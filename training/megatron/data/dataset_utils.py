# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, and NVIDIA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Most of the code here has been copied from:
#   https://github.com/google-research/albert/blob/master/create_pretraining_data.py
# with some modifications.

import math
import os
import time
import collections

import numpy as np
import torch

from megatron import (
    get_args,
    print_rank_0
)
from megatron.core import mpu
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

DSET_TYPE_BERT = 'standard_bert'
DSET_TYPE_ICT = 'ict'
DSET_TYPE_T5  = 't5'

DSET_TYPES = [DSET_TYPE_BERT, DSET_TYPE_ICT, DSET_TYPE_T5]


def get_datasets_weights_and_num_samples(data_prefix,
                                         train_valid_test_num_samples):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0]*num_datasets
    prefixes = [0]*num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2*i])
        prefixes[i] = (data_prefix[2*i+1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    if isinstance(train_valid_test_num_samples, list):
        datasets_train_valid_test_num_samples = []
        for weight in weights:
            datasets_train_valid_test_num_samples.append(
                [int(math.ceil(val * weight * 1.005))
                for val in train_valid_test_num_samples])
    else:
        # Used when separate dataset files are provided for train,
        # valid and test
        datasets_train_valid_test_num_samples = [
            int(math.ceil(train_valid_test_num_samples * weight * 1.005))
            for weight in weights]

    return prefixes, weights, datasets_train_valid_test_num_samples


def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess
    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(['make', '-C', path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys
        sys.exit(1)


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    #print(len_a, len_b, max_num_tokens)
    assert len_a > 0
    if len_a + len_b <= max_num_tokens:
        return False
    while len_a + len_b > max_num_tokens:
        if len_a > len_b:
            len_a -= 1
            tokens = tokens_a
        else:
            len_b -= 1
            tokens = tokens_b
        if np_rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
    return True


def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build tokentypes."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)
    # Segment B.
    for token in tokens_b:
        tokens.append(token)
        tokentypes.append(1)
    if tokens_b:
        # [SEP].
        tokens.append(sep_id)
        tokentypes.append(1)

    return tokens, tokentypes


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False,
                                 geometric_dist=False,
                                 masking_style="bert"):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                not is_start_piece(vocab_id_to_token_dict[token])):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1. / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(ngrams[:len(cand_index_set)],
                              p=pvals[:len(cand_index_set)] /
                              pvals[:len(cand_index_set)].sum(keepdims=True))
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_spans.append(MaskedLmInstance(
            index=index_set,
            label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def build_train_valid_test_datasets(train_data_prefix, valid_data_prefix, data_impl,
                                    train_samples, valid_samples, ul2_type,
                                    max_seq_length, max_seq_length_dec,
                                    seed, skip_warmup):

    # if len(data_prefix) == 1:
    return _build_train_valid_test_datasets(train_data_prefix[0], valid_data_prefix[0],
                                            data_impl, train_samples, valid_samples, ul2_type,
                                            max_seq_length, max_seq_length_dec, seed, skip_warmup)

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i], data_impl, splits_string,
            datasets_train_valid_test_num_samples[i],
            max_seq_length, masked_lm_prob, short_seq_prob,
            seed, skip_warmup, binary_head, max_seq_length_dec,
            dataset_type=dataset_type)
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

    # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)


def _build_train_valid_test_datasets(train_data_prefix, valid_data_prefix, data_impl,
                                     train_samples, valid_samples, ul2_type,
                                     max_seq_length, max_seq_length_dec, seed, skip_warmup):


    # Indexed dataset.
    train_indexed_dataset = get_indexed_dataset_(train_data_prefix,
                                                 data_impl,
                                                 skip_warmup)
    valid_indexed_dataset = get_indexed_dataset_(valid_data_prefix,
                                                 data_impl,
                                                 skip_warmup)

    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_sentences_train = len(train_indexed_dataset)
    total_num_of_sentences_valid = len(valid_indexed_dataset)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, total_num_of_sentences):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     sentence indices in total of {} '
                     'sentences'.format(total_num_of_sentences))
    print_split_stats('train', total_num_of_sentences_train)
    print_split_stats('validation', total_num_of_sentences_valid)

    def build_dataset(name, indexed_dataset, data_prefix, sample_num):
        from megatron.data.t5_dataset import T5Dataset
        dataset = None
        kwargs = dict(
            name=name,
            data_prefix=data_prefix,
            max_num_samples=sample_num,
            seed=seed,
        )
        dataset = T5Dataset(
            indexed_dataset=indexed_dataset,
            encoder_seq_length=max_seq_length, 
            decoder_seq_length=max_seq_length_dec,
            ul2_type=ul2_type,
            **kwargs
        )
        return dataset

    train_dataset = build_dataset('train', train_indexed_dataset, train_data_prefix, train_samples)
    valid_dataset = build_dataset('valid', valid_indexed_dataset, valid_data_prefix, valid_samples)

    return (train_dataset, valid_dataset, None)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):

    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} seconds'.format(time.time() - start_time))

    print_rank_0(' > indexed dataset stats:')
    print_rank_0('    number of sentences: {}'.format( indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index

def get_samples_mapping(indexed_dataset,
                        data_prefix,
                        max_num_samples,
                        seed,
                        name):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    # Filename of the index mapping
    datamap_filename = '{}_{}_datamap_{}mns_{}seed.npy'.format(data_prefix, name, max_num_samples, seed)
    indexmap_filename = '{}_{}_indexmap_{}mns_{}seed.npy'.format(data_prefix, name, max_num_samples, seed)

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and \
       not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        assert indexed_dataset.sizes.dtype == np.int32
        
        # Build samples mapping
        # verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        np_rng = np.random.default_rng(seed=seed)


        spancorr_num = indexed_dataset.get_spancorr_num
        multitask_num = indexed_dataset.get_multitask_num
        
        print_rank_0(' > building samples index mapping for {} - span corruption ...'.format(name))
        print(f"     total number of sentences:      {spancorr_num}\n")

        print_rank_0(' > building samples index mapping for {} - multi task ...'.format(name))
        print(f"     total number of sentences:      {multitask_num}\n")
        
        spancorr_seq = np.arange(spancorr_num, dtype=int)
        multitask_seq = np.arange(multitask_num, dtype=int)
        sub_index_mapping = np.concatenate([spancorr_seq, multitask_seq])
        sub_type_mapping = np.concatenate([np.zeros(spancorr_num, dtype=int), np.ones(multitask_num, dtype=int)])
        permute_num = math.ceil(max_num_samples / (spancorr_num + multitask_num))

        index_mapping_list = []
        type_mapping_list = []
        for _ in range(permute_num):
            p = np_rng.permutation(spancorr_num + multitask_num)
            index_mapping_list.append(sub_index_mapping[p])
            type_mapping_list.append(sub_type_mapping[p])

        index_mapping = np.concatenate(index_mapping_list)
        type_mapping = np.concatenate(type_mapping_list)

        print_rank_0(' > done building samples index maping')
        np.save(datamap_filename, type_mapping, allow_pickle=True)
        np.save(indexmap_filename, index_mapping, allow_pickle=True)
        print_rank_0(' > saved the data mapping in {}'.format(datamap_filename))
        print_rank_0(' > saved the index mapping in {}'.format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elasped time to build and save samples mapping '
                     '(seconds): {:4f}'.format(time.time() - start_time))
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load indexed dataset.
    print_rank_0(' > loading type mapping from {}'.format(datamap_filename))
    print_rank_0(' > loading indexed mapping from {}'.format(indexmap_filename))
    start_time = time.time()
    type_mapping = np.load(datamap_filename, allow_pickle=True, mmap_mode='r')
    samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')

    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(samples_mapping.shape[0]))
    print_rank_0('    number of span corruption: {}'.format((type_mapping <= 6).sum()))
    print_rank_0('    number of multi task: {}'.format((type_mapping == 7).sum()))

    return type_mapping, samples_mapping

def get_finetune_samples_mapping(indexed_dataset,
                        max_num_samples,
                        data_prefix,
                        batch_size,
                        seed,
                        name):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    # Filename of the index mapping
    datamap_filename = '{}_{}_datamap_{}bsz_{}seed.npy'.format(data_prefix, name, batch_size, seed)
    indexmap_filename = '{}_{}_indexmap_{}bsz_{}seed.npy'.format(data_prefix, name, batch_size, seed)

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and \
       not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        assert indexed_dataset.sizes.dtype == np.int32
        
        def get_rand_perm(data_size, total_length, np_rng):
            num_perm = math.ceil(total_length / data_size)
            return np.concatenate([np_rng.permutation(data_size) for _ in range(num_perm)])[:total_length]

        # Build samples mapping
        # verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        np_rng = np.random.default_rng(seed=seed)


        assert indexed_dataset.get_spancorr_num == 0, "there should be no spancorr samples for fine-tune"
        
        multitask_num = indexed_dataset.get_multitask_num // batch_size * batch_size
        
        print_rank_0(' > building samples index mapping for {} - multi task ...'.format(name))
        print(f"     total number of sentences:      {multitask_num}\n")
        
        multitask_seq = get_rand_perm(multitask_num, max_num_samples, np_rng)

        
        type_mapping = np.full(max_num_samples, 7, dtype=int)
        index_mapping = multitask_seq

        print_rank_0(' > done building samples index maping')
        np.save(datamap_filename, type_mapping, allow_pickle=True)
        np.save(indexmap_filename, index_mapping, allow_pickle=True)
        print_rank_0(' > saved the data mapping in {}'.format(datamap_filename))
        print_rank_0(' > saved the index mapping in {}'.format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elasped time to build and save samples mapping '
                     '(seconds): {:4f}'.format(time.time() - start_time))
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load indexed dataset.
    print_rank_0(' > loading type mapping from {}'.format(datamap_filename))
    print_rank_0(' > loading indexed mapping from {}'.format(indexmap_filename))
    start_time = time.time()
    type_mapping = np.load(datamap_filename, allow_pickle=True, mmap_mode='r')
    samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
    
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(samples_mapping.shape[0]))
    print_rank_0('    number of span corruption: {}'.format((type_mapping <= 6).sum()))
    print_rank_0('    number of multi task: {}'.format((type_mapping == 7).sum()))

    return type_mapping, samples_mapping


def t5_trans(input_ids, noise_density, mean_noise_span_length, eos_token_id, bos_token_id, prefix_token_id, max_sentinel_token, np_rng):
    train_sample = {}

    input_ids_mask = np.asarray([random_spans_noise_mask(input_ids.shape[-1], noise_density, mean_noise_span_length, np_rng)])
    labels_mask = ~input_ids_mask

    input_ids_sentinel = create_sentinel_ids(input_ids_mask.astype(np.int8), max_sentinel_token)
    labels_sentinel = create_sentinel_ids(labels_mask.astype(np.int8), max_sentinel_token)

    if prefix_token_id < 0:
        train_sample["text_enc"] = filter_input_ids(input_ids, input_ids_sentinel, eos_token_id)[0]
    else:
        train_sample["text_enc"] = np.concatenate(\
                        (np.array([prefix_token_id]), \
                        filter_input_ids(input_ids, input_ids_sentinel, eos_token_id)[0]), axis = -1)
    train_sample["labels"] = filter_input_ids(input_ids, labels_sentinel, eos_token_id)[0]
    train_sample["text_dec"] = np.concatenate((np.array([bos_token_id]), train_sample["labels"][:-1]), axis = -1)

    return train_sample

def random_spans_noise_mask(length, noise_density, mean_noise_span_length, np_rng):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np_rng.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]

def filter_input_ids(input_ids, sentinel_ids, eos_token_id):
    """
    Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
    This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
    """
    batch_size = input_ids.shape[0]

    input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
    # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
    # masked tokens coming after sentinel tokens and should be removed
    input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
    input_ids = np.concatenate(
        [input_ids, np.full((batch_size, 1), eos_token_id, dtype=np.int32)], axis=-1
    )
    return input_ids

def create_sentinel_ids(mask_indices, max_sentinel_token):
    """
    Sentinel ids creation given the indices that should be masked.
    The start indices of each mask are replaced by the sentinel ids in increasing
    order. Consecutive mask indices to be deleted are replaced with `-1`.
    """
    start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
    start_indices[:, 0] = mask_indices[:, 0]

    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    sentinel_ids = np.where(sentinel_ids != 0, (max_sentinel_token - sentinel_ids), 0)
    sentinel_ids -= mask_indices - start_indices

    return sentinel_ids

def pad_ids(input_ids, expected_length, pad_token_id):
    padding_length = expected_length - input_ids.shape[-1] if expected_length else 0
    assert padding_length >= 0, f"padding_length must > 0, expected_length is {expected_length}, input_ids_length is {input_ids.shape[-1]}"
    if padding_length:
        input_ids = np.concatenate((input_ids, np.array([pad_token_id] * padding_length)), axis = -1)
    return input_ids

def pad_and_make_masks(train_sample, enc_in_length, dec_in_length, pad_token_id):
    # train_sample['text_enc'] = train_sample['text_enc'][:enc_in_length]
    # train_sample['text_dec'] = train_sample['text_dec'][:dec_in_length]
    # train_sample['labels'] = train_sample['labels'][:dec_in_length]
    num_tokens_dec = train_sample['text_dec'].shape[-1]
    if dec_in_length is None:
        padding_length_dec = 0
    else:
        padding_length_dec = dec_in_length - num_tokens_dec

    train_sample['text_enc'] = pad_ids(train_sample['text_enc'], enc_in_length, pad_token_id)
    train_sample['text_dec'] = pad_ids(train_sample['text_dec'], dec_in_length, pad_token_id)
    train_sample['labels'] = pad_ids(train_sample['labels'], dec_in_length, -1)

    tokens_enc = train_sample['text_enc']
    tokens_dec_in = train_sample['text_dec']
    tokens_dec_in4mask = tokens_dec_in.copy()
    tokens_dec_in4mask[0] = 1
    enc_mask = make_attention_mask(tokens_enc, tokens_enc)
    enc_dec_mask = make_attention_mask(tokens_dec_in4mask, tokens_enc)
    dec_mask = make_attention_mask(tokens_dec_in4mask, tokens_dec_in4mask)
    dec_mask = dec_mask * make_history_mask(tokens_dec_in4mask)
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    train_sample['enc_mask'] = enc_mask
    train_sample['enc_dec_mask'] = enc_dec_mask
    train_sample['dec_mask'] = dec_mask
    train_sample['loss_mask'] = loss_mask
    
    return train_sample

def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask

def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (arange[None, ] <= arange[:, None])
    history_mask = history_mask.astype(np.int64)
    return history_mask

def get_enc_dec_length(type_, length, mean_noise_span_length, noise_density):
    if type_ == "S":
        min_dec_lengths, max_enc_lengths = 1 + 3, length - 1 + 2
        max_dec_lengths, min_enc_lengths = min(round(length * noise_density * 2), length) + 3, length - min(round(length * noise_density * 2), length) + 2
        return (f"{min_enc_lengths}-{max_enc_lengths}", f"{min_dec_lengths}-{max_dec_lengths}")
    else:
        num_noise_tokens = int(np.round(length * noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        return (2 + num_nonnoise_tokens + num_noise_spans, num_noise_tokens + num_noise_spans + 1)
    
def s_trans(input_ids, noise_density, eos_token_id, bos_token_id, prefix_token_id, np_rng, sentinel_token):
    train_sample = {}
    length = input_ids.shape[-1]
    max_noise_token_num = min(round(noise_density * length),length)
    random_noise_num = int(np_rng.uniform(1, max_noise_token_num + 1))
    random_unnoise_num = length - random_noise_num
    train_sample['text_enc'] = np.concatenate(\
                    (np.array([prefix_token_id]), \
                    input_ids[0][:random_unnoise_num],\
                    np.array([sentinel_token, eos_token_id]))
                    , axis = -1)
    train_sample['text_dec'] = np.concatenate(\
                    (np.array([bos_token_id, sentinel_token]), \
                    input_ids[0][random_unnoise_num:],\
                    ), axis = -1)
    train_sample['labels'] = np.concatenate(\
                    (np.array([sentinel_token]),\
                    input_ids[0][random_unnoise_num:],\
                    np.array([eos_token_id]),\
                    ), axis = -1)
    return train_sample

def s_trans_multitask(src_tokens, tgt_tokens, S_token_id, enc_in_length, dec_in_length, bos_token_id, eos_token_id, sentinel_token):
    train_sample = {}
    train_sample['text_enc'] = np.concatenate(\
                    (np.array([S_token_id]), \
                     src_tokens[:enc_in_length - 3],\
                     np.array([sentinel_token, eos_token_id]))
                     , axis = -1)

    train_sample['text_dec'] = np.concatenate((np.array([bos_token_id, sentinel_token]), \
                                                tgt_tokens[:dec_in_length - 2]), axis = -1)

    train_sample['labels'] = np.concatenate((np.array([sentinel_token]), \
                                                tgt_tokens[:dec_in_length - 1]), axis = -1)
    return train_sample
