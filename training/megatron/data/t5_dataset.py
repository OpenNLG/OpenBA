# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""T5 Style dataset."""

import collections

import numpy as np
import torch

from megatron import get_tokenizer, print_rank_0
from megatron.data.dataset_utils import (
    create_masked_lm_predictions,
    get_samples_mapping,
    get_finetune_samples_mapping,
    t5_trans,
    pad_and_make_masks,
    get_enc_dec_length,
    s_trans,
    s_trans_multitask,
)

ORIGIN_UL2 = {
    1 : {"type": "R", "mean_noise_span_length": 3, "noise_density":0.15}, 
    2 : {"type": "R", "mean_noise_span_length": 8, "noise_density":0.15}, 
    3 : {"type": "S", "noise_density": 0.25}, 
    4 : {"type": "X", "mean_noise_span_length": 3, "noise_density":0.5}, 
    5 : {"type": "X", "mean_noise_span_length": 8, "noise_density":0.5}, 
    6 : {"type": "X", "mean_noise_span_length": 64, "noise_density":0.15}, 
    7 : {"type": "X", "mean_noise_span_length": 64, "noise_density":0.5}, 
}

NO_UL2 = {"type": "R", "mean_noise_span_length": 3, "noise_density":0.15}

RECORDED = 0

class T5Dataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 max_num_samples, encoder_seq_length, decoder_seq_length,
                 ul2_type, seed):

        # Params to store.
        self.name = name
        self.seed = seed
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.ul2_type = ul2_type

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.type_mapping, self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                                      data_prefix, max_num_samples,
                                                                      seed, name)

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.unk_id = tokenizer.unk_id
        self.rtask_id = tokenizer.rtask_id
        self.stask_id = tokenizer.stask_id
        self.xtask_id = tokenizer.xtask_id
        self.sentinel_tokens = tokenizer.sentinel_tokens_ids
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        # 0: span-corruption, 1: multi-task
        data_type = self.type_mapping[idx]
        data_index = self.samples_mapping[idx]
        src_tokens, tgt_tokens = self.indexed_dataset[(data_type, data_index)]

        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.default_rng(seed=(self.seed + idx))
        s_np_rng = np.random.default_rng(seed=(self.seed + idx))
        if data_type == 1 or self.ul2_type == "sample":
            enc_in_length, dec_in_length = self.encoder_seq_length, self.decoder_seq_length
        else:
            enc_in_length, dec_in_length = None, None

        return build_training_sample(src_tokens,  tgt_tokens,
                                    self.vocab_id_list,
                                    self.vocab_id_to_token_dict,
                                    np_rng, s_np_rng,
                                    self.rtask_id, self.stask_id, self.xtask_id,
                                    self.eos_id, self.bos_id, self.pad_id,
                                    self.sentinel_tokens,
                                    multi_task = (data_type == 1), ul2_type=self.ul2_type,
                                    enc_in_length = enc_in_length, dec_in_length = dec_in_length,)


def build_training_sample(src_tokens,  tgt_tokens,
                        vocab_id_list, vocab_id_to_token_dict,
                        np_rng, s_np_rng,
                        R_token_id, S_token_id, X_token_id,
                        eos_token_id, bos_token_id, pad_token_id,
                        sentinel_tokens=None, expanded_input_length=568,
                        enc_in_length=None, dec_in_length=None, 
                        multi_task = False, ul2_type = "sample"):
    global RECORDED
    truncated = 0
    
    if not multi_task:
        tokens = src_tokens
        if len(tokens) > expanded_input_length: 
            print(f" > warning: ul2's sentence_length = {len(tokens)} > expanded_input_length = {expanded_input_length}, trunct it")
            tokens = tokens[:expanded_input_length]
            truncated = 1 
        elif len(tokens) < expanded_input_length: 
            print(f" > warning: ul2's sentence_length = {len(tokens)} < expanded_input_length = {expanded_input_length}")

    if multi_task: 
        tgt_tokens = tgt_tokens.astype('int64')
        src_tokens = src_tokens.astype('int64')
        train_sample = s_trans_multitask(src_tokens, tgt_tokens, S_token_id, enc_in_length, dec_in_length, bos_token_id, eos_token_id, max(sentinel_tokens))
        # get 'text_enc', 'text_dec', 'labels' with padding and make masks
        train_sample = pad_and_make_masks(train_sample, enc_in_length, dec_in_length, pad_token_id)
        train_sample['truncated'] = (len(src_tokens) > enc_in_length + 3) or (len(tgt_tokens) > dec_in_length + 2)
        assert train_sample['text_enc'].shape == (enc_in_length,)
        assert train_sample['text_dec'].shape == train_sample['labels'].shape == (dec_in_length,)
        
        # print("train_sample['text_enc']:", [vocab_id_to_token_dict[i] for i in train_sample['text_enc']])
        # print("train_sample['text_dec']:", [vocab_id_to_token_dict[i] for i in train_sample['text_dec']])
        # print("train_sample['labels']: ", [vocab_id_to_token_dict[i] for i in train_sample['labels'] if i != -1])
        # print('-' * 100)
        return train_sample
    else:
        if ul2_type == "batch" or ul2_type == "sample":
            if ul2_type == "sample": 
                ul2_task_id = np_rng.integers(1, 8)
            if RECORDED == 0:
                RECORDED = 1
                for i in ORIGIN_UL2:
                    type_, mean_noise_span_length, noise_density = \
                    ORIGIN_UL2[i]["type"],\
                    ORIGIN_UL2[i]["mean_noise_span_length"] if ORIGIN_UL2[i]["type"] != "S" else None,\
                    ORIGIN_UL2[i]["noise_density"]

                    enc_length, dec_length = get_enc_dec_length(type_, expanded_input_length, mean_noise_span_length, noise_density)
                    # print_rank_0("{:<30s}{:<30s}{:<30s}{:<30s}{:<30s}".format(
                    #         f"UL2_Task_Id:{type_}-{i}",
                    #         f"Mean_SPAN_Length:{mean_noise_span_length}",
                    #         f"Noise_Density:{noise_density}",
                    #         f"Enc_Length:{enc_length}",
                    #         f"Dec_Length:{dec_length}"
                    #     ))

            denoiser_type = ORIGIN_UL2[ul2_task_id]['type']
            mean_noise_span_length = ORIGIN_UL2[ul2_task_id]["mean_noise_span_length"] \
            if denoiser_type != "S" else None
            noise_density = ORIGIN_UL2[ul2_task_id]["noise_density"]
            prefix_token_id = X_token_id if denoiser_type == "X" else \
                            R_token_id if denoiser_type == "R" else \
                            S_token_id if denoiser_type == "S" else None
            
            # translate to t5, get 'text_enc', 'text_dec', 'labels' without padding
            if denoiser_type != "S":
                train_sample = t5_trans(np.array([tokens]), noise_density, 
                            mean_noise_span_length, eos_token_id,
                            bos_token_id, prefix_token_id,
                            max(sentinel_tokens) + 1, np_rng)
            else:
                train_sample = s_trans(np.array([tokens]), noise_density, 
                            eos_token_id, bos_token_id, prefix_token_id, s_np_rng, max(sentinel_tokens))
                
            # get 'text_enc', 'text_dec', 'labels' with padding and make masks
            train_sample = pad_and_make_masks(train_sample, enc_in_length, dec_in_length, pad_token_id)
            train_sample['truncated'] = truncated
            # print(train_sample['text_enc'].shape, '\n')
            # print(train_sample['text_dec'].shape, '\n')
            # print(train_sample['labels'].shape, '\n')
            # print('ul2','-'*30)

            return train_sample
        
        elif ul2_type == "no":
            noise_density = NO_UL2["noise_density"]
            mean_noise_span_length = NO_UL2["mean_noise_span_length"]
            train_sample = t5_trans(np.array([tokens]), noise_density, 
                            mean_noise_span_length, eos_token_id,
                            bos_token_id, -1,
                            max(sentinel_tokens) + 1, np_rng)
            train_sample = pad_and_make_masks(train_sample, enc_in_length, dec_in_length, pad_token_id)
            train_sample['truncated'] = truncated
            # print(train_sample['text_enc'].shape, '\n')
            # print(train_sample['text_dec'].shape, '\n')
            # print(train_sample['labels'].shape, '\n')
            # print('no','-'*30)
            return train_sample
        else:
            raise NotImplementedError("Unknown ul2_type")


