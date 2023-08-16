# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        # Encoder.insturct_column = self.args.insturct_column
        Encoder.input_column = self.args.input_column
        Encoder.target_column = self.args.target_column
        Encoder.task_prefix = self.args.task_prefix

    def encode(self, json_line):
        data = json.loads(json_line)
        tot_len  = len(json_line)
        # instruct_text = data[Encoder.insturct_column]
        instruct_text = ""
        source_text = instruct_text + data[Encoder.input_column] + " summarize:"
        target_text = Encoder.task_prefix + data[Encoder.target_column]

        source_sentence_ids = Encoder.tokenizer.tokenize(source_text)
        source_sentence_ids.append(Encoder.tokenizer.eos_id)
        target_sentence_ids = Encoder.tokenizer.tokenize(target_text)
        target_sentence_ids.append(Encoder.tokenizer.eos_id)
        
        return (source_sentence_ids, target_sentence_ids), tot_len

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--json-file', type=str, required=True, help='Path to input JSON')
    group.add_argument('--insturct-column', type=str, default='text')
    group.add_argument('--input-column', type=str, default='text')
    group.add_argument('--target-column', type=str, default='answer')
    group.add_argument('--batch-size', type=int, default=1000)
    group.add_argument('--task-prefix', type=str, default="")

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-model', type=str, required=True)
    group.add_argument('--vocab_extra_ids', type=int, default=0)

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True, help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True, help='Number of worker processes to launch')
    group.add_argument('--chunk-size', type=int, required=True, help='Chunk size assigned to each worker process')
    group.add_argument('--log-interval', type=int, default=100, help='Interval between progress updates')
    args = parser.parse_args()
    # args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.tokenizer_type = "SentencePieceTokenizer"
    # args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.json_file)
    fin = open(args.json_file, 'r', encoding='utf-8')

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    encoded_docs_batch = pool.imap(encoder.encode, fin, args.chunk_size)
    #encoded_docs = map(encoder.encode, fin)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_file = "{}_multitask.bin".format(args.output_prefix)
    output_idx_file = "{}_multitask.idx".format(args.output_prefix)
    builder = indexed_dataset.make_builder(output_bin_file,
                                            impl=args.dataset_impl,
                                            vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_docs_processed, total_sentence_created, total_bytes_processed = 0, 0, 0
    print("Time to startup:", startup_end - startup_start)

    for doc, bytes_processed in encoded_docs_batch:
        source, target = doc
        builder.add_item(
            source_tensor=torch.IntTensor(source),
            target_tensor=torch.IntTensor(target),
            task="multi-task",
        )
        total_docs_processed += 1
        total_sentence_created += 1
        total_bytes_processed += bytes_processed
        
        if total_docs_processed % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"{total_docs_processed} docs, {total_docs_processed / elapsed:.0f} docs/s | ",
                  f"{total_sentence_created} sents, {total_sentence_created / elapsed:.0f} sents/s | ",
                  f"{mbs:.2f} MB/s | ")
    print("Done! Now finalizing.")

    builder.finalize(output_idx_file)

if __name__ == '__main__':
    main()
