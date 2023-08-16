# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Tokenization utilities."""


import torch


from megatron import get_tokenizer
from .communication import broadcast_int_list, broadcast_tensor


def detokenize_generations(tokens_gpu_tensor,
                           lengths_gpu_tensor,
                           return_segments):
    """Detokenize the generated tokens."""

    tokenizer = get_tokenizer()

    prompts_plus_generations = []
    if return_segments:
        prompts_plus_generations_segments = []

    tokens = tokens_gpu_tensor.cpu().numpy().tolist()
    lengths = lengths_gpu_tensor.cpu().numpy().tolist()
    for sequence_tokens, length in zip(tokens, lengths):
        sequence_tokens = sequence_tokens[:length]
        prompts_plus_generations.append(
            tokenizer.detokenize(sequence_tokens, skip_special_tokens=True))
        if return_segments:
            words = []
            for token in sequence_tokens:
                word = tokenizer.tokenizer.decoder[token]
                word = bytearray(
                    [tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                        'utf-8', errors='replace')
                words.append(word)
            prompts_plus_generations_segments.append(words)

    if return_segments:
        return tokens, prompts_plus_generations, \
            prompts_plus_generations_segments

    return tokens, prompts_plus_generations


def tokenize_prompts(encoder_inputs=None, prompts=None, tokens_to_generate=None, encoder_seq_length=None, rank=0):
    """Tokenize prompts and make them avaiable on all ranks."""

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    encoder_inputs_tokens_cuda_long_tensor = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    if torch.distributed.get_rank() == rank:
        assert prompts is not None
        assert tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        encoder_inputs_tokens_cuda_long_tensor = \
            _tokenize_encoder_inputs_and_batch(encoder_inputs, encoder_seq_length)
        prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
            _tokenize_prompts_and_batch(prompts, tokens_to_generate)
        # We need the sizes of these tensors for the boradcast
        sizes_list = [prompts_tokens_cuda_long_tensor.size(0), # Batch size
                      prompts_tokens_cuda_long_tensor.size(1)] # Sequence lenght

    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=rank)

    # Now that we have the sizes, we can boradcast the tokens
    # and length tensors.
    sizes = sizes_tensor.tolist()
    prompts_tokens_cuda_long_tensor = broadcast_tensor(
        sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank)
    prompts_length_cuda_long_tensor = broadcast_tensor(
        sizes[0], torch.int64, tensor=prompts_length_cuda_long_tensor, rank=rank)
    encoder_inputs_tokens_cuda_long_tensor = broadcast_tensor(
        (sizes[0], encoder_seq_length), torch.int64, tensor=encoder_inputs_tokens_cuda_long_tensor, rank=rank)

    return encoder_inputs_tokens_cuda_long_tensor, prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor


def _tokenize_prompts_and_batch(prompts, tokens_to_generate):
    """Given a set of prompts and number of tokens to generate:
        - tokenize prompts
        - set the sequence length to be the max of length of prompts
          plus the number of tokens we would like to generate
        - pad all the sequences to this length so we can convert them
          into a 2D tensor.
    """

    # Tokenize all the prompts.
    tokenizer = get_tokenizer()
    prompts_tokens = [tokenizer.tokenize(prompt) for prompt in prompts]

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.
    prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len = max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = max_prompt_len + tokens_to_generate
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = samples_length - prompt_length
        prompt_tokens.extend([tokenizer.pad_id] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.cuda.LongTensor(prompts_tokens)
    prompts_length_tensor = torch.cuda.LongTensor(prompts_length)

    return prompts_tokens_tensor, prompts_length_tensor

def _tokenize_encoder_inputs_and_batch(encoder_inputs, encoder_seq_length):
    tokenizer = get_tokenizer()
    encoder_inputs_tokens = [tokenizer.tokenize(encoder_input) for encoder_input in encoder_inputs]

    # Now update the list of list to be of the same size: samples_length.
    for idx, encoder_input_tokens in enumerate(encoder_inputs_tokens):
        encoder_input_tokens = [tokenizer.stask_id] + encoder_input_tokens + [max(tokenizer.sentinel_tokens_ids), tokenizer.eos_id]
        padding_size = 0
        encoder_input_tokens += [tokenizer.pad_id] * padding_size
        encoder_inputs_tokens[idx] = encoder_input_tokens

    # Now we are in a structured format, we can convert to tensors.
    encoder_inputs_tokens = torch.cuda.LongTensor(encoder_inputs_tokens)

    return encoder_inputs_tokens

