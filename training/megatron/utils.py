# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import sys

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.model.module import param_is_not_shared


def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)

def throughput_calculator(args, iteration_time, total_iterations):
    batch_size = args.global_batch_size
    elapsed_time_per_iter = iteration_time / total_iterations
    hidden_size = args.hidden_size
    ffn_hidden_size = args.ffn_hidden_size
    decoder_num_layers = args.decoder_num_layers
    encoder_num_layers = args.encoder_num_layers
    vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 250000
    if args.ul2_type == "no":
        decoder_seq_length = 114
        encoder_seq_length = 512
    elif args.ul2_type == "sample":
        decoder_seq_length = args.decoder_seq_length
        encoder_seq_length = args.encoder_seq_length
    else:
        raise NotImplementedError("unknown args.ul2_type")
    
    head_nums = args.num_attention_heads
    head_dim = args.kv_channels

    def cal_flops(B, s, l, h, head_nums, head_dim, V, ffn_h, type, s_in= 0):
        '''
        B: batch size
        s: seq_len
        l: layers
        head_nums: head_nums
        head_dim: kv_channels
        V: vocab size
        ffn_h: feed_foward_hidden_states
        type: encoder or decoder, decoder use cross attention
        s_in: sequence length of input in cross attention
        '''
        head_dim_all = head_nums * head_dim
        self_kqv_trans = 6 * B * s * (h * head_dim_all)
        cross_kqv_trans = 4 * B * s_in * (h * head_dim_all) + 2 * B * s * (h * head_dim_all)
        self_attn_calc = 2 * B * (s ** 2) * head_dim_all
        cross_attn_calc = 2 * B * (s * s_in) * head_dim_all
        attn_v = 2 * B * (s ** 2) * head_dim_all
        cross_attn_v = 2 * B * (s * s_in) * head_dim_all
        post_attn_linear = 2 * B * s * (h * head_dim_all)
        mlp_cal = 2 * (2 * ffn_h * h) * B * s
        V_cal = 6 * B * s * V * h
        if type == "decoder":
            self_attn = self_kqv_trans + self_attn_calc + attn_v + post_attn_linear
            cross_attn = cross_kqv_trans + cross_attn_calc + cross_attn_v + post_attn_linear
            return 4 * ((self_attn + cross_attn) + mlp_cal) * l + V_cal
        elif type == "encoder":
            attn = self_kqv_trans + self_attn_calc + attn_v + post_attn_linear
            return 4 * ((attn + mlp_cal) * l )
        else:
            raise AttributeError("type is wrong")

    flops_encoder_per_iteration = cal_flops(B = batch_size, 
                                    s = encoder_seq_length, 
                                    l = encoder_num_layers,
                                    h = hidden_size,
                                    V = vocab_size,
                                    head_nums=head_nums,
                                    head_dim=head_dim,
                                    ffn_h = ffn_hidden_size,
                                    type="encoder")
    
    flops_decoder_per_iteration = cal_flops(B = batch_size, 
                                    s = decoder_seq_length, 
                                    l = decoder_num_layers,
                                    h = hidden_size,
                                    V = vocab_size,
                                    head_nums=head_nums,
                                    head_dim=head_dim,
                                    ffn_h = ffn_hidden_size,
                                    type="decoder",
                                    s_in= encoder_seq_length)
    
    flops_per_iteration = flops_encoder_per_iteration + flops_decoder_per_iteration
    tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
    tflops_encoder = flops_encoder_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
    tflops_decoder = flops_decoder_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))

    return tflops, tflops_encoder, tflops_decoder