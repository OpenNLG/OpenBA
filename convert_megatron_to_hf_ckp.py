import argparse
import json
import os
import re
import sys
import types

import torch

from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
from OpenBT5 import OpenBT5Config, OpenBT5Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'training')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), 'training')))


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    'input_layernorm': 'input_layernorm',
    'self_attention.query_key_value': 'self_attn.qkv',
    'self_attention.dense': 'self_attn.o',
    'post_attention_layernorm': 'post_attn_layernorm',
    'inter_attention.query': 'inter_attn.q',
    'inter_attention.key_value': 'inter_attn.kv',
    'inter_attention.dense': 'inter_attn.o',
    'post_inter_attention_layernorm': 'post_inter_attn_layernorm',
    'mlp.w1': 'mlp.fc_in_1',
    'mlp.w2': 'mlp.fc_out',
    'mlp.w3': 'mlp.fc_in_2',
}
transformers_to_megatron = {v[1:-1]: k for k, v in megatron_to_transformers.items()}

tensor_parallel_params = [
    'self_attention.query_key_value.weight', 'self_attention.query_key_value.bias',
    'self_attention.dense.weight',
    'inter_attention.query.weight', 'inter_attention.query.bias',
    'inter_attention.key_value.weight', 'inter_attention.key_value.bias',
    'inter_attention.dense.weight',
    'mlp.w1.weight', 'mlp.w1.bias',
    'mlp.w2.weight',
    'mlp.w3.weight', 'mlp.w3.bias',
]


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(60 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = 'model_optim_rng.pt'
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = 'model_optim_rng.pt'
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers GPT2 config from Megatron-LM arguments
    vocab_size = megatron_args.padded_vocab_size
    print("vocab_size:", vocab_size)

    config = OpenBT5Config(
        architectures=["OpenBT5ForConditionalGeneration"],
        vocab_size=vocab_size,
        hidden_size=megatron_args.hidden_size,
        kv_channels=megatron_args.kv_channels,
        ffn_hidden_size=megatron_args.ffn_hidden_size,
        num_layers=megatron_args.num_layers,
        num_decoder_layers=megatron_args.decoder_num_layers,
        hidden_dropout=megatron_args.hidden_dropout,
        attention_dropout=megatron_args.attention_dropout,
        num_heads=megatron_args.num_attention_heads,
        is_encoder_decoder=True,
        tie_word_embeddings=False,
        add_qkv_bias=True,
        add_ffn_bias=False,
        add_lm_head_bias=True,
        model_type="openbt5",
        max_seq_length=megatron_args.encoder_seq_length,
        decoder_max_seq_length = megatron_args.decoder_seq_length,
    )

    output_state_dict = {}

    # checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size
    dtype = torch.float16
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    assert pp_size == 1

    # Convert.
    print("Converting")

    # Convert and store the word embeddings.
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    word_embeddings = word_embeddings[:vocab_size].to(dtype)
    output_state_dict["shared_embedding.weight"] = word_embeddings

    print("Converting transformer layers")
    # Convert and store the transformer Layers
    root_path = "model.language_model"
    for module_name in ["encoder", "decoder"]:
        path = f"{root_path}.{module_name}"
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():    
            m = layer_re.match(key)
            if m is None:
                break

            layer_idx = int(m.group(1))
            op_name = m.group(2)
            weight_or_bias = m.group(3)
            if weight_or_bias != "weight" and weight_or_bias != "bias":
                continue

            layer_name = f"{module_name}.block.{layer_idx}"
            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in ["self_attention.dense", "inter_attention.dense", "mlp.w2"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            hf_module = megatron_to_transformers[op_name]
            hf_path = f"{layer_name}.{hf_module}.{weight_or_bias}"
            output_state_dict[hf_path] = params

        # The final layernorm
        params = get_element_from_dict_by_path(tp_state_dicts[0], path)
        output_state_dict[f"{module_name}.final_layernorm.weight"] = params["final_layernorm.weight"].to(dtype)
        output_state_dict[f"{module_name}.final_layernorm.bias"] = params["final_layernorm.bias"].to(dtype)
        num_layers = config.num_layers if module_name == "encoder" else config.num_decoder_layers

        # merge fc_in_1 and fc_in_2 to fc_in
        for layer_idx in range(num_layers):
            output_state_dict[f'{module_name}.block.{layer_idx}.mlp.fc_in.weight'] = torch.cat(
                [
                    output_state_dict[f'{module_name}.block.{layer_idx}.mlp.fc_in_1.weight'],
                    output_state_dict[f'{module_name}.block.{layer_idx}.mlp.fc_in_2.weight'],
                ],
                dim=0,
            ).to(dtype)
            del output_state_dict[f'{module_name}.block.{layer_idx}.mlp.fc_in_1.weight']
            del output_state_dict[f'{module_name}.block.{layer_idx}.mlp.fc_in_2.weight']

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    output_state_dict["lm_head.weight"] = word_embeddings.to(dtype)
    output_state_dict["lm_head.bias"] = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.lm_head.bias"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Add tokenizer class info to config
    # see https://github.com/huggingface/transformers/issues/13906)

    if args.tokenizer_name is None:
        tokenizer_name = "mt5-base"
    else:
        tokenizer_name = args.tokenizer_name

    tokenizer = OpenBT5Tokenizer.from_pretrained(tokenizer_name)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(args.save_path)

    # Save tokenizer based on args
    if args.tokenizer_name is not None:
        print(f"Adding {tokenizer_class} tokenizer files")
        tokenizer.save_pretrained(args.save_path)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )


def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.

    Args:
        args (argparse.Namespace): the arguments to the script

    """
    pass


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
