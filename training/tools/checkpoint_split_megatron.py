import argparse
import json
import os
import re
import sys
import types

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def add_checkpointing_args(parser):
    parser.add_argument(
        "--megatron-path",
        type=str, default=None,
        help="Base directory of Megatron repository"
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
    return parser


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


def split_megatron_models(args):
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

    orig_tp_size = megatron_args.tensor_model_parallel_size
    split_tp_size = args.target_tensor_model_parallel_size

    # Create Transformers GPT2 config from Megatron-LM arguments
    vocab_size = megatron_args.padded_vocab_size
    print("vocab_size:", vocab_size)

    output_state_dict = [{} for _ in range(split_tp_size)]

    pp_size = megatron_args.pipeline_model_parallel_size
    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    assert pp_size == 1

    # Convert.
    print("Converting")

    # Convert and store the word embeddings.
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, orig_tp_size, pp_size, 0)
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
            for tp_rank in range(orig_tp_size)
        ],
        dim=0,
    )

    # TODO: use tp to compute padded_vocab_size
    megatron_args.padded_vocab_size = 250880
    padding_size = megatron_args.padded_vocab_size - word_embeddings.shape[0]
    word_embeddings = torch.cat((word_embeddings, word_embeddings[-1].unsqueeze(0).expand(padding_size, -1)))
    tp_word_embedding = torch.chunk(word_embeddings, split_tp_size, dim=0)

    for i in range(split_tp_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = tp_word_embedding[i]

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
                if "rotary_embed.inv_freq" in key:
                    for i in range(split_tp_size):
                        module = get_element_from_dict_by_path(output_state_dict[i], path)
                        module[key] = val
                continue

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                for i in range(split_tp_size):
                    module = get_element_from_dict_by_path(output_state_dict[i], path)
                    module[key] = val
            else:
                dim = 1 if op_name in ["self_attention.dense", "inter_attention.dense", "mlp.w2"] else 0
                full_params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, orig_tp_size)
                    ],
                    dim=dim,
                )
                tp_params = torch.chunk(full_params, split_tp_size, dim=dim)
                for i in range(split_tp_size):
                    module = get_element_from_dict_by_path(output_state_dict[i], path)
                    module[key] = tp_params[i]

        # The final layernorm
        params = get_element_from_dict_by_path(tp_state_dicts[0], path)
        for i in range(split_tp_size):
            module = get_element_from_dict_by_path(output_state_dict[i], path)
            module["final_layernorm.weight"] = params["final_layernorm.weight"]
            module["final_layernorm.bias"] = params["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    lm_head_bias = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.lm_head.bias"
            )
            for tp_rank in range(orig_tp_size)
        ],
        dim=0,
    )
    padding_size = megatron_args.padded_vocab_size - lm_head_bias.shape[0]

    lm_head_bias = torch.cat((lm_head_bias, lm_head_bias[-1].expand(padding_size)))
    tp_lm_head_bias = torch.chunk(lm_head_bias, split_tp_size, dim=0)
    for i in range(split_tp_size):
        module = get_element_from_dict_by_path(output_state_dict[i], "model.lm_head")
        module["bias"] = tp_lm_head_bias[i]

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # saving the state dict as per the tp_rank and pp_rank
    for tp_rank in range(split_tp_size):
        megatron_args.tensor_model_parallel_size = split_tp_size
        output_state_dict[tp_rank]["args"] = megatron_args
        output_state_dict[tp_rank]["checkpoint_version"] = 3.0
        output_state_dict[tp_rank]["iteration"] = state_dict["iteration"]
        # output_state_dict[tp_rank]["checkpoint_name"] = state_dict["checkpoint_name"]
        checkpoint_dir = (
            f"mp_rank_{tp_rank:02d}"
        )
        # TODO: save the state of optimizer and distributed optimizer
        # It's OK when there is not need to restore the optimizer state, e.g., finetune
        # output_state_dict[tp_rank]["optimizer"] = {
        #     "step": 0,
        #     "param_groups": [
        #         {
        #             "lr": 0.0,
        #             "beta1": 0.0,
        #             "beta2": 0.0,
        #             "eps": 0.0,
        #             "weight_decay": 0.0,
        #             "correct_bias": False,
        #             "params": [],
        #         }
        #     ],
        # }
        checkpoint_dir = os.path.join(args.save_path, checkpoint_dir)
        checkpoint_name = "model_optim_rng.pt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if args.print_checkpoint_structure:
            print(
                f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
            )
            recursive_print(None, output_state_dict[tp_rank])
        torch.save(output_state_dict[tp_rank], checkpoint_path)


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    args = parser.parse_args()
    split_megatron_models(args)


if __name__ == "__main__":
    main()