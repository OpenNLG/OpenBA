python convert_megatron_to_hf_ckp/convert.py \
    --convert_checkpoint_from_megatron_to_transformers \
    --load_path /public/home/ljt/LEO/checkpoint/14b_flan_final_v1/iter_0004500_model  \
    --save_path /public/home/ljt/LEO/checkpoint/14b_flan_final_v1/iter_0004500_model_hf \
    --tokenizer_name "./tokenizer-ckp" \
    --print-checkpoint-structure \
