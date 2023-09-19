python convert_megatron_to_hf_ckp.py \
    --convert_checkpoint_from_megatron_to_transformers \
    --load_path /data/checkpoint/14b_main_long_final/iter_0020000/  \
    --save_path /opt/dyy/hf_model_stretch \
    --tokenizer_name "OpenBA/OpenBA-LM" \
    --print-checkpoint-structure \
