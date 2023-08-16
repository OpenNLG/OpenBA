python tools/finetune_data_v2.py \
    --json-file /data/all_instruct/less_is_more_40w_zh/clean/share_zh.jsonl  \
    --input-column input \
    --target-column target \
    --tokenizer-model /data/tokenizer/multilingual-spiece.model \
    --vocab_extra_ids 100 \
    --output-prefix /data/all_instruct/less_is_more_40w_zh/clean/bins/share_zh \
    --dataset-impl mmap \
    --workers 32 \
    --log-interval 10 \
    --chunk-size 8 \

    # --input /opt/data/private/Group1/dyy/pile-data/50B.json \

