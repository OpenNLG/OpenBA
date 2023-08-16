python tools/preprocess_data_pretrain.py \
    --json-file /opt/dyy/hf_code_repo/data.jsonl \
    --json-key text \
    --group-size 568 \
    \
    --tokenizer-model /data/tokenizer/entokenizer.model \
    --vocab_extra_ids 100 \
    \
    --output-prefix /data/en_zh/envaltemp \
    --dataset-impl mmap \
    --batch-size 1000 \
    --workers 1 \
    --chunk-size 1 \
    --log-interval 1 \
