python tools/preprocess_data_pretrain.py \
    --json-file /data/pretrain_data.jsonl \
    --json-key text \
    --group-size 568 \
    \
    --tokenizer-model /data/tokenizer/multilingual-spiece.model \
    --vocab_extra_ids 100 \
    \
    --output-prefix /data/pretrain_data \
    --dataset-impl mmap \
    --batch-size 1000 \
    --workers 16 \
    --chunk-size 1 \
    --log-interval 10 \
