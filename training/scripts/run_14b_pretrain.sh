#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export OMP_NUM_THREADS=24

# Change for multinode config
MASTER_ADDR=wxhd00
MASTER_PORT=17099
NNODES=4
NODE_RANK=0
GPUS_PER_NODE=8

LOAD_PATH="/data/checkpoint/14b_main"
CHECKPOINT_PATH="/data/checkpoint/14b_main"
TRAIN_DATA_PATH="/data/en_zh/all_data"
VALID_DATA_PATH="/data/en_zh/val_spancorr"
TOKENIZER_PATH="/data/tokenizer/multilingual-spiece.model"
TESNSORBOARD_PATH=$CHECKPOINT_PATH/tensorboard

mkdir -p ${TESNSORBOARD_PATH}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

T5_ARGS="
    --tensor-model-parallel-size 4 \
    --encoder-num-layers 12 \
    --decoder-num-layers 36 \
    --hidden-size 4096 \
    --num-attention-heads 40 \
    --kv-channels 128 \
    --ffn-hidden-size 16384 \
    --encoder-seq-length 570 \
    --decoder-seq-length 381 \
    --max-position-embeddings 768 \
    --micro-batch-size 16 \
    --global-batch-size 4096 \
    --lr 0.0001 \
    --train-iters 200000 \
    --lr-decay-iters 100000 \
    --lr-decay-style cosine \
    --min-lr 0.00001 \
    --weight-decay 0.1 \
    --lr-warmup-iters 2000 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --clip-grad 1.0 \
    --fp16 \
    --vocab-extra-ids 100 \
    --ul2-type sample \
    --pos-emb-type rotary \
    --mlp-type SwiGLU \
    --use-distributed-optimizer \
    --no-query-key-layer-scaling \
    --recompute-activations \
    --attention-softmax-in-fp32 \
"

DATA_ARGS="
    --train-data-path $TRAIN_DATA_PATH \
    --valid-data-path $VALID_DATA_PATH \
    --tokenizer-model $TOKENIZER_PATH \
    --data-impl mmap \
    --num-workers 32 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 3 \
    --tensorboard-dir $TESNSORBOARD_PATH \
"

torchrun $DISTRIBUTED_ARGS pretrain_t5.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $LOAD_PATH | tee -a $CHECKPOINT_PATH/${NODE_RANK}.log
