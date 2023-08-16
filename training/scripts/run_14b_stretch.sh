#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export OMP_NUM_THREADS=24

# Change for multinode config
MASTER_ADDR=wxhd00
MASTER_PORT=12399
NNODES=4
NODE_RANK=0
GPUS_PER_NODE=8

LOAD_PATH="/data/checkpoint/14b_main"
CHECKPOINT_PATH="/data/checkpoint/14b_main_long_final_new"
TRAIN_DATA_PATH="/data/en_zh/all_data_stretch_2048"
VALID_DATA_PATH="/data/en_zh/all_data_stretch_2048"
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
    --encoder-seq-length 1027 \
    --decoder-seq-length 1025 \
    --max-position-embeddings 2038 \
    --micro-batch-size 4 \
    --global-batch-size 1024 \
    --lr 0.00004 \
    --train-iters 100000 \
    --lr-decay-iters 25000 \
    --lr-decay-style cosine \
    --min-lr 0.00001 \
    --weight-decay 0.1 \
    --lr-warmup-iters 0 \
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
    --attention-softmax-in-fp32 \
    --finetune \
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
    --save-interval 1000 \
    --eval-interval 1000 \
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
