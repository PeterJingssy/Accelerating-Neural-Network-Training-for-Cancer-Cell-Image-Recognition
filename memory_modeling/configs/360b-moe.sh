#!/bin/bash

TEXT_DATA_PATH="
    Wikipedia_2023,/glm_training_testdata/tokenized/150k-tokenizer/merge_wikipedia_2023,binary,0.0213395272811331
"

NAME="360b-moe"
CHECKPOINT_PATH="/glm_training_testdata/checkpoints/360b-moe/360b-moe-0512"
EXP_NAME=$NAME

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$((MLP_WORKER_NUM * 8 * 2))

EP_SIZE=32
PP_SIZE=8
ETP_SIZE=1
TP_SIZE=1
VP_SIZE=2

MOE_ROUTED_EXPERTS=160
MOE_ACTIVE_ROUTED_EXPERTS=8
MOE_SHARED_EXPERTS=1

NHIDDEN=5120
MOE_FFN_HIDDEN=1536
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
FFN_HIDDEN=12288
N_DENSE_LAYERS=1
N_MOE_LAYERS=93
N_REDUCE_LAYERS_FOR_LM_HEAD=2
NHEADS=96

SEQ_LEN=4096

SAVE_INTERVAL=100

TRAIN_TOKENS=10000000000000
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 100 / 100))
LR_WARMUP_SAMPLES=$((GLOBAL_BATCH_SIZE * 100))

script_path="pretrain_glm.py"

OPTIMIZER_ARGS="
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr 1e-4
    --min-lr 1e-5
    --lr-decay-style cosine
    --lr-decay-samples $LR_DECAY_SAMPLES
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --clip-grad 1.0
    --weight-decay 1e-1
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --initial-loss-scale 65536
"

MOE_ARGS="
    --moe-layer-freq [0]*$N_DENSE_LAYERS+[1]*$((N_MOE_LAYERS+N_REDUCE_LAYERS_FOR_LM_HEAD))
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --expert-model-parallel-size $EP_SIZE
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4
    --moe-router-dtype fp32
    --moe-token-dispatcher-type flex
    --moe-enable-deepep
"
#    --combined-1f1b
#    --split-bw
#    --moe-shared-expert-overlap

MODEL_ARGS="
    --bf16
    --num-layers $((N_DENSE_LAYERS+N_MOE_LAYERS+N_REDUCE_LAYERS_FOR_LM_HEAD))
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --seq-length $SEQ_LEN
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings $SEQ_LEN
    --num-attention-heads $NHEADS
    --disable-bias-linear
    --add-qkv-bias
    --rotary-percent 0.5
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --no-position-embedding
    --normalization RMSNorm
    --kv-channels 128
"
    # --reduce-layers-for-lm-head $N_REDUCE_LAYERS_FOR_LM_HEAD
    # --tokenizer-type 150k-tokenizer

TRAINING_ARGS="
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --expert-model-parallel-size $EP_SIZE
    --expert-tensor-parallel-size $ETP_SIZE
    --num-layers-per-virtual-pipeline-stage $VP_SIZE
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
    --manual-gc
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 2
"

DATA_ARGS="
    --num-workers 1
    --train-data-path $TEXT_DATA_PATH
"
# --text-ratio 1.0

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --eval-interval 0 \
    --timing-log-level 0 \
    --save-interval $SAVE_INTERVAL \
    --wandb-save-dir $CHECKPOINT_PATH \
    --wandb-exp-name $NAME
"

gpt_options="
    $MODEL_ARGS
    $MOE_ARGS
    $TRAINING_ARGS
    $OPTIMIZER_ARGS
    $DATA_ARGS
    $OUTPUT_ARGS
    --distributed-timeout-minutes 10
    --init-method-std 0.01
    --ckpt-format torch_dist
    --async-save
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
"
