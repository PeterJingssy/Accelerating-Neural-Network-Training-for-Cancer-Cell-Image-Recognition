#!/bin/bash

VOCAB_FILE=./data/gpt2-vocab.json
MERGE_FILE=./data/gpt2-merges.txt
DATA_PATH=./data/CodeData-gpt2_text_document

EXP_NAME="deepseek-v3"
CHECKPOINT_PATH="/checkpoint/experiments/debug/${EXP_NAME}"

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$((MLP_WORKER_NUM * 8 * 4))

EP_SIZE=32
PP_SIZE=8
ETP_SIZE=1
TP_SIZE=1
VP_SIZE=1

MOE_ROUTED_EXPERTS=256
MOE_ACTIVE_ROUTED_EXPERTS=8
MOE_SHARED_EXPERTS=1
# MOE_ROUTER_NUM_GROUPS=$(($EP_SIZE * $ETP_SIZE / 8))
# # MOE_ROUTER_GROUP_TOPK=$(($MOE_ROUTER_NUM_GROUPS / 2)) # 1/2 Node limit

NHIDDEN=7168
MOE_FFN_HIDDEN=2048
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
FFN_HIDDEN=18432
N_DENSE_LAYERS=3
N_MOE_LAYERS=60
N_REDUCE_LAYERS_FOR_LM_HEAD=1
NHEADS=128

SEQ_LEN=4096

SAVE_INTERVAL=100

TRAIN_TOKENS=1000000000
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 1 / 100))

script_path="pretrain_gpt.py"

TARGET_VOCAB=151552

#MODEL_ARGS=(
#    --disable-bias-linear
#    --seq-length $SEQ_LEN
#    --max-position-embeddings $SEQ_LEN
#    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS + N_REDUCE_LAYERS_FOR_LM_HEAD))
#    --reduce-layers-for-lm-head $N_REDUCE_LAYERS_FOR_LM_HEAD
#    --hidden-size $NHIDDEN
#    --ffn-hidden-size $FFN_HIDDEN
#    --num-attention-heads $NHEADS
#    --kv-channels 128
#    --init-method-std 0.01
#    --attention-dropout 0.0
#    --hidden-dropout 0.0
#    --normalization RMSNorm
#    --position-embedding-type rope
#    --swiglu
#    --untie-embeddings-and-output-weights
#    --group-query-attention
#    --num-query-groups 8
#    --no-masked-softmax-fusion
#    --no-position-embedding
#)

MODEL_ARGS=(
    --transformer-impl local
    --disable-bias-linear
    --seq-length $SEQ_LEN
    --max-position-embeddings $SEQ_LEN
    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS + N_REDUCE_LAYERS_FOR_LM_HEAD))
    # --reduce-layers-for-lm-head $N_REDUCE_LAYERS_FOR_LM_HEAD
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NHEADS
    --kv-channels 128
    --group-query-attention
    --num-query-groups 8
    # --multi-latent-attention
    # --kv-lora-rank 512
    # --q-lora-rank 1536
    # --qk-head-dim 128
    # --qk-layernorm
    # --qk-pos-emb-head-dim 64
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    # --no-rope-fusion
    --swiglu
)

MOE_ARGS=(
    --moe-layer-freq [0]*$N_DENSE_LAYERS+[1]*$((N_MOE_LAYERS+N_REDUCE_LAYERS_FOR_LM_HEAD))
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-grouped-gemm
    --moe-router-pre-softmax
    --moe-router-topk-scaling-factor 2.5
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-score-function sigmoid
    --moe-router-bias-update-rate 0.001
    --moe-router-enable-expert-bias
    --moe-aux-loss-coeff 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-router-force-load-balance
    --recompute-modules layernorm mlp_act
    # --recompute-modules layernorm mla_up_proj
    --moe-token-dispatcher-type flex
    --moe-enable-deepep
    # --moe-token-dispatcher-type alltoall
    --combined-1f1b
    --combined-1f1b-recipe ep_a2a
    --split-bw
    --force-fp8-ctx
    --fp8-ctx-modules Linear GroupedLinear LayerNormLinear # permutation
    --activation-func-fp8-input-store
    # --no-check-for-nan-in-loss-and-grad
)


if [ ! -z $MOE_ROUTER_NUM_GROUPS ]; then
    MOE_ARGS+=(
        --moe-router-num-groups $MOE_ROUTER_NUM_GROUPS
        --moe-router-group-topk $MOE_ROUTER_GROUP_TOPK
    )
fi
#    --moe-shared-expert-overlap
#    --moe-token-dispatcher-type flex
#    --moe-pad-expert-input-to-capacity
#    --moe-router-aux-loss-fusion

DATA_ARGS=(
    --make-vocab-size-divisible-by $((TARGET_VOCAB / TP_SIZE)) \
    --tokenizer-type GPT2BPETokenizer \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr 1e-4
    --train-iters 10000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1e-5
    --weight-decay 0.1
    --lr-warmup-iters 200
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --main-params-dtype fp16
    --exp-avg-dtype fp8
    --exp-avg-sq-dtype fp8
    --manual-gc
    --distributed-timeout-minutes 60
    --activation-offload-ratio 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    --offload-modules Linear LayerNormLinear # Linear GroupedLinear LayerNormLinear permutation
)

#    --recompute-layernorm
#    --moe-layer-recompute
#    --recompute-granularity full
#    --recompute-method uniform
#    --recompute-num-layers 2

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --expert-model-parallel-size $EP_SIZE
    --expert-tensor-parallel-size $ETP_SIZE
    --num-layers-per-virtual-pipeline-stage $VP_SIZE
    --sequence-parallel
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    # --log-interval 1 \
    # --timing-log-level 1 \
    # --save-interval 100 \
    # --eval-interval 10000 \
    # --eval-iters 10 \
    # --log-throughput \
    # --profile
    # --use-pytorch-profiler
    # --profile-step-start 6
    # --profile-step-end 8
    # --profile-ranks 0
)


gpt_options="
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
"
