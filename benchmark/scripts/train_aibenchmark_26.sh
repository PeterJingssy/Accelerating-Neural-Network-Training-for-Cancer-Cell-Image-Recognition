#!/bin/bash

##########################

SEQ=4096
MBS=1
GBS=512

L=8
HS=8192
FFN_HS=32768
H=32
E=128
TopK=4
##########################

TP=4
PP=4
EP=64
ETP=4
VP=1

# recompute="True"

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --tokenizer_path* ]];then
        tokenizer_path=${para#*=}
    elif [[ $para == --checkpoint_path* ]];then
        checkpoint_path=${para#*=}
    elif [[ $para == --launch_with_binding* ]];then
        launch_with_binding=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

# data path
DATA_PATH=${data_path}
TOKENIZER_MODEL_PATH=${tokenizer_path}
CHECKPOINT_PATH=${checkpoint_path}

# default env
DIST_URL=${1}
DIST_PORT=${2}
RANK=$OMPI_COMM_WORLD_RANK
LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
MEGATRON_PATH="/public/home/thu_gmk/dcu_megatron"
export GLOG_minloglevel=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export GPU_MAX_HW_QUEUES=10
export PYTHONPATH=${MEGATRON_PATH}/Megatron-LM:$PYTHONPATH

# int8_simulation_fp8
export NVTE_INT8_SIM_FP8_TENSORWISE=1
export NVTE_DISABLE_NVRTC=1
export NVTE_INT8_SIM_FP8=1

# Fix torch compile cache conflicts in multi-process environment
export TORCHINDUCTOR_CACHE_DIR="/public/home/thu_gmk/tmp/torchinductor_thu_gmk_${RANK}"
export TORCH_COMPILE_CACHE_SIZE_LIMIT=10000000000  # 10GB per process
export TORCHINDUCTOR_MAX_WORKERS=1

# Fix Triton compiler temporary directory conflicts
export TRITON_HOME="/public/home/thu_gmk/tmp/triton_${RANK}"
# export TRITON_CACHE_DIR="/public/home/thu_gmk/tmp/triton_cache_${RANK}"
export TMPDIR="/public/home/thu_gmk/tmp/tmpdir_${RANK}"
mkdir -p ${TMPDIR}

export LD_LIBRARY_PATH=/public/home/thu_gmk/blas/rocblas-install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/thu_gmk/blas/hipblaslt-install/lib:$LD_LIBRARY_PATH


DISTRIBUTED_ARGS=(
    --rank ${RANK}
    --world-size ${WORLD_SIZE}
    --local-rank ${LOCAL_RANK}
    --dist-url tcp://${DIST_URL}:${DIST_PORT}
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQ}
    --max-position-embeddings 32768
    --num-layers ${L}
    --hidden-size ${HS}
    --ffn-hidden-size ${FFN_HS}
    --num-attention-heads ${H}
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --ckpt-format torch
    --use-quantize-comm
    --fp8-format hybrid
    --fp8-recipe tensorwise 
    --fp8-param-gather
    --extra-vocab-size 467
)

if [ -n "${VP}" ]; then
    MODEL_ARGS+=(
        --overlap-moe-expert-parallel-comm
        --schedule-method interleaved_1f1b
    )
fi

if [ -n "${NQG}" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups ${NQG}
    )
fi

# --delay-wgrad-compute
# --vocab-size 128813

MOE_ARGS=(
    --num-experts ${E}
    --moe-router-topk ${TopK}
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-grouped-gemm
    --moe-expert-capacity-factor 1
    --moe-pad-expert-input-to-capacity
    --moe-router-dtype fp32
)

if [ -n "${EG}" ]; then
    MOE_ARGS+=(
        --moe-router-num-groups ${EG}
        --moe-router-group-topk ${GTopK}
    )
fi

#--tokenizer-type Llama2Tokenizer

DATA_ARGS=(
    --tokenizer-type DeepSeekV2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path ${DATA_PATH}
    --split 98,2,0
)

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --lr 1e-4
    --train-iters 20
    --lr-decay-iters 10000
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --weight-decay 0.1
    --lr-warmup-iters 2000
    --clip-grad 1.0
    --bf16
    --overlap-param-gather
    --overlap-grad-reduce
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size ${ETP}
    --context-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

if [ -n "${VP}" ]; then
    MODEL_PARALLEL_ARGS+=(
        --num-layers-per-virtual-pipeline-stage ${VP}
    )
fi

if [ -n "${recompute}" ]; then
    MODEL_PARALLEL_ARGS+=(
        --recompute-activations
    )
fi

MEMORY_SNAPSHOT_PATH="./memory_logs/"

LOGGING_ARGS=(
    --log-throughput \
    --log-interval 1 \
    --log-memory-iters 20 \
    --save-interval 100000 \
    --eval-interval 10000 \
    --eval-iters 0 \
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --record-memory-history \
    --memory-snapshot-path $MEMORY_SNAPSHOT_PATH\
)

TORCH_PROFIE_ARGS=(
    --profile
    --profile-ranks 0 1 2 3 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 128 192 256 320 384 448 512 576 640 704 768 832 896 960
    --profile-step-start 10
    --profile-step-end 15
    --profile-dir ./profiling/torch_profiling_iters_10_15_1
    --use-pytorch-profiler
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"GPT"}
        --wandb-exp-name ${WANDB_NAME:-"GPT_567B"}
    )
fi

APP="python3 -u ${MEGATRON_PATH}/pretrain_gpt.py \
    ${DISTRIBUTED_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    "

if [[ $profiling == "torch" ]]; then
    APP+=" ${TORCH_PROFIE_ARGS[@]}"
elif [[ $profiling == "hip" ]]; then
    mkdir -p hip_prof_data
    APP+=" ${HIP_PROFIE_ARGS[@]}"
    APP="hipprof -d hip_prof_data --hip-trace --trace-off ${APP}"
fi

#for hygon cpu
${launch_with_binding} ${LOCAL_RANK} ${APP}
