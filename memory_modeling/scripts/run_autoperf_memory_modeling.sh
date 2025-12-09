#!/bin/bash

workspace=/public/home/thu_gmk/GLM-MoE

# GPUS
WORLD_SIZE=256

##########################
# Qwen3-235B
SEQ=4096
MBS=1
GBS=512

L=94
HS=4096
FFN_HS=1536
H=64
NQG=16
E=128
TopK=8

TP=4
PP=8
EP=16
ETP=2
VP=6
#########################

for para in $*
do
    if [[ $para == --SEQ=* ]];then
        SEQ=${para#*=}
    elif [[ $para == --L=* ]];then
        L=${para#*=}
    elif [[ $para == --HS=* ]];then
        HS=${para#*=}
    elif [[ $para == --FFN_HS=* ]];then
        FFN_HS=${para#*=}
    elif [[ $para == --H=* ]];then
        H=${para#*=}
    elif [[ $para == --NQG=* ]];then
        NQG=${para#*=}
    elif [[ $para == --E=* ]];then
        E=${para#*=}
    elif [[ $para == --TopK=* ]];then
        TopK=${para#*=}
    elif [[ $para == --EG=* ]];then
        EG=${para#*=}
    elif [[ $para == --GTopK=* ]];then
        GTopK=${para#*=}
    elif [[ $para == --MBS=* ]];then
        MBS=${para#*=}
    elif [[ $para == --GBS=* ]];then
        GBS=${para#*=}
    elif [[ $para == --TP=* ]];then
        TP=${para#*=}
    elif [[ $para == --PP=* ]];then
        PP=${para#*=}
    elif [[ $para == --EP=* ]];then
        EP=${para#*=}
    elif [[ $para == --ETP=* ]];then
        ETP=${para#*=}
    elif [[ $para == --VP=* ]];then
        VP=${para#*=}
    elif [[ $para == --recompute=* ]];then
        recompute=${para#*=}
    fi
done

ARGS="
    --world-size $WORLD_SIZE
    --num-layers $L
    --hidden-size $HS
    --ffn-hidden-size $FFN_HS
    --num-attention-heads $H
    --num-experts $E
    --moe-router-topk $TopK
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --seq-length $SEQ
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --expert-model-parallel-size $EP
    --expert-tensor-parallel-size $ETP
    --bf16
"

# Qwen3-235B
ARGS+="
    --vocab-size 151936
    --swiglu
    --kv-channels 128
"

# Deepseek-v3
# --vocab-size 129280

# aibenchmark
# --untie-embeddings-and-output-weights

if [ -n "${NQG}" ]; then
    ARGS+="
        --group-query-attention
        --num-query-groups ${NQG}
    "
fi
if [ -n "${VP}" ]; then
    ARGS+="
        --num-layers-per-virtual-pipeline-stage ${VP}
    "
fi
if [ -n "${EG}" ]; then
    ARGS+="
        --moe-router-num-groups ${EG}
        --moe-router-group-topk ${GTopK}
    "
fi
if [ -n "${recompute}" ]; then
    ARGS+="
        --recompute-activations
    "
fi

python3 $workspace/memory_modeling/report_theoretical_memory.py $ARGS