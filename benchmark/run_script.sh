set -ex

for para in $*
do
    if [[ $para == --profiling* ]];then
        profiling=${para#*=}
    elif [[ $para == --gpus* ]];then
        custom_gpus=${para#*=}
    fi
done

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# Those variables need to modify
DTK_ENV="/opt/dtk-25.04.1/env.sh"                                                               # where env.sh of dtk
DATA_PATH="/public/home/thu_gmk/datasets/deepseekv3_dataset/mmap_deepseekv3_datasets_text_document"                                                             # path to data files (.bin and .idx)
TOKENIZER_MODEL_PATH="/public/home/thu_gmk/datasets/deepseekv3_dataset"                                                  # path to tokenizer file
CHECKPOINT_PATH="./ckpt"                                                       # path to ckpt
NCCL_ENV=${MEGATRON_PATH}/requirements/env.sh                            # Please adjust the variables based on the actual NET being used
LAUNCH_WITH_BINDING=${MEGATRON_PATH}/requirements/launch_with_binding.sh # Please adjust the variables based on the actual NET being used

# Those variables no need to modify
HOSTFILE="hostfile_aibenchmark"

# Calculate GPUS: use custom_gpus if provided, otherwise auto-calculate from hostfile
if [[ -n "$custom_gpus" ]]; then
    # Validate custom_gpus is a positive integer
    if [[ "$custom_gpus" =~ ^[1-9][0-9]*$ ]]; then
        GPUS=$custom_gpus
        echo "Using custom GPUS count: $GPUS"
    else
        echo "Error: --gpus must be a positive integer, got: $custom_gpus"
        echo "Usage: $0 [--gpus=N] [--profiling=torch|hip]"
        echo "  --gpus=N     : Number of GPUs to use (default: auto-calculate from hostfile)"
        echo "  --profiling  : Enable profiling (torch or hip)"
        exit 1
    fi
else
    GPUS=$(($(cat ${HOSTFILE}|sort|uniq |wc -l)*8))
    echo "Auto-calculated GPUS from hostfile: $GPUS ($(cat ${HOSTFILE}|sort|uniq |wc -l) nodes * 8 GPUs/node)"
fi

HOST="$(cat ${HOSTFILE} |sed -n "1p"|awk -F ' ' '{print $1}')"
PORT="25900"

if [[ -z "$TRAIN_SCRIPT" ]]; then
    TRAIN_SCRIPT="./train_aibenchmark.sh"
fi

# Runs aibenchmark model
source ${NCCL_ENV}
mpirun -np ${GPUS}  --hostfile ${HOSTFILE} \
                    --allow-run-as-root \
                    --bind-to none \
                    --mca plm_rsh_no_tree_spawn 1 \
                    bash -c "
                    exec > >(while read line; do echo \"\$(date +"%Y-%m-%dT%H:%M:%S%z") | \$(hostname) | \$line\"; done) &&
                    exec 2> >(while read line; do echo \"\$(date +"%Y-%m-%dT%H:%M:%S%z") | \$(hostname) | \$line\" >&2; done) &&
                    cd ${CURRENT_DIR} && \
                    source ${NCCL_ENV} && \
                    source ${DTK_ENV} && \
                    source "/public/home/thu_gmk/miniconda3/bin/activate" && \
                    conda activate megatron_new && \
                    ${TRAIN_SCRIPT} \
                    ${HOST} \
                    ${PORT} \
                    --data_path=$DATA_PATH \
                    --tokenizer_path=$TOKENIZER_MODEL_PATH \
                    --checkpoint_path=$CHECKPOINT_PATH \
                    --launch_with_binding=${LAUNCH_WITH_BINDING} \
                    --profiling=$profiling" >> ./log/$((${GPUS} / 8))nodes-`date +%F-%H%M`.log 2>&1

wait
