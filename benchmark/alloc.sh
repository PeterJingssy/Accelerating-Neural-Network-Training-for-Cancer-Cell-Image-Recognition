#!/bin/bash

# ============================================================================
# Modular salloc Resource Allocation Script
# ============================================================================

# 配置参数
JOB_NAME="GMKTest"
PARTITION="ai_custom"
NODES=8
NTASKS_PER_NODE=8
CPUS_PER_TASK=8
GRES="dcu:8"
HOSTFILE_PATH="/public/home/thu_gmk/dcu_megatron/examples/aibenchmark/hostfile_aibenchmark"

# ============================================================================
# 函数定义
# ============================================================================

generate_hostfile() {
    echo "=========================================="
    echo "Generating Hostfile"
    echo "=========================================="
    
    # 创建目标目录
    local hostfile_dir=$(dirname "$HOSTFILE_PATH")
    mkdir -p "$hostfile_dir"
    
    echo "Target hostfile path: $HOSTFILE_PATH"
    echo "Allocated nodes: $SLURM_JOB_NODELIST"
    
    if [ -z "$SLURM_JOB_NODELIST" ]; then
        echo "✗ No node list available"
        return 1
    fi
    
    # 解析节点列表
    if command -v scontrol &> /dev/null; then
        local node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
        
        if [ -n "$node_list" ]; then
            echo "Resolved nodes:"
            echo "$node_list"
            
            # 生成hostfile
            > "$HOSTFILE_PATH"
            for node in $node_list; do
                echo "$node slots=8" >> "$HOSTFILE_PATH"
                echo "  Added: $node slots=8"
            done
            
            echo "✓ Hostfile generated successfully!"
            echo ""
            echo "Hostfile content:"
            cat "$HOSTFILE_PATH"
            return 0
        else
            echo "✗ Failed to resolve node list with scontrol"
            return 1
        fi
    else
        echo "✗ scontrol not available, trying manual parsing..."
        
        # 手动解析
        if [[ "$SLURM_JOB_NODELIST" =~ ^[a-zA-Z0-9,\[\]-]+$ ]]; then
            local node_list=$(echo "$SLURM_JOB_NODELIST" | tr ',' ' ')
            echo "Manually parsed nodes: $node_list"
            
            > "$HOSTFILE_PATH"
            for node in $node_list; do
                echo "$node slots=8" >> "$HOSTFILE_PATH"
                echo "  Added: $node slots=8"
            done
            
            echo "✓ Hostfile generated with manual parsing!"
            echo ""
            echo "Hostfile content:"
            cat "$HOSTFILE_PATH"
            return 0
        else
            echo "✗ Cannot parse node list format: $SLURM_JOB_NODELIST"
            return 1
        fi
    fi
}

setup_environment() {
    echo "=========================================="
    echo "Setting Environment Variables"
    echo "=========================================="
    
    # DCU设备可见性
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
    # 性能优化变量
    export GLOG_minloglevel=3
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export OMP_NUM_THREADS=1
    export GPU_MAX_HW_QUEUES=10
    
    # 缓存目录设置
    export CUDA_CACHE_PATH="$HOME/.cache/cuda"
    export HIP_CACHE_PATH="$HOME/.cache/hip"
    export PYTORCH_KERNEL_CACHE_PATH="$HOME/.cache/torch"
    export TRANSFORMER_ENGINE_CACHE_PATH="$HOME/.cache/transformer_engine"
    export TMPDIR="$HOME/tmp"
    
    # 创建缓存目录
    local cache_dirs=(
        "$HOME/.cache/cuda"
        "$HOME/.cache/hip"
        "$HOME/.cache/torch"
        "$HOME/.cache/transformer_engine"
        "$HOME/tmp"
    )
    
    for dir in "${cache_dirs[@]}"; do
        mkdir -p "$dir"
        echo "  Created cache directory: $dir"
    done
    
    echo "✓ Environment setup complete"
}

show_allocation_info() {
    echo "=========================================="
    echo "Resource Allocation Information"
    echo "=========================================="
    echo "Job Name: $JOB_NAME"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Partition: $PARTITION"
    echo "Nodes: $SLURM_JOB_NODELIST"
    echo "Number of nodes: $NODES"
    echo "Tasks per node: $NTASKS_PER_NODE"
    echo "CPUs per task: $CPUS_PER_TASK"
    echo "GPU/DCU per node: 8"
    echo "Hostfile: $HOSTFILE_PATH"
    echo "=========================================="
}

# ============================================================================
# 主要的salloc启动函数
# ============================================================================

start_interactive_session() {
    echo "Starting interactive resource allocation..."
    echo "Parameters: --job-name=$JOB_NAME --partition=$PARTITION --nodes=$NODES --ntasks-per-node=$NTASKS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --gres=$GRES --exclusive"
    echo ""
    
    # 使用salloc分配资源
    salloc \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --nodes="$NODES" \
        --ntasks-per-node="$NTASKS_PER_NODE" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --gres="$GRES" \
        --exclusive \
        bash -c "
            # 导入函数
            $(declare -f generate_hostfile)
            $(declare -f setup_environment)
            $(declare -f show_allocation_info)
            
            # 设置变量
            HOSTFILE_PATH='$HOSTFILE_PATH'
            JOB_NAME='$JOB_NAME'
            PARTITION='$PARTITION'
            NODES='$NODES'
            NTASKS_PER_NODE='$NTASKS_PER_NODE'
            CPUS_PER_TASK='$CPUS_PER_TASK'
            
            # 执行设置
            generate_hostfile
            setup_environment
            show_allocation_info
            
            echo \"\"
            echo \"Interactive session started. Type 'exit' to release resources.\"
            echo \"You can now run your distributed training commands.\"
            echo \"\"
            
            # 启动交互式shell
            exec bash
        "
}

# ============================================================================
# 脚本执行
# ============================================================================

if [ \"\$1\" = \"--help\" ] || [ \"\$1\" = \"-h\" ]; then
    echo \"Usage: \$0 [options]\"
    echo \"\"
    echo \"Interactive resource allocation with automatic hostfile generation\"
    echo \"\"
    echo \"Options:\"
    echo \"  --help, -h     Show this help message\"
    echo \"  --dry-run      Show the salloc command without executing\"
    echo \"\"
    echo \"Configuration:\"
    echo \"  Job Name: \$JOB_NAME\"
    echo \"  Partition: \$PARTITION\"
    echo \"  Nodes: \$NODES\"
    echo \"  Tasks per node: \$NTASKS_PER_NODE\"
    echo \"  CPUs per task: \$CPUS_PER_TASK\"
    echo \"  GPU/DCU per node: 8\"
    echo \"  Hostfile: \$HOSTFILE_PATH\"
    exit 0
fi

if [ \"\$1\" = \"--dry-run\" ]; then
    echo \"Dry run - would execute:\"
    echo \"salloc --job-name=\$JOB_NAME --partition=\$PARTITION --nodes=\$NODES --ntasks-per-node=\$NTASKS_PER_NODE --cpus-per-task=\$CPUS_PER_TASK --gres=\$GRES --exclusive\"
    exit 0
fi

# 启动交互式会话
start_interactive_session
