"""
计算模型训练的理论内存占用，无需实际实例化模型
"""

import argparse

def get_args():
    """Return arguments."""
    # _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def parse_args():
    """解析命令行参数，对应bash脚本中的各种配置"""
    parser = argparse.ArgumentParser(description="计算模型训练内存占用")
    
    # 模型参数
    parser.add_argument('--num-layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--hidden-size', type=int, default=8192, help='隐藏层维度')
    parser.add_argument('--ffn-hidden-size', type=int, default=32768, help='FFN隐藏层维度')
    parser.add_argument('--seq-length', type=int, default=4096, help='序列长度')
    parser.add_argument('--num-attention-heads', type=int, default=64, help='注意力头数')
    parser.add_argument('--num-query-groups', type=int, default=1)
    parser.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    parser.add_argument('--moe-ffn-hidden-size', type=int, default=None,
                       help='The hidden size of each expert\'s feed-forward network (ffn). '
                       'If not specified, defaults to the ffn_hidden_size.')
    parser.add_argument('--moe-shared-expert-intermediate-size', type=int, default=None,
                       help='Shared expert total ffn hidden size. '
                       'It should be equal to "num_shared_experts * ffn_size_of_each_shared_expert" if there are multiple shared experts. '
                       'None means no shared expert.')
    parser.add_argument('--swiglu', action='store_true',
                       help='Use gated linear units and SiLU activation instead of default gelu')
    parser.add_argument('--group-query-attention', action='store_true',
                          help='Use group-query attention.')
    parser.add_argument('--vocab-size', type=int, default=None,
                       help='Size of vocab before EOD or padding.')
    parser.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                       help='Untie embeddings and output weights.')
    
    # MoE参数
    parser.add_argument('--num-experts', type=int, default=64, help='专家总数')
    parser.add_argument('--moe-router-topk', type=int, default=8, help='激活的专家数')
    
    # 训练参数
    parser.add_argument('--micro-batch-size', type=int, default=1, help='微批次大小')
    parser.add_argument('--global-batch-size', type=int, default=512, help='全局批次大小')
    
    # 并行参数
    parser.add_argument('--tensor-model-parallel-size', type=int, default=4, help='张量并行大小')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=2, help='流水线并行大小')
    parser.add_argument('--expert-model-parallel-size', type=int, default=16, help='专家模型并行大小')
    parser.add_argument('--expert-tensor-parallel-size', type=int, default=2, help='专家张量并行大小')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=1, help='虚拟流水线阶段的层数')
    
    # 精度参数
    parser.add_argument('--bf16', action='store_true', default=True, help='使用bfloat16')
    parser.add_argument('--main-params-dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='主参数数据类型')
    parser.add_argument('--main-grads-dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='主梯度数据类型')
    parser.add_argument('--exp-avg-dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='一阶动量数据类型')
    parser.add_argument('--exp-avg-sq-dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='二阶动量数据类型')
    
    parser.add_argument('--world-size', type=int, help='总进程数')
    
    args, unknown = parser.parse_known_args()
    
    if args.kv_channels is None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.num_experts is not None and args.moe_ffn_hidden_size is None:
        args.moe_ffn_hidden_size = args.ffn_hidden_size
    
    # 打印未知参数（可选，用于调试）
    if unknown:
        print(f"注意: 忽略了未知参数: {unknown}")
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args