#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Megatron-LM log:
1. Extract key hyperparameters (SEQ, MBS, GBS, etc.)
2. Extract performance metrics (Throughput, Params, Memory)
3. Print hyperparameters in fixed format
"""

import os
import re
import argparse
from typing import Dict, Any


# === Megatron参数名 → 简写映射 ===
PARAM_MAP = {
    "seq_length": "SEQ",
    "micro_batch_size": "MBS",
    "global_batch_size": "GBS",
    "num_layers": "L",
    "hidden_size": "HS",
    "ffn_hidden_size": "FFN_HS",
    "num_attention_heads": "H",
    "num_experts": "E",  # ✅ 专家数量
    "moe_router_topk": "TopK",  # ✅ MoE 路由TopK
    "tensor_model_parallel_size": "TP",
    "pipeline_model_parallel_size": "PP",
    "expert_model_parallel_size": "EP",  # ✅ 专家并行大小
    "expert_tensor_parallel_size": "ETP",
    "virtual_pipeline_model_parallel_size": "VP",
    "recompute_granularity": "recompute",  # ✅ 重计算粒度
}


def parse_value(val: str):
    """将字符串值转换为 bool/int/float/str"""
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    if re.match(r"^-?\d+$", val):
        return int(val)
    if re.match(r"^-?\d+\.\d+$", val):
        return float(val)
    return val


def extract_hyperparams(log_file: str) -> Dict[str, Any]:
    """从 Megatron-LM 日志中提取指定超参数"""
    pattern = re.compile(r"\|\s*([a-zA-Z0-9_]+)\s*[.]+\s*(\S+)")
    found = {}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            key, val = match.groups()
            val = parse_value(val)
            if key in PARAM_MAP:
                short = PARAM_MAP[key]
                found[short] = val

    return found


def analyze_logs(log_file: str):
    """从日志中提取性能指标与超参数"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist")
        exit(0)

    # 提取超参数
    params = extract_hyperparams(log_file)

    # 初始化性能指标变量
    throughput_values = []
    total_params = None
    max_allocated_values = []
    max_reserved_values = []

    # 匹配性能指标
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 吞吐率
            t_match = re.search(r'throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)', line)
            if t_match:
                throughput_values.append(float(t_match.group(1)))

            # 模型参数量
            p_match = re.search(r'Total number of parameters in billions:\s*([\d.]+)', line)
            if p_match:
                total_params = float(p_match.group(1))

            # 显存使用
            a_match = re.search(r'max allocated:\s*([\d.]+)', line)
            r_match = re.search(r'max reserved:\s*([\d.]+)', line)
            if a_match and r_match:
                max_allocated_values.append(float(a_match.group(1)))
                max_reserved_values.append(float(r_match.group(1)))

    # 吞吐率均值（跳过前两轮）
    if len(throughput_values) > 2:
        avg_throughput = sum(throughput_values[2:]) / len(throughput_values[2:])
    else:
        avg_throughput = None

    # 显存统计
    actual_memory_stats = {}
    if max_allocated_values:
        actual_memory_stats['max_allocated_mb'] = max(max_allocated_values)
        actual_memory_stats['max_reserved_mb'] = max(max_reserved_values)
        actual_memory_stats['max_allocated_gb'] = actual_memory_stats['max_allocated_mb'] / 1024
        actual_memory_stats['max_reserved_gb'] = actual_memory_stats['max_reserved_mb'] / 1024

    # === 固定顺序输出 ===
    order = [
        "SEQ", "MBS", "GBS", "L", "HS", "FFN_HS", "H", "E",
        "TopK", "TP", "PP", "EP", "ETP", "VP", "recompute"
    ]
    hyper_line = " | ".join(
        f"{k}: {params.get(k, 'N/A')}" for k in order
    ) + " |"

    # === 打印结果 ===
    print("=== Performance ===")
    print(f"Average TGS: {avg_throughput:.2f} TFLOP/s/GPU" if avg_throughput else "Average TGS: N/A")
    print(f"Total Params: {total_params:.2f} B" if total_params else "Total Params: N/A")
    if actual_memory_stats:
        print(f"Max Allocated: {actual_memory_stats['max_allocated_gb']:.2f} GB")
        print(f"Max Reserved: {actual_memory_stats['max_reserved_gb']:.2f} GB")
    print("Hyperparameters:")
    print("  " + hyper_line)


def main():
    parser = argparse.ArgumentParser(description="Analyze Megatron-LM log file")
    parser.add_argument("--log_file", type=str, required=True, help="Path to Megatron log file")
    args = parser.parse_args()

    analyze_logs(args.log_file)


if __name__ == "__main__":
    main()
