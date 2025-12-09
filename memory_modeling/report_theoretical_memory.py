# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training without instantiating
a model and running training iterations on GPU(s)."""

# from megatron.training import get_args
# from megatron.training.initialize import initialize_megatron
from args import (
    parse_args,
    get_args,
)
from gpt_activation import (
    set_activation_model,
    get_activation_model,
    calculate_memory_consumption,
    args_to_params,
)
from gpt_parameter import (
    mlp_params,
    attn_params,
)
from pipeline import (
    warmup_layers,
)
from utils import bytes_dict

def calculate_activation_memory(args, max_vio=0.0, pure_bf16_act=False):
    set_activation_model('aibenchmark', pure_bf16_act)
    # FIXME: support offload in modeling
    args.offload_modules = None
    #在函数开始执行时，它通过这两行代码禁用了 offload 功能，并为未来的开发留下了需要FIXME的标记。

    act_model = get_activation_model()
    act_model.set_values(args_to_params())

    base_ntoken = args.seq_length * args.micro_batch_size * args.moe_router_topk
    ntoken = (1 + max_vio) * base_ntoken
    act_model.set_dynamic_value("ntoken", ntoken)
    act_mem = calculate_memory_consumption(args)
    return act_mem


def calculate_model_states(args):
    world_size = args.world_size
    embedding_params = args.hidden_size * args.vocab_size
    if args.untie_embeddings_and_output_weights:
        embedding_params = 2 * embedding_params
    moe_params = args.num_layers * mlp_params(args)
    dense_params = args.num_layers * attn_params(args)
    total_params = embedding_params + moe_params + dense_params

    params_dtype = 'bf16' if args.bf16 else 'fp32'
    grads_dtype = 'bf16' if args.bf16 else 'fp32'

    bytes_per_params = bytes_dict[params_dtype] + bytes_dict[grads_dtype]
    bytes_per_optimizer_state = bytes_dict[str(args.main_params_dtype)] \
                            + bytes_dict[str(args.main_grads_dtype)] \
                            + bytes_dict[str(args.exp_avg_dtype)] \
                            + bytes_dict[str(args.exp_avg_sq_dtype)]

    params_per_rank = (moe_params / args.expert_model_parallel_size / args.expert_tensor_parallel_size + 
                       dense_params / args.tensor_model_parallel_size) / args.pipeline_model_parallel_size
    opt_per_rank = (moe_params + dense_params) / world_size

    params_per_rank *= bytes_per_params
    opt_per_rank *= bytes_per_optimizer_state

    return total_params, params_per_rank, opt_per_rank


def mem_breakdown(max_vio=0, pure_bf16_act=False):
    args = get_args()

    act_per_batch = calculate_activation_memory(args, max_vio=max_vio, pure_bf16_act=pure_bf16_act)
    mem_act = warmup_layers(args) * act_per_batch / args.tensor_model_parallel_size
    params, mem_weights, mem_optimizer = calculate_model_states(args)

    print(f"pipeline warmup layers: {warmup_layers(args)}")
    
    mem_act /= 1024**3
    mem_weights /= 1024**3
    mem_optimizer /= 1024**3
    params /= 10**9
    print(f"{max_vio=} {pure_bf16_act=}\nparams {params:.2f}B\nact {mem_act:.2f}GB\nweight {mem_weights:.2f}GB\noptimzer {mem_optimizer:.2f}GB")

    print(f"total {mem_act + mem_weights + mem_optimizer:.2f}")
    print("#" * 50)

if __name__ == "__main__":
    parse_args()
    mem_breakdown(max_vio=0.0, pure_bf16_act=True)
    mem_breakdown(max_vio=0.1, pure_bf16_act=True)
    mem_breakdown(max_vio=0.0, pure_bf16_act=False)
    mem_breakdown(max_vio=0.1, pure_bf16_act=False)
