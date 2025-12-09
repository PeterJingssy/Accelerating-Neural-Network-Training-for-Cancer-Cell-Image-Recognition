import sys
import traceback
from megatron.training.initialize import initialize_megatron

from model import build_moe_layer
from dataflow import dataflow_analysis
from schedule import overlap_schedule
from utils import reset

def verify_memory_optimization(policy, verbose):
    reset()

    # assert policy is None, f"{policy=}"
    # modules, tensors = build_moe_layer(policy)
    # build_graph(modules, tensors)
    # return 0

    overall_modules, \
    recompute_modules, fp8_act_modules, \
    offload_modules, offload_ratios, \
    tensors = build_moe_layer(policy)

    mem_cnt = dataflow_analysis(
        overall_modules,
        recompute_modules,
        fp8_act_modules,
        offload_modules,
        tensors,
    )
    g_mem = mem_cnt['gpu'] / 1024**3
    c_mem = mem_cnt['cpu'] / 1024**3
    print(f"{policy=} is valid.\nactivation: device {g_mem:.2f}GiB, host {c_mem:.2f}GiB\n")
    if verbose:
        for k in sorted(mem_cnt.keys()):
            if k in ['cpu', 'gpu']:
                continue
            print(f"{k} {mem_cnt[k]}")

    vpp_1f1b_overlap_plan = overlap_schedule(
        tensors,
        overall_modules,
        recompute_modules,
        fp8_act_modules,
        offload_modules,
        offload_ratios,
    )


def test_poicy(policy, verbose=False):
    try:
        verify_memory_optimization(policy, verbose)
    except AssertionError as e:
        print(f"{policy=} fails with:\n{e=}")
        # Get full traceback as a string
        tb_str = traceback.format_exc()
        print("Full traceback:")
        print(tb_str)

        # Alternatively: get last traceback frame info
        tb = sys.exc_info()[2]
        while tb.tb_next:
            tb = tb.tb_next
        frame = tb.tb_frame
        lineno = tb.tb_lineno
        filename = frame.f_code.co_filename
        funcname = frame.f_code.co_name
        print(f"Error occurred in {filename}, function '{funcname}', line {lineno}")


if __name__ == '__main__':
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    all_linears = ['qkv_linear', 'o_linear',
               'routed_expert_up', 'routed_expert_down',
               'shared_expert_up', 'shared_expert_down']

    print("\n\nFailed Case 1")
    # Failed Case 1
    # Root Cause: core_attn stores output tensor,
    # so the following o_proj should not convert it to fp8.
    policy = {
        'recompute_modules': ['ln', 'mlp_act'],
        'fp8_modules': ['o_linear'],
        'offload_modules': [],
    }
    test_poicy(policy)

    print("\n\nFailed Case 2")
    # Failed Case 2
    # Root Cause: moe router and shared expert share the same input.
    # Converting shared expert activation to fp8 causes duplication.
    policy = {
        'recompute_modules': ['ln', 'mlp_act'],
        'fp8_modules': ['shared_expert_up'],
        'offload_modules': [],
    }
    test_poicy(policy)

    print("\n\nFailed Case 3")
    # Failed Case 3
    # Root Cause: recomputation of mlp_act frees the output tensor,
    # so the following w3 should not conver it to fp8.
    policy = {
        'recompute_modules': ['ln', 'mlp_act'],
        'fp8_modules': ['routed_expert_down'],
        'offload_modules': [],
    }
    test_poicy(policy)

    print("\n\nFailed Case 4")
    # Root Cause: layernorm recomputation frees the output tensor,
    # so the following qkv linear should not convert it to fp8.
    policy = {
        'recompute_modules': ['ln', 'mlp_act'],
        'fp8_modules': ['qkv_linear'],
        'offload_modules': [],
    }
    test_poicy(policy)

    print("\n\nCase 5: pure bf16")
    policy = {
        'recompute_modules': ['ln', 'mlp_act'],
        'fp8_modules': [],
        'offload_modules': [],
    }
    test_poicy(policy, verbose=True)

    print("\n\nCase 6: fp8 act")
    policy = {
        'recompute_modules': ['ln', 'mlp_act'],
        'fp8_modules': ['routed_expert_up', 'recompute'],
        'offload_modules': [],
    }
    test_poicy(policy, verbose=True)
