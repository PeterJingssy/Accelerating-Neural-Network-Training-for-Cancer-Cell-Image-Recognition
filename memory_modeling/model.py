from tensor import Tensor
from module import Module
from utils import OPType


def build_moe_layer(policy):
    tensors = {}
    input_states = Tensor(
        ["batch_size", "seq_len/tp", "hidden_size"],
        tensors,
    )
    attn_ln_output = Tensor(
        ["batch_size", "seq_len/tp", "hidden_size"],
        tensors,
    )
    kv_input = Tensor(
        ["batch_size", "seq_len", "query_group/tp", "kv_channel", 2],
        tensors,
    )
    q_input = Tensor(
        ["batch_size", "seq_len", "attention_head/tp", "kv_channel"],
        tensors,
    )
    core_attn_output = Tensor(
        ["batch_size", "seq_len", "attention_head/tp", "kv_channel"],
        tensors,
    )
    attn_output = Tensor(
        ["batch_size", "seq_len/tp", "hidden_size"],
        tensors,
    )
    mlp_input = Tensor(
        ["batch_size", "seq_len/tp", "hidden_size"],
        tensors,
    )
    post_ln_mlp_input = Tensor(
        ["batch_size", "seq_len/tp", "hidden_size"],
        tensors,
    )
    experts_input = Tensor(
        ["dynamic_ntoken", "hidden_size"],
        tensors,
    )
    routed_experts_w1_output = Tensor(
        ["dynamic_ntoken", "moe_hidden_size"],
        tensors,
    )
    routed_experts_w2_output = Tensor(
        ["dynamic_ntoken", "moe_hidden_size"],
        tensors,
    )
    routed_experts_w2_output_with_score = Tensor(
        ["dynamic_ntoken", "moe_hidden_size"],
        tensors,
    )
    routed_experts_silu_output = Tensor(
        ["dynamic_ntoken", "moe_hidden_size"],
        tensors,
    )
    routed_experts_mul_output = Tensor(
        ["dynamic_ntoken", "moe_hidden_size"],
        tensors,
    )
    routed_experts_w3_output = Tensor(
        ["dynamic_ntoken", "hidden_size"],
        tensors,
    )
    # routed_experts_output_with_score = Tensor(
        # ["dynamic_ntoken", "hidden_size"],
        # tensors,
    # )
    routed_experts_combined_output = Tensor(
        ["batch_size", "seq_len/tp", "hidden_size"],
        tensors,
    )
    shared_experts_w1_output = Tensor(
        ["batch_size", "seq_len", "shared_hidden_size/tp"],
        tensors,
    )
    shared_experts_w2_output = Tensor(
        ["batch_size", "seq_len", "shared_hidden_size/tp"],
        tensors,
    )
    shared_experts_silu_output = Tensor(
        ["batch_size", "seq_len", "shared_hidden_size/tp"],
        tensors,
    )
    shared_experts_mul_output = Tensor(
        ["batch_size", "seq_len", "shared_hidden_size/tp"],
        tensors,
    )
    shared_experts_w3_output = Tensor(
        ["batch_size", "seq_len", "shared_hidden_size/tp"],
        tensors,
    )
    output_states = Tensor(
        ["batch_size", "seq_len/tp", "hidden_size"],
        tensors,
    )

    mem_ops_profile_data = {'fwd': 0, 'dgrad': 0}
    qkv_linear_profile_data = {'fwd': 0.002, 'dgrad': 0.002, 'wgrad': 0.002}
    core_attn_profile_data = {'fwd': 0.002, 'dgrad': 0.003, 'wgrad': 0}
    o_proj_profile_data = {'fwd': 0.002, 'dgrad': 0.002, 'wgrad': 0.002}
    router_profile_data = {'fwd': 0.0001, 'dgrad': 0.0001, 'wgrad': 0.0001}
    shared_experts_w1w2_profile_data = {'fwd': 0.002, 'dgrad': 0.002, 'wgrad': 0.002}
    shared_experts_w3_profile_data = {'fwd': 0.001, 'dgrad': 0.001, 'wgrad': 0.001}
    moe_dispatch_profile_data = {'fwd': 0.004, 'dgrad': 0.004, 'is_comm': True}
    moe_combine_profile_data = {'fwd': 0.006, 'dgrad': 0.006, 'is_comm': True}
    routed_experts_w1w2_profile_data = {'fwd': 0.004, 'dgrad': 0.004, 'wgrad': 0.004}
    routed_experts_w3_profile_data = {'fwd': 0.002, 'dgrad': 0.002, 'wgrad': 0.002}

    # modules
    ## attention
    attn_ln = Module(OPType.Layernorm, "pre_attn_rmsnorm",
            input_tensors=[input_states],
            act_tensors=[input_states],
            output_tensors=[attn_ln_output],
            profile_data=mem_ops_profile_data,
            )
    qkv_linear = Module(OPType.Linear, "qkv_linear",
                        input_tensors=[attn_ln_output],
                        act_tensors=[attn_ln_output],
                        output_tensors=[kv_input, q_input],
                        profile_data=qkv_linear_profile_data,
                        )
    core_attn = Module(OPType.CoreAttn, "core_attn",
                        input_tensors=[kv_input, q_input],
                        # core attn is special, which stores output tensor for backward.
                        act_tensors=[kv_input, q_input, core_attn_output],
                        output_tensors=[core_attn_output],
                        profile_data=core_attn_profile_data,
                        )
    attn_o_proj = Module(OPType.Linear, "o_proj",
                            input_tensors=[core_attn_output],
                            act_tensors=[core_attn_output],
                            output_tensors=[attn_output],
                            profile_data=o_proj_profile_data,
                        )
    post_attn_residual_add = Module(OPType.Add, "residual connection",
                                    input_tensors=[input_states, attn_output],
                                    act_tensors=[],
                                    output_tensors=[mlp_input],
                                    profile_data=mem_ops_profile_data,
                                    )
    mlp_ln = Module(OPType.Layernorm, "pre_mlp_rmsnorm",
                    input_tensors=[mlp_input],
                    act_tensors=[mlp_input],
                    output_tensors=[post_ln_mlp_input],
                    profile_data=mem_ops_profile_data,
                    )
    ## MoE
    ### routed experts
    router = Module(OPType.Linear, "router",
                    input_tensors=[post_ln_mlp_input],
                    act_tensors=[post_ln_mlp_input],
                    output_tensors=[], # ignore small outputs of router
                    profile_data=router_profile_data,
                    )
    token_dispatch = Module(OPType.AlltoAll, "dispatch",
                            input_tensors=[post_ln_mlp_input],
                            act_tensors=[], # pure communication module
                            output_tensors=[experts_input],
                            profile_data=moe_dispatch_profile_data,
                            )
    routed_experts_w1w2 = Module(OPType.GroupedLinear, "moe_w1w2",
                                input_tensors=[experts_input],
                                act_tensors=[experts_input],
                                output_tensors=[routed_experts_w1_output, routed_experts_w2_output],
                                profile_data=routed_experts_w1w2_profile_data,
                                )
    expert_probs = Module(OPType.Mul, "probs",
                          input_tensors=[routed_experts_w2_output],
                          act_tensors=[routed_experts_w2_output],
                          output_tensors=[routed_experts_w2_output_with_score],
                          profile_data=mem_ops_profile_data,
                          )
    routed_experts_silu = Module(OPType.Silu, "moe_silu",
                                    input_tensors=[routed_experts_w1_output],
                                    act_tensors=[routed_experts_w1_output],
                                    output_tensors=[routed_experts_silu_output],
                                    profile_data=mem_ops_profile_data,
                                    )
    routed_experts_mul = Module(OPType.Mul, "moe_mul",
                    input_tensors=[routed_experts_silu_output, routed_experts_w2_output_with_score],
                    act_tensors=[routed_experts_silu_output, routed_experts_w2_output_with_score],
                    output_tensors=[routed_experts_mul_output],
                    profile_data=mem_ops_profile_data,
                    )
    routed_experts_w3 = Module(OPType.GroupedLinear, "moe_w3",
                                input_tensors=[routed_experts_mul_output],
                                act_tensors=[routed_experts_mul_output],
                                output_tensors=[routed_experts_w3_output],
                                profile_data=routed_experts_w3_profile_data,
                                )
    # expert_probs = Module("Probs", "probs",
                            # input_tensors=[routed_experts_w3_output],
                            # act_tensors=[routed_experts_w3_output],
                            # output_tensors=[routed_experts_output_with_score],
                            # )
    token_combine = Module(OPType.AlltoAll, "combine",
                            input_tensors=[routed_experts_w3_output],
                            act_tensors=[],
                            output_tensors=[routed_experts_combined_output],
                            profile_data=moe_combine_profile_data,
                            )
    ### shared experts
    shared_experts_w1w2 = Module(OPType.Linear, "shared_moe_w1w2",
                            input_tensors=[post_ln_mlp_input],
                            act_tensors=[post_ln_mlp_input],
                            output_tensors=[shared_experts_w1_output, shared_experts_w2_output],
                            profile_data=shared_experts_w1w2_profile_data,
                            )
    shared_experts_silu = Module(OPType.Silu, "shared_moe_silu",
                                    input_tensors=[shared_experts_w1_output],
                                    act_tensors=[shared_experts_w1_output],
                                    output_tensors=[shared_experts_silu_output],
                                    profile_data=mem_ops_profile_data,
                                    )
    shared_experts_mul = Module(OPType.Mul, "shared_moe_mul",
                            input_tensors=[shared_experts_silu_output, shared_experts_w2_output],
                            act_tensors=[shared_experts_silu_output, shared_experts_w2_output],
                            output_tensors=[shared_experts_mul_output],
                            profile_data=mem_ops_profile_data,
                            )
    shared_experts_w3 = Module(OPType.Linear, "shared_moe_w3",
                                input_tensors=[shared_experts_mul_output],
                                act_tensors=[shared_experts_mul_output],
                                output_tensors=[shared_experts_w3_output],
                                profile_data=shared_experts_w3_profile_data,
                                )
    final_outputs = Module(OPType.Add, "add routed experts, shared experts, and residual",
                            input_tensors=[routed_experts_combined_output, shared_experts_w3_output, mlp_input],
                            act_tensors=[],
                            output_tensors=[output_states],
                            profile_data=mem_ops_profile_data,
                            )
    v1f1b_buffer = Module(OPType.Mul, "buffer",
                            input_tensors=[output_states],
                            act_tensors=[output_states],
                            output_tensors=[],
                            profile_data=mem_ops_profile_data,
                            )

    overall_modules = [attn_ln, qkv_linear, core_attn, attn_o_proj, post_attn_residual_add] \
                    + [mlp_ln, router] \
                    + [token_dispatch, \
                       routed_experts_w1w2, expert_probs, routed_experts_silu, routed_experts_mul, \
                        routed_experts_w3, token_combine] \
                    + [shared_experts_w1w2, shared_experts_silu, shared_experts_mul, \
                       shared_experts_w3] \
                    + [final_outputs, v1f1b_buffer]

    # return overall_modules, list(tensors.values())

    recompute_modules = []
    fp8_act_modules = []
    offload_modules = []
    offload_ratios = []

    # activation memory optimization
    ln_recompute = 'ln' in policy['recompute_modules']
    mlp_act_recompute = 'mlp_act' in policy['recompute_modules']

    ## recompute modules
    if ln_recompute:
        recompute_modules.append(attn_ln)
        recompute_modules.append(mlp_ln)
    if mlp_act_recompute:
        recompute_modules.append(expert_probs)
        recompute_modules.append(routed_experts_silu)
        recompute_modules.append(routed_experts_mul)
        recompute_modules.append(shared_experts_silu)
        recompute_modules.append(shared_experts_mul)

    ## fp8 modules
    if 'qkv_linear' in policy['fp8_modules']:
        fp8_act_modules.append(qkv_linear)
    if 'o_linear' in policy['fp8_modules']:
        fp8_act_modules.append(attn_o_proj)
    if 'routed_expert_up' in policy['fp8_modules']:
        fp8_act_modules.append(routed_experts_w1w2)
    if 'routed_expert_down' in policy['fp8_modules']:
        fp8_act_modules.append(routed_experts_w3)
    if 'shared_expert_up' in policy['fp8_modules']:
        fp8_act_modules.append(shared_experts_w1w2)
    if 'shared_expert_down' in policy['fp8_modules']:
        fp8_act_modules.append(shared_experts_w3)

    ## offload modules
    # TODO: add offload strategies here

    # assume full offloading
    for _ in offload_modules:
        offload_ratios.append(1.0)

    return overall_modules, recompute_modules, fp8_act_modules, \
        offload_modules, offload_ratios, tensors
