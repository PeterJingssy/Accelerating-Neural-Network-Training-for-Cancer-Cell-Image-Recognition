
def mlp_params(args):
    params = args.hidden_size * args.moe_ffn_hidden_size
    params *= 3 if args.swiglu else 2
    num_shared_experts = args.moe_shared_expert_intermediate_size // args.moe_ffn_hidden_size \
        if args.moe_shared_expert_intermediate_size is not None else 0
    expert_cnt = num_shared_experts + args.num_experts
    params *= expert_cnt
    params += args.hidden_size * args.num_experts # router W
    return params


def attn_params(args):
    if args.group_query_attention:
        wq = wo = args.hidden_size**2
        wk = wv = args.hidden_size * (args.kv_channels * args.num_attention_heads)
    else: 
        # MHA
        wq = wo = wk = wv = args.hidden_size**2

        # MLA
        # assert False
        # not tested. args.attn_type == "mla":
        # wq = (
        #     args.hidden_size * args.q_lora_rank
        #     + args.num_attn_head
        #     * (args.qk_nope_head_dim + args.qk_rope_head_dim)
        #     * args.kv_lora_rank
        # )
        # wo = args.hidden_size * args.head_dim * args.num_attn_head
        # wk = wv = args.kv_lora_rank * args.head_dim * args.num_attn_head
        # wk += args.q_lora_rank * args.head_dim * args.num_attn_head
        # wk += args.hidden_size * (args.kv_lora_rank + args.qk_rope_head_dim)
    return wq + wk + wv + wo
