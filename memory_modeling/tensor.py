from utils import gen_tensor_id, DeviceStatus, DtypeStatus

from megatron.training import get_args

def args_to_params(args=None):
    if args is None:
        args = get_args()

    params = {
        "batch_size": args.micro_batch_size,
        "seq_len": args.seq_length,
        "tp": args.tensor_model_parallel_size,
        "hidden_size": args.hidden_size,

        "query_group": args.num_query_groups,
        "kv_channel": args.kv_channels,
        "attention_head": args.num_attention_heads,

        "topk":args.moe_router_topk,
        "moe_hidden_size": args.moe_ffn_hidden_size,
        "shared_hidden_size": args.moe_shared_expert_intermediate_size,

        "ffn_hidden_size": args.ffn_hidden_size,
    }
    return params

class Tensor():
    def __init__(self, shape, tensor_dict):
        self.tensor_id = gen_tensor_id()
        self.shape = shape
        self.filled_shape = None
        tensor_dict[self.tensor_id] = self
        self.input_modules = [] # modules who use this tensor as input
        self.output_modules = [] # modules who use this tensor as output
        self.act_modules = [] # modules who store this tensor as activation

    def __repr__(self):
        return f"{self.tensor_id=} {self.shape=}"

    def _set_values(self, static_params, dynamic_params):
        assert static_params is not None
        if self.shape is None:
            return

        self.filled_shape = []
        for s in self.shape:
            if isinstance(s, int):
                self.filled_shape.append(s)
            elif s.startswith("dynamic_"):
                param_name = s.replace("dynamic_","")
                self.filled_shape.append(dynamic_params[param_name])
            elif not "/" in s:
                self.filled_shape.append(static_params[s])
            else:
                split = s.split("/")
                assert len(split) == 2
                assert static_params[split[0]] % static_params[split[1]] == 0
                self.filled_shape.append(static_params[split[0]] // static_params[split[1]])

    def get_num_elements(self, max_vio):
        static_params = args_to_params()
        ntoken = static_params['batch_size'] * static_params['seq_len'] \
                * static_params['topk'] * (1 + max_vio)
        dynamic_params = {'ntoken': ntoken}
        self._set_values(static_params, dynamic_params)
        assert self.filled_shape is not None
        size = 1
        for s in self.filled_shape:
            size *= s
        return size

    def activation(self, tensor_status, max_vio = 0):
        dtype, device = tensor_status
        if device == DeviceStatus.freed:
            return 0,0
        bytes_per_element = 2 if dtype == DtypeStatus.bf16 else 1
        volume = bytes_per_element * self.get_num_elements(max_vio)
        if device == DeviceStatus.gpu:
            return volume, 0
        else:
            return 0, volume
