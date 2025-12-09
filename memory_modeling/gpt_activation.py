from args import get_args

from utils import bytes_dict
        
class Module():
    def __init__(self, name, description, shape, layerid, dtype="bf16", recompute=False, offload=0.0, params=None):
        self.name = name
        self.description = description
        if shape is not None and len(shape) == 0:
            shape = None
        self.shape = shape
        self.layerid = layerid
        self.recompute = recompute
        self.filled_shape = None
        self.offload = offload
        
        assert dtype in ("fp32", "bf16", "fp16", "fp8")
        self.bytes_per_element = bytes_dict[dtype]
        
        if params is not None:
            self.set_values(params)


    def set_values(self, params=None):
        assert params is not None
        if self.shape is None:
            return
        
        filled_shape = []
        for s in self.shape:
            if isinstance(s, int):
                filled_shape.append(s)
            elif s.startswith("dynamic_"):
                filled_shape.append(0)
            elif not "/" in s:
                    filled_shape.append(params[s])
            else:
                split = s.split("/")
                assert len(split) == 2
                assert params[split[0]] % params[split[1]] == 0, f"{split[0]} // {split[1]}: {params[split[0]]}, {params[split[1]]}"
                filled_shape.append(params[split[0]] // params[split[1]])

        self.filled_shape = filled_shape
        
    def set_recompute(self, recompute_modules):
        if self.name in recompute_modules:
            self.recompute = True
            
    def set_offload(self, offload_modules, ratio):
        if self.name in offload_modules:
            self.offload = ratio
            
    def set_dynamic_value(self, param_name, value):
        if self.shape is None:
            return
        
        for i, s in enumerate(self.shape):
            if isinstance(s, str) and s.startswith("dynamic_"):
                if s.replace("dynamic_","") == param_name:
                    self.filled_shape[i] = value
    
    def display(self):
        print(f"Layer {self.layerid}: {self.name} ({self.description})")
        print(f"  Shape: {self.shape}\tRecompute: {self.recompute}\tOffload: {self.offload}")
        
        if self.filled_shape is not None:
            size = self.calculate()
            print(f"  Actual Shape: {self.filled_shape}\tdtype(bytes):{self.bytes_per_element}\tSize: {(size/1024**2):2f} MB")

        print("")
        
    def calculate(self, modules=None):
        """
        Calculate the size of the activation memory for this module.
        Recompute, offload, and dynamic parameters are considered.
        If the modules list is not None, only calculate the size if this module is in the list.
        """
        if modules is not None:
            if not self.name in modules:
                return 0
        
        if self.shape is None or self.recompute:
            return 0
        
        assert self.filled_shape is not None
        
        size = self.bytes_per_element
        for s in self.filled_shape:
            size *= s
            
        size = int(size * (1.0 - self.offload))
        return size    


class Model():
    def __init__(self, name, modules):
        self.name = name
        self.modules = modules
        
    def set_values(self, params):
        for module in self.modules:
            module.set_values(params)
            
    def set_recompute(self, recompute_modules):
        assert False
        # There's some problem with automatic recompute setting
        # Firstly, recompute modules have different naming conventions
        # Secondly, when we "recompute" a module, we actually remove the activation storage of the NEXT module
        for module in self.modules:
            module.set_recompute(recompute_modules)
            
    def set_offload(self, offload_modules, ratio):
        for module in self.modules:
            module.set_offload(offload_modules, ratio)
            
    def set_dynamic_value(self, param_name, value):
        for module in self.modules:
            module.set_dynamic_value(param_name, value)
            
    def display(self):
        print(f"Model {self.name}")
        for module in self.modules:
            module.display()
            
    def calculate(self, selected_modules=None, except_modules=None):
        """
        Calculate the size of the activation memory for this model.
        Recompute, offload, and dynamic parameters are considered.
        """
        total = 0
        
        assert selected_modules is None or except_modules is None
        
        if selected_modules is not None:
            for module in self.modules:
                total += module.calculate(selected_modules)
        elif except_modules is not None:
            total_all = 0
            total_except = 0
            for module in self.modules:
                total_all += module.calculate()
                total_except += module.calculate(except_modules)
            total = total_all - total_except
        else:
            for module in self.modules:
                total += module.calculate()
            
        return total


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


class dsv3(Model):
    def __init__(self, pure_bf16_act):
        # TODO: note that recompute need to be registered manually for now
        modules = [
            # Attention
            Module("LayerNormLinear", "rmsnorm", ["batch_size", "seq_len/tp", "hidden_size"], layerid=1, dtype="fp8", recompute=False),
            Module("LayerNormLinear", "qkv_linear", ["batch_size", "seq_len/tp", "hidden_size"], layerid=2, dtype="fp8", recompute=False), # There IS division of tp, as the gather is performed again in backward

            Module("rope", "rope", None, layerid=3),
            Module("core_attn", "attention_kv", ["batch_size", "seq_len", "query_group/tp", "kv_channel", 2], layerid=4, dtype="bf16", recompute=False),
            Module("core_attn", "attention_q", ["batch_size", "seq_len", "attention_head/tp", "kv_channel"], layerid=4, dtype="bf16", recompute=False),
            Module("core_attn", "o_proj", ["batch_size", "seq_len", "attention_head/tp", "kv_channel"], layerid=5, dtype="bf16", recompute=False),
            Module("Linear", "o_proj_dup", ["batch_size", "seq_len", "attention_head/tp", "kv_channel"], layerid=5, dtype="fp8", recompute=False),
            Module("residual", "add", None, layerid=6),
            
            # MoE Core
            Module("fused_layer_norm", "rmsnorm", ["batch_size", "seq_len/tp", "hidden_size"], layerid=7, dtype="fp8", recompute=False),
            # Module("router", ["batch_size", "seq_len/tp", "hidden_size"], layerid=8, dtype="fp32", recompute=False), \
            Module("permutation", "token_permutation", None, layerid=9),
            Module("GroupedLinear", "moe_w1w2", ["dynamic_ntoken", "hidden_size"], layerid=10, dtype="fp8", recompute=False),
            Module("swiglu", "moe_silu", ["dynamic_ntoken", "moe_hidden_size"], layerid=11, dtype="bf16", recompute=True),
            Module("swiglu", "moe_mul", ["dynamic_ntoken", "moe_hidden_size", 2], layerid=12, dtype="fp8", recompute=False),
            Module("GroupedLinear", "moe_w3", ["dynamic_ntoken", "moe_hidden_size"], layerid=13, dtype="fp8", recompute=False),
            Module("permutation", "token_unpermutation", ["dynamic_ntoken", "moe_hidden_size"], layerid=14, dtype="bf16", recompute=False), # combine happens in bf16
            # Module("permutation", "token_unpermutation", ["dynamic_ntoken", "hidden_size"], layerid=14, dtype="bf16", recompute=False), # combine happens in bf16
            
            # FIXME: support dense layers
            # Dense Core
            # Module("Linear", "moe_w1w2", ["dynamic_dense", "batch_size", "seq_len/tp", "hidden_size"], layerid=10, dtype="fp8", recompute=False),
            # Module("swiglu", "moe_silu", ["dynamic_dense", "batch_size", "seq_len/tp", "ffn_hidden_size"], layerid=11, dtype="bf16", recompute=True),
            # Module("swiglu", "moe_mul", ["dynamic_dense", "batch_size", "seq_len/tp", "ffn_hidden_size", 2], layerid=12, dtype="fp8", recompute=False),
            # Module("Linear", "moe_w3", ["dynamic_dense", "batch_size", "seq_len/tp", "ffn_hidden_size"], layerid=13, dtype="fp8", recompute=False),
            
            # MoE Shared
            Module("Linear", "shared_w1w2", ["batch_size", "seq_len/tp", "hidden_size"], layerid=15, dtype="fp8", recompute=True),   # the tp here is due to the same cause as later 2
            Module("swiglu", "shared_silu", ["batch_size", "seq_len", "shared_hidden_size/tp"], layerid=16, dtype="bf16", recompute=True),
            Module("swiglu", "shared_mul", ["batch_size", "seq_len", "shared_hidden_size/tp", 2], layerid=17, dtype="fp8", recompute=False),
            Module("Linear", "shared_w3", ["batch_size", "seq_len", "shared_hidden_size/tp"], layerid=18, dtype="fp8", recompute=False),
            Module("shared", "shared_add", None, layerid=19),
            
            Module("v1f1b", "output_additional", ["batch_size", "seq_len/tp", "hidden_size"], layerid=20, dtype="fp8", recompute=False),
        ]

        if pure_bf16_act:
            for m in modules:
                m.bytes_per_element = bytes_dict['bf16']

        super().__init__("dsv3", modules)
        
class aibenchmark(Model):
    def __init__(self, pure_bf16_act):
        # TODO: note that recompute need to be registered manually for now
        modules = [
            # Attention
            Module("LayerNormLinear", "rmsnorm", ["batch_size", "seq_len/tp", "hidden_size"], layerid=1, dtype="fp8", recompute=False),
            Module("LayerNormLinear", "qkv_linear", ["batch_size", "seq_len/tp", "hidden_size"], layerid=2, dtype="fp8", recompute=False), # There IS division of tp, as the gather is performed again in backward

            Module("rope", "rope", None, layerid=3),
            # Module("core_attn", "attention_kv", ["batch_size", "seq_len", "query_group/tp", "kv_channel", 2], layerid=4, dtype="bf16", recompute=False),
            # Module("core_attn", "attention_q", ["batch_size", "seq_len", "attention_head/tp", "kv_channel"], layerid=4, dtype="bf16", recompute=False),
            Module("core_attn", "attention(qkv)", ["batch_size", "seq_len", "attention_head/tp", "kv_channel", 3], layerid=4, dtype="bf16", recompute=False),
            Module("core_attn", "o_proj", ["batch_size", "seq_len", "attention_head/tp", "kv_channel"], layerid=5, dtype="bf16", recompute=False),
            Module("Linear", "o_proj_dup", ["batch_size", "seq_len", "attention_head/tp", "kv_channel"], layerid=5, dtype="fp8", recompute=False),
            Module("residual", "add", None, layerid=6),
            
            # MoE Core
            Module("fused_layer_norm", "rmsnorm", ["batch_size", "seq_len/tp", "hidden_size"], layerid=7, dtype="fp8", recompute=False),
            # Module("router", ["batch_size", "seq_len/tp", "hidden_size"], layerid=8, dtype="fp32", recompute=False), \
            Module("permutation", "token_permutation", None, layerid=9),
            Module("GroupedLinear", "moe_w1w2", ["dynamic_ntoken", "hidden_size"], layerid=10, dtype="fp8", recompute=False),
            Module("swiglu", "moe_silu", ["dynamic_ntoken", "moe_hidden_size"], layerid=11, dtype="bf16", recompute=True),
            Module("swiglu", "moe_mul", ["dynamic_ntoken", "moe_hidden_size", 2], layerid=12, dtype="fp8", recompute=False),
            Module("GroupedLinear", "moe_w3", ["dynamic_ntoken", "moe_hidden_size"], layerid=13, dtype="fp8", recompute=False),
            Module("permutation", "token_unpermutation", ["dynamic_ntoken", "moe_hidden_size"], layerid=14, dtype="bf16", recompute=False), # combine happens in bf16
            # Module("permutation", "token_unpermutation", ["dynamic_ntoken", "hidden_size"], layerid=14, dtype="bf16", recompute=False), # combine happens in bf16
            
            # FIXME: support dense layers
            # Dense Core
            # Module("Linear", "moe_w1w2", ["dynamic_dense", "batch_size", "seq_len/tp", "hidden_size"], layerid=10, dtype="fp8", recompute=False),
            # Module("swiglu", "moe_silu", ["dynamic_dense", "batch_size", "seq_len/tp", "ffn_hidden_size"], layerid=11, dtype="bf16", recompute=True),
            # Module("swiglu", "moe_mul", ["dynamic_dense", "batch_size", "seq_len/tp", "ffn_hidden_size", 2], layerid=12, dtype="fp8", recompute=False),
            # Module("Linear", "moe_w3", ["dynamic_dense", "batch_size", "seq_len/tp", "ffn_hidden_size"], layerid=13, dtype="fp8", recompute=False),
            
            # MoE Shared
            # Module("Linear", "shared_w1w2", ["batch_size", "seq_len/tp", "hidden_size"], layerid=15, dtype="fp8", recompute=True),   # the tp here is due to the same cause as later 2
            # Module("swiglu", "shared_silu", ["batch_size", "seq_len", "shared_hidden_size/tp"], layerid=16, dtype="bf16", recompute=True),
            # Module("swiglu", "shared_mul", ["batch_size", "seq_len", "shared_hidden_size/tp", 2], layerid=17, dtype="fp8", recompute=False),
            # Module("Linear", "shared_w3", ["batch_size", "seq_len", "shared_hidden_size/tp"], layerid=18, dtype="fp8", recompute=False),
            # Module("shared", "shared_add", None, layerid=19),
            
            Module("v1f1b", "output_additional", ["batch_size", "seq_len/tp", "hidden_size"], layerid=20, dtype="fp8", recompute=False),
        ]

        if pure_bf16_act:
            for m in modules:
                m.bytes_per_element = bytes_dict['bf16']

        super().__init__("aibenchmark", modules)
        

activation_model = None

def set_activation_model(model, pure_bf16_act):
    global activation_model
    if model == "dsv3":
        activation_model = dsv3(pure_bf16_act)
    elif model == "aibenchmark":
        activation_model = aibenchmark(pure_bf16_act)
    else:
        assert False, f"Unknown model {model}"
        
def get_activation_model():
    global activation_model
    if activation_model is None:
        assert False, "Activation model is not set. Please call set_activation_model() first."
    return activation_model

def calculate_memory_consumption(args=None):
    if args is None:
        args = get_args()
    return activation_model.calculate(selected_modules=args.offload_modules)