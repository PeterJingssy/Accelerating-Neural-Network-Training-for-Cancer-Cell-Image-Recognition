from enum import Enum

bytes_dict = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "torch.float32": 4,
}

TENSOR_CNT = 0
MODULE_CNT = 0
NODE_CNT = 0

def reset():
    global TENSOR_CNT, MODULE_CNT, NODE_CNT
    TENSOR_CNT = 0
    MODULE_CNT = 0
    NODE_CNT = 0

def gen_tensor_id():
    global TENSOR_CNT
    TENSOR_CNT += 1
    return TENSOR_CNT

def gen_module_id():
    global MODULE_CNT
    MODULE_CNT += 1
    return MODULE_CNT

def gen_node_id():
    global NODE_CNT
    NODE_CNT += 1
    return NODE_CNT


class OPType(Enum):
    Linear = 0
    Layernorm = 1
    GroupedLinear = 2
    Silu = 3
    Mul = 4
    AlltoAll = 5
    CoreAttn = 6
    Add = 7

    def __repr__(self):
        return f"{self.name}"


class DtypeStatus(Enum):
    bf16 = 1
    fp8 = 2

    def __repr__(self):
        return f"{self.name}"


class DeviceStatus(Enum):
    gpu = 1
    cpu = 2
    freed = 3

    def __repr__(self):
        return f"{self.name}"
