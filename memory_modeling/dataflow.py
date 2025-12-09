from module import Module
from tensor import Tensor
from utils import DeviceStatus, DtypeStatus
from typing import List, Dict


def dataflow_analysis(
        overall_modules: List[Module],
        recompute_modules: List[Module],
        fp8_act_modules: List[Module],
        offload_modules: List[Module],
        tensors: Dict[int, Tensor],
    ):
    recompute_ids = [m.module_id for m in recompute_modules]
    fp8_act_ids = [m.module_id for m in fp8_act_modules]
    offload_ids = [m.module_id for m in offload_modules]
    bf16_act_modules = []

    for m in overall_modules:
        is_recompute = m.module_id in recompute_ids
        is_fp8 = m.module_id in fp8_act_ids
        is_offload = m.module_id in offload_ids
        assert not(is_recompute and is_fp8), f"[BUG: concurrent recompute & fp8 activation] {m=}"
        assert not(is_recompute and is_offload), f"[BUG: concurrent recompute & offload] {m=}"
        if (not is_recompute) and (not is_fp8) and (not is_offload):
            bf16_act_modules.append(m)

    tensor_status = {}
    # writer = {}

    for tid in tensors.keys():
        t = tensors[tid]

        is_act = False
        is_fp8 = False
        is_bf16 = False
        is_gpu = False
        is_cpu = False
        for m in t.act_modules:
            if m not in recompute_ids:
                is_act = True
                if m in fp8_act_ids:
                    is_fp8 = True
                else:
                    is_bf16 = True
                if m in offload_ids:
                    is_cpu = True
                else:
                    is_gpu = True
        is_recompute_ckpt = False
        for m in t.input_modules:
            if m in recompute_ids:
                is_recompute_ckpt = True
                if m in fp8_act_ids:
                    is_fp8 = True
                else:
                    is_bf16 = True
                if m in offload_ids:
                    is_cpu = True
                else:
                    is_gpu = True
        is_recompute_output = False
        for m in t.output_modules:
            if m in recompute_ids:
                is_recompute_output = True

        if (not is_act) and (not is_recompute_ckpt):
            tensor_status[tid] = [None, DeviceStatus.freed]
            continue

        assert not (is_fp8 and is_bf16)
        assert not (is_gpu and is_cpu)

        if is_recompute_output:
            if is_act:
                assert not is_fp8
            if is_recompute_ckpt:
                assert not is_cpu
            tensor_status[tid] = [None, DeviceStatus.freed]
            continue

        tensor_status[tid] = [DtypeStatus.fp8 if is_fp8 else DtypeStatus.bf16,
                              DeviceStatus.gpu if is_gpu else DeviceStatus.cpu]

    for m in overall_modules:
        for t in m.act_ids:
            assert t in tensor_status.keys(), f"[BUG: missing tensor] {tensors[t]} {tensors[t]} {m=}"

    for tid, status in tensor_status.items():
        print(f"{tid=} {status}")

    mem_cnt = {'cpu':0, 'gpu':0}
    for tid, status in tensor_status.items():
        tensor = tensors[tid]
        g,c = tensor.activation(status)
        mem_cnt[f"id={tid:02d} {tensor.shape}"] = f"{g/1024**3:.2f},{c/1024**3:.2f}"
        mem_cnt['gpu'] += g
        mem_cnt['cpu'] += c

    return mem_cnt
