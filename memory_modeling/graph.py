from module import Module
from tensor import Tensor
from typing import List, Dict
from dataclasses import dataclass
from utils import gen_node_id

@dataclass
class Node:
    name: str = ""
    is_comm: bool = False
    t: float = 0

    def __repr__(self):
        return f"{self.name}({self.t*1000:.2f})"

def build_graph(modules, tensors):
    for m in modules:
        print(f"{m.name} {m.description} {m.module_id}")

    edges = []
    for t in tensors:
        assert len(t.output_modules) <= 1, \
            f"tensor {t=} has more than one output modules: {t.output_modules=}"
        for src_module in t.output_modules:
            for dst_module in t.input_modules:
                edges.append((src_module, dst_module))

    for e in edges:
        src, dst = e
        print(f"{src}->{dst}")


def build_nodes(
    tensors: Dict[int, Tensor],
    overall_modules: List[Module],
    recompute_modules: List[Module],
    fp8_act_modules: List[Module],
    offload_modules: List[Module],
    offload_ratios: List[float],
) -> Dict[int, Node]:
    max_vio = 0
    quant_bw = 1024 ** 4 # 1TB/s
    swap_bw = 50 * 1024 ** 3 # 50GB/s
    nodes = {}
    for m in overall_modules:
        fwd_node = Node(f"{m.type}-fwd", is_comm=m.is_comm, t=m.t_fwd)
        fwd_node_id = gen_node_id()
        nodes[fwd_node_id] = fwd_node
        dgrad_node = Node(f"{m.type}-dgrad", is_comm=m.is_comm, t=m.t_dgrad)
        dgrad_node_id = gen_node_id()

        nodes[dgrad_node_id] = dgrad_node
        if m.has_wgrad:
            wgrad_node = Node(f"{m.type}-wgrad", is_comm=m.is_comm, t=m.t_wgrad)
            wgrad_node_id = gen_node_id()
            nodes[wgrad_node_id] = wgrad_node
        else:
            wgrad_node_id = None

        if m in recompute_modules:
            refwd_node = Node(f"{m.type}-refwd", is_comm=m.is_comm, t=m.t_fwd)
            refwd_node_id = gen_node_id()
            nodes[refwd_node_id] = refwd_node
        else:
            refwd_node_id = None

        act_num_element = m.get_activation_num_element(tensors, max_vio)
        if m in fp8_act_modules:
            assert act_num_element is not None, f"[DEBUG] {m=}"
            t_quant = act_num_element / quant_bw
            quant_node = Node(f"{m.type}-quant", is_comm=False, t=t_quant)
            quant_node_id = gen_node_id()
            nodes[quant_node_id] = quant_node
            dequant_node = Node(f"{m.type}-dequant", is_comm=False, t=t_quant)
            dequant_node_id = gen_node_id()
            nodes[dequant_node_id] = dequant_node
        else:
            quant_node_id = None
            dequant_node_id = None

        if m in offload_modules:
            ratio = offload_ratios[offload_modules.index(m)]
            t_swap = ratio * act_num_element / swap_bw
            offload_node = Node(f"{m.type}-offload", is_comm=True, t=t_swap)
            offload_node_id = gen_node_id()
            nodes[offload_node_id] = offload_node
            onload_node = Node(f"{m.type}-onload", is_comm=True, t=t_swap)
            onload_node_id = gen_node_id()
            nodes[onload_node_id] = onload_node
        else:
            offload_node_id = None
            onload_node_id = None

        m.node_ids = {
            'fwd': fwd_node_id,
            'dgrad': dgrad_node_id,
            'wgrad': wgrad_node_id,
            'refwd': refwd_node_id,
            'quant': quant_node_id,
            'dequant': dequant_node_id,
            'offload': offload_node_id,
            'onload': onload_node_id,
        }
    return nodes


def build_edges(
    tensors: Dict[int, Tensor],
    overall_modules: Dict[int, Module],
):
    fwd_edges = []
    bwd_edges = []
    # following rules:
    # 1. fwd order
    # 2. bwd dgrad order
    # 3. bwd wgrad order
    for tid in tensors.keys():
        t = tensors[tid]
        assert len(t.output_modules) <= 1
        if len(t.output_modules) == 1:
            src_module_id = t.output_modules[0]
            src_module = overall_modules[src_module_id]
            src_fwd_node_id = src_module.node_ids['fwd']
            src_dgrad_node_id = src_module.node_ids['dgrad']
            src_wgrad_node_id = src_module.node_ids['wgrad']

            for dst_module_id in t.input_modules:
                dst_module = overall_modules[dst_module_id]
                dst_fwd_node_id = dst_module.node_ids['fwd']
                dst_dgrad_node_id = dst_module.node_ids['dgrad']

                # rule 1
                fwd_edges.append((src_fwd_node_id, dst_fwd_node_id))
                # rule 2
                bwd_edges.append((dst_dgrad_node_id, src_dgrad_node_id))

                if src_wgrad_node_id is not None:
                    # rule 3
                    bwd_edges.append((dst_dgrad_node_id, src_wgrad_node_id))

    # following rules:
    # 1. refwd before dgrad
    # 2. refwd before wgrad
    # 3. offload after fwd
    # 4. onload before dgrad
    # 5. onload before wgrad
    # 6. onload before dequant
    # 7. onload before refwd
    # 8. quant after fwd
    # 9. dequant before dgrad
    # 10.dequant before wgrad
    for _, m in overall_modules.items():
        m_fwd_node_id = m.node_ids['fwd']
        m_dgrad_node_id = m.node_ids['dgrad']
        m_wgrad_node_id = m.node_ids['wgrad']
        m_refwd_node_id = m.node_ids['refwd']
        m_offload_node_id = m.node_ids['offload']
        m_onload_node_id = m.node_ids['onload']
        m_quant_node_id = m.node_ids['quant']
        m_dequant_node_id = m.node_ids['dequant']
        if m_refwd_node_id is not None:
            # rule 1
            bwd_edges.append((m_refwd_node_id, m_dgrad_node_id))
            if m_wgrad_node_id is not None:
                # rule 2
                bwd_edges.append((m_refwd_node_id, m_wgrad_node_id))
        if m_offload_node_id is not None:
            # rule 3
            fwd_edges.append((m_fwd_node_id, m_offload_node_id))
            # rule 4
            bwd_edges.append((m_onload_node_id, m_dgrad_node_id))
            if m_wgrad_node_id is not None:
                # rule 5
                bwd_edges.append((m_onload_node_id, m_wgrad_node_id))
            if m_dequant_node_id is not None:
                # rule 6
                bwd_edges.append((m_onload_node_id, m_dequant_node_id))
            if m_refwd_node_id is not None:
                # rule 7
                bwd_edges.append((m_onload_node_id, m_refwd_node_id))
        if m_quant_node_id is not None:
            # rule 8
            fwd_edges.append((m_fwd_node_id, m_quant_node_id))
            # rule 9
            bwd_edges.append((m_dequant_node_id, m_dgrad_node_id))
            if m_wgrad_node_id is not None:
                # rule 10
                bwd_edges.append((m_dequant_node_id, m_wgrad_node_id))

    return fwd_edges, bwd_edges
