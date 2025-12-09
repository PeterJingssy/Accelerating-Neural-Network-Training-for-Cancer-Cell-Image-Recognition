
from typing import List, Dict, Tuple

from tensor import Tensor
from module import Module

from graph import build_edges, build_nodes, Node


class DualPlan:
    def __init__(self, name):
        self.comp_stream = []
        self.comm_stream = []
        self.comp_dependency = {}
        self.comm_dependency = {}
        self.name = name

    def add_comp_node(self, node: Node):
        assert not node.is_comm
        self.comp_stream.append(node)
        return len(self.comp_stream) - 1

    def add_comm_node(self, node: Node):
        assert node.is_comm
        self.comm_stream.append(node)
        return len(self.comm_stream) - 1

    def add_dependency(self, src, dst, dst_type):
        cur_stream = self.comm_dependency if dst_type == 'comm' \
                    else self.comp_dependency
        if dst not in cur_stream.keys():
            cur_stream[dst] = []
        cur_stream[dst].append(src)

    def _get_previous_idx(self, cur_idx, type):
        dependency = self.comm_dependency if type == 'comm' \
                else self.comp_dependency
        previous_idx = -1
        if cur_idx in dependency.keys():
            previous_idx = max(dependency[cur_idx])
        return previous_idx

    def _get_cur_type(self, comm_idx, comp_idx):
        if comm_idx >= len(self.comm_stream):
            return 'comp'
        if comp_idx >= len(self.comp_stream):
            return 'comm'
        previous_comp_idx = self._get_previous_idx(comm_idx, 'comm')
        previous_comm_idx = self._get_previous_idx(comp_idx, 'comp')
        assert previous_comm_idx < comm_idx or previous_comp_idx < comp_idx
        if previous_comm_idx < comm_idx:
            return 'comp'
        else:
            return 'comm'

    def end_to_end_time(self):
        self.comm_end_time = [None for _ in self.comm_stream]
        self.comp_end_time = [None for _ in self.comp_stream]
        t_comm = 0
        t_comp = 0
        comm_idx = 0
        comp_idx = 0
        comm_cnt = len(self.comm_stream)
        comp_cnt = len(self.comp_stream)
        while comm_idx < comm_cnt or comp_idx < comp_cnt:
            cur_type = self._get_cur_type(comm_idx, comp_idx)
            if cur_type == 'comm':
                cur_stream = self.comm_stream
                cur_idx = comm_idx
                end_time = self.comm_end_time
                previous_end_time = self.comp_end_time
                t_cur_stream = t_comm
            else:
                cur_stream = self.comp_stream
                cur_idx = comp_idx
                end_time = self.comp_end_time
                previous_end_time = self.comm_end_time
                t_cur_stream = t_comp

            previous_idx = self._get_previous_idx(cur_idx, cur_type)
            min_start_time = max(previous_end_time[previous_idx] if previous_idx >= 0 else -1, t_cur_stream)
            end_time[cur_idx] = min_start_time + cur_stream[cur_idx].t
            if cur_type == 'comm':
                t_comm = end_time[comm_idx]
                comm_idx += 1
            else:
                t_comp = end_time[comp_idx]
                comp_idx += 1

        return max(t_comm, t_comp)


    def __repr__(self):
        msg = f"{self.name} e2e={self.end_to_end_time()*1000:.2f}ms\ncomp:"
        for n in self.comp_stream:
            msg += f" {n};"
        comp_end_time = [int(t*100000)/100.0 for t in self.comp_end_time]
        comm_end_time = [int(t*100000)/100.0 for t in self.comm_end_time]
        msg += f"\n{comp_end_time}"
        msg += "\ncomm:"
        for n in self.comm_stream:
            msg += f" {n};"
        msg += f"\n{comm_end_time}"
        return msg


def build_naive_plan(
    nodes: Dict[int, Node],
    fwd_edges: List[Tuple[int,int]],
    bwd_edges: List[Tuple[int,int]],
) -> DualPlan:
    plan = DualPlan("naive schedule")

    fwd_inp_degree = {}
    bwd_inp_degree = {}
    out_nodes = {n:[] for n in nodes.keys()}
    for e in fwd_edges:
        src, dst = e
        if src not in fwd_inp_degree.keys():
            fwd_inp_degree[src] = 0
        if dst not in fwd_inp_degree.keys():
            fwd_inp_degree[dst] = 0
        fwd_inp_degree[dst] += 1
        out_nodes[src].append(dst)

    for e in bwd_edges:
        src, dst = e
        if src not in bwd_inp_degree.keys():
            bwd_inp_degree[src] = 0
        if dst not in bwd_inp_degree.keys():
            bwd_inp_degree[dst] = 0
        bwd_inp_degree[dst] += 1
        out_nodes[src].append(dst)

    fwd_ready = [n for n in fwd_inp_degree.keys() if fwd_inp_degree[n] == 0]
    bwd_ready = [n for n in bwd_inp_degree.keys() if bwd_inp_degree[n] == 0]
    stream_idx = {}
    last_ready = 'fwd'
    while len(fwd_ready) > 0 or len(bwd_ready) > 0:
        if len(fwd_ready) > 0 and len(bwd_ready) > 0:
            if last_ready == 'fwd':
                cur_ready = 'bwd'
            else:
                cur_ready = 'fwd'
        else:
            if len(fwd_ready) > 0:
                cur_ready = 'fwd'
            else:
                cur_ready = 'bwd'
        last_ready = cur_ready

        cur_list = fwd_ready if cur_ready == 'fwd' else bwd_ready
        cur_degree_list = fwd_inp_degree if cur_ready == 'fwd' else bwd_inp_degree

        cur_node_id = cur_list.pop(0)
        cur_node = nodes[cur_node_id]
        if cur_node.is_comm:
            cur_idx = plan.add_comm_node(cur_node)
            stream_idx[cur_node_id] = ('comm', cur_idx)
        else:
            cur_idx = plan.add_comp_node(cur_node)
            stream_idx[cur_node_id] = ('comp', cur_idx)
        for node in out_nodes[cur_node_id]:
            cur_degree_list[node] -= 1
            if cur_degree_list[node] == 0:
                cur_list.append(node)

    for edge in fwd_edges + bwd_edges:
        src, dst = edge
        src_type, src_idx = stream_idx[src]
        dst_type, dst_idx = stream_idx[dst]
        if src_type != dst_type:
            plan.add_dependency(src_idx, dst_idx, dst_type)

    return plan


def overlap_schedule(
    tensors: Dict[int, Tensor],
    overall_modules: List[Module],
    recompute_modules: List[Module],
    fp8_act_modules: List[Module],
    offload_modules: List[Module],
    offload_ratios: List[float],
) -> DualPlan:
    nodes = build_nodes(
        tensors, overall_modules, recompute_modules,
        fp8_act_modules, offload_modules, offload_ratios,
    )
    modules_dict = {m.module_id:m for m in overall_modules}
    fwd_edges, bwd_edges = build_edges(tensors, modules_dict)

    for m in overall_modules:
        print(f"{m=}")
    for nid, node in nodes.items():
        print(f"{nid=} {node.name=}")
    print(f"[DEBUG] {fwd_edges=}\n{bwd_edges=}")

    naive_plan = build_naive_plan(nodes, fwd_edges, bwd_edges)
    print(f"{naive_plan=}")
    return naive_plan
