from utils import OPType, gen_module_id
from typing import List, Dict
from tensor import Tensor

class Module():
    def __init__(self, type: OPType, description,
                 input_tensors: List[Tensor]=[], act_tensors=[], output_tensors=[],
                 profile_data: Dict[str,float] = None):
        self.type = type
        self.description = description
        self.module_id = gen_module_id()
        self.node_ids = {}

        self.input_ids = []
        for t in input_tensors:
            self.input_ids.append(t.tensor_id)
            t.input_modules.append(self.module_id)

        self.act_ids = []
        for t in act_tensors:
            self.act_ids.append(t.tensor_id)
            t.act_modules.append(self.module_id)

        self.output_ids = []
        for t in output_tensors:
            self.act_ids.append(t.tensor_id)
            t.output_modules.append(self.module_id)

        self.t_fwd = profile_data['fwd']
        self.t_dgrad = profile_data['dgrad']
        if 'wgrad' in profile_data.keys():
            self.has_wgrad = True
            self.t_wgrad = profile_data['wgrad']
        else:
            self.has_wgrad = False
            self.t_wgrad = 0
        if 'is_comm' in profile_data.keys():
            self.is_comm = profile_data['is_comm']
        else:
            self.is_comm = False

    def __repr__(self):
        node_ids = {}
        for k,v in self.node_ids.items():
            if v is not None:
                node_ids[k] = v
        return f"{self.type=} {self.description=} {self.module_id=} {node_ids=}"

    def set_offload(self, offload_modules, ratio):
        if self.type in offload_modules:
            self.offload = ratio

    def get_activation_num_element(self, tensors: Dict[int, Tensor], max_vio):
        total = 0
        for tid in self.act_ids:
            t = tensors[tid]
            total += t.get_num_elements(max_vio)
        return total
