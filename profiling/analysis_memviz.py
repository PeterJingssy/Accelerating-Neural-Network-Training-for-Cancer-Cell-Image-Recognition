
import pickle
from tqdm import tqdm
from argparse import ArgumentParser

def main(args):
    with open(args.trace, "rb") as f:
        trace = pickle.load(f)

    tensor_dict = {}
    max_life = 0
    begin_time = None
    current_time = 0
    for device_traces in trace['device_traces']:
        for trace in tqdm(device_traces, desc="read device traces"):
            action = trace['action']
            addr = trace['addr']
            timestamp = trace['time_us']
            current_time = timestamp
            if begin_time is None:
                begin_time = current_time

            if addr in tensor_dict.keys():
                if action == 'free_requested':
                    life_time = timestamp - tensor_dict[addr]['alloc']
                    max_life = max(max_life, life_time)
            else:
                tensor_dict[addr] = {}

            tensor_dict[addr][action] = timestamp
            tensor_dict[addr]['all'] = trace

    for tensor in tensor_dict.keys():
        if 'free_requested' not in tensor_dict[tensor].keys() and 'alloc' in tensor_dict[tensor].keys():
            life_time = current_time - tensor_dict[tensor]['alloc']
            max_life = max(max_life, life_time)

    long_tensor = []
    for tensor in tensor_dict.keys():
        if 'free_requested' not in tensor_dict[tensor].keys():
            if 'alloc' in tensor_dict[tensor].keys():
                long_tensor.append(tensor_dict[tensor]['all'])
    
    print(f"{max_life=}")
    print(f"{len(long_tensor)=}")
    for tensor in long_tensor:
        print(tensor)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--trace", type=str, help="path to memory trace *.pickle file")

    args = parser.parse_args()
    main(args)
