import os, subprocess, time, argparse
shared = os.path.expanduser("~/pipeline_shared")
os.makedirs(shared, exist_ok=True)
in_raw = os.path.join(shared,"raw")
adj = os.path.join(shared,"raw_adj")
seg = os.path.join(shared,"seg")
probs = os.path.join(shared,"probs")
ckpt = os.path.join(shared,"ckpt")
for d in [in_raw,adj,seg,probs,ckpt]: os.makedirs(d, exist_ok=True)

# edit this mapping to match your cluster / nodes
role_map = {
    "contrast": {"node":"fuse0", "cmd":f"python3 contrast.py --in_dir {in_raw} --out_dir {adj} --factor 1.6"},
    "segment": {"node":"fuse0", "cmd":f"python3 segment.py --in_dir {adj} --out_dir {seg} --patch_size 512 --stride 512"},
    "trainer": {"node":"fuse2", "cmd":f"python3 trainer.py --seg_dir {seg} --probs_dir {probs} --ckpt_dir {ckpt} --device cuda:0 --epochs 10"},
    # spawn multiple forward workers; you can place them on different nodes
    "forward1": {"node":"fuse1", "cmd":f"python3 forward_worker.py --seg_dir {seg} --out_dir {probs} --device cuda:0"},
    "forward2": {"node":"fuse1", "cmd":f"python3 forward_worker.py --seg_dir {seg} --out_dir {probs} --device cuda:1"},
    "forward3": {"node":"fuse2", "cmd":f"python3 forward_worker.py --seg_dir {seg} --out_dir {probs} --device cuda:0"},
    "forward4": {"node":"fuse2", "cmd":f"python3 forward_worker.py --seg_dir {seg} --out_dir {probs} --device cuda:1"},
}

def launch(role, info):
    node = info['node']
    cmd = info['cmd']
    out = open(os.path.join(shared, f"{role}.log"), 'w')
    srun_cmd = ["srun","-p","Star","-w",node,"--gres=gpu:1","--pty","bash","-lc", cmd]
    print("launching", role, "on", node)
    p = subprocess.Popen(srun_cmd, stdout=out, stderr=out)
    return p

if __name__ == "__main__":
    procs = {}
    try:
        for r,info in role_map.items():
            procs[r] = launch(r,info)
            time.sleep(0.8)
        print("All roles launched, tail logs in pipeline_shared/*.log")
        # keep manager alive to report status
        while True:
            alive = {r: (p.poll() is None) for r,p in procs.items()}
            print("status:", alive)
            time.sleep(10)
    except KeyboardInterrupt:
        print("terminate children")
        for p in procs.values():
            p.terminate()