import os, sys, time, argparse
from PIL import Image
import numpy as np

def is_tissue_patch(img_np, thresh=0.15):
    gray = img_np.mean(axis=2)
    mask = gray < 240
    return mask.mean() > thresh

def segment_and_save(in_dir, out_dir, patch_size=512, stride=512):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(in_dir):
        if not fname.endswith('.adj.png'): continue
        src = os.path.join(in_dir, fname)
        base = os.path.splitext(fname)[0]
        img = Image.open(src).convert('RGB')
        W,H = img.size
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                if x+patch_size>W or y+patch_size>H: continue
                patch = img.crop((x,y,x+patch_size,y+patch_size))
                np_patch = np.array(patch)
                if not is_tissue_patch(np_patch): continue
                outname = f"{base}_{x}_{y}.png"
                outpath = os.path.join(out_dir, outname)
                if os.path.exists(outpath): continue
                patch.save(outpath)
                open(outpath + '.ready','w').close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    args = p.parse_args()
    while True:
        segment_and_save(args.in_dir, args.out_dir, args.patch_size, args.stride)