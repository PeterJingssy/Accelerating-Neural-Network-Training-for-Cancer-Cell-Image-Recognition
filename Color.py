#!/usr/bin/env python3
import os, sys, time, argparse
from PIL import Image, ImageEnhance

def adjust_contrast(in_dir, out_dir, factor=1.5):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(in_dir):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','.tif','.svs')): continue
        src = os.path.join(in_dir, fname)
        dst = os.path.join(out_dir, fname + '.adj.png')
        if os.path.exists(dst): continue
        try:
            im = Image.open(src).convert('RGB')
            enhancer = ImageEnhance.Contrast(im)
            im2 = enhancer.enhance(factor)
            im2.save(dst)
            open(dst + '.ready','w').close()
        except Exception as e:
            print('contrast error', fname, e)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--factor", type=float, default=1.5)
    args = p.parse_args()
    while True:
        adjust_contrast(args.in_dir, args.out_dir, args.factor)
        time.sleep(2)