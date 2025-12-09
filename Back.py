#!/usr/bin/env python3
import os, sys, time, argparse, json
from PIL import Image
import torch, torch.nn as nn, torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class PatchTrainDataset(Dataset):
    def __init__(self, seg_dir, labels_map=None, transform=None):
        self.files = [f for f in os.listdir(seg_dir) if f.endswith('.png')]
        self.seg_dir = seg_dir
        self.transform = transform
        self.labels_map = labels_map or {}
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(os.path.join(self.seg_dir,f)).convert('RGB')
        if self.transform: img = self.transform(img)
        label = self.labels_map.get(f, 0)
        return img, torch.tensor(label, dtype=torch.float), f

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32,1)
    def forward(self,x):
        x = self.conv(x).flatten(1)
        return self.fc(x).squeeze(1)

def trainer_loop(seg_dir, probs_dir, ckpt_dir, device, epochs=3, bs=8):
    os.makedirs(ckpt_dir, exist_ok=True)
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    ds = PatchTrainDataset(seg_dir, transform=transform)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
    model = SimpleModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for imgs, labels, fnames in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model_ep{ep}.pt'))
        # optional compare with forward probs
        scores = []
        truths = []
        if os.path.isdir(probs_dir):
            for fname in os.listdir(probs_dir):
                if not fname.endswith('.prob'): continue
                try:
                    data = json.load(open(os.path.join(probs_dir,fname)))
                    scores.append(data['prob'])
                except: pass
        print(f"epoch {ep} done, loss {float(loss):.4f}, forward_probs_count {len(scores)}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seg_dir", required=True)
    p.add_argument("--probs_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    trainer_loop(args.seg_dir, args.probs_dir, args.ckpt_dir, args.device, args.epochs)
