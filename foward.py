import os, sys, time, argparse, json
from PIL import Image
import torch, torch.nn as nn, torchvision.transforms as T
import torch.nn.functional as F

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

transform = T.Compose([T.Resize((224,224)), T.ToTensor()])

def worker_loop(seg_dir, out_dir, device, poll_interval=1.0):
    os.makedirs(out_dir, exist_ok=True)
    model = SimpleModel().to(device).eval()
    while True:
        files = [f for f in os.listdir(seg_dir) if f.endswith('.png')]
        for f in files:
            src = os.path.join(seg_dir, f)
            ready = src + '.ready'
            done = os.path.join(out_dir, f + '.prob')
            if not os.path.exists(ready) or os.path.exists(done): continue
            try:
                img = Image.open(src).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    prob = torch.sigmoid(logits).item()
                payload = {"patch": f, "prob": float(prob)}
                with open(done, 'w') as fh:
                    json.dump(payload, fh)
                open(done + '.done','w').close()
            except Exception as e:
                print('forward error', f, e)
        time.sleep(poll_interval)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seg_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    worker_loop(args.seg_dir, args.out_dir, args.device)