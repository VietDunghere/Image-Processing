"""Đo thời gian 1 epoch (train + val) của pipeline CŨ và MỚI trên cùng dữ liệu baseline, cùng cấu hình thí nghiệm.

CŨ  (baseline.ipynb trước khi tách module): ảnh PIL đã resize cache trong RAM -> RandomFlip + ToTensor + Normalize
     trên CPU mỗi ảnh, DataLoader num_workers=0, fp32, cudnn.deterministic=True, benchmark=False.
MỚI (src/training.py): cache uint8 tensor, flip + normalize trên GPU, AMP fp16, channels_last, cudnn.benchmark.

Chạy: python -X utf8 scripts/bench_training_speed.py  -> outputs/baseline/speed_benchmark.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import training as T  # noqa: E402

SEED = 42
dev = torch.device("cuda")
frames = T.load_splits()
y_tr = frames["train"]["label"].to_numpy()

# ------------------------------------------------ MỚI
T.seed_everything(SEED)
t0 = time.time()
cache = T.load_image_cache(frames, None, cache_key="baseline")
t_cache = time.time() - t0
model = T.make_model(dev)
crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda")
sampler = T.make_sampler(y_tr, SEED)
fg = torch.Generator().manual_seed(SEED + 10_000)
bt = {k: T.GPUBatcher(cache[k], dev) for k in ("train", "val")}
lab = torch.as_tensor(y_tr, device=dev)
new = []
torch.cuda.reset_peak_memory_stats()
for ep in range(2):                      # epoch 1 có overhead cudnn.benchmark -> báo cáo cả 2
    torch.cuda.synchronize(); t0 = time.time()
    T.train_one_epoch(model, bt["train"], lab, sampler, fg, crit, opt, scaler)
    torch.cuda.synchronize(); t1 = time.time()
    T.predict(model, bt["val"], frames["val"]["label"].to_numpy(), crit)
    torch.cuda.synchronize(); t2 = time.time()
    new.append({"train_s": t1 - t0, "val_s": t2 - t1, "epoch_s": t2 - t0})
    print("MỚI", new[-1], flush=True)
new_peak = torch.cuda.max_memory_allocated() / 2**20
del model, opt, bt, cache
torch.cuda.empty_cache()

# ------------------------------------------------ CŨ
T.seed_everything(SEED, deterministic=True)
resize = transforms.Resize((224, 224))
norm = [transforms.ToTensor(), transforms.Normalize(T.IMAGENET_MEAN, T.IMAGENET_STD)]
train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), *norm])
eval_tf = transforms.Compose(norm)


def load_resized(paths):
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(resize(im.convert("RGB")))
    return out


class DS(Dataset):
    def __init__(self, ims, labels, tf):
        self.ims, self.labels, self.tf = ims, list(labels), tf

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        return self.tf(self.ims[i]), self.labels[i]


t0 = time.time()
pil = {k: load_resized(frames[k]["path"]) for k in ("train", "val")}
t_cache_old = time.time() - t0
w = (1.0 / np.bincount(y_tr))[y_tr]
tr_loader = DataLoader(DS(pil["train"], y_tr, train_tf), batch_size=32, num_workers=0, pin_memory=True,
                       sampler=WeightedRandomSampler(w, len(y_tr), True, generator=torch.Generator().manual_seed(SEED)))
va_loader = DataLoader(DS(pil["val"], frames["val"]["label"], eval_tf), batch_size=32, num_workers=0, pin_memory=True)
m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = nn.Linear(512, 7)
m = m.to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
old = []
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize(); t0 = time.time()
m.train()
for x, y in tr_loader:
    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
    opt.zero_grad(set_to_none=True)
    loss = crit(m(x), y)
    loss.backward()
    opt.step()
    loss.item()
torch.cuda.synchronize(); t1 = time.time()
m.eval()
with torch.no_grad():
    for x, y in va_loader:
        x, y = x.to(dev), y.to(dev)
        crit(m(x), y).item()
torch.cuda.synchronize(); t2 = time.time()
old.append({"train_s": t1 - t0, "val_s": t2 - t1, "epoch_s": t2 - t0})
print("CŨ", old[-1], flush=True)
old_peak = torch.cuda.max_memory_allocated() / 2**20

res = {
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "n_train": len(y_tr), "n_val": len(frames["val"]),
    "old": {"epochs": old, "peak_vram_alloc_mb": old_peak, "cache_build_s_train_val": t_cache_old},
    "new": {"epochs": new, "peak_vram_alloc_mb": new_peak, "cache_build_or_load_s_all_splits": t_cache},
    "speedup_epoch": old[0]["epoch_s"] / new[-1]["epoch_s"],
    "note": "1 epoch = 1 lượt WRS (|train| mẫu) + 1 lượt val. Thời gian cache ảnh tính riêng.",
}
Path("outputs/baseline").mkdir(parents=True, exist_ok=True)
Path("outputs/baseline/speed_benchmark.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
