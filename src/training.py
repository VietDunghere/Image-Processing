"""Code train / đánh giá DÙNG CHUNG cho baseline và mọi thí nghiệm tiền xử lý.

Mọi thí nghiệm ablation phải gọi `run_experiment` (không sửa module này) để bảo đảm chỉ một yếu tố thay đổi:
hàm `preprocess_fn`. Cấu hình đóng băng nằm trong `FROZEN_CONFIG`.

Luồng dữ liệu (xác định, giống nhau cho train/val/test):
    JPEG gốc 600x450 -> RGB uint8 -> preprocess_fn (nếu có) -> PIL Resize((224,224)) bilinear
    -> cache uint8 (N,3,224,224) trong RAM (+ đĩa: outputs/cache/<key>_224.npy)
Mỗi batch train (trên GPU): flip ngang/dọc ngẫu nhiên p=0.5 -> /255 -> Normalize ImageNet (NCHW).

Ví dụ:
    from src.training import run_experiment
    summary_df = run_experiment("median3", preprocess_fn=lambda im: cv2.medianBlur(im, 3))
"""
from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import WeightedRandomSampler
from torchvision import models

# ---------------------------------------------------------------- hằng số / cấu hình đóng băng
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]      # = sorted(dx), giống mã hoá label
NUM_CLASSES = len(CLASSES)
SPLIT_DIR = Path("outputs/splits")
SPLIT_FILES = {"train": SPLIT_DIR / "train_fit.csv", "val": SPLIT_DIR / "val.csv", "test": SPLIT_DIR / "test.csv"}
CACHE_DIR = Path("outputs/cache")
CKPT_DIR = Path("outputs/checkpoints")
IMG_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

BATCH_SIZE = 32
LR, WEIGHT_DECAY = 1e-4, 1e-4
EPOCHS = 20
SEEDS = (42, 43, 44)

FROZEN_CONFIG = {
    "model": "resnet18 (torchvision, IMAGENET1K_V1), fc -> 7 lớp, fine-tune toàn bộ",
    "img_size": IMG_SIZE,
    "resize": "PIL Image.resize((224,224), BILINEAR) = transforms.Resize((224,224)) trên PIL, không giữ tỉ lệ",
    "normalize": "ImageNet mean/std",
    "augmentation_train": "flip ngang p=0.5, flip dọc p=0.5 (độc lập, mỗi ảnh), làm trên GPU",
    "imbalance": "WeightedRandomSampler, w_i = 1/n_{y_i} (train), num_samples=|train|, replacement=True, "
                 "generator=torch.Generator().manual_seed(seed)",
    "loss": "CrossEntropyLoss (không class weights)",
    "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "scheduler": "không",
    "batch_size": BATCH_SIZE,
    "model_selection": "epoch có macro-F1 cao nhất trên val; test đánh giá 1 lần với checkpoint đó",
    "split_files": [str(p).replace("\\", "/") for p in SPLIT_FILES.values()],
}

SPEED_CONFIG = {
    "amp": "torch.autocast(cuda, float16) + GradScaler cho forward/loss; softmax/metrics tính ở float32",
    "channels_last": False,
    "channels_last_note": "Mặc định tắt: trên RTX 3050 Laptop + torch 2.8 channels_last chậm ~4-5x (386 vs 80 ms/step, "
                          "scripts/bench_layout.py). Trên RTX 5060 Ti + torch 2.14 nhanh hơn ~26% (1 724 so với 1 364 ảnh/s "
                          "ở batch 32) -> bật qua run_experiment(channels_last=True) cho lần chạy outputs/runs_cl",
    "cudnn_benchmark": True,
    "cudnn_deterministic": False,
    "data_pipeline": "ảnh đã resize cache uint8 trong RAM; flip + normalize trên GPU theo batch; "
                     "không DataLoader/PIL trong vòng train",
    "eval_batch_size": 128,
    "reproducibility_note": (
        "Cùng seed cho cùng thứ tự lấy mẫu WRS, cùng flip và cùng khởi tạo fc; nhưng KHÔNG bảo đảm kết quả "
        "trùng từng bit giữa các lần chạy vì cudnn.benchmark chọn thuật toán không tất định và AMP fp16 "
        "làm tròn khác fp32. Sai khác kỳ vọng nhỏ hơn dao động giữa các seed; so sánh thí nghiệm dựa trên "
        "mean ± std qua 3 seed, không dựa trên 1 lần chạy."),
}


# ---------------------------------------------------------------- tiện ích
def seed_everything(seed: int, deterministic: bool = False):
    """Seed random/numpy/torch/cuda. deterministic=False -> cudnn.benchmark (nhanh, không tất định từng bit)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


class _Logger:
    def __init__(self, *files):
        self.files = [Path(f) for f in files if f is not None]
        for f in self.files:
            f.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        for f in self.files:
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# ---------------------------------------------------------------- dữ liệu
def load_splits(subset: Optional[int] = None, subset_seed: int = 0) -> dict[str, pd.DataFrame]:
    """Đọc split cố định outputs/splits/{train_fit,val,test}.csv. subset=n: lấy ngẫu nhiên n ảnh mỗi split (chạy thử)."""
    out = {}
    for k, p in SPLIT_FILES.items():
        df = pd.read_csv(p)
        assert (df["dx"].map({c: i for i, c in enumerate(CLASSES)}) == df["label"]).all(), f"label lệch ở {p}"
        if subset is not None and subset < len(df):
            df = df.sample(n=subset, random_state=subset_seed).sort_index()
        out[k] = df.reset_index(drop=True)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert set(out[a]["lesion_id"]).isdisjoint(out[b]["lesion_id"]), f"Rò rỉ lesion_id {a}/{b}"
    return out


def read_rgb(path) -> np.ndarray:
    """JPEG -> ndarray uint8 RGB (H,W,3) ở độ phân giải gốc."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize_224(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize giống transforms.Resize((224,224)) trên PIL (bilinear + antialias của PIL)."""
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def _load_one(path, preprocess_fn):
    img = read_rgb(path)
    if preprocess_fn is not None:
        h, w = img.shape[:2]
        out = preprocess_fn(img)
        assert isinstance(out, np.ndarray) and out.dtype == np.uint8 and out.shape == img.shape, \
            f"preprocess_fn phải trả về uint8 cùng shape {img.shape}, nhận {getattr(out, 'shape', None)} " \
            f"{getattr(out, 'dtype', None)}"
    else:
        out = img
    return resize_224(out).transpose(2, 0, 1)          # (3,224,224) uint8


def fn_fingerprint(fn) -> Optional[str]:
    """Dấu vân tay của preprocess_fn (mã nguồn + tham số partial). None nếu không xác định được."""
    if fn is None:
        return "none"
    try:
        if isinstance(fn, functools.partial):
            src = inspect.getsource(fn.func) + repr(fn.args) + repr(sorted(fn.keywords.items()))
        else:
            src = inspect.getsource(fn)
            if getattr(fn, "__closure__", None):
                src += repr([c.cell_contents for c in fn.__closure__])
    except (OSError, TypeError):
        return None
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


def load_image_cache(frames: dict[str, pd.DataFrame], preprocess_fn: Optional[Callable] = None,
                     cache_key: Optional[str] = None, n_jobs: int = 12, log=print) -> dict[str, torch.Tensor]:
    """Nạp ảnh (sau preprocess_fn + resize) thành tensor uint8 (N,3,224,224) cho từng split.

    Cache đĩa `outputs/cache/<cache_key>_224.npy` chứa TOÀN BỘ ảnh của 3 split (thứ tự trong `_ids.csv`);
    meta lưu fingerprint của preprocess_fn, nếu mã nguồn hàm đổi thì tính lại. cache_key=None -> không cache đĩa.
    """
    all_ids = pd.concat([f[["image_id", "path"]] for f in frames.values()]).drop_duplicates("image_id")
    fp = fn_fingerprint(preprocess_fn)
    arr, id_pos = None, None
    if cache_key is not None and fp is not None:
        npy, ids_csv, meta_f = (CACHE_DIR / f"{cache_key}_224.npy", CACHE_DIR / f"{cache_key}_224_ids.csv",
                                CACHE_DIR / f"{cache_key}_224_meta.json")
        if npy.exists() and meta_f.exists():
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fp:
                ids = pd.read_csv(ids_csv)["image_id"]
                id_pos = dict(zip(ids, range(len(ids))))
                if all_ids["image_id"].isin(id_pos).all():
                    arr = np.load(npy)            # đọc hết vào RAM (~1.5GB) nhanh hơn memmap + fancy index
                    log(f"Đọc cache {npy} ({arr.shape})")
                else:
                    arr = None
            else:
                log(f"Cache {npy} có fingerprint khác ({meta.get('fingerprint')} != {fp}) -> tính lại")
        if arr is None:
            full = pd.concat([pd.read_csv(p)[["image_id", "path"]] for p in SPLIT_FILES.values()])
            full = full.drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
            t0 = time.time()
            with ThreadPoolExecutor(n_jobs) as ex:
                data = np.stack(list(ex.map(lambda p: _load_one(p, preprocess_fn), full["path"])))
            log(f"Giải mã + preprocess + resize {len(full)} ảnh trong {time.time() - t0:.0f}s")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(npy, data)
            full[["image_id"]].to_csv(ids_csv, index=False)
            meta_f.write_text(json.dumps({"cache_key": cache_key, "fingerprint": fp, "shape": list(data.shape),
                                          "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                              encoding="utf-8")
            log(f"Đã lưu cache {npy}")
            arr, id_pos = data, dict(zip(full["image_id"], range(len(full))))
        return {k: torch.from_numpy(np.ascontiguousarray(arr[[id_pos[i] for i in f["image_id"]]]))
                for k, f in frames.items()}

    if cache_key is not None and fp is None:
        log("Không xác định được fingerprint của preprocess_fn -> không cache đĩa")
    out = {}
    t0 = time.time()
    with ThreadPoolExecutor(n_jobs) as ex:
        for k, f in frames.items():
            out[k] = torch.from_numpy(np.stack(list(ex.map(lambda p: _load_one(p, preprocess_fn), f["path"]))))
    log(f"Giải mã + preprocess + resize {sum(len(v) for v in out.values())} ảnh trong {time.time() - t0:.0f}s")
    return out


class GPUBatcher:
    """Thay DataLoader: lấy batch từ cache uint8, chuyển lên GPU, flip + normalize trên GPU.
    channels_last=True: trả batch ở định dạng NHWC trong bộ nhớ (cùng giá trị, chỉ khác cách xếp) → nhanh hơn
    khoảng 26% trên RTX 5060 Ti; trên RTX 3050 Laptop thì chậm hơn (SPEED_CONFIG)."""

    def __init__(self, images_u8: torch.Tensor, device, pin: bool = True, channels_last: bool = False):
        self.x = images_u8.pin_memory() if (pin and device.type == "cuda") else images_u8
        self.device = device
        self.channels_last = channels_last
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255

    def get(self, idx: torch.Tensor, flip_h: Optional[torch.Tensor] = None, flip_v: Optional[torch.Tensor] = None):
        x = self.x[idx].to(self.device, non_blocking=True)
        if flip_h is not None:
            fh, fv = flip_h.to(self.device).view(-1, 1, 1, 1), flip_v.to(self.device).view(-1, 1, 1, 1)
            x = torch.where(fh, x.flip(3), x)
            x = torch.where(fv, x.flip(2), x)
        x = (x.float() - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x


def make_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    """WRS: w_i = 1/n_{y_i}, num_samples = |train|, có hoàn lại, generator theo seed."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    w = 1.0 / counts[labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels),
                                 replacement=True, generator=g)


# ---------------------------------------------------------------- mô hình / train / eval
def make_model(device, channels_last: bool = False) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)   # 512 -> 7 lớp
    m = m.to(device)
    return m.to(memory_format=torch.channels_last) if channels_last else m


def train_one_epoch(model, batcher: GPUBatcher, labels_gpu, sampler, flip_gen, criterion, optimizer, scaler,
                    batch_size: int = BATCH_SIZE, amp: bool = True) -> float:
    model.train()
    order = torch.as_tensor(list(iter(sampler)), dtype=torch.long)     # 1 lượt WRS = 1 epoch
    total = torch.zeros((), device=batcher.device)
    for s in range(0, len(order), batch_size):                          # drop_last=False như DataLoader
        idx = order[s:s + batch_size]
        flips = torch.rand(2, len(idx), generator=flip_gen) < 0.5
        x = batcher.get(idx, flips[0], flips[1])
        y = labels_gpu[idx.to(batcher.device)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.detach().float() * len(idx)
    return (total / len(order)).item()


@torch.no_grad()
def predict(model, batcher: GPUBatcher, labels: np.ndarray, criterion, batch_size: int = 128, amp: bool = True):
    """Trả về (probs (N,7) float32 numpy, loss trung bình)."""
    model.eval()
    probs, total = [], 0.0
    y_all = torch.as_tensor(labels, dtype=torch.long, device=batcher.device)
    for s in range(0, len(labels), batch_size):
        idx = torch.arange(s, min(s + batch_size, len(labels)))
        x = batcher.get(idx)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
        logits = logits.float()
        total += criterion(logits, y_all[s:s + len(idx)]).item() * len(idx)
        probs.append(torch.softmax(logits, 1).cpu())
    return torch.cat(probs).numpy(), total / len(labels)


def compute_metrics(y_true, y_pred) -> dict:
    """macro_f1, balanced_acc, acc, weighted_f1 + precision/recall/f1 từng lớp."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    m = {"macro_f1": f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average="macro", zero_division=0),
         "balanced_acc": balanced_accuracy_score(y_true, y_pred),
         "acc": accuracy_score(y_true, y_pred),
         "weighted_f1": f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average="weighted", zero_division=0)}
    for i, c in enumerate(CLASSES):
        m[f"f1_{c}"], m[f"precision_{c}"], m[f"recall_{c}"] = float(f[i]), float(p[i]), float(r[i])
    return {k: float(v) for k, v in m.items()}


def save_preds(frame: pd.DataFrame, probs: np.ndarray, path):
    out = frame[["image_id", "lesion_id", "dx", "label"]].copy()
    out["pred"] = probs.argmax(1)
    out["pred_dx"] = out["pred"].map(dict(enumerate(CLASSES)))
    out["correct"] = out["pred"] == out["label"]
    for i, c in enumerate(CLASSES):
        out[f"prob_{c}"] = probs[:, i]
    out.to_csv(path, index=False)


def _summary_row(res: dict) -> dict:
    row = {"seed": res["seed"], "best_epoch": res["best_epoch"]}
    for split in ("val", "test"):
        row.update({f"{split}_{k}": v for k, v in res[split].items()})
    return row


def summarize(summary_df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std (ddof=1) qua các seed cho mọi cột số."""
    num = summary_df.drop(columns=["seed"]).select_dtypes("number")
    return num.agg(["mean", "std"]).T


def train_seed(name: str, seed: int, frames, cache, run_dir: Path, ckpt_path: Path, epochs: int, device,
               log, amp: bool = True, eval_batch_size: int = 128, channels_last: bool = False) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    y = {k: f["label"].to_numpy().copy() for k, f in frames.items()}
    model = make_model(device, channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    sampler = make_sampler(y["train"], seed)
    flip_gen = torch.Generator().manual_seed(seed + 10_000)        # flip độc lập với WRS, vẫn theo seed
    bt = {k: GPUBatcher(cache[k], device, channels_last=channels_last) for k in ("train", "val", "test")}
    labels_gpu = torch.as_tensor(y["train"], dtype=torch.long, device=device)

    history, best_f1, best_state, best_epoch = [], -1.0, None, -1
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, bt["train"], labels_gpu, sampler, flip_gen, criterion, optimizer,
                                  scaler, amp=amp)
        t_train = time.time() - t0
        pv, va_loss = predict(model, bt["val"], y["val"], criterion, eval_batch_size, amp)
        mv = compute_metrics(y["val"], pv.argmax(1))
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_macro_f1": mv["macro_f1"], "val_balanced_acc": mv["balanced_acc"],
                        "val_acc": mv["acc"], "train_time_s": t_train, "epoch_time_s": dt})
        if mv["macro_f1"] > best_f1:
            best_f1, best_epoch = mv["macro_f1"], epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        log(f"[{name} seed {seed}] epoch {epoch:2d}/{epochs} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | "
            f"val_F1 {mv['macro_f1']:.4f} | val_bAcc {mv['balanced_acc']:.4f} | {dt:.1f}s")
    train_total = time.time() - t_start
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Checkpoint tốt nhất theo val -> đánh giá val + test (test chỉ 1 lần)
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "name": name, "seed": seed, "best_epoch": best_epoch}, ckpt_path)
    pv, _ = predict(model, bt["val"], y["val"], criterion, eval_batch_size, amp)
    pt, _ = predict(model, bt["test"], y["test"], criterion, eval_batch_size, amp)
    save_preds(frames["val"], pv, run_dir / "val_preds.csv")
    save_preds(frames["test"], pt, run_dir / "test_preds.csv")
    res = {
        "name": name, "seed": seed, "best_epoch": best_epoch, "epochs": epochs,
        "val": compute_metrics(y["val"], pv.argmax(1)), "test": compute_metrics(y["test"], pt.argmax(1)),
        "val_confusion_matrix": confusion_matrix(y["val"], pv.argmax(1), labels=range(NUM_CLASSES)).tolist(),
        "test_confusion_matrix": confusion_matrix(y["test"], pt.argmax(1), labels=range(NUM_CLASSES)).tolist(),
        "classes": CLASSES,
        "n": {k: len(f) for k, f in frames.items()},
        "time": {"train_total_s": train_total,
                 "epoch_mean_s": float(np.mean([h["epoch_time_s"] for h in history])),
                 "train_epoch_mean_s": float(np.mean([h["train_time_s"] for h in history]))},
        "peak_vram_mb": {"allocated": torch.cuda.max_memory_allocated() / 2**20,
                         "reserved": torch.cuda.max_memory_reserved() / 2**20} if device.type == "cuda" else None,
        "checkpoint": str(ckpt_path).replace("\\", "/"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"[{name} seed {seed}] best epoch {best_epoch} | val F1 {res['val']['macro_f1']:.4f} | "
        f"test F1 {res['test']['macro_f1']:.4f} | test bAcc {res['test']['balanced_acc']:.4f} | "
        f"{train_total / 60:.1f} phút | VRAM đỉnh {res['peak_vram_mb']}")
    del model, best_state, optimizer, bt
    torch.cuda.empty_cache()
    return res


def run_experiment(name: str, preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                   seeds: Iterable[int] = SEEDS, out_dir=None, epochs: int = EPOCHS, subset: Optional[int] = None,
                   retrain: bool = False, *, cache_key: Optional[str] = "auto", description: str = "",
                   extra_config: Optional[dict] = None, log_file=None, n_jobs: int = 12,
                   amp: bool = True, channels_last: bool = False, eval_batch_size: int = 128,
                   out_root=None) -> pd.DataFrame:
    """Chạy 1 thí nghiệm (một cấu hình tiền xử lý) với nhiều seed, lưu kết quả, trả về DataFrame 1 dòng/seed.

    name          : tên thí nghiệm ("baseline", "median3", ...). Là khoá cache ảnh -> phải duy nhất cho 1 preprocess_fn.
    preprocess_fn : f(uint8 RGB (H,W,3) gốc 600x450) -> uint8 cùng shape; áp dụng 1 lần trước resize, cho cả 3 split.
    out_dir       : mặc định outputs/baseline (name="baseline") hoặc outputs/experiments/<name>;
                    nếu subset -> outputs/experiments/_debug/<name>_subset<n>.
    subset        : chỉ lấy n ảnh mỗi split (chạy thử pipeline). Không ghi cache đĩa.
    retrain       : False -> seed đã có metrics.json thì đọc lại.
    cache_key     : "auto" = name (None nếu subset); None = không cache đĩa; str = khoá tuỳ chọn.
    log_file      : file log tiến độ bổ sung (luôn ghi out_dir/train.log).
    channels_last, eval_batch_size : chỉ ảnh hưởng tốc độ, không đổi siêu tham số (batch train, lr giữ nguyên).
    out_root      : thư mục gốc riêng cho một "lần chạy" (vd. outputs/runs_cl): kết quả ở out_root/<name>,
                    checkpoint ở out_root/checkpoints → không đè kết quả của lần chạy khác.
    """
    seeds = list(seeds)
    tag = name if subset is None else f"{name}_subset{subset}"
    if out_dir is None and out_root is not None and subset is None:
        out_dir = Path(out_root) / name
    if out_dir is None:
        out_dir = (Path("outputs/baseline") if name == "baseline" else Path("outputs/experiments") / name) \
            if subset is None else Path("outputs/experiments/_debug") / tag
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _Logger(out_dir / "train.log", log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        amp = False

    todo = [s for s in seeds if retrain or not (out_dir / f"seed{s}" / "metrics.json").exists()]
    results = {}
    for s in seeds:
        if s not in todo:
            log(f"[{tag} seed {s}] đã có metrics.json -> đọc lại")
            results[s] = json.loads((out_dir / f"seed{s}" / "metrics.json").read_text(encoding="utf-8"))

    config = {"name": name, "description": description, **FROZEN_CONFIG, "epochs": epochs, "seeds": seeds,
              "image_processing": "không (baseline)" if preprocess_fn is None else
              getattr(preprocess_fn, "__name__", repr(preprocess_fn)),
              "preprocess_fingerprint": fn_fingerprint(preprocess_fn),
              "preprocess_order": "ảnh gốc 600x450 -> preprocess_fn -> resize 224 -> cache; giống nhau train/val/test",
              "subset": subset, "speed": {**SPEED_CONFIG, "amp_enabled": amp, "channels_last_enabled": channels_last,
                                          "eval_batch_size_used": eval_batch_size},
              "env": {"torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
              **(extra_config or {})}

    if todo:
        frames = load_splits(subset)
        config["split_sizes"] = {k: len(f) for k, f in frames.items()}
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        key = (name if subset is None else None) if cache_key == "auto" else cache_key
        log(f"[{tag}] nạp ảnh (cache_key={key}) | split {config['split_sizes']} | seed cần train: {todo}")
        cache = load_image_cache(frames, preprocess_fn, key, n_jobs=n_jobs, log=log)
        for s in todo:
            ckpt = (Path(out_root) / "checkpoints" if out_root is not None else CKPT_DIR) / f"{tag}_seed{s}.pt"
            results[s] = train_seed(name, s, frames, cache, out_dir / f"seed{s}", ckpt, epochs, device, log, amp,
                                    eval_batch_size=eval_batch_size, channels_last=channels_last)
        del cache
    elif not (out_dir / "config.json").exists():
        (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_df = pd.DataFrame([_summary_row(results[s]) for s in seeds])
    summary_df.insert(0, "name", name)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    summarize(summary_df).to_csv(out_dir / "summary_mean_std.csv")
    return summary_df


def load_results(out_dir, seeds: Iterable[int] = SEEDS) -> list[dict]:
    """Đọc metrics.json của các seed (để vẽ hình/so sánh mà không train)."""
    return [json.loads((Path(out_dir) / f"seed{s}" / "metrics.json").read_text(encoding="utf-8")) for s in seeds]
