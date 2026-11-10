"""Khảo sát kiến trúc mạng tích chập (mục 2.4 & 2.7): so sánh 3 họ CNN đại diện trên cùng cấu hình đóng băng.

Ba kiến trúc được đánh giá trên ảnh không xử lý (chỉ resize 224×224 + Normalize ImageNet):
  1. resnet18           (11,18M tham số) – kết nối tắt (skip connection), trục chính của toàn bộ ablation 2.5;
  2. efficientnet_b0    ( 4,02M tham số) – MBConv đảo ngược + Squeeze-and-Excitation, cân bằng độ sâu/rộng/phân giải;
  3. mobilenet_v3_small ( 1,53M tham số) – tích chập tách chiều sâu nhẹ + h-swish + SE cho thiết bị biên.

Giữ cố định hoàn toàn FROZEN_CONFIG: split theo lesion_id, 224×224, AdamW lr=1e-4, wd=1e-4, batch 32, 20 epoch,
WeightedRandomSampler + CrossEntropyLoss không trọng số lớp, flip ngang/dọc trên train, chọn epoch theo val macro-F1,
seed 42, 43, 44.

    python -X utf8 scripts/run_backbone_comparison.py run
    python -X utf8 scripts/run_backbone_comparison.py analyze
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda.plots import CLASSES, setup_style  # noqa: E402
from src.preprocess.compare import _cluster_boot_idx  # noqa: E402
from src.training import (BATCH_SIZE, EPOCHS, FROZEN_CONFIG, LR, NUM_CLASSES, SEEDS, WEIGHT_DECAY, GPUBatcher, _Logger,
                          _summary_row, compute_metrics, load_image_cache, load_splits, make_sampler, predict,
                          save_preds, seed_everything, summarize, train_one_epoch)  # noqa: E402

OUT_ROOT = Path("outputs/backbones")
FIG_DIR = Path("outputs/figures/baseline")

BACKBONES = {
    "resnet18": ("ResNet-18", "#0072B2"),
    "efficientnet_b0": ("EfficientNet-B0", "#009E73"),
    "mobilenet_v3_small": ("MobileNetV3-Small", "#E69F00"),
}


def build_backbone(arch: str, device: torch.device) -> tuple[nn.Module, int]:
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, NUM_CLASSES)
    else:
        raise ValueError(f"Kiến trúc chưa hỗ trợ: {arch}")
    n_params = int(sum(p.numel() for p in m.parameters()))
    return m.to(device), n_params


def train_one_epoch_accum(model, batcher: GPUBatcher, labels_gpu, sampler, flip_gen, criterion, optimizer, scaler,
                          batch_size: int = BATCH_SIZE, micro_batch: int = 16, amp: bool = True) -> float:
    """Giữ đúng thứ tự WRS, cùng ma trận flip ngẫu nhiên và bước cập nhật hiệu dụng batch=32,
    nhưng chia thành micro_batch=16 để giảm một nửa đỉnh VRAM của MBConv (EfficientNet-B0)."""
    model.train()
    order = torch.as_tensor(list(iter(sampler)), dtype=torch.long)
    total = torch.zeros((), device=batcher.device)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        flips = torch.rand(2, len(idx), generator=flip_gen) < 0.5
        optimizer.zero_grad(set_to_none=True)
        n_b = len(idx)
        for m_s in range(0, n_b, micro_batch):
            sub_idx = idx[m_s:m_s + micro_batch]
            sub_fh, sub_fv = flips[0, m_s:m_s + micro_batch], flips[1, m_s:m_s + micro_batch]
            x = batcher.get(sub_idx, sub_fh, sub_fv)
            y = labels_gpu[sub_idx.to(batcher.device)]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss_raw = criterion(model(x), y)
                loss = loss_raw * (len(sub_idx) / n_b)
            scaler.scale(loss).backward()
            total += loss_raw.detach().float() * len(sub_idx)
        scaler.step(optimizer)
        scaler.update()
    return (total / len(order)).item()


def train_arch(arch: str, frames: dict, cache: dict, seeds=SEEDS, eval_batch_size: int = 16) -> pd.DataFrame:
    """Huấn luyện 3 seed cho một kiến trúc. Với resnet18, đọc lại từ outputs/baseline nếu đã có đủ."""
    if arch == "resnet18" and all((Path("outputs/baseline") / f"seed{s}" / "metrics.json").exists() for s in seeds):
        return pd.read_csv("outputs/baseline/summary.csv")

    out_dir = OUT_ROOT / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = _Logger(out_dir / "train.log", OUT_ROOT / "experiments.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    _, n_params = build_backbone(arch, torch.device("cpu"))
    cfg = {**FROZEN_CONFIG, "model": f"{arch} (IMAGENET1K_V1), head -> 7 lớp, fine-tune toàn bộ",
           "params": n_params, "epochs": EPOCHS, "seeds": list(seeds)}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    results = {}
    y = {k: f["label"].to_numpy().copy() for k, f in frames.items()}
    for seed in seeds:
        run_dir = out_dir / f"seed{seed}"
        m_file = run_dir / "metrics.json"
        if m_file.exists():
            log(f"[{arch} seed {seed}] đã có metrics.json -> đọc lại")
            results[seed] = json.loads(m_file.read_text(encoding="utf-8"))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed, deterministic=False)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            eff_micro_batch = BATCH_SIZE if vram_gb >= 12.0 else 16
        else:
            eff_micro_batch = BATCH_SIZE

        model, _ = build_backbone(arch, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        sampler = make_sampler(y["train"], seed)
        flip_gen = torch.Generator().manual_seed(seed + 10_000)
        bt = {k: GPUBatcher(cache[k], device, pin=False) for k in ("train", "val", "test")}
        labels_gpu = torch.as_tensor(y["train"], dtype=torch.long, device=device)

        history, best_f1, best_state, best_epoch = [], -1.0, None, -1
        t_start = time.time()
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            tr_loss = train_one_epoch_accum(model, bt["train"], labels_gpu, sampler, flip_gen, criterion, optimizer,
                                            scaler, batch_size=BATCH_SIZE, micro_batch=eff_micro_batch, amp=amp)
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
            log(f"[{arch} seed {seed}] ep {epoch:2d}/{EPOCHS} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | "
                f"val_F1 {mv['macro_f1']:.4f} | {dt:.1f}s")
        train_total = time.time() - t_start
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        model.load_state_dict(best_state)
        ckpt_path = ckpt_dir / f"{arch}_seed{seed}.pt"
        torch.save({"state_dict": best_state, "name": arch, "seed": seed, "best_epoch": best_epoch}, ckpt_path)
        pv, _ = predict(model, bt["val"], y["val"], criterion, eval_batch_size, amp)
        pt, _ = predict(model, bt["test"], y["test"], criterion, eval_batch_size, amp)
        save_preds(frames["val"], pv, run_dir / "val_preds.csv")
        save_preds(frames["test"], pt, run_dir / "test_preds.csv")
        res = {
            "name": arch, "seed": seed, "best_epoch": best_epoch, "epochs": EPOCHS, "params": n_params,
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
            "checkpoint": str(ckpt_path),
        }
        m_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        log(f"[{arch} seed {seed}] best ep {best_epoch} | val F1 {res['val']['macro_f1']:.4f} | "
            f"test F1 {res['test']['macro_f1']:.4f} | {train_total:.1f}s")
        results[seed] = res
        del model, best_state, optimizer, bt
        torch.cuda.empty_cache()

    summary_df = pd.DataFrame([_summary_row(results[s]) for s in seeds])
    summary_df.insert(0, "name", arch)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    summarize(summary_df).to_csv(out_dir / "summary_mean_std.csv")
    return summary_df


def _arch_dir(arch: str) -> Path:
    if arch == "resnet18" and (Path("outputs/baseline/summary.csv")).exists():
        return Path("outputs/baseline")
    return OUT_ROOT / arch


def analyze_backbones(n_boot: int = 1000):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    param_counts = {}
    for arch in BACKBONES:
        _, n_p = build_backbone(arch, torch.device("cpu"))
        param_counts[arch] = n_p

    for split in ("val", "test"):
        ref_preds = {s: pd.read_csv(_arch_dir("resnet18") / f"seed{s}" / f"{split}_preds.csv").set_index("image_id").sort_index()
                     for s in SEEDS}
        ids = ref_preds[42].index
        y = ref_preds[42]["label"].to_numpy()
        les = ref_preds[42]["lesion_id"].to_numpy()
        boots = list(_cluster_boot_idx(les, n_boot))
        pb = np.stack([ref_preds[s].loc[ids, "pred"].to_numpy() for s in SEEDS])

        rows = []
        for arch, (label, _) in BACKBONES.items():
            d_dir = _arch_dir(arch)
            S = pd.read_csv(d_dir / "summary.csv").set_index("seed")
            t_mean = np.mean([json.loads((d_dir / f"seed{s}" / "metrics.json").read_text(encoding="utf-8"))["time"]["epoch_mean_s"]
                              for s in SEEDS])
            P = {s: pd.read_csv(d_dir / f"seed{s}" / f"{split}_preds.csv").set_index("image_id").sort_index() for s in SEEDS}
            pa = np.stack([P[s].loc[ids, "pred"].to_numpy() for s in SEEDS])
            rec = {"arch": arch, "label": label, "params_M": round(param_counts[arch] / 1e6, 2),
                   "epoch_s": round(float(t_mean), 2),
                   "acc_mean": S[f"{split}_acc"].mean(), "acc_std": S[f"{split}_acc"].std(ddof=1)}
            for m, fn in (("macro_f1", lambda yt, yp: f1_score(yt, yp, labels=range(7), average="macro", zero_division=0)),
                          ("balanced_acc", balanced_accuracy_score)):
                va = np.array([fn(y, pa[i]) for i in range(len(SEEDS))])
                vb = np.array([fn(y, pb[i]) for i in range(len(SEEDS))])
                rec[f"{m}_mean"], rec[f"{m}_std"] = va.mean(), va.std(ddof=1)
                rec[f"{m}_per_seed"] = "; ".join(f"{v:.4f}" for v in va)
                if arch != "resnet18":
                    d = va - vb
                    bt = [np.mean([fn(y[idx], pa[k][idx]) - fn(y[idx], pb[k][idx]) for k in range(len(SEEDS))]) for idx in boots]
                    lo, hi = np.percentile(bt, [2.5, 97.5])
                    sign = f"{(d > 0).sum()}+/{(d < 0).sum()}-"
                    clear = abs(d.mean()) > vb.std(ddof=1) and (lo > 0 or hi < 0) and sign in ("3+/0-", "0+/3-")
                    rec.update({f"{m}_delta": d.mean(), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi, f"{m}_sign": sign,
                                f"{m}_verdict": ("tốt hơn rõ" if d.mean() > 0 else "tệ hơn rõ") if clear else "không khác biệt rõ"})
            rows.append(rec)
        T = pd.DataFrame(rows)
        T.to_csv(OUT_ROOT / f"comparison_{split}.csv", index=False)
        print(f"=== {split.upper()} ===\n", T.round(4).to_string(index=False))

    # Bảng theo lớp trên test
    pc_rows = []
    for c in CLASSES:
        r = {"dx": c}
        for arch in BACKBONES:
            S = pd.read_csv(_arch_dir(arch) / "summary.csv")
            for k in ("f1", "recall"):
                v = S[f"test_{k}_{c}"]
                r[f"{arch}_{k}_mean"] = v.mean()
                r[f"{arch}_{k}_std"] = v.std(ddof=1)
        pc_rows.append(r)
    pd.DataFrame(pc_rows).to_csv(OUT_ROOT / "per_class_test.csv", index=False)

    # Hình so sánh 3 kiến trúc
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    archs = list(BACKBONES)
    labels = [f"{BACKBONES[a][0]}\n({param_counts[a]/1e6:.1f}M)" for a in archs]
    cols = [BACKBONES[a][1] for a in archs]
    for ax, (sp, met, title) in zip(axes, [("val", "macro_f1", "(a) Validation Macro-F1"),
                                           ("test", "macro_f1", "(b) Test Macro-F1"),
                                           ("test", "balanced_acc", "(c) Test Balanced Accuracy")]):
        all_v = []
        for i, a in enumerate(archs):
            v = pd.read_csv(_arch_dir(a) / "summary.csv")[f"{sp}_{met}"].to_numpy()
            all_v.append(v)
            ax.bar(i, v.mean(), width=0.58, color=cols[i], alpha=0.8, yerr=v.std(ddof=1), capsize=3.5)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=16, color="k", zorder=3)
            ax.text(i, v.mean() + v.std(ddof=1) + 0.008, f"{v.mean():.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(all_v[0].mean(), color="#555555", ls=":", lw=0.9)
        lo, hi = np.concatenate(all_v).min(), np.concatenate(all_v).max()
        ax.set_ylim(max(0.0, lo - 0.05), min(1.0, hi + 0.06))
        ax.set_xticks(range(len(archs)), labels, fontsize=7.5)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Điểm (mean ± std, n = 3 seed)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "backbone_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "backbone_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Đã lưu hình:", FIG_DIR / "backbone_comparison.pdf")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("run", "all"):
        frames = load_splits()
        cache = load_image_cache(frames, None, "baseline", n_jobs=16)
        for arch in BACKBONES:
            train_arch(arch, frames, cache)
        del cache
    if cmd in ("analyze", "all"):
        analyze_backbones()
