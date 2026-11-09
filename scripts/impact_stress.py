"""2.5c tầng C: stress test. Ảnh val/test bị suy giảm có kiểm soát; mô hình KHÔNG huấn luyện lại trên ảnh hỏng.

Với mỗi (loại suy giảm, mức, phép xử lý ghép cặp) so sánh 3 nhánh trên cùng ảnh hỏng (cùng seed mô hình):
    none       baseline, không xử lý
    test_only  baseline, thêm phép xử lý lúc đánh giá (triển khai không cần huấn luyện lại)
    retrain    mô hình huấn luyện trên ảnh đã xử lý (checkpoint của 2.5), thêm phép xử lý lúc đánh giá

Quy tắc kết luận giống 2.5.0: "khác biệt rõ" khi |Δ| > std giữa seed của nhánh none, CI95 bootstrap cụm lesion không chứa 0,
và cả 3 seed cùng dấu.

    python -X utf8 scripts/impact_stress.py run [--subset N] [--jobs 12]   # suy luận, lưu dự đoán (chạy lại sẽ bỏ qua phần đã có)
    python -X utf8 scripts/impact_stress.py analyze                         # bảng + hình từ dự đoán đã lưu
    python -X utf8 scripts/impact_stress.py grid                            # lưới ảnh suy giảm + sau xử lý

Đầu ra: outputs/impact/stress/*.npz, outputs/impact/stress_{metrics,delta}.csv, outputs/figures/impact/fig_c*.{pdf,png}
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda.plots import CLASSES, setup_style  # noqa: E402
from src.preprocess import get_preprocess  # noqa: E402
from src.preprocess.compare import _cluster_boot_idx  # noqa: E402
from src.preprocess.degrade import LEVELS, PAIRS, degrade  # noqa: E402
from src.training import CKPT_DIR, GPUBatcher, compute_metrics, load_splits, predict, read_rgb, resize_224  # noqa: E402

OUT, FIG = Path("outputs/impact"), Path("outputs/figures/impact")
PRED = OUT / "stress"
SEEDS = (42, 43, 44)
KIND_NAMES = {"noise": "Nhiễu Gauss (σ)", "low_contrast": "Nén tương phản (α)", "overexpose": "Phơi sáng quá (hệ số)",
              "uneven": "Chiếu sáng lệch (a)", "vignette": "Viền đen (bán kính tương đối)", "hair": "Lông tổng hợp (số sợi)"}
ARM_STYLE = {"none": ("#555555", "-", "o", "baseline, không xử lý"),
             "test_only": ("#E69F00", "--", "s", "baseline + xử lý lúc đánh giá"),
             "retrain": ("#0072B2", "-", "^", "huấn luyện trên ảnh đã xử lý")}
N_BOOT = 1000


def load_state(name, seed):
    return torch.load(CKPT_DIR / f"{name}_seed{seed}.pt", map_location="cpu", weights_only=False)["state_dict"]


def build(frame, kind, level, method, jobs):
    """Ảnh đánh giá cho một điều kiện. Trả về (tensor uint8 N×3×224×224, tỉ lệ ảnh bị phép xử lý đổi)."""
    fn = None if method == "none" else get_preprocess(method)

    def one(r):
        x = degrade(read_rgb(r.path), kind, level, r.image_id)
        if fn is None:
            return resize_224(x).transpose(2, 0, 1), False
        p = fn(x)
        return resize_224(p).transpose(2, 0, 1), not np.array_equal(p, x)

    with ThreadPoolExecutor(jobs) as ex:
        out = list(ex.map(one, frame.itertuples()))
    return torch.from_numpy(np.stack([o[0] for o in out])), float(np.mean([o[1] for o in out]))


def run(subset, jobs):
    PRED.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    frames = load_splits(subset)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device).eval()
    crit = nn.CrossEntropyLoss()
    states = {}

    def infer(name, X, y):
        out = {}
        bt = GPUBatcher(X, device)
        for s in SEEDS:
            if (name, s) not in states:
                states[(name, s)] = load_state(name, s)
            model.load_state_dict(states[(name, s)])
            probs, _ = predict(model, bt, y, crit, batch_size=128)
            out[s] = probs.argmax(1).astype(np.int8)
        return out

    tag = f"_subset{subset}" if subset else ""
    t0 = time.time()
    for split in ("val", "test"):
        f = frames[split]
        y = f.label.to_numpy()
        for kind in LEVELS:
            for level in range(4):
                base_file = PRED / f"{split}_{kind}_L{level}_none{tag}.npz"
                if not base_file.exists():
                    X, _ = build(f, kind, level, "none", jobs)
                    np.savez_compressed(base_file, fired=0.0, **{f"none_s{s}": v for s, v in infer("baseline", X, y).items()})
                    del X
                for m in PAIRS[kind]:
                    fp = PRED / f"{split}_{kind}_L{level}_{m}{tag}.npz"
                    if fp.exists():
                        continue
                    X, fired = build(f, kind, level, m, jobs)
                    a, b = infer("baseline", X, y), infer(m, X, y)
                    np.savez_compressed(fp, fired=fired, **{f"test_only_s{s}": v for s, v in a.items()}, **{f"retrain_s{s}": v for s, v in b.items()})
                    del X
                    print(f"[{(time.time() - t0) / 60:6.1f} phút] {split} {kind} L{level} {m}: kích hoạt {fired:.2%}", flush=True)
                print(f"[{(time.time() - t0) / 60:6.1f} phút] xong {split} {kind} L{level}", flush=True)


def _f1(y, p):
    cm = np.bincount(y * 7 + p, minlength=49).reshape(7, 7).astype(np.float64)
    tp, pp, ap = np.diag(cm), cm.sum(0), cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(pp + ap > 0, 2 * tp / (pp + ap), 0.0)
    return f.mean()


def _bacc(y, p):
    cm = np.bincount(y * 7 + p, minlength=49).reshape(7, 7).astype(np.float64)
    ap = cm.sum(1)
    return (np.diag(cm)[ap > 0] / ap[ap > 0]).mean()


def analyze():
    FIG.mkdir(parents=True, exist_ok=True)
    frames = load_splits()
    rows, deltas = [], []
    for split in ("val", "test"):
        f = frames[split]
        y = f.label.to_numpy()
        boot = list(_cluster_boot_idx(f.lesion_id.values, N_BOOT))
        for kind in LEVELS:
            for level in range(4):
                base = np.load(PRED / f"{split}_{kind}_L{level}_none.npz")
                none = {s: base[f"none_s{s}"] for s in SEEDS}
                for s in SEEDS:
                    rows.append(dict(split=split, kind=kind, level=level, method="none", arm="none", seed=s, fired=0.0, **compute_metrics(y, none[s])))
                sd = {"macro_f1": np.std([_f1(y, none[s]) for s in SEEDS], ddof=1), "balanced_acc": np.std([_bacc(y, none[s]) for s in SEEDS], ddof=1)}
                for m in PAIRS[kind]:
                    d = np.load(PRED / f"{split}_{kind}_L{level}_{m}.npz")
                    fired = float(d["fired"])
                    for arm in ("test_only", "retrain"):
                        p = {s: d[f"{arm}_s{s}"] for s in SEEDS}
                        for s in SEEDS:
                            rows.append(dict(split=split, kind=kind, level=level, method=m, arm=arm, seed=s, fired=fired, **compute_metrics(y, p[s])))
                        for mname, fn in (("macro_f1", _f1), ("balanced_acc", _bacc)):
                            per = np.array([fn(y, p[s]) - fn(y, none[s]) for s in SEEDS])
                            bs = np.array([np.mean([fn(y[i], p[s][i]) - fn(y[i], none[s][i]) for s in SEEDS]) for i in boot])
                            lo, hi = np.percentile(bs, [2.5, 97.5])
                            sign = f"{(per > 0).sum()}+/{(per < 0).sum()}-"
                            clear = abs(per.mean()) > sd[mname] and (lo > 0 or hi < 0) and sign in ("3+/0-", "0+/3-")
                            deltas.append(dict(split=split, kind=kind, level=level, severity=(LEVELS[kind][level - 1] if level else np.nan), method=m, arm=arm, metric=mname, fired=fired,
                                               delta=per.mean(), ci_lo=lo, ci_hi=hi, sign=sign, none_seed_std=sd[mname],
                                               verdict=("tốt hơn rõ" if per.mean() > 0 else "tệ hơn rõ") if clear else "không khác biệt rõ"))
    R, D = pd.DataFrame(rows), pd.DataFrame(deltas)
    R.to_csv(OUT / "stress_metrics.csv", index=False)
    D.to_csv(OUT / "stress_delta.csv", index=False)
    for split in ("val", "test"):
        _fig_curves(R, split)
    _fig_delta(D)
    pd.set_option("display.width", 230, "display.max_columns", 30, "display.max_rows", 500)
    print(D[(D.metric == "macro_f1")].round(4).to_string())


def _fig_curves(R, split):
    setup_style()
    rows = [(k, m) for k in LEVELS for m in PAIRS[k]]
    n = len(rows)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(12, 3.0 * nrow), sharey=True)
    ax = ax.ravel()
    T = R[R.split == split]
    for a, (k, m) in zip(ax, rows):
        x_lab = ["gốc"] + [str(v) for v in LEVELS[k]]
        for arm, sel in (("none", (T.kind == k) & (T.arm == "none")), ("test_only", (T.kind == k) & (T.method == m) & (T.arm == "test_only")), ("retrain", (T.kind == k) & (T.method == m) & (T.arm == "retrain"))):
            c, ls, mk, lab = ARM_STYLE[arm]
            g = T[sel].groupby("level").macro_f1
            a.errorbar(range(4), g.mean().values, yerr=g.std(ddof=1).values, color=c, ls=ls, marker=mk, ms=4, lw=1.3, capsize=2, label=lab)
            for lv in range(4):
                v = T[sel & (T.level == lv)].macro_f1.values
                a.scatter(np.full(len(v), lv), v, s=5, color=c, alpha=0.45, zorder=3)
        a.set_xticks(range(4), x_lab)
        fired = T[(T.kind == k) & (T.method == m) & (T.arm == "retrain")].groupby("level").fired.first().values
        a.set_title(f"{KIND_NAMES[k]}\n{m} | kích hoạt: " + "/".join(f"{v:.0%}" for v in fired), fontsize=8)
        a.set_ylabel(f"{split} macro-F1")
    for a in ax[n:]:
        a.axis("off")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_c1_stress_curves_{split}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"fig_c1_stress_curves_{split}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _fig_delta(D):
    setup_style()
    T = D[(D.metric == "macro_f1") & (D.level > 0)]
    rows = [(k, m) for k in LEVELS for m in PAIRS[k]]
    fig, ax = plt.subplots(len(("val", "test")), 2, figsize=(11, 8.2), sharex=True, sharey=True)
    for r, split in enumerate(("val", "test")):
        for c, arm in enumerate(("test_only", "retrain")):
            a = ax[r, c]
            H = np.full((len(rows), 3), np.nan)
            V = [["" for _ in range(3)] for _ in rows]
            for i, (k, m) in enumerate(rows):
                for lv in (1, 2, 3):
                    s = T[(T.split == split) & (T.arm == arm) & (T.kind == k) & (T.method == m) & (T.level == lv)].iloc[0]
                    H[i, lv - 1] = s.delta
                    V[i][lv - 1] = f"{s.delta:+.3f}" + ("*" if "rõ" in s.verdict and "không" not in s.verdict else "")
            im = a.imshow(H, cmap="RdBu", vmin=-0.3, vmax=0.3, aspect="auto")
            for i in range(len(rows)):
                for j in range(3):
                    a.text(j, i, V[i][j], ha="center", va="center", fontsize=7, color="k" if abs(H[i, j]) < 0.2 else "w")
            a.set_xticks(range(3), ["mức 1", "mức 2", "mức 3"])
            a.set_yticks(range(len(rows)), [f"{k}: {m}" for k, m in rows], fontsize=7)
            a.grid(False)
            a.set_title(f"{split} | {ARM_STYLE[arm][3]}", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.5, label="Δ macro-F1 so với baseline không xử lý")
    fig.savefig(FIG / "fig_c2_stress_delta_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_c2_stress_delta_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def grid():
    """Lưới minh họa: quy tắc chọn = ảnh val đầu tiên theo image_id mỗi lớp mel/nv/bcc, cố định cho mọi loại suy giảm."""
    setup_style()
    v = pd.read_csv("outputs/splits/val.csv").sort_values("image_id")
    r = v[v.dx == "mel"].iloc[0]
    rgb = read_rgb(r.path)
    rows = [(k, m) for k in LEVELS for m in PAIRS[k][:1]]
    fig, ax = plt.subplots(len(rows), 7, figsize=(11, 1.5 * len(rows)))
    for i, (k, m) in enumerate(rows):
        fn = get_preprocess(m)
        for j, lv in enumerate((0, 1, 2, 3)):
            x = degrade(rgb, k, lv, r.image_id)
            ax[i, j].imshow(x)
            ax[i, j].set_title("gốc" if lv == 0 else f"{k} mức {lv}", fontsize=6.5)
            if lv:
                ax[i, 3 + lv].imshow(fn(x))
                ax[i, 3 + lv].set_title(f"{m}", fontsize=6.5)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([]), a.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c3_degradation_grid.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze", "grid"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.subset, a.jobs)
    elif a.cmd == "analyze":
        analyze()
    else:
        grid()
