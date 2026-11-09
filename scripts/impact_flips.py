"""2.5c tầng B: dự đoán từng ảnh đổi bao nhiêu sau tiền xử lý, so với MỨC NỀN là dao động giữa các seed của baseline.

Chỉ dùng dự đoán ngoài mẫu đã lưu (val_preds.csv, test_preds.csv); không train, không dùng GPU.

Vì sao phải có mức nền: hai lần huấn luyện cùng cấu hình khác seed đã bất đồng ở nhiều ảnh, nên "x% ảnh đổi nhãn sau xử lý"
không tự nó chứng minh phép xử lý có tác động. So sánh công bằng là KHÁC SEED ở cả hai vế:
    flip(cấu hình seed s, baseline seed s')  với s ≠ s'     (6 cặp)
    flip(baseline seed s, baseline seed s')  với s < s'     (3 cặp) = mức nền

Đầu ra: outputs/impact/flips_{detail,excess}.csv; outputs/figures/impact/fig_b*.{pdf,png}

    python -X utf8 scripts/impact_flips.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda.plots import setup_style  # noqa: E402
from src.preprocess.compare import _cluster_boot_idx, run_dir  # noqa: E402

OUT, FIG = Path("outputs/impact"), Path("outputs/figures/impact")
SEEDS = (42, 43, 44)
CONFIGS = {   # cấu hình 3 seed trên máy này → cờ EDA mà nó nhắm tới
    "median3": "F_noise", "clahe_L_c2_t8": "F_low_contrast", "median3+clahe_L_c2_t8": "F_low_contrast",
    "vig_fill": "F_vignette", "gamma_L_auto": "F_overexposed", "flat_L_s60": "F_uneven_illum",
    "flat_L_s60_cond": "F_uneven_illum", "hair_bh17": "F_hair", "hair_bh17_cond": "F_hair",
}
N_BOOT = 2000


def load(name, split, order):
    out = {}
    for s in SEEDS:
        p = pd.read_csv(run_dir(name) / f"seed{s}" / f"{split}_preds.csv", usecols=["image_id", "pred"]).set_index("image_id")
        out[s] = p.loc[order, "pred"].to_numpy()
    return out


def pair_stats(a, b, y, mask=None):
    if mask is not None:
        a, b, y = a[mask], b[mask], y[mask]
    ca, cb = a == y, b == y
    return dict(n=len(y), flip_rate=float((a != b).mean()), wrong_to_right=float((~cb & ca).mean()), right_to_wrong=float((cb & ~ca).mean()),
                acc_a=float(ca.mean()), acc_b=float(cb.mean()))


def holm(p):
    p = np.asarray(p, float)
    idx = np.argsort(p)
    adj = np.empty(len(p))
    run_max = 0.0
    for r, i in enumerate(idx):
        run_max = max(run_max, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run_max
    return adj


def analyze(split, flags):
    frame = pd.read_csv(f"outputs/splits/{split}.csv")
    order, y, les = frame.image_id.values, frame.label.values, frame.lesion_id.values
    B = load("baseline", split, order)
    detail, excess = [], []
    for s, t in itertools.combinations(SEEDS, 2):
        detail.append(dict(split=split, config="baseline", kind="floor", part="all", seed_a=s, seed_b=t, **pair_stats(B[s], B[t], y)))
    boot_idx = list(_cluster_boot_idx(les, N_BOOT))
    floor_pairs = list(itertools.combinations(SEEDS, 2))
    floor_flips = np.stack([B[s] != B[t] for s, t in floor_pairs]).astype(np.float32)
    for name, flag in CONFIGS.items():
        C = load(name, split, order)
        F = flags.loc[order, flag].to_numpy()
        changed_input = flagged_changed(name, order)
        cross = [(s, t) for s in SEEDS for t in SEEDS if s != t]
        cross_flips = np.stack([C[s] != B[t] for s, t in cross]).astype(np.float32)
        for part, mask in (("all", None), ("flagged", F), ("not_flagged", ~F)):
            n = len(y) if mask is None else int(mask.sum())
            if n == 0:
                continue
            for s, t in cross:
                detail.append(dict(split=split, config=name, kind="cross_seed", part=part, seed_a=s, seed_b=t, **pair_stats(C[s], B[t], y, mask)))
            for s in SEEDS:
                detail.append(dict(split=split, config=name, kind="same_seed", part=part, seed_a=s, seed_b=s, **pair_stats(C[s], B[s], y, mask)))
            for s, t in floor_pairs:
                detail.append(dict(split=split, config=name, kind="floor", part=part, seed_a=s, seed_b=t, **pair_stats(B[s], B[t], y, mask)))
            m = np.ones(len(y), bool) if mask is None else mask
            pt = cross_flips[:, m].mean() - floor_flips[:, m].mean()
            boots = np.array([cross_flips[:, idx[m[idx]]].mean() - floor_flips[:, idx[m[idx]]].mean() for idx in boot_idx])
            p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
            excess.append(dict(split=split, config=name, part=part, n=n, flip_cross=cross_flips[:, m].mean(), flip_floor=floor_flips[:, m].mean(), excess=pt,
                               ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5), p_boot=max(p, 1 / N_BOOT),
                               input_changed_share=float(changed_input.loc[order][m].mean()) if changed_input is not None else np.nan))
    return pd.DataFrame(detail), pd.DataFrame(excess)


def flagged_changed(name, order):
    """Bản có điều kiện: ảnh đầu vào của mô hình có thực sự bị đổi không (so cache 224 với baseline)."""
    meta, base = Path(f"outputs/cache/{name}_224_ids.csv"), Path("outputs/cache/baseline_224.npy")
    arr_p = Path(f"outputs/cache/{name}_224.npy")
    if not (meta.exists() and arr_p.exists() and base.exists()):
        return None
    ids = pd.read_csv(meta).image_id.values
    b_ids = pd.read_csv("outputs/cache/baseline_224_ids.csv").image_id.values
    A = np.load(arr_p, mmap_mode="r")
    B = np.load(base, mmap_mode="r")
    assert (ids == b_ids).all()
    pos = pd.Series(np.arange(len(ids)), ids)[order].values
    chunks = [(np.asarray(A[pos[i:i + 256]]) != np.asarray(B[pos[i:i + 256]])).any((1, 2, 3)) for i in range(0, len(pos), 256)]
    return pd.Series(np.concatenate(chunks), index=order)


def fig_flips(E, D, split):
    setup_style()
    names = list(CONFIGS)
    T = E[E.split == split]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for a, part, title in zip(ax, ("all", "flagged", "not_flagged"), ("toàn tập", "ảnh bị cờ vấn đề tương ứng", "ảnh không bị cờ")):
        s = T[T.part == part].set_index("config").reindex(names)
        y = np.arange(len(names))
        a.barh(y, s.flip_cross * 100, color="#0072B2", alpha=0.75, label="cấu hình vs baseline (khác seed)")
        a.scatter(s.flip_floor * 100, y, marker="|", s=170, color="k", zorder=3, label="mức nền: baseline vs baseline")
        d = D[(D.split == split) & (D.kind == "cross_seed") & (D.part == part)]
        for i, n in enumerate(names):
            v = d[d.config == n].flip_rate.values * 100
            a.scatter(v, np.full(len(v), i), s=6, color="k", alpha=0.55, zorder=3)
            r = s.loc[n]
            if pd.notna(r.n):
                a.text(0.985, i, f"n={int(r.n)}", va="center", ha="right", fontsize=6, transform=a.get_yaxis_transform())
        a.set_yticks(y, names)
        a.invert_yaxis()
        a.set_title(f"{split}: {title}")
        a.set_xlabel("% ảnh đổi nhãn dự đoán")
    ax[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_b1_flip_vs_floor_{split}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"fig_b1_flip_vs_floor_{split}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    # Hình B2: sai→đúng và đúng→sai so với mức nền, toàn tập
    D_ = D[(D.split == split) & (D.part == "all")]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(names))
    fl = D_[(D_.kind == "floor") & (D_.config == "baseline")]
    w = D_[D_.kind == "cross_seed"].groupby("config")[["wrong_to_right", "right_to_wrong"]].mean().reindex(names) * 100
    ax.barh(y - 0.18, w.wrong_to_right, 0.34, color="#009E73", label="sai → đúng")
    ax.barh(y + 0.18, w.right_to_wrong, 0.34, color="#D55E00", label="đúng → sai")
    ax.axvline(fl.wrong_to_right.mean() * 100, color="k", ls=":", lw=1, label="mức nền (baseline khác seed)")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel(f"% ảnh {split} (TB 6 cặp khác seed)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_b2_wrong_right_{split}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"fig_b2_wrong_right_{split}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    flags = pd.read_csv("outputs/eda/image_flags.csv").set_index("image_id")
    D, E = zip(*[analyze(s, flags) for s in ("val", "test")])
    D, E = pd.concat(D, ignore_index=True), pd.concat(E, ignore_index=True)
    # Holm trong từng (split, part): 9 phép so sánh
    E["p_holm"] = np.nan
    for _, idx in E.groupby(["split", "part"]).groups.items():
        E.loc[idx, "p_holm"] = holm(E.loc[idx, "p_boot"].values)
    D.to_csv(OUT / "flips_detail.csv", index=False)
    E.to_csv(OUT / "flips_excess.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(E.round(4).to_string())
    for sp in ("val", "test"):
        fig_flips(E, D, sp)
