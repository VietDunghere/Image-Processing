"""Phân tích 2.5 (phần 2): lông / viền tối / quá sáng / sáng không đều.

(A) Đo trên ẢNH (không CNN): phương pháp có sửa được vấn đề theo đúng chỉ số EDA không, và đổi ảnh bình thường bao
    nhiêu. Nhóm cờ ∩ train_fit (tối đa 200 ảnh) + đối chứng cùng cỡ ngẫu nhiên (seed 42) từ phần còn lại của train_fit.
(B) CNN toàn tập: mean ± std 3 seed, Δ ghép cặp seed, CI95 bootstrap cụm lesion, quy tắc "khác biệt rõ" như 2.5.0
    (dùng lại src.preprocess.compare.comparison_table).
(C) CNN trên CHÍNH nhóm có vấn đề (test chính, val kiểm tra nhất quán): Δ accuracy nhóm, Δ phần còn lại, DiD,
    CI95 bootstrap cụm lesion rút trên toàn split; số ảnh sửa/hỏng (WC/CW) từng seed.

    python -X utf8 scripts/analyze_artifacts.py
Đầu ra: outputs/preprocess_artifacts/*.csv, outputs/figures/preprocess_artifacts/*.{pdf,png}
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda.plots import setup_style  # noqa: E402
from src.eda.quality_metrics import compute_one  # noqa: E402
from src.preprocess import ARTIFACT_CANDIDATES  # noqa: E402
from src.preprocess import compare as C  # noqa: E402
from src.training import read_rgb  # noqa: E402

OUT = Path("outputs/preprocess_artifacts")
FIG = Path("outputs/figures/preprocess_artifacts")
SEEDS = (42, 43, 44)
# vấn đề → (cờ EDA, chỉ số EDA, chiều "xấu", các cấu hình)
PROBLEMS = {
    "Lông": ("F_hair", "hair_frac", "high", ["hair_bh17", "hair_bh17_cond"]),
    "Viền tối": ("F_vignette", "vignette_frac", "high", ["vig_fill", "vig_crop"]),
    "Quá sáng": ("F_overexposed", "bright_frac", "high", ["gamma_L_1.5", "gamma_L_auto"]),
    "Sáng không đều": ("F_uneven_illum", "illum_range", "high", ["flat_L_s60", "flat_L_s60_cond"]),
}
CFG2FLAG = {c: v[0] for v in PROBLEMS.values() for c in v[3]}
CONFIGS = [c for v in PROBLEMS.values() for c in v[3]]


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ (A) đo trên ảnh
def _metrics_of(rgb):
    """Chạy lại đúng bộ đo EDA (compute_one) trên ảnh đã xử lý: ghi tạm ra PNG không nén mất mát."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        rec, _ = compute_one(f.name)
    return rec


def _lab_delta(a, b):
    la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    lb = cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    return float(np.linalg.norm(la - lb, axis=2).mean())


def image_level(n_max=200):
    q = pd.read_csv("outputs/eda/quality_metrics.csv")
    fl = pd.read_csv("outputs/eda/image_flags.csv")
    q = q.merge(fl[["image_id"] + [c for c in fl if c.startswith("F_")]], on="image_id")
    q = q[q.image_id.isin(pd.read_csv("outputs/splits/train_fit.csv").image_id)]
    rows = []
    for prob, (flag, metric, _, cfgs) in PROBLEMS.items():
        g = q[q[flag]]
        g = g.sample(min(n_max, len(g)), random_state=42)
        ctrl = q[~q[flag]].sample(len(g), random_state=42)
        for grp, D in [("nhóm cờ", g), ("đối chứng", ctrl)]:
            def work(r):
                im = read_rgb(r.path)
                out = []
                for c in cfgs:
                    o = ARTIFACT_CANDIDATES[c](im)
                    changed = not np.array_equal(o, im)
                    m = _metrics_of(o) if changed else {metric: r[metric]}
                    out.append(dict(problem=prob, group=grp, config=c, image_id=r.image_id, dx=r.dx,
                                    before=r[metric], after=m[metric], changed=changed,
                                    dE76=_lab_delta(im, o) if changed else 0.0))
                return out
            with ThreadPoolExecutor(24) as ex:
                for res in ex.map(work, [r for _, r in D.iterrows()]):
                    rows += res
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "image_effect_per_image.csv", index=False)
    thr = {"Lông": 0.0582, "Viền tối": 0.01, "Quá sáng": 0.10, "Sáng không đều": 38.0}
    R["above_before"] = R.apply(lambda r: r.before > thr[r.problem], axis=1)
    R["above_after"] = R.apply(lambda r: r.after > thr[r.problem], axis=1)
    S = R.groupby(["problem", "config", "group"]).agg(
        n=("image_id", "size"), metric_before_median=("before", "median"), metric_after_median=("after", "median"),
        pct_above_thr_before=("above_before", "mean"), pct_above_thr_after=("above_after", "mean"),
        pct_images_changed=("changed", "mean"), dE76_mean=("dE76", "mean")).reset_index()
    for c in ["pct_above_thr_before", "pct_above_thr_after", "pct_images_changed"]:
        S[c] = (S[c] * 100).round(1)
    S.to_csv(OUT / "image_effect_summary.csv", index=False)
    return S


# ------------------------------------------------------------------ (C) nhóm có vấn đề
def _preds(name, split):
    P = C.load_preds(name, split)
    return {s: P[P.seed == s].set_index("image_id").sort_index() for s in SEEDS}


def subgroup(split, n_boot=1000):
    fl = pd.read_csv("outputs/eda/image_flags.csv").set_index("image_id")
    B = _preds("baseline", split)
    ids = B[42].index
    les = B[42]["lesion_id"].values
    base_c = np.stack([B[s].loc[ids, "correct"].values for s in SEEDS]).astype(float)
    base_sd = {}
    rows = []
    for cfg in CONFIGS:
        flag = CFG2FLAG[cfg]
        g = fl.loc[ids, flag].values.astype(bool)
        A = _preds(cfg, split)
        cc = np.stack([A[s].loc[ids, "correct"].values for s in SEEDS]).astype(float)
        d = cc - base_c                                                     # (3, N)
        dg, dr = d[:, g].mean(1), d[:, ~g].mean(1)                          # theo seed
        boots = []
        for idx in C._cluster_boot_idx(les, n_boot):
            gi = g[idx]
            if gi.sum() == 0:
                continue
            a, b = d[:, idx][:, gi].mean(), d[:, idx][:, ~gi].mean()
            boots.append((a, b, a - b))
        boots = np.array(boots)
        lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
        sd = base_sd.setdefault(flag, base_c[:, g].mean(1).std(ddof=1))
        wc = [int(((base_c[i] == 0) & (cc[i] == 1) & g).sum()) for i in range(3)]
        cw = [int(((base_c[i] == 1) & (cc[i] == 0) & g).sum()) for i in range(3)]
        sign = lambda v: f"{(v > 0).sum()}+/{(v < 0).sum()}-"
        clear = abs(dg.mean()) > sd and (lo[0] > 0 or hi[0] < 0) and sign(dg) in ("3+/0-", "0+/3-") \
            and (lo[2] > 0 or hi[2] < 0)
        rows.append(dict(split=split, config=cfg, flag=flag, n_group=int(g.sum()),
                         n_group_lesions=int(pd.Series(les[g]).nunique()),
                         acc_group_baseline=base_c[:, g].mean(), acc_group_method=cc[:, g].mean(),
                         d_acc_group=dg.mean(), d_acc_group_lo=lo[0], d_acc_group_hi=hi[0], d_acc_group_sign=sign(dg),
                         d_acc_rest=dr.mean(), d_acc_rest_lo=lo[1], d_acc_rest_hi=hi[1], d_acc_rest_sign=sign(dr),
                         did=(dg - dr).mean(), did_lo=lo[2], did_hi=hi[2],
                         baseline_group_seed_std=sd, wc_per_seed="; ".join(map(str, wc)),
                         cw_per_seed="; ".join(map(str, cw)),
                         verdict=("tốt hơn rõ" if dg.mean() > 0 else "tệ hơn rõ") if clear else "không khác biệt rõ"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ hình
def fig_overall(T_val, T_test):
    setup_style()
    names = ["baseline"] + CONFIGS
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=False)
    for ax, split in zip(axes, ["val", "test"]):
        for i, n in enumerate(names):
            v = C.load_summary(n)[f"{split}_macro_f1"].values
            col = "#7A7A7A" if n == "baseline" else C.CFG_COLORS[1 + list(PROBLEMS).index(
                next(p for p, x in PROBLEMS.items() if n in x[3]))]
            ax.bar(i, v.mean(), color=col, alpha=0.55, width=0.7, yerr=v.std(ddof=1), capsize=3)
            ax.scatter(np.full(3, i) + np.array([-0.15, 0, 0.15]), v, s=10, color="k", zorder=3)
        ax.axhline(C.load_summary("baseline")[f"{split}_macro_f1"].mean(), color="#7A7A7A", ls=":", lw=0.8)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        allv = np.concatenate([C.load_summary(n)[f"{split}_macro_f1"].values for n in names])
        ax.set_ylim(allv.min() - 0.03, allv.max() + 0.02)
        ax.set_title(f"{split} macro-F1 (cột = mean, thanh = ±1 std, chấm = seed 42/43/44)")
    fig.tight_layout()
    savefig(fig, "fig_a1_cnn_overall")


def fig_subgroup(SG):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    S = SG[SG.split == "test"].reset_index(drop=True)
    x = np.arange(len(S))
    for k, (col, lab, c) in enumerate([("d_acc_group", "Δ acc nhóm cờ", "#D55E00"),
                                        ("d_acc_rest", "Δ acc phần còn lại", "#7A7A7A")]):
        xx = x + (k - 0.5) * 0.3
        ax.errorbar(xx, S[col], yerr=[S[col] - S[col + "_lo"], S[col + "_hi"] - S[col]], fmt="os"[k], color=c,
                    capsize=2, ms=4, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [f"{r.config}\n(n={r.n_group})" for r in S.itertuples()], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Δ accuracy so với baseline (test)\nTB 3 seed, CI95 bootstrap cụm lesion")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig_a2_subgroup_delta_test")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("(A) đo trên ảnh ...", flush=True)
    print(image_level().to_string())
    for split in ("val", "test"):
        T = C.comparison_table(CONFIGS, split)
        T.to_csv(OUT / f"comparison_{split}.csv", index=False)
    T_val, T_test = (pd.read_csv(OUT / f"comparison_{s}.csv") for s in ("val", "test"))
    cols = ["config", "macro_f1_mean", "macro_f1_std", "macro_f1_delta", "macro_f1_delta_ci_lo",
            "macro_f1_delta_ci_hi", "macro_f1_delta_sign", "macro_f1_verdict"]
    print("(B) val\n", T_val[cols].round(4).to_string(), "\n(B) test\n", T_test[cols].round(4).to_string())
    SG = pd.concat([subgroup("test"), subgroup("val")], ignore_index=True)
    SG.to_csv(OUT / "subgroup.csv", index=False)
    print("(C)\n", SG[["split", "config", "n_group", "acc_group_baseline", "d_acc_group", "d_acc_group_lo",
                       "d_acc_group_hi", "d_acc_group_sign", "did", "did_lo", "did_hi", "wc_per_seed",
                       "cw_per_seed", "verdict"]].round(4).to_string())
    fig_overall(T_val, T_test)
    fig_subgroup(SG)
    print("Xong:", OUT, FIG)
