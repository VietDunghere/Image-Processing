"""Giai đoạn 2 của 2.5: so sánh cấu hình tiền xử lý với baseline (CNN), kiểm định, phân tích theo nhóm ảnh có vấn đề.

Nguồn số liệu: outputs/baseline/ và outputs/experiments/<name>/ (sinh bởi src.training.run_experiment).
- Bảng: mean ± std qua seed, Δ so với baseline (ghép cặp theo seed), bootstrap cụm theo lesion_id trên ảnh.
- Quy tắc kết luận (đặt trước): "khác biệt rõ" chỉ khi |Δ trung bình| > std giữa seed của baseline (cùng split) VÀ
  CI95 bootstrap của Δ không chứa 0 VÀ cả 3 cặp seed cùng dấu; ngược lại "không khác biệt rõ".
- Huấn luyện dùng AMP + cudnn.benchmark → không tất định từng bit; dao động giữa seed đã bao gồm nguồn này.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

from src.eda.plots import CLASS_COLORS, CLASSES, setup_style

from .figures import savefig

EXP_DIR = Path("outputs/experiments")
BASE_DIR = Path("outputs/baseline")
OUT = Path("outputs/preprocess")
SEEDS = (42, 43, 44)
CFG_COLORS = ["#7A7A7A", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#882255"]


def run_dir(name):
    return BASE_DIR if name == "baseline" else EXP_DIR / name


def load_summary(name, seeds=SEEDS) -> pd.DataFrame:
    d = pd.read_csv(run_dir(name) / "summary.csv")
    d = d[d["seed"].isin(seeds)].copy()
    d["name"] = name
    return d


def load_preds(name, split, seeds=SEEDS) -> pd.DataFrame:
    out = []
    for s in seeds:
        f = run_dir(name) / f"seed{s}" / f"{split}_preds.csv"
        if f.exists():
            out.append(pd.read_csv(f, usecols=["image_id", "lesion_id", "dx", "label", "pred", "correct"]).assign(seed=s))
    return pd.concat(out, ignore_index=True)


def available_seeds(name):
    return [s for s in SEEDS if (run_dir(name) / f"seed{s}" / "metrics.json").exists()]


# ---------------- bootstrap cụm theo lesion_id ----------------
def _cluster_boot_idx(lesions: np.ndarray, n_boot: int, seed: int = 42):
    """Sinh chỉ số ảnh cho mỗi lần bootstrap: rút lại lesion (có hoàn lại), lấy mọi ảnh của lesion đó."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(lesions, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([groups[k] for k in pick])


def paired_boot_delta(name, split="test", metric="macro_f1", n_boot=1000, seeds=SEEDS, ref="baseline"):
    """Δ metric (name − ref), trung bình qua các seed ghép cặp, CI95 bootstrap cụm theo lesion_id trên ảnh của split."""
    fn = {"macro_f1": lambda y, p: f1_score(y, p, labels=range(7), average="macro", zero_division=0),
          "balanced_acc": balanced_accuracy_score}[metric]
    A, B = load_preds(name, split, seeds), load_preds(ref, split, seeds)
    ids = A[A.seed == seeds[0]][["image_id", "lesion_id", "label"]].reset_index(drop=True)
    pa = np.stack([A[A.seed == s].set_index("image_id").loc[ids.image_id, "pred"].values for s in seeds])
    pb = np.stack([B[B.seed == s].set_index("image_id").loc[ids.image_id, "pred"].values for s in seeds])
    y = ids["label"].values
    point = np.mean([fn(y, pa[i]) - fn(y, pb[i]) for i in range(len(seeds))])
    boots = []
    for idx in _cluster_boot_idx(ids["lesion_id"].values, n_boot):
        boots.append(np.mean([fn(y[idx], pa[i][idx]) - fn(y[idx], pb[i][idx]) for i in range(len(seeds))]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


# ---------------- bảng so sánh ----------------
METRICS = ["macro_f1", "balanced_acc"] + [f"f1_{c}" for c in CLASSES] + [f"recall_{c}" for c in CLASSES]


def comparison_table(names, split, n_boot=1000):
    base = load_summary("baseline").set_index("seed")
    rows = []
    for name in ["baseline"] + [n for n in names if n != "baseline"]:
        seeds = available_seeds(name)
        S = load_summary(name, seeds).set_index("seed")
        rec = {"config": name, "n_seeds": len(seeds), "seeds": ",".join(map(str, seeds))}
        for m in METRICS:
            col = f"{split}_{m}"
            rec[f"{m}_mean"], rec[f"{m}_std"] = S[col].mean(), S[col].std(ddof=1) if len(seeds) > 1 else np.nan
            if name != "baseline":
                d = S[col] - base.loc[seeds, col]
                rec[f"{m}_delta"] = d.mean()
                rec[f"{m}_delta_sign"] = f"{(d > 0).sum()}+/{(d < 0).sum()}-"
        if name != "baseline" and len(seeds) == 3:
            for m in ("macro_f1", "balanced_acc"):
                pt, lo, hi = paired_boot_delta(name, split, m, n_boot, tuple(seeds))
                rec[f"{m}_delta_ci_lo"], rec[f"{m}_delta_ci_hi"] = lo, hi
                thr = base[f"{split}_{m}"].std(ddof=1)
                clear = abs(pt) > thr and (lo > 0 or hi < 0) and rec[f"{m}_delta_sign"] in ("3+/0-", "0+/3-")
                rec[f"{m}_baseline_seed_std"] = thr
                rec[f"{m}_verdict"] = ("tốt hơn rõ" if pt > 0 else "tệ hơn rõ") if clear else "không khác biệt rõ"
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------- hình cột ----------------
def _bars(ax, names, split, metric, title):
    for i, n in enumerate(names):
        S = load_summary(n, available_seeds(n))
        v = S[f"{split}_{metric}"].values
        ax.bar(i, v.mean(), color=CFG_COLORS[i % len(CFG_COLORS)], alpha=0.55, width=0.7,
               yerr=v.std(ddof=1) if len(v) > 1 else 0, capsize=3, error_kw=dict(lw=1))
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=12, color="k", zorder=3)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_title(title)
    base = load_summary("baseline")[f"{split}_{metric}"].mean()
    ax.axhline(base, color="#7A7A7A", ls=":", lw=0.8)


def fig_overall(names):
    setup_style()
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for a, (split, m, t) in zip(ax, [("val", "macro_f1", "val macro-F1"), ("val", "balanced_acc", "val balanced acc"),
                                     ("test", "macro_f1", "test macro-F1"), ("test", "balanced_acc", "test balanced acc")]):
        _bars(a, names, split, m, t)
        vals = np.concatenate([load_summary(n, available_seeds(n))[f"{split}_{m}"].values for n in names])
        a.set_ylim(vals.min() - 0.03, vals.max() + 0.02)
    ax[0].set_ylabel("điểm (cột = mean, thanh = ±1 std, chấm = từng seed)")
    fig.tight_layout()
    savefig(fig, "fig_p6_cnn_overall")
    return fig


def fig_per_class(names, split="test"):
    setup_style()
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
    w = 0.8 / len(names)
    for r, kind in enumerate(["f1", "recall"]):
        a = ax[r]
        for i, n in enumerate(names):
            S = load_summary(n, available_seeds(n))
            for j, c in enumerate(CLASSES):
                v = S[f"{split}_{kind}_{c}"].values
                x = j + (i - (len(names) - 1) / 2) * w
                a.bar(x, v.mean(), w * 0.95, color=CFG_COLORS[i % len(CFG_COLORS)], alpha=0.6,
                      yerr=v.std(ddof=1) if len(v) > 1 else 0, capsize=1.5, error_kw=dict(lw=0.7),
                      label=n if j == 0 else None)
                a.scatter(np.full(len(v), x), v, s=4, color="k", zorder=3)
        a.set_ylabel(f"{split} {kind} (mean ± std, chấm = seed)")
        a.set_ylim(0, 1.05)
    ax[1].set_xticks(range(len(CLASSES)), CLASSES)
    for t, c in zip(ax[1].get_xticklabels(), CLASSES):
        t.set_color(CLASS_COLORS[c])
    ax[0].legend(ncol=len(names), fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    savefig(fig, f"fig_p7_cnn_per_class_{split}")
    return fig


# ---------------- nhóm ảnh có vấn đề ----------------
def subgroup_errors(names, split, flags: pd.DataFrame, n_boot=1000):
    """Tỉ lệ lỗi (trung bình qua seed) trong nhóm cờ vs phần còn lại; Δ so với baseline trên CÙNG ảnh, CI95 bootstrap cụm lesion."""
    rows = []
    base = load_preds("baseline", split).groupby("image_id")["correct"].mean()
    for name in ["baseline"] + [n for n in names if n != "baseline"]:
        P = load_preds(name, split, available_seeds(name))
        err = (1 - P.groupby("image_id")["correct"].mean()).rename("err")
        D = flags.set_index("image_id").loc[err.index].join(err)
        D["d_err"] = D["err"] - (1 - base.loc[D.index].values)
        for flag in ["F_noise", "F_low_contrast"]:
            # lỗi kỳ vọng nếu nhóm có cùng tỉ lệ lỗi theo lớp như phần còn lại (chuẩn hóa theo thành phần lớp):
            # tách ảnh hưởng của "nhóm khó" khỏi ảnh hưởng của "nhóm nhiều nv / nhiều bcc-akiec"
            rest_c = D[~D[flag]].groupby("dx")["err"].mean()
            exp_err = D[D[flag]]["dx"].map(rest_c).mean()
            for grp, sub in [("nhóm", D[D[flag]]), ("còn lại", D[~D[flag]])]:
                e, de, les = sub["err"].values, sub["d_err"].values, sub["lesion_id"].values
                be, bd = [], []
                for idx in _cluster_boot_idx(les, n_boot):
                    be.append(e[idx].mean()), bd.append(de[idx].mean())
                rows.append(dict(config=name, split=split, flag=flag, group=grp, n_images=len(sub),
                                 n_seeds=P.seed.nunique(), err=e.mean(), err_lo=np.percentile(be, 2.5),
                                 err_hi=np.percentile(be, 97.5),
                                 err_rest_class_matched=exp_err if grp == "nhóm" else np.nan,
                                 d_err_vs_baseline=de.mean() if name != "baseline" else np.nan,
                                 d_lo=np.percentile(bd, 2.5) if name != "baseline" else np.nan,
                                 d_hi=np.percentile(bd, 97.5) if name != "baseline" else np.nan))
    return pd.DataFrame(rows)


def fig_subgroups(T, split="test"):
    setup_style()
    T = T[T.split == split]
    names = list(dict.fromkeys(T.config))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for a, flag, title in zip(ax, ["F_noise", "F_low_contrast"], ["nhóm nhiễu", "nhóm tương phản thấp (mask Otsu thô)"]):
        for k, grp in enumerate(["nhóm", "còn lại"]):
            s = T[(T.flag == flag) & (T.group == grp)].set_index("config").loc[names]
            x = np.arange(len(names)) + (k - 0.5) * 0.35
            a.errorbar(x, s["err"], yerr=[s["err"] - s["err_lo"], s["err_hi"] - s["err"]], fmt="o" if k == 0 else "s",
                       color="#D55E00" if k == 0 else "#7A7A7A", ms=4, capsize=2,
                       label=f"{grp} (n={int(s['n_images'].iloc[0])})")
            if k == 0:
                a.scatter(x, s["err_rest_class_matched"], marker="_", s=120, color="k", zorder=4,
                          label="phần còn lại, chuẩn hóa theo thành phần lớp của nhóm")
        a.set_xticks(range(len(names)), names, rotation=30, ha="right")
        a.set_title(f"{split}: {title}")
        a.legend(fontsize=7)
    ax[0].set_ylabel("tỉ lệ lỗi (TB qua seed), CI95 bootstrap cụm lesion")
    fig.tight_layout()
    savefig(fig, f"fig_p8_subgroup_error_{split}")
    return fig
