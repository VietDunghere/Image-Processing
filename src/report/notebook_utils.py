"""Hàm dùng chung cho 4 notebook "một vấn đề – một notebook" (2.5.3_long … 2.5.6_vien_toi).

Mỗi notebook: áp dụng xử lý ảnh → train ResNet-18 3 seed → so sánh với baseline cùng lần chạy → giải thích.
Mọi kết quả của lần chạy này nằm trong RUN_ROOT (outputs/runs_cl), tách khỏi kết quả cũ (outputs/baseline,
outputs/experiments). Cấu hình train giữ nguyên siêu tham số đóng băng của 2.4 (batch 32, AdamW lr 1e-4, 20 epoch);
chỉ bật các tùy chọn tốc độ không đổi siêu tham số: channels_last, batch đánh giá 512, 32 luồng tạo cache.
"""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.eda.plots import setup_style
from src.eda.quality_metrics import compute_one
from src.preprocess import get_preprocess
from src.training import CLASSES, read_rgb, run_experiment

RUN_ROOT = Path("outputs/runs_cl")
FIG_DIR = RUN_ROOT / "figures"
SEEDS = (42, 43, 44)
SPEED = dict(channels_last=True, eval_batch_size=512, n_jobs=32)
COLORS = {"baseline": "#7A7A7A"}
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

# 4 vấn đề: cờ EDA, chỉ số EDA, ngưỡng cờ, 2 cấu hình (đại trà / có điều kiện hoặc 2 cách sửa)
PROBLEMS = {
    "long": dict(name="Lông che phủ", flag="F_hair", metric="hair_frac", thr=0.0582,
                 configs=["hair_bh17", "hair_bh17_cond"]),
    "sang_khong_deu": dict(name="Sáng không đều", flag="F_uneven_illum", metric="illum_range", thr=38.0,
                           configs=["flat_L_s60", "flat_L_s60_cond"]),
    "qua_sang": dict(name="Quá sáng (bạc màu)", flag="F_overexposed", metric="bright_frac", thr=0.10,
                     configs=["gamma_L_1.5", "gamma_L_auto"]),
    "vien_toi": dict(name="Viền tối do ống soi", flag="F_vignette", metric="vignette_frac", thr=0.01,
                     configs=["vig_fill", "vig_crop"]),
}


def color(name, names):
    if name in COLORS:
        return COLORS[name]
    return PALETTE[[n for n in names if n != "baseline"].index(name) % len(PALETTE)]


def savefig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=200)


# ------------------------------------------------------------------ train (có resume theo seed)
def train(name: str, seeds=SEEDS) -> pd.DataFrame:
    """Train 1 cấu hình × 3 seed vào RUN_ROOT/<name>. Seed đã có metrics.json thì đọc lại, không train lại."""
    fn = None if name == "baseline" else get_preprocess(name)
    return run_experiment(name, preprocess_fn=fn, seeds=seeds, out_root=RUN_ROOT,
                          description=f"notebook 1 vấn đề – {name}", **SPEED)


# ------------------------------------------------------------------ đọc kết quả
def summary(name) -> pd.DataFrame:
    return pd.read_csv(RUN_ROOT / name / "summary.csv")


def preds(name, split) -> dict:
    """{seed: DataFrame dự đoán (index image_id)}."""
    return {s: pd.read_csv(RUN_ROOT / name / f"seed{s}" / f"{split}_preds.csv").set_index("image_id").sort_index()
            for s in SEEDS}


def _boot_idx(lesions, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(lesions, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    return [np.concatenate([groups[k] for k in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]


METRIC_FN = {"macro_f1": lambda y, p: f1_score(y, p, labels=range(7), average="macro", zero_division=0),
             "balanced_acc": balanced_accuracy_score}
METRIC_VI = {"macro_f1": "macro-F1", "balanced_acc": "balanced accuracy"}


def comparison(names, split="test", n_boot=1000) -> pd.DataFrame:
    """mean ± std 3 seed; Δ ghép cặp seed so với baseline; CI95 bootstrap cụm lesion; dấu theo seed; kết luận.
    Quy tắc kết luận (như 2.5.0): "khác biệt rõ" khi |Δ| > std baseline VÀ CI95 không chứa 0 VÀ 3/3 seed cùng dấu."""
    B = preds("baseline", split)
    ids = B[42].index
    y = B[42]["label"].to_numpy()
    les = B[42]["lesion_id"].to_numpy()
    boots = _boot_idx(les, n_boot)
    pb = np.stack([B[s].loc[ids, "pred"].to_numpy() for s in SEEDS])
    rows = []
    for name in names:
        P = preds(name, split)
        pa = np.stack([P[s].loc[ids, "pred"].to_numpy() for s in SEEDS])
        rec = {"cấu hình": name}
        for m, fn in METRIC_FN.items():
            va = np.array([fn(y, pa[i]) for i in range(3)])
            vb = np.array([fn(y, pb[i]) for i in range(3)])
            rec[f"{m}_mean"], rec[f"{m}_std"] = va.mean(), va.std(ddof=1)
            rec[f"{m}_per_seed"] = "; ".join(f"{v:.4f}" for v in va)
            if name == "baseline":
                continue
            d = va - vb
            bt = [np.mean([fn(y[i], pa[k][i]) - fn(y[i], pb[k][i]) for k in range(3)]) for i in boots]
            lo, hi = np.percentile(bt, [2.5, 97.5])
            sign = f"{(d > 0).sum()}+/{(d < 0).sum()}-"
            clear = abs(d.mean()) > vb.std(ddof=1) and (lo > 0 or hi < 0) and sign in ("3+/0-", "0+/3-")
            rec.update({f"{m}_delta": d.mean(), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi, f"{m}_sign": sign,
                        f"{m}_delta_per_seed": "; ".join(f"{v:+.4f}" for v in d),
                        f"{m}_verdict": ("CẢI THIỆN rõ" if d.mean() > 0 else "TỆ ĐI rõ") if clear
                        else "không khác biệt rõ"})
        rows.append(rec)
    T = pd.DataFrame(rows)
    T.to_csv(RUN_ROOT / f"comparison_{'_'.join(n for n in names if n != 'baseline')}_{split}.csv", index=False)
    return T


def per_class(names, split="test") -> pd.DataFrame:
    """Recall từng lớp mean ± std 3 seed và Δ so với baseline (dấu theo seed)."""
    rows = []
    base = summary("baseline")
    for c in CLASSES:
        rec = {"lớp": c}
        for name in names:
            S = summary(name)
            v = S[f"{split}_recall_{c}"].to_numpy()
            txt = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
            if name != "baseline":
                d = v - base[f"{split}_recall_{c}"].to_numpy()
                txt += f" ({d.mean():+.3f}; {(d > 0).sum()}+/{(d < 0).sum()}-)"
            rec[name] = txt
        rows.append(rec)
    return pd.DataFrame(rows)


def subgroup(names, flag, split="test", n_boot=1000) -> pd.DataFrame:
    """Δ accuracy trên CHÍNH nhóm ảnh bị cờ EDA và trên phần còn lại; DiD; số ảnh sửa/hỏng từng seed."""
    fl = pd.read_csv("outputs/eda/image_flags.csv").set_index("image_id")
    B = preds("baseline", split)
    ids = B[42].index
    g = fl.loc[ids, flag].to_numpy(bool)
    les = B[42]["lesion_id"].to_numpy()
    cb = np.stack([B[s].loc[ids, "correct"].to_numpy() for s in SEEDS]).astype(float)
    boots = _boot_idx(les, n_boot)
    rows = []
    for name in names:
        if name == "baseline":
            continue
        P = preds(name, split)
        cc = np.stack([P[s].loc[ids, "correct"].to_numpy() for s in SEEDS]).astype(float)
        d = cc - cb
        dg, dr = d[:, g].mean(1), d[:, ~g].mean(1)
        bt = np.array([(d[:, i][:, g[i]].mean(), d[:, i][:, ~g[i]].mean()) for i in boots if g[i].any()])
        did = bt[:, 0] - bt[:, 1]
        wc = "; ".join(str(int(((cb[k] == 0) & (cc[k] == 1) & g).sum())) for k in range(3))
        cw = "; ".join(str(int(((cb[k] == 1) & (cc[k] == 0) & g).sum())) for k in range(3))
        rows.append({"cấu hình": name, "split": split, "n nhóm": int(g.sum()),
                     "acc nhóm baseline": cb[:, g].mean(), "acc nhóm sau xử lý": cc[:, g].mean(),
                     "Δ acc nhóm": dg.mean(), "CI95 Δ nhóm": f"[{np.percentile(bt[:, 0], 2.5):+.3f}; "
                                                           f"{np.percentile(bt[:, 0], 97.5):+.3f}]",
                     "dấu": f"{(dg > 0).sum()}+/{(dg < 0).sum()}-",
                     "Δ acc còn lại": dr.mean(),
                     "DiD": (dg - dr).mean(), "CI95 DiD": f"[{np.percentile(did, 2.5):+.3f}; "
                                                          f"{np.percentile(did, 97.5):+.3f}]",
                     "sửa (sai→đúng) từng seed": wc, "hỏng (đúng→sai) từng seed": cw})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ hình
def fig_seed_bars(names, split="test", metric="macro_f1", stem=None):
    """Biểu đồ cột nhóm: trục x = seed 42 / 43 / 44 và 'TB 3 seed'; mỗi nhóm có 1 cột cho mỗi cấu hình.
    Cột 'TB 3 seed' có thanh sai số = ±1 std. Đường chấm = mean baseline."""
    setup_style()
    vals = {n: summary(n)[f"{split}_{metric}"].to_numpy() for n in names}
    groups = [f"seed {s}" for s in SEEDS] + ["TB 3 seed"]
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, n in enumerate(names):
        v = vals[n]
        h = list(v) + [v.mean()]
        x = np.arange(len(groups)) + (i - (len(names) - 1) / 2) * w
        ax.bar(x, h, w * 0.95, color=color(n, names), label=n, alpha=0.9)
        sd = v.std(ddof=1)
        ax.errorbar(x[-1], h[-1], yerr=sd, color="k", capsize=3, lw=1)     # chỉ cột TB có thanh ±1 std
        for j, (xi, hi) in enumerate(zip(x, h)):
            top = hi + (sd if j == len(h) - 1 else 0)
            ax.text(xi, top + 0.002, f"{hi:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.axhline(vals["baseline"].mean(), color="#7A7A7A", ls=":", lw=1)
    lo = min(v.min() for v in vals.values())
    hi = max(v.max() for v in vals.values())
    ax.set_ylim(lo - 0.04, hi + 0.04)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(f"{METRIC_VI[metric]} ({split})")
    ax.set_title(f"{METRIC_VI[metric]} trên {split} theo từng lần chạy (seed); cột TB có thanh ±1 std", fontsize=9)
    ax.legend(fontsize=7, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    if stem:
        savefig(fig, stem)
    return fig


def fig_subgroup_seed_bars(names, flag, split="test", stem=None):
    """Biểu đồ cột nhóm cho ACCURACY TRÊN TẬP CON ẢNH CÓ VẤN ĐỀ (cờ EDA): trục x = seed 42 / 43 / 44 và 'TB 3 seed',
    mỗi nhóm 1 cột cho mỗi cấu hình; cột TB có thanh ±1 std; đường chấm = accuracy baseline trên tập con."""
    setup_style()
    fl = pd.read_csv("outputs/eda/image_flags.csv").set_index("image_id")
    vals = {}
    for n in names:
        P = preds(n, split)
        g = fl.loc[P[42].index, flag].to_numpy(bool)
        vals[n] = np.array([P[s].loc[P[42].index, "correct"].to_numpy()[g].mean() for s in SEEDS])
    groups = [f"seed {s}" for s in SEEDS] + ["TB 3 seed"]
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, n in enumerate(names):
        v = vals[n]
        h = list(v) + [v.mean()]
        x = np.arange(len(groups)) + (i - (len(names) - 1) / 2) * w
        ax.bar(x, h, w * 0.95, color=color(n, names), label=n, alpha=0.9)
        sd = v.std(ddof=1)
        ax.errorbar(x[-1], h[-1], yerr=sd, color="k", capsize=3, lw=1)
        for j, (xi, hi) in enumerate(zip(x, h)):
            top = hi + (sd if j == len(h) - 1 else 0)
            ax.text(xi, top + 0.003, f"{hi:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.axhline(vals["baseline"].mean(), color="#7A7A7A", ls=":", lw=1)
    lo = min(v.min() for v in vals.values())
    hi = max(v.max() for v in vals.values())
    ax.set_ylim(max(0, lo - 0.08), min(1.0, hi + 0.08))
    ax.set_xticks(range(len(groups)), groups)
    n_img = int(fl.loc[preds(names[0], split)[42].index, flag].sum())
    ax.set_ylabel(f"accuracy trên tập con ({split})")
    ax.set_title(f"Accuracy trên tập con ảnh có vấn đề ({flag}, n = {n_img} ảnh {split}) theo từng lần chạy; "
                 f"cột TB có thanh ±1 std", fontsize=8.5)
    ax.legend(fontsize=7, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    if stem:
        savefig(fig, stem)
    return fig


def fig_before_after(problem, n_examples=3, stem=None):
    """Ảnh trước/sau: trong nhóm cờ ∩ train_fit, ảnh có chỉ số gần trung vị, gần phân vị 90 và lớn nhất (quy tắc cố định)."""
    P = PROBLEMS[problem]
    q = pd.read_csv("outputs/eda/quality_metrics.csv")
    fl = pd.read_csv("outputs/eda/image_flags.csv")[["image_id", P["flag"]]]
    q = q.merge(fl, on="image_id")
    q = q[q[P["flag"]] & q.image_id.isin(pd.read_csv("outputs/splits/train_fit.csv").image_id)]
    m = q[P["metric"]]
    picks = [q.iloc[(m - m.quantile(t)).abs().argsort().iloc[0]] for t in (0.5, 0.9)] + [q.loc[m.idxmax()]]
    picks = picks[:n_examples]
    cols = ["gốc"] + P["configs"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.6 * len(cols), 2.05 * len(picks)), squeeze=False)
    for r, row in zip(axes, picks):
        im = read_rgb(row.path)
        for ax, c in zip(r, cols):
            out = im if c == "gốc" else get_preprocess(c)(im)
            ax.imshow(out)
            t = f"{row.image_id[5:]} ({row.dx}), {P['metric']}={row[P['metric']]:.3g}" if c == "gốc" else c
            if c != "gốc" and np.array_equal(out, im):
                t += " (không đổi)"
            ax.set_title(t, fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    if stem:
        savefig(fig, stem)
    return fig


# ------------------------------------------------------------------ hiệu ứng trên ảnh (không CNN)
def _metric_after(rgb, metric):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        rec, _ = compute_one(f.name)
    return rec[metric]


def image_effect(problem, n=100) -> pd.DataFrame:
    """Chỉ số EDA trước → sau và ΔE76 trên n ảnh cờ ∩ train_fit và n ảnh đối chứng (rút ngẫu nhiên seed 42)."""
    P = PROBLEMS[problem]
    q = pd.read_csv("outputs/eda/quality_metrics.csv")
    q = q.merge(pd.read_csv("outputs/eda/image_flags.csv")[["image_id", P["flag"]]], on="image_id")
    q = q[q.image_id.isin(pd.read_csv("outputs/splits/train_fit.csv").image_id)]
    rows = []
    for grp, D in [("nhóm cờ", q[q[P["flag"]]]), ("đối chứng", q[~q[P["flag"]]])]:
        D = D.sample(min(n, len(D)), random_state=42)

        def work(r):
            im = read_rgb(r.path)
            lab0 = cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
            out = []
            for c in P["configs"]:
                o = get_preprocess(c)(im)
                same = np.array_equal(o, im)
                de = 0.0 if same else float(np.linalg.norm(
                    cv2.cvtColor(o.astype(np.float32) / 255, cv2.COLOR_RGB2Lab) - lab0, axis=2).mean())
                out.append(dict(nhóm=grp, cấu_hình=c, trước=r[P["metric"]],
                                sau=r[P["metric"]] if same else _metric_after(o, P["metric"]), đổi=not same, dE76=de))
            return out
        with ThreadPoolExecutor(16) as ex:
            for res in ex.map(work, [r for _, r in D.iterrows()]):
                rows += res
    R = pd.DataFrame(rows)
    R["vượt ngưỡng trước"] = R["trước"] > P["thr"]
    R["vượt ngưỡng sau"] = R["sau"] > P["thr"]
    S = R.groupby(["cấu_hình", "nhóm"]).agg(n=("trước", "size"), **{
        f"{P['metric']} trước (trung vị)": ("trước", "median"), f"{P['metric']} sau (trung vị)": ("sau", "median"),
        "% còn vượt ngưỡng": ("vượt ngưỡng sau", "mean"), "% ảnh bị đổi": ("đổi", "mean"),
        "ΔE76 TB": ("dE76", "mean")}).reset_index()
    S["% còn vượt ngưỡng"] *= 100
    S["% ảnh bị đổi"] *= 100
    return S.round(3)


# ------------------------------------------------------------------ câu kết luận tự động từ số liệu
def verdict_text(T, name, split="test") -> str:
    r = T.set_index("cấu hình").loc[name]
    return (f"`{name}` ({split}): macro-F1 {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}, Δ = {r.macro_f1_delta:+.4f} "
            f"(CI95 {r.macro_f1_ci_lo:+.4f}; {r.macro_f1_ci_hi:+.4f}), dấu theo seed {r.macro_f1_sign} "
            f"→ **{r.macro_f1_verdict}**. Balanced accuracy Δ = {r.balanced_acc_delta:+.4f}, {r.balanced_acc_sign} "
            f"→ {r.balanced_acc_verdict}.")


def run_info() -> dict:
    """Cấu hình thực tế của lần chạy (đọc từ config.json của baseline) để in vào notebook."""
    c = json.loads((RUN_ROOT / "baseline" / "config.json").read_text(encoding="utf-8"))
    return {k: c.get(k) for k in ("model", "batch_size", "lr", "epochs", "seeds", "env")} | \
        {"channels_last": c["speed"].get("channels_last_enabled"), "eval_batch": c["speed"].get("eval_batch_size_used")}
