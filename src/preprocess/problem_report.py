"""Phân tích dùng chung cho notebook từng vấn đề của 2.5 (nhiễu Gauss, tương phản tổn thương–da thấp).

Khuôn notebook: mô tả → chứng minh → nhóm ảnh gặp phải → tỉ lệ → phương pháp → thực nghiệm → mẫu 100 ảnh.
Notebook chỉ gọi hàm ở đây để hai vấn đề đi đúng một quy trình và một cách tính.

Lượt chạy lại nằm riêng ở outputs/rerun_2.5/ (kể cả baseline và checkpoint) để không đè kết quả cũ và checkpoint của
app.py. Mọi cấu hình đi qua src.training.run_experiment nên chỉ một yếu tố thay đổi: hàm tiền xử lý.
Chọn phương pháp theo val; test chỉ để báo cáo.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

import src.training as T
from src.eda.plots import CLASS_COLORS, CLASSES, setup_style
from src.eda.quality_metrics import lesion_mask, vignette_mask
from src.eda.stats import holm, one_per_lesion, tukey_fence
from src.preprocess import compare as C
from src.preprocess import get_preprocess

from .subgroup import cliffs_delta, mcnemar_exact

RERUN = Path("outputs/rerun_2.5")
FIG = Path("outputs/figures/rerun_2.5")
SEEDS = (42, 43, 44)
SEED = 42
COLORS = ["#7A7A7A", "#0072B2", "#E69F00", "#009E73"]

PROBLEMS = {
    "noise": dict(
        name="Nhiễu Gauss", flag="F_noise", metric="noise_sigma", side="high", unit="σ nhiễu (mức xám)",
        threshold=2.1599, configs=["gauss3", "median3", "median5"], summary="outputs/preprocess/summary_noise.csv",
        image_cols=["sigma_imm", "sigma_imm_224", "lapvar_les_224_ratio", "contrast_ratio"]),
    "contrast": dict(
        name="Tương phản tổn thương–da thấp", flag="F_low_contrast", metric="contrast", side="low",
        unit="tương phản da − tổn thương (mức xám)", threshold=20.0,
        configs=["stretch_L_1_99", "hist_eq_L", "clahe_L_c2_t8"], summary="outputs/preprocess/summary_contrast.csv",
        image_cols=["contrast", "p5_p95", "dE76", "sigma_imm_224_ratio"]),
}
TRAIN_ORDER = ["baseline"] + [c for p in PROBLEMS.values() for c in p["configs"]]


def use_rerun_paths():
    """Trỏ compare.py và checkpoint sang lượt chạy lại; không sửa file gốc của hai module đó."""
    C.BASE_DIR, C.EXP_DIR = RERUN / "baseline", RERUN
    T.CKPT_DIR = RERUN / "checkpoints"


use_rerun_paths()


def train(names, seeds=SEEDS, subset=None, epochs=T.EPOCHS):
    """Train lần lượt (RAM chỉ giữ 1 cache). Seed đã có metrics.json được đọc lại, nên gọi lại không tốn thời gian."""
    out = []
    for n in names:
        fn = None if n == "baseline" else get_preprocess(n)
        out_dir = RERUN / ("_smoke" if subset else "") / n
        d = T.run_experiment(n, preprocess_fn=fn, seeds=seeds, out_dir=out_dir, subset=subset, epochs=epochs,
                             log_file=RERUN / "experiments.log", description=f"2.5 chạy lại: {n}")
        out.append(d[["name", "seed", "best_epoch", "val_macro_f1", "test_macro_f1"]])
    return pd.concat(out, ignore_index=True)


def savefig(fig, name, raster=False):
    FIG.mkdir(parents=True, exist_ok=True)
    if not raster:
        fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)


def read_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------- dữ liệu
def load_table():
    """Một dòng/ảnh: chỉ số EDA, cờ, metadata, nhóm tuổi (toàn bộ 10 015 ảnh)."""
    q = pd.read_csv("outputs/eda/quality_metrics.csv")
    fl = pd.read_csv("outputs/eda/image_flags.csv")
    D = q.merge(fl[["image_id"] + [c for c in fl if c.startswith("F_")]], on="image_id")
    split = pd.concat([pd.read_csv(T.SPLIT_FILES[k])[["image_id"]].assign(split_exp=k) for k in T.SPLIT_FILES])
    D = D.merge(split, on="image_id", how="left")
    D["age_group"] = pd.cut(D["age"], [-1, 29, 44, 59, 74, 200], labels=["<30", "30–44", "45–59", "60–74", "≥75"])
    return D


# ---------------------------------------------------------------- 2. chứng minh vấn đề
def evidence_table(D, key):
    """Thống kê mô tả chỉ số dùng để gắn cờ. Không so nhóm cờ với phần còn lại trên chính chỉ số này vì sẽ vòng tròn."""
    P = PROBLEMS[key]
    m, f = P["metric"], P["flag"]
    x = D[m].dropna()
    return pd.DataFrame([{
        "n ảnh có đo": len(x), "trung vị": x.median(), "Q1": x.quantile(.25), "Q3": x.quantile(.75),
        "min": x.min(), "max": x.max(), "ngưỡng cờ": P["threshold"], "trung vị nhóm cờ": D.loc[D[f], m].median(),
        "trung vị còn lại": D.loc[~D[f], m].median()}]).T.rename(columns={0: key})


def independent_evidence(D, key):
    """Bằng chứng không vòng tròn: so nhóm cờ và phần còn lại trên chỉ số KHÁC chỉ số gắn cờ.
    Một ảnh/lesion; Cliff's δ: |δ| < 0,147 không đáng kể, < 0,33 nhỏ, < 0,474 vừa, còn lại lớn."""
    f = PROBLEMS[key]["flag"]
    L = one_per_lesion(D.assign(p5_p95=D.gray_p95 - D.gray_p5))
    items = {"noise": [("noise_sigma_224", "σ sau resize 224 (cùng bộ đo)")],
             "contrast": [("gray_std", "độ lệch chuẩn mức xám"), ("p5_p95", "khoảng phân vị 5–95"), ("entropy", "entropy")]}[key]
    rows = []
    for c, lab in items:
        a, b = L.loc[L[f], c].dropna().values, L.loc[~L[f], c].dropna().values
        rows.append({"chỉ số": lab, "nguồn": "toàn bộ dữ liệu", "n cờ": len(a), "n còn lại": len(b), "trung vị cờ": np.median(a),
                     "trung vị còn lại": np.median(b), "Cliff δ": cliffs_delta(a, b)})
    if key == "noise":
        M = pd.read_csv("outputs/preprocess/image_metrics_noise.csv")
        M = M[M.method == "none"].drop_duplicates(["group", "lesion_id"])
        for c, lab in [("sigma_imm", "σ Immerkaer 600×450 (bộ đo độc lập)"), ("sigma_imm_224", "σ Immerkaer sau resize 224")]:
            a, b = M.loc[M.group == "problem", c].values, M.loc[M.group == "control", c].values
            rows.append({"chỉ số": lab, "nguồn": "train_fit, cờ vs đối chứng", "n cờ": len(a), "n còn lại": len(b), "trung vị cờ": np.median(a),
                         "trung vị còn lại": np.median(b), "Cliff δ": cliffs_delta(a, b)})
    return pd.DataFrame(rows)


def fig_evidence(D, key):
    setup_style()
    P = PROBLEMS[key]
    m, f, t = P["metric"], P["flag"], P["threshold"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    x = D[m].dropna()
    ax[0].hist(x, bins=80, color="#7A7A7A")
    ax[0].axvline(t, color="#D55E00", ls="--", label=f"ngưỡng cờ = {t:g}")
    ax[0].set(yscale="log", xlabel=P["unit"], ylabel=f"số ảnh, thang log (n = {len(x)})", title="(a) Phân phối toàn bộ ảnh")
    ax[0].legend()
    for sub, c, lab in [(D[D[f]], "#D55E00", "nhóm cờ"), (D[~D[f]], "#7A7A7A", "còn lại")]:
        v = np.sort(sub[m].dropna().values)
        ax[1].plot(v, np.linspace(0, 1, len(v)), color=c, label=f"{lab} (n = {len(v)})")
    ax[1].set(xlabel=P["unit"], ylabel="tỉ lệ tích lũy", title="(b) ECDF: nhóm cờ và phần còn lại")
    ax[1].legend()
    if key == "noise":
        V = pd.read_csv("outputs/eda/noise_detector_validation.csv")
        V = V[V.kind == "gauss"].groupby("level")["value"].agg(["median", "min", "max"]).reset_index()
        ax[2].errorbar(V.level, V["median"], yerr=[V["median"] - V["min"], V["max"] - V["median"]], fmt="o-", capsize=2)
        ax[2].plot(V.level, V.level, "k:", label="y = x")
        ax[2].axhline(t, color="#D55E00", ls="--", label="ngưỡng cờ")
        ax[2].set(xlabel="σ nhiễu Gauss thêm vào", ylabel="σ đo được", title="(c) Kiểm tra bộ đo bằng nhiễu đã biết")
        ax[2].legend()
    else:
        for sub, c, lab in [(D[D[f]].sort_values(m).iloc[len(D[D[f]]) // 2:len(D[D[f]]) // 2 + 1], "#D55E00", "ảnh cờ, trung vị"),
                            (D[~D[f]].sort_values(m).iloc[len(D[~D[f]]) // 2:len(D[~D[f]]) // 2 + 1], "#7A7A7A", "ảnh thường, trung vị")]:
            r = sub.iloc[0]
            g = cv2.cvtColor(read_image(r.path), cv2.COLOR_RGB2GRAY)
            h = np.bincount(g[~vignette_mask(g)], minlength=256)
            ax[2].plot(h / h.sum(), color=c, label=f"{lab}: {r[m]:.1f}")
        ax[2].set(xlabel="mức xám", ylabel="tỉ lệ pixel", title="(c) Lược đồ xám của ảnh ở trung vị mỗi nhóm")
        ax[2].legend()
    fig.tight_layout()
    savefig(fig, f"{key}_1_evidence")
    return fig


def _even_rows(df, m, n):
    """Chọn có hệ thống: n ảnh cách đều theo thứ tự chỉ số, không chọn ảnh đẹp để kể chuyện."""
    df = df.dropna(subset=[m]).sort_values(m).reset_index(drop=True)
    return df.iloc[np.linspace(0, len(df) - 1, n).round().astype(int)]


def _residual(rgb, gain=8):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.clip(128 + gain * (g.astype(np.float32) - cv2.medianBlur(g, 3)), 0, 255).astype(np.uint8)


def fig_examples(D, key, n=6):
    """Hàng trên: ảnh bị cờ; hàng dưới: minh chứng (phần dư nhiễu ×8 hoặc đường bao mask Otsu thô)."""
    P = PROBLEMS[key]
    m, f = P["metric"], P["flag"]
    rows = pd.concat([_even_rows(D[D[f]], m, n).assign(g="cờ"), _even_rows(D[~D[f]], m, n).assign(g="thường")])
    fig, ax = plt.subplots(4, n, figsize=(n * 1.9, 6.0))
    for a in ax.ravel():
        a.axis("off")
    for k, (_, r) in enumerate(rows.iterrows()):
        im = read_image(r.path)
        i, j = (0 if k < n else 2), k % n
        ax[i, j].imshow(im)
        ax[i, j].set_title(f"{r.g} | {r.dx} | {r[m]:.2f}", fontsize=7, color=CLASS_COLORS[r.dx])
        if key == "noise":
            h, w = im.shape[:2]
            ax[i + 1, j].imshow(_residual(im)[h // 2 - 75:h // 2 + 75, w // 2 - 100:w // 2 + 100], cmap="gray", vmin=0, vmax=255)
        else:
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            les, _ = lesion_mask(g, ~vignette_mask(g))
            cs, _ = cv2.findContours(les.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ax[i + 1, j].imshow(cv2.drawContours(im.copy(), cs, -1, (255, 255, 0), 3))
    fig.tight_layout(pad=0.3, h_pad=1.2)
    savefig(fig, f"{key}_2_examples", raster=True)
    return fig


def mask_audit():
    """Kết quả soát mask Otsu trên ảnh tương phản thấp. Do AI đánh giá bằng mắt, người trong nhóm cần xác nhận."""
    A = pd.read_csv("outputs/preprocess/lowcontrast_mask_audit.csv")
    return A["mask_audit"].value_counts().rename_axis("đánh giá mask").reset_index(name="số ảnh").assign(
        **{"tỉ lệ %": lambda d: (100 * d["số ảnh"] / len(A)).round(1)})


# ---------------------------------------------------------------- 3–4. nhóm ảnh và tỉ lệ
def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


GROUP_COLS = {"dx": "Lớp bệnh", "localization": "Vùng cơ thể", "dx_type": "Cách gán nhãn", "age_group": "Nhóm tuổi", "sex": "Giới"}


def group_rates(D, key, min_n=30):
    """Tỉ lệ ảnh bị cờ theo từng nhóm, CI95 Wilson. Nhóm < min_n ảnh và nhóm 'unknown' không vẽ vì CI quá rộng."""
    f = PROBLEMS[key]["flag"]
    rows = []
    for col, lab in GROUP_COLS.items():
        for v, g in D.dropna(subset=[col]).groupby(col, observed=True):
            if str(v).lower() == "unknown" or len(g) < min_n:
                continue
            k, n = int(g[f].sum()), len(g)
            lo, hi = _wilson(k, n)
            rows.append({"yếu tố": lab, "cột": col, "nhóm": str(v), "n ảnh": n, "n cờ": k, "tỉ lệ %": 100 * k / n, "CI thấp %": 100 * lo, "CI cao %": 100 * hi})
    return pd.DataFrame(rows)


def group_tests(D, key):
    """Chi-square + Cramér's V trên MỘT ảnh/lesion (ảnh cùng lesion không độc lập), hiệu chỉnh Holm."""
    f = PROBLEMS[key]["flag"]
    L = one_per_lesion(D)
    rows = []
    for col, lab in GROUP_COLS.items():
        d = L.dropna(subset=[col])
        d = d[d[col].astype(str).str.lower() != "unknown"]
        tab = pd.crosstab(d[col], d[f])
        tab = tab[tab.sum(1) >= 30]
        chi, p, _, _ = stats.chi2_contingency(tab)
        rows.append({"yếu tố": lab, "n lesion": int(tab.values.sum()), "chi2": chi, "p": p, "Cramér V": np.sqrt(chi / (tab.values.sum() * (min(tab.shape) - 1)))})
    R = pd.DataFrame(rows)
    R["p Holm"] = holm(R["p"].values)
    return R


def fig_groups(D, key):
    setup_style()
    R = group_rates(D, key)
    base = 100 * D[PROBLEMS[key]["flag"]].mean()
    cols = list(GROUP_COLS)
    fig, ax = plt.subplots(1, len(cols), figsize=(14, 4.2), sharex=False, gridspec_kw=dict(width_ratios=[7, 12, 4, 5, 2]))
    for a, c in zip(ax, cols):
        S = R[R["cột"] == c]
        if c == "dx":
            S = S.set_index("nhóm").loc[[x for x in CLASSES if x in set(S["nhóm"])]].reset_index()
        x = np.arange(len(S))
        col = [CLASS_COLORS[v] for v in S["nhóm"]] if c == "dx" else ["#0072B2"] * len(S)
        for i, r in enumerate(S.itertuples()):
            a.errorbar(i, S["tỉ lệ %"].iloc[i], yerr=[[S["tỉ lệ %"].iloc[i] - S["CI thấp %"].iloc[i]], [S["CI cao %"].iloc[i] - S["tỉ lệ %"].iloc[i]]], fmt="o", color=col[i], capsize=2, ms=4)
        a.axhline(base, color="#7A7A7A", ls=":", lw=0.9)
        a.set_ylim(bottom=0)
        a.set_xticks(x, S["nhóm"], rotation=60, ha="right")
        a.set_title(GROUP_COLS[c])
    ax[0].set_ylabel("% ảnh bị cờ (CI95 Wilson)\nđường chấm = toàn bộ dữ liệu")
    fig.tight_layout()
    savefig(fig, f"{key}_3_groups")
    return fig


def prevalence(D, key):
    f = PROBLEMS[key]["flag"]
    rows = [{"tập": "toàn bộ", "n ảnh": len(D), "n cờ": int(D[f].sum()), "tỉ lệ %": 100 * D[f].mean(), "n lesion cờ": D.loc[D[f], "lesion_id"].nunique()}]
    for s in ("train", "val", "test"):
        g = D[D.split_exp == s]
        rows.append({"tập": s, "n ảnh": len(g), "n cờ": int(g[f].sum()), "tỉ lệ %": 100 * g[f].mean(), "n lesion cờ": g.loc[g[f], "lesion_id"].nunique()})
    return pd.DataFrame(rows)


def threshold_sensitivity(D, key):
    """Độ nhạy của tỉ lệ theo ngưỡng: kết luận 'tỉ lệ nhỏ' không được phụ thuộc vào một ngưỡng duy nhất."""
    m = PROBLEMS[key]["metric"]
    rows = []
    if key == "noise":
        t_hair = tukey_fence(D["hair_frac"])
        for k in (1.0, 1.5, 2.0, 3.0):
            t = tukey_fence(D[m], k=k)
            mask = (D[m] > t) & (D["hair_frac"] <= t_hair)
            rows.append({"quy tắc": f"Q3 + {k}·IQR", "ngưỡng σ": t, "n cờ": int(mask.sum()), "tỉ lệ %": 100 * mask.mean()})
    else:
        for t in (15, 20, 25, 30):
            mask = D[m] < t
            rows.append({"quy tắc": f"contrast < {t}", "ngưỡng": t, "n cờ": int(mask.sum()), "tỉ lệ %": 100 * mask.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. phương pháp
def fig_before_after(D, key, n=4):
    """Ảnh bị cờ, chọn cách đều theo chỉ số. Với nhiễu, phóng to vùng giữa vì lọc 3×3 khó thấy ở cả ảnh."""
    P = PROBLEMS[key]
    rows = _even_rows(D[(D[P["flag"]]) & (D.split_exp == "train")], P["metric"], n)
    names = ["gốc"] + P["configs"]
    fig, ax = plt.subplots(n, len(names), figsize=(len(names) * 2.5, n * 1.95))
    for i, (_, r) in enumerate(rows.iterrows()):
        im = read_image(r.path)
        for j, nm in enumerate(names):
            o = im if nm == "gốc" else get_preprocess(nm)(im)
            if key == "noise":
                h, w = o.shape[:2]
                o = o[h // 2 - 56:h // 2 + 56, w // 2 - 75:w // 2 + 75]
            ax[i, j].imshow(o)
            ax[i, j].axis("off")
            ax[i, j].set_title(f"{nm} | {r.dx}" + (f" | {r[P['metric']]:.2f}" if j == 0 else ""), fontsize=8)
    fig.tight_layout(pad=0.3)
    savefig(fig, f"{key}_4_before_after", raster=True)
    return fig


def image_effect(key):
    """Đo trên ảnh trước/sau (train_fit, không dùng CNN), từ file của src.preprocess.image_metrics."""
    P = PROBLEMS[key]
    S = pd.read_csv(P["summary"])
    S = S[S.method.isin(["none"] + P["configs"])]
    return S[["group", "method", "n"] + P["image_cols"]].round(3)


# ---------------------------------------------------------------- 6. thực nghiệm
def result_table(key, n_boot=1000):
    """Δ so với baseline, ghép cặp seed, CI95 bootstrap cụm lesion, quy tắc 2.5.0 (src.preprocess.compare)."""
    cfg = PROBLEMS[key]["configs"]
    out = []
    for split in ("val", "test"):
        Q = C.comparison_table(cfg, split, n_boot=n_boot)
        Q.insert(0, "split", split)
        out.append(Q)
    return pd.concat(out, ignore_index=True)


def result_view(R):
    v = R.copy()
    v["macro-F1"] = v.macro_f1_mean.map("{:.4f}".format) + " ± " + v.macro_f1_std.map("{:.4f}".format)
    v["Δ macro-F1"] = v.macro_f1_delta.map(lambda x: "" if pd.isna(x) else f"{x:+.4f}")
    v["CI95 Δ"] = [("" if pd.isna(a) else f"[{a:+.4f}; {b:+.4f}]") for a, b in zip(v.macro_f1_delta_ci_lo, v.macro_f1_delta_ci_hi)]
    v["dấu"] = v.macro_f1_delta_sign
    v["kết luận"] = v.macro_f1_verdict
    v["Δ bal. acc"] = v.balanced_acc_delta.map(lambda x: "" if pd.isna(x) else f"{x:+.4f}")
    return v[["split", "config", "macro-F1", "Δ macro-F1", "CI95 Δ", "dấu", "kết luận", "Δ bal. acc"]].fillna("")


def pick_on_val(R):
    """Chọn cấu hình để phân tích sâu theo val macro-F1 cao nhất; test không tham gia."""
    v = R[(R.split == "val") & (R.config != "baseline")]
    return v.loc[v.macro_f1_mean.idxmax(), "config"]


def fig_results(key):
    setup_style()
    names = ["baseline"] + PROBLEMS[key]["configs"]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
    for a, (split, m, t) in zip(ax, [("val", "macro_f1", "val macro-F1"), ("test", "macro_f1", "test macro-F1"),
                                     ("val", "balanced_acc", "val balanced acc"), ("test", "balanced_acc", "test balanced acc")]):
        vals = []
        for i, n in enumerate(names):
            v = C.load_summary(n)[f"{split}_{m}"].values
            vals.append(v)
            a.bar(i, v.mean(), color=COLORS[i], alpha=.6, width=.7, yerr=v.std(ddof=1), capsize=3)
            a.scatter(i + np.array([-.15, 0, .15]), v, s=12, color="k", zorder=3)
        a.axhline(vals[0].mean(), color="#7A7A7A", ls=":", lw=.8)
        a.set_ylim(np.concatenate(vals).min() - .03, np.concatenate(vals).max() + .02)
        a.set_xticks(range(len(names)), names, rotation=30, ha="right")
        a.set_title(t)
    ax[0].set_ylabel("cột = mean, thanh = ±1 std, chấm = seed 42/43/44")
    fig.tight_layout()
    savefig(fig, f"{key}_5_results")
    return fig


def per_class(key, split="test"):
    names = ["baseline"] + PROBLEMS[key]["configs"]
    rows = []
    for n in names:
        S = C.load_summary(n)
        for c in CLASSES:
            rows.append({"config": n, "lớp": c, "F1": S[f"{split}_f1_{c}"].mean(), "recall": S[f"{split}_recall_{c}"].mean()})
    Q = pd.DataFrame(rows).pivot(index="lớp", columns="config", values=["F1", "recall"]).round(3)
    return Q.loc[CLASSES, [(a, n) for a in ("F1", "recall") for n in names]]


def _preds(name, split):
    cols = ["image_id", "lesion_id", "dx", "label", "pred", "correct"] + [f"prob_{c}" for c in CLASSES]
    return {s: pd.read_csv(C.run_dir(name) / f"seed{s}" / f"{split}_preds.csv", usecols=cols).set_index("image_id") for s in SEEDS}


def target_group_effect(D, key, n_boot=1000, seed=SEED):
    """Tác động riêng lên nhóm cờ: DiD = Δ lỗi(nhóm cờ) − Δ lỗi(phần còn lại); CI bootstrap rút lại lesion toàn split."""
    P = PROBLEMS[key]
    fl = D.set_index("image_id")[P["flag"]]
    rows = []
    for split in ("val", "test"):
        B = _preds("baseline", split)
        ids = B[42].index
        be = 1 - np.mean([B[s].loc[ids, "correct"].astype(float).values for s in SEEDS], axis=0)
        mask = fl.loc[ids].values.astype(bool)
        les = B[42].loc[ids, "lesion_id"].values
        uniq, inv = np.unique(les, return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        for cfg in P["configs"]:
            A = _preds(cfg, split)
            d = 1 - np.mean([A[s].loc[ids, "correct"].astype(float).values for s in SEEDS], axis=0) - be
            rng = np.random.default_rng(seed)
            bs = []
            for _ in range(n_boot):
                idx = np.concatenate([groups[k] for k in rng.integers(0, len(uniq), len(uniq))])
                m = mask[idx]
                bs.append(d[idx][m].mean() - d[idx][~m].mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"split": split, "config": cfg, "n nhóm cờ": int(mask.sum()), "lỗi baseline nhóm cờ": be[mask].mean(),
                         "Δ lỗi nhóm cờ": d[mask].mean(), "Δ lỗi còn lại": d[~mask].mean(), "DiD": d[mask].mean() - d[~mask].mean(),
                         "CI thấp": lo, "CI cao": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7. mẫu 100 ảnh
def sample_100(D, key, n=100, seed=SEED):
    """Rút ngẫu nhiên n ảnh bị cờ ở val + test, một ảnh/lesion. Dùng cả val vì test có thể không đủ 100 ảnh cờ.
    Việc rút chỉ dựa vào cờ EDA, không nhìn dự đoán."""
    f = PROBLEMS[key]["flag"]
    g = D[D[f] & D.split_exp.isin(["val", "test"])]
    g = g.sample(frac=1, random_state=seed).drop_duplicates("lesion_id").sort_values("image_id")
    if len(g) < n:
        raise ValueError(f"chỉ có {len(g)} lesion bị cờ ở val + test, cần {n}")
    return g.sample(n=n, random_state=seed).sort_values("image_id").reset_index(drop=True)


def sample_detail(S, key):
    """Bảng từng ảnh: dự đoán và xác suất lớp đúng ở từng seed, của baseline và từng phương pháp. Lưu CSV."""
    P_ = {sp: {n: _preds(n, sp) for n in ["baseline"] + PROBLEMS[key]["configs"]} for sp in ("val", "test")}
    rows = []
    for r in S.itertuples():
        rec = dict(image_id=r.image_id, split=r.split_exp, dx=r.dx, path=r.path, metric=getattr(r, PROBLEMS[key]["metric"]))
        for n, by_seed in P_[r.split_exp].items():
            for s in SEEDS:
                x = by_seed[s].loc[r.image_id]
                rec[f"{n}|pred_{s}"] = CLASSES[int(x.pred)]
                rec[f"{n}|ptrue_{s}"] = float(x[f"prob_{r.dx}"])
            rec[f"{n}|n_correct"] = sum(rec[f"{n}|pred_{s}"] == r.dx for s in SEEDS)
        rows.append(rec)
    SD = pd.DataFrame(rows)
    RERUN.mkdir(parents=True, exist_ok=True)
    SD.to_csv(RERUN / f"{key}_sample100.csv", index=False)
    return SD


def _pair(cb, cm):
    wc, cw = int((~cb & cm).sum()), int((cb & ~cm).sum())
    return {"đúng trước": int(cb.sum()), "đúng sau": int(cm.sum()), "sai→đúng": wc, "đúng→sai": cw, "ròng": wc - cw, "McNemar p": mcnemar_exact(wc, cw)}


def sample_summary(SD, key):
    """Đúng/sai trước → sau theo từng seed và đa số 3 seed. Dòng 'baseline seed A→B' là mức dao động khi không xử lý gì."""
    rows = []
    for cfg in PROBLEMS[key]["configs"]:
        for s in SEEDS:
            rows.append({"so sánh": f"{cfg} | seed {s}", **_pair((SD[f"baseline|pred_{s}"] == SD.dx).values, (SD[f"{cfg}|pred_{s}"] == SD.dx).values)})
        rows.append({"so sánh": f"{cfg} | đa số 3 seed", **_pair((SD["baseline|n_correct"] >= 2).values, (SD[f"{cfg}|n_correct"] >= 2).values)})
    for a, b in ((42, 43), (43, 44), (44, 42)):
        rows.append({"so sánh": f"baseline seed {a}→{b}", **_pair((SD[f"baseline|pred_{a}"] == SD.dx).values, (SD[f"baseline|pred_{b}"] == SD.dx).values)})
    return pd.DataFrame(rows)


def prob_change(SD, key):
    """Δ xác suất của lớp đúng (TB 3 seed), kiểm định dấu hạng Wilcoxon ghép cặp theo ảnh, hiệu chỉnh Holm."""
    b = SD[[f"baseline|ptrue_{s}" for s in SEEDS]].mean(axis=1).values
    rows = []
    for cfg in PROBLEMS[key]["configs"]:
        d = SD[[f"{cfg}|ptrue_{s}" for s in SEEDS]].mean(axis=1).values - b
        rows.append({"config": cfg, "Δ prob TB": d.mean(), "Δ prob trung vị": np.median(d), "số ảnh tăng": int((d > 0).sum()), "số ảnh giảm": int((d < 0).sum()), "Wilcoxon p": stats.wilcoxon(d).pvalue})
    R = pd.DataFrame(rows)
    R["p Holm"] = holm(R["Wilcoxon p"].values)
    return R


def sample_by_class(SD, cfg):
    g = SD.assign(trước=SD["baseline|n_correct"] >= 2, sau=SD[f"{cfg}|n_correct"] >= 2).groupby("dx")
    return pd.DataFrame({"n": g.size(), "đúng trước (đa số)": g["trước"].sum(), "đúng sau (đa số)": g["sau"].sum()}).reindex([c for c in CLASSES if c in set(SD.dx)])


def fig_sample(SD, key, cfg):
    setup_style()
    T_ = sample_summary(SD, key)
    T_ = T_[T_["so sánh"].str.startswith(cfg) | T_["so sánh"].str.startswith("baseline")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    x = np.arange(len(T_))
    ax[0].bar(x - .2, T_["sai→đúng"], .4, color="#009E73", label="sai → đúng")
    ax[0].bar(x + .2, T_["đúng→sai"], .4, color="#D55E00", label="đúng → sai")
    ax[0].set_xticks(x, [s.replace(cfg + " | ", "") for s in T_["so sánh"]], rotation=35, ha="right")
    ax[0].set(ylabel="số ảnh / 100", title=f"(a) Đổi đúng/sai: {cfg} so với baseline; ba cột cuối là baseline khác seed")
    ax[0].legend()
    b = SD[[f"baseline|ptrue_{s}" for s in SEEDS]].mean(axis=1).values
    a = SD[[f"{cfg}|ptrue_{s}" for s in SEEDS]].mean(axis=1).values
    for c in sorted(set(SD.dx), key=CLASSES.index):
        k = SD.dx.values == c
        ax[1].scatter(b[k], a[k], s=16, color=CLASS_COLORS[c], label=f"{c} ({k.sum()})")
    ax[1].plot([0, 1], [0, 1], "k:", lw=.8)
    ax[1].set(xlabel="P(lớp đúng), baseline (TB 3 seed)", ylabel=f"P(lớp đúng), {cfg}", title="(b) Xác suất lớp đúng trước và sau")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, f"{key}_6_sample100")
    return fig


def _major(row, cfg):
    v = [row[f"{cfg}|pred_{s}"] for s in SEEDS]
    u, c = np.unique(v, return_counts=True)
    return u[c.argmax()] if c.max() >= 2 else "không đồng thuận"


def fig_sample_grid(SD, key, cfg, k=4):
    """Ảnh đổi đúng/sai theo đa số 3 seed: tối đa k ảnh sai→đúng và k ảnh đúng→sai, lấy theo thứ tự image_id."""
    cb, cm = SD["baseline|n_correct"] >= 2, SD[f"{cfg}|n_correct"] >= 2
    rows = pd.concat([SD[~cb & cm].head(k).assign(kiểu="sai→đúng"), SD[cb & ~cm].head(k).assign(kiểu="đúng→sai")])
    if rows.empty:
        print("Không có ảnh nào đổi đúng/sai theo đa số 3 seed.")
        return None
    fig, ax = plt.subplots(len(rows), 2, figsize=(6.2, len(rows) * 2.05), squeeze=False)
    fn = get_preprocess(cfg)
    for i, (_, r) in enumerate(rows.iterrows()):
        im = read_image(r.path)
        for j, (x, cap) in enumerate([(im, f"gốc → {_major(r, 'baseline')}"), (fn(im), f"{cfg} → {_major(r, cfg)}")]):
            ax[i, j].imshow(x)
            ax[i, j].axis("off")
            ax[i, j].set_title(f"{r['kiểu']} | đúng: {r.dx} | {cap}", fontsize=7.5, color=CLASS_COLORS[r.dx])
    fig.tight_layout(pad=0.3)
    savefig(fig, f"{key}_7_sample_grid", raster=True)
    return fig
