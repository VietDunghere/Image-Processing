"""Hình cho giai đoạn 1 của 2.5 (khử nhiễu, tăng tương phản): xuất PDF vector + PNG 300 dpi vào
outputs/figures/preprocess/. Dùng chung style/màu của EDA (src.eda.plots)."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda.plots import CLASS_COLORS, setup_style

from . import CONTRAST_CANDIDATES, NOISE_CANDIDATES
from .image_metrics import gray_of, load_rgb, reference

FIG_DIR = Path("outputs/figures/preprocess")
GROUP_COLORS = {"problem": "#D55E00", "control": "#7A7A7A"}   # cùng quy ước FLAG/OK của EDA
GROUP_LABEL = {"problem": "nhóm vấn đề", "control": "đối chứng"}


def savefig(fig, name, raster=False):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not raster:
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    return FIG_DIR / f"{name}.png"


def boot_ci_median(x, n=2000, seed=42):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n, len(x))), axis=1)
    return np.median(x), *np.percentile(b, [2.5, 97.5])


def summarize(df, cols, ref="none"):
    """Trung vị + CI bootstrap 95% theo (group, method); thêm tỉ lệ so với 'none' TÍNH THEO TỪNG ẢNH."""
    base = df[df["method"] == ref].set_index("image_id")
    rows = []
    for (g, m), d in df.groupby(["group", "method"], sort=False):
        rec = {"group": g, "method": m, "n": len(d)}
        for c in cols:
            med, lo, hi = boot_ci_median(d[c])
            rec.update({c: med, f"{c}_lo": lo, f"{c}_hi": hi})
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (d.set_index("image_id")[c] / base.loc[d["image_id"], c].values).replace([np.inf, -np.inf], np.nan)
            rec[f"{c}_ratio"] = float(r.median()) if r.notna().any() else np.nan   # vd. Δab của 'none' = 0/0
        rows.append(rec)
    return pd.DataFrame(rows)


def _dotci(ax, S, col, methods, title, xlabel, logx=False):
    y = np.arange(len(methods))
    for k, (g, off) in enumerate([("problem", -0.15), ("control", 0.15)]):
        s = S[S["group"] == g].set_index("method").loc[methods]
        ax.errorbar(s[col], y + off, xerr=[s[col] - s[f"{col}_lo"], s[f"{col}_hi"] - s[col]], fmt="o", ms=4,
                    color=GROUP_COLORS[g], label=f"{GROUP_LABEL[g]} (n={int(s['n'].iloc[0])})", capsize=2)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")


def fig_noise_metrics(df, wn):
    """(a) σ nhiễu ở 600×450 và 224; (b) đánh đổi giảm nhiễu – mất chi tiết ở 224; (c) nhiễu trắng tổng hợp."""
    setup_style()
    methods = list(NOISE_CANDIDATES)
    S = summarize(df, ["sigma_imm", "sigma_imm_224", "lapvar_les_224", "lapvar_224"])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    _dotci(ax[0], S, "sigma_imm", methods, "(a) σ nhiễu ở 600×450", "σ Immerkaer (mức xám), trung vị ± CI95")
    for g in ("problem", "control"):
        s = S[S["group"] == g].set_index("method").loc[methods]
        ax[0].scatter(s["sigma_imm_224"], np.arange(len(methods)) + (-0.15 if g == "problem" else 0.15),
                      marker="x", s=22, color=GROUP_COLORS[g])
    ax[0].plot([], [], "kx", label="cùng ảnh sau resize 224")
    ax[0].legend(loc="lower right", fontsize=6.5)
    # (b) đánh đổi
    for g in ("problem", "control"):
        s = S[S["group"] == g].set_index("method").loc[methods]
        ax[1].plot(s["sigma_imm_224_ratio"] * 100, s["lapvar_les_224_ratio"] * 100, "o-", color=GROUP_COLORS[g],
                   ms=4, lw=0.8, label=GROUP_LABEL[g])
        for m, r in s.iterrows():
            if g == "problem":
                ax[1].annotate(m, (r["sigma_imm_224_ratio"] * 100, r["lapvar_les_224_ratio"] * 100), fontsize=6.5,
                               xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("σ nhiễu ở 224 còn lại (% so với không lọc)")
    ax[1].set_ylabel("phương sai Laplacian vùng tổn thương ở 224\ncòn lại (%)")
    ax[1].set_title("(b) giảm nhiễu đổi lấy mất chi tiết (ở 224)")
    ax[1].legend(fontsize=6.5)
    # (c) nhiễu trắng tổng hợp
    x = np.arange(len(wn))
    ax[2].bar(x - 0.2, wn["ratio_native"] * 100, 0.4, color="#56B4E9", label="sau lọc (600×450)")
    ax[2].bar(x + 0.2, wn["ratio_224"] * 100, 0.4, color="#0072B2", label="sau lọc + resize 224")
    ax[2].set_xticks(x, wn["method"], rotation=30, ha="right")
    ax[2].set_ylabel("σ còn lại (% so với ảnh nhiễu gốc)")
    ax[2].set_title("(c) nhiễu trắng tổng hợp σ=10/kênh, 20 ảnh")
    ax[2].legend(fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "fig_p1_noise_metrics")
    return fig, S


def fig_contrast_metrics(df):
    """(a) tương phản da − tổn thương; (b) σ nhiễu ở 224 (khuếch đại); (c) Δab màu."""
    setup_style()
    methods = list(CONTRAST_CANDIDATES)
    S = summarize(df, ["contrast", "sigma_imm_224", "dab", "lapvar_les_224", "entropy"])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    _dotci(ax[0], S, "contrast", methods, "(a) tương phản da − tổn thương", "mức xám (600×450)")
    ax[0].axvline(20, color="k", ls=":", lw=0.8)
    ax[0].text(20, 0.01, " ngưỡng EDA = 20", fontsize=6.5, transform=ax[0].get_xaxis_transform())
    ax[0].legend(loc="lower right", fontsize=6.5)
    _dotci(ax[1], S, "sigma_imm_224", methods, "(b) nhiễu sau xử lý (ở 224)", "σ Immerkaer (mức xám)")
    _dotci(ax[2], S, "dab", methods, "(c) lệch màu Δab (bỏ L*)", "Δab trung bình trong FOV (CIE76)")
    ax[2].axvline(2.3, color="k", ls=":", lw=0.8)
    ax[2].text(2.3, 0.01, " ΔE≈2.3", fontsize=6.5, ha="right", transform=ax[2].get_xaxis_transform())
    fig.tight_layout()
    savefig(fig, "fig_p2_contrast_metrics")
    return fig, S


# ---------------- lưới ảnh trước/sau ----------------
def pick_examples(df, metric, group="problem"):
    """Quy tắc chọn: trong nhóm vấn đề (train_fit), ảnh có chỉ số gần TRUNG VỊ nhất và ảnh CỰC TRỊ
    (σ lớn nhất với nhiễu, tương phản nhỏ nhất với tương phản thấp)."""
    d = df[(df["group"] == group) & (df["method"] == "none")].dropna(subset=[metric])
    med = d.iloc[(d[metric] - d[metric].median()).abs().argsort()[:1]]
    ext = d.nlargest(1, metric) if metric.startswith("sigma") else d.nsmallest(1, metric)
    return pd.concat([med.assign(rule="trung vị"), ext.assign(rule="cực trị")])


def _flat_crop_center(rgb, size=64):
    """Tâm crop = điểm có mật độ pixel vùng phẳng (mask tham chiếu) cao nhất – nơi nhiễu lộ rõ nhất."""
    R = reference(rgb)
    if R["flat"] is None:
        h, w = rgb.shape[:2]
        return h // 2, w // 2
    dens = cv2.blur(R["flat"].astype(np.float32), (size, size))
    h, w = dens.shape
    dens[: size // 2], dens[-size // 2:], dens[:, : size // 2], dens[:, -size // 2:] = 0, 0, 0, 0
    return np.unravel_index(np.argmax(dens), dens.shape)


def grid_noise(examples, paths, methods=None, size=64):
    setup_style()
    methods = methods or list(NOISE_CANDIDATES)
    n = len(examples)
    fig, ax = plt.subplots(2 * n, len(methods) + 1, figsize=(1.55 * (len(methods) + 1), 1.75 * 2 * n))
    for i, (_, e) in enumerate(examples.iterrows()):
        rgb = load_rgb(paths[e["image_id"]])
        cy, cx = _flat_crop_center(rgb, size)
        ys, xs = slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2)
        th = rgb.copy()
        cv2.rectangle(th, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), (255, 255, 0), 3)
        for r in (2 * i, 2 * i + 1):
            ax[r, 0].axis("off")
        ax[2 * i, 0].imshow(th)
        ax[2 * i, 0].set_title(f"{e['image_id']}\n{e['dx']} · {e['rule']}\nσ={e['sigma_med']:.2f} (EDA)",
                               fontsize=6, color=CLASS_COLORS[e["dx"]])
        for j, m in enumerate(methods):
            p = NOISE_CANDIDATES[m](rgb)
            g = gray_of(p)[ys, xs].astype(np.float32)
            a1, a2 = ax[2 * i, j + 1], ax[2 * i + 1, j + 1]
            a1.imshow(p[ys, xs], interpolation="nearest")
            a1.set_title(m, fontsize=7)
            # phần dư so với làm mịn mạnh: tách "hạt" nhiễu, cùng thang màu ±8 mức xám cho mọi cột
            a2.imshow(g - cv2.GaussianBlur(g, (0, 0), 3), cmap="RdBu_r", vmin=-8, vmax=8, interpolation="nearest")
            for a in (a1, a2):
                a.set_xticks([]), a.set_yticks([])
            if j == 0:
                a1.set_ylabel(f"crop {size}×{size}\n(600×450)", fontsize=6)
                a2.set_ylabel("xám − Gauss(σ=3)\n[−8, 8]", fontsize=6)
    fig.tight_layout(pad=0.3)
    savefig(fig, "grid_p1_noise_before_after", raster=True)
    return fig


def grid_contrast(examples, paths, methods=None):
    setup_style()
    methods = methods or list(CONTRAST_CANDIDATES)
    n = len(examples)
    fig, ax = plt.subplots(2 * n, len(methods), figsize=(1.6 * len(methods), 3.0 * n),
                           gridspec_kw=dict(height_ratios=[1.6, 0.8] * n))
    for i, (_, e) in enumerate(examples.iterrows()):
        rgb = load_rgb(paths[e["image_id"]])
        R = reference(rgb)
        for j, m in enumerate(methods):
            p = CONTRAST_CANDIDATES[m](rgb)
            g = gray_of(p)
            c = g[R["skin"]].mean() - g[R["lesion"]].mean()
            a1, a2 = ax[2 * i, j], ax[2 * i + 1, j]
            a1.imshow(p)
            a1.axis("off")
            a1.set_title(f"{m}\ntương phản={c:.0f}", fontsize=6.3)
            # histogram mức xám vùng da / tổn thương (mask tham chiếu cố định của ảnh gốc)
            bins = np.arange(0, 257, 4)
            a2.hist(g[R["skin"]], bins, color="#E69F00", alpha=0.6, density=True, label="da")
            a2.hist(g[R["lesion"]], bins, color="#0072B2", alpha=0.6, density=True, label="tổn thương")
            a2.set_xlim(0, 255)
            a2.set_yticks([])
            a2.tick_params(labelsize=5.5)
            a2.grid(False)
            if j == 0:
                a2.legend(fontsize=5, loc="upper left")
                a1.text(-0.08, 0.5, f"{e['image_id']} · {e['dx']}\n{e['rule']}", transform=a1.transAxes,
                        rotation=90, va="center", ha="right", fontsize=6.5, color=CLASS_COLORS[e["dx"]])
    fig.tight_layout(pad=0.3, h_pad=1.6)
    savefig(fig, "grid_p2_contrast_before_after", raster=True)
    return fig


def grid_mask_audit(df, paths, n=24, seed=42):
    """Kiểm tra bằng mắt mask tổn thương (Otsu, ước lượng thô) trên n ảnh NGẪU NHIÊN (seed 42) của nhóm tương phản
    thấp ∩ train_fit: đường viền xanh = mask tổn thương, ghi `contrast`. Mục đích: ước lượng tỉ lệ ảnh bị gắn cờ do
    mask sai (vd. chọn nhầm vùng da) chứ không phải do tương phản thấp thật."""
    setup_style()
    d = df[(df["group"] == "problem") & (df["method"] == "none")].sample(n, random_state=seed)
    ncols = 6
    fig, ax = plt.subplots(int(np.ceil(n / ncols)), ncols, figsize=(ncols * 1.75, np.ceil(n / ncols) * 1.5))
    ax = ax.ravel()
    for a in ax:
        a.axis("off")
    for k, (a, (_, r)) in enumerate(zip(ax, d.iterrows())):
        rgb = load_rgb(paths[r["image_id"]])
        R = reference(rgb)
        im = rgb.copy()
        cnts, _ = cv2.findContours(R["lesion"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnts, -1, (0, 255, 255), 4)
        a.imshow(cv2.resize(im, (200, 150), interpolation=cv2.INTER_AREA))
        a.set_title(f"#{k + 1} {r['dx']} · {r['contrast']:.0f}", fontsize=6.5, color=CLASS_COLORS[r["dx"]], pad=2)
    fig.tight_layout(pad=0.3)
    savefig(fig, "grid_p2_lowcontrast_mask_audit", raster=True)
    return fig, d
