"""Quy cách hình cho EDA: cùng thứ tự + màu 7 lớp trên mọi hình, xuất PDF vector + PNG 300 dpi."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FIG_DIR = Path("outputs/figures/eda")

# Thứ tự lớp = sorted(dx) như mã hóa label trong notebook baseline
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# Okabe–Ito (+ #882255), thứ tự đã kiểm tra tách biệt cho mù màu đỏ–lục (deutan ΔE >= 9.6)
CLASS_COLORS = dict(zip(CLASSES, ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                                  "#56B4E9", "#CC79A7", "#882255"]))
FLAG_COLOR, OK_COLOR = "#D55E00", "#7A7A7A"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 80, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })


def savefig(fig, name: str, raster: bool = False):
    """Biểu đồ: PDF vector + PNG 300 dpi. Lưới ảnh chụp (raster=True): chỉ PNG 300 dpi (PDF không thêm gì)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not raster:
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    return FIG_DIR / f"{name}.png"


def image_grid(rows, title_fn, ncols=6, thumb=(200, 150), suptitle=None, name=None):
    """Lưới ảnh có nhãn lớp + giá trị chỉ số. rows: DataFrame có cột path, dx."""
    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.75, nrows * 1.55))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, rows.iterrows()):
        ax.imshow(Image.open(r["path"]).convert("RGB").resize(thumb))
        ax.set_title(title_fn(r), fontsize=6.5, color=CLASS_COLORS.get(r["dx"], "k"), pad=2)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.0)
    fig.tight_layout(pad=0.3)
    if name:
        savefig(fig, name, raster=True)
    return fig
