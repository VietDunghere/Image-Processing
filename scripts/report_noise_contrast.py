"""Hình và bảng phụ cho reports/2.5.1_2.5.2_nhieu_tuong_phan.md.

Đọc output đã lưu trong hai notebook 2.5.1, 2.5.2 nên chạy được cả khi máy chưa có archive/ và outputs/.
Chạy lại notebook thì chạy lại script này để hình và bảng khớp số mới.

    python -X utf8 scripts/report_noise_contrast.py figures   # tách hình notebook, vẽ cột accuracy theo seed, ghi bảng phụ
    python -X utf8 scripts/report_noise_contrast.py check     # chạy hàm xử lý ảnh trên samples/ và ảnh tổng hợp
"""
from __future__ import annotations

import base64
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.eda.plots import setup_style  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures" / "2.5.1_2.5.2"
OUT_DIR = ROOT / "outputs" / "report_noise_contrast"
SEEDS = (42, 43, 44)
PROBLEMS = {
    "noise": dict(notebook="2.5.1_nhieu_gauss.ipynb", title="(a) Nhiễu Gauss", configs=["gauss3", "median3", "median5"]),
    "contrast": dict(notebook="2.5.2_tuong_phan_thap.ipynb", title="(b) Tương phản tổn thương–da thấp",
                     configs=["stretch_L_1_99", "hist_eq_L", "clahe_L_c2_t8"]),
}
FIGURES = {"evidence": "R.fig_evidence", "examples": "R.fig_examples", "before_after": "R.fig_before_after",
           "results": "R.fig_results"}
# Xám sáng + vân gạch cho baseline: tách khỏi xanh cả với người mù màu và khi in đen trắng.
COLORS = ["#B8B8B8", "#0072B2", "#E69F00", "#009E73"]


def find_cell(nb: dict, marker: str) -> dict:
    """Tìm ô theo nội dung thay vì số thứ tự, để thêm ô vào notebook không làm script đọc nhầm."""
    for c in nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return c
    raise KeyError(f"không thấy ô chứa {marker!r}")


def text_of(cell: dict) -> str:
    parts = []
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream":
            parts.append("".join(o["text"]))
        elif "text/plain" in o.get("data", {}):
            parts.append("".join(o["data"]["text/plain"]))
    return "\n".join(parts)


def load_notebook(key: str) -> dict:
    return json.loads((ROOT / PROBLEMS[key]["notebook"]).read_text(encoding="utf-8"))


def extract_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for key in PROBLEMS:
        nb = load_notebook(key)
        for name, marker in FIGURES.items():
            pngs = [o["data"]["image/png"] for o in find_cell(nb, marker).get("outputs", []) if "image/png" in o.get("data", {})]
            (FIG_DIR / f"{key}_{name}.png").write_bytes(base64.b64decode(pngs[0]))


def sample_by_seed(key: str) -> pd.DataFrame:
    """Số ảnh đúng trên mẫu 100 ảnh bị cờ, từng seed, trước (baseline) và sau (cấu hình)."""
    text = text_of(find_cell(load_notebook(key), "R.sample_summary"))
    rows = re.findall(r"^\d+\s+(\S+) \| seed (\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+[\d.]+\s*$", text, re.M)
    d = pd.DataFrame(rows, columns=["config", "seed", "before", "after", "wrong_to_right", "right_to_wrong", "net"])
    return d.astype({c: int for c in d.columns if c != "config"}).assign(problem=key)


def subset_accuracy(key: str) -> pd.DataFrame:
    """Accuracy trên toàn bộ ảnh bị cờ của val/test, trung bình 3 seed: 1 − lỗi, lấy từ bảng DiD của notebook."""
    text = text_of(find_cell(load_notebook(key), "R.target_group_effect"))
    pat = r"^\d\s+(val|test)\s+(\S+)\s+(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$"
    d = pd.DataFrame(re.findall(pat, text, re.M), columns=["split", "config", "n", "err_base", "d_err", "d_rest", "did", "lo", "hi"])
    d = d.astype({c: float for c in d.columns if c not in ("split", "config")})
    d["acc_before_pct"] = 100 * (1 - d.err_base)
    d["acc_after_pct"] = 100 * (1 - d.err_base - d.d_err)
    return d.assign(problem=key)


def fig_seed_bars(tables: dict[str, pd.DataFrame]) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for ax, (key, d) in zip(axes, tables.items()):
        names = ["baseline"] + PROBLEMS[key]["configs"]
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            if name == "baseline":
                v = [d[d.seed == s]["before"].iloc[0] for s in SEEDS]
            else:
                v = [d[(d.seed == s) & (d.config == name)]["after"].iloc[0] for s in SEEDS]
            x = np.arange(len(SEEDS)) + (i - (len(names) - 1) / 2) * width
            ax.bar(x, v, width, color=COLORS[i], edgecolor="white", linewidth=1.2, hatch="///" if i == 0 else None, label=name)
        ax.set_xticks(range(len(SEEDS)), [f"seed {s}" for s in SEEDS])
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        ax.set_title(PROBLEMS[key]["title"])
        ax.legend(ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=7.5)
    axes[0].set_ylabel("số ảnh đúng trên 100 ảnh bị cờ (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_sample100_by_seed.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def run_figures() -> None:
    extract_figures()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = {k: sample_by_seed(k) for k in PROBLEMS}
    fig_seed_bars(seeds)
    pd.concat(seeds.values()).to_csv(OUT_DIR / "sample100_by_seed.csv", index=False)
    acc = pd.concat(subset_accuracy(k) for k in PROBLEMS)
    acc.to_csv(OUT_DIR / "subset_accuracy.csv", index=False)
    print(acc[["problem", "split", "config", "n", "acc_before_pct", "acc_after_pct"]].round(2).to_string(index=False))


def run_check() -> None:
    """Kiểm tra độc lập với dữ liệu: dùng 7 ảnh trong samples/ và ảnh tổng hợp đã biết đáp án."""
    from src.eda.quality_metrics import masks, noise_sigma
    from src.preprocess import get_preprocess
    from src.preprocess import image_metrics as IM

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(glob.glob(str(ROOT / "samples" / "*.jpg")))]
    names = ["gauss3", "median3", "median5", "stretch_L_1_99", "hist_eq_L", "clahe_L_c2_t8"]
    for n in names:
        f = get_preprocess(n)
        for a in imgs:
            o = f(a)
            assert o.shape == a.shape and o.dtype == np.uint8 and np.array_equal(o, f(a)), n
    print(f"{len(names)} hàm: giữ shape và uint8, cùng đầu vào cho cùng đầu ra, trên {len(imgs)} ảnh mẫu")

    impulse = np.zeros((31, 31, 3), np.uint8)
    impulse[15, 15] = 160
    print("đáp ứng xung gauss3 (xung 160):", get_preprocess("gauss3")(impulse)[14:17, 14:17, 0].tolist())

    gain = IM.white_noise_gain(n=20)
    gain.to_csv(OUT_DIR / "check_white_noise_gain.csv", index=False)
    print(gain.round(3).to_string(index=False))

    rng = np.random.default_rng(42)
    rows = []
    for sigma in (1, 2, 5, 10):
        imm, eda = [], []
        for _ in range(20):
            a = np.clip(np.rint(128 + rng.normal(0, sigma, (450, 600))), 0, 255).astype(np.uint8)
            valid = np.ones(a.shape, bool)
            imm.append(IM.sigma_immerkaer(a, valid))
            eda.append(noise_sigma(a, valid))
        rows.append(dict(sigma_true=sigma, immerkaer=np.mean(imm), noise_sigma_eda=np.mean(eda)))
    est = pd.DataFrame(rows)
    est.to_csv(OUT_DIR / "check_noise_estimators.csv", index=False)
    print(est.round(3).to_string(index=False))

    rows = []
    for n in ("stretch_L_1_99", "hist_eq_L", "clahe_L_c2_t8"):
        f, rec = get_preprocess(n), []
        for a in imgs:
            fov = masks(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY))["fov"]
            d = IM.lab_of(f(a)) - IM.lab_of(a)
            rec.append(dict(dL=np.abs(d[..., 0])[fov].mean(), dab=np.hypot(d[..., 1], d[..., 2])[fov].mean(),
                            dE76=np.sqrt((d ** 2).sum(-1))[fov].mean()))
        rows.append(dict(method=n, n_images=len(imgs), **pd.DataFrame(rec).mean().to_dict()))
    lab = pd.DataFrame(rows)
    lab.to_csv(OUT_DIR / "check_lab_split.csv", index=False)
    print(lab.round(2).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "figures":
        run_figures()
    elif cmd == "check":
        run_check()
    else:
        raise SystemExit("dùng: figures | check")
