"""Đánh giá định lượng ảnh trước/sau xử lý (giai đoạn 1 của 2.5, chạy CPU, không cần CNN).

Nguyên tắc:
  - Chỉ dùng ảnh thuộc train_fit (outputs/splits/train_fit.csv) → việc chọn tham số ở đây không nhìn val/test.
  - Nhóm vấn đề = problem_sets/{noise,low_contrast}.csv ∩ train_fit; nhóm đối chứng = cùng cỡ, rút ngẫu nhiên
    (seed 42) từ phần còn lại của train_fit.
  - Mask (FOV, vùng phẳng, tổn thương, da, lông) tính MỘT LẦN trên ảnh GỐC rồi dùng chung cho mọi phương pháp:
    so sánh trên cùng một vùng pixel, không để việc xử lý làm đổi mask (vd. Otsu lệch sau CLAHE).
  - Đo ở độ phân giải gốc 600×450 VÀ sau resize 224×224 (PIL bilinear có antialias – giống
    torchvision.transforms.Resize trên ảnh PIL của baseline), vì CNN chỉ nhìn thấy ảnh 224.

Chạy (từ thư mục gốc repo):
    python -X utf8 -m src.preprocess.image_metrics --kind noise --jobs 12
    python -X utf8 -m src.preprocess.image_metrics --kind contrast --jobs 12
    python -X utf8 -m src.preprocess.image_metrics --kind combo --jobs 12     # khử nhiễu + tăng tương phản
    python -X utf8 -m src.preprocess.image_metrics --kind check --jobs 12     # chạy mọi ứng viên trên 10 015 ảnh
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.eda.quality_metrics import hair_mask, masks, vignette_mask

from . import CANDIDATES, CONTRAST_CANDIDATES, NOISE_CANDIDATES, get_preprocess, identity

OUT_DIR = Path("outputs/preprocess")
SPLIT_DIR = Path("outputs/splits")
PS_DIR = Path("outputs/eda/problem_sets")
IMG_SIZE = 224
SEED = 42
# Kết hợp (thứ tự khử nhiễu → tăng tương phản, vì tăng tương phản nhân cả nhiễu): đo trên nhóm tương phản thấp
COMBO_CANDIDATES = {"none": identity, **{n: get_preprocess(n) for n in [
    "stretch_L_1_99", "gauss3+stretch_L_1_99", "median3+stretch_L_1_99",
    "clahe_L_c2_t8", "gauss3+clahe_L_c2_t8", "median3+clahe_L_c2_t8"]}}
KIND_SET = {"noise": ("noise", NOISE_CANDIDATES), "contrast": ("low_contrast", CONTRAST_CANDIDATES),
            "combo": ("low_contrast", COMBO_CANDIDATES)}
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


# ---------------- tiện ích ----------------
def load_rgb(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize224(rgb: np.ndarray) -> np.ndarray:
    """Giống baseline: transforms.Resize((224, 224)) trên PIL = PIL bilinear (có antialias khi thu nhỏ)."""
    return np.asarray(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


def gray_of(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def flat_mask(gray: np.ndarray, valid: np.ndarray, q: float = 20) -> np.ndarray:
    """Vùng phẳng = q% pixel hợp lệ có độ lớn gradient Sobel nhỏ nhất (cùng định nghĩa EDA)."""
    g = gray.astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return valid & (mag <= np.percentile(mag[valid], q))


def sigma_median_resid(gray, flat) -> float:
    """Bộ đo của EDA: std(ảnh − trung vị 3×3) ở vùng phẳng. LƯU Ý: thiên lệch về 0 sau lọc trung vị
    (trung vị gần như lũy đẳng), nên khi so phương pháp phải xem thêm sigma_immerkaer."""
    resid = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    return float(resid[flat].std())


def sigma_immerkaer(gray, flat) -> float:
    """Immerkaer (1996): σ ≈ sqrt(π/2)/6 · E|I * N|, N = nhân Laplacian hiệu 2 cấp (triệt tiêu mặt phẳng
    tuyến tính). Độc lập với lọc trung vị nên dùng được để so sánh mọi bộ lọc."""
    r = cv2.filter2D(gray.astype(np.float32), -1, IMMERKAER, borderType=cv2.BORDER_REPLICATE)
    return float(np.sqrt(np.pi / 2) / 6 * np.abs(r[flat]).mean())


def entropy(gray, m) -> float:
    p = np.bincount(gray[m], minlength=256).astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def hf_fraction(gray, cutoff: float = 0.25) -> float:
    """Tỉ lệ năng lượng phổ ở tần số > cutoff chu kỳ/pixel (cửa sổ Hann, bỏ DC) – đại diện chi tiết mịn/nhiễu."""
    g = gray.astype(np.float32)
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(g.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(g.shape[1]))[None, :]
    r = np.hypot(fy, fx)
    tot = P[r > 0].sum()
    return float(P[r > cutoff].sum() / tot) if tot > 0 else np.nan


def lab_of(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


# ---------------- mask tham chiếu (ảnh gốc) ----------------
def reference(rgb: np.ndarray) -> dict:
    g = gray_of(rgb)
    M = masks(g)
    valid = M["inner"] & ~M["hair"]
    les = M["lesion"] & ~M["hair"]
    g224 = gray_of(resize224(rgb))
    fov224 = ~vignette_mask(g224, open_k=5)
    in224 = cv2.erode(fov224.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    hair224 = hair_mask(g224, in224, k=7, min_area=10)
    les224 = cv2.resize(M["lesion"].astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
    ok = valid.sum() >= 500
    return dict(
        fov=M["fov"], inner=M["inner"], skin=M["skin"], lesion=les,
        flat=flat_mask(g, valid) if ok else None,
        in224=in224, les224=les224 & in224,
        flat224=flat_mask(g224, in224 & ~hair224) if (in224 & ~hair224).sum() >= 200 else None,
        lab=lab_of(rgb),
    )


def measure(proc: np.ndarray, R: dict) -> dict:
    """Mọi chỉ số trên ảnh đã xử lý `proc`, với mask tham chiếu R của ảnh gốc."""
    g = gray_of(proc)
    p224 = resize224(proc)
    g224 = gray_of(p224)
    fov, skin, les = R["fov"], R["skin"], R["lesion"]
    gf = g[fov]
    lap224 = cv2.Laplacian(g224.astype(np.float32), cv2.CV_32F)
    sx, sy = cv2.Sobel(g224.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(g224.astype(np.float32), cv2.CV_32F, 0, 1)
    lab = lab_of(proc)
    d = lab - R["lab"]
    has_c = skin.sum() > 200 and les.sum() > 200
    x224 = p224.reshape(-1, 3)[R["in224"].ravel()] / 255.0
    rec = dict(
        # nhiễu (gốc + 224), 2 bộ đo
        sigma_med=sigma_median_resid(g, R["flat"]) if R["flat"] is not None else np.nan,
        sigma_imm=sigma_immerkaer(g, R["flat"]) if R["flat"] is not None else np.nan,
        sigma_med_224=sigma_median_resid(g224, R["flat224"]) if R["flat224"] is not None else np.nan,
        sigma_imm_224=sigma_immerkaer(g224, R["flat224"]) if R["flat224"] is not None else np.nan,
        # độ nét / chi tiết ở 224 (cái CNN thấy)
        lapvar_224=float(lap224[R["in224"]].var()),
        lapvar_les_224=float(lap224[R["les224"]].var()) if R["les224"].sum() > 100 else np.nan,
        tenengrad_224=float((sx ** 2 + sy ** 2)[R["in224"]].mean()),
        hf_224=hf_fraction(g224),
        # tương phản (gốc, mask cố định)
        contrast=float(g[skin].mean() - g[les].mean()) if has_c else np.nan,
        gray_std=float(gf.std()), p5_p95=float(np.subtract(*np.percentile(gf, [95, 5]))),
        entropy=entropy(g, fov),
        # màu: ΔE76 (gồm cả thay đổi L* có chủ đích) và Δab (chỉ phần màu, L* bỏ ra)
        dE76=float(np.sqrt((d ** 2).sum(-1))[fov].mean()),
        dab=float(np.hypot(d[..., 1], d[..., 2])[fov].mean()),
        # lông sau xử lý (rủi ro khuếch đại nét tối mảnh)
        hair_frac=float(hair_mask(g, R["inner"]).sum() / max(fov.sum(), 1)),
    )
    for i, c in enumerate("RGB"):   # thống kê kênh ở 224 để so với ImageNet mean/std
        rec[f"mean_{c}"], rec[f"std_{c}"] = float(x224[:, i].mean()), float(x224[:, i].std())
    return rec


# ---------------- chạy theo nhóm ảnh ----------------
def select_images(kind: str, seed: int = SEED) -> pd.DataFrame:
    """Nhóm vấn đề ∩ train_fit + đối chứng cùng cỡ (ngẫu nhiên seed 42) từ phần còn lại của train_fit."""
    key, _ = KIND_SET[kind]
    tf = pd.read_csv(SPLIT_DIR / "train_fit.csv")
    ps = pd.read_csv(PS_DIR / f"{key}.csv")
    prob = tf[tf["image_id"].isin(ps["image_id"])].assign(group="problem")
    ctrl = tf[~tf["image_id"].isin(ps["image_id"])].sample(n=len(prob), random_state=seed).assign(group="control")
    return pd.concat([prob, ctrl], ignore_index=True)[["image_id", "lesion_id", "dx", "path", "group"]]


def _worker(args):
    cv2.setNumThreads(1)
    image_id, path, kind = args
    cands = KIND_SET[kind][1]
    rgb = load_rgb(path)
    R = reference(rgb)
    out = []
    for name, fn in cands.items():
        t0 = time.perf_counter()
        proc = fn(rgb)
        t_ms = (time.perf_counter() - t0) * 1000
        rec = measure(proc, R)
        rec.update(image_id=image_id, method=name, t_ms=t_ms)
        out.append(rec)
    return out


def run(kind: str, n_jobs: int = 12, limit: int | None = None) -> pd.DataFrame:
    sel = select_images(kind)
    if limit:
        sel = sel.groupby("group").head(limit)
    jobs = [(r.image_id, r.path, kind) for r in sel.itertuples()]
    with Pool(n_jobs) as pool:
        res = pool.map(_worker, jobs, chunksize=4)
    df = pd.DataFrame([r for rs in res for r in rs]).merge(sel.drop(columns="path"), on="image_id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"image_metrics_{kind}.csv", index=False)
    return df


# ---------------- nhiễu trắng tổng hợp: resize/lọc giảm σ bao nhiêu? ----------------
def white_noise_gain(sigma: float = 10.0, n: int = 20, seed: int = SEED) -> pd.DataFrame:
    """Ảnh phẳng 128 + N(0, σ²) (600×450, RGB độc lập) → mỗi bộ lọc → (có/không) resize 224.
    Đo std kênh xám ở vùng trong (bỏ 8 px biên). Cho hệ số giảm σ LÝ THUYẾT với nhiễu trắng; nhiễu thật
    (sau demosaic + JPEG) tương quan không gian nên giảm ít hơn – vì vậy vẫn phải đo trên ảnh thật."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        img = np.clip(np.rint(128 + rng.normal(0, sigma, (450, 600, 3))), 0, 255).astype(np.uint8)
        for name, fn in NOISE_CANDIDATES.items():
            p = fn(img)
            g, g224 = gray_of(p).astype(np.float32), gray_of(resize224(p)).astype(np.float32)
            rows.append(dict(method=name, sigma_native=g[8:-8, 8:-8].std(), sigma_224=g224[4:-4, 4:-4].std()))
    df = pd.DataFrame(rows).groupby("method", sort=False).mean().reset_index()
    ref = df.loc[df["method"] == "none", "sigma_native"].item()
    df["ratio_native"], df["ratio_224"] = df["sigma_native"] / ref, df["sigma_224"] / ref
    return df


# ---------------- kiểm tra preprocess_fn trên toàn bộ 10 015 ảnh ----------------
def _check_worker(args):
    cv2.setNumThreads(1)
    image_id, path = args
    rgb = load_rgb(path)
    rec = {"image_id": image_id}
    for name, fn in CANDIDATES.items():
        if name == "none":
            continue
        t0 = time.perf_counter()
        p = fn(rgb)
        rec[f"t_{name}"] = (time.perf_counter() - t0) * 1000
        rec[f"ok_{name}"] = bool(p.shape == rgb.shape and p.dtype == np.uint8)
    return rec


def check_all(n_jobs: int = 12) -> pd.DataFrame:
    idx = pd.concat([pd.read_csv(SPLIT_DIR / f) for f in ("train.csv", "test.csv")], ignore_index=True)
    t0 = time.time()
    with Pool(n_jobs) as pool:
        res = pool.map(_check_worker, list(zip(idx["image_id"], idx["path"])), chunksize=16)
    df = pd.DataFrame(res)
    wall = time.time() - t0
    names = [c[3:] for c in df.columns if c.startswith("ok_")]
    summ = pd.DataFrame({
        "method": names,
        "n_images": len(df),
        "all_ok_shape_dtype": [bool(df[f"ok_{n}"].all()) for n in names],
        "ms_per_image_median": [df[f"t_{n}"].median() for n in names],
        "ms_per_image_p95": [df[f"t_{n}"].quantile(0.95) for n in names],
        "est_total_s_single_thread": [df[f"t_{n}"].sum() / 1000 for n in names],
    })
    summ.attrs["wall_s_all_methods"] = wall
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT_DIR / "preprocess_timing.csv", index=False)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["noise", "contrast", "combo", "check"], required=True)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.kind == "check":
        print(check_all(a.jobs))
    else:
        d = run(a.kind, a.jobs, a.limit)
        print(d.groupby(["group", "method"]).median(numeric_only=True).T)
