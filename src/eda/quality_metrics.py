"""Đo chỉ số chất lượng ảnh cho TOÀN BỘ ảnh HAM10000 (mục 2.2 – EDA).

Phạm vi: các vấn đề chất lượng ảnh của CHƯƠNG 3 (Nâng cao chất lượng ảnh):
    1. Viền tối do ống soi        – làm lệch histogram (đỉnh ở mức xám ~0)
    2. Quá sáng / quá tối          – histogram lệch phải / trái  → gamma, log, cân bằng histogram
    3. Tương phản thấp             – histogram hẹp, 1 đỉnh        → giãn / cân bằng histogram
    4. Sáng không đều              – nền sáng tối khác nhau       → cân bằng histogram cục bộ (CLAHE)
    5. Nhiễu Gauss                 – mô hình g = f + η            → lọc trung bình / Gauss / trung vị
    6. Nhiễu muối–tiêu             – nhiễu xung                   → lọc trung vị
    7. Lông                        – "nhiễu" dạng nét tối mảnh    → black-hat + inpainting

Công cụ đo: histogram, phân ngưỡng, lọc trung vị (Ch.3); Otsu, Sobel (Ch.5); black-hat, đóng/mở (Ch.7);
HSV chỉ dùng để nhận biết vùng "trắng bạc" (Ch.4).

Chạy (từ thư mục gốc repo):
    python -X utf8 -m src.eda.quality_metrics --jobs 8
    python -X utf8 -m src.eda.quality_metrics --limit 50   # thử nhanh

Đầu ra (outputs/eda/):
    quality_metrics.csv  – 1 dòng / ảnh, khóa image_id
    hist_gray.npy        – histogram mức xám 256 bin (tỉ lệ) trong vùng nhìn thấy, cùng thứ tự với CSV
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

OUT_DIR = Path("outputs/eda")
SPLIT_DIR = Path("outputs/splits")
IMG_SIZE = 224

# ---- Tham số (mỗi tham số có lý do) ----
DARK_T = 20          # phân ngưỡng: mức xám <= 20/255 coi là "đen" (viền ống soi gần như 0)
OPEN_K = 15          # phép mở 15x15: xóa sợi lông tối mảnh (rộng 2–10 px), chỉ giữ vùng đen lớn
HAIR_K = 17          # black-hat 17x17: lớn hơn bề rộng sợi lông nên phép đóng "lấp" được lông
HAIR_T = 20          # black-hat >= 20 mức xám mới tính là lông
HAIR_MIN_AREA = 60   # bỏ đốm < 60 px (chấm sắc tố) – sợi lông là nét dài nên diện tích lớn
BRIGHT_V, BRIGHT_S = 235, 40   # "trắng bạc": rất sáng (V>=235) và gần như mất màu (S<=40)
IMPULSE_D = 60       # nhiễu xung: pixel cực trị lệch >= 60 so với trung vị 3x3 lân cận
GRID = 4             # lưới 4x4; dùng 12 ô viền để đo nền (4 ô giữa thường chứa tổn thương)
MIN_CELL_SKIN = 0.3  # ô chỉ được tính nếu >= 30% pixel là da


def vignette_mask(gray: np.ndarray, open_k: int = OPEN_K) -> np.ndarray:
    """Viền tối: phân ngưỡng (<= DARK_T) -> phép mở xóa lông -> giữ vùng đen nối với mép ảnh."""
    dark = (gray <= DARK_T).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,) * 2))
    _, lab = cv2.connectedComponents(dark)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, edge[edge > 0])


def blackhat(gray: np.ndarray, k: int = HAIR_K) -> np.ndarray:
    """Black-hat = đóng(ảnh) − ảnh: phép đóng lấp các nét tối mảnh hơn k, hiệu số chỉ còn lại các nét đó."""
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def hair_mask(gray: np.ndarray, valid: np.ndarray, k: int = HAIR_K, min_area: int = HAIR_MIN_AREA) -> np.ndarray:
    m = ((blackhat(gray, k) >= HAIR_T) & valid).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def lesion_mask(gray: np.ndarray, fov: np.ndarray) -> tuple[np.ndarray, float]:
    """Tổn thương (ƯỚC LƯỢNG THÔ): phép đóng xóa lông -> ngưỡng Otsu trong vùng nhìn thấy -> phép mở
    -> chọn vùng tối lớn và gần tâm -> lấp lỗ. Trả về (mask, ngưỡng Otsu)."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HAIR_K,) * 2))
    t, _ = cv2.threshold(g[fov].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((g <= t) & fov).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros_like(fov), float(t)
    h, w = gray.shape
    d = np.hypot((cen[1:, 0] - w / 2) / w, (cen[1:, 1] - h / 2) / h)
    best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA] / (1 + 4 * d)))   # lớn và gần tâm
    les = (lab == best).astype(np.uint8)
    inv = 1 - les                                                      # lấp lỗ: nền không nối mép
    _, lab2 = cv2.connectedComponents(inv)
    edge = np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
    return les.astype(bool) | ((inv == 1) & ~np.isin(lab2, edge)), float(t)


def noise_sigma(gray: np.ndarray, valid: np.ndarray) -> float:
    """Nhiễu Gauss (mô hình g = f + η): lọc trung vị 3x3 giữ lại f ở vùng phẳng, nên phần dư g − median(g) ≈ η.
    Chỉ đo ở VÙNG PHẲNG (20% pixel có độ lớn gradient Sobel nhỏ nhất) để cạnh/kết cấu không bị tính là nhiễu."""
    if valid.sum() < 500:
        return np.nan
    g = gray.astype(np.float32)
    resid = g - cv2.medianBlur(gray, 3).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = valid & (mag <= np.percentile(mag[valid], 20))
    return float(resid[flat].std())


def impulse_frac(gray: np.ndarray, valid: np.ndarray) -> float:
    """Nhiễu muối–tiêu: pixel cực trị (<= 10 hoặc >= 245) lệch >= 60 so với trung vị 3x3 lân cận."""
    med = cv2.medianBlur(gray, 3)
    imp = ((gray <= 10) | (gray >= 245)) & (np.abs(gray.astype(int) - med.astype(int)) >= IMPULSE_D) & valid
    return float(imp.sum() / max(valid.sum(), 1))


def illum_range(gray: np.ndarray, bg: np.ndarray) -> float:
    """Sáng không đều: lưới 4x4, 12 ô viền, mỗi ô = trung vị mức xám của pixel nền (da).
    Chỉ số = ô sáng nhất − ô tối nhất. Bỏ ô có < 30% nền; cần >= 6 ô."""
    h, w = gray.shape
    vals = []
    for i in range(GRID):
        for j in range(GRID):
            if 0 < i < GRID - 1 and 0 < j < GRID - 1:
                continue
            ys, xs = slice(i * h // GRID, (i + 1) * h // GRID), slice(j * w // GRID, (j + 1) * w // GRID)
            m = bg[ys, xs]
            if m.mean() >= MIN_CELL_SKIN:
                vals.append(np.median(gray[ys, xs][m]))
    return float(max(vals) - min(vals)) if len(vals) >= 6 else np.nan


def masks(gray: np.ndarray) -> dict:
    """Các mask dùng chung: vùng nhìn thấy, vùng trong (bỏ dải sát viền), lông, tổn thương, da."""
    fov = ~vignette_mask(gray)
    inner = cv2.erode(fov.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    hair = hair_mask(gray, inner)
    les, t = lesion_mask(gray, fov)
    les_big = cv2.dilate(les.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    return dict(fov=fov, inner=inner, hair=hair, lesion=les, otsu_t=t, skin=fov & ~les_big & ~hair)


def compute_one(path: str):
    pil = Image.open(path)
    W, H, mode = pil.width, pil.height, pil.mode
    rgb = np.asarray(pil.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    M = masks(gray)
    fov, inner, hair, les, skin = M["fov"], M["inner"], M["hair"], M["lesion"], M["skin"]
    n_fov = max(int(fov.sum()), 1)
    has_skin, has_les = skin.sum() > 200, (les & ~hair).sum() > 200

    gf = gray[fov]
    hist = np.bincount(gf, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    g_skin = gray[skin].mean() if has_skin else np.nan
    g_les = gray[les & ~hair].mean() if has_les else np.nan
    bright = (hsv[..., 2] >= BRIGHT_V) & (hsv[..., 1] <= BRIGHT_S) & fov

    # ảnh 224 (độ phân giải CNN nhìn thấy): nhiễu và lông có còn sau resize không?
    g224 = cv2.cvtColor(np.asarray(pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)), cv2.COLOR_RGB2GRAY)
    fov224 = ~vignette_mask(g224, open_k=5)
    in224 = cv2.erode(fov224.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    hair224 = hair_mask(g224, in224, k=7, min_area=10)

    rec = dict(
        width=W, height=H, mode=mode,
        # 1. viền tối
        vignette_frac=float(1 - fov.mean()),
        # 2. độ sáng (trong vùng nhìn thấy)
        gray_mean=float(gf.mean()), skin_gray=float(g_skin), bright_frac=float(bright.sum() / n_fov),
        skin_S=float(hsv[..., 1][skin].mean()) if has_skin else np.nan,
        # 3. tương phản
        gray_std=float(gf.std()), gray_p5=float(np.percentile(gf, 5)), gray_p95=float(np.percentile(gf, 95)),
        entropy=float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
        otsu_t=M["otsu_t"], lesion_gray=float(g_les), contrast=float(g_skin - g_les),
        # 4. sáng không đều (nền = da sáng hơn ngưỡng Otsu, loại cả nốt/tổn thương khác ở ô viền)
        illum_range=illum_range(gray, skin & (gray > M["otsu_t"])),
        # 5–6. nhiễu
        noise_sigma=noise_sigma(gray, inner & ~hair), noise_sigma_224=noise_sigma(g224, in224 & ~hair224),
        impulse_frac=impulse_frac(gray, inner),
        # 7. lông
        hair_frac=float(hair.sum() / n_fov), hair_frac_224=float(hair224.sum() / max(fov224.sum(), 1)),
        skin_frac=float(skin.sum() / n_fov),
    )
    return rec, p.astype(np.float32)


def analyze_image(path: str) -> dict:
    """Ảnh + các mask trung gian (cùng tham số compute_one) để vẽ hình minh họa cách đo."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return dict(rgb=rgb, gray=gray, blackhat=blackhat(gray), **masks(gray))


def _worker(args):
    image_id, path = args
    try:
        rec, h = compute_one(path)
        rec.update(image_id=image_id, error="")
        return rec, h
    except Exception as e:  # 1 ảnh lỗi không làm hỏng cả lượt chạy
        return {"image_id": image_id, "error": repr(e)}, None


def load_index() -> pd.DataFrame:
    """Ghép split đã lưu (train/test) với metadata."""
    sp = pd.concat([pd.read_csv(SPLIT_DIR / "train.csv").assign(split="train"),
                    pd.read_csv(SPLIT_DIR / "test.csv").assign(split="test")], ignore_index=True)
    meta = pd.read_csv("archive/HAM10000_metadata.csv")
    return sp.merge(meta[["image_id", "dx_type", "age", "sex", "localization"]], on="image_id", how="left")


def compute_all(n_jobs: int = 8, limit: int | None = None, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    idx = load_index().sort_values("image_id").reset_index(drop=True)
    if limit:
        idx = idx.head(limit)
    jobs = list(zip(idx["image_id"], idx["path"]))
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x
    with Pool(n_jobs) as pool:
        res = list(tqdm(pool.imap(_worker, jobs, chunksize=16), total=len(jobs)))
    df = idx.merge(pd.DataFrame([r for r, _ in res]), on="image_id", how="left")
    hist = np.stack([h if h is not None else np.full(256, np.nan, np.float32) for _, h in res])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "quality_metrics.csv", index=False)
    np.save(out_dir / "hist_gray.npy", hist)
    print(f"Đã đo {len(df)} ảnh, lỗi: {(df['error'].fillna('') != '').sum()}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    compute_all(a.jobs, a.limit, a.out)
