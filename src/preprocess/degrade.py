"""Suy giảm ảnh TỔNG HỢP có kiểm soát cho stress test (mục 2.5c): làm hỏng ảnh val/test theo một loại, một mức.

Mục đích: 2.5 cho thấy trên ảnh gốc tiền xử lý không đổi macro-F1. Câu hỏi ở đây khác: khi ảnh đầu vào xấu đi theo
đúng loại vấn đề mà phép xử lý nhắm tới, phép xử lý có giữ được hiệu năng không?

Quy ước:
  - Áp dụng ở 600×450 trên uint8 RGB, TRƯỚC preprocess_fn và resize 224 (cùng vị trí với vấn đề thật của ảnh).
  - Chỉ áp dụng cho val/test lúc đánh giá; mô hình vẫn được huấn luyện trên ảnh không suy giảm.
  - Tất định: mọi yếu tố ngẫu nhiên lấy từ crc32(image_id|loại|mức) → mọi nhánh so sánh nhìn thấy cùng một ảnh hỏng.
  - Các mức (LEVELS) đặt trước khi chạy, không chỉnh theo kết quả.

Hạn chế cần nêu trong báo cáo: đây là ảnh giả lập, không phải vấn đề thật; với viền và lông, phép xử lý gần như là
nghịch đảo của phép suy giảm nên stress test đo trường hợp thuận lợi nhất cho tiền xử lý.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np

from .noise import _check

# Mức 1 → 3 là nhẹ → nặng. Mức 0 (không suy giảm) xử lý ở degrade().
LEVELS = {
    "noise": (10.0, 20.0, 40.0),         # σ nhiễu Gauss, mức xám, độc lập từng kênh
    "low_contrast": (0.7, 0.5, 0.3),     # hệ số nén L* quanh trung bình (1 = giữ nguyên)
    "overexpose": (1.2, 1.4, 1.7),       # hệ số phơi sáng, nhân RGB rồi cắt ở 255 (mất thông tin thật)
    "uneven": (0.3, 0.5, 0.7),           # độ giảm sáng ở phía tối nhất (0 = đều)
    "vignette": (0.90, 0.80, 0.70),      # bán kính vùng nhìn thấy / nửa đường chéo ảnh
    "hair": (10, 30, 60),                # số sợi lông vẽ thêm
}
# Phép xử lý ghép cặp với từng loại suy giảm (đặt trước khi chạy, theo bảng kết luận EDA).
PAIRS = {
    "noise": ("median3",),
    "low_contrast": ("clahe_L_c2_t8",),
    "overexpose": ("gamma_L_auto",),
    "uneven": ("flat_L_s60_cond", "flat_L_s60"),   # bản _cond ít kích hoạt ở ảnh giả lập → thêm bản toàn bộ
    "vignette": ("vig_fill",),
    "hair": ("hair_bh17_cond", "hair_bh17"),
}


def _rng(image_id: str, kind: str, level: int) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{image_id}|{kind}|{level}".encode()))


def add_noise(rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """g = f + η, η ~ N(0, σ²) độc lập từng pixel và từng kênh (mô hình nhiễu Ch.3)."""
    return np.clip(np.rint(rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape)), 0, 255).astype(np.uint8)


def compress_contrast(rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Nén dải động của L* quanh trung bình: L' = m + α(L − m), giữ a*, b* để chỉ đổi một yếu tố là tương phản."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    m = float(lab[..., 0].mean())
    lab[..., 0] = m + alpha * (lab[..., 0] - m)
    return np.clip(np.rint(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0), 0, 255).astype(np.uint8)


def overexpose(rgb: np.ndarray, gain: float) -> np.ndarray:
    """Phơi sáng quá: nhân RGB rồi cắt ở 255. Phần bị cắt mất thông tin, gamma sau đó không khôi phục được."""
    return np.clip(np.rint(rgb.astype(np.float32) * gain), 0, 255).astype(np.uint8)


def uneven_light(rgb: np.ndarray, a: float, rng: np.random.Generator) -> np.ndarray:
    """Chiếu sáng lệch: nhân với trường tuyến tính 1 − a·t, t ∈ [0, 1] theo hướng ngẫu nhiên riêng từng ảnh."""
    h, w = rgb.shape[:2]
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p = np.cos(theta) * xx + np.sin(theta) * yy
    t = (p - p.min()) / (p.max() - p.min())
    return np.clip(np.rint(rgb.astype(np.float32) * (1 - a * t[..., None])), 0, 255).astype(np.uint8)


def add_vignette(rgb: np.ndarray, r_frac: float) -> np.ndarray:
    """Viền đen ống soi: đen ngoài hình tròn đồng tâm bán kính r_frac × nửa đường chéo, biên làm mềm nhẹ."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = r_frac * np.hypot(w, h) / 2
    m = (np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) <= r).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 3.0)[..., None]
    return np.clip(np.rint(rgb.astype(np.float32) * m), 0, 255).astype(np.uint8)


def add_hair(rgb: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Vẽ n sợi lông: đường cong Bézier bậc 2, dài 120–400 px, dày 2–3 px, màu nâu đen, biên làm mềm nhẹ."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    t = np.linspace(0, 1, 60)[:, None]
    for _ in range(int(n)):
        p0 = rng.uniform((0, 0), (w, h))
        ang, length = rng.uniform(0, 2 * np.pi), rng.uniform(120, 400)
        d = np.array([np.cos(ang), np.sin(ang)])
        p2 = p0 + length * d
        c = (p0 + p2) / 2 + np.array([-d[1], d[0]]) * rng.uniform(-0.3, 0.3) * length
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2
        cv2.polylines(mask, [np.rint(pts).astype(np.int32)], False, 255, int(rng.integers(2, 4)), cv2.LINE_AA)
    a = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255.0
    color = np.array([35, 25, 20], np.float32) + rng.uniform(-10, 10)
    return np.clip(np.rint(rgb.astype(np.float32) * (1 - a) + color * a), 0, 255).astype(np.uint8)


def degrade(rgb: np.ndarray, kind: str, level: int, image_id: str) -> np.ndarray:
    """Áp dụng suy giảm `kind` ở mức `level` (1..3); level = 0 trả lại đúng ảnh gốc."""
    _check(rgb)
    if level == 0:
        return rgb
    v = LEVELS[kind][level - 1]
    if kind == "noise":
        return add_noise(rgb, v, _rng(image_id, kind, level))
    if kind == "low_contrast":
        return compress_contrast(rgb, v)
    if kind == "overexpose":
        return overexpose(rgb, v)
    if kind == "uneven":
        return uneven_light(rgb, v, _rng(image_id, kind, level))
    if kind == "vignette":
        return add_vignette(rgb, v)
    if kind == "hair":
        return add_hair(rgb, v, _rng(image_id, kind, level))
    raise ValueError(f"loại suy giảm không hợp lệ: {kind}")
