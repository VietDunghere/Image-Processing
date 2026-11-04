"""Tăng tương phản trên kênh độ sáng để GIỮ MÀU – Chương 3 (toán tử điểm, lược đồ xám) + Chương 4 (hệ màu).

Vì sao chỉ xử lý kênh độ sáng: màu của tổn thương (nâu, xanh xám, đỏ…) là tín hiệu chẩn đoán. Cân bằng
từng kênh R, G, B riêng rẽ làm lệch tỉ lệ R:G:B → đổi màu. Ch.4 (slide 41): HSV "tách biệt giá trị cường độ
ra khỏi các thông tin chứa màu sắc". Hai cách tách được cài:
  - space="lab": kênh L* của CIE L*a*b* (giữ nguyên a*, b*). L* xấp xỉ đều theo cảm nhận; Lab không có trong
    slide nên đây là mở rộng của ý tưởng Ch.4.
  - space="hsv": kênh V = max(R, G, B) của HSV (Ch.4). Thay V bằng V' tương đương nhân cả 3 kênh với V'/V,
    tức giữ nguyên H và S (chính xác, không lượng tử hóa H như cv2 HSV 8-bit).

Vùng nhìn thấy (FOV): ảnh có viền đen ống soi (EDA 2.1, 355 ảnh). Viền chiếm nhiều pixel mức ~0 nên
nếu tính phân vị/CDF trên cả ảnh thì viền "ăn" mất một phần dải mức xám. Vì vậy thống kê (phân vị, CDF)
chỉ tính trên FOV (cùng bộ đo viền với EDA), còn ánh xạ (LUT) áp dụng cho cả ảnh.

Mọi phép biến đổi chỉ dùng thống kê của CHÍNH ảnh đó (không học tham số từ tập dữ liệu) → không rò rỉ,
áp dụng giống hệt nhau cho train/val/test.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.eda.quality_metrics import vignette_mask

from .noise import _check


# ---------------- tách / ghép kênh độ sáng ----------------
def _split_luma(rgb: np.ndarray, space: str):
    """Trả về (Y uint8 0..255, ctx) với Y là L* (thang 0..255) hoặc V."""
    if space == "lab":
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)   # L 0..100, a/b thực
        y = np.clip(np.rint(lab[..., 0] * 2.55), 0, 255).astype(np.uint8)
        return y, lab
    if space == "hsv":
        return rgb.max(axis=2), None
    raise ValueError(f"space phải là 'lab' hoặc 'hsv', nhận {space!r}")


def _merge_luma(rgb: np.ndarray, y_new: np.ndarray, ctx, space: str) -> np.ndarray:
    if space == "lab":
        lab = ctx.copy()
        lab[..., 0] = y_new.astype(np.float32) / 2.55
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0
    else:   # hsv: nhân 3 kênh với V'/V → giữ H, S
        v = rgb.max(axis=2).astype(np.float32)
        gain = np.where(v > 0, y_new.astype(np.float32) / np.maximum(v, 1), 0.0)
        out = rgb.astype(np.float32) * gain[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _fov(rgb: np.ndarray) -> np.ndarray:
    """FOV = phần không thuộc viền đen ống soi (cùng bộ đo vignette_mask của EDA). Ảnh toàn đen → toàn ảnh."""
    fov = ~vignette_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    return fov if fov.sum() >= 1000 else np.ones(fov.shape, bool)


def _apply_lut(rgb, lut_fn, space):
    _check(rgb)
    y, ctx = _split_luma(rgb, space)
    lut = lut_fn(y, _fov(rgb))
    return _merge_luma(rgb, cv2.LUT(y, lut), ctx, space)


# ---------------- các phương pháp ----------------
def stretch_L(rgb: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0, space: str = "lab") -> np.ndarray:
    """Giãn tương phản tuyến tính từng đoạn (Ch.3 slide 21 "Tăng độ tương phản": nhiều hàm tuyến tính trên
    các đoạn): [P_lo, P_hi] của độ sáng trong FOV → [0, 255], ngoài đoạn thì bão hòa (cắt).
    Dùng phân vị (không dùng min/max) để vài pixel lông/phản quang không quyết định hệ số giãn.
    Hệ số giãn = 255/(P_hi − P_lo), tỉ lệ nghịch với dải động của ảnh: ảnh tương phản thấp (P5–P95 ≈ 38
    mức) giãn mạnh hơn ảnh bình thường (nhưng ảnh bình thường của HAM10000 vẫn bị giãn, vì hiếm ảnh phủ
    hết 0–255). Phép biến đổi tuyến tính: nhiễu bị nhân đúng hệ số này, tỉ số tương phản/nhiễu không đổi."""
    def lut_fn(y, fov):
        lo, hi = np.percentile(y[fov], [p_lo, p_hi])
        if hi - lo < 1:
            return np.arange(256, dtype=np.uint8)
        r = np.arange(256, dtype=np.float32)
        return np.clip(np.rint((r - lo) * 255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return _apply_lut(rgb, lut_fn, space)


def hist_eq_L(rgb: np.ndarray, space: str = "lab") -> np.ndarray:
    """Cân bằng lược đồ xám toàn cục (Ch.3 slide 32–36): s_k = (L−1)·Σ_{j≤k} n_j/n, CDF tính trên FOV.
    Làm histogram gần đều → tăng tương phản ở dải mức xám đông pixel nhất (thường là da). Không có tham số,
    nên mức tăng không kiểm soát được; có thể đẩy mạnh khác biệt nhỏ (nhiễu, vân da)."""
    def lut_fn(y, fov):
        h = np.bincount(y[fov], minlength=256).astype(np.float64)
        cdf = np.cumsum(h) / h.sum()
        return np.clip(np.rint(255.0 * cdf), 0, 255).astype(np.uint8)
    return _apply_lut(rgb, lut_fn, space)


def clahe_L(rgb: np.ndarray, clip: float = 2.0, tiles: int = 8, space: str = "lab") -> np.ndarray:
    """CLAHE – cân bằng lược đồ thích nghi có giới hạn tương phản (đề bài DoAn_XuLyAnh.pdf xếp CLAHE vào
    Ch.3; slide Ch.3 dạy cân bằng toàn cục, CLAHE là mở rộng cục bộ của nó). Ảnh chia tiles×tiles ô
    (600×450, tiles=8 → ô 75×56 px); mỗi ô cân bằng riêng, histogram bị cắt ở clip × (trung bình) trước khi
    tính CDF → độ dốc LUT ≤ clip, hạn chế khuếch đại nhiễu; nội suy song tuyến giữa các ô để không lộ ô.
    clip nhỏ (1.0) ≈ gần như không đổi; clip lớn → gần AHE, khuếch đại nhiễu/lông mạnh hơn.
    Lưu ý: CLAHE chạy trên toàn ảnh (cv2 không nhận mask), nên ô góc có viền đen cũng được cân bằng."""
    _check(rgb)
    y, ctx = _split_luma(rgb, space)
    y_new = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles))).apply(y)
    return _merge_luma(rgb, y_new, ctx, space)
