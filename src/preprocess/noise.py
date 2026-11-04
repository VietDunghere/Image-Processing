"""Khử nhiễu Gauss bằng lọc không gian – Chương 3 (slide "Cải thiện ảnh sử dụng cửa sổ" và "Khôi phục ảnh").

Mô hình nhiễu (Ch.3, slide 65): g(x, y) = f(x, y) + η(x, y), η ~ Gauss.
Mọi hàm: vào/ra np.ndarray uint8 RGB (H, W, 3), cùng shape/dtype; lọc từng kênh R, G, B độc lập.
Biên ảnh: lặp lại pixel ở rìa (Ch.3 slide 59, "Lặp lại các pixels ở rìa ảnh") = BORDER_REPLICATE.

Vì sao lọc trên RGB chứ không chỉ trên L: nhiễu cảm biến có ở cả 3 kênh; lọc tuyến tính/trung vị
với cửa sổ nhỏ gần như không đổi màu trung bình của vùng (ΔE được đo ở image_metrics để kiểm chứng).
"""
from __future__ import annotations

import cv2
import numpy as np

_BORDER = cv2.BORDER_REPLICATE


def _check(rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"cần uint8 RGB (H, W, 3), nhận {rgb.dtype} {rgb.shape}")


def denoise_mean(rgb: np.ndarray, k: int = 3) -> np.ndarray:
    """Lọc trung bình số học k×k (Ch.3 slide 50–52, 70): mỗi pixel = trung bình 1/k² của cửa sổ.
    Giảm σ nhiễu trắng ~k lần nhưng làm mờ cạnh đều mọi hướng (slide 52: "chi tiết dần biến mất")."""
    _check(rgb)
    return cv2.blur(rgb, (k, k), borderType=_BORDER)


def denoise_gaussian(rgb: np.ndarray, k: int = 3, sigma: float = 0.0) -> np.ndarray:
    """Lọc trung bình có trọng số / Gauss k×k (Ch.3 slide 53). sigma=0 với k=3 cho đúng nhân của slide
    1/16·[[1,2,1],[2,4,2],[1,2,1]] (OpenCV tự chọn σ = 0.3·((k−1)·0.5 − 1) + 0.8 = 0.8 cho k=3, nhân
    rời rạc [0.25, 0.5, 0.25]). Pixel gần tâm có trọng số lớn hơn → ít mờ hơn lọc trung bình cùng k."""
    _check(rgb)
    return cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=_BORDER)


def denoise_median(rgb: np.ndarray, k: int = 3) -> np.ndarray:
    """Lọc trung vị k×k (Ch.3 slide 56, 73): thay pixel bằng trung vị cửa sổ. Slide: "loại bỏ nhiễu tốt mà
    không có hiệu ứng làm mờ ảnh", giữ cạnh bậc thang; nhưng xóa chi tiết mảnh hơn ~k/2 pixel
    (vd. chấm/nét sắc tố nhỏ). cv2.medianBlur dùng biên lặp lại."""
    _check(rgb)
    return cv2.medianBlur(rgb, k)
