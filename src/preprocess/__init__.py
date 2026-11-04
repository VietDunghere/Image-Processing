"""Xử lý ảnh Chương 3 (+ Ch.4 hệ màu) cho mục 2.5: khử nhiễu Gauss và tăng tương phản tổn thương–da.

Mọi hàm là hàm thuần: np.ndarray uint8 RGB (H, W, 3) → cùng shape/dtype, áp dụng ở độ phân giải gốc
600×450, TRƯỚC resize 224 và ToTensor/Normalize, giống nhau cho train/val/test.

`get_preprocess(name)` trả về callable picklable (functools.partial) để truyền vào
`src.training.run_experiment(preprocess_fn=...)`.
"""
from __future__ import annotations

from functools import partial

from .contrast import clahe_L, hist_eq_L, stretch_L
from .noise import denoise_gaussian, denoise_mean, denoise_median


def identity(rgb):
    """Không xử lý (baseline)."""
    return rgb


# Ứng viên cho đánh giá định lượng ảnh (giai đoạn 1). Tên = tên thí nghiệm ở giai đoạn 2.
NOISE_CANDIDATES = {
    "none": identity,
    "mean3": partial(denoise_mean, k=3),
    "gauss3": partial(denoise_gaussian, k=3, sigma=0.0),      # đúng nhân 1/16[1 2 1; 2 4 2; 1 2 1] của slide 53
    "gauss5_s1": partial(denoise_gaussian, k=5, sigma=1.0),
    "median3": partial(denoise_median, k=3),
    "median5": partial(denoise_median, k=5),
}
CONTRAST_CANDIDATES = {
    "none": identity,
    "stretch_L_1_99": partial(stretch_L, p_lo=1, p_hi=99, space="lab"),
    "hist_eq_L": partial(hist_eq_L, space="lab"),
    "clahe_L_c1_t8": partial(clahe_L, clip=1.0, tiles=8, space="lab"),
    "clahe_L_c2_t8": partial(clahe_L, clip=2.0, tiles=8, space="lab"),
    "clahe_L_c3_t8": partial(clahe_L, clip=3.0, tiles=8, space="lab"),
    "clahe_V_c2_t8": partial(clahe_L, clip=2.0, tiles=8, space="hsv"),   # đối chứng hệ màu: V (HSV) thay L*
}
CANDIDATES = {**NOISE_CANDIDATES, **CONTRAST_CANDIDATES}


def get_preprocess(name: str):
    """Lấy hàm tiền xử lý theo tên (xem NOISE_CANDIDATES, CONTRAST_CANDIDATES). 'none' → None (baseline)."""
    if name == "none":
        return None
    if "+" in name:   # kết hợp, vd. "median3+clahe_L_c2_t8": áp dụng lần lượt trái → phải
        return partial(_compose, fns=tuple(CANDIDATES[n] for n in name.split("+")))
    return CANDIDATES[name]


def _compose(rgb, fns):
    for f in fns:
        rgb = f(rgb)
    return rgb


__all__ = ["denoise_mean", "denoise_gaussian", "denoise_median", "stretch_L", "hist_eq_L", "clahe_L",
           "identity", "NOISE_CANDIDATES", "CONTRAST_CANDIDATES", "CANDIDATES", "get_preprocess"]
