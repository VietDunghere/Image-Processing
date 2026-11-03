"""Quy tắc gắn cờ "ảnh gặp vấn đề" – tái lập được, dùng lại cho mục 2.5.

Cách chọn ngưỡng (đơn giản, dễ giải thích):
- Mặc định: quy tắc ngoại lai của BOX PLOT: > Q3 + 1.5·IQR (hoặc < Q1 − 1.5·IQR).
- Ngưỡng tuyệt đối khi phân phối dồn về 0 (Q3 = 0 nên quy tắc box plot vô nghĩa)
  hoặc khi con số có ý nghĩa trực tiếp (vd. "> 10% diện tích ảnh").
Ngưỡng tính trên toàn bộ 10 015 ảnh – thống kê mô tả cho EDA, không phải tham số mô hình.
"""
from __future__ import annotations

import pandas as pd

from .stats import tukey_fence

VIG_T = 0.01          # > 1% diện tích là viền tối
LOW_CONTRAST_T = 20   # mức xám; độ nhạy 15/20/25 in trong notebook
IMPULSE_T = 0.001     # > 0.1% pixel là xung muối–tiêu


def compute_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (flags: bảng True/False theo ảnh, rules: mô tả quy tắc + ngưỡng)."""
    f = pd.DataFrame(index=df.index)
    rules = []

    def add(key, name, metric, mask, thr, how):
        f[key] = mask.fillna(False).astype(bool)
        rules.append(dict(key=key, problem=name, metric=metric, threshold=round(float(thr), 4), rule=how))

    add("vignette", "Viền tối do ống soi", "vignette_frac", df["vignette_frac"] > VIG_T, VIG_T,
        "> 1% diện tích ảnh là vùng đen (<= 20/255) nối với mép ảnh")

    add("overexposed", "Quá sáng / bạc màu", "bright_frac", df["bright_frac"] > 0.10, 0.10,
        "> 10% vùng nhìn thấy là pixel 'trắng bạc' (V >= 235, S <= 40); Q3 = 0 nên dùng ngưỡng tuyệt đối")

    t = tukey_fence(df["gray_mean"], side="low")
    add("underexposed", "Quá tối", "gray_mean", df["gray_mean"] < t, t,
        "mức xám trung bình < Q1 − 1.5·IQR (box plot)")

    add("low_contrast", "Tương phản tổn thương–da thấp", "contrast", df["contrast"] < LOW_CONTRAST_T,
        LOW_CONTRAST_T, f"mức xám da − mức xám tổn thương < {LOW_CONTRAST_T} (≈ 8% dải 0–255)")

    t = tukey_fence(df["illum_range"])
    add("uneven_illum", "Sáng không đều", "illum_range",
        (df["illum_range"] > t) & (df["vignette_frac"] <= VIG_T), t,
        "ô nền sáng nhất − tối nhất (12 ô viền lưới 4x4) > Q3 + 1.5·IQR; bỏ ảnh đã thuộc nhóm viền tối")

    t_hair = tukey_fence(df["hair_frac"])
    t = tukey_fence(df["noise_sigma"])
    add("noise", "Nhiễu Gauss", "noise_sigma", (df["noise_sigma"] > t) & (df["hair_frac"] <= t_hair), t,
        "std phần dư (ảnh − trung vị 3x3) ở vùng phẳng > Q3 + 1.5·IQR; bỏ ảnh nhiều lông (lông bạc bị tính nhầm là nhiễu)")

    add("impulse", "Nhiễu muối–tiêu", "impulse_frac", df["impulse_frac"] > IMPULSE_T, IMPULSE_T,
        "> 0.1% pixel là xung (cực trị lệch >= 60 so với trung vị 3x3)")

    add("hair", "Lông (nhiễu dạng nét)", "hair_frac", df["hair_frac"] > t_hair, t_hair,
        "tỉ lệ pixel lông (black-hat 17x17 >= 20, bỏ đốm < 60 px) > Q3 + 1.5·IQR")

    return f, pd.DataFrame(rules)
