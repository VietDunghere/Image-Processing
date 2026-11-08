"""Chép 7 ảnh mẫu (1 ảnh/lớp) từ tập TEST vào samples/ cho demo và README (đề bài: 5–10 ảnh mẫu trong samples/).

Quy tắc chọn (đặt trước): mỗi lớp rút ngẫu nhiên 1 ảnh của test, random_state = 42. Ảnh HAM10000 theo giấy phép
CC BY-NC 4.0 (Tschandl và cs., 2018) – ghi nguồn trong README.

    python -X utf8 scripts/make_samples.py
"""
import shutil
from pathlib import Path

import pandas as pd

T = pd.read_csv("outputs/splits/test.csv")
out = Path("samples")
out.mkdir(exist_ok=True)
for dx, g in T.groupby("dx"):
    r = g.sample(1, random_state=42).iloc[0]
    shutil.copy(r.path, out / f"{dx}_{r.image_id}.jpg")
    print(dx, r.image_id)
