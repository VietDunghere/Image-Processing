"""Tính hằng số duy nhất lấy từ dữ liệu của src/preprocess/artifacts.py: L_TARGET (đích của gamma tự động).

L_TARGET = trung vị (qua ảnh) của trung vị L* trong FOV, chỉ trên train_fit và chỉ ảnh KHÔNG bị cờ quá sáng
→ không dùng val/test. Kết quả lưu outputs/preprocess_artifacts/constants.json; dán giá trị vào artifacts.py.

    python -X utf8 scripts/artifact_constants.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.artifacts import _fov, _gray, _to_lab  # noqa: E402
from src.training import read_rgb  # noqa: E402

tr = pd.read_csv("outputs/splits/train_fit.csv")
fl = pd.read_csv("outputs/eda/image_flags.csv").set_index("image_id")
tr = tr[~fl.loc[tr.image_id, "F_overexposed"].values]


def med_L(p):
    rgb = read_rgb(p)
    return float(np.median(_to_lab(rgb)[..., 0][_fov(_gray(rgb))]))


with ThreadPoolExecutor(24) as ex:
    m = np.array(list(ex.map(med_L, tr.path)))
res = {"L_TARGET": round(float(np.median(m)), 2), "n_images": int(len(m)),
       "P10_P90": [round(float(x), 2) for x in np.percentile(m, [10, 90])]}
out = Path("outputs/preprocess_artifacts"); out.mkdir(parents=True, exist_ok=True)
(out / "constants.json").write_text(json.dumps(res, indent=2))
print(res)
