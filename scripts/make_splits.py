"""Tái tạo split cố định outputs/splits/{train,test,train_fit,val}.csv (giống baseline.ipynb, cell 26–28).

Dùng khi chuyển sang máy khác mà không mang theo outputs/. Split là hàm tất định của metadata + SEED,
nên tái tạo được; script so lại số ảnh từng lớp với bảng đã in trong baseline.ipynb để chắc chắn khớp.

    python -X utf8 scripts/make_splits.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED, TEST_SIZE, VAL_SIZE = 42, 0.2, 0.2
DATA_DIR = Path("archive")
SPLIT_DIR = Path("outputs/splits")
COLS = ["image_id", "lesion_id", "dx", "label", "path"]
# Số ảnh/lớp đã in trong baseline.ipynb (cell 29) -> split tái tạo phải khớp
EXPECTED = {"train_fit": [209, 329, 703, 73, 713, 4291, 90],
            "val": [52, 82, 176, 19, 178, 1073, 23],
            "test": [66, 103, 220, 23, 222, 1341, 29]}

df = pd.read_csv(DATA_DIR / "HAM10000_metadata.csv")
paths = {p.stem: p.as_posix() for d in ["HAM10000_images_part_1", "HAM10000_images_part_2"]
         for p in (DATA_DIR / d).glob("*.jpg")}
df["path"] = df["image_id"].map(paths)
assert df["path"].notna().all()
classes = sorted(df["dx"].unique())
df["label"] = df["dx"].map({c: i for i, c in enumerate(classes)})

sgkf = StratifiedGroupKFold(n_splits=int(round(1 / TEST_SIZE)), shuffle=True, random_state=SEED)
tr, te = next(sgkf.split(df, df["label"], groups=df["lesion_id"]))
trainval, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
sgkf_val = StratifiedGroupKFold(n_splits=int(round(1 / VAL_SIZE)), shuffle=True, random_state=SEED)
fi, vi = next(sgkf_val.split(trainval, trainval["label"], groups=trainval["lesion_id"]))
frames = {"train": trainval, "test": test,
          "train_fit": trainval.iloc[fi].reset_index(drop=True), "val": trainval.iloc[vi].reset_index(drop=True)}

for k, exp in EXPECTED.items():
    got = frames[k]["dx"].value_counts().reindex(classes).tolist()
    assert got == exp, f"{k}: số ảnh/lớp {got} khác bảng gốc {exp} (khác phiên bản sklearn?)"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
for k, f in frames.items():
    f[COLS].to_csv(SPLIT_DIR / f"{k}.csv", index=False)
print({k: len(f) for k, f in frames.items()}, "-> khớp bảng gốc, đã lưu", SPLIT_DIR)
