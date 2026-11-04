"""Sinh bảng số liệu cho báo cáo 2.1 (dữ liệu), 2.3 (chuẩn bị/chia tập), 2.4 (baseline: phần cấu hình).

Chỉ ĐỌC: archive/HAM10000_metadata.csv, outputs/splits/*.csv, outputs/data_prep/metadata_clean.csv,
outputs/baseline/{config.json,wrs_one_epoch.csv}. Ghi ra outputs/report_tables/*.csv.
Chạy: python -X utf8 -m src.report.data_tables
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

META = Path("archive/HAM10000_metadata.csv")
SPLITS = Path("outputs/splits")
CLEAN = Path("outputs/data_prep/metadata_clean.csv")
OUT = Path("outputs/report_tables")
SEED = 42
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DX_NAMES = {"nv": "Melanocytic nevi", "mel": "Melanoma", "bkl": "Benign keratosis-like lesions",
            "bcc": "Basal cell carcinoma", "akiec": "Actinic keratoses / Bowen's disease",
            "vasc": "Vascular lesions", "df": "Dermatofibroma"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(META)
    n = len(df)
    log = {}

    # ---- 2.1: phân bố lớp ------------------------------------------------
    vc = df["dx"].value_counts()
    les_per_class = df.groupby("dx")["lesion_id"].nunique()
    t = pd.DataFrame({"ten": [DX_NAMES[c] for c in CLASSES], "n_anh": vc.loc[CLASSES].values,
                      "pct_anh": (vc.loc[CLASSES] / n * 100).round(2).values,
                      "n_lesion": les_per_class.loc[CLASSES].values}, index=CLASSES)
    t["anh_per_lesion"] = (t["n_anh"] / t["n_lesion"]).round(2)
    t.index.name = "dx"
    t.to_csv(OUT / "t21_class_distribution.csv")
    log["imbalance_max_min"] = round(vc.max() / vc.min(), 1)
    log["n_images"], log["n_lesions"] = n, int(df["lesion_id"].nunique())

    # dx_type: tổng và theo lớp
    dt = df["dx_type"].value_counts()
    ct = pd.crosstab(df["dx"], df["dx_type"]).loc[CLASSES]
    ct["tong"] = ct.sum(axis=1)
    ct.loc["tong"] = ct.sum()
    ct.to_csv(OUT / "t21_dx_type_by_class.csv")
    pd.DataFrame({"n_anh": dt, "pct": (dt / n * 100).round(2)}).to_csv(OUT / "t21_dx_type.csv")

    # số ảnh / lesion
    ipl = df.groupby("lesion_id").size()
    ipl_tab = ipl.clip(upper=4).value_counts().sort_index()
    ipl_tab.index = [str(i) if i < 4 else ">=4" for i in ipl_tab.index]
    pd.DataFrame({"n_lesion": ipl_tab, "pct_lesion": (ipl_tab / len(ipl) * 100).round(2)}).to_csv(
        OUT / "t21_images_per_lesion.csv", index_label="anh_per_lesion")
    log["max_images_per_lesion"] = int(ipl.max())
    log["lesions_multi_image"] = int((ipl > 1).sum())
    log["images_in_multi_lesions"] = int(ipl[ipl > 1].sum())
    log["follow_up_classes"] = df.loc[df.dx_type == "follow_up", "dx"].unique().tolist()

    # ---- 2.3: giá trị thiếu, nhất quán, outlier ------------------------------
    miss = pd.DataFrame({
        "nan": df.isna().sum(),
        "unknown": (df.astype(str).apply(lambda s: s.str.lower()) == "unknown").sum()})
    miss["tong"] = miss.sum(axis=1)
    miss["pct"] = (miss["tong"] / n * 100).round(2)
    miss.to_csv(OUT / "t23_missing.csv", index_label="cot")
    miss_cls = pd.DataFrame({
        "age_nan": df[df.age.isna()]["dx"].value_counts(),
        "sex_unknown": df[df.sex == "unknown"]["dx"].value_counts(),
        "loc_unknown": df[df.localization == "unknown"]["dx"].value_counts(),
        "tong_anh": vc}).fillna(0).astype(int).loc[CLASSES]
    miss_cls.to_csv(OUT / "t23_missing_by_class.csv", index_label="dx")
    log["rows_missing_any"] = int((df.age.isna() | (df.sex == "unknown") | (df.localization == "unknown")).sum())

    log["dup_rows"] = int(df.duplicated().sum())
    log["dup_image_id"] = int(df["image_id"].duplicated().sum())
    log["lesions_multi_dx"] = int((df.groupby("lesion_id")["dx"].nunique() > 1).sum())
    inc = df.groupby("lesion_id")[["dx_type", "age", "sex", "localization"]].nunique(dropna=True).gt(1).sum()
    log["lesions_inconsistent_meta"] = {k: int(v) for k, v in inc.items()}

    q1, q3 = df["age"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_mask = (df.age < lo) | (df.age > hi)
    log["age_iqr"] = {"q1": q1, "q3": q3, "iqr": iqr, "lo": lo, "hi": hi,
                      "n_outlier": int(out_mask.sum()),
                      "values": df.loc[out_mask, "age"].value_counts().to_dict(),
                      "dx_outlier": df.loc[out_mask, "dx"].value_counts().to_dict(),
                      "age_min": df.age.min(), "age_max": df.age.max(), "age_median_all": df.age.median()}

    # ---- 2.3: split --------------------------------------------------------
    tr, va, te = (pd.read_csv(SPLITS / f) for f in ("train_fit.csv", "val.csv", "test.csv"))
    trainval = pd.read_csv(SPLITS / "train.csv")
    sp = {"train": tr, "val": va, "test": te}
    dist = pd.DataFrame({k: v["dx"].value_counts() for k, v in sp.items()}).loc[CLASSES].fillna(0).astype(int)
    dist["tong"] = dist.sum(axis=1)
    dist.loc["tong"] = dist.sum()
    dist.to_csv(OUT / "t23_split_class_counts.csv", index_label="dx")
    body = dist.iloc[:-1]
    (body[["train", "val", "test"]] / body[["train", "val", "test"]].sum() * 100).round(2).to_csv(
        OUT / "t23_split_class_pct.csv", index_label="dx")
    (body[["train", "val", "test"]].div(body["tong"], axis=0) * 100).round(1).to_csv(
        OUT / "t23_split_class_share.csv", index_label="dx")
    pd.DataFrame({
        "n_anh": [len(x) for x in sp.values()],
        "pct_anh": [round(len(x) / n * 100, 2) for x in sp.values()],
        "n_lesion": [x["lesion_id"].nunique() for x in sp.values()]}, index=list(sp)).to_csv(
        OUT / "t23_split_overview.csv", index_label="tap")

    # số lesion mỗi lớp mỗi tập: cỡ mẫu hiệu dụng (ảnh cùng lesion không độc lập)
    les = pd.DataFrame({k: v.groupby("dx")["lesion_id"].nunique() for k, v in sp.items()}).loc[CLASSES].fillna(0).astype(int)
    les["tong"] = df.groupby("dx")["lesion_id"].nunique().loc[CLASSES]
    les.to_csv(OUT / "t23_split_class_lesions.csv", index_label="dx")

    # kiểm tra rò rỉ + tái tạo split
    ids = {k: set(v["lesion_id"]) for k, v in sp.items()}
    log["leak_lesion_overlap"] = {f"{a}-{b}": len(ids[a] & ids[b]) for a, b in
                                  [("train", "val"), ("train", "test"), ("val", "test")]}
    log["all_images_covered_once"] = bool(
        len(tr) + len(va) + len(te) == n and set(pd.concat([tr, va, te])["image_id"]) == set(df["image_id"]))
    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)})
    s1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, b = next(s1.split(df, df["label"], groups=df["lesion_id"]))
    log["test_split_reproduced"] = bool(set(df.iloc[b]["image_id"]) == set(te["image_id"]))
    log["trainval_equals_train_csv"] = bool(set(trainval["image_id"]) == set(pd.concat([tr, va])["image_id"]))

    # age: median fit trên train; số ảnh thiếu tuổi mỗi tập
    clean = pd.read_csv(CLEAN)
    tr_age = df[df.image_id.isin(tr.image_id)]["age"]
    log["age_median_train"] = float(tr_age.median())
    log["age_median_trainval"] = float(df[df.image_id.isin(trainval.image_id)]["age"].median())
    log["age_missing_per_split"] = clean.groupby("split")["age_missing"].sum().astype(int).to_dict()
    log["age_outlier_per_split"] = clean.groupby("split")["age_outlier"].sum().astype(int).to_dict()
    log["clean_remaining_nan"] = int(clean.isna().sum().sum())

    # dx_type theo tập (follow_up chỉ ở nv)
    d2 = df.merge(clean[["image_id", "split"]], on="image_id")
    pd.crosstab(d2["split"], d2["dx_type"]).to_csv(OUT / "t23_split_dx_type.csv")

    # ---- WRS & cấu hình -------------------------------------------------------
    cfg = json.loads(Path("outputs/baseline/config.json").read_text(encoding="utf-8"))
    pd.Series(cfg).astype(str).to_frame("gia_tri").to_csv(OUT / "t24_config.csv", index_label="tham_so")
    wrs = pd.read_csv("outputs/baseline/wrs_one_epoch.csv", index_col=0)
    wrs["pct_ky_vong"] = round(100 / 7, 2)
    wrs.to_csv(OUT / "t23_wrs_table.csv", index_label="dx")
    ntr = tr["dx"].value_counts()
    log["train_imbalance_max_min"] = round(ntr.max() / ntr.min(), 1)

    # mốc tham chiếu: bộ phân loại hằng (luôn đoán lớp nv) trên test -> vì sao không dùng riêng accuracy
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    yt = te["dx"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    yp = [CLASSES.index("nv")] * len(yt)
    triv = {"accuracy": accuracy_score(yt, yp), "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
            "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0)}
    pd.Series(triv).round(4).to_frame("luon_doan_nv").to_csv(OUT / "t24_trivial_classifier_test.csv", index_label="chi_so")
    log["trivial_nv_test"] = {k: round(v, 4) for k, v in triv.items()}

    # số tham số ResNet-18 (fc -> 7 lớp); weights=None để không cần tải trọng số
    import torch
    from torch import nn
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    log["resnet18_params_7cls"] = int(sum(p.numel() for p in m.parameters()))

    (OUT / "data_checks.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
