"""Demo 2.8 (Streamlit): kiểm tra chất lượng ảnh (QC) + từng giai đoạn xử lý + dự đoán của baseline.

Theo pipeline đề xuất ở 2.6.3:
  (A) QC: bộ phát hiện + ngưỡng của EDA, chỉ GẮN CỜ, không đổi ảnh đưa vào CNN;
  (B) Phân loại: ảnh GỐC → resize 224 → Normalize ImageNet → ResNet-18 baseline (checkpoint seed 42, chọn trước).
Ảnh sau xử lý chỉ để minh họa; dự đoán trên ảnh đã xử lý hiển thị riêng để so sánh, có ghi chú rằng 2.5 không thấy
khác biệt rõ. KHÔNG dùng cho chẩn đoán.

    streamlit run app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.eda.quality_metrics import compute_one  # noqa: E402
from src.preprocess import CANDIDATES, get_preprocess  # noqa: E402
from src.training import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402

CKPT = ROOT / "outputs/checkpoints/baseline_seed42.pt"
SRC_SIZE = (600, 450)            # kích thước ảnh HAM10000; ngưỡng QC được đặt ở kích thước này
CLASS_NAMES = {"akiec": "Dày sừng quang hóa / Bowen", "bcc": "Ung thư biểu mô tế bào đáy",
               "bkl": "Tổn thương dạng dày sừng lành tính", "df": "U xơ da", "mel": "U hắc tố (melanoma)",
               "nv": "Nốt ruồi (nevus)", "vasc": "Tổn thương mạch máu"}
# Ngưỡng cờ của EDA (src/eda/flags.py; ngưỡng box plot tính trên 10 015 ảnh, xem bảng quy tắc cờ ở mục 2.2)
QC_RULES = [
    ("Viền tối ống soi", "vignette_frac", lambda m: m["vignette_frac"] > 0.01, "> 1% diện tích là viền đen"),
    ("Quá sáng / bạc màu", "bright_frac", lambda m: m["bright_frac"] > 0.10, "> 10% vùng nhìn thấy trắng bạc"),
    ("Lông", "hair_frac", lambda m: m["hair_frac"] > 0.0582, "> 5,82% pixel là lông (black-hat)"),
    ("Sáng không đều", "illum_range", lambda m: m["illum_range"] > 38 and m["vignette_frac"] <= 0.01,
     "chênh nền > 38 mức xám"),
    ("Nhiễu Gauss", "noise_sigma", lambda m: m["noise_sigma"] > 2.16 and m["hair_frac"] <= 0.0582,
     "σ nhiễu > 2,16 mức xám"),
    ("Tương phản tổn thương–da thấp", "contrast", lambda m: m["contrast"] < 20, "< 20 mức xám (mask Otsu thô)"),
]
# Ghi chú từ kết quả 2.5 (3 seed) cho từng bước xử lý minh họa
NOTES = {
    "vig_fill": "2.5.11: không khác biệt rõ so với baseline (test macro-F1 Δ −0,0008).",
    "flat_L_s60_cond": "2.5.11: không khác biệt rõ (Δ +0,0020); trên nhóm sáng không đều tăng nhẹ, CI chứa 0.",
    "hair_bh17": "2.5.11: không khác biệt rõ, nhưng recall mel giảm 0,140 (mel bị đoán thành nv nhiều hơn).",
    "hair_bh17_cond": "2.5.11: không khác biệt rõ (Δ −0,0065).",
    "gamma_L_auto": "2.5.11: không khác biệt rõ; xấu đi trên nhóm quá sáng ở val.",
}


@st.cache_resource
def load_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval(), dev, ck.get("best_epoch")


@torch.no_grad()
def predict(rgb, model, dev):
    """Giống src.training: PIL resize (224, 224) bilinear → /255 → Normalize ImageNet."""
    x = np.asarray(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)).astype(np.float32) / 255
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)
    return torch.softmax(model(t).float(), 1)[0].cpu().numpy()


def qc_metrics(rgb):
    """Chạy đúng bộ đo EDA (compute_one nhận đường dẫn → ghi PNG tạm, không nén mất mát)."""
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        rec, _ = compute_one(f.name)
    return rec


def prob_table(p):
    return pd.DataFrame({"lớp": CLASSES, "tên": [CLASS_NAMES[c] for c in CLASSES], "xác suất": p}) \
        .sort_values("xác suất", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------- giao diện
st.set_page_config(page_title="Demo xử lý ảnh HAM10000", layout="wide")
st.title("Demo: tiền xử lý ảnh dermoscopy và phân loại bằng ResNet-18")
st.warning("Chỉ phục vụ học tập (đồ án Xử lý ảnh). **Không dùng để chẩn đoán.** Mô hình học trên ảnh dermoscopy "
           "HAM10000; ảnh chụp điện thoại thường khác miền dữ liệu nên dự đoán không đáng tin.")
if not CKPT.exists():
    st.error(f"Không tìm thấy checkpoint {CKPT}. Chạy baseline trước: "
             "`python -X utf8 scripts/run_artifact_experiments.py run baseline 42`.")
    st.stop()
model, dev, best_epoch = load_model()

with st.sidebar:
    st.header("Ảnh đầu vào")
    up = st.file_uploader("Tải ảnh (jpg/png)", type=["jpg", "jpeg", "png"])
    samples = sorted((ROOT / "samples").glob("*.jpg")) if (ROOT / "samples").exists() else []
    pick = st.selectbox("…hoặc chọn ảnh mẫu (samples/)", ["(không)"] + [p.name for p in samples])
    st.header("Bước xử lý để minh họa")
    steps = st.multiselect("Áp dụng lần lượt (trái → phải)", [c for c in CANDIDATES if c != "none"],
                           default=["vig_fill", "flat_L_s60_cond"])
    st.caption(f"Mô hình: ResNet-18 baseline, seed 42, epoch {best_epoch}, thiết bị {dev.type}.")

if up is not None:
    rgb = np.asarray(Image.open(up).convert("RGB"))
elif pick != "(không)":
    rgb = np.asarray(Image.open(ROOT / "samples" / pick).convert("RGB"))
else:
    st.info("Tải một ảnh hoặc chọn ảnh mẫu ở thanh bên trái.")
    st.stop()

if rgb.shape[1::-1] != SRC_SIZE:
    st.caption(f"Ảnh {rgb.shape[1]}×{rgb.shape[0]} được resize về {SRC_SIZE[0]}×{SRC_SIZE[1]} "
               "(kích thước HAM10000) để ngưỡng QC và bước xử lý dùng đúng tham số.")
    rgb = np.asarray(Image.fromarray(rgb).resize(SRC_SIZE, Image.BILINEAR))

# (A) QC
st.subheader("(A) Kiểm tra chất lượng ảnh – bộ phát hiện của EDA")
m = qc_metrics(rgb)
qc = pd.DataFrame([dict(vấn_đề=n, chỉ_số=k, giá_trị=round(float(m[k]), 4) if pd.notna(m[k]) else None,
                        ngưỡng=r, gắn_cờ="⚠️ có" if pd.notna(m[k]) and f(m) else "–")
                   for n, k, f, r in QC_RULES])
st.dataframe(qc, hide_index=True, width="stretch")
st.caption("Cờ chỉ là cảnh báo. 2.8: sau khi chuẩn hóa theo thành phần lớp, ảnh 'sáng không đều' và 'quá sáng' có lỗi "
           "baseline cao hơn kỳ vọng (test); đây là liên quan, không phải ước lượng độ tin cậy đã kiểm chứng.")

# Các giai đoạn xử lý
st.subheader("Các giai đoạn xử lý (chỉ minh họa)")
stages = [("Ảnh gốc", rgb)]
cur = rgb
for s in steps:
    cur = get_preprocess(s)(cur)
    stages.append((s, cur))
cols = st.columns(min(len(stages), 4))
for i, (name, img) in enumerate(stages):
    with cols[i % len(cols)]:
        changed = "" if i == 0 else (" (không đổi – ảnh không bị gắn cờ)" if np.array_equal(img, stages[i - 1][1])
                                     else "")
        st.image(img, caption=name + changed, width="stretch")
        if name in NOTES:
            st.caption(NOTES[name])

# (B) Dự đoán
st.subheader("(B) Dự đoán của baseline")
p0 = predict(rgb, model, dev)
c1, c2 = st.columns(2)
with c1:
    top = CLASSES[int(p0.argmax())]
    st.markdown(f"**Trên ảnh gốc (kết quả chính):** `{top}` – {CLASS_NAMES[top]} ({p0.max():.1%})")
    st.bar_chart(prob_table(p0).set_index("lớp")["xác suất"])
if len(stages) > 1:
    p1 = predict(cur, model, dev)
    with c2:
        top1 = CLASSES[int(p1.argmax())]
        st.markdown(f"**Trên ảnh sau xử lý (để so sánh):** `{top1}` – {CLASS_NAMES[top1]} ({p1.max():.1%})")
        st.bar_chart(prob_table(p1).set_index("lớp")["xác suất"])
        st.caption("Mô hình baseline được huấn luyện trên ảnh KHÔNG xử lý, nên ảnh sau xử lý lệch phân phối với dữ "
                   "liệu huấn luyện. Thí nghiệm công bằng (huấn luyện lại trên ảnh đã xử lý, 3 seed) ở mục 2.5: "
                   "không bước nào khác biệt rõ so với baseline.")
