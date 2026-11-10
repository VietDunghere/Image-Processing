"""Demo 2.8 (Streamlit): Hệ thống kiểm tra chất lượng ảnh (QC), trực quan hóa từng bước xử lý ảnh cổ điển,
giải thích Grad-CAM và phân loại tổn thương da dermoscopy (HAM10000) bằng CNN.

Thiết kế bám sát pipeline đề xuất ở mục 2.6.3:
  (A) Nhánh QC: bộ phát hiện + ngưỡng của EDA (mục 2.2), chỉ GẮN CỜ cảnh báo, không đổi ảnh đưa vào CNN;
  (B) Nhánh Phân loại: ảnh GỐC → resize 224×224 → Normalize ImageNet → ResNet-18 baseline (seed 42, chọn trước).
Các bước xử lý ảnh (Ch.3, Ch.4, Ch.7) và bộ mô phỏng suy giảm (mục 2.5c) được hiển thị tách biệt kèm biểu đồ
histogram, mask hình thái học, bản đồ hiệu pixel và Grad-CAM để minh họa trực quan vì sao tiền xử lý đại trà có thể
gây tác dụng phụ và khi nào bước có điều kiện giữ an toàn cho ảnh bình thường.

Chỉ phục vụ học tập (Đồ án Xử lý ảnh – PTIT). KHÔNG dùng cho mục đích y tế hay chẩn đoán lâm sàng.

Chạy thử nghiệm cục bộ (bắt buộc bind localhost khi kiểm thử):
    streamlit run app.py --server.address 127.0.0.1 --server.port 8501
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.eda.quality_metrics import analyze_rgb, compute_from_rgb  # noqa: E402
from src.preprocess import CANDIDATES, get_preprocess  # noqa: E402
from src.preprocess.degrade import LEVELS, degrade  # noqa: E402
from src.training import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402

# TODO(security): Ứng dụng chạy offline phục vụ báo cáo học thuật trên localhost (127.0.0.1) hoặc container nội bộ.
# Nếu triển khai công khai lên môi trường mạng, cần bổ sung lớp reverse-proxy TLS kèm xác thực OAuth2/OIDC (MFA),
# giới hạn tần suất request (rate limiting), quét mã độc tệp tải lên trong sandbox và cấu hình header bảo mật
# nghiêm ngặt (Content-Security-Policy, X-Frame-Options: DENY, X-Content-Type-Options: nosniff).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # Giới hạn 10 MB chống tấn công cạn kiệt bộ nhớ (DoS)
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_PIXELS = 25_000_000

SAMPLES_DIR = (ROOT / "samples").resolve()
DEMO_ASSETS_DIR = (ROOT / "demo_assets").resolve()
SRC_SIZE = (600, 450)  # Kích thước chuẩn HAM10000 (4:3) mà các ngưỡng EDA được thiết lập


def resolve_checkpoint() -> Path:
    """Ưu tiên biến môi trường XLA_CHECKPOINT_PATH -> outputs/checkpoints -> demo_assets (chống Path Traversal)."""
    env_p = os.environ.get("XLA_CHECKPOINT_PATH", "").strip()
    if env_p:
        p = Path(env_p).resolve()
        if p.is_file() and p.suffix == ".pt":
            return p
    for rel in ("outputs/checkpoints/baseline_seed42.pt", "demo_assets/baseline_seed42.pt"):
        p = (ROOT / rel).resolve()
        if p.is_file():
            return p
    return ROOT / "outputs/checkpoints/baseline_seed42.pt"


def resolve_table(out_rel: str, asset_name: str) -> Path | None:
    for p in (ROOT / out_rel, DEMO_ASSETS_DIR / asset_name):
        if p.is_file():
            return p
    return None


CKPT = resolve_checkpoint()

CLASS_NAMES = {
    "akiec": "Dày sừng quang hóa / bệnh Bowen (akiec)",
    "bcc": "Ung thư biểu mô tế bào đáy (bcc)",
    "bkl": "Tổn thương dạng dày sừng lành tính (bkl)",
    "df": "U xơ da (df)",
    "mel": "U hắc tố ác tính – Melanoma (mel)",
    "nv": "Nốt ruồi sắc tố lành tính – Nevus (nv)",
    "vasc": "Tổn thương mạch máu (vasc)",
}

QC_RULES = [
    ("Viền tối ống soi", "vignette_frac", lambda m: m["vignette_frac"] > 0.01, "> 0,0100 (> 1% diện tích)"),
    ("Quá sáng / bạc màu", "bright_frac", lambda m: m["bright_frac"] > 0.10, "> 0,1000 (> 10% FOV trắng bạc)"),
    ("Lông che phủ", "hair_frac", lambda m: m["hair_frac"] > 0.0582, "> 0,0582 (> 5,82% FOV black-hat)"),
    ("Sáng không đều", "illum_range", lambda m: m["illum_range"] > 38 and m["vignette_frac"] <= 0.01,
     "> 38,0 mức xám (không viền)"),
    ("Nhiễu Gauss", "noise_sigma", lambda m: m["noise_sigma"] > 2.16 and m["hair_frac"] <= 0.0582,
     "> 2,16 mức xám (ít lông)"),
    ("Tương phản thấp", "contrast", lambda m: m["contrast"] < 20, "< 20,0 mức xám (mask Otsu thô)"),
]

PRESETS = {
    "Mặc định minh họa (có điều kiện): vig_fill → flat_L_s60_cond": ["vig_fill", "flat_L_s60_cond"],
    "Khử nhiễu + tăng tương phản cục bộ: median3 → clahe_L_c2_t8": ["median3", "clahe_L_c2_t8"],
    "Xóa lông có điều kiện (chỉ ảnh bị cờ): hair_bh17_cond": ["hair_bh17_cond"],
    "Xóa lông đại trà (minh họa tác dụng phụ lên sắc tố): hair_bh17": ["hair_bh17"],
    "Hiệu chỉnh quá sáng tự động: gamma_L_auto": ["gamma_L_auto"],
    "Cân bằng lược đồ toàn cục (minh họa kết quả âm): hist_eq_L": ["hist_eq_L"],
    "Tùy chọn thủ công...": None,
}

DEGRADE_LABELS = {
    "none": "Không làm suy giảm (ảnh gốc)",
    "vignette": "Viền đen ống soi giả lập (vignette)",
    "hair": "Sợi lông giả lập (hair)",
    "noise": "Nhiễu cộng Gauss (noise)",
    "uneven": "Chiếu sáng lệch góc (uneven)",
    "low_contrast": "Nén dải tương phản (low_contrast)",
    "overexpose": "Phơi sáng quá mức (overexpose)",
}

STEP_NOTES = {
    "gauss3": "Ch.3 Lọc Gauss 3×3: giảm σ nhiễu nhưng resize 224 đã khử phần lớn nhiễu mịn; không khác biệt rõ với baseline.",
    "median3": "Ch.3 Lọc trung vị 3×3: khử nhiễu giữ cạnh bậc thang; trên dữ liệu thật không khác biệt rõ.",
    "median5": "Ch.3 Lọc trung vị 5×5: cửa sổ lớn làm mất ~75% phương sai Laplacian vùng tổn thương.",
    "stretch_L_1_99": "Ch.3+4 Giãn tuyến tính [P1, P99] trên L*: làm biến đổi mạnh ảnh bình thường → tệ hơn rõ trên test.",
    "hist_eq_L": "Ch.3+4 Cân bằng lược đồ toàn cục trên L*: khuếch đại nhiễu và lệch màu mạnh → tệ hơn rõ trên cả val & test.",
    "clahe_L_c2_t8": "Ch.3+4 CLAHE (clip=2, lưới 8×8) trên L*: tăng tương phản cục bộ vừa phải, trung tính về macro-F1.",
    "hair_bh17": "Ch.7 Black-hat 17×17 + Telea inpainting (mọi ảnh): bắt nhầm mạng sắc tố tối của mel → giảm recall mel.",
    "hair_bh17_cond": "Ch.7 Black-hat 17×17 có điều kiện (chỉ khi hair_frac > 5,82%): giữ nguyên 100% ảnh đối chứng.",
    "vig_fill": "Ch.3+7 Tô viền đen bằng trung vị da lân cận + làm mềm biên Gauss 21×21: không đổi tỉ lệ tổn thương.",
    "vig_crop": "Ch.3+7 Cắt khung đồng tâm 4:3 bỏ viền đen rồi phóng lại: làm thay đổi tỉ lệ kích thước tổn thương.",
    "gamma_L_1.5": "Ch.3+4 Biến đổi lũy thừa γ=1,5 cho mọi ảnh: làm tối cả ảnh bình thường (ΔE76 đối chứng cao).",
    "gamma_L_auto": "Ch.3+4 Gamma tự động đưa trung vị L* về L_TARGET=66,61 (chỉ ảnh quá sáng): giữ nguyên ảnh thường.",
    "flat_L_s60": "Ch.3+4 Hiệu chỉnh nền phẳng Gauss σ=60 trên L* (mọi ảnh).",
    "flat_L_s60_cond": "Ch.3+4 Hiệu chỉnh nền phẳng có điều kiện (chỉ khi illum_range > 38 và không viền).",
}


def validate_and_decode_upload(uploaded_file) -> np.ndarray:
    """Kiểm tra kích thước, định dạng magic-bytes và cấu trúc tệp ảnh trước khi giải mã RGB."""
    raw = uploaded_file.getvalue()
    if len(raw) == 0:
        raise ValueError("Tệp tải lên rỗng.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Kích thước tệp vượt giới hạn cho phép ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in ALLOWED_IMAGE_FORMATS:
                raise ValueError("Định dạng tệp không thuộc danh sách cho phép (chỉ nhận JPEG hoặc PNG hợp lệ).")
            probe.verify()
        with Image.open(io.BytesIO(raw)) as im:
            return np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Tệp ảnh không hợp lệ hoặc bị hỏng cấu trúc.") from exc


def load_safe_sample(sample_name: str) -> tuple[np.ndarray, str | None]:
    """Đọc ảnh mẫu an toàn chống Path Traversal và trích nhãn thật từ tiền tố tên file."""
    clean_name = Path(sample_name).name
    candidate = (SAMPLES_DIR / clean_name).resolve()
    if candidate.parent != SAMPLES_DIR or not candidate.is_file():
        raise ValueError("Đường dẫn ảnh mẫu không hợp lệ.")
    prefix = clean_name.split("_", 1)[0]
    true_dx = prefix if prefix in CLASS_NAMES else None
    with Image.open(candidate) as im:
        return np.asarray(im.convert("RGB")), true_dx


@st.cache_resource
def load_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    # Dùng weights_only=True để ngăn lỗ hổng giải tuần tự hóa bất an toàn (CWE-502)
    ck = torch.load(CKPT, map_location=dev, weights_only=True)
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval(), dev, ck.get("best_epoch")


def preprocess_tensor(rgb: np.ndarray, dev: torch.device) -> torch.Tensor:
    x = np.asarray(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)


@torch.no_grad()
def predict_probs(rgb: np.ndarray, model: nn.Module, dev: torch.device) -> np.ndarray:
    t = preprocess_tensor(rgb, dev)
    return torch.softmax(model(t).float(), 1)[0].cpu().numpy()


def compute_gradcam(rgb: np.ndarray, model: nn.Module, dev: torch.device, target_idx: int) -> np.ndarray:
    """Tính bản đồ nhiệt Grad-CAM tại tầng tích chập cuối `layer4` (7×7 → nội suy về kích thước ảnh)."""
    feats = []
    hook = model.layer4.register_forward_hook(lambda _m, _i, out: feats.append(out))
    try:
        t = preprocess_tensor(rgb, dev).requires_grad_(True)
        logits = model(t)
        grad = torch.autograd.grad(logits[0, target_idx], feats[0])[0]
        cam7 = F.relu((grad.mean(dim=(2, 3), keepdim=True) * feats[0]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam7, size=(rgb.shape[0], rgb.shape[1]), mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].detach().float().cpu().numpy()
        return cam_np / max(float(cam_np.max()), 1e-8)
    finally:
        hook.remove()


def overlay_cam(rgb: np.ndarray, cam_norm: np.ndarray) -> np.ndarray:
    hm = plt.cm.jet(cam_norm)[..., :3]
    blended = np.clip(0.56 * (rgb.astype(np.float32) / 255.0) + 0.44 * hm, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def qc_metrics(rgb: np.ndarray) -> dict:
    rec, _ = compute_from_rgb(rgb)
    return rec


def crop_to_aspect_4_3(rgb: np.ndarray) -> np.ndarray:
    """Cắt đồng tâm về đúng tỉ lệ khung hình 4:3 trước khi thu phóng về 600×450 để tránh méo hình học tổn thương."""
    h, w = rgb.shape[:2]
    target_ratio = SRC_SIZE[0] / SRC_SIZE[1]  # 4 / 3
    cur_ratio = w / max(h, 1)
    if abs(cur_ratio - target_ratio) < 1e-2:
        return rgb
    if cur_ratio > target_ratio:
        new_w = max(1, int(round(h * target_ratio)))
        x0 = (w - new_w) // 2
        return rgb[:, x0:x0 + new_w]
    new_h = max(1, int(round(w / target_ratio)))
    y0 = (h - new_h) // 2
    return rgb[y0:y0 + new_h, :]


def lab_delta_e(a: np.ndarray, b: np.ndarray) -> float:
    if np.array_equal(a, b):
        return 0.0
    la = cv2.cvtColor(a.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    lb = cv2.cvtColor(b.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return float(np.linalg.norm(la - lb, axis=2).mean())


def plot_hist_and_masks(orig: np.ndarray, proc: np.ndarray):
    info = analyze_rgb(orig)
    g0 = info["gray"]
    g1 = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
    fov = info["fov"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    # (1) Overlay mask viền + lông + biên Otsu trên ảnh gốc
    vis = orig.copy()
    vis[~fov] = (40, 40, 180)
    cnts, _ = cv2.findContours(info["lesion"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    vis[info["hair"]] = (255, 40, 40)
    axes[0].imshow(vis)
    axes[0].set_title("Mask EDA: đỏ=lông, vàng=Otsu thô, xanh=viền", fontsize=8.5)
    axes[0].axis("off")

    # (2) Bản đồ chênh lệch mức xám / màu |sau - trước|
    diff = np.abs(proc.astype(np.int16) - orig.astype(np.int16)).max(axis=2)
    vmax = max(20, int(np.percentile(diff, 99.5)))
    im_d = axes[1].imshow(diff, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title(f"Biên độ thay đổi pixel |sau − trước| (max={int(diff.max())})", fontsize=8.5)
    axes[1].axis("off")
    fig.colorbar(im_d, ax=axes[1], fraction=0.046, pad=0.03)

    # (3) Histogram mức xám trong FOV trước vs sau
    bins = np.arange(0, 257, 4)
    axes[2].hist(g0[fov], bins=bins, density=True, alpha=0.55, color="#7A7A7A", label="Trước xử lý (FOV)")
    axes[2].hist(g1[fov], bins=bins, density=True, alpha=0.55, color="#0072B2", label="Sau xử lý (FOV)")
    axes[2].set_xlim(0, 255)
    axes[2].set_xlabel("Mức xám (0–255)")
    axes[2].set_ylabel("Mật độ")
    axes[2].set_title("Phân bố mức xám trong vùng nhìn thấy", fontsize=8.5)
    axes[2].legend(fontsize=7.5)
    fig.tight_layout()
    return fig, info


# ---------------------------------------------------------------------------------------------------- Giao diện chính
st.set_page_config(page_title="Demo Xử lý ảnh & Phân loại HAM10000", layout="wide")
st.title("🔬 Demo: Đánh giá tác động Tiền xử lý ảnh lên Phân loại tổn thương da (HAM10000)")
st.warning(
    "**Lưu ý học thuật & y tế:** Ứng dụng chỉ phục vụ báo cáo học phần **Xử lý ảnh (PTIT)**, tuyệt đối **không dùng để "
    "chẩn đoán y khoa**. Mô hình được huấn luyện trên ảnh soi da chuyên dụng (*dermoscopy* HAM10000); ảnh chụp điện thoại "
    "thông thường lệch miền dữ liệu nên chỉ mang tính minh họa định tính các bước xử lý ảnh."
)

has_ckpt = CKPT.is_file()
model, dev, best_epoch = (load_model() if has_ckpt else (None, torch.device("cpu"), None))
if not has_ckpt:
    st.info(
        f"ℹ️ Chưa tìm thấy tệp trọng số CNN tại `{CKPT}`. Giao diện đang chạy ở **chế độ Kiểm tra chất lượng ảnh (QC), "
        "Xử lý ảnh cổ điển & Bảng tổng hợp 3 seed**. Để bật suy luận ResNet-18 & Grad-CAM, hãy đặt `baseline_seed42.pt` "
        "vào `outputs/checkpoints/` (hoặc chạy qua Docker image đã đóng gói sẵn)."
    )

with st.sidebar:
    st.header("1. Nguồn ảnh đầu vào")
    up = st.file_uploader("Tải ảnh (JPG/PNG, tối đa 10 MB)", type=["jpg", "jpeg", "png"])
    keep_ratio_upload = st.checkbox("Giữ tỉ lệ khung 4:3 (cắt tâm chống méo hình khi tải ảnh ngoài)", value=True)
    samples = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
    default_idx = 1 if samples else 0
    pick = st.selectbox("…hoặc chọn ảnh mẫu tập test (`samples/`)", ["(không)"] + [p.name for p in samples], index=default_idx)

    st.divider()
    st.header("2. Chuỗi tiền xử lý minh họa")
    preset_key = st.selectbox("Chọn cấu hình mẫu", list(PRESETS.keys()), index=0)
    all_opts = [c for c in CANDIDATES if c != "none"]
    if PRESETS[preset_key] is None:
        steps = st.multiselect("Chọn các bước áp dụng tuần tự (trái → phải)", all_opts, default=["vig_fill", "flat_L_s60_cond"])
    else:
        steps = st.multiselect("Các bước đang bật (có thể chỉnh sửa)", all_opts, default=PRESETS[preset_key])

    st.divider()
    st.header("3. Mô phỏng suy giảm ảnh (2.5c)")
    deg_kind = st.selectbox("Loại suy giảm giả lập", list(DEGRADE_LABELS.keys()), format_func=lambda k: DEGRADE_LABELS[k])
    deg_level = 0
    if deg_kind != "none":
        deg_level = st.slider(
            f"Mức độ suy giảm ({LEVELS[deg_kind][0]} / {LEVELS[deg_kind][1]} / {LEVELS[deg_kind][2]})",
            min_value=1, max_value=3, value=2,
        )

    st.divider()
    if has_ckpt:
        st.caption(f"**Mô hình nhánh (B):** ResNet-18 baseline · seed 42 · epoch tốt nhất {best_epoch} · thiết bị `{dev.type}`.")
    else:
        st.caption("**Mô hình nhánh (B):** Chưa nạp checkpoint (`baseline_seed42.pt`).")
    st.caption(
        "**Nhóm thực hiện (PTIT):** Nguyễn Trung Nam (`B23DCVT299`) · Phan Nguyễn Việt Dũng (`B23DCCN210`) · "
        "Lê Duy Anh (`B23DCCN026`) · Đinh Khánh Sơn (`B23DCDT223`) · Đào Phạm Bích Ngọc (`B23DCVT308`)."
    )

true_dx = None
source_id = "uploaded_image"
if up is not None:
    try:
        rgb_raw = validate_and_decode_upload(up)
        source_id = "user_upload"
    except ValueError as err:
        st.error(str(err))
        st.stop()
elif pick != "(không)":
    try:
        rgb_raw, true_dx = load_safe_sample(pick)
        source_id = Path(pick).stem
    except ValueError as err:
        st.error(str(err))
        st.stop()
else:
    st.info("👈 Vui lòng tải lên một ảnh hợp lệ hoặc chọn một trong 7 ảnh mẫu từ thanh công cụ bên trái.")
    st.stop()

if rgb_raw.shape[1::-1] != SRC_SIZE:
    rgb_src = crop_to_aspect_4_3(rgb_raw) if (up is not None and keep_ratio_upload) else rgb_raw
    crop_note = " (cắt đồng tâm 4:3 trước khi thu phóng)" if (up is not None and keep_ratio_upload) else ""
    st.caption(
        f"ℹ️ Ảnh đầu vào kích thước `{rgb_raw.shape[1]}×{rgb_raw.shape[0]}` được đưa về chuẩn `{SRC_SIZE[0]}×{SRC_SIZE[1]}`"
        f"{crop_note} của HAM10000 để bộ đo QC và tham số hình thái học vận hành đúng tỉ lệ thiết kế."
    )
    rgb = np.asarray(Image.fromarray(rgb_src).resize(SRC_SIZE, Image.BILINEAR))
else:
    rgb = rgb_raw

if deg_kind != "none" and deg_level > 0:
    rgb = degrade(rgb, deg_kind, deg_level, source_id)
    st.info(
        f"🧪 **Đang bật chế độ Stress Test (mục 2.5c):** Ảnh gốc đã được cộng suy giảm tổng hợp "
        f"`{deg_kind}` mức {deg_level} (tham số = `{LEVELS[deg_kind][deg_level - 1]}`)."
    )

# Chạy chuỗi tiền xử lý minh họa
stages = [("Ảnh đầu vào", rgb)]
cur = rgb
for s in steps:
    cur = get_preprocess(s)(cur)
    stages.append((s, cur))

m_before = qc_metrics(rgb)
m_after = qc_metrics(cur) if not np.array_equal(cur, rgb) else m_before
p0 = predict_probs(rgb, model, dev) if has_ckpt else None
p1 = predict_probs(cur, model, dev) if has_ckpt else None

tab_pipe, tab_stages, tab_cam = st.tabs([
    "📊 1. Pipeline Đề xuất: QC & Dự đoán (2.6–2.8)",
    "🖼️ 2. Từng giai đoạn Xử lý ảnh & Histogram (2.5)",
    "🎯 3. Giải thích Grad-CAM & Tổng hợp Thực nghiệm",
])

# ============================================================================== TAB 1
with tab_pipe:
    if true_dx is not None and p0 is not None:
        pred_main = CLASSES[int(p0.argmax())]
        match_txt = "✅ Khớp nhãn thật" if pred_main == true_dx else "❌ Lệch nhãn thật"
        st.success(f"**Ảnh mẫu từ tập Test:** `{pick}` · **Nhãn thật:** `{true_dx}` – {CLASS_NAMES[true_dx]} · **{match_txt}**")
    elif true_dx is not None:
        st.success(f"**Ảnh mẫu từ tập Test:** `{pick}` · **Nhãn thật:** `{true_dx}` – {CLASS_NAMES[true_dx]}")

    st.subheader("(A) Nhánh Kiểm tra Chất lượng ảnh (QC – Bộ phát hiện EDA mục 2.2)")
    qc_rows = []
    n_flags = 0
    for name, key, rule_fn, thr_txt in QC_RULES:
        val_b = float(m_before[key]) if pd.notna(m_before[key]) else np.nan
        val_a = float(m_after[key]) if pd.notna(m_after[key]) else np.nan
        flagged = bool(pd.notna(val_b) and rule_fn(m_before))
        n_flags += int(flagged)
        qc_rows.append({
            "Vấn đề chất lượng": name,
            "Chỉ số đo": key,
            "Giá trị (đầu vào)": round(val_b, 4) if pd.notna(val_b) else None,
            "Ngưỡng cờ EDA": thr_txt,
            "Trạng thái QC": "⚠️ Vượt ngưỡng" if flagged else "✅ Bình thường",
            "Sau bước minh họa": round(val_a, 4) if pd.notna(val_a) else None,
        })
    st.dataframe(pd.DataFrame(qc_rows), hide_index=True, width="stretch")
    st.caption(
        f"Tổng số cờ QC phát hiện trên ảnh đầu vào: **{n_flags}/6**. Theo kết quả mục 2.6 & 2.8, nhánh QC dùng để **cảnh báo "
        "chất lượng ảnh** (đặc biệt nhóm *sáng không đều* và *quá sáng* có tỉ lệ lỗi vượt mức kỳ vọng theo lớp), "
        "không tự động biến đổi ảnh đưa vào CNN vì 0/16 cấu hình tiền xử lý vượt chuẩn cải thiện rõ trên tập validation."
    )

    st.subheader("(B) Dự đoán của ResNet-18 Baseline")
    if p0 is not None and p1 is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            top0 = CLASSES[int(p0.argmax())]
            st.markdown(f"#### Kết quả chính – Trên ảnh đầu vào: `{top0}` ({p0.max():.1%})")
            st.caption(CLASS_NAMES[top0])
            df_p0 = pd.DataFrame({"Lớp": CLASSES, "Xác suất": p0}).set_index("Lớp")
            st.bar_chart(df_p0)
        with col_b:
            top1 = CLASSES[int(p1.argmax())]
            same_img = np.array_equal(cur, rgb)
            tag_cmp = "không đổi pixel → trùng kết quả chính" if same_img else f"`{top1}` ({p1.max():.1%})"
            st.markdown(f"#### Đối chiếu – Trên ảnh sau xử lý minh họa: {tag_cmp}")
            st.caption(CLASS_NAMES[top1])
            df_p1 = pd.DataFrame({"Lớp": CLASSES, "Xác suất": p1}).set_index("Lớp")
            st.bar_chart(df_p1)

        comp_df = pd.DataFrame({
            "Mã lớp": CLASSES,
            "Chẩn đoán": [CLASS_NAMES[c] for c in CLASSES],
            "P(ảnh đầu vào)": [round(float(x), 4) for x in p0],
            "P(sau xử lý)": [round(float(x), 4) for x in p1],
            "Δ xác suất": [round(float(b - a), 4) for a, b in zip(p0, p1)],
        }).sort_values("P(ảnh đầu vào)", ascending=False).reset_index(drop=True)
        with st.expander("Xem bảng chi tiết xác suất 7 lớp (trước & sau xử lý)"):
            st.dataframe(comp_df, hide_index=True, width="stretch")
            st.caption(
                "Lưu ý phương pháp luận: checkpoint baseline được huấn luyện trên ảnh gốc không xử lý. Phép so sánh công bằng "
                "(huấn luyện lại mô hình trên chính ảnh đã xử lý với 3 seed 42, 43, 44) được trình bày ở Tab 3 và mục 2.5 của báo cáo."
            )
    else:
        st.warning("Chưa nạp tệp `baseline_seed42.pt` nên tạm ẩn phần dự đoán xác suất CNN.")

# ============================================================================== TAB 2
with tab_stages:
    st.subheader("Chuỗi ảnh qua từng bước tiền xử lý")
    cols = st.columns(min(len(stages), 4))
    for i, (st_name, img_st) in enumerate(stages):
        with cols[i % len(cols)]:
            if i == 0:
                cap = st_name
            else:
                unchanged = np.array_equal(img_st, stages[i - 1][1])
                de = lab_delta_e(stages[i - 1][1], img_st)
                cap = f"{i}. {st_name}" + (" (giữ nguyên – không vượt ngưỡng cờ)" if unchanged else f" (ΔE76 = {de:.2f})")
            st.image(img_st, caption=cap, width="stretch")
            if st_name in STEP_NOTES:
                st.caption(STEP_NOTES[st_name])

    st.subheader("Chẩn đoán thị giác: Mask hình thái học, Bản đồ thay đổi pixel & Histogram")
    fig_diag, mask_info = plot_hist_and_masks(rgb, cur)
    st.pyplot(fig_diag, width="stretch")
    plt.close(fig_diag)

    c_m1, c_m2, c_m3 = st.columns(3)
    pct_changed = float((cur != rgb).any(axis=2).mean() * 100.0)
    c_m1.metric("Tỉ lệ pixel bị thay đổi", f"{pct_changed:.1f}%")
    c_m2.metric("Độ lệch không gian Lab trung bình ΔE76", f"{lab_delta_e(rgb, cur):.2f}")
    c_m3.metric("Diện tích vùng nhìn thấy (FOV)", f"{float(mask_info['fov'].mean() * 100.0):.1f}%")

# ============================================================================== TAB 3
with tab_cam:
    st.subheader("Giải thích vùng chú ý của mô hình bằng Grad-CAM (`layer4`)")
    if has_ckpt and p0 is not None:
        default_cls_idx = CLASSES.index(true_dx) if true_dx in CLASSES else int(p0.argmax())
        target_cls = st.selectbox(
            "Chọn lớp đích để tính đạo hàm Grad-CAM",
            CLASSES,
            index=default_cls_idx,
            format_func=lambda c: f"{c} – {CLASS_NAMES[c]}",
        )
        t_idx = CLASSES.index(target_cls)
        cam0 = compute_gradcam(rgb, model, dev, t_idx)
        cam1 = compute_gradcam(cur, model, dev, t_idx)

        les_mask = mask_info["lesion"]
        hair_dil = cv2.dilate(mask_info["hair"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        e_les0 = float(cam0[les_mask].sum() / max(float(cam0.sum()), 1e-8) * 100.0)
        e_les1 = float(cam1[les_mask].sum() / max(float(cam1.sum()), 1e-8) * 100.0)
        e_hair0 = float(cam0[hair_dil].sum() / max(float(cam0.sum()), 1e-8) * 100.0)
        e_hair1 = float(cam1[hair_dil].sum() / max(float(cam1.sum()), 1e-8) * 100.0)

        g1, g2 = st.columns(2)
        with g1:
            st.image(
                overlay_cam(rgb, cam0),
                caption=f"Grad-CAM trên ảnh đầu vào (đích `{target_cls}`) · Năng lượng trong tổn thương: {e_les0:.1f}%, trên lông: {e_hair0:.1f}%",
                width="stretch",
            )
        with g2:
            st.image(
                overlay_cam(cur, cam1),
                caption=f"Grad-CAM trên ảnh sau xử lý (đích `{target_cls}`) · Năng lượng trong tổn thương: {e_les1:.1f}%, trên lông: {e_hair1:.1f}%",
                width="stretch",
            )
        st.caption(
            "Grad-CAM tính tại `model.layer4` (lưới đặc trưng 7×7 phóng nội suy về 600×450), thể hiện vùng ảnh đóng góp dương "
            "vào logit của lớp đích đã chọn."
        )
    else:
        st.info("Cần tệp `baseline_seed42.pt` để tính trực tiếp bản đồ nhiệt Grad-CAM.")

    st.divider()
    st.subheader("Tra cứu nhanh Kết quả Thực nghiệm 3 Seed của Dự án (NVIDIA L40S)")
    bb_csv = resolve_table("outputs/backbones/comparison_test.csv", "backbones_comparison_test.csv")
    all16_val_csv = resolve_table("outputs/preprocess/all16_comparison_val.csv", "all16_comparison_val.csv")
    all16_test_csv = resolve_table("outputs/preprocess/all16_comparison_test.csv", "all16_comparison_test.csv")
    if bb_csv is not None:
        st.markdown("**So sánh 3 kiến trúc CNN trên tập Test (3 seed: 42, 43, 44):**")
        st.dataframe(pd.read_csv(bb_csv).round(4), hide_index=True, width="stretch")
    if all16_val_csv is not None and all16_test_csv is not None:
        split_pick = st.radio("Chọn tập xem bảng tổng hợp 16 cấu hình tiền xử lý:", ["Validation (chọn mô hình)", "Test (đánh giá cuối)"], horizontal=True)
        chosen_csv = all16_val_csv if split_pick.startswith("Validation") else all16_test_csv
        df_16 = pd.read_csv(chosen_csv)
        cols_show = [c for c in ["config", "macro_f1_mean", "macro_f1_std", "macro_f1_delta",
                                 "macro_f1_delta_ci_lo", "macro_f1_delta_ci_hi", "macro_f1_delta_sign", "macro_f1_verdict"]
                     if c in df_16.columns]
        st.dataframe(df_16[cols_show].round(4), hide_index=True, width="stretch")
