"""Xử lý 4 vấn đề còn lại của bảng kết luận EDA: lông, sáng không đều, quá sáng (bạc màu), viền tối ống soi.

Chương dùng: Ch.3 (biến đổi lũy thừa/gamma, lọc không gian Gauss, khôi phục ảnh), Ch.4 (tách độ sáng L* khỏi màu),
Ch.7 (black-hat = đóng − ảnh, phép mở, giãn). Inpainting (cv2.inpaint, Telea 2004) KHÔNG có trong slide: dùng
như bước "khôi phục ảnh" (Ch.3) cho vùng bị che, ghi rõ trong báo cáo là mở rộng.

Quy ước giống noise.py/contrast.py: hàm thuần, vào/ra uint8 RGB (H, W, 3) cùng shape, áp dụng ở 600×450 TRƯỚC resize
và Normalize, giống nhau cho train/val/test. Bộ phát hiện (lông, viền, trắng bạc) dùng ĐÚNG hàm/tham số của EDA
(src.eda.quality_metrics) để vấn đề được xử lý truy ngược được về dòng tương ứng của bảng EDA.

Hai kiểu áp dụng (đây là biến được khảo sát, xem 2.5 mới):
  - "toàn bộ": áp dụng cho mọi ảnh (như median3/CLAHE trước đây);
  - "có điều kiện" (_cond / bản chất có điều kiện): chỉ sửa ảnh mà bộ phát hiện của EDA gắn cờ, ảnh khác giữ
    nguyên từng pixel. Lý do: mỗi vấn đề chỉ có 2,5–10% ảnh; phép toàn cục làm lệch 90%+ ảnh bình thường
    (bài học từ stretch/hist_eq ở 2.5.2).
Mọi tham số hoặc cố định trước, hoặc là thống kê của chính ảnh đó; hằng số duy nhất lấy từ dữ liệu
(L_TARGET) tính trên train_fit (scripts/artifact_constants.py) → không rò rỉ val/test.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.eda.flags import VIG_T
from src.eda.quality_metrics import BRIGHT_S, BRIGHT_V, blackhat, lesion_mask, vignette_mask

from .noise import _check

# ---- hằng số ----
HAIR_FRAC_T = 0.0582   # ngưỡng cờ "lông" của EDA (Q3 + 1.5·IQR của hair_frac, outputs/eda/table_flag_rules)
BRIGHT_FRAC_T = 0.10   # ngưỡng cờ "quá sáng" của EDA (> 10% FOV là pixel trắng bạc)
L_TARGET = 66.61       # trung vị (qua ảnh) của trung vị L* trong FOV, ảnh train_fit KHÔNG bị cờ quá sáng
                       # (scripts/artifact_constants.py) → đích của gamma tự động


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _fov(gray):
    fov = ~vignette_mask(gray)
    return fov if fov.sum() >= 1000 else np.ones(gray.shape, bool)


def _to_lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)       # L 0..100


def _from_lab(lab):
    return np.clip(np.rint(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0), 0, 255).astype(np.uint8)


# =====================================================================================================
# 1. LÔNG: black-hat (Ch.7) → ngưỡng → bỏ đốm nhỏ → giãn → inpainting (khôi phục vùng bị che)
# =====================================================================================================
def hair_mask_full(rgb: np.ndarray, k: int = 17, thr: int = 20, min_area: int = 60, dilate: int = 3) -> np.ndarray:
    """Mask lông: giống bộ đo EDA (black-hat k×k elip ≥ thr, bỏ thành phần < min_area px, trong FOV) rồi giãn
    dilate×dilate để phủ cả viền mờ của sợi lông (nếu không, inpainting lấy màu từ chính viền lông tối)."""
    g = _gray(rgb)
    m = ((blackhat(g, k) >= thr) & _fov(g)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    m = keep[lab].astype(np.uint8)
    if dilate > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return m


def remove_hair(rgb: np.ndarray, k: int = 17, thr: int = 20, radius: int = 5, conditional: bool = False) -> np.ndarray:
    """Xóa lông kiểu DullRazor (Lee 1997): tìm nét tối mảnh bằng black-hat, tô lại bằng inpainting Telea.
    k: phần tử cấu trúc phải rộng hơn sợi lông (2–10 px) để phép đóng lấp được lông; k lớn hơn → bắt cả cấu trúc
       sắc tố dạng nét (mạng sắc tố, vệt) trong tổn thương → rủi ro xóa tín hiệu chẩn đoán (giả thuyết H-H2).
    conditional=True: chỉ xử lý ảnh có tỉ lệ pixel lông (trước giãn) > ngưỡng cờ EDA."""
    _check(rgb)
    m = hair_mask_full(rgb, k=k, thr=thr, dilate=1)
    if conditional and m.sum() / max(_fov(_gray(rgb)).sum(), 1) <= HAIR_FRAC_T:
        return rgb
    if m.sum() == 0:
        return rgb
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.inpaint(rgb, m, radius, cv2.INPAINT_TELEA)


# =====================================================================================================
# 2. VIỀN TỐI ỐNG SOI (vốn có điều kiện: ảnh không có viền → giữ nguyên)
# =====================================================================================================
def _vig(rgb):
    g = _gray(rgb)
    v = vignette_mask(g)
    return g, v, v.mean() > VIG_T


def vignette_fill(rgb: np.ndarray, ring: int = 25, feather: int = 21) -> np.ndarray:
    """Tô viền đen bằng màu da: màu = trung vị RGB của dải FOV rộng `ring` px sát viền (da gần viền), biên
    chuyển tiếp làm mềm bằng lọc Gauss feather×feather trên mask (Ch.3) để không tạo cạnh tròn mới.
    Không đổi tỉ lệ/vị trí tổn thương (khác vignette_crop)."""
    _check(rgb)
    g, v, has = _vig(rgb)
    if not has:
        return rgb
    vm = v.astype(np.uint8)
    band = cv2.dilate(vm, np.ones((2 * ring + 1,) * 2, np.uint8)).astype(bool) & ~v
    if band.sum() < 100:
        return rgb
    color = np.median(rgb[band], axis=0)
    a = cv2.GaussianBlur(cv2.dilate(vm, np.ones((5, 5), np.uint8)).astype(np.float32), (feather, feather), 0)[..., None]
    out = rgb.astype(np.float32) * (1 - a) + color[None, None, :] * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def vignette_crop(rgb: np.ndarray, max_dark: float = 0.005, step: int = 4) -> np.ndarray:
    """Cắt bỏ viền: thu nhỏ dần khung chữ nhật đồng tâm, cùng tỉ lệ 4:3 với ảnh, đến khi phần viền đen trong khung
    ≤ max_dark; cắt rồi phóng lại về kích thước gốc (bilinear) để giữ quy ước cùng shape.
    Đánh đổi: tổn thương bị phóng to (đổi tỉ lệ so với ảnh không viền) và mất phần da ở góc."""
    _check(rgb)
    g, v, has = _vig(rgb)
    if not has:
        return rgb
    h, w = v.shape
    ii = cv2.integral(v.astype(np.uint8))
    for d in range(0, min(h, w) // 2 - 20, step):
        y0, x0 = d, int(round(d * w / h))
        y1, x1 = h - y0, w - x0
        area = (y1 - y0) * (x1 - x0)
        dark = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
        if dark / area <= max_dark:
            return cv2.resize(rgb[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb


# =====================================================================================================
# 3. QUÁ SÁNG / BẠC MÀU: biến đổi lũy thừa s = r^γ, γ > 1 (Ch.3) trên L* (Ch.4: giữ a*, b*)
# =====================================================================================================
def _bright_frac(rgb, fov):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    b = (hsv[..., 2] >= BRIGHT_V) & (hsv[..., 1] <= BRIGHT_S) & fov
    return b.sum() / max(fov.sum(), 1)


def _gamma_L(rgb, gamma):
    lab = _to_lab(rgb)
    lab[..., 0] = 100.0 * np.power(np.clip(lab[..., 0] / 100.0, 0, 1), gamma)
    return _from_lab(lab)


def gamma_fixed(rgb: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Gamma cố định cho MỌI ảnh (đối chứng "toàn cục"): γ = 1,5 làm tối mọi ảnh, mạnh nhất ở dải sáng giữa.
    Không đổi a*, b*: vùng bạc màu (L* cao, chroma thấp) tối đi nên không còn "trắng bạc"."""
    _check(rgb)
    return _gamma_L(rgb, gamma)


def gamma_auto(rgb: np.ndarray, target: float = L_TARGET, gmax: float = 3.0) -> np.ndarray:
    """Gamma CÓ ĐIỀU KIỆN: chỉ ảnh bị cờ quá sáng (bright_frac > 10%, bộ đo EDA). γ chọn để trung vị L* của FOV
    về đúng `target` (mức của ảnh bình thường trong train_fit): γ = ln(t)/ln(m), m, t ∈ (0,1); kẹp [1, gmax]
    để chỉ làm tối, không bao giờ làm sáng."""
    _check(rgb)
    fov = _fov(_gray(rgb))
    if _bright_frac(rgb, fov) <= BRIGHT_FRAC_T:
        return rgb
    lab = _to_lab(rgb)
    m = float(np.median(lab[..., 0][fov])) / 100.0
    if not 0 < m < 1:
        return rgb
    gamma = float(np.clip(np.log(target / 100.0) / np.log(m), 1.0, gmax))
    lab[..., 0] = 100.0 * np.power(np.clip(lab[..., 0] / 100.0, 0, 1), gamma)
    return _from_lab(lab)


# =====================================================================================================
# 4. SÁNG KHÔNG ĐỀU: hiệu chỉnh nền (flat-field) L' = L · mean(B) / B, B = nền ước lượng bằng lọc Gauss lớn (Ch.3)
# =====================================================================================================
def _norm_blur(x, w, sigma):
    """Lọc Gauss chuẩn hóa (normalized convolution): B = G*(x·w) / G*(w) – chỉ lấy trung bình trên pixel w=1
    (vd. chỉ da), không để viền đen hay tổn thương tối kéo nền xuống. Tính ở 1/4 độ phân giải cho nhanh
    (nền là thành phần tần số rất thấp nên không mất thông tin)."""
    h, wd = x.shape
    s = 4
    xs = cv2.resize(x * w, (wd // s, h // s), interpolation=cv2.INTER_AREA)
    ws = cv2.resize(w, (wd // s, h // s), interpolation=cv2.INTER_AREA)
    k = sigma / s
    num = cv2.GaussianBlur(xs, (0, 0), k, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(ws, (0, 0), k, borderType=cv2.BORDER_REFLECT)
    b = num / np.maximum(den, 1e-3)
    return cv2.resize(b, (wd, h), interpolation=cv2.INTER_LINEAR)


def flatfield(rgb: np.ndarray, sigma: float = 60.0, skin_only: bool = True, conditional: bool = False,
              illum_t: float = 38.0, max_gain: float = 2.0) -> np.ndarray:
    """Chia cho nền: L' = L · median(B_fov) / B trên L* (giữ a*, b*), pixel viền đen giữ nguyên.
    sigma = 60 px (≈ 1/8 cạnh dài 600): đủ lớn để B không chứa chi tiết tổn thương, đủ nhỏ để theo được độ dốc
    sáng tối dần về một góc. skin_only: ước lượng nền chỉ trên da (bỏ tổn thương bằng mask Otsu thô của EDA, giãn
    31 px) – nếu không, tổn thương lớn tối bị coi là "nền tối" và bị làm sáng lên (giảm tương phản tổn thương).
    conditional: chỉ xử lý khi chênh nền (illum_range của EDA) > ngưỡng cờ 38 mức xám.
    max_gain: kẹp hệ số khuếch đại để vùng rất tối sát viền không bị đẩy lên quá mức."""
    _check(rgb)
    g = _gray(rgb)
    fov = _fov(g)
    if conditional:
        from src.eda.quality_metrics import illum_range, masks
        M = masks(g)
        r = illum_range(g, M["skin"] & (g > M["otsu_t"]))
        if not (np.isfinite(r) and r > illum_t) or (1 - fov.mean()) > VIG_T:
            return rgb
    w = fov.astype(np.float32)
    if skin_only:
        les, _ = lesion_mask(g, fov)
        les = cv2.dilate(les.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        ws = (fov & ~les).astype(np.float32)
        if ws.sum() > 0.1 * fov.sum():
            w = ws
    lab = _to_lab(rgb)
    L = lab[..., 0]
    B = _norm_blur(L, w, sigma)
    ref = float(np.median(B[fov]))
    gain = np.clip(ref / np.maximum(B, 1e-3), 1.0 / max_gain, max_gain)
    Ln = np.where(fov, np.clip(L * gain, 0, 100), L)
    lab[..., 0] = Ln
    return _from_lab(lab)
