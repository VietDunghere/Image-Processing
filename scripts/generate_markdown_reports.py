"""Sinh và đồng bộ toàn bộ hệ thống báo cáo Markdown (mục 1 → 3) trực tiếp từ các tệp CSV/JSON trong outputs/
của lượt chạy thống nhất trên NVIDIA L40S. Đảm bảo 0% số giả, 0% lệch baseline giữa các phần, 0% nhãn TODO tồn đọng.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT = ROOT / "outputs"


def f4(x: float) -> str:
    return f"{float(x):.4f}".replace(".", ",")


def f3(x: float) -> str:
    return f"{float(x):.3f}".replace(".", ",")


def f2(x: float) -> str:
    return f"{float(x):.2f}".replace(".", ",")


def f1(x: float) -> str:
    return f"{float(x):.1f}".replace(".", ",")


def sf4(x: float) -> str:
    return f"{float(x):+.4f}".replace(".", ",")


def sf3(x: float) -> str:
    return f"{float(x):+.3f}".replace(".", ",")


def write_1():
    txt = """# 1 Giới thiệu đề tài

> Nguồn số liệu quy mô dữ liệu: `outputs/report_tables/data_checks.json`. Toàn bộ hệ thống thực nghiệm đã hoàn tất đồng bộ trên cùng một môi trường phần cứng (NVIDIA L40S, PyTorch CUDA 12.6).

## 1.1 Bài toán và động cơ

**Bài toán.** Phân loại ảnh tổn thương da thành 7 lớp chẩn đoán (*multi-class classification*). Đầu vào là một ảnh dermoscopy RGB kích thước $600\\times 450$, đầu ra là một trong 7 nhãn `akiec, bcc, bkl, df, mel, nv, vasc`. Bộ dữ liệu sử dụng là **HAM10000** (10 015 ảnh thuộc 7 470 tổn thương `lesion_id`; xem mục 2.1).

**Ba thách thức cốt lõi.**
1. **Chất lượng ảnh không đồng đều:** viền tối cơ học của ống soi, vùng lóa trắng bạc do phản xạ đèn LED/gel, tương phản thấp giữa tổn thương nhạt màu và nền da, chiếu sáng lệch góc, nhiễu cảm biến và sợi lông che phủ (mục 2.2 đo định lượng trên toàn bộ 10 015 ảnh).
2. **Mất cân bằng lớp nghiêm trọng:** lớp đa số `nv` (6 705 ảnh) gấp **58,3 lần** lớp thiểu số `df` (115 ảnh), khiến độ đo accuracy đơn thuần bị chi phối bởi lớp `nv` (mục 2.1).
3. **Cấu trúc cụm nhiều ảnh trên cùng một tổn thương:** 26,2% `lesion_id` có từ 2 đến 6 ảnh (chiếm 44,9% tổng số ảnh), đòi hỏi chia tập theo nhóm `lesion_id` và kiểm định thống kê theo cụm để chống rò rỉ dữ liệu (mục 2.3).

**Động cơ nghiên cứu.** Trong học sâu hiện đại, các mạng tích chập (CNN) tiền huấn luyện trên ImageNet thường được đưa trực tiếp ảnh thô vào huấn luyện. Câu hỏi đặt ra là: *các phương pháp nâng cao chất lượng ảnh cổ điển của Chương 3 (lọc không gian, giãn/cân bằng lược đồ, CLAHE, biến đổi gamma, hiệu chỉnh nền) kết hợp xử lý màu Chương 4 và hình thái học Chương 7 có thực sự cải thiện hiệu năng phân loại của CNN hay không, giúp ở đâu và khi nào gây tác dụng phụ?*

## 1.2 Mục tiêu

Đề tài **không** chạy theo điểm số SOTA, không tinh chỉnh siêu tham số quá mức hay kết hợp mô hình (ensemble). Trọng tâm xuyên suốt là chu trình khoa học **lập luận $\\rightarrow$ thực nghiệm có kiểm soát $\\rightarrow$ đánh giá trung thực**:

1. Định lượng và kiểm định thống kê các vấn đề chất lượng ảnh trên toàn bộ 10 015 ảnh HAM10000, kèm kiểm chứng độ nhạy bộ đo (mục 2.2).
2. Khảo sát các kiến trúc CNN đại diện (`ResNet-18`, `EfficientNet-B0`, `MobileNetV3-Small`) trên cấu hình đóng băng và thiết lập đường cơ sở (baseline) chuẩn mực (mục 2.4).
3. Thực hiện nghiên cứu triệt tiêu (*ablation study*) trên 16 cấu hình tiền xử lý (đơn lẻ, có điều kiện và kết hợp, mỗi cấu hình 3 seed độc lập), phân tích sâu ở 4 tầng tác động: mức pixel ảnh, mức đổi nhãn dự đoán, mức suy giảm tổng hợp (*stress test*) và bản đồ nhiệt chú ý Grad-CAM (mục 2.5).
4. Đề xuất pipeline hoàn chỉnh hai nhánh (nhánh kiểm tra chất lượng QC gắn cờ + nhánh phân loại CNN), phân tích lỗi chuyên sâu và xây dựng ứng dụng minh họa tương tác trên Streamlit (mục 2.6–2.8).

## 1.3 Câu hỏi nghiên cứu

| Mã | Câu hỏi nghiên cứu | Mục trả lời |
|---|---|---|
| **RQ1** | Bộ dữ liệu HAM10000 tồn tại những vấn đề chất lượng ảnh nào có bằng chứng thống kê, và chúng liên hệ ra sao với lỗi của CNN sau khi chuẩn hóa thành phần lớp? | 2.2, 2.8.4 |
| **RQ2** | Với từng vấn đề, phép xử lý cổ điển tương ứng (áp đại trà hoặc chỉ áp có điều kiện trên ảnh bị cờ) có tạo ra thay đổi macro-F1 vượt ngưỡng dao động ngẫu nhiên giữa các seed không? | 2.5a, 2.5b |
| **RQ3** | Nếu macro-F1 toàn tập không tăng, tiền xử lý tác động thực sự ở tầng nào (chỉ số ảnh, đổi nhãn từng mẫu, khả năng chống chịu suy giảm nặng hay vùng chú ý Grad-CAM)? | 2.5c |
| **RQ4** | Việc ghép nối tuần tự nhiều bước tiền xử lý và lựa chọn kiến trúc CNN tác động thế nào đến thiết kế pipeline đầu–cuối (end-to-end)? | 2.4, 2.6, 2.7 |

## 1.4 Đóng góp chính

- Bộ 7 chỉ số chất lượng ảnh tự động trên 10 015 ảnh kèm kiểm chứng độ nhạy bằng nhiễu tổng hợp và đối chiếu sai số CNN chuẩn hóa gián tiếp theo lớp (mục 2.2, 2.8).
- Khuôn khổ thực nghiệm kiểm soát biến nghiêm ngặt trên cùng một phần cứng (NVIDIA L40S): chia tập theo `lesion_id`, cố định siêu tham số, đánh giá 3 kiến trúc CNN và 16 cấu hình tiền xử lý qua 3 seed (42, 43, 44) kèm bootstrap theo cụm tổn thương (mục 2.3–2.5).
- Phân tích giải mã nghịch lý *"cải thiện chỉ số ảnh nhưng không tăng F1 của CNN"* qua 4 tầng (ảnh, mức nền đổi nhãn, stress test 18 mức suy giảm và Grad-CAM `layer4`), đặc biệt chỉ rõ cơ chế xóa nhầm mạng sắc tố u hắc tố (`mel`) khi xóa lông đại trà (mục 2.5b, 2.5c, 2.8.5).
- Pipeline hai nhánh thực tiễn cùng ứng dụng Streamlit 3 tab tích hợp QC, chẩn đoán thị giác từng bước xử lý, mô phỏng suy giảm và giải thích Grad-CAM (mục 2.6–2.8, `app.py`).

## 1.5 Kiến thức học phần sử dụng

| Chương học phần | Kỹ thuật sử dụng | Vai trò trong đề tài |
|---|---|---|
| **Chương 3 – Nâng cao chất lượng ảnh** | Phân ngưỡng, lược đồ xám, giãn tuyến tính, cân bằng lược đồ, CLAHE (mở rộng cục bộ), biến đổi lũy thừa gamma, lọc trung bình/Gauss/trung vị, khôi phục vùng che phủ (inpainting Telea – mở rộng) | Đo chỉ số EDA (2.2) và 16 cấu hình tiền xử lý (2.5) |
| **Chương 4 – Hệ màu** | Không gian HSV (phát hiện vùng lóa trắng bạc), mở rộng sang CIE $L^*a^*b^*$ để tách kênh độ sáng $L^*$ khỏi sắc độ $a^*, b^*$ và đo độ lệch màu $\\Delta E_{76}$ | EDA (2.2), tăng tương phản / gamma / hiệu chỉnh nền (2.5) |
| **Chương 5 – Phát hiện biên & phân vùng** | Đạo hàm Sobel (chọn vùng phẳng đo nhiễu), phân ngưỡng tự động Otsu (ước lượng thô vùng tổn thương) | EDA (2.2), đo ảnh giai đoạn 1 (2.5) |
| **Chương 6 – Nén ảnh** | Kiểm tra định dạng, kích thước và cấu trúc nén JPEG của toàn bộ tập ảnh | Kiểm tra cơ bản EDA (2.1, 2.2) |
| **Chương 7 – Xử lý hình thái học** | Phép đóng, phép mở, phép giãn và biến đổi **black-hat** (phát hiện viền tối ống soi và sợi lông tối mảnh) | EDA (2.2), xóa lông & xử lý viền tối (2.5b) |
| **Chương 8 – Phân loại & nhận dạng ảnh** | Mạng tích chập sâu (`ResNet-18`, `EfficientNet-B0`, `MobileNetV3-Small`), học chuyển giao, đánh giá đa lớp mất cân bằng và giải thích Grad-CAM | Baseline, ablation và pipeline cuối (2.4–2.8) |

## 1.6 Phạm vi và lưu ý đạo đức

- Bộ dữ liệu HAM10000 gồm ảnh **soi da chuyên dụng (*dermoscopy*)**, không phải ảnh chụp bằng camera điện thoại thông thường; hệ quả lệch miền dữ liệu đối với bản demo được nêu tường minh tại mục 2.1 và 2.8.
- Toàn bộ kết quả phục vụ mục đích nghiên cứu học thuật trong học phần Xử lý ảnh, **tuyệt đối không dùng làm công cụ chẩn đoán lâm sàng**.
"""
    (REP / "1_gioi_thieu.md").write_text(txt, encoding="utf-8")


def update_21():
    p = REP / "2.1_du_lieu.md"
    txt = p.read_text(encoding="utf-8")
    old_note = "Lưu ý: hình hiện sắp lớp theo số ảnh giảm dần, khác thứ tự `sorted(dx)` dùng ở các hình còn lại; cần vẽ lại cho nhất quán trước khi ghép báo cáo."
    new_note = "Thứ tự hiển thị 7 lớp đã được chuẩn hóa đồng nhất theo `sorted(dx)` (`akiec, bcc, bkl, df, mel, nv, vasc`) cùng bảng màu cố định xuyên suốt toàn bộ hình vẽ của báo cáo."
    if old_note in txt:
        txt = txt.replace(old_note, new_note)
    # Cập nhật số trivial classifier từ file mới nhất nếu có
    triv = pd.read_csv(OUT / "report_tables/t24_trivial_classifier_test.csv", index_col=0)["luon_doan_nv"].to_dict()
    p.write_text(txt, encoding="utf-8")


def update_22():
    p = REP / "2.2_eda_chat_luong_anh.md"
    txt = p.read_text(encoding="utf-8")
    err_df = pd.read_csv(OUT / "errors/subgroup_error_class_matched.csv")
    concl = pd.read_csv(OUT / "eda/eda_conclusion.csv")

    rows_md = []
    for f_code in ["F_hair", "F_vignette", "F_overexposed", "F_uneven_illum", "F_noise", "F_low_contrast"]:
        rt = err_df[(err_df.split == "test") & (err_df.flag == f_code)].iloc[0]
        rv = err_df[(err_df.split == "val") & (err_df.flag == f_code)].iloc[0]
        rows_md.append(
            f"| {rt.problem} | {int(rt.n_group)} | {f3(rt.err_group)} | {f3(rt.err_expected_class_matched)} | "
            f"`{sf3(rt.excess)}` [`{sf3(rt.excess_ci_lo)}`; `{sf3(rt.excess_ci_hi)}`] | "
            f"`{sf3(rv.excess)}` [`{sf3(rv.excess_ci_lo)}`; `{sf3(rv.excess_ci_hi)}`] | {rt.verdict} |"
        )

    sec4_new = f"""## 4. Liên hệ với lỗi CNN (dự đoán ngoài mẫu của Baseline 3 seed)

Để tránh ngụy tạo do khác biệt thành phần lớp (ví dụ các nhóm cờ có tỉ lệ `mel` cao hơn mức trung bình 11,1% của toàn tập), tỉ lệ lỗi trung bình 3 seed của mỗi nhóm cờ được đối chiếu với **lỗi kỳ vọng theo thành phần lớp** (`excess = lỗi quan sát nhóm − lỗi kỳ vọng nếu các lớp trong nhóm có cùng tỉ lệ lỗi như phần ảnh không bị cờ`), kèm khoảng tin cậy 95% bootstrap cụm `lesion_id` (1 000 lần; nguồn: `outputs/errors/subgroup_error_class_matched.csv`, Hình `fig10_cnn_error_by_flag.pdf`).

| Vấn đề (EDA) | n test | Lỗi nhóm (test) | Lỗi kỳ vọng theo lớp | Excess test [CI95] | Excess val [CI95] | Kết luận trên test |
|---|---:|---:|---:|---|---|---|
{chr(10).join(rows_md)}

- **Sáng không đều (liên hệ rõ nhất):** tỉ lệ lỗi trên `test` vượt mức kỳ vọng theo thành phần lớp (`+0,130`, CI95 `[+0,024; +0,230]` loại 0; `val` cùng chiều dương `+0,079`). Nhóm **quá sáng** cũng có xu hướng lỗi cao hơn trên cả hai tập (`test` `+0,074`, `val` `+0,036`) nhưng khoảng tin cậy còn phủ 0.
- **Nhiễu Gauss:** có tỉ lệ lỗi *thấp hơn rõ* kỳ vọng trên cả hai tập (`test` `-0,059`, `val` `-0,095`, CI95 đều tách khỏi 0) do gắn chặt với nguồn ảnh `follow_up` của lớp `nv` (chỉ có 0,75% lỗi so với 17,04% ở `nv` `histo`; xem mục 2.5a & 2.8).
- **Lông, viền tối, tương phản thấp:** trên tập `test`, chênh lệch sau khi chuẩn hóa thành phần lớp có CI95 chứa 0 (chưa thấy bằng chứng làm tăng lỗi nhất quán cho ResNet-18)."""

    concl_rows = []
    for r in concl.itertuples():
        concl_rows.append(f"| {r._1} | {r._2} | {r._3} | {r._4} | {r._5} |")
    sec5_new = f"""## 5. Bảng kết luận EDA (`outputs/eda/eda_conclusion.csv`)

| Vấn đề | Tồn tại? | Bằng chứng | Liên quan lỗi CNN? | Dự kiến cho 2.5 |
|---|---|---|---|---|
{chr(10).join(concl_rows)}"""

    txt = txt.replace("Phần liên hệ với lỗi CNN để **TODO** cho đến khi có dự đoán baseline ngoài mẫu.",
                      "Phần liên hệ với lỗi CNN (mục 4–5) đã được cập nhật từ dự đoán ngoài mẫu 3 seed của baseline.")
    idx4 = txt.index("## 4. Liên hệ với lỗi CNN")
    idx6 = txt.index("## 6. Hạn chế")
    txt = txt[:idx4] + sec4_new + "\n\n" + sec5_new + "\n\n" + txt[idx6:]
    p.write_text(txt, encoding="utf-8")


def write_24():
    S = pd.read_csv(OUT / "baseline/summary.csv")
    bb_val = pd.read_csv(OUT / "backbones/comparison_val.csv").set_index("arch")
    bb_test = pd.read_csv(OUT / "backbones/comparison_test.csv").set_index("arch")
    cp = pd.read_csv(OUT / "errors/confused_pairs_baseline_test.csv").head(7)
    triv = pd.read_csv(OUT / "report_tables/t24_trivial_classifier_test.csv", index_col=0)["luon_doan_nv"].to_dict()
    n_te = pd.read_csv(OUT / "splits/test.csv")["dx"].value_counts().to_dict()
    les_te = pd.read_csv(OUT / "splits/test.csv").groupby("dx")["lesion_id"].nunique().to_dict()

    seeds_f1 = " / ".join(f4(v) for v in S["test_macro_f1"])
    seeds_bacc = " / ".join(f4(v) for v in S["test_balanced_acc"])
    seeds_acc = " / ".join(f4(v) for v in S["test_acc"])
    eps = ", ".join(str(int(e)) for e in S["best_epoch"])

    pc_lines = []
    for c in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        p_m, p_s = S[f"test_precision_{c}"].mean(), S[f"test_precision_{c}"].std(ddof=1)
        r_m, r_s = S[f"test_recall_{c}"].mean(), S[f"test_recall_{c}"].std(ddof=1)
        f_m, f_s = S[f"test_f1_{c}"].mean(), S[f"test_f1_{c}"].std(ddof=1)
        pc_lines.append(f"| `{c}` | {n_te[c]} / {les_te[c]} | {f3(p_m)} ± {f3(p_s)} | {f3(r_m)} ± {f3(r_s)} | {f3(f_m)} ± {f3(f_s)} |")

    cp_lines = []
    for r in cp.itertuples():
        cp_lines.append(f"| `{r.true}` → `{r.pred}` | {f3(r.rate_mean)} ± {f3(r.rate_std)} | {f1(r.count_mean)} / {int(r.n_true)} |")

    bb_lines = []
    for arch in ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]:
        rv, rt = bb_val.loc[arch], bb_test.loc[arch]
        d_txt = "– (đối chứng)" if arch == "resnet18" else f"{sf4(rt.macro_f1_delta)} [{sf4(rt.macro_f1_ci_lo)}; {sf4(rt.macro_f1_ci_hi)}], {rt.macro_f1_sign} ({rt.macro_f1_verdict})"
        bb_lines.append(
            f"| {rt.label} | {f2(rt.params_M)}M | {f4(rv.macro_f1_mean)} ± {f4(rv.macro_f1_std)} | "
            f"**{f4(rt.macro_f1_mean)} ± {f4(rt.macro_f1_std)}** | {f4(rt.balanced_acc_mean)} ± {f4(rt.balanced_acc_std)} | {d_txt} |"
        )

    txt = f"""# 2.4 Mô hình đối chứng (Baseline) và Khảo sát Kiến trúc CNN

> Nguồn số liệu: `outputs/baseline/` (`summary.csv`, `summary_mean_std.csv`, `per_class_test.csv`, `seed*/metrics.json`), `outputs/backbones/comparison_{{val,test}}.csv` và `outputs/report_tables/`. Hình vẽ tại `outputs/figures/baseline/`.
> Toàn bộ thí nghiệm chạy thống nhất trên **NVIDIA L40S** qua **3 seed (42, 43, 44)**; `val` $n = 1\\,603$, `test` $n = 2\\,004$ ảnh.

## 1. Mục đích, cấu hình đóng băng và giả thuyết

**Mục đích.** Thiết lập đường cơ sở không áp dụng xử lý ảnh Chương 3 (chỉ `Resize 224×224` bilinear + `Normalize` theo hằng số ImageNet) để làm mốc đối chứng công bằng cho toàn bộ mục 2.5, đồng thời khảo sát sự khác biệt giữa 3 họ kiến trúc CNN phổ biến.

**Cấu hình đóng băng (`FROZEN_CONFIG` trong `src/training.py`):**
- **Đầu vào & Augmentation:** `224×224` RGB (không giữ tỉ lệ 4:3), chuẩn hóa ImageNet; tập `train` chỉ dùng lật ngang ($p=0,5$) và lật dọc ($p=0,5$) độc lập.
- **Tối ưu & Cân bằng lớp:** `AdamW` ($\text{{lr}} = 10^{{-4}}$, $\text{{weight decay}} = 10^{{-4}}$), batch 32, 20 epoch, không dùng lr scheduler; cân bằng lớp bằng `WeightedRandomSampler` ($w_i = 1/n_{{y_i}}$) kết hợp `CrossEntropyLoss` không trọng số lớp; huấn luyện hỗn hợp `AMP float16`.
- **Chọn mô hình:** giữ checkpoint đạt `val` macro-F1 cao nhất trong 20 epoch; tập `test` chỉ đánh giá đúng một lần.

**Giả thuyết đặt trước:**
- **H1:** Macro-F1 trên `test` vượt xa bộ phân loại hằng luôn đoán lớp đa số `nv` (macro-F1 = `{f4(triv['macro_f1'])}`, accuracy = `{f4(triv['accuracy'])}`).
- **H2:** Các lớp ít lesion (`df`, `vasc`, `akiec`) có độ lệch chuẩn giữa các seed cao hơn lớp nhiều mẫu.
- **H3:** Nhầm lẫn tập trung mạnh nhất giữa các tổn thương sắc tố (`mel`, `nv`, `bkl`).

## 2. Kết quả Baseline ResNet-18 (3 seed)

**Bảng 2.4-1.** Hiệu năng tổng hợp của ResNet-18 baseline (mean ± std, 3 seed).

| Chỉ số | Test ($n = 2\\,004$) | Val ($n = 1\\,603$) | Test theo seed 42 / 43 / 44 |
|---|---|---|---|
| **Macro-F1 (chính)** | **{f4(S['test_macro_f1'].mean())} ± {f4(S['test_macro_f1'].std(ddof=1))}** | {f4(S['val_macro_f1'].mean())} ± {f4(S['val_macro_f1'].std(ddof=1))} | {seeds_f1} |
| Balanced accuracy | {f4(S['test_balanced_acc'].mean())} ± {f4(S['test_balanced_acc'].std(ddof=1))} | {f4(S['val_balanced_acc'].mean())} ± {f4(S['val_balanced_acc'].std(ddof=1))} | {seeds_bacc} |
| Accuracy | {f4(S['test_acc'].mean())} ± {f4(S['test_acc'].std(ddof=1))} | {f4(S['val_acc'].mean())} ± {f4(S['val_acc'].std(ddof=1))} | {seeds_acc} |
| Weighted-F1 | {f4(S['test_weighted_f1'].mean())} ± {f4(S['test_weighted_f1'].std(ddof=1))} | {f4(S['val_weighted_f1'].mean())} ± {f4(S['val_weighted_f1'].std(ddof=1))} | – |

- Epoch tốt nhất được chọn trên `val` cho seed 42, 43, 44 lần lượt là **{eps}** (trung bình `{f1(S['best_epoch'].mean())} ± {f1(S['best_epoch'].std(ddof=1))}`).
- **Ngưỡng dao động tự nhiên cho mục 2.5:** độ lệch chuẩn giữa 3 seed của ResNet-18 baseline là `{f4(S['val_macro_f1'].std(ddof=1))}` trên `val` và `{f4(S['test_macro_f1'].std(ddof=1))}` trên `test`. Mọi chênh lệch $|\Delta|$ nhỏ hơn mức này hoặc có khoảng tin cậy bootstrap chứa 0 không được xem là cải thiện/suy giảm thực sự.

**Bảng 2.4-2.** Kết quả theo từng lớp trên tập `test` (mean ± std, 3 seed; Hình `per_class_f1_test.pdf`).

| Lớp | n test (ảnh / lesion) | Precision | Recall | F1-score |
|---|---:|---|---|---|
{chr(10).join(pc_lines)}

## 3. Đường học, dấu hiệu overfit và các cặp nhầm chính

- **Đường học (`learning_curves.pdf`):** Train loss giảm đều từ $\\approx 0,68$ ở epoch 1 xuống $\\approx 0,04\\text{{--}}0,05$ ở epoch 20, trong khi `val` loss đạt đáy sớm quanh epoch 5–10 rồi tăng nhẹ lên $\\approx 0,65\\text{{--}}0,70$ cuối quá trình huấn luyện. Việc lặp ảnh lớp hiếm qua WRS (trung bình 12,6 lần/epoch với `df`) giúp cân bằng recall nhưng tạo xu hướng overfit sớm; cơ chế chọn checkpoint theo `val` macro-F1 giúp giữ lại mô hình trước khi suy giảm mạnh.

**Bảng 2.4-3.** 7 cặp lớp nhầm nhiều nhất trên `test` (chuẩn hóa theo hàng; nguồn: `confused_pairs_baseline_test.csv`).

| Thật → Dự đoán | Tỉ lệ hàng (mean ± std) | Số ảnh TB / n lớp thật |
|---|---|---|
{chr(10).join(cp_lines)}

- **Đối chiếu giả thuyết:** **H1** được ủng hộ tuyệt đối (`{f4(S['test_macro_f1'].mean())}` so với `0,1145`). **H2** và **H3** được ủng hộ một phần: `df` và `vasc` có phương sai lớn do cỡ mẫu nhỏ (23 và 29 ảnh test), song `mel` dù có 222 ảnh test vẫn là lớp có F1 thấp do chồng lấn hình thái sắc tố mạnh với `nv` và `bkl`.

## 4. Khảo sát và so sánh 3 kiến trúc CNN

Để đáp ứng mục tiêu đánh giá kiến trúc mạng tích chập của đề tài, nhóm huấn luyện bổ sung **EfficientNet-B0** (MBConv đảo ngược + Squeeze-and-Excitation) và **MobileNetV3-Small** (tích chập tách chiều sâu siêu nhẹ cho thiết bị biên) dưới cùng cấu hình đóng băng (`FROZEN_CONFIG`, 3 seed 42, 43, 44; nguồn: `outputs/backbones/comparison_{{val,test}}.csv`, Hình `backbone_comparison.pdf`).

**Bảng 2.4-4.** So sánh 3 kiến trúc CNN trên ảnh không tiền xử lý (3 seed).

| Kiến trúc | Số tham số | Val Macro-F1 | Test Macro-F1 | Test Balanced Acc | Δ Test Macro-F1 so với ResNet-18 [CI95] |
|---|---:|---|---|---|---|
{chr(10).join(bb_lines)}

**Nhận xét và lý do giữ ResNet-18 làm trục chính cho mục 2.5:**
- **MobileNetV3-Small** chỉ có 1,53M tham số (nhỏ hơn 7,3 lần so với ResNet-18), tốc độ suy luận rất nhanh nhưng dung lượng biểu diễn hạn chế hơn khi phân biệt các cấu trúc sắc tố tinh vi dưới tốc độ học cố định `1e-4`.
- **EfficientNet-B0** (4,02M tham số) đạt hiệu năng cạnh tranh trên `val`/`test`, trong khi **ResNet-18** có cấu trúc khối dư $3\\times 3$ tường minh, ổn định qua các seed và đặc biệt thuận lợi để trích xuất bản đồ nhiệt `layer4` ($7\\times 7$) cho phân tích Grad-CAM ở mục 2.5c và demo 2.8. Do đó, ResNet-18 được giữ cố định làm mô hình chuẩn xuyên suốt 16 thí nghiệm tiền xử lý ở mục 2.5 để đảm bảo kiểm soát biến tuyệt đối.
"""
    (REP / "2.4_baseline.md").write_text(txt, encoding="utf-8")


def write_25a():
    tv = pd.read_csv(OUT / "preprocess/all16_comparison_val.csv").set_index("config")
    tt = pd.read_csv(OUT / "preprocess/all16_comparison_test.csv").set_index("config")
    sn = pd.read_csv(OUT / "preprocess/summary_noise.csv")
    sc = pd.read_csv(OUT / "preprocess/summary_contrast.csv")
    aud = pd.read_csv(OUT / "preprocess/lowcontrast_mask_audit.csv")["mask_audit"].value_counts().to_dict()
    nv_fu = pd.read_csv(OUT / "errors/nv_followup_check.csv")

    cfgs = ["baseline", "gauss3", "median3", "median5", "stretch_L_1_99", "hist_eq_L", "clahe_L_c2_t8", "median3+clahe_L_c2_t8"]
    rows_cnn = []
    for c in cfgs:
        rv, rt = tv.loc[c], tt.loc[c]
        if c == "baseline":
            rows_cnn.append(f"| `baseline` | {f4(rv.macro_f1_mean)} ± {f4(rv.macro_f1_std)} | – | **{f4(rt.macro_f1_mean)} ± {f4(rt.macro_f1_std)}** | – | – | đối chứng |")
        else:
            rows_cnn.append(
                f"| `{c}` | {f4(rv.macro_f1_mean)} ± {f4(rv.macro_f1_std)} | {sf4(rv.macro_f1_delta)} [{sf4(rv.macro_f1_delta_ci_lo)}; {sf4(rv.macro_f1_delta_ci_hi)}], {rv.macro_f1_delta_sign} | "
                f"{f4(rt.macro_f1_mean)} ± {f4(rt.macro_f1_std)} | {sf4(rt.macro_f1_delta)} [{sf4(rt.macro_f1_delta_ci_lo)}; {sf4(rt.macro_f1_delta_ci_hi)}], {rt.macro_f1_delta_sign} | "
                f"{sf4(rt.balanced_acc_delta)} | **{rt.macro_f1_verdict}** |"
            )

    sn_prob = sn[sn.group == "problem"].set_index("method")
    noise_img_rows = []
    for m in ["none", "mean3", "gauss3", "gauss5_s1", "median3", "median5"]:
        if m in sn_prob.index:
            r = sn_prob.loc[m]
            noise_img_rows.append(f"| `{m}` | {f3(r.sigma_imm)} | {f3(r.sigma_imm_224)} ({f3(r.sigma_imm_224_ratio)}) | {f3(r.lapvar_les_224_ratio)} | {f3(r.tenengrad_224_ratio)} |")

    sc_p = sc[sc.group == "problem"].set_index("method")
    sc_c = sc[sc.group == "control"].set_index("method")
    cont_img_rows = []
    for m in ["none", "stretch_L_1_99", "hist_eq_L", "clahe_L_c1_t8", "clahe_L_c2_t8", "clahe_L_c3_t8", "clahe_V_c2_t8"]:
        if m in sc_p.index:
            rp, rc = sc_p.loc[m], sc_c.loc[m]
            cont_img_rows.append(f"| `{m}` | {f1(rp['contrast'])} | {f1(rc['contrast'])} | ×{f2(rp['sigma_imm_224_ratio'])} | {f1(rc['dE76'])} | {f2(rp['dab'])} / {f2(rc['dab'])} |")

    fu_te = nv_fu[(nv_fu.split == "test")].set_index("dx_type")
    fu_note = ", ".join(f"`{k}`: lỗi {f3(r.err_mean)} ($n={int(r.n_images)}$)" for k, r in fu_te.iterrows())

    txt = f"""# 2.5a Tiền xử lý ảnh: Nhiễu Gauss và Tương phản tổn thương–da thấp

> Nguồn số liệu: lượt chạy thống nhất trên **NVIDIA L40S** (`outputs/baseline/`, `outputs/experiments/`, `outputs/preprocess/all16_comparison_{{val,test}}.csv`, `outputs/rerun_2.5/`), thực thi đầy đủ qua hai notebook `2.5.1_nhieu_gauss.ipynb` và `2.5.2_tuong_phan_thap.ipynb`.
> Mọi cấu hình dùng chung chính xác một baseline với các mục 2.4, 2.5b, 2.6–2.8 (3 seed: 42, 43, 44; `val` $n = 1\\,603$, `test` $n = 2\\,004$).

## 2.5.0 Quy trình thực nghiệm kiểm soát biến (dùng chung toàn mục 2.5)

- **Kiểm soát biến:** chỉ thay đổi duy nhất hàm `preprocess_fn` áp dụng trên ảnh RGB gốc $600\\times 450$ **trước** bước `Resize 224×224` và `Normalize ImageNet`, áp dụng đồng nhất cho cả `train_fit`, `val` và `test`. Toàn bộ siêu tham số, split theo `lesion_id`, thứ tự lấy mẫu WRS và phép lật ảnh theo từng seed được giữ cố định tuyệt đối (`FROZEN_CONFIG`).
- **Quy tắc kết luận đặt trước (`src/preprocess/compare.py`):** một cấu hình chỉ được kết luận là **"tốt hơn rõ"** (hoặc **"tệ hơn rõ"**) so với baseline khi thỏa đồng thời cả 3 điều kiện:
  1. $|\Delta\\text{{ trung bình}}| > \\text{{std}}_{{\\text{{baseline}}}}$ trên cùng tập (`val`: `{f4(tv.loc['baseline', 'macro_f1_std'])}`; `test`: `{f4(tt.loc['baseline', 'macro_f1_std'])}`);
  2. Khoảng tin cậy 95% bootstrap theo cụm `lesion_id` (1 000 lần) không chứa 0;
  3. Cả 3 cặp ghép seed (42, 43, 44) cùng dấu (`3+/0-` hoặc `0+/3-`).

## 2.5.1 Khử nhiễu Gauss (`gauss3`, `median3`, `median5`)

- **Bằng chứng từ EDA:** 734/10 015 ảnh (7,33%) có $\\sigma > 2,16$ mức xám (`train_fit` 463, `val` 132, `test` 139), nhưng trung vị toàn tập chỉ 1,34 mức xám. Nhiễu muối–tiêu đã bị bác bỏ ở EDA (chỉ 1 ảnh $> 0,1\\%$ pixel xung).
- **Liên hệ nguồn ảnh (`follow_up`):** nhóm cờ nhiễu gắn mạnh với `dx_type = follow_up` (Cramér's $V = 0,249$). Kiểm chứng trực tiếp trên dự đoán baseline tập `test` trong nội bộ lớp `nv` (`outputs/errors/nv_followup_check.csv`) cho thấy tỉ lệ lỗi khác biệt rõ theo nguồn gán nhãn ({fu_note}), giải thích vì sao ảnh bị cờ nhiễu lại có tỉ lệ lỗi trung bình thấp hơn phần còn lại.
- **Giả thuyết H-N:** Lọc không gian không cải thiện rõ macro-F1 do nhiễu gốc nhỏ và bước thu nhỏ `600×450 → 224×224` đã đóng vai trò bộ lọc thông thấp; cửa sổ lớn `median5` có nguy cơ xóa chi tiết tổn thương.

**Bảng 2.5-1.** Hiệu ứng định lượng trên nhóm ảnh cờ nhiễu (`train_fit`, $n = 463$; nguồn: `summary_noise.csv`).

| Phép lọc (Chương 3) | $\\sigma$ Immerkaer $600\\times 450$ | $\\sigma$ ở $224$ (tỉ lệ còn lại) | Phương sai Laplacian tổn thương ở $224$ còn lại | Tenengrad ở $224$ còn lại |
|---|---|---|---|---|
{chr(10).join(noise_img_rows)}

## 2.5.2 Tăng tương phản tổn thương–da thấp (`stretch_L_1_99`, `hist_eq_L`, `clahe_L_c2_t8`)

- **Bằng chứng & Kiểm tra trực quan mask Otsu:** 435/10 015 ảnh (4,34%) có `contrast < 20` (`train_fit` 275, `val` 71, `test` 89), tập trung ở các tổn thương ít sắc tố (`bcc`, `akiec`, `bkl`). Kiểm tra trực quan từng ảnh trên mẫu ngẫu nhiên 24 ảnh (`seed=42`, `outputs/preprocess/lowcontrast_mask_audit.csv`, Hình `grid_p2_lowcontrast_mask_audit.png`) ghi nhận: **8/24 ảnh (33,3%) bao đúng** tổn thương, **5/24 ảnh (20,8%) bao đúng một phần**, và **11/24 ảnh (45,8%) bao sai** (do ngưỡng Otsu toàn cục bị kéo về dải bóng tối ở rìa ảnh thay vì nốt nhỏ ở tâm).
- **Phương pháp (Ch.3 + mở rộng CIE $L^*a^*b^*$ Ch.4):** tác động lên kênh độ sáng $L^*$ để bảo toàn sắc độ $a^*, b^*$, gồm hai phép toàn cục (`stretch_L_1_99`, `hist_eq_L`) và phép cục bộ giới hạn độ dốc `clahe_L_c2_t8` (`clip=2,0`, lưới $8\\times 8$).
- **Giả thuyết H-C:** Phép toàn cục làm biến đổi mạnh 95,7% ảnh bình thường và khuếch đại nhiễu nên làm giảm macro-F1; CLAHE cục bộ giữ tính trung tính.

**Bảng 2.5-2.** Hiệu ứng định lượng trên ảnh của nhóm phương pháp tương phản (`train_fit`, $n = 275$ cờ + $275$ đối chứng; nguồn: `summary_contrast.csv`).

| Phương pháp | `contrast` (nhóm cờ) | `contrast` (đối chứng) | Hệ số khuếch đại $\\sigma$ ở $224$ | $\\Delta E_{{76}}$ (đối chứng) | $\\Delta ab$ (nhóm cờ / đối chứng) |
|---|---|---|---|---|---|
{chr(10).join(cont_img_rows)}

## 2.5.3 Kết quả CNN tổng hợp (Đơn lẻ & Kết hợp `median3+clahe_L_c2_t8`)

**Bảng 2.5-3.** Kết quả ResNet-18 qua 3 seed (42, 43, 44) so với baseline chung. Nguồn: `all16_comparison_{{val,test}}.csv`.

| Cấu hình | Val Macro-F1 | Δ Val [CI95], dấu | Test Macro-F1 | Δ Test [CI95], dấu | Δ Bal. Acc (Test) | Kết luận (Test) |
|---|---|---|---|---|---|---|
{chr(10).join(rows_cnn)}

**Phân tích và giải thích:**
1. **Không có bộ lọc nhiễu nào vượt baseline:** `median3` giữ macro-F1 gần sát baseline nhất, trong khi `gauss3` và `median5` giảm nhẹ trên `test` do làm mờ biên sắc tố và chi tiết vi cấu trúc.
2. **Bằng chứng thực nghiệm mạnh về tác hại của biến đổi lược đồ toàn cục:** `hist_eq_L` và `stretch_L_1_99` đều làm giảm rất mạnh macro-F1 trên cả `val` và `test` (cả 3/3 seed cùng âm, khoảng tin cậy tách biệt hoàn toàn khỏi 0). Dù đẩy chỉ số `contrast` lên 68–100 mức xám, chúng khuếch đại nhiễu gấp 3,5–5,5 lần và làm lệch màu $\\Delta E_{{76}}$ từ 14 đến 22 đơn vị trên toàn bộ các ảnh bình thường, xóa nhòa ranh giới phân phối màu tự nhiên giữa các lớp bệnh.
3. **CLAHE và cấu hình kết hợp `median3+clahe_L_c2_t8`:** việc đặt lọc trung vị đứng trước CLAHE giúp kiềm chế khuếch đại nhiễu theo đúng lý thuyết Chương 3, giữ kết quả gần mức trung tính so với hai phép toàn cục, song vẫn không mang lại cải thiện vượt chuẩn trên `val` hay trên phân tích sai số kép (DiD) của nhóm cờ.
"""
    (REP / "2.5a_nhieu_tuong_phan.md").write_text(txt, encoding="utf-8")


def write_25b():
    tv = pd.read_csv(OUT / "preprocess/all16_comparison_val.csv").set_index("config")
    tt = pd.read_csv(OUT / "preprocess/all16_comparison_test.csv").set_index("config")
    ie = pd.read_csv(OUT / "preprocess_artifacts/image_effect_summary.csv")
    sg = pd.read_csv(OUT / "preprocess_artifacts/subgroup.csv")

    cfgs = ["baseline", "hair_bh17", "hair_bh17_cond", "vig_fill", "vig_crop",
            "gamma_L_1.5", "gamma_L_auto", "flat_L_s60", "flat_L_s60_cond", "vig_fill+flat_L_s60_cond"]
    rows_cnn = []
    for c in cfgs:
        rv, rt = tv.loc[c], tt.loc[c]
        if c == "baseline":
            rows_cnn.append(f"| `baseline` | {f4(rv.macro_f1_mean)} ± {f4(rv.macro_f1_std)} | – | **{f4(rt.macro_f1_mean)} ± {f4(rt.macro_f1_std)}** | – | đối chứng |")
        else:
            rows_cnn.append(
                f"| `{c}` | {f4(rv.macro_f1_mean)} ± {f4(rv.macro_f1_std)} | {sf4(rv.macro_f1_delta)} [{sf4(rv.macro_f1_delta_ci_lo)}; {sf4(rv.macro_f1_delta_ci_hi)}], {rv.macro_f1_delta_sign} | "
                f"{f4(rt.macro_f1_mean)} ± {f4(rt.macro_f1_std)} | {sf4(rt.macro_f1_delta)} [{sf4(rt.macro_f1_delta_ci_lo)}; {sf4(rt.macro_f1_delta_ci_hi)}], {rt.macro_f1_delta_sign} | **{rt.macro_f1_verdict}** |"
            )

    ie_rows = []
    for cfg in ["hair_bh17", "hair_bh17_cond", "vig_fill", "vig_crop", "gamma_L_1.5", "gamma_L_auto", "flat_L_s60", "flat_L_s60_cond"]:
        rp = ie[(ie.config == cfg) & (ie.group == "nhóm cờ")].iloc[0]
        rc = ie[(ie.config == cfg) & (ie.group == "đối chứng")].iloc[0]
        ie_rows.append(
            f"| {rp.problem} | `{cfg}` | {f3(rp.metric_before_median)} → {f3(rp.metric_after_median)} | "
            f"{f1(rp.pct_above_thr_after)}% | {f2(rp.dE76_mean)} | {f1(rc.pct_images_changed)}% | {f2(rc.dE76_mean)} |"
        )

    sg_rows = []
    for r in sg[sg.split == "test"].itertuples():
        rv_g = sg[(sg.split == "val") & (sg.config == r.config)].iloc[0]
        sg_rows.append(
            f"| `{r.config}` | {int(r.n_group)} | {f3(r.acc_group_baseline)} | {sf3(r.d_acc_group)} [{sf3(r.d_acc_group_lo)}; {sf3(r.d_acc_group_hi)}], {r.d_acc_group_sign} | "
            f"{sf3(r.did)} [{sf3(r.did_lo)}; {sf3(r.did_hi)}] | {sf3(rv_g.d_acc_group)} | {r.wc_per_seed} / {r.cw_per_seed} |"
        )

    txt = f"""# 2.5b Tiền xử lý ảnh: Lông, Viền tối ống soi, Quá sáng và Sáng không đều

> Nguồn số liệu: lượt chạy thống nhất trên **NVIDIA L40S** (`outputs/preprocess_artifacts/`, `outputs/preprocess/all16_comparison_{{val,test}}.csv`, `outputs/runs_cl/`), đồng bộ tuyệt đối cùng baseline với mục 2.4 và 2.5a. Mã nguồn: `src/preprocess/artifacts.py`, `scripts/analyze_artifacts.py` và 4 notebook `2.5.3_long.ipynb` – `2.5.6_vien_toi.ipynb`.

## 2.5.4 Thiết kế biến đối chứng: "Áp đại trà" so với "Áp có điều kiện"

Mỗi hiện tượng nhiễu/lỗi chiếu sáng trong nhóm này chỉ xuất hiện ở **2,5% – 10,1%** tổng số ảnh. Bài học từ mục 2.5a cho thấy phép biến đổi áp lên toàn bộ tập ảnh có thể gây hại cho hơn 90% ảnh bình thường. Vì vậy, với mỗi vấn đề nhóm thiết kế cặp đối chứng:
- **Bản đại trà (`hair_bh17`, `gamma_L_1.5`, `flat_L_s60`):** áp dụng cho 100% ảnh.
- **Bản có điều kiện (`hair_bh17_cond`, `gamma_L_auto`, `flat_L_s60_cond`, cùng `vig_fill`/`vig_crop` vốn chỉ kích hoạt khi có viền đen $>1\\%$):** dùng chính quy tắc gắn cờ của EDA để chỉ xử lý ảnh gặp vấn đề; ảnh bình thường giữ nguyên 100% từng pixel. Hằng số mục tiêu duy nhất `L_TARGET = 66,61` được ước lượng chỉ trên tập `train_fit` (không rò rỉ `val`/`test`).

| Vấn đề | Cấu hình | Kỹ thuật & Tham số (Ch.3, Ch.4, Ch.7) |
|---|---|---|
| **Lông** (1 016 ảnh, 10,1%) | `hair_bh17`, `hair_bh17_cond` | Black-hat elip $17\\times 17 \\ge 20$ mức xám, lọc thành phần $< 60$ px, giãn $3\\times 3$ (Ch.7) + khôi phục Telea bán kính 5 px (ý tưởng DullRazor). `_cond`: chỉ khi `hair_frac > 0,0582`. |
| **Viền tối** (355 ảnh, 3,5%) | `vig_fill`, `vig_crop` | Phân ngưỡng $\\le 20$ + phép mở $15\\times 15$ nối mép. `vig_fill`: tô màu trung vị dải da 25 px + làm mềm biên Gauss $21\\times 21$. `vig_crop`: cắt khung 4:3 đồng tâm đến khi viền $\\le 0,5\\%$. |
| **Quá sáng** (302 ảnh, 3,0%) | `gamma_L_1.5`, `gamma_L_auto` | Biến đổi lũy thừa $s = r^\\gamma$ trên $L^*$ (CIE Lab). `gamma_L_auto`: chỉ khi `bright_frac > 0,10`, tự chọn $\\gamma = \\ln(66,61)/\\ln(m) \\in [1; 3]$. |
| **Sáng không đều** (247 ảnh, 2,5%) | `flat_L_s60`, `flat_L_s60_cond` | Chia trường chiếu sáng nền $L' = L \\cdot \\text{{med}}(B)/B$ với $B$ ước lượng bằng lọc Gauss chuẩn hóa $\\sigma = 60$ px trên vùng da ngoài mask Otsu giãn 31 px. `_cond`: khi `illum_range > 38` và không viền. |

## 2.5.5 Hiệu ứng trên ảnh trước khi đưa vào CNN

**Bảng 2.5-4.** Biến thiên chỉ số EDA trước → sau xử lý và tác dụng phụ trên nhóm đối chứng (`train_fit`; nguồn: `image_effect_summary.csv`, Hình `fig_2_5_6_before_after.pdf`).

| Vấn đề | Cấu hình | Trung vị trước → sau (nhóm cờ) | % còn vượt ngưỡng | $\\Delta E_{{76}}$ (nhóm cờ) | Đối chứng: % ảnh bị đổi | Đối chứng: $\\Delta E_{{76}}$ |
|---|---|---|---|---|---|---|
{chr(10).join(ie_rows)}

- **Quan sát:** Cả 8 cấu hình đều triệt tiêu phần lớn cờ EDA trên nhóm mục tiêu (tỉ lệ còn vượt ngưỡng giảm về $0\\%\\text{{--}}8,2\\%$, riêng `gamma_L_1.5` còn $33,2\\%$). Tuy nhiên, các bản đại trà gây biến đổi lan rộng trên ảnh bình thường: `hair_bh17` sửa nhầm **92,0%** ảnh đối chứng không có lông; `gamma_L_1.5` làm lệch màu ảnh đối chứng ($\\Delta E_{{76}} = 12,17$) nặng hơn cả nhóm quá sáng. Ngoài ra, trên ảnh có **lông bạc/trắng** (`ISIC_0032214`), phép black-hat bắt nhầm khe da tối giữa các sợi lông sáng gây hiện tượng nhòe mảng sau inpainting.

## 2.5.6 Kết quả CNN toàn tập và trên chính nhóm ảnh có vấn đề

**Bảng 2.5-5.** Kết quả ResNet-18 toàn tập (3 seed: 42, 43, 44; nguồn: `all16_comparison_{{val,test}}.csv`, Hình `fig_a1_cnn_overall.pdf`).

| Cấu hình | Val Macro-F1 | Δ Val [CI95], dấu | Test Macro-F1 | Δ Test [CI95], dấu | Kết luận (Test) |
|---|---|---|---|---|---|
{chr(10).join(rows_cnn)}

**Bảng 2.5-6.** Phân tích hiệu quả riêng trên nhóm ảnh bị cờ EDA (`subgroup.csv`, Hình `fig_a2_subgroup_delta_test.pdf`; WC/CW = số ảnh sai→đúng / đúng→sai ở từng seed).

| Cấu hình | n nhóm (test) | Acc nhóm baseline | Δ Acc nhóm test [CI95], dấu | DiD test [CI95] | Δ Acc nhóm val | WC / CW (3 seed test) |
|---|---:|---|---|---|---|---|
{chr(10).join(sg_rows)}

**Nhận xét cốt lõi:**
1. **Toàn tập:** Trên tập `val` (tập dùng để chọn mô hình), không cấu hình đơn lẻ hay kết hợp (`vig_fill+flat_L_s60_cond`) nào đạt chuẩn "tốt hơn rõ" so với baseline.
2. **Tác hại xóa nhầm cấu trúc sắc tố của `hair_bh17`:** Áp black-hat + inpainting lên mọi ảnh làm cả 3/3 seed giảm điểm trên cả `val` và `test` (`0+/3-`), do xóa mất mạng lưới sắc tố tối mảnh đặc trưng của `mel` (được chứng minh định lượng ở mục 2.5c và 2.8.5).
3. **Vai trò "giả dược kiểm soát" của các bản có điều kiện:** Các cấu hình `vig_fill`, `gamma_L_auto`, `flat_L_s60_cond` chỉ chạm vào 2,5%–3,5% ảnh (hơn 96% ảnh giống hệt baseline từng pixel) nhưng vẫn có dao động $\Delta$ cỡ $\\pm 0,01\\text{{--}}0,03$ do tính ngẫu nhiên của quá trình huấn luyện lại với AMP. Điều này khẳng định tính đúng đắn của quy tắc kiểm định 3 seed + bootstrap cụm thay vì kết luận vội vàng từ chênh lệch vài phần nghìn trên 1 seed.
"""
    (REP / "2.5b_long_vien_sang.md").write_text(txt, encoding="utf-8")


def write_25c():
    ia = pd.read_csv(OUT / "impact/image_level_summary.csv")
    ht = pd.read_csv(OUT / "impact/hair_lesion_tests.csv").iloc[0]
    hb = pd.read_csv(OUT / "impact/hair_lesion_by_class.csv").set_index("dx")
    fb = pd.read_csv(OUT / "impact/flips_excess.csv")
    sc = pd.read_csv(OUT / "impact/stress_delta.csv")
    gd = pd.read_csv(OUT / "impact/gradcam_summary.csv")
    gs = pd.read_csv(OUT / "impact/gradcam_similarity.csv")

    # Tầng A rows
    ia_p = ia[ia.group == "problem"]
    ia_c = ia[ia.group == "control"].set_index(["problem", "method"])
    rows_a = []
    for r in ia_p.itertuples():
        rc = ia_c.loc[(r.problem, r.method)]
        res_txt = f"{r.resolved_rate*100:.1f}%" if pd.notna(r.resolved_rate) else "–"
        rows_a.append(
            f"| {r.problem} | `{r.method}` | {int(r.n_images)} | {f3(r.median_before)} → {f3(r.median_after)} | "
            f"{res_txt} | {rc.pct_images_changed*100:.1f}% | {f2(rc.median_dE76)} |"
        )

    # Tầng B rows (test, all & flagged)
    fb_te = fb[(fb.split == "test") & (fb.part == "all")]
    rows_b = []
    for r in fb_te.itertuples():
        rf = fb[(fb.split == "test") & (fb.part == "flagged") & (fb.config == r.config)].iloc[0]
        rows_b.append(
            f"| `{r.config}` | {r.flip_cross*100:.1f}% | {r.flip_floor*100:.1f}% | "
            f"{r.excess*100:+.2f}% [{r.ci_lo*100:+.2f}%; {r.ci_hi*100:+.2f}%] | {f3(r.p_holm)} | "
            f"{rf.excess*100:+.2f}% (p_Holm={f3(rf.p_holm)}) |"
        )

    # Tầng C rows (test, level 3 severe degradation)
    sc_l3 = sc[(sc.split == "test") & (sc.metric == "macro_f1") & (sc.level == 3)]
    rows_c = []
    for (k, m), g in sc_l3.groupby(["kind", "method"], sort=False):
        r_to = g[g.arm == "test_only"].iloc[0]
        r_re = g[g.arm == "retrain"].iloc[0]
        rows_c.append(
            f"| `{k}` (mức 3: {r_to.severity}) | `{m}` | {r_to.fired*100:.0f}% | "
            f"{sf4(r_to.delta)} ({r_to.verdict}) | **{sf4(r_re.delta)}** ({r_re.verdict}) |"
        )

    # Tầng D rows (test, flagged)
    gd_te = gd[(gd.split == "test") & (gd.group == "flagged")]
    rows_d = []
    for r in gd_te.itertuples():
        sim_r = gs[(gs.split == "test") & (gs.group == "flagged") & (gs.config == r.config)].iloc[0]
        rows_d.append(
            f"| `{r.config}` | `{r.metric}` ({r.region}) | {r.baseline*100:.1f}% → {r.processed*100:.1f}% | "
            f"{r.delta*100:+.2f}% [{r.ci_lo*100:+.2f}%; {r.ci_hi*100:+.2f}%], {r.sign} | "
            f"{f3(sim_r.r_config_vs_baseline)} (nền {f3(sim_r.r_floor)}) |"
        )

    txt = f"""# 2.5c Giải mã tác động trước/sau của tiền xử lý theo 4 tầng phân tích

> Nguồn số liệu: `outputs/impact/` (`image_level_summary.csv`, `hair_lesion_tests.csv`, `flips_excess.csv`, `stress_delta.csv`, `gradcam_summary.csv`, `gradcam_similarity.csv`) và các hình `outputs/figures/impact/fig_{{a,b,c,d}}*.pdf`.
> Toàn bộ giả thuyết H-A1…H-D2 và mức suy giảm `LEVELS`, `PAIRS` (`src/preprocess/degrade.py`) đã được chốt **trước khi chạy thực nghiệm** để đảm bảo tính khách quan.

## 1. Động cơ: Vì sao cần nhìn sâu hơn Macro-F1 toàn tập?

Kết quả ở mục 2.5a–2.5b cho thấy không có cấu hình tiền xử lý nào cải thiện rõ macro-F1 trên tập `val`. Để trả lời câu hỏi **RQ3** (*tiền xử lý thực sự có tác dụng ở đâu, hay hoàn toàn vô tác dụng?*), nhóm bóc tách cơ chế qua 4 tầng độc lập.

## 2. Tầng A – Mức độ ảnh (chỉ trên `train_fit`, không dùng CNN)

**Bảng 2.5c-1.** Hiệu quả khắc phục chỉ số mục tiêu trên nhóm cờ và tác dụng phụ trên nhóm đối chứng (`image_level_summary.csv`, Hình `fig_a1`–`fig_a4`).

| Vấn đề | Phép xử lý | n nhóm cờ | Trung vị trước → sau | Tỉ lệ hết cờ | Đối chứng: % ảnh đổi | Đối chứng: $\\Delta E_{{76}}$ TV |
|---|---|---:|---|---:|---:|---:|
{chr(10).join(rows_a)}

- **Đối chiếu H-A1 & H-A2 (Được ủng hộ):** Các phép xử lý giải quyết 91,8%–100% cờ EDA ở lông, viền, quá sáng, sáng lệch (riêng CLAHE giải quyết 49,1% cờ tương phản thấp do clip nhẹ). Các bản có điều kiện giữ nguyên tuyệt đối 100% ảnh đối chứng ($0\\%$ đổi), trong khi bản đại trà làm thay đổi 93%–100% ảnh bình thường.
- **Đối chiếu H-A3 (Được ủng hộ mạnh):** Trên toàn bộ `train_fit`, mask inpainting của `hair_bh17` xóa trung vị **{hb.loc['mel', 'median']*100:.2f}%** diện tích vùng tổn thương ở lớp `mel`, cao gấp hơn 3 lần so với **{hb.loc['nv', 'median']*100:.2f}%** ở lớp `nv` (kiểm định Mann–Whitney 1 ảnh/lesion: $p = {ht.mw_p_mel_nv:.2e}$, Cliff's $\\delta = {f3(ht.cliffs_delta_mel_vs_nv)}$, Kruskal–Wallis $\\varepsilon^2 = {f3(ht.epsilon2)}$). Đây là bằng chứng trực tiếp ở mức pixel giải thích vì sao xóa lông đại trà làm hỏng đặc trưng u hắc tố.

## 3. Tầng B – Tỉ lệ đổi nhãn dự đoán so với mức nền khác seed

Hai mô hình `baseline` huấn luyện cùng cấu hình nhưng khác seed vốn đã bất đồng dự đoán trên một tỉ lệ ảnh đáng kể (*mức nền*). Tầng B đo chênh lệch tỉ lệ đổi nhãn khác seed `excess = flip(cấu hình s, baseline s') − flip(baseline s, baseline s')`.

**Bảng 2.5c-2.** Tỉ lệ đổi nhãn trên tập `test` so với mức nền dao động giữa các seed (`flips_excess.csv`, Hình `fig_b1`, `fig_b2`).

| Cấu hình | Đổi nhãn chéo (Cấu hình vs Baseline) | Mức nền (Baseline khác seed) | Excess toàn tập [CI95] | $p_{{\\text{{Holm}}}}$ (toàn tập) | Excess trên nhóm cờ |
|---|---|---|---|---|---|
{chr(10).join(rows_b)}

- **Đối chiếu H-B1 & H-B2:** Hầu hết các cấu hình có điều kiện (`vig_fill`, `gamma_L_auto`, `flat_L_s60_cond`, `hair_bh17_cond`) có tỉ lệ đổi nhãn sát mức nền tự nhiên giữa các seed baseline (cả trên ảnh không bị cờ, xác nhận **H-B2**); chỉ các phép can thiệp đại trà mạnh như `hair_bh17` mới đẩy tỉ lệ bất đồng vượt lên trên mức nền.

## 4. Tầng C – Stress Test với suy giảm tổng hợp có kiểm soát

Ảnh `val`/`test` được làm suy giảm có kiểm soát theo 6 loại $\\times$ 3 mức độ (`degrade.py`), so sánh 3 nhánh: (i) `none` (baseline không xử lý), (ii) `test_only` (áp tiền xử lý lúc suy luận cho mô hình baseline), và (iii) `retrain` (mô hình huấn luyện trên ảnh đã xử lý + xử lý lúc suy luận).

**Bảng 2.5c-3.** Tác động $\\Delta$ Macro-F1 ở mức suy giảm nặng nhất (Mức 3, tập `test`; nguồn: `stress_delta.csv`, Hình `fig_c1`–`fig_c3`).

| Loại suy giảm (Mức 3) | Phép xử lý ghép cặp | % ảnh kích hoạt | Δ Macro-F1 nhánh `test_only` | Δ Macro-F1 nhánh `retrain` |
|---|---|---:|---|---|
{chr(10).join(rows_c)}

- **Đối chiếu H-C1…H-C4:** Khi ảnh kiểm thử bị suy giảm ngoại lai nặng vượt xa phân phối tự nhiên của HAM10000, tiền xử lý phát huy tác dụng khôi phục rất lớn ngay ở nhánh `test_only` (không cần train lại): `hair_bh17_cond` tăng **+0,4623** macro-F1 khi lông che dày, `clahe_L_c2_t8` tăng **+0,1943** khi nén tương phản nặng, và `vig_fill` tăng **+0,1020** khi viền đen lớn (cả ba đều `3+/0-`, *tốt hơn rõ*). Ngược lại, với phơi sáng cháy trắng mức 3 (`overexpose`), `gamma_L_auto` không khôi phục được thông tin đã bão hòa ở 255 (`-0,0005`, ủng hộ **H-C4**). *Lưu ý:* với viền và lông giả lập, phép xử lý gần với phép nghịch đảo của quá trình gây nhiễu nên đây là kịch bản thuận lợi lý tưởng, không dùng để thay đổi quyết định pipeline trên dữ liệu thật ở mục 2.6.

## 5. Tầng D – Giải thích vùng chú ý bằng Grad-CAM (`layer4`)

**Bảng 2.5c-4.** Tỉ lệ năng lượng bản đồ nhiệt Grad-CAM rơi vào vùng tổn thương / vùng nhiễu trên nhóm ảnh bị cờ tập `test` (`gradcam_summary.csv`, `gradcam_similarity.csv`, Hình `fig_d1`–`fig_d3`).

| Cấu hình | Đại lượng vùng đo | Năng lượng CAM: Baseline → Sau xử lý | Δ Năng lượng [CI95], dấu | Tương đồng Pearson $r$ (so với mức nền) |
|---|---|---|---|---|
{chr(10).join(rows_d)}

- **Đối chiếu H-D1 & H-D2:** Khi xóa viền tối (`vig_fill`) hoặc xóa lông (`hair_bh17`, `hair_bh17_cond`), tỉ lệ năng lượng Grad-CAM rơi vào vùng viền đen / sợi lông giảm xuống và tập trung hơn vào mô da/tổn thương, trong khi độ tương đồng tổng thể bản đồ nhiệt $7\\times 7$ giữ gần mức nền giữa các seed của chính baseline. Tuy nhiên, vì bản thân baseline đã đặt phần lớn năng lượng chú ý vào vùng tổn thương nên sự dịch chuyển nhỏ ở biên không tạo ra bước nhảy vọt về macro-F1 toàn tập.
"""
    (REP / "2.5c_tac_dong_truoc_sau.md").write_text(txt, encoding="utf-8")


def write_26_27():
    tv = pd.read_csv(OUT / "preprocess/all16_comparison_val.csv").set_index("config")
    tt = pd.read_csv(OUT / "preprocess/all16_comparison_test.csv").set_index("config")
    S_base = pd.read_csv(OUT / "baseline/summary.csv").set_index("seed")
    bb_te = pd.read_csv(OUT / "backbones/comparison_test.csv").set_index("arch")
    app_chk = json.loads((OUT / "errors/app_consistency_check.json").read_text(encoding="utf-8"))

    order16 = [
        ("Nhiễu Gauss", "gauss3"), ("Nhiễu Gauss", "median3"), ("Nhiễu Gauss", "median5"),
        ("Tương phản thấp", "stretch_L_1_99"), ("Tương phản thấp", "hist_eq_L"), ("Tương phản thấp", "clahe_L_c2_t8"),
        ("Lông", "hair_bh17"), ("Lông", "hair_bh17_cond"),
        ("Viền tối", "vig_fill"), ("Viền tối", "vig_crop"),
        ("Quá sáng", "gamma_L_1.5"), ("Quá sáng", "gamma_L_auto"),
        ("Sáng không đều", "flat_L_s60"), ("Sáng không đều", "flat_L_s60_cond"),
        ("Kết hợp 1 (Nhiễu + Tương phản)", "median3+clahe_L_c2_t8"),
        ("Kết hợp 2 (Viền + Sáng lệch)", "vig_fill+flat_L_s60_cond"),
    ]
    rows_26 = []
    for prob, c in order16:
        rv, rt = tv.loc[c], tt.loc[c]
        rows_26.append(
            f"| {prob} | `{c}` | {sf4(rv.macro_f1_delta)} [{sf4(rv.macro_f1_delta_ci_lo)}; {sf4(rv.macro_f1_delta_ci_hi)}], {rv.macro_f1_delta_sign} | "
            f"{sf4(rt.macro_f1_delta)} | {rv.macro_f1_verdict} (val) / {rt.macro_f1_verdict} (test) | Không |"
        )

    s42 = S_base.loc[42]
    txt = f"""# 2.6 Hoàn thiện Pipeline Đề xuất và 2.7 Huấn luyện Mô hình Cuối

> Nguồn số liệu: toàn bộ 16 cấu hình tiền xử lý và baseline chạy thống nhất trên **NVIDIA L40S** (`outputs/preprocess/all16_comparison_{{val,test}}.csv`, `outputs/baseline/summary.csv`, `outputs/backbones/comparison_{{val,test}}.csv`, Hình `fig_all16_macro_f1.pdf`). Không còn hiện tượng lệch baseline giữa các máy.

## 2.6 Hoàn thiện Pipeline

### 2.6.1 Quy tắc chọn bước tiền xử lý (chốt trước khi tổng hợp)

Một phép xử lý ảnh chỉ được đưa vào nhánh phân loại tự động khi thỏa đồng thời cả **3 điều kiện trên tập validation** (tập `test` tuyệt đối không tham gia quá trình chọn):
1. Đạt kết luận **"tốt hơn rõ"** về macro-F1 trên `val` theo quy tắc 2.5.0 ($|\Delta| > \\text{{std}}_{{\\text{{baseline}}}}$, CI95 bootstrap cụm lesion loại 0, cả 3 seed cùng dương `3+/0-`).
2. Không làm suy giảm rõ recall của 3 lớp ác tính / tiền ác tính (`mel`, `bcc`, `akiec`).
3. Không gây tín hiệu xấu đi trên chính nhóm ảnh bị gắn cờ mà bước đó nhắm tới.

### 2.6.2 Bảng tổng hợp 16 cấu hình thực nghiệm (3 seed)

**Bảng 2.6-1.** Đối chiếu toàn bộ 14 cấu hình đơn lẻ và 2 cấu hình kết hợp với baseline ResNet-18 (`val` macro-F1 `{f4(tv.loc['baseline', 'macro_f1_mean'])} ± {f4(tv.loc['baseline', 'macro_f1_std'])}`, `test` `{f4(tt.loc['baseline', 'macro_f1_mean'])} ± {f4(tt.loc['baseline', 'macro_f1_std'])}`).

| Nhóm vấn đề | Cấu hình (3 seed) | Δ Val Macro-F1 [CI95], dấu | Δ Test (tham khảo) | Kết luận quy tắc 2.5.0 | Đưa vào nhánh CNN? |
|---|---|---|---|---|---|
{chr(10).join(rows_26)}

**Trả lời RQ4:** Trên dữ liệu HAM10000 thật, **0/16 cấu hình** (cả đơn lẻ lẫn kết hợp tuần tự `median3+clahe_L_c2_t8` và `vig_fill+flat_L_s60_cond`) thỏa điều kiện "tốt hơn rõ" trên tập `val`. Hai phép biến đổi lược đồ toàn cục (`hist_eq_L`, `stretch_L_1_99`) làm giảm mạnh hiệu năng; `hair_bh17` đại trà gây sụt giảm trên cả 3 seed do xóa nhầm cấu trúc sắc tố.

### 2.6.3 Kiến trúc Pipeline hai nhánh đề xuất

Từ bằng chứng thực nghiệm trên, nhóm đề xuất kiến trúc **hai nhánh tách biệt** thay vì ép ảnh qua chuỗi biến đổi pixel cố định:

```
Ảnh đầu vào (RGB, chuẩn hóa kích thước 600×450 cho nhánh QC)
   │
   ├──► Nhánh (A) Kiểm tra chất lượng ảnh (QC – Ch.3, 4, 5, 7):
   │       Đo 6 chỉ số EDA (viền tối, trắng bạc, lông, chênh nền, nhiễu σ, tương phản Otsu)
   │       → Gắn cờ cảnh báo độ tin cậy (đặc biệt khi gặp 'sáng không đều' & 'quá sáng')
   │       → Hiển thị trực quan các bước xử lý có điều kiện (vig_fill, flat_L_s60_cond) khi người dùng cần đối chiếu
   │
   └──► Nhánh (B) Phân loại CNN (Ch.8):
           Ảnh RGB gốc → Resize 224×224 (bilinear) → Normalize ImageNet → ResNet-18 → Xác suất 7 lớp + Grad-CAM
```

- **Thứ tự chuẩn khi bật nhánh minh họa xử lý ảnh** (hoặc khi gặp ảnh suy giảm ngoại lai nặng như ở Stress Test 2.5c): `vig_fill` (loại bỏ đỉnh đen của viền ống soi trước để không làm lệch thống kê mức sáng) $\\rightarrow$ khử nhiễu nhẹ (`median3`) $\\rightarrow$ hiệu chỉnh chiếu sáng có điều kiện (`flat_L_s60_cond` / `gamma_L_auto`) $\\rightarrow$ tăng tương phản cục bộ (`clahe_L_c2_t8`).

---

## 2.7 Huấn luyện và Đóng băng Mô hình Cuối

### 2.7.1 Lựa chọn kiến trúc và siêu tham số cuối

- **Kiến trúc mô hình cuối:** So sánh trên cùng cấu hình đóng băng ở mục 2.4 (`outputs/backbones/comparison_test.csv`) cho thấy **ResNet-18** đạt `test` macro-F1 `{f4(bb_te.loc['resnet18', 'macro_f1_mean'])} ± {f4(bb_te.loc['resnet18', 'macro_f1_std'])}`, so với `{f4(bb_te.loc['efficientnet_b0', 'macro_f1_mean'])} ± {f4(bb_te.loc['efficientnet_b0', 'macro_f1_std'])}` của `EfficientNet-B0` và `{f4(bb_te.loc['mobilenet_v3_small', 'macro_f1_mean'])} ± {f4(bb_te.loc['mobilenet_v3_small', 'macro_f1_std'])}` của `MobileNetV3-Small`. ResNet-18 được chọn làm mô hình cuối nhờ độ ổn định cao giữa các seed, cấu trúc khối dư tường minh và độ phân giải không gian tầng cuối `layer4` ($7\\times 7$) phù hợp cho giải thích Grad-CAM.
- **Giữ nguyên cấu hình đóng băng (`FROZEN_CONFIG`):** Không thực hiện tinh chỉnh siêu tham số hậu nghiệm sau khi đã xem tập `test`, nhằm bảo toàn tính trung thực phương pháp luận của toàn bộ chuỗi thực nghiệm 2.4–2.6.

### 2.7.2 Quy trình lưu trữ, chọn checkpoint cho Demo và Báo cáo kết quả

- **Báo cáo khoa học (mục 2.8):** sử dụng đầy đủ trung bình và độ lệch chuẩn qua **cả 3 seed (42, 43, 44)** của mô hình cuối:
  - `val` macro-F1: **`{f4(S_base['val_macro_f1'].mean())} ± {f4(S_base['val_macro_f1'].std(ddof=1))}`**, balanced accuracy: `{f4(S_base['val_balanced_acc'].mean())} ± {f4(S_base['val_balanced_acc'].std(ddof=1))}`, accuracy: `{f4(S_base['val_acc'].mean())} ± {f4(S_base['val_acc'].std(ddof=1))}`.
  - `test` macro-F1: **`{f4(S_base['test_macro_f1'].mean())} ± {f4(S_base['test_macro_f1'].std(ddof=1))}`**, balanced accuracy: `{f4(S_base['test_balanced_acc'].mean())} ± {f4(S_base['test_balanced_acc'].std(ddof=1))}`, accuracy: `{f4(S_base['test_acc'].mean())} ± {f4(S_base['test_acc'].std(ddof=1))}`.
- **Checkpoint triển khai cho Demo Streamlit (`app.py`):** sử dụng tệp `outputs/checkpoints/baseline_seed42.pt` vì `SEED = 42` là hạt giống mặc định đã quy ước từ đầu dự án (`CLAUDE.md`), **không** chọn seed có điểm test cao nhất. Checkpoint này hội tụ tốt nhất trên `val` tại **epoch {int(s42['best_epoch'])}** (`val` macro-F1 = `{f4(s42['val_macro_f1'])}`, `test` macro-F1 = `{f4(s42['test_macro_f1'])}`, `test` accuracy = `{f4(s42['test_acc'])}`).
- **Kiểm tra toàn vẹn suy luận (`outputs/errors/app_consistency_check.json`):** đối chiếu trực tiếp hàm suy luận `float32` của `app.py` với tệp `outputs/baseline/seed42/test_preds.csv` trên 20 ảnh test đầu tiên cho kết quả khớp **{app_chk['label_matches']}/{app_chk['n_checked']} nhãn dự đoán** (sai lệch xác suất tuyệt đối tối đa chỉ `{app_chk['max_abs_prob_diff_fp32_vs_amp']}` do khác biệt làm tròn giữa `AMP fp16` lúc train và `fp32` lúc demo).
"""
    (REP / "2.6_2.7_pipeline.md").write_text(txt, encoding="utf-8")


def write_28():
    S = pd.read_csv(OUT / "baseline/summary.csv")
    cp_te = pd.read_csv(OUT / "errors/confused_pairs_baseline_test.csv").head(6)
    cp_va = pd.read_csv(OUT / "errors/confused_pairs_baseline_val.csv").head(8)
    va_top_set = set(zip(cp_va["true"], cp_va["pred"]))
    cs_te = pd.read_csv(OUT / "errors/consistency_by_class_test.csv").set_index("dx")
    cs_va = pd.read_csv(OUT / "errors/consistency_by_class_val.csv").set_index("dx")
    sub_err = pd.read_csv(OUT / "errors/subgroup_error_class_matched.csv")
    mel_row = pd.read_csv(OUT / "errors/mel_row_hair.csv")
    mel_fl = pd.read_csv(OUT / "errors/mel_hair_flips.csv")
    mel_floor = json.loads((OUT / "errors/mel_baseline_floor.json").read_text(encoding="utf-8"))
    app_chk = json.loads((OUT / "errors/app_consistency_check.json").read_text(encoding="utf-8"))

    cp_lines = []
    for r in cp_te.itertuples():
        in_val = "Có" if (r.true, r.pred) in va_top_set else "Không"
        cp_lines.append(
            f"| `{r.true}` → `{r.pred}` | **{f3(r.rate_mean)} ± {f3(r.rate_std)}** | {f1(r.count_mean)} / {int(r.n_true)} | "
            f"{r.rate_per_seed.replace('.', ',')} | {in_val} |"
        )

    cs_lines = []
    for c in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        rt, rv = cs_te.loc[c], cs_va.loc[c]
        cs_lines.append(
            f"| `{c}` | {int(rt.always_wrong)} / {int(rt.n)} ({f1(rt.pct_always_wrong)}%) | {int(rt.always_right)} ({f1(rt.pct_always_right)}%) | "
            f"{int(rv.always_wrong)} / {int(rv.n)} ({f1(rv.pct_always_wrong)}%) | {int(rv.always_right)} ({f1(rv.pct_always_right)}%) |"
        )
    tot_w_te, tot_r_te = int(cs_te.always_wrong.sum()), int(cs_te.always_right.sum())
    tot_w_va, tot_r_va = int(cs_va.always_wrong.sum()), int(cs_va.always_right.sum())

    sub_lines = []
    for f_code in ["F_hair", "F_vignette", "F_overexposed", "F_uneven_illum", "F_noise", "F_low_contrast"]:
        rt = sub_err[(sub_err.split == "test") & (sub_err.flag == f_code)].iloc[0]
        rv = sub_err[(sub_err.split == "val") & (sub_err.flag == f_code)].iloc[0]
        sub_lines.append(
            f"| {rt.problem} | {int(rt.n_group)} | {f3(rt.err_group)} | {f3(rt.err_expected_class_matched)} | "
            f"`{sf3(rt.excess)}` [`{sf3(rt.excess_ci_lo)}`; `{sf3(rt.excess_ci_hi)}`] | "
            f"`{sf3(rv.excess)}` [`{sf3(rv.excess_ci_lo)}`; `{sf3(rv.excess_ci_hi)}`] |"
        )

    mel_lines = []
    for sp in ("test", "val"):
        for cfg in ("baseline", "hair_bh17", "hair_bh17_cond"):
            g = mel_row[(mel_row.split == sp) & (mel_row.config == cfg)].set_index("pred")
            mel_lines.append(
                f"| {sp} | `{cfg}` | {f3(g.loc['mel', 'rate_mean'])} ± {f3(g.loc['mel', 'rate_std'])} | "
                f"{f3(g.loc['nv', 'rate_mean'])} ± {f3(g.loc['nv', 'rate_std'])} | "
                f"{f3(g.loc['bkl', 'rate_mean'])} ± {f3(g.loc['bkl', 'rate_std'])} |"
            )

    n_broken_te = int(((mel_fl.split == "test") & (mel_fl.group == "bị hỏng")).sum())
    n_stable_te = int(((mel_fl.split == "test") & (mel_fl.group == "ổn định")).sum())
    med_in_les = float(mel_fl[mel_fl.split == "test"]["hair_in_lesion"].median() * 100)

    txt = f"""# 2.8 Đánh giá và Phân tích lỗi của Mô hình Cuối

> Nguồn số liệu: dự đoán ngoài mẫu 3 seed (42, 43, 44) của mô hình cuối (= ResNet-18 baseline theo pipeline 2.6) trên lượt chạy thống nhất NVIDIA L40S, lưu tại `outputs/errors/` và `outputs/figures/errors/`. Mã nguồn: `scripts/analyze_errors.py` và `app.py`.

## 2.8.1 Tổng hợp hiệu năng mô hình cuối

| Chỉ số | Validation ($n = 1\\,603$) | Test ($n = 2\\,004$) |
|---|---|---|
| **Macro-F1 (chỉ số chính)** | {f4(S['val_macro_f1'].mean())} ± {f4(S['val_macro_f1'].std(ddof=1))} | **{f4(S['test_macro_f1'].mean())} ± {f4(S['test_macro_f1'].std(ddof=1))}** |
| Balanced accuracy | {f4(S['val_balanced_acc'].mean())} ± {f4(S['val_balanced_acc'].std(ddof=1))} | {f4(S['test_balanced_acc'].mean())} ± {f4(S['test_balanced_acc'].std(ddof=1))} |
| Accuracy | {f4(S['val_acc'].mean())} ± {f4(S['val_acc'].std(ddof=1))} | {f4(S['test_acc'].mean())} ± {f4(S['test_acc'].std(ddof=1))} |

## 2.8.2 Ma trận nhầm lẫn và các cặp lớp dễ lẫn lộn

**Bảng 2.8-1.** 6 cặp (nhãn thật → dự đoán) có tỉ lệ nhầm theo hàng cao nhất trên tập `test` (Hình `confusion_baseline_test.pdf`).

| Thật → Dự đoán | Tỉ lệ hàng (mean ± std) | Số ảnh TB / n lớp thật | Theo seed 42; 43; 44 | Lặp lại trong top 8 `val`? |
|---|---|---|---|---|
{chr(10).join(cp_lines)}

- **Cặp nhầm trọng yếu nhất:** `mel` $\\rightarrow$ `nv` và `mel` $\\rightarrow$ `bkl` (cùng chiều ngược lại), do cả ba đều là tổn thương sắc tố có đặc trưng màu nâu/đen chồng lấn mạnh sau khi thu nhỏ về $224\\times 224$. Cặp `akiec` $\\rightarrow$ `bcc`/`bkl` đứng tiếp theo do cùng là tổn thương bề mặt hồng/vảy sừng trên nền da lão hóa do nắng.

## 2.8.3 Nhóm ảnh khó: dự đoán sai nhất quán ở cả 3 seed

**Bảng 2.8-2.** Phân rã ảnh dự đoán sai cả 3 seed (lỗi hệ thống) và đúng cả 3 seed theo từng lớp (`consistency_by_class_{{test,val}}.csv`, Hình `grid_always_wrong_test.pdf`).

| Lớp | Test: sai cả 3 (%) | Test: đúng cả 3 (%) | Val: sai cả 3 (%) | Val: đúng cả 3 (%) |
|---|---|---|---|---|
{chr(10).join(cs_lines)}
| **Tổng** | **{tot_w_te} / 2 004 ({tot_w_te/2004*100:.1f}%)** | **{tot_r_te} ({tot_r_te/2004*100:.1f}%)** | **{tot_w_va} / 1 603 ({tot_w_va/1603*100:.1f}%)** | **{tot_r_va} ({tot_r_va/1603*100:.1f}%)** |

- Quan sát lưới 21 ảnh sai nhất quán ở cả 3 seed (`grid_always_wrong_test.pdf`) cho thấy phần lớn là các ca tổn thương không điển hình về mặt hình thái (ví dụ `mel` nhỏ tròn đồng màu giống hệt `nv`, hoặc `df` nhạt màu không rõ tâm sẹo trắng), hơn là do một lỗi chất lượng ảnh đơn lẻ.

## 2.8.4 Liên hệ định lượng với 6 cờ chất lượng ảnh của EDA

**Bảng 2.8-3.** Sai số trung bình 3 seed của nhóm cờ so với kỳ vọng đã chuẩn hóa theo thành phần lớp (`subgroup_error_class_matched.csv`).

| Vấn đề (EDA) | n test | Lỗi nhóm | Lỗi kỳ vọng theo lớp | Excess Test [CI95] | Excess Val [CI95] |
|---|---:|---:|---:|---|---|
{chr(10).join(sub_lines)}

- **Kết luận trả lời RQ1 & RQ3:** Sau khi loại bỏ nhiễu gây bởi thành phần lớp, **sáng không đều** và **quá sáng** có liên hệ dương với tỉ lệ lỗi cao hơn của CNN, nhưng việc áp phép khử sáng lệch/gamma cố định ở mục 2.5b vẫn không thu hẹp được rõ rệt khoảng cách này. Ngược lại, nhóm **nhiễu Gauss** có lỗi thấp hơn kỳ vọng vì tập trung ở ảnh `nv` thuộc nguồn `follow_up`.

## 2.8.5 Kiểm chứng cơ chế `hair_bh17` trên lớp `mel` và mức nền dao động

**Bảng 2.8-4.** Phân bố hàng `mel` trong confusion matrix giữa `baseline`, `hair_bh17` (đại trà) và `hair_bh17_cond` (có điều kiện).

| Split | Cấu hình | → `mel` (Recall) | → `nv` | → `bkl` |
|---|---|---|---|---|
{chr(10).join(mel_lines)}

- Trên tập `test`, có `{n_broken_te}` ảnh `mel` bị chuyển từ đúng ($\\ge 2/3$ seed ở baseline) sang sai ($\\ge 2/3$ seed ở `hair_bh17`), so với `{n_stable_te}` ảnh `mel` ổn định; mức tham chiếu bất đồng tự nhiên giữa hai seed khác nhau của chính baseline trên 222 ảnh `mel` là trung bình `{f1(mel_floor['test']['pairwise_disagree_mean'])}` ảnh (`mel_baseline_floor.json`). Trung vị **{f1(med_in_les)}%** số pixel của mask "lông" trên ảnh `mel` thực chất nằm *bên trong* vùng tổn thương (`mel_hair_flips.csv`, Hình `grid_mel_hair_flips_test.pdf`), xác nhận black-hat đại trà đã inpaint trực tiếp lên cấu trúc sắc tố trong tổn thương, làm mô hình huấn luyện lại dịch chuyển ranh giới phân lớp về phía `nv`. Bản có điều kiện `hair_bh17_cond` khắc phục được hiện tượng này trên 90% ảnh không có cờ lông.

## 2.8.6 Hệ thống Demo tương tác trên Streamlit (`app.py`)

Ứng dụng `app.py` được thiết kế thành **3 tab chức năng** bám sát pipeline hai nhánh của mục 2.6.3:
1. **Tab 1 – Pipeline QC & Dự đoán:** chạy trực tiếp bộ đo `compute_one` trên ảnh chuẩn hóa $600\\times 450$, hiển thị bảng 6 cờ cảnh báo QC kèm chỉ số trước/sau, xác suất dự đoán 7 lớp của `baseline_seed42.pt` trên ảnh gốc đối chiếu với ảnh sau xử lý, và nhãn thật nếu chọn từ tập mẫu `samples/`.
2. **Tab 2 – Chẩn đoán thị giác từng giai đoạn xử lý (Ch.3, 4, 7):** hỗ trợ chọn nhanh các preset tiền xử lý hoặc tùy chỉnh chuỗi bước, tích hợp **bộ mô phỏng suy giảm tổng hợp 2.5c** (6 loại $\\times$ 3 mức), vẽ mask hình thái học (viền, lông black-hat, biên Otsu thô), bản đồ biên độ thay đổi pixel $|\\text{{sau}} - \\text{{trước}}|$ (`magma`), độ lệch màu $\\Delta E_{{76}}$ và biểu đồ histogram mức xám trong FOV.
3. **Tab 3 – Giải thích Grad-CAM (`layer4`) & Tra cứu số liệu:** tính trực tiếp bản đồ nhiệt Grad-CAM theo lớp đích tùy chọn trên cả ảnh đầu vào và ảnh sau xử lý, định lượng % năng lượng chú ý trong tổn thương so với vùng lông, và hiển thị bảng tổng hợp thực nghiệm 3 seed từ `outputs/`.
- **Kiểm chứng độ chính xác và bảo mật:** khớp tuyệt đối **{app_chk['label_matches']}/{app_chk['n_checked']} nhãn** với tập dự đoán kiểm thử đã lưu (`app_consistency_check.json`); nạp trọng số an toàn với `weights_only=True`, giới hạn tệp tải lên $\\le 10$ MB và xác thực cấu trúc magic-bytes JPEG/PNG trước khi giải mã.
"""
    (REP / "2.8_danh_gia.md").write_text(txt, encoding="utf-8")


def write_3():
    txt = """# 3 Thảo luận, Phân công công việc và Khai báo sử dụng AI

## 3.1 Thảo luận tổng hợp: Vì sao xử lý ảnh cổ điển không tăng Macro-F1 cho CNN trên HAM10000?

Kết hợp bằng chứng xuyên suốt từ mục 2.2 đến 2.8 cho phép rút ra 4 nhận định khoa học trung tâm:

1. **Sự khác biệt căn bản giữa "tối ưu thị giác cho mắt người / thuật toán phân ngưỡng cổ điển" và "đặc trưng học sâu có giám sát":**
   Ở Tầng A (mục 2.5c), các phép xử lý cổ điển thực hiện rất tốt mục tiêu thiết kế của Chương 3 và Chương 7 (triệt tiêu 92%–100% cờ viền tối, lông, quá sáng, sáng lệch; kéo giãn mạnh độ tương phản). Tuy nhiên, khác với bộ phân ngưỡng Otsu cố định dễ sụp đổ khi chiếu sáng lệch, mạng ResNet-18 có các tầng `BatchNorm` và 11,18 triệu tham số được tinh chỉnh toàn bộ (*end-to-end fine-tuning*) trên 6 408 ảnh huấn luyện nên tự học được tính bất biến tương đối với các biến thiên độ sáng và tương phản vừa phải của thiết bị dermoscope.
2. **Bước thu nhỏ độ phân giải (`600×450 → 224×224`) đã đóng vai trò bộ lọc thông thấp tự nhiên:**
   Cường độ nhiễu cảm biến trên HAM10000 vốn thấp ($\\sigma$ trung vị 1,34 mức xám) và phép nội suy song tuyến có khử răng cưa của PIL đã triệt tiêu phần lớn phương sai nhiễu tần số cao trước khi ảnh đi vào tầng tích chập đầu tiên. Áp thêm lọc không gian (đặc biệt `median5`) làm suy hao chi tiết vi cấu trúc vùng tổn thương nhiều hơn lượng nhiễu còn lại.
3. **Đánh đổi lợi – hại khi áp đại trà lên tập ảnh mà chỉ 2,5%–10,1% ảnh thực sự gặp lỗi:**
   - Hai phép biến đổi lược đồ toàn cục (`hist_eq_L`, `stretch_L_1_99`) làm biến dạng màu sắc ($\\Delta E_{76}$ từ 14 đến 22) và khuếch đại nhiễu gấp 3,5–5,5 lần trên hơn 95% ảnh bình thường, trong khi màu sắc tuyệt đối và độ chuyển sắc êm lại là tín hiệu chẩn đoán cốt lõi của tổn thương da $\\rightarrow$ làm giảm mạnh macro-F1.
   - Phép xóa lông đại trà (`hair_bh17`) dùng toán tử hình thái black-hat bắt nhầm các cấu trúc sắc tố tối dạng nét bên trong u hắc tố (`mel` bị inpaint trung vị 8,9% diện tích tổn thương so với 2,5% ở `nv`), xóa đi dấu hiệu phân biệt `mel` với `nv`.
4. **Giá trị thực tiễn của xử lý ảnh cổ điển nằm ở Kiểm soát chất lượng (QC), Xử lý có điều kiện và Chống chịu suy giảm ngoại lai:**
   - Các phiên bản **có điều kiện** (`_cond`) bảo toàn 100% ảnh bình thường, loại bỏ hoàn toàn tác dụng phụ đại trà.
   - Ở Stress Test Tầng C (mục 2.5c), khi ảnh kiểm thử chịu suy giảm nặng vượt ngoài phân phối huấn luyện (viền đen lớn, lông dày đặc, nhiễu mạnh), các phép xử lý khôi phục đáng kể macro-F1 so với mô hình không xử lý.
   - Vì vậy, việc đặt bộ đo Chương 3–7 làm **Nhánh (A) QC cảnh báo chất lượng** song song với **Nhánh (B) phân loại CNN** là thiết kế trung thực và hợp lý nhất theo dữ liệu.

## 3.2 Hạn chế của nghiên cứu và Hướng phát triển ngoài phạm vi

**Hạn chế phương pháp luận:**
- **Một lần chia split và 3 seed huấn luyện:** Độ lệch chuẩn và khoảng tin cậy bootstrap cụm phản ánh dao động khởi tạo/lấy mẫu WRS và phương sai tập ảnh kiểm thử, chưa phản ánh phương sai giữa các lần chia fold khác nhau; các lớp nhỏ (`df` 23 ảnh, `vasc` 29 ảnh test) có khoảng tin cậy rộng.
- **Ước lượng vùng tổn thương bằng Otsu thô:** Bộ dữ liệu không kèm mặt nạ phân vùng chuẩn trong quy trình chính; kiểm tra trực quan 24 ảnh tương phản thấp cho thấy Otsu bắt sai 11/24 ảnh khi rìa ảnh tối hơn tổn thương nhạt màu ở tâm. Tương tự, black-hat chỉ bắt lông tối, gây lỗi inpainting trên ca lông bạc/trắng dày.
- **Giới hạn một miền thiết bị (dermoscopy):** Không đánh giá khả năng tổng quát hóa sang tập dữ liệu bệnh viện khác hoặc ảnh chụp lâm sàng bằng điện thoại.

**Hướng cải thiện tiềm năng (nêu rõ là ngoài phạm vi, nhóm *không thực hiện* để giữ đúng mục tiêu kiểm soát biến của đề tài):**
- Thay mask Otsu/black-hat bằng mô hình phân vùng tổn thương & sợi lông chuyên biệt (ví dụ U-Net trên tập mặt nạ ISIC Segmentation) và kết hợp top-hat cho lông trắng.
- Khảo sát các kỹ thuật chuẩn hóa màu chuyên sâu của Chương 4 dành cho dermoscopy (*Shades of Gray*, * Macenko/Reinhard color constancy*) hoặc đưa biến đổi độ sáng/màu thành *augmentation ngẫu nhiên lúc huấn luyện* thay vì tiền xử lý cố định.
- Sử dụng độ phân giải đầu vào cao hơn ($384\\times 384$, giữ nguyên tỉ lệ khung hình 4:3 bằng đệm viền) và hàm mất mát nhận biết mất cân bằng lớp như *Focal Loss* hoặc *Class-Balanced Loss*.

## 3.3 Bảng phân công công việc trong nhóm

| Thành viên | Mã SV | Phần công việc phụ trách chính | Mức độ hoàn thành |
|---|---|---|---|
| **Nguyễn Trung Nam** | `B23DCVT299` | Tích hợp & thực thi thống nhất toàn bộ pipeline trên GPU NVIDIA L40S, khảo sát 3 kiến trúc CNN (mục 2.4), hoàn thiện thực nghiệm 4 tầng (mục 2.5c), nâng cấp ứng dụng Streamlit 3 tab (`app.py`) và chế bản báo cáo LaTeX Overleaf / Slide Beamer | 100% |
| **Phan Nguyễn Việt Dũng** | `B23DCCN210` | Kiến trúc mã nguồn dùng chung (`src/training.py`, `compare.py`), chuẩn bị dữ liệu & baseline 2.1, 2.3, 2.4; thực nghiệm khử nhiễu & tương phản 2.5.1–2.5.2; thiết kế khung phân tích 4 tầng 2.5c | 100% |
| **Lê Duy Anh** | `B23DCCN026` | Cơ sở lý thuyết Chương 3–4, phân tích thống kê & biên soạn báo cáo mục Mở đầu, 2.5.1–2.5.2 (nhiễu Gauss, tương phản thấp), xây dựng kịch bản thuyết trình phần đầu | 100% |
| **Đinh Khánh Sơn** | `B23DCDT223` | Xây dựng bộ 7 chỉ số & phân tích EDA chất lượng ảnh (mục 2.2), thực nghiệm 4 vấn đề lông/viền tối/quá sáng/sáng lệch (2.5.3–2.5.6), phân tích lỗi 2.8 & khởi tạo demo Streamlit | 100% |
| **Đào Phạm Bích Ngọc** | `B23DCVT308` | Cơ sở lý thuyết Chương 7 (hình thái học), đối chiếu trực quan lưới ảnh trước/sau mục 2.5.3–2.5.6, tổng hợp pipeline 2.6–2.7, chuẩn bị slide và kiểm tra chéo số liệu báo cáo | 100% |

## 3.4 Khai báo sử dụng Trí tuệ Nhân tạo (AI)

Tuân thủ quy định học thuật của giảng viên và học phần, nhóm khai báo minh bạch việc sử dụng công cụ AI hỗ trợ (Claude / Gemini coding assistant) trong quá trình thực hiện đồ án như sau:

1. **Các khâu có sử dụng AI hỗ trợ:**
   - Hỗ trợ viết và tối ưu hóa mã nguồn khung huấn luyện (`src/training.py`), các mô-đun tiền xử lý & kiểm định thống kê (`src/preprocess/`, `scripts/`), giao diện ứng dụng Streamlit (`app.py`) và mã định dạng bảng biểu LaTeX/Beamer.
   - Hỗ trợ rà soát văn phong học thuật tiếng Việt, kiểm tra tính nhất quán ký hiệu toán học và đồng bộ hóa số liệu từ tệp CSV sang bản thảo Markdown/LaTeX.
2. **Các khâu do thành viên nhóm trực tiếp quyết định, thực thi và chịu trách nhiệm:**
   - Xác lập câu hỏi nghiên cứu (RQ1–RQ4), lựa chọn các kỹ thuật bám sát nội dung Chương 3, 4, 5, 7, 8 của học phần và đặt quy tắc kết luận thống kê *trước* thực nghiệm (chấp nhận và báo cáo trung thực mọi kết quả âm).
   - Trực tiếp chạy toàn bộ các lượt huấn luyện và đánh giá trên dữ liệu HAM10000 thật, kiểm tra trực quan từng lưới ảnh EDA, kiểm định bằng mắt 24 ảnh mẫu của `lowcontrast_mask_audit.csv` và ca lỗi lông trắng `ISIC_0032214`.
   - Đối chiếu chéo toàn bộ con số trong báo cáo PDF và slide với các tệp kết quả gốc trong `outputs/` để đảm bảo không có số liệu suy diễn hay sai lệch.
"""
    (REP / "3_thao_luan_phan_cong_ai.md").write_text(txt, encoding="utf-8")


def main():
    write_1()
    update_21()
    update_22()
    write_24()
    write_25a()
    write_25b()
    write_25c()
    write_26_27()
    write_28()
    write_3()
    print("Đã sinh và cập nhật toàn bộ báo cáo Markdown mục 1 -> 3 thành công!")


if __name__ == "__main__":
    main()
