# TODO

> Cập nhật 10/11/2026 trên `main`. Hạn nộp đồ án: **11/11/2026**.
> **Trạng thái:** Đã hoàn tất 100% toàn bộ thực nghiệm thống nhất trên NVIDIA L40S, báo cáo các mục 1–3, Overleaf LaTeX PDF, Slide Beamer PDF và ứng dụng Demo Streamlit.

## 1. Thực nghiệm & Đồng bộ Số liệu (Hoàn tất 100%)

- [x] Chạy lại thống nhất trên một phần cứng duy nhất (**NVIDIA L40S**, PyTorch 2.13+cu126) toàn bộ `baseline` + 16 cấu hình tiền xử lý × 3 seed (42, 43, 44): xóa bỏ hoàn toàn tình trạng lệch 4 bộ baseline giữa các máy cũ.
- [x] Bổ sung khảo sát 3 kiến trúc CNN (`ResNet-18`, `EfficientNet-B0`, `MobileNetV3-Small` × 3 seed) tại `scripts/run_backbone_comparison.py`.
- [x] **Hoàn tất trọn vẹn 4 tầng mục 2.5c:**
  - [x] Tầng A: hiệu ứng chỉ số ảnh & tác dụng phụ trên đối chứng (`outputs/impact/image_level_summary.csv`, `hair_lesion_tests.csv`).
  - [x] Tầng B: tỉ lệ đổi nhãn dự đoán so với mức nền khác seed (`outputs/impact/flips_excess.csv`).
  - [x] Tầng C: Stress test 48 điều kiện suy giảm tổng hợp 3 nhánh (`outputs/impact/stress_delta.csv`).
  - [x] Tầng D: Grad-CAM `layer4` năng lượng vùng chú ý & tương đồng bản đồ nhiệt (`outputs/impact/gradcam_summary.csv`, `gradcam_similarity.csv`).
- [x] Kiểm tra trực quan từng khung trên 24 ảnh tương phản thấp (`outputs/preprocess/lowcontrast_mask_audit.csv`: 8 đúng, 5 một phần, 11 sai).
- [x] Điền đầy đủ cột "Liên quan lỗi CNN?" trong `outputs/eda/eda_conclusion.csv`, mục 5 notebook `2.2_eda_chat_luong_anh.ipynb` và báo cáo `2.2_eda_chat_luong_anh.md`.
- [x] Kiểm chứng nguồn `follow_up` trong lớp `nv` (`outputs/errors/nv_followup_check.csv`: lỗi 0,75% ở `follow_up` so với 17,04% ở `histo`) và mức nền bất đồng `mel` giữa các seed (`mel_baseline_floor.json`).
- [x] Thực thi lại và đồng bộ nhận xét cho toàn bộ 8 notebook gốc (`2.2`, `2.3_2.4`, `2.5.1`–`2.5.6`).

## 2. Báo cáo, Overleaf, Slide & Demo (Hoàn tất 100%)

- [x] Cập nhật và hoàn thiện toàn bộ báo cáo Markdown trong `reports/`: `1_gioi_thieu.md`, `2.1_du_lieu.md`, `2.2_eda_chat_luong_anh.md`, `2.3_chuan_bi_du_lieu.md`, `2.4_baseline.md`, `2.5a_nhieu_tuong_phan.md`, `2.5b_long_vien_sang.md`, `2.5c_tac_dong_truoc_sau.md`, `2.6_2.7_pipeline.md`, `2.8_danh_gia.md`, `3_thao_luan_phan_cong_ai.md`.
- [x] Xuất trọn bộ dự án Overleaf LaTeX tại `overleaf/` (`main.tex`, `references.bib`, `figures/`, gói tải lên 1-click `Overleaf_Upload_HAM10000.zip` và bản biên dịch `BaoCao_XuLyAnh_HAM10000.pdf`).
- [x] Xuất bộ Slide thuyết trình Beamer 16:9 tại `slides/` (`slides.tex`, `Slide_XuLyAnh_HAM10000.pdf`) kèm kịch bản thuyết trình nhóm `KichBan_ThuyetTrinh_Nhom.md`.
- [x] Nâng cấp và kiểm chứng bảo mật ứng dụng Streamlit `app.py` (3 tab: QC & Dự đoán, Từng giai đoạn xử lý + Stress test + Histogram/Mask, Grad-CAM trực tiếp; khớp 20/20 nhãn với tập dự đoán kiểm thử).

## 3. Thông tin Nhóm Tác giả & Hồ sơ Nộp (Hoàn tất 100%)

- [x] Đã điền đầy đủ họ tên và mã sinh viên của cả 5 thành viên (**Nguyễn Trung Nam – B23DCVT299**, **Phan Nguyễn Việt Dũng – B23DCCN210**, **Lê Duy Anh – B23DCCN026**, **Đinh Khánh Sơn – B23DCDT223**, **Đào Phạm Bích Ngọc – B23DCVT308**) vào `overleaf/main.tex`, `slides/slides.tex`, `reports/3_thao_luan_phan_cong_ai.md`, `app.py` và `README.md`.

## 4. Phản biện học thuật, Chuẩn hóa Diễn giải & Đóng gói Docker (Hoàn tất 100%)

- [x] Chuẩn hóa lập luận chọn kiến trúc `ResNet-18` dựa chặt chẽ trên tập `val` (`0,7254 ± 0,0044` vs `0,7333 ± 0,0146`, độ lệch chuẩn nhỏ hơn 3,3 lần, CI95 hiệu số chứa 0), ghi rõ điều kiện `micro_batch=16` tích lũy gradient ở `run_backbone_comparison.py` (tự động mở batch 32 khi VRAM $\ge 12$ GB và bật lại `cudnn.benchmark`).
- [x] Bổ sung giải thích định lượng cho các hiện tượng chuyên sâu trong báo cáo Markdown, Notebook, Overleaf PDF và Slide Beamer: cơ chế sụt giảm nhóm quá sáng của `gamma_L_auto` (`DiD = -0,209` do *conditional covariate shift* + bệt xám vùng lóa bão hòa), đánh đổi precision–recall của `hair_bh17_cond` trên lớp `mel` (`P: 0,588 → 0,502`, `R: 0,518 → 0,601`), mức nền đổi nhãn Tầng B (`15,34%` test, `+20,53%` excess trên cờ quá sáng), chênh lệch `test_only` (`+0,1943`) vs `retrain` (`-0,0073`) của `clahe_L_c2_t8` ở Stress Test Tầng C, và kiểm chứng ngưỡng Tukey riêng trên `train_fit`.
- [x] Nâng cấp `src/eda/quality_metrics.py` (`compute_from_rgb`, `analyze_rgb` thuần trên bộ nhớ RAM, không ghi tệp tạm ra đĩa), bổ sung cắt khung 4:3 tùy chọn cho ảnh tải lên, đóng gói `demo_assets/` cho bản clone sạch và hoàn thiện `Dockerfile` + `docker-compose.yml` + `.streamlit/config.toml` đạt chuẩn bảo mật (`appuser` non-root, `read_only` rootfs, `127.0.0.1:8501`).
