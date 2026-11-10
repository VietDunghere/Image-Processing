# Đánh giá tác động của xử lý ảnh cổ điển lên phân loại bệnh da bằng CNN

Đồ án học phần **Xử lý ảnh** – Khoa Trí tuệ Nhân tạo, PTIT (Lựa chọn 2: dự án thực tế). Đề tài đánh giá có kiểm soát các phương pháp nâng cao chất lượng ảnh cổ điển (**Chương 3**, kết hợp **Chương 4** hệ màu, **Chương 5** Otsu/Sobel, **Chương 7** hình thái học) trên bộ dữ liệu soi da lâm sàng **HAM10000** sử dụng mạng tích chập sâu (**Chương 8**), theo phương pháp luận **lập luận → thực nghiệm kiểm soát biến → đánh giá trung thực**.

> Chỉ phục vụ mục đích học tập và nghiên cứu học thuật. **Tuyệt đối không dùng để chẩn đoán lâm sàng.**

## Nhóm thực hiện (PTIT)

| Họ và tên | Mã sinh viên | Phụ trách chính |
|---|---|---|
| **Nguyễn Trung Nam** | `B23DCVT299` | Tích hợp thực nghiệm thống nhất trên NVIDIA L40S, khảo sát 3 kiến trúc CNN (2.4), hoàn thiện phân tích 4 tầng (2.5c), nâng cấp Demo Streamlit 3 tab & chế bản LaTeX/Beamer |
| **Phan Nguyễn Việt Dũng** | `B23DCCN210` | Khung huấn luyện & kiểm định dùng chung, chuẩn bị dữ liệu & baseline (2.1, 2.3, 2.4), thực nghiệm 2.5.1–2.5.2 & thiết kế 2.5c |
| **Lê Duy Anh** | `B23DCCN026` | Lý thuyết Chương 3–4, biên soạn báo cáo Mở đầu & mục 2.5.1–2.5.2 (nhiễu Gauss, tương phản thấp), kịch bản thuyết trình |
| **Đinh Khánh Sơn** | `B23DCDT223` | Bộ 7 chỉ số & phân tích EDA (2.2), thực nghiệm 4 vấn đề lông/viền/quá sáng/sáng lệch (2.5.3–2.5.6), phân tích lỗi (2.8) & khởi tạo Demo |
| **Đào Phạm Bích Ngọc** | `B23DCVT308` | Lý thuyết hình thái học Chương 7, đối chiếu trực quan ảnh trước/sau (2.5.3–2.5.6), tổng hợp pipeline (2.6–2.7), kiểm tra chéo số liệu & Slide |

## Kết quả chính (Lượt chạy thống nhất trên NVIDIA L40S, 3 seed: 42, 43, 44)

- **Baseline & So sánh kiến trúc CNN (mục 2.4):** Dưới cùng cấu hình đóng băng (`224×224`, `AdamW lr=1e-4`, batch 32, 20 epoch, `WeightedRandomSampler`), **ResNet-18** đạt validation macro-F1 **0,7254 ± 0,0044** và test macro-F1 **0,6941 ± 0,0160** (balanced accuracy `0,6777 ± 0,0194`, accuracy `0,8315 ± 0,0032`), vượt rõ `MobileNetV3-Small` (`0,6320 ± 0,0125`) và ổn định hơn `EfficientNet-B0` (`0,6732 ± 0,0238`).
- **Ablation 16 cấu hình tiền xử lý trên ResNet-18 (mục 2.5–2.6):** **0/16 cấu hình** (14 đơn lẻ + 2 kết hợp) đạt chuẩn *"tốt hơn rõ"* trên validation. Hai phép biến đổi lược đồ toàn cục (`hist_eq_L` Δ test `−0,1106`, `stretch_L_1_99` Δ test `−0,0786`) tệ hơn rõ do lệch màu $\Delta E_{76}$ và khuếch đại nhiễu; phép xóa lông đại trà (`hair_bh17` Δ test `−0,0363`) inpaint nhầm mạng sắc tố của u hắc tố (`mel`).
- **Giải mã cơ chế qua 4 tầng (mục 2.5c):** Dù cải thiện 92%–100% cờ chất lượng ảnh ở Tầng A, tiền xử lý không làm tăng F1 trên dữ liệu thật vì mạng CNN có `BatchNorm` được fine-tune toàn bộ đã tự thích nghi với biến thiên sáng nhẹ và bước resize `600→224` đã khử nhiễu mịn. Ngược lại, trong **Stress Test suy giảm ngoại lai nặng** (Tầng C), tiền xử lý khôi phục mạnh mẽ macro-F1 ngay ở nhánh `test_only` (`hair_bh17_cond` `+0,4623`, `clahe_L_c2_t8` `+0,1943`, `vig_fill` `+0,1020`).
- **Pipeline hai nhánh & Demo Streamlit (mục 2.6–2.8):** Tách **Nhánh (A) Kiểm tra chất lượng ảnh (QC)** gắn cờ cảnh báo (đặc biệt hiệu quả cho nhóm *sáng không đều* có lỗi cao hơn kỳ vọng `+13,0%`) song song với **Nhánh (B) Phân loại ResNet-18 trên ảnh gốc**, triển khai thành ứng dụng Streamlit 3 tab (`app.py`).

## Tài liệu bàn giao chính trong Repo

| Hạng mục | Đường dẫn | Mô tả |
|---|---|---|
| **Báo cáo PDF (Overleaf)** | [`overleaf/BaoCao_XuLyAnh_HAM10000.pdf`](overleaf/BaoCao_XuLyAnh_HAM10000.pdf) | Bản báo cáo khoa học hoàn chỉnh (biên dịch từ [`overleaf/main.tex`](overleaf/main.tex)); gói upload 1-click lên Overleaf tại [`overleaf/Overleaf_Upload_HAM10000.zip`](overleaf/Overleaf_Upload_HAM10000.zip) |
| **Slide thuyết trình PDF** | [`slides/Slide_XuLyAnh_HAM10000.pdf`](slides/Slide_XuLyAnh_HAM10000.pdf) | Bộ 14 slide Beamer 16:9 ([`slides/slides.tex`](slides/slides.tex)) kèm kịch bản thuyết trình nhóm [`slides/KichBan_ThuyetTrinh_Nhom.md`](slides/KichBan_ThuyetTrinh_Nhom.md) |
| **Báo cáo chi tiết từng mục** | [`reports/`](reports/) | Đầy đủ từ `1_gioi_thieu.md` đến `3_thao_luan_phan_cong_ai.md` (gồm cả `2.5c_tac_dong_truoc_sau.md`) |
| **8 Notebook thực nghiệm** | `2.2_*.ipynb`, `2.3_2.4_*.ipynb`, `2.5.1_*.ipynb`–`2.5.6_*.ipynb` | Đã chạy đồng bộ cùng một baseline trên NVIDIA L40S, có đầy đủ bảng và biểu đồ nhúng sẵn |
| **Demo tương tác Streamlit** | [`app.py`](app.py) | Giao diện 3 tab: (1) QC & Dự đoán, (2) Từng giai đoạn xử lý + Stress Test 2.5c + Histogram/Mask, (3) Giải thích Grad-CAM `layer4` |
| **Bản đồ mã nguồn & Quy tắc** | [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md) | Cấu trúc module, quy tắc thực nghiệm kiểm soát biến và truy xuất số liệu |

## Cài đặt và Chạy lại

```bash
git clone https://github.com/VietDunghere/Image-Processing.git
cd Image-Processing
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dữ liệu HAM10000
Tải từ [Harvard Dataverse (DOI: 10.7910/DVN/DBW86T)](https://doi.org/10.7910/DVN/DBW86T) hoặc [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) và đặt vào `archive/` (`HAM10000_metadata.csv`, `HAM10000_images_part_1/`, `HAM10000_images_part_2/`).

### Chạy toàn bộ thực nghiệm và xuất báo cáo / PDF tự động
```bash
python -X utf8 scripts/run_unified_pipeline.py        # Chạy trọn bộ split, EDA, 17 cấu hình ResNet-18, 3 kiến trúc CNN, 2.8 & 2.5c
python -X utf8 scripts/collect_all_metrics.py         # Tổng hợp bảng so sánh toàn cục & kiểm chứng toàn vẹn
python -X utf8 scripts/finalize_notebooks.py          # Thực thi và đồng bộ nhận xét cả 8 notebook
python -X utf8 scripts/generate_markdown_reports.py   # Xuất hệ thống báo cáo Markdown mục 1 -> 3
python -X utf8 scripts/build_overleaf_and_slides.py   # Biên dịch PDF Overleaf + gói zip + PDF Slide Beamer
```

### Chạy ứng dụng Demo Streamlit
```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```
(Cần tệp checkpoint `outputs/checkpoints/baseline_seed42.pt` sinh ra từ bước huấn luyện baseline; 7 ảnh mẫu mỗi lớp có sẵn tại `samples/`).
