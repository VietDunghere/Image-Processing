# AGENTS.md: bản đồ repo cho agent

Đọc file này trước để **khỏi phải mở lại notebook và báo cáo**. Quy tắc thực nghiệm và cách viết báo cáo ở `CLAUDE.md` (bắt buộc đọc khi chạy thí nghiệm hoặc viết báo cáo). Trả lời bằng tiếng Việt.

**Đề tài:** đánh giá tác động của xử lý ảnh cổ điển (Ch.3) lên phân loại HAM10000 bằng CNN. Mục tiêu là lập luận → thực nghiệm → đánh giá, **không** săn điểm. Kết quả âm vẫn là kết quả. Hạn nộp 11/11/2026.

**Nhánh:** `main` đã gộp toàn bộ công việc. Các nhánh `Dung-Doanh`, `feat/preprocess-artifacts`, `feat/before-after-impact` được giữ lại để tra cứu, không phát triển tiếp trên đó.

**Việc đang dở:** xem `TODO.md` (vấn đề, hướng giải quyết, đã/chưa làm); chi tiết số liệu ở `reports/_handoff_2.5c.md`.

## 1. Quy ước đặt tên

- Tiền tố là **số mục trong kế hoạch nhóm** (`tailieu/ke_hoach_nhom.md`, mục 2.1–2.8), rồi `_mô_tả_không_dấu`.
- **Notebook và báo cáo cùng phần thường trùng tên**, chỉ khác đuôi: `2.2_eda_chat_luong_anh.ipynb` ↔ `reports/2.2_eda_chat_luong_anh.md`. Notebook từng vấn đề của 2.5 đặt là `2.5.<số>_<vấn_đề>.ipynb`, ví dụ `2.5.1_nhieu_gauss.ipynb`.
- Một mục tách nhiều phần: thêm `a`, `b`, `c`. File gộp nhiều mục: `2.6_2.7_…`.
- Tiền tố `_` = ghi chú làm việc, **không** đưa vào báo cáo (`_handoff_2.5c.md`).
- Notebook **phải ở gốc repo**: chúng dùng đường dẫn tương đối (`archive/`, `outputs/`) và `from src...`.
- Slide: `tailieu/ch<N>_….pdf`.

## 2. Bản đồ theo mục kế hoạch

| Mục | Nội dung | Notebook | Báo cáo | Code chính | Đầu ra | Trạng thái |
|---|---|---|---|---|---|---|
| 1 | Giới thiệu, RQ1–RQ4 | – | `1_gioi_thieu.md` | – | – | Có; còn nhãn "[dự kiến]" ở pipeline/demo |
| 2.1 | Mô tả dữ liệu | – | `2.1_du_lieu.md` | `src/report/data_tables.py` | `outputs/report_tables/t21_*` | Xong |
| 2.2 | EDA chất lượng ảnh | `2.2_eda_chat_luong_anh.ipynb` | `2.2_eda_chat_luong_anh.md` | `src/eda/*` | `outputs/eda/`, `outputs/figures/eda/` | Xong, trừ cột "Liên quan lỗi CNN?" còn `TODO` |
| 2.3 | Làm sạch, chia tập, WRS | `2.3_2.4_du_lieu_baseline.ipynb` (mục 3–6) | `2.3_chuan_bi_du_lieu.md` | `scripts/make_splits.py` | `outputs/splits/`, `outputs/data_prep/` | Xong |
| 2.4 | Baseline ResNet-18, 3 seed | `2.3_2.4_du_lieu_baseline.ipynb` (mục 7–11) | `2.4_baseline.md` | `src/training.py` | `outputs/baseline/`, `outputs/figures/baseline/` | Xong |
| 2.5.1, 2.5.2 | Nhiễu Gauss; tương phản thấp (chạy lại) | `2.5.1_nhieu_gauss.ipynb`, `2.5.2_tuong_phan_thap.ipynb` | `2.5a_nhieu_tuong_phan.md` (lượt cũ: `_2.5a_luot_cu.md`); bản giải thích để trình bày: `2.5.1_2.5.2_nhieu_tuong_phan.md`; script nói: `_script_thuyet_trinh_DuyAnh.md` | `src/preprocess/problem_report.py`, `scripts/run_problem_experiments.py`, `scripts/report_noise_contrast.py`, `src/preprocess/{noise,contrast,compare,subgroup}.py` | `outputs/rerun_2.5/`, `outputs/figures/rerun_2.5/`, `reports/figures/2.5.1_2.5.2/` | Xong (3 seed), báo cáo đã cập nhật |
| 2.5b | Lông, viền tối, quá sáng, sáng không đều | – (script) | `2.5b_long_vien_sang.md` | `src/preprocess/artifacts.py`, `scripts/{run_artifact_experiments,analyze_artifacts,artifact_constants}.py` | `outputs/preprocess_artifacts/` (máy khác) | Bản nháp, số từ máy RTX 5060 Ti |
| 2.5.3–2.5.6 | Lông, sáng không đều, quá sáng, viền tối (mỗi vấn đề một notebook) | `2.5.3_long.ipynb`, `2.5.4_sang_khong_deu.ipynb`, `2.5.5_qua_sang.ipynb`, `2.5.6_vien_toi.ipynb` | – | `src/report/notebook_utils.py`, `scripts/make_problem_notebooks.py`, `src/preprocess/artifacts.py` | `outputs/runs_cl/` (máy khác) | Có output 3 seed từ máy RTX 5060 Ti |
| 2.5c | Tác động theo 4 tầng: ảnh, dự đoán, stress test, Grad-CAM | – (script) | – | `src/preprocess/degrade.py`, `scripts/impact_*.py` | `outputs/impact/`, `outputs/figures/impact/` | **Dở**: tầng A xong |
| 2.6, 2.7 | Pipeline đề xuất, huấn luyện cuối | – | `2.6_2.7_pipeline.md` | – | – | Bản nháp; 2.7 chỉ vài dòng |
| 2.8 | Phân tích lỗi, demo | – | `2.8_danh_gia.md` | `scripts/analyze_errors.py`, `app.py` | `outputs/errors/` (chưa có ở máy này) | Code xong, báo cáo bản nháp |
| 3 | Thảo luận, phân công, khai báo AI, Overleaf, slide | – | **chưa có** | – | – | Chưa làm |

**Kết luận chính đến nay:** 0/16 cấu hình 3 seed "tốt hơn rõ" baseline về macro-F1 trên val (`hist_eq_L`, `stretch_L_1_99` tệ hơn rõ trên test), nên pipeline phân loại giữ nguyên baseline, thêm nhánh QC chỉ gắn cờ (2.6).

## 3. Mỗi notebook làm gì

Notebook đã chạy sẵn và có output. **Không mở cả file** vì rất nặng; dùng lệnh ở mục 7 để xem đề mục.

| Notebook | Dung lượng | Các phần | Đọc từ | Ghi ra |
|---|---|---|---|---|
| `2.2_eda_chat_luong_anh.ipynb` | 6,7 MB | 2 cách đo; 3.1–3.7 từng vấn đề (viền, sáng/tối, tương phản, sáng không đều, nhiễu, lông); 4 tổng hợp; 5 liên hệ lỗi CNN (`TODO`); 6–7 bảng kết luận | `outputs/eda/quality_metrics.csv` (sinh bởi `python -X utf8 -m src.eda.quality_metrics --jobs 8`) | `outputs/eda/*` (`image_flags.csv`, `problem_sets/`, `eda_conclusion.csv`, `table_*`), `outputs/figures/eda/` |
| `2.3_2.4_du_lieu_baseline.ipynb` | 340 KB | 3 kiểm tra metadata; 4 mã hóa nhãn; 5 chia train/val/test; 6 WRS; 7 pipeline tốc độ; 8 cấu hình đóng băng; 9 huấn luyện; 10–11 kết quả 3 seed | `archive/` | `outputs/splits/`, `outputs/data_prep/`, `outputs/baseline/`, `outputs/figures/baseline/` |
| `2.5.1_nhieu_gauss.ipynb`, `2.5.2_tuong_phan_thap.ipynb` | – | Cùng khuôn: 1 mô tả; 2 chứng minh; 3 nhóm ảnh; 4 tỉ lệ; 5 phương pháp; 6 thực nghiệm 3 seed; 7 mẫu 100 ảnh trước/sau; 8 nhận xét | `outputs/eda/`, `outputs/preprocess/summary_*.csv`, `outputs/rerun_2.5/` | `outputs/rerun_2.5/<vấn đề>_sample100.csv`, `outputs/figures/rerun_2.5/` |
| `2.5.3_long.ipynb`, `2.5.4_sang_khong_deu.ipynb`, `2.5.5_qua_sang.ipynb`, `2.5.6_vien_toi.ipynb` | – | Cùng khuôn: 1 vấn đề; 2 phương pháp và giả thuyết; 3 ảnh trước/sau; 4 train 3 seed; 5 so với baseline; 6 kết luận | `outputs/eda/image_flags.csv`, `outputs/splits/`, `outputs/runs_cl/` | `outputs/runs_cl/` |

## 4. Script và module

| File | Việc |
|---|---|
| `src/training.py` | **Code train dùng chung.** `run_experiment(name, preprocess_fn, seeds)`, `FROZEN_CONFIG`, cache ảnh. Không sửa khi làm ablation |
| `src/preprocess/__init__.py` | `get_preprocess(tên)`, danh sách cấu hình `CANDIDATES`. Kết hợp bằng `"a+b"` |
| `src/preprocess/noise.py`, `contrast.py`, `artifacts.py` | Các phép xử lý ảnh (lọc; giãn, cân bằng, CLAHE; lông, viền, gamma, flat-field) |
| `src/preprocess/degrade.py` | Suy giảm tổng hợp cho stress test (`LEVELS`, `PAIRS`, chốt trước, không chỉnh theo kết quả) |
| `src/preprocess/image_metrics.py` | Đo ảnh trước/sau (CPU), chỉ trên `train_fit` |
| `src/preprocess/compare.py` | So với baseline: Δ ghép cặp seed, bootstrap cụm lesion, **quy tắc kết luận 2.5.0** |
| `src/preprocess/subgroup.py` | Phân tích trên nhóm ảnh có vấn đề (DiD, chuyển đúng/sai) |
| `src/preprocess/problem_report.py` | Hàm dùng chung cho notebook 2.5.1, 2.5.2; lượt chạy lại ghi vào `outputs/rerun_2.5/` |
| `scripts/run_problem_experiments.py` | Train lại nhiễu/tương phản: `smoke`, `all` (baseline + 6 cấu hình × 3 seed) |
| `scripts/report_noise_contrast.py` | `figures`: tách hình 2.5.1, 2.5.2 từ output notebook và vẽ cột accuracy theo seed; `check`: chạy hàm xử lý ảnh trên `samples/` và ảnh tổng hợp |
| `src/preprocess/figures.py` | Hình giai đoạn 1 của 2.5 |
| `src/eda/quality_metrics.py`, `flags.py`, `stats.py`, `plots.py` | Bộ đo chất lượng, quy tắc gắn cờ, thống kê, style/màu 7 lớp |
| `src/report/notebook_utils.py` | Hàm dùng chung cho notebook 2.5.3–2.5.6; kết quả ghi vào `outputs/runs_cl/` |
| `src/report/data_tables.py` | Bảng số cho 2.1, 2.3, 2.4 |
| `scripts/make_problem_notebooks.py` | Sinh khung notebook 2.5.3–2.5.6. **Ghi đè** cả output và mục 6 đã viết |
| `scripts/make_splits.py` | Tái tạo split cố định |
| `scripts/run_preprocess_experiments.py` | Train nhiễu/tương phản: `smoke`, `screen`, `confirm`, `run` |
| `scripts/run_artifact_experiments.py` | Train lông/viền/gamma/flat: `smoke`, `all`, `run TÊN [seed…]` |
| `scripts/analyze_artifacts.py` | Bảng + hình cho 2.5b (cần `outputs/preprocess_artifacts/` và kết quả train) |
| `scripts/artifact_constants.py` | Tính `L_TARGET` trên `train_fit` |
| `scripts/analyze_errors.py` | 2.8: `cm`, `hair_mel`, `subgroup_norm`, `fig_before_after` |
| `scripts/impact_image_level.py` | 2.5c tầng A: `run`, `hair_lesion`, `figs` |
| `scripts/impact_flips.py` | 2.5c tầng B: đổi nhãn so với mức nền |
| `scripts/impact_stress.py` | 2.5c tầng C: `run`, `analyze`, `grid` |
| `scripts/impact_gradcam.py` | 2.5c tầng D: `run`, `analyze` |
| `scripts/bench_*.py` | Đo tốc độ train, chỉ để tham khảo |
| `scripts/make_samples.py` | Chép 7 ảnh mẫu vào `samples/` |
| `app.py` | Demo Streamlit: QC → các bước xử lý → dự đoán (`streamlit run app.py`) |

## 5. `outputs/` (không commit, có thể khác giữa các máy)

| Thư mục | Nội dung |
|---|---|
| `splits/` | `train_fit`, `val`, `test` (dùng cho thí nghiệm), `train` = `train_fit` + `val`. **Không tạo split mới** |
| `baseline/` | `config.json`, `summary*.csv`, `seed{42,43,44}/{metrics.json,history.csv,val_preds.csv,test_preds.csv}` |
| `experiments/<tên>/` | Cùng cấu trúc baseline. 3 seed: `median3`, `clahe_L_c2_t8`, `median3+clahe_L_c2_t8`, `vig_fill`, `gamma_L_auto`, `flat_L_s60_cond`, `hair_bh17_cond`, `hair_bh17`. 1 seed: `gauss3`, `median5`, `stretch_L_1_99`, `hist_eq_L`. `flat_L_s60`: **chưa có kết quả** |
| `rerun_2.5/` | Lượt chạy lại nhiễu/tương phản, có baseline và `checkpoints/` riêng; không ghép Δ với `baseline/` ở trên |
| `checkpoints/` | `<tên>_seed<s>.pt` |
| `cache/` | Ảnh 224 uint8 đã xử lý, khoảng 1,5 GB/cấu hình |
| `eda/` | Chỉ số từng ảnh, cờ, nhóm vấn đề, bảng kết luận |
| `preprocess/` | Kết quả 2.5a, `comparison_{val,test}.csv`, `subgroup/` |
| `impact/` | Kết quả 2.5c; `_smoke/` chỉ là chạy thử |
| `figures/{eda,baseline,preprocess,impact}/` | Hình cho báo cáo |
| `report_tables/` | Bảng cho 2.1, 2.3 |

## 6. Cấu hình cố định

ResNet-18 `IMAGENET1K_V1`, fine-tune toàn bộ; 224×224; AdamW lr = 1e-4, wd = 1e-4; batch 32; 20 epoch; **WeightedRandomSampler + `CrossEntropyLoss` không class weights**; flip ngang/dọc (chỉ train); chọn epoch theo val macro-F1; seed 42, 43, 44. Khoảng 7 phút/seed trên RTX 3050.

Quy tắc "khác biệt rõ" (2.5.0): |Δ| > std giữa seed của baseline, CI95 bootstrap cụm lesion không chứa 0, cả 3 seed cùng dấu.

## 7. Lệnh hay dùng

```bash
PY=.venv/Scripts/python.exe     # python hệ thống không có pandas; luôn thêm -X utf8

# Xem đề mục notebook mà không mở cả file
$PY -X utf8 -c "import json,sys;[print(l) for c in json.load(open(sys.argv[1],encoding='utf-8'))['cells'] if c['cell_type']=='markdown' for l in ''.join(c['source']).splitlines() if l.startswith('#')]" 2.5.1_nhieu_gauss.ipynb

# Train một cấu hình 3 seed (bỏ qua seed đã có metrics.json)
$PY -X utf8 scripts/run_artifact_experiments.py run <tên> 42 43 44
```

## 8. Cần gì thì đọc gì

| Muốn | Đọc |
|---|---|
| Biết đang làm tới đâu | `reports/_handoff_2.5c.md`, bảng mục 2 ở trên |
| Yêu cầu giảng viên | `tailieu/DoAn_XuLyAnh.pdf` |
| Kế hoạch, phân công nhóm | `tailieu/ke_hoach_nhom.md` |
| Lý thuyết một chương | `tailieu/ch<N>_….pdf` (Ch.3 nâng cao ảnh, Ch.4 hệ màu, Ch.5 biên, Ch.7 hình thái, Ch.8 nhận dạng) |
| Số liệu baseline | `outputs/baseline/summary_mean_std.csv` |
| Thêm thí nghiệm tiền xử lý | Thêm hàm vào `src/preprocess/`, đăng ký trong `__init__.py`, gọi `run_experiment` |
| Quy tắc viết báo cáo | `CLAUDE.md`, mục "Hướng dẫn viết báo cáo" |

## 9. Bẫy cần tránh

- **Đừng sửa mã nguồn hoặc docstring của hàm tiền xử lý đã có cache** (`noise.py`, `contrast.py`, `artifacts.py`). `fn_fingerprint` băm mã nguồn hàm, nên sửa sẽ làm cache sai fingerprint và tính lại. Ngược lại, sửa **hàm phụ** mà hàm chính gọi thì fingerprint không đổi, nên cache cũ vẫn được dùng; khi đó phải xóa cache thủ công. Đó cũng là lý do `tailieu/DoAn_XuLyAnh.pdf` giữ nguyên tên: docstring của `clahe_L` nhắc tới nó.
- **Hai bộ baseline khác máy, không ghép Δ chéo.** Máy này (RTX 3050): test macro-F1 0,6805 ± 0,0192. `2.5b`, `2.6`, `2.7` dùng 0,6785 ± 0,0263 (RTX 5060 Ti). `app.py` dùng checkpoint máy này, nên số ở `2.7` không khớp.
- **Cấu hình kết hợp `"a+b"` luôn bị tính lại cache.** Fingerprint của nó chứa địa chỉ bộ nhớ của hàm, nên khác nhau mỗi lần chạy (ví dụ `median3+clahe_L_c2_t8`). Đây là hành vi sẵn có của `fn_fingerprint`, không phải lỗi dữ liệu.
- **Chạy tác vụ nặng lần lượt.** RAM 23,7 GB; hai tác vụ song song từng bị hệ thống dừng vì thiếu bộ nhớ. Mỗi cache nạp vào RAM khoảng 1,5 GB.
- **Hỏi người dùng trước khi train lâu**; chạy thử tập nhỏ trước. Không commit/push khi chưa được yêu cầu.
- **Test không dùng để chọn phương pháp.** Chia theo `lesion_id`; thống kê lỗi chỉ dùng dự đoán ngoài mẫu.
- Mọi số trong báo cáo phải truy ngược được về file trong `outputs/`.
- Notebook 2.2 mục 5 đang tìm `outputs/baseline/predictions*.csv`, trong khi dự đoán thật nằm ở `outputs/baseline/seed*/test_preds.csv`.
