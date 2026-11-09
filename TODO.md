# TODO

> Cập nhật 9/11/2026, trên `main` (`e851bd2`). Hạn nộp đồ án: **11/11/2026**.
> Agent mới: đọc file này trước, rồi chạy phần "Kiểm tra nhanh" để biết trạng thái có còn đúng không.
> Bản đồ repo ở `AGENTS.md`. Số liệu và lưu ý chi tiết ở `reports/_handoff_2.5c.md`.

## 1. Vấn đề đang giải quyết

Mục 2.5–2.6 cho thấy **12/12 cấu hình tiền xử lý chạy 3 seed đều "không khác biệt rõ" với baseline** về macro-F1 toàn tập. Người dùng thấy kết quả chỉ dừng ở đó thì "hơi vô nghĩa".

Câu hỏi của mục 2.5c: *nếu macro-F1 toàn tập không đổi, tiền xử lý có tác động ở đâu không, và tác động đó thật hay chỉ là dao động giữa các lần huấn luyện?*

Ràng buộc: phải trung thực. Không chọn số đẹp, không thêm thí nghiệm đến khi "ra khác biệt". Kết quả âm vẫn phải báo cáo.

## 2. Hướng giải quyết

Đo tác động ở 4 tầng. Macro-F1 kém nhạy (CI95 của Δ khoảng ±0,03) nên chỉ là một góc nhìn.

| Tầng | Câu hỏi | So với gì | Script |
|---|---|---|---|
| A. Ảnh | Phép xử lý có sửa đúng vấn đề EDA đo không? Tác dụng phụ? | Trước/sau trên `train_fit`; ảnh đối chứng | `scripts/impact_image_level.py` |
| B. Dự đoán | Số ảnh đổi nhãn có vượt dao động tự nhiên không? | **Mức nền:** baseline khác seed | `scripts/impact_flips.py` |
| C. Stress test | Ảnh val/test bị suy giảm tổng hợp thì tiền xử lý có giữ được F1 không? | Baseline không xử lý, 3 nhánh | `scripts/impact_stress.py`, `src/preprocess/degrade.py` |
| D. Grad-CAM | Mô hình có bớt nhìn vào lông/viền không? | Baseline; mức nền khác seed | `scripts/impact_gradcam.py` |

Giả thuyết H-A1…H-D2 và thiết kế đã **chốt trước khi có kết quả** ở mục 2.5c.0 của báo cáo 2.5c cũ. File đã xóa khỏi thư mục làm việc, xem lại bằng `git show 8d69074^:reports/2.5c_tac_dong_truoc_sau.md`. `LEVELS`, `PAIRS` trong `degrade.py` cũng đã chốt. Không sửa để khớp số.

Kết luận theo quy tắc 2.5.0: "khác biệt rõ" khi |Δ| > std giữa seed của baseline, CI95 bootstrap cụm lesion không chứa 0, và cả 3 seed cùng dấu.

Stress test **không** làm đổi kết luận pipeline ở 2.6, vì đo tình huống khác dữ liệu thật.

## 3. Đã làm

- [x] Viết giả thuyết và thiết kế trước khi có kết quả.
- [x] **Tầng A hoàn tất:** `outputs/impact/image_level_*.csv`, `hair_lesion_*.csv`, hình `outputs/figures/impact/fig_a1…a4`.
  - Hết cờ: lông 100%, viền (`vig_fill`) 99,6%, sáng không đều 98,7%, quá sáng (`gamma_L_auto`) 91,8%, tương phản (CLAHE) 49,1%. Nhiễu: σ trung vị 1,58 → 0,44.
  - Bản có điều kiện không đổi ảnh đối chứng nào; `hair_bh17` đổi 93,2% ảnh đối chứng.
  - `hair_bh17` inpaint trung vị 8,9% pixel tổn thương ở `mel` so với 2,5% ở `nv` (Cliff's δ = 0,436). Chưa loại trừ ảnh hưởng vùng cơ thể.
- [x] Train lại trên RTX 3050, 3 seed mỗi cấu hình: `vig_fill`, `gamma_L_auto`, `flat_L_s60_cond`, `hair_bh17_cond`, `hair_bh17`.
- [x] Viết code tầng B, C, D. Phần `run` của C và D đã chạy thử trên tập nhỏ.
- [x] Kiểm tra toàn vẹn: chỉ số tầng A khớp EDA (sai số 0); suy luận lại baseline khớp dự đoán đã lưu 99,95–100%.
- [x] Chuẩn hóa tên file, thêm `AGENTS.md`, `README.md`; merge mọi nhánh vào `main`, đã push.

## 4. Chưa làm (theo thứ tự)

> **Cần hỏi người dùng trước khi làm bước 1–4.** Lần chạy trước bị hệ thống dừng vì thiếu RAM khi hai tác vụ chạy song song; người dùng chưa đồng ý chạy lại. Chạy **lần lượt**, không song song.

```bash
PY=.venv/Scripts/python.exe
```

- [ ] **1. Train `flat_L_s60` (3 seed, khoảng 20–25 phút).** Bị dừng ở bước nạp ảnh; tầng B và C cần cấu hình này.
  `$PY -X utf8 scripts/run_artifact_experiments.py run flat_L_s60 42 43 44`
- [ ] **2. Stress test (ước tính 30–40 phút).** Chạy lại sẽ bỏ qua điều kiện đã có `.npz`.
  `$PY -X utf8 scripts/impact_stress.py run --jobs 12`, rồi `analyze`, rồi `grid`
- [ ] **3. Grad-CAM.** Hiện có CAM của 6/7 cấu hình (thiếu `clahe_L_c2_t8`) nhưng **chưa có CSV**.
  - [ ] Sửa `impact_gradcam.py` trước khi chạy: chỉ ghi CSV ở cuối, không tiếp tục được giữa chừng, và nạp cả cache 1,5 GB cho mỗi cấu hình/split.
  - [ ] `$PY -X utf8 scripts/impact_gradcam.py run`, rồi `analyze`
- [ ] **4. Tầng B.** `$PY -X utf8 scripts/impact_flips.py` (CPU, dùng dự đoán đã lưu).
- [ ] **5. Xem trực quan** các hình `fig_b*`, `fig_c*`, `fig_d*`. Code `impact_flips.py`, `impact_stress.py analyze`/`grid`, `impact_gradcam.py analyze` **chưa chạy lần nào**, có thể còn lỗi.
- [ ] **6. Tính CI cho 5 cấu hình vừa train** bằng `src.preprocess.compare.comparison_table`. Đặc biệt `hair_bh17`: test macro-F1 0,6247 / 0,6290 / 0,5875, so với baseline 0,6921 / 0,6583 / 0,6912; Δ khoảng −0,067, cả 3 seed cùng giảm. Chưa có CI nên chưa kết luận.
- [ ] **7. Viết kết quả vào báo cáo 2.5c (file cũ đã xóa, cần tạo file mới nếu còn làm mục này):** đối chiếu từng giả thuyết, kể cả giả thuyết bị bác. Nêu hạn chế: ảnh giả lập; với viền và lông, phép xử lý gần như nghịch đảo của suy giảm.

## 5. Việc tồn đọng ngoài 2.5c

**Cần cho bài nộp:**
- [ ] Báo cáo Overleaf (4–10 trang); chưa có file `.tex` trong repo.
- [ ] Slide (10–15), video demo; chưa có trong repo.
- [ ] Thảo luận, bảng phân công, khai báo AI chung; hiện chỉ có đoạn khai báo riêng ở cuối `2.5b`.
- [ ] Báo cáo 2.8: chưa có file. Code ở `scripts/analyze_errors.py` chưa chạy trên máy này.
- [ ] Viết đầy đủ 2.7 (hiện vài dòng trong `2.6_2.7_pipeline.md`).

**Sửa số liệu và TODO trong báo cáo:**
- [ ] Thống nhất baseline: máy này 0,6805 ± 0,0192; `2.5b`, `2.6`, `2.7` dùng 0,6785 ± 0,0263 (RTX 5060 Ti). **Không ghép Δ chéo.** `app.py` dùng checkpoint máy này (epoch 9, F1 0,6921) còn `2.7` ghi epoch 5, F1 0,6744.
- [ ] Điền cột "Liên quan lỗi CNN?" trong `outputs/eda/eda_conclusion.csv` và mục 4 của báo cáo 2.2.
- [ ] `2.4_baseline.md` mục 4: nối dự đoán baseline với `quality_metrics.csv`. Notebook 2.2 mục 5 đang tìm sai đường dẫn (`predictions*.csv`).
- [ ] `2.5b`: confusion `mel`→`nv` của `hair_bh17` (`analyze_errors.py hair_mel`); chuẩn hóa lỗi nhóm cờ theo thành phần lớp (`subgroup_norm`); xuất lại Hình 2.5-6 (`fig_before_after`).
- [ ] `2.5a`: xem ảnh `df` bị sai sau CLAHE.
- [ ] `1_gioi_thieu.md`: bỏ nhãn "[dự kiến]" khi pipeline và demo đã xong.

**Cần nhóm quyết định:**
- [ ] So sánh thêm kiến trúc CNN? Mục đích đề tài có nêu, nhưng repo chỉ có ResNet-18 (`2.4_baseline.md` đang `TODO`).

## 6. Kiểm tra nhanh trạng thái

```bash
git status -sb
ls outputs/experiments/flat_L_s60       # có seed42–44 nghĩa là bước 1 xong
ls outputs/impact/stress | wc -l        # 0 nghĩa là bước 2 chưa chạy
ls outputs/impact/gradcam_*.csv         # không có nghĩa là bước 3 chưa xong
ls outputs/impact/flips_*.csv           # không có nghĩa là bước 4 chưa chạy
```

`outputs/` không nằm trong git, nên trạng thái trên chỉ đúng với máy đã chạy.

## 7. Không được làm

- Không sửa mã nguồn hoặc docstring của hàm trong `noise.py`, `contrast.py`, `artifacts.py` (làm hỏng cache; xem `AGENTS.md` mục 9).
- Không sửa giả thuyết, `LEVELS`, `PAIRS` sau khi đã thấy kết quả.
- Không dùng test để chọn phương pháp. Không đổi `FROZEN_CONFIG`.
- Không commit/push khi chưa được yêu cầu.
- Mọi số trong báo cáo phải truy ngược được về file trong `outputs/`; chưa có số thì ghi `TODO`.

Khi xong một mục: đánh dấu `[x]` và cập nhật dòng ngày ở đầu file.
