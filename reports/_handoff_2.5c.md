# Bàn giao: mục 2.5c "Tác động trước/sau của tiền xử lý"

> Cập nhật 9/11/2026. Đây là ghi chú làm việc cho agent/thành viên tiếp theo, **không phải nội dung báo cáo**.
> Hạn nộp đồ án: 11/11/2026. Nhánh: `feat/before-after-impact`, tách từ `feat/preprocess-artifacts` (HEAD `f9e3c1c`).
> **Chưa commit gì.** `CLAUDE.md` cấm commit/push khi chưa được yêu cầu.

## 1. Vì sao có việc này

Ở 2.5–2.6, 12/12 cấu hình 3 seed "không khác biệt rõ" với baseline về macro-F1. Người dùng thấy kết quả đó "hơi vô nghĩa" và muốn làm nổi bật khác biệt trước/sau, nhưng vẫn phải trung thực. Hướng được duyệt là đo tác động ở các tầng khác:

| Tầng | Câu hỏi | Script |
|---|---|---|
| A. Ảnh | Phép xử lý có sửa đúng vấn đề EDA đo không? Tác dụng phụ trên ảnh đối chứng? | `scripts/impact_image_level.py` |
| B. Dự đoán | Số ảnh đổi nhãn có vượt mức nền (baseline khác seed) không? | `scripts/impact_flips.py` |
| C. Stress test | Khi ảnh val/test bị suy giảm tổng hợp, tiền xử lý có giữ được F1 không? | `scripts/impact_stress.py`, `src/preprocess/degrade.py` |
| D. Grad-CAM | Mô hình có bớt nhìn vào lông/viền không? | `scripts/impact_gradcam.py` |

Giả thuyết và thiết kế đã viết **trước khi có kết quả** ở `reports/2.5c_tac_dong_truoc_sau.md` (mục 2.5c.0).
**Không sửa phần giả thuyết, `LEVELS`, `PAIRS` để khớp kết quả.** Nếu buộc phải đổi thiết kế thì ghi rõ lý do và thời điểm, như đã làm với `flat_L_s60`.

## 2. Trạng thái

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Tầng A | **Xong** | `outputs/impact/image_level_*.csv`, `hair_lesion_*.csv`; hình `outputs/figures/impact/fig_a1…a4` |
| Train `vig_fill`, `gamma_L_auto`, `flat_L_s60_cond`, `hair_bh17_cond`, `hair_bh17` (3 seed) | **Xong 15/15** | `outputs/experiments/<tên>/`, `outputs/checkpoints/`, cache `outputs/cache/<tên>_224.npy` |
| Train `flat_L_s60` (3 seed) | **Chưa xong** | Bị hệ thống dừng khi thiếu RAM, ở bước nạp ảnh. Chỉ có `config.json`, `train.log`; chưa có seed, checkpoint, cache |
| Tầng D: tính CAM | **Dở** | Có 12 file `outputs/impact/gradcam/*.npz` của 6/7 cấu hình; thiếu `clahe_L_c2_t8`. `gradcam_per_image.csv` và `gradcam_similarity_per_image.csv` **chưa có** vì chỉ ghi khi chạy hết |
| Tầng B | Chưa chạy | Cần `flat_L_s60` xong |
| Tầng C | Chưa chạy | Cần `flat_L_s60` xong; `outputs/impact/stress/` đang trống |
| Báo cáo `2.5c` | Mới có giả thuyết + thiết kế | Phần kết quả chưa viết |

**Chờ xác nhận:** hai tác vụ bị dừng vì máy thiếu RAM. Người dùng **chưa** đồng ý chạy lại, cần hỏi trước khi chạy.

## 3. Việc tiếp theo, theo thứ tự

Chạy **lần lượt, không song song** (RAM 23,7 GB; lần trước hai tác vụ chạy cùng lúc đã bị dừng, chưa xác định chắc nguyên nhân). Lệnh chạy từ gốc repo:

```bash
PY=.venv/Scripts/python.exe

# 1. Train cấu hình còn thiếu (~20–25 phút; có sẵn cơ chế bỏ qua seed đã có metrics.json)
$PY -X utf8 scripts/run_artifact_experiments.py run flat_L_s60 42 43 44

# 2. Stress test (ước tính 30–40 phút; chạy lại sẽ bỏ qua điều kiện đã có file .npz)
$PY -X utf8 scripts/impact_stress.py run --jobs 12
$PY -X utf8 scripts/impact_stress.py analyze
$PY -X utf8 scripts/impact_stress.py grid

# 3. Grad-CAM (xem lưu ý 4.1 trước khi chạy)
$PY -X utf8 scripts/impact_gradcam.py run
$PY -X utf8 scripts/impact_gradcam.py analyze

# 4. Tầng B (CPU, dùng dự đoán đã lưu)
$PY -X utf8 scripts/impact_flips.py
```

Sau đó:
5. Điền kết quả vào `reports/2.5c_tac_dong_truoc_sau.md`, đối chiếu từng giả thuyết H-A1…H-D2, gồm cả giả thuyết bị bác.
6. Xem trực quan các hình mới (`fig_b*`, `fig_c*`, `fig_d*`) trước khi đưa vào báo cáo.
7. Báo kết quả cho người dùng, hỏi có commit không.

## 4. Lưu ý kỹ thuật

1. **Grad-CAM không tiếp tục được giữa chừng.** `impact_gradcam.py run` chỉ ghi CSV ở cuối và tính lại từ đầu dù đã có `.npz`. Mỗi cấu hình/split còn nạp toàn bộ cache 1,5 GB qua `load_image_cache`. Nên sửa trước khi chạy: ghi CSV theo từng cấu hình và bỏ qua phần đã xong, hoặc chỉ nạp ảnh cần dùng bằng `np.load(..., mmap_mode="r")`.
2. **Code chưa từng chạy thật:** `impact_flips.py` (toàn bộ), `impact_stress.py analyze` và `grid`, `impact_gradcam.py analyze` và phần vẽ hình. Có thể còn lỗi. Phần `impact_stress.py run` và `impact_gradcam.py run` mới chạy thử trên tập nhỏ.
3. **`outputs/impact/_smoke/` là đầu ra chạy thử**, không dùng làm số liệu.
4. **Hai bộ baseline khác máy, không ghép Δ chéo.**
   - `outputs/baseline/` trên máy này (RTX 3050): test macro-F1 0,6805 ± 0,0192. Mọi thí nghiệm mới ở đây so với bộ này.
   - `2.5b`, `2.6`, `2.7` dùng baseline 0,6785 ± 0,0263 từ RTX 5060 Ti; file gốc `outputs/preprocess_artifacts/`, `outputs/errors/` không có trên máy này.
   - Các cấu hình lông/viền/gamma/flat vừa train lại trên RTX 3050 **chưa được đưa vào** `2.5b`, `2.6`. Số trong hai file đó vẫn là của máy kia.
5. **`app.py` dùng `baseline_seed42.pt` của máy này** (epoch 9, test F1 0,6921), trong khi `2.7` ghi epoch 5, F1 0,6744.
6. **Thiết kế stress test:** với viền và lông, phép xử lý gần như là nghịch đảo của suy giảm, nên đây là trường hợp thuận lợi nhất. Phải nêu rõ. Stress test **không** làm đổi kết luận pipeline ở 2.6.
7. **Môi trường:** dùng `.venv/Scripts/python.exe -X utf8`; `python` hệ thống không có pandas.

## 5. Kết quả đã có (nguồn: `outputs/impact/`)

**Tầng A, nhóm cờ `train_fit`** (`image_level_summary.csv`):

| Vấn đề | Phép xử lý | n | Trước → sau (trung vị) | Hết cờ |
|---|---|---|---|---|
| Lông | `hair_bh17` | 644 | 0,098 → 0,000 | 100% |
| Viền tối | `vig_fill` | 223 | 0,056 → 0,000 | 99,6% |
| Viền tối | `vig_crop` | 223 | 0,056 → 0,004 | 98,2% |
| Quá sáng | `gamma_L_auto` | 196 | 0,286 → 0,001 | 91,8% |
| Quá sáng | `gamma_L_1.5` | 196 | 0,286 → 0,033 | 66,8% |
| Sáng không đều | `flat_L_s60_cond` | 150 | 43 → 11 | 98,7% |
| Tương phản thấp | `clahe_L_c2_t8` | 275 | 16,5 → 19,9 | 49,1% |
| Tương phản thấp | `stretch_L_1_99` | 275 | 16,5 → 67,6 | 99,6% |
| Nhiễu (σ Immerkaer) | `median3` | 463 | 1,58 → 0,44 | không áp dụng |

Tác dụng phụ trên ảnh đối chứng:
- Bản có điều kiện (`hair_bh17_cond`, `gamma_L_auto`, `flat_L_s60_cond`) và `vig_fill`, `vig_crop` đổi 0% ảnh.
- `hair_bh17` đổi 93,2% ảnh (ΔE76 trung vị 0,08). `gamma_L_1.5` đổi 100% (ΔE76 12,5). `hist_eq_L` đổi 100% (ΔE76 21,6).

Quan sát định tính cần ghi (`fig_a4`):
- `gamma_L_auto` làm tổn thương tối hẳn ở ảnh rất sáng.
- `hair_bh17` tạo mảng loang ở ảnh lông trắng dày.

**H-A3** (`hair_lesion_*.csv`): trung vị % pixel tổn thương bị inpaint là 8,9% ở `mel` so với 2,5% ở `nv`. Kiểm định một ảnh/lesion: Cliff's δ = 0,436 (n = 394 / 3 458 lesion); Kruskal–Wallis ε² = 0,052 trên 7 lớp. Phù hợp giả thuyết, nhưng chưa loại trừ ảnh hưởng của vùng cơ thể.

**Phát hiện từ kiểm tra 40 ảnh val (trước khi có số CNN, đã ghi trong file giả thuyết):**
- `flat_L_s60_cond` chỉ kích hoạt 25/30/35% ảnh ở 3 mức `uneven`, vì `illum_range` loại pixel tối hơn ngưỡng Otsu.
- `gamma_L_auto` kích hoạt 37,5/75/42,5% ở `overexpose`; ở mức 3 có 57,5% ảnh trung vị L\* = 100 nên hàm trả lại ảnh gốc.

**Tín hiệu từ lần train mới, chưa phân tích** (`outputs/experiments/hair_bh17/`):
- Test macro-F1 của `hair_bh17` là 0,6247 / 0,6290 / 0,5875, so với baseline cùng máy 0,6921 / 0,6583 / 0,6912. Δ trung bình khoảng −0,067, cả 3 seed cùng giảm.
- Báo cáo từ máy kia ghi −0,0237. Chưa tính CI, chưa kết luận theo quy tắc 2.5.0.
- Có thể dùng `src.preprocess.compare.comparison_table` để tính lại cho cả 5 cấu hình mới.

**Kiểm tra toàn vẹn đã đạt:**
- Chỉ số ảnh gốc ở tầng A khớp `quality_metrics.csv` (sai số tối đa 0).
- Suy luận lại baseline trên test khớp dự đoán đã lưu 99,95% (seed 42) và 100% (seed 43, 44).

## 6. File liên quan (chưa commit)

- `src/preprocess/degrade.py`: 6 loại suy giảm × 3 mức, tất định theo `image_id`; `LEVELS`, `PAIRS`.
- `scripts/impact_image_level.py`, `impact_flips.py`, `impact_stress.py`, `impact_gradcam.py`.
- `reports/2.5c_tac_dong_truoc_sau.md`: giả thuyết và thiết kế.
- `reports/_handoff_2.5c.md`: file này.

## 7. Việc còn tồn từ trước (ngoài 2.5c)

- Chưa có `reports/2.8_*.md`; `2.7` chỉ vài dòng; chưa có Thảo luận, phân công, khai báo AI chung, Overleaf, slide, video.
- Cột "Liên quan lỗi CNN?" trong `outputs/eda/eda_conclusion.csv` còn `TODO`; `2.4_baseline.md` mục 4 còn `TODO`.
- `analyze_errors.py` (`hair_mel`, `subgroup_norm`, `fig_before_after`) chưa chạy trên máy này. Nay đã có checkpoint và dự đoán của `hair_bh17` nên có thể chạy được; `fig_before_after` ghi vào `outputs/figures/preprocess_artifacts`.
- Repo chỉ có ResNet-18, trong khi mục đích đề tài có nêu "đánh giá một số mạng CNN" (`2.4_baseline.md` đang `TODO: chờ quyết định của nhóm`).
