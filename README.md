# Đánh giá tác động của xử lý ảnh lên phân loại bệnh da bằng CNN

Đồ án học phần Xử lý ảnh – Khoa Trí tuệ nhân tạo, PTIT (Lựa chọn 2: dự án thực tế). Đề tài đánh giá các phương pháp nâng cao chất lượng ảnh (Chương 3; thêm Chương 4 – hệ màu, Chương 7 – hình thái học) trên bộ dữ liệu HAM10000 bằng ResNet-18, theo hướng **lập luận → thực nghiệm → đánh giá**. Mục tiêu không phải đạt điểm cao nhất, nên kết quả "không cải thiện" cũng được báo cáo.

**Câu hỏi:** các phương pháp này có cải thiện mô hình phân loại 7 lớp tổn thương da hay không, khi được đánh giá có kiểm soát: mỗi thí nghiệm chỉ đổi một bước xử lý, chạy 3 seed, so ghép cặp với baseline.

> Chỉ phục vụ học tập. **Không dùng để chẩn đoán.**

## Kết quả chính

- Baseline ResNet-18 (chỉ resize và normalize): test macro-F1 **0,6805 ± 0,0192** trên 3 seed (`outputs/baseline/summary_mean_std.csv`). Lượt chạy lại phần nhiễu và tương phản có baseline riêng **0,6837 ± 0,0141** (`outputs/rerun_2.5/`). Mỗi phần so với baseline của chính nó.
- Không cấu hình tiền xử lý nào **tốt hơn rõ** baseline về macro-F1 trên val. `hist_eq_L` tệ hơn rõ trên val và test, `stretch_L_1_99` tệ hơn rõ trên test, các cấu hình còn lại không khác biệt rõ (`reports/2.5a_nhieu_tuong_phan.md`, `reports/2.6_2.7_pipeline.md`). Pipeline đề xuất vì vậy giữ nguyên baseline, thêm một nhánh kiểm tra chất lượng ảnh chỉ để gắn cờ.
- Phân tích tác động theo các tầng khác đang thực hiện (`reports/2.5c_tac_dong_truoc_sau.md`).

## Bắt đầu đọc từ đâu

| Bạn muốn | Xem |
|---|---|
| Bản đồ repo: notebook, script, kết quả | [`AGENTS.md`](AGENTS.md) |
| Quy tắc thực nghiệm và viết báo cáo | [`CLAUDE.md`](CLAUDE.md) |
| Nội dung báo cáo từng mục | [`reports/`](reports/) |
| Việc đang dở | [`TODO.md`](TODO.md) |

Notebook và báo cáo cùng phần có tên trùng nhau, bắt đầu bằng số mục kế hoạch. Ví dụ `2.2_eda_chat_luong_anh.ipynb` đi với `reports/2.2_eda_chat_luong_anh.md`.

## Cấu trúc

```
2.x_*.ipynb       Notebook phân tích theo từng mục (chạy từ thư mục gốc)
app.py            Demo Streamlit (2.8)
requirements.txt
src/training.py   Code huấn luyện dùng chung cho mọi thí nghiệm (cấu hình đóng băng, chỉ đổi preprocess_fn)
src/eda/          Đo chất lượng ảnh, gắn cờ, thống kê, hình
src/preprocess/   Các phép xử lý ảnh (Ch.3/4/7), so sánh với baseline, phân tích nhóm
src/report/       Hàm dùng chung cho bảng số và notebook 2.5.3–2.5.6
scripts/          Chạy thí nghiệm và phân tích: split, train các cấu hình, ảnh mẫu
reports/          Bản nháp báo cáo theo mục (Markdown)
samples/          7 ảnh mẫu cho demo (HAM10000, CC BY-NC 4.0)
archive/          Dữ liệu gốc (không commit)
outputs/          Split, kết quả, checkpoint, hình (không commit)
tailieu/          Đề bài, slide, kế hoạch nhóm (không commit)
```

## Môi trường

| | Đã kiểm thử |
|---|---|
| Máy 1 | Windows 11, RTX 3050, Python + `.venv`, torch 2.8.0 (CUDA 12.6) |
| Máy 2 | Ubuntu, RTX 5060 Ti, conda env, torch 2.14.1 (CUDA 13.0) |

GPU dòng RTX 50xx cần bản torch hỗ trợ `sm_120` (CUDA ≥ 12.8). Nếu dùng GPU này, cài torch theo hướng dẫn ở pytorch.org thay cho dòng `torch==2.8.0` trong `requirements.txt`.

## Cài đặt

Máy không có GPU thì bỏ dòng `--extra-index-url` trong `requirements.txt` trước khi cài.

```bash
git clone https://github.com/VietDunghere/Image-Processing.git
cd Image-Processing
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Dữ liệu

**HAM10000:** 10 015 ảnh dermoscopy 600×450, 7 lớp `akiec, bcc, bkl, df, mel, nv, vasc`. Có thể tải từ một trong hai nguồn:
- Kaggle: <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>
- Harvard Dataverse (bản gốc): <https://doi.org/10.7910/DVN/DBW86T>

Giải nén vào thư mục `archive/` ở gốc repo:

```
archive/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/   # .jpg
└── HAM10000_images_part_2/   # .jpg
```

Nếu dữ liệu nằm ở thư mục khác, tạo symlink: `ln -s /đường/dẫn/archive archive`.

Thư mục `archive/` và `outputs/` không được commit (`.gitignore`). Mọi kết quả đều sinh lại được bằng các lệnh ở mục tiếp theo.

**Nguồn và giấy phép:** Tschandl P., Rosendahl C., Kittler H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161. Giấy phép CC BY-NC 4.0. Bảy ảnh trong `samples/` lấy từ bộ dữ liệu này.

## Lệnh chạy (theo thứ tự)

Chạy mọi lệnh **từ thư mục gốc repo**. Luôn dùng `python -X utf8` để tránh lỗi tiếng Việt trên console Windows. Các script train đều **chạy tiếp được**: bấm Ctrl+C rồi chạy lại thì các seed đã có `metrics.json` được bỏ qua.

Thời gian ghi trong ngoặc đo trên RTX 5060 Ti, trừ khi ghi khác. Danh sách đầy đủ các lệnh và cấu hình thí nghiệm ở `AGENTS.md`.

### Chuẩn bị dữ liệu và EDA (mục 2.1–2.3)

```bash
python -X utf8 scripts/make_splits.py                       # split cố định theo lesion_id → outputs/splits/
python -X utf8 -m src.eda.quality_metrics --jobs 8          # chỉ số chất lượng cho 10 015 ảnh (~1 phút)
```

Sau đó chạy notebook `2.2_eda_chat_luong_anh.ipynb`. Notebook này sinh `outputs/eda/image_flags.csv` (cờ "ảnh có vấn đề"), file mà mọi bước sau đều cần.

### Baseline (mục 2.4)

```bash
python -X utf8 scripts/run_artifact_experiments.py run baseline      # ResNet-18, 3 seed (~2 phút/seed)
python -X utf8 scripts/run_artifact_experiments.py run baseline 42   # chỉ một seed (khoảng 7 phút trên RTX 3050)
```

Notebook `2.3_2.4_du_lieu_baseline.ipynb` gồm các phần: kiểm tra dữ liệu, split, cấu hình đóng băng và hình kết quả.

### Tiền xử lý: nhiễu Gauss và tương phản (mục 2.5.1, 2.5.2)

```bash
python -X utf8 scripts/run_problem_experiments.py smoke     # 64 ảnh, 1 epoch, kiểm tra pipeline
python -X utf8 scripts/run_problem_experiments.py all       # baseline + 6 cấu hình × 3 seed → outputs/rerun_2.5/
```

Sau đó chạy `2.5.1_nhieu_gauss.ipynb` và `2.5.2_tuong_phan_thap.ipynb`.

Lượt chạy cũ (sàng lọc và cấu hình kết hợp, kết quả lưu ở `reports/_2.5a_luot_cu.md`) dùng các lệnh sau. Không ghép Δ giữa hai lượt.

```bash
python -X utf8 scripts/run_preprocess_experiments.py screen                         # sàng lọc 4 cấu hình, seed 42
python -X utf8 scripts/run_preprocess_experiments.py confirm median3 clahe_L_c2_t8  # cấu hình thắng: thêm seed 43, 44
python -X utf8 scripts/run_preprocess_experiments.py run "median3+clahe_L_c2_t8"    # cấu hình kết hợp, 3 seed
```

### Tiền xử lý: lông, viền tối, quá sáng, sáng không đều (báo cáo `2.5b_long_vien_sang.md`, mục 2.6)

```bash
python -X utf8 scripts/run_artifact_experiments.py all       # baseline + 8 cấu hình × 3 seed (~50 phút)
python -X utf8 scripts/run_artifact_experiments.py run "vig_fill+flat_L_s60_cond"   # cấu hình kết hợp cho 2.6
python -X utf8 scripts/analyze_artifacts.py                  # đo trên ảnh + bảng so sánh + phân tích nhóm
```

`scripts/artifact_constants.py` tính lại hằng số `L_TARGET` (chỉ dùng `train_fit`). Giá trị đã được ghi sẵn trong `src/preprocess/artifacts.py`.

### Notebook từng vấn đề 2.5.3–2.5.6

`2.5.3_long.ipynb`, `2.5.4_sang_khong_deu.ipynb`, `2.5.5_qua_sang.ipynb`, `2.5.6_vien_toi.ipynb`. Mỗi notebook gồm các phần: vấn đề → phương pháp áp dụng → ảnh trước/sau → train 3 seed → so sánh với baseline (bảng, biểu đồ cột theo từng seed, theo lớp, trên nhóm ảnh có vấn đề) → kết luận và giải thích.

- Kết quả của lần chạy này nằm trong `outputs/runs_cl/`, tách khỏi kết quả cũ. Cấu hình giữ nguyên siêu tham số đóng băng; chỉ bật `channels_last`, batch đánh giá 512 và 32 luồng tạo cache để chạy nhanh hơn trên RTX 5060 Ti.
- Có thể train trước bằng terminal, rồi mở notebook để xem kết quả (ô train chỉ đọc lại kết quả đã có):
  ```bash
  python -X utf8 -c "import sys; sys.path.insert(0,'.'); from src.report import notebook_utils as U; [U.train(n) for n in ['baseline'] + [c for p in U.PROBLEMS.values() for c in p['configs']]]"
  ```
- Chọn kernel trỏ tới đúng môi trường của dự án. Chạy lần lượt từng notebook, không chạy song song, vì các notebook dùng chung baseline.
- `scripts/make_problem_notebooks.py` sinh lại khung 4 notebook. **Lệnh này ghi đè** cả output và mục 6 đã viết.

### Đánh giá và phân tích lỗi (mục 2.8)

```bash
python -X utf8 scripts/analyze_errors.py cm                 # confusion matrix, cặp nhầm, ảnh sai ở cả 3 seed
python -X utf8 scripts/analyze_errors.py hair_mel           # hair_bh17 và cặp mel→nv
python -X utf8 scripts/analyze_errors.py subgroup_norm      # lỗi nhóm cờ EDA, chuẩn hóa theo thành phần lớp
python -X utf8 scripts/analyze_errors.py fig_before_after   # Hình 2.5-6 (trước/sau xử lý)
```

### Demo

```bash
python -X utf8 scripts/make_samples.py     # (đã có sẵn trong repo) 7 ảnh mẫu, 1 ảnh/lớp, từ tập test
streamlit run app.py                       # mở http://localhost:8501
```

Demo cần checkpoint `outputs/checkpoints/baseline_seed42.pt` (sinh ở bước baseline). Giao diện gồm:
- **(A)** bảng kiểm tra chất lượng ảnh (6 vấn đề của EDA);
- **các giai đoạn xử lý** do người dùng chọn;
- **(B)** dự đoán của baseline trên ảnh gốc, kèm dự đoán trên ảnh đã xử lý để so sánh.

### Chạy notebook không cần mở Jupyter

```bash
python -m nbconvert --to notebook --execute <tên>.ipynb --output-dir outputs/notebooks_run \
       --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
```

- Cần `pip install nbconvert ipykernel`.
- Cờ `kernel_name=python3` là bắt buộc nếu máy không có kernel tên `image-processing` (tên kernel ghi trong metadata của `2.2_eda_chat_luong_anh.ipynb` và `2.3_2.4_du_lieu_baseline.ipynb`).
- Thêm `--allow-errors` nếu thiếu file `outputs/preprocess/lowcontrast_mask_audit.csv`. File này chứa nhãn gán bằng mắt cho 24 ảnh, không do code sinh ra.

## Đầu ra mong đợi

| Thư mục | Nội dung |
|---|---|
| `outputs/splits/` | `train_fit.csv` (6 408 ảnh), `val.csv` (1 603), `test.csv` (2 004); chia theo `lesion_id`, không rò rỉ |
| `outputs/eda/` | `quality_metrics.csv` (1 dòng/ảnh), `image_flags.csv`, `eda_conclusion.csv` |
| `outputs/baseline/`, `outputs/experiments/<cấu hình>/` | Mỗi seed: `history.csv`, `metrics.json`, `val_preds.csv`, `test_preds.csv`; `summary.csv` (3 seed) |
| `outputs/rerun_2.5/` | Lượt chạy lại nhiễu và tương phản (notebook 2.5.1, 2.5.2) |
| `outputs/runs_cl/` | Lượt chạy của notebook 2.5.3–2.5.6 |
| `outputs/preprocess/`, `outputs/preprocess_artifacts/` | `comparison_{val,test}.csv` (Δ so với baseline, CI95, dấu theo seed, kết luận), phân tích nhóm, số đo trên ảnh |
| `outputs/errors/` | Confusion matrix, cặp nhầm, ảnh sai nhất quán, lỗi nhóm chuẩn hóa theo lớp |
| `outputs/figures/` | Hình cho báo cáo (PDF + PNG) |
| `outputs/checkpoints/` | Checkpoint tốt nhất theo val macro-F1 của từng cấu hình × seed |

**Số tham chiếu** để kiểm tra đã chạy đúng (baseline, 3 seed, mean ± std):
- val macro-F1: 0,7252 ± 0,0112 trên máy 2; 0,7215 ± 0,0032 trên máy 1;
- test macro-F1: 0,6785 ± 0,0263 trên máy 2; 0,6805 ± 0,0192 trên máy 1.

Kết quả không trùng từng bit giữa các lần chạy, do AMP fp16 và `cudnn.benchmark`. So sánh luôn dựa trên mean ± std của 3 seed.

## Ghi chú kỹ thuật

- **Nguyên tắc thí nghiệm:** mọi cấu hình gọi cùng `src.training.run_experiment`, chỉ khác hàm tiền xử lý. Phép xử lý được áp ở 600×450, **trước** resize 224 và Normalize, giống nhau cho train, val và test. Test chỉ dùng để báo cáo, việc chọn cấu hình dựa trên val.
- **Linux:** `src/preprocess/image_metrics.py` dùng `multiprocessing` kiểu `spawn`. Kiểu `fork` mặc định bị treo khi tiến trình cha đã mở luồng OpenCV.
- **Khai báo sử dụng AI:** một phần mã (`src/preprocess/artifacts.py`, các script phân tích, `app.py`) và bản nháp báo cáo được viết với hỗ trợ của AI (Claude). Nhóm chạy thí nghiệm và kiểm tra số liệu. Chi tiết ở mục khai báo AI của báo cáo.
