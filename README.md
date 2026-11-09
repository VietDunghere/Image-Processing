# Đánh giá tác động của tiền xử lý ảnh lên phân loại tổn thương da (HAM10000) bằng CNN

Đồ án Xử lý ảnh – Khoa Trí tuệ nhân tạo, PTIT (Lựa chọn 2: dự án thực tế).

**Câu hỏi:** các phương pháp nâng cao chất lượng ảnh (Chương 3; thêm Chương 4 – hệ màu, Chương 7 – hình thái học) có cải thiện mô hình phân loại 7 lớp tổn thương da (ResNet-18) hay không, khi được đánh giá có kiểm soát: mỗi thí nghiệm chỉ đổi một bước xử lý, chạy 3 seed, so ghép cặp với baseline.

**Kết quả tóm tắt** (chi tiết trong `reports/`):
- Không phép xử lý nào trong 12 cấu hình (nhiễu, tương phản, lông, viền tối, quá sáng, sáng không đều, và các cấu hình kết hợp) khác biệt rõ so với baseline về macro-F1.
- Pipeline đề xuất gồm hai nhánh: (A) kiểm tra chất lượng ảnh, chỉ gắn cờ; (B) phân loại bằng baseline (mục 2.6).

> Chỉ phục vụ học tập. **Không dùng để chẩn đoán.**

---

## 1. Môi trường

| | Đã kiểm thử |
|---|---|
| Máy 1 | Windows 11, RTX 3050, Python + `.venv`, torch 2.8.0 (CUDA 12.6) |
| Máy 2 | Ubuntu, RTX 5060 Ti, conda env, torch 2.14.1 (CUDA 13.0) |

GPU dòng RTX 50xx cần bản torch hỗ trợ `sm_120` (CUDA ≥ 12.8). Nếu dùng GPU này, cài torch theo hướng dẫn ở pytorch.org thay cho dòng `torch==2.8.0` trong `requirements.txt`.

## 2. Cài đặt

```bash
git clone https://github.com/VietDunghere/Image-Processing.git
cd Image-Processing
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Máy không có GPU: bỏ dòng `--extra-index-url` trong `requirements.txt` trước khi cài.

## 3. Dữ liệu

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

Thư mục `archive/` và `outputs/` không được commit (`.gitignore`). Mọi kết quả đều sinh lại được bằng các lệnh ở mục 4.

**Nguồn và giấy phép:** Tschandl P., Rosendahl C., Kittler H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161. Giấy phép CC BY-NC 4.0. Bảy ảnh trong `samples/` lấy từ bộ dữ liệu này.

## 4. Lệnh chạy (theo thứ tự)

Chạy mọi lệnh **từ thư mục gốc repo**. Windows nên thêm `-X utf8` (console mặc định cp1252 lỗi chữ tiếng Việt). Các script train đều **chạy tiếp được**: bấm Ctrl+C rồi chạy lại thì các seed đã có `metrics.json` được bỏ qua.

Thời gian ghi trong ngoặc đo trên RTX 5060 Ti.

### 4.1 Chuẩn bị dữ liệu và EDA (mục 2.1–2.3)

```bash
python -X utf8 scripts/make_splits.py                       # split cố định theo lesion_id → outputs/splits/
python -X utf8 -m src.eda.quality_metrics --jobs 8          # chỉ số chất lượng cho 10 015 ảnh (~1 phút)
```

Sau đó chạy notebook `eda_image_quality.ipynb`. Notebook này sinh `outputs/eda/image_flags.csv` (cờ "ảnh có vấn đề"), file mà mọi bước sau đều cần.

### 4.2 Baseline (mục 2.4)

```bash
python -X utf8 scripts/run_artifact_experiments.py run baseline     # ResNet-18, 3 seed (~2 phút/seed)
```

Notebook `baseline.ipynb` gồm các phần: kiểm tra dữ liệu, split, cấu hình đóng băng và hình kết quả. Notebook đọc lại kết quả đã train, không train lại.

### 4.3 Tiền xử lý: nhiễu Gauss và tương phản (mục 2.5.1–2.5.7)

```bash
python -X utf8 scripts/run_preprocess_experiments.py screen                        # sàng lọc 4 cấu hình, seed 42
python -X utf8 scripts/run_preprocess_experiments.py confirm median3 clahe_L_c2_t8  # cấu hình thắng: thêm seed 43, 44
python -X utf8 scripts/run_preprocess_experiments.py run "median3+clahe_L_c2_t8"    # cấu hình kết hợp, 3 seed
```

Sau đó chạy `preprocess_noise_contrast.ipynb` (đo trên ảnh và so sánh CNN), rồi `subgroup_before_after.ipynb` (phân tích trên chính nhóm ảnh có vấn đề).

### 4.4 Tiền xử lý: lông, viền tối, quá sáng, sáng không đều (mục 2.5.8–2.5.13, 2.6)

```bash
python -X utf8 scripts/run_artifact_experiments.py all       # baseline + 8 cấu hình × 3 seed (~50 phút)
python -X utf8 scripts/run_artifact_experiments.py run "vig_fill+flat_L_s60_cond"   # cấu hình kết hợp cho 2.6
python -X utf8 scripts/analyze_artifacts.py                  # đo trên ảnh + bảng so sánh + phân tích nhóm
```

`scripts/artifact_constants.py` tính lại hằng số `L_TARGET` (chỉ dùng `train_fit`). Giá trị đã được ghi sẵn trong `src/preprocess/artifacts.py`.

### 4.5 Đánh giá và phân tích lỗi (mục 2.8)

```bash
python -X utf8 scripts/analyze_errors.py cm                 # confusion matrix, cặp nhầm, ảnh sai ở cả 3 seed
python -X utf8 scripts/analyze_errors.py hair_mel           # hair_bh17 và cặp mel→nv
python -X utf8 scripts/analyze_errors.py subgroup_norm      # lỗi nhóm cờ EDA, chuẩn hóa theo thành phần lớp
python -X utf8 scripts/analyze_errors.py fig_before_after   # Hình 2.5-6 (trước/sau xử lý)
```

### 4.6 Demo

```bash
python -X utf8 scripts/make_samples.py     # (đã có sẵn trong repo) 7 ảnh mẫu, 1 ảnh/lớp, từ tập test
streamlit run app.py                       # mở http://localhost:8501
```

Demo cần checkpoint `outputs/checkpoints/baseline_seed42.pt` (sinh ở bước 4.2). Giao diện gồm:
- **(A)** bảng kiểm tra chất lượng ảnh (6 vấn đề của EDA);
- **các giai đoạn xử lý** do người dùng chọn;
- **(B)** dự đoán của baseline trên ảnh gốc, kèm dự đoán trên ảnh đã xử lý để so sánh.

### 4.7 Bốn notebook "một vấn đề – một notebook" (trình bày)

`01_long.ipynb`, `02_sang_khong_deu.ipynb`, `03_qua_sang.ipynb`, `04_vien_toi.ipynb`. Mỗi notebook gồm các phần: vấn đề → phương pháp áp dụng → ảnh trước/sau → train 3 seed → so sánh với baseline (bảng, biểu đồ cột theo từng seed, theo lớp, trên nhóm ảnh có vấn đề) → kết luận và giải thích.

- Kết quả của lần chạy này nằm trong `outputs/runs_cl/`, tách khỏi kết quả cũ. Cấu hình giữ nguyên siêu tham số đóng băng; chỉ bật `channels_last`, batch đánh giá 512 và 32 luồng tạo cache để chạy nhanh hơn trên RTX 5060 Ti.
- Có thể train trước bằng terminal, rồi mở notebook để xem kết quả (ô train chỉ đọc lại kết quả đã có):
  ```bash
  python -X utf8 -c "import sys; sys.path.insert(0,'.'); from src.report import notebook_utils as U; [U.train(n) for n in ['baseline'] + [c for p in U.PROBLEMS.values() for c in p['configs']]]"
  ```
- Chọn kernel trỏ tới đúng môi trường của dự án. Chạy lần lượt từng notebook, không chạy song song, vì các notebook dùng chung baseline.
- `scripts/make_problem_notebooks.py` sinh lại khung 4 notebook. **Lệnh này ghi đè** cả output và mục 6 đã viết.

### Chạy notebook không cần mở Jupyter

```bash
python -m nbconvert --to notebook --execute <tên>.ipynb --output-dir outputs/notebooks_run \
       --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
```

- Cần `pip install nbconvert ipykernel`.
- Cờ `kernel_name=python3` là bắt buộc nếu máy không có kernel tên `image-processing` (tên kernel ghi trong metadata notebook).
- Thêm `--allow-errors` nếu thiếu file `outputs/preprocess/lowcontrast_mask_audit.csv`. File này chứa nhãn gán bằng mắt cho 24 ảnh, không do code sinh ra.

## 5. Đầu ra mong đợi

| Thư mục | Nội dung |
|---|---|
| `outputs/splits/` | `train_fit.csv` (6 408 ảnh), `val.csv` (1 603), `test.csv` (2 004); chia theo `lesion_id`, không rò rỉ |
| `outputs/eda/` | `quality_metrics.csv` (1 dòng/ảnh), `image_flags.csv`, `eda_conclusion.csv` |
| `outputs/baseline/`, `outputs/experiments/<cấu hình>/` | Mỗi seed: `history.csv`, `metrics.json`, `val_preds.csv`, `test_preds.csv`; `summary.csv` (3 seed) |
| `outputs/preprocess/`, `outputs/preprocess_artifacts/` | `comparison_{val,test}.csv` (Δ so với baseline, CI95, dấu theo seed, kết luận), phân tích nhóm, số đo trên ảnh |
| `outputs/errors/` | Confusion matrix, cặp nhầm, ảnh sai nhất quán, lỗi nhóm chuẩn hóa theo lớp |
| `outputs/figures/` | Hình cho báo cáo (PDF + PNG) |
| `outputs/checkpoints/` | Checkpoint tốt nhất theo val macro-F1 của từng cấu hình × seed |

**Số tham chiếu** để kiểm tra đã chạy đúng (baseline, 3 seed, mean ± std):
- val macro-F1: 0,7252 ± 0,0112 trên máy 2; 0,7215 ± 0,0032 trên máy 1;
- test macro-F1: 0,6785 ± 0,0263 trên máy 2; 0,6805 ± 0,0192 trên máy 1.

Kết quả không trùng từng bit giữa các lần chạy, do AMP fp16 và `cudnn.benchmark`. So sánh luôn dựa trên mean ± std của 3 seed.

## 6. Cấu trúc repo

```
├── app.py                         # demo Streamlit (2.8)
├── requirements.txt
├── *.ipynb                        # EDA, baseline, nhiễu/tương phản, phân tích nhóm; 01–04: mỗi vấn đề một notebook
├── src/
│   ├── training.py                # train/đánh giá dùng chung (cấu hình đóng băng, chỉ đổi preprocess_fn)
│   ├── eda/                       # chỉ số chất lượng ảnh, quy tắc cờ, thống kê, hình
│   └── preprocess/                # noise.py, contrast.py, artifacts.py (Ch.3/4/7), so sánh, phân tích nhóm
├── scripts/                       # lệnh chạy: split, train các cấu hình, phân tích, ảnh mẫu
├── reports/                       # bản nháp các mục báo cáo (Markdown)
└── samples/                       # 7 ảnh mẫu (HAM10000, CC BY-NC 4.0)
```

## 7. Ghi chú kỹ thuật

- **Nguyên tắc thí nghiệm:** mọi cấu hình gọi cùng `src.training.run_experiment`, chỉ khác hàm tiền xử lý. Phép xử lý được áp ở 600×450, **trước** resize 224 và Normalize, giống nhau cho train, val và test. Test chỉ dùng để báo cáo, việc chọn cấu hình dựa trên val.
- **Linux:** `src/preprocess/image_metrics.py` dùng `multiprocessing` kiểu `spawn`. Kiểu `fork` mặc định bị treo khi tiến trình cha đã mở luồng OpenCV.
- **Khai báo sử dụng AI:** một phần mã (`src/preprocess/artifacts.py`, các script phân tích, `app.py`) và bản nháp báo cáo được viết với hỗ trợ của AI (Claude). Nhóm chạy thí nghiệm và kiểm tra số liệu. Chi tiết ở mục khai báo AI của báo cáo.
