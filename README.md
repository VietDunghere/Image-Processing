# Đánh giá tác động của xử lý ảnh lên phân loại bệnh da bằng CNN

Đồ án học phần Xử lý ảnh. Đề tài đánh giá các phương pháp nâng cao chất lượng ảnh (Chương 3) trên bộ dữ liệu HAM10000 bằng ResNet-18, theo hướng **lập luận → thực nghiệm → đánh giá**. Mục tiêu không phải đạt điểm cao nhất, nên kết quả "không cải thiện" cũng được báo cáo.

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
app.py            Demo Streamlit
src/training.py   Code huấn luyện dùng chung cho mọi thí nghiệm
src/eda/          Đo chất lượng ảnh, gắn cờ, thống kê
src/preprocess/   Các phép xử lý ảnh, so sánh với baseline
scripts/          Chạy thí nghiệm và phân tích
reports/          Bản nháp báo cáo theo mục
samples/          7 ảnh mẫu cho demo
archive/          Dữ liệu gốc (không commit)
outputs/          Split, kết quả, checkpoint, hình (không commit)
tailieu/          Đề bài, slide, kế hoạch nhóm (không commit)
```

## Cài đặt

Cần Python và GPU CUDA 12.6. Máy không có GPU thì bỏ dòng `--extra-index-url` trong `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tải [HAM10000 từ Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) và giải nén vào `archive/`. Ảnh thuộc giấy phép CC BY-NC 4.0.

## Chạy

Chạy từ thư mục gốc repo. Luôn dùng `python -X utf8` để tránh lỗi tiếng Việt trên console Windows.

```powershell
python -X utf8 scripts/make_splits.py                                # tạo split cố định, chia theo lesion_id
python -X utf8 -m src.eda.quality_metrics --jobs 8                   # đo chất lượng toàn bộ ảnh
python -X utf8 scripts/run_artifact_experiments.py run baseline 42   # train baseline một seed (khoảng 7 phút trên RTX 3050)
streamlit run app.py                                                 # demo, cần outputs/checkpoints/baseline_seed42.pt
```

Danh sách đầy đủ các lệnh và cấu hình thí nghiệm ở `AGENTS.md`.
