# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này. Viết/trả lời bằng **tiếng Việt** (giữ nguyên thuật ngữ kỹ thuật tiếng Anh như CLAHE, ResNet, F1...).

## Đề tài

**Đánh giá các tác động của các phương pháp xử lý ảnh lên mô hình phân loại bệnh trên bộ dữ liệu chụp thực tế sử dụng mạng nơ ron tích chập.**

### Mục đích (quan trọng nhất)

- **KHÔNG** nhằm xây dựng model có độ chính xác cao nhất. Đừng tối ưu hóa số liệu bằng mọi giá (không săn SOTA, không tuning quá mức, không ensemble cho đẹp số).
- Mục tiêu là **lập luận → thực nghiệm → đánh giá**:
  1. Phân tích, lập luận chặt chẽ và đánh giá các phương pháp tăng cường ảnh ở chương 3.
  2. Đánh giá một số kiến trúc mạng tích chập (CNN) trong phân loại ảnh.
  3. Từ đó đề xuất **pipeline hoàn chỉnh end-to-end** để phân loại ảnh.

### Nguyên tắc thực nghiệm

- **Kiểm soát biến (ablation):** mỗi thí nghiệm chỉ thay đổi *một* yếu tố (phương pháp tiền xử lý *hoặc* kiến trúc), giữ nguyên split, seed, siêu tham số, số epoch, augmentation còn lại. Luôn có **baseline không xử lý** (chỉ resize + normalize) để so sánh.
- **Mọi kết luận phải có lập luận + bằng chứng:** giải thích *vì sao* một phương pháp được kỳ vọng giúp/hại (ví dụ: tác động lên histogram, độ tương phản, nhiễu, cạnh), rồi đối chiếu với kết quả đo được. Kết quả "không cải thiện" hay "tệ hơn" cũng là kết quả hợp lệ — báo cáo trung thực, không chọn lọc số liệu.
- **Đánh giá bằng nhiều chỉ số**, không chỉ accuracy (dữ liệu mất cân bằng ~58x): macro-F1, balanced accuracy, precision/recall từng lớp, confusion matrix. Nên chạy nhiều seed (hoặc k-fold) và báo cáo mean ± std để phân biệt cải thiện thật với dao động ngẫu nhiên.
- Có thể bổ sung đánh giá định tính/định lượng ảnh sau xử lý (histogram, entropy, contrast, so sánh trực quan trước–sau) và giải thích bằng Grad-CAM nếu cần lập luận.
- **Không rò rỉ dữ liệu:** phép xử lý có tham số học từ dữ liệu (ví dụ thống kê chuẩn hóa) chỉ được fit trên train. Test set không dùng để chọn phương pháp/siêu tham số — dùng validation tách từ train (vẫn theo `lesion_id`).
- Xử lý ảnh chương 3 áp dụng **trước** `ToTensor/Normalize` (trên ảnh uint8/PIL/ndarray), và phải áp dụng **giống nhau** cho train và test (khác với augmentation ngẫu nhiên chỉ dùng cho train).

## Tài liệu tham chiếu (thư mục `tailieu/`, không commit)

- `DoAn_XuLyAnh.pdf`: yêu cầu của giảng viên (Lựa chọn 2 – Dự án thực tế, báo cáo 4–10 trang, hạn nộp 11/11/2026).
- `ke_hoach_nhom.md`: kế hoạch pipeline 2.1–2.8 và phân công của nhóm.
- `ch1_…pdf` đến `ch8_…pdf`: slide các chương của học phần.

## Dữ liệu

- **HAM10000** (Kaggle), thư mục `archive/` (đã `.gitignore`, ~10k ảnh, không commit):
  - `HAM10000_metadata.csv`: `lesion_id, image_id, dx, dx_type, age, sex, localization`
  - `HAM10000_images_part_1/`, `HAM10000_images_part_2/`: ảnh `.jpg` (ánh xạ qua `image_id`)
  - `hmnist_*.csv`: bản ảnh nhỏ dạng bảng của Kaggle — **không dùng** cho đề tài này (mất chi tiết ảnh).
- 7 lớp (`dx`): `akiec, bcc, bkl, df, mel, nv, vasc`; mã hóa theo `sorted(df["dx"].unique())`. Lớp `nv` chiếm đa số.
- Một `lesion_id` có thể có nhiều ảnh → **bắt buộc chia theo nhóm `lesion_id`** (`StratifiedGroupKFold`, stratify theo `label`) để tránh rò rỉ giữa train/test.
- Split đã lưu ở `outputs/splits/{train_fit,val,test}.csv` (`train` = `train_fit` + `val`; cột `image_id, lesion_id, dx, label, path`). **Dùng lại các file này** cho mọi thí nghiệm để so sánh công bằng; không tạo split mới trừ khi được yêu cầu.

## Cấu trúc repo

**Bản đồ chi tiết ở `AGENTS.md`** (từng notebook làm gì, script, `outputs/`, bẫy cần tránh, lệnh hay dùng). Đọc file đó trước khi mở notebook; notebook rất nặng (tới 6,7 MB). Việc đang dở: `TODO.md`.

Quy ước tên: tiền tố là số mục kế hoạch; notebook và báo cáo cùng phần trùng tên, chỉ khác đuôi; tiền tố `_` là ghi chú làm việc. Notebook phải ở gốc repo.

| Mục | Notebook (gốc repo) | Báo cáo (`reports/`) | Code chính | Trạng thái |
|---|---|---|---|---|
| 1 | – | `1_gioi_thieu.md` | – | Xong |
| 2.1 | – | `2.1_du_lieu.md` | `src/report/data_tables.py` | Xong |
| 2.2 EDA | `2.2_eda_chat_luong_anh.ipynb` | `2.2_eda_chat_luong_anh.md` | `src/eda/` | Xong (kèm liên hệ lỗi CNN) |
| 2.3 | `2.3_2.4_du_lieu_baseline.ipynb` | `2.3_chuan_bi_du_lieu.md` | `scripts/make_splits.py` | Xong |
| 2.4 | `2.3_2.4_du_lieu_baseline.ipynb` | `2.4_baseline.md` | `src/training.py`, `scripts/run_backbone_comparison.py` | Xong (3 kiến trúc CNN × 3 seed) |
| 2.5.1 nhiễu, 2.5.2 tương phản | `2.5.1_nhieu_gauss.ipynb`, `2.5.2_tuong_phan_thap.ipynb` | `2.5a_nhieu_tuong_phan.md`; bản trình bày: `2.5.1_2.5.2_nhieu_tuong_phan.md`; script nói: `_script_thuyet_trinh_DuyAnh.md` | `src/preprocess/problem_report.py` | Xong (đồng bộ L40S) |
| 2.5.3–2.5.6 lông, sáng lệch, quá sáng, viền tối | `2.5.3_long.ipynb` – `2.5.6_vien_toi.ipynb` | `2.5b_long_vien_sang.md` | `src/report/notebook_utils.py`, `src/preprocess/artifacts.py` | Xong (đồng bộ L40S) |
| 2.5c tác động 4 tầng | – | `2.5c_tac_dong_truoc_sau.md` | `src/preprocess/degrade.py`, `scripts/impact_*.py` | Xong (Tầng A–D) |
| 2.6, 2.7 | – | `2.6_2.7_pipeline.md` | `scripts/collect_all_metrics.py` | Xong |
| 2.8 | – | `2.8_danh_gia.md` | `scripts/analyze_errors.py`, `app.py` | Xong (Demo 3 tab + phân tích lỗi) |
| 3 + Overleaf + Slide | – | `3_thao_luan_phan_cong_ai.md`, `overleaf/`, `slides/` | `scripts/build_overleaf_and_slides.py` | Xong (`BaoCao_XuLyAnh_HAM10000.pdf`, `Slide_XuLyAnh_HAM10000.pdf`) |

Khi thêm code mới: ưu tiên tách module/notebook theo vai trò (xử lý ảnh, train, đánh giá) thay vì nhồi vào một notebook; kết quả thí nghiệm lưu vào `outputs/` (CSV/JSON/ảnh) để báo cáo tái tạo được. Thêm file mới thì đặt tên theo quy ước trên và cập nhật bảng này cùng `AGENTS.md`.

## Môi trường & lệnh

- Windows 11, shell chính PowerShell. GPU: RTX 3050 (CUDA 12.6); máy không có GPU thì bỏ dòng `--extra-index-url` trong `requirements.txt`.
- Cài đặt: `python -m venv .venv` → `.venv\Scripts\Activate.ps1` → `pip install -r requirements.txt`
- Chạy notebook từ thư mục gốc repo (đường dẫn tương đối `archive/`, `outputs/`).
- `NUM_WORKERS = 0` (Windows + Jupyter). Nếu chuyển sang script `.py`, có thể tăng `num_workers` và bọc trong `if __name__ == "__main__":`.
- Chạy Python đọc file có tiếng Việt: dùng `python -X utf8` (console mặc định cp1252 sẽ lỗi Unicode).

## Quy ước code

- Tái lập: `SEED = 42`, gọi `seed_everything` (random, numpy, torch, cuda) trước mỗi thí nghiệm.
- Cấu hình hiện tại: `IMG_SIZE=224`, `BATCH_SIZE=32`, chuẩn hóa ImageNet mean/std (dùng pretrained `IMAGENET1K_V1`), `WeightedRandomSampler` + `CrossEntropyLoss` không class weights (tránh bù mất cân bằng hai lần; xem `FROZEN_CONFIG` trong `src/training.py`), `AdamW(lr=1e-4, wd=1e-4)`, 20 epoch, seed 42–44. Đổi các giá trị này chỉ khi có lý do và phải ghi lại, vì sẽ ảnh hưởng tính so sánh giữa các thí nghiệm.
- Augmentation hiện tại (train only): flip ngang/dọc. Giữ cố định giữa các thí nghiệm trừ khi chính augmentation là biến đang khảo sát.
- Comment và markdown trong notebook viết tiếng Việt, ngắn gọn, giải thích *tại sao* chứ không chỉ *làm gì*.
- Không commit dữ liệu, checkpoint, `outputs/` (đã có trong `.gitignore`). Không commit/push nếu chưa được yêu cầu.

## Cách Claude nên làm việc ở đây

- Trước khi code một thí nghiệm, nêu ngắn gọn **giả thuyết** và **biến được thay đổi**; sau khi chạy, nêu kết quả kèm nhận định, kể cả khi kết quả âm.
- Không tự ý chạy huấn luyện dài/tốn tài nguyên khi chưa hỏi; ưu tiên chạy thử trên tập con nhỏ để kiểm tra pipeline trước.
- Đề xuất kết luận cuối cùng (pipeline) phải dựa trên số liệu thí nghiệm, không dựa trên giả định "phương pháp phức tạp hơn thì tốt hơn".

## Hướng dẫn viết báo cáo

Áp dụng khi viết bất kỳ đoạn nào của báo cáo/slide. Báo cáo xuất PDF từ Overleaf (LaTeX), **4–10 trang**, gồm: bài toán, dữ liệu, phương pháp, kết quả, thảo luận, phân công, khai báo sử dụng AI. Phải ghi rõ đã dùng chương nào của học phần (3, 8; thêm 4 – hệ màu, 7 – hình thái nếu dùng) và **tham số + lý do chọn** cho từng bước pipeline.

### Nguyên tắc bắt buộc

1. **Mọi con số phải đến từ file kết quả trong `outputs/`** do code sinh ra. Không bịa, không ước lượng. Chưa có số thì ghi `TODO: chờ <thí nghiệm>`, không điền số giả.
2. **Tách 3 loại câu:** quan sát (có số + hình/bảng), diễn giải (dùng "có thể", "gợi ý"), giả thuyết (viết *trước* thí nghiệm). Không biến giả thuyết thành sự thật khi chưa có bằng chứng.
3. **Kết quả âm vẫn là kết quả:** "X không cải thiện / làm tệ hơn" phải được báo cáo và giải thích. Không chọn seed đẹp. Chênh lệch nhỏ hơn dao động giữa các seed → kết luận "không khác biệt rõ".
4. **Không khẳng định nhân quả** từ tương quan, trừ khi có ablation. Không tuyên bố y khoa.
5. **Cấm câu mơ hồ** ("rõ ràng", "tốt hơn đáng kể", "ảnh bị nhiễu nặng") nếu không kèm số, ngưỡng, tham chiếu hình/bảng. Khuôn: **[số] + [hình/bảng] + [diễn giải thận trọng]**.
6. **Không rò rỉ khi lập luận:** thống kê về lỗi mô hình chỉ dùng dự đoán ngoài mẫu huấn luyện; không dùng test để chọn phương pháp.
7. Chỉ trích dẫn paper/phương pháp chắc chắn tồn tại; không chắc thì ghi "cần kiểm chứng". Mục **khai báo AI** nêu AI dùng ở khâu nào và ai đã kiểm tra.

### Ngân sách trang (tổng ≤ 10)

Giới thiệu 0.5 · Dữ liệu + chuẩn bị 1 · **EDA 1.5–2** · Baseline 0.75 · Tiền xử lý 3 · Kết quả cuối/lỗi/demo 1.5 · Thảo luận 0.5 · Phân công + khai báo AI 0.5. Hình/bảng thừa chuyển phụ lục (hỏi nhóm phụ lục có tính vào 10 trang không).

### EDA (2.2) – phát hiện vấn đề chất lượng ảnh

**Phải trả lời 3 câu:** (1) dataset có những vấn đề chất lượng ảnh nào; (2) mỗi vấn đề có bằng chứng thống kê không (không chỉ nhìn bằng mắt); (3) chúng liên quan thế nào đến lỗi CNN. Phong cách phân tích và chứng minh bằng biểu đồ như paper khoa học.

**Quy trình:**
1. Kiểm tra cơ bản, không đoán: kích thước/mode ảnh có đồng nhất không, ảnh trùng/gần trùng, số ảnh/`lesion_id`, phân bố `dx`, `dx_type`, `localization`, `age`, `sex`, giá trị thiếu.
2. Tính chỉ số chất lượng cho **toàn bộ ảnh** → `outputs/eda/quality_metrics.csv` (một dòng/ảnh, khóa `image_id`); code nằm trong repo, chạy lại được.
3. Thống kê + kiểm định → chọn ảnh minh họa có hệ thống (top/bottom theo chỉ số, không chọn ảnh "đẹp để kể chuyện").
4. Liên hệ với lỗi CNN **chỉ khi đã có dự đoán baseline ngoài mẫu huấn luyện**; chưa có thì để `TODO` đánh dấu rõ, không viết suy đoán thay.
5. Kết thúc bằng **bảng kết luận EDA**.

**Chỉ số ứng viên** (đo rồi mới kết luận có hay không; cột phải là dự kiến cho 2.5, không phải kết luận):

| Vấn đề | Chỉ số gợi ý | Lưu ý | Xử lý dự kiến |
|---|---|---|---|
| Ánh sáng quá sáng/tối | Trung bình L (Lab) hoặc V (HSV); tỉ lệ pixel cháy (≥250) / tối (≤5) | Loại vùng viền đen (vignette) khỏi thống kê | Gamma, cân bằng lược đồ |
| Sáng không đều | Std của trung bình theo lưới (vd. 4×4); nền ước lượng bằng Gauss lớn; so góc với tâm | Phân biệt viền tối quang học với chiếu sáng lệch | CLAHE, chia nền |
| Tương phản thấp | Std của L, khoảng phân vị 5–95, entropy; (nếu có mask thô) chênh tổn thương–da quanh | Mask Otsu/k-means chỉ là ước lượng, nêu rõ | Giãn lược đồ, CLAHE |
| Nhiễu | Ước lượng std nhiễu (Immerkaer / MAD của Laplacian ở vùng phẳng); tỉ lệ pixel cực trị cô lập | Đo cả ở độ phân giải gốc **và** sau resize 224 | Lọc trung vị, Gauss |
| Độ nét (mờ) | Phương sai Laplacian, Tenengrad | Phụ thuộc nội dung; so trong cùng lớp/vùng cơ thể | Unsharp mask |
| Nền phức tạp / vật che | Tỉ lệ pixel lông (black-hat + ngưỡng), thước/mực/bong bóng (kiểm tay trên mẫu), viền tối | Xử lý bằng hình thái học là chương 7 → ghi rõ | Black-hat + inpainting, crop |
| Màu sắc lệch | Trung bình a\*, b\* vùng da quanh; độ lệch gray-world; phân cụm trong không gian màu | Màu là tín hiệu chẩn đoán → cẩn thận khi chuẩn hóa màu | Color constancy (chương 4) |
| Kích thước / góc | Tỉ lệ diện tích tổn thương ước lượng; kích thước ảnh gốc | Nếu ảnh cùng kích thước thì nói rõ | Crop, resize có kiểm soát |

Danh sách vấn đề ở mục 2.5 của file kế hoạch chỉ là ví dụ chung. Kết luận phải từ dữ liệu: vấn đề không tồn tại (vd. nhiễu muối–tiêu) thì ghi bằng chứng là không và loại khỏi 2.5, như đề bài cho phép.

**Bằng chứng thống kê:**
- Tần suất: tỉ lệ ảnh vượt ngưỡng; **ghi rõ ngưỡng và cách chọn** (phân vị, IQR, hoặc nguồn chuẩn); nếu kết luận phụ thuộc ngưỡng thì cho thấy độ nhạy.
- So sánh giữa các lớp `dx` và giữa `dx_type`, `localization`, `sex`, nhóm tuổi: Kruskal–Wallis / Mann–Whitney + **effect size** (epsilon² hoặc Cliff's delta). n lớn thì p-value gần như luôn nhỏ → effect size là chính. Hiệu chỉnh đa so sánh (Holm/Bonferroni).
- Tương quan Spearman giữa các chỉ số để không báo cáo một hiện tượng bằng nhiều chỉ số.
- Một `lesion_id` có nhiều ảnh nên không độc lập: lấy một ảnh/lesion cho kiểm định hoặc nêu rõ hạn chế.
- Cảnh báo confounding: chỉ số chất lượng có thể khác giữa các lớp vì khác nguồn/vùng cơ thể, nguy cơ shortcut learning; không khẳng định khi chưa kiểm chứng.

**Liên hệ lỗi CNN:** so phân phối chỉ số giữa ảnh dự đoán đúng và sai, tách theo lớp; tỉ lệ lỗi theo decile của từng chỉ số kèm khoảng tin cậy bootstrap; có thể dùng hồi quy logistic (lỗi ~ chỉ số + lớp). Nói "liên quan", không "gây ra"; cảnh báo cỡ mẫu nhỏ ở `df`, `vasc`, `akiec`.

**Nhóm ảnh khó:** định nghĩa tiêu chí tái lập được (vd. vượt ngưỡng ≥1 chỉ số, hoặc bị đoán sai nhất quán qua nhiều seed); báo cáo kích thước, thành phần theo lớp, lưới ảnh minh họa, tỉ lệ lỗi so với phần còn lại.

**Bảng kết luận EDA (đầu ra quan trọng nhất, cầu nối sang 2.5):**

| Vấn đề | Tồn tại? (Có/Không/Không rõ) | Bằng chứng (số, hình) | Liên quan lỗi CNN? | Quyết định cho 2.5 |
|---|---|---|---|---|

Vấn đề "Không" cũng phải có dòng và lý do. Mọi vấn đề được xử lý ở 2.5 phải truy ngược về một dòng của bảng này.

**Quy cách biểu đồ như paper:**
- Mỗi hình trả lời một câu hỏi, được trích dẫn và diễn giải trong văn bản. Caption dưới hình, tự đứng được (chỉ số, đối tượng, n, ý chính); caption bảng ở trên.
- Trục có nhãn + đơn vị, ghi n; thang log cho phân phối lệch; không pie, không 3D.
- **Cùng thứ tự và màu cho 7 lớp trên mọi hình**; bảng màu thân thiện người mù màu; chữ đủ lớn khi thu vào cột báo cáo.
- Xuất PDF vector (hoặc PNG ≥300 dpi) vào `outputs/figures/eda/`; script vẽ nằm trong repo.
- So sánh nhóm: box/violin kèm điểm dữ liệu hoặc ECDF, ghi effect size lên hình. Hình ảnh ví dụ: lưới có nhãn lớp + giá trị chỉ số, nêu quy tắc chọn trong caption.
- Bộ gợi ý (4–5 hình, 1–2 bảng): (1) phân phối lớp + ảnh mẫu mỗi lớp; (2) phân phối chỉ số theo lớp; (3) lưới ảnh cực trị cho từng vấn đề; (4) ma trận tương quan chỉ số; (5) tỉ lệ lỗi theo chỉ số (khi có baseline).

### Các mục khác

- **Dữ liệu (2.1, 2.3):** nguồn Kaggle, số ảnh/lớp, cách gán nhãn (`dx_type`), lý do chia theo `lesion_id`, tỉ lệ train/val/test và seed. Nêu trung thực đây là ảnh dermoscopy chứ không phải ảnh điện thoại, và hệ quả cho demo.
- **Baseline (2.4):** bảng cấu hình *đóng băng* (ResNet-18 pretrained, kích thước ảnh, optimizer, lr, epoch, loss/class weights, augmentation, seed); F1-macro **mean ± std trên ≥3 seed**.
- **Tiền xử lý (2.5):** mỗi vấn đề theo khuôn: vấn đề (dẫn bảng kết luận EDA) → phương pháp + lý do + tham số → hình trước/sau → giả thuyết → kết quả so baseline (≥3 seed, biểu đồ cột có sai số + điểm từng lần chạy) → giải thích (kèm F1 theo lớp). Mỗi thí nghiệm chỉ đổi một yếu tố.
- **Pipeline cuối (2.6, 2.7):** thứ tự các bước, tham số, chọn trên validation; test báo cáo một lần; so với baseline được tối ưu tham số tương đương.
- **Đánh giá (2.8):** confusion matrix, phân tích lỗi (cặp lớp hay nhầm, nhóm ảnh khó), demo Streamlit hiển thị từng giai đoạn. Ảnh chụp thực tế ngoài tập chỉ minh họa định tính, ghi rõ lệch miền dữ liệu, không dùng chẩn đoán.
- **Độ đo:** F1-macro (chính), balanced accuracy, precision/recall từng lớp (đặc biệt recall của `mel`, `bcc`, `akiec`), confusion matrix; không chỉ accuracy.
- **Thảo luận:** hạn chế (mất cân bằng, lớp ít ảnh, một tập dữ liệu, ảnh dermoscopy) và hướng cải thiện ngoài phạm vi (kiến trúc lớn hơn, focal loss, ensemble…), nêu rõ là *không thực hiện*.
- **Văn phong:** tiếng Việt học thuật, ngắn gọn; thuật ngữ tiếng Anh kèm giải thích ở lần đầu, dùng nhất quán; số liệu kèm đơn vị, n, và mean ± std ghi rõ trên bao nhiêu lần chạy. EDA nêu vấn đề, phần kết quả nêu hiệu quả xử lý — không lặp ý.

### Checklist trước khi giao bản thảo

- [ ] Mọi số/hình truy ngược được về file trong `outputs/` và script tạo ra nó; không còn số giả hay `TODO` chưa đánh dấu.
- [ ] Mỗi vấn đề có chỉ số, ngưỡng, phân phối, kiểm định + effect size, ảnh minh họa; có bảng kết luận EDA.
- [ ] Thống kê lỗi mô hình dùng dự đoán ngoài mẫu huấn luyện; không chọn phương pháp dựa trên test.
- [ ] Hình đều được trích dẫn, caption tự đứng được, màu/thứ tự lớp nhất quán.
- [ ] Ghi rõ chương đã dùng, tham số + lý do từng bước; có khai báo AI, bảng phân công; 4–10 trang.
