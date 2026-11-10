# 3 Thảo luận, Phân công công việc và Khai báo sử dụng AI

## 3.1 Thảo luận tổng hợp: Vì sao xử lý ảnh cổ điển không tăng Macro-F1 cho CNN trên HAM10000?

Kết hợp bằng chứng xuyên suốt từ mục 2.2 đến 2.8 cho phép rút ra 4 nhận định khoa học trung tâm:

1. **Sự khác biệt căn bản giữa "tối ưu thị giác cho mắt người / thuật toán phân ngưỡng cổ điển" và "đặc trưng học sâu có giám sát":**
   Ở Tầng A (mục 2.5c), các phép xử lý cổ điển thực hiện rất tốt mục tiêu thiết kế của Chương 3 và Chương 7 (triệt tiêu 92%–100% cờ viền tối, lông, quá sáng, sáng lệch; kéo giãn mạnh độ tương phản). Tuy nhiên, khác với bộ phân ngưỡng Otsu cố định dễ sụp đổ khi chiếu sáng lệch, mạng ResNet-18 có các tầng `BatchNorm` và 11,18 triệu tham số được tinh chỉnh toàn bộ (*end-to-end fine-tuning*) trên 6 408 ảnh huấn luyện nên tự học được tính bất biến tương đối với các biến thiên độ sáng và tương phản vừa phải của thiết bị dermoscope.
2. **Bước thu nhỏ độ phân giải (`600×450 → 224×224`) đã đóng vai trò bộ lọc thông thấp tự nhiên:**
   Cường độ nhiễu cảm biến trên HAM10000 vốn thấp ($\sigma$ trung vị 1,34 mức xám) và phép nội suy song tuyến có khử răng cưa của PIL đã triệt tiêu phần lớn phương sai nhiễu tần số cao trước khi ảnh đi vào tầng tích chập đầu tiên. Áp thêm lọc không gian (đặc biệt `median5`) làm suy hao chi tiết vi cấu trúc vùng tổn thương nhiều hơn lượng nhiễu còn lại.
3. **Đánh đổi lợi – hại khi áp đại trà lên tập ảnh mà chỉ 2,5%–10,1% ảnh thực sự gặp lỗi:**
   - Hai phép biến đổi lược đồ toàn cục (`hist_eq_L`, `stretch_L_1_99`) làm biến dạng màu sắc ($\Delta E_{76}$ từ 14 đến 22) và khuếch đại nhiễu gấp 3,5–5,5 lần trên hơn 95% ảnh bình thường, trong khi màu sắc tuyệt đối và độ chuyển sắc êm lại là tín hiệu chẩn đoán cốt lõi của tổn thương da $\rightarrow$ làm giảm mạnh macro-F1.
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
- Sử dụng độ phân giải đầu vào cao hơn ($384\times 384$, giữ nguyên tỉ lệ khung hình 4:3 bằng đệm viền) và hàm mất mát nhận biết mất cân bằng lớp như *Focal Loss* hoặc *Class-Balanced Loss*.

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
