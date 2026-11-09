# 1. Giới thiệu đề tài

> Bản thảo. Mục này chưa chứa kết quả thực nghiệm nên không có con số từ `outputs/` ngoài quy mô dữ liệu (nguồn: `outputs/report_tables/data_checks.json`).
> Phần ghi **[dự kiến]** là kế hoạch, chưa thực hiện, cần cập nhật khi có kết quả.

## 1.1 Bài toán và động cơ

**Bài toán.** Phân loại ảnh tổn thương da thành 7 lớp chẩn đoán (*multi-class classification*). Đầu vào là một ảnh RGB, đầu ra là một trong 7 nhãn `akiec, bcc, bkl, df, mel, nv, vasc`. Dữ liệu là HAM10000 (10 015 ảnh, 7 470 lesion; xem 2.1).

**Khó khăn cốt lõi.**
1. Chất lượng ảnh không đồng đều: viền tối của ống soi, ánh sáng, tương phản, nhiễu, lông. EDA (2.2) đã đo 7 hiện tượng trên toàn bộ 10 015 ảnh.
2. Mất cân bằng lớp: lớp lớn nhất (`nv`) có 6 705 ảnh, lớp nhỏ nhất (`df`) có 115 ảnh, tỉ lệ 58,3 lần (2.1).
3. Một lesion có thể có nhiều ảnh, nên chia tập sai sẽ làm rò rỉ dữ liệu giữa train và test (2.3).

**Động cơ.** Trong thực hành, mạng tích chập (CNN) thường được huấn luyện trực tiếp trên ảnh thô. Câu hỏi của nhóm là các phép xử lý ảnh cổ điển của Chương 3 (cân bằng histogram, CLAHE, gamma, lọc nhiễu, ...) có thực sự giúp CNN hay không, và giúp ở đâu. Câu trả lời không hiển nhiên: CNN tiền huấn luyện có thể đã học được sự bất biến với một phần các biến đổi này, còn một số phép xử lý có thể xóa mất thông tin (màu, kết cấu) mà mô hình đang dùng.

## 1.2 Mục tiêu

Mục tiêu **không** phải xây dựng mô hình có độ chính xác cao nhất. Không tuning siêu tham số, không ensemble, không thay kiến trúc để tăng điểm. Mục tiêu là **lập luận, thực nghiệm, đánh giá**:

1. Phân tích và lập luận có bằng chứng thống kê về các vấn đề chất lượng ảnh của bộ dữ liệu (2.2).
2. Đánh giá tác động của từng phương pháp xử lý ảnh (Chương 3) lên một mô hình CNN cố định (ResNet-18), bằng thí nghiệm kiểm soát biến: mỗi thí nghiệm chỉ đổi một yếu tố, so với baseline không xử lý, chạy từ 3 seed trở lên (2.4, 2.5).
3. Từ số liệu thực nghiệm, đề xuất pipeline end-to-end (2.6–2.8). **[dự kiến]**

## 1.3 Câu hỏi nghiên cứu

| Mã | Câu hỏi | Mục trả lời |
|---|---|---|
| RQ1 | Bộ dữ liệu có những vấn đề chất lượng ảnh nào, và mỗi vấn đề có bằng chứng thống kê không? | 2.2 |
| RQ2 | Với mỗi vấn đề, phương pháp xử lý tương ứng có làm macro-F1 của baseline thay đổi vượt mức dao động giữa các seed không? | 2.5 |
| RQ3 | Ảnh bị gắn cờ có tỉ lệ dự đoán sai khác ảnh còn lại không? Việc xử lý có thu hẹp khác biệt đó không? | 2.2 (mục 4), 2.5, 2.8 |
| RQ4 | Kết hợp các bước có tốt hơn bước đơn lẻ so với baseline không? | 2.6, 2.7 |

Các kết luận chỉ dựa trên chênh lệch lớn hơn độ lệch chuẩn giữa các seed. Kết quả "không cải thiện" hoặc "tệ hơn" vẫn được báo cáo.

## 1.4 Đóng góp

- Bộ chỉ số chất lượng ảnh (7 chỉ số) tính cho toàn bộ ảnh, kèm kiểm chứng độ nhạy của bộ đo bằng nhiễu tổng hợp, và bảng kết luận EDA truy ngược được tới từng thí nghiệm xử lý (2.2).
- Quy trình thực nghiệm không rò rỉ, tái lập được: chia theo `lesion_id`, split và seed cố định, cấu hình baseline đóng băng (2.3, 2.4).
- Đánh giá tác động từng phương pháp bằng mean ± std trên nhiều seed, báo cáo cả F1 từng lớp (2.5). **[dự kiến]**
- Pipeline cuối và bản demo (Streamlit) hiển thị từng giai đoạn xử lý (2.6–2.8). **[dự kiến]**
- Mã nguồn và bảng số liệu tái tạo được (`src/`, `outputs/`).

## 1.5 Kiến thức học phần sử dụng

| Chương | Nội dung dùng | Dùng ở |
|---|---|---|
| 3 – Nâng cao chất lượng ảnh | Histogram, cân bằng/giãn histogram, nhiễu và bộ lọc (trung vị, Gauss) | EDA (đo), 2.5 (xử lý) |
| 8 – Phân loại ảnh | CNN (ResNet-18), độ đo phân loại | 2.4–2.8 |
| 4 – Hệ màu | Không gian HSV để đo vùng "trắng bạc" | EDA |
| 5 – Phát hiện biên | Ngưỡng Otsu, Sobel (ước lượng mask thô, chọn vùng phẳng) | EDA |
| 7 – Xử lý hình thái | Phép mở/đóng, black-hat (đo viền tối, lông) | EDA, 2.5 nếu xử lý lông |
| 6 – Nén ảnh | Chỉ dùng để kiểm tra bảng lượng tử JPEG của ảnh | EDA (kiểm tra cơ bản) |

Danh sách này cần cập nhật theo các phương pháp thực sự được chọn ở 2.5.

## 1.6 Phạm vi và lưu ý

- Dữ liệu là ảnh **dermoscopy** (chụp qua máy soi da), không phải ảnh điện thoại. Hệ quả cho bước demo ở 2.1.
- Báo cáo không đưa ra tuyên bố y khoa; mô hình không dùng để chẩn đoán.
- Bố cục báo cáo: dữ liệu (2.1), EDA (2.2), chuẩn bị dữ liệu (2.3), baseline (2.4), tiền xử lý (2.5), pipeline (2.6, 2.7), đánh giá (2.8), thảo luận.
