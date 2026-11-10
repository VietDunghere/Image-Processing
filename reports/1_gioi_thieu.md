# 1 Giới thiệu đề tài

> Nguồn số liệu quy mô dữ liệu: `outputs/report_tables/data_checks.json`. Toàn bộ hệ thống thực nghiệm đã hoàn tất đồng bộ trên cùng một môi trường phần cứng (NVIDIA L40S, PyTorch CUDA 12.6).

## 1.1 Bài toán và động cơ

**Bài toán.** Phân loại ảnh tổn thương da thành 7 lớp chẩn đoán (*multi-class classification*). Đầu vào là một ảnh dermoscopy RGB kích thước $600\times 450$, đầu ra là một trong 7 nhãn `akiec, bcc, bkl, df, mel, nv, vasc`. Bộ dữ liệu sử dụng là **HAM10000** (10 015 ảnh thuộc 7 470 tổn thương `lesion_id`; xem mục 2.1).

**Ba thách thức cốt lõi.**
1. **Chất lượng ảnh không đồng đều:** viền tối cơ học của ống soi, vùng lóa trắng bạc do phản xạ đèn LED/gel, tương phản thấp giữa tổn thương nhạt màu và nền da, chiếu sáng lệch góc, nhiễu cảm biến và sợi lông che phủ (mục 2.2 đo định lượng trên toàn bộ 10 015 ảnh).
2. **Mất cân bằng lớp nghiêm trọng:** lớp đa số `nv` (6 705 ảnh) gấp **58,3 lần** lớp thiểu số `df` (115 ảnh), khiến độ đo accuracy đơn thuần bị chi phối bởi lớp `nv` (mục 2.1).
3. **Cấu trúc cụm nhiều ảnh trên cùng một tổn thương:** 26,2% `lesion_id` có từ 2 đến 6 ảnh (chiếm 44,9% tổng số ảnh), đòi hỏi chia tập theo nhóm `lesion_id` và kiểm định thống kê theo cụm để chống rò rỉ dữ liệu (mục 2.3).

**Động cơ nghiên cứu.** Trong học sâu hiện đại, các mạng tích chập (CNN) tiền huấn luyện trên ImageNet thường được đưa trực tiếp ảnh thô vào huấn luyện. Câu hỏi đặt ra là: *các phương pháp nâng cao chất lượng ảnh cổ điển của Chương 3 (lọc không gian, giãn/cân bằng lược đồ, CLAHE, biến đổi gamma, hiệu chỉnh nền) kết hợp xử lý màu Chương 4 và hình thái học Chương 7 có thực sự cải thiện hiệu năng phân loại của CNN hay không, giúp ở đâu và khi nào gây tác dụng phụ?*

## 1.2 Mục tiêu

Đề tài **không** chạy theo điểm số SOTA, không tinh chỉnh siêu tham số quá mức hay kết hợp mô hình (ensemble). Trọng tâm xuyên suốt là chu trình khoa học **lập luận $\rightarrow$ thực nghiệm có kiểm soát $\rightarrow$ đánh giá trung thực**:

1. Định lượng và kiểm định thống kê các vấn đề chất lượng ảnh trên toàn bộ 10 015 ảnh HAM10000, kèm kiểm chứng độ nhạy bộ đo (mục 2.2).
2. Khảo sát các kiến trúc CNN đại diện (`ResNet-18`, `EfficientNet-B0`, `MobileNetV3-Small`) trên cấu hình đóng băng và thiết lập đường cơ sở (baseline) chuẩn mực (mục 2.4).
3. Thực hiện nghiên cứu triệt tiêu (*ablation study*) trên 16 cấu hình tiền xử lý (đơn lẻ, có điều kiện và kết hợp, mỗi cấu hình 3 seed độc lập), phân tích sâu ở 4 tầng tác động: mức pixel ảnh, mức đổi nhãn dự đoán, mức suy giảm tổng hợp (*stress test*) và bản đồ nhiệt chú ý Grad-CAM (mục 2.5).
4. Đề xuất pipeline hoàn chỉnh hai nhánh (nhánh kiểm tra chất lượng QC gắn cờ + nhánh phân loại CNN), phân tích lỗi chuyên sâu và xây dựng ứng dụng minh họa tương tác trên Streamlit (mục 2.6–2.8).

## 1.3 Câu hỏi nghiên cứu

| Mã | Câu hỏi nghiên cứu | Mục trả lời |
|---|---|---|
| **RQ1** | Bộ dữ liệu HAM10000 tồn tại những vấn đề chất lượng ảnh nào có bằng chứng thống kê, và chúng liên hệ ra sao với lỗi của CNN sau khi chuẩn hóa thành phần lớp? | 2.2, 2.8.4 |
| **RQ2** | Với từng vấn đề, phép xử lý cổ điển tương ứng (áp đại trà hoặc chỉ áp có điều kiện trên ảnh bị cờ) có tạo ra thay đổi macro-F1 vượt ngưỡng dao động ngẫu nhiên giữa các seed không? | 2.5a, 2.5b |
| **RQ3** | Nếu macro-F1 toàn tập không tăng, tiền xử lý tác động thực sự ở tầng nào (chỉ số ảnh, đổi nhãn từng mẫu, khả năng chống chịu suy giảm nặng hay vùng chú ý Grad-CAM)? | 2.5c |
| **RQ4** | Việc ghép nối tuần tự nhiều bước tiền xử lý và lựa chọn kiến trúc CNN tác động thế nào đến thiết kế pipeline đầu–cuối (end-to-end)? | 2.4, 2.6, 2.7 |

## 1.4 Đóng góp chính

- Bộ 7 chỉ số chất lượng ảnh tự động trên 10 015 ảnh kèm kiểm chứng độ nhạy bằng nhiễu tổng hợp và đối chiếu sai số CNN chuẩn hóa gián tiếp theo lớp (mục 2.2, 2.8).
- Khuôn khổ thực nghiệm kiểm soát biến nghiêm ngặt trên cùng một phần cứng (NVIDIA L40S): chia tập theo `lesion_id`, cố định siêu tham số, đánh giá 3 kiến trúc CNN và 16 cấu hình tiền xử lý qua 3 seed (42, 43, 44) kèm bootstrap theo cụm tổn thương (mục 2.3–2.5).
- Phân tích giải mã nghịch lý *"cải thiện chỉ số ảnh nhưng không tăng F1 của CNN"* qua 4 tầng (ảnh, mức nền đổi nhãn, stress test 18 mức suy giảm và Grad-CAM `layer4`), đặc biệt chỉ rõ cơ chế xóa nhầm mạng sắc tố u hắc tố (`mel`) khi xóa lông đại trà (mục 2.5b, 2.5c, 2.8.5).
- Pipeline hai nhánh thực tiễn cùng ứng dụng Streamlit 3 tab tích hợp QC, chẩn đoán thị giác từng bước xử lý, mô phỏng suy giảm và giải thích Grad-CAM (mục 2.6–2.8, `app.py`).

## 1.5 Kiến thức học phần sử dụng

| Chương học phần | Kỹ thuật sử dụng | Vai trò trong đề tài |
|---|---|---|
| **Chương 3 – Nâng cao chất lượng ảnh** | Phân ngưỡng, lược đồ xám, giãn tuyến tính, cân bằng lược đồ, CLAHE (mở rộng cục bộ), biến đổi lũy thừa gamma, lọc trung bình/Gauss/trung vị, khôi phục vùng che phủ (inpainting Telea – mở rộng) | Đo chỉ số EDA (2.2) và 16 cấu hình tiền xử lý (2.5) |
| **Chương 4 – Hệ màu** | Không gian HSV (phát hiện vùng lóa trắng bạc), mở rộng sang CIE $L^*a^*b^*$ để tách kênh độ sáng $L^*$ khỏi sắc độ $a^*, b^*$ và đo độ lệch màu $\Delta E_{76}$ | EDA (2.2), tăng tương phản / gamma / hiệu chỉnh nền (2.5) |
| **Chương 5 – Phát hiện biên & phân vùng** | Đạo hàm Sobel (chọn vùng phẳng đo nhiễu), phân ngưỡng tự động Otsu (ước lượng thô vùng tổn thương) | EDA (2.2), đo ảnh giai đoạn 1 (2.5) |
| **Chương 6 – Nén ảnh** | Kiểm tra định dạng, kích thước và cấu trúc nén JPEG của toàn bộ tập ảnh | Kiểm tra cơ bản EDA (2.1, 2.2) |
| **Chương 7 – Xử lý hình thái học** | Phép đóng, phép mở, phép giãn và biến đổi **black-hat** (phát hiện viền tối ống soi và sợi lông tối mảnh) | EDA (2.2), xóa lông & xử lý viền tối (2.5b) |
| **Chương 8 – Phân loại & nhận dạng ảnh** | Mạng tích chập sâu (`ResNet-18`, `EfficientNet-B0`, `MobileNetV3-Small`), học chuyển giao, đánh giá đa lớp mất cân bằng và giải thích Grad-CAM | Baseline, ablation và pipeline cuối (2.4–2.8) |

## 1.6 Phạm vi và lưu ý đạo đức

- Bộ dữ liệu HAM10000 gồm ảnh **soi da chuyên dụng (*dermoscopy*)**, không phải ảnh chụp bằng camera điện thoại thông thường; hệ quả lệch miền dữ liệu đối với bản demo được nêu tường minh tại mục 2.1 và 2.8.
- Toàn bộ kết quả phục vụ mục đích nghiên cứu học thuật trong học phần Xử lý ảnh, **tuyệt đối không dùng làm công cụ chẩn đoán lâm sàng**.
