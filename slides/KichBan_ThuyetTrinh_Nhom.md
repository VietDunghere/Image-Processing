# Kịch Bản Thuyết Trình Nhóm (14 Slide – Đồ Án Xử Lý Ảnh HAM10000)

> Đồng bộ 100% với tệp `slides/Slide_XuLyAnh_HAM10000.pdf` và số liệu thống nhất trên NVIDIA L40S. Thời lượng dự kiến: 12–15 phút + Demo trực tiếp `app.py`.

---

## Phần 1 (Lê Duy Anh & Phan Nguyễn Việt Dũng): Slide 1 – 6 (Giới thiệu, Dữ liệu, EDA, Chuẩn bị & Baseline)

- **Slide 1 (Trang bìa):** Chào thầy/cô và các bạn. Nhóm chúng em xin trình bày đồ án học phần Xử lý ảnh với đề tài: *Đánh giá tác động của các phương pháp xử lý ảnh cổ điển lên phân loại tổn thương da HAM10000 bằng mạng tích chập*.
- **Slide 2 (Bài toán & Câu hỏi nghiên cứu):** Điểm quan trọng nhất trong cách tiếp cận của nhóm là **không chạy theo điểm số SOTA**, mà đi theo hướng nghiên cứu khoa học: đặt giả thuyết từ lý thuyết Chương 3, 4, 5, 7, thiết kế thí nghiệm kiểm soát biến trên CNN Chương 8 và báo cáo trung thực cả các kết quả âm. Nhóm đặt 4 câu hỏi RQ1 đến RQ4 xuyên suốt từ EDA đến pipeline cuối.
- **Slide 3 (Dữ liệu HAM10000):** Bộ dữ liệu gồm 10 015 ảnh soi da kích thước 600×450 của 7 470 tổn thương, chia làm 7 lớp. Có 3 điểm đặc biệt: mất cân bằng 58,3 lần (`nv` chiếm 67% nên đoán toàn `nv` vẫn đạt accuracy 66,9% nhưng macro-F1 chỉ 0,1145); 26,2% tổn thương có nhiều ảnh nên bắt buộc chia theo `lesion_id`; và toàn bộ 3 704 ảnh `follow_up` đều thuộc lớp `nv`. Ngoài ra đây là ảnh soi da chuyên dụng (*dermoscopy*), không phải ảnh điện thoại.
- **Slide 4 (EDA Chất lượng ảnh):** Nhóm đo tự động 7 chỉ số trên toàn bộ 10 015 ảnh, kiểm chứng bộ đo bằng nhiễu tổng hợp trước rồi mới kết luận. Kết quả xác nhận 6 vấn đề tồn tại (lông 10,1%, nhiễu Gauss nhẹ 7,3%, tương phản thấp 4,3%, viền tối 3,5%, quá sáng 3,0%, sáng không đều 2,5%) và bác bỏ nhiễu muối–tiêu (chỉ 1 ảnh) cùng quá tối.
- **Slide 5 (Chuẩn bị dữ liệu & WRS):** Dữ liệu chia cố định theo `lesion_id` thành `train_fit` (6 408), `val` (1 603) và `test` (2 004), kiểm tra 0 rò rỉ lesion. Trên tập train, nhóm dùng `WeightedRandomSampler` để cân bằng kỳ vọng mỗi lớp 14,29%/epoch và không gắn thêm trọng số vào `CrossEntropyLoss` để tránh bù hai lần.
- **Slide 6 (Baseline & Khảo sát 3 kiến trúc CNN):** Dưới cùng cấu hình đóng băng 3 seed (42, 43, 44), trên tập `val` (tập chọn mô hình), `ResNet-18` đạt `0,7254 ± 0,0044` với độ lệch chuẩn thấp hơn 3,3 lần so với `EfficientNet-B0` (`0,7355 ± 0,0146`) và vượt xa `MobileNetV3-Small` (`0,6842 ± 0,0306`), đồng thời có tầng `layer4` 7×7 thuận lợi cho Grad-CAM nên được khóa cố định cho toàn bộ 16 thí nghiệm tiền xử lý. Trên tập `test`, `ResNet-18` đạt `0,6941 ± 0,0160` (không khác biệt thống kê với `EfficientNet-B0` `0,6732`, vượt rõ `MobileNetV3-Small` `0,6320`).

---

## Phần 2 (Đinh Khánh Sơn & Đào Phạm Bích Ngọc): Slide 7 – 10 (Thực nghiệm 2.5a, 2.5b và Giải mã 4 tầng 2.5c)

- **Slide 7 (Nhóm 1 – Nhiễu Gauss & Tương phản thấp):** Ở phần nhiễu, bước resize từ 600×450 về 224×224 vốn đã khử phần lớn nhiễu tần số cao, còn lọc `median5` làm mất 75% chi tiết vùng tổn thương. Ở phần tương phản, hai phép biến đổi toàn cục `hist_eq_L` và `stretch_L_1_99` làm giảm rất mạnh Macro-F1 trên cả 3 seed vì khuếch đại nhiễu gấp 3,5–5,5 lần và làm biến thiên mạnh độ chói $\Delta E_{76}$ (chủ yếu $\Delta L^*$) trên 95,7% ảnh bình thường; trong khi CLAHE cục bộ giữ tính trung tính.
- **Slide 8 (Nhóm 2 – Đại trà vs Có điều kiện):** Với lông, viền tối, quá sáng và sáng không đều, bài học quan trọng nhất là phân biệt **áp đại trà** với **áp có điều kiện theo cờ EDA**. Bản `hair_bh17` đại trà sửa nhầm 92% ảnh không có lông và xóa trung vị 8.9% diện tích vùng tổn thương của u hắc tố `mel`, làm cả 3 seed cùng giảm điểm. Các bản có điều kiện (`_cond`) bảo toàn 100% ảnh bình thường; riêng `gamma_L_auto` cho thấy việc cưỡng bức hạ độ sáng vùng cháy trắng mất sắc độ thành xám tro trên 3% dữ liệu có thể gây lệch phân phối cục bộ (`DiD = -0,209`).
- **Slide 9 (Tổng hợp 16 cấu hình):** Nhìn tổng thể cả 16 cấu hình đơn lẻ và kết hợp trên cùng một biểu đồ, 0/16 cấu hình vượt ngưỡng "cải thiện rõ" trên tập validation.
- **Slide 10 (Giải mã ở 4 tầng phân tích 2.5c):** Tại sao chỉ số ảnh ở Tầng A cải thiện 92–100% mà Macro-F1 không tăng? Vì ResNet-18 có BatchNorm được fine-tune toàn mạng nên tự thích nghi với lệch sáng vừa phải, còn mức nền bất đồng tự nhiên giữa 2 seed baseline (Tầng B) vốn đã là 15,3% trên tập test. Tuy nhiên ở **Stress Test Tầng C**, khi ảnh chịu suy giảm ngoại lai nặng (viền che lớn, lông dày, mất tương phản 0,3×), tiền xử lý ngay ở nhánh `test_only` khôi phục rất mạnh điểm F1 (`+0,10` đến `+0,46`)!

---

## Phần 3 (Nguyễn Trung Nam): Slide 11 – 14 & Demo Streamlit Trực Tiếp (Docker)

- **Slide 11 (Pipeline hai nhánh 2.6–2.7):** Từ số liệu trên, nhóm không ép ảnh qua biến đổi pixel cố định trước CNN mà thiết kế **pipeline 2 nhánh**: Nhánh (A) dùng 6 bộ dò Chương 3–7 làm cổng Kiểm tra chất lượng (QC) gắn cờ cảnh báo; Nhánh (B) phân loại trực tiếp trên ảnh gốc bằng ResNet-18 kèm giải thích Grad-CAM.
- **Slide 12 (Phân tích lỗi 2.8):** Cặp nhầm lớn nhất là `mel` sang `nv` do tương đồng sắc tố. Khi chuẩn hóa tỉ lệ lỗi theo thành phần lớp, hai nhóm *sáng không đều* và *quá sáng* thực sự có tỉ lệ lỗi vượt mức kỳ vọng — khẳng định giá trị thực tế của cờ cảnh báo QC ở Nhánh (A).
- **Slide 13 & Chuyển sang Demo trực tiếp (`docker compose up` / `streamlit run app.py`):** Mời thầy/cô xem bản demo trực tiếp (đã đóng gói Docker container non-root an toàn) gồm 3 tab: (1) chọn ảnh mẫu từ tập test để xem bảng cờ QC và đối chiếu xác suất dự đoán; (2) bật thử thanh trượt Stress Test 2.5c và xem mask hình thái học, bản đồ thay đổi pixel cùng histogram trước/sau; (3) xem trực tiếp bản đồ nhiệt Grad-CAM trên vùng tổn thương và bảng tổng hợp 16 cấu hình.
- **Slide 14 (Kết luận, Phân công & Khai báo AI):** Tóm tắt đóng góp, hạn chế phương pháp luận (1 split, mask Otsu/black-hat thô), bảng phân công thành viên và khai báo minh bạch việc sử dụng AI hỗ trợ. Em xin cảm ơn thầy/cô và xin phép nhận câu hỏi ạ!
