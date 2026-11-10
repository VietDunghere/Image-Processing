# Script thuyết trình của Duy Anh

> **Lưu ý đồng bộ (10/11/2026):** Tệp này là bản kịch bản nháp phần riêng của Duy Anh trên nhánh `DuyAnh`. Kịch bản thuyết trình đầy đủ 14 slide cho cả 5 thành viên (đã đồng bộ 100% với bộ slide chính thức `slides/Slide_XuLyAnh_HAM10000.pdf` và số liệu thống nhất trên NVIDIA L40S) nằm tại [`slides/KichBan_ThuyetTrinh_Nhom.md`](../slides/KichBan_ThuyetTrinh_Nhom.md).

Phần trong ngoặc kép là lời nói. Phần *in nghiêng* là thao tác trên slide. Thời lượng ước tính: mở đầu 3 phút, mục 2.1–2.4 khoảng 5 phút, phần của Duy Anh và Dũng 6 phút.

Mọi số trong script có nguồn ở mục "Bảng nguồn số liệu" cuối file.

## Phần 1. Mở đầu

### Chào và giới thiệu đề tài

"Em chào cô và các bạn. Em là Duy Anh, xin đại diện nhóm trình bày đề tài: *Đánh giá các tác động của các phương pháp xử lý ảnh lên mô hình phân loại bệnh trên bộ dữ liệu chụp thực tế sử dụng mạng nơ-ron tích chập*. Đề tài gắn với hai chương chính của học phần là Chương 3, nâng cao chất lượng ảnh, và Chương 8, phân loại ảnh."

### Bài toán và vấn đề

"Bài toán là phân loại ảnh tổn thương da vào 7 lớp bệnh. Nhóm dùng bộ dữ liệu HAM10000, gồm 10 015 ảnh soi da của 7 470 tổn thương."

"Bộ dữ liệu này có ba khó khăn."

"Thứ nhất là mất cân bằng lớp. Lớp nhiều nhất có 6 705 ảnh, lớp ít nhất chỉ 115 ảnh, chênh nhau 58,3 lần. Một mô hình luôn đoán lớp đông nhất vẫn đạt accuracy 66,92%, nhưng macro-F1 chỉ 0,1145. Vì vậy nhóm lấy macro-F1 làm chỉ số chính."

"Thứ hai, một tổn thương có thể có nhiều ảnh. Nhóm chia dữ liệu theo mã tổn thương để tránh rò rỉ: train 6 408 ảnh, validation 1 603 ảnh, test 2 004 ảnh."

"Thứ ba, và là trọng tâm của đề tài, chất lượng ảnh không đồng đều: có ảnh nhiều lông, có viền đen, có ảnh quá sáng, nhiễu, hoặc tổn thương gần như lẫn vào da."

### Mục tiêu

"Mục tiêu của nhóm không phải làm ra mô hình chính xác nhất. Câu hỏi nhóm đặt ra là: các phương pháp xử lý ảnh cổ điển có thực sự giúp mạng CNN phân loại tốt hơn không, và giúp ở đâu. Kết quả không cải thiện cũng là một kết quả, và nhóm sẽ báo cáo trung thực."

"Để trả lời, nhóm cố định một mô hình là ResNet-18 pretrained. Mốc so sánh là baseline chỉ resize và chuẩn hóa, không xử lý ảnh, đạt macro-F1 test 0,6805, độ lệch chuẩn 0,0192 trên 3 lần chạy. Mỗi thí nghiệm sau đó chỉ thêm đúng một bước xử lý ảnh, chạy lại 3 lần, rồi so với baseline."

### Sáu vấn đề và phân công

*Chiếu bảng 6 vấn đề.*

"Từ bước phân tích dữ liệu, nhóm xác định 6 vấn đề có bằng chứng đo được."

"Nhóm bạn Sơn và bạn Ngọc phụ trách 4 vấn đề:"

- "Lông che phủ, 1 016 ảnh. Phương pháp là black-hat để tìm sợi lông, rồi inpainting để tô lại."
- "Viền tối do ống soi, 355 ảnh. Phương pháp là tô viền bằng màu da hoặc cắt bỏ viền."
- "Quá sáng, 302 ảnh. Phương pháp là biến đổi gamma trên kênh độ sáng."
- "Sáng không đều, 247 ảnh. Phương pháp là ước lượng nền bằng lọc Gauss lớn rồi hiệu chỉnh độ sáng."

"Em và bạn Dũng phụ trách 2 vấn đề: nhiễu Gauss, 734 ảnh, và tương phản tổn thương với da thấp, 435 ảnh."

"Ngoài ra nhóm cũng đo quá tối và nhiễu muối tiêu, nhưng chỉ có 40 ảnh và 1 ảnh, nên không xử lý riêng."

## Phần 1B. Mục 2.1–2.4: dữ liệu, EDA, chuẩn bị, baseline

"Trước khi vào từng phương pháp, em xin đi chi tiết hơn vào dữ liệu và mốc so sánh."

### 2.1 Dữ liệu

*Chiếu biểu đồ phân bố lớp.*

"HAM10000 do Tschandl và cộng sự công bố năm 2018, nhóm tải từ Kaggle. Mọi ảnh đều 600 nhân 450, hệ màu RGB. Đây là ảnh chụp qua máy soi da trong thực hành lâm sàng, không phải ảnh điện thoại, nên demo trên ảnh ngoài tập chỉ để minh họa."

"Bảy lớp gồm: nv 6 705 ảnh, chiếm 66,95%; mel 1 113; bkl 1 099; bcc 514; akiec 327; vasc 142; và df 115 ảnh."

"Có hai điểm cần lưu ý. Một là cách gán nhãn không đồng đều: 53,3% ảnh được xác nhận bằng mô bệnh học, 37,0% bằng theo dõi theo thời gian, và toàn bộ 3 704 ảnh theo dõi đều thuộc lớp nv. Hai là có 1 956 tổn thương, tức 26,2%, có từ 2 ảnh trở lên, chiếm 44,9% số ảnh. Các ảnh này không độc lập với nhau, nên ảnh hưởng trực tiếp đến cách chia dữ liệu."

### 2.2 Phân tích chất lượng ảnh (EDA)

*Chiếu bảng tổng hợp 6 vấn đề.*

"Mục này do bạn Sơn thực hiện, trả lời câu hỏi: bộ dữ liệu có những vấn đề gì, và có bằng chứng thống kê không. Nhóm tính 7 chỉ số chất lượng trên toàn bộ 10 015 ảnh, không chỉ nhìn bằng mắt. Ngưỡng gắn cờ mặc định là quy tắc box plot, Q3 cộng 1,5 lần IQR. Khi kiểm định, mỗi tổn thương chỉ lấy một ảnh, và nhóm đọc effect size thay vì chỉ đọc p-value, vì mẫu lớn thì p gần như luôn nhỏ."

"Em xin nêu vài bằng chứng tiêu biểu."

- "Viền tối: nhóm có viền có 5,56% pixel ở mức xám 0 đến 20, nhóm không viền chỉ 0,018%."
- "Quá sáng: 62,2% pixel có mức xám từ 200 trở lên, so với 5,1% ở phần còn lại."
- "Lông: da đầu có 61,7% ảnh nhiều lông, còn lòng bàn tay, bàn chân là 0%, đúng với thực tế, cho thấy chỉ số đo đúng hiện tượng."
- "Sáng không đều: 53,0% ảnh bị cờ có chênh lệch nền lớn hơn một nửa độ tương phản, so với 15,0% ở phần còn lại."

"EDA cũng đưa ra một cảnh báo. Mức nhiễu khác nhau theo nguồn ảnh nhiều hơn theo lớp bệnh, epsilon bình phương 0,187 so với 0,044. Vì vậy CNN có thể học dấu vết cách chụp thay vì đặc điểm bệnh. Đây mới là giả thuyết, nhóm chưa kiểm chứng."

### 2.3 Chuẩn bị dữ liệu

*Chiếu bảng chia tập.*

"Metadata không thiếu nhãn, mọi ảnh đều có file. Tuổi thiếu ở 57 ảnh, giới tính không rõ ở 57 ảnh, vị trí không rõ ở 234 ảnh. Nhóm giữ lại tất cả, vì phần thiếu chỉ nằm ở thông tin phụ, còn ảnh và nhãn đầy đủ."

"Dữ liệu được chia bằng StratifiedGroupKFold với seed 42, nhóm theo mã tổn thương và phân tầng theo nhãn. Kết quả là train 63,98%, validation 16,01%, test 20,01%, tương ứng 4 779, 1 196 và 1 495 tổn thương. Nhóm đã kiểm tra không có tổn thương nào nằm ở hai tập."

"Để xử lý mất cân bằng, nhóm dùng Weighted Random Sampling trên tập train: mỗi ảnh có trọng số bằng 1 chia số ảnh của lớp đó, nên kỳ vọng mỗi lớp chiếm một phần bảy, tức 14,29% mỗi epoch. Đổi lại, ảnh lớp hiếm bị lặp nhiều: trong một epoch mô phỏng, mỗi ảnh df được rút trung bình 12,64 lần. Vì đã cân bằng ở mức dữ liệu, loss không dùng thêm class weights để tránh bù hai lần."

### 2.4 Baseline

*Chiếu bảng cấu hình và kết quả.*

"Baseline là ResNet-18 pretrained trên ImageNet, fine-tune toàn bộ, đầu ra 7 lớp. Ảnh chỉ resize 224 nhân 224 và chuẩn hóa. Nhóm dùng AdamW với learning rate 0,0001, batch 32, 20 epoch, augmentation chỉ có lật ngang và dọc. Cấu hình này được đóng băng cho mọi thí nghiệm, và epoch tốt nhất được chọn theo macro-F1 trên validation."

"Sau 3 seed, macro-F1 test là 0,6805, độ lệch chuẩn 0,0192. Balanced accuracy là 0,6761, còn accuracy là 0,8104. Accuracy cao hơn hẳn vì lớp nv chiếm 1 341 trên 2 004 ảnh test, đó là lý do nhóm không dùng accuracy làm chỉ số chính."

"Theo lớp, F1 cao nhất là nv với 0,909, thấp nhất là mel với 0,530. Cặp nhầm nhiều nhất là mel bị đoán thành nv, tỉ lệ 0,228. Lớp akiec có recall thấp nhất, 0,490."

"Train loss giảm từ khoảng 0,67 xuống 0,043 đến 0,052, trong khi validation loss thấp nhất ở epoch 5, 9, 9 rồi dao động, nên có dấu hiệu overfit. Nhóm vẫn giữ nguyên cấu hình để các thí nghiệm so sánh công bằng."

"Điểm 0,6805 và độ dao động này chính là mốc để đánh giá các phương pháp xử lý ảnh tiếp theo."

## Phần 2. Phần của Duy Anh và Dũng

### Chúng em làm gì trong code

*Chiếu sơ đồ quy trình.*

"Phần của em và bạn Dũng nằm ở hai notebook, 2.5.1 cho nhiễu Gauss và 2.5.2 cho tương phản thấp. Hai notebook đi theo cùng một quy trình: chứng minh vấn đề có thật, đề xuất phương pháp, huấn luyện lại, so với baseline, rồi kiểm tra trên 100 ảnh có vấn đề."

"Nhóm không đề xuất model mới. ResNet-18 được giữ nguyên để chắc chắn chỉ có một yếu tố thay đổi. Cái chúng em đề xuất và thử là 6 thuật toán xử lý ảnh, 3 cho mỗi vấn đề."

"Cách chạy như sau: đọc ảnh gốc 600 nhân 450, áp phép xử lý, resize về 224, chuẩn hóa, rồi đưa vào ResNet-18 huấn luyện 20 epoch. Phép xử lý áp dụng giống nhau cho train, validation và test. Mỗi cấu hình chạy 3 lần với 3 seed, chọn epoch tốt nhất theo validation, test chỉ dùng để báo cáo. Tổng cộng là 21 lượt huấn luyện. Vì kết quả dao động giữa các lần chạy, chúng em train lại baseline ngay trong lượt này, được macro-F1 test 0,6837, độ lệch chuẩn 0,0141, và so sánh với đúng baseline đó."

"Một cấu hình chỉ được kết luận là khác biệt rõ khi đạt cả ba điều kiện: chênh lệch lớn hơn độ lệch chuẩn của baseline, khoảng tin cậy 95% không chứa 0, và cả 3 lần chạy cùng chiều."

### Vấn đề 1: Nhiễu Gauss

*Chiếu hình phân phối σ.*

"Nhiễu là các hạt mịn trên vùng da phẳng. Chúng em đo bằng cách lấy ảnh xám trừ ảnh lọc trung vị, tính độ lệch chuẩn phần dư ở 20% pixel phẳng nhất. Để biết bộ đo có tin được không, chúng em cộng nhiễu đã biết vào ảnh: thêm σ bằng 10 thì đo được 9,91."

"Kết quả có 734 trên 10 015 ảnh, tức 7,33%, có σ vượt ngưỡng 2,16 mức xám. Đối chiếu bằng một công thức khác là Immerkaer, nhóm này có σ 1,582, còn nhóm đối chứng 0,810. Vậy nhóm ảnh nhiễu là có thật, nhưng mức nhiễu nhỏ, chưa tới 1% dải mức xám."

*Chiếu ảnh trước và sau.*

"Chúng em thử 3 thuật toán Chương 3. Lọc Gauss 3 nhân 3 lấy trung bình có trọng số, pixel gần tâm được tính nhiều hơn. Lọc trung vị 3 nhân 3 và 5 nhân 5 sắp xếp các pixel trong cửa sổ rồi lấy giá trị giữa."

"Về mặt ảnh, phép lọc có tác dụng. Sau lọc Gauss, nhiễu ở ảnh 224 của nhóm này giảm từ 0,861 xuống 0,597, ngang với ảnh bình thường là 0,599."

*Chiếu bảng kết quả.*

"Nhưng về phân loại, cả 3 phép đều không khác biệt rõ so với baseline. Lọc Gauss có macro-F1 test tăng 0,0200, cả 3 lần chạy đều tăng, nhưng khoảng tin cậy vẫn chứa 0."

*Chiếu bảng 100 ảnh.*

"Chúng em lấy ngẫu nhiên 100 ảnh bị nhiễu từ validation và test, mỗi tổn thương một ảnh, chọn trước khi xem dự đoán. Một ảnh tính là đúng nếu ít nhất 2 trong 3 lần chạy đoán đúng."

"Trước xử lý, mô hình đúng 92 ảnh, sai 8 ảnh, tức 92%. Sau lọc Gauss, đúng 96 ảnh, sai 4 ảnh: 4 ảnh từ sai thành đúng, không ảnh nào từ đúng thành sai. Lọc trung vị 3 nhân 3 đúng 94 ảnh, trung vị 5 nhân 5 đúng 95 ảnh."

"Tuy vậy, kiểm định McNemar của lọc Gauss có p bằng 0,125, chưa đủ ý nghĩa. Và chỉ đổi seed của chính baseline, số ảnh đúng đã lệch từ âm 2 đến cộng 3 ảnh. Nên chúng em chưa thể khẳng định phép lọc đã khắc phục được lỗi."

"Một lý do có thể là nhiễu vốn nhỏ và bước resize đã làm trơn ảnh sẵn. Kết luận là chưa đưa khử nhiễu vào pipeline."

### Vấn đề 2: Tương phản tổn thương với da thấp

*Chiếu hình phân phối và lược đồ xám.*

"Vấn đề thứ hai là tổn thương có độ sáng gần giống da, ranh giới khó thấy. Chúng em đo bằng chênh lệch mức xám trung bình giữa da và tổn thương. Có 435 ảnh, tức 4,34%, có chênh lệch dưới 20 mức xám."

"Đối chiếu bằng chỉ số không phụ thuộc mask, độ lệch chuẩn mức xám của nhóm này là 12,24, còn phần còn lại 25,96. Khoảng phân vị 5 đến 95 là 38 so với 83. Vấn đề tập trung ở lớp bcc 15,37%, akiec 12,84%, trong khi nv chỉ 2,51%."

"Hạn chế là mask tổn thương chỉ ước lượng thô bằng Otsu. Soát sơ bộ 24 ảnh thấy mask sai 10 ảnh, nên con số 435 chỉ là ước lượng."

*Chiếu ảnh trước và sau.*

"Chúng em thử 3 thuật toán, đều chỉ xử lý kênh độ sáng L của không gian Lab để giữ màu, vì màu là tín hiệu chẩn đoán. Giãn lược đồ kéo khoảng phân vị 1 đến 99 ra toàn thang. Cân bằng lược đồ dùng hàm phân phối tích lũy để làm lược đồ gần đều. CLAHE chia ảnh thành lưới 8 nhân 8, cân bằng từng ô và giới hạn độ dốc để hạn chế khuếch đại nhiễu."

"Về mặt ảnh, cân bằng lược đồ tăng chênh lệch da và tổn thương mạnh nhất, từ 16,5 lên 100,4, nhưng cũng khuếch đại nhiễu 5,51 lần. CLAHE chỉ tăng từ 16,5 lên 19,9."

*Chiếu bảng kết quả.*

"Về phân loại, cân bằng lược đồ làm macro-F1 test giảm 0,1274, giãn lược đồ giảm 0,0724, cả hai được kết luận tệ hơn rõ trên test. CLAHE gần như không đổi, giảm 0,0011."

*Chiếu bảng 100 ảnh.*

"Trên 100 ảnh tương phản thấp, trước xử lý mô hình đúng 73 ảnh, sai 27 ảnh, tức 73%."

- "Sau CLAHE vẫn đúng 73 ảnh, sai 27 ảnh: 5 ảnh được sửa và 5 ảnh bị hỏng."
- "Sau giãn lược đồ, đúng 76 ảnh, sai 24 ảnh: 10 ảnh được sửa, 7 ảnh bị hỏng, p bằng 0,629, không có ý nghĩa."
- "Sau cân bằng lược đồ, chỉ đúng 60 ảnh, sai 40 ảnh: 7 ảnh được sửa nhưng 20 ảnh bị hỏng, p bằng 0,019."

"Bài học ở đây là chỉ số tương phản đẹp hơn không có nghĩa là CNN học tốt hơn. Phép cân bằng lược đồ tăng tương phản nhiều nhất nhưng làm phân loại kém nhất. Nguyên nhân có thể là khuếch đại nhiễu và biến đổi mạnh độ sáng trên mọi ảnh, nhưng thí nghiệm chưa tách riêng được từng yếu tố."

### Chốt phần

"Tóm lại, cả hai vấn đề đều có thật và các phép xử lý thay đổi đúng chỉ số trên ảnh. Tuy nhiên chưa phép nào cải thiện được phân loại ở mức đo được, và hai phép toàn cục còn làm kết quả xấu đi. Vì vậy chúng em đề xuất không đưa hai bước này vào pipeline phân loại."

"Em xin chuyển lời cho bạn tiếp theo."

## Những câu không được nói

| Không nói | Vì sao | Nói thay bằng |
|---|---|---|
| "Lọc Gauss đã cải thiện mô hình" | Khoảng tin cậy chứa 0 | "Có xu hướng tăng nhưng chưa đủ bằng chứng" |
| "Đã chứng minh nhiễu là phân phối Gauss" | Code chưa kiểm định dạng phân phối | "Gauss là giả định theo mô hình Chương 3" |
| "Cân bằng lược đồ làm lệch màu" | Phần màu Δab chỉ 1,22, thay đổi chủ yếu ở độ sáng | "Biến đổi mạnh độ sáng" |
| "Lọc Gauss là tối ưu cho nhiễu Gauss" | Lọc trung bình giảm nhiễu nhiều hơn | "Giữ chi tiết tốt hơn lọc trung bình" |
| "435 ảnh chắc chắn tương phản thấp" | Mask sai ở 10/24 ảnh soát | "Ước lượng 435 ảnh" |
| "Chia theo tổn thương làm điểm thay đổi bao nhiêu" | Nhóm chưa đo bằng thí nghiệm đối chứng | "Về phương pháp, cách chia này tránh được rò rỉ" |
| "WRS tốt hơn class weights" | Chưa chạy so sánh | "Dùng WRS để tránh bù mất cân bằng hai lần" |
| "Vấn đề chất lượng ảnh gây ra lỗi của CNN" | Mới là liên quan, cột liên hệ lỗi CNN ở EDA còn `TODO` | "Có liên quan, chưa khẳng định nguyên nhân" |
| "Mô hình dùng để chẩn đoán" | Đề tài không đưa ra tuyên bố y khoa | "Chỉ phục vụ học tập" |

## Cần làm trước khi nói

- **Soát 24 mask.** Nhãn hiện do AI gán. Nếu nhóm soát lại và số khác 10/24, phải đổi câu trong script.
- **Thống nhất baseline với các bạn.** Mỗi phần so với baseline cùng lượt chạy, nên số khác nhau là bình thường: 0,6805 ở mục 2.4, 0,6837 ở phần của bạn và Dũng, 0,6808 ở 4 notebook của Sơn và Ngọc. Không lấy Δ của phần này so với baseline của phần khác.
- **Kiểm tra số của 4 vấn đề với 2.6.** Báo cáo 2.6 và 4 notebook là hai lượt chạy khác nhau. Ví dụ Δ test của `hair_bh17` là −0,0237 ở 2.6 và −0,0381 ở notebook 2.5.3. Vì vậy script mở đầu chỉ nêu số ảnh và phương pháp của 4 vấn đề, không nêu Δ.
- **Số theo lớp có thể khác slide EDA.** Báo cáo 2.2 ghi `bcc` 14,1%, còn notebook 2.5.2 ghi 15,37% (79/514 ảnh). Script dùng số của notebook.

## Bảng nguồn số liệu

| Số trong script | Nguồn |
|---|---|
| 10 015 ảnh, 7 470 lesion, 6 705 và 115 ảnh, 58,3 lần, 66,92%, 0,1145 | `reports/2.1_du_lieu.md`, mục 1–2 |
| Số ảnh theo lớp, 53,3%, 37,0%, 3 704, 1 956, 26,2%, 44,9% | `reports/2.1_du_lieu.md`, mục 2–4 |
| 5,56%, 0,018%, 62,2%, 5,1%, 61,7%, 53,0%, 15,0%, ε² 0,187 và 0,044 | `reports/2.2_eda_chat_luong_anh.md`, mục 2–3 |
| 6 408 / 1 603 / 2 004; 57, 57, 234; tỉ lệ chia; số lesion; 14,29%; 12,64 | `reports/2.3_chuan_bi_du_lieu.md`, mục 1, 2, 4 |
| 0,6805 ± 0,0192; 0,6761; 0,8104; 1 341; F1, recall theo lớp; 0,228; loss | `reports/2.4_baseline.md`, mục 2–4 |
| 1 016, 734, 435, 355, 302, 247, 40 và 1 ảnh | `reports/2.2_eda_chat_luong_anh.md`, mục 3 |
| Phương pháp của 4 vấn đề | Notebook 2.5.3–2.5.6, mục 2 |
| 0,6837 ± 0,0141; Δ macro-F1; kết luận | Notebook 2.5.1 và 2.5.2, mục 6 |
| 734, 7,33%, 2,16; 435, 4,34% | Notebook 2.5.1 và 2.5.2, mục 4 |
| σ thêm 10 đo được 9,91 | `reports/2.2_eda_chat_luong_anh.md`, mục 2.5 |
| 1,582 và 0,810; 12,24 và 25,96; 38 và 83 | Notebook 2.5.1 và 2.5.2, mục 2 |
| 15,37%, 12,84%, 2,51% | Notebook 2.5.2, mục 3 |
| 10/24 mask sai | Notebook 2.5.2, mục 2 |
| 0,861, 0,597, 0,599; 16,5, 100,4, 19,9; ×5,51 | Notebook 2.5.1 và 2.5.2, mục 5 |
| Kết quả 100 ảnh, McNemar, mức nền −2 đến +3 | Notebook 2.5.1 và 2.5.2, mục 7 |
| Δab 1,22 | `reports/2.5a_nhieu_tuong_phan.md`, Bảng 2.5-2 |
