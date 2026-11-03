# 2.2 – EDA: vấn đề chất lượng ảnh trong HAM10000 (theo Chương 3)

> Mọi con số trong tài liệu này do `eda_image_quality.ipynb` sinh ra (đầu ra in trong notebook và các file ở `outputs/eda/`).
> Hình ở `outputs/figures/eda/`. Phần liên hệ với lỗi CNN để **TODO** cho đến khi có dự đoán baseline ngoài mẫu.

## 1. Phạm vi và cách làm

**Phạm vi.** Chỉ xét các vấn đề chất lượng ảnh mà **Chương 3 – Nâng cao chất lượng ảnh** xử lý được: độ sáng, histogram,
tương phản và nhiễu. Lông được xếp vào nhóm nhiễu, vì nó là nhiễu dạng nét tối mảnh che lên ảnh.
Các vấn đề ngoài Chương 3 không đưa vào: nén JPEG, ảnh trùng, màu da, kích thước tổn thương, độ mờ.

**Dữ liệu.** Toàn bộ 10 015 ảnh (7 470 lesion), mọi ảnh đều 600×450, RGB.

**Đo.** Mỗi ảnh được tính một chỉ số chính cho từng vấn đề (`src/eda/quality_metrics.py`), chỉ dùng kỹ thuật trong slide:

| # | Vấn đề | Chỉ số | Kỹ thuật (chương) |
|---|---|---|---|
| 1 | Viền tối do ống soi | % diện tích vùng đen (mức xám ≤ 20) nối với mép ảnh | Phân ngưỡng (Ch.3), phép mở 15×15 để bỏ sợi lông (Ch.7) |
| 2 | Quá sáng / quá tối | % pixel "trắng bạc" (V ≥ 235 và S ≤ 40); mức xám trung bình | Histogram (Ch.3), HSV (Ch.4) |
| 3 | Tương phản thấp | Mức xám TB da − mức xám TB tổn thương | Ngưỡng Otsu (Ch.5), phép đóng/mở (Ch.7) |
| 4 | Sáng không đều | Ô nền sáng nhất − ô tối nhất, lấy 12 ô viền của lưới 4×4 | Thống kê mức xám cục bộ (Ch.3) |
| 5 | Nhiễu Gauss | Std của phần dư (ảnh − lọc trung vị 3×3), đo ở vùng phẳng | Mô hình g = f + η, lọc trung vị (Ch.3), Sobel (Ch.5) |
| 6 | Nhiễu muối–tiêu | % pixel xung: pixel ≤ 10 hoặc ≥ 245 và lệch ≥ 60 so với trung vị 3×3 | Nhiễu xung, lọc trung vị (Ch.3) |
| 7 | Lông | % pixel lông: black-hat 17×17 ≥ 20, bỏ đốm < 60 px | Black-hat = đóng − ảnh (Ch.7) |

Hình 1 minh họa từng bước đo trên một ảnh. Ảnh này được chọn theo quy tắc: trong nhóm "lông", ảnh có `hair_frac` và `contrast` cùng gần trung vị nhất.

**Bảng nhận biết nhanh** (mô tả chi tiết ở đầu mỗi mục 2.x):

| Lỗi | Nhìn bằng mắt | Trên histogram / số liệu |
|---|---|---|
| Viền tối ống soi | Góc ảnh đen kịt, ranh giới là cung tròn, nối với mép ảnh | Đỉnh nhọn ở mức xám 0–20 |
| Quá sáng (bạc màu) | Nền da gần trắng, mất màu, mất vân da | Histogram dồn sang phải (200–255); HSV: V cao, S thấp |
| Quá tối | Toàn ảnh tối, khó tách tổn thương | Histogram dồn sang trái |
| Tương phản thấp | Tổn thương nhạt, "chìm" vào da, bờ mờ | Histogram hẹp, 1 đỉnh; chênh mức xám da − tổn thương nhỏ |
| Sáng không đều | Nền da tối dần về một góc / rìa, chuyển tiếp mềm | Mức xám nền giữa các ô lưới chênh lệch lớn |
| Nhiễu Gauss | Vùng da phẳng bị sần, lấm tấm hạt mịn | Đỉnh histogram bè rộng; phần dư sau lọc trung vị lớn |
| Nhiễu muối–tiêu | Chấm trắng/đen đơn lẻ, sắc, rải rác ngẫu nhiên | Cột nhọn ở đúng mức 0 và 255 |
| Lông | Nét tối mảnh, dài, cong, đè lên tổn thương | Không thấy qua histogram; phản hồi black-hat cao |

**Ngưỡng.**
- Mặc định dùng quy tắc ngoại lai của **box plot**: lớn hơn Q3 + 1,5·IQR (hoặc nhỏ hơn Q1 − 1,5·IQR).
- Dùng ngưỡng tuyệt đối khi phân phối dồn về 0, vì khi đó Q3 = 0 và quy tắc box plot mất ý nghĩa.
- Toàn bộ quy tắc nằm trong `outputs/eda/table_flag_rules.csv`.

**Kiểm định.**
- Lấy 1 ảnh/lesion (n = 7 470), vì các ảnh cùng một lesion không độc lập.
- Dùng Kruskal–Wallis kèm effect size ε²: 0,01 là nhỏ, 0,08 là vừa, 0,26 là lớn. Hiệu chỉnh đa so sánh bằng Holm.
- Với n lớn như vậy, p-value gần như luôn nhỏ, nên effect size mới là con số chính.

**Ảnh minh họa.** Mỗi vấn đề có một lưới ảnh:
- Hàng 1–2: 12 ảnh nặng nhất theo chỉ số.
- Hàng 3–4: 12 ảnh chọn ngẫu nhiên (seed 42) trong nhóm bị gắn cờ.

Nhóm đã xem từng lưới để kiểm tra chỉ số có đo đúng hiện tượng không.

## 2. Kết quả từng vấn đề

### 2.1 Viền tối do ống soi — **Có**, 355 ảnh (3,5%)

#### Mô tả lỗi và dấu hiệu nhận biết

**Mô tả lỗi.** Bốn góc (hoặc cả viền) ảnh bị **đen kịt**, phần nhìn thấy được có dạng hình tròn/bầu dục ở giữa.

**Nguyên nhân.** Ống kính máy soi da (dermoscope) có miệng tròn; khi cảm biến chữ nhật chụp qua miệng tròn, phần ngoài vòng tròn không nhận được ánh sáng.

**Dấu hiệu nhận biết bằng mắt:**
- Góc ảnh đen (mức xám gần 0), ranh giới giữa vùng đen và vùng da là một **cung tròn**;
- vùng đen nối liền với mép ảnh (khác với tổn thương tối nằm giữa ảnh);
- thường kèm theo da ở gần viền tối dần (tối dần trước khi đen hẳn).

**Dấu hiệu trên histogram / số liệu:** một **đỉnh nhọn ở mức xám 0–20** tách rời khỏi phần histogram của da; tỉ lệ diện tích vùng đen nối mép > 1%.

**Tác hại:** viền đen chiếm dải mức xám thấp → cân bằng histogram / chuẩn hóa độ sáng bị kéo lệch; CNN có thể dùng "có viền tròn" như một dấu hiệu của thiết bị chụp.

#### Kết quả đo

- **Số ảnh:** 355 ảnh có hơn 1% diện tích là vùng đen nối với mép ảnh (Hình 2a).
- **Bằng chứng histogram (Hình 2b):** tính trên toàn ảnh, nhóm có viền có **5,56%** pixel ở mức xám 0–20. Nhóm không viền (355 ảnh ngẫu nhiên) chỉ có **0,018%**.
- **Vì sao là vấn đề:** đỉnh giả ở mức xám ~0 làm cân bằng histogram toàn ảnh dành một phần dải mức xám cho viền đen, thay vì cho da và tổn thương.
- **Theo lớp:** tập trung ở mel (8,6%) và bkl (6,7%), so với nv chỉ 1,4%.
- **Kiểm định:** ε² theo `dx_type` là 0,071, lớn hơn theo `dx` là 0,042. Điều này gợi ý viền tối phụ thuộc thiết bị chụp nhiều hơn bệnh.

### 2.2 Quá sáng (bạc màu) — **Có**, 302 ảnh (3,0%); quá tối — **Không**, 40 ảnh

#### Mô tả lỗi và dấu hiệu nhận biết

**Mô tả lỗi.** Ảnh **quá sáng**: da gần như trắng, màu nhạt "bạc", chi tiết vùng sáng (vân da, mạng sắc tố nhạt) bị mất.
Ảnh **quá tối**: toàn ảnh tối, khó phân biệt tổn thương với da.

**Nguyên nhân.** Đèn LED của máy soi quá mạnh / quá gần, phơi sáng tự động sai, da phản xạ mạnh (có gel, dầu).

**Dấu hiệu nhận biết bằng mắt (quá sáng):**
- nền da trắng hoặc hồng rất nhạt, gần như không còn màu;
- tổn thương trông như "dán" lên nền trắng, viền mờ phần nhạt bị mất;
- không thấy vân da / mao mạch ở vùng sáng.

**Dấu hiệu trên histogram / số liệu:** histogram **dồn về bên phải** (mức xám 200–255), nhiều pixel chạm 250–255;
trong HSV: độ sáng V rất cao và độ bão hòa S rất thấp (pixel "trắng bạc"). Quá tối thì ngược lại: histogram dồn về bên trái.

**Tác hại:** chi tiết nằm trong dải mức xám bị nén (vài mức xám cuối) → mất thông tin kết cấu; màu da khác thường làm lệch đặc trưng màu.

#### Kết quả đo

- **Số ảnh:** 302 ảnh có hơn 10% vùng nhìn thấy là pixel "trắng bạc" (Hình 3a).
- **Bằng chứng histogram (Hình 3b):** histogram lệch phải.
  - Ở nhóm quá sáng, **62,2%** pixel có mức xám ≥ 200, so với 5,1% ở phần còn lại.
  - Mức xám trung bình là 191,2 so với 155,8.
  - Độ bão hòa của da (S) là 32,1 so với 63,1.
- **Vì sao là vấn đề:** chi tiết vùng sáng bị nén vào ít mức xám. Gamma γ > 1 hoặc cân bằng histogram sẽ trải lại dải này.
- **Theo lớp:** tỉ lệ đều giữa các lớp, từ 2,1% đến 4,6%.
- **Kiểm định:** ε² theo `dx_type` là 0,197, gấp đôi theo `dx` là 0,088.
- **Quá tối:** chỉ 40 ảnh (0,4%) có mức xám trung bình dưới ngưỡng box plot 105,6. Vì quá ít ảnh nên không xử lý riêng.

### 2.3 Tương phản tổn thương–da thấp — **Có**, 435 ảnh (4,3%)

#### Mô tả lỗi và dấu hiệu nhận biết

**Mô tả lỗi.** Tổn thương có mức sáng/màu **gần giống da xung quanh** – khó thấy tổn thương bắt đầu và kết thúc ở đâu.

**Nguyên nhân.** Bản chất tổn thương (nhạt màu, phẳng, ít sắc tố – hay gặp ở bcc, akiec, bkl) cộng với điều kiện chụp (ánh sáng khuếch tán, quá sáng).

**Dấu hiệu nhận biết bằng mắt:**
- tổn thương nhạt, hồng/nâu rất nhẹ, "chìm" vào nền da;
- **bờ tổn thương mờ**, không vẽ được đường viền rõ ràng;
- ảnh tổng thể trông "phẳng", ít vùng rất sáng và rất tối.

**Dấu hiệu trên histogram / số liệu:** histogram **hẹp và chỉ có 1 đỉnh** (da và tổn thương chồng lên nhau) – ảnh tương phản tốt có 2 đỉnh tách rời;
khoảng phân vị 5–95 nhỏ, độ lệch chuẩn mức xám nhỏ; chênh lệch mức xám da − tổn thương nhỏ (< 20).

**Tác hại:** bờ và hình dạng tổn thương (dấu hiệu chẩn đoán quan trọng) khó nhìn thấy với CNN; phân ngưỡng tách tổn thương dễ sai.

#### Kết quả đo

- **Số ảnh:** 435 ảnh có độ chênh mức xám giữa da và tổn thương nhỏ hơn 20 (khoảng 8% dải 0–255) (Hình 4a).
- **Độ nhạy theo ngưỡng:** ngưỡng 15 cho 141 ảnh, 20 cho 435 ảnh, 25 cho 859 ảnh. Kết luận "có vấn đề" không phụ thuộc ngưỡng, nhưng số ảnh thì có.
- **Bằng chứng histogram (Hình 4b):**
  - Ảnh tương phản thấp có histogram **một đỉnh, hẹp**. Ảnh tương phản trung vị có hai đỉnh tách rời.
  - Trên toàn nhóm, khoảng phân vị 5–95 của histogram là **38** mức xám, so với **84** ở phần còn lại.
  - Độ lệch chuẩn mức xám là 12,3 so với 26,4.
  - Đây đúng là dấu hiệu "histogram hẹp" mà Chương 3 dùng để nhận ra ảnh tương phản thấp.
- **Theo lớp:** bcc 14,1%, akiec 12,7%, df 9,6%, bkl 8,5%. Trong khi đó mel chỉ 1,6% và nv 2,2%.
- **Kiểm định:** ε² theo `dx` là 0,115 (mức vừa), theo `dx_type` chỉ 0,008. Tương phản thấp gắn với loại tổn thương, không gắn với nguồn ảnh.
- **Hạn chế:** mask Otsu chỉ là ước lượng thô.

### 2.4 Sáng không đều — **Có**, 247 ảnh (2,5%)

#### Mô tả lỗi và dấu hiệu nhận biết

**Mô tả lỗi.** Độ sáng của **nền da** thay đổi theo vị trí: một phía / một góc sáng, phía kia tối, dù da ở đó cùng màu.

**Nguyên nhân.** Đèn chiếu lệch một bên, máy soi đặt nghiêng, bề mặt da cong (tai, mũi, ngón tay), độ sáng giảm dần ra rìa ống kính.

**Dấu hiệu nhận biết bằng mắt:**
- nền da **tối dần** về một góc hoặc về phía rìa (khác viền tối: chưa đen hẳn, chuyển tiếp mềm);
- vùng da tối ở rìa có mức sáng gần bằng tổn thương;
- cùng một tổn thương nhưng nửa sáng, nửa tối.

**Dấu hiệu trên histogram / số liệu:** histogram toàn ảnh bị **giãn rộng / có đuôi** do nền có nhiều mức sáng khác nhau;
chia ảnh thành lưới ô, mức xám trung vị của nền giữa các ô viền chênh lệch lớn (> 38 mức xám).

**Tác hại:** một ngưỡng toàn cục (Otsu) hay cân bằng histogram toàn cục không dùng được cho cả ảnh – vùng da tối bị coi là tổn thương;
CNN có thể nhầm vùng tối do ánh sáng với vùng sắc tố.

#### Kết quả đo

- **Số ảnh:** 247 ảnh có chênh lệch giữa ô nền sáng nhất và tối nhất trên 38 mức xám, tức vượt ngưỡng box plot (Hình 5a).
  - Đã loại các ảnh thuộc nhóm viền tối.
  - Có 204 ảnh không đo được vì có quá ít da.
- **Vì sao là vấn đề (Hình 5b):**
  - Ngưỡng Otsu nằm khoảng giữa mức xám da và mức xám tổn thương.
  - Vì vậy, nếu nền tối đi hơn **một nửa** độ tương phản, da tối sẽ rơi xuống dưới ngưỡng và bị coi là tổn thương.
  - Điều này xảy ra ở **53,0%** ảnh bị gắn cờ, so với 15,0% ở phần còn lại.
  - Kết luận: nhóm này cần xử lý cục bộ (CLAHE), cân bằng toàn cục là không đủ.
- **Theo lớp:** mel 8,1%, so với nv 1,3%.
- **Hạn chế:** khi xem lưới ảnh, hàng ngẫu nhiên đúng là ảnh tối dần về góc. Tuy nhiên, một phần ảnh cực trị là do tổn thương rất lớn tràn vào ô viền.

### 2.5 Nhiễu Gauss — **Có nhưng nhỏ**; nhiễu muối–tiêu — **Không**

#### Mô tả lỗi và dấu hiệu nhận biết

**Mô tả lỗi.** Nhiễu là các biến thiên **ngẫu nhiên** của mức xám không thuộc về nội dung ảnh (mô hình Ch.3: g = f + η).
- **Nhiễu Gauss:** mọi pixel lệch một chút quanh giá trị thật → ảnh "lấm tấm", "hạt" đều khắp ảnh.
- **Nhiễu muối–tiêu (xung):** một số pixel lẻ bị đổi hẳn thành trắng (255) hoặc đen (0).

**Nguyên nhân.** Gauss: nhiễu cảm biến, chụp thiếu sáng phải tăng độ nhạy (ISO). Muối–tiêu: lỗi truyền/ghi dữ liệu, pixel chết của cảm biến.

**Dấu hiệu nhận biết bằng mắt:**
- Gauss: vùng da đáng lẽ phẳng trông **sần, lấm tấm hạt mịn**, rõ nhất ở vùng tối và vùng đồng màu khi phóng to;
- muối–tiêu: **chấm trắng / đen đơn lẻ**, sắc nét, rải rác ngẫu nhiên, không liên quan cấu trúc da.

**Dấu hiệu trên histogram / số liệu:** Gauss làm các đỉnh histogram **bè rộng** ra; ở vùng phẳng, hiệu "ảnh − lọc trung vị 3×3" có độ lệch chuẩn lớn.
Muối–tiêu tạo các **cột nhọn ở đúng mức 0 và 255**; pixel cực trị lệch rất xa trung vị của 8 pixel lân cận.

**Tác hại:** nhiễu tạo cạnh / kết cấu giả, làm sai các bộ lọc phát hiện cạnh (Ch.5); CNN có thể học kết cấu nhiễu đặc trưng của thiết bị.

#### Kết quả đo

**Kiểm chứng bộ đo (Hình 6).** Lấy 30 ảnh ngẫu nhiên, cộng thêm nhiễu tổng hợp theo các mô hình của Chương 3, rồi đo lại:

| Nhiễu cộng thêm | Kết quả đo được (trung vị) |
|---|---|
| Gauss σ = 0 / 2 / 5 / 10 | σ = 1,33 / 2,39 / 5,09 / 9,91 |
| Muối–tiêu p = 0,1% / 1% / 5% | 0,099% / 0,98% / 4,97% |

Cả hai bộ đo đều bám sát lượng nhiễu thật. Vì vậy, nếu kết quả trên dữ liệu thật thấp, đó là do dữ liệu thật ít nhiễu, chứ không phải do bộ đo không nhạy.

**Nhiễu Gauss trên dữ liệu thật (Hình 7a–b).**
- σ có trung vị **1,34** mức xám và phân vị 99 là 2,76. Chỉ 2 ảnh có σ > 5.
- Ảnh 224 có σ trung vị 1,25, tức là resize cũng làm mịn ảnh.
- Có 734 ảnh vượt ngưỡng box plot 2,16. Đã loại các ảnh nhiều lông, vì sợi lông bạc bị tính nhầm là nhiễu.
- Mức nhiễu này nhỏ. Kỳ vọng lọc trung bình, Gauss hay trung vị chỉ tác động nhỏ.
- σ phụ thuộc **nguồn ảnh** (ε² theo `dx_type` = 0,187) nhiều hơn phụ thuộc lớp bệnh (ε² theo `dx` = 0,044).

**Nhiễu muối–tiêu (Hình 7c).**
- 347 ảnh có ít nhất một pixel xung, nhưng chỉ 1 ảnh vượt 0,1%. Mức cao nhất là 0,116%.
- Kết luận: không có nhiễu muối–tiêu, loại khỏi 2.5.

### 2.6 Lông — **Có**, 1 016 ảnh (10,1%)

#### Mô tả lỗi và dấu hiệu nhận biết

**Mô tả lỗi.** Sợi lông (tóc) nằm **đè lên da và tổn thương** – có thể coi là một loại nhiễu dạng nét: không thuộc về đối tượng cần nhận dạng.

**Nguyên nhân.** Tổn thương ở vùng nhiều lông (da đầu, mặt, ngực, chân) và không cạo lông trước khi chụp.

**Dấu hiệu nhận biết bằng mắt:**
- các **nét tối, mảnh (2–10 px), dài, cong**, cắt ngang ảnh, thường nhiều sợi song song hoặc đan chéo;
- lông đè lên tổn thương, che mất một phần mạng sắc tố và bờ tổn thương;
- có cả lông bạc/trắng (sáng hơn da) ở người lớn tuổi.

**Dấu hiệu trên histogram / số liệu:** lông ít ảnh hưởng tới histogram (diện tích nhỏ) nên **không** nhìn thấy qua histogram;
nhận biết bằng phép **black-hat** (đóng − ảnh): nét tối hẹp hơn phần tử cấu trúc cho phản hồi cao; tỉ lệ pixel lông lớn.

**Tác hại:** lọc trung vị 3×3 không xóa được (sợi lông rộng hơn cửa sổ lọc); lông tạo cạnh mạnh giả, che chi tiết chẩn đoán,
và CNN có thể học "nhiều lông ↔ vùng cơ thể ↔ lớp bệnh".

#### Kết quả đo

- **Số ảnh:** 1 016 ảnh có tỉ lệ pixel lông vượt ngưỡng box plot 5,8% (Hình 8a).
- **Kiểm chứng chỉ số đo đúng (Hình 8b):** da đầu (scalp) có **61,7%** ảnh nhiều lông, mặt 19,2%, còn vùng acral (lòng bàn tay, bàn chân) là 0%. Phân bố này khớp với thực tế giải phẫu.
- **Còn sau khi resize 224:** tương quan Spearman giữa tỉ lệ lông đo ở ảnh gốc và ở ảnh 224 là ρ = 0,86. CNN vẫn nhìn thấy lông.
- **Vì sao là vấn đề:** sợi lông dày 2–10 px và dài hàng trăm px. Lọc trung vị 3×3 không xóa được chúng, trong khi chúng che tổn thương và tạo cạnh giả.
- **Theo lớp:** mel 17,4%, bkl 17,1%, akiec 12,7%, so với nv 8,7%.

## 3. Tổng hợp (Hình 9)

- **Số ảnh mỗi vấn đề (Hình 9a).** Có 6 vấn đề với tập ảnh từ ~250 ảnh trở lên:

  | Vấn đề | Số ảnh |
  |---|---|
  | Lông | 1 016 |
  | Nhiễu Gauss | 734 |
  | Tương phản thấp | 435 |
  | Viền tối | 355 |
  | Quá sáng | 302 |
  | Sáng không đều | 247 |

  Hai vấn đề gần như không có: quá tối (40 ảnh) và muối–tiêu (1 ảnh).
- **Không trùng lặp chỉ số (Hình 9c).** Mọi cặp chỉ số có |ρ| ≤ 0,4, nên mỗi chỉ số đo một hiện tượng riêng.
- **Cảnh báo nguồn ảnh (Hình 9b, `table_kruskal_eps2.csv`).** Ba chỉ số khác nhau theo nguồn ảnh (`dx_type`) nhiều hơn theo lớp bệnh:

  | Chỉ số | ε² theo `dx_type` | ε² theo `dx` |
  |---|---|---|
  | Trắng bạc | 0,197 | 0,088 |
  | Nhiễu σ | 0,187 | 0,044 |
  | Viền tối | 0,071 | 0,042 |

  Ảnh `follow_up` chỉ có ở lớp nv. Vì vậy CNN có thể học dấu vết của cách chụp thay vì đặc điểm bệnh. Đây mới là giả thuyết, cần kiểm chứng ở 2.5/2.8.

## 4. Liên hệ với lỗi CNN

**TODO: chờ dự đoán baseline ngoài mẫu.** Mục 5 của notebook sẽ tự tính tỉ lệ lỗi ở ảnh bị gắn cờ so với ảnh không bị gắn cờ, khi có file `outputs/baseline/predictions*.csv`.

## 5. Bảng kết luận EDA (`outputs/eda/eda_conclusion.csv`)

| Vấn đề | Tồn tại? | Bằng chứng | Liên quan lỗi CNN? | Dự kiến cho 2.5 |
|---|---|---|---|---|
| Viền tối ống soi | Có | 355 ảnh; 5,56% pixel ở mức 0–20 so với 0,018% (Hình 2) | TODO | Bỏ viền khỏi ảnh / khỏi thống kê histogram |
| Quá sáng / bạc màu | Có | 302 ảnh; 62,2% pixel ≥ 200 so với 5,1% (Hình 3) | TODO | Gamma γ > 1, cân bằng histogram |
| Quá tối | Không (quá ít) | 40 ảnh dưới ngưỡng box plot (Hình 3) | — | Không xử lý riêng |
| Tương phản thấp | Có | 435 ảnh; histogram 1 đỉnh, P5–P95 là 38 so với 84 (Hình 4) | TODO | Giãn / cân bằng histogram |
| Sáng không đều | Có | 247 ảnh; 53% có chênh nền > ½ tương phản (Hình 5) | TODO | CLAHE |
| Nhiễu Gauss | Có nhưng nhỏ | σ trung vị 1,34 mức xám; 734 ảnh trên ngưỡng (Hình 6–7) | TODO | Lọc trung vị / Gauss, kỳ vọng tác động nhỏ |
| Nhiễu muối–tiêu | Không | 1 ảnh > 0,1% pixel xung; bộ đo nhạy tới 0,1% (Hình 6) | — | Loại khỏi 2.5 |
| Lông | Có | 1 016 ảnh; da đầu 61,7% so với acral 0% (Hình 8) | TODO | Black-hat + inpainting |

## 6. Hạn chế

- **Mask tổn thương chỉ là ước lượng thô.** Nó được tạo bằng Otsu cộng hình thái học, vì bộ dữ liệu không có mask chuẩn. Ba chỉ số phụ thuộc vào mask: tương phản, sáng không đều và vùng da.
- **Ngưỡng là quy ước thống kê.** Quy tắc box plot không phải chuẩn y khoa. Với tương phản thấp, độ nhạy theo ngưỡng đã được báo cáo ở 2.3.
- **Có hiện tượng gây nhiễu chỉ số.** Lông bạc làm tăng σ nhiễu, và tổn thương lớn làm tăng chỉ số sáng không đều. Một phần đã được loại bỏ: ảnh nhiều lông bị loại khỏi cờ nhiễu, và nền chỉ lấy da sáng hơn ngưỡng Otsu.
- **Khác biệt giữa các lớp chưa phải nhân quả.** Các lớp khác nhau cả về nguồn ảnh và vùng cơ thể.

## 7. Danh sách file

- `src/eda/quality_metrics.py`: đo chỉ số. Lệnh chạy: `python -X utf8 -m src.eda.quality_metrics --jobs 8`, mất khoảng 3 phút với 14 tiến trình.
- `src/eda/flags.py`: quy tắc gắn cờ và ngưỡng.
- `src/eda/stats.py`: các hàm thống kê.
- `src/eda/plots.py`: màu và thứ tự lớp, lưới ảnh.
- `eda_image_quality.ipynb`: toàn bộ phân tích, tạo mọi bảng và hình.
- `outputs/eda/`:
  - `quality_metrics.csv`, `hist_gray.npy`
  - `table_flag_rules.csv`, `table_kruskal_eps2.csv`, `noise_detector_validation.csv`
  - `image_flags.csv` (cờ từng vấn đề cho mỗi ảnh), `eda_conclusion.csv`
  - `problem_sets/{vignette, overexposed, underexposed, low_contrast, uneven_illum, noise, hair}.csv`: danh sách ảnh của từng vấn đề, sắp theo mức độ, để dùng ở 2.5.
- `outputs/figures/eda/`:
  - Hình kèm PDF:

    | File | Nội dung |
    |---|---|
    | `fig02_vignette` | Viền tối |
    | `fig03_exposure` | Quá sáng / quá tối |
    | `fig04_low_contrast` | Tương phản thấp |
    | `fig05_uneven_illum` | Sáng không đều |
    | `fig06_noise_validation` | Kiểm chứng bộ đo nhiễu |
    | `fig07_noise` | Nhiễu trên dữ liệu thật |
    | `fig08_hair` | Lông |
    | `fig09_summary` | Tổng hợp |

  - Chỉ có PNG: `fig01_how_we_measure.png` (các bước đo) và `grid_{vignette, overexposed, underexposed, low_contrast, uneven_illum, noise, hair}.png` (lưới ảnh).
  - Thư mục này còn một số file của các phiên bản trước mà tôi không xóa được tự động. Chỉ những file liệt kê ở đây là bản hiện hành.
