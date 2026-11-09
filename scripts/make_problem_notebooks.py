"""Sinh 4 notebook "một vấn đề – một notebook": 01_long, 02_sang_khong_deu, 03_qua_sang, 04_vien_toi.

    python -X utf8 scripts/make_problem_notebooks.py

Notebook chỉ chứa phần trình bày + lời gọi hàm; logic nằm trong src/ (src/report/notebook_utils.py,
src/preprocess/artifacts.py, src/training.py). Chạy lại script sẽ GHI ĐÈ notebook (mất output đã chạy).
"""
import nbformat as nbf

SETUP = '''import os, sys
from pathlib import Path
if not Path("archive").exists() and Path("../archive").exists():
    os.chdir("..")                        # chạy từ thư mục gốc repo
sys.path.insert(0, str(Path.cwd()))
import pandas as pd
from IPython.display import display, Markdown
from src.report import notebook_utils as U
pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 4)
P = U.PROBLEMS[PROBLEM]
NAMES = ["baseline"] + P["configs"]
print("Vấn đề:", P["name"], "| cấu hình so sánh:", NAMES)'''

COUNTS = '''fl = pd.read_csv("outputs/eda/image_flags.csv")
sp = pd.concat([pd.read_csv(f"outputs/splits/{s}.csv")[["image_id"]].assign(split=s) for s in ("train_fit", "val", "test")])
d = sp.merge(fl[["image_id", "dx", P["flag"]]], on="image_id")
tab = d.groupby("split")[P["flag"]].agg(["sum", "size"]).rename(columns={"sum": "số ảnh bị cờ", "size": "tổng"})
tab["%"] = (tab["số ảnh bị cờ"] / tab["tổng"] * 100).round(1)
display(tab.loc[["train_fit", "val", "test"]])
display(d[d[P["flag"]] & (d.split == "test")].dx.value_counts().rename("thành phần lớp của nhóm cờ (test)").to_frame().T)'''

TRAIN = '''# Seed nào đã có kết quả trong outputs/runs_cl/<cấu hình>/seed*/metrics.json thì được đọc lại, không train lại.
# Ctrl+C (hoặc Interrupt kernel) an toàn: chạy lại ô này sẽ tiếp tục từ seed còn thiếu.
for name in NAMES:
    U.train(name)
display(pd.Series(U.run_info(), name="cấu hình lần chạy").to_frame())'''

COMPARE = '''T = U.comparison(NAMES, "test")
Tv = U.comparison(NAMES, "val")
cols = ["cấu hình", "macro_f1_mean", "macro_f1_std", "macro_f1_per_seed", "macro_f1_delta", "macro_f1_ci_lo",
        "macro_f1_ci_hi", "macro_f1_sign", "macro_f1_verdict", "balanced_acc_delta", "balanced_acc_sign", "balanced_acc_verdict"]
print("TEST (n = 2 004)"); display(T[[c for c in cols if c in T]])
print("VAL (n = 1 603) – tập dùng để chọn cấu hình"); display(Tv[[c for c in cols if c in Tv]])'''

BARS = '''U.fig_seed_bars(NAMES, "test", "macro_f1", stem=f"{PROBLEM}_bars_test_macro_f1")
U.fig_seed_bars(NAMES, "val", "macro_f1", stem=f"{PROBLEM}_bars_val_macro_f1")
U.fig_seed_bars(NAMES, "test", "balanced_acc", stem=f"{PROBLEM}_bars_test_bacc");'''

DETAIL = '''print("Recall theo lớp (test): mean ± std 3 seed (Δ so với baseline; dấu theo seed)")
display(U.per_class(NAMES, "test"))
print("Trên CHÍNH nhóm ảnh có vấn đề (cờ EDA) so với phần còn lại")
display(pd.concat([U.subgroup(NAMES, P["flag"], "test"), U.subgroup(NAMES, P["flag"], "val")], ignore_index=True))'''

VERDICT = '''for split, tab in (("test", T), ("val", Tv)):
    for name in P["configs"]:
        display(Markdown("- " + U.verdict_text(tab, name, split)))'''

RULE = """**Cách đọc kết quả (quy tắc đặt trước khi chạy, giống mục 2.5.0 của báo cáo):**
- Mỗi cấu hình train **3 lần** (seed 42, 43, 44) để loại bỏ tính ngẫu nhiên của khởi tạo tham số, thứ tự lấy mẫu và flip.
- Δ = (cấu hình) − (baseline) **ghép cặp theo seed**, trung bình 3 seed. CI95: bootstrap 1 000 lần theo cụm `lesion_id`.
- **"CẢI THIỆN rõ" / "TỆ ĐI rõ"** chỉ khi đủ cả ba: |Δ| > std giữa 3 seed của baseline, CI95 không chứa 0, và cả 3 seed cùng dấu. Không đủ → **"không khác biệt rõ"**.
- Chọn cấu hình theo **val**; test chỉ để báo cáo. Không chọn seed hay ảnh "đẹp"."""

COMMON_TRAIN_MD = """## 4. Áp dụng xử lý ảnh rồi train ResNet-18, 3 lần (seed 42, 43, 44)

- Cấu hình train **giữ nguyên siêu tham số đóng băng** của baseline (mục 2.4): ResNet-18 ImageNet, 224×224, AdamW lr 1e-4, batch 32, 20 epoch, WeightedRandomSampler, flip ngang/dọc, chọn epoch theo val macro-F1. **Chỉ thay đổi một yếu tố**: bước xử lý ảnh, áp ở 600×450 trước resize, giống nhau cho train/val/test.
- Tối ưu tốc độ cho RTX 5060 Ti (không đổi siêu tham số): `channels_last` (+26% thông lượng: 1 724 so với 1 364 ảnh/s ở batch 32), batch đánh giá 512, 32 luồng CPU khi tạo cache ảnh. Đo trên máy này: tăng batch train lên 64–512 **không** nhanh hơn (≈ 1 700–1 850 ảnh/s, GPU đã gần hết công suất tính) nên giữ batch 32.
- Thời gian: ≈ 1,5 phút/seed. Baseline (dùng chung cho 4 notebook) chỉ train một lần ở notebook chạy đầu tiên."""

PROBLEM_TEXT = {
    "long": dict(
        file="01_long.ipynb", title="Vấn đề 1 – Lông che phủ tổn thương",
        problem="""## 1. Vấn đề (từ EDA, mục 2.2)

- **Hiện tượng:** sợi lông (tóc) tối, mảnh (2–10 px), dài, đè lên da và tổn thương; che một phần mạng sắc tố và bờ tổn thương, tạo cạnh giả.
- **Đo bằng:** tỉ lệ pixel "lông" = black-hat 17×17 (phép đóng − ảnh, Ch.7) ≥ 20 mức xám, bỏ đốm < 60 px. **Ngưỡng cờ:** > 5,82% (Q3 + 1,5·IQR) → **1 016/10 015 ảnh (10,1%)**.
- **Kiểm chứng bộ đo:** da đầu 61,7% ảnh nhiều lông, vùng acral (lòng bàn tay/chân) 0%; sau resize 224 tỉ lệ lông vẫn tương quan ρ = 0,86 với ảnh gốc → CNN vẫn "nhìn thấy" lông.""",
        method="""## 2. Áp dụng gì? (Ch.7 hình thái học + khôi phục ảnh Ch.3)

| Cấu hình | Áp cho | Các bước và tham số |
|---|---|---|
| `hair_bh17` | **mọi ảnh** | ảnh xám → **black-hat** elip 17×17 → ngưỡng ≥ 20 → bỏ thành phần < 60 px (trong vùng nhìn thấy) → **giãn** 3×3 → **inpainting Telea** bán kính 5 px trên ảnh màu |
| `hair_bh17_cond` | **chỉ ảnh bị cờ "lông"** (tỉ lệ lông > 5,82%) | như trên; ảnh khác giữ nguyên từng pixel |

- **Lý do tham số:** phần tử cấu trúc 17×17 rộng hơn sợi lông (2–10 px) nên phép đóng lấp được lông → hiệu (black-hat) chỉ còn lại nét tối mảnh; giãn 3×3 để phủ cả viền mờ của sợi lông. Ý tưởng theo DullRazor (Lee và cs., 1997). Inpainting (Telea, 2004) không có trong slide, dùng như bước khôi phục vùng bị che.
- **Giả thuyết (viết trước khi train):**
  - **H1:** xóa lông giúp CNN nhìn rõ cấu trúc tổn thương → kỳ vọng cải thiện **trên nhóm ảnh nhiều lông**.
  - **H2:** black-hat bắt **mọi nét tối mảnh**, kể cả mạng sắc tố trong tổn thương (đặc trưng của `mel`) → bản áp **mọi ảnh** có nguy cơ xóa tín hiệu chẩn đoán, **hại recall `mel`**; bản **có điều kiện** ít rủi ro hơn vì chỉ đụng 10% ảnh.
  - **Hạn chế biết trước:** black-hat chỉ bắt nét **tối**; lông **trắng** (da đầu người lớn tuổi) không được xử lý và có thể làm inpainting lỗi."""),
    "sang_khong_deu": dict(
        file="02_sang_khong_deu.ipynb", title="Vấn đề 2 – Sáng không đều",
        problem="""## 1. Vấn đề (từ EDA, mục 2.2)

- **Hiện tượng:** nền da tối dần về một góc hoặc về phía rìa, chuyển tiếp mềm (khác viền tối: chưa đen hẳn).
- **Đo bằng:** lưới 4×4, 12 ô viền, mỗi ô = trung vị mức xám của **da** (bỏ tổn thương bằng mask Otsu thô); chỉ số = ô sáng nhất − ô tối nhất. **Ngưỡng cờ:** > 38 mức xám (Q3 + 1,5·IQR), bỏ ảnh đã có viền tối → **247/10 015 ảnh (2,5%)**.
- **Liên hệ lỗi CNN (mục 2.8):** sau khi chuẩn hóa theo thành phần lớp, nhóm này có lỗi baseline **cao hơn kỳ vọng** ở cả test (+0,128) và val (+0,145) – vấn đề duy nhất có liên hệ nhất quán ở cả hai tập.""",
        method="""## 2. Áp dụng gì? (Ch.3 lọc không gian + Ch.4 hệ màu)

| Cấu hình | Áp cho | Các bước và tham số |
|---|---|---|
| `flat_L_s60` | **mọi ảnh** | chuyển sang CIE L\\*a\\*b\\*, chỉ xử lý **L\\*** (giữ màu a\\*, b\\*) → ước lượng **nền** B = lọc **Gauss** σ = 60 px chuẩn hóa chỉ trên **da** (bỏ tổn thương + viền đen) → **hiệu chỉnh nền** L' = L · trung vị(B) / B, hệ số kẹp trong [0,5; 2] |
| `flat_L_s60_cond` | **chỉ ảnh bị cờ "sáng không đều"** | như trên; ảnh khác giữ nguyên |

- **Lý do tham số:** σ = 60 px (≈ 1/8 cạnh dài 600) đủ lớn để nền không chứa chi tiết tổn thương, đủ nhỏ để theo được độ dốc sáng; chỉ ước lượng nền trên da để tổn thương tối lớn không bị coi là "nền tối" rồi bị làm sáng lên (mất tương phản); xử lý trên L\\* vì màu là tín hiệu chẩn đoán.
- **Giả thuyết (viết trước khi train):**
  - **H1:** cân bằng nền giúp CNN so sánh tổn thương với da xung quanh nhất quán hơn → kỳ vọng cải thiện **trên nhóm sáng không đều**.
  - **H2:** bản áp **mọi ảnh** đổi nhẹ cả ảnh bình thường (ảnh nào cũng có chút chênh sáng) → hiệu ứng toàn tập nhỏ, có thể lệch khỏi phân phối ImageNet; bản **có điều kiện** chỉ đổi 2,5% ảnh nên tác động toàn tập rất nhỏ."""),
    "qua_sang": dict(
        file="03_qua_sang.ipynb", title="Vấn đề 3 – Quá sáng (bạc màu)",
        problem="""## 1. Vấn đề (từ EDA, mục 2.2)

- **Hiện tượng:** nền da gần trắng, mất màu ("bạc"), mất vân da và chi tiết vùng sáng; tổn thương nhạt màu có bờ mờ.
- **Đo bằng:** tỉ lệ pixel "trắng bạc" trong vùng nhìn thấy (HSV: V ≥ 235 và S ≤ 40, Ch.4). **Ngưỡng cờ:** > 10% → **302/10 015 ảnh (3,0%)**; nhóm này có 62,2% pixel mức xám ≥ 200 so với 5,1% ở phần còn lại.
- **Liên hệ lỗi CNN (mục 2.8):** lỗi baseline của nhóm cao hơn kỳ vọng theo thành phần lớp ở test (+0,167, CI loại 0); val cùng chiều nhưng CI chứa 0.""",
        method="""## 2. Áp dụng gì? (Ch.3 biến đổi lũy thừa + Ch.4 hệ màu)

| Cấu hình | Áp cho | Các bước và tham số |
|---|---|---|
| `gamma_L_1.5` | **mọi ảnh** | **biến đổi lũy thừa** s = r^γ với γ = 1,5 trên L\\* (chuẩn hóa về [0, 1]), giữ a\\*, b\\* |
| `gamma_L_auto` | **chỉ ảnh bị cờ "quá sáng"** | γ = ln(t) / ln(m), m = trung vị L\\* trong vùng nhìn thấy, t = 66,61 (trung vị L\\* của ảnh **không** quá sáng trong `train_fit`), kẹp γ ∈ [1; 3] để chỉ làm tối |

- **Lý do tham số:** γ > 1 nén vùng sáng, kéo dải mức xám cao về giữa → khôi phục chi tiết; đích t lấy từ train_fit (không dùng val/test → không rò rỉ); xử lý trên L\\* để không đổi màu.
- **Giả thuyết (viết trước khi train):**
  - **H1:** đưa độ sáng nhóm quá sáng về mức ảnh bình thường giúp CNN thấy chi tiết sắc tố → kỳ vọng cải thiện **trên nhóm quá sáng**.
  - **H2:** gamma cố định cho **mọi ảnh** làm tối cả ảnh bình thường (lệch phân phối) → có nguy cơ hại toàn tập.
  - **H3 (rủi ro):** một phần ảnh "bạc màu" đến từ **bản chất tổn thương** (nhạt, ít sắc tố – EDA 2.2); làm tối mạnh có thể xóa chính dấu hiệu đó."""),
    "vien_toi": dict(
        file="04_vien_toi.ipynb", title="Vấn đề 4 – Viền tối do ống soi",
        problem="""## 1. Vấn đề (từ EDA, mục 2.2)

- **Hiện tượng:** bốn góc hoặc cả viền ảnh đen kịt, phần nhìn thấy có dạng tròn/bầu dục (khung kính dermoscopy).
- **Đo bằng:** % diện tích vùng đen (mức xám ≤ 20) nối với mép ảnh, sau phép mở 15×15 để bỏ sợi lông (Ch.3 phân ngưỡng + Ch.7). **Ngưỡng cờ:** > 1% → **355/10 015 ảnh (3,5%)**; nhóm có 5,56% pixel ở mức 0–20 so với 0,018%.
- **Tác hại dự kiến:** đỉnh giả ở mức xám ≈ 0 làm lệch histogram (các phép cân bằng lược đồ dành dải mức xám cho viền); CNN có thể dùng "viền tròn" như dấu hiệu thiết bị chụp (ε² theo nguồn ảnh 0,071 > theo lớp 0,042).""",
        method="""## 2. Áp dụng gì? (Ch.3 phân ngưỡng, lọc Gauss + Ch.7)

Cả hai cấu hình chỉ tác động ảnh có viền (> 1% diện tích); ảnh khác giữ nguyên từng pixel.

| Cấu hình | Các bước và tham số |
|---|---|
| `vig_fill` | mask viền (ngưỡng ≤ 20 → mở 15×15 → vùng nối mép) → **tô viền bằng màu da** = trung vị RGB của dải da rộng 25 px sát viền → làm mềm biên bằng **lọc Gauss** 21×21 trên mask. Không đổi tỉ lệ tổn thương |
| `vig_crop` | thu nhỏ khung chữ nhật 4:3 đồng tâm đến khi viền đen ≤ 0,5% khung → **cắt** rồi phóng lại 600×450. Đánh đổi: tổn thương bị **phóng to**, mất da ở góc |

- **Giả thuyết (viết trước khi train):**
  - **H1:** CNN học được bỏ qua vùng đen cố định ở góc → kỳ vọng **ít thay đổi** toàn tập (chỉ 3,5% ảnh bị đổi).
  - **H2:** `vig_crop` đổi đồng thời hai yếu tố (bỏ viền **và** phóng to tổn thương) nên khó diễn giải; `vig_fill` sạch hơn về mặt thí nghiệm."""),
}


def build(key, spec):
    nb = nbf.v4.new_notebook()
    C = nb.cells
    C.append(nbf.v4.new_markdown_cell(f"""# {spec['title']}

**Mục tiêu:** áp dụng phương pháp xử lý ảnh cho đúng vấn đề này → train ResNet-18 **3 lần** → so sánh với baseline (không xử lý) → kết luận **cải thiện / không đổi / tệ đi** và **giải thích vì sao**.

{RULE}

Chạy lần lượt từ trên xuống (Run All). Logic nằm trong `src/`; notebook chỉ gọi hàm và trình bày."""))
    C.append(nbf.v4.new_code_cell(f'PROBLEM = "{key}"\n' + SETUP))
    C.append(nbf.v4.new_markdown_cell(spec["problem"]))
    C.append(nbf.v4.new_code_cell(COUNTS))
    C.append(nbf.v4.new_markdown_cell(spec["method"]))
    C.append(nbf.v4.new_markdown_cell("""## 3. Ảnh trước / sau xử lý

Quy tắc chọn ảnh (cố định, không chọn ảnh "đẹp"): trong nhóm bị cờ ∩ `train_fit`, ảnh có chỉ số EDA gần **trung vị**, gần **phân vị 90** và **lớn nhất**. Bảng dưới: chỉ số EDA trước → sau, % ảnh còn vượt ngưỡng, % ảnh bị đổi và ΔE76 (mức đổi màu trung bình mỗi pixel) trên 100 ảnh bị cờ và 100 ảnh đối chứng (`train_fit`, rút ngẫu nhiên seed 42)."""))
    C.append(nbf.v4.new_code_cell('U.fig_before_after(PROBLEM, stem=f"{PROBLEM}_before_after");'))
    C.append(nbf.v4.new_code_cell("display(U.image_effect(PROBLEM))"))
    C.append(nbf.v4.new_markdown_cell(COMMON_TRAIN_MD))
    C.append(nbf.v4.new_code_cell(TRAIN))
    C.append(nbf.v4.new_markdown_cell("""## 5. So sánh với baseline

### 5.1 Toàn tập: bảng và biểu đồ cột theo từng lần chạy (3 seed)"""))
    C.append(nbf.v4.new_code_cell(COMPARE))
    C.append(nbf.v4.new_code_cell(BARS))
    C.append(nbf.v4.new_markdown_cell("### 5.2 Theo lớp và trên chính nhóm ảnh có vấn đề"))
    C.append(nbf.v4.new_code_cell(DETAIL))
    C.append(nbf.v4.new_markdown_cell("### 5.3 Kết luận tự động từ số liệu (theo quy tắc ở đầu notebook)"))
    C.append(nbf.v4.new_code_cell(VERDICT))
    C.append(nbf.v4.new_markdown_cell("""## 6. Kết luận và giải thích

> **TODO – điền sau khi chạy xong**, dựa trên mục 5 (không điền trước số liệu):
> 1. **Áp dụng gì:** (tóm tắt 1 câu từ mục 2).
> 2. **Kết quả:** cải thiện / không đổi / tệ đi – trích số từ 5.3 (Δ, CI95, dấu theo seed), trên toàn tập **và** trên nhóm ảnh có vấn đề (5.2).
> 3. **Vì sao:** đối chiếu với giả thuyết ở mục 2 và hiệu ứng trên ảnh ở mục 3 (ví dụ: % ảnh bị đổi, ΔE, lớp nào thay đổi).
> 4. **Hạn chế:** cỡ nhóm, 3 seed, bộ phát hiện dùng ngưỡng cố định…"""))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


if __name__ == "__main__":
    for key, spec in PROBLEM_TEXT.items():
        nbf.write(build(key, spec), spec["file"])
        print("Đã tạo", spec["file"])
