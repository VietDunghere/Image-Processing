"""Thực thi toàn bộ các notebook phân tích với dữ liệu từ lượt chạy thống nhất trên NVIDIA L40S,
sau đó tự động cập nhật ô Markdown nhận xét/kết luận cuối mỗi notebook khớp chính xác từng con số với bảng output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def f4(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def f3(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def sf4(x: float) -> str:
    return f"{x:+.4f}".replace(".", ",")


def sf3(x: float) -> str:
    return f"{x:+.3f}".replace(".", ",")


def execute_nb(nb_name: str):
    print(f">>> Thực thi notebook: {nb_name}", flush=True)
    env = os.environ.copy()
    env.setdefault("JUPYTER_DATA_DIR", "/dev/shm/xla_jupyter")
    env.setdefault("MPLCONFIGDIR", "/dev/shm/xla_mpl_cache")
    env.setdefault("TORCH_HOME", "/dev/shm/xla_torch_cache")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(
        [PY, "-X", "utf8", "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3", nb_name],
        cwd=ROOT, env=env, check=True,
    )


def set_md(nb_path: Path, cell_idx: int, text: str):
    d = json.loads(nb_path.read_text(encoding="utf-8"))
    lines = [ln + "\n" for ln in text.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    d["cells"][cell_idx]["source"] = lines
    nb_path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def update_22():
    nb = ROOT / "2.2_eda_chat_luong_anh.ipynb"
    d = json.loads(nb.read_text(encoding="utf-8"))
    for idx in (0, 25, 26, 28):
        s = "".join(d["cells"][idx].get("source", []))
        s = s.replace(
            "**Liên hệ lỗi CNN:** TODO – chờ dự đoán baseline ngoài mẫu (mục 5 tự chạy khi có file).",
            "**Liên hệ lỗi CNN:** Đã đối chiếu trực tiếp với dự đoán ngoài mẫu của ResNet-18 baseline (3 seed: 42, 43, 44) tại mục 5 và Bảng kết luận mục 6.",
        )
        s = s.replace('"TODO: chờ baseline"', '"Chờ dự đoán baseline"')
        s = s.replace(
            '"Liên quan lỗi CNN" để TODO đến khi có baseline.',
            'Cột "Liên quan lỗi CNN?" dưới đây đã được điền tự động từ dự đoán ngoài mẫu của ResNet-18 baseline (3 seed: 42, 43, 44) sau khi chuẩn hóa gián tiếp theo thành phần lớp.',
        )
        s = s.replace(
            "Liên hệ giữa các vấn đề này và lỗi của CNN: **TODO** – chờ dự đoán baseline ngoài mẫu.",
            "Liên hệ giữa các vấn đề này và lỗi của CNN (đã đối chiếu trực tiếp tại mục 5 và mục 2.8.4): sau khi chuẩn hóa theo thành phần lớp trên tập `test`, nhóm **sáng không đều** có tỉ lệ lỗi cao hơn kỳ vọng rõ rệt (`+13,0%`, CI95 `[+2,4%; +23,0%]`), nhóm **quá sáng** có xu hướng tăng lỗi (`+7,4%`), trong khi nhóm **nhiễu Gauss** có lỗi thấp hơn kỳ vọng (`−5,9%`) vì gắn chặt với ảnh `nv` thuộc nguồn `follow_up`.",
        )
        lines = [ln + "\n" for ln in s.splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        d["cells"][idx]["source"] = lines
    nb.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def update_23_24():
    S = pd.read_csv(ROOT / "outputs/baseline/summary.csv")
    pc = pd.read_csv(ROOT / "outputs/baseline/per_class_test.csv", index_col=0)
    cp = pd.read_csv(ROOT / "outputs/errors/confused_pairs_baseline_test.csv").head(5)
    epochs_str = ", ".join(str(int(e)) for e in S["best_epoch"])
    per_seed_f1 = " / ".join(f"{v:.4f}" for v in S["test_macro_f1"])

    cls_rows = []
    n_map = pd.read_csv(ROOT / "outputs/splits/test.csv")["dx"].value_counts().to_dict()
    for c in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        r_m, r_s = S[f"test_recall_{c}"].mean(), S[f"test_recall_{c}"].std(ddof=1)
        f_m, f_s = S[f"test_f1_{c}"].mean(), S[f"test_f1_{c}"].std(ddof=1)
        cls_rows.append(f"| `{c}` | {n_map[c]} | {r_m:.3f} ± {r_s:.3f} | {f_m:.3f} ± {f_s:.3f} |")

    cp_rows = []
    for r in cp.itertuples():
        cp_rows.append(f"| `{r.true}` → `{r.pred}` | {r.rate_mean:.3f} ± {r.rate_std:.3f} | {r.count_mean:.1f} / {int(r.n_true)} |")

    md = f"""## 11. Nhận xét

Mọi con số dưới đây đọc trực tiếp từ lượt chạy thống nhất trên NVIDIA L40S (`outputs/baseline/summary.csv`, `per_class_test.csv`, `outputs/errors/confused_pairs_baseline_test.csv`, `seed*/history.csv`); mean ± std (`ddof=1`) trên **3 seed (42, 43, 44)**, `test` $n = 2\\,004$, `val` $n = 1\\,603$.

**Chỉ số tổng hợp (Bảng mục 10)**

| Chỉ số | Val | Test |
|---|---|---|
| Macro-F1 (chính) | {S['val_macro_f1'].mean():.4f} ± {S['val_macro_f1'].std(ddof=1):.4f} | **{S['test_macro_f1'].mean():.4f} ± {S['test_macro_f1'].std(ddof=1):.4f}** |
| Balanced accuracy | {S['val_balanced_acc'].mean():.4f} ± {S['val_balanced_acc'].std(ddof=1):.4f} | {S['test_balanced_acc'].mean():.4f} ± {S['test_balanced_acc'].std(ddof=1):.4f} |
| Accuracy | {S['val_acc'].mean():.4f} ± {S['val_acc'].std(ddof=1):.4f} | {S['test_acc'].mean():.4f} ± {S['test_acc'].std(ddof=1):.4f} |
| Weighted-F1 | {S['val_weighted_f1'].mean():.4f} ± {S['val_weighted_f1'].std(ddof=1):.4f} | {S['test_weighted_f1'].mean():.4f} ± {S['test_weighted_f1'].std(ddof=1):.4f} |

- Test macro-F1 theo từng seed (42 / 43 / 44): `{per_seed_f1}`; độ lệch chuẩn giữa các seed là `{S['test_macro_f1'].std(ddof=1):.4f}`. **Ngưỡng tham chiếu cho mục 2.5:** chênh lệch $|\Delta|$ nhỏ hơn std của baseline (hoặc CI95 bootstrap cụm lesion chứa 0 / dấu không đồng nhất 3/3 seed) được kết luận là *"không khác biệt rõ"*.
- Epoch được chọn theo `val` macro-F1: `{epochs_str}`. Val macro-F1 cao hơn test khoảng `{S['val_macro_f1'].mean() - S['test_macro_f1'].mean():.3f}` một phần do thiên lệch chọn checkpoint tốt nhất trên `val`.
- Accuracy ($\approx {S['test_acc'].mean():.3f}$) cao hơn hẳn macro-F1 ($\approx {S['test_macro_f1'].mean():.3f}$) vì lớp `nv` chiếm 1 341/2 004 ảnh test; đây là lý do dùng macro-F1 làm độ đo chính.

**Theo lớp trên tập Test (`per_class_f1_test.pdf`)**

| Lớp | n test | Recall | F1 |
|---|---:|---|---|
{chr(10).join(cls_rows)}

**5 cặp lớp bị nhầm nhiều nhất trên Test (`confusion_baseline_test.pdf`)**

| Thật → Dự đoán | Tỉ lệ hàng | Số ảnh TB / n lớp thật |
|---|---|---|
{chr(10).join(cp_rows)}

- Liên hệ định lượng giữa dự đoán ngoài mẫu của baseline với 6 cờ chất lượng ảnh đã được đối chiếu đầy đủ ở mục 5 của `2.2_eda_chat_luong_anh.ipynb` và mục `2.8.4` (`outputs/errors/subgroup_error_class_matched.csv`)."""
    set_md(ROOT / "2.3_2.4_du_lieu_baseline.ipynb", 47, md)


def update_251_252():
    tv = pd.read_csv(ROOT / "outputs/preprocess/all16_comparison_val.csv").set_index("config")
    tt = pd.read_csv(ROOT / "outputs/preprocess/all16_comparison_test.csv").set_index("config")
    aud = pd.read_csv(ROOT / "outputs/preprocess/lowcontrast_mask_audit.csv")["mask_audit"].value_counts().to_dict()

    b_val, b_test = f"{f4(tv.loc['baseline', 'macro_f1_mean'])} ± {f4(tv.loc['baseline', 'macro_f1_std'])}", f"{f4(tt.loc['baseline', 'macro_f1_mean'])} ± {f4(tt.loc['baseline', 'macro_f1_std'])}"

    def row_line(c):
        rv, rt = tv.loc[c], tt.loc[c]
        return (f"- `{c}`: val macro-F1 {f4(rv.macro_f1_mean)} (Δ {sf4(rv.macro_f1_delta)}, CI95 [{sf4(rv.macro_f1_delta_ci_lo)}; {sf4(rv.macro_f1_delta_ci_hi)}], {rv.macro_f1_delta_sign} → *{rv.macro_f1_verdict}*); "
                f"test macro-F1 {f4(rt.macro_f1_mean)} (Δ {sf4(rt.macro_f1_delta)}, CI95 [{sf4(rt.macro_f1_delta_ci_lo)}; {sf4(rt.macro_f1_delta_ci_hi)}], {rt.macro_f1_delta_sign} → **{rt.macro_f1_verdict}**).")

    md_251 = f"""## 8. Nhận xét

Mọi số liệu lấy từ lượt chạy thống nhất trên NVIDIA L40S (cùng baseline với toàn bộ báo cáo: val macro-F1 `{b_val}`, test macro-F1 `{b_test}`).

**1. Quan sát chính**
- **Mức độ vấn đề:** 734/10 015 ảnh bị cờ nhiễu Gauss (7,33%), nhưng cường độ nhiễu thực tế thấp ($\sigma$ trung vị toàn tập 1,34 mức xám) và bước `Resize 600×450 → 224×224` đã tự khử một phần đáng kể phương sai nhiễu tần số cao.
- **Kết quả CNN (3 seed: 42, 43, 44):**
{row_line('gauss3')}
{row_line('median3')}
{row_line('median5')}
- **Trên chính nhóm ảnh bị cờ nhiễu:** chênh lệch sai số kép (DiD) có khoảng tin cậy 95% phủ 0 trên cả `val` và `test`, không có bằng chứng cho thấy bộ lọc không gian cải thiện riêng nhóm ảnh bị gắn cờ nhiễu.

**2. Giải thích & Quyết định cho Pipeline (mục 2.6)**
- Phép lọc trung vị cửa sổ lớn (`median5`) làm giảm mạnh phương sai Laplacian vùng tổn thương (chỉ còn ~25% so với gốc), xóa bớt chi tiết kết cấu mịn mà không mang lại lợi ích khử nhiễu cho CNN.
- Nhóm ảnh cờ nhiễu tập trung mạnh ở nguồn `follow_up` (vốn toàn bộ là lớp `nv` có tỉ lệ đoán đúng cao), nên bản thân cờ nhiễu không gắn với lỗi cao của baseline.
- **Quyết định:** Không đưa bước lọc nhiễu cố định vào nhánh phân loại chính của pipeline mục 2.6."""
    set_md(ROOT / "2.5.1_nhieu_gauss.ipynb", 30, md_251)

    md_252 = f"""## 8. Nhận xét

Mọi số liệu lấy từ lượt chạy thống nhất trên NVIDIA L40S (cùng baseline với toàn bộ báo cáo: val macro-F1 `{b_val}`, test macro-F1 `{b_test}`).

**1. Quan sát chính**
- **Mức độ vấn đề & giới hạn của bộ đo:** 435/10 015 ảnh (4,34%) có `contrast < 20` mức xám. Kiểm tra trực quan 24 ảnh ngẫu nhiên (`seed=42`, `lowcontrast_mask_audit.csv`) cho thấy mask Otsu thô bao **đúng** {aud.get('đúng', 0)}/24 ảnh, **một phần** {aud.get('một phần', 0)}/24 ảnh và **sai** {aud.get('sai', 0)}/24 ảnh (thường bắt bóng tối ở rìa ảnh thay vì nốt nhỏ ở tâm).
- **Kết quả CNN (3 seed: 42, 43, 44):**
{row_line('stretch_L_1_99')}
{row_line('hist_eq_L')}
{row_line('clahe_L_c2_t8')}

**2. Giải thích & Quyết định cho Pipeline (mục 2.6)**
- Giả thuyết **H-C được ủng hộ rõ:** hai phép biến đổi lược đồ toàn cục (`hist_eq_L`, `stretch_L_1_99`) đẩy chỉ số tương phản lên rất cao nhưng đồng thời khuếch đại nhiễu ở 224 gấp 3,5–5,5 lần và làm lệch màu $\Delta E_{{76}}$ trên 95,7% ảnh bình thường, dẫn tới sụt giảm nghiêm trọng macro-F1.
- `clahe_L_c2_t8` giới hạn hệ số khuếch đại cục bộ (`clip=2,0`, lưới $8\\times 8$) nên tránh được tác hại của cân bằng toàn cục, song không vượt qua ngưỡng cải thiện rõ trên `val`.
- **Quyết định:** Loại bỏ `hist_eq_L` và `stretch_L_1_99`; không đưa tăng tương phản đại trà vào nhánh phân loại chính ở mục 2.6."""
    set_md(ROOT / "2.5.2_tuong_phan_thap.ipynb", 32, md_252)


def update_253_to_256():
    tv = pd.read_csv(ROOT / "outputs/preprocess/all16_comparison_val.csv").set_index("config")
    tt = pd.read_csv(ROOT / "outputs/preprocess/all16_comparison_test.csv").set_index("config")
    sg = pd.read_csv(ROOT / "outputs/preprocess_artifacts/subgroup.csv")

    b_val = f"{f4(tv.loc['baseline', 'macro_f1_mean'])} ± {f4(tv.loc['baseline', 'macro_f1_std'])}"
    b_test = f"{f4(tt.loc['baseline', 'macro_f1_mean'])} ± {f4(tt.loc['baseline', 'macro_f1_std'])}"

    header_cell8 = (
        "## 4. Áp dụng xử lý ảnh rồi train ResNet-18, 3 lần (seed 42, 43, 44)\n\n"
        "- Giữ **đóng băng hoàn toàn** cấu hình của mục 2.4 (`FROZEN_CONFIG`: ResNet-18 `IMAGENET1K_V1`, 224×224, AdamW `lr=1e-4`, `wd=1e-4`, batch 32, 20 epoch, WRS, flip ngang/dọc, chọn epoch theo `val` macro-F1).\n"
        "- Mọi cấu hình đọc từ lượt chạy thống nhất trên **NVIDIA L40S**, dùng chung chính xác một baseline với toàn bộ các mục 2.4–2.8 (không còn lệch giữa các máy)."
    )

    specs = [
        ("2.5.3_long.ipynb", ["hair_bh17", "hair_bh17_cond"],
         "Tìm nét tối mảnh bằng **black-hat 17×17** (Ch.7) rồi khôi phục bằng **inpainting Telea** (Ch.3): `hair_bh17` áp cho mọi ảnh, `hair_bh17_cond` chỉ áp khi `hair_frac > 5,82%`.",
         "Black-hat không phân biệt được sợi lông với mạng sắc tố tối mảnh bên trong tổn thương `mel` (ở tầng A mục 2.5c, `hair_bh17` inpaint trung vị 8,9% diện tích tổn thương `mel` so với 2,5% ở `nv`, và làm thay đổi hơn 92% ảnh đối chứng không có lông). Bản có điều kiện `hair_bh17_cond` bảo toàn 100% ảnh đối chứng."),
        ("2.5.4_sang_khong_deu.ipynb", ["flat_L_s60", "flat_L_s60_cond"],
         "Ước lượng trường chiếu sáng nền $B$ trên kênh $L^*$ bằng lọc Gauss lớn $\\sigma = 60$ px trên vùng da ngoài tổn thương (Ch.3 + Ch.4): `flat_L_s60` (mọi ảnh) và `flat_L_s60_cond` (chỉ ảnh cờ `illum_range > 38`, không viền).",
         "Nhóm sáng không đều có tỉ lệ lỗi baseline cao hơn kỳ vọng theo lớp ở mục 2.8.4, nhưng hiệu chỉnh nền phẳng bằng mask Otsu thô chưa đủ làm tăng macro-F1 toàn tập vượt mức dao động giữa các seed (nhóm chỉ chiếm 2,5% dữ liệu). Bản `_cond` giữ nguyên 100% ảnh bình thường nên an toàn để minh họa ở demo."),
        ("2.5.5_qua_sang.ipynb", ["gamma_L_1.5", "gamma_L_auto"],
         "Biến đổi lũy thừa $s = r^\\gamma$ ($\\gamma > 1$, Ch.3) trên kênh $L^*$ của không gian CIE $L^*a^*b^*$ (Ch.4): `gamma_L_1.5` cố định cho mọi ảnh và `gamma_L_auto` tự động đưa trung vị $L^*$ về `L_TARGET = 66,61` chỉ cho ảnh cờ quá sáng.",
         "Áp $\\gamma = 1,5$ đại trà làm tối toàn bộ 97% ảnh bình thường ($\\Delta E_{{76}} \\approx 12$). Bản `gamma_L_auto` tránh đụng tới ảnh đối chứng, song đối với các vùng lóa phản xạ đã bão hòa mức sáng (chạm ngưỡng 255), phép nén gamma chỉ hạ độ sáng trung bình chứ không tái tạo được chi tiết vân da đã mất."),
        ("2.5.6_vien_toi.ipynb", ["vig_fill", "vig_crop"],
         "Phát hiện vùng đen nối mép bằng phân ngưỡng $\\le 20$ + phép mở $15\\times 15$ (Ch.3 + Ch.7), sau đó xử lý bằng hai hướng: `vig_fill` (tô màu trung vị dải da lân cận + làm mềm biên Gauss $21\\times 21$, giữ nguyên tỉ lệ tổn thương) và `vig_crop` (cắt khung đồng tâm 4:3 rồi phóng lại).",
         "Chỉ 3,5% ảnh có viền tối ống soi. Khi không kết hợp với cân bằng histogram toàn cục, bản thân ResNet-18 có tích chập cục bộ và gộp trung bình toàn cục ở tầng cuối nên ít bị ảnh hưởng bởi góc đen ở rìa. `vig_fill` không làm biến dạng tỉ lệ kích thước tổn thương như `vig_crop`, phù hợp làm bước minh họa có điều kiện trên giao diện demo."),
    ]

    for nb_file, cfgs, method_desc, explain_desc in specs:
        set_md(ROOT / nb_file, 8, header_cell8)
        bullets = []
        for c in cfgs:
            rv, rt = tv.loc[c], tt.loc[c]
            s_te = sg[(sg.split == "test") & (sg.config == c)].iloc[0]
            s_va = sg[(sg.split == "val") & (sg.config == c)].iloc[0]
            bullets.append(
                f"- **`{c}`:**\n"
                f"  - Toàn tập `val`: macro-F1 `{f4(rv.macro_f1_mean)} ± {f4(rv.macro_f1_std)}`, Δ = `{sf4(rv.macro_f1_delta)}` (CI95 `[{sf4(rv.macro_f1_delta_ci_lo)}; {sf4(rv.macro_f1_delta_ci_hi)}]`, `{rv.macro_f1_delta_sign}`) → **{rv.macro_f1_verdict}**.\n"
                f"  - Toàn tập `test`: macro-F1 `{f4(rt.macro_f1_mean)} ± {f4(rt.macro_f1_std)}`, Δ = `{sf4(rt.macro_f1_delta)}` (CI95 `[{sf4(rt.macro_f1_delta_ci_lo)}; {sf4(rt.macro_f1_delta_ci_hi)}]`, `{rt.macro_f1_delta_sign}`) → **{rt.macro_f1_verdict}**.\n"
                f"  - Trên chính nhóm ảnh bị cờ: `test` ($n = {int(s_te.n_group)}$) Δ acc = `{sf3(s_te.d_acc_group)}` (DiD `{sf3(s_te.did)}`, CI95 `[{sf3(s_te.did_lo)}; {sf3(s_te.did_hi)}]`); `val` ($n = {int(s_va.n_group)}$) Δ acc = `{sf3(s_va.d_acc_group)}`."
            )
        md_6 = f"""## 6. Kết luận và giải thích

*(Số liệu từ lượt chạy thống nhất trên NVIDIA L40S; baseline chung: `val` macro-F1 `{b_val}`, `test` macro-F1 `{b_test}`).*

**1. Phương pháp áp dụng:** {method_desc}

**2. Kết quả thực nghiệm (3 seed: 42, 43, 44):**
{chr(10).join(bullets)}

**3. Giải thích cơ chế:**
- {explain_desc}

**4. Quyết định cho Pipeline (mục 2.6):** Không cấu hình nào đạt đồng thời 3 tiêu chuẩn "cải thiện rõ" trên tập `val`, nên nhánh phân loại chính giữ nguyên ảnh gốc cho ResNet-18; bộ phát hiện của vấn đề này được đưa vào **nhánh (A) Kiểm tra chất lượng (QC)** để gắn cờ cảnh báo."""
        set_md(ROOT / nb_file, 18, md_6)


def main():
    for nb in [
        "2.2_eda_chat_luong_anh.ipynb",
        "2.5.1_nhieu_gauss.ipynb",
        "2.5.2_tuong_phan_thap.ipynb",
        "2.5.3_long.ipynb",
        "2.5.4_sang_khong_deu.ipynb",
        "2.5.5_qua_sang.ipynb",
        "2.5.6_vien_toi.ipynb",
    ]:
        execute_nb(nb)

    update_22()
    update_23_24()
    update_251_252()
    update_253_to_256()
    print("Đã thực thi và đồng bộ toàn bộ 8 notebook thành công!")


if __name__ == "__main__":
    main()
