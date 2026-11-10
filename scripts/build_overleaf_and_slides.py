"""Sinh trọn bộ dự án Overleaf LaTeX (overleaf/main.tex + figures/ + references.bib -> BaoCao_XuLyAnh_HAM10000.pdf + zip)
và bộ Slide thuyết trình Beamer 16:9 (slides/slides.tex -> Slide_XuLyAnh_HAM10000.pdf + kịch bản thuyết trình nhóm),
đọc trực tiếp 100% con số từ các tệp CSV/JSON thực nghiệm thống nhất trong outputs/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OVL = ROOT / "overleaf"
SLD = ROOT / "slides"


def f4(x: float) -> str:
    return f"{float(x):.4f}".replace(".", "{,}")


def f3(x: float) -> str:
    return f"{float(x):.3f}".replace(".", "{,}")


def f2(x: float) -> str:
    return f"{float(x):.2f}".replace(".", "{,}")


def f1(x: float) -> str:
    return f"{float(x):.1f}".replace(".", "{,}")


def sf4(x: float) -> str:
    s = f"{float(x):+.4f}".replace(".", "{,}")
    return s.replace("-", "$-$") if s.startswith("-") else f"${s}$"


def sf3(x: float) -> str:
    s = f"{float(x):+.3f}".replace(".", "{,}")
    return s.replace("-", "$-$") if s.startswith("-") else f"${s}$"


def tt_code(s: str) -> str:
    return "\\texttt{" + str(s).replace("_", "\\_") + "}"


def tex_esc(s: str) -> str:
    return (str(s)
            .replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("_", "\\_"))


FIG_MAP = {
    "class_distribution.pdf": "figures/baseline/class_distribution.pdf",
    "wrs_distribution.pdf": "figures/baseline/wrs_distribution.pdf",
    "learning_curves.pdf": "figures/baseline/learning_curves.pdf",
    "per_class_f1_test.pdf": "figures/baseline/per_class_f1_test.pdf",
    "backbone_comparison.pdf": "figures/baseline/backbone_comparison.pdf",
    "eda_summary.pdf": "figures/eda/fig09_summary.pdf",
    "eda_noise_val.pdf": "figures/eda/fig06_noise_validation.pdf",
    "eda_cnn_link.pdf": "figures/eda/fig10_cnn_error_by_flag.pdf",
    "eda_how_measure.png": "figures/eda/fig01_how_we_measure.png",
    "all16_macro_f1.pdf": "figures/preprocess/fig_all16_macro_f1.pdf",
    "contrast_metrics.pdf": "figures/preprocess/fig_p2_contrast_metrics.pdf",
    "artifacts_before_after.png": "figures/preprocess_artifacts/fig_2_5_6_before_after.png",
    "subgroup_delta.pdf": "figures/preprocess_artifacts/fig_a2_subgroup_delta_test.pdf",
    "confusion_test.pdf": "figures/errors/confusion_baseline_test.pdf",
    "mel_hair_flips.png": "figures/errors/grid_mel_hair_flips_test.png",
    "always_wrong.png": "figures/errors/grid_always_wrong_test.png",
    "impact_a1_ecdf.pdf": "figures/impact/fig_a1_target_metric_ecdf.pdf",
    "impact_b1_flips.pdf": "figures/impact/fig_b1_flip_vs_floor_test.pdf",
    "impact_c2_stress.pdf": "figures/impact/fig_c2_stress_delta_heatmap.pdf",
    "impact_d1_gradcam.pdf": "figures/impact/fig_d1_gradcam_energy.pdf",
    "impact_d3_grid.png": "figures/impact/fig_d3_gradcam_grid.png",
}


def copy_figures():
    for dst_root in (OVL / "figures", SLD / "figures"):
        dst_root.mkdir(parents=True, exist_ok=True)
        for dst_name, rel_src in FIG_MAP.items():
            src = OUT / rel_src
            if not src.exists() and src.suffix == ".pdf":
                alt = src.with_suffix(".png")
                if alt.exists():
                    src = alt
            if src.exists():
                shutil.copy2(src, dst_root / dst_name)
            else:
                print(f"[CẢNH BÁO] Chưa tìm thấy hình: {src}")


def build_overleaf():
    S = pd.read_csv(OUT / "baseline/summary.csv")
    bb_val = pd.read_csv(OUT / "backbones/comparison_val.csv").set_index("arch")
    bb_test = pd.read_csv(OUT / "backbones/comparison_test.csv").set_index("arch")
    tv = pd.read_csv(OUT / "preprocess/all16_comparison_val.csv").set_index("config")
    tt = pd.read_csv(OUT / "preprocess/all16_comparison_test.csv").set_index("config")
    sub_err = pd.read_csv(OUT / "errors/subgroup_error_class_matched.csv")
    cp_te = pd.read_csv(OUT / "errors/confused_pairs_baseline_test.csv").head(5)
    cs_te = pd.read_csv(OUT / "errors/consistency_by_class_test.csv").set_index("dx")
    ht = pd.read_csv(OUT / "impact/hair_lesion_tests.csv").iloc[0]
    hb = pd.read_csv(OUT / "impact/hair_lesion_by_class.csv").set_index("dx")
    sc_l3 = pd.read_csv(OUT / "impact/stress_delta.csv").query("split == 'test' and metric == 'macro_f1' and level == 3")
    mel_row = pd.read_csv(OUT / "errors/mel_row_hair.csv")
    sg = pd.read_csv(OUT / "preprocess_artifacts/subgroup.csv")
    fb = pd.read_csv(OUT / "impact/flips_excess.csv")
    S_hcond = pd.read_csv(OUT / "runs_cl/hair_bh17_cond/summary.csv")
    app_chk = json.loads((OUT / "errors/app_consistency_check.json").read_text(encoding="utf-8"))
    triv = pd.read_csv(OUT / "report_tables/t24_trivial_classifier_test.csv", index_col=0)["luon_doan_nv"].to_dict()

    # Table 2.2 EDA & CNN Error Link
    eda_meta = [
        ("F_hair", "Lông che phủ", "Black-hat $17{\\times}17 \\ge 20$ ($>5{,}82\\%$ FOV)", "1\\,016 (10{,}1\\%)"),
        ("F_noise", "Nhiễu Gauss", "$\\sigma_{\\text{dư}} > 2{{,}}16$ mức xám (vùng phẳng)", "734 (7{,}3\\%)"),
        ("F_low_contrast", "Tương phản thấp", "$\\bar{g}_{\\text{da}} - \\bar{g}_{\\text{tổn thương}} < 20$ (Otsu)", "435 (4{,}3\\%)"),
        ("F_vignette", "Viền tối ống soi", "Mức xám $\\le 20$ nối mép $>1\\%$ diện tích", "355 (3{,}5\\%)"),
        ("F_overexposed", "Quá sáng (bạc màu)", "Pixel $V{\\ge}235, S{\\le}40 > 10\\%$ FOV", "302 (3{,}0\\%)"),
        ("F_uneven_illum", "Sáng không đều", "Chênh nền lưới $4{\\times}4 > 38$ mức xám", "247 (2{,}5\\%)"),
    ]
    eda_rows_tex = []
    for f_code, v_name, rule_s, cnt_s in eda_meta:
        rt = sub_err[(sub_err.split == "test") & (sub_err.flag == f_code)].iloc[0]
        ex_s = f"{sf3(rt.excess)} [{sf3(rt.excess_ci_lo)}; {sf3(rt.excess_ci_hi)}]"
        eda_rows_tex.append(f"{v_name} & {rule_s} & {cnt_s} & ${f3(rt.err_group)}$ & {ex_s} \\\\")
    eda_rows_tex.append("Quá tối / Muối--tiêu & $\\bar{g}<105{,}6$ / xung $>0{,}1\\%$ & 40 / 1 ảnh & -- & Bác bỏ (quá ít mẫu) \\\\")

    # Table 2.4 Backbones
    bb_rows_tex = []
    for arch in ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]:
        rv, rt = bb_val.loc[arch], bb_test.loc[arch]
        if arch == "resnet18":
            d_s = "Đối chứng"
        else:
            v_short = "k.rõ" if rt.macro_f1_verdict == "không khác biệt rõ" else "tệ rõ"
            d_s = f"{sf4(rt.macro_f1_delta)} ({rt.macro_f1_sign}, {v_short})"
        bb_rows_tex.append(
            f"{rt.label} & ${f2(rt.params_M)}$M & ${f4(rv.macro_f1_mean)} \\pm {f4(rv.macro_f1_std)}$ & "
            f"$\\mathbf{{{f4(rt.macro_f1_mean)} \\pm {f4(rt.macro_f1_std)}}}$ & "
            f"${f4(rt.balanced_acc_mean)} \\pm {f4(rt.balanced_acc_std)}$ & {d_s} \\\\"
        )

    # Table 2.5 All 16 configs
    all16_order = [
        ("Đối chứng", "baseline"),
        ("Nhiễu", "gauss3"), ("Nhiễu", "median3"), ("Nhiễu", "median5"),
        ("T.phản", "stretch_L_1_99"), ("T.phản", "hist_eq_L"), ("T.phản", "clahe_L_c2_t8"),
        ("Lông", "hair_bh17"), ("Lông", "hair_bh17_cond"),
        ("Viền", "vig_fill"), ("Viền", "vig_crop"),
        ("Q.sáng", "gamma_L_1.5"), ("Q.sáng", "gamma_L_auto"),
        ("S.lệch", "flat_L_s60"), ("S.lệch", "flat_L_s60_cond"),
        ("Kết hợp", "median3+clahe_L_c2_t8"), ("Kết hợp", "vig_fill+flat_L_s60_cond"),
    ]
    t16_rows_tex = []
    for grp, cfg in all16_order:
        rv, rt = tv.loc[cfg], tt.loc[cfg]
        if cfg == "baseline":
            t16_rows_tex.append(
                f"{grp} & {tt_code(cfg)} & ${f4(rv.macro_f1_mean)} \\pm {f4(rv.macro_f1_std)}$ & -- & "
                f"${f4(rt.macro_f1_mean)} \\pm {f4(rt.macro_f1_std)}$ & -- & Đối chứng \\\\"
            )
        else:
            verd = "\\textbf{Tệ hơn rõ}" if rt.macro_f1_verdict == "tệ hơn rõ" else ("\\textbf{Tốt hơn rõ}" if rt.macro_f1_verdict == "tốt hơn rõ" else "Không rõ")
            t16_rows_tex.append(
                f"{grp} & {tt_code(cfg)} & ${f4(rv.macro_f1_mean)} \\pm {f4(rv.macro_f1_std)}$ & {sf4(rv.macro_f1_delta)} ({rv.macro_f1_delta_sign}) & "
                f"${f4(rt.macro_f1_mean)} \\pm {f4(rt.macro_f1_std)}$ & {sf4(rt.macro_f1_delta)} [{sf4(rt.macro_f1_delta_ci_lo)}; {sf4(rt.macro_f1_delta_ci_hi)}] & {verd} \\\\"
            )

    # Table Per-class Baseline + consistency
    n_te = pd.read_csv(OUT / "splits/test.csv")["dx"].value_counts().to_dict()
    pc_rows_tex = []
    for c in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        p_m = S[f"test_precision_{c}"].mean()
        r_m, r_s = S[f"test_recall_{c}"].mean(), S[f"test_recall_{c}"].std(ddof=1)
        f_m, f_s = S[f"test_f1_{c}"].mean(), S[f"test_f1_{c}"].std(ddof=1)
        aw = cs_te.loc[c]
        pc_rows_tex.append(
            f"{tt_code(c)} & {n_te[c]} & ${f3(p_m)}$ & ${f3(r_m)} \\pm {f3(r_s)}$ & ${f3(f_m)} \\pm {f3(f_s)}$ & "
            f"{int(aw.always_wrong)} ({f1(aw.pct_always_wrong)}\\%) \\\\"
        )

    # Stress test L3 summary rows
    stress_rows_tex = []
    for (k, m), g in sc_l3.groupby(["kind", "method"], sort=False):
        r_to = g[g.arm == "test_only"].iloc[0]
        r_re = g[g.arm == "retrain"].iloc[0]
        stress_rows_tex.append(
            f"{tt_code(k)} ({r_to.severity}) & {tt_code(m)} & {r_to.fired*100:.0f}\\% & "
            f"{sf4(r_to.delta)} ({r_to.verdict}) & {sf4(r_re.delta)} ({r_re.verdict}) \\\\"
        )

    mel_b = mel_row[(mel_row.split == "test") & (mel_row.config == "baseline")].set_index("pred")
    mel_h = mel_row[(mel_row.split == "test") & (mel_row.config == "hair_bh17")].set_index("pred")
    mel_c = mel_row[(mel_row.split == "test") & (mel_row.config == "hair_bh17_cond")].set_index("pred")
    cp1 = cp_te.iloc[0]
    sg_gauto = sg[(sg.split == "test") & (sg.config == "gamma_L_auto")].iloc[0]
    fb_floor_te = float(fb[(fb.split == "test") & (fb.part == "all")]["flip_floor"].iloc[0]) * 100.0
    fb_gauto_fl = fb[(fb.split == "test") & (fb.part == "flagged") & (fb.config == "gamma_L_auto")].iloc[0]
    p_mel_base = float(S["test_precision_mel"].mean())
    p_mel_hcond = float(S_hcond["test_precision_mel"].mean())
    r_nv_base = float(S["test_recall_nv"].mean())
    r_nv_hcond = float(S_hcond["test_recall_nv"].mean())

    b_val_str = f"{f4(S['val_macro_f1'].mean())} \\pm {f4(S['val_macro_f1'].std(ddof=1))}"
    b_test_str = f"{f4(S['test_macro_f1'].mean())} \\pm {f4(S['test_macro_f1'].std(ddof=1))}"
    b_acc_str = f"{f4(S['test_acc'].mean())} \\pm {f4(S['test_acc'].std(ddof=1))}"
    b_bacc_str = f"{f4(S['test_balanced_acc'].mean())} \\pm {f4(S['test_balanced_acc'].std(ddof=1))}"
    s42 = S.set_index("seed").loc[42]

    main_tex = f"""\\documentclass[10pt,twocolumn,a4paper]{{article}}
\\usepackage[utf8]{{vietnam}}
\\usepackage[top=1.5cm,bottom=1.6cm,left=1.4cm,right=1.4cm,columnsep=0.55cm]{{geometry}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{booktabs,multirow,tabularx,array}}
\\usepackage{{graphicx}}
\\usepackage[font=footnotesize,labelfont=bf,skip=3pt]{{caption}}
\\usepackage{{subcaption}}
\\usepackage{{enumitem}}
\\usepackage{{xcolor}}
\\usepackage{{float}}
\\usepackage{{placeins}}
\\usepackage{{microtype}}
\\usepackage[colorlinks=true,linkcolor=blue!60!black,citecolor=blue!60!black,urlcolor=blue!60!black]{{hyperref}}
\\usepackage{{titlesec}}

\\renewcommand{{\\topfraction}}{{0.92}}
\\renewcommand{{\\bottomfraction}}{{0.85}}
\\renewcommand{{\\textfraction}}{{0.06}}
\\renewcommand{{\\floatpagefraction}}{{0.82}}
\\renewcommand{{\\dbltopfraction}}{{0.92}}
\\renewcommand{{\\dblfloatpagefraction}}{{0.82}}
\\setcounter{{topnumber}}{{3}}
\\setcounter{{dbltopnumber}}{{3}}
\\setcounter{{totalnumber}}{{5}}

\\setlist[itemize]{{nosep,leftmargin=1.2em,topsep=2pt}}
\\setlist[enumerate]{{nosep,leftmargin=1.4em,topsep=2pt}}
\\setlength{{\\textfloatsep}}{{6pt plus 2pt minus 2pt}}
\\setlength{{\\floatsep}}{{5pt plus 2pt minus 2pt}}
\\setlength{{\\intextsep}}{{5pt plus 2pt minus 2pt}}
\\titlespacing*{{\\section}}{{0pt}}{{7pt plus 2pt minus 1pt}}{{3pt}}
\\titlespacing*{{\\subsection}}{{0pt}}{{5pt plus 1pt minus 1pt}}{{2pt}}

\\title{{\\vspace{{-0.8cm}}\\textbf{{\\Large Đánh Giá Tác Động Của Các Phương Pháp Xử Lý Ảnh Cổ Điển Lên Mô Hình Phân Loại Tổn Thương Da HAM10000 Sử Dụng Mạng Nơ-ron Tích Chập}}}}
\\author{{
\\normalsize\\textbf{{Nguyễn Trung Nam (B23DCVT299), Phan Nguyễn Việt Dũng (B23DCCN210),}}\\\\
\\normalsize\\textbf{{Lê Duy Anh (B23DCCN026), Đinh Khánh Sơn (B23DCDT223), Đào Phạm Bích Ngọc (B23DCVT308)}}\\\\
\\small Học phần Xử lý ảnh -- Khoa Trí tuệ Nhân tạo, Học viện Công nghệ Bưu chính Viễn thông (PTIT)\\\\
\\small Mã nguồn \\& thực nghiệm tái lập: \\url{{https://github.com/VietDunghere/Image-Processing}}
}}
\\date{{\\vspace{{-0.3cm}}\\small Tháng 11 năm 2026}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
\\noindent Trong thị giác máy tính hiện đại, mạng tích chập sâu (CNN) thường được huấn luyện trực tiếp trên ảnh thô, đặt ra câu hỏi liệu các phép nâng cao chất lượng ảnh cổ điển còn mang lại lợi ích thực chất hay không. Nghiên cứu này thực hiện đánh giá thực nghiệm có kiểm soát chặt chẽ trên bộ dữ liệu soi da thực tế \\textbf{{HAM10000}} ($10\\,015$ ảnh dermoscopy $600{{\\times}}450$ thuộc $7\\,470$ tổn thương, 7 lớp bệnh mất cân bằng $58{{,}}3$ lần). Trước hết, chúng tôi xây dựng bộ 7 chỉ số định lượng bám sát Chương 3, 4, 5, 7 của học phần để kiểm định các vấn đề chất lượng ảnh và phát hiện 6 hiện tượng có bằng chứng thống kê (lông che phủ $10{{,}}1\\%$, nhiễu Gauss nhẹ $7{{,}}3\\%$, tương phản thấp $4{{,}}3\\%$, viền tối ống soi $3{{,}}5\\%$, quá sáng bạc màu $3{{,}}0\\%$, chiếu sáng lệch $2{{,}}5\\%$), đồng thời bác bỏ nhiễu muối--tiêu và quá tối. Trên phân chia chống rò rỉ theo \\texttt{{lesion\\_id}} và cấu hình đóng băng có lấy mẫu cân bằng WRS qua 3 seed độc lập, chúng tôi so sánh 3 kiến trúc (\\texttt{{ResNet-18}}, \\texttt{{EfficientNet-B0}}, \\texttt{{MobileNetV3-Small}}) và đánh giá 16 cấu hình tiền xử lý trên \\texttt{{ResNet-18}} (baseline đạt test macro-F1 ${b_test_str}$). Kết quả cho thấy \\textbf{{0/16 cấu hình cải thiện rõ}} macro-F1 trên validation; đặc biệt cân bằng lược đồ toàn cục (\\texttt{{hist\\_eq\\_L}}, $\\Delta={sf4(tt.loc['hist_eq_L','macro_f1_delta'])}$) và giãn tuyến tính (\\texttt{{stretch\\_L\\_1\\_99}}, $\\Delta={sf4(tt.loc['stretch_L_1_99','macro_f1_delta'])}$) làm giảm mạnh hiệu năng do lệch màu $\\Delta E_{{76}}$ và khuếch đại nhiễu, còn xóa lông đại trà (\\texttt{{hair\\_bh17}}) inpaint nhầm mạng sắc tố của u hắc tố (\\texttt{{mel}}). Phân tích sâu 4 tầng (ảnh, mức nền đổi nhãn, stress test suy giảm tổng hợp và Grad-CAM) lý giải trọn vẹn cơ chế này và dẫn tới đề xuất \\textbf{{pipeline hai nhánh}}: nhánh Kiểm tra chất lượng (QC) gắn cờ cảnh báo song song với nhánh phân loại CNN trên ảnh gốc, triển khai trực quan trên ứng dụng Streamlit.
\\end{{abstract}}

\\section{{Giới thiệu và Câu hỏi nghiên cứu}}
\\textbf{{Bài toán và động cơ.}} Phân loại đa lớp ảnh tổn thương da (7 lớp: \\texttt{{akiec, bcc, bkl, df, mel, nv, vasc}}) trên tập dữ liệu thực tế HAM10000~\\cite{{tschandl2018ham10000}} đối mặt với ba trở ngại lớn: (i) nhiễu và sai lệch thu nhận ảnh từ máy soi da (viền đen cơ học, phản xạ lóa của gel/LED, sợi lông che khuất, độ tương phản thấp); (ii) mất cân bằng lớp gay gắt ($6\\,705$ ảnh \\texttt{{nv}} so với $115$ ảnh \\texttt{{df}}); và (iii) một \\texttt{{lesion\\_id}} có nhiều ảnh chụp làm phát sinh nguy cơ rò rỉ dữ liệu nếu chia ngẫu nhiên theo ảnh. Mục tiêu của đồ án \\emph{{không}} phải tối ưu hóa điểm số SOTA bằng mọi giá, mà tuân thủ phương pháp luận \\textbf{{lập luận $\\rightarrow$ thực nghiệm kiểm soát biến $\\rightarrow$ đánh giá trung thực}}, trong đó kết quả âm (``không cải thiện'' hoặc ``làm tệ đi'') vẫn là đóng góp khoa học hợp lệ.

\\textbf{{Câu hỏi nghiên cứu (RQ) và ánh xạ học phần.}} Nghiên cứu giải quyết 4 câu hỏi: \\textbf{{RQ1}} -- Tập ảnh có những vấn đề chất lượng nào có bằng chứng thống kê và liên hệ thế nào với lỗi CNN? \\textbf{{RQ2}} -- Từng phương pháp xử lý cổ điển (áp đại trà hoặc có điều kiện) có tạo chênh lệch macro-F1 vượt ngưỡng dao động giữa các seed không? \\textbf{{RQ3}} -- Tác động thực sự diễn ra ở tầng nào (chỉ số ảnh, đổi nhãn dự đoán, stress test suy giảm nặng hay vùng chú ý Grad-CAM)? \\textbf{{RQ4}} -- Ghép nối nhiều bước xử lý và lựa chọn kiến trúc CNN dẫn tới pipeline cuối ra sao? Toàn bộ công cụ bám sát học phần: \\textbf{{Chương 3}} (lược đồ xám, giãn/cân bằng lược đồ, CLAHE~\\cite{{zuiderveld1994clahe}}, gamma, lọc Gauss/trung vị, khôi phục inpainting Telea~\\cite{{telea2004inpainting}}), \\textbf{{Chương 4}} (không gian HSV và mở rộng sang CIE $L^*a^*b^*$), \\textbf{{Chương 5}} (Sobel, phân ngưỡng Otsu~\\cite{{otsu1979threshold}}), \\textbf{{Chương 6}} (kiểm tra nén JPEG), \\textbf{{Chương 7}} (đóng, mở, giãn, black-hat) và \\textbf{{Chương 8}} (CNN~\\cite{{he2016resnet,tan2019efficientnet,howard2019mobilenetv3}} và Grad-CAM~\\cite{{selvaraju2017gradcam}}).

\\section{{Dữ liệu, Phân tích chất lượng ảnh (EDA) và Chuẩn bị}}
\\subsection{{Mô tả dữ liệu HAM10000 và Cảnh báo lệch miền}}
Bộ dữ liệu gồm $10\\,015$ ảnh RGB đồng nhất kích thước $600{{\\times}}450$ từ $7\\,470$ \\texttt{{lesion\\_id}}~\\cite{{tschandl2018ham10000}}. Nhãn được xác nhận qua 4 phương thức (\\texttt{{dx\\_type}}): mô bệnh học \\texttt{{histo}} ($53{{,}}3\\%$), theo dõi số hóa \\texttt{{follow\\_up}} ($37{{,}}0\\%$), đồng thuận \\texttt{{consensus}} ($9{{,}}0\\%$) và kính đồng tiêu \\texttt{{confocal}} ($0{{,}}7\\%$) (Hình~\\ref{{fig:data_wrs}}a). Đáng chú ý, toàn bộ $3\\,704$ ảnh \\texttt{{follow\\_up}} đều thuộc lớp \\texttt{{nv}}, tạo nguy cơ mô hình học dấu vết thiết bị/nguồn chụp (\\emph{{shortcut learning}}). Có $1\\,956$ lesion ($26{{,}}2\\%$) chứa $\\ge 2$ ảnh (chiếm $44{{,}}9\\%$ tổng số ảnh), bao gồm 2 nhóm trùng byte và 19 cặp gần trùng (aHash $\\le 2$ bit) đều nằm trong cùng \\texttt{{lesion\\_id}}. Đặc biệt, đây là ảnh \\textbf{{soi da áp sát (dermoscopy)}} chứ không phải ảnh chụp bằng điện thoại thông thường; do đó mọi thử nghiệm ngoài tập bằng ảnh điện thoại ở phần demo chỉ mang ý nghĩa minh họa định tính giai đoạn xử lý, không dùng để chẩn đoán y khoa.

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=0.98\\linewidth]{{figures/class_distribution.pdf}}
    \\caption{{Phân bố 7 lớp tổn thương theo thứ tự chuẩn \\texttt{{sorted(dx)}} (trái, thang log; tỉ lệ mất cân bằng $58{{,}}3\\times$) và phương thức xác nhận nhãn \\texttt{{dx\\_type}} (phải) trên $n = 10\\,015$ ảnh HAM10000.}}
    \\label{{fig:data_wrs}}
\\end{{figure}}

\\subsection{{Phân tích khám phá chất lượng ảnh (EDA -- Trả lời RQ1)}}
Thay vì nhận định cảm tính, nhóm xây dựng bộ 7 chỉ số định lượng trên toàn bộ $10\\,015$ ảnh (\\texttt{{src/eda/quality\\_metrics.py}}). Ngưỡng gắn cờ mặc định tuân theo quy tắc ngoại lai Tukey ($Q_3 + 1{{,}}5\\cdot\\text{{IQR}}$), hoặc ngưỡng tuyệt đối có kiểm tra độ nhạy khi phân phối dồn về 0. Để tránh vi phạm tính độc lập, kiểm định Kruskal--Wallis giữa các lớp \\texttt{{dx}} và nguồn \\texttt{{dx\\_type}} được tính trên đúng 1 ảnh/lesion ($n = 7\\,470$, hiệu chỉnh Holm) kèm độ lớn hiệu ứng $\\varepsilon^2$. Độ nhạy của bộ đo nhiễu được kiểm chứng trước trên 30 ảnh cộng nhiễu tổng hợp đã biết: với nhiễu Gauss $\\sigma \\in \\{{0,2,5,10\\}}$, bộ đo trả về trung vị $1{{,}}33 / 2{{,}}39 / 5{{,}}09 / 9{{,}}91$; với nhiễu muối--tiêu $p \\in \\{{0{{,}}1\\%, 1\\%, 5\\%\\}}$, bộ đo trả về $0{{,}}099\\% / 0{{,}}98\\% / 4{{,}}97\\%$. Do đó, việc chỉ có 1 ảnh thật vượt $0{{,}}1\\%$ pixel xung chứng minh chắc chắn tập ảnh không bị nhiễu muối--tiêu. Chỉ số lông (black-hat $17{{\\times}}17$) cũng khớp thực tế giải phẫu: vùng da đầu (\\texttt{{scalp}}) có $61{{,}}7\\%$ ảnh nhiều lông so với $0\\%$ ở lòng bàn tay/chân (\\texttt{{acral}}), và tương quan hạng trước--sau resize $224$ đạt $\\rho = 0{{,}}86$. Ba chỉ số (trắng bạc, nhiễu $\\sigma$, viền tối) có $\\varepsilon^2$ theo \\texttt{{dx\\_type}} lớn gấp 2--4 lần theo \\texttt{{dx}} ($0{{,}}197$ vs $0{{,}}088$; $0{{,}}187$ vs $0{{,}}044$), củng cố cảnh báo khác biệt nguồn thu nhận (Hình~\\ref{{fig:eda_sum}}). Bảng~\\ref{{tab:eda_concl}} tổng hợp kết luận EDA cùng mức chênh lệch lỗi CNN sau chuẩn hóa thành phần lớp (chi tiết tại mục~\\ref{{sec:eval}}).

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=0.98\\linewidth]{{figures/eda_summary.pdf}}
    \\caption{{Tổng hợp EDA trên $10\\,015$ ảnh: (a) quy mô 6 vấn đề tồn tại và 2 vấn đề bác bỏ; (b) so sánh effect size $\\varepsilon^2$ theo lớp bệnh \\texttt{{dx}} và nguồn nhãn \\texttt{{dx\\_type}}; (c) ma trận tương quan Spearman $|\\rho| \\le 0{{,}}40$ khẳng định 7 chỉ số không đo trùng hiện tượng.}}
    \\label{{fig:eda_sum}}
\\end{{figure}}

\\begin{{table*}}[!t]
\\centering
\\caption{{Bảng kết luận EDA trên toàn bộ $10\\,015$ ảnh và đối chiếu sai số ResNet-18 baseline ngoài mẫu (chuẩn hóa theo thành phần lớp, tập test $n=2\\,004$, CI95 bootstrap cụm lesion). Mọi thí nghiệm ở mục 2.5 truy ngược trực tiếp về bảng này.}}
\\label{{tab:eda_concl}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrl}}
\\toprule
\\textbf{{Vấn đề chất lượng}} & \\textbf{{Định nghĩa chỉ số \\& Ngưỡng cờ}} & \\textbf{{Số ảnh (\\%)}} & \\textbf{{Lỗi nhóm}} & \\textbf{{Excess lỗi so kỳ vọng lớp [CI95]}} \\\\
\\midrule
{chr(10).join(eda_rows_tex)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}

\\subsection{{Làm sạch metadata, Chia tập theo lesion và WRS}}
Metadata không có dòng nào thiếu nhãn hay thiếu tệp ảnh; 57 ảnh thiếu \\texttt{{age}} được điền bằng trung vị của riêng tập \\texttt{{train\\_fit}} ($50{{,}}0$ tuổi) kèm cột cờ \\texttt{{age\\_missing}} mà không xóa ảnh hợp lệ. Dữ liệu được chia hai tầng bằng \\texttt{{StratifiedGroupKFold}} (nhóm \\texttt{{lesion\\_id}}, phân tầng nhãn, \\texttt{{seed=42}}): \\texttt{{train\\_fit}} $6\\,408$ ảnh ($4\\,779$ lesion, $63{{,}}98\\%$), \\texttt{{val}} $1\\,603$ ảnh ($1\\,196$ lesion, $16{{,}}01\\%$) và \\texttt{{test}} $2\\,004$ ảnh ($1\\,495$ lesion, $20{{,}}01\\%$), kiểm chứng giao \\texttt{{lesion\\_id}} giữa mọi cặp tập bằng $0$. Để xử lý mất cân bằng $58{{,}}3\\times$ trên \\texttt{{train\\_fit}}, nhóm dùng \\texttt{{WeightedRandomSampler}} (WRS) với trọng số $w_i = 1/n_{{y_i}}$ (kỳ vọng mỗi lớp chiếm $14{{,}}29\\%$/epoch) kết hợp \\texttt{{CrossEntropyLoss}} \\emph{{không}} trọng số lớp nhằm tránh bù mất cân bằng hai lần.

\\section{{Baseline và Khảo sát Kiến trúc CNN}}
\\textbf{{Cấu hình đóng băng.}} Mọi thí nghiệm chuẩn hóa ảnh về $224{{\\times}}224$ (PIL bilinear) + ImageNet mean/std, tối ưu bằng AdamW~\\cite{{loshchilov2019adamw}} ($\\text{{lr}}=10^{{-4}}, \\text{{wd}}=10^{{-4}}$), batch $32$, $20$ epoch, AMP float16, lật ngang/dọc ngẫu nhiên trên train, chọn checkpoint đạt \\texttt{{val}} macro-F1 cao nhất và đánh giá \\texttt{{test}} đúng 1 lần qua 3 seed ($42, 43, 44$). Bộ phân loại tầm thường luôn đoán \\texttt{{nv}} đạt accuracy $66{{,}}92\\%$ trên test nhưng macro-F1 chỉ $0{{,}}1145$; do đó \\textbf{{macro-F1}} được dùng làm thước đo chính.

\\begin{{table}}[t]
\\centering
\\caption{{So sánh 3 kiến trúc CNN dưới cùng cấu hình đóng băng (mean $\\pm$ std qua 3 seed: 42, 43, 44; $\\Delta$ ghép cặp theo seed).}}
\\label{{tab:backbones}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrccc}}
\\toprule
\\textbf{{Kiến trúc}} & \\textbf{{Tham số}} & \\textbf{{Val F1}} & \\textbf{{Test F1}} & \\textbf{{Test bAcc}} & \\textbf{{$\\Delta$ Test F1}} \\\\
\\midrule
{chr(10).join(bb_rows_tex)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=0.98\\linewidth]{{figures/backbone_comparison.pdf}}
    \\caption{{So sánh hiệu năng 3 kiến trúc CNN đại diện (cột = trung bình, vạch = $\\pm 1$ std, chấm = từng seed 42/43/44).}}
    \\label{{fig:backbones}}
\\end{{figure}}

Bảng~\\ref{{tab:backbones}} và Hình~\\ref{{fig:backbones}} trình bày kết quả khảo sát 3 họ kiến trúc (với \\texttt{{EfficientNet-B0}} và \\texttt{{MobileNetV3-Small}} dùng tích lũy gradient \\texttt{{micro\\_batch}}$=16$ để tiết kiệm VRAM khối MBConv). Theo nguyên tắc chọn trên \\texttt{{val}} (không dùng \\texttt{{test}}), dù \\texttt{{EfficientNet-B0}} có trung bình \\texttt{{val}} F1 nhỉnh nhẹ ($0{{,}}7355 \\pm 0{{,}}0146$), \\texttt{{ResNet-18}} có độ lệch chuẩn giữa các seed trên \\texttt{{val}} thấp hơn gấp $3{{,}}3$ lần ($0{{,}}0044$), cấu trúc khối dư tường minh và lưới đặc trưng \\texttt{{layer4}} ($7{{\\times}}7$) thuận lợi cho Grad-CAM nên được khóa làm mô hình cơ sở xuyên suốt 16 thí nghiệm mục~\\ref{{sec:ablation}}. Trên \\texttt{{test}}, \\texttt{{ResNet-18}} đạt macro-F1 ${b_test_str}$ (bAcc ${b_bacc_str}$, Acc ${b_acc_str}$), vượt rõ \\texttt{{MobileNetV3-Small}} và không khác biệt thống kê với \\texttt{{EfficientNet-B0}} (CI95 $[{sf4(bb_test.loc['efficientnet_b0','macro_f1_ci_lo'])}; {sf4(bb_test.loc['efficientnet_b0','macro_f1_ci_hi'])}]$ phủ $0$).

\\section{{Thực nghiệm Tiền xử lý Ảnh có Kiểm soát}}
\\label{{sec:ablation}}
\\textbf{{Quy tắc kết luận đặt trước (2.5.0).}} Mọi phép xử lý áp ở độ phân giải gốc $600{{\\times}}450$ trước resize $224$, đồng nhất cho cả 3 tập. Một cấu hình chỉ được kết luận là ``tốt hơn rõ'' hoặc ``tệ hơn rõ'' khi thỏa cả 3 điều kiện: (1) $|\\Delta\\text{{ trung bình}}| > \\text{{std}}_{{\\text{{baseline}}}}$ trên cùng split; (2) khoảng tin cậy $95\\%$ bootstrap theo cụm \\texttt{{lesion\\_id}} ($1\\,000$ lần) không chứa $0$; và (3) cả $3/3$ seed cùng dấu. \\emph{{Lưu ý thống kê:}} phép bootstrap lấy mẫu lại cụm \\texttt{{lesion\\_id}} trên 3 checkpoint cố định nên đo bất định mẫu đánh giá (chưa gộp phương sai huấn luyện lại giữa các seed khi 16 cấu hình cùng so với một bộ 3 seed baseline có test F1 cao ${b_test_str}$); do đó mọi quyết định chọn pipeline ở mục~\\ref{{sec:eval}} chỉ căn cứ trên \\texttt{{val}} kết hợp kiểm định nhóm con DiD.

\\subsection{{Nhiễu Gauss và Tương phản thấp (Chương 3 \\& 4)}}
Với nhiễu Gauss, nhóm đối chiếu lọc Gauss $3{{\\times}}3$ (nhân $\\frac{{1}}{{16}}[1\\,2\\,1;2\\,4\\,2;1\\,2\\,1]$), trung vị $3{{\\times}}3$ và trung vị $5{{\\times}}5$. Đo trực tiếp trên $463$ ảnh cờ nhiễu thuộc \\texttt{{train\\_fit}} cho thấy resize $600{{\\rightarrow}}224$ vốn đã hạ $\\sigma$ Immerkaer~\\cite{{immerkaer1996fast}} từ $1{{,}}58$ xuống $0{{,}}86$; lọc \\texttt{{median5}} hạ tiếp xuống $0{{,}}385$ nhưng làm mất $75\\%$ phương sai Laplacian vùng tổn thương. Với tương phản thấp, nhóm biến đổi kênh độ sáng $L^*$ của không gian CIE $L^*a^*b^*$ để hạn chế bóp méo sắc độ $a^*,b^*$: giãn phân vị $[P_1, P_{{99}}]$ (\\texttt{{stretch\\_L\\_1\\_99}}), cân bằng lược đồ toàn cục (\\texttt{{hist\\_eq\\_L}}) và CLAHE cục bộ \\texttt{{clahe\\_L\\_c2\\_t8}} (clip $2{{,}}0$, lưới $8{{\\times}}8$). Kiểm tra trực quan từng khung trên mẫu ngẫu nhiên 24 ảnh cờ tương phản thấp (\\texttt{{seed=42}}) ghi nhận mask Otsu thô chỉ bao đúng $8/24$ ảnh, đúng một phần $5/24$ và bao sai $11/24$ ảnh (thường bắt bóng tối rìa ảnh). Trên CNN (Bảng~\\ref{{tab:all16}}), \\texttt{{hist\\_eq\\_L}} và \\texttt{{stretch\\_L\\_1\\_99}} đều \\textbf{{tệ hơn rõ}} trên cả \\texttt{{val}} lẫn \\texttt{{test}} ($0+/3-$, CI95 tách rời 0) vì làm biến thiên mạnh độ chói ($\Delta E_{{76}}$ từ $14$ đến $22$, chủ yếu do $\Delta L^*$ xóa thông tin độ tối tuyệt đối kèm trượt sắc độ $\Delta ab \\approx 1{{,}}5$ khi cắt biên sRGB) và nhân nhiễu lên $3{{,}}5\\text{{--}}5{{,}}5$ lần trên $95{{,}}7\\%$ ảnh thường; ngược lại \\texttt{{median3}} và \\texttt{{clahe\\_L\\_c2\\_t8}} giữ trung tính.

\\subsection{{Lông, Viền tối, Quá sáng và Sáng lệch: Đại trà vs Có điều kiện}}
Vì mỗi lỗi chỉ chiếm $2{{,}}5\\%\\text{{--}}10{{,}}1\\%$ dữ liệu, nhóm thiết kế đối chứng giữa \\textbf{{bản đại trà}} và \\textbf{{bản có điều kiện}} (kích hoạt theo ngưỡng cờ không giám sát của EDA -- kiểm tra lại trên riêng \\texttt{{train\\_fit}} cho ngưỡng \\texttt{{illum\\_range}} khớp tuyệt đối $38{{,}}00$ và \\texttt{{hair\\_frac}} lệch chỉ $0{{,}}11\\%$ FOV; hằng số mục tiêu \\texttt{{L\\_TARGET}}$=66{{,}}61$ chỉ tính trên \\texttt{{train\\_fit}}):
\\begin{{itemize}}
    \\item \\textbf{{Lông (Ch.7 + Ch.3):}} Black-hat elip $17{{\\times}}17 \\ge 20$ + giãn $3{{\\times}}3$ + Telea inpainting bán kính 5 px~\\cite{{lee1997dullrazor,telea2004inpainting}} (\\texttt{{hair\\_bh17}} và \\texttt{{hair\\_bh17\\_cond}}).
    \\item \\textbf{{Viền tối (Ch.3 + Ch.7):}} Tô vùng đen nối mép bằng trung vị dải da 25 px + làm mềm biên Gauss $21{{\\times}}21$ (\\texttt{{vig\\_fill}}, giữ nguyên tỉ lệ tổn thương) đối chiếu cắt khung đồng tâm 4:3 (\\texttt{{vig\\_crop}}).
    \\item \\textbf{{Quá sáng \\& Sáng lệch (Ch.3 + Ch.4):}} Biến đổi lũy thừa $s=r^\\gamma$ trên $L^*$ (\\texttt{{gamma\\_L\\_1.5}} vs \\texttt{{gamma\\_L\\_auto}} tự chọn $\\gamma \\in [1;3]$) và chia trường sáng nền Gauss chuẩn hóa $\\sigma=60$ px ngoài vùng tổn thương (\\texttt{{flat\\_L\\_s60}} vs \\texttt{{flat\\_L\\_s60\\_cond}}).
\\end{{itemize}}

\\begin{{table*}}[!t]
\\centering
\\caption{{Kết quả tổng hợp thực nghiệm Ablation 16 cấu hình tiền xử lý trên ResNet-18 (3 seed: 42, 43, 44; cùng một phần cứng NVIDIA L40S). Không cấu hình nào đạt ``tốt hơn rõ'' trên validation; 2 phép toàn cục giảm mạnh hiệu năng.}}
\\label{{tab:all16}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llccccl}}
\\toprule
\\textbf{{Nhóm}} & \\textbf{{Cấu hình}} & \\textbf{{Val Macro-F1}} & \\textbf{{$\\Delta$ Val (dấu)}} & \\textbf{{Test Macro-F1}} & \\textbf{{$\\Delta$ Test [CI95 bootstrap]}} & \\textbf{{Kết luận (Test)}} \\\\
\\midrule
{chr(10).join(t16_rows_tex)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table*}}

\\begin{{figure*}}[!t]
    \\centering
    \\includegraphics[height=0.56\\textheight,keepaspectratio]{{figures/artifacts_before_after.png}}
    \\caption{{Đối chiếu trực quan ảnh gốc, mask phát hiện và ảnh sau xử lý của 4 phép biến đổi có điều kiện (xóa lông \\texttt{{hair\\_bh17\\_cond}}, lấp viền tối \\texttt{{vig\\_fill}}, chỉnh sáng tự động \\texttt{{gamma\\_L\\_auto}} và cân bằng trường sáng \\texttt{{flat\\_L\\_s60\\_cond}}) trên các ca bị cờ EDA.}}
    \\label{{fig:artifacts_ba}}
\\end{{figure*}}

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=0.98\\linewidth]{{figures/all16_macro_f1.pdf}}
    \\caption{{Macro-F1 trên Validation và Test của baseline cùng 16 cấu hình tiền xử lý (dải xám ngang = $\\pm 1$ std của baseline; chấm đen = từng seed 42, 43, 44).}}
    \\label{{fig:all16}}
\\end{{figure}}

Bảng~\\ref{{tab:all16}}, Hình~\\ref{{fig:artifacts_ba}} và Hình~\\ref{{fig:all16}} cho thấy cả 8 cấu hình thuộc nhóm này và 2 cấu hình kết hợp đều không vượt chuẩn cải thiện rõ trên \\texttt{{val}}. Phân tích sai số kép (DiD) trên nhóm ảnh bị cờ có khoảng tin cậy 95\\% phủ 0 ở 7/8 cấu hình, ngoại trừ \\texttt{{gamma\\_L\\_auto}} làm giảm mạnh độ chính xác ngay trên 69 ảnh quá sáng tập test (${f3(sg_gauto.acc_group_baseline)} \\rightarrow {f3(sg_gauto.acc_group_baseline + sg_gauto.d_acc_group)}$, $\\text{{DiD}} = {sf3(sg_gauto.did)}$ $[{sf3(sg_gauto.did_lo)}; {sf3(sg_gauto.did_hi)}]$) do lệch phân phối có điều kiện trên $3\\%$ dữ liệu và kéo vùng lóa mất sắc độ ($a^*,b^*\\approx 0$) thành mảng xám nhân tạo. Các bản có điều kiện giữ nguyên $100\\%$ ảnh đối chứng (các nhãn ``tệ hơn rõ'' ở \\texttt{{vig\\_crop}} hay \\texttt{{vig\\_fill+flat\\_L\\_s60\\_cond}} dù nhóm cờ tăng $+0{{,}}004$ đến $+0{{,}}036$ xuất phát từ nhiễu huấn luyện lại trên các lớp nhỏ không bị cờ), trong khi \\texttt{{hair\\_bh17}} đại trà sửa nhầm $92\\%$ ảnh không lông và giảm điểm cả 3/3 seed.

\\subsection{{Phân tích sâu 4 tầng tác động (Mục 2.5c -- Trả lời RQ3)}}
Để giải thích vì sao cải thiện chỉ số ảnh không chuyển hóa thành tăng macro-F1 trên tập ảnh gốc, nhóm định lượng tiếp trên 4 tầng (Hình~\\ref{{fig:impact4}}):
\\begin{{enumerate}}
    \\item \\textbf{{Tầng A (Mức ảnh trên \\texttt{{train\\_fit}}):}} Các phép xử lý xóa $91{{,}}8\\%\\text{{--}}100\\%$ cờ viền, lông, quá sáng, sáng lệch. Tuy nhiên, kiểm định trên toàn bộ \\texttt{{train\\_fit}} phát hiện \\texttt{{hair\\_bh17}} inpaint trung vị $\\mathbf{{{hb.loc['mel','median']*100:.1f}\\%}}$ diện tích vùng tổn thương của \\texttt{{mel}}, cao gấp hơn 3 lần so với $\\mathbf{{{hb.loc['nv','median']*100:.1f}\\%}}$ ở \\texttt{{nv}} (Mann--Whitney 1 ảnh/lesion $p = {ht.mw_p_mel_nv:.1e}$, Cliff's $\\delta = {f3(ht.cliffs_delta_mel_vs_nv)}$). Ngoài ra, trên ca lông bạc/trắng dày (\\texttt{{ISIC\\_0032214}}), black-hat bắt nhầm nền da tối giữa các sợi lông sáng làm nhòe mảng tổn thương.
    \\item \\textbf{{Tầng B (Đổi nhãn dự đoán so với mức nền):}} Giữa 2 seed khác nhau của chính baseline vốn đã có ${f1(fb_floor_te)}\\%$ ảnh test đổi nhãn dự đoán ($15{{,}}9\\%$ trên \\texttt{{val}}). Tỉ lệ đổi nhãn toàn tập của \\texttt{{vig\\_fill}} ($-0{{,}}67\\%$) và \\texttt{{flat\\_L\\_s60\\_cond}} ($-0{{,}}12\\%$, $p_{{\\text{{Holm}}}}=1{{,}}00$) nằm đúng mức nền tự nhiên này; riêng \\texttt{{gamma\\_L\\_auto}} sát mức nền trên ảnh không cờ ($+0{{,}}71\\%$) nhưng vọt $+{f2(fb_gauto_fl.excess*100)}\\%$ lật nhãn trên nhóm quá sáng ($p_{{\\text{{Holm}}}}={f3(fb_gauto_fl.p_holm)}$), còn \\texttt{{hair\\_bh17}}/\\texttt{{hair\\_bh17\\_cond}} vượt nền rõ ($+2{{,}}35\\%/+2{{,}}45\\%$).
    \\item \\textbf{{Tầng C (Stress Test suy giảm tổng hợp):}} Khi áp 6 loại suy giảm $\\times$ 3 mức độ lên tập đánh giá (Bảng~\\ref{{tab:stress}}), tiền xử lý khôi phục mạnh ở mức suy giảm nặng ngay trên nhánh \\texttt{{test\\_only}}: \\texttt{{hair\\_bh17\\_cond}} tăng $+0{{,}}4623$ macro-F1 (lông dày), \\texttt{{clahe\\_L\\_c2\\_t8}} tăng $+0{{,}}1943$ (nén tương phản $0{{,}}3\\times$) và \\texttt{{vig\\_fill}} tăng $+0{{,}}1020$ (viền lớn), đều đạt ``tốt hơn rõ'' ($3+/0-$). Đáng chú ý, ở \\texttt{{low\\_contrast}}, \\texttt{{clahe\\_L\\_c2\\_t8}} giúp nhánh \\texttt{{test\\_only}} tăng $+0{{,}}1943$ nhưng nhánh \\texttt{{retrain}} chỉ đạt $-0{{,}}0073$ vì CLAHE đưa ảnh bị nén $0{{,}}3\\times$ trở về sát phân phối gốc mà \\texttt{{baseline}} quen thuộc, trong khi mô hình \\texttt{{retrain}} lại kỳ vọng độ tương phản cao hơn nữa. Với \\texttt{{overexpose}} mức 3, \\texttt{{gamma\\_L\\_auto}} không khôi phục được thông tin bão hòa 255 ($-0{{,}}0005$).
    \\item \\textbf{{Tầng D (Grad-CAM \\texttt{{layer4}}):}} Xóa viền tối và xóa lông giúp giảm tỉ lệ năng lượng Grad-CAM rơi vào góc viền/sợi lông và giữ tương đồng bản đồ nhiệt Pearson $r$ ổn định quanh mức nền giữa các seed, song bản thân baseline vốn đã tập trung phần lớn năng lượng vào vùng tổn thương trung tâm.
\\end{{enumerate}}

\\begin{{table*}}[!t]
\\centering
\\caption{{Stress Test tầng C (Mức suy giảm nặng nhất -- Mức 3, tập test): $\\Delta$ Macro-F1 so với baseline không xử lý trên cùng ảnh suy giảm.}}
\\label{{tab:stress}}
\\footnotesize
\\begin{{tabular}}{{llccc}}
\\toprule
\\textbf{{Suy giảm (Mức 3)}} & \\textbf{{Phép xử lý}} & \\textbf{{Tỉ lệ kích hoạt}} & \\textbf{{Nhánh test\\_only}} & \\textbf{{Nhánh retrain}} \\\\
\\midrule
{chr(10).join(stress_rows_tex)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}

\\begin{{figure}}[t]
    \\centering
    \\includegraphics[width=0.98\\linewidth]{{figures/impact_c2_stress.pdf}}
    \\caption{{Bản đồ nhiệt $\\Delta$ Macro-F1 trong Stress Test 18 điều kiện suy giảm tổng hợp (dấu $*$ đánh dấu khác biệt rõ theo quy tắc 2.5.0).}}
    \\label{{fig:impact4}}
\\end{{figure}}

\\section{{Pipeline Đề xuất, Đánh giá Lỗi \\& Demo}}
\\label{{sec:eval}}
\\subsection{{Kiến trúc Pipeline hai nhánh và Mô hình cuối (2.6--2.7)}}
Áp dụng quy tắc chọn 3 điều kiện trên \\texttt{{val}}, không bước biến đổi pixel nào được đưa vào nhánh phân loại tự động (trả lời \\textbf{{RQ4}}: ghép nối \\texttt{{median3+clahe\\_L\\_c2\\_t8}} hay \\texttt{{vig\\_fill+flat\\_L\\_s60\\_cond}} cũng không vượt thành phần đơn lẻ). Pipeline đề xuất gồm \\textbf{{hai nhánh song song}}:
\\begin{{itemize}}
    \\item \\textbf{{Nhánh (A) Kiểm tra chất lượng (QC):}} chạy bộ đo 6 chỉ số Chương 3--7 trên ảnh $600{{\\times}}450$ để gắn cờ cảnh báo cho người dùng (đặc biệt cảnh báo khi gặp \\emph{{sáng không đều}} và \\emph{{quá sáng}}), đồng thời cho phép bật tường minh các bước có điều kiện (\\texttt{{vig\\_fill}}, \\texttt{{flat\\_L\\_s60\\_cond}}) để quan sát đối chiếu.
    \\item \\textbf{{Nhánh (B) Phân loại chính:}} đưa ảnh RGB gốc qua \\texttt{{Resize 224}} $\\rightarrow$ \\texttt{{Normalize ImageNet}} $\\rightarrow$ \\texttt{{ResNet-18}} (giữ nguyên cấu hình đóng băng mục 2.4). Bản demo nạp checkpoint \\texttt{{baseline\\_seed42.pt}} (hạt giống mặc định đặt trước, hội tụ tại epoch {int(s42['best_epoch'])}, test macro-F1 ${f4(s42['test_macro_f1'])}$).
\\end{{itemize}}

\\subsection{{Phân tích lỗi chuyên sâu và Demo Streamlit (2.8)}}
\\begin{{table}}[t]
\\centering
\\caption{{Hiệu năng theo từng lớp trên tập test ($n=2\\,004$, 3 seed) và tỉ lệ ảnh khó bị đoán sai nhất quán ở cả 3 seed.}}
\\label{{tab:perclass}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrccr}}
\\toprule
\\textbf{{Lớp}} & \\textbf{{$n$ test}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1-score}} & \\textbf{{Sai cả 3 seed}} \\\\
\\midrule
{chr(10).join(pc_rows_tex)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}

\\begin{{figure}}[t]
    \\centering
    \\begin{{subfigure}}[b]{{0.49\\linewidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figures/confusion_test.pdf}}
        \\caption{{Confusion matrix test}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[b]{{0.49\\linewidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figures/eda_cnn_link.pdf}}
        \\caption{{Excess lỗi theo cờ EDA}}
    \\end{{subfigure}}
    \\caption{{(a) Ma trận nhầm lẫn chuẩn hóa theo hàng của mô hình cuối (TB 3 seed); (b) Chênh lệch lỗi của nhóm cờ EDA sau khi chuẩn hóa theo thành phần lớp.}}
    \\label{{fig:cm_err}}
\\end{{figure}}

Bảng~\\ref{{tab:perclass}} và Hình~\\ref{{fig:cm_err}} cho thấy cặp nhầm lớn nhất là \\texttt{{mel}}$\\rightarrow$\\texttt{{nv}} (tỉ lệ hàng ${f3(cp1.rate_mean)} \\pm {f3(cp1.rate_std)}$, trung bình ${f1(cp1.count_mean)}/222$ ảnh mỗi seed) do sự tương đồng sắc tố nâu/đen ở độ phân giải $224{{\\times}}224$. Toàn tập test có ${int(cs_te.always_wrong.sum())}/2\\,004$ ảnh (${cs_te.always_wrong.sum()/2004*100:.1f}\\%$) bị đoán sai ở cả 3 seed, tập trung ở các ca không điển hình của \\texttt{{mel}}, \\texttt{{df}}, \\texttt{{akiec}}. Kiểm tra riêng hàng \\texttt{{mel}} giải thích trọn vẹn tác dụng phụ của \\texttt{{hair\\_bh17}}: recall \\texttt{{mel}} trên test bị kéo từ ${f3(mel_b.loc['mel','rate_mean'])}$ (baseline) xuống ${f3(mel_h.loc['mel','rate_mean'])}$ ở bản đại trà (đẩy nhầm sang \\texttt{{bkl}} từ ${f3(mel_b.loc['bkl','rate_mean'])}$ lên ${f3(mel_h.loc['bkl','rate_mean'])}$ và \\texttt{{nv}} từ ${f3(mel_b.loc['nv','rate_mean'])}$ lên ${f3(mel_h.loc['nv','rate_mean'])}$); bản có điều kiện \\texttt{{hair\\_bh17\\_cond}} nâng recall \\texttt{{mel}} lên ${f3(mel_c.loc['mel','rate_mean'])}$ nhưng đánh đổi giảm precision \\texttt{{mel}} (${f3(p_mel_base)} \\rightarrow {f3(p_mel_hcond)}$ do false positive từ \\texttt{{nv}}: recall \\texttt{{nv}} ${f3(r_nv_base)} \\rightarrow {f3(r_nv_hcond)}$), nên không cải thiện F1 tổng thể. Về liên hệ chất lượng ảnh (Hình~\\ref{{fig:cm_err}}b), sau khi chuẩn hóa gián tiếp theo thành phần lớp, nhóm \\emph{{sáng không đều}} có tỉ lệ lỗi vượt rõ kỳ vọng ($+13{{,}}0\\%$, CI95 loại 0; \\emph{{quá sáng}} cùng chiều $+7{{,}}4\\%$), còn cờ \\emph{{nhiễu Gauss}} có lỗi thấp hơn rõ kỳ vọng ($-5{{,}}9\\%$) vì gắn chặt với ảnh \\texttt{{nv}} nguồn \\texttt{{follow\\_up}} (lỗi chỉ $0{{,}}75\\%$ so với $17{{,}}04\\%$ ở \\texttt{{nv}} \\texttt{{histo}}). Hệ thống Demo Streamlit (\\texttt{{app.py}}, đóng gói sẵn qua \\texttt{{Docker Compose}} non-root chỉ đọc) tích hợp 3 tab trực tiếp trên RAM, đạt độ khớp ${app_chk['label_matches']}/{app_chk['n_checked']}$ nhãn kiểm thử và tuân thủ kiểm tra an toàn tệp tải lên cùng \\texttt{{weights\\_only=True}}.

\\section{{Thảo luận, Phân công \\& Khai báo Sử dụng AI}}
\\textbf{{Thảo luận và Hạn chế.}} Nghiên cứu chỉ ra rằng CNN tiền huấn luyện có \\texttt{{BatchNorm}} được fine-tune toàn mạng tự thích nghi tốt với biến thiên độ sáng vừa phải, còn bước resize $600{{\\rightarrow}}224$ đã khử nhiễu mịn; ngược lại, phép biến đổi toàn cục hoặc xóa lông đại trà làm suy hao tín hiệu độ tối và vi cấu trúc sắc tố trên hơn $90\\%$ ảnh bình thường. Tiền xử lý cổ điển phát huy hiệu quả rõ nhất khi đóng vai trò bộ kiểm soát chất lượng (QC), xử lý có điều kiện và khôi phục suy giảm ngoại lai nặng ở nhánh \\texttt{{test\\_only}}. Hạn chế chính của đồ án gồm: chỉ dùng 1 split $\\times$ 3 seed (bootstrap cụm \\texttt{{lesion\\_id}} chưa gộp phương sai tối ưu hóa giữa các seed và so sánh 16 cấu hình với chung 1 bộ baseline nên dễ nhạy ở lớp nhỏ \\texttt{{df, vasc}}), mask Otsu/black-hat ước lượng thô (sai trên ảnh rìa tối hoặc lông bạc, lệch nhẹ viền \\texttt{{fov}} vs \\texttt{{inner}} trên ảnh có viền tối), và chỉ khảo sát trong miền ảnh dermoscopy. Các hướng ngoài phạm vi (\\emph{{không thực hiện}}): phân vùng tổn thương bằng U-Net, chuẩn hóa màu Shades-of-Gray, dùng Focal Loss hoặc tăng độ phân giải $384{{\\times}}384$.

\\textbf{{Bảng phân công công việc (100\\% hoàn thành):}}
\\begin{{itemize}}
    \\item \\textbf{{Nguyễn Trung Nam (B23DCVT299):}} Thực nghiệm thống nhất trên NVIDIA L40S, khảo sát 3 kiến trúc CNN (2.4), hoàn thiện phân tích 4 tầng (2.5c), nâng cấp Demo Streamlit 3 tab, đóng gói Docker và chế bản LaTeX/Beamer.
    \\item \\textbf{{Phan Nguyễn Việt Dũng (B23DCCN210):}} Khung huấn luyện \\& kiểm định dùng chung (\\texttt{{src/training.py}}, \\texttt{{compare.py}}), dữ liệu \\& baseline (2.1, 2.3, 2.4), thực nghiệm 2.5.1--2.5.2 và thiết kế 2.5c.
    \\item \\textbf{{Lê Duy Anh (B23DCCN026):}} Lý thuyết Chương 3--4, phân tích thống kê \\& biên soạn báo cáo Mở đầu, 2.5.1--2.5.2 (nhiễu Gauss, tương phản thấp) và kịch bản thuyết trình.
    \\item \\textbf{{Đinh Khánh Sơn (B23DCDT223):}} Bộ chỉ số \\& phân tích EDA (2.2), thực nghiệm 4 vấn đề lông/viền/quá sáng/sáng lệch (2.5.3--2.5.6), phân tích lỗi 2.8 và khởi tạo ứng dụng Streamlit.
    \\item \\textbf{{Đào Phạm Bích Ngọc (B23DCVT308):}} Lý thuyết hình thái học Chương 7, đối chiếu trực quan lưới ảnh trước/sau 2.5.3--2.5.6, tổng hợp pipeline 2.6--2.7, kiểm tra chéo số liệu và slide.
\\end{{itemize}}

\\textbf{{Khai báo sử dụng AI.}} Nhóm sử dụng trợ lý lập trình AI (Claude / Gemini) hỗ trợ viết mã khung thực nghiệm, tối ưu kịch bản vẽ biểu đồ, xây dựng giao diện \\texttt{{app.py}} và định dạng LaTeX. Toàn bộ câu hỏi nghiên cứu, quy tắc thống kê đặt trước, việc chạy thực nghiệm trên tập HAM10000, kiểm tra trực quan 24 ảnh mask Otsu, ca lỗi lông trắng và kiểm chứng chéo mọi con số từ \\texttt{{outputs/}} đều do các thành viên trực tiếp thực hiện và chịu trách nhiệm.

\\FloatBarrier
\\begingroup
\\renewcommand{{\\section}}[2]{{\\subsection*{{Tài liệu tham khảo}}}}
\\footnotesize
\\bibliographystyle{{unsrt}}
\\bibliography{{references}}
\\endgroup

\\end{{document}}
"""
    (OVL / "main.tex").write_text(main_tex, encoding="utf-8")

    # Compile Overleaf PDF
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=OVL, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    shutil.copy2(OVL / "main.pdf", OVL / "BaoCao_XuLyAnh_HAM10000.pdf")

    # Create 1-click Overleaf upload zip
    zip_path = OVL / "Overleaf_Upload_HAM10000.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OVL / "main.tex", "main.tex")
        zf.write(OVL / "references.bib", "references.bib")
        for f in sorted((OVL / "figures").iterdir()):
            zf.write(f, f"figures/{f.name}")
    print("Đã biên dịch thành công:", OVL / "BaoCao_XuLyAnh_HAM10000.pdf", "và", zip_path)


def build_slides():
    SLD.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(OUT / "baseline/summary.csv")
    bb_val = pd.read_csv(OUT / "backbones/comparison_val.csv").set_index("arch")
    bb_test = pd.read_csv(OUT / "backbones/comparison_test.csv").set_index("arch")
    tv = pd.read_csv(OUT / "preprocess/all16_comparison_val.csv").set_index("config")
    tt = pd.read_csv(OUT / "preprocess/all16_comparison_test.csv").set_index("config")
    sub_err = pd.read_csv(OUT / "errors/subgroup_error_class_matched.csv")
    cp_te = pd.read_csv(OUT / "errors/confused_pairs_baseline_test.csv").head(4)
    hb = pd.read_csv(OUT / "impact/hair_lesion_by_class.csv").set_index("dx")
    fb = pd.read_csv(OUT / "impact/flips_excess.csv")
    fb_floor_te = float(fb[(fb.split == "test") & (fb.part == "all")]["flip_floor"].iloc[0]) * 100.0
    app_chk = json.loads((OUT / "errors/app_consistency_check.json").read_text(encoding="utf-8"))

    b_val = f"{f4(S['val_macro_f1'].mean())} \\pm {f4(S['val_macro_f1'].std(ddof=1))}"
    b_test = f"{f4(S['test_macro_f1'].mean())} \\pm {f4(S['test_macro_f1'].std(ddof=1))}"
    b_bacc = f"{f4(S['test_balanced_acc'].mean())} \\pm {f4(S['test_balanced_acc'].std(ddof=1))}"
    b_acc = f"{f4(S['test_acc'].mean())} \\pm {f4(S['test_acc'].std(ddof=1))}"

    slides_tex = f"""\\documentclass[aspectratio=169,10pt]{{beamer}}
\\usepackage[utf8]{{vietnam}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{booktabs,tabularx,array}}
\\usepackage{{graphicx}}
\\usepackage{{tikz}}
\\usetikzlibrary{{shapes.geometric,arrows.meta,positioning}}

\\usetheme{{Madrid}}
\\definecolor{{ptitred}}{{RGB}}{{180,25,35}}
\\definecolor{{ptitnavy}}{{RGB}}{{20,45,90}}
\\setbeamercolor{{palette primary}}{{bg=ptitnavy,fg=white}}
\\setbeamercolor{{palette secondary}}{{bg=ptitred,fg=white}}
\\setbeamercolor{{palette tertiary}}{{bg=ptitnavy,fg=white}}
\\setbeamercolor{{structure}}{{fg=ptitnavy}}
\\setbeamercolor{{block title}}{{bg=ptitnavy,fg=white}}
\\setbeamercolor{{block body}}{{bg=blue!4,fg=black}}
\\setbeamercolor{{alerted text}}{{fg=ptitred}}
\\setbeamertemplate{{navigation symbols}}{{}}
\\setbeamertemplate{{itemize items}}[circle]

\\title[Xử lý ảnh \\& CNN trên HAM10000]{{\\textbf{{Đánh Giá Tác Động Của Xử Lý Ảnh Cổ Điển Lên Phân Loại Tổn Thương Da HAM10000 Bằng Mạng Tích Chập}}}}
\\subtitle{{Đồ án Học phần Xử lý Ảnh -- Khoa Trí tuệ Nhân tạo, PTIT}}
\\author[Nhóm Đồ án XLA]{{\\small Nguyễn Trung Nam (B23DCVT299) \\quad Phan Nguyễn Việt Dũng (B23DCCN210)\\\\[2pt] Lê Duy Anh (B23DCCN026) \\quad Đinh Khánh Sơn (B23DCDT223) \\quad Đào Phạm Bích Ngọc (B23DCVT308)}}
\\institute[PTIT]{{Học viện Công nghệ Bưu chính Viễn thông}}
\\date{{Tháng 11 / 2026}}

\\begin{{document}}

% Slide 1
\\begin{{frame}}
\\titlepage
\\end{{frame}}

% Slide 2
\\begin{{frame}}{{1. Bài toán, Động cơ \\& Câu hỏi Nghiên cứu}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.54\\textwidth}}
\\begin{{block}}{{Triết lý thực nghiệm xuyên suốt}}
\\textbf{{Không săn điểm SOTA}} -- tập trung \\textbf{{Lập luận $\\rightarrow$ Thực nghiệm kiểm soát biến $\\rightarrow$ Đánh giá trung thực}} (kết quả âm vẫn có giá trị khoa học).
\\end{{block}}
\\vspace{{2pt}}
\\begin{{itemize}}
    \\item \\textbf{{Bài toán:}} Phân loại 7 lớp bệnh da trên tập ảnh soi da lâm sàng \\textbf{{HAM10000}} ($10\\,015$ ảnh $600{{\\times}}450$).
    \\item \\textbf{{Thách thức:}} Mất cân bằng lớp $58{{,}}3\\times$; nhiều ảnh chung \\texttt{{lesion\\_id}}; nhiễu thiết bị (viền đen, lóa gel, lông, tương phản thấp).
    \\item \\textbf{{Kiến thức học phần tích hợp:}} Chương 3 (nâng cao ảnh), Chương 4 (hệ màu), Chương 5 (Otsu/Sobel), Chương 7 (hình thái học), Chương 8 (CNN \\& Grad-CAM).
\\end{{itemize}}
\\end{{column}}
\\begin{{column}}{{0.43\\textwidth}}
\\begin{{block}}{{4 Câu hỏi nghiên cứu (RQ)}}
\\footnotesize
\\begin{{enumerate}}
    \\item \\textbf{{RQ1 (EDA):}} Có những lỗi chất lượng ảnh nào có bằng chứng thống kê \\& liên hệ lỗi CNN?
    \\item \\textbf{{RQ2 (Ablation):}} Xử lý ảnh cổ điển có tạo $\\Delta$ macro-F1 vượt dao động giữa 3 seed?
    \\item \\textbf{{RQ3 (4 tầng):}} Nếu F1 toàn tập không tăng, tiền xử lý tác động ở tầng nào?
    \\item \\textbf{{RQ4 (Pipeline):}} Kết hợp nhiều bước \\& chọn kiến trúc CNN tối ưu ra sao?
\\end{{enumerate}}
\\end{{block}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 3
\\begin{{frame}}{{2.1 Dữ liệu HAM10000 \\& Đặc thù Miền Ảnh Dermoscopy}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.55\\textwidth}}
\\centering
\\includegraphics[width=\\linewidth]{{figures/class_distribution.pdf}}
\\end{{column}}
\\begin{{column}}{{0.42\\textwidth}}
\\footnotesize
\\begin{{itemize}}
    \\item \\textbf{{Quy mô:}} $10\\,015$ ảnh RGB ($600{{\\times}}450$) từ $7\\,470$ \\texttt{{lesion\\_id}}.
    \\item \\textbf{{Mất cân bằng 58,3$\\times$:}} \\texttt{{nv}} chiếm $66{{,}}95\\%$ ($6\\,705$ ảnh), \\texttt{{df}} chỉ $115$ ảnh $\\Rightarrow$ luôn đoán \\texttt{{nv}} đạt Acc $66{{,}}9\\%$ nhưng \\alert{{Macro-F1 chỉ $0{{,}}1145$}}.
    \\item \\textbf{{Đa ảnh/lesion:}} $26{{,}}2\\%$ lesion có $2\\text{{--}}6$ ảnh ($44{{,}}9\\%$ số ảnh) $\\Rightarrow$ bắt buộc chia theo \\texttt{{lesion\\_id}}.
    \\item \\textbf{{Nguồn nhãn:}} $100\\%$ ảnh \\texttt{{follow\\_up}} ($3\\,704$ ảnh) thuộc lớp \\texttt{{nv}} $\\Rightarrow$ nguy cơ shortcut nguồn chụp.
    \\item \\textbf{{Lưu ý miền dữ liệu:}} Ảnh soi da áp sát (\\emph{{dermoscopy}}), khác miền với ảnh điện thoại.
\\end{{itemize}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 4
\\begin{{frame}}{{2.2 Phân tích Chất lượng Ảnh (EDA): 7 Chỉ số Định lượng}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.54\\textwidth}}
\\centering
\\includegraphics[width=\\linewidth]{{figures/eda_summary.pdf}}
\\end{{column}}
\\begin{{column}}{{0.43\\textwidth}}
\\scriptsize
\\begin{{tabular}}{{lrr}}
\\toprule
\\textbf{{Vấn đề (Ch.3, 4, 5, 7)}} & \\textbf{{Số ảnh}} & \\textbf{{Tỉ lệ}} \\\\
\\midrule
1. Lông (Black-hat $17{{\\times}}17$) & $1\\,016$ & $10{{,}}1\\%$ \\\\
2. Nhiễu Gauss ($\\sigma > 2{{,}}16$) & $734$ & $7{{,}}3\\%$ \\\\
3. Tương phản thấp ($<20$) & $435$ & $4{{,}}3\\%$ \\\\
4. Viền tối ống soi ($>1\\%$) & $355$ & $3{{,}}5\\%$ \\\\
5. Quá sáng bạc màu ($>10\\%$) & $302$ & $3{{,}}0\\%$ \\\\
6. Sáng không đều ($>38$) & $247$ & $2{{,}}5\\%$ \\\\
\\midrule
Quá tối / Muối--tiêu & $40$ / $1$ & \\alert{{Bác bỏ}} \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{4pt}}
\\begin{{itemize}}
    \\item Kiểm chứng bộ đo bằng nhiễu tổng hợp: sai số $<0{{,}}1$ mức xám $\\Rightarrow$ bác bỏ muối--tiêu là kết luận thực chứng.
    \\item $|\\rho_{{\\text{{Spearman}}}}| \\le 0{{,}}40$: 7 chỉ số độc lập, không đo trùng.
\\end{{itemize}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 5
\\begin{{frame}}{{2.3 Chuẩn bị Dữ liệu Chống Rò rỉ \\& Cân bằng Lớp WRS}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.48\\textwidth}}
\\footnotesize
\\begin{{block}}{{Chia tập cố định theo \\texttt{{lesion\\_id}} (seed 42)}}
\\texttt{{StratifiedGroupKFold}} 2 tầng (0 lesion trùng lặp):
\\begin{{itemize}}
    \\item \\textbf{{Train\\_fit:}} $6\\,408$ ảnh ($4\\,779$ lesion -- $63{{,}}98\\%$)
    \\item \\textbf{{Validation:}} $1\\,603$ ảnh ($1\\,196$ lesion -- $16{{,}}01\\%$)
    \\item \\textbf{{Test:}} $2\\,004$ ảnh ($1\\,495$ lesion -- $20{{,}}01\\%$)
\\end{{itemize}}
\\end{{block}}
\\vspace{{3pt}}
\\begin{{itemize}}
    \\item \\textbf{{Metadata:}} Điền \\texttt{{age}} thiếu (57 ảnh) bằng median tập train ($50{{,}}0$), giữ nguyên toàn bộ ảnh hợp lệ.
    \\item \\textbf{{Tránh bù 2 lần:}} Dùng \\texttt{{WeightedRandomSampler}} ($w_i = 1/n_{{y_i}}$, mỗi lớp $\\approx 14{{,}}29\\%$/epoch) + \\texttt{{CrossEntropyLoss}} \\emph{{không}} class weights.
\\end{{itemize}}
\\end{{column}}
\\begin{{column}}{{0.49\\textwidth}}
\\centering
\\includegraphics[width=\\linewidth]{{figures/wrs_distribution.pdf}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 6
\\begin{{frame}}{{2.4 Baseline \\& Khảo sát 3 Kiến trúc CNN (3 Seed: 42, 43, 44)}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.52\\textwidth}}
\\centering
\\includegraphics[width=\\linewidth]{{figures/backbone_comparison.pdf}}
\\end{{column}}
\\begin{{column}}{{0.45\\textwidth}}
\\scriptsize
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Kiến trúc}} & \\textbf{{Tham số}} & \\textbf{{Val F1}} & \\textbf{{Test F1}} \\\\
\\midrule
\\textbf{{ResNet-18}} & $11{{,}}18$M & ${f4(bb_val.loc['resnet18','macro_f1_mean'])}$ & $\\mathbf{{{f4(bb_test.loc['resnet18','macro_f1_mean'])}}}$ \\\\
EfficientNet-B0 & $4{{,}}02$M & ${f4(bb_val.loc['efficientnet_b0','macro_f1_mean'])}$ & ${f4(bb_test.loc['efficientnet_b0','macro_f1_mean'])}$ \\\\
MobileNetV3-S & $1{{,}}53$M & ${f4(bb_val.loc['mobilenet_v3_small','macro_f1_mean'])}$ & ${f4(bb_test.loc['mobilenet_v3_small','macro_f1_mean'])}$ \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{3pt}}
\\footnotesize
\\begin{{itemize}}
    \\item \\textbf{{ResNet-18 Baseline:}} Val F1 $= {b_val}$; Test F1 $= \\mathbf{{{b_test}}}$; bAcc $= {b_bacc}$; Acc $= {b_acc}$.
    \\item Chọn trên \\texttt{{val}}: \\textbf{{ResNet-18}} có độ lệch chuẩn thấp hơn $3{{,}}3\\times$ so với \\texttt{{EfficientNet-B0}} ($0{{,}}0044$ vs $0{{,}}0146$) và tầng \\texttt{{layer4}} $7{{\\times}}7$ chuẩn cho Grad-CAM.
    \\item Giữ cố định \\textbf{{ResNet-18}} cho toàn bộ 16 thí nghiệm mục 2.5 để kiểm soát biến tuyệt đối.
\\end{{itemize}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 7
\\begin{{frame}}{{2.5a Ablation Nhóm 1: Khử Nhiễu Gauss \\& Tăng Tương Phản}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.52\\textwidth}}
\\centering
\\includegraphics[width=\\linewidth]{{figures/contrast_metrics.pdf}}
\\end{{column}}
\\begin{{column}}{{0.45\\textwidth}}
\\scriptsize
\\begin{{tabular}}{{lccl}}
\\toprule
\\textbf{{Cấu hình}} & \\textbf{{Val F1}} & \\textbf{{Test F1}} & \\textbf{{Kết luận}} \\\\
\\midrule
\\texttt{{baseline}} & ${f4(tv.loc['baseline','macro_f1_mean'])}$ & ${f4(tt.loc['baseline','macro_f1_mean'])}$ & Đối chứng \\\\
\\texttt{{gauss3}} & ${f4(tv.loc['gauss3','macro_f1_mean'])}$ & ${f4(tt.loc['gauss3','macro_f1_mean'])}$ & {tt.loc['gauss3','macro_f1_verdict']} \\\\
\\texttt{{median3}} & ${f4(tv.loc['median3','macro_f1_mean'])}$ & ${f4(tt.loc['median3','macro_f1_mean'])}$ & {tt.loc['median3','macro_f1_verdict']} \\\\
\\texttt{{median5}} & ${f4(tv.loc['median5','macro_f1_mean'])}$ & ${f4(tt.loc['median5','macro_f1_mean'])}$ & {tt.loc['median5','macro_f1_verdict']} \\\\
\\texttt{{stretch\\_L\\_1\\_99}} & ${f4(tv.loc['stretch_L_1_99','macro_f1_mean'])}$ & ${f4(tt.loc['stretch_L_1_99','macro_f1_mean'])}$ & \\alert{{\\textbf{{Tệ hơn rõ}}}} \\\\
\\texttt{{hist\\_eq\\_L}} & ${f4(tv.loc['hist_eq_L','macro_f1_mean'])}$ & ${f4(tt.loc['hist_eq_L','macro_f1_mean'])}$ & \\alert{{\\textbf{{Tệ hơn rõ}}}} \\\\
\\texttt{{clahe\\_L\\_c2\\_t8}} & ${f4(tv.loc['clahe_L_c2_t8','macro_f1_mean'])}$ & ${f4(tt.loc['clahe_L_c2_t8','macro_f1_mean'])}$ & {tt.loc['clahe_L_c2_t8','macro_f1_verdict']} \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{3pt}}
\\footnotesize
\\begin{{itemize}}
    \\item Resize $600{{\\rightarrow}}224$ đã hạ $\\sigma$ nhiễu; \\texttt{{median5}} xóa $75\\%$ chi tiết Laplacian tổn thương.
    \\item \\alert{{\\texttt{{hist\\_eq\\_L}} \\& \\texttt{{stretch\\_L\\_1\\_99}}}} nhân nhiễu $\\times 3{{,}}5\\text{{--}}5{{,}}5$ và làm biến thiên độ chói $\\Delta E_{{76}} = 14\\text{{--}}22$ ($\\Delta L^*$) trên $95{{,}}7\\%$ ảnh thường!
\\end{{itemize}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 8
\\begin{{frame}}{{2.5b Ablation Nhóm 2: Đại trà vs Có điều kiện (Lông, Viền, Sáng)}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.42\\textwidth}}
\\centering
\\includegraphics[height=0.72\\textheight,keepaspectratio]{{figures/artifacts_before_after.png}}
\\end{{column}}
\\begin{{column}}{{0.55\\textwidth}}
\\scriptsize
\\begin{{block}}{{Bài học lớn nhất: Xử lý Đại trà vs Có điều kiện (\\texttt{{\\_cond}})}}
\\begin{{itemize}}
    \\item \\textbf{{Bản \\texttt{{\\_cond}}}} để nguyên \\textbf{{100\\% ảnh bình thường}} ($\\Delta E_{{76}} = 0$).
    \\item \\alert{{\\texttt{{hair\\_bh17}} (áp đại trà)}} sửa nhầm $92\\%$ ảnh không lông, xóa trung vị \\textbf{{{hb.loc['mel','median']*100:.1f}\\%}} diện tích tổn thương \\texttt{{mel}} (vs {hb.loc['nv','median']*100:.1f}\\% ở \\texttt{{nv}}) $\\Rightarrow$ giảm điểm cả 3/3 seed!
    \\item Ca lỗi kinh điển: \\textbf{{lông bạc/trắng}} bị black-hat bắt ngược khe da tối, gây loang mảng sau inpainting.
\\end{{itemize}}
\\end{{block}}
\\vspace{{2pt}}
\\centering
\\includegraphics[height=0.33\\textheight,keepaspectratio]{{figures/subgroup_delta.pdf}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 9
\\begin{{frame}}{{Tổng hợp 16 Cấu hình Tiền xử lý trên ResNet-18 (NVIDIA L40S)}}
\\centering
\\includegraphics[height=0.61\\textheight,keepaspectratio]{{figures/all16_macro_f1.pdf}}\\par
\\vspace{{3pt}}
\\footnotesize
\\textbf{{Kết luận RQ2 \\& RQ4:}} \\alert{{0/16 cấu hình}} (kể cả 2 cấu hình kết hợp \\texttt{{median3+clahe}} và \\texttt{{vig\\_fill+flat\\_cond}}) vượt chuẩn ``tốt hơn rõ'' trên Validation ($|\\Delta| > \\text{{std}}_{{\\text{{base}}}}$, CI95 loại 0, 3/3 seed dương).
\\end{{frame}}

% Slide 10
\\begin{{frame}}{{2.5c Giải mã Nghịch lý qua 4 Tầng Phân tích (RQ3)}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.48\\textwidth}}
\\footnotesize
\\begin{{enumerate}}
    \\item \\textbf{{Tầng A (Ảnh):}} Triệt tiêu $92\\%\\text{{--}}100\\%$ cờ EDA mục tiêu, nhưng CNN fine-tune toàn mạng có \\texttt{{BatchNorm}} đã tự bù biến thiên sáng nhẹ.
    \\item \\textbf{{Tầng B (Đổi nhãn):}} Hai seed baseline tự bất đồng ${f1(fb_floor_te)}\\%$ ảnh test; \\texttt{{vig\\_fill, flat\\_cond}} sát nền, riêng \\texttt{{gamma\\_L\\_auto}} vọt $+20{{,}}5\\%$ trên nhóm quá sáng.
    \\item \\textbf{{Tầng C (Stress Test):}} Khi suy giảm ngoại lai nặng (viền lớn, lông dày, mất tương phản), nhánh \\texttt{{test\\_only}} \\textbf{{phục hồi mạnh F1}} ($+0{{,}}10$ đến $+0{{,}}46$)!
    \\item \\textbf{{Tầng D (Grad-CAM \\texttt{{layer4}}):}} Giảm chú ý vào viền/lông ở biên, giữ ổn định trọng tâm tổn thương.
\\end{{enumerate}}
\\end{{column}}
\\begin{{column}}{{0.50\\textwidth}}
\\centering
\\includegraphics[width=0.96\\linewidth]{{figures/impact_c2_stress.pdf}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 11
\\begin{{frame}}{{2.6 -- 2.7 Pipeline Đề xuất Hai Nhánh \\& Đóng băng Mô hình Cuối}}
\\centering
\\begin{{tikzpicture}}[node distance=0.8cm and 1.1cm, >=Latex, every node/.style={{font=\\footnotesize}}]
\\node[draw, rounded corners, fill=gray!15, minimum width=2.8cm, minimum height=1.1cm, align=center] (in) {{\\textbf{{Ảnh Dermoscopy}}\\\\RGB $600{{\\times}}450$}};
\\node[draw, rounded corners, fill=orange!15, minimum width=5.6cm, minimum height=1.3cm, align=center, above right=0.25cm and 1.2cm of in] (qc) {{\\textbf{{Nhánh (A): Kiểm tra Chất lượng (QC)}}\\\\6 bộ dò Ch.3--7 $\\rightarrow$ Gắn cờ cảnh báo độ tin cậy\\\\(+ Minh họa bước có điều kiện \\texttt{{vig\\_fill}}, \\texttt{{flat\\_cond}})}};
\\node[draw, rounded corners, fill=blue!12, minimum width=5.6cm, minimum height=1.3cm, align=center, below right=0.25cm and 1.2cm of in] (cnn) {{\\textbf{{Nhánh (B): Phân loại CNN (ResNet-18)}}\\\\Ảnh gốc $\\rightarrow$ Resize $224$ $\\rightarrow$ Normalize ImageNet\\\\$\\rightarrow$ Dự đoán 7 lớp + Bản đồ nhiệt Grad-CAM}};
\\draw[->, thick] (in.east) -- ++(0.45,0) |- (qc.west);
\\draw[->, thick] (in.east) -- ++(0.45,0) |- (cnn.west);
\\end{{tikzpicture}}
\\vspace{{6pt}}
\\begin{{block}}{{Lý do thiết kế từ thực nghiệm}}
\\footnotesize
\\begin{{itemize}}
    \\item Không ép ảnh qua biến đổi pixel cố định ở nhánh (B) vì 0/16 cấu hình giúp tăng F1 trên dữ liệu thật và có nguy cơ làm mất tín hiệu màu/sắc tố.
    \\item Đưa tri thức Chương 3--7 vào \\textbf{{Nhánh (A) QC}} để phát hiện ảnh sáng lệch / quá sáng (nhóm có lỗi thực tế cao hơn kỳ vọng) và hỗ trợ giải thích trực quan.
\\end{{itemize}}
\\end{{block}}
\\end{{frame}}

% Slide 12
\\begin{{frame}}{{2.8 Phân tích Lỗi Chuyên sâu của Mô hình Cuối}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.46\\textwidth}}
\\centering
\\includegraphics[width=0.92\\linewidth]{{figures/confusion_test.pdf}}
\\end{{column}}
\\begin{{column}}{{0.52\\textwidth}}
\\centering
\\includegraphics[width=0.95\\linewidth]{{figures/eda_cnn_link.pdf}}
\\vspace{{2pt}}
\\scriptsize
\\begin{{itemize}}
    \\item \\textbf{{Cặp nhầm số 1:}} \\texttt{{mel}}$\\rightarrow$\\texttt{{nv}} (${f3(cp_te.iloc[0].rate_mean)}$, TB ${f1(cp_te.iloc[0].count_mean)}/222$ ảnh) do chồng lấn sắc tố ở $224{{\\times}}224$.
    \\item \\textbf{{Nhóm ảnh khó:}} $\\approx 8\\%$ ảnh test sai ở cả 3 seed (chủ yếu ca không điển hình của \\texttt{{mel, df, akiec}}).
    \\item \\textbf{{Chuẩn hóa theo lớp:}} \\emph{{Sáng không đều}} \\& \\emph{{quá sáng}} có lỗi cao hơn kỳ vọng; cờ \\emph{{nhiễu}} lỗi thấp do gắn nguồn \\texttt{{follow\\_up}} (\\texttt{{nv}}).
\\end{{itemize}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 13
\\begin{{frame}}{{2.8 Hệ thống Demo Tương tác Streamlit (\\texttt{{app.py}})}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.52\\textwidth}}
\\footnotesize
\\begin{{block}}{{Kiến trúc 3 Tab trực quan \\& An toàn}}
\\begin{{enumerate}}
    \\item \\textbf{{Tab 1 -- QC \\& Dự đoán:}} Chạy trực tiếp 6 bộ dò EDA, cảnh báo cờ vượt ngưỡng, so sánh xác suất 7 lớp của \\texttt{{baseline\\_seed42.pt}} (khớp \\textbf{{{app_chk['label_matches']}/{app_chk['n_checked']}}} nhãn kiểm thử).
    \\item \\textbf{{Tab 2 -- Từng giai đoạn \\& Stress Test:}} Chuỗi ảnh qua từng phép xử lý, bản đồ $|\\text{{sau}}-\\text{{trước}}|$, $\\Delta E_{{76}}$, mask Otsu/lông, histogram FOV \\& thanh trượt giả lập suy giảm 2.5c.
    \\item \\textbf{{Tab 3 -- Grad-CAM \\& Bảng số liệu:}} Nội suy Grad-CAM \\texttt{{layer4}}, đo \\% năng lượng chú ý trong tổn thương vs lông.
\\end{{enumerate}}
\\end{{block}}
\\end{{column}}
\\begin{{column}}{{0.45\\textwidth}}
\\centering
\\includegraphics[height=0.68\\textheight,keepaspectratio]{{figures/impact_d3_grid.png}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

% Slide 14
\\begin{{frame}}{{3. Kết luận, Hạn chế, Phân công \\& Khai báo AI}}
\\begin{{columns}}[T,onlytextwidth]
\\begin{{column}}{{0.53\\textwidth}}
\\footnotesize
\\begin{{block}}{{Tổng kết đóng góp}}
\\begin{{itemize}}
    \\item Chứng minh thực nghiệm trên 51 lượt train ResNet-18 + 6 lượt khảo sát kiến trúc: CNN fine-tune tự thích nghi tốt trên ảnh dermoscopy chuẩn; biến đổi toàn cục/đại trà dễ xóa tín hiệu chẩn đoán.
    \\item Giải mã trọn vẹn cơ chế qua 4 tầng phân tích \\& đề xuất pipeline 2 nhánh (QC + CNN) trung thực với số liệu.
\\end{{itemize}}
\\end{{block}}
\\vspace{{2pt}}
\\scriptsize
\\textbf{{Hạn chế \\& Hướng mở:}} 1 split $\\times$ 3 seed; mask Otsu/black-hat thô (lỗi trên rìa tối \\& lông bạc); hướng mở dùng U-Net segmentation, color constancy Chương 4 và Focal Loss.
\\end{{column}}
\\begin{{column}}{{0.45\\textwidth}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lll}}
\\toprule
\\textbf{{Thành viên}} & \\textbf{{MSV}} & \\textbf{{Phụ trách chính}} \\\\
\\midrule
Nguyễn Trung Nam & B23DCVT299 & Run L40S, 3 CNN, 2.5c, Demo, PDF \\\\
Phan Nguyễn Việt Dũng & B23DCCN210 & Khung train, 2.1--2.4, 2.5a, 2.5c \\\\
Lê Duy Anh & B23DCCN026 & Lý thuyết \\& BC Mở đầu, 2.5a \\\\
Đinh Khánh Sơn & B23DCDT223 & EDA 2.2, 2.5b, lỗi 2.8, Demo \\\\
Đào Phạm Bích Ngọc & B23DCVT308 & Hình thái Ch.7, 2.5b, 2.6--2.7, Slide \\\\
\\bottomrule
\\end{{tabular}}%
}}
\\vspace{{4pt}}
\\begin{{block}}{{Khai báo AI minh bạch}}
\\tiny AI hỗ trợ viết khung mã nguồn, biểu đồ, UI \\texttt{{app.py}} và chế bản LaTeX. Nhóm trực tiếp thiết kế giả thuyết, chạy thực nghiệm HAM10000, kiểm tra trực quan mask/ảnh lỗi và đối soát 100\\% con số từ \\texttt{{outputs/}}.
\\end{{block}}
\\end{{column}}
\\end{{columns}}
\\end{{frame}}

\\end{{document}}
"""
    (SLD / "slides.tex").write_text(slides_tex, encoding="utf-8")

    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "slides.tex"], cwd=SLD, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    shutil.copy2(SLD / "slides.pdf", SLD / "Slide_XuLyAnh_HAM10000.pdf")

    # Kịch bản thuyết trình nhóm đồng bộ 14 slide
    script_md = f"""# Kịch Bản Thuyết Trình Nhóm (14 Slide – Đồ Án Xử Lý Ảnh HAM10000)

> Đồng bộ 100% với tệp `slides/Slide_XuLyAnh_HAM10000.pdf` và số liệu thống nhất trên NVIDIA L40S. Thời lượng dự kiến: 12–15 phút + Demo trực tiếp `app.py`.

---

## Phần 1 (Lê Duy Anh & Phan Nguyễn Việt Dũng): Slide 1 – 6 (Giới thiệu, Dữ liệu, EDA, Chuẩn bị & Baseline)

- **Slide 1 (Trang bìa):** Chào thầy/cô và các bạn. Nhóm chúng em xin trình bày đồ án học phần Xử lý ảnh với đề tài: *Đánh giá tác động của các phương pháp xử lý ảnh cổ điển lên phân loại tổn thương da HAM10000 bằng mạng tích chập*.
- **Slide 2 (Bài toán & Câu hỏi nghiên cứu):** Điểm quan trọng nhất trong cách tiếp cận của nhóm là **không chạy theo điểm số SOTA**, mà đi theo hướng nghiên cứu khoa học: đặt giả thuyết từ lý thuyết Chương 3, 4, 5, 7, thiết kế thí nghiệm kiểm soát biến trên CNN Chương 8 và báo cáo trung thực cả các kết quả âm. Nhóm đặt 4 câu hỏi RQ1 đến RQ4 xuyên suốt từ EDA đến pipeline cuối.
- **Slide 3 (Dữ liệu HAM10000):** Bộ dữ liệu gồm 10 015 ảnh soi da kích thước 600×450 của 7 470 tổn thương, chia làm 7 lớp. Có 3 điểm đặc biệt: mất cân bằng 58,3 lần (`nv` chiếm 67% nên đoán toàn `nv` vẫn đạt accuracy 66,9% nhưng macro-F1 chỉ 0,1145); 26,2% tổn thương có nhiều ảnh nên bắt buộc chia theo `lesion_id`; và toàn bộ 3 704 ảnh `follow_up` đều thuộc lớp `nv`. Ngoài ra đây là ảnh soi da chuyên dụng (*dermoscopy*), không phải ảnh điện thoại.
- **Slide 4 (EDA Chất lượng ảnh):** Nhóm đo tự động 7 chỉ số trên toàn bộ 10 015 ảnh, kiểm chứng bộ đo bằng nhiễu tổng hợp trước rồi mới kết luận. Kết quả xác nhận 6 vấn đề tồn tại (lông 10,1%, nhiễu Gauss nhẹ 7,3%, tương phản thấp 4,3%, viền tối 3,5%, quá sáng 3,0%, sáng không đều 2,5%) và bác bỏ nhiễu muối–tiêu (chỉ 1 ảnh) cùng quá tối.
- **Slide 5 (Chuẩn bị dữ liệu & WRS):** Dữ liệu chia cố định theo `lesion_id` thành `train_fit` (6 408), `val` (1 603) và `test` (2 004), kiểm tra 0 rò rỉ lesion. Trên tập train, nhóm dùng `WeightedRandomSampler` để cân bằng kỳ vọng mỗi lớp 14,29%/epoch và không gắn thêm trọng số vào `CrossEntropyLoss` để tránh bù hai lần.
- **Slide 6 (Baseline & Khảo sát 3 kiến trúc CNN):** Dưới cùng cấu hình đóng băng 3 seed (42, 43, 44), trên tập `val` (tập chọn mô hình), `ResNet-18` đạt `0,7254 ± 0,0044` với độ lệch chuẩn thấp hơn 3,3 lần so với `EfficientNet-B0` (`0,7355 ± 0,0146`) và vượt xa `MobileNetV3-Small` (`0,6842 ± 0,0306`), đồng thời có tầng `layer4` 7×7 thuận lợi cho Grad-CAM nên được khóa cố định cho toàn bộ 16 thí nghiệm tiền xử lý. Trên tập `test`, `ResNet-18` đạt `{f4(S['test_macro_f1'].mean()).replace('{,}', ',')} ± {f4(S['test_macro_f1'].std(ddof=1)).replace('{,}', ',')}` (không khác biệt thống kê với `EfficientNet-B0` `{f4(bb_test.loc['efficientnet_b0','macro_f1_mean']).replace('{,}', ',')}`, vượt rõ `MobileNetV3-Small` `{f4(bb_test.loc['mobilenet_v3_small','macro_f1_mean']).replace('{,}', ',')}`).

---

## Phần 2 (Đinh Khánh Sơn & Đào Phạm Bích Ngọc): Slide 7 – 10 (Thực nghiệm 2.5a, 2.5b và Giải mã 4 tầng 2.5c)

- **Slide 7 (Nhóm 1 – Nhiễu Gauss & Tương phản thấp):** Ở phần nhiễu, bước resize từ 600×450 về 224×224 vốn đã khử phần lớn nhiễu tần số cao, còn lọc `median5` làm mất 75% chi tiết vùng tổn thương. Ở phần tương phản, hai phép biến đổi toàn cục `hist_eq_L` và `stretch_L_1_99` làm giảm rất mạnh Macro-F1 trên cả 3 seed vì khuếch đại nhiễu gấp 3,5–5,5 lần và làm biến thiên mạnh độ chói $\\Delta E_{{76}}$ (chủ yếu $\\Delta L^*$) trên 95,7% ảnh bình thường; trong khi CLAHE cục bộ giữ tính trung tính.
- **Slide 8 (Nhóm 2 – Đại trà vs Có điều kiện):** Với lông, viền tối, quá sáng và sáng không đều, bài học quan trọng nhất là phân biệt **áp đại trà** với **áp có điều kiện theo cờ EDA**. Bản `hair_bh17` đại trà sửa nhầm 92% ảnh không có lông và xóa trung vị {hb.loc['mel','median']*100:.1f}% diện tích vùng tổn thương của u hắc tố `mel`, làm cả 3 seed cùng giảm điểm. Các bản có điều kiện (`_cond`) bảo toàn 100% ảnh bình thường; riêng `gamma_L_auto` cho thấy việc cưỡng bức hạ độ sáng vùng cháy trắng mất sắc độ thành xám tro trên 3% dữ liệu có thể gây lệch phân phối cục bộ (`DiD = -0,209`).
- **Slide 9 (Tổng hợp 16 cấu hình):** Nhìn tổng thể cả 16 cấu hình đơn lẻ và kết hợp trên cùng một biểu đồ, 0/16 cấu hình vượt ngưỡng "cải thiện rõ" trên tập validation.
- **Slide 10 (Giải mã ở 4 tầng phân tích 2.5c):** Tại sao chỉ số ảnh ở Tầng A cải thiện 92–100% mà Macro-F1 không tăng? Vì ResNet-18 có BatchNorm được fine-tune toàn mạng nên tự thích nghi với lệch sáng vừa phải, còn mức nền bất đồng tự nhiên giữa 2 seed baseline (Tầng B) vốn đã là {f1(fb_floor_te).replace('{,}', ',')}% trên tập test. Tuy nhiên ở **Stress Test Tầng C**, khi ảnh chịu suy giảm ngoại lai nặng (viền che lớn, lông dày, mất tương phản 0,3×), tiền xử lý ngay ở nhánh `test_only` khôi phục rất mạnh điểm F1 (`+0,10` đến `+0,46`)!

---

## Phần 3 (Nguyễn Trung Nam): Slide 11 – 14 & Demo Streamlit Trực Tiếp (Docker)

- **Slide 11 (Pipeline hai nhánh 2.6–2.7):** Từ số liệu trên, nhóm không ép ảnh qua biến đổi pixel cố định trước CNN mà thiết kế **pipeline 2 nhánh**: Nhánh (A) dùng 6 bộ dò Chương 3–7 làm cổng Kiểm tra chất lượng (QC) gắn cờ cảnh báo; Nhánh (B) phân loại trực tiếp trên ảnh gốc bằng ResNet-18 kèm giải thích Grad-CAM.
- **Slide 12 (Phân tích lỗi 2.8):** Cặp nhầm lớn nhất là `mel` sang `nv` do tương đồng sắc tố. Khi chuẩn hóa tỉ lệ lỗi theo thành phần lớp, hai nhóm *sáng không đều* và *quá sáng* thực sự có tỉ lệ lỗi vượt mức kỳ vọng — khẳng định giá trị thực tế của cờ cảnh báo QC ở Nhánh (A).
- **Slide 13 & Chuyển sang Demo trực tiếp (`docker compose up` / `streamlit run app.py`):** Mời thầy/cô xem bản demo trực tiếp (đã đóng gói Docker container non-root an toàn) gồm 3 tab: (1) chọn ảnh mẫu từ tập test để xem bảng cờ QC và đối chiếu xác suất dự đoán; (2) bật thử thanh trượt Stress Test 2.5c và xem mask hình thái học, bản đồ thay đổi pixel cùng histogram trước/sau; (3) xem trực tiếp bản đồ nhiệt Grad-CAM trên vùng tổn thương và bảng tổng hợp 16 cấu hình.
- **Slide 14 (Kết luận, Phân công & Khai báo AI):** Tóm tắt đóng góp, hạn chế phương pháp luận (1 split, mask Otsu/black-hat thô), bảng phân công thành viên và khai báo minh bạch việc sử dụng AI hỗ trợ. Em xin cảm ơn thầy/cô và xin phép nhận câu hỏi ạ!
"""
    (SLD / "KichBan_ThuyetTrinh_Nhom.md").write_text(script_md, encoding="utf-8")
    print("Đã biên dịch thành công:", SLD / "Slide_XuLyAnh_HAM10000.pdf", "và kịch bản thuyết trình!")


def main():
    copy_figures()
    build_overleaf()
    build_slides()


if __name__ == "__main__":
    main()
