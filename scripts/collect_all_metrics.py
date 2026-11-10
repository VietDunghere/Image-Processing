"""Tổng hợp toàn bộ số liệu thực nghiệm thống nhất (16 cấu hình tiền xử lý, 3 kiến trúc CNN, lỗi chuẩn hóa theo lớp,
4 tầng 2.5c, kiểm tra shortcut follow_up ở lớp nv, mức nền đổi kết quả mel, và kiểm chứng khớp dự đoán app.py).

Đầu ra:
  - outputs/preprocess/all16_comparison_{val,test}.csv
  - outputs/figures/preprocess/fig_all16_macro_f1.{pdf,png}
  - outputs/errors/nv_followup_check.csv
  - outputs/errors/mel_baseline_floor.json
  - outputs/errors/app_consistency_check.json
  - outputs/unified_digest.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eda.plots import CLASSES, setup_style  # noqa: E402
from src.preprocess import compare as C  # noqa: E402
from src.training import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, read_rgb  # noqa: E402

ALL16 = [
    "gauss3", "median3", "median5",
    "stretch_L_1_99", "hist_eq_L", "clahe_L_c2_t8",
    "hair_bh17", "hair_bh17_cond",
    "flat_L_s60", "flat_L_s60_cond",
    "gamma_L_1.5", "gamma_L_auto",
    "vig_fill", "vig_crop",
    "median3+clahe_L_c2_t8", "vig_fill+flat_L_s60_cond",
]

GROUP_TAG = {
    "gauss3": "Nhiễu", "median3": "Nhiễu", "median5": "Nhiễu",
    "stretch_L_1_99": "Tương phản", "hist_eq_L": "Tương phản", "clahe_L_c2_t8": "Tương phản",
    "hair_bh17": "Lông", "hair_bh17_cond": "Lông",
    "flat_L_s60": "Sáng lệch", "flat_L_s60_cond": "Sáng lệch",
    "gamma_L_1.5": "Quá sáng", "gamma_L_auto": "Quá sáng",
    "vig_fill": "Viền tối", "vig_crop": "Viền tối",
    "median3+clahe_L_c2_t8": "Kết hợp", "vig_fill+flat_L_s60_cond": "Kết hợp",
}

GROUP_COLOR = {
    "Baseline": "#7A7A7A",
    "Nhiễu": "#0072B2",
    "Tương phản": "#D55E00",
    "Lông": "#CC79A7",
    "Sáng lệch": "#009E73",
    "Quá sáng": "#E69F00",
    "Viền tối": "#56B4E9",
    "Kết hợp": "#332288",
}


def build_all16_tables_and_fig():
    C.BASE_DIR = Path("outputs/baseline")
    C.EXP_DIR = Path("outputs/experiments")
    out_p = Path("outputs/preprocess")
    out_p.mkdir(parents=True, exist_ok=True)
    fig_p = Path("outputs/figures/preprocess")
    fig_p.mkdir(parents=True, exist_ok=True)

    tables = {}
    for split in ("val", "test"):
        T = C.comparison_table(ALL16, split, n_boot=1000)
        T.to_csv(out_p / f"all16_comparison_{split}.csv", index=False)
        tables[split] = T

    # Vẽ biểu đồ tổng hợp 17 cấu hình (baseline + 16 cấu hình) cho cả val và test
    setup_style()
    names = ["baseline"] + ALL16
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True)
    for ax, split, title in zip(axes, ("val", "test"),
                                ("(a) Validation Macro-F1 (n = 1 603, dùng để chọn cấu hình)",
                                 "(b) Test Macro-F1 (n = 2 004, chỉ báo cáo 1 lần)")):
        base_vals = C.load_summary("baseline")[f"{split}_macro_f1"].to_numpy()
        all_vals = []
        seen_groups = set()
        for i, nm in enumerate(names):
            v = C.load_summary(nm)[f"{split}_macro_f1"].to_numpy()
            all_vals.append(v)
            grp = "Baseline" if nm == "baseline" else GROUP_TAG[nm]
            col = GROUP_COLOR[grp]
            lbl = grp if grp not in seen_groups else None
            seen_groups.add(grp)
            ax.bar(i, v.mean(), width=0.68, color=col, alpha=0.78, yerr=v.std(ddof=1), capsize=2.5, label=lbl)
            ax.scatter(np.full(3, i) + np.array([-0.14, 0.0, 0.14]), v, s=11, color="k", zorder=3)
        ax.axhline(base_vals.mean(), color="#444444", ls="--", lw=0.9)
        ax.axhspan(base_vals.mean() - base_vals.std(ddof=1), base_vals.mean() + base_vals.std(ddof=1),
                   color="#7A7A7A", alpha=0.14)
        v_cat = np.concatenate(all_vals)
        ax.set_ylim(max(0.45, v_cat.min() - 0.04), min(0.82, v_cat.max() + 0.03))
        ax.set_ylabel("Macro-F1")
        ax.set_title(title, fontsize=9.5)
    axes[0].legend(ncol=8, fontsize=7.5, loc="upper right")
    axes[1].set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_p / "fig_all16_macro_f1.pdf", bbox_inches="tight")
    fig.savefig(fig_p / "fig_all16_macro_f1.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return tables


def check_nv_followup():
    meta = pd.read_csv("archive/HAM10000_metadata.csv")[["image_id", "dx_type"]]
    rows = []
    for split in ("val", "test"):
        ps = [pd.read_csv(f"outputs/baseline/seed{s}/{split}_preds.csv").set_index("image_id") for s in (42, 43, 44)]
        ids = ps[0].index
        err = 1.0 - np.mean([p.loc[ids, "correct"].to_numpy(float) for p in ps], axis=0)
        df = ps[0].loc[ids, ["dx", "lesion_id"]].reset_index().assign(err=err).merge(meta, on="image_id")
        nv = df[df.dx == "nv"]
        for dxt, g in nv.groupby("dx_type"):
            rows.append({"split": split, "dx_type": dxt, "n_images": len(g),
                         "n_lesions": int(g.lesion_id.nunique()), "err_mean": float(g.err.mean())})
    out = pd.DataFrame(rows)
    out.to_csv("outputs/errors/nv_followup_check.csv", index=False)
    return out


def check_mel_baseline_floor():
    res = {}
    for split in ("val", "test"):
        ps = {s: pd.read_csv(f"outputs/baseline/seed{s}/{split}_preds.csv").set_index("image_id") for s in (42, 43, 44)}
        mel_ids = ps[42][ps[42].dx == "mel"].index
        flips = []
        r2w = []
        for a, b in itertools.combinations((42, 43, 44), 2):
            ca = ps[a].loc[mel_ids, "correct"].to_numpy(bool)
            cb = ps[b].loc[mel_ids, "correct"].to_numpy(bool)
            flips.append(int((ca != cb).sum()))
            r2w.append(int((ca & ~cb).sum()))
        res[split] = {"n_mel": len(mel_ids), "pairwise_correct_disagree_counts": flips,
                      "pairwise_disagree_mean": float(np.mean(flips)), "r2w_counts": r2w}
    Path("outputs/errors/mel_baseline_floor.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


@torch.no_grad()
def check_app_consistency():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load("outputs/checkpoints/baseline_seed42.pt", map_location=dev, weights_only=True)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(ck["state_dict"])
    m = m.to(dev).eval()

    te = pd.read_csv("outputs/splits/test.csv").head(20)
    saved = pd.read_csv("outputs/baseline/seed42/test_preds.csv").set_index("image_id")
    matches = 0
    max_prob_diff = 0.0
    for r in te.itertuples():
        rgb = read_rgb(r.path)
        x = np.asarray(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        p = torch.softmax(m(t).float(), 1)[0].cpu().numpy()
        sv = saved.loc[r.image_id]
        sv_p = np.array([sv[f"prob_{c}"] for c in CLASSES], dtype=np.float32)
        matches += int(int(p.argmax()) == int(sv["pred"]))
        max_prob_diff = max(max_prob_diff, float(np.abs(p - sv_p).max()))
    out = {"n_checked": 20, "label_matches": matches, "max_abs_prob_diff_fp32_vs_amp": round(max_prob_diff, 6),
           "seed42_best_epoch": int(ck.get("best_epoch", -1))}
    Path("outputs/errors/app_consistency_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    t16 = build_all16_tables_and_fig()
    nv_fu = check_nv_followup()
    mel_fl = check_mel_baseline_floor()
    app_chk = check_app_consistency()

    digest = {
        "baseline_summary": pd.read_csv("outputs/baseline/summary.csv").to_dict("records"),
        "baseline_mean_std": pd.read_csv("outputs/baseline/summary_mean_std.csv", index_col=0).round(4).to_dict(),
        "backbones_val": pd.read_csv("outputs/backbones/comparison_val.csv").round(4).to_dict("records"),
        "backbones_test": pd.read_csv("outputs/backbones/comparison_test.csv").round(4).to_dict("records"),
        "all16_val": t16["val"][["config", "macro_f1_mean", "macro_f1_std", "macro_f1_delta",
                                 "macro_f1_delta_ci_lo", "macro_f1_delta_ci_hi", "macro_f1_delta_sign",
                                 "macro_f1_verdict", "balanced_acc_delta", "balanced_acc_verdict"]].round(4).to_dict("records"),
        "all16_test": t16["test"][["config", "macro_f1_mean", "macro_f1_std", "macro_f1_delta",
                                   "macro_f1_delta_ci_lo", "macro_f1_delta_ci_hi", "macro_f1_delta_sign",
                                   "macro_f1_verdict", "balanced_acc_delta", "balanced_acc_verdict"]].round(4).to_dict("records"),
        "subgroup_artifacts": pd.read_csv("outputs/preprocess_artifacts/subgroup.csv").round(4).to_dict("records"),
        "subgroup_error_class_matched": pd.read_csv("outputs/errors/subgroup_error_class_matched.csv").round(4).to_dict("records"),
        "confused_pairs_test": pd.read_csv("outputs/errors/confused_pairs_baseline_test.csv").head(10).round(4).to_dict("records"),
        "consistency_test": pd.read_csv("outputs/errors/consistency_by_class_test.csv").to_dict("records"),
        "consistency_val": pd.read_csv("outputs/errors/consistency_by_class_val.csv").to_dict("records"),
        "mel_row_hair": pd.read_csv("outputs/errors/mel_row_hair.csv").round(4).to_dict("records"),
        "nv_followup": nv_fu.round(4).to_dict("records"),
        "mel_baseline_floor": mel_fl,
        "app_consistency": app_chk,
        "impact_a": pd.read_csv("outputs/impact/image_level_summary.csv").round(4).to_dict("records"),
        "impact_hair_tests": pd.read_csv("outputs/impact/hair_lesion_tests.csv").round(5).to_dict("records"),
        "impact_b_excess": pd.read_csv("outputs/impact/flips_excess.csv").round(4).to_dict("records"),
        "impact_c_delta": pd.read_csv("outputs/impact/stress_delta.csv").query("metric == 'macro_f1' and level > 0").round(4).to_dict("records"),
        "impact_d_summary": pd.read_csv("outputs/impact/gradcam_summary.csv").round(4).to_dict("records"),
        "impact_d_sim": pd.read_csv("outputs/impact/gradcam_similarity.csv").round(4).to_dict("records"),
    }
    Path("outputs/unified_digest.json").write_text(json.dumps(digest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Đã xuất outputs/unified_digest.json thành công!")


if __name__ == "__main__":
    main()
