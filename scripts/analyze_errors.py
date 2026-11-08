"""2.8: phân tích lỗi của baseline (ResNet-18, 3 seed) – chỉ dùng dự đoán ngoài mẫu huấn luyện đã lưu, không train.

    python -X utf8 scripts/analyze_errors.py cm        # (1) confusion matrix + cặp lớp hay nhầm + ảnh sai nhất quán
    python -X utf8 scripts/analyze_errors.py hair_mel  # (2) kiểm chứng hair_bh17 giảm recall mel
    python -X utf8 scripts/analyze_errors.py subgroup_norm  # (3) lỗi nhóm cờ EDA so với kỳ vọng theo thành phần lớp
    python -X utf8 scripts/analyze_errors.py fig_before_after  # (4) Hình 2.5-6 bản cuối (PDF + PNG 300 dpi)

Đầu ra: outputs/errors/*.csv, outputs/figures/errors/*.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda.plots import CLASSES, setup_style  # noqa: E402
from src.training import read_rgb  # noqa: E402

OUT, FIG = Path("outputs/errors"), Path("outputs/figures/errors")
SEEDS = (42, 43, 44)


def savefig(fig, name, dpi=300):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def run_dir(name):
    return Path("outputs/baseline") if name == "baseline" else Path("outputs/experiments") / name


def load_preds(name="baseline", split="test"):
    """Bảng rộng 1 dòng/ảnh: label, pred_42/43/44, correct_42/43/44."""
    out = None
    for s in SEEDS:
        P = pd.read_csv(run_dir(name) / f"seed{s}" / f"{split}_preds.csv")
        P = P[["image_id", "lesion_id", "dx", "label", "pred", "correct"]].rename(
            columns={"pred": f"pred_{s}", "correct": f"correct_{s}"})
        out = P if out is None else out.merge(P.drop(columns=["lesion_id", "dx", "label"]), on="image_id")
    out["n_correct"] = out[[f"correct_{s}" for s in SEEDS]].sum(axis=1)
    return out


def confusion(P):
    """(3, 7, 7) số ảnh theo seed: hàng = nhãn thật, cột = dự đoán."""
    C = np.zeros((len(SEEDS), 7, 7))
    for i, s in enumerate(SEEDS):
        np.add.at(C[i], (P["label"].values, P[f"pred_{s}"].values), 1)
    return C


# ------------------------------------------------------------------ (1)
def cmd_cm():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    for split in ("test", "val"):
        P = load_preds("baseline", split)
        C = confusion(P)
        R = C / C.sum(2, keepdims=True)                       # chuẩn hóa theo hàng = recall
        n_cls = C[0].sum(1).astype(int)

        # bảng confusion: mean số ảnh và mean ± std tỉ lệ hàng
        rows = []
        for i, t in enumerate(CLASSES):
            for j, p in enumerate(CLASSES):
                rows.append(dict(split=split, true=t, pred=p, n_true=n_cls[i], count_mean=C[:, i, j].mean(),
                                 rate_mean=R[:, i, j].mean(), rate_std=R[:, i, j].std(ddof=1),
                                 rate_per_seed="; ".join(f"{v:.3f}" for v in R[:, i, j])))
        T = pd.DataFrame(rows)
        T.to_csv(OUT / f"confusion_baseline_{split}.csv", index=False)
        pairs = T[T.true != T.pred].sort_values("rate_mean", ascending=False)
        pairs.to_csv(OUT / f"confused_pairs_baseline_{split}.csv", index=False)

        # hình confusion matrix (test là hình chính)
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        Rm = R.mean(0)
        im = ax.imshow(Rm, cmap="Blues", vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{Rm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if Rm[i, j] > 0.55 else "black")
        ax.set_xticks(range(7), CLASSES)
        ax.set_yticks(range(7), [f"{c} (n={n})" for c, n in zip(CLASSES, n_cls)])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_title(f"Baseline, {split} (n={len(P)}): tỉ lệ theo hàng, TB 3 seed", fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label="tỉ lệ theo hàng (= recall trên đường chéo)")
        savefig(fig, f"confusion_baseline_{split}")

        # ảnh sai nhất quán (sai ở cả 3 seed) – "nhóm ảnh khó", tiêu chí tái lập được
        P["always_wrong"] = P["n_correct"] == 0
        by_cls = P.groupby("dx").agg(n=("image_id", "size"), always_wrong=("always_wrong", "sum"),
                                     always_right=("n_correct", lambda x: int((x == 3).sum())))
        by_cls = by_cls.reindex(CLASSES)
        by_cls["pct_always_wrong"] = (by_cls.always_wrong / by_cls.n * 100).round(1)
        by_cls["pct_always_right"] = (by_cls.always_right / by_cls.n * 100).round(1)
        by_cls.to_csv(OUT / f"consistency_by_class_{split}.csv")
        P.to_csv(OUT / f"preds_wide_baseline_{split}.csv", index=False)

        print(f"\n===== {split}: n = {len(P)} ảnh")
        print("Recall theo lớp (mean ± std):",
              ", ".join(f"{c} {R[:, i, i].mean():.3f}±{R[:, i, i].std(ddof=1):.3f}" for i, c in enumerate(CLASSES)))
        print("8 cặp nhầm nhiều nhất (tỉ lệ hàng):")
        print(pairs.head(8)[["true", "pred", "n_true", "count_mean", "rate_mean", "rate_std"]].round(3)
              .to_string(index=False))
        print("Sai ở cả 3 seed / đúng ở cả 3 seed theo lớp:")
        print(by_cls.to_string())

    # lưới ảnh khó (test): mỗi lớp tối đa 3 ảnh sai cả 3 seed, rút ngẫu nhiên seed 42 (quy tắc đặt trước)
    P = pd.read_csv(OUT / "preds_wide_baseline_test.csv")
    paths = pd.read_csv("outputs/splits/test.csv").set_index("image_id")["path"]
    W = P[P.n_correct == 0]
    pick = pd.concat([W[W.dx == c].sample(min(3, int((W.dx == c).sum())), random_state=42) for c in CLASSES])
    n = len(pick)
    ncol = 6
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(ncol * 1.9, np.ceil(n / ncol) * 1.75))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), pick.itertuples()):
        preds = [CLASSES[getattr(r, f"pred_{s}")] for s in SEEDS]
        ax.imshow(read_rgb(paths[r.image_id]))
        ax.set_title(f"{r.image_id[5:]} thật: {r.dx}\nđoán: {'/'.join(preds)}", fontsize=6.5)
    fig.suptitle("Test: ảnh bị baseline đoán sai ở cả 3 seed (≤ 3 ảnh/lớp, rút ngẫu nhiên seed 42)", fontsize=8)
    fig.tight_layout()
    savefig(fig, "grid_always_wrong_test", dpi=200)
    print("\nĐã lưu:", OUT, FIG)


# ------------------------------------------------------------------ (2)
def _hair_cover(path):
    """Tỉ lệ mask tổn thương (Otsu thô, EDA) bị mask lông của hair_bh17 phủ, và tỉ lệ mask lông nằm TRONG tổn thương.
    Kiểm H-L: nếu black-hat bắt cấu trúc sắc tố thì mask 'lông' tập trung trong tổn thương."""
    import cv2
    from src.eda.quality_metrics import lesion_mask, vignette_mask
    from src.preprocess.artifacts import hair_mask_full
    rgb = read_rgb(path)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    les, _ = lesion_mask(g, ~vignette_mask(g))
    h = hair_mask_full(rgb, k=17, thr=20, dilate=1).astype(bool)
    return dict(lesion_covered=(h & les).sum() / max(les.sum(), 1),
                hair_in_lesion=(h & les).sum() / max(h.sum(), 1), hair_px=int(h.sum()), lesion_px=int(les.sum()))


def cmd_hair_mel():
    """Kiểm chứng: hair_bh17 (toàn bộ) giảm recall mel – ảnh mel đi đâu, và mask lông có phủ lên tổn thương không."""
    from concurrent.futures import ThreadPoolExecutor
    from scipy.stats import mannwhitneyu
    import cv2
    from src.preprocess.artifacts import hair_mask_full, remove_hair
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    mel = CLASSES.index("mel")
    fl = pd.read_csv("outputs/eda/image_flags.csv").set_index("image_id")
    paths = pd.concat([pd.read_csv(f"outputs/splits/{s}.csv") for s in ("val", "test")]).set_index("image_id")["path"]

    rows = []
    for split in ("test", "val"):
        for cfg in ("baseline", "hair_bh17", "hair_bh17_cond"):
            P = load_preds(cfg, split)
            C = confusion(P)
            R = C[:, mel] / C[:, mel].sum(1, keepdims=True)
            for j, c in enumerate(CLASSES):
                rows.append(dict(split=split, config=cfg, pred=c, rate_mean=R[:, j].mean(), rate_std=R[:, j].std(ddof=1),
                                 rate_per_seed="; ".join(f"{v:.3f}" for v in R[:, j])))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "mel_row_hair.csv", index=False)
    print("Hàng 'mel' của confusion matrix (tỉ lệ ảnh mel được đoán thành lớp cột), mean ± std 3 seed:")
    W = T.assign(v=T.rate_mean.map("{:.3f}".format) + "±" + T.rate_std.map("{:.3f}".format)) \
         .pivot_table(index=["split", "config"], columns="pred", values="v", aggfunc="first")[CLASSES]
    print(W.to_string())

    # ảnh mel 'bị hỏng': baseline đúng ≥ 2/3 seed, hair_bh17 sai ≥ 2/3 seed (quy tắc đặt trước); 'ổn định' = đúng ≥2/3 ở cả hai
    res = []
    for split in ("test", "val"):
        B = load_preds("baseline", split)
        H = load_preds("hair_bh17", split).set_index("image_id")
        B = B[B.dx == "mel"].copy()
        B["n_correct_hair"] = H.loc[B.image_id, "n_correct"].values
        B["preds_hair"] = ["/".join(CLASSES[H.loc[i, f"pred_{s}"]] for s in SEEDS) for i in B.image_id]
        B["group"] = np.select([(B.n_correct >= 2) & (B.n_correct_hair <= 1), (B.n_correct >= 2) & (B.n_correct_hair >= 2)],
                               ["bị hỏng", "ổn định"], "khác")
        B["split"] = split
        res.append(B)
    M = pd.concat(res)
    M = M[M.group != "khác"].copy()
    M["F_hair"] = fl.loc[M.image_id, "F_hair"].values
    with ThreadPoolExecutor(24) as ex:
        cov = pd.DataFrame(list(ex.map(_hair_cover, paths.loc[M.image_id])))
    M = pd.concat([M.reset_index(drop=True), cov], axis=1)
    M.to_csv(OUT / "mel_hair_flips.csv", index=False)

    print("\nẢnh mel: 'bị hỏng' (baseline đúng ≥2/3 seed, hair_bh17 sai ≥2/3) so với 'ổn định' (đúng ≥2/3 ở cả hai)")
    for split in ("test", "val"):
        S = M[M.split == split]
        a, b = S[S.group == "bị hỏng"], S[S.group == "ổn định"]
        print(f"[{split}] bị hỏng n={len(a)} (dự đoán hair_bh17 chủ yếu: {a.preds_hair.str.split('/').explode().value_counts().head(3).to_dict()}) | ổn định n={len(b)}")
        print(f"   % ảnh có cờ lông (F_hair): bị hỏng {a.F_hair.mean()*100:.1f}% | ổn định {b.F_hair.mean()*100:.1f}%")
        for col, lab in [("lesion_covered", "tỉ lệ tổn thương bị mask 'lông' phủ"),
                         ("hair_in_lesion", "tỉ lệ mask 'lông' nằm trong tổn thương")]:
            if len(a) >= 3 and len(b) >= 3:
                p = mannwhitneyu(a[col], b[col]).pvalue
                print(f"   {lab}: trung vị bị hỏng {a[col].median():.3f} | ổn định {b[col].median():.3f} | Mann–Whitney p = {p:.3g}")

    # lưới: ảnh mel bị hỏng ở test (tối đa 6, rút ngẫu nhiên seed 42): gốc | mask (đỏ) | sau hair_bh17
    A = M[(M.split == "test") & (M.group == "bị hỏng")]
    A = A.sample(min(6, len(A)), random_state=42)
    if len(A):
        fig, axes = plt.subplots(len(A), 3, figsize=(7.2, 1.85 * len(A)), squeeze=False)
        for r, row in zip(axes, A.itertuples()):
            im = read_rgb(paths[row.image_id])
            m = hair_mask_full(im, k=17, thr=20, dilate=1).astype(bool)
            ov = im.copy()
            ov[m] = (255, 0, 0)
            for ax, img, t in zip(r, [im, ov, remove_hair(im, k=17, thr=20, radius=5)],
                                  [f"{row.image_id[5:]} mel, baseline đúng {row.n_correct}/3",
                                   f"mask 'lông' phủ {row.lesion_covered*100:.0f}% tổn thương",
                                   f"sau hair_bh17: đoán {row.preds_hair}"]):
                ax.imshow(img)
                ax.set_title(t, fontsize=6.5)
                ax.axis("off")
        fig.tight_layout()
        savefig(fig, "grid_mel_hair_flips_test", dpi=200)
    print("\nĐã lưu:", OUT / "mel_row_hair.csv", OUT / "mel_hair_flips.csv", FIG / "grid_mel_hair_flips_test.png")


# ------------------------------------------------------------------ (3)
FLAGS = {"F_hair": "Lông", "F_vignette": "Viền tối", "F_overexposed": "Quá sáng", "F_uneven_illum": "Sáng không đều",
         "F_noise": "Nhiễu Gauss", "F_low_contrast": "Tương phản thấp"}


def _obs_exp(err, cls, g):
    """Lỗi quan sát của nhóm và lỗi kỳ vọng nếu nhóm có tỉ lệ lỗi theo lớp như phần còn lại (chuẩn hóa gián tiếp)."""
    rest = pd.Series(err[~g]).groupby(cls[~g]).mean()
    e = pd.Series(cls[g]).map(rest).to_numpy(float)
    return err[g].mean(), np.nanmean(e)


def cmd_subgroup_norm(n_boot=1000):
    """Cột 'Liên quan lỗi CNN?' của bảng EDA: lỗi baseline của nhóm cờ so với lỗi KỲ VỌNG theo thành phần lớp."""
    OUT.mkdir(parents=True, exist_ok=True)
    fl = pd.read_csv("outputs/eda/image_flags.csv").set_index("image_id")
    rows = []
    for split in ("test", "val"):
        P = load_preds("baseline", split)
        err = 1 - P["n_correct"].to_numpy() / len(SEEDS)              # lỗi TB 3 seed của từng ảnh
        cls, les = P["dx"].to_numpy(), P["lesion_id"].to_numpy()
        uniq, inv = np.unique(les, return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        rng = np.random.default_rng(42)
        boots_idx = [np.concatenate([groups[k] for k in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]
        for f, name in FLAGS.items():
            g = fl.loc[P.image_id, f].to_numpy(bool)
            o, e = _obs_exp(err, cls, g)
            bt = []
            for idx in boots_idx:
                gi = g[idx]
                if gi.sum() == 0:
                    continue
                oo, ee = _obs_exp(err[idx], cls[idx], gi)
                bt.append(oo - ee)
            lo, hi = np.percentile(bt, [2.5, 97.5])
            rows.append(dict(split=split, flag=f, problem=name, n_group=int(g.sum()), err_group=o,
                             err_rest=err[~g].mean(), err_expected_class_matched=e, excess=o - e,
                             excess_ci_lo=lo, excess_ci_hi=hi,
                             verdict=("lỗi cao hơn kỳ vọng" if lo > 0 else "lỗi thấp hơn kỳ vọng" if hi < 0
                                      else "không khác kỳ vọng")))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "subgroup_error_class_matched.csv", index=False)
    print("Lỗi baseline (TB 3 seed) của nhóm cờ EDA so với lỗi kỳ vọng theo thành phần lớp của nhóm.")
    print("excess = lỗi nhóm − lỗi kỳ vọng; CI95 bootstrap cụm lesion (1000 lần); test là kết quả chính.")
    print(T.drop(columns="flag").round(3).to_string(index=False))
    print("\nĐã lưu:", OUT / "subgroup_error_class_matched.csv")


# ------------------------------------------------------------------ (4)
def cmd_fig_before_after():
    """Hình 2.5-6 bản cuối: trước/sau cho 4 vấn đề. Quy tắc chọn (đặt trước): trong nhóm cờ ∩ train_fit, ảnh có chỉ số
    EDA gần trung vị nhóm nhất và ảnh có chỉ số lớn nhất. Cột: gốc | trung gian (mask / nền ước lượng) | phương pháp."""
    import cv2
    from src.eda.quality_metrics import lesion_mask
    from src.preprocess import ARTIFACT_CANDIDATES as CAND
    from src.preprocess.artifacts import _fov, _gray, _norm_blur, _to_lab, hair_mask_full
    from src.eda.quality_metrics import vignette_mask
    setup_style()
    q = pd.read_csv("outputs/eda/quality_metrics.csv")
    fl = pd.read_csv("outputs/eda/image_flags.csv")
    q = q.merge(fl[["image_id"] + [c for c in fl if c.startswith("F_")]], on="image_id")
    q = q[q.image_id.isin(pd.read_csv("outputs/splits/train_fit.csv").image_id)]

    def overlay(im, m):
        o = im.copy()
        o[m.astype(bool)] = (255, 0, 0)
        return o

    def background(im):
        g = _gray(im)
        fov = _fov(g)
        les, _ = lesion_mask(g, fov)
        les = cv2.dilate(les.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        w = (fov & ~les).astype(np.float32)
        B = _norm_blur(_to_lab(im)[..., 0], w, 60.0)
        return np.clip(B / 100 * 255, 0, 255).astype(np.uint8)

    spec = [  # (cờ, chỉ số, tên, [(tiêu đề cột, hàm)])
        ("F_hair", "hair_frac", "Lông", [("mask 'lông' (đỏ)", lambda im: overlay(im, hair_mask_full(im, dilate=1))),
                                          ("hair_bh17", CAND["hair_bh17"])]),
        ("F_vignette", "vignette_frac", "Viền tối", [("vig_fill", CAND["vig_fill"]), ("vig_crop", CAND["vig_crop"])]),
        ("F_overexposed", "bright_frac", "Quá sáng", [("gamma_L_1.5", CAND["gamma_L_1.5"]),
                                                      ("gamma_L_auto", CAND["gamma_L_auto"])]),
        ("F_uneven_illum", "illum_range", "Sáng không đều", [("nền ước lượng B (L*)", background),
                                                             ("flat_L_s60", CAND["flat_L_s60"])]),
    ]
    rows = []
    for flag, met, name, cols in spec:
        g = q[q[flag]]
        med = g.iloc[(g[met] - g[met].median()).abs().argsort().iloc[0]]
        mx = g.loc[g[met].idxmax()]
        rows += [(name, "trung vị", med, met, cols), (name, "lớn nhất", mx, met, cols)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 1.95 * len(rows)))
    for r, (name, kind, row, met, cols) in zip(axes, rows):
        im = read_rgb(row.path)
        panels = [(f"{name} ({kind}): {row.image_id[5:]}, {row.dx}\n{met} = {row[met]:.3g}", im)]
        panels += [(t, f(im)) for t, f in cols]
        for ax, (t, img) in zip(r, panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255) if img.ndim == 2 else ax.imshow(img)   # thang xám cố định, không tự giãn
            ax.set_title(t, fontsize=6.5)
            ax.axis("off")
    fig.tight_layout()
    out = Path("outputs/figures/preprocess_artifacts")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_2_5_6_before_after.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_2_5_6_before_after.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Đã lưu:", out / "fig_2_5_6_before_after.{pdf,png}")


if __name__ == "__main__":
    cmds = {"cm": cmd_cm, "hair_mel": cmd_hair_mel, "subgroup_norm": cmd_subgroup_norm,
            "fig_before_after": cmd_fig_before_after}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(f"cách dùng: python -X utf8 scripts/analyze_errors.py {{{'|'.join(cmds)}}}")
    cmds[sys.argv[1]]()
