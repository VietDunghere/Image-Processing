"""2.5c tầng A: phép xử lý có sửa đúng vấn đề không, và làm đổi bao nhiêu ở ảnh không có vấn đề? (CPU, không dùng CNN)

Chỉ dùng ảnh train_fit (không nhìn val/test). Nhóm = cờ EDA ∩ train_fit; đối chứng = cùng cỡ, ngẫu nhiên seed 42.
Chỉ số mục tiêu đo lại bằng đúng bộ phát hiện EDA trên ảnh ĐÃ xử lý, trừ nhiễu và tương phản:
  - nhiễu: sigma_imm (Immerkaer), vì bộ đo EDA (phần dư trung vị) thiên lệch về 0 sau lọc trung vị;
  - tương phản: mask cố định lấy từ ảnh gốc, để phép xử lý không làm đổi mask Otsu.

    python -X utf8 scripts/impact_image_level.py run [--jobs 8] [--limit N]   # đo, ghi CSV
    python -X utf8 scripts/impact_image_level.py hair_lesion [--jobs 8]       # mask lông nằm trong tổn thương, theo lớp
    python -X utf8 scripts/impact_image_level.py figs                         # vẽ hình từ CSV

Đầu ra: outputs/impact/image_level_{per_image,summary}.csv, hair_lesion_*.csv; outputs/figures/impact/
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda.plots import CLASS_COLORS, CLASSES, setup_style  # noqa: E402
from src.eda.quality_metrics import BRIGHT_S, BRIGHT_V, illum_range, masks  # noqa: E402
from src.preprocess import get_preprocess  # noqa: E402
from src.preprocess.artifacts import hair_mask_full  # noqa: E402
from src.preprocess.image_metrics import measure, reference  # noqa: E402
from src.training import read_rgb  # noqa: E402

OUT, FIG = Path("outputs/impact"), Path("outputs/figures/impact")
PS = Path("outputs/eda/problem_sets")
# vấn đề → (chỉ số mục tiêu, hướng tốt, các phép xử lý, nhãn trục)
TARGETS = {
    "hair": ("hair_frac", "lower", ["hair_bh17", "hair_bh17_cond"], "tỉ lệ pixel lông"),
    "vignette": ("vignette_frac", "lower", ["vig_fill", "vig_crop"], "tỉ lệ diện tích viền đen"),
    "overexposed": ("bright_frac", "lower", ["gamma_L_1.5", "gamma_L_auto"], "tỉ lệ pixel trắng bạc"),
    "uneven_illum": ("illum_range", "lower", ["flat_L_s60", "flat_L_s60_cond"], "chênh nền (mức xám)"),
    "noise": ("sigma_imm", "lower", ["median3", "gauss3"], "σ nhiễu Immerkaer (mức xám)"),
    "low_contrast": ("contrast", "higher", ["clahe_L_c2_t8", "stretch_L_1_99", "hist_eq_L"], "tương phản da − tổn thương (mức xám)"),
}
PRIMARY = {"hair": "hair_bh17", "vignette": "vig_fill", "overexposed": "gamma_L_auto", "uneven_illum": "flat_L_s60_cond",
           "noise": "median3", "low_contrast": "clahe_L_c2_t8"}
METHOD_COLORS = ["#0072B2", "#E69F00", "#009E73"]
NAMES = {"hair": "Lông", "vignette": "Viền tối", "overexposed": "Quá sáng", "uneven_illum": "Sáng không đều",
         "noise": "Nhiễu Gauss", "low_contrast": "Tương phản thấp"}


def eda_targets(rgb: np.ndarray) -> dict:
    """Chỉ số mục tiêu theo đúng công thức của src.eda.quality_metrics.compute_one."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    M = masks(g)
    n = max(int(M["fov"].sum()), 1)
    bright = (hsv[..., 2] >= BRIGHT_V) & (hsv[..., 1] <= BRIGHT_S) & M["fov"]
    return dict(vignette_frac=float(1 - M["fov"].mean()), bright_frac=float(bright.sum() / n),
                hair_frac=float(M["hair"].sum() / n), illum_range=illum_range(g, M["skin"] & (g > M["otsu_t"])))


def select() -> pd.DataFrame:
    """Nhóm vấn đề ∩ train_fit + đối chứng cùng cỡ (seed 42) từ phần còn lại của train_fit."""
    tf = pd.read_csv("outputs/splits/train_fit.csv")
    rows = []
    for p in TARGETS:
        ids = set(pd.read_csv(PS / f"{p}.csv")["image_id"])
        a = tf[tf.image_id.isin(ids)].assign(problem=p, group="problem")
        b = tf[~tf.image_id.isin(ids)].sample(n=len(a), random_state=42).assign(problem=p, group="control")
        rows += [a, b]
    return pd.concat(rows, ignore_index=True)[["image_id", "lesion_id", "dx", "path", "problem", "group"]]


def _target(proc, R, metric):
    if metric in ("sigma_imm", "contrast"):
        return measure(proc, R)
    return {**eda_targets(proc), **measure(proc, R)}


def _worker(a):
    cv2.setNumThreads(1)
    image_id, lesion_id, dx, path, problem, group = a
    metric, _, methods, _ = TARGETS[problem]
    rgb = read_rgb(path)
    R = reference(rgb)
    before = _target(rgb, R, metric)
    orig_eda = eda_targets(rgb)
    out = []
    for m in methods:
        p = get_preprocess(m)(rgb)
        after = _target(p, R, metric)
        diff = (p != rgb).any(2)
        out.append(dict(image_id=image_id, lesion_id=lesion_id, dx=dx, problem=problem, group=group, method=m, metric=metric,
                        before=before[metric], after=after[metric], changed=bool(diff.any()),
                        changed_frac=float(diff[R["fov"]].mean()),
                        changed_frac_lesion=float(diff[R["lesion"]].mean()) if R["lesion"].sum() > 200 else np.nan,
                        dE76=after["dE76"], dab=after["dab"],
                        **{f"orig_{k}": v for k, v in orig_eda.items()}))
    return out


def _boot(values_by_lesion, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    k = len(values_by_lesion)
    return np.percentile([fn(np.concatenate([values_by_lesion[i] for i in rng.integers(0, k, k)])) for _ in range(n)], [2.5, 97.5])


def _groups(sub, col):
    return [g[col].to_numpy() for _, g in sub.groupby("lesion_id")]


def run(jobs, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select()
    if limit:
        sel = sel.groupby(["problem", "group"]).head(limit)
    jobs_a = [(r.image_id, r.lesion_id, r.dx, r.path, r.problem, r.group) for r in sel.itertuples()]
    with Pool(jobs) as pool:
        res = pool.map(_worker, jobs_a, chunksize=4)
    D = pd.DataFrame([r for rs in res for r in rs])
    D.to_csv(OUT / "image_level_per_image.csv", index=False)
    # Kiểm tra toàn vẹn: chỉ số ảnh gốc phải khớp quality_metrics.csv của EDA
    Q = pd.read_csv("outputs/eda/quality_metrics.csv").set_index("image_id")
    C = D.drop_duplicates("image_id").set_index("image_id")
    errs = {k: float(np.nanmax(np.abs(C[f"orig_{k}"] - Q.loc[C.index, k]))) for k in ("vignette_frac", "bright_frac", "hair_frac", "illum_range")}
    print("Sai số tối đa so với quality_metrics.csv:", errs, flush=True)
    assert max(errs.values()) < 1e-6, f"chỉ số lệch EDA: {errs}"
    rules = pd.read_csv("outputs/eda/table_flag_rules.csv").set_index("metric")["threshold"]
    rows = []
    for (problem, method, group), s in D.groupby(["problem", "method", "group"]):
        metric, way, _, _ = TARGETS[problem]
        s = s.assign(gain=(s.before - s.after) if way == "lower" else (s.after - s.before))
        ok = s.dropna(subset=["before", "after"])
        rec = dict(problem=problem, method=method, group=group, metric=metric, n_images=len(s), n_lesions=s.lesion_id.nunique(),
                   median_before=ok.before.median(), median_after=ok.after.median(), median_gain=ok.gain.median())
        rec["median_gain_lo"], rec["median_gain_hi"] = _boot(_groups(ok, "gain"), np.median)
        imp = ok.assign(improved=(ok.gain > 0).astype(float))
        rec["pct_improved"] = imp.improved.mean()
        rec["pct_improved_lo"], rec["pct_improved_hi"] = _boot(_groups(imp, "improved"), np.mean)
        thr = rules.get(metric, np.nan)
        rec["threshold"] = thr
        if group == "problem" and np.isfinite(thr):
            bad = (ok.after > thr) if way == "lower" else (ok.after < thr)
            was = (ok.before > thr) if way == "lower" else (ok.before < thr)
            base = ok[was].assign(resolved=(~bad[was]).astype(float))
            rec["n_flag_before"] = len(base)
            rec["resolved_rate"] = base.resolved.mean()
            rec["resolved_lo"], rec["resolved_hi"] = _boot(_groups(base, "resolved"), np.mean) if len(base) else (np.nan, np.nan)
        rec.update(pct_images_changed=s.changed.mean(), median_changed_frac=s.changed_frac.median(),
                   median_changed_frac_lesion=s.changed_frac_lesion.median(), median_dE76=s.dE76.median(), median_dab=s.dab.median())
        rows.append(rec)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "image_level_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(S[["problem", "method", "group", "n_images", "median_before", "median_after", "pct_improved", "resolved_rate", "pct_images_changed", "median_dE76", "median_changed_frac_lesion"]].round(3).to_string(), flush=True)


def _hair_les(a):
    cv2.setNumThreads(1)
    image_id, lesion_id, dx, path = a
    rgb = read_rgb(path)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    les = masks(g)["lesion"]
    m = cv2.dilate(hair_mask_full(rgb, k=17, thr=20, dilate=1), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    return dict(image_id=image_id, lesion_id=lesion_id, dx=dx, lesion_px=int(les.sum()),
                frac_lesion_inpainted=float(m[les].mean()) if les.sum() > 200 else np.nan, frac_image_inpainted=float(m.mean()))


def hair_lesion(jobs):
    """H-A3: mask lông của hair_bh17 phủ bao nhiêu % vùng tổn thương, theo lớp (toàn bộ train_fit). Một ảnh/lesion khi kiểm định."""
    from scipy.stats import kruskal, mannwhitneyu
    tf = pd.read_csv("outputs/splits/train_fit.csv")
    with Pool(jobs) as pool:
        res = pool.map(_hair_les, [(r.image_id, r.lesion_id, r.dx, r.path) for r in tf.itertuples()], chunksize=16)
    D = pd.DataFrame(res).dropna(subset=["frac_lesion_inpainted"])
    D.to_csv(OUT / "hair_lesion_per_image.csv", index=False)
    one = D.sort_values("image_id").drop_duplicates("lesion_id")
    S = D.groupby("dx")["frac_lesion_inpainted"].agg(n="size", mean="mean", median="median", p90=lambda x: x.quantile(.9)).reindex(CLASSES).reset_index()
    S.to_csv(OUT / "hair_lesion_by_class.csv", index=False)
    H = kruskal(*[g.frac_lesion_inpainted.values for _, g in one.groupby("dx")])
    eps2 = (H.statistic - one.dx.nunique() + 1) / (len(one) - one.dx.nunique())
    a, b = one[one.dx == "mel"].frac_lesion_inpainted.values, one[one.dx == "nv"].frac_lesion_inpainted.values
    U = mannwhitneyu(a, b)
    delta = 2 * U.statistic / (len(a) * len(b)) - 1
    T = pd.DataFrame([dict(n_lesions=len(one), kruskal_p=H.pvalue, epsilon2=eps2, n_mel=len(a), n_nv=len(b), mw_p_mel_nv=U.pvalue, cliffs_delta_mel_vs_nv=delta)])
    T.to_csv(OUT / "hair_lesion_tests.csv", index=False)
    print(S.round(4).to_string(), "\n", T.round(5).to_string(), flush=True)


def figs():
    setup_style()
    FIG.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(OUT / "image_level_per_image.csv")
    S = pd.read_csv(OUT / "image_level_summary.csv")
    # Hình A1: phân phối chỉ số mục tiêu trước/sau trên nhóm có vấn đề
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.6))
    for a, (p, (metric, way, methods, lab)) in zip(ax.ravel(), TARGETS.items()):
        s = D[(D.problem == p) & (D.group == "problem")]
        b = s[s.method == methods[0]].before.dropna().sort_values()
        a.plot(b.values, np.linspace(0, 1, len(b)), color="#555555", lw=1.8, label="trước xử lý")
        for c, m in zip(METHOD_COLORS, methods):
            v = s[s.method == m].after.dropna().sort_values()
            r = S[(S.problem == p) & (S.method == m) & (S.group == "problem")].iloc[0]
            res = f", hết cờ {r.resolved_rate:.0%}" if pd.notna(r.get("resolved_rate")) else ""
            a.plot(v.values, np.linspace(0, 1, len(v)), color=c, lw=1.4, label=f"{m}{res}")
        thr = S[(S.problem == p) & (S.group == "problem")].threshold.iloc[0]
        if pd.notna(thr):
            a.axvline(thr, color="k", ls=":", lw=0.8)
        a.set_title(f"{NAMES[p]} (n = {len(b)})")
        a.set_xlabel(lab)
        a.set_ylabel("tỉ lệ ảnh tích lũy (ECDF)")
        a.legend(fontsize=6.5, loc="lower right")
        if p in ("hair", "vignette", "overexposed"):
            a.set_xscale("symlog", linthresh=0.01)
    fig.tight_layout()
    _save(fig, "fig_a1_target_metric_ecdf")
    # Hình A2: tác dụng phụ trên ảnh đối chứng
    C = S[S.group == "control"].copy()
    C["label"] = C.problem.map(NAMES) + "\n" + C.method
    C = C.sort_values(["problem", "method"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(C))
    ax[0].barh(y, C.pct_images_changed * 100, color="#0072B2")
    ax[0].set_yticks(y, C.label, fontsize=7)
    ax[0].set_xlabel("% ảnh đối chứng bị đổi (≥ 1 pixel)")
    ax[1].barh(y, C.median_dE76, color="#D55E00")
    ax[1].set_xlabel("ΔE76 trung vị trên vùng nhìn thấy (ảnh đối chứng)")
    ax[0].invert_yaxis()
    fig.tight_layout()
    _save(fig, "fig_a2_control_side_effects")
    # Hình A3: lông che vùng tổn thương theo lớp
    if (OUT / "hair_lesion_per_image.csv").exists():
        H = pd.read_csv(OUT / "hair_lesion_per_image.csv")
        fig, a = plt.subplots(figsize=(5.4, 3))
        for i, c in enumerate(CLASSES):
            v = H[H.dx == c].frac_lesion_inpainted.values * 100
            a.boxplot(v, positions=[i], widths=0.6, showfliers=False, patch_artist=True,
                      boxprops=dict(facecolor=CLASS_COLORS[c], alpha=0.7), medianprops=dict(color="k"))
        a.set_xticks(range(len(CLASSES)), [f"{c}\n(n={int((H.dx == c).sum())})" for c in CLASSES])
        a.set_ylabel("% pixel tổn thương bị inpaint\n(hair_bh17, train_fit)")
        fig.tight_layout()
        _save(fig, "fig_a3_hair_mask_in_lesion_by_class")
    # Hình A4: lưới trước/sau/hiệu. Quy tắc: nhóm cờ train_fit, ảnh có chỉ số gần trung vị nhất và ảnh cực trị.
    fig, ax = plt.subplots(len(TARGETS), 6, figsize=(10, 1.6 * len(TARGETS)))
    tf = pd.read_csv("outputs/splits/train_fit.csv").set_index("image_id")
    for i, (p, m) in enumerate(PRIMARY.items()):
        metric, way, _, _ = TARGETS[p]
        s = D[(D.problem == p) & (D.group == "problem") & (D.method == m)].dropna(subset=["before"]).set_index("image_id")
        med = (s.before - s.before.median()).abs().idxmin()
        ext = s.before.idxmax() if way == "lower" else s.before.idxmin()
        for k, iid in enumerate([med, ext]):
            rgb = read_rgb(tf.loc[iid, "path"])
            proc = get_preprocess(m)(rgb)
            dd = np.abs(proc.astype(int) - rgb.astype(int)).max(2)
            ax[i, 3 * k].imshow(rgb)
            ax[i, 3 * k + 1].imshow(proc)
            ax[i, 3 * k + 2].imshow(dd, cmap="magma", vmin=0, vmax=max(20, int(np.percentile(dd, 99.5))))
            tag = "gần trung vị" if k == 0 else "cực trị"
            ax[i, 3 * k].set_title(f"{tag}: {s.loc[iid, 'before']:.3g}", fontsize=7)
            ax[i, 3 * k + 1].set_title(m, fontsize=7)
            ax[i, 3 * k + 2].set_title("|sau − trước|", fontsize=7)
        ax[i, 0].set_ylabel(NAMES[p], fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([]), a.grid(False)
    fig.tight_layout()
    _save(fig, "fig_a4_before_after_diff_grid", raster=True)


def _save(fig, name, raster=False):
    if not raster:
        fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "hair_lesion", "figs"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.jobs, a.limit)
    elif a.cmd == "hair_lesion":
        hair_lesion(a.jobs)
    else:
        figs()
