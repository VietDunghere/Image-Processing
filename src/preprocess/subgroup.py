"""So sánh trước/sau tiền xử lý TRÊN CHÍNH nhóm ảnh có vấn đề (mục 2.5, bổ sung cho compare.py).

Câu hỏi: macro-F1 toàn tập không đổi rõ, nhưng trên ảnh bị nhiễu / tương phản thấp thì phương pháp có sửa được lỗi không?

Thiết kế (đặt TRƯỚC khi xem kết quả, không đổi theo số liệu):
  - Nhóm = cờ của EDA (outputs/eda/image_flags.csv: F_noise, F_low_contrast), KHÔNG tạo nhóm/ngưỡng mới.
  - Đối chứng = phần còn lại của cùng split, cùng cấu hình. Hiệu ứng "riêng cho nhóm" = Δ(nhóm) − Δ(còn lại)
    (difference-in-differences, DiD), tính trên cùng mẫu bootstrap.
  - Chỉ số trên nhóm con chỉ tính các lớp CÓ ảnh thật trong nhóm (lớp vắng không đóng góp F1/recall = 0 giả).
  - Δ ghép cặp theo seed (cùng seed khởi tạo/thứ tự batch); CI95 bootstrap cụm theo lesion_id (1000 lần),
    rút lại lesion của TOÀN split rồi lọc ra nhóm/còn lại → cùng mẫu cho Δ nhóm, Δ còn lại và DiD.
  - Quy tắc kết luận: "cải thiện rõ trên nhóm" chỉ khi Δ > 0, CI95 không chứa 0, 3/3 seed cùng dấu dương và
    |Δ| > std giữa seed của baseline trên chính nhóm đó. "Riêng cho nhóm" cần thêm CI95 của DiD không chứa 0.
    Đối xứng cho "tệ hơn rõ". Ngược lại: "không khác biệt rõ".
  - Test là kết quả chính; val chỉ để kiểm tra tính nhất quán (val đã dùng chọn epoch → lạc quan hơn).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from scipy.stats import binomtest, mannwhitneyu, wilcoxon

from src.eda.plots import CLASS_COLORS, CLASSES, setup_style

from . import get_preprocess
from .compare import SEEDS, _cluster_boot_idx, run_dir

OUT = Path("outputs/preprocess/subgroup")
FIG_DIR = Path("outputs/figures/preprocess/subgroup")
COMBO = "median3+clahe_L_c2_t8"
GROUPS = {   # nhóm → (cờ EDA, các phương pháp so với baseline)
    "noise": dict(flag="F_noise", label="nhóm nhiễu", methods=["median3", COMBO]),
    "low_contrast": dict(flag="F_low_contrast", label="nhóm tương phản thấp", methods=["clahe_L_c2_t8", COMBO]),
}
PARTS = {"group": "nhóm", "rest": "còn lại (đối chứng)"}
CFG_COLORS = {"baseline": "#7A7A7A", "median3": "#0072B2", "clahe_L_c2_t8": "#009E73", COMBO: "#CC79A7"}
K = len(CLASSES)
METRICS3 = ["accuracy", "macro_f1", "balanced_acc"]


def savefig(fig, name, raster=False):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not raster:
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    return FIG_DIR / f"{name}.png"


# ---------------------------------------------------------------- dữ liệu
def load_runs(name, split, seeds=SEEDS, order=None):
    """Dự đoán của 1 cấu hình trên 1 split, xếp theo `order` (image_id) để ghép cặp ảnh giữa các cấu hình.
    Trả về dict: ids, lesion, dx, y (N,), pred (S,N), ptrue (S,N), prob (S,N,7)."""
    pcols = [f"prob_{c}" for c in CLASSES]
    dfs = [pd.read_csv(run_dir(name) / f"seed{s}" / f"{split}_preds.csv").set_index("image_id") for s in seeds]
    order = list(dfs[0].index) if order is None else list(order)
    dfs = [d.loc[order] for d in dfs]
    y = dfs[0]["label"].to_numpy()
    prob = np.stack([d[pcols].to_numpy(float) for d in dfs])
    pred = np.stack([d["pred"].to_numpy() for d in dfs])
    assert (np.stack([d["label"].to_numpy() for d in dfs]) == y).all()
    assert (prob.argmax(-1) == pred).all(), "pred phải = argmax prob"
    return dict(ids=np.array(order), lesion=dfs[0]["lesion_id"].to_numpy(), dx=dfs[0]["dx"].to_numpy(), y=y,
                pred=pred, prob=prob, ptrue=np.take_along_axis(prob, y[None, :, None], -1)[..., 0])


def load_all(split, configs=None):
    configs = configs or ["baseline", "median3", "clahe_L_c2_t8", COMBO]
    R = {"baseline": load_runs("baseline", split)}
    for c in configs:
        if c != "baseline":
            R[c] = load_runs(c, split, order=R["baseline"]["ids"])
    return R


def membership(ids, flags):
    """Mask nhóm cho từng group theo thứ tự ids."""
    f = flags.set_index("image_id").loc[ids]
    return {g: f[v["flag"]].to_numpy(bool) for g, v in GROUPS.items()}


def composition(splits_R, flags, meta):
    """Cỡ mẫu + thành phần lớp + tỉ lệ dx_type từng nhóm/phần trên từng split."""
    dxt = meta.set_index("image_id")["dx_type"]
    rows = []
    for split, R in splits_R.items():
        b = R["baseline"]
        M = membership(b["ids"], flags)
        for g, m in M.items():
            for part, mm in (("group", m), ("rest", ~m)):
                ids = b["ids"][mm]
                rec = dict(group=g, part=part, split=split, n_images=int(mm.sum()),
                           n_lesions=len(np.unique(b["lesion"][mm])))
                for c in CLASSES:
                    rec[f"n_{c}"] = int((b["dx"][mm] == c).sum())
                vc = dxt.loc[ids].value_counts(normalize=True)
                for t in ("histo", "follow_up", "consensus", "confocal"):
                    rec[f"frac_{t}"] = float(vc.get(t, 0.0))
                rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- chỉ số trên nhóm con
def _cm(y, p):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def metrics_from_cm(C):
    """Chỉ tính các lớp có ảnh thật trong tập con (hàng C khác 0)."""
    present = C.sum(1) > 0
    tp = np.diag(C).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(present, tp / C.sum(1), np.nan)
        prec = np.where(C.sum(0) > 0, tp / C.sum(0), 0.0)
        f1 = np.where(present, np.nan_to_num(2 * prec * rec / (prec + rec)), np.nan)
    prec = np.where(present, prec, np.nan)
    out = dict(accuracy=tp.sum() / C.sum(), macro_f1=np.nanmean(f1), balanced_acc=np.nanmean(rec),
               n_classes_present=int(present.sum()))
    for i, c in enumerate(CLASSES):
        out[f"recall_{c}"], out[f"precision_{c}"], out[f"f1_{c}"] = rec[i], prec[i], f1[i]
    return out


def per_seed_metrics(R, M, split):
    rows = []
    for g, spec in GROUPS.items():
        for part, mm in (("group", M[g]), ("rest", ~M[g])):
            for cfg in ["baseline"] + spec["methods"]:
                r = R[cfg]
                for i, s in enumerate(SEEDS):
                    rows.append(dict(group=g, part=part, split=split, config=cfg, seed=s, n_images=int(mm.sum()),
                                     **metrics_from_cm(_cm(r["y"][mm], r["pred"][i][mm]))))
    return pd.DataFrame(rows)


def _metric3(y, P, idx):
    """[(acc, macro_f1, bacc) cho từng seed] trên tập ảnh idx. P: (S,N)."""
    out = []
    for p in P:
        m = metrics_from_cm(_cm(y[idx], p[idx]))
        out.append([m[k] for k in METRICS3])
    return np.array(out)   # (S, 3)


def boot_deltas(R, M, n_boot=1000, seed=42):
    """Bootstrap cụm lesion trên TOÀN split; mỗi lần tính Δ (TB 3 seed) cho nhóm, còn lại và DiD.
    Trả về dict (group, method) → mảng (n_boot, 2 phần, 3 chỉ số)."""
    b = R["baseline"]
    y = b["y"]
    res = {(g, mth): [] for g, sp in GROUPS.items() for mth in sp["methods"]}
    for idx in _cluster_boot_idx(b["lesion"], n_boot, seed):
        cache = {}
        for g, sp in GROUPS.items():
            for part, mm in (("group", M[g]), ("rest", ~M[g])):
                sub = idx[mm[idx]]
                for cfg in ["baseline"] + sp["methods"]:
                    if (g, part, cfg) not in cache:
                        cache[(g, part, cfg)] = _metric3(y, R[cfg]["pred"], sub)
            for mth in sp["methods"]:
                d = [np.nanmean(cache[(g, part, mth)] - cache[(g, part, "baseline")], axis=0) for part in ("group", "rest")]
                res[(g, mth)].append(np.stack(d))
    return {k: np.array(v) for k, v in res.items()}


def summary_table(PS, boots, split):
    """mean ± std qua seed, Δ ghép cặp theo seed, CI95 bootstrap cụm lesion, DiD, verdict."""
    rows = []
    metric_cols = METRICS3 + [f"{k}_{c}" for k in ("recall", "precision", "f1") for c in CLASSES]
    P = PS[PS.split == split]
    for g, sp in GROUPS.items():
        for part in ("group", "rest"):
            base = P[(P.group == g) & (P.part == part) & (P.config == "baseline")].set_index("seed").loc[list(SEEDS)]
            for cfg in ["baseline"] + sp["methods"]:
                S = P[(P.group == g) & (P.part == part) & (P.config == cfg)].set_index("seed").loc[list(SEEDS)]
                for m in metric_cols:
                    v = S[m].to_numpy(float)
                    if np.isnan(v).all():
                        continue
                    rec = dict(group=g, part=part, split=split, config=cfg, metric=m, n_images=int(S.n_images.iloc[0]),
                               mean=v.mean(), std=v.std(ddof=1), seed42=v[0], seed43=v[1], seed44=v[2],
                               baseline_seed_std=base[m].std(ddof=1))
                    if cfg != "baseline":
                        d = v - base[m].to_numpy(float)
                        rec.update(delta_mean=d.mean(), delta_std=d.std(ddof=1),
                                   delta_sign=f"{(d > 0).sum()}+/{(d < 0).sum()}-")
                        if m in METRICS3:
                            B = boots[(g, cfg)][:, 0 if part == "group" else 1, METRICS3.index(m)]
                            rec["ci_lo"], rec["ci_hi"] = np.nanpercentile(B, [2.5, 97.5])
                            if part == "group":
                                D = boots[(g, cfg)][:, 0, METRICS3.index(m)] - boots[(g, cfg)][:, 1, METRICS3.index(m)]
                                rest_d = P[(P.group == g) & (P.part == "rest") & (P.config == cfg)].set_index("seed").loc[list(SEEDS), m].to_numpy() \
                                    - P[(P.group == g) & (P.part == "rest") & (P.config == "baseline")].set_index("seed").loc[list(SEEDS), m].to_numpy()
                                rec["did_mean"] = d.mean() - rest_d.mean()
                                rec["did_lo"], rec["did_hi"] = np.nanpercentile(D, [2.5, 97.5])
                            rec["verdict"] = verdict(rec)
                    rows.append(rec)
    return pd.DataFrame(rows)


def verdict(r):
    dm, lo, hi = r["delta_mean"], r["ci_lo"], r["ci_hi"]
    clear = abs(dm) > r["baseline_seed_std"] and (lo > 0 or hi < 0) and r["delta_sign"] in ("3+/0-", "0+/3-")
    if not clear:
        return "không khác biệt rõ"
    v = "tốt hơn rõ" if dm > 0 else "tệ hơn rõ"
    if r.get("part") == "group" and "did_lo" in r:
        v += "; riêng cho nhóm" if (r["did_lo"] > 0 or r["did_hi"] < 0) else "; KHÔNG riêng (DiD chứa 0)"
    return v


# ---------------------------------------------------------------- confusion matrix
def confusion_long(R, M, split):
    rows = []
    for g, sp in GROUPS.items():
        m = M[g]
        for cfg in ["baseline"] + sp["methods"]:
            C = sum(_cm(R[cfg]["y"][m], p[m]) for p in R[cfg]["pred"])
            for i, t in enumerate(CLASSES):
                for j, pr in enumerate(CLASSES):
                    rows.append(dict(group=g, split=split, config=cfg, true=t, pred=pr, count_sum_3seeds=int(C[i, j]),
                                     row_frac=C[i, j] / C[i].sum() if C[i].sum() else np.nan))
    return pd.DataFrame(rows)


def _cm_from_long(CL, g, split, cfg):
    d = CL[(CL.group == g) & (CL.split == split) & (CL.config == cfg)]
    return d.pivot(index="true", columns="pred", values="count_sum_3seeds").loc[CLASSES, CLASSES].to_numpy()


def fig_confusion(CL, g, split):
    """Hàng = phương pháp; cột = baseline | phương pháp | hiệu (phương pháp − baseline). Số đếm cộng 3 seed + % theo hàng."""
    setup_style()
    methods = GROUPS[g]["methods"]
    Cb = _cm_from_long(CL, g, split, "baseline")
    present = Cb.sum(1) > 0
    n_img = int(Cb.sum() / len(SEEDS))
    fig, ax = plt.subplots(len(methods), 3, figsize=(12, 3.9 * len(methods)))
    for r, mth in enumerate(methods):
        Cm = _cm_from_long(CL, g, split, mth)
        for c, (C, title) in enumerate([(Cb, "baseline"), (Cm, mth), (Cm - Cb, f"hiệu: {mth} − baseline")]):
            a = ax[r, c]
            rs = np.where(present[:, None], Cb.sum(1, keepdims=True), 1)
            frac = C / rs
            lim = max(np.nanmax(np.abs(frac[present])), 1e-3)
            if c < 2:
                a.imshow(np.where(present[:, None], frac, np.nan), cmap="Blues", vmin=0, vmax=1)
            else:
                a.imshow(np.where(present[:, None], frac, np.nan), cmap="PuOr_r", norm=TwoSlopeNorm(0, -lim, lim))
            for i in range(K):
                for j in range(K):
                    if not present[i] or (C[i, j] == 0 and c == 2):
                        continue
                    if C[i, j] == 0:
                        continue
                    txt = f"{C[i, j]:+d}\n{frac[i, j]:+.0%}" if c == 2 else f"{C[i, j]}\n{frac[i, j]:.0%}"
                    dark = frac[i, j] > 0.55 if c < 2 else abs(frac[i, j]) > 0.6 * lim
                    a.text(j, i, txt, ha="center", va="center", fontsize=6, color="w" if dark else "k")
            a.set_xticks(range(K), CLASSES, rotation=45)
            a.set_yticks(range(K), [f"{cl} (n={int(Cb[i].sum() / len(SEEDS))})" for i, cl in enumerate(CLASSES)])
            for t, cl in zip(a.get_xticklabels(), CLASSES):
                t.set_color(CLASS_COLORS[cl])
            for t, cl in zip(a.get_yticklabels(), CLASSES):
                t.set_color(CLASS_COLORS[cl])
            a.grid(False)
            a.set_xlabel("lớp dự đoán")
            if c == 0:
                a.set_ylabel("lớp thật (n ảnh/seed)")
            a.set_title(title)
    fig.suptitle(f"{split}: {GROUPS[g]['label']} (n = {n_img} ảnh × 3 seed; ô = số đếm cộng 3 seed và % theo hàng)", y=1.0)
    fig.tight_layout()
    savefig(fig, f"fig_s2_cm_{g}_{split}")
    return fig


# ---------------------------------------------------------------- chuyển trạng thái từng ảnh
def _trans(cb, cm):
    return dict(CC=int((cb & cm).sum()), WC=int((~cb & cm).sum()), CW=int((cb & ~cm).sum()), WW=int((~cb & ~cm).sum()))


def mcnemar_exact(b, c):
    return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue


def transitions(R, M, split, mask_override=None):
    """Bảng đúng/sai trước→sau, từng seed (cùng seed), đa số phiếu 3 seed; thêm cặp seed của chính baseline
    (42→43, 43→44, 44→42) làm mức dao động ngẫu nhiên. WC = sai→đúng (được sửa), CW = đúng→sai (bị làm hỏng)."""
    rows = []
    b = R["baseline"]
    Cb = b["pred"] == b["y"]
    for g, sp in GROUPS.items():
        parts = (("group", M[g]), ("rest", ~M[g])) if mask_override is None else (("sample100", mask_override[g]),)
        for part, mm in parts:
            for mth in sp["methods"]:
                Cm = R[mth]["pred"] == R[mth]["y"]
                for i, s in enumerate(SEEDS):
                    rows.append(dict(group=g, part=part, split=split, config=mth, compare=f"seed{s}", **_trans(Cb[i][mm], Cm[i][mm])))
                rows.append(dict(group=g, part=part, split=split, config=mth, compare="đa số 3 seed",
                                 **_trans((Cb.sum(0) >= 2)[mm], (Cm.sum(0) >= 2)[mm])))
            for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                rows.append(dict(group=g, part=part, split=split, config="baseline (cặp seed)",
                                 compare=f"seed{SEEDS[a_]}→seed{SEEDS[b_]}", **_trans(Cb[a_][mm], Cb[b_][mm])))
    T = pd.DataFrame(rows)
    T["n"] = T[["CC", "WC", "CW", "WW"]].sum(1)
    T["net_fixed"] = T["WC"] - T["CW"]
    T["flip_rate"] = (T["WC"] + T["CW"]) / T["n"]
    T["mcnemar_p"] = [mcnemar_exact(r.WC, r.CW) for r in T.itertuples()]
    return T


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def consistency(R, M, split):
    """Tách thay đổi NHẤT QUÁN (cả 3 seed) khỏi dao động: n_fix3 = sai ở cả 3 seed baseline và đúng ở cả 3 seed
    phương pháp; n_break3 ngược lại. Tham chiếu: tỉ lệ ảnh baseline không ổn định (đúng ở 1–2/3 seed)."""
    b = R["baseline"]
    Cb = b["pred"] == b["y"]
    rows = []
    for g, sp in GROUPS.items():
        for part, mm in (("group", M[g]), ("rest", ~M[g])):
            nb = Cb[:, mm].sum(0)
            pair_flip = np.mean([(Cb[a][mm] != Cb[c][mm]).mean() for a, c in combinations(range(3), 2)])
            for mth in sp["methods"]:
                Cm = R[mth]["pred"][:, mm] == R[mth]["y"][mm]
                nm = Cm.sum(0)
                n_fix = ((~Cb[:, mm]) & Cm).sum(0)
                n_brk = (Cb[:, mm] & ~Cm).sum(0)
                rows.append(dict(group=g, part=part, split=split, config=mth, n=int(mm.sum()),
                                 baseline_unstable=int(((nb > 0) & (nb < 3)).sum()),
                                 method_unstable=int(((nm > 0) & (nm < 3)).sum()),
                                 baseline_seedpair_flip_rate=pair_flip,
                                 method_vs_baseline_flip_rate=np.mean([(Cb[i][mm] != Cm[i]).mean() for i in range(3)]),
                                 fix_3of3=int(((nb == 0) & (nm == 3)).sum()), break_3of3=int(((nb == 3) & (nm == 0)).sum()),
                                 fix_ge2=int((n_fix >= 2).sum()), break_ge2=int((n_brk >= 2).sum()),
                                 fix_any=int((n_fix >= 1).sum()), break_any=int((n_brk >= 1).sum())))
    D = pd.DataFrame(rows)
    D["net_3of3"] = D["fix_3of3"] - D["break_3of3"]
    return D


# ---------------------------------------------------------------- Δ xác suất lớp đúng
def dprob_per_image(R, M, split):
    b = R["baseline"]
    rows = []
    for g, sp in GROUPS.items():
        for mth in sp["methods"]:
            d = R[mth]["ptrue"] - b["ptrue"]   # (S,N), cùng seed
            for part, mm in (("group", M[g]), ("rest", ~M[g])):
                rows.append(pd.DataFrame(dict(group=g, part=part, split=split, config=mth, image_id=b["ids"][mm],
                                              lesion_id=b["lesion"][mm], dx=b["dx"][mm],
                                              ptrue_base=b["ptrue"][:, mm].mean(0), ptrue_method=R[mth]["ptrue"][:, mm].mean(0),
                                              dprob=d[:, mm].mean(0),
                                              **{f"dprob_seed{s}": d[i, mm] for i, s in enumerate(SEEDS)})))
        # mức dao động: Δ giữa các cặp seed của chính baseline (giá trị tuyệt đối, TB 3 cặp)
        sp_abs = np.mean([np.abs(b["ptrue"][a] - b["ptrue"][c]) for a, c in combinations(range(3), 2)], axis=0)
        for part, mm in (("group", M[g]), ("rest", ~M[g])):
            rows.append(pd.DataFrame(dict(group=g, part=part, split=split, config="baseline (cặp seed)",
                                          image_id=b["ids"][mm], lesion_id=b["lesion"][mm], dx=b["dx"][mm],
                                          ptrue_base=b["ptrue"][:, mm].mean(0), ptrue_method=np.nan, dprob=np.nan,
                                          abs_seedpair=sp_abs[mm])))
    return pd.concat(rows, ignore_index=True)


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))


def dprob_tests(DP):
    """Wilcoxon ghép cặp (H0: trung vị Δ prob_true = 0) ở mức lesion (TB ảnh cùng lesion → quan sát độc lập hơn)
    và mức ảnh; Mann–Whitney + Cliff's δ giữa Δ của nhóm và Δ của phần còn lại (mức lesion)."""
    rows = []
    ref = DP[DP.config == "baseline (cặp seed)"]
    for (g, split, cfg), d in DP[DP.config != "baseline (cặp seed)"].groupby(["group", "split", "config"], sort=False):
        lv = {p: d[d.part == p].groupby("lesion_id")["dprob"].mean().to_numpy() for p in ("group", "rest")}
        for part in ("group", "rest"):
            x = d[d.part == part]
            r = ref[(ref.group == g) & (ref.split == split) & (ref.part == part)]
            rec = dict(group=g, split=split, config=cfg, part=part, n_images=len(x), n_lesions=len(lv[part]),
                       median_dprob=x.dprob.median(), mean_dprob=x.dprob.mean(), frac_pos=(x.dprob > 0).mean(),
                       median_abs_dprob=x.dprob.abs().median(),
                       median_abs_seedpair_baseline=r.abs_seedpair.median(),
                       wilcoxon_p_lesion=wilcoxon(lv[part]).pvalue if np.any(lv[part] != 0) else np.nan,
                       wilcoxon_p_image=wilcoxon(x.dprob).pvalue if np.any(x.dprob != 0) else np.nan)
            if part == "group":
                rec["cliffs_delta_group_vs_rest"] = cliffs_delta(lv["group"], lv["rest"])
                rec["mwu_p_group_vs_rest"] = mannwhitneyu(lv["group"], lv["rest"]).pvalue
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hình
def fig_deltas(SUM, split):
    """Δ (phương pháp − baseline) của accuracy, macro-F1, balanced acc trên nhóm vs còn lại; CI95 + chấm từng seed."""
    setup_style()
    fig, ax = plt.subplots(len(GROUPS), 3, figsize=(11, 2.4 * len(GROUPS) + 0.6), sharex=False)
    for r, (g, sp) in enumerate(GROUPS.items()):
        for c, m in enumerate(METRICS3):
            a = ax[r, c]
            ylab = []
            for k, mth in enumerate(sp["methods"]):
                for j, part in enumerate(("group", "rest")):
                    s = SUM[(SUM.split == split) & (SUM.group == g) & (SUM.part == part) & (SUM.config == mth) & (SUM.metric == m)].iloc[0]
                    yy = k * 2.6 + j
                    col = "#D55E00" if part == "group" else "#7A7A7A"
                    a.errorbar(s.delta_mean, yy, xerr=[[s.delta_mean - s.ci_lo], [s.ci_hi - s.delta_mean]], fmt="o",
                               color=col, ms=4, capsize=2)
                    base = SUM[(SUM.split == split) & (SUM.group == g) & (SUM.part == part) & (SUM.config == "baseline") & (SUM.metric == m)].iloc[0]
                    a.scatter([s[f"seed{x}"] - base[f"seed{x}"] for x in SEEDS], [yy] * 3, marker="|", s=40, color="k", zorder=4)
                    a.fill_betweenx([yy - 0.4, yy + 0.4], -s.baseline_seed_std, s.baseline_seed_std, color=col, alpha=0.12, lw=0)
                    ylab.append((yy, f"{mth}\n{PARTS[part].split(' ')[0]} (n={s.n_images})"))
            a.axvline(0, color="k", lw=0.8)
            a.set_yticks([t for t, _ in ylab], [l for _, l in ylab] if c == 0 else [""] * len(ylab), fontsize=6.5)
            a.invert_yaxis()
            a.set_title(f"{sp['label']}: Δ {m}")
            a.set_xlabel(f"Δ {m} (phương pháp − baseline)")
    fig.suptitle(f"{split}: Δ ghép cặp theo seed (chấm tròn = TB 3 seed, thanh = CI95 bootstrap cụm lesion, vạch đen = từng seed;"
                 f" nền mờ = ±std giữa seed của baseline)", fontsize=8, y=1.0)
    fig.tight_layout()
    savefig(fig, f"fig_s1_subgroup_delta_{split}")
    return fig


def fig_transitions(TR, split):
    """Số ảnh được sửa (WC) và bị làm hỏng (CW) trên 100 ảnh, theo từng seed; so với cặp seed của baseline."""
    setup_style()
    fig, ax = plt.subplots(1, len(GROUPS), figsize=(11, 3.4))
    for a, (g, sp) in zip(ax, GROUPS.items()):
        labels, x = [], 0
        for part in ("group", "rest"):
            d = TR[(TR.split == split) & (TR.group == g) & (TR.part == part)]
            for cfg in sp["methods"] + ["baseline (cặp seed)"]:
                e = d[(d.config == cfg) & (d["compare"] != "đa số 3 seed")]
                wc, cw = 100 * e.WC / e.n, 100 * e.CW / e.n
                col = "#7A7A7A" if cfg.startswith("baseline") else CFG_COLORS[cfg]
                a.bar(x - 0.18, wc.mean(), 0.34, color=col, alpha=0.85, edgecolor="k", lw=0.4)
                a.bar(x + 0.18, cw.mean(), 0.34, color=col, alpha=0.35, edgecolor="k", lw=0.4, hatch="///")
                a.scatter(np.full(3, x - 0.18), wc, s=8, color="k", zorder=3)
                a.scatter(np.full(3, x + 0.18), cw, s=8, color="k", zorder=3)
                labels.append(f"{cfg}\n{PARTS[part].split(' ')[0]} (n={int(e.n.iloc[0])})")
                x += 1
            x += 0.5
        a.set_xticks([i + (0.5 if i >= len(sp['methods']) + 1 else 0) for i in range(len(labels))], labels, rotation=35, ha="right", fontsize=6.5)
        a.set_ylabel("số ảnh / 100 ảnh (TB 3 seed)")
        a.set_title(f"{split}: {sp['label']}")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(facecolor="#BBBBBB", edgecolor="k", label="sai→đúng (được sửa)"),
                          Patch(facecolor="#EEEEEE", edgecolor="k", hatch="///", label="đúng→sai (bị làm hỏng)")], fontsize=7)
    fig.tight_layout()
    savefig(fig, f"fig_s3_transitions_{split}")
    return fig


def fig_dprob(DP, DT, split):
    """ECDF của Δ prob_true theo ảnh (TB 3 seed), nhóm vs còn lại, cho từng phương pháp."""
    setup_style()
    fig, ax = plt.subplots(1, len(GROUPS) * 2, figsize=(12, 3.0), sharey=True)
    k = 0
    for g, sp in GROUPS.items():
        for mth in sp["methods"]:
            a = ax[k]
            k += 1
            for part, col in (("group", "#D55E00"), ("rest", "#7A7A7A")):
                x = np.sort(DP[(DP.split == split) & (DP.group == g) & (DP.config == mth) & (DP.part == part)].dprob)
                t = DT[(DT.split == split) & (DT.group == g) & (DT.config == mth) & (DT.part == part)].iloc[0]
                a.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col,
                       label=f"{PARTS[part].split(' ')[0]} n={len(x)}: trung vị {t.median_dprob:+.3f}, p_W={t.wilcoxon_p_lesion:.2g}")
            a.axvline(0, color="k", lw=0.7)
            a.set_xscale("symlog", linthresh=0.01)
            a.set_xlabel("Δ p(lớp đúng) = sau − trước")
            t = DT[(DT.split == split) & (DT.group == g) & (DT.config == mth) & (DT.part == "group")].iloc[0]
            a.set_title(f"{sp['label']}: {mth}\nCliff δ nhóm vs còn lại = {t.cliffs_delta_group_vs_rest:+.2f}", fontsize=8)
            a.legend(fontsize=5.5, loc="upper left")
    ax[0].set_ylabel("ECDF (tỉ lệ ảnh)")
    fig.tight_layout()
    savefig(fig, f"fig_s4_dprob_ecdf_{split}")
    return fig


# ---------------------------------------------------------------- 100 ảnh
def sample100(flags, R_test, R_val, n=100, seed=42):
    """100 ảnh ngẫu nhiên (seed 42) của mỗi nhóm trên test; thiếu thì lấy hết test rồi bổ sung ngẫu nhiên từ val."""
    rows = []
    for g in GROUPS:
        rng = np.random.default_rng(seed)
        mt = membership(R_test["baseline"]["ids"], flags)[g]
        ids_t = np.sort(R_test["baseline"]["ids"][mt])
        if len(ids_t) >= n:
            pick = [(i, "test") for i in rng.choice(ids_t, n, replace=False)]
        else:
            mv = membership(R_val["baseline"]["ids"], flags)[g]
            ids_v = np.sort(R_val["baseline"]["ids"][mv])
            pick = [(i, "test") for i in ids_t] + [(i, "val") for i in rng.choice(ids_v, n - len(ids_t), replace=False)]
        rows += [dict(group=g, image_id=i, split=s, order=k) for k, (i, s) in enumerate(pick)]
    return pd.DataFrame(rows)


def sample_detail(S100, R_by_split, paths):
    """Chi tiết từng ảnh trong mẫu 100: dự đoán/xác suất lớp đúng của baseline và từng phương pháp ở từng seed."""
    out = []
    for r in S100.itertuples():
        R = R_by_split[r.split]
        i = int(np.flatnonzero(R["baseline"]["ids"] == r.image_id)[0])
        rec = dict(group=r.group, order=r.order, image_id=r.image_id, split=r.split, lesion_id=R["baseline"]["lesion"][i],
                   dx=R["baseline"]["dx"][i], path=paths[r.image_id])
        for cfg in ["baseline"] + GROUPS[r.group]["methods"]:
            for k, s in enumerate(SEEDS):
                rec[f"{cfg}|pred_s{s}"] = CLASSES[R[cfg]["pred"][k, i]]
                rec[f"{cfg}|ptrue_s{s}"] = R[cfg]["ptrue"][k, i]
            rec[f"{cfg}|n_correct"] = int((R[cfg]["pred"][:, i] == R[cfg]["y"][i]).sum())
        out.append(rec)
    return pd.DataFrame(out)


def sample_transitions(SD):
    rows = []
    for g, sp in GROUPS.items():
        d = SD[SD.group == g]
        for mth in sp["methods"]:
            for s in SEEDS:
                cb = (d[f"baseline|pred_s{s}"] == d.dx).to_numpy()
                cm = (d[f"{mth}|pred_s{s}"] == d.dx).to_numpy()
                rows.append(dict(group=g, config=mth, compare=f"seed{s}", **_trans(cb, cm)))
            rows.append(dict(group=g, config=mth, compare="đa số 3 seed",
                             **_trans((d["baseline|n_correct"] >= 2).to_numpy(), (d[f"{mth}|n_correct"] >= 2).to_numpy())))
            rows.append(dict(group=g, config=mth, compare="nhất quán 3/3", CC=np.nan, WW=np.nan,
                             WC=int(((d["baseline|n_correct"] == 0) & (d[f"{mth}|n_correct"] == 3)).sum()),
                             CW=int(((d["baseline|n_correct"] == 3) & (d[f"{mth}|n_correct"] == 0)).sum())))
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            cb = (d[f"baseline|pred_s{SEEDS[a_]}"] == d.dx).to_numpy()
            cm = (d[f"baseline|pred_s{SEEDS[b_]}"] == d.dx).to_numpy()
            rows.append(dict(group=g, config="baseline (cặp seed)", compare=f"seed{SEEDS[a_]}→seed{SEEDS[b_]}", **_trans(cb, cm)))
    T = pd.DataFrame(rows)
    T["n_split"] = [len(SD[SD.group == g]) for g in T.group]
    T["net_fixed"] = T.WC - T.CW
    T["mcnemar_p"] = [mcnemar_exact(int(r.WC), int(r.CW)) if r.compare != "nhất quán 3/3" else np.nan for r in T.itertuples()]
    return T


def _majority(d, cfg):
    preds = d[[f"{cfg}|pred_s{s}" for s in SEEDS]].to_numpy()
    out = []
    for row in preds:
        v, c = np.unique(row, return_counts=True)
        out.append(v[c.argmax()] if c.max() >= 2 else "/".join(row))
    return out


def grid_examples(SD, g, mth, kmax=6, seed=42, thumb=(240, 180), amp=4):
    """Lưới cân bằng: k ảnh sai→đúng nhất quán (3/3 seed) bên trái, k ảnh đúng→sai nhất quán bên phải,
    k = min(kmax, #sửa, #hỏng), chọn ngẫu nhiên seed 42. Mỗi ảnh: trước | sau | |sau − trước| × amp."""
    setup_style()
    d = SD[SD.group == g].copy()
    d["maj_base"], d["maj_m"] = _majority(d, "baseline"), _majority(d, mth)
    d["p_base"] = d[[f"baseline|ptrue_s{s}" for s in SEEDS]].mean(1)
    d["p_m"] = d[[f"{mth}|ptrue_s{s}" for s in SEEDS]].mean(1)
    fix = d[(d["baseline|n_correct"] == 0) & (d[f"{mth}|n_correct"] == 3)]
    brk = d[(d["baseline|n_correct"] == 3) & (d[f"{mth}|n_correct"] == 0)]
    k = min(kmax, len(fix), len(brk))
    rng = np.random.default_rng(seed)
    if k == 0:   # một phía rỗng: vẫn hiện phía còn lại (≤ 3 ảnh) để thấy ví dụ, caption phải nói rõ bất cân xứng
        k_f, k_b = min(3, len(fix)), min(3, len(brk))
    else:
        k_f = k_b = k
    sel_f = fix.iloc[np.sort(rng.choice(len(fix), k_f, replace=False))] if k_f else fix.iloc[:0]
    sel_b = brk.iloc[np.sort(rng.choice(len(brk), k_b, replace=False))] if k_b else brk.iloc[:0]
    info = dict(group=g, config=mth, n_fix3=len(fix), n_break3=len(brk), shown_fix=k_f, shown_break=k_b,
                shown_ids=",".join(list(sel_f.image_id) + list(sel_b.image_id)))
    if k_f + k_b == 0:   # không có ảnh đổi trạng thái nhất quán nào → không vẽ hình rỗng
        return None, info
    nrow = max(k_f, k_b)
    fn = get_preprocess(mth)
    fig, ax = plt.subplots(nrow, 6, figsize=(12, 2.05 * nrow + 0.5), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for side, sel in ((0, sel_f), (1, sel_b)):
        for r, (_, row) in enumerate(sel.iterrows()):
            with Image.open(row.path) as im:
                rgb = np.asarray(im.convert("RGB"))
            after = fn(rgb)
            diff = np.clip(np.abs(after.astype(int) - rgb.astype(int)).mean(-1) * amp, 0, 255).astype(np.uint8)
            ims = [Image.fromarray(rgb).resize(thumb), Image.fromarray(after).resize(thumb), Image.fromarray(diff).resize(thumb)]
            for c, (img, lab) in enumerate(zip(ims, ["trước", "sau", f"|sau−trước|×{amp}"])):
                a = ax[r, side * 3 + c]
                a.imshow(img, cmap="gray" if c == 2 else None, vmin=0, vmax=255)
                a.set_title(lab, fontsize=6, pad=1)
            ax[r, side * 3].text(0, -0.32, f"{row.image_id} ({row.split}) | thật: {row.dx}", transform=ax[r, side * 3].transAxes,
                                 fontsize=6.5, color=CLASS_COLORS[row.dx], fontweight="bold")
            ax[r, side * 3 + 1].text(-0.3, -0.18, f"trước: {row.maj_base} (p_đúng={row.p_base:.2f}) → sau: {row.maj_m} (p_đúng={row.p_m:.2f})",
                                     transform=ax[r, side * 3 + 1].transAxes, fontsize=6)
    fig.suptitle(f"{GROUPS[g]['label']}, {mth}, mẫu 100 ảnh: trái = sai→đúng nhất quán (có {len(fix)}, hiện {k_f}); "
                 f"phải = đúng→sai nhất quán (có {len(brk)}, hiện {k_b}). Dự đoán = đa số 3 seed, p = TB 3 seed",
                 fontsize=7.5, y=1.0)
    fig.tight_layout(h_pad=1.6)
    savefig(fig, f"grid_s5_{g}_{mth.replace('+', '_plus_')}", raster=True)
    return fig, info
