"""2.5c tầng D: Grad-CAM trước/sau. Mô hình nhìn vào đâu, và có đổi khi xử lý ảnh không?

Thực hiện Grad-CAM bằng hook tại model.layer4 (lưới 7×7), đích là LỚP ĐÚNG của ảnh để hai mô hình giải thích cùng một đích.
Ảnh: nhóm cờ tương ứng + đối chứng cùng cỡ (seed 42) trong val và test. Mỗi cấu hình dùng đúng ảnh đầu vào của nó
(cache 224 của 2.5); vùng đo (tổn thương, lông, viền) luôn lấy từ ảnh GỐC.

Đo:
  - energy_R = tổng CAM trong vùng R / tổng CAM (CAM đã upsample 224, chỉ giá trị dương);
  - tương đồng CAM: Pearson của bản đồ 7×7 giữa cấu hình và baseline khác seed, so với mức nền baseline–baseline khác seed.

    python -X utf8 scripts/impact_gradcam.py run [--limit N]
    python -X utf8 scripts/impact_gradcam.py analyze

Đầu ra: outputs/impact/gradcam_{per_image,summary,similarity}.csv, outputs/figures/impact/fig_d*.{pdf,png}
"""
from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eda.plots import setup_style  # noqa: E402
from src.eda.quality_metrics import masks  # noqa: E402
from src.preprocess import get_preprocess  # noqa: E402
from src.preprocess.compare import _cluster_boot_idx  # noqa: E402
from src.training import CKPT_DIR, IMAGENET_MEAN, IMAGENET_STD, load_image_cache, load_splits, read_rgb  # noqa: E402

OUT, FIG = Path("outputs/impact"), Path("outputs/figures/impact")
SEEDS = (42, 43, 44)
CONFIGS = {"hair_bh17": ("F_hair", "hair"), "hair_bh17_cond": ("F_hair", "hair"), "vig_fill": ("F_vignette", "vignette"),
           "gamma_L_auto": ("F_overexposed", None), "flat_L_s60_cond": ("F_uneven_illum", None), "median3": ("F_noise", None),
           "clahe_L_c2_t8": ("F_low_contrast", None)}
NAMES = {"hair": "vị trí lông", "vignette": "vùng viền đen"}
N_BOOT = 2000


def gradcam(model, x, y):
    feats = []
    h = model.layer4.register_forward_hook(lambda m, i, o: feats.append(o))
    logits = model(x)
    h.remove()
    g = torch.autograd.grad(logits.gather(1, y[:, None]).sum(), feats[0])[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * feats[0]).sum(1))
    return cam.detach().float().cpu().numpy()


def select(frame, flags, flag):
    F_ = flags.loc[frame.image_id, flag].to_numpy()
    p = frame[F_].assign(group="flagged")
    c = frame[~F_].sample(n=len(p), random_state=42).assign(group="control")
    return pd.concat([p, c], ignore_index=True)


def region_masks(path):
    """Mask vùng trên ảnh gốc → 224 (nearest). Lông giãn 5×5 vì CAM lưới 7×7 thô, sợi lông mảnh."""
    cv2.setNumThreads(1)
    g = cv2.cvtColor(read_rgb(path), cv2.COLOR_RGB2GRAY)
    M = masks(g)
    hair = cv2.dilate(M["hair"].astype(np.uint8), np.ones((5, 5), np.uint8))
    rs = lambda a: cv2.resize(a.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)  # noqa: E731
    return rs(M["lesion"]), rs(hair), rs(~M["fov"])


def compute_cams(model_name, X, y, device, model):
    out = {}
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255
    for s in SEEDS:
        model.load_state_dict(torch.load(CKPT_DIR / f"{model_name}_seed{s}.pt", map_location="cpu", weights_only=False)["state_dict"])
        model.eval()
        cams = []
        for i in range(0, len(y), 32):
            xb = (X[i:i + 32].to(device).float() - mean) / std
            cams.append(gradcam(model, xb, y[i:i + 32]))
        out[s] = np.concatenate(cams)
    return out


def energy(cam7, R):
    t = torch.from_numpy(cam7).float().unsqueeze(1)
    c = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False).squeeze(1).numpy()
    tot = c.sum((1, 2))
    out = {}
    for k, m in R.items():
        out[k] = np.where(tot > 0, (c * m).sum((1, 2)) / np.maximum(tot, 1e-9), np.nan)
    return out


def pearson(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    d = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(d > 0, (a * b).sum(1) / np.maximum(d, 1e-12), np.nan)


def run(limit):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gradcam").mkdir(exist_ok=True)
    device = torch.device("cuda")
    flags = pd.read_csv("outputs/eda/image_flags.csv").set_index("image_id")
    frames = load_splits()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device).eval()
    rec, sim = [], []
    for name, (flag, region) in CONFIGS.items():
        for split in ("val", "test"):
            S = select(frames[split], flags, flag)
            if limit:
                S = S.groupby("group").head(limit).reset_index(drop=True)
            with ThreadPoolExecutor(12) as ex:
                rm = list(ex.map(region_masks, S.path))
            R = {"lesion": np.stack([r[0] for r in rm]), "hair": np.stack([r[1] for r in rm]), "vignette": np.stack([r[2] for r in rm])}
            y = torch.as_tensor(S.label.values, dtype=torch.long, device=device)
            sub = {split: S[["image_id", "path"]]}
            Xb = load_image_cache(sub, None, "baseline", log=lambda *_: None)[split]
            Xc = load_image_cache(sub, get_preprocess(name), name, log=lambda *_: None)[split]
            Cb, Cc = compute_cams("baseline", Xb, y, device, model), compute_cams(name, Xc, y, device, model)
            np.savez_compressed(OUT / "gradcam" / f"{name}_{split}.npz", image_id=S.image_id.values, group=S.group.values,
                                **{f"b{s}": Cb[s].astype(np.float16) for s in SEEDS}, **{f"c{s}": Cc[s].astype(np.float16) for s in SEEDS})
            for cfg, C in (("baseline", Cb), (name, Cc)):
                for s in SEEDS:
                    E = energy(C[s], R)
                    rec.append(pd.DataFrame({"config_pair": name, "cfg": cfg, "split": split, "seed": s, "image_id": S.image_id, "lesion_id": S.lesion_id, "dx": S.dx, "group": S.group,
                                             "region": region or "lesion", "energy_lesion": E["lesion"], "energy_target": E[region] if region else np.nan,
                                             "area_lesion": R["lesion"].mean((1, 2)), "area_target": R[region].mean((1, 2)) if region else np.nan}))
            for a, b in ((s, t) for s in SEEDS for t in SEEDS if s != t):
                sim.append(pd.DataFrame({"config_pair": name, "split": split, "kind": "config_vs_baseline", "seed_a": a, "seed_b": b, "image_id": S.image_id, "lesion_id": S.lesion_id, "group": S.group, "r": pearson(Cc[a], Cb[b])}))
            for a, b in itertools.combinations(SEEDS, 2):
                sim.append(pd.DataFrame({"config_pair": name, "split": split, "kind": "floor", "seed_a": a, "seed_b": b, "image_id": S.image_id, "lesion_id": S.lesion_id, "group": S.group, "r": pearson(Cb[a], Cb[b])}))
            print(f"{name} {split}: n={len(S)}", flush=True)
    pd.concat(rec, ignore_index=True).to_csv(OUT / "gradcam_per_image.csv", index=False)
    pd.concat(sim, ignore_index=True).to_csv(OUT / "gradcam_similarity_per_image.csv", index=False)


def analyze():
    FIG.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(OUT / "gradcam_per_image.csv")
    Sim = pd.read_csv(OUT / "gradcam_similarity_per_image.csv")
    rows = []
    for (cp, split, grp), s in D.groupby(["config_pair", "split", "group"]):
        for metric in ("energy_lesion", "energy_target"):
            if s[metric].isna().all():
                continue
            w = s.pivot_table(index=["image_id", "lesion_id"], columns="cfg", values=metric, aggfunc="mean").dropna().reset_index()
            d = (w[cp] - w["baseline"]).to_numpy()
            les = w.lesion_id.values
            boots = [d[i].mean() for i in _cluster_boot_idx(les, N_BOOT)]
            per_seed = s.pivot_table(index=["image_id", "seed"], columns="cfg", values=metric).dropna()
            seed_delta = (per_seed[cp] - per_seed["baseline"]).groupby("seed").mean()
            rows.append(dict(config=cp, split=split, group=grp, metric=metric, region=s.region.iloc[0], n_images=len(w), n_lesions=len(set(les)),
                             baseline=w["baseline"].mean(), processed=w[cp].mean(), delta=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                             sign=f"{(seed_delta > 0).sum()}+/{(seed_delta < 0).sum()}-",
                             area_target=s[s.cfg == "baseline"].area_target.mean() if metric == "energy_target" else s[s.cfg == "baseline"].area_lesion.mean()))
    T = pd.DataFrame(rows)
    T["ci_excludes_0"] = (T.ci_lo > 0) | (T.ci_hi < 0)
    T.to_csv(OUT / "gradcam_summary.csv", index=False)
    rs = []
    for (cp, split, grp), s in Sim.groupby(["config_pair", "split", "group"]):
        a = s[s.kind == "config_vs_baseline"].groupby(["image_id", "lesion_id"]).r.mean().reset_index()
        f = s[s.kind == "floor"].groupby(["image_id", "lesion_id"]).r.mean().reset_index()
        m = a.merge(f, on=["image_id", "lesion_id"], suffixes=("_cfg", "_floor")).dropna()
        d = (m.r_cfg - m.r_floor).to_numpy()
        boots = [d[i].mean() for i in _cluster_boot_idx(m.lesion_id.values, N_BOOT)]
        rs.append(dict(config=cp, split=split, group=grp, n_images=len(m), r_config_vs_baseline=m.r_cfg.mean(), r_floor=m.r_floor.mean(), delta=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5)))
    Q = pd.DataFrame(rs)
    Q.to_csv(OUT / "gradcam_similarity.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    print(T.round(4).to_string(), "\n", Q.round(4).to_string())
    setup_style()
    # Hình D1: năng lượng CAM trên vùng mục tiêu, nhóm cờ
    F1 = T[(T.group == "flagged") & (((T.metric == "energy_target")) | (T.metric == "energy_lesion"))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, split in zip(ax, ("val", "test")):
        s = F1[F1.split == split].sort_values(["metric", "config"], ascending=[False, True])
        lab = [f"{r.config}\n{'tổn thương' if r.metric == 'energy_lesion' else NAMES.get(r.region, r.region)}" for r in s.itertuples()]
        y = np.arange(len(s))
        a.barh(y - 0.18, s.baseline * 100, 0.34, color="#555555", label="baseline")
        a.barh(y + 0.18, s.processed * 100, 0.34, color="#0072B2", label="sau xử lý")
        for i, r in enumerate(s.itertuples()):
            a.text(max(r.baseline, r.processed) * 100 + 1, i, f"Δ {r.delta * 100:+.1f} [{r.ci_lo * 100:+.1f}; {r.ci_hi * 100:+.1f}]", va="center", fontsize=6)
        a.set_yticks(y, lab, fontsize=6.5)
        a.invert_yaxis()
        a.set_xlabel("% năng lượng Grad-CAM trong vùng (TB 3 seed)")
        a.set_title(split)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d1_gradcam_energy.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_d1_gradcam_energy.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    # Hình D2: tương đồng CAM so với mức nền
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for a, split in zip(ax, ("val", "test")):
        s = Q[(Q.split == split) & (Q.group == "flagged")].sort_values("config")
        y = np.arange(len(s))
        a.barh(y, s.r_config_vs_baseline, color="#0072B2", alpha=0.75, label="cấu hình vs baseline (khác seed)")
        a.scatter(s.r_floor, y, marker="|", s=170, color="k", zorder=3, label="mức nền: baseline vs baseline")
        a.set_yticks(y, s.config)
        a.invert_yaxis()
        a.set_xlabel("Pearson r giữa bản đồ Grad-CAM 7×7")
        a.set_title(f"{split}: ảnh bị cờ")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_d2_gradcam_similarity.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_d2_gradcam_similarity.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    grid(D)


def overlay(rgb, cam):
    c = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    c = c / max(c.max(), 1e-9)
    hm = plt.cm.jet(c)[..., :3]
    return np.clip(0.55 * rgb / 255.0 + 0.45 * hm, 0, 1)


def grid(D):
    """Lưới: nhóm cờ tập test, quy tắc chọn = ảnh có energy_target của baseline gần trung vị nhất và cao nhất (hair, vig_fill).
    CAM là trung bình 3 seed để không phụ thuộc một lần huấn luyện."""
    frames = load_splits()["test"].set_index("image_id")
    cfgs = ["hair_bh17", "vig_fill"]
    fig, ax = plt.subplots(len(cfgs) * 2, 4, figsize=(8.6, 2.05 * len(cfgs) * 2))
    for i, name in enumerate(cfgs):
        z = np.load(OUT / "gradcam" / f"{name}_test.npz", allow_pickle=True)
        ids, grp = z["image_id"], z["group"]
        s = D[(D.config_pair == name) & (D.cfg == "baseline") & (D.split == "test") & (D.group == "flagged")].groupby("image_id").energy_target.mean()
        picks = [(s - s.median()).abs().idxmin(), s.idxmax()]
        for k, iid in enumerate(picks):
            j = np.flatnonzero(ids == iid)[0]
            rgb = read_rgb(frames.loc[iid, "path"])
            proc = get_preprocess(name)(rgb)
            cb, cc = (np.mean([z[f"{p}{sd}"][j].astype(np.float32) for sd in SEEDS], 0) for p in ("b", "c"))
            row = 2 * i + k
            ax[row, 0].imshow(rgb)
            ax[row, 0].set_title(f"{frames.loc[iid, 'dx']} | {'gần trung vị' if k == 0 else 'cao nhất'}", fontsize=7)
            ax[row, 1].imshow(overlay(rgb, cb))
            ax[row, 1].set_title("Grad-CAM baseline (TB 3 seed)", fontsize=7)
            ax[row, 2].imshow(proc)
            ax[row, 2].set_title(name, fontsize=7)
            ax[row, 3].imshow(overlay(proc, cc))
            ax[row, 3].set_title(f"Grad-CAM {name}", fontsize=7)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([]), a.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d3_gradcam_grid.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.limit) if a.cmd == "run" else analyze()
