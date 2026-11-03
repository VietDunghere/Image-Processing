"""Thống kê tối thiểu cho EDA: 1 ảnh/lesion, ngưỡng Tukey (box plot), Kruskal–Wallis + ε², Holm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42


def one_per_lesion(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Lấy ngẫu nhiên (seed cố định) 1 ảnh / lesion_id: các ảnh cùng lesion không độc lập."""
    return (df.sample(frac=1, random_state=seed).drop_duplicates("lesion_id")
              .sort_values("image_id").reset_index(drop=True))


def tukey_fence(x, k: float = 1.5, side: str = "high", log: bool = False) -> float:
    """Ngưỡng ngoại lai của box plot: Q3 + 1.5·IQR (high) hoặc Q1 − 1.5·IQR (low).
    log=True: tính trên log10(x) rồi đổi ngược – cho chỉ số lệch phải mạnh (vd. phương sai Laplacian)."""
    x = np.asarray(x, float)
    z = np.log10(x[x > 0]) if log else x
    q1, q3 = np.nanpercentile(z, [25, 75])
    f = q3 + k * (q3 - q1) if side == "high" else q1 - k * (q3 - q1)
    return float(10 ** f) if log else float(f)


def holm(pvals) -> np.ndarray:
    """Hiệu chỉnh Holm–Bonferroni khi làm nhiều kiểm định cùng lúc."""
    p = np.asarray(pvals, float)
    order, m = np.argsort(p), len(p)
    adj, run = np.empty(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def kruskal_eps2(df: pd.DataFrame, metric: str, group: str, min_n: int = 10) -> dict:
    """Kruskal–Wallis (so trung vị nhiều nhóm, không giả định phân phối chuẩn) + effect size
    ε² = H / (n − 1): 0.01 nhỏ, 0.08 vừa, 0.26 lớn. Bỏ nhóm < min_n ảnh và nhóm 'unknown'."""
    d = df[[metric, group]].dropna()
    d = d[d[group].astype(str).str.lower() != "unknown"]
    vc = d[group].value_counts()
    d = d[d[group].isin(vc[vc >= min_n].index)]
    groups = [g[metric].values for _, g in d.groupby(group)]
    if len(groups) < 2:
        return dict(H=np.nan, p=np.nan, eps2=np.nan, n=len(d))
    H, p = stats.kruskal(*groups)
    return dict(H=H, p=p, eps2=H / (len(d) - 1), n=len(d))


def eps2_magnitude(e: float) -> str:
    return "không đáng kể" if e < 0.01 else "nhỏ" if e < 0.08 else "vừa" if e < 0.26 else "lớn"
