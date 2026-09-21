"""Redundancy diagnostic (brief §2, §3.4). PCA here is a DIAGNOSTIC ONLY.

Outputs
  * correlation matrix of the standardised set
  * explained-variance share (how much of the set is one common factor)
  * near-duplicate pairs above a |corr| threshold -> candidates to cut
  * PC1 sign/loading stability across expanding windows -> quantifies the
    "loadings flip as data are added" objection instead of just asserting it
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pca(z: pd.DataFrame, min_rows: int = 36):
    """PCA on the complete-case window of a standardised panel."""
    x = z.dropna()
    if len(x) < min_rows:
        raise ValueError(f"Only {len(x)} complete rows; need {min_rows} for PCA.")
    x = (x - x.mean()) / x.std()
    u, s, vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
    var = s**2 / (s**2).sum()
    loadings = pd.DataFrame(vt.T, index=x.columns,
                            columns=[f"PC{i+1}" for i in range(len(s))])
    # sign convention: PC1 loads positively on the average (hotter = up)
    if loadings["PC1"].sum() < 0:
        loadings["PC1"] *= -1
    return {"explained": pd.Series(var, index=loadings.columns), "loadings": loadings,
            "window": (x.index.min(), x.index.max()), "n_rows": len(x)}


def near_duplicates(z: pd.DataFrame, thresh: float = 0.9) -> pd.DataFrame:
    c = z.corr()
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(c.columns)
             for b in c.columns[i + 1:] if abs(c.loc[a, b]) >= thresh]
    return pd.DataFrame(pairs, columns=["a", "b", "corr"]).sort_values(
        "corr", key=np.abs, ascending=False)


def pc1_stability(z: pd.DataFrame, start_rows: int = 60, step: int = 12) -> pd.DataFrame:
    """Refit PCA on expanding windows WITHOUT the sign convention and record
    raw PC1 loadings. Sign flips / rank changes = instability of PCA as a signal."""
    x_all = z.dropna()
    rec = {}
    for end in range(start_rows, len(x_all) + 1, step):
        x = x_all.iloc[:end]
        x = (x - x.mean()) / x.std()
        _, _, vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
        rec[x_all.index[end - 1]] = vt[0]
    out = pd.DataFrame(rec, index=x_all.columns).T
    return out


def stability_summary(stab: pd.DataFrame) -> dict:
    if stab.empty:
        return {}
    raw_sign = np.sign(stab.sum(axis=1))
    flips = int((raw_sign.diff().abs() > 0).sum())
    # after normalising sign, how often does an individual indicator's loading change sign?
    norm = stab.mul(raw_sign, axis=0)
    ind_flips = (np.sign(norm).diff().abs() > 0).sum().sort_values(ascending=False)
    return {"refits": len(stab), "raw_pc1_sign_flips": flips,
            "indicator_loading_flips": ind_flips[ind_flips > 0].to_dict()}


def run(z: pd.DataFrame, dup_thresh: float = 0.9) -> dict:
    p = pca(z)
    stab = pc1_stability(z)
    return {"corr": z.corr(), "pca": p, "duplicates": near_duplicates(z, dup_thresh),
            "stability": stab, "stability_summary": stability_summary(stab),
            "common_factor_share": float(p["explained"].iloc[0])}
