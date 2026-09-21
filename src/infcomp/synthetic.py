"""SYNTHETIC panel for pipeline smoke-tests only. Contains no real data.
Every figure produced from it is watermarked. Never feed this to the engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

WATERMARK = "SYNTHETIC DATA — PIPELINE TEST ONLY"


def make(n_months: int = 360, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2026-08-31", periods=n_months, freq="ME")
    common = np.cumsum(rng.normal(0, 0.25, n_months)); common -= common.mean()
    groups = {"headline": 4, "pipeline": 2, "underlying": 3, "components": 5,
              "expectations": 2, "market": 2, "wages": 1}
    cols, meta = {}, {}
    for g, k in groups.items():
        gshock = np.cumsum(rng.normal(0, 0.12, n_months))
        for j in range(k):
            key = f"syn_{g}_{j+1}"
            beta = rng.uniform(0.4, 1.2)
            cols[key] = 2.5 + beta * common + 0.6 * gshock + rng.normal(0, 0.3, n_months)
            meta[key] = {"name": f"Synthetic {g} {j+1}", "group": g,
                         "role": "orthogonal" if g in ("market", "wages") else "core", "sign": 1}
    inputs = pd.DataFrame(cols, index=idx)
    targets = pd.DataFrame({
        "tgt_be5": 2.2 + 0.3 * common + np.cumsum(rng.normal(0, 0.05, n_months)),
        "tgt_ust2": 3.0 + 0.5 * common + np.cumsum(rng.normal(0, 0.1, n_months)),
        "tgt_ust10": np.abs(80 + 10 * common + rng.normal(0, 8, n_months)),
        "tgt_fedfunds": 3.0 + 0.5 * common + np.cumsum(rng.normal(0, 0.1, n_months)),
    }, index=idx)
    return inputs, targets, pd.DataFrame(meta).T


def make_benchmarks(idx: pd.DatetimeIndex, seed: int = 11) -> pd.DataFrame:
    """Synthetic 3M/6M/1Y curve for market-benchmark pipeline/validation tests only.
    Not linked to `make()`'s common factor — it only needs to exercise
    validation.market_policy_change(), not to resemble a real yield curve."""
    rng = np.random.default_rng(seed)
    level = 3.0 + np.cumsum(rng.normal(0, 0.08, len(idx)))
    slope = rng.normal(0, 0.05, len(idx))
    return pd.DataFrame({"bmk_dgs3mo": level, "bmk_dgs6mo": level + slope,
                         "bmk_dgs1": level + 2 * slope}, index=idx)
