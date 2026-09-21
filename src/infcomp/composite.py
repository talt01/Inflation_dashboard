"""Composite construction (brief §2, §3.5).

Production: expanding-window z-score per indicator, then equal-weight average.
  * Expanding (not full-sample) stats: each month's z uses only history up to
    that month, so the historical series is free of look-ahead and does not
    rewrite itself when new data arrive (full-sample z-scores do).
  * Contribution_i = z_i / N_available  ->  contributions sum exactly to the composite.

Cross-check: 0-5 Level / Trend / Momentum, continuity with Heatmap_indicators_v10.
  !! The workbook's exact bucket formulas are not in the brief. The version
  here is expanding-percentile based and must be aligned with the workbook
  before the two are compared as like-for-like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- production
def expanding_z(df: pd.DataFrame, min_periods: int = 60, clip: float = 4.0,
                window: int | None = None) -> pd.DataFrame:
    """window=None -> expanding; int -> rolling N months (still look-ahead free).
    Expanding z on series whose history includes the 1970s-80s will read a 3%
    CPI print as 'cool'. That is a regime-anchoring choice, not a bug: see
    settings.yaml `zscore.window`."""
    roll = df.rolling(window, min_periods=min_periods) if window else \
        df.expanding(min_periods=min_periods)
    mu, sd = roll.mean(), roll.std()
    return ((df - mu) / sd).clip(-clip, clip)


def zscore_composite(z: pd.DataFrame, min_coverage: float = 0.6):
    n_avail = z.notna().sum(axis=1)
    ok = n_avail >= np.ceil(min_coverage * z.shape[1])
    comp = z.mean(axis=1).where(ok)
    contrib = z.div(n_avail, axis=0).where(ok, axis=0)
    return comp.rename("composite_z"), contrib, n_avail.rename("n_indicators")


def diffusion(z: pd.DataFrame, lookback_m: int = 3) -> pd.DataFrame:
    """Breadth. heating_breadth = % of indicators whose z rose over lookback;
    above_avg = % with z > 0. 50 = balanced."""
    chg = z.diff(lookback_m)
    valid = chg.notna().sum(axis=1).replace(0, np.nan)
    heat = (chg > 0).sum(axis=1) / valid * 100
    above = (z > 0).sum(axis=1) / z.notna().sum(axis=1).replace(0, np.nan) * 100
    return pd.DataFrame({"heating_breadth": heat, "above_avg": above})


def ranked_contributions(z: pd.DataFrame, contrib: pd.DataFrame, meta: pd.DataFrame,
                         as_of=None, neutral_band: float = 0.5) -> pd.DataFrame:
    """The triage table: which indicators are driving the reading right now."""
    as_of = as_of or contrib.dropna(how="all").index[-1]
    tbl = pd.DataFrame({
        "indicator": meta["name"],
        "group": meta["group"],
        "z": z.loc[as_of],
        "z_3m_chg": z.diff(3).loc[as_of],
        "contribution": contrib.loc[as_of],
    }).dropna(subset=["contribution"])
    tbl["share_of_abs"] = tbl["contribution"].abs() / tbl["contribution"].abs().sum()
    # colour convention (duration lens): cooling = supportive, heating = detracting
    tbl["signal"] = np.select([tbl["z"] >= neutral_band, tbl["z"] <= -neutral_band],
                              ["heating", "cooling"], "neutral")
    tbl = tbl.sort_values("contribution", key=np.abs, ascending=False)
    tbl.index.name = "key"
    tbl.attrs["as_of"] = as_of
    return tbl


def concentration_flags(tbl: pd.DataFrame, top_n: int = 2, max_share: float = 0.5) -> dict:
    """Brief §4: no single input carries the result."""
    top = tbl["share_of_abs"].head(top_n).sum()
    return {"top_n": top_n, "top_share": float(top), "carried_by_few": bool(top > max_share),
            "top_keys": list(tbl.index[:top_n])}


# ----------------------------------------------------------------- 0-5 cross-check
def _expanding_pct_rank(s: pd.Series, min_periods: int) -> pd.Series:
    vals = s.to_numpy()
    out = np.full(len(vals), np.nan)
    hist: list[float] = []
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        hist.append(v)
        if len(hist) >= min_periods:
            out[i] = (np.asarray(hist) <= v).mean()
    return pd.Series(out, index=s.index)


def ltm_scores(df: pd.DataFrame, trend_m: int = 12, momentum_m: int = 3,
               min_periods: int = 60) -> dict[str, pd.DataFrame]:
    L = df.apply(_expanding_pct_rank, min_periods=min_periods) * 5
    Tr = df.diff(trend_m).apply(_expanding_pct_rank, min_periods=min_periods) * 5
    M = df.diff(momentum_m).apply(_expanding_pct_rank, min_periods=min_periods) * 5
    return {"level": L, "trend": Tr, "momentum": M, "ltm": (L + Tr + M) / 3}


def ltm_composite(ltm: pd.DataFrame, overrides: list[dict] | None = None,
                  cap: float = 1.25) -> pd.DataFrame:
    model = ltm.mean(axis=1)
    adj = pd.Series(0.0, index=model.index)
    for o in overrides or []:
        ts = pd.Timestamp(o["as_of"]) + pd.offsets.MonthEnd(0)
        if ts in adj.index:
            adj.loc[ts] = float(np.clip(o["value"], -cap, cap))
    return pd.DataFrame({"ltm_model": model, "manual_adj": adj,
                         "ltm_final": (model + adj).clip(0, 5)})
