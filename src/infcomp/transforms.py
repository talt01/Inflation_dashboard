"""All transforms live here, once (brief §3.3). Input: month-end indexed series."""
from __future__ import annotations

import numpy as np
import pandas as pd

_FREQ_PERIODS = {"M": 12, "Q": 4}


def to_monthly(s: pd.Series, agg: str = "mean", native_freq: str | None = None) -> pd.Series:
    """Collapse any frequency to month-end.

    daily -> mean (default) / last / rvol (annualised std of daily changes, bp)
    monthly -> as is; quarterly -> kept at quarter-end then forward-filled 3 months
    (the reading stays "current" until the next release; limit=3 covers a full
    quarter so a single missing/late print doesn't leave a mid-history NaN hole).
    Forward-fill only, never interpolation: holding the last *published* value
    flat is exactly what was knowable in real time; interpolating between
    quarter-ends would pull a future release into the interior months.
    """
    s = s.dropna().sort_index()
    f = (native_freq or "").upper()[:1]
    if f == "Q":
        q = s.resample("QE").last()
        idx = pd.date_range(q.index.min(), q.index.max() + pd.offsets.MonthEnd(2), freq="ME")
        return q.reindex(idx).ffill(limit=3)
    if agg == "rvol":
        d = s.diff() * 100  # yield pct-pts -> bp
        return d.resample("ME").std() * np.sqrt(252)
    if agg == "last":
        return s.resample("ME").last()
    return s.resample("ME").mean()


def yoy(s: pd.Series, native_freq: str = "M") -> pd.Series:
    """YoY % on the month-end grid (quarterly data were ffilled, so 12 steps works)."""
    return (s / s.shift(12) - 1.0) * 100.0


def ann3m(s: pd.Series) -> pd.Series:
    return ((s / s.shift(3)) ** 4 - 1.0) * 100.0


def ann_to_12m(s: pd.Series) -> pd.Series:
    """Monthly annualised % changes -> approx YoY via 12m rolling mean."""
    return s.rolling(12, min_periods=12).mean()


def apply(s: pd.Series, transform: str, native_freq: str = "M") -> pd.Series:
    if transform == "yoy":
        return yoy(s, native_freq)
    if transform == "ann3m":
        return ann3m(s)
    if transform == "ann_to_12m":
        return ann_to_12m(s)
    if transform == "level":
        return s
    raise ValueError(f"Unknown/unresolved transform: {transform}")
