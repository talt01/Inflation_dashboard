"""Data layer (brief §3.2): verified map -> transformed monthly panel.

Only rows passing `verify.active_rows` are fetched. Returns
  inputs     : DataFrame of transformed composite inputs (roles core + orthogonal)
  targets    : DataFrame of raw monthly target series (role target)
  benchmarks : DataFrame of raw monthly benchmark-curve series (role benchmark);
               never transformed/z-scored, never a composite input
  meta       : per-indicator metadata used downstream (name, group, sign, lag)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import transforms as T
from .verify import active_rows, load_map


def _load_manual(key: str, manual_dir: str) -> pd.Series:
    df = pd.read_csv(Path(manual_dir) / f"{key}.csv", parse_dates=["date"])
    return df.set_index("date")["value"].astype(float).rename(key)


def build_panel(client, verified: pd.DataFrame, map_path="config/series_map.yaml",
                manual_dir="data/manual", realtime_end: str | None = None,
                point_in_time_lag: bool = False):
    spec = {d["key"]: d for d in load_map(map_path)}
    act = active_rows(verified, manual_dir)

    inputs, targets, benchmarks, meta = {}, {}, {}, {}
    for _, row in act.iterrows():
        d = spec[row["key"]]
        if row["source"] == "fred":
            raw = client.observations(row["series_id"], realtime_end=realtime_end)
        else:
            raw = _load_manual(row["key"], manual_dir)
        freq = (row.get("frequency") or "Monthly")
        m = T.to_monthly(raw, agg=d.get("agg", "mean"), native_freq=freq)

        if d["role"] == "target":
            targets[row["key"]] = m
            continue
        if d["role"] == "benchmark":
            benchmarks[row["key"]] = m           # raw level, no z, no sign flip
            continue

        x = T.apply(m, row["resolved_transform"], freq) * d.get("sign", 1)
        if point_in_time_lag:
            # value for reference month t is only knowable at t + pub_lag
            x = x.shift(d.get("pub_lag_m", 0))
        inputs[row["key"]] = x
        meta[row["key"]] = {"name": d["name"], "group": d["group"],
                            "role": d["role"], "sign": d.get("sign", 1)}

    inputs = pd.DataFrame(inputs).sort_index()
    targets = pd.DataFrame(targets).sort_index()
    benchmarks = pd.DataFrame(benchmarks).sort_index()
    return inputs, targets, benchmarks, pd.DataFrame(meta).T
