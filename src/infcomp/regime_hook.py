"""Clean output for the top-level fusion layer (brief §3.8).

Every domain (inflation, labour/growth, credit, policy...) emits the same
schema, so cloning a domain = one more row in the fusion table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def emit(domain: str, comp: pd.Series, diff: pd.DataFrame, table: pd.DataFrame,
         conc: dict, ltm: pd.DataFrame | None, validation: dict | None,
         band: float = 0.25) -> dict:
    as_of = table.attrs["as_of"]
    c = comp.loc[:as_of]
    chg = {f"chg_{m}m": float(c.loc[as_of] - c.shift(m).loc[as_of])
           if len(c) > m and not np.isnan(c.shift(m).loc[as_of]) else None
           for m in (3, 6, 12)}
    d3 = chg["chg_3m"]
    traj = None if d3 is None else ("heating" if d3 > band else "cooling" if d3 < -band else "stable")
    return {
        "domain": domain,
        "as_of": str(pd.Timestamp(as_of).date()),
        "level_z": float(c.loc[as_of]),
        **chg,
        "trajectory": traj,
        "trajectory_band_note": f"|3m Δz| > {band} (a chosen parameter, not calibrated)",
        "diffusion_heating_pct": float(diff["heating_breadth"].loc[as_of]),
        "top_drivers": table.head(3)[["indicator", "z", "contribution"]]
                        .round(3).reset_index().to_dict("records"),
        "concentration": conc,
        "ltm_0_5": None if ltm is None else {k: round(float(v), 3) for k, v in ltm.loc[as_of].items()},
        # UNVALIDATED until validation.run() has been executed against a committed gate
        "status": (validation or {}).get("composite_label", "UNVALIDATED"),
        "production_method": (validation or {}).get("production_method", "zscore_avg"),
    }


def save(payload: dict, out_dir="outputs") -> Path:
    p = Path(out_dir) / f"{payload['domain']}_regime_hook.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p
