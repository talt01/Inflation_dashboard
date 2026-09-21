"""Task 1: build the verified series map.

For every `source: fred` row, query FRED live and record what the ID actually
resolves to. Status logic:

  VERIFIED      ID exists AND every `expect_title` substring appears in the title
  MISMATCH      ID exists but the title does not match -> almost certainly the
                wrong series; human review required, excluded from composite
  UNVERIFIED    ID not found on FRED (or call failed); excluded
  OFF_FRED      free source loaded from data/manual/<key>.csv
  PROPRIETARY   ISM / Conference Board / Truflation; excluded unless supplied

A VERIFIED row still needs you to set `confirmed` = TRUE in the output CSV
after eyeballing title/units/frequency. The composite reads only rows that are
VERIFIED + confirmed (or OFF_FRED/PROPRIETARY with a CSV present + confirmed).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .fred import FredError

COLUMNS = ["key", "indicator", "source", "series_id", "title", "frequency",
           "history_start", "last_obs", "units", "seasonal_adj", "transform",
           "resolved_transform", "role", "group", "status", "note", "confirmed"]


def load_map(path: str | Path = "config/series_map.yaml") -> list[dict]:
    return yaml.safe_load(Path(path).read_text())["indicators"]


def resolve_transform(transform: str, units: str | None, frequency: str | None) -> str:
    """Turn `auto` into a concrete transform using the VERIFIED units string."""
    if transform != "auto":
        return transform
    u = (units or "").lower()
    if u.startswith("index"):
        return "yoy"
    if "percent change at annual rate" in u:
        return "ann_to_12m"
    if "percent change from year ago" in u:
        return "level"
    if u.startswith("percent"):
        return "level"
    return "UNRESOLVED"


def verify(client, map_path="config/series_map.yaml", manual_dir="data/manual",
           previous: str | Path | None = None) -> pd.DataFrame:
    rows = []
    prior_conf = {}
    if previous and Path(previous).exists():
        p = pd.read_csv(previous)
        prior_conf = dict(zip(p["key"] + "|" + p["series_id"].fillna(""),
                              p["confirmed"].astype(str).str.upper() == "TRUE"))

    for ind in load_map(map_path):
        r = {c: None for c in COLUMNS}
        r.update(key=ind["key"], indicator=ind["name"], source=ind["source"],
                 series_id=ind.get("series_id"), transform=ind["transform"],
                 role=ind["role"], group=ind["group"], confirmed=False)

        if ind["source"] == "fred":
            try:
                info = client.series_info(ind["series_id"])
                title = info.get("title", "")
                r.update(title=title, frequency=info.get("frequency"),
                         history_start=info.get("observation_start"),
                         last_obs=info.get("observation_end"),
                         units=info.get("units"),
                         seasonal_adj=info.get("seasonal_adjustment_short"))
                missing = [k for k in ind.get("expect_title", [])
                           if k.lower() not in title.lower()]
                if missing:
                    r.update(status="MISMATCH",
                             note=f"title lacks {missing}; likely wrong series")
                else:
                    r["status"] = "VERIFIED"
                r["resolved_transform"] = resolve_transform(
                    ind["transform"], r["units"], r["frequency"])
                if r["resolved_transform"] == "UNRESOLVED":
                    r["note"] = (r["note"] or "") + " transform unresolved from units"
            except (FredError, Exception) as e:  # noqa: BLE001  (network, 4xx, parse)
                r.update(status="UNVERIFIED", note=str(e)[:160])
        else:
            csv = Path(manual_dir) / f"{ind['key']}.csv"
            r["status"] = "OFF_FRED" if ind["source"] == "file" else "PROPRIETARY"
            r["resolved_transform"] = ind["transform"]
            r["note"] = (f"CSV present: {csv}" if csv.exists()
                         else f"awaiting {csv} (columns: date,value)")

        # carry forward a prior human confirmation only if the ID is unchanged
        r["confirmed"] = prior_conf.get(f"{r['key']}|{r['series_id'] or ''}", False) \
            and r["status"] in ("VERIFIED", "OFF_FRED", "PROPRIETARY")
        rows.append(r)

    return pd.DataFrame(rows, columns=COLUMNS)


def active_rows(verified: pd.DataFrame, manual_dir="data/manual") -> pd.DataFrame:
    """Rows allowed into the model: verified/confirmed, or confirmed file with CSV."""
    v = verified.copy()
    v["confirmed"] = v["confirmed"].astype(str).str.upper() == "TRUE"
    fred_ok = (v["status"] == "VERIFIED") & v["confirmed"]
    file_ok = v["status"].isin(["OFF_FRED", "PROPRIETARY"]) & v["confirmed"] & \
        v["key"].map(lambda k: (Path(manual_dir) / f"{k}.csv").exists())
    return v[fred_ok | file_ok]
