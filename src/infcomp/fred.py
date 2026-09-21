"""Minimal FRED / ALFRED client with a vintage-keyed local cache.

Designed to be swapped for (or wrapped around) the TYCCLES ingestion module:
anything exposing `series_info(id)` and `observations(id, realtime_end=None)`
works with the rest of the package.

Revision policy (brief §3.2)
----------------------------
* Default = LATEST vintage (what FRED shows today). Fine for the dashboard.
* For validation, pass `realtime_end` to pull a point-in-time vintage from
  ALFRED. Data revise (PCE especially), so a backtest on latest-vintage data
  overstates what was knowable. `data.build_panel(..., realtime_end=...)`
  threads this through; the validator records which policy was used.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # picks up FRED_API_KEY from a git-ignored .env at the repo root, if present

BASE = "https://api.stlouisfed.org/fred"


class FredError(RuntimeError):
    pass


class FredClient:
    def __init__(self, api_key: str | None = None, cache_dir: str | Path = "data/cache",
                 timeout: int = 30):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise FredError(
                "FRED_API_KEY not set. Get a free key at fredaccount.stlouisfed.org, then "
                "either `export FRED_API_KEY=...` or add it to a .env file at the repo root "
                "(FRED_API_KEY=..., already git-ignored).")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    # ------------------------------------------------------------------ raw
    def _get(self, endpoint: str, **params) -> dict:
        params.update(api_key=self.api_key, file_type="json")
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=self.timeout)
        if r.status_code == 400:
            # FRED returns 400 for a non-existent series ID
            raise FredError(r.json().get("error_message", "Bad request"))
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ public
    def series_info(self, series_id: str) -> dict:
        """Metadata for one series. Raises FredError if the ID does not exist."""
        js = self._get("series", series_id=series_id)
        seriess = js.get("seriess") or []
        if not seriess:
            raise FredError(f"No series returned for {series_id}")
        return seriess[0]

    def observations(self, series_id: str, realtime_end: str | None = None,
                     use_cache: bool = True) -> pd.Series:
        vintage = realtime_end or dt.date.today().isoformat()
        path = self.cache / f"{series_id}__{vintage}.csv"
        if use_cache and path.exists():
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = series_id
            return s

        params = {"series_id": series_id}
        if realtime_end:
            params.update(realtime_start=realtime_end, realtime_end=realtime_end)
        js = self._get("series/observations", **params)
        obs = pd.DataFrame(js["observations"])
        s = pd.to_numeric(obs["value"].replace(".", pd.NA), errors="coerce")
        s.index = pd.to_datetime(obs["date"])
        s.name = series_id
        s = s.dropna()
        s.to_frame().to_csv(path)
        return s
