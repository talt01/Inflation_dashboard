"""Fetchers for off-FRED sources distributed as hosted spreadsheets (`source: url`
in series_map.yaml — e.g. the SF Fed cyclical/acyclical core PCE decomposition,
the NY Fed Global Supply Chain Pressure Index).

Unlike `data/manual/<key>.csv` drops (`source: file`), these are pulled live from
the publisher's URL at verify/build/validate time — no local cache, so a stale
read never silently serves old data. If the host is unreachable or the sheet's
columns don't match the configured `date_col`/`value_col`, this raises loudly
with the actual columns found, the same "never guess, show what's really there"
rule verify.py already applies to FRED title mismatches.
"""
from __future__ import annotations

import io

import pandas as pd
import requests


class ExternalFetchError(RuntimeError):
    pass


def fetch_xlsx_series(url: str, date_col: str, value_col: str,
                      sheet: str | int = 0, skiprows: int = 0,
                      timeout: int = 30) -> pd.Series:
    """Download a hosted .xlsx and return one column as a date-indexed Series."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise ExternalFetchError(f"could not fetch {url}: {e}") from e
    try:
        df = pd.read_excel(io.BytesIO(r.content), sheet_name=sheet, skiprows=skiprows)
    except Exception as e:  # noqa: BLE001  (bad/unexpected xlsx structure)
        raise ExternalFetchError(f"could not parse {url} (sheet={sheet!r}) as xlsx: {e}") from e
    missing = [c for c in (date_col, value_col) if c not in df.columns]
    if missing:
        raise ExternalFetchError(
            f"{url} (sheet={sheet!r}): column(s) {missing} not found; "
            f"sheet actually has {list(df.columns)}")
    s = df.set_index(date_col)[value_col]
    s.index = pd.to_datetime(s.index)
    return pd.to_numeric(s, errors="coerce").dropna().sort_index().rename(value_col)
