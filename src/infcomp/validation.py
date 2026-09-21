"""Out-of-sample validation against an external target (brief §2 gate, §3.6).

Nothing here decides the threshold. It reads config/gate.yaml, refuses to run
if any field is unfilled, and applies the pre-committed rules mechanically.

No look-ahead:
  * every predictor is built from expanding-window statistics (z, percentiles,
    and PCA refit at each date on data up to that date);
  * the regression at date t uses only (x_s, y_s) pairs with s + h <= t, i.e.
    forward targets that had actually been realised by t;
  * the benchmark is selected by `config/gate.yaml: benchmark` — hist_mean
    (expanding historical mean of the target, Campbell-Thompson OOS R^2),
    zero (random-walk "no change" sanity check), or market (curve-implied
    expected policy move, see market_policy_change()). OOS R^2 > 0 means the
    composite beat that benchmark's own prediction.
Caveat recorded in output: overlapping h-month targets make observations
autocorrelated; OOS R^2 is still valid as a point estimate but naive standard
errors are not, so none are reported.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import composite as C

REQUIRED = ["target", "horizon_m", "metric", "pca_promotion_margin",
            "signal_min_oos_r2", "min_train_months", "committed_on", "committed_by"]


class GateNotCommitted(RuntimeError):
    pass


def load_gate(path="config/gate.yaml") -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    g = yaml.safe_load(raw)
    missing = [k for k in REQUIRED if g.get(k) in (None, "")]
    if missing:
        raise GateNotCommitted(
            f"Gate not pre-committed; fill {missing} in {path} and commit it to git "
            "BEFORE running validation.")
    return g, hashlib.sha256(raw).hexdigest()


def build_target(targets: pd.DataFrame, name: str, h: int) -> pd.Series:
    if name == "be5_change":
        s = targets["tgt_be5"]; return ((s.shift(-h) - s) * 100).rename(name)
    if name == "ust2_change":
        s = targets["tgt_ust2"]; return ((s.shift(-h) - s) * 100).rename(name)
    if name == "rate_vol":
        s = targets["tgt_ust10"]; return s.rolling(h).mean().shift(-h).rename(name)
    if name == "fedfunds_change":
        s = targets["tgt_fedfunds"]; return ((s.shift(-h) - s) * 100).rename(name)
    raise ValueError(f"unknown target {name}")


def market_policy_change(bmk: pd.DataFrame, h: int) -> pd.Series:
    """Curve-implied expected change in the policy rate over the next `h` months, in bp.
    Approximation, not fed-funds futures: forward short rate (h..2h) minus spot short rate.
    For h=6: f(6->12) = 2*DGS1 - DGS6MO ; spot = DGS3MO.  Expected change = f - spot.
    A proxy for the priced Fed path — good enough to be the consensus benchmark, not exact.
    Only supports h=6 (the two-point forward construction below is specific to that horizon)."""
    if h != 6:
        raise ValueError("market_policy_change only supports horizon_m=6 "
                          "(2*DGS1 - DGS6MO forward construction is horizon-specific)")
    spot = bmk["bmk_dgs3mo"]
    fwd = 2.0 * bmk["bmk_dgs1"] - bmk["bmk_dgs6mo"]        # linear forward approx
    return ((fwd - spot) * 100).rename("market_implied")  # bp, matches target units


def pca_pc1_expanding(z: pd.DataFrame, min_rows: int) -> pd.Series:
    """PC1 score at each t from a PCA fitted on complete rows up to t.
    Sign aligned so PC1 loads positively on the equal-weight average."""
    out = pd.Series(np.nan, index=z.index)
    for i, t in enumerate(z.index):
        hist = z.iloc[: i + 1].dropna()
        if len(hist) < min_rows:
            continue
        mu, sd = hist.mean(), hist.std().replace(0, np.nan)
        x = ((hist - mu) / sd).fillna(0)
        _, _, vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
        w = vt[0] if vt[0].sum() >= 0 else -vt[0]
        cur = ((z.loc[t] - mu) / sd).fillna(0)  # missing indicator -> neutral
        out.loc[t] = float(cur.to_numpy() @ w)
    return out


def oos_r2(x: pd.Series, y: pd.Series, h: int, min_train: int,
           benchmark: pd.Series | None = None) -> dict:
    """benchmark=None -> Campbell-Thompson (expanding historical mean of y).
    benchmark=Series -> the benchmark's own prediction of y at each t (e.g. the
    market-implied policy path, or a zero/"no change" series); OOS R^2 then
    measures edge over THAT prediction instead of over the historical mean."""
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    if benchmark is not None:
        df = df.join(benchmark.rename("bmk"))
    dates = df.index
    preds, bench, actual = [], [], []
    for i in range(len(dates)):
        if pd.isna(df["x"].iloc[i]) or pd.isna(df["y"].iloc[i]):
            continue
        train = df.iloc[: max(i - h + 1, 0)].dropna(subset=["x", "y"])   # s + h <= t
        if len(train) < min_train:
            continue
        # resolve the benchmark's prediction for this date FIRST so a missing
        # benchmark value skips the whole observation, keeping preds/bench/actual aligned
        if benchmark is not None:
            bp = df["bmk"].iloc[i]
            if pd.isna(bp):
                continue
            bench.append(bp)                 # market's own prediction of y at t
        else:
            bench.append(train["y"].mean())  # Campbell-Thompson historical mean
        b, a = np.polyfit(train["x"], train["y"], 1)
        preds.append(a + b * df["x"].iloc[i])
        actual.append(df["y"].iloc[i])
    if not actual:
        return {"oos_r2": np.nan, "n_oos": 0}
    p, bm, yv = map(np.asarray, (preds, bench, actual))
    return {"oos_r2": float(1 - ((yv - p) ** 2).sum() / ((yv - bm) ** 2).sum()),
            "rmse_ratio": float(np.sqrt(((yv - p) ** 2).mean() / ((yv - bm) ** 2).mean())),
            "n_oos": int(len(yv))}


def run(inputs: pd.DataFrame, targets: pd.DataFrame, settings: dict,
        gate_path="config/gate.yaml", vintage_policy="latest",
        benchmarks: pd.DataFrame | None = None) -> dict:
    gate, gate_hash = load_gate(gate_path)
    h, mt = int(gate["horizon_m"]), int(gate["min_train_months"])
    X = inputs.drop(columns=[c for c in gate.get("exclude_inputs", []) if c in inputs])
    zs = settings["zscore"]

    z = C.expanding_z(X, zs["min_periods"], zs["clip"], zs.get("window"))
    comp, _, _ = C.zscore_composite(z, settings["composite"]["min_coverage"])
    ltm = C.ltm_scores(X, settings["ltm"]["trend_m"], settings["ltm"]["momentum_m"],
                       zs["min_periods"])["ltm"].mean(axis=1)
    pc1 = pca_pc1_expanding(z, min_rows=zs["min_periods"])
    y = build_target(targets, gate["target"], h)

    bmode = gate.get("benchmark", "hist_mean")
    if bmode == "market":
        if benchmarks is None or benchmarks.empty:
            raise ValueError("gate benchmark: market requires the benchmark curve "
                             "(bmk_dgs3mo/6mo/1) — pass `benchmarks` from data.build_panel")
        bench_series = market_policy_change(benchmarks, h)
    elif bmode == "zero":
        bench_series = pd.Series(0.0, index=targets.index)
    else:
        bench_series = None   # hist_mean (Campbell-Thompson, original behaviour)

    res = {m: oos_r2(s, y, h, mt, benchmark=bench_series) for m, s in
           {"zscore_avg": comp, "ltm_0_5": ltm, "pca_pc1": pc1}.items()}

    margin = res["pca_pc1"]["oos_r2"] - res["zscore_avg"]["oos_r2"]
    promote = bool(margin >= gate["pca_promotion_margin"])
    production = "pca_pc1" if promote else "zscore_avg"
    label = "SIGNAL" if res[production]["oos_r2"] >= gate["signal_min_oos_r2"] else "DESCRIPTIVE"

    return {"run_at": dt.datetime.now().isoformat(timespec="seconds"),
            "gate": gate, "gate_sha256": gate_hash, "vintage_policy": vintage_policy,
            "benchmark": bmode, "inputs_used": list(X.columns), "results": res,
            "pca_minus_zavg": float(margin), "pca_promoted": promote,
            "production_method": production, "composite_label": label,
            "caveat": "Overlapping forward targets -> autocorrelated errors; no SEs reported."}


def save(result: dict, out_dir="outputs") -> Path:
    p = Path(out_dir) / "validation_result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, default=str))
    return p
