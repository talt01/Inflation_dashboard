"""CLI — runs brief §3 in order.

  python -m infcomp verify     # Task 1: live FRED check -> config/series_map_verified.csv
  python -m infcomp build      # data -> transforms -> diagnostics -> composite -> dashboard -> hook
  python -m infcomp validate   # OOS test vs the pre-committed gate (config/gate.yaml)
  python -m infcomp demo       # synthetic smoke test (watermarked, no FRED needed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import composite as C
from . import diagnostics as D
from . import regime_hook as R
from . import validation as V
from . import visuals as VIS

VERIFIED = Path("config/series_map_verified.csv")


def _settings():
    return yaml.safe_load(Path("config/settings.yaml").read_text())


def _client(s):
    from .fred import FredClient
    return FredClient(cache_dir=s["cache_dir"])


def _load_panel(s, pit=False, realtime_end=None):
    from .data import build_panel
    if not VERIFIED.exists():
        sys.exit("Run `python -m infcomp verify` first and confirm rows.")
    v = pd.read_csv(VERIFIED)
    return build_panel(_client(s), v, manual_dir=s["manual_dir"],
                       realtime_end=realtime_end, point_in_time_lag=pit)


def compute_and_render(inputs, targets, meta, s, out: Path, watermark=None, validation=None):
    out.mkdir(parents=True, exist_ok=True)
    zs, cs = s["zscore"], s["composite"]
    disp = inputs.ffill(limit=2)  # ragged edge: carry last print until next release
    z = C.expanding_z(disp, zs["min_periods"], zs["clip"], zs.get("window"))
    comp, contrib, n = C.zscore_composite(z, cs["min_coverage"])
    diff = C.diffusion(z, s["diffusion"]["lookback_m"])
    table = C.ranked_contributions(z, contrib, meta, neutral_band=cs["neutral_band"])
    conc = C.concentration_flags(table)
    ltm_parts = C.ltm_scores(disp, s["ltm"]["trend_m"], s["ltm"]["momentum_m"], zs["min_periods"])
    overrides = (yaml.safe_load(Path("config/overrides.yaml").read_text()) or {}).get("adjustments") \
        if Path("config/overrides.yaml").exists() else []
    ltm = C.ltm_composite(ltm_parts["ltm"], overrides)

    # diagnostics (PCA = diagnostic only)
    try:
        diag = D.run(z, s["diagnostics"]["duplicate_corr"])
        diag["corr"].round(3).to_csv(out / "diag_correlation.csv")
        diag["pca"]["loadings"].iloc[:, :3].round(3).to_csv(out / "diag_pca_loadings.csv")
        diag["duplicates"].round(3).to_csv(out / "diag_near_duplicates.csv", index=False)
        diag["stability"].round(3).to_csv(out / "diag_pc1_stability.csv")
        (out / "diag_summary.json").write_text(json.dumps({
            "common_factor_share_pc1": diag["common_factor_share"],
            "explained_top5": diag["pca"]["explained"].head(5).round(3).to_dict(),
            "pca_window": [str(d.date()) for d in diag["pca"]["window"]],
            "stability": diag["stability_summary"]}, indent=2, default=str))
    except ValueError as e:
        print(f"[diagnostics skipped] {e}")

    hist = pd.concat([comp, n, diff, ltm], axis=1)
    hist.round(4).to_csv(out / "composite_history.csv")
    table.round(4).to_csv(out / "ranked_contributions.csv")

    label = (validation or {}).get("composite_label", "UNVALIDATED")
    fig = VIS.dashboard(comp, diff, z, table, s["gauge"]["needles_m"], label=label,
                        watermark=watermark, ltm=ltm)
    fig.savefig(out / "inflation_dashboard.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "inflation_dashboard.pdf", bbox_inches="tight")

    hook = R.emit("inflation", comp, diff, table, conc, ltm, validation,
                  s["regime_hook"]["trajectory_band"])
    if watermark:
        hook["status"] = "SYNTHETIC"
    R.save(hook, out)
    return hook


def main(argv=None):
    ap = argparse.ArgumentParser(prog="infcomp")
    ap.add_argument("cmd", choices=["verify", "build", "validate", "demo"])
    ap.add_argument("--pit", action="store_true",
                    help="validate: shift inputs by publication lag")
    ap.add_argument("--vintage", default=None,
                    help="ALFRED realtime_end (YYYY-MM-DD) for point-in-time data")
    ap.add_argument("--eval-start", default=None,
                    help="validate: restrict OOS scoring to dates >= this (YYYY-MM-DD); "
                         "overrides gate.yaml's eval_start if both are set")
    a = ap.parse_args(argv)
    s = _settings()

    if a.cmd == "verify":
        from .verify import verify
        df = verify(_client(s), manual_dir=s["manual_dir"], previous=VERIFIED)
        df.to_csv(VERIFIED, index=False)
        print(df[["key", "series_id", "status", "title", "frequency", "history_start",
                  "units", "resolved_transform"]].to_string(index=False))
        print(f"\nWrote {VERIFIED}. Review each row, then set confirmed=TRUE for the ones "
              "you accept. MISMATCH/UNVERIFIED rows stay out of the composite.")

    elif a.cmd == "build":
        inputs, targets, _benchmarks, meta = _load_panel(s)
        vpath = Path(s["output_dir"]) / "validation_result.json"
        val = json.loads(vpath.read_text()) if vpath.exists() else None
        hook = compute_and_render(inputs, targets, meta, s, Path(s["output_dir"]), validation=val)
        print(json.dumps(hook, indent=2))

    elif a.cmd == "validate":
        inputs, targets, benchmarks, _ = _load_panel(s, pit=a.pit, realtime_end=a.vintage)
        policy = f"{'vintage ' + a.vintage if a.vintage else 'latest'}" + (" + pub-lag" if a.pit else "")
        res = V.run(inputs, targets, s, vintage_policy=policy, benchmarks=benchmarks,
                    eval_start=a.eval_start)
        print(f"Saved {V.save(res, s['output_dir'])}")
        print(json.dumps({k: res[k] for k in ("results_common", "n_common", "min_common_oos",
                                              "underpowered", "pca_promoted", "production_method",
                                              "composite_label", "benchmark", "eval_start")},
                         indent=2))
        print("\n(results_own — per-method sample context, NOT comparable across methods):")
        print(json.dumps(res["results_own"], indent=2))

    elif a.cmd == "demo":
        from . import synthetic
        inputs, targets, meta = synthetic.make()
        out = Path(s["output_dir"]) / "demo_synthetic"
        hook = compute_and_render(inputs, targets, meta, s, out, watermark=synthetic.WATERMARK)
        print(f"Synthetic smoke test written to {out}/ (status={hook['status']})")


if __name__ == "__main__":
    main()
