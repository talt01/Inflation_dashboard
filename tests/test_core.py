import numpy as np
import pandas as pd
import pytest
import yaml

from infcomp import composite as C, data as D, transforms as T, validation as V, synthetic
from infcomp.verify import resolve_transform, verify


def test_yoy_and_ann_to_12m():
    idx = pd.date_range("2000-01-31", periods=36, freq="ME")
    s = pd.Series(100 * 1.002 ** np.arange(36), index=idx)
    assert T.yoy(s).iloc[-1] == pytest.approx((1.002**12 - 1) * 100)
    assert T.ann_to_12m(pd.Series(3.0, index=idx)).iloc[-1] == pytest.approx(3.0)


def test_quarterly_to_monthly_ffills_within_a_quarter():
    idx = pd.date_range("2020-03-31", periods=4, freq="QE")
    m = T.to_monthly(pd.Series([1, 2, 3, 4.0], index=idx), native_freq="Quarterly")
    assert m.loc["2020-05-31"] == 1 and m.loc["2020-06-30"] == 2


def test_quarterly_to_monthly_survives_one_missing_quarter():
    # a genuinely missing quarter (real FRED gap, e.g. ECI) used to leave a
    # mid-history NaN hole under ffill(limit=2); limit=3 closes one more
    # month of it without becoming interpolation (still a strict ffill).
    idx = pd.to_datetime(["2018-01-01", "2018-04-01", "2018-10-01",  # 2018Q3 missing
                          "2019-01-01"])
    s = pd.Series([100.0, 101.0, 103.0, 104.0], index=idx)
    m = T.to_monthly(s, native_freq="Quarterly")
    assert m.loc["2018-09-30"] == 101.0          # now covered (was NaN under limit=2)
    assert m.loc["2018-10-31":"2018-11-30"].isna().all()  # a 5-month gap still isn't fully bridged


def test_expanding_z_has_no_lookahead():
    inp, _, _ = synthetic.make()
    pd.testing.assert_frame_equal(C.expanding_z(inp, 60).iloc[:200],
                                  C.expanding_z(inp.iloc[:200], 60))
    pd.testing.assert_frame_equal(C.expanding_z(inp, 60, window=120).iloc[:200],
                                  C.expanding_z(inp.iloc[:200], 60, window=120))


def test_contributions_sum_to_composite():
    inp, _, _ = synthetic.make()
    comp, contrib, _ = C.zscore_composite(C.expanding_z(inp, 60))
    assert (contrib.sum(axis=1) - comp).dropna().abs().max() < 1e-12


def test_override_is_clamped_and_separate():
    inp, _, _ = synthetic.make()
    out = C.ltm_composite(C.ltm_scores(inp)["ltm"], [{"as_of": "2026-08-31", "value": -3}])
    assert out.loc["2026-08-31", "manual_adj"] == -1.25
    assert out["manual_adj"].drop(pd.Timestamp("2026-08-31")).eq(0).all()


def test_gate_refuses_when_uncommitted(tmp_path):
    g = tmp_path / "gate.yaml"
    g.write_text(yaml.safe_dump({"target": None, "horizon_m": 6}))
    with pytest.raises(V.GateNotCommitted):
        V.load_gate(g)


def test_shipped_gate_is_committed():
    # config/gate.yaml now carries Tim's filled, git-committed decisions
    # (market-benchmark patch), so it must load without raising.
    gate, gate_sha = V.load_gate("config/gate.yaml")
    assert gate["target"] == "fedfunds_change"
    assert gate["benchmark"] == "market"
    assert gate["committed_by"] == "Tim"
    assert len(gate_sha) == 64


def test_resolve_transform_from_units():
    assert resolve_transform("auto", "Index 1982-1984=100", "Monthly") == "yoy"
    assert resolve_transform("auto", "Percent Change at Annual Rate", "Monthly") == "ann_to_12m"
    assert resolve_transform("auto", "Percent Change from Year Ago", "Monthly") == "level"
    assert resolve_transform("auto", "Thousands of Units", "Monthly") == "UNRESOLVED"


class FakeClient:
    """Resolves one ID correctly, one to the WRONG series, one not at all."""
    def series_info(self, sid):
        if sid == "GOOD":
            return {"title": "Consumer Price Index for All Urban Consumers: All Items",
                    "frequency": "Monthly", "units": "Index 1982-1984=100",
                    "observation_start": "1947-01-01", "observation_end": "2026-08-01"}
        if sid == "WRONG":
            return {"title": "Unemployment Rate", "frequency": "Monthly", "units": "Percent"}
        raise RuntimeError("Bad Request. The series does not exist.")


def test_verify_statuses(tmp_path):
    m = tmp_path / "map.yaml"
    base = dict(source="fred", transform="auto", sign=1, role="core", group="g", pub_lag_m=1,
                expect_title=["consumer price index", "all items"])
    m.write_text(yaml.safe_dump({"indicators": [
        {**base, "key": "a", "name": "A", "series_id": "GOOD"},
        {**base, "key": "b", "name": "B", "series_id": "WRONG"},
        {**base, "key": "c", "name": "C", "series_id": "MISSING"},
        {"key": "d", "name": "D", "source": "proprietary", "transform": "level",
         "sign": 1, "role": "core", "group": "g"}]}))
    df = verify(FakeClient(), m, manual_dir=tmp_path).set_index("key")
    assert df.loc["a", "status"] == "VERIFIED" and df.loc["a", "resolved_transform"] == "yoy"
    assert df.loc["b", "status"] == "MISMATCH"
    assert df.loc["c", "status"] == "UNVERIFIED"
    assert df.loc["d", "status"] == "PROPRIETARY"
    assert not df["confirmed"].any()   # nothing is auto-confirmed


def test_validation_runs_on_committed_gate(tmp_path):
    inp, tgt, _ = synthetic.make(240)
    g = tmp_path / "gate.yaml"
    g.write_text(yaml.safe_dump(dict(target="be5_change", horizon_m=6, metric="oos_r2",
        pca_promotion_margin=0.02, signal_min_oos_r2=0.0, min_train_months=60,
        exclude_inputs=["syn_market_1", "syn_market_2"], committed_on="2026-09-21",
        committed_by="test")))
    s = yaml.safe_load(open("config/settings.yaml"))
    r = V.run(inp, tgt, s, gate_path=g)
    assert r["composite_label"] in ("SIGNAL", "DESCRIPTIVE", "UNDERPOWERED")
    assert "syn_market_1" not in r["inputs_used"]
    assert len(r["gate_sha256"]) == 64
    assert r["benchmark"] == "hist_mean"   # default when gate omits `benchmark`
    # results_own is per-method context and not bounded by the common sample;
    # results_common (the decision-grade comparison) can only be <= each method's own n
    assert set(r["results_own"]) == set(r["results_common"]) == {"zscore_avg", "ltm_0_5", "pca_pc1"}
    for m in r["results_common"]:
        assert r["results_common"][m]["n_oos"] <= r["results_own"][m]["n_oos"]
    assert r["n_common"] == min(v["n_oos"] for v in r["results_common"].values())


# ------------------------------------------------------------- market benchmark patch

def test_market_policy_change_formula():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    bmk = pd.DataFrame({"bmk_dgs3mo": [1.0, 1.5, 2.0],
                        "bmk_dgs6mo": [1.2, 1.6, 2.1],
                        "bmk_dgs1":   [1.5, 1.8, 2.3]}, index=idx)
    out = V.market_policy_change(bmk, 6)
    expected = ((2.0 * bmk["bmk_dgs1"] - bmk["bmk_dgs6mo"]) - bmk["bmk_dgs3mo"]) * 100
    pd.testing.assert_series_equal(out, expected.rename("market_implied"))


def test_market_policy_change_only_supports_h6():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    bmk = pd.DataFrame({"bmk_dgs3mo": [1.0] * 3, "bmk_dgs6mo": [1.0] * 3,
                        "bmk_dgs1": [1.0] * 3}, index=idx)
    with pytest.raises(ValueError):
        V.market_policy_change(bmk, 3)


def test_oos_r2_benchmark_stays_aligned_through_nan_gaps():
    # regression test: a NaN in the benchmark series used to `continue` the loop
    # AFTER `preds` had already been appended, desyncing preds/bench/actual and
    # crashing (or silently misaligning) the R^2 calculation.
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=200), index=idx)
    y = pd.Series(rng.normal(size=200), index=idx)
    bmk = pd.Series(rng.normal(size=200), index=idx)
    bmk.iloc[::7] = np.nan
    res = V.oos_r2(x, y, h=6, min_train=60, benchmark=bmk)
    assert res["n_oos"] > 0
    assert np.isfinite(res["oos_r2"])


def test_build_target_fedfunds_change():
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    tgt = pd.DataFrame({"tgt_fedfunds": np.arange(8, dtype=float)}, index=idx)
    out = V.build_target(tgt, "fedfunds_change", h=2)
    assert out.iloc[0] == pytest.approx(200.0)  # (2-0)*100 bp
    assert np.isnan(out.iloc[-1])


def _market_gate_dict(**overrides):
    g = dict(target="fedfunds_change", horizon_m=6, metric="oos_r2",
             benchmark="market", pca_promotion_margin=0.02, signal_min_oos_r2=0.02,
             min_train_months=60, exclude_inputs=["syn_market_1", "syn_market_2"],
             committed_on="2026-09-21", committed_by="test")
    g.update(overrides)
    return g


def test_validation_market_benchmark_end_to_end(tmp_path):
    inp, tgt, _ = synthetic.make(240)
    bmk = synthetic.make_benchmarks(tgt.index)
    g = tmp_path / "gate.yaml"
    g.write_text(yaml.safe_dump(_market_gate_dict()))
    s = yaml.safe_load(open("config/settings.yaml"))
    r = V.run(inp, tgt, s, gate_path=g, benchmarks=bmk)
    assert r["benchmark"] == "market"
    assert r["composite_label"] in ("SIGNAL", "DESCRIPTIVE", "UNDERPOWERED")
    assert np.isfinite(r["results_common"]["zscore_avg"]["oos_r2"])
    assert np.isfinite(r["results_own"]["zscore_avg"]["oos_r2"])


# ------------------------------------------------------ Fix A: common-sample comparison

def test_oos_r2_restrict_only_scores_given_dates():
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=200), index=idx)
    y = pd.Series(rng.normal(size=200), index=idx)
    full = V.oos_r2(x, y, h=6, min_train=60)
    half = V.oos_r2(x, y, h=6, min_train=60, restrict=idx[100:])
    assert half["n_oos"] < full["n_oos"]
    assert half["n_oos"] > 0


def test_run_flags_underpowered_when_common_sample_is_thin(tmp_path):
    # PCA needs complete rows across all composite inputs, so on synthetic data
    # (240m, min_periods=60) its own sample is far smaller than z-avg/L-T-M's —
    # this reproduces the exact bug the patch describes: a method-vs-method
    # comparison on mismatched sample sizes flashing a false SIGNAL.
    inp, tgt, _ = synthetic.make(240)
    bmk = synthetic.make_benchmarks(tgt.index)
    g = tmp_path / "gate.yaml"
    g.write_text(yaml.safe_dump(_market_gate_dict()))  # default min_common_oos=60
    s = yaml.safe_load(open("config/settings.yaml"))
    r = V.run(inp, tgt, s, gate_path=g, benchmarks=bmk)
    assert r["results_own"]["pca_pc1"]["n_oos"] < r["results_own"]["zscore_avg"]["n_oos"]
    assert r["n_common"] == r["results_common"]["pca_pc1"]["n_oos"]  # PCA is the binding constraint
    assert r["underpowered"] is True
    assert r["composite_label"] == "UNDERPOWERED"
    assert r["pca_promoted"] is False   # UNDERPOWERED blocks promotion outright


def test_run_min_common_oos_from_gate_lifts_underpowered_guard(tmp_path):
    inp, tgt, _ = synthetic.make(240)
    bmk = synthetic.make_benchmarks(tgt.index)
    g = tmp_path / "gate.yaml"
    g.write_text(yaml.safe_dump(_market_gate_dict(min_common_oos=1)))
    s = yaml.safe_load(open("config/settings.yaml"))
    r = V.run(inp, tgt, s, gate_path=g, benchmarks=bmk)
    assert r["underpowered"] is False
    assert r["composite_label"] in ("SIGNAL", "DESCRIPTIVE")


def test_run_eval_start_restricts_common_sample(tmp_path):
    inp, tgt, _ = synthetic.make(240)
    bmk = synthetic.make_benchmarks(tgt.index)
    g = tmp_path / "gate.yaml"
    g.write_text(yaml.safe_dump(_market_gate_dict(min_common_oos=1)))
    s = yaml.safe_load(open("config/settings.yaml"))
    r_full = V.run(inp, tgt, s, gate_path=g, benchmarks=bmk)
    r_restricted = V.run(inp, tgt, s, gate_path=g, benchmarks=bmk, eval_start="2023-01-01")
    assert r_restricted["n_common"] < r_full["n_common"]
    assert r_restricted["eval_start"] == "2023-01-01"
    assert r_full["eval_start"] is None
    # CLI --eval-start overrides a gate-level eval_start when both are given
    g2 = tmp_path / "gate2.yaml"
    g2.write_text(yaml.safe_dump(_market_gate_dict(min_common_oos=1, eval_start="2015-01-01")))
    r_override = V.run(inp, tgt, s, gate_path=g2, benchmarks=bmk, eval_start="2023-01-01")
    assert r_override["eval_start"] == "2023-01-01"


class FakePanelClient:
    def __init__(self, series):
        self.series = series

    def observations(self, series_id, realtime_end=None):
        return self.series[series_id]


def test_build_panel_routes_benchmark_role(tmp_path):
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    m = tmp_path / "map.yaml"
    m.write_text(yaml.safe_dump({"indicators": [
        {"key": "core_a", "name": "Core A", "source": "fred", "series_id": "COREA",
         "transform": "level", "sign": 1, "role": "core", "group": "g", "pub_lag_m": 1},
        {"key": "tgt_x", "name": "Target X", "source": "fred", "series_id": "TGTX",
         "transform": "level", "sign": 1, "role": "target", "group": "target", "pub_lag_m": 0},
        {"key": "bmk_y", "name": "Benchmark Y", "source": "fred", "series_id": "BMKY",
         "transform": "level", "sign": 1, "role": "benchmark", "group": "benchmark", "pub_lag_m": 0},
    ]}))
    verified = pd.DataFrame([
        {"key": "core_a", "series_id": "COREA", "source": "fred", "status": "VERIFIED",
         "confirmed": True, "frequency": "Monthly", "resolved_transform": "level"},
        {"key": "tgt_x", "series_id": "TGTX", "source": "fred", "status": "VERIFIED",
         "confirmed": True, "frequency": "Monthly", "resolved_transform": "level"},
        {"key": "bmk_y", "series_id": "BMKY", "source": "fred", "status": "VERIFIED",
         "confirmed": True, "frequency": "Monthly", "resolved_transform": "level"},
    ])
    client = FakePanelClient({
        "COREA": pd.Series(np.arange(6, dtype=float), index=idx, name="COREA"),
        "TGTX": pd.Series(np.arange(6, dtype=float) + 1, index=idx, name="TGTX"),
        "BMKY": pd.Series(np.arange(6, dtype=float) + 2, index=idx, name="BMKY"),
    })
    inputs, targets, benchmarks, meta = D.build_panel(client, verified, map_path=m, manual_dir=tmp_path)
    assert list(inputs.columns) == ["core_a"]
    assert list(targets.columns) == ["tgt_x"]
    assert list(benchmarks.columns) == ["bmk_y"]
    assert "bmk_y" not in meta.index and "tgt_x" not in meta.index


def test_validation_market_benchmark_requires_benchmarks_frame(tmp_path):
    inp, tgt, _ = synthetic.make(240)
    g = tmp_path / "gate.yaml"
    g.write_text(yaml.safe_dump(dict(target="fedfunds_change", horizon_m=6, metric="oos_r2",
        benchmark="market", pca_promotion_margin=0.02, signal_min_oos_r2=0.02,
        min_train_months=60, exclude_inputs=[], committed_on="2026-09-21",
        committed_by="test")))
    s = yaml.safe_load(open("config/settings.yaml"))
    with pytest.raises(ValueError):
        V.run(inp, tgt, s, gate_path=g)  # benchmarks=None
