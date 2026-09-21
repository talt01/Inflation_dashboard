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


def test_quarterly_to_monthly_ffills_two_months():
    idx = pd.date_range("2020-03-31", periods=4, freq="QE")
    m = T.to_monthly(pd.Series([1, 2, 3, 4.0], index=idx), native_freq="Quarterly")
    assert m.loc["2020-05-31"] == 1 and m.loc["2020-06-30"] == 2


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
    assert r["composite_label"] in ("SIGNAL", "DESCRIPTIVE")
    assert "syn_market_1" not in r["inputs_used"]
    assert len(r["gate_sha256"]) == 64
    assert r["benchmark"] == "hist_mean"   # default when gate omits `benchmark`


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


def test_validation_market_benchmark_end_to_end(tmp_path):
    inp, tgt, _ = synthetic.make(240)
    bmk = synthetic.make_benchmarks(tgt.index)
    g = tmp_path / "gate.yaml"
    g.write_text(yaml.safe_dump(dict(target="fedfunds_change", horizon_m=6, metric="oos_r2",
        benchmark="market", pca_promotion_margin=0.02, signal_min_oos_r2=0.02,
        min_train_months=60, exclude_inputs=["syn_market_1", "syn_market_2"],
        committed_on="2026-09-21", committed_by="test")))
    s = yaml.safe_load(open("config/settings.yaml"))
    r = V.run(inp, tgt, s, gate_path=g, benchmarks=bmk)
    assert r["benchmark"] == "market"
    assert r["composite_label"] in ("SIGNAL", "DESCRIPTIVE")
    assert np.isfinite(r["results"]["zscore_avg"]["oos_r2"])


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
