"""M3 report tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bagua_weather.ingest import STATIONS
from bagua_weather.report import dwell_times, run_eval

FREIBURG = STATIONS["01443"]


def _series(vals):
    idx = pd.date_range("2020-06-01", periods=len(vals), freq="h", tz="UTC")
    return pd.Series(vals, index=idx, dtype="Int64")


def test_dwell_times_runs_and_gaps():
    # 11111 then a 2h gap (NA + missing timestamp) then 22222
    idx_a = pd.date_range("2020-06-01 08:00", periods=5, freq="h", tz="UTC")
    idx_b = pd.date_range("2020-06-02 08:00", periods=5, freq="h", tz="UTC")
    s = pd.Series([1] * 5 + [2] * 5, index=idx_a.append(idx_b))
    runs = dwell_times(s)
    assert runs[1] == [5] and runs[2] == [5]


def test_dwell_times_breaks_on_na():
    s = _series([3, 3, None, 3, 3])
    runs = dwell_times(s)
    assert runs[3] == [2, 2]


def test_dwell_times_breaks_on_nonconsecutive_index():
    idx = pd.date_range("2020-06-01", periods=3, freq="h", tz="UTC")
    s = pd.Series([0, 0], index=idx.delete(1))  # missing middle hour
    runs = dwell_times(s)
    assert runs[0] == [1, 1]  # two runs, not one run of 2


def _synthetic_df(days=30):
    idx = pd.date_range("2020-06-01", periods=24 * days, freq="h", tz="UTC")
    h = idx.hour.to_numpy()
    sun = np.where((h >= 9) & (h <= 17), 40.0, 0.0)
    precip = np.where((np.arange(len(idx)) // 72) % 3 == 0, 0.0, 0.6)
    return pd.DataFrame(
        {"sun_minutes": sun, "wind_ms": np.full(len(idx), 2.5), "precip_mm": precip},
        index=idx,
    )


def test_run_eval_writes_metrics_and_figures(tmp_path):
    out = tmp_path / "reports"
    summary = run_eval(
        _synthetic_df(90), FREIBURG["lat"], FREIBURG["lon"],
        split="2020-07-01", reports_dir=out,
    )
    metrics = json.loads((out / "metrics.json").read_text())
    for fig in summary["figures"]:
        assert (out / fig).stat().st_size > 1000, fig
    for model in ("climatology", "persistence", "markov_global", "markov_contextual"):
        assert metrics["models"][model]["log_loss"] > 0
        assert 0.0 <= metrics["models"][model]["top1_acc"] <= 1.0
    assert metrics["n_train_pairs"] > 0 and metrics["n_test_pairs"] > 0
    assert len(metrics["dwell"]) > 0
    for d in metrics["dwell"].values():
        assert d["empirical_mean_h"] > 0 and d["geometric_mean_h"] > 0


def test_run_eval_empty_test_split_raises():
    with pytest.raises(ValueError):
        run_eval(
            _synthetic_df(30), FREIBURG["lat"], FREIBURG["lon"],
            split="2021-01-01", reports_dir="/tmp/nope",
        )
