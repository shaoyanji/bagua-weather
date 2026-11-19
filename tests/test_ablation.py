"""M5 ablation tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bagua_weather.ablation import VARIANTS, run_ablation, verdict
from bagua_weather.ingest import STATIONS
from bagua_weather.model import fit_transition_matrix
from bagua_weather.states import classify

FREIBURG = STATIONS["01443"]


def _df(n=48, temp=20.0):
    idx = pd.date_range("2020-06-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "sun_minutes": 45.0,
            "wind_ms": 7.0,
            "precip_mm": 0.0,
            "temp_c": np.full(n, temp),
        },
        index=idx,
    )


def test_classify_default_matches_eight_state():
    s = classify(_df())
    assert set(s.dropna().unique()) == {6}  # sun+wind, no wet -> 0b110


def test_classify_bit_subsets():
    df = _df()
    s4 = classify(df, bits=("sun", "wet"))
    assert set(s4.dropna().unique()) == {2}  # sun bit only -> 0b10
    s4w = classify(df, bits=("wind", "wet"))
    assert set(s4w.dropna().unique()) == {2}  # wind bit only -> 0b10
    s16 = classify(df, bits=("sun", "wind", "wet", "temp"))
    assert set(s16.dropna().unique()) == {14}  # sun+wind+temp, no wet -> 0b1110
    cold = classify(_df(temp=5.0), bits=("sun", "wind", "wet", "temp"))
    assert set(cold.dropna().unique()) == {6}  # temp bit off -> same as 8-state


def test_classify_16_requires_temp_column():
    with pytest.raises(KeyError, match="temp_c"):
        classify(_df().drop(columns="temp_c"), bits=("sun", "wind", "wet", "temp"))


def test_classify_unknown_bit_raises():
    with pytest.raises(ValueError, match="unknown bits"):
        classify(_df(), bits=("sun", "bogus"))


def test_fit_transition_matrix_n_states():
    idx = pd.date_range("2020-01-01", periods=20, freq="h", tz="UTC")
    pairs = pd.DataFrame({"prev": np.arange(20) % 16, "nxt": (np.arange(20) + 1) % 16},
                       index=idx)
    P = fit_transition_matrix(pairs, alpha=1.0, n_states=16)
    assert P.shape == (16, 16) and np.allclose(P.sum(axis=1), 1)


def _synthetic_df(days=40):
    idx = pd.date_range("2020-06-01", periods=24 * days, freq="h", tz="UTC")
    h = idx.hour.to_numpy()
    return pd.DataFrame(
        {
            "sun_minutes": np.where((h >= 9) & (h <= 17), 40.0, 0.0),
            "wind_ms": np.where(np.arange(len(idx)) % 96 < 48, 6.0, 2.0),
            "precip_mm": np.where(np.arange(len(idx)) % 72 < 24, 0.5, 0.0),
            "temp_c": np.where(np.arange(len(idx)) % 48 < 24, 20.0, 10.0),
        },
        index=idx,
    )


def test_run_ablation_writes_json(tmp_path, monkeypatch):
    # Speed: the ablation invokes evaluate() which runs GBM if sklearn is
    # installed; stub it out to keep the test fast.
    import bagua_weather.evaluate as ev

    monkeypatch.setattr(ev.baselines, "gbm", lambda *a, **k: None)
    out = run_ablation(
        _synthetic_df(), FREIBURG["lat"], FREIBURG["lon"],
        split="2020-06-21", reports_dir=tmp_path,
    )
    rows = out["variants"]
    assert {r["n_states"] for r in rows} == {4, 8, 16}
    assert sum(r["n_states"] == 4 for r in rows) == 3  # all three drops
    for r in rows:
        assert r["skill_nats"] >= 0 and 0 < r["rel_ll"] <= 1
    js = json.loads((tmp_path / "ablation.json").read_text())
    assert len(js["variants"]) == len(VARIANTS)
    assert "skill" in verdict(rows)
