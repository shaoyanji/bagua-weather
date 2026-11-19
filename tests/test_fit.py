"""M2 fit pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bagua_weather import cli
from bagua_weather.fit import fit_pipeline, load_fit, save_fit
from bagua_weather.ingest import STATIONS

FIXTURES = Path(__file__).parent / "fixtures"
FREIBURG = STATIONS["01443"]


def _synthetic_df(days=20):
    """Daylight-heavy synthetic frame: alternating wet/dry spells."""
    idx = pd.date_range("2020-06-01", periods=24 * days, freq="h", tz="UTC")
    h = idx.hour.to_numpy()
    sun = np.where((h >= 9) & (h <= 17), 45.0, 0.0)
    wind = np.full(len(idx), 3.0)
    precip = np.where((np.arange(len(idx)) // 48) % 2 == 0, 0.0, 0.5)
    return pd.DataFrame(
        {"sun_minutes": sun, "wind_ms": wind, "precip_mm": precip}, index=idx
    )


def test_fit_pipeline_classifies_masks_and_fits():
    fit = fit_pipeline(_synthetic_df(), FREIBURG["lat"], FREIBURG["lon"])
    P, cov = fit["P"], fit["coverage"]
    assert np.allclose(P.sum(axis=1), 1)
    assert cov["classified_states"] > cov["after_night_mask"]  # night masked out
    assert cov["transition_pairs"] > 0
    # Every context matrix is also row-stochastic.
    for label, m in fit["contextual"].items():
        assert np.allclose(m.sum(axis=1), 1), label
    # No night states survive the mask.
    from bagua_weather.solar import daylight_mask

    day = daylight_mask(fit["states"].index, FREIBURG["lat"], FREIBURG["lon"])
    assert fit["states"].notna().to_numpy()[~day].sum() == 0


def test_fit_pipeline_empty_pairs_raises():
    idx = pd.date_range("2020-06-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"sun_minutes": np.nan, "wind_ms": 1.0, "precip_mm": 0.0}, index=idx
    )
    with pytest.raises(ValueError, match="no consecutive"):
        fit_pipeline(df, FREIBURG["lat"], FREIBURG["lon"])


def test_save_and_load_fit_roundtrip(tmp_path):
    fit = fit_pipeline(_synthetic_df(), FREIBURG["lat"], FREIBURG["lon"])
    save_fit(fit, tmp_path)
    back = load_fit(tmp_path)
    assert np.allclose(back["P"], fit["P"])
    for label, m in fit["contextual"].items():
        assert np.allclose(back["contextual"][label], m)
    assert (tmp_path / "fit_report.json").exists()


def test_cli_fit_end_to_end(tmp_path, monkeypatch, capsys):
    """`bagua fit` on the fixture cache writes matrices + a coverage report."""
    out = tmp_path / "processed"
    # The fixture has only a few daylight hours; the pipeline must still run.
    cli.main([
        "fit", "--station", "01443", "--cache", str(FIXTURES),
        "--out", str(out), "--since", "2003-08-30",
    ])
    assert (out / "fit.npz").exists() and (out / "fit_report.json").exists()
    back = load_fit(out)
    assert np.allclose(back["P"].sum(axis=1), 1)


def test_cli_rejects_unknown_station(capsys):
    with pytest.raises(SystemExit):
        cli.main(["fit", "--station", "99999", "--cache", str(FIXTURES)])
