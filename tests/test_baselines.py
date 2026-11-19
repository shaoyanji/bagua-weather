"""Baseline tests (M3/M4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bagua_weather import baselines
from bagua_weather.evaluate import evaluate
from bagua_weather.states import N_STATES


def _pairs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    prev = rng.integers(0, N_STATES, n)
    nxt = (prev + rng.integers(0, 3, n)) % N_STATES
    return pd.DataFrame({"prev": prev, "nxt": nxt}, index=idx)


def test_climatology_normalizes():
    p = baselines.climatology(_pairs())
    assert np.isclose(p.sum(), 1) and p.shape == (N_STATES,)


def test_persistence_row_stochastic():
    P = baselines.persistence_matrix(_pairs())
    assert np.allclose(P.sum(axis=1), 1)
    assert (np.diag(P) == P.max(axis=1)).all()  # stay prob dominates each row


def test_gbm_missing_sklearn_returns_none(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pairs = _pairs()
    assert baselines.gbm(pairs.iloc[:200], pairs.iloc[200:]) is None


def test_gbm_proba_matrix_when_sklearn_present():
    sklearn = pytest.importorskip("sklearn")
    del sklearn
    pairs = _pairs(2000)
    train, test = pairs.iloc[:1500], pairs.iloc[1500:]
    probs = baselines.gbm(train, test)
    assert probs.shape == (len(test), N_STATES)
    assert np.allclose(probs.sum(axis=1), 1)
    assert (probs >= 0).all()


def test_evaluate_includes_gbm_when_available():
    sklearn = pytest.importorskip("sklearn")
    del sklearn
    idx = pd.date_range("2020-01-01", periods=48 * 30, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    # persistent-ish chain so all classes appear
    states = np.zeros(len(idx), dtype=int)
    for i in range(1, len(idx)):
        states[i] = states[i - 1] if rng.random() < 0.8 else rng.integers(0, N_STATES)
    s = pd.Series(states, index=idx)
    out = evaluate(s, "2020-01-20")
    assert "gbm" in out
    assert out["gbm"]["log_loss"] > 0
    assert 0 <= out["gbm"]["top1_acc"] <= 1
