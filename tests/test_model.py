import numpy as np
import pandas as pd

from bagua_weather import model
from bagua_weather.evaluate import evaluate


def _chain(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    P = 0.5 * np.eye(8) + 0.5 * rng.dirichlet(np.ones(8) * 0.3, size=8)
    path = model.sample_path(P, 0, n, rng)
    idx = pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC")
    return P, pd.Series(path, index=idx).astype("Int8")


def test_fit_recovers_matrix():
    P, s = _chain()
    P_hat = model.fit_transition_matrix(model.transition_pairs(s), alpha=0.01)
    assert np.abs(P - P_hat).max() < 0.06
    assert np.allclose(P_hat.sum(axis=1), 1)


def test_pairs_drop_gaps_and_na():
    idx = pd.to_datetime(
        ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 03:00", "2020-01-01 04:00"], utc=True
    )
    s = pd.Series([1, 2, 3, pd.NA], index=idx, dtype="Int8")
    pairs = model.transition_pairs(s)
    assert len(pairs) == 1 and (pairs.iloc[0]["prev"], pairs.iloc[0]["nxt"]) == (1, 2)


def test_stationary_distribution_sums_to_one():
    P, _ = _chain(n=10)
    pi = model.stationary_distribution(P)
    assert np.isclose(pi.sum(), 1) and np.allclose(pi @ P, pi, atol=1e-8)


def test_markov_beats_baselines_on_synthetic_chain():
    _, s = _chain()
    res = evaluate(s, split="2017-01-01")
    assert res["markov_global"]["log_loss"] < res["climatology"]["log_loss"]
    assert res["markov_global"]["log_loss"] < res["persistence"]["log_loss"]
