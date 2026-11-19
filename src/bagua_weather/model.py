"""First-order Markov model of the 8 weather states, optionally conditioned on context."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .states import N_STATES

_SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def transition_pairs(states: pd.Series) -> pd.DataFrame:
    """Consecutive-hour (prev, nxt) pairs; drops gaps and missing values. Indexed by prev time."""
    arr = states.to_numpy(dtype=float, na_value=np.nan)
    idx = states.index
    prev, nxt = arr[:-1], arr[1:]
    consecutive = np.asarray((idx[1:] - idx[:-1]) == pd.Timedelta(hours=1))
    ok = consecutive & ~np.isnan(prev) & ~np.isnan(nxt)
    return pd.DataFrame(
        {"prev": prev[ok].astype(int), "nxt": nxt[ok].astype(int)}, index=idx[:-1][ok]
    )


def context_labels(index: pd.DatetimeIndex) -> pd.Series:
    """Season x 6-hour-of-day bucket (UTC), e.g. 'JJA-h2'."""
    labels = [f"{_SEASON[m]}-h{h // 6}" for m, h in zip(index.month, index.hour)]
    return pd.Series(labels, index=index)


def fit_transition_matrix(pairs: pd.DataFrame, alpha: float = 1.0) -> np.ndarray:
    """Row-stochastic 8x8 matrix with Laplace smoothing `alpha`."""
    counts = np.zeros((N_STATES, N_STATES))
    np.add.at(counts, (pairs["prev"].to_numpy(), pairs["nxt"].to_numpy()), 1)
    counts += alpha
    return counts / counts.sum(axis=1, keepdims=True)


def fit_contextual(pairs: pd.DataFrame, alpha: float = 1.0) -> dict:
    """One matrix per context label, plus a global fallback under key None."""
    ctx = context_labels(pairs.index)
    models = {None: fit_transition_matrix(pairs, alpha)}
    for label, sub in pairs.groupby(ctx.to_numpy()):
        models[label] = fit_transition_matrix(sub, alpha)
    return models


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmax(np.real(vals))])
    return v / v.sum()


def sample_path(P: np.ndarray, start: int, n: int, rng: np.random.Generator) -> np.ndarray:
    path = np.empty(n, dtype=int)
    path[0] = start
    for t in range(1, n):
        path[t] = rng.choice(N_STATES, p=P[path[t - 1]])
    return path
