"""Baselines the Markov model must beat.

M4 adds a gradient-boosted classifier as the "stronger learned baseline"
(optional `scikit-learn` extra; returns None when it is not installed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .states import N_STATES


def _features(idx: pd.DatetimeIndex, prev: np.ndarray) -> np.ndarray:
    """Feature matrix: previous state + cyclical hour/month encodings."""
    h = idx.hour.to_numpy(dtype=float) * 2 * np.pi / 24
    m = idx.month.to_numpy(dtype=float) * 2 * np.pi / 12
    return np.column_stack([prev, np.sin(h), np.cos(h), np.sin(m), np.cos(m)])


def gbm(train: pd.DataFrame, test: pd.DataFrame, n_states: int = N_STATES) -> np.ndarray | None:
    """HistGradientBoosting next-state classifier trained on the train pairs.

    Returns an (len(test), N_STATES) probability matrix, or None when
    scikit-learn is unavailable. Unseen classes get probability 0 (smoothed
    at scoring time via the clip floor).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return None
    # `prev` as a native categorical + l2 + early stopping (validation drawn
    # only from the train slice — test data never enters fitting). Without
    # the categorical hint the GBM underfits badly (LL 1.18 vs 0.76 tuned).
    clf = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, categorical_features=[0],
        early_stopping=True, validation_fraction=0.15, random_state=42,
    )
    clf.fit(_features(train.index, train["prev"].to_numpy()), train["nxt"].to_numpy())
    proba = clf.predict_proba(_features(test.index, test["prev"].to_numpy()))
    full = np.zeros((len(test), n_states))
    full[:, clf.classes_.astype(int)] = proba
    return full


def climatology(train: pd.DataFrame, alpha: float = 1.0, n_states: int = N_STATES) -> np.ndarray:
    """Marginal state distribution, ignoring the previous state."""
    p = np.bincount(train["nxt"], minlength=n_states) + alpha
    return p / p.sum()


def persistence_matrix(train: pd.DataFrame, n_states: int = N_STATES) -> np.ndarray:
    """'Same as last hour', smoothed: stay with the empirical stay rate, else spread evenly."""
    stay = float((train["prev"] == train["nxt"]).mean())
    stay = min(max(stay, 1e-3), 1 - 1e-3)
    P = np.full((n_states, n_states), (1 - stay) / (n_states - 1))
    np.fill_diagonal(P, stay)
    return P
