"""Baselines the Markov model must beat. TODO(v0.1.0 M4): add an HMM / gradient-boosting baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .states import N_STATES


def climatology(train: pd.DataFrame, alpha: float = 1.0) -> np.ndarray:
    """Marginal state distribution, ignoring the previous state."""
    p = np.bincount(train["nxt"], minlength=N_STATES) + alpha
    return p / p.sum()


def persistence_matrix(train: pd.DataFrame) -> np.ndarray:
    """'Same as last hour', smoothed: stay with the empirical stay rate, else spread evenly."""
    stay = float((train["prev"] == train["nxt"]).mean())
    stay = min(max(stay, 1e-3), 1 - 1e-3)
    P = np.full((N_STATES, N_STATES), (1 - stay) / (N_STATES - 1))
    np.fill_diagonal(P, stay)
    return P
