"""Chronological train/test evaluation of the Markov model against baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import baselines, model


def _score(probs: np.ndarray, nxt: np.ndarray) -> dict:
    p = probs[np.arange(len(nxt)), nxt]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(p, 1e-12, None)))),
        "top1_acc": float(np.mean(probs.argmax(axis=1) == nxt)),
    }


def evaluate(
    states: pd.Series,
    split: str | pd.Timestamp,
    alpha: float = 1.0,
    n_states: int = baselines.N_STATES,
) -> dict:
    """Train on transitions before `split`, test on the rest. Lower log_loss is better."""
    pairs = model.transition_pairs(states)
    split = pd.Timestamp(split)
    if pairs.index.tz is not None and split.tzinfo is None:
        split = split.tz_localize(pairs.index.tz)
    train, test = pairs[pairs.index < split], pairs[pairs.index >= split]
    if train.empty or test.empty:
        raise ValueError("empty train or test set; check split date and data coverage")
    prev, nxt = test["prev"].to_numpy(), test["nxt"].to_numpy()

    clim = np.tile(baselines.climatology(train, alpha, n_states), (len(test), 1))
    pers = baselines.persistence_matrix(train, n_states)[prev]
    P_glob = model.fit_transition_matrix(train, alpha, n_states)[prev]

    ctx_models = model.fit_contextual(train, alpha, n_states)
    ctx = model.context_labels(test.index).to_numpy()
    P_ctx = np.empty((len(test), P_glob.shape[1]))
    for i, (p, c) in enumerate(zip(prev, ctx)):
        P_ctx[i] = ctx_models.get(c, ctx_models[None])[p]

    out = {
        "n_train": len(train),
        "n_test": len(test),
        "climatology": _score(clim, nxt),
        "persistence": _score(pers, nxt),
        "markov_global": _score(P_glob, nxt),
        "markov_contextual": _score(P_ctx, nxt),
    }
    gbm_probs = baselines.gbm(train, test, n_states)
    if gbm_probs is not None:
        out["gbm"] = _score(gbm_probs, nxt)
    return out
