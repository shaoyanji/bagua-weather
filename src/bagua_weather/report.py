"""M3 evaluation report: metrics.json + figures for the chronological eval.

Figures: transition-matrix heatmap (trigram labels), empirical state
frequencies vs the fitted stationary distribution, and dwell-time
histograms against the geometric dwell a first-order chain implies.
All numbers in metrics.json come from real runs on the loaded frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; never requires a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import evaluate
from .model import (
    fit_transition_matrix,
    stationary_distribution,
    transition_pairs,
)
from .solar import mask_night
from .states import DEFAULT_THRESHOLDS, N_STATES, Thresholds, classify, label

FIGS = ("transition_matrix.png", "state_frequencies.png", "dwell_times.png")


def dwell_times(states: pd.Series) -> dict[int, list[int]]:
    """Run lengths per state. A run ends on state change, <NA>, or a
    non-consecutive timestamp (night masked -> the gap between days)."""
    arr = states.to_numpy(dtype=float, na_value=np.nan)
    idx = states.index
    runs: dict[int, list[int]] = {s: [] for s in range(N_STATES)}
    run_state, run_len = np.nan, 0
    for i in range(len(arr)):
        gap = i > 0 and (idx[i] - idx[i - 1]) != pd.Timedelta(hours=1)
        if gap or arr[i] != run_state:
            if not np.isnan(run_state) and run_len > 0:
                runs[int(run_state)].append(run_len)
            run_state, run_len = arr[i], 1
        else:
            run_len += 1
    if not np.isnan(run_state) and run_len > 0:
        runs[int(run_state)].append(run_len)
    return runs


def _geometric_pmf(p_stay: float, n: int) -> np.ndarray:
    """P(dwell = k) for k = 1..n under a first-order chain."""
    p = np.clip(p_stay, 1e-9, 1 - 1e-9)
    k = np.arange(1, n + 1)
    return (p ** (k - 1)) * (1 - p)


def _fig_transition_matrix(P: np.ndarray, path: Path) -> None:
    labels = [label(s) for s in range(N_STATES)]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(P, cmap="viridis", vmin=0, vmax=max(0.2, P.max()))
    ax.set_xticks(range(N_STATES), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N_STATES), labels, fontsize=8)
    ax.set_xlabel("next state")
    ax.set_ylabel("previous state")
    ax.set_title("Transition matrix (train fit, trigram labels)")
    fig.colorbar(im, ax=ax, label="P(next | prev)")
    for i in range(N_STATES):
        for j in range(N_STATES):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if P[i, j] < P.max() * 0.6 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_frequencies(empirical: np.ndarray, stationary: np.ndarray, path: Path) -> None:
    x = np.arange(N_STATES)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, empirical, width=0.4, label="empirical (test)")
    ax.bar(x + 0.2, stationary, width=0.4, label="stationary (train fit)")
    ax.set_xticks(x, [label(s) for s in range(N_STATES)], rotation=45, ha="right",
                  fontsize=8)
    ax.set_ylabel("frequency")
    ax.set_title("State frequencies: empirical vs stationary distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_dwell(runs: dict[int, list[int]], P: np.ndarray, path: Path,
               max_h: int = 48) -> dict:
    """Per-state dwell histogram vs the geometric dwell P[i,i] implies."""
    n_states_present = sum(1 for r in runs.values() if r)
    cols = 4
    rows = int(np.ceil(max(n_states_present, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows), squeeze=False)
    kmax = 0
    stats = {}
    ax_i = 0
    for s in range(N_STATES):
        r = sorted(runs[s])
        if not r:
            continue
        ax = axes[ax_i // cols][ax_i % cols]
        ax_i += 1
        r = np.asarray(r)
        p_stay = float(P[s, s])
        geo = _geometric_pmf(p_stay, max_h)
        ax.hist(r, bins=np.arange(1, max_h + 2) - 0.5, density=True,
                alpha=0.65, label="empirical")
        ax.plot(np.arange(1, max_h + 1), geo, "r-", lw=1.5, label="geometric")
        ax.set_title(label(s), fontsize=9)
        ax.set_xlim(0, max_h)
        if ax_i % cols == 1:
            ax.set_ylabel("P(dwell)")
        if ax_i == 1:
            ax.legend(fontsize=7)
        stats[label(s)] = {
            "n_runs": len(r),
            "empirical_mean_h": float(r.mean()),
            "geometric_mean_h": float(1 / (1 - p_stay)),
            "empirical_frac_gt24h": float((r > 24).mean()),
            "geometric_frac_gt24h": float(p_stay**24),
        }
        kmax = max(kmax, len(r))
    for ax in axes.flat[ax_i:]:
        ax.axis("off")
    fig.suptitle("Dwell-time distributions vs geometric (first-order) fit", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return stats


def run_eval(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    split: str = "2016-01-01",
    alpha: float = 1.0,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    reports_dir: str | Path = "reports",
) -> dict:
    """classify -> mask night -> chronological eval -> metrics.json + figures."""
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    states = mask_night(classify(df, thresholds), lat, lon)
    metrics = evaluate(states, split, alpha)

    pairs = transition_pairs(states)
    split_ts = pd.Timestamp(split, tz="UTC")
    train = pairs[pairs.index < split_ts]
    test = pairs[pairs.index >= split_ts]
    P = fit_transition_matrix(train, alpha)
    pi = stationary_distribution(P)
    empirical = np.bincount(test["nxt"], minlength=N_STATES) / len(test)
    tvd = float(0.5 * np.abs(pi - empirical).sum())

    runs = dwell_times(states)
    dwell_stats = _fig_dwell(runs, P, out / FIGS[2])
    _fig_transition_matrix(P, out / FIGS[0])
    _fig_frequencies(empirical, pi, out / FIGS[1])

    summary = {
        "station_split": str(split_ts),
        "alpha": alpha,
        "n_states_classified": int(states.notna().sum()),
        "n_train_pairs": len(train),
        "n_test_pairs": len(test),
        "models": {k: metrics[k] for k in metrics
                   if k not in ("n_train", "n_test")},
        "stationary_vs_empirical": {
            "stationary": pi.tolist(),
            "empirical": empirical.tolist(),
            "total_variation_distance": tvd,
        },
        "dwell": dwell_stats,
        "figures": list(FIGS),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
