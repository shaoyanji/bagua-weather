"""M5 stretch: state-count ablation — is the 8-state space justified?

Runs the same classify -> night mask -> chronological eval for each bit
selection (all three 4-state drops, the 8-state model, and the 16-state
variant that adds the `temp` bit from TT_TU).

Comparison method: raw log loss is NOT comparable across different class
counts (a 4-class problem has a lower entropy ceiling than a 16-class one),
so the headline metric is `skill_nats = LL(climatology) - LL(markov_global)`
— how much information the previous state buys over the unconditioned
marginal, within each state space. Raw LL and top-1 are reported alongside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import evaluate
from .model import fit_transition_matrix, transition_pairs
from .solar import mask_night
from .states import BIT_ORDER, DEFAULT_THRESHOLDS, Thresholds, classify

# All three 4-state drops (not just the obvious one) — shows which bit
# carries the signal — plus the 8-state model and the 16-state +temp variant.
VARIANTS = [
    ("4  (sun x wet)", ("sun", "wet")),
    ("4  (sun x wind)", ("sun", "wind")),
    ("4  (wind x wet)", ("wind", "wet")),
    ("8  (sun x wind x wet)", ("sun", "wind", "wet")),
    ("16 (+ temp)", ("sun", "wind", "wet", "temp")),
]


def _marginal_map(from_bits: tuple[str, ...], to_bits: tuple[str, ...]) -> np.ndarray:
    """from-state -> to-state mapping when `to_bits` is a subset of `from_bits`."""
    pos_f = {b: i for i, b in enumerate(sorted(from_bits, key=BIT_ORDER.index))}
    pos_t = {b: i for i, b in enumerate(sorted(to_bits, key=BIT_ORDER.index))}
    m = np.zeros(2 ** len(from_bits), dtype=int)
    for s in range(len(m)):
        m[s] = sum(((s >> pos_f[b]) & 1) << pos_t[b] for b in to_bits)
    return m


def _score_rows(probs: np.ndarray, nxt: np.ndarray) -> dict:
    p = probs[np.arange(len(nxt)), nxt]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(p, 1e-12, None)))),
        "top1_acc": float(np.mean(probs.argmax(axis=1) == nxt)),
    }


def _cross_compare(
    fitted: dict[tuple[str, ...], dict], alpha: float
) -> list[dict]:
    """Does a richer chain predict a coarser target better than the native one?

    For each pair (big ⊃ small): on the test timestamps BOTH models share,
    score the small model natively and the big model with its predicted
    distribution marginalized onto the small state space. Identical rows ->
    a directly comparable log loss.
    """
    rows = []
    fitted_items = list(fitted.items())
    for big_bits, big in fitted_items:
        for small_bits, small in fitted_items:
            if not set(small_bits) < set(big_bits):
                continue
            common = big["test"].index.intersection(small["test"].index)
            if len(common) == 0:
                continue
            bt, st = big["test"].loc[common], small["test"].loc[common]
            m = _marginal_map(big_bits, small_bits)
            n_s = len(small["P"])
            big_probs = np.zeros((len(common), n_s))
            for j in range(big["P"].shape[1]):
                big_probs[:, m[j]] += big["P"][bt["prev"].to_numpy(), j]
            small_probs = small["P"][st["prev"].to_numpy()]
            nxt = st["nxt"].to_numpy()
            small_sc = _score_rows(small_probs, nxt)
            big_sc = _score_rows(big_probs, nxt)
            rows.append({
                "target": f"{'×'.join(small_bits)} ({len(small['P'])} states)",
                "predicted_by": f"{'×'.join(big_bits)} ({big['P'].shape[0]} states)",
                "n_test_common": len(common),
                "native_ll": small_sc["log_loss"],
                "marginalized_ll": big_sc["log_loss"],
                "delta_ll": big_sc["log_loss"] - small_sc["log_loss"],
            })
    return rows


def run_ablation(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    split: str = "2016-01-01",
    alpha: float = 1.0,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    reports_dir: str | Path = "reports",
) -> dict:
    """Evaluate every state-count variant on the same split; write ablation.json."""
    rows = []
    fitted: dict[tuple[str, ...], dict] = {}
    split_ts = pd.Timestamp(split, tz="UTC")
    for name, bits in VARIANTS:
        states = mask_night(classify(df, thresholds, bits=bits), lat, lon)
        n_states = 2 ** len(bits)
        m = evaluate(states, split, alpha, n_states)
        clim_ll = m["climatology"]["log_loss"]
        glob_ll = m["markov_global"]["log_loss"]
        row = {
            "variant": name,
            "bits": list(bits),
            "n_states": n_states,
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "climatology_ll": clim_ll,
            "markov_global_ll": glob_ll,
            "markov_contextual_ll": m["markov_contextual"]["log_loss"],
            "markov_global_top1": m["markov_global"]["top1_acc"],
            "skill_nats": clim_ll - glob_ll,
            "rel_ll": glob_ll / clim_ll,
        }
        if "gbm" in m:
            row["gbm_ll"] = m["gbm"]["log_loss"]
            row["gbm_top1"] = m["gbm"]["top1_acc"]
            row["gbm_skill_nats"] = clim_ll - row["gbm_ll"]
        rows.append(row)
        pairs = transition_pairs(states)
        fitted[bits] = {
            "P": fit_transition_matrix(pairs[pairs.index < split_ts], alpha, n_states),
            "test": pairs[pairs.index >= split_ts],
        }

    cross = _cross_compare(fitted, alpha)
    out = {
        "split": str(split_ts),
        "alpha": alpha,
        "variants": rows,
        "cross_comparisons": cross,
    }
    path = Path(reports_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "ablation.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def verdict(rows: list[dict], cross: list[dict] | None = None) -> str:
    """One honest sentence on whether 8 states is justified."""
    base = next(r for r in rows if r["n_states"] == 8)
    low = max((r for r in rows if r["n_states"] < 8), key=lambda r: r["skill_nats"])
    high = next(r for r in rows if r["n_states"] == 16)
    s = (
        f"skill: 8-state {base['skill_nats']:.3f} nats; "
        f"best 4-state ({low['variant'].strip()}) {low['skill_nats']:.3f} "
        f"({low['skill_nats'] - base['skill_nats']:+.3f}); "
        f"16-state {high['skill_nats']:.3f} ({high['skill_nats'] - base['skill_nats']:+.3f})"
    )
    if cross:
        x8 = next(r for r in cross if "8 states" in r["target"]
                  and "16 states" in r["predicted_by"])
        s += (f"; on the 8-state target itself, the 16-chain marginalized scores "
              f"{x8['marginalized_ll']:.3f} vs native {x8['native_ll']:.3f} "
              f"({x8['delta_ll']:+.3f})")
    return s
