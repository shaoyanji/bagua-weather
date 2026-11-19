"""M2 fit pipeline: classify -> mask night -> transition pairs -> fit -> save.

The saved matrices are for inspection and the v0.2.0 simulation path. The
evaluation path (evaluate.py) refits on the training slice only — it must not
see these matrices, which were fitted on the whole series.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import fit_contextual, fit_transition_matrix, transition_pairs
from .solar import mask_night
from .states import DEFAULT_THRESHOLDS, Thresholds, classify


def fit_pipeline(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    alpha: float = 1.0,
    min_elevation_deg: float = 5.0,
) -> dict:
    """Hourly observations -> fitted model + coverage report.

    Returns {states, pairs, P, contextual, coverage}. `contextual` maps
    context label -> matrix, plus None -> global fallback.
    """
    states = classify(df, thresholds)
    masked = mask_night(states, lat, lon, min_elevation_deg)
    pairs = transition_pairs(masked)
    if pairs.empty:
        raise ValueError("no consecutive daytime transition pairs — check data coverage")
    P = fit_transition_matrix(pairs, alpha)
    ctx = fit_contextual(pairs, alpha)
    coverage = {
        "rows": len(df),
        "classified_states": int(states.notna().sum()),
        "after_night_mask": int(masked.notna().sum()),
        "night_or_missing_dropped": int((states.notna() & masked.isna()).sum()),
        "transition_pairs": len(pairs),
        "window": [str(masked.index.min()), str(masked.index.max())],
        "alpha": alpha,
        "min_elevation_deg": min_elevation_deg,
    }
    return {"states": masked, "pairs": pairs, "P": P, "contextual": ctx, "coverage": coverage}


def save_fit(fit: dict, out_dir: str | Path) -> Path:
    """Persist matrices to <out_dir>/fit.npz and coverage to fit_report.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = {f"ctx__{label}": m for label, m in fit["contextual"].items() if label is not None}
    np.savez_compressed(out / "fit.npz", P_global=fit["P"], **npz)
    report = dict(fit["coverage"])
    report["contexts"] = sorted(k for k in fit["contextual"] if k is not None)
    (out / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return out


def load_fit(out_dir: str | Path) -> dict:
    """Read back what save_fit wrote: {P, contextual, coverage}."""
    out = Path(out_dir)
    z = np.load(out / "fit.npz")
    ctx = {k[5:]: z[k] for k in z.files if k.startswith("ctx__")}
    ctx[None] = z["P_global"]
    coverage = json.loads((out / "fit_report.json").read_text())
    return {"P": z["P_global"], "contextual": ctx, "coverage": coverage}
