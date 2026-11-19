"""Binarise weather into 3 bits -> 8 states, named after the bagua trigrams.

Bit order is (bottom, middle, top) = (sun, wind, wet); state = bottom*4 + middle*2 + top.
The trigram names are a naming/visualisation layer only; they carry no modelling meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_STATES = 8

# Fu Xi ("earlier heaven") sequence, lines read bottom -> top.
TRIGRAMS = {
    0b111: ("☰", "Qian"),
    0b110: ("☱", "Dui"),
    0b101: ("☲", "Li"),
    0b100: ("☳", "Zhen"),
    0b011: ("☴", "Xun"),
    0b010: ("☵", "Kan"),
    0b001: ("☶", "Gen"),
    0b000: ("☷", "Kun"),
}


@dataclass(frozen=True)
class Thresholds:
    """Thresholds for the three bits. Defaults are starting points, tune in config."""

    sun_min_per_hour: float = 30.0  # sunshine minutes in the hour; >= this -> sunny
    wind_ms: float = 5.0  # mean wind speed m/s; >= this -> windy
    precip_mm: float = 0.1  # hourly precipitation mm; >= this -> wet


def encode(sun: bool, wind: bool, wet: bool) -> int:
    return (int(sun) << 2) | (int(wind) << 1) | int(wet)


def decode(state: int) -> tuple[bool, bool, bool]:
    if not 0 <= state < N_STATES:
        raise ValueError(f"state must be in 0..7, got {state}")
    return bool(state >> 2 & 1), bool(state >> 1 & 1), bool(state & 1)


def label(state: int) -> str:
    glyph, name = TRIGRAMS[state]
    return f"{glyph} {name}"


def classify(df: pd.DataFrame, th: Thresholds = Thresholds()) -> pd.Series:
    """Map hourly observations to states (nullable Int8; <NA> if any input is missing).

    Expects columns: sun_minutes, wind_ms, precip_mm (DWD -999 already converted to NaN).
    """
    cols = ["sun_minutes", "wind_ms", "precip_mm"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    valid = df[cols].notna().all(axis=1)
    sun = (df["sun_minutes"] >= th.sun_min_per_hour).astype(int)
    wind = (df["wind_ms"] >= th.wind_ms).astype(int)
    wet = (df["precip_mm"] >= th.precip_mm).astype(int)
    state = sun * 4 + wind * 2 + wet
    out = pd.Series(np.where(valid, state, -1), index=df.index).astype("Int8")
    return out.mask(~valid, pd.NA).rename("state")
