"""Binarise weather into 3 bits -> 8 states, named after the bagua trigrams.

Bit order is (bottom, middle, top) = (wet, wind, sun); state = sun*4 + wind*2 + wet.
The trigram names are a naming/visualisation layer only; they carry no modelling meaning.

M5: `classify` also accepts a `bits` subset/superset for the state-count
ablation — 4 states (drop one bit) and 16 states (add the `temp` bit).
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
    """Thresholds for the bits. Defaults are starting points, tune in config."""

    sun_min_per_hour: float = 30.0  # sunshine minutes in the hour; >= this -> sunny
    wind_ms: float = 5.0  # mean wind speed m/s; >= this -> windy
    precip_mm: float = 0.1  # hourly precipitation mm; >= this -> wet
    temp_c: float = 15.0  # air temperature deg C; ~daylight median on 01443 (verified)


DEFAULT_THRESHOLDS = Thresholds()

# Canonical bit order; a `bits` selection compacts to positions 0..k-1
# preserving this order, so ("sun","wet") -> sun*2+wet and the full four
# -> sun*4+wind*2+wet+temp*8.
BIT_ORDER = ("wet", "wind", "sun", "temp")
DEFAULT_BITS = ("sun", "wind", "wet")

_BIT_COLUMN = {"sun": "sun_minutes", "wind": "wind_ms", "wet": "precip_mm", "temp": "temp_c"}
_BIT_THRESHOLD = {
    "sun": "sun_min_per_hour",
    "wind": "wind_ms",
    "wet": "precip_mm",
    "temp": "temp_c",
}


def encode(sun: bool, wind: bool, wet: bool) -> int:
    return (int(sun) << 2) | (int(wind) << 1) | int(wet)


def decode(state: int) -> tuple[bool, bool, bool]:
    if not 0 <= state < N_STATES:
        raise ValueError(f"state must be in 0..7, got {state}")
    return bool(state >> 2 & 1), bool(state >> 1 & 1), bool(state & 1)


def label(state: int) -> str:
    if state in TRIGRAMS:
        glyph, name = TRIGRAMS[state]
        return f"{glyph} {name}"
    return f"s{state}"  # ablation variants (4/16 states) have no trigram names


def classify(
    df: pd.DataFrame,
    th: Thresholds = DEFAULT_THRESHOLDS,
    bits: tuple[str, ...] = DEFAULT_BITS,
) -> pd.Series:
    """Map hourly observations to states (nullable Int8; <NA> if any input is missing).

    `bits` selects which signals enter the state: the default ("sun","wind","wet")
    reproduces the 8-state model; ("sun","wet") and ("wind","wet") are the 4-state
    drops; ("sun","wind","wet","temp") is the 16-state variant (needs temp_c).
    """
    unknown = [b for b in bits if b not in _BIT_COLUMN]
    if unknown:
        raise ValueError(f"unknown bits: {unknown}; known: {list(_BIT_COLUMN)}")
    cols = [_BIT_COLUMN[b] for b in bits]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    valid = df[cols].notna().all(axis=1)
    state = np.zeros(len(df), dtype=int)
    for i, b in enumerate(sorted(bits, key=BIT_ORDER.index)):
        bit = (df[_BIT_COLUMN[b]] >= getattr(th, _BIT_THRESHOLD[b])).to_numpy().astype(int)
        state |= bit << i
    out = pd.Series(np.where(valid, state, -1), index=df.index).astype("Int8")
    return out.mask(~valid, pd.NA).rename("state")
