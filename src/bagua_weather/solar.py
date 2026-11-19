"""Daylight masking. The sun bit is meaningless at night, so night hours are excluded."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daylight_mask(
    index: pd.DatetimeIndex, lat: float, lon: float, min_elevation_deg: float = 5.0
) -> np.ndarray:
    """True where the sun is above `min_elevation_deg`. Index must be tz-aware (UTC)."""
    from pvlib.solarposition import get_solarposition

    if index.tz is None:
        raise ValueError("index must be timezone-aware (UTC)")
    sp = get_solarposition(index, lat, lon)
    return (sp["apparent_elevation"] > min_elevation_deg).to_numpy()


def mask_night(states: pd.Series, lat: float, lon: float, min_elevation_deg: float = 5.0):
    """Set night-hour states to <NA> so no transition crosses or lands in the night."""
    keep = daylight_mask(states.index, lat, lon, min_elevation_deg)
    return states.mask(~keep, pd.NA)
