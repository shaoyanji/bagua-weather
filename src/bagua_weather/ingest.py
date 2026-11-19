"""DWD Climate Data Center ingestion. TODO(v0.1.0 M1): implement, see README section 'Data'."""

from __future__ import annotations

import pandas as pd


def fetch_station(station_id: str, cache_dir: str = "data/raw") -> dict[str, str]:
    """Download hourly sunshine, wind and precipitation zips for one station into cache_dir.

    Must discover file names by listing the DWD directory (do not hard-code them), cache
    downloads, and be idempotent. Returns {variable: local_path}.
    """
    raise NotImplementedError


def load_hourly(station_id: str, cache_dir: str = "data/raw") -> pd.DataFrame:
    """Merge the cached files into one UTC-indexed frame with columns
    sun_minutes, wind_ms, precip_mm. Convert DWD's -999 sentinel to NaN.
    """
    raise NotImplementedError
