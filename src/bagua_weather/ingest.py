"""DWD Climate Data Center ingestion.

Hourly station observations live under
  {DWD_BASE}/<product>/<period>/stundenwerte_<letter>_<station>_<range>.zip
with <period> in {historical, recent}. File names are discovered by listing the
directory — never hard-coded. Each zip contains a produkt_*.txt data file plus
Metadaten_* documentation. Downloads are cached under cache_dir and the fetch
is idempotent: existing files are not re-downloaded.

Verified against the DESCRIPTION PDFs and the station-01443 metadata
(2026-11-19):

- Columns: sun -> SD_SO (min/h), wind -> F (m/s; direction is column D),
  precipitation -> R1 (mm; RS_IND is the 0/1 indicator). MESS_DATUM is
  YYYYMMDDHH, sentinel -999.
- Timestamps are NOT uniformly UTC. Freiburg (01443) sun data is WOZ (local
  apparent solar time) before 2002-07-01, wind is MEZ before 2003-09-01,
  precipitation is UTC throughout. The all-UTC overlap era begins
  2003-09-01; earlier rows cannot be interpreted as UTC, so load_hourly
  takes a `since` cutoff instead of silently misreading the clock.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

DWD_BASE = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
)
PERIODS = ("historical", "recent")

# product dir -> (value column, our column name)
PRODUCTS = {
    "sun": ("SD_SO", "sun_minutes"),
    "wind": ("F", "wind_ms"),
    "precipitation": ("R1", "precip_mm"),
}

# Verified 2026-11-19: station 01443 is UTC across all three products only from
# 2003-09-01 (sun: WOZ earlier; wind: MEZ earlier). Default analysis window.
UTC_ERA_START = "2003-09-01"

_MISSING = -999.0


def _list_hrefs(url: str) -> list[str]:
    """All href targets in a DWD directory index."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return re.findall(r'href="([^"]+)"', resp.text)


def _station_zips(product: str, period: str, station_id: str) -> list[str]:
    """Zip file names for one station in one product/period directory."""
    url = f"{DWD_BASE}/{product}/{period}/"
    wanted = f"_{station_id}_"
    return sorted(
        h
        for h in _list_hrefs(url)
        if h.endswith(".zip") and wanted in h
    )


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _extract_produkts(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract produkt_*.txt members; skip existing. Returns the paths."""
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith("produkt_") or not name.endswith(".txt"):
                continue
            dest = dest_dir / name
            if not dest.exists():
                with zf.open(name) as fh:
                    dest.write_bytes(fh.read())
            out.append(dest)
    return out


def fetch_station(station_id: str, cache_dir: str = "data/raw") -> dict[str, list[str]]:
    """Download + extract hourly sun, wind and precipitation for one station.

    Discovers zip names by listing the DWD directories (historical + recent),
    caches everything under cache_dir, and is idempotent — a second call
    issues directory listings only (no re-downloads). Returns
    {product: [produkt txt paths]}.
    """
    cache = Path(cache_dir)
    found: dict[str, list[str]] = {}
    for product in PRODUCTS:
        paths: list[Path] = []
        for period in PERIODS:
            period_dir = cache / product / period
            for zip_name in _station_zips(product, period, station_id):
                zip_path = _download(
                    f"{DWD_BASE}/{product}/{period}/{zip_name}", period_dir / zip_name
                )
                paths.extend(_extract_produkts(zip_path, period_dir))
        found[product] = [str(p) for p in sorted(set(paths))]
    return found


def _read_produkt(path: Path, column: str) -> pd.Series:
    """One produkt file -> tz-aware (UTC) Series indexed by MESS_DATUM.

    Timestamps are read as UTC unconditionally; whether that is true depends
    on the era — callers must apply the `since` cutoff (see module docstring).
    """
    df = pd.read_csv(
        path,
        sep=";",
        skipinitialspace=True,
        na_values=[_MISSING],
        dtype={"STATIONS_ID": str, "MESS_DATUM": str},
    )
    df = df.drop(columns=[c for c in df.columns if c.strip() == "eor"], errors="ignore")
    df.columns = [c.strip() for c in df.columns]
    if column not in df.columns:
        raise KeyError(f"{path.name}: expected column {column!r}, have {list(df.columns)}")
    ts = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", utc=True)
    s = pd.Series(pd.to_numeric(df[column], errors="coerce").to_numpy(), index=ts)
    return s.rename(column)


def _product_frame(cache: Path, product: str, station_id: str) -> pd.Series:
    """Merge historical + recent produkt files for one product.

    On overlapping timestamps the 'recent' file wins (it carries the latest
    quality control); within a file the first occurrence wins.
    """
    col, _ = PRODUCTS[product]
    recent, historical = [], []
    for p in _produkt_files(cache, product, station_id):
        s = _read_produkt(p, col)
        s = s[~s.index.duplicated(keep="first")]
        (recent if p.parent.name == "recent" else historical).append(s)
    series = pd.concat(historical + recent)
    # Prefer 'recent' values on boundary overlap: it was concatenated last,
    # so keep='last' picks it. Within-era duplicates keep the first file.
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _produkt_files(cache: Path, product: str, station_id: str) -> list[Path]:
    """produkt_*.txt for a station under cache/<product>/<period>/, both id forms."""
    pats = [f"{product}/*/produkt_*_{station_id}.txt", f"{product}/*/produkt_*_{station_id.lstrip('0')}.txt"]
    files = {p for pat in pats for p in cache.glob(pat)}
    return sorted(files)


def load_hourly(
    station_id: str, cache_dir: str = "data/raw", since: str | None = UTC_ERA_START
) -> pd.DataFrame:
    """Merge cached produkt files into one UTC-indexed hourly frame.

    Columns: sun_minutes, wind_ms, precip_mm. -999 -> NaN (handled at read).
    `since` is an inclusive UTC cutoff applied *before* merging; the default
    is the verified all-UTC era start for Freiburg (see module docstring).
    Pass since=None only if you know your station is UTC from genesis.
    """
    cache = Path(cache_dir)
    cols = {}
    for product, (col, out_name) in PRODUCTS.items():
        files = _produkt_files(cache, product, station_id)
        if not files:
            raise FileNotFoundError(
                f"no produkt files for {product}/{station_id} under {cache} — "
                "run `bagua fetch` first"
            )
        cols[out_name] = _product_frame(cache, product, station_id)
    df = pd.concat(cols, axis=1)
    df.index.name = "timestamp"
    if since is not None:
        df = df[df.index >= pd.Timestamp(since, tz="UTC")]
    return df.sort_index()
