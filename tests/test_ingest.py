"""M1 ingest tests — fixtures only, no network."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bagua_weather import ingest

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_produkt_columns_and_sentinel():
    s = ingest._read_produkt(
        FIXTURES / "sun/historical/produkt_sd_stunde_20030829_20030831_01443.txt", "SD_SO"
    )
    assert s.index.tz is not None and str(s.index.tz) == "UTC"
    assert s.loc["2003-08-30 09:00"] == 45.0
    # -999 -> NaN
    assert pd.isna(s.loc["2003-08-30 10:00"])


def test_read_produkt_missing_column_raises():
    with pytest.raises(KeyError):
        ingest._read_produkt(
            FIXTURES / "sun/historical/produkt_sd_stunde_20030829_20030831_01443.txt", "F"
        )


def test_load_hourly_merges_recent_over_historical():
    df = ingest.load_hourly("01443", cache_dir=str(FIXTURES), since=None)
    assert list(df.columns) == ["sun_minutes", "wind_ms", "precip_mm"]
    assert not df.index.duplicated().any()
    # The 2003-08-31 09:00 overlap must carry the 'recent' file's values.
    row = df.loc["2003-08-31 09:00"]
    assert row["sun_minutes"] == 55.0 and row["wind_ms"] == 9.9 and row["precip_mm"] == 0.7


def test_load_hourly_default_since_drops_pre_utc_era():
    df_all = ingest.load_hourly("01443", cache_dir=str(FIXTURES), since=None)
    df = ingest.load_hourly("01443", cache_dir=str(FIXTURES))
    assert df.index.min() >= pd.Timestamp("2003-09-01", tz="UTC")
    assert len(df) < len(df_all)


def test_load_hourly_missing_cache_raises():
    with pytest.raises(FileNotFoundError):
        ingest.load_hourly("01443", cache_dir=str(FIXTURES / "nonexistent"))


def test_extract_produkts_is_idempotent(tmp_path):
    src = FIXTURES / "sun/historical/produkt_sd_stunde_20030829_20030831_01443.txt"
    zip_path = tmp_path / "stundenwerte_SD_01443_fake_hist.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src, arcname=src.name)
        zf.writestr("Metadaten_Stationsname_01443.txt", "meta")
    out1 = ingest._extract_produkts(zip_path, tmp_path)
    assert len(out1) == 1 and out1[0].name == src.name
    mtime = out1[0].stat().st_mtime_ns
    out2 = ingest._extract_produkts(zip_path, tmp_path)
    assert out2[0].stat().st_mtime_ns == mtime  # not rewritten


def test_station_zip_discovery(tmp_path, monkeypatch):
    """Discovery lists the dir; matching is by station id, nothing hard-coded."""
    fake_listing = [
        "../",
        "stundenwerte_SD_01443_19510101_20251231_hist.zip",
        "stundenwerte_SD_00001_19510101_20251231_hist.zip",
        "BESCHREIBUNG.pdf",
    ]
    monkeypatch.setattr(ingest, "_list_hrefs", lambda url: fake_listing)
    found = ingest._station_zips("sun", "historical", "01443")
    assert found == ["stundenwerte_SD_01443_19510101_20251231_hist.zip"]


def test_fetch_station_end_to_end_offline(tmp_path, monkeypatch):
    """Full fetch path against a fake listing + local zip payload."""
    src = FIXTURES / "sun/recent/produkt_sd_stunde_20030831_20030902_01443.txt"
    payload = tmp_path / "payload.zip"
    with zipfile.ZipFile(payload, "w") as zf:
        zf.write(src, arcname=src.name)
        zf.writestr("Metadaten.txt", "m")

    monkeypatch.setattr(
        ingest,
        "_station_zips",
        lambda product, period, station_id: (
            [f"stundenwerte_XX_{station_id}_fake_{period}.zip"] if product == "sun" else []
        ),
    )
    def fake_download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload.read_bytes())
        return dest

    monkeypatch.setattr(ingest, "_download", fake_download)

    first = ingest.fetch_station("01443", cache_dir=str(tmp_path / "raw"))
    assert len(first["sun"]) == 2  # one produkt file per period
    assert all(p.endswith(src.name) for p in first["sun"])
    second = ingest.fetch_station("01443", cache_dir=str(tmp_path / "raw"))
    assert second == first  # idempotent

    df = ingest.load_hourly("01443", cache_dir=str(FIXTURES), since=None)
    assert not df.empty and df["sun_minutes"].notna().any()
    assert np.isfinite(df["wind_ms"].dropna()).all()
