import numpy as np
import pandas as pd
import pytest

from bagua_weather.states import Thresholds, classify, decode, encode, label


def test_roundtrip():
    for s in range(8):
        assert encode(*decode(s)) == s


def test_labels():
    assert label(0b111).endswith("Qian")
    assert label(0b000).endswith("Kun")
    assert label(0b101).endswith("Li")


def test_decode_rejects_out_of_range():
    with pytest.raises(ValueError):
        decode(8)


def test_classify_with_missing():
    df = pd.DataFrame(
        {"sun_minutes": [60, 0, np.nan], "wind_ms": [10, 1, 3], "precip_mm": [0, 2, 0]},
        index=pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC"),
    )
    out = classify(df, Thresholds())
    assert out.iloc[0] == 0b110  # sunny, windy, dry
    assert out.iloc[1] == 0b001  # dull, calm, wet
    assert pd.isna(out.iloc[2])


def test_classify_requires_columns():
    with pytest.raises(KeyError):
        classify(pd.DataFrame({"sun_minutes": [1]}))
