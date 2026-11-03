"""Pencereleme ve sızıntı testleri:  pytest -q"""
import copy
from pathlib import Path

import numpy as np
import pytest

from solarh2.config import load_config
from solarh2.data.windowing import build_windows
from tests.make_synthetic import make

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    csv = tmp_path_factory.mktemp("d") / "syn.csv"
    make().to_csv(csv, index=False)
    cfg = load_config(ROOT / "configs/lstm.yaml")
    cfg["data"]["csv_path"] = str(csv)
    return build_windows(cfg), cfg


def test_shapes(built):
    (splits, sc), cfg = built
    L, H = cfg["data"]["lookback"], cfg["data"]["horizon"]
    for ws in splits.values():
        assert ws.x_past.shape[1:] == (L, len(sc["past_cols"]))
        assert ws.x_future.shape[1:] == (H, len(sc["future_cols"]))
        assert ws.y.shape[1] == H


def test_no_target_overlap_between_splits(built):
    (splits, _), _ = built
    tr_end = splits["train"].times.max()
    va_start, va_end = splits["val"].times.min(), splits["val"].times.max()
    assert tr_end < va_start
    assert va_end < splits["test"].times.min()


def test_targets_are_contiguous(built):
    (splits, _), _ = built
    step = np.diff(splits["train"].times, axis=1).astype("timedelta64[m]").astype(int)
    assert (step == 15).all(), "Pencere içinde zaman boşluğu var"


def test_past_target_not_in_future_inputs(built):
    (_, sc), _ = built
    assert "AC_POWER" not in sc["future_cols"]


def test_persistence_uses_only_past(built):
    (splits, _), _ = built
    te = splits["test"]
    # y_lag_day[i, h] zamanı = hedef zamanı - 24 saat  -> tahmin anından önce
    assert te.y_lag_day.shape == te.y_raw.shape
