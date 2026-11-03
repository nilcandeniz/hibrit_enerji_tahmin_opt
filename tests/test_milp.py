"""MILP doğruluk testleri:  pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarh2.config import load_config
from solarh2.optimization.evaluate import grid_only_plan, realize
from solarh2.optimization.milp import solve_day_ahead
from solarh2.optimization.params import from_config
from solarh2.optimization.prices import align_prices, load_epias_csv, synthetic_ptf

ROOT = Path(__file__).resolve().parents[1]
T = 96


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs/optimization.yaml")


def day_pv(peak=8000.0):
    h = np.arange(T) / 4
    return np.clip(np.sin(np.pi * (h - 6) / 12.5), 0, None) * peak


def day_price():
    return align_prices(pd.date_range("2026-07-02", periods=T, freq="15min").to_numpy(), synthetic_ptf(5))


def test_plan_satisfies_all_constraints(cfg):
    p = from_config(cfg, 8000.0)
    pv, price = day_pv(), day_price()
    plan = solve_day_ahead(pv, price, p)
    tol = 1e-4
    assert (plan.p_grid + plan.p_ely <= pv + tol).all()                       # güç dengesi
    assert (plan.p_ely <= p.ely_max_kw * plan.u + tol).all()                  # kapasite
    assert (plan.p_ely >= p.ely_min_kw * plan.u - tol).all()                  # min yük
    assert (plan.tank >= p.tank_min_kg - tol).all() and (plan.tank <= p.tank_max_kg + tol).all()
    assert plan.tank[-1] >= p.tank_init_kg - tol                              # gün sonu stoğu
    prod = plan.p_ely * p.dt / p.kwh_per_kg
    assert np.allclose(plan.tank[1:], plan.tank[:-1] + prod - plan.sale_kg, atol=1e-3)  # tank dengesi
    starts = np.maximum(np.diff(np.concatenate([[0], plan.u])), 0)
    assert (plan.start >= starts - 1e-6).all()


def test_no_hydrogen_when_h2_worthless(cfg):
    p = from_config(cfg, 8000.0, h2_price=0.0)
    plan = solve_day_ahead(day_pv(), day_price(), p)
    assert plan.p_ely.sum() < 1e-6


def test_hydrogen_used_when_valuable(cfg):
    p = from_config(cfg, 8000.0, h2_price=400.0)
    plan = solve_day_ahead(day_pv(), day_price(), p)
    assert plan.p_ely.max() == pytest.approx(p.ely_max_kw, rel=1e-3)


def test_perfect_forecast_realizes_planned_value(cfg):
    p = from_config(cfg, 8000.0)
    pv, price = day_pv(), day_price()
    plan = solve_day_ahead(pv, price, p)
    r = realize(plan, pv, price, p)
    assert r["net_tl"] == pytest.approx(plan.objective, rel=1e-3)
    assert r["imbalance_penalty_tl"] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("bias", [0.7, 1.3])
def test_biased_forecast_never_beats_oracle(cfg, bias):
    p = from_config(cfg, 8000.0, h2_price=250.0)
    pv, price = day_pv(), day_price()
    oracle = realize(solve_day_ahead(pv, price, p), pv, price, p)["net_tl"]
    biased = realize(solve_day_ahead(pv * bias, price, p), pv, price, p)["net_tl"]
    assert biased <= oracle + 1e-3


def test_optimizer_beats_grid_only_with_valuable_h2(cfg):
    p = from_config(cfg, 8000.0, h2_price=150.0)
    pv, price = day_pv(), day_price()
    opt = realize(solve_day_ahead(pv, price, p), pv, price, p)["net_tl"]
    base = realize(grid_only_plan(pv, p), pv, price, p)["net_tl"]
    assert opt > base


def test_epias_csv_turkish_format(tmp_path):
    f = tmp_path / "ptf.csv"
    f.write_text("Tarih;Saat;PTF (TL/MWh)\n01.07.2026;00:00;\"2.300,50\"\n01.07.2026;01:00;\"1.999,00\"\n",
                 encoding="utf-8")
    s = load_epias_csv(str(f))
    assert list(s.round(2)) == [2300.5, 1999.0]
    assert s.index[1] - s.index[0] == pd.Timedelta("1h")


def test_cycle_alignment_keeps_hourly_shape():
    ptf = synthetic_ptf(3)
    times = pd.date_range("2020-06-12 21:00", periods=T, freq="15min").to_numpy()
    out = align_prices(times, ptf, "cycle")
    assert out.shape == (T,) and not np.isnan(out).any()
    assert out[0] == out[3]            # saat içinde adım-sabit
