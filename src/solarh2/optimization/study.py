"""Tahmin -> MILP -> gerçekleşme değerlendirmesi (yeniden kullanılabilir çekirdek).

`run_optimization.py`, `forecast_value_sweep.py` ve ileride Streamlit arayüzü aynı
fonksiyonları kullanır; böylece tüm sonuçlar tek bir tanımdan üretilir.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from solarh2.optimization.evaluate import grid_only_plan, realize
from solarh2.optimization.milp import solve_day_ahead
from solarh2.optimization.params import Params, from_config
from solarh2.optimization.prices import align_prices, load_epias_series, synthetic_ptf

ORACLE = "Oracle (mükemmel tahmin)"


@dataclass
class StudyContext:
    sel: np.ndarray               # seçilen pencere indeksleri
    capacity: float
    params: Params
    ptf: dict                     # pencere -> (T,) TL/MWh
    smf: dict | None              # pencere -> (T,) TL/MWh ya da None (SMF = PTF)
    price_source: str


def load_prices(cfg: dict):
    pr = cfg["prices"]
    ptf = load_epias_series(pr["epias_csv"], "ptf") if pr.get("epias_csv") else synthetic_ptf()
    smf = load_epias_series(pr["smf_csv"], "smf") if pr.get("smf_csv") else None
    src = ("EPİAŞ PTF" if pr.get("epias_csv") else "SENTETİK PTF") + \
          (" + EPİAŞ SMF" if smf is not None else " (SMF = PTF varsayımı)")
    return ptf, smf, src


def select_windows(times: np.ndarray, st: dict) -> np.ndarray:
    sel = np.arange(0, len(times), st.get("window_stride", 1))
    if st.get("origin_hours") is not None:
        hrs = pd.DatetimeIndex(times[:, 0]).hour.to_numpy()
        sel = np.array([i for i in sel if hrs[i] in st["origin_hours"]], dtype=int)
    if len(sel) == 0:
        raise ValueError("Seçilen pencere yok (origin_hours/stride'ı kontrol edin)")
    return sel


def build_context(times: np.ndarray, y_true: np.ndarray, cfg: dict, prices=None) -> StudyContext:
    sel = select_windows(times, cfg["study"])
    cap_cfg = cfg["plant"]["capacity_kw"]
    capacity = float(y_true.max()) if cap_cfg == "auto" else float(cap_cfg)
    ptf_s, smf_s, src = prices or load_prices(cfg)
    mode = cfg["prices"]["align_mode"]
    day0 = pd.Timestamp(times.min())
    ptf = {i: align_prices(times[i], ptf_s, mode, day0) for i in sel}
    smf = None if smf_s is None else {i: align_prices(times[i], smf_s, mode, day0) for i in sel}
    return StudyContext(sel, capacity, from_config(cfg, capacity), ptf, smf, src)


def run_policy(ctx: StudyContext, y_true: np.ndarray, forecast: np.ndarray | None,
               params: Params | None = None, grid_only: bool = False, recourse: bool = False,
               time_limit_s: float = 60) -> pd.DataFrame:
    """forecast=None -> Oracle (gerçek üretim tahmin olarak kullanılır)."""
    p = params or ctx.params
    rows = []
    for i in ctx.sel:
        fc = y_true[i] if forecast is None else forecast[i]
        plan = grid_only_plan(fc, p) if grid_only else solve_day_ahead(fc, ctx.ptf[i], p, time_limit_s)
        smf = None if ctx.smf is None else ctx.smf[i]
        r = realize(plan, y_true[i], ctx.ptf[i], p, smf, recourse=recourse)
        r["window"] = int(i)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(results: dict[str, pd.DataFrame], oracle_key: str = ORACLE) -> pd.DataFrame:
    oracle = results[oracle_key]["net_tl"].mean()
    return pd.DataFrame({
        name: {"Net gelir (TL/gün)": d["net_tl"].mean(),
               "Oracle'a göre kayıp (TL/gün)": oracle - d["net_tl"].mean(),
               "Oracle'a göre kayıp (%)": 100 * (1 - d["net_tl"].mean() / oracle),
               "H2 (kg/gün)": d["h2_kg"].mean(),
               "Dengesizlik net cezası (TL/gün)": d["imbalance_penalty_tl"].mean(),
               "Eksik enerji (kWh/gün)": d["shortfall_kwh"].mean(),
               "Fazla enerji (kWh/gün)": d["surplus_kwh"].mean(),
               "H2'ye emilen sapma (kWh/gün)": d["absorbed_kwh"].mean(),
               "Başlatma (adet/gün)": d["starts"].mean()}
        for name, d in results.items()}).T


def evaluate_forecasts(ctx: StudyContext, y_true: np.ndarray, forecasts: dict[str, np.ndarray],
                       recourse_modes=(False,), grid_only_for: str | None = None,
                       time_limit_s: float = 60):
    """Her tahmin edici (ve her gerçek zamanlı mod) için MILP planı + gerçekleşme."""
    res = {ORACLE: run_policy(ctx, y_true, None, time_limit_s=time_limit_s)}
    for name, fc in forecasts.items():
        for rc in recourse_modes:
            label = f"MILP + {name}" + (" + RT düzeltme" if rc else "")
            res[label] = run_policy(ctx, y_true, fc, recourse=rc, time_limit_s=time_limit_s)
    if grid_only_for is not None:
        res[f"Sadece şebeke ({grid_only_for} tahmini)"] = run_policy(
            ctx, y_true, forecasts[grid_only_for], grid_only=True)
        res["Sadece şebeke (oracle)"] = run_policy(ctx, y_true, None, grid_only=True)
    return summarize(res), res
