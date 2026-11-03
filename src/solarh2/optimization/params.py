"""Optimizasyon parametreleri (YAML -> tip güvenli dataclass, türetilmiş büyüklükler dahil)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    dt: float                 # h
    grid_limit_kw: float
    ely_max_kw: float
    ely_min_kw: float
    kwh_per_kg: float         # HHV / verim  (spesifik tüketim)
    startup_cost: float       # TL / başlatma
    var_cost: float           # TL / kWh
    tank_max_kg: float
    tank_min_kg: float
    tank_init_kg: float
    terminal_ge_initial: bool
    sale_max_kg_per_step: float
    h2_price: float           # TL / kg
    short_buy_factor: float
    long_sell_factor: float

    @property
    def kg_per_kwh(self) -> float:
        return 1.0 / self.kwh_per_kg


def from_config(cfg: dict, capacity_kw: float, h2_price: float | None = None) -> Params:
    e, t, pr, im = cfg["electrolyzer"], cfg["tank"], cfg["prices"], cfg["imbalance"]
    dt = cfg["timing"]["dt_hours"]
    ely_max = e["p_max_ratio"] * capacity_kw
    kwh_per_kg = e["hhv_kwh_per_kg"] / e["efficiency_hhv"]
    full_rate_kg_h = ely_max / kwh_per_kg
    tank_max = t["max_hours_full_load"] * full_rate_kg_h
    return Params(
        dt=dt,
        grid_limit_kw=cfg["plant"]["grid_limit_ratio"] * capacity_kw,
        ely_max_kw=ely_max,
        ely_min_kw=e["p_min_ratio"] * ely_max,
        kwh_per_kg=kwh_per_kg,
        startup_cost=e["startup_cost_tl"],
        var_cost=e["var_cost_tl_per_kwh"],
        tank_max_kg=tank_max,
        tank_min_kg=t["min_fraction"] * tank_max,
        tank_init_kg=t["initial_fraction"] * tank_max,
        terminal_ge_initial=t["terminal_ge_initial"],
        sale_max_kg_per_step=t["max_sale_hours_ratio"] * full_rate_kg_h * dt,
        h2_price=pr["h2_tl_per_kg"] if h2_price is None else h2_price,
        short_buy_factor=im["short_buy_factor"],
        long_sell_factor=im["long_sell_factor"],
    )
