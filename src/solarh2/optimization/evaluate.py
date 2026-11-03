"""Planı gerçekleşen üretime karşı değerlendirir (tahmin hatasının TL karşılığı).

İşleyiş: plan TAHMİNLE yapılır; gerçekleşme gerçek PV ile ölçülür.
  * Elektrolizör önceliklidir: p_ely_gerçek = min(plan, mevcut); min yükün altına
    düşerse açılamaz (0).
  * Şebeke taahhüdü gerçek PV'ye sığmazsa eksik enerji dengesizlikte
    (fiyat x short_buy_factor) ile kapatılır.
  * Taahhüt üstü fazla enerji (fiyat x long_sell_factor) ile satılır.
  * H2 satışı, gerçek tank stoğuyla sınırlıdır.
  * Gün sonu stoğu başlangıç stoğunun altına düşerse (plan bunu garanti ederdi, gerçekleşme
    garanti etmez) eksik miktar H2 satış fiyatıyla cezalandırılır. Aksi halde başlangıç
    stoğu "bedavaya" satılıp kötü tahmin iyi görünebilirdi (Oracle'ı yenerdi).
  * `imbalance_penalty_tl`: sapmanın net maliyeti = (buy-1)*fiyat*eksik + (1-sell)*fiyat*fazla.
    NOT: SMF=PTF varsayılır; gerçekte SMF-PTF farkı bu cezayı büyütür (alt sınırdır).
"""
from __future__ import annotations

import numpy as np

from solarh2.optimization.milp import Plan
from solarh2.optimization.params import Params


def realize(plan: Plan, pv_actual_kw: np.ndarray, price_tl_mwh: np.ndarray, p: Params) -> dict:
    T = len(pv_actual_kw)
    pv = np.clip(pv_actual_kw, 0, None)
    pk = price_tl_mwh / 1000.0
    tank = p.tank_init_kg
    rev_grid = rev_h2 = cost_imb = cost_ops = 0.0
    h2_prod = short_kwh = imb_pen = 0.0
    starts, on_prev = 0, 0

    for t in range(T):
        ely = min(plan.p_ely[t], pv[t])
        ely = ely if ely >= p.ely_min_kw - 1e-9 and plan.u[t] > 0.5 else 0.0
        on = int(ely > 0)
        starts += int(on and not on_prev)
        on_prev = on

        committed = plan.p_grid[t]
        free = pv[t] - ely
        delivered = min(committed, free)
        shortfall = committed - delivered                    # kW
        surplus = max(free - committed, 0.0)                 # kW (taahhüt üstü)

        rev_grid += committed * p.dt * pk[t]                 # taahhüt gelir olarak yazılır
        cost_imb += shortfall * p.dt * pk[t] * p.short_buy_factor
        rev_grid += surplus * p.dt * pk[t] * p.long_sell_factor
        short_kwh += shortfall * p.dt
        imb_pen += (shortfall * (p.short_buy_factor - 1) + surplus * (1 - p.long_sell_factor)) * p.dt * pk[t]

        produced = ely * p.dt * p.kg_per_kwh
        h2_prod += produced
        tank = min(tank + produced, p.tank_max_kg)
        sold = min(plan.sale_kg[t], tank)
        tank -= sold
        rev_h2 += sold * p.h2_price
        cost_ops += ely * p.dt * p.var_cost
    cost_ops += starts * p.startup_cost
    stock_penalty = max(p.tank_init_kg - tank, 0.0) * p.h2_price
    net = rev_grid + rev_h2 - cost_imb - cost_ops - stock_penalty
    return dict(net_tl=net, rev_grid=rev_grid, rev_h2=rev_h2, cost_imbalance=cost_imb,
                cost_ops=cost_ops, imbalance_penalty_tl=imb_pen, stock_penalty_tl=stock_penalty, h2_kg=h2_prod, shortfall_kwh=short_kwh, starts=starts,
                end_tank_kg=tank)


def grid_only_plan(pv_forecast_kw: np.ndarray, p: Params) -> Plan:
    """Referans politika: tüm öngörülen üretimi şebekeye taahhüt et, H2 yok."""
    T = len(pv_forecast_kw)
    z = np.zeros(T)
    return Plan(np.minimum(np.clip(pv_forecast_kw, 0, None), p.grid_limit_kw), z, z, z, z,
                np.full(T + 1, p.tank_init_kg), 0.0, "Optimal")
