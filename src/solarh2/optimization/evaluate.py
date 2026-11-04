"""Planı gerçekleşen üretime karşı değerlendirir (tahmin hatasının TL karşılığı).

Plan TAHMİNLE ve yalnızca PTF ile yapılır (SMF gün öncesinde bilinmez). Gerçekleşme gerçek
PV ile ölçülür ve sapmalar Türkiye dengesizlik kuralıyla fiyatlanır:

    negatif dengesizlik (eksik) fiyatı  = (1 + k) * max(PTF, SMF)     k = %3
    pozitif dengesizlik (fazla) fiyatı  = (1 - k) * min(PTF, SMF)

SMF verilmezse SMF = PTF alınır (eski, iyimser davranış).

Gerçek zamanlı davranış (`recourse`):
  * False (plana sadık): elektrolizör planlanan gücü korur (PV yetmezse düşer, min yükün
    altındaysa kapanır); tüm sapma şebekeye yansır.
  * True (gerçek zamanlı düzeltme): operatör o saatin marjinal değerlerine bakar —
      - PV açığı varsa ve eksik enerjinin bedeli (negatif dengesizlik fiyatı) H2'nin
        kWh başına değerinden yüksekse, elektrolizör kısılarak şebeke taahhüdü korunur;
      - PV fazlası varsa ve H2 değeri pozitif dengesizlik fiyatından yüksekse, ÇALIŞAN
        elektrolizör tank/kapasite elverdiğince yukarı çekilir.
    Kapalı elektrolizör gerçek zamanda açılmaz (başlatma kararı gün öncesinde verilir).
    Bu, hibrit sistemin "dengesizlik tamponu" olarak ek değerini ölçer.

Muhasebe: taahhüt PTF'den gelir yazılır; eksik, negatif dengesizlik fiyatıyla geri alınır;
fazla, pozitif dengesizlik fiyatıyla satılır. Gün sonu stoğu başlangıcın altına düşerse eksik
kg H2 fiyatıyla cezalandırılır (aksi halde başlangıç stoğu bedavaya satılıp kötü tahmin
Oracle'ı yenebilirdi).
"""
from __future__ import annotations

import numpy as np

from solarh2.optimization.milp import Plan
from solarh2.optimization.params import Params


def imbalance_prices(ptf: np.ndarray, smf: np.ndarray | None, p: Params):
    """(negatif dengesizlik fiyatı, pozitif dengesizlik fiyatı) — TL/MWh."""
    smf = ptf if smf is None else smf
    neg = p.short_buy_factor * np.maximum(ptf, smf)
    pos = p.long_sell_factor * np.minimum(ptf, smf)
    return neg, pos


def realize(plan: Plan, pv_actual_kw: np.ndarray, price_tl_mwh: np.ndarray, p: Params,
            smf_tl_mwh: np.ndarray | None = None, recourse: bool = False) -> dict:
    T = len(pv_actual_kw)
    pv = np.clip(pv_actual_kw, 0, None)
    pk = price_tl_mwh / 1000.0                               # TL/kWh
    neg, pos = imbalance_prices(price_tl_mwh, smf_tl_mwh, p)
    neg_k, pos_k = neg / 1000.0, pos / 1000.0
    v_h2 = p.h2_price * p.kg_per_kwh - p.var_cost            # 1 kWh'ın H2 olarak net değeri

    tank = p.tank_init_kg
    extra_kg = 0.0                                           # düzeltmeyle fazladan üretilip henüz satılmayan
    sale_room = np.maximum(p.sale_max_kg_per_step - plan.sale_kg, 0.0)
    rev_grid = rev_h2 = cost_imb = cost_ops = 0.0
    h2_prod = short_kwh = surplus_kwh = imb_pen = absorbed_kwh = curtail_kwh = 0.0
    starts, on_prev = 0, 0

    for t in range(T):
        committed = plan.p_grid[t]
        planned_on = plan.u[t] > 0.5
        ely = min(plan.p_ely[t], pv[t]) if planned_on else 0.0

        if recourse and planned_on:
            free = pv[t] - ely
            if free < committed and neg_k[t] > v_h2:
                # açığı kapatmak için elektrolizörü kıs (min yükün altına inemezse kapanır)
                ely = max(pv[t] - committed, 0.0)
            elif free > committed and v_h2 > pos_k[t]:
                # fazlayı tank/kapasite elverdiğince H2'ye çevir
                room_kg = p.tank_max_kg - tank + plan.sale_kg[t]
                room_kw = max(room_kg, 0.0) / (p.dt * p.kg_per_kwh)
                sellable_kg = max(sale_room[t:].sum() - extra_kg, 0.0)   # günün kalanında satılabilir
                sellable_kw = sellable_kg / (p.dt * p.kg_per_kwh)
                up = min(free - committed, p.ely_max_kw - ely, max(room_kw - ely, 0.0), sellable_kw)
                absorbed_kwh += max(up, 0.0) * p.dt
                ely += max(up, 0.0)
        if ely < p.ely_min_kw - 1e-9:
            ely = 0.0

        on = int(ely > 0)
        starts += int(on and not on_prev)
        on_prev = on

        free = pv[t] - ely
        delivered = min(committed, free)
        shortfall = committed - delivered                    # kW
        # taahhüt üstü fazla; şebeke bağlantı limitini aşan kısım satılamaz -> kısıntı
        surplus = min(max(free - committed, 0.0), max(p.grid_limit_kw - committed, 0.0))
        curtail_kwh += (max(free - committed, 0.0) - surplus) * p.dt

        rev_grid += committed * p.dt * pk[t]
        cost_imb += shortfall * p.dt * neg_k[t]
        rev_grid += surplus * p.dt * pos_k[t]
        short_kwh += shortfall * p.dt
        surplus_kwh += surplus * p.dt
        # sapmanın, aynı enerjiyi PTF'den teslim etmeye göre net maliyeti
        imb_pen += (shortfall * (neg_k[t] - pk[t]) + surplus * (pk[t] - pos_k[t])) * p.dt

        produced = ely * p.dt * p.kg_per_kwh
        h2_prod += produced
        if recourse:
            planned_prod = (min(plan.p_ely[t], pv[t]) if planned_on else 0.0) * p.dt * p.kg_per_kwh
            extra_kg += max(produced - planned_prod, 0.0)
        tank = min(tank + produced, p.tank_max_kg)
        target = plan.sale_kg[t] + (extra_kg if recourse else 0.0)
        sold = min(target, p.sale_max_kg_per_step, tank)
        if recourse:
            extra_kg = max(extra_kg - max(sold - plan.sale_kg[t], 0.0), 0.0)
        tank -= sold
        rev_h2 += sold * p.h2_price
        cost_ops += ely * p.dt * p.var_cost
    cost_ops += starts * p.startup_cost
    stock_penalty = max(p.tank_init_kg - tank, 0.0) * p.h2_price
    net = rev_grid + rev_h2 - cost_imb - cost_ops - stock_penalty
    return dict(net_tl=net, rev_grid=rev_grid, rev_h2=rev_h2, cost_imbalance=cost_imb,
                cost_ops=cost_ops, imbalance_penalty_tl=imb_pen, stock_penalty_tl=stock_penalty,
                h2_kg=h2_prod, shortfall_kwh=short_kwh, surplus_kwh=surplus_kwh,
                absorbed_kwh=absorbed_kwh, curtail_kwh=curtail_kwh, starts=starts, end_tank_kg=tank)


def grid_only_plan(pv_forecast_kw: np.ndarray, p: Params) -> Plan:
    """Referans politika: tüm öngörülen üretimi şebekeye taahhüt et, H2 yok."""
    T = len(pv_forecast_kw)
    z = np.zeros(T)
    return Plan(np.minimum(np.clip(pv_forecast_kw, 0, None), p.grid_limit_kw), z, z, z, z,
                np.full(T + 1, p.tank_init_kg), 0.0, "Optimal")
