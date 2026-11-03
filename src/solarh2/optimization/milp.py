"""Gün öncesi karar motoru: MILP (PuLP).

Gemini taslağına göre bilinçli farklar
--------------------------------------
1. Zaman adımı dt açıkça yer alır. Güç (kW) ile enerji (kWh) ve kütle (kg) karıştırılmaz:
       enerji = P * dt,   m_H2 = P_ely * dt / (HHV / eta)
2. Hidrojen geliri ÜRETİMDEN değil SATIŞTAN (d_t) doğar. Taslakta hem "üretim x fiyat"
   hem de "tank dengesi" vardı; bu çifte sayıma ve tankın anlamsızlaşmasına yol açar.
3. "u(t)*C_deg" (açık saat başına maliyet) yerine BAŞLATMA değişkeni y_t ve
   başlatma maliyeti kullanılır; bozunma enerjiye bağlı değişken maliyet olarak girer.
4. Tank için gün sonu koşulu S_T >= S_0: aksi halde model günün sonunda stoku bedavaya
   satıp "kazanç" üretir (ufuk-sonu etkisi).

Çözücü: HiGHS varsa HiGHS, yoksa CBC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from solarh2.optimization.params import Params


@dataclass
class Plan:
    p_grid: np.ndarray    # kW   gün öncesi şebeke taahhüdü
    p_ely: np.ndarray     # kW
    u: np.ndarray         # {0,1}
    start: np.ndarray     # {0,1}
    sale_kg: np.ndarray   # kg / adım
    tank: np.ndarray      # kg, uzunluk T+1
    objective: float
    status: str


def get_solver(time_limit_s: float = 60, gap: float = 1e-4):
    avail = pulp.listSolvers(onlyAvailable=True)
    if "HiGHS" in avail:
        return pulp.HiGHS(msg=False, timeLimit=time_limit_s, gapRel=gap)
    if "PULP_CBC_CMD" in avail:
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s, gapRel=gap)
    raise RuntimeError("Hiçbir MILP çözücüsü bulunamadı: `pip install highspy` deneyin.")


def solve_day_ahead(pv_kw: np.ndarray, price_tl_mwh: np.ndarray, p: Params,
                    time_limit_s: float = 60) -> Plan:
    """pv_kw: öngörülen kullanılabilir PV gücü (T,). price_tl_mwh: PTF (T,)."""
    T = len(pv_kw)
    assert len(price_tl_mwh) == T
    pv = np.clip(pv_kw, 0, None)
    price_kwh = price_tl_mwh / 1000.0           # TL/kWh
    R = range(T)

    m = pulp.LpProblem("pv_grid_h2", pulp.LpMaximize)
    # LpVariable.dicts PuLP 4.x'te yok -> sürümden bağımsız, tek tek oluşturulur
    def var(name, idx, lo=None, hi=None, cat="Continuous"):
        return {i: pulp.LpVariable(f"{name}_{i}", lo, hi, cat=cat) for i in idx}

    g = var("p_grid", R, 0, p.grid_limit_kw)
    e = var("p_ely", R, 0, p.ely_max_kw)
    u = var("u", R, cat="Binary")
    y = var("start", R, cat="Binary")
    d = var("sale_kg", R, 0, p.sale_max_kg_per_step)
    s = var("tank", range(T + 1), p.tank_min_kg, p.tank_max_kg)

    # Amaç: şebeke geliri + H2 satışı - başlatma - değişken maliyet
    m += pulp.lpSum(
        g[t] * p.dt * price_kwh[t] + d[t] * p.h2_price
        - y[t] * p.startup_cost - e[t] * p.dt * p.var_cost
        for t in R
    )

    m += s[0] == p.tank_init_kg
    for t in R:
        m += g[t] + e[t] <= float(pv[t])                       # güç dengesi (fazlası kısıntı)
        m += e[t] <= p.ely_max_kw * u[t]                       # kapasite
        m += e[t] >= p.ely_min_kw * u[t]                       # minimum yük
        m += y[t] >= u[t] - (u[t - 1] if t > 0 else 0)         # başlatma tespiti (u_{-1}=0)
        m += s[t + 1] == s[t] + e[t] * p.dt * p.kg_per_kwh - d[t]   # tank dengesi
    if p.terminal_ge_initial:
        m += s[T] >= p.tank_init_kg

    status = m.solve(get_solver(time_limit_s))
    st = pulp.LpStatus[status]
    if st != "Optimal":
        raise RuntimeError(f"MILP çözümü optimal değil: {st}")

    v = lambda dct, n: np.array([pulp.value(dct[i]) or 0.0 for i in range(n)])
    return Plan(v(g, T), v(e, T), np.round(v(u, T)), np.round(v(y, T)), v(d, T),
                v(s, T + 1), float(pulp.value(m.objective)), st)
