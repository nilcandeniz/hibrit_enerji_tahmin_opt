"""PVGIS seriescalc CSV biçiminde (UTC, HH:10, W, W/m², footer'lı) sentetik veri üretici.

Gerçek PVGIS/EPİAŞ dosyası olmadan veri hattını (ayrıştırma, UTC->yerel dönüşüm, takvim
eşleştirme) uçtan uca sınamak içindir. Konum: İskenderun. Yalnızca test amaçlıdır.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

LAT, LON = 36.587, 36.173


def make_pvgis_csv(path: str, start_year: int = 2021, end_year: int = 2023,
                   peak_kw: float = 10000.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{start_year}-01-01 00:10", f"{end_year}-12-31 23:10", freq="h")  # UTC
    doy = idx.dayofyear.to_numpy()
    utc_h = (idx.hour + idx.minute / 60).to_numpy()
    solar_h = (utc_h + LON / 15) % 24
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.deg2rad(15 * (solar_h - 12))
    phi = np.deg2rad(LAT)
    sin_el = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    elev = np.rad2deg(np.arcsin(np.clip(sin_el, -1, 1)))
    clear = 1000 * np.clip(sin_el, 0, None) ** 1.15
    n_days = (idx.normalize().nunique())
    k_day = np.clip(0.75 + 0.2 * rng.standard_normal(n_days), 0.25, 1.0)
    k = np.repeat(k_day, 24)[: len(idx)]
    k = np.clip(k * (1 - np.abs(rng.normal(0, 0.08, len(idx)))), 0.15, 1.0)
    g = clear * k
    t2m = (19 + 9 * np.sin(2 * np.pi * (doy - 110) / 365)
           + 5 * np.sin(2 * np.pi * (solar_h - 9) / 24) + rng.normal(0, 1.0, len(idx)))
    ws = np.clip(rng.gamma(2.5, 1.2, len(idx)), 0, 20)
    tcell = t2m + 0.03 * g
    p_w = peak_kw * 1000 * 0.86 * (g / 1000) * (1 - 0.004 * (tcell - 25))
    p_w = np.clip(p_w, 0, None)
    rows = pd.DataFrame({"time": idx.strftime("%Y%m%d:%H%M"), "P": p_w.round(2), "G(i)": g.round(2),
                         "H_sun": elev.round(2), "T2m": t2m.round(2), "WS10m": ws.round(2),
                         "Int": 0.0})
    with open(path, "w", newline="") as f:
        f.write(f"Latitude (decimal degrees):\t{LAT}\n")
        f.write(f"Longitude (decimal degrees):\t{LON}\n")
        f.write("Elevation (m):\t12.0\n")
        f.write("Radiation database:\tPVGIS-SARAH3\n")
        f.write(f"Nominal power of the PV system (c-Si) (kWp):\t{peak_kw}\n")
        f.write("System losses (%):\t14.0\n\n")
        f.write("time,P,G(i),H_sun,T2m,WS10m,Int\n")
        rows.to_csv(f, header=False, index=False, lineterminator="\n")
        f.write("\nP: PV system power (W)\nG(i): Global in-plane irradiance (W/m2)\n"
                "H_sun: Sun height (degree)\nT2m: 2-m air temperature (degree Celsius)\n"
                "WS10m: 10-m total wind speed (m/s)\nInt: 1 means solar radiation values are reconstructed\n")


def make_epias_like(path: str, start: str, days: int, seed: int = 0, xlsx: bool = False) -> None:
    """EPİAŞ'a benzer export: Tarih;Saat;PTF (TL/MWh) ve Türkçe sayı biçimi ('1.234,56')."""
    from solarh2.optimization.prices import synthetic_ptf
    s = synthetic_ptf(days, seed, start=start)
    if xlsx:
        pd.DataFrame({"Tarih": s.index.strftime("%d.%m.%Y"), "Saat": s.index.strftime("%H:%M"),
                      "PTF (TL/MWh)": s.to_numpy().round(2)}).to_excel(path, index=False)
        return
    tr = lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("Tarih;Saat;PTF (TL/MWh)\n")
        for t, v in s.items():
            f.write(f"{t:%d.%m.%Y};{t:%H:%M};\"{tr(v)}\"\n")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/pvgis_synthetic.csv"
    make_pvgis_csv(out)
    print("yazıldı:", out)
