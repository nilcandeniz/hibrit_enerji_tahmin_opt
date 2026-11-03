"""Tahmin -> MILP -> gerçekleşme zinciri; 'tahmin değeri' (value of forecast) analizi.

Kullanım:  python scripts/run_optimization.py --config configs/optimization.yaml
Önkoşul:   scripts/train_compare.py çıktısı olan test_predictions.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarh2.config import load_config
from solarh2.optimization.evaluate import grid_only_plan, realize
from solarh2.optimization.milp import solve_day_ahead
from solarh2.optimization.params import from_config
from solarh2.optimization.prices import align_prices, load_epias_csv, synthetic_ptf



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optimization.yaml")
    ap.add_argument("--npz", default=None, help="config'teki predictions_npz'yi geçersiz kıl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    npz = np.load(args.npz or cfg["predictions_npz"], allow_pickle=True)
    out = Path(args.out or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    times, y_true = npz["times"], npz["y_true"]
    forecasters = [k for k in ["LSTM", "GRU", "XGBoost", "Persistence"] if k in npz.files]
    st = cfg["study"]
    sel = np.arange(0, len(times), st["window_stride"])
    if st["origin_hours"] is not None:
        hrs = pd.DatetimeIndex(times[:, 0]).hour.to_numpy()
        sel = np.array([i for i in sel if hrs[i] in st["origin_hours"]])
    assert len(sel), "Seçilen pencere yok (origin_hours/stride'ı kontrol edin)"

    main_fc = "LSTM" if "LSTM" in forecasters else forecasters[0]   # duyarlılık/grafik için
    capacity = float(y_true.max()) if cfg["plant"]["capacity_kw"] == "auto" else cfg["plant"]["capacity_kw"]
    base = from_config(cfg, capacity)
    ptf = load_epias_csv(cfg["prices"]["epias_csv"]) if cfg["prices"]["epias_csv"] else synthetic_ptf()
    day0 = pd.Timestamp(times.min())
    print(f"{len(sel)} pencere | kapasite {capacity:.0f} kW | elektrolizör {base.ely_max_kw:.0f} kW "
          f"| {base.kwh_per_kg:.1f} kWh/kg | tank {base.tank_max_kg:.0f} kg | "
          f"fiyat kaynağı: {'EPİAŞ' if cfg['prices']['epias_csv'] else 'SENTETİK'}")

    tl = st["solver_time_limit_s"]
    prices = {i: align_prices(times[i], ptf, cfg["prices"]["align_mode"], day0) for i in sel}

    def run(forecast_key: str | None, p, grid_only=False):
        rows = []
        for i in sel:
            fc = y_true[i] if forecast_key is None else npz[forecast_key][i]
            plan = grid_only_plan(fc, p) if grid_only else solve_day_ahead(fc, prices[i], p, tl)
            r = realize(plan, y_true[i], prices[i], p)
            r["window"] = i
            rows.append(r)
        return pd.DataFrame(rows)

    # ---- Ana karşılaştırma ----------------------------------------------------
    res = {"Oracle (mükemmel tahmin)": run(None, base)}
    for k in forecasters:
        res[f"MILP + {k}"] = run(k, base)
    res[f"Sadece şebeke ({main_fc} tahmini)"] = run(main_fc, base, grid_only=True)
    res["Sadece şebeke (oracle)"] = run(None, base, grid_only=True)

    oracle = res["Oracle (mükemmel tahmin)"]["net_tl"].mean()
    summ = pd.DataFrame({
        name: {"Net gelir (TL/gün)": d["net_tl"].mean(),
               "Oracle'a göre kayıp (%)": 100 * (1 - d["net_tl"].mean() / oracle),
               "H2 (kg/gün)": d["h2_kg"].mean(),
               "Dengesizlik net cezası (TL/gün)": d["imbalance_penalty_tl"].mean(),
               "Eksik enerji (kWh/gün)": d["shortfall_kwh"].mean(),
               "Başlatma (adet/gün)": d["starts"].mean()}
        for name, d in res.items()}).T.round(2)
    print("\nKARAR MOTORU SONUÇLARI (gün başına ortalama)\n", summ.to_string())
    summ.to_csv(out / "optimization_summary.csv")

    # ---- Hidrojen fiyatı duyarlılığı ------------------------------------------
    sweep = []
    for h in st["h2_price_sweep"]:
        p = from_config(cfg, capacity, h2_price=h)
        for name, key in [("Oracle", None), (main_fc, main_fc)]:
            d = run(key, p)
            sweep.append(dict(h2_price=h, forecaster=name, net_tl=d["net_tl"].mean(),
                              h2_kg=d["h2_kg"].mean()))
    sw = pd.DataFrame(sweep)
    sw.to_csv(out / "h2_price_sweep.csv", index=False)
    print("\nH2 FİYAT DUYARLILIĞI\n", sw.round(1).to_string(index=False))
    print(f"\nBaşabaş referansı: elektrik maliyeti ≈ {base.kwh_per_kg:.1f} kWh/kg x PTF; "
          f"PTF=1000 TL/MWh -> {base.kwh_per_kg:.1f} TL/kg, 3000 TL/MWh -> {3*base.kwh_per_kg:.0f} TL/kg")

    # ---- Grafikler -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    names = list(summ.index)
    ax[0].barh(names, summ["Net gelir (TL/gün)"], color="#4C78A8")
    ax[0].set_title("Ortalama net gelir (TL/gün)")
    ax[0].invert_yaxis()
    for f in ["Oracle", main_fc]:
        d = sw[sw.forecaster == f]
        ax[1].plot(d.h2_price, d.net_tl, "o-", label=f)
    ax[1].set_xlabel("H2 satış fiyatı (TL/kg)"); ax[1].set_ylabel("Net gelir (TL/gün)")
    ax[1].set_title("Hidrojen fiyatı duyarlılığı"); ax[1].legend()

    i = sel[0]; fc = npz[main_fc][i]
    plan = solve_day_ahead(fc, prices[i], base, tl)
    h = np.arange(len(fc)) * base.dt
    ax[2].stackplot(h, plan.p_grid / 1000, plan.p_ely / 1000, labels=["Şebeke", "Elektrolizör"],
                    colors=["#4C78A8", "#54A24B"], alpha=0.8)
    ax[2].plot(h, y_true[i] / 1000, "k--", lw=1.2, label="Gerçek PV")
    ax[2].plot(h, fc / 1000, "r:", lw=1.2, label=f"{main_fc} tahmini")
    ax[2].set_ylabel("MW"); ax[2].set_xlabel("saat"); ax[2].set_title("Örnek gün: dağıtım planı")
    tw = ax[2].twinx(); tw.plot(h, prices[i], color="orange", lw=1.5, label="PTF"); tw.set_ylabel("TL/MWh")
    ax[2].legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "optimization.png", dpi=150)
    print(f"\nÇıktılar: {out.resolve()}")


if __name__ == "__main__":
    main()
