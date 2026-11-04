"""Tahmin -> MILP -> gerçekleşme zinciri; 'tahmin değeri' (value of forecast) analizi.

Kullanım:  python scripts/run_optimization.py --config configs/optimization_pvgis.yaml
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
from solarh2.optimization.milp import solve_day_ahead
from solarh2.optimization.params import from_config
from solarh2.optimization.study import ORACLE, build_context, evaluate_forecasts, run_policy


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
    main_fc = "LSTM" if "LSTM" in forecasters else forecasters[0]
    st = cfg["study"]
    tl = st["solver_time_limit_s"]
    modes = tuple(st.get("recourse_modes", [False]))

    ctx = build_context(times, y_true, cfg)
    p = ctx.params
    print(f"{len(ctx.sel)} pencere | kapasite {ctx.capacity:.0f} kW | elektrolizör {p.ely_max_kw:.0f} kW "
          f"| {p.kwh_per_kg:.1f} kWh/kg | tank {p.tank_max_kg:.0f} kg | fiyat: {ctx.price_source}")

    # ---- Ana karşılaştırma ----------------------------------------------------
    summ, res = evaluate_forecasts(ctx, y_true, {k: npz[k] for k in forecasters},
                                   recourse_modes=modes, grid_only_for=main_fc, time_limit_s=tl)
    summ = summ.round(2)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\nKARAR MOTORU SONUÇLARI (gün başına ortalama)\n", summ.to_string())
    summ.to_csv(out / "optimization_summary.csv")
    pd.concat({k: v for k, v in res.items()}, names=["strateji"]).to_csv(out / "optimization_daily.csv")

    # ---- Hidrojen fiyatı duyarlılığı ------------------------------------------
    sweep = []
    for h in st["h2_price_sweep"]:
        ph = from_config(cfg, ctx.capacity, h2_price=h)
        for name, key in [("Oracle", None), (main_fc, main_fc)]:
            d = run_policy(ctx, y_true, None if key is None else npz[key], ph, time_limit_s=tl)
            sweep.append(dict(h2_price=h, forecaster=name, net_tl=d["net_tl"].mean(),
                              h2_kg=d["h2_kg"].mean()))
    sw = pd.DataFrame(sweep)
    sw.to_csv(out / "h2_price_sweep.csv", index=False)
    print("\nH2 FİYAT DUYARLILIĞI\n", sw.round(1).to_string(index=False))

    # ---- Grafikler -------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    ax[0].barh(list(summ.index), summ["Net gelir (TL/gün)"], color="#4C78A8")
    ax[0].set_title("Ortalama net gelir (TL/gün)")
    ax[0].invert_yaxis()
    ax[0].tick_params(axis="y", labelsize=8)
    for f in ["Oracle", main_fc]:
        d = sw[sw.forecaster == f]
        ax[1].plot(d.h2_price, d.net_tl, "o-", label=f)
    ax[1].set_xlabel("H2 satış fiyatı (TL/kg)"); ax[1].set_ylabel("Net gelir (TL/gün)")
    ax[1].set_title("Hidrojen fiyatı duyarlılığı"); ax[1].legend()

    i = ctx.sel[0]; fc = npz[main_fc][i]
    plan = solve_day_ahead(fc, ctx.ptf[i], p, tl)
    h = np.arange(len(fc)) * p.dt
    ax[2].stackplot(h, plan.p_grid / 1000, plan.p_ely / 1000, labels=["Şebeke", "Elektrolizör"],
                    colors=["#4C78A8", "#54A24B"], alpha=0.8)
    ax[2].plot(h, y_true[i] / 1000, "k--", lw=1.2, label="Gerçek PV")
    ax[2].plot(h, fc / 1000, "r:", lw=1.2, label=f"{main_fc} tahmini")
    ax[2].set_ylabel("MW"); ax[2].set_xlabel("saat"); ax[2].set_title("Örnek gün: dağıtım planı")
    tw = ax[2].twinx(); tw.plot(h, ctx.ptf[i], color="orange", lw=1.5, label="PTF")
    if ctx.smf is not None:
        tw.plot(h, ctx.smf[i], color="purple", lw=1, ls="--", label="SMF")
    tw.set_ylabel("TL/MWh"); tw.legend(loc="upper right", fontsize=8)
    ax[2].legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "optimization.png", dpi=150)
    print(f"\nÇıktılar: {out.resolve()}")


if __name__ == "__main__":
    main()
