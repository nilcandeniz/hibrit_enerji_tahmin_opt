"""Tahmin hatası taraması: hava tahmini hata düzeyi -> tahmin doğruluğu -> TL kaybı.

Her hata düzeyinde (noise_scale) modeller yeniden eğitilir, test tahminleri MILP ile
planlanır ve gerçekleşen gelir Oracle ile kıyaslanır. Sonuç: "tahmin hatası (nRMSE) 1 puan
arttığında günlük gelir kaybı ~X TL" eğrisi — tezin iki katmanını bağlayan ana bulgu.

Kullanım:
  python scripts/forecast_value_sweep.py --config configs/pvgis_forecast.yaml \
      --opt-config configs/optimization_pvgis.yaml \
      --tuned results/pvgis_run/tuning/best_params.json --levels 0 0.5 1 1.5 2

Süre: her düzey bir train_compare koşusu kadar sürer (modeller x düzey). --models ile
azaltılabilir (örn. --models xgboost gru).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarh2.config import load_config, set_seed
from solarh2.data.windowing import build_windows
from solarh2.evaluation.metrics import compute_metrics
from solarh2.optimization.study import build_context, load_prices, run_policy
from solarh2.pipeline import fit_and_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pvgis_forecast.yaml")
    ap.add_argument("--opt-config", default="configs/optimization_pvgis.yaml")
    ap.add_argument("--tuned", default=None)
    ap.add_argument("--levels", nargs="*", type=float, default=[0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--models", nargs="*", default=["xgboost", "lstm", "gru"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, opt = load_config(args.config), load_config(args.opt_config)
    if cfg["data"]["weather_forecast"]["mode"] != "lead_time":
        raise SystemExit("Tarama için config'te weather_forecast.mode: lead_time olmalı.")
    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned else {}
    out = Path(args.out or Path(cfg["output_dir"]) / "forecast_value_sweep")
    out.mkdir(parents=True, exist_ok=True)
    prices = load_prices(opt)
    modes = tuple(opt["study"].get("recourse_modes", [False]))
    tl = opt["study"]["solver_time_limit_s"]
    base_std = cfg["data"]["weather_forecast"]["noise_std"]

    rows, ctx, oracle = [], None, None
    for lvl in args.levels:
        c = copy.deepcopy(cfg)
        c["data"]["weather_forecast"]["noise_scale"] = float(lvl)
        set_seed(cfg["seed"])
        splits, scalers = build_windows(c, seed=cfg["seed"])
        te = splits["test"]
        print(f"\n=== Hata düzeyi x{lvl}  (ışınım std h=1: {lvl*base_std[0]:.0%}, h=24: {lvl*base_std[1]:.0%}) ===")
        preds, _, _ = fit_and_predict(c, splits, scalers, tuned, args.models,
                                      max_epochs=args.max_epochs, verbose=False)
        if ctx is None:
            ctx = build_context(te.times, te.y_raw, opt, prices)
            oracle = run_policy(ctx, te.y_raw, None, time_limit_s=tl)["net_tl"].mean()
            print(f"{len(ctx.sel)} gün | fiyat: {ctx.price_source} | Oracle {oracle:,.0f} TL/gün")
        cap = float(np.nanmax(splits["train"].y_raw))
        sel = ctx.sel
        for name, p in preds.items():
            if name == "Persistence" and lvl != args.levels[0]:
                continue                       # persistence hava tahmininden bağımsız
            m = compute_metrics(te.y_raw[sel], p[sel], cap, te.daytime[sel])
            # Dengesizlik kuralı asimetrik (açık 1.03*max, fazla 0.97*min): fazla tahmin (bias>0)
            # eksik tahminden pahalıdır -> kaybı RMSE kadar yön hatası da belirler.
            dm = te.daytime[sel]
            bias = float(np.mean(p[sel][dm] - te.y_raw[sel][dm]))
            for rc in modes:
                net = run_policy(ctx, te.y_raw, p, recourse=rc, time_limit_s=tl)["net_tl"].mean()
                rows.append(dict(level=lvl, irr_std_h24=lvl * base_std[1], model=name, recourse=rc,
                                 nRMSE=m["nRMSE_%"], RMSE_kW=m["RMSE_kW"], bias_day_kW=bias, net_tl=net,
                                 loss_tl=oracle - net, loss_pct=100 * (oracle - net) / oracle))
            r = [x for x in rows if x["model"] == name and x["level"] == lvl]
            print(f"  {name:12s} nRMSE {m['nRMSE_%']:5.2f}%  bias {bias:+7.1f} kW | kayıp " +
                  "  ".join(f"{'RT' if x['recourse'] else 'plan'}: {x['loss_tl']:8,.0f} TL/gün" for x in r))
        pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)   # her düzeyden sonra kaydet

    df = pd.DataFrame(rows)
    learn = df[(df.model != "Persistence") & (~df.recourse)]
    slope, icpt = np.polyfit(learn.nRMSE, learn.loss_tl, 1)
    summary = {"TL_per_nRMSE_point": float(slope), "intercept_tl": float(icpt), "oracle_tl": float(oracle)}
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nEĞİM: nRMSE 1 puan artınca günlük kayıp ≈ {slope:,.0f} TL (plana sadık, öğrenen modeller)")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, g in df[~df.recourse].groupby("model"):
        ax[0].plot(g.nRMSE, g.loss_tl, "o-" if name != "Persistence" else "x", ms=8, label=name)
    xs = np.linspace(0, df.nRMSE.max() * 1.05, 50)
    ax[0].plot(xs, slope * xs + icpt, "k--", lw=1, label=f"doğrusal uyum: {slope:,.0f} TL / puan")
    ax[0].set_xlabel("Tahmin hatası nRMSE (%)"); ax[0].set_ylabel("Oracle'a göre gelir kaybı (TL/gün)")
    ax[0].set_title("Tahmin doğruluğunun ekonomik değeri"); ax[0].legend(fontsize=8)
    if True in modes:
        for name, g in df[df.model != "Persistence"].groupby("model"):
            gp, gr = g[~g.recourse].sort_values("level"), g[g.recourse].sort_values("level")
            ax[1].plot(gp.level, gp.loss_tl - gr.loss_tl.to_numpy(), "o-", label=name)
        ax[1].set_xlabel("Hata düzeyi (noise_scale)"); ax[1].set_ylabel("RT düzeltmenin kazandırdığı (TL/gün)")
        ax[1].set_title("Elektrolizörün dengesizlik tamponu değeri"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "forecast_value_sweep.png", dpi=150)
    print(f"Çıktılar: {out.resolve()}")


if __name__ == "__main__":
    main()
