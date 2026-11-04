"""Çoklu seed ile model kıyası + Diebold-Mariano testi.

Hava tahmini hatası (pencereler) sabit kalır; yalnızca model rastgeleliği (ağırlık
başlatma, mini-batch sırası, XGBoost alt-örneklemesi) değişir. Böylece "GRU, LSTM'den iyi"
iddiasının rastlantı olup olmadığı sınanır.

Kullanım:
  python scripts/multi_seed.py --config configs/pvgis_forecast.yaml \
      --tuned results/pvgis_run/tuning/best_params.json --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from solarh2.config import load_config
from solarh2.data.windowing import build_windows
from solarh2.evaluation.metrics import compute_metrics, daily_mse, diebold_mariano
from solarh2.pipeline import fit_and_predict

METRICS = ["RMSE_kW", "MAE_kW", "nRMSE_%", "R2", "RMSE_day_kW", "SkillScore", "nRMSE_all_%"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pvgis_forecast.yaml")
    ap.add_argument("--tuned", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=["xgboost", "lstm", "gru"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned else {}
    out = Path(args.out or Path(cfg["output_dir"]) / "multi_seed")
    out.mkdir(parents=True, exist_ok=True)

    splits, scalers = build_windows(cfg, seed=cfg["seed"])      # pencereler tüm seed'lerde aynı
    tr, te = splits["train"], splits["test"]
    cap = float(np.nanmax(tr.y_raw))
    day0 = np.where(pd.DatetimeIndex(te.times[:, 0]).hour == 0)[0]   # örtüşmeyen günlük pencereler
    print(f"test: {len(te)} pencere, DM için {len(day0)} örtüşmeyen gün | seed'ler: {args.seeds}")

    runs, losses = [], {}
    for s in args.seeds:
        preds, _, _ = fit_and_predict(cfg, splits, scalers, tuned, args.models, seed=s,
                                      max_epochs=args.max_epochs, verbose=False)
        for name, p in preds.items():
            # Ana metrikler, DM testi ve MILP ile AYNI kümede: her gün 00:00'da başlayan
            # örtüşmeyen gün-öncesi pencereleri. Tüm (saatlik başlangıçlı) pencereler ayrıca.
            m = compute_metrics(te.y_raw[day0], p[day0], cap, te.daytime[day0], preds["Persistence"][day0])
            m["nRMSE_all_%"] = compute_metrics(te.y_raw, p, cap)["nRMSE_%"]
            runs.append(dict(seed=s, model=name, **{k: m[k] for k in METRICS}))
            losses[(name, s)] = daily_mse(te.y_raw[day0], p[day0])
        msg = "  ".join(f"{r['model']}: {r['nRMSE_%']:.3f}%" for r in runs if r["seed"] == s)
        print(f"seed {s}: {msg}")
        pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)

    df = pd.DataFrame(runs)
    agg = df.groupby("model")[METRICS].agg(["mean", "std"])
    order = agg[("nRMSE_%", "mean")].sort_values().index
    agg = agg.loc[order]
    agg.to_csv(out / "summary.csv")
    print(f"\nÖZET (ortalama ± std, n = {len(args.seeds)} seed; {len(day0)} gün, 00:00 pencereleri"
          " = DM ve MILP ile aynı küme; nRMSE_all = tüm saatlik başlangıçlar)")
    for model in order:
        print(f"  {model:12s} " + "  ".join(
            f"{k}: {agg.loc[model, (k, 'mean')]:.4f} ± {agg.loc[model, (k, 'std')]:.4f}"
            for k in ["nRMSE_%", "R2", "SkillScore", "nRMSE_all_%"]))

    learned = [m for m in order if m != "Persistence"]
    tests = []
    for a, b in itertools.combinations(learned, 2):
        per_seed = [diebold_mariano(losses[(a, s)], losses[(b, s)]) for s in args.seeds]
        avg = diebold_mariano(np.mean([losses[(a, s)] for s in args.seeds], axis=0),
                              np.mean([losses[(b, s)] for s in args.seeds], axis=0))
        tests.append(dict(A=a, B=b, DM_seed_avg=avg["DM"], p_seed_avg=avg["p_value"],
                          A_better_in_seeds=sum(r["mean_diff"] < 0 for r in per_seed),
                          significant_seeds=sum(r["p_value"] < 0.05 for r in per_seed),
                          n_seeds=len(args.seeds)))
    tdf = pd.DataFrame(tests)
    tdf.to_csv(out / "diebold_mariano.csv", index=False)
    print("\nDIEBOLD-MARIANO (H0: eşit doğruluk; DM<0 -> A daha iyi)")
    for t in tests:
        verdict = "anlamlı fark" if t["p_seed_avg"] < 0.05 else "istatistiksel olarak ayırt edilemez"
        print(f"  {t['A']} vs {t['B']}: DM={t['DM_seed_avg']:+.2f}, p={t['p_seed_avg']:.3f} -> {verdict} "
              f"| A {t['A_better_in_seeds']}/{t['n_seeds']} seed'de daha iyi, "
              f"{t['significant_seeds']} seed'de p<0.05")
    print(f"\nÇıktılar: {out.resolve()}")


if __name__ == "__main__":
    main()
