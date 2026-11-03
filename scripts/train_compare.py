"""Persistence, XGBoost, LSTM ve GRU'yu aynı test pencerelerinde eğitip kıyaslar.

Kullanım:
  python scripts/train_compare.py --config configs/lstm.yaml
  python scripts/train_compare.py --config configs/lstm.yaml --tuned results/lstm_run/tuning/best_params.json
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
import torch

from solarh2.config import load_config, set_seed
from solarh2.data.windowing import build_windows
from solarh2.evaluation.metrics import compute_metrics, rmse_by_horizon
from solarh2.models.baselines import XGBHorizonModel, persistence
from solarh2.models.seq2seq import Seq2SeqRNN
from solarh2.training.trainer import predict, train_model

RNN_HP = ("hidden_size", "num_layers", "dropout")
TRAIN_HP = ("lr", "weight_decay", "batch_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lstm.yaml")
    ap.add_argument("--tuned", default=None, help="tune.py çıktısı (best_params.json)")
    ap.add_argument("--models", nargs="*", default=["xgboost", "lstm", "gru"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned else {}

    splits, scalers = build_windows(cfg, seed=cfg["seed"])
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print({k: len(v) for k, v in splits.items()}, "pencere |",
          "x_past", tr.x_past.shape, "x_future", tr.x_future.shape,
          "| hiperparametreler:", "OPTUNA" if tuned else "config varsayılanı")
    capacity = float(np.nanmax(tr.y_raw))

    preds = {"Persistence": persistence(te)}
    histories = {}

    if "xgboost" in args.models:
        xp = {k: v for k, v in tuned.get("xgboost", {}).get("params", {}).items()}
        xgb = XGBHorizonModel(random_state=cfg["seed"], **xp).fit(tr, va)
        preds["XGBoost"] = xgb.predict(te)

    for cell in ("lstm", "gru"):
        if cell not in args.models:
            continue
        set_seed(cfg["seed"])
        hp = dict(cfg["model"])
        c = copy.deepcopy(cfg)
        for k, v in tuned.get(cell, {}).get("params", {}).items():
            (hp if k in RNN_HP else c["training"])[k] = v
        print(f"\n--- {cell.upper()} ---  {hp}  lr={c['training']['lr']:.2e} bs={c['training']['batch_size']}")
        model = Seq2SeqRNN(tr.x_past.shape[-1], tr.x_future.shape[-1], cell=cell, **hp)
        model, hist = train_model(model, tr, va, c)
        preds[cell.upper()] = predict(model, te, scalers["y"], cfg["training"]["device"])
        histories[cell] = hist
        torch.save(model.state_dict(), out / f"{cell}_best.pt")

    rows = {name: compute_metrics(te.y_raw, p, capacity, te.daytime, preds["Persistence"])
            for name, p in preds.items()}
    table = pd.DataFrame(rows).T.round(4)
    print("\nTEST SONUÇLARI\n", table.to_string())
    table.to_csv(out / "metrics.csv")
    (out / "history.json").write_text(json.dumps(histories, indent=2))
    np.savez_compressed(out / "test_predictions.npz", times=te.times.astype("datetime64[ns]"),
                        y_true=te.y_raw, **preds)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    i = 0
    ax[0].plot(te.times[i], te.y_raw[i], "k-", lw=2, label="Gerçek")
    for name, p in preds.items():
        ax[0].plot(te.times[i], p[i], label=name, alpha=0.85)
    ax[0].set_title("24 saatlik tahmin örneği (ilk test penceresi)")
    ax[0].set_ylabel("AC güç (kW)")
    ax[0].legend()
    ax[0].tick_params(axis="x", rotation=30)
    step_h = pd.Timedelta(cfg["data"]["freq"]) / pd.Timedelta("1h")
    hours = np.arange(1, te.y_raw.shape[1] + 1) * step_h
    for name, p in preds.items():
        ax[1].plot(hours, rmse_by_horizon(te.y_raw, p), label=name)
    ax[1].set_title("Ufuk adımına göre RMSE")
    ax[1].set_xlabel("Tahmin ufku (saat)")
    ax[1].set_ylabel("RMSE (kW)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    print(f"\nÇıktılar: {out.resolve()}")


if __name__ == "__main__":
    main()
