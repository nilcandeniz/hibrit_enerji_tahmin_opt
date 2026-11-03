"""Optuna ile XGBoost, LSTM ve GRU hiperparametre araması (yalnızca val seti kullanılır).

Kullanım:  python scripts/tune.py --config configs/lstm.yaml
Çıktı:     <output_dir>/tuning/best_params.json  (+ trials_*.csv)
Sonra:     python scripts/train_compare.py --config configs/lstm.yaml \
                  --tuned results/lstm_run/tuning/best_params.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from solarh2.config import load_config, set_seed
from solarh2.data.windowing import build_windows
from solarh2.tuning.search import save_results, tune_rnn, tune_xgboost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lstm.yaml")
    ap.add_argument("--trials-xgb", type=int, default=None)
    ap.add_argument("--trials-rnn", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=["xgboost", "lstm", "gru"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg.get("tuning", {})
    n_xgb = args.trials_xgb or tc.get("trials_xgboost", 30)
    n_rnn = args.trials_rnn or tc.get("trials_rnn", 20)
    seed = cfg["seed"]
    set_seed(seed)

    splits, _ = build_windows(cfg, seed=seed)
    tr, va = splits["train"], splits["val"]
    print(f"train {len(tr)} | val {len(va)} pencere  (test'e dokunulmuyor)")

    studies = {}
    if "xgboost" in args.only:
        studies["xgboost"] = tune_xgboost(tr, va, n_xgb, seed)
    for cell in ("lstm", "gru"):
        if cell in args.only:
            studies[cell] = tune_rnn(cell, tr, va, cfg, n_rnn, seed,
                                     tc.get("max_epochs", 30), tc.get("patience", 5))
    out = Path(cfg["output_dir"]) / "tuning"
    best = save_results(studies, out)
    for k, v in best.items():
        print(f"\n[{k}] en iyi val = {v['best_value']:.5f} "
              f"({v['n_trials']} deneme, {v['n_pruned']} budandı)\n  {v['params']}")
    print(f"\nKaydedildi: {out.resolve() / 'best_params.json'}")


if __name__ == "__main__":
    main()
