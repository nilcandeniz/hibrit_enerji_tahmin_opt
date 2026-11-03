"""Optuna ile hiperparametre araması (yalnızca VALIDATION üzerinde; test'e asla bakılmaz).

- XGBoost: doğrudan kW cinsinden val RMSE'yi minimize eder.
- LSTM / GRU: her hücre için AYRI bir çalışma (study) ve eşit deneme bütçesi -> adil kıyas.
  Median pruner, kötü gidişatlı denemeleri erken keser (epoch başına val kaybı raporlanır).
- Tüm örnekleyiciler seed'lidir (TPESampler) -> aynı seed aynı arama.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import optuna

from solarh2.data.windowing import WindowSet
from solarh2.models.baselines import XGBHorizonModel
from solarh2.models.seq2seq import Seq2SeqRNN
from solarh2.training.trainer import train_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sampler(seed: int):
    return optuna.samplers.TPESampler(seed=seed)


def tune_xgboost(train: WindowSet, val: WindowSet, n_trials: int, seed: int = 42):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        model = XGBHorizonModel(random_state=seed, **params).fit(train, val)
        pred = model.predict(val)
        return float(np.sqrt(np.mean((pred - val.y_raw) ** 2)))

    study = optuna.create_study(direction="minimize", sampler=_sampler(seed),
                                study_name="xgboost")
    study.optimize(objective, n_trials=n_trials)
    return study


def tune_rnn(cell: str, train: WindowSet, val: WindowSet, cfg: dict, n_trials: int,
             seed: int = 42, max_epochs: int = 30, patience: int = 5):
    n_past, n_future = train.x_past.shape[-1], train.x_future.shape[-1]

    def objective(trial: optuna.Trial) -> float:
        import torch
        torch.manual_seed(seed + trial.number)
        hp = dict(
            hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
        )
        tr = dict(
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        )
        c = copy.deepcopy(cfg)
        c["training"].update(tr, max_epochs=max_epochs, patience=patience)
        model = Seq2SeqRNN(n_past, n_future, cell=cell, **hp)

        def report(epoch: int, val_loss: float):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, hist = train_model(model, train, val, c, on_epoch=report, verbose=False)
        return float(min(h["val_loss"] for h in hist))

    study = optuna.create_study(
        direction="minimize", sampler=_sampler(seed), study_name=cell,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials)
    return study


def save_results(studies: dict[str, optuna.Study], out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = {}
    for name, st in studies.items():
        best[name] = {"params": st.best_params, "best_value": st.best_value,
                      "n_trials": len(st.trials),
                      "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in st.trials)}
        st.trials_dataframe().to_csv(out / f"trials_{name}.csv", index=False)
    (out / "best_params.json").write_text(json.dumps(best, indent=2))
    return best
