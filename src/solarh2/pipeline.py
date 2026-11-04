"""Model eğitimi + test tahmini (train_compare, multi_seed ve sweep aynı fonksiyonu kullanır)."""
from __future__ import annotations

import copy

from solarh2.config import set_seed
from solarh2.models.baselines import XGBHorizonModel, persistence
from solarh2.models.seq2seq import Seq2SeqRNN
from solarh2.training.trainer import predict, train_model

RNN_HP = ("hidden_size", "num_layers", "dropout")


def fit_and_predict(cfg: dict, splits: dict, scalers: dict, tuned: dict | None = None,
                    models=("xgboost", "lstm", "gru"), seed: int | None = None,
                    max_epochs: int | None = None, verbose: bool = True):
    """Persistence + seçili modelleri eğitir, TEST tahminlerini (kW) döndürür.

    seed: model başlatma/örnekleme seed'i (pencerelerdeki hava tahmini hatası config seed'iyle
    sabit kalır -> çoklu seed denemesinde yalnızca model rastgeleliği değişir).
    Dönüş: (preds: {ad: (N,H)}, histories: {hücre: [...]}, fitted: {ad: model})
    """
    tuned = tuned or {}
    seed = cfg["seed"] if seed is None else seed
    tr, va, te = splits["train"], splits["val"], splits["test"]
    preds, histories, fitted = {"Persistence": persistence(te)}, {}, {}

    if "xgboost" in models:
        xp = dict(tuned.get("xgboost", {}).get("params", {}))
        xgb = XGBHorizonModel(random_state=seed, **xp).fit(tr, va)
        preds["XGBoost"] = xgb.predict(te)
        fitted["XGBoost"] = xgb

    for cell in ("lstm", "gru"):
        if cell not in models:
            continue
        set_seed(seed)
        hp = dict(cfg["model"])
        c = copy.deepcopy(cfg)
        for k, v in tuned.get(cell, {}).get("params", {}).items():
            (hp if k in RNN_HP else c["training"])[k] = v
        if max_epochs is not None:
            c["training"]["max_epochs"] = max_epochs
        if verbose:
            print(f"\n--- {cell.upper()} (seed {seed}) ---  {hp}  "
                  f"lr={c['training']['lr']:.2e} bs={c['training']['batch_size']}")
        model = Seq2SeqRNN(tr.x_past.shape[-1], tr.x_future.shape[-1], cell=cell, **hp)
        model, hist = train_model(model, tr, va, c, verbose=verbose)
        preds[cell.upper()] = predict(model, te, scalers["y"], cfg["training"]["device"])
        histories[cell] = hist
        fitted[cell.upper()] = model
    return preds, histories, fitted
