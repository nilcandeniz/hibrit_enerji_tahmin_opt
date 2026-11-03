"""LSTM ile adil kıyaslama için baseline'lar.

Her model aynı WindowSet'i, yani aynı bilgi kümesini kullanır. Böylece kıyaslama
"aynı soru, farklı model" olur.
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor

from solarh2.data.windowing import WindowSet


def persistence(ws: WindowSet) -> np.ndarray:
    """Gün öncesi persistence: ŷ(t) = y(t − 24 saat). Skill score referansı."""
    return ws.y_lag_day.copy()


def _tabular(ws: WindowSet) -> np.ndarray:
    """Her (pencere, ufuk adımı h) çifti için tek satır.

    Özellikler: h anındaki bilinen gelecek değişkenleri + h/H + y(t+h−24s) + son gözlem.
    """
    N, H, F = ws.x_future.shape
    h = np.tile(np.arange(H) / H, N)[:, None]
    fut = ws.x_future.reshape(N * H, F)
    lag = ws.y_lag_day.reshape(-1, 1)
    last = np.repeat(ws.x_past[:, -1, 0], H)[:, None]  # ölçeklenmiş son hedef değer
    return np.hstack([fut, h, lag, last]).astype(np.float32)


class XGBHorizonModel:
    """Ufuk adımını özellik olarak alan tek bir XGBoost (direct multi-horizon)."""

    def __init__(self, **params):
        defaults = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
        defaults.update(params)
        self.model = XGBRegressor(**defaults, early_stopping_rounds=30)

    def fit(self, train: WindowSet, val: WindowSet):
        self.model.fit(_tabular(train), train.y_raw.ravel(),
                       eval_set=[(_tabular(val), val.y_raw.ravel())], verbose=False)
        return self

    def predict(self, ws: WindowSet) -> np.ndarray:
        pred = self.model.predict(_tabular(ws)).reshape(ws.y_raw.shape)
        return np.clip(pred, 0, None)
