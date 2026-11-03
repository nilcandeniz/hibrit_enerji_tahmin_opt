"""Güneş tahmini literatürüne uygun metrikler.

- Gece değerleri (≈0) hatayı yapay olarak düşürür; bu yüzden gündüz-maskeli metrikler
  de raporlanır.
- nRMSE kurulu güce (kapasite) göre normalize edilir — farklı santrallerle kıyaslanabilir.
- Skill score, persistence'a göre iyileşmeyi verir: SS = 1 − RMSE_model / RMSE_persist.
"""
from __future__ import annotations

import numpy as np


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity: float,
                    daytime: np.ndarray | None = None,
                    y_ref: np.ndarray | None = None) -> dict:
    yt, yp = y_true.ravel(), y_pred.ravel()
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    out = {
        "RMSE_kW": _rmse(yt, yp),
        "MAE_kW": float(np.mean(np.abs(yt - yp))),
        "nRMSE_%": 100 * _rmse(yt, yp) / capacity,
        "R2": float(1 - ss_res / ss_tot),
    }
    if daytime is not None:
        m = daytime.ravel()
        out["RMSE_day_kW"] = _rmse(yt[m], yp[m])
        out["MAE_day_kW"] = float(np.mean(np.abs(yt[m] - yp[m])))
    if y_ref is not None:
        out["SkillScore"] = 1 - out["RMSE_kW"] / _rmse(yt, y_ref.ravel())
    return out


def rmse_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Her ufuk adımı için RMSE — hatanın ufukla nasıl büyüdüğünü gösterir."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
