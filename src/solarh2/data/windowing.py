"""Zaman serisini 3B kayan pencerelere (samples, timesteps, features) dönüştürür.

Tasarım ilkeleri
----------------
1. Düzenli zaman ızgarası: Ham veride eksik 15 dk'lık damgalar var. Seri tam ızgaraya
   yeniden indekslenir; boşluk içeren pencereler atılır (sessizce kaydırma yapılmaz).
2. Sızıntısız bölme: Bölme, tahmin *hedef* aralıklarına göre yapılır. Bir train
   penceresinin hedefi hiçbir zaman val/test bölgesine taşmaz. Ölçekleyiciler yalnızca
   train satırlarına fit edilir.
3. Geçmiş / gelecek ayrımı: Encoder yalnızca t anına kadar ölçülenleri görür;
   decoder yalnızca tahmin anında gerçekten bilinebilecek değişkenleri görür.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowSet:
    x_past: np.ndarray      # (N, lookback, n_past)
    x_future: np.ndarray    # (N, horizon, n_future)
    y: np.ndarray           # (N, horizon)   ölçeklenmiş hedef
    y_raw: np.ndarray       # (N, horizon)   kW cinsinden hedef
    y_lag_day: np.ndarray   # (N, horizon)   24 saat önceki gerçek değer (kW) -> persistence
    daytime: np.ndarray     # (N, horizon)   bool, ışınım > 0
    times: np.ndarray       # (N, horizon)   hedef zaman damgaları

    def __len__(self) -> int:
        return len(self.y)


def load_frame(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = pd.read_csv(d["csv_path"], parse_dates=[d["time_col"]])
    df = df.set_index(d["time_col"]).sort_index()
    df = df[~df.index.duplicated()]
    full_idx = pd.date_range(df.index.min().floor("D"), df.index.max().ceil("D"),
                             freq=d["freq"], inclusive="left")
    df = df.reindex(full_idx)
    # Kısa boşluklar (varsayılan ≤ 1 saat) zaman-doğrusal interpolasyonla doldurulur;
    # daha uzun boşluklar NaN kalır ve o boşluğa değen pencereler atılır.
    gap = d.get("max_interp_gap", 4)
    num = df.select_dtypes("number").columns
    df[num] = df[num].interpolate(method="time", limit=gap, limit_area="inside")

    minutes = df.index.hour * 60 + df.index.minute
    df["sin_time"] = np.sin(2 * np.pi * minutes / 1440)
    df["cos_time"] = np.cos(2 * np.pi * minutes / 1440)
    doy = df.index.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def _future_columns(d: dict) -> list[str]:
    mode = d["weather_forecast"]["mode"]
    cols = list(d["future_calendar"])
    if mode in ("oracle", "noisy"):
        cols += [f"{c}__fcst" for c in d["future_weather"]]
    return cols


def build_windows(cfg: dict, df: pd.DataFrame | None = None, seed: int = 0):
    """Train/val/test WindowSet'lerini ve ölçekleyicileri döndürür."""
    d = cfg["data"]
    df = load_frame(cfg) if df is None else df.copy()
    target = d["target"]
    L, H = d["lookback"], d["horizon"]
    DAY = int(pd.Timedelta("1D") / pd.Timedelta(d["freq"]))  # 15 dk'da 96 adım
    assert H <= DAY, "Persistence referansı geleceği görmesin diye horizon <= 1 gün olmalı"

    # --- Simüle edilmiş hava tahmini (gelecek değişkenleri) -------------------
    # IRRADIATION ve diğerleri: çarpımsal hata (noise_std). AMBIENT_TEMPERATURE:
    # `temp_noise_std_c` verilmişse toplamsal (°C), verilmemişse eski davranış (çarpımsal).
    # `rho` > 0 ise hata AR(1) otokorelasyonludur (gerçek NWP hataları saatten saate
    # bağımsız değildir: bulut zamanlaması hatası birkaç saat sürer). rho=0 -> iid.
    rng = np.random.default_rng(seed)
    wf = d["weather_forecast"]
    rho = float(wf.get("rho", 0.0))

    def unit_noise(n: int) -> np.ndarray:
        z = rng.normal(0.0, 1.0, n)
        if rho > 0:
            from scipy.signal import lfilter
            z = lfilter([np.sqrt(1 - rho ** 2)], [1, -rho], z)   # birim varyans korunur
        return z

    for c in d["future_weather"]:
        fc = df[c].copy()
        if wf["mode"] == "noisy":
            z = unit_noise(len(fc))
            if c == "AMBIENT_TEMPERATURE" and "temp_noise_std_c" in wf:
                fc = fc + wf["temp_noise_std_c"] * z
            else:
                fc = fc * (1 + wf["noise_std"] * z)
            if c == "IRRADIATION":
                fc = fc.clip(lower=0)
        df[f"{c}__fcst"] = fc

    past_cols = [target] + [c for c in d["past_features"] if c != target]
    fut_cols = _future_columns(d)

    # --- Kronolojik bölme sınırları (satır indeksi) ---------------------------
    n = len(df)
    r_tr, r_va, _ = d["split"]
    b1, b2 = int(n * r_tr), int(n * (r_tr + r_va))

    # Ölçekleyiciler SADECE train bölgesine fit edilir
    sc_past = StandardScaler().fit(df[past_cols].iloc[:b1].dropna())
    sc_fut = StandardScaler().fit(df[fut_cols].iloc[:b1].dropna())
    sc_y = StandardScaler().fit(df[[target]].iloc[:b1].dropna())

    past = sc_past.transform(df[past_cols].fillna(0)).astype(np.float32)
    fut = sc_fut.transform(df[fut_cols].fillna(0)).astype(np.float32)
    y_raw = df[target].to_numpy(dtype=np.float32)
    y_sc = sc_y.transform(df[[target]].fillna(0)).ravel().astype(np.float32)
    valid = df[past_cols + fut_cols].notna().all(axis=1).to_numpy()
    irr = df["IRRADIATION"].to_numpy() if "IRRADIATION" in df else y_raw
    times = df.index.to_numpy()

    def make(origin_lo: int, origin_hi: int, stride: int) -> WindowSet:
        # origin = tahmin edilen ilk adımın indeksi; hedef aralığı [o, o+H)
        idx = []
        lo = max(origin_lo, L, DAY)  # persistence için 24 saat geriye erişim gerekir
        for o in range(lo, origin_hi - H + 1, stride):
            if valid[o - max(L, DAY): o + H].all():
                idx.append(o)
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            raise ValueError("Bu bölmede hiç geçerli pencere yok — lookback/horizon/split'i kontrol edin.")
        p = np.stack([past[o - L:o] for o in idx])
        f = np.stack([fut[o:o + H] for o in idx])
        tgt = np.stack([y_sc[o:o + H] for o in idx])
        tgt_raw = np.stack([y_raw[o:o + H] for o in idx])
        lag = np.stack([y_raw[o - DAY + np.arange(H)] for o in idx])
        day = np.stack([irr[o:o + H] > 0 for o in idx])
        t = np.stack([times[o:o + H] for o in idx])
        return WindowSet(p, f, tgt, tgt_raw, lag, day, t)

    splits = {
        "train": make(0, b1, d["train_stride"]),
        "val": make(b1, b2, d["eval_stride"]),
        "test": make(b2, n, d["eval_stride"]),
    }
    scalers = {"past": sc_past, "future": sc_fut, "y": sc_y,
               "past_cols": past_cols, "future_cols": fut_cols}
    return splits, scalers
