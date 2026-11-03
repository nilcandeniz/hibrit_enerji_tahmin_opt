"""PVGIS saatlik veri: indirme, ayrıştırma ve proje şemasına dönüştürme.

PVGIS `seriescalc` (saatlik) çıktısı:
  * `time`  : UTC (!), biçim YYYYMMDD:HHMM. SARAH için dakika genelde :10'dur.
  * `P`     : PV sistem gücü [W]   (pvcalculation=1 ise)
  * `G(i)`  : eğik düzlem global ışınım [W/m²]
  * `T2m`   : 2 m hava sıcaklığı [°C],  `WS10m`: 10 m rüzgâr hızı [m/s]

KRİTİK: EPİAŞ fiyatları Türkiye YEREL saatindedir (UTC+3, yaz saati uygulaması yok),
PVGIS ise UTC. Dönüştürmeden eşleştirirsen üretim/fiyat 3 saat kayar ve ördek eğrisinin
anlamı bozulur. Bu modül çıktıyı yerel saate çevirir ve tz bilgisini atar (naive).

Çıktı şeması (Kaggle `df_merged.csv` ile uyumlu adlar):
  DATE_TIME (yerel, naive) | AC_POWER [kW] | IRRADIATION [kW/m²] | AMBIENT_TEMPERATURE [°C]
  | WIND_SPEED [m/s]

Not: PVGIS'in P değeri bir MODEL çıktısıdır (ölçüm değil): ışınım + sıcaklıktan
hesaplanır. Bu yüzden PVGIS üzerindeki tahmin başarısı, gerçek santral telemetrisinin
(Kaggle) yerine geçmez; ikisi birbirini tamamlar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TZ = "Europe/Istanbul"


def to_project_frame(data: pd.DataFrame, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """pvlib'in döndürdüğü (map_variables=True) tabloyu proje şemasına çevirir."""
    cols = {"P": "AC_POWER", "poa_global": "IRRADIATION",
            "temp_air": "AMBIENT_TEMPERATURE", "wind_speed": "WIND_SPEED"}
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"PVGIS çıktısında beklenen sütunlar yok: {missing}. "
                         "pvcalculation=1 ve components=0 ile indirdiniz mi?")
    df = data[list(cols)].rename(columns=cols).copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    # UTC -> yerel -> naive; SARAH :10 dakika ofsetini saate yuvarla
    df.index = idx.tz_convert(tz).tz_localize(None).floor("h")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # UTC->yerel kayması baştaki/sondaki günleri kısmi bırakır (örn. 2023-12-31 23:10 UTC =
    # 2024-01-01 02:10 yerel). Takvim eşleştirmede kenar sorunu çıkmasın diye yalnızca TAM
    # yerel günler tutulur.
    full = df.groupby(df.index.normalize()).size()
    df = df[df.index.normalize().isin(full[full == 24].index)]
    df["AC_POWER"] = df["AC_POWER"] / 1000.0          # W -> kW
    df["IRRADIATION"] = df["IRRADIATION"] / 1000.0    # W/m² -> kW/m²
    df.index.name = "DATE_TIME"
    return df.reset_index()


def load_pvgis_csv(path: str | Path, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """PVGIS web arayüzünden (veya API'den) indirilmiş ham CSV'yi okur."""
    from pvlib.iotools import read_pvgis_hourly
    out = read_pvgis_hourly(str(path), pvgis_format="csv", map_variables=True)
    return to_project_frame(out[0], tz)    # dönüş demeti pvlib sürümüne göre 2 veya 3 elemanlı


def download_pvgis(lat: float, lon: float, start: int, end: int, peakpower_kw: float,
                   out_csv: str | Path, loss: float = 14.0, tz: str = DEFAULT_TZ,
                   raw_csv: str | Path | None = None) -> pd.DataFrame:
    """PVGIS API'den saatlik veriyi indirir (kendi bilgisayarında, internetle çalıştır).

    Eğim/azimut PVGIS'e optimum olarak hesaplatılır (optimalangles). Sistem kayıpları
    varsayılan %14 (PVGIS'in standart önerisi). Yıl aralığı veritabanına bağlıdır;
    kapsam dışı yıl istenirse PVGIS hata mesajı döner.
    """
    from pvlib.iotools import get_pvgis_hourly
    out = get_pvgis_hourly(lat, lon, start=start, end=end, pvcalculation=True,
                           peakpower=peakpower_kw, loss=loss, optimalangles=True,
                           components=False, map_variables=True)
    data = out[0]
    if raw_csv is not None:
        data.to_csv(raw_csv)
    df = to_project_frame(data, tz)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
