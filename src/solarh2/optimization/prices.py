"""PTF (piyasa takas fiyatı) yükleme ve zaman hizalama.

PTF saatliktir (gün öncesi piyasasında saat başına tek fiyat); 15 dk'lık ızgaraya
adım-sabit (forward fill) yayılır.

Tarih uyuşmazlığı hakkında ÖNEMLİ not
-------------------------------------
Kaggle santrali 2020'de Hindistan'da ölçülmüş; EPİAŞ fiyatları Türkiye'de ve başka bir
dönemde. İkisini takvim tarihine göre eşleştirmek anlamsızdır. Bu yüzden:
  * mode="cycle":    test günleri i = 0,1,2,... -> fiyat dosyasının i. günü (mod N). Gün-içi
                     şekil (ördek eğrisi) korunur, tarih rastlantısal eşleşir. Tezde
                     "senaryo tabanlı değerlendirme" olarak açıkça belirtilmelidir.
  * mode="calendar": zaman damgasıyla birebir eşleştirir (yalnızca tarihler örtüşüyorsa).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tr_number(x) -> float:
    """'1.234,56' -> 1234.56 ; zaten sayıysa dokunmaz."""
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(".", "").replace(",", ".")
    return float(s)


def load_epias_csv(path: str) -> pd.Series:
    """EPİAŞ Şeffaflık Platformu'ndan indirilen PTF dosyasını (csv/xlsx) saatlik Series'e çevirir.

    Beklenen: bir tarih(+saat) sütunu ve adında 'PTF' geçen bir fiyat sütunu. Ayraç ve
    Türkçe ondalık biçimi otomatik tanınır. DOSYANIZIN BAŞLIKLARI FARKLIYSA bu fonksiyonu
    uyarlayın — gerçek bir export ile ilk kez çalıştırırken çıktıyı elle kontrol edin.
    """
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)          # openpyxl gerekir
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    price_col = next(c for c in df.columns if "ptf" in c.lower() or "mcp" in c.lower())
    date_col = next(c for c in df.columns if c.lower().startswith(("tarih", "date")))
    ts = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce", utc=False)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    if "saat" in " ".join(df.columns).lower():  # tarih ve saat ayrı sütunlarda
        hcol = next(c for c in df.columns if "saat" in c.lower() or c.lower() == "hour")
        ts = ts.dt.normalize() + pd.to_timedelta(
            df[hcol].astype(str).str.extract(r"(\d+)")[0].astype(int), unit="h")
    s = pd.Series(df[price_col].map(_tr_number).to_numpy(), index=ts).sort_index()
    s = s[~s.index.duplicated()].dropna()
    return s


def validate_ptf(s: pd.Series, start=None, end=None) -> dict:
    """Yüklenen PTF serisinin kapsama/aralık raporu (biçim hatalarını erken yakalamak için)."""
    start = pd.Timestamp(start) if start is not None else s.index.min().floor("D")
    end = pd.Timestamp(end) if end is not None else s.index.max().ceil("D")
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    missing = expected.difference(s.index)
    return dict(n=len(s), first=str(s.index.min()), last=str(s.index.max()),
                expected_hours=len(expected), missing_hours=len(missing),
                min=float(s.min()), max=float(s.max()), mean=float(s.mean()),
                zero_hours=int((s == 0).sum()),
                missing_sample=[str(t) for t in missing[:5]])


def synthetic_ptf(days: int = 60, seed: int = 0, start: str = "2026-07-01") -> pd.Series:
    """Test amaçlı sentetik PTF (TL/MWh): öğle çukuru + akşam puantı ('ördek eğrisi').

    Şekil, EPİAŞ günlük raporlarında 2026 yazında görülen yapıya benzer: gündüz saatleri
    düşük, 17-21 puant yüksek. Seviyeler gerçek veriyle DEĞİŞTİRİLMELİDİR.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    h = idx.hour.to_numpy()
    base = np.interp(h, [0, 5, 8, 12, 16, 19, 21, 23], [2300, 2300, 1500, 600, 1500, 3800, 3600, 2800])
    daily = rng.normal(1.0, 0.12, days).repeat(24)
    noise = rng.normal(0, 150, len(idx))
    return pd.Series(np.clip(base * daily + noise, 0, 4500), index=idx, name="PTF")


def align_prices(times: np.ndarray, ptf: pd.Series, mode: str = "cycle",
                 day0: pd.Timestamp | None = None) -> np.ndarray:
    """times: (T,) datetime64 hedef zaman damgaları -> (T,) TL/MWh (adım-sabit)."""
    t = pd.DatetimeIndex(times)
    if mode == "calendar":
        out = ptf.reindex(t.floor("h"))
        if out.isna().any():
            raise ValueError("Fiyat serisi test tarihlerini kapsamıyor; mode='cycle' kullanın.")
        return out.to_numpy(dtype=float)
    # cycle
    ptf_days = ptf.index.normalize().unique().sort_values()
    n_days = len(ptf_days)
    day0 = (day0 or t.min()).normalize()
    off = ((t.normalize() - day0).days.to_numpy()) % n_days
    mapped = ptf_days[off] + pd.to_timedelta(t.hour.to_numpy(), unit="h")
    out = ptf.reindex(mapped)
    if out.isna().any():  # fiyat dosyasında eksik saat -> aynı güne ait en yakın dolgu
        out = out.ffill().bfill()
    return out.to_numpy(dtype=float)
