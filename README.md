# PV Tahmini + MILP Tabanlı Şebeke / Yeşil Hidrojen Karar Destek Sistemi

İskenderun Teknik Üniversitesi, Bilgisayar Mühendisliği Bitirme Projesi (2026-2027 Güz).
**Sistem tanımı:** PV üretimi + PEM elektrolizör + hidrojen depolama hibrit entegrasyonu
(rüzgâr türbini kapsam dışıdır; rüzgâr hızı yalnızca meteorolojik girdi olarak kullanılır).

## İki veri hattı (bilinçli ayrım)

| Hat | Veri | Rolü |
|---|---|---|
| **Doğrulama** | Kaggle Plant_1 (gerçek santral telemetrisi, 15 dk, 34 gün) | Tahmin yönteminin *ölçülmüş* veride sınanması |
| **Karar katmanı** | PVGIS saatlik (İskenderun, çok yıllı) + EPİAŞ PTF (gerçek tarihler) | Tahmin → MILP → ekonomik değer |

PVGIS'teki PV gücü bir model çıktısıdır (ışınım ve sıcaklıktan hesaplanır, ölçüm değildir);
bu yüzden PVGIS üzerindeki tahmin başarısı gerçek telemetrinin yerine geçmez.

## Kurulum
```
pip install -e ".[dev,data]"      # pulp<4 sabitlidir (PuLP 4.x klasik API'yi kırar)
```

## Akış

**A. Doğrulama hattı (Kaggle)**
```
python scripts/tune.py          --config configs/lstm.yaml                 # Optuna (XGBoost, LSTM, GRU)
python scripts/train_compare.py --config configs/lstm.yaml --tuned results/lstm_run/tuning/best_params.json
```

**B. Karar katmanı (PVGIS + EPİAŞ)**
```
python scripts/fetch_pvgis.py --lat 36.587 --lon 36.173 --start 2019 --end 2023 --peak-kw 10000 \
       --out data/processed/df_pvgis.csv             # internet gerekir; PVGIS yıl kapsamını denetleyin
# EPİAŞ Şeffaflık Platformu -> Gün Öncesi Piyasası -> PTF -> dışa aktar -> data/raw/ptf_2023.csv|xlsx
python scripts/check_ptf.py data/raw/ptf_2023.csv --pv data/processed/df_pvgis.csv   # kapsama kontrolü
python scripts/tune.py          --config configs/pvgis_forecast.yaml
python scripts/train_compare.py --config configs/pvgis_forecast.yaml --tuned results/pvgis_run/tuning/best_params.json
python scripts/run_optimization.py --config configs/optimization_pvgis.yaml
```

**C. Senaryo analizleri (önerideki 11-12. hafta)**
```
# (c) Çoklu seed + Diebold-Mariano: "GRU > LSTM" iddiasının sağlamlığı
python scripts/multi_seed.py --config configs/pvgis_forecast.yaml \
       --tuned results/pvgis_run/tuning/best_params.json --seeds 0 1 2 3 4
# (b) Tahmin hatası taraması: hata düzeyi -> nRMSE -> TL kaybı (her düzeyde modeller yeniden eğitilir)
python scripts/forecast_value_sweep.py --config configs/pvgis_forecast.yaml \
       --opt-config configs/optimization_pvgis.yaml \
       --tuned results/pvgis_run/tuning/best_params.json --levels 0 0.5 1 1.5 2
```
(a) SMF'li dengesizlik ve gerçek zamanlı düzeltme `run_optimization.py` içinde kendiliğinden çalışır
(`prices.smf_csv`, `study.recourse_modes`).

**Colab (GPU):** `!git clone <repo> && cd <repo> && pip install -e ".[data]"`; veriyi `data/` altına
yükleyin (Drive'dan kopyalayın). LSTM/GRU GPU'yu kendiliğinden kullanır; MILP CPU'da çalışır.
Uzun koşularda `results/`'ı Drive'a kopyalamayı unutmayın (Colab oturumu kapanınca silinir).

**Optuna notları:** Arama `<output_dir>/tuning/optuna.db` dosyasına kaydedilir; Ctrl+C ile kesip
aynı komutu tekrar çalıştırırsan kaldığı yerden devam eder. Modeller parça parça da koşulabilir
(`--only xgboost`, `--only lstm`, `--only gru`); sonuçlar `best_params.json` içinde birleşir.

`pytest -q` — pencereleme/sızıntı, MILP kısıtları, PVGIS ayrıştırma, EPİAŞ okuma, Optuna testleri.

## Öneri raporu ↔ kod

| Rapor | Kod |
|---|---|
| Veri ön işleme, sin/cos, lag, normalizasyon | `data/windowing.py` (lag → geçmiş penceresi + `y(t−24s)`; ölçekleme: StandardScaler, yalnızca train'e fit) |
| XGBoost vs LSTM/GRU, 24 saat | `models/baselines.py`, `models/seq2seq.py` |
| Hiperparametre optimizasyonu (Optuna) | `tuning/search.py`, `scripts/tune.py` |
| RMSE / MAE / R² | `evaluation/metrics.py` (+ nRMSE, gündüz maskeli, skill score) |
| MILP (PuLP), min yük, başlatma/aşınma, tank | `optimization/milp.py` |
| PTF (saatlik) | `optimization/prices.py` |
| Dashboard (Streamlit) | henüz yok — son aşama |

## MILP formülasyonunun öneri raporundan farkları (tezde gerekçelendirilmeli)
1. Zaman adımı `dt` açık: `enerji = P·dt`, `m_H2 = P_ely·dt / (HHV/η)`.
2. H2 geliri **satıştan** (`d_t`) doğar; üretimden değil (aksi halde tank dengesiyle çifte sayım).
3. `u(t)·C_deg` yerine başlatma ikilisi `y_t` + başlatma maliyeti + enerjiye bağlı bozunma maliyeti.
4. Gün sonu stoğu `S_T ≥ S_0` (ufuk-sonu boşaltma hilesini engeller).

## Bilinen sınırlar
- `PLACEHOLDER` işaretli parametreler (elektrolizör boyutu/verimi, başlatma maliyeti, H2 fiyatı)
  literatürle doldurulmalı ve duyarlılık analizine sokulmalıdır.
- Dengesizlik: eksik 1,03·max(PTF,SMF), fazla 0,97·min(PTF,SMF). %3 marjın 2023 için geçerliliği
  resmi kaynaktan doğrulanmalı. SMF verilmezse SMF = PTF (iyimser) varsayılır.
- Gerçek zamanlı düzeltme miyoptur (yalnızca o saatin marjinal değerlerine bakar) ve kapalı
  elektrolizörü açmaz; bu yüzden hibrit sistemin tampon değerini *alt sınırdan* ölçer.
- Hava tahmini girdisi **simüle** edilir (gerçek NWP arşivi yok): her pencere ayrı bir tahmin
  yayınıdır, hata ufukla doğrusal büyür (`lead_time`). Düzeyler NWP literatürüyle kalibre edilmeli.
- Gün öncesi planı 00:00 penceresiyle yapılır; gerçek piyasada teklif ~ önceki gün öğlen verilir
  (son gözlemden ~14 saat boşluk). Bu boşluk modellenmemiştir.
- EPİAŞ okuyucusu gerçek bir export ile henüz denenmemiştir (sentetik csv/xlsx ile test edildi).
