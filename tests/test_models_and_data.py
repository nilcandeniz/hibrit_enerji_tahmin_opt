"""GRU/LSTM, PVGIS ayrıştırma (UTC->yerel), EPİAŞ csv/xlsx, Optuna duman testleri."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from solarh2.config import load_config
from solarh2.data.pvgis import load_pvgis_csv
from solarh2.data.windowing import build_windows
from solarh2.models.seq2seq import Seq2SeqLSTM, Seq2SeqRNN
from solarh2.optimization.prices import load_epias_csv, validate_ptf
from tests.make_synthetic import make
from tests.make_synthetic_pvgis import make_epias_like, make_pvgis_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_rnn_forward_shape(cell):
    m = Seq2SeqRNN(4, 3, hidden_size=16, num_layers=2, cell=cell)
    out = m(torch.randn(5, 12, 4), torch.randn(5, 8, 3))
    assert out.shape == (5, 8)


def test_gru_differs_from_lstm_and_alias_works():
    g = Seq2SeqRNN(4, 3, 16, 2, cell="gru")
    l = Seq2SeqLSTM(4, 3, 16, 2)
    assert l.cell == "lstm"
    assert sum(p.numel() for p in g.parameters()) < sum(p.numel() for p in l.parameters())
    with pytest.raises(ValueError):
        Seq2SeqRNN(4, 3, cell="transformer")


@pytest.fixture(scope="module")
def pvgis_df(tmp_path_factory):
    f = tmp_path_factory.mktemp("pv") / "raw.csv"
    make_pvgis_csv(str(f), 2021, 2022)
    return load_pvgis_csv(f)


def test_pvgis_units_and_full_local_days(pvgis_df):
    df = pvgis_df
    assert df.DATE_TIME.dt.tz is None
    assert df.DATE_TIME.iloc[0] == pd.Timestamp("2021-01-02 00:00")     # kısmi ilk gün atıldı
    assert len(df) % 24 == 0
    assert df.IRRADIATION.max() < 1.5 and df.AC_POWER.max() < 10_000     # kW/m², kW


def test_pvgis_utc_to_local_shifts_solar_noon(pvgis_df):
    # İskenderun'da güneş öğlesi UTC ≈ 9.6 -> yerel (UTC+3) ≈ 12.6. Dönüşüm yoksa tepe 9-10'da olurdu.
    peak_hour = pvgis_df.groupby(pvgis_df.DATE_TIME.dt.hour).AC_POWER.mean().idxmax()
    assert peak_hour in (12, 13)


def test_epias_csv_and_xlsx_agree(tmp_path):
    make_epias_like(str(tmp_path / "p.csv"), "2023-03-01", 3, seed=1)
    make_epias_like(str(tmp_path / "p.xlsx"), "2023-03-01", 3, seed=1, xlsx=True)
    a, b = load_epias_csv(str(tmp_path / "p.csv")), load_epias_csv(str(tmp_path / "p.xlsx"))
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=0.01)
    rep = validate_ptf(a)
    assert rep["n"] == 72 and rep["missing_hours"] == 0


def test_optuna_smoke(tmp_path):
    from solarh2.tuning.search import save_results, tune_rnn, tune_xgboost
    csv = tmp_path / "s.csv"
    make(days=14).to_csv(csv, index=False)
    cfg = load_config(ROOT / "configs/lstm.yaml")
    cfg["data"].update(csv_path=str(csv), lookback=8, horizon=8, eval_stride=4)
    cfg["training"].update(max_epochs=2, patience=2, device="cpu")
    splits, _ = build_windows(cfg)
    tr, va = splits["train"], splits["val"]
    studies = {"xgboost": tune_xgboost(tr, va, 2), "gru": tune_rnn("gru", tr, va, cfg, 2, max_epochs=2)}
    best = save_results(studies, tmp_path / "t")
    assert set(best) == {"xgboost", "gru"} and (tmp_path / "t/best_params.json").exists()
    assert "hidden_size" in best["gru"]["params"] and "max_depth" in best["xgboost"]["params"]


# ---------------- ufka bağlı hata / DM testi ----------------
def _lead_cfg(tmp_path):
    from tests.make_synthetic_pvgis import make_pvgis_csv
    raw = tmp_path / "raw.csv"
    make_pvgis_csv(str(raw), 2022, 2022)
    csv = tmp_path / "pv.csv"
    load_pvgis_csv(raw).to_csv(csv, index=False)
    cfg = load_config(ROOT / "configs/pvgis_forecast.yaml")
    cfg["data"]["csv_path"] = str(csv)
    return cfg


def test_lead_time_noise_grows_with_horizon_and_scale_zero_is_exact(tmp_path):
    import copy
    cfg = _lead_cfg(tmp_path)
    (sp, sc) = build_windows(cfg, seed=1)
    c0 = copy.deepcopy(cfg); c0["data"]["weather_forecast"]["noise_scale"] = 0.0
    (sp0, sc0) = build_windows(c0, seed=1)
    j = sc["future_cols"].index("IRRADIATION__fcst")
    inv = lambda ws, s: s["future"].inverse_transform(
        ws.x_future.reshape(-1, ws.x_future.shape[-1])).reshape(ws.x_future.shape)
    f, f0 = inv(sp["test"], sc), inv(sp0["test"], sc0)
    day = sp["test"].daytime
    rel = lambda h: np.std((f[:, h, j] - f0[:, h, j])[day[:, h]] / np.maximum(f0[:, h, j][day[:, h]], 1e-3))
    assert rel(23) > 2 * rel(0) > 0
    assert np.array_equal(sp["test"].times, sp0["test"].times)        # aynı test pencereleri
    # noise_scale=0 -> decoder'a giren "tahmin" ışınımı = gerçekleşen ışınım
    actual = pd.read_csv(cfg["data"]["csv_path"], parse_dates=["DATE_TIME"]).set_index("DATE_TIME")
    true_irr = actual["IRRADIATION"].reindex(pd.DatetimeIndex(sp0["test"].times.ravel())).to_numpy()
    assert np.allclose(f0[:, :, j].ravel(), true_irr, atol=1e-5)


def test_diebold_mariano_detects_and_rejects():
    from solarh2.evaluation.metrics import diebold_mariano
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 1.0, 365)
    same = diebold_mariano(base + rng.normal(0, 0.3, 365), base + rng.normal(0, 0.3, 365))
    better = diebold_mariano(base * 0.8, base)
    assert same["p_value"] > 0.05
    assert better["p_value"] < 0.01 and better["DM"] < 0
