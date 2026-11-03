"""df_merged.csv ile aynı şemaya sahip sentetik veri (Kaggle Plant_1 benzeri, 34 gün, boşluklu).

Gerçek veri olmadan pipeline'ı test etmek içindir. Kullanım:
    python tests/make_synthetic.py data/processed/df_synthetic.csv
"""
import sys

import numpy as np
import pandas as pd


def make(days: int = 34, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-05-15", periods=days * 96, freq="15min")
    hour = np.asarray(idx.hour + idx.minute / 60)
    clear = np.clip(np.sin(np.pi * (hour - 6) / 12.5), 0, None)
    cloud_day = rng.uniform(0.5, 1.0, days).repeat(96)
    cloud_noise = np.clip(1 - np.abs(rng.normal(0, 0.15, len(idx))), 0.3, 1)
    irr = 1.1 * clear * cloud_day * cloud_noise
    amb = 25 + 8 * clear + rng.normal(0, 1, len(idx))
    mod = amb + 25 * irr + rng.normal(0, 1, len(idx))
    ac = np.clip(12000 * irr * (1 - 0.004 * (mod - 25)) + rng.normal(0, 150, len(idx)), 0, None)
    ac[irr == 0] = 0
    df = pd.DataFrame({"DATE_TIME": idx, "DC_POWER": ac * 10.2, "AC_POWER": ac,
                       "AMBIENT_TEMPERATURE": amb, "MODULE_TEMPERATURE": mod, "IRRADIATION": irr})
    drop = rng.choice(len(df), size=100, replace=False)   # gerçek veride ~100 eksik damga var
    return df.drop(index=drop).reset_index(drop=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/processed/df_synthetic.csv"
    make().to_csv(out, index=False)
    print("yazıldı:", out)
