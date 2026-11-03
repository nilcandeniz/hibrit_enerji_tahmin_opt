"""EPİAŞ PTF dosyasını (csv/xlsx) okuyup kapsama raporu verir; PVGIS test yılıyla örtüşmeyi sınar.

Kullanım:  python scripts/check_ptf.py data/raw/ptf_2023.csv --pv data/processed/df_pvgis.csv
"""
import argparse

import pandas as pd

from solarh2.optimization.prices import load_epias_csv, validate_ptf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptf")
    ap.add_argument("--pv", default=None)
    args = ap.parse_args()

    ptf = load_epias_csv(args.ptf)
    rep = validate_ptf(ptf)
    print("PTF raporu:", rep)
    print("İlk 3 satır (elle kontrol edin):\n", ptf.head(3).to_string())
    if rep["missing_hours"]:
        print("UYARI: eksik saat var; Şubat/Mart ve Ekim sonu gibi geçiş günlerine bakın.")
    if args.pv:
        pv = pd.read_csv(args.pv, parse_dates=["DATE_TIME"]).set_index("DATE_TIME")
        both = pv.index.intersection(ptf.index)
        print(f"PV ile örtüşen saat: {len(both)} / {len(pv)}  "
              f"({both.min()} -> {both.max()})" if len(both) else "PV ile örtüşme YOK")


if __name__ == "__main__":
    main()
