"""PVGIS saatlik veriyi indirir ve proje şemasına çevirir (kendi bilgisayarında çalıştır).

Kullanım (İskenderun, 10 MWp, 2019-2023):
  python scripts/fetch_pvgis.py --lat 36.587 --lon 36.173 --start 2019 --end 2023 \
         --peak-kw 10000 --out data/processed/df_pvgis.csv

İnternet yoksa/PVGIS arayüzünden elle indirdiysen:
  python scripts/fetch_pvgis.py --from-csv data/raw/pvgis_raw.csv --out data/processed/df_pvgis.csv
"""
import argparse

from solarh2.data.pvgis import download_pvgis, load_pvgis_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=36.587)     # İskenderun
    ap.add_argument("--lon", type=float, default=36.173)
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--peak-kw", type=float, default=10000.0)
    ap.add_argument("--loss", type=float, default=14.0)
    ap.add_argument("--out", default="data/processed/df_pvgis.csv")
    ap.add_argument("--from-csv", default=None, help="elle indirilmiş ham PVGIS CSV")
    args = ap.parse_args()

    if args.from_csv:
        df = load_pvgis_csv(args.from_csv)
        df.to_csv(args.out, index=False)
    else:
        df = download_pvgis(args.lat, args.lon, args.start, args.end, args.peak_kw, args.out,
                            loss=args.loss)
    print(f"{len(df)} satır | {df.DATE_TIME.min()} -> {df.DATE_TIME.max()} | "
          f"AC maks {df.AC_POWER.max():.0f} kW | yazıldı: {args.out}")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
