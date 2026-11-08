"""Download free macro data from FRED (Federal Reserve Bank of St. Louis). No account needed.

Each series is saved untouched in data/macro/ as fred_<SERIES>.csv, replacing the previous copy
(FRED republishes the whole history, so the newest download is always the complete one).
"""
from pathlib import Path

import pandas as pd
import requests

SERIES = {
    "CBBTCUSD": "Bitcoin price (Coinbase), daily since Dec 2014",
    "CBETHUSD": "Ether price (Coinbase), daily since 2016",
    "NASDAQCOM": "NASDAQ Composite stock index, daily",
    "DTWEXBGS": "US dollar against a broad basket of currencies, daily",
    "DFF": "US federal funds interest rate, daily",
    "M2SL": "US M2 money supply, monthly",
}
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

PROJECT_DIR = Path(__file__).resolve().parent.parent
MACRO_DIR = PROJECT_DIR / "data" / "macro"


def main():
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    for series, label in SERIES.items():
        response = requests.get(URL, params={"id": series}, timeout=60)
        response.raise_for_status()
        path = MACRO_DIR / f"fred_{series}.csv"
        path.write_bytes(response.content)
        df = pd.read_csv(path)
        values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        print(f"{series:10s} {len(df):6,} rows  {df.iloc[0, 0]} to {df.iloc[-1, 0]}  "
              f"({values.isna().sum()} blank)  {label}")


if __name__ == "__main__":
    main()
