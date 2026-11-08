"""Download months of hourly price history from Kraken's free quarterly archives.

Kraken's live API only returns the latest 720 candles (30 days of hourly data).
Its quarterly archives hold every candle for every coin, so we download them once,
keep only the hourly files for our five coins and delete the large ZIP afterwards.

Archive rows have no header: timestamp, open, high, low, close, volume, trades.
"""
import zipfile
from pathlib import Path

import requests

QUARTERS = ["2026Q1", "2026Q2", "2026Q3"]
URL = "https://assets.kraken.com/marketing/institutions/Kraken_OHLCVT_{quarter}.zip"

# Kraken uses two spellings for some coins, so we accept either.
PAIRS = {
    "XBTEUR": ["XBTEUR", "XXBTZEUR"],
    "ETHEUR": ["ETHEUR", "XETHZEUR"],
    "SOLEUR": ["SOLEUR"],
    "XRPEUR": ["XRPEUR", "XXRPZEUR"],
    "XDGEUR": ["XDGEUR", "XXDGZEUR"],
}
INTERVAL = 60   # minutes: hourly candles

PROJECT_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_DIR / "data" / "archive"   # temporary, not committed
HISTORY_DIR = PROJECT_DIR / "data" / "history"


def download(quarter):
    """Stream one quarterly ZIP to disk, showing progress."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE_DIR / f"Kraken_OHLCVT_{quarter}.zip"
    if path.exists():
        print(f"{path.name} already downloaded")
        return path

    with requests.get(URL.format(quarter=quarter), stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        print(f"Downloading {path.name} ({total / 1e9:.2f} GB)")
        done = 0
        partial = path.with_suffix(".part")
        with open(partial, "wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:6.1%}", end="")
        print()
    partial.rename(path)   # only a finished download gets the real name
    return path


def extract(zip_path, quarter):
    """Copy out only the hourly CSV for each of our coins."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = {Path(n).name.upper(): n for n in archive.namelist()}
        for pair, spellings in PAIRS.items():
            member = next((names[f"{s}_{INTERVAL}.CSV"] for s in spellings
                           if f"{s}_{INTERVAL}.CSV" in names), None)
            if member is None:
                close = sorted(n for n in names if n.startswith(pair[:3]))[:10]
                print(f"  {pair}: not found. Similar names: {close}")
                continue
            out = HISTORY_DIR / f"{pair}_{INTERVAL}m_{quarter}.csv"
            out.write_bytes(archive.read(member))
            rows = out.read_text().count("\n")
            print(f"  {pair}: {rows:,} hourly rows -> {out.relative_to(PROJECT_DIR)}")


def main():
    for quarter in QUARTERS:
        zip_path = download(quarter)
        extract(zip_path, quarter)
        zip_path.unlink()   # the ZIP holds every coin and is large; we keep only what we need
        print(f"Deleted {zip_path.name}")


if __name__ == "__main__":
    main()
