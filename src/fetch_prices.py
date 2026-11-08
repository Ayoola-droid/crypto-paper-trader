"""Download recent prices for several coins from Kraken and take a first look.

No account or API key is needed: Kraken's price data is public.
Each run saves new CSVs in data/raw/ and never overwrites an earlier one.
Kraken returns the latest 720 candles, so:
  1-minute candles  = the last 12 hours
  15-minute candles = the last 7.5 days
"""
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Kraken's names: XBT = Bitcoin, ETH = Ether, SOL = Solana, XRP = Ripple, XDG = Dogecoin
PAIRS = ["XBTEUR", "ETHEUR", "SOLEUR", "XRPEUR", "XDGEUR"]
INTERVALS = [1, 15]      # minutes per candle
URL = "https://api.kraken.com/0/public/OHLC"

# Kraken Pro spot fee for the lowest tier (checked 4 Nov 2026 on kraken.com/features/fee-schedule).
TAKER_FEE = 0.0080
ROUND_TRIP_FEE = 2 * TAKER_FEE

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"


def fetch_candles(pair, interval):
    """Ask Kraken for the latest candles and return them as a table."""
    response = requests.get(URL, params={"pair": pair, "interval": interval}, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if payload["error"]:
        raise RuntimeError(f"Kraken returned an error for {pair}: {payload['error']}")

    result = payload["result"]
    pair_key = next(key for key in result if key != "last")
    columns = ["time", "open", "high", "low", "close", "vwap", "volume", "trades"]
    df = pd.DataFrame(result[pair_key], columns=columns)

    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # The last candle is still forming, so drop it: every row we keep is final.
    return df.iloc[:-1].reset_index(drop=True)


def save_raw(df, pair, interval, stamp):
    """Save a new, timestamped CSV so raw data is never overwritten."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{pair}_{interval}m_{stamp}.csv"
    df.to_csv(path, index=False)
    return path


def inspect(df, pair, interval):
    """Return one row of first-look measures for the summary table."""
    moves = df["close"].pct_change().abs().dropna()
    gaps = df["time"].diff().dt.total_seconds().gt(interval * 60).sum()
    return {
        "pair": pair,
        "candle": f"{interval}m",
        "rows": len(df),
        "from": df["time"].iloc[0].strftime("%d %b %H:%M"),
        "to": df["time"].iloc[-1].strftime("%d %b %H:%M"),
        "gaps": int(gaps),
        "no_trade": int((df["trades"] == 0).sum()),
        "avg_move_%": round(moves.mean() * 100, 3),
        "change_%": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2),
        "beat_fees": f"{(moves > ROUND_TRIP_FEE).sum()} of {len(moves)}",
    }


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = []
    for interval in INTERVALS:
        for pair in PAIRS:
            df = fetch_candles(pair, interval)
            save_raw(df, pair, interval, stamp)
            summary.append(inspect(df, pair, interval))
            time.sleep(1)   # be polite: Kraken limits how fast public requests can be made

    print(f"Saved {len(summary)} files in data/raw/ (stamp {stamp})")
    print(f"Fees to buy and sell once: {ROUND_TRIP_FEE:.2%}")
    print()
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
