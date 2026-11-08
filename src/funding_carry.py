"""Lever 4: funding-rate carry. Earn money without predicting the price.

How it works, in plain words:
  Perpetual futures never expire, so the exchange keeps their price close to the real coin price
  with "funding" payments. When most traders bet on a rise, the futures price drifts above the
  coin price, and the people betting up (longs) pay the people betting down (shorts) every period.
  Carry trade: buy EUR 100 of the coin AND short EUR 100 of its perpetual future. If the price
  rises, one side gains what the other loses, so price moves roughly cancel. What is left is the
  funding you collect as the short, minus fees for opening and closing both sides.

This script downloads Kraken Futures' public funding history (no account needed) and measures:
  - always on:  hold the carry trade for the whole period
  - switching:  hold it only in weeks after funding was positive on average (decided using only
                the past week), paying fees every time the trade is opened or closed
Results are per EUR 100 in each leg.

Kraken's perpetuals are priced in US dollars, so a real version also carries euro/dollar risk.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SYMBOLS = {"XBTEUR": "PF_XBTUSD", "ETHEUR": "PF_ETHUSD", "SOLEUR": "PF_SOLUSD",
           "XRPEUR": "PF_XRPUSD", "XDGEUR": "PF_DOGEUSD"}
URLS = [   # Kraken documents more than one path for this endpoint; try them in turn
    "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates",
    "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates",
    "https://futures.kraken.com/derivatives/api/v3/historicalfundingrates",
]
STAKE = 100.0
COSTS = {   # fee per buy or sell, per leg: (spot leg, futures leg)
    "taker (market orders)": (0.0080, 0.0005),
    "maker (limit orders)": (0.0040, 0.0002),
}
SLIPPAGE = 0.0005          # per buy or sell, market orders only
SWITCH_LOOKBACK_DAYS = 7

PROJECT_DIR = Path(__file__).resolve().parent.parent
FUNDING_DIR = PROJECT_DIR / "data" / "funding"
OUTPUT_DIR = PROJECT_DIR / "output"


def download(symbol):
    """Download the funding history for one perpetual and save it untouched."""
    last_error = None
    for url in URLS:
        try:
            response = requests.get(url, params={"symbol": symbol}, timeout=30)
            if response.status_code != 200:
                last_error = f"{url} -> HTTP {response.status_code}"
                continue
            rates = response.json().get("rates", [])
            if rates:
                df = pd.DataFrame(rates)
                FUNDING_DIR.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
                df.to_csv(FUNDING_DIR / f"{symbol}_funding_{stamp}.csv", index=False)
                return df
            last_error = f"{url} -> no rates"
        except requests.RequestException as error:
            last_error = f"{url} -> {error}"
    raise RuntimeError(f"Could not download {symbol}: {last_error}")


def prepare(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["timestamp"], utc=True)
    # relativeFundingRate is a fraction of the position per period, paid by longs to shorts if positive
    df["rate"] = df["relativeFundingRate"].astype(float)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def round_trip_cost(spot_fee, futures_fee, limit):
    """Open and close both legs once: 4 fills in total."""
    slip = 0.0 if limit else SLIPPAGE
    return STAKE * (2 * (spot_fee + slip) + 2 * (futures_fee + slip))


def analyse(pair, df):
    periods_per_day = round(pd.Timedelta(days=1) / df["time"].diff().median())
    days = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 86400
    earned = STAKE * df["rate"]                   # what the short leg receives each period
    rows = []
    for cost_label, (spot_fee, fut_fee) in COSTS.items():
        trip = round_trip_cost(spot_fee, fut_fee, cost_label.startswith("maker"))

        # Always on: open once at the start, close once at the end
        always = earned.sum() - trip

        # Switching: decide each week from the previous week's average funding only
        daily = df.set_index("time")["rate"].resample("1D").sum()
        weekly_signal = daily.rolling(SWITCH_LOOKBACK_DAYS).mean().shift(1)   # known at the start of the day
        on = (weekly_signal > 0).astype(int)
        opens = int(((on == 1) & (on.shift(fill_value=0) == 0)).sum())   # each open is later closed
        switching = (STAKE * daily * on).sum() - opens * trip

        rows.append({
            "pair": pair, "costs": cost_label,
            "from": f"{df['time'].iloc[0]:%d %b %Y}", "to": f"{df['time'].iloc[-1]:%d %b %Y}",
            "periods_per_day": periods_per_day,
            "share_positive_%": round(100 * (df["rate"] > 0).mean(), 1),
            "funding_EUR": round(earned.sum(), 2),
            "round_trip_cost_EUR": round(trip, 2),
            "always_on_EUR": round(always, 2),
            "always_on_per_year_%": round(100 * always / STAKE * 365 / days, 1) if days else 0.0,
            "days_to_earn_EUR1": round(days / earned.sum(), 1) if earned.sum() > 0 else None,
            "switching_EUR": round(switching, 2),
            "times_opened": opens,
        })
    return rows


def main():
    rows = []
    for pair, symbol in SYMBOLS.items():
        try:
            df = prepare(download(symbol))
        except RuntimeError as error:
            print(error)
            continue
        print(f"{pair} ({symbol}): {len(df):,} funding periods")
        rows += analyse(pair, df)
    if not rows:
        raise SystemExit("No funding data downloaded.")

    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    table.to_csv(OUTPUT_DIR / f"funding_carry_{stamp}.csv", index=False)
    pd.set_option("display.width", 200)
    for cost_label in COSTS:
        print(f"\n=== Carry trade with {cost_label}, EUR {STAKE:.0f} in each leg ===")
        print(table[table["costs"] == cost_label].drop(columns="costs").to_string(index=False))
    print(f"\nSaved: output/funding_carry_{stamp}.csv; raw funding in data/funding/")


if __name__ == "__main__":
    main()
