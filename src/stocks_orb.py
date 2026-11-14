"""Opening-range breakout on US stocks: does a cheaper market make EUR 1 per trade possible?

The idea (the strategy at the heart of the day-trader-skill pack):
  1. The opening range is the highest and lowest price in the first 15 or 30 minutes of trading.
  2. Buy when a 5-minute candle CLOSES above the range (or sell short below it, if allowed).
  3. Stop at the middle of the range, or at its other side. Target: 1x or 2x the risk, or the
     price that makes EUR 1 on EUR 100. Close everything by 3:55 pm New York time.

Why stocks: US brokers such as Alpaca charge no commission on shares, so the main cost is the
spread (the gap between buy and sell prices), about 0.02% per buy or sell for big, busy stocks,
instead of Kraken's 0.80% crypto fee.

Honest test: Yahoo keeps 60 days of 5-minute bars. The first half of the sessions picks the best
of 24 settings; the second half, which played no part in choosing, shows how it really does.

    py src\\stocks_orb.py download     # saves Yahoo's 5-minute bars to data/stocks/ (once a month is enough)
    py src\\stocks_orb.py test         # runs the study on everything downloaded so far

Prices are in US dollars; profits are shown per 100 of stake, ignoring the euro/dollar rate.
"""
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
           "AMD", "NFLX", "AVGO", "JPM", "BAC", "XOM", "KO", "PEP", "WMT", "COST",
           "UNH", "LLY", "V", "MA", "DIS", "INTC", "CSCO", "ORCL", "CRM", "ADBE"]
URL = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (crypto-paper-trader research script)"}

STAKE = 100.0
TARGET_PROFIT = 1.0
COST_LEVELS = {"spread 0.02% (big stocks)": 0.0002, "spread 0.05% (cautious)": 0.0005}
COMMISSION = 0.0                 # Alpaca: no commission on US shares
ENTRY_DEADLINE = "11:30"         # no new entries after this New York time
FLAT_BY = "15:55"                # close any open trade at the close of this bar
BARS_PER_SESSION = 78            # 9:30 to 16:00 in 5-minute bars
MIN_SEARCH_TRADES = 20

SETTINGS = {
    "range_minutes": [15, 30],
    "stop": ["range middle", "other side of range"],
    "target": ["1R", "2R", "EUR 1"],
    "direction": ["long only", "long and short"],
}

PROJECT_DIR = Path(__file__).resolve().parent.parent
STOCKS_DIR = PROJECT_DIR / "data" / "stocks"
OUTPUT_DIR = PROJECT_DIR / "output"


# =============================================================================
# Data
# =============================================================================
def download():
    STOCKS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    for ticker in TICKERS:
        response = requests.get(URL.format(ticker=ticker), params={"range": "60d", "interval": "5m"},
                                headers=HEADERS, timeout=30)
        if response.status_code != 200:
            print(f"{ticker}: HTTP {response.status_code}, skipped")
            continue
        path = STOCKS_DIR / f"{ticker}_5m_{stamp}.json"
        path.write_bytes(response.content)
        n = len(response.json()["chart"]["result"][0].get("timestamp", []))
        print(f"{ticker}: {n:,} five-minute bars -> {path.relative_to(PROJECT_DIR)}")
        time.sleep(1.0)   # be polite: one request a second


def load_ticker(ticker):
    """All downloads for one ticker, combined, as full New York sessions only."""
    frames = []
    for f in sorted(STOCKS_DIR.glob(f"{ticker}_5m_*.json")):
        r = json.loads(f.read_text())["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        frames.append(pd.DataFrame({"ts": r["timestamp"], "open": q["open"], "high": q["high"],
                                    "low": q["low"], "close": q["close"], "volume": q["volume"]}))
    if not frames:
        return None
    df = pd.concat(frames).dropna().drop_duplicates("ts", keep="last").sort_values("ts")
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df = df[df["time"].dt.second == 0]                      # drop Yahoo's live, unfinished bar
    df["date"] = df["time"].dt.date
    df["clock"] = df["time"].dt.strftime("%H:%M")
    df = df[(df["clock"] >= "09:30") & (df["clock"] <= "15:55")]
    full = df.groupby("date")["ts"].transform("size") == BARS_PER_SESSION   # no half days or gaps
    return df[full].reset_index(drop=True)


# =============================================================================
# The strategy, one session at a time
# =============================================================================
def trade_session(day, range_minutes, stop_rule, target_rule, direction, cost):
    """Return None (no trade) or a dict describing the one trade taken that day."""
    o, h, l, c, clock = (day[k].to_numpy() for k in ["open", "high", "low", "close", "clock"])
    n_range = range_minutes // 5
    top, bottom = h[:n_range].max(), l[:n_range].min()
    middle = (top + bottom) / 2
    for i in range(n_range, len(c)):
        if clock[i] > ENTRY_DEADLINE:
            return None
        side = 1 if c[i] > top else (-1 if c[i] < bottom and direction == "long and short" else 0)
        if side:
            break
    else:
        return None

    entry = c[i]
    stop = middle if stop_rule == "range middle" else (bottom if side == 1 else top)
    risk = abs(entry - stop)
    if risk <= 0 or risk / entry < 2 * cost:     # a stop inside the spread is not a real plan
        return None
    if target_rule == "EUR 1":
        # price move that leaves EUR 1 on EUR 100 after paying the cost on the way in and out
        move = (1 + TARGET_PROFIT / STAKE) * (1 + cost) / ((1 - cost) * (1 - COMMISSION) ** 2) - 1
        target = entry * (1 + side * move)
    else:
        target = entry + side * float(target_rule[0]) * risk

    exit_price, reason = c[-1], "flat by 3:55"
    for j in range(i + 1, len(c)):
        hit_stop = l[j] <= stop if side == 1 else h[j] >= stop
        hit_target = h[j] >= target if side == 1 else l[j] <= target
        if hit_stop:                                   # same bar: assume the stop came first
            exit_price, reason = stop, "stop"
            break
        if hit_target:
            exit_price, reason = target, "target"
            break
    gross = side * (exit_price / entry - 1)
    net = (1 + gross) * (1 - cost) / (1 + cost) - 1 - 2 * COMMISSION
    return {"date": day["date"].iloc[0], "entry_time": clock[i], "side": "long" if side == 1 else "short",
            "entry": entry, "stop": stop, "target": target, "exit": exit_price, "reason": reason,
            "R": side * (exit_price - entry) / risk, "profit": STAKE * net}


def run_setting(sessions, setting, cost):
    trades = [t for day in sessions if (t := trade_session(day, *setting, cost)) is not None]
    return pd.DataFrame(trades)


def summarise(trades):
    if trades.empty:
        return {"trades": 0, "target_%": 0.0, "stop_%": 0.0, "avg_R": 0.0, "profit": 0.0, "per_trade": 0.0}
    return {"trades": len(trades),
            "target_%": round(100 * (trades["reason"] == "target").mean(), 1),
            "stop_%": round(100 * (trades["reason"] == "stop").mean(), 1),
            "avg_R": round(trades["R"].mean(), 3),
            "profit": round(trades["profit"].sum(), 2),
            "per_trade": round(trades["profit"].mean(), 3)}


def test():
    data = {t: d for t in TICKERS if (d := load_ticker(t)) is not None and not d.empty}
    if not data:
        raise SystemExit("No stock data. Run py src\\stocks_orb.py download first.")
    dates = sorted(set().union(*[set(d["date"]) for d in data.values()]))
    split = dates[len(dates) // 2]
    print(f"{len(data)} tickers, {len(dates)} sessions ({dates[0]} to {dates[-1]}).")
    print(f"Search: {dates[0]} to {dates[len(dates) // 2 - 1]}   Test: {split} to {dates[-1]}")

    sessions = {t: [day for _, day in d.groupby("date")] for t, d in data.items()}
    grid = list(itertools.product(*SETTINGS.values()))
    rows, chosen_rows, all_trades = [], [], []
    for cost_label, cost in COST_LEVELS.items():
        per_setting = []
        for setting in grid:
            trades = pd.concat([run_setting(s, setting, cost).assign(ticker=t) for t, s in sessions.items()],
                               ignore_index=True)
            if trades.empty:
                continue
            search, later = trades[trades["date"] < split], trades[trades["date"] >= split]
            a, b = summarise(search), summarise(later)
            row = {"costs": cost_label, **dict(zip(SETTINGS, setting)),
                   **{f"search_{k}": v for k, v in a.items()}, **{f"test_{k}": v for k, v in b.items()}}
            per_setting.append(row)
            all_trades.append(trades.assign(costs=cost_label, setting=" | ".join(map(str, setting))))
        table = pd.DataFrame(per_setting)
        rows.append(table)
        ok = table[table["search_trades"] >= MIN_SEARCH_TRADES]
        best = ok.loc[ok["search_profit"].idxmax()]
        chosen_rows.append(best)

        print(f"\n=== {cost_label}: all {len(table)} settings ===")
        show = list(SETTINGS) + ["search_trades", "search_target_%", "search_per_trade", "test_trades",
                                 "test_target_%", "test_stop_%", "test_avg_R", "test_profit", "test_per_trade"]
        print(table.sort_values("search_profit", ascending=False)[show].to_string(index=False))
        agree = table["search_per_trade"].corr(table["test_per_trade"])
        print(f"Do good settings stay good? Correlation of per-trade results, search vs test: {agree:+.2f}")
        print(f"Settings that made money in the test half: {(table['test_profit'] > 0).sum()} of {len(table)}")
        print(f"Chosen on the search half: {', '.join(str(best[k]) for k in SETTINGS)} -> "
              f"search {best['search_per_trade']:+.3f}/trade, TEST {best['test_per_trade']:+.3f}/trade "
              f"over {int(best['test_trades'])} trades ({best['test_profit']:+.2f} per 100 stake each)")

    # Benchmark: holding SPY through the test half
    if "SPY" in data:
        spy = data["SPY"]
        later = spy[spy["date"] >= split]
        hold = STAKE * (later["close"].iloc[-1] / later["open"].iloc[0] - 1)
        print(f"\nBenchmark: holding SPY through the test half made {hold:+.2f} per 100.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.concat(rows).to_csv(OUTPUT_DIR / f"stocks_orb_settings_{stamp}.csv", index=False)
    pd.concat(all_trades).to_csv(OUTPUT_DIR / f"stocks_orb_trades_{stamp}.csv", index=False)
    print(f"\nSaved: output/stocks_orb_settings_{stamp}.csv, output/stocks_orb_trades_{stamp}.csv")


if __name__ == "__main__":
    {"download": download, "test": test}.get(sys.argv[1] if len(sys.argv) > 1 else "test", test)()
