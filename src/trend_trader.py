"""Lesson 4: paper-trade a trend-following strategy that goes long or short, on several coins.

Indicators used (all standard ones traders watch):
  EMA crossover: fast average above slow average = uptrend, below = downtrend
  RSI:           momentum from 0 to 100; above 70 is "overbought", below 30 "oversold"
  ATR:           average size of a candle, used to place the stop loss

Each coin gets its own pretend EUR 100 and is compared with simply buying and holding it.
"""
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PAIRS = ["XBTEUR", "ETHEUR", "SOLEUR", "XRPEUR", "XDGEUR"]
INTERVAL_MINUTES = 15    # trend-following works on slower candles; 720 x 15 min = 7.5 days

# --- Money and costs -------------------------------------------------------
STARTING_CASH = 100.0
SLIPPAGE = 0.0002        # 0.02% on every buy and sell (an assumption)
# Shorting means borrowing coins to sell now and buy back later. Kraken charges for borrowing:
MARGIN_OPEN_FEE = 0.0002         # 0.02% when a short opens (Kraken's example rate; it changes)
MARGIN_ROLLOVER_FEE = 0.0002     # 0.02% for every 4 hours the short stays open
FEE_LEVELS = [
    ("Kraken spot, lowest tier (taker)", 0.0080),
    ("Kraken derivatives, lowest tier (taker)", 0.0005),
]

# --- Strategy ----------------------------------------------------------------
EMA_FAST = 20            # candles
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_STOP = 2.0           # stop loss sits 2 average candles away from the entry price
LONG_RSI = (50, 70)      # go long only with upward momentum that is not yet overbought
SHORT_RSI = (30, 50)     # go short only with downward momentum that is not yet oversold

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "output"


def load_prices(pair):
    """Combine every saved file for one coin and remove minutes that appear twice."""
    files = sorted(RAW_DIR.glob(f"{pair}_{INTERVAL_MINUTES}m_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    gaps = df["time"].diff().dt.total_seconds().gt(INTERVAL_MINUTES * 60).sum()
    if gaps:
        print(f"Warning: {pair} has {gaps} gap(s). Results across a gap are less reliable.")
    return df


def add_indicators(df):
    """Add the EMA, RSI and ATR columns."""
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    change = df["close"].diff()
    avg_gain = change.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = (-change.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + avg_gain / avg_loss)

    prev_close = df["close"].shift()
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    return df


def close_trade(pos, exit_price, exit_time, reason, cash, fee):
    """Work out the money for one finished trade."""
    entry = pos["entry"]
    fee_open = cash * fee
    hours = (exit_time - pos["time"]).total_seconds() / 3600

    if pos["side"] == "long":
        units = (cash - fee_open) / (entry * (1 + SLIPPAGE))
        proceeds = units * exit_price * (1 - SLIPPAGE)
        fee_close = proceeds * fee
        borrow = 0.0
        cash_after = proceeds - fee_close
        gross = cash * (exit_price / entry - 1)
    else:
        sell_fill = entry * (1 - SLIPPAGE)
        buy_fill = exit_price * (1 + SLIPPAGE)
        units = (cash - fee_open) / sell_fill
        fee_close = units * buy_fill * fee
        borrow = cash * (MARGIN_OPEN_FEE + MARGIN_ROLLOVER_FEE * math.ceil(hours / 4))
        cash_after = cash - fee_open + units * (sell_fill - buy_fill) - fee_close - borrow
        gross = cash * (1 - exit_price / entry)

    fees = fee_open + fee_close
    net = cash_after - cash
    return {
        "side": pos["side"],
        "entry_time": pos["time"],
        "exit_time": exit_time,
        "hours_held": round(hours, 2),
        "entry_price": entry,
        "exit_price": round(exit_price, 6),
        "exit_reason": reason,
        "gross_profit": round(gross, 4),
        "fees": round(fees, 4),
        "borrowing": round(borrow, 4),
        "slippage": round(gross - net - fees - borrow, 4),
        "net_profit": round(net, 4),
        "cash_after": round(cash_after, 4),
    }


def simulate(df, fee):
    """Replay the candles in order and return a table of trades."""
    cash = STARTING_CASH
    pos = None
    trades = []
    for i in range(EMA_SLOW, len(df)):
        row = df.iloc[i]
        uptrend = row["ema_fast"] > row["ema_slow"]
        closed_now = False

        # 1. Manage an open trade: stop loss first (the cautious choice), then trend change.
        if pos is not None:
            exit_price, reason = None, None
            if pos["side"] == "long" and row["low"] <= pos["stop"]:
                exit_price, reason = pos["stop"], "stop loss"
            elif pos["side"] == "short" and row["high"] >= pos["stop"]:
                exit_price, reason = pos["stop"], "stop loss"
            elif pos["side"] == "long" and not uptrend:
                exit_price, reason = row["close"], "trend turned down"
            elif pos["side"] == "short" and uptrend:
                exit_price, reason = row["close"], "trend turned up"
            if exit_price is not None:
                trade = close_trade(pos, exit_price, row["time"], reason, cash, fee)
                trades.append(trade)
                cash = trade["cash_after"]
                pos = None
                closed_now = True

        # 2. Look for a new trade (at most one at a time, not in the same candle as an exit).
        if pos is None and not closed_now:
            if uptrend and LONG_RSI[0] < row["rsi"] < LONG_RSI[1]:
                pos = {"side": "long", "entry": row["close"], "time": row["time"],
                       "stop": row["close"] - ATR_STOP * row["atr"]}
            elif not uptrend and SHORT_RSI[0] < row["rsi"] < SHORT_RSI[1]:
                pos = {"side": "short", "entry": row["close"], "time": row["time"],
                       "stop": row["close"] + ATR_STOP * row["atr"]}

    # 3. Close anything still open at the last price, so every trade is counted.
    if pos is not None:
        last = df.iloc[-1]
        trades.append(close_trade(pos, last["close"], last["time"], "end of data", cash, fee))

    return pd.DataFrame(trades)


def main():
    results = []
    all_trades = []
    for pair in PAIRS:
        df = load_prices(pair)
        if df is None or len(df) <= EMA_SLOW:
            print(f"{pair}: not enough {INTERVAL_MINUTES}-minute data yet, skipped")
            continue
        df = add_indicators(df)
        move = df["close"].iloc[-1] / df["close"].iloc[EMA_SLOW] - 1

        for label, fee in FEE_LEVELS:
            trades = simulate(df, fee)
            end = trades["cash_after"].iloc[-1] if not trades.empty else STARTING_CASH
            hold = STARTING_CASH * (1 + move) * (1 - fee) ** 2
            results.append({
                "pair": pair,
                "fees": label,
                "price_change_%": round(move * 100, 2),
                "trades": len(trades),
                "longs": int((trades["side"] == "long").sum()) if not trades.empty else 0,
                "shorts": int((trades["side"] == "short").sum()) if not trades.empty else 0,
                "won": int((trades["net_profit"] > 0).sum()) if not trades.empty else 0,
                "fees_paid": round(trades["fees"].sum(), 2) if not trades.empty else 0.0,
                "strategy_EUR": round(end, 2),
                "buy_and_hold_EUR": round(hold, 2),
            })
            if not trades.empty:
                all_trades.append(trades.assign(pair=pair, fee_level=label))

    if not results:
        raise SystemExit("No 15-minute data found. Run py src\\fetch_prices.py first.")

    table = pd.DataFrame(results)
    for label, _ in FEE_LEVELS:
        print()
        print(f"--- {label} ---")
        print(table[table["fees"] == label].drop(columns="fees").to_string(index=False))

    if all_trades:
        OUTPUT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"trend_trades_{stamp}.csv"
        pd.concat(all_trades, ignore_index=True).to_csv(path, index=False)
        print()
        print(f"Trade log saved: {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
