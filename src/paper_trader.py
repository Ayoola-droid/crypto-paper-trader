"""Lesson 2: paper-trade a simple scalping strategy on saved prices, with fees and slippage.

Paper trading means pretend money: no account, no real orders.
The script reads every price file in data/raw/, replays the minutes in order,
and writes a log of every trade to output/.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PAIR = "XBTEUR"
INTERVAL_MINUTES = 1

# --- Money and costs -------------------------------------------------------
STARTING_CASH = 100.0    # euro
TAKER_FEE = 0.0080       # 0.80% per buy or sell (Kraken Pro lowest tier, checked 4 Nov 2026)
SLIPPAGE = 0.0002        # 0.02%: we assume we buy a little above and sell a little below the price we saw

# --- Strategy: "buy the dip" (mean reversion) -----------------------------
MA_MINUTES = 20          # the moving average: average closing price of the last 20 minutes
ENTRY_DIP = 0.0005       # buy when the price is 0.05% below that average
TAKE_PROFIT = 0.0010     # sell when the price is 0.10% above where we bought
STOP_LOSS = 0.0010       # sell when the price is 0.10% below where we bought
MAX_HOLD_MINUTES = 30    # otherwise sell after 30 minutes

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "output"


def load_prices():
    """Combine every saved price file and remove minutes that appear in more than one file."""
    files = sorted(RAW_DIR.glob(f"{PAIR}_{INTERVAL_MINUTES}m_*.csv"))
    if not files:
        raise SystemExit("No price files in data/raw. Run py src\\fetch_prices.py first.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    rows_before = len(df)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    print(f"Loaded {len(files)} file(s): {rows_before} rows, {len(df)} unique minutes after removing overlaps")
    print(f"Period: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # Data-quality check: if files do not join up, the replay jumps across the missing time.
    gaps = df["time"].diff().dt.total_seconds().gt(INTERVAL_MINUTES * 60).sum()
    if gaps:
        print(f"Warning: {gaps} gap(s) in the minutes. Results across a gap are less reliable.")
    return df


def simulate(df, fee):
    """Replay the prices minute by minute and return a table of trades."""
    df = df.copy()
    df["ma"] = df["close"].rolling(MA_MINUTES).mean()

    cash = STARTING_CASH
    trades = []
    i = MA_MINUTES
    while i < len(df) - 1:
        row = df.iloc[i]
        if row["close"] >= row["ma"] * (1 - ENTRY_DIP):
            i += 1
            continue

        # Buy at the close of this minute.
        entry = row["close"]
        target = entry * (1 + TAKE_PROFIT)
        stop = entry * (1 - STOP_LOSS)
        last = min(i + MAX_HOLD_MINUTES, len(df) - 1)

        # Watch the following minutes until the target, the stop or the time limit is hit.
        # If one minute touches both, we assume the stop came first (the cautious choice).
        for j in range(i + 1, last + 1):
            candle = df.iloc[j]
            if candle["low"] <= stop:
                exit_price, reason = stop, "stop loss"
                break
            if candle["high"] >= target:
                exit_price, reason = target, "take profit"
                break
        else:
            exit_price, reason = df.iloc[last]["close"], "time limit"
            j = last

        # Work out the money, step by step.
        fee_buy = cash * fee
        buy_fill = entry * (1 + SLIPPAGE)
        coins = (cash - fee_buy) / buy_fill
        sell_fill = exit_price * (1 - SLIPPAGE)
        proceeds = coins * sell_fill
        fee_sell = proceeds * fee
        cash_after = proceeds - fee_sell

        gross = cash * (exit_price / entry - 1)      # profit if trading were free
        net = cash_after - cash                      # profit we actually keep
        fees = fee_buy + fee_sell
        trades.append({
            "entry_time": row["time"],
            "exit_time": df.iloc[j]["time"],
            "minutes_held": j - i,
            "entry_price": round(entry, 2),
            "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "cash_before": round(cash, 4),
            "gross_profit": round(gross, 4),
            "fees": round(fees, 4),
            "slippage": round(gross - net - fees, 4),
            "net_profit": round(net, 4),
            "cash_after": round(cash_after, 4),
        })
        cash = cash_after
        i = j + 1    # one trade at a time: look for the next dip after we have sold

    return pd.DataFrame(trades)


def summarise(trades):
    if trades.empty:
        print("No trades: the price never dipped enough below its average.")
        return
    wins = (trades["net_profit"] > 0).sum()
    print(f"Trades: {len(trades)}  (won {wins}, lost {len(trades) - wins})")
    print(f"Exit reasons: {trades['exit_reason'].value_counts().to_dict()}")
    print(f"Profit before costs: EUR {trades['gross_profit'].sum():.2f}")
    print(f"Fees paid:           EUR {trades['fees'].sum():.2f}")
    print(f"Slippage cost:       EUR {trades['slippage'].sum():.2f}")
    print(f"Profit after costs:  EUR {trades['net_profit'].sum():.2f}")
    print(f"EUR {STARTING_CASH:.2f} became EUR {trades['cash_after'].iloc[-1]:.2f}")


def main():
    df = load_prices()

    print()
    print(f"--- Strategy with Kraken's real fee ({TAKER_FEE:.2%} per side) ---")
    trades = simulate(df, TAKER_FEE)
    summarise(trades)

    if not trades.empty:
        OUTPUT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"trades_{stamp}.csv"
        trades.to_csv(path, index=False)
        print(f"Trade log saved: {path.relative_to(PROJECT_DIR)}")

    # Same trades, different fee levels: how much do fees decide the result?
    print()
    print("--- Same strategy, different fee levels ---")
    scenarios = [
        ("Kraken Pro lowest tier, taker", 0.0080),
        ("Kraken Pro lowest tier, maker", 0.0040),
        ("Very large trader", 0.0010),
        ("No fees (slippage only)", 0.0),
    ]
    for label, fee in scenarios:
        result = simulate(df, fee)
        end = result["cash_after"].iloc[-1] if not result.empty else STARTING_CASH
        print(f"{label:<32} fee {fee:.2%}  ->  EUR {end:.2f}")


if __name__ == "__main__":
    main()
