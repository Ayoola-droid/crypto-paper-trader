"""Indicator lab: which popular indicators, alone or combined, can make EUR 1 per trade after costs?

How it works, in plain words:
  1. Ten popular indicators each "vote" every hour: +1 (price should rise), -1 (fall) or 0 (no view).
  2. A combination only trades when ALL its indicators agree.
  3. Every trade aims for exactly EUR 1 profit after fees on a EUR 100 stake.
     The stop loss sits the same price distance away on the other side.
  4. Walk-forward testing (the "test running in the background"):
     look back 3 weeks, pick the combination that did best there, trade it for the next week,
     then move forward a week and pick again. Each week's trades use only information
     the bot could have had at the time, so the results are honest (out-of-sample).
  5. The bot sits out a week if no combination made money in the previous 3 weeks,
     and skips a trade if the market is too calm for EUR 1 to be reachable.
"""
import itertools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ["XBTEUR", "ETHEUR", "SOLEUR", "XRPEUR", "XDGEUR"]

# --- Money and costs -------------------------------------------------------
STAKE = 100.0            # euro per trade (not compounded, so every trade is comparable)
TARGET_PROFIT = 1.0      # euro after all costs
SLIPPAGE = 0.0005        # 0.05% per buy or sell: higher than before, as XRP and DOGE trade thinly
MARGIN_OPEN_FEE = 0.0002         # shorts: 0.02% to borrow
MARGIN_ROLLOVER_FEE = 0.0002     # shorts: 0.02% per 4 hours
FEE_LEVELS = {"spot 0.80%": 0.0080, "derivatives 0.05%": 0.0005}

# --- Trade rules -------------------------------------------------------------
MAX_HOLD_HOURS = 48      # close the trade after 2 days if neither target nor stop is hit
WARMUP_HOURS = 80        # indicators need history before they are valid (Ichimoku needs 78)
MAX_FILL_HOURS = 6       # fill up to 6 empty hours in a row; a longer run is a hole in the data

# --- Walk-forward testing ---------------------------------------------------
TRAIN_HOURS = 24 * 21    # look back 3 weeks to choose
TEST_HOURS = 24 * 7      # then trade 1 week
MIN_TRAIN_TRADES = 5     # a combination needs at least 5 trades in the look-back to be trusted

# --- Trend ranking ------------------------------------------------------------
TREND_LOOKBACK_HOURS = 24 * 7   # judge each coin's trend over the past week
TOP_TREND_COINS = 2             # trade only the 2 most strongly trending coins that week
MIN_EFFICIENCY = 0.10           # ...and only if their trend is at least this straight
POOLS = ["single indicator", "combination", "combination, trending coins only"]

# Indicator families. Combining two from the same family repeats the same information
# (multicollinearity), so a combination may use at most one indicator from each family.
FAMILIES = {
    "EMA 20/50": "trend", "MACD": "trend", "Supertrend": "trend",
    "Ichimoku": "trend", "ADX/DMI": "trend",
    "RSI": "momentum",
    "Bollinger": "mean reversion",
    "Donchian (support/resistance)": "support/resistance",
    "OBV": "volume", "VWAP 24h": "volume",
}
MAX_COMBO_SIZE = 3

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
HISTORY_DIR = PROJECT_DIR / "data" / "history"
OUTPUT_DIR = PROJECT_DIR / "output"
COLUMNS = ["time", "open", "high", "low", "close", "volume", "trades"]


# =============================================================================
# 1. Data
# =============================================================================
def load_hourly(pair):
    """Combine archive history and live downloads into one gap-free hourly table."""
    frames = []
    for f in sorted(HISTORY_DIR.glob(f"{pair}_60m_*.csv")):   # archive: no header
        df = pd.read_csv(f, header=None, names=COLUMNS)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        frames.append(df)
    for f in sorted(RAW_DIR.glob(f"{pair}_60m_*.csv")):       # live API: has a header
        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        frames.append(df[COLUMNS])
    if not frames:
        return None, 0, 0, None

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="time", keep="last").sort_values("time").set_index("time")

    # Kraken leaves out hours with no trades. Fill them: price unchanged, volume zero.
    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    missing = ~full.isin(df.index)
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    for col in ["open", "high", "low"]:
        df[col] = df[col].fillna(df["close"])
    df[["volume", "trades"]] = df[["volume", "trades"]].fillna(0)
    df = df.rename_axis("time").reset_index()

    # A long run of missing hours is a hole in our data, not a quiet market
    # (e.g. after the last quarterly archive and before the live API's 30 days begin).
    # No trading inside a hole, or until the indicators have rebuilt after it.
    run_id = np.cumsum(~missing)
    run_len = pd.Series(missing).groupby(run_id).transform("sum").to_numpy()
    hole = missing & (run_len > MAX_FILL_HOURS)
    blocked = hole.copy()
    for end in np.flatnonzero(hole[:-1] & ~hole[1:]):
        blocked[end + 1:end + 1 + WARMUP_HOURS] = True      # indicators rebuilding
    for start in np.flatnonzero(~hole[:-1] & hole[1:]) + 1:
        blocked[max(0, start - MAX_HOLD_HOURS):start] = True  # a trade would run into the hole
    return df, int(missing.sum()), int(hole.sum()), blocked


# =============================================================================
# 2. Indicators: each returns +1, -1 or 0 for every hour
# =============================================================================
def wilder(series, period):
    return series.ewm(alpha=1 / period, adjust=False).mean()


def supertrend_direction(high, low, close, atr, mult=3.0):
    hl2 = (high + low) / 2
    upper, lower = hl2 + mult * atr, hl2 - mult * atr
    final_up, final_low = upper.copy(), lower.copy()
    direction = np.ones(len(close))
    for i in range(1, len(close)):
        final_up[i] = upper[i] if (upper[i] < final_up[i - 1] or close[i - 1] > final_up[i - 1]) else final_up[i - 1]
        final_low[i] = lower[i] if (lower[i] > final_low[i - 1] or close[i - 1] < final_low[i - 1]) else final_low[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_low[i] else 1
        else:
            direction[i] = 1 if close[i] > final_up[i] else -1
    return direction


def compute_signals(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    prev_c = c.shift()
    true_range = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = wilder(true_range, 14)
    s = {}

    # Trend
    s["EMA 20/50"] = np.sign(c.ewm(span=20, adjust=False).mean() - c.ewm(span=50, adjust=False).mean())
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    s["MACD"] = np.sign(macd - macd.ewm(span=9, adjust=False).mean())
    s["Supertrend"] = pd.Series(supertrend_direction(h.to_numpy(), l.to_numpy(), c.to_numpy(), atr.to_numpy()))

    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)          # the cloud is drawn 26 hours ahead,
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)   # so shift it back
    cloud_top, cloud_bottom = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    s["Ichimoku"] = np.where((c > cloud_top) & (tenkan > kijun), 1,
                    np.where((c < cloud_bottom) & (tenkan < kijun), -1, 0))

    up_move, down_move = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))
    plus_di, minus_di = 100 * wilder(plus_dm, 14) / atr, 100 * wilder(minus_dm, 14) / atr
    adx = wilder(100 * (plus_di - minus_di).abs() / (plus_di + minus_di), 14)
    s["ADX/DMI"] = np.where(adx > 20, np.sign(plus_di - minus_di), 0)   # only when a trend is strong

    # Momentum
    change = c.diff()
    rsi = 100 - 100 / (1 + wilder(change.clip(lower=0), 14) / wilder(-change.clip(upper=0), 14))
    s["RSI"] = np.where(rsi > 55, 1, np.where(rsi < 45, -1, 0))

    # Mean reversion: price stretched outside its normal band tends to snap back
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    s["Bollinger"] = np.where(c < mid - 2 * sd, 1, np.where(c > mid + 2 * sd, -1, 0))

    # Support and resistance: breaking the highest high (resistance) or lowest low (support)
    resistance, support = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    s["Donchian (support/resistance)"] = np.where(c > resistance, 1, np.where(c < support, -1, 0))

    # Volume
    obv = (np.sign(change).fillna(0) * v).cumsum()
    s["OBV"] = np.sign(obv - obv.ewm(span=20, adjust=False).mean())
    typical = (h + l + c) / 3
    vwap = (typical * v).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    s["VWAP 24h"] = np.sign(c - vwap)

    out = {}
    for name, values in s.items():
        arr = np.nan_to_num(np.asarray(values, dtype=float)).astype(np.int8)
        arr[:WARMUP_HOURS] = 0
        out[name] = arr
    return out, (atr / c).to_numpy()


def build_combos():
    names = list(FAMILIES)
    singles = [(n,) for n in names]
    multis = [combo for size in range(2, MAX_COMBO_SIZE + 1)
              for combo in itertools.combinations(names, size)
              if len({FAMILIES[n] for n in combo}) == size]      # one per family
    return singles, multis


def combo_signal(signals, combo):
    stack = np.vstack([signals[n] for n in combo])
    return np.where((stack == 1).all(axis=0), 1, np.where((stack == -1).all(axis=0), -1, 0))


# =============================================================================
# 3. Trade outcomes
# =============================================================================
def target_prices(entry, fee):
    """Prices at which a trade makes exactly EUR 1 after costs, and the matching stop."""
    s, p = SLIPPAGE, TARGET_PROFIT / STAKE
    long_tp = entry * (1 + p) * (1 + s) / ((1 - fee) ** 2 * (1 - s))
    k = (1 + s) * (1 + fee) / (1 - s)
    borrow = MARGIN_OPEN_FEE + MARGIN_ROLLOVER_FEE
    short_tp = entry * (1 - (p + fee + borrow) / (1 - fee)) / k
    long_sl = entry - (long_tp - entry)
    short_sl = entry + (entry - short_tp)
    return long_tp, long_sl, short_tp, short_sl


def net_profit(side, entry, exit_price, hours, fee):
    s = SLIPPAGE
    if side == 1:
        units = STAKE * (1 - fee) / (entry * (1 + s))
        return units * exit_price * (1 - s) * (1 - fee) - STAKE
    sell_fill, buy_fill = entry * (1 - s), exit_price * (1 + s)
    units = STAKE * (1 - fee) / sell_fill
    borrow = STAKE * (MARGIN_OPEN_FEE + MARGIN_ROLLOVER_FEE * np.ceil(hours / 4))
    return -STAKE * fee + units * (sell_fill - buy_fill) - units * buy_fill * fee - borrow


def precompute_outcomes(df, fee):
    """For every hour, what happens if we go long or short there? (Same for every combination.)"""
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(close)
    result = {}
    for side in (1, -1):
        exit_idx = np.zeros(n, dtype=int)
        profit = np.zeros(n)
        reason = np.empty(n, dtype=object)
        for i in range(n - 1):
            entry = close[i]
            long_tp, long_sl, short_tp, short_sl = target_prices(entry, fee)
            last = min(i + MAX_HOLD_HOURS, n - 1)
            hi, lo = high[i + 1:last + 1], low[i + 1:last + 1]
            if side == 1:
                tp, sl = long_tp, long_sl
                hit_sl, hit_tp = lo <= sl, hi >= tp
            else:
                tp, sl = short_tp, short_sl
                hit_sl, hit_tp = hi >= sl, lo <= tp
            first_sl = hit_sl.argmax() if hit_sl.any() else len(hi)
            first_tp = hit_tp.argmax() if hit_tp.any() else len(hi)
            if first_sl <= first_tp and first_sl < len(hi):     # same hour: assume stop first
                j, price, why = i + 1 + first_sl, sl, "stop loss"
            elif first_tp < len(hi):
                j, price, why = i + 1 + first_tp, tp, "EUR 1 target"
            else:
                j, price, why = last, close[last], "time limit"
            exit_idx[i], reason[i] = j, why
            profit[i] = net_profit(side, entry, price, j - i, fee)
        result[side] = (exit_idx, profit, reason)
    # The move needed for EUR 1, as a fraction of the price
    needed = target_prices(1.0, fee)[0] - 1.0
    return result, needed


def run(signal, start, end, outcomes, reachable, cursor=0):
    """Trade one signal between two hours, one trade at a time. Returns trades and the next free hour."""
    trades = []
    candidates = np.flatnonzero((signal[start:end] != 0) & reachable[start:end]) + start
    for i in candidates:
        if i < cursor:
            continue
        side = int(signal[i])
        exit_idx, profit, reason = outcomes[side]
        trades.append((i, side, exit_idx[i], profit[i], reason[i]))
        cursor = exit_idx[i] + 1
    return trades, cursor


# =============================================================================
# 4. Walk-forward test
# =============================================================================
def filtered_signal(signals, combo, allowed):
    """A combination's signal, kept only where it matches the allowed direction (if any)."""
    sig = combo_signal(signals, combo)
    return sig if allowed is None else np.where(sig == allowed, sig, 0)


def walk_forward(df, signals, pool, outcomes, reachable, allowed=None):
    pool_signals = {combo: filtered_signal(signals, combo, allowed) for combo in pool}
    windows, trades = [], []
    cursor = 0
    for test_start in range(WARMUP_HOURS + TRAIN_HOURS, len(df) - 1, TEST_HOURS):
        train_start = test_start - TRAIN_HOURS
        test_end = min(test_start + TEST_HOURS, len(df) - 1)

        best, best_profit, best_count = None, 0.0, 0
        for combo, sig in pool_signals.items():
            train_trades, _ = run(sig, train_start, test_start, outcomes, reachable)
            total = sum(t[3] for t in train_trades)
            if len(train_trades) >= MIN_TRAIN_TRADES and total > best_profit:
                best, best_profit, best_count = combo, total, len(train_trades)

        week_trades = []
        if best is not None:     # otherwise: not convinced, sit the week out
            week_trades, cursor = run(pool_signals[best], test_start, test_end,
                                      outcomes, reachable, cursor)
        windows.append({
            "week_start": df["time"].iloc[test_start],
            "chosen": " + ".join(best) if best else "none (sat out)",
            "lookback_trades": best_count,
            "lookback_profit": round(best_profit, 2),
            "week_trades": len(week_trades),
            "week_profit": round(sum(t[3] for t in week_trades), 2),
        })
        for i, side, j, profit, why in week_trades:
            trades.append({
                "entry_time": df["time"].iloc[i], "exit_time": df["time"].iloc[j],
                "side": "long" if side == 1 else "short", "combo": " + ".join(best),
                "entry_price": df["close"].iloc[i], "exit_reason": why,
                "net_profit": round(profit, 4),
            })
    return pd.DataFrame(windows), pd.DataFrame(trades)


def hindsight_best(df, signals, pool, outcomes, reachable, start, allowed=None):
    """The best combination over the whole test period, chosen AFTER seeing it. Not achievable."""
    best_name, best_total = None, -np.inf
    for combo in pool:
        tr, _ = run(filtered_signal(signals, combo, allowed), start, len(df) - 1, outcomes, reachable)
        total = sum(t[3] for t in tr)
        if len(tr) >= MIN_TRAIN_TRADES and total > best_total:
            best_name, best_total = combo, total
    if best_name is None:
        return "none", 0.0
    return " + ".join(best_name), best_total


# =============================================================================
# 5. Trend ranking: prefer coins that are moving strongly up or down
# =============================================================================
def efficiency_ratio(close, hours):
    """How straight the price path was over the last `hours`: 1 = a straight line, 0 = pure zig-zag.

    Net change divided by the total distance travelled (Kaufman's efficiency ratio).
    """
    net = (close - close.shift(hours)).abs()
    path = close.diff().abs().rolling(hours).sum()
    return (net / path).fillna(0)


def trend_masks(data):
    """For every hour: +1 = this coin may only go long, -1 = only short, 0 = not trending enough.

    At the start of each week, rank all coins by efficiency ratio over the past week, using only
    prices up to that moment. The top TOP_TREND_COINS coins may trade, and only in the direction
    of their trend; sideways coins are left alone for that week.
    """
    pairs = list(data)
    n = min(len(data[p]["df"]) for p in pairs)
    er = {}
    for p in pairs:
        ratio = efficiency_ratio(data[p]["df"]["close"], TREND_LOOKBACK_HOURS).to_numpy()
        # A look-back week that touches a hole in the data gives a false reading: treat it as no trend.
        touches_hole = pd.Series(data[p]["blocked"]).rolling(TREND_LOOKBACK_HOURS, min_periods=1).max().to_numpy() > 0
        er[p] = np.where(touches_hole, 0.0, ratio)
    direction = {p: np.sign((data[p]["df"]["close"] - data[p]["df"]["close"].shift(TREND_LOOKBACK_HOURS))
                            .fillna(0).to_numpy()) for p in pairs}
    masks = {p: np.zeros(len(data[p]["df"]), dtype=np.int8) for p in pairs}
    ranking = []
    for week_start in range(WARMUP_HOURS + TREND_LOOKBACK_HOURS, n, TEST_HOURS):
        scores = sorted(pairs, key=lambda p: er[p][week_start], reverse=True)
        top = [p for p in scores[:TOP_TREND_COINS] if er[p][week_start] >= MIN_EFFICIENCY]
        for p in top:
            masks[p][week_start:week_start + TEST_HOURS] = direction[p][week_start]
        ranking.append({"week_start": data[pairs[0]]["df"]["time"].iloc[week_start],
                        **{p: round(er[p][week_start], 3) for p in pairs},
                        "trending": ", ".join(f"{p} {'up' if direction[p][week_start] > 0 else 'down'}"
                                              for p in top) or "none"})
    return masks, pd.DataFrame(ranking)


def indicator_accuracy(df, signals, valid, horizon=24):
    """How often each indicator's vote matched the price direction over the next `horizon` hours."""
    future = np.sign(np.log(df["close"].shift(-horizon) / df["close"]).to_numpy())
    rows = {}
    for name, sig in signals.items():
        m = (sig != 0) & valid & ~np.isnan(future)
        rows[name] = (100 * (future[m] == sig[m]).mean()) if m.any() else np.nan
    return rows


# =============================================================================
# 6. Run everything (used by the command line and by the Streamlit app)
# =============================================================================
def load_all(pairs):
    """Load each coin and compute its indicators once."""
    data = {}
    for pair in pairs:
        df, filled, hole_hours, blocked = load_hourly(pair)
        if df is None or len(df) < WARMUP_HOURS + TRAIN_HOURS + 24:
            continue
        signals, atr_pct = compute_signals(df)
        data[pair] = {"df": df, "signals": signals, "atr_pct": atr_pct, "blocked": blocked,
                      "quiet_filled": filled - hole_hours, "hole_hours": hole_hours}
    return data


def run_lab(data, fee_levels, pools, progress=None):
    """Walk-forward test for every coin, fee level and pool. Returns tables for printing or charts."""
    singles, multis = build_combos()
    pool_defs = {"single indicator": (singles, False), "combination": (multis, False),
                 "combination, trending coins only": (multis, True)}
    masks, ranking = trend_masks(data)
    summary, all_windows, all_trades, accuracy = [], [], [], []
    steps = len(data) * len(fee_levels) * len(pools)
    done = 0

    for pair, d in data.items():
        df, signals = d["df"], d["signals"]
        valid = (df["volume"].to_numpy() > 0) & ~d["blocked"]
        trending = valid & (masks[pair] != 0)
        for name, acc in indicator_accuracy(df, signals, valid).items():
            accuracy.append({"pair": pair, "indicator": name, "family": FAMILIES[name],
                             "all_hours_%": round(acc, 1),
                             "trending_weeks_%": round(indicator_accuracy(df, {name: signals[name]},
                                                                           trending)[name], 1)})
        test_start = WARMUP_HOURS + TRAIN_HOURS
        price_change = df["close"].iloc[-1] / df["close"].iloc[test_start] - 1

        for fee_label in fee_levels:
            fee = FEE_LEVELS[fee_label]
            outcomes, needed = precompute_outcomes(df, fee)
            # A stopped-out trade loses more than EUR 1 (fees are paid either way),
            # so the bot must hit its target more than half the time just to break even.
            stop_loss = -net_profit(1, 1.0, target_prices(1.0, fee)[1], 1, fee)
            break_even = stop_loss / (stop_loss + TARGET_PROFIT)
            # Skip a trade when the market is too calm: a typical 2-day range must exceed the move needed.
            reachable = (d["atr_pct"] * np.sqrt(MAX_HOLD_HOURS) >= needed) & ~d["blocked"]
            for pool_name in pools:
                pool, use_trend = pool_defs[pool_name]
                allowed = masks[pair] if use_trend else None
                windows, trades = walk_forward(df, signals, pool, outcomes, reachable, allowed)
                hs_name, hs_profit = hindsight_best(df, signals, pool, outcomes, reachable,
                                                    test_start, allowed)
                n = len(trades)
                wins = int((trades["net_profit"] > 0.99).sum()) if n else 0
                summary.append({
                    "pair": pair, "fees": fee_label, "pool": pool_name,
                    "move_needed_%": round(needed * 100, 2),
                    "weeks": len(windows),
                    "weeks_traded": int((windows["week_trades"] > 0).sum()),
                    "trades": n,
                    "hit_EUR1": wins,
                    "hit_rate_%": round(100 * wins / n, 1) if n else 0.0,
                    "break_even_%": round(100 * break_even, 1),
                    "profit_EUR": round(trades["net_profit"].sum(), 2) if n else 0.0,
                    "per_trade_EUR": round(trades["net_profit"].mean(), 2) if n else 0.0,
                    "hindsight_best_EUR": round(hs_profit, 2),
                    "hindsight_best": hs_name,
                    "buy_hold_EUR": round(STAKE * (1 + price_change) * (1 - fee) ** 2 - STAKE, 2),
                })
                all_windows.append(windows.assign(pair=pair, fees=fee_label, pool=pool_name))
                if n:
                    all_trades.append(trades.assign(pair=pair, fees=fee_label, pool=pool_name))
                done += 1
                if progress:
                    progress(done / steps, f"{pair}, {fee_label}, {pool_name}")

    return {
        "summary": pd.DataFrame(summary),
        "weeks": pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame(),
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        "accuracy": pd.DataFrame(accuracy),
        "ranking": ranking,
    }


def main():
    singles, multis = build_combos()
    print(f"Testing {len(singles)} single indicators and {len(multis)} combinations "
          f"(at most one indicator per family)")
    data = load_all(PAIRS)
    if not data:
        raise SystemExit("Not enough hourly data. Run py src\\download_history.py first.")
    for pair, d in data.items():
        df = d["df"]
        print(f"{pair}: {len(df):,} hours, {df['time'].iloc[0]:%d %b %Y} to "
              f"{df['time'].iloc[-1]:%d %b %Y}; {d['quiet_filled']:,} quiet hours filled")
        if d["hole_hours"]:
            print(f"  Warning: {d['hole_hours']:,} hours missing in long runs (holes in the data). "
                  f"No trading in them, for {MAX_HOLD_HOURS} hours before or {WARMUP_HOURS} hours after.")

    results = run_lab(data, list(FEE_LEVELS), POOLS)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, table in results.items():
        table.to_csv(OUTPUT_DIR / f"lab_{name}_{stamp}.csv", index=False)

    table = results["summary"]
    show = ["pair", "pool", "weeks_traded", "trades", "hit_rate_%", "break_even_%",
            "profit_EUR", "per_trade_EUR", "hindsight_best_EUR", "buy_hold_EUR"]
    for fee_label in FEE_LEVELS:
        print(f"\n=== Out-of-sample results, {fee_label} fees (EUR 100 per trade) ===")
        print(table[table["fees"] == fee_label][show].to_string(index=False))

    acc = results["accuracy"].pivot(index="indicator", columns="pair", values="all_hours_%")
    print("\n=== How often each indicator called the next 24 hours' direction correctly (%) ===")
    print(acc.to_string())
    trend_acc = results["accuracy"].groupby("indicator")[["all_hours_%", "trending_weeks_%"]].mean().round(1)
    print("\n=== Average accuracy: all hours vs weeks when the coin was among the strongest trends ===")
    print(trend_acc.to_string())
    print(f"\nSaved: output/lab_*_{stamp}.csv (summary, weeks, trades, accuracy, ranking)")


if __name__ == "__main__":
    main()
