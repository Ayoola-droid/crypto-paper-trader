"""Two tools borrowed (as ideas, rewritten for crypto) from the day-trader-skill pack (MIT):

Time Machine: before taking a trade, replay the next 48 hours thousands of times, each hour drawn
  from the coin's own recent hours at a similar time of day, and count how often the EUR 1 target
  or the stop is reached first. Also checks the same trade at every past hour of the period.
Risk manager: apply simple limits to a list of backtest trades (most trades per day, a daily
  loss limit) and show what they would have changed.
"""
import numpy as np
import pandas as pd

import indicator_lab as lab


def time_machine(df, side, fees, replays=10_000, lookback_days=60, hours=None, seed=19):
    """Simulate a trade opened at the last close. Returns a dict of odds and the paths' end prices."""
    hours = hours or lab.MAX_HOLD_HOURS
    rng = np.random.default_rng(seed)
    recent = df.iloc[-24 * lookback_days:].reset_index(drop=True)
    prev = recent["close"].shift().to_numpy()
    ratios = np.stack([recent["high"] / prev, recent["low"] / prev, recent["close"] / prev], axis=1)[1:]
    hour_of_day = recent["time"].dt.hour.to_numpy()[1:]
    ok = recent["volume"].to_numpy()[1:] > 0               # skip filled, empty hours
    pools = {h: np.flatnonzero(ok & (np.abs(((hour_of_day - h) + 12) % 24 - 12) <= 1)) for h in range(24)}

    entry = float(df["close"].iloc[-1])
    long_tp, long_sl, short_tp, short_sl = lab.target_prices(entry, fees)
    target, stop = (long_tp, long_sl) if side == 1 else (short_tp, short_sl)
    price = np.full(replays, entry)
    outcome = np.zeros(replays, dtype=np.int8)               # 0 open, 1 target, -1 stop
    start_hour = (int(df["time"].iloc[-1].hour) + 1) % 24
    for k in range(hours):
        pool = pools[(start_hour + k) % 24]
        pick = ratios[rng.choice(pool, size=replays)]
        hi, lo, cl = price * pick[:, 0], price * pick[:, 1], price * pick[:, 2]
        open_ = outcome == 0
        hit_stop = open_ & ((lo <= stop) if side == 1 else (hi >= stop))
        hit_target = open_ & ~hit_stop & ((hi >= target) if side == 1 else (lo <= target))
        outcome[hit_stop], outcome[hit_target] = -1, 1
        price = np.where(open_, cl, price)
    t, s = (outcome == 1).mean(), (outcome == -1).mean()
    decided = t + s
    return {"entry": entry, "target": target, "stop": stop, "target_first_%": 100 * t,
            "stop_first_%": 100 * s, "neither_%": 100 * (1 - decided),
            "target_share_of_decided_%": 100 * t / decided if decided else float("nan"),
            "break_even_%": 100 * lab.break_even_rate(fees), "end_prices": price,
            "pool_hours": int(ok.sum()), "replays": replays}


def history_check(df, side, fees, lookback_days=60):
    """The same trade, opened at every past hour of the look-back period: how did it go?"""
    outcomes, _ = lab.precompute_outcomes(df, fees)
    exit_idx, profit, reason, filled = outcomes[side]
    start = max(0, len(df) - 24 * lookback_days)
    idx = np.arange(start, len(df) - lab.MAX_HOLD_HOURS)
    idx = idx[filled[idx] & (df["volume"].to_numpy()[idx] > 0)]
    reasons = pd.Series(reason[idx])
    return {"trades": len(idx), "target_%": 100 * (reasons == lab.TARGET_REASON).mean(),
            "stop_%": 100 * (reasons == "stop loss").mean(), "avg_profit": float(profit[idx].mean())}


def verdict(share, break_even):
    """Plain-words reading of the odds, in the spirit of the pack's verdict line."""
    gap = share - break_even
    if np.isnan(gap):
        return "Too few replays reached the target or the stop to judge."
    if abs(gap) < 5:
        return "Close to break-even: no real edge either way."
    lean = "the target" if gap > 0 else "the stop"
    strength = "Leans toward" if abs(gap) < 15 else "Strongly toward"
    return f"{strength} {lean} ({gap:+.0f} points against the {break_even:.0f}% needed)."


def apply_risk_limits(trades, max_trades_per_day=None, daily_loss_limit=None):
    """Replay trades in time order across all coins; skip any that would break a limit."""
    if trades.empty:
        return trades.assign(taken=pd.Series(dtype=bool), skipped_because=pd.Series(dtype=object))
    t = trades.sort_values("entry_time").reset_index(drop=True).copy()
    t["day"] = pd.to_datetime(t["entry_time"]).dt.floor("D")
    taken, why = [], []
    kept = []          # (day, exit_time, profit) of trades already taken
    for row in t.itertuples():
        today = [k for k in kept if k[0] == row.day]
        realised = sum(p for d, ex, p in today if ex <= row.entry_time)
        if max_trades_per_day and len(today) >= max_trades_per_day:
            taken.append(False); why.append("trades-per-day limit")
        elif daily_loss_limit and realised <= -daily_loss_limit:
            taken.append(False); why.append("daily loss limit")
        else:
            taken.append(True); why.append("")
            kept.append((row.day, row.exit_time, row.net_profit))
    t["taken"], t["skipped_because"] = taken, why
    return t


def risk_summary(t):
    def stats(frame):
        if frame.empty:
            return {"trades": 0, "profit_EUR": 0.0, "worst_day_EUR": 0.0, "biggest_drop_EUR": 0.0}
        daily = frame.groupby("day")["net_profit"].sum()
        curve = frame.sort_values("exit_time")["net_profit"].cumsum()
        return {"trades": len(frame), "profit_EUR": round(frame["net_profit"].sum(), 2),
                "worst_day_EUR": round(daily.min(), 2),
                "biggest_drop_EUR": round((curve.cummax().clip(lower=0) - curve).max(), 2)}
    return pd.DataFrame([{"bot": "no limits", **stats(t)}, {"bot": "with your limits", **stats(t[t["taken"]])}])
