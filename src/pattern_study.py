"""Pattern study: what do the trades that made EUR 1 have in common, and can a bot use it?

Plain-words plan:
  1. Every hour, for every coin, imagine a long trade and a short trade with the usual EUR 1 target
     and matching stop. Record whether each one would have hit the target (a "win").
  2. Describe the market at that moment with 15 measurements (volatility, trend, momentum, volume,
     time of day, how many indicators agreed, what Bitcoin was doing, and so on).
  3. Look for patterns ONLY in January to June. Patterns found by searching are easy to fool
     yourself with, so...
  4. ...test the best "scenario" on July to November, which the search never saw. If it still
     wins at least as often as break-even there, it might be real. If not, it was noise.

Note: neighbouring hours share most of the same future, so the samples overlap. That makes patterns
look more certain than they are, which is another reason the test period matters.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

import indicator_lab as lab

SPLIT = pd.Timestamp("2026-07-01", tz="UTC")     # search before, test from here
FEES = "derivatives 0.05%"
FEATURES = {
    "volatility_%": "Average hourly range (ATR) as % of price",
    "trend_straightness": "Efficiency ratio over the past week (0 zig-zag, 1 straight line)",
    "with_trend_7d_%": "Past week's price change in the trade's direction",
    "with_trend_24h_%": "Past day's price change in the trade's direction",
    "rsi_in_direction": "RSI, flipped for shorts (above 50 = momentum in the trade's direction)",
    "stretch_from_ema50_%": "Distance from the 50-hour average, in the trade's direction",
    "volume_vs_week": "Last 24h volume compared with an average day of the past week",
    "band_width_rank": "Bollinger band width vs the past 30 days (0 = calmest, 1 = widest)",
    "indicators_agreeing": "How many of the 10 indicators pointed in the trade's direction",
    "indicators_against": "How many pointed the other way",
    "bitcoin_24h_in_direction_%": "Bitcoin's past-day change in the trade's direction",
    "hour_utc": "Hour of the day (UTC)",
    "weekday": "Day of the week (0 = Monday)",
    "is_short": "1 for a short trade, 0 for a long",
    "coin": "Which coin (0 Bitcoin, 1 Ether, 2 Solana, 3 XRP, 4 Dogecoin)",
}


def build_dataset(data, fees):
    btc = data["XBTEUR"]["df"]["close"]
    btc_24h = (btc / btc.shift(24) - 1).to_numpy()
    rows = []
    for coin_no, (pair, d) in enumerate(data.items()):
        df, sig = d["df"], d["signals"]
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        ema50 = c.ewm(span=50, adjust=False).mean()
        change = c.diff()
        rsi = 100 - 100 / (1 + lab.wilder(change.clip(lower=0), 14) / lab.wilder(-change.clip(upper=0), 14))
        bw = 4 * c.rolling(20).std() / c.rolling(20).mean()
        base = pd.DataFrame({
            "time": df["time"], "pair": pair, "coin": coin_no,
            "volatility_%": 100 * d["atr_pct"],
            "trend_straightness": lab.efficiency_ratio(c, 24 * 7),
            "ret_7d": 100 * (c / c.shift(24 * 7) - 1),
            "ret_24h": 100 * (c / c.shift(24) - 1),
            "rsi": rsi,
            "stretch": 100 * (c / ema50 - 1),
            "volume_vs_week": v.rolling(24).sum() / (v.rolling(24 * 7).sum() / 7).replace(0, np.nan),
            "band_width_rank": bw.rolling(24 * 30).rank(pct=True),
            "btc_24h": 100 * btc_24h[:len(df)],
            "hour_utc": df["time"].dt.hour, "weekday": df["time"].dt.weekday,
            "up_votes": np.sum([s == 1 for s in sig.values()], axis=0),
            "down_votes": np.sum([s == -1 for s in sig.values()], axis=0),
        })
        outcomes, _ = lab.precompute_outcomes(df, fees)
        valid = (~d["blocked"]) & (np.arange(len(df)) >= lab.WARMUP_HOURS + 24 * 30) & (np.arange(len(df)) < len(df) - lab.MAX_HOLD_HOURS)
        for side in (1, -1):
            exit_idx, profit, reason, filled = outcomes[side]
            part = base.copy()
            part["side"], part["is_short"] = side, int(side == -1)
            part["with_trend_7d_%"] = side * part["ret_7d"]
            part["with_trend_24h_%"] = side * part["ret_24h"]
            part["rsi_in_direction"] = 50 + side * (part["rsi"] - 50)
            part["stretch_from_ema50_%"] = side * part["stretch"]
            part["bitcoin_24h_in_direction_%"] = side * part["btc_24h"]
            part["indicators_agreeing"] = part["up_votes"] if side == 1 else part["down_votes"]
            part["indicators_against"] = part["down_votes"] if side == 1 else part["up_votes"]
            part["profit"], part["exit_idx"] = profit, exit_idx
            part["win"] = (reason == lab.TARGET_REASON).astype(int)   # reached the target
            part["i"] = np.arange(len(df))
            rows.append(part[valid & filled])
    return pd.concat(rows, ignore_index=True).dropna(subset=list(FEATURES))


def lift_table(train, overall):
    """Win rate in each fifth (quintile) of every measurement, January to June only."""
    out = []
    for f in FEATURES:
        if train[f].nunique() <= 10:
            groups = train.groupby(f)["win"]
        else:
            groups = train.groupby(pd.qcut(train[f], 5, duplicates="drop"), observed=True)["win"]
        stats = groups.agg(["mean", "size"])
        best = stats["mean"].idxmax()
        out.append({"measurement": f, "best_group": str(best),
                    "win_rate_there_%": round(100 * stats["mean"].max(), 1),
                    "win_rate_worst_group_%": round(100 * stats["mean"].min(), 1),
                    "spread_points": round(100 * (stats["mean"].max() - stats["mean"].min()), 1),
                    "samples_in_best_group": int(stats.loc[best, "size"])})
    t = pd.DataFrame(out).sort_values("spread_points", ascending=False)
    t.insert(3, "overall_%", round(100 * overall, 1))
    return t


def trade_when(frame, mask):
    """Trade only where the rule fires, one trade at a time per coin; return wins, trades, profit."""
    trades = []
    for pair, part in frame[mask].sort_values("i").groupby("pair"):
        cursor = -1
        for row in part.itertuples():
            if row.i <= cursor:
                continue
            trades.append((row.win, row.profit))
            cursor = row.exit_idx
    if not trades:
        return {"trades": 0, "hit_%": 0.0, "profit_EUR": 0.0, "per_trade_EUR": 0.0}
    wins, profits = zip(*trades)
    return {"trades": len(trades), "hit_%": round(100 * np.mean(wins), 1),
            "profit_EUR": round(sum(profits), 2), "per_trade_EUR": round(np.mean(profits), 2)}


def main():
    data = lab.load_all(lab.PAIRS)
    fees = lab.FEE_LEVELS[FEES]
    break_even = lab.break_even_rate(fees)
    ds = build_dataset(data, fees)
    train, test = ds[ds["time"] < SPLIT], ds[ds["time"] >= SPLIT]
    print(f"{len(ds):,} possible trades ({len(train):,} Jan-Jun to search, {len(test):,} Jul-Nov to test)")
    print(f"Win = reached the EUR 1 target. Break-even win rate at {FEES}: {100 * break_even:.1f}%")
    print(f"Overall win rate: search period {100 * train['win'].mean():.1f}%, test period {100 * test['win'].mean():.1f}%")

    print("\n=== 1. Which measurements separate wins from losses? (search period only) ===")
    lifts = lift_table(train, train["win"].mean())
    print(lifts.to_string(index=False))

    print("\n=== 2. Simple rules found by a decision tree (search period only) ===")
    X_cols = list(FEATURES)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, random_state=0)
    tree.fit(train[X_cols], train["win"])
    print(export_text(tree, feature_names=X_cols, show_weights=False, decimals=2))
    leaf_train = tree.apply(train[X_cols])
    leaf_rates = train.groupby(leaf_train)["win"].mean()
    good_leaves = leaf_rates[leaf_rates >= break_even].index.tolist()
    print(f"Groups that beat break-even in the search period: {len(good_leaves)} of {len(leaf_rates)}")

    print("\n=== 3. A model that weighs every measurement at once (search period only) ===")
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           min_samples_leaf=300, random_state=0)
    model.fit(train[X_cols], train["win"])
    p_train, p_test = model.predict_proba(train[X_cols])[:, 1], model.predict_proba(test[X_cols])[:, 1]
    print(f"Hours where the model expects a win rate above break-even: "
          f"{(p_train >= break_even).mean():.1%} of search hours, {(p_test >= break_even).mean():.1%} of test hours")

    print("\n=== 4. The scenarios, traded one at a time: search period vs the unseen test period ===")
    scenarios = {
        "every hour (no filter)": (np.ones(len(train), bool), np.ones(len(test), bool)),
        "best single measurement group": None,
        "decision-tree groups above break-even": (np.isin(leaf_train, good_leaves),
                                                  np.isin(tree.apply(test[X_cols]), good_leaves)),
        "model: expected win rate >= break-even": (p_train >= break_even, p_test >= break_even),
        "model: top 5% most confident hours": (p_train >= np.quantile(p_train, 0.95),
                                               p_test >= np.quantile(p_train, 0.95)),
    }
    top = lifts.iloc[0]
    f = top["measurement"]
    if train[f].nunique() <= 10:
        val = type(train[f].iloc[0])(float(top["best_group"]))
        scenarios["best single measurement group"] = (train[f] == val, test[f] == val)
    else:
        edges = pd.qcut(train[f], 5, duplicates="drop", retbins=True)[1]
        bins_tr = pd.cut(train[f], edges, include_lowest=True).astype(str)
        bins_te = pd.cut(test[f], edges, include_lowest=True).astype(str)
        scenarios["best single measurement group"] = (bins_tr == top["best_group"], bins_te == top["best_group"])
    rows = []
    for name, (m_tr, m_te) in scenarios.items():
        a, b = trade_when(train, np.asarray(m_tr)), trade_when(test, np.asarray(m_te))
        rows.append({"scenario": name, "search_trades": a["trades"], "search_hit_%": a["hit_%"],
                     "search_per_trade_EUR": a["per_trade_EUR"], "test_trades": b["trades"],
                     "test_hit_%": b["hit_%"], "test_profit_EUR": b["profit_EUR"],
                     "test_per_trade_EUR": b["per_trade_EUR"]})
    result = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(result.to_string(index=False))
    print(f"(break-even hit rate: {100 * break_even:.1f}%)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lab.OUTPUT_DIR.mkdir(exist_ok=True)
    lifts.to_csv(lab.OUTPUT_DIR / f"patterns_lifts_{stamp}.csv", index=False)
    result.to_csv(lab.OUTPUT_DIR / f"patterns_scenarios_{stamp}.csv", index=False)
    print(f"\nSaved: output/patterns_lifts_{stamp}.csv, output/patterns_scenarios_{stamp}.csv")


if __name__ == "__main__":
    main()
