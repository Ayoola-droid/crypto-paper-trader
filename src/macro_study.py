"""Macro study: do big-picture economic factors help the bot make EUR 1 per trade?

Three parts:
  A. Event filter. US interest-rate decisions (FOMC), inflation figures (CPI) and jobs reports move
     crypto within minutes. Do trades that run through one of these releases do worse, and does
     the bot do better if it stops opening trades in the 24 hours before a release?
     (A fixed rule, decided before looking, so the whole period counts as a fair test.)
  B. Macro regimes. Add stock-market, dollar, interest-rate and money-supply measurements to the
     pattern study. Search January-June, test July-November, exactly as before.
  C. Seasonality. With Bitcoin and Ether daily prices back to 2015, check month-of-year,
     day-of-week and halving-cycle patterns in 2015-2020, then see if they held in 2021-2026.

Needs data/macro/events.csv (official dates) and py src\\download_macro.py run first.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import indicator_lab as lab
import pattern_study as ps

MACRO_DIR = lab.PROJECT_DIR / "data" / "macro"
FEES = "derivatives 0.05%"
MAIN_POOLS = ["single indicator", "combination", "combination, trending coins only", "volume surge + breakout"]
HOURS_BEFORE_EVENT = 24      # rule: no new trade in the 24 hours before a release...
HOURS_AFTER_EVENT = 2        # ...or the 2 hours after it
HALVINGS = ["2016-07-09", "2020-05-11", "2024-04-20"]   # Bitcoin's supply cuts
SEASON_SPLIT_YEAR = 2021     # find patterns in 2015-2020, test them in 2021-2026
M2_PUBLICATION_DELAY_DAYS = 55   # monthly M2 is published several weeks after the month ends


# =============================================================================
# Data
# =============================================================================
def load_fred(series):
    path = MACRO_DIR / f"fred_{series}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def load_events():
    events = pd.read_csv(MACRO_DIR / "events.csv")
    new_york = ZoneInfo("America/New_York")
    events["time_utc"] = [pd.Timestamp(f"{d} {t}").tz_localize(new_york).tz_convert("UTC")
                          for d, t in zip(events["date"], events["time_new_york"])]
    events["hour_utc"] = events["time_utc"].dt.floor("h")    # the hourly candle the release falls in
    return events


def event_positions(df, events):
    """Row numbers of the hourly candles that contain a release, with the event type."""
    lookup = pd.Series(np.arange(len(df)), index=df["time"])
    found = events[events["hour_utc"].isin(lookup.index)].copy()
    found["i"] = lookup.loc[found["hour_utc"]].to_numpy()
    return found


# =============================================================================
# A. Event filter
# =============================================================================
def part_a(data, events):
    print("\n" + "=" * 90 + "\nA. US ECONOMIC RELEASES\n" + "=" * 90)

    # A1: how much bigger are price moves in the release hour?
    rows = []
    for pair, d in data.items():
        df = d["df"]
        ret = df["close"].pct_change().abs()
        ev = event_positions(df, events)
        ok = ~d["blocked"]
        for kind, part in ev.groupby("event"):
            idx = part["i"].to_numpy()
            idx = idx[ok[idx]]
            hours = df["time"].dt.hour.to_numpy()
            weekday = df["time"].dt.day_of_week.to_numpy()
            same_hour = np.isin(hours, np.unique(hours[idx])) & (weekday < 5) & ok
            same_hour[idx] = False
            rows.append({"pair": pair, "event": kind, "releases": len(idx),
                         "move_in_release_hour_%": round(100 * ret.iloc[idx].mean(), 2),
                         "same_hour_other_weekdays_%": round(100 * ret[same_hour].mean(), 2)})
    a1 = pd.DataFrame(rows)
    a1["times_bigger"] = (a1["move_in_release_hour_%"] / a1["same_hour_other_weekdays_%"]).round(1)
    print("\nA1. Average price move in the hour of the release vs the same hour on other weekdays")
    print(a1.to_string(index=False))

    # A2: trades whose holding period contains a release
    fees = lab.FEE_LEVELS[FEES]
    ds = ps.build_dataset(data, fees)
    event_rows = {pair: np.sort(event_positions(d["df"], events)["i"].to_numpy()) for pair, d in data.items()}
    contains = np.zeros(len(ds), dtype=bool)
    for pair, part in ds.groupby("pair"):
        ev = event_rows[pair]
        nxt = np.searchsorted(ev, part["i"].to_numpy(), side="right")      # first release after entry
        first_after = np.where(nxt < len(ev), ev[np.minimum(nxt, len(ev) - 1)], np.iinfo(np.int64).max)
        contains[part.index] = first_after <= part["exit_idx"].to_numpy()
    a2 = pd.DataFrame({
        "trades": ["run through a release", "no release"],
        "possible_trades": [int(contains.sum()), int((~contains).sum())],
        "hit_rate_%": [round(100 * ds["win"][contains].mean(), 1), round(100 * ds["win"][~contains].mean(), 1)],
        "avg_profit_EUR": [round(ds["profit"][contains].mean(), 3), round(ds["profit"][~contains].mean(), 3)],
    })
    print(f"\nA2. Every possible hourly trade ({FEES}, EUR 1 target): did a release during the trade matter?")
    print(a2.to_string(index=False))

    # A3: the bot with and without the rule
    results = []
    for label, block in [("normal", False), (f"no new trades {HOURS_BEFORE_EVENT}h before to {HOURS_AFTER_EVENT}h after a release", True)]:
        run_data = {}
        for pair, d in data.items():
            blocked = d["blocked"].copy()
            if block:
                for i in event_rows[pair]:
                    blocked[max(0, i - HOURS_BEFORE_EVENT):i + HOURS_AFTER_EVENT + 1] = True
            run_data[pair] = {**d, "blocked": blocked}
        s = lab.run_lab(run_data, [FEES], MAIN_POOLS)["summary"]
        trades, hits, profit = s["trades"].sum(), s["hit_EUR1"].sum(), s["profit_EUR"].sum()
        results.append({"bot": label, "trades": int(trades), "hit_rate_%": round(100 * hits / trades, 1),
                        "profit_EUR": round(profit, 2), "per_trade_EUR": round(profit / trades, 2)})
    a3 = pd.DataFrame(results)
    print(f"\nA3. The walk-forward bot (four main options, {FEES}) with and without the release rule")
    print(a3.to_string(index=False))
    return a1, a2, a3, ds


# =============================================================================
# B. Macro regimes in the pattern study
# =============================================================================
MACRO_FEATURES = {
    "nasdaq_20d_%": "US tech stocks' change over the past 20 trading days",
    "bitcoin_nasdaq_corr_60d": "How closely Bitcoin and the Nasdaq moved together over ~60 days",
    "dollar_20d_%": "US dollar's change against other currencies over 20 trading days",
    "fed_rate_change_90d": "Change in the US interest rate over 90 days (percentage points)",
    "m2_yoy_%": "US money supply growth over 12 months (as published at the time)",
}


def macro_table():
    """One row per day: each value as it was known at the start of that day (UTC)."""
    nasdaq, dollar, fed, m2, btc = (load_fred(s) for s in ["NASDAQCOM", "DTWEXBGS", "DFF", "M2SL", "CBBTCUSD"])
    if any(x is None for x in [nasdaq, dollar, fed, m2, btc]):
        return None
    t = pd.DataFrame(index=pd.date_range("2014-01-01", pd.Timestamp.today().normalize(), freq="D"))
    t["nasdaq_20d_%"] = (100 * nasdaq.pct_change(20)).reindex(t.index).ffill()
    both = pd.concat([btc.pct_change(), nasdaq.pct_change()], axis=1, join="inner").dropna()
    t["bitcoin_nasdaq_corr_60d"] = both.iloc[:, 0].rolling(40).corr(both.iloc[:, 1]).reindex(t.index).ffill()
    t["dollar_20d_%"] = (100 * dollar.pct_change(20)).reindex(t.index).ffill()
    t["fed_rate_change_90d"] = (fed - fed.shift(90)).reindex(t.index).ffill()
    m2_yoy = 100 * m2.pct_change(12)
    m2_yoy.index = m2_yoy.index + pd.Timedelta(days=M2_PUBLICATION_DELAY_DAYS)
    t["m2_yoy_%"] = m2_yoy.reindex(t.index, method="ffill")
    # A day's closing values are only known the next day: shift everything forward by one day.
    t = t.shift(1)
    t.index = t.index.tz_localize("UTC")
    return t


def part_b(ds, macro):
    print("\n" + "=" * 90 + "\nB. MACRO REGIMES IN THE PATTERN STUDY\n" + "=" * 90)
    ds = ds.copy()
    ds["day"] = ds["time"].dt.floor("D")
    ds = ds.join(macro, on="day").dropna(subset=list(MACRO_FEATURES))
    train, test = ds[ds["time"] < ps.SPLIT], ds[ds["time"] >= ps.SPLIT]
    break_even = lab.break_even_rate(lab.FEE_LEVELS[FEES])

    lifts = []
    for f in MACRO_FEATURES:
        groups = train.groupby(pd.qcut(train[f], 5, duplicates="drop"), observed=True)["win"].agg(["mean", "size"])
        lifts.append({"measurement": f, "best_group": str(groups["mean"].idxmax()),
                      "win_rate_there_%": round(100 * groups["mean"].max(), 1),
                      "worst_group_%": round(100 * groups["mean"].min(), 1),
                      "spread_points": round(100 * (groups["mean"].max() - groups["mean"].min()), 1)})
    lifts = pd.DataFrame(lifts).sort_values("spread_points", ascending=False)
    print(f"\nB1. Win rate by fifth of each macro measurement, January-June only (overall {100 * train['win'].mean():.1f}%)")
    print(lifts.to_string(index=False))

    rows = []
    for name, cols in [("price measurements only", list(ps.FEATURES)),
                       ("price + macro measurements", list(ps.FEATURES) + list(MACRO_FEATURES))]:
        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                               min_samples_leaf=300, random_state=0)
        model.fit(train[cols], train["win"])
        p_tr, p_te = model.predict_proba(train[cols])[:, 1], model.predict_proba(test[cols])[:, 1]
        for rule, m_tr, m_te in [("expected hit rate >= break-even", p_tr >= break_even, p_te >= break_even),
                                 ("top 5% most confident hours", p_tr >= np.quantile(p_tr, 0.95),
                                  p_te >= np.quantile(p_tr, 0.95))]:
            a, b = ps.trade_when(train, m_tr), ps.trade_when(test, m_te)
            rows.append({"model": name, "rule": rule, "search_hit_%": a["hit_%"], "search_per_trade_EUR": a["per_trade_EUR"],
                         "test_trades": b["trades"], "test_hit_%": b["hit_%"],
                         "test_profit_EUR": b["profit_EUR"], "test_per_trade_EUR": b["per_trade_EUR"]})
    b2 = pd.DataFrame(rows)
    print(f"\nB2. Search Jan-Jun, test Jul-Nov: does adding macro help? (break-even {100 * break_even:.1f}%)")
    print(b2.to_string(index=False))
    return lifts, b2


# =============================================================================
# C. Seasonality over many years
# =============================================================================
def part_c():
    print("\n" + "=" * 90 + f"\nC. SEASONALITY: PATTERNS FOUND IN 2015-{SEASON_SPLIT_YEAR - 1}, CHECKED IN {SEASON_SPLIT_YEAR}-2026\n" + "=" * 90)
    out = {}
    for series, name in [("CBBTCUSD", "Bitcoin"), ("CBETHUSD", "Ether")]:
        price = load_fred(series)
        if price is None:
            continue
        price = price[price.index >= "2015-01-01"]
        early = lambda idx: idx.year < SEASON_SPLIT_YEAR

        # Month of year
        monthly = price.resample("ME").last().pct_change().dropna() * 100
        by_month = pd.DataFrame({
            f"2015-{SEASON_SPLIT_YEAR - 1} avg %": monthly[early(monthly.index)].groupby(monthly[early(monthly.index)].index.month).mean(),
            f"{SEASON_SPLIT_YEAR}-2026 avg %": monthly[~early(monthly.index)].groupby(monthly[~early(monthly.index)].index.month).mean(),
        }).round(1)
        by_month.index.name = "month"
        best = by_month.iloc[:, 0].nlargest(6).index
        later = monthly[~early(monthly.index)]
        on = later.index.month.isin(best)
        switches = int(np.abs(np.diff(np.r_[0, on.astype(int), 0])).sum())     # each buy or sell
        hold_all = (1 + later / 100).prod() - 1
        hold_best = (1 + later[on] / 100).prod() * (1 - 0.0080) ** switches - 1   # spot fee on every buy and sell
        corr_m = by_month.iloc[:, 0].corr(by_month.iloc[:, 1])

        # Day of week
        daily = price.pct_change().dropna() * 100
        by_day = pd.DataFrame({
            f"2015-{SEASON_SPLIT_YEAR - 1} avg %": daily[early(daily.index)].groupby(daily[early(daily.index)].index.day_of_week).mean(),
            f"{SEASON_SPLIT_YEAR}-2026 avg %": daily[~early(daily.index)].groupby(daily[~early(daily.index)].index.day_of_week).mean(),
        }).round(3)
        by_day.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        corr_d = by_day.iloc[:, 0].corr(by_day.iloc[:, 1])

        print(f"\n--- {name}: month of year ---")
        print(by_month.T.to_string())
        print(f"Do the two periods agree? Correlation of the 12 monthly averages: {corr_m:+.2f}")
        print(f"Rule 'hold only in the 6 best months of 2015-{SEASON_SPLIT_YEAR - 1}' ({', '.join(str(m) for m in sorted(best))}) "
              f"in {SEASON_SPLIT_YEAR}-2026: {100 * hold_best:+.0f}% after spot fees, vs {100 * hold_all:+.0f}% for holding all along")
        print(f"\n--- {name}: day of week ---")
        print(by_day.T.to_string())
        print(f"Correlation of the 7 weekday averages between the two periods: {corr_d:+.2f}")
        out[name] = {"months": by_month, "days": by_day, "month_corr": corr_m, "day_corr": corr_d,
                     "rule_best_months_%": 100 * hold_best, "hold_all_%": 100 * hold_all}

        if name == "Bitcoin":
            yearly = []
            for k, h in enumerate(HALVINGS):
                start = pd.Timestamp(h)
                for phase in range(4):
                    a, b = start + pd.DateOffset(years=phase), start + pd.DateOffset(years=phase + 1)
                    if b > price.index[-1]:
                        break
                    yearly.append({"halving": h, "year_after_halving": phase + 1,
                                   "return_%": round(100 * (price.asof(b) / price.asof(a) - 1))})
            halving = pd.DataFrame(yearly).pivot(index="year_after_halving", columns="halving", values="return_%")
            print("\n--- Bitcoin: return in each year after a halving (%) ---")
            print(halving.to_string())
            print("Only 2-3 cycles: far too few to tell a real cycle from coincidence.")
            out["halving"] = halving
    return out


def main():
    data = lab.load_all(lab.PAIRS)
    if not data:
        raise SystemExit("Not enough hourly data. Run py src\\download_history.py first.")
    events = load_events()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lab.OUTPUT_DIR.mkdir(exist_ok=True)

    a1, a2, a3, ds = part_a(data, events)
    pd.concat([a1.assign(table="A1"), a2.assign(table="A2"), a3.assign(table="A3")]).to_csv(
        lab.OUTPUT_DIR / f"macro_events_{stamp}.csv", index=False)

    macro = macro_table()
    if macro is None:
        print("\nB and C skipped: run py src\\download_macro.py first.")
        return
    lifts, b2 = part_b(ds, macro)
    pd.concat([lifts.assign(table="B1"), b2.assign(table="B2")]).to_csv(
        lab.OUTPUT_DIR / f"macro_regimes_{stamp}.csv", index=False)

    seasons = part_c()
    rows = []
    for name, r in seasons.items():
        if name != "halving":
            rows.append({"asset": name, "month_pattern_agreement": round(r["month_corr"], 2),
                         "weekday_pattern_agreement": round(r["day_corr"], 2),
                         "best_months_rule_%": round(r["rule_best_months_%"]), "hold_all_%": round(r["hold_all_%"])})
    pd.DataFrame(rows).to_csv(lab.OUTPUT_DIR / f"macro_seasonality_{stamp}.csv", index=False)
    print(f"\nSaved: output/macro_events_{stamp}.csv, macro_regimes_{stamp}.csv, macro_seasonality_{stamp}.csv")


if __name__ == "__main__":
    main()
