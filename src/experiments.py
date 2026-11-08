"""Try the levers that might make EUR 1 per trade more likely, one at a time and then together.

Each experiment re-runs the walk-forward indicator lab with one setting changed and compares the
totals across all five coins. Every number is out-of-sample: chosen on past weeks, traded on the next.

Levers:
  1. Costs:       limit orders (maker fees) instead of market orders
  2. Exits:       a tighter or wider stop compared with the EUR 1 target
  3. Pickiness:   only trust a combination whose look-back hit rate beat break-even by a margin
  5. Fading:      bet that surge breakouts fail (biased: the idea came from this same data)
Lever 4 (funding-rate carry) is a different kind of strategy: see src/funding_carry.py.

Testing many settings and keeping the best is itself a form of overfitting, so the summary also
says how many settings were tried.
"""
from datetime import datetime, timezone

import pandas as pd

import indicator_lab as lab

BASE = {"fees": "derivatives 0.05%", "stop": 1.0, "picky": None}


def run_one(data, name, fees, stop, picky):
    lab.STOP_MULTIPLE = stop
    lab.PICKY_MARGIN = picky
    res = lab.run_lab(data, [fees], lab.POOLS)["summary"]
    rows = []
    for pool, part in res.groupby("pool", sort=False):
        trades = int(part["trades"].sum())
        hits = int(part["hit_EUR1"].sum())
        profit = part["profit_EUR"].sum()
        rows.append({
            "experiment": name, "fees": fees, "stop_x": stop,
            "picky": "off" if picky is None else f"+{picky * 100:g} pts",
            "pool": pool, "trades": trades,
            "hit_%": round(100 * hits / trades, 1) if trades else 0.0,
            "break_even_%": part["break_even_%"].iloc[0],
            "profit_EUR": round(profit, 2),
            "per_trade_EUR": round(profit / trades, 2) if trades else 0.0,
            "coins_in_profit": int((part["profit_EUR"] > 0).sum()),
        })
    return rows


def main():
    data = lab.load_all(lab.PAIRS)
    if not data:
        raise SystemExit("Not enough hourly data. Run py src\\download_history.py first.")

    plan = [("baseline", BASE["fees"], BASE["stop"], BASE["picky"])]
    for fees in lab.FEE_LEVELS:                                   # lever 1
        if fees != BASE["fees"]:
            plan.append(("1 costs", fees, BASE["stop"], BASE["picky"]))
    for stop in [0.5, 1.5, 2.0, 3.0]:                             # lever 2
        plan.append(("2 exits", BASE["fees"], stop, BASE["picky"]))
    for margin in [0.0, 0.05, 0.10]:                              # lever 3
        plan.append(("3 picky", BASE["fees"], BASE["stop"], margin))
    plan.append(("all levers", "derivatives maker 0.02%", 1.0, 0.05))   # combined, chosen up front

    rows = []
    for i, (name, fees, stop, picky) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {name}: {fees}, stop x{stop:g}, picky {picky}")
        rows += run_one(data, name, fees, stop, picky)
    lab.STOP_MULTIPLE, lab.PICKY_MARGIN = 1.0, None

    table = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lab.OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(lab.OUTPUT_DIR / f"experiments_{stamp}.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["experiment", "fees", "stop_x", "picky", "pool", "trades", "hit_%", "break_even_%",
            "profit_EUR", "per_trade_EUR", "coins_in_profit"]
    for pool in lab.POOLS:
        print(f"\n=== {pool} ===")
        print(table[table["pool"] == pool][cols].drop(columns="pool").to_string(index=False))

    winners = table[(table["profit_EUR"] > 0) & (table["trades"] >= 30)]
    print(f"\n{len(plan)} settings x {len(lab.POOLS)} options = {len(table)} results tried.")
    if winners.empty:
        print("None made money over at least 30 trades.")
    else:
        print("Made money over at least 30 trades (check on new data before trusting):")
        print(winners[cols].to_string(index=False))
    print(f"\nSaved: output/experiments_{stamp}.csv")


if __name__ == "__main__":
    main()
