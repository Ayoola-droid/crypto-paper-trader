# Crypto Paper Trader: Can Automated Trading Beat the Fees?

**Business question:** Can an automated bot make money trading crypto with €100 once real exchange fees, borrowing costs and slippage are counted? Two approaches are tested: many small trades (scalping) and fewer, longer trades that follow the trend, long or short.

**Data:** 1-minute and 15-minute prices for Bitcoin, Ether, Solana, XRP and Dogecoin in euro, from [Kraken's public API](https://docs.kraken.com/api/docs/rest-api/get-ohlc-data). No account or API key is needed. Fees from [Kraken's spot fee schedule](https://www.kraken.com/features/fee-schedule) (checked 4 November 2026) and [derivatives fees](https://support.kraken.com/articles/360048917612).

**Tools:** Python (pandas, requests)

**Status:** In progress. Both strategies built and tested on one week of prices; collecting more weeks before drawing conclusions.

> No real money is traded. "Paper trading" replays real prices with pretend money.

## Key findings (preliminary: 1–8 November 2026)

1. **Scalping cannot beat the fees.** One buy and one sell costs 1.6% at Kraken's lowest spot tier, but the average one-minute Bitcoin move was 0.01%, and 0 of 719 minutes moved more than the fees. Over 21 trades, €100 became €70.76; fees took €28.52 of the €29.24 lost, and every trade lost money, including the 8 that hit their profit target.
2. **At spot fees, trend-following lost on all five coins** (€100 became €67–€84) and did worse than simply holding each coin, because fees took €22–€30 per coin.
3. **At derivatives fees (0.05%), trend-following beat buy-and-hold on all five coins**, by 1.4 to 5.4 euro per €100. It made money on Bitcoin (€101.59) and Ether (€105.00) even though both fell, and lost less than holding on Solana, XRP and Dogecoin.
4. **The gains came from shorts in a falling week.** Short trades made €16.20 before costs and long trades lost €7.09. This may not repeat in a rising market.
5. **It wins rarely but big, which is typical of trend-following:** 23% of trades were winners, but the average win (€1.65) was nearly three times the average loss (€0.58). Most trades (54 of 93) ended at the stop loss.

### Further detail

- **Before costs, only Bitcoin (+€4.41) and Ether (+€7.73) showed an edge**; Solana, XRP and Dogecoin each lost about €1 before any fees.
- **One week is not enough to trust these results.** Kraken returns only the latest 720 candles, so history is built by running the download regularly.
- **Shorting with real money needs margin or futures trading**, which brings leverage risk and funding or borrowing costs, and may not be available to every retail customer.

## Skills demonstrated

- **Python and pandas:** calling a web API, parsing JSON, combining files, time-series indicators (EMA, RSI, ATR)
- **Simulation:** event-by-event backtests with explicit fee, slippage and borrowing-cost modelling, long and short
- **Analysis:** benchmarking against buy-and-hold, fee-sensitivity scenarios, breaking results down by coin and trade side
- **Data quality:** gap and no-trade checks; raw data never overwritten; duplicate candles removed when files are combined
- **Judgement:** cautious assumptions stated openly, and limits of a one-week sample spelled out

## The output

Both trading scripts print a summary and write a trade log to `output/`, one row per trade:

| Column | Meaning |
|---|---|
| side (trend only) | long (bet on a rise) or short (bet on a fall) |
| entry_time, exit_time | When the trade opened and closed |
| entry_price, exit_price, exit_reason | Prices, and why it closed: take profit, stop loss, time limit or trend change |
| gross_profit | Profit if trading were free |
| fees, borrowing, slippage | The costs (borrowing applies to shorts only) |
| net_profit, cash_after | What was actually kept, and the running balance |

## Method

1. **Download:** `fetch_prices.py` asks Kraken for the latest 720 candles of each coin at 1-minute (12 hours) and 15-minute (7.5 days) intervals. The last candle is still forming, so it is dropped. Each run saves new timestamped files in `data/raw/`.
2. **Combine:** each trading script reads every saved file for a coin, removes candles that appear in more than one file and warns about gaps.
3. **Scalping strategy (`paper_trader.py`, Bitcoin, 1-minute):** buy when the price is 0.05% below its 20-minute average; sell at +0.10% (take profit), −0.10% (stop loss) or after 30 minutes.
4. **Trend strategy (`trend_trader.py`, five coins, 15-minute):**
   - **Trend:** uptrend when the 20-candle exponential moving average (EMA) is above the 50-candle EMA, downtrend when below.
   - **Momentum filter:** go long only when the RSI is between 50 and 70; go short only when it is between 30 and 50.
   - **Exit:** a stop loss 2 × ATR (average candle size) from the entry, or when the trend flips.
   - Each coin starts with its own €100 and is compared with buying and holding it.
5. **Costs:** 0.80% per buy or sell at spot fees (0.05% in the derivatives scenario), 0.02% slippage per side, and for shorts a 0.02% opening fee plus 0.02% for every 4 hours held.

### Judgement calls

- If one candle touches both the stop and the target, the stop is assumed to come first. This is the cautious choice, as candle data does not show which happened first.
- The derivatives scenario reuses the spot borrowing fees for shorts. Real futures charge a funding rate instead, which can be higher or lower.
- Slippage of 0.02% is an assumption, not a measurement. It is likely too low for Dogecoin and XRP, which trade thinly in euro (see the data-quality log).
- The 0.40% "maker" scalping scenario assumes limit orders always fill.

## Data-quality log

| Check | Result (8 Nov 2026 download) |
|---|---|
| Missing candles | 0 in every file |
| 1-minute candles with no trades (of 720) | Bitcoin 10, Ether 111, Solana 130, XRP 331, Dogecoin 560 |
| 15-minute candles with no trades (of 720) | 0 for all coins except Dogecoin (17) |
| Duplicate candles across files | Removed when files are combined |

The euro markets for XRP and Dogecoin are thin: most of their minutes have no trades, so 1-minute results for those coins are unreliable and real slippage would be higher.

## Running it

```powershell
py src\fetch_prices.py
py src\paper_trader.py
py src\trend_trader.py
```

Run `fetch_prices.py` at least once a week, as the 15-minute history only reaches back 7.5 days; runs that overlap are combined automatically.

## Setup

Python 3 on Windows.

```powershell
git clone https://github.com/Ayoola-droid/crypto-paper-trader.git
cd crypto-paper-trader
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Scripts

| Script | What it does |
|---|---|
| `src/fetch_prices.py` | Downloads the latest 1-minute and 15-minute prices for five coins and prints a first inspection |
| `src/paper_trader.py` | Scalping backtest on Bitcoin with €100 of pretend money, at four fee levels |
| `src/trend_trader.py` | Long/short trend-following backtest on five coins at spot and derivatives fees, against buy-and-hold |

## Project structure

```
crypto-paper-trader/
├── data/raw/          one CSV per coin, candle size and download; never edited
├── output/            trade logs (not committed)
├── src/
│   ├── fetch_prices.py
│   ├── paper_trader.py
│   └── trend_trader.py
├── requirements.txt
└── README.md
```

## Commit convention

Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), with one project-specific type: `data` for adding raw data files.
