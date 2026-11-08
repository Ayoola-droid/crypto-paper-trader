# Crypto Paper Trader: Can Automated Trading Beat the Fees?

**Business question:** Can an automated bot make money trading crypto with €100 once real exchange fees, borrowing costs and slippage are counted? And can popular technical indicators, alone or combined, predict price moves well enough to make €1 on every trade?

**Data:** 10 months of hourly prices (January to November 2026) plus 1-minute and 15-minute prices for Bitcoin, Ether, Solana, XRP and Dogecoin in euro, from [Kraken's quarterly archives](https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data) and [public API](https://docs.kraken.com/api/docs/rest-api/get-ohlc-data). Fees from [Kraken's spot fee schedule](https://www.kraken.com/features/fee-schedule) (checked 4 November 2026) and [derivatives fees](https://support.kraken.com/articles/360048917612).

**Tools:** Python (pandas, NumPy, requests), Streamlit, Plotly

**Status:** In progress. Three strategies tested, with a Streamlit front end for the indicator lab.

> No real money is traded. "Paper trading" replays real prices with pretend money.

## Key findings

1. **No indicator, alone or combined, could make €1 per trade.** Across 10 popular indicators and 86 combinations, tested week by week on prices the bot had not yet seen, every coin lost money at both fee levels. At derivatives fees, combinations lost €319 over 948 trades (€100 stake each); single indicators lost €245 over 869 trades.
2. **The indicators are close to coin flips.** When an indicator pointed up or down, the price moved that way over the next 24 hours only 44–51% of the time for trend indicators (EMA, MACD, Supertrend, Ichimoku, ADX). The best was Bollinger Bands at 54% on XRP and Dogecoin.
3. **A €1 target needs a high hit rate, because losing trades pay fees too.** At derivatives fees the bot must hit its target 58% of the time to break even (81% at spot fees); it managed 29–39%.
4. **Combining indicators did not help.** Combinations lost about as much per trade as single indicators (€0.30–€0.37 against €0.25–€0.32), and traded more often. Even the best combination picked in hindsight lost money on 4 of 5 coins.
5. **Trading only strongly trending coins cut the losses but did not make money.** Ranking coins each week by how straight their price path was (efficiency ratio) and trading only the top 2, in their trend's direction, lost €10.38 over 119 trades at derivatives fees (−€0.09 a trade, against −€0.34 for all coins). The hit rate rose from 36% to 47%, still short of the 58% needed, and the bot traded in only 10 of 210 coin-weeks, so the sample is small.
6. **Scalping cannot beat spot fees.** One buy and one sell costs 1.6%, but the average one-minute Bitcoin move was 0.01%. Over 21 scalping trades, €100 became €70.76, with fees taking €28.52 of the €29.24 lost.

### Further detail

- **Support/resistance breakouts worked best on trending coins:** the Donchian channel called the next 24 hours correctly 53.5% of the time in trending weeks, against 48.6% overall. Every other indicator stayed below 52%.

- **At spot fees the lab's bot mostly refused to trade**, as designed: it traded in only 10 of 210 coin-weeks, because no combination had made money in the previous 3 weeks. The trades it did make still lost.
- **The 15-minute trend strategy beat buy-and-hold at derivatives fees in one falling week (1–8 November)**, mainly through short trades. The 10-month hourly test above suggests this does not hold up over time.
- **Holding was better than trading in 2026:** all five coins fell, but holding lost €9–€39 per €100, less than the indicator strategies lost on most coins.

## Skills demonstrated

- **Python and pandas:** web APIs, large archive downloads, combining files, time-series indicators (EMA, MACD, RSI, ATR, ADX, Supertrend, Ichimoku, Bollinger, Donchian, OBV, VWAP)
- **Backtesting done properly:** walk-forward (out-of-sample) testing, hindsight benchmarks, buy-and-hold benchmarks, fee and borrowing-cost modelling for long and short trades
- **Statistics:** directional accuracy of each indicator, break-even hit rates
- **Data quality:** detecting a 206-hour hole between two data sources and excluding it rather than filling it with fake prices; raw data never overwritten
- **Judgement:** avoiding multicollinearity and hindsight bias, and reporting a negative result clearly

## The app

`streamlit run app.py` opens the indicator lab in the browser. Choose coins, fee level, what the bot may choose from, the profit target, the look-back and the number of trending coins in the sidebar; the results update on their own. A **Download latest prices** button runs `fetch_prices.py`.

| Tab | Shows |
|---|---|
| Results | Profit by coin for each option against just holding the coin, and hit rate against the break-even line |
| Week by week | Running total of profit, and which combination the bot picked each week |
| Trades on the chart | Every trade marked on the price chart: ▲ long, ▼ short, blue for a hit, red for a miss |
| Indicator accuracy | How often each indicator called the next 24 hours correctly, coin by coin |
| Trend ranking | Each coin's trend strength every week and which coins were traded |

**Design direction: "Night ledger"**, a research notebook for a sceptical trader. Blue-black page, warm paper-coloured text, one amber accent, Fraunces serif headlines, Source Sans 3 body text and JetBrains Mono for data labels. Fonts are bundled in `static/fonts` (SIL Open Font Licence) so the app works offline. Colours live in `.streamlit/config.toml` and the `THEME` dictionary in `app.py`; the three chart colours were checked for colour-blind separation.

## The indicator lab (`src/indicator_lab.py`)

1. **Votes:** each of 10 indicators votes every hour: +1 (up), −1 (down) or 0 (no view).
2. **Families:** indicators are grouped into trend, momentum, mean reversion, support/resistance and volume. A combination uses at most one per family, because two from the same family repeat the same information (multicollinearity). A combination trades only when all its indicators agree.
3. **€1 target:** every trade risks €100 and closes at the price that gives exactly €1 after fees and slippage, at a stop the same price distance away, or after 48 hours.
4. **Walk-forward test:** each week the bot looks back 3 weeks, picks the combination that made the most there (at least 5 trades), and trades it for the next week. If nothing made money, it sits the week out.
5. **Reachability check:** a trade is skipped if the market is too calm for the €1 move to be likely within 2 days.
6. **Trending coins only (optional pool):** at the start of each week, coins are ranked by efficiency ratio over the previous week (net price change ÷ total distance travelled; 1 = a straight line, 0 = zig-zag). Only the top 2, with a ratio of at least 0.10, may trade that week, and only in the direction of their trend. A look-back week that touches a hole in the data counts as no trend.

| Output | Contents |
|---|---|
| `output/lab_summary_*.csv` | One row per coin, fee level and pool: trades, hit rate, break-even hit rate, profit, hindsight best, buy-and-hold |
| `output/lab_weeks_*.csv` | Which combination was chosen each week, its look-back profit and its profit that week |
| `output/lab_trades_*.csv` | Every out-of-sample trade |
| `output/lab_accuracy_*.csv` | How often each indicator called the next 24 hours correctly, overall and in trending weeks |
| `output/lab_ranking_*.csv` | Each coin's weekly trend strength and which coins were traded |

### Judgement calls

- If one candle touches both the stop and the target, the stop is assumed to come first (the cautious choice).
- Slippage is set at 0.05% per side for hourly trades, as XRP and Dogecoin trade thinly in euro.
- Shorts pay Kraken's example margin fees (0.02% to open, 0.02% per 4 hours). Real futures charge a funding rate instead.
- Up to 6 empty hours in a row are treated as quiet (price unchanged). Longer runs are holes in the data; there is no trading inside them, for 48 hours before (a trade could run into the hole) or for 80 hours after (indicators rebuilding).

## Other strategies

- **`paper_trader.py`:** scalping on 1-minute Bitcoin prices. Buy 0.05% below the 20-minute average; sell at +0.10%, −0.10% or after 30 minutes.
- **`trend_trader.py`:** long/short trend-following on 15-minute prices. Uptrend when the 20-candle EMA is above the 50-candle EMA; RSI filter; stop 2 × ATR away; compared with buy-and-hold.

## Data-quality log

| Check | Result |
|---|---|
| Hourly archive rows | Complete: one row per hour for every coin in Q1 (2,160), Q2 (2,184) and Q3 (2,208) 2026 |
| Hole between sources | 1–9 October 2026 (206 hours): after the last published archive (Q3) and before the live API's 30 days. Excluded from trading |
| 1-minute candles with no trades (of 720, 8 Nov) | Bitcoin 9, Ether 107, Solana 124, XRP 324, Dogecoin 557: thin euro markets, so 1-minute results for XRP and Dogecoin are unreliable |
| Duplicate candles across files | Removed when files are combined |

## Running it

```powershell
py src\download_history.py   # once: hourly archives for Jan-Sep 2026 (about 1.6 GB downloaded, then deleted)
py src\fetch_prices.py       # latest 1-minute, 15-minute and hourly prices
py src\indicator_lab.py      # walk-forward indicator test
py src\paper_trader.py       # scalping test
py src\trend_trader.py       # 15-minute trend test
streamlit run app.py         # the front end
```

Run `fetch_prices.py` at least weekly; overlapping downloads are combined automatically.

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
| `app.py` | Streamlit front end for the indicator lab |
| `src/download_history.py` | Downloads Kraken's quarterly archives and keeps only hourly prices for the five coins |
| `src/fetch_prices.py` | Downloads the latest 1-minute, 15-minute and hourly prices and prints a first inspection |
| `src/indicator_lab.py` | Walk-forward test of 10 indicators and 86 combinations with a €1 profit target, with optional trending-coins filter |
| `src/paper_trader.py` | Scalping backtest on Bitcoin, at four fee levels |
| `src/trend_trader.py` | Long/short 15-minute trend backtest on five coins, against buy-and-hold |

## Project structure

```
crypto-paper-trader/
├── data/
│   ├── history/       hourly prices from Kraken's quarterly archives
│   ├── raw/           one CSV per coin, candle size and API download; never edited
│   └── archive/       temporary ZIP downloads (not committed)
├── output/            results and trade logs (not committed)
├── static/fonts/      bundled fonts for the app, with licences
├── .streamlit/        app theme (colours and fonts)
├── src/
├── app.py
├── requirements.txt
└── README.md
```

## Commit convention

Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), with one project-specific type: `data` for adding raw data files.
