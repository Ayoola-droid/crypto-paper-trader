"""Streamlit front end for the indicator lab.

Run from the project folder:  streamlit run app.py
Pick coins, fees and settings in the sidebar, press "Run test", and read the results in the tabs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
import fetch_prices  # noqa: E402
import indicator_lab as lab  # noqa: E402

# --- Theme: one place for every colour (matches .streamlit/config.toml) ------
THEME = {
    "page": "#0E1217", "surface": "#151A21", "ink": "#ECE6DA", "ink_2": "#B9B2A4",
    "muted": "#8A8478", "grid": "#252C35", "axis": "#3A4350", "accent": "#E9A23B",
    "gain": "#3987E5", "loss": "#E66767", "neutral": "#383835",
}
# Each pool keeps its colour whatever is selected (colour follows the entity, not its position).
# Checked with a colour-blind simulator: the four pass as neighbouring bars. Lines and dots also get
# their own dash style and marker shape, so no two options are told apart by colour alone.
POOL_COLOURS = {"single indicator": "#3987E5", "combination": "#D95926",
                "combination, trending coins only": "#199E70", "volume surge + breakout": "#C98500",
                "fade surge breakout": "#9085E9"}
POOL_DASH = {"single indicator": "solid", "combination": "dash",
             "combination, trending coins only": "dot", "volume surge + breakout": "dashdot",
             "fade surge breakout": "longdash"}
POOL_SYMBOL = {"single indicator": "circle", "combination": "square",
               "combination, trending coins only": "diamond", "volume surge + breakout": "triangle-up",
               "fade surge breakout": "x"}
COIN_NAMES = {"XBTEUR": "Bitcoin", "ETHEUR": "Ether", "SOLEUR": "Solana",
              "XRPEUR": "XRP", "XDGEUR": "Dogecoin"}

st.set_page_config(page_title="Indicator Lab · Crypto Paper Trader", page_icon="◐", layout="wide")


# =============================================================================
# Styling the theme can't do: layered background, header, stat tiles, one entrance
# =============================================================================
st.markdown(f"""
<style>
.stApp {{
  background:
    radial-gradient(1100px 500px at 85% -10%, rgba(233,162,59,0.08), transparent 60%),
    radial-gradient(900px 600px at -10% 110%, rgba(57,135,229,0.07), transparent 60%),
    repeating-linear-gradient(0deg, rgba(236,230,218,0.025) 0 1px, transparent 1px 28px),
    {THEME['page']};
}}
.eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; letter-spacing: 0.16em;
           color: {THEME['accent']}; text-transform: uppercase; margin-bottom: 0.4rem; }}
.headline {{ font-family: 'Fraunces', serif; font-weight: 800; font-size: 3.1rem; line-height: 1.05;
            color: {THEME['ink']}; margin: 0 0 0.6rem 0; }}
.headline em {{ font-weight: 300; font-style: normal; color: {THEME['accent']}; }}
.intro {{ font-size: 1.12rem; color: {THEME['ink_2']}; max-width: 54rem; margin-bottom: 1.4rem; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.8rem; margin-bottom: 1.2rem; }}
.tile {{ background: rgba(23,29,37,0.85); border: 1px solid #2A323D; border-radius: 0.35rem;
        padding: 0.9rem 1rem; }}
.tile .label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
               text-transform: uppercase; color: {THEME['muted']}; }}
.tile .value {{ font-weight: 700; font-size: 2.1rem; color: {THEME['ink']}; line-height: 1.2; }}
.tile .note {{ font-size: 0.9rem; color: {THEME['ink_2']}; }}
.tile .value.accent {{ color: {THEME['accent']}; }}
@media (prefers-reduced-motion: no-preference) {{
  .rise {{ opacity: 0; animation: rise 0.6s ease-out forwards; }}
  .d1 {{ animation-delay: 0.05s; }} .d2 {{ animation-delay: 0.15s; }} .d3 {{ animation-delay: 0.25s; }}
  .d4 {{ animation-delay: 0.35s; }} .d5 {{ animation-delay: 0.42s; }} .d6 {{ animation-delay: 0.49s; }}
  .d7 {{ animation-delay: 0.56s; }}
  @keyframes rise {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
}}
@media (max-width: 900px) {{ .tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .headline {{ font-size: 2.2rem; }} }}
</style>
""", unsafe_allow_html=True)


def euro(value):
    """Format money with the minus sign in front: -EUR 5.20 rather than EUR -5.20."""
    return f"{'−' if value < 0 else ''}€{abs(value):,.2f}"


def chart_layout(fig, height=380, y_title=None, x_title=None):
    """Apply the same quiet chart style everywhere: hairline grid, one axis, theme fonts."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["surface"],
        font=dict(family="SourceSans3, sans-serif", color=THEME["ink_2"], size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor="#0B0E12", bordercolor=THEME["axis"],
                        font=dict(family="SourceSans3, sans-serif", color=THEME["ink"])),
        bargap=0.25, bargroupgap=0.08,
    )
    fig.update_xaxes(gridcolor=THEME["grid"], linecolor=THEME["axis"], zeroline=False, title=x_title,
                     tickfont=dict(family="JetBrains Mono, monospace", size=11))
    fig.update_yaxes(gridcolor=THEME["grid"], linecolor=THEME["axis"], zerolinecolor=THEME["axis"],
                     zerolinewidth=1, title=y_title, tickfont=dict(family="JetBrains Mono, monospace", size=11))
    return fig


# =============================================================================
# Data and running the lab (cached, so changing tabs doesn't re-run anything)
# =============================================================================
def data_signature():
    """Changes whenever a price file is added or updated, so the cache knows to reload."""
    files = sorted(lab.HISTORY_DIR.glob("*_60m_*.csv")) + sorted(lab.RAW_DIR.glob("*_60m_*.csv"))
    return tuple((f.name, f.stat().st_mtime) for f in files)


@st.cache_resource(show_spinner=False)
def load_data(signature):
    return lab.load_all(lab.PAIRS)


@st.cache_data(show_spinner=False)
def run_test(signature, pairs, fees, pools, target, lookback_weeks, top_coins, stop_multiple, picky):
    lab.TARGET_PROFIT = target
    lab.STOP_MULTIPLE = stop_multiple
    lab.PICKY_MARGIN = 0.05 if picky else None
    lab.TRAIN_HOURS = 24 * 7 * lookback_weeks
    lab.TOP_TREND_COINS = top_coins
    data = load_data(signature)
    subset = {p: data[p] for p in pairs if p in data}
    return lab.run_lab(subset, list(fees), list(pools))


signature = data_signature()
data = load_data(signature)

# =============================================================================
# Sidebar: settings, grouped in the order you use them
# =============================================================================
with st.sidebar:
    st.markdown("### 1 · Prices")
    if data:
        first = min(d["df"]["time"].iloc[0] for d in data.values())
        last = max(d["df"]["time"].iloc[-1] for d in data.values())
        st.caption(f"Hourly prices from **{first:%d %b %Y}** to **{last:%d %b %Y %H:%M}** UTC")
    if st.button("Download latest prices", width="stretch",
                 help="Runs fetch_prices.py: the latest 12 hours (1-minute), 7.5 days (15-minute) and 30 days (hourly)"):
        with st.spinner("Asking Kraken for the latest candles…"):
            fetch_prices.main()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.markdown("### 2 · Test settings")
    pairs = st.multiselect("Coins", list(COIN_NAMES), default=list(COIN_NAMES),
                           format_func=lambda p: COIN_NAMES[p])
    fee = st.radio("Fees", list(lab.FEE_LEVELS), index=1,
                   help="Kraken's lowest tier. Market orders (taker): spot 0.80%, derivatives 0.05% per buy or sell. "
                        "Maker levels use limit orders for the entry and target: cheaper and no slippage, "
                        "but an entry only fills if the price comes back to it within the hour.")
    pools = st.multiselect("What the bot may choose from", lab.POOLS, default=lab.POOLS[:4],
                           help="'Fade surge breakout' bets that breakouts fail. That idea came from this same data, "
                                "so treat its result as a hypothesis to check on new prices.")
    target = st.number_input("Profit target per trade (EUR, on a EUR 100 stake)", 0.25, 5.0, 1.0, 0.25)
    lookback = st.slider("Look-back before each week (weeks)", 1, 6, 3,
                         help="Each week the bot picks the combination that did best over this many past weeks.")
    stop_multiple = st.select_slider("Stop distance compared with the target", [0.5, 1.0, 1.5, 2.0, 3.0], 1.0,
                                     format_func=lambda x: f"{x:g}x",
                                     help="1x: the stop is as far from the entry as the target. A wider stop wins more "
                                          "often but loses more each time.")
    picky = st.toggle("Stricter confidence rule",
                      help="Only trade a combination whose look-back hit rate beat break-even by 5 points over at "
                           "least 10 trades.")
    top_coins = st.slider("Trending coins traded per week", 1, 4, 2,
                          help="For the 'trending coins only' option: how many of the strongest-trending coins may trade.")
    st.button("Run test", type="primary", width="stretch")   # any change re-runs; this makes it explicit
    st.caption("Paper trading only: pretend money, real Kraken prices. Not financial advice.")

if not data:
    st.error("No hourly prices found. In PowerShell run `py src\\download_history.py`, then `py src\\fetch_prices.py`.")
    st.stop()
if not pairs or not pools:
    st.warning("Pick at least one coin and one option in the sidebar.")
    st.stop()

with st.spinner("Testing every indicator and combination, week by week…"):
    results = run_test(signature, tuple(pairs), (fee,), tuple(pools), target, lookback, top_coins,
                       stop_multiple, picky)

summary, weeks, trades = results["summary"], results["weeks"], results["trades"]
accuracy, ranking = results["accuracy"], results["ranking"]

# =============================================================================
# Header with headline numbers
# =============================================================================
n_trades = int(summary["trades"].sum())
hits = int(summary["hit_EUR1"].sum())
hit_rate = 100 * hits / n_trades if n_trades else 0.0
break_even = summary["break_even_%"].iloc[0]
profit = summary["profit_EUR"].sum()
best = summary.groupby("pool")["profit_EUR"].sum().sort_values(ascending=False)

st.markdown(f"""
<div class="eyebrow rise d1">Crypto paper trader · indicator lab</div>
<div class="headline rise d2">Can popular indicators earn <em>€{target:g} a trade?</em></div>
<div class="intro rise d3">Ten indicators traders watch on TradingView vote every hour. Each week the bot picks the
indicator or combination that made the most money over the previous {lookback} weeks, then trades it on prices it has
not seen yet, aiming for €{target:g} after fees on every €100 trade. These are the honest, out-of-sample results.</div>
<div class="tiles">
  <div class="tile rise d4"><div class="label">Trades made</div><div class="value">{n_trades:,}</div>
    <div class="note">{len(pairs)} coins · {fee} fees</div></div>
  <div class="tile rise d5"><div class="label">Hit the €{target:g} target</div>
    <div class="value">{hit_rate:.0f}%</div><div class="note">needs {break_even:.0f}% to break even</div></div>
  <div class="tile rise d6"><div class="label">Profit after costs</div>
    <div class="value accent">{euro(profit)}</div><div class="note">€100 per trade, not compounded</div></div>
  <div class="tile rise d7"><div class="label">Least-bad option</div>
    <div class="value" style="font-size:1.35rem;padding-top:0.45rem">{best.index[0]}</div>
    <div class="note">{euro(best.iloc[0])} in total</div></div>
</div>
""", unsafe_allow_html=True)

tab_results, tab_weeks, tab_trades, tab_ind, tab_trend = st.tabs(
    ["Results", "Week by week", "Trades on the chart", "Indicator accuracy", "Trend ranking"])

# --- Results ------------------------------------------------------------------
with tab_results:
    st.markdown("#### Profit after costs, by coin")
    fig = go.Figure()
    for pool in pools:
        part = summary[summary["pool"] == pool]
        fig.add_bar(x=[COIN_NAMES[p] for p in part["pair"]], y=part["profit_EUR"], name=pool,
                    marker_color=POOL_COLOURS[pool], marker_line_width=0,
                    customdata=np.stack([part["trades"], part["hit_rate_%"]], axis=1),
                    hovertemplate="%{x}<br>" + pool + "<br>€%{y:.2f} from %{customdata[0]} trades"
                                  "<br>hit rate %{customdata[1]}%<extra></extra>")
    hold = summary.drop_duplicates("pair")
    fig.add_bar(x=[COIN_NAMES[p] for p in hold["pair"]], y=hold["buy_hold_EUR"], name="just holding the coin",
                marker_color=THEME["muted"], marker_line_width=0,
                hovertemplate="%{x}: holding €100 for the whole test → €%{y:.2f}<extra></extra>")
    st.plotly_chart(chart_layout(fig, y_title="EUR"), width="stretch")

    st.markdown("#### Hit rate against the break-even line")
    fig = go.Figure()
    for pool in pools:
        part = summary[(summary["pool"] == pool) & (summary["trades"] > 0)]
        fig.add_scatter(x=[COIN_NAMES[p] for p in part["pair"]], y=part["hit_rate_%"], mode="markers",
                        name=pool, marker=dict(size=13, color=POOL_COLOURS[pool], symbol=POOL_SYMBOL[pool],
                                               line=dict(width=2, color=THEME["surface"])),
                        customdata=part["trades"],
                        hovertemplate="%{x}: %{y}% of %{customdata} trades hit the target<extra>" + pool + "</extra>")
    fig.add_hline(y=break_even, line_color=THEME["ink"], line_width=1.5,
                  annotation_text=f"break-even {break_even:.0f}%", annotation_position="top left",
                  annotation_font_color=THEME["ink"])
    st.plotly_chart(chart_layout(fig, height=320, y_title="% of trades"), width="stretch")
    st.caption("A losing trade costs more than a winning trade earns, because fees are paid either way. "
               "Dots below the break-even line lose money overall.")
    with st.expander("Table view"):
        st.dataframe(summary.drop(columns=["fees"]), hide_index=True, width="stretch")

# --- Week by week -------------------------------------------------------------
with tab_weeks:
    st.markdown("#### Running total of profit, all selected coins")
    fig = go.Figure()
    for pool in pools:
        part = weeks[weeks["pool"] == pool].groupby("week_start")["week_profit"].sum().cumsum()
        fig.add_scatter(x=part.index, y=part.values, mode="lines", name=pool,
                        line=dict(color=POOL_COLOURS[pool], width=2, dash=POOL_DASH[pool]),
                        hovertemplate="week of %{x|%d %b}: €%{y:.2f} so far<extra>" + pool + "</extra>")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(chart_layout(fig, y_title="EUR"), width="stretch")

    c1, c2 = st.columns(2)
    coin_w = c1.selectbox("Coin", pairs, format_func=lambda p: COIN_NAMES[p], key="weeks_coin")
    pool_w = c2.selectbox("Option", pools, key="weeks_pool")
    picks = weeks[(weeks["pair"] == coin_w) & (weeks["pool"] == pool_w)]
    st.dataframe(
        picks[["week_start", "chosen", "lookback_trades", "lookback_profit", "week_trades", "week_profit"]],
        hide_index=True, width="stretch",
        column_config={
            "week_start": st.column_config.DatetimeColumn("Week starting", format="D MMM YYYY"),
            "chosen": st.column_config.TextColumn("Picked by the bot", width="large"),
            "lookback_trades": st.column_config.NumberColumn("Look-back trades"),
            "lookback_profit": st.column_config.NumberColumn("Look-back profit", format="€%.2f"),
            "week_trades": st.column_config.NumberColumn("Trades that week"),
            "week_profit": st.column_config.NumberColumn("Profit that week", format="€%.2f"),
        })

# --- Trades on the chart --------------------------------------------------------
with tab_trades:
    c1, c2 = st.columns(2)
    coin_t = c1.selectbox("Coin", pairs, format_func=lambda p: COIN_NAMES[p], key="trades_coin")
    pool_t = c2.selectbox("Option", pools, key="trades_pool")
    df = data[coin_t]["df"]
    fig = go.Figure()
    fig.add_scatter(x=df["time"], y=df["close"], mode="lines", name="price (EUR)",
                    line=dict(color=THEME["muted"], width=1.5), hovertemplate="%{x|%d %b %H:%M}: €%{y:,.4g}<extra></extra>")
    sel = trades[(trades["pair"] == coin_t) & (trades["pool"] == pool_t)] if not trades.empty else trades
    if not sel.empty:
        for won, colour, label in [(True, THEME["gain"], "hit the target"), (False, THEME["loss"], "missed")]:
            part = sel[(sel["exit_reason"] == lab.TARGET_REASON) == won]
            fig.add_scatter(
                x=part["entry_time"], y=part["entry_price"], mode="markers", name=label,
                marker=dict(size=11, color=colour, line=dict(width=2, color=THEME["surface"]),
                            symbol=np.where(part["side"] == "long", "triangle-up", "triangle-down")),
                customdata=np.stack([part["side"], part["combo"], part["exit_reason"], part["net_profit"]], axis=1),
                hovertemplate="%{x|%d %b %H:%M} · %{customdata[0]}<br>%{customdata[1]}<br>"
                              "%{customdata[2]}: €%{customdata[3]:.2f}<extra></extra>")
    st.plotly_chart(chart_layout(fig, height=460, y_title="EUR"), width="stretch")
    st.caption("▲ long (bet on a rise) · ▼ short (bet on a fall). Blue = reached the target, red = did not.")
    if not sel.empty:
        with st.expander(f"All {len(sel)} trades"):
            st.dataframe(sel.drop(columns=["fees", "pool", "pair"]), hide_index=True, width="stretch")

# --- Indicator accuracy -----------------------------------------------------------
with tab_ind:
    st.markdown("#### How often each indicator called the next 24 hours correctly")
    grid = accuracy.pivot(index="indicator", columns="pair", values="all_hours_%")
    grid = grid[[p for p in pairs if p in grid.columns]]
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[COIN_NAMES[p] for p in grid.columns], y=grid.index, zmid=50, zmin=40, zmax=60,
        colorscale=[[0, THEME["loss"]], [0.5, THEME["neutral"]], [1, THEME["gain"]]],
        texttemplate="%{z:.1f}%", textfont=dict(color=THEME["ink"]),
        xgap=2, ygap=2, colorbar=dict(title="%", ticksuffix="%"),
        hovertemplate="%{y} on %{x}: right %{z:.1f}% of the time<extra></extra>"))
    st.plotly_chart(chart_layout(fig, height=440), width="stretch")
    st.caption("50% is a coin flip (grey). Blue is better than a coin flip, red is worse. "
               f"With a €{target:g} target the bot needs about {break_even:.0f}% to break even.")
    comp = accuracy.groupby(["indicator", "family"])[["all_hours_%", "trending_weeks_%"]].mean().round(1).reset_index()
    st.markdown("#### All hours vs weeks when the coin was among the strongest trends")
    st.dataframe(comp, hide_index=True, width="stretch", column_config={
        "all_hours_%": st.column_config.NumberColumn("All hours", format="%.1f%%"),
        "trending_weeks_%": st.column_config.NumberColumn("Trending weeks", format="%.1f%%")})

# --- Trend ranking ------------------------------------------------------------------
with tab_trend:
    st.markdown("#### Which coins were trending each week")
    st.caption("Trend strength (efficiency ratio): how straight the price path was over the previous week. "
               "100 = a straight line up or down, 0 = pure zig-zag. Uses only prices before the week starts.")
    show = ranking[["week_start", "trending"] + [p for p in pairs if p in ranking.columns]].copy()
    for p in pairs:
        if p in show.columns:
            show[p] = (show[p] * 100).round(0)
            show["trending"] = show["trending"].str.replace(p, COIN_NAMES[p])
    st.dataframe(show.iloc[::-1], hide_index=True, width="stretch", column_config={
        "week_start": st.column_config.DatetimeColumn("Week starting", format="D MMM YYYY"),
        "trending": st.column_config.TextColumn("Traded that week", width="medium"),
        **{p: st.column_config.ProgressColumn(COIN_NAMES[p], min_value=0, max_value=100, format="%d")
           for p in pairs if p in show.columns}})
