"""
SIGNAL Strategy Backtester
===========================
Backtests the exact current strategy logic against Alpaca historical data.

Tests:
- +5% profit target
- -5% stop loss (or -2% in bear conditions)
- +4% peak → sell at +2% trailing protection
- 60/40 or 40/60 TA/Fund blend (simulated via TA score only for speed)
- SPY-based market regime filter

Outputs:
- Win rate, avg gain, avg loss, expectancy
- Sharpe ratio, Sortino ratio
- Max drawdown
- Monthly returns breakdown
- Best/worst trades
- Kelly optimal position size

Usage:
    pip install alpaca-py pandas numpy
    python backtest.py

Environment variables required:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backtest")

# ── Config ─────────────────────────────────────────────────────
ALPACA_KEY    = os.environ.get("ALPACA_API_KEY", "PKRTBHHT6MGSTV6NJ6HU45PVW4")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "6QaZu55EcVuEiEYWZFZksFAUxFE3WxMSHvWky49ULah3")
DATA_URL      = "https://data.alpaca.markets/v2"
HEADERS       = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# Strategy parameters — mirror exact bot settings
PROFIT_TARGET  =  0.05
STOP_LOSS      = -0.05
PEAK_TRIGGER   =  0.04
TRAIL_SELL     =  0.02
BEAR_STOP      = -0.02
SPY_CAUTION    = -0.01
SPY_BEAR       = -0.02
MIN_TA_SCORE   =  2.0    # minimum taScore to consider a BUY signal

# Backtest universe — same as bot
UNIVERSE = [
    "NVDA","AAPL","MSFT","AMZN","META","GOOG","TSLA","AMD",
    "AVGO","QCOM","ARM","PANW","ASML","MU","ORCL","CRM",
    "SNOW","PLTR","LLY","NVO","ABBV","UNH","JPM","GS",
    "V","MA","BLK","XOM","CVX","NEE","ENPH","GE","CAT",
    "UBER","SPOT","NFLX","CRWD","NET","DDOG","NOW","SPY",
]

# ── Data fetching ───────────────────────────────────────────────
def fetch_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV bars from Alpaca."""
    try:
        r = requests.get(
            f"{DATA_URL}/stocks/{symbol}/bars",
            headers=HEADERS,
            params={"timeframe": "1Day", "start": start, "end": end,
                    "limit": 10000, "feed": "iex"},
            timeout=15
        )
        r.raise_for_status()
        bars = r.json().get("bars", [])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        df.set_index("t", inplace=True)
        df.index = df.index.tz_localize(None)
        return df[["o","h","l","c","v"]]
    except Exception as e:
        log.warning(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()

# ── Technical indicator computation ────────────────────────────
def compute_ta_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes the same taScore as the Cloudflare Worker for each day.
    Returns a Series of scores indexed by date.
    """
    closes = df["c"]
    highs  = df["h"]
    lows   = df["l"]
    vols   = df["v"]

    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    def rsi(series, period=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    e20  = ema(closes, 20)
    e50  = ema(closes, 50)
    e200 = ema(closes, 200)
    rsi_vals = rsi(closes)

    # MACD
    macd_line   = ema(closes, 12) - ema(closes, 26)
    signal_line = ema(macd_line, 9)
    macd_hist   = macd_line - signal_line

    # Bollinger Bands
    bb_mid   = closes.rolling(20).mean()
    bb_std   = closes.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # Volume ratio
    avg_vol  = vols.rolling(10).mean()
    vol_ratio = vols / avg_vol

    # Score each day
    scores = pd.Series(0.0, index=df.index)

    scores += np.where(closes > e20,  1, -1)
    scores += np.where(e20 > e50,     1, -1)
    scores += np.where(closes > e200, 1, -1)

    # RSI
    scores += np.where(rsi_vals < 30,  2,
              np.where(rsi_vals > 70, -2,
              np.where(rsi_vals > 50,  0.5, -0.5)))

    # MACD crossover
    prev_hist = macd_hist.shift(1)
    scores += np.where((macd_hist > 0) & (prev_hist <= 0),  2,
              np.where((macd_hist < 0) & (prev_hist >= 0), -2,
              np.where(macd_line > signal_line,  1, -1)))

    # Bollinger
    scores += np.where(closes < bb_lower,  1.5,
              np.where(closes > bb_upper, -1.5, 0))

    # Volume
    pct_1d = closes.pct_change()
    scores += np.where((vol_ratio > 1.5) & (pct_1d > 0),  1,
              np.where((vol_ratio > 1.5) & (pct_1d < 0), -1, 0))

    return scores.clip(-10, 10)

# ── Trade simulation ────────────────────────────────────────────
@dataclass
class Trade:
    symbol:      str
    entry_date:  pd.Timestamp
    entry_price: float
    exit_date:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    exit_reason: str                    = ""
    pnl_pct:     float                  = 0.0
    peak_pct:    float                  = 0.0
    market_mode: str                    = "BULL"

def simulate_trade(
    symbol: str,
    entry_date: pd.Timestamp,
    entry_price: float,
    price_data: pd.DataFrame,
    spy_data: pd.DataFrame,
    market_mode: str
) -> Trade:
    """
    Simulates holding a position from entry_date until one of the exit rules triggers.
    Uses actual daily OHLC data — checks intraday H/L for trigger accuracy.
    """
    trade = Trade(symbol=symbol, entry_date=entry_date,
                  entry_price=entry_price, market_mode=market_mode)

    # Determine stop loss based on market mode
    active_stop  = BEAR_STOP if market_mode == "BEAR" else STOP_LOSS
    active_peak  = 0.02 if market_mode in ("BEAR","CAUTION") else PEAK_TRIGGER
    active_trail = 0.01 if market_mode in ("BEAR","CAUTION") else TRAIL_SELL

    peak_pct = 0.0
    future_bars = price_data[price_data.index > entry_date]

    for date, bar in future_bars.iterrows():
        high_pct = (bar["h"] - entry_price) / entry_price
        low_pct  = (bar["l"] - entry_price) / entry_price
        close_pct = (bar["c"] - entry_price) / entry_price

        # Update peak
        if high_pct > peak_pct:
            peak_pct = high_pct

        # Rule 1: Profit target hit (use high of day)
        if high_pct >= PROFIT_TARGET:
            trade.exit_date   = date
            trade.exit_price  = entry_price * (1 + PROFIT_TARGET)
            trade.exit_reason = "PROFIT_TARGET"
            trade.pnl_pct     = PROFIT_TARGET
            trade.peak_pct    = peak_pct
            return trade

        # Rule 2: Trailing protection (use close)
        if peak_pct >= active_peak and close_pct <= active_trail:
            trade.exit_date   = date
            trade.exit_price  = bar["c"]
            trade.exit_reason = "TRAILING"
            trade.pnl_pct     = close_pct
            trade.peak_pct    = peak_pct
            return trade

        # Rule 3: Stop loss hit (use low of day)
        if low_pct <= active_stop:
            trade.exit_date   = date
            trade.exit_price  = entry_price * (1 + active_stop)
            trade.exit_reason = "STOP_LOSS"
            trade.pnl_pct     = active_stop
            trade.peak_pct    = peak_pct
            return trade

    # Still open — close at last available price
    if len(future_bars) > 0:
        last_bar = future_bars.iloc[-1]
        trade.exit_date   = future_bars.index[-1]
        trade.exit_price  = last_bar["c"]
        trade.exit_reason = "END_OF_DATA"
        trade.pnl_pct     = (last_bar["c"] - entry_price) / entry_price
        trade.peak_pct    = peak_pct

    return trade

# ── Main backtest engine ────────────────────────────────────────
def run_backtest(start_date: str = "2022-01-01", end_date: str = "2025-12-31"):
    log.info("=" * 60)
    log.info("SIGNAL Strategy Backtester")
    log.info(f"  Period:    {start_date} to {end_date}")
    log.info(f"  Universe:  {len(UNIVERSE)-1} stocks + SPY")
    log.info(f"  Targets:   +{PROFIT_TARGET*100:.0f}% profit | {STOP_LOSS*100:.0f}% stop")
    log.info("=" * 60)

    # Fetch all data
    log.info("Fetching historical data...")
    all_data = {}
    for symbol in UNIVERSE:
        df = fetch_bars(symbol, start_date, end_date)
        if len(df) >= 200:
            all_data[symbol] = df
            log.info(f"  {symbol}: {len(df)} bars")
        else:
            log.warning(f"  {symbol}: insufficient data ({len(df)} bars) — skipping")
        time.sleep(0.3)  # rate limit

    spy_data = all_data.get("SPY", pd.DataFrame())
    tickers  = [s for s in all_data if s != "SPY"]
    log.info(f"Data loaded: {len(tickers)} tickers")

    # Compute TA scores for all tickers
    log.info("Computing technical indicators...")
    ta_scores = {}
    for symbol in tickers:
        ta_scores[symbol] = compute_ta_score(all_data[symbol])

    # Compute SPY daily returns for market regime
    spy_returns = pd.Series(dtype=float)
    if not spy_data.empty:
        spy_returns = spy_data["c"].pct_change()

    # Run simulation
    log.info("Simulating trades...")
    trades = []
    all_dates = sorted(set(
        date for df in all_data.values() for date in df.index
    ))

    for date in all_dates:
        # Determine market mode from SPY
        if date in spy_returns.index:
            spy_chg = spy_returns[date]
            if spy_chg <= SPY_BEAR:
                market_mode = "BEAR"
            elif spy_chg <= SPY_CAUTION:
                market_mode = "CAUTION"
            else:
                market_mode = "BULL"
        else:
            market_mode = "BULL"

        # Skip non-bull days for new entries
        if market_mode != "BULL":
            continue

        # Find BUY signals on this date
        for symbol in tickers:
            if date not in ta_scores[symbol].index:
                continue
            score = ta_scores[symbol][date]
            if score < MIN_TA_SCORE:
                continue

            # Enter at next day's open
            df = all_data[symbol]
            future = df[df.index > date]
            if future.empty:
                continue
            entry_bar   = future.iloc[0]
            entry_date  = future.index[0]
            entry_price = entry_bar["o"]
            if entry_price <= 0:
                continue

            trade = simulate_trade(
                symbol, entry_date, entry_price,
                df, spy_data, market_mode
            )
            if trade.exit_date:
                trades.append(trade)

    if not trades:
        log.error("No trades generated — check data and parameters")
        return

    # ── Analytics ───────────────────────────────────────────────
    log.info(f"\nAnalysing {len(trades)} trades...")
    df_trades = pd.DataFrame([{
        "symbol":      t.symbol,
        "entry_date":  t.entry_date,
        "exit_date":   t.exit_date,
        "entry_price": t.entry_price,
        "exit_price":  t.exit_price,
        "pnl_pct":     t.pnl_pct,
        "peak_pct":    t.peak_pct,
        "exit_reason": t.exit_reason,
        "market_mode": t.market_mode,
        "hold_days":   (t.exit_date - t.entry_date).days if t.exit_date else 0,
    } for t in trades])

    wins    = df_trades[df_trades["pnl_pct"] > 0]
    losses  = df_trades[df_trades["pnl_pct"] <= 0]
    win_rate = len(wins) / len(df_trades) * 100

    avg_win  = wins["pnl_pct"].mean() * 100 if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() * 100 if len(losses) > 0 else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Kelly Criterion
    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = (win_rate/100) - ((1 - win_rate/100) / win_loss_ratio)
        kelly_pct = max(0, kelly * 100)
    else:
        kelly_pct = 0

    # Equity curve
    df_trades_sorted = df_trades.sort_values("entry_date")
    equity = 100000.0
    equity_curve = []
    for _, t in df_trades_sorted.iterrows():
        position_size = equity * 0.5  # current 50% rule
        profit = position_size * t["pnl_pct"]
        equity += profit
        equity_curve.append({"date": t["exit_date"], "equity": equity})

    eq_df = pd.DataFrame(equity_curve)
    if not eq_df.empty:
        eq_df.set_index("date", inplace=True)
        rolling_max  = eq_df["equity"].cummax()
        drawdown     = (eq_df["equity"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        final_equity = eq_df["equity"].iloc[-1]
        total_return = (final_equity - 100000) / 100000 * 100
    else:
        max_drawdown = 0
        total_return = 0
        final_equity = 100000

    # Sharpe ratio (annualised, assuming risk-free = 4%)
    daily_returns = df_trades_sorted["pnl_pct"]
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() - 0.04/252) / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino ratio
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (daily_returns.mean() - 0.04/252) / downside.std() * np.sqrt(252)
    else:
        sortino = 0

    # Monthly returns
    if not eq_df.empty:
        monthly = eq_df["equity"].resample("ME").last().pct_change() * 100
    else:
        monthly = pd.Series()

    # Exit reason breakdown
    exit_counts = df_trades["exit_reason"].value_counts()

    # Best and worst trades
    best_trades  = df_trades.nlargest(5, "pnl_pct")[["symbol","entry_date","pnl_pct","exit_reason","hold_days"]]
    worst_trades = df_trades.nsmallest(5, "pnl_pct")[["symbol","entry_date","pnl_pct","exit_reason","hold_days"]]

    # Per-symbol performance
    symbol_perf = df_trades.groupby("symbol").agg(
        trades=("pnl_pct","count"),
        win_rate=("pnl_pct", lambda x: (x>0).mean()*100),
        avg_pnl=("pnl_pct", lambda x: x.mean()*100),
        total_pnl=("pnl_pct", "sum")
    ).sort_values("total_pnl", ascending=False)

    # ── Print results ────────────────────────────────────────────
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print(f"Period: {start_date} to {end_date}")
    print("="*60)

    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"  Total trades:       {len(df_trades)}")
    print(f"  Win rate:           {win_rate:.1f}%")
    print(f"  Avg win:            +{avg_win:.2f}%")
    print(f"  Avg loss:           {avg_loss:.2f}%")
    print(f"  Expectancy/trade:   {expectancy:.2f}%")
    print(f"  Total return:       {total_return:.1f}%")
    print(f"  Final equity:       ${final_equity:,.0f}")

    print(f"\n📈 RISK METRICS")
    print(f"  Sharpe ratio:       {sharpe:.2f}")
    print(f"  Sortino ratio:      {sortino:.2f}")
    print(f"  Max drawdown:       {max_drawdown:.1f}%")
    print(f"  Kelly optimal size: {kelly_pct:.1f}% of portfolio per trade")
    print(f"  Current bot size:   50% — {'OPTIMAL' if 45 <= kelly_pct <= 55 else 'ADJUST to '+str(round(kelly_pct))+'%'}")

    print(f"\n🚪 EXIT REASONS")
    for reason, count in exit_counts.items():
        pct = count / len(df_trades) * 100
        avg = df_trades[df_trades["exit_reason"]==reason]["pnl_pct"].mean()*100
        print(f"  {reason:<20} {count:>4} trades ({pct:.0f}%) avg {avg:+.2f}%")

    print(f"\n📅 MONTHLY RETURNS")
    if not monthly.empty:
        for date, ret in monthly.items():
            if not pd.isna(ret):
                bar = "█" * int(abs(ret)/2) if abs(ret) < 40 else "█"*20
                sign = "+" if ret >= 0 else ""
                print(f"  {date.strftime('%Y-%m')}  {sign}{ret:.1f}%  {bar}")

    print(f"\n🏆 BEST 5 TRADES")
    for _, t in best_trades.iterrows():
        print(f"  {t['symbol']:<6} {str(t['entry_date'])[:10]}  +{t['pnl_pct']*100:.1f}%  {t['exit_reason']}  {t['hold_days']}d")

    print(f"\n💀 WORST 5 TRADES")
    for _, t in worst_trades.iterrows():
        print(f"  {t['symbol']:<6} {str(t['entry_date'])[:10]}  {t['pnl_pct']*100:.1f}%  {t['exit_reason']}  {t['hold_days']}d")

    print(f"\n🏅 TOP 10 SYMBOLS BY TOTAL PNL")
    for symbol, row in symbol_perf.head(10).iterrows():
        print(f"  {symbol:<6} {row['trades']:>3} trades  {row['win_rate']:.0f}% WR  avg {row['avg_pnl']:+.1f}%")

    print(f"\n💡 KEY INSIGHTS")
    if expectancy > 0:
        print(f"  ✅ Strategy has POSITIVE expectancy ({expectancy:.2f}% per trade)")
    else:
        print(f"  ❌ Strategy has NEGATIVE expectancy ({expectancy:.2f}% per trade) — edge not confirmed")

    if sharpe > 1:
        print(f"  ✅ Sharpe ratio {sharpe:.2f} — good risk-adjusted returns")
    elif sharpe > 0.5:
        print(f"  ⚠️  Sharpe ratio {sharpe:.2f} — acceptable but improvable")
    else:
        print(f"  ❌ Sharpe ratio {sharpe:.2f} — poor risk-adjusted returns")

    if abs(max_drawdown) > 20:
        print(f"  ❌ Max drawdown {max_drawdown:.1f}% — high risk, consider tighter stops")
    elif abs(max_drawdown) > 10:
        print(f"  ⚠️  Max drawdown {max_drawdown:.1f}% — moderate, monitor closely")
    else:
        print(f"  ✅ Max drawdown {max_drawdown:.1f}% — well controlled")

    if kelly_pct < 30:
        print(f"  ⚠️  Kelly says {kelly_pct:.0f}% per trade — current 50% rule is OVERSIZING")
    elif kelly_pct > 60:
        print(f"  ✅ Kelly says {kelly_pct:.0f}% per trade — current 50% rule is conservative")
    else:
        print(f"  ✅ Kelly says {kelly_pct:.0f}% per trade — current 50% rule is appropriate")

    # Save full results to CSV
    df_trades.to_csv("/mnt/user-data/outputs/backtest_trades.csv", index=False)
    print(f"\n  Full trade log saved to backtest_trades.csv")
    print("="*60)

    return df_trades

if __name__ == "__main__":
    run_backtest(
        start_date="2022-01-01",
        end_date="2025-12-31"
    )
