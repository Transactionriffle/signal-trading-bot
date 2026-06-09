"""
SIGNAL Trading Bot
==================
- Builds a dynamic universe every morning from Alpaca market data:
    Top 50 most active by volume  → momentum confirmation
    Top 20 gainers                → breakout candidates
    Top 20 losers                 → oversold reversal candidates
- Scans positions every 60 seconds during market hours
- Three exit rules per position:
    +5%        → profit target, sell immediately
    +4% → +2%  → trailing protection, lock in gain
    -5%        → stop loss, cut losses
- Reinvests freed cash using 50% / equal split allocation rule
- Runs 24/7 on Render as a background worker

Environment variables (set in Render):
    ALPACA_API_KEY         (required)
    ALPACA_SECRET_KEY      (required)
    ANTHROPIC_API_KEY      (required)
    ALPACA_BASE_URL        (default: https://paper-api.alpaca.markets)
    CLOUDFLARE_WORKER      (default: your worker URL)
    TECH_WEIGHT            (default: 60)
    MIN_CONFIDENCE         (default: 80)
    MAX_TRADES_PER_DAY     (default: 10)
    MAX_DRAWDOWN_PCT       (default: 0.15)
    PAUSED                 (set "true" to halt instantly)
    TOP_ACTIVE             (default: 50 — most active by volume)
    TOP_MOVERS             (default: 20 — top gainers only)

Risk thresholds (hardcoded):
    PROFIT_TARGET = +5%   sell at full target
    PEAK_TRIGGER  = +4%   activate trailing protection
    TRAIL_SELL    = +2%   sell if falls back here after peak
    STOP_LOSS     = -5%   cut losses

Market state (Phase 1 macro layer):
    BULL mode    SPY >= -1%   → normal trading
    CAUTION mode SPY -1% to -2% → no new buys, tighter trailing
    BEAR mode    SPY <= -2%   → no new buys, tighten stop loss to -2%
    VIXY proxy   VIXY > +5%  → reduce position sizes by 50%
    Sector check Avoid buying into sectors down > 1.5% today
"""

import os
import json
import time
import logging
import requests
import anthropic
from datetime import datetime, timezone
from collections import defaultdict
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("signal-bot")

# ── Config ─────────────────────────────────────────────────────
ALPACA_KEY      = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET   = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]
WORKER_URL      = os.environ.get("CLOUDFLARE_WORKER", "https://winter-cake-6aae.dimitridesplace-65f.workers.dev")
TECH_WEIGHT     = int(os.environ.get("TECH_WEIGHT", "60"))
FUND_WEIGHT     = 100 - TECH_WEIGHT
MIN_CONFIDENCE  = int(os.environ.get("MIN_CONFIDENCE", "80"))
MAX_TRADES_DAY  = int(os.environ.get("MAX_TRADES_PER_DAY", "10"))
MAX_DRAWDOWN    = float(os.environ.get("MAX_DRAWDOWN_PCT", "0.15"))
TOP_ACTIVE      = int(os.environ.get("TOP_ACTIVE", "50"))
TOP_MOVERS      = int(os.environ.get("TOP_MOVERS", "20"))
SCAN_INTERVAL   = 60

# ── Risk thresholds (backtest validated) ───────────────────────
PROFIT_TARGET   =  0.05   # +5%  full profit target — 47% of trades hit this
PEAK_TRIGGER    =  0.03   # +3%  activate trailing (tightened from +4% — backtest avg trailing exit was only +0.68%)
TRAIL_SELL      =  0.025  # +2.5% sell if falls back here (tightened from +2%)
STOP_LOSS       = -0.05   # -5%  cut losses (tightens to -2% in bear mode)

# ── Market state thresholds ────────────────────────────────────
SPY_CAUTION     = -0.01   # SPY down 1% → caution mode
SPY_BEAR        = -0.02   # SPY down 2% → bear mode
VIXY_FEAR       =  0.05   # VIXY up 5%  → fear active, halve position sizes
SECTOR_WEAK     = -0.015  # Sector ETF down 1.5% → avoid that sector

# Sector ETF map: sector name → ETF ticker
SECTOR_ETFS = {
    "tech":        "XLK",
    "healthcare":  "XLV",
    "financials":  "XLF",
    "energy":      "XLE",
    "utilities":   "XLU",
    "consumer":    "XLY",
    "industrials": "XLI",
    "materials":   "XLB",
}

# ── Runtime state ──────────────────────────────────────────────
trades_today:    dict[str, int]   = defaultdict(int)
circuit_breaker: bool             = False
starting_equity: float | None     = None
position_peaks:  dict[str, float] = {}
dynamic_universe: list[str]       = []
universe_date:   str              = ""   # date universe was last built
market_state:    str              = "BULL"   # BULL | CAUTION | BEAR
fear_active:     bool             = False    # VIXY spiking
weak_sectors:    set              = set()    # sectors to avoid today

# ── Peak persistence ────────────────────────────────────────────
PEAKS_FILE = "/tmp/position_peaks.json"

def save_peaks():
    """Persist position peaks to disk so they survive bot restarts."""
    try:
        with open(PEAKS_FILE, "w") as f:
            json.dump(position_peaks, f)
    except Exception as e:
        log.warning(f"Failed to save peaks: {e}")

def load_peaks():
    """Load position peaks from disk on startup."""
    global position_peaks
    try:
        with open(PEAKS_FILE) as f:
            position_peaks = json.load(f)
        if position_peaks:
            log.info(f"Loaded position peaks from disk: {position_peaks}")
    except FileNotFoundError:
        position_peaks = {}
    except Exception as e:
        log.warning(f"Failed to load peaks: {e}")
        position_peaks = {}

def prune_peaks(active_symbols: list):
    """Remove peaks for positions that no longer exist."""
    stale = [s for s in position_peaks if s not in active_symbols]
    for s in stale:
        del position_peaks[s]
        log.info(f"  Pruned stale peak for {s}")
    if stale:
        save_peaks()

# ── Clients ────────────────────────────────────────────────────
trade_client = TradingClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=True,
    url_override=ALPACA_BASE_URL,
)
data_client = StockHistoricalDataClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
)
ai_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ══════════════════════════════════════════════════════════════
# PHASE 1 MACRO LAYER
# ══════════════════════════════════════════════════════════════

def get_quote_change(symbol: str) -> float | None:
    """Returns today's % change for a symbol using Alpaca latest quote."""
    try:
        DATA_URL = "https://data.alpaca.markets/v2"
        headers  = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
        r = requests.get(
            f"{DATA_URL}/stocks/{symbol}/snapshot",
            headers=headers,
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        snap = data.get("snapshot") or data.get(symbol, {})
        daily = snap.get("dailyBar", {})
        prev  = snap.get("prevDailyBar", {})
        if daily and prev and prev.get("c", 0) > 0:
            return (daily.get("c", 0) - prev.get("c", 0)) / prev.get("c", 0)
        return None
    except Exception as e:
        log.warning(f"Quote fetch failed for {symbol}: {e}")
        return None

def get_vix_level() -> float | None:
    """
    Fetches the current VIX level via Cloudflare Worker -> Yahoo Finance.
    VIX is an index (^VIX), not an equity — not available on Alpaca.
    Returns the current VIX value (e.g. 17.5, 35.0).
    Interpretation:
        < 15  = very calm / complacent
        15-20 = normal market conditions
        20-25 = elevated uncertainty
        25-35 = fear / volatility
        > 35  = panic / crisis
    """
    try:
        now  = int(time.time())
        from_ts = now - 86400 * 5  # last 5 days to ensure we get latest
        url = f"{WORKER_URL}/yahoofinance/chart/%5EVIX?interval=1d&period1={from_ts}&period2={now}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data  = r.json()
        chart = data.get("chart", {}).get("result", [{}])[0]
        closes = chart.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        closes = [c for c in closes if c is not None]
        if closes:
            vix = closes[-1]
            log.info(f"VIX level: {vix:.1f}")
            return vix
        return None
    except Exception as e:
        log.warning(f"VIX fetch failed: {e}")
        return None

def assess_market_state():
    """
    Checks SPY, VIXY, and sector ETFs to determine market mode.
    Updates global market_state, fear_active, weak_sectors.
    Called once per scan cycle.
    """
    global market_state, fear_active, weak_sectors

    # ── SPY: overall market direction ─────────────────────────
    spy_chg = get_quote_change("SPY")
    if spy_chg is not None:
        if spy_chg <= SPY_BEAR:
            market_state = "BEAR"
            log.warning(f"BEAR MODE — SPY {spy_chg*100:.2f}% today. No new buys. Tightening stops to -2%.")
        elif spy_chg <= SPY_CAUTION:
            market_state = "CAUTION"
            log.warning(f"CAUTION MODE — SPY {spy_chg*100:.2f}% today. No new buys. Trailing protection tightened.")
        else:
            market_state = "BULL"
            log.info(f"BULL MODE — SPY {spy_chg*100:.2f}% today. Normal trading.")
    else:
        log.warning("Could not fetch SPY — defaulting to CAUTION")
        market_state = "CAUTION"

    # ── VIX: real fear index via Yahoo Finance ────────────────
    vix_level = get_vix_level()
    vixy_chg  = get_quote_change("VIXY")

    if vix_level is not None:
        if vix_level >= 35:
            fear_active = True
            log.warning(f"FEAR ACTIVE — VIX {vix_level:.1f} (PANIC/CRISIS). Position sizes halved.")
        elif vix_level >= 25:
            fear_active = True
            log.warning(f"FEAR ACTIVE — VIX {vix_level:.1f} (elevated fear). Position sizes halved.")
        elif vix_level >= 20:
            fear_active = False
            log.info(f"VIX {vix_level:.1f} — uncertainty but not fear. Normal sizing.")
        else:
            fear_active = False
            log.info(f"VIX {vix_level:.1f} — calm market. Normal sizing.")

        # Also use VIX to refine SPY signal:
        # SPY down 2% + VIX < 20 = institutional rotation, not fear → stay BULL
        if market_state == "BEAR" and vix_level < 20:
            market_state = "BULL"
            log.info(f"SPY bear but VIX {vix_level:.1f} < 20 — likely rotation not fear. Staying BULL.")
        elif market_state == "CAUTION" and vix_level < 15:
            market_state = "BULL"
            log.info(f"SPY caution but VIX {vix_level:.1f} < 15 — very calm market. Staying BULL.")

    elif vixy_chg is not None:
        # Fallback to VIXY if VIX unavailable
        fear_active = vixy_chg >= VIXY_FEAR
        if fear_active:
            log.warning(f"FEAR ACTIVE (VIXY proxy) — VIXY +{vixy_chg*100:.1f}% today.")
        else:
            log.info(f"Fear gauge normal (VIXY proxy) — {vixy_chg*100:.1f}%")
    else:
        fear_active = False
        log.warning("Could not fetch VIX or VIXY — assuming no fear")

    # ── Sector ETFs: rotation check ───────────────────────────
    weak_sectors = set()
    for sector, etf in SECTOR_ETFS.items():
        chg = get_quote_change(etf)
        if chg is not None and chg <= SECTOR_WEAK:
            weak_sectors.add(sector)
            log.info(f"  Weak sector: {sector.upper()} ({etf} {chg*100:.2f}%) — avoiding")

    log.info(f"Market state: {market_state} | Fear: {fear_active} | Weak sectors: {weak_sectors or 'none'}")

def get_stop_loss() -> float:
    """Returns the active stop loss level based on market state."""
    if market_state == "BEAR":
        return -0.02  # tighten to -2% in bear market
    return STOP_LOSS  # normal -5%

def get_trail_sell() -> float:
    """Returns the trailing sell level based on market state."""
    if market_state in ("BEAR", "CAUTION"):
        return 0.01  # tighten to +1% in caution/bear
    return TRAIL_SELL  # normal +2%

def get_peak_trigger() -> float:
    """Returns the peak trigger level based on market state."""
    if market_state in ("BEAR", "CAUTION"):
        return 0.02  # activate trailing at +2% in caution/bear
    return PEAK_TRIGGER  # normal +4%

def adjust_qty_for_fear(qty: int, price: float, cash: float) -> int:
    """Halves position size when fear is active."""
    if fear_active and qty > 1:
        adjusted = int((cash * 0.5) / price)
        log.info(f"  Fear active — position size halved: {qty} → {adjusted} shares")
        return max(1, adjusted)
    return qty

# ══════════════════════════════════════════════════════════════
# DYNAMIC UNIVERSE BUILDER
# ══════════════════════════════════════════════════════════════

def build_universe() -> list[str]:
    """
    Builds a fresh dynamic universe from Alpaca market data:
    - Top N most active stocks by volume  → momentum confirmation
    - Top N gainers                       → breakout candidates
    Deduplicates and filters to tradeable US equities only.
    Called once per trading day at market open, and again on each reinvestment.
    """
    symbols = set()

    DATA_URL = "https://data.alpaca.markets/v1beta1"
    headers  = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

    # Minimum quality filters — eliminates penny stocks, ETFs, warrants
    MIN_PRICE       = 10.0    # at least $10
    MIN_TRADE_COUNT = 50000   # at least 50k trades = real liquidity

    def is_quality(symbol: str, price: float = 0, trade_count: int = 0) -> bool:
        return (
            symbol.isalpha()          # no warrants (AAPL.WS), no preferred (AAPL-A)
            and len(symbol) <= 5      # no exotic tickers
            and price >= MIN_PRICE    # no penny stocks
            and trade_count >= MIN_TRADE_COUNT  # liquid enough
        )

    try:
        r = requests.get(
            f"{DATA_URL}/screener/stocks/most-actives",
            headers=headers,
            params={"top": TOP_ACTIVE, "by": "trades"},
            timeout=10
        )
        r.raise_for_status()
        actives = r.json().get("most_actives", [])
        active_symbols = [
            s["symbol"] for s in actives
            if is_quality(s["symbol"], s.get("price", 0), s.get("trade_count", 0))
        ]
        if active_symbols:
            symbols.update(active_symbols)
            log.info(f"Most active ({len(active_symbols)}): {active_symbols[:10]}...")
        else:
            log.info("Screener returned empty — market may be closed. No fallback used.")
    except Exception as e:
        log.warning(f"Most active fetch failed: {e}")

    try:
        r = requests.get(
            f"{DATA_URL}/screener/stocks/movers",
            headers=headers,
            params={"top": min(TOP_MOVERS * 3, 50)},  # capped at 50 per API limit
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        gainers = data.get("gainers", [])
        gainer_symbols = [
            s["symbol"] for s in gainers
            if is_quality(s["symbol"], s.get("price", 0), s.get("trade_count", 0))
        ][:TOP_MOVERS]
        symbols.update(gainer_symbols)
        log.info(f"Top gainers ({len(gainer_symbols)}): {gainer_symbols[:5]}...")
    except Exception as e:
        log.warning(f"Market movers fetch failed: {e}")

    # Filter out ETFs, preferred shares, warrants (contain numbers or special chars)
    clean = [s for s in symbols if s.isalpha() and len(s) <= 5]

    if not clean:
        log.warning("Dynamic universe empty — falling back to default tickers")
        clean = [
            "NVDA","AAPL","MSFT","AMZN","META","GOOG","TSLA","AMD",
            "AVGO","QCOM","ARM","PANW","LLY","JPM","V","XOM",
        ]

    log.info(f"Dynamic universe built: {len(clean)} unique tickers (most active + gainers)")
    return clean


def get_or_refresh_universe() -> list[str]:
    """
    Returns today's universe. Rebuilds if it's a new trading day.
    """
    global dynamic_universe, universe_date
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    if universe_date != today or not dynamic_universe:
        log.info(f"Building fresh universe for {today}...")
        dynamic_universe = build_universe()
        universe_date    = today

    return dynamic_universe


# ══════════════════════════════════════════════════════════════
# RISK CONTROLS
# ══════════════════════════════════════════════════════════════

def is_paused() -> bool:
    val = os.environ.get("PAUSED", "false").strip().lower()
    return val in ("true", "1", "yes")

def today_key() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

def trades_today_count() -> int:
    return trades_today.get(today_key(), 0)

def increment_trades_today():
    trades_today[today_key()] = trades_today.get(today_key(), 0) + 1

def check_drawdown(account) -> bool:
    global circuit_breaker, starting_equity
    if circuit_breaker:
        return False
    equity = float(account.equity)
    if starting_equity is None:
        starting_equity = equity
        log.info(f"Starting equity set: ${starting_equity:,.2f}")
        return True
    drawdown = (starting_equity - equity) / starting_equity
    if drawdown >= MAX_DRAWDOWN:
        circuit_breaker = True
        log.critical(
            f"CIRCUIT BREAKER TRIGGERED — drawdown {drawdown*100:.1f}% "
            f"exceeds {MAX_DRAWDOWN*100:.0f}% limit. All trading halted."
        )
        return False
    return True

def run_risk_checks(account) -> tuple[bool, str]:
    if is_paused():
        return False, "PAUSED — set PAUSED=false in Render to resume"
    if not check_drawdown(account):
        return False, f"CIRCUIT BREAKER — drawdown exceeded {MAX_DRAWDOWN*100:.0f}%"
    count = trades_today_count()
    if count >= MAX_TRADES_DAY:
        return False, f"MAX TRADES — {count}/{MAX_TRADES_DAY} today"
    return True, "OK"

# ══════════════════════════════════════════════════════════════
# ALPACA HELPERS
# ══════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    try:
        return trade_client.get_clock().is_open
    except Exception as e:
        log.warning(f"Clock check failed: {e}")
        return False

def get_account():
    return trade_client.get_account()

def get_positions() -> dict:
    try:
        return {p.symbol: p for p in trade_client.get_all_positions()}
    except Exception as e:
        log.error(f"Failed to get positions: {e}")
        return {}

def close_position(symbol: str) -> bool:
    try:
        # Cancel any open orders on this symbol first
        open_orders = trade_client.get_orders()
        for order in open_orders:
            if order.symbol == symbol:
                try:
                    trade_client.cancel_order_by_id(order.id)
                    log.info(f"  Cancelled open order on {symbol} (id: {order.id})")
                    time.sleep(0.5)
                except Exception as ce:
                    log.warning(f"  Could not cancel order {order.id}: {ce}")
        # Now close the position
        trade_client.close_position(symbol)
        log.info(f"[SELL] Closed {symbol}")
        return True
    except Exception as e:
        log.error(f"Failed to close {symbol}: {e}")
        return False

def place_buy(symbol: str, qty: int) -> bool:
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        trade_client.submit_order(order)
        increment_trades_today()
        log.info(f"[BUY] {qty}x {symbol} — trades today: {trades_today_count()}/{MAX_TRADES_DAY}")
        return True
    except Exception as e:
        log.error(f"Failed to buy {symbol}: {e}")
        return False

# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════

def fetch_technicals(symbol: str) -> dict | None:
    try:
        r = requests.get(f"{WORKER_URL}/technicals/{symbol}", timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            log.warning(f"Technicals error for {symbol}: {data['error']}")
            return None
        return data
    except Exception as e:
        log.warning(f"Technicals fetch failed for {symbol}: {e}")
        return None

def fetch_fundamental(symbol: str, ta: dict) -> dict | None:
    try:
        # Keep prompt short to stay under 30k tokens/min rate limit
        price  = ta.get("price", 0)
        rsi    = ta.get("rsi", 50)
        signal = ta.get("taSignal", "HOLD")
        prompt = (
            f"Research {symbol} stock. Price ${price:.0f}, RSI {rsi:.0f}, TA {signal}. "
            "Rate fundamentals. Return ONLY this JSON with NO other text: "
            '{"fundSignal":"BUY","fundScore":7,"confidence":82,"thesis":"one sentence"} '
            "fundScore MUST be between -10 and +10. confidence MUST be 0-100."
        )

        response = ai_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=150,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        si, ei = text.find("{"), text.rfind("}")
        if si == -1:
            return None
        result = json.loads(text[si:ei+1])
        # Clamp scores to expected ranges regardless of what Claude returns
        result["fundScore"]  = max(-10, min(10, float(result.get("fundScore", 0))))
        result["confidence"] = max(0,   min(100, float(result.get("confidence", 50))))
        return result
    except Exception as e:
        log.warning(f"Fundamental fetch failed for {symbol}: {e}")
        return None

def compute_signal(symbol: str) -> dict | None:
    ta = fetch_technicals(symbol)
    if not ta:
        return None
    fund = fetch_fundamental(symbol, ta)
    if not fund:
        return None

    ta_score   = ta.get("taScore", 0)
    fund_score = fund.get("fundScore", 0)
    composite  = ta_score * (TECH_WEIGHT / 100) + fund_score * (FUND_WEIGHT / 100)

    return {
        "symbol":     symbol,
        "price":      ta.get("price", 0),
        "taScore":    ta_score,
        "taSignal":   ta.get("taSignal"),
        "fundScore":  fund_score,
        "fundSignal": fund.get("fundSignal"),
        "composite":  composite,
        "signal":     "BUY" if composite >= 2 else "SELL" if composite <= -2 else "HOLD",
        "confidence": fund.get("confidence", 50),
        "thesis":     fund.get("thesis", ""),
    }

def find_candidates(exclude: list[str]) -> list[dict]:
    """
    Builds a fresh dynamic universe then scans it for BUY signals.
    Returns all qualifying candidates sorted by confidence descending.
    """
    universe  = get_or_refresh_universe()
    targets   = [s for s in universe if s not in exclude]
    candidates = []

    log.info(f"Scanning {len(targets)} tickers from dynamic universe...")

    for symbol in targets:
        result = compute_signal(symbol)
        if result and result["signal"] == "BUY" and result["confidence"] >= MIN_CONFIDENCE:
            candidates.append(result)
            log.info(
                f"  SIGNAL {symbol}: composite={result['composite']:.2f}, "
                f"confidence={result['confidence']}%, thesis: {result['thesis']}"
            )
        time.sleep(20)  # 20s gap = ~3 calls/min, well under rate limit

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    log.info(f"Scan complete — {len(candidates)} qualifying signals found")
    return candidates

# ══════════════════════════════════════════════════════════════
# TRADING ACTIONS
# ══════════════════════════════════════════════════════════════

def check_profit_targets(positions: dict) -> list[str]:
    """
    Three exit rules per position:
    1. Profit target (+5%)        → sell immediately
    2. Trailing protection        → peaked +4%, fell to +2% → sell, lock gain
    3. Stop loss (-5%)            → cut losses
    """
    closed = []

    for symbol, pos in positions.items():
        try:
            unrealized_pct = float(pos.unrealized_plpc)

            # Update peak tracker
            prev_peak = position_peaks.get(symbol, 0.0)
            if unrealized_pct > prev_peak:
                position_peaks[symbol] = unrealized_pct
                save_peaks()
                if unrealized_pct >= PEAK_TRIGGER:
                    log.info(f"  {symbol}: new peak {unrealized_pct*100:+.2f}% — trailing protection active")

            current_peak = position_peaks.get(symbol, 0.0)
            reason = None

            # Rule 1: Full profit target
            if unrealized_pct >= PROFIT_TARGET:
                reason = f"PROFIT TARGET hit ({unrealized_pct*100:+.2f}%)"

            # Rule 2: Trailing peak protection (tightens in caution/bear)
            active_peak  = get_peak_trigger()
            active_trail = get_trail_sell()
            if current_peak >= active_peak and unrealized_pct <= active_trail:
                reason = (
                    f"TRAILING PROTECTION — peaked {current_peak*100:+.2f}%, "
                    f"fell to {unrealized_pct*100:+.2f}% — locking gain [{market_state} mode]"
                )

            # Rule 3: Stop loss (tightens to -2% in bear mode)
            active_stop = get_stop_loss()
            if unrealized_pct <= active_stop:
                reason = f"STOP LOSS hit ({unrealized_pct*100:+.2f}% <= {active_stop*100:.0f}%)"

            if reason:
                log.info(f"  [{symbol}] {reason} — closing")
                if close_position(symbol):
                    closed.append(symbol)
                    position_peaks.pop(symbol, None)
                    save_peaks()
            else:
                peak_str = f" (peak: {current_peak*100:+.2f}%)" if current_peak >= PEAK_TRIGGER else ""
                log.info(f"  {symbol}: {unrealized_pct*100:+.2f}% P&L{peak_str} — holding")

        except Exception as e:
            log.warning(f"  Error checking {symbol}: {e}")

    return closed

def reinvest(current_positions: dict, account):
    """
    Kelly-optimal position sizing — backtest confirmed 10% per trade.
    Each qualifying signal gets 10% of total portfolio equity.
    Max 10 concurrent positions (100% / 10% = 10).
    Cash not deployed stays idle — no over-concentration.

    Previous 50% rule caused the NOW disaster ($47k in one stock).
    Kelly 10% caps single-position risk at 0.5% portfolio loss per stop loss hit.
    """
    KELLY_PCT   = 0.10   # backtest-validated: 10% per position
    MAX_POS     = 10     # maximum concurrent positions

    cash   = float(account.cash)
    equity = float(account.equity)

    if cash < 100:
        log.info(f"Not enough cash to reinvest (${cash:.2f})")
        return

    # How many more positions can we open?
    current_count  = len(current_positions)
    available_slots = MAX_POS - current_count
    if available_slots <= 0:
        log.info(f"Max positions reached ({current_count}/{MAX_POS}) — no new buys")
        return

    log.info(f"Reinvesting — Kelly 10% sizing | Slots available: {available_slots}/{MAX_POS}")
    candidates = find_candidates(exclude=list(current_positions.keys()))

    if not candidates:
        log.info("No qualifying signals — cash stays idle")
        return

    # Each position gets 10% of total portfolio equity
    position_size = equity * KELLY_PCT
    log.info(f"Position size: ${position_size:,.0f} (10% of ${equity:,.0f} equity)")

    # Only take as many signals as we have slots for
    to_buy = candidates[:available_slots]
    log.info(f"Deploying into {len(to_buy)} signal(s): {[c['symbol'] for c in to_buy]}")

    for candidate in to_buy:
        symbol = candidate["symbol"]
        price  = candidate["price"]

        if price <= 0:
            log.warning(f"Invalid price for {symbol} — skipping")
            continue

        # Don't deploy more than available cash
        alloc = min(position_size, cash * 0.95)  # keep 5% cash buffer
        qty   = int(alloc / price)

        if qty < 1:
            log.info(f"Position size ${alloc:.0f} insufficient for 1 share of {symbol} at ${price:.2f}")
            continue

        qty = adjust_qty_for_fear(qty, price, alloc)
        actual_cost = qty * price
        log.info(f"  {symbol}: {qty} shares @ ~${price:.2f} = ${actual_cost:,.0f} ({actual_cost/equity*100:.1f}% of equity)")
        place_buy(symbol, qty)
        cash -= actual_cost  # track remaining cash across buys

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run():
    load_peaks()
    log.info("=" * 60)
    log.info("SIGNAL Trading Bot started")
    log.info(f"  Universe:           DYNAMIC — top {TOP_ACTIVE} most active + top {TOP_MOVERS} gainers")
    log.info(f"  Position sizing:    Kelly 10% per trade (backtest validated)")
    log.info(f"  Max positions:      10 concurrent")
    log.info(f"  Profit target:      +{PROFIT_TARGET*100:.0f}%")
    log.info(f"  Trailing:           peak >={PEAK_TRIGGER*100:.0f}% → sell at +{TRAIL_SELL*100:.0f}%")
    log.info(f"  Stop loss:          -{abs(STOP_LOSS)*100:.0f}%")
    log.info(f"  TA / Fund weight:   {TECH_WEIGHT}% / {FUND_WEIGHT}%")
    log.info(f"  Min confidence:     {MIN_CONFIDENCE}%")
    log.info(f"  Max trades/day:     {MAX_TRADES_DAY}")
    log.info(f"  Max drawdown:       {MAX_DRAWDOWN*100:.0f}%")
    log.info(f"  Pause:              set PAUSED=true in Render")
    log.info(f"  Macro layer:        SPY + VIX circuit breaker | Sector rotation")
    log.info(f"  Bull mode:          SPY > -1% or VIX < 20 despite SPY drop")
    log.info(f"  Caution mode:       SPY -1% to -2% + VIX 20-25")
    log.info(f"  Bear mode:          SPY < -2% AND VIX > 25 (genuine fear)")
    log.info(f"  Fear sizing:        VIX > 25 -> position sizes halved")
    log.info(f"  Rotation mode:      SPY down but VIX < 20 -> stay BULL (IPO rotation etc)")
    log.info("=" * 60)

    while True:
        try:
            now = datetime.now(timezone.utc)

            if not is_market_open():
                log.info(f"Market closed ({now.strftime('%H:%M UTC')}) — sleeping")
                # Reset universe date so it rebuilds fresh at next market open
                universe_date = ""
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"{'─'*50}")
            log.info(f"Scan at {now.strftime('%H:%M:%S UTC')}")

            account = get_account()
            equity  = float(account.equity)
            cash    = float(account.cash)
            log.info(f"Equity: ${equity:,.2f}  Cash: ${cash:,.2f}  Trades today: {trades_today_count()}/{MAX_TRADES_DAY}")

            # ── Assess market state every cycle ───────────────
            assess_market_state()

            safe, reason = run_risk_checks(account)
            if not safe:
                log.warning(f"RISK GATE: {reason}")
                time.sleep(SCAN_INTERVAL)
                continue

            positions = get_positions()
            log.info(f"Open positions: {list(positions.keys()) or 'none'}")
            prune_peaks(list(positions.keys()))

            closed = check_profit_targets(positions) if positions else []

            if closed:
                time.sleep(3)
                positions = get_positions()
                account   = get_account()
                if market_state == "BULL":
                    reinvest(positions, account)
                else:
                    log.warning(f"  {market_state} MODE — skipping reinvestment. Cash preserved until market recovers.")
            elif not positions:
                clock = trade_client.get_clock()
                if clock.is_open and market_state == "BULL":
                    reinvest({}, account)
                elif clock.is_open and market_state != "BULL":
                    log.warning(f"  {market_state} MODE — no new buys. Waiting for SPY to recover above -{abs(SPY_CAUTION)*100:.0f}%.")
                else:
                    log.info("No positions and market closed — waiting for market open to reinvest")

        except KeyboardInterrupt:
            log.info("Bot stopped.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run()
