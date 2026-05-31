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

# ── Risk thresholds ────────────────────────────────────────────
PROFIT_TARGET   =  0.05   # +5%  full profit target
PEAK_TRIGGER    =  0.04   # +4%  activate trailing protection
TRAIL_SELL      =  0.02   # +2%  sell if falls back here after peak
STOP_LOSS       = -0.05   # -5%  cut losses

# ── Runtime state ──────────────────────────────────────────────
trades_today:    dict[str, int]   = defaultdict(int)
circuit_breaker: bool             = False
starting_equity: float | None     = None
position_peaks:  dict[str, float] = {}
dynamic_universe: list[str]       = []
universe_date:   str              = ""   # date universe was last built

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

    try:
        # Most active by volume
        active = trade_client.get_most_active_stocks(top=TOP_ACTIVE)
        active_symbols = [s.symbol for s in active]
        symbols.update(active_symbols)
        log.info(f"Most active ({len(active_symbols)}): {active_symbols[:10]}...")
    except Exception as e:
        log.warning(f"Most active fetch failed: {e}")

    try:
        # Top gainers only — momentum aligned with strategy
        movers = trade_client.get_market_movers(top=TOP_MOVERS)
        if hasattr(movers, 'gainers') and movers.gainers:
            gainer_symbols = [s.symbol for s in movers.gainers]
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
        prompt = f"""You are a senior equity analyst. Research {symbol} using web search — recent earnings, revenue, margins, analyst consensus, news.

Technical context:
Price: ${ta.get('price', 0):.2f}, RSI: {ta.get('rsi', 50):.1f}, MACD Hist: {ta.get('macdHist', 0):.3f}
EMA20: ${ta.get('ema20', 0):.2f}, EMA50: ${ta.get('ema50', 0):.2f}, EMA200: ${ta.get('ema200', 0):.2f}
TA Signal: {ta.get('taSignal', 'HOLD')} (score {ta.get('taScore', 0):.1f}/10)

Return ONLY valid JSON (no markdown):
{{
  "fundSignal": "BUY or HOLD or SELL",
  "fundScore": <number -10 to 10>,
  "confidence": <number 0 to 100>,
  "thesis": "1-2 sentence summary"
}}"""

        response = ai_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        si, ei = text.find("{"), text.rfind("}")
        if si == -1:
            return None
        return json.loads(text[si:ei+1])
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
        time.sleep(8)  # respect Anthropic rate limits

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
                if unrealized_pct >= PEAK_TRIGGER:
                    log.info(f"  {symbol}: new peak {unrealized_pct*100:+.2f}% — trailing protection active")

            current_peak = position_peaks.get(symbol, 0.0)
            reason = None

            # Rule 1: Full profit target
            if unrealized_pct >= PROFIT_TARGET:
                reason = f"PROFIT TARGET hit ({unrealized_pct*100:+.2f}%)"

            # Rule 2: Trailing peak protection
            elif current_peak >= PEAK_TRIGGER and unrealized_pct <= TRAIL_SELL:
                reason = (
                    f"TRAILING PROTECTION — peaked {current_peak*100:+.2f}%, "
                    f"fell to {unrealized_pct*100:+.2f}% — locking gain"
                )

            # Rule 3: Stop loss
            elif unrealized_pct <= STOP_LOSS:
                reason = f"STOP LOSS hit ({unrealized_pct*100:+.2f}%)"

            if reason:
                log.info(f"  [{symbol}] {reason} — closing")
                if close_position(symbol):
                    closed.append(symbol)
                    position_peaks.pop(symbol, None)
            else:
                peak_str = f" (peak: {current_peak*100:+.2f}%)" if current_peak >= PEAK_TRIGGER else ""
                log.info(f"  {symbol}: {unrealized_pct*100:+.2f}% P&L{peak_str} — holding")

        except Exception as e:
            log.warning(f"  Error checking {symbol}: {e}")

    return closed

def reinvest(current_positions: dict, account):
    """
    Allocation rule:
    - 1 qualifying signal  → 100% of cash
    - 2+ qualifying signals → 50% into highest confidence, rest split equally
    """
    cash = float(account.cash)
    if cash < 100:
        log.info(f"Not enough cash to reinvest (${cash:.2f})")
        return

    log.info(f"Reinvesting ${cash:,.2f} — pulling fresh universe...")
    candidates = find_candidates(exclude=list(current_positions.keys()))

    if not candidates:
        log.info("No qualifying signals — cash stays idle")
        return

    # Build allocation
    allocations = {}
    if len(candidates) == 1:
        allocations[candidates[0]["symbol"]] = cash
        log.info(f"Single signal — 100% into {candidates[0]['symbol']}")
    else:
        top       = candidates[0]
        rest      = candidates[1:]
        top_cash  = cash * 0.50
        rest_cash = (cash * 0.50) / len(rest)
        allocations[top["symbol"]] = top_cash
        for c in rest:
            allocations[c["symbol"]] = rest_cash
        log.info(
            f"Split: {top['symbol']} 50% (${top_cash:,.0f}) | "
            f"{[c['symbol'] for c in rest]} share 50% (${rest_cash:,.0f} each)"
        )

    price_map = {c["symbol"]: c["price"] for c in candidates}

    for symbol, alloc in allocations.items():
        price = price_map.get(symbol, 0)
        if price <= 0:
            log.warning(f"Invalid price for {symbol} — skipping")
            continue
        qty = int(alloc / price)
        if qty < 1:
            log.info(f"Insufficient allocation (${alloc:.0f}) for 1 share of {symbol} at ${price:.2f}")
            continue
        log.info(f"  {symbol}: ${alloc:,.0f} → {qty} shares @ ~${price:.2f}")
        place_buy(symbol, qty)

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run():
    log.info("=" * 60)
    log.info("SIGNAL Trading Bot started")
    log.info(f"  Universe:           DYNAMIC — top {TOP_ACTIVE} most active + top {TOP_MOVERS} gainers")
    log.info(f"  Profit target:      +{PROFIT_TARGET*100:.0f}%")
    log.info(f"  Trailing:           peak >={PEAK_TRIGGER*100:.0f}% → sell at +{TRAIL_SELL*100:.0f}%")
    log.info(f"  Stop loss:          -{abs(STOP_LOSS)*100:.0f}%")
    log.info(f"  TA / Fund weight:   {TECH_WEIGHT}% / {FUND_WEIGHT}%")
    log.info(f"  Min confidence:     {MIN_CONFIDENCE}%")
    log.info(f"  Max trades/day:     {MAX_TRADES_DAY}")
    log.info(f"  Max drawdown:       {MAX_DRAWDOWN*100:.0f}%")
    log.info(f"  Pause:              set PAUSED=true in Render")
    log.info("=" * 60)

    while True:
        try:
            now = datetime.now(timezone.utc)

            if not is_market_open():
                log.info(f"Market closed ({now.strftime('%H:%M UTC')}) — sleeping")
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"{'─'*50}")
            log.info(f"Scan at {now.strftime('%H:%M:%S UTC')}")

            account = get_account()
            equity  = float(account.equity)
            cash    = float(account.cash)
            log.info(f"Equity: ${equity:,.2f}  Cash: ${cash:,.2f}  Trades today: {trades_today_count()}/{MAX_TRADES_DAY}")

            safe, reason = run_risk_checks(account)
            if not safe:
                log.warning(f"RISK GATE: {reason}")
                time.sleep(SCAN_INTERVAL)
                continue

            positions = get_positions()
            log.info(f"Open positions: {list(positions.keys()) or 'none'}")

            closed = check_profit_targets(positions) if positions else []

            if closed:
                time.sleep(3)
                positions = get_positions()
                account   = get_account()
                reinvest(positions, account)
            elif not positions:
                reinvest({}, account)

        except KeyboardInterrupt:
            log.info("Bot stopped.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run()
