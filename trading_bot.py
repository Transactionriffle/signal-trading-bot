"""
SIGNAL Trading Bot
==================
- Scans positions every 60 seconds during market hours
- Closes any position that hits +5% gain
- Reinvests freed cash into the next highest-signal stock
- Runs 24/7 on Render as a background worker

Risk Controls (borrowed from fail-safe design principles):
- PAUSED env var: set to "true" to instantly halt all trading
- MAX_TRADES_PER_DAY: hard cap on daily buy orders
- MAX_DRAWDOWN_PCT: circuit breaker — disarms if portfolio drops X% from start
- CIRCUIT_BREAKER_TRIGGERED: auto-set internally when drawdown fires

Requirements:
    pip install alpaca-py anthropic requests

Environment variables (set in Render):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    ALPACA_BASE_URL        (default: https://paper-api.alpaca.markets)
    ANTHROPIC_API_KEY
    CLOUDFLARE_WORKER      (your worker URL for technicals)
    TECH_WEIGHT            (default: 60)
    PROFIT_TARGET          (default: 0.05 = 5%)
    MIN_CONFIDENCE         (default: 80)
    MAX_TRADES_PER_DAY     (default: 10)
    MAX_DRAWDOWN_PCT       (default: 0.15 = 15%)
    PAUSED                 (set to "true" to pause — no restart needed)
    SCAN_UNIVERSE          (comma-separated tickers)
"""

import os
import json
import time
import logging
import requests
import anthropic
from datetime import datetime, timezone, date
from collections import defaultdict
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("signal-bot")

# ── Config from environment ────────────────────────────────────
ALPACA_KEY       = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET    = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL  = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ANTHROPIC_KEY    = os.environ["ANTHROPIC_API_KEY"]
WORKER_URL       = os.environ.get("CLOUDFLARE_WORKER", "https://winter-cake-6aae.dimitridesplace-65f.workers.dev")
TECH_WEIGHT      = int(os.environ.get("TECH_WEIGHT", "60"))
FUND_WEIGHT      = 100 - TECH_WEIGHT
PROFIT_TARGET    = float(os.environ.get("PROFIT_TARGET", "0.05"))
MIN_CONFIDENCE   = int(os.environ.get("MIN_CONFIDENCE", "80"))
MAX_TRADES_DAY   = int(os.environ.get("MAX_TRADES_PER_DAY", "10"))
MAX_DRAWDOWN     = float(os.environ.get("MAX_DRAWDOWN_PCT", "0.15"))
SCAN_INTERVAL    = 60
SCAN_UNIVERSE    = os.environ.get("SCAN_UNIVERSE",
    "NVDA,AAPL,MSFT,AMZN,META,GOOG,TSLA,AMD,AVGO,QCOM,ARM,PANW,ASML,SMCI,MU,"
    "ORCL,CRM,SNOW,PLTR,LLY,NVO,ABBV,UNH,JPM,GS,V,MA,XOM,CVX,NEE,ENPH,GE,CAT,UBER,SPOT"
).split(",")

# ── Runtime state ──────────────────────────────────────────────
trades_today:      dict[str, int] = defaultdict(int)   # date_str -> count
circuit_breaker:   bool           = False               # trips on max drawdown
starting_equity:   float | None   = None               # set on first run

# ── Clients ────────────────────────────────────────────────────
trade_client = TradingClient(
    api_key=ALPACA_KEY,
    secret_key=ALPACA_SECRET,
    paper=True,
    url_override=ALPACA_BASE_URL,
)
ai_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ══════════════════════════════════════════════════════════════
# RISK CONTROLS
# ══════════════════════════════════════════════════════════════

def is_paused() -> bool:
    """
    Hot-reload pause: change PAUSED=true in Render env vars.
    No restart needed — checked every scan cycle.
    """
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
    """
    Returns True (safe) or False (circuit breaker triggered).
    Compares current equity to starting equity.
    """
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
            f"exceeds limit {MAX_DRAWDOWN*100:.0f}%. All trading halted."
        )
        return False

    return True

def run_risk_checks(account) -> tuple[bool, str]:
    """
    Master risk gate. Returns (safe: bool, reason: str).
    All checks must pass for trading to proceed.
    """
    # 1. Manual pause
    if is_paused():
        return False, "PAUSED — manual pause active (set PAUSED=false in Render to resume)"

    # 2. Circuit breaker
    if not check_drawdown(account):
        return False, f"CIRCUIT BREAKER — drawdown exceeded {MAX_DRAWDOWN*100:.0f}% limit"

    # 3. Max trades per day
    count = trades_today_count()
    if count >= MAX_TRADES_DAY:
        return False, f"MAX TRADES REACHED — {count}/{MAX_TRADES_DAY} trades today"

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

def find_best_entry(exclude: list[str]) -> dict | None:
    candidates = []
    targets = [s for s in SCAN_UNIVERSE if s not in exclude]
    log.info(f"Scanning {len(targets)} tickers...")

    for symbol in targets:
        result = compute_signal(symbol)
        if result and result["signal"] == "BUY" and result["confidence"] >= MIN_CONFIDENCE:
            candidates.append(result)
            log.info(f"  {symbol}: composite={result['composite']:.2f}, confidence={result['confidence']}%")
        time.sleep(8)

    if not candidates:
        log.info("No qualifying BUY signals found.")
        return None

    best = max(candidates, key=lambda x: x["confidence"])
    log.info(f"Best entry: {best['symbol']} @ {best['confidence']}% confidence — {best['thesis']}")
    return best

# ══════════════════════════════════════════════════════════════
# TRADING ACTIONS
# ══════════════════════════════════════════════════════════════

def check_profit_targets(positions: dict) -> list[str]:
    closed = []
    for symbol, pos in positions.items():
        try:
            unrealized_pct = float(pos.unrealized_plpc)
            log.info(f"  {symbol}: {unrealized_pct*100:+.2f}% P&L")
            if unrealized_pct >= PROFIT_TARGET:
                log.info(f"  {symbol} hit +{PROFIT_TARGET*100:.0f}% target — closing")
                if close_position(symbol):
                    closed.append(symbol)
        except Exception as e:
            log.warning(f"  Error checking {symbol}: {e}")
    return closed

def reinvest(current_positions: dict, account):
    cash = float(account.cash)
    if cash < 100:
        log.info(f"Not enough cash to reinvest (${cash:.2f})")
        return

    log.info(f"Reinvesting ${cash:,.2f}...")
    best = find_best_entry(exclude=list(current_positions.keys()))
    if not best:
        log.info("No qualifying signal — cash stays idle")
        return

    symbol, price = best["symbol"], best["price"]
    if price <= 0:
        log.warning(f"Invalid price for {symbol}")
        return

    qty = int(cash / price)
    if qty < 1:
        log.info(f"Not enough cash to buy 1 share of {symbol} at ${price:.2f}")
        return

    log.info(f"Deploying ${cash:,.2f} → {qty}x {symbol} @ ~${price:.2f}")
    place_buy(symbol, qty)

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run():
    log.info("=" * 60)
    log.info("SIGNAL Trading Bot started")
    log.info(f"  Profit target:      +{PROFIT_TARGET*100:.0f}%")
    log.info(f"  TA / Fund weight:   {TECH_WEIGHT}% / {FUND_WEIGHT}%")
    log.info(f"  Min confidence:     {MIN_CONFIDENCE}%")
    log.info(f"  Max trades/day:     {MAX_TRADES_DAY}")
    log.info(f"  Max drawdown:       {MAX_DRAWDOWN*100:.0f}%")
    log.info(f"  Scan interval:      {SCAN_INTERVAL}s")
    log.info(f"  Universe:           {len(SCAN_UNIVERSE)} tickers")
    log.info(f"  Manual pause:       set PAUSED=true in Render to halt")
    log.info("=" * 60)

    while True:
        try:
            now = datetime.now(timezone.utc)

            # ── Market hours gate ──────────────────────────────
            if not is_market_open():
                log.info(f"Market closed ({now.strftime('%H:%M UTC')}) — sleeping")
                time.sleep(SCAN_INTERVAL)
                continue

            log.info(f"{'─'*50}")
            log.info(f"Scan at {now.strftime('%H:%M:%S UTC')}")

            # ── Get account ────────────────────────────────────
            account = get_account()
            equity  = float(account.equity)
            cash    = float(account.cash)
            log.info(f"Equity: ${equity:,.2f}  Cash: ${cash:,.2f}  Trades today: {trades_today_count()}/{MAX_TRADES_DAY}")

            # ── Risk gate ──────────────────────────────────────
            safe, reason = run_risk_checks(account)
            if not safe:
                log.warning(f"RISK GATE BLOCKED: {reason}")
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Get positions ──────────────────────────────────
            positions = get_positions()
            log.info(f"Open positions: {list(positions.keys()) or 'none'}")

            # ── Check profit targets ───────────────────────────
            closed = check_profit_targets(positions) if positions else []

            # ── Reinvest or find new entry ─────────────────────
            if closed:
                time.sleep(3)  # let fills settle
                positions = get_positions()
                account   = get_account()
                reinvest(positions, account)
            elif not positions:
                reinvest({}, account)

        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run()
