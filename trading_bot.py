"""
SIGNAL Trading Bot
==================
Architecture:
  Pre-market scan (Sunday 8pm or Monday 6am ET):
    → Build 50-ticker universe (35 curated + 15 dynamic gainers)
    → Run full fundamental + technical scan on all tickers
    → Cache ranked BUY list with scores
    → Deploy capital at market open from cached list

  During market hours:
    → Monitor positions every 60 seconds (zero Claude calls)
    → Exit rules: +5% profit, trailing +3%→+2.5%, -5% stop loss
    → Position closes → deploy into next ranked signal from cache
    → Only re-scan if cache exhausted (max once per 30 minutes)

  Every 90 days — Curated ticker audit:
    Criterion 1 — Volume & Liquidity: avg daily volume dropped?
                  Institutional money left = choppy, unpredictable → swap out
    Criterion 2 — Strategy Alignment: does stock still respect TA setups?
                  Regulatory change or market cap shift → remove
    Criterion 3 — Personal Performance: negative win rate on ticker
                  over last 3 months despite following rules → cut immediately

Environment variables (set in Render):
    ALPACA_API_KEY         (required)
    ALPACA_SECRET_KEY      (required)
    ANTHROPIC_API_KEY      (required)
    ALPACA_BASE_URL        (default: https://paper-api.alpaca.markets)
    CLOUDFLARE_WORKER
    TECH_WEIGHT            (default: 40 — 40% TA / 60% fundamental)
    MIN_CONFIDENCE         (default: 80)
    MAX_TRADES_PER_DAY     (default: 10)
    MAX_DRAWDOWN_PCT       (default: 0.15)
    PAUSED                 (set "true" to halt instantly)

Risk thresholds (backtest validated):
    PROFIT_TARGET = +5%    47% of trades hit this
    PEAK_TRIGGER  = +3%    activate trailing
    TRAIL_SELL    = +2.5%  sell if falls back here after peak
    STOP_LOSS     = -5%    cut losses (tightens to -2% in bear mode)

Macro layer:
    BULL  SPY >= -2% or VIX < 25       → normal trading
    BEAR  SPY < -2% AND VIX >= 25      → no new buys, stops tighten to -2%
    Fear  VIX >= 25                     → position sizes halved
    RS    stock % - SPY % boost/penalty per ticker
"""

import os
import json
import time
import logging
import requests
import anthropic
import threading
import websocket
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
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

# ── Config ─────────────────────────────────────────────────────
ALPACA_KEY      = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET   = os.environ["ALPACA_SECRET_KEY"]
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]
WORKER_URL      = os.environ.get("CLOUDFLARE_WORKER", "https://winter-cake-6aae.dimitridesplace-65f.workers.dev")
TECH_WEIGHT     = int(os.environ.get("TECH_WEIGHT", "40"))
FUND_WEIGHT     = 100 - TECH_WEIGHT
MIN_CONFIDENCE  = int(os.environ.get("MIN_CONFIDENCE", "85"))  # raised from 80% — filters weak signals like NFLX (82%)
MAX_TRADES_DAY  = int(os.environ.get("MAX_TRADES_PER_DAY", "10"))
MAX_DRAWDOWN    = float(os.environ.get("MAX_DRAWDOWN_PCT", "0.15"))
SCAN_INTERVAL   = 60
ET              = ZoneInfo("America/New_York")

# ── Risk thresholds ────────────────────────────────────────────
PROFIT_TARGET   =  0.05    # +5%   sell immediately
PEAK_TRIGGER    =  0.03    # +3%   activate trailing protection
TRAIL_SELL      =  0.025   # +2.5% sell if falls back here after peak
STOP_LOSS       = -0.05    # -5%   hard stop (before breakeven activates)
BREAKEVEN_TRIGGER = 0.01   # +1%   once hit, stop shifts to +0.5%
BREAKEVEN_STOP    = 0.005  # +0.5% minimum locked-in gain after breakeven

# ── Macro thresholds ───────────────────────────────────────────
SPY_BEAR        = -0.02
VIXY_FEAR       =  0.05
SECTOR_WEAK     = -0.015
SECTOR_ETFS     = {
    "tech": "XLK", "healthcare": "XLV", "financials": "XLF",
    "energy": "XLE", "utilities": "XLU", "consumer": "XLY",
    "industrials": "XLI", "materials": "XLB",
}

# ── Curated universe (35 tickers — sector diversified) ────────
# Rebuilt June 2026 with proper sector diversification and mid-cap exposure
# Audit criteria: Volume/liquidity, TA alignment, personal win rate
# Max 6 tickers per sector — no single sector dominates
# Mix of large-cap stability + mid-cap growth ($15B-$80B)
CURATED_TICKERS = [
    # ── AI / Semiconductors (5) — core AI infrastructure ─────
    # Large-cap anchors with proven +5% exits
    "NVDA",   # $5.2T — AI GPU monopoly, multiple exits
    "AVGO",   # $1.9T — networking chips, AI custom silicon
    "TSM",    # $2.2T — manufactures everything, 88% confidence score
    "MU",     # $1.1T — HBM memory, multiple +5% exits
    "AMD",    # $806B — data center CPU/GPU, strong momentum

    # ── Mega-cap Tech (4) — liquid, consistent signals ────────
    "GOOG",   # $4.4T — Cloud +63%, AI search dominance
    "META",   # $1.5T — 33% revenue growth, PE 21 below average
    "MSFT",   # $2.9T — Azure +40%, enterprise AI
    "AMZN",   # $2.6T — AWS reaccelerating, retail margins expanding

    # ── Financials (4) — rate sensitive, macro diversifier ───
    "JPM",    # $876B — consistent 85% confidence scores
    "V",      # $617B — payment rails, recession resistant, multiple +5% exits
    "MA",     # $510B — same, multiple +5% exits
    "GS",     # $210B — trading revenue, M&A advisory cycle
    "BAC",    # $350B — rate sensitive, improving ROE

    # ── Healthcare (4) — defensive + biotech catalyst ────────
    "LLY",    # $1.04T — GLP-1 monopoly, multiple +5% exits
    "UNH",    # $440B — managed care, raised guidance
    "ABBV",   # $370B — Skyrizi/Rinvoq growth replacing Humira
    "ISRG",   # $200B — surgical robotics, recurring revenue

    # ── Energy (4) — best performing sector 2026 YTD +19.9% ─
    "XOM",    # $613B — Iran war premium, strong FCF
    "CVX",    # $270B — integrated major, dividend growth
    "COP",    # $110B — pure upstream E&P, leveraged to oil price
    "SLB",    # $55B — oilfield services, AI drilling tech (mid-cap)

    # ── Industrials (4) — AI infrastructure buildout play ────
    "GEV",    # $90B — GE Vernova, gas turbines + nuclear (mid-cap)
    "CAT",    # $185B — construction equipment, data center buildout
    "RTX",    # $190B — defense/aerospace, Iran war spending
    "HON",    # $130B — industrial automation, building tech

    # ── Consumer / Distribution (4) — FMCG + logistics ──────
    "COST",   # $440B — membership model, recession proof
    "WMT",    # $780B — supply chain dominance, grocery
    "MCD",    # $210B — global franchise, pricing power
    "FDX",    # $65B — logistics, e-commerce backbone (mid-cap)

    # ── Mid-cap Growth (6) — less efficient, more alpha ──────
    "PLTR",   # $450B — AI software, government + enterprise
    "CRWD",   # $120B — cybersecurity market leader
    "SPOT",   # $80B — profitability inflection, subscriber growth
    # NFLX removed — Jun 15 trade hit -4%, weak signal (composite 2.50, min threshold)
    "NUVL",   # $10B  — acquisition target, biotech catalyst
    "DECK",   # $22B  — UGG/HOKA, consistent earnings beats (mid-cap)
    "IOT",    # $19B  — Samsara, Connected Operations platform, 30% ARR growth,
              #          3rd consecutive GAAP profitable quarter, composite 7.90 Jul 1
              #          Physical switching costs + AI monetisation upside
              #          Next earnings: Sep 3, 2026
]  # 37 tickers

# Sector breakdown:
# Semis: 5 (14%) | Tech: 4 (11%) | Financials: 5 (14%)
# Healthcare: 4 (11%) | Energy: 4 (11%) | Industrials: 4 (11%)
# Consumer/Distribution: 4 (11%) | Mid-cap Growth: 6 (17%)
# Large-cap (>$100B): 26 | Mid-cap ($15B-$100B): 10

# ── Sector map — used for weak sector filtering ────────────────
SECTOR_MAP: dict[str, str] = {
    # Semis
    "NVDA":"tech","AVGO":"tech","TSM":"tech","MU":"tech","AMD":"tech",
    # Mega-cap Tech
    "GOOG":"tech","META":"tech","MSFT":"tech","AMZN":"tech",
    "AAPL":"tech","TSLA":"tech","ORCL":"tech",
    # Software / Cyber
    "PLTR":"tech","CRWD":"tech","PANW":"tech","NET":"tech",
    "CRM":"tech","SNOW":"tech",
    # Financials
    "JPM":"financials","V":"financials","MA":"financials",
    "GS":"financials","BAC":"financials","MS":"financials","BLK":"financials",
    # Healthcare
    "LLY":"healthcare","UNH":"healthcare","ABBV":"healthcare",
    "ISRG":"healthcare","NUVL":"healthcare",
    # Energy
    "XOM":"energy","CVX":"energy","COP":"energy","SLB":"energy",
    # Industrials
    "GEV":"industrials","CAT":"industrials","RTX":"industrials","HON":"industrials",
    "IOT":"industrials",  # Samsara — Connected Operations, fleet/physical asset management
    # Consumer / Distribution
    "COST":"consumer","WMT":"consumer","MCD":"consumer","FDX":"consumer",
    # Mid-cap
    "SPOT":"consumer","NFLX":"consumer","UBER":"consumer",
    "DECK":"consumer","ARM":"tech","MRVL":"tech","QCOM":"tech",
    "ASML":"tech","ADBE":"tech","INTC":"tech",
}

# ── ETF exclusions — never trade these ────────────────────────
ETF_EXCLUSIONS = {
    "SPY","QQQ","IWM","EEM","VOO","VTI","VEA","VWO","IVV","DIA",
    "XLF","XLK","XLE","XLV","XLU","XLY","XLI","XLB","XLC","XLRE",
    "GLD","SLV","TLT","HYG","LQD","AGG","BND",
    "SOXS","SOXL","TQQQ","SQQQ","UVIX","UVXY","VIXY","SPDN","TZA",
    "IBIT","BITO","MSTU","TSLL","DRIP","QID","NVD","TSLS","NVDL",
    "NVDS","LABU","LABD","TECL","TECS","FAS","FAZ","UPRO","SPXU",
    "UDOW","SDOW","BOIL","KOLD","UGAZ","DGAZ",
    "DRAM","SMH","SOXX","XSD","PSI","FTXL","SOXQ",
    # Tradr 2X leveraged ETFs — new series, not recognisable by ticker name
    "CRDU","NVDU","AAPU","AMZU","MSFU","GOGU","METAU","TSLU",
    "AMZD","NVDD","AAPD","MSFD","GOGD","METAD","CRDL",
}

# Suffixes that identify leveraged/inverse ETFs not in the exclusion list
# Used as a fallback check in is_likely_etf()
ETF_NAME_KEYWORDS = [
    "2x long", "2x short", "3x long", "3x short",
    "leveraged", "inverse", "ultra", "bear", "bull etf",
    "daily etf", "proshares", "direxion", "tradr",
]

def is_likely_etf(symbol: str) -> bool:
    """
    Secondary ETF check using Alpaca asset data.
    Catches new leveraged ETFs not yet in ETF_EXCLUSIONS.
    Called once per ticker, result cached in fund_cache.
    """
    cache_key = f"etf_check_{symbol}"
    if cache_key in fund_cache:
        _, result = fund_cache[cache_key]
        return result

    try:
        asset = trade_client.get_asset(symbol)
        name  = (asset.name or "").lower()
        is_etf = any(kw in name for kw in ETF_NAME_KEYWORDS)
        if is_etf:
            log.warning(f"  {symbol}: identified as ETF/leveraged product ('{asset.name}') — blocking")
        fund_cache[cache_key] = (time.time(), is_etf)
        return is_etf
    except Exception:
        fund_cache[cache_key] = (time.time(), False)
        return False

# ── Runtime state ──────────────────────────────────────────────
trades_today:       dict[str, int]   = defaultdict(int)
circuit_breaker:    bool             = False
starting_equity:    float | None     = None
position_peaks:     dict[str, float] = {}
market_state:       str              = "BULL"
fear_active:        bool             = False
weak_sectors:       set              = set()
spy_change:         float            = 0.0
fund_cache:         dict             = {}        # {symbol: (timestamp, result)}
signal_cache:       list             = []        # ranked BUY signals from pre-market scan
signal_cache_time:  float            = 0.0       # when cache was built
signal_cache_date:  str              = ""        # date of last scan
last_rescan_time:   float            = 0.0       # last emergency rescan

# ── Re-entry cooldown tracking ─────────────────────────────────
# Option 1: Post-profit cooldown — 4hr block after +5% exit
# Option 3: Re-entry price gate — only re-enter if price pulled back 2%
# Option 5: Stop loss cooldown — 24hr block after -5% stop exit
reentry_cooldown:   dict[str, dict]  = {}
# {symbol: {"type": "profit"|"stop", "time": timestamp, "exit_price": float}}

PROFIT_COOLDOWN_SECS = 14400   # 4 hours after profit target exit
STOP_COOLDOWN_SECS   = 86400   # 24 hours after stop loss exit
PRICE_GATE_PCT       = 0.02    # must pull back 2% from exit price to re-enter

# ── News WebSocket state ───────────────────────────────────────
news_triggered:     dict[str, float] = {}        # {symbol: timestamp} — news-triggered tickers
news_queue:         list             = []         # pending news signals to process
NEWS_COOLDOWN       = 3600                        # 1 hour before same ticker triggers again

# High-value news keywords that warrant immediate signal scoring
NEWS_BULLISH_KEYWORDS = [
    "acqui", "merger", "buyout", "takeover",       # M&A
    "beats", "beat", "exceeded", "surpassed",      # earnings beats
    "raises guidance", "raised guidance",           # guidance upgrade
    "fda approved", "fda approval",                # FDA
    "partnership", "contract", "deal",             # business wins
    "buyback", "repurchase",                       # shareholder returns
    "upgrade", "outperform", "overweight",         # analyst upgrades
]
NEWS_BEARISH_KEYWORDS = [
    "miss", "missed", "below expectations",        # earnings miss
    "lowered guidance", "cuts guidance",           # guidance cut
    "investigation", "lawsuit", "sec charges",     # legal issues
    "recall", "safety concern",                    # product issues
    "downgrade", "underperform", "sell rating",   # analyst downgrades
]

PEAKS_FILE   = "/tmp/position_peaks.json"
JOURNAL_FILE = "/tmp/trade_journal.json"

# ══════════════════════════════════════════════════════════════
# NEWS WEBSOCKET LAYER
# ══════════════════════════════════════════════════════════════

def classify_news(headline: str, summary: str) -> str:
    """
    Classifies news as BULLISH, BEARISH, or NEUTRAL.
    Returns classification string.
    """
    text = (headline + " " + summary).lower()
    for kw in NEWS_BULLISH_KEYWORDS:
        if kw in text:
            return "BULLISH"
    for kw in NEWS_BEARISH_KEYWORDS:
        if kw in text:
            return "BEARISH"
    return "NEUTRAL"

def extract_tickers(symbols: list, headline: str) -> list:
    """
    Returns tickers from the news article that are in our universe.
    Filters to curated tickers + news-triggered universe.
    """
    universe = set(CURATED_TICKERS) | set(news_triggered.keys())
    matched  = [s for s in symbols if s in universe and s not in ETF_EXCLUSIONS]
    return matched

def on_news_message(ws, message):
    """
    Handles incoming news from Alpaca WebSocket stream.
    Filters for high-value events on universe tickers.
    Adds to news_queue for main thread to process.
    """
    global news_triggered, news_queue
    try:
        data = json.loads(message)
        if not isinstance(data, list):
            data = [data]

        for article in data:
            if article.get("T") != "n":  # only news type messages
                continue

            headline = article.get("headline", "")
            summary  = article.get("summary", "")
            symbols  = article.get("symbols", [])

            # Filter to universe tickers
            matched = extract_tickers(symbols, headline)
            if not matched:
                continue

            # Classify sentiment
            sentiment = classify_news(headline, summary)
            if sentiment == "NEUTRAL":
                continue

            # Check cooldown — don't re-trigger same ticker within 1 hour
            now = time.time()
            for symbol in matched:
                last_trigger = news_triggered.get(symbol, 0)
                if now - last_trigger < NEWS_COOLDOWN:
                    continue

                news_triggered[symbol] = now
                news_queue.append({
                    "symbol":    symbol,
                    "headline":  headline,
                    "sentiment": sentiment,
                    "timestamp": now,
                })
                log.info(
                    f"📰 NEWS TRIGGER [{sentiment}] {symbol}: {headline[:80]}..."
                )

    except Exception as e:
        log.warning(f"News message error: {e}")

def on_news_open(ws):
    log.info("📰 News WebSocket connected — subscribing to all news")
    ws.send(json.dumps({
        "action":  "auth",
        "key":     ALPACA_KEY,
        "secret":  ALPACA_SECRET,
    }))
    ws.send(json.dumps({
        "action": "subscribe",
        "news":   ["*"],  # subscribe to all news
    }))

def on_news_error(ws, error):
    log.warning(f"News WebSocket error: {error}")

def on_news_close(ws, close_status_code, close_msg):
    log.warning(f"News WebSocket closed: {close_status_code} {close_msg}")

def start_news_stream():
    """
    Starts news WebSocket in a background daemon thread.
    Automatically reconnects on disconnect.
    """
    def run():
        while True:
            try:
                log.info("📰 Starting news WebSocket stream...")
                ws = websocket.WebSocketApp(
                    "wss://stream.data.alpaca.markets/v1beta1/news",
                    on_open    = on_news_open,
                    on_message = on_news_message,
                    on_error   = on_news_error,
                    on_close   = on_news_close,
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                log.warning(f"News stream crashed: {e} — reconnecting in 30s")
            time.sleep(30)  # reconnect delay

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    log.info("📰 News WebSocket thread started")

def process_news_queue(positions: dict, account) -> bool:
    """
    Processes pending news signals from the queue.
    Called from main loop every cycle.
    Returns True if any news-triggered trade was placed.
    """
    global news_queue, signal_cache

    if not news_queue:
        return False

    traded = False
    to_process = news_queue.copy()
    news_queue.clear()

    for item in to_process:
        symbol    = item["symbol"]
        headline  = item["headline"]
        sentiment = item["sentiment"]

        # Skip if already in a position
        if symbol in positions:
            log.info(f"📰 {symbol} already held — skipping news trigger")
            continue

        # Skip if market is not open
        if not is_market_open():
            log.info(f"📰 {symbol} news trigger queued — market closed")
            news_queue.append(item)  # requeue for when market opens
            continue

        if market_state == "BEAR":
            log.info(f"📰 {symbol} news trigger skipped — BEAR mode")
            continue

        log.info(f"📰 Processing news trigger for {symbol}: {headline[:60]}...")

        # Score the ticker immediately
        spy_chg = spy_change or 0.0

        # Invalidate fund cache for this ticker — news changes the score
        if symbol in fund_cache:
            del fund_cache[symbol]

        result = compute_signal(symbol, spy_chg)

        if result and result["signal"] == "BUY" and result["confidence"] >= MIN_CONFIDENCE and result["composite"] >= 3.0:
            log.info(
                f"📰 NEWS BUY [{sentiment}] {symbol}: "
                f"composite={result['composite']:.2f}, confidence={result['confidence']}% "
                f"— {result['thesis']}"
            )
            # Insert at top of signal cache — but check for existing entry first
            already_cached = any(s["symbol"] == symbol for s in signal_cache)
            if already_cached:
                # Update existing entry with fresh score rather than duplicating
                signal_cache[:] = [s for s in signal_cache if s["symbol"] != symbol]
                log.info(f"📰 {symbol}: updated existing cache entry with fresh news score")
            signal_cache.insert(0, result)
            traded = True
        elif sentiment == "BULLISH" and result:
            log.info(
                f"📰 {symbol} news bullish but below threshold "
                f"(composite={result['composite']:.2f}, confidence={result['confidence']}%)"
            )
        else:
            log.info(f"📰 {symbol} news scored HOLD/SELL — no action")

    return traded
def load_journal() -> dict:
    try:
        with open(JOURNAL_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def record_trade(symbol: str, pnl_pct: float, exit_reason: str):
    """Record every trade outcome for 90-day audit."""
    journal = load_journal()
    if symbol not in journal:
        journal[symbol] = []
    journal[symbol].append({
        "date":   datetime.now(ET).strftime("%Y-%m-%d"),
        "pnl":    round(pnl_pct * 100, 2),
        "reason": exit_reason,
    })
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(journal, f)
    except Exception as e:
        log.warning(f"Failed to save journal: {e}")

def run_90_day_audit():
    """
    Audits curated tickers every 90 days against three criteria:
    1. Volume & Liquidity — has institutional volume dropped?
    2. Strategy Alignment — does stock still respect TA setups?
    3. Personal Performance — negative win rate over last 90 days?
    """
    journal  = load_journal()
    cutoff   = datetime.now(ET) - timedelta(days=90)
    flagged  = []

    log.info("=" * 60)
    log.info("90-DAY CURATED TICKER AUDIT")
    log.info("=" * 60)

    for symbol in CURATED_TICKERS:
        issues = []

        # Criterion 3: Personal Performance (from journal)
        trades = journal.get(symbol, [])
        recent = [t for t in trades if datetime.strptime(t["date"], "%Y-%m-%d").replace(tzinfo=ET) >= cutoff]
        if len(recent) >= 5:  # need at least 5 trades for meaningful data
            wins     = sum(1 for t in recent if t["pnl"] > 0)
            win_rate = wins / len(recent)
            avg_pnl  = sum(t["pnl"] for t in recent) / len(recent)
            if win_rate < 0.40:  # below 40% win rate
                issues.append(f"Poor win rate {win_rate*100:.0f}% over {len(recent)} trades (avg {avg_pnl:.1f}%)")

        # Criterion 1: Volume & Liquidity
        try:
            r = requests.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
                params={"timeframe": "1Day", "limit": 20, "feed": "sip"},
                timeout=10
            )
            if r.ok:
                bars = r.json().get("bars", [])
                if bars:
                    avg_vol = sum(b["v"] for b in bars) / len(bars)
                    if avg_vol < 500_000:
                        issues.append(f"Low volume {avg_vol/1e6:.1f}M avg — institutional money may have left")
        except Exception:
            pass

        if issues:
            flagged.append((symbol, issues))
            log.warning(f"  ⚠️  {symbol}: {' | '.join(issues)}")
        else:
            log.info(f"  ✅ {symbol}: passes all criteria")

    log.info("=" * 60)
    if flagged:
        log.warning(f"AUDIT COMPLETE — {len(flagged)} tickers flagged for review:")
        for sym, issues in flagged:
            log.warning(f"  → {sym}: {issues[0]}")
        log.warning("Update CURATED_TICKERS in the bot to remove flagged tickers")
    else:
        log.info("AUDIT COMPLETE — all tickers pass. No changes needed.")
    log.info("=" * 60)

# ── Peak persistence ───────────────────────────────────────────
def save_peaks():
    try:
        with open(PEAKS_FILE, "w") as f:
            json.dump(position_peaks, f)
    except Exception as e:
        log.warning(f"Failed to save peaks: {e}")

def load_peaks():
    global position_peaks
    try:
        with open(PEAKS_FILE) as f:
            position_peaks = json.load(f)
        if position_peaks:
            log.info(f"Loaded peaks: {position_peaks}")
    except FileNotFoundError:
        position_peaks = {}
    except Exception as e:
        log.warning(f"Failed to load peaks: {e}")
        position_peaks = {}

def prune_peaks(active_symbols: list):
    stale = [s for s in position_peaks if s not in active_symbols]
    for s in stale:
        del position_peaks[s]
    if stale:
        save_peaks()

# ── Sector concentration cap ───────────────────────────────────
# Backtest validated: max 2 per sector outperforms max 3
# Apr 26 +$1,196 better, May 26 +$1,914 better vs uncapped
# Prevents NVDA×3 type concentration losses regardless of signal quality
MAX_SECTOR_POSITIONS = 2
SECTOR_MAP = {
    "NVDA":"semis","AMD":"semis","AVGO":"semis","ASML":"semis","MU":"semis",
    "TSM":"semis","QCOM":"semis","ARM":"semis","MRVL":"semis",
    "AAPL":"tech","MSFT":"tech","GOOG":"tech","META":"tech",
    "AMZN":"tech","TSLA":"tech","ORCL":"tech",
    "PLTR":"software","PANW":"cyber","CRWD":"cyber","NET":"cyber",
    "CRM":"software","SNOW":"software",
    "JPM":"financials","V":"financials","MA":"financials",
    "GS":"financials","BAC":"financials","BLK":"financials",
    "LLY":"healthcare","UNH":"healthcare","ABBV":"healthcare",
    "ISRG":"healthcare","NUVL":"healthcare",
    "XOM":"energy","CVX":"energy","COP":"energy","SLB":"energy",
    "GEV":"industrials","CAT":"industrials","RTX":"industrials","HON":"industrials",
    "COST":"consumer","WMT":"consumer","MCD":"consumer","FDX":"consumer",
    "DECK":"consumer","SPOT":"media","NFLX":"media","UBER":"consumer",
}

# ── Alpaca client ──────────────────────────────────────────────
trade_client = TradingClient(
    api_key=ALPACA_KEY, secret_key=ALPACA_SECRET,
    paper=True, url_override=ALPACA_BASE_URL,
)
ai_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ══════════════════════════════════════════════════════════════
# MACRO LAYER
# ══════════════════════════════════════════════════════════════

def get_quote_change(symbol: str) -> float | None:
    try:
        headers = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
        r = requests.get(
            "https://data.alpaca.markets/v2/stocks/snapshots",
            headers=headers,
            params={"symbols": symbol, "feed": "sip"},
            timeout=15
        )
        if not r.ok:
            return None
        snap  = r.json().get(symbol, {})
        daily = snap.get("dailyBar", {})
        prev  = snap.get("prevDailyBar", {})
        if daily and prev and prev.get("c", 0) > 0:
            return (daily.get("c", 0) - prev.get("c", 0)) / prev.get("c", 0)
        latest = snap.get("latestTrade", {})
        if latest and prev and prev.get("c", 0) > 0:
            return (latest.get("p", 0) - prev.get("c", 0)) / prev.get("c", 0)
        return None
    except Exception as e:
        log.warning(f"Quote fetch failed for {symbol}: {e}")
        return None

def get_vix_level() -> float | None:
    try:
        now     = int(time.time())
        from_ts = now - 86400 * 5
        url     = f"{WORKER_URL}/yahoofinance/chart/%5EVIX?interval=1d&period1={from_ts}&period2={now}"
        r       = requests.get(url, timeout=20)
        r.raise_for_status()
        chart  = r.json().get("chart", {}).get("result", [{}])[0]
        closes = chart.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        closes = [c for c in closes if c is not None]
        if closes:
            log.info(f"VIX level: {closes[-1]:.1f}")
            return closes[-1]
        return None
    except Exception as e:
        log.warning(f"VIX fetch failed: {e}")
        return None

def assess_market_state():
    global market_state, fear_active, weak_sectors, spy_change

    vix_now  = get_vix_level()
    vixy_chg = get_quote_change("VIXY")

    if vix_now is not None:
        fear_active = vix_now >= 25
        level = "PANIC" if vix_now >= 35 else "FEAR" if vix_now >= 25 else "uncertainty" if vix_now >= 20 else "calm"
        if fear_active:
            log.warning(f"FEAR ACTIVE — VIX {vix_now:.1f} ({level}). Position sizes halved.")
        else:
            log.info(f"VIX {vix_now:.1f} — {level}. Normal sizing.")
    elif vixy_chg is not None:
        fear_active = vixy_chg >= VIXY_FEAR
        vix_now     = None
    else:
        fear_active = False
        vix_now     = None

    spy_chg = get_quote_change("SPY")
    if spy_chg is not None:
        spy_change = spy_chg
        if spy_chg <= SPY_BEAR and vix_now and vix_now >= 25:
            market_state = "BEAR"
            log.warning(f"BEAR MODE — SPY {spy_chg*100:.2f}% + VIX {vix_now:.1f}. Cash preserved.")
        else:
            market_state = "BULL"
            tag = "(rotation)" if spy_chg <= SPY_BEAR else "(mild weakness)" if spy_chg <= -0.01 else ""
            log.info(f"BULL MODE {tag} — SPY {spy_chg*100:.2f}%.")
    else:
        log.warning("Could not fetch SPY — market state unchanged")

    weak_sectors = set()
    for sector, etf in SECTOR_ETFS.items():
        chg = get_quote_change(etf)
        if chg is not None and chg <= SECTOR_WEAK:
            weak_sectors.add(sector)
            log.info(f"  Weak sector: {sector.upper()} ({etf} {chg*100:.2f}%) — avoiding")

    log.info(f"Market state: {market_state} | Fear: {fear_active} | Weak sectors: {weak_sectors or 'none'}")

def get_stop_loss() -> float:
    return -0.02 if market_state == "BEAR" else STOP_LOSS

def adjust_qty_for_fear(qty: int, price: float, alloc: float) -> int:
    if fear_active and qty > 1:
        adjusted = max(1, int((alloc * 0.5) / price))
        log.info(f"  Fear active — size halved: {qty} → {adjusted}")
        return adjusted
    return qty

# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════

def fetch_technicals(symbol: str) -> dict | None:
    try:
        r = requests.get(f"{WORKER_URL}/technicals/{symbol}", timeout=15)
        r.raise_for_status()
        data = r.json()
        return None if "error" in data else data
    except Exception as e:
        log.warning(f"Technicals fetch failed for {symbol}: {e}")
        return None

def fetch_fundamental(symbol: str, ta: dict) -> dict | None:
    global fund_cache
    # 2-hour cache — avoid re-researching same stock multiple times
    if symbol in fund_cache:
        cached_time, cached_result = fund_cache[symbol]
        if time.time() - cached_time < 7200:
            return cached_result

    try:
        price  = ta.get("price", 0)
        rsi    = ta.get("rsi", 50)
        signal = ta.get("taSignal", "HOLD")
        prompt = (
            f"Research {symbol} stock. Price ${price:.0f}, RSI {rsi:.0f}, TA {signal}. "
            "Rate fundamentals. Return ONLY this JSON: "
            '{"fundSignal":"BUY","fundScore":7,"confidence":82,"thesis":"one sentence"} '
            "fundScore -10 to +10. confidence 0-100."
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
        result["fundScore"]  = max(-10, min(10, float(result.get("fundScore", 0))))
        result["confidence"] = max(0,   min(100, float(result.get("confidence", 50))))
        fund_cache[symbol]   = (time.time(), result)
        return result
    except Exception as e:
        log.warning(f"Fundamental fetch failed for {symbol}: {e}")
        return None

def check_earnings_proximity(symbol: str, today: str) -> tuple[bool, str]:
    """
    Checks if a ticker has earnings within 3 days using Claude web search.
    Returns (has_earnings_soon, earnings_date_or_empty).
    Uses a separate lightweight cache to avoid repeated checks.
    """
    # Lightweight earnings cache — 24 hours
    cache_key = f"earnings_{symbol}"
    if cache_key in fund_cache:
        cached_time, cached_result = fund_cache[cache_key]
        if time.time() - cached_time < 86400:
            return cached_result

    try:
        prompt = (
            f"Does {symbol} have earnings announcement within the next 3 days from {today}? "
            "Return ONLY this JSON with NO other text: "
            '{"earnings_within_3_days":false,"earnings_date":null} '
            "Set earnings_within_3_days to true only if earnings are confirmed within 3 calendar days."
        )
        response = ai_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=60,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        si, ei = text.find("{"), text.rfind("}")
        if si == -1:
            fund_cache[cache_key] = (time.time(), (False, ""))
            return False, ""
        result = json.loads(text[si:ei+1])
        has_earnings = bool(result.get("earnings_within_3_days", False))
        earnings_date = result.get("earnings_date") or ""
        fund_cache[cache_key] = (time.time(), (has_earnings, earnings_date))
        return has_earnings, earnings_date
    except Exception as e:
        log.warning(f"Earnings check failed for {symbol}: {e}")
        return False, ""

def compute_signal(symbol: str, spy_chg: float = 0.0) -> dict | None:
    # Skip known ETFs
    if symbol in ETF_EXCLUSIONS:
        return None

    # Secondary ETF check — catches new leveraged ETFs not in exclusion list (e.g. CRDU)
    if is_likely_etf(symbol):
        ETF_EXCLUSIONS.add(symbol)  # add to exclusion list for this session
        return None

    ta = fetch_technicals(symbol)

    ipo_mode = False
    if not ta:
        if symbol in ETF_EXCLUSIONS:
            return None
        log.info(f"  {symbol}: no technicals — IPO mode")
        ta = {"price":0,"taScore":0,"taSignal":"HOLD","rsi":50,
              "macdHist":0,"ema20":0,"ema50":0,"ema200":0,
              "pct1d":0,"pct5d":0,"volRatio":1}
        ipo_mode = True

    # ── Earnings proximity check — applies to ALL tickers (curated + dynamic) ──
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    has_earnings, earnings_date = check_earnings_proximity(symbol, today_str)
    if has_earnings:
        log.warning(
            f"  {symbol}: earnings within 3 days ({earnings_date}) — "
            f"skipping to avoid pre-earnings selloff risk"
        )
        return None

    fund = fetch_fundamental(symbol, ta)
    if not fund:
        return None

    ta_score   = ta.get("taScore", 0)
    fund_score = fund.get("fundScore", 0)
    composite  = ta_score * (TECH_WEIGHT / 100) + fund_score * (FUND_WEIGHT / 100)

    # Relative strength vs SPY
    stock_pct = ta.get("pct1d", 0) / 100
    rel_str   = stock_pct - spy_chg
    if rel_str >= 0.02:
        rs_boost, rs_label = 1.5, f"STRONG RS +{rel_str*100:.1f}%"
    elif rel_str >= 0.01:
        rs_boost, rs_label = 0.75, f"GOOD RS +{rel_str*100:.1f}%"
    elif rel_str >= 0.0:
        rs_boost, rs_label = 0.0, f"NEUTRAL RS {rel_str*100:.1f}%"
    elif rel_str >= -0.01:
        rs_boost, rs_label = -0.5, f"WEAK RS {rel_str*100:.1f}%"
    else:
        rs_boost, rs_label = -1.5, f"POOR RS {rel_str*100:.1f}%"

    composite_adj = composite + rs_boost
    signal        = "BUY" if composite_adj >= 2 else "SELL" if composite_adj <= -2 else "HOLD"

    if rs_boost != 0:
        log.info(f"  {symbol} RS: {rs_label} → composite {composite:.2f} → {composite_adj:.2f}")

    # ATR as % of price — used for adaptive profit targets in deploy_from_cache
    # Cloudflare worker may return atr14 (14-day ATR in $ terms); convert to %
    atr_raw = ta.get("atr14", 0) or ta.get("atr", 0)
    atr_pct = (atr_raw / ta.get("price", 1)) if atr_raw and ta.get("price", 0) > 0 else 0.02

    return {
        "symbol":     symbol,
        "price":      ta.get("price", 0),
        "taScore":    ta_score,
        "fundScore":  fund_score,
        "composite":  composite_adj,
        "signal":     signal,
        "confidence": fund.get("confidence", 50),
        "thesis":     fund.get("thesis", ""),
        "ipo_mode":   ipo_mode,
        "atr_pct":    round(atr_pct, 4),   # adaptive profit target input
    }

# ══════════════════════════════════════════════════════════════
# PRE-MARKET SCAN
# ══════════════════════════════════════════════════════════════

def build_universe() -> list[str]:
    """
    Curated tickers + dynamic universe from 3 sources:

    Source 1 — Top gainers (movers): catches post-earnings spikes,
               news catalysts. Already used. Filtered to price ≥ $15
               and min gain ≥ 3% to exclude penny stock noise.

    Source 2 — Most active by TRADE COUNT (not volume): trade count
               is a better proxy for institutional interest than raw
               volume. High trade count = many separate orders =
               institutional accumulation. Filters out leveraged ETFs
               and penny stocks. This is where IOT-type stocks appear.

    Source 3 — 5-day momentum scan: stocks up 5-15% over 5 days
               with consistent daily gains. Captures steady institutional
               buying like IOT's post-earnings drift. Uses curated-adjacent
               tickers from a broader watchlist of $5B+ market cap stocks.
    """
    symbols = set(CURATED_TICKERS)
    added   = 0
    headers = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
    DATA_URL = "https://data.alpaca.markets/v1beta1"

    # ── Source 1: Top gainers (movers) ─────────────────────────
    try:
        r = requests.get(
            f"{DATA_URL}/screener/stocks/movers",
            headers=headers,
            params={"top": 50},
            timeout=10
        )
        if r.ok:
            gainers = r.json().get("gainers", [])
            for g in gainers:
                sym = g.get("symbol", "")
                pct = g.get("percent_change", 0)
                px  = g.get("price", 0)
                if (sym.isalpha() and len(sym) <= 5
                        and sym not in ETF_EXCLUSIONS
                        and sym not in symbols
                        and px >= 15           # no penny stocks
                        and pct >= 3.0         # min 3% gain — filters penny stock noise
                        and added < 8):
                    symbols.add(sym)
                    added += 1
            log.info(f"  Source 1 (movers): {added} tickers added")
    except Exception as e:
        log.warning(f"Movers fetch failed: {e}")

    # ── Source 2: Most active by TRADE COUNT ───────────────────
    # Trade count = institutional interest proxy (many orders = algos/funds)
    # Much better than volume for quality screening
    s2_added = 0
    try:
        r = requests.get(
            f"{DATA_URL}/screener/stocks/most-actives",
            headers=headers,
            params={"by": "trades", "top": 50},
            timeout=10
        )
        if r.ok:
            actives = r.json().get("most_actives", [])
            for a in actives:
                sym   = a.get("symbol", "")
                count = a.get("trade_count", 0)
                if (sym.isalpha() and len(sym) <= 5
                        and sym not in ETF_EXCLUSIONS
                        and sym not in symbols
                        and count >= 50000     # min 50k trades = real institutional interest
                        and s2_added < 8):
                    # Quick price check — skip penny stocks
                    try:
                        snap = requests.get(
                            f"{DATA_URL}/stocks/{sym}/snapshot",
                            headers=headers, timeout=5
                        )
                        if snap.ok:
                            px = snap.json().get("latestTrade", {}).get("p", 0)
                            if px >= 15:
                                symbols.add(sym)
                                s2_added += 1
                    except Exception:
                        pass
            log.info(f"  Source 2 (trade-count): {s2_added} tickers added")
    except Exception as e:
        log.warning(f"Most-actives fetch failed: {e}")

    # ── Source 3: 5-day momentum — quality mid-cap watchlist ───
    # These are $5B-$50B companies in growth sectors that don't
    # always make the top movers but have steady institutional buying.
    # IOT, SNOW, NET, DDOG, SHOP, ZS, PANW, HUBS, MSTR, ARM etc.
    # Claude scores these well when they're moving — they just need
    # to enter the universe first.
    MOMENTUM_WATCHLIST = [
        # Enterprise SaaS / Cybersecurity
        "IOT","SNOW","NET","DDOG","ZS","PANW","HUBS","GTLB","BILL","MDB",
        # Fintech
        "SQ","AFRM","COIN","HOOD",
        # Semiconductors (mid-cap)
        "ARM","MRVL","QCOM","ON","SMCI",
        # Healthcare / Biotech
        "MRNA","BNTX","INCY","ACAD","RARE",
        # Energy (mid-cap)
        "DVN","HAL","MPC","VLO",
        # Consumer / Retail
        "SHOP","ABNB","UBER","LYFT","DASH",
        # Industrials / Defence
        "HEI","AXON","TDG","LDOS",
    ]
    s3_added = 0
    try:
        # Fetch 5-day bars for watchlist in one call
        watch_syms = [s for s in MOMENTUM_WATCHLIST if s not in symbols and s not in ETF_EXCLUSIONS]
        if watch_syms:
            r = requests.get(
                f"{DATA_URL}/stocks/bars",
                headers=headers,
                params={
                    "symbols": ",".join(watch_syms[:40]),
                    "timeframe": "1Day",
                    "limit": 5 * len(watch_syms[:40]),
                    "start": (datetime.now(ET) - timedelta(days=8)).strftime("%Y-%m-%d"),
                },
                timeout=15
            )
            if r.ok:
                bars = r.json().get("bars", {})
                for sym, sym_bars in bars.items():
                    if len(sym_bars) < 3:
                        continue
                    # 5-day return
                    first_close = sym_bars[0].get("c", 0)
                    last_close  = sym_bars[-1].get("c", 0)
                    if first_close <= 0 or last_close < 10:
                        continue
                    ret_5d = (last_close - first_close) / first_close
                    # Volume check — above-average volume in last 2 days
                    recent_vols = [b.get("v", 0) for b in sym_bars[-2:]]
                    avg_vol     = sum(b.get("v", 0) for b in sym_bars) / len(sym_bars)
                    vol_surge   = avg_vol > 0 and (sum(recent_vols)/len(recent_vols)) > avg_vol * 0.8
                    # Add if up 3%+ over 5 days with decent volume
                    if ret_5d >= 0.03 and vol_surge and sym not in symbols and s3_added < 8:
                        symbols.add(sym)
                        s3_added += 1
                        log.info(f"  Source 3 (momentum): {sym} +{ret_5d*100:.1f}% 5d → added")
        log.info(f"  Source 3 (5-day momentum): {s3_added} tickers added")
    except Exception as e:
        log.warning(f"Momentum scan failed: {e}")

    total_dynamic = len(symbols) - len(CURATED_TICKERS)
    clean = [s for s in symbols if s.isalpha() and len(s) <= 5 and s not in ETF_EXCLUSIONS]
    log.info(
        f"Universe: {len(clean)} tickers "
        f"({len(CURATED_TICKERS)} curated + {total_dynamic} dynamic "
        f"[{added} movers, {s2_added} trade-active, {s3_added} momentum])"
    )
    return clean[:55]  # max 55 — allow headroom for all 3 sources

def run_premarket_scan():
    """
    Full universe scan — runs pre-market (Sunday 8pm or Monday 6am ET).
    Results cached for the trading day. Zero Claude calls during market hours
    unless cache is exhausted.
    """
    global signal_cache, signal_cache_time, signal_cache_date, fund_cache

    now_et = datetime.now(ET)
    today  = now_et.strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("PRE-MARKET SCAN STARTING")
    log.info(f"  Time: {now_et.strftime('%A %Y-%m-%d %H:%M ET')}")
    log.info(f"  Universe: 35 curated + up to 15 dynamic = max 50 tickers")
    log.info("=" * 60)

    # Clear yesterday's cache
    fund_cache = {}

    # Assess market state for RS calculation
    spy_chg = get_quote_change("SPY") or 0.0

    universe   = build_universe()
    candidates = []
    seen_symbols = set()  # deduplicate — LLY can appear in both curated and dynamic
    all_scored   = []     # track all scored results for diversity fallback

    for i, symbol in enumerate(universe):
        if symbol in seen_symbols:
            log.info(f"  {symbol}: already scanned — skipping duplicate")
            continue
        seen_symbols.add(symbol)
        result = compute_signal(symbol, spy_chg)
        if result:
            all_scored.append(result)
            if result["signal"] == "BUY" and result["confidence"] >= MIN_CONFIDENCE and result["composite"] >= 3.0:
                candidates.append(result)
                ipo_tag = " [IPO]" if result.get("ipo_mode") else ""
                log.info(
                    f"  ✅ {symbol}: composite={result['composite']:.2f}, "
                    f"confidence={result['confidence']}%{ipo_tag} — {result['thesis']}"
                )
        time.sleep(20)  # ~3 calls/min, well under rate limit

    # ── Sector diversity enforcement ──────────────────────────
    # If cache is all-tech, force add the best non-tech signal
    # even if it's below threshold — prevents $85k idle cash on tech weakness days
    DIVERSITY_SECTORS = ["financials", "healthcare", "energy", "industrials", "consumer"]
    cached_sectors = {SECTOR_MAP.get(c["symbol"]) for c in candidates}

    for sector in DIVERSITY_SECTORS:
        if sector not in cached_sectors:
            # Find best scoring signal in this sector from all_scored
            sector_best = sorted(
                [r for r in all_scored if SECTOR_MAP.get(r["symbol"]) == sector
                 and r["signal"] != "SELL"],
                key=lambda x: x["composite"], reverse=True
            )
            if sector_best:
                best = sector_best[0]
                # Only add if composite >= 1.5 (some positive signal, not just random)
                if best["composite"] >= 1.5:
                    candidates.append(best)
                    log.info(
                        f"  📊 DIVERSITY {best['symbol']} ({sector}): "
                        f"composite={best['composite']:.2f} — added for sector diversity"
                    )

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    signal_cache      = candidates
    signal_cache_time = time.time()
    signal_cache_date = today

    log.info("=" * 60)
    log.info(f"PRE-MARKET SCAN COMPLETE — {len(candidates)} BUY signals")
    for c in candidates:
        log.info(f"  #{candidates.index(c)+1} {c['symbol']}: {c['composite']:.2f} composite, {c['confidence']}% confidence")
    log.info("=" * 60)

def should_run_premarket_scan() -> bool:
    """
    Returns True ONLY in designated pre-market windows.
    TIME IS THE PRIMARY GATE — cache state is secondary.

    Scan windows:
    - Sunday 8pm-10pm ET  → Monday preparation
    - Mon-Fri 6am-9:29am ET → morning preparation

    NEVER scans:
    - During market hours (9:30am-4pm ET) — cache only
    - After 4pm ET — wait for 6am next morning
    - Weekends outside Sunday 8-10pm
    - On bot restart mid-session (even with empty cache)
    """
    now_et = datetime.now(ET)
    today  = now_et.strftime("%Y-%m-%d")
    hour   = now_et.hour
    minute = now_et.minute
    day    = now_et.weekday()  # 0=Mon, 6=Sun

    # ── Step 1: Time gate — must be in allowed window ──────────
    in_window = False

    # Sunday 8pm-10pm ET
    if day == 6 and 20 <= hour < 22:
        in_window = True

    # Weekday 6am-9:29am ET only
    elif 0 <= day <= 4 and 6 <= hour <= 9:
        market_open = hour == 9 and minute >= 30
        if not market_open:
            in_window = True

    # Not in any allowed window — never scan
    if not in_window:
        return False

    # ── Step 2: Already scanned today — don't repeat ───────────
    if signal_cache_date == today and signal_cache:
        return False

    # In window + cache stale → scan
    return True

# ══════════════════════════════════════════════════════════════
# ALPACA HELPERS
# ══════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    try:
        return trade_client.get_clock().is_open
    except Exception:
        return False

def get_account():
    return trade_client.get_account()

def get_positions() -> dict:
    try:
        return {p.symbol: p for p in trade_client.get_all_positions()}
    except Exception as e:
        log.error(f"Failed to get positions: {e}")
        return {}

def register_reentry_cooldown(symbol: str, exit_reason: str, exit_price: float):
    """
    Registers a re-entry cooldown after a position closes.
    Option 1: PROFIT_TARGET exit → 4hr cooldown
    Option 5: STOP_LOSS exit    → 24hr cooldown
    Option 3: Price gate stored for all exits (2% pullback required)
    """
    global reentry_cooldown
    now = time.time()

    if "PROFIT_TARGET" in exit_reason:
        reentry_cooldown[symbol] = {
            "type":       "profit",
            "time":       now,
            "exit_price": exit_price,
            "expires":    now + PROFIT_COOLDOWN_SECS,
        }
        log.info(
            f"  ⏳ {symbol}: profit cooldown active — "
            f"no re-entry for 4hrs, price gate ${exit_price*(1-PRICE_GATE_PCT):.2f} "
            f"(2% below exit ${exit_price:.2f})"
        )
    elif "STOP_LOSS" in exit_reason:
        reentry_cooldown[symbol] = {
            "type":       "stop",
            "time":       now,
            "exit_price": exit_price,
            "expires":    now + STOP_COOLDOWN_SECS,
        }
        log.info(
            f"  ⏳ {symbol}: stop loss cooldown active — "
            f"no re-entry for 24hrs (thesis failed)"
        )
    else:
        # Trailing/breakeven exits — only price gate applies, no time cooldown
        reentry_cooldown[symbol] = {
            "type":       "trailing",
            "time":       now,
            "exit_price": exit_price,
            "expires":    0,  # no time cooldown
        }

def check_reentry_allowed(symbol: str, current_price: float) -> tuple[bool, str]:
    """
    Checks Options 1, 3, 5 before allowing re-entry.
    Returns (allowed, reason_if_blocked).

    Price gate expiry: 2 trading days (48 hours).
    Prevents stale cooldowns from permanently blocking stocks
    that have genuinely moved on (e.g. AMD $526 → $581 weeks later).
    """
    if symbol not in reentry_cooldown:
        return True, ""

    cd      = reentry_cooldown[symbol]
    now     = time.time()
    cdtype  = cd["type"]
    expires = cd["expires"]
    exit_px = cd["exit_price"]

    # Option 1 — profit cooldown (4hr)
    if cdtype == "profit" and expires > 0 and now < expires:
        remaining = (expires - now) / 3600
        return False, f"profit cooldown ({remaining:.1f}hrs remaining after +5% exit)"

    # Option 5 — stop loss cooldown (24hr)
    if cdtype == "stop" and expires > 0 and now < expires:
        remaining = (expires - now) / 3600
        return False, f"stop loss cooldown ({remaining:.1f}hrs remaining — thesis failed)"

    # Option 3 — price gate (2% pullback required)
    # Expires after 48 hours — prevents stale gates blocking re-entry indefinitely
    PRICE_GATE_EXPIRY_SECS = 48 * 3600
    gate_set_time = cd.get("time", 0)
    gate_expired  = (now - gate_set_time) > PRICE_GATE_EXPIRY_SECS

    if not gate_expired:
        gate_price = exit_px * (1 - PRICE_GATE_PCT)
        if current_price > gate_price:
            hours_remaining = max(0, PRICE_GATE_EXPIRY_SECS - (now - gate_set_time)) / 3600
            return False, (
                f"price gate: current ${current_price:.2f} > gate ${gate_price:.2f} "
                f"(need 2% pullback from exit ${exit_px:.2f}, "
                f"or gate expires in {hours_remaining:.1f}hrs)"
            )

    # All checks passed — clear the cooldown
    del reentry_cooldown[symbol]
    return True, ""

def close_position(symbol: str, pnl_pct: float = 0.0, exit_reason: str = "") -> bool:
    try:
        # Cancel open orders first
        orders = trade_client.get_orders()
        for o in orders:
            if o.symbol == symbol:
                try:
                    trade_client.cancel_order_by_id(o.id)
                    time.sleep(0.5)
                except Exception:
                    pass

        # Get current price before closing for cooldown registration
        try:
            pos = trade_client.get_open_position(symbol)
            current_price = float(pos.current_price)
        except Exception:
            current_price = 0.0

        trade_client.close_position(symbol)
        log.info(f"[SELL] Closed {symbol} — {exit_reason} ({pnl_pct*100:+.2f}%)")
        record_trade(symbol, pnl_pct, exit_reason)

        # Register re-entry cooldown based on exit type
        register_reentry_cooldown(symbol, exit_reason, current_price)
        return True
    except Exception as e:
        log.error(f"Failed to close {symbol}: {e}")
        return False

def place_buy(symbol: str, qty: int) -> bool:
    try:
        # ── Layer 1: Duplicate position guard ─────────────────
        # Prevents buying a ticker already held (NVDA ×3 bug Jun 23)
        existing = get_positions()
        if symbol in existing:
            log.warning(f"[SKIP] {symbol} already held — duplicate position blocked")
            return False

        # ── Layer 2: Open orders guard ────────────────────────
        # Prevents buying if a pending buy order already exists for same ticker
        try:
            open_orders = trade_client.get_orders()
            for o in open_orders:
                if o.symbol == symbol and o.side.value == "buy":
                    log.warning(f"[SKIP] {symbol} has pending buy order — duplicate order blocked")
                    return False
        except Exception:
            pass  # if check fails, proceed cautiously

        order = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
        )
        trade_client.submit_order(order)
        today_key = datetime.now(ET).strftime("%Y-%m-%d")
        trades_today[today_key] = trades_today.get(today_key, 0) + 1
        log.info(f"[BUY] {qty}x {symbol} — trades today: {trades_today.get(today_key,0)}/{MAX_TRADES_DAY}")
        return True
    except Exception as e:
        log.error(f"Failed to buy {symbol}: {e}")
        return False

# ══════════════════════════════════════════════════════════════
# RISK CONTROLS
# ══════════════════════════════════════════════════════════════

def is_paused() -> bool:
    return os.environ.get("PAUSED", "false").strip().lower() in ("true", "1", "yes")

def trades_today_count() -> int:
    return trades_today.get(datetime.now(ET).strftime("%Y-%m-%d"), 0)

def check_drawdown(account) -> bool:
    global circuit_breaker, starting_equity
    if circuit_breaker:
        return False
    equity = float(account.equity)
    if starting_equity is None:
        starting_equity = equity
        log.info(f"Starting equity: ${starting_equity:,.2f}")
        return True
    drawdown = (starting_equity - equity) / starting_equity
    if drawdown >= MAX_DRAWDOWN:
        circuit_breaker = True
        log.critical(f"CIRCUIT BREAKER — drawdown {drawdown*100:.1f}% > {MAX_DRAWDOWN*100:.0f}%")
        return False
    return True

def run_risk_checks(account) -> tuple[bool, str]:
    if is_paused():
        return False, "PAUSED"
    if not check_drawdown(account):
        return False, "CIRCUIT BREAKER"
    if trades_today_count() >= MAX_TRADES_DAY:
        return False, f"MAX TRADES {trades_today_count()}/{MAX_TRADES_DAY}"
    return True, "OK"

# ══════════════════════════════════════════════════════════════
# POSITION MONITORING
# ══════════════════════════════════════════════════════════════

def check_profit_targets(positions: dict) -> list[str]:
    """
    Exit rules — no Claude calls needed.
    Four exit conditions:
      1. Profit target +5%
      2. Trailing protection: peak ≥ +3% → sell if falls to +2.5%
      3. Breakeven stop: peak ≥ +1% → stop shifts to +0.5%
      4. Hard stop loss: -5% (tightens to -2% in BEAR mode)
    Plus:
      5. Weak sector mid-day exit: if sector turns weak AND position
         is at breakeven (+0.5%) or better → exit to protect gains.
         If position is negative → hold (don't crystallise a loss).
    """
    closed      = []
    active_stop = get_stop_loss()

    for symbol, pos in positions.items():
        try:
            pnl_pct = float(pos.unrealized_plpc)

            # Update peak
            prev_peak = position_peaks.get(symbol, 0.0)
            if pnl_pct > prev_peak:
                position_peaks[symbol] = pnl_pct
                save_peaks()
                if pnl_pct >= PEAK_TRIGGER:
                    log.info(f"  {symbol}: new peak {pnl_pct*100:+.2f}% — trailing active")
                elif pnl_pct >= BREAKEVEN_TRIGGER:
                    log.info(f"  {symbol}: new peak {pnl_pct*100:+.2f}% — breakeven stop active (+0.5%)")

            current_peak = position_peaks.get(symbol, 0.0)
            reason       = None

            # ── Adaptive profit target — ATR-based per position ──
            # Stored at buy time: position_peaks[f"{symbol}_target"]
            # High-vol stocks (MU, NVDA, AMD): ATR ~4% → target ~10%
            # Low-vol stocks (JPM, V, MA):     ATR ~1% → target ~3%
            # Falls back to fixed PROFIT_TARGET if no ATR stored
            profit_target = position_peaks.get(f"{symbol}_target", PROFIT_TARGET)

            # ── Standard exit rules ────────────────────────────
            if pnl_pct >= profit_target:
                reason = f"PROFIT_TARGET ({profit_target*100:.1f}%)"
            elif current_peak >= PEAK_TRIGGER and pnl_pct <= TRAIL_SELL:
                reason = f"TRAILING (peaked {current_peak*100:+.2f}%)"
            elif current_peak >= BREAKEVEN_TRIGGER and pnl_pct <= BREAKEVEN_STOP:
                reason = f"BREAKEVEN_STOP (peaked {current_peak*100:+.2f}%, locked +0.5%)"
            elif current_peak < BREAKEVEN_TRIGGER and pnl_pct <= active_stop:
                reason = f"STOP_LOSS ({active_stop*100:.0f}%)"

            # ── Weak sector mid-day exit ───────────────────────
            # Only exit if position is at breakeven (+0.5%) or better
            # Never crystallise a loss due to sector rotation
            if not reason and weak_sectors:
                sector = SECTOR_MAP.get(symbol)
                if sector and sector in weak_sectors:
                    if pnl_pct >= BREAKEVEN_STOP:
                        reason = (
                            f"WEAK_SECTOR ({sector.upper()} weak, "
                            f"P&L {pnl_pct*100:+.2f}% ≥ +0.5% — exiting)"
                        )
                    else:
                        log.info(
                            f"  {symbol}: sector '{sector}' weak but "
                            f"P&L {pnl_pct*100:+.2f}% below breakeven — holding"
                        )

            if reason:
                if close_position(symbol, pnl_pct, reason):
                    closed.append(symbol)
                    position_peaks.pop(symbol, None)
                    save_peaks()
                    # Remove from signal cache so it's not immediately re-bought
                    global signal_cache
                    signal_cache = [s for s in signal_cache if s["symbol"] != symbol]
            else:
                peak_str = f" (peak: {current_peak*100:+.2f}%)" if current_peak >= PEAK_TRIGGER else ""
                log.info(f"  {symbol}: {pnl_pct*100:+.2f}% P&L{peak_str} — holding")

        except Exception as e:
            log.warning(f"  Error checking {symbol}: {e}")

    return closed

# ══════════════════════════════════════════════════════════════
# DEPLOYMENT
# ══════════════════════════════════════════════════════════════

def deploy_from_cache(positions: dict, account):
    """
    Deploy capital from cached signal list — zero Claude calls.
    Signal cache built pre-market contains ranked BUY signals.
    """
    global signal_cache, last_rescan_time

    KELLY_PCT   = 0.10
    MAX_POS     = 10
    equity      = float(account.equity)
    cash        = float(account.cash)
    open_slots  = MAX_POS - len(positions)

    if open_slots <= 0:
        log.info(f"All {MAX_POS} slots filled")
        return

    if cash < equity * KELLY_PCT:
        log.info(f"Insufficient cash (${cash:.0f}) for Kelly position (${equity*KELLY_PCT:.0f})")
        return

    if market_state == "BEAR":
        log.warning("BEAR MODE — no new positions")
        return

    # Use cached signals — deduplicated by symbol, no Claude calls
    seen = set()
    available = []
    for s in signal_cache:
        if s["symbol"] not in positions and s["symbol"] not in seen:
            seen.add(s["symbol"])
            available.append(s)

    if not available:
        # Cache exhausted — emergency rescan max once per hour
        time_since = time.time() - last_rescan_time
        if time_since > 3600:  # 1 hour
            log.info("Signal cache exhausted — running emergency rescan (1hr cooldown)")
            last_rescan_time = time.time()
            spy_chg = get_quote_change("SPY") or 0.0
            universe = build_universe()
            new_signals = []
            for symbol in universe:
                if symbol in positions:
                    continue
                result = compute_signal(symbol, spy_chg)
                if result and result["signal"] == "BUY" and result["confidence"] >= MIN_CONFIDENCE and result["composite"] >= 3.0:
                    new_signals.append(result)
                time.sleep(20)
            signal_cache = sorted(new_signals, key=lambda x: x["confidence"], reverse=True)
            available    = signal_cache
        else:
            log.info(f"Cache empty — next emergency rescan in {max(0,(3600-time_since)/60):.0f}min")
            return

    to_buy = available[:open_slots]
    log.info(f"Deploying into {len(to_buy)} signal(s) from cache: {[s['symbol'] for s in to_buy]}")

    for candidate in to_buy:
        symbol = candidate["symbol"]
        price  = candidate["price"]
        if price <= 0:
            continue

        # ── Sector cap — max 2 positions per sector (backtest validated) ──
        # Apr 26 +$1,196, May 26 +$1,914 better vs uncapped
        # Prevents NVDA×3 concentration regardless of signal quality
        sector = SECTOR_MAP.get(symbol)
        if sector:
            sector_count = sum(1 for s in positions if SECTOR_MAP.get(s) == sector)
            if sector_count >= MAX_SECTOR_POSITIONS:
                log.info(
                    f"  {symbol}: sector '{sector}' capped — "
                    f"{sector_count}/{MAX_SECTOR_POSITIONS} positions held — skipping"
                )
                continue

        # ── Weak sector filter — block buys when sector ETF down >1.5% ──
        if sector and sector in weak_sectors:
            log.info(
                f"  {symbol}: sector '{sector}' is weak today — "
                f"skipping new buy (cache signal preserved for tomorrow)"
            )
            continue

        # ── Re-entry cooldown checks (Options 1, 3, 5) ────────
        # Fetch current live price for Option 3 price gate
        try:
            snap  = trade_client.get_stock_latest_bar(symbol)
            live_price = float(snap[symbol].c) if snap and symbol in snap else price
        except Exception:
            live_price = price

        allowed, block_reason = check_reentry_allowed(symbol, live_price)
        if not allowed:
            log.info(f"  {symbol}: re-entry BLOCKED — {block_reason}")
            continue

        # ── Confidence-based sizing (backtest validated: marginal improvement) ──
        # High conviction (composite ≥ 6.0, confidence ≥ 90%) → 12% Kelly
        # Normal (composite ≥ 4.5) → 10% Kelly
        # Marginal (composite < 4.5) → 8% Kelly
        composite  = candidate.get("composite", 0)
        confidence = candidate.get("confidence", 85)
        if composite >= 6.0 and confidence >= 90:
            kelly = 0.12
        elif composite >= 4.5:
            kelly = 0.10
        else:
            kelly = 0.08

        alloc  = min(equity * kelly, cash * 0.95)
        qty    = int(alloc / live_price)
        if qty < 1:
            continue
        qty = adjust_qty_for_fear(qty, live_price, alloc)

        # ── ATR-based profit target (backtest validated: +$45 avg win) ──
        # Stores target in peaks dict so check_profit_targets can use it
        atr = candidate.get("atr_pct", None)
        if atr and atr > 0:
            atr_target = min(0.12, max(0.03, atr * 2.5))
            position_peaks[f"{symbol}_target"] = atr_target
            save_peaks()
            log.info(f"  {symbol}: {qty} shares @ ~${live_price:.2f} = ${qty*live_price:,.0f} ({kelly*100:.0f}% Kelly, ATR target {atr_target*100:.1f}%)")
        else:
            log.info(f"  {symbol}: {qty} shares @ ~${live_price:.2f} = ${qty*live_price:,.0f} ({kelly*100:.0f}% Kelly)")

        place_buy(symbol, qty)
        cash -= qty * live_price

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run():
    load_peaks()
    start_news_stream()  # Start news WebSocket in background thread

    log.info("=" * 60)
    log.info("SIGNAL Trading Bot started")
    log.info(f"  Universe:       36 curated + 15 dynamic = 51 max")
    log.info(f"  Scan timing:    Pre-market (Sun 8pm / Mon-Fri 6am ET)")
    log.info(f"  Position size:  Kelly 8-12% (confidence-based sizing)")
    log.info(f"  Profit target:  ATR×2.5 per position (3-12% range, fallback +5%)")
    log.info(f"  Sector cap:     Max 2 positions per sector (backtest validated)")
    log.info(f"  Re-entry rules: +5% exit → 4hr cooldown + 2% price gate")
    log.info(f"                  -5% stop → 24hr cooldown + 2% price gate")
    log.info(f"  Max positions:  10 concurrent")
    log.info(f"  Profit target:  +{PROFIT_TARGET*100:.0f}%")
    log.info(f"  Trailing:       peak >={PEAK_TRIGGER*100:.0f}% → sell at +{TRAIL_SELL*100:.0f}%")
    log.info(f"  Stop loss:      -{abs(STOP_LOSS)*100:.0f}%")
    log.info(f"  TA/Fund weight: {TECH_WEIGHT}% / {FUND_WEIGHT}%")
    log.info(f"  Min confidence: {MIN_CONFIDENCE}%")
    log.info(f"  Max drawdown:   {MAX_DRAWDOWN*100:.0f}%")
    log.info(f"  90-day audit:   Volume, TA alignment, win rate")
    log.info(f"  Pause:          set PAUSED=true in Render")
    log.info("=" * 60)

    # 90-day audit check
    audit_file = "/tmp/last_audit.txt"
    try:
        with open(audit_file) as f:
            last_audit = datetime.strptime(f.read().strip(), "%Y-%m-%d").replace(tzinfo=ET)
        if (datetime.now(ET) - last_audit).days >= 90:
            run_90_day_audit()
            with open(audit_file, "w") as f:
                f.write(datetime.now(ET).strftime("%Y-%m-%d"))
    except FileNotFoundError:
        with open(audit_file, "w") as f:
            f.write(datetime.now(ET).strftime("%Y-%m-%d"))

    while True:
        try:
            now_et = datetime.now(ET)

            # ── Pre-market scan window ─────────────────────────
            if should_run_premarket_scan():
                assess_market_state()
                run_premarket_scan()
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Market closed — sleep ──────────────────────────
            if not is_market_open():
                log.info(f"Market closed ({now_et.strftime('%H:%M ET')}) — sleeping")
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Market open — monitor positions ────────────────
            log.info(f"{'─'*50}")
            log.info(f"Scan at {now_et.strftime('%H:%M:%S ET')}")

            account = get_account()
            equity  = float(account.equity)
            cash    = float(account.cash)
            log.info(f"Equity: ${equity:,.2f}  Cash: ${cash:,.2f}  Trades: {trades_today_count()}/{MAX_TRADES_DAY}")

            safe, reason = run_risk_checks(account)
            if not safe:
                log.warning(f"RISK GATE: {reason}")
                time.sleep(SCAN_INTERVAL)
                continue

            # Assess market state every cycle
            assess_market_state()

            positions = get_positions()
            log.info(f"Open positions: {list(positions.keys()) or 'none'}")
            prune_peaks(list(positions.keys()))

            # Check exits
            closed = check_profit_targets(positions) if positions else []

            if closed:
                time.sleep(3)
                positions = get_positions()
                account   = get_account()

            # ── Process news triggers (real-time events) ───────
            news_traded = process_news_queue(positions, account)
            if news_traded:
                time.sleep(3)
                positions = get_positions()
                account   = get_account()

            # Deploy from cache (no Claude calls during market hours)
            open_slots = 10 - len(positions)
            if open_slots > 0 and float(account.cash) >= equity * 0.10:
                if not signal_cache:
                    log.info(
                        f"  {open_slots} slot(s) available but signal cache is empty — "
                        f"waiting for tomorrow's 6am ET pre-market scan. "
                        f"News WebSocket will trigger immediate re-score on breaking events."
                    )
                else:
                    deploy_from_cache(positions, account)

        except KeyboardInterrupt:
            log.info("Bot stopped.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run()
