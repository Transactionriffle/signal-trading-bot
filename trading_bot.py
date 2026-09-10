"""
SIGNAL Trading Bot
==================
Architecture:
  Pre-market scan (Sunday 8pm ET, or dynamic ~9:20am ET weekdays):
    → Build up to 55-ticker universe (37 curated + up to 24 dynamic:
      8 movers + 8 trade-count-active + 8 five-day-momentum)
    → TA pre-filter (skip Claude entirely if taScore < 1.5)
    → Run fundamental scan (Claude + web search) on surviving tickers
    → Hard vetoes applied before scoring: ETF/leveraged-product,
      SPAC/blank-check, general structural-risk (recent IPO, pending
      M&A, reverse split, halted, thin ADR), earnings within 3 days
    → Cache ranked BUY list (composite >= MIN_COMPOSITE, confidence
      >= MIN_CONFIDENCE) with scores; persisted to disk so a mid-day
      redeploy doesn't trigger a wasted re-scan
    → Deploy capital at market open from cached list

  During market hours:
    → Monitor positions every 60 seconds (zero Claude calls)
    → Exit priority: +60% hard ceiling (the only fixed profit exit —
      ATR/fixed target removed Sep 10 2026) > stop-loss-magnitude
      override (always labelled STOP_LOSS regardless of prior peak,
      for accurate attribution) > dynamic-width trail (arms at +1.0%
      peak, gap widens with peak size; fresh tiers at +5/10/15/20/40%
      so a runner is never sold at a fixed level below +60%) > old
      fixed trail (+3%→+2.5%, rarely reached first) > breakeven stop
      (VIX-aware: wider band when VIX<18) > composite-tiered stop
      loss (-0.75%/-1.0%/-1.5% by entry conviction, dynamic downside
      floor tightens as loss deepens) > weak-sector mid-day exit
    → First 30 min after entry: full noise tolerance, nothing fires.
      After that: ATR-scaled early-exit stop (hard-capped -0.5%) AND
      the composite-tiered stop are both active — whichever is
      tighter effectively governs for a position with zero validation
    → Held positions get re-evaluated (not ignored) on bearish/
      material-adverse news, and on a periodic stale-fundamentals
      recheck independent of news
    → Position closes → re-entry cooldown (type-specific duration +
      escalating strikes for repeat stop-losses) → deploy into next
      ranked signal from cache
    → Only re-scan if cache exhausted (max once per hour)
    → Systemic de-risk (severe VIX + 3+ weak sectors): blocks new
      buys, allows early loss-cutting up to -2% instead of riding
      each position to its full stop distance one at a time

  Every 90 days — Curated ticker + universe-composition audit:
    Criterion 1 — Volume & Liquidity: avg daily volume dropped?
    Criterion 2 — Strategy Alignment: does stock still respect TA setups?
    Criterion 3 — Personal Performance: negative win rate over last 90 days?
    Criterion 4 — Universe-level AI concentration: has the curated
                  LIST itself drifted too AI/tech-heavy over time?

Environment variables (set in Render):
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY  (required)
    ALPACA_BASE_URL        (default: https://paper-api.alpaca.markets)
    CLOUDFLARE_WORKER
    TECH_WEIGHT            (default: 40 — 40% TA / 60% fundamental)
    MIN_CONFIDENCE         (default: 85)
    MIN_COMPOSITE          (default: 4.0)
    MAX_TRADES_PER_DAY     (default: 10)
    MAX_DRAWDOWN_PCT       (default: 0.15)
    PAUSED                 (set "true" to halt instantly)

Macro layer:
    BULL  SPY >= -2% or VIX < 25       → normal trading
    BEAR  SPY < -2% AND VIX >= 25      → no new buys, stops tighten to -2%
    Fear  VIX >= 25                     → position sizes halved
    Systemic de-risk  VIX>=28 + 3+ weak sectors → no new buys, faster loss-cutting
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
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

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

# Raised from 3.0 → 4.0 (Jul 29 review). Rationale: realised trades were
# clustering at +0.3-0.5% wins vs -5% stop losses — a ratio that needs a
# ~93% win rate just to break even, which no live sample sustained. Every
# marginal (3.0-4.0) entry was negative expected value at that risk/reward.
# Raising the floor cuts trade count but concentrates capital in the
# higher-conviction setups the composite score is actually meant to find.
MIN_COMPOSITE   = float(os.environ.get("MIN_COMPOSITE", "4.0"))
MAX_TRADES_DAY  = int(os.environ.get("MAX_TRADES_PER_DAY", "10"))
MAX_DRAWDOWN    = float(os.environ.get("MAX_DRAWDOWN_PCT", "0.15"))
SCAN_INTERVAL   = 60
ET              = ZoneInfo("America/New_York")

# ── Risk thresholds ────────────────────────────────────────────
# Sep 10 2026 — fixed profit target REMOVED. Realised wins were being
# capped at +0.1-0.5% (MSFT +0.11%) while losses ran to -1.7% (SLB) —
# a badly inverted win/loss ratio. Winners are now only closed by the
# escalating dynamic trail (see DYNAMIC_TRAIL_TABLE) or the hard
# ceiling below. Nothing sells a position at a fixed +5% any more.
HARD_SELL_CEILING = 0.60   # +60%  the ONLY fixed profit exit — sell immediately, no trail given
PEAK_TRIGGER    =  0.03    # +3%   activate trailing protection (legacy fallback path)
TRAIL_SELL      =  0.025   # +2.5% sell if falls back here after peak (legacy fallback path)
# Sep 10 2026 — hard stop tightened from -5% to -1.5%. This is the CEILING
# for the highest-conviction tier; the composite tiers below it are
# tighter still (-0.75% / -1.0% / -1.5%). Target risk:reward is ~2:1 —
# with wins allowed to run via the trail, a stop this tight only needs
# a ~34% win rate to break even, versus the ~93% the old -5% flat
# stop against +0.4% typical wins demanded.
STOP_LOSS       = -0.015   # -1.5% hard stop ceiling (before breakeven activates)
STOP_LOSS_ACTIVATION_MINUTES = 30  # composite-tiered stop loss only fires after
                                     # this many minutes held — avoids cutting a
                                     # fresh position on opening-print/spread noise
                                     # before the thesis has had time to develop
BREAKEVEN_TRIGGER = 0.01   # +1%   once hit, stop shifts to +0.5% (default / high-VIX)
BREAKEVEN_STOP    = 0.005  # +0.5% minimum locked-in gain after breakeven (default / high-VIX)

# ── Dynamic-width trailing stop (Aug 2026) ──────────────────────
# Sep 10 2026: arm raised from +0.2% to +1.0%, first gap widened from
# 0.1% to 0.4%. The +0.2%/0.1% row was producing the +0.1% "wins"
# (MSFT +0.11% today) — it locked winners in before they had any room
# to become real gains. Now a position must peak +1% before the trail
# arms, and its first floor sits at +0.6% — just above the +0.5%
# breakeven lock, so the two rules stay coherent rather than fighting.
# Arms as soon as peak reaches +1.0%. The gap between peak and the
# sell-floor is NOT constant — it widens as the peak grows, so small
# moves get locked in tight (protect against noise) while genuine
# runners (like MU's +19% day) get progressively more room to
# breathe instead of being stopped out on the first 0.1% wobble.
# Table is peak-threshold -> trail gap. Highest matching threshold
# the peak has reached determines the active gap.
MICRO_TRAIL_ARM_PCT = 0.01    # +1.0% peak required to arm the trail at all (was +0.2%)
# Sep 10 2026 — table extended upward. With the fixed profit target
# gone, a runner past +5% keeps trailing under the SAME mechanic, with
# a fresh (wider) tier arming at +10%, +20% and +40%. +60% is the hard
# ceiling (HARD_SELL_CEILING) where the position is sold outright.
DYNAMIC_TRAIL_TABLE = [
    # (peak_threshold, gap_below_peak)
    (0.01,   0.004),   # peak +1%    → floor 0.4% behind   (first lock at +0.6%, just above breakeven's +0.5%)
    (0.02,   0.005),   # peak +2%    → floor 0.5% behind
    (0.05,   0.015),   # peak +5%    → floor 1.5% behind   (old profit-target level — now just another trail tier)
    (0.10,   0.03),    # peak +10%   → floor 3.0% behind
    (0.15,   0.05),    # peak +15%   → floor 5.0% behind   (room for a runner)
    (0.20,   0.06),    # peak +20%   → floor 6.0% behind
    (0.40,   0.10),    # peak +40%   → floor 10.0% behind  (last trail tier before the +60% hard ceiling)
]

def get_dynamic_trail_gap(peak_pct: float) -> float:
    """Returns the trail gap for the current peak — picks the gap
    from the highest threshold in DYNAMIC_TRAIL_TABLE the peak has
    reached. E.g. peak=+7% → uses the +5% row → gap=1.5%."""
    gap = DYNAMIC_TRAIL_TABLE[0][1]
    for threshold, table_gap in DYNAMIC_TRAIL_TABLE:
        if peak_pct >= threshold:
            gap = table_gap
        else:
            break
    return gap

# ── Dynamic downside floor (Sep 2026) ────────────────────────────
# Mirrors the dynamic trail concept, inverted: the trail widens its
# gap as a GAIN grows, giving winners more room the further they run.
# This does the opposite for LOSSES — the effective stop TIGHTENS
# (moves closer to current price) the deeper a loss gets, rather than
# giving the position the full remaining distance to its tier ceiling
# (-3%/-4%/-5%) regardless of how the decline is behaving.
#
# Rationale (Sep 8 case): NVDA sat at -2.01% for over 3 hours, still
# well inside its tier ceiling the whole time — the flat-tier approach
# would let it ride the FULL remaining distance to -3%/-4%/-5% before
# doing anything, treating "-0.5% and drifting" identically to "-2.5%
# and accelerating." A trader reads the RATE and DEPTH of a decline as
# informative, the same way the trail reads the SIZE of a peak as
# informative — this generalises that same instinct to the downside,
# independent of the systemic-derisk exception (which only applies
# during a severe VIX + multi-sector event).
#
# Table is fraction-of-tier-consumed -> effective stop at that point.
# E.g. tier=-4%: at -1% (25% of tier consumed) still get the full -4%
# ceiling; at -2.5% (62.5% consumed) effective stop tightens to -3%;
# beyond -3% it tightens further toward the tier itself. Keeps the
# 30-minute activation delay untouched — this only changes WHAT the
# stop level is once the delay has passed, not WHEN it can first fire.
DYNAMIC_DOWNSIDE_TABLE = [
    # (fraction_of_tier_consumed, effective_stop_as_fraction_of_tier)
    (0.0,  1.00),   # 0-25% into the tier   → full tier ceiling, give it room
    (0.25, 0.85),   # 25-50% into the tier  → tighten to 85% of tier
    (0.50, 0.65),   # 50-75% into the tier  → tighten to 65% of tier
    (0.75, 0.50),   # 75%+ into the tier    → tighten to 50% of tier — cut sooner
                     #                          if it's already deep and not stabilising
]

def get_dynamic_downside_floor(pnl_pct: float, tier_stop: float) -> float:
    """
    Returns the EFFECTIVE stop for the current loss depth, tightening
    progressively as pnl_pct approaches tier_stop (the tier ceiling,
    e.g. -0.03/-0.04/-0.05). Both inputs are negative fractions.
    Returns a value between tier_stop and 0 — always at least as tight
    as (closer to zero than) the flat tier, never looser.
    """
    if tier_stop >= 0:
        return tier_stop  # guard against misuse — tier stops are always negative
    fraction_consumed = min(1.0, pnl_pct / tier_stop)  # both negative → ratio is positive, 0..1+
    multiplier = DYNAMIC_DOWNSIDE_TABLE[0][1]
    for threshold, table_mult in DYNAMIC_DOWNSIDE_TABLE:
        if fraction_consumed >= threshold:
            multiplier = table_mult
        else:
            break
    return tier_stop * multiplier

# ── VIX-aware breakeven — calm markets need more room before locking in ──
# Rationale: in a low-VIX (<18) tape, stocks oscillate ±1% on pure noise.
# The tight 1%/0.5% breakeven was catching that noise and exiting winners
# early (avg realised gain was 0.36-0.46% instead of riding toward +3-5%).
CALM_VIX_THRESHOLD      = 18
BREAKEVEN_TRIGGER_CALM  = 0.02    # +2%   needs more confirmation in calm markets
BREAKEVEN_STOP_CALM     = 0.01    # +1%   locked-in gain, still meaningfully positive

# ── Macro thresholds ───────────────────────────────────────────
SPY_BEAR        = -0.02
VIXY_FEAR       =  0.05
SECTOR_WEAK     = -0.015
SECTOR_ETFS     = {
    "tech": "XLK", "healthcare": "XLV", "financials": "XLF",
    "energy": "XLE", "utilities": "XLU", "consumer": "XLY",
    "industrials": "XLI", "materials": "XLB",
    # Sep 2026 audit fix: SECTOR_MAP classifies NVDA/AMD/MU/TSM as "semis",
    # PLTR/CRM/SNOW as "software", PANW/CRWD/NET as "cyber", and SPOT/NFLX
    # as "media" — but none of those four buckets had a proxy ETF here,
    # so weak_sectors could NEVER contain them and the weak-sector buy
    # filter / mid-day exit was structurally inapplicable to the entire
    # semiconductor curated list (arguably the highest-volatility, most
    # news-sensitive part of the universe) plus software/cyber/media.
    # Verified liquid, real ETFs for each (checked directly, not guessed):
    # SMH ($65-71B AUM, ~11M/day), IGV ($12-15B AUM, ~15M/day), CIBR
    # ($9-14B AUM, ~1.2-1.76M/day). "media" specifically has no good
    # dedicated ETF (PBS/IEME are thin-to-liquidated) — SPOT/NFLX are
    # standard-classified under Communication Services, so XLC (State
    # Street, $22B AUM) is the correct, real proxy, not a media-only fund.
    "semis": "SMH", "software": "IGV", "cyber": "CIBR", "media": "XLC",
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
    "CPRX",   # $10B  — Catalyst Pharmaceuticals, rare disease portfolio
              #          (Firdapse, Agamree, Fycompa), +24% revenue YoY,
              #          +45.9% EPS growth, profitable (not binary-catalyst
              #          dependent like NUVL was) — replaces NUVL Aug 21 2026
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

# NOTE: SECTOR_MAP is defined once, further below (see "Sector concentration
# cap" section) with the full semis/tech/software/cyber/financials/healthcare/
# energy/industrials/consumer split that AI_CORRELATED_SECTORS and get_sector()
# both depend on. (Sep 2026 audit: a duplicate, incompatible definition used
# to live here — everything mapped to "tech" — and was silently shadowed by
# the real one 800+ lines later. Removed to eliminate the trap of someone
# editing this dead copy and wondering why nothing changes.)

# ── Dynamic sector classification (Sep 2026) ─────────────────────
# Root cause: PHVS (Pharvaris, a biotech) was bought Sep 8 while
# healthcare (XLV) was already -1.72% at market open — well past the
# -1.5% SECTOR_WEAK threshold. The weak-sector filter in
# deploy_from_cache() correctly checked `if sector and sector in
# weak_sectors`, but PHVS has NO entry in SECTOR_MAP (it entered the
# universe via the dynamic momentum screener reacting to Phase 3 trial
# news, not the curated list), so `sector` was None — falsy — and the
# entire check was silently skipped. The filter didn't fail; it was
# never applicable to a ticker it had no classification for.
#
# Fix: any ticker not in the static SECTOR_MAP gets a live sector
# lookup via Yahoo Finance's quoteSummary (assetProfile module, proxied
# through the same Cloudflare Worker already used for VIX/technicals),
# cached for 30 days per symbol since a company's sector classification
# essentially never changes. Falls back to "unknown" (not None) if the
# lookup fails — "unknown" is still a hashable value the weak_sectors
# set will simply never contain, but at minimum makes the gap visible
# in logs rather than silently invisible.
DYNAMIC_SECTOR_MAP: dict[str, str] = {}   # {symbol: sector} — runtime cache, session-lived
DYNAMIC_SECTOR_TTL = 30 * 86400            # 30 days — sector classification rarely changes

# Maps Yahoo Finance's broad "sector" field to this bot's sector buckets
# so a dynamically-looked-up ticker lines up with SECTOR_ETFS/weak_sectors
YAHOO_SECTOR_TO_BUCKET = {
    "technology": "tech",
    "healthcare": "healthcare",
    "financial services": "financials",
    "financial": "financials",
    "energy": "energy",
    "utilities": "utilities",
    "consumer cyclical": "consumer",
    "consumer defensive": "consumer",
    "industrials": "industrials",
    "basic materials": "materials",
}

def get_sector(symbol: str) -> str | None:
    """
    Returns this bot's sector bucket for `symbol` — checks the static
    SECTOR_MAP first (free, instant), then falls back to a live lookup
    for any ticker not hand-curated there (e.g. dynamically-discovered
    momentum/news tickers like PHVS). Cached in-session; unresolved
    lookups are cached as "unknown" so a failed lookup doesn't retry
    every single cycle.
    """
    if symbol in SECTOR_MAP:
        return SECTOR_MAP[symbol]

    cache_key = f"sector_lookup_{symbol}"
    if cache_key in fund_cache:
        cached_time, cached_sector = fund_cache[cache_key]
        if time.time() - cached_time < DYNAMIC_SECTOR_TTL:
            return cached_sector if cached_sector != "unknown" else None

    try:
        url = f"{WORKER_URL}/yahoofinance/quoteSummary/{symbol}?modules=assetProfile"
        r   = requests.get(url, timeout=10)
        if r.ok:
            data = r.json()
            profile = (
                data.get("quoteSummary", {})
                    .get("result", [{}])[0]
                    .get("assetProfile", {})
            )
            yahoo_sector = (profile.get("sector") or "").strip().lower()
            bucket = YAHOO_SECTOR_TO_BUCKET.get(yahoo_sector)
            if bucket:
                fund_cache[cache_key] = (time.time(), bucket)
                log.info(f"  {symbol}: dynamically classified as '{bucket}' (Yahoo sector: '{yahoo_sector}')")
                return bucket
            else:
                log.warning(f"  {symbol}: Yahoo sector '{yahoo_sector}' has no bucket mapping — treating as unknown")
    except Exception as e:
        log.warning(f"  {symbol}: dynamic sector lookup failed: {e}")

    fund_cache[cache_key] = (time.time(), "unknown")
    return None

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

# Keywords/patterns identifying SPACs (blank-check acquisition companies).
# Root cause Aug 21: RFAI/RFAIU (RF Acquisition Corp II) spiked +230-473%
# on a merger-vote approval — a shell company with "no significant
# operations" that still cleared the composite floor because its extreme
# TA/momentum score outweighed near-zero fundamentals in the 40/60 blend.
# This is a hard veto, independent of composite math — same category as
# the ETF exclusion, for a different structurally-risky instrument type.
SPAC_NAME_KEYWORDS = [
    "acquisition corp", "acquisition corporation", "acquisition co",
    "blank check", "spac", "special purpose acquisition",
    "capital corp ii", "capital corp iii",  # common serial-SPAC-sponsor naming
]

def is_likely_spac(symbol: str) -> bool:
    """
    Detects SPACs / blank-check companies via Alpaca asset name lookup,
    same mechanism as is_likely_etf(). SPACs have no operating business —
    price action is driven entirely by merger-vote speculation, which can
    produce an extreme TA/momentum score that masks near-zero fundamentals
    in the composite blend. Hard veto: blocked regardless of composite score.
    Result cached in fund_cache, same TTL pattern as the ETF check.
    """
    cache_key = f"spac_check_{symbol}"
    if cache_key in fund_cache:
        _, result = fund_cache[cache_key]
        return result

    try:
        asset = trade_client.get_asset(symbol)
        name  = (asset.name or "").lower()
        is_spac = any(kw in name for kw in SPAC_NAME_KEYWORDS)
        if is_spac:
            log.warning(f"  {symbol}: identified as SPAC/blank-check company ('{asset.name}') — blocking")
        fund_cache[cache_key] = (time.time(), is_spac)
        return is_spac
    except Exception:
        fund_cache[cache_key] = (time.time(), False)
        return False

# ── Runtime state ──────────────────────────────────────────────
trades_today:       dict[str, int]   = defaultdict(int)
circuit_breaker:    bool             = False
api_credit_exhausted: bool           = False  # set True on Anthropic 'credit balance too low' — halts trading, not silent HOLD
starting_equity:    float | None     = None
position_peaks:     dict[str, float] = {}
market_state:       str              = "BULL"
fear_active:        bool             = False
weak_sectors:       set              = set()
spy_change:         float            = 0.0
current_vix:        float | None     = None   # latest VIX reading — used for VIX-aware breakeven stop
systemic_derisk_active: bool         = False  # True when VIX severely elevated + multiple sectors weak at once

# ── Systemic de-risk thresholds ──────────────────────────────────
SYSTEMIC_DERISK_VIX              = 28   # VIX level considered "severe", above the existing FEAR_VIX=25
SYSTEMIC_DERISK_MIN_WEAK_SECTORS = 3    # 3+ simultaneously weak sectors = broad event, not rotation
SYSTEMIC_DERISK_MAX_LOSS_EXIT    = -0.005 # during systemic de-risk, allow weak-sector exit up to -0.5% loss
                                           # (Sep 10: was -2%; tiers are now -0.75%/-1.0%/-1.5%, so this stays
                                           # inside them — it only ever exits EARLIER/SMALLER than the hard stop would)
# GAP_RISK_THRESHOLD removed Aug 31 — the check could not fire until market
# open regardless (no code runs while market is closed), so it never had
# any head start over the ordinary composite-tiered stop loss, which would
# catch the same overnight gap on the exact same first post-open price tick.
# Redundant complexity with no genuine protective value; removed.
fund_cache:         dict             = {}        # {symbol: (timestamp, result)}
signal_cache:       list             = []        # ranked BUY signals from pre-market scan
signal_cache_time:  float            = 0.0       # when cache was built
signal_cache_date:  str              = ""        # date of last scan
last_rescan_time:   float            = 0.0       # last emergency rescan
closed_this_session: set             = set()     # symbols closed THIS 60s cycle — blocks immediate re-buy

# ── Re-entry cooldown tracking ─────────────────────────────────
# Option 1: Post-profit cooldown — 4hr block after +5% exit
# Option 3: Re-entry price gate — only re-enter if price pulled back 2%
# Option 5: Stop loss cooldown — 24hr block after -5% stop exit
reentry_cooldown:   dict[str, dict]  = {}
# {symbol: {"type": "profit"|"stop", "time": timestamp, "exit_price": float}}

PROFIT_COOLDOWN_SECS = 14400   # 4 hours after profit target exit
STOP_COOLDOWN_SECS   = 86400   # 24 hours after stop loss exit

# ── Strike system — repeated stop-loss failures escalate cooldown ──
# Problem observed: MU stopped out -5% on Jul 24, re-bought, stopped out
# -5% again on Jul 28. The 24hr cooldown resets the slate each time with
# no memory that this ticker just failed. A human would sit it out longer
# after 2+ failures in a short window — this gives the bot the same instinct.
STRIKE_WINDOW_SECS   = 7 * 86400   # look back 7 days for strike counting
STRIKE_COOLDOWNS     = {           # strikes within window → cooldown duration
    1: 86400,        # 1st stop loss this week  → 24hr  (unchanged)
    2: 3 * 86400,    # 2nd stop loss this week   → 72hr
    3: 14 * 86400,   # 3rd+ stop loss this week  → 14 days (effectively benched)
}
stop_loss_strikes: dict = {}   # {symbol: [timestamp, timestamp, ...]} — rolling log
PRICE_GATE_PCT       = 0.02    # must pull back 2% from exit price to re-enter

# ── News WebSocket state ───────────────────────────────────────
news_triggered:     dict[str, float] = {}        # {symbol: timestamp} — news-triggered tickers
news_queue:         list             = []         # pending news signals to process
held_symbols:       set              = set()       # currently-held position symbols, refreshed each
                                                     # main-loop cycle — lets the WebSocket thread check
                                                     # "is this ticker held?" without an API call per message
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

# ── Material adverse event tier (Sep 2026) ──────────────────────
# Root cause: AMZN was hit with an FTC + 22-state lawsuit alleging
# $20B+ in deceptive ad-auction pricing (Aug 31 2026) while HELD in
# the portfolio. The news WAS correctly classified BEARISH via the
# "lawsuit" keyword above, but process_news_queue() discarded it on
# sight because the ticker was already in positions — the news
# pipeline only ever triggered NEW buys, never reviewed EXISTING
# holdings. Separately, PLTR lost -5.92% overnight to a Google
# DeepMind product launch that was reported same-day in mainstream
# press but never reached the bot at all, since it wasn't phrased as
# any of the keywords above.
#
# This tier identifies events severe enough to warrant treating a
# HELD position's bearish news differently from routine sentiment —
# regulatory/legal action, credible new competitive threats, and
# executive-level shocks. It does not replace NEWS_BEARISH_KEYWORDS;
# it's checked in addition, to flag which bearish items should
# trigger an immediate re-evaluation of an existing position rather
# than just being logged.
NEWS_MATERIAL_ADVERSE_KEYWORDS = [
    "ftc sues", "ftc lawsuit", "antitrust", "doj sues",           # regulatory/legal
    "state attorneys general", "class action", "sec charges",
    "sec investigation", "criminal probe", "subpoena",
    "competing product", "competitor launch", "rival launches",   # competitive threats
    "unveiled", "enters the market", "encroach",
    "ceo resigns", "ceo steps down", "cfo resigns",               # leadership shocks
    "resigns amid", "fired amid", "ousted",
    "data breach", "hack", "cyberattack",                         # security incidents
    "halted trading", "trading suspended",                        # exchange actions
]

def is_material_adverse(headline: str, summary: str) -> bool:
    text = (headline + " " + summary).lower()
    return any(kw in text for kw in NEWS_MATERIAL_ADVERSE_KEYWORDS)


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
    Filters to curated tickers + news-triggered universe + currently-held
    positions (Sep 2026 fix — a held position outside the curated list
    could otherwise never have its news seen at all, and even for curated
    tickers this makes the "is this ticker held" state explicit downstream
    rather than implicit).
    """
    universe = set(CURATED_TICKERS) | set(news_triggered.keys()) | held_symbols
    matched  = [s for s in symbols if s in universe and s not in ETF_EXCLUSIONS]
    return matched

def on_news_message(ws, message):
    """
    Handles incoming news from Alpaca WebSocket stream.
    Sends subscribe after auth confirmation.
    Filters for high-value events on universe tickers.
    """
    global news_triggered, news_queue
    try:
        data = json.loads(message)
        if not isinstance(data, list):
            data = [data]

        for article in data:
            msg_type = article.get("T")

            # ── Auth/subscription confirmation ────────────────
            if msg_type == "success":
                if article.get("msg") == "connected":
                    log.info("📰 News WebSocket connection confirmed")
                elif article.get("msg") == "authenticated":
                    log.info("📰 News WebSocket authenticated — subscribing to all news")
                    ws.send(json.dumps({
                        "action": "subscribe",
                        "news":   ["*"],
                    }))
                continue

            if msg_type == "subscription":
                log.info(f"📰 News WebSocket subscribed: {article.get('news', [])}")
                continue

            if msg_type == "error":
                log.warning(f"📰 News WebSocket error message: {article.get('msg')}")
                continue

            if msg_type != "n":  # only news type messages
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

            material_adverse = sentiment == "BEARISH" and is_material_adverse(headline, summary)

            # Check cooldown — don't re-trigger same ticker within 1 hour
            # (material adverse events bypass the cooldown for HELD positions —
            # a held ticker shouldn't wait up to an hour to be reviewed after
            # regulatory/legal/competitive news, even if it triggered recently)
            now = time.time()
            for symbol in matched:
                last_trigger = news_triggered.get(symbol, 0)
                is_held      = symbol in held_symbols
                if now - last_trigger < NEWS_COOLDOWN and not (material_adverse and is_held):
                    continue

                news_triggered[symbol] = now
                news_queue.append({
                    "symbol":           symbol,
                    "headline":         headline,
                    "sentiment":        sentiment,
                    "timestamp":        now,
                    "material_adverse": material_adverse,
                })
                tag = " [MATERIAL ADVERSE]" if material_adverse else ""
                log.info(
                    f"📰 NEWS TRIGGER [{sentiment}]{tag} {symbol}: {headline[:80]}..."
                )

    except Exception as e:
        log.warning(f"News message error: {e}")

def on_news_open(ws):
    log.info("📰 News WebSocket connected — authenticating...")
    # Send auth first — subscribe is sent after auth confirmation in on_news_message
    ws.send(json.dumps({
        "action": "auth",
        "key":    ALPACA_KEY,
        "secret": ALPACA_SECRET,
    }))

def on_news_error(ws, error):
    log.warning(f"News WebSocket error: {error}")

def on_news_close(ws, close_status_code, close_msg):
    log.warning(f"News WebSocket closed: {close_status_code} {close_msg}")

def start_news_stream():
    """
    Starts news WebSocket in a background daemon thread.
    Auth is sent first; subscribe is sent after auth confirmation.
    Longer reconnect delay prevents connection limit errors on Render redeploy.
    """
    import ssl

    def run():
        # Initial delay — prevents connection limit errors when Render
        # restarts quickly (old process may not have closed yet)
        time.sleep(5)
        consecutive_failures = 0

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
                ws.run_forever(
                    ping_interval = 20,
                    ping_timeout  = 10,
                    sslopt        = {"cert_reqs": ssl.CERT_NONE},
                )
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                log.warning(f"News stream crashed: {e} — reconnecting...")

            # Exponential backoff — prevents hammering Alpaca on repeated failures
            delay = min(30 * consecutive_failures, 120)
            log.info(f"📰 News WebSocket reconnecting in {delay}s...")
            time.sleep(delay)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    log.info("📰 News WebSocket thread started")

def process_news_queue(positions: dict, account) -> bool:
    """
    Processes pending news signals from the queue.
    Called from main loop every cycle.
    Returns True if any news-triggered trade (buy OR exit) was placed.

    Sep 2026 fix: previously, ANY news item for an already-held symbol
    was discarded outright (`if symbol in positions: skip`) — this is
    what let AMZN ride an FTC/22-state lawsuit (Aug 31) down to a -4.25%
    loss uncaught, and PLTR miss a same-day Google DeepMind competitive
    product launch entirely. The news pipeline only ever triggered NEW
    buys; it never reviewed EXISTING holdings on bad news.

    Held positions now get a re-evaluation path: bearish or material-
    adverse news re-runs the fundamental check (not the stale entry-time
    score) and can trigger an early exit via close_position(), separate
    from the normal composite-tiered stop-loss/trailing logic.
    """
    global news_queue, signal_cache

    if not news_queue:
        return False

    if api_credit_exhausted:
        log.warning(
            f"⏸ {len(news_queue)} news trigger(s) queued but Anthropic API credit "
            f"is exhausted — holding queue rather than processing blind (was: silently "
            f"treated as HOLD/SELL, losing real signals like today's AMD/NVDA news)"
        )
        return False  # queue is preserved — not cleared — will be retried once credit restored

    traded = False
    to_process = news_queue.copy()
    news_queue.clear()

    for item in to_process:
        symbol           = item["symbol"]
        headline         = item["headline"]
        sentiment        = item["sentiment"]
        material_adverse = item.get("material_adverse", False)

        # ── Held position review (new path) ────────────────────
        if symbol in positions:
            if sentiment != "BEARISH":
                log.info(f"📰 {symbol} held — {sentiment} news, no action needed")
                continue

            if not is_market_open():
                log.info(f"📰 {symbol} held, bearish news queued — market closed")
                news_queue.append(item)
                continue

            urgency = "MATERIAL ADVERSE" if material_adverse else "bearish"
            log.warning(
                f"📰 HELD POSITION REVIEW [{urgency}] {symbol}: {headline[:70]}... "
                f"— re-running fundamentals (not trusting stale entry score)"
            )

            # Force a fresh fundamental read — the entry-time score could be
            # days old and doesn't know about news that just broke.
            if symbol in fund_cache:
                del fund_cache[symbol]
            spy_chg = spy_change or 0.0
            result  = compute_signal(symbol, spy_chg)

            pos = positions[symbol]
            try:
                pnl_pct = float(pos.unrealized_plpc)
            except (AttributeError, TypeError, ValueError):
                pnl_pct = 0.0

            # Exit if fresh fundamentals confirm deterioration (SELL signal
            # or fundScore has turned negative), OR if it's a material
            # adverse event regardless of composite (regulatory/legal/
            # competitive shocks deserve a lower bar than routine sentiment
            # — same "hard veto over blended score" principle as the SPAC
            # and structural-risk checks).
            should_exit = False
            exit_note   = ""
            if result and result.get("signal") == "SELL":
                should_exit = True
                exit_note   = f"fresh fundamentals turned SELL (composite={result['composite']:.2f})"
            elif material_adverse:
                should_exit = True
                exit_note   = "material adverse event (regulatory/legal/competitive)"

            if should_exit:
                reason = f"NEWS_ADVERSE ({exit_note}) — {headline[:60]}"
                if close_position(symbol, pnl_pct, reason):
                    traded = True
                    log.warning(f"📰 [SELL] {symbol} exited on adverse news review — {exit_note}")
            else:
                log.info(
                    f"📰 {symbol} held, bearish news reviewed but fundamentals still "
                    f"intact — holding (composite={result['composite']:.2f} if scored)"
                    if result else f"📰 {symbol} held, bearish news reviewed — holding"
                )
            continue

        # ── New-buy candidate path (existing behaviour) ────────
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

        if result and result["signal"] == "BUY" and result["confidence"] >= MIN_CONFIDENCE and result["composite"] >= MIN_COMPOSITE:
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
    """
    Record every trade outcome for 90-day audit AND signal-attribution
    analysis — joins the entry-time composite/confidence/ta/fund scores
    (captured in entry_signals at buy time) with the realised outcome.
    This is what makes it possible to answer "does a high composite
    score actually predict a winning trade?" instead of guessing.
    """
    journal = load_journal()
    if symbol not in journal:
        journal[symbol] = []

    entry = entry_signals.pop(symbol, {})  # consume + remove — one snapshot per trade
    save_entry_signals()

    journal[symbol].append({
        "date":        datetime.now(ET).strftime("%Y-%m-%d"),
        "pnl":         round(pnl_pct * 100, 2),
        "reason":      exit_reason,
        "composite":   entry.get("composite"),
        "confidence":  entry.get("confidence"),
        "ta_score":    entry.get("ta_score"),
        "fund_score":  entry.get("fund_score"),
        "sector":      entry.get("sector"),
        "kelly_pct":   entry.get("kelly_pct"),
    })
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(journal, f)
    except Exception as e:
        log.warning(f"Failed to save journal: {e}")

def analyse_signal_attribution():
    """
    Answers: 'does a high composite/confidence score actually predict
    a winning trade, or is TA/fundamentals just noise?'

    Groups all journalled trades (that have entry-signal data attached)
    into composite-score buckets and reports win rate + avg P&L per
    bucket. Call this manually or on a schedule once enough trades
    have accumulated with the new entry_signals tracking (old trades
    won't have this data — only trades closed AFTER this fix shipped).
    """
    journal = load_journal()
    all_trades = [t for trades in journal.values() for t in trades if t.get("composite") is not None]

    if len(all_trades) < 10:
        log.info(
            f"Signal attribution: only {len(all_trades)} trades with entry-signal "
            f"data so far — need 10+ for a meaningful read. (Old trades before this "
            f"fix don't have composite/confidence attached.)"
        )
        return

    def bucket(trades, lo, hi):
        b = [t for t in trades if lo <= t["composite"] < hi]
        if not b:
            return None
        wins = sum(1 for t in b if t["pnl"] > 0)
        return {
            "n": len(b),
            "win_rate": wins / len(b) * 100,
            "avg_pnl": sum(t["pnl"] for t in b) / len(b),
        }

    log.info("=" * 60)
    log.info("SIGNAL ATTRIBUTION — does composite score predict outcome?")
    log.info("=" * 60)
    for lo, hi, label in [(3.0, 4.0, "3.0-4.0 (marginal)"),
                           (4.0, 5.5, "4.0-5.5 (normal)"),
                           (5.5, 100, "5.5+ (high conviction)")]:
        r = bucket(all_trades, lo, hi)
        if r:
            log.info(f"  {label}: {r['n']} trades, {r['win_rate']:.0f}% win rate, avg P&L {r['avg_pnl']:+.2f}%")
        else:
            log.info(f"  {label}: no trades yet")

    # Same breakdown by confidence
    conf_trades = [t for t in all_trades if t.get("confidence") is not None]
    for lo, hi, label in [(85, 88, "85-88% conf"), (88, 92, "88-92% conf"), (92, 101, "92%+ conf")]:
        b = [t for t in conf_trades if lo <= t["confidence"] < hi]
        if b:
            wins = sum(1 for t in b if t["pnl"] > 0)
            log.info(f"  {label}: {len(b)} trades, {wins/len(b)*100:.0f}% win rate, avg P&L {sum(t['pnl'] for t in b)/len(b):+.2f}%")
    log.info("=" * 60)

def run_90_day_audit():
    """
    Audits curated tickers every 90 days against four criteria:
    1. Volume & Liquidity — has institutional volume dropped?
    2. Strategy Alignment — does stock still respect TA setups?
    3. Personal Performance — negative win rate over last 90 days?
    4. Universe-level AI concentration — has the curated list itself
       drifted toward over-concentration in AI-correlated sectors?
       (Added Aug 2026 alongside the live AI-correlated position cap —
       that cap limits how many AI-correlated positions can be HELD at
       once, but doesn't address the curated LIST itself skewing further
       AI-heavy over time as tickers get added. This is the periodic
       check on the universe design, not the runtime position count.)
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

    # ── Criterion 4: Universe-level AI concentration ────────────
    ai_count    = sum(1 for s in CURATED_TICKERS if SECTOR_MAP.get(s) in AI_CORRELATED_SECTORS)
    total_count = len(CURATED_TICKERS)
    ai_share    = ai_count / total_count if total_count else 0
    log.info("-" * 60)
    log.info(f"UNIVERSE COMPOSITION: {ai_count}/{total_count} curated tickers "
             f"({ai_share*100:.0f}%) are AI-correlated (semis/tech/software/cyber)")
    if ai_share > UNIVERSE_AI_SHARE_WARN:
        log.warning(
            f"⚠️  Curated universe is {ai_share*100:.0f}% AI-correlated — above the "
            f"{UNIVERSE_AI_SHARE_WARN*100:.0f}% review threshold. The live "
            f"MAX_AI_CORRELATED_POSITIONS cap limits concurrent exposure, but a "
            f"universe this skewed means most signals scored highly will be in this "
            f"bucket, making the cap bind often. Consider adding non-AI-correlated "
            f"tickers to CURATED_TICKERS to give the scanner more to choose from."
        )

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
    "ISRG":"healthcare","CPRX":"healthcare",
    "XOM":"energy","CVX":"energy","COP":"energy","SLB":"energy",
    "GEV":"industrials","CAT":"industrials","RTX":"industrials","HON":"industrials",
    "COST":"consumer","WMT":"consumer","MCD":"consumer","FDX":"consumer",
    "DECK":"consumer","SPOT":"media","NFLX":"media","UBER":"consumer",
}

# ── AI-correlated concentration cap (Aug 2026) ──────────────────
# The per-sector cap (2 max) treats "semis", "tech", "software", "cyber"
# as independent buckets — but in a genuine AI-sector selloff these move
# in near-lockstep (NVDA, MSFT, PLTR, CRWD are all "the AI trade" despite
# different sector labels). A bot could hold 2 semis + 2 tech + 2 software
# + 2 cyber = 8 of 10 positions all riding the same underlying bet with
# no guardrail noticing, because each individual sector count looks fine.
# This is a SEPARATE, stricter cap layered on top of the per-sector one.
AI_CORRELATED_SECTORS  = {"semis", "tech", "software", "cyber"}
MAX_AI_CORRELATED_POSITIONS = 4  # combined cap across all AI-correlated sectors
UNIVERSE_AI_SHARE_WARN = 0.45    # if 45%+ of the curated LIST is AI-correlated,
                                  # flag it in the 90-day audit for manual review

def count_ai_correlated_positions(positions_or_symbols) -> int:
    """Counts how many current positions fall in an AI-correlated sector,
    regardless of which specific sector label they carry."""
    return sum(
        1 for s in positions_or_symbols
        if SECTOR_MAP.get(s) in AI_CORRELATED_SECTORS
    )

# ── Alpaca client ──────────────────────────────────────────────
trade_client = TradingClient(
    api_key=ALPACA_KEY, secret_key=ALPACA_SECRET,
    paper=True, url_override=ALPACA_BASE_URL,
)
ai_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

def safe_claude_call(**kwargs):
    """
    Thin wrapper around ai_client.messages.create() that detects
    Anthropic credit exhaustion specifically (vs a generic transient
    error) and sets a global flag instead of letting every caller
    silently degrade to 'HOLD/SELL — no action'.

    Root cause fixed Aug 7: the bot ran out of API credit mid-session.
    Every subsequent Claude call failed with a 400, and every caller's
    bare except-clause swallowed it as a neutral/no-signal result —
    which is NOT neutral, it's unknown. News-triggered buys on real
    bullish headlines (AMD, NVDA, GOOG, META, MSFT) were silently
    skipped as 'HOLD/SELL' for hours with no alerting.
    """
    global api_credit_exhausted
    try:
        response = ai_client.messages.create(**kwargs)
        if api_credit_exhausted:
            api_credit_exhausted = False  # recovered — clear the flag
            log.info("✅ Anthropic API credit restored — resuming normal operation")
        return response
    except anthropic.APIStatusError as e:
        if e.status_code == 400 and "credit balance is too low" in str(e).lower():
            if not api_credit_exhausted:
                api_credit_exhausted = True
                log.critical(
                    "🚨 ANTHROPIC API CREDIT EXHAUSTED — all Claude-dependent "
                    "signals (fundamentals, earnings checks, news scoring) will "
                    "be BLOCKED (not silently skipped) until credit is restored. "
                    "Go to console.anthropic.com → Plans & Billing."
                )
        raise

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
    global market_state, fear_active, weak_sectors, spy_change, current_vix

    vix_now  = get_vix_level()
    vixy_chg = get_quote_change("VIXY")
    current_vix = vix_now  # stored globally — used by check_profit_targets for VIX-aware breakeven stop

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

    # ── Systemic de-risking (Aug 2026) ──────────────────────────
    # VIX≥25 + normal sector cap alone doesn't distinguish "one sector
    # rotating out" from "everything crashing together" (e.g. an AI-bubble
    # burst hitting semis+tech+software+cyber+consumer simultaneously).
    # If VIX is severely elevated AND 3+ sectors are weak at once, this is
    # a systemic event, not a rotation — force-reduce exposure regardless
    # of individual stop levels rather than waiting for each position's
    # own stop to fire one at a time.
    global systemic_derisk_active
    systemic_derisk_active = (
        current_vix is not None
        and current_vix >= SYSTEMIC_DERISK_VIX
        and len(weak_sectors) >= SYSTEMIC_DERISK_MIN_WEAK_SECTORS
    )
    if systemic_derisk_active:
        log.critical(
            f"🚨 SYSTEMIC DE-RISK ACTIVE — VIX {current_vix:.1f} + "
            f"{len(weak_sectors)} sectors weak simultaneously ({', '.join(weak_sectors)}). "
            f"Treating as broad market event, not sector rotation. "
            f"New buys blocked; existing positions' exits will run more aggressively."
        )

    log.info(f"Market state: {market_state} | Fear: {fear_active} | Weak sectors: {weak_sectors or 'none'}")

def get_stop_loss() -> float:
    # BEAR mode: tightest stop of all. Was -2% when the tiers were
    # -3/-4/-5%; with tiers now -0.75/-1.0/-1.5% that would have been
    # LOOSER than every tier, so it's pinned to the tightest tier.
    return -0.0075 if market_state == "BEAR" else STOP_LOSS

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
        response = safe_claude_call(
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
        response = safe_claude_call(
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

def check_structural_risk(symbol: str, today: str) -> tuple[bool, str]:
    """
    Generalized hard-veto check: does this instrument have a structural
    reason its price may not reflect normal business fundamentals?

    Root cause Aug 21: RFAI/RFAIU (a SPAC) spiked +230-473% on a merger
    vote and cost a combined -$4,788 realised loss because its extreme
    TA/momentum score outweighed near-zero fundamentals in the composite
    blend. The SPAC name-check (is_likely_spac) fixed that ONE case, but
    the same failure mode applies to a wider category: any instrument
    where price action is decoupled from normal operating fundamentals
    (recent IPO with thin float, pending/announced M&A trading on deal
    odds, reverse stock split, recently halted, SPAC unit/right/warrant,
    thinly-traded ADR). Rather than add a new special-case filter each
    time one of these bites, this asks Claude directly and treats any
    "yes" as a hard veto — independent of composite score, same pattern
    as check_earnings_proximity but for structural rather than timing risk.

    Returns (has_structural_risk, reason).
    Cached 24h per symbol — same TTL as the earnings check.
    """
    cache_key = f"structural_risk_{symbol}"
    if cache_key in fund_cache:
        cached_time, cached_result = fund_cache[cache_key]
        if time.time() - cached_time < 86400:
            return cached_result

    try:
        prompt = (
            f"For the stock ticker {symbol} as of {today}, is there any structural "
            "reason its current price may NOT reflect normal business fundamentals? "
            "Specifically check: (1) is it a SPAC, blank-check company, or a SPAC "
            "unit/right/warrant; (2) did it IPO within the last 180 days with a very "
            "small public float; (3) is it currently the target of an announced-but-"
            "not-closed M&A deal, trading on deal-completion odds rather than "
            "fundamentals; (4) did it recently do a reverse stock split; (5) was "
            "trading recently halted for volatility (LULD circuit breaker); (6) is it "
            "a thinly-traded foreign ADR with very low US daily volume. "
            'Return ONLY this JSON with NO other text: '
            '{"structural_risk":false,"reason":""} '
            "Set structural_risk to true if ANY of the above applies, and give a "
            "short reason (under 10 words)."
        )
        response = safe_claude_call(
            model="claude-sonnet-4-5",
            max_tokens=80,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        si, ei = text.find("{"), text.rfind("}")
        if si == -1:
            fund_cache[cache_key] = (time.time(), (False, ""))
            return False, ""
        result = json.loads(text[si:ei+1])
        has_risk = bool(result.get("structural_risk", False))
        reason   = result.get("reason") or ""
        if has_risk:
            log.warning(f"  {symbol}: structural risk flagged — {reason} — blocking")
        fund_cache[cache_key] = (time.time(), (has_risk, reason))
        return has_risk, reason
    except Exception as e:
        log.warning(f"Structural risk check failed for {symbol}: {e}")
        return False, ""

def compute_signal(symbol: str, spy_chg: float = 0.0, prefetched_ta: dict | None = None) -> dict | None:
    # Skip known ETFs
    if symbol in ETF_EXCLUSIONS:
        return None

    # Secondary ETF check — catches new leveraged ETFs not in exclusion list (e.g. CRDU)
    if is_likely_etf(symbol):
        ETF_EXCLUSIONS.add(symbol)  # add to exclusion list for this session
        return None

    # SPAC / blank-check company check — hard veto, independent of composite
    # score. Root cause Aug 21: RFAI (RF Acquisition Corp II) spiked
    # +230-473% on a merger-vote approval and cleared the composite floor
    # via an extreme TA score despite having "no significant operations" —
    # cost a $712 realised loss on a ~2-minute round trip. This blocks the
    # ticker before compute_signal ever runs TA/fundamentals on it.
    if is_likely_spac(symbol):
        return None

    # Generalized structural risk check — catches the WIDER category the
    # SPAC check only partially covers (recent IPO/thin float, pending M&A
    # deal-odds pricing, reverse split, recent volatility halt, thinly-
    # traded ADR, SPAC units/rights/warrants the name-check might miss).
    # One Claude call, cached 24h, same cost profile as the earnings check.
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    has_structural_risk, risk_reason = check_structural_risk(symbol, today_str)
    if has_structural_risk:
        return None

    # Use prefetched TA if provided (avoids double-fetching during pre-market scan)
    ta = prefetched_ta if prefetched_ta is not None else fetch_technicals(symbol)

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

    if api_credit_exhausted:
        log.critical(
            "🚨 Skipping pre-market scan — Anthropic API credit exhausted. "
            "Cache will stay empty/stale until credit is restored and the next "
            "scan window (or emergency rescan) runs. Check console.anthropic.com."
        )
        return

    now_et = datetime.now(ET)
    today  = now_et.strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("PRE-MARKET SCAN STARTING")
    log.info(f"  Time: {now_et.strftime('%A %Y-%m-%d %H:%M ET')}")
    log.info(f"  Universe: {len(CURATED_TICKERS)} curated + up to 24 dynamic (8+8+8) = max 55 tickers")
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

        # ── TA pre-filter — skip Claude if TA is clearly bearish ──
        # Saves ~12 minutes by avoiding Claude calls on obvious non-signals
        # Fetch technicals first (fast — Cloudflare Worker, <1s)
        ta = fetch_technicals(symbol)
        if ta:
            ta_score = ta.get("taScore", 0)
            # Skip Claude if TA score is weak (below all MAs, bearish MACD)
            # Threshold 1.5 means: some positive technical signal required
            # At 82% live win rate this filter should be additive not subtractive
            if ta_score < 1.5:
                log.info(f"  {symbol}: taScore={ta_score:.1f} — TA too weak, skipping Claude")
                continue  # no sleep needed — skipping Claude call

        result = compute_signal(symbol, spy_chg, prefetched_ta=ta)
        if result:
            all_scored.append(result)
            if result["signal"] == "BUY" and result["confidence"] >= MIN_CONFIDENCE and result["composite"] >= MIN_COMPOSITE:
                candidates.append(result)
                ipo_tag = " [IPO]" if result.get("ipo_mode") else ""
                log.info(
                    f"  ✅ {symbol}: composite={result['composite']:.2f}, "
                    f"confidence={result['confidence']}%{ipo_tag} — {result['thesis']}"
                )
        time.sleep(3)   # Tier 1 = 50 RPM (1 req/1.2s min). sleep(3) = 4x safety margin. Was 20s.

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
    save_scan_state()  # persist immediately — a redeploy right after this point must NOT re-scan

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
    - Mon-Fri: dynamic start = 9:30am minus scan duration minus buffer
      Scan duration estimate: ~8 min (TA pre-filter + sleep 3s on 55 tickers)
      Buffer: 10 min safety margin
      → Scan starts at ~9:12am ET, finishes just before market open

    This ensures signals are as fresh as possible at 9:30am open.
    Old approach (6am scan) left cache stale for 3+ hours.

    NEVER scans:
    - During market hours (9:30am-4pm ET) — cache only
    - After 4pm ET — wait for next morning
    - Weekends outside Sunday 8-10pm
    - On bot restart mid-session (even with empty cache)
    """
    now_et = datetime.now(ET)
    today  = now_et.strftime("%Y-%m-%d")
    hour   = now_et.hour
    minute = now_et.minute
    day    = now_et.weekday()  # 0=Mon, 6=Sun

    # ── Scan duration config ───────────────────────────────────
    ESTIMATED_SCAN_MINUTES = 4    # measured Jul 2 — TA pre-filter + sleep(3s) = ~4min actual
    BUFFER_MINUTES         = 6    # safety margin before open
    TOTAL_LEAD_MINUTES     = ESTIMATED_SCAN_MINUTES + BUFFER_MINUTES  # 10 min

    # Market open = 9:30am ET
    # Scan should start at: 9:30 - 10 min = 9:20am ET
    market_open_minutes   = 9 * 60 + 30          # 570
    scan_start_minutes    = market_open_minutes - TOTAL_LEAD_MINUTES  # 560 = 9:20am
    now_minutes           = hour * 60 + minute

    # ── Step 1: Time gate ──────────────────────────────────────
    in_window = False

    # Sunday 8pm-10pm ET — prepare for Monday
    if day == 6 and 20 <= hour < 22:
        in_window = True

    # Weekday: dynamic window from scan_start_minutes to 9:29am
    elif 0 <= day <= 4:
        market_open = (hour == 9 and minute >= 30) or hour >= 10
        if not market_open and now_minutes >= scan_start_minutes:
            in_window = True

    if not in_window:
        return False

    # ── Step 2: Already scanned today — don't repeat ──────────
    if signal_cache_date == today and signal_cache:
        return False

    # In window + cache stale → scan
    scan_start_str = f"{scan_start_minutes // 60}:{scan_start_minutes % 60:02d}am ET"
    log.info(f"Pre-market scan window active (start={scan_start_str}, "
             f"est. duration={ESTIMATED_SCAN_MINUTES}min, buffer={BUFFER_MINUTES}min)")
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

COOLDOWN_FILE = "/tmp/reentry_cooldowns.json"
STRIKES_FILE  = "/tmp/stop_loss_strikes.json"
ENTRY_SIGNALS_FILE = "/tmp/entry_signals.json"
SCAN_STATE_FILE = "/tmp/scan_state.json"

# ── Scan-gate persistence (Sep 2026) ──────────────────────────────
# Root cause: signal_cache, signal_cache_date, and last_rescan_time
# were plain in-memory globals with NO persistence — same class of bug
# as the earlier entry_composite/NVDA and cooldown/AMZN incidents.
# should_run_premarket_scan()'s "already scanned today, don't repeat"
# check (signal_cache_date == today and signal_cache) relies entirely
# on this state surviving. A Render redeploy during or shortly after
# the scan window wipes it back to signal_cache_date="" and
# signal_cache=[] — the gate then correctly (by its own logic) sees
# "haven't scanned today" and fires a brand new scan, burning a full
# round of Claude API calls that had already just completed moments
# before the upload. Persisting this the same way entry_signals and
# reentry_cooldown already are closes the gap.
def save_scan_state():
    try:
        with open(SCAN_STATE_FILE, "w") as f:
            json.dump({
                "signal_cache_date": signal_cache_date,
                "signal_cache":      signal_cache,
                "last_rescan_time":  last_rescan_time,
            }, f)
    except Exception as e:
        log.warning(f"Failed to save scan state: {e}")

def load_scan_state():
    global signal_cache, signal_cache_date, last_rescan_time
    try:
        with open(SCAN_STATE_FILE) as f:
            data = json.load(f)
        today = datetime.now(ET).strftime("%Y-%m-%d")
        if data.get("signal_cache_date") == today:
            signal_cache      = data.get("signal_cache", [])
            signal_cache_date = data.get("signal_cache_date", "")
            last_rescan_time  = data.get("last_rescan_time", 0.0)
            log.info(
                f"Restored today's scan state — {len(signal_cache)} cached signal(s), "
                f"scan already completed today (redeploy will NOT trigger a fresh scan)"
            )
        else:
            log.info("Restored scan state is from a prior day — will scan fresh at next window")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Failed to load scan state: {e}")

entry_signals: dict = {}   # {symbol: {composite, confidence, ta_score, fund_score, sector, ...}}
                            # captured at buy time, consumed + cleared at exit time
                            # so the journal can answer "did high-composite trades win more?"

def save_entry_signals():
    try:
        with open(ENTRY_SIGNALS_FILE, "w") as f:
            json.dump(entry_signals, f)
    except Exception as e:
        log.warning(f"Failed to save entry signals: {e}")

def load_entry_signals():
    global entry_signals
    try:
        with open(ENTRY_SIGNALS_FILE) as f:
            entry_signals = json.load(f)
        if entry_signals:
            log.info(f"Restored entry signal snapshots for {len(entry_signals)} open position(s)")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Failed to load entry signals: {e}")

def save_cooldowns():
    """Persist re-entry cooldowns so a bot restart doesn't forget a 24hr
    stop-loss block. Note: /tmp survives process crashes but NOT Render
    redeploys (new container). Best-effort protection."""
    try:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(reentry_cooldown, f)
        with open(STRIKES_FILE, "w") as f:
            json.dump(stop_loss_strikes, f)
    except Exception as e:
        log.warning(f"Failed to save cooldowns: {e}")

def load_cooldowns():
    global reentry_cooldown, stop_loss_strikes
    try:
        with open(COOLDOWN_FILE) as f:
            data = json.load(f)
        now = time.time()
        # Drop already-expired time cooldowns; keep price gates < 48h old
        reentry_cooldown = {
            sym: cd for sym, cd in data.items()
            if (cd.get("expires", 0) > now) or (now - cd.get("time", 0) < 48 * 3600)
        }
        if reentry_cooldown:
            log.info(f"Restored {len(reentry_cooldown)} re-entry cooldown(s): {list(reentry_cooldown.keys())}")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Failed to load cooldowns: {e}")

    try:
        with open(STRIKES_FILE) as f:
            data = json.load(f)
        now = time.time()
        # Prune strikes older than the rolling window on load
        stop_loss_strikes = {
            sym: [t for t in times if now - t < STRIKE_WINDOW_SECS]
            for sym, times in data.items()
        }
        stop_loss_strikes = {k: v for k, v in stop_loss_strikes.items() if v}
        flagged = {k: len(v) for k, v in stop_loss_strikes.items() if len(v) >= 2}
        if flagged:
            log.info(f"Restored stop-loss strike history — repeat offenders: {flagged}")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"Failed to load strikes: {e}")

def register_reentry_cooldown(symbol: str, exit_reason: str, exit_price: float):
    """
    Registers a re-entry cooldown after a position closes.

    HARD_CEILING   → 4hr cooldown + 2% price gate (the +60% forced exit — replaces PROFIT_TARGET)
    STOP_LOSS      → 24hr cooldown + 2% price gate
    WEAK_SECTOR    → 2hr cooldown + 2% price gate (sector still weak — don't re-enter)
    TRAILING       → 2% price gate only (no time cooldown — partial win)
    BREAKEVEN_STOP → 1hr cooldown + 2% price gate (momentum stalled)
    DRAG           → 1hr cooldown (flat position freed up)

    CRITICAL: All exits also add to closed_this_session set.
    Bot will NOT re-buy any symbol closed in the current 60s cycle.
    """
    global reentry_cooldown, closed_this_session
    now = time.time()

    # Always add to closed_this_session — prevents same-cycle re-entry
    closed_this_session.add(symbol)

    # Also remove from signal cache immediately — don't redeploy a just-closed position
    global signal_cache
    signal_cache = [s for s in signal_cache if s["symbol"] != symbol]

    if "HARD_CEILING" in exit_reason or "PROFIT_TARGET" in exit_reason:
        reentry_cooldown[symbol] = {
            "type":       "profit",
            "time":       now,
            "exit_price": exit_price,
            "expires":    now + PROFIT_COOLDOWN_SECS,
        }
        log.info(
            f"  ⏳ {symbol}: profit cooldown — "
            f"4hr block + price gate ${exit_price*(1-PRICE_GATE_PCT):.2f}"
        )
    elif "STOP_LOSS" in exit_reason:
        # ── Strike tracking — escalate cooldown on repeated failures ──
        global stop_loss_strikes
        history = stop_loss_strikes.get(symbol, [])
        history = [t for t in history if now - t < STRIKE_WINDOW_SECS]  # prune old strikes
        history.append(now)
        stop_loss_strikes[symbol] = history
        strike_count = len(history)

        cooldown_secs = STRIKE_COOLDOWNS.get(strike_count, STRIKE_COOLDOWNS[3])
        reentry_cooldown[symbol] = {
            "type":       "stop",
            "time":       now,
            "exit_price": exit_price,
            "expires":    now + cooldown_secs,
        }
        cooldown_days = cooldown_secs / 86400
        log.info(
            f"  ⏳ {symbol}: stop loss cooldown — strike {strike_count} in past 7 days "
            f"→ {cooldown_days:.0f}-day block (escalating penalty for repeat failures)"
        )
    elif "WEAK_SECTOR" in exit_reason:
        reentry_cooldown[symbol] = {
            "type":       "weak_sector",
            "time":       now,
            "exit_price": exit_price,
            "expires":    now + 7200,  # 2hr — sector may recover by then
        }
        log.info(
            f"  ⏳ {symbol}: weak sector cooldown — 2hr block + price gate"
        )
    elif "BREAKEVEN" in exit_reason or "DRAG" in exit_reason:
        reentry_cooldown[symbol] = {
            "type":       "breakeven",
            "time":       now,
            "exit_price": exit_price,
            "expires":    now + 3600,  # 1hr — momentum stalled
        }
        log.info(
            f"  ⏳ {symbol}: breakeven cooldown — 1hr block + price gate"
        )
    else:
        # Trailing exits — price gate only, no time cooldown
        reentry_cooldown[symbol] = {
            "type":       "trailing",
            "time":       now,
            "exit_price": exit_price,
            "expires":    0,
        }

    save_cooldowns()  # persist across restarts (best-effort)

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

    # Insufficient buying power cooldown (15min) — Sep 2026 fix for the
    # NVDA retry-loop incident (bot hammered buy attempts every ~1s with
    # shrinking quantities, all failing identically). Checked explicitly
    # by type, same as profit/stop above — falling through to the price
    # gate below would use exit_price=0, which happens to always block
    # but for the wrong reason and ignores the actual 15min expiry.
    if cdtype == "insufficient_funds" and expires > 0 and now < expires:
        remaining = (expires - now) / 60
        return False, f"insufficient funds cooldown ({remaining:.1f}min remaining)"

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

        # ── Settlement guard ─────────────────────────────────────
        # Wait 2 seconds after close before allowing re-buy of same symbol.
        # Prevents race condition where bot re-buys before close settles,
        # causing Alpaca to execute as a short sell instead of long.
        time.sleep(2)

        # Register re-entry cooldown based on exit type
        register_reentry_cooldown(symbol, exit_reason, current_price)
        return True
    except Exception as e:
        log.error(f"Failed to close {symbol}: {e}")
        return False

def place_buy(symbol: str, qty: int, composite: float | None = None) -> bool:
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

        # ── Layer 3: SPAC guard (defense in depth) ────────────
        # compute_signal() already blocks SPACs before scoring, but this
        # catches any stale cache entry built before this fix deployed,
        # or any path that bypasses compute_signal entirely.
        if is_likely_spac(symbol):
            log.warning(f"[SKIP] {symbol} is a SPAC/blank-check company — blocked at buy time")
            return False

        # ── Layer 4: Structural risk guard (cache-only, no extra API call) ──
        # compute_signal() already ran check_structural_risk() this scan
        # cycle and cached the result (24h TTL) — re-check the cache here
        # as a defense-in-depth backstop without paying for a second
        # Claude+web-search call per trade.
        cached = fund_cache.get(f"structural_risk_{symbol}")
        if cached and cached[1][0]:
            log.warning(f"[SKIP] {symbol} structural risk ({cached[1][1]}) — blocked at buy time")
            return False

        # ── Durable composite storage (Aug 24 fix) ──────────────
        # Root cause: NVDA (bought Aug 20) had its entry_composite lost
        # when the bot redeployed 3+ times before it was sold Aug 24 —
        # entry_signals lives in /tmp, which is wiped on every Render
        # redeploy (new container). The stop-loss tiering silently fell
        # back to the loosest -5% stop with no entry data, and NVDA rode
        # the full -5.03% instead of the -3%/-4% its actual composite
        # score should have used.
        #
        # Fix: encode the composite score into Alpaca's own
        # client_order_id field. This is stored in Alpaca's database,
        # not the bot's filesystem — it survives every redeploy because
        # it never lives in the container at all. Format: "c{composite}"
        # e.g. composite 5.40 -> "c540" (2 decimal places, dot removed,
        # kept short since client_order_id has a 128-char limit).
        client_id = None
        if composite is not None:
            client_id = f"c{round(composite*100):04d}_{int(time.time())}"[:48]

        order = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            position_intent="buy_to_open",   # explicitly prevents short sell execution
            client_order_id=client_id,
        )
        trade_client.submit_order(order)
        today_key = datetime.now(ET).strftime("%Y-%m-%d")
        trades_today[today_key] = trades_today.get(today_key, 0) + 1
        log.info(f"[BUY] {qty}x {symbol} — trades today: {trades_today.get(today_key,0)}/{MAX_TRADES_DAY}")
        return True
    except Exception as e:
        log.error(f"Failed to buy {symbol}: {e}")
        # ── Permanent failure detection ────────────────────────
        # "asset is not active" (code 40010001) means the ticker is
        # delisted/halted/suspended — retrying every 60s cycle forever
        # wastes time and clutters logs (observed: NUVL retried 10+
        # times across 10 minutes on Aug 21 before being caught here).
        # Remove it from the cache immediately so it isn't retried
        # again until the next pre-market scan re-evaluates it fresh.
        err_str = str(e).lower()
        if "not active" in err_str or "40010001" in err_str:
            global signal_cache
            before = len(signal_cache)
            signal_cache = [s for s in signal_cache if s["symbol"] != symbol]
            if len(signal_cache) < before:
                log.warning(
                    f"  {symbol}: asset not active — removed from cache "
                    f"(will not retry until next scan)"
                )

        # "insufficient buying power" — Sep 2026 fix. Root cause: the bot
        # was observed hammering NVDA buy attempts every ~1 second for 30+
        # seconds straight, shrinking the quantity each time and retrying
        # immediately, always failing the same way. deploy_from_cache()
        # recalculates qty fresh from account.cash every cycle — if
        # Alpaca's reported cash doesn't match what its own order engine
        # validates against at submission time (e.g. a pending order or
        # margin calc lag), every recalculated qty fails identically,
        # burning API calls and cluttering logs with zero progress.
        #
        # Fix: register a short re-entry cooldown on insufficient-funds
        # failures specifically, same mechanism as a stop-loss/profit
        # cooldown, so this ticker is skipped for a cooling-off period
        # instead of being retried every single cycle.
        elif "insufficient buying power" in err_str or "insufficient_buying_power" in err_str:
            global reentry_cooldown
            reentry_cooldown[symbol] = {
                "type":       "insufficient_funds",
                "time":       time.time(),
                "exit_price": 0,
                "expires":    time.time() + 900,  # 15 min cooldown — enough for cash/margin state to settle
            }
            save_cooldowns()
            log.warning(
                f"  {symbol}: insufficient buying power — 15min cooldown applied "
                f"(was retrying every cycle with no progress)"
            )
        return False

# ══════════════════════════════════════════════════════════════
# RISK CONTROLS
# ══════════════════════════════════════════════════════════════

def is_paused() -> bool:
    return os.environ.get("PAUSED", "false").strip().lower() in ("true", "1", "yes")

def trades_today_count() -> int:
    return trades_today.get(datetime.now(ET).strftime("%Y-%m-%d"), 0)

drawdown_anchor_date: str = ""  # trading day the drawdown anchor was set

def check_drawdown(account) -> bool:
    """
    Daily drawdown circuit breaker.
    - Anchor equity resets each new trading day (uses last_equity = prior close)
    - Breaker auto-resets on a new day, so one bad day doesn't disable
      the bot forever (old behaviour required a manual restart)
    """
    global circuit_breaker, starting_equity, drawdown_anchor_date
    today = datetime.now(ET).strftime("%Y-%m-%d")

    # New trading day → re-anchor and clear breaker
    if drawdown_anchor_date != today:
        drawdown_anchor_date = today
        try:
            starting_equity = float(account.last_equity)  # prior close — clean daily anchor
        except Exception:
            starting_equity = float(account.equity)
        if circuit_breaker:
            log.info("Circuit breaker RESET — new trading day")
        circuit_breaker = False
        log.info(f"Daily drawdown anchor: ${starting_equity:,.2f}")

    if circuit_breaker:
        return False

    equity   = float(account.equity)
    drawdown = (starting_equity - equity) / starting_equity if starting_equity else 0
    if drawdown >= MAX_DRAWDOWN:
        circuit_breaker = True
        log.critical(f"CIRCUIT BREAKER — intraday drawdown {drawdown*100:.1f}% > {MAX_DRAWDOWN*100:.0f}% — trading halted until tomorrow")
        return False
    return True

def run_risk_checks(account) -> tuple[bool, str]:
    """
    HALTS EVERYTHING (monitoring + new entries) — pause and circuit
    breaker are genuine stop-all conditions.
    """
    if is_paused():
        return False, "PAUSED"
    if not check_drawdown(account):
        return False, "CIRCUIT BREAKER"
    return True, "OK"

def can_open_new_position() -> tuple[bool, str]:
    """
    Blocks NEW ENTRIES ONLY — the daily trade cap should never stop
    the bot from managing risk on positions it already holds.

    Fix (Sep 2026 audit): MAX_TRADES_DAY was previously folded into
    run_risk_checks(), which gates the ENTIRE cycle — hitting the daily
    cap silently disabled check_profit_targets()/check_max_losing_hold()/
    check_stale_holds() for every open position until midnight ET. A
    busy day (exactly when the cap is most likely to be hit) is exactly
    when continued stop-loss/trailing monitoring matters most. Split so
    the trade-count limit only ever suppresses deploy_from_cache().
    """
    if trades_today_count() >= MAX_TRADES_DAY:
        return False, f"MAX TRADES {trades_today_count()}/{MAX_TRADES_DAY}"
    return True, "OK"

# ══════════════════════════════════════════════════════════════
# POSITION MONITORING
# ══════════════════════════════════════════════════════════════

def get_durable_composite(symbol: str) -> float | None:
    """
    Recovers a position's entry composite score from Alpaca's own order
    history via client_order_id, when the /tmp-based entry_signals cache
    has lost it (e.g. after a Render redeploy — see place_buy() for the
    full root-cause writeup). This is the durable fallback: Alpaca's
    order database survives every bot redeploy since it's never stored
    in the container's filesystem at all.

    Looks up the most recent FILLED buy order for this symbol and
    decodes the composite from its client_order_id (format: "cNNNN_ts").
    Returns None if no matching order is found or decoding fails —
    callers should then fall back to the flat base stop, same as before.
    """
    cache_key = f"durable_composite_{symbol}"
    if cache_key in fund_cache:
        cached_time, cached_result = fund_cache[cache_key]
        if time.time() - cached_time < 3600:  # 1hr cache — avoid hammering the orders API
            return cached_result

    try:
        # Sep 2026 audit fix: get_orders() with NO arguments defaults to
        # status=OPEN in the Alpaca SDK — this function was filtering for
        # status=="filled" against a query that can only ever return open
        # orders, so `matching` was guaranteed empty every single call.
        # The entire durable-composite recovery (built specifically to
        # prevent a repeat of the NVDA flat-stop incident) has likely
        # never actually worked since it shipped. Explicitly requesting
        # CLOSED orders fixes this.
        request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=50)
        orders  = trade_client.get_orders(filter=request)
        matching = [
            o for o in orders
            if o.symbol == symbol and o.side.value == "buy"
            and str(o.status).lower() in ("filled", "orderstatus.filled")
            and o.client_order_id and o.client_order_id.startswith("c")
        ]
        if not matching:
            fund_cache[cache_key] = (time.time(), None)
            return None
        # Most recent matching buy order
        matching.sort(key=lambda o: o.submitted_at or o.created_at, reverse=True)
        client_id = matching[0].client_order_id
        composite_part = client_id.split("_")[0][1:]  # strip leading "c"
        composite = int(composite_part) / 100
        fund_cache[cache_key] = (time.time(), composite)
        return composite
    except Exception as e:
        log.warning(f"Durable composite lookup failed for {symbol}: {e}")
        fund_cache[cache_key] = (time.time(), None)
        return None

STALE_HOLD_HOURS       = 4        # re-check fundamentals if held longer than this
STALE_HOLD_RECHECK_SECS = 3600    # don't re-check the SAME position more than once per hour
last_staleness_check: dict = {}   # {symbol: timestamp of last re-check}

# ── Never-touched-breakeven early exit (Sep 2026) ────────────────
# ── Early-hold tight floor (Sep 2026, simplified) ────────────────
# Root cause: real intraday bar data confirmed COST and PLTR were BOTH
# 100% red — never once traded above entry — for their entire hold.
# The original fix gated a tight ATR-scaled stop behind "has this
# position EVER touched breakeven" — but that created two competing
# clocks (a 30-min noise-tolerance delay on the composite stop, and a
# separate "hours on probation since never green" clock), and since
# the tight floor's ceiling (originally -2.5%) is always reached before
# the loosest composite tier (-3%), the 30-min delay was effectively
# dead code for any position that never went green — it never got the
# chance to matter.
#
# Simplified to ONE clock: minutes since entry. For the first
# STOP_LOSS_ACTIVATION_MINUTES, NOTHING fires — full noise tolerance
# while the position settles in, whether it's touched green or not.
# After the delay passes, BOTH the ATR-scaled early-exit stop (now
# hard-capped at -0.5%, tightened further from -1.5%/-2.5% for a
# faster, cheaper cut) and the composite-tiered stop (checked in
# check_profit_targets(), with the dynamic downside floor already
# layered on it) become active together — whichever threshold is
# tighter for a given position effectively governs from that point on.
def get_early_exit_stop(symbol: str) -> float:
    """
    ATR-scaled stop that becomes active once STOP_LOSS_ACTIVATION_MINUTES
    has passed since entry — tighter than the composite-tiered stop,
    designed to minimise loss impact on a position that's still shown
    no sign of validating (never touched breakeven) once the initial
    noise-tolerance grace period has elapsed.

    Tightened further (Sep 2026): hard-capped at -0.5% regardless of
    ATR. A position with zero validation after the grace period is cut
    fast and cheap — this makes the early-exit stop meaningfully
    tighter than the loosest composite tier (-3%) across the board,
    not just for low-volatility names, closing the coherence gap where
    the two rules previously fought each other on timing.
    """
    atr = entry_signals.get(symbol, {}).get("atr_pct", 0.02)
    return -min(0.005, max(0.003, atr * 0.75))

def check_max_losing_hold(positions: dict) -> list[str]:
    """
    Unified early-hold policy (Sep 2026): NOTHING fires in the first
    STOP_LOSS_ACTIVATION_MINUTES of a position's life — full noise
    tolerance while the position settles in, exactly like the composite
    stop's original intent. After that window passes, the ATR-scaled
    early-exit stop (get_early_exit_stop) becomes active ALONGSIDE the
    composite-tiered stop in check_profit_targets() — whichever is
    tighter effectively governs, since both are checked every cycle
    from that point on.

    This replaces the earlier (inverted) version where the tight
    ATR-scaled stop fired ONLY during the first 30 minutes and then
    handed off to the looser composite tier — which defeated the
    purpose of the 30-minute grace period, since the tightest rule was
    the one active exactly when we wanted maximum tolerance, and the
    loosest rule took over once tolerance was no longer the goal.
    """
    closed = []
    now    = time.time()

    for symbol, pos in positions.items():
        try:
            pnl_pct = float(pos.unrealized_plpc)
        except (AttributeError, TypeError, ValueError):
            continue

        if pnl_pct >= 0:
            continue  # only a downside check — profit targets handled elsewhere

        entry_time = entry_signals.get(symbol, {}).get("entry_time")
        if entry_time is None:
            continue  # legacy position, no entry timestamp — skip

        minutes_held = (now - entry_time) / 60
        if minutes_held < STOP_LOSS_ACTIVATION_MINUTES:
            continue  # still inside the grace period — nothing fires yet, by design

        early_stop = get_early_exit_stop(symbol)
        if pnl_pct <= early_stop:
            reason = (
                f"EARLY_EXIT_STOP (ATR-scaled {early_stop*100:.1f}% stop hit at "
                f"{minutes_held:.1f}min held — minimising loss, {pnl_pct*100:+.2f}%)"
            )
            if close_position(symbol, pnl_pct, reason):
                closed.append(symbol)
                log.warning(f"  [SELL] {symbol} exited — {reason}")

    return closed

def check_stale_holds(positions: dict) -> list[str]:
    """
    Periodically re-runs fundamentals on positions held longer than
    STALE_HOLD_HOURS, independent of whether news happened to fire.

    Root cause: entry-time composite/fundamental scores are trusted for
    the ENTIRE duration of a hold — AMZN's FTC lawsuit news reached the
    news-driven review path (see process_news_queue), but a position
    could just as easily deteriorate from something that never crosses
    the WebSocket's keyword filters (a slow-building competitive
    narrative, a sector-wide re-rating, analyst commentary that isn't
    phrased as any of the tracked keywords). This is the general-purpose
    backstop: don't trust a multi-hour-old fundamental score forever.

    Rate-limited to once per hour per symbol to control Claude API cost —
    this is NOT meant to run every 60s cycle like the price-based checks.
    """
    closed = []
    now    = time.time()

    for symbol, pos in positions.items():
        entry = entry_signals.get(symbol, {})
        entry_time = entry.get("entry_time")
        if entry_time is None:
            continue  # no entry timestamp (legacy position) — nothing to compare against

        held_hours = (now - entry_time) / 3600
        if held_hours < STALE_HOLD_HOURS:
            continue

        last_check = last_staleness_check.get(symbol, 0)
        if now - last_check < STALE_HOLD_RECHECK_SECS:
            continue  # already re-checked this symbol recently

        last_staleness_check[symbol] = now
        log.info(f"  {symbol}: held {held_hours:.1f}hrs — re-running stale fundamentals check")

        if symbol in fund_cache:
            del fund_cache[symbol]
        result = compute_signal(symbol, spy_change or 0.0)

        if result and result.get("signal") == "SELL":
            try:
                pnl_pct = float(pos.unrealized_plpc)
            except (AttributeError, TypeError, ValueError):
                pnl_pct = 0.0
            reason = f"STALE_FUNDAMENTALS_DETERIORATED (composite={result['composite']:.2f} after {held_hours:.1f}hr hold)"
            if close_position(symbol, pnl_pct, reason):
                closed.append(symbol)
                log.warning(f"  [SELL] {symbol} exited — fundamentals deteriorated since entry ({held_hours:.1f}hrs ago)")
        elif result:
            log.info(f"  {symbol}: fundamentals still support hold (composite={result['composite']:.2f})")

    return closed

def check_profit_targets(positions: dict) -> list[str]:
    """
    Exit rules — no Claude calls needed.
    Priority order (first match wins):
      1. Hard ceiling: +60% → sell immediately (the ONLY fixed profit
         exit — the fixed/ATR profit target was removed Sep 10 2026)
      2. Dynamic-width trail: peak ≥ +1.0% arms it. Gap between peak
         and sell-floor widens as the peak grows (see DYNAMIC_TRAIL_TABLE):
         tight (0.4%) near breakeven to protect against noise, wide
         (up to 10%) at high peaks so a genuine runner (like MU's +19%
         day) gets room to keep going instead of being stopped out on
         the first small wobble. Fresh, wider tiers arm at +5%, +10%,
         +15%, +20% and +40% — a runner is never sold at a fixed level
         below +60%.
      3. Old fixed trailing: peak ≥ +3% → sell if falls to +2.5%
         (kept as a fallback path; in practice #2 fires first since
         it arms earlier at +1.0% and its floor is above +2.5% by then)
      4. Breakeven stop: peak ≥ +1% (or +2% calm-VIX) → stop shifts
         to +0.5% (or +1% calm-VIX)
      5. Hard stop loss: composite-tiered -0.75%/-1.0%/-1.5% (-0.75% in BEAR mode)
    Plus:
      6. Weak sector mid-day exit: if sector turns weak AND position
         is at breakeven or better → exit to protect gains.
         If position is negative → hold (don't crystallise a loss).
    """
    closed      = []
    base_stop   = get_stop_loss()   # -1.5% normal, -0.75% in BEAR mode — the ceiling
    global signal_cache  # declared once here — was previously declared twice, nested
                          # inside conditional blocks below, which is a SyntaxError in
                          # Python if any use of the name in this scope could precede a
                          # later 'global' statement. Aug 31 production crash: bot failed
                          # to start at all with "name 'signal_cache' is used prior to
                          # global declaration" — single declaration at top fixes this.

    # ── VIX-aware breakeven thresholds ─────────────────────────
    # Calm market (VIX < 18): widen the breakeven band so normal
    # intraday noise doesn't trigger an early exit on a real winner.
    is_calm = current_vix is not None and current_vix < CALM_VIX_THRESHOLD
    be_trigger = BREAKEVEN_TRIGGER_CALM if is_calm else BREAKEVEN_TRIGGER
    be_stop    = BREAKEVEN_STOP_CALM    if is_calm else BREAKEVEN_STOP

    # ── Overnight-hold prioritisation (Sep 2026) ────────────────
    # Root cause: PLTR (bought Sep 1, held overnight) breached its own
    # -4% composite-tier stop and closed at -5.92% the next morning —
    # exchange gaps at the open can blow through a percentage-based
    # check before this polling loop even gets to look at the position.
    # We can't prevent the gap itself (no resting broker-side stop order
    # exists — that's the real structural fix, tracked separately), but
    # we CAN make sure an overnight-held position is the very first thing
    # evaluated each cycle rather than being processed in arbitrary dict
    # order alongside same-day entries — so if the bot's loop is even
    # slightly delayed, the position most exposed to gap risk is checked
    # with priority, not last.
    #
    # Detection: a position opened TODAY has unrealized_intraday_plpc
    # equal to unrealized_plpc (today's move IS the total move so far).
    # A position held OVERNIGHT has these diverge, since total P&L
    # includes days before today. No extra API call needed — both
    # fields are already present on the position object.
    def _is_overnight_hold(pos) -> bool:
        try:
            total_pct    = float(pos.unrealized_plpc)
            intraday_pct = float(pos.unrealized_intraday_plpc)
            return abs(total_pct - intraday_pct) > 0.0005  # >0.05% divergence = pre-existing position
        except (AttributeError, TypeError, ValueError):
            return False  # if the field is missing, don't assume — process in normal order

    ordered_positions = sorted(
        positions.items(),
        key=lambda item: (not _is_overnight_hold(item[1]), item[0])  # overnight holds first, then alphabetical
    )
    overnight_symbols = [sym for sym, pos in ordered_positions if _is_overnight_hold(pos)]
    if overnight_symbols:
        log.info(f"  Overnight-held positions checked first this cycle: {overnight_symbols}")

    for symbol, pos in ordered_positions:
        try:
            pnl_pct = float(pos.unrealized_plpc)

            # ── Composite-aware stop loss ──────────────────────
            # Jul 29 review: realised wins clustered at +0.3-0.5% against a
            # flat -5% stop — that ratio needs ~93% win rate to break even.
            # Fix: entries with a weaker composite (closer to the MIN_COMPOSITE
            # floor) get a TIGHTER stop, since they're lower-conviction and
            # shouldn't be given as much rope. High-conviction entries (which
            # should also be sized larger via Kelly) get the full -1.5% to let
            # the thesis play out. BEAR mode's -0.75% ceiling always wins (tightest).
            #
            # Sep 10 2026: tiers tightened from -3%/-4%/-5% to -0.75%/-1.0%/-1.5%.
            # The loss side was ~16x the win side (SLB -1.71% vs MSFT +0.11%
            # the same day). Winners can now run via the trail, so the stop
            # is sized to what the strategy's wins actually look like.
            #
            # Aug 24 fix: entry_signals (/tmp-backed) is lost on every Render
            # redeploy — this caused NVDA to silently fall back to the flat
            # -5% stop instead of its actual tier, riding the full loss
            # before exiting. Now falls back to get_durable_composite()
            # (reads Alpaca's own order history via client_order_id) before
            # giving up and using the flat stop.
            entry_composite = entry_signals.get(symbol, {}).get("composite")
            if entry_composite is None:
                entry_composite = get_durable_composite(symbol)
                if entry_composite is not None:
                    log.info(f"  {symbol}: entry composite recovered from order history ({entry_composite:.2f}) — /tmp cache was empty")
            if entry_composite is not None and market_state != "BEAR":
                if entry_composite < 4.5:
                    active_stop = -0.0075    # marginal entry (4.0-4.5) → tighter -0.75%
                elif entry_composite < 6.0:
                    active_stop = -0.01      # normal entry (4.5-6.0)  → -1.0%
                else:
                    active_stop = base_stop  # high conviction (6.0+)  → full -1.5%
            else:
                active_stop = base_stop      # no entry data anywhere, or BEAR mode

            # Update peak
            prev_peak = position_peaks.get(symbol, 0.0)
            if pnl_pct > prev_peak:
                position_peaks[symbol] = pnl_pct
                save_peaks()
                if pnl_pct >= MICRO_TRAIL_ARM_PCT:
                    gap   = get_dynamic_trail_gap(pnl_pct)
                    floor = pnl_pct - gap
                    log.info(
                        f"  {symbol}: new peak {pnl_pct*100:+.2f}% — dynamic trail active, "
                        f"gap {gap*100:.1f}%, floor {floor*100:+.2f}%"
                    )
                elif pnl_pct >= be_trigger:
                    log.info(
                        f"  {symbol}: new peak {pnl_pct*100:+.2f}% — breakeven stop active "
                        f"(+{be_stop*100:.1f}%{' calm-VIX' if is_calm else ''})"
                    )

            current_peak = position_peaks.get(symbol, 0.0)
            reason       = None

            # ── Profit side (Sep 10 2026) ─────────────────────
            # The ATR-based / fixed profit target that used to live here
            # was REMOVED — it was selling winners at +3-5% while losses
            # ran to the full tier. A winner now only closes via the
            # escalating dynamic trail, or at the +60% hard ceiling.

            # ── Standard exit rules ────────────────────────────
            # Priority order: hard ceiling > stop-loss-magnitude override >
            # micro-trail > old fixed trail > breakeven stop > hard stop loss
            #
            # Sep 2026 audit fix: the DYNAMIC_TRAIL branch below only checks
            # "has price fallen more than the trail gap below its peak" — it
            # has NO awareness of the position's own stop-loss tier. Once a
            # position has EVER peaked >= MICRO_TRAIL_ARM_PCT (1.0%), its
            # current_peak never resets down, so ANY subsequent crash — even
            # one that blows straight through the -0.75%/-1.0%/-1.5% composite tier
            # — gets caught and labelled "DYNAMIC_TRAIL" instead of the more
            # accurate "STOP_LOSS". This didn't cost money (the position still
            # closes at the same time either way), but it corrupts
            # analyse_signal_attribution()'s reporting: a real stop-loss-
            # magnitude loss gets bucketed as a trailing exit, making the
            # trail's stats look worse and the stop-loss tier's stats look
            # artificially better than they really are.
            #
            # Fix: if the loss has already fallen past the position's own
            # composite-tiered stop level, classify it as STOP_LOSS
            # regardless of prior peak — the magnitude of the loss is what
            # matters for accurate attribution, not whether it once ticked
            # positive first.
            if pnl_pct >= HARD_SELL_CEILING:
                reason = f"HARD_CEILING (+{HARD_SELL_CEILING*100:.0f}% reached — forced exit, no trail)"
            elif pnl_pct <= active_stop:
                # Loss has reached stop-loss-tier magnitude — always label
                # it STOP_LOSS for accurate attribution, even if this
                # position peaked positive earlier in its life. Still
                # respects the same activation delay as the dedicated
                # STOP_LOSS branch further below.
                entry_time = entry_signals.get(symbol, {}).get("entry_time")
                minutes_held = (time.time() - entry_time) / 60 if entry_time else None
                if minutes_held is None or minutes_held >= STOP_LOSS_ACTIVATION_MINUTES:
                    peak_note = f", peaked {current_peak*100:+.2f}% earlier" if current_peak >= MICRO_TRAIL_ARM_PCT else ""
                    reason = f"STOP_LOSS ({active_stop*100:.0f}%{' composite-tiered' if entry_composite is not None else ''}{peak_note})"
                # else: falls through to noise-tolerance window below via the
                # ordinary STOP_LOSS branch's own delay check — no action here,
                # just don't claim it as DYNAMIC_TRAIL in the meantime either
                elif current_peak < MICRO_TRAIL_ARM_PCT:
                    log.info(
                        f"  {symbol}: at {pnl_pct*100:+.2f}% (below {active_stop*100:.0f}% stop) but only "
                        f"{minutes_held:.1f}min held — stop-loss activates at {STOP_LOSS_ACTIVATION_MINUTES}min, holding"
                    )
            elif current_peak >= MICRO_TRAIL_ARM_PCT and pnl_pct <= (current_peak - get_dynamic_trail_gap(current_peak)):
                gap = get_dynamic_trail_gap(current_peak)
                reason = (
                    f"DYNAMIC_TRAIL (peaked {current_peak*100:+.2f}%, "
                    f"gap {gap*100:.1f}% behind peak)"
                )
            elif current_peak >= PEAK_TRIGGER and pnl_pct <= TRAIL_SELL:
                reason = f"TRAILING (peaked {current_peak*100:+.2f}%)"
            elif current_peak >= be_trigger and pnl_pct <= be_stop:
                reason = (
                    f"BREAKEVEN_STOP (peaked {current_peak*100:+.2f}%, "
                    f"locked +{be_stop*100:.1f}%{' calm-VIX' if is_calm else ''})"
                )
            elif current_peak < be_trigger and active_stop < pnl_pct < get_dynamic_downside_floor(pnl_pct, active_stop):
                # ── Dynamic downside floor (Sep 2026) ───────────
                # Position hasn't hit the flat tier ceiling yet, but HAS
                # crossed the tightened floor for how deep it already is.
                # Same activation delay applies — a position can't be cut
                # by this any earlier than the flat stop could be.
                entry_time = entry_signals.get(symbol, {}).get("entry_time")
                minutes_held = (time.time() - entry_time) / 60 if entry_time else None
                if minutes_held is not None and minutes_held >= STOP_LOSS_ACTIVATION_MINUTES:
                    dyn_floor = get_dynamic_downside_floor(pnl_pct, active_stop)
                    reason = (
                        f"DYNAMIC_STOP ({pnl_pct*100:+.2f}% crossed tightened floor "
                        f"{dyn_floor*100:.1f}% — tier ceiling was {active_stop*100:.0f}%)"
                    )

            # ── Weak sector mid-day exit ───────────────────────
            # Normally only exits at breakeven or better — never crystallise
            # a loss due to ordinary sector rotation (a single sector dipping
            # is common and often reverses).
            #
            # EXCEPTION (Aug 2026): during a systemic de-risk event (severe
            # VIX + 3+ sectors weak simultaneously — see assess_market_state),
            # this is not rotation, it's a broad selloff. Waiting for each
            # position's own composite-tiered hard stop (-0.75% to -1.5%) to fire
            # one at a time means riding the full stop distance on every
            # position during exactly the scenario where speed matters most.
            # In that mode, allow exiting a small loss (bounded by
            # SYSTEMIC_DERISK_MAX_LOSS_EXIT) rather than holding for the
            # full stop distance.
            if not reason and weak_sectors:
                sector = get_sector(symbol)
                if sector and sector in weak_sectors:
                    if pnl_pct >= be_stop:
                        reason = (
                            f"WEAK_SECTOR ({sector.upper()} weak, "
                            f"P&L {pnl_pct*100:+.2f}% ≥ +{be_stop*100:.1f}% — exiting)"
                        )
                    elif systemic_derisk_active and pnl_pct >= SYSTEMIC_DERISK_MAX_LOSS_EXIT:
                        reason = (
                            f"WEAK_SECTOR_SYSTEMIC ({sector.upper()} weak during systemic "
                            f"de-risk, P&L {pnl_pct*100:+.2f}% — exiting early rather than "
                            f"riding to full stop distance)"
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

    KELLY_PCT   = 0.10  # minimum-cash gate — aligned with new lowest tier (was 0.10, tiers now 10/13/16%)
    MAX_POS     = 10
    equity      = float(account.equity)
    cash        = float(account.regt_buying_power or account.cash)  # intraday buying power
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

    if systemic_derisk_active:
        log.warning("SYSTEMIC DE-RISK ACTIVE — no new positions until conditions normalise")
        return

    # Use cached signals — deduplicated by symbol, no Claude calls
    seen = set()
    available = []
    for s in signal_cache:
        sym = s["symbol"]
        if sym not in positions and sym not in seen and sym not in closed_this_session:
            seen.add(sym)
            available.append(s)
        elif sym in closed_this_session:
            log.info(f"  {sym}: closed this cycle — skipping redeploy (re-entry loop prevention)")

    if not available:
        # Cache exhausted — emergency rescan max once per hour
        time_since = time.time() - last_rescan_time
        if time_since > 3600:  # 1 hour
            log.info("Signal cache exhausted — running emergency rescan (1hr cooldown)")
            last_rescan_time = time.time()
            spy_chg = get_quote_change("SPY") or 0.0
            universe = build_universe()
            new_signals = []
            skipped_ta  = 0
            for symbol in universe:
                if symbol in positions:
                    continue
                # ── TA pre-filter (Aug 24 fix) ─────────────────
                # The emergency rescan was missing the same TA pre-filter
                # the pre-market scan uses — calling Claude on every ticker
                # in the universe unconditionally. Observed Aug 24: this
                # took 7+ minutes mid-session (19:47:44 → 19:54:59) scoring
                # ~40 tickers sequentially with no skip, well over the
                # ~4min the optimised pre-market scan takes. Same fix:
                # fetch cheap TA first, skip the paid Claude call entirely
                # if taScore < 1.5 (clearly bearish / no signal).
                ta = fetch_technicals(symbol)
                if ta and ta.get("taScore", 0) < 1.5:
                    skipped_ta += 1
                    continue  # no sleep needed — skipping Claude call
                result = compute_signal(symbol, spy_chg, prefetched_ta=ta)
                if result and result["signal"] == "BUY" and result["confidence"] >= MIN_CONFIDENCE and result["composite"] >= MIN_COMPOSITE:
                    new_signals.append(result)
                time.sleep(3)
            log.info(f"Emergency rescan: {skipped_ta} tickers skipped on weak TA, {len(new_signals)} signal(s) found")
            signal_cache = sorted(new_signals, key=lambda x: x["confidence"], reverse=True)
            available    = signal_cache
        else:
            log.info(f"Cache empty — next emergency rescan in {max(0,(3600-time_since)/60):.0f}min")
            return

    to_buy = available[:open_slots]
    log.info(f"Deploying into {len(to_buy)} signal(s) from cache: {[s['symbol'] for s in to_buy]}")

    # Sector counts tracked across the WHOLE deploy loop — incremented after
    # each buy so 3 same-sector candidates can't all pass the cap in one cycle
    sector_counts: dict = {}
    for s in positions:
        sec = get_sector(s)
        if sec:
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    # AI-correlated count tracked the same way, across the whole loop
    ai_correlated_count = count_ai_correlated_positions(positions.keys())

    for candidate in to_buy:
        symbol = candidate["symbol"]
        price  = candidate["price"]
        if price <= 0:
            continue

        # ── Sector cap — max 2 positions per sector (backtest validated) ──
        # Apr 26 +$1,196, May 26 +$1,914 better vs uncapped
        # Prevents NVDA×3 concentration regardless of signal quality
        # NOTE: counts include buys made earlier in THIS loop (same-cycle fix)
        # NOTE: get_sector() falls back to a live lookup for tickers not in
        # the static SECTOR_MAP (e.g. dynamically-discovered PHVS Sep 8) —
        # this is the actual fix for the weak-sector filter below silently
        # not applying to unmapped tickers.
        sector = get_sector(symbol)
        if sector:
            sector_count = sector_counts.get(sector, 0)
            if sector_count >= MAX_SECTOR_POSITIONS:
                log.info(
                    f"  {symbol}: sector '{sector}' capped — "
                    f"{sector_count}/{MAX_SECTOR_POSITIONS} positions held — skipping"
                )
                continue

        # ── AI-correlated concentration cap ─────────────────────
        # Separate, stricter cap spanning semis/tech/software/cyber
        # combined. Prevents e.g. 2 semis + 2 tech + 2 software all
        # being "the AI trade" under different sector labels while
        # each individual sector count looks compliant.
        if sector in AI_CORRELATED_SECTORS:
            if ai_correlated_count >= MAX_AI_CORRELATED_POSITIONS:
                log.info(
                    f"  {symbol}: AI-correlated concentration capped — "
                    f"{ai_correlated_count}/{MAX_AI_CORRELATED_POSITIONS} positions "
                    f"already in semis/tech/software/cyber — skipping"
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
            r = requests.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
                timeout=5,
            )
            live_price = float(r.json().get("trade", {}).get("p", price)) if r.ok else price
        except Exception:
            live_price = price

        allowed, block_reason = check_reentry_allowed(symbol, live_price)
        if not allowed:
            log.info(f"  {symbol}: re-entry BLOCKED — {block_reason}")
            continue

        # ── Confidence-based sizing (raised Aug 2026 — moderate increase) ──
        # Previous tiers (8/10/12%) left ~80% of the account idle on most
        # days (Aug 13-20 sample: only 2-3 positions open at a time out of
        # a $103k account). Raising tiers increases capital utilisation per
        # trade without changing which signals qualify or how fast the
        # cache is exhausted — a separate lever (composite floor / cooldowns)
        # from the idle-cash problem, addressed independently.
        # High conviction (composite ≥ 6.0, confidence ≥ 90%) → 16% Kelly (was 12%)
        # Normal (composite ≥ 4.5)                            → 13% Kelly (was 10%)
        # Marginal (composite < 4.5)                          → 10% Kelly (was 8%)
        composite  = candidate.get("composite", 0)
        confidence = candidate.get("confidence", 85)
        if composite >= 6.0 and confidence >= 90:
            kelly = 0.16
        elif composite >= 4.5:
            kelly = 0.13
        else:
            kelly = 0.10

        alloc  = min(equity * kelly, cash * 0.95)
        qty    = int(alloc / live_price)
        if qty < 1:
            continue
        qty = adjust_qty_for_fear(qty, live_price, alloc)

        # ATR-based profit target removed Sep 10 2026 — check_profit_targets()
        # no longer sells at a fixed level; the escalating trail governs
        # the profit side. atr_pct is still stored in entry_signals below
        # for the early-exit stop.
        log.info(f"  {symbol}: {qty} shares @ ~${live_price:.2f} = ${qty*live_price:,.0f} ({kelly*100:.0f}% Kelly)")

        # ── Store entry signal snapshot — required to analyse which
        # analysis (TA vs fundamentals) actually drives outcomes.
        # Without this, exit-time journal entries have no link back to
        # what the composite/confidence/scores were when the bot bought.
        entry_signals[symbol] = {
            "composite":  composite,
            "confidence": confidence,
            "ta_score":   candidate.get("ta_score"),
            "fund_score": candidate.get("fund_score"),
            "sector":     sector,
            "kelly_pct":  kelly,
            "entry_price": live_price,
            "entry_time":  time.time(),  # Sep 2026 — enables staleness check in check_profit_targets
            "atr_pct":     atr if atr and atr > 0 else 0.02,  # for the never-touched-breakeven early exit
        }
        save_entry_signals()

        if place_buy(symbol, qty, composite=composite):
            cash -= qty * live_price
            if sector:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1  # same-cycle cap tracking
                if sector in AI_CORRELATED_SECTORS:
                    ai_correlated_count += 1  # same-cycle AI-correlated cap tracking

# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run():
    global last_rescan_time
    load_peaks()
    load_cooldowns()     # restore stop-loss/profit cooldowns after restart
    load_entry_signals()  # restore entry snapshots for signal-attribution analysis
    load_scan_state()     # restore today's scan cache — prevents a redeploy from
                           # triggering a wasted re-scan if one already ran today
    # (early-hold tight floor no longer needs separate persisted state —
    # it derives purely from entry_time, already loaded via entry_signals)
    start_news_stream()  # Start news WebSocket in background thread

    log.info("=" * 60)
    log.info("SIGNAL Trading Bot started")
    log.info(f"  Universe:       {len(CURATED_TICKERS)} curated + up to 24 dynamic (8+8+8) = 55 max")
    log.info(f"  Scan timing:    Sun 8pm ET / Mon-Fri 9:20am ET (dynamic) + restart rescan")
    log.info(f"  Position size:  Kelly 10-16% (confidence-based sizing, raised Aug 2026)")
    log.info(f"  Profit target:  ATR×2.5 per position (3-12% range, fallback +5%)")
    log.info(f"  Sector cap:     Max 2 positions per sector (backtest validated)")
    log.info(f"  AI concentration cap: Max {MAX_AI_CORRELATED_POSITIONS} combined across semis/tech/software/cyber")
    log.info(f"  Systemic de-risk: VIX≥{SYSTEMIC_DERISK_VIX} + {SYSTEMIC_DERISK_MIN_WEAK_SECTORS}+ weak sectors → blocks new buys, allows early loss-cutting")
    log.info(f"  Re-entry rules: +60% ceiling exit → 4hr cooldown + 2% price gate")
    log.info(f"                  stop loss → 24hr cooldown + 2% price gate (escalating strikes)")
    log.info(f"  Max positions:  10 concurrent")
    log.info(f"  Profit target:  NONE — trail-only. Hard ceiling +{HARD_SELL_CEILING*100:.0f}% (sell immediately)")
    log.info(f"  Trail tiers:    " + " | ".join(f"peak>={t*100:g}%→{g*100:g}% behind" for t, g in DYNAMIC_TRAIL_TABLE))
    log.info(f"  Trailing:       peak >={PEAK_TRIGGER*100:.0f}% → sell at +{TRAIL_SELL*100:.0f}%")
    log.info(f"  Breakeven:      calm(VIX<{CALM_VIX_THRESHOLD}) peak>={BREAKEVEN_TRIGGER_CALM*100:.0f}%→lock+{BREAKEVEN_STOP_CALM*100:.0f}% | else peak>={BREAKEVEN_TRIGGER*100:.0f}%→lock+{BREAKEVEN_STOP*100:.1f}%")
    log.info(f"  Stop loss:      -{abs(STOP_LOSS)*100:.1f}% ceiling")
    log.info(f"  TA/Fund weight: {TECH_WEIGHT}% / {FUND_WEIGHT}%")
    log.info(f"  Min confidence: {MIN_CONFIDENCE}%")
    log.info(f"  Min composite:  {MIN_COMPOSITE} (raised from 3.0 — Jul 29 review)")
    log.info(f"  Stop loss:      tiered by entry composite — <4.5: -0.75% | 4.5-6.0: -1.0% | 6.0+: -1.5%")
    log.info(f"  Stop loss delay: activates {STOP_LOSS_ACTIVATION_MINUTES}min after entry (avoids opening-print noise)")
    log.info(f"  Dynamic downside floor: tightens toward tier ceiling as loss deepens (mirrors trail, inverted)")
    log.info(f"  Max drawdown:   {MAX_DRAWDOWN*100:.0f}%")
    log.info(f"  90-day audit:   Volume, TA alignment, win rate")
    log.info(f"  Held-position news review: bearish/material-adverse news on a HELD symbol re-runs fundamentals, can trigger early exit")
    log.info(f"  Stale-hold check: fundamentals re-checked after {STALE_HOLD_HOURS}hr hold, max 1x/hr per symbol")
    log.info(f"  Early-hold policy: no exit fires in first {STOP_LOSS_ACTIVATION_MINUTES}min (noise tolerance); after that, ATR-scaled early-exit stop (hard-capped -0.5%) AND composite tier both active")
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

    # Weekly signal-attribution check — faster feedback loop than the 90-day
    # audit while enough entry-tagged trades accumulate to be meaningful.
    attribution_file = "/tmp/last_attribution.txt"
    try:
        with open(attribution_file) as f:
            last_attr = datetime.strptime(f.read().strip(), "%Y-%m-%d").replace(tzinfo=ET)
        if (datetime.now(ET) - last_attr).days >= 7:
            analyse_signal_attribution()
            with open(attribution_file, "w") as f:
                f.write(datetime.now(ET).strftime("%Y-%m-%d"))
    except FileNotFoundError:
        with open(attribution_file, "w") as f:
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

            # ── Clear per-cycle state ──────────────────────────
            closed_this_session.clear()  # reset every 60s cycle

            positions = get_positions()
            held_symbols.clear()
            held_symbols.update(positions.keys())  # keeps WebSocket thread's view current
            log.info(f"Open positions: {list(positions.keys()) or 'none'}")
            prune_peaks(list(positions.keys()))

            # Check exits
            closed = check_profit_targets(positions) if positions else []

            # Force-exit positions that have never touched breakeven since
            # entry, using an ATR-scaled loss-minimising stop — catches the
            # COST/AMZN/PLTR pattern (real data showed these were 89-100%
            # red for their entire hold, never once green)
            if positions:
                maxhold_closed = check_max_losing_hold(positions)
                if maxhold_closed:
                    closed.extend(maxhold_closed)

            # Re-check fundamentals on positions held beyond STALE_HOLD_HOURS
            # (independent of news — catches slow-building deterioration that
            # never crosses the news WebSocket's keyword filters)
            if positions:
                stale_closed = check_stale_holds(positions)
                if stale_closed:
                    closed.extend(stale_closed)

            if closed:
                time.sleep(3)
                positions = get_positions()
                held_symbols.clear()
                held_symbols.update(positions.keys())
                account   = get_account()

            # ── Process news triggers (real-time events) ───────
            news_traded = process_news_queue(positions, account)
            if news_traded:
                time.sleep(3)
                positions = get_positions()
                held_symbols.clear()
                held_symbols.update(positions.keys())
                account   = get_account()

            # Deploy from cache (no Claude calls during market hours)
            open_slots = 10 - len(positions)
            can_open, cannot_open_reason = can_open_new_position()
            if not can_open:
                if open_slots > 0:
                    log.info(f"  {cannot_open_reason} — new entries blocked, monitoring continues normally")
            elif open_slots > 0 and float(account.regt_buying_power or account.cash) >= equity * 0.10:
                if not signal_cache:
                    time_since_rescan = time.time() - last_rescan_time
                    if time_since_rescan > 3600:
                        log.info(
                            f"  Cache empty mid-session (likely post-restart) — "
                            f"running one-time emergency rescan"
                        )
                        last_rescan_time = time.time()
                        run_premarket_scan()
                    else:
                        log.info(
                            f"  {open_slots} slot(s) available but signal cache is empty — "
                            f"next emergency rescan in {max(0,(3600-time_since_rescan)/60):.0f}min. "
                            f"News WebSocket active for breaking events."
                        )
                elif closed_this_session:
                    log.info(
                        f"  Skipping deploy — {len(closed_this_session)} position(s) just closed "
                        f"this cycle {closed_this_session} — deploying next cycle to avoid re-entry loop"
                    )
                else:
                    deploy_from_cache(positions, account)

            # Persist scan-gate state every cycle — cheap (one small JSON
            # write), and covers every place signal_cache/last_rescan_time
            # gets mutated during the day (deploys, exits, emergency
            # rescans, news-triggered inserts) without needing a save call
            # at each individual mutation site.
            save_scan_state()

        except KeyboardInterrupt:
            log.info("Bot stopped.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run()
