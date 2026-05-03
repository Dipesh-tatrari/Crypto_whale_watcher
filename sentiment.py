"""
sentiment.py — Real-time Sentiment Scoring  (Phase 5 — ML-powered)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE:
  ┌──────────────────────────────────────────────────────────────┐
  │  ml/predictor.py  — loads fine-tuned FinBERT (or fallback)  │
  │    • ml/crypto_finbert/  (after running  ml/train.py)        │
  │    • ProsusAI/finbert    (base model, no fine-tuning)        │
  │    • Lexicon scorer      (pure Python, always works)         │
  └────────────────────────────┬─────────────────────────────────┘
                               │  predict(text) → PredictorResult
  ┌────────────────────────────▼─────────────────────────────────┐
  │  SentimentBackgroundFetcher (daemon thread)                  │
  │    • Seeds from SYNTHETIC_HEADLINES immediately at startup   │
  │    • Fetches RSS headlines every REFRESH_SECS                │
  │    • Calls ml.predictor.predict() per headline               │
  │    • Writes SentimentResult into _SENTIMENT_STORE (dict)     │
  └────────────────────────────┬─────────────────────────────────┘
                               │  thread-safe plain dict (GIL)
  ┌────────────────────────────▼─────────────────────────────────┐
  │  Streamlit main thread                                       │
  │    • get_cached_sentiment(symbol) reads _SENTIMENT_STORE     │
  │    • Mirrors into st.session_state[K_SENTIMENT_CACHE]        │
  │    • Renders gauge / cards in UI                             │
  └──────────────────────────────────────────────────────────────┘

HOW TO UPGRADE SENTIMENT QUALITY:
  Step 1:  pip install transformers torch scikit-learn
  Step 2:  python ml/train.py          (fine-tunes FinBERT, ~5 min CPU)
  Step 3:  restart  streamlit run main.py
  → The gauge now uses your trained model automatically.
"""

import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

# ── ML predictor (loaded once at startup) ────────────────────
try:
    from ml import predictor as _predictor_mod
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SENTIMENT_REFRESH_SECS = 90
HEADLINE_CACHE_TTL     = 600
FETCH_TIMEOUT          = 10

NEWS_FEEDS = [
    "https://feeds.feedburner.com/CoinDesk",
    "https://cointelegraph.com/rss",
    "https://cryptonews.com/news/feed/",
    "https://decrypt.co/feed",
]

SYMBOLS_MONITORED = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

SYMBOL_ALIASES: dict[str, list[str]] = {
    "BTC/USDT": ["bitcoin", "btc", "₿"],
    "ETH/USDT": ["ethereum", "eth", "ether"],
    "SOL/USDT": ["solana", "sol"],
    "BNB/USDT": ["binance coin", "bnb", "binance"],
}

# Synthetic seed headlines — used immediately at startup so gauges
# are never empty, and as fallback when RSS is unreachable.
SYNTHETIC_HEADLINES: list[str] = [
    "Bitcoin rallies above key resistance as institutional inflows surge",
    "BTC breaks out to new monthly highs amid ETF approval optimism",
    "Bitcoin drops sharply on Fed rate decision fears and crypto selloff",
    "Whale accumulation detected — Bitcoin hodlers add to positions",
    "Bitcoin faces correction risk as overbought RSI signals bearish divergence",
    "BlackRock Bitcoin ETF records massive inflow in record trading session",
    "Ethereum surges on layer-2 adoption growth and DeFi recovery",
    "ETH staking yields rise as validators accumulate ahead of upgrade",
    "Ethereum faces bearish pressure as SEC scrutiny increases",
    "Ether rebounds strongly after exploit concerns prove overblown",
    "Solana breaks all-time high as DeFi and NFT activity explodes",
    "SOL rally continues on institutional partnership announcements",
    "Solana network congestion raises vulnerability concerns among traders",
    "Solana price plunges amid broader crypto market selloff and panic",
    "BNB coin gains on Binance expansion and new partnership launch",
    "Binance faces crackdown risk as regulators target exchange operations",
    "BNB uptrend supported by growing BNB Chain DeFi ecosystem",
    "Crypto market green across the board as risk appetite returns",
    "Altcoins surge on Bitcoin ETF approval momentum and bullish outlook",
    "Crypto fear and greed index hits extreme greed — correction risk grows",
    "Liquidations spike as crypto market dumps on macro uncertainty",
    "DeFi total value locked surges to new highs amid yield farming boom",
    "Stablecoin outflows signal bearish repositioning by smart money",
]

# ─────────────────────────────────────────────────────────────
# THREAD-SAFE STORE
# ─────────────────────────────────────────────────────────────
_SENTIMENT_STORE: dict[str, "SentimentResult"] = {}
_STORE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────
# SENTIMENT RESULT  (public dataclass used by UI)
# ─────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    symbol:     str
    score:      float          # [-1.0, +1.0]
    label:      str            # BEARISH | NEUTRAL | BULLISH
    confidence: float          # 0.0–1.0
    headline:   str            # source text
    source:     str = "synthetic"   # synthetic | live | finetuned | finbert | lexicon
    scored_at:  float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.scored_at

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > HEADLINE_CACHE_TTL

    @property
    def label_emoji(self) -> str:
        return {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(self.label, "")

    @property
    def backend_badge(self) -> str:
        return {
            "finetuned": "🧠 Fine-tuned",
            "finbert":   "🤖 FinBERT",
            "lexicon":   "📖 Lexicon",
            "synthetic": "🔄 Synthetic",
            "live":      "🌐 Live",
        }.get(self.source, self.source)


# ─────────────────────────────────────────────────────────────
# CORE SCORE FUNCTION  — routes to ML model or lexicon
# ─────────────────────────────────────────────────────────────

def score_text(text: str, symbol: str = "") -> SentimentResult:
    """
    Score a single headline using the best available backend.

    Routing:
      If ml.predictor is loaded and is a transformer → use it.
      Otherwise → lexicon scorer (pure Python, always works).
    """
    if _ML_AVAILABLE and _predictor_mod.is_transformer():
        try:
            result = _predictor_mod.predict(text)
            return SentimentResult(
                symbol     = symbol,
                score      = result.score,
                label      = result.label,
                confidence = result.confidence,
                headline   = text[:200],
                source     = result.backend,
            )
        except Exception:
            pass

    # Lexicon fallback
    if _ML_AVAILABLE:
        result = _predictor_mod.predict(text)   # returns lexicon result
        return SentimentResult(
            symbol=symbol, score=result.score, label=result.label,
            confidence=result.confidence, headline=text[:200],
            source="lexicon",
        )

    # No ml module at all — inline lexicon
    return _inline_lexicon(text, symbol)


def _inline_lexicon(text: str, symbol: str) -> SentimentResult:
    """Minimal inline lexicon — used only if ml/ package is missing."""
    import re, math
    POS = {"rally":1.5,"surge":1.5,"bullish":1.5,"gain":0.8,"approval":1.2,
           "inflow":1.0,"etf":1.3,"rebound":1.0,"boom":1.5,"record":0.8}
    NEG = {"crash":-2.0,"plunge":-1.8,"dump":-1.5,"bearish":-1.5,"hack":-2.0,
           "selloff":-1.5,"ban":-1.5,"fear":-1.0,"panic":-1.5,"loss":-1.0}
    tokens = re.findall(r"[\w'-]+", text.lower())
    raw    = sum(POS.get(t,0)+NEG.get(t,0) for t in tokens)
    score  = math.tanh(raw*0.4)
    label  = "BULLISH" if score>0.15 else "BEARISH" if score<-0.15 else "NEUTRAL"
    return SentimentResult(
        symbol=symbol, score=round(score,4), label=label,
        confidence=round(min(abs(score)+0.1,1.0),4),
        headline=text[:200], source="lexicon",
    )


# ─────────────────────────────────────────────────────────────
# RSS FETCHER
# ─────────────────────────────────────────────────────────────

def _fetch_headlines(url: str) -> list[str]:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; WhaleWatcher/5.0; "
                    "+https://github.com/crypto-whale-watcher)"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            }
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            raw = resp.read()
        root   = ET.fromstring(raw)
        titles = []
        # RSS 2.0
        for item in root.iter("item"):
            t = item.find("title")
            if t is not None and t.text and len(t.text.strip()) > 10:
                titles.append(t.text.strip())
        # Atom
        if not titles:
            for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                t = entry.find("{http://www.w3.org/2005/Atom}title")
                if t is not None and t.text and len(t.text.strip()) > 10:
                    titles.append(t.text.strip())
        return titles[:40]
    except Exception:
        return []




def _score_best_for_symbol(
    pool: list[str], symbol: str, exclude: set[str]
) -> "SentimentResult":
    """
    Score all headlines in pool for symbol, skip any in exclude set,
    return the SentimentResult with highest |score|.
    Falls back to synthetic pool if nothing matches.
    """
    aliases  = SYMBOL_ALIASES.get(symbol, [])

    # Prefer symbol-specific headlines from the pool
    relevant = [h for h in pool
                if h not in exclude and any(a in h.lower() for a in aliases)]

    # Fall back to anything in pool not yet used
    if not relevant:
        relevant = [h for h in pool if h not in exclude]

    # Fall back to symbol-specific synthetic
    if not relevant:
        relevant = [h for h in SYNTHETIC_HEADLINES
                    if h not in exclude and any(a in h.lower() for a in aliases)]

    # Final fallback: all synthetic
    if not relevant:
        relevant = [h for h in SYNTHETIC_HEADLINES if h not in exclude]

    if not relevant:
        relevant = SYNTHETIC_HEADLINES   # last resort, allow duplicates

    # Score up to 15 candidates, pick strongest signal
    candidates = relevant[:15]
    scored = [(h, score_text(h, symbol)) for h in candidates]
    best_h, best_r = max(scored, key=lambda x: abs(x[1].score))
    best_r.headline = best_h
    return best_r


# ─────────────────────────────────────────────────────────────
# SEED + BACKGROUND FETCHER
# ─────────────────────────────────────────────────────────────

def _seed_from_synthetic() -> None:
    """
    Score synthetic headlines immediately at startup.
    Each symbol gets a DIFFERENT headline — no duplicates shown in the UI.
    """
    used: set[str] = set()
    with _STORE_LOCK:
        for symbol in SYMBOLS_MONITORED:
            result = _score_best_for_symbol(SYNTHETIC_HEADLINES, symbol, used)
            result.source    = "synthetic"
            result.scored_at = time.time()
            _SENTIMENT_STORE[symbol] = result
            used.add(result.headline)


class SentimentBackgroundFetcher(threading.Thread):
    """
    Daemon thread that keeps _SENTIMENT_STORE up to date.

    On start:
      1. Calls _seed_from_synthetic() — instant scores, no network needed.
      2. Loads the ML predictor model (may take 10-30 seconds first time).
      3. Enters the RSS fetch loop every REFRESH_SECS.

    Never writes to st.session_state — only the Streamlit main thread
    does that (in get_cached_sentiment()).
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        # Step 1: seed immediately so UI has data on first render
        _seed_from_synthetic()

        # Step 2: load ML model in background (non-blocking for UI)
        if _ML_AVAILABLE:
            try:
                backend = _predictor_mod.load()
                # Re-seed with the now-loaded model
                _seed_from_synthetic()
            except Exception as e:
                pass   # predictor.load() already prints its own status

        # Step 3: RSS fetch loop
        while not self._stop.is_set():
            self._tick()
            for _ in range(SENTIMENT_REFRESH_SECS * 10):
                if self._stop.is_set():
                    break
                time.sleep(0.1)

    def _tick(self) -> None:
        all_headlines: list[str] = []
        for url in NEWS_FEEDS:
            all_headlines.extend(_fetch_headlines(url))

        pool = all_headlines if all_headlines else SYNTHETIC_HEADLINES
        src  = "live" if all_headlines else "synthetic"

        # Deduplicate: each symbol gets a different headline
        used: set[str] = set()
        with _STORE_LOCK:
            for symbol in SYMBOLS_MONITORED:
                result = _score_best_for_symbol(pool, symbol, used)
                # Source: keep finetuned/finbert label if transformer was used,
                # otherwise mark as live/synthetic based on headline origin
                if result.source not in ("finetuned", "finbert"):
                    result.source = src
                result.scored_at = time.time()
                _SENTIMENT_STORE[symbol] = result
                used.add(result.headline)


# ─────────────────────────────────────────────────────────────
# PUBLIC API  (called from main thread only)
# ─────────────────────────────────────────────────────────────

def get_cached_sentiment(symbol: str) -> Optional[SentimentResult]:
    """
    Read latest sentiment from thread-safe store and mirror to session_state.
    Returns None only if store is completely empty (first few ms of startup).
    """
    with _STORE_LOCK:
        result = _SENTIMENT_STORE.get(symbol)

    if result:
        # Mirror into session_state so processor._attach_sentiment() works
        try:
            import streamlit as st
            from config import K_SENTIMENT_CACHE
            cache = st.session_state.get(K_SENTIMENT_CACHE, {})
            cache[symbol] = result
            st.session_state[K_SENTIMENT_CACHE] = cache
        except Exception:
            pass
        return result

    # Legacy fallback
    try:
        import streamlit as st
        from config import K_SENTIMENT_CACHE
        return st.session_state.get(K_SENTIMENT_CACHE, {}).get(symbol)
    except Exception:
        return None


def get_all_scores() -> dict[str, SentimentResult]:
    with _STORE_LOCK:
        return dict(_SENTIMENT_STORE)


def force_refresh() -> None:
    """Reseed from synthetic headlines immediately. Call from UI button."""
    _seed_from_synthetic()


def current_backend() -> str:
    """Return which ML backend is active."""
    if _ML_AVAILABLE:
        return _predictor_mod.backend_name()
    return "lexicon"


def sentiment_badge_html(result: Optional[SentimentResult]) -> str:
    if result is None:
        return "<span style='color:#2a6080'>— no data</span>"
    color = {"BULLISH":"#00ff88","BEARISH":"#ff4466","NEUTRAL":"#ffaa00"}.get(
        result.label, "#2a6080")
    sign    = "+" if result.score >= 0 else ""
    age_s   = int(result.age_seconds)
    age_str = f"{age_s}s ago" if age_s < 60 else f"{age_s//60}m ago"
    return (
        f"<span style='color:{color};font-size:0.75rem;'>"
        f"{result.label_emoji} {result.label} {sign}{result.score:.2f}</span>"
        f"<span style='color:#1a4060;font-size:0.65rem;'>"
        f" · {age_str} · {result.backend_badge}</span>"
    )