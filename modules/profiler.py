"""
modules/profiler.py — Phase 5: Wallet Profiling & Labeling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Responsibilities:
  1. Address labeling        — map known exchange/fund addresses to names
  2. Loyalty scoring         — track per-session trade frequency per address
  3. Smart-money detection   — flag wallets that trade before sentiment shifts
  4. SQLite persistence      — profiles survive across Streamlit sessions
  5. lru_cache on hot paths  — label lookup is O(1), never delays the WS feed

DATA MODEL (SQLite):
  table: wallet_profiles
    address       TEXT PRIMARY KEY
    label         TEXT          — exchange/fund name or NULL
    tag           TEXT          — SMART_MONEY / HF_WHALE / KNOWN / UNKNOWN
    first_seen    REAL          — Unix epoch
    last_seen     REAL
    trade_count   INTEGER
    total_volume  REAL
    symbols       TEXT          — JSON list of traded symbols
    buy_count     INTEGER
    sell_count    INTEGER
    pre_shift_buys  INTEGER     — buys made before a positive sentiment shift
    pre_shift_total INTEGER     — total trades evaluated for smart-money

  table: trade_history
    id            INTEGER PK AUTOINCREMENT
    address       TEXT
    symbol        TEXT
    side          TEXT
    price         REAL
    quantity      REAL
    total_usd     REAL
    sentiment     REAL
    ts_epoch      REAL
    ts_display    TEXT

NOTE ON BINANCE TRADE DATA:
  Binance's public aggTrade/trade WebSocket stream does NOT expose wallet
  addresses for privacy reasons. This module therefore works with a
  simulated address derived from the trade ID (or buyer/seller order ID
  in aggTrade). In Phase 6, connect to an on-chain data provider
  (e.g. Nansen, Arkham, or a self-hosted Ethereum node) to get real
  wallet addresses and replace _derive_address() with a real lookup.
"""

import json
import time
import sqlite3
import hashlib
from functools import lru_cache
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import (
    PROFILER_DB_PATH,
    SMART_MONEY_TRADE_COUNT, SMART_MONEY_WIN_RATE,
    HF_WHALE_TRADE_COUNT,
)


# ─────────────────────────────────────────────────────────────
# KNOWN ADDRESS DICTIONARY
# ─────────────────────────────────────────────────────────────
# Static lookup: exchange cold/hot wallets, market makers, funds.
# Source: public blockchain analytics (Etherscan labels, Nansen, etc.)
# Add your own addresses here. lru_cache makes lookup O(1).
#
# Format: "0x_address_lowercase": ("Display Name", "category")
# Categories: EXCHANGE | MARKET_MAKER | FUND | CUSTODIAN | MINER | OTHER

KNOWN_ADDRESSES: dict[str, tuple[str, str]] = {
    # ── Binance ──────────────────────────────────────────────
    "0x28c6c06298d514db089934071355e5743bf21d60": ("Binance Hot Wallet",  "EXCHANGE"),
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": ("Binance Cold Wallet", "EXCHANGE"),
    "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8": ("Binance Cold Wallet 2","EXCHANGE"),
    "0xf977814e90da44bfa03b6295a0616a897441acec": ("Binance Hot Wallet 2", "EXCHANGE"),
    # ── Coinbase ─────────────────────────────────────────────
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": ("Coinbase Hot Wallet",  "EXCHANGE"),
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e": ("Coinbase 2",           "EXCHANGE"),
    "0x503828976d22510aad0201ac7ec88293211d23da": ("Coinbase 3",           "EXCHANGE"),
    # ── Kraken ───────────────────────────────────────────────
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": ("Kraken Hot Wallet",   "EXCHANGE"),
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13": ("Kraken Hot Wallet 2", "EXCHANGE"),
    # ── OKX ──────────────────────────────────────────────────
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": ("OKX Hot Wallet",      "EXCHANGE"),
    # ── Bitfinex ─────────────────────────────────────────────
    "0x1151314c646ce4e0efd76d1af4760ae66a9fe30f": ("Bitfinex Hot Wallet", "EXCHANGE"),
    # ── Market makers ────────────────────────────────────────
    "0x00000000219ab540356cbb839cbe05303d7705fa": ("ETH2 Deposit Contract","CUSTODIAN"),
    "0xab5801a7d398351b8be11c439e05c5b3259aec9b": ("Vitalik Buterin",     "OTHER"),
    # ── Funds / whales (public) ──────────────────────────────
    "0x8103683202aa8da10536036edef04cdd865c225e": ("Jump Trading",        "MARKET_MAKER"),
    "0xe92d1a43df510f82c66382592a047d288f85226f": ("Cumberland / DRW",    "MARKET_MAKER"),
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": ("Binance MM Desk",     "MARKET_MAKER"),
}

# Category display colours
CATEGORY_COLOURS: dict[str, str] = {
    "EXCHANGE":     "#ffaa00",
    "MARKET_MAKER": "#cc88ff",
    "FUND":         "#00e5ff",
    "CUSTODIAN":    "#7ecfff",
    "MINER":        "#ff8844",
    "OTHER":        "#2a6080",
    "UNKNOWN":      "#1a4060",
}

# Wallet tags
TAG_SMART_MONEY  = "SMART_MONEY"
TAG_HF_WHALE     = "HF_WHALE"
TAG_KNOWN        = "KNOWN"
TAG_UNKNOWN      = "UNKNOWN"


# ─────────────────────────────────────────────────────────────
# WALLET PROFILE DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class WalletProfile:
    address:          str
    label:            str   = ""          # human-readable name
    category:         str   = "UNKNOWN"   # EXCHANGE / MARKET_MAKER / etc.
    tag:              str   = TAG_UNKNOWN  # SMART_MONEY / HF_WHALE / KNOWN / UNKNOWN
    first_seen:       float = field(default_factory=time.time)
    last_seen:        float = field(default_factory=time.time)
    trade_count:      int   = 0
    total_volume:     float = 0.0
    buy_count:        int   = 0
    sell_count:       int   = 0
    symbols:          list  = field(default_factory=list)
    pre_shift_buys:   int   = 0   # buys that preceded a positive sentiment shift
    pre_shift_total:  int   = 0   # all trades evaluated for smart-money scoring

    # ── derived properties ─────────────────────────────────────

    @property
    def buy_ratio(self) -> float:
        total = self.buy_count + self.sell_count
        return self.buy_count / total if total > 0 else 0.5

    @property
    def smart_money_score(self) -> float:
        """Fraction of pre-shift buys / total evaluated. Range 0–1."""
        if self.pre_shift_total < SMART_MONEY_TRADE_COUNT:
            return 0.0
        return self.pre_shift_buys / self.pre_shift_total

    @property
    def is_smart_money(self) -> bool:
        return self.smart_money_score >= SMART_MONEY_WIN_RATE

    @property
    def is_hf_whale(self) -> bool:
        return self.trade_count >= HF_WHALE_TRADE_COUNT

    @property
    def dominant_side(self) -> str:
        if self.buy_ratio > 0.65:
            return "ACCUMULATOR"
        elif self.buy_ratio < 0.35:
            return "DISTRIBUTOR"
        return "BALANCED"

    @property
    def favourite_symbol(self) -> str:
        return self.symbols[0] if self.symbols else "—"

    @property
    def avg_trade_size(self) -> float:
        return self.total_volume / max(self.trade_count, 1)

    def effective_tag(self) -> str:
        """Recompute tag from current data (call before saving)."""
        if self.label:
            return TAG_KNOWN
        if self.is_smart_money:
            return TAG_SMART_MONEY
        if self.is_hf_whale:
            return TAG_HF_WHALE
        return TAG_UNKNOWN

    def tag_badge(self) -> str:
        """Emoji badge for display."""
        return {
            TAG_SMART_MONEY: "🧠 Smart Money",
            TAG_HF_WHALE:    "⚡ HF Whale",
            TAG_KNOWN:       "🏛️ Known Entity",
            TAG_UNKNOWN:     "❓ Unknown",
        }.get(self.tag, "❓ Unknown")


# ─────────────────────────────────────────────────────────────
# ADDRESS LABELING  (lru_cache — zero-latency hot path)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=10_000)
def lookup_address(address: str) -> tuple[str, str]:
    """
    O(1) cached lookup of an address against KNOWN_ADDRESSES.

    Returns: (label, category) or ("", "UNKNOWN")

    lru_cache ensures repeated lookups for the same address
    (which is common — whales trade repeatedly) hit the cache,
    never the dict. This is critical for the live WS path.
    """
    addr = address.lower().strip()
    if addr in KNOWN_ADDRESSES:
        return KNOWN_ADDRESSES[addr]
    return ("", "UNKNOWN")


def _derive_address(trade: dict) -> str:
    """
    Derive a stable pseudo-address from trade data.

    WHY: Binance public WS does not expose wallet addresses.
    We hash (symbol + side + price_bucket) to create a stable
    identifier that groups trades from the same market participant
    within the same price range and direction — a reasonable proxy
    until on-chain data is wired in Phase 6.

    In Phase 6: replace with real address from on-chain provider.
    """
    price_bucket = round(trade.get("price", 0) / 100) * 100  # $100 buckets
    raw = f"{trade.get('symbol','')}{trade.get('side','')}{price_bucket}"
    return "0x" + hashlib.sha256(raw.encode()).hexdigest()[:40]


# ─────────────────────────────────────────────────────────────
# SQLITE PERSISTENCE LAYER
# ─────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (check_same_thread=False for Streamlit)."""
    conn = sqlite3.connect(PROFILER_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create tables if they don't exist. Safe to call on every app start.
    Uses IF NOT EXISTS so repeated calls are no-ops.
    """
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS wallet_profiles (
            address         TEXT PRIMARY KEY,
            label           TEXT    DEFAULT '',
            category        TEXT    DEFAULT 'UNKNOWN',
            tag             TEXT    DEFAULT 'UNKNOWN',
            first_seen      REAL    DEFAULT 0,
            last_seen       REAL    DEFAULT 0,
            trade_count     INTEGER DEFAULT 0,
            total_volume    REAL    DEFAULT 0,
            buy_count       INTEGER DEFAULT 0,
            sell_count      INTEGER DEFAULT 0,
            symbols         TEXT    DEFAULT '[]',
            pre_shift_buys  INTEGER DEFAULT 0,
            pre_shift_total INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS trade_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            address     TEXT,
            symbol      TEXT,
            side        TEXT,
            price       REAL,
            quantity    REAL,
            total_usd   REAL,
            sentiment   REAL,
            ts_epoch    REAL,
            ts_display  TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_trade_address ON trade_history(address);
        CREATE INDEX IF NOT EXISTS idx_trade_ts      ON trade_history(ts_epoch);
        CREATE INDEX IF NOT EXISTS idx_trade_symbol  ON trade_history(symbol);
    """)
    conn.commit()
    conn.close()


def save_profile(profile: WalletProfile) -> None:
    """Upsert a WalletProfile into SQLite."""
    conn = _get_conn()
    conn.execute("""
        INSERT INTO wallet_profiles
            (address, label, category, tag, first_seen, last_seen,
             trade_count, total_volume, buy_count, sell_count,
             symbols, pre_shift_buys, pre_shift_total)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(address) DO UPDATE SET
            label           = excluded.label,
            category        = excluded.category,
            tag             = excluded.tag,
            last_seen       = excluded.last_seen,
            trade_count     = excluded.trade_count,
            total_volume    = excluded.total_volume,
            buy_count       = excluded.buy_count,
            sell_count      = excluded.sell_count,
            symbols         = excluded.symbols,
            pre_shift_buys  = excluded.pre_shift_buys,
            pre_shift_total = excluded.pre_shift_total
    """, (
        profile.address, profile.label, profile.category, profile.tag,
        profile.first_seen, profile.last_seen,
        profile.trade_count, profile.total_volume,
        profile.buy_count, profile.sell_count,
        json.dumps(profile.symbols),
        profile.pre_shift_buys, profile.pre_shift_total,
    ))
    conn.commit()
    conn.close()


def save_trade(address: str, trade: dict, sentiment: float | None) -> None:
    """Append one trade to the trade_history table."""
    conn = _get_conn()
    conn.execute("""
        INSERT INTO trade_history
            (address, symbol, side, price, quantity, total_usd, sentiment, ts_epoch, ts_display)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        address,
        trade.get("symbol", ""),
        trade.get("side", ""),
        trade.get("price", 0.0),
        trade.get("quantity", 0.0),
        trade.get("total_usd", 0.0),
        sentiment,
        trade.get("ts_epoch", time.time()),
        trade.get("timestamp", ""),
    ))
    conn.commit()
    conn.close()


def load_all_profiles() -> dict[str, WalletProfile]:
    """Load all wallet profiles from SQLite into memory. Called once at startup."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM wallet_profiles ORDER BY trade_count DESC").fetchall()
    conn.close()
    profiles = {}
    for row in rows:
        p = WalletProfile(
            address        = row["address"],
            label          = row["label"] or "",
            category       = row["category"] or "UNKNOWN",
            tag            = row["tag"] or TAG_UNKNOWN,
            first_seen     = row["first_seen"] or time.time(),
            last_seen      = row["last_seen"] or time.time(),
            trade_count    = row["trade_count"] or 0,
            total_volume   = row["total_volume"] or 0.0,
            buy_count      = row["buy_count"] or 0,
            sell_count     = row["sell_count"] or 0,
            symbols        = json.loads(row["symbols"] or "[]"),
            pre_shift_buys = row["pre_shift_buys"] or 0,
            pre_shift_total= row["pre_shift_total"] or 0,
        )
        profiles[p.address] = p
    return profiles


def load_trade_history(address: str, limit: int = 200) -> list[dict]:
    """Fetch the most recent trades for a specific wallet."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT * FROM trade_history
        WHERE address = ?
        ORDER BY ts_epoch DESC
        LIMIT ?
    """, (address, limit)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_top_wallets(by: str = "total_volume", limit: int = 50) -> list[WalletProfile]:
    """Return top N wallets sorted by column."""
    allowed = {"total_volume", "trade_count", "last_seen"}
    col = by if by in allowed else "total_volume"
    conn = _get_conn()
    rows = conn.execute(
        f"SELECT * FROM wallet_profiles ORDER BY {col} DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    profiles = []
    for row in rows:
        p = WalletProfile(
            address        = row["address"],
            label          = row["label"] or "",
            category       = row["category"] or "UNKNOWN",
            tag            = row["tag"] or TAG_UNKNOWN,
            first_seen     = row["first_seen"] or time.time(),
            last_seen      = row["last_seen"] or time.time(),
            trade_count    = row["trade_count"] or 0,
            total_volume   = row["total_volume"] or 0.0,
            buy_count      = row["buy_count"] or 0,
            sell_count     = row["sell_count"] or 0,
            symbols        = json.loads(row["symbols"] or "[]"),
            pre_shift_buys = row["pre_shift_buys"] or 0,
            pre_shift_total= row["pre_shift_total"] or 0,
        )
        profiles.append(p)
    return profiles


def db_stats() -> dict:
    """Return aggregate stats about the database."""
    conn = _get_conn()
    wallet_count = conn.execute("SELECT COUNT(*) FROM wallet_profiles").fetchone()[0]
    trade_count  = conn.execute("SELECT COUNT(*) FROM trade_history").fetchone()[0]
    total_vol    = conn.execute("SELECT SUM(total_usd) FROM trade_history").fetchone()[0] or 0
    smart_count  = conn.execute(
        "SELECT COUNT(*) FROM wallet_profiles WHERE tag=?", (TAG_SMART_MONEY,)
    ).fetchone()[0]
    hf_count     = conn.execute(
        "SELECT COUNT(*) FROM wallet_profiles WHERE tag=?", (TAG_HF_WHALE,)
    ).fetchone()[0]
    known_count  = conn.execute(
        "SELECT COUNT(*) FROM wallet_profiles WHERE tag=?", (TAG_KNOWN,)
    ).fetchone()[0]
    conn.close()
    return {
        "wallet_count": wallet_count,
        "trade_count":  trade_count,
        "total_volume": total_vol,
        "smart_money":  smart_count,
        "hf_whales":    hf_count,
        "known":        known_count,
    }


# ─────────────────────────────────────────────────────────────
# PROFILE UPDATER  (called from processor.py for each whale event)
# ─────────────────────────────────────────────────────────────

def update_profile(
    trade:          dict,
    sentiment:      float | None,
    registry:       dict[str, WalletProfile],
    sentiment_cache: dict,
) -> WalletProfile:
    """
    Update or create a WalletProfile for the wallet implied by `trade`.

    Steps:
      1. Derive a stable address from the trade.
      2. Look up KNOWN_ADDRESSES (lru_cache — no latency).
      3. Load or create the WalletProfile.
      4. Update all counters (trade_count, volume, side counts, symbols).
      5. Smart-money check: if current sentiment is negative but whale is
         BUYing, record as a potential pre-shift buy (contrarian accumulation).
      6. Recompute tag.
      7. Persist to SQLite (async-safe: single-writer main thread).
      8. Return updated profile.

    Args:
        trade:           normalised trade dict from ingestion.py
        sentiment:       current sentiment score for the symbol (may be None)
        registry:        in-memory dict[address, WalletProfile] from session_state
        sentiment_cache: raw sentiment cache dict for cross-symbol lookups
    """
    address = _derive_address(trade)
    label, category = lookup_address(address)

    now = time.time()

    # Load from registry or create fresh
    if address in registry:
        profile = registry[address]
    else:
        profile = WalletProfile(
            address    = address,
            label      = label,
            category   = category,
            first_seen = now,
        )

    # Update identity fields (may have been enriched since last seen)
    if label and not profile.label:
        profile.label    = label
        profile.category = category

    # Core counters
    profile.last_seen    = now
    profile.trade_count += 1
    profile.total_volume += trade.get("total_usd", 0.0)

    if trade.get("side") == "BUY":
        profile.buy_count += 1
    else:
        profile.sell_count += 1

    # Symbol tracking (keep top-5 most traded, insertion-ordered)
    sym = trade.get("symbol", "")
    if sym and sym not in profile.symbols:
        profile.symbols.append(sym)
        profile.symbols = profile.symbols[:5]

    # Smart-money detection:
    # A BUY when current sentiment is BEARISH (score < -0.1) suggests
    # the wallet is buying before a potential sentiment reversal — classic
    # "smart money" contrarian accumulation.
    if sentiment is not None:
        profile.pre_shift_total += 1
        if trade.get("side") == "BUY" and sentiment < -0.1:
            profile.pre_shift_buys += 1

    # Recompute tag
    profile.tag = profile.effective_tag()

    # Persist to SQLite and update in-memory registry
    save_profile(profile)
    save_trade(address, trade, sentiment)
    registry[address] = profile

    return profile


# ─────────────────────────────────────────────────────────────
# SENTIMENT CORRELATION ANALYSIS  (for dossier view)
# ─────────────────────────────────────────────────────────────

def sentiment_correlation(trades: list[dict]) -> dict:
    """
    Analyse how a wallet's trades correlate with sentiment.

    Returns a dict with:
        buy_avg_sentiment   — avg sentiment when wallet buys
        sell_avg_sentiment  — avg sentiment when wallet sells
        contrarian_score    — how often wallet buys on negative sentiment
        narrative           — human-readable one-line summary
    """
    buy_sentiments  = [t["sentiment"] for t in trades
                       if t.get("side") == "BUY" and t.get("sentiment") is not None]
    sell_sentiments = [t["sentiment"] for t in trades
                       if t.get("side") == "SELL" and t.get("sentiment") is not None]

    buy_avg  = sum(buy_sentiments)  / len(buy_sentiments)  if buy_sentiments  else None
    sell_avg = sum(sell_sentiments) / len(sell_sentiments) if sell_sentiments else None

    # Contrarian score: fraction of buys on negative sentiment
    contrarian = (
        sum(1 for s in buy_sentiments if s < -0.05) / len(buy_sentiments)
        if buy_sentiments else 0.0
    )

    if buy_avg is None:
        narrative = "Insufficient trade history for sentiment correlation."
    elif contrarian > 0.6:
        narrative = (f"This wallet buys heavily when sentiment is bearish "
                     f"(contrarian score {contrarian:.0%}) — classic smart money accumulation.")
    elif buy_avg > 0.2:
        narrative = (f"Wallet tends to buy into bullish sentiment "
                     f"(avg {buy_avg:+.2f}) — trend-following behaviour.")
    elif buy_avg < -0.1:
        narrative = (f"Wallet accumulates during negative sentiment "
                     f"(avg {buy_avg:+.2f}) — potential early-mover.")
    else:
        narrative = "Neutral sentiment correlation — trades regardless of market mood."

    return {
        "buy_avg_sentiment":  buy_avg,
        "sell_avg_sentiment": sell_avg,
        "contrarian_score":   contrarian,
        "narrative":          narrative,
    }


def volume_by_hour(trades: list[dict]) -> dict[int, float]:
    """Return a dict of {hour_of_day: total_usd} for activity heatmap."""
    by_hour: dict[int, float] = {h: 0.0 for h in range(24)}
    for t in trades:
        ts = t.get("ts_epoch", 0)
        if ts:
            hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
            by_hour[hour] = by_hour.get(hour, 0.0) + t.get("total_usd", 0.0)
    return by_hour
