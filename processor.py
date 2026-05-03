"""
processor.py — Real-time Kappa Filter & Behavioural Clustering  (Phase 5)
Adds: wallet profile enrichment on every whale/cluster event.
"""
import time
import queue
import streamlit as st
from dataclasses import dataclass

import state
import utils
from config import (
    K_TRADE_QUEUE, K_WHALE_FEED, K_CLUSTER_FEED, K_UNIFIED_FEED, K_CLUSTER_RAW,
    K_TOTAL_SCANNED, K_TOTAL_WHALES, K_TOTAL_CLUSTERS,
    K_TOTAL_VOLUME, K_CLUSTER_BUFFER, K_SENTIMENT_CACHE, K_LAST_ALERT_TIME,
    K_WALLET_REGISTRY,
    DEFAULT_WHALE_THRESHOLD, DEFAULT_CLUSTER_THRESHOLD, DEFAULT_CLUSTER_WINDOW,
)

TOAST_DEBOUNCE_SEC = 8


@dataclass
class WhaleEvent:
    trade:     dict
    sentiment: float | None = None
    detection: str = "SINGLE"


@dataclass
class ClusterWhaleEvent:
    symbol:      str
    trades:      list
    cluster_usd: float
    window_sec:  float
    trade_count: int
    first_ts:    str
    last_ts:     str
    sentiment:   float | None = None
    detection:   str = "CLUSTER"


def _attach_sentiment(symbol: str) -> float | None:
    cache: dict = state.get(K_SENTIMENT_CACHE)
    result = cache.get(symbol)
    return getattr(result, "score", None) if result else None


def _evict_old_trades(buffer: list, window_sec: float, now: float) -> list:
    return [t for t in buffer if t["ts_epoch"] >= now - window_sec]


def _check_cluster(symbol: str, cluster_threshold: float,
                   cluster_window: float) -> "ClusterWhaleEvent | None":
    cluster_buf: dict = st.session_state[K_CLUSTER_BUFFER]
    buf = cluster_buf.get(symbol, [])
    if not buf:
        return None
    cluster_usd = sum(t["total_usd"] for t in buf)
    if cluster_usd < cluster_threshold:
        return None
    window_sec = buf[-1]["ts_epoch"] - buf[0]["ts_epoch"]
    event = ClusterWhaleEvent(
        symbol=symbol, trades=list(buf),
        cluster_usd=round(cluster_usd, 2),
        window_sec=round(window_sec, 3),
        trade_count=len(buf),
        first_ts=buf[0]["timestamp"], last_ts=buf[-1]["timestamp"],
        sentiment=_attach_sentiment(symbol),
    )
    cluster_buf[symbol] = []
    st.session_state[K_CLUSTER_BUFFER] = cluster_buf
    return event


def drain_trade_queue(
    whale_threshold:   float = DEFAULT_WHALE_THRESHOLD,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    cluster_window:    float = DEFAULT_CLUSTER_WINDOW,
    max_feed_rows:     int   = 100,
) -> None:
    tq: queue.Queue | None = state.get(K_TRADE_QUEUE)
    if tq is None:
        return
    if not isinstance(st.session_state.get(K_CLUSTER_BUFFER), dict):
        st.session_state[K_CLUSTER_BUFFER] = {}

    now = time.time()

    while True:
        try:
            trade: dict = tq.get_nowait()
        except queue.Empty:
            break

        st.session_state[K_TOTAL_SCANNED] += 1
        symbol = trade["symbol"]

        cluster_buf: dict = st.session_state[K_CLUSTER_BUFFER]
        buf = _evict_old_trades(cluster_buf.get(symbol, []), cluster_window, now)
        buf.append(trade)
        cluster_buf[symbol] = buf
        st.session_state[K_CLUSTER_BUFFER] = cluster_buf

        sentiment = _attach_sentiment(symbol)

        if trade["total_usd"] >= whale_threshold:
            event = WhaleEvent(trade=trade, sentiment=sentiment)
            _emit_whale_event(event, max_feed_rows)

        cluster_event = _check_cluster(symbol, cluster_threshold, cluster_window)
        if cluster_event:
            _emit_cluster_event(cluster_event, max_feed_rows)


def _get_profile(trade: dict, sentiment: float | None):
    """Enrich trade with a wallet profile. Returns profile or None on error."""
    try:
        from modules.profiler import update_profile
        registry: dict = st.session_state.get(K_WALLET_REGISTRY, {})
        sentiment_cache: dict = st.session_state.get(K_SENTIMENT_CACHE, {})
        profile = update_profile(trade, sentiment, registry, sentiment_cache)
        st.session_state[K_WALLET_REGISTRY] = registry
        return profile
    except Exception:
        return None


def _emit_whale_event(event: WhaleEvent, max_rows: int) -> None:
    t         = event.trade
    profile   = _get_profile(t, event.sentiment)
    row       = utils.format_whale_row(t, event.sentiment, profile)

    for key in (K_WHALE_FEED, K_UNIFIED_FEED):
        feed: list = st.session_state[key]
        feed.insert(0, row)
        st.session_state[key] = feed[:max_rows]

    st.session_state[K_TOTAL_WHALES] += 1
    st.session_state[K_TOTAL_VOLUME] += t["total_usd"]

    label_str = f" [{profile.label}]" if profile and profile.label else ""
    tag_str   = f" {profile.tag_badge()}" if profile else ""
    icon = "🟢" if t["side"] == "BUY" else "🔴"
    state.append_log(
        f"🐋 [{t['timestamp']}] WHALE {icon} {t['symbol']} "
        f"${t['total_usd']:,.0f}{label_str}{tag_str}"
        + (f" | s={event.sentiment:+.2f}" if event.sentiment is not None else "")
    )


def _emit_cluster_event(event: ClusterWhaleEvent, max_rows: int) -> None:
    # Use the first trade in the cluster to derive a profile
    rep_trade = event.trades[0] if event.trades else {}
    profile   = _get_profile(rep_trade, event.sentiment) if rep_trade else None
    row       = utils.format_cluster_row(event, profile)
    raw       = utils.format_cluster_raw(event)

    cluster_feed: list = st.session_state[K_CLUSTER_FEED]
    cluster_feed.insert(0, row)
    st.session_state[K_CLUSTER_FEED] = cluster_feed[:max_rows]

    unified: list = st.session_state[K_UNIFIED_FEED]
    unified.insert(0, row)
    st.session_state[K_UNIFIED_FEED] = unified[:max_rows]

    cluster_raw: list = st.session_state[K_CLUSTER_RAW]
    cluster_raw.insert(0, raw)
    st.session_state[K_CLUSTER_RAW] = cluster_raw[:max_rows]

    st.session_state[K_TOTAL_CLUSTERS] += 1
    st.session_state[K_TOTAL_VOLUME]   += event.cluster_usd

    state.append_log(
        f"🔍 [{event.last_ts}] CLUSTER {event.symbol} "
        f"{event.trade_count} trades · ${event.cluster_usd:,.0f} in {event.window_sec:.1f}s"
        + (f" | s={event.sentiment:+.2f}" if event.sentiment is not None else "")
    )

    import time as _t
    last = st.session_state.get(K_LAST_ALERT_TIME, 0.0)
    if _t.time() - last > TOAST_DEBOUNCE_SEC and "HIGH" in raw.get("suspicion", ""):
        tag_str = f" [{profile.tag_badge()}]" if profile else ""
        state.push_toast(
            f"🔍 Stealth Whale{tag_str}: {event.symbol} — "
            f"${event.cluster_usd:,.0f} in {event.trade_count} trades ({event.window_sec:.1f}s)",
            icon="🚨"
        )
        st.session_state[K_LAST_ALERT_TIME] = _t.time()
