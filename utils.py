"""
utils.py — Display Formatting & Data Cleaning Helpers  (Phase 5)
Pure stateless functions — no Streamlit imports, no session_state reads.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from processor import ClusterWhaleEvent
    from modules.profiler import WalletProfile


def format_sentiment(score: float | None) -> str:
    if score is None:
        return "—"
    emoji = "📈" if score > 0.15 else "📉" if score < -0.15 else "➡️ "
    return f"{emoji} {score:+.2f}"


def format_whale_row(trade: dict, sentiment: float | None = None,
                     profile: "WalletProfile | None" = None) -> dict:
    side_icon = "🟢 BUY" if trade["side"] == "BUY" else "🔴 SELL"
    wallet_label = _wallet_label(profile)
    wallet_tag   = profile.tag_badge() if profile else "❓ Unknown"
    return {
        "Type":           "🐋 SINGLE",
        "⏱ Time":        trade["timestamp"],
        "Symbol":         trade["symbol"],
        "Side":           side_icon,
        "Price (USDT)":   f"${trade['price']:>13,.2f}",
        "Quantity":       f"{trade['quantity']:>14,.6f}",
        "💰 Total USD":   f"${trade['total_usd']:>16,.2f}",
        "Sentiment":      format_sentiment(sentiment),
        "🏷 Wallet":      wallet_label,
        "Tag":            wallet_tag,
        "_total_usd":     trade["total_usd"],
        "_sentiment":     sentiment if sentiment is not None else 0.0,
        "_side":          trade["side"],
        "_detection":     "SINGLE",
        "_address":       profile.address if profile else "",
    }


def format_cluster_row(event: "ClusterWhaleEvent",
                       profile: "WalletProfile | None" = None) -> dict:
    avg_size   = event.cluster_usd / max(event.trade_count, 1)
    suspicion  = _suspicion_label(event.trade_count, avg_size)
    wallet_label = _wallet_label(profile)
    wallet_tag   = profile.tag_badge() if profile else "❓ Unknown"
    return {
        "Type":           "🔍 CLUSTER",
        "⏱ Time":        event.last_ts,
        "Symbol":         event.symbol,
        "Side":           "🔀 MIXED",
        "Price (USDT)":   "—",
        "Quantity":       f"{event.trade_count} trades",
        "💰 Total USD":   f"${event.cluster_usd:>16,.2f}",
        "Sentiment":      format_sentiment(event.sentiment),
        "🏷 Wallet":      wallet_label,
        "Tag":            wallet_tag,
        "_total_usd":     event.cluster_usd,
        "_sentiment":     event.sentiment if event.sentiment is not None else 0.0,
        "_side":          "CLUSTER",
        "_detection":     "CLUSTER",
        "_suspicion":     suspicion,
        "_window_sec":    event.window_sec,
        "_trade_count":   event.trade_count,
        "_address":       profile.address if profile else "",
    }


def format_cluster_raw(event: "ClusterWhaleEvent") -> dict:
    return {
        "symbol":      event.symbol,
        "cluster_usd": event.cluster_usd,
        "trade_count": event.trade_count,
        "window_sec":  event.window_sec,
        "first_ts":    event.first_ts,
        "last_ts":     event.last_ts,
        "sentiment":   event.sentiment,
        "trades":      list(event.trades),
        "suspicion":   _suspicion_label(event.trade_count,
                           event.cluster_usd / max(event.trade_count, 1)),
    }


def _wallet_label(profile: "WalletProfile | None") -> str:
    """Return display name for a wallet.
    Known wallets -> their human label (e.g. 'Binance Hot Wallet').
    Unknown wallets -> full address so the user can copy/verify it.
    """
    if profile is None:
        return "—"
    if profile.label:
        return profile.label
    return profile.address   # full address — never truncate unknown wallets


def _suspicion_label(trade_count: int, avg_size: float) -> str:
    if trade_count >= 10 and avg_size < 50_000:
        return "🔴 HIGH"
    elif trade_count >= 5 or avg_size < 150_000:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def log_css_class(line: str) -> str:
    if "🐋" in line or "🔍" in line:
        return "log-whale"
    if "error" in line.lower() or "✖" in line or "Failed" in line or "gave up" in line.lower():
        return "log-error"
    if "📡" in line or "⟳" in line or "Reconnect" in line or "Sentiment" in line:
        return "log-alert"
    return "log-scan"


def normalise_symbol(raw: str) -> str:
    s = raw.upper().strip()
    if "/" in s:
        return s
    for quote in ("USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"):
        if s.endswith(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s


def format_usd(value: float, decimals: int = 0) -> str:
    return f"${value:,.{decimals}f}"


def format_pct(numerator: float, denominator: float, decimals: int = 3) -> str:
    if denominator == 0:
        return f"0.{'0'*decimals}%"
    return f"{numerator / denominator * 100:.{decimals}f}%"


def ts_to_display(epoch: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S %d %b")