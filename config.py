"""
config.py — Centralised constants & session-state schema  (Phase 5)
"""

# WebSocket
BINANCE_WS_BASE      = "wss://stream.binance.com:9443/ws"
MAX_RECONNECT        = 10
RECONNECT_BASE_DELAY = 2
RECONNECT_MAX_DELAY  = 60

def binance_ws_url(slug: str) -> str:
    return f"{BINANCE_WS_BASE}/{slug.lower()}@trade"

# Connection tokens
STATUS_STOPPED      = "stopped"
STATUS_CONNECTING   = "connecting"
STATUS_CONNECTED    = "connected"
STATUS_RECONNECTING = "reconnecting"
STATUS_ERROR        = "error"

# Processor defaults
DEFAULT_WHALE_THRESHOLD   = 500_000
DEFAULT_CLUSTER_THRESHOLD = 1_000_000
DEFAULT_CLUSTER_WINDOW    = 3.0
DEFAULT_REFRESH_INTERVAL  = 1.0
DEFAULT_MAX_FEED_ROWS     = 100

# ── Phase 5: Profiler / persistence ────────────────────────
import os as _os

# On Streamlit Cloud the working directory is read-only in some contexts.
# Use /tmp which is always writable, falling back to local for development.
_DB_FILENAME = "whale_profiles.db"
PROFILER_DB_PATH = (
    _os.path.join("/tmp", _DB_FILENAME)
    if _os.getenv("STREAMLIT_SHARING_MODE") or not _os.access(".", _os.W_OK)
    else _DB_FILENAME
)
SMART_MONEY_TRADE_COUNT   = 5     # min trades before smart-money tagging
SMART_MONEY_WIN_RATE      = 0.6   # fraction of pre-sentiment-shift buys
HF_WHALE_TRADE_COUNT      = 20    # trades/session to be "high-frequency"
LOYALTY_WINDOW_HOURS      = 24    # hours of history to consider for loyalty

# Session state keys
K_RUNNING           = "running"
K_WS_THREAD         = "ws_thread"
K_TRADE_QUEUE       = "trade_queue"
K_STATUS_QUEUE      = "status_queue"
K_WS_STATUS         = "ws_status"
K_WS_MSG            = "ws_msg"
K_WHALE_FEED        = "whale_feed"
K_CLUSTER_FEED      = "cluster_feed"
K_UNIFIED_FEED      = "unified_feed"
K_CLUSTER_RAW       = "cluster_raw"
K_TOTAL_SCANNED     = "total_scanned"
K_TOTAL_WHALES      = "total_whales"
K_TOTAL_CLUSTERS    = "total_clusters"
K_TOTAL_VOLUME      = "total_volume"
K_EVENT_LOG         = "event_log"
K_SENTIMENT_CACHE   = "sentiment_cache"
K_CLUSTER_BUFFER    = "cluster_buffer"
K_ACTIVE_SYMBOL     = "active_symbol"
K_TOAST_QUEUE       = "toast_queue"
K_SENTIMENT_FILTER  = "sentiment_filter"
K_LAST_ALERT_TIME   = "last_alert_time"
# Phase 5 new keys
K_WALLET_REGISTRY   = "wallet_registry"    # dict[address, WalletProfile]
K_SELECTED_WALLET   = "selected_wallet"    # address string for dossier view
K_PROFILER_READY    = "profiler_ready"     # bool — DB initialised

SESSION_DEFAULTS: dict = {
    K_RUNNING:           False,
    K_ACTIVE_SYMBOL:     "btcusdt",
    K_WS_THREAD:         None,
    K_TRADE_QUEUE:       None,
    K_STATUS_QUEUE:      None,
    K_WS_STATUS:         STATUS_STOPPED,
    K_WS_MSG:            "",
    K_WHALE_FEED:        [],
    K_CLUSTER_FEED:      [],
    K_UNIFIED_FEED:      [],
    K_CLUSTER_RAW:       [],
    K_TOTAL_SCANNED:     0,
    K_TOTAL_WHALES:      0,
    K_TOTAL_CLUSTERS:    0,
    K_TOTAL_VOLUME:      0.0,
    K_EVENT_LOG:         [],
    K_SENTIMENT_CACHE:   {},
    K_CLUSTER_BUFFER:    {},
    K_TOAST_QUEUE:       [],
    K_SENTIMENT_FILTER:  0.0,
    K_LAST_ALERT_TIME:   0.0,
    K_WALLET_REGISTRY:   {},
    K_SELECTED_WALLET:   "",
    K_PROFILER_READY:    False,
}