"""
main.py — Streamlit Entry Point · Crypto Whale Watcher Phase 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run with:
    streamlit run main.py

Phase 5 additions vs Phase 4:
  • modules/profiler.py   — SQLite wallet registry, address labeling (lru_cache),
                            loyalty scoring, smart-money detection
  • Wallet Intelligence tab — leaderboard of all profiled wallets
  • Wallet Dossier expander — per-wallet deep-dive (volume, assets, sentiment
                              correlation, hourly heatmap, trade history)
  • DB stats in sidebar    — live count of profiled wallets / stored trades
  • All feed rows enriched  — 🏷 Wallet label + Tag column in every table
  • Persistent storage      — wallet_profiles.db survives across sessions

ANTI-FLICKER STRATEGY (unchanged from Phase 4):
  All session_state mutations happen before any st.* render call.
"""

import time
from datetime import datetime
import streamlit as st

import state
import ingestion
import processor
import sentiment
import utils
from modules import visualization as viz
from modules.profiler import (
    init_db, load_all_profiles, get_top_wallets,
    load_trade_history, db_stats,
)
from config import (
    K_RUNNING, K_WS_STATUS, K_WS_MSG,
    K_WHALE_FEED, K_CLUSTER_FEED, K_UNIFIED_FEED, K_CLUSTER_RAW,
    K_TOTAL_SCANNED, K_TOTAL_WHALES, K_TOTAL_CLUSTERS, K_TOTAL_VOLUME,
    K_EVENT_LOG, K_SENTIMENT_CACHE, K_TRADE_QUEUE, K_ACTIVE_SYMBOL,
    K_TOAST_QUEUE, K_SENTIMENT_FILTER,
    K_WALLET_REGISTRY, K_SELECTED_WALLET, K_PROFILER_READY,
    STATUS_STOPPED, STATUS_CONNECTING, STATUS_CONNECTED,
    STATUS_RECONNECTING, STATUS_ERROR,
    DEFAULT_WHALE_THRESHOLD, DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_CLUSTER_WINDOW, DEFAULT_REFRESH_INTERVAL, DEFAULT_MAX_FEED_ROWS,
)


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Whale Watcher · Phase 5",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    background-color: #030d14 !important;
    color: #c8f0ff !important;
    font-family: 'Share Tech Mono', monospace !important;
}
section[data-testid="stSidebar"] {
    background: #020a10 !important;
    border-right: 1px solid #0a4060 !important;
}
section[data-testid="stSidebar"] * { color: #7ecfff !important; }
[data-testid="collapsedControl"] { color:#00e5ff !important; background:#020a10 !important; }

.whale-header {
    font-family:'Orbitron',monospace; font-size:2rem; font-weight:900;
    color:#00e5ff; text-shadow:0 0 20px #00e5ff88,0 0 50px #00e5ff22;
    letter-spacing:0.12em; margin-bottom:0;
}
.whale-sub {
    font-size:0.72rem; color:#2a6080;
    letter-spacing:0.22em; text-transform:uppercase; margin-bottom:0;
}
.badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    padding:0.25rem 0.85rem; border-radius:2px;
    font-size:0.68rem; letter-spacing:0.15em; margin-bottom:0.5rem;
}
.badge-live       { background:#001a0d; border:1px solid #00ff44; color:#00ff44; }
.badge-connecting { background:#0d0d00; border:1px solid #ffcc00; color:#ffcc00; }
.badge-paused     { background:#1a0a00; border:1px solid #ff6600; color:#ff6600; }
.badge-error      { background:#1a0008; border:1px solid #ff0044; color:#ff0044; }
.live-dot {
    width:7px; height:7px; background:#00ff44; border-radius:50%;
    animation:blink 1s infinite; display:inline-block;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.15} }
.section-label {
    font-size:0.6rem; color:#2a6080; letter-spacing:0.25em; text-transform:uppercase;
    border-bottom:1px solid #0a2030; padding-bottom:0.28rem; margin-bottom:0.75rem;
}
.stTabs [data-baseweb="tab-list"] {
    background:transparent !important; border-bottom:1px solid #0a2030; gap:0.5rem;
}
.stTabs [data-baseweb="tab"] {
    color:#2a6080!important; font-family:'Share Tech Mono',monospace;
    font-size:0.76rem; letter-spacing:0.1em; padding:0.4rem 1rem;
    border-radius:2px 2px 0 0;
}
.stTabs [aria-selected="true"] {
    color:#00e5ff!important; border-bottom:2px solid #00e5ff!important;
    background:#041822!important;
}
.stDataFrame { border:1px solid #0a4060!important; border-radius:2px!important; }
iframe { background:#020d16!important; }
details {
    border:1px solid #0a4060!important; background:#041822!important;
    border-radius:3px!important; margin-bottom:0.5rem!important;
}
summary {
    color:#c8f0ff!important; font-family:'Share Tech Mono',monospace!important;
    font-size:0.78rem!important; padding:0.6rem 1rem!important;
}
summary:hover { background:#061f30!important; }
.stVegaLiteChart { background:#041822!important; }
.stButton > button {
    font-family:'Share Tech Mono',monospace!important;
    background:transparent!important; border:1px solid #00e5ff!important;
    color:#00e5ff!important; letter-spacing:0.12em!important;
    border-radius:2px!important; padding:0.45rem 1rem!important;
    transition:all 0.2s!important; width:100%;
}
.stButton > button:hover { background:#00e5ff18!important; }
.stButton > button[kind="primary"] { border-color:#ff4466!important; color:#ff4466!important; }
.stButton > button[kind="primary"]:hover { background:#ff446618!important; }
div[data-testid="stInfo"] {
    background:#041822!important; border:1px solid #0a4060!important;
    color:#7ecfff!important; font-family:'Share Tech Mono',monospace; font-size:0.75rem;
}
div[data-testid="stToast"] {
    background:#0a1e2a!important; border:1px solid #cc88ff!important;
    color:#c8f0ff!important; font-family:'Share Tech Mono',monospace!important;
    font-size:0.75rem!important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#00e5ff!important; }
.sent-card {
    background:#041822; border:1px solid #0a4060; border-radius:3px;
    padding:0.5rem 0.8rem; margin-bottom:0.5rem;
}
.sent-symbol { font-size:0.65rem; color:#2a6080; letter-spacing:0.15em; }
#MainMenu, footer,{ visibility:hidden; }
header { visibility:transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Init session state
# ─────────────────────────────────────────────────────────────
state.init()


# ─────────────────────────────────────────────────────────────
# STEP 2 — Init SQLite DB + load persisted wallet profiles (once)
# ─────────────────────────────────────────────────────────────
if not state.get(K_PROFILER_READY):
    init_db()
    persisted = load_all_profiles()
    # Merge persisted profiles into the in-memory registry
    registry: dict = st.session_state.get(K_WALLET_REGISTRY, {})
    registry.update(persisted)
    st.session_state[K_WALLET_REGISTRY] = registry
    state.set(K_PROFILER_READY, True)


# ─────────────────────────────────────────────────────────────
# STEP 3 — Start sentiment background fetcher (once)
# ─────────────────────────────────────────────────────────────
if "sentiment_thread" not in st.session_state:
    sent_thread = sentiment.SentimentBackgroundFetcher()
    sent_thread.start()
    st.session_state["sentiment_thread"] = sent_thread


# ─────────────────────────────────────────────────────────────
# STEP 4 — Read persisted slider values
# ─────────────────────────────────────────────────────────────
_threshold         = st.session_state.get("_threshold",         DEFAULT_WHALE_THRESHOLD)
_cluster_threshold = st.session_state.get("_cluster_threshold", DEFAULT_CLUSTER_THRESHOLD)
_cluster_window    = st.session_state.get("_cluster_window",    DEFAULT_CLUSTER_WINDOW)
_max_rows          = st.session_state.get("_max_rows",          DEFAULT_MAX_FEED_ROWS)
_sentiment_filter  = st.session_state.get("_sentiment_filter",  0.0)


# ─────────────────────────────────────────────────────────────
# STEP 5 — Drain queues BEFORE rendering (anti-flicker)
# ─────────────────────────────────────────────────────────────
if state.get(K_RUNNING):
    ingestion.drain_status_queue()
    processor.drain_trade_queue(
        whale_threshold   = _threshold,
        cluster_threshold = _cluster_threshold,
        cluster_window    = _cluster_window,
        max_feed_rows     = _max_rows,
    )


# ─────────────────────────────────────────────────────────────
# STEP 6 — Fire pending toasts
# ─────────────────────────────────────────────────────────────
for _toast in st.session_state.get(K_TOAST_QUEUE, []):
    st.toast(_toast["msg"], icon=_toast.get("icon", "🐋"))
st.session_state[K_TOAST_QUEUE] = []


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Orbitron,monospace;font-size:0.95rem;font-weight:700;
         color:#00e5ff;letter-spacing:0.1em;padding-bottom:0.6rem;
         border-bottom:1px solid #0a4060;margin-bottom:1.1rem;
         text-shadow:0 0 12px #00e5ff66;'>
        🐋 WHALE WATCHER
    </div>
    """, unsafe_allow_html=True)

    # ── Start / Stop ──────────────────────────────────────────
    running = state.get(K_RUNNING)
    if running:
        if st.button("⏹  STOP TRACKING", use_container_width=True, type="primary"):
            ingestion.stop_stream()
            st.rerun()
    else:
        if st.button("▶  START TRACKING", use_container_width=True):
            ingestion.start_stream(slug=st.session_state.get("_symbol_slug", "btcusdt"))
            st.rerun()

    if st.button("🗑  CLEAR LOG", use_container_width=True):
        state.reset_stats()
        st.rerun()

    _div = "<div style='margin:0.75rem 0;border-top:1px solid #0a2030'></div>"

    st.markdown(_div, unsafe_allow_html=True)

    # ── Symbol selector ───────────────────────────────────────
    st.markdown('<div class="section-label">Symbol</div>', unsafe_allow_html=True)
    symbol_options = {
        "BTC/USDT":  "btcusdt",
        "ETH/USDT":  "ethusdt",
        "SOL/USDT":  "solusdt",
        "BNB/USDT":  "bnbusdt",
        "XRP/USDT":  "xrpusdt",
        "AVAX/USDT": "avaxusdt",
        "DOGE/USDT": "dogeusdt",
        "ARB/USDT":  "arbusdt",
    }
    slug_to_display = {v: k for k, v in symbol_options.items()}
    _active_slug    = state.get(K_ACTIVE_SYMBOL)
    _active_display = slug_to_display.get(_active_slug, "BTC/USDT")
    _default_idx    = list(symbol_options.keys()).index(_active_display)

    selected_symbol = st.selectbox(
        "Trading Pair", options=list(symbol_options.keys()),
        index=_default_idx, key="ctrl_symbol", label_visibility="collapsed",
    )
    selected_slug = symbol_options[selected_symbol]

    if selected_slug != _active_slug and state.get(K_RUNNING):
        ingestion.switch_symbol(selected_slug)
        state.append_log(
            f"📡 [{datetime.utcnow().strftime('%H:%M:%S')}] "
            f"Switched stream → {selected_symbol}"
        )
        st.rerun()

    st.session_state["_symbol_slug"] = selected_slug
    st.markdown(_div, unsafe_allow_html=True)

    # ── Single-trade threshold ────────────────────────────────
    st.markdown('<div class="section-label">Single-Trade Threshold</div>',
                unsafe_allow_html=True)
    whale_threshold = float(st.slider(
        "whale_threshold", min_value=10_000, max_value=5_000_000,
        value=int(_threshold), step=10_000, format="$%d",
        help="Flag any single trade above this USD value.",
        key="ctrl_whale_threshold", label_visibility="collapsed",
    ))
    st.markdown(
        f"<div style='font-family:Orbitron,monospace;font-size:0.82rem;"
        f"color:#00e5ff;margin-top:-0.3rem;'>{utils.format_usd(whale_threshold)}</div>",
        unsafe_allow_html=True
    )
    st.session_state["_threshold"] = whale_threshold
    st.markdown(_div, unsafe_allow_html=True)

    # ── Cluster threshold ─────────────────────────────────────
    st.markdown('<div class="section-label">Cluster Threshold</div>',
                unsafe_allow_html=True)
    cluster_threshold = float(st.slider(
        "cluster_threshold", min_value=100_000, max_value=5_000_000,
        value=int(_cluster_threshold), step=50_000, format="$%d",
        key="ctrl_cluster_threshold", label_visibility="collapsed",
    ))
    st.markdown(
        f"<div style='font-family:Orbitron,monospace;font-size:0.82rem;"
        f"color:#cc88ff;margin-top:-0.3rem;'>{utils.format_usd(cluster_threshold)}</div>",
        unsafe_allow_html=True
    )
    st.session_state["_cluster_threshold"] = cluster_threshold
    st.markdown(_div, unsafe_allow_html=True)

    # ── Cluster window ────────────────────────────────────────
    st.markdown('<div class="section-label">Cluster Window</div>',
                unsafe_allow_html=True)
    cluster_window = float(st.slider(
        "cluster_window", min_value=1.0, max_value=30.0,
        value=float(_cluster_window), step=0.5,
        key="ctrl_cluster_window", label_visibility="collapsed",
    ))
    st.markdown(
        f"<div style='font-family:Orbitron,monospace;font-size:0.82rem;"
        f"color:#ffaa00;margin-top:-0.3rem;'>{cluster_window:.1f}s</div>",
        unsafe_allow_html=True
    )
    st.session_state["_cluster_window"] = cluster_window
    st.markdown(_div, unsafe_allow_html=True)

    # ── Sentiment filter ──────────────────────────────────────
    st.markdown('<div class="section-label">Min Sentiment Strength</div>',
                unsafe_allow_html=True)
    sentiment_filter = float(st.slider(
        "sentiment_filter", min_value=0.0, max_value=1.0,
        value=float(_sentiment_filter), step=0.05, format="%.2f",
        key="ctrl_sentiment_filter", label_visibility="collapsed",
    ))
    st.markdown(
        f"<div style='font-family:Orbitron,monospace;font-size:0.82rem;"
        f"color:#7ecfff;margin-top:-0.3rem;'>"
        f"{'All' if sentiment_filter == 0 else f'|s| ≥ {sentiment_filter:.2f}'}</div>",
        unsafe_allow_html=True
    )
    st.session_state["_sentiment_filter"] = sentiment_filter
    state.set(K_SENTIMENT_FILTER, sentiment_filter)
    st.markdown(_div, unsafe_allow_html=True)

    # ── Feed settings ─────────────────────────────────────────
    st.markdown('<div class="section-label">Feed Settings</div>',
                unsafe_allow_html=True)
    refresh_interval = float(st.slider(
        "UI refresh (sec)", min_value=0.5, max_value=5.0,
        value=DEFAULT_REFRESH_INTERVAL, step=0.5, key="ctrl_refresh",
    ))
    max_rows = int(st.slider(
        "Max feed rows", min_value=10, max_value=200,
        value=int(_max_rows), key="ctrl_maxrows",
    ))
    st.session_state["_max_rows"] = max_rows
    st.markdown(_div, unsafe_allow_html=True)

    # ── Wallet DB stats ───────────────────────────────────────
    st.markdown('<div class="section-label">Wallet Intelligence DB</div>',
                unsafe_allow_html=True)
    try:
        _stats = db_stats()
        viz.render_db_stats(_stats)
    except Exception:
        st.markdown(
            "<div style='font-size:0.62rem;color:#1a4060;'>DB initialising…</div>",
            unsafe_allow_html=True
        )
    st.markdown(_div, unsafe_allow_html=True)

    # ── Connection ────────────────────────────────────────────
    st.markdown('<div class="section-label">Connection</div>', unsafe_allow_html=True)
    viz.render_connection_status(state.get(K_WS_STATUS), state.get(K_WS_MSG))

    st.markdown(_div, unsafe_allow_html=True)

    # ── ML Model status ───────────────────────────────────────
    st.markdown('<div class="section-label">Sentiment Model</div>',
                unsafe_allow_html=True)
    _backend = sentiment.current_backend()
    _backend_colour = {
        "finetuned": "#00ff88",
        "finbert":   "#cc88ff",
        "lexicon":   "#ffaa00",
        "loading":   "#ffcc00",
        "not_loaded":"#2a6080",
    }.get(_backend, "#2a6080")
    _backend_label = {
        "finetuned": "🧠 Fine-tuned FinBERT",
        "finbert":   "🤖 Base FinBERT",
        "lexicon":   "📖 Lexicon Scorer",
        "loading":   "⏳ Loading model…",
        "not_loaded":"⚪ Not loaded",
    }.get(_backend, _backend)

    # Training metadata if fine-tuned model exists
    try:
        from ml.predictor import training_meta
        meta = training_meta()
    except Exception:
        meta = None

    _train_hint = (
        "<div style='color:#2a6080;margin-top:0.25rem;'>"
        "Train your own model:<br>"
        "<code style='color:#7ecfff;'>python ml/train.py</code></div>"
        if _backend != "finetuned" else ""
    )
    _meta_html = (
        f"<div style='color:#2a6080;margin-top:0.25rem;'>"
        f"Val F1: <b style='color:{_backend_colour}'>{meta['best_val_f1']:.3f}</b>"
        f" &nbsp;·&nbsp; Acc: <b style='color:{_backend_colour}'>{meta['best_val_acc']:.3f}</b><br>"
        f"Trained: {meta['trained_at']}</div>"
        if meta else ""
    )
    st.markdown(
        f"<div style='background:#041822;border:1px solid #0a4060;"
        f"border-left:3px solid {_backend_colour};"
        f"padding:0.55rem 0.85rem;border-radius:2px;font-size:0.65rem;'>"
        f"<div style='color:{_backend_colour};font-family:Orbitron,monospace;"
        f"font-size:0.75rem;font-weight:700;'>{_backend_label}</div>"
        f"{_train_hint}{_meta_html}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style='margin-top:1.5rem;font-size:0.56rem;color:#1a3a50;line-height:1.9;'>
        PHASE 5 · LIVE MODE<br>
        ─────────────────────<br>
        Source: Binance WS<br>
        Filter: single + cluster<br>
        Sentiment: ML model<br>
        Wallets: SQLite profiles<br>
        Labels: lru_cache O(1)<br>
        Smart money: detected<br><br>
        Phase 6 →<br>
        On-chain wallet data<br>
        Kafka + ClickHouse
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
hdr_col, badge_col = st.columns([3, 1])
with hdr_col:
    st.markdown('<div class="whale-header">🐋 CRYPTO WHALE WATCHER</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="whale-sub">Kappa Architecture · Phase 5 · '
        'Wallet Intelligence · Profiling · Smart Money Detection</div>',
        unsafe_allow_html=True
    )

ws_status    = state.get(K_WS_STATUS)
_sym_display = st.session_state.get("ctrl_symbol", "BTC/USDT")
with badge_col:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if ws_status == STATUS_CONNECTED:
        badge = (f'<div class="badge badge-live">'
                 f'<span class="live-dot"></span>LIVE · {_sym_display}</div>')
    elif ws_status in (STATUS_CONNECTING, STATUS_RECONNECTING):
        badge = (f'<div class="badge badge-connecting">'
                 f'⟳ {ws_status.upper()} · {_sym_display}</div>')
    elif ws_status == STATUS_ERROR:
        badge = '<div class="badge badge-error">✖ CONNECTION ERROR</div>'
    else:
        badge = '<div class="badge badge-paused">⏸ PAUSED</div>'
    st.markdown(badge, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────
tq = state.get(K_TRADE_QUEUE)
viz.render_metric_cards(
    total_scanned   = state.get(K_TOTAL_SCANNED),
    total_whales    = state.get(K_TOTAL_WHALES),
    total_clusters  = state.get(K_TOTAL_CLUSTERS),
    total_volume    = state.get(K_TOTAL_VOLUME),
    queue_depth     = tq.qsize() if tq else 0,
    whale_threshold = whale_threshold,
)


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT — feed tabs (left) + sentiment panel (right)
# ─────────────────────────────────────────────────────────────
feed_col, sent_col = st.columns([3, 1])

with feed_col:
    (tab_unified, tab_whale, tab_cluster,
     tab_breakdown, tab_wallets, tab_dossier, tab_log) = st.tabs([
        "📊  Live Feed",
        "🐋  Single Whales",
        "🔍  Clusters",
        "🧩  Cluster Breakdown",
        "👤  Wallet Intelligence",
        "🗂️  Wallet Dossier",
        "📡  Activity Log",
    ])

    # ── Tab 1: Unified feed ───────────────────────────────────
    with tab_unified:
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem">'
            'All whale events — single trades & clusters — newest first</div>',
            unsafe_allow_html=True
        )
        viz.render_styled_feed(
            rows=state.get(K_UNIFIED_FEED),
            sentiment_filter=sentiment_filter,
            min_usd=0.0, max_rows=max_rows,
        )

    # ── Tab 2: Single-trade whales ────────────────────────────
    with tab_whale:
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem">'
            f'Individual trades ≥ {utils.format_usd(whale_threshold)} '
            f'on {selected_symbol}</div>',
            unsafe_allow_html=True
        )
        viz.render_styled_feed(
            rows=state.get(K_WHALE_FEED),
            sentiment_filter=sentiment_filter,
            min_usd=0.0, max_rows=max_rows,
        )

    # ── Tab 3: Cluster table ──────────────────────────────────
    with tab_cluster:
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem">'
            'Stealth whale clusters — iceberg & layering detection</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <div style='font-size:0.66rem;color:#2a6080;line-height:1.7;
             background:#020d16;border:1px solid #0a2030;
             border-left:3px solid #cc88ff;
             padding:0.55rem 0.9rem;border-radius:2px;margin-bottom:0.8rem;'>
            <b style='color:#cc88ff;'>ANTI-EVASION</b> — Rolling
            <b style='color:#c8f0ff'>{cluster_window:.1f}s</b> window.
            Sum ≥ <b style='color:#c8f0ff'>{utils.format_usd(cluster_threshold)}</b>
            triggers a cluster event. HIGH suspicion clusters fire a
            <b style='color:#cc88ff'>toast alert</b>.
        </div>
        """, unsafe_allow_html=True)
        viz.render_styled_feed(
            rows=state.get(K_CLUSTER_FEED),
            sentiment_filter=sentiment_filter,
            min_usd=0.0, max_rows=max_rows,
        )

    # ── Tab 4: Cluster breakdown ──────────────────────────────
    with tab_breakdown:
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem">'
            'Per-cluster drill-down — expand to inspect individual orders</div>',
            unsafe_allow_html=True
        )
        viz.render_cluster_breakdown(
            cluster_raw_list=state.get(K_CLUSTER_RAW),
            max_clusters=20,
        )

    # ── Tab 5: Wallet Intelligence leaderboard ────────────────
    with tab_wallets:
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem">'
            'Wallet Intelligence — all profiled wallets ranked by volume</div>',
            unsafe_allow_html=True
        )

        # Tag legend
        st.markdown("""
        <div style='font-size:0.63rem;color:#2a6080;line-height:1.8;
             background:#010a10;border:1px solid #0a2030;
             border-left:3px solid #2a6080;
             padding:0.5rem 0.9rem;border-radius:2px;margin-bottom:0.8rem;'>
            🧠 <b style='color:#00ff88'>Smart Money</b> — buys on bearish sentiment
            (contrarian, pre-shift accumulation) &nbsp;·&nbsp;
            ⚡ <b style='color:#cc88ff'>HF Whale</b> — high-frequency trader
            (&gt;{hf} trades/session) &nbsp;·&nbsp;
            🏛️ <b style='color:#ffaa00'>Known Entity</b> — exchange / market maker
            (address in static registry)
        </div>
        """.format(hf=20), unsafe_allow_html=True)

        # Sort control
        sort_by = st.selectbox(
            "Sort by", ["total_volume", "trade_count", "last_seen"],
            format_func=lambda x: {
                "total_volume": "💰 Total Volume",
                "trade_count":  "📈 Trade Count",
                "last_seen":    "🕐 Last Seen",
            }[x],
            key="wallet_sort",
            label_visibility="collapsed",
        )

        top_profiles = get_top_wallets(by=sort_by, limit=50)
        viz.render_wallet_leaderboard(profiles=top_profiles)

        # Quick-select for dossier
        if top_profiles:
            st.markdown(
                "<div style='font-size:0.6rem;color:#2a6080;margin-top:0.6rem;"
                "letter-spacing:0.15em;'>SELECT WALLET FOR DOSSIER</div>",
                unsafe_allow_html=True
            )
            wallet_names = {
                p.address: (p.label or p.address[:14] + "…")
                for p in top_profiles
            }
            selected_addr = st.selectbox(
                "Wallet", options=list(wallet_names.keys()),
                format_func=lambda a: wallet_names[a],
                key="wallet_select_leaderboard",
                label_visibility="collapsed",
            )
            if st.button("🗂️  Open Dossier", key="btn_open_dossier"):
                state.set(K_SELECTED_WALLET, selected_addr)
                st.info(f"Dossier loaded. Switch to the 🗂️ Wallet Dossier tab.")

    # ── Tab 6: Wallet Dossier ─────────────────────────────────
    with tab_dossier:
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem">'
            'Wallet Dossier — deep-dive for a single address</div>',
            unsafe_allow_html=True
        )

        selected_addr = state.get(K_SELECTED_WALLET)
        registry: dict = st.session_state.get(K_WALLET_REGISTRY, {})

        # Address picker at the top of the tab
        all_addrs = list(registry.keys())
        if all_addrs:
            def _fmt_addr(a):
                p = registry.get(a)
                if p and p.label:
                    return f"{p.label} ({a[:8]}…)"
                return a[:14] + "…" + a[-6:]

            # Default to whatever was set from the leaderboard tab
            default_idx = 0
            if selected_addr and selected_addr in all_addrs:
                default_idx = all_addrs.index(selected_addr)

            chosen = st.selectbox(
                "Select wallet address",
                options=all_addrs,
                index=default_idx,
                format_func=_fmt_addr,
                key="dossier_addr_picker",
            )
            state.set(K_SELECTED_WALLET, chosen)

            profile      = registry.get(chosen)
            trade_hist   = load_trade_history(chosen, limit=200)
            viz.render_wallet_dossier(profile=profile, trade_history=trade_hist)
        else:
            st.info(
                "No wallets profiled yet. Start tracking — every whale trade "
                "automatically creates and updates a wallet profile in the SQLite DB."
            )

    # ── Tab 7: Activity log ───────────────────────────────────
    with tab_log:
        st.markdown(
            '<div class="section-label" style="margin-top:0.5rem">'
            'Stream event log — real-time pipeline activity</div>',
            unsafe_allow_html=True
        )
        viz.render_activity_log(event_log=state.get(K_EVENT_LOG), height_px=460)


# ─────────────────────────────────────────────────────────────
# RIGHT COLUMN — Sentiment panel
# ─────────────────────────────────────────────────────────────
with sent_col:
    st.markdown(
        '<div class="section-label" style="margin-top:0.5rem">'
        '📰 Market Sentiment</div>',
        unsafe_allow_html=True
    )

    active_display = st.session_state.get("ctrl_symbol", "BTC/USDT")
    active_result  = sentiment.get_cached_sentiment(active_display)
    active_score   = active_result.score if active_result else None

    # ── SVG Gauge ─────────────────────────────────────────────
    viz.render_sentiment_gauge(score=active_score, symbol=active_display)

    # ── Backend badge ─────────────────────────────────────────
    src_label = active_result.backend_badge if active_result else "⏳ Loading…"
    age_main  = ""
    if active_result:
        _a = int(active_result.age_seconds)
        age_main = f"{_a}s ago" if _a < 60 else f"{_a//60}m ago"
    st.markdown(
        f"<div style='font-size:0.56rem;color:#1a4060;text-align:center;"
        f"margin-bottom:0.4rem;letter-spacing:0.08em;'>"
        f"{src_label}"
        f"{'  ·  ' + age_main if age_main else ''}</div>",
        unsafe_allow_html=True
    )

    # ── Refresh button ────────────────────────────────────────
    if st.button("🔄 Refresh Sentiment", key="btn_refresh_sentiment",
                 use_container_width=True):
        if hasattr(sentiment, "force_refresh"):
            sentiment.force_refresh()
        st.rerun()

    # ── Latest headline ───────────────────────────────────────
    if active_result and active_result.headline:
        st.markdown(
            f"<div style='margin:0.6rem 0;font-size:0.6rem;color:#1a4060;"
            f"line-height:1.6;padding:0.45rem 0.7rem;background:#010a10;"
            f"border:1px solid #0a2030;border-left:3px solid #2a6080;border-radius:2px;'>"
            f"<span style='color:#2a6080;letter-spacing:0.1em;font-size:0.55rem;'>LATEST HEADLINE</span><br>"
            f"<span style='color:#7ecfff;'>{active_result.headline[:120]}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Per-symbol compact cards ──────────────────────────────
    monitored = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    for sym in monitored:
        if sym == active_display:
            continue
        result = sentiment.get_cached_sentiment(sym)
        if result:
            colour = {"BULLISH": "#00ff88", "BEARISH": "#ff4466",
                      "NEUTRAL": "#ffaa00"}.get(result.label, "#2a6080")
            sign  = "+" if result.score >= 0 else ""
            age_s = int(result.age_seconds)
            age   = f"{age_s}s ago" if age_s < 60 else f"{age_s//60}m ago"
            bar_w = max(2, int(abs(result.score) * 44))
            st.markdown(
                f"<div class='sent-card'>"
                f"<div class='sent-symbol'>{sym}"
                f"<span style='float:right;font-size:0.5rem;color:#1a4060;'>{result.backend_badge}</span></div>"
                f"<div style='display:flex;align-items:center;gap:0.4rem;margin:0.2rem 0;'>"
                f"<div style='font-family:Orbitron,monospace;font-size:0.8rem;"
                f"color:{colour};font-weight:700;min-width:60px;'>"
                f"{result.label_emoji} {sign}{result.score:.2f}</div>"
                f"<div style='flex:1;height:4px;background:#0a2030;border-radius:2px;'>"
                f"<div style='width:{bar_w}px;height:4px;background:{colour};border-radius:2px;'></div>"
                f"</div></div>"
                f"<div style='font-size:0.56rem;color:#1a4060;'>{result.label} · {age}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='sent-card'><div class='sent-symbol'>{sym}</div>"
                f"<div style='font-size:0.66rem;color:#1a4060;'>⏳ Loading…</div></div>",
                unsafe_allow_html=True
            )

    # ── Top wallet mini card ──────────────────────────────────
    st.markdown(
        "<div style='margin:0.6rem 0;border-top:1px solid #0a2030'></div>",
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-label">👤 Top Wallet</div>',
                unsafe_allow_html=True)
    try:
        top = get_top_wallets(by="total_volume", limit=1)
        if top:
            p    = top[0]
            name = p.label if p.label else p.address
            st.markdown(
                f"<div style='background:#041822;border:1px solid #0a4060;"
                f"border-left:3px solid #ffaa00;padding:0.5rem 0.75rem;"
                f"border-radius:2px;font-size:0.62rem;'>"
                f"<div style='color:#ffaa00;font-family:Orbitron,monospace;"
                f"font-size:0.72rem;'>{p.tag_badge()}</div>"
                f"<div style='color:#c8f0ff;margin-top:0.2rem;"
                f"word-break:break-all;font-size:0.6rem;'>{name}</div>"
                f"<div style='color:#2a6080;margin-top:0.15rem;'>"
                f"Vol: ${p.total_volume:,.0f} · {p.dominant_side}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='font-size:0.62rem;color:#1a4060;'>No wallets yet.</div>",
                unsafe_allow_html=True
            )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# POLLING LOOP
# ─────────────────────────────────────────────────────────────
# Always poll on a timer so sentiment refreshes even when the
# WebSocket stream is paused. When streaming, use the user's
# refresh_interval. When paused, poll every 10 s (light).
# ─────────────────────────────────────────────────────────────

# Track last sentiment update time to detect when thread writes new scores
_store_snapshot = {
    sym: getattr(sentiment.get_cached_sentiment(sym), "scored_at", 0)
    for sym in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
}
st.session_state["_sent_snapshot"] = _store_snapshot

if state.get(K_RUNNING):
    # Stream active — fast polling
    time.sleep(refresh_interval)
    st.rerun()
else:
    # Stream paused — slow polling just for sentiment updates
    time.sleep(10)
    st.rerun()