"""
state.py — Streamlit session-state lifecycle manager  (Phase 5)
"""
import streamlit as st
from config import SESSION_DEFAULTS


def init():
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get(key: str):
    if key not in SESSION_DEFAULTS:
        raise KeyError(f"Unknown session key '{key}'. Register it in config.py first.")
    return st.session_state.get(key, SESSION_DEFAULTS[key])


def set(key: str, value) -> None:
    if key not in SESSION_DEFAULTS:
        raise KeyError(f"Unknown session key '{key}'. Register it in config.py first.")
    st.session_state[key] = value


def reset_stats() -> None:
    from config import (
        K_WHALE_FEED, K_CLUSTER_FEED, K_UNIFIED_FEED, K_CLUSTER_RAW,
        K_TOTAL_SCANNED, K_TOTAL_WHALES, K_TOTAL_CLUSTERS, K_TOTAL_VOLUME,
        K_EVENT_LOG, K_CLUSTER_BUFFER, K_TOAST_QUEUE, K_LAST_ALERT_TIME,
    )
    st.session_state[K_WHALE_FEED]      = []
    st.session_state[K_CLUSTER_FEED]    = []
    st.session_state[K_UNIFIED_FEED]    = []
    st.session_state[K_CLUSTER_RAW]     = []
    st.session_state[K_TOTAL_SCANNED]   = 0
    st.session_state[K_TOTAL_WHALES]    = 0
    st.session_state[K_TOTAL_CLUSTERS]  = 0
    st.session_state[K_TOTAL_VOLUME]    = 0.0
    st.session_state[K_EVENT_LOG]       = []
    st.session_state[K_CLUSTER_BUFFER]  = {}
    st.session_state[K_TOAST_QUEUE]     = []
    st.session_state[K_LAST_ALERT_TIME] = 0.0
    # Note: wallet registry is NOT reset — profiles persist across clears


def append_log(line: str, max_lines: int = 500) -> None:
    from config import K_EVENT_LOG
    log: list = st.session_state[K_EVENT_LOG]
    log.append(line)
    if len(log) > max_lines:
        st.session_state[K_EVENT_LOG] = log[-300:]


def push_toast(msg: str, icon: str = "🐋") -> None:
    from config import K_TOAST_QUEUE
    q: list = st.session_state[K_TOAST_QUEUE]
    q.append({"msg": msg, "icon": icon})
    st.session_state[K_TOAST_QUEUE] = q[-5:]
