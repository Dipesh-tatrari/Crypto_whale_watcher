"""
modules/visualization.py — Phase 4 Visualization Components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All visual rendering functions for the dashboard.  Each function
is self-contained: takes data in, renders Streamlit/HTML out.

Components:
  render_sentiment_gauge()    — SVG arc gauge for market mood
  render_styled_feed()        — colour-coded Pandas-styled dataframe
  render_cluster_breakdown()  — expandable per-cluster trade drill-down
  render_metric_cards()       — top KPI strip
  render_activity_log()       — live scrolling terminal log
  render_connection_status()  — badge + diagnostics
"""

from __future__ import annotations
import math
import pandas as pd
import streamlit as st
from typing import Any


# ─────────────────────────────────────────────────────────────
# 1. SENTIMENT GAUGE
# ─────────────────────────────────────────────────────────────

def render_sentiment_gauge(score: float | None, symbol: str = "") -> None:
    """
    Render an SVG arc gauge showing the current sentiment score.
    All text elements are fully inside the viewBox — no overflow leakage.
    """
    cx, cy, r = 120, 95, 78
    stroke_w  = 16

    def _arc_point(angle_deg: float) -> tuple[float, float]:
        rad = math.radians(180 - angle_deg)
        return cx + r * math.cos(rad), cy - r * math.sin(rad)

    def _arc_path(start_deg: float, end_deg: float) -> str:
        x1, y1 = _arc_point(start_deg)
        x2, y2 = _arc_point(end_deg)
        large  = 1 if (end_deg - start_deg) > 180 else 0
        return f"M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}"

    bg_path    = _arc_path(0, 180)
    zone_paths = [
        (_arc_path(0,   60),  "#ff4466", "0.75"),
        (_arc_path(60, 120),  "#ffaa00", "0.75"),
        (_arc_path(120, 180), "#00ff88", "0.75"),
    ]

    if score is None:
        needle_deg = 90
        score_text = "—"
        label_col  = "#2a6080"
        mood_text  = "NO DATA"
        opacity    = "0.3"
    else:
        needle_deg = (score + 1.0) / 2.0 * 180.0
        score_text = f"{score:+.2f}"
        label_col  = "#00ff88" if score > 0.2 else "#ff4466" if score < -0.2 else "#ffaa00"
        mood_text  = "BULLISH"  if score > 0.2 else "BEARISH"  if score < -0.2 else "NEUTRAL"
        opacity    = "1.0"

    nx, ny = _arc_point(needle_deg)

    ticks_svg = ""
    for deg in [0, 45, 90, 135, 180]:
        ox, oy = _arc_point(deg)
        ix = cx + (r - 11) * math.cos(math.radians(180 - deg))
        iy = cy - (r - 11) * math.sin(math.radians(180 - deg))
        ticks_svg += (
            f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{ix:.1f}" y2="{iy:.1f}" '
            f'stroke="#0a4060" stroke-width="1.5"/>'
        )

    zone_svg = "".join(
        f'<path d="{p}" fill="none" stroke="{c}" stroke-width="{stroke_w}" '
        f'stroke-linecap="round" opacity="{o}"/>'
        for p, c, o in zone_paths
    )

    # viewBox height = 210 — all text comfortably inside
    svg = (
        f'<div style="background:#020d16;border:1px solid #0a4060;'
        f'border-radius:4px;padding:0.5rem 0.3rem 0.3rem;margin-bottom:0.3rem;">'
        f'<svg viewBox="0 0 240 190" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;display:block;">'

        # Background track
        f'<path d="{bg_path}" fill="none" stroke="#041822" '
        f'stroke-width="{stroke_w + 4}" stroke-linecap="round"/>'

        # Coloured zones
        f'{zone_svg}'

        # Tick marks
        f'{ticks_svg}'

        # Needle
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" '
        f'stroke="#c8f0ff" stroke-width="2.5" stroke-linecap="round" opacity="{opacity}"/>'

        # Hub
        f'<circle cx="{cx}" cy="{cy}" r="5" fill="#c8f0ff" opacity="{opacity}"/>'

        # BEAR / BULL axis labels
        f'<text x="18" y="112" font-family="monospace" font-size="9" '
        f'fill="#ff4466" font-weight="bold">BEAR</text>'
        f'<text x="196" y="112" font-family="monospace" font-size="9" '
        f'fill="#00ff88" font-weight="bold">BULL</text>'

        # Score value — large, centred
        f'<text x="{cx}" y="138" text-anchor="middle" '
        f'font-family="monospace" font-size="26" font-weight="700" '
        f'fill="{label_col}" opacity="{opacity}">{score_text}</text>'

        # Mood label
        f'<text x="{cx}" y="158" text-anchor="middle" '
        f'font-family="monospace" font-size="11" letter-spacing="3" '
        f'fill="{label_col}" opacity="{opacity}">{mood_text}</text>'

        # Symbol
        f'<text x="{cx}" y="178" text-anchor="middle" '
        f'font-family="monospace" font-size="9" letter-spacing="2" '
        f'fill="#2a6080">{symbol}</text>'

        f'</svg></div>'
    )
    st.markdown(svg, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 2. STYLED FEED DATAFRAME
# ─────────────────────────────────────────────────────────────

# Columns shown in the styled table (hide internal _ prefixed cols)
_DISPLAY_COLS = [
    "Type", "⏱ Time", "Symbol", "Side",
    "Price (USDT)", "Quantity", "💰 Total USD", "Sentiment",
]

def _row_style(row: pd.Series) -> list[str]:
    """
    Pandas Styler row-level function.
    Colour logic:
      SINGLE BUY  → deep green tint
      SINGLE SELL → deep red tint
      CLUSTER     → purple tint (intensity by suspicion)
    """
    detection = row.get("_detection", "SINGLE")
    side      = row.get("_side", "")

    if detection == "CLUSTER":
        bg = "background-color: #1a0a2e; color: #cc88ff;"
    elif side == "BUY":
        bg = "background-color: #001a0d; color: #c8f0ff;"
    else:
        bg = "background-color: #1a0006; color: #c8f0ff;"

    return [bg] * len(row)


def render_styled_feed(
    rows: list[dict],
    sentiment_filter: float = 0.0,
    min_usd: float = 0.0,
    max_rows: int = 50,
) -> None:
    """
    Render a colour-coded Pandas Styler dataframe.

    Args:
        rows:             list of row dicts (from unified_feed / whale_feed)
        sentiment_filter: only show rows where abs(sentiment) >= this value
        min_usd:          only show rows where _total_usd >= this value
        max_rows:         truncate display to this many rows
    """
    if not rows:
        st.info("No events yet. Start tracking to see whale activity.")
        return

    df = pd.DataFrame(rows)

    # Apply sentiment + USD filters on internal cols if present
    if "_sentiment" in df.columns and sentiment_filter > 0:
        df = df[df["_sentiment"].abs() >= sentiment_filter]
    if "_total_usd" in df.columns and min_usd > 0:
        df = df[df["_total_usd"] >= min_usd]

    if df.empty:
        st.info("No events match the current filters. Try relaxing the thresholds.")
        return

    df = df.head(max_rows)

    # Subset to display columns only (drop internal _ cols from view)
    display_cols = [c for c in _DISPLAY_COLS if c in df.columns]
    internal_cols = [c for c in df.columns if c.startswith("_")]

    try:
        styled = (
            df[display_cols + internal_cols]
            .style
            .apply(_row_style, axis=1)
            .set_properties(**{
                "font-family": "Share Tech Mono, monospace",
                "font-size":   "0.75rem",
                "border":      "1px solid #0a2030",
            })
            .hide(subset=internal_cols, axis="columns")
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        # Fallback if styling fails (older pandas versions)
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# 3. CLUSTER BREAKDOWN VIEW
# ─────────────────────────────────────────────────────────────

def render_cluster_breakdown(cluster_raw_list: list[dict], max_clusters: int = 10) -> None:
    """
    Render a drill-down breakdown for each detected cluster event.

    Each cluster gets an st.expander showing:
      - Summary: symbol, total USD, trade count, time window, suspicion
      - A mini-table of all individual trades that formed the cluster
      - A simple bar chart of per-trade USD values (visual pattern detection)

    Args:
        cluster_raw_list: list of raw cluster dicts from K_CLUSTER_RAW
        max_clusters:     show at most this many recent clusters
    """
    if not cluster_raw_list:
        st.info("No cluster events detected yet. Clusters appear when many small trades "
                "sum above the cluster threshold within the time window.")
        return

    st.markdown("""
    <div style='font-size:0.65rem; color:#2a6080; line-height:1.7;
         background:#010a10; border:1px solid #0a2030; border-left:3px solid #cc88ff;
         padding:0.5rem 0.9rem; border-radius:2px; margin-bottom:0.8rem;'>
        <b style='color:#cc88ff;'>STEALTH WHALE DETECTION</b> — Each card below shows a
        cluster of small trades that collectively exceeded the cluster threshold.
        High trade count + small avg size = 🔴 iceberg / layering pattern.
    </div>
    """, unsafe_allow_html=True)

    for i, cluster in enumerate(cluster_raw_list[:max_clusters]):
        sym          = cluster.get("symbol", "—")
        total        = cluster.get("cluster_usd", 0)
        n_trades     = cluster.get("trade_count", 0)
        window       = cluster.get("window_sec", 0)
        suspicion    = cluster.get("suspicion", "—")
        first_ts     = cluster.get("first_ts", "—")
        last_ts      = cluster.get("last_ts", "—")
        sentiment    = cluster.get("sentiment")
        trades       = cluster.get("trades", [])
        avg_size     = total / max(n_trades, 1)

        susp_colour = {"🔴 HIGH": "#ff4466", "🟡 MEDIUM": "#ffaa00", "🟢 LOW": "#00ff88"}.get(suspicion, "#2a6080")
        sent_str    = f"{sentiment:+.2f}" if sentiment is not None else "—"

        expander_label = (
            f"{'🔴' if 'HIGH' in suspicion else '🟡' if 'MEDIUM' in suspicion else '🟢'}"
            f"  {sym}  ·  ${total:,.0f}  ·  {n_trades} trades  ·  {window:.1f}s  ·  {suspicion}"
        )

        with st.expander(expander_label, expanded=(i == 0)):
            # ── Summary cards ─────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(_mini_card("Total USD",  f"${total:,.0f}",  "#00e5ff"), unsafe_allow_html=True)
            with c2:
                st.markdown(_mini_card("# Trades",   str(n_trades),    "#cc88ff"), unsafe_allow_html=True)
            with c3:
                st.markdown(_mini_card("Window",     f"{window:.2f}s", "#ffaa00"), unsafe_allow_html=True)
            with c4:
                st.markdown(_mini_card("Avg Size",   f"${avg_size:,.0f}", "#7ecfff"), unsafe_allow_html=True)
            with c5:
                st.markdown(_mini_card("Suspicion",  suspicion,        susp_colour), unsafe_allow_html=True)

            st.markdown(
                f"<div style='font-size:0.6rem; color:#1a4060; margin:0.4rem 0 0.6rem;'>"
                f"First: {first_ts} · Last: {last_ts} · Sentiment: {sent_str}</div>",
                unsafe_allow_html=True
            )

            # ── Trade-by-trade table ───────────────────────────
            if trades:
                trade_rows = []
                for t in trades:
                    trade_rows.append({
                        "Time":      t.get("timestamp", "—"),
                        "Side":      "🟢 BUY" if t.get("side") == "BUY" else "🔴 SELL",
                        "Price":     f"${t.get('price', 0):,.2f}",
                        "Qty":       f"{t.get('quantity', 0):,.6f}",
                        "USD Value": f"${t.get('total_usd', 0):,.2f}",
                    })
                df_trades = pd.DataFrame(trade_rows)
                st.dataframe(df_trades, use_container_width=True,
                             hide_index=True, height=180)

                # ── Bar chart: per-trade USD size ──────────────
                usd_values = [t.get("total_usd", 0) for t in trades]
                if len(usd_values) > 1:
                    df_chart = pd.DataFrame({
                        "Trade #": [f"#{j+1}" for j in range(len(usd_values))],
                        "USD":     usd_values,
                    }).set_index("Trade #")
                    st.markdown(
                        "<div style='font-size:0.6rem; color:#2a6080; "
                        "letter-spacing:0.15em; text-transform:uppercase; "
                        "margin-bottom:0.2rem;'>Trade Size Distribution</div>",
                        unsafe_allow_html=True
                    )
                    st.bar_chart(df_chart, height=120, use_container_width=True)


def _mini_card(label: str, value: str, colour: str) -> str:
    return (
        f"<div style='background:#041822; border:1px solid #0a4060; "
        f"border-left:3px solid {colour}; padding:0.5rem 0.7rem; border-radius:2px;'>"
        f"<div style='font-size:0.55rem; color:#2a6080; letter-spacing:0.15em; "
        f"text-transform:uppercase;'>{label}</div>"
        f"<div style='font-family:Orbitron,monospace; font-size:0.9rem; "
        f"color:{colour}; font-weight:700;'>{value}</div>"
        f"</div>"
    )


# ─────────────────────────────────────────────────────────────
# 4. METRIC CARDS STRIP
# ─────────────────────────────────────────────────────────────

def render_metric_cards(
    total_scanned:  int,
    total_whales:   int,
    total_clusters: int,
    total_volume:   float,
    queue_depth:    int,
    whale_threshold: float,
) -> None:
    hit_rate   = (total_whales + total_clusters) / max(total_scanned, 1) * 100
    qdepth_cls = "red" if queue_depth > 500 else ""

    st.markdown(f"""
    <div style="display:flex; gap:0.65rem; margin-bottom:1.1rem; flex-wrap:wrap;">
      {_kpi("Trades Scanned", f"{total_scanned:,}", "#00e5ff")}
      {_kpi("Single Whales",  f"{total_whales:,}",  "#00ff88")}
      {_kpi("Clusters",       f"{total_clusters:,}","#cc88ff")}
      {_kpi("Whale Volume",   f"${total_volume:,.0f}", "#ffaa00")}
      {_kpi("Detection Rate", f"{hit_rate:.3f}%",   "#00e5ff")}
      {_kpi("Queue Depth",    f"{queue_depth:,}",   "#ff4466" if queue_depth>500 else "#2a6080")}
      {_kpi("Threshold",      f"${whale_threshold:,.0f}", "#7ecfff")}
    </div>
    """, unsafe_allow_html=True)


def _kpi(label: str, value: str, colour: str) -> str:
    return (
        f"<div style='background:#041822; border:1px solid #0a4060; "
        f"border-left:3px solid {colour}; padding:0.6rem 0.85rem; "
        f"flex:1; min-width:95px; border-radius:2px;'>"
        f"<div style='font-size:0.56rem; color:#2a6080; letter-spacing:0.18em; "
        f"text-transform:uppercase;'>{label}</div>"
        f"<div style='font-family:Orbitron,monospace; font-size:1.05rem; "
        f"color:{colour}; font-weight:700;'>{value}</div></div>"
    )


# ─────────────────────────────────────────────────────────────
# 5. ACTIVITY LOG
# ─────────────────────────────────────────────────────────────

_LOG_CSS = {
    "log-whale": "#00ff88",
    "log-alert": "#ffaa00",
    "log-error": "#ff4466",
    "log-scan":  "#1a4060",
}

def render_activity_log(event_log: list[str], height_px: int = 400) -> None:
    """Render a scrollable terminal-style activity log."""
    import utils as _utils
    lines = event_log[-80:]
    html_lines = []
    for line in reversed(lines):
        css   = _utils.log_css_class(line)
        color = _LOG_CSS.get(css, "#1a4060")
        html_lines.append(
            f'<div style="color:{color}; line-height:1.65;">{line}</div>'
        )
    st.markdown(
        f'<div style="background:#010a10; border:1px solid #0a2030; '
        f'border-left:3px solid #003050; padding:0.6rem 1rem; '
        f'height:{height_px}px; overflow-y:auto; font-size:0.68rem; '
        f'font-family:Share Tech Mono,monospace; border-radius:2px;">'
        f'{"".join(html_lines)}</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# 6. CONNECTION STATUS
# ─────────────────────────────────────────────────────────────

_STATUS_COLOURS = {
    "stopped":     "#2a6080",
    "connecting":  "#ffcc00",
    "connected":   "#00ff88",
    "reconnecting":"#ffaa00",
    "error":       "#ff4466",
}

def render_connection_status(ws_status: str, ws_msg: str) -> None:
    colour = _STATUS_COLOURS.get(ws_status, "#2a6080")
    st.markdown(f"""
    <div style='font-size:0.7rem; color:{colour}; letter-spacing:0.08em;'>
        ● {ws_status.upper()}
    </div>
    <div style='font-size:0.58rem; color:#1a3a50; margin-top:0.3rem;
         word-break:break-word; line-height:1.5;'>
        {ws_msg[:140]}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 7. WALLET LEADERBOARD
# ─────────────────────────────────────────────────────────────

def render_wallet_leaderboard(profiles: list, on_select=None) -> None:
    """
    Render a ranked table of the most active wallets seen this session.

    Args:
        profiles:  list of WalletProfile objects (sorted by volume)
        on_select: optional callback(address) when user clicks a wallet
    """
    import pandas as pd
    from modules.profiler import CATEGORY_COLOURS, TAG_SMART_MONEY, TAG_HF_WHALE, TAG_KNOWN

    if not profiles:
        st.info("No wallets profiled yet. Start tracking to begin building the registry.")
        return

    rows = []
    for p in profiles[:50]:
        tag_emoji = {
            TAG_SMART_MONEY: "🧠",
            TAG_HF_WHALE:    "⚡",
            TAG_KNOWN:       "🏛️",
        }.get(p.tag, "❓")

        name = p.label if p.label else p.address[:10] + "…"
        rows.append({
            "Tag":         tag_emoji,
            "Wallet":      name,
            "Category":    p.category,
            "Trades":      p.trade_count,
            "Volume (USD)":f"${p.total_volume:,.0f}",
            "Avg Size":    f"${p.avg_trade_size:,.0f}",
            "Buy %":       f"{p.buy_ratio*100:.0f}%",
            "Behaviour":   p.dominant_side,
            "SM Score":    f"{p.smart_money_score:.2f}" if p.pre_shift_total >= 3 else "—",
            "_address":    p.address,
        })

    df = pd.DataFrame(rows)

    def _style_row(row):
        tag = row.get("Tag", "")
        if "🧠" in tag:
            return ["background-color:#0d1a0d; color:#00ff88;"] * len(row)
        if "⚡" in tag:
            return ["background-color:#1a0d1a; color:#cc88ff;"] * len(row)
        if "🏛️" in tag:
            return ["background-color:#0d1020; color:#ffaa00;"] * len(row)
        return [""] * len(row)

    display_cols = [c for c in df.columns if not c.startswith("_")]
    internal_cols = [c for c in df.columns if c.startswith("_")]
    try:
        styled = (
            df[display_cols + internal_cols]
            .style
            .apply(_style_row, axis=1)
            .set_properties(**{"font-family": "Share Tech Mono, monospace", "font-size": "0.72rem"})
            .hide(subset=internal_cols, axis="columns")
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=380)
    except Exception:
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True, height=380)


# ─────────────────────────────────────────────────────────────
# 8. WALLET DOSSIER
# ─────────────────────────────────────────────────────────────

def render_wallet_dossier(profile, trade_history: list[dict]) -> None:
    """
    Full dossier view for a single wallet — called inside an st.expander
    or on the Wallet Intelligence tab.

    Args:
        profile:       WalletProfile dataclass
        trade_history: list of trade dicts from trade_history table
    """
    import pandas as pd
    from modules.profiler import (
        sentiment_correlation, volume_by_hour, CATEGORY_COLOURS,
    )

    if profile is None:
        st.info("Select a wallet from the leaderboard to view its dossier.")
        return

    # ── Identity header ────────────────────────────────────────
    cat_colour = CATEGORY_COLOURS.get(profile.category, "#2a6080")
    name       = profile.label or (profile.address[:12] + "…" + profile.address[-6:])

    st.markdown(f"""
    <div style='background:#041822; border:1px solid #0a4060;
         border-left:4px solid {cat_colour}; padding:1rem 1.2rem;
         border-radius:3px; margin-bottom:1rem;'>
        <div style='font-family:Orbitron,monospace; font-size:1.1rem;
             font-weight:700; color:{cat_colour}; letter-spacing:0.08em;'>
            {profile.tag_badge()} &nbsp; {name}
        </div>
        <div style='font-size:0.65rem; color:#2a6080; margin-top:0.3rem;
             font-family:Share Tech Mono,monospace; letter-spacing:0.1em;'>
            {profile.address} &nbsp;·&nbsp; {profile.category}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "Total Volume",    f"${profile.total_volume:,.0f}",  "#ffaa00"),
        (k2, "Trade Count",     str(profile.trade_count),         "#00e5ff"),
        (k3, "Avg Trade Size",  f"${profile.avg_trade_size:,.0f}","#7ecfff"),
        (k4, "Buy Ratio",       f"{profile.buy_ratio*100:.0f}%",  "#00ff88"),
        (k5, "SM Score",        f"{profile.smart_money_score:.2f}","#cc88ff"),
        (k6, "Behaviour",       profile.dominant_side,            "#ffaa00"),
    ]
    for col, label, value, colour in kpis:
        with col:
            st.markdown(
                f"<div style='background:#041822;border:1px solid #0a4060;"
                f"border-left:3px solid {colour};padding:0.55rem 0.8rem;border-radius:2px;'>"
                f"<div style='font-size:0.55rem;color:#2a6080;letter-spacing:0.15em;"
                f"text-transform:uppercase;'>{label}</div>"
                f"<div style='font-family:Orbitron,monospace;font-size:0.95rem;"
                f"color:{colour};font-weight:700;'>{value}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Favourite assets ───────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            "<div style='font-size:0.6rem;color:#2a6080;letter-spacing:0.2em;"
            "text-transform:uppercase;border-bottom:1px solid #0a2030;"
            "padding-bottom:0.25rem;margin-bottom:0.6rem;'>Favourite Assets</div>",
            unsafe_allow_html=True
        )
        if profile.symbols:
            for sym in profile.symbols:
                st.markdown(
                    f"<div style='font-size:0.78rem;color:#c8f0ff;"
                    f"padding:0.2rem 0;'>▸ {sym}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<div style='font-size:0.7rem;color:#1a4060;'>No data yet</div>",
                        unsafe_allow_html=True)

    with right:
        st.markdown(
            "<div style='font-size:0.6rem;color:#2a6080;letter-spacing:0.2em;"
            "text-transform:uppercase;border-bottom:1px solid #0a2030;"
            "padding-bottom:0.25rem;margin-bottom:0.6rem;'>Activity Window</div>",
            unsafe_allow_html=True
        )
        import utils as _utils
        from datetime import datetime, timezone
        first = datetime.fromtimestamp(profile.first_seen, tz=timezone.utc).strftime("%H:%M %d %b")
        last  = datetime.fromtimestamp(profile.last_seen,  tz=timezone.utc).strftime("%H:%M %d %b")
        st.markdown(
            f"<div style='font-size:0.7rem;color:#c8f0ff;line-height:1.9;'>"
            f"First seen: <b>{first}</b><br>"
            f"Last seen:  <b>{last}</b><br>"
            f"Buys: <b style='color:#00ff88'>{profile.buy_count}</b> &nbsp; "
            f"Sells: <b style='color:#ff4466'>{profile.sell_count}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Sentiment correlation ──────────────────────────────────
    if trade_history:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.6rem;color:#2a6080;letter-spacing:0.2em;"
            "text-transform:uppercase;border-bottom:1px solid #0a2030;"
            "padding-bottom:0.25rem;margin-bottom:0.6rem;'>Sentiment Correlation</div>",
            unsafe_allow_html=True
        )
        corr = sentiment_correlation(trade_history)
        buy_s  = f"{corr['buy_avg_sentiment']:+.2f}" if corr["buy_avg_sentiment"]  is not None else "—"
        sell_s = f"{corr['sell_avg_sentiment']:+.2f}" if corr["sell_avg_sentiment"] is not None else "—"
        cs     = f"{corr['contrarian_score']:.0%}"

        st.markdown(f"""
        <div style='background:#010a10;border:1px solid #0a2030;
             border-left:3px solid #cc88ff;padding:0.7rem 1rem;border-radius:2px;'>
            <div style='font-size:0.68rem;color:#c8f0ff;line-height:1.9;'>
                Avg sentiment when <b style='color:#00ff88'>BUY</b>: <b>{buy_s}</b>
                &nbsp;·&nbsp;
                Avg sentiment when <b style='color:#ff4466'>SELL</b>: <b>{sell_s}</b>
                &nbsp;·&nbsp;
                Contrarian score: <b style='color:#cc88ff'>{cs}</b>
            </div>
            <div style='font-size:0.65rem;color:#7ecfff;margin-top:0.4rem;font-style:italic;'>
                {corr["narrative"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Hourly activity heatmap ────────────────────────────
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.6rem;color:#2a6080;letter-spacing:0.2em;"
            "text-transform:uppercase;border-bottom:1px solid #0a2030;"
            "padding-bottom:0.25rem;margin-bottom:0.6rem;'>Hourly Activity (UTC)</div>",
            unsafe_allow_html=True
        )
        by_hour = volume_by_hour(trade_history)
        df_hour = pd.DataFrame({
            "Hour (UTC)": [f"{h:02d}:00" for h in range(24)],
            "Volume USD": [by_hour.get(h, 0.0) for h in range(24)],
        }).set_index("Hour (UTC)")
        st.bar_chart(df_hour, height=130, use_container_width=True)

        # ── Recent trade table ─────────────────────────────────
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.6rem;color:#2a6080;letter-spacing:0.2em;"
            "text-transform:uppercase;border-bottom:1px solid #0a2030;"
            "padding-bottom:0.25rem;margin-bottom:0.6rem;'>Recent Trades</div>",
            unsafe_allow_html=True
        )
        df_trades = pd.DataFrame([{
            "Time":      t.get("ts_display", "—"),
            "Symbol":    t.get("symbol", "—"),
            "Side":      "🟢 BUY" if t.get("side") == "BUY" else "🔴 SELL",
            "Price":     f"${t.get('price', 0):,.2f}",
            "Qty":       f"{t.get('quantity', 0):,.6f}",
            "USD":       f"${t.get('total_usd', 0):,.0f}",
            "Sentiment": f"{t.get('sentiment'):+.2f}" if t.get("sentiment") is not None else "—",
        } for t in trade_history[:30]])
        st.dataframe(df_trades, use_container_width=True, hide_index=True, height=240)

    else:
        st.info("No trade history in the database for this wallet yet.")


# ─────────────────────────────────────────────────────────────
# 9. PROFILER DB STATS CARD
# ─────────────────────────────────────────────────────────────

def render_db_stats(stats: dict) -> None:
    """Compact card showing SQLite database health."""
    st.markdown(f"""
    <div style='background:#020d16;border:1px solid #0a2030;
         border-left:3px solid #2a6080;padding:0.6rem 1rem;
         border-radius:2px;font-size:0.65rem;color:#2a6080;line-height:1.9;
         font-family:Share Tech Mono,monospace;'>
        <b style='color:#7ecfff;letter-spacing:0.1em;'>WALLET DB</b><br>
        Profiles: <b style='color:#00e5ff'>{stats.get("wallet_count",0):,}</b>
        &nbsp;·&nbsp;
        Trades stored: <b style='color:#00e5ff'>{stats.get("trade_count",0):,}</b><br>
        🧠 Smart Money: <b style='color:#00ff88'>{stats.get("smart_money",0)}</b>
        &nbsp;·&nbsp;
        ⚡ HF Whales: <b style='color:#cc88ff'>{stats.get("hf_whales",0)}</b>
        &nbsp;·&nbsp;
        🏛️ Known: <b style='color:#ffaa00'>{stats.get("known",0)}</b>
    </div>
    """, unsafe_allow_html=True)