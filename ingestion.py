"""
ingestion.py — Stream Source: Binance WebSocket Consumer  (Cloud-safe)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY BINANCE BLOCKS STREAMLIT CLOUD:
  Streamlit Community Cloud runs on AWS (us-east-1).
  Binance actively blocks WebSocket connections from AWS IP ranges
  to prevent automated trading bots. This means:
    wss://stream.binance.com:9443  →  Connection refused on Cloud

SOLUTION — Two WS URLs tried in order:
  1. wss://data-stream.binance.vision  (Binance market data endpoint —
     designed for read-only consumers, less aggressively blocked)
  2. wss://stream.binance.com:9443     (standard, blocked on AWS)
  3. wss://stream.binance.com:443      (port 443 fallback, sometimes works)
  4. Simulation mode                   (if ALL above fail — generates
     realistic fake trades so the UI is fully functional on Cloud)

SIMULATION MODE:
  Uses real BTC price ranges and realistic trade size distributions.
  Whale events still trigger correctly. The UI is 100% functional.
  A banner in the app indicates simulation mode is active.
"""

import json
import math
import os
import queue
import random
import threading
import time
from datetime import datetime, timezone

import websocket

import state
from config import (
    binance_ws_url,
    MAX_RECONNECT, RECONNECT_BASE_DELAY, RECONNECT_MAX_DELAY,
    STATUS_STOPPED, STATUS_CONNECTING, STATUS_CONNECTED,
    STATUS_RECONNECTING, STATUS_ERROR,
    K_RUNNING, K_WS_THREAD, K_TRADE_QUEUE, K_STATUS_QUEUE,
    K_WS_STATUS, K_ACTIVE_SYMBOL,
)

# Alternative Binance endpoints (tried in order before simulation fallback)
BINANCE_ENDPOINTS = [
    "wss://data-stream.binance.vision/ws/{slug}@trade",  # market data mirror
    "wss://stream.binance.com:9443/ws/{slug}@trade",     # standard
    "wss://stream.binance.com:443/ws/{slug}@trade",      # port 443
]

# Realistic price ranges per symbol for simulation mode
SYMBOL_PRICE_RANGES = {
    "btcusdt":  (55000,  80000),
    "ethusdt":  (2500,   4500),
    "solusdt":  (100,    250),
    "bnbusdt":  (350,    650),
    "xrpusdt":  (0.40,   1.20),
    "avaxusdt": (20,     60),
    "dogeusdt": (0.06,   0.25),
    "arbusdt":  (0.60,   2.00),
}


# ─────────────────────────────────────────────────────────────
# SCHEMA PARSER
# ─────────────────────────────────────────────────────────────

def parse_binance_trade(raw_msg: str) -> dict | None:
    try:
        data     = json.loads(raw_msg)
        price    = float(data["p"])
        quantity = float(data["q"])
        total    = round(price * quantity, 2)
        side     = "SELL" if data.get("m", False) else "BUY"
        ts_ms    = data.get("T", 0)
        ts_epoch = ts_ms / 1000.0
        dt       = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
        symbol   = data.get("s", "BTCUSDT")
        if "/" not in symbol and len(symbol) > 4:
            symbol = symbol[:-4] + "/" + symbol[-4:]
        return {
            "symbol":    symbol,
            "side":      side,
            "price":     price,
            "quantity":  quantity,
            "total_usd": total,
            "ts_epoch":  ts_epoch,
            "timestamp": dt.strftime("%H:%M:%S.%f")[:-3],
        }
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────
# SIMULATION GENERATOR  (used when all WS endpoints fail)
# ─────────────────────────────────────────────────────────────

def _simulate_trade(slug: str) -> dict:
    """
    Generate a realistic fake trade for the given symbol.
    Price walks randomly within the symbol's typical range.
    ~4% of trades are large enough to trigger whale detection.
    """
    low, high = SYMBOL_PRICE_RANGES.get(slug, (100, 1000))

    # Random walk: price stays near midpoint with realistic volatility
    mid   = (low + high) / 2
    sigma = (high - low) * 0.05
    price = round(max(low, min(high, random.gauss(mid, sigma))), 4)

    # Trade size: bimodal — mostly small retail, occasional whale
    if random.random() < 0.04:   # 4% whale trades
        qty = round(random.uniform(10, 100), 4)
    else:
        qty = round(random.uniform(0.001, 5), 4)

    total    = round(price * qty, 2)
    side     = random.choice(["BUY", "SELL"])
    now      = time.time()
    dt       = datetime.fromtimestamp(now, tz=timezone.utc)
    symbol   = slug.upper()
    if len(symbol) > 4:
        symbol = symbol[:-4] + "/" + symbol[-4:]

    return {
        "symbol":    symbol,
        "side":      side,
        "price":     price,
        "quantity":  qty,
        "total_usd": total,
        "ts_epoch":  now,
        "timestamp": dt.strftime("%H:%M:%S.%f")[:-3],
    }


# ─────────────────────────────────────────────────────────────
# SIMULATION THREAD
# ─────────────────────────────────────────────────────────────

class SimulationThread(threading.Thread):
    """
    Generates synthetic trades at ~5 trades/second.
    Used automatically when all Binance WebSocket endpoints fail.
    """
    def __init__(self, trade_queue: queue.Queue,
                 status_queue: queue.Queue, slug: str):
        super().__init__(daemon=True)
        self.trade_queue  = trade_queue
        self.status_queue = status_queue
        self.slug         = slug.lower()
        self._stop        = threading.Event()

    def stop(self):
        self._stop.set()

    def _push(self, status: str, msg: str = "") -> None:
        try:
            self.status_queue.put_nowait({"status": status, "msg": msg})
        except queue.Full:
            pass

    def run(self):
        self._push(
            STATUS_CONNECTED,
            "⚠️ SIMULATION MODE — Binance WS blocked on Cloud. "
            "Generating realistic synthetic trades. "
            "All whale detection & clustering fully functional."
        )
        while not self._stop.is_set():
            trade = _simulate_trade(self.slug)
            try:
                self.trade_queue.put_nowait(trade)
            except queue.Full:
                pass
            time.sleep(0.2)   # ~5 trades/second

        self._push(STATUS_STOPPED, "Simulation stopped.")


# ─────────────────────────────────────────────────────────────
# WEBSOCKET THREAD  (tries multiple endpoints then falls back)
# ─────────────────────────────────────────────────────────────

class BinanceStreamThread(threading.Thread):
    """
    Tries each Binance endpoint in order.
    Falls back to SimulationThread if all endpoints fail.
    """

    def __init__(self, trade_queue: queue.Queue,
                 status_queue: queue.Queue, slug: str = "btcusdt"):
        super().__init__(daemon=True)
        self.trade_queue  = trade_queue
        self.status_queue = status_queue
        self.slug         = slug.lower()
        self._stop        = threading.Event()
        self.attempt      = 0
        self._ws_app      = None
        self._connected   = False
        # Build endpoint list for this slug
        self.endpoints    = [
            url.format(slug=self.slug)
            for url in BINANCE_ENDPOINTS
        ]
        self._current_url = self.endpoints[0]

    def stop(self):
        self._stop.set()
        if self._ws_app:
            try:
                self._ws_app.close()
            except Exception:
                pass

    def _push(self, status: str, msg: str = "") -> None:
        try:
            self.status_queue.put_nowait({"status": status, "msg": msg})
        except queue.Full:
            pass

    def _on_open(self, ws):
        self.attempt    = 0
        self._connected = True
        self._push(STATUS_CONNECTED,
                   f"Connected → {self._current_url}")

    def _on_message(self, ws, message: str):
        if self._stop.is_set():
            ws.close()
            return
        trade = parse_binance_trade(message)
        if trade:
            try:
                self.trade_queue.put_nowait(trade)
            except queue.Full:
                pass

    def _on_error(self, ws, error):
        self._push(STATUS_RECONNECTING,
                   f"WS error: {type(error).__name__}: {error}")

    def _on_close(self, ws, code, msg):
        if not self._stop.is_set():
            self._push(STATUS_RECONNECTING,
                       f"Closed (code={code}). Reconnecting…")

    def _try_connect(self, url: str) -> bool:
        """
        Attempt a single WebSocket connection.
        Returns True if successfully connected (on_open fired),
        False if connection was refused immediately.
        """
        self._current_url = url
        self._connected   = False
        self._push(STATUS_CONNECTING, f"Trying {url}…")

        try:
            self._ws_app = websocket.WebSocketApp(
                url,
                on_open    = self._on_open,
                on_message = self._on_message,
                on_error   = self._on_error,
                on_close   = self._on_close,
            )
            self._ws_app.run_forever(
                ping_interval=20,
                ping_timeout=10,
            )
        except Exception as e:
            self._push(STATUS_RECONNECTING, f"Exception on {url}: {e}")

        return self._connected

    def run(self):
        self._push(STATUS_CONNECTING, "Starting WebSocket connection…")

        # Round-robin through all endpoints
        endpoint_idx    = 0
        total_attempts  = 0
        max_total       = MAX_RECONNECT * len(self.endpoints)

        while not self._stop.is_set():
            if total_attempts >= max_total:
                # All endpoints exhausted — fall back to simulation
                self._push(
                    STATUS_CONNECTED,
                    f"All {len(self.endpoints)} Binance endpoints failed "
                    f"(likely AWS IP block). Starting simulation mode…"
                )
                sim = SimulationThread(
                    self.trade_queue, self.status_queue, self.slug
                )
                sim.start()
                # Wait for stop signal, then stop simulation too
                while not self._stop.is_set():
                    time.sleep(0.5)
                sim.stop()
                return

            url = self.endpoints[endpoint_idx % len(self.endpoints)]
            connected = self._try_connect(url)

            if self._stop.is_set():
                break

            total_attempts  += 1
            endpoint_idx    += 1

            # If it connected but then dropped, prefer the same URL
            if connected:
                endpoint_idx -= 1   # retry the same endpoint first

            delay = min(
                RECONNECT_BASE_DELAY * (2 ** (total_attempts - 1)),
                RECONNECT_MAX_DELAY
            )
            self._push(
                STATUS_RECONNECTING,
                f"Attempt {total_attempts}/{max_total} — "
                f"next retry in {delay:.0f}s…"
            )
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                if self._stop.is_set():
                    break
                time.sleep(0.1)

        self._push(STATUS_STOPPED, "Stream stopped.")


# ─────────────────────────────────────────────────────────────
# LIFECYCLE HELPERS
# ─────────────────────────────────────────────────────────────

def start_stream(slug: str = "btcusdt") -> None:
    if state.get(K_RUNNING):
        return
    tq = queue.Queue(maxsize=2000)
    sq = queue.Queue(maxsize=100)
    t  = BinanceStreamThread(trade_queue=tq, status_queue=sq, slug=slug)
    t.start()
    state.set(K_TRADE_QUEUE,   tq)
    state.set(K_STATUS_QUEUE,  sq)
    state.set(K_WS_THREAD,     t)
    state.set(K_RUNNING,       True)
    state.set(K_ACTIVE_SYMBOL, slug)
    state.set(K_WS_STATUS,     STATUS_CONNECTING)


def stop_stream() -> None:
    t = state.get(K_WS_THREAD)
    if t and t.is_alive():
        t.stop()
    state.set(K_RUNNING,      False)
    state.set(K_WS_STATUS,    STATUS_STOPPED)
    state.set(K_WS_THREAD,    None)
    state.set(K_TRADE_QUEUE,  None)
    state.set(K_STATUS_QUEUE, None)


def switch_symbol(new_slug: str) -> None:
    stop_stream()
    import state as _state
    _state.reset_stats()
    start_stream(slug=new_slug)


def drain_status_queue() -> None:
    sq = state.get(K_STATUS_QUEUE)
    if not sq:
        return
    while True:
        try:
            item = sq.get_nowait()
            state.set(K_WS_STATUS, item["status"])
            from config import K_WS_MSG
            state.set(K_WS_MSG, item["msg"])
            ts = datetime.utcnow().strftime("%H:%M:%S")
            state.append_log(f"📡 [{ts}] {item['msg']}")
        except queue.Empty:
            break
