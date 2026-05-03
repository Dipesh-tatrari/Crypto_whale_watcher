"""
ingestion.py — Stream Source: Binance WebSocket Consumer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KAPPA ARCHITECTURE ROLE: Source Operator
─────────────────────────────────────────
In a production Kappa pipeline this module is replaced by a
Kafka consumer subscribed to a `raw-trades` topic. The WebSocket
thread here IS that consumer for the prototype — it runs in a
background daemon thread and pushes normalised trade dicts into
a thread-safe queue.Queue that the Streamlit main loop drains.

THREAD SAFETY:
  • BinanceStreamThread is a daemon thread (auto-killed on exit)
  • Communication to Streamlit happens only via queue.Queue
  • No direct st.session_state writes from the thread (not safe)
  • Status updates flow through a separate status_queue

RECONNECT STRATEGY: Exponential backoff
  delay = min(BASE * 2^attempt, MAX_DELAY)
  → 2s, 4s, 8s, 16s, 32s, 60s, 60s …
"""

import json
import queue
import threading
import time
from datetime import datetime, timezone

import websocket  # pip install websocket-client

import state
from config import (
    binance_ws_url,
    MAX_RECONNECT, RECONNECT_BASE_DELAY, RECONNECT_MAX_DELAY,
    STATUS_STOPPED, STATUS_CONNECTING, STATUS_CONNECTED,
    STATUS_RECONNECTING, STATUS_ERROR,
    K_RUNNING, K_WS_THREAD, K_TRADE_QUEUE, K_STATUS_QUEUE, K_WS_STATUS,
    K_ACTIVE_SYMBOL,
)


# ── Schema parser ────────────────────────────────────────────

def parse_binance_trade(raw_msg: str) -> dict | None:
    """
    KAPPA NOTE: Deserialisation / Schema Normalisation
    ───────────────────────────────────────────────────
    Converts the raw Binance JSON wire format into our internal
    trade schema. In production this is a Kafka Deserializer with
    an Avro/JSON schema registry. The internal schema is:

        symbol      str   "BTC/USDT"
        side        str   "BUY" | "SELL"
        price       float USD per unit
        quantity    float units
        total_usd   float price × quantity
        ts_epoch    float Unix epoch seconds (float for sub-second res)
        timestamp   str   "HH:MM:SS.mmm" UTC display string

    Binance trade stream fields:
        p  price (string)       q  quantity (string)
        T  trade time (ms)      m  is-buyer-maker (bool)
        s  symbol (e.g. BTCUSDT)
    """
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


# ── Background WebSocket thread ──────────────────────────────

class BinanceStreamThread(threading.Thread):
    """
    Producer half of the Producer-Consumer pattern.

    Lifecycle:
        start_stream() → BinanceStreamThread.start()
                       → run() blocks on ws.run_forever()
                       → on each message: trade_queue.put_nowait(trade)
        stop_stream()  → self._stop.set() → ws.close() → thread exits
    """

    def __init__(self, trade_queue: queue.Queue, status_queue: queue.Queue, slug: str = "btcusdt"):
        super().__init__(daemon=True)
        self.trade_queue  = trade_queue
        self.status_queue = status_queue
        self.slug         = slug.lower()
        self.ws_url       = binance_ws_url(self.slug)
        self._stop        = threading.Event()
        self.attempt      = 0
        self._ws_app      = None   # held so stop() can close it

    def stop(self):
        """Signal shutdown; safe to call from any thread."""
        self._stop.set()
        if self._ws_app:
            try:
                self._ws_app.close()
            except Exception:
                pass

    # ── websocket-client callbacks ───────────────────────────

    def _push_status(self, status: str, msg: str = "") -> None:
        """Non-blocking — drops the update if the queue is full."""
        try:
            self.status_queue.put_nowait({"status": status, "msg": msg})
        except queue.Full:
            pass

    def _on_open(self, ws):
        self.attempt = 0
        self._push_status(STATUS_CONNECTED,
                          f"Connected → {self.ws_url}")

    def _on_message(self, ws, message: str):
        if self._stop.is_set():
            ws.close()
            return
        trade = parse_binance_trade(message)
        if trade:
            try:
                self.trade_queue.put_nowait(trade)
            except queue.Full:
                pass   # backpressure: drop oldest via maxsize

    def _on_error(self, ws, error):
        self._push_status(STATUS_RECONNECTING,
                          f"WS error [{type(error).__name__}]: {error}")

    def _on_close(self, ws, code, msg):
        if not self._stop.is_set():
            self._push_status(STATUS_RECONNECTING,
                              f"Closed (code={code}). Scheduling reconnect…")

    # ── Main loop with exponential backoff ───────────────────

    def run(self):
        self._push_status(STATUS_CONNECTING, "Establishing WebSocket connection…")

        while not self._stop.is_set():
            if self.attempt >= MAX_RECONNECT:
                self._push_status(STATUS_ERROR,
                    f"Gave up after {MAX_RECONNECT} attempts. "
                    "Click STOP then START to retry.")
                break

            try:
                self._ws_app = websocket.WebSocketApp(
                    self.ws_url,
                    on_open    = self._on_open,
                    on_message = self._on_message,
                    on_error   = self._on_error,
                    on_close   = self._on_close,
                )
                # Blocks until disconnected
                self._ws_app.run_forever(ping_interval=20, ping_timeout=10)

            except Exception as exc:
                self._push_status(STATUS_RECONNECTING,
                                  f"Unhandled exception: {exc}")

            if self._stop.is_set():
                break

            # Exponential backoff sleep (interruptible)
            self.attempt += 1
            delay = min(RECONNECT_BASE_DELAY * (2 ** (self.attempt - 1)),
                        RECONNECT_MAX_DELAY)
            self._push_status(STATUS_RECONNECTING,
                f"Attempt {self.attempt}/{MAX_RECONNECT} — "
                f"retrying in {delay:.0f}s…")
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                if self._stop.is_set():
                    break
                time.sleep(0.1)

        self._push_status(STATUS_STOPPED, "WebSocket thread stopped.")


# ── Lifecycle helpers (called from main.py) ──────────────────

def start_stream(slug: str = "btcusdt") -> None:
    """
    Spawn the background thread and wire up queues.
    Idempotent — safe to call if already running.
    """
    if state.get(K_RUNNING):
        return

    tq = queue.Queue(maxsize=2000)
    sq = queue.Queue(maxsize=100)
    t  = BinanceStreamThread(trade_queue=tq, status_queue=sq, slug=slug)
    t.start()

    state.set(K_TRADE_QUEUE,  tq)
    state.set(K_STATUS_QUEUE, sq)
    state.set(K_WS_THREAD,    t)
    state.set(K_RUNNING,      True)
    state.set(K_ACTIVE_SYMBOL, slug)
    state.set(K_WS_STATUS,    STATUS_CONNECTING)


def stop_stream() -> None:
    """Signal the background thread to stop and clean up state."""
    t = state.get(K_WS_THREAD)
    if t and t.is_alive():
        t.stop()
    state.set(K_RUNNING,      False)
    state.set(K_WS_STATUS,    STATUS_STOPPED)
    state.set(K_WS_THREAD,    None)
    state.set(K_TRADE_QUEUE,  None)
    state.set(K_STATUS_QUEUE, None)


def switch_symbol(new_slug: str) -> None:
    """
    Stop the current stream and immediately restart on a new symbol.
    Clears feeds so stale data from the old symbol doesn't persist.
    """
    stop_stream()
    import state as _state
    _state.reset_stats()
    start_stream(slug=new_slug)


def drain_status_queue() -> None:
    """
    Pull all pending status messages from the background thread.
    Called once per Streamlit rerun, from the main thread only.
    The last status wins (most-recent connection state).
    """
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
