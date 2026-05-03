"""
ml/predictor.py — Inference Wrapper for the Fine-tuned Crypto FinBERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This module is the single interface between the trained model and the
rest of the app. sentiment.py calls predict() and never touches
transformers directly.

LOADING PRIORITY:
  1. ml/crypto_finbert/   — your fine-tuned model (after running train.py)
  2. ProsusAI/finbert     — base financial BERT (no fine-tuning, still good)
  3. Lexicon fallback      — pure Python, zero dependencies, always works

The model is loaded ONCE into _PREDICTOR at module level.
Subsequent calls to predict() reuse the same loaded model — no
repeated disk I/O on every Streamlit rerun.

THREAD SAFETY:
  The model is loaded once before the Streamlit server starts accepting
  connections (STEP 3 in main.py). After that, predict() is called
  only from the SentimentBackgroundFetcher daemon thread which runs
  one inference at a time. No concurrent model calls → no locking needed.
"""

from __future__ import annotations

import os
import math
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────
ML_DIR        = Path(__file__).parent
FINETUNED_DIR = ML_DIR / "crypto_finbert"
BASE_MODEL    = "ProsusAI/finbert"

LABEL_MAP = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

# ─────────────────────────────────────────────────────────────
# PREDICTOR RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class PredictorResult:
    text:       str
    label:      str        # BEARISH | NEUTRAL | BULLISH
    score:      float      # [-1, +1] — negative=bearish, positive=bullish
    confidence: float      # [0, 1]
    backend:    str        # "finetuned" | "finbert" | "lexicon"
    latency_ms: float      # inference time

    @property
    def label_emoji(self) -> str:
        return {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(self.label, "")


# ─────────────────────────────────────────────────────────────
# LEXICON FALLBACK  (zero-dependency, always available)
# ─────────────────────────────────────────────────────────────

_POS = {
    "rally":1.5,"surge":1.5,"soar":1.5,"breakout":1.5,"bullish":1.5,
    "bull":1.0,"uptrend":1.2,"adoption":1.0,"approval":1.2,"etf":1.3,
    "institutional":1.0,"recovery":1.0,"rebound":1.0,"highs":1.2,
    "ath":2.0,"accumulate":1.0,"moon":1.5,"gain":0.8,"profit":1.0,
    "growth":1.0,"inflow":1.0,"momentum":0.8,"boom":1.5,"record":0.8,
    "skyrocket":2.0,"hodl":0.5,"green":0.5,"optimism":1.0,"expand":0.7,
}
_NEG = {
    "crash":-2.0,"plunge":-1.8,"collapse":-2.0,"dump":-1.5,"bearish":-1.5,
    "bear":-1.0,"downtrend":-1.2,"selloff":-1.5,"hack":-2.0,"exploit":-1.8,
    "breach":-1.5,"ban":-1.5,"crackdown":-1.5,"lawsuit":-1.2,"fraud":-2.0,
    "scam":-2.0,"rug":-2.0,"liquidation":-1.5,"fear":-1.0,"panic":-1.5,
    "fud":-1.0,"outflow":-1.0,"loss":-1.0,"drop":-1.0,"decline":-0.8,
    "correction":-0.7,"congestion":-0.6,"scrutiny":-0.8,"uncertainty":-0.7,
    "overbought":-0.6,"capitulate":-1.2,"insolvency":-1.8,"bankrupt":-2.0,
}

def _lexicon_predict(text: str) -> PredictorResult:
    import re
    t0     = time.time()
    lower  = text.lower()
    tokens = re.findall(r"[\w'-]+", lower)
    raw    = sum(_POS.get(tok, 0) + _NEG.get(tok, 0) for tok in tokens)
    score  = math.tanh(raw * 0.4)
    label  = "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL"
    return PredictorResult(
        text=text, label=label, score=round(score, 4),
        confidence=round(min(abs(score) + 0.1, 1.0), 4),
        backend="lexicon", latency_ms=round((time.time() - t0) * 1000, 2),
    )


# ─────────────────────────────────────────────────────────────
# TRANSFORMER-BASED PREDICTOR
# ─────────────────────────────────────────────────────────────

class TransformerPredictor:
    """
    Loads a FinBERT-style model once and exposes a predict() method.

    Score mapping (FinBERT outputs 3 classes):
        FinBERT label 0 → BEARISH  → score in [-1, -0.15]
        FinBERT label 1 → NEUTRAL  → score in (-0.15, +0.15)
        FinBERT label 2 → BULLISH  → score in [+0.15, +1]

    We convert softmax probabilities to a scalar score:
        score = P(BULLISH) - P(BEARISH)
    This gives a continuous [-1, +1] signal instead of a hard class.
    """

    def __init__(self, model_path: str, backend_name: str):
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )
        import torch

        self.backend = backend_name
        self.device  = (
            "cuda"  if torch.cuda.is_available() else
            "mps"   if (hasattr(torch.backends, "mps") and
                        torch.backends.mps.is_available()) else
            "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.model.eval()
        self.model.to(self.device)

        # Read label mapping from model config if available
        cfg = self.model.config
        if hasattr(cfg, "id2label") and cfg.id2label:
            self._id2label = {int(k): v.upper() for k, v in cfg.id2label.items()}
        else:
            self._id2label = LABEL_MAP

    def predict(self, text: str) -> PredictorResult:
        import torch
        t0 = time.time()

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        # Normalise to exactly 3 values regardless of model output size
        if len(probs) == 3:
            p_bear, p_neut, p_bull = probs
        else:
            p_bear = probs[0]
            p_bull = probs[-1]
            p_neut = 1.0 - p_bear - p_bull

        # Continuous score: P(bull) - P(bear), clamped to [-1, 1]
        raw_score = float(p_bull - p_bear)
        score     = max(-1.0, min(1.0, raw_score))
        label     = "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL"
        conf      = float(max(p_bear, p_neut, p_bull))

        return PredictorResult(
            text=text, label=label, score=round(score, 4),
            confidence=round(conf, 4), backend=self.backend,
            latency_ms=round((time.time() - t0) * 1000, 2),
        )


# ─────────────────────────────────────────────────────────────
# SINGLETON LOADER
# ─────────────────────────────────────────────────────────────

_predictor: TransformerPredictor | None = None
_predictor_lock   = threading.Lock()
_load_status: str = "not_loaded"   # not_loaded | loading | finetuned | finbert | lexicon
_load_error:  str = ""


def load(force_backend: str = "auto") -> str:
    """
    Load the best available model into the module-level singleton.

    Args:
        force_backend: "finetuned" | "finbert" | "lexicon" | "auto"
                       "auto" tries finetuned → finbert → lexicon

    Returns the backend name that was loaded.
    Called once at app startup from main.py STEP 3.
    """
    global _predictor, _load_status, _load_error

    with _predictor_lock:
        if _load_status not in ("not_loaded", "error") and force_backend == "auto":
            return _load_status   # already loaded

        _load_status = "loading"

        # ── 1. Try fine-tuned model ───────────────────────────
        if force_backend in ("auto", "finetuned"):
            if FINETUNED_DIR.exists() and (FINETUNED_DIR / "config.json").exists():
                try:
                    print(f"🧠 Loading fine-tuned model from {FINETUNED_DIR}…")
                    _predictor   = TransformerPredictor(str(FINETUNED_DIR), "finetuned")
                    _load_status = "finetuned"
                    print(f"   ✅ Fine-tuned model loaded on {_predictor.device}")
                    return _load_status
                except Exception as e:
                    _load_error = str(e)
                    print(f"   ⚠️  Fine-tuned load failed: {e}")

        # ── 2. Try base FinBERT from HuggingFace ─────────────
        if force_backend in ("auto", "finbert"):
            # Skip on Streamlit Cloud — HF downloads hang the app
            import os
            on_cloud = bool(os.getenv("STREAMLIT_SHARING_MODE"))
            if not on_cloud:
                try:
                    print(f"🧠 Loading base FinBERT ({BASE_MODEL})…")
                    _predictor   = TransformerPredictor(BASE_MODEL, "finbert")
                    _load_status = "finbert"
                    print(f"   ✅ Base FinBERT loaded on {_predictor.device}")
                    return _load_status
                except Exception as e:
                    _load_error = str(e)
                    print(f"   ⚠️  FinBERT load failed: {e}")

        # ── 3. Lexicon fallback ───────────────────────────────
        _predictor   = None
        _load_status = "lexicon"
        print("   ℹ️  Using lexicon fallback (no transformer model available)")
        return _load_status


def predict(text: str) -> PredictorResult:
    """
    Run sentiment inference on a single text string.
    Uses whichever backend was loaded. Falls back to lexicon
    if no model is loaded yet.
    """
    if _predictor is not None:
        try:
            return _predictor.predict(text)
        except Exception:
            pass
    return _lexicon_predict(text)


def backend_name() -> str:
    """Return the name of the currently loaded backend."""
    return _load_status


def is_transformer() -> bool:
    """True if a transformer model is loaded (not just lexicon)."""
    return _load_status in ("finetuned", "finbert")


def get_load_error() -> str:
    return _load_error


def training_meta() -> dict | None:
    """Return training metadata if a fine-tuned model exists."""
    import json
    meta_path = FINETUNED_DIR / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None