"""
ml/predictor.py — Inference Wrapper  (Cloud-safe version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
On Streamlit Cloud: torch/transformers are NOT installed.
  → Falls straight to lexicon scorer. Fast, zero memory overhead.

On your local machine (after pip install torch transformers):
  → Tries fine-tuned model first, then base FinBERT, then lexicon.

This file is safe to import regardless of what is installed.
"""

from __future__ import annotations
import os
import math
import re
import time
import threading
from dataclasses import dataclass
from pathlib import Path

ML_DIR        = Path(__file__).parent
FINETUNED_DIR = ML_DIR / "crypto_finbert"
BASE_MODEL    = "ProsusAI/finbert"
LABEL_MAP     = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

# ── Detect environment ────────────────────────────────────────
def _on_cloud() -> bool:
    """True when running on Streamlit Cloud (no GPU, limited RAM)."""
    return bool(
        os.getenv("STREAMLIT_SHARING_MODE") or
        os.getenv("IS_STREAMLIT_CLOUD") or
        not os.access(".", os.W_OK)
    )

def _torch_available() -> bool:
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class PredictorResult:
    text:       str
    label:      str
    score:      float
    confidence: float
    backend:    str
    latency_ms: float

    @property
    def label_emoji(self) -> str:
        return {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}.get(self.label, "")


# ─────────────────────────────────────────────────────────────
# LEXICON SCORER  (zero dependencies — always available)
# ─────────────────────────────────────────────────────────────

_POS = {
    "rally":1.5,"surge":1.5,"soar":1.5,"breakout":1.5,"bullish":1.5,
    "bull":1.0,"uptrend":1.2,"adoption":1.0,"approval":1.2,"etf":1.3,
    "institutional":1.0,"recovery":1.0,"rebound":1.0,"highs":1.2,
    "ath":2.0,"accumulate":1.0,"moon":1.5,"gain":0.8,"profit":1.0,
    "growth":1.0,"inflow":1.0,"momentum":0.8,"boom":1.5,"record":0.8,
    "skyrocket":2.0,"hodl":0.5,"green":0.5,"optimism":1.0,"expand":0.7,
    "partnership":1.0,"launch":0.8,"upgrade":0.8,"support":0.6,
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
    t0     = time.time()
    tokens = re.findall(r"[\w'-]+", text.lower())
    raw    = sum(_POS.get(t, 0) + _NEG.get(t, 0) for t in tokens)
    score  = math.tanh(raw * 0.4)
    label  = "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL"
    return PredictorResult(
        text=text, label=label, score=round(score, 4),
        confidence=round(min(abs(score) + 0.1, 1.0), 4),
        backend="lexicon",
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


# ─────────────────────────────────────────────────────────────
# TRANSFORMER PREDICTOR  (only used when torch is available)
# ─────────────────────────────────────────────────────────────

class TransformerPredictor:
    def __init__(self, model_path: str, backend_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.backend  = backend_name
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, text: str) -> PredictorResult:
        import torch
        t0  = time.time()
        enc = self.tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=128, padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs      = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        p_bear, p_neut, p_bull = (probs + [0.0, 0.0, 0.0])[:3]
        score = max(-1.0, min(1.0, float(p_bull - p_bear)))
        label = "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL"
        return PredictorResult(
            text=text, label=label, score=round(score, 4),
            confidence=round(float(max(p_bear, p_neut, p_bull)), 4),
            backend=self.backend,
            latency_ms=round((time.time() - t0) * 1000, 2),
        )


# ─────────────────────────────────────────────────────────────
# SINGLETON LOADER
# ─────────────────────────────────────────────────────────────

_predictor: TransformerPredictor | None = None
_predictor_lock = threading.Lock()
_load_status    = "not_loaded"
_load_error     = ""


def load(force_backend: str = "auto") -> str:
    global _predictor, _load_status, _load_error

    with _predictor_lock:
        if _load_status not in ("not_loaded",) and force_backend == "auto":
            return _load_status

        # Skip ALL transformer loading on Cloud or when torch not installed
        if _on_cloud() or not _torch_available():
            _load_status = "lexicon"
            print("  ℹ️  Cloud/no-torch detected — using lexicon scorer")
            return _load_status

        _load_status = "loading"

        # 1. Fine-tuned local model
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

        # 2. HuggingFace Hub model (env var set)
        hf_repo = os.getenv("HF_MODEL_REPO", "")
        if hf_repo:
            try:
                hf_token = os.getenv("HF_TOKEN", "")
                if hf_token:
                    from huggingface_hub import login
                    login(token=hf_token, add_to_git_credential=False)
                print(f"🧠 Loading fine-tuned model from Hub: {hf_repo}…")
                _predictor   = TransformerPredictor(hf_repo, "finetuned")
                _load_status = "finetuned"
                print(f"   ✅ Hub model loaded on {_predictor.device}")
                return _load_status
            except Exception as e:
                _load_error = str(e)
                print(f"   ⚠️  Hub model load failed: {e}")

        # 3. Base FinBERT
        try:
            print(f"🧠 Loading base FinBERT ({BASE_MODEL})…")
            _predictor   = TransformerPredictor(BASE_MODEL, "finbert")
            _load_status = "finbert"
            print(f"   ✅ Base FinBERT loaded on {_predictor.device}")
            return _load_status
        except Exception as e:
            _load_error = str(e)
            print(f"   ⚠️  FinBERT load failed: {e}")

        # 4. Lexicon fallback
        _predictor   = None
        _load_status = "lexicon"
        print("   ℹ️  Using lexicon fallback")
        return _load_status


def predict(text: str) -> PredictorResult:
    if _predictor is not None:
        try:
            return _predictor.predict(text)
        except Exception:
            pass
    return _lexicon_predict(text)


def backend_name() -> str:
    return _load_status

def is_transformer() -> bool:
    return _load_status in ("finetuned", "finbert")

def get_load_error() -> str:
    return _load_error

def training_meta() -> dict | None:
    import json
    meta_path = FINETUNED_DIR / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None
