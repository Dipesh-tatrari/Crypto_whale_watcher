"""
ml/train.py — Fine-tune FinBERT on Crypto Sentiment Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run from the project root:
    python ml/train.py

What this does:
  1. Loads ProsusAI/finbert (pre-trained on financial news)
  2. Adds a classification head for 3 classes (BEARISH/NEUTRAL/BULLISH)
  3. Fine-tunes on our 300 crypto-specific labeled headlines
  4. Evaluates on a held-out validation split
  5. Saves the model to  ml/crypto_finbert/
  6. Generates a classification report

Requirements (install once):
    pip install transformers torch scikit-learn datasets

After training, sentiment.py automatically loads the saved model
if ml/crypto_finbert/ exists, otherwise falls back to the base
FinBERT or the lexicon scorer.

TRAINING TIME:
  CPU only: ~5-10 minutes
  GPU (CUDA): ~30 seconds
  MPS (Apple Silicon): ~1-2 minutes
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent   # project root
ML_DIR   = Path(__file__).parent
MODEL_OUT = ML_DIR / "crypto_finbert"

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Hyperparameters ──────────────────────────────────────────
BASE_MODEL     = "ProsusAI/finbert"   # pre-trained financial BERT
MAX_LEN        = 128       # max tokens — headlines are short
BATCH_SIZE     = 8         # small batch for CPU compatibility
EPOCHS         = 6         # enough for fine-tuning on small dataset
LEARNING_RATE  = 2e-5      # standard for BERT fine-tuning
WARMUP_RATIO   = 0.1       # 10% warmup steps
WEIGHT_DECAY   = 0.01
VAL_SPLIT      = 0.15      # 15% of data for validation
LABEL_NAMES    = ["BEARISH", "NEUTRAL", "BULLISH"]
NUM_LABELS     = 3


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> tuple[list[str], list[int]]:
    """Load and return (texts, labels) from the dataset module."""
    sys.path.insert(0, str(ROOT))
    from ml.crypto_sentiment_dataset import get_dataset, get_stats
    data  = get_dataset()
    stats = get_stats()
    print(f"\n📊 Dataset loaded: {stats['total']} examples")
    print(f"   BULLISH {stats['bullish']} ({stats['bullish_pct']})  "
          f"NEUTRAL {stats['neutral']} ({stats['neutral_pct']})  "
          f"BEARISH {stats['bearish']} ({stats['bearish_pct']})")
    random.shuffle(data)
    texts  = [t for t, _ in data]
    labels = [l for _, l in data]
    return texts, labels


def build_dataset(texts, labels, tokenizer, max_len: int):
    """Build a PyTorch Dataset from texts and labels."""
    import torch
    from torch.utils.data import Dataset

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    class CryptoSentimentDataset(Dataset):
        def __init__(self, enc, labs):
            self.enc    = enc
            self.labels = torch.tensor(labs, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "token_type_ids": self.enc.get("token_type_ids",
                                    torch.zeros_like(self.enc["input_ids"]))[idx],
                "labels":         self.labels[idx],
            }

    return CryptoSentimentDataset(encodings, labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute accuracy and per-class F1."""
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report
    )
    pred_classes = np.argmax(preds, axis=1)
    acc   = accuracy_score(labels, pred_classes)
    f1    = f1_score(labels, pred_classes, average="macro")
    report = classification_report(
        labels, pred_classes, target_names=LABEL_NAMES
    )
    return {"accuracy": acc, "macro_f1": f1, "report": report}


def train():
    import torch
    from torch.utils.data import DataLoader, random_split
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    set_seed(SEED)

    # ── Device ────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n🖥️  Training on GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🖥️  Training on Apple MPS")
    else:
        device = torch.device("cpu")
        print("\n🖥️  Training on CPU (this will take ~5-10 minutes)")

    # ── Tokenizer & Model ─────────────────────────────────────
    print(f"\n📥 Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
        ignore_mismatched_sizes=True,   # replaces FinBERT's 3-class head
    )
    model.to(device)

    # ── Data ──────────────────────────────────────────────────
    texts, labels = load_data()
    dataset = build_dataset(texts, labels, tokenizer, MAX_LEN)

    val_size   = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"\n✂️  Split: {train_size} train / {val_size} val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # ── Optimiser & Scheduler ─────────────────────────────────
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)

    optimizer  = AdamW(model.parameters(), lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DECAY)
    scheduler  = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────
    print(f"\n🚀 Training for {EPOCHS} epochs "
          f"({total_steps} total steps, {warmup_steps} warmup)…\n")

    best_val_f1  = 0.0
    best_epoch   = 0
    train_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            batch    = {k: v.to(device) for k, v in batch.items()}
            outputs  = model(**batch)
            loss     = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 5 == 0 or step == len(train_loader):
                avg = epoch_loss / step
                print(f"  Epoch {epoch}/{EPOCHS}  "
                      f"Step {step:3d}/{len(train_loader)}  "
                      f"Loss {avg:.4f}  "
                      f"LR {scheduler.get_last_lr()[0]:.2e}",
                      end="\r")

        # ── Validation ────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch   = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits  = outputs.logits.cpu().numpy()
                labs    = batch["labels"].cpu().numpy()
                all_preds.append(logits)
                all_labels.append(labs)

        all_preds  = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics    = compute_metrics(all_preds, all_labels)
        elapsed    = time.time() - t0
        train_losses.append(epoch_loss / len(train_loader))

        print(f"\n  Epoch {epoch}/{EPOCHS}  "
              f"Train Loss {train_losses[-1]:.4f}  "
              f"Val Acc {metrics['accuracy']:.3f}  "
              f"Val F1 {metrics['macro_f1']:.3f}  "
              f"({elapsed:.0f}s)")

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            best_epoch  = epoch
            # Save best checkpoint
            MODEL_OUT.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_OUT)
            tokenizer.save_pretrained(MODEL_OUT)
            print(f"  ✅ New best model saved (F1={best_val_f1:.3f})")

    # ── Final report ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Best model: Epoch {best_epoch}  Val F1 = {best_val_f1:.3f}")
    print(f"   Saved to: {MODEL_OUT}")
    print(f"\n📋 Final validation classification report:")
    print(metrics["report"])

    # Save training metadata
    meta = {
        "base_model":    BASE_MODEL,
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "best_epoch":    best_epoch,
        "best_val_f1":   best_val_f1,
        "best_val_acc":  metrics["accuracy"],
        "label_names":   LABEL_NAMES,
        "trained_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_size":    train_size,
        "val_size":      val_size,
    }
    with open(MODEL_OUT / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n📄 Training metadata saved to {MODEL_OUT / 'training_meta.json'}")
    print(f"\n🎯 Run your app — sentiment.py will automatically use the new model.")
    return best_val_f1


if __name__ == "__main__":
    try:
        score = train()
        sys.exit(0 if score > 0.5 else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
