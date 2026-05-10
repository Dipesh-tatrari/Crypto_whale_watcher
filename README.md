<div align="center">

# 🐋 Crypto Whale Watcher

### Real-time Crypto Trade Intelligence · Kappa Architecture · ML-powered Sentiment

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cryptowhalewatcher-7vm7meslhibz2lhranarp5.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Binance](https://img.shields.io/badge/Binance-F0B90B?style=for-the-badge&logo=binance&logoColor=black)](https://binance.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**[🚀 Live Demo](https://cryptowhalewatcher-7vm7meslhibz2lhranarp5.streamlit.app/)** · **[📖 Documentation](#architecture)** · **[🐛 Report Bug](https://github.com/Dipesh-tatrari/Crypto_whale_watcher/issues)**

![Crypto Whale Watcher Dashboard](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

</div>

---

## 📌 What is Crypto Whale Watcher?

Crypto Whale Watcher is a **real-time market intelligence dashboard** that detects large cryptocurrency trades — called "whale trades" — as they happen on Binance. It goes beyond simple price tracking by identifying **stealth whale behaviour** (large orders disguised as many small ones), profiling repeat traders, and scoring market sentiment using a **fine-tuned FinBERT model** trained on crypto-specific data.

Built on a **Kappa Architecture** — all data is processed as a live stream with no separate batch layer — the system is designed to be extended toward a production-grade pipeline using Kafka, Flink, and ClickHouse.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🐋 **Single Whale Detection** | Flags individual trades above a configurable USD threshold in real time |
| 🔍 **Stealth Whale Clustering** | Groups small trades within a time window to catch iceberg/layering patterns |
| ⚠️ **Suspicion Scoring** | Rates each cluster `🔴 HIGH / 🟡 MEDIUM / 🟢 LOW` based on trade count and average size |
| 📰 **ML Sentiment Analysis** | Fine-tuned FinBERT scores crypto news headlines as BULLISH / NEUTRAL / BEARISH |
| 📊 **SVG Sentiment Gauge** | Real-time arc gauge showing market mood for each symbol |
| 👤 **Wallet Profiling** | Tracks repeat traders, labels known exchange wallets, detects Smart Money |
| 🧠 **Smart Money Detection** | Flags wallets that consistently buy during bearish sentiment (contrarian accumulation) |
| 🗂️ **Wallet Dossier** | Per-wallet deep-dive: volume, favourite assets, hourly activity heatmap, sentiment correlation |
| 💾 **SQLite Persistence** | Wallet profiles and trade history survive across sessions |
| 🔔 **Toast Alerts** | Popup notifications for high-suspicion stealth whale clusters |
| 🔄 **Live Symbol Switching** | Switch between BTC, ETH, SOL, BNB, XRP, AVAX, DOGE, ARB without restarting |
| 🌐 **Cloud-safe Fallback** | Simulation mode activates automatically when Binance WebSocket is blocked (e.g. AWS) |

---

## 🏗️ Architecture

This project implements a **Kappa Architecture** — a streaming-only data pipeline where all processing happens on the live event stream.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KAPPA ARCHITECTURE                           │
│                                                                     │
│  SOURCE          PROCESS              ENRICH            SINK        │
│                                                                     │
│  Binance    →   Whale Filter    →   Sentiment     →   Streamlit    │
│  WebSocket      (stateless)         (FinBERT)         Dashboard    │
│                      ↓                                              │
│               Cluster Detection  →  Wallet          →   SQLite     │
│               (sliding window)      Profiler            DB         │
│                                                                     │
│  Phase 6:  Kafka → Flink → ClickHouse → Grafana                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Map

```
crypto_whale_watcher/
│
├── main.py                    # Streamlit entry point — UI, tabs, polling loop
├── config.py                  # All constants, session state keys, defaults
├── state.py                   # Typed session state init/get/set
├── ingestion.py               # Binance WebSocket thread + simulation fallback
├── processor.py               # Whale filter + cluster aggregation + toast alerts
├── sentiment.py               # Background sentiment scorer (FinBERT / lexicon)
├── utils.py                   # Pure formatting helpers (no Streamlit imports)
├── requirements.txt
│
├── modules/
│   ├── visualization.py       # All UI components (gauge, styled feed, dossier)
│   └── profiler.py            # Wallet profiling, SQLite layer, smart money logic
│
└── ml/
    ├── crypto_sentiment_dataset.py   # 243 labeled crypto headlines
    ├── train.py                      # Fine-tunes FinBERT on the dataset
    └── predictor.py                  # Model loader + inference wrapper
```

---

## 🚀 Quick Start

### Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/Dipesh-tatrari/Crypto_whale_watcher.git
cd Crypto_whale_watcher
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run main.py
```

The app opens at `http://localhost:8501`. Click **▶ START TRACKING** in the sidebar.

---

### Optional: Enable ML Sentiment Model

By default the app uses a fast lexicon scorer. To use the fine-tuned FinBERT:

**1. Install ML dependencies**
```bash
pip install torch==2.1.2 transformers==4.37.2 scikit-learn
```

**2. Train the model** (~5 min on CPU, ~30s on GPU)
```bash
python ml/train.py
```

The trained model is saved to `ml/crypto_finbert/`. The app detects it automatically on next start and the sidebar shows `🧠 Fine-tuned FinBERT`.

---

## ☁️ Deployment

### Streamlit Community Cloud (Free)

**1. Fork or push to GitHub**
```bash
git add .
git commit -m "deploy"
git push origin main
```

**2. Deploy**
- Go to [share.streamlit.io](https://share.streamlit.io)
- New app → select your repo → Main file: `main.py`
- Click Deploy

**3. Add secrets** (Settings → Secrets):
```toml
HF_TOKEN      = "hf_your_token_here"
HF_MODEL_REPO = "your_hf_username/crypto-whale-finbert"
```

> **Note:** Streamlit Cloud runs on AWS. Binance blocks WebSocket connections from AWS IPs. The app automatically falls back to **Simulation Mode** — all features work identically with realistic synthetic data.

---

### HuggingFace Model Hosting

Upload your trained model so the cloud app can use it:

**1. Login**
```bash
hf auth login
```

**2. Create a private model repo**
```bash
hf repo create crypto-whale-finbert --type model --private
```

**3. Upload the trained model**
```bash
hf upload YOUR_HF_USERNAME/crypto-whale-finbert ml/crypto_finbert .
```

The app reads `HF_TOKEN` and `HF_MODEL_REPO` from Streamlit secrets and downloads the model at startup.

---

## 🎛️ Configuration

All thresholds are adjustable from the sidebar without restarting:

| Control | Default | Description |
|---|---|---|
| Single-Trade Threshold | $500,000 | Minimum trade size to flag as a single whale |
| Cluster Threshold | $1,000,000 | Cluster sum above this triggers a stealth whale event |
| Cluster Window | 3 seconds | Rolling time window for grouping trades |
| Min Sentiment Strength | 0.0 | Filter feed by minimum absolute sentiment score |
| UI Refresh Interval | 1.0s | How often the dashboard polls for new data |
| Max Feed Rows | 100 | Rolling history window depth |

---

## 📊 What the App Shows

### 📊 Live Feed Tab
Combined real-time table of all whale events (single trades + clusters), colour-coded:
- 🟩 Green rows — large BUY orders
- 🟥 Red rows — large SELL orders
- 🟪 Purple rows — cluster events (stealth whale)

### 🐋 Single Whales Tab
Individual trades above the threshold. Each row shows price, quantity, total USD, sentiment score at time of trade, and wallet label.

### 🔍 Clusters Tab
Stealth whale events where many small trades summed above the cluster threshold. Shows trade count, time window, and suspicion rating.

### 🧩 Cluster Breakdown Tab
Expandable drill-down for each cluster — individual trades, size distribution bar chart, and pattern analysis.

### 👤 Wallet Intelligence Tab
Leaderboard of all wallets profiled this session, sortable by volume, trade count, or last seen. Tag legend:

| Tag | Meaning |
|---|---|
| 🧠 Smart Money | Buys during bearish sentiment — contrarian accumulation |
| ⚡ HF Whale | High-frequency trader (>20 trades/session) |
| 🏛️ Known Entity | Exchange or market maker (address in registry) |
| ❓ Unknown | Unidentified wallet |

### 🗂️ Wallet Dossier Tab
Deep-dive for any selected wallet:
- 6-metric KPI strip (volume, trade count, avg size, buy ratio, smart money score, behaviour)
- Favourite trading pairs
- Sentiment correlation analysis with narrative
- Hourly activity heatmap (UTC)
- Last 30 trades table

### 📰 Market Sentiment Panel (right column)
- SVG arc gauge for the active symbol
- Per-symbol sentiment cards (BTC, ETH, SOL, BNB)
- Latest scored headline
- Backend badge showing which model is active

---

## 🧠 ML Sentiment Model

The sentiment system has three backends loaded in priority order:

```
🧠 Fine-tuned FinBERT  →  fastest on GPU, most accurate for crypto
🤖 Base FinBERT         →  good general financial sentiment
📖 Lexicon Scorer       →  pure Python, zero dependencies, always works
```

### Dataset
243 hand-labeled crypto headlines covering:
- 83 BULLISH examples (rallies, ETF approvals, accumulation)
- 83 NEUTRAL examples (routine updates, governance votes)
- 77 BEARISH examples (crashes, hacks, regulatory crackdowns)

### Training
```bash
python ml/train.py
```
- Base model: `ProsusAI/finbert` (pre-trained on financial news)
- Fine-tuning: 6 epochs, batch size 8, learning rate 2e-5
- Output: continuous score in [-1.0, +1.0] via `P(BULLISH) - P(BEARISH)`
- Typical results: Val F1 ~0.80, Val Accuracy ~0.78

---

## 🔬 Detection Logic

### Single Whale
```
trade.total_usd >= whale_threshold  →  emit WhaleEvent
```

### Stealth Whale Cluster (Anti-Evasion)
```
For each new trade on symbol S:
  1. Evict trades older than cluster_window seconds
  2. Append new trade to buffer[S]
  3. If sum(buffer[S]) >= cluster_threshold:
       emit ClusterWhaleEvent
       flush buffer[S]
```

### Suspicion Rating
```
trade_count >= 10 AND avg_size < $50,000  →  🔴 HIGH  (iceberg pattern)
trade_count >= 5  OR  avg_size < $150,000 →  🟡 MEDIUM
otherwise                                 →  🟢 LOW
```

### Smart Money Detection
```
For each whale trade:
  if side == BUY and current_sentiment < -0.1:
    wallet.pre_shift_buys += 1
  wallet.pre_shift_total += 1

smart_money_score = pre_shift_buys / pre_shift_total
if score >= 0.6 and total_trades >= 5:
  tag = SMART_MONEY
```

---

## 🗺️ Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Static prototype — simulated data, Streamlit UI |
| Phase 2 | ✅ Complete | Live Binance WebSocket, real BTC/USDT data |
| Phase 3 | ✅ Complete | Stream processing — whale filter + cluster detection |
| Phase 4 | ✅ Complete | Visualization — SVG gauge, styled feed, cluster breakdown |
| Phase 5 | ✅ Complete | Wallet profiling, SQLite persistence, smart money detection |
| Phase 6 | 🔜 Planned | Kafka producer → ClickHouse sink → Grafana dashboard |
| Phase 7 | 🔜 Planned | On-chain wallet data (Nansen / Arkham API integration) |
| Phase 8 | 🔜 Planned | FinBERT v2 trained on larger labeled dataset (10K+ examples) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit, Pandas Styler, Custom SVG |
| **Stream Source** | Binance WebSocket (`websocket-client`) |
| **Stream Processing** | Python threading, `queue.Queue` (Kappa pattern) |
| **ML Model** | FinBERT (`ProsusAI/finbert`) fine-tuned on crypto data |
| **Model Hosting** | HuggingFace Hub (private repo) |
| **Persistence** | SQLite (`whale_profiles.db`) |
| **Address Labeling** | Static registry + `functools.lru_cache` |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | GitHub |

---

## 📁 Key Files Reference

| File | Purpose |
|---|---|
| `ingestion.py` | WebSocket consumer — tries 3 Binance endpoints, falls back to simulation |
| `processor.py` | Core Kappa operators: filter, cluster aggregation, sentiment hook, toast queue |
| `sentiment.py` | Background fetcher: seeds synthetic data → loads ML model → polls RSS |
| `modules/profiler.py` | Full wallet intelligence: SQLite CRUD, lru_cache labeling, smart money scoring |
| `modules/visualization.py` | All UI components: SVG gauge, Pandas Styler feed, cluster expander, dossier |
| `ml/train.py` | Fine-tuning script: loads FinBERT, trains 6 epochs, saves best checkpoint |
| `ml/predictor.py` | Inference singleton: local → HF Hub → base FinBERT → lexicon |

---

## ⚙️ Environment Variables / Secrets

| Variable | Where to set | Purpose |
|---|---|---|
| `HF_TOKEN` | Streamlit secrets or `.env` | Authenticates with HuggingFace Hub |
| `HF_MODEL_REPO` | Streamlit secrets or `.env` | `username/crypto-whale-finbert` |

For local development, create a `.env` file:
```
HF_TOKEN=hf_your_token_here
HF_MODEL_REPO=your_username/crypto-whale-finbert
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

To extend the sentiment dataset, add labeled examples to `ml/crypto_sentiment_dataset.py` and rerun `python ml/train.py`.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Dipesh Tatrari**
- GitHub: [@Dipesh-tatrari](https://github.com/Dipesh-tatrari)
- Live App: [cryptowhalewatcher.streamlit.app](https://cryptowhalewatcher-7vm7meslhibz2lhranarp5.streamlit.app/)

---

<div align="center">

Built with ❤️ using Python, Streamlit, and a Kappa Architecture

⭐ Star this repo if you found it useful

</div>
