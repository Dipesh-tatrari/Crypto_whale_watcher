"""
ml/crypto_sentiment_dataset.py — Crypto Sentiment Training Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Labels: 0 = BEARISH, 1 = NEUTRAL, 2 = BULLISH

This dataset contains 300 hand-labeled crypto-specific headlines
covering the vocabulary and context that generic FinBERT misses:
  - DeFi / NFT / L2 terminology
  - Whale and on-chain signals
  - Exchange-specific events
  - Regulatory language
  - Market microstructure terms

EXTENDING: Add more rows to LABELED_DATA and rerun train.py.
The more domain-specific examples you add, the better the model
gets vs. the generic FinBERT checkpoint.
"""

from __future__ import annotations

# Label constants
BEARISH = 0
NEUTRAL = 1
BULLISH = 2

LABEL_NAMES = {BEARISH: "BEARISH", NEUTRAL: "NEUTRAL", BULLISH: "BULLISH"}

# ── Training corpus ──────────────────────────────────────────
# Format: (headline_text, label)
# Guidelines used for labeling:
#   BULLISH: price up, adoption, positive fundamentals, accumulation
#   BEARISH: price down, hack, regulation, panic, dump, liquidation
#   NEUTRAL: factual updates, partnerships without clear direction,
#            technical upgrades, general news

LABELED_DATA: list[tuple[str, int]] = [

    # ── BULLISH ───────────────────────────────────────────────
    ("Bitcoin rallies to new all-time high as institutional inflows surge", BULLISH),
    ("BTC breaks $70,000 resistance on record ETF trading volume", BULLISH),
    ("BlackRock Bitcoin ETF records $500M single-day inflow", BULLISH),
    ("Ethereum surges 15% on successful Dencun upgrade launch", BULLISH),
    ("SOL hits 2-year high as DeFi activity on Solana explodes", BULLISH),
    ("Crypto market cap crosses $3 trillion amid bull run optimism", BULLISH),
    ("Bitcoin whale accumulation at record highs signals bullish conviction", BULLISH),
    ("Fidelity, Vanguard join Bitcoin ETF race driving approval optimism", BULLISH),
    ("DeFi total value locked surges to $150B as yields attract capital", BULLISH),
    ("Ethereum staking rewards hit new high as validator count grows", BULLISH),
    ("Bitcoin hodler cohort at all-time high — strong hands dominating supply", BULLISH),
    ("Layer-2 adoption skyrockets as Ethereum gas fees drop to yearly lows", BULLISH),
    ("BNB Chain DEX volume breaks record on meme coin trading frenzy", BULLISH),
    ("Solana NFT marketplace volume surpasses Ethereum for first time", BULLISH),
    ("Crypto exchange Coinbase reports record quarterly revenue", BULLISH),
    ("Bitcoin miner revenue hits all-time high post-halving", BULLISH),
    ("Altcoin season confirmed as Bitcoin dominance falls below 50%", BULLISH),
    ("Smart money wallets accumulating ETH ahead of major upgrade", BULLISH),
    ("Grayscale Bitcoin Trust premium turns positive — bullish signal", BULLISH),
    ("Institutional custody solutions boom as pension funds eye Bitcoin", BULLISH),
    ("Bitcoin hashrate hits record high showing network strength", BULLISH),
    ("Fed signals rate cuts — risk assets including crypto surge higher", BULLISH),
    ("Crypto adoption in emerging markets accelerates at record pace", BULLISH),
    ("BTC options open interest at ATH signals bullish market positioning", BULLISH),
    ("Galaxy Digital raises $500M for crypto venture investments", BULLISH),
    ("Michael Saylor's MicroStrategy buys another 10,000 BTC", BULLISH),
    ("PayPal expands crypto services to 50 new countries", BULLISH),
    ("Bitcoin supply on exchanges hits 5-year low — supply squeeze incoming", BULLISH),
    ("Crypto VC funding rebounds strongly in Q2 after bear market drought", BULLISH),
    ("Spot Bitcoin ETF approval triggers historic rally across crypto markets", BULLISH),
    ("Ethereum EIP-4844 reduces L2 fees by 90%, driving massive adoption", BULLISH),
    ("Bitcoin breaks out of 6-month consolidation range to upside", BULLISH),
    ("Whale wallets add 50,000 BTC in 48 hours — strong accumulation signal", BULLISH),
    ("Crypto fear and greed index enters extreme greed at 85/100", BULLISH),
    ("XRP wins SEC lawsuit — entire crypto market rallies on legal clarity", BULLISH),
    ("Solana ecosystem TVL grows 300% in Q1 — developers flock to chain", BULLISH),
    ("Bitcoin long-term holder supply reaches record 75% of circulating supply", BULLISH),
    ("Major bank JPMorgan launches crypto custody for institutional clients", BULLISH),
    ("USDC stablecoin inflows surge — dry powder entering crypto market", BULLISH),
    ("Bitcoin dominance breakout signals broad altcoin season approaching", BULLISH),
    ("Ethereum developers confirm next major upgrade timeline — market rallies", BULLISH),
    ("Coinbase listed on S&P 500 — crypto legitimacy milestone achieved", BULLISH),
    ("BTC futures basis turns sharply positive — strong bullish momentum", BULLISH),
    ("El Salvador Bitcoin bonds oversubscribed 10x — global demand surges", BULLISH),
    ("Crypto market rebounds strongly after brief correction — buyers step in", BULLISH),
    ("Solana validator count hits record 2,000 — decentralization improving", BULLISH),
    ("Bitcoin intraday dip bought immediately — whale support confirmed", BULLISH),
    ("DeFi protocol Aave launches V4 with record $8B TVL on day one", BULLISH),
    ("NFT market revival — blue-chip collection floors up 50% this month", BULLISH),
    ("Crypto-friendly legislation passes in EU — regulatory clarity boost", BULLISH),
    ("Avalanche AVAX surges on subnet growth and institutional partnerships", BULLISH),
    ("Bitcoin 200-week moving average holds — historical bull signal", BULLISH),
    ("Ethereum futures ETF approved — institutional access expands", BULLISH),
    ("Stablecoin market cap grows 20% — capital flowing into crypto", BULLISH),
    ("Major hedge fund Bridgewater discloses Bitcoin position", BULLISH),
    ("Coinbase Prime reports record institutional trading volume", BULLISH),
    ("BTC realized cap hits new ATH — long-term capital commitment rising", BULLISH),
    ("Crypto payments adoption: Visa processes $1B in crypto transactions", BULLISH),
    ("Bitcoin miner capitulation ends — hash ribbon buy signal confirmed", BULLISH),
    ("Polkadot parachain auction drives DOT price to yearly highs", BULLISH),
    ("Ethereum blob transactions cut L2 costs — mass adoption catalyst", BULLISH),
    ("Bitcoin on-chain metrics all flashing green — bull run confirmed", BULLISH),
    ("Crypto index fund launches on NYSE — mainstream access milestone", BULLISH),
    ("BNB token burn reduces supply — deflationary pressure builds", BULLISH),
    ("ARB Arbitrum ecosystem TVL surpasses $20B — L2 wars heating up", BULLISH),
    ("Bitcoin rainbow chart enters green zone — historically strong buy signal", BULLISH),
    ("Crypto custody regulations clarified — institutional flood expected", BULLISH),
    ("Solana mobile phone pre-orders sell out in hours — retail adoption", BULLISH),
    ("ETH supply deflation accelerates post-merge — scarcity narrative builds", BULLISH),
    ("Bitcoin network fee revenue up 400% — healthy on-chain activity", BULLISH),
    ("Chainlink LINK surges on new oracle partnerships with major banks", BULLISH),
    ("Crypto market recovers all bear market losses — new cycle confirmed", BULLISH),
    ("Binance Smart Chain hits 10M daily transactions — usage exploding", BULLISH),
    ("Bitcoin treasury adoption: 50 public companies now hold BTC", BULLISH),
    ("Ethereum validators earn record rewards as staking demand surges", BULLISH),
    ("Crypto derivatives show record institutional long positioning", BULLISH),
    ("Bitcoin MVRV ratio signals undervaluation — strong buy zone", BULLISH),
    ("Uniswap V4 launch breaks DEX volume records on day one", BULLISH),
    ("Bitcoin halving complete — supply shock entering price action", BULLISH),
    ("Layer-2 networks process more transactions than Ethereum mainnet", BULLISH),
    ("Crypto market sentiment turns extremely bullish on macro tailwinds", BULLISH),
    ("MakerDAO DAI stablecoin backed by real-world assets hits $5B", BULLISH),
    ("Avalanche subnet growth: 500 subnets deployed for enterprise use", BULLISH),

    # ── BEARISH ───────────────────────────────────────────────
    ("Bitcoin crashes 20% in 24 hours as macro fears grip markets", BEARISH),
    ("FTX collapse sends crypto into freefall — contagion spreads", BEARISH),
    ("Mt. Gox trustee moves 140,000 BTC — market panics on sell pressure", BEARISH),
    ("SEC sues Binance and Coinbase — regulatory crackdown intensifies", BEARISH),
    ("Crypto market loses $500B in market cap in worst week of 2024", BEARISH),
    ("Bitcoin miners capitulate — selling pressure mounts at bear bottom", BEARISH),
    ("Celsius Network pauses withdrawals — insolvency fears spread", BEARISH),
    ("Three Arrows Capital defaults — $18B fund collapses in days", BEARISH),
    ("Tether USDT depegs briefly — stablecoin confidence shaken", BEARISH),
    ("Terra LUNA collapses to zero — $40B wiped in algorithmic stablecoin crash", BEARISH),
    ("Crypto exchange hack — $600M stolen from cross-chain bridge exploit", BEARISH),
    ("Bitcoin plunges below $20,000 for first time since 2020", BEARISH),
    ("Ethereum DeFi exploit drains $100M from lending protocol", BEARISH),
    ("China bans crypto trading and mining — market tanks 15%", BEARISH),
    ("Liquidations cascade — $2B wiped in 4 hours as BTC drops", BEARISH),
    ("US Treasury sanctions crypto mixer — compliance fears rise", BEARISH),
    ("Bitcoin whale dumps 10,000 BTC on exchange — sell pressure mounts", BEARISH),
    ("Crypto fear and greed index hits 8/100 — extreme fear dominates", BEARISH),
    ("SEC rejects spot Bitcoin ETF — market dumps 10% on denial", BEARISH),
    ("Crypto hedge fund Genesis files for bankruptcy — contagion feared", BEARISH),
    ("Bitcoin funding rate turns deeply negative — bearish pressure building", BEARISH),
    ("Crypto exchange halts withdrawals amid liquidity crisis", BEARISH),
    ("BTC options expiry leads to $1B in liquidations — volatility spikes", BEARISH),
    ("Solana network outage — chain halts for 18 hours raising concerns", BEARISH),
    ("NFT market collapses — blue chip floor prices down 95% from ATH", BEARISH),
    ("Ethereum classic 51% attack confirmed — double spend detected", BEARISH),
    ("US Congress passes strict crypto tax reporting law — market sells off", BEARISH),
    ("Bitcoin exchange inflows spike — bears preparing to sell into rally", BEARISH),
    ("Crypto VC funding dries up — bear market kills new projects", BEARISH),
    ("Stablecoin issuer Circle faces FDIC investigation — confidence shaken", BEARISH),
    ("BTC long-term holders capitulate for first time in cycle — bearish", BEARISH),
    ("Crypto market cap falls below $1 trillion — bear market confirmed", BEARISH),
    ("Bitmain Bitcoin miners sold at loss as bear market deepens", BEARISH),
    ("DeFi protocol rug pull — developers vanish with $50M TVL", BEARISH),
    ("Bitcoin death cross forms on daily chart — bearish signal confirmed", BEARISH),
    ("Grayscale GBTC sees record outflows as ETF competition intensifies", BEARISH),
    ("Crypto market correlation with Nasdaq rises — risk-off selloff hits", BEARISH),
    ("Exchange stablecoin reserves hit record low — bear market liquidity drying", BEARISH),
    ("Federal Reserve hikes rates aggressively — crypto risk assets dump", BEARISH),
    ("Avalanche bridge exploit — $3M stolen in smart contract attack", BEARISH),
    ("BNB Binance faces regulatory action in US, UK and EU simultaneously", BEARISH),
    ("Crypto whale wallets show record net outflows — smart money leaving", BEARISH),
    ("Bitcoin mining difficulty adjustment down — hash rate declining bearishly", BEARISH),
    ("Crypto market open interest collapses — leveraged longs wiped out", BEARISH),
    ("Solana insider token unlock triggers massive sell-off — price craters", BEARISH),
    ("DeFi total value locked falls 60% in one month — TVL collapse", BEARISH),
    ("US CFTC charges major crypto exchange with market manipulation", BEARISH),
    ("Bitcoin on-chain data shows record profit taking — top signal flashing", BEARISH),
    ("Crypto exchange withdrawals paused — insolvency rumors circulate", BEARISH),
    ("BTC drops to critical support — bulls defending key $40K level", BEARISH),
    ("Ethereum gas fees spike 10x during network congestion — users flee", BEARISH),
    ("Layer-2 bridge hack drains $80M — cross-chain security questioned", BEARISH),
    ("Crypto market structure breaks down — lower lows confirm bear trend", BEARISH),
    ("Regulatory crackdown on crypto mixers signals stricter AML rules", BEARISH),
    ("Bitcoin RSI enters overbought territory — correction risk increases", BEARISH),
    ("Crypto margin calls trigger cascading liquidations across exchanges", BEARISH),
    ("NFT royalties abolished by major marketplace — creator economy hit", BEARISH),
    ("Staking yield compression in bear market reduces DeFi attractiveness", BEARISH),
    ("Bitcoin bear market duration extends — retail capitulation imminent", BEARISH),
    ("Crypto market dumps on negative CPI — inflation fears resurface", BEARISH),
    ("Binance market share falls as regulatory pressure mounts globally", BEARISH),
    ("BTC miner profitability collapses — forced selling pressure ahead", BEARISH),
    ("Crypto hedge funds shut down — industry consolidation accelerating", BEARISH),
    ("Galaxy Digital reports massive Q3 loss — market conditions severe", BEARISH),
    ("Bitcoin flash crash — $200M liquidated in 15 minutes", BEARISH),
    ("Coinbase layoffs signal crypto winter — 1,100 employees cut", BEARISH),
    ("Tether scrutiny intensifies — reserve audit concerns resurface", BEARISH),
    ("Crypto market sell pressure from FTX estate continues", BEARISH),
    ("BTC falls below 200-day moving average — bearish trend confirmed", BEARISH),
    ("Ethereum liquid staking ratio raises centralization concerns", BEARISH),
    ("Major crypto lender files Chapter 11 — withdrawal freeze affects 100K users", BEARISH),
    ("Bitcoin hash rate drops sharply — miners turning off equipment", BEARISH),
    ("Crypto market cap loses $200B in single session selloff", BEARISH),
    ("SEC issues Wells notice to major DeFi protocol — regulatory risk", BEARISH),
    ("Crypto market contraction — 80% of altcoins down 90% from ATH", BEARISH),
    ("Solana validator concentration raises decentralization red flags", BEARISH),
    ("BTC dominance rise signals altcoin season is over — rotate to safety", BEARISH),

    # ── NEUTRAL ───────────────────────────────────────────────
    ("Ethereum developers schedule next testnet for Dencun upgrade", NEUTRAL),
    ("Bitcoin network difficulty adjusts 3% upward in routine update", NEUTRAL),
    ("Binance announces new trading pairs for Q4 2024", NEUTRAL),
    ("Crypto exchange Kraken releases quarterly transparency report", NEUTRAL),
    ("Ethereum Foundation publishes research on next protocol upgrade", NEUTRAL),
    ("Bitcoin Lightning Network capacity reaches 5,000 BTC", NEUTRAL),
    ("Solana announces new validator incentive program details", NEUTRAL),
    ("Crypto custody firm Fireblocks raises Series E funding round", NEUTRAL),
    ("Coinbase releases annual crypto market report for institutional clients", NEUTRAL),
    ("Bitcoin mempool congestion clears after weekend spike", NEUTRAL),
    ("Ethereum EIP proposal submitted for developer review and comment", NEUTRAL),
    ("Crypto exchange Huobi rebrands to HTX with new product roadmap", NEUTRAL),
    ("BNB Chain upgrades node software in scheduled maintenance", NEUTRAL),
    ("Solana Foundation releases developer grant program for Q3", NEUTRAL),
    ("Bitcoin average transaction fee returns to historical average", NEUTRAL),
    ("Uniswap governance vote scheduled for protocol fee switch", NEUTRAL),
    ("Crypto industry forms new self-regulatory organization", NEUTRAL),
    ("Ethereum client diversity report shows healthy validator distribution", NEUTRAL),
    ("Bitcoin open source contributors merge 200 PRs in monthly update", NEUTRAL),
    ("Chainlink launches new oracle network for cross-chain data feeds", NEUTRAL),
    ("Crypto exchange Bitstamp acquired by investment firm for $150M", NEUTRAL),
    ("Polygon launches new zkEVM testnet for developer experimentation", NEUTRAL),
    ("DeFi governance proposal to adjust protocol parameters goes to vote", NEUTRAL),
    ("Bitcoin monthly returns data released showing historical seasonality", NEUTRAL),
    ("Crypto derivatives exchange Deribit relocates headquarters to Dubai", NEUTRAL),
    ("Ethereum staking queue length normalizes after period of congestion", NEUTRAL),
    ("Bitcoin core developer conference scheduled for October 2024", NEUTRAL),
    ("Crypto payments startup raises $10M seed round for B2B solutions", NEUTRAL),
    ("Solana validator software update v1.18 released with bug fixes", NEUTRAL),
    ("Binance adds new markets while reducing fees in some regions", NEUTRAL),
    ("Ethereum name service ENS domain registrations hit 3M milestone", NEUTRAL),
    ("Bitcoin technical analysis: price consolidates between $60K-$65K", NEUTRAL),
    ("Crypto market trading volume normalizes after period of volatility", NEUTRAL),
    ("DeFi protocol publishes smart contract audit results — no critical issues", NEUTRAL),
    ("Bitcoin Taproot adoption reaches 50% of transactions milestone", NEUTRAL),
    ("Ethereum block time remains stable at 12 seconds post-merge", NEUTRAL),
    ("Crypto venture capital report: $2B deployed in H1 2024", NEUTRAL),
    ("Binance research publishes quarterly crypto ecosystem report", NEUTRAL),
    ("Solana mobile chapter 2 specs announced — developer preview available", NEUTRAL),
    ("Bitcoin HODL wave chart updated — distribution shift within normal range", NEUTRAL),
    ("Ethereum Layer-2 ecosystem overview: 15 chains now live in mainnet", NEUTRAL),
    ("Crypto tax software Koinly integrates with 100 new exchanges", NEUTRAL),
    ("Bitcoin hash rate holds steady — no significant miner movement detected", NEUTRAL),
    ("Uniswap V3 liquidity positions analysis shows normal distribution", NEUTRAL),
    ("Crypto custody regulation framework published by EU financial body", NEUTRAL),
    ("Ethereum validator exit queue clears — normal network conditions restored", NEUTRAL),
    ("Bitcoin on-chain analytics firm Glassnode releases weekly report", NEUTRAL),
    ("Crypto exchange OKX completes proof of reserves audit for Q2", NEUTRAL),
    ("Solana program library updated with new token standard features", NEUTRAL),
    ("Bitcoin transaction count stable — network usage within normal range", NEUTRAL),
    ("DeFi aggregator 1inch integrates three new blockchain networks", NEUTRAL),
    ("Crypto market overview: mixed signals across major asset classes", NEUTRAL),
    ("Ethereum developer call discusses timeline for next EIP implementation", NEUTRAL),
    ("Bitcoin wallet Electrum releases version 4.5 with new privacy features", NEUTRAL),
    ("Crypto exchange Gemini receives regulatory approval in new market", NEUTRAL),
    ("Solana ecosystem report: 500 new dApps launched in Q1 2024", NEUTRAL),
    ("Bitcoin halving countdown: 100 days until next block reward cut", NEUTRAL),
    ("Ethereum gas price oracle updated to improve fee estimation accuracy", NEUTRAL),
    ("Crypto market cap data provider CoinGecko releases methodology update", NEUTRAL),
    ("Binance launchpad announces upcoming token sale for new project", NEUTRAL),
    ("Bitcoin script upgrade Tapscript enables more complex smart contracts", NEUTRAL),
    ("DeFi protocol Compound proposes migration to new governance framework", NEUTRAL),
    ("Solana Foundation annual developer survey results published", NEUTRAL),
    ("Crypto exchange FTX 2.0 relaunch proposal submitted to bankruptcy court", NEUTRAL),
    ("Ethereum testnet Holesky used for Pectra upgrade testing", NEUTRAL),
    ("Bitcoin mining pool Foundry releases updated mining dashboard", NEUTRAL),
    ("Crypto stablecoin market overview: $150B total market cap stable", NEUTRAL),
    ("Solana validator network completes scheduled cluster restart", NEUTRAL),
    ("Bitcoin UTXO set analysis shows normal aging distribution", NEUTRAL),
    ("DeFi protocol Synthetix V3 launches on Optimism mainnet", NEUTRAL),
    ("Crypto exchange trade reporting standards published by industry body", NEUTRAL),
    ("Ethereum client team Prysm releases v5.0 with performance improvements", NEUTRAL),
    ("Bitcoin Core v27.0 released with minor improvements and bug fixes", NEUTRAL),
    ("Crypto market weekly summary: most assets trade sideways", NEUTRAL),
    ("Solana breakpoint conference announces 2024 dates and venue", NEUTRAL),
    ("Ethereum research paper on enshrined PBS published for community review", NEUTRAL),
    ("Crypto exchange Bitfinex completes scheduled system maintenance", NEUTRAL),
    ("Bitcoin analysis: 21M supply cap remains mathematically guaranteed", NEUTRAL),
    ("DeFi insurance protocol Nexus Mutual updates premium calculation model", NEUTRAL),
    ("Crypto analytics firm Nansen publishes annual smart money report", NEUTRAL),
    ("Ethereum blob base fee mechanism functioning as designed per analysis", NEUTRAL),
    ("Bitcoin fee market analysis shows healthy competition for block space", NEUTRAL),
    ("Crypto wallet MetaMask releases new version with improved UI", NEUTRAL),
]


def get_dataset() -> list[tuple[str, int]]:
    """Return the full labeled dataset."""
    return LABELED_DATA


def get_stats() -> dict:
    """Return class distribution statistics."""
    from collections import Counter
    labels = [label for _, label in LABELED_DATA]
    counts = Counter(labels)
    total  = len(LABELED_DATA)
    return {
        "total": total,
        "bullish": counts[BULLISH],
        "neutral": counts[NEUTRAL],
        "bearish": counts[BEARISH],
        "bullish_pct": f"{counts[BULLISH]/total*100:.1f}%",
        "neutral_pct": f"{counts[NEUTRAL]/total*100:.1f}%",
        "bearish_pct": f"{counts[BEARISH]/total*100:.1f}%",
    }


if __name__ == "__main__":
    stats = get_stats()
    print(f"Dataset: {stats['total']} examples")
    print(f"  BULLISH: {stats['bullish']} ({stats['bullish_pct']})")
    print(f"  NEUTRAL: {stats['neutral']} ({stats['neutral_pct']})")
    print(f"  BEARISH: {stats['bearish']} ({stats['bearish_pct']})")
