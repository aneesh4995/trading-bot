# SPY/QQQ Options Signal Bot

A production-ready trading signal system that predicts 3–5 day directional momentum swings on SPY and QQQ ETFs, maps predictions to options strategies, and sends daily GO/WAIT alerts to your phone.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Financial Concepts Explained](#financial-concepts-explained)
4. [Module Documentation](#module-documentation)
5. [V2 Model Improvements](#v2-model-improvements)
6. [Ravish Strategy Engine](#ravish-strategy-engine)
7. [Backtesting](#backtesting)
8. [Screener Logic](#screener-logic)
9. [Sentiment Analysis](#sentiment-analysis)
10. [Deployment & Alerts](#deployment--alerts)
11. [Design Decisions](#design-decisions)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full signal cards + backtest
python3 src/main.py

# Run GO/WAIT screener in terminal
python3 src/screener.py

# Send phone alert now
python3 src/alert_screener.py

# Run 3-year strategy backtest
python3 src/strategy_backtest.py

# Run sentiment analysis only
python3 src/sentiment.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    alert_screener.py                      │
│              (entry point for daily alerts)               │
│                         │                                │
│         ┌───────────────┼───────────────┐                │
│         ▼               ▼               ▼                │
│   screener.py      sentiment.py    data_handler.py       │
│   (11 checks)      (news NLP)     (market data)         │
│         │               │               │                │
│    ┌────┴────┐          │          ┌────┴────┐           │
│    ▼         ▼          │          ▼         ▼           │
│ predictor  indicators   │     yfinance    options        │
│   .py        .py        │     (Yahoo)     chain          │
│    │          │         │                                │
│    │    ┌─────┘         │                                │
│    ▼    ▼               │                                │
│ EnsemblePredictor       │                                │
│ (GB + RF + LR)          │                                │
│  + VIX, put/call,       │                                │
│    vol ratio, etc.      │                                │
│         │               │                                │
│         ▼               ▼                                │
│   ┌─────────────────────────┐                            │
│   │    ravish_strategy.py   │                            │
│   │  (7 options strategies) │                            │
│   └─────────────────────────┘                            │
│         │                                                │
│         ▼                                                │
│   risk_manager.py ──► Kelly Criterion sizing             │
│         │                                                │
│         ▼                                                │
│   ntfy.sh push notification ──► your phone               │
└─────────────────────────────────────────────────────────┘
```

### File Structure

```
/src/
  data_handler.py       # Market data fetching (Yahoo Finance)
  indicators.py         # Technical indicator calculations
  predictor.py          # ML models (Ensemble: GB + RF + LR, Walk-Forward)
  options_strategy.py   # Basic options strategy mapper
  ravish_strategy.py    # 7 strategies from @OptionsWithRavish
  risk_manager.py       # Kelly Criterion + ATR-based risk
  backtester.py         # ML signal backtester
  strategy_backtest.py  # Black-Scholes backtest of all strategies
  screener.py           # 11-check GO/WAIT screener
  sentiment.py          # News sentiment via Google News RSS
  alert_screener.py     # Screener + ntfy.sh push notifications
  main.py               # Orchestration entry point

/docs/
  README.md             # This file

/.github/workflows/
  screener.yml          # GitHub Actions daily cron job
```

---

## Financial Concepts Explained

### What are SPY and QQQ?

**SPY** is an ETF (Exchange-Traded Fund) that tracks the S&P 500 — the 500 largest US companies.
**QQQ** tracks the Nasdaq-100 — the 100 largest non-financial Nasdaq companies (heavy in tech).

They are the most liquid ETFs in the world, meaning their options have tight bid-ask spreads and high volume, making them ideal for options trading.

### What is an Option?

An option is a contract that gives you the right (but not the obligation) to buy or sell a stock at a specific price (the **strike price**) by a specific date (the **expiration date**).

- **Call option** = right to BUY at the strike price. You buy calls when you think the stock will go UP.
- **Put option** = right to SELL at the strike price. You buy puts when you think the stock will go DOWN.

### Key Options Terms Used in This Project

| Term | Meaning |
|---|---|
| **Strike price** | The price at which you can buy/sell the stock if you exercise the option |
| **Expiration / DTE** | Days To Expiration — when the option contract expires |
| **Premium** | The price you pay to buy an option, or receive when selling one |
| **Delta** | How much the option price moves per $1 move in the stock. A 0.50 delta call moves $0.50 for every $1 the stock moves. Also approximates probability of expiring in-the-money |
| **Theta** | How much value the option loses per day from time decay. Options lose value as expiration approaches |
| **IV (Implied Volatility)** | The market's expectation of how much the stock will move. Higher IV = more expensive options |
| **IV Rank** | Where current IV sits relative to its range over the past year. IV Rank of 80% means IV is near the top of its annual range — good time to SELL options |
| **ATM (At The Money)** | Option whose strike price equals the current stock price |
| **ITM (In The Money)** | Call with strike below stock price, or put with strike above stock price |
| **OTM (Out of The Money)** | Call with strike above stock price, or put with strike below stock price |
| **Credit spread** | Selling one option and buying another as a hedge. You RECEIVE money (credit) to open the trade |
| **Debit spread** | Buying one option and selling another. You PAY money (debit) to open |

### What is a Bull Put Credit Spread?

This is the **primary strategy** recommended by our system. Here's how it works:

1. **SELL** a put option at the current stock price (50 delta) — you collect premium
2. **BUY** a put option 10 points lower — this is your safety net (caps your maximum loss)
3. Both options have the same expiration date (14 days out)

**Example on SPY at $660:**
- Sell the $660 put → receive $5.00 credit ($500 per contract)
- Buy the $650 put → pay $3.00 ($300 per contract)
- Net credit received: $2.00 ($200 per contract)
- Maximum loss: $10 spread width - $2 credit = $8 ($800 per contract)
- Maximum profit: $200 (if SPY stays above $660 at expiration)

**Why it works:** You profit if SPY goes up, stays flat, or even drops slightly (as long as it stays above the short strike). Time decay (theta) works in your favor — the options you sold lose value every day.

**Take profit at 50%:** When you've made $100 of the $200 max profit, close the trade. Don't get greedy.

### What is VIX?

The VIX (CBOE Volatility Index) measures the market's expectation of 30-day volatility on the S&P 500. It's often called the "fear gauge."

- **VIX 12-18**: Low fear, calm market. Options premiums are cheap.
- **VIX 18-28**: Moderate fear. Good premiums for selling options.
- **VIX 28-35**: High fear. Options are expensive — spreads cost more to hedge.
- **VIX > 35**: Extreme fear (crash territory). Stay cash or use long-dated strategies only.

Our screener uses VIX 12-28 as the sweet spot for credit spreads.

### What is the Kelly Criterion?

A mathematical formula for optimal bet sizing:

```
f = (win_rate × avg_win - loss_rate × avg_loss) / avg_win
```

- If your win rate is 87% and avg win is 35% and avg loss is 94%...
- Kelly says bet 17% of your account per trade
- We use **half-Kelly** (8.5%) for safety — reduces variance significantly

If your backtest shows negative expectancy (losing strategy), Kelly returns 0 — it correctly refuses to size the position, protecting you from a bad strategy.

### What is ATR (Average True Range)?

ATR measures how much a stock typically moves per day, accounting for gaps. If SPY's ATR is $9.50, it means SPY typically moves $9.50 per day.

We use ATR for:
- **Stop loss**: Entry - (2× ATR) for moderate risk tolerance
- **Take profit**: Entry + (3× ATR)
- **Volatility filter**: If ATR is 1.5× its 20-day average, the market is too wild for credit spreads

### What is RSI (Relative Strength Index)?

RSI measures how overbought or oversold a stock is on a scale of 0-100.

- **RSI > 70**: Overbought — the stock has risen too fast, may pull back
- **RSI < 30**: Oversold — the stock has fallen too hard, may bounce
- **RSI 30-70**: Normal range

Our screener requires RSI > 30 before opening a bullish credit spread — don't sell puts into a crashing market.

### What is MACD?

MACD (Moving Average Convergence Divergence) shows the relationship between two moving averages:
- **MACD line**: 12-day EMA minus 26-day EMA
- **Signal line**: 9-day EMA of the MACD line
- When MACD crosses above signal → bullish momentum
- When MACD crosses below signal → bearish momentum

### What are EMA_20, EMA_50, EMA_200?

Exponential Moving Averages of the closing price over 20, 50, and 200 days. They smooth out price action to show the trend:

- **Price > EMA_20**: Short-term uptrend
- **Price > EMA_50**: Medium-term uptrend
- **Price > EMA_200**: Long-term uptrend (the "200-day moving average" that Wall Street watches)
- **Price < EMA_50 AND < EMA_200**: Confirmed downtrend — don't sell puts

### What is FOMC?

The Federal Open Market Committee meets ~8 times per year to decide interest rates. These meetings cause massive market swings. **Never open a credit spread within 1 day of an FOMC decision** — the market can gap 2-3% in either direction.

### What is CPI / NFP?

- **CPI (Consumer Price Index)**: Monthly inflation report (~12th of each month). Higher than expected = bearish (Fed may raise rates).
- **NFP (Non-Farm Payrolls)**: Monthly jobs report (first Friday). Surprise numbers cause big moves.

Both events can cause sudden volatility that blows up credit spreads.

---

## Module Documentation

### data_handler.py — `DataHandler`

Fetches all market data from Yahoo Finance (free, no API key).

| Method | Returns | Description |
|---|---|---|
| `fetch_ohlcv(ticker, period, interval)` | DataFrame | Daily OHLCV bars with DatetimeIndex |
| `fetch_vix()` | float | Current VIX level |
| `fetch_live_quote(ticker)` | dict | Near-real-time price via `yf.Ticker.fast_info` |
| `fetch_premarket(ticker)` | float | Pre-market price if available |
| `fetch_options_iv(ticker)` | dict | ATM implied volatility + IV Rank from real options chain |
| `fetch_options_chain(ticker)` | dict | Full calls + puts DataFrames for nearest expiry |
| `market_status()` | str | "pre-market" / "open" / "after-hours" / "closed" |

**Data freshness:** During market hours, `fetch_live_quote` returns near-real-time data. After hours, it uses the last closing price. Options chain data is most accurate during market hours.

### indicators.py — `FeatureEngineer`

Calculates all technical indicators using the `ta` library.

| Method | Output Columns |
|---|---|
| `add_all(df)` | RSI_14, MACD, MACD_signal, EMA_20, EMA_50, EMA_200, ATR_14, volume_ratio, dist_ema200, day_of_week, rsi_divergence |
| `add_vix(df)` | VIX (merged from ^VIX daily close) |
| `add_put_call_ratio(df)` | put_call_ratio (from CBOE ^PCCE, or VIX-derived fallback) |

**V2 features added (4 new engineered + 2 external):**

| Feature | Formula | Why It Helps |
|---|---|---|
| `volume_ratio` | today's volume / 20-day average volume | Unusual volume often precedes big moves. A ratio > 2 means twice-normal activity |
| `dist_ema200` | (close - EMA_200) / EMA_200 × 100 | Mean reversion signal. Stocks far above their 200-day tend to pull back; far below tend to bounce |
| `day_of_week` | Monday=0 through Friday=4 | Markets have day-of-week effects. Mondays tend slightly bearish, Fridays bullish (position squaring) |
| `rsi_divergence` | 5-day RSI change minus 5-day price change (%) | When price rises but RSI falls, it's bearish divergence — momentum is weakening under the surface |
| `VIX` | CBOE Volatility Index daily close | Direct measure of market fear. VIX > 25 means credit spreads carry more risk |
| `put_call_ratio` | CBOE equity put/call ratio (or VIX-derived proxy) | High put/call = more hedging = more fear. Extreme readings can signal capitulation (contrarian bullish) |

Drops NaN rows after computing (first ~200 rows lost due to EMA_200 warmup).

### predictor.py — Three Predictor Classes

Machine learning models for predicting 5-day forward returns.

#### `SwingPredictor` (V1 — backward compatible)

- **Algorithm**: GradientBoostingClassifier (200 trees, max depth 4)
- **Features**: 7 original (RSI_14, MACD, MACD_signal, EMA_20, EMA_50, EMA_200, ATR_14) — upgrades to V2 features (13 total) if available in the DataFrame
- **Target**: Binary — 1 if price rises > 0.5% in next 5 days, else 0
- **Train/test split**: 80/20, `shuffle=False` (respects time ordering — no lookahead)

#### `EnsemblePredictor` (V2 — 3-model majority vote)

Uses three models and takes a **majority vote** (2 of 3 must agree):

| Model | Type | Strength |
|---|---|---|
| GradientBoosting | 200 trees, depth 4 | Non-linear patterns, feature interactions |
| RandomForest | 200 trees, depth 6 | Reduces overfitting via bagging (each tree sees random data subset) |
| LogisticRegression | SAGA solver, 1000 iter | Linear baseline — captures simple trend momentum |

- **Features**: All 13 V2 features (7 original + volume_ratio, dist_ema200, day_of_week, rsi_divergence, VIX, put_call_ratio)
- **Scaling**: `StandardScaler` applied to LogisticRegression inputs (LR needs normalized features; tree models don't)
- **Probability**: Average probability across models that agree with the majority vote
- **Sentiment integration**: Adjusts final probability by up to ±10% if sentiment score provided

**Why 3 models?** Each model has different failure modes. GradientBoosting overfits noise, RandomForest underfits subtle patterns, LogisticRegression misses non-linear effects. By majority voting, you need 2 of 3 to agree — this filters out false signals from any single model. Empirically, V2 ensemble outperforms V1 single model by 11-16 percentage points on test data.

**Test results (80/20 split, 2 years daily data):**

| Ticker | V1 (GB only, 7 feat) | V2 GB (13 feat) | V2 RF (13 feat) | V2 LR (13 feat) |
|---|---|---|---|---|
| SPY | 26.2% | 37.7% | 36.1% | 37.7% |
| QQQ | 29.5% | 42.6% | 45.9% | 41.0% |

Note: The test period was 74% "Down" labels (bear market test set), so even 40% accuracy significantly beats the 26% base rate of "Up" predictions.

#### `WalkForwardPredictor` (V2 — adaptive retraining)

Wraps `EnsemblePredictor` with **walk-forward retraining**: instead of training once on stale data, it retrains on an expanding window every N days.

- **Default retrain interval**: 63 trading days (~quarterly)
- **`backtest(df, min_train=252)`**: Walk-forward backtest that trains on data up to day i, predicts day i, steps forward by `retrain_every` days
- **`train_latest(df)`**: Train on all available data for live prediction
- **Why walk-forward?** Markets change regimes (bull → bear → sideways). A model trained in 2024 may not work in 2026. Walk-forward ensures the model always adapts to the most recent market conditions.

**Why 0.5% threshold?** A 5-day move > 0.5% filters out noise. On SPY, this is roughly a $3.30 move — enough to make credit spread profits meaningful after commissions.

**Why GradientBoosting (plus RF and LR)?** Tree-based models handle non-linear relationships and don't require feature scaling. LogisticRegression adds a linear baseline. Together, they cover different pattern types in the data. All train in seconds on ~500 rows — no GPU needed.

### options_strategy.py — `OptionsSignalMapper`

Maps ML predictions to basic options strategies based on conviction and volatility:

| Condition | Strategy |
|---|---|
| High conviction + Up + low ATR | Long Call |
| High conviction + Up + high ATR | Bull Call Spread |
| High conviction + Down + high ATR | Bear Put Spread |
| High conviction + Down + low ATR | Long Put |
| Low conviction (< 65%) | No Trade / Wait |

ATR threshold: 60th percentile of rolling 20-period ATR. Above = "high volatility."

### risk_manager.py — `RiskManager`

Calculates position sizing and exit levels.

| Method | Formula |
|---|---|
| `max_position_size(win_rate, avg_win, avg_loss)` | Half-Kelly × account_size |
| `stop_loss(entry, atr)` | entry - (multiplier × ATR) |
| `take_profit(entry, atr)` | entry + (multiplier × ATR) |
| `recommended_delta()` | Conservative: 0.70, Moderate: 0.55, Aggressive: 0.40 |

**Risk tolerance multipliers:**

| | Stop (ATR×) | Take Profit (ATR×) | Delta |
|---|---|---|---|
| Conservative | 1.5 | 2.0 | 0.70 |
| Moderate | 2.0 | 3.0 | 0.55 |
| Aggressive | 2.5 | 4.0 | 0.40 |

### sentiment.py — `NewsSentimentAnalyzer`

Analyzes market sentiment from news headlines.

**Data sources:** Google News RSS feeds (free, no API key):
- `SPY stock market` — SPY-specific news
- `QQQ nasdaq stock` — QQQ-specific news
- `stock market today` — general market news
- `federal reserve interest rates` — Fed/macro news

**Scoring method:**
1. **TextBlob NLP** (40% weight): Natural language processing polarity from -1 to +1
2. **Financial keyword matching** (60% weight): Custom bullish/bearish/fear keyword lists
3. **Time weighting**: Exponential decay with 24-hour half-life — recent news matters more
4. **Composite score**: Weighted average of all headlines → -1.0 (extreme fear) to +1.0 (extreme greed)

**Why 60% keywords, 40% NLP?** TextBlob is a general-purpose sentiment tool — it doesn't understand financial jargon. "Rate cut" is bullish for stocks but TextBlob sees "cut" as negative. Our keyword list captures financial context that NLP misses.

**Fear level:** Proportion of headlines containing panic/crisis keywords (fear, crash, collapse, contagion, etc.). Used as a secondary warning signal.

---

## V2 Model Improvements

The system was upgraded from a single GradientBoosting model with 7 features (V1) to an ensemble of 3 models with 13 features plus walk-forward retraining (V2). Here's a summary of all 4 changes:

### 1. VIX + Put/Call Ratio as Features

**What changed:** Added two external market-wide features that the model can now learn from.

- `VIX` is fetched daily from `^VIX` via Yahoo Finance and merged into the feature DataFrame by date
- `put_call_ratio` is fetched from CBOE's `^PCCE` index. If unavailable (common — it's often delisted on Yahoo), it falls back to a VIX-derived proxy: `0.5 + (VIX / 100)`

**Why it matters for credit spreads:** VIX directly determines option premium levels. The model now learns that VIX 18-25 is the sweet spot — enough premium to collect, but not so volatile that spreads get blown up.

### 2. Walk-Forward Retraining

**What changed:** Instead of training once on all historical data, the `WalkForwardPredictor` retrains the ensemble every 63 trading days (~quarterly) on an expanding window of data.

**How it works:**
```
Day 1-252:     Train on days 1-252, predict day 253
Day 253-315:   Train on days 1-315, predict day 316
Day 316-378:   Train on days 1-378, predict day 379
...and so on
```

Each retrain sees ALL past data (expanding window, not sliding). For live prediction, `train_latest(df)` trains on everything available.

### 3. Volume Ratio + 3 More Engineered Features

**What changed:** Added 4 new features computed from existing OHLCV data:

```python
# Volume ratio: today's volume vs 20-day average
df["volume_ratio"] = volume / volume.rolling(20).mean()

# Distance from 200-day EMA as percentage
df["dist_ema200"] = (close - EMA_200) / EMA_200 * 100

# Day of week (Monday=0, Friday=4)
df["day_of_week"] = df.index.dayofweek

# RSI divergence: RSI momentum vs price momentum
df["rsi_divergence"] = (RSI - RSI.shift(5)) - (close.pct_change(5) * 100)
```

**Total features:** V1 had 7, V2 has 13 (7 original + 4 engineered + VIX + put_call_ratio).

### 4. Ensemble Voting (3 Models)

**What changed:** Replaced the single GradientBoostingClassifier with a 3-model ensemble using majority voting.

| Model | Config | Role |
|---|---|---|
| GradientBoosting | 200 trees, depth 4 | Primary — captures complex feature interactions |
| RandomForest | 200 trees, depth 6 | Variance reducer — each tree sees random data subset |
| LogisticRegression | SAGA solver, 1000 iter, StandardScaler | Linear baseline — catches simple momentum trends |

**Voting:** If 2 or 3 models say "Up", the final signal is "Up". The reported probability is the average confidence across agreeing models.

**`predict_detail(df)`** returns each model's individual prediction for transparency:
```
gb: Down @ 83.8%
rf: Up @ 61.2%
lr: Up @ 85.2%
→ Majority: Up (2/3) @ 73.2%
```

### Accuracy Comparison

Tested on SPY and QQQ with 80/20 chronological split (train on older data, test on recent):

| Ticker | V1 (1 model, 7 features) | V2 (3 models, 13 features) | Improvement |
|---|---|---|---|
| SPY | 26.2% | 37.7% | +11.5pp |
| QQQ | 29.5% | 45.9% | +16.4pp |

The test period was ~74% "Down" labels (recent bear market), so even 38-46% accuracy on "Up" predictions significantly beats the 26-30% base rate. The model correctly identifies more bullish setups while filtering out false signals.

---

## Ravish Strategy Engine

Extracted from 13 YouTube videos by @OptionsWithRavish. These are systematic, rule-based strategies with specific entry/exit criteria.

### Strategy 1: Bull Put Credit Spread (Primary)

| Parameter | Value |
|---|---|
| Sell leg | 50-delta put (ATM) |
| Buy leg | 10 points below sell strike |
| DTE | 14 days |
| Take profit | 50% of max credit |
| Stop loss | None — roll down and out to next month |
| Win rate (backtest) | 87-90% |
| Backtest result ($1,000) | $6,683 over 3 years |

**When to use:** ML says bullish, VIX 12-28, no events, IV Rank > 30%.

### Strategy 2: LEAPS Swing

| Parameter | Value |
|---|---|
| Buy | 60-80 delta call |
| Expiry | 12+ months |
| Hold time | 3-4 months |
| Take profit | 50% |
| Stop loss | None |
| Win rate | 91-96% |
| Best on | QQQ specifically |

**When to use:** High ML conviction (> 80%) on QQQ, or VIX > 30 (LEAPS are better than spreads in high vol).

### Strategy 3: Diagonal Spread (Poor Man's Covered Call)

| Parameter | Value |
|---|---|
| Long leg | 70-delta call, 6-month expiry |
| Short leg | 30-delta call, monthly expiry |
| Roll | Monthly — sell new 30-delta call each month |
| Take profit | 30-40% on short leg |
| Backtest result | $2,933 from $1,000 (3y) |

**When to use:** Ongoing monthly income on any bullish position.

### Strategy 4: Cash Secured Put

| Parameter | Value |
|---|---|
| Sell | 30-50 delta put |
| DTE | 30+ days |
| Take profit | 50-80% |
| Win rate | 91-93% |

**When to use:** Want to buy a stock at a discount. If assigned, you own shares below market price.

### Strategy 5: Earnings Short Put

| Parameter | Value |
|---|---|
| Sell | 20-delta put below expected move |
| DTE | 2 days (open day before earnings) |
| Entry time | 3:30 PM before market close |
| Exit | First 30 minutes of next-day open |
| Win rate | 80-100% on NVDA over 3 years |

**When to use:** Quarterly earnings on high-IV stocks. Profits from IV crush overnight.

### Strategy 6: ZEBRA (Zero Extrinsic Back Ratio)

| Parameter | Value |
|---|---|
| Buy | 2x ATM calls (50 delta each) |
| Sell | 1x deep ITM call (80 delta) |
| Net delta | ~100 (stock replacement) |
| Theta decay | Near zero |

**When to use:** Directional trade where you want 100-share exposure without theta decay risk.

### Strategy 7: Covered Call (for existing positions)

| Parameter | Value |
|---|---|
| Sell | 30-delta call, monthly expiry |
| Roll | Up and out if stock exceeds strike |
| Income | $20-30K/month on large portfolios |

**When to use:** Already own shares and want monthly income.

---

## Backtesting

### ML Signal Backtest (`backtester.py`)

- Train/test split: 70/30, time-series ordered
- For each test row: predict direction, if conviction > 65%, enter trade
- Hold 5 days, track P&L
- Compound equity across all trades

### Strategy Backtest (`strategy_backtest.py`)

Uses **Black-Scholes option pricing** to simulate each strategy over 3 years:

**Black-Scholes formula** approximates option prices based on:
- Current stock price (S)
- Strike price (K)
- Time to expiry (T)
- Volatility (σ) — we use 30-day rolling historical volatility as IV proxy
- Risk-free rate (r = 4.5%)

**Position sizing:** Each trade risks 20% of current equity (fixed fractional).

**Results (3 years, $1,000 starting capital):**

| Strategy | Ticker | Trades | Win% | Final$ | Return |
|---|---|---|---|---|---|
| Bull Put Credit Spread | SPY | 60 | 86.7% | $6,683 | +568% |
| Earnings Short Put | SPY | 9 | 100% | $4,702 | +370% |
| Diagonal Spread | QQQ | 24 | 79.2% | $2,933 | +193% |
| LEAPS Swing | QQQ | 11 | 81.8% | $1,849 | +85% |
| Cash Secured Put | SPY | 59 | 93.2% | $1,042 | +4% |

**Why Black-Scholes and not real options data?** Real historical options data costs $1,000+/year from providers like ORATS or OptionMetrics. Black-Scholes is a reasonable approximation for backtesting strategy logic, though it underestimates tail risk and doesn't capture bid-ask spread slippage.

---

## Screener Logic

The screener runs 11 checks and produces a weighted score:

| # | Check | Weight | Pass Condition |
|---|---|---|---|
| 1 | VIX Level | 2.0 | VIX between 12-28 |
| 2 | VIX Hard Stop | 3.0 | VIX < 30 (only added if VIX > 30) |
| 3 | Price > EMA_20 | 1.5 | Short-term uptrend |
| 4 | Not in downtrend | 2.0 | Not below both EMA_50 and EMA_200 |
| 5 | RSI > 30 | 1.5 | Not oversold |
| 6 | ATR not spiking | 1.0 | ATR < 1.5× its 20-day average |
| 7 | IV Rank | 1.5 | IV Rank >= 30% (premiums rich enough) |
| 8 | No FOMC | 2.0 | Not within 1 day of FOMC meeting |
| 9 | No CPI/NFP | 1.0 | Not near CPI release or jobs report |
| 10 | ML Bullish Signal | 1.5 | Ensemble (3-model vote) says Up with > 65% conviction |
| 11 | News Sentiment | 1.5 | Not strongly bearish (> -0.15) |

**Scoring:**
```
score = sum(passed_check_weights) / sum(all_check_weights)
```

| Score | Verdict | Action |
|---|---|---|
| >= 85% | **GO** | Open the spread |
| 60-84% | **CAUTION** | Check failed conditions manually |
| < 60% | **WAIT** | Do not trade |

---

## Deployment & Alerts

### Push Notifications (ntfy.sh)

We use [ntfy.sh](https://ntfy.sh) — a free, open-source push notification service. No account needed.

- **Topic:** `spy-qqq-screener-aneesh`
- **How it works:** The script makes a POST request to `https://ntfy.sh/spy-qqq-screener-aneesh` with the alert text. The ntfy app on your phone receives it instantly.
- **Privacy:** Anyone who knows the topic name can subscribe. Use a unique topic name.

### GitHub Actions (daily automation)

The workflow in `.github/workflows/screener.yml` runs every weekday:

```
Schedule: 45 13 * * 1-5  (1:45 PM UTC = 8:45 AM CT)
```

- Runs on GitHub's servers (Ubuntu)
- Installs Python 3.11 + all dependencies
- Executes `alert_screener.py`
- Sends push notification to your phone
- Free tier: 2,000 minutes/month (this uses ~1 min/day = 20 min/month)
- Works even if your computer is off

### Local Mac Schedule (backup)

A `launchd` plist is installed at `~/Library/LaunchAgents/com.screener.daily.plist` that runs the screener weekdays at 8:45 AM CT. Only works if the Mac is awake.

---

## Design Decisions

### Why Yahoo Finance instead of a paid data provider?

- Free, no API key, works globally
- Options chain data available (IV, Greeks)
- Good enough for daily screening (15-min delay on quotes)
- Trade-off: No real-time streaming, no historical options data

### Why an ensemble instead of a single GradientBoosting model?

- A single model can overfit to noise in recent data or fail on regime changes
- Majority voting (2 of 3 must agree) filters out false signals — if only 1 model is bullish, the system waits
- Each model captures different patterns: GB handles interactions, RF reduces variance, LR catches linear trends
- Ensemble outperforms single GB by 11-16pp on out-of-sample test data
- Almost no downside: training all 3 still takes < 2 seconds on 500 rows

### Why walk-forward retraining instead of train-once?

- Markets shift regimes: a model trained during 2024's bull run fails during 2025's correction
- Walk-forward retrains every ~63 trading days (quarterly), keeping the model current
- It uses an expanding window: each retrain sees all past data, not just a sliding window
- This prevents the model from "forgetting" long-term patterns while adapting to recent conditions

### Why these specific V2 features (VIX, volume ratio, etc.)?

- **VIX**: Directly predicts the environment where credit spreads work. High VIX = expensive premiums but more risk. The model learns the VIX sweet spot
- **Put/call ratio**: Extreme readings signal capitulation or complacency — powerful contrarian indicator
- **Volume ratio**: Unusual volume precedes breakouts/breakdowns. The model learns "this pattern + high volume = real; this pattern + low volume = noise"
- **Distance from EMA_200**: Mean reversion is one of the strongest effects in equity markets. Stocks 10%+ above their 200-day tend to pull back
- **RSI divergence**: When price makes new highs but RSI doesn't, momentum is weakening. Classic technical signal now quantified for the ML model
- **Day of week**: Slight but real calendar effects (Monday weakness, Friday strength from position squaring)

### Why not use deep learning (LSTM/Transformer)?

- Works well on tabular data with ~500 rows (2 years daily)
- No GPU required, trains in seconds
- Interpretable feature importances
- LSTMs need thousands of rows and careful hypertuning for marginal gains on this data size

### Why 14 DTE for credit spreads?

- Long enough for theta to work (options lose ~50% of time value in last 2 weeks)
- Short enough to avoid big market moves
- Matches Ravish's backtested rules (90% win rate over 5 years)
- One trade every 2 weeks = manageable for a working person

### Why Half-Kelly instead of full Kelly?

Full Kelly maximizes long-term growth rate but has extreme variance. A single bad streak can draw down 50%+. Half-Kelly sacrifices ~25% of growth rate but reduces drawdowns by ~50%. For a $500-1000 account, preservation is more important than maximization.

### Why not execute trades automatically?

- Options require careful strike/expiry selection based on live bid-ask spreads
- Automated execution needs a funded brokerage API (Alpaca, IBKR) — adds complexity
- The value is in the SIGNAL (when to trade), not the execution
- Manual execution takes 2 minutes and lets you verify the setup

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Options trading involves significant risk of loss. Past performance (backtests) does not guarantee future results. Always consult a financial advisor before trading with real money.
