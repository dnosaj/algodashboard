---
name: Z-Score Probability Indicator
description: Rolling Z-Score with normal distribution probability bands. Measures price extension from mean in standard deviations. Potential entry gate or exhaustion filter.
type: reference
---

# Z-Score Probability Indicator

## Source

- **Author**: steversteves (TradingView, 17.4K followers, 104 scripts)
- **URL**: https://www.tradingview.com/script/zrc6tWT4-Z-Score-Probability-Indicator/
- **Published**: June 3, 2023
- **License**: Open source (Pine Script viewable on TradingView)
- **Engagement**: 105K views, 6.9K comments, 2.4K followers
- **Derivatives**: HMA Z-Score (Erika Barker), Fetch Z-Score (FetchTeam), ThinkOrSwim port (Sam4Cok)

## Discovery Context

Jason found this indicator on TradingView (Mar 14, 2026). Interested in the statistical approach to measuring price extension — conceptually similar to our ADR exhaustion gate but more granular and adaptable.

## What It Does

Measures how far price deviates from its rolling mean in standard deviations (Z-Score), then maps those deviations to cumulative probability levels from the standard normal distribution.

Two components:
1. **Oscillator (lower pane)**: Raw Z-Score value over time, colored green/red, with SMA trend line
2. **Price levels (upper pane)**: Horizontal bands at each SD level (mean +/- 1/2/3 SD) with probability labels

### Math

```
Z = (close - SMA(close, lookback)) / StDev(close, lookback)

price_at_N_sd = SMA(close, lookback) + (N * StDev(close, lookback))
```

Probability labels are hardcoded Z-table values:

| Z-Score | Cumulative Prob | Interpretation |
|---------|----------------|----------------|
| -3 SD   | 0.13%          | Extreme oversold |
| -2 SD   | 2.28%          | Very oversold |
| -1 SD   | 16.0%          | Moderately oversold |
| 0 (mean)| 50.0%          | Fair value |
| +1 SD   | 85.0%          | Moderately overbought |
| +2 SD   | 98.0%          | Very overbought |
| +3 SD   | 99.9%          | Extreme overbought |

### Buy/Sell Signal Logic

```
Buy:  (price crosses above Z-Score MA) AND (Z rising: Z >= highest(avg(Z,5), 5)) AND (Z > 0.5)
Sell: (price crosses below Z-Score MA) AND (Z falling: Z <= lowest(avg(Z,5), 5)) AND (Z < -0.5)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Source | close | Price source |
| Lookback Length | 75 | Bars for SMA and StDev |
| SMA Length | 14 | Smoothing for Z-Score MA line |
| Plot Type | Area | "Area" or "Candles" |
| Show Z-Table | yes | Probability labels |
| SD Bands for SMA | off | Added Nov 2023 |

## Code

### Pine Script (core logic)

```pinescript
//@version=5
// Z-Score Probability Indicator — steversteves (simplified core)

indicator("Z-Score Probability", overlay=false)

source = input.source(close, "Source")
lookback = input.int(75, "Lookback Length")
sma_len = input.int(14, "SMA Length")

// Core Z-Score
cl_sma = ta.sma(source, lookback)
cl_sd = ta.stdev(source, lookback)
z = (source - cl_sma) / cl_sd

// Z-Score SMA (trend line)
z_sma = ta.sma(z, sma_len)

// Price levels at each standard deviation
mean_price = cl_sma
onesd_up = cl_sma + cl_sd
onesd_dn = cl_sma - cl_sd
twosd_up = cl_sma + (2 * cl_sd)
twosd_dn = cl_sma - (2 * cl_sd)
threesd_up = cl_sma + (3 * cl_sd)
threesd_dn = cl_sma - (3 * cl_sd)

// Signal logic
z_rising = z >= ta.highest(ta.sma(z, 5), 5)
z_falling = z <= ta.lowest(ta.sma(z, 5), 5)
buy = ta.crossover(source, cl_sma) and z_rising and z > 0.5
sell = ta.crossunder(source, cl_sma) and z_falling and z < -0.5

// Plot
plot(z, "Z-Score", color=z > 0 ? color.green : color.red)
plot(z_sma, "Z-SMA", color=color.cyan)
hline(0, "Zero", color=color.gray)
hline(2, "+2 SD", color=color.yellow, linestyle=hline.style_dotted)
hline(-2, "-2 SD", color=color.yellow, linestyle=hline.style_dotted)
```

### Python (for backtesting)

```python
import numpy as np

def compute_zscore(closes: np.ndarray, lookback: int = 75) -> np.ndarray:
    """Compute rolling Z-Score of price vs its SMA."""
    n = len(closes)
    z = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        window = closes[i - lookback + 1 : i + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=0)  # population std to match Pine
        if std > 0:
            z[i] = (closes[i] - mean) / std
    return z


def compute_zscore_levels(closes: np.ndarray, lookback: int = 75):
    """Compute price levels at each SD band."""
    n = len(closes)
    mean = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        window = closes[i - lookback + 1 : i + 1]
        mean[i] = np.mean(window)
        sd[i] = np.std(window, ddof=0)
    return mean, sd  # levels = mean + N * sd for N in [-3,-2,-1,0,1,2,3]


def zscore_signals(closes: np.ndarray, lookback: int = 75, sma_len: int = 14,
                   z_buy_thresh: float = 0.5, z_sell_thresh: float = -0.5):
    """Generate buy/sell signals per the indicator's logic."""
    z = compute_zscore(closes, lookback)
    n = len(z)

    # Z-Score SMA
    z_sma = np.full(n, np.nan)
    for i in range(sma_len - 1, n):
        z_sma[i] = np.mean(z[i - sma_len + 1 : i + 1])

    # Z rising/falling (highest/lowest of 5-bar avg over 5 bars)
    z_avg5 = np.full(n, np.nan)
    for i in range(4, n):
        z_avg5[i] = np.mean(z[i - 4 : i + 1])

    z_rising = np.full(n, False)
    z_falling = np.full(n, False)
    for i in range(8, n):
        window = z_avg5[i - 4 : i + 1]
        if not np.any(np.isnan(window)):
            z_rising[i] = z[i] >= np.max(window)
            z_falling[i] = z[i] <= np.min(window)

    # Mean price (SMA)
    mean_price = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        mean_price[i] = np.mean(closes[i - lookback + 1 : i + 1])

    # Cross above/below mean
    buy = np.full(n, False)
    sell = np.full(n, False)
    for i in range(1, n):
        cross_above = closes[i - 1] < mean_price[i - 1] and closes[i] > mean_price[i]
        cross_below = closes[i - 1] > mean_price[i - 1] and closes[i] < mean_price[i]
        if not np.isnan(z[i]):
            buy[i] = cross_above and z_rising[i] and z[i] > z_buy_thresh
            sell[i] = cross_below and z_falling[i] and z[i] < z_sell_thresh

    return z, buy, sell
```

## Analysis

### Useful

- **Z-Score calculation is mathematically sound** — `(x - mean) / stddev` is the standard formula for measuring deviation
- **Adaptable**: Rolling window adjusts to recent volatility regime automatically, unlike fixed-range indicators
- **Conceptually aligned with our ADR gate**: Both answer "is price overextended?" but Z-Score is more granular — continuous value vs binary threshold
- **Cheap to compute**: One SMA + one StDev per bar, O(1) with running accumulators. No overhead concern for live engine

### Flawed

- **Normality assumption is wrong for financial returns.** Fat tails mean 3 SD events happen ~10-20x more often than the stated 0.13%. The probability labels are misleading for real markets.
- **Probabilities are hardcoded Z-table values, not empirically measured.** A rigorous version would compute actual historical frequencies.
- **Rolling window creates non-stationarity.** The Z-Score at bar N is computed against a different distribution than bar N+1. The fixed probability labels don't account for this.
- **No adjustment for autocorrelation.** Financial prices exhibit serial correlation. With 75 consecutive bars, the effective sample size is much smaller, making StDev unreliable.
- **Buy/sell signals are basic MA crossovers** dressed up with statistical language. Nothing novel.
- **75-bar lookback is arbitrary.** On 1-min charts = 75 minutes. On daily = 3.5 months. Behavior varies drastically.

### Related Indicators

- **Bollinger Bands**: Same concept (mean +/- N*SD), different visualization. Z-Score is the normalized version.
- **Keltner Channels**: Mean +/- N*ATR instead of SD. Less sensitive to outliers.
- **RSI**: Both measure "how extended" but RSI uses momentum ratio, Z-Score uses statistical deviation.
- **Our ADR exhaustion gate**: Similar philosophy (block chasing extended moves) but ADR uses daily range %, Z-Score uses rolling SD.

### Correlation with Our Existing Indicators

- **SM (Squeeze Momentum)**: Low expected correlation. SM measures momentum direction/magnitude, Z-Score measures deviation from mean. Complementary axes.
- **RSI**: Moderate expected correlation. Both measure "overbought/oversold" but from different angles (momentum ratio vs statistical deviation). Need to test empirically.
- **ADR gate**: High expected correlation on 1-min data with similar lookback. Both measure "price has moved too far." Z-Score may subsume ADR gate if proven superior.

### Instrument / Timeframe Notes

- Designed for any timeframe. On our 1-min bars with lookback=75, it measures deviation over ~75 minutes of trading.
- For NQ/MNQ: High volatility means SD bands are wide. 2 SD moves happen more often than theoretical 2.28%.
- For MES: Lower point value but similar volatility profile. Should behave similarly.
- **Must test on 1-min data** — most community use is on daily/4H charts.

### Parameter Sensitivity

- **Lookback (75 default)**: Most important param. Shorter = more responsive but noisier. Longer = smoother but slower. Sweep range: [20, 30, 50, 75, 100, 150, 200].
- **Z threshold for gate**: If used as entry gate, the threshold matters. Sweep: [1.0, 1.5, 2.0, 2.5, 3.0].
- **SMA length (14)**: Only matters for the trend line / signal generation. Less important if we use Z-Score as a raw gate value.

### Computational Cost

Minimal. Rolling SMA and StDev are O(1) per bar with running accumulators. Already have SMA infrastructure in the live engine. No concern for per-bar computation.

## Fit With Our System

### Option 1: Z-Score Entry Gate (most natural fit)

Block entries when Z-Score is extreme in the entry direction:
- **Long entry + Z > threshold**: Price already extended above mean — don't chase
- **Short entry + Z < -threshold**: Price already extended below mean — don't chase

This is philosophically identical to the ADR exhaustion gate but uses rolling statistics instead of a fixed daily range percentage. Could potentially replace or complement the ADR gate.

**Test plan**: Pre-filter on SM+RSI entries. Sweep Z thresholds [1.0, 1.5, 2.0, 2.5] and lookbacks [50, 75, 100]. Evaluate as IS/OOS gate like our other filters.

### Option 2: Z-Score Mean-Reversion Entry

Use extreme Z-Score as a contrarian entry signal:
- Z crosses back below +2 from above = short
- Z crosses back above -2 from below = long

This is a different strategy entirely — mean-reversion rather than momentum. Would need its own TP/SL calibration. Higher risk given our system is momentum-based.

### Option 3: Z-Score Exit Enhancement

Use Z-Score to detect when a runner has reached an extreme and should exit:
- vScalpC runner at Z > 2 = take profit (move is extended)
- Could complement structure exit with a statistical overlay

**Recommendation**: Start with Option 1 (entry gate). It's the lowest-risk integration, follows our established gate testing methodology, and directly answers "would Z-Score improve our existing entries?"

## Status

**Untested** — backtest sweep needed against MNQ and MES 1-min data.

## Test Results

*(To be filled after backtesting)*

## Backtest Priority

**Medium** — Sound statistical concept, good philosophical fit with our "don't chase" approach. But we already have the ADR gate covering similar ground. Test to see if Z-Score adds value beyond ADR, or could replace it with a more principled measure.
