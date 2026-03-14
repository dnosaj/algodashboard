---
name: Divergence for Many Indicators
description: Multi-oscillator divergence scanner checking price vs up to 9 built-in oscillators. Potential entry gate to block on bearish divergence.
type: reference
---

# Divergence for Many Indicators

## Source

- **Author**: LonesomeTheBlue (TradingView)
- **Popularity**: ~721 likes, 193 boosts
- **License**: Open source (Pine Script viewable)

## Discovery Context

Surveyed Mar 6, 2026 as part of the TradingView open-source indicator research. Rated LOW priority. The survey noted we already use RSI and that divergence-type filters have mixed OOS results in our prior research.

## What It Does

Scans for divergences between price and multiple oscillators simultaneously:

1. Detects pivot highs/lows on both price and oscillator
2. Compares: price making higher high but oscillator making lower high = **bearish regular divergence** (weakening momentum)
3. Also detects hidden divergences (trend continuation signals)
4. Checks across up to 9 built-in oscillators + 1 external

### Supported Oscillators
RSI, MACD, MACD Histogram, Stochastic, CCI, Momentum, OBV, VWMACD, CMF

### Signal Logic
- Scan last N pivot points (default 16)
- If divergence detected on `min_divergences` or more oscillators simultaneously, flag it
- Can require consensus across multiple oscillators for higher-confidence signals

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Pivot Period | (varies) | Lookback for pivot detection |
| Source | Close or H/L | Price reference for pivots |
| Max Pivots to Check | 16 | Number of historical pivots to scan |
| Max Bars to Check | (varies) | Lookback window in bars |
| Min # Divergences | (varies) | Required oscillator agreement count |
| Show Hidden Divergences | off | Trend continuation divergences |

## Code

No Pine Script extracted during survey. The full indicator is viewable on TradingView. Python implementation would require pivot detection + divergence comparison logic for each oscillator.

## Analysis

### Useful

- **Multi-oscillator consensus is theoretically stronger** than single-oscillator divergence. If RSI, CCI, and Stochastic all show bearish divergence, the signal has more weight.
- **Covers oscillators we don't currently use** (CCI, Stochastic, OBV, CMF). Could capture divergences RSI misses.

### Flawed

- **We already use RSI as our primary oscillator.** Adding CCI or Stochastic divergence adds complexity for diminishing returns.
- **Divergence detection requires pivot identification**, which is inherently lagging (need N bars after pivot to confirm it).
- **Our prior research shows divergence-type filters have mixed OOS results.** Rally exhaustion (conceptually related to divergence) was marginal at best for MES v2.
- **On 1-min bars, divergences are extremely noisy.** Most community use of this indicator is on 4H/daily charts.
- **Combined/stacked filters always fail in our system** due to geometric trade count reduction. Multi-oscillator consensus is just another form of stacking.

### Related Indicators

- **RSI (our existing filter)**: Already captures momentum extremes. Divergence adds the "momentum weakening while price advances" dimension.
- **Leledc Exhaustion (adopted)**: Different approach to the same problem -- counts consecutive directional bars rather than measuring oscillator divergence.
- **CCI Divergence Detector (sizzlinsoft)**: CCI-only variant. Narrower scope.

### Correlation with Our Existing Indicators

- **RSI**: High correlation for RSI-based divergence signals (same base oscillator). CCI/Stochastic divergences would have moderate correlation with RSI divergence.
- **SM**: Low-moderate correlation. SM measures trend strength, divergence measures trend weakening. They are conceptually complementary but divergence signals tend to fire before SM flips.

## Fit With Our System

### Entry Gate
Block entries when multiple oscillators show bearish divergence against entry direction.

Problem: This is complex to implement (pivot detection on multiple oscillators), likely to overlap significantly with our existing RSI filter, and our research consistently shows that stacking oscillator-based filters degrades performance due to trade count loss.

## Status

**Untested** -- not backtested. Not mentioned in the decision log. Low priority per the original survey.

## Test Results

*(None)*

## Backtest Priority

**Low** -- Three factors argue against testing: (1) We already have RSI. (2) Our research shows adding more oscillators has diminishing returns. (3) Stacked/combined filters consistently fail in our system. Only worth revisiting if a specific failure mode is identified where RSI misses divergences that other oscillators catch.
