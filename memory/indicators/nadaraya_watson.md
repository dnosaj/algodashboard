---
name: Nadaraya-Watson Envelope
description: Kernel smoothing-based price envelope using non-parametric regression. Rejected due to repainting in default mode.
type: reference
---

# Nadaraya-Watson Envelope

## Source

- **Author**: LuxAlgo (TradingView)
- **Popularity**: ~30K likes, 1.15M views
- **License**: Open source (Pine Script viewable)

## Discovery Context

Surveyed Mar 6, 2026 as part of the TradingView open-source indicator research. Rated LOW priority due to repainting concerns. The survey concluded that the non-repainting mode is "essentially a smoothed moving average envelope -- not much different from Bollinger Bands."

## What It Does

Estimates the underlying trend via Nadaraya-Watson kernel regression (a non-parametric statistical method), then constructs an envelope around it using mean absolute deviation.

### Math

Nadaraya-Watson estimator:
```
Y_hat(x) = sum(K(x - x_i) * y_i) / sum(K(x - x_i))
```
where K is a kernel function (typically Gaussian or Epanechnikov) and the bandwidth controls smoothness.

Envelope:
```
Upper = Y_hat + Mult * MAD
Lower = Y_hat - Mult * MAD
```

Signals: Price touching upper envelope = overbought, lower envelope = oversold.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Bandwidth | (varies) | Controls smoothness of the kernel regression |
| Mult | (varies) | Envelope width as multiple of MAD |
| Source | close | Price input |
| Repainting | on (default) | Toggle for repaint vs non-repaint mode |

## Code

No Pine Script extracted during survey. The full indicator is viewable on TradingView. The core kernel regression is standard but computationally heavier than simple moving averages (O(bandwidth) per bar vs O(1) for EMAs).

## Analysis

### Useful

- **Mathematically elegant.** Kernel regression is a well-established non-parametric method. Adapts to data shape without assuming a functional form.
- **Envelope provides natural overbought/oversold levels** based on the fitted curve, not arbitrary standard deviations.

### Flawed

- **Default mode REPAINTS.** This is the critical problem. The kernel regression uses future data in its calculation, so historical signals change as new bars arrive. Any backtest using the default mode produces unrealistically good results.
- **Non-repainting mode is just a lagged smoother.** When you constrain Nadaraya-Watson to only use past data, it degenerates into a weighted moving average with a kernel-shaped weight function. This is functionally similar to Bollinger Bands (mean + SD envelope) with slightly different smoothing.
- **Computational cost.** Full kernel regression is O(N * bandwidth) per bar. On 1-min data with large bandwidth, this is significantly more expensive than our current indicators. Not a dealbreaker but adds overhead for marginal benefit.
- **The 30K likes are largely driven by the repainting version.** The visually impressive overlays that attract users rely on future data. The non-repainting version looks much worse and gets less attention.

### Related Indicators

- **Bollinger Bands**: Mean +/- N*SD. Functionally equivalent to the non-repainting Nadaraya-Watson envelope. Same concept, simpler math.
- **Keltner Channels**: Mean +/- N*ATR. Similar envelope concept, less sensitive to outliers.
- **Z-Score Probability**: Normalized version of Bollinger Bands. Our existing file covers this.
- **Squeeze Momentum (TTM)**: Uses the relationship between BB and KC. Rejected for our system.

### Correlation with Our Existing Indicators

- **SM**: Low correlation. SM measures momentum, Nadaraya-Watson measures price position within a smoothed envelope.
- **RSI**: Moderate correlation. Both identify overbought/oversold. The envelope approach and momentum-ratio approach can diverge in trending markets.

## Fit With Our System

### Entry Gate
Block entries when price is at envelope extremes. But this is equivalent to what we already tested with VWAP Z-Score (block when price is overextended from mean), which was REJECTED on Mar 6 -- vScalpA OOS always degraded.

The non-repainting version would behave similarly to a Bollinger Band gate, which is a component of the Squeeze (TTM) gate that was also REJECTED (redundant with SM+RSI).

## Status

**Not tested, effectively rejected.** The repainting issue makes the default mode unsuitable for backtesting or live trading. The non-repainting mode reduces to a standard envelope indicator, which overlaps with already-rejected concepts (VWAP Z-Score, Squeeze/BB).

## Test Results

*(None -- not backtested)*

## Backtest Priority

**None** -- Repainting disqualifies the default mode. Non-repainting mode is a Bollinger Band variant. We already tested Squeeze (TTM) which uses BB, and VWAP Z-Score which measures the same thing (price deviation from smoothed mean). Both were rejected. No reason to test a third variant of the same concept.
