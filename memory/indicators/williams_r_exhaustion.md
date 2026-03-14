---
name: "%R Trend Exhaustion"
description: Dual-period Williams %R confluence for detecting overextension. Potential entry gate to block entries when both short and long %R are overbought/oversold.
type: reference
---

# %R Trend Exhaustion

## Source

- **Author**: upslidedown (TradingView)
- **Popularity**: ~5.4K likes, 7.5K boosts
- **License**: Open source (Pine Script viewable)

## Discovery Context

Surveyed Mar 6, 2026 as part of the TradingView open-source indicator research. Rated MEDIUM priority. Noted as similar concept to our RSI filter but using Williams %R, which may capture a different exhaustion signature.

## What It Does

Combines two Williams %R periods to detect trend exhaustion via confluence:

1. **Short-period %R**: Fast oscillator for immediate overbought/oversold
2. **Long-period %R** (default 112): Slow oscillator for trend-level overextension
3. **Confluence zone**: When BOTH periods are simultaneously OB/OS, price is at an "area of interest"
4. **Signal**: Break from the OB/OS confluence triggers a potential reversal signal

Williams %R formula: `%R = (Highest High - Close) / (Highest High - Lowest Low) * -100`

Range: -100 (oversold) to 0 (overbought). Functionally equivalent to an inverted Stochastic %K.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Short %R Period | (varies) | Fast period for immediate overextension |
| Long %R Period | 112 | Slow period for trend-level overextension |
| Smoothing | 3 | Smoothing applied to %R values |

## Code

No Pine Script extracted during survey. Python implementation would be straightforward:

```python
def williams_r(highs, lows, closes, period):
    """Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100"""
    n = len(closes)
    wr = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = np.max(highs[i - period + 1 : i + 1])
        ll = np.min(lows[i - period + 1 : i + 1])
        if hh != ll:
            wr[i] = (hh - closes[i]) / (hh - ll) * -100
    return wr
```

## Analysis

### Useful

- **Confluence concept is sound.** Dual-timeframe agreement (short and long period both OB/OS) is a stronger signal than either alone.
- **Cheap to compute.** Rolling highest/lowest are O(1) with deques. No overhead concern.
- **Different math than RSI.** %R uses highest-high/lowest-low range, RSI uses average gains/losses. They can diverge, especially in trending markets where RSI stays OB/OS longer than %R.

### Flawed

- **We already have RSI as our exhaustion filter.** RSI(8/60-40) for V15 and RSI(8/55-45) for vScalpB already gate on momentum extremes. Adding %R may be redundant.
- **Long period = 112 on 1-min bars = ~2 hours.** This is measuring session-level trend exhaustion on 1-min data. May be too slow for scalping entries.
- **Stochastic equivalence.** Williams %R is mathematically identical to Stochastic %K (just inverted). No novel information content.

### Related Indicators

- **RSI (our existing filter)**: Both measure "how extended is the move." RSI uses momentum ratio, %R uses price position within range.
- **Stochastic**: Mathematically equivalent to %R. Same information, different scale.
- **ADR exhaustion gate (our existing filter)**: Measures extension as % of average daily range. Different approach (session-anchored) vs %R (rolling window).

### Correlation with Our Existing Indicators

- **RSI**: Moderate-to-high expected correlation. Both measure overbought/oversold. On trending days they diverge (%R reverts to center faster than RSI). On choppy days they agree.
- **SM**: Low expected correlation. SM measures direction/magnitude, %R measures position within range.

## Fit With Our System

### Entry Gate
Block entries when both short-period and long-period %R show overextension in the entry direction (long entry blocked when both %R near 0/overbought, short entry blocked when both near -100/oversold).

This is conceptually identical to what our RSI filter already does. The question is whether %R captures different bad entries than RSI. Given the mathematical similarity to Stochastic and conceptual overlap with RSI, the marginal value is likely low.

## Status

**Untested** -- not backtested. Not mentioned in the decision log.

## Test Results

*(None)*

## Backtest Priority

**Low** -- We already have RSI as our primary oscillator filter and it works well (RSI 8/60-40 for V15, 8/55-45 for vScalpB). The prior research conclusion that "adding more oscillators has diminishing returns" applies here. Would only test if RSI showed gaps in specific failure modes that %R might address differently.
