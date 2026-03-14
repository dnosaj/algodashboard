---
name: Chandelier Exit
description: ATR-based trailing stop that tracks local extremes and creates dynamic exit levels. Potential exit signal for MES v2 runner.
type: reference
---

# Chandelier Exit

## Source

- **Author**: everget (TradingView)
- **License**: Open source (Pine Script viewable)
- **Note**: Based on the original concept by Charles Le Beau, popularized in Alexander Elder's trading books.

## Discovery Context

Surveyed Mar 6, 2026 as part of the TradingView open-source indicator research. Identified as a potential exit signal for MES v2 runner leg, to replace or supplement BE_TIME.

## What It Does

Creates a dynamic trailing stop based on ATR:

### Long Exit
```
Chandelier_Long = Highest_High(N) - Mult * ATR(N)
```

### Short Exit
```
Chandelier_Short = Lowest_Low(N) + Mult * ATR(N)
```

The exit level ratchets in the direction of the trade (only moves in your favor, never against). When price crosses the chandelier level, exit the trade.

Key property: the stop distance adapts to current volatility (ATR). In high volatility, the stop is wider. In low volatility, it tightens.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| ATR Period | 22 | Lookback for ATR calculation |
| ATR Multiplier | 3.0 | Distance from extreme in ATR units |
| Use Close | true | Use close vs high/low for signals |

## Code

### Python (core logic)

```python
def chandelier_exit(highs, lows, closes, period=22, mult=3.0):
    """Compute Chandelier Exit levels."""
    n = len(closes)
    atr = compute_atr(highs, lows, closes, period)  # standard ATR

    long_exit = np.full(n, np.nan)
    short_exit = np.full(n, np.nan)

    for i in range(period - 1, n):
        hh = np.max(highs[i - period + 1 : i + 1])
        ll = np.min(lows[i - period + 1 : i + 1])
        long_exit[i] = hh - mult * atr[i]
        short_exit[i] = ll + mult * atr[i]

    return long_exit, short_exit
```

## Analysis

### Useful

- **Volatility-adaptive stop.** Unlike our fixed SL (10-40 pts), Chandelier adjusts to market conditions. During quiet periods the stop tightens; during volatile periods it widens.
- **Established methodology.** Chandelier Exit has decades of use in trend-following systems. Well-understood behavior.
- **Simple to compute.** Just ATR + highest-high/lowest-low. Minimal overhead.
- **Natural fit for runner exit.** Could replace BE_TIME (75 bars) on MES v2 or complement vScalpC structure exit with a volatility-aware trailing stop.

### Flawed

- **We already have fixed SL/TP exits that work.** Our backtesting consistently shows fixed exits outperform dynamic exits on these strategies. SM flip exit failed OOS. Trailing stops failed for MNQ in v14/v15 testing (PF 0.933).
- **ATR-based exits were explicitly tested and rejected** for MNQ (Mar 9, atr_adx_research.md). ATR-scaled exits are "just bigger fixed exits" without capturing meaningful information.
- **The ratcheting behavior can create premature exits.** During normal pullbacks within a trend, the chandelier level may be hit if the ATR multiplier is too tight.
- **Parameter sensitivity.** ATR period and multiplier need careful tuning. On 1-min bars, ATR(22) = 22 minutes of data. Mult=3.0 may be too wide for TP=20 MES trades.

### Related Indicators

- **Our fixed SL/TP exits**: Simpler, validated approach. TP=7/SL=40 (V15), TP=3/SL=10 (vScalpB), TP=20/SL=35 (MES v2).
- **vScalpC Structure Exit (adopted)**: Pivot-based swing detection. Not volatility-adaptive but captures structural turning points.
- **BE_TIME (MES v2, adopted)**: Time-based exit after 75 bars. Simpler than Chandelier but addresses the same problem (closing stale trades).
- **ATR-scaled exits (rejected, Mar 9)**: Same concept of using ATR to size exits. Failed.

### Correlation with Our Existing Indicators

- **SM**: Low correlation. SM measures momentum, Chandelier measures distance from recent extreme.
- **ATR gate (prior-day ATR for vScalpC)**: Same ATR calculation but used as entry gate, not exit.

## Fit With Our System

### MES v2 Runner Exit
Replace or supplement BE_TIME with Chandelier trailing stop for the runner leg (after TP1). The runner currently rides to TP2=20 or SL or BE_TIME=75 bars.

**Problem**: Our prior research (Mar 9, Mar 17) established that dynamic exits consistently underperform fixed exits on these strategies. The SM flip exit (dynamic, trend-following exit) was the pivotal failure that drove our entire TP-exit architecture. Chandelier is another form of dynamic exit.

### vScalpC Runner
Could add a Chandelier trailing stop alongside structure exit. But structure exit (pivot-based) already adapts to market structure, and adding a second dynamic exit layer increases complexity without clear benefit.

## Status

**Untested** -- not backtested for our strategies. However, the category (ATR-based dynamic exits) has been tested and rejected multiple times:
- Trailing stops rejected for MNQ (Feb 17, PF 0.933)
- ATR-scaled exits rejected (Mar 9, "just bigger fixed exits")
- SM flip exit (dynamic) rejected OOS (the foundational finding)

## Test Results

*(None for Chandelier specifically)*

## Backtest Priority

**Very Low** -- The entire category of dynamic/trailing exits has been rejected for our system. SM flip exit failed OOS. Trailing stops failed for MNQ. ATR-scaled exits failed. Our architecture is built on the finding that fixed TP exits outperform dynamic exits. Testing Chandelier would be re-litigating a settled question unless new evidence emerges that volatility-adaptive exits behave fundamentally differently from the trailing stops already tested.
