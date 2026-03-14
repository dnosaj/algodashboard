---
name: VWAP Deviation Oscillator
description: Measures price deviation from session VWAP in Z-Score units. Entry gate candidate to block overextended entries. REJECTED Mar 6, 2026 — vScalpA OOS always degrades.
type: reference
---

# VWAP Deviation Oscillator [BackQuant]

## Source

- **Author**: BackQuant (TradingView)
- **URL**: https://www.tradingview.com/script/ (search "VWAP Deviation Oscillator BackQuant")
- **Popularity**: ~7.7K likes, 44K boosts
- **License**: Open source (Pine Script viewable)

## Discovery Context

Catalogued during Mar 6, 2026 TradingView indicator survey. Rated HIGH priority for Round 1 backtesting. The concept is philosophically aligned with our "don't chase extended moves" approach -- block entries when price is statistically far from session VWAP.

## What It Does

Measures how far price deviates from session VWAP, expressed as a Z-Score. Sessions reset at configurable boundaries (daily, 4H, weekly, rolling). The Z-Score standardizes the deviation by its own rolling mean and standard deviation, making it comparable across different volatility regimes.

### Math

```
deviation = close - VWAP

rolling_mean = SMA(deviation, window)
rolling_std  = StDev(deviation, window)

z_score = (deviation - rolling_mean) / rolling_std
```

Fixed bucket edges at 0.5, 1.0, 2.0, 2.8 sigma for visual classification.

Three deviation modes available:
1. **Percent**: `(close - VWAP) / VWAP * 100`
2. **Absolute**: `close - VWAP` (in points)
3. **Z-Score**: Standardized as above (our tested mode)

### Gate Logic (as tested)

```
gate_open = |z_score| <= max_z
```

Block entry when price is more than `max_z` standard deviations from VWAP. Session resets at 18:00 ET (futures session boundary).

## Parameters

| Parameter | Default | Tested Range | Description |
|-----------|---------|-------------|-------------|
| VWAP Mode | Daily | Daily | Session anchor (4H/Daily/Weekly/Rolling) |
| Price Reference | HLC3 | close | Price source for deviation |
| Deviation Method | Z-Score | Z-Score | Percent, Absolute, or Z-Score |
| Z/Std Window | 60 | 60 | Rolling window for z-score calculation |
| Min Sigma Guard | varies | n/a | Floor for std to prevent division issues |
| max_z (gate) | n/a | [1.0, 1.5, 2.0, 2.5, 3.0] | Gate threshold (our addition) |

## Code

### Pine Script (core logic)

```pinescript
//@version=5
// VWAP Deviation Oscillator — BackQuant (simplified Z-Score mode)

indicator("VWAP Dev", overlay=false)

vwap_val = ta.vwap(hlc3)  // session-anchored VWAP
deviation = close - vwap_val

// Z-Score of the deviation
window = input.int(60, "Z Window")
dev_mean = ta.sma(deviation, window)
dev_std = ta.stdev(deviation, window)
z_score = dev_std > 0 ? (deviation - dev_mean) / dev_std : 0

plot(z_score, "Z-Score", color=z_score > 0 ? color.green : color.red)
hline(2.0, "+2 SD", color=color.yellow)
hline(-2.0, "-2 SD", color=color.yellow)
```

### Python (backtesting implementation)

```python
def compute_vwap_zscore(closes, vwap, times, rolling_window=60):
    """Compute z-score of the close-VWAP deviation within each futures session.

    Sessions reset at 18:00 ET (1080 minutes from midnight). Within each
    session the deviation (close - VWAP) is z-scored using a rolling mean and
    std over rolling_window bars.

    Returns:
        z_score: 1-D numpy float64 array. Bars with insufficient data get z_score = 0.
    """
    n = len(closes)
    z_score = np.zeros(n, dtype=np.float64)
    et_mins = compute_et_minutes(times)

    # Forward-fill VWAP NaN within sessions
    # ... (session boundary detection, NaN handling)

    # Rolling z-score within each session
    for i in range(n):
        if bars_in_session >= rolling_window:
            window_devs = deviations[i - rolling_window + 1 : i + 1]
            mean_dev = np.mean(window_devs)
            std_dev = np.std(window_devs, ddof=0)
            if std_dev > 1e-10:
                z_score[i] = (deviations[i] - mean_dev) / std_dev

    return z_score
```

Full implementation: `backtesting_engine/strategies/sr_vwap_zscore_sweep.py`

## Analysis

### Useful

- **Mathematically sound.** Z-Score of VWAP deviation is a principled measure of "how overextended is price relative to session fair value."
- **Session-anchored.** VWAP resets each session, making the deviation measurement relevant to current market conditions (not stale historical data).
- **Regime-adaptive.** The Z-Score normalization automatically adjusts for different volatility regimes. 2 sigma in a low-vol session is a different absolute deviation than 2 sigma in a high-vol session.
- **Philosophically aligned with our system.** "Don't chase overextended moves" is exactly our ADR gate philosophy, measured differently.

### Flawed

- **vScalpA OOS consistently degrades at every threshold.** This is the dealbreaker. Even at z3.0 (very loose, blocks 0-2 trades), vScalpA OOS PF still slightly degrades. The filter systematically removes vScalpA's profitable entries.
- **Root cause**: vScalpA (SM_T=0.0) enters on ANY SM crossover, including breakout entries at extended VWAP levels. These extended entries are often the highest-quality breakout trades (strong momentum, clear direction). The VWAP gate removes them because they're "far from mean" -- but being far from mean IS the breakout.
- **Rolling window (60 bars) creates instability.** The Z-Score at bar N is computed against a different distribution than bar N+1. As the session progresses, the rolling window slides and the Z-Score can jump without price moving.
- **1.6% NaN VWAP values on MNQ** required forward-filling. This is a data quality issue, not an indicator flaw, but adds implementation complexity.

### Related Indicators

- **Z-Score Probability [steversteves]**: Same Z-Score concept but against rolling SMA, not VWAP. See `zscore_probability.md`.
- **Bollinger Bands**: Mean +/- N*SD is the price-level equivalent of Z-Score. Z-Score is the normalized version.
- **Our ADR gate**: Similar philosophy (block chasing extended moves) but uses daily range percentage, not VWAP deviation. ADR gate PASSED where VWAP Z-Score FAILED.

### Correlation with Our Existing Indicators

- **SM**: Low correlation. SM measures trend direction via EMA. VWAP Z-Score measures price distance from volume-weighted mean. SM can be strongly positive while Z-Score is near zero (trending near VWAP) or extreme (trending away from VWAP).
- **RSI**: Moderate correlation. Both measure "overbought/oversold" from different angles. RSI uses momentum ratio, VWAP Z-Score uses statistical deviation from fair value.
- **ADR gate**: High expected correlation. Both answer "has price moved too far?" ADR uses daily range percentage, VWAP Z-Score uses session deviation. On 1-min data with similar lookbacks, they likely flag similar conditions.

### Instrument / Timeframe Notes

- On 1-min bars with 60-bar rolling window, the Z-Score measures deviation over ~60 minutes of trading.
- For NQ/MNQ: High intraday volatility means the VWAP deviation can be large in absolute terms but still modest in Z-Score terms. This is by design (normalization).
- MES: z2.0 showed MARGINAL improvement (OOS PF +3.3%), better than MNQ, but not strong enough to adopt.
- VWAP is most meaningful during RTH (9:30-16:00 ET). Pre-RTH and globex session VWAP is less reliable due to lower volume.

### Parameter Sensitivity

- **max_z threshold**: Sweept [1.0, 1.5, 2.0, 2.5, 3.0]. Tighter thresholds block more trades and hurt performance more. Even the loosest threshold (3.0) doesn't help.
- **Rolling window (60)**: Fixed. Shorter windows would be more responsive but noisier. Not tested.
- The failure is not parameter-sensitive -- vScalpA degrades at every tested threshold. This suggests a fundamental mismatch, not a tuning problem.

### Computational Cost

Low-moderate. Requires VWAP computation (running sum of price*volume / running sum of volume, reset per session) plus rolling mean and std of the deviation. Already have VWAP tracking in the live engine for the Digest Agent's forensic tools.

## Fit With Our System

### Tested Use Case: VWAP Z-Score Entry Gate

Block entries when `|z_score| > max_z`. Tested as part of Round 3 S/R filter research alongside Leledc, Squeeze, IB, Prior Day Levels, and Intraday Pivots.

### Why It Failed

The VWAP Z-Score gate removes profitable breakout entries from vScalpA. Our strategies are momentum-based -- they enter on SM crossovers in the direction of the trend. When a trend is strong, price moves away from VWAP, and the Z-Score increases. But these are exactly the conditions where our entries work best (strong trend, clear direction). The gate fights our own edge.

The ADR directional gate (adopted Mar 10) addresses the same "don't chase extended moves" concern but does so relative to the daily range budget, which is a better frame of reference for our TP=5-7 scalps than VWAP Z-Score.

### VWAP Still Useful in Our System

Despite the Z-Score gate failing as an entry filter, VWAP itself is used in the live engine:
- VWAP accumulation in SafetyManager (RTH-only) for Digest Agent forensic analysis
- `get_level_proximity` tool reports distance to VWAP at entry
- `gate_state_snapshots` records vwap_close for post-hoc analysis
- VWAP is useful as a reference level, just not as an entry gate threshold

## Status

**REJECTED** -- Mar 6, 2026. vScalpA OOS degrades at every threshold. The gate removes profitable breakout entries.

## Test Results

**Script**: `backtesting_engine/strategies/sr_vwap_zscore_sweep.py`
**Date**: March 6, 2026
**Data**: MNQ+MES 1-min bars, 12.5 months, chronological IS/OOS split

### Sweep: max_z = [1.0, 1.5, 2.0, 2.5, 3.0] (5 configs)

**All-3-Strategy Results**:

| Config | vScalpA IS/OOS dPF% | vScalpB IS/OOS dPF% | MES_V2 IS/OOS dPF% | Portfolio OOS Sharpe |
|--------|---------------------|---------------------|---------------------|---------------------|
| z1.0 | -4.2%/-16.9% | +3.0%/-8.1% | -17.0%/-14.2% | 0.402 |
| z1.5 | +9.9%/-10.0% | +0.1%/-9.7% | -6.8%/-4.7% | 0.815 |
| z2.0 | +18.3%/-6.1% | +3.4%/-0.7% | -0.5%/+3.3% | 1.234 |
| z2.5 | +0.9%/-5.7% | -0.3%/-4.9% | +2.7%/-0.2% | 1.082 |
| z3.0 | +5.1%/+0.7% | -1.1%/-2.6% | +1.0%/-0.2% | 1.209 |

**Verdict: FAIL** -- vScalpA OOS always degrades. Consistent IS-to-OOS degradation at every threshold.

### MES-Only Evaluation

| Config | IS PF | OOS PF | OOS dPF% | OOS Sharpe | Verdict |
|--------|-------|--------|----------|------------|---------|
| z2.0 | 1.360 | 1.335 | +3.3% | 1.720 | MARGINAL |
| z2.5 | 1.404 | 1.289 | -0.2% | 1.502 | MARGINAL |
| z3.0 | 1.380 | 1.289 | -0.2% | 1.511 | MARGINAL |

MES shows marginal improvement at z2.0 but not enough to adopt (prior-day level gate is STRONG PASS at +9.0%).

### Decision Log Entry

From `decision_log.md`, Mar 6:
> **REJECTED** VWAP Z-Score, Squeeze (TTM), Intraday Pivots, combined IB+Leledc.

## Backtest Priority

**None** -- conclusively rejected. The failure is fundamental (removes profitable breakout entries), not parametric. Further threshold tuning would not fix the mismatch between a mean-reversion gate and a momentum-based strategy.

The ADR directional gate (see `adr_exhaustion_research.md`) already addresses the same concern with better results. VWAP is useful as a reference level for analysis, but not as an entry gate.
