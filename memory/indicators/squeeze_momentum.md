---
name: Squeeze Momentum Indicator
description: TTM Squeeze — detects low-volatility compression (BB inside KC) followed by expansion. Entry gate candidate to block during choppy consolidation. REJECTED Mar 6, 2026.
type: reference
---

# Squeeze Momentum Indicator [LazyBear]

## Source

- **Author**: LazyBear (TradingView)
- **URL**: https://www.tradingview.com/script/ (search "Squeeze Momentum LazyBear")
- **Popularity**: ~109K likes, 2.7M views -- one of the most popular indicators on TradingView
- **License**: Open source (Pine Script viewable)
- **Based on**: John Carter's TTM Squeeze (from "Mastering the Trade")

## Discovery Context

Catalogued during Mar 6, 2026 TradingView indicator survey. Rated HIGH priority for Round 1 backtesting based on conceptual fit: block entries during choppy consolidation (squeeze state) where SM+RSI signals are expected to whipsaw.

## What It Does

Detects volatility compression ("squeeze") by comparing Bollinger Band width to Keltner Channel width. When Bollinger Bands sit entirely inside Keltner Channels, volatility is compressed and a breakout is building. When BBs break outside KC, the squeeze "fires" and expansion begins.

### Math

**Bollinger Bands:**
```
BB_mid = SMA(close, bb_len)
BB_upper = BB_mid + bb_mult * StdDev(close, bb_len)
BB_lower = BB_mid - bb_mult * StdDev(close, bb_len)
```

**Keltner Channels:**
```
KC_mid = SMA(close, kc_len)
KC_upper = KC_mid + kc_mult * ATR_Wilder(kc_len)
KC_lower = KC_mid - kc_mult * ATR_Wilder(kc_len)
```

**Squeeze Detection:**
```
Squeeze ON  = (BB_upper < KC_upper) AND (BB_lower > KC_lower)
Squeeze OFF = NOT Squeeze ON
```

**Momentum Histogram** (linear regression of price minus midline):
```
val = linreg(close - avg(avg(highest(high, kc_len), lowest(low, kc_len)), SMA(close, kc_len)), kc_len, 0)
```
Histogram color = momentum direction (dark/light green for positive rising/falling, dark/light red for negative falling/rising).

### Signal Logic

- **Black dots** = squeeze ON (BB inside KC, consolidation)
- **Gray dots** = no squeeze (normal or expansion)
- **First gray dot after black** = squeeze just fired (potential breakout entry)
- Histogram direction indicates breakout direction

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| BB Length | 20 | Bollinger Band SMA lookback |
| BB MultFactor | 2.0 | BB standard deviation multiplier |
| KC Length | 20 | Keltner Channel SMA lookback |
| KC MultFactor | 1.5 | KC ATR multiplier |
| Use TrueRange | true | Use True Range vs simple H-L for KC |

## Code

### Pine Script (core logic)

```pinescript
//@version=5
// Squeeze Momentum — LazyBear (simplified core)

indicator("Squeeze Momentum", overlay=false)

bb_len = input.int(20, "BB Length")
bb_mult = input.float(2.0, "BB MultFactor")
kc_len = input.int(20, "KC Length")
kc_mult = input.float(1.5, "KC MultFactor")

// Bollinger Bands
bb_basis = ta.sma(close, bb_len)
bb_dev = bb_mult * ta.stdev(close, bb_len)
bb_upper = bb_basis + bb_dev
bb_lower = bb_basis - bb_dev

// Keltner Channels
kc_basis = ta.sma(close, kc_len)
kc_range = ta.atr(kc_len)
kc_upper = kc_basis + kc_mult * kc_range
kc_lower = kc_basis - kc_mult * kc_range

// Squeeze detection
sqz_on  = (bb_lower > kc_lower) and (bb_upper < kc_upper)
sqz_off = not sqz_on
no_sqz  = (bb_lower < kc_lower) and (bb_upper > kc_upper)

// Momentum
val = ta.linreg(close - math.avg(math.avg(ta.highest(high, kc_len), ta.lowest(low, kc_len)), ta.sma(close, kc_len)), kc_len, 0)
```

### Python (backtesting implementation)

```python
def compute_squeeze(closes, highs, lows,
                    bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5):
    """Detect TTM Squeeze -- Bollinger Bands inside Keltner Channels.

    Returns:
        Boolean numpy array (True = squeeze ON, False = squeeze OFF).
    """
    n = len(closes)
    squeeze_on = np.zeros(n, dtype=bool)

    # Bollinger Bands: rolling SMA and StdDev
    for i in range(bb_len - 1, n):
        window = closes[i - bb_len + 1 : i + 1]
        bb_mid = np.mean(window)
        bb_std = np.std(window, ddof=0)
        bb_upper = bb_mid + bb_mult * bb_std
        bb_lower = bb_mid - bb_mult * bb_std

        # Keltner Channels: SMA + ATR (Wilder)
        kc_mid = bb_mid  # same SMA if bb_len == kc_len
        atr = compute_atr_wilder(highs, lows, closes, kc_len)  # precomputed
        kc_upper = kc_mid + kc_mult * atr[i]
        kc_lower = kc_mid - kc_mult * atr[i]

        squeeze_on[i] = (bb_upper < kc_upper) and (bb_lower > kc_lower)

    return squeeze_on
```

Full implementation: `backtesting_engine/strategies/sr_squeeze_gate_sweep.py`

## Analysis

### Useful

- **Clean binary signal** -- squeeze ON/OFF is unambiguous. Easy to use as entry gate.
- **Sound theoretical basis** -- volatility mean-reverts. Compression precedes expansion. Well-established in quantitative finance (GARCH models capture same phenomenon).
- **Simple to implement** -- just BB and KC, both standard calculations.
- **Widely used** -- 109K likes, extensive community validation.

### Flawed

- **Redundant with our SM+RSI stack.** Our SM (Squeeze Momentum / EMA crossover) already avoids choppy periods implicitly -- SM stays near zero during consolidation, so entries don't fire. RSI bands (55/45 or 60/40) add another choppy-period filter. Adding BB-vs-KC squeeze triple-filters for the same condition.
- **Lagging indicator.** By the time BBs exit KCs (squeeze fires), the breakout move is already underway. For our TP=3-7 scalps, the first few points of the move are the ones we need.
- **KC multiplier is the critical parameter** and there's no principled way to set it for 1-min NQ bars. 1.5 is calibrated for daily charts.

### Related Indicators

- **Bollinger Bands**: BB component of the squeeze
- **Keltner Channels**: KC component of the squeeze
- **ATR**: Used in KC calculation
- **Our SM (Squeeze Momentum)**: Despite similar names, completely different -- our SM is EMA-based trend momentum, not BB-vs-KC squeeze detection

### Correlation with Our Existing Indicators

- **SM**: Moderate-high correlation expected. When BB is inside KC (squeeze ON), SM tends to oscillate near zero (no trend). Both indicate "don't enter" in the same regimes.
- **RSI**: Moderate correlation. During squeeze, RSI tends to stay in the neutral zone (40-60), which already blocks entries via our RSI band filter. Double-filtering.

### Instrument / Timeframe Notes

- Designed for daily/4H charts. On 1-min NQ, squeeze states may be very short-lived or very frequent depending on KC multiplier.
- kc_mult=1.5 (default) on 1-min NQ may produce nearly continuous squeeze states during lunch hours, then never trigger during morning volatility.
- MES and MNQ would likely show similar squeeze patterns (same underlying, different multipliers).

### Parameter Sensitivity

- **KC MultFactor is the most important parameter.** Lower = tighter KC = more frequent squeeze = more blocking. Higher = wider KC = rare squeeze = less blocking.
- Sweep tested: kc_mult = [1.0, 1.5, 2.0] x min_bars_off = [0, 5, 10] = 9 configs.
- No config passed for any strategy.

### Computational Cost

Low. BB and KC are standard rolling calculations. One SMA + one StDev + one ATR per bar. Already have ATR infrastructure in the live engine.

## Fit With Our System

### Tested Use Case: Squeeze Entry Gate

Block entries during squeeze (BB inside KC). Allow entries when squeeze has fired (expansion). Optionally keep blocking for `min_bars_off` bars after squeeze releases.

### Why It Failed

The filter is **redundant with SM+RSI**:
- Our strategies already avoid choppy periods via SM threshold (SM near zero = no entries) and RSI bands (RSI 55/45 or 60/40 blocks entries in neutral momentum).
- Adding squeeze as a third filter for the same condition just removes additional trades without improving the ones that remain.
- Squeeze detection is also **lagging** -- by the time BBs exit KCs, the breakout move has already started. Our SM detects trend direction earlier via EMA slope.

## Status

**REJECTED** -- Mar 6, 2026. Tested in Round 3 of S/R filter research. Every configuration hurt all strategies.

## Test Results

**Script**: `backtesting_engine/strategies/sr_squeeze_gate_sweep.py`
**Date**: March 6, 2026
**Data**: MNQ+MES 1-min bars, 12.5 months, chronological IS/OOS split

### Sweep: kc_mult x min_bars_off (9 configs)

**All-3-Strategy Verdict: TOTAL FAIL -- hurts all strategies**

Key findings:
- Every config failed at portfolio level (best: kc1.0_off10 OOS +$2,483 vs baseline $2,864)
- vScalpB universally harmed -- every config degraded both IS and OOS PF
- Only 3 MARGINAL PASS verdicts across 27 strategy-config combinations (9 configs x 3 strategies)
- Zero STRONG PASS verdicts

**Root cause**: Strategies already avoid choppy periods via SM threshold + RSI bands. Squeeze double-filters for the same condition with worse precision. Squeeze is also lagging -- by the time BB exits KC, the move is underway.

### Decision Log Entry

From `decision_log.md`, Mar 6:
> **REJECTED** VWAP Z-Score, Squeeze (TTM), Intraday Pivots, combined IB+Leledc.

## Backtest Priority

**None** -- conclusively rejected. No further testing planned. The mechanism is sound in theory but redundant with our existing indicator stack. If we ever remove SM or RSI from the entry logic, squeeze could be revisited as a replacement, but that's not on the roadmap.
