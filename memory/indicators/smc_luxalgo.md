---
name: Smart Money Concepts (SMC)
description: All-in-one market structure indicator — BOS, CHoCH, order blocks, FVGs, premium/discount zones, swing point labels. The most popular indicator on TradingView.
type: reference
---

# Smart Money Concepts (SMC) [LuxAlgo]

## Source

- **Author**: LuxAlgo (TradingView)
- **URL**: https://www.tradingview.com/script/ (search "Smart Money Concepts LuxAlgo")
- **Popularity**: ~123K likes, 12K boosts -- #1 most-liked indicator on TradingView
- **License**: Open source (Pine Script viewable)

## Discovery Context

Catalogued during Mar 6, 2026 TradingView indicator survey. Filed under "Market Structure Indicators" as the most comprehensive SMC implementation available. Assessed alongside simpler alternatives (mickes Market Structure, LuxAlgo Fractal CHoCH/BOS).

## What It Does

All-in-one Smart Money Concepts indicator combining multiple ICT/institutional trading concepts:

1. **Break of Structure (BOS)**: Price breaks a swing high/low in the direction of the existing trend, confirming trend continuation.
2. **Change of Character (CHoCH)**: Price breaks a swing high/low AGAINST the existing trend, signaling potential reversal.
3. **Order Blocks**: Zones where institutional buying/selling occurred. Last bullish candle before a down move = bearish OB, and vice versa.
4. **Fair Value Gaps (FVGs)**: Three-bar imbalance zones (current low > high 2 bars ago for bullish FVG).
5. **Premium/Discount Zones**: Fibonacci-based zones above/below equilibrium (50% of range). Premium = sell zone, Discount = buy zone.
6. **Equal Highs/Lows**: Detects clusters of similar swing points (liquidity pools).
7. **Swing Point Labels**: HH/HL/LH/LL classification of every swing point.

### Math/Logic

BOS/CHoCH detection uses pivot-based swing points at configurable lookback lengths:
- Internal structure: shorter pivot length (default ~5) for granular moves
- Swing structure: longer pivot length (default ~50) for macro trend

The indicator maintains a running "trend state" (bullish/bearish) based on swing sequence. A break of a swing high in a bullish trend = BOS (continuation). A break of a swing low in a bullish trend = CHoCH (reversal).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Internal Structure Pivot Length | ~5 | Lookback for internal swing detection |
| Swing Structure Pivot Length | ~50 | Lookback for swing-level structure |
| Order Block Settings | various | Show/hide, mitigation method |
| FVG Threshold | auto | Min FVG height (auto = cumulative mean) |
| Premium/Discount | on/off | Show Fibonacci equilibrium zones |
| Equal H/L Detection | on/off | Cluster detection for liquidity pools |

## Code

### Pine Script

Not extracted -- the full LuxAlgo script is 1000+ lines. The core BOS/CHoCH logic requires maintaining swing state, trend direction, and multiple pivot buffers. Too complex to meaningfully excerpt.

### Python

No Python port exists. Would require:
- Pivot detection (swing high/low at configurable lookback)
- Trend state machine (bullish/bearish based on swing sequence)
- BOS/CHoCH classification (break direction vs trend direction)
- Order block zone tracking (last opposing candle before structure break)
- FVG detection (3-bar pattern, separate indicator file)

Estimated porting effort: 300-500 lines of Python, moderate complexity.

## Analysis

### Useful

- **CHoCH as exit signal**: When market structure shifts against our position, it's a genuine reversal signal. Could complement our structure-based exit for vScalpC.
- **BOS as entry confirmation**: Entry only when structure confirms trend direction. Adds confluence to SM signal.
- **Most battle-tested SMC implementation**: 123K likes means extensive community validation and bug-fixing over years.

### Flawed

- **Massive overlap with our SM indicator.** Our Squeeze Momentum (SM) already handles trend detection via EMA crossover. BOS/CHoCH is a different approach to the same question: "which direction is the trend?"
- **Very complex to port.** The indicator is a monolith. Extracting just the CHoCH component requires understanding the full state machine.
- **Pivot-based detection is inherently lagging.** Pivot confirmation requires N bars after the swing point, adding latency that matters on 1-min bars.
- **On 1-min bars, swing detection is very noisy.** Designed for higher timeframes (15-min, 1H, 4H). On 1-min NQ, every 5-bar pullback creates a "swing."
- **The order block and FVG components are better tested independently** (see `fair_value_gap.md`).

### Related Indicators

- **Market Structure by mickes**: Simpler BOS/CHoCH only, pivot lengths 5/50
- **Market Structure CHoCH/BOS (Fractal) [LuxAlgo]**: Fractal-based (not swing), may be noisier on 1-min
- **Our SM (Squeeze Momentum)**: Different mechanism (EMA-based) but answers same question
- **Fair Value Gap [LuxAlgo]**: FVG component extracted as standalone (see `fair_value_gap.md`)

### Correlation with Our Existing Indicators

- **SM**: High expected correlation. Both detect trend direction. SM uses EMA momentum, SMC uses swing structure. Likely 0.5-0.7 correlation on trend direction signals.
- **RSI**: Low correlation. RSI measures momentum exhaustion, SMC measures structural breaks. Different axes.

### Instrument / Timeframe Notes

- Designed for any timeframe but optimized for 15-min+ charts
- On 1-min NQ/MNQ: swing detection would generate excessive noise with short pivot lengths
- On 1-min: would need pivot length >= 20-50 to produce meaningful structure, but this adds significant lag
- Better suited for MES v2 (longer holds, TP=20) than MNQ scalps (TP=3-7)

### Parameter Sensitivity

- **Pivot lengths are critical.** Too short = noise, too long = lagging. Finding the right balance on 1-min futures is the main challenge.
- No established parameter ranges for 1-min NQ. Would need a full sweep.

### Computational Cost

High relative to our other indicators. Maintaining swing state, trend classification, order block zones, and FVG tracking on every bar would add meaningful overhead to the live engine. Estimated 20-50 operations per bar (vs 5-10 for SM+RSI combined).

## Fit With Our System

### Potential Use Cases

1. **CHoCH exit signal for vScalpC runner** -- exit when structure shifts against position. But we already have pivot-based structure exit implemented (LB=50, PR=2) which is conceptually similar.

2. **BOS entry confirmation** -- only enter when recent BOS confirms SM direction. Would reduce false entries but likely cuts trade count significantly. Given that SM magnitude and SM slope filters already failed for MES (see `mes_v2_entry_filter_research.md`), adding another trend-direction filter is unlikely to help.

3. **Order blocks as S/R levels** -- use order block zones as entry gates (don't enter inside an order block against the order flow). Conceptually similar to prior-day level gate, which already works for MES.

### Assessment

The SMC indicator is too complex to port for marginal benefit. Our existing indicators (SM for trend, RSI for exhaustion, Leledc for directional exhaustion, prior-day levels for S/R) already cover the same ground through simpler, tested mechanisms. The individual components (FVG, IB) are better tested as standalone indicators.

## Status

**Not tested** -- too complex to port to Python. Individual components (FVG, structure levels) are better tested as standalone indicators.

## Test Results

None.

## Backtest Priority

**Low** -- Our SM already handles trend detection. The porting effort (300-500 lines, complex state machine) is not justified given the overlap. If we need better structure detection, the simpler pivot-based approach we already use for vScalpC exits is more practical. Revisit only if our structure exit underperforms and we need a fundamentally different approach to market structure.
