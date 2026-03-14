---
name: Fair Value Gap (FVG)
description: Detects 3-bar price imbalance zones (gaps between candle wicks). Entry gate candidate to block entries inside unfilled FVGs. Not tested.
type: reference
---

# Fair Value Gap [LuxAlgo]

## Source

- **Author**: LuxAlgo (TradingView)
- **URL**: https://www.tradingview.com/script/ (search "Fair Value Gap LuxAlgo")
- **Popularity**: ~7.7K likes, 515 boosts
- **License**: Open source (Pine Script viewable)

### Related Indicator
- **Multitimeframe Fair Value Gap [Zeiierman]**: Premium author. Multi-timeframe FVG with smart volume logic. Concept is portable even though indicator is premium.

## Discovery Context

Catalogued during Mar 6, 2026 TradingView indicator survey. Rated MEDIUM priority for Round 2 backtesting. FVGs are a core ICT (Inner Circle Trader) concept -- zones where price moved so fast that it left a gap between candle wicks, which the market often returns to fill.

## What It Does

Detects Fair Value Gaps -- three-bar price imbalance patterns where there's a gap between the high of bar[i-2] and the low of bar[i]:

### Math

**Bullish FVG** (gap up, price moved up too fast):
```
FVG exists when: low[i] > high[i-2]
FVG zone: from high[i-2] to low[i]
```
The gap represents buyers stepping in aggressively -- no trading occurred in that price zone. Price often returns to "fill" the gap (retrace into it).

**Bearish FVG** (gap down, price moved down too fast):
```
FVG exists when: high[i] < low[i-2]
FVG zone: from low[i-2] to high[i]
```

**Mitigation** (filling):
```
FVG is mitigated when price returns to trade within the gap zone.
Partial mitigation: price enters the zone but doesn't cross it.
Full mitigation: price crosses the entire FVG zone.
```

**Instantaneous mitigation**: FVG created and filled within a few bars -- signals rapid reversal (price rejected the gap fill). This is a reversal signal.

### Key Concepts

- **Unmitigated FVGs**: Gaps that haven't been filled yet. Price tends to return to these zones -- they act as magnets.
- **FVG as S/R**: Unmitigated FVGs function like support (bullish FVG below price) and resistance (bearish FVG above price).
- **FVG quality**: Larger FVGs (more points) on higher timeframes are more significant than small 1-min FVGs.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Threshold % | auto | Minimum FVG height (auto = cumulative mean of all FVGs) |
| Auto Threshold | on | Use cumulative mean to filter small FVGs |
| Unmitigated Levels | on | Show only unfilled FVGs |
| Timeframe | chart | Timeframe for FVG detection (MTF available) |

## Code

### Pine Script (core logic)

```pinescript
//@version=5
// Fair Value Gap — simplified core detection

indicator("FVG", overlay=true)

// Bullish FVG: gap between bar[i-2] high and bar[i] low
bull_fvg = low > high[2]
bull_top = low
bull_bot = high[2]

// Bearish FVG: gap between bar[i-2] low and bar[i] high
bear_fvg = high < low[2]
bear_top = low[2]
bear_bot = high

// Track unmitigated FVGs (simplified — full version uses arrays)
// Mitigation: price re-enters the gap zone
// bull_mitigated = low <= bull_top (price came back down into the gap)
// bear_mitigated = high >= bear_bot (price came back up into the gap)
```

### Python

No Python implementation exists. The core detection is trivial (3-bar pattern), but tracking unmitigated FVGs over time requires maintaining a list of active zones and checking each bar against all active FVGs.

**Estimated implementation**:
```python
def detect_fvg(highs, lows):
    """Detect Fair Value Gaps (3-bar imbalance zones)."""
    n = len(highs)
    bull_fvg = np.zeros(n, dtype=bool)
    bear_fvg = np.zeros(n, dtype=bool)
    fvg_top = np.full(n, np.nan)
    fvg_bot = np.full(n, np.nan)

    for i in range(2, n):
        if lows[i] > highs[i - 2]:  # Bullish FVG
            bull_fvg[i] = True
            fvg_top[i] = lows[i]
            fvg_bot[i] = highs[i - 2]
        elif highs[i] < lows[i - 2]:  # Bearish FVG
            bear_fvg[i] = True
            fvg_top[i] = lows[i - 2]
            fvg_bot[i] = highs[i]

    return bull_fvg, bear_fvg, fvg_top, fvg_bot

def build_fvg_gate(highs, lows, closes):
    """Block entries inside unmitigated FVGs."""
    bull_fvg, bear_fvg, fvg_top, fvg_bot = detect_fvg(highs, lows)
    n = len(closes)
    gate = np.ones(n, dtype=bool)  # True = allow

    # Track active (unmitigated) FVGs
    active_fvgs = []  # list of (top, bot, direction)

    for i in range(n):
        # Create new FVGs
        if bull_fvg[i]:
            active_fvgs.append((fvg_top[i], fvg_bot[i], 'bull'))
        if bear_fvg[i]:
            active_fvgs.append((fvg_top[i], fvg_bot[i], 'bear'))

        # Check mitigation (remove filled FVGs)
        remaining = []
        for top, bot, direction in active_fvgs:
            if direction == 'bull' and lows[i] <= top:
                continue  # mitigated
            if direction == 'bear' and highs[i] >= bot:
                continue  # mitigated
            remaining.append((top, bot, direction))
        active_fvgs = remaining

        # Block if price is inside any active FVG
        for top, bot, direction in active_fvgs:
            if bot <= closes[i] <= top:
                gate[i] = False
                break

    return gate
```

## Analysis

### Useful

- **Sound market microstructure concept.** FVGs represent genuine supply/demand imbalances. When price moves too fast, orders are left unfilled, creating a "vacuum" that price tends to return to fill.
- **Works as S/R levels.** Unmitigated FVGs function as support/resistance -- a more dynamic, market-generated alternative to prior-day levels.
- **3-bar pattern is simple to detect.** Core detection is trivial. The complexity is in tracking active zones and mitigation.
- **Multi-timeframe FVGs are more meaningful.** 1-min FVGs are likely noise, but 5-min or 15-min FVGs projected onto 1-min could provide meaningful imbalance zones.

### Flawed

- **1-min FVGs are probably too noisy on NQ.** NQ has tight spreads and high liquidity. True 3-bar gaps (no wick overlap) on 1-min are either very rare (genuine gaps = meaningful) or very common during fast moves (which is when our SM entries fire = would block good entries).
- **FVG density is the key unknown.** If there are dozens of active FVGs at any given time on 1-min NQ, the gate would either block everything (tight proximity) or block nothing (loose proximity). Without testing, we don't know the density.
- **Same fundamental risk as VWAP Z-Score gate.** FVGs form during strong moves. Our SM entries also fire during strong moves. A gate that blocks entries during/near strong moves fights our own momentum-based edge. The VWAP Z-Score failed for exactly this reason.
- **State management complexity.** Tracking active FVGs, checking mitigation per bar, aging out old FVGs -- more complex than our existing gates (Leledc = one counter, prior-day = 5 fixed levels).
- **No established parameter calibration for 1-min NQ.** All community use is on 15-min+ charts. Thresholds, minimum FVG size, maximum active FVGs, mitigation rules would all need calibration.

### Related Indicators

- **Smart Money Concepts [LuxAlgo]**: Includes FVG detection as a component (see `smc_luxalgo.md`)
- **Prior-day levels**: Similar concept (block near known levels). Prior-day levels are simpler and already adopted for MES.
- **IB levels**: Another "block near known levels" gate (see `initial_balance.md`). Simpler, already tested.

### Correlation with Our Existing Indicators

- **SM**: Potentially high inverse correlation as a gate. FVGs form during SM crossover events (strong moves). Blocking near FVGs could systematically remove our best entries.
- **Prior-day levels**: Moderate correlation. Both identify S/R zones, but FVGs are intraday and dynamic while prior-day levels are static.
- **Leledc**: Low correlation. Leledc measures consecutive directional closes; FVGs measure wick gaps. Different mechanisms.

### Instrument / Timeframe Notes

- **1-min NQ/MNQ**: FVG density unknown. High liquidity on NQ may mean true 1-min FVGs are rare (good -- each one is meaningful) or that they form and fill so fast they're noise.
- **5-min FVGs projected onto 1-min**: More promising approach. 5-min FVGs represent larger institutional imbalances. Could be computed on 5-min aggregated bars then checked on each 1-min entry.
- **MES**: Same underlying (ES), so FVG behavior should be identical to MNQ scaled by contract multiplier.

### Parameter Sensitivity

- **Minimum FVG size**: Critical and unknown. Need to calibrate for 1-min NQ. The auto-threshold (cumulative mean) from the TradingView indicator is a starting point.
- **Timeframe**: 1-min vs 5-min vs 15-min FVGs will behave very differently. Higher TF = fewer, more meaningful FVGs.
- **Proximity buffer**: How close to an FVG edge should trigger the gate. Would need a sweep similar to IB/prior-day level testing.
- **Max age**: How long an unmitigated FVG remains "active." FVGs from 3 days ago may no longer be relevant.

### Computational Cost

Moderate. Core detection is O(1) per bar, but tracking active FVGs requires iterating through all unmitigated zones per bar. With a cap on active FVGs (e.g., last 20), this is bounded but more expensive than our existing gates.

## Fit With Our System

### Potential Use Case: FVG Proximity Entry Gate

Block entries when price is inside an unmitigated FVG (imbalance zone likely to be revisited/rejected). Or block entries immediately after FVG creation (price moving too fast, likely to retrace).

### Concerns

1. **Same risk as VWAP Z-Score.** Our entries fire during strong moves. FVGs form during strong moves. Blocking near FVGs could systematically remove our best momentum entries. VWAP Z-Score failed for this exact reason.
2. **Prior-day levels already cover S/R proximity for MES.** FVG would add intraday S/R, but IB already covers that and was not adopted (superseded by Leledc and prior-day).
3. **Implementation complexity is higher than alternatives.** State management (active FVG list, mitigation tracking) is more complex than any gate we've built. Higher maintenance burden for uncertain benefit.
4. **5-min MTF FVGs would require bar aggregation.** Our data pipeline is 1-min bars. Computing 5-min FVGs requires aggregating 5 bars, adding a data transformation step.

### If Testing

- Start with 5-min FVGs projected onto 1-min (skip 1-min FVGs, likely too noisy)
- Test on MES first (level proximity matters more for TP=20 than TP=5-7)
- Use the same sweep methodology as prior-day levels (buffer sweep [0, 2, 5, 8, 10])
- Compare directly to prior-day level gate results (the FVG gate would need to beat buf5 STRONG PASS)

## Status

**Not tested** -- catalogued but never backtested. Deferred due to:
1. Higher-priority filters (Leledc, VWAP, Squeeze, IB, prior-day) tested first
2. Implementation complexity relative to simpler alternatives
3. Risk of same failure mode as VWAP Z-Score (blocking momentum entries)

## Test Results

None.

## Backtest Priority

**Low** -- multiple concerns suggest it may fail for the same reasons as VWAP Z-Score (blocking momentum entries during strong moves). If testing, use 5-min MTF FVGs on MES, not 1-min FVGs on MNQ. Prior-day levels already provide a STRONG PASS S/R gate for MES, so FVGs would need to meaningfully outperform an already-working solution.

The only scenario where FVGs become higher priority is if prior-day levels are removed/weakened and we need an alternative intraday S/R mechanism.
