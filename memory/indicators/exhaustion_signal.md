---
name: Exhaustion Signal (ChartingCycles)
description: Simple consecutive bar counter measuring close vs close[4]. The simpler variant of the Leledc exhaustion concept. Leledc V4 (Joy_Bangla) was tested and adopted; this is the stripped-down version.
type: reference
---

# Exhaustion Signal (ChartingCycles)

## Source

- **Author**: ChartingCycles (TradingView)
- **Popularity**: ~1.4K likes
- **License**: Open source (Pine Script viewable)

## Discovery Context

Surveyed Mar 6, 2026 as part of the TradingView open-source indicator research. Rated MEDIUM priority. Described as "simpler version of Leledc. Worth testing as a quick filter." The Leledc V4 variant (Joy_Bangla) was tested and adopted before this was independently evaluated.

## What It Does

Counts consecutive bars where close > close[4] (bullish) or close < close[4] (bearish). When the count hits a configurable level, signals exhaustion.

This is the core counting mechanism from Leledc, stripped of the "major" vs "minor" quality/length distinction. Pure count.

### Signal Levels
- **Level 1** (default 9): First exhaustion warning
- **Level 2** (default 12): Stronger exhaustion
- **Level 3** (custom): Extreme exhaustion

### Logic

```
count = 0
for each bar:
    if close > close[4]:
        count += 1
    else:
        count = 0
    if count >= level_1:
        signal = exhaustion
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Level 1 | 9 | First exhaustion threshold |
| Level 2 | 12 | Second exhaustion threshold |
| Level 3 | custom | Third exhaustion threshold |

## Code

### Python (core logic)

```python
def exhaustion_signal(closes, level=9, lookback=4):
    """Count consecutive bars where close > close[lookback]."""
    n = len(closes)
    bull_count = np.zeros(n, dtype=int)
    bear_count = np.zeros(n, dtype=int)
    exhaustion = np.full(n, False)

    for i in range(lookback, n):
        if closes[i] > closes[i - lookback]:
            bull_count[i] = bull_count[i - 1] + 1
        else:
            bull_count[i] = 0

        if closes[i] < closes[i - lookback]:
            bear_count[i] = bear_count[i - 1] + 1
        else:
            bear_count[i] = 0

        if bull_count[i] >= level or bear_count[i] >= level:
            exhaustion[i] = True

    return exhaustion, bull_count, bear_count
```

## Analysis

### Useful

- **Extremely simple.** One counter, one comparison per bar. Trivial to implement and debug.
- **Conceptually identical to what we already adopted.** Our Leledc gate (mq9_p1) uses the same core logic: count bars where close > close[4], block at count >= 9.

### Flawed

- **This IS what we adopted, just branded differently.** The Leledc V4 (Joy_Bangla) that we tested includes this exact counting mechanism as its "major exhaustion" component. Our implemented gate (maj_qual=9, lookback=4, persistence=1) is functionally identical to ChartingCycles Level 1 = 9.
- **The "minor" exhaustion (shorter lookback/lower threshold) in Leledc V4 was not adopted.** We only use the major component, which is exactly this indicator.
- **Level 2 (12) and Level 3 are too rare on 1-min bars.** Our sweep showed mq10+ events are so rare they stop helping MES. Level 12 would fire almost never.

### Related Indicators

- **Leledc Exhaustion V4 (Joy_Bangla)** -- ADOPTED as mq9_p1. Superset of this indicator with major/minor quality and length parameters.
- **Rally Exhaustion Gate (tested for MES v2)** -- Different approach: measures total price displacement over a lookback window rather than counting directional bars.
- **Williams %R Exhaustion**: Uses oscillator-based overextension rather than bar counting.

### Correlation with Our Existing Indicators

This indicator is essentially our Leledc gate, so correlation is definitional (100%). No additional signal content.

## Fit With Our System

Already adopted in substance. Our Leledc gate (mq9_p1) for all MNQ strategies is the same counting mechanism with the same threshold and lookback.

## Status

**Adopted (as Leledc V4 mq9_p1)** -- The core logic of this indicator was tested on Mar 6, 2026 as part of the Round 3 S/R filter research. The Leledc V4 (Joy_Bangla) variant was used for the backtest, which includes this exact counting mechanism.

## Test Results

See Leledc Exhaustion results in `round3_sr_filter_research.md`:

- **Config**: maj_qual=9, persistence=1, lookback=4
- **vScalpA**: IS PF +7.2%, OOS PF +11.1% -- STRONG PASS
- **vScalpB**: IS PF -1.0%, OOS PF +5.9% -- MARGINAL PASS
- **MES_V2**: IS PF -3.8%, OOS PF +4.7% -- MARGINAL PASS
- **Portfolio OOS Sharpe**: 1.623 (+32% over baseline)
- **Blocks**: 6-11% of trades

The extended sweep confirmed mq9 is the peak -- mq10+ reverses at portfolio level.

## Backtest Priority

**None** -- Already tested and adopted under the Leledc V4 name. No separate test needed. The ChartingCycles variant is functionally identical to what we use.
