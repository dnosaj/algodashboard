---
name: Initial Balance / Opening Range
description: RTH first 30-60 min high/low as institutional reaction zones. Entry gate candidate to block near IB extremes. Passed all-3 criteria at ib60_buf5 but not adopted — Leledc chosen for MNQ, prior-day levels for MES.
type: reference
---

# Initial Balance / Opening Range

## Source

### Primary: Opening Range, Initial Balance, Opening Price [PtGambler]
- **Author**: PtGambler (TradingView)
- **Popularity**: ~656 likes, 1.56M views
- **License**: Open source (Pine Script viewable)
- **Focus**: Clean implementation of OR, IB, and opening price. Futures-aware with timezone support.

### Secondary: Initial Balance Breakout Signals [LuxAlgo]
- **Author**: LuxAlgo (TradingView)
- **Popularity**: ~6.7K likes, 1.7K boosts
- **License**: Open source
- **Focus**: More sophisticated -- auto-detects IB, marks extensions (1.5x, 2x), Fibonacci levels within IB, weekday-filtered forecasts.

## Discovery Context

Catalogued during Mar 6, 2026 TradingView indicator survey. Rated MEDIUM priority for Round 2 backtesting. IB levels are well-known institutional reaction zones on ES/NQ -- the RTH first-hour high/low often acts as a magnet/resistance for the rest of the day.

## What It Does

Computes the high and low price range established during the first N minutes of Regular Trading Hours (RTH, starting 9:30 ET). Two common periods:

1. **30-min IB**: 9:30-10:00 ET (fully formed before our entries start at 10:00 ET)
2. **60-min IB**: 9:30-10:30 ET (developing IB during 10:00-10:30, finalized after 10:30)

### Math

```
During IB period (9:30 to 9:30 + ib_period_minutes):
    ib_high = running max(bar.high) since 9:30
    ib_low  = running min(bar.low) since 9:30

After IB period:
    ib_high = finalized (fixed for rest of day)
    ib_low  = finalized (fixed for rest of day)

Gate:
    block = (|close - ib_high| <= buffer_pts) OR (|close - ib_low| <= buffer_pts)
```

### IB Extensions (LuxAlgo variant)

```
IB_range = ib_high - ib_low
extension_1.5 = ib_high + 0.5 * IB_range  (or ib_low - 0.5 * IB_range)
extension_2.0 = ib_high + 1.0 * IB_range  (or ib_low - 1.0 * IB_range)
```

Price often stalls at 1.5x and 2.0x IB extensions. These were NOT tested in our sweep.

## Parameters

| Parameter | Default | Tested Range | Description |
|-----------|---------|-------------|-------------|
| IB Period | 30 min | [30, 60] | Duration of IB window from RTH open |
| Buffer (pts) | 0 | [0, 2, 5, 8, 10] | Proximity threshold for gate |
| Session Start | 9:30 ET | fixed | RTH opening time |

## Code

### Pine Script

The PtGambler indicator is a standard session-tracking script. Core logic:

```pinescript
//@version=5
// Initial Balance — simplified core

var float ib_high = na
var float ib_low = na
var bool ib_finalized = false

is_rth_open = (hour == 9 and minute >= 30) or (hour >= 10)
is_ib_period = is_rth_open and (hour < 10 or (hour == 10 and minute < 30))  // 60-min IB

if is_new_session
    ib_high := high
    ib_low := low
    ib_finalized := false
else if is_ib_period and not ib_finalized
    ib_high := math.max(ib_high, high)
    ib_low := math.min(ib_low, low)
else
    ib_finalized := true

plot(ib_high, "IB High", color=color.blue)
plot(ib_low, "IB Low", color=color.blue)
```

### Python (backtesting implementation)

```python
def compute_initial_balance(times, highs, lows, et_mins, ib_period_minutes=30):
    """Compute developing/finalized Initial Balance high and low arrays.

    RTH starts at 9:30 ET (570 minutes from midnight). During IB period,
    levels are developing (running max/min). After IB period, finalized.

    Returns:
        (ib_high, ib_low): numpy float64 arrays, same length as times.
    """
    RTH_START = 570  # 9:30 ET
    ib_end = RTH_START + ib_period_minutes

    n = len(times)
    ib_high = np.full(n, np.nan, dtype=np.float64)
    ib_low = np.full(n, np.nan, dtype=np.float64)

    current_date = None
    running_high = np.nan
    running_low = np.nan
    ib_finalized = False

    for i in range(n):
        bar_date = pd.Timestamp(times[i]).date()
        bar_et = et_mins[i]

        if bar_date != current_date:
            current_date = bar_date
            running_high = np.nan
            running_low = np.nan
            ib_finalized = False

        if RTH_START <= bar_et < ib_end:
            running_high = highs[i] if np.isnan(running_high) else max(running_high, highs[i])
            running_low = lows[i] if np.isnan(running_low) else min(running_low, lows[i])
        elif bar_et >= ib_end:
            ib_finalized = True

        if not np.isnan(running_high):
            ib_high[i] = running_high
            ib_low[i] = running_low

    return ib_high, ib_low
```

Full implementation: `backtesting_engine/strategies/sr_initial_balance_sweep.py`

### Live Engine

IB/Opening Range tracking is already partially implemented in the live engine for Digest Agent forensic tools:
- `engine/safety_manager.py`: Opening range tracked during 10:00-10:30 ET (first 30 min of our entry window, slightly different from traditional 9:30 IB)
- `gate_state_snapshots`: Records `opening_range` columns
- Not used as an active entry gate

## Analysis

### Useful

- **Institutional standard.** IB is a well-established concept in market profile / auction market theory. The first hour's range captures the initial balance between buyers and sellers.
- **Simple to compute.** Running max/min for 30-60 bars, then fixed for the rest of the day. Trivial implementation.
- **60-min IB passed all-3-strategy criteria.** ib60_buf2 and ib60_buf5 both pass for all three strategies simultaneously (one of only two filters to achieve this, alongside Leledc mq9_p1).
- **MES shows consistent OOS improvement.** Across multiple IB configs, MES OOS PF improves despite IS degradation -- suggests the filter is removing IS overfit trades.
- **Complementary to Leledc.** IB catches "near a level" (location-based). Leledc catches "end of a run" (momentum-based). Different failure modes.

### Flawed

- **Marginal statistical significance.** vScalpA ib60_buf5 is the only STRONG PASS. vScalpB and MES are MARGINAL. With ~230 trades per half, the improvements are borderline noise.
- **30-min IB vs 60-min IB discrepancy.** 30-min IB fully formed before entries start (clean), but 60-min IB overlaps with our first 30 min of entries (10:00-10:30). During those 30 minutes, the IB is still developing -- gate uses an incomplete level. Despite this, 60-min IB outperforms 30-min consistently.
- **Only 2 levels.** IB high and IB low give just 2 reference points per day. Prior-day levels give 5 (H, L, VPOC, VAH, VAL). For MES, the prior-day level gate provides more granular level awareness.
- **No IB extensions tested.** 1.5x and 2.0x IB extensions are important institutional levels. We only tested proximity to IB high/low, not extensions. This is an untested dimension.
- **Not adopted despite passing.** While ib60_buf5 passes all-3 criteria, Leledc mq9_p1 has stronger portfolio OOS Sharpe (1.623 vs 1.459). The "one filter per instrument" principle means IB wasn't needed after Leledc was chosen for MNQ and prior-day levels for MES.

### Related Indicators

- **Prior-Day Levels**: Same concept (block near known levels) but uses yesterday's data. IB uses today's data. Both are S/R proximity gates.
- **Intraday Pivots**: Multi-window pivot detection. Much more complex, much higher false positive rate. TOTAL FAIL in testing.
- **VWAP**: Session-anchored fair value. VWAP Z-Score gate also failed.
- **NQ 65 Point Futures Session Opening Range [Bostonshamrock]**: 30-second opening range with 65-pt projections. Too granular for 1-min bars.

### Correlation with Our Existing Indicators

- **SM**: Low correlation. SM measures trend direction, IB measures proximity to a price level. Orthogonal.
- **Leledc**: Low correlation. Leledc measures consecutive directional closes (momentum persistence). IB measures location relative to session range. Different dimensions.
- **Prior-day levels**: Moderate correlation. Both are S/R proximity gates, but IB uses today's levels (session-specific) while prior-day uses yesterday's levels (carryover). Some overlap near unchanged open.

### Instrument / Timeframe Notes

- **MNQ**: 60-min IB at buf5 is STRONG PASS for vScalpA, MARGINAL for vScalpB. The IB range on MNQ (typically 50-150 pts) means a 5-pt buffer captures ~3-10% of the IB range.
- **MES**: IB is MARGINAL across all tested configs. MES's TP=20 means it needs 20 pts of runway -- IB levels are less constraining than prior-day levels for this purpose.
- **1-min bars**: IB levels are set once per day (30 or 60 min after open). The gate is straightforward on 1-min since levels don't change after the IB period.

### Parameter Sensitivity

- **IB period**: 60-min consistently outperforms 30-min. Likely because the 60-min range captures more of the morning auction and produces more meaningful levels.
- **Buffer**: 5 pts is the sweet spot. Smaller buffers (0, 2) don't block enough near-level entries. Larger buffers (8, 10) block too many trades (MES drops below 70% count).

| Config | vScalpA IS/OOS dPF% | vScalpB IS/OOS dPF% | MES IS/OOS dPF% | Portfolio OOS Sharpe |
|--------|---------------------|---------------------|-----------------|---------------------|
| ib60_buf2 | +6.6%/+3.5% | -0.2%/+0.4% | -2.5%/+3.4% | 1.392 |
| **ib60_buf5** | **+11.2%/+8.7%** | **+1.1%/-0.9%** | **-5.0%/+4.4%** | **1.459** |
| ib60_buf8 | +5.5%/+16.3% | -3.1%/+2.6% | N<70% | 1.591 |

### Computational Cost

Negligible. Running max/min for 30-60 bars at session start, then two fixed values for the rest of the day. The cheapest level-based indicator possible.

## Fit With Our System

### Tested Use Case: IB Proximity Entry Gate

Block entries when close is within `buffer_pts` of either IB high or IB low. Tested with 30-min and 60-min IB periods.

### Why Not Adopted

IB passed all-3 criteria (ib60_buf5) but was not adopted because:

1. **Leledc mq9_p1 has stronger portfolio impact.** OOS Sharpe 1.623 vs 1.459. Leledc is the better single filter for MNQ.
2. **Stacking IB+Leledc fails.** Combined filter pushes MES IS PF below -5% threshold. AND-ing filters compounds the trade count problem.
3. **MES has a better option.** Prior-day level gate (buf5) is STRONG PASS for MES (+9.0% OOS PF vs +4.4% for IB).
4. **One filter per instrument principle.** The signal is thin -- each filter catches different rare edge cases. Stacking dilutes rather than compounds.

### Combined Test Results (IB + Leledc)

| Config | vScalpA IS/OOS dPF% | vScalpB IS/OOS dPF% | MES IS/OOS dPF% | Portfolio OOS Sharpe |
|--------|---------------------|---------------------|-----------------|---------------------|
| ib only | +11.2%/+8.7% | +1.1%/-0.9% | -5.0%/+4.4% | 1.459 |
| leledc only | +7.2%/+11.1% | -1.0%/+5.9% | -3.8%/+4.7% | 1.623 |
| **ib+leledc** | **+12.3%/+17.1%** | **+0.2%/+0.4%** | **-5.6%/+6.0%** | **1.657** |

ib+leledc has the best portfolio Sharpe (1.657) but MES IS PF -5.6% triggers the fail threshold.

### Partial IB Tracking in Live Engine

The live engine already tracks opening range (10:00-10:30 ET) in SafetyManager for Digest Agent analysis. This is slightly different from traditional IB (9:30-10:00 or 9:30-10:30) because our entry window starts at 10:00 ET. The opening range is recorded in `gate_state_snapshots` but NOT used as an active entry gate.

## Status

**Tested, not adopted** -- passed all-3-strategy criteria at ib60_buf5 but outperformed by Leledc (for MNQ) and prior-day levels (for MES). Combined IB+Leledc fails MES IS threshold.

## Test Results

**Script**: `backtesting_engine/strategies/sr_initial_balance_sweep.py`
**Date**: March 6, 2026
**Data**: MNQ+MES 1-min bars, 12.5 months, chronological IS/OOS split

### Full Sweep: ib_period x buffer_pts (10 configs)

Two configs pass all-3-strategy criteria:
- **ib60_buf2**: All 3 MARGINAL. Portfolio OOS Sharpe 1.392 (+13% over baseline)
- **ib60_buf5**: vScalpA STRONG, vScalpB+MES MARGINAL. Portfolio OOS Sharpe 1.459 (+18% over baseline)

30-min IB universally weaker than 60-min IB.

### Decision Log Entry

IB was not explicitly rejected in the decision log -- it passed but was superseded by Leledc and prior-day levels. From `decision_log.md`, Mar 6:
> **REJECTED** VWAP Z-Score, Squeeze (TTM), Intraday Pivots, combined IB+Leledc.

The combined IB+Leledc was rejected; IB standalone passed but wasn't adopted.

## Backtest Priority

**Low** -- IB passed testing but is superseded by adopted filters. Revisit only if:
1. Leledc is removed/changed and we need a replacement MNQ filter
2. We want to test IB extensions (1.5x, 2.0x) as additional levels
3. "Require consensus" gating (block only when BOTH IB and Leledc agree) -- this was identified as an untested approach in the research
4. Direction-aware IB gating (block longs near IB high only, not near IB low)
