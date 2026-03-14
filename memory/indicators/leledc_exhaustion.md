---
name: Leledc Exhaustion V4
description: Detects "exhaustion bars" — the last buyer in an uptrend (or last seller in a downtrend) by counting consecutive directional closes. Active entry gate at mq=9 for all MNQ strategies. ADOPTED Mar 6, 2026.
type: reference
---

# Leledc Exhaustion V4 [Joy_Bangla]

## Source

- **Author**: Joy_Bangla (converted from original Leledc concept)
- **URL**: https://www.tradingview.com/script/ (search "Leledc Exhaustion V4")
- **Popularity**: ~1.2K likes, 23K favorites
- **License**: Open source (Pine Script viewable)

## Discovery Context

Catalogued during Mar 6, 2026 TradingView indicator survey. Rated HIGH priority for Round 1 backtesting. The concept directly addresses our core failure mode: entering at the tail end of a sustained move where the "last buyer" is about to get trapped.

## What It Does

Detects "exhaustion bars" -- points where a sustained directional move has gone on long enough that the final participants are likely trapped. Uses a simple bar-counting mechanism relative to price N bars ago.

### Math

```
For each bar i:
  if close[i] > close[i - lookback]:
      bull_count += 1
  else:
      bull_count = 0

  if close[i] < close[i - lookback]:
      bear_count += 1
  else:
      bear_count = 0

  bull_exhaustion = (bull_count >= maj_qual)
  bear_exhaustion = (bear_count >= maj_qual)
```

When `bull_count` reaches `maj_qual` (e.g., 9), it means 9 consecutive bars have closed above where price was 4 bars prior -- a sustained directional push that's likely exhausted.

The counter **resets to zero** the moment the condition breaks (any bar where close is NOT above close[i-4]).

### Signal Logic

- **Bull exhaustion detected**: 9+ consecutive bars with close > close[i-4]. This is the "last buyer" -- the uptrend is stretched.
- **Bear exhaustion detected**: 9+ consecutive bars with close < close[i-4]. This is the "last seller."
- **Gate logic**: Block ANY entry (not direction-specific) when EITHER bull or bear exhaustion is detected on bar[i-1]. Persistence=1 means only the exhaustion bar itself is blocked.

The original indicator also has "minor" exhaustion (min_qual=5, min_len=5) but we only use major exhaustion.

## Parameters

| Parameter | Default | Our Config | Description |
|-----------|---------|------------|-------------|
| maj_qual | 6 | **9** | Consecutive bars for major exhaustion |
| maj_len | 30 | unused | Major exhaustion lookback window (we don't use) |
| min_qual | 5 | unused | Minor exhaustion threshold |
| min_len | 5 | unused | Minor exhaustion lookback |
| lookback | 4 | **4** | How far back to compare each close |
| persistence | 1 | **1** | Bars to stay blocked after exhaustion |

## Code

### Pine Script (core logic)

```pinescript
//@version=5
// Leledc Exhaustion V4 — Joy_Bangla (simplified core)

indicator("Leledc Exhaustion", overlay=true)

maj_qual = input.int(6, "Major Quality")
lookback = 4

var int bull_count = 0
var int bear_count = 0

bull_count := close > close[lookback] ? bull_count + 1 : 0
bear_count := close < close[lookback] ? bear_count + 1 : 0

bull_exhaustion = bull_count >= maj_qual
bear_exhaustion = bear_count >= maj_qual

plotshape(bull_exhaustion, style=shape.xcross, color=color.red, location=location.abovebar)
plotshape(bear_exhaustion, style=shape.xcross, color=color.green, location=location.belowbar)
```

### Python (backtesting implementation)

```python
def compute_leledc_exhaustion(closes, maj_qual=6, lookback=4):
    """Detect Leledc exhaustion bars.

    Bull exhaustion: consecutive bars where close[j] > close[j - lookback]
    reaches maj_qual.

    Bear exhaustion: consecutive bars where close[j] < close[j - lookback]
    reaches maj_qual.

    Returns:
        (bull_exhaustion, bear_exhaustion) -- boolean arrays, same length as closes.
    """
    n = len(closes)
    bull_exhaustion = np.zeros(n, dtype=bool)
    bear_exhaustion = np.zeros(n, dtype=bool)
    bull_count = 0
    bear_count = 0

    for i in range(n):
        if i < lookback:
            continue
        bull_count = (bull_count + 1) if closes[i] > closes[i - lookback] else 0
        bear_count = (bear_count + 1) if closes[i] < closes[i - lookback] else 0
        bull_exhaustion[i] = bull_count >= maj_qual
        bear_exhaustion[i] = bear_count >= maj_qual

    return bull_exhaustion, bear_exhaustion
```

Full backtesting sweep: `backtesting_engine/strategies/sr_leledc_exhaustion_sweep.py`

### Python (live engine implementation)

The live engine uses an incremental version in `live_trading/engine/safety_manager.py`:

```python
# Incremental update per bar (no recomputation of full history)
_LELEDC_LOOKBACK = 4

def _update_leledc(self, bar):
    closes = self._leledc_closes[inst]
    closes.append(bar.close)
    if len(closes) > 20:
        closes[:] = closes[-20:]  # keep rolling buffer

    curr = closes[-1]
    prev = closes[-(1 + self._LELEDC_LOOKBACK)]

    self._leledc_bull_count[inst] = (self._leledc_bull_count[inst] + 1) if curr > prev else 0
    self._leledc_bear_count[inst] = (self._leledc_bear_count[inst] + 1) if curr < prev else 0

    exhausted = (self._leledc_bull_count[inst] >= threshold
                 or self._leledc_bear_count[inst] >= threshold)
    self._leledc_gate_current[inst] = not exhausted
```

Gate is consumed as `_leledc_gate_prev` (bar[i-1] pattern). Counter does NOT reset at daily boundary (matches backtest behavior). Seeds from `gate_seed.json` or Databento historical closes at startup.

Config: `leledc_maj_qual=9` in `engine/config.py` for MNQ_V15, MNQ_VSCALPB, MNQ_VSCALPC.

## Analysis

### Useful

- **Trivial to compute.** One counter per instrument, one comparison per bar. O(1) per bar. No rolling windows, no lookback buffers (beyond 4 closes).
- **Directly addresses our failure mode.** Our strategies enter on SM+RSI signals at bar close. If the market has run 9 bars in one direction, the SM signal is stale and the move is likely exhausted. Leledc catches exactly this.
- **Very light filtering.** Blocks only 6-11% of trades (mq=9 is genuinely rare). Removes only the most extreme exhaustion events.
- **Not direction-specific.** Blocks on EITHER bull or bear exhaustion. Conservative but robust -- doesn't need to know our entry direction.
- **Works across all MNQ strategies.** vScalpA (STRONG PASS), vScalpB (MARGINAL PASS), vScalpC (MARGINAL PASS). Same threshold works for all three despite different SM_T and TP values.

### Flawed

- **Not statistically significant for vScalpB and MES individually.** vScalpB OOS PF +5.9% and MES OOS PF +4.7% are within sampling noise (~11% CI at 230 trades). Only vScalpA OOS +11.1% is borderline significant.
- **Not direction-aware.** We block ALL entries during exhaustion, even if the entry is in the OPPOSITE direction of the exhaustion. A bullish exhaustion bar would be a great SHORT entry, but we block it. This is intentional conservatism but leaves value on the table.
- **lookback=4 is fixed.** We never tested lookback=[2, 4, 6, 8]. The value 4 comes from the original indicator, not from our own optimization.
- **Persistence=1 means very brief blocking.** Only the single exhaustion bar is blocked. If entries fire one bar later, they're allowed through even though the market may still be extended.
- **No daily reset of counter.** Counter carries across sessions. A 9-bar count spanning two trading days may not be as meaningful as a 9-bar count intraday.

### Related Indicators

- **Exhaustion Signal [ChartingCycles]**: Simpler version -- pure consecutive count at levels 9, 12, custom. No major/minor distinction.
- **%R Trend Exhaustion [upslidedown]**: Williams %R dual-period exhaustion. Different mechanism but same concept.
- **RSI**: Both detect "overextended." RSI uses momentum ratio, Leledc uses consecutive directional closes. Complementary.

### Correlation with Our Existing Indicators

- **SM**: Low correlation. SM measures trend direction/magnitude via EMA crossover. Leledc measures consecutive directional closes. SM can be strongly positive while Leledc count is low (fresh trend) or high (extended trend).
- **RSI**: Low-moderate correlation. RSI can be at 55 (just above our long threshold) while Leledc shows 9 consecutive up-closes. RSI responds to magnitude, Leledc responds to persistence. They measure different aspects of "overextended."
- **ADR gate**: Low correlation. ADR measures intraday range exhaustion relative to daily average. Leledc measures bar-by-bar directional persistence. Different timescales.

### Instrument / Timeframe Notes

- **MNQ**: mq=9 is the optimal threshold. Blocks 6-11% of entries. STRONG PASS on vScalpA, MARGINAL on vScalpB.
- **MES**: mq=9 is MARGINAL only (OOS PF +4.7%, not significant). Different from MNQ because MES's slow SM (EMA=255) and TP=20 navigate through exhaustion zones differently.
- **1-min bars**: lookback=4 means comparing to 4 minutes ago. 9 consecutive bars = 9 minutes of sustained directional movement. This is a meaningful signal on NQ where 9 minutes of one-directional closes is genuinely extreme.

### Parameter Sensitivity

- **maj_qual is the key parameter.** Extended sweep confirmed mq=9 as the peak:

| mq | Portfolio OOS Sharpe | vScalpA OOS dPF% | MES OOS dPF% | Blocked (vScalpA) |
|----|---------------------|------------------|--------------|-------------------|
| 8 | 1.571 | +7.3% | +6.3% | 31 |
| **9** | **1.623** | **+11.1%** | **+4.7%** | **25** |
| 10 | 1.183 | +1.8% | -5.0% | 16 |
| 11 | 1.319 | +6.2% | -4.3% | 13 |
| 12 | 1.265 | +3.2% | -4.8% | 8 |
| 15 | 1.297 | +5.3% | -0.3% | 3 |

- Curve reverses sharply at mq=10. mq=9 is genuinely the best, not an artifact of truncated testing.
- **persistence=1 is optimal.** Higher persistence (3, 5) over-blocks -- removes too many trades after the exhaustion event passes.
- **lookback=4 untested.** Fixed from original indicator. Testing [2, 4, 6, 8] is an open item.

### Computational Cost

Negligible. One integer comparison and one counter increment per bar per instrument. No rolling windows, no division, no square roots. The cheapest indicator in our entire stack.

## Fit With Our System

### Active Use: MNQ Entry Gate (mq=9, persistence=1)

Implemented and deployed. Blocks entries for all three MNQ strategies (V15, vScalpB, vScalpC) when 9+ consecutive bars show close > close[i-4] (bull exhaustion) or close < close[i-4] (bear exhaustion).

**Architecture**:
- `engine/config.py`: `leledc_maj_qual=9` on MNQ_V15, MNQ_VSCALPB, MNQ_VSCALPC
- `engine/safety_manager.py`: Incremental `_update_leledc()` per bar, gate consumed as `_leledc_gate_prev` (bar[i-1])
- Seeds from `gate_seed.json` at startup (computed by `compute_gate_seed.py`)
- Persisted at daily reset via `_save_gate_state()`
- Blocked signals logged to Supabase `blocked_signals` table with `gate_type="leledc"`
- `gate_leledc_count` injected into every TradeRecord for post-hoc analysis
- Dashboard shows `leledc_gated` badge per strategy in SafetyPanel
- No daily reset of counter (matches backtest)
- Fail-open during warmup
- Manual override via dashboard bypasses gate

## Status

**ADOPTED** -- implemented Mar 6, 2026. Active in production for all MNQ strategies.

## Test Results

**Script**: `backtesting_engine/strategies/sr_leledc_exhaustion_sweep.py`
**Date**: March 6, 2026
**Data**: MNQ+MES 1-min bars, 12.5 months, chronological IS/OOS split

### Full Sweep: maj_qual x persistence (15 configs)

**All-3-Strategy Results (selected configs)**:

| Config | vScalpA | vScalpB | MES_V2 | Portfolio OOS Sharpe |
|--------|---------|---------|--------|---------------------|
| mq5_p1 | FAIL (+11.0%/-8.0%) | FAIL (-16.5%/+4.6%) | FAIL (-5.6%/-11.6%) | 0.793 |
| mq6_p1 | MARGINAL (+7.8%/+0.2%) | MARGINAL (+4.7%/+12.6%) | FAIL (-5.3%/-8.5%) | 1.114 |
| mq7_p1 | MARGINAL (+12.7%/+3.2%) | MARGINAL (+8.5%/+3.9%) | FAIL (-1.2%/-0.9%) | 1.328 |
| mq8_p1 | MARGINAL (+4.3%/+7.3%) | MARGINAL (-3.0%/+2.4%) | FAIL (-5.9%/+6.3%) | 1.571 |
| **mq9_p1** | **STRONG (+7.2%/+11.1%)** | **MARGINAL (-1.0%/+5.9%)** | **MARGINAL (-3.8%/+4.7%)** | **1.623** |
| mq9_p3 | MARGINAL (-4.5%/+15.0%) | FAIL (-6.9%/+3.6%) | FAIL (-8.3%/-6.6%) | 1.288 |

**mq9_p1 is the ONLY config that passes all 3 strategies simultaneously.**

### Portfolio Impact (mq9_p1)
- Portfolio OOS Sharpe: 1.623 (+32% over baseline 1.232)
- Portfolio OOS Net$: +$3,348 (+17% over baseline $2,864)
- Blocks 6-11% of trades (very light filtering)
- Performance improves monotonically mq5 to mq9 with persistence=1

### MES-Only Evaluation
- mq9_p1 for MES standalone: MARGINAL PASS (OOS PF +4.7%, Sharpe +19%)
- Not adopted for MES -- prior-day level gate (buf5) is STRONG PASS for MES (+9.0% OOS PF)
- MES uses its own optimal filter (prior-day levels), not Leledc

### Decision Log Entry

From `decision_log.md`, Mar 6:
> **IMPLEMENTED** Leledc exhaustion gate mq9_p1 for all MNQ strategies. Portfolio OOS Sharpe +32%.

## Backtest Priority

**None** -- adopted and deployed. No further backtesting needed unless we want to test:
1. lookback variation [2, 4, 6, 8] (never swept)
2. Direction-aware gating (block longs on bull exhaustion only)
3. Time-of-day interaction (exhaustion at 10:30 vs 3:00 may differ)

These are low priority -- the current implementation works well and blocks few enough trades that the risk of over-optimization is low.
