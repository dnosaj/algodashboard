---
name: Liquidity Sweep / Stop Hunt Detectors
description: Detect price sweeping prior swing highs/lows and reversing -- the "stop hunt" pattern. Potential exit signal or entry gate.
type: reference
---

# Liquidity Sweep / Stop Hunt Detectors

## Source

Multiple authors surveyed:
- **Quantura** (TradingView)
- **wateriskey6689** (TradingView)
- **H1 Liquidity Sweep Tracker** (TradingView)

No single dominant implementation. This is a category of indicators rather than one specific script.

## Discovery Context

Surveyed Mar 6, 2026 as part of the TradingView open-source indicator research. Noted as "interesting for MES v2 (longer hold time). Not useful for MNQ TP=5 scalps (too slow)."

## What It Does

Detects the liquidity sweep pattern:
1. Identifies a prior swing high or swing low (recent pivot)
2. Watches for price to briefly break above/below that level (sweeping the stops placed there)
3. Detects the reversal back inside the range (the "trap")

The concept comes from ICT/Smart Money methodology: institutional players push price through known stop-loss clusters to fill large orders, then reverse.

### Signal Logic
- **Bullish sweep**: Price dips below a prior swing low, triggers stops, then closes back above the level. Potential long entry.
- **Bearish sweep**: Price spikes above a prior swing high, triggers stops, then closes back below the level. Potential short entry or exit signal for longs.

## Parameters

Vary by implementation, but typically:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Swing Lookback | (varies) | Bars to identify swing H/L |
| Sweep Threshold | (varies) | How far past the level price must go |
| Reversal Confirmation | (varies) | How quickly price must reverse |
| Minimum Sweep Duration | (varies) | Bars the sweep must last |

## Code

No Pine Script extracted during survey. Multiple implementations exist on TradingView. Python implementation would require:
1. Swing point detection (same as Market Structure CHoCH/BOS)
2. Level tracking (maintain list of unswept swing H/L)
3. Sweep detection (price crosses level)
4. Reversal confirmation (price returns back within N bars)

## Analysis

### Useful

- **Captures a real institutional pattern.** Stop hunts are a documented phenomenon in futures markets. Sweeps of prior swing H/L followed by reversals happen frequently on NQ/ES.
- **Could serve as exit signal for runners.** If our vScalpC runner is long and price sweeps a prior high then reverses, exiting before the reversal deepens.
- **Could serve as entry gate.** Wait for a liquidity sweep before entering -- enter the reversal rather than chasing the initial move.

### Flawed

- **Too slow for MNQ scalps.** TP=3-7 trades are done in minutes. A liquidity sweep pattern takes many bars to develop, confirm, and reverse. By the time the sweep is detected, our scalp is already closed.
- **Complex to implement reliably.** Needs swing detection, level management, sweep detection, and reversal confirmation. Many edge cases (partial sweeps, failed reversals, overlapping levels).
- **Confirmation lag.** Detecting the reversal requires waiting for the close back inside the range. By then, the opportunity may be stale for our timeframe.
- **vScalpC already has structure exit.** Our pivot-based structure exit (LB=50, PR=2) already exits when price reaches swing levels. Liquidity sweep detection would be a more sophisticated version of the same concept.

### Related Indicators

- **vScalpC Structure Exit (adopted)**: Uses pivot-based swing levels for runner exits. Conceptually overlapping.
- **Market Structure CHoCH/BOS**: Uses the same swing point infrastructure. CHoCH is essentially what happens after a liquidity sweep fails.
- **Prior-Day Level Gate (adopted for MES)**: Blocks entries near prior-day levels. Sweep detection is the dynamic intraday version of the same concept.

### Correlation with Our Existing Indicators

- **SM**: Moderate correlation. A bearish liquidity sweep (fake breakout above, then reversal) often coincides with SM flipping to negative.
- **vScalpC structure monitor**: High overlap. Both track swing levels and react when price reaches them.

## Fit With Our System

### Option 1: Exit Signal (vScalpC runner)
Exit runner when a bearish sweep is detected (price sweeps a prior high and reverses). Would need to compare with current structure exit (LB=50, PR=2, buffer=2pts) which already covers similar ground.

### Option 2: Entry Gate (MES v2)
Wait for a liquidity sweep pattern before allowing entry. Could improve MES v2 entries by entering after the "trap" reversal. But MES already has prior-day level gate and entry cutoff.

### Option 3: Entry Confirmation (post-sweep entry)
Only enter after a sweep + reversal pattern. High-conviction entries but very few signals.

## Status

**Untested** -- not backtested. Not mentioned in the decision log.

## Test Results

*(None)*

## Backtest Priority

**Low** -- Three concerns: (1) Too slow for our scalp timeframe (TP=3-7 on MNQ). (2) Complex implementation with many edge cases. (3) Significant overlap with our existing vScalpC structure exit and prior-day level gate. Only worth exploring if MES v2 or vScalpC runner performance degrades and we need a more sophisticated exit mechanism.
