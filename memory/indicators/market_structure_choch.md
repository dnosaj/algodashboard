---
name: Market Structure CHoCH/BOS
description: Swing-based and fractal-based Break of Structure (BOS) and Change of Character (CHoCH) detection. Potential directional gate or exit signal.
type: reference
---

# Market Structure CHoCH/BOS

## Source

Two variants surveyed:

### mickes — Market Structure
- **Author**: mickes (TradingView)
- **Popularity**: ~3.2K likes, 18K boosts
- **License**: Open source (Pine Script viewable)

### LuxAlgo — Market Structure CHoCH/BOS (Fractal)
- **Author**: LuxAlgo (TradingView)
- **Popularity**: ~12K likes, 3.8K boosts
- **License**: Open source (Pine Script viewable)
- **Note**: LuxAlgo also has a full SMC indicator (~123K likes) but that is far more complex (order blocks, FVGs, premium/discount zones). These two are the structure-only variants.

## Discovery Context

Surveyed Mar 6, 2026 as part of the TradingView open-source indicator research. Categorized as Round 3 (complex, lower priority) for backtesting. The survey noted our SM indicator already handles trend detection, making CHoCH/BOS partially redundant.

## What It Does

Detects two structural events using swing point analysis:

- **BOS (Break of Structure)**: Price breaks a prior swing high (bullish BOS) or swing low (bearish BOS). Confirms trend continuation.
- **CHoCH (Change of Character)**: Price breaks a swing point in the opposite direction of the prevailing trend. Signals potential trend reversal.

### mickes Variant (Swing-Based)
Uses ta.pivothigh / ta.pivotlow with configurable pivot lengths:
- Internal structure: pivot length 5 (short-term swings)
- Swing structure: pivot length 50 (major swings)
- Tracks HH/HL (bullish) and LH/LL (bearish) sequences
- Also detects equal H/L zones and liquidity levels
- ATR factor % for level proximity

### LuxAlgo Fractal Variant
Uses fractal pattern detection instead of swing pivots:
- Fractal highs/lows identified by the Length parameter
- More adaptive than fixed pivot lengths
- Can detect shifts faster but may generate more noise
- Includes S/R levels derived from structure

## Parameters

### mickes
| Parameter | Default | Description |
|-----------|---------|-------------|
| Internal Pivot Length | 5 | Short-term swing detection |
| Swing Pivot Length | 50 | Major swing detection |
| ATR Factor % | 10% | Proximity threshold for level classification |
| Liquidation Detection | on | Detect liquidity sweep zones |

### LuxAlgo Fractal
| Parameter | Default | Description |
|-----------|---------|-------------|
| Length | (varies) | Fractal pattern detection window |

## Code

No Pine Script extracted during survey. Both are viewable on TradingView. Python implementation would require:
1. Swing point detection (pivothigh/pivotlow equivalent)
2. HH/HL/LH/LL tracking state machine
3. BOS/CHoCH event classification based on which swing point is broken

## Analysis

### Useful

- **CHoCH as exit signal**: When trend reverses against our position, exit the runner. Could complement vScalpC structure exit.
- **BOS as entry confirmation**: Require BOS in entry direction before taking SM+RSI signal. Adds structural context.
- **Two granularities**: Internal (fast) vs swing (slow) structure captures both micro and macro shifts.

### Flawed

- **Complex to port to Python.** Swing point detection + state machine + BOS/CHoCH classification is non-trivial compared to our simple counter-based gates.
- **Our SM indicator already handles trend detection.** SM measures momentum direction and strength. Adding structure detection is potentially redundant.
- **Fractal-based variant may be noisy on 1-min.** Fractal patterns on high-frequency data produce many false signals.
- **Pivot length sensitivity**: Internal=5 on 1-min bars = 5-minute swings. Very noisy. Swing=50 = 50 minutes. More meaningful but slow.

### Related Indicators

- **Our SM (Squeeze Momentum)**: Already measures momentum direction. BOS/CHoCH adds structural confirmation but may not add information beyond what SM already provides.
- **vScalpC Structure Exit**: Uses pivot-based swing detection (LB=50, PR=2) for runner exits. Conceptually similar to the swing structure variant.
- **LuxAlgo SMC (full)**: Superset that includes order blocks, FVGs, and premium/discount zones.

### Correlation with Our Existing Indicators

- **SM**: High expected correlation. Both measure trend direction. CHoCH often coincides with SM zero-cross. Adding CHoCH on top of SM is likely redundant for entry gating.
- **RSI**: Low expected correlation. RSI measures overbought/oversold, structure measures trend breaks.

## Fit With Our System

### Option 1: CHoCH Entry Gate
Block entries when a CHoCH against entry direction occurred within N bars. Problem: our SM already captures trend reversals, so this likely adds little.

### Option 2: BOS Entry Confirmation
Require a BOS in entry direction before allowing entry. Problem: may filter too many trades (BOS confirmation often comes after the move starts).

### Option 3: CHoCH Exit Signal (for vScalpC runner)
Exit runner when CHoCH detected against position direction. We already have pivot-based structure exits for vScalpC. Would need to compare with current LB=50/PR=2 approach.

## Status

**Untested** -- categorized as Round 3 (low priority) in the Mar 6 survey. Never backtested.

## Test Results

*(None)*

## Backtest Priority

**Low** -- Our SM indicator already provides trend direction. The vScalpC structure exit already uses pivot-based swing detection. CHoCH/BOS would need to demonstrate value beyond what we already have. The implementation complexity (swing detection + state machine) is higher than simpler gates like Leledc.
