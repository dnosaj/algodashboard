---
name: ICT Concepts (Smart Money Concepts)
description: All-in-one ICT implementation — BOS/CHoCH, order blocks, FVGs, liquidity sweeps, fib OTE. Primary candidate for trade forensics and structural context.
type: reference
---

# ICT Concepts (Smart Money Concepts)

## Source

- **TradingView**: "ICT Concepts [LuxAlgo]" (open source, ~123K likes) + "ICT Concepts [UAlgo]" (invite-only)
- **Python library**: `joshyattridge/smart-money-concepts` (GitHub) — full pandas/numpy implementation
- **Pine Script source**: [GitHub Gist by niquedegraaff](https://gist.github.com/niquedegraaff/8c2f45dc73519458afeae14b0096d719) — 1,129 lines
- **Author of methodology**: Michael Huddlestone (Inner Circle Trader)

## Discovery Context

Jason identified this as the framework for understanding why trades win/lose relative to market structure (Mar 14, 2026). Goal: map our SM+RSI entries against ICT structural features to find patterns in our losers.

## What It Does

Multiple ICT concepts in one framework:

### 1. Market Structure: BOS and MSS/CHoCH

**BOS (Break of Structure)**: Price closes beyond a swing point in the trend direction → continuation signal.
**CHoCH/MSS (Change of Character / Market Structure Shift)**: Price closes beyond a swing point against the trend → reversal signal.

Detection uses two timeframes:
- **Internal structure** (5-bar lookback) — minor swings
- **Swing structure** (50-bar lookback) — major swings

```python
# Swing detection (simplified from LuxAlgo)
# A swing high is confirmed when high[len] > highest of surrounding bars
# BOS: close crosses above stored swing high while trend is already bullish
# CHoCH: close crosses above stored swing high while trend was bearish (reversal)
```

### 2. Order Blocks

The last opposing candle before a displacement move. Created at BOS/CHoCH events.

- **Bullish OB**: Last bearish candle before a strong move up → future support zone
- **Bearish OB**: Last bullish candle before a strong move down → future resistance zone
- **Size filter**: Candle range must be < 2x ATR(200) to qualify
- **Mitigated**: Price touches the OB zone (wick enters)
- **Broken**: Close moves through the OB zone completely → OB removed

### 3. Fair Value Gaps (FVG)

3-bar imbalance pattern — no overlap between candle 1 and candle 3 shadows.

```python
# Bullish FVG: candle_3_low > candle_1_high (gap up)
bullish_fvg = (high.shift(1) < low.shift(-1)) & (close > open)

# Bearish FVG: candle_3_high < candle_1_low (gap down)
bearish_fvg = (low.shift(1) > high.shift(-1)) & (close < open)
```

**Auto-threshold**: Only counts FVGs where middle candle body % > 2x cumulative average (filters noise).
**Mitigated**: Price fills the gap → FVG removed.

### 4. BPR and CE

- **BPR (Balanced Price Range)**: Overlap zone between a bullish and bearish FVG → equilibrium point
- **CE (Consequent Encroachment)**: Midpoint of any FVG → acts as price magnet

### 5. Liquidity Sweeps

Price temporarily breaches a swing high/low (taking stops) then closes back inside.

```python
# Buy-side liquidity sweep
sweep_up = (high > prev_high) and (close < prev_high)  # wick above, close back inside

# Sell-side liquidity sweep
sweep_down = (low < prev_low) and (close > prev_low)   # wick below, close back inside
```

### 6. Fibonacci / OTE

ICT-specific Fibonacci levels with the Optimal Trade Entry zone:

| Level | Name |
|-------|------|
| 0.0 | Swing extreme |
| 0.236 | Discount zone |
| 0.382 | Discount zone |
| 0.5 | Equilibrium |
| 0.618 | OTE zone start |
| 0.705 | **Optimal Trade Entry** |
| 0.79 | OTE zone end |
| 1.0 | Swing extreme |

**Premium zone**: Above 0.5 (overpriced for longs)
**Discount zone**: Below 0.5 (underpriced for longs)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Internal swing length | 5 | Bars for minor structure detection |
| Swing structure length | 50 | Bars for major structure detection |
| OB filter method | ATR(200) | Filter noisy candles |
| OB max shown | 5 | Per direction |
| FVG threshold | Auto (2x cumulative mean) | Min displacement size |
| Liquidity lookback | 10 | Bars to find prior swing for sweep detection |

## Code

### Python library (ready to use)

```bash
pip install smartmoneyconcepts
```

```python
from smartmoneyconcepts import smc
import pandas as pd

# Requires DataFrame with columns: open, high, low, close, volume
df = pd.DataFrame(...)

# Swing highs and lows
swing_highs_lows = smc.swing_highs_lows(df, swing_length=50)

# BOS and CHoCH
bos_choch = smc.bos_choch(df, swing_highs_lows)

# Order blocks
order_blocks = smc.ob(df, swing_highs_lows)

# Fair value gaps
fvg = smc.fvg(df)

# Liquidity sweeps
liquidity = smc.liquidity(df, swing_highs_lows)

# Fibonacci retracements
retracements = smc.retracements(df, swing_highs_lows)
```

### Custom implementation (if library doesn't fit)

FVG is trivial (~10 lines numpy). Order blocks need swing detection first. BOS/CHoCH uses the same swing infrastructure our structure exit already has (PivotExitTracker with LB=50).

## Analysis

### Useful

- **Comprehensive framework** — covers S/R (OBs), imbalance (FVGs), momentum (displacement), and structure (BOS/CHoCH) in one coherent system
- **Python library exists** — `smartmoneyconcepts` gives us pandas/numpy implementation ready for our backtest data
- **Our existing infrastructure overlaps significantly** — pivot swing detection (structure_monitor.py), prior-day levels (safety_manager.py), VWAP tracking are all partial ICT implementations
- **FVGs are computationally trivial** — 3-bar pattern, no lookback lag, no repainting
- **Order blocks are meaningful S/R** — based on actual price displacement, not arbitrary levels
- **Confluence approach is novel for us** — we've always tested features in isolation; ICT's value is in stacking (OB + FVG + fib OTE = high probability zone)

### Flawed

- **Internal structure (5-bar) is too noisy for 1-min NQ** — would fire constantly. Only swing structure (50-bar) is viable.
- **50-bar swing = 50-minute confirmation delay** — acceptable for forensics but slow for real-time entry decisions
- **Swing point selection for fibs is subjective** — which swing high/low do you measure from? Different choices give different OTE zones. The Python library handles this algorithmically but the choice of `swing_length` matters enormously.
- **Liquidity sweep detection is simplistic** — `high > prevHigh and close < prevHigh` catches many false positives on volatile NQ. No volume confirmation.
- **No displacement quantification** — ICT methodology says OBs need "strong displacement" but the code only uses ATR-based candle size, not momentum measurement.
- **Our research history: every level-based filter has been marginal** — VWAP Z-Score rejected, IB superseded, only VPOC+VAL survived for MES. But we've never tested confluence of multiple structural features.

### Related Indicators

- **Our PivotExitTracker** (structure_monitor.py): Same swing detection, LB=50, already parity-tested
- **Prior-day levels** (safety_manager.py): VPOC, VAL — ICT-equivalent institutional reference levels
- **Leledc exhaustion**: Partial overlap with displacement detection
- **Fair Value Gap** (already in indicator library): standalone FVG analysis
- **Market Structure CHoCH** (already in indicator library): standalone BOS/CHoCH

### Correlation with Our Existing Indicators

- **SM (Squeeze Momentum)**: SM detects momentum shifts; BOS/CHoCH detects structural shifts. Should be complementary — SM fires on momentum, structure confirms direction. Need to test correlation.
- **Prior-day levels**: Our levels are daily-anchored (VPOC, VAL). ICT order blocks are intraday-anchored. Different time horizons, likely low correlation.
- **PivotExitTracker**: Same swing detection infrastructure (LB=50, prominence-based). Could potentially share code.

### Instrument / Timeframe Notes

- **1-min NQ/MNQ**: Internal structure (5-bar) too noisy. Swing structure (50-bar) viable. FVGs viable with auto-threshold filter. Order blocks viable at swing timeframe.
- **FVGs on 1-min may be too numerous** — the auto-threshold filter is critical. May want to compute on 5-min bars and project onto 1-min (same approach suggested in the original indicator survey for FVGs).
- **Liquidity sweeps need longer lookback on NQ** — default 10 bars (10 min) is too short. 30-60 bars more appropriate for NQ's volatility.

### Parameter Sensitivity

- **Swing length (50 default)**: Most important. Shorter = more structure events but noisier. Longer = fewer, more meaningful. Our structure exit uses 50 — keep consistent.
- **FVG auto-threshold**: Critical for filtering 1-min noise. Test with and without.
- **OB ATR filter period (200)**: Standard. Our SM already uses 200-bar normalization.
- **Liquidity lookback (10)**: Too short for NQ 1-min. Sweep [20, 30, 50, 100].

### Computational Cost

- **FVG**: O(1) per bar — trivial 3-bar pattern check
- **Swing detection**: O(n) with deque — same as our PivotExitTracker, already proven fast
- **Order blocks**: O(k) per structure break where k = bars between swings — infrequent events
- **Liquidity sweeps**: O(lookback) per bar — rolling max/min
- **Overall**: Very lightweight. No concern for live engine or backtest.

## Fit With Our System

### Phase 3 (Forensics First) — PRIMARY USE

Map every trade in our 12.8-month backtest against ICT structural context at entry:
- Was the entry near a bearish/bullish order block?
- Was the entry inside an unmitigated FVG?
- Did a liquidity sweep just occur?
- What was the BOS/MSS state (trending or just reversed)?
- Was the entry in the fib OTE zone (62-79%) of the most recent swing?
- Was the entry in premium or discount zone?

Then: **Do our SL trades cluster at specific structural features?** If yes, those features become gate candidates.

### Phase 1 (Backtest) — AFTER forensics reveals patterns

Test promising structural features as pre-filters with IS/OOS validation. Same methodology as all our other gates.

### Phase 2 (Implementation) — IF validated

Two possible implementations:
- **Entry gate**: Block entries at adverse structural features (e.g., don't go long at a bearish OB)
- **Entry confirmation**: Only enter when structure supports (e.g., require BOS in trade direction)

## UAlgo vs Python Library Comparison

The UAlgo Pine Script code (provided by Jason, Mar 14) implements ICT differently from the Python library in several key ways.

### Feature Matrix

| Feature | UAlgo Pine Script | Python `smartmoneyconcepts` | Key Difference |
|---------|-------------------|----------------------------|----------------|
| **BOS/MSS** | `ta.pivothigh/low` with configurable pivot length (default 5). Tracks trend state, labels BOS (continuation) vs MSS (reversal). | `smc.bos_choch()` uses 4-point swing pattern matching. | UAlgo is simpler/more direct. Both produce equivalent signals. |
| **Order Blocks** | **3-bar engulfing pattern**: bar[2] opposing, bar[1] engulfing, bar[0] confirming. Independent of BOS events. Uses `request.security` for MTF OBs. Overlap filter prevents stacking. Max 2 active per side. | Tied to BOS/CHoCH events. Searches backwards from structure break for candle with extreme H/L. ATR size filter. | **Fundamentally different detection.** UAlgo = pattern-based (simpler). Python = structure-break-based (more contextual). Both valid but will produce different OBs. |
| **FVG** | `low > high[2]` / `high < low[2]`. No auto-threshold. Tracks mitigation (remove when filled). Max 3 active per side. | Same 3-bar logic. Optional `join_consecutive` merges adjacent FVGs. | Equivalent core logic. UAlgo has no displacement filter — may be noisier on 1-min. |
| **BPR** | Built-in. Actively finds overlapping bullish+bearish FVGs, computes intersection zone. | **Not implemented.** | UAlgo advantage. Would need ~20 lines custom code for Python. |
| **CE** | Built-in as display mode. Midpoint of FVG zones with dedicated line + box. | **Not implemented.** | UAlgo advantage. Trivial to add: `(top + bottom) / 2`. |
| **Liquidity Sweeps** | 20-bar pivot lookback. Sweep = wick beyond pivot but close back inside. Visual: line from pivot to sweep bar + highlight box. | Cluster-based: groups swing H/L within `range_percent` tolerance. | Different approaches. UAlgo is simpler/more intuitive. |
| **Fibonacci/OTE** | **Structure-anchored**: auto-draws from `lastStructBreakExtreme` to `lastStructOppPrice`. Updates with each new BOS/MSS. Levels: 0.382, 0.5 (EQ), 0.618, 0.705, 0.79 (OTE). Two custom extras. | `smc.retracements()` gives raw pullback %, no fib level computation. | UAlgo is much more complete. Python would need ~40 lines custom fib code. |
| **SMT Divergence** | Full implementation. Compares pivot H/L of current symbol vs user-selected symbol. Bearish SMT = symbol A higher high + symbol B lower high. Mini chart panel. | **Not implemented.** | UAlgo advantage. Would need ~60-80 lines for Python. Could compare MNQ vs MES or NQ vs ES. |
| **Killzones** | Asian (20:00-00:00), London (02:00-05:00), NY AM (08:30-11:00), NY PM (13:30-16:00). UTC-5. Box overlay with H/L range lines. | `smc.sessions()` supports same 4 killzones + custom. | Equivalent. |
| **MTF Support** | OBs use `request.security` for multi-timeframe detection. | No built-in MTF. Must resample DataFrame manually. | UAlgo advantage for OBs. For our use: compute on 5-min/15-min DataFrame, project onto 1-min. |
| **Mitigation Tracking** | Both OBs and FVGs tracked — removed when price fills them. | Both return `MitigatedIndex`. | Equivalent. |
| **Lookahead Bias** | Uses `[2]`, `[1]`, `[0]` bar references — no forward look. Pivots use `ta.pivothigh(len, len)` which confirms `len` bars late (causal). | **HAS LOOKAHEAD BIAS** in `fvg()` (`shift(-1)`) and `swing_highs_lows()` (centered window). Open PR #95 for `causal=True` fix NOT merged. | **CRITICAL**: Python library needs patching before backtesting use. UAlgo is clean. |

### UAlgo Order Block Detection (key logic)

```pine
detectLogic() =>
    // Bullish OB: bar[2] bearish, bar[1] bullish engulfing, bar[0] bullish confirming
    bool isBull = open[2] > close[2]       // bar[2] is bearish (red)
                  and close[1] > open[1]    // bar[1] is bullish (green) — the OB candle
                  and close > open           // bar[0] confirms direction
                  and low[1] < low[2]        // bar[1] sweeps below bar[2] low
                  and close > high[1]        // bar[0] closes above bar[1] high (displacement)

    // Bearish OB: mirror logic
    bool isBear = open[2] < close[2]       // bar[2] is bullish
                  and close[1] < open[1]    // bar[1] is bearish — the OB candle
                  and close < open           // bar[0] confirms
                  and high[1] > high[2]      // bar[1] sweeps above bar[2] high
                  and close < low[1]         // bar[0] closes below bar[1] low

    [isBull, high[1], low[1], time[1], isBear, high[1], low[1], time[1]]
    // OB zone = [low[1], high[1]] of the middle engulfing candle
```

This is simpler than the Python library's approach and more aligned with how ICT practitioners actually identify OBs — a clear rejection + displacement pattern, not tied to higher-level structure breaks.

### UAlgo Fibonacci (structure-anchored)

```pine
// Fibs auto-anchor to the last structure break
float a0 = lastStructBreakExtreme  // the extreme of the impulse move
float a1 = lastStructOppPrice      // the swing point before it
float r = a1 - a0                  // range
// Levels computed as: a0 + r * level
// EQ = 0.5, OTE zone = 0.618 to 0.79
```

This solves the "which swing points?" subjectivity problem — fibs always anchor to the most recent confirmed structure break.

### Recommendation for Our Implementation

| Feature | Use UAlgo Logic | Use Python Library | Custom Build |
|---------|----------------|-------------------|--------------|
| **FVG** | | Y (with causal fix) | |
| **BPR** | | | Y (~20 lines) |
| **CE** | | | Y (~3 lines) |
| **Order Blocks** | **Y** (3-bar pattern is cleaner) | | Port to Python (~40 lines) |
| **BOS/MSS** | **Y** (pivot-based, matches our infrastructure) | | Use our PivotExitTracker |
| **Fib/OTE** | **Y** (structure-anchored) | | Port to Python (~50 lines) |
| **Liquidity Sweeps** | **Y** (20-bar pivot, simpler) | | Port to Python (~30 lines) |
| **SMT Divergence** | **Y** | | Port to Python (~60 lines) |
| **Killzones** | | Y | |

### Multi-Timeframe Approach (Jason's insight)

ICT is fundamentally HTF→LTF. For our system:
1. Compute BOS/MSS + OBs + FVGs on **5-min bars** (resample from 1-min)
2. Project those structural zones onto the **1-min chart** where our entries happen
3. At entry time, check: is this 1-min entry inside/near a 5-min OB? A 5-min FVG? In the OTE zone of a 5-min structure break?

This mirrors how we already handle RSI (computed on 5-min, mapped to 1-min). Our infrastructure supports this pattern.

## Status

**Untested** — forensic analysis needed first. Python library available (needs causal fix). UAlgo Pine Script source available for porting key functions.

## Test Results

*(To be filled after forensic analysis)*

## Backtest Priority

**HIGH** — This is the next major research initiative. Forensics-first approach (3→1→2). The Python library + our existing swing infrastructure makes implementation fast. The confluence hypothesis (stacking multiple ICT features) has never been tested in our system.
