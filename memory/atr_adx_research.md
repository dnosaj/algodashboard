# ATR / ADX Research

## Completed: Basic Entry Filter Sweep (Mar 9)

**Script**: `backtesting_engine/strategies/atr_adx_entry_sweep.py`

Tested ATR and ADX at entry as entry gates across V15, vScalpB, MES_V2.
- Timeframes: 1-min, 5-min mapped to 1-min
- Periods: 7, 10, 14, 20, 30
- Phase 1: Correlation with trade P&L + quintile breakdown
- Phase 2: Threshold sweep with IS/OOS validation

**Result: FAIL.** No actionable entry filter found.
- ATR mildly predictive for MES_V2 (higher vol = better, r=+0.12) — intuitive for TP=20 needing movement. But improvement marginal (+3-6% PF).
- ADX: no useful signal for any strategy.
- vScalpB immune to both (SM_T=0.25 already filters well).
- Best combo (MES_V2 ATR>p50 on 5min p=30) only +6.5% OOS PF — not worth blocking 15% of trades.

## Test Queue (prioritized)

### 1. ATR-Scaled Exits (HIGH PRIORITY)
**Idea**: Replace fixed SL/TP with ATR-multiple exits. SL = N × ATR, TP = M × ATR.
**Why**: Fixed SL ignores volatility regime. vScalpB 15pt SL gets clipped in 3 bars 68% of the time — in high ATR, 15pts is noise. ATR-scaled SL adapts automatically.
**Sweep plan**:
- ATR periods: [7, 10, 14, 20] on 1-min and 5-min
- SL multiples: [1.0, 1.5, 2.0, 2.5, 3.0] × ATR
- TP multiples: [0.5, 1.0, 1.5, 2.0, 2.5] × ATR
- Cap SL at reasonable max (e.g., 50pts MNQ, 40pts MES) to avoid blow-up bars
- Baseline: current fixed SL/TP for each strategy
- Validate: IS/OOS split, compare PF + Sharpe + MaxDD

### 2. Relative ATR — Expansion/Contraction Ratio (MEDIUM PRIORITY)
**Idea**: ATR(fast) / ATR(slow) as volatility regime signal.
**Why**: Different from absolute ATR. Ratio > 1.3 = expansion (breakout starting). Ratio < 0.7 = contraction (squeeze). This is what good squeeze detectors actually measure.
**Sweep plan**:
- Fast/slow pairs: [5/20, 7/30, 10/30, 7/20, 10/50]
- Entry gate: only enter when ratio > threshold (expansion) or < threshold (contraction)
- Expansion thresholds: [1.0, 1.2, 1.3, 1.5, 2.0]
- Contraction thresholds: [0.5, 0.6, 0.7, 0.8, 0.9]
- Also test as exit signal: exit when ratio crosses below 0.8 (contraction after entry during expansion)

### 3. DI+/DI- Alignment with Trade Direction (MEDIUM PRIORITY)
**Idea**: Require DI+ > DI- for LONG entries, DI- > DI+ for SHORT entries. Directional confirmation.
**Why**: ADX measures strength but not direction. SM says LONG but if DI- > DI+ the trend is actually bearish. Alignment could filter false signals.
**Sweep plan**:
- ADX/DI periods: [7, 10, 14, 20] on 1-min and 5-min
- Gate: only enter if DI spread aligns with SM direction
- Also test: minimum DI spread (|DI+ - DI-|) > threshold for conviction
- DI spread thresholds: [0, 5, 10, 15, 20]

### 4. ADX Slope as Exit Signal for MES_V2 (MEDIUM PRIORITY)
**Idea**: Exit MES_V2 when ADX is falling (trend exhaustion) instead of fixed BE_TIME=75 bars.
**Why**: MES_V2 has the stale trade problem. BE_TIME is a blunt instrument. Falling ADX detects trend exhaustion adaptively.
**Sweep plan**:
- ADX periods: [7, 10, 14, 20] on 1-min and 5-min
- Exit when: ADX[i] < ADX[i-N] (falling over N bars)
- N (lookback for slope): [3, 5, 10, 15, 20]
- Also test: exit when ADX drops below absolute threshold after being above it
- Thresholds: [15, 20, 25, 30]
- Compare vs current BE_TIME=75 baseline

### 5. ADX Rate of Change as Entry Filter (LOW PRIORITY)
**Idea**: Rising ADX = trend strengthening = good entry. Not the level, but the direction.
**Why**: ADX going 15→25 is bullish for trend-following even though starting level was low. The sweep tested levels only, not momentum.
**Sweep plan**:
- ADX periods: [10, 14, 20] on 5-min
- Rising = ADX[i] > ADX[i-N] where N = [3, 5, 10]
- Gate: only enter if ADX is rising
- Also test: ADX acceleration (rising faster)

### 6. Time-of-Day Normalized ATR (LOW PRIORITY)
**Idea**: Normalize ATR by time-of-day average. A 10pt ATR at 10AM is normal; at 2PM it's extreme.
**Why**: Raw ATR is confounded by intraday volatility patterns. Normalized ATR isolates "unusually volatile for this time" which is a better signal.
**Sweep plan**:
- Compute rolling 20-day average ATR for each 1-min time slot (e.g., avg ATR at 10:01, 10:02, etc.)
- Z-score = (current ATR - time_avg) / time_std
- Gate entries when z-score > threshold or < threshold
- Thresholds: [-2, -1, -0.5, 0, 0.5, 1, 2]

## Completed: ATR-Scaled Exits (Mar 9) — Test #1

**Script**: `backtesting_engine/strategies/atr_scaled_exits_sweep.py`

Custom backtest loop replacing fixed SL/TP with ATR-multiple exits. ATR periods [10,14,20] on 1-min, SL mults [1.0-4.0], TP mults [0.5-3.0].

**vScalpA**: Best ATR(20) SL=3.0x TP=2.0x → effective ~SL=50 TP=30. Sharpe 1.96 vs 1.48 (+32%), P&L +$4,206 vs +$1,948. OOS PF > IS PF (1.40 vs 1.21). BUT WR 84.5%→66.4%, MaxDD doubles. ATR values all hit caps — really just "use bigger fixed exits."
**vScalpB**: Best ATR(20) SL=4.0x TP=0.5x → effective ~SL=50 TP=8. IS/OOS solid (ratio 1.03). Key insight: SL=15 is ~1.0x ATR — too tight.
**MES_V2**: FAIL. 1-min ATR (mean 3.5pts) wrong scale for 75-bar holds. All OOS PF < 1.0.
**Verdict**: Not implementing ATR scaling. Real insight = current fixed exits may not be optimal (V15 TP too small, vScalpB SL too tight). Consider direct fixed exit sweep.

## Completed: ATR Expansion Ratio (Mar 9) — Test #2

**Script**: `backtesting_engine/strategies/atr_ratio_entry_sweep.py`

ATR(fast)/ATR(slow) ratio as entry gate. 5 pairs × 2 timeframes × 12 thresholds.

**vScalpA**: One STRONG PASS: contraction gate ATR(10/50) < 1.0 on 1-min. IS +7.2%, OOS +7.7% PF. But blocks 26% trades for ~$500 marginal gain.
**vScalpB**: FAIL. No config passed IS/OOS.
**MES_V2**: Marginal expansion gate ATR(10/50) > 1.0 on 5-min. +1% OOS PF — noise.
**Verdict**: FAIL. Not implementing.

## Completed: DI+/DI- Alignment (Mar 9) — Test #3

**Script**: `backtesting_engine/strategies/di_alignment_entry_sweep.py`

Custom backtest loop requiring DI+ > DI- for LONG, DI- > DI+ for SHORT. Periods [7,10,14,20] on 1-min and 5-min, spread thresholds [0-20].

**All strategies**: FAIL. DI alignment redundant with SM signal. 1-min DI too noisy, 5-min configs block too many trades. No config passed IS/OOS.
**Verdict**: FAIL. Not implementing.

## Structural Lessons (all tests combined)
- ATR/vol-normalization at entry doesn't help (confirmed 3 times now)
- ATR-scaled exits don't work as dynamic adapters — they're just selecting larger fixed exits
- DI alignment is redundant with SM momentum direction
- The positive ATR correlation for MES_V2 makes sense: TP=20 needs price movement
- vScalpB's SM_T=0.25 already selects high-conviction entries that are uncorrelated with volatility
- **V15 TP=5 may be suboptimal** — ATR sweep showed TP=20-30 with SL=50 doubles P&L (but halves WR)
- **vScalpB SL=15 is too tight** at ~1.0x mean ATR — wider SL lets more trades reach TP
- MES_V2's fixed exits (SL=35, TP=20) are well-calibrated for its hold time
- 1-min ATR is the wrong volatility scale for MES_V2 (multi-hour holds). Would need 5-min or daily ATR.
- Exit optimization > entry filtering for ATR/ADX

## Completed: Fixed Exit Sweep (Mar 9)

**Script**: `backtesting_engine/strategies/fixed_exit_sweep.py`

Full TP×SL grid for V15 (108 combos) and vScalpB (90 combos) with IS/OOS validation.

**V15 findings**:
- TP=7 SL=40: Sharpe 2.73 (vs 2.08), PF 1.36 (vs 1.29), MaxDD -$523 (vs -$570). STRONG IS/OOS.
- TP=7 SL=50: Sharpe 2.93, PF 1.41, MaxDD -$869. STRONG IS/OOS (OOS PF > IS PF).
- Runner cluster (TP 25+): avg Sharpe 1.69, best 2.91. Foundation for vScalpC.
- TP=5 is NOT optimal. TP=7 is the sweet spot for scalp exits.

**vScalpB findings**:
- TP=3 SL=10 dominates everything: Sharpe 3.29 (vs 1.49), PF 1.47 (vs 1.19), MaxDD -$281 (vs -$502). STRONG IS/OOS.
- SM_T=0.25 entries are quick pops — tighter exits capture more reliably.
- Runner TPs (25+) completely fail for vScalpB. Pure scalper only.

**Action items**: V15 TP=7 upgrade, vScalpB TP=3/SL=10 upgrade, vScalpC runner design. Details: `memory/vscalpc_runner_design.md`
