# MES v9.4 Exit Research (Feb 20, 2026)

## Context

MES v9.4 uses SM flip as its only exit. During paper trading on Feb 20, a live
trade showed SM staying positive while RSI crashed below 34 — price gave back
all gains while SM was obliviously bullish. SM(20/12/400/255) with EMA=255 is
intentionally slow for entries but exits lag badly on momentum reversals.

Baseline: 359 trades, PF 1.228, WR 54.9%, +$1,129 (6 months, Databento 1-min)

## RSI-Based Exits — REJECTED

### Scripts
- `v94_rsi_exit_test.py` — v1, simple 50/50 split (flawed, Sept poisoned early half)
- `v94_rsi_exit_test_v2.py` — v2, LOMO + walk-forward + counterfactual

### Configs Tested (17 total)
- **RSI cross exits**: Exit when 5-min RSI crosses below 35/40/45 (longs) or above 55/60/65 (shorts)
- **Profit-gated variants**: Only fire RSI exit when trade is currently profitable
- **RSI drop from peak**: Track RSI peak during trade, exit when RSI fell 15/20/25 pts from peak
- **Combos**: cross + drop, with and without profit gating

### Results
- **Best RSI config (R4: cross<45, profit-gated)**: 3/7 LOMO wins, only 9 exits over 6 months
- **EOD 15:30** emerged as strongest: 4/7 LOMO wins, avg dPF +0.109
- RSI drop from peak: mostly hurt performance
- **Core finding**: RSI exit cuts winners roughly as often as it saves losers

### Counterfactual (R4)
9 RSI exits fired: 6/9 better than SM flip, net delta +$58.75. But trade #7
cost $62.50 by cutting a big winner. Too few firings to be reliable.

## Hold Time Analysis — INFORMATIVE BUT NOT ACTIONABLE

### Script: `v94_exit_deep_dive.py`, `v94_hold_time_sweep.py`

### Findings
- MES trades capture only **17% of MFE** overall
- 31+ bar trades (118 of 359) capture only 7-12% of MFE, net -$51
- Of 118 trades past 30 bars: 58 helped by holding, 53 hurt, net +$236 (coin flip)
- Max hold sweep: best at 40 bars ($1,155 vs $1,129 baseline) — marginal
- **September 2025**: 68 trades, 41% WR, shorts 33% WR, SM signals near zero

### Conditional Losing Hold Cap — Post-Filter vs Pre-Filter Trap

**Post-filter simulation** (from frozen baseline curves) looked great:
- Cap=25: 70 trades exit early, Saved $1,911, Lost $1,222, **Net +$688**
- Cap=30: 61 trades exit early, Saved $1,478, Lost $911, **Net +$567**

**Pre-filter validation** (actual backtest, LOMO): ALL FAILED
- LC25: 1/7 LOMO wins, PF 1.044, -$918 vs baseline
- LC30: 2/7 LOMO wins, PF 1.167, -$399 vs baseline
- LC35: 2/7 LOMO wins, PF 1.208, -$201 vs baseline

### Why Post-Filter ≠ Pre-Filter (Re-Entry Analysis)

**Script**: `v94_reentry_analysis.py`

The losing cap creates a feedback loop:
1. Cut loser at bar 25 — **SM still agrees 98-100% of the time** (hasn't flipped)
2. Cooldown starts ~31 bars earlier than baseline (avg)
3. Creates **13 phantom trades** (LC25) that only exist because of early exit
4. Phantom trades win only **22-31%**, total P&L -$104 to -$125
5. LC25 creates 366 trades vs 359 baseline (7 extra, mostly losers)

### Directional Suppression — ZERO EFFECT

**Script**: `v94_losecap_directional.py`

Hypothesis: block same-direction re-entries after losing cap exit until SM flips.
Result: **Zero trades suppressed.** LC25+SUPP = LC25 exactly.

Reason: The engine's `long_used`/`short_used` flags already prevent same-direction
re-entry within the same SM cycle. `long_used` resets on `sm_flipped_bull`, which
is the same event that would clear the suppression flag. They're completely redundant.

The same-direction re-entries in the data (47 for LC25) happen in **new** SM cycles
(SM flipped bear then back to bull), not within the same cycle.

## EOD 15:30 — VALIDATED AND IMPLEMENTED

### LOMO Results
- **4/7 PF wins** (best of all configs)
- **Only config with positive avg dPF** (+0.109)
- Same 359 trades (no phantom entries)
- PF 1.228 → 1.266, net +$59

### Why It Works
- Closes 20 more trades as EOD (38 vs 18) — captures late-session mean reversion
- **Doesn't create new entries** — no cooldown cascade, no phantom trades
- November was the one bad month (-$425 delta) — other 6 months break even or improve

### Implementation
`session_close_et="15:30"` added to `MES_V94` config in `live_trading/engine/config.py`

### MH40+EOD (Runner-Up)
Max hold 40 + EOD 15:30: best absolute delta (+$104) but only 3/7 LOMO wins,
avg dPF -0.043. Not robust enough to implement.

## Key Lessons

1. **Post-filter ≠ Pre-filter.** This is now the THIRD time this pattern has appeared
   (SM threshold for MES, SM death zone band exclusion, losing hold cap). Simulating
   from frozen baseline curves always overstates the benefit because early exits shift
   cooldowns and create new (usually worse) entries.

2. **`long_used`/`short_used` already prevent same-cycle re-entry.** Any directional
   suppression scheme is redundant with the engine's existing episode tracking.

3. **MES captures only 17% of MFE.** The exit problem is real but no indicator-based
   or mechanical exit improves on SM flip except EOD timing.

4. **EOD 15:30 works because it has no entry-side effects.** It just closes positions
   earlier. Every other exit modifier creates a cascading cooldown problem.

## Scripts Created
| File | Purpose |
|------|---------|
| `v94_rsi_exit_test.py` | RSI exit v1 (simple split, flawed) |
| `v94_rsi_exit_test_v2.py` | RSI exit v2 (LOMO, walk-forward, counterfactual) |
| `v94_exit_deep_dive.py` | Hold time, MFE, September analysis, exit ideas |
| `v94_hold_time_sweep.py` | Granular hold sweep, P&L curves, conditional cap |
| `v94_lomo_validation.py` | LOMO validation of all promising configs |
| `v94_reentry_analysis.py` | Re-entry pattern analysis after losing cap |
| `v94_losecap_directional.py` | Directional suppression test (no effect) |
