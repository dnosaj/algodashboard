# v10 Feature Candidates — Updated Feb 13, 2026

Based on v9/v9.4 forward testing, Python backtests, and MES/MNQ trade analysis.

## Current State

- **v9**: Original validated strategy (DO NOT modify)
- **v9.4**: v9 + max loss stop via fully-gated strategy.close() (current best for MNQ)
- Forward testing on MNQ + MES since Feb 13

## Critical Pine Script Constraints

Any v10 feature MUST follow these rules or it will break the strategy:
1. NO extra stateful functions (ta.rsi, ta.ema, etc.) unless they change trade behavior
2. ALL optional code MUST be inside `if feature_enabled` blocks — Pine does NOT short-circuit
3. Use `strategy.close()` not `strategy.exit()` for stops that need episode flags to persist
4. Never access `strategy.position_avg_price` outside an `if` guard
5. New strategy name = TradingView resets Properties — always verify date range

---

## TIER 1: Proven / High Confidence

### 1a. Max Loss Stop (DONE in v9.4)
- **Status**: Implemented and validated
- MNQ: 50pts, improves PF from 1.529 to 1.649, cuts MaxDD from $471 to $421
- MES: DISABLED (any value hurts — MES moves are too small for fixed-point stops)
- Uses strategy.close() inside `if max_loss_pts > 0` block

### 1b. SM Flip Reversal Entry
- **Concept**: When SM flips and exits a trade, immediately enter in the new direction
  (instead of waiting for a fresh RSI cross)
- **Evidence**: Feb 13 MES — after Trade 2 exit at 14:56 (SM flip bear), price dropped
  another 32 pts to 6833.75. That short was free money but the strategy missed it
  because it requires an RSI cross for entry.
- **Why it could work**: The SM flip IS the signal. If SM just flipped bearish from bullish,
  that's the strongest directional signal the indicator produces. Requiring an additional
  RSI cross adds lag and misses the move.
- **Risk**: SM flip reversals near EOD (15:45+) could enter positions with no time to play out.
  Need to respect the entry window.
- **Implementation**: On SM flip, if in_entry_window and cooldown_ok, enter the opposite
  direction. No RSI cross required. Episode flag for the new direction gets set.
- **Test plan**: Python backtest first — add reversal entries to v9 backtest, measure PF/WR.
  Only implement in Pine if Python shows improvement.

---

## TIER 2: Promising / Needs Testing

### 2a. RSI Momentum Direction at Entry
- **Concept**: Only enter if 5-min RSI is moving in trade direction (not just crossed level)
- **Evidence**: Feb 13 both MNQ and MES — winning trades entered with RSI rising,
  losing trades entered with RSI already falling.
  - Trade 1 (winner): RSI 59.36 and rising -> 63 -> 65 -> 70
  - Trade 2 (loser): RSI 53.28 and falling -> 47 -> 45 -> 41 -> 25
- **Why it could work**: RSI cross fires on the 5-min bar, but the 1-min entry happens
  1-5 bars later. If momentum reversed in that window, the trade is already stale.
- **PINE DANGER**: Cannot add ta.rsi() or any extra indicator. Must use the existing
  rsi_5m and rsi_5m_prev values. Check: rsi_5m > rsi_5m_prev for longs (RSI rising),
  rsi_5m < rsi_5m_prev for shorts (RSI falling).
- **Implementation**: This only uses values already computed (rsi_5m, rsi_5m_prev).
  No new stateful functions. Should be safe as a boolean condition on entry.
  But verify: with max_loss=0, results must match v9 baseline exactly.
- **Test plan**: Python backtest first. If promising, add to Pine with regression test.

### 2b. Cross-Instrument SM Confirmation
- **Concept**: When running on MNQ, also read MES SM (or vice versa). Only enter when
  both instruments' SM agrees on direction.
- **Evidence**: Feb 13 — at MNQ Trade 16's worst point, MNQ SM was +0.37 (bullish)
  but MES SM had already flipped to -0.003 (bearish). Cross-checking would have
  provided an early warning.
- **Why it could work**: Different instruments seeing the same flow direction = higher
  conviction. When they disagree, one instrument's SM is wrong — skip the trade.
- **Implementation**: request.security() on the other instrument's SM.
  Requires AlgoAlpha on both charts or native SM computation.
  Pine: `sm_other = request.security("MES1!", "1", nz(sm_buy_other) + nz(sm_sell_other))`
  — but this requires input.source() which only works on the current chart's indicators.
  Would need native SM computation for the other instrument.
- **Complexity**: HIGH. Requires computing SM natively (PVI/NVI) for a different instrument
  via request.security(). Volume data availability is a concern.
- **Test plan**: Python backtest with both MNQ and MES data aligned by timestamp.
  Check: how often do they disagree? What's the win rate when they agree vs disagree?

### 2c. Adaptive Stop Loss (ATR-based)
- **Concept**: Instead of fixed-point stop, use ATR to scale stop to instrument volatility
- **Evidence**: 50pt stop works on MNQ but hurts MES. ATR would auto-adapt.
- **Implementation**: Compute ATR(14) on 1-min bars, set stop at 3x ATR.
  MNQ 1-min ATR ~3-5pts -> stop = 9-15pts (tighter than 50pt, more adaptive)
  MES 1-min ATR ~1-2pts -> stop = 3-6pts
- **PINE DANGER**: ta.atr() is a stateful function. MUST be inside `if use_atr_stop` block.
  When disabled, ta.atr() must never execute.
- **Test plan**: Python first with ATR from data. Compare vs fixed stops on both instruments.

---

## TIER 3: Investigated / Lower Priority

### 3a. HTF SM Direction Filter (15-min/1-hour)
- **Status**: Investigated Feb 13. 15-min SM was too slow to help.
- **Finding**: 15-min SM stayed bullish ALL DAY on Feb 13 even during the selloff.
  SM_15m went from +0.30 to +0.22 while price dropped 200 pts.
- **Conclusion**: Higher TF SM is even more lagging. Does NOT detect profit-taking
  or intraday reversals. May work for multi-day trends but not intraday scalping.
- **Action**: DEPRIORITIZED unless more data shows otherwise.

### 3b. Daily Loss Limit / Circuit Breaker
- **Concept**: Stop trading after X% daily loss
- **Why**: Safety net for bad days. Feb 13 MNQ lost ~$455 on one trade.
- **Implementation**: Track daily equity, pause entries if drawdown exceeds threshold.
  Uses strategy.equity and time("D") — no stateful functions needed.
- **Test plan**: Low urgency — add after core improvements.

### 3c. Time-of-Day Filter
- **Concept**: Different behavior for morning (10:00-12:00) vs afternoon (12:00-15:45)
- **Evidence**: Feb 13 — morning trades were clean, afternoon trades were losers on
  both MNQ and MES. Could tighten stops or reduce position in afternoon.
- **Complexity**: Would need to analyze more days to see if this pattern holds.
  1 day of data is not enough to conclude "afternoons are worse."
- **Test plan**: Analyze all v9 backtest trades by hour. If clear hourly pattern exists,
  implement as a simple time gate or afternoon-specific stop tightening.

---

## REJECTED / Do Not Implement

### RSI Confirmation on 1-min
- Adding ta.rsi(close, rsi_len) on 1-min causes Pine side effects even behind toggles
- Does not block any trades (same trade count on 5-min bars)
- Tested and FAILED — degrades results

### SM Momentum Filter
- Requiring SM to be trending in trade direction (SM[0] > SM[5])
- Python backtest: PF drops from 2.325 to ~1.7
- Tested and HURTS — cuts good trades more than bad ones

### strategy.exit() for stops
- Fires silently, does not interact with episode flags
- Causes re-entry in same SM episode (52 trades vs 49)
- Use strategy.close() instead for any stop that should "spend" the episode

---

## Testing Order

1. **RSI momentum direction** (Tier 2a) — lowest risk, uses existing variables
2. **SM flip reversal entry** (Tier 1b) — addresses a known missed-opportunity pattern
3. **Cross-instrument SM** (Tier 2b) — novel signal, needs data alignment work
4. **ATR adaptive stop** (Tier 2c) — nice-to-have, mainly for multi-instrument deployment
5. Everything else — wait for more forward test data before prioritizing

## Testing Protocol

For EVERY feature:
1. Python backtest first (strategies/scalp_v9_smflip_backtest.py as base)
2. Must show improvement on Jan 19 - Feb 12 in-sample data
3. Must not degrade Feb 13 forward test results
4. Pine implementation: verify max_loss=0 + feature OFF = identical to v9 (49 trades)
5. Pine implementation: verify feature ON does not introduce stateful side effects
6. Create new version file (v9.5, v10, etc.) — never overwrite existing versions
