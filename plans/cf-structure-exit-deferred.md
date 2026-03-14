# DEFERRED C1: vScalpC Structure Exit in CF Simulator

**Status**: DEFERRED (revisit after structure exit params stabilize, ~4 weeks or 50+ vScalpC blocked signals)
**Date**: March 14, 2026

**Why deferred**: vScalpC started paper trading March 13. Structure exit params (LB=50, PR=2, buffer=2pts) may change based on paper trading results. Only ~4-8 blocked signals will accumulate before params stabilize. The CF engine currently simulates vScalpC with `tp2_pts=60` (the OCO crash-safety cap) and `be_time_bars=0`, which means the runner rides until TP2=60, SL/BE, or EOD — missing the structure exit that would typically fire earlier. Results can be reprocessed with `--force` once this fix ships.

---

## Current Behavior (without fix)

In `engine.py:374-375`, vScalpC blocked signals are simulated with:
```python
effective_be_time = 0 if cfg.structure_exit_type else cfg.breakeven_after_bars
```

This correctly skips BE_TIME (which vScalpC doesn't use), but `simulate_partial_exit()` has no structure exit logic. The runner leg exits only via TP2=60 (crash cap), SL/BE, or EOD. In reality, most vScalpC runners exit via structure (pivot-based swing level) well before the 60pt cap. CF results for vScalpC are therefore **knowingly inaccurate** — runners are held too long, overstating both wins and losses.

---

## Three Implementation Options

### Option A: Import IncrementalSwingTracker (Architect + Senior Dev recommendation)

Import `IncrementalSwingTracker` from `engine.structure_monitor` — zero external dependencies, already parity-tested against the vectorized `compute_swing_levels()` on the full 12.5-month MNQ CSV.

**How it works**:
1. Before simulating exit, instantiate `IncrementalSwingTracker(lookback=50, pivot_right=2)`
2. Warm up with ALL available pre-entry bars (not just 50) — the tracker needs `lookback + pivot_right + 1 = 53` bars minimum to confirm its first pivot, but feeding more history produces better swing levels at entry time
3. In `simulate_partial_exit`, after leg 2 checks SL/BE but BEFORE BE_TIME, capture `swing_high_prev`/`swing_low_prev` BEFORE calling `tracker.update(highs[i], lows[i])`, then check structure exit against `prev_bar.close`
4. Exit priority in leg 2: SL/BE > TP2_cap(60) > Structure > (BE_TIME skipped, =0 for vScalpC)

**Estimated change**: ~50-60 lines added to `exit_simulator.py` (new `simulate_partial_exit_structure()` function or parameter extension to existing function).

**Pros**:
- Single source of truth — same `IncrementalSwingTracker` used by live engine
- No code duplication, no drift risk
- Already parity-tested against vectorized backtest implementation

**Cons**:
- Couples CF simulator (`agents/counterfactual/`) to live engine module (`engine/structure_monitor.py`)
- Import path: `from live_trading.engine.structure_monitor import IncrementalSwingTracker` (works because CF already imports from `engine.config`)

### Option B: Vectorized swing detection (Creative recommendation)

Pure numpy function, ~25 lines, computes `swing_high[i]`/`swing_low[i]` arrays for the entire bar series in one pass. Identical logic to `compute_swing_levels()` in `backtesting_engine/strategies/structure_exit_common.py`.

**How it works**:
1. Before the exit loop, compute swing level arrays over the full bar data: `sh, sl = compute_swing_levels_vec(highs, lows, lookback=50, pivot_right=2)`
2. In the leg 2 exit loop, check `closes[i-1] >= sh[i-1] - buffer` (long) or `closes[i-1] <= sl[i-1] + buffer` (short)
3. No warm-up concept — the vectorized function handles the initial NaN window internally

**Estimated change**: ~30 lines (standalone function + check in exit loop).

**Pros**:
- Simpler implementation, no state management
- No dependency on live engine module
- Natural fit for CF's array-based design (all data already loaded as numpy arrays)
- Easier to test — pure function, deterministic on same inputs

**Cons**:
- Third implementation of swing detection (backtest vectorized, live incremental, CF vectorized) — must validate parity independently
- If swing detection logic changes, three places to update
- Slightly different code path than live engine (vectorized vs incremental), though mathematically identical

### Option C: Use TP2=25 as approximation (simple fallback)

Use the pre-structure-exit fixed TP2=25 value as a rough proxy for structure exit behavior. Quick parameter change, zero implementation effort.

**Pros**:
- Instant — change one number in engine.py
- No new code to test or maintain

**Cons**:
- Knowingly inaccurate — structure exit was adopted specifically because it's better than fixed TP2=25
- Structure exit is adaptive (follows swing levels), TP2=25 is static
- Defeats the purpose of CF accuracy for vScalpC
- Would need to be replaced anyway when structure exit params stabilize

---

## Recommended Approach: Option A

Import is the cleanest path — one source of truth, already parity-tested, no risk of implementations drifting apart. The coupling to `engine.structure_monitor` is acceptable because the CF engine already imports from `engine.config` (same package boundary).

Option B is viable if coupling to the live engine module is undesirable (e.g., if structure_monitor gains heavy dependencies in the future). In that case, copy the ~25-line vectorized function into `exit_simulator.py` and add a parity test.

Option C is not recommended — it sacrifices accuracy for convenience and would need replacement.

---

## Key Implementation Details

### Warm-up strategy
Feed ALL available pre-entry bars to the tracker, not just the minimum 53. The Databento CSV has the full session (~400+ bars per day), plus prior days. More history means the tracker has confirmed swing levels ready at entry time, matching how the live engine operates (tracker runs from session start, long before any entry).

### bar[i-1] convention
This is critical and must match the live engine (`structure_monitor.py:206-208`):
1. Capture `swing_high_prev = tracker.swing_high` and `swing_low_prev = tracker.swing_low`
2. THEN call `tracker.update(highs[i], lows[i])`
3. Check exit: `closes[i-1] >= swing_high_prev - buffer` (long) or `closes[i-1] <= swing_low_prev + buffer` (short)
4. Fill at `opens[i]`

This ensures the exit decision uses swing levels as of bar[i-1], not bar[i] — no look-ahead.

### Exit priority in leg 2
```
SL/BE         — prev bar close breaches stop → fill at next open
TP2_cap(60)   — prev bar close reaches 60pt cap → fill at next open
Structure     — prev bar close reaches swing level - buffer → fill at next open
(BE_TIME      — skipped, =0 for vScalpC)
```

Structure exit sits AFTER TP2 cap check but BEFORE BE_TIME. In practice, structure fires much earlier than the 60pt cap, so priority between TP2 and Structure rarely matters. But if both trigger on the same bar, TP2 cap wins (matches the live engine where the exchange OCO bracket would fill before the structure monitor fires).

### No min_profit_pts guard
The live engine's `StructureExitMonitor.check_exit()` does not enforce a minimum profit threshold — it exits whenever close reaches the swing level. The CF simulator must match this behavior. Structure exits near entry price are valid (the swing level itself provides the profit logic).

### Short-side handling
Short runner: exit when `closes[i-1] <= swing_low_prev + buffer`. The tracker tracks both swing_high and swing_low simultaneously, so short-side exits work with no additional logic.

### Structure exit only fires on leg 2 (runner)
Leg 1 (scalp) always exits via TP1 or SL. Structure exit only applies to the runner leg after TP1 has filled. This is already enforced by the leg separation in `simulate_partial_exit`.

### No confirmed swing level yet
If the tracker hasn't confirmed any pivot yet (first 53 bars of warm-up), `swing_high`/`swing_low` are `None`. Structure exit doesn't fire — the runner falls through to TP2 cap, SL/BE, or EOD. This matches the live engine behavior.

---

## Validation Plan

### 1. Differential test (primary)
Run both `run_backtest_structure_exit()` (from `backtesting_engine/strategies/structure_exit_common.py`) and the new CF `simulate_partial_exit` with structure exit on every vScalpC entry from the 12.5-month MNQ CSV. Compare exit_reason, exit_bar, and pnl_pts. Any mismatch is a bug.

**Expected**: exact match on all 469 trades (12-month sample from MEMORY.md).

### 2. Swing tracker parity test
Run `IncrementalSwingTracker` bar-by-bar and `compute_swing_levels()` vectorized on 500+ bars of MNQ data. Assert `swing_high[i]` and `swing_low[i]` are identical at every bar. This test already exists conceptually (parity was tested during structure_monitor development) but should be formalized.

### 3. Edge case tests
- **Early-session entry**: Entry in first 53 bars of session — no confirmed swing level → runner should fall through to TP2/SL/EOD
- **SL/BE vs structure same-bar priority**: If both SL/BE and structure trigger on the same bar, SL/BE must win
- **Short-side exit**: Verify `swing_low + buffer` logic is correct for short runners
- **60pt cap reached before structure**: If price moves 60pts before any swing level is reached, TP2 cap fires (not structure)
- **Structure fires at a loss**: Entry at 21000, swing_high at 20998 with buffer=2 → target = 20996. Close >= 20996 triggers exit at a loss. This is valid behavior — verify CF matches.

---

## Trigger to Revisit

Whichever comes first:
- **50+ vScalpC blocked signals accumulate** (est. ~4-6 weeks from March 14, based on ~5-25% block rate and 1-3 entries/day)
- **Structure exit params confirmed stable** through paper trading (10+ days acceptance criteria met per `memory/MEMORY.md` open items)

When triggered:
1. Re-read this document
2. Confirm structure exit params haven't changed from LB=50, PR=2, buffer=2pts
3. Implement Option A (or Option B if circumstances changed)
4. Run validation plan
5. Reprocess all existing vScalpC blocked signals with `--force`
6. Update `plans/counterfactual-engine.md` to remove C1 from deferred items
