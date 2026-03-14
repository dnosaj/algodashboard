---
name: Counterfactual blocked trade calculation — design requirements
description: Robust P&L calculation for blocked signals, accounting for cooldown, deduplication, and sequential fill simulation. Requested Mar 13, 2026.
type: project
---

# Counterfactual Blocked Trade Calculation

**Why:** Jason needs to know if gates are helping or hurting. The current `blocked_signals` table has `cf_exit_price` and `cf_exit_reason` columns but they are NULL — never populated. The `gate_effectiveness` view shows block counts but no P&L impact.

**How to apply:** This is a prerequisite for meaningful gate analysis in the Digest Agent and for Frontier to evaluate gate threshold changes.

## Requirements (from Jason, Mar 13)

1. **Deduplication**: Multiple blocked signals in a 3-5 minute window → only the FIRST one counts. The others would not have been taken because cooldown would be active after the first entry.

2. **Cooldown simulation**: After a counterfactual fill, apply the strategy's cooldown (CD=20 bars for MNQ, CD=25 for MES) before the next counterfactual entry is eligible. This matches how the live engine works.

3. **Sequential fill simulation**: Process blocked signals chronologically. For each:
   - Is cooldown expired since last (real or counterfactual) trade? If no, skip.
   - Simulate entry at signal price → apply TP/SL exit logic → record counterfactual P&L.
   - Start cooldown timer.

4. **Exit simulation**: Use the same logic as the backtest engine — next-bar-open fill after entry, TP/SL based on strategy config. For partial-exit strategies (vScalpC, MES_V2), simulate both legs.

5. **Track and persist**: Write `cf_exit_price`, `cf_exit_reason`, `cf_pnl_net`, `cf_bars_held` back to the `blocked_signals` table. Null = not yet calculated. This enables the `gate_effectiveness` view to show actual P&L impact.

## Complexity Notes

- The backtest engine (`v10_test_common.py`) already has the TP/SL exit simulation logic. Could reuse `run_backtest_v10` or extract the exit logic.
- Need access to 1-min bar data after the blocked signal time to simulate the exit. Options:
  - (a) Run counterfactual batch replay nightly using local Databento CSVs (most accurate, requires local data)
  - (b) Engine computes counterfactual in real-time as bars arrive (adds complexity to hot path — probably not)
  - (c) Digest Agent queries bars from Supabase (but we decided not to store 1-min bars)
- Option (a) is most aligned with the architecture. Could be a step in the overnight pipeline (after EOD, before morning).

## Open Questions

- Should counterfactual simulation use signal_price or next-bar-open after signal? (Backtest uses next-bar-open.)
- For partial-exit strategies, simulate both TP1 and TP2/runner? Or just TP1 for simplicity?
- How far back should we backfill? All historical blocked signals, or rolling 30 days?
