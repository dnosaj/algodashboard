# Heartbeat, Reconciliation, and Emergency Flatten Fixes (Mar 2, 2026)

All changes in a single file: `live_trading/engine/runner.py`

## What Was Broken

### 1. Reconciliation false alarms (300+ warnings/session)
`MockOrderManager.reconcile_positions()` returned positions keyed by `strategy_id`
(e.g., `MNQ_V15`, `MNQ_VSCALPB`), but the recon loop expected positions keyed by
`instrument` (e.g., `MNQ`, `MES`). Every recon cycle produced false mismatch warnings
because broker keys never matched strategy-side instrument keys.

### 2. Daily zombie engine after CME maintenance
CME equity futures close ~16:59 ET, reopen ~18:00 ET. Heartbeat fired at 17:04 ET
(90s after last bar), then emergency flatten at 17:04+300s. `trading_active` set to
`False` permanently. Engine streamed data but processed nothing for the rest of the
evening session.

### 3. No recovery after feed gap
After connection_timeout flatten, `trading_active = False` with no path back. Even
when the data feed resumed, the engine stayed dead until manual restart or API resume.

### 4. Emergency flatten risks reversing OCO-protected positions
Mid-session feed gap while positioned: emergency flatten sends a market close while
OCO brackets (SL + TP) are still resting on exchange. If the market close fills and
then the SL fills, the account ends up reversed.

## What Was Fixed

### Fix 1: CME Maintenance Window Suppression
- New helper `_in_cme_maintenance_window(state)` — returns True during 16:57-18:03 ET
- Uses `zoneinfo.ZoneInfo("America/New_York")` for automatic DST handling
- **Safety guard**: if ANY strategy has `position != 0`, suppression is disabled
  (means EOD close failed — heartbeat is the last safety net)
- Heartbeat loop `continue`s during the window — no warnings, no flatten
- Module-level `_ET` timezone and `_cme_suppression_logged` flag for once-per-window logging

### Fix 2: Reconciliation Key Namespace
- `MockOrderManager.reconcile_positions()` now aggregates `_positions` by instrument
- Algorithm: sum signed quantities per instrument, convert back to `{side, qty, avg_price}`
- Matches `TastytradeBroker.reconcile_positions()` format exactly
- Fixed stale comment on `_positions` dict (was "instrument →", now "strategy_id →")
- Verified compatible with recon loop: all 9 position scenarios tested (flat, long, short,
  multi-strategy same instrument, opposing positions, partial close, multi-instrument, etc.)

### Fix 3: trading_active Auto-Recovery
- New field `EngineState._flatten_reason: str = ""`
- Set in `_emergency_flatten(reason)` — stores "connection_timeout", "kill_switch", "shutdown"
- Set in `pause_trading()` as "manual_pause" — prevents auto-recovery from overriding user pause
- Cleared in `resume_trading()` — clears stale state on manual resume
- **Recovery branch** in `heartbeat_loop`: when `_flatten_reason == "connection_timeout"` and
  `check_heartbeat()` returns healthy → sets `trading_active = True`, clears reason, logs + emits
- Only recovers for "connection_timeout" — kill_switch, shutdown, manual_pause all excluded
- **Critical fix during review**: `bar_processing_loop` now calls `safety.on_bar(bar)` and
  updates `last_prices` BEFORE the `trading_active` gate. Without this, `_last_bar_time` was
  never updated during pause → `check_heartbeat()` always reported stale → recovery was dead.
  Found by edge-case review agent. `safety.on_bar()` is idempotent (timestamp overwrite only).

### Fix 3e: Intra-Bar Monitor Recovery (Bug #36)
- In `intra_bar_exit_loop`, when quotes resume (`bid > 0, ask > 0`) and
  `state.intrabar_monitor_active` is False, re-enables the global flag and per-strategy
  flags for monitored (non-OCO) strategies
- OCO strategies are correctly excluded (their flags were never cleared by staleness handler)
- Guard `not state.intrabar_monitor_active` prevents repeated firing on every quote

### Fix 4: OCO-Aware Emergency Flatten
- New helper `_has_oco_protection(state, strategy_id)` — checks broker's `_brackets` dict
  for active unfilled brackets via `getattr` (returns False for MockOrderManager)
- Handles composite bracket keys (e.g., `MES_V2__tp1`, `MES_V2__tp2`)
- In `_emergency_flatten`, for `reason == "connection_timeout"` only: if strategy has
  active OCO brackets, skip the flatten for that strategy (brackets protect on exchange)
- For "shutdown" and "kill_switch": always flatten everything (existing behavior)
- Log message changed from "All positions flattened" to "Emergency flatten complete"

## What Was NOT Fixed (Deferred Concerns)

### Concern 1: Stale indicators on recovery (LOW risk, ~0.4% per event)
**What**: When `trading_active=False`, `strategy.on_bar()` is never called. SM EMA and RSI
freeze. On recovery, first bars have stale indicators. Phantom 5-min RSI bar can form
from incomplete pre-pause data.

**Why it's low risk**: Entry requires BOTH SM regime AND RSI cross simultaneously. SM is
sluggish (not wrong) after gap. RSI cross requires being within ~2 points of threshold at
pause time. Cooldown (20-25 bars) provides incidental warm-up if a trade exited recently.
Episode flags (long_used/short_used) are conservative (block entries, not enable them).

**Max loss if false entry**: Bounded by SL ($30-175 depending on strategy).

**Recommended fix**: Add `_recovery_warmup_remaining` counter per strategy. Set to 25 on
auto-recovery. Decrement on each `on_bar()`. Suppress entries while > 0. Simple, no side
effects. Not urgent — implement before scaling past 1 contract.

### Concern 2: EOD close doesn't fire while paused
**What**: EOD close (15:30 ET) is checked inside `strategy.on_bar()`. If trading_active=False
at 15:30, position stays open past EOD. For OCO-protected strategies, SL bracket on exchange
provides protection.

**When it matters**: Only if a connection_timeout flatten happens between ~15:25-15:30 AND the
feed doesn't recover before 15:30. Very narrow window.

**Risk**: Position carries overnight with SL protection. Not ideal but bounded.

**Possible fix**: Timer-based EOD close independent of bar_processing_loop. Not urgent for
paper trading.

### Concern 3: Dashboard doesn't show _flatten_reason
**What**: `_flatten_reason` is not exposed in the status API. User sees `trading_active: false`
but can't distinguish auto-recoverable pause from manual pause.

**Impact**: UX only. User might unnecessarily click "Resume" during connection timeout (benign —
resume also sets trading_active=True and clears _flatten_reason).

**Fix**: Add `flatten_reason` to `get_status()` response, add TypeScript type, show in dashboard.
Low priority.

### Concern 4: Recon loop has no CME window suppression
**What**: Reconciliation runs every 60s regardless of CME maintenance window. During 17:00-18:00,
broker API may return empty positions (market closed), causing false mismatch warnings for
OCO-protected positions that are intentionally left open.

**Impact**: Log noise only during CME maintenance. No trading impact.

**Fix**: Add `_in_cme_maintenance_window(state)` check at top of recon iteration. Quick fix,
low priority.

## Review Summary

### Round 9 (heartbeat/recon/flatten fixes — Mar 2, 2026)

**Initial review**: 3 opus agents (heartbeat/recovery logic, OCO flatten + recon, edge cases).
- Agent 1: 0 bugs. All heartbeat/recovery checks pass.
- Agent 2: 0 bugs. OCO detection, recon aggregation, format match all correct.
- Agent 3: **1 HIGH bug found** — auto-recovery dead because `_last_bar_time` never updated
  during pause. Fixed immediately (call `safety.on_bar(bar)` before `trading_active` gate).

**Full team audit**: 6 opus agents (bar gate, entry flow, exit/position, recon scenarios,
CME subsystems, indicator staleness). 36 items checked.
- bar-gate-auditor: 5/5 PASS
- entry-flow-auditor: 3/5 PASS, 2 CONCERN (stale indicators — low risk)
- exit-auditor: 5/6 PASS, 1 CONCERN (EOD close during pause)
- recon-auditor: 9/9 PASS
- cme-subsystem-auditor: 4/6 PASS, 2 CONCERN (dashboard reason, recon CME suppression)
- indicator-auditor: LOW overall risk (~0.4% per recovery, SL-bounded)

**Total: 0 bugs, 4 deferred concerns (all low/moderate risk, none affect strategy execution).**

## Scope of Changes

~85 lines new, ~15 lines modified, all in `runner.py`. No changes to:
- `config.py` — no new config fields
- `safety_manager.py` — no safety logic changes
- `tastytrade_broker.py` — no broker logic changes
- `strategy.py` — no strategy/indicator logic changes
- Dashboard — no frontend changes

Strategy execution (signal generation, entry/exit logic, indicator computation, position
management) is completely unchanged during normal operation.
