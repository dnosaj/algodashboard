# Critical Engine Bugs Fixed

## DST Bug in Session Filtering (Feb 14)
- Hardcoded UTC offsets (15:00 UTC = 10 AM ET) only correct for EST
- During EDT (Mar-Nov), 10 AM ET = 14:00 UTC
- **Fix**: compute_et_minutes() with zoneinfo, use ET constants

## Look-Ahead Bias in ALL Exits (Feb 14)
- Stop loss, underwater, price structure exits checked current bar's low/close but filled at current bar's open (before low/close is known)
- **Fix**: Check bar i-1 data, fill at bar i open

## RSI Mapping Look-Ahead (Feb 14)
- Mid-window 1-min bars (e.g., 10:01) were getting current in-progress 5-min RSI instead of previous completed bar's RSI
- **Fix**: Always use rsi_5m_vals[j-1]

## RSI Cross Persistence Mismatch (Feb 14)
- Python fired RSI cross on 1 bar only at 5-min boundary
- Pine's request.security() persists cross across all 5 bars in window
- **Fix**: Added rsi_5m_curr/rsi_5m_prev mapped arrays

## Episode Reset Flickering with Threshold > 0 (Feb 19)
- `not sm_bull` (SM <= threshold) flickers in the 0-to-threshold zone
- Caused repeated entries in choppy conditions
- **Fix**: Use zero-crossing only: `sm_prev <= 0` instead of `not sm_bull`
- Fixed in both Pine v11.1 and live engine strategy.py

## Daily Reset UTC Bug (Feb 19)
- `_check_daily_reset` compared UTC dates — fired at midnight UTC (7-8 PM ET, mid-session)
- Spuriously cleared all drawdown protection, circuit breakers, and P&L counters mid-session
- Missed reset at actual session boundary (~16:00 ET)
- **Fix**: Convert bar.timestamp to ET via `zoneinfo.ZoneInfo("America/New_York")` before date comparison

## trade_closed Event Ordering (Feb 19)
- `strategy._close_position()` emitted `trade_closed` BEFORE setting `state.position = 0`
- SafetyManager subscribers saw stale position during their handlers
- **Fix**: Moved `state.position = 0` (and related resets) BEFORE the `event_bus.emit("trade_closed", trade)` call

## SafetyManager Review Bugs (Feb 19)
- 12 bugs found by parallel review agents after initial implementation. Key ones:
  - WS broadcast blanked dashboard (safety-only dict sent as full "status")
  - Rule 1 triggered on wrong strategy (checked instrument, not strategy_id)
  - sl_count_today incremented on any loss, not just SL stops
  - Extended pause never auto-cleared (zero-trade days skipped history append)
  - force_resume_all set trading_active=True (bypassed kill switch)
  - Session filename path traversal vulnerability
  - Advisor wrapper didn't enforce qty_locked
- All fixed. See plan file for full details.

## Bracket Cancel Race Condition (Feb 20)
- `_cancel_bracket` called `self._brackets.pop()` BEFORE cancel API confirmed
- If cancel fails (stop already filled on exchange), AlertStreamer fill is orphaned
- Then `close_position()` sends market close on already-flat account → unintended new position
- **Fix**: `get()` not `pop()`. Only pop on success. On failure w/ fill: pop + log. On failure w/o fill: keep bracket for reconciliation, log CRITICAL.
- Also: `close_position` re-checks bracket after cancel returns; market close wrapped in try/except (no position restore on failure — order may be sent but fill polling timed out)

## Emergency Flatten Robustness (Feb 20)
- `_emergency_flatten` used 0.0 as price when `last_prices` had no entry
- No try/except on `close_position()` — one failure blocked remaining strategies
- No fill patching — P&L records reflected estimated price, not actual fill
- **Fix**: Handle None price (warn + attempt anyway), try/except per strategy, fill patching with `_pre_correction_pnl` stamp

## Intra-Bar Monitor Error Handling (Feb 20)
- `_execute_exit` had no try/except on `close_position()` call
- `on_quote` called `_execute_exit` without guard — one strategy's error blocked others
- **Fix**: Both wrapped in try/except; emit error event on failure

## P&L Correction Handler Was Log-Only (Feb 20)
- `trade_corrected` event handler only logged — didn't adjust SafetyManager P&L
- If actual fill was significantly worse than estimated, circuit breakers wouldn't fire
- **Fix**: Delta-adjustment handler: `delta = new_pnl - old_pnl`, adjusts `_global_daily_pnl` + per-strategy, re-checks circuit breaker (both halt and un-halt paths)

## WS Bar Handler Used UTC Instead of Server ET (Feb 20)
- Dashboard `useWebSocket.ts` bar handler parsed `d.timestamp` (ISO → UTC) instead of using `d.time` (ET epoch from server)
- Caused bars to shift by 5h on chart
- **Fix**: Use `d.time` with `typeof` guard for backward compat

## DST-Hardcoded Offset in PriceChart (Feb 20)
- Backward compat for old sessions used `-5 * 3600` (EST only, wrong during EDT)
- `getEtEpoch` fallback also returned UTC not ET
- **Fix**: DST-aware `getEtOffset()` using `Intl.DateTimeFormat.formatToParts()` with `timeZone: 'America/New_York'`

## Quote Feed Staleness (Feb 20)
- If quote feed dies, `intrabar_monitor_active` stays True permanently
- Intra-bar exits silently stop working with no indication
- **Fix**: Staleness check in `bar_processing_loop` — if >60s since last quote, disable monitor. Re-enables automatically when quotes resume.

## OCO Bracket Wiring (Feb 22)
- After entry fill, `place_market_order()` now places OCO bracket (LIMIT TP + STOP SL) on exchange instead of simple stop
- `_process_alert_fill()` uses price proximity to classify fills as TP vs SL
- Runner maps fill type to `ExitReason.TAKE_PROFIT` or `ExitReason.STOP_LOSS`
- IntraBarExitMonitor excludes OCO-bracketed strategies in live mode (via `exclude_sids`)
- Staleness handler skips OCO strategies when disabling monitor (exchange handles exits regardless)
- Falls back to simple stop if OCO placement fails; runner re-enables bar-close TP/SL
- Paper mode completely unaffected (all gated on `hasattr(..., 'place_oco_bracket')`)

## Paper Mode SL Not Enforced (Feb 22 — pre-existing bug fixed)
- `IntraBarExitMonitor._check_exit()` only checked TP and trail, NOT SL
- When `intrabar_monitor_active=True` in tastytrade paper mode, `on_bar()` SL check was also skipped
- Per-trade SL was silently unenforced — positions could lose far more than `max_loss_pts`
- **Fix**: Added SL check to `_check_exit()`: `unrealized <= -strat.config.max_loss_pts`
- Safe in live mode: OCO strategies are excluded from the monitor entirely

## Unmatched AlertStreamer Fill Warning (Feb 22)
- If AlertStreamer detects a fill matching no active bracket while OCO brackets exist, it was silently ignored
- **Fix**: Warning log at end of `_process_alert_fill()` with order ID, complex_order_id, and active OCO strategy list
- Defense-in-depth observability for matching failures

## Double Failure Protection — Engine Halt (Feb 22)
- If both `place_oco_bracket()` AND fallback `_place_stop()` fail, position has no exchange protection
- Also: SL-only path (`tp_pts == 0`) had unprotected `_place_stop()` — same vulnerability
- **Fix**: Both paths wrapped in try/except. On failure, sid added to `_protection_failed_sids`.
  Runner checks `pop_protection_failures()` after each entry fill, halts engine immediately.
  `_oco_failed_sids` only populated after successful fallback (no double-processing).
- Fill always returned so position is visible to dashboard/safety

## Key Lesson
Always backtest on the SAME timeframe as Pine (1-min), not resampled 5-min.
