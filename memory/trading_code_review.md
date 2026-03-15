# Trading Code Review Checklist

**When**: After writing or modifying ANY live engine, safety, order management,
sizing, or drawdown logic. Before telling the user "done."

**How**: Spawn 3-5 parallel review agents (opus model) with specific audit scopes.
Each agent reads the modified files + their dependencies and reports bugs.

## Review Scopes

### 1. Trade Flow Correctness
- Does the signal → pre_order → safety check → fill pipeline execute in the right order?
- Are strategy_id, instrument, qty passed correctly at every step?
- Can an advisor silently overwrite a safety override? (check qty_locked)
- Does the exposure check sum ALL strategies for the instrument, not just one?
- If a strategy is paused, can it still receive exit signals for open positions?

### 2. Drawdown Rule Logic
- Which strategy's loss triggers each rule? Is the condition on the RIGHT strategy_id?
- After Rule 1 fires: is the paused strategy's open position handled? (exposure collision)
- max_position_size vs. rule-driven qty increases — trace the math end-to-end
- Does manual_override prevent re-pause? Is it cleared at the right time?
- Does daily reset preserve vs. clear the right things? (extended pause survives, Rule 1 doesn't)
- Rolling window: history appended BEFORE clearing? Correct slice size? No-trade days?

### 3. State Broadcasting & Dashboard Sync
- When safety state changes, does the WS broadcast contain the FULL status or just a fragment?
- Could a partial broadcast overwrite the dashboard's full state? (setStatus vs. merge)
- Are per-strategy positions correctly mapped to UI cards? (multi-strategy per instrument)
- Do frontend TypeScript types match backend dict shapes field-for-field?

### 4. API Input Validation & Safety
- Are all API body parameters validated (type, range, nullability)?
- Can malformed WS messages cause silent misbehavior? (e.g., qty="abc" defaults to 1)
- Does force_resume accidentally re-enable something that should stay disabled?
- Path traversal / injection on any user-supplied strings?

### 5. Research-to-Implementation Parity
- If the feature is based on forensic/backtest findings, does the implementation match the EXACT methodology?
- **Timeframe**: Research used 5-min bars? Implementation must resample to 5-min. NOT run on 1-min.
- **Lookback**: Research used lookback=50? Implementation must use 50.
- **Thresholds**: Research used 5pts proximity? Implementation must use 5pts.
- **Data source**: Research used RTH-only bars? Implementation must filter to RTH.
- **Convention**: Research used bar[i-1] for signals? Implementation must match.
- Cross-reference the research document/script, not just the implementation plan.
- (Added Mar 14 2026: OB detection was implemented on 1-min but forensics validated on 5-min. Jason caught it, not the review agents.)

### 6. Edge Cases & Timing
- What happens if on_trade_closed fires for an unknown strategy_id?
- What if two strategies on the same instrument both have open positions during a pause?
- Commission lookup: per-strategy or per-instrument? Does last-writer-wins cause errors?
- Duplicated calculations: will they diverge if one is updated and the other isn't?
- Dead code / old files that could shadow new imports?

## Bugs Found in Feb 2026 SafetyManager Review

Captured here as examples of the class of bugs to watch for:

### Round 1 (plan review — 12 bugs)
1. **Partial WS broadcast blanks dashboard** — safety.get_status() emitted as "status_change",
   dashboard treated it as full StatusData and blanked positions/instruments.
2. **Rule 1 triggered by wrong strategy** — checked `strat.instrument == "MNQ"` instead of
   `strat.strategy_id == "MNQ_V11"`, so V15 losses could pause V11.
3. **max_position_size blocked Rule 1 sizing** — Rule 1 sets V15 qty=2 but doesn't close V11's
   open position, so exposure = 1+2 = 3 > max_position_size=2. Entry blocked.
4. **Extended pause never auto-clears** — daily_reset didn't check rolling count < 4.
5. **force_resume_all re-enabled trading_active** — overrode manual pause / kill switch state.
6. **No input validation on API qty/enabled** — body:dict bypasses Pydantic, truthy strings pass.
7. **sl_count_today on any loss** — incremented on negative PnL, not just SL stops.
8. **Rolling sum included V15** — V15 SL counts polluted the V11-only rolling window.
9. **Zero-trade days skipped history append** — old SL entries never aged out, extended pause permanent.

### Round 2 (trading code review — 5 additional bugs)
10. **CRITICAL: Daily reset at midnight UTC** — `_check_daily_reset` compared UTC dates; fired
    mid-session (7-8 PM ET), clearing all drawdown protection. Fixed: use ET dates.
11. **trade_closed before position=0** — strategy emitted event before resetting state.position,
    so SafetyManager subscribers saw stale position. Fixed: reset position before emit.
12. **Advisor wrapper ignores qty_locked** — on_pre_order didn't enforce qty_locked, allowing
    future advisors to silently overwrite SafetyManager qty overrides. Fixed: added guard.
13. **Session filename path traversal** — `/api/session/{filename}` accepted `../` in filename.
    Fixed: reject `/`, `\`, `..` in filename before filesystem access.
14. **_apply_drawdown_rules called for non-V11** — wasted function call. Fixed: pre-filter to V11.

### Round 3 (post-v2-upgrades review — Feb 20)
Reviewed: bracket cancel, close_position, emergency flatten, intra-bar monitor, P&L correction, trade_update WS event.

15. **CRITICAL: _cancel_bracket pops bracket before cancel API confirms** — if cancel fails (stop filled
    on exchange), AlertStreamer fill is orphaned. `close_position()` then sends market close on flat
    account → unintended new position. Fixed: `get()` not `pop()`, only pop on success.
16. **HIGH: _cancel_bracket else branch (fail + not filled) also popped bracket** — initial fix still
    removed the bracket, orphaning future AlertStreamer fills. Fixed: keep bracket, log CRITICAL.
17. **HIGH: P&L correction circuit breaker only halts, never un-halts** — if correction improves P&L
    above threshold, system stays halted until manual intervention. Fixed: added un-halt path when
    `safety._halted and "Daily loss limit" in safety._halt_reason`.
18. **MEDIUM: Emergency flatten with price 0.0** — creates wrong P&L records. Accepted: broker needs to
    execute regardless; reconciliation corrects afterward.
19. **MEDIUM: Orphaned fill entries in `_bracket_fill_queue`** — concurrent close paths (intra-bar +
    manual) may leave unprocessed fills. Deferred: requires periodic queue drain task.
20. **MEDIUM: `trade_update` match on `entry_time` string is fragile** — string comparison of datetime.
    Deferred: consider adding a unique `trade_id` field.

### Round 4 (full application audit — Feb 20 night)
Full review of all engine, broker, safety, API, dashboard, and feed code.

**CRITICAL:**
21. **`_emergency_flatten` uses price=0.0 when no last price** — creates garbage TradeRecord
    with wrong P&L that feeds SafetyManager, can trigger false circuit breaker halt. Broker
    market order executes at real price, but local accounting is wrong until correction.
    `runner.py:600-602`. Fix: skip TradeRecord creation when price is zero, or skip
    force_close entirely and let broker + reconciliation handle it.

**HIGH:**
22. **Failed `_cancel_bracket` leaves stale bracket indefinitely** — no cleanup path if cancel
    throws and stop isn't confirmed filled. Stale bracket persists across future trades.
    `tastytrade_broker.py:582-592`. Fix: add reconciliation query for order status, or
    time-based expiry on stale brackets.
23. **`close_position` proceeds to market close after bracket cancel failure** — cancel fails,
    stop potentially still live on exchange, market close executes → double close → reverse
    position. `tastytrade_broker.py:371-381`. Fix: do NOT proceed to market close if cancel
    is unconfirmed. Raise or retry.
24. **~~`close_strategy_position` missing try/except around broker close~~** — FIXED Feb 21.
    Added try/except matching `_emergency_flatten` pattern. Returns success with warning
    message if broker call fails. `runner.py:801-803`.

**MEDIUM:**
25. **Extended pause bypasses `manual_override` in `check_can_trade`** — per-strategy Resume
    button updates state but trades still blocked by `_extended_pause` gate. UI shows
    resumed, actually still blocked. Only `force_resume_all` works during extended pause.
    `safety_manager.py:201-202`. Fix: check `manual_override` before `_extended_pause`.
26. **Exposure check uses gross (abs) sum** — if V11 long 1 and V15 short 1, exposure=2 not 0.
    Unlikely with same-direction strategies, but logically wrong. `runner.py:338-344`.
27. **Status poll can overwrite fresh WS safety_status** — 5s poll fetches full status,
    blindly replaces, can overwrite WS update that arrived moments earlier. Stale up to 5s.
    `useWebSocket.ts:212-219`.

**LOW:**
28. **History date stamped UTC not ET** — cosmetic, rolling window math unaffected.
    `safety_manager.py:339`.
29. **Un-halt uses fragile string match** — `"Daily loss limit" in _halt_reason`. Breaks if
    wording changes. `runner.py:1008`. Fix: use `_halt_type` enum.
30. **Bar fallback uses UTC epoch** — when `d.time` missing. Live server always sends it.
    `useWebSocket.ts:144`.
31. **~~`getEtOffset` day comparison fails at month boundaries~~** — FIXED Feb 21. Replaced
    day-of-month comparison with full Date.UTC diff using formatToParts. `PriceChart.tsx:21-40`.
32. **`safety_status` WS doesn't update top-level StatusData fields** — `daily_pnl`,
    `consecutive_losses` stale for up to 5s until next poll. `useWebSocket.ts:104`.
33. **`_on_trade_corrected` crashes on None timestamps** — missing `.isoformat()` guard.
    `server.py:335-336`. Same issue in `_on_trade_closed` at `server.py:308-309`.
34. **`_pending_bars` deque unbounded** — legacy path, ~390 bars/day. `tastytrade_feed.py:93`.
35. **`_bracket_fill_queue` unbounded** — should be dict keyed by strategy_id.
    `tastytrade_broker.py:149`.
36. **No re-enable for intra-bar monitor after quote recovery** — once disabled by 60s
    staleness, V15 degrades to bar-boundary exits permanently for rest of session.
    `runner.py:428-434`. Fix: re-enable when `last_quote_time` is fresh again.

**Verified correct:** signal→pre_order→safety→fill ordering, qty_locked enforcement,
_pre_correction_pnl stamping (all 3 sites), session_close_et="15:30" parsing, paused
strategy exit signals, per-strategy commission, drawdown rule strategy_id targeting,
Rule 1 exposure math, manual_override in drawdown rules, daily reset ET timezone,
rolling window slice, trade_lock on all close paths, path traversal rejection, DST
via formatToParts, force_resume behavior, TS types match backend shapes, trade_update
WS match-and-update-in-place.

### Round 5 (OCO bracket wiring — Feb 22)
Reviewed: OCO bracket placement, TP/SL fill detection, monitor exclusion, staleness handler,
OCO fallback, paper mode SL, unmatched fill warning. 9 review agents total (5 safety/plan-match
for initial OCO wiring, 4 for post-review fixes).

**All checks PASS.** No bugs found in the implementation.

**Findings addressed:**
37. **MEDIUM: OCO fallback silently loses TP monitoring** — if `place_oco_bracket()` fails and
    falls back to simple stop, the strategy stays in `oco_sids` (computed at startup). No entity
    monitors TP. **Fixed**: broker tracks failures in `_oco_failed_sids`, runner clears
    `intrabar_monitor_active` so bar-close TP/SL resume.
38. **HIGH (pre-existing): Paper mode SL not enforced** — `IntraBarExitMonitor._check_exit()`
    didn't check SL. With `intrabar_monitor_active=True`, bar-close SL also skipped.
    **Fixed**: added SL check to `_check_exit()`.
39. **LOW: Unmatched AlertStreamer fill silent** — no log when fill matches no bracket while
    OCO brackets active. **Fixed**: warning log with order ID and complex_order_id.

**Previously tracked, now fixed:**
40. **~~MEDIUM: Double failure (OCO + fallback stop both fail)~~** — FIXED Feb 22.
    Both `_place_stop()` call sites (TP+SL fallback and SL-only) wrapped in try/except.
    On failure: sid added to `_protection_failed_sids`, runner halts engine via
    `safety._halted = True`. `_oco_failed_sids` only populated after successful fallback
    stop (no cross-cleaning needed). 3 review agents verified all paths.

### Round 6 (sizing override controls — Feb 23)
Reviewed: strategy override properties, runner set_strategy_sizing, safety manager status,
intra-bar monitor partial qty, API endpoint + WS command, dashboard SafetyPanel UI.
2 review agents (engine + API/dashboard).

**All checks PASS.** One minor concern, one bug caught and fixed during review.

**Findings addressed:**
41. **HIGH: Missing `config_partial_qty` in API response** — SafetyPanel used
    `config_entry_qty - 1` as TP1 fallback, which is only correct by accident for the
    default MES_V2 config (entry=2). If config ever has entry=3, partial_qty=1, UI would
    show TP1=2 instead of 1. **Fixed**: added `config_partial_qty` to safety_manager
    get_status(), types.ts, and SafetyPanel.
42. **LOW: force_resume_all clearing order** — runner clears strategy overrides before
    calling safety.force_resume_all(). Semantically backwards but safe in single-threaded
    asyncio. Both end up None. Acceptable.

**Verified correct:** active_entry_qty/active_partial_qty properties used at all 6 config
read sites, set_strategy_sizing atomic validation (partial < entry), block-while-positioned
guard, backward-compat strategy_qty routing, force_resume + daily_reset clear both strategy
and SafetyManager overrides, TypeScript types match Python response, SizingInput UI min/max
constraints, Pydantic SizingOverrideBody validation.

### Round 7 (dual OCO brackets for MES v2 partial exit — Feb 26)
Reviewed: dual OCO placement, composite bracket keys, _process_alert_fill partial/SL classification,
check_bracket_fills list return, close_position composite key handling, runner partial fill loop.
5 review agents total (3 round 1, 2 round 2 verifying fixes).

**Round 1 findings (3 agents):**
43. **CRITICAL: `_process_alert_fill` popped position → `check_bracket_fills` marked all fills
    as duplicates** — `_process_alert_fill` did `self._positions.pop(sid)` for full exits. Later,
    `check_bracket_fills` checked `self._positions.get(sid)` and found `None`, marking every fill
    as `_duplicate`. The runner NEVER received bracket fill notifications. Affected ALL OCO
    strategies (single and multi-bracket). **Fixed**: removed position pop from `_process_alert_fill`;
    `check_bracket_fills` is sole owner of position cleanup. SL dedup uses sibling `bracket.filled`
    state + price-proximity instead.
44. **HIGH: `close_position` used stale `pos["qty"]` after partial TP1** — if TP1 filled via
    AlertStreamer but runner hadn't polled `check_bracket_fills` yet, EOD close would market-close
    2 contracts when only 1 was still open (net short 1). **Fixed**: `_process_alert_fill` immediately
    reduces `pos["qty"]` for partial fills.
45. **MEDIUM: Same-cycle SL + TP1 in runner** — if SL processed before TP1 in fill loop,
    `_partial_close()` ran on flat position (`position=0`), recording phantom trade with wrong side.
    **Fixed**: guard `if strategy.state.position == 0: continue` before partial close.
46. **LOW: Dead code `return result` (undefined var)** — unreachable line after `return results`.
    **Fixed**: removed.
47. **LOW: `_place_stop` BracketState qty=1 default** — didn't pass actual `qty` to BracketState
    constructor. **Fixed**: added `qty=qty`.

**Round 2 verification (2 agents):**
All 7 scenarios traced through fixed code: single-bracket SL, single-bracket TP, dual-bracket SL
gap-down, TP1+TP2 normal, TP1+SL runner, EOD after TP1, same-cycle SL+stale TP1. All pass.
Orphaned exchange order analysis: both OCOs share identical SL price → exchange fills both SL legs
and auto-cancels both TP legs. Fallback (OCO #1 + simple STOP) keyed correctly. No orphans.

**Pre-existing bug fixed:**
48. **HIGH: `response.order.id` should be `response.complex_order.id`** — `place_complex_order()`
    returns `PlacedComplexOrderResponse` with `complex_order` field (not `order`). Would crash on
    first live OCO placement. Masked in paper mode. `tastytrade_broker.py:595`. **Fixed**.

### Round 8 (phantom trade fix — Feb 27)
Reviewed: `reject_entry()` in strategy.py, runner.py veto/safety/order-failure paths.
4 opus review agents total (1 initial + 3 full review).

**Bug fixed:**
49. **CRITICAL: Phantom trade on safety-blocked entry** — Strategy `_open_position()` mutates state
    (position=1/-1, entry_price, qty_remaining, etc.) before runner checks safety. When safety blocks,
    strategy has phantom position; intra-bar monitor generates exit → phantom trade recorded, P&L
    tracked. Found in production Feb 27 (vScalpB trade #4). **Fixed**: added `reject_entry()` to
    strategy, called from runner at all 4 rejection points (advisor veto, safety block,
    order_manager missing, place_market_order exception).

**Initial review findings (3 issues, all fixed):**
50. **MEDIUM: Missing `reject_entry()` on `place_market_order()` exception** — Network/broker errors
    would leave phantom position with no exchange order. **Fixed**: try/except around order placement
    calls `reject_entry()` + emits CRITICAL error.
51. **MEDIUM: Exposure sum double-counted current strategy** — `_open_position()` already set
    `qty_remaining` before safety check, inflating `current_exposure`. Could falsely reject valid
    trades. **Fixed**: excluded current `strategy_id` from exposure sum.
52. **LOW: `"signal"` event emitted before veto** — Dashboard receives BUY/SELL notification for
    entries that are subsequently rejected. Cosmetic only — no trade recorded. **Deferred**.

**Full 3-agent review findings (Feb 27):**
Agent 1 (trade flow correctness): ALL CLEAR. reject_entry() correct at all 4 sites.
Agent 3 (regression check): ALL CLEAR. No regressions on happy path.
Agent 2 (live mode / broker integration): Found issues on order placement failure path:

53. **CRITICAL (OPEN): `reject_entry()` after broker failure can orphan real position** — In live
    mode, if `place_market_order()` sends the order to tastytrade but `_wait_for_fill` times out or
    throws, `reject_entry()` makes strategy think flat. But the order may have filled on the exchange.
    Result: real position with no OCO bracket protection, strategy won't manage it. Two scenarios:
    (a) `_wait_for_fill` throws after order sent — order executed on exchange, confirmation lost.
    (b) Network timeout during `place_order()` HTTP call — order may have reached tastytrade.
    `runner.py:~430`. **NOT YET FIXED** — needs "sent but unconfirmed" state. See live_launch_risks.md.
54. **HIGH (OPEN): Reconciliation is detect-only** — Position reconciliation logs a warning on
    mismatch but doesn't auto-correct. An orphaned position from #53 would be flagged but not fixed
    until manual intervention. **NOT YET FIXED**.
55. **MEDIUM-HIGH (OPEN): AlertStreamer silently drops unmatched fills** — If an orphaned position's
    bracket fills fire, AlertStreamer finds no matching bracket and silently ignores. Should alert on
    ALL unexpected fills. **NOT YET FIXED**.
56. **MEDIUM (OPEN): No manual close-by-instrument endpoint** — Dashboard cannot close an orphaned
    position that no strategy owns. **NOT YET FIXED**.

### Round 9 (heartbeat, recon, emergency flatten fixes — Mar 2, 2026)
Reviewed: CME maintenance suppression, recon key aggregation, auto-recovery, OCO-aware flatten,
bar processing gate, indicator staleness, all exit paths during pause/recovery.
9 opus review agents total (3 initial, 6 full team audit). 36 items checked.

**Initial review (3 agents):**
57. **HIGH: Auto-recovery dead on arrival** — `bar_processing_loop` skipped `process_bar()`
    when `trading_active=False`, so `safety.on_bar()` was never called, `_last_bar_time` froze,
    `check_heartbeat()` always reported stale. Recovery condition could never be satisfied.
    **Fixed**: call `safety.on_bar(bar)` and update `last_prices` BEFORE the `trading_active`
    gate. `on_bar` is idempotent (timestamp overwrite only). `runner.py:549-553`.

**Full team audit (6 agents) — 0 additional bugs:**
- bar-gate-auditor: 5/5 PASS. No double-counting, no strategy mutation pre-gate.
- entry-flow-auditor: 3/5 PASS, 2 CONCERN (stale indicators ~0.4% risk, no warm-up gate).
- exit-auditor: 5/6 PASS, 1 CONCERN (EOD close doesn't fire while paused).
- recon-auditor: 9/9 PASS. All position scenarios verified, format matches TastytradeBroker.
- cme-subsystem-auditor: 4/6 PASS, 2 CONCERN (_flatten_reason not in API, recon no CME suppression).
- indicator-auditor: LOW risk. SM sluggish not wrong. RSI phantom 5-min bar possible but
  requires threshold boundary + SM regime simultaneously. SL bounds loss.

**Previously tracked, now fixed:**
58. **~~Bug #36: Intra-bar monitor permanent disable~~** — FIXED Mar 2.
    Re-enables `intrabar_monitor_active` when quotes resume in `intra_bar_exit_loop`.
    Guard prevents repeated firing. OCO strategies correctly excluded.

**Not fixed (deferred):**
59. **LOW: Stale indicators on recovery** — SM/RSI not updated during pause. ~0.4% false
    entry probability per recovery event, SL-bounded. Recommend warm-up gate before scaling.
60. **LOW: EOD close skipped while paused** — OCO brackets protect on exchange. Only triggers
    if outage spans 15:30 exactly. Accept risk.
61. **LOW: `_flatten_reason` not in dashboard API** — UX only. User can't distinguish auto-
    recoverable pause from manual pause. No trading impact.
62. **LOW: Recon loop no CME suppression** — False mismatch warnings during 17:00-18:00 for
    OCO-protected positions. Log noise only.

Details: `memory/heartbeat_recon_fixes.md`

### Round 10 (vScalpC implementation + SL→BE — Mar 9, 2026)
Reviewed: MNQ_VSCALPC config, replace_bracket_sl(), runner SL→BE trigger, paper mode breakeven SL,
dashboard marker colors. 4 review agents (trade flow, edge cases, dashboard, config correctness).

**Config fixes (found by config review agent):**
63. **MEDIUM: VIX death zone missing from vScalpC** — Same entries as V15 (SM_T=0.0, RSI 60/40)
    should share the same VIX gate. **Fixed**: added `vix_death_zone_min=19, max=22`.
64. **MEDIUM: max_strategy_daily_loss too low** — 2-contract SL = $160, exceeds $100 limit.
    Auto-pauses on every single SL with no recovery. **Fixed**: raised to $200.

**Edge case fixes (found by edge case agent):**
65. **CRITICAL: Orphan OCO on race in replace_bracket_sl** — Old bracket popped before new OCO
    placed. If SL fills during window, fill is dropped, new OCO placed on flat position. **Fixed**:
    keep old bracket in dict during window, post-placement check for `old_bracket.filled`, cancel
    orphan OCO if detected. `tastytrade_broker.py:replace_bracket_sl`.
66. **HIGH: reset_daily() doesn't clear partial_filled** — If position survives EOD, `partial_filled`
    persists into next day. **Fixed**: clear `partial_filled`, `qty_remaining`, `max_favorable`,
    `trail_activated` in `reset_daily()` when `position == 0`. `strategy.py:829-838`.

**Trade flow fixes (found by trade flow agent):**
67. **HIGH: replace_bracket_sl failure doesn't re-enable bar-close TP** — If OCO replacement fails,
    `_oco_failed_sids` only consumed on next entry. Runner stuck with no TP exit. **Fixed**: immediately
    set `strategy.intrabar_monitor_active = False` on failure. `runner.py:425-440`.

**Not fixed (deferred/pre-existing):**
68. **HIGH: `_place_dual_oco` uses config `partial_qty`, ignoring dashboard override** — Pre-existing,
    affects MES_V2 too. Dashboard partial_qty override not propagated to OCO bracket sizing in live mode.
    `tastytrade_broker.py:731`. Defer: low impact, dashboard override rarely used.
69. **HIGH: Cancel-replace gap in replace_bracket_sl** — Brief unprotected window (~1 network round-trip)
    between cancel and replace. Inherent to cancel-replace pattern. Mitigated by race checks and fallback
    stop. Accept risk for paper trading; consider atomic modify if tastytrade supports it.
70. **MEDIUM: SL=0 at breakeven exits on exact entry touch** — Costs commission ($1.04) with zero P&L.
    Design decision: matches validated backtest (WR 77.3% includes these exits). No change.

**Dashboard (found by dashboard agent):**
71. **LOW: SessionTradeList hardcoded colors/labels** — Only V15 had clean label and unique color.
    **Fixed**: added vC/vB/MES labels and strategy-specific badge colors.
72. **MEDIUM (pre-existing): InstrumentCard shows 1 of 3 MNQ positions** — Single position display
    per instrument. Amplified by 3rd MNQ strategy but not a regression. Deferred.

### Round 11 (Prior-Day ATR gate for vScalpC — Mar 9, 2026)
Reviewed: StrategyConfig field, SafetyManager ATR tracking/gate/status, dashboard types+badge.
3 review agents (trade flow, state/dashboard, backtest match).

**Bug found by backtest-match agent:**
73. **HIGH: Backtest UTC vs live ET date boundary** — `compute_prior_day_atr()` in `htf_common.py`
    used `pd.DatetimeIndex(df_1m.index).date` on UTC timestamps, grouping bars 7pm-midnight ET
    into the wrong calendar date. Live engine uses ET dates. Daily ranges diverged, producing
    different ATR values and gate decisions. **Fixed**: added `_get_et_dates()` helper using
    `tz_localize('UTC').tz_convert(_ET)`. Same fix applied to `compute_volatility_regime()`.
    Sweep re-run: threshold shifted 252.9 → 263.8. Both functions now match live engine.

**Dashboard fix (found by state/dashboard agent):**
74. **MEDIUM: Missing ATR gate in dashboard** — `atr_gated` boolean and `prior_day_atr` dict
    not in TypeScript types or SafetyPanel badge chain. **Fixed**: added to `types.ts` and
    purple "ATR" badge after LEVEL in SafetyPanel.

**Verified correct:**
- Gate uses `_gate_prev` (bar[i-1]) consistently, swap before update
- Wilder ATR formula matches backtest (seed = mean of first 14, recursive step)
- Fail-open during warmup (first 14 days, `atr_val is None`)
- Manual override bypasses gate, daily reset doesn't touch ATR state
- `_atr_daily_ranges` trimmed to 20 entries (no memory leak)
- Multi-strategy isolation: V15 (threshold=0) unaffected, only vScalpC gated
- Deduplication via `(inst, bar.timestamp)` key prevents multi-call corruption

**Not fixed (deferred):**
75. **LOW: 14-day warmup after restart** — ATR state lost on restart, fail-open for ~3 weeks.
    Acceptable for paper trading. Consider persisting ATR state to disk before live.

### Key pattern
The most dangerous bugs are **timing/ordering** issues (position state during events,
UTC vs ET dates, history append vs clear ordering), **wrong scope** (instrument vs
strategy_id, all strategies vs V11-only), and **race conditions in async broker calls**
(cancel vs fill, concurrent close paths). Always trace the exact execution order,
verify which strategy_id triggers each rule, and check what happens when async
operations interleave.
