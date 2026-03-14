# Known Risks for Live Launch

Risk register for going live. Target portfolio: vScalpA(1) + vScalpB(1) + MES v2(1).
Dashboard sizing controls allow scaling up any strategy intraday.

## Deferred Bugs (Accepted Risk at 1 Contract)

### Bug #22/#23: Bracket Cancel Race Condition (TastytradeBroker)

- **What**: When cancelling an OCO bracket, one leg could fill before the cancel completes
- **Risk level**: Very low at 1 contract with 5pt TP scalps lasting 3-5 bars
- **Why deferred**: The window for the race is tiny (milliseconds), and at 1 contract the exposure is bounded by SL
- **Mitigation**: Position reconciliation on each bar catches any orphaned positions
- **Action**: Fix before scaling past 5 contracts

### ~~Bug #36: Intra-Bar Monitor Permanent Disable After Quote Staleness~~ — FIXED Mar 2

- **What**: If DXLink quotes go stale, the intra-bar monitor disabled itself permanently
- **Fix**: `intra_bar_exit_loop` re-enables `intrabar_monitor_active` when quotes resume
- **Status**: FIXED. Reviewed by 9 agents (3 initial + 6 full team audit). Details: `memory/heartbeat_recon_fixes.md`

### ~~Bug: Phantom Trade on Safety-Blocked Entry~~ — FIXED Feb 27

- **What**: When the daily P&L limiter pauses a strategy, the runner blocks the entry but the strategy state machine still opens the position internally.
- **Found**: Feb 27, 2026 — vScalpB trade #4 was blocked by safety but recorded as a -$65 loss.
- **Fix**: Added `reject_entry()` to strategy, called at all 4 rejection points in runner (advisor veto, safety block, order_manager missing, place_market_order exception). Also fixed exposure double-counting (excluded current strategy from sum).
- **Status**: FIXED. Reviewed by 4 agents (1 initial + 3 full review). Trade flow and regression agents all clear.

### Bug #53: Order Failure Can Orphan Real Position (CRITICAL for live)

- **What**: In live mode, if `place_market_order()` sends the order to tastytrade but the confirmation fails (network timeout, `_wait_for_fill` throws), `reject_entry()` makes the strategy think it's flat. But the order may have filled on the exchange — creating a real unprotected position with no OCO bracket.
- **Found**: Feb 27, 2026 — discovered during 3-agent review of phantom trade fix.
- **Risk level**: CRITICAL conceptually, but narrow blast radius. Only triggers on broker API failures (network drops mid-request), not on safety blocks or vetoes. At 1-contract scale, worst case is a single unprotected contract detected by reconciliation within ~1 bar (1 minute).
- **Recommended fix**: Add "sent but unconfirmed" state before calling `reject_entry()` on broker exceptions. On failure: poll `tastytrade_broker.get_positions()` to check actual fill status. If filled → proceed normally (place brackets). If not filled → then `reject_entry()`. If indeterminate → halt strategy and alert.
- **Related issues**:
  - Reconciliation is detect-only, doesn't auto-correct (#54)
  - AlertStreamer silently drops fills that match no bracket (#55)
  - No manual close-by-instrument API endpoint (#56)
- **Mitigation at 1 contract**: Reconciliation detects mismatch within 1 bar. Manual intervention via tastytrade app. Be at desk during market hours week 1.
- **Action**: Fix before scaling past 1 contract. Acceptable risk at 1 contract with manual monitoring.

### Bug #21: Emergency Flatten with price=0

- **What**: If emergency flatten triggers and can't get a market price, it may submit with price=0
- **Risk level**: Very low --- tastytrade rejects orders with price=0, and market orders don't use price field
- **Why deferred**: Emergency flatten is a last resort, and reconciliation catches any issues
- **Mitigation**: Manual oversight during initial live trading
- **Action**: Fix to use market order explicitly

## Operational Risks

### VIX Death Zone Gate — IMPLEMENTED Mar 3

- **What**: VIX prior-day close 19-22 blocks MNQ entries (V15 + vScalpB). MES unaffected.
- **Status**: IMPLEMENTED. Fetches VIX via yfinance at startup + daily reset. Fail-open (None = no gate).
- **Dashboard**: Header shows VIX close, per-strategy amber "VIX GATE" badge.
- **Phantom signal log**: Blocked entries logged to `logs/vix_blocked_signals.csv` for post-hoc validation.
- **Manual override**: Resume button bypasses gate; re-engages on daily reset.
- **Details**: `memory/vix_death_zone_gate.md`

### No Automated News Day Filter Yet

- vScalpA loses on FOMC days, vScalpB loses on Retail Sales days
- **Mitigation**: Manual pause on known news days until filter is implemented
- **Key dates to watch**: FOMC, NFP, CPI, Retail Sales

### First Week of Live Trading

- Higher monitoring needed --- watch for fill quality, slippage, bracket placement timing
- **Plan**: Be at desk during market hours for first 3 days minimum
- Dashboard at localhost:3000 shows real-time status

### Launcher Silent Crash — Orphaned Processes (Low Priority)

- **What**: Launcher (.app) disappears from dock silently (cause unknown — possibly macOS app nap, memory pressure, or crash). Engine + dashboard keep running as orphaned processes with no cleanup control.
- **Risk level**: Low — engine is fully independent of launcher after startup. Trades, brackets, safety rules all unaffected.
- **Annoyance**: Next morning double-click reuses orphaned engine (port 8000 detected), but launcher can't clean up those processes on quit. User must manually kill PIDs.
- **Action**: Investigate crash cause (check Console.app for crash reports). Consider replacing wait-loop with something more robust, or adding a launchd plist for the engine so it doesn't depend on the launcher staying alive.

### Stale Indicators After Auto-Recovery (~0.4% false entry per event)

- **What**: After connection_timeout flatten + auto-recovery, SM/RSI indicators are stale. First bars after recovery could produce a false entry if RSI was near threshold boundary at pause time.
- **Risk level**: Low (~0.4% probability per recovery event). Requires SM regime + RSI cross simultaneously.
- **Max loss**: Bounded by SL ($30-175 depending on strategy).
- **Mitigation**: Cooldown provides incidental warm-up (20-25 bars) if a trade exited recently. Episode flags are conservative.
- **Action**: Add `_recovery_warmup_remaining` counter (suppress entries for 25 bars after recovery). Implement before scaling past 1 contract.

### EOD Close Skipped While Trading Paused

- **What**: If connection_timeout flatten happens between ~15:25-15:30 ET and feed doesn't recover before 15:30, EOD close never fires. OCO-protected positions carry overnight.
- **Risk level**: Very low — requires outage spanning 15:30 exactly. Position has SL protection on exchange.
- **Mitigation**: SL bracket on exchange bounds overnight risk. Reconciliation detects position.
- **Action**: Consider timer-based EOD close independent of bar_processing_loop. Not urgent.

### DXLink Connection Stability

- Reconnection logic exists but untested over multi-day live sessions
- **Mitigation**: Daily session rotation (engine restarts each morning)
- **Action**: Monitor connection drops in first week

### MES v2 Dual OCO — Step 0 Validated (Feb 27, 2026)

- **What**: MES v2 partial exit places two independent OCO brackets (1 contract each).
- **Status**: VALIDATED — tastytrade accepts two simultaneous OCO brackets. Tested with 2x M6E on production API: both OCOs accepted, cancelled, and position closed cleanly.
- **Remaining**: Paper trade MES v2 with dual OCO brackets on actual MES contracts (requires account funding for 2-contract MES margin ~$3,000+).

## Action Items from Feb 27 Paper Trading Review

### ~~1. Fix phantom trade bug (CRITICAL — blocks live)~~ — DONE
Fixed with `reject_entry()` method. Reviewed by 4 agents. See above.

### 2. Daily limit scaling decision for vScalpB at 2x
At 2x, SL costs ~$63 vs ~$31 at 1x. Current $100 limit = ~1.5 SLs before pause.
**Recommendation: keep $100.** Today it paused after 1 TP + 2 SLs = 3 trades. This is tight
but matches "chip away" philosophy — limits damage on bad draws. The limiter correctly
prevented a 3rd consecutive SL.

### ~~3. Implement BE_TIME in live engine~~ — DONE (Mar 4)
Implemented at N=75 (re-swept from original N=275). OOS Sharpe 1.502, PF 1.292.
Bar counter in `strategy.py`, config `breakeven_after_bars=75` on MES_V2.
Off-by-one between backtest and live engine found and fixed during code review.

### 4. MES v2 running at 2x during paper — should be 1x for target portfolio
Today's MES loss was -$285 at 2x. Target portfolio is 1x = -$145.
Verify dashboard sizing is set correctly for paper trading validation.

## Go-Live Checklist

- [x] Merge go-live-prep branch (commit 8bb82db, Feb 24)
- [x] Fix heartbeat zombie engine (CME maintenance suppression + auto-recovery) — FIXED Mar 2
- [x] Fix reconciliation false alarms (MockOrderManager key namespace) — FIXED Mar 2
- [x] Fix Bug #36 (intra-bar monitor permanent disable) — FIXED Mar 2
- [x] OCO-aware emergency flatten (skip bracket-protected positions on timeout) — FIXED Mar 2
- [x] Implement VIX death zone gate (19-22) for MNQ strategies — DONE Mar 3
- [ ] Restart engine, verify config loads with new fields
- [ ] Check dashboard SL counters visible + VIX close visible
- [ ] Paper trade for 2-3 days with new safety rules active
- [x] Verify P&L limit auto-pause triggers correctly — CONFIRMED Feb 27 (vScalpB paused at -$105)
- [x] Fix phantom trade bug (strategy state desync on blocked entry) — FIXED Feb 27 (reject_entry)
- [x] Implement BE_TIME N=75 for MES v2 in live engine — DONE Mar 4
- [ ] Fix order failure orphan risk (#53) — acceptable at 1 contract, fix before scaling
- [x] Validate tastytrade accepts dual OCO brackets (Step 0 for MES v2) — PASSED Feb 27
- [ ] Paper trade MES v2 with dual OCO brackets — verify TP1 partial, TP2 full, SL, EOD flows
- [ ] `touch live_trading/.live` to enable live mode
- [ ] Verify logs show "LIVE" mode
- [ ] Fund tastytrade account appropriately
- [ ] Set up news day calendar alerts for manual pausing

## Suggested Go-Live Path

1. **Before live**: Fund account, `touch .live`, verify LIVE in logs, check dashboard
2. **Week 1 live**: **vScalpA(1) + vScalpB(1)** on MNQ — single-contract OCO, well-tested
3. **Keep MES v2 in paper** until dual OCO is paper-tested on actual MES contracts (~$3k+ margin)
4. **Scale**: Use dashboard sizing controls to increase qty intraday when confident
5. **MES v2 live**: After paper validation of dual OCO on MES
6. **News days**: Manual pause on FOMC/NFP/CPI/Retail Sales until automated filter built
