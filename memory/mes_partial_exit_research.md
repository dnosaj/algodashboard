# MES v2 Partial Exit Research (Feb 23, 2026)

## Context

Live paper trade: MES v2 entered long at 6,857.75, rallied to MFE of +11.50 pts
(high 6,869.25 at 13:28), then reversed. After 105 bars, trade was -$69. TP=20
was never close — needed 8.50 more pts. This is the recurring frustration:
trades that go 10-15 pts in your favor then give it all back.

## MFE Distribution (382 trades, 12 months)

| MFE Level | Trades | % |
|-----------|--------|---|
| >= 5 pts  | 311    | 81.4% |
| >= 10 pts | 251    | 65.7% |
| >= 12 pts | 223    | 58.4% |
| >= 15 pts | 187    | 49.0% |
| >= 20 pts | 154    | 40.3% |

- **104 trades (27.2%) went 10+ pts in favor but never hit TP=20.** Avg final P&L: -$23.
- By exit reason: TP 38.5%, SL 11.5%, EOD 50.0%.
- EOD trades avg P&L = -2.4 pts (biggest drag).

## TP Sweep (single contract)

| TP | Trades | WR | PF | Net |
|----|--------|----|----|-----|
| 8  | 477 | 70.6% | 1.363 | $2,795 |
| 10 | 452 | 67.7% | 1.348 | $2,989 |
| 12 | 432 | 64.4% | 1.271 | $2,414 |
| 16 | 398 | 62.8% | 1.314 | $3,191 |
| 18 | 389 | 61.4% | 1.347 | $3,684 |
| 20 | 382 | 60.7% | 1.429 | $4,718 |

TP=20 wins on raw P&L. Lower TPs boost WR but reduce net.

## Partial Exit Analysis (2 contracts: 1@TP1, runner@TP20, SL=35)

Baseline: 2@TP20 = $9,435, WR 60.7%, PF 1.429, MaxDD -$2,055

| TP1 | Net | vs BL | WR | PF | MaxDD | Rescued |
|-----|-----|-------|----|----|-------|---------|
| 5   | $8,041 | -$1,394 | 65.2% | **1.584** | **-$1,584** | 143 |
| 9   | $8,225 | -$1,210 | 65.2% | 1.488 | **-$1,560** | 97 |
| 10  | $7,740 | -$1,695 | 64.9% | 1.439 | -$1,736 | 81 |
| 16  | $8,069 | -$1,366 | 62.0% | 1.391 | -$2,169 | 18 |

### Key findings

- **No partial exit beats baseline on raw P&L.** TP=20 on all contracts maximizes total return.
- **TP1=5 is the risk-adjusted winner**: PF 1.584 (+11% vs baseline), MaxDD -$1,584 (23% better), WR 65.2%.
- **TP1=9 is the P&L-maximizing partial**: closest to baseline at -$1,210, good DD.
- The rescued trades (TP1 hit, TP2 didn't) are the trades that would have been losers — partial exit turns them breakeven or small winners.
- The cost: on the 147 trades where both would hit, you give up (TP2-TP1) x $5 on one contract.

### At scale (20 contracts)

| Metric | 2@TP20 | 1@TP5 + 1@TP20 |
|--------|--------|-----------------|
| MaxDD (20x) | -$41,100 | -$31,680 |
| Net (20x) | $94,350 | $80,410 |

The DD difference matters more at scale.

## Decision

Favor risk-adjusted returns over maximum extraction. See trading philosophy in MEMORY.md.
Next step: paper trade MES v2 with partial exits before committing to the approach.

## Implementation (Feb 23, 2026)

Engine multi-contract partial exit support is **IMPLEMENTED** across 11 files:

### Config (`engine/config.py`)
- `entry_qty=2, partial_tp_pts=10, partial_qty=1` on MES_V2
- MNQ configs unchanged (`entry_qty=1, partial_tp_pts=0`) — completely unaffected

### Data model (`engine/events.py`)
- `PARTIAL_CLOSE_LONG/SHORT` signal types, `TAKE_PROFIT_PARTIAL` exit reason
- `TradeRecord.qty` and `TradeRecord.is_partial` fields

### Strategy state machine (`engine/strategy.py`)
- `TradeState.qty_remaining` and `partial_filled` fields
- `_partial_close()` method: decrements qty, marks partial_filled, does NOT reset position
- `_close_position()` scales P&L by `qty_remaining`
- TP1 check fires before TP2 in `on_bar()`

### Runner (`engine/runner.py`)
- `MockOrderManager.partial_close_position()` decrements qty
- Partial close signal handler in `process_bar()`
- `entry_qty` preserved via `qty_locked` (prevents FixedSizeAdvisor override)
- Exposure, reconciliation, unrealized P&L all use `qty_remaining`
- Emergency flatten, manual close, trade correction all scale by `trade.qty`

### Intra-bar monitor (`engine/intra_bar_monitor.py`)
- `_check_exit()` returns `(reason, is_partial)` tuple
- TP1 fires before TP2, dispatches to `_execute_partial_exit()`
- `partial_filled=True` prevents re-trigger

### Safety (`engine/safety_manager.py`)
- Commission: `2 * trade.qty * commission_per_side`
- Partial trades: skip trade_count and consecutive_losses, DO contribute to daily_pnl

### API + Dashboard
- All 4 trade serialization points include `qty` and `is_partial`
- InstrumentCard: "2x LONG" + PARTIAL badge
- TradeLog: Qty column + TP1 badge
- PriceChart: Square orange markers for partial exits

### Known limitations
- At most 2 exit legs (partial at TP1 + remainder). Generalizing to N legs is overkill.

### Code review findings (3 parallel agents — paper mode, Feb 23)
- All safety/P&L scaling correct
- All dashboard components correct
- All intra-bar monitor logic correct

## Live OCO Partial Exit (Feb 26, 2026)

TastytradeBroker now supports two independent OCO brackets for partial-exit strategies.

### Design
After a 2-contract MES_V2 entry fills:
- **OCO #1 (tp1)**: 1 contract — LIMIT @ entry+10 (TP1), STOP @ entry-35 (SL)
- **OCO #2 (tp2)**: 1 contract — LIMIT @ entry+20 (TP2), STOP @ entry-35 (SL)

Both are fully independent, self-contained on the exchange. Engine crash = both SLs resting.

### Implementation (`tastytrade_broker.py`)
- `BracketState` has `tag` ("tp1"/"tp2"/"") and `qty` fields
- Composite bracket keys: `"MES_V2__tp1"`, `"MES_V2__tp2"` (single-bracket: plain `"MNQ_V15"`)
- `_bracket_keys(sid)` helper finds all keys for a strategy
- `_place_dual_oco()`: places two OCOs, graceful degradation on partial failure
- `_process_alert_fill()`: classifies fills (tp1+TP → `take_profit_partial`, tp2+TP → `take_profit`, SL → `stop_loss`). SL dedup via sibling bracket state.
- `check_bracket_fills()` returns `list[dict]`, handles partial (bracket pop only) vs full (position pop)
- `partial_close_position()`: market order fallback for bar-close partial exits
- Pre-existing `response.order.id` bug fixed → `response.complex_order.id`

### Runner (`runner.py`)
- `process_bar()` loops over fill list. Partial TP1 → `_partial_close()` (position stays open). Full exit → `force_close()` (returns). Guard against stale partial on flat position.

### Failure handling
- OCO #1 fails → single stop for all qty, `_oco_failed_sids` (bar-close TP re-enables)
- OCO #1 OK, OCO #2 fails → keep OCO #1, simple stop for runner, `_oco_failed_sids`
- Both fail → `_protection_failed_sids`, engine halt

### Code review (5 agents, 2 rounds)
Round 1 found 5 bugs (1 CRITICAL, 1 HIGH, 1 MEDIUM, 2 LOW). All fixed.
Round 2 verified all 7 edge-case scenarios pass. No orphaned exchange orders.
See `memory/trading_code_review.md` Round 7 for full findings.

### Step 0 validation — PASSED (Feb 27, 2026)
Validated that tastytrade accepts two simultaneous `NewComplexOrder` OCO brackets on the
same underlying position. Test used M6E (Micro Euro FX) due to account margin constraints:
- Bought 2x /M6EH6 @ 1.1821
- OCO #1 (1x LIMIT @ 1.1851 + STOP @ 1.1771) → accepted, ID 5751306
- OCO #2 (1x LIMIT @ 1.1881 + STOP @ 1.1771) → accepted, ID 5751307
- Both cancelled cleanly, position closed

**Result**: Dual OCO brackets confirmed working. No fallback needed.
Script: `live_trading/validate_dual_oco.py` (reverted to MES for future use).
Next step: paper trade MES v2 with dual OCO brackets on MES once account is funded.

## Dashboard Sizing Controls (Feb 23, 2026)

Added dashboard UI to override entry_qty and partial_qty per strategy without restarting:

- **MES_V2**: Shows `Entry: [-][2][+]  TP1: [-][1][+]` — can scale to e.g. Entry:4, TP1:2
- **MNQ strategies**: Shows `Qty: [-][1][+]` — single-contract control
- Strategy properties `active_entry_qty` / `active_partial_qty` check override first, fall back to config
- Blocked while positioned (prevents mid-trade qty confusion)
- Cleared on Force Resume and daily reset
- Entry signal logic completely untouched — only affects contract count and exit leg management
- **Open question**: Should Entry=1 on partial strategies be allowed (implicitly disabling TP1)? Currently blocked. Revisit after paper trading.

## Breakeven-After-N-Bars (BE_TIME) — Feb 27, 2026

### Problem
MES v2 trades can stall 200-300 bars without hitting TP=20 or SL=35, then drift to EOD at a loss. 50% of exits are EOD with avg P&L = -2.4 pts.

### Original coarse sweep (Feb 27)
After N bars, close at next open (exit_reason=BE_TIME). Pre-filter: closing early frees cooldown, enabling re-entries. Implemented in `run_backtest_tp_exit()` param `breakeven_after_bars`.

| N | OOS Sharpe | OOS PF | OOS MaxDD | OOS P&L | BE_TIME | Re-entries |
|---|-----------|--------|-----------|---------|---------|------------|
| 0 | 0.318 | 1.050 | -$1,028 | +$335 | 0 | — |
| 150 | **1.060** | **1.186** | -$1,179 | +$1,149 | 70 | +37 |
| 275 | 0.808 | 1.133 | **-$999** | +$829 | 29 | +3 |
| 300 | 0.470 | 1.075 | -$1,041 | +$486 | 21 | +1 |

Original recommendation was N=275. But runner analysis (Mar 4) showed this was too conservative — median winning runner resolves TP1→TP2 in 20 bars, average runner goes negative 7 bars after TP1. The original sweep was too coarse (jumped from 150 to 275, skipping the 30-120 range).

### Fine-grained re-sweep (Mar 4)

| N | OOS Sharpe | OOS PF | OOS MaxDD | OOS P&L | BE_TIME | Re-entries |
|---|-----------|--------|-----------|---------|---------|------------|
| 0 | 0.318 | 1.050 | -$1,028 | +$335 | 0 | — |
| 30 | 0.942 | 1.180 | -$674 | +$834 | 241 | +117 |
| 45 | 0.979 | 1.186 | -$774 | +$881 | 209 | +96 |
| 60 | 0.779 | 1.143 | -$574 | +$770 | 186 | +87 |
| **75** | **1.502** | **1.292** | **-$673** | **+$1,563** | 160 | +76 |
| 90 | 1.159 | 1.208 | -$918 | +$1,248 | 131 | +64 |
| 105 | 1.047 | 1.185 | -$930 | +$1,130 | 113 | +52 |
| 120 | 0.624 | 1.106 | -$1,033 | +$679 | 99 | +42 |
| 150 | 1.060 | 1.186 | -$1,179 | +$1,149 | 70 | +37 |
| 275 | 0.808 | 1.133 | -$999 | +$829 | 29 | +3 |

### Decision (revised Mar 4)
**N=75.** Best OOS Sharpe (1.502), best OOS PF (1.292), best OOS MaxDD (-$673). IS still strong (Sharpe 1.986, PF 1.394). 76 re-entries are net positive — they're real signals freed by closing stale positions. Aligns with runner data: at bar 75, if the trade hasn't resolved, it almost certainly won't.

### Runner analysis supporting N=75 (Mar 4)
Of 382 trades, 230 reach TP1 (+10 pts). Of those, 147 (64%) hit TP2, 83 (36%) don't.
- Median winning runner: 26 bars to TP1, then 20 bars to TP2 (total ~46 bars)
- Average runner goes negative at bar 7 after TP1
- Fast TP1 (0-10 bars): 77% hit TP2. Slow TP1 (60+ bars): 49% — coin flip.
- Morning TP1 (10-12): 71-77% hit TP2. Afternoon 15:00+: only 17%.
- 89% of failed runners exit at EOD for a loss. They just drift.

### Status — IMPLEMENTED (Mar 4)
**Config**: `breakeven_after_bars=75` on MES_V2 in `config.py`
**Live engine**: Bar counter in `strategy.py` on_bar(), after TP/trail exits. Uses `(bar_idx - entry_bar_idx - 1)` to match backtest's `(i-1) - entry_idx`. Fires as bar-close only (not intra-bar). Off-by-one found and fixed during code review.
**Backtest**: `MESV2_BREAKEVEN_BARS = 75` in `generate_session.py`, passed through `run_and_save_portfolio.py`.
**ExitReason**: `BE_TIME` added to events.py enum.
**Safety**: BE_TIME exits handled correctly — no SL counter increment, P&L tracked normally.
**OCO brackets**: Cancelled correctly on BE_TIME exit (tastytrade_broker `_cancel_bracket` handles composite keys).
**MNQ**: Unaffected (`breakeven_after_bars=0` default).

## TP1 Re-sweep — Mar 11, 2026

Full 2-contract bar-by-bar simulation revealed TP1=10 was suboptimal:
- Only 39% of trades reached +10 pts for TP1 fill
- 78% of SLs had MFE < 10 pts → both contracts ate full SL (-$387 avg)

**TP1=6 adopted**: PF 1.239→1.341 (+8.2%), Sharpe 1.27→1.63 (+28%), TP1 fill rate 39%→60%, MaxDD 17% better. IS/OOS consistent. Config updated `partial_tp_pts=6`.

Also red-teamed VIX [20-25] and entry delay +30min gates for MES — both FAILED bootstrap significance (p=0.29, p=0.33). Breakeven escape rejected (IS/OOS divergence).

Full details: `memory/mes_v2_tp1_sweep.md`

## Caveats

- Backtest uses bar-close TP detection, not intra-bar. Live paper mode uses intra-bar quotes.
- **Next step**: Paper trade MES v2 with TP1=6 + BE_TIME=75 to validate before live.
