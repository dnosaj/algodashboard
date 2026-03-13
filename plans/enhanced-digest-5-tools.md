# Enhanced Digest Agent: 6 Forensic Tools — Implementation Plan

**Status: Phase 6 COMPLETE — ALL PHASES DONE. Ready for deployment.**
**Date: Mar 13, 2026**
**Current workflow phase: COMPLETE**

## Phase 5 Review Findings (all applied)
### Round 1 (initial review)
- H1: Near-gate-miss P&L deduplication — added `unique_trades` count + deduped `net_counterfactual_pnl`
- H2: Removed "MES prior-day levels 5-10pts" from near-gate-miss description (belongs in `get_level_proximity`)
- H3: Documented `daily_limit = 650` constant in entry_clustering with update reminder
- M1: SL velocity BE exit heuristic now checks `abs(pnl) < 5.0` to avoid misclassifying full runner SL exits
- M2: Prompt generic SL thresholds replaced with reference to strategy-specific thresholds
### Round 2 (from full agent reports)
- S1 CRITICAL: `get_market_regime` hist_ranges index correlation bug — fixed with `hist_range_by_date` dict
- S2 HIGH: ADR near-miss now direction-aware (checks `side` + sign of `adr`)
- S3 HIGH: `get_level_proximity` `adr or 1` → `None` when unavailable (no misleading values)
- Q1 MEDIUM: `get_level_proximity` now includes VWAP + opening range distances

---

## Vision (Phase 1 — Confirmed)

Enhance the existing Digest Agent with forensic analysis tools that let it report *why* things happened, not just *what* happened. No new agent, no new tables. Minimal engine additions (VWAP, opening range, leledc count — all trivial compute). Structured `flags_for_frontier` output for the Frontier Agent pipeline. Feed real insights to Jason — quality-gated, not quantity-gated.

---

## Tool Set (6 tools — nothing deferred)

### Tool 1: `get_sl_velocity(date, days=1)`
- Classify SL exits by speed: rapid/normal/gradual
- **Strategy-specific thresholds** (Domain Specialist):
  - vScalpB: rapid <3 bars, normal 3-10, gradual 10+
  - V15: rapid <3 bars, normal 3-15, gradual 15+
  - MES_V2: fast <30 bars, normal 30-75, extended 75+ (BE_TIME boundary)
  - vScalpC: rapid <3 bars = TP1 never filled (2x loss)
- **Optional rolling window**: `days` param for multi-day aggregation
- Include `mae_velocity` = mae_pts / bars_held (Creative)
- Include bars_held distribution, not just counts

### Tool 2: `get_entry_clustering(date)`
- Flag simultaneous entries across strategies (same 1-min bar)
- **Enhanced with dollar-risk exposure** (Creative + Product Strategist):
  - `combined_sl_risk`: sum of SL distances * point values * qty
  - `max_concurrent_dollar_risk`: peak $ at risk vs $600 daily limit
  - `correlation_realized`: did clustered trades all win or all lose?
- Distinguish "correlated cluster" (V15+vScalpC, same signal) vs "coincident cluster" (MNQ+MES, different signals)
- Flag when combined SL risk exceeds 50% of daily circuit breaker ($300+)
- **Pre-launch risk catch**: A(1)+B(1)+C(2)+MES(2) worst-case simultaneous SL = $610, exceeds $600 limit

### Tool 3: `get_near_gate_miss(date)`
- Trades that entered just outside gate thresholds
- **Cross-reference with trade outcome** (Domain Specialist): only flag near-misses that LOST money
- **Counterfactual P&L** (Creative): "If threshold were X instead of Y, this $Z loss would have been blocked"
- Near-miss windows per gate (Domain Specialist):
  - VIX: within 1.0 of 19 or 22
  - ATR: within 10% above 263.8
  - ADR ratio: within 0.05 of 0.3
  - Leledc: count at 7 or 8 (threshold 9) — uses per-trade `gate_leledc_count` (engine change, see below)
  - Prior-day levels (MES): 5-10pts from level (1-2x buffer)
- Noise filter: skip near-misses on trades that won

### Tool 4: `get_level_proximity(date)`
- Distance from each trade's entry to nearest prior-day level
- **Express as % of TP** (Domain Specialist): "50% of TP consumed by level proximity"
- **Signed distance with direction** (Creative): approaching vs departing level
- **ADR-normalized distance** (Creative): distance_pts / adr_value for cross-volatility comparison
- Report distance to EACH level individually (not just nearest)
- Depends on `gate_state_snapshots` having data — prerequisite to verify/fix

### Tool 5: `get_day_summary_extended(date)`
- Session OHLC + VWAP close + opening range H/L + trend score + body%
- OHLC already in `gate_state_snapshots`. VWAP, opening range, trend score from new engine fields.
- Day archetype classification: TREND / RANGE / BREAKOUT / REVERSAL / MIXED (Creative)
- VWAP position: close above/below VWAP and by how much

### Tool 6: `get_multi_day_context(days=10)`
- Rolling session context from `gate_state_snapshots` data (last N days)
- Returns: OHLC, range, body%, day type, direction, VWAP, opening range for each day
- Lets Digest say "3rd consecutive choppy day" or "range expanding for 5 days"
- Temporal patterns the single-day tools can't see

---

## Engine Changes (all trivial)

### In `safety_manager.py` — `on_bar()` additions (after existing gate swaps)

1. **VWAP accumulation** (~2 lines per bar):
   ```
   self._vwap_num[inst] += typical_price * volume
   self._vwap_den[inst] += volume
   ```
   Reset on date change. Final VWAP = num / den. Written to gate_state_snapshots at daily reset.

2. **Opening range tracking** (~4 lines per bar):
   ```
   if et_mins < 630 and not or_finalized:  # Before 10:30 ET
       or_high = max(or_high, bar.high)
       or_low = min(or_low, bar.low)
   ```
   Finalized at 10:30. Written to gate_state_snapshots at daily reset.

3. **Leledc count at entry** (~1 line in strategy.py):
   ```
   trade.gate_leledc_count = max(safety._leledc_bull_count.get(inst, 0), safety._leledc_bear_count.get(inst, 0))
   ```

### In `db_logger.py` — `_gate_state_rows()` additions
- Add `vwap_close`, `opening_range_high`, `opening_range_low`, `body_pct`, `trend_score` to snapshot row
- Add `gate_leledc_count` to `_trade_to_row()`

### In `engine/events.py`
- Add `gate_leledc_count: int | None = None` to TradeRecord dataclass

---

## Schema Migration

```sql
-- 005_forensic_tools.sql
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS vwap_close double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS opening_range_high double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS opening_range_low double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS body_pct double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS trend_score double precision;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS gate_leledc_count smallint;
```

Purely additive. Nullable columns. Safe while engine is running.

---

## `flags_for_frontier` — Structured Pipeline Output

Added to EOD `save_digest` content schema. Each flag is a structured object:

```json
{
  "flags_for_frontier": [
    {
      "tool": "get_near_gate_miss",
      "hypothesis": "Tighten ADR gate from 0.30 to 0.25 for V15",
      "evidence": "3 near-miss losses at ADR 0.26-0.29 this week",
      "suggested_test": "sweep adr_directional_threshold 0.20-0.35 step 0.05",
      "priority": "high",
      "sample_size": 7,
      "recurrence": 3
    }
  ]
}
```

- `recurrence` computed by scanning last N digests' flags via `get_recent_digests`
- Minimum `sample_size` reported so Frontier (and Jason) can judge statistical weight
- EOD prompt instructs: generate flags when forensic tools reveal actionable patterns. No minimum, no maximum — quality only.

---

## Prompt Strategy

- **Quality-gated, not quantity-gated**: Surface every genuine insight. No cap on forensic bullets. Skip routine confirmations ("no SL velocity anomalies" = don't mention it).
- **"Quiet day" clause**: <3 trades = skip forensic tools entirely (not enough data to be meaningful)
- **Conditional tool calls**: Only call `get_sl_velocity` if SL exits exist, only call `get_entry_clustering` if 3+ trades, etc.
- **Exception-based**: "V15 SL velocity shifted from 34% rapid to 60% rapid this week" = include. "vScalpB SL velocity normal" = omit.

---

## Prerequisites

1. **Verify `gate_state_snapshots` is populating.** Tools 4, 5, 6 depend on it. If empty, debug and fix first.
2. **Fix if not populating.** Tools 1, 2, 3 work without it (pure trades queries).

---

## Code Changes Summary

| File | Change | Magnitude |
|------|--------|-----------|
| `agents/digest/tools.py` | Add 6 tool definitions + implementations + TOOL_DISPATCH | ~300 lines |
| `agents/digest/prompts.py` | Add tool references to EOD process + flags_for_frontier schema + quality-gated reporting | ~30 lines |
| `engine/safety_manager.py` | VWAP accumulation (2 lines/bar), opening range (4 lines/bar), reset on date change | ~15 lines |
| `engine/strategy.py` | Populate gate_leledc_count at entry | 1 line |
| `engine/events.py` | Add gate_leledc_count to TradeRecord | 1 line |
| `engine/db_logger.py` | Add 5 columns to _gate_state_rows, 1 to _trade_to_row | ~10 lines |
| `supabase/migrations/005_forensic_tools.sql` | ALTER TABLE x2 (6 new columns total) | 6 lines |

**Total: ~360 lines of new code, ~15 lines of engine changes.**

---

## Deployment Plan

1. Push all code to `main` (after hours)
2. Run migration `005_forensic_tools.sql` in Supabase SQL Editor
3. Manual `workflow_dispatch` with `dry_run: true` to verify agent + new tools
4. Manual `workflow_dispatch` without dry_run to verify Supabase write + Intel panel
5. Engine picks up safety_manager/db_logger changes on next natural restart
6. First daily reset writes VWAP/OR/leledc to gate_state_snapshots
7. Next EOD digest uses all 6 tools + flags_for_frontier

---

## Key Decisions (revised after Jason review, Mar 13)

1. **VWAP and opening range IN the engine** — 2 float additions per bar is not "hot path danger." It's trivial compute. Overblown risk assessment corrected.
2. **`flags_for_frontier` ships now** — Frontier Agent is coming soon. Build the structured handoff now so it's ready.
3. **`gate_leledc_count` on trades** — 1 line at entry time. No reason to defer.
4. **6 tools, not 5** — both `get_day_summary_extended` (single-day with VWAP/OR) AND `get_multi_day_context` (temporal patterns). They serve different purposes.
5. **No cap on forensic insights** — quality-gated, not quantity-gated. If there are 7 genuine insights, show all 7. If nothing interesting, say nothing.
6. **Strategy-specific SL velocity thresholds** — Domain Specialist validated, not one-size-fits-all.
7. **Dollar-risk exposure in clustering** — $610 worst-case vs $600 limit is a real pre-launch finding.
8. **Outcome-weighted near-gate misses** — only flag losers, skip winners.

---

## Risk Register (recalibrated)

| Risk | Severity | Mitigation |
|------|----------|------------|
| gate_state_snapshots empty | HIGH | Verify/fix before building tools 4, 5, 6 |
| LLM hallucinating patterns from N=3-7 | MEDIUM | Quality-gated reporting, quiet day clause, recurrence tracking in flags |
| Token budget bloat (24 tools) | LOW | Conditional tool calls, measure before/after. Current run was 258K/300K budget — some headroom. |
| Digest getting too long on busy days | LOW | Exception-based reporting handles this naturally |
| Threshold duplication (config.py vs tools.py) | LOW | Document, accept for now |
| Tool call cost increase | LOW | Expected ~$0.90-1.10/run vs current ~$0.83 |

---

## Phase 3 Review Findings (Full Team Review, Mar 13)

**Verdict: APPROVE with revisions.**

10 review agents (2 rounds x 5 roles: Architect, Senior Dev, Domain+Product, QA+DevOps, Critic) performed deep codebase analysis. Findings synthesized below by consensus strength.

---

### MUST-FIX Before Execution (universal consensus — all 10 agents agree)

**1. Opening range needs RTH lower bound**
- Plan says `if et_mins < 630`. Must be `600 <= et_mins < 630` (10:00-10:30 ET).
- Without lower bound, pre-market/overnight bars contaminate the opening range.
- Fix: 1 line.

**2. `gate_leledc_count` injection is in runner.py, not strategy.py**
- Strategy has no `safety` reference. Gate state injection happens in `runner.py:576-581`.
- Correct implementation: (a) `TradeState` field in strategy.py, (b) injection in runner.py, (c) `TradeRecord` field in events.py, (d) passthrough in `_close_position` + `_partial_close` in strategy.py, (e) `_trade_to_row` in db_logger.py.
- Total: ~6 lines across 4 files, not "1 line in strategy.py."

**3. VWAP div-by-zero guard**
- Synthetic bars from IntraBarExitMonitor and runner have `volume=0`. These are deduped and won't reach VWAP in normal flow, but final division needs `if den > 0` guard.
- Also: VWAP should be **RTH-only** (10:00-16:00 ET) — standard VWAP definition. Plan's pseudocode has no RTH filter.
- Fix: `if volume > 0 and RTH:` guard + `if den > 0` at finalization. ~3 lines.

**4. `trend_score` is undefined**
- Mentioned in migration + snapshot but no computation specified anywhere.
- **Decision: DROP from migration.** Compute `body_pct` in the tool (like `get_market_regime` already does). `body_pct` is derivable from existing OHLC columns — storing it is redundant. `trend_score` adds nothing `body_pct * sign(close-open)` doesn't.
- Migration drops from 6 columns to 4 (vwap_close, opening_range_high, opening_range_low, gate_leledc_count). All 4 carry genuinely novel data.

**5. Verify `gate_state_snapshots` with a 30-second query**
- Code path exists: `runner.py:811` → `db_logger.snapshot_gate_state()` → `_gate_state_rows()`.
- MEMORY.md says "not populated" but multiple agents traced the code and believe it IS populating.
- **Before coding anything**, run: `SELECT COUNT(*), MIN(snapshot_date), MAX(snapshot_date) FROM gate_state_snapshots`. This resolves the #1 risk item instantly.

---

### SHOULD-FIX (strong consensus — 7+ of 10 agents agree)

**6. MES has no RTH OHLC in snapshots**
- `_gate_state_rows` reads OHLC from `_adr_rth_session` — only populated for MNQ (has `adr_lookback_days`).
- MES uses `_current_rth` for prior-day level tracking, but its OHLC isn't pulled into the snapshot.
- Fix: Add fallback to `_current_rth` in `_gate_state_rows`. ~3 lines.

**7. `get_day_summary_extended` overlaps with `get_market_regime`**
- Both query `gate_state_snapshots`, both compute day type, both return OHLC.
- **Decision: Extend `get_market_regime`** to include `vwap_close`, `opening_range_high`, `opening_range_low`. Add a `days` parameter (default 1) for multi-day context.
- This merges tools 5+6 into the existing tool. Tool count: 22 instead of 24 (18 existing + 4 new).
- Reduces LLM confusion from two partially-overlapping day-summary tools.

**8. SL velocity thresholds need recalibration (Domain Specialist finding)**
- **V15**: Plan says rapid <3 bars, but reversal diagnostic shows median=6.5 bars, only 34.2% within 3 bars. **Change to: rapid <5, normal 5-15, gradual 15+.**
- **MES_V2**: Plan says fast <30 bars, but diagnostic shows 100% gradual, median=100 bars. <30 would capture nothing. **Change to: fast <60, normal 60-100, extended 100+.**
- vScalpB (<3 bars) and vScalpC (<3 bars) are correct.

**9. `flags_for_frontier` schema improvements**
- Add `strategy_id: str | null` — Frontier needs to know which strategy to sweep.
- Add `flags_for_frontier` to the `save_digest` tool description (currently absent).
- Add `gate_leledc_count` to `_TRADE_COLUMNS` in tools.py.
- Add prompt guardrail: "If previous digests lack `flags_for_frontier`, treat recurrence as 0."

**10. Near-gate-miss should show BOTH outcomes, not losers-only**
- Filtering to losers-only introduces systematic bias toward tighter gates.
- A gate that lets near-miss losers AND near-miss winners through is working correctly.
- **Change: Report both outcomes with net counterfactual P&L.** Let the net number speak.

**11. SL distance for dollar-risk in clustering tool**
- Not in trades table — it's in strategy config (`max_loss_pts`).
- Hardcode per-strategy constants in the tool (V15=40, vScalpB=10, vScalpC=40, MES=35). Acceptable — these change very rarely.

---

### NICE-TO-FIX (minority or low-impact findings)

**12. ATR near-miss window too wide**: 10% above 263.8 = up to 290.2, captures ~15% of days. Tighten to 5% (up to 277.0).

**13. vScalpC runner BE exits misclassified**: SL-at-breakeven after TP1 is a success, not a loss. Tool 1 should filter `exit_reason=SL` with `is_runner=True` as BE exits.

**14. Drop BREAKOUT/REVERSAL day archetypes**: Keep existing 3-type system (TREND/CHOPPY/MIXED). Detecting breakouts/reversals needs intraday path tracking not proposed.

**15. Rename "Opening Range" to "Initial Balance"**: Standard terminology for 30-min RTH window. Code comment + plan terminology.

**16. $610 worst-case is actually $619.16 with commissions**: V15=$81.04, vScalpB=$21.04, vScalpC=$162.08, MES=$355.00. Total=$619.16.

**17. VWAP partial-day after mid-day restart**: Accumulators reset to zero. Consider writing `vwap_bar_count` alongside `vwap_close` so tools can assess reliability.

**18. `get_multi_day_context` contradicts regime detection research**: MEMORY.md explicitly says "Regime detection doesn't work." Multi-day regime patterns are exactly the signal that failed walk-forward. If kept (merged into `get_market_regime`), scope as "attribution only, not predictive."

**19. Leledc count off-by-one**: Gate updates use bar[i], but entry decisions use bar[i-1]. The count captured at entry time will be bar[i]'s count. Document this and adjust near-miss thresholds by +1 if needed.

---

### Concerns Killed (overblown per pragmatic risk feedback)

1. **"Engine hot path danger"** — 2 float additions + 1 comparison per bar in a function that already does 50+ operations. Not a risk.
2. **"Threshold duplication"** — Tools hardcode near-miss windows related to config.py values. These are analysis classification boundaries, not duplicated trading thresholds. Acceptable.
3. **"Token budget bloat"** — 42K headroom. Adding 6 tools consumes ~15-18K additional tokens (definitions + results). Estimated 276K/300K on typical days. Conditional tool calls keep busy days within budget.
4. **"Migration safety"** — All `ADD COLUMN IF NOT EXISTS` with nullable defaults. Zero-downtime. No table rewrite. Safe while engine is running.
5. **"Digest getting too long"** — Exception-based reporting + quality gating + `max_tokens=4096` per response. Self-correcting.
6. **"Rollback complexity"** — Agent tools (tools.py) are independently revertible from engine changes. Clean separation. Zero live trading impact in any failure mode.

---

### Revised Estimates

| Item | Plan Estimate | Revised Estimate | Reason |
|------|--------------|------------------|--------|
| Engine changes | ~15 lines | ~30-35 lines | Init dicts, RTH guards, reset, snapshot, leledc passthrough across 4 files |
| Tool implementations | ~300 lines | ~350-400 lines | Clustering dollar-risk + near-gate-miss multi-join are more complex |
| Migration columns | 6 columns | 4 columns | Dropped body_pct + trend_score (derivable/undefined) |
| Tools count | 6 new (24 total) | 4 new (22 total) | Merged tools 5+6 into existing `get_market_regime` |
| **Grand total** | ~360 lines | ~400-450 lines | Still very manageable |

---

### Revised Deployment Plan

1. **Verify gate_state_snapshots** — single Supabase query. 30 seconds.
2. Push all code to `main` (after hours)
3. Run migration `005_forensic_tools.sql` **immediately** (before engine restart)
4. Manual `workflow_dispatch` with `dry_run: true` to verify agent + new tools
5. Manual `workflow_dispatch` without dry_run to verify Supabase write + Intel panel
6. Engine picks up safety_manager/db_logger changes on next natural restart
7. First daily reset writes VWAP/OR/leledc to gate_state_snapshots
8. Next EOD digest uses all forensic tools + flags_for_frontier

**Split deployment option**: If gate_state_snapshots IS empty, ship tools 1-4 first (pure trades queries), fix the snapshot pipeline, then extend `get_market_regime` with VWAP/OR.

---

### Decisions (Jason, Mar 13)

1. **$619 vs $600 daily limit**: **Raise to $650.** Config change in `config.py`.
2. **Near-gate-miss**: **Show both outcomes with net P&L.** No losers-only filter.
3. **Merge tools 5+6 into `get_market_regime`**: **Yes.** 22 tools total (18 existing + 4 new).
