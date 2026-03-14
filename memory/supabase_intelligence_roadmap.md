---
name: Supabase Intelligence Layer Roadmap
description: Phased plan for all Algo Trading Expert suggestions — enrichment fields, views, and functions that make future intelligence queries possible
type: project
---

# Supabase Intelligence Layer — Phased Roadmap

From the Algo Trading Expert's review of Phase 1 schema. These are the enrichments and views that turn a trade database into an analytical engine.

## Phase 2A: Quick Schema Enrichments (next session, ~30 min)

These are single-column additions or generated columns. No engine changes needed — just SQL ALTER TABLE + future population.

### 1. Commission column on `trades`
- **What**: `commission numeric(8,2)` — actual commission paid per trade
- **Why**: `pnl_net` currently approximates. Real commission varies by broker, account tier, and routing. Needed for accurate P&L at scale.
- **How**: ALTER TABLE trades ADD COLUMN commission numeric(8,2); Engine populates from broker fill report. Until then, NULL (pnl_net still accurate enough with fixed $0.52/side MNQ, $1.25/side MES).
- **Populate**: Read from tastytrade fill data (available in `OrderFill` response). Set `pnl_net = pnl_gross - commission`.

### 2. `rth_close` on `gate_state_snapshots` — populate or remove
- **What**: Column exists but is never populated
- **How**: In `_gate_state_rows()`, read `rth_close` from SafetyManager's RTH session tracking (already tracked for ADR gate). Either populate it or drop the column to avoid confusion.
- **Decision**: Populate — needed for daily bar reconstruction and gap analysis.

### 3. `day_of_week` generated column — DONE (this session)
- Already added: `day_of_week smallint GENERATED ALWAYS AS (EXTRACT(isodow FROM trade_date)::smallint) STORED`
- 1=Monday through 5=Friday. Enables "Friday performance is worse" queries with zero compute.

## Phase 2B: Entry Context Enrichment (~1 hour)

Fields that capture the market state at the moment of entry. Enables "what conditions lead to winners vs losers?" analysis.

### 4. `concurrent_open_positions` field
- **What**: `concurrent_positions smallint` on `trades` — how many other positions were open when this trade entered
- **Why**: Portfolio heat matters. "Do we perform worse when 3+ strategies are in trades simultaneously?"
- **How**: In `_open_position()` or the entry fill handler in runner.py, count `sum(1 for s in strategies if s.state.position != 0)`. Write to TradeState, flow through to TradeRecord → db_logger.
- **Complexity**: Low. One line in runner.py post-fill.

### 5. `streak_at_entry` field
- **What**: `streak_at_entry smallint` on `trades` — current consecutive win/loss streak when this trade was entered
- **Why**: "Do we take worse trades when on a losing streak?" (tilt detection). "Do we overtrade after winning streaks?"
- **How**: Track per-strategy running streak in `IncrementalStrategy` (already has `self.trades` list). Write to TradeState at entry.
- **Complexity**: Low. Counter incremented in `_close_position`, reset on streak break.

### 6. `entry_bar_range` field
- **What**: `entry_bar_range double precision` on `trades` — the 1-min bar's high-low range at entry
- **Why**: Wide entry bars correlate with slippage and volatility. "Are our SL hits concentrated on wide-bar entries?"
- **How**: `bar.high - bar.low` at entry time. One line in `_open_position()`.
- **Complexity**: Trivial.

### 7. `entry_spread` for live trades
- **What**: `entry_spread double precision` on `trades` — bid-ask spread at entry
- **Why**: Real execution quality analysis. Paper trades have zero spread; live trades may have 0.25-2.0 pts spread on MNQ.
- **How**: Requires reading spread from DXLink quote data at fill time. Tastytrade AlertStreamer already receives quotes. Store `ask - bid` at the moment of fill.
- **Complexity**: Medium. Need to capture spread from quote stream, not just OHLCV bars.
- **Defer until**: Live trading at scale (spread matters more with multiple contracts).

## Phase 2C: Analytical Views (~1 hour)

### 8. `tp1_runner_pairs` view — DONE (this session)
- Already added. Links TP1 partial + runner legs via `trade_group_id`.
- Enables: "TP1 fills 60% of the time. When it fills, the runner adds +$X on average."

### 9. Statistical Z-score in `live_vs_backtest`
- **What**: Replace fixed 5%/10% deviation bands with proper binomial Z-score
- **Why**: 5% WR deviation on 20 trades is noise. 5% deviation on 200 trades is a real signal. Current bands don't account for sample size.
- **How**: Z = (live_wr - backtest_wr) / sqrt(backtest_wr * (1-backtest_wr) / n). Status based on Z thresholds (|Z|<1.96 = GREEN, <2.58 = YELLOW, else RED).
- **SQL**: Update `live_vs_backtest` view with the Z-score formula. Pure SQL, no engine changes.
- **Complexity**: Low. One view rewrite.

### 10. Time-of-day performance bucketing view
- **What**: New view `tod_performance` — aggregate stats by 30-min or 1-hour entry time buckets
- **Why**: "We lose money in the first 30 minutes" or "our best trades are 11:00-12:00 ET". Already known from backtests but needs live validation.
- **How**: `date_trunc('hour', entry_time AT TIME ZONE 'America/New_York')` or `EXTRACT(hour FROM ...)` grouping. Win rate, PF, avg P&L per bucket per strategy.
- **SQL**: Pure view on `trades` table. No engine changes.
- **Complexity**: Low.

### 11. Cross-strategy correlation view
- **What**: New view showing daily P&L correlation between strategy pairs
- **Why**: "Are vScalpA and vScalpC actually diversifying or just doubling down on the same trades?"
- **How**: Self-join `daily_stats` on `trade_date`, compute Pearson correlation of daily P&L across rolling windows. Complex SQL but doable with window functions.
- **Complexity**: Medium. Correlation in SQL is verbose but well-documented.
- **Alternative**: Compute offline in Python and store in a results table. Faster to build, easier to extend.

### 12. Portfolio daily view — DONE (this session)
- Already added as `portfolio_daily`. Aggregates `daily_stats` across strategies.

### 13. Drawdown/HWM view — DONE (this session)
- Already added as `drawdown_hwm`. Running cumulative P&L, high water mark, drawdown amount and %.

## Phase 2D: Enhanced Functions (~30 min)

### 14. Enhanced `f_morning_briefing` — PARTIALLY DONE (this session)
- Added: `yesterday_portfolio`, `drawdown_status`, `yesterday_blocked`
- **Still needed**: Rolling 30-day performance per strategy (requires querying `rolling_performance` matview for latest row per strategy). Add as `rolling_30d` key.
- **Complexity**: Low. One more subquery in the function.

### 15. `config_id` FK population
- **What**: `trades.config_id` is defined but never populated
- **Why**: Links each trade to the exact strategy config version that produced it. Enables "did the config change improve results?"
- **How**: At trade entry, look up the active `strategy_configs` row for this `strategy_id`. Requires `strategy_configs` to be populated first (not yet done — no backfill script yet).
- **Dependency**: Needs the backfill script to seed `strategy_configs` with current configs. Until then, column stays NULL.
- **Complexity**: Low once configs exist. One query in db_logger or runner.

## Phase 2E: Blocked Signal Enrichment (~30 min)

### 16. `cf_mfe_pts` / `cf_mae_pts` on `blocked_signals`
- **What**: Two new columns for counterfactual MFE and MAE
- **Why**: "The gate blocked a signal that would have hit TP and then run 20 more points" vs "the signal would have barely moved before SL." Counterfactual P&L alone doesn't capture opportunity quality.
- **How**: ALTER TABLE blocked_signals ADD COLUMN cf_mfe_pts double precision, ADD COLUMN cf_mae_pts double precision;
- **Populate**: Batch replay script (future session) computes these alongside `cf_pnl_pts`.
- **Complexity**: Trivial schema change. Computation depends on batch replay infrastructure.

### 17. `gate_leledc_active` on `blocked_signals` — DONE (this session)
- Already added to schema, runner.py, and db_logger.

## Build Order — Remaining

| Session | Items | Time | Dependencies |
|---------|-------|------|--------------|
| Next | #11 Cross-strategy correlation view | 30 min | Daily stats populated |
| Backfill | #15 config_id FK population | 15 min | strategy_configs seeded |
| Live scaling | #7 entry_spread | 30 min | Live trading + DXLink quotes |

Phases 2A, 2B, 2C all complete. Only 3 items remain.

## What's Already Done (Mar 11 Sessions)

### Pre-2A (from prior session)
- `trade_group_id`, `signal_price` on trades + TradeRecord
- Gate state capture at entry time (not exit)
- Lambda closure fixes, upsert dedup, retry logic
- `day_of_week` generated column on trades
- `gate_leledc_active` on blocked_signals (schema + runner + db_logger)
- `exit_time` index on trades
- `portfolio_daily` view (with trade-level PF, pre-agg CTE for multi-instrument safety)
- `drawdown_hwm` materialized view (with CONCURRENTLY refresh)
- `tp1_runner_pairs` view
- Enhanced `f_morning_briefing` (portfolio total, drawdown status, blocked count, rolling perf)
- Correction zero-rows-matched warning
- Queue size 1000 → 5000

### Phase 2A (completed + reviewed by 3 agents)
- `commission numeric(8,2)` column on trades
- `entry_bar_range` field (TradeState → TradeRecord → db_logger → SQL)
- `rth_close` population in gate state snapshots (SafetyManager + db_logger)
- `cf_mfe_pts` / `cf_mae_pts` columns on blocked_signals
- Seed init fixed (`"close": None`)
- `_blocked_to_row` None guard + strip

### Phase 2B (completed + reviewed by 2 agents)
- `concurrent_positions` (runner injects post-fill, TradeState → TradeRecord → SQL)
- `streak_at_entry` (IncrementalStrategy._streak counter, captured at entry)
- `concurrent_positions` reset in `_close_position()` for hygiene

### Phase 2C (completed + reviewed by 2 agents)
- `live_vs_backtest` Z-score upgrade (binomial proportion test, GREEN/YELLOW/RED)
- `live_vs_backtest` now groups by source (separate live vs paper rows)
- `tod_performance` view (hourly ET entry buckets, WR/PF/avg P&L/MFE/MAE)
- `rolling_performance` in morning briefing (latest per strategy, prefers live source)

### Review coverage
- 8 review agents total (3 for 2A, 2 for 2B+2C, 3 in prior session)
- All CRITICAL/HIGH findings fixed
- Composite index for trade_group_id (includes is_partial + source)
