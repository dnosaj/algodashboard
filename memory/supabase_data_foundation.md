---
name: Supabase Data Foundation
description: Complete implementation details for Phase 1 Supabase infrastructure — schema, backfill scripts, dashboard analytics, known limitations, review findings
type: project
---

# Supabase Data Foundation (Mar 11-12, 2026)

## What Was Built

The "self-awareness layer" for the trading system — a Supabase database that enables drift detection, gate effectiveness analysis, and portfolio analytics. Every trade, blocked signal, and research run becomes part of a permanent, queryable institutional memory.

### Infrastructure

| File | Purpose |
|------|---------|
| `supabase/migrations/001_initial_schema.sql` | Full DDL: 10 tables, 4 views, 3 functions, indexes, RLS |
| `supabase/migrations/002_review_fixes.sql` | Standalone patch: blocked signal dedup index + anon RLS policies |
| `engine/db.py` | Supabase client singleton (lazy init from env vars, graceful degradation) |
| `engine/db_logger.py` | Async EventBus subscriber: trades, blocked signals, gate snapshots |
| `supabase/seed_strategy_configs.py` | Seeds strategy_configs with current configs + backtest benchmarks |
| `supabase/backfill_paper_trades.py` | Loads 20 session JSONs → trades + blocked_signals tables |
| `supabase/backfill_backtest_trades.py` | Loads backtest CSVs → trades table + research_runs |
| `dashboard/src/lib/supabase.ts` | Client singleton from VITE_ env vars |
| `dashboard/src/hooks/useSupabase.ts` | React hook: fetches drift, rolling perf, daily stats (2-min poll) |
| `dashboard/src/components/AnalyticsPanel.tsx` | 3-tab panel: Drift / Equity / Daily |

### Database Schema (10 Tables)

1. **`strategy_configs`** — versioned config snapshots with backtest benchmarks (WR, PF, Sharpe, MaxDD)
2. **`research_runs`** — backtest sweep metadata (strategy, params, param_hash, aggregate metrics)
3. **`trades`** — unified trade log (paper + live + backtest), dedup on `(strategy_id, entry_time, is_partial, source)`
4. **`blocked_signals`** — counterfactual logging with gate_type extraction
5. **`research_results`** — individual parameter combos within a sweep
6. **`decisions`** — structured decision log (implement/reject/shelve/change/reverse)
7. **`gate_state_snapshots`** — daily gate state per instrument
8. **`claude_sessions`** — AI session continuity (summary, topics, open_questions, next_steps)
9. **`bars_daily`** — daily OHLCV summary per instrument
10. **`trade_annotations`** — post-trade analysis (investigation, pattern, lesson, correction)

### Analytical Views

- **`daily_stats`** (materialized) — per-strategy per-day: trade count, WR, PF, exit reason breakdown
- **`rolling_performance`** (materialized) — rolling 20-trade WR/PF, cumulative equity curve, backtest benchmark comparison
- **`gate_effectiveness`** (view) — per-gate-type per-week: signals blocked, counterfactual net PnL
- **`live_vs_backtest`** (view) — drift detection: WR/PF deviation, binomial Z-score, GREEN/YELLOW/RED status
- **`portfolio_daily`** (view) — aggregated daily portfolio stats
- **`drawdown_hwm`** (materialized) — running cumulative PnL, high water mark, drawdown
- **`tp1_runner_pairs`** (view) — links TP1 partial + runner legs
- **`tod_performance`** (view) — hourly ET entry buckets

### Functions

- **`f_morning_briefing()`** — JSON with gate state, yesterday's trades, rolling performance, drawdown, blocked signals
- **`f_has_been_tested(strategy_id, param_key, param_value)`** — checks if a parameter was already tested
- **`f_trade_context(trade_id)`** — full enriched context for a trade

## Current Data State (Mar 12, updated)

| Source | Trades | Strategies |
|--------|--------|------------|
| paper | 143 | V15 (59), vScalpB (25), MES_V2 (24+37), vScalpC (2), V11 (15) |
| backtest | 4,604 | V15 (471), vScalpB (385), vScalpC (465), MES_V2 (545), +research variants (2,738) |

- 33 research_runs (4 prod strategies × FULL/IS/OOS + MES research variants)
- 5 strategy_configs seeded (4 active + 1 shelved V11)
- 6 blocked signals from CSV
- vScalpC backtest gap (K2) resolved: added to `run_and_save_portfolio.py` with `run_backtest_partial_exit`

### Drift Detection Status (Mar 12)

| Strategy | Paper WR | Backtest WR | Z-Score | Status |
|----------|----------|-------------|---------|--------|
| MNQ_V15 | 78.0% | 83.0% | -1.02 | GREEN |
| MNQ_VSCALPB | 68.0% | 73.2% | -0.59 | GREEN |
| MES_V2 | 54.2% | 56.0% | -0.18 | GREEN |
| MNQ_VSCALPC | 0.0% (2 trades) | 77.0% | — | INSUFFICIENT_DATA |

## P&L Data Flow (Critical to Understand)

- **Session JSON `pnl`** = GROSS (pts × dollar_per_pt × qty, no commission)
- **Backtest CSV `pnl_dollar`** = NET (pts × dollar_per_pt - 2 × commission_per_side × qty)
- **DB `pnl_net`** = NET (both sources deduct commission before insert)
- **Commission rates**: MNQ $0.52/side, MES $1.25/side
- `backfill_paper_trades.py` deducts commission from gross session JSON pnl
- `backfill_backtest_trades.py` uses CSV pnl_dollar as-is (already net), reads `qty` from CSV metadata for commission/qty fields
- `db_logger.py` deducts commission from gross `trade.pnl_dollar`

## Dashboard Analytics Panel

Three tabs at the bottom of the dashboard:

1. **Drift** — per-strategy cards showing live WR/PF vs backtest, Z-score, GREEN/YELLOW/RED
2. **Equity** — cumulative P&L curves per strategy (Recharts)
3. **Daily** — last 7 trading days with W/L, WR, PF, exit breakdown

Env vars: `VITE_SUPABASE_URL` + `VITE_SUPABASE_PUBLISHABLE_KEY` in `dashboard/.env`

## Known Limitations

### Structural (Won't Fix — Fundamental Design)

1. **Backtest fills at next-bar open, live fills at limit/stop price.** Backtest TP trades can overshoot (e.g., vScalpB TP=3 but backtest records pts=28). Systematically inflates backtest PF. Drift detection will show live slightly underperforming backtest for TP-heavy strategies. This is a backtest engine design issue, not fixable in the data layer.

2. ~~**No vScalpC backtest CSV.**~~ **FIXED (Mar 12)**: `run_and_save_portfolio.py` now runs 4 strategies including vScalpC via `run_backtest_partial_exit`. 465 backtest trades loaded. `save_backtest` updated with `qty` param for correct 2-contract commission. Drift detection shows INSUFFICIENT_DATA (N=2 paper trades, needs 20+).

3. **Backtest = single-contract, live = multi-contract.** Backtest trades are all qty=1, is_partial=false. Live partial exit strategies produce paired TP1 (is_partial=true) + runner (is_partial=false) trades. The `live_vs_backtest` view handles this by filtering `NOT is_partial` for live trades. But this means the backtest baseline includes both TP1 and runner outcomes in a single trade, while live separates them.

4. **Backtest omits entry gates.** Backtest trades include signals that would be blocked by Leledc, ADR, ATR, and prior-day gates in live. Backtest has ~5-25% more trades than live would. Known pre-gate baselines.

5. **Materialized views refresh once/day.** Equity curve and daily stats show yesterday's data during trading hours. Real-time SafetyPanel (separate from AnalyticsPanel) handles actual risk management via EventBus events, not Supabase.

### Fixed by Review (Mar 12)

6. **C1: `_apply_correction` had no `source` filter** — could overwrite backtest trades with live corrections. Fixed: added `.eq("source", self._source)`.

7. **C2: Blocked signals used `insert()` not `upsert()`** — would fail on restart with duplicate key. Fixed: changed to `upsert(..., ignore_duplicates=True)`.

8. **C3: App.tsx hid Supabase errors** — `analytics.loaded &&` prevented error display. Fixed: `(analytics.loaded || analytics.error) &&`.

9. **H1-H2: Unbounded/unfiltered Supabase queries** — rolling_performance had no limit, live_vs_backtest fetched all sources. Fixed: added `.limit(2000)` and `.in('source', ['paper', 'live'])`.

10. **H3: `date.today()` used system TZ** — backfill today-exclusion would be wrong on UTC server. Fixed: `datetime.now(_ET).date()`.

11. **H4: Split casing inconsistency** — hash computed with "FULL" but stored as "full". Fixed: normalize to lowercase before hashing.

12. **H5: MFE/MAE=0.0 lost** — `if mfe:` is falsy for 0.0. Fixed: `if mfe is not None:`.

13. **H6: Shutdown drain broke on first error** — remaining queue items abandoned. Fixed: `continue` with 5-failure cap.

## Operational Notes

- **Graceful degradation**: Engine runs normally without Supabase. db_logger logs warning and continues.
- **Zero latency impact**: Async queue pattern — trade processing never waits for Supabase.
- **Backup**: Session JSONs + blocked_signals.csv preserved as local backup. All data recoverable via backfill scripts.
- **Idempotent**: All scripts use ON CONFLICT DO NOTHING. Safe to re-run.
- **Today exclusion**: Backfill excludes today's trades to avoid overwriting enriched live-logged data.
- **Publishable key**: Safe for client-side use (RLS limits to read-only). Anon SELECT policies on all tables + views.

## Deferred Items

- **config_id FK population** — link each trade to the strategy config version that produced it
- **Cross-strategy correlation view** — daily P&L correlation between strategy pairs
- **Counterfactual batch replay** — simulate blocked signals against bar data to fill cf_* columns
- **Intraday matview refresh** — refresh every N trades or every 30 min during RTH
- **Restore script** — single script that runs all 3 backfill scripts in order
