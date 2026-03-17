# Decision Log

Chronological record of all decisions (implement, reject, change, shelve).
Updated after each session. Rejections are as important as implementations.

---

## Feb 14, 2026
- **FIXED** DST bug, look-ahead bias in exits, RSI mapping look-ahead. (`engine_bugs.md`)
- **REJECTED** all 11 v10 MNQ feature variants (underwater exit, OR alignment, VWAP, etc). v9 baseline was negative on 6-month 1-min data. (`strategy_versions.md`)
- **ADOPTED** separate SM params per instrument: MNQ SM(10/12/200/100), MES SM(20/12/400/255). MES sweep of 6,090 combos all failed; v9.4 genuinely optimal. (`cross_instrument_analysis.md`)

## Feb 15
- **ADOPTED** v11 MNQ with native SM in Pine. Walk-forward OOS PF 1.713 > IS 1.634. (`strategy_versions.md`)

## Feb 17
- **REJECTED** CVD/Volume Delta as entry filter or SM replacement. SM and CVD uncorrelated (r=-0.029). (`cvd_research.md`)
- **REJECTED** all v14 tick microstructure exits (trade imbalance, large trade ratio, tick intensity, delta accel, iceberg). All failed OOS. (`v14_v15_exit_research.md`)
- **REJECTED** trailing stops as standalone MNQ exit (PF 0.933). (`v14_v15_exit_research.md`)
- **REJECTED** all trailing stop and TP configs for MES. All hurt MES. (`v14_v15_exit_research.md`)
- **ADOPTED** TP=5 as MNQ v15 exit. 97-100% of trades go profitable; TV validated 86.67% WR. (`v14_v15_exit_research.md`)
- **REJECTED** regime detection for MNQ (London range, prior-day range, rolling vol, first-30-min, overnight gap). All flip direction between train/test. (`v14_v15_exit_research.md`)

## Feb 18
- **BUILT** Renko bar data from 42.6M NQ ticks (5pt box). Shelved for future use. (`renko_data.md`)
- **NOTED** SM threshold 0.15 beneficial for v11 but deferred. Removes profitable trades too. (`sm_strength_research.md`)
- **REJECTED** SM threshold for MES v9.4 and volume delta on top of SM for MNQ. (`sm_strength_research.md`)

## Feb 19
- **BUILT** React + Vite dashboard with Lightweight Charts v4. (`dashboard.md`)
- **FIXED** episode reset flickering, daily reset UTC bug, trade_closed event ordering, 12 SafetyManager review bugs. (`engine_bugs.md`)
- **SHELVED** v11.1 MNQ (SM flip exit: IS +$3,529, OOS -$509). (`strategy_versions.md`)

## Feb 20
- **REJECTED** RSI-based exits for MES (17 configs), conditional losing hold cap (post-filter != pre-filter). (`mes_exit_research.md`)
- **ADOPTED** EOD 15:30 close for MES v9.4. 4/7 LOMO wins, positive avg dPF. (`mes_exit_research.md`)
- **FIXED** bracket cancel race, emergency flatten robustness, quote feed staleness detection. (`engine_bugs.md`)

## Feb 21
- **CONFIRMED** SM entry signal works IS+OOS; SM flip exit is the problem. All 3 strategies fail OOS with SM flip. (`oos_analysis_feb21.md`)
- **ADOPTED** vScalpB (SM_T=0.25, RSI 8/55-45, TP=5, SL=15, CD=20). Only cluster profitable on both IS and OOS. (`strategy_versions.md`)
- **ADOPTED** MES v2 with TP=20 exit (SM_T=0.0, RSI=12/55-45, CD=25, SL=35, EOD=15:30). (`strategy_versions.md`)
- **SHELVED** MES v9.4 (replaced by v2). (`strategy_versions.md`)

## Feb 22
- **BUILT** OCO bracket system for live trading (exchange-side, crash-safe). (`engine_bugs.md`)
- **FIXED** paper mode SL not enforced, added double failure protection (engine halt if both OCO and fallback fail). (`engine_bugs.md`)
- **BUILT** news day impact analysis (107-event calendar). vScalpA loses on FOMC (-$454). (`news_day_analysis.md`)
- **REJECTED** blanket news blackout (portfolio would lose $929; MES profits from news). (`news_day_analysis.md`)

## Feb 23
- **ADOPTED** TPX L=30 S=12 as vScalpB regime gate candidate (pending paper trade). (`tpx_research.md`)
- **REJECTED** TPX for vScalpA and MES v2. (`tpx_research.md`)
- **IMPLEMENTED** MES v2 partial exit (entry_qty=2, TP1=10, runner to TP2=20). PF +11%. (`mes_partial_exit_research.md`)
- **BUILT** dashboard sizing controls (entry_qty + partial_qty overrides per strategy). (`dashboard.md`)

## Feb 24
- **BUILT** NQ Trading.app launcher with smart process reuse. (`dashboard.md`)
- **IMPLEMENTED** daily session rotation (auto-save + clear at ET day boundary). (`dashboard.md`)

## Feb 26
- **CHANGED** V15 SL from 50 to 40 (best by Sharpe, never re-optimized for TP=5). (`strategy_versions.md`)
- **SET** target portfolio B(2)+MES(1), dropping vScalpA. (SUPERSEDED Mar 4). (`portfolio_weighting_analysis.md`)
- **IMPLEMENTED** dual OCO brackets for MES v2 partial exit in TastytradeBroker. (`mes_partial_exit_research.md`)

## Feb 27
- **VALIDATED** tastytrade accepts two simultaneous OCO brackets (Step 0 test with M6E). (`mes_partial_exit_research.md`)
- **IMPLEMENTED** BE_TIME=275 for MES v2 (REVISED to N=75 on Mar 4). (`mes_partial_exit_research.md`)
- **FIXED** phantom trade bug (strategy state desync on blocked entry). (`trading_code_review.md`)

## Mar 2
- **FIXED** CME maintenance window heartbeat, recon key namespace, auto-recovery after timeout, OCO-aware emergency flatten. (`heartbeat_recon_fixes.md`)
- **DEFERRED** recovery warm-up gate (~0.4% false entry risk, fix before scaling). (`heartbeat_recon_fixes.md`)

## Mar 3
- **IMPLEMENTED** VIX death zone gate 19-22 (blocks V15+vScalpB). (`vix_death_zone_gate.md`)
- **IMPLEMENTED** V15 13:00 ET entry cutoff (PF 1.07->1.326, Sharpe 0.39->1.648). (`late_day_entry_analysis.md`)
- **REJECTED** late-day cutoff for vScalpB (afternoon entries are vScalpB's BEST, PF 2.075-2.260). (`late_day_entry_analysis.md`)
- **REJECTED** late-day cutoff for MES (break-even, not worth gating). (`late_day_entry_analysis.md`)

## Mar 4
- **CHANGED** MES v2 BE_TIME from 275 to 75 (best OOS Sharpe/PF/MaxDD; original sweep too coarse). (`mes_partial_exit_research.md`)
- **REVERSED** "drop vScalpA" — new target A(1)+B(1)+MES(1), Sharpe 3.05. De-risked V15 now strong. (`portfolio_weighting_analysis.md`)

## Mar 5
- **FIXED** VIX gate silently broken (yfinance return type change, caught by fail-open). (`vix_death_zone_gate.md`)
- **REMOVED** VIX gate from vScalpB (OOS borderline on only 22 trades). Kept on V15 only. (`vix_death_zone_gate.md`)
- **REWROTE** VIX fetcher to use tastytrade DXLink Summary primary, yfinance fallback. (`vix_death_zone_gate.md`)
- **FIXED** launcher quit handling (PID files + watchdog). (`MEMORY.md`)
- **REJECTED** SM magnitude, volume climax, all exit-side filters for MES v2. (`mes_v2_entry_filter_research.md`)
- **REJECTED** combined filter stacking for MES (AND-ing compounds trade count problem). (`mes_v2_entry_filter_research.md`)

## Mar 6
- **IMPLEMENTED** Leledc exhaustion gate mq9_p1 for all MNQ strategies. Portfolio OOS Sharpe +32%. (`round3_sr_filter_research.md`)
- **IMPLEMENTED** prior-day level gate buf5 for MES v2 (H/L/VPOC/VAH/VAL within 5 pts). IS PF +10.4%, OOS PF +9.0%. (`round3_sr_filter_research.md`)
- **REJECTED** VWAP Z-Score, Squeeze (TTM), Intraday Pivots, combined IB+Leledc. (`round3_sr_filter_research.md`)
- **ADOPTED** separate filter per instrument (Leledc for MNQ, Prior Day for MES). (`round3_sr_filter_research.md`)
- **RAISED** MES daily loss limit $200->$400 (one 2-contract SL is $355). (`MEMORY.md`)
- **RAISED** global daily loss $500->$600. (`MEMORY.md`)
- **REJECTED** MES 14:00 ET cutoff (IS overfit) and intraday VIX monitoring (VIX reactive, not predictive). (`MEMORY.md`)

## Mar 8
- **ADDED** Pivot Point Supertrend overlay to dashboard. (`dashboard.md`)
- **FIXED** auto-save to only save if trades from today (ET). (`dashboard.md`)

## Mar 9
- **REJECTED** ATR/ADX entry filters (3 tests), ATR-scaled exits (just bigger fixed exits), DI alignment (redundant with SM). (`atr_adx_research.md`)
- **FOUND** V15 TP=7 and vScalpB TP=3/SL=10 as upgrade candidates from fixed exit sweep. (`atr_adx_research.md`)
- **IMPLEMENTED** vScalpC runner (TP1=7/TP2=25/SL=40/BE45/SL-to-BE). 463 trades, PF 1.452, Sharpe 2.25. (`vscalpc_runner_design.md`)
- **IMPLEMENTED** SL-to-breakeven after TP1 for vScalpC (WR 67%->77.3%, MaxDD -$271). (`vscalpc_runner_design.md`)
- **IMPLEMENTED** prior-day ATR gate >=263.8 for vScalpC (STRONG PASS IS+OOS). (`vscalpc_runner_design.md`)
- **ADDED** VIX death zone gate to vScalpC (same entries as V15). (`trading_code_review.md`)
- **RAISED** vScalpC daily loss limit $100->$200 (2-contract SL = $160). (`trading_code_review.md`)

## Mar 10
- **APPLIED** V15 exit upgrade TP=5->7 (Sharpe 2.73 vs 2.08, PF 1.36 vs 1.29). Config updated for paper trading.
- **APPLIED** vScalpB exit upgrade TP=5/SL=15 -> TP=3/SL=10 (Sharpe 3.29 vs 1.49, half drawdown). Config updated for paper trading.
- **IMPLEMENTED** ADR directional gate for all MNQ (lookback=14, threshold=0.3). STRONG PASS IS/OOS. (`adr_exhaustion_research.md`)
- **REJECTED** ADR gate for MES v2 (universal failure — slow SM + TP=20 trades larger moves). (`adr_exhaustion_research.md`)
- **IMPLEMENTED** gate seed + auto-persist system (bootstrap from databento, persist daily at reset).
- **CREATED** trade post-mortem template (8 required sections). (`trade_postmortem_template.md`)
- **CREATED** decision log (this file), reconstructed from 30 topic files.
- **RULE** "Paper trade it = apply config changes immediately."

## Mar 11
- **RAN** comprehensive MES v2 gate sweep (10 gate types, 45 configs). Motivated by 3 MES full-SL days in 5 trading days (-$1,067).
- **FOUND** VIX [20-25] STRONG PASS for MES (IS PF +7.6%, OOS PF +8.4%). Different band than MNQ's [19-22].
- **FOUND** Entry delay +30-45min MARGINAL PASS for MES (OOS PF +8-12%). 67% of MES SLs happen 10:00-11:00 ET.
- **FOUND** Leledc mq=7 MARGINAL PASS for MES (OOS PF +8.9%). Different threshold than MNQ (mq=9).
- **CONFIRMED** SM threshold ALL FAIL for MES (all 5 values kill OOS). "Weak entries profitable" validated again.
- **CONFIRMED** Shorter TP ALL FAIL for MES (TP=3 through TP=15 all worse). MES needs TP=20+.
- **CONFIRMED** Tighter SL ALL FAIL for MES (SL=15 through SL=30 all hurt OOS). SL=35 optimal.
- **CONFIRMED** Entry cutoff ALL FAIL for MES (IS overfit, same as Mar 6 finding).
- **CONFIRMED** Overnight range gate ALL FAIL for MES (no sweet spot).
- **CONFIRMED** ADR gate FAIL for MES at thresholds 0.1-0.3. ADR=0.4 marginal only.
- **RED TEAMED** VIX [20-25] gate for MES → FAIL. Bootstrap p=0.29 (not significant). Alternative bands show [11-16] has higher dPF than [20-25] — pattern is noise. (`mes_v2_tp1_sweep.md`)
- **RED TEAMED** Entry delay +30min for MES → FAIL. Bootstrap p=0.33 (not significant). IS/OOS divergence when stacked with VIX (IS -7.4%, classic overfit). (`mes_v2_tp1_sweep.md`)
- **ADOPTED** MES v2 TP1=6 (was TP1=10). 2-contract sweep: PF 1.239→1.341 (+8.2%), Sharpe 1.27→1.63 (+28%), TP1 fill rate 39%→60%, MaxDD -$1,420→-$1,175. IS/OOS consistent (OOS PF > IS PF). Config updated. (`mes_v2_tp1_sweep.md`)
- **REJECTED** breakeven escape (close wandering trades near breakeven). All configs show IS/OOS divergence (hurts IS, helps OOS). Same red flag as VIX gate. TP1 reduction already captures the benefit. (`mes_v2_tp1_sweep.md`)

## Mar 11 (evening)
- **IMPLEMENTED** Supabase data foundation (Phase 1 of Trading Copilot). Schema: 10 tables (trades, blocked_signals, research_runs, research_results, decisions, strategy_configs, gate_state_snapshots, claude_sessions, bars_daily, trade_annotations), 4 analytical views (daily_stats, rolling_performance, gate_effectiveness, live_vs_backtest), 3 functions (f_morning_briefing, f_has_been_tested, f_trade_context). Engine integration: db_logger.py (async EventBus subscriber), db.py (lazy client singleton), trade context enrichment (SM/RSI/volume at entry, SM/RSI at exit, MFE/MAE, gate state injection). Graceful degradation — engine runs normally without Supabase. Zero latency impact (async queue pattern).
- **IMPLEMENTED** MAE tracking in strategy.py. Tracks worst adverse excursion alongside existing MFE. Both populated on every TradeRecord (partial and full closes).
- **IMPLEMENTED** TradeRecord context fields (events.py). 17 new optional fields: entry context (sm, velocity, rsi, volume, minutes_from_open), exit context (sm, rsi), gate state (vix, leledc, atr, adr), MFE/MAE, source. All backward-compatible with None defaults.

## Mar 12
- **FIXED** Stale backtest constants in `generate_session.py`. V15 TP=5→7, vScalpB TP=5→3/SL=15→10. All existing backtest CSVs were generated under wrong config. Re-ran `run_and_save_portfolio.py --split` to produce correct-config baselines.
- **FIXED** db_logger.py writing gross P&L as `pnl_net`. Added commission deduction: `pnl_net = pnl_dollar - commission_per_side * 2 * qty`. Session JSON `pnl` confirmed gross via arithmetic (7.5pts × $2/pt = $15.00 exactly).
- **IMPLEMENTED** Supabase Session 1 — seed + backfill paper data. Strategy configs seeded (5 strategies with backtest benchmarks). 143 paper trades backfilled from 20 session JSONs. 6 blocked signals backfilled. Anon RLS policies + view grants deployed for dashboard access.
- **IMPLEMENTED** Supabase Session 2 — backfill backtest trades. 1,401 backtest trades loaded from 9 correct-config CSVs (V15/vScalpB/MES_V2 × FULL/IS/OOS). 9 research_runs created with aggregate metrics. Drift detection operational: all active strategies GREEN (V15 z=-1.02, vScalpB z=-0.59, MES_V2 z=-0.18).
- **IMPLEMENTED** Supabase Session 3 — dashboard AnalyticsPanel wired to Supabase. 3 tabs: Drift (per-strategy cards, WR/PF vs backtest, Z-score, GREEN/YELLOW/RED), Equity (cumulative P&L curves per strategy), Daily (last 7 days with W/L, WR, PF). Installed @supabase/supabase-js, created client + hook + component. (`supabase_data_foundation.md`)
- **FIXED** 5-agent review findings (9 issues). CRITICAL: `_apply_correction` cross-source contamination (added source filter), blocked signal `insert` → `upsert`. HIGH: unbounded rolling_performance query (+limit), unfiltered live_vs_backtest query (+source filter), App.tsx error state hidden, `date.today()` → ET-aware, split casing inconsistency, MFE/MAE=0 lost (falsy check), shutdown drain break → continue. (`supabase_data_foundation.md`)
- **NOTED** Known backtest limitations for drift detection: next-bar-open fills inflate backtest PF, single-contract vs multi-contract structural mismatch, backtest omits entry gates. All documented, not fixable in data layer. (`supabase_data_foundation.md`)
- **FIXED** vScalpC backtest gap (K2). Added vScalpC to `run_and_save_portfolio.py` using `run_backtest_partial_exit` from sweep script. Updated `save_results.py` with `qty` param for correct 2-contract commission. Updated `backfill_backtest_trades.py` to read `qty` from CSV metadata. 465 vScalpC backtest trades + 3 research_runs loaded into Supabase. vScalpC benchmarks updated: WR 77.0%, PF 1.425, Sharpe 2.139, MaxDD -$1,079. Drift detection shows INSUFFICIENT_DATA (2 paper trades, needs 20+).
- **IMPLEMENTED** Digest Agent (agents/digest/). 4 files: prompts.py (EOD + Morning system prompts), tools.py (12 Supabase tools + dispatch), agent.py (Anthropic tool_use loop with metadata tracking, dry-run, cost estimation), cli.py (CLI entry point). Digests table migration created (004). Pending: apply migration, add ANTHROPIC_API_KEY, test with --dry-run.
- **ADOPTED** Strategist Agent layer in pipeline architecture. Strategist is the ONLY agent authorized to recommend parameter changes — runs on deepest reasoning model (Opus), reads ALL agent outputs, weighs competing evidence with full portfolio context. Individual agents (Digest, Investigation, Frontier) observe/investigate/test but do not recommend. Morning Digest surfaces Strategist recommendations. Rationale: build for the vision (multi-instrument, many strategies) not current state. (`evening_morning_pipeline.md`)
- **BUG FOUND** ADR directional gate was not operational on March 11 despite config being correct. Root cause: `gate_seed.json` was created at 16:12 ET on March 10, but the engine started at 08:50 ET — before the file existed. Engine was never restarted. Without seed data, 14-day lookback was never satisfied (only 1 day accumulated). Gate was in fail-open (ratio=0.0) all day. Estimated cost: $205.86 net (would have blocked 6 shorts on a -1.68% selloff day). Fix: restart engine. Full postmortem in `logs/trade_deconstructions.md`.
- **DESIGNED** Strategist Agent — full technical architecture. Three-pass reasoning (evidence gathering, evaluation+challenge, decision) with forced checkpoints. 19 tools (11 inherited from Digest + 8 new: decision_history, parameter_history, check_has_been_tested, get_active_recommendations, preflight_check, save_recommendation, begin_phase, get_frontier/investigation outputs). 5 action tiers (implement, paper_trade, investigate, monitor, no_action). Preflight guardrails: rate limit (max 3 action recs/day, max 1 implement/day), contradiction detection (rejected params, conflicting active recs, decision log checks), OOS trade minimums (15 hard, 30 preferred). Devil's advocate mandatory for every recommendation. Supabase table `strategist_recommendations` with full lifecycle tracking. Runs on Opus, ~$0.50-1.50/run. Advisory only — recommends but never executes.
- **CONSOLIDATED** Strategist Agent design — merged PM design doc + architect doc into single definitive specification. Promoted 5 features from "3 months" to core v1: temporal reasoning tools, recommendation chains, ghost portfolio, counterfactual replay against live trades, confidence calibration tracking. Expanded from 19 to 24+ tools, 5 to 7 action tiers (added SHELVE, ALERT, REDUCE). Added 4 new Supabase tables/views (recommendation_chains, ghost_portfolio_entries, ghost_portfolio_results, strategist_calibration). (`strategist_agent_design.md`)
- **FIXED** MAX_TURNS 15→25. Sonnet calls tools one-at-a-time (no parallel tool_use), so 15 turns was too tight for 14 data-gathering steps + synthesis + save_digest. Dry run hit max_turns at turn 15. Bumped to 25, next run completed in 17 turns.
- **DEPLOYED** GitHub Actions cron for Digest Agent. Workflow: `.github/workflows/digest.yml`. EOD at 21:00 UTC (17:00 ET EDT / 16:00 ET EST — both after 15:30 EOD exit), morning at 11:00 UTC (07:00 ET EDT / 06:00 ET EST). Mon-Fri only. Manual dispatch with mode/date/dry-run inputs. Secrets: ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY in GitHub repo settings. First successful dry run: 17 turns, 16 tool calls, 80s, $0.77. First GitHub Actions run: green check.
- **IMPLEMENTED** Intel panel in dashboard (`dashboard/src/components/IntelPanel.tsx`). Accordion feed of digests from Supabase `digests` table. Auto-expands most recent. Inline markdown rendering (bold, italic, code, headers, lists, P&L auto-colored). EOD/AM badges. 5-min polling. Last 20 digests. Sits below Analytics panel.
- **INSIGHT** Karpathy's autoresearch pattern maps directly to the Frontier Agent. Autonomous loop: read hypothesis → modify backtest config → run experiment (~30-60s) → evaluate IS/OOS → accept/reject → log → repeat. ~50-100 tests overnight. Key difference from Digest: must run locally (needs Databento data + backtesting engine), not on GitHub Actions cloud. Self-hosted GH Actions runner recommended. Investigation Agent dropped from pipeline — Digest enhanced tools feed hypotheses directly to Frontier. (`frontier_autoresearch_pattern.md`)
- **DEBATED** Investigation Agent — 3-agent design review (architect, trading domain expert, critic). Critic argued persuasively against a separate agent: Digest already has 70% of the data, remaining 30% can be 3-5 new Digest tools, bottleneck is testing (Frontier) not hypothesis generation. Recommended path: enhance Digest with level proximity + near-gate-miss tools, build Frontier next, revisit Investigation when system scales. Bar data decision: do NOT store 1-min bars in Supabase; pre-compute context in engine. Full debate: `memory/investigation_vs_enhanced_digest.md`. Architect's full spec preserved: `memory/investigation_agent_design.md`.
- **IMPLEMENTED** Digest Agent team review — all 23 fixes from 5-agent review. 7 CRITICAL: C1 save_digest error-as-data, C2 _digest_saved flag, C3 API error handling (4 exception types), C4 token budget 300k + max turns 15, C5 query limits (.limit(100) + explicit column select), C6 .env loader (quote stripping, export handling), C7 sync not async. 9 IMPORTANT: I1 max_tokens stop handling, I2 tool result truncation (8000 chars), I3 @functools.wraps, I4 cost_usd in all returns, I5 explicit traffic light thresholds (RED/YELLOW/GREEN with Z-score, drawdown, streak criteria), I6 save_digest content schema docs, I7 7-day lookback for recent digests, I8 date.fromisoformat, I9 proper rank calculation. 7 NEW TOOLS: M1 get_tod_performance, M2 commission enrichment in daily stats, M3 get_market_regime, M4 get_gate_effectiveness, M5 get_streak_status, M6 get_runner_stats, M7 get_dow_performance. Prompts updated with new tool references in analysis process + pattern detection checklist. All 4 files verified (18 tools, imports clean, CLI operational).

## Mar 13, 2026
- **IMPLEMENTED** 4 forensic analysis tools for Digest Agent: `get_sl_velocity` (per-strategy SL speed classification), `get_entry_clustering` (same-bar correlated entry detection), `get_near_gate_miss` (near-threshold counterfactual P&L), `get_level_proximity` (entry distance to levels/VWAP/OR). Enhanced `get_market_regime` with VWAP close, opening range, multi-day `days` param. 22 tools total. Full 6-phase agent workflow.
- **IMPLEMENTED** Engine additions: VWAP accumulation (RTH-only), opening range tracking (10:00-10:30 ET), `gate_leledc_count` per-trade injection (runner→strategy→TradeRecord→Supabase). Migration `005_forensic_tools.sql` (4 columns).
- **ADOPTED** `max_daily_loss` $600→$650. Worst-case A(1)+B(1)+C(2)+MES(2) simultaneous SL = $619.16 with commissions.
- **ADOPTED** `flags_for_frontier` structured output in save_digest. Schema: hypothesis, evidence, suggested_test, priority(1-5), sample_size, recurrence, strategy_id. Quality-gated: sample_size≥5 only.
- **ADOPTED** Quality-gated forensic insights: no filler, no cap on count, skip on quiet days (<3 trades).
- **REVIEW FIXES** (Phase 5, 9 issues): near-gate-miss P&L dedup, MES prior-day levels removed from near-miss description, BE exit pnl check, prompt SL threshold cleanup, market_regime index correlation bug, ADR near-miss direction-awareness, level_proximity adr→None fallback, level_proximity now includes VWAP+OR distances.
- **NOTED** `gate_state_snapshots` table still empty — `get_level_proximity` and `get_market_regime` VWAP/OR fields will return empty until engine runs post-migration. Pre-existing known gap.
- **IMPLEMENTED** Counterfactual Trade Engine (`agents/counterfactual/`). 3 files: engine.py (fetch→interleave→simulate→write), exit_simulator.py (single-leg + partial exit), cli.py (CLI entry point). Cooldown-aware timeline walk, cross-day continuous cooldown, signal_group_id for V15↔vScalpC correlation. Session JSON primary data source (no Databento cost), Databento CSV fallback. Migration `006_cf_signal_price.sql` (signal_price + signal_group_id columns).
- **FIXED** `db_logger.py` gate_type "unknown" for prior_day_level blocks. Reason string starts with "Near prior-day level" but prefix map only had "Prior-day level". Added "Near prior-day level" prefix.
- **CHANGED** MES_V2 session timing. (1) Removed forced close: `session_close_et` 15:30→16:00. The 15:30 close was validated on v9.4 (SM flip exit) but never triggered in v2 backtests (0 EOD exits out of 552 trades) — TP/SL/BE_TIME resolve all trades before 15:30. In live trading Mar 13, it killed a winner at -$35. Dead weight that actively hurts. (2) Added entry cutoff: `session_end_et="14:15"`. Time bucket analysis showed entries 14:30+ are net losers (PF 0.78-0.87, collectively -$295). 14:15 cutoff: IS PF +17.6%, OOS PF -2.7% (within noise). Script: `mes_v2_session_timing_sweep.py`.
- **CONFIRMED** Cooldown values (MNQ=20, MES=25) are optimal. Full IS/OOS sweep of CD=[0,5,8,10,12,15,18,20,25,30]. Lower CDs inflate IS but degrade OOS. No change needed. Script: `cooldown_sweep.py`.
- **INVESTIGATED** Prior-day level gate per-level breakdown for MES_V2 (buf=5). Script: `sr_prior_day_level_breakdown.py`. VPOC+VAL are the only levels removing losing trades. H/L and VAH remove **profitable** breakout trades ($311, $530, $883 respectively). VPOC+VA combo gets best OOS Sharpe (1.567 vs 1.548 for all-5). Recommendation: test VPOC+VAL only gate as a leaner alternative. Full analysis in `round3_sr_filter_research.md`.

## Mar 13 (continued)
- **REJECTED** Structure exit for MES_V2. Full-period: P&L -$1,460, Sharpe +5%, MaxDD -19%. OOS numbers (Sharpe +60%, MaxDD -47%) are misleading — full period tells the real story.
- **IMPLEMENTED** Structure exit for vScalpC runner. Pivot LB=50, PR=2, Buffer=2pts, cap=60pts. Full-period: P&L +$406, PF +2.3%, Sharpe +3%, MaxDD -12%, WR +1.3pp. IS/OOS PF ratio 97.3%. All metrics improve. Cap=60 backtest validated (-$36 vs no-cap). Parity test: zero mismatches across 375,211 bars.
- **CHANGED** vScalpC tp_pts: 25→60. Now serves as crash-safety cap on exchange. Structure monitor exits before this in normal operation.
- **CHANGED** vScalpC BE_TIME: skipped when structure_exit_type is set. Structure monitor + 60pt cap replace the stale-runner timeout.
- **REJECTED** Phantom OCO Ladder. tastytrade has no partial OCO modification API. Cancel-replace each bar creates 100ms unprotected gap. Active monitoring + 60pt cap is simpler.
- **FIXED** _prev_bar look-ahead in runner.py. `strategy._prev_bar` updates inside `on_bar()`. Must capture BEFORE calling `on_bar()` to pass bar[i-1] to structure monitor. Caught by mandatory review.
- **IMPLEMENTED** Dashboard structure overlay (Stage 2): STRUCT toggle on PriceChart (swing H/L lines), STRUCT badge in SafetyPanel (target level + distance, clickable toggle), STR badge in TradeLog. Runtime disable via WS `structure_exit_toggle` command.
- **IMPLEMENTED** Observation data logging (Stage 3): Per-bar structure level logging to Supabase `structure_bar_logs` table. Near-miss detection (within 3pts of trigger). Fire-and-forget queue. Migration `007_structure_bar_logs.sql`.
- **REVIEW FIXES** (Phase 5, 3-agent team): (1) Moved `tracker.update()` above `_enabled` check — prevents 53-bar stale swing levels on re-enable. (2) Added daily reset comment documenting intentional non-reset of structure monitors (matches backtest). (3) Removed unused `close_sig` variable. (4) ACCEPTED: BE_TIME disabled by static config (not runtime state) — 60pt cap + EOD + SL@BE provide 3 safety nets. (5) NOTED: No unit tests for check_exit() — paper trading is the functional test.

## Mar 14, 2026
- **DEPLOYED** All Supabase migrations (004-007). Structure exits, forensic tools, CF signal_price, structure bar logs.
- **IMPLEMENTED** CF engine launchd plist (`~/Library/LaunchAgents/com.nqtrading.counterfactual.plist`). 18:00 ET Mon-Fri, TimeOut=600s. First run Monday Mar 16.
- **FIXED** CF engine BE_TIME: (1) vScalpC now passes `be_time_bars=0` when `structure_exit_type` is set (matches live engine which skips BE_TIME for structure exit strategies). (2) MES v2 BE_TIME now closes ALL remaining legs after 75 bars regardless of TP1 status (previously required `not leg1_active`).
- **FIXED** CF engine COOLDOWN_SUPPRESSED idempotency: writes `cf_exit_price=NULL` (was 0.0), fetch filter changed to `cf_exit_reason IS NULL`, `_clear_results` updated to match.
- **IMPLEMENTED** `cf_pnl_dollar` column on `blocked_signals`. Dollar conversion at write time: `cf_pnl_pts * dollar_per_pt` (no entry_qty — pnl_pts already sums both legs). Migration `008_cf_dollar_and_view.sql` recreates `gate_effectiveness` view with dollar column + COOLDOWN_SUPPRESSED exclusion.
- **FIXED** CF engine signal_group_id now set for ALL signals (including COOLDOWN_SUPPRESSED), not just simulated ones.
- **CONFIRMED** CF commission calculation is already correct. `_COMMISSION_PTS` stores roundtrip per contract (0.52 for MNQ = $1.04/$2.00/pt). Formula `roundtrip * entry_qty` gives correct totals. 8-agent team review verified independently. No change needed.
- **DEFERRED** CF structure exit simulation for vScalpC (C1). Reason: vScalpC started paper trading Mar 13, structure params may change, only ~4-8 blocked signals before stabilization. Uses 60pt crash cap as TP2 until then. Revisit after 50+ blocked signals or params confirmed. 3 implementation options documented: `plans/cf-structure-exit-deferred.md`. Can reprocess with `--force`.
- **IMPLEMENTED** Gated portfolio backtest runner. Added all 5 entry gates (Leledc, ADR, ATR, prior-day levels, VIX death zone) + vScalpC structure exit + MES v2 parameter corrections (EOD 16:00, entry cutoff 14:15) to `run_and_save_portfolio.py`. Monthly breakdown output. Full-period gated portfolio: $12,939, Sharpe 4.29, MaxDD -$1,420, 12/14 months positive.
- **REMOVED** Leledc + ADR gates from vScalpB. Gated vs ungated analysis: gates cost $483/yr (31% of vScalpB P&L) for only +0.6pp WR. Blocked 109 winners vs 43 losers. vScalpB's SM_T=0.25 already filters for high-conviction entries — additional gates are redundant and harmful. Original research flagged vScalpB as "filter-resistant," now confirmed with full portfolio backtest.
- **COMPLETED** ICT Trade Forensics (v3) — mapped 1,123 trades against 7 ICT features (OBs, FVGs, fib/OTE, liquidity sweeps, BOS/MSS, weekly PVP, session H/L) on 5-min and 1-min timeframes. Three bugs found in audit and fixed (BOS bars_ago variable leak, undirected sweep, MNQ bin_width). Key findings: (1) London H/L is best gate candidate (WR -12.2pp, PF 0.924, N=137). (2) Weekly VAL dangerous (WR -23.5pp, PF 0.622, N=22). (3) Weekly VPOC is a sweet spot (WR 90.5%, PF 3.891, N=42). (4) OBs show strongest effect (-37pp) but N=11. (5) FVGs/sweeps/confluence not useful. (6) System is momentum-based — premium entries outperform. Script: `backtesting_engine/strategies/ict_forensics.py`. Results: `backtesting_engine/results/ict_forensics_v3_output.txt`. Full analysis: `memory/ict_forensics_results.md`.
- **CREATED** Research queue (`memory/research_queue.md`) — rolling prioritized list of findings to test, monitor, or defer. London H/L gate is #1 ready-to-test item.
- **CREATED** Indicator library — 16 indicators analyzed with structured templates. ICT Concepts and PVP tested via forensics. Index: `memory/indicators/INDEX.md`.
- **COMPLETED** London H/L directional decomposition. Both directions near both levels underperform: short near London LOW worst (WR -28.7pp, PF 0.630), long near London LOW also bad (WR -20.6pp, PF 0.812). Undirectional gate is appropriate. Script output: `backtesting_engine/results/london_hl_gate_sweep_output.txt`.
- **TESTED** London H/L gate sweep (buf=[3,5,7,10]). vScalpC buf=3: STRONG PASS (IS PF +7.7%, OOS PF +1.2%). vScalpA: inconsistent IS/OOS. Gate blocks <5% of entries (~1-2 trades/month). Decision: implement as observation (dashboard display + trade tagging) first, monitor before promoting to hard gate.
- **DESIGNED** ICT Dashboard Levels (9-agent Phase 2). London H/L DROPPED after gate sweep showed <1% block rate and no directional edge. Final scope: Weekly VPOC (green) + Weekly VAL (red) + Order Block zones (red/green).
- **IMPLEMENTED** ICT Dashboard Levels (Phase 4). 419 lines across 11 files. Engine: weekly VPOC/VAL (Monday computation from prior week RTH, gate_state persistence, bin_width per instrument), Order Block tracking (3-bar UAlgo engulfing, max 2/direction, mitigate on close-through), get_ict_proximity (observation only). Dashboard: ICT toggle, green wPOC line (conviction opacity + proximity style toggle), red wVAL line, OB zones as 3 price lines (top/bottom/mid). Trade tagging at entry time. Migration 009. Plan: `plans/ict-dashboard-levels.md`.

## Mar 15, 2026
- **IMPLEMENTED** Developing Daily VPOC engine accumulator. SafetyManager: daily volume profile bins (RTH-only, per-instrument), `_compute_value_area()[0]` for VPOC, VCR (volume concentration ratio = max_bin_vol/total_vol), stability index (bars since last dPOC shift). Resets daily. ~46 lines in safety_manager.py + 1 line server.py session save.
- **IMPLEMENTED** Dashboard dPOC line. Cyan `#00cccc` dotted line under ICT toggle, label "dPOC". Types: `dvpoc_strength`, `dvpoc_stability` added to `ICTLevelData`. Session replay works automatically.
- **IMPLEMENTED** VCR badge on SafetyPanel. Green (<0.12), yellow (0.12-0.26), red (>0.26) per instrument. Observation only.
- **FIXED** `_finalize_prior_day` bin_width bug. Hardcoded `bin_width=5.0` → `self._weekly_bin_width.get(inst, 5.0)`. MNQ now uses bin_width=2.0 for prior-day VPOC.
- **COMPLETED** Developing VPOC Forensics (1,088 trades, 12.8 months). Key findings: (1) dVPOC proximity is NEGATIVE for momentum entries overall (WR -5.4pp at 0-5pts, +3.5pp at 20+pts). (2) **VCR regime is strongest signal** — Q4 (concentrated, >0.263) WR 62.9% vs Q1 (dispersed) 78.3% (-15.4pp). (3) dVPOC-VWAP consensus is second strongest — when dPOC and VWAP agree (<3pts), WR drops 6.7pp. (4) vScalpA does BETTER near dVPOC (TP=7 captures magnet bounce). (5) MES shows strongest distance effect (+9.6pp from 0-5 to 20+). (6) Stability index NOT a signal (-1.5pp). (7) Prior-day VPOC for MNQ NOT actionable (94% of entries >10pts away). Script: `backtesting_engine/strategies/developing_vpoc_forensics.py`.
- **COMPLETED** VCR + dVPOC gate sweep (IS/OOS, pre-filter). VCR thresholds [0.10-0.30] × min_bars [0,30,60] × 4 strategies + MES dVPOC distance [5,7,10]. Script: `backtesting_engine/strategies/vcr_dvpoc_gate_sweep.py`.
- **REJECTED** VCR as hard gate for vScalpA/vScalpB. vScalpA: IS often negative, tiny improvements (1-4% PF). vScalpB: filter-resistant as always, IS degrades at every threshold.
- **REJECTED** VCR as hard gate for MES v2. Blocks 56% of entries at best threshold (0.25), loses raw P&L on both IS (-$871) and OOS (-$330). MES continues to reject all gate candidates.
- **REJECTED** MES dVPOC distance gate. All 3 thresholds show IS/OOS divergence. FAIL.
- **SHELVED** VCR gate for vScalpC. VCR>0.15 shows STRONG IS/OOS PF improvement (+5.6%/+32.6%) but IS P&L drops -$111. VCR>0.30 adds P&L on both sides but only blocks 5 trades/year. Adopted as observation (dashboard badge + Supabase logging) — Digest Agent to monitor and flag if pattern strengthens. Revisit when 50+ VCR>0.26 observations collected.
- **IMPLEMENTED** VCR data pipeline to Supabase. Migration 010: `dvpoc_price`, `dvpoc_strength`, `dvpoc_stability` columns on `gate_state_snapshots`. Digest Agent `get_market_regime` and `get_gate_state` tools can query VCR data.
- **COMPLETED** RSI Trendline Breakout backtest (CAE-ATL). Pine indicator (`strategies/cae_auto_trendlines.pine`) + standalone backtest engine (`backtesting_engine/strategies/rsi_trendline_backtest.py`). Based on Jason's manual breakout trading methodology — auto-detect descending RSI peaks (long setup) and ascending RSI troughs (short setup), signal on trendline break.
- **BACKTESTED** RSI Trendline single-exit: Best config RSI(8) TP=15 SL=30 CD=20 Cutoff=13:00. 1,573 trades, WR 65.6%, PF 1.084, Sharpe 0.591, +$3,416. IS Sharpe 0.756 / OOS Sharpe 0.432. OOS holds — WR stable at 65.5%, PF degrades modestly.
- **BACKTESTED** RSI Trendline runner: Best config TP1=7 TP2=20 SL=40 CD=30. 2,314 trades, WR 71.8%, PF 1.140, Sharpe 0.699, +$7,686. IS Sharpe 0.496 / OOS Sharpe 0.893 — **OOS stronger than IS** (opposite of overfitting). Runner mode captures genuine momentum continuation.
- **NOTED** RSI(8) wins over RSI(11) and RSI(14). 13:00 ET cutoff helps. ~40k raw signals, ~1,500-2,300 trades after filtering — abundant signal, standalone strategy viable. Correlation with existing portfolio not yet tested.
- **DECIDED** Redesign Digest Agent as Daily Log + Weekly Insight Scan. Daily: pure code or Ollama, postmortem table saved to Supabase ($0.10 or free). Weekly: Claude scan across 20-40 trades for pattern detection + `flags_for_frontier` ($0.50-1.00). Total ~$1.50/week vs ~$10.50/week. Current 22-tool Digest paused. Daily log doubles as first Ollama learning project.
- **TESTED** RSI entry filter for RSI TL (require RSI > X for longs, < Y for shorts). Symmetric [50/50 through 65/35] and asymmetric [60/45, 63/40, 65/40, etc.]. ALL configs degrade OOS PF except Long>=50/Short<=50 (+0.019, marginal). Post-filter observation showed weak entries near RSI 50 (PF 0.919-0.973) but pre-filter backtest shows removing them changes cooldown cascade and makes overall system worse. **REJECTED** — post-filter ≠ pre-filter strikes again.
- **INVESTIGATED** Missed RSI TL long at 12:35 ET (Mar 16 live). Root cause: CD=30 cooldown. Prior trade SL'd at 12:21, cooldown blocked entry until 12:51. The missed long would have caught the move from ~24650 to 24716 (+$60-100). Next entry after cooldown was a SHORT at 12:58 that also SL'd (-$174). Also: the 12:19 big loser entered at RSI=49.5 which is near-neutral — not a strong trendline break signal. The 12:58 short that followed (RSI=52.2) was also weak. Both losing trades had RSI near 50. Worth investigating: should RSI TL require RSI to be further from 50 at entry (e.g., RSI > 55 for longs, RSI < 45 for shorts)?
- **FOUND** SM alignment predicts RSI TL trade quality. Backtest on 2,314 trades: SM-aligned entries PF 1.344 (+$7.24/trade) vs SM-opposed PF 1.104 (+$2.58/trade) — 2.8x more profit per trade. At high SM conviction (|SM|>0.50): aligned PF 1.633 vs opposed 1.234, WR delta +4.5pp. Opposed entries are still profitable (don't hard-block), but significantly weaker. Options to explore: (1) drop to 1 contract on opposed entries, (2) tighter SL on opposed, (3) SM as conviction filter for position sizing. Live observation Mar 16: RSI TL missed a long entry that aligned with SM — chart showed BRK signal on RSI trendline right as SM turned bullish.
- **TESTED** Conditional counter-signal flip for MES v2. Sweep: min_bars [30,45,60,75] × min_loss [0,5,10]. ALL configs degrade FULL PF (1.413→1.160-1.398). Counter-signals during losing trades are whipsaw noise from slow SM(EMA=255) — new trades after flip average negative P&L. Aggressive flipping (min_bars=30) nearly doubles MaxDD. **REJECTED** — current behavior (hold to SL/TP/BE_TIME) is correct. Script: `mes_conditional_flip_sweep.py`.
- **PRIOR RESEARCH ITEM** Conditional counter-signal flip for MES v2. Currently, while in a position, ALL counter-signals are ignored (strategy.py line 681: entries only when `position == 0`). Jason observed a 60+ bar losing long blocking a valid short signal. Test: if trade is old (60+ bars) AND negative AND a counter-signal fires, close the loser and enter the new direction. This is NOT the blanket SM flip exit (which failed OOS) — it's conditional on trade age + P&L. Would require backtest of "stale losing trade flip" vs current hold-to-SL/TP/BE_TIME behavior.
- **TESTED** MES v2 SL→BE after TP1. Full: PF 1.413→1.396, Net $7,706→$6,931 (-$775). IS and OOS both degrade. WR increases (73.6%→80.8%) but raw P&L drops — runner needs room to breathe below entry then recover to TP2=20. **REJECTED** — current config (no BE) is correct.
- **OBSERVED** (live, Mar 16): MES runner pulled back to dPOC after being profitable — confirming magnet effect in real time. Jason's idea: dynamic runner exit using dPOC proximity + time in trade ("if runner open 30+ bars and dPOC within 5pts, take profit instead of waiting for TP2"). Logged for research after paper trading accumulates data. Supabase dVPOC pipeline already capturing the data needed.
- **NOTED** Domain insight: VPOC magnet effect opposes SM momentum entries. Entries near developing VPOC fight mean-reversion gravity. Effect strongest on MES (wide TP=20 needs room to run) and vScalpC runner (long exposure). vScalpA's TP=7 actually benefits from VPOC proximity (captures magnet bounce).

---

## Meta-Patterns

- **Rejections outnumber implementations ~2:1.** Most ideas fail walk-forward validation.
- **Post-filter != pre-filter** recurs at least 3 times. Blocking entries changes cooldowns/episodes.
- **SM flip exit failure on OOS** is the pivotal finding, driving the entire TP-exit architecture.
- **Per-instrument specialization**: What works on MNQ fails on MES and vice versa (ATR, ADX, cutoffs, VIX, exits).
- **Exit-side filters universally fail** for these strategies. Only entry-side gates show value.
- **Combined/stacked filters always fail** due to geometric trade count reduction.
- **vScalpB (SM_T=0.25) is filter-resistant.** High-conviction entries are uncorrelated with most volatility/momentum features.
- **VPOC magnet opposes momentum entries.** Developing daily VPOC acts as mean-reversion anchor — SM momentum entries near it underperform (except vScalpA's TP=7 which captures the bounce).
- **VCR (volume concentration) is a real regime signal** but too sparse to gate. Monitor as observation, revisit when data accumulates.


## Mar 15, 2026
- **IMPLEMENTED** CAE-ATL Pine Script indicator: RSI(11) Chop-and-Explode with auto-trendlines. Descending peak trendlines (blue, long setups), ascending trough trendlines (orange, short setups). Breakout detection, configurable styles/colors/widths. (`strategies/cae_auto_trendlines.pine`)
- **LEARNED** Pine v6: `line.new()` inside helper functions does not render visible lines. Must inline at top-level scope. Discovered after 3 debugging iterations.
- **LEARNED** Pine v6: trendline grace period needed — trendlines break immediately on creation bar if RSI has already moved past the projected line. Grace = `min_spacing + 2*lb_right` bars.
- **LEARNED** Pine v6: `var int` cannot hold return of function producing `series int`. Inline ternaries work.
