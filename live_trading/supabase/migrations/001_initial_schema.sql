-- ============================================================================
-- NQ Trading — Supabase Schema (Phase 1)
-- 10 tables, 4 analytical views, 3 functions
-- ============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 1. strategy_configs — Versioned config snapshots
-- ============================================================================

CREATE TABLE strategy_configs (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    strategy_id text NOT NULL,
    config jsonb NOT NULL,
    config_hash text NOT NULL,
    active boolean DEFAULT true,
    -- Backtest benchmarks (the numbers this config was validated against)
    backtest_expected_wr double precision,
    backtest_expected_pf double precision,
    backtest_expected_sharpe double precision,
    backtest_max_dd double precision,
    decision_id uuid,  -- FK added after decisions table created
    notes text,
    created_at timestamptz DEFAULT now()
);

CREATE UNIQUE INDEX idx_strategy_configs_active
    ON strategy_configs (strategy_id) WHERE active = true;
CREATE INDEX idx_strategy_configs_hash ON strategy_configs (config_hash);

-- ============================================================================
-- 2. research_runs — Each backtest sweep/validation run
-- ============================================================================

CREATE TABLE research_runs (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    strategy_id text NOT NULL,
    run_name text NOT NULL,
    script text,
    data_range text,           -- e.g. "2025-02-17 to 2026-03-05"
    split text,                -- "full", "IS", "OOS"
    params jsonb NOT NULL,
    param_hash text NOT NULL,
    -- Aggregate results
    trade_count int,
    total_pnl double precision,
    win_rate double precision,
    profit_factor double precision,
    sharpe double precision,
    max_drawdown double precision,
    notes text,
    created_at timestamptz DEFAULT now()
);

CREATE UNIQUE INDEX idx_research_runs_param_hash ON research_runs (param_hash);
CREATE INDEX idx_research_runs_strategy ON research_runs (strategy_id);

-- ============================================================================
-- 3. trades — Every trade ever taken, unified
-- ============================================================================

CREATE TABLE trades (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    strategy_id text NOT NULL,
    instrument text NOT NULL,
    side text NOT NULL CHECK (side IN ('long', 'short')),
    entry_price double precision NOT NULL,
    exit_price double precision NOT NULL,
    entry_time timestamptz NOT NULL,
    exit_time timestamptz,
    pts double precision NOT NULL,
    pnl_net numeric(12,2) NOT NULL,
    exit_reason text NOT NULL,
    bars_held int DEFAULT 0,
    qty int DEFAULT 1,
    is_partial boolean DEFAULT false,

    -- Entry context (populated by strategy at entry time)
    entry_sm_value double precision,
    entry_sm_velocity double precision,
    entry_rsi_value double precision,
    entry_bar_volume double precision,
    entry_minutes_from_open int,

    -- Exit context (populated by strategy at exit time)
    exit_sm_value double precision,
    exit_rsi_value double precision,
    is_runner boolean,  -- True if TP1 already filled (this is the runner leg)

    -- Gate state snapshot at entry
    gate_vix_close double precision,
    gate_leledc_active boolean,
    gate_atr_value double precision,
    gate_adr_ratio double precision,

    -- MFE/MAE in points
    mfe_pts double precision,
    mae_pts double precision,
    -- Entry bar context
    entry_bar_range double precision,  -- bar high-low at entry (volatility proxy)
    concurrent_positions smallint,    -- other open positions at entry time
    streak_at_entry smallint,         -- win/loss streak at entry (+win, -loss)

    -- Commission (actual broker commission, NULL until populated from fill data)
    commission numeric(8,2),

    -- Trade grouping (links TP1 partial + runner legs of same position)
    trade_group_id text,  -- e.g. "MNQ_VSCALPC_2026-03-11T14:32:00"
    -- Signal price for slippage analysis
    signal_price double precision,

    -- Metadata
    source text NOT NULL DEFAULT 'paper' CHECK (source IN ('live', 'paper', 'backtest')),
    trade_date date NOT NULL,  -- ET date, computed by application
    day_of_week smallint GENERATED ALWAYS AS (EXTRACT(isodow FROM trade_date)::smallint) STORED,
    config_id uuid REFERENCES strategy_configs(id),
    research_run_id uuid REFERENCES research_runs(id),
    notes text,

    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_trades_strategy_date ON trades (strategy_id, trade_date);
CREATE INDEX idx_trades_instrument_date ON trades (instrument, trade_date);
CREATE INDEX idx_trades_source ON trades (source);
CREATE INDEX idx_trades_exit_reason ON trades (exit_reason);
CREATE INDEX idx_trades_entry_time ON trades (entry_time);
CREATE INDEX idx_trades_exit_time ON trades (exit_time);
CREATE INDEX idx_trades_group ON trades (trade_group_id, is_partial, source) WHERE trade_group_id IS NOT NULL;
-- Dedup: prevent duplicate inserts from re-fired events
CREATE UNIQUE INDEX idx_trades_dedup
    ON trades (strategy_id, entry_time, is_partial, source);

-- ============================================================================
-- 4. blocked_signals — Counterfactual logging
-- ============================================================================

CREATE TABLE blocked_signals (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    signal_time timestamptz NOT NULL,
    strategy_id text NOT NULL,
    instrument text NOT NULL,
    side text NOT NULL CHECK (side IN ('long', 'short')),
    price double precision NOT NULL,
    sm_value double precision,
    rsi_value double precision,

    -- Gate info
    gate_type text NOT NULL,  -- vix_death_zone, leledc, prior_day_level, adr_directional, etc.
    reason text NOT NULL,
    gate_vix_close double precision,
    gate_leledc_active boolean,
    gate_atr_value double precision,
    gate_adr_ratio double precision,

    -- Counterfactual outcome (NULL initially, filled by batch replay)
    cf_exit_price double precision,
    cf_exit_reason text,
    cf_pnl_pts double precision,
    cf_bars_held int,
    cf_mfe_pts double precision,
    cf_mae_pts double precision,

    signal_date date NOT NULL,  -- ET date, computed by application
    source text NOT NULL DEFAULT 'paper' CHECK (source IN ('live', 'paper')),
    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_blocked_signals_strategy_date ON blocked_signals (strategy_id, signal_date);
CREATE INDEX idx_blocked_signals_gate_type ON blocked_signals (gate_type);
CREATE INDEX idx_blocked_signals_time ON blocked_signals (signal_time);
-- Dedup: prevent duplicate blocked signal inserts on re-run
CREATE UNIQUE INDEX idx_blocked_signals_dedup
    ON blocked_signals (signal_time, strategy_id, instrument, source);

-- ============================================================================
-- 5. research_results — Individual parameter combos within a sweep
-- ============================================================================

CREATE TABLE research_results (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    sweep_params jsonb NOT NULL,
    -- Metrics
    trade_count int,
    win_rate double precision,
    profit_factor double precision,
    sharpe double precision,
    total_pnl double precision,
    max_drawdown double precision,
    -- IS/OOS paired results
    is_pf double precision,
    is_sharpe double precision,
    is_wr double precision,
    oos_pf double precision,
    oos_sharpe double precision,
    oos_wr double precision,
    -- Verdict
    verdict text CHECK (verdict IN ('PASS', 'FAIL', 'MARGINAL')),
    evidence_summary text,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_research_results_run ON research_results (run_id);
CREATE INDEX idx_research_results_params ON research_results USING gin (sweep_params);
CREATE INDEX idx_research_results_verdict ON research_results (verdict);

-- ============================================================================
-- 6. decisions — Structured decision log
-- ============================================================================

CREATE TABLE decisions (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    decision_date date NOT NULL DEFAULT CURRENT_DATE,
    action text NOT NULL CHECK (action IN (
        'implement', 'reject', 'shelve', 'change', 'reverse'
    )),
    category text NOT NULL CHECK (category IN (
        'strategy_params', 'entry_filter', 'exit_logic',
        'risk_management', 'infrastructure', 'portfolio',
        'research', 'operations'
    )),
    strategy_id text,
    summary text NOT NULL,
    rationale text,
    evidence text,
    research_run_id uuid REFERENCES research_runs(id),
    superseded_by uuid REFERENCES decisions(id),
    claude_session_id uuid,  -- FK added after claude_sessions table
    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_decisions_date ON decisions (decision_date);
CREATE INDEX idx_decisions_strategy ON decisions (strategy_id);
CREATE INDEX idx_decisions_action ON decisions (action);

-- Add FK from strategy_configs to decisions
ALTER TABLE strategy_configs
    ADD CONSTRAINT fk_strategy_configs_decision
    FOREIGN KEY (decision_id) REFERENCES decisions(id);

-- ============================================================================
-- 7. gate_state_snapshots — Daily gate state for debugging
-- ============================================================================

CREATE TABLE gate_state_snapshots (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    snapshot_date date NOT NULL,
    instrument text NOT NULL,
    -- Gate values
    vix_close double precision,
    adr_value double precision,
    atr_value double precision,
    leledc_bull_count int,
    leledc_bear_count int,
    prior_day_high double precision,
    prior_day_low double precision,
    prior_day_vpoc double precision,
    prior_day_vah double precision,
    prior_day_val double precision,
    -- Session info
    rth_open double precision,
    rth_high double precision,
    rth_low double precision,
    rth_close double precision,
    created_at timestamptz DEFAULT now(),
    UNIQUE (snapshot_date, instrument)
);

CREATE INDEX idx_gate_snapshots_date ON gate_state_snapshots (snapshot_date);

-- ============================================================================
-- 8. claude_sessions — AI session continuity
-- ============================================================================

CREATE TABLE claude_sessions (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    session_date date NOT NULL DEFAULT CURRENT_DATE,
    summary text,
    topics text[],
    files_modified text[],
    decisions_made text[],
    open_questions text[],
    next_steps text[],
    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_claude_sessions_date ON claude_sessions (session_date);

-- Add FK from decisions
ALTER TABLE decisions
    ADD CONSTRAINT fk_decisions_claude_session
    FOREIGN KEY (claude_session_id) REFERENCES claude_sessions(id);

-- ============================================================================
-- 9. bars_daily — Daily OHLCV summary
-- ============================================================================

CREATE TABLE bars_daily (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    bar_date date NOT NULL,
    instrument text NOT NULL,
    session_type text DEFAULT 'rth' CHECK (session_type IN ('rth', 'full')),
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume double precision,
    daily_range double precision GENERATED ALWAYS AS (high - low) STORED,
    created_at timestamptz DEFAULT now(),
    UNIQUE (bar_date, instrument, session_type)
);

CREATE INDEX idx_bars_daily_instrument_date ON bars_daily (instrument, bar_date);

-- ============================================================================
-- 10. trade_annotations — Post-trade analysis
-- ============================================================================

CREATE TABLE trade_annotations (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    trade_id uuid NOT NULL REFERENCES trades(id) ON DELETE CASCADE,
    annotation_type text NOT NULL CHECK (annotation_type IN (
        'investigation', 'pattern', 'lesson', 'correction'
    )),
    content text NOT NULL,
    created_by text,  -- claude_session_id or 'manual'
    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_trade_annotations_trade ON trade_annotations (trade_id);
CREATE INDEX idx_trade_annotations_type ON trade_annotations (annotation_type);

-- ============================================================================
-- V1. daily_stats — Materialized view
-- ============================================================================

CREATE MATERIALIZED VIEW daily_stats AS
SELECT
    t.strategy_id,
    t.trade_date,
    t.instrument,
    t.source,
    COUNT(*) AS trade_count,
    COUNT(*) FILTER (WHERE t.pnl_net > 0) AS wins,
    COUNT(*) FILTER (WHERE t.pnl_net <= 0) AS losses,
    ROUND(
        (COUNT(*) FILTER (WHERE t.pnl_net > 0))::numeric
        / NULLIF(COUNT(*), 0) * 100, 1
    ) AS win_rate,
    ROUND(SUM(t.pnl_net)::numeric, 2) AS total_pnl,
    ROUND(
        (SUM(t.pnl_net) FILTER (WHERE t.pnl_net > 0))::numeric
        / NULLIF(ABS(SUM(t.pnl_net) FILTER (WHERE t.pnl_net <= 0)), 0)::numeric,
        3
    ) AS profit_factor,
    -- Exit reason breakdown
    COUNT(*) FILTER (WHERE t.exit_reason = 'TP') AS tp_count,
    COUNT(*) FILTER (WHERE t.exit_reason = 'TP1') AS tp1_count,
    COUNT(*) FILTER (WHERE t.exit_reason = 'SL') AS sl_count,
    COUNT(*) FILTER (WHERE t.exit_reason = 'BE_TIME') AS be_time_count,
    COUNT(*) FILTER (WHERE t.exit_reason = 'EOD') AS eod_count,
    -- Averages
    ROUND(AVG(t.bars_held)::numeric, 1) AS avg_bars_held,
    ROUND(AVG(t.mfe_pts)::numeric, 2) AS avg_mfe,
    ROUND(AVG(t.mae_pts)::numeric, 2) AS avg_mae
FROM trades t
WHERE t.trade_date IS NOT NULL
GROUP BY t.strategy_id, t.trade_date, t.instrument, t.source;

CREATE UNIQUE INDEX idx_daily_stats_pk
    ON daily_stats (strategy_id, trade_date, instrument, source);

-- ============================================================================
-- V2. rolling_performance — Materialized view
-- ============================================================================

CREATE MATERIALIZED VIEW rolling_performance AS
WITH numbered AS (
    SELECT
        t.strategy_id,
        t.source,
        t.pnl_net,
        t.exit_time,
        t.trade_date,
        ROW_NUMBER() OVER (
            PARTITION BY t.strategy_id, t.source
            ORDER BY t.exit_time
        ) AS trade_num,
        SUM(t.pnl_net) OVER (
            PARTITION BY t.strategy_id, t.source
            ORDER BY t.exit_time
        ) AS cumulative_pnl,
        -- Rolling 20-trade stats using window frames (O(n), not O(n²))
        COUNT(*) OVER w20 AS r20_count,
        COUNT(*) FILTER (WHERE t.pnl_net > 0) OVER w20 AS r20_wins,
        SUM(t.pnl_net) OVER w20 AS r20_pnl,
        SUM(t.pnl_net) FILTER (WHERE t.pnl_net > 0) OVER w20 AS r20_gross_profit,
        SUM(t.pnl_net) FILTER (WHERE t.pnl_net <= 0) OVER w20 AS r20_gross_loss
    FROM trades t
    WHERE t.exit_time IS NOT NULL
    WINDOW w20 AS (
        PARTITION BY t.strategy_id, t.source
        ORDER BY t.exit_time
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    )
)
SELECT
    n.strategy_id,
    n.source,
    n.trade_num,
    n.exit_time,
    n.trade_date,
    n.cumulative_pnl,
    ROUND((n.r20_wins)::numeric / NULLIF(n.r20_count, 0) * 100, 1) AS rolling_20_wr,
    ROUND(
        (n.r20_gross_profit)::numeric
        / NULLIF(ABS(n.r20_gross_loss), 0)::numeric,
        3
    ) AS rolling_20_pf,
    ROUND(n.r20_pnl::numeric, 2) AS rolling_20_pnl,
    ROUND((n.r20_pnl / NULLIF(n.r20_count, 0))::numeric, 2) AS rolling_20_avg_pnl,
    sc.backtest_expected_wr,
    sc.backtest_expected_pf,
    sc.backtest_expected_sharpe,
    sc.backtest_max_dd
FROM numbered n
LEFT JOIN strategy_configs sc ON sc.strategy_id = n.strategy_id AND sc.active = true;

CREATE UNIQUE INDEX idx_rolling_performance_pk
    ON rolling_performance (strategy_id, source, trade_num);

-- ============================================================================
-- V3. gate_effectiveness — View (not materialized, always fresh)
-- ============================================================================

CREATE VIEW gate_effectiveness AS
SELECT
    bs.gate_type,
    bs.strategy_id,
    date_trunc('week', bs.signal_date) AS week_start,
    COUNT(*) AS signals_blocked,
    COUNT(bs.cf_pnl_pts) AS counterfactuals_computed,
    ROUND(COALESCE(SUM(bs.cf_pnl_pts), 0)::numeric, 2) AS counterfactual_net_pnl,
    ROUND(
        COALESCE(SUM(bs.cf_pnl_pts) FILTER (WHERE bs.cf_pnl_pts < 0), 0)::numeric, 2
    ) AS losses_avoided,
    ROUND(
        COALESCE(SUM(bs.cf_pnl_pts) FILTER (WHERE bs.cf_pnl_pts > 0), 0)::numeric, 2
    ) AS gains_missed
FROM blocked_signals bs
GROUP BY bs.gate_type, bs.strategy_id, date_trunc('week', bs.signal_date::timestamptz);

-- ============================================================================
-- V4. live_vs_backtest — View
-- ============================================================================

CREATE VIEW live_vs_backtest AS
WITH live_stats AS (
    SELECT
        t.strategy_id,
        t.source,
        COUNT(*) AS total_trades,
        ROUND(
            (COUNT(*) FILTER (WHERE t.pnl_net > 0))::numeric
            / NULLIF(COUNT(*), 0) * 100, 1
        ) AS live_wr,
        ROUND(
            (SUM(t.pnl_net) FILTER (WHERE t.pnl_net > 0))::numeric
            / NULLIF(ABS(SUM(t.pnl_net) FILTER (WHERE t.pnl_net <= 0)), 0)::numeric,
            3
        ) AS live_pf,
        ROUND(SUM(t.pnl_net)::numeric, 2) AS live_total_pnl,
        MIN(t.trade_date) AS first_trade_date,
        MAX(t.trade_date) AS last_trade_date
    FROM trades t
    WHERE t.source IN ('live', 'paper')
      AND t.exit_time IS NOT NULL
      AND NOT t.is_partial  -- exclude TP1 partial legs; count each position once
    GROUP BY t.strategy_id, t.source
)
SELECT
    ls.strategy_id,
    ls.source,
    ls.total_trades,
    ls.live_wr,
    ls.live_pf,
    ls.live_total_pnl,
    ls.first_trade_date,
    ls.last_trade_date,
    sc.backtest_expected_wr,
    sc.backtest_expected_pf,
    sc.backtest_expected_sharpe,
    sc.backtest_max_dd,
    -- Deviations
    ROUND((ls.live_wr - sc.backtest_expected_wr)::numeric, 1) AS wr_deviation,
    ROUND((ls.live_pf - sc.backtest_expected_pf)::numeric, 3) AS pf_deviation,
    -- Statistical Z-score: accounts for sample size
    -- Z = (live_wr/100 - bt_wr/100) / sqrt(bt_wr/100 * (1 - bt_wr/100) / n)
    ROUND(
        CASE WHEN sc.backtest_expected_wr IS NOT NULL
                  AND sc.backtest_expected_wr > 0
                  AND sc.backtest_expected_wr < 100
                  AND ls.total_trades >= 10
            THEN (ls.live_wr / 100.0 - sc.backtest_expected_wr / 100.0)
                 / NULLIF(SQRT(
                     (sc.backtest_expected_wr / 100.0)
                     * (1 - sc.backtest_expected_wr / 100.0)
                     / ls.total_trades
                 ), 0)
            ELSE NULL
        END::numeric, 2
    ) AS wr_z_score,
    -- Status based on Z-score (one-tailed: only alarm on underperformance)
    -- Outperformance is never a concern; only flag when live WR < backtest WR significantly
    CASE
        WHEN sc.backtest_expected_wr IS NULL THEN 'NO_BENCHMARK'
        WHEN ls.total_trades < 20 THEN 'INSUFFICIENT_DATA'
        WHEN (ls.live_wr / 100.0 - sc.backtest_expected_wr / 100.0)
             / NULLIF(SQRT(
                 (sc.backtest_expected_wr / 100.0)
                 * (1 - sc.backtest_expected_wr / 100.0)
                 / ls.total_trades
             ), 0) >= -1.645 THEN 'GREEN'   -- not significantly below (p>0.05)
        WHEN (ls.live_wr / 100.0 - sc.backtest_expected_wr / 100.0)
             / NULLIF(SQRT(
                 (sc.backtest_expected_wr / 100.0)
                 * (1 - sc.backtest_expected_wr / 100.0)
                 / ls.total_trades
             ), 0) >= -2.326 THEN 'YELLOW'  -- significantly below at 5% but not 1%
        ELSE 'RED'                           -- significantly below at 1%
    END AS status
FROM live_stats ls
LEFT JOIN strategy_configs sc ON sc.strategy_id = ls.strategy_id AND sc.active = true;

-- ============================================================================
-- V4b. tod_performance — Time-of-day performance buckets (1-hour ET)
-- ============================================================================

CREATE VIEW tod_performance AS
SELECT
    t.strategy_id,
    t.source,
    EXTRACT(hour FROM t.entry_time AT TIME ZONE 'America/New_York')::int AS entry_hour_et,
    COUNT(*) AS trade_count,
    COUNT(*) FILTER (WHERE t.pnl_net > 0) AS wins,
    ROUND(
        (COUNT(*) FILTER (WHERE t.pnl_net > 0))::numeric
        / NULLIF(COUNT(*), 0) * 100, 1
    ) AS win_rate,
    ROUND(SUM(t.pnl_net)::numeric, 2) AS total_pnl,
    ROUND(AVG(t.pnl_net)::numeric, 2) AS avg_pnl,
    ROUND(
        (SUM(t.pnl_net) FILTER (WHERE t.pnl_net > 0))::numeric
        / NULLIF(ABS(SUM(t.pnl_net) FILTER (WHERE t.pnl_net <= 0)), 0)::numeric,
        3
    ) AS profit_factor,
    ROUND(AVG(t.mfe_pts)::numeric, 2) AS avg_mfe,
    ROUND(AVG(t.mae_pts)::numeric, 2) AS avg_mae,
    ROUND(AVG(t.bars_held)::numeric, 1) AS avg_bars_held
FROM trades t
WHERE t.exit_time IS NOT NULL
  AND t.entry_time IS NOT NULL
  AND NOT t.is_partial  -- exclude TP1 partial legs; count each position once
GROUP BY t.strategy_id, t.source,
         EXTRACT(hour FROM t.entry_time AT TIME ZONE 'America/New_York')::int;

-- ============================================================================
-- F1. f_morning_briefing() — Everything a new session needs
-- ============================================================================

CREATE OR REPLACE FUNCTION f_morning_briefing()
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    result jsonb;
    today date := (now() AT TIME ZONE 'America/New_York')::date;
    yesterday date := today - 1;
BEGIN
    SELECT jsonb_build_object(
        'date', today,

        -- Yesterday's trades
        'yesterday_trades', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'strategy_id', t.strategy_id,
                'side', t.side,
                'pnl_net', t.pnl_net,
                'exit_reason', t.exit_reason,
                'entry_time', t.entry_time,
                'exit_time', t.exit_time
            ) ORDER BY t.exit_time)
            FROM trades t
            WHERE t.trade_date = yesterday
              AND t.source IN ('live', 'paper')
        ), '[]'::jsonb),

        -- Yesterday's P&L by strategy
        'yesterday_pnl', COALESCE((
            SELECT jsonb_object_agg(
                ds.strategy_id,
                jsonb_build_object(
                    'pnl', ds.total_pnl,
                    'trades', ds.trade_count,
                    'win_rate', ds.win_rate,
                    'pf', ds.profit_factor
                )
            )
            FROM daily_stats ds
            WHERE ds.trade_date = yesterday
              AND ds.source IN ('live', 'paper')
        ), '{}'::jsonb),

        -- Today's gate state
        'gate_state', COALESCE((
            SELECT jsonb_object_agg(
                gs.instrument,
                jsonb_build_object(
                    'vix_close', gs.vix_close,
                    'adr_value', gs.adr_value,
                    'atr_value', gs.atr_value,
                    'leledc_bull', gs.leledc_bull_count,
                    'leledc_bear', gs.leledc_bear_count,
                    'prior_day_high', gs.prior_day_high,
                    'prior_day_low', gs.prior_day_low
                )
            )
            FROM gate_state_snapshots gs
            WHERE gs.snapshot_date = yesterday
        ), '{}'::jsonb),

        -- Live vs backtest status
        'strategy_status', COALESCE((
            SELECT jsonb_object_agg(
                lv.strategy_id,
                jsonb_build_object(
                    'live_wr', lv.live_wr,
                    'backtest_wr', lv.backtest_expected_wr,
                    'live_pf', lv.live_pf,
                    'backtest_pf', lv.backtest_expected_pf,
                    'status', lv.status,
                    'total_trades', lv.total_trades
                )
            )
            FROM live_vs_backtest lv
        ), '{}'::jsonb),

        -- Gate effectiveness last 7 days
        'gate_effectiveness_7d', COALESCE((
            SELECT jsonb_object_agg(sub.gate_type, sub.stats)
            FROM (
                SELECT
                    bs.gate_type,
                    jsonb_build_object(
                        'blocked', COUNT(*),
                        'cf_pnl', ROUND(COALESCE(SUM(bs.cf_pnl_pts), 0)::numeric, 2)
                    ) AS stats
                FROM blocked_signals bs
                WHERE bs.signal_date >= today - 7
                GROUP BY bs.gate_type
            ) sub
        ), '{}'::jsonb),

        -- Portfolio total yesterday (prefer 'live' over 'paper')
        'yesterday_portfolio', COALESCE((
            SELECT jsonb_build_object(
                'pnl', pd.portfolio_pnl,
                'trades', pd.total_trades,
                'wr', pd.portfolio_wr,
                'pf', pd.portfolio_pf,
                'source', pd.source
            )
            FROM portfolio_daily pd
            WHERE pd.trade_date = yesterday
              AND pd.source IN ('live', 'paper')
            ORDER BY (pd.source = 'live') DESC
            LIMIT 1
        ), '{}'::jsonb),

        -- Drawdown status (current drawdown per strategy, prefer live source)
        'drawdown_status', COALESCE((
            SELECT jsonb_object_agg(sub.strategy_id, sub.dd_info)
            FROM (
                SELECT DISTINCT ON (d.strategy_id)
                    d.strategy_id,
                    jsonb_build_object(
                        'cumulative_pnl', d.cumulative_pnl,
                        'high_water_mark', d.high_water_mark,
                        'drawdown', d.drawdown,
                        'drawdown_pct', d.drawdown_pct,
                        'source', d.source
                    ) AS dd_info
                FROM drawdown_hwm d
                WHERE d.source IN ('live', 'paper')
                ORDER BY d.strategy_id, (d.source = 'live') DESC, d.trade_num DESC
            ) sub
        ), '{}'::jsonb),

        -- Yesterday's blocked signals count
        'yesterday_blocked', COALESCE((
            SELECT jsonb_object_agg(sub.gate_type, sub.cnt)
            FROM (
                SELECT bs.gate_type, COUNT(*) AS cnt
                FROM blocked_signals bs
                WHERE bs.signal_date = yesterday
                GROUP BY bs.gate_type
            ) sub
        ), '{}'::jsonb),

        -- Rolling performance (latest row per strategy from rolling_performance matview)
        'rolling_performance', COALESCE((
            SELECT jsonb_object_agg(sub.strategy_id, sub.perf)
            FROM (
                SELECT DISTINCT ON (rp.strategy_id)
                    rp.strategy_id,
                    jsonb_build_object(
                        'rolling_20_wr', rp.rolling_20_wr,
                        'rolling_20_pf', rp.rolling_20_pf,
                        'rolling_20_pnl', rp.rolling_20_pnl,
                        'cumulative_pnl', rp.cumulative_pnl,
                        'trade_num', rp.trade_num,
                        'backtest_wr', rp.backtest_expected_wr,
                        'backtest_pf', rp.backtest_expected_pf
                    ) AS perf
                FROM rolling_performance rp
                WHERE rp.source IN ('live', 'paper')
                ORDER BY rp.strategy_id, (rp.source = 'live') DESC, rp.trade_num DESC
            ) sub
        ), '{}'::jsonb),

        -- Last 3 Claude sessions
        'recent_sessions', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'date', cs.session_date,
                'summary', cs.summary,
                'open_questions', cs.open_questions,
                'next_steps', cs.next_steps
            ) ORDER BY cs.created_at DESC)
            FROM (
                SELECT * FROM claude_sessions
                ORDER BY created_at DESC LIMIT 3
            ) cs
        ), '[]'::jsonb)

    ) INTO result;

    RETURN result;
END;
$$;

-- ============================================================================
-- F2. f_has_been_tested() — Prevent re-testing rejected ideas
-- ============================================================================

CREATE OR REPLACE FUNCTION f_has_been_tested(
    p_strategy_id text,
    p_param_key text,
    p_param_value text
)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    result jsonb;
BEGIN
    SELECT jsonb_build_object(
        'tested', true,
        'run_id', rr.run_id,
        'verdict', rr.verdict,
        'evidence', rr.evidence_summary,
        'run_name', r.run_name,
        'run_date', r.created_at
    ) INTO result
    FROM research_results rr
    JOIN research_runs r ON r.id = rr.run_id
    WHERE r.strategy_id = p_strategy_id
      AND rr.sweep_params @> jsonb_build_object(p_param_key, p_param_value)
    ORDER BY r.created_at DESC
    LIMIT 1;

    IF result IS NULL THEN
        -- Try numeric comparison
        BEGIN
            SELECT jsonb_build_object(
                'tested', true,
                'run_id', rr.run_id,
                'verdict', rr.verdict,
                'evidence', rr.evidence_summary,
                'run_name', r.run_name,
                'run_date', r.created_at
            ) INTO result
            FROM research_results rr
            JOIN research_runs r ON r.id = rr.run_id
            WHERE r.strategy_id = p_strategy_id
              AND rr.sweep_params @> jsonb_build_object(p_param_key, p_param_value::numeric)
            ORDER BY r.created_at DESC
            LIMIT 1;
        EXCEPTION WHEN OTHERS THEN
            NULL;  -- Not a number, skip numeric comparison
        END;
    END IF;

    RETURN COALESCE(result, jsonb_build_object('tested', false));
END;
$$;

-- ============================================================================
-- F3. f_trade_context() — Full enriched context for a trade
-- ============================================================================

CREATE OR REPLACE FUNCTION f_trade_context(p_trade_id uuid)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    result jsonb;
BEGIN
    SELECT jsonb_build_object(
        'trade', jsonb_build_object(
            'id', t.id,
            'strategy_id', t.strategy_id,
            'instrument', t.instrument,
            'side', t.side,
            'entry_price', t.entry_price,
            'exit_price', t.exit_price,
            'entry_time', t.entry_time,
            'exit_time', t.exit_time,
            'pts', t.pts,
            'pnl_net', t.pnl_net,
            'exit_reason', t.exit_reason,
            'bars_held', t.bars_held,
            'qty', t.qty,
            'is_partial', t.is_partial,
            'source', t.source,
            'trade_date', t.trade_date
        ),
        'entry_context', jsonb_build_object(
            'sm_value', t.entry_sm_value,
            'sm_velocity', t.entry_sm_velocity,
            'rsi_value', t.entry_rsi_value,
            'bar_volume', t.entry_bar_volume,
            'minutes_from_open', t.entry_minutes_from_open,
            'bar_range', t.entry_bar_range,
            'concurrent_positions', t.concurrent_positions,
            'streak_at_entry', t.streak_at_entry
        ),
        'exit_context', jsonb_build_object(
            'sm_value', t.exit_sm_value,
            'rsi_value', t.exit_rsi_value
        ),
        'gate_state', jsonb_build_object(
            'vix_close', t.gate_vix_close,
            'leledc_active', t.gate_leledc_active,
            'atr_value', t.gate_atr_value,
            'adr_ratio', t.gate_adr_ratio
        ),
        'excursion', jsonb_build_object(
            'mfe_pts', t.mfe_pts,
            'mae_pts', t.mae_pts
        ),
        'annotations', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'type', ta.annotation_type,
                'content', ta.content,
                'created_at', ta.created_at,
                'created_by', ta.created_by
            ) ORDER BY ta.created_at)
            FROM trade_annotations ta
            WHERE ta.trade_id = t.id
        ), '[]'::jsonb),
        -- Nearby trades (same strategy, +/- 2 hours)
        'nearby_trades', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'id', nt.id,
                'side', nt.side,
                'pnl_net', nt.pnl_net,
                'exit_reason', nt.exit_reason,
                'entry_time', nt.entry_time
            ) ORDER BY nt.entry_time)
            FROM trades nt
            WHERE nt.strategy_id = t.strategy_id
              AND nt.id != t.id
              AND nt.entry_time BETWEEN t.entry_time - interval '2 hours'
                                    AND t.entry_time + interval '2 hours'
        ), '[]'::jsonb)
    ) INTO result
    FROM trades t
    WHERE t.id = p_trade_id;

    RETURN result;
END;
$$;

-- ============================================================================
-- Row Level Security (service key bypasses, but policies for future use)
-- ============================================================================

ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE blocked_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE gate_state_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE claude_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE bars_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_annotations ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users full access (service key bypasses RLS anyway)
CREATE POLICY "Authenticated full access" ON trades
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON blocked_signals
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON research_runs
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON research_results
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON decisions
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON strategy_configs
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON gate_state_snapshots
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON claude_sessions
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON bars_daily
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated full access" ON trade_annotations
    FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- Anon read-only access (for React dashboard via Supabase JS anon key)
CREATE POLICY "Anon read access" ON trades
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON blocked_signals
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON strategy_configs
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON gate_state_snapshots
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON research_runs
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON research_results
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON decisions
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON claude_sessions
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON bars_daily
    FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read access" ON trade_annotations
    FOR SELECT TO anon USING (true);

-- Materialized views don't have RLS but anon needs SELECT grant
GRANT SELECT ON daily_stats TO anon;
GRANT SELECT ON rolling_performance TO anon;
GRANT SELECT ON drawdown_hwm TO anon;

-- ============================================================================
-- V5. portfolio_daily — Cross-strategy daily P&L summary
-- ============================================================================

CREATE VIEW portfolio_daily AS
WITH strat_day AS (
    -- Pre-aggregate daily_stats across instruments per strategy
    -- (handles future case where one strategy trades multiple instruments)
    SELECT
        ds.strategy_id,
        ds.trade_date,
        ds.source,
        SUM(ds.trade_count) AS trade_count,
        SUM(ds.wins) AS wins,
        SUM(ds.losses) AS losses,
        ROUND(SUM(ds.wins)::numeric / NULLIF(SUM(ds.trade_count), 0) * 100, 1) AS win_rate,
        ROUND(SUM(ds.total_pnl)::numeric, 2) AS total_pnl,
        ROUND(
            (SUM(ds.total_pnl) FILTER (WHERE ds.total_pnl > 0))::numeric
            / NULLIF(ABS(SUM(ds.total_pnl) FILTER (WHERE ds.total_pnl <= 0)), 0)::numeric,
            3
        ) AS profit_factor
    FROM daily_stats ds
    GROUP BY ds.strategy_id, ds.trade_date, ds.source
),
trade_pf AS (
    -- Compute portfolio PF from trade-level gross profit / gross loss (not strategy-day net)
    SELECT
        t.trade_date,
        t.source,
        ROUND(
            SUM(t.pnl_net) FILTER (WHERE t.pnl_net > 0)::numeric
            / NULLIF(ABS(SUM(t.pnl_net) FILTER (WHERE t.pnl_net <= 0)), 0)::numeric,
            3
        ) AS portfolio_pf
    FROM trades t
    WHERE t.trade_date IS NOT NULL
    GROUP BY t.trade_date, t.source
)
SELECT
    sd.trade_date,
    sd.source,
    SUM(sd.trade_count) AS total_trades,
    SUM(sd.wins) AS total_wins,
    SUM(sd.losses) AS total_losses,
    ROUND(SUM(sd.wins)::numeric / NULLIF(SUM(sd.trade_count), 0) * 100, 1) AS portfolio_wr,
    ROUND(SUM(sd.total_pnl)::numeric, 2) AS portfolio_pnl,
    tpf.portfolio_pf,
    -- Per-strategy breakdown as JSONB (unique keys guaranteed by strat_day pre-agg)
    jsonb_object_agg(
        sd.strategy_id,
        jsonb_build_object(
            'pnl', sd.total_pnl,
            'trades', sd.trade_count,
            'wr', sd.win_rate,
            'pf', sd.profit_factor
        )
    ) AS strategy_breakdown
FROM strat_day sd
LEFT JOIN trade_pf tpf ON tpf.trade_date = sd.trade_date AND tpf.source = sd.source
GROUP BY sd.trade_date, sd.source, tpf.portfolio_pf;

-- ============================================================================
-- V6. drawdown_hwm — Running high-water mark + drawdown tracking
-- ============================================================================

CREATE MATERIALIZED VIEW drawdown_hwm AS
WITH cum AS (
    SELECT
        t.strategy_id,
        t.source,
        t.exit_time,
        t.trade_date,
        SUM(t.pnl_net) OVER (
            PARTITION BY t.strategy_id, t.source ORDER BY t.exit_time
        ) AS cumulative_pnl,
        ROW_NUMBER() OVER (
            PARTITION BY t.strategy_id, t.source ORDER BY t.exit_time
        ) AS trade_num
    FROM trades t
    WHERE t.exit_time IS NOT NULL
),
hwm AS (
    SELECT
        c.*,
        MAX(c.cumulative_pnl) OVER (
            PARTITION BY c.strategy_id, c.source ORDER BY c.exit_time
        ) AS high_water_mark
    FROM cum c
)
SELECT
    h.strategy_id,
    h.source,
    h.trade_num,
    h.exit_time,
    h.trade_date,
    ROUND(h.cumulative_pnl::numeric, 2) AS cumulative_pnl,
    ROUND(h.high_water_mark::numeric, 2) AS high_water_mark,
    ROUND((h.cumulative_pnl - h.high_water_mark)::numeric, 2) AS drawdown,
    -- drawdown_pct: use absolute HWM to handle negative HWM (early drawdown)
    ROUND(
        CASE WHEN h.high_water_mark != 0
            THEN ((h.cumulative_pnl - h.high_water_mark) / ABS(h.high_water_mark) * 100)::numeric
            ELSE 0
        END, 2
    ) AS drawdown_pct
FROM hwm h;

CREATE UNIQUE INDEX idx_drawdown_hwm_pk
    ON drawdown_hwm (strategy_id, source, trade_num);

-- ============================================================================
-- V7. tp1_runner_pairs — Links partial exit legs via trade_group_id
-- ============================================================================

CREATE VIEW tp1_runner_pairs AS
SELECT
    tp1.trade_group_id,
    tp1.strategy_id,
    tp1.instrument,
    tp1.side,
    tp1.entry_time,
    tp1.entry_price,
    tp1.exit_price AS tp1_exit_price,
    tp1.pts AS tp1_pts,
    tp1.pnl_net AS tp1_pnl,
    tp1.exit_reason AS tp1_exit_reason,
    tp1.bars_held AS tp1_bars_held,
    runner.exit_price AS runner_exit_price,
    runner.pts AS runner_pts,
    runner.pnl_net AS runner_pnl,
    runner.exit_reason AS runner_exit_reason,
    runner.bars_held AS runner_bars_held,
    ROUND((tp1.pnl_net + COALESCE(runner.pnl_net, 0))::numeric, 2) AS combined_pnl,
    tp1.trade_date,
    tp1.source
FROM trades tp1
LEFT JOIN trades runner ON runner.trade_group_id = tp1.trade_group_id
    AND runner.is_partial = false
    AND runner.source = tp1.source
WHERE tp1.trade_group_id IS NOT NULL
  AND tp1.is_partial = true;

-- ============================================================================
-- Refresh helper for materialized views
-- ============================================================================

CREATE OR REPLACE FUNCTION refresh_trading_views()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY rolling_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY drawdown_hwm;
END;
$$;
