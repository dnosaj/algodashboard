-- ============================================================================
-- 002: Review Agent Fixes
-- Blocked signals dedup index + anon RLS policies for React dashboard
-- Run this in Supabase SQL Editor
-- ============================================================================

-- 1. Blocked signals dedup index (prevents duplicate rows on backfill re-runs)
CREATE UNIQUE INDEX IF NOT EXISTS idx_blocked_signals_dedup
    ON blocked_signals (signal_time, strategy_id, instrument, source);

-- 2. Anon read-only policies (React dashboard uses anon key, not service key)
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

-- 3. Materialized views: anon needs SELECT grant (no RLS on matviews)
GRANT SELECT ON daily_stats TO anon;
GRANT SELECT ON rolling_performance TO anon;
GRANT SELECT ON drawdown_hwm TO anon;

-- 4. Regular views also need SELECT grant for anon
GRANT SELECT ON portfolio_daily TO anon;
GRANT SELECT ON gate_effectiveness TO anon;
GRANT SELECT ON live_vs_backtest TO anon;
GRANT SELECT ON tp1_runner_pairs TO anon;
GRANT SELECT ON tod_performance TO anon;
