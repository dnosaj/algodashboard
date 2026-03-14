-- ============================================================================
-- 003: Cross-strategy daily P&L correlation view
-- Shows Pearson correlation between each strategy pair over rolling windows.
-- ============================================================================

-- Regular view (not materialized — correlations should always be fresh).
-- Uses daily_stats as the source. Requires paper/live trades with trade_date.

CREATE OR REPLACE VIEW cross_strategy_correlation AS
WITH daily_pnl AS (
    -- Get daily P&L per strategy, only paper/live, exclude partials
    SELECT
        strategy_id,
        trade_date,
        source,
        total_pnl
    FROM daily_stats
    WHERE source IN ('paper', 'live')
),
-- Generate all strategy pairs
pairs AS (
    SELECT DISTINCT
        a.strategy_id AS strategy_a,
        b.strategy_id AS strategy_b,
        a.source
    FROM daily_pnl a
    JOIN daily_pnl b ON a.trade_date = b.trade_date
        AND a.source = b.source
        AND a.strategy_id < b.strategy_id  -- avoid duplicates and self-pairs
),
-- Compute correlation for each pair
corr AS (
    SELECT
        p.strategy_a,
        p.strategy_b,
        p.source,
        COUNT(*) AS overlap_days,
        ROUND(CORR(a.total_pnl, b.total_pnl)::numeric, 3) AS pearson_r,
        ROUND(AVG(a.total_pnl)::numeric, 2) AS avg_pnl_a,
        ROUND(AVG(b.total_pnl)::numeric, 2) AS avg_pnl_b,
        -- How often both lose on the same day (joint drawdown risk)
        COUNT(*) FILTER (WHERE a.total_pnl < 0 AND b.total_pnl < 0) AS both_lose_days,
        -- How often one wins while the other loses (diversification)
        COUNT(*) FILTER (
            WHERE (a.total_pnl > 0 AND b.total_pnl < 0)
               OR (a.total_pnl < 0 AND b.total_pnl > 0)
        ) AS divergent_days
    FROM pairs p
    JOIN daily_pnl a ON a.strategy_id = p.strategy_a
        AND a.source = p.source
    JOIN daily_pnl b ON b.strategy_id = p.strategy_b
        AND b.trade_date = a.trade_date
        AND b.source = p.source
    GROUP BY p.strategy_a, p.strategy_b, p.source
    HAVING COUNT(*) >= 5  -- need at least 5 overlapping days
)
SELECT
    strategy_a,
    strategy_b,
    source,
    overlap_days,
    pearson_r,
    avg_pnl_a,
    avg_pnl_b,
    both_lose_days,
    divergent_days,
    -- Classification
    CASE
        WHEN pearson_r IS NULL THEN 'INSUFFICIENT_DATA'
        WHEN pearson_r >= 0.7 THEN 'HIGH'
        WHEN pearson_r >= 0.3 THEN 'MODERATE'
        WHEN pearson_r >= -0.3 THEN 'LOW'
        ELSE 'NEGATIVE'
    END AS correlation_level
FROM corr
ORDER BY source, strategy_a, strategy_b;

-- RLS policy for dashboard read access
GRANT SELECT ON cross_strategy_correlation TO anon;
