-- Add dollar P&L column to blocked_signals
ALTER TABLE blocked_signals ADD COLUMN IF NOT EXISTS cf_pnl_dollar double precision;

-- Recreate gate_effectiveness view with dollar column and COOLDOWN_SUPPRESSED exclusion
DROP VIEW IF EXISTS gate_effectiveness;
CREATE VIEW gate_effectiveness AS
SELECT
    bs.gate_type,
    bs.strategy_id,
    date_trunc('week', bs.signal_date::timestamptz) AS week_start,
    COUNT(*) AS signals_blocked,
    COUNT(bs.cf_pnl_pts) FILTER (WHERE bs.cf_exit_reason IS DISTINCT FROM 'COOLDOWN_SUPPRESSED') AS counterfactuals_computed,
    ROUND(COALESCE(SUM(bs.cf_pnl_pts) FILTER (WHERE bs.cf_exit_reason IS DISTINCT FROM 'COOLDOWN_SUPPRESSED'), 0)::numeric, 2) AS counterfactual_net_pnl,
    ROUND(COALESCE(SUM(bs.cf_pnl_dollar) FILTER (WHERE bs.cf_exit_reason IS DISTINCT FROM 'COOLDOWN_SUPPRESSED'), 0)::numeric, 2) AS counterfactual_net_dollar,
    ROUND(COALESCE(SUM(bs.cf_pnl_pts) FILTER (WHERE bs.cf_pnl_pts < 0 AND bs.cf_exit_reason IS DISTINCT FROM 'COOLDOWN_SUPPRESSED'), 0)::numeric, 2) AS losses_avoided,
    ROUND(COALESCE(SUM(bs.cf_pnl_pts) FILTER (WHERE bs.cf_pnl_pts > 0 AND bs.cf_exit_reason IS DISTINCT FROM 'COOLDOWN_SUPPRESSED'), 0)::numeric, 2) AS gains_missed
FROM blocked_signals bs
GROUP BY bs.gate_type, bs.strategy_id, date_trunc('week', bs.signal_date::timestamptz);

-- Re-apply anon grant (dropped with the view)
GRANT SELECT ON gate_effectiveness TO anon;
