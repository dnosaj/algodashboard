-- Structure bar logs: per-bar observation data for active runners
-- High volume (~1 row/min per active runner). Used by Observation Agent for structure exit analysis.

CREATE TABLE IF NOT EXISTS structure_bar_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    bar_time TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    instrument TEXT NOT NULL,
    bar_close DOUBLE PRECISION NOT NULL,
    swing_high DOUBLE PRECISION,
    swing_low DOUBLE PRECISION,
    position INTEGER NOT NULL DEFAULT 0,  -- 0=flat, 1=long, -1=short
    entry_price DOUBLE PRECISION,
    runner_profit_pts DOUBLE PRECISION,  -- current unrealized on runner
    distance_to_level_pts DOUBLE PRECISION,  -- how close to exit trigger
    near_miss BOOLEAN DEFAULT FALSE,  -- close but didn't fire
    trade_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Partition-friendly index for date-range + strategy queries
CREATE INDEX idx_structure_bar_logs_date ON structure_bar_logs (trade_date, strategy_id);

-- Dedup index: one row per bar per strategy
CREATE UNIQUE INDEX idx_structure_bar_logs_dedup ON structure_bar_logs (bar_time, strategy_id);

-- Retention: only keep 30 days (observation data is high volume)
-- Cleanup via scheduled function or manual DELETE WHERE trade_date < NOW() - INTERVAL '30 days'
