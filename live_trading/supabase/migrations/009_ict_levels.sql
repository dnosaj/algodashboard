-- ICT level proximity tags on trades
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ict_near_levels text[];
CREATE INDEX IF NOT EXISTS idx_trades_ict_levels ON trades USING GIN (ict_near_levels);

-- Weekly levels in gate state snapshots
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS weekly_vpoc double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS weekly_val double precision;
