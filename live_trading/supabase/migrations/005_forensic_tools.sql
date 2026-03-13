-- 005_forensic_tools.sql
-- Add columns for forensic digest tools: VWAP, opening range, per-trade leledc count.
-- All nullable, purely additive. Safe while engine is running.

ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS vwap_close double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS opening_range_high double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS opening_range_low double precision;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS gate_leledc_count smallint;
