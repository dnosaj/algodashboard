-- Developing daily VPOC observation data
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS dvpoc_price double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS dvpoc_strength double precision;  -- VCR: 0-1
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS dvpoc_stability integer;  -- bars since last shift
