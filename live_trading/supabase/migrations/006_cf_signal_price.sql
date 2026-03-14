-- Add signal_price (bar.open at signal time, matching actual entry logic)
-- and signal_group_id (for correlated V15↔vScalpC deduplication)
ALTER TABLE blocked_signals ADD COLUMN IF NOT EXISTS signal_price double precision;
ALTER TABLE blocked_signals ADD COLUMN IF NOT EXISTS signal_group_id text;
