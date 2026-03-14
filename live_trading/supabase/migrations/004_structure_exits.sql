-- Migration 004: Structure-based exit support
--
-- Adds structure_exit_level column to trades table for tracking
-- which swing level triggered a structure exit.

-- Add structure exit level to trades
ALTER TABLE trades
ADD COLUMN IF NOT EXISTS structure_exit_level DOUBLE PRECISION;

-- Index for querying structure exits
CREATE INDEX IF NOT EXISTS idx_trades_structure_exit
ON trades (exit_reason)
WHERE exit_reason = 'STRUCTURE';

-- Comment
COMMENT ON COLUMN trades.structure_exit_level IS 'Pivot swing H/L that triggered a structure exit (NULL for non-structure exits)';
