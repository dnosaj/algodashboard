-- ============================================================================
-- 004: Digests table — stores AI-generated trading digests (EOD + morning)
-- ============================================================================

CREATE TABLE digests (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    digest_date date NOT NULL,
    digest_type text NOT NULL CHECK (digest_type IN ('eod', 'morning')),

    -- Structured output from the agent (full analysis)
    content jsonb NOT NULL,

    -- Human-readable markdown (rendered from content)
    markdown text NOT NULL,

    -- Agent metadata
    model text NOT NULL DEFAULT 'claude-sonnet',
    tokens_in int,
    tokens_out int,
    cost_usd numeric(8,4),
    duration_sec numeric(8,2),
    tool_calls int DEFAULT 0,

    -- Linked agent outputs (nullable until those agents exist)
    investigation_digest_id uuid,
    frontier_digest_id uuid,

    -- Versioning
    agent_version text NOT NULL DEFAULT '1.0',

    created_at timestamptz DEFAULT now(),
    UNIQUE (digest_date, digest_type)
);

CREATE INDEX idx_digests_date ON digests (digest_date);
CREATE INDEX idx_digests_type ON digests (digest_type);

-- RLS
ALTER TABLE digests ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anon read access" ON digests
    FOR SELECT TO anon USING (true);
CREATE POLICY "Service write access" ON digests
    FOR ALL USING (true) WITH CHECK (true);
