---
name: Investigation Agent — Design Debate & Recommended Path
description: Three-perspective review of whether to build Investigation Agent as separate agent or enhance the Digest. Includes critic's argument against, domain expert's pre-compute approach, architect's full spec, and recommended path forward. Mar 12, 2026.
type: project
---

# Investigation Agent — Design Debate & Recommended Path

Mar 12, 2026. Three agents reviewed the Investigation Agent design in parallel: an architect, a trading domain expert, and a critic. This document captures the debate and the recommended path.

## The Question

Should we build the Investigation Agent as a separate agent in the pipeline (Digest → Investigation → Frontier → Strategist → Morning), or enhance the Digest Agent with more tools?

## Three Perspectives

### Architect — "Build the full agent"

Full spec written to `memory/investigation_agent_design.md` (1,056 lines). Key points:

- **14 tools** across 4 categories: data access, computation, level analysis, output
- **`bars_1min` table** in Supabase: engine writes ~780 rows/day at daily reset, 30-day retention
- **Per-trade forensics**: proximity to prior-day H/L/VPOC/VAH/VAL, Fibonacci levels, VWAP/bands, round numbers, multi-TF trend alignment, volume profile position
- **Tomorrow's key levels**: prior-day levels + Fibonacci + developing VWAP + round numbers + confluence zones scored by overlap count
- **New schema**: `bars_1min`, `key_levels`, `investigation_runs` tables, `trade_annotations.metadata` JSONB column
- **Cost**: ~$0.32/run Sonnet, ~$7/month
- **Engine changes**: `export_session_bars()` in DbLogger, two new drain loop item types

### Trading Domain Expert — "Pre-compute, don't ship raw bars"

Agreed on forensic value but proposed a fundamentally simpler data approach:

**Key insight: Have the engine pre-compute a daily context package at EOD, not ship raw bars.**

The engine already tracks everything needed. At session end, write a single structured row per instrument to Supabase with:
- RTH OHLC, VWAP at close, VWAP std dev (already available in engine)
- Opening range H/L (first 30 min of RTH, easy to track)
- Trend score (cumulative directional bars / total bars)
- Direction changes per 30-min block
- Per-trade level context: for each trade, pre-compute distances to all known levels at entry time

**This eliminates the `bars_1min` table entirely.** ~10 extra fields on `gate_state_snapshots` + a JSONB `trade_level_context` array. No new pipeline, no bar data in the cloud, no 780 rows/day.

**What actually matters at 3-7pt MNQ / 6-20pt MES scale:**
1. Prior-day levels (already tracked) — most validated levels in research history
2. Overnight high/low — institutional reference points
3. Opening range (first 30 min) — intraday S/R
4. Session VWAP — mean-reversion reference
5. Round numbers — attract resting orders
6. Day type classification (richer) — trend score, direction changes

**What doesn't matter:**
- Fibonacci (subjective swing points, every level filter has failed or been marginal)
- Weekly/monthly pivots (too far apart for 3-7pt trades)
- Intraday volume profile (noisy, POC moves throughout session)
- Multi-TF trend (already rejected in research — "regime detection doesn't work")
- Order flow / tape / sentiment (no data access)

**Three focused capabilities if built:**
1. Trade forensics: distance to known levels for each SL, classify SL speed, flag proximity
2. Pattern aggregation: do SLs cluster near specific level types across 5-10 days?
3. Level map: tomorrow's reference levels with confluence scoring

### Critic — "Don't build it. Enhance the Digest instead."

**The strongest argument.** Key points:

1. **The Digest already does 70% of what Investigation would do.** Entry prices, exit prices, prior-day levels, regime classification, gate values, drift, streaks, blocked signals — all already in Supabase and queried by the Digest.

2. **The remaining 30% is either theater or 3-5 tools.** Level proximity is a computation, not an investigation — it's `entry_price - vpoc`. That's one SQL query, not a separate agent.

3. **The bottleneck is testing, not generating hypotheses.** MEMORY.md already has a long Open Items list. The Digest already flags patterns. What's missing is the ability to automatically TEST hypotheses — that's Frontier, not Investigation.

4. **Historical hit rate of "key level" filters: ~15-20%.** And those were carefully designed, backtest-validated hypotheses. Investigation-generated hypotheses from single-day observations would have even lower signal quality. Estimate: ~5-10% conversion. ~1-2 useful findings per month.

5. **Cost isn't money, it's engineering time.** The agent costs $7/month. Building and maintaining the bar data pipeline, new tables, new agent code, debugging chain failures — that's the real cost.

**Critic's alternative — 5 new Digest tools (no new agent, no new tables):**
1. `get_level_proximity(date)` — distance from each trade's entry to nearest prior-day level
2. `get_sl_velocity(date)` — classify each SL as rapid/normal/gradual from bars_held
3. `get_entry_clustering(date)` — flag correlated entries (V15+vScalpC same bar)
4. `get_day_summary_extended(date)` — session OHLC, VWAP, body%, IB range (needs ~5 extra fields from engine)
5. `get_near_gate_miss(date)` — trades that entered just outside a gate threshold (ADR ratio 0.28 vs 0.30 threshold)

These use existing Supabase data. No bar pipeline. ~2-3 hours to implement.

## Recommended Path (Synthesis)

**Phase 1: Enhance the Digest (do now)**
- Add the critic's 5 tools to the Digest Agent
- Have the engine write ~5 extra fields to `gate_state_snapshots` at daily reset (opening range H/L, VWAP close, trend score)
- The Digest gets 80% of Investigation's value with 20% of the effort
- ~2-3 hours of work

**Phase 2: Build Frontier (do next)**
- The agent that takes hypotheses and runs backtests to validate them
- This is the REAL missing piece — the testing bottleneck
- The Digest flags patterns → Frontier tests them → Strategist judges them
- Skip Investigation in the chain: Digest → Frontier → Strategist → Morning

**Phase 3: Revisit Investigation (do later, if needed)**
- When the system scales to more instruments/strategies
- When daily trade volume justifies dedicated forensics
- Build as a "deep Digest mode" — same agent, extended tools, heavier analysis prompt
- Only runs on trigger (3+ SLs, portfolio loss > $100), not every day

## Bar Data Decision

**Do NOT store 1-min bars in Supabase.** Instead:
- Extend `gate_state_snapshots` with pre-computed session context (~5-10 extra fields)
- Optionally add a `trade_level_context` JSONB column with per-trade distances
- If raw bars are ever needed (for Frontier backtesting), use a self-hosted GitHub Actions runner with access to local Databento CSVs — not cloud-based bar storage

## What the Digest EOD Prompt Should Flag for Frontier

Add a `flags_for_frontier` field to the structured `content` output of save_digest:
```json
{
  "flags_for_frontier": [
    {
      "hypothesis": "V15 SLs cluster within 5pts of prior-day VAH",
      "evidence": "3 of 7 SLs this week entered within 5pts of VAH",
      "suggested_test": "Test VAH proximity gate with buffer 5-15",
      "priority": "medium",
      "sample_size": 7
    }
  ]
}
```

This feeds directly to Frontier without an intermediary agent.

## Files Referenced

- `memory/investigation_agent_design.md` — Architect's full 1,056-line spec (preserved for future reference)
- `memory/evening_morning_pipeline.md` — Pipeline architecture
- `memory/strategist_agent_design.md` — Strategist spec
- `agents/digest/tools.py` — Current Digest tools (18 tools)
- `agents/digest/prompts.py` — Current Digest prompts
