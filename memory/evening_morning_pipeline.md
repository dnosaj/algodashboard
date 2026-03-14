---
name: Evening-to-Morning Intelligence Pipeline
description: Architecture for the nightly agent pipeline — Digest (EOD) → Investigation → Frontier → Strategist (deep reasoning) → Digest (morning). Includes Strategist layer as the only agent authorized to recommend parameter changes. Designed Mar 12, 2026.
type: project
---

# Evening-to-Morning Intelligence Pipeline

Designed Mar 12, 2026. Updated Mar 12 to add Strategist Agent layer.

## Core Principle

**Build for the vision, not the current state.** This system will scale to multiple instruments and many more strategies. The architecture must support that without retrofitting.

## Pipeline Flow

```
  EVENING                 OVERNIGHT                            MORNING
  ═══════                 ═════════                            ═══════
  16:30 ET                ~17:00-06:00 ET                      ~06:30-07:00 ET

  ┌──────────┐   triggers   ┌──────────────────┐
  │  DIGEST   │────────────▶│  INVESTIGATION   │
  │  (EOD)    │             │     AGENT        │
  │           │             │                  │
  │ Observe   │             │ Deep dive flagged│
  │ Detect    │             │ patterns         │
  │ Flag      │             │ Map levels       │
  └─────┬─────┘             └────────┬─────────┘
        │                            │ findings feed
        │                            ▼
        │                   ┌──────────────────┐
        │                   │    FRONTIER       │
        │                   │     AGENT         │
        │                   │                  │
        │                   │ Test hypotheses  │
        │                   │ Run backtests    │
        │                   │ Parameter sweeps │
        │                   └────────┬─────────┘
        │                            │
        │                            │ all outputs feed
        ▼                            ▼
  ┌─────────────────────────────────────────────┐
  │              STRATEGIST AGENT                │
  │         (deepest reasoning model)            │
  │                                              │
  │  Reads ALL agent outputs:                    │
  │  • Digest EOD (patterns, attribution)        │
  │  • Investigation (levels, forensics)         │
  │  • Frontier (backtest results, hypotheses)   │
  │  • Drift / Drawdown / Correlation state      │
  │  • Gate effectiveness aggregate              │
  │  • Portfolio-level risk context               │
  │                                              │
  │  ONLY agent authorized to recommend:          │
  │  • Parameter changes (with IS/OOS evidence)  │
  │  • Strategy additions/removals               │
  │  • Gate threshold adjustments                │
  │  • Position sizing changes                   │
  │  • Risk limit adjustments                    │
  │                                              │
  │  Can also say: "NOT NOW" with reasoning      │
  │  (evidence exists but timing/context wrong)  │
  └──────────────────┬────────────────────────────┘
                     │
                     ▼
           ┌──────────────────┐
           │     DIGEST       │
           │  (morning mode)  │
           │                  │
           │ Present:         │
           │ • Traffic light  │
           │ • Yesterday recap│
           │ • Gate status    │
           │ • Strategist     │
           │   recommendations│
           │ • Watchlist      │
           └──────────────────┘

  All agents write structured output to Supabase ──▶ [Supabase]
```

## Agent Roles & Boundaries

| Agent | Role | Recommends changes? | Model tier |
|-------|------|---------------------|------------|
| **Digest (EOD)** | Observe, detect, flag patterns | No — flags anomalies, defers to specialists | Sonnet |
| **Investigation** | Deep-dive flagged patterns, map levels | No — produces evidence and forensics | Sonnet |
| **Frontier** | Run backtests, validate hypotheses | No — produces IS/OOS results tagged CONFIRMED/PROMISING/REJECTED | Sonnet |
| **Strategist** | Synthesize ALL agent outputs, weigh competing evidence, make recommendations | **YES** — the ONLY agent that recommends changes, with full portfolio context | **Opus** (deepest thinker) |
| **Digest (Morning)** | Present the Strategist's synthesis + operational briefing | No — surfaces Strategist recommendations, doesn't generate them | Sonnet |

### Why the Strategist is the only recommender

Individual agents optimize their own slice. The Frontier might say "this filter passes IS/OOS" without knowing the Investigation found the pattern is regime-specific, while the Digest flagged that drawdown is at 40% of max. Only an agent with the FULL picture can decide:
- "Evidence is strong AND timing is right" → Recommend paper trading
- "Evidence is strong BUT drawdown is elevated" → Shelve for 2 weeks, re-evaluate
- "Two agents found conflicting signals" → Needs more data, flag for manual review
- "Frontier result is PROMISING but only 30 OOS trades" → Insufficient evidence, wait

This prevents the system from overreacting to narrow evidence and aligns with the trading philosophy: **optimize for risk-adjusted returns, not maximum extraction.**

## What Each Agent Produces (Supabase output)

### Digest Agent — EOD Mode (16:30 ET)
- **Writes to**: `digests` table (type='eod')
- **Schema**: trades analyzed, per-trade attribution (entry quality, exit type, SM/RSI at entry), patterns detected (time clustering, streak context, day-of-week), rolling stats vs backtest, gate effectiveness
- **Flags**: Specific patterns for Investigation (e.g., "V15 3 SLs in first hour", "MFE efficiency dropping")

### Investigation Agent (~16:45 ET)
- **Reads**: Digest output (flagged patterns), bar data, prior-day levels
- **Writes to**: `trade_annotations` (forensics per trade) + level maps
- **Output**: Where was price relative to key levels? Fib, VWAP, volume profile, multi-TF trend. Tomorrow's key levels with confluence scoring.
- **Feeds Frontier**: "Losses cluster near fib 61.8%" → Frontier tests a fib proximity filter

### Frontier Agent (Overnight)
- **Reads**: Investigation findings, pending research queue, latest backtest data
- **Writes to**: `research_runs` + `research_results`
- **Output**: Each finding tagged CONFIRMED / PROMISING / REJECTED with IS/OOS results
- **Does NOT recommend** — just produces evidence with statistical rigor

### Strategist Agent (~06:30 ET)
- **Reads**: ALL of the above + drift, drawdown, correlation, gate effectiveness, portfolio context, decision log
- **Writes to**: `strategist_recommendations` (new table)
- **Output**: 7 recommendation types (IMPLEMENT_NOW, PAPER_TRADE, SHELVE, REJECT, MONITOR, ALERT, REDUCE) with:
  - Evidence chain (which agents contributed what)
  - Risk assessment (case against, what could go wrong)
  - Timing judgment (why now, or why not now)
  - Confidence level (HIGH / MEDIUM / LOW with reasoning)
  - Specific action (parameter value, gate threshold, sizing change)
  - Review triggers (success/failure criteria, auto-revert thresholds)
- **Decision framework**: 6 sequential questions — statistical sufficiency, portfolio impact, timing, reversibility, complexity audit, decision log check
- **Guardrails**: Rate limiter (max 2 changes/10 trading days), recency bias filter, complexity ceiling (20 gates), confirmation bias check (case against required), decision log consistency, human approval gates, post-implementation review triggers
- **5 Promoted Features (core to v1, not deferred)**:
  - *Temporal reasoning tools*: Pre-computed drawdown age, decision velocity, recommendation half-life, macro event proximity
  - *Recommendation chains*: Multi-step sequential validation (prevents "implement 5 things at once")
  - *Ghost portfolio*: Shadow portfolio tracking counterfactual performance of all recommendations
  - *Counterfactual replay*: Replay proposed gate configs against LIVE trades (stronger than backtest)
  - *Confidence calibration*: Track acceptance rate and outcome accuracy per confidence level, feed back into Strategist context
- **Model**: Opus — deepest reasoning model. ~$0.50-1.50/run, ~$15-45/month
- **Full specification**: `memory/strategist_agent_design.md`

### Digest Agent — Morning Mode (~07:00 ET)
- **Reads**: EOD digest, Strategist recommendations, gate state, VIX, drawdown
- **Writes to**: `digests` table (type='morning')
- **Output**: Operational briefing:
  - Traffic light (GREEN/YELLOW/RED)
  - Yesterday's summary (from EOD digest)
  - Strategist recommendations (presented clearly, not regenerated)
  - Today's setup: key levels, gate status, VIX, drawdown context
  - Watchlist: specific things to monitor

## How This Relates to the Copilot (Phase 6)

The Copilot is the **conversational layer** on top of this pipeline. The pipeline is **push** (runs automatically). The Copilot is **pull** (trader asks questions). When built, the Copilot reads from the same Supabase tables — digests, trade_annotations, research_results, strategist_recommendations. It doesn't recompute; it explains and answers follow-ups.

## Trigger Mechanisms

| Agent | Trigger | Fallback |
|-------|---------|----------|
| Digest (EOD) | Cron at 16:30 ET | Manual: `python -m agents.digest.cli --mode eod` |
| Investigation | Triggered by Digest completion | Manual: `/investigate [trade_id]` |
| Frontier | Triggered by Investigation completion + overnight cron | Manual: `/frontier [hypothesis]` |
| Strategist | Triggered by Frontier completion OR cron at 06:30 ET | Manual: `/strategist` |
| Digest (morning) | Triggered by Strategist completion OR cron at 07:00 ET | Manual: `python -m agents.digest.cli --mode morning` |

## Dependencies

- ✅ trades table with full context
- ✅ blocked_signals with gate enrichment
- ✅ daily_stats, rolling_performance, live_vs_backtest views
- ✅ Cross-strategy correlation view
- ✅ config_id FK population
- ✅ Digest Agent (EOD + Morning) — built Mar 12
- ⬜ `strategist_recommendations` table
- ⬜ `recommendation_chains` table (Strategist v1 — recommendation chains)
- ⬜ `ghost_portfolio_entries` + `ghost_portfolio_results` tables (Strategist v1 — ghost portfolio)
- ⬜ `strategist_calibration` view (Strategist v1 — confidence calibration)
- ⬜ Decision log backfill script (markdown → `decisions` table)
- ⬜ Investigation Agent
- ⬜ Frontier Agent
- ⬜ Strategist Agent
- ⬜ Re-backfill to sync session files ↔ Supabase trade counts

## Key Design Principles

1. **Structured output schemas are the contract between agents.** Each agent writes to a known Supabase format. Downstream agents read structured results — they don't need to know HOW upstream agents arrived at their findings.

2. **Only the Strategist recommends.** Individual agents observe, investigate, and test. The Strategist synthesizes and judges. This prevents overreaction to narrow evidence.

3. **Build for the vision.** This architecture supports scaling to many instruments, many strategies, and many more agents without redesign. The Strategist pattern holds whether you have 4 strategies or 40.
