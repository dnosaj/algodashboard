---
name: Frontier Agent — Autoresearch Pattern
description: Maps Karpathy's autoresearch autonomous experiment loop to the Frontier Agent design. Key architectural insight for how the Frontier Agent should work — autonomous hypothesis testing overnight with accept/reject loop. Includes runtime decision (local vs cloud). Mar 12, 2026.
type: project
---

# Frontier Agent — Autoresearch Pattern

Mar 12, 2026. Insight from Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).

## The Autoresearch Pattern

Karpathy's autoresearch is an autonomous AI research loop for ML experiments:

```
while True:
    1. AI agent reads program.md (instructions/constraints)
    2. Agent modifies train.py (the only editable file)
    3. Run experiment (fixed 5-min time budget)
    4. Evaluate result (validation bits-per-byte — single metric, lower is better)
    5. Accept or reject based on improvement
    6. Log results
    7. Repeat → ~12 experiments/hour, ~100 overnight
```

Key design principles:
- **Single editable file** — keeps diffs reviewable, prevents sprawl
- **Fixed time budget** — ensures comparability across experiments
- **Single evaluation metric** — no ambiguity about "better"
- **program.md as instructions** — human writes intent in markdown, agent executes in code
- **Overnight autonomy** — "you wake up to a log of experiments and hopefully a better model"
- **Accept/reject gate** — only improvements persist, everything else is discarded

## Mapping to the Frontier Agent

| Autoresearch | Frontier Agent |
|---|---|
| `program.md` (human instructions) | System prompt: what can be modified, guardrails, evaluation criteria |
| `train.py` (editable file) | Backtest config/parameters (gate thresholds, TP/SL values, filters) |
| `prepare.py` (fixed infrastructure) | `v10_test_common.py`, `run_backtest_v10()`, Databento data |
| 5-min training run | ~30-60s backtest run (12-month 1-min bars) |
| Validation BPB (single metric) | OOS Profit Factor or OOS Sharpe (primary), with WR/MaxDD as constraints |
| Accept/reject gate | IS/OOS walk-forward: accept if OOS PF > 1.1 AND OOS Sharpe > 1.0 AND OOS trades > 30 |
| Experiment log | `research_runs` + `research_results` tables in Supabase |
| ~100 experiments overnight | ~50-100 hypothesis tests overnight (depends on backtest speed) |

## The Loop

```
FRONTIER AGENT LOOP:

    1. READ hypothesis source:
       - Digest EOD flags_for_frontier (e.g., "SLs cluster near VPOC")
       - Open items from decision log
       - Strategist-queued research items
       - Self-generated follow-ups from prior runs

    2. DESIGN experiment:
       - Define parameter to test (e.g., vpoc_buffer = [3, 5, 7, 10, 15])
       - Define which strategy/strategies to test on
       - Define IS/OOS split (first 9 months / last 3 months)

    3. RUN backtest:
       - Call run_backtest_v10() or run_backtest_tp_exit() with modified config
       - Fixed time budget (~30-60s per parameter set)
       - Full 12-month 1-min data

    4. EVALUATE:
       - Primary: OOS PF and OOS Sharpe
       - Constraints: OOS trades >= 30, OOS WR within 10% of IS WR
       - Compare to baseline (current config without the change)
       - Tag result: CONFIRMED / PROMISING / REJECTED

    5. ACCEPT or REJECT:
       - CONFIRMED: OOS PF > IS PF (or within 5%), both > 1.1, sufficient trades
       - PROMISING: OOS improvement but insufficient trades or marginal
       - REJECTED: OOS degradation, IS overfit, or insufficient improvement

    6. LOG to Supabase:
       - research_runs: hypothesis, parameters tested, model, cost
       - research_results: per-parameter IS/OOS metrics, tag, evidence

    7. REPEAT with next hypothesis
```

## What the Frontier Agent Produces

Each run writes to Supabase with structured results:

```json
{
    "hypothesis": "VPOC proximity gate for V15 — block entries within N pts of prior-day VPOC",
    "source": "digest_eod_2026-03-12",
    "strategy": "MNQ_V15",
    "parameters_tested": [3, 5, 7, 10, 15],
    "best_parameter": 7,
    "baseline": {"is_pf": 1.34, "oos_pf": 1.29, "is_sharpe": 2.73},
    "best_result": {"is_pf": 1.41, "oos_pf": 1.35, "is_sharpe": 2.89},
    "tag": "PROMISING",
    "reason": "OOS PF improves 1.29→1.35 (+4.7%) but only 28 OOS trades (need 30+)",
    "trades_blocked_pct": 12,
    "next_step": "Paper trade for 2 weeks to accumulate more trades"
}
```

The Strategist reads these results and decides whether to implement, paper trade, shelve, or reject — with full portfolio context.

## Runtime Decision: Local, Not Cloud

**Critical difference from Digest Agent**: The Frontier Agent CANNOT run on GitHub Actions (cloud).

Why:
- Needs Databento 1-min CSV data (~500MB across instruments)
- Needs the backtesting engine (`v10_test_common.py`, `run_backtest_v10()`)
- Needs numpy/pandas for computation
- Each backtest takes 30-60s with real data — GitHub Actions timeout is fine, but data shipping is not

**Options:**

1. **Self-hosted GitHub Actions runner on Mac** — GitHub Actions workflow triggers, but executes on a runner process on the MacBook. Gets the cron scheduling + secrets management of GH Actions with local file access. Mac must be on (can be asleep — runner wakes on job).

2. **launchd cron on Mac** — Simpler. `launchd` plist runs `python -m agents.frontier.cli` at a set time (e.g., 22:00 ET). No GitHub dependency. Mac must be on.

3. **Dedicated cloud VM** — Upload Databento data to a small VM (e.g., $5/month DigitalOcean). Runs overnight. Most reliable but adds infrastructure.

**Recommended: Option 1 (self-hosted runner) for now.** Keeps the GitHub Actions pattern consistent with the Digest Agent, uses the same secrets management, and the Mac is already on overnight (sleeps but doesn't shut down). If reliability becomes an issue, graduate to Option 3.

## How This Changes the Pipeline

The critic's argument against Investigation was: "the bottleneck is testing, not generating hypotheses." Autoresearch validates this — the value is in the autonomous experiment loop, not in a separate forensic analysis layer.

Revised pipeline:
```
  EVENING              OVERNIGHT                      MORNING
  ═══════              ═════════                      ═══════
  17:00 ET             ~18:00-06:00 ET                07:00 ET

  ┌──────────┐         ┌──────────────────┐
  │  DIGEST   │────────▶│    FRONTIER      │
  │  (EOD)    │  flags  │     AGENT        │
  │           │────────▶│                  │
  │ Observe   │         │ Test hypotheses  │
  │ Detect    │         │ Run backtests    │
  │ Flag      │         │ Accept/reject    │
  └───────────┘         │ ~50-100 tests    │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   STRATEGIST     │
                        │ (Opus, morning)  │
                        │                  │
                        │ Synthesize ALL:  │
                        │ • Digest flags   │
                        │ • Frontier results│
                        │ • Portfolio state │
                        │ • Decision log   │
                        │ RECOMMEND or NOT │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   DIGEST (AM)    │
                        │ Morning briefing │
                        └──────────────────┘
```

Investigation is dropped from the chain. The Digest's enhanced tools (level proximity, near-gate-miss, SL velocity) handle the forensic layer. Frontier picks up the testable hypotheses directly.

## Relationship to Enhanced Digest

The 5 new Digest tools (from `investigation_vs_enhanced_digest.md`) produce the `flags_for_frontier` that feed this loop:

1. `get_level_proximity` → "SLs cluster near VPOC" → Frontier tests VPOC buffer gate
2. `get_near_gate_miss` → "ADR ratio was 0.28, threshold 0.30" → Frontier tests tighter ADR threshold
3. `get_sl_velocity` → "68% of vScalpB SLs hit in <3 bars" → Frontier tests entry delay
4. `get_entry_clustering` → "V15+vScalpC simultaneous entry → worse outcomes" → Frontier tests stagger gate
5. `get_day_summary_extended` → "runners only convert on trend days" → Frontier tests trend gate for vScalpC

## Key Guardrails (from Strategist design)

The Frontier Agent tests. It does NOT implement. Guardrails:
- Never modify live config
- Never recommend — only tag CONFIRMED/PROMISING/REJECTED with evidence
- Always run IS/OOS split (never IS-only)
- Minimum 30 OOS trades for CONFIRMED tag
- Log everything to Supabase for Strategist review
- Rate limit: max 100 experiments per night (cost cap)
- Each experiment must have a clear hypothesis and evaluation criteria BEFORE running

## Cost Estimate

- Sonnet for hypothesis design + result evaluation: ~$0.15-0.30 per experiment
- 50 experiments/night: ~$7.50-15.00
- Monthly (22 trading days): ~$165-330
- This is 10-20x more expensive than the Digest — the cost is in the volume of API calls, not individual run cost
- **Optimization**: Batch multiple parameter sweeps per API call (test 5 values, evaluate all at once)
- **With batching**: ~$3-5/night, ~$66-110/month

## Build Order

1. **Enhanced Digest tools** (prerequisite — 2-3 hours)
2. **Frontier Agent core** — hypothesis → backtest → evaluate → log loop
3. **Self-hosted runner** — GitHub Actions runner on Mac for overnight execution
4. **`flags_for_frontier` in Digest output** — structured hypothesis handoff
5. **Strategist reads Frontier results** — already designed in Strategist spec

---

## Research Queue

Hypotheses queued for Frontier testing. Each entry has a clear test design.

### RQ-1: Prior-day level gate — decompose by level type (Mar 13, 2026)

**Hypothesis**: The MES prior-day level gate (buf=5) blocks profitable breakout
trades near H/L and VAH. A leaner gate using only VPOC+VAL may outperform.

**Evidence**: `sr_prior_day_level_breakdown.py` (Mar 13) found:
- H/L blocks remove +$841 in winners (63.8% WR high, 58.5% WR low)
- VAH blocks remove +$883 in winners (71.0% WR — the worst offender)
- VPOC blocks remove -$144 in losers (48.3% WR — actually helping)
- VAL blocks remove -$173 in losers (45.3% WR — marginally helping)
- VPOC+VA combo has best OOS Sharpe (1.567 vs 1.548 for all-5)

**Test design**:
1. Run MES_V2 backtest with 4 gate variants: VPOC-only, VAL-only, VPOC+VAL, All-5
2. IS/OOS split at midpoint (same as original Round 3)
3. Evaluate: OOS PF, OOS Sharpe, trade count, MaxDD
4. Walk-forward validation: 3-month rolling windows (4 folds)
5. If VPOC+VAL matches or beats All-5: recommend dropping H/L+VAH from gate

**Priority**: 3 (high — gate is actively blocking $500/day winners in paper trading)
**Sample size**: ~200 trades per half (sufficient)
**Strategy**: MES_V2 only
