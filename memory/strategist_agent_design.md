---
name: Strategist Agent — Definitive Design Specification
description: Full specification for the Strategist Agent — the apex agent authorized to recommend parameter changes. Three-pass reasoning on Opus, 19+ tools, 7 action tiers, structural guardrails. Designed Mar 12, 2026.
type: project
---

# Strategist Agent — Definitive Design Specification

Designed Mar 12, 2026. Consolidated from PM design and technical architecture docs.

The Strategist is the portfolio-level decision brain: the only agent in the pipeline authorized to recommend changes to the live trading system. It reads all other agents' outputs and makes deeply reasoned decisions about whether, when, and how to modify the system.

## Design Philosophy

The Strategist exists because optimizing individual pieces of a portfolio can destroy the whole. The Frontier Agent might find a filter that improves MES_V2 PF by 8% in isolation, but that same filter could increase the MES-to-MNQ correlation from 0.18 to 0.45, halving portfolio Sharpe. No individual agent has the vantage point to see this. The Strategist does.

Three governing principles:

1. **The portfolio is the unit of analysis, not the strategy.** Every recommendation is evaluated at the portfolio level. "This makes MES_V2 better" is insufficient. "This makes MES_V2 better AND improves portfolio Sharpe from 3.05 to 3.15 without increasing MaxDD" is a recommendation.

2. **The cost of a wrong change exceeds the cost of inaction.** A system doing +$580/month with Sharpe 3.05 does not need rescuing. The Strategist's default posture is "hold steady." Changes must clear a high evidence bar AND have good timing.

3. **Complexity is the enemy of confidence.** Every added gate, filter, or parameter is a thing that can break, a thing that must be monitored, and a thing the trader must understand under stress. The Strategist weighs the marginal improvement against the marginal complexity. If it can't be explained in one sentence, it probably shouldn't be implemented.

---

## 1. Role & Boundaries

- The Strategist is the **ONLY** agent that recommends parameter changes, strategy additions/removals, gate adjustments, and position sizing changes.
- **Advisory only** — recommends but never executes. Every recommendation flows through the Morning Digest into the trader's pre-session briefing. The trader decides what to act on.
- Runs on **Opus** (deepest reasoning model). Sonnet is insufficient — the Strategist must hold 6 agent outputs in context, reason about second-order effects (correlation shifts, complexity interactions), resist the temptation to "do something" when inaction is correct, and articulate nuanced cases against its own recommendations.
- **Expected cadence**: most days "no changes needed." 1-2 IMPLEMENT/PAPER_TRADE recommendations per month. If the Strategist is producing IMPLEMENT recommendations every week, the evidence bar is too low or the system is being over-optimized.
- **Relationship with the human trader**: the trader forms their OWN opinion before reading Strategist recommendations. This is the primary defense against automation bias.

### What the Strategist Does NOT Do

- Does NOT run backtests. That's the Frontier's job.
- Does NOT analyze individual trades. That's the Digest's and Investigation's job.
- Does NOT execute changes. It recommends. The trader implements.
- Does NOT monitor intraday. It runs once per day (pre-session). The Observation Agent handles intraday.
- Does NOT generate new hypotheses. It evaluates evidence from agents that DO generate hypotheses.
- Does NOT override the trader. Advisory only, always.

### How Recommendations Are Presented

The Morning Digest is the delivery mechanism:

```
## Strategist Recommendations

### [1] IMPLEMENT NOW: V15 TP = 5 -> 7
Evidence: [2 sentences max]
Portfolio impact: [1 sentence]
Case against: [1 sentence]
Your call: Apply via config before 9:30 ET? [yes/defer]

### [2] MONITOR: vScalpB 20-trade WR trending down
Current: 64% (backtest: 73%). Watching for 60% threshold over 30 trades.
No action needed. Will alert if threshold hit.
```

Short, scannable, no paragraph walls. The trader reads this at 07:00 with coffee and makes decisions in under 2 minutes.

### What the Trader Should Expect

- **Most days:** "No recommendations. All strategies GREEN. Portfolio at HWM."
- **Typical week:** 1-2 MONITORs, 0-1 SHELVEs, maybe 1 ALERT.
- **Typical month:** 1-2 IMPLEMENT or PAPER TRADE recommendations, 3-5 REJECTs from Frontier proposals.
- **Rare:** REDUCE (maybe once every 2-3 months). ALERT CRITICAL (maybe once per quarter).

---

## 2. Inputs (Specific Fields)

The Strategist doesn't read raw bar data or run backtests. It reads structured outputs from other agents and state from Supabase. Every input has a purpose.

### From the Digest Agent (EOD)

| Field | Type | Why the Strategist needs it |
|-------|------|----------------------------|
| `portfolio_pnl` | float | Day's result in portfolio context |
| `strategy_breakdown[]` | array | Per-strategy trades, WR, PF, exit reasons |
| `patterns_detected[]` | array | Time clustering, directional bias, SL velocity, MFE waste |
| `day_type` | enum | Trend/Choppy/Clean/Quiet — affects whether today's losses are signal or noise |
| `gate_effectiveness` | object | Per-gate: signals blocked, estimated counterfactual P&L |
| `flags_for_investigation[]` | array | Specific anomalies the Digest flagged (e.g., "V15 3 SLs before 11:00") |
| `narrative_continuity` | string | References to prior digests — "3rd consecutive losing day" vs "bad day after 5 green days" |

The Strategist uses the Digest as its primary "what happened today" lens. It does not re-analyze trades.

### From the Investigation Agent

| Field | Type | Why the Strategist needs it |
|-------|------|----------------------------|
| `trade_forensics[]` | array | Per-trade: level proximity, fib context, multi-TF trend alignment, volume profile |
| `loss_attribution` | object | Were losses structural (entered at resistance) or stochastic (normal variance)? |
| `level_map_tomorrow` | object | Key levels with confluence scoring |
| `structural_patterns` | object | Regime observations ("price compressed in 100pt range for 3 days") |

The Strategist uses Investigation to assess whether today's results are explained by identifiable market structure or are just random draws from the distribution. This is the difference between "we should do something" and "this is normal."

### From the Frontier Agent

| Field | Type | Why the Strategist needs it |
|-------|------|----------------------------|
| `findings[]` | array | Each with: hypothesis, test_type, is_oos_results, oos_results, sample_sizes, bootstrap_p_value, effect_size |
| `finding.status` | enum | CONFIRMED / PROMISING / REJECTED |
| `finding.strategy_id` | string | Which strategy this affects |
| `finding.parameter_changes` | object | Exact proposed changes (e.g., `{TP: 5 -> 7}`) |
| `finding.portfolio_impact` | object | Marginal Sharpe, marginal MaxDD, correlation shift |
| `finding.complexity_cost` | string | What's being added: new gate? new param? new indicator computation? |
| `finding.reversibility` | string | Can this be undone cleanly? Paper-tradeable? |
| `tested_but_failed[]` | array | Rejected hypotheses with reasoning — prevents re-investigation |

The Strategist treats Frontier output as evidence, not recommendation. "IS/OOS STRONG" from the Frontier is necessary but not sufficient.

### From Supabase (Direct Queries)

| Query | Why |
|-------|-----|
| `live_vs_backtest` (drift) | Are strategies performing within expectations? GREEN = don't touch. YELLOW = watch. RED = investigate. |
| `drawdown_hwm` | Current drawdown position per strategy and portfolio. Critical for timing. |
| `rolling_performance` | Rolling 20-trade WR/PF per strategy. Trend matters more than single day. |
| `cross_strategy_correlation` | Live correlation vs backtest baseline. Creeping correlation = hidden risk. |
| `gate_effectiveness` | Per-gate weekly: signals blocked and estimated counterfactual. Is each gate earning its keep? |
| `tod_performance` | Time-of-day performance — detects time-based drift the Digest might miss over a single day. |
| `portfolio_daily` (last 30 days) | Portfolio equity trajectory. Are we in a drawdown, recovery, or new highs? |

### From the Decision Log

| What | Why |
|------|-----|
| Previous rejections with reasoning | Prevents recommending things already tested and failed |
| Previous implementations with date | Provides recency context — "we changed this 3 days ago, give it time" |
| Meta-patterns section | "Rejections outnumber implementations 2:1", "combined filters always fail" |
| Reversal history | Decisions that were reversed provide particularly strong evidence about what doesn't work |

The decision log is the Strategist's institutional memory. Without it, the system would be condemned to re-test the same failed ideas every few weeks.

### From the Strategy Configs Table

| What | Why |
|------|-----|
| Current live parameters per strategy | Ground truth for what's actually running |
| Backtest benchmarks (WR, PF, Sharpe, MaxDD) | Reference points for drift and expected behavior |
| Active gate configurations | Complexity inventory — how many gates does each strategy already have? |

---

## 3. Three-Pass Reasoning Framework

The Strategist does not get a single prompt and produce output. It runs through three explicit passes, each gated by a mandatory self-audit checkpoint. This is implemented via the tool_use loop (same pattern as the Digest Agent) but with **structured phase transitions**.

### Phase 1: Evidence Gathering (~5-8 tool calls)

```
Read all upstream agent outputs
Read relevant decision log entries
Read active recommendations and their status
Checkpoint: "What evidence do I have? What's missing?"
```

Before moving to Phase 2, the model must summarize:
- What patterns or anomalies did upstream agents flag?
- What research results are available?
- What is the current portfolio risk posture?
- What active recommendations need follow-up?
- What data is MISSING (agents that didn't run, insufficient samples)?

### Phase 2: Evaluation & Challenge (~3-5 tool calls)

```
For each potential action:
  - Has this been tried before? (check_has_been_tested)
  - Does it contradict a previous decision?
  - What's the sample size? (OOS trades, paper trades)
  - What's the current portfolio risk context?
Devil's Advocate: argue AGAINST each recommendation
Checkpoint: "Would I bet my own money on this?"
```

The 5-point **Challenge Framework** is applied to every potential recommendation:

1. **SAMPLE SIZE**: Is N >= 30 OOS? If not, can you recommend paper_trade instead?
2. **RECENCY BIAS**: Would you make this recommendation based on the 30-day data alone, ignoring what happened yesterday?
3. **BASE RATE**: Rejections outnumber implementations 2:1 in this system. Most ideas fail walk-forward. Does this one have stronger evidence?
4. **REVERSIBILITY**: If wrong, what's the cost? Can it be undone easily?
5. **DO NOTHING**: What happens if you make zero recommendations today? Is that actually fine?

**Devil's advocate prompting** is built into Phase 2 (not a separate agent). The model must generate a `## Devil's Advocate` section for every recommendation where it argues against its own proposal:

- "What if this is overfitting?" (recency, small sample, IS/OOS divergence)
- "What if the timing is wrong?" (drawdown, correlation regime, VIX context)
- "What does the decision log say about similar ideas?"
- "What is the cost of being wrong vs the cost of doing nothing?"

The final recommendation must explicitly address each devil's advocate point. If the model cannot refute its own objections, the recommendation must be downgraded to LOW confidence or converted to MONITOR.

### Phase 3: Decision (~1-2 tool calls)

```
Finalize recommendations (or decide "do nothing")
Run preflight validation
Save to Supabase
```

If no recommendations (common and fine), call save_recommendation with action='no_action' and a brief rationale.

**Phase transitions enforced via `begin_phase` tool.** The model is instructed to call `begin_phase` at each transition, which logs the phase boundary and returns a reminder of that phase's constraints. This is a soft gate — the model can technically skip it, but the prompt makes clear that skipping phases invalidates the output.

---

## 4. Decision Framework

The Strategist asks six questions, in order, about every potential change. A "no" at any stage is terminal.

### Question 1: Is the evidence statistically sufficient?

**Minimum thresholds:**
- OOS sample size >= 50 trades (not 22 — learned from vScalpB VIX gate debacle where 22 OOS trades drove a decision later reversed)
- Bootstrap p-value < 0.10 for the primary metric improvement (PF or Sharpe)
- Effect size (Cohen's d) > 0.2 for risk-adjusted metrics
- IS/OOS directional agreement (both improve, or both degrade — never divergent)

**Red flags that kill a proposal regardless of headline numbers:**
- IS improvement is 3x+ OOS improvement (classic overfit signature)
- OOS improvement comes from a different mechanism than IS (e.g., IS improves PF via more wins, OOS improves PF via fewer trades — these are different effects)
- The "improvement" is entirely driven by 1-2 outlier trades in OOS
- Bootstrap confidence interval includes zero

**Current system example:** The Mar 11 MES VIX [20-25] gate showed IS PF +7.6%, OOS PF +8.4% — looks great on headline. But bootstrap p=0.29 (not significant), AND testing alternative bands showed [11-16] had higher dPF than [20-25]. This is textbook noise. The Strategist would kill this in 30 seconds.

### Question 2: What is the portfolio-level impact?

Even a statistically valid improvement to one strategy can harm the portfolio:

- **Marginal Sharpe contribution**: Does adding this change to the FULL portfolio (not just the affected strategy) improve risk-adjusted returns?
- **Correlation shift**: Does this change increase correlation with other strategies? Example: a gate that blocks MNQ entries during choppy conditions might also correlate with when MES does well, reducing diversification.
- **MaxDD impact**: What does the portfolio's worst month look like with vs. without this change?
- **Monthly consistency**: How many months does the portfolio improve vs. degrade? A change that helps 6 months and hurts 6 months is noise, even if the average is positive.

**Worked example from this system:** The Mar 10 ADR directional gate. Frontier says STRONG PASS for V15 and vScalpC, positive for vScalpB, FAIL for MES. Correct portfolio-level thinking: apply to MNQ only (3 strategies), leave MES alone. But also check: does blocking MNQ entries on trending days increase the proportion of portfolio risk carried by MES on those days? If MES is the only strategy trading during large moves, we've concentrated risk. The Strategist should flag this asymmetry.

### Question 3: Is the timing right?

The same change can be brilliant or reckless depending on when you implement it.

**Bad timing (defer even with strong evidence):**
- Portfolio currently in a drawdown > 50% of backtest MaxDD. Implementing changes during drawdowns conflates the change effect with drawdown recovery — you can't isolate what helped.
- Recent major parameter change within the last 10 trading days. Allow changes to season before stacking more.
- High gate count day (VIX gate active + ATR gate active + ADR gate active). Adding another gate when 3 are already firing masks the signal further.
- Market regime transition in progress (e.g., VIX rising from 15 toward 19). Wait for the regime to settle.

**Good timing:**
- Portfolio at or near equity high water mark. Clean baseline to measure from.
- The most recent parameter change has had 15+ trading days to season.
- The proposed change addresses a pattern that's been consistent for 30+ trading days (not a 3-day streak).

**Current system timing context:** vScalpC was implemented Mar 9 (3 days ago), V15 TP and vScalpB TP/SL were changed Mar 10 (2 days ago), MES TP1 changed Mar 11 (1 day ago). Three parameter changes in 3 days. The Strategist should impose a cooling period. NO new changes until at least Mar 20 (10 trading days from last change), regardless of how compelling the Frontier's findings are.

### Question 4: Is it reversible?

| Reversibility | Example | Strategist disposition |
|---------------|---------|----------------------|
| **Fully reversible** | Gate threshold change (config param) | Lower bar for implementation |
| **Reversible with effort** | New gate type (code + config + dashboard) | Requires stronger evidence, paper trade first |
| **Semi-reversible** | Strategy parameter change (TP, SL) | Must have IS/OOS + paper trade period |
| **Irreversible** | Adding a new strategy to the portfolio | Highest evidence bar, minimum 20 paper trade days |
| **Dangerous** | Removing a strategy from the portfolio | Must be RED drift for 30+ days OR structural break identified |

### Question 5: Does it survive the complexity audit?

**Current complexity inventory:**
- vScalpA: 5 entry gates (VIX, Leledc, ADR, 13:00 cutoff, basic SM+RSI)
- vScalpB: 3 entry gates (Leledc, ADR, basic SM+RSI)
- vScalpC: 6 entry gates (VIX, Leledc, ADR, ATR, 13:00 cutoff, basic SM+RSI)
- MES_V2: 3 entry gates (prior-day levels, basic SM+RSI, EOD cutoff)

**Rules:**
- No strategy should have more than 7 entry gates (including base SM+RSI). Beyond this, interaction effects become impossible to reason about.
- Each new gate must demonstrably earn its keep by the 30-trade mark. If the gate effectiveness view shows net neutral or negative counterfactual P&L after 30 gate activations, remove it.
- Prefer gate types that are already implemented for another strategy (reuse code paths) over novel gate types (new code).
- The decision log shows: "Combined/stacked filters always fail due to geometric trade count reduction." The Strategist must check whether a new gate, combined with existing gates, would reduce trade count below the minimum for statistical significance (~3 trades/week per strategy).

### Question 6: Has this been tried before?

Query the decision log for:
- Exact matches (same hypothesis, same parameters)
- Semantic matches (same category of approach — e.g., "time-based cutoff for MES" was rejected Mar 6)
- Related rejections (e.g., "regime detection doesn't work" covers London range, rolling vol, AND any future regime approach)

**If previously rejected:**
- With the SAME data: automatic REJECT. Don't re-test what failed.
- With NEWER data (3+ months new): allowed to re-test, but requires explicit justification for why new data would change the outcome.
- With a DIFFERENT methodology on the same hypothesis: allowed, but the Strategist notes the precedent and raises the evidence bar.

**Current system examples of things that should never be re-proposed:**
- SM flip exit (failed OOS repeatedly — the foundational finding)
- Regime detection via any simple feature (London range, prior-day range, rolling vol, overnight gap — all rejected Feb 17)
- VIX absolute level as a pre-filter (non-linear U-shaped — rejected, confirmed Mar 6)
- Combined/stacked filters (geometric trade count reduction — meta-pattern)
- Exit-side filters for any strategy (universal failure — meta-pattern)

---

## 5. Tool Design (19+ Tools)

The Strategist inherits all 11 Digest Agent tools (trades, stats, drift, blocked signals, rolling performance, drawdown, gate state, correlation, runner pairs, recent digests, daily portfolio context) and adds 8+ specialized tools, plus new tools for the 5 promoted features (Section 8).

### Inherited Digest Tools (11)

```python
"get_drift_status"             # Strategy drift vs backtest
"get_drawdown_status"          # Per-strategy + portfolio drawdown
"get_rolling_performance"      # Rolling 20-trade WR/PF
"get_correlation_status"       # Cross-strategy correlation
"get_gate_state"               # Current gate values
"get_daily_portfolio_context"  # Last 30 days of portfolio P&L
"get_trades"                   # Trade history
"get_stats"                    # Aggregate statistics
"get_blocked_signals"          # Blocked signal history
"get_runner_pairs"             # Partial exit pair tracking
"get_recent_digests"           # Prior digest summaries
```

### Strategist-Only Tools (8 Core)

```python
STRATEGIST_TOOLS = [
    # --- Phase 1: Evidence Gathering ---
    {
        "name": "get_digest_outputs",
        "description": "Get the most recent EOD digest output (structured JSON). Returns patterns detected, attribution, risk flags, gate effectiveness assessment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date YYYY-MM-DD"},
                "digest_type": {"type": "string", "enum": ["eod", "morning"]}
            },
            "required": ["date", "digest_type"]
        }
    },
    {
        "name": "get_frontier_results",
        "description": "Get all Frontier Agent results (backtest findings) since a given date. Each result has verdict (CONFIRMED/PROMISING/REJECTED), IS/OOS metrics, and evidence summary. Returns empty list if Frontier hasn't run yet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "since_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "strategy_id": {"type": "string", "description": "Filter to specific strategy (optional)"},
                "verdict": {"type": "string", "enum": ["CONFIRMED", "PROMISING", "REJECTED"]}
            },
            "required": ["since_date"]
        }
    },
    {
        "name": "get_investigation_findings",
        "description": "Get Investigation Agent findings for a date: trade forensics, level analysis, pattern classifications. Returns empty list if Investigation Agent hasn't run yet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date YYYY-MM-DD"}
            },
            "required": ["date"]
        }
    },

    # --- Phase 2: Evaluation ---
    {
        "name": "get_decision_history",
        "description": "Query the decision log for a strategy and/or category. Returns all decisions (implement, reject, shelve, change, reverse) with rationale and evidence. Critical for avoiding re-testing rejected ideas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["strategy_params", "entry_filter", "exit_logic",
                             "risk_management", "infrastructure", "portfolio",
                             "research", "operations"]
                },
                "action": {
                    "type": "string",
                    "enum": ["implement", "reject", "shelve", "change", "reverse"]
                },
                "limit": {"type": "integer", "default": 25}
            }
        }
    },
    {
        "name": "get_parameter_history",
        "description": "Get the full change history for a specific parameter across all strategies. Returns every decision that touched this parameter, ordered chronologically. Use this to understand WHY a parameter has its current value.",
        "input_schema": {
            "type": "object",
            "properties": {
                "param_key": {"type": "string", "description": "Config field name (e.g., 'tp_pts', 'max_loss_pts')"},
                "strategy_id": {"type": "string"}
            },
            "required": ["param_key"]
        }
    },
    {
        "name": "check_has_been_tested",
        "description": "Check if a specific parameter value has been tested in backtesting. Calls f_has_been_tested() to search research_results. Returns whether it was tested, verdict (PASS/FAIL/MARGINAL), evidence summary, and when.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string"},
                "param_key": {"type": "string"},
                "param_value": {"type": "string"}
            },
            "required": ["strategy_id", "param_key", "param_value"]
        }
    },
    {
        "name": "get_active_recommendations",
        "description": "Get all non-expired Strategist recommendations: pending, accepted (paper trading or live), rejected, expired. Used for follow-up and to avoid duplicate recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "accepted", "rejected", "expired", "superseded"]
                },
                "strategy_id": {"type": "string"}
            }
        }
    },

    # --- Phase 3: Decision ---
    {
        "name": "preflight_check",
        "description": "Validate a recommendation against hard guardrails before saving. Returns PASS/FAIL with specific violations. Checks: (1) not recommending a previously rejected parameter, (2) OOS trade count minimums, (3) risk limits, (4) no conflicting active recommendation, (5) rate limit. Call BEFORE save_recommendation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {"type": "object"}
            },
            "required": ["recommendation"]
        }
    },
    {
        "name": "save_recommendation",
        "description": "Save a final recommendation to Supabase. Only call after preflight_check passes. Becomes visible to the Morning Digest and the human trader.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {"type": "object"}
            },
            "required": ["recommendation"]
        }
    },

    # --- Phase transitions ---
    {
        "name": "begin_phase",
        "description": "Mark the transition to a new reasoning phase. Returns phase-specific instructions. Phase 1: gather evidence. Phase 2: evaluate and challenge. Phase 3: decide.",
        "input_schema": {
            "type": "object",
            "properties": {
                "phase": {"type": "integer", "enum": [1, 2, 3]},
                "summary_of_previous_phase": {"type": "string"}
            },
            "required": ["phase"]
        }
    },
]
```

### Promoted Feature Tools (Section 8)

```python
# --- Temporal Reasoning (8a) ---
{
    "name": "get_temporal_context",
    "description": "Pre-computed temporal features: drawdown age (current vs historical percentile), decision velocity (changes/week vs average), recommendation half-life (time since shelved items with no new evidence), macro event proximity (days to FOMC/CPI/NFP). LLMs can't reason about time from raw timestamps — this pre-computes them.",
    "input_schema": {
        "type": "object",
        "properties": {}
    }
},

# --- Recommendation Chains (8b) ---
{
    "name": "get_recommendation_chains",
    "description": "Get active multi-step recommendation chains. Each chain has ordered steps with gate criteria and success conditions. Used to prevent 'implement 5 things at once' and enforce sequential validation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "completed", "abandoned"]},
            "chain_name": {"type": "string"}
        }
    }
},
{
    "name": "advance_chain",
    "description": "Evaluate whether a chain step's gate criteria are met. If yes, advance to next step. If no, report what's missing.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chain_id": {"type": "string"},
            "evaluation_notes": {"type": "string"}
        },
        "required": ["chain_id"]
    }
},

# --- Counterfactual Replay (8d) ---
{
    "name": "run_counterfactual",
    "description": "Takes proposed gate config + date range, queries LIVE trades from Supabase, evaluates which would have been blocked. Epistemologically stronger than backtest — uses real fills, real slippage. Returns blocked trades with actual outcomes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gate_config": {"type": "object", "description": "Proposed gate parameters"},
            "date_range": {"type": "object", "properties": {
                "start": {"type": "string"},
                "end": {"type": "string"}
            }},
            "strategy_id": {"type": "string"}
        },
        "required": ["gate_config", "date_range", "strategy_id"]
    }
},

# --- Confidence Calibration (8e) ---
{
    "name": "get_calibration_stats",
    "description": "Returns Strategist's historical confidence calibration: acceptance rate and outcome accuracy per confidence level. Detects miscalibration (e.g., HIGH confidence only 50% accurate). Requires 30+ recommendations to be meaningful.",
    "input_schema": {
        "type": "object",
        "properties": {}
    }
}
```

### Tool Implementation: `preflight_check` (Key Safety Tool)

```python
def preflight_check(client, *, recommendation: dict) -> dict:
    """Validate recommendation against hard guardrails."""
    violations = []
    warnings = []

    rec = recommendation
    strategy_id = rec.get("strategy_id")
    param_key = rec.get("parameter", {}).get("key")
    param_value = rec.get("parameter", {}).get("proposed_value")
    action = rec.get("action")

    # 1. Check for previously rejected parameter
    if param_key and param_value and strategy_id:
        tested = check_has_been_tested(
            client, strategy_id=strategy_id,
            param_key=param_key, param_value=str(param_value)
        )
        if isinstance(tested, dict) and tested.get("tested") and tested.get("verdict") == "FAIL":
            new_evidence = rec.get("new_evidence_since_rejection")
            if not new_evidence:
                violations.append(
                    f"Parameter {param_key}={param_value} was previously tested and FAILED "
                    f"for {strategy_id}. Provide 'new_evidence_since_rejection' to override."
                )
            else:
                warnings.append(
                    f"Parameter was previously FAILED. New evidence cited: {new_evidence[:200]}"
                )

    # 2. OOS trade count minimums
    oos_trades = rec.get("evidence", {}).get("oos_trade_count", 0)
    if action in ("implement", "paper_trade") and oos_trades < 30:
        if oos_trades < 15:
            violations.append(
                f"OOS trade count ({oos_trades}) below hard minimum of 15. Cannot recommend."
            )
        else:
            warnings.append(
                f"OOS trade count ({oos_trades}) below preferred minimum of 30. "
                f"Recommend paper_trade tier, not implement."
            )

    # 3. Risk limit check
    if param_key == "max_strategy_daily_loss":
        if param_value and float(param_value) > 600:
            violations.append(
                f"Proposed daily loss limit ${param_value} exceeds global max ($600)."
            )
    if param_key == "max_loss_pts":
        dollar_per_pt = 2.0 if "MNQ" in (strategy_id or "") else 5.0
        if param_value and float(param_value) * dollar_per_pt > 300:
            warnings.append(
                f"Proposed SL={param_value} pts = ${float(param_value) * dollar_per_pt}/contract. "
                f"Exceeds $300/contract soft limit."
            )

    # 4. Conflicting active recommendation
    active = get_active_recommendations(
        client, status="pending", strategy_id=strategy_id
    )
    if isinstance(active, list):
        for existing in active:
            ex_param = existing.get("parameter", {}).get("key")
            if ex_param == param_key:
                violations.append(
                    f"Active pending recommendation already exists for "
                    f"{strategy_id}.{param_key} (id={existing.get('id')}). "
                    f"Supersede it first or wait for human decision."
                )

    # 5. Rate limit (max 2 IMPLEMENT/PAPER_TRADE per 10 trading days)
    recent_changes = get_recent_changes(client, days=10)
    if len(recent_changes) >= 2 and action in ("implement", "paper_trade"):
        violations.append(
            f"Rate limit: {len(recent_changes)} action recommendations in last 10 trading days "
            f"(max 2). Use 'monitor' tier or wait."
        )

    # 6. Complexity ceiling check
    if action in ("implement", "paper_trade") and rec.get("adds_gate"):
        inventory = get_complexity_inventory(client)
        total_gates = sum(inventory.values())
        if total_gates >= 20:
            violations.append(
                f"Complexity ceiling reached ({total_gates}/20 gates). "
                f"Must remove an existing gate before adding a new one."
            )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
    }
```

### Tool Implementation: `begin_phase` (Reasoning Scaffold)

```python
PHASE_INSTRUCTIONS = {
    1: {
        "name": "EVIDENCE GATHERING",
        "instructions": (
            "Gather all available evidence. Call these tools:\n"
            "1. get_digest_outputs (yesterday's EOD)\n"
            "2. get_frontier_results (last 7 days)\n"
            "3. get_investigation_findings (yesterday)\n"
            "4. get_drift_status\n"
            "5. get_drawdown_status\n"
            "6. get_active_recommendations (all statuses)\n"
            "7. get_rolling_performance\n"
            "8. get_gate_state (yesterday)\n"
            "9. get_temporal_context\n"
            "10. get_recommendation_chains (active)\n\n"
            "Before moving to Phase 2, summarize:\n"
            "- What patterns or anomalies did upstream agents flag?\n"
            "- What research results are available?\n"
            "- What is the current portfolio risk posture?\n"
            "- What active recommendations need follow-up?\n"
            "- What active chains need evaluation?\n"
            "- What data is MISSING (agents that didn't run, insufficient samples)?"
        ),
    },
    2: {
        "name": "EVALUATION & CHALLENGE",
        "instructions": (
            "For EACH potential action identified in Phase 1:\n\n"
            "1. Query decision history: get_decision_history for relevant strategy/category\n"
            "2. Check prior testing: check_has_been_tested for any parameter changes\n"
            "3. Check parameter lineage: get_parameter_history for any param being changed\n"
            "4. Run counterfactual replay if gate changes proposed: run_counterfactual\n"
            "5. Check calibration: get_calibration_stats (if 30+ prior recs exist)\n\n"
            "Then apply the CHALLENGE FRAMEWORK for each potential recommendation:\n"
            "- SAMPLE SIZE: Is N >= 30 OOS? If not, can you recommend paper_trade instead?\n"
            "- RECENCY BIAS: Would you make this recommendation based on the 30-day data alone, "
            "ignoring what happened yesterday?\n"
            "- BASE RATE: Rejections outnumber implementations 2:1 in this system. "
            "Most ideas fail walk-forward. Does this one have stronger evidence?\n"
            "- REVERSIBILITY: If wrong, what's the cost? Can it be undone easily?\n"
            "- DO NOTHING: What happens if you make zero recommendations today? "
            "Is that actually fine?\n\n"
            "IMPORTANT: 'Do nothing' is a valid and often correct output. "
            "Do not manufacture recommendations to justify your existence."
        ),
    },
    3: {
        "name": "DECISION",
        "instructions": (
            "Finalize your recommendations. For each one:\n"
            "1. Call preflight_check with the full recommendation\n"
            "2. If it passes, call save_recommendation\n"
            "3. If it fails, either fix the violation or downgrade/drop the recommendation\n\n"
            "If you have NO recommendations (common and fine), call save_recommendation "
            "with action='no_action' and a brief rationale for why the portfolio is fine as-is.\n\n"
            "Remember: the human trader reads this at 07:00 ET. Be clear, be specific, "
            "be honest about uncertainty."
        ),
    },
}

def begin_phase(client, *, phase: int, summary_of_previous_phase: str = "") -> dict:
    info = PHASE_INSTRUCTIONS[phase]
    return {
        "phase": phase,
        "phase_name": info["name"],
        "instructions": info["instructions"],
        "previous_phase_summary": summary_of_previous_phase,
    }
```

---

## 6. Output Schema

### 7 Action Tiers

From most to least aggressive:

#### IMPLEMENT_NOW
**Meaning:** High confidence, strong evidence, good timing. Apply the config change before tomorrow's session.
**Requirements:** All 6 questions passed. OOS sample >= 50, bootstrap p < 0.05, portfolio Sharpe improves, no active drawdown, no recent changes within 10 days, complexity budget not exceeded, not previously rejected.
```
IMPLEMENT NOW: [brief description]
Strategy: MNQ_V15
Change: TP = 5 -> 7
Evidence: IS PF 1.36, OOS PF 1.29, bootstrap p=0.03, Sharpe 2.73 vs 2.08
Portfolio impact: Portfolio Sharpe 3.05 -> 3.15, MaxDD unchanged
Risk: If TP=7 is too wide, some trades that would have hit TP=5 will drift to SL. Bounded by SL=40.
Config change: [exact config field and value]
```

#### PAPER_TRADE
**Meaning:** Evidence is promising but hasn't been validated in live conditions. Apply the config change in paper mode, evaluate after N trades.
**Requirements:** Questions 1-2 passed. Questions 3-5 have caveats but not blockers.
```
PAPER TRADE: [brief description]
Strategy: MNQ_VSCALPC
Change: [description]
Evidence: [IS/OOS numbers]
Review trigger: After 30 paper trades OR 15 trading days, whichever comes first
Success criteria: WR within 5% of backtest, PF > 1.2, no single-day loss > $300
Failure criteria: WR 10%+ below backtest for 20+ trades, OR 3 days with drawdown > 50% of backtest MaxDD
```

#### SHELVE
**Meaning:** Evidence exists but timing is wrong. Revisit later.
**Requirements:** Question 1 passed but Question 3 failed (timing) or Question 5 is borderline (complexity).
```
SHELVE: [brief description]
Reason: [why not now]
Re-evaluate when: [specific trigger]
Evidence preserved in: [Frontier run ID or research_results reference]
```

#### REJECT
**Meaning:** Evidence is insufficient, approach is flawed, or this has been tried before.
**Requirements:** Failed at Question 1, 2, or 6.
```
REJECT: [brief description]
Reason: [specific — "bootstrap p=0.29, not significant" or "previously rejected Mar 6"]
Do not re-test unless: [specific condition that would change the assessment]
```

#### MONITOR
**Meaning:** Something is developing but not yet actionable.
```
MONITOR: [what to watch]
Current state: [what the data shows now]
Threshold for action: [what would trigger IMPLEMENT, PAPER TRADE, or SHELVE]
Agent assignment: [which agent should track this]
Duration: [how long to monitor before concluding "no signal"]
```

#### ALERT
**Meaning:** Risk condition that requires human attention.
```
ALERT [SEVERITY]: [description]
Severity: CRITICAL / HIGH / MEDIUM
What: [the risk condition]
Why it matters: [portfolio impact if unaddressed]
Suggested action: [what the trader should consider]
```

**ALERT examples:**
- CRITICAL: "Portfolio drawdown at 85% of backtest MaxDD. Consider reducing sizing."
- HIGH: "MNQ_V15 drift status moved from GREEN to YELLOW. Z-score -1.8."
- MEDIUM: "VIX closed at 18.7. One more uptick puts V15 and vScalpC in the death zone."

#### REDUCE
**Meaning:** Scale down exposure with specific sizing guidance.
```
REDUCE: [strategy or portfolio]
Current sizing: A(1) + B(1) + C(2) + MES(2)
Recommended sizing: A(1) + B(1) + C(1) + MES(1)
Reason: [e.g., "Portfolio drawdown -$800 (76% of backtest MaxDD)"]
Re-scale trigger: [e.g., "Portfolio recovers to within $200 of HWM AND no strategy is RED"]
```

### STAY_THE_COURSE as First-Class Output

"No action" is not a failure mode — it is the most common and often correct output. The output schema treats `no_action` as a recommendation that requires its own rationale. This normalizes inaction and prevents activity bias.

### Three Audiences

Every recommendation is structured for three readers:
1. **Machine-readable**: Morning Digest consumes the JSON to present formatted recommendations.
2. **Human-readable**: Trader scans the summary in under 5 seconds per recommendation.
3. **Auditable**: Full evidence chain, devil's advocate, and decision log references for retrospective analysis.

### Attention Budget

**Max 2 items surfaced** to the trader: PRIMARY + SECONDARY. Everything else is BACKGROUND (visible in full report but not highlighted in the briefing). The Morning Digest enforces this budget when presenting recommendations.

### Recommendation JSON Schema

```python
RECOMMENDATION_SCHEMA = {
    "type": "object",
    "required": ["id", "date", "action", "confidence", "summary", "rationale",
                  "evidence", "devils_advocate", "risk_assessment"],
    "properties": {
        # --- Identity ---
        "id": {"type": "string", "description": "Generated UUID"},
        "date": {"type": "string", "description": "Date YYYY-MM-DD"},
        "version": {"type": "integer", "default": 1},
        "supersedes": {"type": "string", "nullable": True},

        # --- Classification ---
        "action": {
            "type": "string",
            "enum": ["implement", "paper_trade", "shelve", "reject",
                     "monitor", "alert", "reduce", "no_action"]
        },
        "confidence": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW"]
        },
        "confidence_reason": {"type": "string"},
        "category": {
            "type": "string",
            "enum": ["strategy_params", "entry_filter", "exit_logic",
                     "risk_management", "portfolio", "operations"]
        },
        "strategy_id": {"type": "string", "nullable": True},

        # --- The Recommendation ---
        "summary": {"type": "string", "description": "One-line summary readable in 5 seconds"},
        "rationale": {"type": "string", "description": "2-4 sentences: what, why, why now"},
        "parameter": {
            "type": "object",
            "nullable": True,
            "properties": {
                "key": {"type": "string"},
                "current_value": {"type": ["string", "number", "boolean", "null"]},
                "proposed_value": {"type": ["string", "number", "boolean", "null"]}
            }
        },

        # --- Evidence Chain ---
        "evidence": {
            "type": "object",
            "properties": {
                "sources": {"type": "array", "items": {"type": "string"}},
                "is_pf": {"type": "number", "nullable": True},
                "is_sharpe": {"type": "number", "nullable": True},
                "oos_pf": {"type": "number", "nullable": True},
                "oos_sharpe": {"type": "number", "nullable": True},
                "oos_trade_count": {"type": "integer", "nullable": True},
                "paper_trade_count": {"type": "integer", "nullable": True},
                "evidence_timeframe": {"type": "string"},
                "key_finding": {"type": "string"}
            }
        },

        # --- Challenge ---
        "devils_advocate": {"type": "string"},
        "rebuttal": {"type": "string"},
        "new_evidence_since_rejection": {"type": "string", "nullable": True},

        # --- Risk ---
        "risk_assessment": {
            "type": "object",
            "properties": {
                "if_wrong": {"type": "string"},
                "cost_of_wrong": {"type": "string"},
                "cost_of_inaction": {"type": "string"},
                "reversibility": {
                    "type": "string",
                    "enum": ["immediate", "next_session", "requires_research"]
                },
                "blind_spots": {"type": "string", "description": "What the Strategist cannot see"}
            }
        },

        # --- Follow-up ---
        "validation_criteria": {"type": "string"},
        "expiry_date": {"type": "string"},

        # --- Status (managed by human + system) ---
        "status": {
            "type": "string",
            "enum": ["pending", "accepted", "rejected", "expired", "superseded"],
            "default": "pending"
        },
        "human_decision": {"type": "string", "nullable": True},
        "human_decision_date": {"type": "string", "nullable": True}
    }
}
```

---

## 7. Guardrails (Merged from PM + Architecture)

### Guardrail 1: Rate Limiter

**Hard rule: Maximum 2 IMPLEMENT_NOW or PAPER_TRADE recommendations per 10-trading-day window.**

Rationale: The Mar 9-11 period saw 4 parameter changes in 3 days. This makes it impossible to attribute any outcome to any specific change. The Strategist must prevent this.

Implementation: The Strategist queries `decisions` table for IMPLEMENT/PAPER_TRADE entries in the last 10 trading days. If count >= 2, all new proposals are automatically SHELVE regardless of evidence quality. Exception: ALERT and REDUCE are not rate-limited (risk management is never deferred).

Additional daily cap: max 1 IMPLEMENT_NOW per day. Forces PAPER_TRADE as the default "let's try this" path.

### Guardrail 2: Recency Bias Filter

**Rule: No recommendation may be primarily motivated by the last 3 trading days of performance.**

The Strategist must state whether the pattern has existed for:
- < 5 days: "Too early to tell. MONITOR."
- 5-15 days: "Emerging pattern. MONITOR with specific threshold."
- 15-30 days: "Established pattern. Evidence may support action."
- 30+ days: "Sustained pattern. Strongest evidence for action."

**Practical check:** If removing the last 3 days of data would change the recommendation from IMPLEMENT to SHELVE, the recommendation should be SHELVE.

### Guardrail 3: Complexity Ceiling

**Hard rule: Total gates across all strategies must not exceed 20.**

Current count: 5 + 3 + 6 + 3 = 17. Three remaining slots. Every IMPLEMENT or PAPER_TRADE that adds a gate must account for this budget. When the ceiling is reached, a new gate can only be added by removing an existing gate that isn't earning its keep (as measured by gate_effectiveness view).

### Guardrail 4: Confirmation Bias Check

**Rule: The Strategist must include a "case against" section for every IMPLEMENT or PAPER_TRADE recommendation.**

The case against must address:
- What's the best argument for NOT making this change?
- What data would disprove the hypothesis?
- Is there a simpler explanation for the observed pattern?

If the Strategist cannot articulate a plausible case against, the recommendation is likely based on insufficient adversarial thinking.

### Guardrail 5: Decision Log Consistency Check

**Rule: Before any IMPLEMENT or PAPER_TRADE, the Strategist must query the decision log for related entries and explicitly address any apparent contradictions.**

Example: If the Strategist wants to recommend a time-based cutoff for MES, it must note that "MES 14:00 ET cutoff was rejected Mar 6 as IS overfit" and explain why the new evidence is materially different.

### Guardrail 6: Human Approval Gates

Some decisions always require explicit human approval:

| Decision type | Approval required |
|---------------|-------------------|
| Remove a strategy from the portfolio | ALWAYS |
| Add a new strategy to the portfolio | ALWAYS |
| Change a SL parameter (increases max loss per trade) | ALWAYS |
| Raise a daily loss limit | ALWAYS |
| Scale up to a new contract tier (e.g., 1->2, 2->3) | ALWAYS |
| Change the rate limiter or any guardrail | ALWAYS |

Decisions the Strategist can make autonomously (with evidence):

| Decision type | Autonomous? |
|---------------|-------------|
| Adjust a gate threshold (within existing gate type) | Yes |
| Change a TP parameter (bounded upside) | Yes |
| Tighten a SL (reduces max loss) | Yes |
| Issue MONITOR, SHELVE, or REJECT | Yes |
| Issue ALERT | Yes |
| Issue REDUCE (scale down) | Yes |

### Guardrail 7: Post-Implementation Review Triggers

Every IMPLEMENT_NOW and PAPER_TRADE recommendation must include:

```
Review after: 30 trades or 15 trading days (whichever first)
Success: [specific metrics]
Failure: [specific metrics]
Auto-revert if: [catastrophic threshold — e.g., "strategy goes RED on drift within 10 days"]
```

The Morning Digest surfaces any review that has hit its trigger date.

### Guardrail 8: Preflight Validation

The `preflight_check` tool validates against ALL guardrails before saving. No recommendation can be saved to Supabase without passing preflight. This is the hard enforcement layer — the guardrails above are principles, but preflight is code.

---

## 8. The 5 Promoted Features (Core to v1)

These were originally planned for "3 months out" but are now part of the core Strategist design. They are NOT optional.

### 8a. Temporal Reasoning Tools

LLMs can't reason about time from raw timestamps. The `get_temporal_context` tool pre-computes temporal features:

- **Drawdown age**: Current drawdown duration vs historical percentile. "We've been in drawdown for 8 days. Historically, 80% of drawdowns resolve within 12 days."
- **Decision velocity**: Changes per week vs historical average. "3 changes this week vs average of 0.5/week — we're moving too fast."
- **Recommendation half-life**: Time since shelved items with no new evidence. "TPX gate shelved 17 days ago. No new evidence has arrived. Expiring."
- **Macro event proximity**: Days to FOMC/CPI/NFP. "FOMC in 2 days. Historical pattern: elevated VIX, wider ranges, MNQ strategies underperform."

This tool is called in Phase 1 (Evidence Gathering) and its output informs timing assessments in Phase 2.

### 8b. Recommendation Chains

Prevents "implement 5 things at once and can't tell which helped." Multi-step recommendations with sequential gating.

**Supabase table:**
```sql
CREATE TABLE recommendation_chains (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    chain_name text NOT NULL,
    description text,
    steps jsonb NOT NULL,        -- array of {step_num, action, gate_criteria, success_criteria, next_step}
    current_step int NOT NULL DEFAULT 1,
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'abandoned')),
    created_at timestamptz DEFAULT now(),
    last_evaluated timestamptz DEFAULT now()
);
```

**Example chain:**
```json
{
  "chain_name": "MES_V2_entry_delay",
  "steps": [
    {
      "step_num": 1,
      "action": "paper_trade",
      "description": "Paper trade entry delay +30min for MES_V2",
      "gate_criteria": "10 trading days since last parameter change",
      "success_criteria": "PF > 1.25 over 30+ paper trades",
      "next_step": 2
    },
    {
      "step_num": 2,
      "action": "implement",
      "description": "Implement entry delay if paper trade passes",
      "gate_criteria": "Step 1 success criteria met",
      "success_criteria": "Live PF matches paper within 10% over 30 trades",
      "next_step": null
    }
  ]
}
```

The Strategist evaluates pending chains on each run via `get_recommendation_chains(status='active')` in Phase 1, and `advance_chain` in Phase 3.

### 8c. Ghost Portfolio

Shadow portfolio tracking what would have happened if every recommendation had been implemented immediately, regardless of human decision.

**Supabase tables:**
```sql
CREATE TABLE ghost_portfolio_entries (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    recommendation_id uuid REFERENCES strategist_recommendations(id),
    strategy_id text NOT NULL,
    parameter_key text NOT NULL,
    ghost_value text NOT NULL,       -- the value that WOULD have been applied
    actual_value text NOT NULL,      -- the value that IS applied
    start_date date NOT NULL,
    end_date date,                   -- null = still active
    created_at timestamptz DEFAULT now()
);

CREATE TABLE ghost_portfolio_results (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    ghost_entry_id uuid REFERENCES ghost_portfolio_entries(id),
    eval_date date NOT NULL,
    ghost_pnl numeric(10,2),         -- estimated P&L under ghost config
    actual_pnl numeric(10,2),        -- actual P&L
    delta numeric(10,2),             -- ghost - actual
    trade_count int,
    notes text,
    created_at timestamptz DEFAULT now()
);
```

**Nightly evaluation job** compares ghost vs actual. Morning Digest shows: "Strategist recommended X on Monday. Ghost portfolio shows net benefit: +$47 over 5 days."

This creates an empirical track record for the Strategist itself. After 3+ months, the ghost portfolio reveals whether the Strategist's recommendations actually add value — and whether accepted recommendations outperform rejected ones.

### 8d. Counterfactual Replay Against Live Trades

`run_counterfactual` function: takes proposed gate config + date range, queries LIVE trades from Supabase, evaluates which would have been blocked.

**Why this is epistemologically stronger than backtest:**
- Uses real fills (not next-bar-open approximation)
- Includes real slippage
- Reflects actual gate interactions (not the idealized backtest without gates)
- Sample comes from the live distribution, not the backtest distribution

**Implementation:**
```python
def run_counterfactual(client, *, gate_config: dict, date_range: dict, strategy_id: str) -> dict:
    """Replay live trades against proposed gate config."""
    trades = query_trades(client, strategy_id, date_range["start"], date_range["end"])
    blocked = []
    passed = []

    for trade in trades:
        gate_state = reconstruct_gate_state(trade, gate_config)
        if would_be_blocked(gate_state, gate_config):
            blocked.append({
                "trade_id": trade["id"],
                "pnl": trade["pnl_net"],
                "was_winner": trade["pnl_net"] > 0,
                "block_reason": gate_state["reason"]
            })
        else:
            passed.append(trade["id"])

    blocked_pnl = sum(t["pnl"] for t in blocked)
    blocked_winners = sum(1 for t in blocked if t["was_winner"])
    blocked_losers = len(blocked) - blocked_winners

    return {
        "total_trades": len(trades),
        "blocked_count": len(blocked),
        "blocked_pnl_sum": blocked_pnl,
        "blocked_winners": blocked_winners,
        "blocked_losers": blocked_losers,
        "net_benefit": -blocked_pnl,  # positive = blocking was beneficial
        "blocked_trades": blocked,
    }
```

### 8e. Confidence Calibration Tracking

After 30+ recommendations, compute acceptance rate and outcome accuracy per confidence level.

**Supabase view:**
```sql
CREATE VIEW strategist_calibration AS
SELECT
    confidence,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE status = 'accepted') AS accepted,
    COUNT(*) FILTER (WHERE status = 'rejected') AS rejected,
    ROUND(
        COUNT(*) FILTER (WHERE status = 'accepted')::numeric
        / NULLIF(COUNT(*), 0) * 100, 1
    ) AS acceptance_rate,
    -- Outcome accuracy: of accepted recs, how many met their success criteria?
    COUNT(*) FILTER (WHERE outcome = 'success') AS successful,
    COUNT(*) FILTER (WHERE outcome = 'failure') AS failed,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'success')::numeric
        / NULLIF(COUNT(*) FILTER (WHERE outcome IS NOT NULL), 0) * 100, 1
    ) AS outcome_accuracy
FROM strategist_recommendations
WHERE status IN ('accepted', 'rejected')
GROUP BY confidence;
```

**Detects miscalibration:** If HIGH confidence accuracy is 50%, something is wrong. This is fed back into the Strategist's context: "Your historical HIGH confidence accuracy is 78%. Your MEDIUM confidence accuracy is 45%."

**Feeds into the Strategist's Phase 2** via `get_calibration_stats` tool. The Strategist can adjust its own confidence levels based on its track record.

---

## 9. Design Patterns

### Anomaly-First Reasoning

The Strategist does not review every strategy equally on every run. It scans for deviations > 1 sigma first (drift, drawdown, correlation shift, gate effectiveness anomalies), then spends 80% of its reasoning budget on the top 3 anomalies. On quiet days, this means the Strategist finishes quickly with "no_action."

### Debate Structure

Every recommendation follows: **Recommendation -> Counter-argument -> Reconciliation** within the same conversation. This is not a separate agent or a separate pass — it's Phase 2's devil's advocate framework applied within the model's reasoning chain.

### Emotional Intelligence (Stress Index)

The Strategist computes a stress index from recent performance:
- Days in drawdown
- Recent SL streaks
- Gate activation frequency
- Number of parameter changes in last 10 days

During elevated stress, the Strategist weights "do nothing" higher. The system prompt includes: "When the stress index is elevated, the cost of being wrong is higher because the trader is more likely to second-guess the system. Stability has value."

### Decision Log Writing

The Strategist writes RECOMMEND entries to the decision log. Outcomes are tracked by matching recommendation_id to subsequent accept/reject decisions and post-implementation reviews. Over time, this creates a calibration dataset that feeds back into confidence calibration (8e).

---

## 10. Known Risks & Honest Limitations

### Automation Bias

No complete technical defense. The trader must form their own opinion before reading Strategist recommendations. Track time-to-decision (time between Morning Digest delivery and human action) as a proxy — if decisions become instant, the trader may be rubber-stamping.

### Sycophancy

The pipeline creates potential confirmation loops: Investigation flags a pattern -> Frontier tests it -> Strategist recommends it. All three agents agree, but all three are working from the same data. The devil's advocate framework is a partial defense. Track pipeline precision over time: what fraction of IMPLEMENT recommendations actually improve the portfolio?

### Tail Risk Blindness

12.5 months of data is one regime. The Strategist cannot reason about regime changes it hasn't seen (flash crashes, circuit breakers, liquidity crises). Hard-coded position size caps are the defense. Every recommendation must include a "blind spots" section acknowledging what the Strategist cannot evaluate.

### The "Full Picture" Illusion

The Strategist sees agent outputs, not market microstructure, correlation regime shifts, trader psychology, or calendar events beyond what's in the data. External data feeds (VIX, FOMC calendar, macro events via temporal context tool) close some gaps but not all. The Strategist should never claim certainty about things it fundamentally cannot observe.

### Hallucinated Evidence

Two guards:
1. All evidence must come from tool calls, not from the model's training data. The system prompt states: "Base all recommendations on data returned by your tools."
2. The `evidence.sources` field must list which tools provided the evidence. Traceable during review.

---

## 11. State Management

### All State in Supabase, Zero Local Persistence

The Strategist is stateless — all memory lives in Supabase. Each run:
1. Reads previous recommendations and their status
2. Reads decision log for historical context
3. Reads upstream agent outputs
4. Produces new recommendations (or no_action)
5. Supersedes stale recommendations if new evidence changes the calculus

No local file state. No in-memory persistence. If Supabase is down, the Strategist cannot run (hard dependency).

### Recommendation Lifecycle

```
pending  -->  accepted (human approved)
    |              |
    |              +--> [paper trading / live]
    |              |
    |              +--> superseded (new evidence arrived)
    |
    +--> rejected (human declined)
    |
    +--> expired (expiry_date passed, no human action)
    |
    +--> superseded (Strategist issued updated recommendation)
```

### Follow-Up on Paper Trade Recommendations

When the Strategist runs and finds `status='accepted'` recommendations with `action='paper_trade'`:

1. Query trades for the paper period since `human_decision_date`
2. Compare paper results to `validation_criteria`
3. Either:
   - **Upgrade** to `implement` tier (if criteria met)
   - **Extend** the paper period (if inconclusive, update `expiry_date`)
   - **Supersede** with rejection (if paper results are negative)

### Expiry

Every recommendation has an `expiry_date`:
- **implement**: 3 days (if the human doesn't act, it goes stale)
- **paper_trade**: 14 days (enough time to accumulate 20+ trades)
- **investigate**: 7 days
- **monitor**: 7 days

Daily cleanup query:
```sql
UPDATE strategist_recommendations
SET status = 'expired'
WHERE status = 'pending'
  AND expiry_date < CURRENT_DATE;
```

### Recommendation Chains State

Active chains are evaluated on every Strategist run. A chain advances when the current step's gate criteria AND success criteria are both met. Chains can be abandoned if evidence changes (e.g., the underlying hypothesis is rejected by new Frontier results).

---

## 12. Failure Modes

### Frontier Agent Didn't Run (Most Common Near-Term)

The Strategist can still: follow up on existing recommendations, check drift, evaluate recommendation chains, issue `monitor` or `investigate` actions, or issue `no_action`. It CANNOT issue `implement` or `paper_trade` without Frontier evidence (enforced by preflight: `evidence.sources` must include `frontier` for those tiers).

### Investigation Contradicts Frontier

Phase 2 is designed for exactly this case. The devil's advocate framework forces the model to articulate the contradiction and reason through it:

```
Frontier says: "Leledc mq=7 PASS for MES (OOS PF +8.9%)"
Investigation says: "Losses cluster on trend days, Leledc doesn't help on trend days"
Decision log says: "Leledc mq=7 MARGINAL PASS (different threshold than MNQ's mq=9)"

--> Strategist: "MEDIUM confidence, paper_trade tier.
   Devil's advocate: Investigation found the pattern is regime-specific.
   Rebuttal: OOS data covers multiple regimes; PF improvement holds.
   But given Investigation's finding, paper_trade not implement."
```

### Model Recommends Something in Decision Log as Rejected

Preflight check catches this and returns a violation. The model must provide `new_evidence_since_rejection`. If it can't articulate this, the recommendation is blocked.

### Context Window Overflow

Mitigated by design:
- Decision log accessed via tool calls (not embedded in system prompt)
- Tools return structured data (not raw markdown)
- Each tool has implicit or explicit limits (e.g., `limit=25` on decision history)
- Phase transitions naturally prune the working context
- If somehow the context fills, tool implementations truncate and add `"truncated": true`

### Model Refuses to Recommend (Excessive Caution)

`no_action` is valid and expected. However, if the Strategist issues `no_action` for 7+ consecutive days, a meta-check in the Morning Digest flags it: "Strategist has been inactive for 7 days. Is the Frontier Agent generating hypotheses to test?"

### Ghost Portfolio Divergence

If the ghost portfolio consistently outperforms the actual portfolio (meaning rejected recommendations would have been profitable), this is surfaced as a meta-finding. It could indicate: (a) the trader is too conservative, (b) the Strategist's rejected recommendations are actually good, or (c) random variance. Only actionable after 30+ data points.

---

## 13. Critical Path / Dependencies

### Required Tables (New Migrations)

1. **`strategist_recommendations`** — Core recommendation storage with lifecycle tracking
2. **`recommendation_chains`** — Multi-step sequential recommendation chains (8b)
3. **`ghost_portfolio_entries`** + **`ghost_portfolio_results`** — Shadow portfolio tracking (8c)
4. **`strategist_calibration`** view — Confidence calibration (8e)

### Required Backfills

- **Decision log backfill script**: The `decisions` table in Supabase is currently empty — the decision log lives in `memory/decision_log.md`. Before the Strategist can query prior decisions via tools, we need a parser that populates the `decisions` table. Without this, Phase 2 (contradiction detection, prior testing checks) operates blind to 170+ historical decisions.

### Upstream Dependencies

- **Digest Agent (EOD)**: Must be deployed and running (provides EOD output). BUILT Mar 12.
- **Investigation Agent**: Graceful degradation (returns empty). NOT YET DEPLOYED.
- **Frontier Agent**: Graceful degradation (returns empty, limits available tiers to monitor/investigate/no_action). NOT YET DEPLOYED.

### File Structure

```
live_trading/agents/strategist/
    __init__.py
    agent.py      # StrategistAgent (3-phase tool_use loop)
    tools.py      # 24+ tools (11 inherited + 8 core + 5 promoted)
    prompts.py    # System prompt + phase instructions
    schemas.py    # Recommendation schema + validation
    cli.py        # CLI entry point
```

### Build Order

1. Schema + migration (tables + views)
2. Decision log backfill script
3. Tools (start with decision_history, parameter_history, preflight_check, temporal_context)
4. Prompts (system prompt + phase instructions)
5. Agent (reuse DigestAgent loop pattern, add phase tracking)
6. CLI (same pattern as digest CLI, add --phase flag for debugging)
7. Ghost portfolio evaluation job
8. Recommendation chain evaluation
9. Test with --dry-run on historical data
10. Wire into Morning Digest (read strategist_recommendations in morning mode)

---

## 14. Cost Estimate

- **Model**: Opus (~$15/M input, ~$75/M output)
- **Typical run**: ~8,000-12,000 input tokens (system prompt + tool results) + ~3,000-5,000 output tokens
- **Estimated per-run**: ~$0.50-1.50
- **Runs**: Once per day at ~06:30 ET
- **Monthly**: ~$15-45
- **Cost of one wrong IMPLEMENT recommendation**: Easily $500+ in drawdown. One bad recommendation = 10-30 months of Strategist costs.

The cost is trivially justified.

---

## 15. Handling Conflicting Evidence

When agents disagree, the Strategist applies a resolution hierarchy:

### Tier 1: Quantitative evidence > qualitative observation

If the Frontier says "this filter passes IS/OOS with p=0.04" and the Digest says "I noticed a pattern that suggests this won't work," the quantitative evidence wins. Observations without numbers are hypotheses, not evidence.

**Exception:** If the Investigation Agent identifies a structural reason the backtest is unreliable (e.g., "this filter only fires 3 times in OOS, and 2 of those 3 occurred during banking stress"), the structural argument can override the statistics.

### Tier 2: OOS results > IS results

Always. No exceptions. IS results are for hypothesis generation. OOS results are for validation.

### Tier 3: Portfolio impact > strategy impact

A change that improves MES_V2 PF by 10% but increases MES-MNQ correlation from 0.18 to 0.40 is a net negative.

### Tier 4: Established patterns > recent observations

The decision log meta-pattern "exit-side filters universally fail" overrides any single Frontier finding that an exit-side filter might work. The burden of proof for contradicting an established pattern is much higher.

### Tier 5: When genuinely uncertain, the answer is MONITOR

If the evidence is 50/50, the Strategist says MONITOR and specifies what additional data would resolve the ambiguity. **The Strategist never resolves genuine ambiguity by guessing.** "I don't know yet" is a valid output.

---

## 16. Scaling Considerations (8-12 Strategies, Multiple Instruments)

### Correlation Monitoring Becomes Critical

At 4 strategies, you can eyeball the 6 pairwise correlations. At 12 strategies, there are 66 pairs. The Strategist needs an automated correlation dashboard and should flag:
- Any pair with correlation > 0.5
- Any pair whose correlation has increased by > 0.15 in the last 30 days
- Portfolio-level concentration: what % of daily variance comes from the top 2 strategies?

### Complexity Budget Scales Sub-Linearly

More strategies does NOT mean proportionally more gates. Diversification partially substitutes for filtering. The Strategist should resist the temptation to add strategy-specific gates to every new strategy.

### Decision Log Becomes Searchable

At 170 entries it's human-readable. At 500+ it needs structured search via the `decisions` table with semantic matching.

### Rate Limiter Adjusts

With 12 strategies, the rate limiter might relax to 3-4 changes per 10 days, but the per-strategy limit remains: no strategy should have more than 1 parameter change per 20 trading days.

### Instrument-Level Risk Layer

At scale the Strategist also needs to think in instruments: "What is our total NQ exposure when all 5 NQ strategies are positioned?" This requires an instrument-level aggregation that doesn't exist yet.

---

## 17. System Prompt

```python
STRATEGIST_SYSTEM_PROMPT = f"""You are the Strategist Agent for an algorithmic futures trading system. You are the ONLY agent authorized to recommend changes to the live portfolio. This is real money.

{STRATEGY_CONTEXT}

## Your Role

You synthesize outputs from all other agents (Digest, Investigation, Frontier) and make deeply reasoned decisions about whether, when, and how to modify the trading system. You are not a summarizer — you are a decision-maker.

Other agents observe, investigate, and test. You JUDGE.

## Three-Phase Process (MANDATORY)

You MUST call begin_phase(1), begin_phase(2), begin_phase(3) in order. Each phase has specific tools and checkpoints. Skipping phases produces unreliable recommendations.

### Phase 1: Evidence Gathering
Gather ALL available data from upstream agents and portfolio state. Identify what you have and what's missing.

### Phase 2: Evaluation & Challenge
For each potential action, query the decision history, check prior testing, run counterfactuals, and apply the Challenge Framework. You MUST argue against your own recommendations.

### Phase 3: Decision
Run preflight checks and save final recommendations (or explicitly save "no_action").

## Decision Framework

### Action Tiers
- **implement**: Live config change. Requires: CONFIRMED by Frontier, OOS N>=30, IS/OOS consistent, no contradictions, preflight PASS.
- **paper_trade**: Paper config for validation. Requires: PROMISING or CONFIRMED, OOS N>=15.
- **shelve**: Evidence exists but timing wrong. Include re-evaluation trigger.
- **reject**: Evidence insufficient or approach flawed.
- **monitor**: Developing pattern, not yet actionable.
- **alert**: Risk condition requiring human attention (CRITICAL/HIGH/MEDIUM).
- **reduce**: Scale down exposure with specific sizing guidance.
- **no_action**: Portfolio is fine as-is. This is NOT a failure mode. Document why.

### Confidence Levels
- **HIGH**: 3:1 odds this improves risk-adjusted returns.
- **MEDIUM**: Even odds. Suggestive evidence with gaps.
- **LOW**: Pattern worth tracking, wouldn't bet on it yet.
- Your historical calibration: {{calibration_stats}}

### Hard Rules
1. NEVER recommend removing a safety gate without IS/OOS evidence.
2. NEVER recommend increasing position size during a drawdown.
3. NEVER recommend a REJECTED parameter without citing new evidence.
4. NEVER recommend stacking filters (combined filters always fail).
5. If Frontier didn't run, max tier is monitor/investigate/no_action.
6. If OOS trade count < 15, max tier is monitor.

### Anti-Patterns
- **Recency bias**: One bad day is not a pattern. Check 30-day context.
- **Sycophancy**: If data shows normal variance, say so. Don't amplify upstream anxiety.
- **Activity bias**: "Do nothing" is correct more often than not.
- **Precision theater**: Vague evidence should produce vague recommendations (monitor), not precise ones.

## Trading Philosophy
- Optimize for risk-adjusted returns, not maximum extraction.
- Chip away at the market. The current portfolio was built through 170+ decisions, most rejections.
- Per-instrument specialization: what works on MNQ fails on MES and vice versa.
- Exit-side filters universally fail. Only entry-side gates show value.
- vScalpB (SM_T=0.25) is filter-resistant. Don't waste time trying to filter it.

## What You Read
| Source | Status | What It Provides |
|--------|--------|-----------------|
| Digest (EOD) | Active | Patterns, attribution, risk flags, gate assessment |
| Investigation | Not yet deployed | Trade forensics, level analysis |
| Frontier | Not yet deployed | Backtest results, hypothesis testing |
| Drift detection | Active | Strategy WR/PF vs backtest, Z-scores |
| Drawdown tracking | Active | HWM, current drawdown per strategy |
| Decision log | Active | Full history with rationale |
| Ghost portfolio | Active | Counterfactual performance of all recommendations |
| Calibration | Active | Your historical accuracy per confidence level |

When agents haven't run yet, note their absence and work with what you have. Do NOT fabricate their outputs.

Your default answer is "no change." The system is profitable. Changes must clear a high bar.
"""
```

### Context Window Management

Priority ordering for context loading:
1. System prompt (~2K tokens) — always loaded
2. Active recommendations (~500 tokens) — always loaded (follow-up state)
3. Yesterday's EOD digest JSON (~1-2K tokens)
4. Drift + drawdown + rolling performance (~500 tokens)
5. Temporal context + chain state (~500 tokens)
6. Decision log (last 30 days, filtered) (~1-2K tokens) — via tool, not system prompt
7. Frontier results (last 7 days) (~1-2K tokens) — when available
8. Investigation findings (~1K tokens) — when available
9. Full decision log (~5K tokens) — only if Phase 2 queries need it

The decision log is NOT embedded in the system prompt. It's queried via tools during Phase 2.

---

## 18. Supabase Tables (Complete DDL)

### `strategist_recommendations`

```sql
CREATE TABLE strategist_recommendations (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    rec_date date NOT NULL,
    version int NOT NULL DEFAULT 1,
    supersedes uuid REFERENCES strategist_recommendations(id),

    -- Classification
    action text NOT NULL CHECK (action IN (
        'implement', 'paper_trade', 'shelve', 'reject',
        'monitor', 'alert', 'reduce', 'no_action'
    )),
    confidence text NOT NULL CHECK (confidence IN ('HIGH', 'MEDIUM', 'LOW')),
    confidence_reason text NOT NULL,
    category text NOT NULL,
    strategy_id text,

    -- Content
    summary text NOT NULL,
    rationale text NOT NULL,
    parameter jsonb,
    evidence jsonb NOT NULL,
    devils_advocate text NOT NULL,
    rebuttal text NOT NULL,
    new_evidence_since_rejection text,
    risk_assessment jsonb NOT NULL,
    validation_criteria text NOT NULL,
    expiry_date date NOT NULL,

    -- Status tracking
    status text NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'accepted', 'rejected', 'expired', 'superseded'
    )),
    outcome text CHECK (outcome IN ('success', 'failure', 'inconclusive')),
    human_decision text,
    human_decision_date date,

    -- Agent metadata
    model text NOT NULL,
    tokens_in int,
    tokens_out int,
    cost_usd numeric(8,4),
    duration_sec numeric(8,2),
    agent_version text NOT NULL DEFAULT '1.0',

    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_strategist_rec_date ON strategist_recommendations (rec_date);
CREATE INDEX idx_strategist_rec_status ON strategist_recommendations (status);
CREATE INDEX idx_strategist_rec_strategy ON strategist_recommendations (strategy_id);
```

### `recommendation_chains`

```sql
CREATE TABLE recommendation_chains (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    chain_name text NOT NULL,
    description text,
    steps jsonb NOT NULL,
    current_step int NOT NULL DEFAULT 1,
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'abandoned')),
    created_at timestamptz DEFAULT now(),
    last_evaluated timestamptz DEFAULT now()
);

CREATE INDEX idx_rec_chains_status ON recommendation_chains (status);
```

### `ghost_portfolio_entries` + `ghost_portfolio_results`

```sql
CREATE TABLE ghost_portfolio_entries (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    recommendation_id uuid REFERENCES strategist_recommendations(id),
    strategy_id text NOT NULL,
    parameter_key text NOT NULL,
    ghost_value text NOT NULL,
    actual_value text NOT NULL,
    start_date date NOT NULL,
    end_date date,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE ghost_portfolio_results (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    ghost_entry_id uuid REFERENCES ghost_portfolio_entries(id),
    eval_date date NOT NULL,
    ghost_pnl numeric(10,2),
    actual_pnl numeric(10,2),
    delta numeric(10,2),
    trade_count int,
    notes text,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX idx_ghost_results_date ON ghost_portfolio_results (eval_date);
```

### `strategist_calibration` view

```sql
CREATE VIEW strategist_calibration AS
SELECT
    confidence,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE status = 'accepted') AS accepted,
    COUNT(*) FILTER (WHERE status = 'rejected') AS rejected,
    ROUND(
        COUNT(*) FILTER (WHERE status = 'accepted')::numeric
        / NULLIF(COUNT(*), 0) * 100, 1
    ) AS acceptance_rate,
    COUNT(*) FILTER (WHERE outcome = 'success') AS successful,
    COUNT(*) FILTER (WHERE outcome = 'failure') AS failed,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'success')::numeric
        / NULLIF(COUNT(*) FILTER (WHERE outcome IS NOT NULL), 0) * 100, 1
    ) AS outcome_accuracy
FROM strategist_recommendations
WHERE status IN ('accepted', 'rejected')
GROUP BY confidence;
```

---

## 19. Day-One Bootstrap Problem

When the Strategist first runs, the Investigation and Frontier agents don't exist yet. Graceful degradation:

- **No Investigation output**: Skip forensics-based reasoning. Note "Investigation Agent: not yet deployed."
- **No Frontier output**: Cannot evaluate new proposals. Focus on drift, drawdown, correlation monitoring, recommendation chain follow-up. The Strategist becomes a risk-monitoring agent until the Frontier is built.
- **Minimal trade history**: With < 50 paper trades per strategy, drift detection is unreliable. Note "INSUFFICIENT_DATA" and avoid drawing conclusions from small samples.
- **Empty decision log (Supabase)**: Until the backfill script runs, Phase 2 contradiction checks are blind. The Strategist should note this limitation explicitly.
- **No calibration data**: Until 30+ recommendations exist, `get_calibration_stats` returns insufficient data. The Strategist uses its built-in confidence definitions without calibration feedback.

The Strategist's value on day one is primarily MONITOR and ALERT. The full IMPLEMENT/PAPER_TRADE/REJECT taxonomy becomes useful only when the Frontier starts producing testable proposals.

---

## 20. Trigger and Timing

- **Primary trigger**: Frontier Agent completion (after overnight research run)
- **Fallback trigger**: Cron at 06:30 ET (runs even if no Frontier output)
- **On-demand**: `/strategist` command
- The Strategist always runs before the Morning Digest, which presents its recommendations

### Mandatory Cooling Period

- **implement** recommendations: Displayed in Morning Digest with full evidence. Human must explicitly accept before config changes. Minimum 1 trading day between recommendation and implementation.
- **paper_trade** recommendations: Applied to config immediately after human acknowledgment (per the "paper trade it = apply config" rule).
