---
name: Pillar reassessment Mar 15
description: Honest reassessment of 8 pillars — Frontier Agent is the priority, Digest needs fixing or pausing, Observation/Copilot deferred
type: project
---

## Pillar Reassessment — March 15, 2026

**Why:** After building the Digest Agent (22 tools, $0.75/run) and running it for several weeks, Jason questioned the value. The digest summarizes what he can already see on the dashboard, sometimes has errors, and hasn't changed a single trading decision. Meanwhile, the real value has come from manual research sessions (forensics, gate sweeps, IS/OOS validation).

**How to apply:** Prioritize Frontier Agent as the next major build. Fix or pause the Digest. Defer Observation and Copilot agents.

## Revised Priority Order

### Build Next
1. **Frontier Agent** — Automates what we do manually in sessions (hypothesis → backtest → IS/OOS → report). Aligns with learning goals: Ollama, LMStudio, local models, agentic architecture, model routing. Highest value + highest learning.
2. **Fix Digest Agent** — Focus on multi-week pattern detection (not daily summaries). Fix accuracy issues. Consider weekly cadence instead of daily when <5 trades/day. If it can't deliver actionable insights, pause it.
3. **Dashboard improvements (Pillars 1+2)** — Loss day messaging, rolling 30-day context. No AI, pure UI, immediate value.

### Defer
4. **Observation Agent** — Too complex, risk of noise, needs multi-instrument feeds. Build after Frontier works and local model stack is learned.
5. **Copilot Agent** — Only as good as the agents it synthesizes. Build the inputs first.
6. **Sizing Agent** — Pure rules in Python until running 10+ contracts.

## Key Insight
The real value comes from RESEARCH, not summarization. The Frontier Agent automates research. The Digest Agent summarizes. Prioritize accordingly.

## Learning Goals (all served by Frontier Agent)
- Ollama / LMStudio — local model deployment
- Model routing — Claude for complex reasoning, local models for batch classification
- Agentic architecture — autonomous hypothesis → test → evaluate loops
- Agent SDK patterns — tool use, structured output, long-running tasks
