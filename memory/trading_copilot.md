# Trading Copilot — Intelligence System Architecture

Created Mar 7, 2026. Revised Mar 10, 2026.

This project has two goals:
1. **Trading**: Build systems that keep the trader confident, informed, and scaling intelligently
2. **Learning**: Use this as an educational project to learn agentic AI architecture, edge technologies, model routing, and modern deployment

---

## The Seven Pillars

### Pillar 1: Confidence Dashboard
**Problem**: A -$475 month feels like the system is broken, even when it's within backtest expectations.
**Solution**: Surface rolling stats that contextualize bad days/weeks/months.

Features:
- **Rolling 30-day P&L** alongside daily — a -$200 day next to "+$1,400 last 30 days"
- **Monthly equity curve** — backtest overlay vs live actuals. If live tracks inside the backtest drawdown envelope, the system is working
- **System Health indicator** — green/yellow/red based on whether current drawdown and win rate are within 1-2 sigma of backtest. Green = normal. Yellow = unusual but seen before. Red = outside parameters, investigate
- **Streak context** — "Current: 2 consecutive losses. Backtest worst: 5." Normalizes bad days
- **Win rate tracker** — rolling 20-trade win rate vs backtest expected, per strategy

Implementation: mostly frontend (React dashboard additions) + Supabase queries. Light lift, high impact.

### Pillar 2: Loss Day Messaging
**Problem**: "PAUSED" after a daily limit hit feels like failure. The trader needs context, not silence.
**Solution**: Contextual, factual messaging when limits trigger.

Features:
- **Context line**: "Daily limit reached (-$100). This month: +$840. Last 30 days: +$1,200."
- **Frequency normalization**: "3rd loss day in 22 trading days (14%). Backtest avg: 15%."
- **Digest commitment**: "End-of-day analysis runs at 16:30. Check tomorrow morning."
- **Tone**: Co-pilot giving facts, not patronizing. The emotional brain is loud — make the rational case louder.

Implementation: dashboard UI enhancement + small engine-side context payload when daily limit triggers. Medium lift.

### Pillar 3: Automated Trade Digest (THE BIG ONE)
**Problem**: Losses feel random and uncontrollable. The trader can't see patterns across days/weeks.
**Solution**: AI agent that analyzes every trading day and generates a structured digest.

Features:
- **Daily digest** (runs at 16:30 ET or on demand):
  - Each trade with full context: SM value, RSI, time, VIX, market conditions
  - Entry quality assessment: was the signal within normal parameters?
  - Win/loss attribution: TP hit, SL hit (how fast?), EOD exit, BE_TIME exit
  - Comparison to backtest: "V15 went 1/4 today. Backtest WR for similar VIX/time conditions: 78%"
- **Pattern detection** (compounding over time):
  - Time-of-day clustering: "3 of last 5 V15 SLs were 12:00-13:00"
  - Day-of-week patterns: "vScalpB losing on Tuesdays"
  - Market condition correlation: "MES losses clustering on days with >1% gap"
  - Strategy-specific drift: "V15 win rate dropped from 85% to 72% over last 30 trades"
- **Weekly summary**: Aggregates daily digests into themes and actionable recommendations
- **Recommendations**: Ranked by confidence. "Consider: tighten V15 cutoff to 12:30" or "vScalpB performing above backtest — hold steady"
- **Digest storage**: Supabase `digests` table + markdown files for morning coffee

**This is the highest-value feature** — it makes the system smarter over time AND gives the trader something concrete to review instead of just worrying.

### Pillar 4: Investigation Agent — "Why did we lose? What's the structure?"
**Problem**: After a loss, the trader can see the P&L but not the *why*. Was price at a key level? Did we enter into a fib retracement? Was there a VWAP rejection we should have seen?
**Solution**: An agent that performs deep forensic analysis on trades and the surrounding market structure.

This is distinct from the Digest Agent. The digest is a *reporter* ("here's what happened"). The Investigation Agent is a *detective* ("here's why, and here's what we should watch tomorrow").

Features:
- **Post-trade forensics** (runs after each trade or on demand):
  - Where was price relative to key levels? Prior-day H/L, VWAP, VAH/VAL, round numbers
  - Fibonacci retracement/extension levels from recent swing points — did we enter at a 61.8% retrace into resistance?
  - Volume profile context: were we trading into a high-volume node (support) or low-volume gap?
  - Multi-timeframe context: what was the 4H and daily trend? Did we take a long scalp against a 4H downtrend?
  - Order flow clues: was there a volume climax bar before our entry? Absorption patterns?
- **Level mapping for tomorrow**:
  - "Tomorrow's key levels: PDH 21,450 / PDL 21,280 / VWAP 21,370 / Fib 61.8% 21,320"
  - Flag zones where multiple levels converge (confluence = stronger S/R)
  - Dashboard overlay showing these levels on the price chart (extends existing prior-day level overlay)
- **Real-time analysis engine** (stretch goal):
  - Live market structure scoring: "price approaching fib 78.6% + PDH confluence — high rejection probability"
  - Not for blocking trades (the system handles entries), but for the trader's situational awareness
  - Think of it as a heads-up display layered on top of the existing dashboard

Tools this agent needs: market data queries (Supabase + live bars), technical analysis library (fib, pivot points, VWAP), chart annotation API.

### Pillar 5: Observation Agent — "What's happening RIGHT NOW?"
**Problem**: The rule-based system enters and exits on fixed parameters (TP, SL, BE_TIME). It can't see that price just hit a fib 78.6% into weekly resistance, or that the Russell is rolling over while Gold spikes — context that a skilled discretionary trader would use to manage the position.
**Solution**: A live agent that watches open positions in real-time with broader market awareness than any rule-based exit can have.

This is the only agent that runs DURING the session, not after. It sits beside the trader as a co-pilot.

**What it watches:**
- **Open position context**:
  - Current P&L, time in trade, distance to TP/SL
  - SM value evolution since entry — is conviction strengthening or fading?
  - RSI divergence: price making new highs but RSI declining?
  - Volume pattern: is the move being confirmed by volume or drying up?
- **Key level proximity**:
  - Is price approaching prior-day H/L, VWAP, round numbers, fib levels?
  - Confluence zones: multiple levels stacking = higher rejection probability
  - "Price 3 pts from PDH + fib 61.8% confluence — watch for rejection"
- **Cross-market signals**:
  - Russell (RTY/M2K) rolling over while NQ holds — divergence warning
  - Gold/GC spiking — risk-off signal
  - VIX intraday spike — volatility expansion incoming
  - DXY (dollar) sharp move — correlated pressure on equities
  - Bonds (ZN/ZB) breaking out — rate sensitivity moment
  - "Russell diverging bearish, Gold +1.2%, VIX +8% intraday — risk-off cluster forming"
- **Exhaustion detection**:
  - Price extended beyond Bollinger Bands or Keltner Channels
  - Volume climax bars (huge volume + rejection wick)
  - Consecutive bars in one direction without pullback
  - "NQ up 180 pts from open with declining volume — exhaustion signature"

**What it can suggest (human approval required):**
- **Early exit**: "Consider closing V15 long — price at fib 78.6% + PDH, Russell diverging. Current P&L: +$8"
- **Add to position**: "vScalpB long +$4, price breaking above PDH with volume confirmation, next resistance 50 pts away. Consider adding 1 contract"
- **Tighten stop**: "MES runner approaching weekly resistance. Suggest moving SL to +5 pts (lock in $25)"
- **Hold steady**: "V15 long looks good — SM strengthening, volume confirming, no nearby resistance until 21,500"
- **Do nothing differently**: Most of the time, the observation agent should confirm the system is working as designed

**Key constraints:**
- **NEVER auto-executes.** Surfaces observations + suggestions to dashboard. Trader decides.
- **Confidence scoring**: Each suggestion rated LOW/MEDIUM/HIGH based on how many signals agree
- **Bias awareness**: Explicitly flags when it's seeing contradictory signals ("cross-market says risk-off but SM is strengthening — mixed signal, hold current plan")
- **Learning over time**: Tracks which suggestions the trader acted on and whether they helped. Over months, learns which cross-market patterns actually predict NQ moves vs noise.

**Architecture considerations:**
- Needs real-time market data feeds (not just NQ — Russell, Gold, VIX, DXY, Bonds)
- Runs on every bar (1-min) for open positions, idle otherwise
- Must be FAST — suggestions need to arrive while they're still actionable
- Local model (Ollama) for the per-bar classification ("is this noteworthy?"), Claude API only for the natural-language suggestion when something IS noteworthy
- Dashboard: dedicated panel showing live observations, color-coded by urgency

Tools this agent needs: live multi-instrument data feed, technical analysis toolkit, cross-market correlation engine, Supabase for historical pattern lookup.

### Pillar 6: Frontier Agent — "What's the next edge?"
**Problem**: The current system is optimized, but markets evolve. Today's edge can decay. Nobody is actively looking for improvements.
**Solution**: An autonomous research agent that continuously explores new opportunities, parameter spaces, and strategies.

This is the agent that works *between* trading sessions — overnight, weekends — running experiments and presenting findings for human review.

Features:
- **Parameter optimization sweeps**:
  - Periodically re-sweep SM and RSI parameters on the latest data window
  - Detect parameter drift: "SM(10/12) was optimal 6 months ago, SM(12/14) now shows higher Sharpe on recent 3 months"
  - Walk-forward validation built in — no IS-only results, always IS+OOS split
  - Alert: "RSI(8/60/40) still optimal for V15. No drift detected." (confidence-building even when nothing changes)
- **New filter/indicator exploration**:
  - Test indicators we haven't tried: ATR-based volatility gates, Bollinger Band squeeze, market breadth
  - Cross-reference with academic papers and quant forums for ideas
  - Each exploration produces a structured report: hypothesis → test → IS result → OOS result → recommendation
- **Multi-timeframe analysis**:
  - Daily/Weekly/4H chart pattern detection: is NQ in a channel? At a weekly resistance?
  - Regime classification: trending vs ranging vs volatile (using the data, not curve-fit thresholds)
  - "NQ has been in a 4H uptrend for 12 days. Historical: V15 WR is 89% in sustained uptrends vs 78% in chop."
- **Cross-instrument scouting**:
  - Monitor performance of our SM signal on ES, YM, RTY, NKD
  - "SM(10/12) on RTY shows PF 1.4 over the last 6 months — worth deeper investigation?"
  - Track correlation shifts between instruments
- **Competitive intelligence** (stretch):
  - Monitor public quant research, trading forums, new indicator publications
  - "New paper on order flow imbalance as a short-term predictor — relevant to our SM signal"
- **Output**: Weekly frontier report. Ranked opportunities with effort/impact estimates.
  - Each finding tagged: CONFIRMED (implement), PROMISING (needs more testing), REJECTED (with reasoning)

Tools this agent needs: backtest engine access, data loaders, parameter sweep framework, web search (for research), long-running compute (can run overnight).

**Key constraint**: This agent RECOMMENDS. It never changes the live system. Every finding goes through human review. The trader decides what to test further and what to implement.

### Pillar 7: Intelligent Position Sizing
**Problem**: Starting at 2x, when do you scale to 3x? Based on what? Gut feel leads to scaling at the wrong time.
**Solution**: Data-driven scaling recommendations based on live performance vs backtest expectations.

Features:
- **Scale-up criteria** (ALL must be true):
  - Account balance above threshold for next tier (margin + 2x max DD buffer)
  - Live Sharpe > 1.5 over last 60 trading days
  - Live win rate within 5% of backtest per strategy
  - Not currently in drawdown (equity at or near high-water mark)
  - At least 4 weeks since last scale change
- **Scale-down criteria** (ANY triggers):
  - Drawdown exceeds 15% of account
  - Live win rate drops >10% below backtest for 30+ days
  - System Health indicator is RED for 5+ consecutive days
- **Graduated scaling**: 2x → 3x → 4x → 5x with 2-4 week confirmation at each level
- **Per-strategy**: MES v2 might earn scale-up before vScalpB based on live performance
- **Dashboard notification**: "Account $14,200. MES v2 live PF 1.41 (backtest 1.33). Recommendation: scale MES 1 → 2 contracts. Expected: +$320/mo return, +$390 max DD."
- **Requires human approval**: System recommends, trader decides. Never auto-scales.

### Pillar 8: Copilot Agent — "Talk to me like a trading partner"
**Problem**: The Observation Agent is a HUD — terse, per-bar alerts for in-the-moment decisions. But at 1:00 PM the trader wants to sit down and have a conversation: "How's it going today? What have you noticed? What's the macro picture? What should I be thinking about this afternoon?"
**Solution**: A conversational agent that aggregates output from ALL other agents and talks to the trader like a knowledgeable partner.

This is distinct from every other agent. The Digest is a reporter (end of day). The Investigation Agent is a detective (post-trade). The Observation Agent is a HUD (real-time alerts). The Copilot is a **conversation partner** — on-demand, wide lens, synthesizing everything into a coherent picture.

**What it synthesizes (reads from all other agents):**
- **From Stats**: Today's P&L, rolling metrics, health score, streak context
- **From Observation**: What's been flagged today, which suggestions were acted on, outcomes
- **From Alert**: News calendar, upcoming events, VIX status
- **From Digest history**: Pattern context ("this looks like Tuesday's setup where we lost on the late entry")
- **From Investigation**: Key levels for today, market structure assessment
- **From Frontier**: Overnight research results, pending findings
- **From Sizing**: Account health, any pending recommendations

**Example conversations:**

*Trader at 1:00 PM*: "How's it going today?"
> "Good morning so far. V15 went 2/3, both wins in the first hour before your 13:00 cutoff. vScalpB is 0/1 — the loss was fast, SL hit in 2 bars, SM was barely above the 0.25 threshold. Net +$17 after commissions.
>
> NQ is up 120 pts but volume has been declining since 11:30. Russell diverged bearish at 12:15 and Gold is up 0.8% — the Observation Agent flagged a risk-off cluster forming at 12:20 but no positions were open at the time.
>
> Fed speaker at 2:00 PM. Historically our MNQ strategies lose on Fed speech days — vScalpA averaged -$45 on the last 8 Fed days. MES tends to be fine. V15 is past its 13:00 cutoff so it won't enter anyway. vScalpB and MES are active."

*Trader*: "Should I be worried about that Russell divergence?"
> "Not yet. The divergence has been mild — Russell down 0.3% while NQ up 0.5%. In the last 60 trading days, we've seen 11 similar divergences. 7 resolved without NQ following Russell down. The 4 that did resolve bearishly averaged -80 pts on NQ over 2 hours, which would be within our SL ranges.
>
> The more notable signal is the volume decline. NQ rallying on declining volume after 11:30 has historically been a setup for afternoon mean reversion. If MES gets a long signal this afternoon, the Investigation Agent's level map shows resistance at 21,450 (PDH + round number confluence). Worth watching."

*Trader at EOD*: "Quick debrief before I close up?"
> "Final: +$32 after commissions. V15 2/3, vScalpB 0/1, vScalpC 1/2 (runner hit BE_TIME), MES 1/1.
>
> The Fed speech at 2:00 caused a 40-pt spike then full retrace — classic Fed noise. MES caught the retrace perfectly, +$100 on that trade. The Observation Agent suggested tightening the MES stop during the spike but you held — good call, it would have stopped out +$15 instead of the full +$100.
>
> Frontier Agent finished overnight — tested SM(12/14) on recent 3 months. Slightly better Sharpe but not statistically significant. Full report in the Frontier tab. No action recommended.
>
> Tomorrow: CPI at 8:30 AM. Historically high-impact for our strategies. Prior-day levels are PDH 21,450 / PDL 21,280. VIX closed at 18.2 — below the death zone, all gates clear."

**The Copilot's three roles:**

1. **Synthesizer** — weaves separate agent outputs into a coherent picture. "Here's what Stats, Observation, Investigation, and Frontier are each saying, and here's how it all fits together."

2. **Original thinker** — this is the most valuable role. No single agent has the cross-domain view. The Stats Agent sees numbers. The Observation Agent sees market structure. The Investigation Agent sees levels. The Frontier Agent sees historical patterns. The Copilot sees ALL of them at once and reasons about the intersection. It generates insights that no individual agent — and no human looking at 8 separate dashboards — would produce.

   Example: "The Observation Agent flagged Russell divergence at 12:15. The Investigation Agent's level map shows NQ at fib 78.6% of the weekly swing. VIX ticked up 0.5 in the last hour. And the Frontier Agent's overnight research showed V15 win rate drops to 68% when all three conditions are present simultaneously. None of those are actionable alone. Together, they're telling you this afternoon's long signals are lower quality than usual."

   That insight doesn't exist in any individual agent's output. It emerges from the intersection. This is what makes the Copilot more than a dashboard — it thinks.

3. **Conversation partner** — explains its reasoning, answers follow-ups, adapts depth to the question, remembers what you discussed earlier in the day.

**Key characteristics:**
- **Conversational, not alert-based.** The trader initiates. The Copilot responds with context-appropriate depth.
- **Produces original analysis.** It doesn't just summarize — it reasons across agent boundaries to find patterns, conflicts, and confluences that no single agent can see.
- **Adapts depth to the question.** "How's it going?" gets a summary. "Should I be worried about X?" gets a deep answer with historical context and cross-agent reasoning. "Quick debrief?" gets bullet points.
- **Opinionated but transparent.** It can say "I wouldn't worry about that" but always shows its reasoning. Never hides uncertainty. When cross-agent signals conflict, it says so explicitly.
- **Remembers the conversation.** If you asked about Russell divergence at 1:00 PM, and Russell reverses at 2:30 PM, the EOD debrief references it: "The Russell divergence you asked about resolved — Russell recovered by 2:30, NQ unaffected."
- **Gets smarter over time.** As more trades, digests, investigations, and frontier research accumulate in Supabase, the Copilot's cross-domain reasoning has a deeper pool to draw from. Day 1 it's connecting 3 dots. Month 6 it's connecting 300.

**Architecture:**
- **Model**: Claude (Opus or Sonnet) — this is pure reasoning and synthesis, needs the best language model
- **Framework**: Agent SDK — needs tool access to query all other agents' outputs from Supabase
- **Tools**: `get_today_stats`, `get_observation_log`, `get_alerts`, `get_investigation_levels`, `get_frontier_status`, `get_digest_history`, `get_account_health`
- **Interface**: Chat panel in dashboard OR standalone chat (could even be a Slack bot, iMessage, etc.)
- **Memory**: Conversation context persists within a trading day. Resets at session boundary but can reference prior days via Supabase.
- **Trigger**: On-demand only. The trader talks when they want to talk.

---

## Revised Stack (Mar 10, 2026)

### Design Priorities
1. **Learn edge technologies** over rolling our own
2. **Supabase** already in toolkit — Postgres, realtime, auto-generated API
3. **Vercel** already in toolkit — cron, edge functions, deployment
4. **Model diversity** — Claude API for complex reasoning, local models (Ollama/LMStudio) for cheap batch work
5. **Agent SDK vs LangChain** — learn both on real problems, use each where it fits

### Data Layer — Supabase (Postgres)

```
Tables:
  trades         — every fill with full context (SM, RSI, VIX, time, MFE, MAE, strategy)
  daily_stats    — daily aggregates per strategy
  digests        — AI-generated daily/weekly analyses
  research       — frontier agent findings
  sizing_history — scaling decisions + outcomes
  gate_state     — persisted gate values (replaces gate_seed.json)
```

Supabase gives us:
- **Auto REST API** — agents query trades via `supabase.from('trades').select()` instead of writing endpoints
- **Realtime** — dashboard subscribes to new trades as they insert. No polling.
- **Postgres functions** — rolling P&L, win rates, health scores as SQL functions (Stats Agent becomes mostly SQL)
- **Cron via pg_cron or Vercel** — trigger digest agent at 16:30 ET

### Agent Framework — Learn Both, Use Each Where It Fits

| | **Anthropic Agent SDK** | **LangChain** |
|---|---|---|
| **Strength** | Clean, minimal, purpose-built for Claude. Tool use is first-class. You write Python, not framework DSL | Massive ecosystem, 500+ integrations, model-agnostic. Swap Claude for Ollama with one line |
| **Weakness** | Claude-only. If you want to route to Ollama/LMStudio, you're writing that plumbing yourself | Abstraction tax. Lots of wrappers around simple things. Debugging is harder because you're 3 layers deep |
| **Best for** | High-stakes agents where you want full control (Digest, Investigation) | Experimentation, model routing, when you want to try 5 models on the same task quickly |
| **Learning value** | Understand how agents actually work under the hood — tools, loops, structured output | Understand the patterns the industry uses — chains, memory, retrieval, orchestration |

**Assignment:**
- **Digest Agent + Investigation Agent** → Agent SDK. Critical-path agents. Full control, Claude's best reasoning, no framework overhead.
- **Observation Agent** → Agent SDK for Claude calls + local model for per-bar classification. Latency matters — no framework overhead.
- **Frontier Agent** → LangChain. Benefits from model routing (Claude for complex analysis, Ollama for cheap parameter sweep summaries).
- **Stats Agent** → Neither. Pure SQL/Python. No LLM needed.
- **Sizing Agent** → Start with rules in Python, add Claude reasoning later via whichever framework you prefer by then.

### Model Routing — The Real-World Pattern

```
┌─────────────────────────────────────────────────┐
│                  MODEL ROUTER                     │
│  Routes tasks to the right model by complexity    │
├─────────────────────────────────────────────────┤
│                                                   │
│  Claude Opus/Sonnet (API)                        │
│  ├── Trade digest generation                     │
│  ├── Investigation forensics                     │
│  ├── Frontier research synthesis                 │
│  ├── Observation agent suggestions               │
│  └── Complex reasoning about scaling decisions   │
│                                                   │
│  Local Model — Ollama/LMStudio (free)            │
│  ├── Trade classification (win attribution)      │
│  ├── Simple summarization                        │
│  ├── Structured extraction from bar data         │
│  ├── Parameter sweep result ranking              │
│  ├── Per-bar observation screening ("noteworthy?")│
│  └── Batch processing (100s of trades)           │
│                                                   │
│  No LLM (pure code)                              │
│  ├── Rolling stats, health scores                │
│  ├── Pattern detection (time clustering, streaks)│
│  ├── Gate state computation                      │
│  └── Alert rules (VIX, news, limits)             │
│                                                   │
└─────────────────────────────────────────────────┘
```

Principle: **use the cheapest thing that works.** Claude for reasoning, local models for grunt work, pure code for math.

### Deployment

```
Supabase (hosted)          — Postgres, realtime, auth
Vercel (cron + functions)  — Scheduled agent triggers, API routes
Local Mac (engine)         — Live trading engine (needs low latency to tastytrade)
Ollama (local)             — Cheap model inference for batch tasks + per-bar screening
```

The trading engine stays local (latency matters for order execution). Everything else can be cloud. Vercel cron triggers the Digest Agent at 16:30 ET. It queries Supabase, calls Claude API, writes the digest back to Supabase. Dashboard picks it up via realtime subscription.

### Agent Table (Revised)

| Agent | Role | Trigger | Model | Framework |
|-------|------|---------|-------|-----------|
| **Stats** | Rolling metrics, health scores | Every trade / hourly | None — pure SQL/Python | Supabase functions |
| **Digest** | Daily trade analysis, pattern detection | 16:30 ET daily / on loss day | Claude | Agent SDK |
| **Investigation** | Post-trade forensics, market structure | After each trade / on demand | Claude | Agent SDK |
| **Observation** | Live position monitoring, cross-market | Every bar (open positions) | Ollama (screening) + Claude (suggestions) | Agent SDK |
| **Copilot** | Conversational synthesis, on-demand briefings | Trader-initiated | Claude (Opus/Sonnet) | Agent SDK |
| **Frontier** | Parameter sweeps, new indicators, scouting | Overnight / weekends | Claude + Ollama (batch) | LangChain |
| **Sizing** | Account health, scale recommendations | Daily / on threshold cross | Rules + Claude for reasoning | TBD |
| **Alert** | News, VIX, system health notifications | Event-driven | None — rule-based | Pure Python |

### Agent Interaction Patterns

Agents aren't isolated — they feed each other:
- **Stats → Digest**: Stats computes the numbers, Digest interprets them in context
- **Stats → Sizing**: Sizing uses health scores and rolling metrics for scale recommendations
- **Digest → Investigation**: Digest flags "V15 3 SLs today" → Investigation does deep dive on those 3 trades
- **Investigation → Frontier**: Investigation finds "losses cluster near fib 61.8%" → Frontier tests a fib-proximity filter
- **Investigation → Observation**: Investigation maps tomorrow's key levels → Observation watches them live
- **Observation → Digest**: Observation logs all suggestions + outcomes → Digest reviews which suggestions were acted on and whether they helped
- **Frontier → Digest**: Frontier completes a sweep → next digest mentions "Frontier found RSI(10) slightly better, report ready"
- **All → Copilot**: Copilot has read access to every agent's output. It synthesizes on demand — the trader's conversational interface to the entire system.
- **Copilot → Investigation**: Trader asks "why did that V15 trade lose?" → Copilot triggers Investigation deep dive if one hasn't run yet
- **All → Alert**: Any agent can emit an alert to the dashboard (context messages, recommendations, warnings)

---

## Revised Build Order

**Phase 1 — Data Foundation (Supabase)**
- [ ] Schema design + tables in Supabase
- [ ] Trade logger: engine writes every fill to Supabase (alongside existing session JSONs)
- [ ] Backfill: load all backtest trades (~1,300+) for pattern baseline
- [ ] SQL functions for rolling stats (replaces Stats Agent)
- [ ] Dashboard wired to Supabase realtime for live trade feed
- **Learning**: Supabase, Postgres, SQL functions, realtime subscriptions
- **Delivers**: Data layer all agents depend on

**Phase 2 — Confidence Dashboard + Loss Messaging (React + Supabase)**
- [ ] Rolling 30-day P&L, system health indicator, streak context (Pillars 1+2)
- [ ] Loss day contextual messaging
- [ ] Mostly frontend + Supabase queries. No AI yet.
- **Learning**: data visualization, dashboard UX for trading context
- **Delivers**: Pillars 1 + 2

**Phase 3 — Digest Agent (Agent SDK + Claude API)**
- [ ] Design digest prompt + structured output schema (JSON → markdown)
- [ ] Build Digest Agent using Anthropic Agent SDK with tool use (Supabase queries, pattern detection)
- [ ] Tools: `query_trades`, `get_daily_stats`, `get_rolling_metrics`, `save_digest`
- [ ] Vercel cron triggers at 16:30 ET
- [ ] Loss day trigger: immediate context message when daily limit hits
- [ ] Dashboard: digest viewer panel
- **Learning**: Agent SDK, Claude API tool use, prompt engineering, structured output
- **Delivers**: Pillar 3

**Phase 4 — Investigation Agent (Agent SDK + Claude API)**
- [ ] Technical analysis toolkit: fib retracement, pivot points, VWAP bands, volume profile
- [ ] Build Investigation Agent: post-trade forensics using market structure context
- [ ] Tools: `get_bars_around_trade`, `compute_fib_levels`, `find_nearby_levels`, `get_volume_profile`
- [ ] Trigger: after each trade (lightweight scan) + on-demand deep dive
- [ ] Dashboard: level overlay enhancements, investigation report viewer
- [ ] Agent-to-agent: Digest flags anomalies → Investigation auto-runs on flagged trades
- **Learning**: technical analysis computation, agent-to-agent communication, tool design
- **Delivers**: Pillar 4

**Phase 5 — Observation Agent (Agent SDK + Ollama + live feeds)**
- [ ] Multi-instrument data feeds: RTY, GC, VIX, DXY, ZN (via tastytrade DXLink or other)
- [ ] Per-bar screening model (Ollama): "is this bar noteworthy for any open position?"
- [ ] Claude API for natural-language suggestions when screening flags something
- [ ] Dashboard: observation panel with live suggestions, confidence scoring, color-coded urgency
- [ ] Suggestion tracking: log what was suggested, what trader did, outcome
- [ ] Set up Ollama locally, learn local model inference patterns
- **Learning**: local model deployment (Ollama/LMStudio), real-time inference, model routing, multi-instrument data
- **Delivers**: Pillar 5

**Phase 6 — Copilot Agent (Agent SDK + Claude)**
- [ ] Design Copilot system prompt: trading partner persona, synthesis rules, depth adaptation
- [ ] Tools: read access to all Supabase tables (trades, stats, digests, observations, alerts, research, sizing)
- [ ] Conversation memory: persists within trading day, references prior days via DB
- [ ] Dashboard: chat panel (or Slack bot, or both)
- [ ] Wire up agent-to-agent triggers: Copilot can invoke Investigation on demand
- [ ] Test conversation quality: morning briefing, midday check-in, EOD debrief scenarios
- **Learning**: conversational AI, context management, multi-source synthesis, chat UX
- **Delivers**: Pillar 8

**Phase 7 — Frontier Agent (LangChain + model routing)**
- [ ] Build Frontier Agent using LangChain for model-agnostic orchestration
- [ ] Route between Claude (complex analysis) and Ollama (batch summarization)
- [ ] Tools: `run_backtest`, `parameter_sweep`, `walk_forward_validate`, `web_search`, `save_research`
- [ ] Overnight/weekend scheduling via Vercel cron
- [ ] Weekly frontier report: ranked findings with CONFIRMED/PROMISING/REJECTED tags
- [ ] Agent-to-agent: Investigation finds pattern → Frontier tests as filter
- **Learning**: LangChain, LangGraph, model routing, long-running tasks, agent coordination
- **Delivers**: Pillar 6

**Phase 8 — Sizing Agent + Account Management**
- [ ] Build Sizing Agent: account health monitoring, scaling recommendations
- [ ] tastytrade API integration for live account balance/margin (no manual input)
- [ ] Dashboard: scaling notifications with approve/defer workflow
- [ ] Historical tracking: every scale decision logged with context + outcome
- **Learning**: financial risk modeling, approval workflows
- **Delivers**: Pillar 7

**Phase 9 — Production Hardening**
- [ ] Monitoring: agent health dashboard, run logs, error tracking, cost tracking (API usage per agent)
- [ ] CI/CD: automated testing of agent outputs, regression detection
- [ ] Consider containerization if deployment complexity warrants it
- **Learning**: observability, production operations

---

## Key Principles

1. **Learn edge technologies over rolling your own.** Supabase over SQLite. Vercel over cron scripts. Agent SDK/LangChain over raw HTTP calls.
2. **Every agent must justify its existence.** If a SQL function does the job, don't wrap it in an agent framework.
3. **Human in the loop for all decisions.** Agents recommend, trader decides. No agent changes the live system autonomously. The Observation Agent suggests — it never executes.
4. **Data is the foundation.** Nothing works without clean, queryable trade data. Phase 1 is non-negotiable.
5. **The digest is the killer feature.** It's the thing that actually makes you a better trader over time. Prioritize it.
6. **Agents feed each other.** The system gets smarter when investigation triggers frontier research, observation logs feed the digest, and frontier findings inform everyone. Design for agent-to-agent communication from the start.
7. **Separate speed from depth.** The Observation Agent runs per-bar (fast). The Digest Agent runs daily (thorough). The Investigation Agent goes deep (forensic). The Frontier Agent goes wide (overnight sweeps). Don't try to make one agent do all of them.
8. **Use the cheapest model that works.** Claude for reasoning, Ollama for screening/classification, pure code for math. Model routing is a first-class concern.
9. **Log everything, summarize intelligently.** Raw data is cheap to store. The agents' job is to turn noise into signal — but never throw away the noise.

---

## Open Questions

- **Digest delivery**: Dashboard page? Email? Markdown file for morning coffee? All three?
- **Historical backfill**: Populate DB with all backtest trades for pattern detection baseline? (Probably yes — gives agents 1,300+ trades to learn from day one)
- **Observation Agent data feeds**: tastytrade DXLink can stream Russell, Gold, VIX, DXY, Bonds — but is the latency sufficient? Or do we need a separate feed for cross-market data?
- **Observation Agent bias risk**: An agent that suggests "exit early" or "add to position" could create a discretionary override habit that undermines the systematic approach. Need guardrails: suggestion frequency limits, confidence thresholds, tracking of suggestion quality over time.
- **Local model selection**: Which Ollama models work best for financial classification? Mistral, Llama 3, Phi-3? Need to benchmark on trade classification tasks.
- **Account tracking**: Pull from tastytrade API (`Account.get_balances()`) — no manual input.
- **Frontier compute budget**: Parameter sweeps are CPU-intensive. How much overnight compute is acceptable? Local machine vs Vercel serverless?
- **Agent memory**: Should agents have persistent memory across runs? (e.g., Frontier remembers what it already tested, Observation learns which cross-market patterns matter). Supabase tables for agent state.
- **Cost tracking**: Claude API calls cost money. Track per-agent API usage and set budgets?
- **Investigation depth**: How deep should post-trade forensics go? Quick scan (fib + levels, 10s) vs full analysis (volume profile + multi-TF + order flow, 2-3 min)?
- **Dashboard real estate**: With 8 agent types potentially surfacing information, how do we avoid information overload? Priority-based notification system?
- **Copilot interface**: Chat panel in dashboard? Slack bot? iMessage? Voice? Start with dashboard chat, expand later.
- **Copilot personality**: How opinionated should it be? "I wouldn't worry" vs "data shows X, you decide." Probably start factual, let personality emerge over time.
- **Copilot → other agents**: Should the Copilot be able to trigger any agent on demand ("run a frontier sweep on this idea") or just Investigation? Start narrow, expand as trust builds.
