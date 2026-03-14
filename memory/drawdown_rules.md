# Drawdown Rule Options

Reference for drawdown protection design across the active portfolio (vScalpA + vScalpB + MES v2).

## Current State

- **Per-strategy daily P&L limit** being implemented via `max_strategy_daily_loss` in `engine/config.py`
  - MNQ strategies (vScalpA, vScalpB): $100/day limit
  - MES v2: $200/day limit
- **Global circuit breakers** (already active): max daily loss $500, max 5 consecutive losses
- **V11-specific rules dormant**: Rule 1 (SL->pause) and Rule 2 (rolling 5-day SL count) only trigger on MNQ_V11, which is not in active config

## Rule Options Compared

### 1. P&L-Based Per Strategy (chosen for initial go-live)

Auto-pause strategy when daily realized P&L hits -$X.

- **Pros**: Simple, directly limits dollar risk, easy to understand and tune
- **Cons**: Doesn't distinguish between many small losses vs one big SL hit; a strategy could hit 10 losing trades that each lose $9 and never trigger the $100 limit
- **Current thresholds**: $100 MNQ, $200 MES

### 2. SL-Count Based (V11 style, currently dormant)

Pause after N stop-losses in a day, or after rolling 5-day SL count exceeds threshold.

- **Pros**: Catches "death spiral" patterns where strategy keeps hitting SL regardless of dollar amount
- **Cons**: Doesn't account for dollar magnitude; a TP=5 win and SL=50 loss are very different. A single SL hit on a good day could prematurely pause.

### 3. Combined (Belt and Suspenders)

Both P&L limit AND SL count can independently trigger pause.

- **Pros**: Catches both dollar-risk and pattern-risk scenarios
- **Cons**: More complex, more false pauses possible, harder to reason about interactions
- **Recommended**: Adopt when scaling past 5 contracts

## Recommended Thresholds by Contract Scale

| Scale | P&L Limit (MNQ) | P&L Limit (MES) | SL Count/Day | Rolling 5-Day SL |
|-------|-----------------|-----------------|-------------|------------------|
| 1 contract | $100 | $200 | 3 | 10 |
| 5 contracts | $500 | $1,000 | 3 | 10 |
| 10 contracts | $1,000 | $2,000 | 3 | 8 |
| 20 contracts | $2,000 | $4,000 | 2 | 6 |

As scale increases: tighten SL thresholds (fewer losses tolerated before pause) but scale P&L limits proportionally with contract count.

## Implementation Notes

- P&L limit resets daily at session start (engine restart via daily rotation)
- SL count rules live in `engine/safety_manager.py` — currently wired to V11 config only
- Force Resume button on dashboard overrides any pause (use with caution)
- Global $500 daily loss circuit breaker is the hard backstop regardless of per-strategy rules
