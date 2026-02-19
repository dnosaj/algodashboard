# MYM (Micro Dow) Analysis — Why It's Different & How to Adapt

## Summary

MYM is profitable with the SM+RSI strategy but weaker than MNQ/ES. The baseline v9 config (RSI14 60/40 SM>0 CD6 cross) gives PF 1.559, 54% WR, 28 trades, +$201 on ~25 days of data. ES with the same config gives PF 3.18.

The strategy works on MYM, but requires parameter adjustments to match MNQ/ES performance levels.

---

## Root Causes

### 1. Commission Drag (biggest factor)

| Metric | MYM | MNQ | ES |
|--------|-----|-----|-----|
| $/point | $0.50 | $2.00 | $50.00 |
| Commission/side | $0.52 | $0.52 | $1.25 |
| Round-trip in points | **2.08 pts** | 0.52 pts | 0.05 pts |
| Avg 5-min bar range | 31 pts | 26 pts | 4.9 pts |
| **Commission as % of range** | **6.7%** | 2.0% | 1.0% |

MYM's commission eats 3.3x more of each trade's potential than MNQ and 6.7x more than ES. Every MYM trade starts 2.08 points in the hole. In dollar terms it's only $1.04 — but the strategy thinks in points, not dollars, so commission drag is proportionally much larger.

### 2. Weaker SM Signal Predictiveness

SM flip directional accuracy (next-bar hit rate):
- MYM: SM flip bull → 48% win rate (below random), SM flip bear → 52%
- ES: SM flip bull → 49%, SM flip bear → 45%

The SM indicator is barely predictive on MYM. The Dow's price action may be less driven by the smart/dumb money divergence that PVI/NVI captures. Possible reasons:
- Dow has 30 large-cap stocks vs Nasdaq's 100 (more tech-heavy, more retail flow)
- Dow components are less retail-traded, so the PVI/NVI smart vs dumb money distinction is weaker
- Dow is more institutionally driven — both sides are "smart money"

### 3. Lower Volatility

- MYM avg 5-min return: 0.032%
- MNQ avg 5-min return: 0.050%
- ES avg 5-min return: 0.035%

The Dow moves ~36% less per bar than Nasdaq. Smaller moves + higher relative commission = tighter margin for error.

### 4. SM Episode Behavior Is Actually Similar

Despite weaker predictiveness, SM structural behavior is comparable across instruments:
- MYM: 608 episodes, median 6 bars (30 min), 14% choppy (≤2 bars)
- ES: 643 episodes, median 6 bars (30 min), 15% choppy (≤2 bars)

So the SM flip frequency isn't the problem — it's the quality of the signal, not the timing.

---

## What Works on MYM

### Optimized Configs (from 6,720-combo sweep)

| Approach | Config | Trades | WR% | PF | Net$ |
|----------|--------|--------|-----|-----|------|
| High selectivity | RSI16 70/30 SM>0.20 CD10 zone | 6 | 67% | 11.56 | +$217 |
| Balanced + SL | RSI20 60/40 SM>0.00 CD1 zone SL50 | 46 | 61% | 2.64 | +$651 |
| Fast SM standalone | RSI14 65/35 CD3 cross (SM 15/10/300/150) | 24 | 75% | 3.45 | +$458 |
| Volume + SL | RSI8 60/40 SM>0.05 CD10 zone SL150 | 47 | 55% | 2.13 | +$647 |

### Key Parameter Shifts vs MNQ/ES Defaults

| Parameter | MNQ/ES Default | MYM Optimized | Why |
|-----------|---------------|---------------|-----|
| RSI Length | 14 | 16-20 | Dow trends more slowly, longer RSI smooths noise |
| RSI Levels | 60/40 | 65/35 or 70/30 | Wait for strong conviction only |
| Entry Mode | Cross | Zone | More flexible, captures Dow's gradual momentum |
| SM Threshold | 0.00 | 0.20-0.30 | Filter out weak SM noise (SM is less predictive here) |
| Stop Loss | OFF | 50 pts | Cuts losers that SM flip misses |
| SM Params (standalone) | 25/14/500/255 | 15/10/300/150 | Fast SM dramatically improves MYM results |

### Standalone SM Is Better on MYM

Using the standalone Pine Script with fast SM params (15/10/300/150) instead of AA defaults:

| Config | AA Default SM | Fast SM (15/10/300/150) |
|--------|--------------|------------------------|
| RSI14 60/40 CD6 cross | PF 1.27, 24 trades | **PF 2.30, 31 trades** |
| RSI14 65/35 CD3 cross | PF 1.36, 24 trades | **PF 3.45, 24 trades** |
| RSI14 60/40 CD6 zone | PF 1.53, 45 trades | **PF 2.21, 61 trades** |

Fast SM params nearly double the profit factor on MYM.

---

## MYM vs MNQ vs ES Baseline Comparison

All using AlgoAlpha SM, RSI cross entry, SM flip exit:

| Config | MYM (PF / WR / Trades) | ES (PF / WR / Trades) |
|--------|------------------------|----------------------|
| RSI14 60/40 SM>0.00 CD6 | 1.56 / 54% / 28 | **3.18 / 62% / 29** |
| RSI14 60/40 SM>0.05 CD6 | 1.32 / 57% / 21 | **2.93 / 63% / 24** |
| RSI14 65/35 SM>0.00 CD6 | **2.13 / 61% / 18** | 1.68 / 55% / 22 |
| RSI10 55/45 SM>0.00 CD3 | 1.00 / 44% / 48 | **1.64 / 60% / 57** |
| RSI14 60/40 SM>0.00 CD6 zone | **1.48 / 52% / 50** | 1.47 / 59% / 56 |

Notable: RSI14 65/35 is the one config where MYM (PF 2.13) actually outperforms ES (PF 1.68). Tighter RSI filters help MYM more than ES because they compensate for the weaker SM signal.

---

## MYM Trade Log — RSI14 60/40 SM>0.00 CD6 Cross (28 trades)

Largest winners:
- 02/05 short: +313 pts ($+155, SM_FLIP, 6 bars) — big Dow selloff day
- 01/30 short: +165 pts ($+81, SM_FLIP, 6 bars)
- 02/05 short: +135 pts ($+66, EOD, 7 bars)
- 01/20 short: +119 pts ($+58, SM_FLIP, 3 bars)

Largest losers:
- 01/30 short: -233 pts ($-118, SM_FLIP, 18 bars) — SM flip too late
- 01/20 long: -171 pts ($-87, SM_FLIP, 23 bars) — SM flip too late
- 01/28 long: -83 pts ($-43, SM_FLIP, 14 bars)

Pattern: Losses tend to be SM flip exits that fire too late on MYM. Wins tend to be sharp moves where SM correctly identifies direction. Adding a fixed stop loss (50 pts) would have capped the -233 and -171 losers.

---

## Recommendations

1. **Use standalone Pine Script with fast SM params (15/10/300/150) on MYM** — doubles PF
2. **Widen RSI to 65/35 or 70/30** — only trade strong conviction signals
3. **Enable hard stop loss at 50 pts** — caps the slow SM flip losers
4. **Consider zone entry instead of cross** — more trades, similar quality
5. **Raise SM threshold to 0.10-0.20** — filter out noise
6. **Use YM instead of MYM if possible** — YM has $5/pt so commission is only 0.50 pts round-trip (4x less drag than MYM's 2.08 pts)

---

## Data Notes

- MYM CSV: `CBOT_MINI_MYM1!, 1_b3131.csv` — 25,370 bars, Jan 19 - Feb 13, 2026
- Column layout matches ES (SM Net Index col 9, Net Buy Line col 18, Net Sell Line col 24)
- MYM price level: ~49,000-50,500 (1 point = $0.50)
- MYM mintick: 1 point
- Full test script: `strategies/v10_mym_investigation.py`
