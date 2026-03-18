---
name: SM 3-Year Regime Analysis
description: SM edge validated across 3 years (Feb 2023 - Mar 2026). Current params optimized for high-vol. Robust cluster at SM_Index=12, EMA=80-120.
type: project
---

# SM 3-Year Regime Analysis — March 17, 2026

## Summary
Swept 576 SM configs across 3 independent years of MNQ data. The SM entry signal is real and persists across volatility regimes — 10.2% of configs produce PF > 1.0 on ALL 3 years. Our current params are not overfit to the dev period, but they are optimized for high-vol conditions.

## Data
| Period | Bars | MNQ Price Range | Avg Daily ATR | Regime |
|--------|------|-----------------|---------------|--------|
| Year 1 (Feb 2023 - Feb 2024) | 351K | 11,712 - 18,118 | 213 pts | LOW |
| Year 2 (Feb 2024 - Feb 2025) | 350K | 17,121 - 22,217 | 279 pts | MEDIUM |
| Year 3 (Feb 2025 - Mar 2026) | 375K | 16,500 - 26,396 | 369 pts | HIGH |

## vScalpA Results (TP=7, SL=40, RSI 8/60/40, CD=20)

**59 of 576 configs pass all 3 years (10.2%)**

### Top 5 Robust Configs (by worst-year PF)
| Rank | SM Config | SM_T | Worst PF | Y1 Net | Y2 Net | Y3 Net | Total |
|------|-----------|------|----------|--------|--------|--------|-------|
| 1 | 14/10/150/120 | 0.25 | 1.091 | +$228 | +$310 | +$689 | +$1,227 |
| 2 | 8/10/250/120 | 0.15 | 1.083 | +$251 | +$285 | +$623 | +$1,158 |
| 3 | 12/14/150/80 | 0.25 | 1.080 | +$306 | +$226 | +$797 | +$1,328 |
| 4 | 12/10/150/80 | 0.25 | 1.076 | +$283 | +$322 | +$618 | +$1,223 |
| 5 | 14/14/150/80 | 0.25 | 1.072 | +$345 | +$192 | +$735 | +$1,272 |

### Current Config: 10/12/200/100, SM_T=0.0
- Y1: PF 0.999, -$5 (barely misses)
- Y2: PF 1.018, +$124
- Y3: PF 1.296, +$2,297
- Total: +$2,415
- **Does NOT pass the 3-year test** — fails Y1 by $5

### Robust Cluster Analysis (what params do passing configs prefer?)
| Parameter | Most common in passing configs | Our current | Action |
|-----------|-------------------------------|-------------|--------|
| SM_Index | **12** (58%), 14 (29%) | 10 (8%) | Consider 12 |
| SM_Flow | 10 (49%), 12 (41%) | 12 ✓ | Fine |
| SM_Norm | Even (150-300 all ~25%) | 200 ✓ | Fine |
| SM_EMA | **80** (31%), 120 (31%) | 100 (20%) | Consider 80 |
| SM_T | **0.0** (61%), 0.25 (27%) | 0.0 ✓ | Fine |

### The Trade-off
- **Current params**: +$2,415 over 3 years, but $0 in low-vol Year 1. Boom-or-bust.
- **Robust params (e.g., 12/10/150/80)**: +$1,223 over 3 years, profitable every year. Consistent but lower ceiling.
- Decision: optimize for consistency or maximize high-vol returns?

## vScalpB Results (TP=3, SL=10, RSI 8/55/45, CD=20)

**Only 9 of 768 configs pass all 3 years (1.2%)**

### Top 3 Robust Configs
| Rank | SM Config | SM_T | Worst PF | Total Net |
|------|-----------|------|----------|-----------|
| 1 | 10/14/200/80 | 0.25 | 1.023 | +$1,338 |
| 2 | **10/12/200/80** | **0.25** | **1.018** | **+$1,162** |
| 3 | 14/14/300/80 | 0.25 | 1.013 | +$940 |

### Current vScalpB Config: 10/12/200/100, SM_T=0.25
- Fails on Year 2 (PF 0.938, -$200)
- The #2 passing config is our exact params with **EMA=80 instead of 100**
- One parameter shift makes it regime-robust

### vScalpB Key Insight
- 100% of passing configs use SM_T=0.25 — threshold is essential
- 56% use SM_EMA=80 — faster EMA is more robust for tight exits
- The edge per year is very thin ($200-$400) — tight TP=3 leaves little margin

## Cross-Style Overlap
Only 3 of 192 SM combos work for BOTH vScalpA AND vScalpB:
1. 14/10/300/100 (A_T=0.15, B_T=0.25)
2. 10/12/200/80 (A_T=0.0, B_T=0.25)
3. 10/12/250/80 (A_T=0.0, B_T=0.25)

**10/12/200/80** is noteworthy — it's our current config with just EMA changed from 100 to 80.

## Conclusions
1. The SM signal is NOT overfit — it works across 3 years of different regimes
2. Our specific params are high-vol optimized, not universally optimal
3. SM_Index=12 and SM_EMA=80 are the regime-robust cluster center
4. SM_T=0.25 is essential for tight-exit strategies (vScalpB)
5. The edge is thin but persistent — consistency across regimes comes at the cost of peak performance
6. No config is a "home run" across all years — best worst-year PFs are ~1.09

## Open Decision
Should we shift to regime-robust params (SM 12/12/200/80) or keep current high-vol params (10/12/200/100)?
- If we expect continued high vol: keep current (more profitable)
- If we want insurance against vol compression: shift to robust params (less upside, no downside)
- Middle ground: keep current params + ATR gate that pauses in low-vol months (already have this for vScalpC at ATR min 263.8)
