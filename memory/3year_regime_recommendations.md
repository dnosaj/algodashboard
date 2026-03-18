---
name: 3-Year Regime Recommendations
description: Pragmatic parameter change recommendations based on 3-year SM sweeps across MNQ and MES
type: project
---

# 3-Year Regime Analysis — Parameter Recommendations (March 17, 2026)

## The data

3 independent years of 1-min data (Feb 2023 - Mar 2026). Each year tested independently. No config saw all 3 years during development — Year 1 and Year 2 are truly unseen.

| Year | MNQ ATR | MES ATR | Regime |
|------|---------|---------|--------|
| Feb 2023 - Feb 2024 | 213 | 46 | Low vol |
| Feb 2024 - Feb 2025 | 279 | 58 | Medium vol |
| Feb 2025 - Mar 2026 | 369 | 81 | High vol (dev period) |

## Current production performance across 3 years

| Strategy | Year 1 | Year 2 | Year 3 | 3Y Total | Pass all 3? |
|----------|--------|--------|--------|----------|-------------|
| vScalpA | -$5 | +$124 | +$2,297 | +$2,415 | NO (Y1 fails) |
| vScalpB | +$215 | -$200 | +$1,536 | +$1,551 | NO (Y2 fails) |
| vScalpC | +$93 | +$1,383 | +$6,099 | +$7,575 | YES (barely) |
| MES v2 | -$1,706 | +$1,235 | +$4,602 | +$4,131 | NO (Y1 fails) |

Current portfolio total: +$15,672 over 3 years. Two strategies fail at least one year.

## Recommendations

### Change 1: MES v2 — SM_T 0.0 → 0.25, SM flow 12→14, SM EMA 255→300, TP1 6→8

**Why:** Current MES v2 loses $1,652 in Year 1 (low vol) with -$2,952 drawdown because SM_T=0.0 takes every signal including weak ones. 43% of 326 passing MES configs use SM_T=0.25 vs only 23% for SM_T=0.0.

**What changes:**
- SM(20/12/400/255) → SM(20/14/400/300)
- SM_T=0.0 → SM_T=0.25
- TP1=6 → TP1=8

**Validated comparison (runner backtest, 3 years):**

| | Current | Robust |
|---|---|---|
| Y1 (low vol) | -$1,652, PF 0.901, DD -$2,952 | **+$1,476**, PF 1.193, DD **-$1,450** |
| Y2 (med vol) | +$3,189, PF 1.199 | +$2,015, PF 1.233 |
| Y3 (high vol) | +$8,701, PF 1.473 | +$2,375, PF 1.248 |
| 3Y Total | **+$10,238** | +$5,866 |
| Losing years | 1 | **0** |
| Worst DD | -$2,952 | **-$1,450** |
| Trades/year | ~350 | ~163 |

**The trade-off is stark:** Current makes $4,371 more over 3 years but has a losing year and 2x the drawdown. Robust is steady but leaves ~$1,457/year on the table.

**Decision factors:**
- If you believe high vol continues: keep current (more profit now)
- If you want insurance: change to robust (consistent but lower ceiling)
- Middle ground: run both configs as separate strategies at 1 contract each

### Change 2: vScalpC — SM Index 10→12, EMA 100→80, TP1 7→10, TP2 25→30, SL 40→30

**Why:** vScalpC already passes all 3 years, but barely (Y1 PF 1.007). The robust config moves worst-year PF to 1.169. ALL 36 exit configs tested with robust SM pass all 3 years — the entry signal is genuinely robust with these params.

**What changes:**
- SM(10/12/200/100) → SM(12/12/200/80)
- TP1=7 → TP1=10
- TP2=25 → TP2=30
- SL=40 → SL=30

**Expected impact:**
- Current 3Y: +$7,575, worst-year PF 1.007
- Robust 3Y: +$8,455, worst-year PF 1.169

**The trade-off:** Wider TP1 (10 vs 7) means fewer TP1 fills. Tighter SL (30 vs 40) means more stops. But the net is better because the tighter SL cuts losses faster on bad entries while the wider TP1 captures more on good entries.

**This is the safest change.** It improves every metric (total P&L, worst-year PF, consistency) without a significant downside.

### Change 3: vScalpA — SM Index 10→12, EMA 100→80

**Why:** 59 of 576 configs pass all 3 years. The robust cluster center is SM_Index=12 (58% of winners) and SM_EMA=80 (31%). Our current config misses Year 1 by $5.

**What changes:**
- SM(10/12/200/100) → SM(12/12/200/80)
- Exits unchanged (TP=7, SL=40)

**Expected impact:**
- Current 3Y: +$2,415 (Y1 -$5)
- Robust 3Y: +$3,352 (Y1 +$592)

**The trade-off:** Modest. +$937 over 3 years, fixes Year 1. Year 3 goes from +$2,297 to +$1,814 — you lose ~$480 in the current high-vol period. Whether this is worth it depends on whether $480/year of upside is worth the insurance against a zero year.

### Change 4: vScalpB — SM EMA 100→80

**Why:** Only 9 of 768 configs pass all 3 years. The #2 passing config is exactly our params with EMA=80 instead of 100. One parameter change.

**What changes:**
- SM(10/12/200/100) → SM(10/12/200/80)
- Everything else unchanged

**Expected impact:** Fixes Year 2 (PF 0.938 → ~1.02). Very thin edge either way. vScalpB's tight TP=3/SL=10 leaves almost no margin for parameter variation.

**The trade-off:** Marginal improvement in bad years, marginal degradation in good years. The edge is so thin that this change barely matters either way.

**Recommendation: lowest priority.** The improvement is too small to be confident it's signal vs noise.

## Priority order

1. **vScalpC** — safest change, improves everything, no downside. Do this.
2. **MES v2** — biggest impact but real trade-off (high-vol performance drops). Decide based on vol outlook.
3. **vScalpA** — moderate improvement, small trade-off. Consider.
4. **vScalpB** — marginal, low confidence. Skip unless other changes prove out.

### Change 5: RSI TL — RSI 8→10, lb_right 3→4, min_spacing 10→15, SL 40→35, TP2 20→25, REMOVE SM filter

**Why:** Current RSI TL config fails Years 1+2. The overnight 3-year sweep (320 signal × 144 exit configs) found the robust config. 3.1% of signal configs pass all 3 years. The key shifts are more confirmation bars (lb_right 4 vs 3) and wider pivot spacing (15 vs 10) — the signal needs to be more selective to work across regimes.

**What changes:**
- RSI period: 8 → 10
- lb_right: 3 → 4 (more confirmation, less repainting risk)
- min_spacing: 10 → 15 (wider pivot spacing, fewer but better trendlines)
- SL: 40 → 35
- TP2: 20 → 25
- REMOVE the SM-aware tighter SL (implemented Mar 16 — does NOT hold on 3-year data, Year 3 goes negative)

**Expected impact:**
- Current 3Y: +$4,385 (fails Y1 -$2,441, Y2 -$860)
- Robust 3Y: +$8,973 (passes all 3 years, worst PF 1.059)

**The trade-off:** Year 3 drops from +$7,686 to +$3,717. You lose $3,969 in the best year but gain $5,742 in the two losing years. Net improvement +$4,588 over 3 years.

## What a veteran algo trader would consider

- These backtests have no slippage, no partial fills, no market impact. Real performance will be worse than backtest by some amount. The thin edges (PF 1.02-1.09 in bad years) may not survive execution costs.
- The "robust" configs trade less (half the MES trades). Fewer trades = higher variance around the expected value. A year could still be negative with a robust config — just less likely.
- Running current AND robust as separate strategies (split the capital) hedges the regime bet. You underperform in high vol (half capital on each) but never have a zero year.
- The 3-year window is still limited. 2023 had a specific macro regime (rate hikes ending). 2024 had rate cuts. 2025-2026 has tariff uncertainty. None of these regimes may repeat.
- The SM signal is real (10-36% of configs pass 3 years) but thin. This is a scalping edge, not a trend-following edge. Respect the thin margins.
