# SM Strength at Entry & Volume Delta Research (Feb 18, 2026)

## Context

After Feb 12-17 drawdown (8 ML stops, -$977 combined), chart data revealed 45% of MNQ entries had |SM| < 0.1 ("weak conviction"). Three of five weak entries during that week hit max loss. This prompted two investigations:

1. **SM Strength Analysis**: Does |SM| at entry predict trade quality over 6 months?
2. **Volume Delta vs SM**: Does real order flow (42.6M NQ ticks with aggressor side) tell a different story than computed SM?

## SM Strength Analysis — MNQ v11

### SM Distribution at Entry (368 trades, 6 months)

| |SM| Range | Trades | % | PF | WR | Avg$/trade | SL exits |
|-----------|--------|---|-----|-----|-----------|----------|
| 0.00-0.05 | 101 | 27% | 1.65 | 54.5% | +$10.45 | 8 |
| 0.05-0.10 | 54 | 15% | 1.46 | 55.6% | +$9.33 | 6 |
| **0.10-0.15** | **37** | **10%** | **0.52** | **45.9%** | **-$13.63** | **7** |
| 0.15-0.20 | 46 | 13% | 2.36 | 58.7% | +$17.01 | 1 |
| 0.20-0.30 | 52 | 14% | 1.83 | 57.7% | +$18.82 | 5 |
| 0.30-0.50 | 48 | 13% | 2.79 | 70.8% | +$26.22 | 5 |
| 0.50-1.00 | 30 | 8% | 1.85 | 66.7% | +$16.44 | 4 |

**Key finding**: Relationship is NOT monotonic. The 0.00-0.05 bucket is decent (PF 1.65). The **0.10-0.15 "death zone"** is the only money-losing bucket (PF 0.52, -$504 net). Above 0.15, quality improves substantially.

### Threshold Sweep

| Min |SM| | Trades | PF | $/trade | Net | SL | Train PF | Test PF |
|---------|--------|-----|---------|------|----|---------:|--------:|
| 0.00 | 368 | 1.669 | $12.41 | $4,567 | 36 | 1.530 | 1.781 |
| 0.10 | 213 | 1.733 | $14.12 | $3,008 | 22 | 1.592 | 1.863 |
| **0.15** | **176** | **2.154** | **$19.96** | **$3,513** | **15** | **2.081** | **2.219** |
| 0.20 | 130 | 2.106 | $21.00 | $2,730 | 14 | 2.002 | 2.203 |
| 0.22 | 120 | 2.555 | $26.01 | $3,121 | — | — | — |
| 0.30 | 78 | 2.367 | $22.46 | $1,752 | 9 | 3.267 | 1.903 |

**Sweet spot: 0.15** — PF +29%, $/trade +61%, SL exits 36→15 (-58%). Train/test validates (+6.6%, no degradation). Thresholds up to 0.25 pass validation; 0.30 overfits (-42% degradation).

**Trade-off**: Filtering at 0.15 removes 192 trades that are collectively profitable ($1,054 net, PF 1.28). You gain quality but lose quantity.

### Monthly Stability (MNQ, threshold 0.15)

All 7 months positive except Dec (essentially flat at +$30). Oct and Nov improved dramatically (PF 4.47 and 3.06 vs baseline 2.22 and 1.53).

### MES v9.4 — Thesis REJECTED

No clear benefit from SM threshold filtering on MES. Removed trades are actually among the best (|SM| < 0.05: PF 1.66). The slower SM params (20/12/400/255) produce fewer weak entries. **Do NOT add SM threshold to MES.**

## Volume Delta vs SM — MNQ v11

### Data

- 42.6M NQ ticks with CME aggressor side ('B'=buy, 'A'=sell, 'N'=neutral)
- 99.99% side coverage (only 143 neutral ticks)
- Aggregated to 1-min bars: buy_vol, sell_vol, delta, CVD (session-reset at 18:00 ET)
- All 368 trades matched 100% to delta data

### Key Findings

**1. SM agrees with real delta 77% of the time (5-min window)**
SM is a good proxy for actual order flow direction. Not perfect, but strong agreement.

**2. When SM and delta DISAGREE, trades are MORE profitable**
- 5-min: Agree avg +$11.00 vs Disagree avg +$17.26
- 20-min: Agree +$11.67 vs Disagree +$13.95
- Interpretation: SM *leads* delta. SM detects regime shifts before raw volume accumulation reflects it.

**3. The 0.10-0.15 "death zone" is NOT a flow direction problem**
Both agree (-$13.47) and disagree (-$14.23) subsets lose money in that SM range. Delta confirms the direction — the issue is structural to that SM magnitude.

**4. Adding delta/CVD to SM filtering does NOT improve results**

| Filter | Trades | PF | $/trade | Net |
|--------|--------|-----|---------|------|
| Baseline | 368 | 1.669 | +$12.41 | +$4,567 |
| |SM| >= 0.15 alone | 176 | 2.154 | +$19.96 | +$3,513 |
| |SM|>=0.15 + delta agrees | 140 | 2.105 | +$18.98 | +$2,657 |
| |SM|>=0.15 + CVD aligned | 122 | 2.009 | +$19.86 | +$2,423 |

SM threshold alone is the best single filter. Delta/CVD add no independent value on top.

**5. Divergence trades are profitable**
43 trades where SM and 5-min delta strongly disagreed (|delta| > 100): avg +$19.61, WR 60%. Better than average. SM appears to be "right" more often than raw delta when they conflict.

**6. Delta strength has mild standalone value**
Weakest delta quartile (|5m delta| < 84): avg +$7.40, WR 48.9%. Middle quartiles: +$14-16. Strongest: +$11.37. Not strong enough for a filter.

## Conclusions

1. **SM works well** — it captures real institutional flow direction 77% of the time, and leads raw delta
2. **SM magnitude at entry matters for MNQ** — the 0.15 threshold is validated and would improve PF by 29%
3. **Raw volume delta does not add value** beyond what SM already captures
4. **The 0.10-0.15 death zone is structural**, not a flow direction error — possibly SM oscillating around zero
5. **MES does not benefit from SM strength filtering** — different dynamics with slower SM params
6. **Implementation priority**: SM threshold is a validated, ready-to-use improvement for MNQ. Delta/CVD research can be shelved.

## Files

| File | Purpose |
|------|---------|
| `strategies/sm_strength_analysis.py` | SM magnitude sweep, bucketed analysis, train/test validation |
| `strategies/volume_delta_vs_sm.py` | Real order flow vs computed SM comparison |
| `strategies/chart_deep_dive.py` | Chart data analysis (Feb 12-18 MES + MNQ exports) |

## Decision Status

- **SM threshold 0.15 for MNQ**: VALIDATED on train/test. Not yet implemented. Available for future strategy version.
- **Volume delta as filter**: REJECTED. Does not improve upon SM alone.
- **SM threshold for MES**: REJECTED. No benefit.
