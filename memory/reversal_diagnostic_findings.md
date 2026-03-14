# Reversal Diagnostic Findings

**Date**: Mar 3, 2026
**Script**: `backtesting_engine/strategies/reversal_diagnostic.py`
**Charts**: `backtesting_engine/results/reversal_analysis/` (3 PNGs + enriched CSV)
**Data**: 1,499 trades (12 months), 697 matched to 1-min bar context (6-month Databento window)

## Step 0: SL Exit Speed Diagnostic

**Each strategy has a completely different SL problem.**

| Strategy | SL pts | SL exits | Median bars to SL | Immediate (≤3 bars) | Gradual (>10 bars) |
|---|---|---|---|---|---|
| vScalpB | 15 | 98 (27.8%) | **2 bars** | **68.1%** | 10.6% |
| MNQ_V15 | 40 | 103 (13.5%) | 6.5 bars | 34.2% | 42.1% |
| MES_V2 | 35 | 44 (11.5%) | 100 bars | 0% | **100%** |

### Implications

- **vScalpB**: 2/3 of all SLs hit within 3 minutes. This IS an entry quality problem. Pre-filters have the best chance of helping here.
- **MNQ_V15**: Mixed — half fast, half slow. Both entry quality and position management.
- **MES_V2**: All SLs are slow grinds over 1.5+ hours. NOT an entry quality problem — position management issue (BE_TIME already addresses this).

### MAE Distributions

| Strategy | Winners median MAE | Losers median MAE |
|---|---|---|
| vScalpB | 6.4 pts | 26.5 pts |
| MNQ_V15 | 9.2 pts | 64.0 pts |
| MES_V2 | 6.0 pts | 21.2 pts |

---

## Step 1: Entry-Level Signals

697 trades matched to 1-min bar context (Databento data: Aug 2025 – Feb 2026).

### 1A: SM Slope at Entry — STRONG for MES_V2

SM aligned slope = SM slope in the direction of the trade (positive = SM strengthening in trade direction).

**Portfolio:**

| SM Slope Bucket | N | WR% | PF | Avg$/trade | SL% |
|---|---|---|---|---|---|
| Strong against (<-0.02) | 73 | 64.4% | 0.635 | -$12.83 | 12.3% |
| Weak against (-0.02 to -0.005) | 122 | 78.7% | 1.703 | +$10.67 | 9.8% |
| Flat (±0.005) | 160 | 70.0% | 0.887 | -$2.70 | 20.6% |
| Weak with (0.005 to 0.02) | 191 | 79.6% | 1.508 | +$7.05 | 16.2% |
| Strong with (>0.02) | 151 | 84.8% | 1.634 | +$8.07 | 10.6% |

**MES_V2 (strongest differentiation):**

| SM Slope Bucket | N | WR% | PF | Avg$/trade | SL% |
|---|---|---|---|---|---|
| Strong against | 42 | 47.6% | **0.563** | **-$21.04** | 14.3% |
| Weak with | 28 | 67.9% | **2.079** | **+$25.76** | 7.1% |

$46.80/trade swing between best and worst bucket.

**MNQ_V15:** "Weak against" is actually best (PF 2.728, WR 92.2%), "strong against" worst (PF 0.639). SM gently declining doesn't hurt MNQ scalps but hard declines do.

**vScalpB:** "Flat" is worst (PF 0.731, SL 36.1%). "Strong with" is best (PF 2.945). "Weak with" is terrible (PF 0.535, SL 43.5%). Needs strong conviction.

### 1B: SM Zero-Crossing Count (60 bars) — Moderate

| Flips in 60 bars | N | WR% | PF | Sharpe | SL% |
|---|---|---|---|---|---|
| 0 flips | 79 | 74.7% | 1.034 | 0.21 | 13.9% |
| 1 flip | 271 | 79.3% | **1.624** | **3.94** | 11.8% |
| 2 flips | 207 | 75.8% | **0.880** | **-0.89** | 15.9% |
| 3-4 flips | 131 | 72.5% | 1.165 | 1.19 | 19.1% |

1 flip is the sweet spot. 2 flips is the danger zone. Not monotonic — "SM confused" hypothesis doesn't hold cleanly. MES_V2 specifically: 2 flips = PF 0.711, SL 18.9% (vs baseline 9.8%).

### 1C: SM Signal Freshness — Non-Obvious Pattern

| Bars Since SM Flip | N | WR% | PF | Avg$/trade | SL% |
|---|---|---|---|---|---|
| 0-5 (very fresh) | 131 | 78.6% | **1.692** | +$10.26 | 13.7% |
| 6-15 (fresh) | 165 | 78.8% | 1.149 | +$2.93 | 15.2% |
| 16-30 (moderate) | 167 | 71.3% | **0.740** | **-$6.41** | **19.2%** |
| 31-60 (stale) | 158 | 79.1% | 1.761 | +$10.07 | 9.5% |
| 60+ (very stale) | 76 | 76.3% | 1.105 | +$1.98 | 14.5% |

Worst bucket is **16-30 bars** (moderate staleness), not "very stale." MES_V2 amplifies this: 0-5 bars = PF 2.236 (+$26.76/trade), 16-30 bars = PF 0.517 (-$25.58/trade). Interpretation: late enough to miss the initial move but not stale enough for a new regime to establish.

### 1D: Pre-Entry Momentum (Chasing) — SL Rate Effect

Quartiles of 10-bar price move in trade direction before entry.

| Strategy | Q1 SL% (counter-move) | Q4 SL% (chasing) | SL increase |
|---|---|---|---|
| MES_V2 | 2.3% | **22.0%** | **9.6x** |
| MNQ_V15 | 7.6% | **16.5%** | 2.2x |
| vScalpB | 23.3% | **35.7%** | 1.5x |

Chasing increases SL rate dramatically, especially for MES_V2. But WR doesn't collapse as much — it's specifically about hitting the stop, not about winning less.

### 1E: Volume at Entry — STRONG for MES_V2

| Volume Ratio | N | WR% | PF | Avg$/trade | SL% |
|---|---|---|---|---|---|
| <0.5x (quiet) | 21 | 71.4% | 0.627 | -$12.84 | 23.8% |
| 0.5-0.8x (low) | 171 | 77.2% | 0.879 | -$2.53 | 15.2% |
| 0.8-1.2x (normal) | 259 | 77.6% | 1.295 | +$5.14 | 14.7% |
| 1.2-2.0x (elevated) | 189 | 77.8% | **1.708** | **+$11.05** | 12.2% |
| 2.0x+ (spike) | 57 | 70.2% | 0.799 | -$3.80 | 15.8% |

**MES_V2 specifically:**
- Low volume (<0.8x): PF 0.582, Avg -$23.04, SL 22.9%
- Elevated (1.2-2x): PF **2.057**, Avg **+$24.43**, SL 6.0%

$47/trade swing. Low volume = no institutional participation to sustain the SM signal.

**vScalpB:** Volume spike (2x+) is bad: PF 0.652, SL 38.5%. Exhaustion/capitulation at entry.

**Hypothesis inverted:** Moderate volume elevation is GOOD (confirms institutional flow). Only extreme spikes are bad (exhaustion).

### 1F: Entry Bar Rejection Wick — Inconsistent

Results vary by strategy — not a reliable signal. MES_V2 shows <15% wick = PF 2.179 (clean entry bars are best), but vScalpB shows the opposite (<15% wick = PF 0.672). Not actionable as a universal filter.

---

## Step 2: Daily Regime (Lagged Predictors)

All 1,475 trades with market data.

### 2A: Rule of 16 (Realized vs Implied Vol) — Hypothesis Inverted

Prior day's actual range / VIX-implied expected range.

| Prior Day Regime | N | WR% | PF | Avg$/trade | Sharpe |
|---|---|---|---|---|---|
| <0.5 (very quiet) | 65 | 72.3% | 1.308 | +$6.26 | 2.92 |
| 0.5-0.75 (calm) | 270 | 75.6% | **0.934** | **-$1.37** | -0.76 |
| 0.75-1.0 (normal) | 372 | 73.4% | 1.005 | +$0.10 | 0.06 |
| 1.0-1.5 (hot) | 450 | 77.8% | 1.296 | +$5.62 | 3.17 |
| 1.5+ (over-delivering) | 318 | 76.1% | **1.577** | **+$12.44** | **6.13** |

**Opposite of hypothesis.** When the market over-delivered on volatility yesterday, today is BETTER for our strategies. Our SM signal thrives in momentum/trending environments, not calm ones. The "calm" bucket (0.5-0.75) is the worst.

MES_V2 amplifies: over-delivering = PF 1.728, Avg +$25.53. Calm = PF 0.931.

### 2B: VIX Overnight Gap — STRONG for vScalpB

| VIX Gap | N | WR% | PF | Avg$/trade | Sharpe | SL% |
|---|---|---|---|---|---|---|
| Gap down >2% | 242 | 76.4% | **1.625** | +$11.91 | **6.31** | 16.9% |
| Gap down 0.5-2% | 320 | 77.8% | 1.236 | +$4.19 | 2.44 | 13.8% |
| Flat gap (±0.5%) | 244 | 75.4% | 1.289 | +$6.03 | 2.70 | 16.4% |
| Gap up 0.5-2% | 242 | 74.4% | 1.075 | +$1.52 | 0.90 | 14.9% |
| Gap up >2% | 427 | 74.5% | **1.050** | **+$1.16** | **0.67** | **19.0%** |

**vScalpB specifically:**
- Gap down 0.5-2%: PF **1.692**, SL 20.3% — best regime
- Gap up >2%: PF **0.824**, SL **33.7%**, Avg **-$2.86** — loses money

**MES_V2:** Gap down >2% = PF 1.890, Avg +$27.52. Gap up 0.5-2% = PF 0.969.

Knowable at session start. VIX gapping up = overnight hedging demand = more whippy environment for scalps.

### 2C: Multi-Day VIX Momentum — Interesting but Non-Monotonic

**3-day VIX change (prior day's value):**

| VIX 3d Momentum | N | PF | Sharpe | SL% |
|---|---|---|---|---|
| Crashing (<-5%) | 497 | 1.095 | 1.11 | 17.1% |
| Falling (-5 to -2%) | 169 | **1.949** | **7.60** | **7.1%** |
| Flat (±2%) | 157 | 1.327 | 2.85 | 13.4% |
| Rising (2-5%) | 182 | **0.751** | **-3.78** | **20.3%** |
| Surging (>5%) | 456 | 1.380 | 4.28 | 18.4% |

"Falling moderately" is best. "Rising moderately" is worst. But "surging" is also good — pattern is non-linear. The 5-day version doesn't confirm this pattern (rising 2-5% is actually good on the 5-day window). Likely not robust enough for a filter.

### 2D: VIX Death Zone (19-22) — VALIDATES ON IS AND OOS

**This is the most robust finding in the entire analysis.**

| vScalpB | Period | Outside 19-22 | Death Zone 19-22 |
|---|---|---|---|
| **PF** | IS | 1.462 | 0.845 |
| **PF** | **OOS** | **1.486** | **0.597** |
| **SL%** | IS | 24.1% | 39.5% |
| **SL%** | **OOS** | **23.0%** | **57.7%** |

vScalpB SL rate nearly **triples** in the death zone. Effect is **stronger** on OOS than IS — not overfit.

| MNQ_V15 | Period | Outside 19-22 | Death Zone 19-22 |
|---|---|---|---|
| PF | IS | 0.980 | 0.864 |
| PF | **OOS** | **1.495** | **0.716** |
| SL% | IS | 15.3% | 17.3% |
| SL% | **OOS** | **9.2%** | **21.3%** |

Also validates. OOS PF more than halves in the death zone.

| MES_V2 | Period | Outside 19-22 | Death Zone 19-22 |
|---|---|---|---|
| PF | IS | 1.600 | **1.771** (good!) |
| PF | OOS | 1.137 | 0.813 |

MES_V2 is inconsistent — IS says death zone is fine, OOS says it's bad. Not a clean signal for MES.

**Mechanistic explanation:** VIX 19-22 = enough volatility to whip past tight SL (15pt for vScalpB, 40pt for V15), but not enough to create strong directional trends that MES_V2 captures with its wider TP=20.

### 2E: VIX Prior Day Range — Moderate, Strategy-Dependent

| Prior Day VIX Range | N | PF | Sharpe | SL% |
|---|---|---|---|---|
| Tight | 492 | 1.226 | 2.44 | 12.8% |
| Normal | 496 | 0.997 | -0.04 | 18.5% |
| Wide | 487 | 1.417 | 4.65 | 17.9% |

"Normal" VIX range days are worst. Both tight and wide are fine. Pattern reverses for vScalpB (tight is best, wide is worst). Conflicting across strategies — not actionable as a universal filter.

---

## Step 3: Interaction Effects

### SM Weakening × VIX Death Zone

| Combo | N | WR% | PF | Avg$/trade | SL% |
|---|---|---|---|---|---|
| VIX safe + SM steady | 447 | 78.7% | 1.287 | +$4.63 | 15.0% |
| VIX safe + SM weakening | 155 | 76.8% | 1.174 | +$3.42 | 7.1% |
| VIX 19-22 + SM steady | 51 | 70.6% | 0.963 | -$0.86 | 25.5% |
| VIX 19-22 + SM weakening | 37 | 64.9% | 0.926 | -$2.63 | 27.0% |

VIX death zone is the dominant factor. SM state adds marginal differentiation within it.

### Combined Skip Filter Results

**vScalpB:** Filter (skip death zone + SM weakening or SM confused) would skip 44 trades. SL avoidance ratio 34.1% — catches more TP winners than SL losers. **Not effective.** The VIX death zone alone would be a cleaner filter.

**MNQ_V15:** Filter separates well. SKIP trades: PF 0.807, Avg -$4.30. KEEP trades: PF 1.576, Avg +$6.19. Better result but needs validation as a proper pre-filter.

---

## Actionability Tiers

### Tier 1 — Validated, implementable now

1. **VIX Death Zone gate (19-22) for vScalpB + MNQ_V15**: Skip trading when prior day VIX closed 19-22. Validates on both IS and OOS. Knowable at session start. Would affect ~20% of trading days. vScalpB SL rate: 23% → 57.7% in death zone.

2. **VIX overnight gap >2% gate for vScalpB**: Skip when VIX opens >2% above prior close. PF drops to 0.824, SL rises to 33.7%. Knowable at session start. Needs IS/OOS validation.

### Tier 2 — Strong signal, needs IS/OOS validation

3. **SM slope filter for MES_V2**: Skip entries when SM aligned slope < -0.02 (rolling over hard). PF 0.563 vs 1.120 baseline. Entry-level, implementable in strategy logic.

4. **Volume minimum for MES_V2**: Skip entries when volume ratio < 0.8x (low participation). PF 0.582 vs 1.120 baseline. Entry-level.

5. **Pre-entry momentum (chasing) for MES_V2**: Q4 chasing entries have 22% SL rate vs 9.8% baseline.

### Tier 3 — Interesting but not clean enough

6. SM freshness 16-30 bar danger zone (strong for MES_V2 but non-monotonic)
7. Rule of 16 over-delivering is GOOD (opposite of hypothesis — useful insight but not a filter)
8. 3-day VIX momentum (strong differentiation but doesn't confirm on 5-day window)
9. SM zero-crossing count (2 flips is bad but pattern isn't monotonic)

### Not actionable

- Entry bar rejection wick (inconsistent across strategies)
- VIX prior day range (conflicting between strategies)
- VIX death zone for MES_V2 (inconsistent IS vs OOS)

---

## Key Insights

1. **The 3 strategies have fundamentally different SL profiles.** Any filter must be strategy-specific.
2. **vScalpB's problem is entry quality** (68% immediate SL). Pre-filters can help.
3. **MES_V2's problem is position management** (100% slow SL). Entry-level signals (SM slope, volume) still matter but won't prevent the majority of losses.
4. **The Rule of 16 result inverted our hypothesis.** Our strategies LIKE momentum/volatility. The danger is calm/choppy markets, not volatile ones. This makes sense — SM is a momentum indicator.
5. **VIX 19-22 is the Goldilocks zone of bad** — enough vol to hit stops, not enough for directional follow-through. This is mechanistically sound, not just a statistical artifact.
