# Round 3: S/R and Price Structure Entry Gates — Full Research Report

**Date**: March 6, 2026
**Data**: MNQ+MES 1-min bars, Feb 17 2025 – Mar 5 2026 (12.5 months)
**Split**: Chronological midpoint per instrument (~Aug 25-26, 2025)
**Scripts**: `backtesting_engine/strategies/sr_*.py` (7 study files + `sr_common.py`)

---

## Context & Motivation

On Mar 6, the user observed long entries firing directly into obvious local highs (intraday
resistance). Round 2 had proven prior-session S/R has a real OOS edge but was too aggressive
on trade count (30-50% blocked). Round 3 extends the filter battery with 6 new
price-structure gates tested across ALL 3 strategies simultaneously.

These are **generic filters computed from price/volume** — not SM-specific. The goal is
finding pre-entry gates that improve risk-adjusted returns without destroying trade count.

---

## Methodology

### Test Framework
- **Pre-filter validation**: Gates block entries (change cooldowns/episodes), not post-filter
- **IS/OOS chronological midpoint split**: ~6 months each per instrument
- **SM computed on full data**: Avoids EMA warm-up artifacts at split boundary
- **RSI remapped within each split**: 5-min RSI to 1-min mapping done separately per half
- **Gate architecture**: Single boolean array per instrument, `gate[i-1]` checked at entry bar
- **Not direction-aware**: Level-based gates block near ANY level (conservative)

### Pass/Fail Criteria
- **STRONG PASS**: IS PF ≥+5% AND OOS PF ≥+5%, both Sharpe non-negative, N ≥ 70% baseline, both profitable
- **MARGINAL PASS**: PF within ±5%, at least one Sharpe improves, N ≥ 70%, both profitable
- **FAIL**: PF degrades >5% either half, OR net negative, OR N < 70%

### Baseline Stats
| Strategy | Trades (IS/OOS) | WR | PF (IS/OOS) | Net$ | Sharpe (IS/OOS) |
|----------|-----------------|------|-------------|-------|-----------------|
| vScalpA | 472 (235/237) | 84.5% | 1.289 (1.341/1.236) | +$1,948 | 1.734/1.224 |
| vScalpB | 376 (181/195) | 71.5% | 1.186 (1.252/1.129) | +$889 | 1.444/0.744 |
| MES_V2 | 540 (279/261) | 56.1% | 1.332 (1.367/1.292) | +$4,331 | 1.865/1.522 |
| **Portfolio** | — | — | — | **+$7,168** | **1.662/1.232** |

---

## Phase 1: All-3-Must-Pass Results

Initial evaluation required every config to pass for ALL 3 strategies simultaneously.

### Study 1: Prior Day H/L + Volume Profile (VPOC/VAH/VAL)

**Theory**: Prior-day high/low are institutional reaction zones. Adding VPOC/VAH/VAL
(volume profile from prior RTH session) creates more granular levels.

**Implementation**:
- `compute_rth_volume_profile()`: Bins 1-min closes into fixed-width buckets per RTH day (10:00-16:00 ET). VPOC = max volume bucket. VAH/VAL = 70% value area expanding from VPOC.
- Bin widths: 2 pts MNQ, 5 pts MES
- Gate blocks if close within buffer_pts of ANY prior-day level (H, L, VPOC, VAH, VAL)

**Sweep**: buffer_pts = [0, 2, 5, 8, 10, 15] → 6 configs + baseline

**Full Results (IS dPF% / OOS dPF%)**:
| Config | vScalpA | vScalpB | MES_V2 | Portfolio OOS Sharpe |
|--------|---------|---------|--------|---------------------|
| buf0 | MARGINAL (+0.0%/+0.7%) | MARGINAL (+3.0%/+0.0%) | FAIL (-0.3%/-0.1%) | 1.241 |
| buf2 | MARGINAL (-0.4%/+0.1%) | FAIL (-2.2%/-0.4%) | FAIL (-2.0%/-4.7%) | 1.091 |
| buf5 | MARGINAL (-2.3%/+0.6%) | FAIL (-6.9%/+5.5%) | **STRONG (+10.4%/+9.0%)** | 1.506 |
| buf8 | FAIL (-5.7%/+2.4%) | FAIL (-9.6%/+8.1%) | FAIL (N<70%, 63%/59%) | 1.335 |
| buf10 | FAIL (-7.3%/-2.8%) | FAIL (-11.2%/-4.2%) | FAIL (N<70%, 54%/52%) | 0.671 |
| buf15 | MARGINAL (+3.4%/+18.0%) | MARGINAL (+8.0%/-4.3%) | FAIL (N<70%, 41%/40%) | 1.359 |

**All-3 Verdict**: **FAIL** — No config passes all 3 strategies
- buf5 is excellent for MES_V2 (STRONG PASS) but kills vScalpB (IS -6.9%)
- buf15 has great vScalpA OOS (+18.0%) but drops MES below 70% trade count
- The 5 levels per day are too many for MNQ's tight SL strategies

---

### Study 2: VWAP Z-Score Gate

**Theory**: Block entries when price is >N sigma from session VWAP (overextended, likely
to mean-revert against entry). VWAP is session-anchored and regime-adaptive.

**Implementation**:
- `compute_vwap_zscore()`: Session-anchored (reset at 18:00 ET), 60-bar rolling window for mean/std of (close - VWAP). Min 10 bars before computing (else z=0).
- Gate: `|z_score| ≤ max_z`
- MNQ had 1.6% NaN VWAP values (forward-filled within session)

**Sweep**: max_z = [1.0, 1.5, 2.0, 2.5, 3.0] → 5 configs + baseline

**Full Results**:
| Config | vScalpA | vScalpB | MES_V2 | Portfolio OOS Sharpe |
|--------|---------|---------|--------|---------------------|
| z1.0 | FAIL (-4.2%/-16.9%) | FAIL (+3.0%/-8.1%) | FAIL (-17.0%/-14.2%) | 0.402 |
| z1.5 | FAIL (+9.9%/-10.0%) | FAIL (+0.1%/-9.7%) | FAIL (-6.8%/-4.7%) | 0.815 |
| z2.0 | FAIL (+18.3%/-6.1%) | MARGINAL (+3.4%/-0.7%) | MARGINAL (-0.5%/+3.3%) | 1.234 |
| z2.5 | FAIL (+0.9%/-5.7%) | FAIL (-0.3%/-4.9%) | MARGINAL (+2.7%/-0.2%) | 1.082 |
| z3.0 | MARGINAL (+5.1%/+0.7%) | FAIL (-1.1%/-2.6%) | MARGINAL (+1.0%/-0.2%) | 1.209 |

**All-3 Verdict**: **FAIL** — vScalpA OOS always degrades
- Consistent IS→OOS degradation for vScalpA at every threshold
- z2.0 shows marginal promise for vScalpB+MES but fails vScalpA
- Even z3.0 (very loose, blocks 0-2 trades) only gets vScalpA to marginal

---

### Study 3: Squeeze Momentum (TTM Squeeze)

**Theory**: Block entries during BB-inside-KC squeeze (low volatility compression). SM+RSI
signals expected to whipsaw during these choppy periods.

**Implementation**:
- BB: SMA(20) ± 2.0 × StdDev(20). KC: SMA(20) ± mult × ATR_Wilder(20)
- Squeeze ON: BB_upper < KC_upper AND BB_lower > KC_lower
- Gate: Block during squeeze + min_bars_off bars after release

**Sweep**: kc_mult = [1.0, 1.5, 2.0] × min_bars_off = [0, 5, 10] → 9 configs + baseline

**All-3 Verdict**: **TOTAL FAIL** — Hurts all strategies
- Every config failed at portfolio level (best: kc1.0_off10 OOS +$2,483 vs baseline $2,864)
- vScalpB universally harmed — every config degraded both IS and OOS PF
- Only 3 MARGINAL PASS verdicts across 27 strategy-config combinations
- **Why**: Strategies already avoid choppy periods via SM threshold + RSI bands.
  Squeeze double-filters for the same condition with worse precision.
  Squeeze is also lagging — by the time BB exits KC, the move is underway.

---

### Study 4: Leledc Exhaustion

**Theory**: Block entries on "exhaustion" bars. Count consecutive bars where close >
close[lookback=4]. When count hits maj_qual threshold = exhaustion detected. The "last
buyer" at the end of a sustained move.

**Implementation**:
- `compute_leledc_exhaustion(closes, maj_qual, lookback=4)`: Counts consecutive bars where
  close > close[i-4] (bullish) or close < close[i-4] (bearish)
- Gate blocks for `persistence` bars after either direction exhaustion detected
- Not direction-aware: blocks if EITHER bull or bear exhaustion

**Sweep**: maj_qual = [5, 6, 7, 8, 9] × persistence = [1, 3, 5] → 15 configs + baseline

**Full Results (selected configs)**:
| Config | vScalpA | vScalpB | MES_V2 | Portfolio OOS Sharpe |
|--------|---------|---------|--------|---------------------|
| mq5_p1 | FAIL (+11.0%/-8.0%) | FAIL (-16.5%/+4.6%) | FAIL (-5.6%/-11.6%) | 0.793 |
| mq6_p1 | MARGINAL (+7.8%/+0.2%) | MARGINAL (+4.7%/+12.6%) | FAIL (-5.3%/-8.5%) | 1.114 |
| mq7_p1 | MARGINAL (+12.7%/+3.2%) | MARGINAL (+8.5%/+3.9%) | FAIL (-1.2%/-0.9%) | 1.328 |
| mq7_p5 | MARGINAL (-2.4%/+16.7%) | STRONG (+12.5%/+15.7%) | FAIL (+3.0%/-7.6%) | 1.401 |
| mq8_p1 | MARGINAL (+4.3%/+7.3%) | MARGINAL (-3.0%/+2.4%) | FAIL (-5.9%/+6.3%) | 1.571 |
| mq8_p5 | FAIL (-9.3%/+16.7%) | MARGINAL (-1.4%/+4.8%) | MARGINAL (-4.2%/+0.9%) | 1.564 |
| **mq9_p1** | **STRONG (+7.2%/+11.1%)** | **MARGINAL (-1.0%/+5.9%)** | **MARGINAL (-3.8%/+4.7%)** | **1.623** |
| mq9_p3 | MARGINAL (-4.5%/+15.0%) | FAIL (-6.9%/+3.6%) | FAIL (-8.3%/-6.6%) | 1.288 |
| mq9_p5 | FAIL (-5.1%/+17.5%) | MARGINAL (-4.0%/+6.6%) | FAIL (-5.3%/+1.5%) | 1.617 |

**All-3 Verdict**: **PASS** — mq9_p1 is the ONLY config that passes all 3
- Portfolio OOS Sharpe: 1.623 (+32% over baseline 1.232)
- Portfolio OOS Net$: +$3,348 (+17% over baseline $2,864)
- Blocks 6-11% of trades (very light filtering)
- Performance improves monotonically mq5→mq9 with persistence=1
- Higher persistence (3, 5) tends to fail — over-blocking after exhaustion
- Lower maj_qual (5, 6) is too sensitive — blocks good trades

**Why mq9_p1 works**:
- 9 consecutive bars where close > close[4] is genuinely rare and extreme
- persistence=1 means only the exhaustion bar itself is blocked (not sustained blocking)
- Low false positive rate (6-11% blocked) = only truly extreme conditions filtered

---

### Study 5: Initial Balance Proximity

**Theory**: Block entries near RTH Initial Balance (first 30/60 min) high/low. IB levels
are institutional reaction zones, often acting as magnets and reversal points.

**Implementation**:
- `compute_initial_balance()`: Tracks developing IB during first N minutes of RTH (9:30 ET)
- After IB period, levels finalized for rest of day
- 30-min IB fully formed before entries start at 10:00 ET
- 60-min IB uses developing IB for bars 10:00-10:30
- Gate: Block if close within buffer_pts of IB high or IB low

**Sweep**: ib_period = [30, 60] × buffer_pts = [0, 2, 5, 8, 10] → 10 configs + baseline

**Full Results (selected configs)**:
| Config | vScalpA | vScalpB | MES_V2 | Portfolio OOS Sharpe |
|--------|---------|---------|--------|---------------------|
| ib30_buf2 | MARGINAL (-1.0%/+6.0%) | MARGINAL (+0.6%/-1.5%) | FAIL (-4.1%/-0.7%) | 1.277 |
| ib30_buf5 | MARGINAL (+2.1%/+5.7%) | FAIL (-0.9%/-3.3%) | MARGINAL (+0.7%/+14.5%) | 1.612 |
| ib30_buf8 | MARGINAL (-0.7%/+19.0%) | FAIL (-6.3%/-8.4%) | FAIL (N<70%) | 2.098 |
| ib60_buf0 | MARGINAL (+0.0%/+0.1%) | FAIL (+0.0%/+0.0%) | FAIL (+0.0%/-0.1%) | 1.232 |
| **ib60_buf2** | **MARGINAL (+6.6%/+3.5%)** | **MARGINAL (-0.2%/+0.4%)** | **MARGINAL (-2.5%/+3.4%)** | **1.392** |
| **ib60_buf5** | **STRONG (+11.2%/+8.7%)** | **MARGINAL (+1.1%/-0.9%)** | **MARGINAL (-5.0%/+4.4%)** | **1.459** |
| ib60_buf8 | STRONG (+5.5%/+16.3%) | MARGINAL (-3.1%/+2.6%) | FAIL (N<70%, 66%/56%) | 1.591 |
| ib60_buf10 | MARGINAL (+0.1%/+11.9%) | FAIL (-3.3%/-2.6%) | FAIL (N<70%, 59%/49%) | 1.376 |

**All-3 Verdict**: **PASS** — ib60_buf2 and ib60_buf5 pass all 3
- ib60_buf5 is the stronger config (vScalpA STRONG PASS)
- 60-min IB outperforms 30-min IB consistently
- MES_V2 shows consistent OOS improvement across IB configs despite IS degradation
- ib60_buf8 has best vScalpA numbers but drops MES trade count below 70%

---

### Study 6: Intraday Pivots (Multi-Window + Scoring)

**Theory**: Block entries near confirmed intraday pivot highs/lows. Score levels by
multi-timeframe confirmation (detected at multiple window sizes) and clustering (retested
multiple times). Higher-scored levels = stronger rejection probability.

**Implementation**:
- `compute_pivots_fast()`: scipy maximum_filter1d/minimum_filter1d, windows [10,20,30,50]
- `score_pivots()`: +1 per additional window confirmation, +1 per clustering, cap 5
- `precompute_nearest_levels()`: O(log M) bisect lookup per bar
- Gate: Block if nearest S/R within buffer_pts of close

**Sweep**: buffer_pts = [2, 5, 8, 10, 15] × min_score = [1, 2, 3] → 15 configs + baseline

**Key numbers**:
- MNQ: 26,458 resistance pivots, 26,224 support pivots
- MES: 36,824 resistance pivots, 34,845 support pivots
- buf2_ms1: vScalpA OOS -$381, MES OOS -$259 (from +$1,740 baseline)
- buf15_ms1: MES OOS had ZERO trades
- Even loosest config (buf2_ms3) blocked 70-90% of trades

**All-3 Verdict**: **TOTAL FAIL** — Too many levels, too aggressive blocking
- Level density: 26K-37K pivots per instrument = nearest pivot almost always within 2-15 pts
- min_score filtering (2, 3) barely helps — even scored levels are dense
- Would need fundamentally different approach (much larger windows, time-decay on old levels)

---

### Study 7: Combined Best Filters (IB+Leledc)

Tested combinations of the two all-3 passing configs (Leledc mq9_p1 + IB ib60_buf5).

| Config | vScalpA IS/OOS dPF% | vScalpB IS/OOS dPF% | MES IS/OOS dPF% | Portfolio OOS Sharpe |
|--------|---------------------|---------------------|-----------------|---------------------|
| ib only | +11.2%/+8.7% | +1.1%/-0.9% | -5.0%/+4.4% | 1.459 |
| leledc only | +7.2%/+11.1% | -1.0%/+5.9% | -3.8%/+4.7% | 1.623 |
| **ib+leledc** | **+12.3%/+17.1%** | **+0.2%/+0.4%** | **-5.6%/+6.0%** | **1.657** |

**All-3 Verdict**: ib+leledc **FAILS** — MES_V2 IS PF -5.6% triggers fail threshold
- vScalpA gets even better (STRONG, +17.1% OOS PF)
- vScalpB marginal but passes
- MES_V2 IS degradation crosses -5% → automatic fail
- Portfolio Sharpe is actually the best at 1.657, but doesn't meet all-3 criteria

---

## Phase 1 Summary Scorecard

| Study | Filter | Best Config | All-3 Pass? | Portfolio OOS Sharpe |
|-------|--------|-------------|:-----------:|:--------------------:|
| 1 | Prior Day H/L + VP | buf5 | NO (MES STRONG but vScalpB FAIL) | 1.506 |
| 2 | VWAP Z-Score | z2.0/z3.0 | NO (vScalpA always degrades OOS) | 1.234/1.209 |
| 3 | Squeeze (TTM) | — | NO (hurts everything) | <baseline |
| **4** | **Leledc Exhaustion** | **mq9_p1** | **YES** | **1.623 (+32%)** |
| **5** | **Initial Balance** | **ib60_buf5** | **YES** | **1.459 (+18%)** |
| 6 | Intraday Pivots | — | NO (blocks 70-99% of trades) | <0 |
| 7 | Combined IB+Leledc | — | NO (MES IS PF -5.6%) | 1.657 |

---

## Phase 2: Re-Examination with Separate Instrument Criteria

### Why Separate Criteria

The all-3-must-pass requirement was reconsidered for two reasons:

1. **vScalpB is not "weak" — it's different.** vScalpB was selected as the safe, low-variance
   strategy (SM_T=0.25, SL=15, fewer trades). Its OOS stats (PF 1.129, Sharpe 0.744) are lower
   than vScalpA/MES, but its portfolio value comes from low correlation (A-B 0.26, B-MES 0.18),
   not standalone performance. The tight SL makes each trade more binary, so small samples
   swing PF significantly. A handful of SLs going differently would change its PF by 10%.

2. **MNQ and MES are different products.** MNQ strategies (SL=15-40, TP=5) care about
   momentum quality. MES (SL=35, TP=20) cares about level proximity — it needs 20 pts of
   runway to reach TP, so entering near resistance is fatal. Different failure modes →
   different optimal filters.

### New Framework
- **MNQ**: Both vScalpA and vScalpB share the same filter (same instrument/timeframe).
  Filter must help vScalpA and not hurt vScalpB.
- **MES**: Evaluated independently with its own pass/fail criteria.

### MES_V2 — Independent Evaluation (All Studies)

**Baseline**: 540 trades (IS 279 / OOS 261), PF 1.332 (IS 1.367 / OOS 1.292), Sharpe IS 1.865 / OOS 1.522

| Study | Config | IS Trades | IS PF | IS dPF% | OOS Trades | OOS PF | OOS dPF% | OOS Sharpe | Verdict |
|-------|--------|-----------|-------|---------|------------|--------|----------|------------|---------|
| 1 | buf0 | 279 | 1.363 | -0.3% | 261 | 1.291 | -0.1% | 1.515 | FAIL |
| 1 | buf2 | 267 | 1.339 | -2.0% | 252 | 1.231 | -4.7% | 1.247 | FAIL |
| **1** | **buf5** | **212** | **1.509** | **+10.4%** | **195** | **1.408** | **+9.0%** | **2.042** | **STRONG** |
| 1 | buf8 | 177 | 1.492 | +9.1% | 154 | 1.285 | -0.5% | 1.534 | FAIL (N<70%) |
| 1 | buf10 | 152 | 1.380 | +1.0% | 135 | 1.087 | -15.9% | 0.509 | FAIL (N<70%) |
| 1 | buf15 | 114 | 1.497 | +9.5% | 104 | 1.240 | -4.0% | 1.282 | FAIL (N<70%) |
| 2 | z1.0 | 209 | 1.134 | -17.0% | 200 | 1.109 | -14.2% | 0.618 | FAIL |
| 2 | z1.5 | 248 | 1.274 | -6.8% | 226 | 1.231 | -4.7% | 1.231 | FAIL |
| 2 | z2.0 | 269 | 1.360 | -0.5% | 246 | 1.335 | +3.3% | 1.720 | MARGINAL |
| 2 | z2.5 | 274 | 1.404 | +2.7% | 258 | 1.289 | -0.2% | 1.502 | MARGINAL |
| 2 | z3.0 | 277 | 1.380 | +1.0% | 261 | 1.289 | -0.2% | 1.511 | MARGINAL |
| 3 | all | — | — | — | — | — | — | — | FAIL |
| **4** | **mq9_p1** | **260** | **1.315** | **-3.8%** | **244** | **1.353** | **+4.7%** | **1.810** | **MARGINAL** |
| 4 | mq8_p1 | 257 | 1.287 | -5.9% | 240 | 1.373 | +6.3% | 1.895 | FAIL (IS) |
| **5** | **ib30_buf5** | **229** | **1.377** | **+0.7%** | **198** | **1.479** | **+14.5%** | **2.351** | **MARGINAL** |
| **5** | **ib60_buf2** | **272** | **1.333** | **-2.5%** | **244** | **1.336** | **+3.4%** | **1.746** | **MARGINAL** |
| **5** | **ib60_buf5** | **231** | **1.299** | **-5.0%** | **201** | **1.349** | **+4.4%** | **1.817** | **MARGINAL** |
| 5 | ib30_buf8 | 191 | 1.364 | -0.2% | 148 | 1.748 | +35.3% | 3.446 | FAIL (N<70%) |
| 6 | all | — | — | — | — | — | — | — | FAIL |

**MES Standout: Prior Day buf5** — the ONLY MES STRONG PASS across all studies:
- IS PF 1.367→1.509 (+10.4%), OOS PF 1.292→1.408 (+9.0%)
- IS Sharpe 1.865→2.492 (+33.6%), OOS Sharpe 1.522→2.042 (+34.2%)
- Trade count: 76%/75% of baseline (above 70% threshold)
- Both IS AND OOS improve on every metric simultaneously
- This filter failed the old all-3 criteria only because vScalpB (MNQ) doesn't like the
  5 prior-day levels, but MES with its 35pt SL and 20pt TP navigates around levels fine.

### MES-Only Combination Test

Tested all combinations of the three MES-passing filters (Prior Day buf5, Leledc mq9_p1,
IB ib60_buf5) for MES_V2 independently:

| Config | IS Trades | IS PF | IS dPF% | OOS Trades | OOS PF | OOS dPF% | OOS Sharpe | Verdict |
|--------|-----------|-------|---------|------------|--------|----------|------------|---------|
| **prior_day** | **212** | **1.509** | **+10.4%** | **195** | **1.408** | **+9.0%** | **2.042** | **STRONG** |
| leledc | 260 | 1.315 | -3.8% | 244 | 1.353 | +4.7% | 1.810 | MARGINAL |
| ib | 231 | 1.299 | -5.0% | 201 | 1.349 | +4.4% | 1.817 | MARGINAL |
| prior_day+leledc | 197 | 1.452 | +6.2% | 179 | 1.437 | +11.2% | 2.207 | FAIL (N 71%/69%) |
| prior_day+ib | 181 | 1.436 | +5.0% | 147 | 1.246 | -3.6% | 1.339 | FAIL (N 65%/56%) |
| leledc+ib | 214 | 1.290 | -5.6% | 185 | 1.369 | +6.0% | 1.918 | FAIL (IS -5.6%) |
| prior_day+leledc+ib | 168 | 1.337 | -2.2% | 130 | 1.163 | -10.0% | 0.924 | FAIL (N 60%/50%) |

**Key finding: Stacking filters ALWAYS fails for MES.** Each filter blocks different trades
(17-67 each), and ANDing them pushes total blocked trades past the 70% threshold. The
filters aren't catching the same bad trades — they're catching different, rare edge cases.

prior_day+leledc is interesting (IS+OOS both improve) but OOS trades drop to 69% — just
barely below the 70% threshold. This is a judgment call.

---

## Deep Analysis & Critical Examination

### 1. Leledc mq9_p1 — Confirmed as Peak (Extended Sweep)
The winning config uses maj_qual=9, persistence=1. Initial sweep only went to mq9,
raising concern the optimum might be beyond the tested range. **Extended sweep to
mq10, 11, 12, 15 confirms mq9 IS the peak:**

| mq | Portfolio OOS Sharpe | vScalpA OOS dPF% | MES OOS dPF% | Blocked (vScalpA) |
|----|---------------------|------------------|--------------|-------------------|
| 8 | 1.571 | +7.3% | +6.3% | 31 |
| **9** | **1.623** | **+11.1%** | **+4.7%** | **25** |
| 10 | 1.183 | +1.8% | -5.0% | 16 |
| 11 | 1.319 | +6.2% | -4.3% | 13 |
| 12 | 1.265 | +3.2% | -4.8% | 8 |
| 15 | 1.297 | +5.3% | -0.3% | 3 |

The curve reverses at mq10 (Sharpe drops from 1.623 to 1.183). mq10+ still helps MNQ
strategies but HURTS MES (all fail). mq9 is the sweet spot where exhaustion detection
works across both instruments. Above mq9, events are so rare they stop helping MES.
**Yellow flag resolved — mq9 is genuinely the best, not an artifact of truncated testing.**

### 2. Statistical significance problem
With ~230-260 trades per strategy per half, a PF change of ±5% is within normal sampling
variation. Rough 95% CI for PF at ~230 trades with PF ~1.3 is approximately ±0.15 (±11%):
- vScalpA OOS PF +11.1% (Leledc) — **barely significant**
- MES_V2 OOS PF +9.0% (Prior Day) — **borderline significant**
- MES_V2 OOS PF +4.7% (Leledc) — **NOT significant** (could be noise)
- vScalpB OOS PF +5.9% (Leledc) — **NOT significant**

Only vScalpA (Leledc) and MES (Prior Day) clearly exceed noise. The other improvements
could vanish with more data. This doesn't mean the filters are useless, but we should be
humble about the marginal pass verdicts.

### 3. We never tested "require BOTH signals to agree before blocking"
Our AND combination blocks if EITHER filter blocks (gate = gate1 & gate2, where True=allow).
We never tested the opposite: block ONLY when both conditions are simultaneously true
(require consensus). This would be a more selective gate — only blocking trades where
there's both exhaustion AND proximity to a level. Not tested.

### 4. MES consistently shows IS degradation / OOS improvement
Across Leledc, IB, and combined tests, MES IS PF goes down while OOS goes up. Three
possible explanations:
- **(a) Real regime shift**: IS (Feb-Aug 2025) had different structure than OOS (Aug 2025-Mar 2026)
- **(b) Sampling noise**: IS 279 trades, OOS 261 — not huge samples
- **(c) Filter correctly removes IS overfit**: Baseline MES overfits IS slightly
  (IS PF 1.367 > OOS 1.292). Filter removes trades that worked in IS but not OOS.

Interpretation (c) is bullish — suggests the filter is doing its job. Prior Day buf5 is
the exception: BOTH IS and OOS improve, making it the most robust result.

### 5. vScalpB is not weak — it's different
vScalpB was selected as the conservative, low-variance strategy. SM_T=0.25 already acts
as a quality filter, rejecting marginal entries that vScalpA takes. When you layer
additional filters on top, you're double-filtering — vScalpB's entries were already
pre-filtered by the SM threshold. This explains why filters help vScalpA more: vScalpA
(SM_T=0.0) has more marginal entries to remove.

vScalpB's portfolio value comes from low correlation (A-B 0.26, B-MES 0.18), not
standalone stats. Its OOS PF of 1.129 and Sharpe of 0.744 are "weaker" than vScalpA/MES,
but the tight SL=15 makes each trade binary (TP=5 or SL=15, no in-between) and a handful
of SLs going differently swings PF by 10%+.

### 6. Why MNQ and MES need different filters
MNQ (TP=5, SL=15-40) needs **momentum quality** — entering at the top of a 9-bar run is
fatal because there's only 5 pts of upside to capture. Leledc exhaustion catches exactly
this: "the market just ran 9 bars in one direction, don't chase."

MES (TP=20, SL=35) needs **level clearance** — it needs 20 pts of runway to reach TP, so
entering 5 pts below yesterday's high is fatal (ceiling blocks the target). Prior Day
levels catch exactly this: "there's a known ceiling/floor within your TP distance."

The two filters address different failure modes tied to each instrument's risk profile.

### 7. The 70% trade count threshold
Prior Day buf5 blocks 24-25% of MES trades. The prior_day+leledc combo hits 69% OOS
(just below 70%). The threshold is somewhat arbitrary:
- It protects against overfitting to "only take the best trades"
- But it may be too conservative when the blocked trades are genuinely bad
- Prior Day buf5 MES at 75-76% retention is safely above the threshold

### 8. Are we just removing unlucky trades?
Blocking 6-25% of trades and seeing PF improvement doesn't prove the filter identifies
"bad" conditions. It could be that the blocked trades happened to be losers by chance.
To validate:
- Walk-forward test (rolling windows, not just one IS/OOS split)
- Analyze blocked trades specifically — are they consistently worse?
- Permutation test: randomly block same NUMBER of trades 1000x, see how often random
  blocking produces similar PF improvement

### 9. Things we didn't test
- **Direction-aware Leledc**: Block longs only on bull exhaustion (our strategies are
  long-biased on SM bull signals). Could be more targeted.
- **ATR-scaled buffers**: Fixed 5-pt buffer regardless of volatility. ATR-adaptive
  could be more robust across regimes.
- **Time-of-day interaction**: Exhaustion at 10:30 AM vs 3:00 PM may have different
  predictive power.
- **Leledc lookback variation**: Fixed lookback=4. Testing [2,4,6,8] unexplored.
- **Walk-forward validation**: Single IS/OOS split. Rolling windows give stronger evidence.
- **Cross-instrument gates**: Could MNQ exhaustion predict MES entry quality?

---

## Final Conclusions

### Filters are mutually exclusive — pick one per instrument, don't stack

The signal is thin. Each individual filter finds a small set of bad trades to remove
(6-25% of entries). When you AND two filters, you're not removing the same bad trades
twice — you're removing different trades, cutting total volume past the point of usefulness.

This tells us these filters aren't catching the same underlying problem. They each catch
different, rare edge cases. One filter per instrument is the right approach.

### What works (individually)

| Instrument | Filter | Type | What it catches | OOS Improvement |
|-----------|--------|------|-----------------|-----------------|
| MNQ | Leledc mq9_p1 | Momentum exhaustion | "Last buyer" after 9 consecutive up-closes | vScalpA PF +11.1%, Sharpe +49% |
| MES | Prior Day buf5 | Level proximity | Entry within 5 pts of prior day H/L/VPOC/VAH/VAL | PF +9.0%, Sharpe +34% |
| MES | Leledc mq9_p1 | Momentum exhaustion | Same as MNQ, weaker effect | PF +4.7%, Sharpe +19% |
| MES | IB ib60_buf5 | Level proximity | Entry within 5 pts of today's 60-min IB H/L | PF +4.4%, Sharpe +19% |

### What doesn't work
- **Stacking filters**: Every AND combination fails (trade count or IS degradation)
- **Squeeze (TTM)**: Hurts all strategies (redundant with SM+RSI)
- **VWAP Z-Score**: vScalpA OOS always degrades; MES marginal at best
- **Intraday Pivots**: Too many levels, blocks 70-99% of trades

### Recommended Implementation

| Instrument | Filter | Config | Blocked% | Complexity |
|-----------|--------|--------|----------|------------|
| **MNQ** | Leledc mq9_p1 | 9 consecutive close>close[4], block 1 bar | 6-11% | Trivial (one counter) |
| **MES** | Prior Day buf5 | Block within 5 pts of prior day H/L/VPOC/VAH/VAL | 24-25% | Moderate (volume profile) |

These are independent, simple, and address different instruments' different risk profiles.
They are conceptually complementary — Leledc catches momentum exhaustion, Prior Day
catches level proximity.

### Implementation notes
- **Leledc** is trivial: one counter per instrument, when it hits 9 → block next bar.
  No levels, no VWAP, no volume data needed. Negligible latency.
- **Prior Day buf5** requires computing prior-day volume profile (VPOC/VAH/VAL) at
  session open. Bin closes into 5-pt buckets, find max-volume bucket, compute 70% value
  area. Do this once per day. Then check 5 levels (H, L, VPOC, VAH, VAL) at each entry.

---

## Prior Day Level Breakdown — Per-Level-Type Analysis (Mar 13, 2026)

**Script**: `backtesting_engine/strategies/sr_prior_day_level_breakdown.py`
**Motivation**: CF engine revealed $500 in MES TP1+TP2 winners blocked on Mar 12, all
triggered by VPOC proximity (within 1.25-4.75 pts). User questioned whether prior-day
levels as a concept are too coarse — are all 5 level types pulling their weight?

### Bar-Level Block Attribution (buf=5, MES full dataset)
| Level | Bars blocked | % of total |
|-------|-------------|-----------|
| VAH | 52,071 | 13.9% |
| VPOC | 50,158 | 13.3% |
| High | 41,911 | 11.2% |
| VAL | 41,344 | 11.0% |
| Low | 23,544 | 6.3% |
| **Any** | **159,452** | **42.4%** |

### IS/OOS Results by Level Combination
| Config | IS PF | IS dPF% | OOS PF | OOS dPF% | OOS Sharpe | Verdict |
|--------|-------|---------|--------|----------|------------|---------|
| Baseline | 1.385 | — | 1.195 | — | 1.067 | — |
| H/L only | 1.432 | +3.4% | 1.154 | **-3.4%** | 0.860 | MARGINAL |
| VPOC only | 1.401 | +1.2% | 1.239 | +3.7% | 1.282 | MARGINAL |
| VAH/VAL only | 1.444 | +4.3% | 1.272 | +6.4% | 1.448 | MARGINAL |
| H/L + VPOC | 1.479 | +6.8% | 1.260 | +5.4% | 1.380 | **STRONG** |
| VPOC + VA | 1.465 | +5.8% | 1.295 | +8.4% | 1.567 | **STRONG** |
| All 5 | 1.526 | +10.2% | 1.292 | +8.1% | 1.548 | **STRONG** |

### Blocked Trade Quality Analysis (FULL dataset)
| Level | Blocked trades | WR% | Avg $/trade | Total $ | Quality |
|-------|---------------|-----|-------------|---------|---------|
| **High** | **58** | **63.8%** | **+$5.37** | **+$311** | **HURTING** |
| **Low** | **41** | **58.5%** | **+$12.93** | **+$530** | **HURTING** |
| VPOC | 58 | 48.3% | -$2.48 | -$144 | HELPING |
| **VAH** | **62** | **71.0%** | **+$14.23** | **+$883** | **HURTING** |
| VAL | 53 | 45.3% | -$3.25 | -$173 | HELPING |

### Key Findings

1. **H/L alone HURTS OOS** (dPF% -3.4%). The blocked trades near yesterday's H/L are
   actually profitable (63.8% WR, +$311 high; 58.5% WR, +$530 low). These are breakout
   trades that clear the level and run — exactly what MES TP=20 needs.

2. **VAH is the worst offender**: 71.0% WR, +$883 total. The gate is removing the
   strategy's best entries. Trades near yesterday's value area high are typically
   breakouts from the prior-day range — high-quality trend entries.

3. **VPOC is the only clearly helpful single level** (48.3% WR, -$144). When price is
   near yesterday's VPOC, it's in a mean-reversion zone where both sides chop.

4. **VAL is marginally helpful** (45.3% WR, -$173). Price near the prior-day value area
   low acts as a pivot — some bounce, some break.

5. **The paradox**: All 5 combined gets STRONG PASS despite 3 of 5 levels hurting
   individually. This is because the composite gate blocks more total trades (69 IS, 65 OOS)
   and the selection effect at scale works differently than per-level analysis.

### Recommendation: Test VPOC+VAL Only

The data suggests a leaner gate using only VPOC + VAL (the two levels that actually
remove losing trades) could outperform the full 5-level gate:
- VPOC+VA already gets STRONG PASS with +8.4% OOS dPF and Sharpe 1.567 (best of all combos)
- Dropping VAH (the biggest offender at +$883 in lost winners) while keeping VPOC+VAL
  is the natural next experiment
- This would also reduce the block rate, keeping more of the good breakout trades

### Open Question for Strategist/Research Agent

**Should the prior-day level gate use VPOC+VAL only instead of all 5 levels?**
- VPOC+VA combo has the best OOS Sharpe (1.567 vs 1.548 for all-5)
- H/L and VAH individually remove profitable breakout trades
- But the all-5 combo still passes STRONG — the IS improvement (+10.2%) gives more room
- Need to weigh: is +0.019 OOS Sharpe improvement (VPOC+VA vs all-5) significant enough
  to justify the change, given the smaller N of blocked trades?
- **Walk-forward validation** across rolling windows would settle this definitively
