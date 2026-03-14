# OOS Analysis Session — Feb 21, 2026

## Context

Downloaded 6 months of prior data (Feb 17 – Aug 17, 2025) from Databento for $1.28.
Tested all 3 production strategies on this TRUE out-of-sample data.
All strategies failed OOS. Regime analysis determined it's primarily regime change,
not overfitting. User pushed back on some conclusions — correctly.

## OOS Validation Results

| Strategy | Period | Trades | WR% | PF | Net $ | SL count |
|----------|--------|--------|-----|----|-------|----------|
| MNQ v11.1 | IS | 226 | 62.4% | 1.797 | +$3,529 | 23 |
| MNQ v11.1 | OOS | 220 | 44.1% | 0.600 | -$3,434 | 52 |
| MNQ v15 | IS | 363 | 88.2% | 1.272 | +$1,324 | 38 |
| MNQ v15 | OOS | 385 | 81.8% | 0.932 | -$509 | 60 |
| MES v9.4 | IS | 359 | 54.9% | 1.266 | +$1,188 | 0 |
| MES v9.4 | OOS | 374 | 50.8% | 0.897 | -$849 | 0 |

12-month combined check: EXACT match. No SM warmup artifact.

## 4-Step Deep Dive Results (Feb 21)

### Step 1: Crash Exclusion + Trade Autopsy

**Big-move days (range > 500 pts):** 15 days in OOS (Feb 27, Mar 3-10, Apr 4-16, Apr 30).
April 7-9 had 1,210-2,174 pt ranges (tariff crash).

**Crash exclusion helps marginally:** OOS minus big-move: -$2,545 (PF 0.602) vs full OOS -$3,434.
OOS minus Mar+Apr: -$1,969 (PF 0.512). Strategy still fails even without crash.

**Critical MFE/MAE autopsy:**
- **98% of SL trades were profitable first** (avg MFE 24.1 pts before reversing)
- TP=5 would have saved 71% of SL trades, TP=3 saves 77%
- SL trades avg MAE = 77.4 pts (all are >50 pts, which is the SL)
- **FINDING: Problem is exit timing (SM too slow), not entry quality**
- Tighter SL alone won't help; faster exits (TP or faster SM) are the lever

### Step 2: SL Sweep

| SL | IS PF | IS Net$ | OOS PF | OOS Net$ | OOS SL# |
|----|-------|---------|--------|----------|---------|
| 0 | 1.552 | +$2,871 | 0.612 | -$3,568 | 0 |
| 15 | 1.573 | +$2,388 | 0.588 | -$2,461 | 131 |
| 25 | 1.861 | +$3,507 | 0.616 | -$2,710 | 94 |
| 35 | 1.805 | +$3,540 | 0.648 | -$2,676 | 73 |
| 50 | 1.797 | +$3,529 | 0.600 | -$3,434 | 52 |
| 100 | 1.557 | +$2,899 | 0.564 | -$4,270 | 26 |

- IS-OOS PF correlation: 0.521 (moderate)
- **Best OOS PF: SL=35 (0.648)**
- **Best OOS Net$: SL=15 (-$2,461)**
- Wider SL always worse on OOS. No SL value makes OOS profitable.

### Step 3: Exit Model Comparison (8 models)

| Model | IS PF | IS Net$ | OOS PF | OOS Net$ |
|-------|-------|---------|--------|----------|
| 1. SM flip (SL=50) | 1.797 | +$3,529 | 0.600 | -$3,434 |
| **2. TP=5 (thresh=0.15)** | **1.554** | **+$1,547** | **0.827** | **-$782** |
| 3. SM flip (SL=35) | 1.805 | +$3,540 | 0.648 | -$2,676 |
| 5. Fast SM (exit=50) | 1.546 | +$2,158 | 0.697 | -$2,261 |
| 6. FastSM=50+SL=35 | 1.524 | +$2,071 | 0.711 | -$2,000 |
| 7. Trail 3pts | 1.455 | +$805 | 0.517 | -$1,592 |
| 8. Breakeven 3pts | 1.432 | +$1,198 | 0.518 | -$1,970 |

- **TP=5 is the clear best exit model** — loses least on OOS (-$782), PF 0.827
- TP=5 on April (crash month): PF 1.865, +$455 — actually PROFITABLE
- Fast SM (EMA=50) whipsaw: 54% of fast exits would have been profitable on slow SM

### Step 4: Param Sweep (1,575 combos, TP=5 exit)

- **374 valid combos** (≥100 trades, monthly consistency)
- **All top 20 pass walk-forward** (profitable in both OOS halves)
- IS-OOS PF correlation: **0.067** (very weak — params don't generalize)
- **191/374 combos profitable on BOTH IS and OOS** (PF > 1.0)
- Stability scores: 0.84-0.87 (good plateau, not spikey optima)

**Best OOS combos (TP=5 exit, crash days excluded):**

| SM_T | RSI_P | B/S | CD | SL | OOS PF | OOS Net$ | IS PF | IS Net$ |
|------|-------|-----|----|----|--------|----------|-------|---------|
| 0.10 | 12 | 65/35 | 10 | 35 | 1.486 | +$674 | 0.958 | -$99 |
| 0.10 | 12 | 65/35 | 15 | 35 | 1.477 | +$661 | 0.967 | -$77 |
| **0.25** | **8** | **55/45** | **15** | **15** | **1.383** | **+$605** | **1.201** | **+$405** |
| **0.25** | **8** | **55/45** | **30** | **15** | **1.373** | **+$574** | **1.292** | **+$519** |

**Key finding:** SM_T=0.25 / RSI=8/55-45 / SL=15 is the only cluster profitable on BOTH
IS and OOS, with 6/7 OOS months profitable. Higher threshold + wider RSI + much tighter SL.

## v10_test_common.py Modification

Added `sm_exit=None` parameter to `run_backtest_v10` for dual-SM exit detection.
Entry/episode reset uses `sm` (slow), exit flip uses `sm_exit` (fast) when provided.
Reviewed and verified: sm_exit=None produces identical results (226 trades, PF 1.797).

## Data Files

- OOS: `databento_MNQ_1min_2025-02-17_to_2025-08-17.csv` (175,839 bars)
- IS: `databento_MNQ_1min_2025-08-17_to_2026-02-13.csv` (172,582 bars)
- MES OOS: `databento_MES_1min_2025-02-17_to_2025-08-17.csv` (175,831 bars)
- MES IS: `databento_MES_1min_2025-08-17_to_2026-02-13.csv` (172,495 bars)

## Scripts

| File | Purpose |
|------|---------|
| `oos_validation.py` | Runs all 3 strategies on IS+OOS, comparison table |
| `oos_12month_check.py` | Combined 12-month run to verify no SM warmup effect |
| `oos_regime_analysis.py` | Volatility, SM quality, SL, direction, regime analysis |
| `oos_step1_crash_exclusion.py` | Crash filter + MAE/MFE autopsy |
| `oos_step2_sl_sweep.py` | SL sweep grid (0-100) |
| `oos_step3_exit_models.py` | 8 exit model comparison + whipsaw analysis |
| `oos_step4_param_sweep.py` | 1,575-combo param sweep + walk-forward validation |
