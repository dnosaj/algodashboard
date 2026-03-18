---
name: RSI Trendline Backtest Results
description: First backtest of RSI trendline breakout strategy on MNQ — sweep results, key findings, IS/OOS
type: project
---

# RSI Trendline Breakout — Backtest Results (Mar 15, 2026)

## Strategy Definition

**Entry**: RSI trendline breakout on 1-min bars.
- Compute RSI on 1-min closes (sweepable period, best=8)
- Detect pivots: lb_left=10, lb_right=3, min_spacing=10
- Connect each new pivot to last 5 qualifying pivots (piv_lookback=5)
- Descending peak trendline → LONG when RSI breaks above
- Ascending trough trendline → SHORT when RSI breaks below
- Grace period: min_spacing + 2*lb_right = 16 bars before checking breaks
- Entry at bar[i] open using signal from bar[i-1]

**Exit**: TP/SL/EOD. Fill at next bar open. No SM requirement.

**Data**: MNQ 1-min, 375,211 bars, 2025-02-17 to 2026-03-12 (~12.5 months)

## Signal Volume

| RSI Period | Long Signals | Short Signals | Total |
|------------|-------------|---------------|-------|
| 8 | 19,820 | 19,708 | 39,528 |
| 11 | 19,901 | 19,889 | 39,790 |
| 14 | 19,811 | 19,919 | 39,730 |

## Full Sweep: 630 Configs

Swept: RSI [8, 11, 14] x TP [3, 5, 7, 10, 15, 20, 25] x SL [10, 15, 20, 30, 40] x CD [10, 20, 30] x CutOff [15:45, 13:00]

### Top 10 by Sharpe (full dataset)

| # | RSI | TP | SL | CD | CutOff | Trades | WR% | PF | Sharpe | Net$ | MaxDD$ |
|---|-----|----|----|-----|--------|--------|------|------|--------|------|--------|
| 1 | 8 | 15 | 30 | 20 | 13:00 | 1573 | 65.6 | 1.084 | 0.591 | +$3,416 | -$1,369 |
| 2 | 8 | 10 | 40 | 30 | 13:00 | 1292 | 75.4 | 1.091 | 0.579 | +$2,731 | -$1,922 |
| 3 | 8 | 15 | 40 | 30 | 13:00 | 1236 | 70.1 | 1.081 | 0.551 | +$2,813 | -$1,882 |
| 4 | 14 | 20 | 10 | 20 | 13:00 | 1753 | 41.8 | 1.079 | 0.550 | +$2,882 | -$1,979 |
| 5 | 8 | 25 | 40 | 30 | 13:00 | 1136 | 61.4 | 1.073 | 0.534 | +$3,005 | -$2,387 |
| 6 | 8 | 25 | 30 | 10 | 13:00 | 1839 | 56.4 | 1.070 | 0.522 | +$4,317 | -$2,319 |
| 7 | 8 | 20 | 40 | 30 | 13:00 | 1187 | 65.5 | 1.069 | 0.495 | +$2,668 | -$2,110 |
| 8 | 8 | 25 | 30 | 20 | 13:00 | 1428 | 56.0 | 1.064 | 0.477 | +$3,043 | -$1,879 |
| 9 | 14 | 25 | 15 | 20 | 13:00 | 1621 | 43.2 | 1.061 | 0.444 | +$2,552 | -$2,107 |
| 10 | 8 | 20 | 30 | 30 | 13:00 | 1239 | 60.4 | 1.060 | 0.441 | +$2,241 | -$1,428 |

### IS/OOS — Best Single-Exit (RSI=8 TP=15 SL=30 CD=20 CutOff=13:00)

| Period | Trades | WR% | PF | Sharpe | Net$ | MaxDD$ | Exits |
|--------|--------|------|------|--------|------|--------|-------|
| IS | 782 | 65.7% | 1.109 | 0.756 | +$2,149 | -$1,248 | TP:513 SL:264 EOD:5 |
| OOS | 791 | 65.5% | 1.061 | 0.432 | +$1,267 | -$1,369 | TP:517 SL:271 EOD:3 |

OOS holds: WR stable, PF degrades modestly (1.109→1.061).

## Runner Mode: 180 Configs

Swept: RSI=8 only, TP1 [3, 5, 7] x TP2 [15, 20, 25, 30] x SL [30, 40] x CD [20, 30] + CutOff 13:00

### Top 5 Runner Configs

| # | TP1 | TP2 | SL | CD | Trades | WR% | PF | Sharpe | Net$ | MaxDD$ |
|---|-----|-----|----|----|--------|------|------|--------|------|--------|
| 1 | 7 | 20 | 40 | 30 | 2314 | 71.8 | 1.140 | 0.699 | +$7,686 | -$3,174 |
| 2 | 7 | 25 | 40 | 30 | 2279 | 68.5 | 1.133 | 0.673 | +$7,448 | -$3,972 |
| 3 | 5 | 20 | 40 | 30 | 2386 | 70.1 | 1.124 | 0.597 | +$6,428 | -$3,278 |
| 4 | 3 | 25 | 40 | 30 | 2421 | 66.2 | 1.127 | 0.590 | +$6,219 | -$3,282 |
| 5 | 3 | 20 | 40 | 30 | 2452 | 68.9 | 1.121 | 0.561 | +$5,875 | -$3,436 |

### IS/OOS — Best Runner (TP1=7 TP2=20 SL=40 CD=30)

| Period | Trades | WR% | PF | Sharpe | Net$ | MaxDD$ | Exits |
|--------|--------|------|------|--------|------|--------|-------|
| IS | 1120 | 71.4% | 1.096 | 0.496 | +$2,674 | -$2,005 | TP1:491 TP2:306 SL:135 BE:185 EOD:3 |
| OOS | 1194 | 72.2% | 1.185 | 0.893 | +$5,012 | -$3,174 | TP1:534 TP2:323 BE:210 SL:121 EOD:6 |

OOS STRONGER than IS — no overfitting. Runner captures real momentum continuation.

## Scalp Analysis (TP 3-7)

### Key Finding: Cooldown is critical for small TP

| TP | SL | CD=10 Sharpe | CD=20 Sharpe | CD=30 Sharpe |
|----|-----|-------------|-------------|-------------|
| 3 | 30 | -0.095 | +0.001 | **+0.175** |
| 3 | 40 | -0.082 | -0.200 | **+0.153** |
| 5 | 30 | -0.205 | +0.169 | **+0.408** |
| 5 | 40 | -0.185 | -0.048 | **+0.415** |
| 7 | 30 | -0.026 | +0.341 | **+0.369** |
| 7 | 40 | -0.041 | +0.169 | **+0.301** |

- CD=10 is almost always negative for TP 3-7. Too many rapid re-entries after losses.
- CD=30 is consistently best for small TP scalps.
- Wide SL (30-40) with small TP (5-7) works — tight SL kills profitability (noise stops).

### Best Scalp Configs (TP 5-7, CD=30)

| TP | SL | CD | Trades | WR% | PF | Sharpe | Net$ |
|----|----|----|--------|------|------|--------|------|
| 5 | 40 | 30 | 1364 | 81.0% | 1.072 | 0.415 | +$1,798 |
| 5 | 30 | 30 | 1408 | 77.1% | 1.068 | 0.408 | +$1,661 |
| 7 | 15 | 30 | 1448 | 63.7% | 1.062 | 0.414 | +$1,485 |
| 7 | 30 | 30 | 1377 | 74.6% | 1.058 | 0.369 | +$1,548 |

## Key Findings

1. **RSI(8) dominates** — matches existing SM+RSI strategies. RSI(11) and RSI(14) worse.
2. **13:00 ET cutoff helps** — consistent with V15/vScalpC finding.
3. **Runner mode is the play** — $7,686 vs $3,416, OOS stronger than IS.
4. **CD=30 required for scalps** — CD=10 destroys edge at small TP.
5. **Signal is abundant** — ~40k raw signals, 1200-2300 trades after filtering.
6. **Standalone edge exists** — no SM requirement. Pure RSI trendline breakout.
7. **Wide SL with small TP** — counterintuitive but data confirms. SL=30-40 with TP=5-7.

## 3-Year Validation (Mar 17-18, 2026)

### Current params fail on prior years
| Period | Trades | WR% | PF | Net$ |
|--------|--------|------|------|------|
| Year 1 (Feb23-24) | 1,897 | 68.1% | 0.941 | -$2,441 |
| Year 2 (Feb24-25) | 2,020 | 69.3% | 0.982 | -$860 |
| Year 3 (Feb25-26) | 2,314 | 71.8% | 1.140 | +$7,686 |

### 3-Year signal sweep: 10 of 320 pass (3.1%)
Best signal config: **RSI(10), lb_left=10, lb_right=4, min_spacing=15**
- vs current: RSI(8), lb_left=10, lb_right=3, min_spacing=10
- Key shifts: RSI 8→10, lb_right 3→4, min_spacing 10→15
- Cluster: RSI 10 (40%), lb_right=4 (50%), min_spacing=15 (60%)

### 3-Year exit sweep: 26 of 144 pass (18.1%)
Best exit: **TP1=7, TP2=25, SL=35, CD=30** (worst-year PF 1.059, +$8,973 total)
- CD=30 mandatory (100%)
- SL=35 preferred (38%)
- Current exits (TP1=7/TP2=20/SL=40) rank #2

### SM filter: does NOT help on 3-year data
- No filter: +$8,973 (best)
- Block SM-opposed: +$3,038 (kills Year 1)
- Tighter SL on opposed: +$2,952 (Year 3 goes negative)
- The SM-aware tighter SL implemented on Mar 16 should be REMOVED

### Robust RSI TL Config
RSI(10), lb_left=10, lb_right=4, min_spacing=15, TP1=7, TP2=25, SL=35, CD=30, entry 9:30-13:00, SL→BE after TP1, NO SM filter.

## Next Steps

- [x] IS/OOS on scalp configs — done Mar 16
- [x] Pine Script strategy — done Mar 15
- [x] Correlation with portfolio — done Mar 16 (near-zero, 0.024 avg)
- [x] SM alignment analysis — done Mar 16 (helps on dev-only, hurts on 3-year)
- [x] 3-year signal+exit sweep — done Mar 17-18
- [ ] Implement robust config in live engine
- [ ] Remove SM-aware tighter SL (doesn't hold on 3-year data)
- [ ] Test overnight trendlines breaking during NY session

## Files

- Backtest script: `backtesting_engine/strategies/rsi_trendline_backtest.py`
- 3-year sweep: `backtesting_engine/strategies/rsi_tl_3year_sweep.py`
- Correlation analysis: `backtesting_engine/strategies/rsi_tl_correlation_analysis.py`
- SM conviction sweep: `backtesting_engine/strategies/rsi_tl_sm_conviction_sweep.py`
- Pine indicator: `strategies/cae_auto_trendlines.pine`
- Pine strategy: `strategies/rsi_trendline_strategy.pine`
- Live engine: `live_trading/engine/rsi_trendline_strategy.py`
- Plan: `plans/cae-atl-indicator.md`
