# Strategy Version History

## MNQ Strategies

### v9 SM-Flip (Jan 2026)
- Architecture: 1-min chart, AlgoAlpha SM via input.source(), 5-min RSI via request.security()
- TV validated (Jan 19 - Feb 12): 45 trades, 62.22% WR, PF 2.027, +$1,196.70
- Python backtest: 44 trades, 63.6% WR, PF 2.325, +$1,583/1lot
- File: `strategies/scalp_v9_smflip.pine` (DO NOT modify)

### v9.4 SM-Flip + Stop (Feb 14)
- v9 + max loss stop 50pts via strategy.close() inside fully-gated if block
- TV (Jan 19 - Feb 13): 50 trades, 60% WR, PF 1.649, +$1,071.50, MaxDD $421
- Superseded by v11 for MNQ; still optimal for MES
- File: `strategies/scalp_v9.4_smflip.pine`

### v10 Feature Variants (Feb 14) — ALL REJECTED
- 11 features tested: underwater exit, OR alignment, VWAP filter, prior day levels, price structure exit, SM reversal, RSI momentum
- v9 baseline NEGATIVE on 6-month 1-min data: PF 0.871, -$1,385
- ALL 11: REJECT (0 ADOPT, 0 MAYBE, 11 REJECT)

### v11 MNQ (Feb 15) — TV VALIDATED
- SM computed natively in Pine (no external indicator)
- Params: SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50
- 6-month: 368 trades, 57.9% WR, PF 1.669, +$4,567, DD -$567
- Walk-forward: OOS PF 1.713 (better than IS 1.634), NOT overfit
- TV validated (Jan 19 - Feb 13): 66 trades, PF 1.681, +$1,174.86
- File: `strategies/scalp_v11MNQ.pine` (superseded by v11.1)

### v11.1 MNQ (Feb 19) — SHELVED (Feb 21)
- v11 + SM threshold 0.15 (skip death zone entries)
- Pre-filter: 234 trades, PF 1.592, WR 61.1%, +$3,004, SL=28
- Episode reset fixed to zero-crossing only (not threshold)
- **SHELVED:** SM flip exit profitable on IS (+$3,529) but loses on OOS (-$509).
  Regime detection failed. At scale, tail risk is unbounded.
- File: `strategies/scalp_v11.1MNQ.pine`
- Config: `MNQ_V11_1` commented out in `engine/config.py`

### v15 TP=5 Scalp (Feb 17) — vScalpA, ACTIVE
- TP=5 exit exploiting near-perfect entry MFE (97% trades go profitable)
- TV (Jan 25 - Feb 17): 60 trades, 86.67% WR, PF 1.272, +$264.60
- 12-month (SL=40): 750 trades, WR 85.0%, PF 1.070, +$880, MaxDD -$1,065, Sharpe 0.39
- Runs as baseline scalp strategy
- File: `strategies/scalp_v15_trail_MNQ.pine`
- Config: `MNQ_V15` in `engine/config.py`

#### SL Sweep (Feb 26) — SL reduced from 50 to 40
SL=50 was inherited from SM-flip era (v9/v11), never re-optimized for TP=5. Swept SL=[10..75] with TP=5 fixed.

| SL | Net | Trades | WR | PF | MaxDD | Sharpe |
|---|---|---|---|---|---|---|
| 10 | -$542 | 1,059 | 90.6% | 0.930 | -$1,148 | -0.22 |
| 15 | +$303 | 888 | 88.2% | 1.013 | -$1,102 | 0.07 |
| 20 | +$555 | 822 | 86.8% | 1.029 | -$920 | 0.17 |
| 25 | +$604 | 783 | 85.8% | 1.037 | -$982 | 0.22 |
| 30 | +$549 | 757 | 85.3% | 1.027 | -$880 | 0.20 |
| 35 | +$792 | 753 | 85.1% | 1.054 | -$905 | 0.33 |
| 40 | +$880 | 750 | 85.0% | 1.070 | -$1,065 | 0.39 |
| 50 | +$613 | 748 | 84.9% | 1.048 | -$1,689 | 0.25 |
| 60 | +$816 | 748 | 84.9% | 1.066 | -$1,543 | 0.32 |
| 75 | +$816 | 748 | 84.9% | 1.066 | -$1,543 | 0.32 |

Findings:
- SL=40 best by Sharpe (0.39) and PF (1.070). SL=30 best MaxDD (-$880).
- SL=10 net loser -- too tight for MNQ noise.
- SL=60 and SL=75 identical to SL=50 in trades/WR -- no trades reach 50-75pt MAE.
- Sweet spot: SL=35 to SL=40.
- ALL SL values fail IS-OOS consistency (IS period negative for all). Edge is in entries, not SL sizing.
- Decision: SL=50 -> SL=40. Better risk-adjusted return, tighter loss cap.

### vScalpB MNQ (Feb 21) — ACTIVE
- High-conviction TP scalp: SM_T=0.25, RSI 8/55-45, TP=5, SL=15, CD=20
- Same SM engine as vScalpA but filters to strong SM signals + tighter stop
- 12-month: 345 trades, WR 72.8%, PF 1.273, +$1,106, MaxDD -$358, Sharpe 1.55
- IS: +$461, OOS: +$645 — profitable on BOTH periods
- 73% trade overlap with vScalpA but different outcomes due to SL=15 vs SL=50
- File: `strategies/scalp_vScalpB_MNQ.pine`
- Config: `MNQ_VSCALPB` in `engine/config.py`
- Validation: `strategies/OOS_DEEP_DIVE_FEB21.md` Steps 5-8

## MES Strategies

### v9.4 MES (Feb 15, updated Feb 20) — REPLACED by MES v2 (Feb 21)
- Params: SM(20/12/400/255) RSI(10/55/45) CD=15 SL=0 EOD=15:30 (stop HURTS MES)
- 6-month: 359 trades, 54.9% WR, PF 1.228, +$1,129 (Databento data, $1.25/side)
- 12-month: 733 trades, PF 1.027, +$339 (OOS -$849)
- SM threshold REJECTED — weak entries are profitable
- EOD 15:30 VALIDATED
- **REPLACED:** SM flip exit same problem as MNQ vWinners. TP=20 exit massively better.
- File: `strategies/scalp_v94MES.pine`
- Config: `MES_V94` commented out in `engine/config.py`
- Full research: `memory/mes_exit_research.md`

### MES v2 (Feb 21) — ACTIVE
- Replaces v9.4. Same SM engine (20/12/400/255), new exit + RSI tuning.
- Params: SM_T=0.0, RSI=12/55-45, CD=25, TP=20 ($100/contract), SL=35 ($175/contract), EOD=15:30
- 12-month: +$5,380, OOS +$4,665, IS +$715
- Why TP=20 (not TP=5 like MNQ): MES $5/pt so TP=20=$100; SM EMA=255 trends persist longer
- SL=35 caps worst-case (v9.4 had uncapped losses up to -60 pts)
- Commission fixed: 1.25/side (v9.4 Pine incorrectly had 0.52)
- No SM flip exit — exits ONLY on TP, SL, or EOD
- File: `strategies/scalp_MES_v2.pine`
- Config: `MES_V2` in `engine/config.py`
- Validation: `strategies/OOS_DEEP_DIVE_FEB21.md` Step 8

## Validation Results

### TradingView Validation (Feb 15)
- MNQ v11: TV 66 trades, PF 1.681 vs Python 57 trades, PF 2.297 — $16 PnL diff through Feb 12
- MES v9.4: TV 59 trades, PF 2.089 vs Python 55 trades, PF 2.318 — $20 PnL diff through Feb 12

### Cross-Instrument (Feb 14)
- MNQ: v11 walk-forward OOS PF 1.713, 6/7 months profitable, NOT overfit
- MES: v9.4 train/test PF 1.686, sweep configs ALL FAILED test
- Details: `memory/cross_instrument_analysis.md`

### SM Computation Verification (Feb 14)
- Python compute_smart_money() with AA params (20,12,400,255) matches AlgoAlpha TradingView: r=0.985
- Fast params (15,10,300,150) diverge: r=0.87 — do NOT use for backtesting
- v11 params (10/12/200/100) are the optimized set for MNQ

## Portfolio Weighting Analysis (Feb 26)

Six configurations tested to determine optimal portfolio composition. All 3 strategies run on same 12-month dataset.

| Config | vScalpA | vScalpB | MES v2 | P&L | Sharpe | MaxDD | PF |
|--------|---------|---------|--------|-----|--------|-------|-----|
| Baseline | 1 | 1 | 1 | $6,552 | 2.37 | -$1,388 | 1.48 |
| Drop A | 0 | 1 | 1 | $5,671 | 2.60 | -$1,065 | 1.52 |
| Scale B only | 0 | 2 | 1 | $6,625 | 2.64 | -$1,228 | 1.53 ← TARGET |
| Keep both, B 2x | 1 | 2 | 1 | $7,506 | 2.42 | -$1,596 | 1.49 |
| Keep both, B 3x | 1 | 3 | 1 | $8,460 | 2.41 | -$1,804 | 1.49 |
| Scale both 2x | 2 | 2 | 1 | $8,386 | 2.08 | -$1,967 | 1.42 |

**Key Findings:**

1. **vScalpB(2) + MES(1) is optimal** on risk-adjusted metrics: Sharpe 2.64, PF 1.53, MaxDD -$1,228
2. **vScalpA regime insurance is real but costly**: Provides +$1,654 profit in volatile months (Apr, Nov) but drawdown cost is $0.42 per dollar vs $0.17 for vScalpB
3. **vScalpB scales cleanly**: Sharpe remains 2.64 at 2x and only degrades to 2.41 at 3x, demonstrating no overfitting
4. **Never scale vScalpA**: Going from A(1) to A(2) with B(2) drops Sharpe from 2.64 to 2.08; worst MaxDD of all multi-strategy configs (-$1,967)
5. **Correlation matrix**: A-B 0.236, A-MES 0.179, B-MES 0.161 (all low, good diversification)

**IS-OOS Split Sensitivity** (5 split points tested):
- vScalpB passes Sharpe + PF at ALL 5 splits — robust and consistent
- vScalpA passes only at 40% IS split — edge is fragile under different train/test boundaries

**Decision**: Paper trade all 3 strategies through Feb 28, then drop vScalpA and go live with vScalpB(2) + MES(1).

Script: `backtesting_engine/strategies/weighted_portfolio_comparison.py`
