# Late-Day Entry Analysis (Mar 3, 2026)

## Finding

V15 and vScalpB have OPPOSITE late-day patterns. V15 late entries are consistently
bad; vScalpB late entries are consistently good. MES_V2 is neutral.

Triggered by Mar 3 paper trading: both MNQ strategies SL'd at 14:36 ET (V15's worst hour).

## 12-Month Results by Entry Hour (ET)

### MNQ_V15 — Late entries clearly bad

| Hour (ET) | N | WR% | PF | Total$ | Avg$ | SL% |
|-----------|---|-----|-----|--------|------|-----|
| 10:00-10:59 | 169 | 84.0% | 1.189 | +$584 | +$3.46 | 15.4% |
| 11:00-11:59 | 127 | 86.6% | 1.207 | +$388 | +$3.06 | 13.4% |
| **12:00-12:59** | **148** | **91.2%** | **1.729** | **+$1,037** | **+$7.01** | **7.4%** |
| 13:00-13:59 | 131 | 81.7% | 0.823 | -$462 | -$3.53 | 16.8% |
| **14:00-14:59** | **104** | **81.7%** | **0.651** | **-$816** | **-$7.84** | **16.3%** |
| 15:00-15:59 | 86 | 79.1% | 0.924 | -$119 | -$1.38 | 11.6% |

Sweet spot: 12:00 hour. Cliff at 13:00+. Worst hour: 14:00-14:59.

### MNQ_VSCALPB — Late entries are the BEST (do NOT apply cutoff)

| Hour (ET) | N | WR% | PF | Total$ | Avg$ | SL% |
|-----------|---|-----|-----|--------|------|-----|
| 10:00-10:59 | 54 | 70.4% | 1.191 | +$158 | +$2.92 | 29.6% |
| 11:00-11:59 | 64 | 73.4% | 1.300 | +$214 | +$3.35 | 26.6% |
| 12:00-12:59 | 65 | 69.2% | 0.949 | -$44 | -$0.67 | 30.8% |
| 13:00-13:59 | 73 | 67.1% | 0.777 | -$251 | -$3.44 | 32.9% |
| **14:00-14:59** | **52** | **78.8%** | **2.075** | **+$410** | **+$7.88** | **21.2%** |
| **15:00-15:59** | **44** | **77.3%** | **2.260** | **+$467** | **+$10.61** | **22.7%** |

SM_T=0.25 filter selects higher-quality afternoon signals. Weak hours are 12:00-13:59.

### MES_V2 — Neutral late-day

| Split | Early PF | Late PF | Late $ |
|-------|----------|---------|--------|
| FULL | 1.397 | 0.973 | -$44 |

Late entries break even. Not worth gating.

## IS/OOS Validation (V15 and vScalpB)

### V15: Late entries bad on BOTH halves

| Period | Early PF | Late PF | Late $ |
|--------|----------|---------|--------|
| IS | 1.008 | **0.742** | -$569 |
| OOS | 1.419 | **0.786** | -$366 |

### vScalpB: Late entries good on BOTH halves

| Period | Early PF | Late PF | Late $ |
|--------|----------|---------|--------|
| IS | 1.103 | 1.810 | +$351 |
| OOS | 0.940 | 2.648 | +$526 |

## Why the Difference

V15 (SM_T=0.0) takes every SM signal. In the afternoon, market structure shifts to
mean-reversion — V15 catches weak signals that reverse. vScalpB (SM_T=0.25) filters
those out, and SL=15 cuts losers fast before they become expensive.

## Data Notes

- V15 CSV uses SL=50 (pre-Feb 26). Current live is SL=40. Entry patterns same; P&L magnitude slightly different.
- vScalpB and MES_V2 CSVs match current live config.
- Data: Feb 2025 — Feb 2026, `run_and_save_portfolio.py` output (Feb 23, 2026 run).
- Entry times converted UTC→ET with DST handling via `zoneinfo`.

## Pre-Filter Validation (Mar 3, 2026)

Post-filter ≠ pre-filter. Blocking entries changes cooldowns and episodes. Ran TRUE
pre-filter backtests with `entry_end_et` parameter (not post-hoc trade filtering).

### V15 with SL=40 — two cutoff candidates

| Cutoff | Split | Trades | WR% | PF | Net $ | MaxDD | Sharpe |
|--------|-------|--------|-----|-----|-------|-------|--------|
| Baseline (15:45) | FULL | 781 | 82.5% | 1.070 | +$880 | -$1,065 | 0.391 |
| Baseline (15:45) | IS | 397 | 80.4% | 0.996 | -$30 | -$1,065 | -0.025 |
| Baseline (15:45) | OOS | 384 | 84.6% | 1.168 | +$910 | -$652 | 0.882 |
| **14:00** | FULL | 585 | 83.1% | 1.142 | +$1,280 | -$654 | 0.763 |
| **14:00** | IS | 295 | 81.0% | 1.048 | +$242 | -$654 | 0.270 |
| **14:00** | OOS | 290 | 85.2% | 1.263 | +$1,038 | -$534 | 1.338 |
| **13:00** | **FULL** | **452** | **85.0%** | **1.326** | **+$2,042** | **-$542** | **1.648** |
| **13:00** | **IS** | **227** | **84.1%** | **1.305** | **+$1,009** | **-$404** | **1.568** |
| **13:00** | **OOS** | **225** | **85.8%** | **1.350** | **+$1,033** | **-$542** | **1.737** |

### Decision: 13:00 ET cutoff

13:00 is clearly better than 14:00:
- IS is borderline at 14:00 (PF 1.048, +$242) vs robust at 13:00 (PF 1.305, +$1,009)
- Sharpe 4x improvement over baseline (0.39 → 1.648)
- MaxDD halved (-$1,065 → -$542)
- Net P&L more than doubles (+$880 → +$2,042)

### Control check

vScalpB and MES_V2 trade counts identical across all cutoff runs (352 and 382 respectively).
Independent strategies, independent cooldowns — cutoff only affects V15.

## Implementation Status — IMPLEMENTED (Mar 3, 2026)

**Config**: `session_end_et="13:00"` on `MNQ_V15` in `live_trading/engine/config.py`

**Backtest engine**:
- `run_backtest_tp_exit()` gained `entry_end_et` param (default `NY_LAST_ENTRY_ET` for backward compat)
- `VSCALPA_ENTRY_END_ET = 13 * 60` in `generate_session.py`
- `VSCALPA_MAX_LOSS_PTS` updated 50→40 (was already 40 in live config)
- `run_and_save_portfolio.py` passes `entry_end_et` for V15 only

**Live engine**: No code changes needed. `strategy.py` already parses `session_end_et` into
minutes and gates entries via `in_session = self._session_start <= bar_mins <= self._session_end`.
EOD close uses separate `session_close_et` (default "16:00"), so V15 positions entered before
13:00 still exit at TP/SL/EOD normally.

**Do NOT apply to vScalpB** — late entries are its best (PF 2.17).
**Do NOT apply to MES_V2** — neutral.
