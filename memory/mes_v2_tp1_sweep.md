---
name: MES v2 TP1 Sweep & Red Team (Mar 11)
description: Red team of VIX/delay gates (both FAIL), TP1 sweep finding TP1=6 optimal, breakeven escape rejected
type: project
---

# MES v2 TP1 Sweep & Red Team (Mar 11, 2026)

## Context

Mar 11 trading session: -$577 portfolio, -$552 from MES alone. Two MES trades:
1. LONG ~6771→SL (-$355, 2 contracts, 15 bars) — full SL, MFE only +0.75 pts. No TP1 would have helped.
2. LONG 6777→BE_TIME 6757.75 (-$198, 2 contracts, 76 bars) — stale exit, wandered 75 bars without resolution

Motivated comprehensive investigation: red team prior gate findings, analyze TP1 fill rates, and sweep for better partial exit parameters.

## Red Team: VIX [20-25] Gate — FAIL

Prior sweep (Mar 11 morning) found VIX [20-25] as STRONG PASS for MES (IS PF +7.6%, OOS PF +8.4%).

**Bootstrap significance test**: 10,000 random samples of same size as blocked trades. Result: **p=0.29** — blocked trades are NOT statistically worse than random. The VIX band just happened to contain some bad trades, but any random band of similar size would show similar "improvement."

**Alternative band test**: Tested all 5-pt VIX bands [10-15] through [35-40]. The [11-16] band showed *higher* dPF than [20-25], destroying the "death zone" narrative. The pattern is noise.

## Red Team: Entry Delay +30min — FAIL

Prior sweep found entry delay +30-45min as MARGINAL PASS (OOS PF +8-12%).

**Bootstrap significance test**: **p=0.33** — not significant. 67% of MES SLs happen 10:00-11:00 ET, but blocking early entries also blocks early *winners*.

**IS/OOS divergence**: Stacking VIX+delay: IS PF *decreases* (-7.4%) while OOS increases. Classic overfitting signal — filter fits OOS noise, not a real pattern.

## TP1 Sweep (2-contract simulation)

Full bar-by-bar simulation with 2 contracts: TP1 partial → SL-to-BE → runner to TP2=20/SL/BE_TIME/EOD.

### The TP1=10 Problem

- **Only 39% of trades reach +10 pts** for TP1 partial fill
- **78% of SLs have MFE < 10 pts** — both contracts eat full SL (-$387 avg vs -$193 at 1 contract)
- **55% of trades (298/545) exit stale via BE_TIME** with avg -$5.9 — biggest drag

### TP1 Sweep Results (FULL period, 2 contracts)

| TP1 | Net     | PF    | Sharpe | WR    | TP1 Fill% | MaxDD    |
|-----|---------|-------|--------|-------|-----------|----------|
| 3   | $5,525  | 1.270 | 1.27   | 58.7% | 77%       | -$1,430  |
| 4   | $6,095  | 1.303 | 1.46   | 58.0% | 70%       | -$1,295  |
| 5   | $6,305  | 1.313 | 1.51   | 57.1% | 65%       | -$1,310  |
| **6** | **$6,780** | **1.341** | **1.63** | 56.5% | **60%** | **-$1,175** |
| 7   | $6,490  | 1.325 | 1.53   | 55.8% | 54%       | -$1,225  |
| 8   | $6,215  | 1.305 | 1.44   | 55.2% | 48%       | -$1,305  |
| 10  | $5,875  | 1.239 | 1.27   | 54.5% | 39%       | -$1,420  |

### IS/OOS Split

| TP1 | IS PF | OOS PF | IS Sharpe | OOS Sharpe |
|-----|-------|--------|-----------|------------|
| 6   | 1.271 | 1.430  | 1.42      | 1.89       |
| 10  | 1.195 | 1.298  | 1.05      | 1.53       |

TP1=6: OOS PF > IS PF — no overfitting signal. Consistent across both halves.

### Why TP1=6 Works

1. **TP1 fill rate 60% vs 39%** — 21% more trades get the partial exit de-risk
2. **Converts BE_TIME/BE_TIME trades** (-$42 avg) into TP1/SL_BE trades (+$26 avg) — $68/trade swing on ~60 trades
3. **SL damage reduction**: When TP1 fills, SL moves to breakeven — runner SL is risk-free
4. **NOTE**: Today's Trade 1 had MFE +0.75 pts — TP1=6 would NOT have helped. The benefit is statistical across the population, not guaranteed on any individual trade.

### Exit Pattern Breakdown (TP1=6)

| Exit Pattern (C1/C2) | Count | Avg P&L | Description |
|-----------------------|-------|---------|-------------|
| TP1/TP2               | 98    | +$127   | Best case: both TPs hit |
| TP1/SL_BE             | 51    | +$26    | TP1 hit, runner stopped at breakeven |
| TP1/BE_TIME           | 42    | +$18    | TP1 hit, runner timed out near entry |
| SL/SL                 | 121   | -$387   | Worst case: both contracts full SL |
| BE_TIME/BE_TIME       | 183   | -$42    | Stale: neither contract resolved |
| TP1/EOD               | 28    | +$8     | TP1 hit, runner closed at EOD |

## Breakeven Escape — REJECTED

Tested closing trades when |unrealized| <= N pts after M bars (before TP1 fills).

**All configs showed IS/OOS divergence** — hurts IS, helps OOS. Same red flag pattern as VIX gate. The TP1 reduction already captures most of the benefit (converts BE_TIME trades into TP1 partials).

| Escape Config | IS PF | OOS PF | IS dPF | OOS dPF |
|---------------|-------|--------|--------|---------|
| 3pts after 30 | 1.245 | 1.445  | -2.1%  | +1.0%   |
| 5pts after 45 | 1.260 | 1.460  | -0.9%  | +2.1%   |
| 3pts after 45 | 1.255 | 1.450  | -1.3%  | +1.4%   |

Not implementing — same IS/OOS divergence red flag pattern seen with VIX gate.

## Decision

- **ADOPTED**: TP1=6 for MES v2 (was TP1=10). Config: `partial_tp_pts=6`.
- **REJECTED**: VIX [20-25] gate for MES (p=0.29, not significant).
- **REJECTED**: Entry delay +30min for MES (p=0.33, not significant).
- **REJECTED**: Breakeven escape (IS/OOS divergence).

## Impact

- PF: 1.239 → 1.341 (+8.2%)
- Sharpe: 1.27 → 1.63 (+28%)
- Net: +$5,875 → +$6,780 (+15.4%)
- TP1 fill rate: 39% → 60%
- MaxDD: -$1,420 → -$1,175 (17% better)

Config updated in `config.py`. Takes effect on next engine restart.
