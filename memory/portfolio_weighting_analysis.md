# Portfolio Weighting Analysis

## Current Target: A(1) + B(1) + MES(1) — decided Mar 4, 2026

### Why this changed from the Feb 26 recommendation

The Feb 26 analysis recommended dropping vScalpA and running vScalpB(2)+MES(1).
That was based on the OLD V15 (PF 1.07, Sharpe 0.39, +$880). De-risking V15 with
the 13:00 ET cutoff + SL=40 made it a legitimately strong strategy (PF 1.45,
Sharpe 2.22, +$2,042), completely changing the portfolio math.

### Mar 4 analysis — with de-risked V15 + MES BE_TIME=75

Individual strategy metrics (12 months, all with current params):

| Strategy | Trades | WR | PF | P&L | Sharpe | MaxDD |
|----------|--------|-----|------|---------|--------|---------|
| vScalpA (SL=40, 13:00 cutoff) | 452 | 85.0% | 1.45 | +$2,042 | 2.22 | -$513 |
| vScalpB | 352 | 72.2% | 1.22 | +$954 | 1.53 | -$358 |
| MES v2 (BE_TIME=75) | 519 | 55.9% | 1.33 | +$4,015 | 1.67 | -$1,169 |

IS/OOS splits — all three positive on BOTH halves:

| Strategy | IS PF | IS P&L | OOS PF | OOS P&L |
|----------|-------|--------|--------|---------|
| vScalpA | 1.305 | +$1,009 | 1.35 | +$1,033 |
| vScalpB | 1.277 | +$592 | 1.168 | +$361 |
| MES v2 | 1.394 | +$2,688 | 1.242 | +$1,328 |

### Portfolio comparison — 6 configurations

| Config | P&L | Sharpe | Sortino | MaxDD | PF | Lose Mo |
|--------|-----|--------|---------|-------|-----|---------|
| **A(1)+B(1)+MES(1) Baseline** | **$7,011** | **3.05** | **3.12** | **-$1,057** | **1.67** | 4 |
| Drop A: B(1)+MES(1) | $4,969 | 2.61 | 2.77 | -$1,141 | 1.55 | 5 |
| Scale B: B(2)+MES(1) | $5,923 | 2.60 | 2.68 | -$1,198 | 1.56 | 4 |
| A(1)+B(2)+MES(1) | $7,965 | 2.98 | 2.97 | -$1,233 | 1.66 | 4 |
| A(1)+B(3)+MES(1) | $8,919 | 2.86 | 2.74 | -$1,430 | 1.63 | 4 |
| A(2)+B(2)+MES(1) | $10,008 | 3.05 | 2.77 | -$1,605 | 1.68 | 4 |

### Decision rationale

**Baseline A(1)+B(1)+MES(1) wins on risk-adjusted metrics:**
- Highest Sharpe (3.05) and Sortino (3.12)
- Smallest MaxDD (-$1,057)
- Only 4 losing months

Scaling vScalpB to 2x adds $954 P&L but worsens Sharpe (-0.07), Sortino (-0.15),
and MaxDD (-$176). Adding V15 to B(2)+MES(1) adds $2,042 P&L and improves Sharpe
(+0.38), but Baseline still has the best risk-adjusted profile.

### Correlations (daily P&L)

| | A | B | MES |
|---|------|------|------|
| A | 1.00 | 0.26 | 0.16 |
| B | 0.26 | 1.00 | 0.18 |
| MES | 0.16 | 0.18 | 1.00 |

Low correlations across all pairs — good diversification.

### Monthly breakdown — vScalpA marginal contribution

Adding vScalpA to B(2)+MES(1):
- Helped in **9 of 13 months** (total: +$2,357)
- Hurt in **4 months** (total: -$314)
- Net marginal: +$2,042

Best months for vScalpA: Apr 2025 (+$499), Nov 2025 (+$484), Jan 2026 (+$432)
Worst months: Jul 2025 (-$140), Oct 2025 (-$102)

### Dashboard sizing controls

User can adjust contract sizes per-strategy in real time via dashboard:
- MES_V2: `Entry: [-][2][+]  TP1: [-][1][+]`
- MNQ strategies: `Qty: [-][1][+]`
- Blocked while positioned (prevents mid-trade confusion)
- Cleared on Force Resume and daily reset

This allows scaling up intraday when conditions are favorable without changing
the baseline config.

### Scaling notes

- vScalpB scales cleanly: B(2) Sharpe 2.60, B(3) Sharpe 2.86 — barely degrades
- vScalpA scaling to 2x: Sharpe drops from 3.05 to 3.05 (tied) but MaxDD worsens -$548
- Never scale all strategies simultaneously without reviewing MaxDD impact

---

## Historical: Feb 26 analysis (OLD — superseded)

Based on old V15 (PF 1.07, +$880, no entry cutoff):
- Baseline (A=1, B=1, M=1): +$6,552, Sharpe 2.37, MaxDD -$1,388
- Target was B(2)+MES(1): +$6,625, Sharpe 2.64, MaxDD -$1,228
- vScalpA cost $0.42 per dollar of profit in added drawdown
- Recommendation was to drop vScalpA

This recommendation is **no longer valid** — de-risked V15 reverses the conclusion.
