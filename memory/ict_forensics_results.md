# ICT Trade Forensics — Results (Mar 14, 2026)

## Overview

Mapped 1,123 trades (12.8 months, 4 strategies with production gates) against ICT structural features on 5-min and 1-min timeframes. Script: `backtesting_engine/strategies/ict_forensics.py`. Full output: `backtesting_engine/results/ict_forensics_v3_output.txt`.

## Baseline

1,123 trades, WR 73.5%, PF 1.58, SL rate 13.7%

## Gate Candidates (ranked by effect size)

| # | Feature | WR | WR Delta | PF | Sample | Status |
|---|---------|------|----------|------|--------|--------|
| 1 | **Inside order block zone** | 36.4% | -37.1pp | -- | 11 | Monitor — huge effect but N=11 |
| 2 | **Near weekly VAL (<10pts)** | 50.0% | -23.5pp | 0.622 | 22 | Monitor — strong but N=22 |
| 3 | **Near London H/L (<5pts)** | 61.3% | -12.2pp | 0.924 | 137 | **TEST NEXT** — good sample, PF < 1 |
| 4 | Near pre-NY H/L (<5pts) | 69.4% | -4.1pp | 1.284 | 85 | Weak |
| 5 | Near Asia H/L (<5pts) | 70.2% | -3.3pp | 1.391 | 47 | Weak |
| 6 | Sweep against entry direction | 70.4% | -3.1pp | 1.395 | 226 | Mild, not standalone |
| 7 | Outside weekly VA | 71.8% | -1.7pp | 1.523 | 373 | Marginal |

## Features That Confirm Our Edge (NOT gates — positive signals)

| Feature | WR | PF | Sample | Interpretation |
|---------|------|------|--------|----------------|
| Near weekly VPOC (<10pts) | 90.5% | 3.891 | 42 | VPOC is our sweet spot — institutional volume magnet |
| Premium zone entries (longs) | 76.7% | -- | 412 | Our system is momentum, not mean-reversion |
| Inside weekly VA | 74.3% | 1.823 | 750 | Better than outside VA |
| OTE zone (62-79% fib) | 75.3% | 1.777 | 73 | Slight edge, small sample |
| Entries against 5-min trend | 74.5% | 1.740 | -- | SM+RSI catches reversals well |
| Not near any session H/L | 76.6% | 1.796 | 829 | Clean entries outperform |

## Features That Don't Discriminate

| Feature | Finding | Why |
|---------|---------|-----|
| FVGs (5-min) | Too short-lived (~2 bar median) | NQ fills gaps within minutes |
| Liquidity sweeps | Too common (>50% of trades) | Not discriminating on NQ 1-min |
| Confluence count | No monotonic improvement | Features are mostly independent |
| BOS vs MSS recency | Mild effect | SM already captures momentum shifts |

## Per-Strategy Highlights

- **vScalpA**: London H/L entries have WR 58% vs 86% baseline. Strongest per-strategy signal.
- **vScalpB**: Filter-resistant as expected. No ICT feature significantly affects it.
- **vScalpC**: OB proximity matters (low WR near opposing OBs). Structure exit may already handle this.
- **MES v2**: Weekly VAL proximity is dangerous (PF < 1 near VAL). Daily VPOC+VAL gate partially covers this.

## Key Insight

Our SM+RSI system is momentum-based, not mean-reversion. ICT concepts designed for pullback entries (buy in discount, OTE zone) are neutral or positive for us because we're entering breakouts. The value of ICT for our system is **defensive** — identifying zones where our momentum entries are more likely to fail (OBs, London H/L, weekly VAL).

## Code Audit (Mar 14)

Three bugs found and fixed:
1. BOS/MSS `bars_ago` used wrong variable (`s_idx` → `t_idx`) — corrected
2. Session sweep mixed directions — split into bullish/bearish with direction parameter
3. MNQ bin_width was 5 (should be 2 matching daily) — corrected

All weekly PVP and session H/L proximity numbers verified clean post-audit.

## Next Steps

1. **Test London H/L as entry gate** — IS/OOS validation, same methodology as other gates
2. **Monitor OB and weekly VAL** — accumulate more sample before testing
3. **Dashboard overlays** — show weekly VPOC/VAH/VAL + session H/L on price chart for visual confirmation
