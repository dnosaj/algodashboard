# Robust Portfolio Rollout Plan

**Date**: March 18, 2026
**Status**: Ready to implement
**Account**: $150K, mini contracts

## Portfolio Design

Robust configs as primary (higher size), vScalpC current as only satellite (earned it — 0 losing years across 3 years).

| Strategy | Config | Contracts | SM Params | Exits | Notes |
|----------|--------|-----------|-----------|-------|-------|
| vScalpA | ROBUST | 2 | SM(12/12/200/80) T=0.0 | TP=7 SL=40 CD=20 | Index 10→12, EMA 100→80 |
| vScalpB | CURRENT | 1 | SM(10/12/200/100) T=0.25 | TP=3 SL=10 CD=20 | No change |
| vScalpC | ROBUST | 3 (scalp 2, run 1) | SM(12/12/200/80) T=0.0 | TP1=10 TP2=30 SL=30 CD=20 | Scalp 2 at TP1, 1 runner. +$40 locked after TP1. |
| vScalpC-SAT | CURRENT | 1 | SM(10/12/200/100) T=0.0 | TP1=7 TP2=25 SL=40 CD=20 | Satellite, 0 losing years |
| MES v2 | ROBUST | 2 | SM(20/14/400/300) T=0.25 | TP1=8 TP2=20 SL=35 CD=25 | SM_T 0→0.25, flow 12→14, EMA 255→300 |
| RSI TL | ROBUST | 2 | N/A (RSI entry) | TP1=7 TP2=25 SL=35 CD=30 | RSI(10), lb_right=4, min_spacing=15, no SM filter |

All MNQ: RSI(8/60/40) on 5-min mapped to 1-min, cutoff 13:00 ET (except vScalpB no cutoff)
MES: RSI(12/55/45), cutoff 14:15 ET, BE_TIME=75
RSI TL: entry 9:30-13:00 ET, SL→BE after TP1

## Max simultaneous exposure

| Instrument | Strategies | Max contracts |
|------------|-----------|---------------|
| MNQ | A(2) + B(1) + C-robust(3) + C-sat(1) + RSI_TL(2) | 9 |
| MES | MES-robust(2) | 2 |

At $150K: MNQ margin ~$1,900/contract × 9 = $17,100. MES margin ~$1,500/contract × 2 = $3,000. Total ~$20,100 margin (13.4% of account). Very comfortable.

## Operational changes needed

| Setting | Current | New | Why |
|---------|---------|-----|-----|
| max_position_size | 10 | 12 | 9 MNQ possible |
| max_daily_loss | $800 | $1,200 | Worst case: A(2×$80) + B($20) + C-rob(3×$60) + C-sat(1×$80) + RSI(2×$70) + MES(2×$175) = $1,070 |
| MNQ_V15 daily limit | $100 | $200 | 2 contracts |
| MNQ_VSCALPC daily limit | $200 | $400 | 3 contracts |
| MES_V2 daily limit | $400 | $400 | Same (2 contracts but tighter SL) |

## 3-Year validated performance (backtested)

### MNQ Robust strategies:
| Strategy | Y1 | Y2 | Y3 | 3Y Total |
|----------|-----|-----|-----|---------|
| vScalpA robust (2x) | +$660 | +$158 | +$5,886 | +$6,704 |
| vScalpB current (1x) | +$215 | -$200 | +$1,536 | +$1,551 |
| vScalpC robust (3x) | +$8,685 | +$9,597 | +$10,977 | +$29,259 |
| vScalpC-sat current (1x) | +$480 | +$1,785 | +$6,111 | +$8,375 |
| RSI TL robust (2x) | +$4,382 | +$6,130 | +$7,434 | +$17,946 |
| MES robust (2x) | +$2,952 | +$4,030 | +$4,750 | +$11,732 |
| **TOTAL** | **+$17,374** | **+$21,500** | **+$36,694** | **+$75,567** |

## Implementation steps

1. Create new strategy configs in config.py:
   - MNQ_V15_ROBUST (SM 12/12/200/80, entry_qty=2)
   - MNQ_VSCALPC_ROBUST (SM 12/12/200/80, TP1=10/TP2=30/SL=30, entry_qty=3)
   - MNQ_VSCALPC_SAT (current params, entry_qty=1) — rename current
   - MES_V2_ROBUST (SM 20/14/400/300, SM_T=0.25, TP1=8, entry_qty=2)
   - MNQ_RSI_TL_ROBUST (RSI=10, lb_right=4, min_spacing=15, TP2=25, SL=35, entry_qty=2, remove SM filter)
2. Update DEFAULT_CONFIG and INSTRUMENT_CONFIGS in run.py
3. Update operational limits (max_position_size, max_daily_loss, per-strategy limits)
4. Remove SM-aware tighter SL from rsi_trendline_strategy.py
5. Paper trade for 2 weeks minimum
6. Review performance vs backtest expectations
7. Go live

## Risk notes
- Paper trade first — backtest fills at next-bar-open, live may have slippage
- vScalpC at 3 contracts is the largest single position — one full SL = $180
- Monitor correlation between vScalpC robust and vScalpC satellite — they share the same RSI/cutoff, different SM and exits
- RSI TL is the newest and least proven — 2 contracts is appropriate, not more
