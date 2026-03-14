# RedK TPX Research (Feb 23, 2026)

## What TPX Is

RedK Trader Pressure Index — measures buying vs selling pressure from bar
structure (high-to-high, low-to-low movement normalized by 2-bar range).
WMA smoothed. TPX > 0 = bullish (green), TPX < 0 = bearish (red).

- **Not volume-based** — purely OHLC-derived, same family as RSI
- Pine source: `strategies/RedK_TPX_v5.pine`
- Python port: `backtesting_engine/strategies/compute_tpx.py`
- `compute_tpx(highs, lows, length=7, smooth=3)` → (tpx, avgbulls, avgbears)

## What We Tested

### Test 1: Binary agree/disagree at entry bar (`tpx_filter_test.py`)
- Checked if TPX sign at entry bar predicts trade outcome
- **Result: No useful signal.** Disagree trades performed about the same
  (or better for vScalpB) than agree trades. TPX value at a single bar
  doesn't predict the trade.

### Test 2: TPX as regime gate — pre-filter (`tpx_regime_test.py`)
- Only allow LONGs in TPX green zones (>0), SHORTs in red zones (<0)
- Swept 7 param combos: L=7/S=3 through L=30/S=12
- **This is the right way to use it** — wide stable zones as directional regime

#### Results by strategy:
- **vScalpA**: Marginal. Best at L=10 S=4 (+$150, PF 1.048→1.064). Slower hurts.
- **vScalpB**: Strong quality improvement. L=30 S=12: same P&L, PF 1.22→1.42,
  Sharpe 1.29→2.20, MaxDD -$358→-$315. Cuts trades from 352 to 203.
- **MES v2**: Every setting hurts. Do NOT use TPX gate on MES.

### Test 3: Split-half stability for vScalpB L=30 S=12 (`tpx_split_check.py`)

| Metric | IS Baseline | IS Gated | OOS Baseline | OOS Gated |
|--------|-------------|----------|--------------|-----------|
| Trades | 176 | 108 | 176 | 95 |
| WR | 72.7% | 76.9% | 71.6% | 68.4% |
| PF | 1.277 | 1.657 | 1.168 | 1.194 |
| Sharpe | 1.571 | 3.200 | 0.993 | 1.120 |
| MaxDD | -$358 | -$205 | -$293 | -$216 |
| Net $ | +$592 | +$736 | +$361 | +$238 |

- **PF improved on both halves: PASS**
- **Sharpe improved on both halves: PASS**
- **MaxDD improved on both halves: PASS**
- IS effect is stronger (trending market). OOS still positive but weaker.
- OOS net P&L drops $124 (fewer trades), but per-trade quality improves.

## Conclusion

TPX regime gate is a **trade quality filter**, not a profit booster. For vScalpB
it consistently improves PF, Sharpe, and MaxDD across both market halves by
filtering out trades that go against the broader pressure regime.

## Next Steps

1. **Paper trade vScalpB with and without TPX gate** — compare after 1-2 months
2. If validated, add TPX computation to live engine for vScalpB only
3. Could sweep more TPX params later, but L=30 S=12 is the candidate
4. Do NOT apply to MES v2 or vScalpA
