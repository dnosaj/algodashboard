# SM + RSI Parameter Sweep Results
## Date: Feb 14, 2026
## Data: Databento MNQ 1-min, 172,582 bars, Aug 17 2025 - Feb 12 2026

---

## Executive Summary

The v9.4 strategy was losing money over 6 months because the AlgoAlpha default SM params (20/12/400/255) are too slow for 1-min intraday trading. Sweeping 6,180 parameter combinations on 1-min bars found settings that are **profitable in 6 of 7 months** with PF 1.669 and +$4,567 net (vs baseline -$984).

| Metric | Baseline (production) | Winner | Delta |
|--------|----------------------|--------|-------|
| SM Params | 20/12/400/255 | **10/12/200/100** | Faster index + shorter EMA |
| RSI (5-min) | RSI(10) 55/45 | **RSI(8) 60/40** | Faster RSI + wider levels |
| Cooldown | 15 bars | **20 bars** | +5 bars patience |
| Stop Loss | 50 pts | **50 pts** | Same |
| Trades | 345 | 368 | +23 |
| Win Rate | 52.5% | **57.9%** | +5.4% |
| Profit Factor | 0.906 | **1.669** | +0.763 |
| Net P&L | -$984 | **+$4,567** | +$5,551 |
| Max Drawdown | -$2,163 | **-$567** | 74% less DD |

---

## Winning Configuration

```
SM Index Period:    10    (was 20 -- half the lookback)
SM Flow Period:    12    (unchanged)
SM Norm Period:   200    (was 400 -- but 200/300/400/500 all identical)
SM EMA Length:    100    (was 255 -- much shorter smoothing)

5-min RSI Length:   8    (was 10)
5-min RSI Buy:     60    (was 55 -- higher threshold = stronger momentum)
5-min RSI Sell:    40    (was 45 -- lower threshold = stronger momentum)

Cooldown:          20    (was 15 -- 20 min between trades)
Max Loss Stop:     50    (unchanged)
ATR Trail:        OFF    (not needed with these params)
1-min RSI Filter: OFF    (not needed for best combo)
```

### Why These Params Work

1. **SM index_period=10 vs 20**: The SM indicator accumulates buy/sell pressure over `index_period` bars. At 20, it's looking back 20 minutes -- too much smoothing for 1-min scalping. At 10, it reacts twice as fast to Smart Money direction changes, catching flips earlier and exiting losers sooner.

2. **SM ema_len=100 vs 255**: The EMA smoothing on PVI/NVI determines how quickly the "smart" vs "dumb" money separation adapts. 100 bars = ~1.5 hours of context vs 255 = ~4 hours. Shorter EMA means SM direction changes are detected faster.

3. **RSI(8) 60/40 vs RSI(10) 55/45**: Faster RSI with wider levels means we're entering on stronger momentum signals -- RSI has to push above 60 (not just 55) to trigger a long. This filters out weak signals at the cost of slightly fewer trades.

4. **Cooldown 20 vs 15**: An extra 5 minutes between trades avoids re-entering too quickly after a losing exit.

---

## Monthly Performance Comparison

### Winner: SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50

| Month | Trades | Win Rate | PF | Net P&L | Max DD |
|-------|--------|----------|-----|---------|--------|
| Aug 25 | 28 | 46.4% | 1.018 | +$7 | -$170 |
| Sep 25 | 68 | 61.8% | 1.519 | +$469 | -$338 |
| Oct 25 | 58 | 60.3% | 2.215 | +$1,068 | -$337 |
| Nov 25 | 64 | 57.8% | 1.532 | +$876 | -$497 |
| Dec 25 | 59 | 47.5% | 0.948 | -$80 | -$567 |
| Jan 26 | 59 | 61.0% | 2.041 | +$902 | -$284 |
| Feb 26 | 32 | 68.8% | 3.179 | +$1,326 | -$216 |
| **TOTAL** | **368** | **57.9%** | **1.669** | **+$4,567** | **-$567** |

### Baseline: SM(20/12/400/255) RSI(10/55/45) CD=15 SL=50

| Month | Trades | Win Rate | PF | Net P&L | Max DD |
|-------|--------|----------|-----|---------|--------|
| Aug 25 | 31 | 67.7% | 1.660 | +$271 | -$154 |
| Sep 25 | 51 | 49.0% | 0.580 | -$621 | -$666 |
| Oct 25 | 59 | 44.1% | 0.662 | -$716 | -$1,079 |
| Nov 25 | 60 | 43.3% | 0.901 | -$233 | -$747 |
| Dec 25 | 66 | 51.5% | 0.687 | -$593 | -$758 |
| Jan 26 | 55 | 65.5% | 1.777 | +$953 | -$414 |
| Feb 26 | 23 | 56.5% | 0.953 | -$44 | -$351 |
| **TOTAL** | **345** | **52.5%** | **0.906** | **-$984** | **-$2,163** |

### Month-by-Month Delta (Winner - Baseline)

| Month | PF Delta | P&L Delta |
|-------|----------|-----------|
| Aug 25 | -0.642 | -$264 (baseline better) |
| Sep 25 | +0.939 | +$1,090 |
| Oct 25 | +1.553 | +$1,784 |
| Nov 25 | +0.631 | +$1,109 |
| Dec 25 | +0.261 | +$513 |
| Jan 26 | +0.264 | -$51 (roughly equal) |
| Feb 26 | +2.226 | +$1,370 |

The winner sacrifices some Aug performance but massively improves Sep-Dec (the losing months). Aug is only slightly negative for the baseline anyway.

---

## Phase 1: SM Parameter Sweep (400 combos)

Fixed RSI 10/55/45, CD=15, SL=50. Swept all SM params.

### Key Finding: index_period=10 dominates

Every single top-20 combo has `index_period=10`. No other value comes close.

| index_period | Best Aug-Dec PF | Best Full PF | Profitable in Both? |
|-------------|----------------|-------------|-------------------|
| **10** | **1.255** | **1.364** | **Yes (28 combos)** |
| 15 | 1.053 | 1.116 | Yes (limited) |
| 20 | 0.771 | 0.906 | No |
| 25 | 0.639 | 0.755 | No |
| 30 | 0.546 | 0.659 | No |

### SM norm_period is insensitive
200, 300, 400, and 500 all produce identical results with index=10. The normalization window settles quickly at this scale.

### SM ema_len ranking
100 > 150 > 200 >> 255 >> 350. Shorter EMA = faster adaptation = better.

---

## Phase 2: RSI Sweep + 1-min RSI Filter (5,600 combos)

Top 10 SM param sets x 5 RSI lengths x 4 level pairs x 4 cooldowns x 7 1-min RSI configs.

### 4,848 combos profitable in both periods

The strategy is broadly robust once you fix the SM params.

### Best 5-min RSI configs

| Rank | RSI Length | Buy/Sell | Notes |
|------|-----------|----------|-------|
| 1 | 8 | 60/40 | Best absolute P&L |
| 2 | 14 | 50/50 | Best gMean PF (with 1m RSI filter) |
| 3 | 10 | 60/40 | Close to #1 |
| 4 | 8 | 55/45 | Slightly worse than 60/40 |

### 1-min RSI Filter Results

The 1-min RSI filter dramatically increases the number of profitable configurations:
- **With 1-min RSI ON: 4,138 profitable combos**
- **With 1-min RSI OFF: 710 profitable combos**

However, the absolute best configuration (PF 1.669) does NOT use the 1-min RSI filter. The filter helps weaker SM/RSI combos survive but isn't needed when the SM params are already optimal.

Best config WITH 1-min RSI filter:
```
SM(10/8/200/150) RSI5m(14/50/50) RSI1m(10/45/55) CD=15
281 trades, PF 1.419, +$2,312, DD -$674
Aug-Dec PF 1.214, Jan-Feb PF 2.978
```

This is a more conservative config: fewer trades, lower Aug-Dec PF, but extremely strong Jan-Feb performance. Could be useful as a "confirmation mode" setting.

---

## Phase 3: ATR Trailing Stop (180 combos)

Tested top 10 configs with ATR trail multipliers and stop loss levels.

### Finding: ATR trail does NOT help the winning config

| Config | PF | Net | Max DD |
|--------|-----|------|--------|
| Winner (no ATR, SL=50) | **1.669** | **+$4,567** | **-$567** |
| + ATR 2.5x20 (no SL) | 1.498 | +$3,327 | -$576 |
| + ATR 2.5x20 + SL=50 | 1.474 | +$3,205 | -$576 |
| + ATR 2.0x20 (no SL) | 1.441 | +$2,824 | -$573 |
| + ATR 1.5x20 (no SL) | 1.302 | +$1,778 | -$573 |

ATR trailing stops cut into profits without meaningfully improving drawdown (DD is already low at $567). The SM flip exit with faster SM params is already doing a good job of cutting losers.

### SL=50 is optimal

| Stop Loss | PF | Net | Max DD |
|-----------|-----|------|--------|
| SL=0 (none) | 1.503 | +$3,735 | **-$1,119** |
| **SL=50** | **1.669** | **+$4,567** | **-$567** |
| SL=75 | 1.520 | +$3,890 | -$686 |

SL=50 is clearly the best -- it's the sweet spot between catching catastrophic losers and not clipping normal trades.

---

## Next Steps

### 1. TradingView Validation (Critical)
Build Pine Script with SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50 and validate on TradingView against AlgoAlpha indicator. The Python backtest uses computed SM from Databento volume -- need to confirm AlgoAlpha with these params produces similar signals.

**IMPORTANT**: AlgoAlpha's default SM params on TradingView may not be adjustable to index=10/ema=100. If AlgoAlpha locks params, we'd need to:
- Check if AlgoAlpha has configurable settings
- Or compute SM natively in Pine (port compute_smart_money to Pine v6)

### 2. Forward Test
Run both configs side-by-side:
- Production: SM(20/12/400/255) RSI(10/55/45) -- the current known config
- Optimized: SM(10/12/200/100) RSI(8/60/40) -- the sweep winner

### 3. Cross-Instrument Validation
Test optimized params on MES, ES, MYM to confirm they generalize.

### 4. Overfitting Check
The sweep tested 6,180 parameter combinations. While the winner is consistent across months and the SM index_period=10 pattern is strong (dominates all top-20), we should:
- Run walk-forward validation (train on Aug-Nov, test on Dec-Feb)
- Check that the improvement isn't driven by a few lucky trades
- Verify on an independent data period when available

---

## Files

| File | Description |
|------|-------------|
| `v10_param_sweep.py` | Sweep script (Phase 1-4) |
| `v10_test_common.py` | Shared engine with 1-min RSI filter added |
| `PARAM_SWEEP_RESULTS.md` | This document |

---

## Raw Data

### Sweep Configuration
- **Phase 1**: 400 SM combos (5 index x 4 flow x 4 norm x 5 ema), fixed RSI 10/55/45
- **Phase 2**: 5,600 combos (10 SM x 5 RSI len x 4 levels x 4 cooldowns x 7 1-min RSI)
- **Phase 3**: 180 combos (10 configs x 6 ATR x 3 SL)
- **Total runtime**: 26.2 minutes on 172,582 1-min bars
- **Commission**: $0.52/side ($1.04 round trip) on MNQ ($2/point)
