---
name: Periodic Volume Profile (PVP)
description: Weekly volume profile (VAH/POC/VAL) as higher-timeframe S/R levels. Same math as our daily VPOC — just weekly aggregation. Testing as forensic/gate feature.
type: reference
---

# Periodic Volume Profile (PVP)

## Source

- **TradingView built-in**: Premium feature, but algorithm is fully documented and replicable
- **Open-source Pine Script**: PtGambler's Periodic Volume Profile
- **Documentation**: https://www.tradingview.com/support/solutions/43000703071-periodic-volume-profile/
- **Strategy video**: Scott Taylor's 2-step PVP + session liquidity strategy

## Discovery Context

Jason found the Scott Taylor strategy video (Mar 14, 2026) using weekly PVP levels + session-based liquidity sweeps. The strategy trades away from value area levels and toward them as targets, combining with Asia/London/NY session H/L sweeps for entries.

## What It Does

Exactly the same as our daily volume profile, but aggregated over a full week:

1. Group all bars within a week (Mon-Fri)
2. Bin prices and accumulate volume per bin
3. POC = bin with highest volume
4. Value Area (70%) = expand above/below POC until 70% of volume is captured
5. VAH = top of value area, VAL = bottom

**We already have this algorithm** in `sr_prior_day_levels_sweep.py` (`compute_rth_volume_profile`). Weekly is a ~30-line aggregation change.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Period | Week | Aggregation window (can be day/week/month) |
| Value Area % | 70% | Standard |
| Row size | Auto or fixed ticks | Price bin granularity |

## Code

No new code needed — reuse `compute_rth_volume_profile` with weekly grouping instead of daily.

## Analysis

### Useful

- **5x volume = higher conviction**: Weekly POC has a full week of institutional volume behind it vs one day. More stable, less noise.
- **Complementary to daily levels**: Weekly is a different timeframe layer — could catch S/R that daily misses.
- **We already have the algorithm**: Zero implementation effort. Just change the aggregation window.
- **Proven concept**: VPOC+VAL is our only adopted S/R gate (MES, ~20% block rate). Weekly could extend this.

### Flawed

- **Wider zones on weekly**: Weekly range is larger, so VAH-VAL spread is wider. Proximity thresholds need re-calibration.
- **Levels change less frequently**: Only updates weekly — could be stale by Thursday/Friday.
- **The video's strategy is discretionary**: Variable TP targets + judgment calls on session sweeps. Doesn't map directly to our systematic fixed-TP approach.
- **Overlap with daily**: If weekly and daily VPOC are close, they're redundant. If they differ, which matters more?

### Related Indicators

- **Our daily VPOC/VAH/VAL gate** (MES): Same algorithm, daily window
- **VWAP**: Running mean price weighted by volume — continuous, not binned
- **Prior-day levels** (H/L): Not volume-weighted

### Correlation with Our Existing Indicators

- **Daily VPOC/VAL**: High expected correlation (weekly POC often near daily POC). Need to test whether weekly adds value beyond daily.
- **ADR gate**: Low correlation (range-based vs volume-based)

### Fit With Our System

**Gate candidate**: Block entries near weekly VPOC/VAH/VAL (same pattern as our daily gate for MES). Test whether weekly levels have stronger predictive power for SL clustering.

**NOT adopting the full video strategy** (discretionary, variable TP, session sweep entries). That's a different trading system.

## Status

**Testing** — added to ICT forensics for SL clustering analysis alongside daily levels and session H/L.

## Test Results

*(Being computed — added to ict_forensics.py re-run)*

## Backtest Priority

**Medium** — the weekly aggregation is trivial to add, and we already know daily VPOC+VAL works for MES. The question is whether weekly adds incremental value.
