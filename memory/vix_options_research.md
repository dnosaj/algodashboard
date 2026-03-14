# VIX & Options Data Research Plan

**Started**: Mar 3, 2026
**Status**: Phase 1 COMPLETE — no actionable filter found

## Goal

Determine if VIX levels, options flow, or derivatives data can improve
our SM+RSI scalping strategies — specifically as a pre-filter to avoid
bad trades or as a regime gate.

## Trading context

- Strategies already struggle on high-vol news days (FOMC: -$454, Retail Sales: -$265 for vScalpA)
- Regime detection has been tested and rejected before (London range, prior-day range, rolling volatility — all fail walk-forward)
- Any new signal must pass IS/OOS split-half validation before paper trading

## Phases

### Phase 1 — Historical VIX correlation (COMPLETE Mar 3, 2026)

**Script**: `backtesting_engine/strategies/vix_correlation_analysis.py`
**Data**: 1,476 trades (12 months) + daily VIX from Yahoo Finance (^VIX via yfinance)
**Charts**: `backtesting_engine/results/vix_analysis/` (5 PNGs + merged CSV)

#### Key findings

**1. VIX level vs trade P&L is NON-LINEAR (not "high VIX = bad")**

| VIX Quintile | Range | PF | Sharpe | P&L | Interpretation |
|---|---|---|---|---|---|
| Q1 | 13.5-15.8 | 0.981 | -0.23 | -$95 | Low VIX = losing |
| Q2 | 15.9-16.9 | 1.077 | 0.81 | +$401 | Marginal |
| Q3 | 16.9-18.7 | **1.612** | **5.64** | **+$2,963** | Sweet spot |
| Q4 | 19.0-22.3 | 0.966 | -0.45 | -$264 | Death zone |
| Q5 | 22.3-52.3 | **1.507** | **6.24** | **+$3,639** | High VIX = great |

**2. Strategy-specific patterns**
- **MES_V2**: Loves high VIX (Q5: +$2,525, Sharpe 6.28). Low VIX terrible (Q1: WR 49%, -$562).
- **MNQ_VSCALPB**: Death zone Q4 (VIX 19-22): SL rate 45%, WR 54.5%, -$369.
- **MNQ_V15**: Q3 best (+$1,220), Q4 worst (-$716).

**3. VIX day-over-day change matters more**
- Big drops (<-5%): PF 1.713, Sharpe 6.67 — best
- Big rises (>+5%): PF 1.370, Sharpe 4.12 — also good
- Flat/small moves: PF ~0.9, negative Sharpe — worst
- **Big VIX moves in either direction = profitable. Choppy = bad.**

**4. No simple threshold filter works**
- No VIX≤X threshold improved combined portfolio Sharpe (baseline 2.44).
- VIX≤15 on individual strategies showed high Sharpe but tiny N (35-68 trades).

**5. Correlations all weak**: |r| < 0.13 for all VIX metrics vs P&L.

**6. SL clustering**: Q4 (VIX 19-22) has highest SL rate (24.7%). vScalpB spikes to 45% in Q4.

#### Decision

**VIX as a simple pre-filter is NOT actionable.**
- The relationship is non-linear — can't use a threshold
- A Q4 band-exclusion (skip VIX 19-22) is too narrow and likely overfit
- VIX change is interesting but not predictive (you'd need to know today's change in advance)
- Follows the established pattern: simple regime detection doesn't survive walk-forward

**Phase 2 pivot**: Instead of VIX filtering, the more promising direction is
**intraday options data** (GEX, put/call flow) which changes within the session
and could provide real-time regime signals. This requires live data collection
(Phase 2) since there's no historical data to backtest against.

### Phase 2 — Live options data collection (logging only)

Prerequisites: Phase 1 shows VIX signal
**No strategy changes** — just collect and observe for 2+ weeks.

Data to collect (via existing DXLink streamer):
- `imp_volatility` from Candle events (already streaming, just not extracted)
- DXLink `Underlying` event for NQ/QQQ → real-time put/call ratio, front/back IV
- VIX quote via `$VIX.X` streamer symbol
- tastytrade REST `market-metrics` (IV rank, IV percentile, IV-HV spread)

Log alongside each trade entry/exit in session files.

### Phase 3 — 0DTE / GEX analysis

Prerequisites: Phase 1-2 show signal worth pursuing.

- Use QQQ equity options as proxy (avoids DXLink index option restrictions)
- Subscribe to Greeks + Summary for ATM ± 10 strikes on 0DTE expiry
- Compute simplified GEX: positive = mean-reverting (good for scalps), negative = trending
- Test as regime filter
- Heavy lift: ~100+ option symbol subscriptions

### Phase 4 — Backtest validation & pre-filter implementation

Prerequisites: Phase 3 confirms signal.

- Build winning signal into backtest engine as pre-filter
- IS/OOS split-half validation (same rigor as SM threshold, TPX gate)
- Paper trade the filtered strategy
- If Sharpe/PF improves on OOS → add to live engine

## Available data infrastructure

### What tastytrade/DXLink CAN stream (not currently used)

| Event | Fields | Use case |
|-------|--------|----------|
| Greeks | delta, gamma, theta, vega, rho, IV | Per-strike options analytics |
| TheoPrice | price, underlying_price, delta, gamma | Model-free theoretical pricing |
| Underlying | put_call_ratio, call_volume, put_volume, front/back IV | Aggregate options sentiment |
| Summary | open_interest, daily OHLC | OI changes per strike |
| TimeAndSale | price, size, aggressor_side | Order flow analysis |
| Trade | price, size, day_volume | Tick-level trades |

### What's already streaming but unused

- `imp_volatility` on Candle events
- `bid_volume` / `ask_volume` on Candle events

### Key constraints

- VIX index not available on Databento (no CBOE dataset). Use yfinance for daily, DXLink `$VIX.X` for live.
- Index options (SPX/NDX) may be restricted on DXLink. Use QQQ/SPY equity options as proxy.
- No historical options data in our stack. Phase 1 uses VIX as proxy. Phase 2+ requires live collection.
- GEX requires subscribing to 100+ individual option symbols simultaneously.

## Files

| File | Purpose |
|------|---------|
| `backtesting_engine/strategies/vix_correlation_analysis.py` | Phase 1 analysis script |
| `memory/vix_options_research.md` | This plan (update with findings) |
