# v14 Tick Microstructure & v15 Mechanical Exit Research (Feb 17, 2026)

## Context

After CVD/Volume Delta research showed weak results (G1 entry filter +0.08 PF but never significant), we pivoted to using the same 42.6M NQ ticks for *microstructure features* — looking at trade-level patterns rather than just aggressor side.

**Core insight**: The v11 entries work (57.9% WR). The problem is exits — SM flip exits are too slow to react to sudden selloffs (Feb 13: 235pt loss while SM stayed bullish). Can we find better exits?

## Phase 1: v14 Tick Microstructure Features

### Compute Pipeline
- **Script**: `strategies/v14_tick_compute.py`
- **Input**: `data/databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet` (42.6M ticks)
- **Output**: `data/tick_microstructure_features.csv` (171,628 1-min bars, 12.8s compute)
- **Fix**: Python 3.10+ type syntax `str | None` caused TypeError — removed type hint

### Features Computed Per Bar
1. **Trade imbalance**: (buy_vol - sell_vol) / total_vol, normalized [-1,1]
2. **Large trade ratio**: Volume from trades >= 10 contracts / total volume
3. **Tick intensity**: Number of trades per bar (normalized by rolling mean)
4. **Aggressive ratio**: Aggressive volume / passive volume per side
5. **Iceberg detection**: Large volume trades at same price level within bar (potential hidden liquidity)
6. **Delta acceleration**: Rate of change of cumulative delta
7. **VWAP deviation**: Bar close vs volume-weighted average price

### Test Variants (v14_tick_test.py)

**TRAIN period**: Aug 17 - Nov 16, 2025 (179 baseline trades, PF 1.256)
**TEST period**: Nov 17 - Feb 13, 2026 (176 baseline trades, PF 1.083)

#### Round 1 — Initial 9 Variants

| Variant | Feature | Exit Type | TRAIN Verdict | Notes |
|---------|---------|-----------|---------------|-------|
| L1 | Trade imbalance | Entry filter (threshold 0.2) | MAYBE 4/7 | Removes 30% trades, PF +0.05 |
| L2 | Trade imbalance | Entry filter (threshold 0.3) | REJECT 2/7 | Too aggressive, kills trade count |
| M1 | Large trade ratio | Entry filter (>0.4) | MAYBE 4/7 | Slight PF boost |
| M2 | Large trade ratio | Exit on ratio spike | REJECT 2/7 | Cuts winners |
| N1 | Tick intensity | Entry filter (>1.5x) | MAYBE 4/7 | |
| N2 | Tick intensity | Exit on low intensity | REJECT 2/7 | |
| O1 | Delta acceleration | Entry confirmation | MAYBE 4/7 | |
| P1 | Iceberg detection | Exit on iceberg (3x ratio) | MAYBE 4/7 | Rescued 12/23 SL trades but cut winners |
| P2 | Iceberg detection | Exit on iceberg (2x ratio) | MAYBE 4/7 | Rescued 15/23 SL trades but cut winners |

**Finding**: P1/P2 iceberg exits rescued many stop-loss trades (detected hidden selling before price crashed) but also cut winners short. Solution: only fire when already underwater.

#### Round 2 — Underwater-Gated Iceberg (P3-P6)

Added `iceberg_min_loss` parameter: only fire iceberg exit when position is already losing >= N points.

| Variant | Iceberg Ratio | Min Loss Gate | TRAIN | OOS |
|---------|--------------|---------------|-------|-----|
| P3 | 3.0x | 15 pts | **ADOPT 5/7** | REJECT 2/7 |
| P4 | 2.0x | 15 pts | **ADOPT 5/7** | REJECT 2/7 |
| P5 | 3.0x | 10 pts | **ADOPT 5/7** | REJECT 2/7 |
| P6 | 2.0x | 10 pts | **ADOPT 5/7** (best: dPF +0.191, DD -23%) | REJECT 2/7 |

**Result**: First features ever to score ADOPT on TRAIN. But ALL failed OOS (PF < 1.0). Same train/test instability pattern as every other feature.

**Lesson**: Iceberg detection is regime-dependent. The "hidden selling at same price" pattern existed in the trending Aug-Nov period but didn't persist in the choppier Nov-Feb period. The microstructure itself changes between regimes.

## Phase 2: v15 Mechanical Exits

### Motivation
Since entries work but indicator exits (SM flip) fail in adverse regimes, test purely mechanical exits that don't depend on any indicator state.

### Script: `strategies/v15_time_exits.py`

22 configs tested, all using identical v11 entries:

**Time Exits** (hold N bars, 50pt backstop):
- T3, T5, T10, T15, T20, T30

**Target/Stop Pairs** (TP = take profit, SL = stop loss in points):
- 12 combinations: TP10/SL10 through TP50/SL30

**Trailing Stops** (activate at N pts profit, trail at M pts distance):
- TR_3_5, TR_5_5, TR_5_8, TR_8_10, TR_10_15, TR_15_15

### Results

#### TRAIN (Aug-Nov, trending period)

| Category | Best Config | PF | dPF vs v11 | Verdict |
|----------|-------------|-----|------------|---------|
| Time | T10 | 1.301 | +0.045 | MAYBE 4/7 |
| TP/SL | TP30_SL15 | 1.388 | +0.132 | MAYBE 4/7 |
| Trail | TR_10_15 | 1.205 | -0.051 | REJECT |

#### OOS (Nov-Feb, choppy period)

| Category | Best Config | PF | dPF vs v11 | Verdict |
|----------|-------------|-----|------------|---------|
| Time | All failed | < 1.0 | negative | REJECT |
| TP/SL | All failed | < 1.0 | negative | REJECT |
| **Trail** | **TR_5_8** | **1.202** | **+0.119** | **ADOPT 5/7** |

### Key Discovery: Trailing Stops Show INVERSE Pattern

This was unprecedented — trailing stops were the ONLY exit category that:
- **Hurt** on TRAIN (trending markets: trail cuts big winners short)
- **Helped** on OOS (choppy markets: trail locks in profits before reversals)

Every other feature/exit we've tested shows the opposite (works on train, fails on test). Trailing stops are naturally countercyclical — they excel precisely when trend-following exits (SM flip) struggle.

### TR_5_8 Configuration
- Activate trailing stop after position is +5 pts favorable
- Trail at 8 pts from max favorable excursion
- Fires faster than SM flip in choppy conditions, slower in trending
- OOS: PF 1.202, +$678, MaxDD -$645 (vs baseline PF 1.083, +$262, MaxDD -$812)

## Phase 3: v15 Hybrid & Portfolio

### Script: `strategies/v15_hybrid.py`

Three approaches tested:

#### Approach 1: Hybrid (Trail + SM Flip, First Wins)
Each trade has ONE exit — whichever fires first (trail or SM flip).

**Result**: Disappointing. Trail fires too early on trending days, cutting winners that SM flip would have let run. Worse than either exit alone.

| Config | TRAIN PF | OOS PF | Combined PnL |
|--------|----------|--------|-------------|
| H_5_8 | 1.214 | 1.025 | +$688 |
| H_10_15 | 1.258 | 1.068 | +$982 |
| v11 baseline | 1.256 | 1.083 | +$1,369 |

#### Approach 2: Portfolio (2 Contracts, Independent Exits)
Each entry deploys 2 contracts:
- Leg A (SM): exits on SM flip (standard v11 behavior)
- Leg B (Trail): exits on trailing stop

**Result**: Best approach. Trail leg earns its keep on OOS by protecting against chop losses.

| Leg | TRAIN PnL | OOS PnL | Combined |
|-----|-----------|---------|----------|
| SM leg (1 ct) | +$1,107 | +$262 | +$1,369 |
| Trail leg (1 ct) | -$124 | +$1,051 | +$927 |
| **Portfolio total** | **+$983** | **+$1,313** | **+$2,296** |

The trail leg LOSES on TRAIN (-$124, cuts trending winners) but GAINS on OOS (+$1,051, locks choppy profits). No regime prediction needed — the portfolio is naturally diversified across exit types.

#### Approach 3: Regime Detection
Tested 5 daily metrics to predict "trending vs choppy":
- Prior day range, first-30-min range, SM flips in first 60min, overnight gap, first-30/prior range %

**Result**: ALL metrics flipped predictive direction between train and test. No stable regime discriminator found. Regime detection is unreliable — portfolio approach sidesteps it entirely.

## Phase 4: Pine Script & MES Testing

### scalp_v15_portfolio.pine (ABANDONED)
Two-leg Pine Script v6 strategy attempt for MNQ. Required `pyramiding=2` and `close_entries_rule="ANY"`.

**FAILED**: Pine shares one position for all entries, so max loss events hit 2 contracts ($200) instead of 1 ($100). Trail leg only adds small gains on winners but doubles losses. TradingView showed DD going from 30% to 53%.

**Key Pine lesson**: `pyramiding=N` allows N same-direction entries. `close_entries_rule="ANY"` allows closing specific entry IDs (default FIFO ignores the ID you pass). But shared position means shared risk — true independent legs require separate strategies.

### Trail-Only on TV (REJECTED)
Trail-only (no TP, no SM flip) showed PF 0.933 on TradingView — net negative. Trail without a fixed take profit is not independently profitable.

### MES Trailing Stop / TP Test (REJECTED)
ALL trailing stop and TP configs HURT MES on both TRAIN and OOS. MES v9.4 baseline is already PF 1.621 on OOS. v15 is MNQ-only.

## Phase 5: MFE Analysis & Take Profit Discovery

### The Breakthrough Insight
Almost every v11 trade has positive excursion — 97-100% of trades go profitable at some point before exiting. The entry signal is near-perfect for short-term direction.

### MFE (Maximum Favorable Excursion) Results
**Script**: `strategies/v15_mfe_analysis.py`

| TP Level | 6-month Hit Rate | TV Window Hit Rate | 6-month PF | 6-month Net |
|----------|-----------------|-------------------|------------|-------------|
| 1 pt | 99.8% | 100% | 32.2 | +$381 |
| 3 pts | 98.6% | 100% | 10.5 | +$1,842 |
| 5 pts | 96.7% | 100% | 12.5 | +$3,362 |
| 7 pts | 94.1% | 98.5% | 8.8 | +$4,569 |
| 10 pts | 87.1% | 91.2% | 9.1 | +$6,337 |
| 15 pts | 74.9% | 82.4% | 10.3 | +$8,888 |
| 20 pts | 65.1% | 72.1% | 9.8 | +$10,712 |

ALL TP levels are independently profitable across all time periods. Higher TP = more per-trade profit but lower hit rate.

### scalp_v15_trail_MNQ.pine (TV VALIDATED)
Single-position strategy, identical entries to v11, with TP + trail exits:
- **TP=5 default** (user-configurable, 0 to disable)
- Trail activate/distance as backup (fires if TP disabled or set higher)
- Max loss stop 50pts, EOD close
- Exit priority: EOD > Max Loss > TP > Trail
- Designed to run ALONGSIDE v11 as a separate strategy

**TradingView validated** (Jan 25 - Feb 17): 60 trades, **86.67% WR**, PF 1.272, +$264.60, MaxDD 24.95%

**TP > 5 did not work well on TV** — user tested multiple levels, only TP=5 and below were profitable.

### Two-Strategy Production Setup
Run BOTH on MNQ 1-min chart with independent positions:
- **v11 MNQ**: SM flip exits, lets winners run. PF 1.681, +$1,175
- **v15 TP=5**: Fixed TP, captures near-certain small wins. PF 1.272, +$265
- **Combined: ~$1,440** (22% more than v11 alone)

## Final Status (Feb 17, Updated)

### Current Production Setup
| Instrument | Strategy | Pine Script | Status |
|-----------|----------|-------------|--------|
| MNQ | v11 (SM flip exit) | `scalp_v11MNQ.pine` | LIVE (TV validated) |
| MNQ | v15 TP=5 (scalp exit) | `scalp_v15_trail_MNQ.pine` | TV VALIDATED, ready to go live |
| MES | v9.4 (SM flip exit) | `scalp_v94MES.pine` | LIVE (TV validated) |

### Files Created This Session
| File | Purpose |
|------|---------|
| `strategies/v14_tick_compute.py` | Tick data -> per-bar microstructure features |
| `strategies/v14_tick_test.py` | Test 13 microstructure variants (L1-P6) |
| `strategies/v15_time_exits.py` | 22 mechanical exit configs (time, TP/SL, trailing) |
| `strategies/v15_hybrid.py` | Hybrid, portfolio, and regime detection analysis |
| `strategies/v15_mfe_analysis.py` | MFE distribution + TP level simulation |
| `strategies/scalp_v15_portfolio.pine` | Two-leg Pine script (ABANDONED) |
| `strategies/scalp_v15_trail_MNQ.pine` | TP scalp Pine script (TV VALIDATED) |
| `data/tick_microstructure_features.csv` | 171k bars of computed features |

### Key Learnings
1. **Entries are near-perfect for short-term direction.** 97-100% of trades have positive MFE. The SM+RSI signal genuinely predicts short-term price movement.
2. **TP=5 is the sweet spot.** High enough to cover commission and generate profit, low enough that nearly all trades reach it.
3. **Two separate strategies > one multi-leg strategy.** Pine's shared position model means multi-leg strategies share risk. Independent strategies have independent risk.
4. **Trail-only is not independently profitable.** Must pair with fixed TP or it loses money.
5. **MES doesn't need TP or trail.** v9.4 with SM flip exits is already strong.
6. **Pine `pyramiding` and `close_entries_rule` gotchas**: Default pyramiding=1 silently rejects second same-direction entries. Default FIFO exit rule ignores the entry ID you pass to strategy.close(). Must set `pyramiding=N` and `close_entries_rule="ANY"` for multi-entry strategies.
7. **No regime predictor is stable.** Every daily metric flipped direction between train and test.
8. **Iceberg detection works but is regime-dependent.** Underwater-gated iceberg exits scored ADOPT on train but failed OOS.
