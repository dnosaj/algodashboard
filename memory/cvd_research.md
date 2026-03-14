# CVD / Volume Delta Research (Feb 17, 2026)

## Motivation
On Feb 13, SM stayed bullish through a 235pt NQ selloff, causing 3 consecutive max-loss stops (-$308). SM is a *derived* indicator — it infers institutional flow from price/volume using RSI-of-money-flow + normalization. It cannot see actual aggressor side. Goal: fetch real order flow (tick-level with CME aggressor tag) and test whether it adds value.

## Data Pipeline

### Phase 1: NQ Tick Data
- **Source**: Databento `trades` schema for `NQ.c.0` (full-size, NOT MNQ — more institutional volume, same price movement)
- **Dataset**: GLBX.MDP3 (CME Globex)
- **Side field**: CME native aggressor tag: 'B' (buy), 'A' (sell), 'N' (undefined)
- **Pilot** (Feb 3-8): $2.50, 2M ticks, 99.99% side coverage (only 3 'N' ticks)
- **Full 6 months** (Aug 17 - Feb 13): $53.31, 42.6M ticks, 99.9997% side coverage
- **File**: `data/databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet` (330 MB)
- **Script**: `strategies/fetch_databento_ticks.py`
- **Stats**: 155 trading days, 274k ticks/day avg, size mean=1.5 contracts/tick

### Phase 2: Volume Delta Bars
- Aggregate ticks into 1-min bars: buy_vol, sell_vol, delta, total_vol
- CVD (Cumulative Volume Delta) with session reset at 6 PM ET (DST-aware)
- CVD_norm: rolling-max normalized to [-1,1] (like SM normalization)
- **File**: `data/databento_NQ_delta_1min_2025-08-17_to_2026-02-13.csv` (7.9 MB)
- **Script**: `strategies/compute_volume_delta.py`
- **Stats**: 171,628 1-min bars, avg delta/bar -0.3, std 80.8
- **Bug fixed**: Databento `size` field is uint64 — must cast to int64 before subtraction or buy-sell overflows

## Phase 3: CVD vs SM Analysis (TRAIN: Aug 17 - Nov 16)

### Key Findings
1. **Correlation**: SM vs CVD_norm Pearson r = **-0.029** (essentially uncorrelated — captures completely different signal)
2. **Lead-lag**: CVD flips before SM by ~1.4 bars on average (46% CVD leads, 35% SM leads, 19% simultaneous)
3. **Sign agreement**: Only 47.5% (worse than coin flip — they disagree more than agree)
4. **Trade-level**: CVD-agree trades PF **1.434** (+$1,181), CVD-disagree PF **0.954** (-$74). PF spread = +0.48
5. **Divergence**: SM Bull + CVD Bear has 15-bar forward return of -0.40pts vs +4.19pts baseline (strong warning signal)
6. **Monthly stability**: Correlation and agreement stable across all 4 train months

### What CVD Actually Measures
- "Buy aggressor" = someone used a market order to buy (lifted the ask). Does NOT mean institutional buying.
- Institutions often use LIMIT orders (passive side) — they show up as NON-aggressor
- CVD tracks *urgency/aggressiveness*, not necessarily "smart money"
- SM's derived PVI/NVI model is "wrong" in theory but the smoothing creates a useful oscillator that captures regime changes
- SM and CVD being uncorrelated (r=-0.03) is the BEST case — they capture genuinely different information

### Recommendation
Proceed to Phase 4: CVD shows clear independent signal (3/4 criteria passed).

## Phase 4: Backtest Integration

### Round 1 — Original Variants (FAILED)
Used 400-bar CVD_norm normalization (6.5 hours on 1-min bars — FAR too slow).
Entry filter implementation was flawed: modified SM array globally, breaking exit logic.

| Variant | Description | Verdict | dPF | Trades |
|---------|-------------|---------|-----|--------|
| A | CVD entry filter (sign) | REJECT 2/7 | -0.015 | 76 |
| A2-A4 | Entry filter (thresholds 0.1-0.3) | REJECT 2/7 | ~0 | 69-72 |
| B | CVD replaces SM | MAYBE 4/7 | +0.266 | 103 |
| C | SM+CVD confluence | REJECT 2/7 | -0.015 | 76 |
| D | CVD flip exit | MAYBE 4/7 | +0.091 | 179 |
| E | CVD divergence exit (0.3) | MAYBE 4/7 | +0.006 | 180 |
| E2 | CVD divergence exit (0.5) | MAYBE 3/7 | -0.115 | 180 |

**Lessons**: 400-bar norm too slow. Entry filter corrupted SM exits (179->76 trades was a bug, not a feature). No variant reached statistical significance.

### Round 2 — Revised Variants
Fixed: fast 20-30 bar normalization, proper entry filter (CVD checked ONLY at entry bar, exits untouched), raw delta burst detector, conviction sizing.

#### TRAIN Results (Aug 17 - Nov 16, 179 baseline trades, PF 1.256)

| Variant | Description | Verdict | dPF | Trades | Key |
|---------|-------------|---------|-----|--------|-----|
| F1 | Burst exit (5bar/300) | REJECT 1/7 | -0.221 | 183 | 80 BURST exits, cuts winners |
| F2 | Burst exit (10bar/300) | MAYBE 3/7 | -0.168 | 183 | Still cuts too many winners |
| F3 | Burst exit (10bar/500) | REJECT 2/7 | -0.162 | 181 | DD worse (-$922 vs -$648) |
| G1 | Entry filter (fast CVD 20) | **MAYBE 4/7** | +0.081 | 129 | **DD improved: -$505 vs -$648** |
| G2 | Entry filter (fast CVD 30) | MAYBE 4/7 | +0.081 | 129 | Identical to G1 |
| FG | Burst + filter combined | MAYBE 3/7 | -0.150 | 132 | Burst still hurts |
| H | Conviction sizing (2x agree) | — | — | 179 | **$2,275 vs $1,107 (+105%)** |

**H breakdown**: 123 agree (PF 1.426), 56 disagree (PF 0.961). PF spread +0.465. STRONG separation.

#### OOS Results (Nov 17 - Feb 13, 176 baseline trades, PF 1.083)

| Variant | Verdict | dPF | Trades | Key |
|---------|---------|-----|--------|-----|
| F1-F3 | REJECT 1-2/7 | -0.13 to -0.31 | 180-182 | All burst exits FAILED OOS |
| G1 | **MAYBE 4/7** | **+0.089** | 106 | **PF 1.172 vs 1.083, DD -$808 vs -$812** |
| G2 | MAYBE 4/7 | +0.089 | 106 | Identical to G1 |
| FG | REJECT 1/7 | -0.327 | 107 | Burst kills it OOS too |
| H | — | — | 176 | PF spread collapsed: +0.099 (vs +0.465 train) |

### Critical OOS Findings
1. **Delta burst exits: definitively REJECT.** Raw delta threshold exits cut winners short consistently. The problem: a 300-unit rolling delta is only ~1.5 sigma on 1-min bars — it triggers on routine moves, not just "flushes." Would need much higher thresholds but then fires too rarely to matter.
2. **G1 entry filter: HELD UP on OOS.** PF improved +0.089 on both train AND test. Win rate unchanged. Drawdown slightly better. But never reached statistical significance (p=0.40). Consistent improvement but weak.
3. **Conviction sizing (H): DEGRADED on OOS.** Train PF spread was +0.465 (strong). Test PF spread collapsed to +0.099 (weak). The CVD agree/disagree separation doesn't persist across regimes. This is the key finding — CVD's trade-level filtering power may be regime-dependent.

## Summary Assessment

### What Works (weakly)
- **G1 (fast CVD entry filter)**: Consistent +0.08 PF improvement on both train and test, with better drawdown. Removes ~28% of trades but keeps win rate. Not statistically significant but directionally correct on both periods.

### What Doesn't Work
- **Delta burst exits**: Raw delta is too noisy for exit signals at the 1-min level. Cuts winners as often as losers.
- **CVD as SM replacement (B)**: Wins bigger but wins much less often (-13.5% WR). Different character entirely.
- **Conviction sizing (H)**: Strong separation on train but collapsed on test. Regime-dependent.
- **Confluence/agreement filters**: SM and CVD only agree 47.5% of the time, so any "both must agree" filter kills trade count.

### Why CVD Underwhelms Despite Being "Real" Data
1. Buy aggressor != institutional buying. Institutions use limit orders (passive side).
2. SM's smoothing (RSI + EMA + normalization) accidentally creates a useful oscillator. CVD is "too real" — too noisy for 1-min algorithmic trading.
3. The asymmetric divergence results (SM Bull+CVD Bear is more damaging than SM Bear+CVD Bull) is a regime artifact of the training period's uptrend, not a structural edge. Would reverse in a bear market.
4. 400-bar normalization (first attempt) was too slow. 20-bar normalization (second attempt) was better but CVD_norm still has high inertia from cumulative nature.

### Open Questions / Future Directions
- Could delta at KEY PRICE LEVELS (prior day high/low, round numbers) be more informative than bar-by-bar delta?
- Is tick imbalance (large trades vs small trades) more useful than simple aggressor side?
- Would a different aggregation (e.g., 5-min delta bars instead of 1-min) smooth enough noise to be useful?
- The G1 filter + conviction sizing together (filter entries AND size up on conviction) was not tested as a combination

### Files Created
- `strategies/fetch_databento_ticks.py` — Download NQ tick data
- `strategies/compute_volume_delta.py` — Tick -> 1-min delta bars -> CVD
- `strategies/cvd_analysis.py` — SM vs CVD correlation/divergence analysis
- `strategies/v12_cvd_test.py` — Backtest integration (revised with all variants)

### Data Files
- `data/databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet` (330 MB, 42.6M ticks)
- `data/databento_NQ_ticks_2026-02-03_to_2026-02-08.parquet` (15.5 MB, pilot)
- `data/databento_NQ_delta_1min_2025-08-17_to_2026-02-13.csv` (7.9 MB, 171k bars)
