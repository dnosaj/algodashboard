# vScalpC Runner Strategy — Design & Research

## Status: Phase 1+2 IMPLEMENTED (Mar 9). Phase 2 ATR gate live. Ready for paper trading.

## Motivation

The SM+RSI entry signal works at both scalp AND swing TP levels on MNQ. Current strategies (V15, vScalpB) optimize for high WR with tight TPs, leaving larger moves on the table. vScalpC complements them by capturing 20-30pt moves using the same proven entry with a partial exit structure.

## Winning Configuration (Phase 1 — Mar 9)

**Script**: `backtesting_engine/strategies/vscalpc_partial_exit_sweep.py` (400 combos tested)

### Production config (recommended):
```
Entry:  SM(10/12/200/100) SM_T=0.0  RSI(8/60/40)  CD=20  Entry cutoff 13:00 ET
Exit:   TP1=7  TP2=25  SL=40  BE_TIME=45  SL→BE after TP1=Yes
Qty:    2 contracts (1 closes at TP1, 1 rides to TP2)
```

### Performance:
| Metric | vScalpC | V15 current | V15 upgrade |
|--------|---------|-------------|-------------|
| Trades | 463 | 472 | 469 |
| WR | 77.3% | 84.5% | 83.2% |
| PF | 1.452 | 1.289 | 1.358 |
| P&L | +$6,469 | +$1,948 | +$2,612 |
| Sharpe | 2.25 | 1.48 | 1.89 |
| MaxDD | -$1,074 | -$570 | -$523 |

### IS/OOS: STRONG
All top 10 configs have OOS PF > IS PF. This is not overfit. The runner extension captures a real, persistent edge.

### Why this config:
- **TP1=7** dominates TP1=5 across all combos (+0.3 Sharpe, +$500-1000 P&L)
- **TP2=25** is the sweet spot. TP2=30 falls off (too ambitious). TP2=20 leaves money.
- **SL=40** same as V15 — proven level
- **SL→BE=Yes** after TP1: boosts WR 67%→77%, reduces MaxDD by $271. Runner becomes risk-free after scalp hits.
- **BE_TIME=45**: closes stale runners. Minimal impact vs disabled — runner either reaches TP2 quickly or doesn't.

### Alternate config (max Sharpe, no SL→BE):
```
TP1=7  TP2=25  SL=40  BE_TIME=0  SL→BE=No
```
- 445 trades, WR 66.5%, PF 1.384, +$7,192, Sharpe 2.35, MaxDD -$1,345
- More P&L (+$723) but worse WR and deeper drawdown. Keep as reference.

## Portfolio Impact

### Correlation
- vScalpC ↔ V15: **0.80** (high — same entries)
- vScalpC ↔ vScalpB: **0.24** (low — different SM threshold)
- vScalpC ↔ MES_V2: expected ~0.15-0.20 (different instrument)

### Combined portfolio (4 strategies):
| Portfolio | P&L | Sharpe | MaxDD |
|-----------|-----|--------|-------|
| V15 + vScalpB (MNQ only) | $2,837 | 2.21 | -$729 |
| V15 + vScalpB + vScalpC | $10,029 | 2.95 | -$1,791 |
| V15 + vScalpB + MES_V2 (current target) | $7,011 | 3.05 | -$1,057 |
| V15 + vScalpB + MES_V2 + vScalpC | ~$13,480 | ~3.2 | ~-$2,100 |

vScalpC adds +$6,469 P&L but deepens MaxDD by ~$1,062 (MNQ-only) due to 0.80 V15 correlation. The full 4-strategy portfolio needs validation.

## Phase 2: Higher Timeframe Research — COMPLETE (Mar 9)

Goal: use higher TF context to improve vScalpC entry quality (TP2 hit rate).

### Infrastructure built
- `htf_common.py`: Shared utilities — `resample_to_timeframe()`, `map_htf_to_1min()` (j-1 offset, no look-ahead), `compute_htf_indicators()` (batch SM/RSI/ATR/EMA on 5m/15m/30m/1H/4H), `compute_volatility_regime()` (prior-day classification, fixed look-ahead), `compute_prior_day_atr()` (14-day Wilder on daily ranges)
- `htf_diagnostic.py`: Diagnostic analysis — captures 94 features at each of 463 trade entries, correlates with both `leg2_pts` (continuous) and `tp2_hit` (binary)
- `htf_filter_atr_sweep.py`: Sweep script — 28 configs across 3 sweep types (intraday ATR, prior-day ATR, vol regime)

### Diagnostic findings
- **ATR is the dominant predictor** of TP2 success. Higher ATR (more volatility) = more likely to reach TP2=25.
- Binary target (`tp2_hit`) reveals much stronger correlations than continuous `leg2_pts` (censoring in continuous target masks the relationship).
- Top correlations with `tp2_hit`: 30min_atr r=0.183, 1h_atr r=0.171, 15min_atr r=0.157, prior_day_atr r=0.142, 5min_atr r=0.137
- All ATR features IS/OOS stable (r=0.12-0.18 on both halves)
- SM alignment, RSI, session context, vol regime — all weak (|r| < 0.10)
- **Vol regime look-ahead bug found and fixed**: original used today's range to classify today's bars. Fixed to use prior-day range. Correlation dropped from 0.15 to 0.10 after fix.

### Sweep results (28 configs tested)
7 STRONG PASS configs found. Summary of best per category:
- **PriorDay ATR p20 (>=263.8)**: STRONG PASS, IS PF +11.1%, OOS PF +11.5%. FULL: 383 trades (17% blocked), PF 1.615, $6,802, Sharpe 2.903, MaxDD -$1,074
- **15min ATR p20**: STRONG PASS, IS PF +11.3%, OOS PF +9.2%
- **30min ATR p15**: STRONG PASS, IS PF +9.2%, OOS PF +8.5%
- **Vol regime Block Low**: MARGINAL PASS only
- **Vol regime Block Low+Med**: FAIL (blocks too many trades)

### Selected config: PriorDay ATR p20 (>=263.8)
Sweep re-run after fixing UTC→ET date boundary bug in backtest (threshold shifted from 252.9 to 263.8).
- STRONG PASS: IS PF +10.1%, OOS PF +12.5%
- FULL: 384 trades (79 blocked, 17%), PF 1.615, $6,910, Sharpe 2.925, MaxDD -$1,074
- IS: 199 trades, PF 1.526, $3,195, Sharpe 2.540
- OOS: 185 trades, PF 1.720, $3,715, Sharpe 3.368

**Why this one** (aligns with trading philosophy):
1. Symmetric IS/OOS improvement (+10% / +13%) — p15 was asymmetric (+6% / +17%)
2. OOS Sharpe 3.37 (baseline 2.55) — substantial improvement
3. Prior-day ATR is known before the session opens (no intrabar computation)
4. Blocks low-vol days where TP2=25 is structurally unlikely
5. 17% block rate is reasonable — removes 79 trades, concentrates on high-opportunity days
6. Easy to implement (one number, computed daily)

### Implementation: `prior_day_atr_min=263.8` on MNQ_VSCALPC
- New field on StrategyConfig: `prior_day_atr_min` (default 0 = disabled)
- SafetyManager tracks daily ranges, computes Wilder ATR(14) on daily ranges
- Gate checks `_prior_day_atr_prev >= threshold` (prior-bar pattern, matches backtest)
- Fail-open during warmup (first 15 days — need 14 ranges for ATR seed)

## Phase 3: Structure-Based Runner Exit — IMPLEMENTED (Mar 13)

Replaces fixed TP2=25 with adaptive pivot-based swing exits. Full 6-phase workflow completed.

### Architecture
- **IncrementalSwingTracker** (`engine/structure_monitor.py`): Deque-based rolling buffer (maxlen=53). Pivot detection: candidate at `i - pivot_right`, confirmed when >= all left AND right neighbors. Parity-tested: zero mismatches across 375,211 bars vs vectorized backtest.
- **StructureExitMonitor** (`engine/structure_monitor.py`): Wraps tracker. `check_exit()` uses bar[i-1] close vs swing level - buffer. `get_bar_observation()` produces per-bar data for Supabase.
- **Runner integration** (`engine/runner.py`): Captures `prev_bar_for_struct` BEFORE `on_bar()`, then checks structure exit. Observation data emitted for active runners (partial_filled=True) on every bar.
- **OCO crash-safety cap**: tp_pts=60 on exchange. Structure monitor exits runner before cap. If engine dies, 60pt OCO catches tail risk.
- **BE_TIME disabled**: Skipped when `structure_exit_type` is set. Structure monitor + 60pt cap + EOD + SL@BE replace it.
- **Runtime toggle**: Dashboard STRUCT badge → WS `structure_exit_toggle` → `StructureExitMonitor.set_enabled()`. Fallback to 60pt cap when disabled. Tracker keeps updating even when disabled (Phase 5 fix).

### Config (MNQ_VSCALPC)
```
structure_exit_type="pivot"
structure_exit_lookback=50
structure_exit_pivot_right=2
structure_exit_buffer_pts=2.0
tp_pts=60                    # Was 25 — now crash-safety cap
trail_activate_pts=60        # Matches tp_pts
```

### Backtest — FULL PERIOD (12.5 months, 2 contracts)
```
                         P&L      PF   Sharpe    MaxDD      WR   Trades
Baseline (TP2=25)    $+5,693   1.374    1.914  $-1,586   76.7%     472
Structure exit       $+6,099   1.406    1.975  $-1,394   78.0%     469
─────────────────────────────────────────────────────────────────────────
  P&L:    +$406       (more money)
  MaxDD:  -12%        ($193 less drawdown)
  Sharpe: +3%
  PF:     +2.3%
  WR:     +1.3pp      (higher win rate)
```
IS/OOS PF ratio: 97.3%. 19/20 top configs pass IS/OOS.

### Dashboard (Stage 2)
- **PriceChart**: STRUCT toggle button (green). Draws swing high (#00cc6688 "SH") and swing low (#cc333388 "SL") price lines.
- **SafetyPanel**: Clickable STRUCT badge per strategy. Shows target level + distance for active runners. Click toggles structure exit on/off via WS.
- **TradeLog**: Green "STR" badge on trades with `exit_reason === 'STRUCTURE'`.

### Supabase (Stage 3)
- **Migration 004**: `structure_exit_level` column on `trades` table. Partial index on `exit_reason = 'STRUCTURE'`.
- **Migration 007**: `structure_bar_logs` table for per-bar observation data. Columns: bar_time, strategy_id, instrument, bar_close, swing_high, swing_low, position, entry_price, runner_profit_pts, distance_to_level_pts, near_miss, trade_date. Unique index on `(bar_time, strategy_id)`. 30-day retention.
- **db_logger.py**: `on_structure_bar()` handler, fire-and-forget queue. Upsert with dedup.

### Rejected alternatives
- **MES structure exit**: Full-period P&L -$1,460, Sharpe +5%, MaxDD -19%. Not justified.
- **Phantom OCO Ladder**: tastytrade has no partial OCO modification. Cancel-replace creates 100ms unprotected gap.
- **Donchian channel**: Not needed since MES rejected (pivot-only).

### Phase 5 Review (3-agent team, Mar 13)
- **APPROVED for paper trading** — all 3 agents (Senior Dev, Security+QA, Architect+Critic)
- **FIXED**: Tracker update moved above `_enabled` check (prevents 53-bar stale levels on re-enable)
- **FIXED**: Added daily reset comment (intentional non-reset matches backtest)
- **FIXED**: Removed unused `close_sig` variable
- **ACCEPTED**: BE_TIME disabled by static config (60pt cap + EOD + SL@BE provide 3 safety nets)
- **NOTED**: No unit tests for check_exit() — parity test covers swing detection, paper trading covers exit logic

### Paper trade criteria
- Notify Jason after 10 structure exits observed — review and discuss performance together
- Hard fail: position mismatch, OCO cancel failure, crash during structure processing
- Track via: `SELECT COUNT(*) FROM trades WHERE exit_reason = 'STRUCTURE'`

### Pending (completed)
- ~~Run Supabase migrations 004 + 007~~ — deployed Mar 14
- ~~caffeinate -di in NQ Trading.app launcher~~ — added Mar 14

## Phase 4: Portfolio Integration (LATER)

- Full 4-strategy portfolio backtest with optimized vScalpC
- Correlation matrix with structure exit applied
- Position sizing optimization
- Drawdown contribution analysis
- Combined equity curve + Monte Carlo

## V15 and vScalpB Exit Upgrades (separate from vScalpC)

### V15: TP=5 → TP=7 upgrade
- **TP=7 SL=40**: Sharpe 2.73 vs 2.08, PF 1.36 vs 1.29, MaxDD -$523 vs -$570
- Strictly better on every metric. IS/OOS STRONG (OOS PF > IS PF).
- **Action**: Update backtest constants + live config. Paper trade to confirm.

### vScalpB: TP=5/SL=15 → TP=3/SL=10 upgrade
- **TP=3 SL=10**: Sharpe 3.29 vs 1.49, PF 1.47 vs 1.19, MaxDD -$281 vs -$502
- Dominates on every metric. IS/OOS STRONG (OOS PF 1.49 > IS PF 1.45).
- SM_T=0.25 entries are quick pops — tighter exits capture more reliably.
- Commission drag: TP=3 = $6 gross, $1.04 RT = 17%. Acceptable.
- **Action**: Update backtest constants + live config. Paper trade to confirm.

## Implementation (Mar 9 — DONE)

### Config: `MNQ_VSCALPC` in `engine/config.py`
- `entry_qty=2`, `partial_tp_pts=7`, `tp_pts=60` (crash-safety cap), `max_loss_pts=40`
- `breakeven_after_bars=45`, `move_sl_to_be_after_tp1=True`
- `structure_exit_type="pivot"`, `structure_exit_lookback=50`, `structure_exit_pivot_right=2`, `structure_exit_buffer_pts=2.0`
- `max_strategy_daily_loss=200` (one 2-contract SL = $160)
- `vix_death_zone_min=19, vix_death_zone_max=22` (same VIX gate as V15)
- `leledc_maj_qual=9`, `session_end_et="13:00"`
- Added to `DEFAULT_CONFIG` → 4 strategies: V15, vScalpB, vScalpC, MES_V2

### New feature: `move_sl_to_be_after_tp1`
- New field on `StrategyConfig` (default False, True for vScalpC)
- **Live mode**: `replace_bracket_sl()` in `tastytrade_broker.py` — cancel runner OCO, replace with SL=entry_price + same TP. Race-safe: old bracket kept in dict during cancel-replace window. Post-placement check for late fills.
- **Paper mode**: `strategy.py` bar-close SL and `intra_bar_monitor.py` intra-bar SL both check `partial_filled + move_sl_to_be_after_tp1` → use `sl_pts=0` (breakeven)
- **Runner trigger**: `runner.py` after bracket TP1 fill processing, calls `replace_bracket_sl(sid, "tp2", entry_price)`. On failure, re-enables bar-close TP/SL.

### Dashboard
- PriceChart: purple/pink markers (`#bb77ff` / `#ff77bb`) for vScalpC entries
- SessionTradeList: `vC` label, purple badge color
- SafetyPanel: auto-discovers 4th strategy (data-driven)

### Code review findings (4 agents, Mar 9)
- **Fixed**: daily loss limit raised $100→$200, VIX gate added, reset_daily clears partial state, replace_bracket_sl race protection, runner re-enables bar-close TP on SL→BE failure
- **Pre-existing (not blocking)**: `_place_dual_oco` ignores dashboard partial_qty override (affects MES_V2 too)
- **Live-only concern**: cancel-replace gap leaves runner briefly unprotected (~1 network round-trip). Mitigated by fallback stop + race checks.
- Margin: 2 additional MNQ contracts ~$1,700
