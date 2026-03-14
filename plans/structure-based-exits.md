# Structure-Based Exit Research Plan

**Phase**: 2 (TEAM PLAN)
**Vision**: Use recent swing highs/lows as adaptive exit targets instead of fixed TP or lagging SM flip. Exit when price approaches a structural level it's likely to stall at.
**Date**: 2026-03-13

---

## 1. Creative Agent — Defining "Structure"

### Swing Detection Methods (from simplest to most complex)

**A. N-Bar Pivot (Fractal Pivot)**
The most intuitive definition. A swing high is a bar whose high is higher than the N bars on each side. For exits: find the most recent confirmed swing high/low relative to current price.
- Parameters: `pivot_left` (bars left), `pivot_right` (bars right for confirmation)
- Typical values: left=3..10, right=1..5
- Trade-off: larger N = fewer, more meaningful pivots but more lag in confirmation. right=1 confirms fastest but catches noise.
- **Key insight**: `pivot_right` introduces confirmation delay. A swing high confirmed with right=2 means we know about it 2 bars after the peak. This is fine for exits (we only need it once the level exists) but the delay matters for how close price can get before we react.

**B. N-Bar Highest High / Lowest Low (Donchian Channel)**
Rolling max(high, N) and min(low, N) computed over a lookback window. The "exit at the N-bar low" for longs is the classic turtle/trend-following exit.
- Parameters: `lookback` (bars)
- Typical values: 10..50
- Simpler than pivots — no confirmation delay. But the level moves every bar, so it's essentially a trailing stop with variable distance.
- **Distinction from trailing stop**: A fixed trailing stop trails by a constant distance from MFE. A Donchian exit trails by tracking the actual lowest low over N bars, which adapts to market structure (wider in volatile periods, tighter in consolidation).

**C. ATR-Offset from Swing**
Find the swing high/low per method A or B, then place the exit target N*ATR away from it. This accounts for the fact that price often overshoots structure by a noise margin before reversing.
- Parameters: `swing_lookback`, `atr_period`, `atr_mult`
- Adds complexity but may improve robustness.

**D. Rolling Support/Resistance Zones**
Cluster nearby swing highs/lows into zones (within X points of each other). Exit when price enters a zone. More sophisticated but also more parameters.
- Probably overkill for this research — save for later if pivots work.

**E. Prior-Bar-Close vs. Prior-Bar-High/Low for Level Comparison**
When checking if price "reached" a swing level, we can compare:
- `close[i-1]` >= swing_high (conservative, matches existing engine convention)
- `high[i-1]` >= swing_high (more aggressive, catches intra-bar touches)
Both should be tested.

**F. Hybrid: Structure + Minimum Profit Lock**
Only exit at a structure level if the trade is already profitable by at least X points. This prevents structure-based exits from cutting trades that haven't had a chance to run.
- Parameters: `min_profit_for_structure_exit` (pts)
- This is important — without a floor, a nearby swing level could exit a trade at +1 pt that was heading for +20.

**G. Hybrid: Structure Level as Dynamic TP (not trailing stop)**
Instead of exiting when price reaches a nearby swing level, use the swing level AS the TP target at entry time. Long entry -> find nearest swing high above entry -> set that as TP. This is different from the trailing approach — it's a one-time target.
- Problem: what if the nearest swing high is only 3 pts away? Need a minimum distance filter.
- Problem: swing levels found at entry time may become stale. But this avoids the "moving target" issue.

### Recommendation
Start with **Method A (N-Bar Pivot)** and **Method B (N-Bar High/Low)** as they're the simplest and most interpretable. Method F (profit lock) should be tested as an overlay on both. Method G (structure as TP target) is a fundamentally different approach and should be a separate test.

---

## 2. Architect Agent — Backtest Structure

### Current Architecture

The codebase has two backtest engines:
1. `run_backtest_v10()` in `v10_test_common.py` — SM flip exit, used for vWinners (v11.1)
2. `run_backtest_tp_exit()` in `generate_session.py` — TP/SL/EOD exits, used for V15/vScalpB/MES v2

Structure-based exits need to be tested on **both**:
- **vWinners revival**: Replace SM flip exit with structure exit in `run_backtest_v10`
- **Runner improvement**: Replace fixed TP2 with structure exit in `run_backtest_partial_exit` (from `vscalpc_partial_exit_sweep.py`)

### Proposed Implementation

**Option A: New standalone function (RECOMMENDED)**

Create `run_backtest_structure_exit()` as a new function, not a modification of existing ones. Reasons:
1. Structure exits fundamentally change the exit decision tree — bolting them onto `run_backtest_tp_exit` via flags would make that function even more complex
2. We need to test structure exits both as the ONLY exit (vWinners replacement) and as the runner exit in partial strategies — these are different enough to warrant separate functions or at least a clear mode switch
3. Preserves existing validated code untouched (matches "never overwrite strategy versions" rule)

**Function signature sketch:**
```python
def run_backtest_structure_exit(
    opens, highs, lows, closes, sm, times,
    rsi_5m_curr, rsi_5m_prev,
    rsi_buy, rsi_sell, sm_threshold,
    cooldown_bars, max_loss_pts,
    # Structure exit params
    swing_lookback,           # N-bar lookback for swing detection
    swing_type="pivot",       # "pivot" or "donchian"
    pivot_right=2,            # Confirmation bars for pivot type
    swing_buffer_pts=0,       # Exit N pts before swing level (0 = at level)
    min_profit_pts=0,         # Only structure-exit if profitable by >= X pts
    use_high_low=False,       # True = compare H/L to level; False = compare close
    # Fallback exits
    max_tp_pts=0,             # Hard cap TP (0 = no cap, structure-only)
    eod_minutes_et=NY_CLOSE_ET,
    breakeven_after_bars=0,
    entry_end_et=NY_LAST_ENTRY_ET,
    entry_gate=None,
):
```

**For partial exit testing**, create a second function:
```python
def run_backtest_partial_structure_exit(
    # ... same entry params ...
    sl_pts, tp1_pts,          # Scalp leg: fixed TP1 + SL (unchanged)
    # Runner leg: structure-based exit
    swing_lookback, swing_type, pivot_right,
    swing_buffer_pts, min_profit_pts,
    max_tp2_pts=0,            # Hard cap on runner (0 = structure-only)
    sl_to_be_after_tp1=False,
    be_time_bars=0,
    # ...
):
```

**Option B: Add structure exit as a mode in existing functions**

Add `exit_mode="structure"` alongside `"sm_flip"` and `"tp_scalp"`. This is cleaner architecturally but risks breaking existing validated paths.

**Recommendation**: Option A. The research code should live in a new file (`structure_exit_sweep.py`) with its own backtest function(s). If structure exits prove valuable, we can later refactor into the main engine.

### Swing Detection Implementation

The swing detection must be computed as a pre-pass over the full price series, NOT inside the bar-by-bar loop. This keeps the loop clean and makes lookback handling correct.

```python
def compute_swing_levels(highs, lows, lookback, swing_type="pivot", pivot_right=2):
    """Pre-compute swing high/low arrays.

    Returns:
        swing_highs: array where swing_highs[i] = most recent confirmed
                     swing high as of bar i (NaN if none yet)
        swing_lows:  array where swing_lows[i] = most recent confirmed
                     swing low as of bar i (NaN if none yet)
    """
```

For **pivot type**: A swing high at bar j is confirmed at bar j + pivot_right. So `swing_highs[i]` for i >= j + pivot_right reflects that level.

For **donchian type**: `swing_highs[i] = max(highs[i-lookback:i])` (exclusive of current bar to avoid look-ahead).

### File Organization

```
backtesting_engine/strategies/
    structure_exit_common.py      # compute_swing_levels(), helpers
    structure_exit_sweep.py       # vWinners-style sweep (SM flip entry, structure exit)
    structure_exit_runner_sweep.py  # Runner sweep (partial exit with structure TP2)
```

---

## 3. Domain Specialist Agent — Literature & Precedent

### Swing-Based Exits in Practice

**Donchian Channel Exit (Turtle Trading)**: The original trend-following exit. Long exits when price breaks below the 10-day low. Richard Dennis and the Turtles used 20-day entry channel, 10-day exit channel. This is well-documented and robust across decades of commodity futures data. The key insight: the exit channel is SHORTER than the entry channel — you enter on larger breakouts but exit on smaller pullbacks.

**Relevance to our system**: Our entries are SM+RSI momentum signals, not channel breakouts. But the exit logic is separable. A 10-20 bar low exit on 1-min bars would be analogous to a 10-20 minute structure exit.

**Chandelier Exit**: Developed by Charles LeBeau. Trail by N * ATR from the highest high since entry. This is essentially Method C from the Creative section. Well-studied, robust, but parameter-sensitive to ATR period and multiplier.

**Swing Point Trailing Stop**: Trail the stop to the most recent swing low (for longs). This is the exact mechanism Jason described. Common in discretionary trading but less studied in systematic literature because "swing point" definitions vary.

### Lookback Periods for 1-Min Bars

Academic literature on microstructure suggests that meaningful support/resistance on 1-min bars forms over 5-15 bar windows. The Pivot Point concept (left=3, right=3) on 1-min is effectively a 7-minute swing structure. For intraday futures:
- **5-10 bars**: Very short-term structure, frequent pivots, tight trailing
- **15-30 bars**: Moderate structure, captures 15-30 min swings
- **50-100 bars**: Captures session-level structure (1-2 hour swings)

Given that MES trades average 30-75 bars held and MNQ vWinners averaged ~80 bars, lookbacks of 10-50 bars are the relevant range.

### Fixed TP vs. Structure Exit: Robustness

Fixed TP is robust because it has ONE parameter and the optimal value is determined by the MFE distribution (a function of the entry signal, not market structure). Structure exits add parameters (lookback, buffer, type) that may overfit.

**However**: Fixed TP leaves money on the table when the market trends. The MES v2 MFE data shows 65.7% of trades reach +10 pts but only 40.3% reach +20 pts. A structure exit that captures +15 on a trade that reverses at +16 (instead of holding for +20 and giving it all back) is the potential edge.

### The Donchian Exit vs. Swing Pivot Exit

These are NOT the same thing:
- **Donchian exit**: Exits when price makes a new N-bar low. This fires DURING a pullback — it's reactive.
- **Swing pivot exit**: Exits when price APPROACHES a pre-identified level. This fires BEFORE the reversal — it's anticipatory.

The anticipatory nature of swing pivot exits is what makes them interesting but also what makes them harder to validate. You're betting that identified structure will hold, which is a prediction.

---

## 4. Senior Developer Agent — Implementation Details

### Changes to Backtest Engine

**No changes to existing files.** All new code goes in new files. This is critical because:
1. `run_backtest_tp_exit` is validated and produces known-good results for 4 active strategies
2. `run_backtest_v10` is the reference SM flip engine
3. Modifying either risks regression bugs

### Swing Level Computation (Pre-Pass)

```python
def compute_swing_highs_lows(highs, lows, left, right):
    """Pivot-based swing detection. Returns (swing_high_levels, swing_low_levels).

    swing_high_levels[i] = most recent confirmed swing high price as of bar i.
    swing_low_levels[i] = most recent confirmed swing low price as of bar i.

    A swing high at bar j requires:
      highs[j] >= highs[j-left:j] AND highs[j] >= highs[j+1:j+right+1]
    Confirmed at bar j + right.
    """
    n = len(highs)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    last_sh = np.nan
    last_sl = np.nan

    for j in range(left, n - right):
        # Check swing high
        is_sh = True
        for k in range(1, left + 1):
            if highs[j - k] > highs[j]:
                is_sh = False
                break
        if is_sh:
            for k in range(1, right + 1):
                if highs[j + k] > highs[j]:
                    is_sh = False
                    break
        if is_sh:
            # Confirmed at bar j + right
            confirm_bar = j + right
            last_sh = highs[j]

        # Check swing low
        is_sl = True
        for k in range(1, left + 1):
            if lows[j - k] < lows[j]:
                is_sl = False
                break
        if is_sl:
            for k in range(1, right + 1):
                if lows[j + k] < lows[j]:
                    is_sl = False
                    break
        if is_sl:
            confirm_bar = j + right
            last_sl = lows[j]

        # Fill from confirmation bar onward
        # (actually need to fill in a second pass or track confirm_bar)

    # Second pass: forward-fill confirmed levels
    # ...
    return sh, sl
```

**Important**: The forward-fill must happen ONLY from the confirmation bar onward. Before confirmation, the swing level is not known (no look-ahead).

### Exit Logic Integration

In the bar-by-bar loop, after SL check and before EOD:

```python
# Structure exit for longs
if swing_highs[i - 1] is not np.nan:  # We have a known swing high
    target = swing_highs[i - 1] - swing_buffer_pts
    current_profit = closes[i - 1] - entry_price
    if current_profit >= min_profit_pts:
        if closes[i - 1] >= target:  # or highs[i-1] >= target
            close_trade("long", entry_price, opens[i], ...)
```

For shorts, mirror with swing_lows.

### Key Decision: Which Swing Level to Use

For a long trade, the exit target should be the **nearest swing high ABOVE the current price** (or above entry price). We do NOT want to use a swing high below the entry — that would be a stale level.

This means the precomputed `swing_highs[i]` array needs to track ALL recent swing highs, and at exit time we need to find the one nearest-above the current price. This is more complex than a single "most recent" level.

**Simpler alternative**: Use the most recent swing high regardless of relative position. If price has already exceeded the most recent swing high, there's no structure target above — use a fallback (max TP or just hold).

**Even simpler**: Use the rolling N-bar high (Donchian) as the exit. For longs: exit when price drops below the N-bar low. This avoids the "which swing level" problem entirely.

### Performance Considerations

- Precomputing swing levels is O(N * left) — fast for N=300K bars, left<=50
- Bar-by-bar loop already iterates 300K bars — adding one comparison per bar is negligible
- Total sweep compute: depends on parameter grid (see QA section)

---

## 5. Product Strategist Agent — Strategy & Success Criteria

### Which Strategy First?

**Test on MES v2 runner first.** Reasons:

1. **Highest impact**: MES v2 runner currently uses fixed TP2=20. MES captures only 17% of MFE (per `mes_exit_research.md`). Structure-based exits could capture more of the available move.
2. **Existing partial exit infrastructure**: `run_backtest_partial_exit()` already handles 2-leg trades. We just replace the runner TP2 logic.
3. **Larger sample size**: MES v2 has 382 trades over 12 months. vWinners was on IS/OOS split data.
4. **Direct comparison**: We can compare structure runner vs. fixed TP2=20 runner using the exact same entries and TP1 scalp leg. Apples to apples.
5. **Jason's trigger**: The +48 pt manual exit was an MES trade.

**Second test: vWinners revival on MNQ.** If structure exits work on MES runner, test as full exit for MNQ vWinners (replacing SM flip). This has more upside (potentially revives a shelved strategy) but also more risk (the entry signal's OOS degradation may be the real problem, not the exit).

**Third test: vScalpC runner.** If MES works, test on vScalpC with TP2=25 replaced by structure exit. Same 2-leg framework.

### Success Criteria

**Primary metric: OOS Profit Factor > baseline** with consistent IS/OOS behavior.

For MES v2 runner (2-contract partial):
- Baseline: TP1=6, TP2=20, SL=35, BE_TIME=75 — PF ~1.34, Sharpe ~1.63
- **PASS**: OOS PF >= 1.30 AND OOS Sharpe >= 1.50 AND IS PF within 20% of OOS PF
- **STRONG PASS**: OOS PF > 1.40 AND OOS Sharpe > 1.70 AND MaxDD <= baseline MaxDD
- **FAIL**: OOS PF < 1.20 OR IS/OOS PF ratio > 1.5x (overfit signal)

For vWinners revival:
- Baseline: SM flip exit — IS +$3,529, OOS -$509
- **PASS**: OOS PF > 1.0 (profitable on OOS) AND OOS Sharpe > 0.5
- **STRONG PASS**: OOS PF > 1.15 AND consistent with IS

### What We're Optimizing For

Per trading philosophy: **risk-adjusted returns, not maximum extraction**. We want:
1. Higher PF (more profit per dollar of loss)
2. Better Sharpe (more consistent returns)
3. Same or better MaxDD
4. NOT necessarily higher total P&L — if structure exits capture $300 less but cut MaxDD by $500 and raise PF by 0.10, that's a win

### The "Let Winners Run" Problem Reframed

The original vWinners vision was "let winners run using SM flip." SM flip is a lagging indicator that gives back profits. Structure exits flip this: instead of waiting for an indicator to tell you the trend ended, exit when price reaches a level where it's likely to stall.

This is philosophically different:
- SM flip = "the trend is over, get out" (reactive, always late)
- Structure exit = "price is approaching resistance, take profits before it stalls" (anticipatory)

The risk is that structure exits could be TOO eager — exiting at every minor swing when price is actually trending through multiple swings. This is why the profit lock parameter and lookback window matter enormously.

---

## 6. QA / Test Strategist Agent — Validation Framework

### Parameter Grid

**MES v2 Runner (primary sweep):**

| Parameter | Values | Count |
|-----------|--------|-------|
| swing_lookback | 10, 15, 20, 30, 50 | 5 |
| swing_type | "pivot", "donchian" | 2 |
| pivot_right (pivot only) | 1, 2, 3 | 3 |
| swing_buffer_pts | 0, 1, 2 | 3 |
| min_profit_pts | 0, 5, 10 | 3 |
| use_high_low | True, False | 2 |

Pivot combos: 5 lookbacks * 3 pivot_rights * 3 buffers * 3 min_profits * 2 high_low = 270
Donchian combos: 5 lookbacks * 3 buffers * 3 min_profits * 2 high_low = 90
**Total: 360 combos** (manageable)

But wait — we also need to decide whether to keep a hard TP2 cap alongside the structure exit. Test with:
- max_tp2_pts = 0 (structure-only, no cap)
- max_tp2_pts = 30 (generous cap as safety net)
- max_tp2_pts = 50 (very generous cap)

That triples to ~1,080 combos. Still manageable at <1 second per run.

**Reduction strategy**: Run the full grid once on FULL data, then validate top 10 on IS/OOS split.

### Validation Methodology

1. **Full-period sweep** (360-1080 combos): Identify top 20 configs by PF, Sharpe, MaxDD
2. **IS/OOS split** (top 20): 60/40 chronological split, same as all prior research
3. **LOMO** (top 5): Leave-one-month-out on 12 months. Must win 7+/12 months.
4. **Robustness check**: Neighboring parameter values should produce similar results. If lookback=20 is great but lookback=15 and lookback=25 are terrible, it's overfit.
5. **Trade-by-trade comparison**: For the best config, compare every trade vs. fixed TP2=20 baseline. How many trades improve? How many get worse? Is the improvement concentrated in a few outliers or spread across many trades?

### IS/OOS Split

Use the same split as all prior research (first 60% = IS, last 40% = OOS). This is approximately:
- IS: Feb 2025 - Sep 2025 (~7.5 months)
- OOS: Oct 2025 - Mar 2026 (~5 months)

### Diagnostic Outputs

For each top config, output:
- Trade-level CSV with: entry/exit/pts/mfe/mae/bars/swing_level_used/exit_reason
- Distribution of exit reasons: structure vs SL vs EOD vs TP_cap vs BE_TIME
- MFE capture ratio: (actual_pts / mfe) — compare to baseline 17%
- Histogram of distance-to-swing-level at exit (did we actually exit near structure?)

---

## 7. Security Auditor Agent — Bias & Data Integrity

### Look-Ahead Bias Risks

**CRITICAL: Swing confirmation timing.** A pivot swing high at bar j with right=2 is CONFIRMED at bar j+2. The exit logic at bar i must only use swing levels confirmed at bar i-1 or earlier (matching the engine's prev-bar convention).

Implementation check: `swing_highs[i-1]` must NOT include any swing whose confirmation bar is > i-1. The pre-computation must respect this strictly.

**Donchian look-ahead**: `max(highs[i-lookback:i])` must use `i` exclusive (not inclusive). Including bar i's high in the lookback would create look-ahead. Use `highs[i-lookback:i]` (Python slice, exclusive of i).

**Bar-close vs. next-open fill**: All existing exits use the convention "signal from bar[i-1] close, fill at bar[i] open." Structure exits must follow the same convention. If `closes[i-1] >= swing_target`, fill at `opens[i]`.

### Data Snooping Concerns

1. **Parameter count**: With 360-1080 combos, we're testing many hypotheses. The top config will look good by chance alone. LOMO and neighbor stability checks mitigate this, but we should report the expected number of "false positive" configs (at p=0.05 with 360 tests, expect ~18 spurious passes).

2. **Lookback window and data length**: If swing_lookback=50 and the typical trade lasts 75 bars, the swing levels are updated several times during the trade. This is fine — the levels are computed from past data. But if someone later tries to optimize lookback by looking at the result, that's circular.

3. **Cross-contamination with existing research**: The MES v2 TP2=20 was itself determined by a sweep. Using the same data to find a structure exit that "beats" TP2=20 is a second optimization pass on the same data. The IS/OOS split helps, but be aware that both the baseline and the test used the same IS/OOS boundary.

### Mitigation

- Report the IS/OOS PF ratio for every config. Ratios > 1.3 are suspect.
- Check that the chosen config's neighbors (lookback +/- 5, buffer +/- 1) also pass. If it's an isolated peak, it's likely overfit.
- For the final chosen config, run a bootstrap test: randomly shuffle trade dates and check if the improvement vs. baseline is significant at p < 0.05.

---

## 8. DevOps / Operations Agent — Compute & Logistics

### Estimated Compute Time

Each `run_backtest_tp_exit` call on 300K bars takes ~0.3-0.5 seconds (pure Python loop). The partial exit variant is slightly slower (~0.6 seconds).

- Swing level pre-computation: O(N * left) = 300K * 50 = 15M ops, ~0.1 sec
- Precompute once, reuse for all lookback values? No — different lookbacks need different precomputation. But we can batch: 5 lookback values = 5 precomputations = 0.5 sec total.
- Per-config backtest: ~0.6 sec
- **Full sweep (1080 combos): ~11 minutes**
- IS/OOS validation (top 20): 20 * 0.6 * 2 = 24 sec
- LOMO (top 5): 5 * 12 * 0.6 = 36 sec

**Total estimated: ~15 minutes.** Very manageable.

### Environment

- Run in the existing `backtesting_engine/strategies/` directory
- Data: same Databento 1-min CSVs already in `backtesting_engine/data/`
- No new dependencies needed (numpy + pandas only)
- Save results using existing `save_results.py` framework

### Output Files

```
backtesting_engine/results/structure_exit/
    mes_v2_runner_full_sweep.csv       # All 1080 configs, full period
    mes_v2_runner_top20_isoos.csv      # Top 20, IS/OOS split
    mes_v2_runner_top5_lomo.csv        # Top 5, LOMO
    mes_v2_runner_best_trades.csv      # Trade-level detail for best config
```

---

## 9. Critic / Red Team Agent — Why This Might Fail

### Concern 1: "This is just a trailing stop with extra steps"

A Donchian exit (exit at N-bar low for longs) IS a trailing stop. The distance from price to the N-bar low varies with recent volatility, so it's an adaptive trailing stop — but it's still mechanically a trailing stop.

The v10 engine already tested trailing stops (`trailing_stop_pts`) and ATR trailing stops (`atr_trail_exit`). Both are available in `run_backtest_v10`. If these were never adopted, what makes Donchian trailing different?

**Counter-argument**: The existing trailing stop uses a FIXED distance from MFE. The Donchian/pivot exit uses ACTUAL market structure. These are genuinely different. A fixed trailing stop of 10 pts will whipsaw in volatile conditions and be too loose in quiet ones. A 20-bar low adapts.

**Test**: Run the existing ATR trailing stop in `run_backtest_v10` on vWinners parameters and compare. If ATR trail already captures most of the structure exit's benefit, there's no need for the more complex approach.

### Concern 2: IS/OOS Degradation (The vWinners Problem)

vWinners failed on OOS not because of the exit — it failed because the SM flip exit amplified the entry signal's marginal OOS performance. The SM entry signal at threshold=0.15 had IS PF 1.59 but the OOS edge was thin.

If structure exits are applied to the SAME entry signal, the OOS entry degradation is still present. A better exit can only help if the entry is sound. For MES v2, the entry signal is validated (OOS PF 1.05 with SM flip, 1.29+ with TP=20). For vWinners, the entry signal's OOS is questionable.

**Implication**: If structure exits fail on vWinners but succeed on MES v2, the problem was always the entry, not the exit. This would confirm that vWinners should stay shelved.

### Concern 3: Swing Level Lookback Overfitting

With 5 lookback values, 3 pivot_right values, 3 buffers, and 3 min_profit levels, we have 135 parameter combos for the pivot type alone. The "best" combo will look good by construction. The LOMO and neighbor checks are essential — without them, we're just curve-fitting.

**Historical precedent**: Every exit optimization in this project has either failed OOS or produced marginal improvement. RSI exits (REJECTED), hold time caps (REJECTED), losing hold cap (REJECTED), max hold (MARGINAL). The only exit that WORKED was the simplest possible: fixed TP at a level suggested by the MFE distribution, plus EOD time cutoff.

Structure exits add 3-5 parameters. The more parameters, the more likely the best combo is overfit.

### Concern 4: The 17% MFE Capture Problem

MES captures only 17% of MFE. Is this because the EXIT is bad (SM flip/fixed TP), or because the ENTRY timing is imprecise?

If a trade enters at 6000, rallies to 6020 (MFE=+20), then reverses to 5990 (final P&L=-10), the problem could be:
- **Exit problem**: We should have exited near 6020. Structure exit might help.
- **Entry problem**: We entered at 6000 but the "real" move started at 5990. We caught a temporary bounce, not a trend. No exit can fix a bad entry.

The MFE distribution data (65.7% reach +10, 40.3% reach +20) suggests many trades DO have the potential for +10-15 pts. Capturing even 50% of MFE instead of 17% would be a major improvement. This gives reason for cautious optimism.

### Concern 5: Intraday Structure is Noisy

On 1-min bars, swing highs and lows form constantly. A 10-bar swing high on 1-min data is just a 10-minute peak — these are plentiful and often meaningless. The signal-to-noise ratio of 1-min structure is much lower than daily or hourly structure.

**Mitigation**: Use longer lookbacks (30-50 bars) to find more meaningful structure. But longer lookbacks increase the lag.

### Concern 6: The +48pt Trade Was an Outlier

Jason's manual +48pt exit was exceptional. The MES MFE distribution shows only 15-20% of trades reach +48 pts. Designing an exit system optimized for capturing these rare large moves may not improve the TYPICAL trade. The median trade may be better served by the simple TP=20.

**Counter-argument**: We're not trying to capture +48 on every trade. We're trying to avoid the scenario where a trade goes +15, then reverses to -10. If structure exits capture +12 instead of -10 on those trades, that's +$110/contract improvement even though the trade never reached +48.

### Bottom Line

This research has a **moderate probability of producing actionable results for MES v2 runner**, and a **low probability of reviving vWinners**. The MES v2 case is more promising because:
- The entry signal is validated on OOS
- The partial exit framework limits downside (TP1 scalp is unaffected)
- We're only changing the runner leg, which is already the "speculative" leg
- The existing TP2=20 leaves known money on the table (65.7% reach +10 but only 40.3% reach +20)

The vWinners case is high-risk because the entry signal's OOS performance is the unresolved problem.

---

## Research Plan — Numbered Steps

### Phase A: Infrastructure (Steps 1-3)

1. **Create `structure_exit_common.py`**: Implement `compute_swing_levels()` with both pivot and Donchian methods. Include unit tests that verify no look-ahead bias (a confirmed swing at bar j+right should not appear in the output for bars before j+right).

2. **Create `structure_exit_sweep.py`**: MES v2 runner sweep script. Uses `run_backtest_partial_exit()` from `vscalpc_partial_exit_sweep.py` as a template but replaces the runner's fixed TP2 with structure-based exit. Entry logic and TP1 scalp leg remain IDENTICAL to production MES v2.

3. **Baseline verification**: Run the new script with `max_tp2_pts=20, swing_lookback=0` (effectively disabling structure exit, using hard cap only) and verify it produces IDENTICAL results to the existing MES v2 partial exit baseline. Any discrepancy means a bug.

### Phase B: Diagnostic (Steps 4-5)

4. **MFE/structure diagnostic**: For each MES v2 trade in the baseline, compute the nearest swing high (for longs) and swing low (for shorts) at key moments during the trade. Specifically:
   - At entry: what's the nearest swing level above (longs) / below (shorts)?
   - At MFE: what's the nearest swing level? Was the MFE near a swing level?
   - At exit: was there a swing level between entry and MFE that could have been used?
   This diagnostic tells us whether structure levels are actually relevant to these trades before we run any sweep.

5. **ATR trailing stop comparison**: Run the existing `run_backtest_v10()` with `atr_trail_exit=True` on MES v9.4 params (SM flip entry + ATR trail exit) as a quick sanity check. If ATR trailing already captures most of the structure exit's potential benefit, the added complexity of swing detection may not be justified. This takes 30 seconds and provides a useful comparison point.

### Phase C: MES v2 Runner Sweep (Steps 6-9)

6. **Full-period sweep**: Run all 360 parameter combos (or 1080 with max_tp2 variants) on full 12-month MES data. Record: trades, WR, PF, net P&L, Sharpe, MaxDD, avg bars held, % exits by reason (structure/SL/EOD/TP_cap/BE_TIME), MFE capture ratio.

7. **Top-20 IS/OOS validation**: Take top 20 configs by composite score (PF * Sharpe / (1 + MaxDD_ratio)). Run IS/OOS split. Apply PASS/FAIL criteria from Section 5.

8. **Neighbor stability check**: For each of the top 5 IS/OOS configs, verify that configs with lookback +/-5, buffer +/-1, min_profit +/-2 also pass. If the peak is isolated, flag as overfit.

9. **LOMO validation**: Run top 3 surviving configs through 12-month LOMO. Must win 7+/12 months.

### Phase D: Trade-Level Analysis (Step 10)

10. **Head-to-head comparison**: For the best structure exit config vs. baseline TP2=20:
    - Trade-by-trade P&L difference
    - Scatter plot: baseline P&L vs. structure P&L per trade
    - Count: how many trades improve, how many degrade, by how much
    - Identify the "rescued" trades (would have been losers with TP2=20, now profitable with structure exit) and "missed" trades (would have hit TP2=20 but structure exit cut them short)
    - Distribution of swing levels used for exit (are they clustered or random?)

### Phase E: vWinners Test (Steps 11-12, conditional)

11. **Only if MES runner passes**: Apply structure exit to MNQ vWinners (v11.1 params, SM flip entry, structure exit instead of SM flip exit). Run the same sweep grid but adapted for MNQ parameters.

12. **IS/OOS validation**: Same framework. The bar for vWinners is higher: OOS must be profitable (PF > 1.0), which it never was with SM flip exit.

### Phase F: vScalpC Runner Test (Step 13, conditional)

13. **Only if MES runner passes**: Test structure exit on vScalpC runner (replace TP2=25 with structure exit). Compare to current TP2=25 + SL-to-BE + BE_TIME=45 baseline.

### Phase G: Decision & Documentation (Step 14)

14. **Decision and documentation**:
    - If PASS: Document the winning config, write to `memory/structure_exit_research.md`, update MEMORY.md, identify next steps for live engine implementation
    - If FAIL: Document why, add to the "rejected exits" list in `mes_exit_research.md`, update vWinners status in MEMORY.md
    - Either way: save all sweep results to `backtesting_engine/results/structure_exit/`

---

## Estimated Timeline

| Phase | Steps | Time |
|-------|-------|------|
| A: Infrastructure | 1-3 | 45 min |
| B: Diagnostic | 4-5 | 30 min |
| C: MES Sweep | 6-9 | 45 min |
| D: Trade Analysis | 10 | 20 min |
| E: vWinners (conditional) | 11-12 | 30 min |
| F: vScalpC (conditional) | 13 | 20 min |
| G: Decision | 14 | 15 min |
| **Total** | | **~3 hours** |

This is research — correctness matters more than speed. Budget extra time for debugging the swing detection and validating the no-look-ahead requirement.
