# ADR Exhaustion Entry Filter — Complete Research (Mar 10, 2026)

## Status: RESEARCH COMPLETE. Directional ADR has signal but evidence is weak. Paper trade before implementing.

## Concept

"If the product has already moved its typical daily range, don't re-enter positions chasing the same direction."

The idea: compute Average Daily Range (ADR) as a rolling N-day mean of prior completed RTH daily ranges. Track today's intraday range and directional move from open in real time. Block entries when the range budget is consumed or the move is already extended in the entry direction.

## Study Design — 4 Layers

All layers use RTH session tracking (10:00-16:00 ET). ADR uses prior days only (no look-ahead). Today's range/move updates bar-by-bar. All tested as pre-filters (entry gates) with IS/OOS split-half validation.

**Scripts**: `adr_common.py` (shared computation), `adr_exhaustion_sweep.py` (V15+vScalpB+MES), `adr_exhaustion_vscalpc_sweep.py` (vScalpC partial exit engine)

**Full sweep outputs saved**: `backtesting_engine/results/adr_exhaustion/main_sweep_output.txt`, `backtesting_engine/results/adr_exhaustion/vscalpc_sweep_output.txt`

### Layer 1: Basic Range Gate
Block ALL entries when `today_range / ADR >= threshold`.
- Sweep: lookback [5, 10, 14, 20] x threshold [0.7, 0.8, 0.9, 1.0, 1.1, 1.2] = 24 configs

### Layer 2: Directional Gate (KEY INSIGHT)
Block LONGS when `(close - today_open) / ADR >= +threshold` (already rallied).
Block SHORTS when `(close - today_open) / ADR <= -threshold` (already sold off).
SM sign determines entry direction (sm > 0 = potential long, sm < 0 = potential short).
- Sweep: lookback [5, 10, 14, 20] x threshold [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] = 24 configs

### Layer 3: Combined (best L1 + L2)
Require BOTH range consumed AND directional move. ~6-9 combos from top L1+L2 results.

### Layer 4: Remaining Range vs TP
Block when `ADR - today_range < TP target` (strategy-specific: MNQ TP=5, MES TP=20, vScalpC TP2=25).
- 4 lookbacks x 1-2 TP thresholds

## Results Summary

### V15 (vScalpA) — WORKS (Layer 2 best)
Baseline: 472 trades, WR 84.5%, PF 1.289, +$1,948

| Config | Layer | IS dPF | OOS dPF | Verdict |
|--------|-------|--------|---------|---------|
| dir_lb14_t0.3 | L2 | +12.8% | +15.8% | **STRONG PASS** |
| dir_lb10_t0.3 | L2 | +10.3% | +15.1% | **STRONG PASS** |
| dir_lb20_t0.3 | L2 | +15.2% | +6.9% | **STRONG PASS** |
| dir_lb5_t0.3 | L2 | +5.6% | +12.3% | **STRONG PASS** |
| rng_lb10_t0.9 | L1 | +5.6% | +21.5% | **STRONG PASS** |
| Multiple L2 configs | L2 | +1-5% | +5-15% | MARGINAL PASS |

### vScalpB — WORKS (Layer 2 best, fewer passes)
Baseline: 376 trades, WR 71.5%, PF 1.186, +$889

| Config | Layer | IS dPF | OOS dPF | Verdict |
|--------|-------|--------|---------|---------|
| dir_lb20_t0.4 | L2 | +5.2% | +28.1% | **STRONG PASS** |
| dir_lb5_t0.7 | L2 | +5.2% | +14.8% | **STRONG PASS** |
| Various L2 | L2 | varied | varied | MARGINAL PASS |

### vScalpC — WORKS (best overall, 2 STRONG + 10 MARGINAL)
Baseline: 463 trades, WR 77.3%, PF 1.452, +$6,469, Sharpe 2.253

| Config | Layer | IS dPF | OOS dPF | OOS Sharpe | Verdict |
|--------|-------|--------|---------|------------|---------|
| dir_lb14_t0.3 | L2 | +6.6% | +15.7% | 3.490 | **STRONG PASS** |
| dir_lb20_t0.3 | L2 | +8.5% | +5.9% | 2.942 | **STRONG PASS** |
| rng_lb10_t0.9 | L1 | +2.5% | +26.1% | 3.941 | MARGINAL |
| dir_lb5_t0.6 | L2 | +1.6% | +23.7% | 3.840 | MARGINAL |
| dir_lb14_t0.6 | L2 | +1.2% | +22.6% | 3.801 | MARGINAL |
| dir_lb5_t0.5 | L2 | +3.4% | +19.6% | 3.685 | MARGINAL |
| comb_lb10_r0.9_d0.6 | L3 | +2.0% | +21.6% | 3.748 | MARGINAL |
| remain_TP1=7_lb10 | L4 | +1.0% | +14.1% | 3.370 | MARGINAL |
| 4 more | various | +0-3% | +5-15% | 2.9-3.4 | MARGINAL |

### MES_V2 — FAIL across ALL layers, ALL configs
Baseline: 540 trades, WR 56.1%, PF 1.332, +$4,331. No config passed on MES. The filter doesn't work for TP=20 swing trades with slow SM (EMA=255).

### Portfolio Aggregate (best multi-strategy)
- `dir_lb14_t0.7`: Best portfolio OOS Sharpe (IS $+3,844, OOS $+3,920, OOS Sharpe 1.878)
- `rng_lb5_t1.2`: Only config with MARGINAL PASS on all 3 strategies simultaneously

### Layer Ranking
1. **Layer 2 (Directional)** — best across all MNQ strategies. Multiple STRONG PASSes.
2. **Layer 1 (Basic Range)** — works for V15 and vScalpC but fewer passes.
3. **Layer 3 (Combined)** — MARGINAL only. AND logic blocks fewer entries, dilutes signal.
4. **Layer 4 (Remaining Range)** — weak. Only marginal on vScalpC TP1=7. Fails everywhere else.

## Code Audit Results (4 parallel agents, Mar 10)

### Agent 1: Look-ahead Bias & Timing Audit — ALL CLEAN
- **ADR Look-ahead**: CLEAN. Day D's ADR uses only days D-N through D-1. Off-by-one verified correct. The `date_list` iteration starts at `i_date=1`, window is `range(start, end)` where `end = i_date` (exclusive). Day `i_date` itself is never included in its own ADR. First date with valid ADR is `date_list[lookback_days]`.
- **Session Tracking Look-ahead**: CLEAN. Running max/min only uses bars 0..i. At bar i, `session_high[i]` uses only bars 0 through i. The loop processes strictly forward.
- **Gate-to-Entry Timing**: CLEAN. Gate consumed as `entry_gate[i-1]`. At bar i-1, session_high[i-1] includes bar i-1's high (which is known at bar close, matching live trading). Entry on bar i uses previous bar's gate value.
- **SM Direction Match**: CLEAN. `sm[i-1]` in gate matches `sm[i-1]` in entry logic. vScalpB's `sm_threshold=0.25` approximated as 0.0 in gate (conservative — over-blocks bars that wouldn't trigger entries anyway).
- **Gate Application Order**: CLEAN. `gate_ok = entry_gate is None or entry_gate[i - 1]` is checked BEFORE `sm_bull`, `rsi_long_trigger`, etc. in both `run_backtest_tp_exit` and `run_backtest_partial_exit`.
- **RTH Boundary**: CLEAN. Pre-RTH bars get NaN → fail-open. Entries independently blocked by session time check. First RTH bar (10:00 ET) has today_range ≈ 0, gate = True (range not exhausted at open).

### Agent 2: Baseline Verification — MATCH
- **vScalpC**: EXACT match to MEMORY (463 trades, PF 1.452, +$6,469, Sharpe 2.253)
- **V15**: 472 vs MEMORY 452 (+20 trades). Data extended to Mar 5 vs Feb 19. Not a bug — extra ~2 weeks of data.
- **vScalpB**: 376 vs MEMORY 352 (+24 trades). Same data extension.
- **MES_V2**: 540 vs MEMORY 519 (+21 trades). Same data extension.
- **Baseline code path verified**: `run_sweep()` line 315 passes `entry_gate=None` for baseline. vScalpC sweep passes no `entry_gate` arg (defaults to None).
- **Strategy params verified**: All match MEMORY.md production config exactly.

### Agent 3: Red Team — CRITICAL FINDINGS

**1. IS/OOS Asymmetry — BIGGEST RED FLAG**
Top vScalpC configs show IS improvement +2-6% but OOS improvement +15-25%. This 3-10x gap is suspicious.

IS/OOS detail for top 5 vScalpC passing configs:

| Config | IS dPF | OOS dPF | IS Sharpe Δ | OOS Sharpe Δ |
|--------|--------|---------|-------------|--------------|
| dir_lb14_t0.3 (STRONG) | +6.6% | +15.7% | +0.385 | +0.943 |
| dir_lb20_t0.3 (STRONG) | +8.5% | +5.9% | +0.465 | +0.395 |
| dir_lb5_t0.5 (MARGINAL) | +3.4% | +19.6% | +0.203 | +1.138 |
| dir_lb5_t0.6 (MARGINAL) | +1.6% | +23.7% | +0.100 | +1.293 |
| dir_lb14_t0.6 (MARGINAL) | +1.2% | +22.6% | +0.076 | +1.254 |

Most likely cause: **Seasonal regime bias**. IS = Feb-Aug 2025 (includes low-vol summer). OOS = Aug 2025-Mar 2026 (higher-vol fall/winter with more trending days). The filter naturally helps more in high-vol regimes where range gets consumed early. OOS blocking rates are 5-10% higher than IS for the same threshold, confirming the regime difference.

Only `dir_lb20_t0.3` shows roughly symmetric IS/OOS improvement (+8.5%/+5.9%), but it blocks 24% of trades.

**2. Multiple Comparisons — MODERATE CONCERN**
61 configs tested for vScalpC. Only 2 STRONG PASSes. Under null hypothesis (no real effect), expected false STRONG PASSes with 61 tests ≈ 1.2-2.4. The 2 observed STRONG PASSes fall within the false positive envelope. The 10 MARGINAL PASSes use a weak threshold (essentially "not worse").

**3. MES Universal Failure**
MES fails on every config across all 4 layers. If "range exhaustion" were a universal market property, MES should benefit too. The MES failure suggests the filter is specific to tight-TP scalps (TP=5-7 pts) where remaining range matters more.

**4. Percentile Calibration on Full Data**
The percentile thresholds (e.g., p20 = 263.8 for prior-day ATR) are computed on the full dataset including OOS data. This is a minor information leak — the threshold value itself incorporates OOS volatility levels. Same methodology as the already-approved prior-day ATR gate, so consistent but worth noting.

**5. Directional Gate is Conceptually Distinct from SM Magnitude**
Despite both using SM direction, the directional gate adds "how much of the daily range has been consumed in the entry direction" — a session-anchored, ADR-normalized perspective that SM magnitude alone lacks. A stock can have SM=+0.8 early (range not consumed) vs SM=+0.8 late (range exhausted). This explains why directional ADR shows signal while SM magnitude filters failed.

**Red Team Overall Verdict**: The mechanism is sound but statistical evidence is weaker than raw numbers suggest. The dramatic OOS improvements are likely inflated by a favorable volatility regime in the OOS period. Expect ~5-8% PF improvement in practice, not 15-25%.

### Agent 4: Edge Case & Gate Array Audit — ALL CORRECT
- **NaN propagation**: Pre-RTH NaN → fail-open everywhere. First 14 days NaN ADR → fail-open. First RTH bar today_range ≈ 0 → gate True. All correct.
- **Session reset**: Date boundary handled naturally. Friday-to-Monday gap: loop skips to next bar, resets at Monday 10:00 ET. Overnight bars skipped (NaN).
- **ADR RTH vs full range**: `adr_common.py` uses RTH-only (10:00-16:00 ET) for both ADR and session tracking — internally consistent. Different from `htf_common.py` which uses all bars. Deliberate design difference for different purposes.
- **Division by zero**: Guarded — `adr[i] <= 0` check prevents division. ADR = 0 would require 14 consecutive zero-range days (impossible for NQ).
- **Combined gate AND logic**: Verified correct. Blocks ONLY when BOTH conditions met. NaN → fail-open.
- **Gate slicing**: `is_len` consistent within each module. No cross-module contamination.

## Key Structural Lessons

1. **Directional ADR dominates basic range gate.** The asymmetric "don't chase the move" logic is more selective than symmetric "range is consumed."
2. **Combined filters (L3) don't improve over L2 alone.** AND logic blocks fewer entries, diluting the signal. The directional component already captures the important dimension.
3. **Remaining range vs TP (L4) is weak.** The theoretical appeal ("not enough range left for TP") doesn't hold because ADR is an average — actual daily ranges have high variance. A day that's already moved 1.0 ADR can still move 1.5 ADR.
4. **MES doesn't benefit.** Slow SM (EMA=255) and TP=20 mean MES trades larger moves that can continue beyond ADR. The filter is scalp-specific.
5. **IS/OOS asymmetry is a persistent pattern in this dataset.** The OOS half (Aug-Mar) has consistently higher baseline performance AND higher filter benefit. This is a data characteristic, not a filter characteristic.

## Practical Recommendation

If implementing:
- **Only directional gate (Layer 2)** — strongest signal, soundest economic logic
- **Only MNQ strategies** — fails on MES
- **Loose threshold (t=0.6-0.7)** — blocks <10% of trades, defensive layer
- **Expect ~5-8% PF improvement**, not the 15-25% seen in OOS
- **Paper trade 2+ months** spanning different volatility regimes before live
- **`dir_lb14_t0.3`** is the most symmetric IS/OOS config (STRONG PASS on V15 + vScalpC)
- **`dir_lb20_t0.3`** is the only config with roughly equal IS/OOS improvement for vScalpC

## Implementation Notes (if proceeding)

### Config fields needed:
- `adr_lookback_days: int` (default 0 = disabled, 14 for active)
- `adr_directional_threshold: float` (0.3-0.7 range, 0.0 = disabled)

### SafetyManager additions:
- Track RTH session open/high/low per instrument (reset at 10:00 ET)
- Compute ADR as rolling mean of prior N RTH daily ranges (not Wilder — simple mean)
- Gate: `move_from_open / ADR >= threshold` for longs when SM > 0
- Use `_gate_prev` (bar[i-1]) pattern matching existing gates
- Fail-open during warmup (first N days)
- Log blocked signals to `logs/blocked_signals.csv`

### Interaction with existing gates:
- Independent of leledc, prior-day level, VIX, prior-day ATR gates
- Applied after those gates (AND with all other gates)
- vScalpC already has prior-day ATR gate (blocks low-vol days). ADR directional gate is orthogonal (blocks specific entries on normal/high-vol days where the move already happened)

## Files

| File | Purpose |
|------|---------|
| `backtesting_engine/strategies/adr_common.py` | Shared ADR computation + gate builders |
| `backtesting_engine/strategies/adr_exhaustion_sweep.py` | V15 + vScalpB + MES sweep |
| `backtesting_engine/strategies/adr_exhaustion_vscalpc_sweep.py` | vScalpC partial exit sweep |
| `backtesting_engine/results/adr_exhaustion/main_sweep_output.txt` | Full V15+vScalpB+MES output |
| `backtesting_engine/results/adr_exhaustion/vscalpc_sweep_output.txt` | Full vScalpC output |
