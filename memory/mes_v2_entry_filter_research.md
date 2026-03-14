# MES v2 Entry Filter Research

Comprehensive log of all entry-side filters tested for MES v2.
Baseline: ~519 trades, PF 1.326, Sharpe 1.667 (FULL 12-month).
IS: PF 1.394, Sharpe 1.986, N=270. OOS: PF 1.292, Sharpe 1.502, N=249.
All tests use pre-filter validation (entry_gate), breakeven_after_bars=75.

**Pass criteria**: Both IS and OOS PF improve >= 5% (strong) or within 5% (marginal),
Sharpe non-negative change, trade count >= 70% of baseline each half, both halves profitable.

---

## Round 1 (Mar 4, 2026)

### 1. SM Magnitude Gate — FAIL
**Script**: `mes_v2_sm_magnitude_sweep.py`
**Idea**: Block entries when SM absolute value is too small (barely crossed zero).
**Sweep**: sm_min = [0.001, 0.005, 0.01, 0.02, 0.05]
**Result**: Works on IS but fails OOS. PF improvement doesn't transfer.
Pass rate: low. Overfits to IS regime.

### 2. SM Slope Gate — STRONG OOS SIGNAL
**Script**: `mes_v2_sm_slope_sweep.py`
**Idea**: Require SM to be moving away from zero at entry (positive slope for longs).
**Sweep**: slope_threshold = [0.0, 0.001, 0.002, 0.005, 0.01]
**Result**: Best filter in Round 1. slope=0.005: OOS PF 1.54, Sharpe 2.50.
BUT halves trade count (N=137 OOS, 55% of baseline). Fails the >= 70% count rule.
Monotonically improving PF with tighter threshold — genuine signal.

### 3. Rally Exhaustion Gate — MARGINAL
**Script**: `mes_v2_rally_exhaustion_sweep.py`
**Idea**: Block entries when price already moved too far in entry direction over lookback window.
**Sweep**: lookback = [5, 10, 15, 20] x max_move = [10, 15, 20, 25, 30, 35] = 24 configs.
**Result**: 2 of 24 passed (marginal). Tiny effect. Fixed lookback window doesn't anchor
to the right event. Motivated the "move-since-flip" variant in Round 2.

### 4. Volume Climax Gate — FAIL OOS
**Script**: `mes_v2_volume_climax_sweep.py`
**Idea**: Block entries on bars with extreme volume (climax move already happened).
**Result**: Passed IS, failed OOS. Volume climax pattern not stable across regimes.

### 5. Quick Stop Gate — FAIL (exit-side)
**Script**: `mes_v2_quick_stop_sweep.py`
**Idea**: Exit if trade goes underwater immediately (within N bars).
**Result**: Exit-side filter — kills winners more than losers. Fundamental problem:
winners and losers look identical in the first few bars.

### 6. Engulfing Bar Gate — FAIL (exit-side)
**Idea**: Exit on bearish engulfing bar (for longs) or bullish engulfing (for shorts).
**Result**: Same problem as quick stop — exit-side filters can't distinguish winners
from losers at exit time. Kills winners proportionally.

### 7. Consecutive Adverse Bars — FAIL (exit-side)
**Idea**: Exit after N consecutive bars moving against the position.
**Result**: Same exit-side failure pattern. All exit-side filters fail for the same reason.

### Round 1 Conclusions
- **All exit-side filters failed.** They kill winners at the same rate as losers.
- **SM slope is the only strong signal** but costs too many trades (halves count).
- **SM magnitude failed OOS** — IS-only signal.
- **Rally exhaustion is marginal** — fixed lookback doesn't anchor to signal event.
- Core trap pattern: SM barely crosses zero, RSI triggers, but price already rallied
  30+ pts to get there. Entry is chasing a move that's already done.

---

## Round 2 (Mar 5, 2026)

Tested 3 new filters attacking the trap from different angles, plus combinations.

### 1. Move-Since-SM-Flip Gate — MARGINAL PASS (best: max_move=5)
**Script**: `mes_v2_move_since_flip_sweep.py`
**Idea**: Track price at the SM zero-crossing (the "signal start"). Block entries if
price already moved too far since that flip. Anchors to the actual signal event rather
than an arbitrary lookback window.
**Sweep**: max_move_pts = [5, 10, 15, 20, 25, 30, 35, 40]

**Full results (max_move=5, best config)**:
- FULL: 416 trades, WR 56.7%, PF 1.366, +$3,409, Sharpe 1.829
- IS: 218 trades, WR 57.3%, PF 1.374, +$2,036, Sharpe 1.905
- OOS: 198 trades, WR 56.1%, PF 1.355, +$1,373, Sharpe 1.751
- Blocked: 103 entries (20%), retains 80% of trades (**passes count threshold**)

**Pass/fail per config**:
| max_move | IS PF chg | OOS PF chg | IS N% | OOS N% | Verdict |
|---|---|---|---|---|---|
| 5 | -1.4% | +4.9% | 81% | 80% | MARGINAL PASS |
| 10 | -5.5% | -11.3% | 92% | 92% | FAIL |
| 15 | -7.3% | -3.2% | 96% | 96% | FAIL |
| 20 | -6.7% | -7.9% | 97% | 98% | FAIL |
| 25 | -4.1% | -7.3% | 97% | 99% | FAIL |
| 30 | -4.1% | -1.9% | 97% | 100% | FAIL |
| 35 | -1.6% | +0.0% | 99% | 100% | MARGINAL PASS |
| 40 | -2.2% | +0.0% | 99% | 100% | MARGINAL PASS |

**Key observations**:
- NOT monotonic — tight filters don't consistently improve PF.
- max_move=5 is clearly the best OOS config (Sharpe 1.751 vs 1.502 baseline).
- max_move=10 is surprisingly the worst — suggests a non-linear relationship.
- max_move=35 and 40 barely block any trades (0-4 in OOS), effectively no-ops.
- Exit breakdown shows proportional reduction in all exit types (no selective improvement).
- Pass rate: 3/8 (38%), all marginal.

**Why this didn't work as well as hoped**: The SM zero-crossing doesn't always mark the
start of the move. SM can oscillate near zero, creating many small "flips" within a larger
trend, resetting the anchor price frequently. The 5-pt threshold is tight enough to still
catch some traps but the non-monotonic behavior suggests it's partially noise.

### 2. ATR-Normalized Extension Gate — FAIL
**Script**: `mes_v2_atr_extension_sweep.py`
**Idea**: Same as rally exhaustion but normalize by ATR. 30 pts when ATR=15 is extreme
(2x extension); 30 pts when ATR=40 is normal (0.75x). Adapts to volatility regime.
**Sweep**: lookback = [5, 10, 15, 20] x max_atr_mult = [1.0, 1.5, 2.0, 2.5, 3.0] = 20 configs.

**OOS PF grid**:
| LB \ Mult | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|
| 5 | 1.236 | 1.179 | 1.222 | 1.269 | 1.303 |
| 10 | 1.193 | 1.184 | 1.205 | 1.169 | 1.225 |
| 15 | **0.956** | 1.007 | 1.050 | 1.109 | 1.173 |
| 20 | 1.121 | 1.265 | 1.024 | 1.123 | 1.218 |

**OOS Sharpe grid**:
| LB \ Mult | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|
| 5 | 1.242 | 0.963 | 1.168 | 1.395 | 1.551 |
| 10 | 1.012 | 0.990 | 1.096 | 0.917 | 1.187 |
| 15 | **-0.255** | 0.038 | 0.284 | 0.604 | 0.934 |
| 20 | 0.657 | 1.379 | 0.139 | 0.674 | 1.138 |

**Pass rate: 2/20 (10%)** — only LB=5/Mult=2.5 and LB=5/Mult=3.0 pass marginally.
LB=5/Mult=3.0 blocks only 1 trade total — effectively a no-op.

**Key observations**:
- All OOS PFs below baseline (1.292) except LB=5/Mult=3.0 (1.303, blocks 1 trade).
- Longer lookbacks are worse — LB=15 produces a net-negative OOS at Mult=1.0.
- Tight multipliers block too many trades and degrade performance.
- Loose multipliers block almost nothing.
- No sweet spot exists — the filter doesn't capture the right information.

**Why ATR normalization failed**: ATR adapts to regime but the problem isn't regime-dependent.
The trap (chasing a move after SM flip) is about the relationship between SM signal timing
and price displacement, not about absolute or ATR-relative move size. The same 2x ATR
extension can be a fresh breakout (good) or a stale chase (bad) depending on context.

### 3. Prior Session S/R Gate — INTERESTING SIGNAL BUT FAIL
**Script**: `mes_v2_prior_session_sr_sweep.py`
**Idea**: Block longs near prior session high (resistance) and shorts near prior session
low (support). Classic "don't buy resistance / don't sell support" rule.
**Sweep**: buffer_pts = [0, 2, 5, 8, 10, 15, 20]

**Full results**:
| Buffer | FULL PF | IS PF | OOS PF | OOS Sharpe | IS N (%) | OOS N (%) |
|---|---|---|---|---|---|---|
| None | 1.326 | 1.394 | 1.292 | 1.502 | 270 (100%) | 249 (100%) |
| 0 | 1.329 | 1.380 | 1.308 | 1.587 | 199 (74%) | 174 (70%) |
| 2 | 1.355 | 1.397 | 1.348 | 1.777 | 196 (73%) | 166 (67%) |
| 5 | 1.364 | 1.369 | 1.358 | 1.826 | 188 (70%) | 156 (63%) |
| 8 | 1.339 | 1.312 | 1.379 | 1.904 | 180 (67%) | 152 (61%) |
| 10 | 1.304 | 1.285 | 1.334 | 1.718 | 177 (66%) | 146 (59%) |
| 15 | 1.311 | 1.257 | 1.396 | 2.012 | 166 (61%) | 131 (53%) |
| 20 | 1.320 | 1.213 | **1.516** | **2.468** | 156 (58%) | 121 (49%) |

**Diagnostics (distance to prior levels for blocked/allowed entries)**:
- At buffer=0: blocked longs avg 28 pts ABOVE prior high (already broken through).
  Allowed longs avg 48 pts below prior high.
- At buffer=20: blocked longs avg 15 pts above/near prior high.
  Allowed longs avg 69 pts below prior high.
- The filter correctly targets entries near/at resistance.

**Pass rate: 0/7 (0%)** — all fail on trade count (OOS N < 70% at all buffer levels).

**Key observations**:
- **Every buffer level improves OOS PF over baseline.** This is rare and noteworthy.
- OOS Sharpe monotonically improves (mostly) from 1.587 to 2.468.
- OOS PF improves from 1.308 to 1.516 (+17.3% at buffer=20).
- BUT IS PF degrades from 1.380 to 1.213 — IS/OOS divergence.
- AND trade count drops to 49-70% of baseline — too aggressive.
- Nearly monotonic OOS PF improvement (one dip at buffer=10).

**Why it fails pass/fail despite strong OOS signal**:
1. Blocks too many trades (30-50% loss of trade count).
2. IS/OOS divergence — IS PF degrades while OOS improves. This pattern often indicates
   the filter is sensitive to regime changes between halves rather than a stable edge.
3. NaN gap: 8,495 NaN bars after first valid bar = weekends/holidays between trading days
   where prior day levels carry over. Not a bug but worth noting.

**Salvageable?** Possibly. The OOS signal is real but:
- Could combine with other filters that preserve more trades.
- Could use it as a "proceed with caution" flag rather than a hard gate.
- The IS/OOS divergence needs explanation — may be related to changing market structure
  in the two halves (different rate environments, VIX regimes, etc.).

### 4. Combined Filters — FAIL (all combos fail trade count)
**Script**: `mes_v2_round2_combined.py`
**Constants**: max_move=5, ATR=skipped, S/R buffer=0, SM slope=0.005

**Results**:
| Combo | IS PF | OOS PF | OOS Sharpe | IS N (%) | OOS N (%) | Verdict |
|---|---|---|---|---|---|---|
| Baseline | 1.394 | 1.292 | 1.502 | 270 (100%) | 249 (100%) | -- |
| Flip only | 1.374 | 1.355 | 1.751 | 218 (81%) | 198 (80%) | **MARGINAL PASS** |
| S/R only | 1.380 | 1.308 | 1.587 | 199 (74%) | 174 (70%) | FAIL (OOS N=70%) |
| Slope only | 1.472 | 1.307 | 1.594 | 161 (60%) | 137 (55%) | FAIL (trade count) |
| Flip+S/R | 1.364 | 1.137 | 0.728 | 152 (56%) | 134 (54%) | FAIL |
| Flip+Slope | 1.156 | 1.383 | 1.883 | 127 (47%) | 116 (47%) | FAIL |
| S/R+Slope | 1.221 | 1.050 | 0.286 | 120 (44%) | 103 (41%) | FAIL |
| ALL(Flip+S/R+Slope) | 0.954 | 1.247 | 1.277 | 94 (35%) | 86 (35%) | FAIL (IS negative) |

**Key observations**:
- AND-ing filters compounds the trade count problem geometrically.
- Flip+Slope has best OOS Sharpe (1.883) but only 47% of trades — too few.
- Flip+S/R performs WORSE than either alone (OOS PF 1.137) — filters conflict.
- S/R+Slope is near-flat (OOS PF 1.050) — again, combining degrades both.
- ALL three together makes IS go negative — classic over-filtering.
- **Only Flip alone passes** — single filters preserve enough trades.

---

## Master Summary (All Filters Tested, Rounds 1+2)

| # | Filter | Type | OOS PF | OOS Sharpe | OOS N% | Verdict | Script |
|---|---|---|---|---|---|---|---|
| 1 | SM magnitude | Entry gate | varies | varies | varies | FAIL (OOS) | `mes_v2_sm_magnitude_sweep.py` |
| 2 | SM slope (0.005) | Entry gate | 1.307 | 1.594 | 55% | STRONG signal, low N | `mes_v2_sm_slope_sweep.py` |
| 3 | Rally exhaustion | Entry gate | ~baseline | ~baseline | >90% | MARGINAL (2/24) | `mes_v2_rally_exhaustion_sweep.py` |
| 4 | Volume climax | Entry gate | <baseline | <baseline | varies | FAIL (OOS) | `mes_v2_volume_climax_sweep.py` |
| 5 | Quick stop | Exit gate | <baseline | <baseline | 100% | FAIL (exit-side) | `mes_v2_quick_stop_sweep.py` |
| 6 | Engulfing bar | Exit gate | <baseline | <baseline | 100% | FAIL (exit-side) | (in quick stop script) |
| 7 | Consecutive adverse | Exit gate | <baseline | <baseline | 100% | FAIL (exit-side) | (in quick stop script) |
| 8 | Move-since-flip (5) | Entry gate | 1.355 | 1.751 | 80% | **MARGINAL PASS** | `mes_v2_move_since_flip_sweep.py` |
| 9 | ATR extension | Entry gate | <baseline | <baseline | varies | FAIL (2/20) | `mes_v2_atr_extension_sweep.py` |
| 10 | Prior session S/R | Entry gate | 1.308-1.516 | 1.587-2.468 | 49-70% | FAIL (trade count) | `mes_v2_prior_session_sr_sweep.py` |
| 11 | Flip+Slope combo | Entry gate | 1.383 | 1.883 | 47% | FAIL (trade count) | `mes_v2_round2_combined.py` |
| 12 | Other combos | Entry gate | varies | varies | 35-54% | FAIL (trade count) | `mes_v2_round2_combined.py` |

---

## Structural Lessons

1. **Exit-side filters fundamentally can't work for this strategy.** Winners and losers
   are indistinguishable in the first few bars. Any exit filter kills both proportionally.

2. **The trade count vs quality tradeoff is steep.** Filters that improve PF enough to
   matter (SM slope, prior S/R) also cut 30-50% of trades. The only filter that improves
   PF while keeping 80% of trades (move-since-flip) only improves by 5%.

3. **Combining filters compounds the problem.** Each filter independently blocks 20-50%
   of trades. AND-ing two filters blocks 50-65%. Three filters blocks 65%+. The remaining
   trades are a small, unrepresentative sample.

4. **The SM signal is inherently noisy.** SM oscillates near zero creating many false
   "flips." Filters that anchor to the flip point (move-since-flip, SM slope) capture
   some of the trap pattern but can't fully separate good from bad entries because the
   signal itself is noisy at the boundary.

5. **ATR/vol-normalization doesn't help.** The trap pattern isn't volatility-dependent.
   It's about signal timing (SM flip) vs price displacement, which varies with trend
   context rather than vol regime.

6. **Prior session S/R has a real edge but it's orthogonal to the SM signal.** The
   OOS improvement is consistent and near-monotonic. This filter works by removing a
   different class of bad entries (approaching resistance/support) rather than addressing
   the SM trap directly. Worth revisiting if trade count can be preserved.

---

## What's Left to Try

- **SM slope as a soft signal** (reduce position size rather than hard gate)
- **Prior S/R as a soft gate** (reduce size near levels rather than block)
- **Intraday S/R** (current day's developing high/low, not just prior session)
- **Time-of-day interaction** — are trap entries concentrated at specific times?
- **SM cross quality** — measure how "clean" the zero-crossing is (sustained vs oscillating)
- **News day filter** (already researched separately, see `news_day_analysis.md`)
- **TPX regime gate** (already researched separately, see `tpx_research.md`)
