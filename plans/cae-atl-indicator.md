# CAE-ATL: Chop and Explode with Auto-Trendlines — Implementation Plan

**Phase**: 3 (Team Reviewed)
**Date**: 2026-03-15
**Status**: Ready for Phase 4 execution

## Phase 3 Review Amendments

1. Added `max_age` input (500 bars) + age-based pruning
2. Changed `broken` arrays from `bool` to `int` (stores break bar_index, 0=unbroken)
3. Added `offset=-lb_right` to pivot marker plotshapes
4. Added session awareness: vertical lines at 09:30/18:00 ET, toggleable
5. Removed unnecessary `rsi_5m_prev` request.security call
6. Changed break threshold default from 1.0 to 0 (visual validation tool)
7. Added "Both" option to Trendline Source dropdown
8. Broken trendlines freeze at break point (stop extending)
9. Use `input.timeframe()` for sweepable HTF (not input.string)
10. Note: `request.security()` must be called unconditionally
11. Use `color.new()` for line transparency (no `transp=` in Pine v6)

---

## Vision Recap

Pine Script v6 **indicator** (not strategy) for TradingView:
- RSI(11) on 1-min + RSI on 5-min via `request.security()`, both sweepable
- Chop and Explode zones (green/yellow/red/black core)
- Auto-detect RSI pivot peaks and troughs
- Auto-draw trendlines: descending peaks (long setup), ascending troughs (short setup)
- Trendlines persist; new ones added as pivots form
- Breakout signals when RSI crosses through a trendline
- End goal: visual validation on TradingView, then backtest, then port to live dashboard

---

## Agent-by-Agent Analysis

### 1. CREATIVE

**Innovative approaches to auto-trendline detection on RSI:**

**Pivot detection method — "Confirmed Pivot with Lookback"**: Rather than the classic `ta.pivothigh(rsi, lb, lb)` which has a fixed symmetric lookback (and inherent `lb`-bar lag), I recommend an **asymmetric confirmed pivot**: `ta.pivothigh(rsi, lb_left, lb_right)` where `lb_left` controls how many prior bars must be lower (governs "significance") and `lb_right` controls confirmation delay (governs "how quickly we know"). Default: `lb_left=10, lb_right=3`. This gives significant pivots (10 bars of prior context) confirmed quickly (3 bars). The lag is only 3 bars vs 10 for symmetric.

**Trendline presentation ideas:**
- **Color-code by age**: Fresh trendlines bright, older ones fade (increase transparency by 10 per new line, floor at 80). Gives instant visual hierarchy — "which trendline is the active one?"
- **Dashed vs solid**: Active (unbroken) trendlines = solid. Broken trendlines = switch to dashed, then auto-delete after N bars. This visually distinguishes "live setups" from "historical breaks."
- **Breakout flash**: On the bar RSI breaks a trendline, draw a small circle/diamond on the RSI line at the break point + optional bgcolor flash (1 bar, very transparent). Catches the eye without clutter.
- **Trendline label**: Each trendline gets a tiny label at its right endpoint showing "L1", "L2" (long setup 1, 2) or "S1", "S2" (short setup), making it clear which direction the setup implies.

**Rejected ideas** (over-engineering for an indicator):
- Dynamic pivot lookback based on ATR — adds 15 lines, marginal visual improvement, sweep the input instead.
- Channel/parallel lines — doubles line count, exceeds 500 limit fast.

**Quantified complexity**: The color-fading + dashed-on-break adds ~20 lines of code. The label system adds ~15 lines. Both are within budget.

---

### 2. ARCHITECT

**Data Flow Diagram:**

```
[1-min close] --> ta.rsi(close, rsi_len_1m) --> rsi_1m
[1-min close] --> request.security("5", ta.rsi(close, rsi_len_5m)) --> rsi_5m

rsi_1m --> pivot detection --> peak_arrays / trough_arrays
                                    |
                          trendline construction
                                    |
                          line.new() + array management
                                    |
                          breakout detection (rsi vs trendline value)
                                    |
                          visual signals (shapes, bgcolor)

Zone rendering: hline() + fill() -- static, no state
```

**Component Boundaries (6 logical sections):**

| Section | Lines (est.) | State | Description |
|---------|-------------|-------|-------------|
| 1. Inputs | 40 | none | All sweepable parameters |
| 2. RSI Computation | 15 | none | 1-min + 5-min RSI via request.security |
| 3. Zone Rendering | 20 | none | hline/fill for chop-and-explode zones |
| 4. Pivot Detection | 30 | var arrays | Detect peaks/troughs, store in arrays |
| 5. Trendline Management | 80 | var arrays | Build, draw, extend, prune trendlines |
| 6. Breakout Detection | 35 | var bools | Detect RSI crossing trendlines, signal |
| 7. Info Table | 30 | var table | Config display |
| **Total** | **~250** | | |

**State management:**
- `var array<float> peak_bars` / `peak_vals` / `trough_bars` / `trough_vals` — store recent pivot locations/values
- `var array<line> long_lines` / `short_lines` — active trendline line objects
- `var array<float> long_slopes` / `long_intercepts` / `short_slopes` / `short_intercepts` — trendline math for breakout detection (faster than recalculating from endpoints each bar)
- `var array<int> long_start_bars` / `short_start_bars` — for pruning old trendlines
- Max pivots stored: 20 per side (configurable). Max trendlines: 30 per side = 60 total (well under 500 line limit).

**Key architectural decision**: Store trendline slope/intercept at creation time so breakout detection is O(1) per trendline per bar (one multiply + one compare), NOT O(n) recalculation. On 1-min charts with 60 trendlines, that's 60 multiplies per bar — negligible.

---

### 3. DOMAIN SPECIALIST (Algo Trading / Technical Analysis)

**What makes a valid RSI trendline?**

RSI trendlines are a well-established technique (Constance Brown, Andrew Cardwell). The methodology:

1. **For long setups (bearish-to-bullish transition)**: Connect 2+ *descending* peaks in RSI. When RSI breaks ABOVE this descending line, it signals momentum shifting bullish. These peaks typically form while RSI is in or near the bearish zone (30-50).

2. **For short setups (bullish-to-bearish transition)**: Connect 2+ *ascending* troughs in RSI. When RSI breaks BELOW this ascending line, it signals momentum shifting bearish. These troughs typically form while RSI is in or near the bullish zone (50-70).

**Critical rule**: The trendline must be **sloped against the current RSI direction**. A descending peak line is drawn while RSI is making lower highs (bearish behavior) — the break signals reversal. An ascending trough line is drawn while RSI is making higher lows (bullish behavior) — the break signals reversal.

**Pivot significance thresholds:**
- `lb_left=10, lb_right=3` default on 1-min = peaks that dominated 10 preceding bars, confirmed after 3.
- On 5-min RSI (if used for trendlines), `lb_left=5, lb_right=2` would be equivalent.
- **Minimum pivot separation**: Pivots used for a trendline should be at least `min_pivot_spacing` bars apart (default 10). Two pivots 2 bars apart don't form a meaningful trendline — they're noise.

**False break handling:**
- A "break" should require RSI to close **above/below** the trendline value, not just touch it. One bar is sufficient for visual purposes (this is a study, not a strategy).
- Optional: require break by `break_threshold` RSI points (default 0, sweepable up to 5). E.g., RSI must exceed the trendline by 2 points to count. This filters whisker touches.
- Once broken, a trendline is "spent" — mark it broken, switch to dashed, stop checking it for new breaks.

**RSI period choice:**
- RSI(11) on 1-min is a reasonable choice for trendline analysis. RSI(14) is the classic, RSI(8) is what the strategies use on 5-min. Making it sweepable (2-50) is correct.
- The 5-min RSI overlay is for context (matching what the trading strategies see), not necessarily for trendlines. Trendlines should default to the 1-min RSI since that's the chart timeframe and has more pivot resolution.
- Input toggle: "Draw trendlines on: 1m RSI / 5m RSI / Both" — default 1m only.

**Lookback for pivots:**
- 10 left / 3 right = confirmed within 3 bars. On 1-min, that's 3 minutes of lag. Acceptable for visual analysis.
- Store up to last 20 peaks and 20 troughs. Only the most recent 2-5 pivots on each side are typically relevant for trendlines, but keeping 20 allows seeing older structure.

---

### 4. SENIOR DEVELOPER (Pine v6 Implementation)

**Critical Pine v6 constraints and solutions:**

**A. line.new() 500-object limit:**
- Each trendline = 1 `line.new()`. Each label = 1 `label.new()`.
- Budget: 30 long trendlines + 30 short trendlines + 30 long labels + 30 short labels = 120 objects. Well under 500.
- But if we also mark break points (circles), that's more objects. Use `plotshape()` for break signals instead (no object limit).
- **Pruning strategy**: When array length exceeds `max_trendlines` (default 30 per side), delete the oldest trendline (`line.delete()` + `array.shift()`). This is O(1).

**B. Trendline extension and breakout math:**
- A trendline from pivot at (bar1, val1) to (bar2, val2) has slope = (val2 - val1) / (bar2 - bar1).
- Current trendline value at bar_index: `val1 + slope * (bar_index - bar1)`.
- This is 1 multiply + 1 add per trendline per bar. 60 trendlines = 60 mults — trivial.
- Use `line.set_x2()` and `line.set_y2()` each bar to extend the line to current bar. This is the standard Pine pattern for extending lines.

**C. request.security() for 5-min RSI:**
- Follow existing pattern: `request.security(syminfo.tickerid, "5", ta.rsi(close, rsi_len_5m), lookahead=barmerge.lookahead_off)`
- Also need previous bar: `request.security(syminfo.tickerid, "5", ta.rsi(close, rsi_len_5m)[1], lookahead=barmerge.lookahead_off)`
- The `[1]` inside the expression is evaluated on the 5-min timeframe, so it gives the prior 5-min bar's RSI. No look-ahead bias.
- Pine gotcha: `hline()` does NOT support `display=display.pane`. We're in a separate pane (non-overlay indicator), so `hline()` works fine for zone boundaries.

**D. Pivot detection using ta.pivothigh/ta.pivotlow:**
- `ta.pivothigh(rsi_1m, lb_left, lb_right)` returns the pivot value when confirmed, `na` otherwise.
- The pivot is located at `bar_index - lb_right` (it was confirmed lb_right bars ago).
- Store: `array.push(peak_bars, bar_index - lb_right)` and `array.push(peak_vals, pivot_val)`.

**E. Trendline construction logic:**
- When a new peak is detected, check the previous peak. If the new peak is LOWER (descending), draw a trendline from previous peak to new peak = long setup line.
- When a new trough is detected, check the previous trough. If the new trough is HIGHER (ascending), draw a trendline from previous trough to new trough = short setup line.
- Only connect consecutive qualifying pivots. Don't try to connect pivot[0] to pivot[3] skipping non-qualifying ones — that creates ambiguity and adds ~40 lines of combinatorial logic for minimal visual improvement.

**F. Array management pattern:**
```pine
var peak_bars = array.new<int>(0)
var peak_vals = array.new<float>(0)
var long_lines = array.new<line>(0)
// ...on new peak:
array.push(peak_bars, bar_index - lb_right)
array.push(peak_vals, pv)
if array.size(peak_bars) > max_pivots
    array.shift(peak_bars)
    array.shift(peak_vals)
```

**G. Performance on 1-min charts:**
- 1-min MNQ loads ~20,000 bars in TradingView (approx 2 weeks). At 60 trendlines checked per bar, that's 1.2M operations over the chart. Each operation is 1 multiply + 1 compare. Pine handles this without issue — I've seen indicators with 200+ operations per bar run on 50k bars.
- The expensive operations are `line.new()` and `line.set_x2()`/`line.set_y2()`. These are called at most once per trendline per bar. With 60 trendlines, that's 60 line updates per bar — well within Pine's performance budget.

---

### 5. PRODUCT STRATEGIST / USER ADVOCATE

**Is this the right visualization?**

Yes. Jason's existing strategies (vScalpA, vScalpB, vScalpC) all use RSI crosses as entry triggers. The CAE indicator shows the *context* around those crosses — is the RSI in chop (yellow zone, avoid) or transitioning out of an extreme (green/red zone, opportunity)? The auto-trendlines add the *momentum shift detection* that manual chart analysis provides but is hard to systematize.

**Controls/toggles to expose (priority order):**

| Input | Type | Default | Why |
|-------|------|---------|-----|
| RSI Length (1-min) | int | 11 | Primary analysis period, sweepable |
| RSI Length (5-min) | int | 8 | Matches existing strategy RSI, sweepable |
| Show 5-min RSI | bool | true | Toggle overlay on/off |
| Pivot Left Bars | int | 10 | Controls pivot significance |
| Pivot Right Bars | int | 3 | Controls confirmation speed |
| Min Pivot Spacing | int | 10 | Filters noise pivots |
| Max Trendlines per Side | int | 15 | Controls visual density |
| Break Threshold (RSI pts) | int | 0 | Filters false breaks (0 = any cross) |
| Show Long Setup Lines | bool | true | Toggle descending peak lines |
| Show Short Setup Lines | bool | true | Toggle ascending trough lines |
| Show Break Signals | bool | true | Toggle breakout markers |
| Show Broken Lines | bool | true | Keep broken trendlines (dashed) or delete |
| Broken Line Max Bars | int | 50 | Delete broken trendlines after N bars |
| Trendline Source | dropdown | "1-min RSI" | Which RSI to draw trendlines on |

**Visual clarity principles:**
- Long setup lines (descending peaks) = **blue** (cool = buy)
- Short setup lines (ascending troughs) = **orange** (warm = sell)
- 1-min RSI line = white (primary)
- 5-min RSI line = gray, thinner (context overlay)
- Break signals: green triangle up (bullish break of descending line) / red triangle down (bearish break of ascending line)
- Zone fills should be semi-transparent enough that trendlines are clearly visible on top

**What NOT to build in v1:**
- No alert conditions (add in v2 once visual validation confirms the signals are meaningful)
- No automatic long/short bias output (this is a study, not a signal generator)
- No ATR-adaptive anything (sweep the inputs instead)

---

### 6. QA / TEST STRATEGIST

**Verification approach:**

**A. Visual comparison (primary method):**
1. Load the indicator on a 1-min MNQ chart
2. Manually identify 5-10 obvious RSI peaks and troughs
3. Verify the indicator detects the same pivots (within lb_right bars of confirmation)
4. Manually draw trendlines connecting the same descending peaks
5. Verify the indicator's auto-trendlines match the manual ones in slope and position
6. Check 3-5 obvious trendline breaks and verify the indicator marks them

**B. Edge cases to test:**

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| Flat RSI in chop zone (45-55) | Many false pivots clustered together | `min_pivot_spacing` filter (default 10 bars minimum between pivots) |
| Sharp RSI spike (0->100 in 5 bars) | Single extreme pivot, no second pivot to connect | Trendline only drawn when 2+ qualifying pivots exist — no line from single pivot |
| RSI exactly on trendline | Ambiguous break | Use strict `>` for bull break, `<` for bear break (not `>=`/`<=`) |
| Gap open (Monday, holidays) | RSI jumps, pivot detection confused | `ta.pivothigh/low` handles gaps naturally — the gap creates a valid pivot if it's an extremum |
| Chart loads with 500 bars only | Not enough history for meaningful trendlines | Trendlines start forming as soon as 2 pivots exist — within ~50 bars typically |
| 5-min RSI becomes `na` (before enough bars) | Plotting error | Guard with `not na(rsi_5m)` before plotting |
| Max trendlines hit (30 per side) | Oldest deleted, but it might be visually important | 30 is generous — in practice 5-10 active trendlines per side. User can increase to 50 if needed. |

**C. Regression checklist:**
- Zone fills render correctly (green 60-70, yellow 40-60, red 30-40, black 45-55)
- RSI(11) 1-min matches manual RSI(11) calculation (spot-check 3 bars)
- RSI(8) 5-min matches what existing strategies show (cross-reference vScalpA chart)
- Trendlines don't disappear on chart scroll/zoom
- No Pine compilation errors
- No runtime errors on 20k+ bar chart

---

### 7. SECURITY AUDITOR

**Data integrity concerns:**

**A. Multi-timeframe look-ahead bias:**
- Using `lookahead=barmerge.lookahead_off` (matching existing strategy pattern) prevents look-ahead. The 5-min RSI value on a 1-min bar reflects the RSI as of the *last completed* 5-min bar. This means within a 5-min candle, the value stays constant (updates only when the 5-min bar closes). This is correct and safe.
- The `[1]` operator inside `request.security()` gives the *prior completed* 5-min bar's value, which is always historical. No look-ahead.

**B. Pivot detection look-ahead:**
- `ta.pivothigh(rsi, lb_left, lb_right)` with `lb_right >= 1` means the pivot is only confirmed AFTER `lb_right` bars have passed. The pivot value is placed at `bar_index - lb_right`. This is NOT look-ahead — it's delayed confirmation. The trendline appears on the chart `lb_right` bars after the actual peak occurred. This is correct behavior.
- **Risk if lb_right=0**: The pivot would be confirmed on the same bar it occurs, which could repaint (next bar might be higher). Default `lb_right=3` avoids this entirely. Add `minval=1` to the input to prevent users from setting it to 0.

**C. Trendline value projection:**
- When we calculate "RSI should be X if it's on the trendline", this is a mathematical projection, not a data leak. The trendline is constructed from past pivots only.

**D. No secrets/credentials**: This is a standalone Pine indicator — no API keys, no external data, no network calls beyond TradingView's built-in `request.security()`.

**Quantified risk**: The only real risk is `lb_right=0` causing repainting. Mitigation: `minval=1` on the input. Cost: 0 lines (just a parameter on the existing input definition). Probability of user hitting this: 0% with the guard.

---

### 8. DEVOPS / OPERATIONS

**File naming and location:**
- Save as: `/Users/jasongeorge/Desktop/NQ trading/strategies/cae_auto_trendlines.pine`
- Consistent with existing `.pine` files in `strategies/` directory
- No version suffix on v1 — add versioning if/when modifications warrant it

**Loading in TradingView:**
1. Open TradingView, navigate to MNQ 1-min chart
2. Pine Editor > New > Paste full code
3. Click "Add to chart" (adds as a separate pane below price chart — it's an `indicator()` with `overlay=false`)
4. Adjust inputs via the gear icon
5. Save the script in TradingView's Pine Editor (cloud save) for persistence across sessions

**Chart setup:**
- The indicator will appear in its own pane (like the existing CAE study)
- It can coexist with the price chart strategies (vScalpA, vScalpB, etc.) since those are overlay strategies
- It can coexist with RedK_TPX (another non-overlay indicator) — TradingView allows multiple panes

**No CI/CD impact**: This is a TradingView-only script. No engine changes, no deployment, no Supabase schema changes.

**Git**: Add the `.pine` file and the plan file. Standard commit.

---

### 9. CRITIC / RED TEAM

**Challenge 1: Are we over-engineering the trendline management?**

The Architect proposes storing slope/intercept arrays separately from line arrays, plus start_bar arrays, plus label arrays, plus broken-state arrays. That's 10+ arrays per side, 20+ total. For a visualization tool, this is heavy.

**Counter-proposal**: Store only what's needed. Each trendline needs: (1) the line object, (2) x1/y1/x2/y2 for breakout math, (3) a "broken" flag. Instead of parallel arrays, use a simpler approach:

```
var long_lines = array.new<line>(0)
var long_x1 = array.new<int>(0)
var long_y1 = array.new<float>(0)
var long_slope = array.new<float>(0)
var long_broken = array.new<bool>(0)
```

That's 5 arrays per side, 10 total. The labels can be created at trendline creation time and forgotten (no array needed — they don't move). This cuts the state management by ~40%.

**Quantified**: Reduces ~30 lines of array management code.

**Challenge 2: Do we really need the 5-min RSI overlay?**

The vision says "overlay both," and Jason's strategies use 5-min RSI for entries. However, plotting 5-min RSI on a 1-min chart creates a step-function (constant for 5 bars, then jumps). This could clutter the trendline analysis if trendlines are drawn on 1-min RSI.

**Verdict**: Keep the 5-min overlay but make it toggleable (default ON) and visually subdued (gray, thin, `plot.style_stepline`). The "Trendline Source" dropdown defaults to 1-min only. This serves the vision without cluttering.

**Challenge 3: Will 250 lines be enough?**

Looking at the component estimates: Inputs (40) + RSI (15) + Zones (20) + Pivots (30) + Trendlines (80) + Breakouts (35) + Info Table (30) = 250. With Pine's verbose syntax (every `if` needs indentation, no compact ternary across lines), actual code tends to run 20% over estimates.

**Revised estimate**: 280-320 lines. Still well within Pine's 500-line indicator comfort zone and not a concern.

**Challenge 4: Is the break threshold (default 0) the right call?**

Setting break threshold to 0 means ANY cross of the trendline is a signal. On 1-min charts, RSI can flicker around a trendline for 3-5 bars before committing. This will generate noisy signals.

**Recommendation**: Default break threshold to 1 (RSI point), not 0. One RSI point filters the worst noise without missing real breaks. The user can set to 0 for maximum sensitivity or 3+ for high conviction.

**Challenge 5: What's the simplest viable version?**

If we cut everything to the bare minimum that delivers visual validation value:
- RSI(11) 1-min (no 5-min overlay)
- Chop-and-explode zones
- Pivot detection (ta.pivothigh/low)
- Simple trendline: connect last two descending peaks, last two ascending troughs
- Only 2 active trendlines (1 long, 1 short) at a time
- Breakout marker when RSI crosses

This would be ~120 lines. But it loses the "trendlines persist" requirement and the visual richness that makes this useful for pattern study. The full 280-line version is not excessive for the stated goals.

**Final verdict**: Build the full version. The 280-line scope is standard for a Pine indicator with this feature set. No component is gratuitous.

---

## Synthesized Implementation Plan

### File Structure

```
strategies/cae_auto_trendlines.pine    <-- The indicator (single file)
plans/cae-atl-indicator.md             <-- This plan document
```

### Input Parameters

```
GROUP: RSI SETTINGS
  rsi_len_1m    int     11    (2-50)     "RSI Length (1-min)"
  rsi_len_5m    int     8     (2-50)     "RSI Length (5-min)"
  show_5m_rsi   bool    true             "Show 5-min RSI Overlay"

GROUP: PIVOT DETECTION
  lb_left       int     10    (3-50)     "Pivot Left Bars (significance)"
  lb_right      int     3     (1-10)     "Pivot Right Bars (confirmation)"
  min_spacing   int     10    (1-50)     "Min Bars Between Pivots"

GROUP: TRENDLINES
  max_lines     int     15    (1-50)     "Max Trendlines per Side"
  show_long_tl  bool    true             "Show Long Setup Lines (desc peaks)"
  show_short_tl bool    true             "Show Short Setup Lines (asc troughs)"
  show_broken   bool    true             "Show Broken Trendlines (dashed)"
  broken_max    int     50    (10-200)   "Remove Broken Lines After N Bars"
  tl_source     string  "1-min RSI"      "Trendline Source" (options: "1-min RSI", "5-min RSI")

GROUP: BREAKOUT DETECTION
  break_thresh  float   1.0   (0-10)     "Break Threshold (RSI points)"
  show_breaks   bool    true             "Show Breakout Signals"

GROUP: ZONE DISPLAY
  show_zones    bool    true             "Show Chop/Explode Zones"
  show_core     bool    true             "Show Black Core (45-55)"
```

### Algorithm: RSI Computation

```
// 1-min RSI (native)
rsi_1m = ta.rsi(close, rsi_len_1m)

// 5-min RSI (multi-timeframe)
rsi_5m = request.security(syminfo.tickerid, "5",
    ta.rsi(close, rsi_len_5m), lookahead=barmerge.lookahead_off)

// Select which RSI is used for trendline analysis
rsi_tl = tl_source == "5-min RSI" ? rsi_5m : rsi_1m
```

### Algorithm: Pivot Detection

```
// Detect pivots on the selected RSI source
ph = ta.pivothigh(rsi_tl, lb_left, lb_right)
pl = ta.pivotlow(rsi_tl, lb_left, lb_right)

// On new peak (ph is not na):
//   1. Check spacing: bar_index - lb_right - last_peak_bar >= min_spacing
//   2. If passes, push to peak_bars / peak_vals arrays
//   3. If array exceeds max_pivots (20), shift oldest

// On new trough (pl is not na):
//   Same logic for trough_bars / trough_vals arrays
```

### Algorithm: Trendline Construction

```
// When a new PEAK is confirmed:
//   Look at previous peak (peak_vals[size-2])
//   If new peak < previous peak (descending):
//     slope = (new_val - prev_val) / (new_bar - prev_bar)
//     Draw line from (prev_bar, prev_val) to (new_bar, new_val)
//     Color: blue (long setup — descending resistance line)
//     Push to long_lines, long_x1, long_y1, long_slope, long_broken arrays
//     Add label "L" + count at the line's right end

// When a new TROUGH is confirmed:
//   Look at previous trough (trough_vals[size-2])
//   If new trough > previous trough (ascending):
//     slope = (new_val - prev_val) / (new_bar - prev_bar)
//     Draw line from (prev_bar, prev_val) to (new_bar, new_val)
//     Color: orange (short setup — ascending support line)
//     Push to short_lines, short_x1, short_y1, short_slope, short_broken arrays
//     Add label "S" + count at the line's right end
```

### Algorithm: Trendline Extension (every bar)

```
// For each active (unbroken) trendline:
//   Compute projected_y = y1 + slope * (bar_index - x1)
//   If projected_y is within 0-100 (valid RSI range):
//     line.set_x2(the_line, bar_index)
//     line.set_y2(the_line, projected_y)
//   Else:
//     Trendline has left the RSI range — stop extending (mark broken)
```

### Algorithm: Breakout Detection

```
// For each UNBROKEN long setup line (descending peaks, blue):
//   projected_y = y1 + slope * (bar_index - x1)
//   If rsi_tl > projected_y + break_thresh:
//     BULLISH BREAK — mark trendline broken
//     Signal: green triangle up below the RSI line
//     If show_broken: switch line style to dashed
//     If not show_broken: line.delete()

// For each UNBROKEN short setup line (ascending troughs, orange):
//   projected_y = y1 + slope * (bar_index - x1)
//   If rsi_tl < projected_y - break_thresh:
//     BEARISH BREAK — mark trendline broken
//     Signal: red triangle down above the RSI line
//     If show_broken: switch line style to dashed
//     If not show_broken: line.delete()

// Cleanup: Remove broken trendlines older than broken_max bars
```

### Algorithm: Trendline Pruning

```
// On each bar, for each side (long/short):
//   1. If array size > max_lines:
//        Delete oldest line object, shift all parallel arrays
//   2. For broken trendlines:
//        If bar_index - break_bar > broken_max:
//          Delete line object, remove from all arrays
```

### Visual Design

```
PANE LAYOUT (non-overlay, separate from price chart):

RSI Lines:
  - 1-min RSI: white, linewidth=2, solid
  - 5-min RSI: gray (#888888), linewidth=1, style_stepline (shows 5-min holding pattern)

Zone Fills (matching original CAE):
  - 60-70: green,  transp=90  (bullish zone)
  - 40-60: yellow, transp=90  (chop zone)
  - 30-40: red,    transp=90  (bearish zone)
  - 45-55: black,  transp=60  (core chop — slightly more opaque than original's 50)

Horizontal Lines:
  - 70, 60, 55, 50, 45, 40, 30: hline() with appropriate colors

Trendlines:
  - Long setup (descending peaks): color=#2196F3 (blue), linewidth=2, solid
  - Long setup BROKEN:             color=#2196F3 (blue), linewidth=1, dashed, transp=50
  - Short setup (ascending troughs): color=#FF9800 (orange), linewidth=2, solid
  - Short setup BROKEN:              color=#FF9800 (orange), linewidth=1, dashed, transp=50

Breakout Signals:
  - Bullish break: plotshape, shape.triangleup, color=green, below RSI line
  - Bearish break: plotshape, shape.triangledown, color=red, above RSI line

RSI Line Coloring (from original CAE):
  - RSI > 60: blue segment (linewidth=3, linebr style)
  - RSI < 40: purple segment (linewidth=3, linebr style)
  - 40-60: default white

Pivot Markers:
  - Detected peaks: small red circle on RSI (shape.circle, above)
  - Detected troughs: small green circle on RSI (shape.circle, below)
```

### Pine v6 Specific Considerations

1. **`//@version=6`** and `indicator()` not `strategy()`
2. **`overlay=false`** — renders in separate pane
3. **`max_bars_back=5000`** — ensure enough history for pivots
4. **`hline()` for zone boundaries** — no `display` parameter needed (already in own pane)
5. **`fill()` between hlines** — same pattern as original CAE code
6. **No short-circuit `and`** — but we have no stateful functions in boolean expressions (pivots are computed unconditionally), so this is not a concern here
7. **No unicode** — all ASCII comments and labels
8. **No multi-line ternary** — keep all ternaries on single lines
9. **`line.new()` syntax**: `line.new(x1, y1, x2, y2, xloc=xloc.bar_index, color=..., style=..., width=...)`
10. **`line.set_*()` for extension**: `line.set_x2(ln, bar_index)`, `line.set_y2(ln, projected_y)`
11. **`line.set_style()` for broken lines**: `line.set_style(ln, line.style_dashed)`
12. **Array type syntax**: `array.new<float>(0)`, `array.new<line>(0)`, etc.

### Edge Cases and Mitigations

| Edge Case | Mitigation |
|-----------|------------|
| Chart has < lb_left + lb_right bars of history | Pivots simply won't detect — no error. Trendlines appear once enough bars load. |
| RSI stuck at 50 (dead flat) | No pivots detected (no local max/min) — no trendlines drawn. Correct behavior. |
| Two pivots at identical value (horizontal trendline) | Slope=0, which is valid. Trendline extends horizontally. Break detected normally. |
| Trendline projects to RSI > 100 or < 0 | Stop extending the line. Mark as expired (treated like broken for pruning). |
| 5-min RSI is `na` on early bars | Guard: `plot(show_5m_rsi and not na(rsi_5m) ? rsi_5m : na, ...)` |
| Many pivots cluster in a range (chop) | `min_spacing` filter prevents drawing 10 trendlines from noise pivots 3 bars apart. |
| User sets max_lines very high (e.g., 200) | 200 lines + 200 labels = 400 objects, still under 500 limit. But approaching it. Tooltip warns "Total objects = 2x this value; stay under 200." |

### Estimated Complexity

| Section | Lines |
|---------|-------|
| Header + indicator declaration | 8 |
| Inputs (4 groups, 16 parameters) | 45 |
| RSI computation (1-min + 5-min) | 12 |
| Zone rendering (hlines + fills) | 22 |
| RSI plotting (white + colored segments + 5m overlay) | 15 |
| Pivot detection + array management | 40 |
| Trendline construction (on new pivot) | 50 |
| Trendline extension (every bar loop) | 25 |
| Breakout detection (every bar loop) | 35 |
| Trendline pruning (max count + broken expiry) | 25 |
| Pivot markers (plotshape) | 8 |
| Breakout signal markers | 10 |
| Info table | 35 |
| **Total** | **~330** |

This is a moderate-size Pine indicator. The existing strategy files are ~300 lines each, so this is comparable.

---

## Critic's Accepted Amendments

1. **Break threshold default**: Changed from 0 to 1.0 RSI points (filters noise)
2. **Array count**: Reduced from 10+ per side to 5 per side (line, x1, y1, slope, broken)
3. **Labels**: Created once at trendline birth, not tracked in arrays
4. **lb_right minval**: Set to 1 (prevents repainting)

---

## Open Questions — RESOLVED (Mar 15)

1. **RSI default period**: **11** confirmed
2. **Trendline source**: **1-min RSI** confirmed
3. **Max trendline age**: **500 bars (~8 hours)** — added as input
4. **Pivot markers**: **Yes** — show dots, toggleable
5. **Break signal**: **One-bar marker** only
6. **5-min timeframe**: **Sweepable input** (default "5")

---

*Phase 2 complete. Proceeding to Phase 3 (Team Review).*
