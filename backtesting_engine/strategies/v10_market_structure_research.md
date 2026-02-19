# v10 Market Structure Research -- MNQ SM-Flip Strategy Improvements

**Date:** Feb 13, 2026
**Baseline:** v9 SM-Flip, 57 trades, 78.9% WR, PF 2.725, +$2,469, MaxDD -$452
**Data:** MNQ 5-min, Jan 18 - Feb 13, 2026 (20 trading days)

---

## 1. Trade Pattern Analysis

### 1.1 The 12 Losing Trades

Out of 57 baseline trades, 12 lost money. Total losses: -$1,432. Total wins: +$3,900.
Seven of the 12 losers were "big" (worse than -$50), and those 7 accounted for -$1,399
of the -$1,432 total loss. The remaining 5 losers lost a combined -$33 -- essentially
breakeven noise. The strategy's weakness is concentrated in a handful of catastrophic
trades, not a pattern of small losses.

**The 7 big losers:**

| # | Dir | Date | Time | P&L | MAE | Hold | Exit | Key Context |
|---|-----|------|------|-----|-----|------|------|-------------|
| 1 | short | Jan 19 | 11:50 | -$66 | -$121 | 26 bars | SM Flip | First day of data, no prior day context |
| 15 | short | Jan 23 | 14:35 | -$452 | -$470 | 16 bars | SM Flip | Worst trade. SM stayed bearish during 225pt rally |
| 29 | short | Feb 03 | 11:55 | -$76 | -$87 | 9 bars | SM Flip | Weak SM (-0.075), entered on pullback that reversed |
| 30 | long | Feb 03 | 13:25 | -$68 | -$99 | 12 bars | SM Flip | Weak SM (+0.040), long above prev day high |
| 36 | long | Feb 05 | 12:00 | -$388 | -$554 | 8 bars | SM Flip | Second-worst. Strong SM (+0.410) but 275pt crash |
| 56 | short | Feb 13 | 14:55 | -$150 | -$208 | 3 bars | SM Flip | Short far below prev day low (-272pts) |
| 57 | short | Feb 13 | 15:45 | -$128 | -$221 | 4 bars | EOD | Late entry (15:45), no time to recover |

### 1.2 What the Losers Have in Common

**Pattern A: SM lagged during sharp counter-moves (Trades 15, 36)**
The two worst losses (-$452 and -$388) share the same root cause: SM stayed
directional while price made a violent move in the opposite direction. Trade 15
shorted at 14:35, SM was -0.123, then price rallied 225+ points while SM
remained bearish. Trade 36 went long at 12:00, SM was +0.410 (strong bullish),
then price crashed 275+ points while SM stayed bullish. These are
institutional-flow-lagging-price events -- exactly the problem identified in
forward testing.

**Pattern B: Weak SM at entry (Trades 29, 30)**
Both Feb 03 losers entered with very weak SM readings (|SM| < 0.10). SM at
-0.075 and +0.040 represent marginal conviction. Compare to winning trades
where average |SM| at entry was 0.193. While the current SM threshold is 0.0,
these trades suggest a minimum |SM| of ~0.05-0.10 might filter noise.

**Pattern C: Longs above prior day high (Trades 11, 30, 47)**
Four of the 6 long losers entered above the prior day's session high. This
is a "profit-taking zone" where institutions who bought the prior day may
sell into strength. The filter simulation shows this blocks 4 losers but
ALSO blocks 8 winners including some big ones (+$553, +$202, +$87). Net
effect is negative (-$1,365 of good trades removed). The filter is too
broad as-is.

**Pattern D: Shorts far below prior day low (Trades 56, 57)**
Both Feb 13 afternoon losers entered 227-272 points below the prior day
low. This is extreme overextension -- the easy short money was already
made hours ago. When price is this far below prior day support, a bounce
is likely.

**Pattern E: Late afternoon entries (Trades 15, 56, 57)**
Three big losers entered after 14:30. Afternoon win rate is 74% vs 81% for
AM and 82% for midday. The afternoon losers are also larger (-$155 average
vs -$61 AM and -$118 midday). Afternoon has more profit-taking, end-of-day
positioning, and news-driven reversals.

**Pattern F: Losers hold longer**
Losing trades held an average of 9.9 bars (50 min) vs 5.0 bars (25 min)
for winners. Winners resolve quickly -- if the trade is going to work, it
works within 15-25 minutes. Losers linger.

### 1.3 What Winners Look Like

- **Fast resolution:** Median hold time of 3 bars (15 min). Winners move in
  the trade direction quickly.
- **Moderate SM conviction:** Average |SM| of 0.193 at entry. Not extreme,
  but clearly directional.
- **Wide day ranges favor the strategy:** Days with 300+ point range
  produced the most profit. The strategy thrives on trending days.
- **Trades below the opening range** averaged $133/trade (10 trades) vs
  $53/trade above OR (20 trades) and $11/trade inside OR (24 trades).
  Below-OR trades were fewer but much more profitable per trade.
- **Direction-aligned trades** (longs above OR or shorts below OR) averaged
  $112/trade vs $31/trade for counter-direction trades. When price breaks
  the OR and SM confirms direction, the expected value is 3.6x higher.

### 1.4 Day Type Analysis

| Day Type | Trades | W/L | P&L | Avg/Trade |
|----------|--------|-----|-----|-----------|
| WIDE (>300pts) | 32 | 28/4 | +$2,055 | +$64 |
| MEDIUM (150-300) | 18 | 14/4 | +$391 | +$22 |
| NARROW (<150) | 7 | 5/2 | +$22 | +$3 |

Wide-range days produce 83% of the profit. Narrow days barely break even.
This is not surprising for a trend-following strategy -- it needs movement
to profit.

---

## 2. Market Structure Filter Ideas

### 2a. Prior Day High/Low as Support/Resistance

**Concept:** Prior day session high and low are key institutional reference
levels. Do not go long when price is already 50+ points above prior day high
(profit-taking zone). Do not go short when price is already 50+ points
below prior day low (capitulation exhaustion zone).

**Evidence from the data:** Trades 56 and 57 (Feb 13 afternoon) entered
short 227-272 points below prior day low and both lost. The raw "skip all
longs above prev high / all shorts below prev low" filter kills too many
winners (-$1,365), but a DISTANCE threshold could work. The 4 long losers
above prev day high were 62-200 points above it. Adding a distance buffer
(e.g., skip only when >150pts past prev day high/low) would be more
surgical.

**How it would work in Pine v6:** Use `request.security(syminfo.tickerid,
"D", high[1])` and `request.security(syminfo.tickerid, "D", low[1])` to
get prior day high/low. Compare entry price distance from these levels.
If `close > prev_day_high + buffer` and signal is long, skip. If
`close < prev_day_low - buffer` and signal is short, skip.

**Risks:** Prior day levels are only meaningful in context. On multi-day
trend moves (like the Jan 20-28 rally or the Feb 3-5 selloff), price
naturally trades well above prior day high or below prior day low. A
fixed buffer could block good trend-continuation trades. Need to test
multiple buffer values (25, 50, 100, 150 pts) and check which balances
protection vs missed opportunity.

**Risk of overfitting:** LOW. Prior day high/low is a universal market
structure concept that all institutional traders reference. Not a
parameter-sensitive construct.

---

### 2b. Current Day High/Low Proximity Filter

**Concept:** If price is within X points of the current session high,
skip long entries (the move may be exhausted). If within X points of
session low, skip short entries.

**Evidence from the data:** Only 1 long loser (#11) was within 20pts of
session high at entry. But 2 long winners were also near session high
and won. At wider thresholds, the data is inconclusive -- losers average
distance from session high is similar to winners. This filter would not
have caught the big losers (Trade 15 was 66pts from session high when it
entered short, Trade 36 was 62pts from session high when it entered long).

**How it would work in Pine v6:** Track running session high/low with
`var float sess_high = na` reset on each new day. Update with
`sess_high := math.max(sess_high, high)`. Skip longs if
`close > sess_high - buffer`.

**Risks:** Session high/low changes throughout the day. A trade near
session high at 10:30 is very different from near session high at 15:00.
Early session high proximity is less meaningful because the range hasn't
developed yet. Also, breakout trades (which are among the best trades in
this strategy) by definition occur at or near session high/low.

**Risk of overfitting:** LOW mechanically, but the buffer parameter is
sensitive and the data doesn't show a clear edge.

---

### 2c. Opening Range Breakout Filter

**Concept:** Use the first 30-minute range (10:00-10:30) as context.
Trades outside the opening range that are ALIGNED with the breakout
direction (long above OR, short below OR) get priority. Trades inside
the OR or counter-direction get tighter stops or are skipped.

**Evidence from the data:** This is one of the strongest signals in the
dataset:
- **Direction-aligned OR breakout trades:** 18 trades, 83% WR, $112/trade
- **Counter-direction trades:** 12 trades, 83% WR, $31/trade
- **Inside OR trades:** 24 trades, 79% WR, $11/trade

Trades that break the OR in SM's direction produce 3.6x more per trade
than counter-direction trades and 10x more than inside-OR trades.
Inside-OR trades are essentially noise -- 24 trades averaging $11/trade
is barely above commission.

**How it would work in Pine v6:** Calculate OR high/low from the first
30 min of the session using `ta.highest(high, 6)` and `ta.lowest(low, 6)`
anchored to the session open (or use `var` to track). Then:
- If `close > or_high` and signal is long: take the trade (aligned breakout)
- If `close < or_low` and signal is short: take the trade (aligned breakout)
- If inside OR: option to skip entirely, or tighten stop, or reduce size

**Risks:** The opening range varies enormously (from 14pts on a quiet day
to 175pts on a volatile day). A narrow OR makes "inside OR" meaningless
(everything breaks out). A wide OR makes "breakout" meaningless (price
stays inside all day). Could add an OR width filter: only apply OR logic
when OR range is between 20-100pts.

**Risk of overfitting:** MODERATE. The concept is well-established in
trading literature, but the specific time window and range thresholds
need careful testing. The 30-min window may not be optimal for MNQ.

---

### 2d. VWAP Proximity Filter

**Concept:** VWAP (Volume Weighted Average Price) is the most important
institutional reference price for the current session. Longs above VWAP
have institutional tailwind (price is above average execution). Shorts
below VWAP have tailwind. Counter-VWAP trades fight the flow.

**Evidence from the data:** Cannot directly measure (no volume data in
CSV), but the session open can serve as a rough VWAP proxy for early-day
trades. Trades below the session open that shorted averaged $189/trade
(6 trades). The aligned direction with respect to session open strongly
predicts profitability.

**How it would work in Pine v6:** `ta.vwap(close)` is built-in on Pine
and resets each session. Filter: only take longs when `close > vwap` and
only take shorts when `close < vwap`. Alternatively, use VWAP as a
softer filter: trades aligned with VWAP direction get normal treatment,
counter-VWAP trades get tighter stops.

**Risks:** In a strong trend, price can stay above or below VWAP all day.
The filter would block counter-trend mean-reversion trades that sometimes
produce huge winners (e.g., Trade 6: long at 15:20 after a selloff day,
+$239). Also, VWAP requires real volume data which is available on
TradingView for MNQ futures but NOT in the CSV for Python backtesting.

**Risk of overfitting:** LOW. VWAP is the most widely used institutional
benchmark. Every institutional algorithm references it. However, testing
in Python is harder without volume data.

---

### 2e. Higher Timeframe Candle Structure

**Concept:** Check if the current 15-min or 1-hour candle is bullish or
bearish. Do not take longs during a bearish HTF candle (fighting the
immediate flow). Do not take shorts during a bullish HTF candle.

**Evidence from the data:** The v10_features.md already notes that
15-min SM was too slow to help (stayed bullish all day during a selloff).
However, simple candle direction (close vs open of the current 15-min
bar) is different from 15-min SM. A 15-min bar that is closing bearish
means price has been falling for the last 15 minutes -- this is
information the strategy does not currently use.

**How it would work in Pine v6:** Use `request.security(syminfo.tickerid,
"15", close > open ? 1 : -1)` to get 15-min candle direction. Only take
longs when the 15-min candle is bullish, shorts when bearish. Could also
use 1-hour direction.

**Risks:** Candle direction mid-bar is noisy. A 15-min candle might be
bearish at minute 5 but flip bullish by minute 15. Using `request.security`
with `lookahead=barmerge.lookahead_off` means you get the PREVIOUS
completed bar's direction, which adds lag. Also, this is conceptually
similar to the RSI momentum filter which already exists and showed no
improvement on the 5-min RSI.

**Risk of overfitting:** MODERATE. Candle direction is simple, but the
timeframe choice (15-min vs 1-hour) adds a parameter to optimize.

---

### 2f. ATR-Based Overextension Filter

**Concept:** If price has already moved more than Nx the daily ATR from
the session open, the "easy" move is done. Skip new entries in the
direction of the move, or tighten stops.

**Evidence from the data:** The ATR filter test was disappointing. At
1.5x and 2.0x daily ATR, zero trades were filtered. At 1.0x, only 1
trade was filtered (a winner). The problem: MNQ daily ranges are large
(average ~400pts during this period) and the session open is already in
the middle of the range. Trades rarely exceed 1.5x ATR from open.

The more interesting finding is that losers average 36pts from open while
winners average 46pts from open. This is counter-intuitive -- losers are
closer to open, not farther. The overextension thesis does not hold in
this dataset at the daily ATR scale.

**However,** a 5-bar ATR filter is more promising. Losers entered after
5-bar ranges averaging 89pts vs 58pts for winners. Wide pre-entry ranges
correlate with worse outcomes. This makes intuitive sense: if the market
has been volatile right before entry, the SM signal is less reliable.

**How it would work in Pine v6:** Use `ta.atr(14)` on the 5-min chart
(inside a feature-gated block). Skip entries when the 5-bar range before
entry exceeds some multiple of ATR. Alternatively, skip entries when the
immediate 5-bar range exceeds a fixed threshold (e.g., 100pts).

**Risks:** The PINE DANGER is real -- `ta.atr()` is stateful and MUST
be gated. Also, wide 5-bar ranges sometimes precede the best trades
(Trade 37: 310pt range, then +$553). Filtering on volatility could cut
the biggest winners along with losers.

**Risk of overfitting:** MODERATE. The threshold for "too volatile" is
a tunable parameter with limited data to calibrate.

---

### 2g. Session High/Low Breakout Entry

**Concept:** Instead of only entering on RSI cross, also enter when
price breaks the current session high/low AND SM confirms the direction.
This captures breakout momentum that the RSI cross might miss.

**Evidence from the data:** The SM Flip Reversal test (Feature A)
attempted something similar by entering on SM flip. It produced 141
trades but the reversal trades were net negative (PF 0.94, -$438).
The problem: SM flips too frequently intraday (84 extra reversal trades).
A session high/low breakout is a much higher bar -- it only triggers
once per direction per day.

The current strategy already captures some breakout moves via RSI cross
(e.g., Trade 12: short after breaking OR low, +$298). But it misses
breakouts where RSI doesn't cross the threshold.

**How it would work in Pine v6:** Track `var float sess_high = na` and
`var float sess_low = na`, reset daily. On `high > sess_high[1]` and
SM bullish, enter long. On `low < sess_low[1]` and SM bearish, enter
short. Require breakout to be confirmed on a closed bar (not just an
intra-bar poke).

**Risks:** False breakouts are common. Price can poke above session high
and immediately reverse. Requiring SM confirmation helps, but SM already
lags. Also, breakouts near the session high/low tend to have poor
risk/reward -- you're buying at the day's highest price. Need a
minimum breakout distance (e.g., must close 5+ points above prior high).

**Risk of overfitting:** LOW. Session breakouts are a standard concept.
The risk is in the parameter tuning (minimum breakout distance, etc.).

---

### 2h. Consolidation Detection

**Concept:** If price has been in a tight range for N bars (low
volatility, building energy), the next SM signal could be a real
breakout. Prioritize entries after consolidation.

**Evidence from the data:** Pre-entry 5-bar range analysis shows
winners have a median of 47pts while losers have 35pts. This is
surprising and OPPOSITE to the expected pattern. Winners actually
entered after MORE volatile periods, not less. The average is skewed
by a few losers with very wide pre-entry ranges (Trades 56, 57 with
271 and 206 pts), but the median tells a different story.

This suggests that for SM-flip specifically, the SM signal is more
reliable when there's already momentum (higher pre-entry range) rather
than when the market is quiet. The SM indicator needs movement to
generate meaningful readings.

**How it would work in Pine v6:** Compute range of last N bars:
`ta.highest(high, N) - ta.lowest(low, N)`. If below a threshold,
flag as consolidation. Could be used as a positive filter (only enter
during consolidation) or negative (skip consolidation).

**Risks:** The data suggests this filter would not help and might hurt.
The strategy's winners actually come from momentum-rich environments,
not quiet consolidation periods. Filtering based on pre-entry range
could exclude the best setups.

**Risk of overfitting:** HIGH. The optimal range threshold and lookback
period would be heavily sample-dependent.

---

### 2i. Gap Fill / Gap-and-Go Detection

**Concept:** If there's an overnight gap (session open vs prior day
close), use SM direction after 10:00 to determine if it's a gap-fill
day (price retraces back to prior close) or gap-and-go day (price
continues in gap direction).

**Evidence from the data:** Looking at the daily OHLC, significant
gaps occurred frequently. Feb 02 gapped down ~80pts and then rallied
+798pts (gap and go up). Feb 05 gapped up ~30pts and then sold off
-876pts (gap and go down). The strategy captured big moves on these
days, but the gap direction itself did not predict the day type. SM
direction by 10:30 might be a better predictor.

**How it would work in Pine v6:** Compute gap size as
`open - request.security(syminfo.tickerid, "D", close[1])`. If SM
aligns with gap direction after the first 30 min, treat as gap-and-go
(let trades run longer). If SM opposes gap, treat as gap-fill (tighter
stops or skip counter-gap entries).

**Risks:** Gap classification requires looking at both gap size and
subsequent price action, which means you need to wait for the opening
range to complete before trading. This conflicts with the current 10:00
entry start. Also, not all days have meaningful gaps -- MNQ often gaps
by only 5-20pts which is noise.

**Risk of overfitting:** MODERATE. The gap size threshold and
classification logic have multiple parameters.

---

## 3. Exit Improvements (Non-Indicator)

### 3a. Time-Based Exit (Underwater Timer)

**Concept:** If a trade is underwater (negative P&L) after N bars, close
it. Good trades resolve quickly; bad trades linger.

**Evidence from the data:** This is one of the strongest findings.
Simulating different bar cutoffs:

| Cutoff | P&L | vs Baseline | Winners Cut | Losers Saved |
|--------|-----|-------------|-------------|--------------|
| 2 bars (10min) | $2,284 | -$185 | 0 | 2 |
| 3 bars (15min) | $2,799 | +$330 | 0 | 3 |
| 4 bars (20min) | $2,749 | +$281 | 0 | 1 |
| 5 bars (25min) | $2,721 | +$253 | 0 | 1 |
| 6 bars (30min) | $2,671 | +$203 | 0 | 0 |

**The 3-bar (15min) cutoff is the sweet spot.** It adds +$330 to the
baseline with ZERO winners cut short. After 15 minutes, if the trade
is still underwater, it is statistically unlikely to recover enough to
be a good trade.

The price path analysis confirms this: all big losers were already
negative at the 15-minute mark, while winners were already positive.
The 15-minute mark is a natural decision point where trade thesis
validation occurs.

**How it would work in Pine v6:** Track entry bar with
`var int entry_bar = 0`. Set `entry_bar := bar_index` on entry. On each
bar, if `bar_index - entry_bar >= 3` and unrealized P&L < 0, call
`strategy.close()`. This is similar to the max_loss stop but
time-triggered rather than price-triggered. Would need to be inside a
feature-gated `if` block.

**Risks:** The P&L check on bar close (not intra-bar) means a trade
could briefly dip negative then recover and still get stopped. Using
close-based P&L reduces this risk. Also, on very wide-range days,
15 minutes of underwater time might be normal for trades that eventually
win big. The 3-bar filter catches 0 winners in this dataset, but with
more data, some false positives will appear.

**Risk of overfitting:** LOW-MODERATE. The concept is sound (cut losers
early), but the exact cutoff (3 bars vs 4 vs 5) is sample-dependent.
The fact that 0 winners are cut is encouraging but may not hold OOS.

---

### 3b. Price Structure Exit (Prior Candle Breach)

**Concept:** If price breaks below the prior 15-min candle's low while
long (or above prior 15-min candle's high while short), exit. This uses
candle structure rather than SM to detect direction failure.

**Evidence from the data:** For Trade 15 (the worst loser at -$452),
the short entered at 14:35. The prior 15-min candle (14:15-14:30) had
a high of ~25,595. Price broke above this within the first 5 minutes
of the trade. A candle-breach exit would have capped the loss at ~$46
instead of $452.

For Trade 36 (-$388), the long entered at 12:00. The prior 15-min
candle had a low of ~25,020. Price broke below this after 15 minutes.
A candle-breach exit would have capped the loss at ~$110 vs $388.

**How it would work in Pine v6:** Use `request.security(syminfo.tickerid,
"15", low[1])` and `request.security(syminfo.tickerid, "15", high[1])`
to get the prior 15-min candle's range. If long and `low < prev_15m_low`,
exit. If short and `high > prev_15m_high`, exit.

**Risks:** This is effectively a tight trailing stop based on HTF candle
structure. It could exit trades prematurely during normal retracements.
Many winning trades dip below the prior 15-min candle low briefly before
continuing higher. The MAE data shows winners average worst excursion of
~$40-60, which likely includes intra-bar dips below prior candle levels.

**Risk of overfitting:** LOW mechanically, but the timeframe choice
(15-min vs 30-min vs 1-hour) matters.

---

### 3c. Trailing Structure Stop (Higher Lows / Lower Highs)

**Concept:** For longs, trail the stop below the most recent higher low
in the 5-min chart. For shorts, trail above the most recent lower high.
This lets winners run while protecting profits when the trend structure
breaks.

**Evidence from the data:** Winners that had large MFE (max favorable
excursion) sometimes gave back significant portions before SM flipped.
Trade 43 reached MFE of +$202 but SM didn't flip until exactly +$202 --
perfect. But Trade 47 reached MFE of +$96 then reversed to -$33 before
exit. A trailing structure stop could have locked in the $96 win.

**How it would work in Pine v6:** Track the most recent swing low (for
longs) using a simple N-bar lookback: `ta.lowest(low, 3)[1]` -- the
lowest low of the prior 3 bars. Place the stop 2-3 points below this
level. Update each bar. This creates a natural trailing stop that follows
the trend structure without a fixed distance.

**Risks:** In choppy markets, swing lows can be very close together,
leading to premature exits. The lookback period (3 bars vs 5 vs 7)
dramatically affects behavior. Too tight = stopped out of winners. Too
loose = no better than SM flip exit. Also, this adds a `ta.lowest()`
call which is stateful and must be properly gated.

**Risk of overfitting:** MODERATE. Swing detection parameters are
sensitive.

---

## 4. New Entry Types

### 4a. Breakout Entries

**Concept:** When price breaks the current session high/low and SM
confirms direction, enter without waiting for RSI cross. This captures
breakout momentum.

**Evidence:** The direction-aligned OR breakout analysis shows these
are the highest-expectancy setups: $112/trade for aligned breakouts vs
$11/trade for inside-OR trades. Currently, the strategy only captures
these via RSI cross, which may fire too late.

**Risk:** False breakouts. Need SM confirmation + closed bar above level.
Adding a new entry type increases trade frequency, which is good only if
the new entries have positive expectancy.

---

### 4b. Pullback Entries

**Concept:** After an OR breakout, price often pulls back to test the
breakout level. Enter on the pullback when SM still confirms direction
and RSI re-crosses the threshold.

**Evidence:** Several of the winning trades are essentially pullback
entries (e.g., Trade 9: long at 13:05 with SM at 0.241, this was after
a morning selloff and bounce). The strategy naturally catches some
pullbacks via RSI cross, but doesn't explicitly target them.

**Risk:** Pullback entries are harder to define precisely. How deep must
the pullback be? How do you distinguish a pullback from a reversal?
Adding pullback logic significantly increases complexity.

---

### 4c. Opening Drive Entries

**Concept:** If the first 15-30 minutes show a strong directional move
(large candle + SM confirms), enter on the first pullback within the
opening drive.

**Evidence:** The current 10:00 entry window means the strategy catches
opening drives, but only if RSI happens to cross during that window.
Morning trades (10:00-12:00) have 81% WR, so the window is already
productive. The main improvement would be allowing an earlier start
(09:45 or 09:50) to catch the initial momentum.

**Risk:** Pre-10:00 entries are noisier. The current 10:00 start exists
because the first few minutes after open are chaotic. Entering earlier
adds risk of stop-hunting wicks.

---

## 5. Priority Ranking

Ranked by (a) simplicity, (b) likelihood of improvement from the data,
(c) risk of overfitting:

### TIER 1: Test First

**1. Time-Based Exit -- 3-bar underwater cutoff (Section 3a)**
- Simplicity: HIGH (track bar count, check P&L, call strategy.close())
- Evidence: +$330 improvement, 0 winners cut, 3 big losers saved
- Overfit risk: LOW-MODERATE (concept is universal, exact bar count tunable)
- Pine complexity: One `var int`, one `if` block
- **Why first:** Best risk/reward of all ideas. Data strongly supports it.
  Even if the optimal cutoff is 4 or 5 bars instead of 3, the concept
  still works. And it requires zero new indicators -- just bar counting.

**2. Opening Range Direction Alignment (Section 2c)**
- Simplicity: MEDIUM (need OR high/low tracking + direction check)
- Evidence: 3.6x expected value for aligned trades vs counter-direction
- Overfit risk: LOW-MODERATE (OR breakout is a well-known concept)
- Pine complexity: Two `var float` variables, reset daily
- **Why second:** The data shows the clearest differentiation here.
  Inside-OR trades ($11/trade) are barely worth taking. As a soft filter
  (tighten stops for inside-OR trades rather than skip them), this could
  improve without losing trade count.

### TIER 2: Test After Tier 1

**3. VWAP Direction Filter (Section 2d)**
- Simplicity: HIGH (Pine has built-in `ta.vwap()`)
- Evidence: Cannot fully validate in Python (no volume), but session-open
  proxy shows strong directional bias
- Overfit risk: LOW (VWAP is the gold standard institutional reference)
- Pine complexity: One line for VWAP, one boolean condition
- **Why Tier 2:** Cannot backtest in Python, so must go straight to Pine.
  But the concept is strong enough to justify Pine-only testing.

**4. Prior Day Levels with Distance Buffer (Section 2a)**
- Simplicity: HIGH (request.security for prev day high/low)
- Evidence: 2 big losers (Feb 13 shorts) were 227-272pts below prev low.
  A 200pt buffer would catch these without blocking many winners
- Overfit risk: LOW (prior day levels are universal S/R)
- Pine complexity: Two `request.security()` calls, two conditions
- **Why Tier 2:** The raw filter hurts more than helps, but with a
  distance buffer it becomes surgical. Need to find the right buffer.

**5. Price Structure Exit (Section 3b)**
- Simplicity: MEDIUM (need 15-min candle data via request.security)
- Evidence: Would have saved $400+ on worst trade (#15)
- Overfit risk: LOW (candle structure exits are standard)
- Pine complexity: request.security for 15-min high/low[1]
- **Why Tier 2:** Promising but interacts with the SM flip exit. Need
  to ensure candle exit doesn't fire before SM flip on winning trades.

### TIER 3: Test Later

**6. Session Breakout Entry (Section 2g / 4a)**
- Adds new entry type = more complexity and interaction effects
- Need Tier 1/2 exit improvements stable before adding entries

**7. ATR / Volatility Filter on 5-bar range (Section 2f)**
- Data shows correlation (losers had wider pre-entry ranges) but also
  shows biggest winners came from wide ranges. Net effect unclear.

**8. Higher Timeframe Candle Direction (Section 2e)**
- Already investigated for 15-min SM (failed). Simple candle direction
  is different but likely has similar lag problems.

### DO NOT IMPLEMENT

**- Consolidation Detection (Section 2h):** Data shows opposite of expected
  pattern. Winners come from higher-volatility environments, not quiet
  consolidation. Would hurt.

**- Prior Day Filter Without Distance Buffer (Section 2a raw):** Kills
  $1,365 in winning trades to save $156 in losses. Net -$1,209.

**- SM Flip Reversal as Entry (from v10_features.md Tier 1b):** Already
  tested. Produces 84 extra trades at PF 0.94. The SM flips too often
  intraday to be a reliable standalone entry signal. RSI cross entry
  remains essential.

---

## Summary: What to Build Next

The v9 strategy's core logic is sound (78.9% WR, PF 2.725). The problem
is concentrated in 7 catastrophic trades that account for 97% of all
losses. The most impactful improvements target these big losers
specifically:

1. **3-bar underwater exit** -- cuts the 3 worst losers with zero cost to
   winners. Test in Python, then Pine. This is the single highest-ROI
   change available.

2. **OR direction alignment** -- reduces low-value trades (inside-OR) and
   prioritizes high-value setups (aligned breakouts). Can be implemented
   as a soft filter (tighter stops for inside-OR) to avoid reducing trade
   count.

3. **VWAP + prior day levels** -- standard institutional reference levels
   that can prevent entries in exhaustion zones. Test in Pine (VWAP needs
   real volume data).

The common thread: these are all about **trade quality filtering** -- not
adding new indicators, but using price structure context that every
institutional trader already watches. The strategy enters trades blind to
where price is relative to key levels. Adding that awareness should
reduce big losers without adding complexity.
