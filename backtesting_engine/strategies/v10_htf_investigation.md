# Higher Timeframe (HTF) Extensions for SM+RSI v10

## Context

The v10 strategy already uses a basic multi-timeframe approach: 1-min bars with 5-min RSI via `request.security()`. These extensions take that further by using higher timeframe SM data to filter entries, improve exits, or replace the base timeframe entirely.

---

## Approach 1: HTF SM Direction Filter (Recommended First Test)

**Concept:** Compute SM on 15-min or 1-hour. Only allow entries that align with the HTF SM direction.

- HTF SM > 0 → only allow longs
- HTF SM < 0 → only allow shorts

**Why it should work:**
- Filters counter-trend trades where LTF SM has a brief blip against the bigger trend
- Should improve win rate at the cost of trade count
- Especially promising on MYM where 1-min SM is barely predictive (48-52% hit rate) — HTF SM may be more reliable

**Pine implementation (add to standalone):**
```pine
// ── HTF Filter ──
use_htf_filter = input.bool(false, "Enable HTF SM Filter", group="Higher Timeframe")
htf_timeframe  = input.timeframe("15", "HTF Timeframe", group="Higher Timeframe")

sm_htf = request.security(syminfo.tickerid, htf_timeframe, sm_net_index, lookahead=barmerge.lookahead_off)
htf_long_ok  = not use_htf_filter or sm_htf > 0
htf_short_ok = not use_htf_filter or sm_htf < 0
```

Then add to entry conditions:
```pine
long_entry  = [existing conditions] and htf_long_ok
short_entry = [existing conditions] and htf_short_ok
```

**Testing plan:**
1. Load standalone on MNQ 1-min chart with HTF filter OFF → record baseline trades/PF/WR
2. Enable HTF filter with 15-min → compare
3. Try 60-min → compare
4. Repeat on ES and MYM
5. Look for: fewer trades, higher win rate, similar or better PF

**Expected outcome:** Trade count drops 30-50%, win rate increases 5-10%, PF improves.

---

## Approach 2: Run Strategy on 15-min Bars Directly

**Concept:** Instead of 1-min bars + 5-min RSI, use 15-min bars + 1-hour RSI.

**Why it should work:**
- Commission drag drops dramatically — avg 15-min move is ~3-4x larger than 5-min, so fixed $0.52 becomes proportionally smaller
- SM signal should be cleaner with more data per bar
- Fewer trades but higher quality

**Setup on TradingView:**
1. Switch chart to 15-min timeframe
2. Load standalone Pine Script
3. Change RSI `request.security()` timeframe from "5" to "60" (1-hour)
4. SM computes natively on 15-min bars (no change needed)
5. Adjust cooldown from 15 bars (15 min on 1-min chart) to 3 bars (45 min on 15-min chart)

**Pine changes needed:**
```pine
// Change RSI timeframe input default
rsi_tf = input.timeframe("60", "RSI Timeframe")  // was "5"

// Adjust cooldown default
cooldown_bars = input.int(3, "Cooldown Bars")  // was 15
```

**Testing plan:**
1. Load on 15-min MNQ chart with 1-hour RSI
2. Compare PF, WR, trade count vs 1-min baseline
3. Try RSI on 30-min as middle ground
4. Test same configs on ES and MYM
5. MYM should benefit most — commission drag (currently 6.7% of 5-min range) would drop to ~2% on 15-min

**Expected outcome:** Fewer trades (5-10 per week vs 20-30), higher average profit per trade, lower commission impact. May need wider RSI levels (55/45) to maintain trade count.

**Tradeoff:** Harder to validate with current data — 25 days of 15-min bars gives very few trades to draw conclusions from. Need more data before trusting results.

---

## Approach 3: Multi-Timeframe SM Confluence

**Concept:** Score SM alignment across 2-3 timeframes. Only enter on high-confluence setups.

| 5-min SM | 15-min SM | 1-hour SM | Score | Action |
|----------|-----------|-----------|-------|--------|
| Bullish  | Bullish   | Bullish   | 3/3   | Strong entry |
| Bullish  | Bullish   | Bearish   | 2/3   | Optional — test both ways |
| Bullish  | Bearish   | Bearish   | 1/3   | No entry |

**Pine implementation:**
```pine
sm_5m  = sm_net_index  // already computed on base timeframe
sm_15m = request.security(syminfo.tickerid, "15", sm_net_index, lookahead=barmerge.lookahead_off)
sm_60m = request.security(syminfo.tickerid, "60", sm_net_index, lookahead=barmerge.lookahead_off)

sm_bull_score = (sm_5m > 0 ? 1 : 0) + (sm_15m > 0 ? 1 : 0) + (sm_60m > 0 ? 1 : 0)
sm_bear_score = (sm_5m < 0 ? 1 : 0) + (sm_15m < 0 ? 1 : 0) + (sm_60m < 0 ? 1 : 0)

min_confluence = input.int(2, "Min SM Confluence (1-3)", minval=1, maxval=3)
htf_long_ok  = sm_bull_score >= min_confluence
htf_short_ok = sm_bear_score >= min_confluence
```

**Testing plan:**
1. Test with min_confluence = 2 (majority rules) vs 3 (full alignment)
2. Compare trade count and quality vs single-timeframe baseline
3. Full alignment (3/3) will be very selective — may only get 3-5 trades per week

**Expected outcome:** Highest win rate of all approaches, but lowest trade count. Best suited for larger contracts (NQ, ES, YM) where fewer high-quality trades matter more than volume.

---

## Approach 4: HTF SM as Early Exit Signal

**Concept:** Use HTF SM flip to exit positions faster, rather than waiting for LTF SM to fully reverse.

**Why it should work:**
- Directly addresses the MYM problem where losers were caused by slow LTF SM flip exits
- A 15-min SM flip fires sooner than waiting for 1-min SM to fully reverse
- Doesn't change entry logic at all — only affects exits

**Pine implementation:**
```pine
use_htf_exit = input.bool(false, "Enable HTF SM Exit", group="Higher Timeframe")
htf_timeframe = input.timeframe("15", "HTF Exit Timeframe", group="Higher Timeframe")

sm_htf = request.security(syminfo.tickerid, htf_timeframe, sm_net_index, lookahead=barmerge.lookahead_off)

// Exit long if HTF SM turns bearish (even if LTF SM still bullish)
if use_htf_exit and strategy.position_size > 0 and sm_htf < 0
    strategy.close("Long", comment="HTF_SM_FLIP")

if use_htf_exit and strategy.position_size < 0 and sm_htf > 0
    strategy.close("Short", comment="HTF_SM_FLIP")
```

**Testing plan:**
1. Enable on MYM first (where slow exits are the known problem)
2. Compare exit timing — does HTF exit fire before the big losers develop?
3. Check if it also cuts winners short (the risk)
4. Try 15-min and 30-min as HTF exit timeframes

**Expected outcome:** Reduces max loss per trade, may slightly reduce average winner. Net effect should be positive on MYM, unclear on MNQ/ES where exits already work well.

---

## Testing Priority

| Priority | Approach | Effort | Expected Impact |
|----------|----------|--------|-----------------|
| 1 | HTF SM Direction Filter | Low — one toggle + one `request.security()` | High — filters bad trades |
| 2 | HTF SM Early Exit | Low — one toggle + exit condition | Medium — fixes MYM losers |
| 3 | 15-min Direct | Medium — timeframe + parameter adjustments | Medium — needs more data |
| 4 | MTF Confluence | Medium — multiple `request.security()` calls | High but very few trades |

**Start with #1.** It's the simplest change, fully toggleable, and doesn't alter core logic. If it improves results on all three instruments, consider combining it with #2 for a "HTF-informed" mode.

---

## Important Notes

- All `request.security()` calls MUST use `lookahead=barmerge.lookahead_off` to prevent future data leakage
- HTF SM values only update when the HTF bar closes — there will be lag. This is a feature (stability) not a bug
- When backtesting on TradingView, HTF data availability depends on your chart's history depth
- Python backtesting of HTF approaches requires resampling the 1-min data to 15-min/60-min timeframes — doable but adds complexity
- These approaches are additive — they can be combined (e.g., HTF filter for entries + HTF exit for exits)
