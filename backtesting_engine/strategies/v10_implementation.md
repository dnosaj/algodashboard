# SM+RSI v10 Production — Implementation Summary

**File:** `scalp_v10_production.pine`
**Lines:** 579
**Based on:** `scalp_v9_smflip.pine` (226 lines)

## What's preserved (zero changes to signal logic):
- SM computation, direction, and flip detection
- 5-min RSI via `request.security()` with `lookahead_off`
- RSI cross entry logic
- Episode flags (`long_used` / `short_used`)
- SM flip exit via `strategy.close()`
- EOD close at 16:00 ET
- Session filter and cooldown mechanism

13 sections in the code are explicitly marked with `// (PRESERVED EXACTLY FROM v9)` comments for auditability.

## New safety layers added:

| Feature | Default | Details |
|---------|---------|---------|
| **Hard Stop Loss** | ON, ATR 3x14 | ~9-15 pt stop on MNQ. Wide enough SM flip fires first normally. ATR or Fixed mode. |
| **Trailing Stop** | OFF | 15 pt activation, 8 pt trail. Enable after forward-testing. |
| **Breakeven Stop** | OFF | 8 pt trigger, 0.5 pt offset (covers commission). Only applies if tighter than hard SL. |
| **Daily Loss Limit** | ON, 3% | Blocks new entries when daily equity drops 3%. Existing positions stay open. |
| **Max Consecutive Losses** | OFF (0) | Optional pause after N losses. Streak spans overnight intentionally. |
| **Slippage** | 1 tick | Was 0 in v9 — adds realism. |
| **5 Alert Conditions** | Yes | Long entry, short entry, SM flip exit, daily limit, consec limit. |
| **16-row Live Dashboard** | Yes | Position, open P&L, stop levels/status, daily stats, risk status, config. |
| **Stop Level Lines** | Yes | Red (hard SL), orange (trail), blue (BE) lines on chart. |

## 8 input groups:
1. Smart Money (3 inputs)
2. RSI Settings (3 inputs)
3. Hard Stop Loss (6 inputs)
4. Trailing Stop (3 inputs)
5. Breakeven Stop (3 inputs)
6. Daily Risk Management (3 inputs)
7. Session Filter (3 inputs)
8. Display & Alerts (5 inputs)

## Exit priority (highest first):
1. **EOD Close** — non-negotiable 16:00 ET
2. **Hard Stop Loss** — `strategy.exit()` with `loss=` ticks (fires intrabar)
3. **Trailing Stop** — `strategy.exit()` with `trail_points`/`trail_offset` (fires intrabar)
4. **Breakeven Stop** — `strategy.exit()` with `stop=` price (fires intrabar)
5. **SM Flip Exit** — `strategy.close()` (fires at bar close)

## Daily risk tracking subsystem:
- `day_start_equity` resets on each new day via `ta.change(time("D"))`
- Running `day_pnl` and `day_pnl_pct` computed every bar
- Consecutive losses tracked via `strategy.closedtrades.profit()` — intentionally NOT reset on new day
- Two risk gates: `daily_paused` and `consec_paused` → combined into `risk_ok`
- When all risk features are OFF, `risk_ok` is always `true` (exact v9 behavior)

## Dashboard rows:
| Row | Label | Content |
|-----|-------|---------|
| 0 | Version | v10 Production + AA link status |
| 1 | SM Index | Current value, green/red coloring |
| 2 | 5m RSI | Current value, colored by buy/sell level |
| 3 | Position | LONG / SHORT / FLAT |
| 4 | Open P&L | Dollar amount, green/red |
| 5 | Hard SL | Distance in pts + mode (ATR/Fixed) or OFF |
| 6 | Trail | OFF / ARMED / ACTIVE |
| 7 | Breakeven | OFF / ARMED / LOCKED |
| 8 | Separator | --- Daily --- |
| 9 | Day P&L | Dollar + percentage |
| 10 | Day W/L | Win/Loss count |
| 11 | Consec L | Current streak (/ max if configured) |
| 12 | Risk | OK / DAILY LIMIT / CONSEC LIMIT |
| 13 | Separator | --- Config --- |
| 14 | RSI Cfg | RSI length + buy/sell levels |
| 15 | CD / Thr | Cooldown bars + SM threshold |

## Regression test:
To verify v10 matches v9 exactly, set these in TradingView:
- Hard Stop Loss: **OFF**
- Trailing Stop: **OFF**
- Breakeven Stop: **OFF**
- Daily Loss Limit: **OFF**
- Max Consecutive Losses: **0**
- Strategy Properties → Slippage: **0**

Trade count, win rate, and profit factor should be identical to v9.
