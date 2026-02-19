# Plan: SM+RSI v10 Production Pine Script

## Context

The SM+RSI v9 SM-Flip strategy has proven itself across two instruments (MNQ: PF 2.10, MES: PF 2.38) with nearly identical settings. The core signal logic — SM direction + RSI cross entry, SM flip exit — is sound and should not be touched. However, v9 runs with **no stop loss** and **no daily risk limits**, making it vulnerable to black swan events and cascading losses in production. This plan adds layered safety features while preserving every line of signal logic.

## What Changes (and What Doesn't)

### Preserved Exactly (zero changes):
- SM computation (`nz(sm_buy_src) + nz(sm_sell_src)`)
- SM direction/flip detection
- 5-min RSI via `request.security()` with `lookahead_off`
- RSI cross entry logic
- Episode flags (`long_used` / `short_used`)
- SM flip exit (`strategy.close()` on zero-cross)
- EOD close at 16:00 ET
- Session filter (entry window + close window)
- Cooldown mechanism

### New Safety Layers:

| Feature | Default | Rationale |
|---------|---------|-----------|
| **Hard Stop Loss** (ATR-based) | ON, 3x ATR(14) | Wide enough to almost never fire during normal SM-flip exits. Catches flash crashes. ATR adapts to any instrument automatically. |
| **Trailing Stop** | OFF | Available but off — SM-flip exit already handles most profit-taking well. User enables after forward-testing. |
| **Breakeven Stop** | OFF | Available but off — same reasoning. Moves stop to entry+offset after trade moves X pts in favor. |
| **Daily Loss Limit** | ON, 3% | Stops new entries if daily equity drops 3%. MNQ worst day was ~1-2%, so 3% is conservative circuit breaker. |
| **Max Consecutive Losses** | OFF (0) | Available if user wants to pause after N losses in a row. |
| **Slippage** | 1 tick | Was 0 in v9 — adds realism without being pessimistic. |
| **Alert Conditions** | Yes | Entry, SM flip exit, risk events — for mobile notifications in live trading. |
| **Live Dashboard** | Yes | Real-time table showing position, open P&L, stop levels, daily stats, risk status. |
| **Stop Level Lines** | Yes | Visual lines on chart showing hard SL, trail, and BE levels when in a position. |

## Architecture

**File:** `scalp_v10_production.pine`

**Exit Priority (highest first):**
1. EOD Close — non-negotiable 16:00 ET
2. Hard Stop Loss — `strategy.exit()` with `loss=` ticks (fires intrabar)
3. Trailing Stop — `strategy.exit()` with `trail_points`/`trail_offset` (fires intrabar)
4. Breakeven Stop — `strategy.exit()` with `stop=` price (fires intrabar)
5. SM Flip Exit — `strategy.close()` (fires at bar boundary)

This priority is natural: `strategy.exit()` orders are standing broker-side orders that fire intrabar, while `strategy.close()` fires at bar close. So hard stops always protect before SM flip evaluates.

**Entry Gate Addition:**
```
risk_ok = not daily_paused and not consec_paused
long_entry = [all v9 conditions] and risk_ok
```

When all risk features are disabled, `risk_ok` is always `true` and behavior is identical to v9.

## Input Groups (8 total)

1. **Smart Money** — unchanged (3 inputs)
2. **RSI Settings** — unchanged (3 inputs)
3. **Hard Stop Loss** — enable toggle, ATR vs Fixed mode, ATR length/multiplier, fixed points (6 inputs)
4. **Trailing Stop** — enable toggle, activation threshold, trail distance (3 inputs)
5. **Breakeven Stop** — enable toggle, trigger points, offset above entry (3 inputs)
6. **Daily Risk Management** — daily loss %, max consec losses, cooldown (3 inputs)
7. **Session Filter** — unchanged (3 inputs)
8. **Display & Alerts** — expanded with stop lines and dashboard toggles (5 inputs)

## Key Defaults

- **Hard SL:** ATR mode, 14-period, 3.0x multiplier. On MNQ 1-min (ATR ~3-5 pts), this gives ~9-15 pt stop. Wide enough for normal SM-flip to fire first.
- **Trail:** 15 pts activation, 8 pts trail distance. Only locks in on trades that are already big winners.
- **Breakeven:** 8 pts trigger, 0.5 pts offset (covers $1.04 round-trip commission).
- **Daily loss:** 3% of starting daily equity.
- **Adaptive SM Threshold:** OFF by default. When enabled, uses ATR ratio to scale SM threshold — higher vol = require more SM conviction, lower vol = accept less.

## Daily Risk Tracking (new subsystem)

- Track `day_start_equity` on each new day (`ta.change(time("D"))`)
- Compute running `day_pnl = strategy.equity - day_start_equity`
- Use `strategy.closedtrades.profit()` to detect trade outcomes for consecutive loss counting
- `consec_losses` intentionally NOT reset on new day (a streak spanning overnight is still a streak)

## Dashboard Table (16 rows)

Real-time display at top-right: version/link status, SM value, RSI value, position state, open P&L, hard SL distance, trail status (OFF/ARMED/ACTIVE), BE status (OFF/ARMED/LOCKED), daily P&L, day trades (W/L), consecutive losses, daily limit status, RSI config, cooldown.

## Instrument Compatibility

Uses `syminfo.mintick` for all tick conversions. ATR-based stops auto-scale across MNQ/NQ/MES/ES. Only `commission_value` in the strategy properties needs adjustment per instrument.

## Edge Cases Handled

- ATR not calculated on first bars → `nz(atr_val, 10.0)` fallback
- BE stop worse than hard SL → only apply BE if it's tighter than SL
- Daily limit hit mid-trade → existing positions stay open, only new entries blocked
- Multiple `strategy.exit()` on same entry → Pine auto-cancels the losing one
- Position goes flat from `strategy.exit()` → cooldown still resets correctly

## Verification

1. **Regression:** Load v10 with hard SL OFF, trail OFF, BE OFF, daily limit OFF, slippage=0. Results should match v9 exactly (same trade count, WR, PF).
2. **Hard SL only:** Enable ATR 3x. Trade count should be nearly identical to v9 (stop almost never fires in this data).
3. **Cross-instrument:** Load on MES — ATR stop should auto-adapt to S&P volatility.
4. **Daily limit:** Enable 3%. Confirm no day in backtest triggers it (conservative setting).
