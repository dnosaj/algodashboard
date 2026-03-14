# Pine Script v6 Gotchas

Accumulated bugs and lessons from implementing strategies in TradingView Pine v6.

## Strategy Declaration
- `margin_long=0, margin_short=0` REQUIRED in strategy() or orders silently rejected
- `fill_orders_on_standard_ohlc=true` REQUIRED for consistent fills
- When pasting new code with a different strategy name, TradingView resets Properties (date range etc.) — always re-check Properties tab

## Syntax
- NO unicode characters (em-dashes, arrows, emoji) — ASCII only
- NO multi-line ternary operators
- `hline()` does NOT support `display=display.pane` (but `plot()` does)
- `input.source()` returns `close` (default) for `na` bars when reading cross-timeframe with `plot.style_linebr`

## Short-Circuit Bug (CRITICAL)
- Pine does NOT short-circuit `and` — ALL operands are evaluated
- Accessing `strategy.position_avg_price` or calling `ta.rsi()` even in false expressions STILL causes side effects
- Adding stateful functions (ta.rsi, ta.ema, etc.) even behind toggles can change results
- **Fix**: Wrap ALL related code inside `if feature_enabled` blocks so Pine never executes the inner code

## Stop Loss Implementation
- Use `strategy.close()` not `strategy.exit()` for stop losses when you need episode flags (long_used/short_used) to remain set after the stop fires
- `strategy.exit()` fires silently outside your code flow
- Stop loss implementation history: v9.1 (strategy.exit = silent re-entry), v9.2 (var bool = side effects), v9.3 (boolean expressions = side effects), v9.4 (fully-gated if block = WORKS)

## Episode Reset with SM Threshold
- `not sm_bull` (SM <= threshold) flickers in the 0-to-threshold zone, causing repeated entries in choppy conditions
- **Fix**: Only reset on zero-crossing: `if sm_flipped_bull` not `if not sm_bull`
- Same bug existed in live engine strategy.py — fixed Feb 19

## Multi-Entry Strategies
- `pyramiding=N` needed (default 1 silently rejects 2nd entry)
- `close_entries_rule="ANY"` needed (default FIFO ignores entry ID in strategy.close)
- BUT shared position = shared risk — use separate strategies for truly independent legs
