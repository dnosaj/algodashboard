# Backtesting Engine — TradingView Matching Notes

## Order of Operations Per Bar (matching TV)
1. Fill any pending orders at this bar's **Open** price
2. Update equity mark-to-market at bar's **Close**
3. Detect signals (crossover/crossunder) at bar's **Close**
4. Queue pending orders for next bar

## EMA Calculation
- Multiplier: `2 / (length + 1)`
- Seed value: SMA of the first `length` bars
- Applied from index `length-1` onward
- Seed location doesn't matter after ~100 bars (converges exponentially)

## Position Sizing (percent_of_equity = 100%)
- Commission-adjusted: `trade_value = equity / (1 + commission_rate)`
- This ensures `trade_value + entry_commission = equity` (total outlay never exceeds equity)
- `qty = trade_value / fill_price`
- Matches TradingView's internal sizing behaviour

## Commission
- Percent-based: `commission_value / 100` (e.g., 0.1% → 0.001)
- Applied on **both** entry and exit
- Entry commission = `position_value * rate`
- Exit commission = `exit_value * rate`
- Net PnL = gross PnL - entry_commission - exit_commission

## Crossover / Crossunder Detection
- Crossover: `prev_fast <= prev_slow AND curr_fast > curr_slow`
- Crossunder: `prev_fast >= prev_slow AND curr_fast < curr_slow`
- Uses shift(1) for previous bar values

## TradingView PineScript Settings (must match)
- `calc_on_every_tick=false` → signals on bar close, fill on next bar open
- `fill_orders_on_standard_ohlc=true` → fill at open price
- **`margin_long=0, margin_short=0`** → CRITICAL: set to 0% to avoid margin call mini-trades
- With 100% margin, TV generates ~27 spurious margin-call trades (tiny qty, -0.20% PnL each)
- These margin calls happen when entry signal fires and TV does a margin check
- They alter equity slightly, causing trade count and PnL to diverge from our engine

## Data Source
- Use TradingView CSV export for exact OHLC match (e.g., `INDEX_BTCUSD, 1D.csv`)
- TV export format: `time` (unix ts), `open`, `high`, `low`, `close` — no volume
- Bitstamp API available as fallback but prices differ slightly from INDEX:BTCUSD
- Yahoo Finance BTC-USD also differs — not recommended

## Reversal Behavior (Long ↔ Short)
- TV's `strategy.entry("Short")` implicitly reverses: closes any open long AND opens short on the same bar
- The engine supports this via reversal detection at step 3:
  - If both `long_exit` and `short_entry` fire on the same bar, both are queued
  - On the next bar's Open: exit fills first (position → flat), then entry fills immediately after
- Strategy signals must include cross-triggers: `short_entry` should also mark `long_exit`, and vice versa
- Without reversal signals, the engine treats long and short as independent (positions never overlap)

## Net Profit vs Open P&L
- `net_profit` = sum of **closed** trade PnLs only (matches TV's "Net Profit")
- `open_profit` = unrealised P&L of any position still open when data ends
- `final_equity` = `initial_capital + net_profit + open_profit`
- TV reports these separately: "Net Profit" (closed) and "Open P&L" (unrealised)

## Validation Approach
- Compare trade count (must match exactly with 0% margin)
- Compare first 5 trades (entry date, price, qty)
- Compare net profit, PF, max DD, win rate
- Small qty differences OK (commission drag accumulates slightly differently)
