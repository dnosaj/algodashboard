# VIX Death Zone Gate (19-22) for MNQ Strategies

Implemented Mar 3, 2026. Blocks MNQ entries when prior-day VIX closed in 19-22.

## Research Summary

Reversal diagnostic (Mar 3) found VIX 19-22 is NET NEGATIVE for MNQ strategies.
Original analysis had a **look-ahead bias** (used same-day `vix_close` instead of
`vix_prev_close`). Fixed and re-validated — results hold with prior-day VIX.

### Re-validated Results (prior-day VIX, corrected)

| Strategy | Period | Normal PF | Death Zone PF | Death Zone $ | N |
|----------|--------|-----------|---------------|-------------|---|
| MNQ_V15 | IS | 1.019 | **0.731** | -$461 | 77 |
| MNQ_V15 | OOS | 1.361 | **0.849** | -$187 | 52 |
| MNQ_VSCALPB | IS | 1.497 | **0.763** | -$140 | 40 |
| MNQ_VSCALPB | OOS | 1.135 | 1.360 | +$139 | 22 |
| MES_V2 | IS | 1.777 | 1.252 | +$481 | 44 |
| MES_V2 | OOS | 1.083 | 0.959 | -$65 | 32 |

**V15**: Death zone PF < 1.0 on both IS and OOS. Clear signal.
**vScalpB**: IS clearly bad (PF 0.763). OOS flipped positive but on only 22 trades
(noisy). SL rate jumps on both splits (25%→32.5% IS, 27%→36.4% OOS). Gated
anyway — will track blocked signals to validate during paper trading.
**MES_V2**: Mixed results, NOT gated.

## Implementation

### Config (`engine/config.py`)
- `StrategyConfig` has `vix_death_zone_min` / `vix_death_zone_max` (default 0.0 = disabled)
- MNQ_V15: 19.0 / 22.0
- MNQ_VSCALPB: 0.0 / 0.0 (disabled — removed Mar 5, OOS was borderline on 22 trades)
- MES_V2: 0.0 / 0.0 (disabled)

### VIX Fetcher (`engine/vix_gate.py`) — rewritten Mar 6 (tastytrade primary)
- `async fetch_prior_day_vix_close(session=None)` → `float | None`
- **Primary: tastytrade DXLink Summary** (Mar 6): `Equity.get(session, 'VIX')` returns
  an index equity with `streamer_symbol='VIX'` and `is_index=True`. Subscribe to
  `Summary` event → `prev_day_close_price` field. The original Mar 5 failure was
  caused by guessing symbol formats (`$VIX.X`, `.VIX`, etc.) instead of looking up
  via `Equity.get()`. Tested live: returns correct prior-day close (e.g., 23.75).
- **Fallback: yfinance** if tastytrade fails or session unavailable.
- 15-second `asyncio.wait_for` timeout on tastytrade path.
- yfinance fallback: 10-second socket timeout, 7-day window, `.item()` pandas fix.
- **Fail-open**: returns `None` on any error → gate disabled, trading unblocked.
- Called at engine startup (with `tt_session`) + daily reset via `asyncio.create_task`.

### Runner (`engine/runner.py`) — updated Mar 5
- Stores `tt_session` on EngineState after tastytrade auth
- Startup VIX fetch: `await fetch_prior_day_vix_close(session=state.tt_session)`
- Daily reset VIX re-fetch: background `asyncio.create_task(_refetch_vix())`
  - Task reference stored in `state._vix_refetch_task` to prevent GC
- **Phantom signal logging**: when VIX gate blocks an entry, appends to
  `logs/vix_blocked_signals.csv` (timestamp, strategy, instrument, side, price,
  SM value, RSI value, reason). For post-hoc "would have entered" analysis.

### Safety Manager (`engine/safety_manager.py`)
- `_vix_close: float | None` field, `set_vix_close()` setter
- Gate in `check_can_trade()`: blocks entry if VIX in strategy's death zone range
- `manual_override=True` (from Resume button) bypasses the gate
- `reset_daily()` clears `manual_override` → gate re-engages next day
- `get_status()` exposes `vix_close` (global) + `vix_gated` (per-strategy boolean)

### Dashboard
- `types.ts`: `vix_close: number | null` in SafetyStatusData, `vix_gated: boolean` in SafetyStrategyStatus
- `SafetyPanel.tsx`: Header shows `VIX: XX.X` (or `—` if null). Per-strategy badge
  priority: PAUSED (red) > VIX GATE (amber) > ACTIVE (green).

## Safety Audit (Mar 3)

3 agents reviewed. All items SAFE except one minor RISK:
- Fail-open: SAFE
- Manual override bypass: SAFE
- Daily reset re-engagement: SAFE
- MES exclusion: SAFE
- No position/exit interference: SAFE
- Socket timeout: RISK (minor) — 10s sync call at daily reset blocks event loop.
  Fires at 18:00 ET during session gap, operationally acceptable.
- Config correctness: SAFE
- Status reporting: SAFE
- Dashboard types/layout/colors: all OK

## Paper Trading Validation Plan

1. Monitor `logs/vix_blocked_signals.csv` for blocked V15 entries
2. Cross-reference blocked signals against 1-min bars to compute MFE/MAE
3. Track: did portfolio Sharpe improve with the gate active?
4. vScalpB gate already removed (Mar 5) — OOS was borderline positive on only 22 trades

## Bug History

**Mar 4-5**: VIX gate silently broken. yfinance changed return type — `vix["Close"].iloc[-1]`
now returns a pandas Series instead of a scalar. `float(series)` raised ValueError,
caught by the fail-open handler. VIX was 21.15 (in death zone 19-22). Gate would have
blocked all MNQ entries. V15 had one losing trade (-$92.50) that would have been prevented.
Fixed Mar 5 by rewriting `vix_gate.py` to use tastytrade DXLink as primary source.

## Diagnostic Script Fix

`reversal_diagnostic.py` line 905: changed `vix_close` → `vix_prev_close` in the
death zone filter to eliminate look-ahead bias (VIX closes at 4:15 PM ET, after
most trades). The script already computed `vix_prev_close` via `shift(1)`.
