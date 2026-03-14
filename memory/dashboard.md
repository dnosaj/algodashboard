# Dashboard & Session Replay System

Built Feb 19, 2026.

## Architecture

### Stack
- **Frontend**: React + Vite + TypeScript, port 3000
- **Charting**: Lightweight Charts v4 (TradingView open-source, ~45KB)
- **Backend**: FastAPI (same `api/server.py` that serves live engine)
- **Proxy**: Vite dev server proxies `/api` -> `localhost:8000`, `/ws` -> `ws://localhost:8000`

### Key Files
| File | Purpose |
|------|---------|
| `dashboard/src/App.tsx` | Main layout: status, instruments, chart, signals, trades, P&L, safety |
| `dashboard/src/components/PriceChart.tsx` | Candlestick chart + session save/load/replay + trade list |
| `dashboard/src/components/SafetyPanel.tsx` | Per-strategy pause/resume, qty override, drawdown toggle, force resume |
| `dashboard/src/components/TradeLog.tsx` | Trade table with strategy column + per-strategy P&L + portfolio total |
| `dashboard/src/components/SessionTradeList.tsx` | TradingView-style trade readout (MFE/MAE) |
| `dashboard/src/hooks/useWebSocket.ts` | WS connection + bar/trade/safety_status state + REST poll |
| `dashboard/src/types.ts` | BarData, Trade, Position (with strategy_id), SafetyStatusData, etc. |
| `api/server.py` | REST + WS endpoints, EventBridge (safety_status merge), Pydantic validation |
| `api/models.py` | Pydantic models: QtyOverrideBody, DrawdownToggleBody, etc. |
| `dashboard/src/components/AnalyticsPanel.tsx` | Supabase-powered: Drift cards, Equity curves, Daily stats (3 tabs) |
| `dashboard/src/components/IntelPanel.tsx` | Agent digest feed from Supabase `digests` table. Accordion, markdown rendering, 5-min poll |
| `dashboard/src/hooks/useSupabase.ts` | Supabase hook for analytics data (drift, rolling, dailyStats). 2-min poll |
| `dashboard/src/lib/supabase.ts` | Supabase client (VITE_SUPABASE_URL + VITE_SUPABASE_PUBLISHABLE_KEY) |
| `generate_session.py` | Offline session generation from Databento backtest data |

## Live Chart
- Bars stored in `TastytradeDataFeed._all_bars` (deque maxlen=50000)
- On dashboard connect: REST fetch `/api/bars/MNQ` for history, then WS `bar` messages for live updates
- Incremental update: if same timestamp, update last bar; otherwise append
- Chart always renders (even with 0 bars) so session LOAD controls are accessible on weekends
- Trade markers: green arrowUp (v15 LONG), red arrowDown (v15 SHORT), blue/orange for vScalpB
- Exit markers: circles with P&L text
- TODO: vScalpB needs unique color pair (currently shares v15 blue/orange)

### Pivot Point Supertrend Overlay (Mar 8)
- Port of LonesomeTheBlue's Pine v4 indicator, computed client-side from bar data
- Toggle: **PPST** button in chart header; gear icon expands settings panel
- Parameters (adjustable in UI): Pivot Period (default 3), ATR Factor (default 2), ATR Period (default 6)
- Renders as a line series on the candlestick chart; green when bullish, red when bearish
- Recomputes on parameter change or new bars
- Implementation: `computePivotPointSupertrend()` in `PriceChart.tsx`
- Single color line (follows current trend direction); not bi-color per-segment

## Session Save/Load
- **Save**: POST `/api/session/save` snapshots current bars + trades to `sessions/session_YYYY-MM-DD.json`
- **Auto-save**: Background asyncio task saves every 5 minutes + on shutdown. Only saves if at least one trade exited TODAY (ET date). Prevents overwriting prior-day sessions when engine runs across date boundaries (weekends, overnight). Additional guard in `_do_save_session`: refuses to overwrite a file whose stored date doesn't match today.
- **Load**: GET `/api/session/{filename}` returns session JSON; dashboard enters replay mode
- **List**: GET `/api/sessions` returns all saved session files for dashboard dropdown
- **Replay mode**: orange REPLAY badge, LIVE button to return, chart reloads full data
- Sessions stored in `live_trading/sessions/` (gitignored)

## Session Generation (Offline)
- `generate_session.py` loads Databento 1-min CSVs for MNQ + MES, runs all 3 strategies:
  - **vScalpA**: `run_backtest_tp_exit()` SM_T=0.0, RSI 8/60-40, TP=5, SL=50
  - **vScalpB**: `run_backtest_tp_exit()` SM_T=0.25, RSI 8/55-45, TP=5, SL=15
  - **MES v2**: `run_backtest_tp_exit()` SM_T=0.0, RSI 12/55-45, TP=20, SL=35, EOD 15:30
- Computes MFE/MAE for all trades (max favorable/adverse excursion from bar highs/lows)
- Trades labeled with `strategy_id` (MNQ_V15, MNQ_VSCALPB, MES_V2), sorted by entry time
- Includes MES bars in session JSON
- Usage: `python3 generate_session.py --date 2026-02-19` or `--days 5`

## Trade Readout (SessionTradeList)
Columns: # | Strategy (vA/vB/MES badge) | Side | Entry (date/time/price) | Exit (date/time/price) | Signal (TP/Max Loss/EOD) | Pts | P&L | MFE | MAE | Cumulative P&L

Summary header shows: trade count, W/L, net P&L, avg MFE, avg MAE.

## TP Backtest Engine (`run_backtest_tp_exit`)
- Same entry logic as v10 engine (SM direction, RSI cross, cooldown, session times)
- Configurable: sm_threshold, rsi_buy/sell, tp_pts, max_loss_pts, eod_minutes_et
- Exit on TP: prev bar close >= entry + tp_pts (fill at next open)
- Exit on SL: prev bar close breaches max_loss_pts (fill at next open)
- Exit on EOD: configurable (16:00 for MNQ, 15:30 for MES)
- NO SM flip exit

## Databento Data Pipeline
- MNQ: 12-month data (Feb 2025 – Feb 2026), multiple CSVs concatenated
- MES: 12-month data (Feb 2025 – Feb 2026), multiple CSVs concatenated
- Fetch script: `backtesting_engine/strategies/fetch_databento_data.py`
- API key in `live_trading/.env` (DATABENTO_API_KEY) — free tier has delayed RTH data

## SafetyPanel
- Per-strategy rows: pause/resume button, sizing controls, SL count, daily P&L, EXIT button
- **VIX Gate display** (Mar 3):
  - Header: `VIX: XX.X` (or `—` if fetch failed) next to 5d SL counter
  - Per-strategy badge priority: PAUSED (red) > VIX GATE (amber `#ffaa00`) > ACTIVE (green)
  - VIX GATE badge shows when strategy is blocked by VIX death zone but not otherwise paused
  - Driven by `vix_close` (SafetyStatusData) + `vix_gated` (SafetyStrategyStatus) from engine
- **Sizing controls** (Feb 23):
  - Strategies WITH partial exit (MES_V2): `Entry: [-][2][+]  TP1: [-][1][+]`
  - Strategies WITHOUT partial (MNQ_V15, MNQ_VSCALPB): `Qty: [-][1][+]`
  - Values: override if set, else config default. Min Entry=2 for partial strategies, Min Qty=1 for others
  - Sends `strategy_sizing` WS command with `{strategy_id, entry_qty?, partial_qty?}`
  - Blocked while positioned (server rejects if `strategy.position != 0`)
  - Cleared on Force Resume and daily reset
- Drawdown badge (NORMAL / REDUCED / EXTENDED PAUSE)
- Drawdown toggle (DD ON/OFF)
- Force Resume All: sends `force_resume` + `resume` (clears safety + re-enables trading)

## Sizing Override Architecture (Feb 23)
- **Strategy owns properties**: `active_entry_qty` / `active_partial_qty` — override first, fallback to config
- **Dual storage**: strategy `_entry_qty_override` + SafetyManager `StrategyStatus.qty_override` kept in sync
- **Single entry point**: `set_strategy_sizing()` in runner.py validates pair atomically, updates both
- **Immediate propagation**: direct attribute set (no "sync on next bar"), intra-bar monitor sees current values
- **API**: `POST /api/safety/strategy/{sid}/sizing` + `strategy_sizing` WS command
- **Backward compat**: old `strategy_qty` WS command routes through `set_strategy_sizing(sid, entry_qty=qty)`
- **Validation**: entry_qty >= 1, partial_qty < entry_qty (for partial strategies), blocked while positioned
- Entry logic completely untouched — only affects qty and exit leg management

## WS Event Flow
- Engine SafetyManager emits `"status_change"` on EventBus after every mutation
- EventBridge converts to `"safety_status"` WS event (NOT `"status"` — avoids blanking dashboard)
- Dashboard merges `safety_status` into existing StatusData (`{ ...prev, safety: data }`)
- 5s REST poll (`/api/status`) provides authoritative full state
- `"trade_update"` WS event (Feb 20): emitted on fill correction, matches trade by `(entry_time, strategy_id)` and replaces in-place (no duplicate)

## ET Time Handling (Feb 20)
- Server sends `d.time` as ET epoch (seconds) on bar WS events
- Dashboard uses `d.time` directly (with `typeof` guard for backward compat with pre-restart servers)
- `getEtOffset(utcEpoch)` in PriceChart.tsx: DST-aware offset via `Intl.DateTimeFormat.formatToParts()` — used for old sessions without `timezone: "ET"` field
- `getEtEpoch(isoTime, etEpoch?)`: returns ET epoch; uses server value if available, falls back to DST-aware conversion
- Session save now includes `entry_time_et_epoch` / `exit_time_et_epoch` on trade dicts
- `generate_session.py` now adds `"timezone": "ET"` + ET-shifted bar epochs + ET epoch trade fields

## News Event Alert Banner (Mar 6)
- `dashboard/src/lib/newsCalendar.ts`: Static 2026 calendar (FOMC, NFP, CPI, Retail Sales) with `getTodayEvents()` helper
- `dashboard/src/components/NewsAlert.tsx`: Amber banner between StatusPanel and InstrumentCards, hidden on quiet days
- ET date comparison via `Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York' })` — DST-safe
- Informational only — no automated pausing. Shows event, time, and which strategy to consider pausing
- MES v2 profits from all news events, so banner notes "MES v2 unaffected"
- Calendar needs manual update for 2027 dates

## Running the Dashboard
```bash
# Terminal 1: Engine (all 3 strategies)
cd live_trading && python3 run.py --broker tastytrade --instruments MNQ,MES

# Terminal 2: Dashboard dev server
cd live_trading/dashboard && npm run dev

# Open http://localhost:3000
```
