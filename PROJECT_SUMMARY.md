# NQ Trading Project — Complete Summary

## What This Project Is

A systematic effort to build and validate a Smart Money + RSI trading strategy for MNQ (Micro E-mini Nasdaq-100 futures) using a custom Python backtesting engine that exactly matches TradingView's Pine Script V6 execution model.

---

## The Strategy Concept

**Core Idea:** Use institutional money flow (Smart Money indicator) combined with RSI momentum to time entries, and exit when institutional conviction reverses.

**Signal Logic:**
- LONG: Smart Money net_index > 0 (institutions buying) AND RSI crosses above threshold
- SHORT: Smart Money net_index < 0 (institutions selling) AND RSI crosses below threshold
- EXIT: Smart Money flips direction (crosses zero) — lets winners run with the institutional flow

**Smart Money Indicator (AlgoAlpha "Smart Money Volume Index"):**
- Separates institutional vs retail money flow using PVI (Positive Volume Index) and NVI (Negative Volume Index)
- Applies RSI to each flow, computes buying/selling ratios
- Sums ratios over an index period, normalizes by peak value
- Output: net_index — positive means institutional buying, negative means institutional selling
- Source code in `backtesting_engine/smart_money_ind.txt`

---

## Key Findings

### 1. Commission Sensitivity
- 0.1% commission completely destroys profitability on 1-min MNQ (avg move ~0.02%)
- Realistic MNQ commission: ~0.005% ($0.52 per contract on ~$12,500 notional)
- Always use 0.005% or $0.52/contract for MNQ backtests

### 2. Optimal Timeframe
- **5-minute bars are optimal** for this strategy
- 1-min: Too noisy, too many false signals, commission drag
- 5-min: Good signal quality, enough trades for statistical significance
- 15-min+: Too few signals, misses intraday moves

### 3. The Multi-Timeframe Edge
- **1-min SM on a 5-min trading chart** was the best-performing SM source ("AA_prebaked")
- Higher-resolution SM signal (1-min) provides earlier detection of institutional flow changes
- 5-min RSI provides momentum confirmation without noise
- This MTF approach consistently outperformed same-timeframe SM computation

### 4. SM Flip Exit is the Key Innovation
- Fixed TP/SL (e.g., 15pt TP / 7pt SL) leaves massive profits on the table
- SM flip exit (close when Smart Money changes direction) lets winners run
- Trades catch moves of +112, +156, +210, +218 points when SM stays directional
- Average hold time ~7 bars (35 min on 5-min chart) — exits quickly when wrong
- SM flip exit produced PF 5.852 vs ~2.5 for fixed TP/SL

### 5. Fast SM Parameters Work Better for Intraday
- Fast params: index=15, flow=10, norm=300, ema=150
- Default (AlgoAlpha): index=25, flow=14, norm=500, ema=255
- Fast params respond quicker to intraday flow changes
- Both were tested; fast params consistently outperformed for intraday trading

### 6. RSI Threshold Tradeoffs
- RSI 55/45: More trades, lower quality per trade
- RSI 60/40: Balanced (used in F11 winner)
- RSI 65/35: Highest quality signals but fewer trades (24 vs 44)
- RSI cross (ta.crossover) better than RSI zone (RSI > level) for entry timing

### 7. Volume Synthesis for Futures
- MNQ 1-min CSV from TradingView has NO volume column
- Range method (High - Low) works as volume proxy for PVI/NVI calculation
- Range-weighted (range * |close - open|) is slightly better but marginal improvement

### 8. Look-Ahead Bias is Critical
- v7 backtest had look-ahead: used bar[i] values to decide entry at bar[i] open
- v8 fix: use bar[i-1] values for signals, enter at bar[i] open
- Look-ahead inflated results significantly — always use i-1 signals

---

## Strategy Evolution

### Round 1: Foundation (smart_money_rsi.py, V1-V6)
- **Commission:** 0.1% (unrealistic)
- **Finding:** Strategy concept works but commission kills it
- **Lesson:** Must use realistic commission for futures

### Round 2: Commission Fix + Timeframe (smart_money_rsi_v2.py, V7-V18)
- **Commission:** 0.005% (realistic)
- **Finding:** 5-min timeframe is optimal; fast SM params outperform defaults
- **Winner:** V15 (5-min, fast SM params)

### Round 3: Exit Mode Optimization (smart_money_rsi_final.py, F1-F20)
- **Focus:** Test different exit strategies (reversal, SM flip, RSI neutral, RSI color loss)
- **Winner:** F11 — SM flip exit
  - PF: 5.852 | Win Rate: 69.23% | Sharpe: 0.999 | MaxDD: -0.80% | Net: +$47.08 (4.71%)
  - 26 trades over 7 trading days

### Round 4: User Settings (smart_money_rsi_round4.py, U1-U25)
- **Focus:** Test user's custom TradingView settings (RSI=11, SM index=10, flow=8, norm=150)
- **Status:** Investigation phase

### Scalper Series (scalper v1-v5, scalp15pt v5-v8)
- **Focus:** Fixed TP/SL approach (10-20pt TP, 5-15pt SL)
- **Finding:** Fixed TP/SL underperforms SM flip exit significantly
- **v8 Key Fix:** Look-ahead bias corrected, validated against TradingView

### v9: SM-Flip Strategy (scalp_v9_smflip_backtest.py)
- **Focus:** Combine the best learnings: 1-min SM + 5-min RSI + SM flip exit
- **SM Sources Tested:**
  - AA_prebaked (1-min AlgoAlpha SM pre-baked into 5-min CSV) — BEST
  - AA_resamp (1-min AlgoAlpha resampled to 5-min)
  - Ours_fast (our SM with fast params on 1-min, resampled)
  - Ours_AA (our SM with AA-like params on 1-min, resampled)
- **Best Configs (AA_prebaked, RSI cross entry):**

| RSI | Levels | SM Thr | CD | Trades | WR% | PF | Net $1lot |
|-----|--------|--------|----|--------|-----|----|-----------|
| 10 | 55/45 | 0.00 | 3 | 44 | 63.6% | 2.325 | +$1,583 |
| 11 | 55/45 | 0.00 | 3 | 38 | 60.5% | 2.528 | +$1,477 |
| 12 | 55/45 | 0.05 | 3 | 36 | 63.9% | 2.514 | +$1,458 |
| 10 | 65/35 | 0.05 | 6 | 24 | 54.2% | 3.503 | +$1,142 |

- **Cross entry >> Zone entry** (70% of cross configs profitable vs 51% for zone)
- **AA_prebaked >> Other SM sources** (80% profitable, avg PF 1.49)

---

## TradingView Pine Script Implementation

### Current Strategy: scalp_v9_smflip.pine

**Architecture:**
- Runs on **1-minute** MNQ chart
- Reads AlgoAlpha SM via `input.source()` (same timeframe, no cross-TF issues)
- Computes 5-min RSI via `request.security(syminfo.tickerid, "5", ta.rsi(...))`
- SM flip exit on 1-min precision

**Critical strategy() Parameters (learned the hard way):**
```pine
strategy("SM+RSI v9 SM-Flip", overlay=true,
     initial_capital=1000,
     default_qty_type=strategy.fixed,
     default_qty_value=1,
     commission_type=strategy.commission.cash_per_contract,
     commission_value=0.52,
     slippage=0,
     margin_long=0,          // REQUIRED — without this, orders silently rejected
     margin_short=0,         // REQUIRED — same
     process_orders_on_close=false,
     calc_on_every_tick=false,
     fill_orders_on_standard_ohlc=true,  // REQUIRED — use standard OHLC fills
     max_bars_back=5000)
```

**TradingView Validation (Jan 19 - Feb 12, 2026):**

| Metric | Python Backtest | TradingView |
|--------|----------------|-------------|
| Trades | 44 | 45 |
| Win Rate | 63.6% | 62.22% |
| PF | 2.325 | 2.027 |
| Net P&L | +$1,583 | +$1,196.70 |

Close match confirms the strategy logic is correctly implemented.

### Pine Script Lessons Learned
- `input.source()` with `plot.style_linebr` returns `close` (default) for `na` bars when reading cross-timeframe — must use same-timeframe reads
- `margin_long=0, margin_short=0` is essential — without it TradingView silently rejects all orders
- `fill_orders_on_standard_ohlc=true` needed for consistent fill behavior
- Pine v6 quirks: no multi-line ternaries, no unicode in code, `hline` doesn't support `display.pane`, nested ternaries in function args can fail

---

## Overfit Analysis (RESOLVED)

### v9.4 on Short Window (Jan-Feb 2026)
- 44-45 trades on 17 trading days — statistically thin
- TradingView validated: PF 2.027, but only one window of data
- Could be regime-specific (high volatility, strong trends in Jan-Feb)

### Extended 6-Month Backtest (Aug 2025 - Feb 2026, Databento)
- v9.4 with AA defaults SM(20/12/400/255): 345 trades, PF 0.906, **-$984** over 6 months
- Monthly: Aug +$222, Sep -$873, Oct -$1,678, Nov -$1,018, Dec -$1,014, Jan +$1,173, Feb +$617
- Strategy has real regime sensitivity — profitable in 2/7 months with original params

### Resolution: v11 Parameter Sweep
- 6,180-combo parameter sweep on 6 months of MNQ 1-min Databento data
- Winner: SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50
- **v11**: 368 trades, 57.9% WR, PF 1.669, +$4,567, DD -$567
- AlgoAlpha hardcodes ema_len=255 — only way to use ema=100 is native SM computation
- **Walk-forward validation PASSED**: OOS PF 1.713 (better than IS 1.634), 6/7 months profitable
- **Lucky trade removal PASSED**: PF stays > 1.3 after removing top 5 trades
- **v11 is NOT overfit on MNQ**

---

## Feb 13 Forward Test & v9.4 Update

### v9.4: Max Loss Stop
- Added 50pt max loss stop via strategy.close() inside fully-gated `if max_loss_pts > 0` block
- With max_loss=0: identical to v9 (49 trades, same results)
- **MNQ** (Jan 19 - Feb 13): 50 trades, 60% WR, PF 1.649, +$1,071.50, MaxDD $421
- **MES** (Jan 19 - Feb 13): RSI 55/40, Max Loss=0, 35/55 trades, 63.64% WR, PF 2.165, +$904

### Stop Loss Implementation Lessons (v9.1-v9.4)
- v9.1: strategy.exit() fires silently, resets episode flags, caused re-entry (52 vs 49 trades)
- v9.2: var bool state tracking caused Pine side effects (37 trades)
- v9.3: boolean expressions with strategy.position_avg_price evaluated every bar (37 trades)
- v9.4: ALL stop code inside `if max_loss_pts > 0` -- Pine never touches strategy.position_avg_price when disabled = WORKS

### Forward Test Results (Feb 13, Day 1)
- **MNQ**: 5 trades, small net win. Trade 16 lost -$455 (SM stayed bullish through 235pt selloff). v9.4 stop would have capped at ~$100.
- **MES**: 2 trades, -$48 net. Same pattern: winner + afternoon loser held too long by SM.

### Key Discovery: MES SM != MNQ SM
- At MNQ Trade 16 exit, MNQ SM was +0.37 (bullish) but MES SM was -0.003 (bearish)
- Different instruments have different order flow -- SM reads them independently

---

## v10 Feature Validation (Feb 14) -- ALL REJECTED

Tested 11 feature variants on corrected 1-min engine across 7 criteria:
- Features tested: underwater exit, OR alignment, VWAP filter, prior day levels, price structure exit, SM reversal, RSI momentum
- **v9 baseline itself is NEGATIVE on 6-month 1-min data**: 308 trades, PF 0.871, -$1,385
- **ALL 11 features: REJECT** (0 ADOPT, 0 MAYBE, 11 REJECT)
- No feature has confidence interval excluding zero
- **Conclusion**: No bolt-on feature can fix the base strategy. The solution was parameter optimization (v11).

---

## Critical Engine Bugs Fixed (Feb 14)

1. **DST bug in session filtering**: Hardcoded UTC offsets (15:00 UTC = 10 AM ET) only correct for EST. During EDT (Mar-Nov), 10 AM ET = 14:00 UTC. Fixed with zoneinfo.
2. **Look-ahead bias in ALL exits**: Stop loss, underwater, price structure exits checked current bar's data but filled at current bar's open. Fixed: check bar i-1, fill bar i.
3. **RSI mapping look-ahead**: Mid-window 1-min bars getting current in-progress 5-min RSI. Fixed: always use rsi_5m_vals[j-1].
4. **RSI cross persistence mismatch**: Python fired RSI cross on 1 bar only at 5-min boundary. Pine persists across all 5 bars. Fixed: added rsi_5m_curr/rsi_5m_prev mapped arrays.
5. **CRITICAL LESSON**: Always backtest on 1-min bars (same as Pine). Never resample to 5-min.

---

## SM Computation Validation (Feb 14)

- Python compute_smart_money() with **AA params (20,12,400,255) matches AlgoAlpha TradingView**: r=0.985, mean abs diff=0.007
- 32/33 trades IDENTICAL between AlgoAlpha SM and Databento-computed SM
- Volume data: 100% match between TradingView and Databento (19,316/19,319 exact)
- **Fast params (15,10,300,150) diverge significantly** (r=0.87) -- do NOT use for backtesting against AlgoAlpha
- First ~400 bars show warmup discrepancy, then converge

---

## v11 Parameter Sweep & Native SM (Feb 14)

### The Sweep
- 6,180 parameter combinations on 6 months of MNQ 1-min Databento data (172K bars)
- Phase 1: 400 SM combos (index x flow x norm x ema)
- Phase 2: 5,600 RSI combos (top SM x RSI x levels x cooldown)
- Phase 3: 180 ATR/SL combos
- Winner: SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50

### Native SM Computation
- AlgoAlpha hardcodes ema_len=255 in their indicator
- The sweep found ema=100 is optimal for MNQ -- impossible to achieve with AlgoAlpha
- v11 computes SM natively using AlgoAlpha's open-source algorithm (MPL 2.0)
- Uses Pine's built-in `ta.pvi` and `ta.nvi` -- no external indicator required
- **SM params are now configurable inputs** in the strategy

### Walk-Forward Validation
- Split 1 (IS Aug-Nov, OOS Dec-Feb): OOS PF 1.713 (better than IS 1.634) = PASS
- Split 2 (IS Aug-Oct, OOS Nov-Feb): OOS PF 1.649 = PASS
- Leave-One-Month-Out: 6/7 months profitable, avg +$652/month
- Lucky trade removal: PF stays > 1.3 after removing top 5 trades
- **v11 is NOT overfit**

---

## Cross-Instrument Analysis (Feb 14) -- COMPLETE

### MES vs MNQ: Different Instruments Need Different Params
- v11 MNQ params SM(10/12/200/100) on MES: PF 1.080, +$388 (poor)
- v9.4 defaults SM(20/12/400/255) on MES: PF 1.351, +$1,653 (4.3x better)

### MES Parameter Sweep (Train/Test Split)
- Train: Aug-Nov, Test: Dec-Feb, 6,090 combos total
- **ALL 20 top sweep configs FAILED test validation** (PF < 1.0, ~55% degradation)
- v9.4 baseline: Train PF 1.16, Test PF 1.686 (negative degradation = PASS)
- The "slow" AA defaults genuinely suit MES's lower volatility

### Final Production Configuration

| Setting | MNQ (v11MNQ) | MES (v94MES) |
|---------|-------------|-------------|
| SM Index | 10 | 20 |
| SM Flow | 12 | 12 |
| SM Norm | 200 | 400 |
| SM EMA | 100 | 255 |
| RSI Length | 8 | 10 |
| RSI Buy/Sell | 60/40 | 55/45 |
| Cooldown | 20 bars | 15 bars |
| Max Loss | 50 pts | OFF |
| SM Source | Native (built-in) | Native (built-in) |
| Validation | Walk-forward PASS | Train/Test PASS |

Both strategies compute SM natively -- no AlgoAlpha indicator required.

---

## TradingView Validation (Feb 15, 2026) -- COMPLETE

### MNQ v11
| Metric | Python Backtest | TradingView |
|--------|----------------|-------------|
| Trades | 57 | 66 |
| Win Rate | -- | -- |
| PF | 2.297 | 1.681 |
| Net P&L | +$1,499 | +$1,174.86 |

- Through Feb 12 only: $16 PnL difference (1.1%). 10+ perfect trade matches.
- 9 extra trades in TV from data feed differences (net -$175, mostly 1-bar scratches)
- Feb 13 was a bad day: 3 consecutive max loss stops = -$308

### MES v9.4
| Metric | Python Backtest | TradingView |
|--------|----------------|-------------|
| Trades | 55 | 59 |
| PF | 2.318 | 2.089 |
| Net P&L | +$860 | +$846 |

- Through Feb 12: $20 PnL difference (2.3%). Very tight match.
- 5 extra trades in TV (net +$107), 4 missing (net +$111)

**Both VALIDATED and LIVE on TradingView from Feb 15, 2026.**

---

## Live Trading Bot (Feb 15, 2026)

### Architecture
Automated execution engine for MNQ + MES, designed to replace manual TradingView monitoring:

```
Dashboard (React + Vite, localhost:3000)
    |  WebSocket (real-time bar/signal/trade/status events)
API Server (FastAPI, localhost:8000)
    |  /api/status, /api/trades, /api/daily_pnl, /api/control/*
Trading Engine (Python, async)
    |  BarBuilder -> IncrementalStrategy -> OrderManager
    |  SafetyLayer (daily loss limit, position recon, kill switch)
    |  EventBus + Advisor plugins (extensibility for future AI)
Webull OpenAPI (HTTP bars + gRPC order status) [PENDING API ACCESS]
```

### Components Built
- **Incremental Strategy Engine** (`engine/strategy.py`): Bar-by-bar SM + RSI computation, validated trade-for-trade against vectorized backtest (368/368 trades match on 6-month data)
- **Event Bus** (`engine/events.py`): Pub/sub for bar, signal, trade, fill, status events. Advisors subscribe without touching core logic.
- **Safety Layer** (`engine/safety.py`): Daily P&L limits, consecutive loss circuit breaker, position reconciliation, heartbeat monitor, kill switch
- **API Server** (`api/server.py`): FastAPI with REST + WebSocket. EngineHandle abstraction decouples server from engine internals. EventBridge forwards sync engine events to async WebSocket broadcasts.
- **React Dashboard** (`dashboard/`): Status panel, trade log, daily P&L chart, pause/resume/kill controls. Dark theme, JetBrains Mono font.
- **Advisor System** (`advisors/`): Plugin architecture for future AI modules. Day 1: FixedSizeAdvisor (qty=1). Future: AI sizing, regime detection, smart exits.
- **Configuration** (`engine/config.py`): MNQ_V11 and MES_V94 presets with all params from validated Pine scripts.

### Current State
- Engine runs in paper mode with MockDataFeed (synthetic bars) and MockOrderManager (simulated fills)
- Dashboard is live at localhost:3000, API at localhost:8000
- Strategy parity tests pass: 368/368 trades match vectorized engine
- **Blocked on**: Webull API access approval (1-3 business days)

### What Happens When Webull API Arrives
1. Fill in `WebullDataFeed` (~100 lines): authenticate, poll 1-min bars, parse response
2. Fill in `WebullOrderManager` (~100 lines): place market orders, subscribe to gRPC fills
3. Run paper mode 3-5 days alongside TradingView for signal parity validation
4. Flip paper_mode=false for live execution

### Key Scripts
- `v10_test_common.py` -- Shared backtesting engine (corrected 1-min, all bug fixes)
- `v10_param_sweep.py` -- MNQ parameter sweep (6,180 combos)
- `v11_mes_param_sweep.py` -- MES parameter sweep with train/test split (6,090 combos)
- `v11_walk_forward.py` -- Walk-forward validation of v11 on MNQ
- `v11_mes_backtest.py` -- v11 cross-instrument test on MES
- `fetch_databento_data.py` -- Extended data download from Databento API

## Current Status (Feb 15, 2026)

### Production Pine Scripts (LIVE on TradingView)
- **`scalp_v11MNQ.pine`** -- MNQ: native SM(10/12/200/100), RSI(8/60/40), CD=20, SL=50
- **`scalp_v94MES.pine`** -- MES: native SM(20/12/400/255), RSI(10/55/45), CD=15, SL=0
- Both self-contained, no external indicators needed

### Live Trading Bot (BUILT, waiting for Webull API)
- Full-stack: Python engine + FastAPI + React dashboard
- Strategy parity validated (368/368 trades match)
- Advisor plugin architecture ready for future AI modules
- Blocked on Webull API access for real data feed + order execution

---

## File Reference

```
/Users/jasongeorge/Desktop/NQ trading/
├── PROJECT_SUMMARY.md                    <- This file
├── backtesting_engine/
│   ├── README.md                         -- Engine feature overview
│   ├── smart_money_ind.txt               -- AlgoAlpha indicator source (Pine)
│   ├── rsi_ind.txt                       -- RSI indicator source
│   ├── requirements.txt                  -- Python dependencies
│   ├── engine/
│   │   ├── engine.py                     -- Core backtest logic + indicators
│   │   ├── data.py                       -- Data loaders
│   │   └── __init__.py                   -- Public API
│   ├── data/
│   │   ├── databento_MNQ_1min_*.csv      -- 6-month MNQ 1-min (Databento, 172K bars)
│   │   ├── databento_MES_1min_*.csv      -- 6-month MES 1-min (Databento, 172K bars)
│   │   ├── CME_MINI_MNQ1!, 1_b2119.csv  -- MNQ 1-min with VWAP+Volume+SM (Jan 25-Feb 13)
│   │   ├── CME_MINI_MES1!, 1_cca38.csv  -- MES 1-min with VWAP+Volume+SM (Jan 25-Feb 13)
│   │   ├── CME_MINI_MNQ1!, 5_46a9d.csv  -- 5-min MNQ with pre-baked SM (Nov-Feb)
│   │   └── (older data files...)
│   └── strategies/
│       ├── -- PRODUCTION PINE SCRIPTS --
│       ├── scalp_v11MNQ.pine             -- v11 MNQ: native SM(10/12/200/100), PRODUCTION
│       ├── scalp_v94MES.pine             -- v9.4 MES: native SM(20/12/400/255), PRODUCTION
│       ├── -- ARCHIVED PINE SCRIPTS --
│       ├── scalp_v9_smflip.pine          -- v9 original (DO NOT modify)
│       ├── scalp_v9.4_smflip.pine        -- v9.4 with AlgoAlpha input.source()
│       ├── scalp_v11_standalone.pine     -- v11 prototype (before MNQ/MES split)
│       ├── scalp_v9.1_smflip.pine        -- BROKEN (strategy.exit stop)
│       ├── scalp_v9.2_smflip.pine        -- BROKEN (var bool tracking)
│       ├── scalp_v9.3_smflip.pine        -- BROKEN (boolean expression stop)
│       ├── scalp_v10_smflip.pine         -- v10: price structure exit (INVALIDATED)
│       ├── -- BACKTESTING SCRIPTS --
│       ├── v10_test_common.py            -- Shared engine (corrected, all bug fixes)
│       ├── v10_param_sweep.py            -- MNQ parameter sweep (6,180 combos)
│       ├── v11_mes_param_sweep.py        -- MES sweep with train/test (6,090 combos)
│       ├── v11_walk_forward.py           -- Walk-forward validation of v11
│       ├── v11_mes_backtest.py           -- v11 cross-instrument test on MES
│       ├── fetch_databento_data.py       -- Databento API data download
│       ├── scalp_v9_smflip_backtest.py   -- Original v9 Python backtest
│       ├── scalp_v9_signal_quality_test.py -- Signal quality filter tests
│       ├── PARAM_SWEEP_RESULTS.md        -- Full MNQ sweep results
│       ├── -- EARLIER ITERATIONS --
│       ├── smart_money_rsi.py            -- Round 1 (V1-V6, 0.1% commission)
│       ├── smart_money_rsi_v2.py         -- Round 2 (V7-V18, 0.005% commission)
│       ├── smart_money_rsi_final.py      -- Round 3 (F1-F20, best: F11)
│       └── (older scalper/pine files...)
├── live_trading/                          -- Live execution engine + dashboard
│   ├── run.py                            -- Entry point (python run.py --paper true)
│   ├── requirements.txt                  -- Python deps (fastapi, uvicorn, etc.)
│   ├── engine/
│   │   ├── config.py                     -- MNQ_V11 + MES_V94 presets, SafetyConfig
│   │   ├── events.py                     -- EventBus, Bar, Signal, TradeRecord types
│   │   ├── strategy.py                   -- IncrementalStrategy (bar-by-bar SM+RSI)
│   │   ├── bar_builder.py                -- 1-min bar aggregation + 5-min window tracking
│   │   ├── runner.py                     -- Main async loop, EngineState, orchestration
│   │   ├── safety.py                     -- Position recon, daily loss, kill switch
│   │   ├── order_manager.py              -- Order placement + confirmation (Webull stub)
│   │   └── data_feed.py                  -- DataFeed base, WebullDataFeed stub, MockDataFeed
│   ├── advisors/
│   │   ├── base.py                       -- Abstract Advisor interface
│   │   └── sizing.py                     -- FixedSizeAdvisor (qty=1)
│   ├── api/
│   │   ├── server.py                     -- FastAPI + WebSocket + EngineHandle + EventBridge
│   │   └── models.py                     -- Pydantic response models
│   ├── dashboard/
│   │   ├── package.json                  -- React + Vite + Recharts
│   │   ├── vite.config.ts                -- Proxy /api->:8000, /ws->ws://:8000
│   │   └── src/
│   │       ├── App.tsx                   -- Main layout
│   │       ├── types.ts                  -- TypeScript types matching API contract
│   │       ├── hooks/useWebSocket.ts     -- WS connection + REST polling
│   │       └── components/
│   │           ├── StatusPanel.tsx        -- Connection, positions, P&L
│   │           ├── TradeLog.tsx           -- Scrollable trade table
│   │           ├── DailyPnL.tsx          -- Bar chart + cumulative line
│   │           └── Controls.tsx          -- Pause/Resume/Kill buttons
│   └── tests/
│       ├── test_strategy.py              -- Parity: incremental == vectorized (368/368)
│       ├── test_safety.py                -- Safety layer unit tests
│       └── test_order_manager.py         -- Order flow mock tests
```
