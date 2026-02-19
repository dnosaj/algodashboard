# Backtest Engine

A Python backtesting engine that **exactly matches TradingView's Strategy Tester** results — same fills, same KPIs, same drawdown numbers.

Built and validated against TradingView's PineScript V6 strategy execution model.

## Project Structure

```
backtest_engine/
├── engine/                  ← business logic
│   ├── __init__.py          ← public API (import everything from here)
│   ├── engine.py            ← core backtest engine + indicator library
│   └── data.py              ← data loaders (TV CSV export + Bitstamp API)
├── data/                    ← chart data
│   └── INDEX_BTCUSD, 1D.csv
├── strategies/              ← strategy scripts
│   └── example_ema_cross.py
├── MEMORY.md                ← project memory for Claude AI sessions
├── README.md
├── requirements.txt
└── backtesting-notes.md
```

## Features

- **TradingView-matched execution**: signals on bar close, fills on next bar open (matching `calc_on_every_tick=false`, `fill_orders_on_standard_ohlc=true`)
- **Intrabar max drawdown**: uses bar Low for worst-case equity during open positions, matching TV's "Max equity drawdown (intrabar)" methodology
- **Indicator library**: EMA, SMMA/RMA, WMA, HMA, EHMA, THMA, Gaussian filter, crossover/crossunder detection, price source selector
- **Long + Short backtesting**: `run_backtest()` for long-only, `run_backtest_long_short()` for independent long+short positions (never simultaneous). Short PnL uses bar High for intrabar drawdown.
- **Strategy-agnostic**: write a signal function, pass a DataFrame with signal columns to `run_backtest()` or `run_backtest_long_short()`
- **Dual data sources**: TradingView CSV export (recommended for exact match) or Bitstamp API (free, no auth)

## Quick Start

```bash
pip install -r requirements.txt
python strategies/example_ema_cross.py
```

This runs an EMA 9/21 crossover strategy on the included INDEX:BTCUSD daily data and prints full KPIs.

## How to Add Your Own Strategy

1. Create a new file in `strategies/` with the sys.path setup and your signal function:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (
    load_tv_export,
    BacktestConfig, run_backtest, print_kpis,
    calc_ema, detect_crossover, detect_crossunder,
)

def my_strategy_signals(df, fast=9, slow=21):
    df = df.copy()
    df["fast_ema"] = calc_ema(df["Close"], fast)
    df["slow_ema"] = calc_ema(df["Close"], slow)
    df["long_entry"] = detect_crossover(df["fast_ema"], df["slow_ema"])
    df["long_exit"] = detect_crossunder(df["fast_ema"], df["slow_ema"])
    return df

def main():
    df = load_tv_export("INDEX_BTCUSD, 1D.csv")
    df = my_strategy_signals(df)

    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.1,
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        start_date="2018-01-01",
        end_date="2069-12-31",
    )

    kpis = run_backtest(df, config)
    print_kpis(kpis)

if __name__ == "__main__":
    main()
```

2. Run it: `python strategies/my_strategy.py`

## Available Indicators

All indicators match their TradingView PineScript equivalents:

| Function | PineScript Equivalent | Description |
|---|---|---|
| `calc_ema(series, length)` | `ta.ema()` | Exponential Moving Average |
| `calc_smma(series, length)` | `ta.rma()` | Smoothed MA / RMA / Wilder's |
| `calc_wma(series, length)` | `ta.wma()` | Weighted Moving Average |
| `calc_hma(series, length)` | — | Hull Moving Average |
| `calc_ehma(series, length)` | — | Exponential Hull MA |
| `calc_thma(series, length)` | — | Triple Hull MA |
| `calc_gaussian(series, length, poles)` | — | Gaussian filter (1-4 poles) |
| `detect_crossover(fast, slow)` | `ta.crossover()` | Cross above detection |
| `detect_crossunder(fast, slow)` | `ta.crossunder()` | Cross below detection |
| `get_source(df, source)` | `input.source()` | Price source selector (close/hl2/hlc3/ohlc4/etc.) |

All indicators handle NaN-leading input safely (can be chained/cascaded).

## Data Sources

**TradingView CSV export (recommended):**
Use this for an exact OHLC match with TradingView. On the TV chart, click the **Export chart data** button (small download icon in the bottom-right of the chart pane), save the CSV, and place it in the `data/` directory.

**Bitstamp API (fallback):**
Free, no authentication. Prices differ slightly from INDEX:BTCUSD but good for quick testing.

```python
from engine import fetch_btc_daily
df = fetch_btc_daily(start="2017-01-01", end="2026-12-31")
```

## TradingView Settings for Matching

To get identical results between this engine and TradingView, set these in your TV strategy properties:

| Setting | Value |
|---|---|
| Margin Long | **0%** |
| Margin Short | **0%** |
| Slippage | **0** |
| Commission | Match your `commission_pct` (e.g. 0.1%) |
| `calc_on_every_tick` | `false` |
| `fill_orders_on_standard_ohlc` | `true` |

**Important:** Setting margin to 100% (TV default) causes spurious margin-call mini-trades that inflate trade count.

## Slippage Note

Slippage is **not simulated** because it requires expensive tick-level / order-book data. Always set slippage to 0 in both the engine and TradingView for matching.

## Required DataFrame Columns

**`run_backtest()` (long-only):**

| Column | Type | Description |
|---|---|---|
| `Open` | float | Bar open price |
| `High` | float | Bar high price (used for intrabar drawdown) |
| `Low` | float | Bar low price (used for intrabar drawdown) |
| `Close` | float | Bar close price |
| `long_entry` | bool | True on bars where a long entry signal fires |
| `long_exit` | bool | True on bars where a long exit signal fires |

**`run_backtest_long_short()` — adds these columns:**

| Column | Type | Description |
|---|---|---|
| `short_entry` | bool | True on bars where a short entry signal fires |
| `short_exit` | bool | True on bars where a short exit signal fires |

Long and short positions are independent — a `long_exit` does NOT open a short, and vice versa. Positions are never held simultaneously.

## Requirements

- Python 3.10+
- pandas, numpy, requests (see `requirements.txt`)
