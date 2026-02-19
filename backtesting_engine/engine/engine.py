"""
Backtesting engine that matches TradingView's PineScript strategy behavior.

Replicates these TradingView settings:
- calc_on_every_tick = false   → signals on bar close, orders fill on next bar open
- fill_orders_on_standard_ohlc = true  → fills at the Open price
- margin_long = 0, margin_short = 0    → no margin calls
- commission_type = percent    → applied on both entry and exit

Usage:
    from engine import BacktestConfig, run_backtest
    from engine import calc_ema, detect_crossover, detect_crossunder, ema_cross_signals

    df = load_your_data()                        # DataFrame with Open, High, Low, Close
    df = ema_cross_signals(df, fast_len=9, slow_len=21)  # adds long_entry / long_exit columns
    kpis = run_backtest(df, BacktestConfig())    # run the backtest
    print_kpis(kpis)
"""

__version__ = "4.0.0"

import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Backtest settings matching TradingView's strategy() properties."""
    initial_capital: float = 1000.0
    commission_pct: float = 0.1       # e.g. 0.1 = 0.1%
    slippage_ticks: int = 0
    qty_type: str = "percent_of_equity"
    qty_value: float = 100.0          # 100 = 100% of equity
    pyramiding: int = 1
    start_date: str = "2018-01-01"
    end_date: str = "2069-12-31"


@dataclass
class Trade:
    """A single completed (or open) trade."""
    entry_date: pd.Timestamp
    entry_price: float
    entry_qty: float                  # asset units (e.g. BTC, fractional)
    direction: str = "long"           # "long" or "short"
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    entry_commission: float = 0.0
    exit_commission: float = 0.0


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """
    EMA matching TradingView's ``ta.ema()``.

    - Multiplier: ``2 / (length + 1)``
    - Seed: SMA of the first *length* **valid** values

    Handles NaN-leading input (e.g. when chaining indicators).
    """
    multiplier = 2.0 / (length + 1)
    ema = pd.Series(np.nan, index=series.index, dtype=float)
    vals = series.values

    # Find the first index with `length` consecutive non-NaN values for the seed
    valid = ~np.isnan(vals)
    start = -1
    count = 0
    for i in range(len(vals)):
        if valid[i]:
            count += 1
            if count == length:
                start = i - length + 1
                break
        else:
            count = 0

    if start < 0:
        return ema  # not enough data

    seed_idx = start + length - 1
    ema.iloc[seed_idx] = np.mean(vals[start:start + length])
    for i in range(seed_idx + 1, len(vals)):
        if np.isnan(vals[i]):
            break
        ema.iloc[i] = vals[i] * multiplier + ema.iloc[i - 1] * (1 - multiplier)

    return ema


def detect_crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on bars where *fast* crosses **above** *slow*."""
    return (fast.shift(1) <= slow.shift(1)) & (fast > slow)


def detect_crossunder(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on bars where *fast* crosses **below** *slow*."""
    return (fast.shift(1) >= slow.shift(1)) & (fast < slow)


def calc_smma(series: pd.Series, length: int) -> pd.Series:
    """
    Smoothed Moving Average matching TradingView's ``ta.rma()``.

    Also known as RMA / Wilder's smoothing.
    Formula: ``smma[i] = (smma[i-1] * (length - 1) + src[i]) / length``
    Seed: SMA of the first *length* **valid** values.
    """
    smma = pd.Series(np.nan, index=series.index)
    vals = series.values

    # Find seed from first `length` consecutive non-NaN values
    valid = ~np.isnan(vals)
    start = -1
    count = 0
    for i in range(len(vals)):
        if valid[i]:
            count += 1
            if count == length:
                start = i - length + 1
                break
        else:
            count = 0

    if start < 0:
        return smma

    seed_idx = start + length - 1
    smma.iloc[seed_idx] = np.mean(vals[start:start + length])
    for i in range(seed_idx + 1, len(vals)):
        if np.isnan(vals[i]):
            break
        smma.iloc[i] = (smma.iloc[i - 1] * (length - 1) + vals[i]) / length

    return smma


def calc_wma(series: pd.Series, length: int) -> pd.Series:
    """
    Weighted Moving Average matching TradingView's ``ta.wma()``.

    Weights increase linearly: ``[1, 2, 3, ..., length]``.
    """
    weights = np.arange(1, length + 1, dtype=float)
    weight_sum = weights.sum()

    def _weighted_avg(window):
        return np.dot(window, weights) / weight_sum

    return series.rolling(window=length, min_periods=length).apply(_weighted_avg, raw=True)


def calc_hma(series: pd.Series, length: int) -> pd.Series:
    """
    Hull Moving Average (HMA) by Alan Hull.

    ``HMA = WMA( 2·WMA(n/2) − WMA(n) , √n )``
    """
    half = length // 2
    sqrt = round(math.sqrt(length))
    diff = 2 * calc_wma(series, half) - calc_wma(series, length)
    return calc_wma(diff, sqrt)


def calc_ehma(series: pd.Series, length: int) -> pd.Series:
    """
    Exponential Hull Moving Average (EHMA).

    Same structure as HMA but uses EMA instead of WMA.
    """
    half = length // 2
    sqrt = round(math.sqrt(length))
    diff = 2 * calc_ema(series, half) - calc_ema(series, length)
    return calc_ema(diff, sqrt)


def calc_thma(series: pd.Series, length: int) -> pd.Series:
    """
    Triple Hull Moving Average (THMA).

    ``THMA = WMA( 3·WMA(n/3) − WMA(n/2) − WMA(n) , n )``
    """
    len3 = length // 3
    half = length // 2
    inner = 3 * calc_wma(series, len3) - calc_wma(series, half) - calc_wma(series, length)
    return calc_wma(inner, length)


def calc_gaussian(series: pd.Series, length: int, poles: int = 1) -> pd.Series:
    """
    Gaussian filter approximated by cascading EMAs.

    *poles* (1–4) controls smoothness: more poles → smoother curve.
    """
    poles = max(1, min(poles, 4))
    result = calc_ema(series, length)
    for _ in range(poles - 1):
        result = calc_ema(result, length)
    return result


# ---------------------------------------------------------------------------
# Source selector (matches PineScript input.source())
# ---------------------------------------------------------------------------

def get_source(df: pd.DataFrame, source: str = "Close") -> pd.Series:
    """
    Return a price series from *df* matching a PineScript source string.

    Accepted values (case-insensitive):
        ``close``, ``open``, ``high``, ``low``,
        ``hl2``, ``hlc3``, ``ohlc4``
    """
    key = source.strip().lower()
    if key == "close":
        return df["Close"]
    if key == "open":
        return df["Open"]
    if key == "high":
        return df["High"]
    if key == "low":
        return df["Low"]
    if key == "hl2":
        return (df["High"] + df["Low"]) / 2
    if key == "hlc3":
        return (df["High"] + df["Low"] + df["Close"]) / 3
    if key == "ohlc4":
        return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    raise ValueError(
        f"Unknown source '{source}'. "
        "Use: close, open, high, low, hl2, hlc3, ohlc4"
    )


# ---------------------------------------------------------------------------
# Strategy signal generators (add more here for new strategies)
# ---------------------------------------------------------------------------

def ema_cross_signals(df: pd.DataFrame, fast_len: int = 9, slow_len: int = 21) -> pd.DataFrame:
    """
    Add EMA-crossover entry/exit signals to *df* (in-place + returned).

    Columns added:
        fast_ema, slow_ema, long_entry, long_exit
    """
    df = df.copy()
    df["fast_ema"] = calc_ema(df["Close"], fast_len)
    df["slow_ema"] = calc_ema(df["Close"], slow_len)
    df["long_entry"] = detect_crossover(df["fast_ema"], df["slow_ema"])
    df["long_exit"] = detect_crossunder(df["fast_ema"], df["slow_ema"])
    return df


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> dict:
    """
    Run a long-only backtest matching TradingView behaviour.

    **Required columns** in *df*:
        ``Open``, ``High``, ``Low``, ``Close``,
        ``long_entry`` (bool), ``long_exit`` (bool)

    The DataFrame should include warmup bars **before** ``config.start_date``
    so that indicator values are accurate when the trading window begins.

    Returns a dict of KPIs (see ``compute_kpis``) including a ``trades`` list.
    """
    # --- input validation ---------------------------------------------------
    required = {"Open", "High", "Low", "Close", "long_entry", "long_exit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            "Add signal columns (e.g. via ema_cross_signals()) before calling run_backtest()."
        )

    df = df.copy()
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    # --- start-date safeguard ------------------------------------------------
    # If the data doesn't go back far enough, adjust start_date to the first
    # available bar.  This prevents mismatched KPIs when the user's CSV export
    # begins later than expected (TV always trades from the first bar).
    data_first = df.index[0]
    if data_first > start:
        print(f"\n  ⚠  DATA STARTS AFTER start_date!")
        print(f"     start_date was:  {config.start_date}")
        print(f"     data starts at:  {data_first.date()}")
        print(f"     → Adjusted start_date to {data_first.date()}")
        print(f"     To match TradingView, set the same start date in your")
        print(f"     TV strategy properties (Date Range → Start Date).\n")
        start = data_first

    # --- state ---------------------------------------------------------------
    equity = config.initial_capital
    cash = config.initial_capital
    position_qty = 0.0
    position_entry_price = 0.0
    trades: list[Trade] = []
    current_trade: Optional[Trade] = None

    pending_entry = False
    pending_exit = False

    equity_curve: list[dict] = []
    commission_rate = config.commission_pct / 100.0

    # --- intrabar drawdown tracking (TV methodology) -------------------------
    # Peak equity only updates when flat (no open position).
    # TV defines "max_equity" as peak from initial capital + closed trades.
    # Intrabar trough uses bar["Low"] for open long positions.
    # Max DD ($) and max DD (%) are tracked independently — they may occur
    # at different points in time, matching TV's reporting.
    peak_equity = config.initial_capital
    max_intrabar_dd = 0.0       # worst absolute drawdown (negative or zero)
    max_intrabar_dd_pct = 0.0   # worst percentage drawdown (negative or zero)

    # --- bar-by-bar loop (matches TV execution order) ------------------------
    for i in range(len(df)):
        bar = df.iloc[i]
        bar_date = df.index[i]
        bar_in_range = start <= bar_date <= end

        # 1) FILL pending orders at this bar's Open
        if pending_entry and position_qty == 0:
            fill_price = bar["Open"]
            # TV sizes so trade_value + commission = equity
            trade_value = equity / (1 + commission_rate)
            qty = trade_value / fill_price
            entry_commission = trade_value * commission_rate

            position_qty = qty
            position_entry_price = fill_price
            cash = equity - trade_value - entry_commission

            current_trade = Trade(
                entry_date=bar_date,
                entry_price=fill_price,
                entry_qty=qty,
                direction="long",
                entry_commission=entry_commission,
            )
            pending_entry = False

        if pending_exit and position_qty > 0:
            fill_price = bar["Open"]
            trade_value = position_qty * fill_price
            exit_commission = trade_value * commission_rate

            gross_pnl = position_qty * (fill_price - position_entry_price)
            net_pnl = gross_pnl - current_trade.entry_commission - exit_commission

            cash += trade_value - exit_commission
            equity = cash

            current_trade.exit_date = bar_date
            current_trade.exit_price = fill_price
            current_trade.pnl = net_pnl
            entry_value = current_trade.entry_qty * current_trade.entry_price
            current_trade.pnl_pct = (net_pnl / entry_value) * 100
            current_trade.exit_commission = exit_commission
            trades.append(current_trade)
            current_trade = None

            position_qty = 0.0
            position_entry_price = 0.0
            pending_exit = False

        # 2a) Intrabar drawdown check (only while holding a long position)
        if bar_in_range and position_qty > 0:
            equity_at_low = cash + position_qty * bar["Low"]
            dd = equity_at_low - peak_equity
            dd_pct = (dd / peak_equity) * 100 if peak_equity != 0 else 0.0
            if dd < max_intrabar_dd:
                max_intrabar_dd = dd
            if dd_pct < max_intrabar_dd_pct:
                max_intrabar_dd_pct = dd_pct

        # 2b) Mark-to-market equity at Close
        if position_qty > 0:
            equity = cash + position_qty * bar["Close"]
        else:
            equity = cash

        if bar_in_range:
            equity_curve.append({"date": bar_date, "equity": equity})

        # 2c) Update peak equity — ONLY when flat (no open position).
        # TV's "max_equity" = peak from initial capital + all closed trades.
        # Unrealised mark-to-market equity does NOT update the peak.
        if bar_in_range and position_qty == 0 and equity > peak_equity:
            peak_equity = equity

        # 3) Detect signals at Close (only inside trading window)
        pending_entry = False
        pending_exit = False

        if bar_in_range:
            if bar["long_entry"] and position_qty == 0:
                pending_entry = True
            if bar["long_exit"] and position_qty > 0:
                pending_exit = True

    # Record any open position at end of data
    if current_trade is not None:
        trades.append(current_trade)

    equity_df = pd.DataFrame(equity_curve)
    kpis = compute_kpis(trades, equity_df, config,
                        max_intrabar_dd, max_intrabar_dd_pct)
    # Include the actual start/end dates used (may differ from config if adjusted)
    kpis["actual_start_date"] = str(start.date())
    kpis["actual_end_date"] = str(end.date())
    return kpis


def run_backtest_long_short(df: pd.DataFrame, config: BacktestConfig) -> dict:
    """
    Run a long+short backtest matching TradingView behaviour.

    Supports separate long and short positions (never simultaneously).
    Long and short signals are completely independent — a long_exit does
    NOT open a short, and vice versa.

    **Required columns** in *df*:
        ``Open``, ``High``, ``Low``, ``Close``,
        ``long_entry`` (bool), ``long_exit`` (bool),
        ``short_entry`` (bool), ``short_exit`` (bool)

    TradingView matching notes:
    - Short PnL: qty * (entry_price - exit_price) - commissions
    - Intrabar DD for shorts uses bar High (worst case for short = price spike up)
    - Signals on bar close, fill on next bar open (calc_on_every_tick=false)
    - margin_long = 0, margin_short = 0

    Returns a dict of KPIs (see ``compute_kpis``) including a ``trades`` list.
    """
    # --- input validation ---------------------------------------------------
    required = {"Open", "High", "Low", "Close",
                "long_entry", "long_exit", "short_entry", "short_exit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            "Add all signal columns before calling run_backtest_long_short()."
        )

    df = df.copy()
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    # --- start-date safeguard ------------------------------------------------
    data_first = df.index[0]
    if data_first > start:
        print(f"\n  ⚠  DATA STARTS AFTER start_date!")
        print(f"     start_date was:  {config.start_date}")
        print(f"     data starts at:  {data_first.date()}")
        print(f"     → Adjusted start_date to {data_first.date()}")
        print(f"     To match TradingView, set the same start date in your")
        print(f"     TV strategy properties (Date Range → Start Date).\n")
        start = data_first

    # --- state ---------------------------------------------------------------
    equity = config.initial_capital
    cash = config.initial_capital
    position_qty = 0.0          # positive = long, negative = short
    position_entry_price = 0.0
    position_side = ""          # "long" or "short" or ""
    trades: list[Trade] = []
    current_trade: Optional[Trade] = None

    pending_long_entry = False
    pending_long_exit = False
    pending_short_entry = False
    pending_short_exit = False

    equity_curve: list[dict] = []
    commission_rate = config.commission_pct / 100.0

    # --- intrabar drawdown tracking ------------------------------------------
    peak_equity = config.initial_capital
    max_intrabar_dd = 0.0
    max_intrabar_dd_pct = 0.0

    # --- bar-by-bar loop -----------------------------------------------------
    for i in range(len(df)):
        bar = df.iloc[i]
        bar_date = df.index[i]
        bar_in_range = start <= bar_date <= end

        # 1) FILL pending orders at this bar's Open

        # --- Close existing positions first ---
        if pending_long_exit and position_side == "long" and position_qty > 0:
            fill_price = bar["Open"]
            trade_value = position_qty * fill_price
            exit_commission = trade_value * commission_rate
            gross_pnl = position_qty * (fill_price - position_entry_price)
            net_pnl = gross_pnl - current_trade.entry_commission - exit_commission
            cash += trade_value - exit_commission
            equity = cash

            current_trade.exit_date = bar_date
            current_trade.exit_price = fill_price
            current_trade.pnl = net_pnl
            entry_value = current_trade.entry_qty * current_trade.entry_price
            current_trade.pnl_pct = (net_pnl / entry_value) * 100
            current_trade.exit_commission = exit_commission
            trades.append(current_trade)
            current_trade = None
            position_qty = 0.0
            position_entry_price = 0.0
            position_side = ""
            pending_long_exit = False

        if pending_short_exit and position_side == "short" and position_qty < 0:
            fill_price = bar["Open"]
            abs_qty = abs(position_qty)
            trade_value = abs_qty * fill_price
            exit_commission = trade_value * commission_rate
            # Short PnL: profit when price goes down
            gross_pnl = abs_qty * (position_entry_price - fill_price)
            net_pnl = gross_pnl - current_trade.entry_commission - exit_commission
            # Cash: settle short from actual cash (not mark-to-market equity).
            # At short entry cash = equity - entry_commission (collateral).
            # At exit: return collateral + gross_pnl - exit_commission.
            cash = cash + gross_pnl - exit_commission
            equity = cash

            current_trade.exit_date = bar_date
            current_trade.exit_price = fill_price
            current_trade.pnl = net_pnl
            entry_value = abs_qty * current_trade.entry_price
            current_trade.pnl_pct = (net_pnl / entry_value) * 100
            current_trade.exit_commission = exit_commission
            trades.append(current_trade)
            current_trade = None
            position_qty = 0.0
            position_entry_price = 0.0
            position_side = ""
            pending_short_exit = False

        # --- Open new positions ---
        # TV sizes positions so that trade_value + entry_commission = equity,
        # i.e. trade_value = equity / (1 + commission_rate).  This ensures
        # the total outlay never exceeds available equity.
        if pending_long_entry and position_qty == 0:
            fill_price = bar["Open"]
            trade_value = equity / (1 + commission_rate)
            qty = trade_value / fill_price
            entry_commission = trade_value * commission_rate
            position_qty = qty
            position_entry_price = fill_price
            position_side = "long"
            cash = equity - trade_value - entry_commission
            current_trade = Trade(
                entry_date=bar_date, entry_price=fill_price,
                entry_qty=qty, direction="long",
                entry_commission=entry_commission,
            )
            pending_long_entry = False

        if pending_short_entry and position_qty == 0:
            fill_price = bar["Open"]
            trade_value = equity / (1 + commission_rate)
            abs_qty = trade_value / fill_price
            entry_commission = trade_value * commission_rate
            position_qty = -abs_qty  # negative = short
            position_entry_price = fill_price
            position_side = "short"
            # Cash model: we hold equity as collateral, commission deducted
            cash = equity - entry_commission
            current_trade = Trade(
                entry_date=bar_date, entry_price=fill_price,
                entry_qty=abs_qty, direction="short",
                entry_commission=entry_commission,
            )
            pending_short_entry = False

        # 2a) Intrabar drawdown check
        if bar_in_range and position_qty != 0:
            if position_side == "long":
                # Worst case for long: price drops to bar Low
                equity_at_worst = cash + position_qty * bar["Low"]
            else:
                # Worst case for short: price spikes to bar High
                abs_qty = abs(position_qty)
                unrealised_pnl = abs_qty * (position_entry_price - bar["High"])
                equity_at_worst = cash + unrealised_pnl

            dd = equity_at_worst - peak_equity
            dd_pct = (dd / peak_equity) * 100 if peak_equity != 0 else 0.0
            if dd < max_intrabar_dd:
                max_intrabar_dd = dd
            if dd_pct < max_intrabar_dd_pct:
                max_intrabar_dd_pct = dd_pct

        # 2b) Mark-to-market equity at Close
        if position_side == "long" and position_qty > 0:
            equity = cash + position_qty * bar["Close"]
        elif position_side == "short" and position_qty < 0:
            abs_qty = abs(position_qty)
            unrealised_pnl = abs_qty * (position_entry_price - bar["Close"])
            equity = cash + unrealised_pnl
        else:
            equity = cash

        if bar_in_range:
            equity_curve.append({"date": bar_date, "equity": equity})

        # 2c) Update peak equity — ONLY when flat
        if bar_in_range and position_qty == 0 and equity > peak_equity:
            peak_equity = equity

        # 3) Detect signals at Close (only inside trading window)
        pending_long_entry = False
        pending_long_exit = False
        pending_short_entry = False
        pending_short_exit = False

        if bar_in_range:
            # Long signals
            if bar["long_entry"] and position_qty == 0:
                pending_long_entry = True
            if bar["long_exit"] and position_side == "long" and position_qty > 0:
                pending_long_exit = True
            # Short signals (independent from long)
            if bar["short_entry"] and position_qty == 0:
                pending_short_entry = True
            if bar["short_exit"] and position_side == "short" and position_qty < 0:
                pending_short_exit = True

            # Reversal: if exiting one side and entering the other on the
            # same bar, allow both to queue.  The exit fills first on the
            # next bar's Open (setting position_qty=0), then the entry fills
            # immediately after — matching TV's strategy.entry() reversal.
            if pending_long_exit and bar["short_entry"]:
                pending_short_entry = True
            if pending_short_exit and bar["long_entry"]:
                pending_long_entry = True

    # Record any open position at end of data
    if current_trade is not None:
        trades.append(current_trade)

    equity_df = pd.DataFrame(equity_curve)
    kpis = compute_kpis(trades, equity_df, config,
                        max_intrabar_dd, max_intrabar_dd_pct)
    kpis["actual_start_date"] = str(start.date())
    kpis["actual_end_date"] = str(end.date())
    return kpis


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(
    trades: list[Trade],
    equity_df: pd.DataFrame,
    config: BacktestConfig,
    max_intrabar_dd: float = 0.0,
    max_intrabar_dd_pct: float = 0.0,
) -> dict:
    """Compute performance KPIs matching TradingView's Strategy Tester.

    Max drawdown uses TradingView's intrabar methodology:
    - Uses bar Low prices for worst-case equity during open long positions
    - Peak equity only updates when flat (from closed-trade equity)
    - Max DD ($) and max DD (%) are independent maximums
    """

    initial_capital = config.initial_capital

    if not trades:
        return {"error": "No trades executed"}

    final_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_capital

    # Separate closed vs open trades
    closed_trades = [t for t in trades if t.exit_date is not None]
    open_trades = [t for t in trades if t.exit_date is None]

    # Profit / loss — net_profit based on CLOSED trades only (matches TV)
    winning_trades = [t for t in closed_trades if t.pnl > 0]
    losing_trades = [t for t in closed_trades if t.pnl <= 0]

    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = sum(t.pnl for t in losing_trades)
    net_profit = gross_profit + gross_loss
    net_profit_pct = (net_profit / initial_capital) * 100

    # Open P&L = equity beyond what closed trades account for
    open_profit = (final_equity - initial_capital) - net_profit

    # Total P&L = closed + open (matches TV Overview "Total P&L")
    total_pnl = net_profit + open_profit
    total_pnl_pct = (total_pnl / initial_capital) * 100

    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    # Drawdown — intrabar methodology matching TradingView
    max_drawdown = max_intrabar_dd
    max_drawdown_pct = max_intrabar_dd_pct

    # Trade statistics (closed trades only — matching TV)
    total_trades = len(closed_trades)
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    win_rate = (num_winning / total_trades) * 100 if total_trades > 0 else 0

    avg_trade = net_profit / total_trades if total_trades > 0 else 0
    avg_trade_pct = sum(t.pnl_pct for t in closed_trades) / total_trades if total_trades > 0 else 0
    avg_winning = gross_profit / num_winning if num_winning > 0 else 0
    avg_losing = gross_loss / num_losing if num_losing > 0 else 0
    avg_win_loss_ratio = abs(avg_winning / avg_losing) if avg_losing != 0 else float("inf")

    largest_winning = max((t.pnl for t in winning_trades), default=0)
    largest_losing = min((t.pnl for t in losing_trades), default=0)

    # Consecutive wins / losses
    max_consec_wins = max_consec_losses = 0
    cur_w = cur_l = 0
    for t in closed_trades:
        if t.pnl > 0:
            cur_w += 1; cur_l = 0
            max_consec_wins = max(max_consec_wins, cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_consec_losses = max(max_consec_losses, cur_l)

    total_commission = sum(t.entry_commission + t.exit_commission for t in closed_trades)

    # Risk-adjusted ratios (annualised, 365-day year for crypto)
    sharpe = sortino = 0.0
    if len(equity_df) > 1:
        daily_returns = equity_df["equity"].pct_change().dropna()
        std = daily_returns.std()
        if std != 0:
            sharpe = (daily_returns.mean() / std) * np.sqrt(365)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() != 0:
            sortino = (daily_returns.mean() / downside.std()) * np.sqrt(365)

    return {
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "net_profit": net_profit,
        "net_profit_pct": net_profit_pct,
        "open_profit": open_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "total_trades": total_trades,
        "num_winning": num_winning,
        "num_losing": num_losing,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "avg_trade_pct": avg_trade_pct,
        "avg_winning": avg_winning,
        "avg_losing": avg_losing,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "largest_winning": largest_winning,
        "largest_losing": largest_losing,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "total_commission": total_commission,
        "final_equity": final_equity,
        "initial_capital": initial_capital,
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_kpis(kpis: dict):
    """Print KPIs in a format similar to TradingView's Strategy Tester.

    Display order matches what users see in TV:
    - Total P&L (incl. open) shown first — matches TV Overview headline
    - Net Profit (closed only) shown second — matches TV Excel export
    - % always shown alongside $ for Net Profit and Max Drawdown
    """
    print("=" * 60)
    print("  STRATEGY PERFORMANCE SUMMARY")
    print("=" * 60)
    print()
    has_open = abs(kpis.get('open_profit', 0)) > 0.005
    if has_open:
        print(f"  Total P&L (incl. open): ${kpis['total_pnl']:>10,.2f}  ({kpis['total_pnl_pct']:>8.2f}%)")
    print(f"  Net Profit (closed):    ${kpis['net_profit']:>10,.2f}  ({kpis['net_profit_pct']:>8.2f}%)")
    if has_open:
        open_pct = (kpis['open_profit'] / kpis['initial_capital']) * 100
        print(f"  Open P&L:               ${kpis['open_profit']:>10,.2f}  ({open_pct:>8.2f}%)")
    print(f"  Gross Profit:           ${kpis['gross_profit']:>10,.2f}")
    print(f"  Gross Loss:             ${kpis['gross_loss']:>10,.2f}")
    print()
    print(f"  Profit Factor:         {kpis['profit_factor']:>12.3f}")
    print(f"  Max Drawdown:         ${kpis['max_drawdown']:>12,.2f}  ({kpis['max_drawdown_pct']:>8.2f}%)")
    print(f"  Sharpe Ratio:          {kpis['sharpe_ratio']:>12.3f}")
    print(f"  Sortino Ratio:         {kpis['sortino_ratio']:>12.3f}")
    print()
    print(f"  Total Trades:          {kpis['total_trades']:>12d}")
    print(f"  Winning Trades:        {kpis['num_winning']:>12d}  ({kpis['win_rate']:>6.2f}%)")
    print(f"  Losing Trades:         {kpis['num_losing']:>12d}")
    print()
    print(f"  Avg Trade:            ${kpis['avg_trade']:>12,.2f}  ({kpis['avg_trade_pct']:>8.2f}%)")
    print(f"  Avg Winning Trade:    ${kpis['avg_winning']:>12,.2f}")
    print(f"  Avg Losing Trade:     ${kpis['avg_losing']:>12,.2f}")
    print(f"  Avg Win/Loss Ratio:    {kpis['avg_win_loss_ratio']:>12.3f}")
    print()
    print(f"  Largest Win:          ${kpis['largest_winning']:>12,.2f}")
    print(f"  Largest Loss:         ${kpis['largest_losing']:>12,.2f}")
    print()
    print(f"  Max Consec. Wins:      {kpis['max_consec_wins']:>12d}")
    print(f"  Max Consec. Losses:    {kpis['max_consec_losses']:>12d}")
    print()
    print(f"  Total Commission:     ${kpis['total_commission']:>12,.2f}")
    print(f"  Initial Capital:      ${kpis['initial_capital']:>12,.2f}")
    print(f"  Final Equity:         ${kpis['final_equity']:>12,.2f}")
    print("=" * 60)


def print_trades(trades: list[Trade], max_trades: int = 0):
    """Print trade list.

    Automatically shows a 'Dir' column when short trades are present.
    """
    print()
    has_shorts = any(t.direction == "short" for t in trades)

    if has_shorts:
        header = (f"  {'#':>3}  {'Dir':>5}  {'Entry Date':>12}  {'Entry $':>10}  {'Exit Date':>12}  "
                  f"{'Exit $':>10}  {'Qty':>12}  {'PnL $':>12}  {'PnL %':>8}")
        print(header)
        print("  " + "-" * 103)
    else:
        header = (f"  {'#':>3}  {'Entry Date':>12}  {'Entry $':>10}  {'Exit Date':>12}  "
                  f"{'Exit $':>10}  {'Qty':>12}  {'PnL $':>12}  {'PnL %':>8}")
        print(header)
        print("  " + "-" * 95)

    display = trades if max_trades == 0 else trades[:max_trades]
    for i, t in enumerate(display, 1):
        exit_date = t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "OPEN"
        exit_price = f"{t.exit_price:>10,.2f}" if t.exit_price else "      OPEN"
        pnl = f"{t.pnl:>12,.2f}" if t.pnl is not None else "        N/A"
        pnl_pct = f"{t.pnl_pct:>8.2f}" if t.pnl_pct is not None else "     N/A"
        direction = t.direction.upper()[:5]

        if has_shorts:
            print(f"  {i:>3}  {direction:>5}  {t.entry_date.strftime('%Y-%m-%d'):>12}  {t.entry_price:>10,.2f}  "
                  f"{exit_date:>12}  {exit_price}  {t.entry_qty:>12.6f}  {pnl}  {pnl_pct}")
        else:
            print(f"  {i:>3}  {t.entry_date.strftime('%Y-%m-%d'):>12}  {t.entry_price:>10,.2f}  "
                  f"{exit_date:>12}  {exit_price}  {t.entry_qty:>12.6f}  {pnl}  {pnl_pct}")

    if max_trades and len(trades) > max_trades:
        print(f"  ... ({len(trades) - max_trades} more trades)")
