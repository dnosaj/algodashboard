"""
Smart Money + RSI Scalper V4 — Selective entries, real TV data.

Problem with V3: Too many signals (100-400+), low win rate (~30%).
User sees ~5 high-quality trades per day visually.

New approaches:
A) "Strong SM + RSI flip" — require SM to be strongly positive/negative (>0.3, >0.4, >0.5)
   AND RSI flip. This dramatically reduces signal count.

B) "RSI reversal from extreme" — RSI must first go to an extreme (<30 or >70), then
   flip to blue/purple. This catches the "explode" part of "Chop and Explode".

C) "SM trend + RSI momentum" — SM must be trending (increasing for green, decreasing
   for red), not just positive/negative. Catches fresh momentum.

D) "First signal of SM regime" — Only take the FIRST RSI flip after SM changes from
   red to green (or green to red). Avoids repetitive signals in same SM regime.

E) "SM high interest zone" — SM must be above threshold (0.85 user setting matches
   the "High Interest Threshold" in the indicator). Combined with RSI flip.

F) "Cooldown" — After a trade, wait N bars before taking another signal.
   Prevents rapid-fire entries.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine.engine import Trade, compute_kpis, BacktestConfig
from engine import print_kpis, print_trades
from typing import Optional


# ---------------------------------------------------------------------------
# Load the indicator-enriched CSV
# ---------------------------------------------------------------------------

def load_indicator_csv(filename: str) -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    filepath = data_dir / filename
    df = pd.read_csv(filepath)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time",
        cols[1]: "Open",
        cols[2]: "High",
        cols[3]: "Low",
        cols[4]: "Close",
        cols[5]: "RSI",
        cols[16]: "SM_Buy",
        cols[22]: "SM_Sell",
    })
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    sm_buy = pd.to_numeric(df["SM_Buy"], errors="coerce").fillna(0.0)
    sm_sell = pd.to_numeric(df["SM_Sell"], errors="coerce").fillna(0.0)
    df["net_index"] = sm_buy + sm_sell
    df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")
    df = df.iloc[:-1]
    return df[["Open", "High", "Low", "Close", "RSI", "net_index"]]


# ---------------------------------------------------------------------------
# Scalper engine (same as V3)
# ---------------------------------------------------------------------------

def run_scalper(df: pd.DataFrame, config: BacktestConfig,
                tp_points: float, sl_points: float,
                max_hold_bars: int = 0,
                exit_on_sm_flip: bool = False) -> dict:
    df = df.copy()
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    data_first = df.index[0]
    if data_first > start:
        start = data_first

    equity = config.initial_capital
    cash = config.initial_capital
    position_qty = 0.0
    position_entry_price = 0.0
    position_side = ""
    entry_bar_idx = 0
    trades: list[Trade] = []
    current_trade: Optional[Trade] = None

    pending_long = False
    pending_short = False

    equity_curve: list[dict] = []
    commission_rate = config.commission_pct / 100.0

    peak_equity = config.initial_capital
    max_intrabar_dd = 0.0
    max_intrabar_dd_pct = 0.0

    sm_values = df["net_index"].values

    def close_position(bar, bar_date, exit_price):
        nonlocal equity, cash, position_qty, position_entry_price, position_side
        nonlocal current_trade, peak_equity

        if position_side == "long":
            trade_value = position_qty * exit_price
            exit_commission = trade_value * commission_rate
            gross_pnl = position_qty * (exit_price - position_entry_price)
            net_pnl = gross_pnl - current_trade.entry_commission - exit_commission
            cash += trade_value - exit_commission
            equity = cash
        else:
            abs_qty = abs(position_qty)
            trade_value = abs_qty * exit_price
            exit_commission = trade_value * commission_rate
            gross_pnl = abs_qty * (position_entry_price - exit_price)
            net_pnl = gross_pnl - current_trade.entry_commission - exit_commission
            cash = cash + gross_pnl - exit_commission
            equity = cash

        current_trade.exit_date = bar_date
        current_trade.exit_price = exit_price
        current_trade.pnl = net_pnl
        entry_value = current_trade.entry_qty * current_trade.entry_price
        current_trade.pnl_pct = (net_pnl / entry_value) * 100
        current_trade.exit_commission = exit_commission
        trades.append(current_trade)
        current_trade = None
        position_qty = 0.0
        position_entry_price = 0.0
        position_side = ""

    for i in range(len(df)):
        bar = df.iloc[i]
        bar_date = df.index[i]
        bar_in_range = start <= bar_date <= end

        if pending_long and position_qty == 0:
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
            entry_bar_idx = i
            pending_long = False

        if pending_short and position_qty == 0:
            fill_price = bar["Open"]
            trade_value = equity / (1 + commission_rate)
            abs_qty = trade_value / fill_price
            entry_commission = trade_value * commission_rate
            position_qty = -abs_qty
            position_entry_price = fill_price
            position_side = "short"
            cash = equity - entry_commission
            current_trade = Trade(
                entry_date=bar_date, entry_price=fill_price,
                entry_qty=abs_qty, direction="short",
                entry_commission=entry_commission,
            )
            entry_bar_idx = i
            pending_short = False

        if position_qty != 0:
            bars_held = i - entry_bar_idx
            exited = False

            if position_side == "long":
                tp_price = position_entry_price + tp_points
                sl_price = position_entry_price - sl_points
                if bar["Low"] <= sl_price:
                    close_position(bar, bar_date, sl_price)
                    exited = True
                elif bar["High"] >= tp_price:
                    close_position(bar, bar_date, tp_price)
                    exited = True
            else:
                tp_price = position_entry_price - tp_points
                sl_price = position_entry_price + sl_points
                if bar["High"] >= sl_price:
                    close_position(bar, bar_date, sl_price)
                    exited = True
                elif bar["Low"] <= tp_price:
                    close_position(bar, bar_date, tp_price)
                    exited = True

            if not exited and exit_on_sm_flip:
                sm_now = sm_values[i]
                if position_side == "long" and sm_now < 0:
                    close_position(bar, bar_date, bar["Close"])
                    exited = True
                elif position_side == "short" and sm_now > 0:
                    close_position(bar, bar_date, bar["Close"])
                    exited = True

            if not exited and max_hold_bars > 0 and bars_held >= max_hold_bars:
                close_position(bar, bar_date, bar["Close"])
                exited = True

        if bar_in_range and position_qty != 0:
            if position_side == "long":
                equity_at_worst = cash + position_qty * bar["Low"]
            else:
                abs_qty = abs(position_qty)
                unrealised_pnl = abs_qty * (position_entry_price - bar["High"])
                equity_at_worst = cash + unrealised_pnl
            dd = equity_at_worst - peak_equity
            dd_pct = (dd / peak_equity) * 100 if peak_equity != 0 else 0.0
            if dd < max_intrabar_dd:
                max_intrabar_dd = dd
            if dd_pct < max_intrabar_dd_pct:
                max_intrabar_dd_pct = dd_pct

        if position_side == "long" and position_qty > 0:
            equity = cash + position_qty * bar["Close"]
        elif position_side == "short" and position_qty < 0:
            abs_qty = abs(position_qty)
            equity = cash + abs_qty * (position_entry_price - bar["Close"])
        else:
            equity = cash

        if bar_in_range:
            equity_curve.append({"date": bar_date, "equity": equity})
        if bar_in_range and position_qty == 0 and equity > peak_equity:
            peak_equity = equity

        pending_long = False
        pending_short = False
        if bar_in_range and position_qty == 0:
            if bar["long_entry"]:
                pending_long = True
            elif bar["short_entry"]:
                pending_short = True

    if current_trade is not None:
        trades.append(current_trade)

    equity_df = pd.DataFrame(equity_curve)
    kpis = compute_kpis(trades, equity_df, config,
                        max_intrabar_dd, max_intrabar_dd_pct)
    kpis["actual_start_date"] = str(start.date())
    kpis["actual_end_date"] = str(end.date())
    return kpis


# ---------------------------------------------------------------------------
# Signal generators — multiple creative approaches
# ---------------------------------------------------------------------------

def signals_basic(df, rsi_buy=60, rsi_sell=40, sm_thr=0.0):
    """Standard: SM above threshold + RSI crossover."""
    df = df.copy()
    rsi = df["RSI"]
    sm = df["net_index"]
    df["long_entry"] = (sm > sm_thr) & (rsi > rsi_buy) & (rsi.shift(1) <= rsi_buy)
    df["short_entry"] = (sm < -sm_thr) & (rsi < rsi_sell) & (rsi.shift(1) >= rsi_sell)
    return df


def signals_extreme_reversal(df, rsi_buy=60, rsi_sell=40, sm_thr=0.0,
                              extreme_lo=30, extreme_hi=70, lookback=10):
    """
    RSI must have been at an extreme (below extreme_lo or above extreme_hi)
    within the last `lookback` bars before flipping to blue/purple.
    This catches the "explode" from a chop zone.
    """
    df = df.copy()
    rsi = df["RSI"]
    sm = df["net_index"]

    rsi_flips_blue = (rsi > rsi_buy) & (rsi.shift(1) <= rsi_buy)
    rsi_flips_purple = (rsi < rsi_sell) & (rsi.shift(1) >= rsi_sell)

    # Was RSI recently at an extreme?
    rsi_was_low = rsi.rolling(lookback).min() < extreme_lo
    rsi_was_high = rsi.rolling(lookback).max() > extreme_hi

    df["long_entry"] = (sm > sm_thr) & rsi_flips_blue & rsi_was_low
    df["short_entry"] = (sm < -sm_thr) & rsi_flips_purple & rsi_was_high
    return df


def signals_first_of_regime(df, rsi_buy=60, rsi_sell=40, sm_thr=0.0):
    """
    Only take the FIRST RSI flip after SM changes regime (crosses zero).
    Once a signal fires, ignore all further signals until SM flips again.
    """
    df = df.copy()
    rsi = df["RSI"].values
    sm = df["net_index"].values

    long_entry = np.zeros(len(df), dtype=bool)
    short_entry = np.zeros(len(df), dtype=bool)

    sm_regime = 0  # 0=none, 1=green, -1=red
    took_long = False
    took_short = False

    for i in range(1, len(df)):
        # Track SM regime changes
        if sm[i] > sm_thr and sm_regime != 1:
            sm_regime = 1
            took_long = False
        elif sm[i] < -sm_thr and sm_regime != -1:
            sm_regime = -1
            took_short = False
        elif -sm_thr <= sm[i] <= sm_thr:
            sm_regime = 0
            took_long = False
            took_short = False

        # RSI flips
        rsi_flips_blue = rsi[i] > rsi_buy and rsi[i-1] <= rsi_buy
        rsi_flips_purple = rsi[i] < rsi_sell and rsi[i-1] >= rsi_sell

        if sm_regime == 1 and rsi_flips_blue and not took_long:
            long_entry[i] = True
            took_long = True

        if sm_regime == -1 and rsi_flips_purple and not took_short:
            short_entry[i] = True
            took_short = True

    df["long_entry"] = long_entry
    df["short_entry"] = short_entry
    return df


def signals_sm_momentum(df, rsi_buy=60, rsi_sell=40, sm_thr=0.0, sm_slope_bars=5):
    """
    SM must be not only positive but INCREASING (for longs) or DECREASING (shorts).
    This catches fresh momentum, not stale trends.
    """
    df = df.copy()
    rsi = df["RSI"]
    sm = df["net_index"]

    rsi_flips_blue = (rsi > rsi_buy) & (rsi.shift(1) <= rsi_buy)
    rsi_flips_purple = (rsi < rsi_sell) & (rsi.shift(1) >= rsi_sell)

    sm_slope = sm - sm.shift(sm_slope_bars)
    sm_rising = (sm > sm_thr) & (sm_slope > 0)
    sm_falling = (sm < -sm_thr) & (sm_slope < 0)

    df["long_entry"] = sm_rising & rsi_flips_blue
    df["short_entry"] = sm_falling & rsi_flips_purple
    return df


def signals_with_cooldown(df, rsi_buy=60, rsi_sell=40, sm_thr=0.0, cooldown_bars=30):
    """
    Standard signals but with a cooldown period after each trade signal.
    Prevents rapid-fire entries.
    """
    df = df.copy()
    rsi = df["RSI"].values
    sm = df["net_index"].values

    long_entry = np.zeros(len(df), dtype=bool)
    short_entry = np.zeros(len(df), dtype=bool)

    last_signal_bar = -cooldown_bars - 1

    for i in range(1, len(df)):
        if i - last_signal_bar < cooldown_bars:
            continue

        rsi_flips_blue = rsi[i] > rsi_buy and rsi[i-1] <= rsi_buy
        rsi_flips_purple = rsi[i] < rsi_sell and rsi[i-1] >= rsi_sell

        if sm[i] > sm_thr and rsi_flips_blue:
            long_entry[i] = True
            last_signal_bar = i
        elif sm[i] < -sm_thr and rsi_flips_purple:
            short_entry[i] = True
            last_signal_bar = i

    df["long_entry"] = long_entry
    df["short_entry"] = short_entry
    return df


def signals_sm_high_interest(df, rsi_buy=60, rsi_sell=40, sm_hi=0.5):
    """
    SM must be in the "high interest" zone (|net_index| > threshold).
    User uses 0.85 as high interest threshold in the indicator.
    We test various levels.
    """
    df = df.copy()
    rsi = df["RSI"]
    sm = df["net_index"]

    rsi_flips_blue = (rsi > rsi_buy) & (rsi.shift(1) <= rsi_buy)
    rsi_flips_purple = (rsi < rsi_sell) & (rsi.shift(1) >= rsi_sell)

    df["long_entry"] = (sm > sm_hi) & rsi_flips_blue
    df["short_entry"] = (sm < -sm_hi) & rsi_flips_purple
    return df


def signals_rsi_sustained(df, rsi_buy=60, rsi_sell=40, sm_thr=0.0, sustain_bars=3):
    """
    RSI must stay above rsi_buy (or below rsi_sell) for N consecutive bars
    after the flip. Signal fires on the Nth bar of sustained RSI.
    Filters out quick spikes that immediately reverse.
    """
    df = df.copy()
    rsi = df["RSI"].values
    sm = df["net_index"].values

    long_entry = np.zeros(len(df), dtype=bool)
    short_entry = np.zeros(len(df), dtype=bool)

    for i in range(sustain_bars, len(df)):
        # Check RSI sustained above rsi_buy for sustain_bars
        rsi_sustained_blue = all(rsi[i-j] > rsi_buy for j in range(sustain_bars))
        rsi_just_entered = rsi[i - sustain_bars] <= rsi_buy  # was below before

        rsi_sustained_purple = all(rsi[i-j] < rsi_sell for j in range(sustain_bars))
        rsi_just_entered_sell = rsi[i - sustain_bars] >= rsi_sell

        if sm[i] > sm_thr and rsi_sustained_blue and rsi_just_entered:
            long_entry[i] = True
        if sm[i] < -sm_thr and rsi_sustained_purple and rsi_just_entered_sell:
            short_entry[i] = True

    df["long_entry"] = long_entry
    df["short_entry"] = short_entry
    return df


def signals_combined_best(df, rsi_buy=60, rsi_sell=40, sm_thr=0.3,
                          cooldown_bars=20, first_only=True):
    """
    Combined approach: strong SM + RSI flip + cooldown + optionally first-of-regime only.
    """
    df = df.copy()
    rsi = df["RSI"].values
    sm = df["net_index"].values

    long_entry = np.zeros(len(df), dtype=bool)
    short_entry = np.zeros(len(df), dtype=bool)

    last_signal_bar = -cooldown_bars - 1
    sm_regime = 0
    took_long = False
    took_short = False

    for i in range(1, len(df)):
        # Track regime
        if sm[i] > sm_thr and sm_regime != 1:
            sm_regime = 1
            took_long = False
        elif sm[i] < -sm_thr and sm_regime != -1:
            sm_regime = -1
            took_short = False
        elif -sm_thr <= sm[i] <= sm_thr:
            sm_regime = 0
            took_long = False
            took_short = False

        if i - last_signal_bar < cooldown_bars:
            continue

        rsi_flips_blue = rsi[i] > rsi_buy and rsi[i-1] <= rsi_buy
        rsi_flips_purple = rsi[i] < rsi_sell and rsi[i-1] >= rsi_sell

        if sm_regime == 1 and rsi_flips_blue:
            if not first_only or not took_long:
                long_entry[i] = True
                last_signal_bar = i
                took_long = True

        if sm_regime == -1 and rsi_flips_purple:
            if not first_only or not took_short:
                short_entry[i] = True
                last_signal_bar = i
                took_short = True

    df["long_entry"] = long_entry
    df["short_entry"] = short_entry
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_indicator_csv("CME_MINI_MNQ1!, 1_9f80c.csv")
    print(f"\nData: MNQ 1-min with REAL TV indicators")
    print(f"  Bars: {len(df)} | {df.index[0]} to {df.index[-1]}")

    # Count trading days
    trading_days = df.index.normalize().nunique()
    print(f"  Trading days: ~{trading_days}")

    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.005,
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        pyramiding=1,
        start_date="2020-01-01",
        end_date="2069-12-31",
    )

    # Build all signal variations
    signal_configs = []

    # A) Strong SM threshold (fewer, higher quality signals)
    for sm in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for rb, rs in [(60, 40), (65, 35)]:
            name = f"Basic RSI{rb}/{rs} SM>{sm}"
            signal_configs.append((name, lambda d, rb=rb, rs=rs, sm=sm:
                signals_basic(d, rb, rs, sm)))

    # B) Extreme reversal
    for lb in [5, 10, 15, 20]:
        for sm in [0.0, 0.1, 0.2]:
            name = f"ExtrRev lb{lb} SM>{sm}"
            signal_configs.append((name, lambda d, lb=lb, sm=sm:
                signals_extreme_reversal(d, 60, 40, sm, 30, 70, lb)))

    # C) First of regime
    for sm in [0.0, 0.1, 0.2, 0.3]:
        for rb, rs in [(60, 40), (65, 35)]:
            name = f"1stRegime RSI{rb}/{rs} SM>{sm}"
            signal_configs.append((name, lambda d, rb=rb, rs=rs, sm=sm:
                signals_first_of_regime(d, rb, rs, sm)))

    # D) SM momentum
    for slope in [3, 5, 10]:
        for sm in [0.0, 0.1, 0.2]:
            name = f"SMMomentum s{slope} SM>{sm}"
            signal_configs.append((name, lambda d, slope=slope, sm=sm:
                signals_sm_momentum(d, 60, 40, sm, slope)))

    # E) Cooldown
    for cd in [15, 30, 60, 120]:
        for sm in [0.0, 0.1, 0.2]:
            name = f"Cooldown {cd}bar SM>{sm}"
            signal_configs.append((name, lambda d, cd=cd, sm=sm:
                signals_with_cooldown(d, 60, 40, sm, cd)))

    # F) High interest zone
    for hi in [0.3, 0.4, 0.5, 0.6, 0.7, 0.85]:
        for rb, rs in [(60, 40), (65, 35)]:
            name = f"HighInt>{hi} RSI{rb}/{rs}"
            signal_configs.append((name, lambda d, hi=hi, rb=rb, rs=rs:
                signals_sm_high_interest(d, rb, rs, hi)))

    # G) RSI sustained
    for sb in [2, 3, 5]:
        for sm in [0.0, 0.1, 0.2]:
            name = f"RSIsustain {sb}bar SM>{sm}"
            signal_configs.append((name, lambda d, sb=sb, sm=sm:
                signals_rsi_sustained(d, 60, 40, sm, sb)))

    # H) Combined best
    for sm in [0.1, 0.2, 0.3, 0.4]:
        for cd in [15, 30, 60]:
            for first in [True, False]:
                fo_str = "1st" if first else "all"
                name = f"Combined SM>{sm} cd{cd} {fo_str}"
                signal_configs.append((name, lambda d, sm=sm, cd=cd, first=first:
                    signals_combined_best(d, 60, 40, sm, cd, first)))

    # Exit configurations — focused on your 10-15pt scalp targets
    exit_configs = [
        ("TP10 SL5",           dict(tp_points=10, sl_points=5,  max_hold_bars=0, exit_on_sm_flip=False)),
        ("TP15 SL7",           dict(tp_points=15, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=False)),
        ("TP15 SL10",          dict(tp_points=15, sl_points=10, max_hold_bars=0, exit_on_sm_flip=False)),
        ("TP10 SL5 +SM",      dict(tp_points=10, sl_points=5,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP15 SL7 +SM",      dict(tp_points=15, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP15 SL10 +SM",     dict(tp_points=15, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP20 SL10 +SM",     dict(tp_points=20, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),
        ("SM exit SL7",        dict(tp_points=9999, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("SM exit SL10",       dict(tp_points=9999, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),
        ("SM exit SL15",       dict(tp_points=9999, sl_points=15, max_hold_bars=0, exit_on_sm_flip=True)),
    ]

    results = []
    sig_trade_counts = {}

    for sig_name, sig_fn in signal_configs:
        try:
            df_sig = sig_fn(df.copy())
        except Exception:
            continue
        n_long = df_sig["long_entry"].sum()
        n_short = df_sig["short_entry"].sum()
        n_total = n_long + n_short

        sig_trade_counts[sig_name] = n_total

        if n_total == 0:
            continue

        # Target ~5 trades/day = ~50 over 10 days. Skip if >200 signals.
        if n_total > 300:
            continue

        for exit_name, exit_params in exit_configs:
            name = f"{sig_name} | {exit_name}"
            try:
                kpis = run_scalper(df_sig, config, **exit_params)
                results.append({"name": name, "kpis": kpis,
                                "sig": sig_name, "exit": exit_name})
            except Exception:
                pass

    # === Signal count summary ===
    print(f"\n{'='*80}")
    print(f"  SIGNAL COUNT SUMMARY (target: ~50 signals over ~{trading_days} days)")
    print(f"{'='*80}")
    sorted_sigs = sorted(sig_trade_counts.items(), key=lambda x: x[1])
    for name, count in sorted_sigs:
        per_day = count / trading_days if trading_days > 0 else 0
        marker = "***" if 20 <= count <= 80 else "   "
        if count == 0:
            marker = " X "
        print(f"  {marker} {name:<45} {count:>5} signals ({per_day:.1f}/day)")

    # === COMPARISON TABLE ===
    print(f"\n\n{'='*165}")
    print(f"  SCALPER V4 — Selective Entries with REAL TV Data (1-min MNQ)")
    print(f"{'='*165}")
    header = (f"  {'Signals':<38} {'Exit':<18} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} "
              f"{'PF':>7} {'MaxDD%':>8} {'Sharpe':>7} {'Avg$':>8}")
    print(header)
    print("  " + "-" * 157)

    ranked = []
    profitable = []

    for r in results:
        k = r["kpis"]
        if k is None or "error" in k or k.get("total_trades", 0) == 0:
            continue

        pf = k["profit_factor"]
        pf_display = f"{pf:>7.2f}" if pf != float("inf") else "    inf"
        n_trades = k["total_trades"]

        marker = " "
        if k["net_profit"] > 0:
            marker = "+"
            profitable.append(r)

        line = (f"{marker} {r['sig']:<38} {r['exit']:<18} "
                f"${k['net_profit']:>8,.2f} "
                f"{k['net_profit_pct']:>6.2f}% "
                f"{n_trades:>5d} "
                f"{k['win_rate']:>6.2f}% "
                f"{pf_display} "
                f"{k['max_drawdown_pct']:>7.2f}% "
                f"{k['sharpe_ratio']:>7.3f} "
                f"${k['avg_trade']:>6,.2f}")
        print(line)

        if n_trades >= 5:
            score = (min(pf, 10) * 2 + k["sharpe_ratio"] * 3 +
                     k["win_rate"] / 25 + min(n_trades, 50) / 15)
        elif n_trades >= 2:
            score = (min(pf, 10) + k["sharpe_ratio"]) * 0.5
        else:
            score = -100
        if k["net_profit"] < 0:
            score *= 0.3
        ranked.append((score, r))

    print(f"{'='*165}")
    print(f"\n  Total combos tested: {len(results)}")
    print(f"  Profitable: {len(profitable)}")
    print(f"  Unprofitable: {len(results) - len(profitable)}")

    ranked.sort(key=lambda x: x[0], reverse=True)
    print(f"\n  TOP 20 STRATEGIES:")
    print(f"  {'#':<4} {'Signals':<38} {'Exit':<18} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} {'PF':>7} {'MaxDD%':>8} {'Sharpe':>7}")
    print(f"  {'-'*130}")
    for i, (score, r) in enumerate(ranked[:20], 1):
        k = r["kpis"]
        pf = k["profit_factor"]
        pf_str = f"{pf:>7.2f}" if pf != float("inf") else "    inf"
        marker = "+" if k["net_profit"] > 0 else " "
        print(f"  {i:<4}{marker}{r['sig']:<38} {r['exit']:<18} "
              f"${k['net_profit']:>8,.2f} "
              f"{k['net_profit_pct']:>6.2f}% "
              f"{k['total_trades']:>5d} "
              f"{k['win_rate']:>6.2f}% "
              f"{pf_str} "
              f"{k['max_drawdown_pct']:>7.2f}% "
              f"{k['sharpe_ratio']:>7.3f}")

    # Show champion details
    if ranked:
        best = ranked[0][1]
        k = best["kpis"]
        print(f"\n\n{'='*70}")
        print(f"  CHAMPION: [{best['sig']}] + [{best['exit']}]")
        print(f"{'='*70}")
        print_kpis(k)
        print_trades(k["trades"], max_trades=50)

    # Also show the best with fixed TP
    scalp_ranked = [(s, r) for s, r in ranked
                    if "SM exit" not in r["exit"] and r["kpis"]["total_trades"] >= 5]
    if scalp_ranked and scalp_ranked[0] != ranked[0]:
        best_scalp = scalp_ranked[0][1]
        k2 = best_scalp["kpis"]
        print(f"\n\n{'='*70}")
        print(f"  BEST FIXED-TP SCALPER: [{best_scalp['sig']}] + [{best_scalp['exit']}]")
        print(f"{'='*70}")
        print_kpis(k2)
        print_trades(k2["trades"], max_trades=50)

    # Show profitable strategies summary
    if profitable:
        print(f"\n\n{'='*70}")
        print(f"  ALL PROFITABLE STRATEGIES ({len(profitable)} total):")
        print(f"{'='*70}")
        for r in sorted(profitable, key=lambda x: x["kpis"]["net_profit"], reverse=True):
            k = r["kpis"]
            pf = k["profit_factor"]
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(f"  [{r['sig']}] + [{r['exit']}]")
            print(f"    Net: ${k['net_profit']:,.2f} ({k['net_profit_pct']:.2f}%) | "
                  f"Trades: {k['total_trades']} | Win: {k['win_rate']:.1f}% | "
                  f"PF: {pf_str} | MaxDD: {k['max_drawdown_pct']:.2f}% | "
                  f"Sharpe: {k['sharpe_ratio']:.3f}")


if __name__ == "__main__":
    main()
