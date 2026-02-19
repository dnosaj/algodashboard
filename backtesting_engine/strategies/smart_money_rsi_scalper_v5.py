"""
Smart Money + RSI Scalper V5 — Deep dive on winning approaches from V4.

Winners from V4 (real TV data, 1-min MNQ):
1. "Extreme Reversal" — RSI was recently at extreme before flipping
   - ExtrRev lb5 + TP15/SL7: 53.8% win, PF 1.54, 13 trades
2. "SM Momentum" — SM must be rising (for longs) with SM exit
   - SMMomentum s3 SM>0.2 + SM exit SL15: 38.9% win, PF 1.25, 72 trades
3. "High Interest" — SM in high zone (>0.5) with tight SL
   - HighInt>0.5 RSI65/35 + SM exit SL7: 27.3% win, PF 1.23, 11 trades

This version:
- Tests many more variations of these 3 winning approaches
- Tests combinations of them (e.g., extreme reversal + SM momentum)
- Uses $2/point for MNQ to show real dollar P&L with 10 contracts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine.engine import Trade, compute_kpis, BacktestConfig
from engine import print_kpis, print_trades
from typing import Optional


def load_indicator_csv(filename: str) -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    filepath = data_dir / filename
    df = pd.read_csv(filepath)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close", cols[5]: "RSI",
        cols[16]: "SM_Buy", cols[22]: "SM_Sell",
    })
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    sm_buy = pd.to_numeric(df["SM_Buy"], errors="coerce").fillna(0.0)
    sm_sell = pd.to_numeric(df["SM_Sell"], errors="coerce").fillna(0.0)
    df["net_index"] = sm_buy + sm_sell
    df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")
    df = df.iloc[:-1]
    return df[["Open", "High", "Low", "Close", "RSI", "net_index"]]


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
# Signal generators — focused on V4 winners
# ---------------------------------------------------------------------------

def signals_extreme_reversal(df, rsi_buy=60, rsi_sell=40, sm_thr=0.0,
                              extreme_lo=30, extreme_hi=70, lookback=5,
                              sm_slope_bars=0, sm_slope_min=0.0,
                              cooldown=0, first_of_regime=False):
    """
    Full-featured extreme reversal signal with optional SM momentum filter,
    cooldown, and first-of-regime restriction.
    """
    df = df.copy()
    rsi = df["RSI"].values
    sm = df["net_index"].values
    n = len(df)

    # Pre-compute rolling min/max for RSI extremes
    rsi_roll_min = pd.Series(rsi).rolling(lookback, min_periods=1).min().values
    rsi_roll_max = pd.Series(rsi).rolling(lookback, min_periods=1).max().values

    # SM slope if needed
    if sm_slope_bars > 0:
        sm_series = pd.Series(sm)
        sm_slope = (sm_series - sm_series.shift(sm_slope_bars)).values
    else:
        sm_slope = np.zeros(n)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    last_signal_bar = -cooldown - 1
    sm_regime = 0
    took_long = False
    took_short = False

    for i in range(1, n):
        # Track SM regime for first_of_regime
        if first_of_regime:
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

        # Cooldown check
        if cooldown > 0 and i - last_signal_bar < cooldown:
            continue

        # RSI flips
        rsi_flips_blue = rsi[i] > rsi_buy and rsi[i-1] <= rsi_buy
        rsi_flips_purple = rsi[i] < rsi_sell and rsi[i-1] >= rsi_sell

        # RSI extreme check
        rsi_was_low = rsi_roll_min[i] < extreme_lo
        rsi_was_high = rsi_roll_max[i] > extreme_hi

        # SM conditions
        sm_ok_long = sm[i] > sm_thr
        sm_ok_short = sm[i] < -sm_thr

        # SM slope condition
        if sm_slope_bars > 0:
            sm_ok_long = sm_ok_long and sm_slope[i] > sm_slope_min
            sm_ok_short = sm_ok_short and sm_slope[i] < -sm_slope_min

        # Generate signals
        if sm_ok_long and rsi_flips_blue and rsi_was_low:
            if not first_of_regime or not took_long:
                long_entry[i] = True
                last_signal_bar = i
                took_long = True

        if sm_ok_short and rsi_flips_purple and rsi_was_high:
            if not first_of_regime or not took_short:
                short_entry[i] = True
                last_signal_bar = i
                took_short = True

    df["long_entry"] = long_entry
    df["short_entry"] = short_entry
    return df


def signals_sm_momentum_extreme(df, rsi_buy=60, rsi_sell=40, sm_thr=0.0,
                                 sm_slope_bars=3, extreme_lo=30, extreme_hi=70,
                                 lookback=5, cooldown=0):
    """
    Combined: SM momentum + extreme reversal.
    SM must be positive AND rising, plus RSI must have been at extreme.
    """
    df = df.copy()
    rsi = df["RSI"].values
    sm = df["net_index"].values
    n = len(df)

    sm_series = pd.Series(sm)
    sm_slope = (sm_series - sm_series.shift(sm_slope_bars)).values

    rsi_roll_min = pd.Series(rsi).rolling(lookback, min_periods=1).min().values
    rsi_roll_max = pd.Series(rsi).rolling(lookback, min_periods=1).max().values

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    last_signal_bar = -cooldown - 1

    for i in range(1, n):
        if cooldown > 0 and i - last_signal_bar < cooldown:
            continue

        rsi_flips_blue = rsi[i] > rsi_buy and rsi[i-1] <= rsi_buy
        rsi_flips_purple = rsi[i] < rsi_sell and rsi[i-1] >= rsi_sell

        rsi_was_low = rsi_roll_min[i] < extreme_lo
        rsi_was_high = rsi_roll_max[i] > extreme_hi

        sm_rising = sm[i] > sm_thr and sm_slope[i] > 0
        sm_falling = sm[i] < -sm_thr and sm_slope[i] < 0

        if sm_rising and rsi_flips_blue and rsi_was_low:
            long_entry[i] = True
            last_signal_bar = i

        if sm_falling and rsi_flips_purple and rsi_was_high:
            short_entry[i] = True
            last_signal_bar = i

    df["long_entry"] = long_entry
    df["short_entry"] = short_entry
    return df


def signals_high_interest_extreme(df, rsi_buy=60, rsi_sell=40, sm_hi=0.5,
                                   extreme_lo=30, extreme_hi=70, lookback=5):
    """
    SM must be in high interest zone AND RSI came from extreme.
    """
    df = df.copy()
    rsi = df["RSI"]
    sm = df["net_index"]

    rsi_flips_blue = (rsi > rsi_buy) & (rsi.shift(1) <= rsi_buy)
    rsi_flips_purple = (rsi < rsi_sell) & (rsi.shift(1) >= rsi_sell)

    rsi_was_low = rsi.rolling(lookback).min() < extreme_lo
    rsi_was_high = rsi.rolling(lookback).max() > extreme_hi

    df["long_entry"] = (sm > sm_hi) & rsi_flips_blue & rsi_was_low
    df["short_entry"] = (sm < -sm_hi) & rsi_flips_purple & rsi_was_high
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_indicator_csv("CME_MINI_MNQ1!, 1_9f80c.csv")
    trading_days = df.index.normalize().nunique()
    print(f"\nData: MNQ 1-min REAL TV indicators | {len(df)} bars | ~{trading_days} days")

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

    signal_configs = []

    # =====================================================================
    # A) Extreme Reversal — focused sweep (reduced from 270 to ~60)
    # =====================================================================
    for lb in [3, 5, 10, 15]:
        for exlo, exhi in [(30, 70), (35, 65)]:
            for sm in [0.0, 0.1, 0.2]:
                for rb, rs in [(60, 40), (65, 35)]:
                    name = f"ER lb{lb} ex{exlo}/{exhi} SM>{sm} RSI{rb}/{rs}"
                    signal_configs.append((name, lambda d, lb=lb, exlo=exlo, exhi=exhi,
                        sm=sm, rb=rb, rs=rs:
                        signals_extreme_reversal(d, rb, rs, sm, exlo, exhi, lb)))

    # B) Extreme Reversal + SM momentum
    for lb in [3, 5, 10]:
        for slope in [3, 5]:
            for sm in [0.0, 0.1, 0.2]:
                name = f"ER+Mom lb{lb} s{slope} SM>{sm}"
                signal_configs.append((name, lambda d, lb=lb, slope=slope, sm=sm:
                    signals_extreme_reversal(d, 60, 40, sm, 30, 70, lb,
                                           sm_slope_bars=slope, sm_slope_min=0)))

    # C) Extreme Reversal + cooldown (reduced)
    for lb in [5, 10]:
        for cd in [15, 30]:
            for sm in [0.0, 0.1]:
                name = f"ER+CD lb{lb} cd{cd} SM>{sm}"
                signal_configs.append((name, lambda d, lb=lb, cd=cd, sm=sm:
                    signals_extreme_reversal(d, 60, 40, sm, 30, 70, lb,
                                           cooldown=cd)))

    # D) Extreme Reversal + first of regime
    for lb in [5, 10]:
        for sm in [0.0, 0.1]:
            name = f"ER+1stReg lb{lb} SM>{sm}"
            signal_configs.append((name, lambda d, lb=lb, sm=sm:
                signals_extreme_reversal(d, 60, 40, sm, 30, 70, lb,
                                       first_of_regime=True)))

    # E) SM Momentum + Extreme combined (reduced)
    for lb in [3, 5]:
        for slope in [3, 5]:
            for sm in [0.0, 0.1, 0.2]:
                for cd in [0, 15]:
                    name = f"Mom+ER lb{lb} s{slope} SM>{sm} cd{cd}"
                    signal_configs.append((name, lambda d, lb=lb, slope=slope, sm=sm, cd=cd:
                        signals_sm_momentum_extreme(d, 60, 40, sm, slope, 30, 70, lb, cd)))

    # F) High Interest + Extreme (reduced)
    for hi in [0.3, 0.5, 0.6]:
        for lb in [5, 10]:
            name = f"HI>{hi}+ER lb{lb}"
            signal_configs.append((name, lambda d, hi=hi, lb=lb:
                signals_high_interest_extreme(d, 60, 40, hi, 30, 70, lb)))

    # Exit configurations
    exit_configs = [
        ("TP10 SL5",       dict(tp_points=10, sl_points=5,  exit_on_sm_flip=False)),
        ("TP10 SL7",       dict(tp_points=10, sl_points=7,  exit_on_sm_flip=False)),
        ("TP12 SL5",       dict(tp_points=12, sl_points=5,  exit_on_sm_flip=False)),
        ("TP12 SL7",       dict(tp_points=12, sl_points=7,  exit_on_sm_flip=False)),
        ("TP15 SL5",       dict(tp_points=15, sl_points=5,  exit_on_sm_flip=False)),
        ("TP15 SL7",       dict(tp_points=15, sl_points=7,  exit_on_sm_flip=False)),
        ("TP15 SL10",      dict(tp_points=15, sl_points=10, exit_on_sm_flip=False)),
        ("TP20 SL7",       dict(tp_points=20, sl_points=7,  exit_on_sm_flip=False)),
        ("TP20 SL10",      dict(tp_points=20, sl_points=10, exit_on_sm_flip=False)),
        ("TP10 SL5 +SM",   dict(tp_points=10, sl_points=5,  exit_on_sm_flip=True)),
        ("TP15 SL7 +SM",   dict(tp_points=15, sl_points=7,  exit_on_sm_flip=True)),
        ("TP15 SL10 +SM",  dict(tp_points=15, sl_points=10, exit_on_sm_flip=True)),
        ("TP20 SL10 +SM",  dict(tp_points=20, sl_points=10, exit_on_sm_flip=True)),
        ("SM exit SL7",    dict(tp_points=9999, sl_points=7,  exit_on_sm_flip=True)),
        ("SM exit SL10",   dict(tp_points=9999, sl_points=10, exit_on_sm_flip=True)),
        ("SM exit SL15",   dict(tp_points=9999, sl_points=15, exit_on_sm_flip=True)),
    ]

    results = []

    for sig_name, sig_fn in signal_configs:
        try:
            df_sig = sig_fn(df.copy())
        except Exception:
            continue
        n_long = df_sig["long_entry"].sum()
        n_short = df_sig["short_entry"].sum()
        n_total = n_long + n_short
        if n_total == 0 or n_total > 200:
            continue

        for exit_name, exit_params in exit_configs:
            try:
                kpis = run_scalper(df_sig, config, **exit_params)
                results.append({"name": f"{sig_name} | {exit_name}",
                                "kpis": kpis,
                                "sig": sig_name, "exit": exit_name,
                                "n_signals": n_total})
            except Exception:
                pass

    # Rank all results
    ranked = []
    for r in results:
        k = r["kpis"]
        if k is None or "error" in k or k.get("total_trades", 0) == 0:
            continue
        pf = k["profit_factor"]
        n_trades = k["total_trades"]
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

    ranked.sort(key=lambda x: x[0], reverse=True)

    # Summary
    profitable = [r for _, r in ranked if r["kpis"]["net_profit"] > 0]
    print(f"\n  Total combos tested: {len(results)}")
    print(f"  Profitable: {len(profitable)} / {len(results)}")

    # TOP 30
    print(f"\n{'='*175}")
    print(f"  SCALPER V5 — Deep Dive on Winners | 1-min MNQ | REAL TV Data")
    print(f"{'='*175}")
    print(f"  {'#':<4} {'Signals':<48} {'Exit':<16} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} {'PF':>7} {'MaxDD%':>8} {'Sharpe':>7} {'Sig':>5}")
    print(f"  {'-'*140}")

    for i, (score, r) in enumerate(ranked[:30], 1):
        k = r["kpis"]
        pf = k["profit_factor"]
        pf_str = f"{pf:>7.2f}" if pf != float("inf") else "    inf"
        marker = "+" if k["net_profit"] > 0 else " "
        print(f"  {i:<4}{marker}{r['sig']:<48} {r['exit']:<16} "
              f"${k['net_profit']:>8,.2f} "
              f"{k['net_profit_pct']:>6.2f}% "
              f"{k['total_trades']:>5d} "
              f"{k['win_rate']:>6.2f}% "
              f"{pf_str} "
              f"{k['max_drawdown_pct']:>7.2f}% "
              f"{k['sharpe_ratio']:>7.3f} "
              f"{r['n_signals']:>5d}")

    # Champion details
    if ranked:
        best = ranked[0][1]
        k = best["kpis"]
        print(f"\n\n{'='*70}")
        print(f"  CHAMPION: [{best['sig']}] + [{best['exit']}]")
        print(f"{'='*70}")
        print_kpis(k)

        # Calculate MNQ 10-lot dollar values
        # MNQ is $2 per point per contract, 10 contracts
        print(f"\n  === WITH 10 MNQ CONTRACTS ($2/pt/contract = $20/point total) ===")
        for t in k["trades"]:
            pts = (t.exit_price - t.entry_price) if t.direction == "long" else (t.entry_price - t.exit_price)
            dollar_pnl = pts * 20  # $2/pt * 10 contracts
            commission = 0.52 * 10 * 2  # $0.52/contract * 10 contracts * round trip
            net = dollar_pnl - commission
            print(f"    {t.direction:>5s} {str(t.entry_date):>20s} -> {str(t.exit_date):>20s}  "
                  f"Entry: {t.entry_price:>10,.2f}  Exit: {t.exit_price:>10,.2f}  "
                  f"Pts: {pts:>+7.2f}  $PnL: ${net:>+8.2f}")

        total_pts = sum(
            (t.exit_price - t.entry_price) if t.direction == "long" else (t.entry_price - t.exit_price)
            for t in k["trades"]
        )
        total_dollar = total_pts * 20
        total_commission = 0.52 * 10 * 2 * len(k["trades"])
        total_net = total_dollar - total_commission
        print(f"\n    TOTAL: {total_pts:>+.2f} pts | "
              f"Gross: ${total_dollar:>+,.2f} | "
              f"Commission: ${total_commission:>,.2f} | "
              f"Net: ${total_net:>+,.2f}")
        print(f"    Per day (~{trading_days} days): ${total_net/trading_days:>+,.2f}/day")

    # Show ALL profitable strategies
    print(f"\n\n{'='*70}")
    print(f"  ALL PROFITABLE ({len(profitable)} strategies):")
    print(f"{'='*70}")
    for r in sorted(profitable, key=lambda x: x["kpis"]["net_profit"], reverse=True)[:30]:
        k = r["kpis"]
        pf = k["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  [{r['sig']}] + [{r['exit']}]")
        print(f"    Net: ${k['net_profit']:,.2f} ({k['net_profit_pct']:.2f}%) | "
              f"Trades: {k['total_trades']} | Win: {k['win_rate']:.1f}% | "
              f"PF: {pf_str} | MaxDD: {k['max_drawdown_pct']:.2f}% | "
              f"Sharpe: {k['sharpe_ratio']:.3f}")

    # Best fixed-TP scalper (your style - TP10-15)
    fixed_tp = [(s, r) for s, r in ranked
                if r["kpis"]["net_profit"] > 0
                and "SM exit" not in r["exit"]
                and r["kpis"]["total_trades"] >= 5]
    if fixed_tp:
        best_scalp = fixed_tp[0][1]
        k2 = best_scalp["kpis"]
        print(f"\n\n{'='*70}")
        print(f"  BEST FIXED-TP SCALPER: [{best_scalp['sig']}] + [{best_scalp['exit']}]")
        print(f"{'='*70}")
        print_kpis(k2)
        print_trades(k2["trades"], max_trades=50)

        # MNQ 10-lot
        print(f"\n  === WITH 10 MNQ CONTRACTS ===")
        for t in k2["trades"]:
            pts = (t.exit_price - t.entry_price) if t.direction == "long" else (t.entry_price - t.exit_price)
            dollar_pnl = pts * 20
            commission = 0.52 * 10 * 2
            net = dollar_pnl - commission
            print(f"    {t.direction:>5s} {str(t.entry_date):>20s} -> {str(t.exit_date):>20s}  "
                  f"Pts: {pts:>+7.2f}  10-lot $: ${net:>+8.2f}")

        total_pts = sum(
            (t.exit_price - t.entry_price) if t.direction == "long" else (t.entry_price - t.exit_price)
            for t in k2["trades"]
        )
        total_net = total_pts * 20 - 0.52 * 10 * 2 * len(k2["trades"])
        print(f"\n    TOTAL: {total_pts:>+.2f} pts | Net 10-lot: ${total_net:>+,.2f} | "
              f"Per day: ${total_net/trading_days:>+,.2f}")


if __name__ == "__main__":
    main()
