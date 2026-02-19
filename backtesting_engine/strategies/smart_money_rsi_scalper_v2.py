"""
Smart Money + RSI Scalper V2 — Filtered entries, smarter exits.

Problems with V1: too many signals (300+), basically random on 1-min.

Fixes:
- Require SM to be "strongly" green/red (above threshold, not just barely > 0)
- Require RSI to STAY blue/purple (not just flip then immediately reverse)
- Add time-based max hold (don't hold a scalp for hours)
- Test TP/SL with the SM flip as a trailing exit
- Test fewer entries by requiring SM to be in agreement for N bars
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine import (
    load_tv_export, BacktestConfig, calc_ema, calc_smma,
    print_kpis, print_trades,
)
from engine.engine import Trade, compute_kpis
from typing import Optional


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    change = series.diff()
    up = change.clip(lower=0)
    dn = -change.clip(upper=0)
    up_smooth = calc_smma(up, length)
    dn_smooth = calc_smma(dn, length)
    rs = up_smooth / dn_smooth
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(dn_smooth != 0, 100.0)
    rsi = rsi.where(up_smooth != 0, 0.0)
    return rsi


def calc_pvi_nvi(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = df["Close"].values
    volume = df["Volume"].values
    pvi = np.full(len(close), np.nan)
    nvi = np.full(len(close), np.nan)
    pvi[0] = 1.0
    nvi[0] = 1.0
    for i in range(1, len(close)):
        pct = (close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0
        pvi[i] = pvi[i-1] + pct * pvi[i-1] if volume[i] > volume[i-1] else pvi[i-1]
        nvi[i] = nvi[i-1] + pct * nvi[i-1] if volume[i] < volume[i-1] else nvi[i-1]
    return pd.Series(pvi, index=df.index), pd.Series(nvi, index=df.index)


def calc_smart_money(df: pd.DataFrame, index_period=10, flow_period=8,
                     norm_period=150, ema_len=255) -> pd.Series:
    pvi, nvi = calc_pvi_nvi(df)
    dumb = pvi - calc_ema(pvi, ema_len)
    smart = nvi - calc_ema(nvi, ema_len)
    drsi = calc_rsi(dumb, flow_period)
    srsi = calc_rsi(smart, flow_period)
    r_buy = (srsi / drsi).replace([np.inf, -np.inf], np.nan).fillna(0)
    r_sell = ((100 - srsi) / (100 - drsi)).replace([np.inf, -np.inf], np.nan).fillna(0)
    sums_buy = r_buy.rolling(window=index_period, min_periods=index_period).sum()
    sums_sell = r_sell.rolling(window=index_period, min_periods=index_period).sum()
    combined_max = pd.concat([sums_buy, sums_sell], axis=1).max(axis=1)
    peak = combined_max.rolling(window=norm_period, min_periods=1).max()
    return sums_buy / peak - sums_sell / peak


def synthesize_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Volume"] = (df["High"] - df["Low"]).clip(lower=0.25)
    return df


# ---------------------------------------------------------------------------
# Scalper engine with TP/SL + SM flip exit + max hold time
# ---------------------------------------------------------------------------

def run_scalper(df: pd.DataFrame, config: BacktestConfig,
                tp_points: float, sl_points: float,
                max_hold_bars: int = 0,
                exit_on_sm_flip: bool = False) -> dict:
    """
    TP/SL scalper with optional SM flip exit and max hold time.

    max_hold_bars: close after N bars if still open (0 = disabled)
    exit_on_sm_flip: close if SM changes sign while in trade
    """
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

    sm_values = df["smart_money"].values if "smart_money" in df.columns else None

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

        # 1) Fill pending entry
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

        # 2) Check exits for open positions
        if position_qty != 0:
            bars_held = i - entry_bar_idx
            exited = False

            if position_side == "long":
                tp_price = position_entry_price + tp_points
                sl_price = position_entry_price - sl_points

                # Check SL first (conservative)
                if bar["Low"] <= sl_price:
                    close_position(bar, bar_date, sl_price)
                    exited = True
                elif bar["High"] >= tp_price:
                    close_position(bar, bar_date, tp_price)
                    exited = True
            else:
                abs_qty = abs(position_qty)
                tp_price = position_entry_price - tp_points
                sl_price = position_entry_price + sl_points

                if bar["High"] >= sl_price:
                    close_position(bar, bar_date, sl_price)
                    exited = True
                elif bar["Low"] <= tp_price:
                    close_position(bar, bar_date, tp_price)
                    exited = True

            # SM flip exit (at close price)
            if not exited and exit_on_sm_flip and sm_values is not None:
                sm_now = sm_values[i]
                if position_side == "long" and sm_now < 0:
                    close_position(bar, bar_date, bar["Close"])
                    exited = True
                elif position_side == "short" and sm_now > 0:
                    close_position(bar, bar_date, bar["Close"])
                    exited = True

            # Max hold time exit (at close price)
            if not exited and max_hold_bars > 0 and bars_held >= max_hold_bars:
                close_position(bar, bar_date, bar["Close"])
                exited = True

        # Drawdown tracking
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

        # Mark-to-market
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

        # 3) Signal detection (only when flat)
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
# Signal generation with filters
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, rsi_len=11, rsi_buy=60, rsi_sell=40,
                     sm_index=10, sm_flow=8, sm_norm=150, sm_ema=255,
                     sm_thr=0.0, sm_bars_confirm=1) -> pd.DataFrame:
    """
    sm_bars_confirm: require SM to be green/red for N consecutive bars before entry.
    """
    df = synthesize_volume(df)
    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index, sm_flow, sm_norm, sm_ema)

    rsi_flips_blue = (df["rsi"] > rsi_buy) & (df["rsi"].shift(1) <= rsi_buy)
    rsi_flips_purple = (df["rsi"] < rsi_sell) & (df["rsi"].shift(1) >= rsi_sell)

    sm_green = df["smart_money"] > sm_thr
    sm_red = df["smart_money"] < -sm_thr

    # SM confirmation: must be green/red for N bars in a row
    if sm_bars_confirm > 1:
        sm_green_confirmed = sm_green.copy()
        sm_red_confirmed = sm_red.copy()
        for j in range(1, sm_bars_confirm):
            sm_green_confirmed = sm_green_confirmed & sm_green.shift(j)
            sm_red_confirmed = sm_red_confirmed & sm_red.shift(j)
    else:
        sm_green_confirmed = sm_green
        sm_red_confirmed = sm_red

    df["long_entry"] = sm_green_confirmed & rsi_flips_blue
    df["short_entry"] = sm_red_confirmed & rsi_flips_purple
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_tv_export("CME_MINI_MNQ1!, 1_e8c40.csv")
    print(f"\nData: MNQ 1-min | {len(df)} bars | {df.index[0]} to {df.index[-1]}")

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

    user = dict(rsi_len=11, sm_index=10, sm_flow=8, sm_norm=150, sm_ema=255)

    # Build signal sets with different filters
    signal_configs = [
        ("Base (RSI 60/40, no filter)",
         dict(**user, rsi_buy=60, rsi_sell=40, sm_thr=0.0, sm_bars_confirm=1)),
        ("SM thr 0.05",
         dict(**user, rsi_buy=60, rsi_sell=40, sm_thr=0.05, sm_bars_confirm=1)),
        ("SM thr 0.1",
         dict(**user, rsi_buy=60, rsi_sell=40, sm_thr=0.1, sm_bars_confirm=1)),
        ("SM 3-bar confirm",
         dict(**user, rsi_buy=60, rsi_sell=40, sm_thr=0.0, sm_bars_confirm=3)),
        ("SM 5-bar confirm",
         dict(**user, rsi_buy=60, rsi_sell=40, sm_thr=0.0, sm_bars_confirm=5)),
        ("RSI 65/35",
         dict(**user, rsi_buy=65, rsi_sell=35, sm_thr=0.0, sm_bars_confirm=1)),
        ("RSI 65/35 + SM thr 0.05",
         dict(**user, rsi_buy=65, rsi_sell=35, sm_thr=0.05, sm_bars_confirm=1)),
        ("RSI 65/35 + SM 3-bar",
         dict(**user, rsi_buy=65, rsi_sell=35, sm_thr=0.0, sm_bars_confirm=3)),
        ("RSI 65/35 + SM 5-bar",
         dict(**user, rsi_buy=65, rsi_sell=35, sm_thr=0.0, sm_bars_confirm=5)),
        ("RSI 65/35 + SM thr 0.05 + 3-bar",
         dict(**user, rsi_buy=65, rsi_sell=35, sm_thr=0.05, sm_bars_confirm=3)),
    ]

    # Exit configurations
    exit_configs = [
        ("TP15 SL7",           dict(tp_points=15, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=False)),
        ("TP15 SL7 +SMflip",   dict(tp_points=15, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP15 SL10 +SMflip",  dict(tp_points=15, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP10 SL5 +SMflip",   dict(tp_points=10, sl_points=5,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP20 SL10 +SMflip",  dict(tp_points=20, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP15 SL7 +30bar",    dict(tp_points=15, sl_points=7,  max_hold_bars=30, exit_on_sm_flip=False)),
        ("TP15 SL7 +SM+30bar", dict(tp_points=15, sl_points=7,  max_hold_bars=30, exit_on_sm_flip=True)),
        ("TP15 SL10 +SM+60bar",dict(tp_points=15, sl_points=10, max_hold_bars=60, exit_on_sm_flip=True)),
        ("TP20 SL10 +SM+60bar",dict(tp_points=20, sl_points=10, max_hold_bars=60, exit_on_sm_flip=True)),
        ("SMflip only SL10",   dict(tp_points=9999, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),
        ("SMflip only SL15",   dict(tp_points=9999, sl_points=15, max_hold_bars=0, exit_on_sm_flip=True)),
        ("SMflip only SL7",    dict(tp_points=9999, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=True)),
    ]

    results = []

    for sig_name, sig_params in signal_configs:
        df_sig = generate_signals(df.copy(), **sig_params)
        n_long = df_sig["long_entry"].sum()
        n_short = df_sig["short_entry"].sum()

        if n_long == 0 and n_short == 0:
            continue

        for exit_name, exit_params in exit_configs:
            name = f"{sig_name} | {exit_name}"
            try:
                kpis = run_scalper(df_sig, config, **exit_params)
                results.append({"name": name, "kpis": kpis,
                                "sig": sig_name, "exit": exit_name})
            except Exception as e:
                pass

    # === COMPARISON TABLE (only show profitable or near-breakeven) ===
    print("\n\n" + "=" * 155)
    print("  SCALPER V2 — Filtered Entries + Smart Exits")
    print("=" * 155)
    header = (f"  {'Signals':<30} {'Exit':<22} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} "
              f"{'PF':>7} {'MaxDD%':>8} {'Sharpe':>7} {'Avg$':>8}")
    print(header)
    print("  " + "-" * 147)

    ranked = []

    for r in results:
        k = r["kpis"]
        if k is None or "error" in k or k.get("total_trades", 0) == 0:
            continue

        pf = k["profit_factor"]
        pf_display = f"{pf:>7.2f}" if pf != float("inf") else "    inf"
        n_trades = k["total_trades"]

        line = (f"  {r['sig']:<30} {r['exit']:<22} "
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
            score *= 0.5
        ranked.append((score, r))

    print("=" * 155)

    ranked.sort(key=lambda x: x[0], reverse=True)
    print("\n  TOP 10 STRATEGIES:")
    for i, (score, r) in enumerate(ranked[:10], 1):
        k = r["kpis"]
        pf = k["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  #{i}: [{r['sig']}] + [{r['exit']}]")
        print(f"      Net: ${k['net_profit']:,.2f} ({k['net_profit_pct']:.2f}%) | "
              f"Trades: {k['total_trades']} | Win: {k['win_rate']:.1f}% | "
              f"PF: {pf_str} | MaxDD: {k['max_drawdown_pct']:.2f}% | "
              f"Sharpe: {k['sharpe_ratio']:.3f}")

    # Show champion details
    if ranked:
        best = ranked[0][1]
        k = best["kpis"]
        print(f"\n\n{'='*60}")
        print(f"  CHAMPION: [{best['sig']}] + [{best['exit']}]")
        print(f"{'='*60}")
        print_kpis(k)
        print_trades(k["trades"], max_trades=30)


if __name__ == "__main__":
    main()
