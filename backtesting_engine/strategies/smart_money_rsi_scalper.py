"""
Smart Money + RSI Scalper for MNQ — 1-minute chart.

User wants:
- Scalp 10-15 MNQ points per trade
- Cut losses early (tight stop loss)
- Not always-in — discrete trades with TP/SL
- Use their indicator settings: RSI 11, SM(10, 8, 150, 255)

Entry:
  LONG  — SM green (net_index > 0) + RSI crosses above 60 (flips blue)
  SHORT — SM red (net_index < 0) + RSI crosses below 40 (flips purple)

Exit:
  Take Profit: +10 to +20 points from entry
  Stop Loss: -5 to -15 points from entry
  (Test various TP/SL combos)

MNQ point value: $0.50 per tick, 4 ticks per point = $2.00 per point per contract
At ~$25,000 price, 10 points = 0.04% move, 15 points = 0.06% move
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


def synthesize_volume(df: pd.DataFrame, method="range") -> pd.DataFrame:
    df = df.copy()
    if method == "range":
        df["Volume"] = (df["High"] - df["Low"]).clip(lower=0.25)
    elif method == "range_weighted":
        df["Volume"] = ((df["High"] - df["Low"]) * (df["Close"] - df["Open"]).abs()).clip(lower=0.25)
    return df


# ---------------------------------------------------------------------------
# Scalper backtest engine with TP/SL (point-based)
# ---------------------------------------------------------------------------

def run_scalper_backtest(df: pd.DataFrame, config: BacktestConfig,
                         tp_points: float, sl_points: float) -> dict:
    """
    Backtest with fixed take-profit and stop-loss in price points.

    TP/SL are checked intrabar using High/Low:
    - Long: TP hit if High >= entry + tp_points, SL hit if Low <= entry - sl_points
    - Short: TP hit if Low <= entry - tp_points, SL hit if High >= entry + sl_points

    When both TP and SL could be hit on the same bar, SL is assumed to hit first
    (conservative / worst-case assumption).

    Signals detected on bar close, entry fills on next bar open (matching TV).
    """
    required = {"Open", "High", "Low", "Close", "long_entry", "short_entry"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

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
    trades: list[Trade] = []
    current_trade: Optional[Trade] = None

    pending_long = False
    pending_short = False

    equity_curve: list[dict] = []
    commission_rate = config.commission_pct / 100.0

    peak_equity = config.initial_capital
    max_intrabar_dd = 0.0
    max_intrabar_dd_pct = 0.0

    for i in range(len(df)):
        bar = df.iloc[i]
        bar_date = df.index[i]
        bar_in_range = start <= bar_date <= end

        # 1) FILL pending entry at this bar's Open
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
            pending_short = False

        # 2) Check TP/SL intrabar for open positions
        if position_side == "long" and position_qty > 0:
            tp_price = position_entry_price + tp_points
            sl_price = position_entry_price - sl_points

            hit_tp = bar["High"] >= tp_price
            hit_sl = bar["Low"] <= sl_price

            exit_price = None
            if hit_sl and hit_tp:
                # Both possible — assume SL hits first (worst case)
                exit_price = sl_price
            elif hit_sl:
                exit_price = sl_price
            elif hit_tp:
                exit_price = tp_price

            if exit_price is not None:
                trade_value = position_qty * exit_price
                exit_commission = trade_value * commission_rate
                gross_pnl = position_qty * (exit_price - position_entry_price)
                net_pnl = gross_pnl - current_trade.entry_commission - exit_commission
                cash += trade_value - exit_commission
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

        elif position_side == "short" and position_qty < 0:
            abs_qty = abs(position_qty)
            tp_price = position_entry_price - tp_points
            sl_price = position_entry_price + sl_points

            hit_tp = bar["Low"] <= tp_price
            hit_sl = bar["High"] >= sl_price

            exit_price = None
            if hit_sl and hit_tp:
                exit_price = sl_price
            elif hit_sl:
                exit_price = sl_price
            elif hit_tp:
                exit_price = tp_price

            if exit_price is not None:
                trade_value = abs_qty * exit_price
                exit_commission = trade_value * commission_rate
                gross_pnl = abs_qty * (position_entry_price - exit_price)
                net_pnl = gross_pnl - current_trade.entry_commission - exit_commission
                cash = cash + gross_pnl - exit_commission
                equity = cash

                current_trade.exit_date = bar_date
                current_trade.exit_price = exit_price
                current_trade.pnl = net_pnl
                entry_value = abs_qty * current_trade.entry_price
                current_trade.pnl_pct = (net_pnl / entry_value) * 100
                current_trade.exit_commission = exit_commission
                trades.append(current_trade)
                current_trade = None
                position_qty = 0.0
                position_entry_price = 0.0
                position_side = ""

        # 2a) Intrabar drawdown check
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

        if bar_in_range and position_qty == 0 and equity > peak_equity:
            peak_equity = equity

        # 3) Detect signals at Close (only if flat — no pyramiding)
        pending_long = False
        pending_short = False

        if bar_in_range and position_qty == 0:
            if bar["long_entry"]:
                pending_long = True
            elif bar["short_entry"]:
                pending_short = True

    # Record any open position at end
    if current_trade is not None:
        trades.append(current_trade)

    equity_df = pd.DataFrame(equity_curve)
    kpis = compute_kpis(trades, equity_df, config,
                        max_intrabar_dd, max_intrabar_dd_pct)
    kpis["actual_start_date"] = str(start.date())
    kpis["actual_end_date"] = str(end.date())
    return kpis


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, rsi_len=11, rsi_buy=60, rsi_sell=40,
                     sm_index=10, sm_flow=8, sm_norm=150, sm_ema=255,
                     sm_thr=0.0, vol_method="range") -> pd.DataFrame:
    df = synthesize_volume(df, method=vol_method)
    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index, sm_flow, sm_norm, sm_ema)

    rsi_flips_blue = (df["rsi"] > rsi_buy) & (df["rsi"].shift(1) <= rsi_buy)
    rsi_flips_purple = (df["rsi"] < rsi_sell) & (df["rsi"].shift(1) >= rsi_sell)
    sm_green = df["smart_money"] > sm_thr
    sm_red = df["smart_money"] < -sm_thr

    df["long_entry"] = sm_green & rsi_flips_blue
    df["short_entry"] = sm_red & rsi_flips_purple
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

    # User's settings
    user = dict(rsi_len=11, sm_index=10, sm_flow=8, sm_norm=150, sm_ema=255)

    # TP/SL combinations to test (in MNQ points)
    tp_sl_combos = [
        # (TP points, SL points, description)
        (10, 5,  "TP 10 / SL 5  (2:1)"),
        (10, 7,  "TP 10 / SL 7  (1.4:1)"),
        (10, 10, "TP 10 / SL 10 (1:1)"),
        (12, 5,  "TP 12 / SL 5  (2.4:1)"),
        (12, 8,  "TP 12 / SL 8  (1.5:1)"),
        (15, 5,  "TP 15 / SL 5  (3:1)"),
        (15, 7,  "TP 15 / SL 7  (2.1:1)"),
        (15, 10, "TP 15 / SL 10 (1.5:1)"),
        (15, 15, "TP 15 / SL 15 (1:1)"),
        (20, 7,  "TP 20 / SL 7  (2.9:1)"),
        (20, 10, "TP 20 / SL 10 (2:1)"),
        (20, 15, "TP 20 / SL 15 (1.3:1)"),
        (8,  5,  "TP 8  / SL 5  (1.6:1)"),
        (8,  3,  "TP 8  / SL 3  (2.7:1)"),
    ]

    # Also test RSI 65/35 (tighter entry)
    rsi_variants = [
        ("RSI 60/40", dict(rsi_buy=60, rsi_sell=40)),
        ("RSI 65/35", dict(rsi_buy=65, rsi_sell=35)),
    ]

    results = []

    for rsi_name, rsi_params in rsi_variants:
        merged = {**user, **rsi_params}
        df_sig = generate_signals(df.copy(), **merged)
        n_long = df_sig["long_entry"].sum()
        n_short = df_sig["short_entry"].sum()
        print(f"\n{'='*70}")
        print(f"  {rsi_name}: {n_long} long signals, {n_short} short signals")
        print(f"{'='*70}")

        for tp, sl, desc in tp_sl_combos:
            name = f"{rsi_name}, {desc}"
            try:
                kpis = run_scalper_backtest(df_sig, config, tp_points=tp, sl_points=sl)
                results.append({"name": name, "kpis": kpis, "tp": tp, "sl": sl,
                                "rsi": rsi_name})
                n_trades = kpis.get("total_trades", 0)
                if n_trades > 0:
                    pf = kpis["profit_factor"]
                    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
                    print(f"  {desc:<25} | Trades: {n_trades:>3} | "
                          f"Win: {kpis['win_rate']:>5.1f}% | PF: {pf_str:>6} | "
                          f"Net: ${kpis['net_profit']:>8,.2f} ({kpis['net_profit_pct']:>5.2f}%) | "
                          f"MaxDD: {kpis['max_drawdown_pct']:>6.2f}%")
                else:
                    print(f"  {desc:<25} | No trades")
            except Exception as e:
                print(f"  {desc:<25} | ERROR: {e}")
                results.append({"name": name, "kpis": None})

    # === COMPARISON TABLE ===
    print("\n\n" + "=" * 145)
    print("  SCALPER STRATEGY COMPARISON — MNQ 1-min, User Settings (RSI 11, SM 10/8/150/255)")
    print("=" * 145)
    header = (f"  {'Version':<40} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} "
              f"{'PF':>7} {'MaxDD%':>8} {'Sharpe':>7} {'Avg$':>8} {'TP':>4} {'SL':>4}")
    print(header)
    print("  " + "-" * 137)

    best_score = -999
    best_version = None
    ranked = []

    for r in results:
        k = r["kpis"]
        if k is None or "error" in k:
            continue

        pf = k["profit_factor"]
        pf_display = f"{pf:>7.2f}" if pf != float("inf") else "    inf"
        n_trades = k["total_trades"]
        if n_trades == 0:
            continue

        line = (f"  {r['name']:<40} "
                f"${k['net_profit']:>8,.2f} "
                f"{k['net_profit_pct']:>6.2f}% "
                f"{n_trades:>5d} "
                f"{k['win_rate']:>6.2f}% "
                f"{pf_display} "
                f"{k['max_drawdown_pct']:>7.2f}% "
                f"{k['sharpe_ratio']:>7.3f} "
                f"${k['avg_trade']:>6,.2f} "
                f"{r['tp']:>4} "
                f"{r['sl']:>4}")
        print(line)

        # Score
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
        if score > best_score:
            best_score = score
            best_version = r

    print("=" * 145)

    ranked.sort(key=lambda x: x[0], reverse=True)
    print("\n  TOP 5 SCALPER STRATEGIES:")
    for i, (score, r) in enumerate(ranked[:5], 1):
        k = r["kpis"]
        if k is None:
            continue
        pf = k["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(f"  #{i}: {r['name']}")
        print(f"      Net: ${k['net_profit']:,.2f} ({k['net_profit_pct']:.2f}%) | "
              f"Trades: {k['total_trades']} | Win: {k['win_rate']:.1f}% | "
              f"PF: {pf_str} | MaxDD: {k['max_drawdown_pct']:.2f}% | "
              f"Sharpe: {k['sharpe_ratio']:.3f} | Score: {score:.2f}")

    if best_version and best_version["kpis"]:
        print(f"\n\n{'='*60}")
        print(f"  CHAMPION: {best_version['name']}")
        print(f"{'='*60}")
        print_kpis(best_version["kpis"])
        print_trades(best_version["kpis"]["trades"])


if __name__ == "__main__":
    main()
