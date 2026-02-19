"""
Smart Money + RSI Scalper V3 — Uses REAL TradingView indicator data.

KEY DIFFERENCE from V1/V2: Instead of synthesizing volume and computing SM
indicator from scratch, this reads the actual TradingView SM Net Index and RSI
values exported directly from the chart. This means the indicator values match
EXACTLY what you see on screen.

CSV columns (from TradingView export with indicators):
  0: time, 1: open, 2: high, 3: low, 4: close
  5: Plot (RSI)  [6,7 are duplicates]
  8: Zero Line
 16: Net Buy Line   (= net_index when net_index > 0, else empty)
 22: Net Sell Line  (= net_index when net_index < 0, else empty)
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
    """Load TradingView CSV that includes SM + RSI indicator columns."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    filepath = data_dir / filename
    df = pd.read_csv(filepath)

    # Rename columns by position (headers have duplicate "Plot" names)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time",
        cols[1]: "Open",
        cols[2]: "High",
        cols[3]: "Low",
        cols[4]: "Close",
        cols[5]: "RSI",       # First "Plot" = RSI
        cols[16]: "SM_Buy",   # Net Buy Line
        cols[22]: "SM_Sell",  # Net Sell Line
    })

    # Parse timestamp
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")

    # Build net_index from the two columns
    sm_buy = pd.to_numeric(df["SM_Buy"], errors="coerce").fillna(0.0)
    sm_sell = pd.to_numeric(df["SM_Sell"], errors="coerce").fillna(0.0)
    # SM_Buy has positive values when SM > 0, SM_Sell has negative values when SM < 0
    df["net_index"] = sm_buy + sm_sell

    # RSI
    df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")

    # Drop last bar (may be incomplete, matching engine convention)
    df = df.iloc[:-1]

    return df[["Open", "High", "Low", "Close", "RSI", "net_index"]]


# ---------------------------------------------------------------------------
# Scalper engine with TP/SL + SM flip exit + max hold time
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

            # SM flip exit (at close price)
            if not exited and exit_on_sm_flip:
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
# Signal generation using REAL TV indicator data
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, rsi_buy=60, rsi_sell=40,
                     sm_thr=0.0, sm_bars_confirm=1) -> pd.DataFrame:
    """
    Generate entry signals from the real TradingView RSI & SM data.

    rsi_buy / rsi_sell: RSI crossover thresholds (blue flip / purple flip)
    sm_thr: minimum |net_index| to consider SM as green/red
    sm_bars_confirm: require SM to be green/red for N consecutive bars
    """
    df = df.copy()

    rsi = df["RSI"]
    sm = df["net_index"]

    # RSI crossover/crossunder detection
    rsi_flips_blue = (rsi > rsi_buy) & (rsi.shift(1) <= rsi_buy)
    rsi_flips_purple = (rsi < rsi_sell) & (rsi.shift(1) >= rsi_sell)

    # SM conditions
    sm_green = sm > sm_thr
    sm_red = sm < -sm_thr

    # SM multi-bar confirmation
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
    df = load_indicator_csv("CME_MINI_MNQ1!, 1_9f80c.csv")
    print(f"\nData: MNQ 1-min with REAL TV indicators")
    print(f"  Bars: {len(df)} | {df.index[0]} to {df.index[-1]}")
    print(f"  RSI range: {df['RSI'].min():.1f} to {df['RSI'].max():.1f}")
    print(f"  SM net_index range: {df['net_index'].min():.4f} to {df['net_index'].max():.4f}")
    sm_pos = (df['net_index'] > 0).sum()
    sm_neg = (df['net_index'] < 0).sum()
    print(f"  SM green bars: {sm_pos} | SM red bars: {sm_neg}")

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

    # Signal configurations — your TradingView settings: RSI 11, SM (10,8,150,255)
    # The RSI and SM values are already computed by TV, so we just vary the thresholds
    signal_configs = [
        # --- RSI 60/40 (standard blue/purple) ---
        ("RSI 60/40",
         dict(rsi_buy=60, rsi_sell=40, sm_thr=0.0, sm_bars_confirm=1)),
        ("RSI 60/40 SM>0.05",
         dict(rsi_buy=60, rsi_sell=40, sm_thr=0.05, sm_bars_confirm=1)),
        ("RSI 60/40 SM>0.1",
         dict(rsi_buy=60, rsi_sell=40, sm_thr=0.1, sm_bars_confirm=1)),
        ("RSI 60/40 SM>0.15",
         dict(rsi_buy=60, rsi_sell=40, sm_thr=0.15, sm_bars_confirm=1)),
        ("RSI 60/40 SM>0.2",
         dict(rsi_buy=60, rsi_sell=40, sm_thr=0.2, sm_bars_confirm=1)),
        ("RSI 60/40 3bar",
         dict(rsi_buy=60, rsi_sell=40, sm_thr=0.0, sm_bars_confirm=3)),
        ("RSI 60/40 5bar",
         dict(rsi_buy=60, rsi_sell=40, sm_thr=0.0, sm_bars_confirm=5)),
        ("RSI 60/40 SM>0.1 3bar",
         dict(rsi_buy=60, rsi_sell=40, sm_thr=0.1, sm_bars_confirm=3)),

        # --- RSI 65/35 (tighter filter) ---
        ("RSI 65/35",
         dict(rsi_buy=65, rsi_sell=35, sm_thr=0.0, sm_bars_confirm=1)),
        ("RSI 65/35 SM>0.05",
         dict(rsi_buy=65, rsi_sell=35, sm_thr=0.05, sm_bars_confirm=1)),
        ("RSI 65/35 SM>0.1",
         dict(rsi_buy=65, rsi_sell=35, sm_thr=0.1, sm_bars_confirm=1)),
        ("RSI 65/35 SM>0.15",
         dict(rsi_buy=65, rsi_sell=35, sm_thr=0.15, sm_bars_confirm=1)),
        ("RSI 65/35 SM>0.2",
         dict(rsi_buy=65, rsi_sell=35, sm_thr=0.2, sm_bars_confirm=1)),
        ("RSI 65/35 3bar",
         dict(rsi_buy=65, rsi_sell=35, sm_thr=0.0, sm_bars_confirm=3)),
        ("RSI 65/35 5bar",
         dict(rsi_buy=65, rsi_sell=35, sm_thr=0.0, sm_bars_confirm=5)),
        ("RSI 65/35 SM>0.1 3bar",
         dict(rsi_buy=65, rsi_sell=35, sm_thr=0.1, sm_bars_confirm=3)),

        # --- RSI 55/45 (wider net) ---
        ("RSI 55/45",
         dict(rsi_buy=55, rsi_sell=45, sm_thr=0.0, sm_bars_confirm=1)),
        ("RSI 55/45 SM>0.1",
         dict(rsi_buy=55, rsi_sell=45, sm_thr=0.1, sm_bars_confirm=1)),
        ("RSI 55/45 SM>0.2",
         dict(rsi_buy=55, rsi_sell=45, sm_thr=0.2, sm_bars_confirm=1)),

        # --- RSI 70/30 (very strict) ---
        ("RSI 70/30",
         dict(rsi_buy=70, rsi_sell=30, sm_thr=0.0, sm_bars_confirm=1)),
        ("RSI 70/30 SM>0.1",
         dict(rsi_buy=70, rsi_sell=30, sm_thr=0.1, sm_bars_confirm=1)),
        ("RSI 70/30 SM>0.2",
         dict(rsi_buy=70, rsi_sell=30, sm_thr=0.2, sm_bars_confirm=1)),
    ]

    # Exit configurations — scalper-focused with your 10-15pt targets
    exit_configs = [
        # Pure TP/SL
        ("TP10 SL5",            dict(tp_points=10, sl_points=5,  max_hold_bars=0, exit_on_sm_flip=False)),
        ("TP10 SL7",            dict(tp_points=10, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=False)),
        ("TP15 SL5",            dict(tp_points=15, sl_points=5,  max_hold_bars=0, exit_on_sm_flip=False)),
        ("TP15 SL7",            dict(tp_points=15, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=False)),
        ("TP15 SL10",           dict(tp_points=15, sl_points=10, max_hold_bars=0, exit_on_sm_flip=False)),

        # TP/SL + SM flip
        ("TP10 SL5 +SM",        dict(tp_points=10, sl_points=5,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP10 SL7 +SM",        dict(tp_points=10, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP15 SL5 +SM",        dict(tp_points=15, sl_points=5,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP15 SL7 +SM",        dict(tp_points=15, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP15 SL10 +SM",       dict(tp_points=15, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP20 SL7 +SM",        dict(tp_points=20, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("TP20 SL10 +SM",       dict(tp_points=20, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),

        # SM flip only (no TP, just SL protection)
        ("SM exit SL5",          dict(tp_points=9999, sl_points=5,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("SM exit SL7",          dict(tp_points=9999, sl_points=7,  max_hold_bars=0, exit_on_sm_flip=True)),
        ("SM exit SL10",         dict(tp_points=9999, sl_points=10, max_hold_bars=0, exit_on_sm_flip=True)),
        ("SM exit SL15",         dict(tp_points=9999, sl_points=15, max_hold_bars=0, exit_on_sm_flip=True)),

        # With max hold time (don't hold scalps too long)
        ("TP15 SL7 +SM 30bar",  dict(tp_points=15, sl_points=7,  max_hold_bars=30, exit_on_sm_flip=True)),
        ("TP15 SL7 +SM 60bar",  dict(tp_points=15, sl_points=7,  max_hold_bars=60, exit_on_sm_flip=True)),
        ("TP10 SL5 +SM 20bar",  dict(tp_points=10, sl_points=5,  max_hold_bars=20, exit_on_sm_flip=True)),
        ("TP15 SL10 +SM 60bar", dict(tp_points=15, sl_points=10, max_hold_bars=60, exit_on_sm_flip=True)),
    ]

    results = []

    for sig_name, sig_params in signal_configs:
        df_sig = generate_signals(df.copy(), **sig_params)
        n_long = df_sig["long_entry"].sum()
        n_short = df_sig["short_entry"].sum()

        if n_long == 0 and n_short == 0:
            print(f"  SKIP {sig_name}: 0 signals")
            continue

        for exit_name, exit_params in exit_configs:
            name = f"{sig_name} | {exit_name}"
            try:
                kpis = run_scalper(df_sig, config, **exit_params)
                results.append({"name": name, "kpis": kpis,
                                "sig": sig_name, "exit": exit_name})
            except Exception as e:
                pass

    # === COMPARISON TABLE ===
    print(f"\n\n{'='*165}")
    print("  SCALPER V3 — REAL TradingView Indicator Data (1-min MNQ)")
    print(f"{'='*165}")
    header = (f"  {'Signals':<28} {'Exit':<22} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} "
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

        line = (f"{marker} {r['sig']:<28} {r['exit']:<22} "
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
    print(f"\n  TOP 15 STRATEGIES:")
    print(f"  {'#':<4} {'Signals':<28} {'Exit':<22} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} {'PF':>7} {'MaxDD%':>8} {'Sharpe':>7}")
    print(f"  {'-'*120}")
    for i, (score, r) in enumerate(ranked[:15], 1):
        k = r["kpis"]
        pf = k["profit_factor"]
        pf_str = f"{pf:>7.2f}" if pf != float("inf") else "    inf"
        print(f"  {i:<4} {r['sig']:<28} {r['exit']:<22} "
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
        print_trades(k["trades"], max_trades=40)

    # Also show the best scalper-style result (with fixed TP, not SM-exit-only)
    scalp_ranked = [(s, r) for s, r in ranked if "SM exit" not in r["exit"]]
    if scalp_ranked and scalp_ranked[0] != ranked[0]:
        best_scalp = scalp_ranked[0][1]
        k2 = best_scalp["kpis"]
        print(f"\n\n{'='*70}")
        print(f"  BEST SCALPER (fixed TP): [{best_scalp['sig']}] + [{best_scalp['exit']}]")
        print(f"{'='*70}")
        print_kpis(k2)
        print_trades(k2["trades"], max_trades=40)


if __name__ == "__main__":
    main()
