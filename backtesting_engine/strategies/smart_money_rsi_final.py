"""
Smart Money + RSI Strategy — Final Round.

Best performers from Round 2:
- V15: 5-min, fast SM params (PF 2.655, 17 trades, $32.76 profit)
- V7/V11: 1-min basic with low commission (PF 1.328, 67 trades, $20.61)
- V14: 5-min, RSI 65/35, SM 0.05 (PF inf but only 2 trades)

This round explores variations around V15 (the best balanced result):
- Different SM parameter combinations on 5-min
- Different RSI thresholds
- EMA trend filter on 5-min
- Combined filters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine import (
    load_tv_export, BacktestConfig, calc_ema, calc_smma,
    run_backtest_long_short, print_kpis, print_trades,
)


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


def calc_smart_money(df: pd.DataFrame, index_period=25, flow_period=14,
                     norm_period=500, ema_len=255) -> pd.Series:
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


def resample_ohlc(df: pd.DataFrame, period: str) -> pd.DataFrame:
    resampled = df.resample(period).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
    }).dropna()
    if "Volume" in df.columns:
        resampled["Volume"] = df["Volume"].resample(period).sum()
    return resampled


# ---------------------------------------------------------------------------
# Strategy signal generators — Final round
# ---------------------------------------------------------------------------

def make_signals(df: pd.DataFrame, timeframe: str = "5min",
                 rsi_len: int = 14, rsi_buy: float = 60, rsi_sell: float = 40,
                 sm_index: int = 15, sm_flow: int = 10, sm_norm: int = 300,
                 sm_ema: int = 150, sm_thr: float = 0.0,
                 vol_method: str = "range", trend_ema: int = 0,
                 exit_mode: str = "reversal") -> pd.DataFrame:
    """
    Unified signal generator with all configurable parameters.

    exit_mode:
      "reversal" — exit only when opposite entry fires
      "rsi_neutral" — exit when RSI crosses 50
      "rsi_color_loss" — exit when RSI drops below 60 (long) or rises above 40 (short)
      "sm_flip" — exit when SM changes sign
    """
    # Resample if needed
    if timeframe != "1min":
        df = resample_ohlc(df, timeframe)

    df = synthesize_volume(df, method=vol_method)
    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index, sm_flow, sm_norm, sm_ema)

    # RSI flip conditions
    rsi_flips_blue = (df["rsi"] > rsi_buy) & (df["rsi"].shift(1) <= rsi_buy)
    rsi_flips_purple = (df["rsi"] < rsi_sell) & (df["rsi"].shift(1) >= rsi_sell)

    # SM conditions
    sm_green = df["smart_money"] > sm_thr
    sm_red = df["smart_money"] < -sm_thr

    # Trend filter
    if trend_ema > 0:
        df["trend"] = calc_ema(df["Close"], trend_ema)
        uptrend = df["Close"] > df["trend"]
        downtrend = df["Close"] < df["trend"]
        df["long_entry"] = uptrend & sm_green & rsi_flips_blue
        df["short_entry"] = downtrend & sm_red & rsi_flips_purple
    else:
        df["long_entry"] = sm_green & rsi_flips_blue
        df["short_entry"] = sm_red & rsi_flips_purple

    # Exit logic
    if exit_mode == "reversal":
        df["long_exit"] = df["short_entry"].copy()
        df["short_exit"] = df["long_entry"].copy()
    elif exit_mode == "rsi_neutral":
        rsi_below_50 = (df["rsi"] < 50) & (df["rsi"].shift(1) >= 50)
        rsi_above_50 = (df["rsi"] > 50) & (df["rsi"].shift(1) <= 50)
        df["long_exit"] = rsi_below_50 | df["short_entry"]
        df["short_exit"] = rsi_above_50 | df["long_entry"]
    elif exit_mode == "rsi_color_loss":
        rsi_loses_blue = (df["rsi"] < rsi_buy) & (df["rsi"].shift(1) >= rsi_buy)
        rsi_loses_purple = (df["rsi"] > rsi_sell) & (df["rsi"].shift(1) <= rsi_sell)
        df["long_exit"] = rsi_loses_blue | df["short_entry"]
        df["short_exit"] = rsi_loses_purple | df["long_entry"]
    elif exit_mode == "sm_flip":
        sm_to_red = (df["smart_money"] < 0) & (df["smart_money"].shift(1) >= 0)
        sm_to_green = (df["smart_money"] > 0) & (df["smart_money"].shift(1) <= 0)
        df["long_exit"] = sm_to_red | df["short_entry"]
        df["short_exit"] = sm_to_green | df["long_entry"]

    return df


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

    # Define all test configurations
    tests = [
        # --- V15 base (winner from round 2) ---
        ("F1: V15 base (5m, fast SM, reversal)", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150)),

        # --- Vary SM parameters around V15 ---
        ("F2: 5m, SM(20,12,400,200), reversal", dict(
            timeframe="5min", sm_index=20, sm_flow=12, sm_norm=400, sm_ema=200)),
        ("F3: 5m, SM(10,8,200,100), reversal", dict(
            timeframe="5min", sm_index=10, sm_flow=8, sm_norm=200, sm_ema=100)),
        ("F4: 5m, SM(15,14,300,150), reversal", dict(
            timeframe="5min", sm_index=15, sm_flow=14, sm_norm=300, sm_ema=150)),

        # --- V15 with SM threshold ---
        ("F5: V15 + SM thr 0.05, reversal", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150, sm_thr=0.05)),
        ("F6: V15 + SM thr 0.02, reversal", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150, sm_thr=0.02)),

        # --- V15 with different RSI levels ---
        ("F7: V15 + RSI 55/45, reversal", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            rsi_buy=55, rsi_sell=45)),
        ("F8: V15 + RSI 65/35, reversal", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            rsi_buy=65, rsi_sell=35)),

        # --- V15 with different exits ---
        ("F9: V15 + RSI neutral exit", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            exit_mode="rsi_neutral")),
        ("F10: V15 + RSI color loss exit", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            exit_mode="rsi_color_loss")),
        ("F11: V15 + SM flip exit", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            exit_mode="sm_flip")),

        # --- V15 with trend filter ---
        ("F12: V15 + EMA 20 trend filter", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            trend_ema=20)),
        ("F13: V15 + EMA 50 trend filter", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            trend_ema=50)),

        # --- V15 with range-weighted volume ---
        ("F14: V15 + range-weighted vol", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            vol_method="range_weighted")),

        # --- Best combo attempts ---
        ("F15: 5m, SM(15,10,300,150), RSI 65/35, SM thr 0.05", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            sm_thr=0.05, rsi_buy=65, rsi_sell=35)),
        ("F16: 5m, SM(10,8,200,100), RSI 55/45, reversal", dict(
            timeframe="5min", sm_index=10, sm_flow=8, sm_norm=200, sm_ema=100,
            rsi_buy=55, rsi_sell=45)),
        ("F17: 5m, SM(15,10,300,150), EMA20, SM flip exit", dict(
            timeframe="5min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
            trend_ema=20, exit_mode="sm_flip")),
        ("F18: 5m, SM(20,12,400,200), RSI 55/45, EMA 20", dict(
            timeframe="5min", sm_index=20, sm_flow=12, sm_norm=400, sm_ema=200,
            rsi_buy=55, rsi_sell=45, trend_ema=20)),

        # --- 3-min and 10-min timeframes ---
        ("F19: 3m, SM(15,10,300,150), reversal", dict(
            timeframe="3min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150)),
        ("F20: 10m, SM(15,10,300,150), reversal", dict(
            timeframe="10min", sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150)),
    ]

    results = []
    for name, params in tests:
        print(f"\n--- {name} ---")
        try:
            df_sig = make_signals(df.copy(), **params)
            n_long = df_sig["long_entry"].sum()
            n_short = df_sig["short_entry"].sum()
            print(f"  Signals: {n_long} long, {n_short} short on {len(df_sig)} bars")

            if n_long == 0 and n_short == 0:
                results.append({"name": name, "kpis": None})
                continue

            kpis = run_backtest_long_short(df_sig, config)
            results.append({"name": name, "kpis": kpis})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": name, "kpis": None})

    # === COMPARISON TABLE ===
    print("\n\n" + "=" * 140)
    print("  FINAL STRATEGY COMPARISON (MNQ 1-min data, 0.005% commission)")
    print("=" * 140)
    header = (f"  {'Version':<52} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} "
              f"{'PF':>7} {'MaxDD%':>8} {'Sharpe':>7} {'Avg$':>8} {'W/L':>6}")
    print(header)
    print("  " + "-" * 132)

    best_score = -999
    best_version = None
    ranked = []

    for r in results:
        k = r["kpis"]
        if k is None or "error" in k:
            print(f"  {r['name']:<52} {'NO TRADES / ERROR':>10}")
            continue

        pf = k["profit_factor"]
        pf_display = f"{pf:>7.3f}" if pf != float("inf") else "    inf"
        wl = k["avg_win_loss_ratio"]
        wl_display = f"{wl:>6.2f}" if wl != float("inf") else "   inf"

        line = (f"  {r['name']:<52} "
                f"${k['net_profit']:>8,.2f} "
                f"{k['net_profit_pct']:>6.2f}% "
                f"{k['total_trades']:>5d} "
                f"{k['win_rate']:>6.2f}% "
                f"{pf_display} "
                f"{k['max_drawdown_pct']:>7.2f}% "
                f"{k['sharpe_ratio']:>7.3f} "
                f"${k['avg_trade']:>6,.2f} "
                f"{wl_display}")
        print(line)

        # Scoring: balance PF, trade count, win rate, and Sharpe
        n_trades = k["total_trades"]
        if n_trades >= 5:
            score = (min(pf, 10) * 2 +        # profit factor (capped)
                     k["sharpe_ratio"] * 3 +   # risk-adjusted return
                     k["win_rate"] / 25 +      # win rate contribution
                     min(n_trades, 30) / 15)   # trade count (diminishing)
        elif n_trades >= 2:
            score = (min(pf, 10) + k["sharpe_ratio"]) * 0.5  # penalize low count
        else:
            score = -100  # too few trades

        ranked.append((score, r))
        if score > best_score:
            best_score = score
            best_version = r

    print("=" * 140)

    # Top 3
    ranked.sort(key=lambda x: x[0], reverse=True)
    print("\n  TOP 3 STRATEGIES:")
    for i, (score, r) in enumerate(ranked[:3], 1):
        k = r["kpis"]
        if k is None:
            continue
        pf = k["profit_factor"]
        pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
        print(f"  #{i}: {r['name']}")
        print(f"      Net: ${k['net_profit']:,.2f} ({k['net_profit_pct']:.2f}%) | "
              f"Trades: {k['total_trades']} | Win: {k['win_rate']:.1f}% | "
              f"PF: {pf_str} | MaxDD: {k['max_drawdown_pct']:.2f}% | "
              f"Sharpe: {k['sharpe_ratio']:.3f} | Score: {score:.2f}")

    if best_version and best_version["kpis"]:
        print(f"\n\n{'='*60}")
        print(f"  🏆 CHAMPION: {best_version['name']}")
        print(f"{'='*60}")
        print_kpis(best_version["kpis"])
        print_trades(best_version["kpis"]["trades"])


if __name__ == "__main__":
    main()
