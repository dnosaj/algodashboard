"""
Smart Money + RSI Strategy — Round 4: User's Real Settings.

The user's actual TradingView settings on 1-min MNQ:
  - RSI Length: 11
  - SM Index Period: 10
  - SM Volume Flow Period: 8
  - SM Normalization Period: 150
  - SM EMA Length: 255 (default)
  - SM High Interest Threshold: 0.85

This round tests the user's exact params and variations around them,
across 1-min, 3-min, and 5-min timeframes, with different exit strategies.
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
# Unified signal generator
# ---------------------------------------------------------------------------

def make_signals(df: pd.DataFrame, timeframe: str = "1min",
                 rsi_len: int = 11, rsi_buy: float = 60, rsi_sell: float = 40,
                 sm_index: int = 10, sm_flow: int = 8, sm_norm: int = 150,
                 sm_ema: int = 255, sm_thr: float = 0.0,
                 vol_method: str = "range", trend_ema: int = 0,
                 exit_mode: str = "reversal") -> pd.DataFrame:
    """
    exit_mode:
      "reversal" — exit only when opposite entry fires
      "sm_flip" — exit when SM changes sign
      "rsi_neutral" — exit when RSI crosses 50
      "rsi_color_loss" — exit when RSI drops below 60 (long) or rises above 40 (short)
      "sm_flip_or_reversal" — exit on SM flip OR opposite entry (whichever first)
    """
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
    elif exit_mode == "sm_flip":
        sm_to_red = (df["smart_money"] < 0) & (df["smart_money"].shift(1) >= 0)
        sm_to_green = (df["smart_money"] > 0) & (df["smart_money"].shift(1) <= 0)
        df["long_exit"] = sm_to_red | df["short_entry"]
        df["short_exit"] = sm_to_green | df["long_entry"]
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
    elif exit_mode == "sm_flip_or_reversal":
        sm_to_red = (df["smart_money"] < 0) & (df["smart_money"].shift(1) >= 0)
        sm_to_green = (df["smart_money"] > 0) & (df["smart_money"].shift(1) <= 0)
        df["long_exit"] = sm_to_red | df["short_entry"]
        df["short_exit"] = sm_to_green | df["long_entry"]

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

    # USER'S EXACT PARAMS: RSI 11, SM(10, 8, 150, 255)
    user_sm = dict(rsi_len=11, sm_index=10, sm_flow=8, sm_norm=150, sm_ema=255)

    tests = [
        # === USER'S EXACT SETTINGS ===
        ("U1: User exact, 1m, reversal exit",
         dict(**user_sm, timeframe="1min", exit_mode="reversal")),
        ("U2: User exact, 1m, SM flip exit",
         dict(**user_sm, timeframe="1min", exit_mode="sm_flip")),
        ("U3: User exact, 1m, RSI neutral exit",
         dict(**user_sm, timeframe="1min", exit_mode="rsi_neutral")),
        ("U4: User exact, 1m, RSI color loss exit",
         dict(**user_sm, timeframe="1min", exit_mode="rsi_color_loss")),

        # === USER PARAMS ON HIGHER TIMEFRAMES ===
        ("U5: User params, 3m, reversal",
         dict(**user_sm, timeframe="3min", exit_mode="reversal")),
        ("U6: User params, 3m, SM flip exit",
         dict(**user_sm, timeframe="3min", exit_mode="sm_flip")),
        ("U7: User params, 5m, reversal",
         dict(**user_sm, timeframe="5min", exit_mode="reversal")),
        ("U8: User params, 5m, SM flip exit",
         dict(**user_sm, timeframe="5min", exit_mode="sm_flip")),

        # === USER PARAMS + SM THRESHOLD (filter weak SM signals) ===
        ("U9: User params, 1m, SM thr 0.05, reversal",
         dict(**user_sm, timeframe="1min", sm_thr=0.05, exit_mode="reversal")),
        ("U10: User params, 1m, SM thr 0.05, SM flip",
         dict(**user_sm, timeframe="1min", sm_thr=0.05, exit_mode="sm_flip")),
        ("U11: User params, 5m, SM thr 0.05, reversal",
         dict(**user_sm, timeframe="5min", sm_thr=0.05, exit_mode="reversal")),
        ("U12: User params, 5m, SM thr 0.05, SM flip",
         dict(**user_sm, timeframe="5min", sm_thr=0.05, exit_mode="sm_flip")),

        # === USER PARAMS + RSI LEVEL TWEAKS ===
        ("U13: User params, 1m, RSI 65/35, reversal",
         dict(**user_sm, timeframe="1min", rsi_buy=65, rsi_sell=35, exit_mode="reversal")),
        ("U14: User params, 1m, RSI 65/35, SM flip",
         dict(**user_sm, timeframe="1min", rsi_buy=65, rsi_sell=35, exit_mode="sm_flip")),
        ("U15: User params, 5m, RSI 65/35, SM flip",
         dict(**user_sm, timeframe="5min", rsi_buy=65, rsi_sell=35, exit_mode="sm_flip")),

        # === USER PARAMS + TREND FILTER ===
        ("U16: User params, 1m, EMA 50 trend, reversal",
         dict(**user_sm, timeframe="1min", trend_ema=50, exit_mode="reversal")),
        ("U17: User params, 1m, EMA 50 trend, SM flip",
         dict(**user_sm, timeframe="1min", trend_ema=50, exit_mode="sm_flip")),
        ("U18: User params, 5m, EMA 20 trend, SM flip",
         dict(**user_sm, timeframe="5min", trend_ema=20, exit_mode="sm_flip")),

        # === RANGE-WEIGHTED VOLUME ===
        ("U19: User params, 1m, range-wt vol, reversal",
         dict(**user_sm, timeframe="1min", vol_method="range_weighted", exit_mode="reversal")),
        ("U20: User params, 1m, range-wt vol, SM flip",
         dict(**user_sm, timeframe="1min", vol_method="range_weighted", exit_mode="sm_flip")),
        ("U21: User params, 5m, range-wt vol, SM flip",
         dict(**user_sm, timeframe="5min", vol_method="range_weighted", exit_mode="sm_flip")),

        # === BEST COMBOS ===
        ("U22: User, 1m, RSI 65/35, SM thr 0.05, SM flip",
         dict(**user_sm, timeframe="1min", rsi_buy=65, rsi_sell=35, sm_thr=0.05,
              exit_mode="sm_flip")),
        ("U23: User, 3m, RSI 65/35, SM flip",
         dict(**user_sm, timeframe="3min", rsi_buy=65, rsi_sell=35, exit_mode="sm_flip")),
        ("U24: User, 5m, RSI 65/35, SM thr 0.05, SM flip",
         dict(**user_sm, timeframe="5min", rsi_buy=65, rsi_sell=35, sm_thr=0.05,
              exit_mode="sm_flip")),

        # === COMPARE: Previous F11 winner with user's RSI len ===
        ("U25: F11 params but RSI 11 (hybrid), 5m, SM flip",
         dict(rsi_len=11, sm_index=15, sm_flow=10, sm_norm=300, sm_ema=150,
              timeframe="5min", exit_mode="sm_flip")),
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
    print("\n\n" + "=" * 145)
    print("  ROUND 4: USER'S SETTINGS (RSI 11, SM 10/8/150/255) — 0.005% commission")
    print("=" * 145)
    header = (f"  {'Version':<55} {'Net$':>10} {'Net%':>7} {'#Tr':>5} {'Win%':>7} "
              f"{'PF':>7} {'MaxDD%':>8} {'Sharpe':>7} {'Avg$':>8}")
    print(header)
    print("  " + "-" * 137)

    best_score = -999
    best_version = None
    ranked = []

    for r in results:
        k = r["kpis"]
        if k is None or "error" in k:
            print(f"  {r['name']:<55} {'NO TRADES / ERROR':>10}")
            continue

        pf = k["profit_factor"]
        pf_display = f"{pf:>7.3f}" if pf != float("inf") else "    inf"

        line = (f"  {r['name']:<55} "
                f"${k['net_profit']:>8,.2f} "
                f"{k['net_profit_pct']:>6.2f}% "
                f"{k['total_trades']:>5d} "
                f"{k['win_rate']:>6.2f}% "
                f"{pf_display} "
                f"{k['max_drawdown_pct']:>7.2f}% "
                f"{k['sharpe_ratio']:>7.3f} "
                f"${k['avg_trade']:>6,.2f}")
        print(line)

        n_trades = k["total_trades"]
        if n_trades >= 5:
            score = (min(pf, 10) * 2 + k["sharpe_ratio"] * 3 +
                     k["win_rate"] / 25 + min(n_trades, 30) / 15)
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
    print("\n  TOP 5 STRATEGIES:")
    for i, (score, r) in enumerate(ranked[:5], 1):
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
        print(f"  CHAMPION: {best_version['name']}")
        print(f"{'='*60}")
        print_kpis(best_version["kpis"])
        print_trades(best_version["kpis"]["trades"])


if __name__ == "__main__":
    main()
