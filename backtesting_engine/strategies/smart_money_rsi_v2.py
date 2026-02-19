"""
Smart Money + RSI Strategy for MNQ — Optimized Versions.

Round 2: Address issues from round 1:
1. Commission is killing profits — test with futures-realistic commission ($0.52/contract)
   which on MNQ at ~$25,000 is ~0.002%, NOT 0.1%
2. Too many signals on 1-min — use higher timeframe resampling (5min, 15min)
3. Need stronger confirmation — require SM to be firmly green/red, not just barely above 0
4. Hold trades longer — don't exit on every opposite signal

Versions:
- V7:  Original logic but with realistic MNQ commission (0.002%)
- V8:  5-min resampled bars with realistic commission
- V9:  15-min resampled bars with realistic commission
- V10: SM strength filter (SM > 0.1 for green, < -0.1 for red)
- V11: Hold until opposite entry signal (no independent exits)
- V12: Best combo — 5-min bars, SM strength filter, hold until reversal
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from engine import (
    load_tv_export, BacktestConfig, calc_ema, calc_smma,
    detect_crossover, detect_crossunder,
    run_backtest_long_short, print_kpis, print_trades,
)


# ---------------------------------------------------------------------------
# Indicators (same as v1 file)
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


def calc_smart_money(df: pd.DataFrame,
                     index_period: int = 25,
                     flow_period: int = 14,
                     norm_period: int = 500,
                     ema_len: int = 255) -> pd.Series:
    pvi, nvi = calc_pvi_nvi(df)
    dumb = pvi - calc_ema(pvi, ema_len)
    smart = nvi - calc_ema(nvi, ema_len)
    drsi = calc_rsi(dumb, flow_period)
    srsi = calc_rsi(smart, flow_period)
    r_buy = srsi / drsi
    r_sell = (100 - srsi) / (100 - drsi)
    r_buy = r_buy.replace([np.inf, -np.inf], np.nan).fillna(0)
    r_sell = r_sell.replace([np.inf, -np.inf], np.nan).fillna(0)
    sums_buy = r_buy.rolling(window=index_period, min_periods=index_period).sum()
    sums_sell = r_sell.rolling(window=index_period, min_periods=index_period).sum()
    combined_max = pd.concat([sums_buy, sums_sell], axis=1).max(axis=1)
    peak = combined_max.rolling(window=norm_period, min_periods=1).max()
    index_buy = sums_buy / peak
    index_sell = sums_sell / peak
    return index_buy - index_sell


def synthesize_volume(df: pd.DataFrame, method: str = "range") -> pd.DataFrame:
    df = df.copy()
    if method == "range":
        df["Volume"] = (df["High"] - df["Low"]).clip(lower=0.25)
    elif method == "range_weighted":
        df["Volume"] = ((df["High"] - df["Low"]) * (df["Close"] - df["Open"]).abs()).clip(lower=0.25)
    return df


# ---------------------------------------------------------------------------
# Timeframe resampling
# ---------------------------------------------------------------------------

def resample_ohlc(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Resample 1-min OHLC to higher timeframe."""
    resampled = df.resample(period).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()
    if "Volume" in df.columns:
        resampled["Volume"] = df["Volume"].resample(period).sum()
    return resampled


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def base_signals(df: pd.DataFrame, rsi_len: int = 14,
                 sm_index_period: int = 25, sm_flow_period: int = 14,
                 sm_norm_period: int = 500, sm_ema_len: int = 255,
                 sm_threshold: float = 0.0,
                 rsi_buy: float = 60, rsi_sell: float = 40,
                 vol_method: str = "range") -> pd.DataFrame:
    """Core signal generation — shared by all versions."""
    df = synthesize_volume(df, method=vol_method)
    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index_period, sm_flow_period,
                                          sm_norm_period, sm_ema_len)

    rsi_flips_blue = (df["rsi"] > rsi_buy) & (df["rsi"].shift(1) <= rsi_buy)
    rsi_flips_purple = (df["rsi"] < rsi_sell) & (df["rsi"].shift(1) >= rsi_sell)
    sm_green = df["smart_money"] > sm_threshold
    sm_red = df["smart_money"] < -sm_threshold

    df["long_entry"] = sm_green & rsi_flips_blue
    df["short_entry"] = sm_red & rsi_flips_purple
    # Default: reversal exits
    df["long_exit"] = df["short_entry"].copy()
    df["short_exit"] = df["long_entry"].copy()
    return df


def v7_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V7: Same as V1 but with realistic commission — test if commission was the issue."""
    return base_signals(df)


def v8_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V8: 5-min bars — fewer signals, bigger moves per trade."""
    df_5m = resample_ohlc(df, "5min")
    df_5m = base_signals(df_5m)
    return df_5m


def v9_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V9: 15-min bars — even fewer signals, hold trades longer."""
    df_15m = resample_ohlc(df, "15min")
    df_15m = base_signals(df_15m)
    return df_15m


def v10_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V10: SM strength filter — require SM > 0.1 or < -0.1 (not just above/below 0)."""
    return base_signals(df, sm_threshold=0.1)


def v11_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V11: Hold until opposite signal — no independent exit, only reversal."""
    df = base_signals(df)
    # Exits only fire when the opposite entry fires (already the default in base_signals)
    return df


def v12_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V12: 5-min + SM strength filter + reversal exits only."""
    df_5m = resample_ohlc(df, "5min")
    df_5m = base_signals(df_5m, sm_threshold=0.1)
    return df_5m


def v13_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V13: 15-min + SM strength filter."""
    df_15m = resample_ohlc(df, "15min")
    df_15m = base_signals(df_15m, sm_threshold=0.1)
    return df_15m


def v14_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V14: 5-min + wider RSI levels (65/35) + SM threshold 0.05."""
    df_5m = resample_ohlc(df, "5min")
    df_5m = base_signals(df_5m, sm_threshold=0.05, rsi_buy=65, rsi_sell=35)
    return df_5m


def v15_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V15: 5-min + shorter SM params (faster for intraday)."""
    df_5m = resample_ohlc(df, "5min")
    df_5m = base_signals(df_5m, sm_index_period=15, sm_flow_period=10,
                         sm_norm_period=300, sm_ema_len=150)
    return df_5m


def v16_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V16: 15-min + shorter SM params + SM threshold."""
    df_15m = resample_ohlc(df, "15min")
    df_15m = base_signals(df_15m, sm_index_period=15, sm_flow_period=10,
                          sm_norm_period=300, sm_ema_len=150, sm_threshold=0.05)
    return df_15m


def v17_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V17: 5-min + exit when RSI returns to neutral (50) instead of reversal only."""
    df_5m = resample_ohlc(df, "5min")
    df_5m = synthesize_volume(df_5m, method="range")
    df_5m["rsi"] = calc_rsi(df_5m["Close"], 14)
    df_5m["smart_money"] = calc_smart_money(df_5m)

    rsi_flips_blue = (df_5m["rsi"] > 60) & (df_5m["rsi"].shift(1) <= 60)
    rsi_flips_purple = (df_5m["rsi"] < 40) & (df_5m["rsi"].shift(1) >= 40)
    sm_green = df_5m["smart_money"] > 0
    sm_red = df_5m["smart_money"] < 0

    df_5m["long_entry"] = sm_green & rsi_flips_blue
    df_5m["short_entry"] = sm_red & rsi_flips_purple

    # Exit when RSI crosses back through 50
    rsi_cross_below_50 = (df_5m["rsi"] < 50) & (df_5m["rsi"].shift(1) >= 50)
    rsi_cross_above_50 = (df_5m["rsi"] > 50) & (df_5m["rsi"].shift(1) <= 50)

    df_5m["long_exit"] = rsi_cross_below_50 | df_5m["short_entry"]
    df_5m["short_exit"] = rsi_cross_above_50 | df_5m["long_entry"]

    return df_5m


def v18_signals(df: pd.DataFrame) -> pd.DataFrame:
    """V18: 30-min bars — even longer holding period, fewer whipsaws."""
    df_30m = resample_ohlc(df, "30min")
    df_30m = base_signals(df_30m)
    return df_30m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_version(name: str, df_or_sig: pd.DataFrame, signal_fn, config: BacktestConfig) -> dict:
    """Run a single strategy version."""
    df_sig = signal_fn(df_or_sig.copy())
    n_long = df_sig["long_entry"].sum()
    n_short = df_sig["short_entry"].sum()
    print(f"\n  Signals: {n_long} long entries, {n_short} short entries on {len(df_sig)} bars")

    if n_long == 0 and n_short == 0:
        print("  ⚠ No signals generated — skipping backtest")
        return {"name": name, "kpis": None}

    kpis = run_backtest_long_short(df_sig, config)
    return {"name": name, "kpis": kpis}


def main():
    df = load_tv_export("CME_MINI_MNQ1!, 1_e8c40.csv")
    print(f"\nData: MNQ 1-min | {len(df)} bars | {df.index[0]} to {df.index[-1]}")

    # Realistic MNQ commission: $0.52 round-trip per contract
    # At $25,000 * $0.50 multiplier = $12,500 notional, commission = 0.52/12500 ≈ 0.004%
    # But we're trading fractional in the engine, so let's use 0.005% to be conservative
    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.005,    # Realistic futures commission
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        pyramiding=1,
        start_date="2020-01-01",
        end_date="2069-12-31",
    )

    versions = [
        ("V7:  1-min, basic, low commission", v7_signals),
        ("V8:  5-min bars, basic", v8_signals),
        ("V9:  15-min bars, basic", v9_signals),
        ("V10: 1-min, SM threshold 0.1", v10_signals),
        ("V11: 1-min, reversal-only exits", v11_signals),
        ("V12: 5-min + SM threshold 0.1", v12_signals),
        ("V13: 15-min + SM threshold 0.1", v13_signals),
        ("V14: 5-min, RSI 65/35, SM 0.05", v14_signals),
        ("V15: 5-min, fast SM params", v15_signals),
        ("V16: 15-min, fast SM + threshold", v16_signals),
        ("V17: 5-min, exit at RSI 50", v17_signals),
        ("V18: 30-min bars, basic", v18_signals),
    ]

    results = []
    for name, fn in versions:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        result = run_version(name, df, fn, config)
        results.append(result)
        if result["kpis"] and "error" not in result["kpis"]:
            print_kpis(result["kpis"])

    # === COMPARISON TABLE ===
    print("\n\n" + "=" * 130)
    print("  STRATEGY COMPARISON — Round 2 (Realistic Commission: 0.005%)")
    print("=" * 130)
    header = f"  {'Version':<42} {'Net Profit':>12} {'Net %':>8} {'Trades':>7} {'Win%':>7} {'PF':>8} {'MaxDD%':>9} {'Sharpe':>8} {'AvgTrade':>10}"
    print(header)
    print("  " + "-" * 122)

    best_score = -999
    best_version = None

    for r in results:
        k = r["kpis"]
        if k is None or "error" in k:
            print(f"  {r['name']:<42} {'NO TRADES':>12}")
            continue

        pf = k["profit_factor"]
        line = (f"  {r['name']:<42} "
                f"${k['net_profit']:>10,.2f} "
                f"{k['net_profit_pct']:>7.2f}% "
                f"{k['total_trades']:>7d} "
                f"{k['win_rate']:>6.2f}% "
                f"{pf:>8.3f} "
                f"{k['max_drawdown_pct']:>8.2f}% "
                f"{k['sharpe_ratio']:>8.3f} "
                f"${k['avg_trade']:>8,.2f}")
        print(line)

        # Combined score: PF weighted by trade count and win rate
        if k["total_trades"] >= 3:
            score = pf * (1 + k["win_rate"] / 100)
        else:
            score = pf * 0.3
        if k["net_profit"] < 0:
            score *= 0.5  # Penalize losing strategies

        if score > best_score and k["total_trades"] > 0:
            best_score = score
            best_version = r

    print("=" * 130)

    if best_version and best_version["kpis"]:
        print(f"\n  🏆 BEST VERSION: {best_version['name']}")
        print(f"\n  Detailed results:")
        print_kpis(best_version["kpis"])
        print_trades(best_version["kpis"]["trades"])


if __name__ == "__main__":
    main()
