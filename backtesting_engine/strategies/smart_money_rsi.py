"""
Smart Money + RSI (Chop & Explode) Strategy for MNQ.

Entry rules (from user's chart analysis):
- BUY:  Smart Money indicator is GREEN (net_index > 0) AND RSI flips to BLUE (crosses above 60)
- SELL: Smart Money indicator is RED (net_index < 0) AND RSI flips to PURPLE (crosses below 40)

The Smart Money Volume Index uses PVI/NVI to distinguish institutional vs retail flow.
Since MNQ 1-min data lacks volume, we synthesize it from bar range * tick activity,
a standard futures proxy.

Multiple versions are tested with different parameters and filters.
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
# Indicator: RSI (Wilder's / RMA-based) — matches TradingView's ta.rsi()
# ---------------------------------------------------------------------------

def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """RSI matching TradingView's ta.rsi() using Wilder's smoothing (RMA)."""
    change = series.diff()
    up = change.clip(lower=0)
    dn = -change.clip(upper=0)
    up_smooth = calc_smma(up, length)
    dn_smooth = calc_smma(dn, length)
    rs = up_smooth / dn_smooth
    rsi = 100 - (100 / (1 + rs))
    # Handle edge cases
    rsi = rsi.where(dn_smooth != 0, 100.0)
    rsi = rsi.where(up_smooth != 0, 0.0)
    return rsi


# ---------------------------------------------------------------------------
# Indicator: PVI / NVI — matches TradingView's ta.pvi / ta.nvi
# ---------------------------------------------------------------------------

def calc_pvi_nvi(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Positive Volume Index (PVI) and Negative Volume Index (NVI).

    PVI: cumulates on bars where volume increases from prior bar.
    NVI: cumulates on bars where volume decreases from prior bar.

    TradingView formula:
      PVI starts at 1.0
      if volume > volume[1]: PVI += (close - close[1]) / close[1] * PVI[1]
      NVI starts at 1.0
      if volume < volume[1]: NVI += (close - close[1]) / close[1] * NVI[1]
    """
    close = df["Close"].values
    volume = df["Volume"].values

    pvi = np.full(len(close), np.nan)
    nvi = np.full(len(close), np.nan)
    pvi[0] = 1.0
    nvi[0] = 1.0

    for i in range(1, len(close)):
        price_change_pct = (close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0

        if volume[i] > volume[i-1]:
            pvi[i] = pvi[i-1] + price_change_pct * pvi[i-1]
        else:
            pvi[i] = pvi[i-1]

        if volume[i] < volume[i-1]:
            nvi[i] = nvi[i-1] + price_change_pct * nvi[i-1]
        else:
            nvi[i] = nvi[i-1]

    return pd.Series(pvi, index=df.index), pd.Series(nvi, index=df.index)


# ---------------------------------------------------------------------------
# Indicator: Smart Money Volume Index
# ---------------------------------------------------------------------------

def calc_smart_money(df: pd.DataFrame,
                     index_period: int = 25,
                     flow_period: int = 14,
                     norm_period: int = 500,
                     ema_len: int = 255) -> pd.Series:
    """
    Smart Money Volume Index [AlgoAlpha] — Net mode.

    Returns net_index: positive = smart money buying (green), negative = selling (red).
    """
    pvi, nvi = calc_pvi_nvi(df)

    # Dumb money = PVI deviation from its EMA
    dumb = pvi - calc_ema(pvi, ema_len)
    # Smart money = NVI deviation from its EMA
    smart = nvi - calc_ema(nvi, ema_len)

    # RSI on the volume flows
    drsi = calc_rsi(dumb, flow_period)
    srsi = calc_rsi(smart, flow_period)

    # Ratios: smart buying from dumb selling, and vice versa
    r_buy = srsi / drsi
    r_sell = (100 - srsi) / (100 - drsi)

    # Replace inf/nan from division by zero
    r_buy = r_buy.replace([np.inf, -np.inf], np.nan).fillna(0)
    r_sell = r_sell.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Summation over index period
    sums_buy = r_buy.rolling(window=index_period, min_periods=index_period).sum()
    sums_sell = r_sell.rolling(window=index_period, min_periods=index_period).sum()

    # Peak normalization
    combined_max = pd.concat([sums_buy, sums_sell], axis=1).max(axis=1)
    peak = combined_max.rolling(window=norm_period, min_periods=1).max()

    # Normalized indices
    index_buy = sums_buy / peak
    index_sell = sums_sell / peak

    # Net index: positive = smart money buying, negative = selling
    net_index = index_buy - index_sell

    return net_index


# ---------------------------------------------------------------------------
# Volume synthesis for MNQ (no volume in data)
# ---------------------------------------------------------------------------

def synthesize_volume(df: pd.DataFrame, method: str = "range") -> pd.DataFrame:
    """
    Add synthetic Volume column for MNQ data that lacks real volume.

    Methods:
    - "range": bar range (High - Low) as proxy — wider bars = more activity
    - "range_weighted": range * absolute return to emphasize directional bars
    - "tick_count": constant + noise (baseline volume with random variation)
    """
    df = df.copy()
    if method == "range":
        df["Volume"] = (df["High"] - df["Low"]).clip(lower=0.25)
    elif method == "range_weighted":
        bar_range = (df["High"] - df["Low"]).clip(lower=0.25)
        abs_return = (df["Close"] - df["Open"]).abs().clip(lower=0.25)
        df["Volume"] = bar_range * abs_return
    elif method == "tick_count":
        np.random.seed(42)
        base = 100.0
        noise = np.random.uniform(0.5, 1.5, len(df))
        df["Volume"] = base * noise
    else:
        raise ValueError(f"Unknown volume method: {method}")
    return df


# ---------------------------------------------------------------------------
# Strategy signal generators
# ---------------------------------------------------------------------------

def smart_money_rsi_signals_v1(df: pd.DataFrame,
                                rsi_len: int = 14,
                                sm_index_period: int = 25,
                                sm_flow_period: int = 14,
                                sm_norm_period: int = 500,
                                sm_ema_len: int = 255,
                                vol_method: str = "range") -> pd.DataFrame:
    """
    V1: Basic — Buy when SM green + RSI flips blue. Sell when SM red + RSI flips purple.

    "Flips to blue" = RSI crosses above 60 on close
    "Flips to purple" = RSI crosses below 40 on close
    """
    df = synthesize_volume(df, method=vol_method)

    # Calculate indicators
    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index_period, sm_flow_period,
                                          sm_norm_period, sm_ema_len)

    # RSI color transitions
    rsi_above_60 = df["rsi"] > 60
    rsi_below_40 = df["rsi"] < 40
    rsi_was_below_60 = df["rsi"].shift(1) <= 60
    rsi_was_above_40 = df["rsi"].shift(1) >= 40

    rsi_flips_blue = rsi_above_60 & rsi_was_below_60     # crosses above 60
    rsi_flips_purple = rsi_below_40 & rsi_was_above_40   # crosses below 40

    # Smart Money conditions
    sm_green = df["smart_money"] > 0
    sm_red = df["smart_money"] < 0

    # Signals
    df["long_entry"] = sm_green & rsi_flips_blue
    df["long_exit"] = sm_red & rsi_flips_purple   # exit long on sell signal
    df["short_entry"] = sm_red & rsi_flips_purple
    df["short_exit"] = sm_green & rsi_flips_blue  # exit short on buy signal

    # Also exit long when short entry fires, and vice versa (reversal)
    df["long_exit"] = df["long_exit"] | df["short_entry"]
    df["short_exit"] = df["short_exit"] | df["long_entry"]

    return df


def smart_money_rsi_signals_v2(df: pd.DataFrame,
                                rsi_len: int = 14,
                                sm_index_period: int = 25,
                                sm_flow_period: int = 14,
                                sm_norm_period: int = 500,
                                sm_ema_len: int = 255,
                                sm_threshold: float = 0.0,
                                rsi_buy_level: float = 60,
                                rsi_sell_level: float = 40,
                                vol_method: str = "range") -> pd.DataFrame:
    """
    V2: Relaxed exits — Same entry conditions but exits are independent.

    Exits:
    - Long exit: RSI drops below 50 (momentum fading) OR SM flips red
    - Short exit: RSI rises above 50 (momentum fading) OR SM flips green

    This lets winners run longer while cutting losers faster.
    """
    df = synthesize_volume(df, method=vol_method)

    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index_period, sm_flow_period,
                                          sm_norm_period, sm_ema_len)

    # Entry conditions (same as V1)
    rsi_flips_blue = (df["rsi"] > rsi_buy_level) & (df["rsi"].shift(1) <= rsi_buy_level)
    rsi_flips_purple = (df["rsi"] < rsi_sell_level) & (df["rsi"].shift(1) >= rsi_sell_level)

    sm_green = df["smart_money"] > sm_threshold
    sm_red = df["smart_money"] < -sm_threshold

    df["long_entry"] = sm_green & rsi_flips_blue
    df["short_entry"] = sm_red & rsi_flips_purple

    # Independent exits — more responsive
    rsi_crosses_below_50 = (df["rsi"] < 50) & (df["rsi"].shift(1) >= 50)
    rsi_crosses_above_50 = (df["rsi"] > 50) & (df["rsi"].shift(1) <= 50)

    sm_flips_red = (df["smart_money"] < 0) & (df["smart_money"].shift(1) >= 0)
    sm_flips_green = (df["smart_money"] > 0) & (df["smart_money"].shift(1) <= 0)

    df["long_exit"] = rsi_crosses_below_50 | sm_flips_red | df["short_entry"]
    df["short_exit"] = rsi_crosses_above_50 | sm_flips_green | df["long_entry"]

    return df


def smart_money_rsi_signals_v3(df: pd.DataFrame,
                                rsi_len: int = 14,
                                sm_index_period: int = 25,
                                sm_flow_period: int = 14,
                                sm_norm_period: int = 500,
                                sm_ema_len: int = 255,
                                trend_ema_len: int = 50,
                                vol_method: str = "range") -> pd.DataFrame:
    """
    V3: Trend-aligned — Only take trades in the direction of the EMA trend.

    - Buy only when price > EMA(50) (uptrend) + SM green + RSI flips blue
    - Sell only when price < EMA(50) (downtrend) + SM red + RSI flips purple
    - Exit when RSI returns to neutral zone (45-55) OR opposite signal fires
    """
    df = synthesize_volume(df, method=vol_method)

    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index_period, sm_flow_period,
                                          sm_norm_period, sm_ema_len)
    df["trend_ema"] = calc_ema(df["Close"], trend_ema_len)

    # Trend filter
    uptrend = df["Close"] > df["trend_ema"]
    downtrend = df["Close"] < df["trend_ema"]

    # Entry conditions
    rsi_flips_blue = (df["rsi"] > 60) & (df["rsi"].shift(1) <= 60)
    rsi_flips_purple = (df["rsi"] < 40) & (df["rsi"].shift(1) >= 40)

    sm_green = df["smart_money"] > 0
    sm_red = df["smart_money"] < 0

    df["long_entry"] = uptrend & sm_green & rsi_flips_blue
    df["short_entry"] = downtrend & sm_red & rsi_flips_purple

    # Exits: RSI returns to neutral or opposite signal
    rsi_neutral_from_high = (df["rsi"] < 45) & (df["rsi"].shift(1) >= 45)
    rsi_neutral_from_low = (df["rsi"] > 55) & (df["rsi"].shift(1) <= 55)

    df["long_exit"] = rsi_neutral_from_high | df["short_entry"]
    df["short_exit"] = rsi_neutral_from_low | df["long_entry"]

    return df


def smart_money_rsi_signals_v4(df: pd.DataFrame,
                                rsi_len: int = 14,
                                sm_index_period: int = 25,
                                sm_flow_period: int = 14,
                                sm_norm_period: int = 500,
                                sm_ema_len: int = 255,
                                vol_method: str = "range") -> pd.DataFrame:
    """
    V4: RSI-is-blue/purple state (not just flip) — Stay in trade while RSI color holds.

    - Buy when SM green AND RSI is blue (>60) — enter on the flip, stay while blue
    - Sell when SM red AND RSI is purple (<40) — enter on the flip, stay while purple
    - Exit long when RSI is no longer blue (drops below 60) regardless of SM
    - Exit short when RSI is no longer purple (rises above 40) regardless of SM
    """
    df = synthesize_volume(df, method=vol_method)

    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index_period, sm_flow_period,
                                          sm_norm_period, sm_ema_len)

    # Entry: flip moment
    rsi_flips_blue = (df["rsi"] > 60) & (df["rsi"].shift(1) <= 60)
    rsi_flips_purple = (df["rsi"] < 40) & (df["rsi"].shift(1) >= 40)

    sm_green = df["smart_money"] > 0
    sm_red = df["smart_money"] < 0

    df["long_entry"] = sm_green & rsi_flips_blue
    df["short_entry"] = sm_red & rsi_flips_purple

    # Exit: RSI loses its color
    rsi_drops_from_blue = (df["rsi"] < 60) & (df["rsi"].shift(1) >= 60)
    rsi_rises_from_purple = (df["rsi"] > 40) & (df["rsi"].shift(1) <= 40)

    df["long_exit"] = rsi_drops_from_blue | df["short_entry"]
    df["short_exit"] = rsi_rises_from_purple | df["long_entry"]

    return df


def smart_money_rsi_signals_v5(df: pd.DataFrame,
                                rsi_len: int = 14,
                                sm_index_period: int = 15,
                                sm_flow_period: int = 10,
                                sm_norm_period: int = 300,
                                sm_ema_len: int = 150,
                                vol_method: str = "range") -> pd.DataFrame:
    """
    V5: Tuned for 1-min timeframe — Shorter SM lookbacks for faster signals.

    Same logic as V1 but with parameters optimized for 1-minute MNQ bars.
    Shorter normalization and EMA periods to be more responsive on intraday data.
    """
    df = synthesize_volume(df, method=vol_method)

    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index_period, sm_flow_period,
                                          sm_norm_period, sm_ema_len)

    rsi_flips_blue = (df["rsi"] > 60) & (df["rsi"].shift(1) <= 60)
    rsi_flips_purple = (df["rsi"] < 40) & (df["rsi"].shift(1) >= 40)

    sm_green = df["smart_money"] > 0
    sm_red = df["smart_money"] < 0

    df["long_entry"] = sm_green & rsi_flips_blue
    df["long_exit"] = sm_red & rsi_flips_purple
    df["short_entry"] = sm_red & rsi_flips_purple
    df["short_exit"] = sm_green & rsi_flips_blue

    df["long_exit"] = df["long_exit"] | df["short_entry"]
    df["short_exit"] = df["short_exit"] | df["long_entry"]

    return df


def smart_money_rsi_signals_v6(df: pd.DataFrame,
                                rsi_len: int = 14,
                                sm_index_period: int = 25,
                                sm_flow_period: int = 14,
                                sm_norm_period: int = 500,
                                sm_ema_len: int = 255,
                                vol_method: str = "range_weighted") -> pd.DataFrame:
    """
    V6: Range-weighted volume synthesis — Better volume proxy for Smart Money.

    Uses range * |close - open| as volume proxy instead of just range.
    This gives more weight to bars with strong directional movement.
    Same entry/exit logic as V1.
    """
    df = synthesize_volume(df, method=vol_method)

    df["rsi"] = calc_rsi(df["Close"], rsi_len)
    df["smart_money"] = calc_smart_money(df, sm_index_period, sm_flow_period,
                                          sm_norm_period, sm_ema_len)

    rsi_flips_blue = (df["rsi"] > 60) & (df["rsi"].shift(1) <= 60)
    rsi_flips_purple = (df["rsi"] < 40) & (df["rsi"].shift(1) >= 40)

    sm_green = df["smart_money"] > 0
    sm_red = df["smart_money"] < 0

    df["long_entry"] = sm_green & rsi_flips_blue
    df["long_exit"] = sm_red & rsi_flips_purple
    df["short_entry"] = sm_red & rsi_flips_purple
    df["short_exit"] = sm_green & rsi_flips_blue

    df["long_exit"] = df["long_exit"] | df["short_entry"]
    df["short_exit"] = df["short_exit"] | df["long_entry"]

    return df


# ---------------------------------------------------------------------------
# Main: run all versions and compare
# ---------------------------------------------------------------------------

def run_version(name: str, df: pd.DataFrame, signal_fn, config: BacktestConfig, **kwargs) -> dict:
    """Run a single strategy version and return KPIs."""
    df_sig = signal_fn(df.copy(), **kwargs)

    # Count signals for diagnostics
    n_long = df_sig["long_entry"].sum()
    n_short = df_sig["short_entry"].sum()
    print(f"\n  Signals: {n_long} long entries, {n_short} short entries")

    if n_long == 0 and n_short == 0:
        print("  ⚠ No signals generated — skipping backtest")
        return {"name": name, "kpis": None}

    kpis = run_backtest_long_short(df_sig, config)
    return {"name": name, "kpis": kpis}


def main():
    # Load MNQ data
    df = load_tv_export("CME_MINI_MNQ1!, 1_e8c40.csv")
    print(f"\nData: MNQ 1-min | {len(df)} bars | {df.index[0]} to {df.index[-1]}")

    config = BacktestConfig(
        initial_capital=1000.0,
        commission_pct=0.1,
        slippage_ticks=0,
        qty_type="percent_of_equity",
        qty_value=100.0,
        pyramiding=1,
        start_date="2020-01-01",
        end_date="2069-12-31",
    )

    versions = [
        ("V1: Basic SM Green + RSI Blue/Purple", smart_money_rsi_signals_v1, {}),
        ("V2: Relaxed Exits (RSI 50 / SM flip)", smart_money_rsi_signals_v2, {}),
        ("V3: Trend-Aligned (EMA 50 filter)", smart_money_rsi_signals_v3, {}),
        ("V4: RSI State Hold (exit on color loss)", smart_money_rsi_signals_v4, {}),
        ("V5: Fast Params (tuned for 1-min)", smart_money_rsi_signals_v5, {}),
        ("V6: Range-Weighted Volume", smart_money_rsi_signals_v6, {}),
    ]

    results = []
    for name, fn, kwargs in versions:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        result = run_version(name, df, fn, config, **kwargs)
        results.append(result)
        if result["kpis"] and "error" not in result["kpis"]:
            print_kpis(result["kpis"])
            print_trades(result["kpis"]["trades"], max_trades=5)

    # === COMPARISON TABLE ===
    print("\n\n" + "=" * 120)
    print("  STRATEGY COMPARISON")
    print("=" * 120)
    header = f"  {'Version':<45} {'Net Profit':>12} {'Net %':>8} {'Trades':>7} {'Win%':>7} {'PF':>8} {'MaxDD%':>9} {'Sharpe':>8}"
    print(header)
    print("  " + "-" * 112)

    best_pf = -999
    best_version = None

    for r in results:
        k = r["kpis"]
        if k is None or "error" in k:
            print(f"  {r['name']:<45} {'NO TRADES':>12}")
            continue

        pf = k["profit_factor"]
        line = (f"  {r['name']:<45} "
                f"${k['net_profit']:>10,.2f} "
                f"{k['net_profit_pct']:>7.2f}% "
                f"{k['total_trades']:>7d} "
                f"{k['win_rate']:>6.2f}% "
                f"{pf:>8.3f} "
                f"{k['max_drawdown_pct']:>8.2f}% "
                f"{k['sharpe_ratio']:>8.3f}")
        print(line)

        # Score: profit factor, but penalize very few trades
        score = pf if k["total_trades"] >= 3 else pf * 0.5
        if score > best_pf and k["total_trades"] > 0:
            best_pf = score
            best_version = r

    print("=" * 120)

    if best_version and best_version["kpis"]:
        print(f"\n  🏆 BEST VERSION: {best_version['name']}")
        print(f"\n  Detailed results:")
        print_kpis(best_version["kpis"])
        print_trades(best_version["kpis"]["trades"])


if __name__ == "__main__":
    main()
