"""
v11 MES Cross-Instrument Backtest
===================================
Tests v11 optimized params on MES 1-min Databento data and compares
against the v9.4 baseline.

Architecture (matches Pine Script exactly):
  1. Load MES 1-min Databento data (172,496 bars, Aug 17 2025 - Feb 13 2026)
  2. Compute SM on 1-min bars using compute_smart_money()
  3. Resample closes to 5-min, compute RSI on 5-min, map RSI curr+prev back to 1-min
  4. Run backtest engine on 1-min bars

Configs tested:
  v11: SM(10/12/200/100) RSI(8/60/40) CD=20 with SL=[0, 15, 20, 25, 50]
  v9.4 baseline: SM(20/12/400/255) RSI(10/55/45) CD=15 with SL=[0, 50]
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
)

# ---------------------------------------------------------------------------
# MES Instrument Config
# ---------------------------------------------------------------------------
COMMISSION_PER_SIDE = 0.52
DOLLAR_PER_PT = 5.0  # MES = $5 per point


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

def load_mes_databento():
    """Load MES 1-min Databento data. Returns DataFrame with Open, High, Low, Close, Volume."""
    filepath = (Path(__file__).resolve().parent.parent
                / "data" / "databento_MES_1min_2025-08-17_to_2026-02-13.csv")
    df = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    return result


# ---------------------------------------------------------------------------
# Run Config Helper
# ---------------------------------------------------------------------------

def run_config(opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
               rsi_buy, rsi_sell, cooldown, max_loss_pts, label=""):
    """Run a single config on pre-computed SM and RSI arrays. Returns (trades, score)."""
    trades = run_backtest_v10(
        opens, highs, lows, closes, sm, rsi_curr, times,
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=0.0, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )
    sc = score_trades(trades, commission_per_side=COMMISSION_PER_SIDE,
                      dollar_per_pt=DOLLAR_PER_PT)
    return trades, sc


# ---------------------------------------------------------------------------
# Printing Helpers
# ---------------------------------------------------------------------------

def print_result(sc, label="", show_exits=True):
    """Print a single result line."""
    if sc is None:
        print(f"  {label}: NO TRADES")
        return
    line = (f"  {label:<35}  {sc['count']:>4} trades  "
            f"WR {sc['win_rate']:>5.1f}%  PF {sc['pf']:>6.3f}  "
            f"Net ${sc['net_dollar']:>+9.2f}  MaxDD ${sc['max_dd_dollar']:>8.2f}  "
            f"Sharpe {sc['sharpe']:>6.3f}")
    if show_exits and sc.get('exits'):
        exits_str = "  ".join(f"{k}:{v}" for k, v in sorted(sc['exits'].items()))
        line += f"  | {exits_str}"
    print(line)


def monthly_breakdown(trades, label=""):
    """Print monthly performance table."""
    if not trades:
        print(f"  {label}: NO TRADES")
        return

    months = {}
    for t in trades:
        ts = pd.Timestamp(t['entry_time'])
        key = ts.strftime('%Y-%m')
        if key not in months:
            months[key] = []
        months[key].append(t)

    print(f"\n  Monthly Breakdown: {label}")
    print(f"  {'Month':>8}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Net$':>9}  {'MaxDD$':>8}")
    print(f"  {'-'*50}")

    for month_key in sorted(months.keys()):
        sc = score_trades(months[month_key], commission_per_side=COMMISSION_PER_SIDE,
                          dollar_per_pt=DOLLAR_PER_PT)
        if sc:
            print(f"  {month_key:>8}  {sc['count']:>6}  {sc['win_rate']:>5.1f}%  "
                  f"{sc['pf']:>6.3f}  ${sc['net_dollar']:>+8.2f}  ${sc['max_dd_dollar']:>7.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("MES v11 CROSS-INSTRUMENT BACKTEST")
    print("Data: Databento MES 1-min, 172,496 bars, Aug 17 2025 - Feb 13 2026")
    print(f"Commission: ${COMMISSION_PER_SIDE:.2f}/side, ${DOLLAR_PER_PT:.2f}/point")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading MES 1-min Databento data...")
    df_1m = load_mes_databento()
    print(f"  Loaded: {len(df_1m)} bars, {df_1m.index[0].date()} to {df_1m.index[-1].date()}")

    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    volumes = df_1m['Volume'].values
    times = df_1m.index.values

    # Resample to 5-min for RSI computation
    # Need a DataFrame with 'Close' and 'SM_Net' columns for resample_to_5min
    # We add a placeholder SM_Net since resample_to_5min expects it
    df_for_resample = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_resample['SM_Net'] = 0.0  # placeholder, not used for RSI
    df_5m = resample_to_5min(df_for_resample)
    fivemin_times = df_5m.index.values
    fivemin_closes = df_5m['Close'].values
    print(f"  5-min bars: {len(df_5m)} (for RSI computation)")

    # ------------------------------------------------------------------
    # Pre-compute SM arrays (one per unique SM param set)
    # ------------------------------------------------------------------
    print("\nComputing Smart Money indicators on 1-min bars...")

    # v11 SM params: (10, 12, 200, 100)
    print("  SM(10/12/200/100) for v11...")
    sm_v11 = compute_smart_money(closes, volumes, 10, 12, 200, 100)

    # v9.4 SM params: (20, 12, 400, 255)
    print("  SM(20/12/400/255) for v9.4 baseline...")
    sm_v94 = compute_smart_money(closes, volumes, 20, 12, 400, 255)

    # ------------------------------------------------------------------
    # Pre-compute RSI mappings (one per unique RSI length)
    # ------------------------------------------------------------------
    print("\nComputing 5-min RSI and mapping to 1-min bars...")

    # v11 RSI length=8
    print("  RSI(8) for v11...")
    rsi_curr_8, rsi_prev_8 = map_5min_rsi_to_1min(
        times, fivemin_times, fivemin_closes, rsi_len=8)

    # v9.4 RSI length=10
    print("  RSI(10) for v9.4 baseline...")
    rsi_curr_10, rsi_prev_10 = map_5min_rsi_to_1min(
        times, fivemin_times, fivemin_closes, rsi_len=10)

    # ==================================================================
    # SECTION 1: v11 Params with Multiple Stop Loss Values
    # ==================================================================
    print("\n" + "=" * 100)
    print("SECTION 1: v11 SM(10/12/200/100) RSI(8/60/40) CD=20 -- Stop Loss Sweep")
    print("=" * 100)

    v11_sl_values = [0, 15, 20, 25, 50]
    v11_results = {}

    for sl in v11_sl_values:
        sl_label = f"SL={sl}" if sl > 0 else "SL=OFF"
        label = f"v11 {sl_label}"
        trades, sc = run_config(
            opens, highs, lows, closes, sm_v11,
            rsi_curr_8, rsi_prev_8, times,
            rsi_buy=60, rsi_sell=40, cooldown=20,
            max_loss_pts=sl, label=label)
        v11_results[sl] = (trades, sc)
        print_result(sc, label)

    # ==================================================================
    # SECTION 2: v9.4 Baseline Params
    # ==================================================================
    print("\n" + "=" * 100)
    print("SECTION 2: v9.4 BASELINE SM(20/12/400/255) RSI(10/55/45) CD=15")
    print("=" * 100)

    v94_sl_values = [0, 50]
    v94_results = {}

    for sl in v94_sl_values:
        sl_label = f"SL={sl}" if sl > 0 else "SL=OFF"
        label = f"v9.4 {sl_label}"
        trades, sc = run_config(
            opens, highs, lows, closes, sm_v94,
            rsi_curr_10, rsi_prev_10, times,
            rsi_buy=55, rsi_sell=45, cooldown=15,
            max_loss_pts=sl, label=label)
        v94_results[sl] = (trades, sc)
        print_result(sc, label)

    # ==================================================================
    # SECTION 3: Monthly Breakdown
    # ==================================================================
    print("\n" + "=" * 100)
    print("SECTION 3: MONTHLY BREAKDOWN")
    print("=" * 100)

    # Find best v11 config by net dollar
    best_v11_sl = None
    best_v11_net = -1e9
    for sl, (trades, sc) in v11_results.items():
        if sc is not None and sc['net_dollar'] > best_v11_net:
            best_v11_net = sc['net_dollar']
            best_v11_sl = sl

    if best_v11_sl is not None:
        sl_label = f"SL={best_v11_sl}" if best_v11_sl > 0 else "SL=OFF"
        monthly_breakdown(
            v11_results[best_v11_sl][0],
            f"v11 BEST: SM(10/12/200/100) RSI(8/60/40) CD=20 {sl_label}")

    # Baseline monthly (SL=0 as the simpler reference, plus SL=50 if different)
    for sl in v94_sl_values:
        sl_label = f"SL={sl}" if sl > 0 else "SL=OFF"
        monthly_breakdown(
            v94_results[sl][0],
            f"v9.4 BASELINE: SM(20/12/400/255) RSI(10/55/45) CD=15 {sl_label}")

    # ==================================================================
    # SECTION 4: Comparison Table
    # ==================================================================
    print("\n" + "=" * 100)
    print("SECTION 4: COMPARISON -- v11 vs v9.4 Baseline on MES")
    print("=" * 100)

    # Build comparison rows
    rows = []

    # All v11 configs
    for sl in v11_sl_values:
        trades, sc = v11_results[sl]
        sl_label = f"SL={sl}" if sl > 0 else "SL=OFF"
        rows.append((f"v11 SM(10/12/200/100) {sl_label}", sc))

    # All v9.4 configs
    for sl in v94_sl_values:
        trades, sc = v94_results[sl]
        sl_label = f"SL={sl}" if sl > 0 else "SL=OFF"
        rows.append((f"v9.4 SM(20/12/400/255) {sl_label}", sc))

    print(f"\n  {'Config':<35} | {'Trades':>6} | {'WR%':>6} | {'PF':>7} | {'Net$':>10} | {'MaxDD$':>9} | {'Sharpe':>7}")
    print(f"  {'-'*93}")

    for label, sc in rows:
        if sc is None:
            print(f"  {label:<35} |    N/A |    N/A |     N/A |        N/A |       N/A |     N/A")
        else:
            print(f"  {label:<35} | {sc['count']:>6} | {sc['win_rate']:>5.1f}% | "
                  f"{sc['pf']:>7.3f} | ${sc['net_dollar']:>+9.2f} | "
                  f"${sc['max_dd_dollar']:>8.2f} | {sc['sharpe']:>7.3f}")

    # Highlight winner
    print()
    if best_v11_sl is not None:
        best_v11_sc = v11_results[best_v11_sl][1]
        # Best baseline is whichever SL has higher net dollar
        best_v94_sl = max(v94_results.keys(),
                         key=lambda s: v94_results[s][1]['net_dollar']
                         if v94_results[s][1] else -1e9)
        best_v94_sc = v94_results[best_v94_sl][1]

        if best_v11_sc and best_v94_sc:
            sl_label_v11 = f"SL={best_v11_sl}" if best_v11_sl > 0 else "SL=OFF"
            sl_label_v94 = f"SL={best_v94_sl}" if best_v94_sl > 0 else "SL=OFF"
            v11_better = best_v11_sc['net_dollar'] > best_v94_sc['net_dollar']
            winner = "v11" if v11_better else "v9.4"
            diff = best_v11_sc['net_dollar'] - best_v94_sc['net_dollar']
            print(f"  WINNER: {winner} by ${abs(diff):+.2f}")
            print(f"    v11 best:  {sl_label_v11} -> ${best_v11_sc['net_dollar']:+.2f}, "
                  f"PF {best_v11_sc['pf']}, WR {best_v11_sc['win_rate']}%")
            print(f"    v9.4 best: {sl_label_v94} -> ${best_v94_sc['net_dollar']:+.2f}, "
                  f"PF {best_v94_sc['pf']}, WR {best_v94_sc['win_rate']}%")

    print("\n" + "=" * 100)
    print("DONE -- v11 MES cross-instrument backtest complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
