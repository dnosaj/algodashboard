"""
v10 1-min Validation
====================
Run backtest on 1-MIN bars (matching Pine Script architecture) with
5-min RSI mapped back to each 1-min bar via prepare_backtest_arrays_1min().

Compares v9.4 baseline vs v10 (price structure exit) at 1-min resolution.
This should match TradingView results more closely than the old 5-min approach.

TradingView reference (Jan 19 - Feb 13, 2026):
  v9.4:  50 trades, PF 1.649, +$1,071.50, MaxDD $421.58
  v10:   53 trades, PF 1.429, +$365.38,   MaxDD $353.46
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_instrument_1min, resample_to_5min,
    prepare_backtest_arrays, prepare_backtest_arrays_1min,
    run_backtest_v10, run_v9_baseline, compute_rsi,
    score_trades, fmt_score, fmt_exits,
)

def run_v10_1min(arr, price_struct_bars, max_loss_pts=50):
    """Run v10 on 1-min arrays with mapped RSI."""
    return run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], arr['rsi'], arr['times'],
        rsi_buy=55, rsi_sell=45, sm_threshold=0.0,
        cooldown_bars=15, max_loss_pts=max_loss_pts,
        rsi_5m_curr=arr.get('rsi_5m_curr'),
        rsi_5m_prev=arr.get('rsi_5m_prev'),
        price_structure_exit=(price_struct_bars > 0),
        price_structure_bars=price_struct_bars if price_struct_bars > 0 else 3,
    )


def main():
    print("=" * 80)
    print("v10 1-MIN VALIDATION (matching Pine Script architecture)")
    print("=" * 80)

    # Load MNQ 1-min data
    df_1m = load_instrument_1min('MNQ')
    # Filter to Jan 19 - Feb 13 to match TradingView date range
    start = pd.Timestamp("2026-01-19")
    end = pd.Timestamp("2026-02-14")  # exclusive
    df_1m = df_1m[(df_1m.index >= start) & (df_1m.index < end)]
    print(f"\nData: {len(df_1m)} 1-min bars, {df_1m.index[0].date()} to {df_1m.index[-1].date()}")

    # Prepare 1-min arrays with 5-min RSI mapped back
    arr_1m = prepare_backtest_arrays_1min(df_1m, rsi_len=10)
    print(f"  RSI mapped from 5-min: rsi_5m_curr + rsi_5m_prev ({len(arr_1m['rsi_5m_curr'])} values)")

    # Also prepare the old 5-min path for comparison
    df_5m = resample_to_5min(df_1m)
    arr_5m = prepare_backtest_arrays(df_5m)
    print(f"  5-min bars: {len(df_5m)}")

    print("\n" + "=" * 80)
    print("SECTION 1: v9.4 BASELINE COMPARISON (1-min vs 5-min engine)")
    print("=" * 80)

    # v9.4 on 1-min (cooldown=15 bars = 15 min)
    trades_v94_1m = run_v9_baseline(arr_1m, cooldown=15, max_loss_pts=50)
    sc_v94_1m = score_trades(trades_v94_1m)
    print(f"\n  v9.4 on 1-MIN bars: {fmt_score(sc_v94_1m)}")
    if sc_v94_1m:
        print(f"    Exits: {fmt_exits(sc_v94_1m['exits'])}")

    # v9.4 on 5-min (cooldown=3 bars = 15 min)
    trades_v94_5m = run_v9_baseline(arr_5m, cooldown=3, max_loss_pts=50)
    sc_v94_5m = score_trades(trades_v94_5m)
    print(f"  v9.4 on 5-MIN bars: {fmt_score(sc_v94_5m)}")
    if sc_v94_5m:
        print(f"    Exits: {fmt_exits(sc_v94_5m['exits'])}")

    print(f"\n  TradingView ref:    50 trades, WR 60%, PF 1.649, +$1,071.50, MaxDD $421.58")

    # Show first 10 trades from 1-min engine for manual comparison
    print(f"\n  First 10 v9.4 trades (1-min engine):")
    for j, t in enumerate(trades_v94_1m[:10]):
        entry_ts = pd.Timestamp(t['entry_time'])
        exit_ts = pd.Timestamp(t['exit_time'])
        pnl = (t['pts'] - 0.52) * 2
        print(f"    #{j+1:2d} {t['side']:5s} {entry_ts.strftime('%m/%d %H:%M')} -> "
              f"{exit_ts.strftime('%m/%d %H:%M')} | entry={t['entry']:.2f} "
              f"exit={t['exit']:.2f} pts={t['pts']:+.2f} ${pnl:+.2f} [{t['result']}]")

    print("\n" + "=" * 80)
    print("SECTION 2: v10 PRICE STRUCTURE EXIT ON 1-MIN BARS")
    print("  Testing bars = [2, 3, 4, 5, 10, 15, 20, 25, 30]")
    print("=" * 80)

    # Test various price_struct_bars values on 1-min
    test_bars = [0, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    for psb in test_bars:
        trades = run_v10_1min(arr_1m, psb)
        sc = score_trades(trades)
        label = f"  struct={psb:2d}" if psb > 0 else "  struct=OFF"
        print(f"{label}:  {fmt_score(sc)}")
        if sc:
            print(f"            Exits: {fmt_exits(sc['exits'])}")

    print(f"\n  TradingView v10 ref (struct=3): 53 trades, PF 1.429, +$365.38, MaxDD $353.46")

    print("\n" + "=" * 80)
    print("SECTION 3: v9 BASELINE WITHOUT STOP (matching v9 original)")
    print("=" * 80)

    # v9 baseline (no stop, cooldown=15 on 1-min)
    trades_v9_1m = run_v9_baseline(arr_1m, cooldown=15, max_loss_pts=0)
    sc_v9_1m = score_trades(trades_v9_1m)
    print(f"\n  v9 (no stop) on 1-MIN: {fmt_score(sc_v9_1m)}")
    if sc_v9_1m:
        print(f"    Exits: {fmt_exits(sc_v9_1m['exits'])}")
    print(f"  TradingView v9 ref:    45-50 trades, PF ~2.027, +$1,196.70")

    print("\n" + "=" * 80)
    print("SECTION 4: 6-MONTH FULL DATA (1-min)")
    print("=" * 80)

    # Full 6-month range on 1-min
    df_1m_full = load_instrument_1min('MNQ')
    print(f"\n  Full data: {len(df_1m_full)} 1-min bars, {df_1m_full.index[0].date()} to {df_1m_full.index[-1].date()}")

    arr_full = prepare_backtest_arrays_1min(df_1m_full, rsi_len=10)

    # v9.4 baseline on 1-min full
    trades_base_full = run_v9_baseline(arr_full, cooldown=15, max_loss_pts=50)
    sc_base = score_trades(trades_base_full)
    print(f"  v9.4 baseline (1-min): {fmt_score(sc_base)}")
    if sc_base:
        print(f"    Exits: {fmt_exits(sc_base['exits'])}")

    # Price structure on 1-min full - sweep values
    print(f"\n  Price structure sweep on 6-month 1-min:")
    for psb in [3, 5, 10, 15, 20, 25, 30]:
        trades = run_v10_1min(arr_full, psb)
        sc = score_trades(trades)
        print(f"    struct={psb:2d}:  {fmt_score(sc)}")
        if sc:
            print(f"              Exits: {fmt_exits(sc['exits'])}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
