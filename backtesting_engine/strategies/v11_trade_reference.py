"""
v11 Trade Reference
====================
Run the v11 params SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50
on 1-min Databento data and output a full trade list for
cross-referencing against TradingView.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_databento_1min, resample_to_5min,
    compute_smart_money, compute_rsi, map_5min_rsi_to_1min,
    run_backtest_v10, score_trades,
)


def main():
    print("=" * 100)
    print("v11 TRADE REFERENCE -- SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50")
    print("=" * 100)

    # Load data
    df_1m = load_databento_1min('MNQ')
    print(f"\n1-min: {len(df_1m)} bars, {df_1m.index[0].date()} to {df_1m.index[-1].date()}")

    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    volumes = df_1m['Volume'].values
    times = df_1m.index.values

    # Compute SM with v11 params
    print("Computing SM with params index=10, flow=12, norm=200, ema=100...")
    sm = compute_smart_money(closes, volumes, 10, 12, 200, 100)
    print(f"  SM range: [{sm.min():.4f}, {sm.max():.4f}], mean={sm.mean():.4f}")

    # 5-min RSI(8) mapped to 1-min
    df_5m = resample_to_5min(df_1m)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=8)

    # Run backtest
    print("Running backtest...")
    trades = run_backtest_v10(
        opens, highs, lows, closes, sm, rsi_curr, times,
        rsi_buy=60, rsi_sell=40, sm_threshold=0.0,
        cooldown_bars=20, max_loss_pts=50,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )

    sc = score_trades(trades)
    print(f"\nTOTAL: {sc['count']} trades, WR {sc['win_rate']}%, PF {sc['pf']}, "
          f"${sc['net_dollar']:+.2f}, DD ${sc['max_dd_dollar']:.2f}")
    print(f"Exits: {sc['exits']}")

    # Full trade list
    print(f"\n{'#':>3}  {'Side':>5}  {'Entry Time':>16}  {'Exit Time':>16}  "
          f"{'Entry':>10}  {'Exit':>10}  {'Pts':>7}  {'Net$':>8}  {'Exit Type':>10}")
    print("-" * 100)

    cum_pnl = 0.0
    for i, t in enumerate(trades):
        entry_ts = pd.Timestamp(t['entry_time'])
        exit_ts = pd.Timestamp(t['exit_time'])
        net_pts = t['pts'] - 0.52  # commission per side
        net_dollar = net_pts * 2.0
        cum_pnl += net_dollar
        print(f"{i+1:>3}  {t['side']:>5}  {entry_ts.strftime('%m/%d %H:%M'):>16}  "
              f"{exit_ts.strftime('%m/%d %H:%M'):>16}  "
              f"{t['entry']:>10.2f}  {t['exit']:>10.2f}  {t['pts']:>+7.2f}  "
              f"${net_dollar:>+7.2f}  {t['result']:>10}")

    print("-" * 100)
    print(f"{'':>3}  {'':>5}  {'':>16}  {'':>16}  {'':>10}  {'':>10}  "
          f"{'':>7}  ${cum_pnl:>+7.2f}  {'TOTAL':>10}")

    # Also print the Jan 19 - Feb 13 subset for direct comparison with v9.4 TV range
    print(f"\n\n{'='*100}")
    print("SUBSET: Jan 19 - Feb 13 (matching v9.4 TradingView date range)")
    print(f"{'='*100}")

    start = pd.Timestamp("2026-01-19").date()
    end = pd.Timestamp("2026-02-14").date()
    subset = [t for t in trades
              if start <= pd.Timestamp(t['entry_time']).date() < end]
    sc_sub = score_trades(subset)
    if sc_sub:
        print(f"\n{sc_sub['count']} trades, WR {sc_sub['win_rate']}%, PF {sc_sub['pf']}, "
              f"${sc_sub['net_dollar']:+.2f}, DD ${sc_sub['max_dd_dollar']:.2f}")

    print(f"\n{'#':>3}  {'Side':>5}  {'Entry Time':>16}  {'Exit Time':>16}  "
          f"{'Entry':>10}  {'Exit':>10}  {'Pts':>7}  {'Net$':>8}  {'Exit Type':>10}")
    print("-" * 100)

    cum_sub = 0.0
    for i, t in enumerate(subset):
        entry_ts = pd.Timestamp(t['entry_time'])
        exit_ts = pd.Timestamp(t['exit_time'])
        net_pts = t['pts'] - 0.52
        net_dollar = net_pts * 2.0
        cum_sub += net_dollar
        print(f"{i+1:>3}  {t['side']:>5}  {entry_ts.strftime('%m/%d %H:%M'):>16}  "
              f"{exit_ts.strftime('%m/%d %H:%M'):>16}  "
              f"{t['entry']:>10.2f}  {t['exit']:>10.2f}  {t['pts']:>+7.2f}  "
              f"${net_dollar:>+7.2f}  {t['result']:>10}")

    print("-" * 100)
    print(f"{'':>3}  {'':>5}  {'':>16}  {'':>16}  {'':>10}  {'':>10}  "
          f"{'':>7}  ${cum_sub:>+7.2f}  {'TOTAL':>10}")

    # Print SM values at a few key points for manual spot-checking
    print(f"\n\n{'='*100}")
    print("SM SPOT-CHECK VALUES (for verifying Pine SM matches Python)")
    print(f"{'='*100}")
    print(f"\nSM values at first 5 trade entries:")
    for i, t in enumerate(trades[:5]):
        entry_ts = pd.Timestamp(t['entry_time'])
        # Find the bar index
        idx = np.searchsorted(times, np.datetime64(entry_ts))
        if idx > 0:
            idx -= 1
        sm_val = sm[idx] if idx < len(sm) else 0
        rsi_val = rsi_curr[idx] if idx < len(rsi_curr) else 0
        print(f"  Trade {i+1}: {entry_ts.strftime('%m/%d %H:%M')} | SM[i-1]={sm_val:+.4f} | "
              f"RSI5m={rsi_val:.2f} | side={t['side']}")

    print(f"\nSM values at specific timestamps (for Pine comparison):")
    check_times = [
        "2026-01-27 15:30:00",
        "2026-01-28 16:00:00",
        "2026-01-29 15:00:00",
        "2026-02-03 16:30:00",
        "2026-02-10 15:00:00",
    ]
    for ct in check_times:
        ts = pd.Timestamp(ct)
        idx = np.searchsorted(times, np.datetime64(ts))
        if idx < len(sm):
            print(f"  {ct}: SM={sm[idx]:+.6f}, RSI5m={rsi_curr[idx]:.2f}, "
                  f"Close={closes[idx]:.2f}")


if __name__ == "__main__":
    main()
