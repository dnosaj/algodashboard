"""
v11 Walk-Forward Validation on MNQ
====================================
Tests whether the v11 sweep-optimized params are overfit to MNQ.

Splits:
  In-Sample  (IS): Aug 17 - Nov 30, 2025 (3.5 months)
  Out-of-Sample (OOS): Dec 1, 2025 - Feb 12, 2026 (2.5 months)

The v11 params were optimized on the FULL 6-month period. If OOS
performance holds (PF > 1.0, degradation < 30%), the edge is likely
real and not just curve-fitting.

Also runs v9.4 baseline on both splits for comparison.

Architecture: 1-MIN bars, 5-min RSI mapped back (matches Pine exactly).
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
)

COMMISSION = 0.52
DOLLAR_PER_PT = 2.0  # MNQ


def load_mnq_databento():
    filepath = (Path(__file__).resolve().parent.parent
                / "data" / "databento_MNQ_1min_2025-08-17_to_2026-02-13.csv")
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


def run_on_slice(df_slice, sm_params, rsi_len, rsi_buy, rsi_sell,
                 cooldown, max_loss_pts, full_sm=None, full_times=None):
    """Run backtest on a date-sliced 1-min DataFrame.

    SM must be computed on FULL data for warmup, then sliced.
    RSI is resampled to 5-min from the slice (RSI warmup is short).
    """
    opens = df_slice['Open'].values
    highs = df_slice['High'].values
    lows = df_slice['Low'].values
    closes = df_slice['Close'].values
    times = df_slice.index.values

    # SM: use pre-computed full-period SM, sliced to this window
    if full_sm is not None and full_times is not None:
        # Find indices in full data that correspond to this slice
        mask = (full_times >= times[0]) & (full_times <= times[-1])
        sm = full_sm[mask]
    else:
        sm = compute_smart_money(closes, df_slice['Volume'].values, *sm_params)

    # RSI: resample slice to 5-min, compute RSI, map back
    df_for_5m = df_slice[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_5m['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_for_5m)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=rsi_len)

    trades = run_backtest_v10(
        opens, highs, lows, closes, sm, rsi_curr, times,
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=0.0, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )
    sc = score_trades(trades, commission_per_side=COMMISSION,
                      dollar_per_pt=DOLLAR_PER_PT)
    return trades, sc


def print_row(label, sc):
    if sc is None:
        print(f"  {label:<40}  NO TRADES")
        return
    print(f"  {label:<40}  {sc['count']:>4} trades  "
          f"WR {sc['win_rate']:>5.1f}%  PF {sc['pf']:>6.3f}  "
          f"Net ${sc['net_dollar']:>+9.2f}  MaxDD ${sc['max_dd_dollar']:>8.2f}  "
          f"Sharpe {sc['sharpe']:>6.3f}")


def main():
    print("=" * 100)
    print("v11 WALK-FORWARD VALIDATION -- MNQ 1-min")
    print("=" * 100)

    # Load full data
    print("\nLoading MNQ 1-min Databento data...")
    df_full = load_mnq_databento()
    print(f"  {len(df_full)} bars, {df_full.index[0].date()} to {df_full.index[-1].date()}")

    # Pre-compute SM on FULL data (needs warmup from beginning)
    closes_full = df_full['Close'].values
    volumes_full = df_full['Volume'].values
    times_full = df_full.index.values

    print("\nComputing SM on full data for warmup...")
    sm_v11_full = compute_smart_money(closes_full, volumes_full, 10, 12, 200, 100)
    sm_v94_full = compute_smart_money(closes_full, volumes_full, 20, 12, 400, 255)

    # Define splits
    splits = {
        'IS: Aug-Nov': (pd.Timestamp('2025-08-17'), pd.Timestamp('2025-12-01')),
        'OOS: Dec-Feb': (pd.Timestamp('2025-12-01'), pd.Timestamp('2026-02-13')),
    }

    # Also do a different split for robustness check
    splits_alt = {
        'IS: Aug-Oct': (pd.Timestamp('2025-08-17'), pd.Timestamp('2025-11-01')),
        'OOS: Nov-Feb': (pd.Timestamp('2025-11-01'), pd.Timestamp('2026-02-13')),
    }

    configs = {
        'v11 SM(10/12/200/100) RSI8 60/40 CD20 SL50': {
            'sm_params': (10, 12, 200, 100),
            'rsi_len': 8, 'rsi_buy': 60, 'rsi_sell': 40,
            'cooldown': 20, 'max_loss_pts': 50,
            'sm_full': sm_v11_full,
        },
        'v11 SM(10/12/200/100) RSI8 60/40 CD20 SL0': {
            'sm_params': (10, 12, 200, 100),
            'rsi_len': 8, 'rsi_buy': 60, 'rsi_sell': 40,
            'cooldown': 20, 'max_loss_pts': 0,
            'sm_full': sm_v11_full,
        },
        'v9.4 SM(20/12/400/255) RSI10 55/45 CD15 SL50': {
            'sm_params': (20, 12, 400, 255),
            'rsi_len': 10, 'rsi_buy': 55, 'rsi_sell': 45,
            'cooldown': 15, 'max_loss_pts': 50,
            'sm_full': sm_v94_full,
        },
        'v9.4 SM(20/12/400/255) RSI10 55/45 CD15 SL0': {
            'sm_params': (20, 12, 400, 255),
            'rsi_len': 10, 'rsi_buy': 55, 'rsi_sell': 45,
            'cooldown': 15, 'max_loss_pts': 0,
            'sm_full': sm_v94_full,
        },
    }

    # =====================================================================
    # Primary Split: Aug-Nov (IS) vs Dec-Feb (OOS)
    # =====================================================================
    print("\n" + "=" * 100)
    print("SPLIT 1: In-Sample Aug-Nov | Out-of-Sample Dec-Feb")
    print("=" * 100)

    results_1 = {}
    for config_name, cfg in configs.items():
        results_1[config_name] = {}
        for split_name, (start, end) in splits.items():
            df_slice = df_full[(df_full.index >= start) & (df_full.index < end)]
            trades, sc = run_on_slice(
                df_slice, cfg['sm_params'], cfg['rsi_len'],
                cfg['rsi_buy'], cfg['rsi_sell'], cfg['cooldown'],
                cfg['max_loss_pts'],
                full_sm=cfg['sm_full'], full_times=times_full)
            results_1[config_name][split_name] = (trades, sc)

    for config_name in configs:
        print(f"\n  {config_name}")
        for split_name in splits:
            _, sc = results_1[config_name][split_name]
            print_row(f"    {split_name}", sc)

        # Degradation analysis
        is_sc = results_1[config_name]['IS: Aug-Nov'][1]
        oos_sc = results_1[config_name]['OOS: Dec-Feb'][1]
        if is_sc and oos_sc and is_sc['pf'] > 0:
            degrad = (1 - oos_sc['pf'] / is_sc['pf']) * 100
            print(f"    -> OOS/IS PF ratio: {oos_sc['pf']/is_sc['pf']:.3f}  "
                  f"Degradation: {degrad:+.1f}%  "
                  f"{'PASS' if oos_sc['pf'] >= 1.0 and degrad < 30 else 'WATCH' if oos_sc['pf'] >= 1.0 else 'FAIL'}")

    # =====================================================================
    # Alternate Split: Aug-Oct (IS) vs Nov-Feb (OOS)
    # =====================================================================
    print("\n" + "=" * 100)
    print("SPLIT 2: In-Sample Aug-Oct | Out-of-Sample Nov-Feb")
    print("=" * 100)

    results_2 = {}
    for config_name, cfg in configs.items():
        results_2[config_name] = {}
        for split_name, (start, end) in splits_alt.items():
            df_slice = df_full[(df_full.index >= start) & (df_full.index < end)]
            trades, sc = run_on_slice(
                df_slice, cfg['sm_params'], cfg['rsi_len'],
                cfg['rsi_buy'], cfg['rsi_sell'], cfg['cooldown'],
                cfg['max_loss_pts'],
                full_sm=cfg['sm_full'], full_times=times_full)
            results_2[config_name][split_name] = (trades, sc)

    for config_name in configs:
        print(f"\n  {config_name}")
        for split_name in splits_alt:
            _, sc = results_2[config_name][split_name]
            print_row(f"    {split_name}", sc)

        is_sc = results_2[config_name]['IS: Aug-Oct'][1]
        oos_sc = results_2[config_name]['OOS: Nov-Feb'][1]
        if is_sc and oos_sc and is_sc['pf'] > 0:
            degrad = (1 - oos_sc['pf'] / is_sc['pf']) * 100
            print(f"    -> OOS/IS PF ratio: {oos_sc['pf']/is_sc['pf']:.3f}  "
                  f"Degradation: {degrad:+.1f}%  "
                  f"{'PASS' if oos_sc['pf'] >= 1.0 and degrad < 30 else 'WATCH' if oos_sc['pf'] >= 1.0 else 'FAIL'}")

    # =====================================================================
    # Leave-One-Month-Out Cross-Validation
    # =====================================================================
    print("\n" + "=" * 100)
    print("LEAVE-ONE-MONTH-OUT CROSS-VALIDATION")
    print("=" * 100)

    months = [
        ('Aug', pd.Timestamp('2025-08-17'), pd.Timestamp('2025-09-01')),
        ('Sep', pd.Timestamp('2025-09-01'), pd.Timestamp('2025-10-01')),
        ('Oct', pd.Timestamp('2025-10-01'), pd.Timestamp('2025-11-01')),
        ('Nov', pd.Timestamp('2025-11-01'), pd.Timestamp('2025-12-01')),
        ('Dec', pd.Timestamp('2025-12-01'), pd.Timestamp('2026-01-01')),
        ('Jan', pd.Timestamp('2026-01-01'), pd.Timestamp('2026-02-01')),
        ('Feb', pd.Timestamp('2026-02-01'), pd.Timestamp('2026-02-13')),
    ]

    # Focus on v11 SL50 and v9.4 SL50 for LOMO
    lomo_configs = {
        'v11 SL50': configs['v11 SM(10/12/200/100) RSI8 60/40 CD20 SL50'],
        'v9.4 SL50': configs['v9.4 SM(20/12/400/255) RSI10 55/45 CD15 SL50'],
    }

    for config_name, cfg in lomo_configs.items():
        print(f"\n  {config_name}")
        print(f"  {'Month':>5}  {'Trades':>6}  {'WR%':>6}  {'PF':>7}  {'Net$':>9}  {'MaxDD$':>8}")
        print(f"  {'-'*55}")

        monthly_nets = []
        for month_name, start, end in months:
            df_slice = df_full[(df_full.index >= start) & (df_full.index < end)]
            if len(df_slice) < 100:
                continue
            trades, sc = run_on_slice(
                df_slice, cfg['sm_params'], cfg['rsi_len'],
                cfg['rsi_buy'], cfg['rsi_sell'], cfg['cooldown'],
                cfg['max_loss_pts'],
                full_sm=cfg['sm_full'], full_times=times_full)
            if sc:
                monthly_nets.append(sc['net_dollar'])
                print(f"  {month_name:>5}  {sc['count']:>6}  {sc['win_rate']:>5.1f}%  "
                      f"{sc['pf']:>7.3f}  ${sc['net_dollar']:>+8.2f}  ${sc['max_dd_dollar']:>7.2f}")
            else:
                monthly_nets.append(0)
                print(f"  {month_name:>5}  {'N/A':>6}")

        pos_months = sum(1 for x in monthly_nets if x > 0)
        print(f"  {'-'*55}")
        print(f"  Profitable months: {pos_months}/{len(monthly_nets)}  "
              f"Avg monthly: ${np.mean(monthly_nets):+.2f}  "
              f"Std: ${np.std(monthly_nets):.2f}")

    # =====================================================================
    # Lucky Trade Removal Test
    # =====================================================================
    print("\n" + "=" * 100)
    print("LUCKY TRADE REMOVAL TEST (v11 SL50)")
    print("Remove top N profitable trades and check if PF stays > 1.3")
    print("=" * 100)

    # Run v11 SL50 on full data
    df_full_copy = df_full.copy()
    trades_v11_full, sc_full = run_on_slice(
        df_full_copy, (10, 12, 200, 100), 8, 60, 40, 20, 50,
        full_sm=sm_v11_full, full_times=times_full)

    if trades_v11_full:
        pts_arr = np.array([t['pts'] for t in trades_v11_full])
        comm_pts = (COMMISSION * 2) / DOLLAR_PER_PT
        net_arr = pts_arr - comm_pts

        print(f"\n  Full data baseline: {sc_full['count']} trades, "
              f"PF {sc_full['pf']}, Net ${sc_full['net_dollar']:+.2f}")

        sorted_idx = np.argsort(net_arr)[::-1]  # best trades first

        for remove_n in [1, 2, 3, 5]:
            if remove_n >= len(net_arr):
                continue
            remaining = np.delete(net_arr, sorted_idx[:remove_n])
            w = remaining[remaining > 0].sum()
            ls = abs(remaining[remaining <= 0].sum())
            pf = w / ls if ls > 0 else 999.0
            net = remaining.sum() * DOLLAR_PER_PT
            removed_pts = net_arr[sorted_idx[:remove_n]]
            print(f"  Remove top {remove_n}: PF {pf:.3f}, Net ${net:+.2f}  "
                  f"(removed: {', '.join(f'${p*DOLLAR_PER_PT:+.0f}' for p in removed_pts)})  "
                  f"{'PASS' if pf > 1.3 else 'FRAGILE' if pf > 1.0 else 'FAIL'}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("""
  Walk-forward tests whether the v11 edge is real or curve-fit noise.

  PASS criteria:
    1. OOS PF >= 1.0 (profitable out of sample)
    2. OOS PF degradation < 30% from IS
    3. Profitable in >= 5/7 months
    4. PF stays > 1.3 after removing top 3 trades
  """)


if __name__ == "__main__":
    main()
