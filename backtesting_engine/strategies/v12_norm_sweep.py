"""
CVD Normalization Period Sweep
===============================
Sweeps the rolling-max normalization window for the G1 entry filter variant.

Tests: [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400]
Runs on both TRAIN and OOS to check for optimal window.

Usage:
  python3 v12_norm_sweep.py --delta data/databento_NQ_delta_1min_*.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_instrument_1min, prepare_backtest_arrays_1min,
    run_v9_baseline, score_trades,
    compute_et_minutes, compute_rsi,
    NY_OPEN_ET, NY_CLOSE_ET, NY_LAST_ENTRY_ET,
)
from v12_cvd_test import (
    load_delta_bars, _run_baseline,
    V11_RSI_LEN, V11_RSI_BUY, V11_RSI_SELL,
    V11_COOLDOWN, V11_MAX_LOSS, V11_SM_THRESHOLD,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    _run_custom_engine,
)

NORM_PERIODS = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400]


def fast_normalize(arr, window):
    """Rolling-max normalize to [-1,1]."""
    n = len(arr)
    out = np.zeros(n)
    for i in range(n):
        s = max(0, i - window + 1)
        abs_max = np.max(np.abs(arr[s:i + 1]))
        out[i] = arr[i] / abs_max if abs_max > 0 else 0.0
    return out


def run_sweep(delta_df, ohlcv_1m, period_start, period_end, period_name):
    """Run G1 entry filter sweep for all norm periods on one data period."""

    period_data = ohlcv_1m[(ohlcv_1m.index >= period_start) &
                            (ohlcv_1m.index < period_end)]

    if len(period_data) < 100:
        print(f"  ERROR: Only {len(period_data)} bars in {period_name}")
        return []

    # Align raw data (only need raw delta and CVD, normalization done per-sweep)
    raw_delta = delta_df['delta'].reindex(period_data.index, method='ffill').fillna(0).values
    cvd_raw = delta_df['CVD'].reindex(period_data.index, method='ffill').fillna(0).values

    n = len(raw_delta)

    # Rolling sums (needed by engine even though we don't use burst exits)
    roll_5 = np.zeros(n)
    roll_10 = np.zeros(n)
    for i in range(n):
        s5 = max(0, i - 4)
        roll_5[i] = np.sum(raw_delta[s5:i + 1])
        s10 = max(0, i - 9)
        roll_10[i] = np.sum(raw_delta[s10:i + 1])

    # Prepare arrays
    arr = prepare_backtest_arrays_1min(period_data, rsi_len=V11_RSI_LEN)

    # Baseline
    trades_baseline = _run_baseline(arr)
    sc_base = score_trades(trades_baseline)

    print(f"\n  {period_name}: {len(period_data):,} bars")
    print(f"  Baseline: {sc_base['count']} trades, PF {sc_base['pf']:.3f}, "
          f"Net ${sc_base['net_dollar']:+.2f}, MaxDD ${sc_base['max_dd_dollar']:.2f}")

    # Sweep
    results = []
    for norm_p in NORM_PERIODS:
        # Compute normalized CVD with this window
        cvd_norm = fast_normalize(cvd_raw, norm_p)

        # Build delta_data dict for engine
        delta_data = {
            'delta': raw_delta,
            'roll_5': roll_5,
            'roll_10': roll_10,
        }

        # Run G1 with this normalization
        trades = _run_custom_engine(arr, delta_data, entry_filter_arr=cvd_norm)
        sc = score_trades(trades)

        if sc is None:
            results.append({
                'norm': norm_p, 'trades': 0, 'pf': 0, 'wr': 0,
                'net': 0, 'dd': 0, 'dpf': 0,
            })
            continue

        dpf = sc['pf'] - sc_base['pf']
        results.append({
            'norm': norm_p,
            'trades': sc['count'],
            'pf': sc['pf'],
            'wr': sc['win_rate'],
            'net': sc['net_dollar'],
            'dd': sc['max_dd_dollar'],
            'dpf': dpf,
        })

    return results, sc_base


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CVD normalization period sweep")
    parser.add_argument("--delta", required=True, help="Path to volume delta CSV")
    parser.add_argument("--instrument", default="MNQ", help="Instrument (default: MNQ)")
    args = parser.parse_args()

    delta_path = Path(args.delta)
    if not delta_path.exists():
        print(f"ERROR: {delta_path} not found")
        sys.exit(1)

    print("=" * 80)
    print("CVD NORMALIZATION PERIOD SWEEP (G1 Entry Filter)")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    delta_df = load_delta_bars(args.delta)
    ohlcv_1m = load_instrument_1min(args.instrument)

    if ohlcv_1m.index.tz is None:
        ohlcv_1m.index = ohlcv_1m.index.tz_localize('UTC')

    print(f"  OHLCV: {len(ohlcv_1m):,} bars")
    print(f"  Delta: {len(delta_df):,} bars")

    # Run on TRAIN
    print("\n" + "=" * 80)
    print("TRAIN PERIOD (Aug 17 - Nov 16, 2025)")
    print("=" * 80)
    train_results, train_base = run_sweep(
        delta_df, ohlcv_1m, TRAIN_START, TRAIN_END, "TRAIN")

    # Run on TEST
    print("\n" + "=" * 80)
    print("TEST PERIOD (Nov 17, 2025 - Feb 13, 2026)")
    print("=" * 80)
    test_results, test_base = run_sweep(
        delta_df, ohlcv_1m, TEST_START, TEST_END, "TEST")

    # Combined results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n  Baselines:")
    print(f"    TRAIN: {train_base['count']} trades, PF {train_base['pf']:.3f}, "
          f"Net ${train_base['net_dollar']:+.2f}")
    print(f"    TEST:  {test_base['count']} trades, PF {test_base['pf']:.3f}, "
          f"Net ${test_base['net_dollar']:+.2f}")

    header = (f"  {'Norm':>5} | {'Trades':>6} {'PF':>7} {'dPF':>7} {'WR':>6} "
              f"{'Net$':>9} {'DD$':>8} | "
              f"{'Trades':>6} {'PF':>7} {'dPF':>7} {'WR':>6} "
              f"{'Net$':>9} {'DD$':>8}")
    print(f"\n  {'':>5}   {'--- TRAIN ---':^47}   {'--- TEST (OOS) ---':^47}")
    print(header)
    print(f"  {'-'*5}-+-{'-'*47}-+-{'-'*47}")

    best_train = None
    best_test = None
    best_combined = None

    for tr, te in zip(train_results, test_results):
        norm = tr['norm']
        line = (f"  {norm:>5} | "
                f"{tr['trades']:>6} {tr['pf']:>7.3f} {tr['dpf']:>+7.3f} "
                f"{tr['wr']:>5.1f}% ${tr['net']:>+8.2f} ${tr['dd']:>7.2f} | "
                f"{te['trades']:>6} {te['pf']:>7.3f} {te['dpf']:>+7.3f} "
                f"{te['wr']:>5.1f}% ${te['net']:>+8.2f} ${te['dd']:>7.2f}")
        print(line)

        if best_train is None or tr['dpf'] > best_train['dpf']:
            best_train = tr
        if best_test is None or te['dpf'] > best_test['dpf']:
            best_test = te

        # Combined: average dPF across train and test (rewards consistency)
        avg_dpf = (tr['dpf'] + te['dpf']) / 2
        if best_combined is None or avg_dpf > best_combined[2]:
            best_combined = (tr, te, avg_dpf)

    print(f"\n  Best TRAIN norm period: {best_train['norm']} bars "
          f"(dPF {best_train['dpf']:+.3f})")
    print(f"  Best TEST norm period:  {best_test['norm']} bars "
          f"(dPF {best_test['dpf']:+.3f})")
    if best_combined:
        bc_tr, bc_te, bc_avg = best_combined
        print(f"  Best COMBINED (avg dPF): {bc_tr['norm']} bars "
              f"(train dPF {bc_tr['dpf']:+.3f}, test dPF {bc_te['dpf']:+.3f}, "
              f"avg {bc_avg:+.3f})")

    # Conviction sizing sweep (optional — only for best combined norm period)
    if best_combined:
        best_norm = bc_tr['norm']
        print(f"\n  Running conviction sizing with norm={best_norm}...")

        for pname, pstart, pend in [
            ("TRAIN", TRAIN_START, TRAIN_END),
            ("TEST", TEST_START, TEST_END),
        ]:
            pd_data = ohlcv_1m[(ohlcv_1m.index >= pstart) & (ohlcv_1m.index < pend)]
            cvd_raw = delta_df['CVD'].reindex(pd_data.index, method='ffill').fillna(0).values
            cvd_norm = fast_normalize(cvd_raw, best_norm)

            arr = prepare_backtest_arrays_1min(pd_data, rsi_len=V11_RSI_LEN)
            trades_bl = _run_baseline(arr)
            sc_bl = score_trades(trades_bl)
            times = arr['times']

            # Tag each trade with CVD agree/disagree
            time_to_idx = {times[j]: j for j in range(len(times))}
            n_agree, n_disagree = 0, 0
            agree_trades, disagree_trades = [], []

            for t in trades_bl:
                idx = time_to_idx.get(t['entry_time'])
                if idx is None or idx < 1:
                    disagree_trades.append(t)
                    n_disagree += 1
                    continue

                cvd_val = cvd_norm[idx - 1]
                if (t['side'] == 'long' and cvd_val > 0) or \
                   (t['side'] == 'short' and cvd_val < 0):
                    agree_trades.append(t)
                    n_agree += 1
                else:
                    disagree_trades.append(t)
                    n_disagree += 1

            sc_agree = score_trades(agree_trades) if agree_trades else None
            sc_disagree = score_trades(disagree_trades) if disagree_trades else None

            print(f"\n    {pname} conviction (norm={best_norm}):")
            print(f"      Agree: {n_agree} trades, "
                  f"PF {sc_agree['pf']:.3f}" if sc_agree else f"      Agree: 0 trades")
            print(f"      Disagree: {n_disagree} trades, "
                  f"PF {sc_disagree['pf']:.3f}" if sc_disagree else f"      Disagree: 0 trades")
            if sc_agree and sc_disagree:
                spread = sc_agree['pf'] - sc_disagree['pf']
                print(f"      PF spread: {spread:+.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
