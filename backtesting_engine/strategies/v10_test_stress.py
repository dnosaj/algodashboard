"""
v10 Stress Test Suite (File 3 of 3)
====================================
Red-team / stress tests for all v10 features.

Tests:
  S1 - Walk-Forward Validation (IS vs OOS)
  S2 - Leave-One-Day-Out Cross-Validation
  S3 - Monte Carlo Trade Sequence (10,000 shuffles)
  S4 - Parameter Sensitivity Sweep
  S5 - Overfitting Detection (lucky trades, random baseline, temporal stability)
  S6 - Regime Analysis

Usage:  python3 v10_test_stress.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from v10_test_common import (
    load_instrument_1min, load_mnq_5min_prebaked, resample_to_5min,
    compute_rsi, compute_opening_range, compute_prior_day_levels,
    run_backtest_v10, score_trades, fmt_score, fmt_exits,
    bootstrap_ci, monte_carlo_equity, permutation_test,
    random_feature_baseline, prepare_backtest_arrays, run_v9_baseline,
    NY_OPEN_UTC, NY_LAST_ENTRY_UTC, NY_CLOSE_UTC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_feature(arr, feature_name, **feature_kwargs):
    """Run backtest with a specific feature enabled. Returns trades list."""
    rsi = compute_rsi(arr['closes'], 10)
    kwargs = dict(
        rsi_buy=55, rsi_sell=45, sm_threshold=0.0,
        cooldown_bars=3, use_rsi_cross=True,
    )
    kwargs.update(feature_kwargs)
    return run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], rsi, arr['times'], **kwargs,
    )


def build_feature_kwargs(arr, feature_name):
    """Return the kwargs dict needed to enable a specific feature.

    Some features require pre-computed arrays (OR direction, prior day levels,
    VWAP).  This function computes them from *arr* and returns a ready-to-use
    kwargs dict.
    """
    kw = {}
    if feature_name == 'underwater_exit':
        kw['underwater_exit_bars'] = 3
    elif feature_name == 'or_direction':
        or_dir = compute_opening_range(arr['times'], arr['highs'], arr['lows'],
                                       or_minutes=30)
        kw['or_direction'] = or_dir
    elif feature_name == 'vwap_filter':
        if 'vwap' in arr:
            kw['vwap'] = arr['vwap']
            kw['vwap_filter'] = True
        else:
            kw['vwap_filter'] = False  # no vwap data -- feature is a no-op
    elif feature_name == 'prior_day_buffer':
        pdh, pdl, pdc = compute_prior_day_levels(
            arr['times'], arr['highs'], arr['lows'], arr['closes'])
        kw['prior_day_high'] = pdh
        kw['prior_day_low'] = pdl
        kw['prior_day_buffer'] = True
        kw['prior_day_buffer_pts'] = 50
    elif feature_name == 'price_structure_exit':
        kw['price_structure_exit'] = True
        kw['price_structure_bars'] = 3
    elif feature_name == 'sm_reversal_entry':
        kw['sm_reversal_entry'] = True
    elif feature_name == 'rsi_momentum_filter':
        kw['rsi_momentum_filter'] = True
    else:
        raise ValueError(f"Unknown feature: {feature_name}")
    return kw


FEATURE_NAMES = [
    'underwater_exit',
    'or_direction',
    'vwap_filter',
    'prior_day_buffer',
    'price_structure_exit',
    'sm_reversal_entry',
    'rsi_momentum_filter',
]


def run_named_feature(arr, feature_name):
    """Run backtest with a single named feature. Returns trades."""
    kw = build_feature_kwargs(arr, feature_name)
    return run_feature(arr, feature_name, **kw)


def safe_pf(sc):
    """Extract PF from score dict, returning 0.0 if no trades."""
    if sc is None:
        return 0.0
    return sc['pf']


def safe_wr(sc):
    if sc is None:
        return 0.0
    return sc['win_rate']


def safe_net(sc):
    if sc is None:
        return 0.0
    return sc['net_dollar']


def filter_df_by_dates(df_5m, start_date, end_date):
    """Filter a DataFrame index to [start_date, end_date)."""
    return df_5m[(df_5m.index >= pd.Timestamp(start_date)) &
                 (df_5m.index < pd.Timestamp(end_date))]


def header(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ===========================================================================
# S1: Walk-Forward Validation
# ===========================================================================

def test_s1_walk_forward(df_5m):
    header("S1: WALK-FORWARD VALIDATION (IS vs OOS)")

    # Split dates
    # Use 70/30 IS/OOS split based on available trading days
    all_dates = sorted(set(df_5m.index.date))
    split_idx = int(len(all_dates) * 0.7)
    is_start = str(all_dates[0])
    is_end = str(all_dates[split_idx])
    oos_start = is_end
    oos_end = str(all_dates[-1] + pd.Timedelta(days=1))

    df_is = filter_df_by_dates(df_5m, is_start, is_end)
    df_oos = filter_df_by_dates(df_5m, oos_start, oos_end)

    print(f"  In-Sample:      {is_start} to {is_end}  ({len(df_is)} bars)")
    print(f"  Out-of-Sample:  {oos_start} to {oos_end}  ({len(df_oos)} bars)")

    arr_is = prepare_backtest_arrays(df_is)
    arr_oos = prepare_backtest_arrays(df_oos)

    results = {}  # feature -> {is_pf, oos_pf, pass}

    # Baseline
    for label, arr_split in [("IS", arr_is), ("OOS", arr_oos)]:
        trades = run_v9_baseline(arr_split)
        sc = score_trades(trades)
        results.setdefault('BASELINE', {})[label] = sc

    # Features
    for feat in FEATURE_NAMES:
        for label, arr_split in [("IS", arr_is), ("OOS", arr_oos)]:
            trades = run_named_feature(arr_split, feat)
            sc = score_trades(trades)
            results.setdefault(feat, {})[label] = sc

    # Print table
    print()
    print(f"  {'Feature':<24s} {'IS PF':>8s} {'IS N':>6s} {'OOS PF':>8s} "
          f"{'OOS N':>6s} {'Degrad%':>8s} {'Result':>8s}")
    print("  " + "-" * 72)

    verdicts = {}
    for feat in ['BASELINE'] + FEATURE_NAMES:
        is_sc = results[feat].get('IS')
        oos_sc = results[feat].get('OOS')
        is_pf = safe_pf(is_sc)
        oos_pf = safe_pf(oos_sc)
        is_n = is_sc['count'] if is_sc else 0
        oos_n = oos_sc['count'] if oos_sc else 0

        if is_pf > 0:
            degradation = (1 - oos_pf / is_pf) * 100 if is_pf != 999.0 else 0
        else:
            degradation = 100

        passed = oos_pf >= 1.0 and oos_pf >= 0.7 * is_pf
        verdicts[feat] = passed

        pf_is_str = f"{is_pf:.3f}" if is_pf < 100 else "inf"
        pf_oos_str = f"{oos_pf:.3f}" if oos_pf < 100 else "inf"
        result_str = "PASS" if passed else "FAIL"

        print(f"  {feat:<24s} {pf_is_str:>8s} {is_n:>6d} {pf_oos_str:>8s} "
              f"{oos_n:>6d} {degradation:>7.1f}% {result_str:>8s}")

    return verdicts


# ===========================================================================
# S2: Leave-One-Day-Out Cross-Validation
# ===========================================================================

def test_s2_loocv(df_5m):
    header("S2: LEAVE-ONE-DAY-OUT CROSS-VALIDATION")

    # Identify unique trading days
    dates = sorted(set(df_5m.index.date))
    print(f"  Trading days in dataset: {len(dates)}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")

    results = {}  # feature -> list of (date, net_dollar) for held-out day

    for feat in ['BASELINE'] + FEATURE_NAMES:
        daily_pnl = []
        for hold_date in dates:
            # Training set: all days except hold_date
            df_train = df_5m[df_5m.index.date != hold_date]
            # Held-out day
            df_hold = df_5m[df_5m.index.date == hold_date]

            if len(df_hold) < 5:
                continue

            arr_hold = prepare_backtest_arrays(df_hold)

            if feat == 'BASELINE':
                trades_hold = run_v9_baseline(arr_hold)
            else:
                trades_hold = run_named_feature(arr_hold, feat)

            sc = score_trades(trades_hold)
            net = safe_net(sc)
            daily_pnl.append((hold_date, net))

        results[feat] = daily_pnl

    # Print table
    print()
    print(f"  {'Feature':<24s} {'Avg P&L':>10s} {'Std Dev':>10s} "
          f"{'% Days+':>8s} {'Days':>6s}")
    print("  " + "-" * 62)

    verdicts = {}
    for feat in ['BASELINE'] + FEATURE_NAMES:
        pnls = [p for _, p in results[feat]]
        if pnls:
            avg_pnl = np.mean(pnls)
            std_pnl = np.std(pnls) if len(pnls) > 1 else 0
            pct_profitable = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        else:
            avg_pnl = std_pnl = pct_profitable = 0

        verdicts[feat] = avg_pnl > 0 and pct_profitable >= 50
        result = "PASS" if verdicts[feat] else "FAIL"

        print(f"  {feat:<24s} ${avg_pnl:>9.2f} ${std_pnl:>9.2f} "
              f"{pct_profitable:>7.1f}% {len(pnls):>6d}  {result}")

    # Detail per day for baseline
    print()
    print("  BASELINE daily detail:")
    for dt, pnl in results['BASELINE']:
        marker = "+" if pnl > 0 else ("-" if pnl < 0 else " ")
        print(f"    {dt}  ${pnl:>+9.2f}  {marker}")

    return verdicts


# ===========================================================================
# S3: Monte Carlo Trade Sequence
# ===========================================================================

def test_s3_monte_carlo(df_5m):
    header("S3: MONTE CARLO TRADE SEQUENCE (10,000 shuffles)")

    arr = prepare_backtest_arrays(df_5m)

    results = {}

    # Baseline
    trades_base = run_v9_baseline(arr)
    mc = monte_carlo_equity(trades_base, n_sims=10000)
    results['BASELINE'] = mc

    # Features
    for feat in FEATURE_NAMES:
        trades = run_named_feature(arr, feat)
        mc = monte_carlo_equity(trades, n_sims=10000)
        results[feat] = mc

    # Print max drawdown table
    print()
    print("  MAX DRAWDOWN Distribution:")
    print(f"  {'Feature':<24s} {'P5':>10s} {'P25':>10s} {'P50':>10s} "
          f"{'P75':>10s} {'P95':>10s}")
    print("  " + "-" * 68)

    for feat in ['BASELINE'] + FEATURE_NAMES:
        mc = results[feat]
        if mc is None:
            print(f"  {feat:<24s}  (no trades)")
            continue
        dd = mc['max_dd']
        print(f"  {feat:<24s} ${dd['P5']:>9.2f} ${dd['P25']:>9.2f} "
              f"${dd['P50']:>9.2f} ${dd['P75']:>9.2f} ${dd['P95']:>9.2f}")

    # Print final equity table
    print()
    print("  FINAL EQUITY Distribution:")
    print(f"  {'Feature':<24s} {'P5':>10s} {'P25':>10s} {'P50':>10s} "
          f"{'P75':>10s} {'P95':>10s}")
    print("  " + "-" * 68)

    for feat in ['BASELINE'] + FEATURE_NAMES:
        mc = results[feat]
        if mc is None:
            print(f"  {feat:<24s}  (no trades)")
            continue
        eq = mc['final_equity']
        print(f"  {feat:<24s} ${eq['P5']:>9.2f} ${eq['P25']:>9.2f} "
              f"${eq['P50']:>9.2f} ${eq['P75']:>9.2f} ${eq['P95']:>9.2f}")

    return results


# ===========================================================================
# S4: Parameter Sensitivity
# ===========================================================================

def test_s4_param_sensitivity(df_5m):
    header("S4: PARAMETER SENSITIVITY SWEEP")

    arr = prepare_backtest_arrays(df_5m)

    rsi_lengths = [8, 10, 12, 14]
    rsi_levels = [(55, 45), (60, 40), (65, 35)]
    sm_thresholds = [0.0, 0.05, 0.10]
    cooldowns = [2, 3, 4, 6]

    total_combos = len(rsi_lengths) * len(rsi_levels) * len(sm_thresholds) * len(cooldowns)
    print(f"  Parameter grid: {total_combos} combinations per feature")

    verdicts = {}

    for feat in ['BASELINE'] + FEATURE_NAMES:
        pf_values = []

        # Build feature-specific kwargs (that don't depend on param sweep)
        if feat != 'BASELINE':
            feat_kw = build_feature_kwargs(arr, feat)
        else:
            feat_kw = {}

        for rsi_len in rsi_lengths:
            rsi = compute_rsi(arr['closes'], rsi_len)
            for rsi_buy, rsi_sell in rsi_levels:
                for sm_thresh in sm_thresholds:
                    for cd in cooldowns:
                        base_kw = dict(
                            rsi_buy=rsi_buy, rsi_sell=rsi_sell,
                            sm_threshold=sm_thresh, cooldown_bars=cd,
                            use_rsi_cross=True,
                        )
                        base_kw.update(feat_kw)
                        trades = run_backtest_v10(
                            arr['opens'], arr['highs'], arr['lows'],
                            arr['closes'], arr['sm'], rsi, arr['times'],
                            **base_kw,
                        )
                        sc = score_trades(trades)
                        pf_values.append(safe_pf(sc))

        pf_arr = np.array(pf_values)
        # Cap extreme PF values for std calculation
        pf_capped = np.clip(pf_arr, 0, 20)
        pct_profitable = (pf_arr > 1.0).sum() / len(pf_arr) * 100
        pf_std = np.std(pf_capped)
        pf_mean = np.mean(pf_capped)
        pf_median = np.median(pf_capped)

        if pct_profitable >= 70 and pf_std < 0.5:
            stability = "STABLE"
        elif pct_profitable < 50 or pf_std > 1.0:
            stability = "FRAGILE"
        else:
            stability = "MIXED"

        verdicts[feat] = stability

        print(f"\n  {feat}:")
        print(f"    Profitable variants: {pct_profitable:.1f}% ({int((pf_arr > 1.0).sum())}/{len(pf_arr)})")
        print(f"    PF mean={pf_mean:.3f}  median={pf_median:.3f}  std={pf_std:.3f}")
        print(f"    PF min={pf_arr.min():.3f}  max={min(pf_arr.max(), 999):.3f}")
        print(f"    Stability: {stability}")

    # Summary table
    print()
    print(f"  {'Feature':<24s} {'% Prof':>8s} {'PF Std':>8s} {'Stability':>10s}")
    print("  " + "-" * 54)
    for feat in ['BASELINE'] + FEATURE_NAMES:
        # Re-extract from verdicts -- we already printed details above
        print(f"  {feat:<24s} {'':>8s} {'':>8s} {verdicts[feat]:>10s}")

    return verdicts


# ===========================================================================
# S5: Overfitting Detection
# ===========================================================================

def test_s5_overfitting(df_5m):
    header("S5: OVERFITTING DETECTION")

    arr = prepare_backtest_arrays(df_5m)

    # -----------------------------------------------------------------------
    # 5a: Lucky Trade Test
    # -----------------------------------------------------------------------
    print("\n  --- 5a: Lucky Trade Test ---")
    print(f"  {'Feature':<24s} {'Base PF':>8s} {'Drop1':>8s} {'Drop2':>8s} "
          f"{'Drop3':>8s} {'Result':>8s}")
    print("  " + "-" * 62)

    lucky_verdicts = {}
    for feat in ['BASELINE'] + FEATURE_NAMES:
        if feat == 'BASELINE':
            trades = run_v9_baseline(arr)
        else:
            trades = run_named_feature(arr, feat)

        sc_full = score_trades(trades)
        base_pf = safe_pf(sc_full)

        # Sort trades by P&L descending, remove top 1, 2, 3
        if trades:
            sorted_trades = sorted(trades, key=lambda t: t['pts'], reverse=True)
            drop_pfs = []
            for k in [1, 2, 3]:
                remaining = sorted_trades[k:]
                sc_drop = score_trades(remaining)
                drop_pfs.append(safe_pf(sc_drop))

            # FAIL if any drop sends PF below 1.3
            fragile = any(pf < 1.3 for pf in drop_pfs)
            result = "FRAGILE" if fragile else "PASS"
            lucky_verdicts[feat] = not fragile

            print(f"  {feat:<24s} {base_pf:>8.3f} {drop_pfs[0]:>8.3f} "
                  f"{drop_pfs[1]:>8.3f} {drop_pfs[2]:>8.3f} {result:>8s}")
        else:
            lucky_verdicts[feat] = False
            print(f"  {feat:<24s}  (no trades)  FAIL")

    # -----------------------------------------------------------------------
    # 5b: Random Feature Baseline
    # -----------------------------------------------------------------------
    print("\n  --- 5b: Random Feature Baseline ---")
    print(f"  {'Feature':<24s} {'Real PF':>8s} {'Rand P95':>9s} {'Result':>8s}")
    print("  " + "-" * 53)

    # Get baseline trades for random feature comparison
    trades_base = run_v9_baseline(arr)
    rand_pfs = random_feature_baseline(trades_base, n_sims=200)
    p95_random = np.percentile(rand_pfs, 95) if len(rand_pfs) > 0 else 999

    random_verdicts = {}
    for feat in ['BASELINE'] + FEATURE_NAMES:
        if feat == 'BASELINE':
            trades = trades_base
        else:
            trades = run_named_feature(arr, feat)

        sc = score_trades(trades)
        real_pf = safe_pf(sc)
        passed = real_pf > p95_random
        random_verdicts[feat] = passed
        result = "PASS" if passed else "FAIL"

        print(f"  {feat:<24s} {real_pf:>8.3f} {p95_random:>9.3f} {result:>8s}")

    # -----------------------------------------------------------------------
    # 5c: Temporal Stability (4 windows)
    # -----------------------------------------------------------------------
    print("\n  --- 5c: Temporal Stability (4 windows) ---")

    dates = sorted(set(df_5m.index.date))
    n_dates = len(dates)
    window_size = n_dates // 4
    windows = []
    for w in range(4):
        start_idx = w * window_size
        end_idx = start_idx + window_size if w < 3 else n_dates
        window_dates = dates[start_idx:end_idx]
        windows.append(window_dates)

    print(f"  Windows: ", end="")
    for w_idx, w_dates in enumerate(windows):
        print(f"W{w_idx+1}({w_dates[0]} to {w_dates[-1]}, {len(w_dates)}d)  ", end="")
    print()

    print(f"\n  {'Feature':<24s}", end="")
    for w_idx in range(4):
        print(f" {'W'+str(w_idx+1)+' PF':>8s}", end="")
    print(f" {'#Prof':>6s} {'Result':>8s}")
    print("  " + "-" * 66)

    temporal_verdicts = {}
    for feat in ['BASELINE'] + FEATURE_NAMES:
        window_pfs = []
        for w_dates in windows:
            w_date_set = set(w_dates)
            df_window = df_5m[[d in w_date_set for d in df_5m.index.date]]
            if len(df_window) < 5:
                window_pfs.append(0.0)
                continue
            arr_w = prepare_backtest_arrays(df_window)
            if feat == 'BASELINE':
                trades_w = run_v9_baseline(arr_w)
            else:
                trades_w = run_named_feature(arr_w, feat)
            sc_w = score_trades(trades_w)
            window_pfs.append(safe_pf(sc_w))

        profitable_windows = sum(1 for pf in window_pfs if pf > 1.0)
        passed = profitable_windows >= 3
        temporal_verdicts[feat] = passed
        result = "PASS" if passed else "FAIL"

        print(f"  {feat:<24s}", end="")
        for pf in window_pfs:
            pf_str = f"{pf:.3f}" if pf < 100 else "inf"
            print(f" {pf_str:>8s}", end="")
        print(f" {profitable_windows:>6d} {result:>8s}")

    return lucky_verdicts, random_verdicts, temporal_verdicts


# ===========================================================================
# S6: Regime Analysis
# ===========================================================================

def test_s6_regime(df_5m):
    header("S6: REGIME ANALYSIS")

    # Classify each trading day
    dates = sorted(set(df_5m.index.date))

    day_stats = {}
    for dt in dates:
        df_day = df_5m[df_5m.index.date == dt]
        # Filter to RTH bars for classification
        rth = df_day[
            (df_day.index.hour * 60 + df_day.index.minute >= NY_OPEN_UTC) &
            (df_day.index.hour * 60 + df_day.index.minute < NY_CLOSE_UTC)
        ]
        if len(rth) == 0:
            continue
        day_high = rth['High'].max()
        day_low = rth['Low'].min()
        day_open = rth['Open'].iloc[0]
        day_close = rth['Close'].iloc[-1]
        day_range = day_high - day_low
        day_stats[dt] = {
            'high': day_high, 'low': day_low, 'open': day_open,
            'close': day_close, 'range': day_range,
        }

    # Compute median range
    ranges = [s['range'] for s in day_stats.values()]
    median_range = np.median(ranges)

    print(f"  Median daily range: {median_range:.2f} pts")
    print(f"  Wide threshold: > {median_range * 1.25:.2f} pts")
    print(f"  Narrow threshold: < {median_range * 0.75:.2f} pts")

    # Classify
    regimes = {}  # date -> (range_regime, trend_regime)
    for dt, s in day_stats.items():
        # Range classification
        if s['range'] > 1.25 * median_range:
            range_regime = 'wide'
        elif s['range'] < 0.75 * median_range:
            range_regime = 'narrow'
        else:
            range_regime = 'normal'

        # Trend classification
        body_thresh = 0.3 * s['range']
        if s['close'] > s['open'] + body_thresh:
            trend_regime = 'up'
        elif s['close'] < s['open'] - body_thresh:
            trend_regime = 'down'
        else:
            trend_regime = 'choppy'

        regimes[dt] = (range_regime, trend_regime)

    # Print regime distribution
    print()
    regime_counts = {}
    for dt, (rr, tr) in regimes.items():
        key = f"{rr}/{tr}"
        regime_counts[key] = regime_counts.get(key, 0) + 1
    for key in sorted(regime_counts.keys()):
        print(f"    {key}: {regime_counts[key]} days")

    # Run per-regime analysis
    range_regimes = ['wide', 'normal', 'narrow']
    trend_regimes = ['up', 'down', 'choppy']

    print()
    print("  --- By Range Regime ---")
    print(f"  {'Feature':<24s}", end="")
    for rr in range_regimes:
        print(f" {rr+' PF':>10s} {rr+' WR':>8s} {rr+' N':>5s}", end="")
    print()
    print("  " + "-" * 95)

    regime_verdicts = {}

    for feat in ['BASELINE'] + FEATURE_NAMES:
        regime_pfs = {}
        print(f"  {feat:<24s}", end="")

        for rr in range_regimes:
            rr_dates = set(dt for dt, (r, t) in regimes.items() if r == rr)
            if not rr_dates:
                print(f" {'n/a':>10s} {'n/a':>8s} {'0':>5s}", end="")
                regime_pfs[rr] = 0.0
                continue

            df_regime = df_5m[[d in rr_dates for d in df_5m.index.date]]
            if len(df_regime) < 5:
                print(f" {'n/a':>10s} {'n/a':>8s} {'0':>5s}", end="")
                regime_pfs[rr] = 0.0
                continue

            arr_r = prepare_backtest_arrays(df_regime)
            if feat == 'BASELINE':
                trades_r = run_v9_baseline(arr_r)
            else:
                trades_r = run_named_feature(arr_r, feat)
            sc_r = score_trades(trades_r)
            pf = safe_pf(sc_r)
            wr = safe_wr(sc_r)
            n = sc_r['count'] if sc_r else 0
            regime_pfs[rr] = pf

            pf_str = f"{pf:.3f}" if pf < 100 else "inf"
            print(f" {pf_str:>10s} {wr:>7.1f}% {n:>5d}", end="")

        # Flag if feature only works in one regime
        profitable_regimes = sum(1 for pf in regime_pfs.values() if pf > 1.0)
        total_with_data = sum(1 for pf in regime_pfs.values() if pf > 0)
        one_regime_only = (profitable_regimes == 1 and total_with_data >= 2)
        regime_verdicts[feat] = not one_regime_only

        print()

    print()
    print("  --- By Trend Regime ---")
    print(f"  {'Feature':<24s}", end="")
    for tr in trend_regimes:
        print(f" {tr+' PF':>10s} {tr+' WR':>8s} {tr+' N':>5s}", end="")
    print()
    print("  " + "-" * 95)

    for feat in ['BASELINE'] + FEATURE_NAMES:
        trend_pfs = {}
        print(f"  {feat:<24s}", end="")

        for tr in trend_regimes:
            tr_dates = set(dt for dt, (r, t) in regimes.items() if t == tr)
            if not tr_dates:
                print(f" {'n/a':>10s} {'n/a':>8s} {'0':>5s}", end="")
                trend_pfs[tr] = 0.0
                continue

            df_regime = df_5m[[d in tr_dates for d in df_5m.index.date]]
            if len(df_regime) < 5:
                print(f" {'n/a':>10s} {'n/a':>8s} {'0':>5s}", end="")
                trend_pfs[tr] = 0.0
                continue

            arr_r = prepare_backtest_arrays(df_regime)
            if feat == 'BASELINE':
                trades_r = run_v9_baseline(arr_r)
            else:
                trades_r = run_named_feature(arr_r, feat)
            sc_r = score_trades(trades_r)
            pf = safe_pf(sc_r)
            wr = safe_wr(sc_r)
            n = sc_r['count'] if sc_r else 0
            trend_pfs[tr] = pf

            pf_str = f"{pf:.3f}" if pf < 100 else "inf"
            print(f" {pf_str:>10s} {wr:>7.1f}% {n:>5d}", end="")

        profitable_trends = sum(1 for pf in trend_pfs.values() if pf > 1.0)
        total_with_data = sum(1 for pf in trend_pfs.values() if pf > 0)
        one_trend_only = (profitable_trends == 1 and total_with_data >= 2)
        if one_trend_only:
            regime_verdicts[feat] = False  # Override to fail

        print()

    # Print flags
    print()
    for feat in ['BASELINE'] + FEATURE_NAMES:
        if not regime_verdicts.get(feat, True):
            print(f"  WARNING: {feat} only profitable in one regime -- regime-dependent!")

    return regime_verdicts


# ===========================================================================
# MAIN: Run all stress tests
# ===========================================================================

def main():
    print()
    print("*" * 80)
    print("*  v10 STRESS TEST SUITE")
    print("*  Red-team validation for all candidate features")
    print("*" * 80)

    # Load data
    print("\n  Loading MNQ 1-min data and resampling to 5-min...")
    df_1m = load_instrument_1min('MNQ')
    df_5m = resample_to_5min(df_1m)
    print(f"  5-min bars: {len(df_5m)}, range: {df_5m.index[0].date()} to {df_5m.index[-1].date()}")
    print(f"  Trading days: {len(set(df_5m.index.date))}")

    # Run all tests
    s1_verdicts = test_s1_walk_forward(df_5m)
    s2_verdicts = test_s2_loocv(df_5m)
    s3_results = test_s3_monte_carlo(df_5m)
    s4_verdicts = test_s4_param_sensitivity(df_5m)
    s5_lucky, s5_random, s5_temporal = test_s5_overfitting(df_5m)
    s6_verdicts = test_s6_regime(df_5m)

    # -----------------------------------------------------------------------
    # STRESS TEST SUMMARY
    # -----------------------------------------------------------------------
    header("STRESS TEST SUMMARY")

    tests = ['S1:WalkFwd', 'S2:LOOCV', 'S4:Stability', 'S5a:Lucky',
             'S5b:Random', 'S5c:Temporal', 'S6:Regime']

    print(f"  {'Feature':<24s}", end="")
    for t in tests:
        print(f" {t:>12s}", end="")
    print(f" {'OVERALL':>10s}")
    print("  " + "-" * (24 + 12 * len(tests) + 10))

    for feat in ['BASELINE'] + FEATURE_NAMES:
        verdicts = {
            'S1:WalkFwd': s1_verdicts.get(feat, False),
            'S2:LOOCV': s2_verdicts.get(feat, False),
            'S4:Stability': s4_verdicts.get(feat, 'FRAGILE') in ('STABLE', 'MIXED'),
            'S5a:Lucky': s5_lucky.get(feat, False),
            'S5b:Random': s5_random.get(feat, False),
            'S5c:Temporal': s5_temporal.get(feat, False),
            'S6:Regime': s6_verdicts.get(feat, False),
        }

        all_pass = all(verdicts.values())
        pass_count = sum(1 for v in verdicts.values() if v)

        print(f"  {feat:<24s}", end="")
        for t in tests:
            v = verdicts[t]
            label = "PASS" if v else "FAIL"
            print(f" {label:>12s}", end="")

        overall = f"{pass_count}/{len(tests)}"
        if all_pass:
            overall += " OK"
        print(f" {overall:>10s}")

    print()
    print("  Legend:")
    print("    S1: OOS PF >= 1.0 AND >= 70% of IS PF")
    print("    S2: Avg daily P&L > 0, >= 50% days profitable")
    print("    S4: >= 70% param combos profitable, PF std < 0.5 = STABLE")
    print("    S5a: PF stays >= 1.3 after removing top 1-3 trades")
    print("    S5b: Real PF beats P95 of random feature baseline")
    print("    S5c: Profitable in >= 3 of 4 time windows")
    print("    S6: Not dependent on a single regime")
    print()


if __name__ == "__main__":
    main()
