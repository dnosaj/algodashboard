"""
v10 Cross-Instrument Validation (File 4 of v10 Feature Suite)
=============================================================
Runs v9 baseline + each v10 feature on MNQ, MES, ES, and MYM to validate
cross-instrument robustness.

Features must be profitable (PF > 1.0) on MNQ plus at least 2 of 3 other
instruments to PASS.

Point-based parameters are scaled by average daily range relative to MNQ.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from v10_test_common import (
    INSTRUMENTS, load_instrument_1min, resample_to_5min,
    prepare_backtest_arrays, compute_rsi, compute_opening_range,
    compute_prior_day_levels, run_backtest_v10, run_v9_baseline,
    score_trades, fmt_score, fmt_exits,
)

# ---------------------------------------------------------------------------
# v9 baseline parameters
# ---------------------------------------------------------------------------
RSI_LEN = 10
RSI_BUY = 55
RSI_SELL = 45
SM_THRESHOLD = 0.0
COOLDOWN = 3
USE_RSI_CROSS = True

# MNQ base point parameters (will be scaled for other instruments)
BASE_PRIOR_DAY_BUFFER_PTS = 50
BASE_VWAP_BAND_PTS = 0  # 0 = no limit in base config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_avg_daily_range(df_5m):
    """Compute average daily range from 5-min data.

    Groups by date, finds daily high and low from all bars, computes range.
    Returns average range in points.
    """
    df_5m = df_5m.copy()
    df_5m['date'] = df_5m.index.date
    daily = df_5m.groupby('date').agg({'High': 'max', 'Low': 'min'})
    daily['range'] = daily['High'] - daily['Low']
    return daily['range'].mean()


def scale_pts(base_pts, inst_avg_range, mnq_avg_range):
    """Scale a point-based parameter from MNQ to another instrument."""
    if mnq_avg_range <= 0 or base_pts == 0:
        return base_pts
    scale_factor = inst_avg_range / mnq_avg_range
    return int(round(base_pts * scale_factor))


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

FEATURES = [
    'baseline',
    'underwater_exit_3',
    'or_alignment_30',
    'vwap_filter',
    'prior_day_buffer',
    'price_structure_exit',
    'sm_reversal_entry',
    'rsi_momentum',
]

FEATURE_LABELS = {
    'baseline':             'Baseline (v9)',
    'underwater_exit_3':    'Underwater exit (3)',
    'or_alignment_30':      'OR alignment (30)',
    'vwap_filter':          'VWAP filter',
    'prior_day_buffer':     'Prior day buffer',
    'price_structure_exit': 'Price struct exit',
    'sm_reversal_entry':    'SM reversal entry',
    'rsi_momentum':         'RSI momentum',
}


def run_feature(feature_name, arr, rsi, cfg, scaled_buffer_pts,
                or_direction=None, prev_high=None, prev_low=None):
    """Run a single feature test. Returns list of trades."""
    # Common kwargs for all runs
    base = dict(
        opens=arr['opens'], highs=arr['highs'], lows=arr['lows'],
        closes=arr['closes'], sm=arr['sm'], rsi=rsi, times=arr['times'],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL, sm_threshold=SM_THRESHOLD,
        cooldown_bars=COOLDOWN, use_rsi_cross=USE_RSI_CROSS,
    )

    if feature_name == 'baseline':
        return run_backtest_v10(**base)

    elif feature_name == 'underwater_exit_3':
        return run_backtest_v10(**base, underwater_exit_bars=3)

    elif feature_name == 'or_alignment_30':
        return run_backtest_v10(**base, or_direction=or_direction)

    elif feature_name == 'vwap_filter':
        if 'vwap' not in arr or arr['vwap'] is None:
            return None  # Not applicable
        return run_backtest_v10(**base, vwap=arr.get('vwap'),
                                vwap_filter=True, vwap_band_pts=0)

    elif feature_name == 'prior_day_buffer':
        return run_backtest_v10(**base, prior_day_buffer=True,
                                prior_day_high=prev_high,
                                prior_day_low=prev_low,
                                prior_day_buffer_pts=scaled_buffer_pts)

    elif feature_name == 'price_structure_exit':
        return run_backtest_v10(**base, price_structure_exit=True,
                                price_structure_bars=3)

    elif feature_name == 'sm_reversal_entry':
        return run_backtest_v10(**base, sm_reversal_entry=True)

    elif feature_name == 'rsi_momentum':
        return run_backtest_v10(**base, rsi_momentum_filter=True)

    return None


# ---------------------------------------------------------------------------
# Per-instrument test runner
# ---------------------------------------------------------------------------

def run_instrument_tests(instrument, mnq_avg_range=None):
    """Run all feature tests on an instrument.

    Returns dict: feature_name -> {'trades': list, 'score': dict or None, 'applicable': bool}
    Also returns avg_daily_range for scaling reference.
    """
    cfg = INSTRUMENTS[instrument]
    print(f"\n{'='*70}")
    print(f"  {instrument} -- commission=${cfg['commission']}, "
          f"${cfg['dollar_per_pt']}/pt, VWAP={'Yes' if cfg['has_vwap'] else 'No'}")
    print(f"{'='*70}")

    # Load and prepare data
    try:
        df_1m = load_instrument_1min(instrument)
    except Exception as e:
        print(f"  ERROR loading {instrument}: {e}")
        return {}, 0.0

    df_5m = resample_to_5min(df_1m)
    arr = prepare_backtest_arrays(df_5m)
    avg_range = compute_avg_daily_range(df_5m)

    print(f"  Data: {len(df_1m)} 1-min bars -> {len(df_5m)} 5-min bars")
    print(f"  Date range: {df_5m.index[0].date()} to {df_5m.index[-1].date()}")
    print(f"  Avg daily range: {avg_range:.2f} pts")

    # Compute RSI
    rsi = compute_rsi(arr['closes'], RSI_LEN)

    # Pre-compute feature arrays
    or_direction = compute_opening_range(arr['times'], arr['highs'],
                                          arr['lows'], or_minutes=30)
    prev_high, prev_low, prev_close = compute_prior_day_levels(
        arr['times'], arr['highs'], arr['lows'], arr['closes'])

    # Scale point-based parameters
    if mnq_avg_range is not None and mnq_avg_range > 0:
        scaled_buffer_pts = scale_pts(BASE_PRIOR_DAY_BUFFER_PTS,
                                       avg_range, mnq_avg_range)
    else:
        scaled_buffer_pts = BASE_PRIOR_DAY_BUFFER_PTS

    print(f"  Scaled prior_day_buffer_pts: {scaled_buffer_pts} "
          f"(base={BASE_PRIOR_DAY_BUFFER_PTS})")

    results = {}
    for feat in FEATURES:
        # Check applicability
        if feat == 'vwap_filter' and not cfg['has_vwap']:
            results[feat] = {'trades': None, 'score': None, 'applicable': False}
            continue

        trades = run_feature(feat, arr, rsi, cfg, scaled_buffer_pts,
                             or_direction=or_direction,
                             prev_high=prev_high, prev_low=prev_low)

        if trades is None:
            results[feat] = {'trades': None, 'score': None, 'applicable': False}
            continue

        sc = score_trades(trades, commission_per_side=cfg['commission'],
                          dollar_per_pt=cfg['dollar_per_pt'])
        results[feat] = {'trades': trades, 'score': sc, 'applicable': True}

        label = FEATURE_LABELS.get(feat, feat)
        print(f"\n  {fmt_score(sc, label)}")
        if sc is not None and sc.get('exits'):
            print(f"    Exits: {fmt_exits(sc['exits'])}")

    return results, avg_range


# ---------------------------------------------------------------------------
# Cross-instrument comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(all_results, instruments):
    """Print the cross-instrument comparison table."""
    print("\n")
    print("=" * 100)
    print("CROSS-INSTRUMENT RESULTS")
    print("=" * 100)

    # Header
    hdr_inst = " | ".join(f"{inst:>7s}" for inst in instruments)
    hdr = f"{'Feature':<22s} | {hdr_inst} | Pass?"
    print(hdr)
    print("-" * len(hdr))

    verdicts = {}

    for feat in FEATURES:
        label = FEATURE_LABELS.get(feat, feat)
        cells = []
        applicable_count = 0
        profitable_count = 0
        mnq_profitable = False

        for inst in instruments:
            res = all_results.get(inst, {}).get(feat)
            if res is None or not res['applicable']:
                cells.append("    N/A")
                continue

            applicable_count += 1
            sc = res['score']
            if sc is None:
                cells.append("  0 trd")
                continue

            pf = sc['pf']
            cells.append(f"  {pf:5.3f}")

            if pf > 1.0:
                profitable_count += 1
                if inst == 'MNQ':
                    mnq_profitable = True

        cells_str = " | ".join(cells)

        # Decision: MNQ profitable + at least 2 of 3 others
        other_instruments = [i for i in instruments if i != 'MNQ']
        other_profitable = 0
        other_applicable = 0
        for inst in other_instruments:
            res = all_results.get(inst, {}).get(feat)
            if res is None or not res['applicable']:
                continue
            other_applicable += 1
            sc = res['score']
            if sc is not None and sc['pf'] > 1.0:
                other_profitable += 1

        if feat == 'baseline':
            verdict = f"{profitable_count}/{applicable_count}"
        elif not mnq_profitable:
            verdict = f"{other_profitable}/{other_applicable} FAIL (MNQ<1)"
        elif other_applicable == 0:
            # Only MNQ is applicable (shouldn't happen, but just in case)
            verdict = "1/1 PASS" if mnq_profitable else "0/1 FAIL"
        else:
            needed = min(2, other_applicable)
            if other_profitable >= needed:
                verdict = f"{other_profitable}/{other_applicable} PASS"
            else:
                verdict = f"{other_profitable}/{other_applicable} FAIL"

        verdicts[feat] = verdict
        print(f"{label:<22s} | {cells_str} | {verdict}")

    print("-" * len(hdr))
    return verdicts


def print_detailed_results(all_results, instruments):
    """Print detailed per-instrument results for each feature."""
    print("\n")
    print("=" * 100)
    print("DETAILED PER-INSTRUMENT RESULTS")
    print("=" * 100)

    for feat in FEATURES:
        label = FEATURE_LABELS.get(feat, feat)
        print(f"\n--- {label} ---")
        print(f"  {'Instrument':<6s} | {'Trades':>6s} | {'WR%':>6s} | "
              f"{'PF':>7s} | {'Net $':>10s} | {'MaxDD $':>9s} | "
              f"{'Sharpe':>7s} | {'AvgPts':>7s}")
        print(f"  {'-'*6} | {'-'*6} | {'-'*6} | {'-'*7} | {'-'*10} | "
              f"{'-'*9} | {'-'*7} | {'-'*7}")

        for inst in instruments:
            res = all_results.get(inst, {}).get(feat)
            if res is None or not res['applicable']:
                print(f"  {inst:<6s} |    N/A |    N/A |     N/A |        N/A |       N/A |     N/A |     N/A")
                continue

            sc = res['score']
            if sc is None:
                print(f"  {inst:<6s} |      0 |    N/A |     N/A |        N/A |       N/A |     N/A |     N/A")
                continue

            print(f"  {inst:<6s} | {sc['count']:>6d} | {sc['win_rate']:>5.1f}% | "
                  f"{sc['pf']:>7.3f} | {sc['net_dollar']:>+10.2f} | "
                  f"{sc['max_dd_dollar']:>9.2f} | {sc['sharpe']:>7.3f} | "
                  f"{sc['avg_pts']:>7.2f}")


def print_summary(verdicts):
    """Print final pass/fail summary."""
    print("\n")
    print("=" * 100)
    print("VERDICT SUMMARY")
    print("=" * 100)

    passed = []
    failed = []

    for feat in FEATURES:
        if feat == 'baseline':
            continue
        label = FEATURE_LABELS.get(feat, feat)
        v = verdicts.get(feat, '')
        if 'PASS' in v:
            passed.append(label)
            print(f"  [PASS] {label:.<40s} {v}")
        else:
            failed.append(label)
            print(f"  [FAIL] {label:.<40s} {v}")

    print(f"\n  PASSED: {len(passed)}/{len(passed)+len(failed)} features")
    if passed:
        print(f"    -> {', '.join(passed)}")
    if failed:
        print(f"  FAILED: {len(failed)}/{len(passed)+len(failed)} features")
        print(f"    -> {', '.join(failed)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("v10 CROSS-INSTRUMENT VALIDATION")
    print("=" * 100)
    print(f"\nInstruments: {', '.join(INSTRUMENTS.keys())}")
    print(f"Baseline: RSI({RSI_LEN}) buy={RSI_BUY}/sell={RSI_SELL}, "
          f"SM thresh={SM_THRESHOLD}, cooldown={COOLDOWN}, rsi_cross={USE_RSI_CROSS}")
    print(f"Features: {len(FEATURES)-1} (excluding baseline)")

    instruments = list(INSTRUMENTS.keys())
    all_results = {}
    avg_ranges = {}

    # First pass: run MNQ to get reference avg daily range
    print("\n>>> Running MNQ first to establish scaling reference...")
    mnq_results, mnq_avg_range = run_instrument_tests('MNQ', mnq_avg_range=None)
    all_results['MNQ'] = mnq_results
    avg_ranges['MNQ'] = mnq_avg_range
    print(f"\n  MNQ avg daily range (scaling reference): {mnq_avg_range:.2f} pts")

    # Second pass: run remaining instruments with scaling
    for inst in instruments:
        if inst == 'MNQ':
            continue
        inst_results, inst_avg_range = run_instrument_tests(
            inst, mnq_avg_range=mnq_avg_range)
        all_results[inst] = inst_results
        avg_ranges[inst] = inst_avg_range

    # Print scaling summary
    print("\n")
    print("-" * 60)
    print("SCALING SUMMARY")
    print("-" * 60)
    for inst in instruments:
        rng = avg_ranges.get(inst, 0)
        scale = rng / mnq_avg_range if mnq_avg_range > 0 else 0
        scaled_buf = scale_pts(BASE_PRIOR_DAY_BUFFER_PTS, rng, mnq_avg_range)
        print(f"  {inst:<4s}: avg_range={rng:>8.2f}, "
              f"scale={scale:.4f}, buffer_pts={scaled_buf}")

    # Print comparison table
    verdicts = print_comparison_table(all_results, instruments)

    # Print detailed results
    print_detailed_results(all_results, instruments)

    # Print summary
    print_summary(verdicts)


if __name__ == "__main__":
    main()
