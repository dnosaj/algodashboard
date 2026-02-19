"""
v10 Feature Validation -- Final Decision Matrix & Report Generator
===================================================================
CORRECTED: Runs on 1-MIN bars with 5-min RSI mapped back, matching
the actual Pine Script architecture (1-min SM, 1-min OHLC for exits,
5-min RSI via request.security() for entries).

Master runner: executes ALL tests (feature performance, walk-forward,
cross-instrument, sensitivity, overfitting) and produces a final Go/No-Go
decision matrix for each candidate feature.

Standalone runnable:  python3 v10_test_report.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Import everything from the common module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_test_common import (
    load_instrument_1min, resample_to_5min,
    compute_rsi, compute_opening_range, compute_prior_day_levels,
    run_backtest_v10, score_trades,
    prepare_backtest_arrays_1min, map_5min_rsi_to_1min, run_v9_baseline,
    bootstrap_ci, monte_carlo_equity, permutation_test, random_feature_baseline,
    INSTRUMENTS, fmt_score, fmt_exits,
    NY_OPEN_UTC, NY_LAST_ENTRY_UTC, NY_CLOSE_UTC,
)


# ---------------------------------------------------------------------------
# Constants -- NOTE: cooldown=15 is in 1-min bars (= 15 min = 3 on 5-min)
# ---------------------------------------------------------------------------

V9_PARAMS = dict(
    rsi_len=10, rsi_buy=55, rsi_sell=45,
    sm_threshold=0.0, cooldown=15, use_rsi_cross=True,
)

N_BOOT = 5000
N_PERMS = 5000

# Date ranges computed dynamically from data in main()
FULL_START = None
FULL_END   = None
IS_START = None
IS_END   = None
OOS_START = None
OOS_END   = None
TEMPORAL_WINDOWS = []


def compute_date_splits(df):
    """Compute walk-forward and temporal window splits from data."""
    global FULL_START, FULL_END, IS_START, IS_END, OOS_START, OOS_END, TEMPORAL_WINDOWS
    all_dates = sorted(set(df.index.date))
    FULL_START = pd.Timestamp(str(all_dates[0]))
    FULL_END = pd.Timestamp(str(all_dates[-1])) + pd.Timedelta(days=1)

    # 70/30 walk-forward split
    split_idx = int(len(all_dates) * 0.7)
    IS_START = FULL_START
    IS_END = pd.Timestamp(str(all_dates[split_idx]))
    OOS_START = IS_END
    OOS_END = FULL_END

    # 4 equal temporal windows
    TEMPORAL_WINDOWS.clear()
    window_size = len(all_dates) // 4
    for w in range(4):
        s = w * window_size
        e = s + window_size if w < 3 else len(all_dates)
        ws = pd.Timestamp(str(all_dates[s]))
        we = pd.Timestamp(str(all_dates[e - 1])) + pd.Timedelta(days=1)
        TEMPORAL_WINDOWS.append((ws, we))

# Features under evaluation
FEATURES = {
    'underwater_3':     {'underwater_exit_bars': 3},
    'underwater_4':     {'underwater_exit_bars': 4},
    'or_align_30':      {},   # needs or_direction precomputed
    'or_align_45':      {},   # needs or_direction precomputed
    'vwap_filter':      {'vwap_filter': True},
    'prior_day_50':     {'prior_day_buffer': True, 'prior_day_buffer_pts': 50},
    'prior_day_75':     {'prior_day_buffer': True, 'prior_day_buffer_pts': 75},
    'price_struct_3':   {'price_structure_exit': True, 'price_structure_bars': 3},
    'price_struct_4':   {'price_structure_exit': True, 'price_structure_bars': 4},
    'sm_reversal':      {'sm_reversal_entry': True},
    'rsi_momentum':     {'rsi_momentum_filter': True},
}

# Sensitivity sweep grid
SENS_RSI_LENS   = [8, 10, 12, 14]
SENS_RSI_LEVELS = [(55, 45), (60, 40), (65, 35)]
SENS_THRESHOLDS = [0.0, 0.05, 0.10]

# Adoption thresholds
MIN_TRADES          = 20
WF_MIN_OOS_PF       = 1.0
WF_MAX_DEGRADATION   = 0.30     # 30%
SENS_STABILITY_MIN   = 0.70     # 70% profitable
SENS_PF_STD_MAX      = 0.5
LUCKY_REMOVAL_PF_MIN = 1.3
ADOPT_THRESHOLD      = 7        # must pass all 7 criteria
MAYBE_THRESHOLD      = 5        # pass 5-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_feature(arr, feature_name, feature_kwargs,
                 or_dir_30=None, or_dir_45=None,
                 pdh=None, pdl=None, pdc=None,
                 **extra_v9):
    """Run a single feature on prepared 1-min arrays. Returns trades list.

    arr must contain rsi_5m_curr and rsi_5m_prev (from prepare_backtest_arrays_1min
    or manually mapped via map_5min_rsi_to_1min).
    """
    rsi_buy  = extra_v9.get('rsi_buy',  V9_PARAMS['rsi_buy'])
    rsi_sell = extra_v9.get('rsi_sell', V9_PARAMS['rsi_sell'])
    sm_threshold = extra_v9.get('sm_threshold', V9_PARAMS['sm_threshold'])
    cooldown = extra_v9.get('cooldown', V9_PARAMS['cooldown'])
    use_rsi_cross = extra_v9.get('use_rsi_cross', V9_PARAMS['use_rsi_cross'])

    kw = dict(
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        use_rsi_cross=use_rsi_cross,
        rsi_5m_curr=arr.get('rsi_5m_curr'),
        rsi_5m_prev=arr.get('rsi_5m_prev'),
    )

    # Merge feature-specific kwargs
    kw.update(feature_kwargs)

    # Wire up preprocessed arrays for special features
    if feature_name == 'or_align_30' and or_dir_30 is not None:
        kw['or_direction'] = or_dir_30
    elif feature_name == 'or_align_45' and or_dir_45 is not None:
        kw['or_direction'] = or_dir_45

    if feature_name == 'vwap_filter' and 'vwap' in arr:
        kw['vwap'] = arr['vwap']

    if feature_name in ('prior_day_50', 'prior_day_75'):
        if pdh is not None:
            kw['prior_day_high'] = pdh
            kw['prior_day_low'] = pdl

    return run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], arr['rsi'], arr['times'], **kw,
    )


def _precompute_indicators(arr):
    """Precompute OR direction, prior day levels for a given array set."""
    or_dir_30 = compute_opening_range(arr['times'], arr['highs'], arr['lows'], 30)
    or_dir_45 = compute_opening_range(arr['times'], arr['highs'], arr['lows'], 45)
    pdh, pdl, pdc = compute_prior_day_levels(
        arr['times'], arr['highs'], arr['lows'], arr['closes'])
    return or_dir_30, or_dir_45, pdh, pdl, pdc


def _slice_df(df, start, end):
    """Slice a DataFrame by timestamp range."""
    return df[(df.index >= start) & (df.index < end)]


# ---------------------------------------------------------------------------
# Section 1: Feature Performance
# ---------------------------------------------------------------------------

def run_feature_performance(arr, baseline_trades, baseline_sc,
                            or_dir_30, or_dir_45, pdh, pdl, pdc):
    """Run each feature on MNQ 1-min, compute PF delta, CI, perm test."""
    print("\n" + "=" * 80)
    print("SECTION 1: FEATURE PERFORMANCE (MNQ 1-min)")
    print("=" * 80)
    print(f"\nBaseline: {fmt_score(baseline_sc, 'v9')}")

    results = {}
    for fname, fkw in FEATURES.items():
        trades = _run_feature(
            arr, fname, fkw,
            or_dir_30=or_dir_30, or_dir_45=or_dir_45,
            pdh=pdh, pdl=pdl, pdc=pdc,
        )
        sc = score_trades(trades)

        pf_base = baseline_sc['pf'] if baseline_sc else 0
        pf_feat = sc['pf'] if sc else 0
        pf_delta = round(pf_feat - pf_base, 3)

        # Bootstrap CI on PF delta
        if sc and sc['count'] >= 3:
            _, ci_lo_raw, ci_hi_raw = bootstrap_ci(
                trades, metric="pf", n_boot=N_BOOT)
            ci_lo = round(ci_lo_raw - pf_base, 3)
            ci_hi = round(ci_hi_raw - pf_base, 3)
            ci_excl_zero = ci_lo > 0
        else:
            ci_lo, ci_hi = 0.0, 0.0
            ci_excl_zero = False

        # Permutation test
        if trades and baseline_trades:
            _, perm_p = permutation_test(
                baseline_trades, trades, metric="pf", n_perms=N_PERMS)
        else:
            perm_p = 1.0

        results[fname] = {
            'sc': sc, 'trades': trades,
            'pf_delta': pf_delta,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'ci_excl_zero': ci_excl_zero,
            'perm_p': perm_p,
        }

        ct = sc['count'] if sc else 0
        wr = sc['win_rate'] if sc else 0
        pf = sc['pf'] if sc else 0
        net = sc['net_dollar'] if sc else 0
        dd = sc['max_dd_dollar'] if sc else 0
        exc = "YES" if ci_excl_zero else "NO"
        print(f"  {fname:20s}  {ct:3d} trades  WR {wr:5.1f}%  PF {pf:6.3f}  "
              f"dPF {pf_delta:+.3f}  CI [{ci_lo:+.3f},{ci_hi:+.3f}] excl0={exc}  "
              f"perm_p={perm_p:.4f}  Net ${net:+.2f}  DD ${dd:.2f}")

    return results


# ---------------------------------------------------------------------------
# Section 2: Walk-Forward
# ---------------------------------------------------------------------------

def run_walk_forward(df_1m_full):
    """In-sample / Out-of-sample walk-forward on 1-min data."""
    print("\n" + "=" * 80)
    print("SECTION 2: WALK-FORWARD VALIDATION")
    print(f"  IS: {IS_START.date()} to {IS_END.date()}")
    print(f"  OOS: {OOS_START.date()} to {OOS_END.date()}")
    print("=" * 80)

    results = {}

    for label, start, end, tag in [
        ("IS", IS_START, IS_END, "is"),
        ("OOS", OOS_START, OOS_END, "oos"),
    ]:
        df_slice = _slice_df(df_1m_full, start, end)
        if len(df_slice) < 50:
            print(f"  SKIP {label}: only {len(df_slice)} bars")
            continue

        arr = prepare_backtest_arrays_1min(df_slice, V9_PARAMS['rsi_len'])
        or_30, or_45, pdh, pdl, pdc = _precompute_indicators(arr)

        for fname, fkw in FEATURES.items():
            trades = _run_feature(
                arr, fname, fkw,
                or_dir_30=or_30, or_dir_45=or_45,
                pdh=pdh, pdl=pdl, pdc=pdc,
            )
            sc = score_trades(trades)
            pf = sc['pf'] if sc else 0
            if fname not in results:
                results[fname] = {}
            results[fname][f'{tag}_pf'] = pf
            results[fname][f'{tag}_count'] = sc['count'] if sc else 0

    # Compute degradation and pass/fail
    print(f"\n  {'Feature':20s}  {'IS PF':>8s}  {'OOS PF':>8s}  {'Degrad%':>8s}  {'Result':>8s}")
    print("  " + "-" * 60)
    for fname in FEATURES:
        r = results.get(fname, {})
        is_pf = r.get('is_pf', 0)
        oos_pf = r.get('oos_pf', 0)
        if is_pf > 0:
            degradation = (is_pf - oos_pf) / is_pf
        else:
            degradation = 1.0
        passed = (oos_pf >= WF_MIN_OOS_PF and degradation < WF_MAX_DEGRADATION)
        r['degradation'] = round(degradation, 3)
        r['pass'] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  {fname:20s}  {is_pf:8.3f}  {oos_pf:8.3f}  "
              f"{degradation * 100:7.1f}%  {status:>8s}")

    return results


# ---------------------------------------------------------------------------
# Section 3: Cross-Instrument
# ---------------------------------------------------------------------------

def run_cross_instrument():
    """Test each feature on MNQ, MES, ES, MYM using 1-min bars."""
    print("\n" + "=" * 80)
    print("SECTION 3: CROSS-INSTRUMENT VALIDATION")
    print("=" * 80)

    # MNQ avg daily range for scaling point-based params
    mnq_1m = load_instrument_1min('MNQ')
    mnq_arr = prepare_backtest_arrays_1min(mnq_1m, V9_PARAMS['rsi_len'])
    daily_ranges = {}
    for i in range(len(mnq_arr['times'])):
        d = pd.Timestamp(mnq_arr['times'][i]).date()
        if d not in daily_ranges:
            daily_ranges[d] = (mnq_arr['highs'][i], mnq_arr['lows'][i])
        else:
            h, l = daily_ranges[d]
            daily_ranges[d] = (max(h, mnq_arr['highs'][i]),
                               min(l, mnq_arr['lows'][i]))
    mnq_avg_range = np.mean([h - l for h, l in daily_ranges.values()])

    results = {}

    for inst in INSTRUMENTS:
        cfg = INSTRUMENTS[inst]
        dollar_per_pt = cfg['dollar_per_pt']
        commission = cfg['commission']

        try:
            df_1m = load_instrument_1min(inst)
        except Exception as e:
            print(f"  SKIP {inst}: {e}")
            continue

        df_1m = _slice_df(df_1m, FULL_START, FULL_END)
        if len(df_1m) < 250:
            print(f"  SKIP {inst}: only {len(df_1m)} bars")
            continue

        arr = prepare_backtest_arrays_1min(df_1m, V9_PARAMS['rsi_len'])
        or_30, or_45, pdh, pdl, pdc = _precompute_indicators(arr)

        # Compute avg daily range for scaling
        inst_daily = {}
        for j in range(len(arr['times'])):
            d = pd.Timestamp(arr['times'][j]).date()
            if d not in inst_daily:
                inst_daily[d] = (arr['highs'][j], arr['lows'][j])
            else:
                h, l = inst_daily[d]
                inst_daily[d] = (max(h, arr['highs'][j]),
                                 min(l, arr['lows'][j]))
        inst_avg_range = np.mean([h - l for h, l in inst_daily.values()])
        scale = inst_avg_range / mnq_avg_range if mnq_avg_range > 0 else 1.0

        # Baseline
        base_trades = run_v9_baseline(arr, **V9_PARAMS)
        base_sc = score_trades(base_trades, commission_per_side=commission,
                               dollar_per_pt=dollar_per_pt)

        print(f"\n  {inst} ({len(df_1m)} bars, scale={scale:.3f})  "
              f"Baseline: {base_sc['count'] if base_sc else 0} trades, "
              f"PF {base_sc['pf'] if base_sc else 0:.3f}")

        for fname, fkw in FEATURES.items():
            # Scale point-based params
            scaled_kw = dict(fkw)
            if 'prior_day_buffer_pts' in scaled_kw:
                scaled_kw['prior_day_buffer_pts'] = round(
                    scaled_kw['prior_day_buffer_pts'] * scale)

            trades = _run_feature(
                arr, fname, scaled_kw,
                or_dir_30=or_30, or_dir_45=or_45,
                pdh=pdh, pdl=pdl, pdc=pdc,
            )
            sc = score_trades(trades, commission_per_side=commission,
                              dollar_per_pt=dollar_per_pt)
            pf = sc['pf'] if sc else 0

            if fname not in results:
                results[fname] = {}
            results[fname][inst] = pf

    # Determine pass/fail
    others = [k for k in INSTRUMENTS if k != 'MNQ']
    print(f"\n  {'Feature':20s}", end="")
    for inst in INSTRUMENTS:
        print(f"  {inst:>8s}", end="")
    print(f"  {'Others':>8s}  {'Result':>8s}")
    print("  " + "-" * 80)

    for fname in FEATURES:
        r = results.get(fname, {})
        mnq_pf = r.get('MNQ', 0)
        other_ok = sum(1 for o in others if r.get(o, 0) > 1.0)
        passed = mnq_pf > 1.0 and other_ok >= 2
        r['other_ok'] = other_ok
        r['pass'] = passed
        status = "PASS" if passed else "FAIL"

        print(f"  {fname:20s}", end="")
        for inst in INSTRUMENTS:
            pf = r.get(inst, 0)
            print(f"  {pf:8.3f}", end="")
        print(f"  {other_ok:>5d}/3  {status:>8s}")

    return results


# ---------------------------------------------------------------------------
# Section 4: Parameter Sensitivity
# ---------------------------------------------------------------------------

def run_sensitivity(df_1m_full):
    """Sweep RSI length, levels, threshold for each feature on 1-min data.

    Uses map_5min_rsi_to_1min() to efficiently recompute RSI mapping
    for each rsi_len without full data reload.
    """
    print("\n" + "=" * 80)
    print("SECTION 4: PARAMETER SENSITIVITY")
    print(f"  RSI lengths: {SENS_RSI_LENS}")
    print(f"  RSI levels:  {SENS_RSI_LEVELS}")
    print(f"  Thresholds:  {SENS_THRESHOLDS}")
    print("=" * 80)

    # Prepare base 1-min arrays once (RSI will be remapped per rsi_len)
    arr_base = prepare_backtest_arrays_1min(df_1m_full, rsi_len=10)
    or_30, or_45, pdh, pdl, pdc = _precompute_indicators(arr_base)

    # Pre-resample to 5-min once for efficient RSI remapping
    df_5m = resample_to_5min(df_1m_full)
    fivemin_times = df_5m.index.values
    fivemin_closes = df_5m['Close'].values
    onemin_times = arr_base['times']

    results = {}

    for fname, fkw in FEATURES.items():
        pfs = []
        total_variants = 0
        profitable_variants = 0

        for rsi_len in SENS_RSI_LENS:
            # Remap RSI for this length
            rsi_curr, rsi_prev = map_5min_rsi_to_1min(
                onemin_times, fivemin_times, fivemin_closes, rsi_len)

            # Build arr with remapped RSI
            arr = dict(arr_base)
            arr['rsi'] = rsi_curr
            arr['rsi_5m_curr'] = rsi_curr
            arr['rsi_5m_prev'] = rsi_prev

            for (buy, sell) in SENS_RSI_LEVELS:
                for thresh in SENS_THRESHOLDS:
                    trades = _run_feature(
                        arr, fname, fkw,
                        or_dir_30=or_30, or_dir_45=or_45,
                        pdh=pdh, pdl=pdl, pdc=pdc,
                        rsi_buy=buy, rsi_sell=sell,
                        sm_threshold=thresh,
                    )
                    sc = score_trades(trades)
                    pf = sc['pf'] if sc else 0
                    pfs.append(pf)
                    total_variants += 1
                    if pf > 1.0:
                        profitable_variants += 1

        stability = profitable_variants / total_variants if total_variants > 0 else 0
        pf_std = float(np.std(pfs)) if pfs else 999.0
        passed = stability >= SENS_STABILITY_MIN and pf_std < SENS_PF_STD_MAX

        results[fname] = {
            'stability_pct': round(stability * 100, 1),
            'pf_std': round(pf_std, 3),
            'pf_mean': round(float(np.mean(pfs)), 3) if pfs else 0,
            'n_variants': total_variants,
            'n_profitable': profitable_variants,
            'pass': passed,
        }

        tag = "Stable" if passed else "Fragile"
        print(f"  {fname:20s}  {profitable_variants:2d}/{total_variants:2d} profitable "
              f"({stability * 100:5.1f}%)  PF std={pf_std:.3f}  mean={results[fname]['pf_mean']:.3f}  "
              f"-> {tag}")

    return results


# ---------------------------------------------------------------------------
# Section 5: Overfitting Checks
# ---------------------------------------------------------------------------

def run_overfitting(arr, baseline_trades, baseline_sc,
                    or_dir_30, or_dir_45, pdh, pdl, pdc,
                    df_1m_full):
    """Lucky trade removal, random feature baseline, temporal stability."""
    print("\n" + "=" * 80)
    print("SECTION 5: OVERFITTING CHECKS")
    print("=" * 80)

    results = {}

    for fname, fkw in FEATURES.items():
        # --- A) Lucky trade removal ---
        trades = _run_feature(
            arr, fname, fkw,
            or_dir_30=or_dir_30, or_dir_45=or_dir_45,
            pdh=pdh, pdl=pdl, pdc=pdc,
        )
        sc = score_trades(trades)

        lucky_pass = False
        if trades and len(trades) >= 5:
            pts_list = [(t['pts'], idx) for idx, t in enumerate(trades)]
            pts_list.sort(key=lambda x: x[0], reverse=True)
            all_pass = True
            for n_remove in [1, 2, 3]:
                remove_idxs = set(idx for _, idx in pts_list[:n_remove])
                remaining = [t for idx, t in enumerate(trades) if idx not in remove_idxs]
                sc_r = score_trades(remaining)
                if sc_r is None or sc_r['pf'] < LUCKY_REMOVAL_PF_MIN:
                    all_pass = False
                    break
            lucky_pass = all_pass

        # --- B) Random feature baseline ---
        random_pass = False
        if baseline_trades and len(baseline_trades) >= 5:
            rand_pfs = random_feature_baseline(
                baseline_trades, n_sims=100)
            p95 = np.percentile(rand_pfs, 95) if len(rand_pfs) > 0 else 999
            feat_pf = sc['pf'] if sc else 0
            random_pass = feat_pf > p95

        # --- C) Temporal stability (on 1-min data) ---
        temporal_profitable = 0
        for (ws, we) in TEMPORAL_WINDOWS:
            df_w = _slice_df(df_1m_full, ws, we)
            if len(df_w) < 50:
                continue
            w_arr = prepare_backtest_arrays_1min(df_w, V9_PARAMS['rsi_len'])
            w_or30, w_or45, w_pdh, w_pdl, w_pdc = _precompute_indicators(w_arr)
            w_trades = _run_feature(
                w_arr, fname, fkw,
                or_dir_30=w_or30, or_dir_45=w_or45,
                pdh=w_pdh, pdl=w_pdl, pdc=w_pdc,
            )
            w_sc = score_trades(w_trades)
            if w_sc and w_sc['net_pts'] > 0:
                temporal_profitable += 1

        temporal_pass = temporal_profitable >= 3

        overall = lucky_pass and random_pass and temporal_pass
        results[fname] = {
            'lucky_pass': lucky_pass,
            'random_pass': random_pass,
            'temporal_windows': temporal_profitable,
            'temporal_pass': temporal_pass,
            'pass': overall,
        }

        tag = "Clean" if overall else "Flag"
        print(f"  {fname:20s}  Lucky={'PASS' if lucky_pass else 'FAIL'}  "
              f"Random={'PASS' if random_pass else 'FAIL'}  "
              f"Temporal={temporal_profitable}/4  -> {tag}")

    return results


# ---------------------------------------------------------------------------
# Section 6: FINAL DECISION MATRIX
# ---------------------------------------------------------------------------

def build_decision_matrix(perf, wf, cross, sens, overfit, baseline_sc):
    """Aggregate results into a final decision matrix."""
    print("\n" + "=" * 80)
    print("FINAL DECISION MATRIX -- v10 Feature Candidates (1-MIN ENGINE)")
    print("=" * 80)

    header = (f"  {'FEATURE':20s} | {'PF Delta':>9s} | {'CI excl 0?':>10s} | "
              f"{'Walk-Fwd':>8s} | {'Cross-Inst':>10s} | {'Sensitivity':>11s} | "
              f"{'Overfit':>8s} | {'VERDICT':>8s}")
    sep = "  " + "-" * len(header)

    print(header)
    print(sep)

    decisions = {}

    for fname in FEATURES:
        p = perf.get(fname, {})
        ci_excl = p.get('ci_excl_zero', False)
        pf_delta = p.get('pf_delta', 0)

        w = wf.get(fname, {})
        wf_pass = w.get('pass', False)

        c = cross.get(fname, {})
        cross_pass = c.get('pass', False)
        other_ok = c.get('other_ok', 0)

        s = sens.get(fname, {})
        sens_pass = s.get('pass', False)

        o = overfit.get(fname, {})
        random_pass = o.get('random_pass', False)
        lucky_pass = o.get('lucky_pass', False)

        sc = p.get('sc', None)
        count_ok = sc is not None and sc['count'] >= MIN_TRADES

        criteria = [ci_excl, wf_pass, cross_pass, sens_pass,
                    random_pass, lucky_pass, count_ok]
        score = sum(criteria)

        if score >= ADOPT_THRESHOLD:
            verdict = "ADOPT"
        elif score >= MAYBE_THRESHOLD:
            verdict = "MAYBE"
        else:
            verdict = "REJECT"

        if pf_delta < 0:
            verdict = "REJECT"

        ci_str  = "YES" if ci_excl else "NO"
        wf_str  = "PASS" if wf_pass else "FAIL"
        cr_str  = f"{other_ok}/3"
        se_str  = "Stable" if sens_pass else "Fragile"
        ov_str  = "Clean" if (random_pass and lucky_pass) else "Flag"

        print(f"  {fname:20s} | {pf_delta:+9.3f} | {ci_str:>10s} | "
              f"{wf_str:>8s} | {cr_str:>10s} | {se_str:>11s} | "
              f"{ov_str:>8s} | {verdict:>8s}")

        decisions[fname] = {
            'pf_delta': pf_delta,
            'ci_excl_zero': ci_excl,
            'wf_pass': wf_pass,
            'cross_pass': cross_pass,
            'cross_others': other_ok,
            'sens_pass': sens_pass,
            'overfit_pass': random_pass and lucky_pass,
            'count_ok': count_ok,
            'score': score,
            'verdict': verdict,
        }

    print(sep)
    print()

    adopted = [f for f, d in decisions.items() if d['verdict'] == 'ADOPT']
    maybe   = [f for f, d in decisions.items() if d['verdict'] == 'MAYBE']
    rejected = [f for f, d in decisions.items() if d['verdict'] == 'REJECT']

    print("  ADOPTION CRITERIA (must pass ALL 7):")
    print("    1. PF improvement with 95% CI excluding zero")
    print("    2. Walk-forward OOS PF >= 1.0, degradation < 30%")
    print("    3. Profitable on MNQ + 2 of 3 other instruments")
    print("    4. Stable parameter sensitivity (>70% variants profitable, PF std < 0.5)")
    print("    5. Beats random feature baseline (PF > P95 of random)")
    print("    6. Survives lucky trade removal (PF > 1.3 after removing top 3)")
    print("    7. Trade count >= 20")
    print()
    print(f"  ADOPT  ({len(adopted)}): {', '.join(adopted) if adopted else 'None'}")
    print(f"  MAYBE  ({len(maybe)}):  {', '.join(maybe) if maybe else 'None'}")
    print(f"  REJECT ({len(rejected)}): {', '.join(rejected) if rejected else 'None'}")
    print()

    return decisions


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 80)
    print("v10 FEATURE VALIDATION -- FINAL REPORT (1-MIN ENGINE)")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # --- Load MNQ 1-min data (primary dataset) ---
    print("\nLoading MNQ 1-min data...")
    df_1m_full = load_instrument_1min('MNQ')
    compute_date_splits(df_1m_full)
    print(f"  {len(df_1m_full)} 1-min bars, {df_1m_full.index[0].date()} to "
          f"{df_1m_full.index[-1].date()}")

    # Prepare 1-min arrays with mapped 5-min RSI
    print("  Preparing 1-min arrays with mapped 5-min RSI...")
    arr = prepare_backtest_arrays_1min(df_1m_full, V9_PARAMS['rsi_len'])
    or_dir_30, or_dir_45, pdh, pdl, pdc = _precompute_indicators(arr)

    # --- v9 Baseline ---
    print("\nRunning v9 baseline on 1-min bars (cooldown=15)...")
    baseline_trades = run_v9_baseline(arr, **V9_PARAMS)
    baseline_sc = score_trades(baseline_trades)
    print(f"  {fmt_score(baseline_sc, 'v9 Baseline')}")
    if baseline_sc:
        print(f"  Exits: {fmt_exits(baseline_sc['exits'])}")

    if baseline_sc is None or baseline_sc['count'] < 10:
        print("\nERROR: Baseline has too few trades. Aborting.")
        sys.exit(1)

    # --- Section 1: Feature Performance ---
    t1 = time.time()
    perf = run_feature_performance(
        arr, baseline_trades, baseline_sc,
        or_dir_30, or_dir_45, pdh, pdl, pdc,
    )
    print(f"\n  [Section 1 complete in {time.time() - t1:.1f}s]")

    # --- Section 2: Walk-Forward ---
    t2 = time.time()
    wf = run_walk_forward(df_1m_full)
    print(f"\n  [Section 2 complete in {time.time() - t2:.1f}s]")

    # --- Section 3: Cross-Instrument ---
    t3 = time.time()
    cross = run_cross_instrument()
    print(f"\n  [Section 3 complete in {time.time() - t3:.1f}s]")

    # --- Section 4: Parameter Sensitivity ---
    t4 = time.time()
    sens = run_sensitivity(df_1m_full)
    print(f"\n  [Section 4 complete in {time.time() - t4:.1f}s]")

    # --- Section 5: Overfitting Checks ---
    t5 = time.time()
    overfit = run_overfitting(
        arr, baseline_trades, baseline_sc,
        or_dir_30, or_dir_45, pdh, pdl, pdc,
        df_1m_full,
    )
    print(f"\n  [Section 5 complete in {time.time() - t5:.1f}s]")

    # --- Section 6: Final Decision Matrix ---
    decisions = build_decision_matrix(perf, wf, cross, sens, overfit, baseline_sc)

    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 80)

    return decisions


if __name__ == "__main__":
    decisions = main()
