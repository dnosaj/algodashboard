"""
v10 Feature Validation Suite -- File 2
=======================================
Tests 7 individual features against the v9 baseline using bootstrap CI
and permutation tests. Runs on MNQ 1-min (resampled to 5-min) as primary
dataset, and MNQ 5-min prebaked (55-day) for features that don't need VWAP.

Usage:
    python3 v10_test_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from v10_test_common import (
    load_instrument_1min, load_mnq_5min_prebaked, resample_to_5min,
    compute_rsi, compute_opening_range, compute_prior_day_levels,
    run_backtest_v10, score_trades, bootstrap_ci, permutation_test,
    prepare_backtest_arrays, run_v9_baseline, fmt_score, fmt_exits,
    DATA_DIR, INSTRUMENTS, NY_OPEN_UTC, NY_LAST_ENTRY_UTC, NY_CLOSE_UTC,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMISSION = 0.52     # MNQ commission per side
DOLLAR_PT = 2.0       # MNQ dollar per point
RSI_LEN = 10
RSI_BUY = 55
RSI_SELL = 45
SM_THRESHOLD = 0.0
COOLDOWN = 3
USE_RSI_CROSS = True

N_BOOT = 10000
N_PERMS = 10000
CI_LEVEL = 0.95

# Separator widths
SEP = "=" * 100
THIN = "-" * 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_primary_dataset():
    """Load MNQ 1-min -> resample to 5-min. Returns (df_5m, arr dict)."""
    print("Loading MNQ 1-min data (b2119)...")
    df_1m = load_instrument_1min('MNQ')
    df_5m = resample_to_5min(df_1m)
    print(f"  1-min bars: {len(df_1m)}, 5-min bars: {len(df_5m)}")
    print(f"  Date range: {df_5m.index[0].date()} to {df_5m.index[-1].date()}")
    arr = prepare_backtest_arrays(df_5m)
    return df_5m, arr


def load_prebaked_dataset():
    """Load MNQ 5-min prebaked (55-day, Nov 25 - Feb 12). Returns (df_5m, arr dict)."""
    print("Loading MNQ 5-min prebaked (55-day)...")
    df_5m = load_mnq_5min_prebaked()
    print(f"  5-min bars: {len(df_5m)}")
    print(f"  Date range: {df_5m.index[0].date()} to {df_5m.index[-1].date()}")
    arr = prepare_backtest_arrays(df_5m)
    return df_5m, arr


def get_baseline_trades(arr):
    """Run v9 baseline with standard params."""
    return run_v9_baseline(
        arr, rsi_len=RSI_LEN, rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=SM_THRESHOLD, cooldown=COOLDOWN, max_loss_pts=0,
        use_rsi_cross=USE_RSI_CROSS,
    )


def run_feature_variant(arr, rsi, **feature_kwargs):
    """Run v10 engine with a feature enabled. Passes standard v9 params + feature."""
    return run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], rsi, arr['times'],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=SM_THRESHOLD, cooldown_bars=COOLDOWN,
        max_loss_pts=0, use_rsi_cross=USE_RSI_CROSS,
        **feature_kwargs,
    )


def compute_delta(baseline_val, feature_val):
    """Compute delta between two values."""
    if baseline_val is None or feature_val is None:
        return 0.0
    return feature_val - baseline_val


def significance_marker(ci_lo, ci_hi):
    """Return 'YES' if CI doesn't cross zero, else 'No'."""
    if ci_lo > 0 or ci_hi < 0:
        return "YES"
    return "No"


def print_comparison_header():
    """Print the header for feature comparison tables."""
    print(f"  {'Metric':<18} {'Baseline':>12} {'Feature':>12} {'Delta':>12} {'95% CI':>22} {'Sig?':>6}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*12} {'-'*22} {'-'*6}")


def print_comparison_row(metric_name, base_val, feat_val, delta, ci_lo, ci_hi, sig):
    """Print a single comparison row."""
    if isinstance(base_val, int) and isinstance(feat_val, int):
        print(f"  {metric_name:<18} {base_val:>12d} {feat_val:>12d} {delta:>+12d} [{ci_lo:>+9.1f}, {ci_hi:>+9.1f}] {sig:>6}")
    else:
        print(f"  {metric_name:<18} {base_val:>12.3f} {feat_val:>12.3f} {delta:>+12.3f} [{ci_lo:>+9.3f}, {ci_hi:>+9.3f}] {sig:>6}")


def run_full_comparison(label, base_trades, feat_trades, base_sc, feat_sc):
    """Run full comparison with bootstrap CI and permutation test for one variant."""
    print(f"\nFEATURE: {label}")

    if feat_sc is None:
        print("  ** No trades produced -- feature too restrictive **")
        return None

    print_comparison_header()

    # Trades (simple count, no CI needed)
    dt = feat_sc['count'] - base_sc['count']
    print(f"  {'Trades':<18} {base_sc['count']:>12d} {feat_sc['count']:>12d} {dt:>+12d} {'N/A':>22} {'--':>6}")

    # Win Rate
    b_wr = base_sc['win_rate']
    f_wr = feat_sc['win_rate']
    _, bci_lo, bci_hi = bootstrap_ci(feat_trades, 'win_rate', N_BOOT, CI_LEVEL, COMMISSION, DOLLAR_PT)
    _, bbci_lo, bbci_hi = bootstrap_ci(base_trades, 'win_rate', N_BOOT, CI_LEVEL, COMMISSION, DOLLAR_PT)
    dwr = f_wr - b_wr
    # CI on the difference (approximation: use feature CI spread relative to point)
    wr_ci_lo = bci_lo - b_wr
    wr_ci_hi = bci_hi - b_wr
    sig = significance_marker(wr_ci_lo, wr_ci_hi)
    print(f"  {'Win Rate %':<18} {b_wr:>12.1f} {f_wr:>12.1f} {dwr:>+12.1f} [{wr_ci_lo:>+9.1f}, {wr_ci_hi:>+9.1f}] {sig:>6}")

    # Profit Factor
    b_pf = base_sc['pf']
    f_pf = feat_sc['pf']
    _, pf_ci_lo, pf_ci_hi = bootstrap_ci(feat_trades, 'pf', N_BOOT, CI_LEVEL, COMMISSION, DOLLAR_PT)
    dpf = f_pf - b_pf
    pf_dci_lo = pf_ci_lo - b_pf
    pf_dci_hi = pf_ci_hi - b_pf
    pf_sig = significance_marker(pf_dci_lo, pf_dci_hi)
    print(f"  {'Profit Factor':<18} {b_pf:>12.3f} {f_pf:>12.3f} {dpf:>+12.3f} [{pf_dci_lo:>+9.3f}, {pf_dci_hi:>+9.3f}] {pf_sig:>6}")

    # Permutation test for PF
    perm_diff, perm_p = permutation_test(base_trades, feat_trades, 'pf', N_PERMS, COMMISSION, DOLLAR_PT)
    print(f"  {'  Permutation PF':<18} {'':>12} {'':>12} {perm_diff:>+12.3f} {'p=':>12}{perm_p:<10.4f} {'YES' if perm_p < 0.05 else 'No':>6}")

    # Net P&L ($)
    b_net = base_sc['net_dollar']
    f_net = feat_sc['net_dollar']
    _, net_ci_lo, net_ci_hi = bootstrap_ci(feat_trades, 'net_pts', N_BOOT, CI_LEVEL, COMMISSION, DOLLAR_PT)
    net_ci_lo_d = net_ci_lo * DOLLAR_PT
    net_ci_hi_d = net_ci_hi * DOLLAR_PT
    dnet = f_net - b_net
    net_dci_lo = net_ci_lo_d - b_net
    net_dci_hi = net_ci_hi_d - b_net
    net_sig = significance_marker(net_dci_lo, net_dci_hi)
    print(f"  {'Net P&L ($)':<18} {b_net:>12.2f} {f_net:>12.2f} {dnet:>+12.2f} [{net_dci_lo:>+9.2f}, {net_dci_hi:>+9.2f}] {net_sig:>6}")

    # Max DD ($)
    b_dd = base_sc['max_dd_dollar']
    f_dd = feat_sc['max_dd_dollar']
    _, dd_ci_lo, dd_ci_hi = bootstrap_ci(feat_trades, 'max_dd_pts', N_BOOT, CI_LEVEL, COMMISSION, DOLLAR_PT)
    dd_ci_lo_d = dd_ci_lo * DOLLAR_PT
    dd_ci_hi_d = dd_ci_hi * DOLLAR_PT
    ddd = f_dd - b_dd
    dd_dci_lo = dd_ci_lo_d - b_dd
    dd_dci_hi = dd_ci_hi_d - b_dd
    dd_sig = significance_marker(dd_dci_lo, dd_dci_hi)
    print(f"  {'Max DD ($)':<18} {b_dd:>12.2f} {f_dd:>12.2f} {ddd:>+12.2f} [{dd_dci_lo:>+9.2f}, {dd_dci_hi:>+9.2f}] {dd_sig:>6}")

    # Sharpe
    b_sh = base_sc['sharpe']
    f_sh = feat_sc['sharpe']
    _, sh_ci_lo, sh_ci_hi = bootstrap_ci(feat_trades, 'sharpe', N_BOOT, CI_LEVEL, COMMISSION, DOLLAR_PT)
    dsh = f_sh - b_sh
    sh_dci_lo = sh_ci_lo - b_sh
    sh_dci_hi = sh_ci_hi - b_sh
    sh_sig = significance_marker(sh_dci_lo, sh_dci_hi)
    print(f"  {'Sharpe':<18} {b_sh:>12.3f} {f_sh:>12.3f} {dsh:>+12.3f} [{sh_dci_lo:>+9.3f}, {sh_dci_hi:>+9.3f}] {sh_sig:>6}")

    # Exits breakdown
    print(f"  Exits: {fmt_exits(feat_sc['exits'])}")

    return {
        'label': label,
        'trades': feat_sc['count'],
        'wr': f_wr,
        'pf': f_pf,
        'net': f_net,
        'dd': f_dd,
        'sharpe': f_sh,
        'perm_p': perm_p,
        'dpf': dpf,
        'dnet': dnet,
    }


# ===========================================================================
# T1: Underwater Exit
# ===========================================================================

def test_underwater_exit(arr, rsi, base_trades, base_sc, dataset_label):
    """Test underwater exit with bars=[2,3,4,5,6]."""
    print(f"\n{SEP}")
    print(f"T1: UNDERWATER EXIT  [{dataset_label}]")
    print(SEP)
    print(f"  Baseline: {fmt_score(base_sc, 'v9')}")

    variants = [2, 3, 4, 5, 6]
    results = []

    for bars in variants:
        trades = run_feature_variant(arr, rsi, underwater_exit_bars=bars)
        sc = score_trades(trades, COMMISSION, DOLLAR_PT)
        res = run_full_comparison(
            f"Underwater Exit (bars={bars})",
            base_trades, trades, base_sc, sc,
        )
        if res:
            results.append(res)

    # T1 Extra: Bars-held-while-underwater analysis for baseline trades
    print(f"\n{THIN}")
    print("T1 EXTRA: Bars-Held-While-Underwater Analysis (Baseline Trades)")
    print(THIN)

    comm_pts = (COMMISSION * 2) / DOLLAR_PT
    uw_winners = []
    uw_losers = []

    for t in base_trades:
        ei = t['entry_idx']
        xi = t['exit_idx']
        ep = t['entry']
        side = t['side']
        net_pts = t['pts'] - comm_pts

        uw_count = 0
        for j in range(ei, xi + 1):
            if side == "long":
                if arr['closes'][j] < ep:
                    uw_count += 1
            else:
                if arr['closes'][j] > ep:
                    uw_count += 1

        if net_pts > 0:
            uw_winners.append(uw_count)
        else:
            uw_losers.append(uw_count)

    if uw_winners:
        w_arr = np.array(uw_winners)
        print(f"  Winners ({len(w_arr)} trades): avg underwater bars = {w_arr.mean():.1f}, "
              f"median = {np.median(w_arr):.0f}, max = {w_arr.max()}")
        # Histogram
        bins = [0, 1, 2, 3, 4, 5, 10, 50]
        hist, _ = np.histogram(w_arr, bins=bins)
        print(f"    Histogram: ", end="")
        for k in range(len(hist)):
            lo_b = bins[k]
            hi_b = bins[k + 1] - 1 if k < len(hist) - 1 else bins[k + 1]
            if lo_b == hi_b:
                print(f"[{lo_b}]:{hist[k]}  ", end="")
            else:
                print(f"[{lo_b}-{hi_b}]:{hist[k]}  ", end="")
        print()

    if uw_losers:
        l_arr = np.array(uw_losers)
        print(f"  Losers  ({len(l_arr)} trades): avg underwater bars = {l_arr.mean():.1f}, "
              f"median = {np.median(l_arr):.0f}, max = {l_arr.max()}")
        bins = [0, 1, 2, 3, 4, 5, 10, 50]
        hist, _ = np.histogram(l_arr, bins=bins)
        print(f"    Histogram: ", end="")
        for k in range(len(hist)):
            lo_b = bins[k]
            hi_b = bins[k + 1] - 1 if k < len(hist) - 1 else bins[k + 1]
            if lo_b == hi_b:
                print(f"[{lo_b}]:{hist[k]}  ", end="")
            else:
                print(f"[{lo_b}-{hi_b}]:{hist[k]}  ", end="")
        print()

    return results


# ===========================================================================
# T2: Opening Range Alignment
# ===========================================================================

def test_opening_range(arr, rsi, base_trades, base_sc, dataset_label):
    """Test opening range alignment filter with minutes=[15,30,45,60]."""
    print(f"\n{SEP}")
    print(f"T2: OPENING RANGE ALIGNMENT  [{dataset_label}]")
    print(SEP)
    print(f"  Baseline: {fmt_score(base_sc, 'v9')}")

    variants = [15, 30, 45, 60]
    results = []

    for mins in variants:
        or_dir = compute_opening_range(arr['times'], arr['highs'], arr['lows'], or_minutes=mins)
        trades = run_feature_variant(arr, rsi, or_direction=or_dir)
        sc = score_trades(trades, COMMISSION, DOLLAR_PT)
        res = run_full_comparison(
            f"Opening Range Align (OR={mins}min)",
            base_trades, trades, base_sc, sc,
        )
        if res:
            results.append(res)

    # T2 Extra: OR alignment vs counter-OR for baseline trades
    print(f"\n{THIN}")
    print("T2 EXTRA: Baseline Trades -- OR-Aligned vs Counter-OR (30-min OR)")
    print(THIN)

    or_dir_30 = compute_opening_range(arr['times'], arr['highs'], arr['lows'], or_minutes=30)
    comm_pts = (COMMISSION * 2) / DOLLAR_PT

    aligned_trades = []
    counter_trades = []
    neutral_trades = []

    for t in base_trades:
        ei = t['entry_idx']
        or_val = or_dir_30[ei - 1] if ei > 0 else 0  # use signal bar's OR direction
        side = t['side']

        if or_val == 0:
            neutral_trades.append(t)
        elif (side == "long" and or_val == 1) or (side == "short" and or_val == -1):
            aligned_trades.append(t)
        else:
            counter_trades.append(t)

    for group_label, group_trades in [("OR-Aligned", aligned_trades),
                                       ("Counter-OR", counter_trades),
                                       ("No Breakout", neutral_trades)]:
        if group_trades:
            gsc = score_trades(group_trades, COMMISSION, DOLLAR_PT)
            print(f"  {group_label}: {fmt_score(gsc, '')}")
        else:
            print(f"  {group_label}: No trades")

    return results


# ===========================================================================
# T3: VWAP Filter
# ===========================================================================

def test_vwap_filter(arr, rsi, base_trades, base_sc, dataset_label):
    """Test VWAP filter (binary + band_pts=[10,25,50])."""
    print(f"\n{SEP}")
    print(f"T3: VWAP FILTER  [{dataset_label}]")
    print(SEP)
    print(f"  Baseline: {fmt_score(base_sc, 'v9')}")

    if 'vwap' not in arr:
        print("  ** VWAP not available in this dataset -- SKIPPING **")
        return []

    results = []

    # Binary VWAP filter (no band limit)
    trades = run_feature_variant(arr, rsi, vwap_filter=True, vwap=arr['vwap'], vwap_band_pts=0)
    sc = score_trades(trades, COMMISSION, DOLLAR_PT)
    res = run_full_comparison("VWAP Filter (binary)", base_trades, trades, base_sc, sc)
    if res:
        results.append(res)

    # VWAP band variants
    for band_pts in [10, 25, 50]:
        trades = run_feature_variant(
            arr, rsi, vwap_filter=True, vwap=arr['vwap'], vwap_band_pts=band_pts,
        )
        sc = score_trades(trades, COMMISSION, DOLLAR_PT)
        res = run_full_comparison(
            f"VWAP Filter (band={band_pts}pts)",
            base_trades, trades, base_sc, sc,
        )
        if res:
            results.append(res)

    # T3 Extra: VWAP distance analysis for baseline trades
    print(f"\n{THIN}")
    print("T3 EXTRA: Baseline Trades -- Entry Distance from VWAP")
    print(THIN)

    comm_pts = (COMMISSION * 2) / DOLLAR_PT
    vwap_dists = []
    pnls = []

    for t in base_trades:
        ei = t['entry_idx']
        vw = arr['vwap'][ei - 1] if ei > 0 else arr['vwap'][ei]
        if np.isnan(vw) or vw == 0:
            continue
        entry_p = t['entry']
        side = t['side']
        # Signed distance: positive = above VWAP
        if side == "long":
            dist = entry_p - vw
        else:
            dist = vw - entry_p  # positive = below VWAP (favorable)
        vwap_dists.append(dist)
        pnls.append(t['pts'] - comm_pts)

    if vwap_dists:
        dists = np.array(vwap_dists)
        pnl_arr = np.array(pnls)
        print(f"  Avg VWAP distance at entry: {dists.mean():.1f} pts (pos = favorable side)")
        print(f"  Median VWAP distance: {np.median(dists):.1f} pts")

        # Split into quantiles
        q_labels = ["Q1 (near VWAP)", "Q2", "Q3", "Q4 (far from VWAP)"]
        try:
            quartiles = np.percentile(np.abs(dists), [25, 50, 75])
            abs_dists = np.abs(dists)
            masks = [
                abs_dists <= quartiles[0],
                (abs_dists > quartiles[0]) & (abs_dists <= quartiles[1]),
                (abs_dists > quartiles[1]) & (abs_dists <= quartiles[2]),
                abs_dists > quartiles[2],
            ]
            for ql, mask in zip(q_labels, masks):
                grp_pnl = pnl_arr[mask]
                if len(grp_pnl) > 0:
                    wr = (grp_pnl > 0).sum() / len(grp_pnl) * 100
                    net = grp_pnl.sum() * DOLLAR_PT
                    w = grp_pnl[grp_pnl > 0].sum()
                    ls = abs(grp_pnl[grp_pnl <= 0].sum())
                    pf = w / ls if ls > 0 else 999.0
                    print(f"    {ql}: {len(grp_pnl)} trades, WR {wr:.0f}%, PF {pf:.2f}, Net ${net:.2f}")
        except Exception:
            print("  (Quartile analysis not possible with current data)")

    return results


# ===========================================================================
# T4: Prior Day Levels
# ===========================================================================

def test_prior_day_levels(arr, rsi, base_trades, base_sc, dataset_label):
    """Test prior day level buffer filter with buffer=[25,50,75,100] pts."""
    print(f"\n{SEP}")
    print(f"T4: PRIOR DAY LEVELS  [{dataset_label}]")
    print(SEP)
    print(f"  Baseline: {fmt_score(base_sc, 'v9')}")

    prev_high, prev_low, prev_close = compute_prior_day_levels(
        arr['times'], arr['highs'], arr['lows'], arr['closes'],
    )

    results = []
    for buffer_pts in [25, 50, 75, 100]:
        trades = run_feature_variant(
            arr, rsi,
            prior_day_high=prev_high, prior_day_low=prev_low,
            prior_day_buffer=True, prior_day_buffer_pts=buffer_pts,
        )
        sc = score_trades(trades, COMMISSION, DOLLAR_PT)
        res = run_full_comparison(
            f"Prior Day Buffer ({buffer_pts}pts)",
            base_trades, trades, base_sc, sc,
        )
        if res:
            results.append(res)

    # T4 Extra: Distance from prior day high/low for baseline trades
    print(f"\n{THIN}")
    print("T4 EXTRA: Baseline Trades -- Distance from Prior Day High/Low")
    print(THIN)

    comm_pts = (COMMISSION * 2) / DOLLAR_PT
    dist_from_pdh = []
    dist_from_pdl = []
    pnl_list = []

    for t in base_trades:
        ei = t['entry_idx']
        if np.isnan(prev_high[ei]) or np.isnan(prev_low[ei]):
            continue
        entry_p = t['entry']
        dist_h = entry_p - prev_high[ei]
        dist_l = entry_p - prev_low[ei]
        dist_from_pdh.append(dist_h)
        dist_from_pdl.append(dist_l)
        pnl_list.append(t['pts'] - comm_pts)

    if pnl_list:
        pdh = np.array(dist_from_pdh)
        pdl = np.array(dist_from_pdl)
        pnl_arr = np.array(pnl_list)
        print(f"  Avg distance from Prev Day High: {pdh.mean():.1f} pts")
        print(f"  Avg distance from Prev Day Low:  {pdl.mean():.1f} pts")

        # Above PDH vs between vs below PDL
        above_pdh = pdh > 0
        below_pdl = pdl < 0
        between = (~above_pdh) & (~below_pdl)

        for grp_label, mask in [("Above Prev High", above_pdh),
                                 ("Between H/L", between),
                                 ("Below Prev Low", below_pdl)]:
            grp_pnl = pnl_arr[mask]
            if len(grp_pnl) > 0:
                wr = (grp_pnl > 0).sum() / len(grp_pnl) * 100
                net = grp_pnl.sum() * DOLLAR_PT
                w = grp_pnl[grp_pnl > 0].sum()
                ls = abs(grp_pnl[grp_pnl <= 0].sum())
                pf = w / ls if ls > 0 else 999.0
                print(f"    {grp_label}: {len(grp_pnl)} trades, WR {wr:.0f}%, PF {pf:.2f}, Net ${net:.2f}")
            else:
                print(f"    {grp_label}: No trades")

    return results


# ===========================================================================
# T5: Price Structure Exit
# ===========================================================================

def test_price_structure_exit(arr, rsi, base_trades, base_sc, dataset_label):
    """Test price structure exit with bars=[2,3,4,5]."""
    print(f"\n{SEP}")
    print(f"T5: PRICE STRUCTURE EXIT  [{dataset_label}]")
    print(SEP)
    print(f"  Baseline: {fmt_score(base_sc, 'v9')}")

    variants = [2, 3, 4, 5]
    results = []

    for bars in variants:
        trades = run_feature_variant(
            arr, rsi, price_structure_exit=True, price_structure_bars=bars,
        )
        sc = score_trades(trades, COMMISSION, DOLLAR_PT)
        res = run_full_comparison(
            f"Price Structure Exit (bars={bars})",
            base_trades, trades, base_sc, sc,
        )
        if res:
            results.append(res)

    return results


# ===========================================================================
# T6: SM Flip Reversal
# ===========================================================================

def test_sm_reversal(arr, rsi, base_trades, base_sc, dataset_label):
    """Test SM flip reversal entry (binary)."""
    print(f"\n{SEP}")
    print(f"T6: SM FLIP REVERSAL ENTRY  [{dataset_label}]")
    print(SEP)
    print(f"  Baseline: {fmt_score(base_sc, 'v9')}")

    trades = run_feature_variant(arr, rsi, sm_reversal_entry=True)
    sc = score_trades(trades, COMMISSION, DOLLAR_PT)
    res = run_full_comparison(
        "SM Flip Reversal (on)",
        base_trades, trades, base_sc, sc,
    )

    return [res] if res else []


# ===========================================================================
# T7: RSI Momentum Direction
# ===========================================================================

def test_rsi_momentum(arr, rsi, base_trades, base_sc, dataset_label):
    """Test RSI momentum direction filter (binary)."""
    print(f"\n{SEP}")
    print(f"T7: RSI MOMENTUM DIRECTION FILTER  [{dataset_label}]")
    print(SEP)
    print(f"  Baseline: {fmt_score(base_sc, 'v9')}")

    trades = run_feature_variant(arr, rsi, rsi_momentum_filter=True)
    sc = score_trades(trades, COMMISSION, DOLLAR_PT)
    res = run_full_comparison(
        "RSI Momentum Filter (on)",
        base_trades, trades, base_sc, sc,
    )

    return [res] if res else []


# ===========================================================================
# Summary Table
# ===========================================================================

def print_summary_table(all_results, base_sc):
    """Print a summary table of all feature tests with their best variant."""
    print(f"\n\n{'#' * 100}")
    print("SUMMARY: ALL FEATURES -- BEST VARIANT PER FEATURE")
    print(f"{'#' * 100}")
    print(f"\n  Baseline: {base_sc['count']} trades, WR {base_sc['win_rate']}%, "
          f"PF {base_sc['pf']}, Net ${base_sc['net_dollar']:+.2f}, "
          f"MaxDD ${base_sc['max_dd_dollar']:.2f}, Sharpe {base_sc['sharpe']}")
    print()

    header = (f"  {'#':<4} {'Feature':<40} {'Trades':>7} {'WR%':>7} {'PF':>8} "
              f"{'Net$':>10} {'dPF':>8} {'dNet$':>10} {'p-val':>8} {'Sig?':>6}")
    print(header)
    print(f"  {'-'*4} {'-'*40} {'-'*7} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")

    for idx, (test_name, test_results) in enumerate(all_results, 1):
        if not test_results:
            print(f"  T{idx:<3} {test_name:<40} {'-- NO RESULTS --':>60}")
            continue

        # Find best variant by PF
        best = max(test_results, key=lambda r: r['pf'])
        sig = "YES" if best['perm_p'] < 0.05 else "No"
        print(f"  T{idx:<3} {best['label']:<40} {best['trades']:>7d} {best['wr']:>7.1f} "
              f"{best['pf']:>8.3f} {best['net']:>+10.2f} {best['dpf']:>+8.3f} "
              f"{best['dnet']:>+10.2f} {best['perm_p']:>8.4f} {sig:>6}")

    print()

    # Highlight features worth combining
    print("  RECOMMENDATION: Features worth combining (p < 0.10 and dPF > 0):")
    any_good = False
    for idx, (test_name, test_results) in enumerate(all_results, 1):
        if not test_results:
            continue
        best = max(test_results, key=lambda r: r['pf'])
        if best['perm_p'] < 0.10 and best['dpf'] > 0:
            any_good = True
            print(f"    -> T{idx}: {best['label']} (PF {best['pf']:.3f}, p={best['perm_p']:.4f})")

    if not any_good:
        print("    (None pass the threshold -- no features significantly improve v9)")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(SEP)
    print("v10 FEATURE VALIDATION SUITE")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    # -----------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------
    print("\n--- Loading Datasets ---\n")
    df_5m_primary, arr_primary = load_primary_dataset()
    df_5m_prebaked, arr_prebaked = load_prebaked_dataset()

    # -----------------------------------------------------------------------
    # Run v9 baseline on both datasets
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("V9 BASELINE")
    print(SEP)

    rsi_primary = compute_rsi(arr_primary['closes'], RSI_LEN)
    base_trades_primary = get_baseline_trades(arr_primary)
    base_sc_primary = score_trades(base_trades_primary, COMMISSION, DOLLAR_PT)
    print(f"  Primary (1-min resampled):  {fmt_score(base_sc_primary, 'v9')}")
    if base_sc_primary:
        print(f"  Exits: {fmt_exits(base_sc_primary['exits'])}")

    rsi_prebaked = compute_rsi(arr_prebaked['closes'], RSI_LEN)
    base_trades_prebaked = get_baseline_trades(arr_prebaked)
    base_sc_prebaked = score_trades(base_trades_prebaked, COMMISSION, DOLLAR_PT)
    print(f"  Prebaked (55-day 5-min):    {fmt_score(base_sc_prebaked, 'v9')}")
    if base_sc_prebaked:
        print(f"  Exits: {fmt_exits(base_sc_prebaked['exits'])}")

    # -----------------------------------------------------------------------
    # Run all feature tests on PRIMARY dataset
    # -----------------------------------------------------------------------
    all_results = []

    # T1: Underwater Exit
    t1_results = test_underwater_exit(
        arr_primary, rsi_primary, base_trades_primary, base_sc_primary,
        "MNQ 1-min resampled",
    )
    all_results.append(("Underwater Exit", t1_results))

    # T2: Opening Range Alignment
    t2_results = test_opening_range(
        arr_primary, rsi_primary, base_trades_primary, base_sc_primary,
        "MNQ 1-min resampled",
    )
    all_results.append(("Opening Range Alignment", t2_results))

    # T3: VWAP Filter (primary only -- needs VWAP)
    t3_results = test_vwap_filter(
        arr_primary, rsi_primary, base_trades_primary, base_sc_primary,
        "MNQ 1-min resampled",
    )
    all_results.append(("VWAP Filter", t3_results))

    # T4: Prior Day Levels
    t4_results = test_prior_day_levels(
        arr_primary, rsi_primary, base_trades_primary, base_sc_primary,
        "MNQ 1-min resampled",
    )
    all_results.append(("Prior Day Levels", t4_results))

    # T5: Price Structure Exit
    t5_results = test_price_structure_exit(
        arr_primary, rsi_primary, base_trades_primary, base_sc_primary,
        "MNQ 1-min resampled",
    )
    all_results.append(("Price Structure Exit", t5_results))

    # T6: SM Flip Reversal
    t6_results = test_sm_reversal(
        arr_primary, rsi_primary, base_trades_primary, base_sc_primary,
        "MNQ 1-min resampled",
    )
    all_results.append(("SM Flip Reversal", t6_results))

    # T7: RSI Momentum Direction
    t7_results = test_rsi_momentum(
        arr_primary, rsi_primary, base_trades_primary, base_sc_primary,
        "MNQ 1-min resampled",
    )
    all_results.append(("RSI Momentum Direction", t7_results))

    # -----------------------------------------------------------------------
    # Also run features on PREBAKED for cross-validation (non-VWAP features)
    # -----------------------------------------------------------------------
    print(f"\n\n{'#' * 100}")
    print("CROSS-VALIDATION ON PREBAKED 55-DAY DATASET")
    print(f"{'#' * 100}")

    # T1 on prebaked
    test_underwater_exit(
        arr_prebaked, rsi_prebaked, base_trades_prebaked, base_sc_prebaked,
        "MNQ 55-day prebaked",
    )

    # T2 on prebaked
    test_opening_range(
        arr_prebaked, rsi_prebaked, base_trades_prebaked, base_sc_prebaked,
        "MNQ 55-day prebaked",
    )

    # T4 on prebaked
    test_prior_day_levels(
        arr_prebaked, rsi_prebaked, base_trades_prebaked, base_sc_prebaked,
        "MNQ 55-day prebaked",
    )

    # T5 on prebaked
    test_price_structure_exit(
        arr_prebaked, rsi_prebaked, base_trades_prebaked, base_sc_prebaked,
        "MNQ 55-day prebaked",
    )

    # T6 on prebaked
    test_sm_reversal(
        arr_prebaked, rsi_prebaked, base_trades_prebaked, base_sc_prebaked,
        "MNQ 55-day prebaked",
    )

    # T7 on prebaked
    test_rsi_momentum(
        arr_prebaked, rsi_prebaked, base_trades_prebaked, base_sc_prebaked,
        "MNQ 55-day prebaked",
    )

    # -----------------------------------------------------------------------
    # Summary table (primary dataset results)
    # -----------------------------------------------------------------------
    print_summary_table(all_results, base_sc_primary)

    print(f"\n{SEP}")
    print("v10 Feature Validation Complete")
    print(SEP)


if __name__ == "__main__":
    main()
