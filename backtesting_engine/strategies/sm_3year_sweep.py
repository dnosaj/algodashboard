"""
SM Parameter Sweep — 3-Year Independent Validation
====================================================
Tests 576 SM configurations across 3 independent years of MNQ data.
Answers: "Is there ANY SM configuration that works on all 3 years?"

Years:
  Year 1 (Feb 2023 - Feb 2024): prior2_databento_MNQ_1min_2023-02-17_to_2024-02-16.csv
  Year 2 (Feb 2024 - Feb 2025): prior_databento_MNQ_1min_2024-02-17_to_2025-02-16.csv
  Year 3 (Feb 2025 - Mar 2026): loaded via load_instrument_1min('MNQ')

Part 1: vScalpA-style (TP=7, SL=40, RSI 8/60/40, CD=20, cutoff 13:00)
Part 3: vScalpB-style (TP=3, SL=10, RSI 8/55/45, CD=20, SM_T sweep)

Usage:
    cd backtesting_engine && python3 strategies/sm_3year_sweep.py
"""

import sys
import time
import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

# --- Path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    compute_rsi,
    compute_et_minutes,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
    NY_OPEN_ET,
    NY_CLOSE_ET,
)

from generate_session import run_backtest_tp_exit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"
MNQ_DOLLAR_PER_PT = 2.0
MNQ_COMMISSION = 0.52

# vScalpA fixed params
VSCALPA_RSI_LEN = 8
VSCALPA_RSI_BUY = 60
VSCALPA_RSI_SELL = 40
VSCALPA_COOLDOWN = 20
VSCALPA_MAX_LOSS_PTS = 40
VSCALPA_TP_PTS = 7
VSCALPA_ENTRY_END_ET = 13 * 60  # 13:00 ET

# vScalpB fixed params
VSCALPB_RSI_LEN = 8
VSCALPB_RSI_BUY = 55
VSCALPB_RSI_SELL = 45
VSCALPB_COOLDOWN = 20
VSCALPB_MAX_LOSS_PTS = 10
VSCALPB_TP_PTS = 3

# Sweep ranges
SM_INDEX_RANGE = [8, 10, 12, 14]
SM_FLOW_RANGE = [10, 12, 14]
SM_NORM_RANGE = [150, 200, 250, 300]
SM_EMA_RANGE = [80, 100, 120, 150]
SM_THRESHOLD_RANGE_A = [0.0, 0.15, 0.25]
SM_THRESHOLD_RANGE_B = [0.15, 0.25, 0.35, 0.50]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_year_data(filepath):
    """Load a year of MNQ data from a CSV file."""
    df_raw = pd.read_csv(filepath)

    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df_raw['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df_raw['open'], errors='coerce')
    result['High'] = pd.to_numeric(df_raw['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df_raw['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df_raw['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df_raw['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    result = result[~result.index.duplicated(keep='first')]
    result = result.sort_index()

    return result


def load_all_years():
    """Load all 3 years of MNQ data."""
    print("\n--- Loading Data ---")

    # Year 1: Feb 2023 - Feb 2024
    y1_path = DATA_DIR / "prior2_databento_MNQ_1min_2023-02-17_to_2024-02-16.csv"
    df_y1 = load_year_data(y1_path)
    print(f"  Year 1 (Feb23-24): {len(df_y1):,} bars, "
          f"{df_y1.index[0]} to {df_y1.index[-1]}, "
          f"Price: {df_y1['Close'].min():.0f}-{df_y1['Close'].max():.0f}")

    # Year 2: Feb 2024 - Feb 2025
    y2_path = DATA_DIR / "prior_databento_MNQ_1min_2024-02-17_to_2025-02-16.csv"
    df_y2 = load_year_data(y2_path)
    print(f"  Year 2 (Feb24-25): {len(df_y2):,} bars, "
          f"{df_y2.index[0]} to {df_y2.index[-1]}, "
          f"Price: {df_y2['Close'].min():.0f}-{df_y2['Close'].max():.0f}")

    # Year 3: Feb 2025 - Mar 2026 (dev period, loaded same way)
    # Concatenate all databento_MNQ_1min_*.csv files (excluding prior/prior2)
    db_files = sorted(DATA_DIR.glob("databento_MNQ_1min_*.csv"))
    dfs = []
    for f in db_files:
        df_tmp = load_year_data(f)
        dfs.append(df_tmp)
    df_y3 = pd.concat(dfs)
    df_y3 = df_y3[~df_y3.index.duplicated(keep='last')]
    df_y3 = df_y3.sort_index()
    print(f"  Year 3 (Feb25-26): {len(df_y3):,} bars, "
          f"{df_y3.index[0]} to {df_y3.index[-1]}, "
          f"Price: {df_y3['Close'].min():.0f}-{df_y3['Close'].max():.0f}")

    return df_y1, df_y2, df_y3


# ---------------------------------------------------------------------------
# Precompute SM + RSI for each year and SM combo
# ---------------------------------------------------------------------------

def precompute_year(df, sm_index, sm_flow, sm_norm, sm_ema, rsi_len):
    """Compute SM and 5-min RSI mapping for one year + one SM config.

    Returns dict with arrays needed for run_backtest_tp_exit.
    """
    closes = df['Close'].values
    volumes = df['Volume'].values

    sm = compute_smart_money(
        closes, volumes,
        index_period=sm_index, flow_period=sm_flow,
        norm_period=sm_norm, ema_len=sm_ema,
    )

    # 5-min RSI mapping
    df_copy = df.copy()
    df_copy['SM_Net'] = sm
    df_5m = resample_to_5min(df_copy)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values,
        df_5m['Close'].values, rsi_len=rsi_len,
    )

    return {
        'opens': df['Open'].values,
        'highs': df['High'].values,
        'lows': df['Low'].values,
        'closes': closes,
        'sm': sm,
        'times': df.index,
        'rsi_curr': rsi_curr,
        'rsi_prev': rsi_prev,
    }


# ---------------------------------------------------------------------------
# Run a single backtest config
# ---------------------------------------------------------------------------

def run_single_backtest(arrays, sm_threshold, rsi_buy, rsi_sell, cooldown,
                        max_loss_pts, tp_pts, entry_end_et):
    """Run one backtest and return (trades_count, win_rate, pf, net_dollar)."""
    trades = run_backtest_tp_exit(
        arrays['opens'], arrays['highs'], arrays['lows'],
        arrays['closes'], arrays['sm'], arrays['times'],
        arrays['rsi_curr'], arrays['rsi_prev'],
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts, tp_pts=tp_pts,
        entry_end_et=entry_end_et,
    )
    sc = score_trades(trades, commission_per_side=MNQ_COMMISSION,
                      dollar_per_pt=MNQ_DOLLAR_PER_PT)
    if sc is None:
        return 0, 0.0, 0.0, 0.0
    return sc['count'], sc['win_rate'], sc['pf'], sc['net_dollar']


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def print_top_configs(results, label, n=10):
    """Print top N configs by worst-year PF."""
    # Filter: PF > 1.0 on all 3 years
    passing = []
    for cfg, years in results.items():
        pfs = [years[y]['pf'] for y in ['y1', 'y2', 'y3']]
        if all(pf > 1.0 for pf in pfs):
            worst_pf = min(pfs)
            total_net = sum(years[y]['net'] for y in ['y1', 'y2', 'y3'])
            passing.append({
                'cfg': cfg,
                'worst_pf': worst_pf,
                'total_net': total_net,
                'y1_pf': pfs[0], 'y2_pf': pfs[1], 'y3_pf': pfs[2],
                'y1_net': years['y1']['net'],
                'y2_net': years['y2']['net'],
                'y3_net': years['y3']['net'],
                'y1_wr': years['y1']['wr'],
                'y2_wr': years['y2']['wr'],
                'y3_wr': years['y3']['wr'],
                'y1_n': years['y1']['n'],
                'y2_n': years['y2']['n'],
                'y3_n': years['y3']['n'],
            })

    total_configs = len(results)
    print(f"\n{'='*100}")
    print(f"{label} — RESULTS")
    print(f"{'='*100}")
    print(f"\n  {len(passing)} of {total_configs} configs pass all 3 years (PF > 1.0)")
    print(f"  Pass rate: {len(passing)/total_configs*100:.1f}%")

    if not passing:
        print("  NO CONFIGS PASS ALL 3 YEARS.")
        return passing

    # Sort by worst-year PF
    passing.sort(key=lambda x: x['worst_pf'], reverse=True)

    print(f"\n  TOP {min(n, len(passing))} CONFIGS (by worst-year PF):")
    print(f"  {'Rank':>4}  {'SM_Idx':>6} {'SM_Flw':>6} {'SM_Nrm':>6} {'SM_EMA':>6} {'SM_T':>5}  "
          f"{'Y1 PF':>7} {'Y2 PF':>7} {'Y3 PF':>7} {'Worst':>7}  "
          f"{'Y1 Net$':>9} {'Y2 Net$':>9} {'Y3 Net$':>9} {'Total$':>9}  "
          f"{'Y1 WR':>5} {'Y2 WR':>5} {'Y3 WR':>5}  "
          f"{'Y1 N':>4} {'Y2 N':>4} {'Y3 N':>4}")
    print(f"  {'-'*4}  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}  "
          f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}  "
          f"{'-'*9} {'-'*9} {'-'*9} {'-'*9}  "
          f"{'-'*5} {'-'*5} {'-'*5}  "
          f"{'-'*4} {'-'*4} {'-'*4}")

    for i, p in enumerate(passing[:n]):
        sm_idx, sm_flw, sm_nrm, sm_ema, sm_t = p['cfg']
        print(f"  {i+1:>4}  {sm_idx:>6} {sm_flw:>6} {sm_nrm:>6} {sm_ema:>6} {sm_t:>5.2f}  "
              f"{p['y1_pf']:>7.3f} {p['y2_pf']:>7.3f} {p['y3_pf']:>7.3f} {p['worst_pf']:>7.3f}  "
              f"${p['y1_net']:>+8.0f} ${p['y2_net']:>+8.0f} ${p['y3_net']:>+8.0f} ${p['total_net']:>+8.0f}  "
              f"{p['y1_wr']:>5.1f} {p['y2_wr']:>5.1f} {p['y3_wr']:>5.1f}  "
              f"{p['y1_n']:>4} {p['y2_n']:>4} {p['y3_n']:>4}")

    return passing


def find_current_config_rank(passing, current_cfg):
    """Find where the current config ranks in the passing list."""
    for i, p in enumerate(passing):
        if p['cfg'] == current_cfg:
            return i + 1
    return None


def find_overfit_config(results):
    """Find the config with the best COMBINED P&L across all 3 years."""
    best_cfg = None
    best_total = -999999
    for cfg, years in results.items():
        total = sum(years[y]['net'] for y in ['y1', 'y2', 'y3'])
        if total > best_total:
            best_total = total
            best_cfg = cfg
    return best_cfg, best_total


def cluster_analysis(passing):
    """Analyze whether passing configs cluster around similar params."""
    if not passing:
        print("\n  CLUSTER ANALYSIS: No passing configs to analyze.")
        return

    print(f"\n  CLUSTER ANALYSIS ({len(passing)} passing configs):")

    # Count frequency of each param value
    param_names = ['SM_Index', 'SM_Flow', 'SM_Norm', 'SM_EMA', 'SM_T']
    for pidx, pname in enumerate(param_names):
        counts = defaultdict(int)
        for p in passing:
            val = p['cfg'][pidx]
            counts[val] += 1
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        total = len(passing)
        dist_str = "  ".join(f"{v}:{c}({c/total*100:.0f}%)" for v, c in sorted_counts)
        print(f"    {pname:>10}: {dist_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 100)
    print("SM PARAMETER SWEEP — 3-YEAR INDEPENDENT VALIDATION")
    print("=" * 100)

    df_y1, df_y2, df_y3 = load_all_years()
    years_data = {'y1': df_y1, 'y2': df_y2, 'y3': df_y3}
    year_labels = {'y1': 'Feb23-24', 'y2': 'Feb24-25', 'y3': 'Feb25-26'}

    # Build unique SM param combos (without threshold)
    sm_combos = list(product(SM_INDEX_RANGE, SM_FLOW_RANGE, SM_NORM_RANGE, SM_EMA_RANGE))
    n_sm = len(sm_combos)
    print(f"\n  SM combos (idx x flow x norm x ema): {n_sm}")
    print(f"  vScalpA threshold values: {SM_THRESHOLD_RANGE_A} -> {n_sm * len(SM_THRESHOLD_RANGE_A)} total configs")
    print(f"  vScalpB threshold values: {SM_THRESHOLD_RANGE_B} -> {n_sm * len(SM_THRESHOLD_RANGE_B)} total configs")

    # ===================================================================
    # PRECOMPUTE: SM for each unique combo x each year
    # ===================================================================
    print(f"\n--- Precomputing SM for {n_sm} combos x 3 years = {n_sm * 3} computations ---")

    # Cache: (sm_idx, sm_flow, sm_norm, sm_ema, year_key) -> arrays dict
    # We need separate caches for RSI len 8 (both strategies use RSI len 8)
    sm_cache = {}

    for combo_idx, (sm_idx, sm_flow, sm_norm, sm_ema) in enumerate(sm_combos):
        if (combo_idx + 1) % 48 == 0 or combo_idx == 0:
            elapsed = time.time() - t_start
            print(f"  SM combo {combo_idx+1}/{n_sm} "
                  f"({sm_idx}/{sm_flow}/{sm_norm}/{sm_ema}) ... [{elapsed:.0f}s]")

        for year_key, df in years_data.items():
            cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
            sm_cache[cache_key] = precompute_year(
                df, sm_idx, sm_flow, sm_norm, sm_ema, rsi_len=VSCALPA_RSI_LEN
            )

    precompute_time = time.time() - t_start
    print(f"\n  Precompute done in {precompute_time:.1f}s")

    # ===================================================================
    # PART 1: vScalpA sweep (TP=7, SL=40, RSI 8/60/40)
    # ===================================================================
    print(f"\n{'='*100}")
    print("PART 1: vScalpA-style sweep (TP=7, SL=40, RSI 8/60/40, CD=20, cutoff 13:00)")
    print(f"{'='*100}")

    results_a = {}
    total_a = n_sm * len(SM_THRESHOLD_RANGE_A)
    done = 0

    for sm_idx, sm_flow, sm_norm, sm_ema in sm_combos:
        for sm_t in SM_THRESHOLD_RANGE_A:
            cfg = (sm_idx, sm_flow, sm_norm, sm_ema, sm_t)
            year_results = {}

            for year_key in ['y1', 'y2', 'y3']:
                cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
                arrays = sm_cache[cache_key]
                n_trades, wr, pf, net = run_single_backtest(
                    arrays, sm_t,
                    VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
                    VSCALPA_COOLDOWN, VSCALPA_MAX_LOSS_PTS,
                    VSCALPA_TP_PTS, VSCALPA_ENTRY_END_ET,
                )
                year_results[year_key] = {'n': n_trades, 'wr': wr, 'pf': pf, 'net': net}

            results_a[cfg] = year_results
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  vScalpA: {done}/{total_a} configs done [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"  vScalpA sweep done: {total_a} configs in {elapsed:.0f}s")

    # --- Print results for Part 1 ---
    passing_a = print_top_configs(results_a, "PART 1: vScalpA (TP=7, SL=40)", n=10)

    # Current config ranking
    current_cfg_a = (10, 12, 200, 100, 0.0)
    rank_a = find_current_config_rank(passing_a, current_cfg_a)
    current_in_results = results_a.get(current_cfg_a)
    if current_in_results:
        y1 = current_in_results['y1']
        y2 = current_in_results['y2']
        y3 = current_in_results['y3']
        print(f"\n  CURRENT CONFIG (10/12/200/100/0.0):")
        print(f"    Y1: {y1['n']} trades, WR {y1['wr']:.1f}%, PF {y1['pf']:.3f}, Net ${y1['net']:+.0f}")
        print(f"    Y2: {y2['n']} trades, WR {y2['wr']:.1f}%, PF {y2['pf']:.3f}, Net ${y2['net']:+.0f}")
        print(f"    Y3: {y3['n']} trades, WR {y3['wr']:.1f}%, PF {y3['pf']:.3f}, Net ${y3['net']:+.0f}")
        all_pass = all(current_in_results[y]['pf'] > 1.0 for y in ['y1', 'y2', 'y3'])
        if rank_a:
            print(f"    RANK: #{rank_a} of {len(passing_a)} passing configs (by worst-year PF)")
        else:
            print(f"    DOES NOT PASS — PF < 1.0 on at least one year")

    # Overfit config
    best_cfg_a, best_total_a = find_overfit_config(results_a)
    if best_cfg_a:
        bf = results_a[best_cfg_a]
        print(f"\n  OVERFIT CONFIG (best combined P&L): {best_cfg_a}")
        print(f"    Y1: PF {bf['y1']['pf']:.3f}, Net ${bf['y1']['net']:+.0f}")
        print(f"    Y2: PF {bf['y2']['pf']:.3f}, Net ${bf['y2']['net']:+.0f}")
        print(f"    Y3: PF {bf['y3']['pf']:.3f}, Net ${bf['y3']['net']:+.0f}")
        print(f"    Total: ${best_total_a:+,.0f}")
        is_near = (abs(best_cfg_a[0] - 10) <= 4 and abs(best_cfg_a[1] - 12) <= 4
                   and abs(best_cfg_a[2] - 200) <= 100 and abs(best_cfg_a[3] - 100) <= 50)
        print(f"    Near current params? {'YES' if is_near else 'NO — wildly different'}")

    cluster_analysis(passing_a)

    # ===================================================================
    # PART 3: vScalpB sweep (TP=3, SL=10, RSI 8/55/45)
    # ===================================================================
    print(f"\n{'='*100}")
    print("PART 3: vScalpB-style sweep (TP=3, SL=10, RSI 8/55/45, CD=20)")
    print(f"{'='*100}")

    results_b = {}
    total_b = n_sm * len(SM_THRESHOLD_RANGE_B)
    done = 0

    for sm_idx, sm_flow, sm_norm, sm_ema in sm_combos:
        for sm_t in SM_THRESHOLD_RANGE_B:
            cfg = (sm_idx, sm_flow, sm_norm, sm_ema, sm_t)
            year_results = {}

            for year_key in ['y1', 'y2', 'y3']:
                cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
                arrays = sm_cache[cache_key]
                n_trades, wr, pf, net = run_single_backtest(
                    arrays, sm_t,
                    VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
                    VSCALPB_COOLDOWN, VSCALPB_MAX_LOSS_PTS,
                    VSCALPB_TP_PTS,
                    entry_end_et=15 * 60 + 45,  # No cutoff for vScalpB (default)
                )
                year_results[year_key] = {'n': n_trades, 'wr': wr, 'pf': pf, 'net': net}

            results_b[cfg] = year_results
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  vScalpB: {done}/{total_b} configs done [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"  vScalpB sweep done: {total_b} configs in {elapsed:.0f}s")

    # --- Print results for Part 3 ---
    passing_b = print_top_configs(results_b, "PART 3: vScalpB (TP=3, SL=10)", n=10)

    # Current vScalpB config ranking
    current_cfg_b = (10, 12, 200, 100, 0.25)
    rank_b = find_current_config_rank(passing_b, current_cfg_b)
    current_in_results_b = results_b.get(current_cfg_b)
    if current_in_results_b:
        y1 = current_in_results_b['y1']
        y2 = current_in_results_b['y2']
        y3 = current_in_results_b['y3']
        print(f"\n  CURRENT CONFIG (10/12/200/100/0.25):")
        print(f"    Y1: {y1['n']} trades, WR {y1['wr']:.1f}%, PF {y1['pf']:.3f}, Net ${y1['net']:+.0f}")
        print(f"    Y2: {y2['n']} trades, WR {y2['wr']:.1f}%, PF {y2['pf']:.3f}, Net ${y2['net']:+.0f}")
        print(f"    Y3: {y3['n']} trades, WR {y3['wr']:.1f}%, PF {y3['pf']:.3f}, Net ${y3['net']:+.0f}")
        if rank_b:
            print(f"    RANK: #{rank_b} of {len(passing_b)} passing configs (by worst-year PF)")
        else:
            print(f"    DOES NOT PASS — PF < 1.0 on at least one year")

    # Overfit config
    best_cfg_b, best_total_b = find_overfit_config(results_b)
    if best_cfg_b:
        bf = results_b[best_cfg_b]
        print(f"\n  OVERFIT CONFIG (best combined P&L): {best_cfg_b}")
        print(f"    Y1: PF {bf['y1']['pf']:.3f}, Net ${bf['y1']['net']:+.0f}")
        print(f"    Y2: PF {bf['y2']['pf']:.3f}, Net ${bf['y2']['net']:+.0f}")
        print(f"    Y3: PF {bf['y3']['pf']:.3f}, Net ${bf['y3']['net']:+.0f}")
        print(f"    Total: ${best_total_b:+,.0f}")

    cluster_analysis(passing_b)

    # ===================================================================
    # CROSS-STYLE ANALYSIS
    # ===================================================================
    print(f"\n{'='*100}")
    print("CROSS-STYLE: SM combos that work for BOTH vScalpA AND vScalpB")
    print(f"{'='*100}")

    # For each SM combo (ignoring threshold), check if there exists
    # any threshold that passes for vScalpA AND any threshold for vScalpB
    both_pass = []
    for sm_idx, sm_flow, sm_norm, sm_ema in sm_combos:
        # Check vScalpA thresholds
        a_best_worst_pf = 0
        a_best_t = None
        for sm_t in SM_THRESHOLD_RANGE_A:
            cfg_a = (sm_idx, sm_flow, sm_norm, sm_ema, sm_t)
            if cfg_a in results_a:
                r = results_a[cfg_a]
                pfs = [r[y]['pf'] for y in ['y1', 'y2', 'y3']]
                if all(pf > 1.0 for pf in pfs):
                    worst = min(pfs)
                    if worst > a_best_worst_pf:
                        a_best_worst_pf = worst
                        a_best_t = sm_t

        # Check vScalpB thresholds
        b_best_worst_pf = 0
        b_best_t = None
        for sm_t in SM_THRESHOLD_RANGE_B:
            cfg_b = (sm_idx, sm_flow, sm_norm, sm_ema, sm_t)
            if cfg_b in results_b:
                r = results_b[cfg_b]
                pfs = [r[y]['pf'] for y in ['y1', 'y2', 'y3']]
                if all(pf > 1.0 for pf in pfs):
                    worst = min(pfs)
                    if worst > b_best_worst_pf:
                        b_best_worst_pf = worst
                        b_best_t = sm_t

        if a_best_t is not None and b_best_t is not None:
            both_pass.append({
                'sm': (sm_idx, sm_flow, sm_norm, sm_ema),
                'a_t': a_best_t, 'a_worst_pf': a_best_worst_pf,
                'b_t': b_best_t, 'b_worst_pf': b_best_worst_pf,
                'combined_worst': min(a_best_worst_pf, b_best_worst_pf),
            })

    print(f"\n  {len(both_pass)} of {n_sm} SM combos pass for BOTH strategies (with best threshold)")

    if both_pass:
        both_pass.sort(key=lambda x: x['combined_worst'], reverse=True)
        print(f"\n  TOP 10 SM combos (by min of worst-year PFs across both strategies):")
        print(f"  {'Rank':>4}  {'SM_Idx':>6} {'SM_Flw':>6} {'SM_Nrm':>6} {'SM_EMA':>6}  "
              f"{'A_T':>5} {'A_WrstPF':>8}  {'B_T':>5} {'B_WrstPF':>8}")
        for i, bp in enumerate(both_pass[:10]):
            sm = bp['sm']
            print(f"  {i+1:>4}  {sm[0]:>6} {sm[1]:>6} {sm[2]:>6} {sm[3]:>6}  "
                  f"{bp['a_t']:>5.2f} {bp['a_worst_pf']:>8.3f}  "
                  f"{bp['b_t']:>5.2f} {bp['b_worst_pf']:>8.3f}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"SWEEP COMPLETE — {total_time:.0f}s total")
    print(f"  vScalpA: {len(passing_a)}/{total_a} configs pass 3 years ({len(passing_a)/total_a*100:.1f}%)")
    print(f"  vScalpB: {len(passing_b)}/{total_b} configs pass 3 years ({len(passing_b)/total_b*100:.1f}%)")
    print(f"  Both strategies: {len(both_pass)}/{n_sm} SM combos work for both")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
