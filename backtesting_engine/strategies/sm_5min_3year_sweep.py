"""
SM on 5-Min Bars — 3-Year Independent Validation
==================================================
Hypothesis: SM computed on 5-min bars (instead of 1-min) may produce a more
regime-robust entry signal because 5-min bars have 5x more price movement,
potentially reducing noise in low-vol periods.

Approach:
  1. Resample 1-min OHLCV to 5-min bars
  2. Compute SM on 5-min closes and volumes
  3. Map 5-min SM values back to 1-min bars (forward-fill, no look-ahead)
  4. Run vScalpA-style backtest using 5-min SM but all else on 1-min

Sweep: sm_index [8,10,12,14] x sm_flow [10,12,14] x sm_norm [100,150,200,250]
       x sm_ema [40,60,80,100] x sm_threshold [0.0, 0.15, 0.25]
     = 4 x 3 x 4 x 4 x 3 = 576 configs x 3 years

Comparison baseline: 59 of 576 1-min SM configs pass all 3 years (10.2%)
(from sm_3year_sweep.py)

Usage:
    cd backtesting_engine && python3 strategies/sm_5min_3year_sweep.py
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

# Sweep ranges for 5-min SM
SM_INDEX_RANGE = [8, 10, 12, 14]
SM_FLOW_RANGE = [10, 12, 14]
SM_NORM_RANGE = [100, 150, 200, 250]     # Lower norms: 5-min has fewer bars
SM_EMA_RANGE = [40, 60, 80, 100]         # Faster EMAs: 5x fewer bars at 5-min
SM_THRESHOLD_RANGE = [0.0, 0.15, 0.25]

# 1-min SM baseline ranges (from sm_3year_sweep.py)
SM_1MIN_INDEX_RANGE = [8, 10, 12, 14]
SM_1MIN_FLOW_RANGE = [10, 12, 14]
SM_1MIN_NORM_RANGE = [150, 200, 250, 300]
SM_1MIN_EMA_RANGE = [80, 100, 120, 150]
SM_1MIN_THRESHOLD_RANGE = [0.0, 0.15, 0.25]


# ---------------------------------------------------------------------------
# Data Loading (same as sm_3year_sweep.py)
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

    # Year 3: Feb 2025 - Mar 2026
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
# 5-min SM computation + mapping back to 1-min
# ---------------------------------------------------------------------------

def resample_to_5min_for_sm(df_1m):
    """Resample 1-min bars to 5-min for SM computation.

    Unlike resample_to_5min from v10_test_common (which expects SM_Net column),
    this only needs OHLCV.
    """
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    df_5m = df_1m.resample('5min').agg(agg).dropna(subset=['Open'])
    return df_5m


def map_5min_sm_to_1min(onemin_times, fivemin_times, fivemin_sm):
    """Map 5-min SM values back to 1-min bars.

    Each 1-min bar gets the SM value from the most recent COMPLETED 5-min bar.
    This means the 5-min bar's SM value is available at the START of the next
    5-min window (same as how RSI mapping works for RSI_prev).

    For the SM entry check: bar[i] uses sm[i-1], and sm[i-1] will hold the
    5-min SM from the most recently completed 5-min bar before bar i-1.
    This is correct and matches how the live engine would work.

    Uses forward-fill: the 5-min SM at time T applies to all 1-min bars
    until the next 5-min bar completes.
    """
    n_1m = len(onemin_times)
    sm_mapped = np.zeros(n_1m)

    j = 0  # pointer into 5-min array
    for i in range(n_1m):
        # Advance j to the latest 5-min bar whose timestamp <= current 1-min bar
        while j + 1 < len(fivemin_times) and fivemin_times[j + 1] <= onemin_times[i]:
            j += 1
        # Use the COMPLETED 5-min bar (j-1), not the current one being formed
        # This prevents look-ahead: the current 5-min bar includes future 1-min bars
        if j >= 1:
            sm_mapped[i] = fivemin_sm[j - 1]
        # else: sm_mapped[i] stays 0.0 (no completed 5-min bar yet)

    return sm_mapped


# ---------------------------------------------------------------------------
# Precompute 5-min SM + 5-min RSI for each year and SM config
# ---------------------------------------------------------------------------

def precompute_year_5min_sm(df, sm_index, sm_flow, sm_norm, sm_ema, rsi_len):
    """Compute SM on 5-min bars and RSI on 5-min bars, both mapped to 1-min.

    Returns dict with arrays needed for run_backtest_tp_exit.
    """
    # Step 1: Resample 1-min to 5-min
    df_5m = resample_to_5min_for_sm(df)

    # Step 2: Compute SM on 5-min bars
    sm_5min = compute_smart_money(
        df_5m['Close'].values, df_5m['Volume'].values,
        index_period=sm_index, flow_period=sm_flow,
        norm_period=sm_norm, ema_len=sm_ema,
    )

    # Step 3: Map 5-min SM back to 1-min bars (forward-fill, no look-ahead)
    sm_1min = map_5min_sm_to_1min(
        df.index.values, df_5m.index.values, sm_5min
    )

    # Step 4: RSI on 5-min mapped to 1-min (same as production)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values,
        df_5m['Close'].values, rsi_len=rsi_len,
    )

    return {
        'opens': df['Open'].values,
        'highs': df['High'].values,
        'lows': df['Low'].values,
        'closes': df['Close'].values,
        'sm': sm_1min,
        'times': df.index,
        'rsi_curr': rsi_curr,
        'rsi_prev': rsi_prev,
    }


def precompute_year_1min_sm(df, sm_index, sm_flow, sm_norm, sm_ema, rsi_len):
    """Compute SM on 1-min bars (baseline). Same as sm_3year_sweep.py."""
    closes = df['Close'].values
    volumes = df['Volume'].values

    sm = compute_smart_money(
        closes, volumes,
        index_period=sm_index, flow_period=sm_flow,
        norm_period=sm_norm, ema_len=sm_ema,
    )

    # 5-min RSI mapping (same for both — RSI always runs on 5-min)
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
# Run a single backtest
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
    print(f"\n{'='*110}")
    print(f"{label} — RESULTS")
    print(f"{'='*110}")
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


def cluster_analysis(passing):
    """Analyze whether passing configs cluster around similar params."""
    if not passing:
        print("\n  CLUSTER ANALYSIS: No passing configs to analyze.")
        return

    print(f"\n  CLUSTER ANALYSIS ({len(passing)} passing configs):")

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
    print("=" * 110)
    print("SM ON 5-MIN BARS — 3-YEAR INDEPENDENT VALIDATION")
    print("Hypothesis: 5-min SM reduces noise, improves regime robustness")
    print("=" * 110)

    df_y1, df_y2, df_y3 = load_all_years()
    years_data = {'y1': df_y1, 'y2': df_y2, 'y3': df_y3}
    year_labels = {'y1': 'Feb23-24', 'y2': 'Feb24-25', 'y3': 'Feb25-26'}

    # Build unique SM param combos (without threshold)
    sm_5min_combos = list(product(SM_INDEX_RANGE, SM_FLOW_RANGE,
                                   SM_NORM_RANGE, SM_EMA_RANGE))
    n_5min = len(sm_5min_combos)

    sm_1min_combos = list(product(SM_1MIN_INDEX_RANGE, SM_1MIN_FLOW_RANGE,
                                   SM_1MIN_NORM_RANGE, SM_1MIN_EMA_RANGE))
    n_1min = len(sm_1min_combos)

    n_5min_total = n_5min * len(SM_THRESHOLD_RANGE)
    n_1min_total = n_1min * len(SM_1MIN_THRESHOLD_RANGE)
    print(f"\n  5-min SM combos (idx x flow x norm x ema): {n_5min}")
    print(f"  5-min threshold values: {SM_THRESHOLD_RANGE} -> {n_5min_total} total configs")
    print(f"  1-min SM combos (baseline): {n_1min}")
    print(f"  1-min threshold values: {SM_1MIN_THRESHOLD_RANGE} -> {n_1min_total} total configs")

    # ===================================================================
    # PRECOMPUTE: 5-min SM for each unique combo x each year
    # ===================================================================
    print(f"\n--- Precomputing 5-min SM for {n_5min} combos x 3 years ---")

    sm_5min_cache = {}

    for combo_idx, (sm_idx, sm_flow, sm_norm, sm_ema) in enumerate(sm_5min_combos):
        if (combo_idx + 1) % 48 == 0 or combo_idx == 0:
            elapsed = time.time() - t_start
            print(f"  5-min SM combo {combo_idx+1}/{n_5min} "
                  f"({sm_idx}/{sm_flow}/{sm_norm}/{sm_ema}) ... [{elapsed:.0f}s]")

        for year_key, df in years_data.items():
            cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
            sm_5min_cache[cache_key] = precompute_year_5min_sm(
                df, sm_idx, sm_flow, sm_norm, sm_ema, rsi_len=VSCALPA_RSI_LEN
            )

    precompute_5min_time = time.time() - t_start
    print(f"\n  5-min SM precompute done in {precompute_5min_time:.1f}s")

    # ===================================================================
    # PRECOMPUTE: 1-min SM baseline for comparison
    # ===================================================================
    print(f"\n--- Precomputing 1-min SM (baseline) for {n_1min} combos x 3 years ---")

    sm_1min_cache = {}

    for combo_idx, (sm_idx, sm_flow, sm_norm, sm_ema) in enumerate(sm_1min_combos):
        if (combo_idx + 1) % 48 == 0 or combo_idx == 0:
            elapsed = time.time() - t_start
            print(f"  1-min SM combo {combo_idx+1}/{n_1min} "
                  f"({sm_idx}/{sm_flow}/{sm_norm}/{sm_ema}) ... [{elapsed:.0f}s]")

        for year_key, df in years_data.items():
            cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
            sm_1min_cache[cache_key] = precompute_year_1min_sm(
                df, sm_idx, sm_flow, sm_norm, sm_ema, rsi_len=VSCALPA_RSI_LEN
            )

    precompute_1min_time = time.time() - t_start
    print(f"\n  1-min SM precompute done in {precompute_1min_time:.1f}s (cumulative)")

    # ===================================================================
    # SWEEP: 5-min SM configs
    # ===================================================================
    print(f"\n{'='*110}")
    print("5-MIN SM SWEEP (TP=7, SL=40, RSI 8/60/40, CD=20, cutoff 13:00)")
    print(f"{'='*110}")

    results_5min = {}
    done = 0

    for sm_idx, sm_flow, sm_norm, sm_ema in sm_5min_combos:
        for sm_t in SM_THRESHOLD_RANGE:
            cfg = (sm_idx, sm_flow, sm_norm, sm_ema, sm_t)
            year_results = {}

            for year_key in ['y1', 'y2', 'y3']:
                cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
                arrays = sm_5min_cache[cache_key]
                n_trades, wr, pf, net = run_single_backtest(
                    arrays, sm_t,
                    VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
                    VSCALPA_COOLDOWN, VSCALPA_MAX_LOSS_PTS,
                    VSCALPA_TP_PTS, VSCALPA_ENTRY_END_ET,
                )
                year_results[year_key] = {'n': n_trades, 'wr': wr, 'pf': pf, 'net': net}

            results_5min[cfg] = year_results
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  5-min: {done}/{n_5min_total} configs done [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"  5-min sweep done: {n_5min_total} configs in {elapsed:.0f}s")

    # ===================================================================
    # SWEEP: 1-min SM baseline
    # ===================================================================
    print(f"\n{'='*110}")
    print("1-MIN SM BASELINE (TP=7, SL=40, RSI 8/60/40, CD=20, cutoff 13:00)")
    print(f"{'='*110}")

    results_1min = {}
    done = 0

    for sm_idx, sm_flow, sm_norm, sm_ema in sm_1min_combos:
        for sm_t in SM_1MIN_THRESHOLD_RANGE:
            cfg = (sm_idx, sm_flow, sm_norm, sm_ema, sm_t)
            year_results = {}

            for year_key in ['y1', 'y2', 'y3']:
                cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
                arrays = sm_1min_cache[cache_key]
                n_trades, wr, pf, net = run_single_backtest(
                    arrays, sm_t,
                    VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
                    VSCALPA_COOLDOWN, VSCALPA_MAX_LOSS_PTS,
                    VSCALPA_TP_PTS, VSCALPA_ENTRY_END_ET,
                )
                year_results[year_key] = {'n': n_trades, 'wr': wr, 'pf': pf, 'net': net}

            results_1min[cfg] = year_results
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  1-min: {done}/{n_1min_total} configs done [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"  1-min sweep done: {n_1min_total} configs in {elapsed:.0f}s")

    # ===================================================================
    # RESULTS: 5-min SM
    # ===================================================================
    passing_5min = print_top_configs(
        results_5min, "5-MIN SM (TP=7, SL=40, RSI 8/60/40, CD=20, cutoff 13:00)", n=10
    )
    cluster_analysis(passing_5min)

    # ===================================================================
    # RESULTS: 1-min SM baseline
    # ===================================================================
    passing_1min = print_top_configs(
        results_1min, "1-MIN SM BASELINE (TP=7, SL=40, RSI 8/60/40, CD=20, cutoff 13:00)", n=10
    )
    cluster_analysis(passing_1min)

    # Current 1-min config
    current_cfg_1min = (10, 12, 200, 100, 0.0)
    current_in_1min = results_1min.get(current_cfg_1min)
    if current_in_1min:
        y1 = current_in_1min['y1']
        y2 = current_in_1min['y2']
        y3 = current_in_1min['y3']
        print(f"\n  CURRENT 1-MIN CONFIG (10/12/200/100/0.0):")
        print(f"    Y1: {y1['n']} trades, WR {y1['wr']:.1f}%, PF {y1['pf']:.3f}, Net ${y1['net']:+.0f}")
        print(f"    Y2: {y2['n']} trades, WR {y2['wr']:.1f}%, PF {y2['pf']:.3f}, Net ${y2['net']:+.0f}")
        print(f"    Y3: {y3['n']} trades, WR {y3['wr']:.1f}%, PF {y3['pf']:.3f}, Net ${y3['net']:+.0f}")
        all_pass = all(current_in_1min[y]['pf'] > 1.0 for y in ['y1', 'y2', 'y3'])
        if all_pass:
            rank = None
            for i, p in enumerate(passing_1min):
                if p['cfg'] == current_cfg_1min:
                    rank = i + 1
                    break
            if rank:
                print(f"    RANK: #{rank} of {len(passing_1min)} passing configs")
        else:
            print(f"    DOES NOT PASS all 3 years")

    # ===================================================================
    # HEAD-TO-HEAD COMPARISON
    # ===================================================================
    print(f"\n{'='*110}")
    print("HEAD-TO-HEAD COMPARISON: 5-min SM vs 1-min SM")
    print(f"{'='*110}")

    n_pass_5min = len(passing_5min)
    n_pass_1min = len(passing_1min)
    rate_5min = n_pass_5min / n_5min_total * 100
    rate_1min = n_pass_1min / n_1min_total * 100

    print(f"\n  PASS RATE:")
    print(f"    5-min SM: {n_pass_5min} of {n_5min_total} pass all 3 years ({rate_5min:.1f}%)")
    print(f"    1-min SM: {n_pass_1min} of {n_1min_total} pass all 3 years ({rate_1min:.1f}%)")
    if rate_5min > rate_1min and rate_1min > 0:
        print(f"    -> 5-min SM has {rate_5min/rate_1min:.2f}x higher pass rate")
    elif rate_1min > rate_5min and rate_5min > 0:
        print(f"    -> 1-min SM has {rate_1min/rate_5min:.2f}x higher pass rate")
    elif rate_1min > 0 and rate_5min == 0:
        print(f"    -> 1-min SM dominates (5-min has ZERO passing configs)")
    elif rate_5min > 0 and rate_1min == 0:
        print(f"    -> 5-min SM dominates (1-min has ZERO passing configs)")
    else:
        print(f"    -> Same pass rate")

    # Best configs comparison
    if passing_5min and passing_1min:
        best_5min = passing_5min[0]  # sorted by worst-year PF
        best_1min = passing_1min[0]
        print(f"\n  BEST CONFIG (by worst-year PF):")
        print(f"    5-min: SM({best_5min['cfg'][0]}/{best_5min['cfg'][1]}/{best_5min['cfg'][2]}/{best_5min['cfg'][3]}) T={best_5min['cfg'][4]}")
        print(f"      Worst PF: {best_5min['worst_pf']:.3f}  "
              f"Y1: PF {best_5min['y1_pf']:.3f} ${best_5min['y1_net']:+.0f}  "
              f"Y2: PF {best_5min['y2_pf']:.3f} ${best_5min['y2_net']:+.0f}  "
              f"Y3: PF {best_5min['y3_pf']:.3f} ${best_5min['y3_net']:+.0f}  "
              f"Total: ${best_5min['total_net']:+.0f}")
        print(f"    1-min: SM({best_1min['cfg'][0]}/{best_1min['cfg'][1]}/{best_1min['cfg'][2]}/{best_1min['cfg'][3]}) T={best_1min['cfg'][4]}")
        print(f"      Worst PF: {best_1min['worst_pf']:.3f}  "
              f"Y1: PF {best_1min['y1_pf']:.3f} ${best_1min['y1_net']:+.0f}  "
              f"Y2: PF {best_1min['y2_pf']:.3f} ${best_1min['y2_net']:+.0f}  "
              f"Y3: PF {best_1min['y3_pf']:.3f} ${best_1min['y3_net']:+.0f}  "
              f"Total: ${best_1min['total_net']:+.0f}")

    # Profitability comparison: median total P&L of passing configs
    if passing_5min:
        totals_5min = sorted([p['total_net'] for p in passing_5min])
        median_5min = totals_5min[len(totals_5min) // 2]
        best_total_5min = max(totals_5min)
        worst_total_5min = min(totals_5min)
    else:
        median_5min = 0
        best_total_5min = 0
        worst_total_5min = 0

    if passing_1min:
        totals_1min = sorted([p['total_net'] for p in passing_1min])
        median_1min = totals_1min[len(totals_1min) // 2]
        best_total_1min = max(totals_1min)
        worst_total_1min = min(totals_1min)
    else:
        median_1min = 0
        best_total_1min = 0
        worst_total_1min = 0

    print(f"\n  PROFITABILITY OF PASSING CONFIGS (3-year total $):")
    print(f"    5-min SM: median ${median_5min:+,.0f}  best ${best_total_5min:+,.0f}  worst ${worst_total_5min:+,.0f}")
    print(f"    1-min SM: median ${median_1min:+,.0f}  best ${best_total_1min:+,.0f}  worst ${worst_total_1min:+,.0f}")

    # Consistency comparison: average of worst-year PFs
    if passing_5min:
        avg_worst_5min = np.mean([p['worst_pf'] for p in passing_5min])
        max_worst_5min = max(p['worst_pf'] for p in passing_5min)
    else:
        avg_worst_5min = 0
        max_worst_5min = 0

    if passing_1min:
        avg_worst_1min = np.mean([p['worst_pf'] for p in passing_1min])
        max_worst_1min = max(p['worst_pf'] for p in passing_1min)
    else:
        avg_worst_1min = 0
        max_worst_1min = 0

    print(f"\n  CONSISTENCY (worst-year PF of passing configs):")
    print(f"    5-min SM: avg worst-year PF {avg_worst_5min:.3f}, best worst-year PF {max_worst_5min:.3f}")
    print(f"    1-min SM: avg worst-year PF {avg_worst_1min:.3f}, best worst-year PF {max_worst_1min:.3f}")

    # Trade count comparison
    if passing_5min:
        avg_trades_5min = np.mean([p['y1_n'] + p['y2_n'] + p['y3_n'] for p in passing_5min])
    else:
        avg_trades_5min = 0
    if passing_1min:
        avg_trades_1min = np.mean([p['y1_n'] + p['y2_n'] + p['y3_n'] for p in passing_1min])
    else:
        avg_trades_1min = 0

    print(f"\n  TRADE FREQUENCY (avg total trades across 3 years for passing configs):")
    print(f"    5-min SM: {avg_trades_5min:.0f} trades")
    print(f"    1-min SM: {avg_trades_1min:.0f} trades")

    # ===================================================================
    # PER-YEAR BREAKDOWN: How many configs are profitable each year?
    # ===================================================================
    print(f"\n  PER-YEAR PROFITABILITY (PF > 1.0):")
    for ykey, ylabel in year_labels.items():
        n_prof_5min = sum(1 for r in results_5min.values() if r[ykey]['pf'] > 1.0 and r[ykey]['n'] >= 10)
        n_prof_1min = sum(1 for r in results_1min.values() if r[ykey]['pf'] > 1.0 and r[ykey]['n'] >= 10)
        print(f"    {ylabel}: 5-min {n_prof_5min}/{n_5min_total} ({n_prof_5min/n_5min_total*100:.1f}%)  "
              f"1-min {n_prof_1min}/{n_1min_total} ({n_prof_1min/n_1min_total*100:.1f}%)")

    # ===================================================================
    # VERDICT
    # ===================================================================
    print(f"\n{'='*110}")
    print("VERDICT")
    print(f"{'='*110}")

    if rate_5min > rate_1min * 1.2:
        print(f"\n  5-min SM shows MEANINGFULLY HIGHER regime robustness ({rate_5min:.1f}% vs {rate_1min:.1f}% pass rate)")
        print(f"  -> Worth investigating further for production use")
    elif rate_5min > rate_1min:
        print(f"\n  5-min SM shows SLIGHTLY HIGHER regime robustness ({rate_5min:.1f}% vs {rate_1min:.1f}% pass rate)")
        print(f"  -> Marginal improvement, may not justify the added complexity")
    elif rate_5min == rate_1min:
        print(f"\n  5-min SM shows EQUAL regime robustness ({rate_5min:.1f}% = {rate_1min:.1f}% pass rate)")
        print(f"  -> No benefit to computing SM on 5-min bars")
    else:
        print(f"\n  5-min SM shows LOWER regime robustness ({rate_5min:.1f}% vs {rate_1min:.1f}% pass rate)")
        print(f"  -> Hypothesis rejected: 1-min SM is more robust")

    if passing_5min and passing_1min:
        if max_worst_5min > max_worst_1min:
            print(f"  Best 5-min worst-year PF ({max_worst_5min:.3f}) > best 1-min ({max_worst_1min:.3f})")
        else:
            print(f"  Best 1-min worst-year PF ({max_worst_1min:.3f}) >= best 5-min ({max_worst_5min:.3f})")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*110}")
    print(f"SWEEP COMPLETE — {total_time:.0f}s total")
    print(f"  5-min SM: {n_pass_5min}/{n_5min_total} configs pass 3 years ({rate_5min:.1f}%)")
    print(f"  1-min SM: {n_pass_1min}/{n_1min_total} configs pass 3 years ({rate_1min:.1f}%)")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
