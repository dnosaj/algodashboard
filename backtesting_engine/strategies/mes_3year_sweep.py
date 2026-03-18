"""
MES SM Parameter Sweep — 3-Year Independent Validation
=======================================================
Tests 900 SM configurations across 3 independent years of MES data.
Answers: "Is there ANY SM configuration that works on all 3 years?"

Years:
  Year 1 (Feb 2023 - Feb 2024): prior2_databento_MES_1min_2023-02-17_to_2024-02-16.csv
  Year 2 (Feb 2024 - Feb 2025): prior_databento_MES_1min_2024-02-17_to_2025-02-16.csv
  Year 3 (Feb 2025 - Mar 2026): loaded via load_instrument_1min('MES')

Part 1: MES v2 single-exit (TP=20, SL=35, RSI 12/55/45, CD=25, cutoff 14:15)
Part 3: Runner exit sweep (TP1/TP2/SL sweep with partial exit)

Usage:
    cd backtesting_engine && python3 strategies/mes_3year_sweep.py
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
MES_DOLLAR_PER_PT = 5.0
MES_COMMISSION = 1.25

# MES v2 fixed params
MES_RSI_LEN = 12
MES_RSI_BUY = 55
MES_RSI_SELL = 45
MES_COOLDOWN = 25
MES_MAX_LOSS_PTS = 35
MES_TP_PTS = 20
MES_ENTRY_END_ET = 14 * 60 + 15  # 14:15 ET
MES_EOD_ET = 16 * 60             # 16:00 ET

# Sweep ranges (MES uses larger params than MNQ)
SM_INDEX_RANGE = [16, 18, 20, 22, 24]
SM_FLOW_RANGE = [10, 12, 14]
SM_NORM_RANGE = [300, 350, 400, 450, 500]
SM_EMA_RANGE = [150, 200, 255, 300]
SM_THRESHOLD_RANGE = [0.0, 0.15, 0.25]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_year_data(filepath):
    """Load a year of MES data from a CSV file."""
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
    """Load all 3 years of MES data."""
    print("\n--- Loading Data ---")

    # Year 1: Feb 2023 - Feb 2024
    y1_path = DATA_DIR / "prior2_databento_MES_1min_2023-02-17_to_2024-02-16.csv"
    df_y1 = load_year_data(y1_path)
    print(f"  Year 1 (Feb23-24): {len(df_y1):,} bars, "
          f"{df_y1.index[0]} to {df_y1.index[-1]}, "
          f"Price: {df_y1['Close'].min():.0f}-{df_y1['Close'].max():.0f}")

    # Year 2: Feb 2024 - Feb 2025
    y2_path = DATA_DIR / "prior_databento_MES_1min_2024-02-17_to_2025-02-16.csv"
    df_y2 = load_year_data(y2_path)
    print(f"  Year 2 (Feb24-25): {len(df_y2):,} bars, "
          f"{df_y2.index[0]} to {df_y2.index[-1]}, "
          f"Price: {df_y2['Close'].min():.0f}-{df_y2['Close'].max():.0f}")

    # Year 3: Feb 2025 - Mar 2026 (dev period)
    # Concatenate all databento_MES_1min_*.csv files (excluding prior/prior2)
    db_files = sorted(DATA_DIR.glob("databento_MES_1min_*.csv"))
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
                        max_loss_pts, tp_pts, entry_end_et, eod_et=MES_EOD_ET,
                        breakeven_after_bars=0):
    """Run one backtest and return (trades_count, win_rate, pf, net_dollar)."""
    trades = run_backtest_tp_exit(
        arrays['opens'], arrays['highs'], arrays['lows'],
        arrays['closes'], arrays['sm'], arrays['times'],
        arrays['rsi_curr'], arrays['rsi_prev'],
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts, tp_pts=tp_pts,
        entry_end_et=entry_end_et,
        eod_minutes_et=eod_et,
        breakeven_after_bars=breakeven_after_bars,
    )
    sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                      dollar_per_pt=MES_DOLLAR_PER_PT)
    if sc is None:
        return 0, 0.0, 0.0, 0.0
    return sc['count'], sc['win_rate'], sc['pf'], sc['net_dollar']


# ---------------------------------------------------------------------------
# Runner (partial exit) backtest
# ---------------------------------------------------------------------------

def run_partial_exit_backtest(arrays, sm_threshold, rsi_buy, rsi_sell, cooldown,
                              sl_pts, tp1_pts, tp2_pts, entry_end_et,
                              eod_et=MES_EOD_ET, be_time_bars=75):
    """Run one partial-exit backtest (2 contracts) and return metrics.

    Returns (trades_count, win_rate, pf, net_dollar) with 2-contract P&L.
    """
    from vscalpc_partial_exit_sweep import run_backtest_partial_exit

    trades = run_backtest_partial_exit(
        arrays['opens'], arrays['highs'], arrays['lows'],
        arrays['closes'], arrays['sm'], arrays['times'],
        arrays['rsi_curr'], arrays['rsi_prev'],
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        sl_pts=sl_pts, tp1_pts=tp1_pts, tp2_pts=tp2_pts,
        sl_to_be_after_tp1=True,
        be_time_bars=be_time_bars,
        entry_end_et=entry_end_et,
        eod_minutes_et=eod_et,
    )

    if not trades:
        return 0, 0.0, 0.0, 0.0

    # Score with 2-contract partial P&L
    comm_per_leg = MES_COMMISSION * 2  # entry + exit per contract
    pnl_list = []
    for t in trades:
        leg1_pnl = t["leg1_pts"] * MES_DOLLAR_PER_PT - comm_per_leg
        leg2_pnl = t["leg2_pts"] * MES_DOLLAR_PER_PT - comm_per_leg
        pnl_list.append(leg1_pnl + leg2_pnl)

    pnl_arr = np.array(pnl_list)
    n = len(pnl_arr)
    total_pnl = pnl_arr.sum()
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100

    return n, wr, pf, total_pnl


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
# Runner exit sweep
# ---------------------------------------------------------------------------

def run_runner_sweep(sm_cache, best_sm_cfg, years_data):
    """Sweep runner exit params for the best SM config.

    Sweeps TP1, TP2, SL with fixed CD=25, BE_TIME=75, RSI(12/55/45), cutoff 14:15.
    """
    sm_idx, sm_flow, sm_norm, sm_ema, sm_t = best_sm_cfg

    TP1_RANGE = [4, 6, 8, 10]
    TP2_RANGE = [15, 20, 25, 30]
    SL_RANGE = [25, 30, 35, 40]

    runner_combos = list(product(TP1_RANGE, TP2_RANGE, SL_RANGE))
    total_runner = len(runner_combos)

    print(f"\n{'='*100}")
    print(f"PART 3: RUNNER EXIT SWEEP (SM config: {sm_idx}/{sm_flow}/{sm_norm}/{sm_ema}/{sm_t})")
    print(f"  TP1: {TP1_RANGE}, TP2: {TP2_RANGE}, SL: {SL_RANGE}")
    print(f"  Fixed: CD=25, BE_TIME=75, RSI(12/55/45), cutoff 14:15, EOD 16:00")
    print(f"  {total_runner} configs x 3 years = {total_runner * 3} backtests")
    print(f"{'='*100}")

    results = {}
    done = 0

    for tp1, tp2, sl in runner_combos:
        # Skip if TP1 >= TP2 (makes no sense)
        if tp1 >= tp2:
            continue

        cfg = (tp1, tp2, sl)
        year_results = {}

        for year_key in ['y1', 'y2', 'y3']:
            cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
            arrays = sm_cache[cache_key]
            n_trades, wr, pf, net = run_partial_exit_backtest(
                arrays, sm_t,
                MES_RSI_BUY, MES_RSI_SELL, MES_COOLDOWN,
                sl_pts=sl, tp1_pts=tp1, tp2_pts=tp2,
                entry_end_et=MES_ENTRY_END_ET,
                eod_et=MES_EOD_ET,
                be_time_bars=75,
            )
            year_results[year_key] = {'n': n_trades, 'wr': wr, 'pf': pf, 'net': net}

        results[cfg] = year_results
        done += 1

        if done % 16 == 0:
            print(f"  Runner: {done}/{total_runner} configs done")

    print(f"  Runner sweep done: {done} configs")

    # Filter and rank
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

    print(f"\n  {len(passing)} of {done} runner configs pass all 3 years (PF > 1.0)")

    if not passing:
        print("  NO RUNNER CONFIGS PASS ALL 3 YEARS.")
        return

    passing.sort(key=lambda x: x['worst_pf'], reverse=True)

    print(f"\n  TOP {min(10, len(passing))} RUNNER CONFIGS (by worst-year PF):")
    print(f"  {'Rank':>4}  {'TP1':>4} {'TP2':>4} {'SL':>4}  "
          f"{'Y1 PF':>7} {'Y2 PF':>7} {'Y3 PF':>7} {'Worst':>7}  "
          f"{'Y1 Net$':>9} {'Y2 Net$':>9} {'Y3 Net$':>9} {'Total$':>9}  "
          f"{'Y1 WR':>5} {'Y2 WR':>5} {'Y3 WR':>5}  "
          f"{'Y1 N':>4} {'Y2 N':>4} {'Y3 N':>4}")
    print(f"  {'-'*4}  {'-'*4} {'-'*4} {'-'*4}  "
          f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}  "
          f"{'-'*9} {'-'*9} {'-'*9} {'-'*9}  "
          f"{'-'*5} {'-'*5} {'-'*5}  "
          f"{'-'*4} {'-'*4} {'-'*4}")

    for i, p in enumerate(passing[:10]):
        tp1, tp2, sl = p['cfg']
        print(f"  {i+1:>4}  {tp1:>4} {tp2:>4} {sl:>4}  "
              f"{p['y1_pf']:>7.3f} {p['y2_pf']:>7.3f} {p['y3_pf']:>7.3f} {p['worst_pf']:>7.3f}  "
              f"${p['y1_net']:>+8.0f} ${p['y2_net']:>+8.0f} ${p['y3_net']:>+8.0f} ${p['total_net']:>+8.0f}  "
              f"{p['y1_wr']:>5.1f} {p['y2_wr']:>5.1f} {p['y3_wr']:>5.1f}  "
              f"{p['y1_n']:>4} {p['y2_n']:>4} {p['y3_n']:>4}")

    # Current runner config
    current_runner = (6, 20, 35)
    rank = None
    for i, p in enumerate(passing):
        if p['cfg'] == current_runner:
            rank = i + 1
            break

    if current_runner in results:
        cr = results[current_runner]
        print(f"\n  CURRENT RUNNER CONFIG (TP1=6, TP2=20, SL=35):")
        for yk, yl in [('y1', 'Feb23-24'), ('y2', 'Feb24-25'), ('y3', 'Feb25-26')]:
            print(f"    {yl}: {cr[yk]['n']} trades, WR {cr[yk]['wr']:.1f}%, "
                  f"PF {cr[yk]['pf']:.3f}, Net ${cr[yk]['net']:+.0f}")
        if rank:
            print(f"    RANK: #{rank} of {len(passing)} passing configs")
        else:
            print(f"    DOES NOT PASS — PF < 1.0 on at least one year")

    # Best overfit runner
    best_cfg = None
    best_total = -999999
    for cfg, years in results.items():
        total = sum(years[y]['net'] for y in ['y1', 'y2', 'y3'])
        if total > best_total:
            best_total = total
            best_cfg = cfg
    if best_cfg:
        bf = results[best_cfg]
        print(f"\n  OVERFIT RUNNER CONFIG (best combined P&L): TP1={best_cfg[0]}, TP2={best_cfg[1]}, SL={best_cfg[2]}")
        print(f"    Y1: PF {bf['y1']['pf']:.3f}, Net ${bf['y1']['net']:+.0f}")
        print(f"    Y2: PF {bf['y2']['pf']:.3f}, Net ${bf['y2']['net']:+.0f}")
        print(f"    Y3: PF {bf['y3']['pf']:.3f}, Net ${bf['y3']['net']:+.0f}")
        print(f"    Total: ${best_total:+,.0f}")

    # Cluster analysis for runner params
    if passing:
        print(f"\n  RUNNER CLUSTER ANALYSIS ({len(passing)} passing configs):")
        for pidx, pname in enumerate(['TP1', 'TP2', 'SL']):
            counts = defaultdict(int)
            for p in passing:
                val = p['cfg'][pidx]
                counts[val] += 1
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            total = len(passing)
            dist_str = "  ".join(f"{v}:{c}({c/total*100:.0f}%)" for v, c in sorted_counts)
            print(f"    {pname:>4}: {dist_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 100)
    print("MES SM PARAMETER SWEEP — 3-YEAR INDEPENDENT VALIDATION")
    print("=" * 100)

    df_y1, df_y2, df_y3 = load_all_years()
    years_data = {'y1': df_y1, 'y2': df_y2, 'y3': df_y3}
    year_labels = {'y1': 'Feb23-24', 'y2': 'Feb24-25', 'y3': 'Feb25-26'}

    # Build unique SM param combos (without threshold)
    sm_combos = list(product(SM_INDEX_RANGE, SM_FLOW_RANGE, SM_NORM_RANGE, SM_EMA_RANGE))
    n_sm = len(sm_combos)
    print(f"\n  SM combos (idx x flow x norm x ema): {n_sm}")
    print(f"  Threshold values: {SM_THRESHOLD_RANGE} -> {n_sm * len(SM_THRESHOLD_RANGE)} total configs")

    # ===================================================================
    # PRECOMPUTE: SM for each unique combo x each year
    # ===================================================================
    print(f"\n--- Precomputing SM for {n_sm} combos x 3 years = {n_sm * 3} computations ---")

    # Cache: (sm_idx, sm_flow, sm_norm, sm_ema, year_key) -> arrays dict
    sm_cache = {}

    for combo_idx, (sm_idx, sm_flow, sm_norm, sm_ema) in enumerate(sm_combos):
        if (combo_idx + 1) % 60 == 0 or combo_idx == 0:
            elapsed = time.time() - t_start
            print(f"  SM combo {combo_idx+1}/{n_sm} "
                  f"({sm_idx}/{sm_flow}/{sm_norm}/{sm_ema}) ... [{elapsed:.0f}s]")

        for year_key, df in years_data.items():
            cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
            sm_cache[cache_key] = precompute_year(
                df, sm_idx, sm_flow, sm_norm, sm_ema, rsi_len=MES_RSI_LEN
            )

    precompute_time = time.time() - t_start
    print(f"\n  Precompute done in {precompute_time:.1f}s")

    # ===================================================================
    # PART 1: MES v2 SM sweep (TP=20, SL=35, RSI 12/55/45, CD=25)
    # ===================================================================
    print(f"\n{'='*100}")
    print("PART 1: MES v2 single-exit sweep (TP=20, SL=35, RSI 12/55/45, CD=25, cutoff 14:15, EOD 16:00)")
    print(f"{'='*100}")

    results = {}
    total_configs = n_sm * len(SM_THRESHOLD_RANGE)
    done = 0

    for sm_idx, sm_flow, sm_norm, sm_ema in sm_combos:
        for sm_t in SM_THRESHOLD_RANGE:
            cfg = (sm_idx, sm_flow, sm_norm, sm_ema, sm_t)
            year_results = {}

            for year_key in ['y1', 'y2', 'y3']:
                cache_key = (sm_idx, sm_flow, sm_norm, sm_ema, year_key)
                arrays = sm_cache[cache_key]
                n_trades, wr, pf, net = run_single_backtest(
                    arrays, sm_t,
                    MES_RSI_BUY, MES_RSI_SELL,
                    MES_COOLDOWN, MES_MAX_LOSS_PTS,
                    MES_TP_PTS, MES_ENTRY_END_ET,
                    eod_et=MES_EOD_ET,
                )
                year_results[year_key] = {'n': n_trades, 'wr': wr, 'pf': pf, 'net': net}

            results[cfg] = year_results
            done += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  MES v2: {done}/{total_configs} configs done [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"  MES v2 sweep done: {total_configs} configs in {elapsed:.0f}s")

    # --- Print results for Part 1 ---
    passing = print_top_configs(results, "PART 1: MES v2 (TP=20, SL=35)", n=10)

    # Current config ranking
    current_cfg = (20, 12, 400, 255, 0.0)
    rank = find_current_config_rank(passing, current_cfg)
    current_in_results = results.get(current_cfg)
    if current_in_results:
        y1 = current_in_results['y1']
        y2 = current_in_results['y2']
        y3 = current_in_results['y3']
        print(f"\n  CURRENT CONFIG (20/12/400/255/0.0):")
        print(f"    Y1: {y1['n']} trades, WR {y1['wr']:.1f}%, PF {y1['pf']:.3f}, Net ${y1['net']:+.0f}")
        print(f"    Y2: {y2['n']} trades, WR {y2['wr']:.1f}%, PF {y2['pf']:.3f}, Net ${y2['net']:+.0f}")
        print(f"    Y3: {y3['n']} trades, WR {y3['wr']:.1f}%, PF {y3['pf']:.3f}, Net ${y3['net']:+.0f}")
        all_pass = all(current_in_results[y]['pf'] > 1.0 for y in ['y1', 'y2', 'y3'])
        if rank:
            print(f"    RANK: #{rank} of {len(passing)} passing configs (by worst-year PF)")
        else:
            print(f"    DOES NOT PASS — PF < 1.0 on at least one year")

    # Overfit config
    best_cfg, best_total = find_overfit_config(results)
    if best_cfg:
        bf = results[best_cfg]
        print(f"\n  OVERFIT CONFIG (best combined P&L): {best_cfg}")
        print(f"    Y1: PF {bf['y1']['pf']:.3f}, Net ${bf['y1']['net']:+.0f}")
        print(f"    Y2: PF {bf['y2']['pf']:.3f}, Net ${bf['y2']['net']:+.0f}")
        print(f"    Y3: PF {bf['y3']['pf']:.3f}, Net ${bf['y3']['net']:+.0f}")
        print(f"    Total: ${best_total:+,.0f}")
        is_near = (abs(best_cfg[0] - 20) <= 4 and abs(best_cfg[1] - 12) <= 4
                   and abs(best_cfg[2] - 400) <= 100 and abs(best_cfg[3] - 255) <= 100)
        print(f"    Near current params? {'YES' if is_near else 'NO — wildly different'}")

    cluster_analysis(passing)

    # ===================================================================
    # PART 3: Runner exit sweep (if passing configs exist)
    # ===================================================================
    if passing:
        # Use the top SM config for the runner sweep
        best_sm_cfg = passing[0]['cfg']
        print(f"\n  Using top SM config for runner sweep: {best_sm_cfg}")
        run_runner_sweep(sm_cache, best_sm_cfg, years_data)

        # Also run the runner sweep with the CURRENT config if it passes
        if current_cfg in results and rank is not None and current_cfg != best_sm_cfg:
            print(f"\n  Also running runner sweep with CURRENT config: {current_cfg}")
            run_runner_sweep(sm_cache, current_cfg, years_data)
    else:
        print(f"\n  SKIPPING runner sweep — no SM configs pass all 3 years.")
        # Still run runner with current config even if it doesn't "pass"
        # to see how close it gets
        print(f"\n  Running runner sweep with CURRENT config anyway: {current_cfg}")
        run_runner_sweep(sm_cache, current_cfg, years_data)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"SWEEP COMPLETE — {total_time:.0f}s total")
    print(f"  MES v2: {len(passing)}/{total_configs} configs pass 3 years "
          f"({len(passing)/total_configs*100:.1f}%)")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
