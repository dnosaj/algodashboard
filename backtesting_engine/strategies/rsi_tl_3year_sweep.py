"""
RSI Trendline Breakout — 3-Year Independent Validation Sweep
=============================================================
Tests RSI trendline breakout signal configs and exit configs across 3 independent
years of MNQ data. Answers: "Is there a robust RSI TL config that works on all 3 years?"

Years:
  Year 1 (Feb 2023 - Feb 2024): prior2_databento_MNQ_1min_2023-02-17_to_2024-02-16.csv
  Year 2 (Feb 2024 - Feb 2025): prior_databento_MNQ_1min_2024-02-17_to_2025-02-16.csv
  Year 3 (Feb 2025 - Mar 2026): all databento_MNQ_1min_*.csv concatenated

Part 1: Signal config sweep (320 configs) with fixed runner exits
Part 2: Exit sweep on best signal config (144 configs)
Part 3: SM alignment filter test (4 modes)

Usage:
    cd backtesting_engine && python3 strategies/rsi_tl_3year_sweep.py
"""

import sys
import time
import traceback
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

from rsi_trendline_backtest import (
    compute_rsi,
    generate_signals,
    run_backtest_runner,
    score_trades,
    compute_et_minutes,
    NY_OPEN_ET,
    NY_CLOSE_ET,
    COMMISSION_PER_SIDE,
    DOLLAR_PER_PT,
)

from v10_test_common import compute_smart_money

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"
MNQ_DOLLAR_PER_PT = 2.0
MNQ_COMMISSION = 0.52

ENTRY_START_ET = 9 * 60 + 30   # 09:30 ET
ENTRY_END_ET = 13 * 60         # 13:00 ET

# Part 1: Signal sweep ranges
RSI_PERIOD_RANGE = [6, 8, 10, 12, 14]
LB_LEFT_RANGE = [8, 10, 12, 15]
LB_RIGHT_RANGE = [2, 3, 4, 5]
MIN_SPACING_RANGE = [8, 10, 12, 15]

# Part 1 fixed exits
FIXED_TP1 = 7
FIXED_TP2 = 20
FIXED_SL = 40
FIXED_CD = 30

# Part 2: Exit sweep ranges
TP1_RANGE = [5, 7, 10]
TP2_RANGE = [15, 20, 25, 30]
SL_RANGE = [25, 30, 35, 40]
CD_RANGE = [20, 25, 30]

# Part 3: SM params (same as live strategies)
SM_INDEX = 10
SM_FLOW = 12
SM_NORM = 200
SM_EMA = 100


# ---------------------------------------------------------------------------
# Data Loading (matches sm_3year_sweep.py pattern)
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

    # Year 3: Feb 2025 - Mar 2026 (concatenate all databento_MNQ files)
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
# Helpers
# ---------------------------------------------------------------------------

def run_single_config(opens, closes, times, long_sig, short_sig,
                      tp1, tp2, sl, cd):
    """Run one runner backtest and return metrics dict or None."""
    trades = run_backtest_runner(
        opens, closes, times, long_sig, short_sig,
        cooldown_bars=cd, max_loss_pts=sl,
        tp1_pts=tp1, tp2_pts=tp2,
        entry_start_et=ENTRY_START_ET,
        entry_end_et=ENTRY_END_ET,
        eod_minutes_et=NY_CLOSE_ET,
    )
    sc = score_trades(trades, commission_per_side=MNQ_COMMISSION,
                      dollar_per_pt=MNQ_DOLLAR_PER_PT)
    if sc is None:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'net': 0.0, 'sharpe': 0.0, 'mdd': 0.0}
    return {
        'n': sc['count'],
        'wr': sc['win_rate'],
        'pf': sc['pf'],
        'net': sc['net_dollar'],
        'sharpe': sc['sharpe'],
        'mdd': sc['max_dd_dollar'],
    }


def print_top_configs(results, label, n=10, cfg_formatter=None):
    """Print top N configs by worst-year PF.

    results: dict of cfg -> {'y1': metrics, 'y2': metrics, 'y3': metrics}
    cfg_formatter: function(cfg) -> str for printing the config params
    """
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
                'y1_sharpe': years['y1']['sharpe'],
                'y2_sharpe': years['y2']['sharpe'],
                'y3_sharpe': years['y3']['sharpe'],
            })

    total_configs = len(results)
    print(f"\n{'='*120}")
    print(f"{label} -- RESULTS")
    print(f"{'='*120}")
    print(f"\n  {len(passing)} of {total_configs} configs pass all 3 years (PF > 1.0)")
    print(f"  Pass rate: {len(passing)/total_configs*100:.1f}%" if total_configs > 0 else "")

    if not passing:
        print("  NO CONFIGS PASS ALL 3 YEARS.")
        return passing

    passing.sort(key=lambda x: x['worst_pf'], reverse=True)

    show = min(n, len(passing))
    print(f"\n  TOP {show} CONFIGS (by worst-year PF):")

    if cfg_formatter:
        # Use custom header from formatter
        header = cfg_formatter(None)  # Get header string
        print(f"  {'Rank':>4}  {header}  "
              f"{'Y1 PF':>7} {'Y2 PF':>7} {'Y3 PF':>7} {'Worst':>7}  "
              f"{'Y1 Net$':>9} {'Y2 Net$':>9} {'Y3 Net$':>9} {'Total$':>9}  "
              f"{'Y1 WR':>5} {'Y2 WR':>5} {'Y3 WR':>5}  "
              f"{'Y1 N':>4} {'Y2 N':>4} {'Y3 N':>4}")
    else:
        print(f"  {'Rank':>4}  {'Config':>30}  "
              f"{'Y1 PF':>7} {'Y2 PF':>7} {'Y3 PF':>7} {'Worst':>7}  "
              f"{'Y1 Net$':>9} {'Y2 Net$':>9} {'Y3 Net$':>9} {'Total$':>9}  "
              f"{'Y1 WR':>5} {'Y2 WR':>5} {'Y3 WR':>5}  "
              f"{'Y1 N':>4} {'Y2 N':>4} {'Y3 N':>4}")

    for i, p in enumerate(passing[:show]):
        if cfg_formatter:
            cfg_str = cfg_formatter(p['cfg'])
        else:
            cfg_str = f"{str(p['cfg']):>30}"
        print(f"  {i+1:>4}  {cfg_str}  "
              f"{p['y1_pf']:>7.3f} {p['y2_pf']:>7.3f} {p['y3_pf']:>7.3f} {p['worst_pf']:>7.3f}  "
              f"${p['y1_net']:>+8.0f} ${p['y2_net']:>+8.0f} ${p['y3_net']:>+8.0f} ${p['total_net']:>+8.0f}  "
              f"{p['y1_wr']:>5.1f} {p['y2_wr']:>5.1f} {p['y3_wr']:>5.1f}  "
              f"{p['y1_n']:>4} {p['y2_n']:>4} {p['y3_n']:>4}")

    return passing


def cluster_analysis(passing, param_names, param_indices):
    """Analyze parameter clustering among passing configs."""
    if not passing:
        print("\n  CLUSTER ANALYSIS: No passing configs to analyze.")
        return

    print(f"\n  CLUSTER ANALYSIS ({len(passing)} passing configs):")

    for pidx, pname in zip(param_indices, param_names):
        counts = defaultdict(int)
        for p in passing:
            val = p['cfg'][pidx]
            counts[val] += 1
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        total = len(passing)
        dist_str = "  ".join(f"{v}:{c}({c/total*100:.0f}%)" for v, c in sorted_counts)
        print(f"    {pname:>12}: {dist_str}")


# ---------------------------------------------------------------------------
# Part 1: Signal Config Sweep
# ---------------------------------------------------------------------------

def run_part1(years_data):
    """Sweep RSI period + pivot params with fixed runner exits.

    Returns results dict and the best signal config tuple.
    """
    print(f"\n{'='*120}")
    print("PART 1: SIGNAL CONFIG SWEEP (320 configs)")
    print(f"  RSI periods: {RSI_PERIOD_RANGE}")
    print(f"  lb_left:     {LB_LEFT_RANGE}")
    print(f"  lb_right:    {LB_RIGHT_RANGE}")
    print(f"  min_spacing: {MIN_SPACING_RANGE}")
    print(f"  Fixed exits: TP1={FIXED_TP1}, TP2={FIXED_TP2}, SL={FIXED_SL}, CD={FIXED_CD}")
    print(f"  Entry window: 9:30-13:00 ET")
    print(f"{'='*120}")

    signal_combos = list(product(RSI_PERIOD_RANGE, LB_LEFT_RANGE,
                                  LB_RIGHT_RANGE, MIN_SPACING_RANGE))
    total = len(signal_combos)
    print(f"\n  Total signal configs: {total}")

    results = {}
    errors = []
    done = 0
    t0 = time.time()

    for rsi_period, lb_left, lb_right, min_spacing in signal_combos:
        cfg = (rsi_period, lb_left, lb_right, min_spacing)

        try:
            year_results = {}
            for year_key, df in years_data.items():
                closes = df['Close'].values
                opens = df['Open'].values
                times = df.index

                rsi = compute_rsi(closes, rsi_period)
                long_sig, short_sig = generate_signals(
                    rsi,
                    lb_left=lb_left,
                    lb_right=lb_right,
                    min_spacing=min_spacing,
                    piv_lookback=5,
                    max_age=2000,
                    break_thresh=0.0,
                )

                metrics = run_single_config(
                    opens, closes, times,
                    long_sig, short_sig,
                    tp1=FIXED_TP1, tp2=FIXED_TP2,
                    sl=FIXED_SL, cd=FIXED_CD,
                )
                year_results[year_key] = metrics

            results[cfg] = year_results

        except Exception as e:
            errors.append((cfg, str(e)))
            traceback.print_exc()

        done += 1
        if done % 50 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  Part 1: {done}/{total} configs done "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    elapsed = time.time() - t0
    print(f"\n  Part 1 sweep done: {len(results)} configs in {elapsed:.0f}s")
    if errors:
        print(f"  ERRORS: {len(errors)} configs failed")
        for cfg, err in errors[:5]:
            print(f"    {cfg}: {err}")

    # Signal config formatter
    def signal_cfg_fmt(cfg):
        if cfg is None:
            return f"{'RSI':>3} {'LbL':>3} {'LbR':>3} {'Spc':>3}"
        rsi_p, lbl, lbr, spc = cfg
        return f"{rsi_p:>3} {lbl:>3} {lbr:>3} {spc:>3}"

    passing = print_top_configs(results, "PART 1: Signal Config Sweep", n=10,
                                cfg_formatter=signal_cfg_fmt)

    # Cluster analysis
    cluster_analysis(passing, ['RSI_Period', 'lb_left', 'lb_right', 'min_spacing'],
                     [0, 1, 2, 3])

    # Find best config by worst-year PF
    best_cfg = None
    best_worst_pf = 0
    if passing:
        best_cfg = passing[0]['cfg']
        best_worst_pf = passing[0]['worst_pf']

    # Also show signal counts for top configs
    if passing:
        print(f"\n  SIGNAL COUNTS for top 3 configs:")
        for p in passing[:3]:
            rsi_p, lbl, lbr, spc = p['cfg']
            for yk, df in years_data.items():
                rsi = compute_rsi(df['Close'].values, rsi_p)
                long_sig, short_sig = generate_signals(
                    rsi, lb_left=lbl, lb_right=lbr,
                    min_spacing=spc, piv_lookback=5,
                )
                print(f"    RSI={rsi_p} LbL={lbl} LbR={lbr} Spc={spc} {yk}: "
                      f"{long_sig.sum()} long + {short_sig.sum()} short = "
                      f"{long_sig.sum() + short_sig.sum()} raw signals")

    return results, best_cfg


# ---------------------------------------------------------------------------
# Part 2: Exit Sweep on Best Signal Config
# ---------------------------------------------------------------------------

def run_part2(years_data, best_signal_cfg):
    """Sweep exit params using the best signal config from Part 1."""
    if best_signal_cfg is None:
        print("\n  PART 2 SKIPPED -- no passing signal configs from Part 1")
        return {}, None

    rsi_period, lb_left, lb_right, min_spacing = best_signal_cfg

    print(f"\n{'='*120}")
    print(f"PART 2: EXIT SWEEP on best signal config")
    print(f"  Signal config: RSI={rsi_period}, lb_left={lb_left}, "
          f"lb_right={lb_right}, min_spacing={min_spacing}")
    print(f"  TP1: {TP1_RANGE}")
    print(f"  TP2: {TP2_RANGE}")
    print(f"  SL:  {SL_RANGE}")
    print(f"  CD:  {CD_RANGE}")
    print(f"  Entry window: 9:30-13:00 ET, SL->BE after TP1")
    print(f"{'='*120}")

    exit_combos = list(product(TP1_RANGE, TP2_RANGE, SL_RANGE, CD_RANGE))
    # Filter: TP2 must be > TP1
    exit_combos = [(tp1, tp2, sl, cd) for tp1, tp2, sl, cd in exit_combos
                   if tp2 > tp1]
    total = len(exit_combos)
    print(f"\n  Total exit configs (TP2 > TP1): {total}")

    # Precompute signals for each year (same signal config)
    year_signals = {}
    for year_key, df in years_data.items():
        closes = df['Close'].values
        rsi = compute_rsi(closes, rsi_period)
        long_sig, short_sig = generate_signals(
            rsi, lb_left=lb_left, lb_right=lb_right,
            min_spacing=min_spacing, piv_lookback=5,
            max_age=2000, break_thresh=0.0,
        )
        year_signals[year_key] = (long_sig, short_sig)

    results = {}
    errors = []
    done = 0
    t0 = time.time()

    for tp1, tp2, sl, cd in exit_combos:
        cfg = (tp1, tp2, sl, cd)

        try:
            year_results = {}
            for year_key, df in years_data.items():
                opens = df['Open'].values
                closes = df['Close'].values
                times = df.index
                long_sig, short_sig = year_signals[year_key]

                metrics = run_single_config(
                    opens, closes, times,
                    long_sig, short_sig,
                    tp1=tp1, tp2=tp2, sl=sl, cd=cd,
                )
                year_results[year_key] = metrics

            results[cfg] = year_results

        except Exception as e:
            errors.append((cfg, str(e)))
            traceback.print_exc()

        done += 1
        if done % 50 == 0 or done == total:
            elapsed = time.time() - t0
            print(f"  Part 2: {done}/{total} configs done [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\n  Part 2 sweep done: {len(results)} configs in {elapsed:.0f}s")
    if errors:
        print(f"  ERRORS: {len(errors)} configs failed")

    # Exit config formatter
    def exit_cfg_fmt(cfg):
        if cfg is None:
            return f"{'TP1':>3} {'TP2':>3} {'SL':>3} {'CD':>3}"
        tp1, tp2, sl, cd = cfg
        return f"{tp1:>3} {tp2:>3} {sl:>3} {cd:>3}"

    passing = print_top_configs(results, "PART 2: Exit Sweep", n=10,
                                cfg_formatter=exit_cfg_fmt)

    # Cluster analysis
    cluster_analysis(passing, ['TP1', 'TP2', 'SL', 'CD'], [0, 1, 2, 3])

    # Find best exit config
    best_exit = None
    if passing:
        best_exit = passing[0]['cfg']

    # Also print Sharpe ranking for top exit configs
    if passing:
        print(f"\n  TOP 5 BY SHARPE (sum across 3 years):")
        sharpe_ranked = sorted(passing,
                               key=lambda x: x['y1_sharpe'] + x['y2_sharpe'] + x['y3_sharpe'],
                               reverse=True)
        for i, p in enumerate(sharpe_ranked[:5]):
            tp1, tp2, sl, cd = p['cfg']
            total_sharpe = p['y1_sharpe'] + p['y2_sharpe'] + p['y3_sharpe']
            print(f"    {i+1}. TP1={tp1} TP2={tp2} SL={sl} CD={cd}: "
                  f"Sharpe Y1={p['y1_sharpe']:.2f} Y2={p['y2_sharpe']:.2f} "
                  f"Y3={p['y3_sharpe']:.2f} Sum={total_sharpe:.2f}")

    return results, best_exit


# ---------------------------------------------------------------------------
# Part 3: SM Alignment Filter
# ---------------------------------------------------------------------------

def run_part3(years_data, best_signal_cfg, best_exit_cfg):
    """Test whether SM alignment filtering improves the RSI TL strategy."""
    if best_signal_cfg is None or best_exit_cfg is None:
        print("\n  PART 3 SKIPPED -- no best config from Parts 1-2")
        return

    rsi_period, lb_left, lb_right, min_spacing = best_signal_cfg
    tp1, tp2, sl, cd = best_exit_cfg

    print(f"\n{'='*120}")
    print(f"PART 3: SM ALIGNMENT FILTER TEST")
    print(f"  Signal: RSI={rsi_period}, lb_left={lb_left}, "
          f"lb_right={lb_right}, min_spacing={min_spacing}")
    print(f"  Exits: TP1={tp1}, TP2={tp2}, SL={sl}, CD={cd}")
    print(f"  SM params: {SM_INDEX}/{SM_FLOW}/{SM_NORM}/{SM_EMA}")
    print(f"  4 modes: no filter, block SM-opposed (T=0.0), "
          f"block strongly opposed (T=0.25), tighter SL=25 on opposed")
    print(f"{'='*120}")

    # Precompute RSI signals and SM for each year
    year_data_prepped = {}
    for year_key, df in years_data.items():
        closes = df['Close'].values
        opens = df['Open'].values
        volumes = df['Volume'].values
        times = df.index

        rsi = compute_rsi(closes, rsi_period)
        long_sig, short_sig = generate_signals(
            rsi, lb_left=lb_left, lb_right=lb_right,
            min_spacing=min_spacing, piv_lookback=5,
            max_age=2000, break_thresh=0.0,
        )

        sm = compute_smart_money(
            closes, volumes,
            index_period=SM_INDEX, flow_period=SM_FLOW,
            norm_period=SM_NORM, ema_len=SM_EMA,
        )

        year_data_prepped[year_key] = {
            'opens': opens,
            'closes': closes,
            'times': times,
            'long_sig': long_sig,
            'short_sig': short_sig,
            'sm': sm,
        }

    modes = [
        ("No SM filter (baseline)", "baseline"),
        ("Block SM-opposed (SM_T=0.0)", "block_0.0"),
        ("Block SM strongly-opposed (SM_T=0.25)", "block_0.25"),
        ("Tighter SL=25 on SM-opposed", "tighter_sl"),
    ]

    print(f"\n  {'Mode':<45}  "
          f"{'Y1 PF':>7} {'Y1 N':>4} {'Y1 Net$':>9}  "
          f"{'Y2 PF':>7} {'Y2 N':>4} {'Y2 Net$':>9}  "
          f"{'Y3 PF':>7} {'Y3 N':>4} {'Y3 Net$':>9}  "
          f"{'Total$':>9}")
    print(f"  {'-'*45}  "
          f"{'-'*7} {'-'*4} {'-'*9}  "
          f"{'-'*7} {'-'*4} {'-'*9}  "
          f"{'-'*7} {'-'*4} {'-'*9}  "
          f"{'-'*9}")

    for mode_label, mode_key in modes:
        try:
            year_metrics = {}
            for year_key, yd in year_data_prepped.items():
                long_sig_filtered = yd['long_sig'].copy()
                short_sig_filtered = yd['short_sig'].copy()
                sm = yd['sm']
                use_sl = sl

                if mode_key == "block_0.0":
                    # Block entries where SM is on the opposite side (net < 0 for long, net > 0 for short)
                    for i in range(len(long_sig_filtered)):
                        if long_sig_filtered[i] and sm[i] < 0.0:
                            long_sig_filtered[i] = False
                        if short_sig_filtered[i] and sm[i] > 0.0:
                            short_sig_filtered[i] = False

                elif mode_key == "block_0.25":
                    # Block entries where SM is strongly opposed
                    for i in range(len(long_sig_filtered)):
                        if long_sig_filtered[i] and sm[i] < -0.25:
                            long_sig_filtered[i] = False
                        if short_sig_filtered[i] and sm[i] > 0.25:
                            short_sig_filtered[i] = False

                elif mode_key == "tighter_sl":
                    # We need to run two backtests:
                    # 1. SM-aligned trades with normal SL
                    # 2. SM-opposed trades with tighter SL=25
                    # Then combine the trade lists

                    long_aligned = long_sig_filtered.copy()
                    short_aligned = short_sig_filtered.copy()
                    long_opposed = long_sig_filtered.copy()
                    short_opposed = short_sig_filtered.copy()

                    for i in range(len(long_sig_filtered)):
                        # Aligned: SM agrees with direction
                        if long_aligned[i] and sm[i] < 0.0:
                            long_aligned[i] = False
                        if short_aligned[i] and sm[i] > 0.0:
                            short_aligned[i] = False
                        # Opposed: SM disagrees
                        if long_opposed[i] and sm[i] >= 0.0:
                            long_opposed[i] = False
                        if short_opposed[i] and sm[i] <= 0.0:
                            short_opposed[i] = False

                    trades_aligned = run_backtest_runner(
                        yd['opens'], yd['closes'], yd['times'],
                        long_aligned, short_aligned,
                        cooldown_bars=cd, max_loss_pts=sl,
                        tp1_pts=tp1, tp2_pts=tp2,
                        entry_start_et=ENTRY_START_ET,
                        entry_end_et=ENTRY_END_ET,
                        eod_minutes_et=NY_CLOSE_ET,
                    )
                    trades_opposed = run_backtest_runner(
                        yd['opens'], yd['closes'], yd['times'],
                        long_opposed, short_opposed,
                        cooldown_bars=cd, max_loss_pts=25,  # Tighter SL
                        tp1_pts=tp1, tp2_pts=tp2,
                        entry_start_et=ENTRY_START_ET,
                        entry_end_et=ENTRY_END_ET,
                        eod_minutes_et=NY_CLOSE_ET,
                    )
                    combined_trades = trades_aligned + trades_opposed
                    sc = score_trades(combined_trades,
                                      commission_per_side=MNQ_COMMISSION,
                                      dollar_per_pt=MNQ_DOLLAR_PER_PT)
                    if sc is None:
                        year_metrics[year_key] = {'n': 0, 'wr': 0.0, 'pf': 0.0, 'net': 0.0}
                    else:
                        year_metrics[year_key] = {
                            'n': sc['count'], 'wr': sc['win_rate'],
                            'pf': sc['pf'], 'net': sc['net_dollar'],
                        }
                    continue

                # For baseline, block_0.0, block_0.25 modes
                metrics = run_single_config(
                    yd['opens'], yd['closes'], yd['times'],
                    long_sig_filtered, short_sig_filtered,
                    tp1=tp1, tp2=tp2, sl=sl, cd=cd,
                )
                year_metrics[year_key] = metrics

            total_net = sum(year_metrics[y]['net'] for y in ['y1', 'y2', 'y3'])
            print(f"  {mode_label:<45}  "
                  f"{year_metrics['y1']['pf']:>7.3f} {year_metrics['y1']['n']:>4} "
                  f"${year_metrics['y1']['net']:>+8.0f}  "
                  f"{year_metrics['y2']['pf']:>7.3f} {year_metrics['y2']['n']:>4} "
                  f"${year_metrics['y2']['net']:>+8.0f}  "
                  f"{year_metrics['y3']['pf']:>7.3f} {year_metrics['y3']['n']:>4} "
                  f"${year_metrics['y3']['net']:>+8.0f}  "
                  f"${total_net:>+8.0f}")

        except Exception as e:
            print(f"  {mode_label:<45}  ERROR: {e}")
            traceback.print_exc()

    # Additional analysis: What fraction of signals are SM-opposed?
    print(f"\n  SM ALIGNMENT BREAKDOWN:")
    for year_key, yd in year_data_prepped.items():
        sm = yd['sm']
        long_sig = yd['long_sig']
        short_sig = yd['short_sig']

        n_long = long_sig.sum()
        n_short = short_sig.sum()
        n_long_opposed = sum(1 for i in range(len(long_sig))
                             if long_sig[i] and sm[i] < 0.0)
        n_short_opposed = sum(1 for i in range(len(short_sig))
                              if short_sig[i] and sm[i] > 0.0)
        n_long_strong_opp = sum(1 for i in range(len(long_sig))
                                if long_sig[i] and sm[i] < -0.25)
        n_short_strong_opp = sum(1 for i in range(len(short_sig))
                                 if short_sig[i] and sm[i] > 0.25)

        total_signals = n_long + n_short
        total_opposed = n_long_opposed + n_short_opposed
        total_strong_opp = n_long_strong_opp + n_short_strong_opp

        pct_opp = total_opposed / total_signals * 100 if total_signals > 0 else 0
        pct_strong = total_strong_opp / total_signals * 100 if total_signals > 0 else 0

        print(f"    {year_key}: {total_signals} signals, "
              f"{total_opposed} SM-opposed ({pct_opp:.1f}%), "
              f"{total_strong_opp} strongly opposed ({pct_strong:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 120)
    print("RSI TRENDLINE BREAKOUT — 3-YEAR INDEPENDENT VALIDATION SWEEP")
    print("=" * 120)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        df_y1, df_y2, df_y3 = load_all_years()
    except Exception as e:
        print(f"\nFATAL: Failed to load data: {e}")
        traceback.print_exc()
        return

    years_data = {'y1': df_y1, 'y2': df_y2, 'y3': df_y3}

    # ===================================================================
    # PART 1: Signal Config Sweep
    # ===================================================================
    try:
        results_p1, best_signal_cfg = run_part1(years_data)
    except Exception as e:
        print(f"\nFATAL: Part 1 failed: {e}")
        traceback.print_exc()
        best_signal_cfg = None
        results_p1 = {}

    # ===================================================================
    # PART 2: Exit Sweep on best signal config
    # ===================================================================
    try:
        results_p2, best_exit_cfg = run_part2(years_data, best_signal_cfg)
    except Exception as e:
        print(f"\nFATAL: Part 2 failed: {e}")
        traceback.print_exc()
        best_exit_cfg = None
        results_p2 = {}

    # If Part 2 had no passing configs, fall back to fixed exits for Part 3
    if best_exit_cfg is None and best_signal_cfg is not None:
        print("\n  Part 2 had no passing configs. Using fixed exits for Part 3.")
        best_exit_cfg = (FIXED_TP1, FIXED_TP2, FIXED_SL, FIXED_CD)

    # ===================================================================
    # PART 3: SM Alignment Filter
    # ===================================================================
    try:
        run_part3(years_data, best_signal_cfg, best_exit_cfg)
    except Exception as e:
        print(f"\nFATAL: Part 3 failed: {e}")
        traceback.print_exc()

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*120}")
    print(f"SWEEP COMPLETE — {total_time:.0f}s total ({total_time/60:.1f} min)")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if results_p1:
        passing_p1 = sum(1 for cfg, years in results_p1.items()
                         if all(years[y]['pf'] > 1.0 for y in ['y1', 'y2', 'y3']))
        print(f"  Part 1 (Signal sweep): {passing_p1}/{len(results_p1)} configs pass 3 years "
              f"({passing_p1/len(results_p1)*100:.1f}%)")
    if best_signal_cfg:
        print(f"  Best signal config: RSI={best_signal_cfg[0]}, lb_left={best_signal_cfg[1]}, "
              f"lb_right={best_signal_cfg[2]}, min_spacing={best_signal_cfg[3]}")

    if results_p2:
        passing_p2 = sum(1 for cfg, years in results_p2.items()
                         if all(years[y]['pf'] > 1.0 for y in ['y1', 'y2', 'y3']))
        print(f"  Part 2 (Exit sweep): {passing_p2}/{len(results_p2)} configs pass 3 years "
              f"({passing_p2/len(results_p2)*100:.1f}%)")
    if best_exit_cfg:
        print(f"  Best exit config: TP1={best_exit_cfg[0]}, TP2={best_exit_cfg[1]}, "
              f"SL={best_exit_cfg[2]}, CD={best_exit_cfg[3]}")

    print(f"{'='*120}")


if __name__ == "__main__":
    main()
