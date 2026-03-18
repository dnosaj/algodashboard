"""
vScalpC Runner — 3-Year Independent Validation
================================================
Tests vScalpC partial exit (2-contract runner) across 3 independent years of MNQ data.

Years:
  Year 1 (Feb 2023 - Feb 2024): prior2_databento_MNQ_1min_2023-02-17_to_2024-02-16.csv
  Year 2 (Feb 2024 - Feb 2025): prior_databento_MNQ_1min_2024-02-17_to_2025-02-16.csv
  Year 3 (Feb 2025 - Mar 2026): loaded via load_all_years (concat databento files)

Test A: SM(10/12/200/100) vs SM(12/12/200/80) — head-to-head on fixed exits
Test B: Exit sweep (TP1 x TP2 x SL) using SM(12/12/200/80) — 36 configs x 3 years

Usage:
    cd backtesting_engine && python3 strategies/vscalpc_3year_sweep.py
"""

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

# --- Path setup (reuse pattern from sm_3year_sweep.py) ---
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

from vscalpc_partial_exit_sweep import (
    run_backtest_partial_exit,
    score_partial_trades,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"
MNQ_DOLLAR_PER_PT = 2.0
MNQ_COMMISSION = 0.52

# vScalpC fixed entry params
RSI_LEN = 8
RSI_BUY = 60
RSI_SELL = 40
COOLDOWN = 20
SM_THRESHOLD = 0.0
ENTRY_END_ET = 13 * 60  # 13:00 ET

# Current production exits
CURRENT_TP1 = 7
CURRENT_TP2 = 25
CURRENT_SL = 40


# ---------------------------------------------------------------------------
# Data Loading (reuse from sm_3year_sweep.py)
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

    # Year 3: Feb 2025 - Mar 2026 (concat all databento_MNQ_1min_*.csv)
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
# Precompute SM + RSI for one year and SM config
# ---------------------------------------------------------------------------

def precompute_year(df, sm_index, sm_flow, sm_norm, sm_ema, rsi_len):
    """Compute SM and 5-min RSI mapping for one year + one SM config.

    Returns dict with arrays needed for run_backtest_partial_exit.
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
# Run a single partial exit backtest
# ---------------------------------------------------------------------------

def run_single_partial(arrays, sm_threshold, rsi_buy, rsi_sell, cooldown,
                       sl_pts, tp1_pts, tp2_pts, entry_end_et,
                       sl_to_be=True, be_time_bars=0):
    """Run one partial exit backtest and return scored metrics."""
    trades = run_backtest_partial_exit(
        arrays['opens'], arrays['highs'], arrays['lows'],
        arrays['closes'], arrays['sm'], arrays['times'],
        arrays['rsi_curr'], arrays['rsi_prev'],
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold,
        cooldown_bars=cooldown,
        sl_pts=sl_pts, tp1_pts=tp1_pts, tp2_pts=tp2_pts,
        sl_to_be_after_tp1=sl_to_be,
        be_time_bars=be_time_bars,
        entry_end_et=entry_end_et,
    )

    sc = score_partial_trades(trades,
                              dollar_per_pt=MNQ_DOLLAR_PER_PT,
                              commission_per_side=MNQ_COMMISSION)
    if sc is None:
        return 0, 0.0, 0.0, 0.0, 0.0
    return sc['count'], sc['win_rate'], sc['pf'], sc['net_dollar'], sc['sharpe']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 110)
    print("vScalpC RUNNER — 3-YEAR INDEPENDENT VALIDATION")
    print("  Partial exit: entry_qty=2, TP1 scalp + runner to TP2/SL, SL->BE after TP1")
    print("=" * 110)

    df_y1, df_y2, df_y3 = load_all_years()
    years_data = {'y1': df_y1, 'y2': df_y2, 'y3': df_y3}
    year_labels = {'y1': 'Y1 (Feb23-24)', 'y2': 'Y2 (Feb24-25)', 'y3': 'Y3 (Feb25-26)'}

    # ==================================================================
    # TEST A: SM(10/12/200/100) vs SM(12/12/200/80)
    # ==================================================================
    print(f"\n{'='*110}")
    print("TEST A: SM Config Comparison for vScalpC Runner")
    print("  Fixed exits: TP1=7, TP2=25, SL=40, CD=20, SM_T=0.0, RSI(8/60/40), cutoff 13:00, SL->BE")
    print(f"{'='*110}")

    sm_configs = {
        'Current SM(10/12/200/100)': (10, 12, 200, 100),
        'Robust SM(12/12/200/80)':   (12, 12, 200, 80),
    }

    # Precompute for both SM configs
    sm_precomputed = {}
    for sm_label, (sm_idx, sm_flow, sm_norm, sm_ema) in sm_configs.items():
        for year_key, df in years_data.items():
            key = (sm_label, year_key)
            sm_precomputed[key] = precompute_year(
                df, sm_idx, sm_flow, sm_norm, sm_ema, rsi_len=RSI_LEN
            )
            elapsed = time.time() - t_start
            print(f"  Precomputed {sm_label} / {year_labels[year_key]} [{elapsed:.0f}s]")

    # Run both configs on fixed exits
    print(f"\n  {'SM Config':>30}  {'Year':>15}  {'N':>5} {'WR%':>6} {'PF':>7} {'Net$':>10} {'Sharpe':>7}")
    print(f"  {'-'*30}  {'-'*15}  {'-'*5} {'-'*6} {'-'*7} {'-'*10} {'-'*7}")

    test_a_results = {}
    for sm_label in sm_configs:
        test_a_results[sm_label] = {}
        for year_key in ['y1', 'y2', 'y3']:
            key = (sm_label, year_key)
            arrays = sm_precomputed[key]
            n, wr, pf, net, sharpe = run_single_partial(
                arrays, SM_THRESHOLD, RSI_BUY, RSI_SELL, COOLDOWN,
                sl_pts=CURRENT_SL, tp1_pts=CURRENT_TP1, tp2_pts=CURRENT_TP2,
                entry_end_et=ENTRY_END_ET, sl_to_be=True,
            )
            test_a_results[sm_label][year_key] = {
                'n': n, 'wr': wr, 'pf': pf, 'net': net, 'sharpe': sharpe
            }
            print(f"  {sm_label:>30}  {year_labels[year_key]:>15}  "
                  f"{n:>5} {wr:>5.1f}% {pf:>7.3f} ${net:>+9.0f} {sharpe:>7.3f}")

    # Summary comparison
    print(f"\n  --- SUMMARY ---")
    print(f"  {'SM Config':>30}  {'Total$':>10} {'WorstPF':>8} {'AllPass':>8}")
    print(f"  {'-'*30}  {'-'*10} {'-'*8} {'-'*8}")
    for sm_label in sm_configs:
        r = test_a_results[sm_label]
        total_net = sum(r[y]['net'] for y in ['y1', 'y2', 'y3'])
        pfs = [r[y]['pf'] for y in ['y1', 'y2', 'y3']]
        worst_pf = min(pfs)
        all_pass = "YES" if all(pf > 1.0 for pf in pfs) else "NO"
        print(f"  {sm_label:>30}  ${total_net:>+9.0f} {worst_pf:>8.3f} {all_pass:>8}")

    # ==================================================================
    # TEST B: Exit Sweep with SM(12/12/200/80)
    # ==================================================================
    print(f"\n{'='*110}")
    print("TEST B: Exit Sweep — SM(12/12/200/80), 36 configs x 3 years")
    print("  TP1: [5, 7, 10]  |  TP2: [15, 20, 25, 30]  |  SL: [30, 35, 40]")
    print("  Fixed: CD=20, SM_T=0.0, RSI(8/60/40), cutoff 13:00, SL->BE after TP1")
    print(f"{'='*110}")

    TP1_VALUES = [5, 7, 10]
    TP2_VALUES = [15, 20, 25, 30]
    SL_VALUES = [30, 35, 40]

    combos = list(product(TP1_VALUES, TP2_VALUES, SL_VALUES))
    total_combos = len(combos)
    total_tests = total_combos * 3
    print(f"  {total_combos} exit configs x 3 years = {total_tests} backtests")

    # Use SM(12/12/200/80) precomputed arrays
    robust_label = 'Robust SM(12/12/200/80)'

    results = {}
    done = 0
    for tp1, tp2, sl in combos:
        cfg_key = (tp1, tp2, sl)
        results[cfg_key] = {}

        for year_key in ['y1', 'y2', 'y3']:
            arrays = sm_precomputed[(robust_label, year_key)]
            n, wr, pf, net, sharpe = run_single_partial(
                arrays, SM_THRESHOLD, RSI_BUY, RSI_SELL, COOLDOWN,
                sl_pts=sl, tp1_pts=tp1, tp2_pts=tp2,
                entry_end_et=ENTRY_END_ET, sl_to_be=True,
            )
            results[cfg_key][year_key] = {
                'n': n, 'wr': wr, 'pf': pf, 'net': net, 'sharpe': sharpe
            }
            done += 1

        if done % 30 == 0:
            elapsed = time.time() - t_start
            print(f"  ... {done}/{total_tests} backtests done [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"  Sweep complete: {total_tests} backtests in {elapsed:.0f}s")

    # --- Filter: all 3 years PF > 1.0 ---
    passing = []
    for cfg_key, year_results in results.items():
        pfs = [year_results[y]['pf'] for y in ['y1', 'y2', 'y3']]
        if all(pf > 1.0 for pf in pfs):
            worst_pf = min(pfs)
            total_net = sum(year_results[y]['net'] for y in ['y1', 'y2', 'y3'])
            worst_net = min(year_results[y]['net'] for y in ['y1', 'y2', 'y3'])
            sharpes = [year_results[y]['sharpe'] for y in ['y1', 'y2', 'y3']]
            worst_sharpe = min(sharpes)
            passing.append({
                'cfg': cfg_key,
                'worst_pf': worst_pf,
                'total_net': total_net,
                'worst_net': worst_net,
                'worst_sharpe': worst_sharpe,
                **{f'{y}_pf': year_results[y]['pf'] for y in ['y1', 'y2', 'y3']},
                **{f'{y}_net': year_results[y]['net'] for y in ['y1', 'y2', 'y3']},
                **{f'{y}_wr': year_results[y]['wr'] for y in ['y1', 'y2', 'y3']},
                **{f'{y}_n': year_results[y]['n'] for y in ['y1', 'y2', 'y3']},
                **{f'{y}_sharpe': year_results[y]['sharpe'] for y in ['y1', 'y2', 'y3']},
            })

    print(f"\n  {len(passing)} of {total_combos} exit configs pass ALL 3 years (PF > 1.0)")
    print(f"  Pass rate: {len(passing)/total_combos*100:.1f}%")

    if not passing:
        print("  NO CONFIGS PASS ALL 3 YEARS.")
    else:
        # Sort by worst-year PF
        passing.sort(key=lambda x: x['worst_pf'], reverse=True)

        # --- Full table of all passing configs ---
        print(f"\n  ALL PASSING CONFIGS (sorted by worst-year PF):")
        print(f"  {'Rank':>4}  {'TP1':>4} {'TP2':>4} {'SL':>4}  "
              f"{'Y1 PF':>7} {'Y2 PF':>7} {'Y3 PF':>7} {'WrstPF':>7}  "
              f"{'Y1 Net$':>9} {'Y2 Net$':>9} {'Y3 Net$':>9} {'Total$':>9}  "
              f"{'Y1 Sh':>6} {'Y2 Sh':>6} {'Y3 Sh':>6}  "
              f"{'Y1 N':>4} {'Y2 N':>4} {'Y3 N':>4}")
        print(f"  {'-'*4}  {'-'*4} {'-'*4} {'-'*4}  "
              f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}  "
              f"{'-'*9} {'-'*9} {'-'*9} {'-'*9}  "
              f"{'-'*6} {'-'*6} {'-'*6}  "
              f"{'-'*4} {'-'*4} {'-'*4}")

        for i, p in enumerate(passing):
            tp1, tp2, sl = p['cfg']
            print(f"  {i+1:>4}  {tp1:>4} {tp2:>4} {sl:>4}  "
                  f"{p['y1_pf']:>7.3f} {p['y2_pf']:>7.3f} {p['y3_pf']:>7.3f} {p['worst_pf']:>7.3f}  "
                  f"${p['y1_net']:>+8.0f} ${p['y2_net']:>+8.0f} ${p['y3_net']:>+8.0f} ${p['total_net']:>+8.0f}  "
                  f"{p['y1_sharpe']:>6.2f} {p['y2_sharpe']:>6.2f} {p['y3_sharpe']:>6.2f}  "
                  f"{p['y1_n']:>4} {p['y2_n']:>4} {p['y3_n']:>4}")

        # --- Top 5 by worst-year PF ---
        print(f"\n  TOP 5 CONFIGS (by worst-year PF):")
        for i, p in enumerate(passing[:5]):
            tp1, tp2, sl = p['cfg']
            print(f"\n  #{i+1}: TP1={tp1} TP2={tp2} SL={sl}")
            print(f"    Y1: {p['y1_n']} trades, WR {p['y1_wr']:.1f}%, "
                  f"PF {p['y1_pf']:.3f}, ${p['y1_net']:+.0f}, Sharpe {p['y1_sharpe']:.3f}")
            print(f"    Y2: {p['y2_n']} trades, WR {p['y2_wr']:.1f}%, "
                  f"PF {p['y2_pf']:.3f}, ${p['y2_net']:+.0f}, Sharpe {p['y2_sharpe']:.3f}")
            print(f"    Y3: {p['y3_n']} trades, WR {p['y3_wr']:.1f}%, "
                  f"PF {p['y3_pf']:.3f}, ${p['y3_net']:+.0f}, Sharpe {p['y3_sharpe']:.3f}")
            print(f"    3-Year Total: ${p['total_net']:+,.0f}, "
                  f"Worst PF: {p['worst_pf']:.3f}, Worst Sharpe: {p['worst_sharpe']:.3f}")

    # ==================================================================
    # FULL TABLE: All 36 configs with per-year detail
    # ==================================================================
    print(f"\n{'='*110}")
    print("FULL RESULTS TABLE — All 36 exit configs x 3 years")
    print(f"{'='*110}")

    print(f"\n  {'TP1':>4} {'TP2':>4} {'SL':>4}  "
          f"{'Y1_N':>5} {'Y1_WR':>6} {'Y1_PF':>7} {'Y1_Net$':>9} {'Y1_Sh':>6}  "
          f"{'Y2_N':>5} {'Y2_WR':>6} {'Y2_PF':>7} {'Y2_Net$':>9} {'Y2_Sh':>6}  "
          f"{'Y3_N':>5} {'Y3_WR':>6} {'Y3_PF':>7} {'Y3_Net$':>9} {'Y3_Sh':>6}  "
          f"{'Pass':>5}")
    sep_year = f"{'-'*5} {'-'*6} {'-'*7} {'-'*9} {'-'*6}  "
    print(f"  {'-'*4} {'-'*4} {'-'*4}  " + sep_year * 3 + f"{'-'*5}")

    for tp1, tp2, sl in combos:
        cfg_key = (tp1, tp2, sl)
        yr = results[cfg_key]
        pfs = [yr[y]['pf'] for y in ['y1', 'y2', 'y3']]
        all_pass = "YES" if all(pf > 1.0 for pf in pfs) else "no"
        print(f"  {tp1:>4} {tp2:>4} {sl:>4}  "
              f"{yr['y1']['n']:>5} {yr['y1']['wr']:>5.1f}% {yr['y1']['pf']:>7.3f} ${yr['y1']['net']:>+8.0f} {yr['y1']['sharpe']:>6.2f}  "
              f"{yr['y2']['n']:>5} {yr['y2']['wr']:>5.1f}% {yr['y2']['pf']:>7.3f} ${yr['y2']['net']:>+8.0f} {yr['y2']['sharpe']:>6.2f}  "
              f"{yr['y3']['n']:>5} {yr['y3']['wr']:>5.1f}% {yr['y3']['pf']:>7.3f} ${yr['y3']['net']:>+8.0f} {yr['y3']['sharpe']:>6.2f}  "
              f"{all_pass:>5}")

    # ==================================================================
    # COMPARISON: Best robust exit vs current config
    # ==================================================================
    print(f"\n{'='*110}")
    print("COMPARISON: Best Robust Exit Config vs Current Production Config")
    print(f"{'='*110}")

    # Current config on SM(10/12/200/100)
    current_label = 'Current SM(10/12/200/100)'
    print(f"\n  CURRENT: SM(10/12/200/100), TP1=7, TP2=25, SL=40")
    for year_key in ['y1', 'y2', 'y3']:
        r = test_a_results[current_label][year_key]
        print(f"    {year_labels[year_key]}: {r['n']} trades, WR {r['wr']:.1f}%, "
              f"PF {r['pf']:.3f}, ${r['net']:+.0f}, Sharpe {r['sharpe']:.3f}")
    current_total = sum(test_a_results[current_label][y]['net'] for y in ['y1', 'y2', 'y3'])
    current_pfs = [test_a_results[current_label][y]['pf'] for y in ['y1', 'y2', 'y3']]
    current_worst_pf = min(current_pfs)
    current_all_pass = all(pf > 1.0 for pf in current_pfs)
    print(f"    Total: ${current_total:+,.0f}, Worst PF: {current_worst_pf:.3f}, "
          f"All Years Pass: {'YES' if current_all_pass else 'NO'}")

    # Current exits on SM(12/12/200/80) (from Test A)
    robust_label_short = 'Robust SM(12/12/200/80)'
    print(f"\n  CURRENT EXITS ON ROBUST SM: SM(12/12/200/80), TP1=7, TP2=25, SL=40")
    for year_key in ['y1', 'y2', 'y3']:
        r = test_a_results[robust_label_short][year_key]
        print(f"    {year_labels[year_key]}: {r['n']} trades, WR {r['wr']:.1f}%, "
              f"PF {r['pf']:.3f}, ${r['net']:+.0f}, Sharpe {r['sharpe']:.3f}")
    robust_current_total = sum(test_a_results[robust_label_short][y]['net'] for y in ['y1', 'y2', 'y3'])
    robust_current_pfs = [test_a_results[robust_label_short][y]['pf'] for y in ['y1', 'y2', 'y3']]
    robust_current_worst_pf = min(robust_current_pfs)
    robust_current_all_pass = all(pf > 1.0 for pf in robust_current_pfs)
    print(f"    Total: ${robust_current_total:+,.0f}, Worst PF: {robust_current_worst_pf:.3f}, "
          f"All Years Pass: {'YES' if robust_current_all_pass else 'NO'}")

    # Best robust exit config (from Test B, if any pass)
    if passing:
        best = passing[0]
        tp1, tp2, sl = best['cfg']
        print(f"\n  BEST ROBUST EXIT: SM(12/12/200/80), TP1={tp1}, TP2={tp2}, SL={sl}")
        for year_key in ['y1', 'y2', 'y3']:
            print(f"    {year_labels[year_key]}: {best[f'{year_key}_n']} trades, "
                  f"WR {best[f'{year_key}_wr']:.1f}%, "
                  f"PF {best[f'{year_key}_pf']:.3f}, "
                  f"${best[f'{year_key}_net']:+.0f}, "
                  f"Sharpe {best[f'{year_key}_sharpe']:.3f}")
        print(f"    Total: ${best['total_net']:+,.0f}, Worst PF: {best['worst_pf']:.3f}, "
              f"All Years Pass: YES")

        # Delta
        delta_net = best['total_net'] - current_total
        delta_worst_pf = best['worst_pf'] - current_worst_pf
        print(f"\n  DELTA vs Current:")
        print(f"    Total Net$: ${delta_net:+,.0f}")
        print(f"    Worst PF:   {delta_worst_pf:+.3f}")
        if not current_all_pass:
            print(f"    Current FAILS at least one year. Best robust PASSES all 3.")
    else:
        print(f"\n  No exit configs pass all 3 years with SM(12/12/200/80).")

    # ==================================================================
    # CLUSTER ANALYSIS
    # ==================================================================
    if passing:
        print(f"\n{'='*110}")
        print("CLUSTER ANALYSIS — What exit param values appear in passing configs?")
        print(f"{'='*110}")

        from collections import defaultdict
        for pidx, pname in enumerate(['TP1', 'TP2', 'SL']):
            counts = defaultdict(int)
            for p in passing:
                val = p['cfg'][pidx]
                counts[val] += 1
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            total = len(passing)
            dist_str = "  ".join(f"{v}:{c}({c/total*100:.0f}%)" for v, c in sorted_counts)
            print(f"  {pname:>4}: {dist_str}")

    # ==================================================================
    # DONE
    # ==================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*110}")
    print(f"SWEEP COMPLETE — {total_time:.0f}s total")
    print(f"  Test A: SM head-to-head on fixed exits (2 configs x 3 years)")
    print(f"  Test B: {total_combos} exit configs x 3 years = {total_tests} backtests")
    print(f"  Passing (all 3 years PF > 1.0): {len(passing)}/{total_combos} ({len(passing)/total_combos*100:.1f}%)")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
