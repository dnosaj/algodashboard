#!/usr/bin/env python3
"""
Structure Exit Validation on Robust SM(12/12/200/80) — 3 Years
===============================================================
Validates the structure exit (pivot-based swing detection) with the NEW
robust SM(12/12/200/80) across 3 years of MNQ data.

Test A: Fixed TP2=30 vs Structure Exit (LB=50, PR=2, buffer=2, cap=30)
Test B: Structure exit param sweep (LB x PR x buffer = 24 configs x 3 years)

All configs use:
  SM(12/12/200/80) SM_T=0.0 RSI(8/60/40) CD=20
  TP1=10 SL=30 Entry cutoff 13:00 ET
  entry_qty=2 partial_qty=1 SL->BE after TP1

Usage:
    cd backtesting_engine && python3 strategies/structure_exit_robust_validation.py
"""

import sys
import time
import warnings
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

from structure_exit_common import (
    compute_swing_levels,
    run_backtest_structure_exit,
    score_structure_trades,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"
MNQ_DOLLAR_PER_PT = 2.0
MNQ_COMMISSION = 0.52

# Robust SM params
SM_INDEX = 12
SM_FLOW = 12
SM_NORM = 200
SM_EMA = 80

# vScalpC entry params
RSI_LEN = 8
RSI_BUY = 60
RSI_SELL = 40
COOLDOWN = 20
SM_THRESHOLD = 0.0
ENTRY_END_ET = 13 * 60  # 13:00 ET

# Exit params from the validated robust sweep
TP1_PTS = 10
SL_PTS = 30
FIXED_TP2 = 30  # Baseline fixed TP2

# Structure exit current params
CURRENT_LB = 50
CURRENT_PR = 2
CURRENT_BUF = 2.0
MAX_TP2_CAP = 30  # Crash-safety cap for structure exit


# ---------------------------------------------------------------------------
# Data Loading (reuse from vscalpc_3year_sweep.py)
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
# Precompute SM + RSI for one year
# ---------------------------------------------------------------------------

def precompute_year(df):
    """Compute SM(12/12/200/80) and 5-min RSI(8) mapping for one year.

    Returns dict with arrays needed for run_backtest_structure_exit.
    """
    closes = df['Close'].values
    volumes = df['Volume'].values

    sm = compute_smart_money(
        closes, volumes,
        index_period=SM_INDEX, flow_period=SM_FLOW,
        norm_period=SM_NORM, ema_len=SM_EMA,
    )

    # 5-min RSI mapping
    df_copy = df.copy()
    df_copy['SM_Net'] = sm
    df_5m = resample_to_5min(df_copy)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values,
        df_5m['Close'].values, rsi_len=RSI_LEN,
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
# Run fixed TP2 backtest (baseline)
# ---------------------------------------------------------------------------

def run_fixed_tp2(arrays, tp2_pts):
    """Run structure exit backtest with NaN swings = effectively fixed TP2 only."""
    n = len(arrays['opens'])
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    trades = run_backtest_structure_exit(
        arrays['opens'], arrays['highs'], arrays['lows'],
        arrays['closes'], arrays['sm'], arrays['times'],
        arrays['rsi_curr'], arrays['rsi_prev'],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=SM_THRESHOLD,
        cooldown_bars=COOLDOWN,
        max_loss_pts=SL_PTS,
        tp1_pts=TP1_PTS,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        swing_buffer_pts=0,
        min_profit_pts=0,
        use_high_low=False,
        max_tp2_pts=tp2_pts,
        move_sl_to_be_after_tp1=True,
        breakeven_after_bars=0,  # No BE_TIME for vScalpC
        eod_minutes_et=NY_CLOSE_ET,
        entry_end_et=ENTRY_END_ET,
    )

    return trades


# ---------------------------------------------------------------------------
# Run structure exit backtest
# ---------------------------------------------------------------------------

def run_structure(arrays, lb, pr, buf, cap):
    """Run structure exit backtest with given swing params."""
    swing_highs, swing_lows = compute_swing_levels(
        arrays['highs'], arrays['lows'],
        lookback=lb,
        swing_type="pivot",
        pivot_right=pr,
    )

    trades = run_backtest_structure_exit(
        arrays['opens'], arrays['highs'], arrays['lows'],
        arrays['closes'], arrays['sm'], arrays['times'],
        arrays['rsi_curr'], arrays['rsi_prev'],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=SM_THRESHOLD,
        cooldown_bars=COOLDOWN,
        max_loss_pts=SL_PTS,
        tp1_pts=TP1_PTS,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        swing_buffer_pts=buf,
        min_profit_pts=0,
        use_high_low=False,
        max_tp2_pts=cap,
        move_sl_to_be_after_tp1=True,
        breakeven_after_bars=0,  # No BE_TIME for vScalpC
        eod_minutes_et=NY_CLOSE_ET,
        entry_end_et=ENTRY_END_ET,
    )

    return trades


# ---------------------------------------------------------------------------
# Format runner exit breakdown
# ---------------------------------------------------------------------------

def fmt_runner_exits(sc):
    """Format runner exit reason breakdown."""
    if sc is None:
        return ""
    exits = sc.get("runner_exits", {})
    parts = []
    for key in ["structure", "TP_cap", "SL", "BE", "EOD"]:
        if key in exits:
            parts.append(f"{key}:{exits[key]}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 120)
    print("STRUCTURE EXIT VALIDATION — Robust SM(12/12/200/80) x 3 Years")
    print(f"  SM(12/12/200/80) SM_T=0.0 RSI(8/60/40) CD=20")
    print(f"  TP1={TP1_PTS} SL={SL_PTS} Entry cutoff 13:00 ET")
    print(f"  entry_qty=2 partial_qty=1 SL->BE after TP1")
    print("=" * 120)

    df_y1, df_y2, df_y3 = load_all_years()
    years_data = {'y1': df_y1, 'y2': df_y2, 'y3': df_y3}
    year_labels = {'y1': 'Y1 (Feb23-24)', 'y2': 'Y2 (Feb24-25)', 'y3': 'Y3 (Feb25-26)'}

    # Precompute SM + RSI for all 3 years
    precomputed = {}
    for year_key, df in years_data.items():
        precomputed[year_key] = precompute_year(df)
        elapsed = time.time() - t_start
        print(f"  Precomputed {year_labels[year_key]} [{elapsed:.0f}s]")

    # ==================================================================
    # TEST A: Fixed TP2=30 vs Structure Exit (current params)
    # ==================================================================
    print(f"\n{'='*120}")
    print("TEST A: Fixed TP2=30 vs Structure Exit (LB=50, PR=2, Buf=2.0, Cap=30)")
    print(f"{'='*120}")

    print(f"\n  {'Config':>35}  {'Year':>15}  {'N':>5} {'WR%':>6} {'PF':>7} "
          f"{'Net$':>10} {'MaxDD$':>10} {'Sharpe':>7}  {'Runner Exits':>35}")
    print(f"  {'-'*35}  {'-'*15}  {'-'*5} {'-'*6} {'-'*7} "
          f"{'-'*10} {'-'*10} {'-'*7}  {'-'*35}")

    test_a_results = {}

    # --- Fixed TP2=30 ---
    test_a_results['fixed_tp2'] = {}
    for year_key in ['y1', 'y2', 'y3']:
        trades = run_fixed_tp2(precomputed[year_key], FIXED_TP2)
        sc = score_structure_trades(trades, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
        test_a_results['fixed_tp2'][year_key] = sc
        if sc:
            print(f"  {'Fixed TP2=30':>35}  {year_labels[year_key]:>15}  "
                  f"{sc['count']:>5} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                  f"${sc['net_dollar']:>+9.0f} ${sc['max_dd_dollar']:>9.0f} "
                  f"{sc['sharpe']:>7.3f}  {fmt_runner_exits(sc):>35}")
        else:
            print(f"  {'Fixed TP2=30':>35}  {year_labels[year_key]:>15}  NO TRADES")

    # --- Structure Exit (current params) ---
    test_a_results['structure_current'] = {}
    for year_key in ['y1', 'y2', 'y3']:
        trades = run_structure(precomputed[year_key], CURRENT_LB, CURRENT_PR,
                               CURRENT_BUF, MAX_TP2_CAP)
        sc = score_structure_trades(trades, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
        test_a_results['structure_current'][year_key] = sc
        if sc:
            print(f"  {'Structure LB=50/PR=2/Buf=2.0/Cap=30':>35}  {year_labels[year_key]:>15}  "
                  f"{sc['count']:>5} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                  f"${sc['net_dollar']:>+9.0f} ${sc['max_dd_dollar']:>9.0f} "
                  f"{sc['sharpe']:>7.3f}  {fmt_runner_exits(sc):>35}")
        else:
            print(f"  {'Structure LB=50/PR=2/Buf=2.0/Cap=30':>35}  {year_labels[year_key]:>15}  NO TRADES")

    # --- Test A Summary ---
    print(f"\n  --- TEST A SUMMARY ---")
    print(f"  {'Config':>35}  {'Total$':>10} {'WorstPF':>8} {'WorstSharpe':>12} {'AllPass':>8}")
    print(f"  {'-'*35}  {'-'*10} {'-'*8} {'-'*12} {'-'*8}")

    for label, key in [("Fixed TP2=30", "fixed_tp2"),
                       ("Structure LB=50/PR=2/Buf=2.0", "structure_current")]:
        r = test_a_results[key]
        total_net = sum(r[y]['net_dollar'] for y in ['y1', 'y2', 'y3'] if r[y])
        pfs = [r[y]['pf'] for y in ['y1', 'y2', 'y3'] if r[y]]
        sharpes = [r[y]['sharpe'] for y in ['y1', 'y2', 'y3'] if r[y]]
        worst_pf = min(pfs) if pfs else 0
        worst_sharpe = min(sharpes) if sharpes else 0
        all_pass = "YES" if pfs and all(pf > 1.0 for pf in pfs) else "NO"
        print(f"  {label:>35}  ${total_net:>+9.0f} {worst_pf:>8.3f} {worst_sharpe:>12.3f} {all_pass:>8}")

    # --- Delta comparison ---
    print(f"\n  DELTA (Structure vs Fixed TP2):")
    for year_key in ['y1', 'y2', 'y3']:
        f_sc = test_a_results['fixed_tp2'][year_key]
        s_sc = test_a_results['structure_current'][year_key]
        if f_sc and s_sc:
            d_net = s_sc['net_dollar'] - f_sc['net_dollar']
            d_pf = s_sc['pf'] - f_sc['pf']
            d_sharpe = s_sc['sharpe'] - f_sc['sharpe']
            d_dd = s_sc['max_dd_dollar'] - f_sc['max_dd_dollar']
            print(f"    {year_labels[year_key]:>15}: Net$ {d_net:>+8.0f}, "
                  f"PF {d_pf:>+.3f}, Sharpe {d_sharpe:>+.3f}, MaxDD {d_dd:>+.0f}")

    # ==================================================================
    # TEST B: Structure exit param sweep
    # ==================================================================
    print(f"\n{'='*120}")
    print("TEST B: Structure Exit Param Sweep — 24 configs x 3 years")
    print(f"  LB: [30, 40, 50, 60]  |  PR: [2, 3]  |  Buffer: [1.0, 2.0, 3.0]")
    print(f"  All with cap={MAX_TP2_CAP} | TP1={TP1_PTS} SL={SL_PTS}")
    print(f"{'='*120}")

    LB_VALUES = [30, 40, 50, 60]
    PR_VALUES = [2, 3]
    BUF_VALUES = [1.0, 2.0, 3.0]

    combos = list(product(LB_VALUES, PR_VALUES, BUF_VALUES))
    total_combos = len(combos)
    total_tests = total_combos * 3
    print(f"  {total_combos} structure configs x 3 years = {total_tests} backtests")

    results = {}
    done = 0
    for lb, pr, buf in combos:
        cfg_key = (lb, pr, buf)
        results[cfg_key] = {}

        for year_key in ['y1', 'y2', 'y3']:
            trades = run_structure(precomputed[year_key], lb, pr, buf, MAX_TP2_CAP)
            sc = score_structure_trades(trades, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
            results[cfg_key][year_key] = sc
            done += 1

        if done % 12 == 0:
            elapsed = time.time() - t_start
            print(f"  ... {done}/{total_tests} backtests done [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"  Sweep complete: {total_tests} backtests in {elapsed:.0f}s")

    # ==================================================================
    # FULL TABLE: All 24 configs with per-year detail
    # ==================================================================
    print(f"\n{'='*120}")
    print("FULL RESULTS TABLE — All 24 structure configs x 3 years")
    print(f"{'='*120}")

    print(f"\n  {'LB':>4} {'PR':>3} {'Buf':>5}  "
          f"{'Y1_N':>5} {'Y1_WR':>6} {'Y1_PF':>7} {'Y1_Net$':>9} {'Y1_DD$':>8} {'Y1_Sh':>6}  "
          f"{'Y2_N':>5} {'Y2_WR':>6} {'Y2_PF':>7} {'Y2_Net$':>9} {'Y2_DD$':>8} {'Y2_Sh':>6}  "
          f"{'Y3_N':>5} {'Y3_WR':>6} {'Y3_PF':>7} {'Y3_Net$':>9} {'Y3_DD$':>8} {'Y3_Sh':>6}  "
          f"{'Pass':>5}")
    sep_y = f"{'-'*5} {'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*6}  "
    print(f"  {'-'*4} {'-'*3} {'-'*5}  " + sep_y * 3 + f"{'-'*5}")

    for lb, pr, buf in combos:
        cfg_key = (lb, pr, buf)
        yr = results[cfg_key]
        pfs = []
        for y in ['y1', 'y2', 'y3']:
            if yr[y]:
                pfs.append(yr[y]['pf'])
        all_pass = "YES" if pfs and len(pfs) == 3 and all(pf > 1.0 for pf in pfs) else "no"

        row_parts = []
        for y in ['y1', 'y2', 'y3']:
            if yr[y]:
                row_parts.append(
                    f"{yr[y]['count']:>5} {yr[y]['win_rate']:>5.1f}% {yr[y]['pf']:>7.3f} "
                    f"${yr[y]['net_dollar']:>+8.0f} ${yr[y]['max_dd_dollar']:>7.0f} "
                    f"{yr[y]['sharpe']:>6.2f}"
                )
            else:
                row_parts.append(f"{'--':>5} {'--':>6} {'--':>7} {'--':>9} {'--':>8} {'--':>6}")

        print(f"  {lb:>4} {pr:>3} {buf:>5.1f}  {'  '.join(row_parts)}  {all_pass:>5}")

    # ==================================================================
    # Filter: all 3 years PF > 1.0
    # ==================================================================
    passing = []
    for cfg_key, year_results in results.items():
        pfs = []
        nets = []
        sharpes = []
        for y in ['y1', 'y2', 'y3']:
            if year_results[y]:
                pfs.append(year_results[y]['pf'])
                nets.append(year_results[y]['net_dollar'])
                sharpes.append(year_results[y]['sharpe'])
            else:
                pfs.append(0)
                nets.append(0)
                sharpes.append(0)

        if all(pf > 1.0 for pf in pfs):
            worst_pf = min(pfs)
            total_net = sum(nets)
            worst_sharpe = min(sharpes)
            passing.append({
                'cfg': cfg_key,
                'worst_pf': worst_pf,
                'total_net': total_net,
                'worst_sharpe': worst_sharpe,
                'year_results': year_results,
            })

    print(f"\n  {len(passing)} of {total_combos} structure configs pass ALL 3 years (PF > 1.0)")
    print(f"  Pass rate: {len(passing)/total_combos*100:.1f}%")

    # Sort by worst-year PF
    passing.sort(key=lambda x: x['worst_pf'], reverse=True)

    if passing:
        print(f"\n  ALL PASSING CONFIGS (sorted by worst-year PF):")
        print(f"  {'Rank':>4}  {'LB':>4} {'PR':>3} {'Buf':>5}  "
              f"{'Y1 PF':>7} {'Y2 PF':>7} {'Y3 PF':>7} {'WrstPF':>7}  "
              f"{'Y1 Net$':>9} {'Y2 Net$':>9} {'Y3 Net$':>9} {'Total$':>9}  "
              f"{'Y1 Sh':>6} {'Y2 Sh':>6} {'Y3 Sh':>6}")
        print(f"  {'-'*4}  {'-'*4} {'-'*3} {'-'*5}  "
              f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}  "
              f"{'-'*9} {'-'*9} {'-'*9} {'-'*9}  "
              f"{'-'*6} {'-'*6} {'-'*6}")

        for i, p in enumerate(passing):
            lb, pr, buf = p['cfg']
            yr = p['year_results']
            print(f"  {i+1:>4}  {lb:>4} {pr:>3} {buf:>5.1f}  "
                  f"{yr['y1']['pf']:>7.3f} {yr['y2']['pf']:>7.3f} {yr['y3']['pf']:>7.3f} "
                  f"{p['worst_pf']:>7.3f}  "
                  f"${yr['y1']['net_dollar']:>+8.0f} ${yr['y2']['net_dollar']:>+8.0f} "
                  f"${yr['y3']['net_dollar']:>+8.0f} ${p['total_net']:>+8.0f}  "
                  f"{yr['y1']['sharpe']:>6.2f} {yr['y2']['sharpe']:>6.2f} "
                  f"{yr['y3']['sharpe']:>6.2f}")

        # --- Top 5 detailed ---
        print(f"\n  TOP 5 CONFIGS (by worst-year PF) — detailed breakdown:")
        for i, p in enumerate(passing[:5]):
            lb, pr, buf = p['cfg']
            yr = p['year_results']
            print(f"\n  #{i+1}: LB={lb} PR={pr} Buffer={buf}")
            for year_key in ['y1', 'y2', 'y3']:
                sc = yr[year_key]
                if sc:
                    print(f"    {year_labels[year_key]}: {sc['count']} trades, "
                          f"WR {sc['win_rate']:.1f}%, PF {sc['pf']:.3f}, "
                          f"${sc['net_dollar']:+.0f}, MaxDD ${sc['max_dd_dollar']:.0f}, "
                          f"Sharpe {sc['sharpe']:.3f}")
                    print(f"      Runner exits: {fmt_runner_exits(sc)}")
            print(f"    3-Year Total: ${p['total_net']:+,.0f}, "
                  f"Worst PF: {p['worst_pf']:.3f}, Worst Sharpe: {p['worst_sharpe']:.3f}")
    else:
        print(f"\n  NO CONFIGS PASS ALL 3 YEARS.")

    # ==================================================================
    # COMPARISON: Structure exit vs Fixed TP2
    # ==================================================================
    print(f"\n{'='*120}")
    print("COMPARISON: Best Structure Config vs Fixed TP2=30 vs Current Structure (LB=50/PR=2/Buf=2)")
    print(f"{'='*120}")

    # Fixed TP2 summary
    print(f"\n  BASELINE: Fixed TP2={FIXED_TP2}")
    for year_key in ['y1', 'y2', 'y3']:
        sc = test_a_results['fixed_tp2'][year_key]
        if sc:
            print(f"    {year_labels[year_key]}: {sc['count']} trades, WR {sc['win_rate']:.1f}%, "
                  f"PF {sc['pf']:.3f}, ${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']:.3f}")
    fixed_total = sum(test_a_results['fixed_tp2'][y]['net_dollar']
                      for y in ['y1', 'y2', 'y3']
                      if test_a_results['fixed_tp2'][y])
    fixed_pfs = [test_a_results['fixed_tp2'][y]['pf']
                 for y in ['y1', 'y2', 'y3']
                 if test_a_results['fixed_tp2'][y]]
    fixed_worst_pf = min(fixed_pfs) if fixed_pfs else 0
    print(f"    Total: ${fixed_total:+,.0f}, Worst PF: {fixed_worst_pf:.3f}")

    # Current structure params
    print(f"\n  CURRENT STRUCTURE: LB=50, PR=2, Buf=2.0, Cap=30")
    for year_key in ['y1', 'y2', 'y3']:
        sc = test_a_results['structure_current'][year_key]
        if sc:
            print(f"    {year_labels[year_key]}: {sc['count']} trades, WR {sc['win_rate']:.1f}%, "
                  f"PF {sc['pf']:.3f}, ${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']:.3f}")
    current_total = sum(test_a_results['structure_current'][y]['net_dollar']
                        for y in ['y1', 'y2', 'y3']
                        if test_a_results['structure_current'][y])
    current_pfs = [test_a_results['structure_current'][y]['pf']
                   for y in ['y1', 'y2', 'y3']
                   if test_a_results['structure_current'][y]]
    current_worst_pf = min(current_pfs) if current_pfs else 0
    print(f"    Total: ${current_total:+,.0f}, Worst PF: {current_worst_pf:.3f}")

    # Best structure from sweep
    if passing:
        best = passing[0]
        lb, pr, buf = best['cfg']
        yr = best['year_results']
        print(f"\n  BEST SWEEP: LB={lb}, PR={pr}, Buf={buf}, Cap={MAX_TP2_CAP}")
        for year_key in ['y1', 'y2', 'y3']:
            sc = yr[year_key]
            if sc:
                print(f"    {year_labels[year_key]}: {sc['count']} trades, WR {sc['win_rate']:.1f}%, "
                      f"PF {sc['pf']:.3f}, ${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']:.3f}")
        print(f"    Total: ${best['total_net']:+,.0f}, Worst PF: {best['worst_pf']:.3f}")

        # Deltas
        print(f"\n  DELTA: Best Sweep vs Fixed TP2:")
        print(f"    Net$: ${best['total_net'] - fixed_total:+,.0f}")
        print(f"    Worst PF: {best['worst_pf'] - fixed_worst_pf:+.3f}")

        print(f"\n  DELTA: Best Sweep vs Current Structure:")
        print(f"    Net$: ${best['total_net'] - current_total:+,.0f}")
        print(f"    Worst PF: {best['worst_pf'] - current_worst_pf:+.3f}")

        # Is best == current?
        if best['cfg'] == (CURRENT_LB, CURRENT_PR, CURRENT_BUF):
            print(f"\n  >>> CURRENT PARAMS ARE ALREADY OPTIMAL in the sweep grid.")
        else:
            print(f"\n  >>> BEST STRUCTURE PARAMS DIFFER from current "
                  f"(LB={lb}/PR={pr}/Buf={buf} vs LB={CURRENT_LB}/PR={CURRENT_PR}/Buf={CURRENT_BUF})")

    # ==================================================================
    # CLUSTER ANALYSIS
    # ==================================================================
    if passing:
        print(f"\n{'='*120}")
        print("CLUSTER ANALYSIS — Which param values appear in passing configs?")
        print(f"{'='*120}")

        from collections import defaultdict
        param_names = ['LB', 'PR', 'Buffer']
        for pidx, pname in enumerate(param_names):
            counts = defaultdict(int)
            for p in passing:
                val = p['cfg'][pidx]
                counts[val] += 1
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            total = len(passing)
            dist_str = "  ".join(f"{v}:{c}({c/total*100:.0f}%)" for v, c in sorted_counts)
            print(f"  {pname:>8}: {dist_str}")

    # ==================================================================
    # RUNNER EXIT BREAKDOWN (for best config)
    # ==================================================================
    if passing:
        print(f"\n{'='*120}")
        print("RUNNER EXIT BREAKDOWN — Best config across 3 years")
        print(f"{'='*120}")

        best = passing[0]
        lb, pr, buf = best['cfg']
        yr = best['year_results']

        for year_key in ['y1', 'y2', 'y3']:
            sc = yr[year_key]
            if sc:
                scalp_exits = sc.get('scalp_exits', {})
                runner_exits = sc.get('runner_exits', {})
                print(f"\n  {year_labels[year_key]}:")
                print(f"    Scalp exits:  {scalp_exits}")
                print(f"    Runner exits: {runner_exits}")
                # Calculate structure exit rate
                total_runner = sum(runner_exits.values())
                struct_count = runner_exits.get('structure', 0)
                struct_rate = struct_count / total_runner * 100 if total_runner > 0 else 0
                print(f"    Structure exit rate: {struct_count}/{total_runner} "
                      f"({struct_rate:.1f}%)")

    # ==================================================================
    # DONE
    # ==================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*120}")
    print(f"VALIDATION COMPLETE — {total_time:.0f}s total")
    print(f"  Test A: Fixed TP2=30 vs Structure LB=50/PR=2/Buf=2.0 x 3 years")
    print(f"  Test B: {total_combos} structure configs x 3 years = {total_tests} backtests")
    if passing:
        print(f"  Passing (all 3 years PF > 1.0): {len(passing)}/{total_combos} ({len(passing)/total_combos*100:.1f}%)")
        best = passing[0]
        lb, pr, buf = best['cfg']
        print(f"  Best: LB={lb} PR={pr} Buf={buf} — Worst PF {best['worst_pf']:.3f}, "
              f"Total ${best['total_net']:+,.0f}")
    else:
        print(f"  Passing: 0/{total_combos} — no configs pass all 3 years")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
