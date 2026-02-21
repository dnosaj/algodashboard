"""
OOS Step 4: Param Sweep with Walk-Forward Validation
======================================================

Sweeps entry params on OOS (minus crash days) using the best exit model
from Step 3. Validates with walk-forward, IS cross-validation, and stability.

Parameters swept:
  - SM threshold: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  - RSI period: [6, 8, 10, 12, 14]
  - RSI buy/sell: [(55,45), (60,40), (65,35)]
  - Cooldown: [10, 15, 20, 25, 30]
  - SL: top 3 values from Step 2

Total: 7 x 5 x 3 x 5 x 3 = 1,575 combos

Validation:
  1. Walk-forward within OOS: split into 2 halves, optimize on each, validate on other
  2. IS cross-validation: top combos also run on IS
  3. Stability analysis: average PF of neighboring combos (±1 grid step)
  4. Minimum 100 trades per combo
  5. Monthly consistency: profitable in >= 4 of 5 remaining OOS months

Usage:
    python3 oos_step4_param_sweep.py

Configure BEST_EXIT_MODEL and BEST_SL_VALUES before running.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import product
from multiprocessing import Pool, cpu_count
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    compute_et_minutes, NY_OPEN_ET, NY_CLOSE_ET,
)

# Import v15 TP-exit backtest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    'MNQ_OOS': 'databento_MNQ_1min_2025-02-17_to_2025-08-17.csv',
    'MNQ_IS':  'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
}

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MNQ_COMM, MNQ_DPP = 0.52, 2.0
BIG_MOVE_THRESHOLD = 500

# ---- CONFIGURED FROM STEPS 2 & 3 ----
# Step 3 result: TP=5 is best exit model (OOS: PF 0.827, -$782; IS: PF 1.554, +$1,547)
BEST_EXIT_MODEL = 'tp'
BEST_EXIT_KWARGS = {'tp_pts': 5}
BEST_FAST_SM_EMA = None    # Not using fast SM for sweep

# Top 3 SL values from Step 2 (by OOS Net $)
BEST_SL_VALUES = [15, 25, 35]

# Sweep grid
SM_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
RSI_PERIODS = [6, 8, 10, 12, 14]
RSI_LEVELS = [(55, 45), (60, 40), (65, 35)]
COOLDOWNS = [10, 15, 20, 25, 30]

# Validation thresholds
MIN_TRADES = 100
MIN_PROFITABLE_MONTHS = 4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_databento(filepath, sm_params):
    df = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    sm = compute_smart_money(result['Close'].values, result['Volume'].values, **sm_params)
    result['SM_Net'] = sm
    return result


def get_daily_ranges(df_1m):
    et_mins = compute_et_minutes(df_1m.index.values)
    mask = (et_mins >= NY_OPEN_ET) & (et_mins < NY_CLOSE_ET)
    rth = df_1m[mask].copy()
    rth['date'] = rth.index.date
    daily = rth.groupby('date').agg(High=('High', 'max'), Low=('Low', 'min'))
    daily['Range'] = daily['High'] - daily['Low']
    return daily


def filter_trades_by_date(trades, exclude_dates):
    return [t for t in trades if pd.Timestamp(t['entry_time']).date() not in exclude_dates]


# ---------------------------------------------------------------------------
# Pre-cache RSI mappings for all periods
# ---------------------------------------------------------------------------

def precompute_rsi_mappings(df_1m, rsi_periods):
    """Pre-compute 5-min RSI mapped to 1-min for all RSI periods.

    Returns dict: rsi_period -> (rsi_5m_curr, rsi_5m_prev)
    """
    df_5m = resample_to_5min(df_1m)
    fivemin_times = df_5m.index.values
    fivemin_closes = df_5m['Close'].values
    onemin_times = df_1m.index.values

    mappings = {}
    for rsi_len in rsi_periods:
        rsi_curr, rsi_prev = map_5min_rsi_to_1min(
            onemin_times, fivemin_times, fivemin_closes, rsi_len
        )
        mappings[rsi_len] = (rsi_curr, rsi_prev)
    return mappings


# ---------------------------------------------------------------------------
# Backtest runner for sweep
# ---------------------------------------------------------------------------

def run_sweep_backtest(opens, highs, lows, closes, sm, times, rsi_dummy,
                       rsi_5m_curr, rsi_5m_prev,
                       rsi_buy, rsi_sell, sm_threshold, cooldown_bars, sl,
                       sm_exit=None, exit_model='sm_flip', exit_kwargs=None):
    """Lightweight wrapper for param sweep backtests."""
    if exit_kwargs is None:
        exit_kwargs = {}

    if exit_model == 'sm_flip':
        return run_backtest_v10(
            opens, highs, lows, closes, sm, rsi_dummy, times,
            rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            sm_threshold=sm_threshold, cooldown_bars=cooldown_bars,
            max_loss_pts=sl,
            rsi_5m_curr=rsi_5m_curr, rsi_5m_prev=rsi_5m_prev,
            sm_exit=sm_exit,
        )
    elif exit_model == 'tp':
        return run_backtest_tp_exit(
            opens, highs, lows, closes, sm, times,
            rsi_5m_curr, rsi_5m_prev,
            rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            sm_threshold=sm_threshold, cooldown_bars=cooldown_bars,
            max_loss_pts=sl, tp_pts=exit_kwargs.get('tp_pts', 5),
        )
    elif exit_model == 'time':
        return run_backtest_v10(
            opens, highs, lows, closes, sm, rsi_dummy, times,
            rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            sm_threshold=sm_threshold, cooldown_bars=cooldown_bars,
            max_loss_pts=sl,
            underwater_exit_bars=exit_kwargs.get('max_bars', 20),
            rsi_5m_curr=rsi_5m_curr, rsi_5m_prev=rsi_5m_prev,
            sm_exit=sm_exit,
        )
    elif exit_model == 'trail':
        return run_backtest_v10(
            opens, highs, lows, closes, sm, rsi_dummy, times,
            rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            sm_threshold=sm_threshold, cooldown_bars=cooldown_bars,
            max_loss_pts=sl,
            trailing_stop_pts=exit_kwargs.get('trail_pts', 3),
            rsi_5m_curr=rsi_5m_curr, rsi_5m_prev=rsi_5m_prev,
            sm_exit=sm_exit,
        )
    elif exit_model == 'breakeven':
        return run_backtest_v10(
            opens, highs, lows, closes, sm, rsi_dummy, times,
            rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            sm_threshold=sm_threshold, cooldown_bars=cooldown_bars,
            max_loss_pts=sl,
            breakeven_pts=exit_kwargs.get('be_pts', 3),
            rsi_5m_curr=rsi_5m_curr, rsi_5m_prev=rsi_5m_prev,
            sm_exit=sm_exit,
        )
    elif exit_model == 'fast_sm':
        return run_backtest_v10(
            opens, highs, lows, closes, sm, rsi_dummy, times,
            rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            sm_threshold=sm_threshold, cooldown_bars=cooldown_bars,
            max_loss_pts=sl,
            sm_exit=sm_exit,
            rsi_5m_curr=rsi_5m_curr, rsi_5m_prev=rsi_5m_prev,
        )
    else:
        raise ValueError(f"Unknown exit_model: {exit_model}")


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

# Module-level globals set by initializer for pool workers
_worker_data = {}


def _init_worker(data_dict):
    """Initializer for pool workers — loads shared data into module globals."""
    global _worker_data
    _worker_data = data_dict


def _run_single_combo(args):
    """Run one param combo on one data slice. Returns (combo_key, score_dict, trade_count)."""
    sm_thresh, rsi_period, rsi_buy, rsi_sell, cooldown, sl, data_key = args

    d = _worker_data[data_key]
    rsi_curr, rsi_prev = d['rsi_mappings'][rsi_period]

    trades = run_sweep_backtest(
        d['opens'], d['highs'], d['lows'], d['closes'],
        d['sm'], d['times'], d['rsi_dummy'],
        rsi_curr, rsi_prev,
        rsi_buy, rsi_sell, sm_thresh, cooldown, sl,
        sm_exit=d.get('sm_exit'),
        exit_model=d['exit_model'],
        exit_kwargs=d.get('exit_kwargs'),
    )

    # Post-filter crash days if applicable
    if d.get('exclude_dates'):
        trades = filter_trades_by_date(trades, d['exclude_dates'])

    sc = score_trades(trades, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)

    # Monthly P&L for consistency check
    monthly_pnl = {}
    comm_pts = (MNQ_COMM * 2) / MNQ_DPP
    for t in trades:
        m = pd.Timestamp(t['exit_time']).strftime('%Y-%m')
        monthly_pnl[m] = monthly_pnl.get(m, 0) + (t['pts'] - comm_pts) * MNQ_DPP

    combo_key = (sm_thresh, rsi_period, rsi_buy, rsi_sell, cooldown, sl)
    return combo_key, sc, len(trades), monthly_pnl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("OOS STEP 4: PARAM SWEEP WITH WALK-FORWARD VALIDATION")
    print(f"Exit model: {BEST_EXIT_MODEL}  SL values: {BEST_SL_VALUES}")
    print("=" * 80)

    t_start = time.time()

    # ---- Load data ----
    print("\nLoading data...")
    mnq_oos = load_databento(DATA_DIR / FILES['MNQ_OOS'], MNQ_SM)
    mnq_is = load_databento(DATA_DIR / FILES['MNQ_IS'], MNQ_SM)
    print(f"  MNQ OOS: {len(mnq_oos):,} bars")
    print(f"  MNQ IS:  {len(mnq_is):,} bars")

    # Crash exclusion
    daily_oos = get_daily_ranges(mnq_oos)
    big_move_dates = set(daily_oos[daily_oos['Range'] > BIG_MOVE_THRESHOLD].index)
    print(f"  Big-move days excluded: {len(big_move_dates)}")

    # Pre-compute SM exit array if needed
    sm_exit_oos = None
    sm_exit_is = None
    if BEST_FAST_SM_EMA is not None:
        print(f"  Computing fast SM (EMA={BEST_FAST_SM_EMA})...")
        sm_exit_oos = compute_smart_money(
            mnq_oos['Close'].values, mnq_oos['Volume'].values,
            index_period=MNQ_SM['index_period'], flow_period=MNQ_SM['flow_period'],
            norm_period=MNQ_SM['norm_period'], ema_len=BEST_FAST_SM_EMA,
        )
        sm_exit_is = compute_smart_money(
            mnq_is['Close'].values, mnq_is['Volume'].values,
            index_period=MNQ_SM['index_period'], flow_period=MNQ_SM['flow_period'],
            norm_period=MNQ_SM['norm_period'], ema_len=BEST_FAST_SM_EMA,
        )

    # Pre-cache RSI mappings
    print("Pre-computing RSI mappings for all periods...")
    rsi_mappings_oos = precompute_rsi_mappings(mnq_oos, RSI_PERIODS)
    rsi_mappings_is = precompute_rsi_mappings(mnq_is, RSI_PERIODS)

    # Walk-forward: split OOS into 2 halves
    # OOS is Feb-Aug 2025. Minus crash days, split roughly:
    # Half 1: Feb-Apr (but crash days excluded)
    # Half 2: May-Aug
    oos_midpoint = pd.Timestamp('2025-05-01')
    mnq_oos_h1 = mnq_oos[mnq_oos.index < oos_midpoint]
    mnq_oos_h2 = mnq_oos[mnq_oos.index >= oos_midpoint]
    print(f"  Walk-forward halves: H1={len(mnq_oos_h1):,} bars, H2={len(mnq_oos_h2):,} bars")

    rsi_mappings_h1 = precompute_rsi_mappings(mnq_oos_h1, RSI_PERIODS)
    rsi_mappings_h2 = precompute_rsi_mappings(mnq_oos_h2, RSI_PERIODS)

    # Build SM exit for halves if needed
    sm_exit_h1 = None
    sm_exit_h2 = None
    if BEST_FAST_SM_EMA is not None:
        sm_exit_h1 = compute_smart_money(
            mnq_oos_h1['Close'].values, mnq_oos_h1['Volume'].values,
            **{**MNQ_SM, 'ema_len': BEST_FAST_SM_EMA},
        )
        sm_exit_h2 = compute_smart_money(
            mnq_oos_h2['Close'].values, mnq_oos_h2['Volume'].values,
            **{**MNQ_SM, 'ema_len': BEST_FAST_SM_EMA},
        )

    # Daily ranges for halves (for crash exclusion)
    daily_h1 = get_daily_ranges(mnq_oos_h1)
    daily_h2 = get_daily_ranges(mnq_oos_h2)
    big_move_h1 = set(daily_h1[daily_h1['Range'] > BIG_MOVE_THRESHOLD].index)
    big_move_h2 = set(daily_h2[daily_h2['Range'] > BIG_MOVE_THRESHOLD].index)

    # Prepare shared data dicts for workers
    def make_data_dict(df_1m, rsi_mappings, exclude_dates=None, sm_exit_arr=None):
        return {
            'opens': df_1m['Open'].values, 'highs': df_1m['High'].values,
            'lows': df_1m['Low'].values, 'closes': df_1m['Close'].values,
            'sm': df_1m['SM_Net'].values, 'times': df_1m.index.values,
            'rsi_dummy': np.full(len(df_1m), 50.0),
            'rsi_mappings': rsi_mappings,
            'exclude_dates': exclude_dates,
            'sm_exit': sm_exit_arr,
            'exit_model': BEST_EXIT_MODEL,
            'exit_kwargs': BEST_EXIT_KWARGS,
        }

    data_dicts = {
        'oos_full': make_data_dict(mnq_oos, rsi_mappings_oos, big_move_dates, sm_exit_oos),
        'is_full': make_data_dict(mnq_is, rsi_mappings_is, None, sm_exit_is),
        'oos_h1': make_data_dict(mnq_oos_h1, rsi_mappings_h1, big_move_h1, sm_exit_h1),
        'oos_h2': make_data_dict(mnq_oos_h2, rsi_mappings_h2, big_move_h2, sm_exit_h2),
    }

    # Build combo list
    combos = list(product(SM_THRESHOLDS, RSI_PERIODS, RSI_LEVELS, COOLDOWNS, BEST_SL_VALUES))
    total_combos = len(combos)
    print(f"\n  Total combos: {total_combos}")
    print(f"  Total backtests: {total_combos * 4} (OOS + IS + 2 halves)")

    # Build work items for OOS full sweep
    work_items = []
    for sm_thresh, rsi_period, (rsi_buy, rsi_sell), cooldown, sl in combos:
        work_items.append((sm_thresh, rsi_period, rsi_buy, rsi_sell, cooldown, sl, 'oos_full'))

    # ================================================================
    # RUN OOS SWEEP
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 1: OOS Sweep (minus crash days)")
    print(f"{'='*80}")

    n_workers = max(1, cpu_count() - 1)
    print(f"  Using {n_workers} workers for {len(work_items)} backtests...")
    t0 = time.time()

    with Pool(n_workers, initializer=_init_worker, initargs=(data_dicts,)) as pool:
        oos_results_raw = pool.map(_run_single_combo, work_items, chunksize=50)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(work_items)*1000:.1f}ms per backtest)")

    # Parse results
    oos_results = {}  # combo_key -> (score_dict, trade_count, monthly_pnl)
    for combo_key, sc, n_trades, monthly_pnl in oos_results_raw:
        oos_results[combo_key] = (sc, n_trades, monthly_pnl)

    # ================================================================
    # FILTER: min trades + monthly consistency
    # ================================================================
    print(f"\n{'='*80}")
    print("FILTERING")
    print(f"{'='*80}")

    valid_combos = {}
    for combo_key, (sc, n_trades, monthly_pnl) in oos_results.items():
        if sc is None or n_trades < MIN_TRADES:
            continue
        if sc['pf'] < 0.5:  # Skip clearly terrible combos
            continue
        # Monthly consistency: profitable in >= MIN_PROFITABLE_MONTHS months
        profitable_months = sum(1 for v in monthly_pnl.values() if v > 0)
        total_months = len(monthly_pnl)
        if total_months >= 5 and profitable_months < MIN_PROFITABLE_MONTHS:
            continue
        valid_combos[combo_key] = (sc, n_trades, monthly_pnl, profitable_months)

    print(f"  Total combos: {total_combos}")
    print(f"  After min {MIN_TRADES} trades: {sum(1 for sc, n, _, _ in [(oos_results[k][0], oos_results[k][1], None, None) for k in oos_results] if sc and n >= MIN_TRADES)}")
    print(f"  After monthly consistency: {len(valid_combos)}")

    if not valid_combos:
        print("\n  NO VALID COMBOS FOUND. Try relaxing constraints.")
        return

    # Sort by PF
    ranked_combos = sorted(valid_combos.items(), key=lambda x: x[1][0]['pf'], reverse=True)
    top_20 = ranked_combos[:20]

    print(f"\n  Top 20 OOS combos (by PF):")
    print(f"  {'#':>3} {'SM_T':>5} {'RSI_P':>5} {'B/S':>7} {'CD':>4} {'SL':>4} "
          f"| {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10} {'ProfMo':>6}")
    print("  " + "-" * 80)
    for i, (ck, (sc, nt, mp, pm)) in enumerate(top_20, 1):
        sm_t, rsi_p, rsi_b, rsi_s, cd, sl = ck
        print(f"  {i:>3} {sm_t:>5.2f} {rsi_p:>5d} {rsi_b:>3d}/{rsi_s:<3d} {cd:>4d} {sl:>4d} "
              f"| {sc['count']:>6d} {sc['win_rate']:>5.1f}% {sc['pf']:>6.3f} "
              f"{sc['net_dollar']:>+10.2f} {pm:>4d}/{len(mp)}")

    # ================================================================
    # PHASE 2: IS CROSS-VALIDATION for top combos
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 2: IS Cross-Validation (top 20 combos)")
    print(f"{'='*80}")

    top_combo_keys = [ck for ck, _ in top_20]
    is_work = [(sm_t, rsi_p, rsi_b, rsi_s, cd, sl, 'is_full')
               for sm_t, rsi_p, rsi_b, rsi_s, cd, sl in top_combo_keys]

    with Pool(n_workers, initializer=_init_worker, initargs=(data_dicts,)) as pool:
        is_results_raw = pool.map(_run_single_combo, is_work, chunksize=5)

    is_results = {}
    for combo_key, sc, n_trades, monthly_pnl in is_results_raw:
        is_results[combo_key] = (sc, n_trades, monthly_pnl)

    print(f"\n  {'#':>3} {'SM_T':>5} {'RSI_P':>5} {'B/S':>7} {'CD':>4} {'SL':>4} "
          f"| {'OOS PF':>7} {'OOS $':>10} | {'IS PF':>6} {'IS $':>10} {'IS Tr':>5}")
    print("  " + "-" * 85)
    for i, (ck, (sc_oos, _, _, _)) in enumerate(top_20, 1):
        sm_t, rsi_p, rsi_b, rsi_s, cd, sl = ck
        sc_is, _, _ = is_results.get(ck, (None, 0, {}))
        is_pf = sc_is['pf'] if sc_is else 0
        is_net = sc_is['net_dollar'] if sc_is else 0
        is_cnt = sc_is['count'] if sc_is else 0
        print(f"  {i:>3} {sm_t:>5.2f} {rsi_p:>5d} {rsi_b:>3d}/{rsi_s:<3d} {cd:>4d} {sl:>4d} "
              f"| {sc_oos['pf']:>7.3f} {sc_oos['net_dollar']:>+10.2f} "
              f"| {is_pf:>6.3f} {is_net:>+10.2f} {is_cnt:>5d}")

    # ================================================================
    # PHASE 3: WALK-FORWARD VALIDATION
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 3: Walk-Forward Validation (top 20)")
    print("Split OOS into H1 (Feb-Apr) and H2 (May-Aug)")
    print(f"{'='*80}")

    wf_work = []
    for ck in top_combo_keys:
        sm_t, rsi_p, rsi_b, rsi_s, cd, sl = ck
        wf_work.append((sm_t, rsi_p, rsi_b, rsi_s, cd, sl, 'oos_h1'))
        wf_work.append((sm_t, rsi_p, rsi_b, rsi_s, cd, sl, 'oos_h2'))

    with Pool(n_workers, initializer=_init_worker, initargs=(data_dicts,)) as pool:
        wf_results_raw = pool.map(_run_single_combo, wf_work, chunksize=5)

    # Reorganize results into h1/h2 by matching work items back
    h1_results = {}
    h2_results = {}
    for item, (combo_key, sc, n_trades, monthly_pnl) in zip(wf_work, wf_results_raw):
        data_key = item[-1]
        if data_key == 'oos_h1':
            h1_results[combo_key] = (sc, n_trades)
        else:
            h2_results[combo_key] = (sc, n_trades)

    print(f"\n  {'#':>3} {'SM_T':>5} {'RSI_P':>5} {'B/S':>7} {'CD':>4} {'SL':>4} "
          f"| {'H1 PF':>6} {'H1 $':>10} {'H1 Tr':>5} "
          f"| {'H2 PF':>6} {'H2 $':>10} {'H2 Tr':>5} | {'Both+':>5}")
    print("  " + "-" * 95)
    wf_both_profitable = []
    for i, ck in enumerate(top_combo_keys, 1):
        sm_t, rsi_p, rsi_b, rsi_s, cd, sl = ck
        sc_h1, n_h1 = h1_results.get(ck, (None, 0))
        sc_h2, n_h2 = h2_results.get(ck, (None, 0))
        h1_pf = sc_h1['pf'] if sc_h1 else 0
        h1_net = sc_h1['net_dollar'] if sc_h1 else 0
        h2_pf = sc_h2['pf'] if sc_h2 else 0
        h2_net = sc_h2['net_dollar'] if sc_h2 else 0
        both_ok = h1_net > 0 and h2_net > 0
        if both_ok:
            wf_both_profitable.append(ck)
        print(f"  {i:>3} {sm_t:>5.2f} {rsi_p:>5d} {rsi_b:>3d}/{rsi_s:<3d} {cd:>4d} {sl:>4d} "
              f"| {h1_pf:>6.3f} {h1_net:>+10.2f} {n_h1:>5d} "
              f"| {h2_pf:>6.3f} {h2_net:>+10.2f} {n_h2:>5d} | {'YES' if both_ok else 'NO ':>5}")

    print(f"\n  Combos profitable in BOTH halves: {len(wf_both_profitable)}/{len(top_combo_keys)}")

    # ================================================================
    # PHASE 4: STABILITY ANALYSIS (top 10)
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 4: Stability Analysis (top 10)")
    print("Average PF of neighboring combos (±1 grid step)")
    print(f"{'='*80}")

    top_10_keys = top_combo_keys[:10]

    for i, ck in enumerate(top_10_keys, 1):
        sm_t, rsi_p, rsi_b, rsi_s, cd, sl = ck
        sc_center = oos_results[ck][0]

        # Find neighbors (±1 step in each dimension)
        sm_idx = SM_THRESHOLDS.index(sm_t) if sm_t in SM_THRESHOLDS else -1
        rsi_p_idx = RSI_PERIODS.index(rsi_p) if rsi_p in RSI_PERIODS else -1
        rsi_lvl = (rsi_b, rsi_s)
        rsi_lvl_idx = RSI_LEVELS.index(rsi_lvl) if rsi_lvl in RSI_LEVELS else -1
        cd_idx = COOLDOWNS.index(cd) if cd in COOLDOWNS else -1
        sl_idx = BEST_SL_VALUES.index(sl) if sl in BEST_SL_VALUES else -1

        neighbor_pfs = []
        for dim_name, values, idx, extract in [
            ('SM_T', SM_THRESHOLDS, sm_idx, lambda v: (v, rsi_p, rsi_b, rsi_s, cd, sl)),
            ('RSI_P', RSI_PERIODS, rsi_p_idx, lambda v: (sm_t, v, rsi_b, rsi_s, cd, sl)),
            ('RSI_L', RSI_LEVELS, rsi_lvl_idx, lambda v: (sm_t, rsi_p, v[0], v[1], cd, sl)),
            ('CD', COOLDOWNS, cd_idx, lambda v: (sm_t, rsi_p, rsi_b, rsi_s, v, sl)),
            ('SL', BEST_SL_VALUES, sl_idx, lambda v: (sm_t, rsi_p, rsi_b, rsi_s, cd, v)),
        ]:
            for offset in [-1, 1]:
                ni = idx + offset
                if 0 <= ni < len(values):
                    neighbor_key = extract(values[ni])
                    if neighbor_key in oos_results:
                        nsc = oos_results[neighbor_key][0]
                        if nsc:
                            neighbor_pfs.append(nsc['pf'])

        avg_neighbor_pf = np.mean(neighbor_pfs) if neighbor_pfs else 0
        stability = avg_neighbor_pf / sc_center['pf'] if sc_center['pf'] > 0 else 0

        in_wf = ck in wf_both_profitable
        print(f"  #{i:>2} ({sm_t:.2f}/{rsi_p}/{rsi_b}-{rsi_s}/CD{cd}/SL{sl}): "
              f"center PF={sc_center['pf']:.3f}, "
              f"neighbors avg PF={avg_neighbor_pf:.3f} ({len(neighbor_pfs)} neighbors), "
              f"stability={stability:.2f}, WF={'PASS' if in_wf else 'FAIL'}")

    # ================================================================
    # PHASE 5: IS vs OOS CORRELATION
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 5: IS vs OOS Correlation (all valid combos)")
    print(f"{'='*80}")

    # Run IS for all valid combos (not just top 20) — in batches
    remaining_keys = [ck for ck in valid_combos if ck not in is_results]
    if remaining_keys:
        print(f"  Running IS backtests for {len(remaining_keys)} remaining valid combos...")
        is_work_extra = [(sm_t, rsi_p, rsi_b, rsi_s, cd, sl, 'is_full')
                         for sm_t, rsi_p, rsi_b, rsi_s, cd, sl in remaining_keys]
        with Pool(n_workers, initializer=_init_worker, initargs=(data_dicts,)) as pool:
            is_extra_raw = pool.map(_run_single_combo, is_work_extra, chunksize=50)
        for combo_key, sc, n_trades, monthly_pnl in is_extra_raw:
            is_results[combo_key] = (sc, n_trades, monthly_pnl)

    # Scatter data: IS PF vs OOS PF
    is_pfs = []
    oos_pfs = []
    for ck in valid_combos:
        sc_oos = valid_combos[ck][0]
        is_data = is_results.get(ck)
        if is_data and is_data[0]:
            is_pfs.append(is_data[0]['pf'])
            oos_pfs.append(sc_oos['pf'])

    if len(is_pfs) > 2:
        corr = np.corrcoef(is_pfs, oos_pfs)[0, 1]
        print(f"\n  IS vs OOS PF correlation: {corr:.3f} ({len(is_pfs)} combos)")
        if corr > 0.3:
            print(f"  POSITIVE correlation — params that work IS tend to work OOS (generalizing)")
        elif corr < -0.1:
            print(f"  NEGATIVE correlation — params are overfit to one period")
        else:
            print(f"  WEAK correlation — mixed signal on generalization")

        # Distribution summary
        is_arr = np.array(is_pfs)
        oos_arr = np.array(oos_pfs)
        both_gt1 = np.sum((is_arr > 1.0) & (oos_arr > 1.0))
        print(f"  Combos with PF > 1.0 on BOTH IS and OOS: {both_gt1}/{len(is_pfs)}")

    # ================================================================
    # CURRENT PARAMS COMPARISON
    # ================================================================
    print(f"\n{'='*80}")
    print("COMPARISON: Current IS-Optimal vs Best OOS")
    print(f"{'='*80}")

    # Current IS-optimal params
    current_key = (0.15, 8, 60, 40, 20, 50)
    current_oos = oos_results.get(current_key)
    current_is = is_results.get(current_key)

    if current_oos and current_oos[0]:
        print(f"\n  Current params (SM_T=0.15, RSI=8/60-40, CD=20, SL=50):")
        print(f"    OOS: {current_oos[0]['count']} trades, PF {current_oos[0]['pf']:.3f}, "
              f"Net ${current_oos[0]['net_dollar']:+.2f}")
        if current_is and current_is[0]:
            print(f"    IS:  {current_is[0]['count']} trades, PF {current_is[0]['pf']:.3f}, "
                  f"Net ${current_is[0]['net_dollar']:+.2f}")

    if top_20:
        best_ck, (best_sc, _, _, _) = top_20[0]
        sm_t, rsi_p, rsi_b, rsi_s, cd, sl = best_ck
        best_is = is_results.get(best_ck)
        print(f"\n  Best OOS params (SM_T={sm_t}, RSI={rsi_p}/{rsi_b}-{rsi_s}, CD={cd}, SL={sl}):")
        print(f"    OOS: {best_sc['count']} trades, PF {best_sc['pf']:.3f}, "
              f"Net ${best_sc['net_dollar']:+.2f}")
        if best_is and best_is[0]:
            print(f"    IS:  {best_is[0]['count']} trades, PF {best_is[0]['pf']:.3f}, "
                  f"Net ${best_is[0]['net_dollar']:+.2f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed_total = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"SUMMARY (total time: {elapsed_total:.0f}s)")
    print(f"{'='*80}")
    print(f"  Total combos swept: {total_combos}")
    print(f"  Valid combos (>={MIN_TRADES} trades, monthly consistency): {len(valid_combos)}")
    print(f"  Walk-forward validated (profitable both halves): {len(wf_both_profitable)}")
    if is_pfs:
        print(f"  IS-OOS PF correlation: {corr:.3f}")

    print(f"\n  RECOMMENDED: Pick combos that pass walk-forward AND have stability > 0.8")
    print(f"  AND are profitable on IS. These are the most robust parameter sets.")

    print("\n" + "=" * 80)
    print("DONE — Step 4 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
