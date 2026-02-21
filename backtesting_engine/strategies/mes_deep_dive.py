"""
Task 4: MES Deep Dive
=======================
Run equivalent of Steps 1-3 on MES v9.4. MES has fundamentally different
params and may have different findings.

MES v9.4 current params:
  SM(20/12/400/255), RSI(10/55/45), CD=15, SL=0, EOD=15:30
  No stop loss (exits only on SM flip or EOD at 15:30 ET)
  $1.25/side commission, $5.00/pt (corrected from memory: MES comm varies by broker)

Step A: MFE/MAE Autopsy
Step B: Exit Model Comparison (TP exits)
Step C: Param Sweep (if TP works)
Step D: Regime interaction (does MNQ regime detector predict MES?)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    fmt_score, compute_et_minutes,
    NY_OPEN_ET, NY_CLOSE_ET, NY_LAST_ENTRY_ET,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit, compute_mfe_mae

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MES_SM = dict(index_period=20, flow_period=12, norm_period=400, ema_len=255)
MES_COMM, MES_DPP = 1.25, 5.0
MES_EOD_ET = 15 * 60 + 30  # 15:30 ET (930 minutes from midnight)

SPLIT = pd.Timestamp("2025-08-17", tz='UTC')

# MES crash exclusion: range > 150 pts (proportional to MES point value)
BIG_MOVE_THRESHOLD = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_combined():
    files = sorted(DATA_DIR.glob("databento_MES_1min_2025-*.csv"))
    print(f"  Loading MES: {[f.name for f in files]}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        result = pd.DataFrame()
        result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
        result['Open'] = pd.to_numeric(df['open'], errors='coerce')
        result['High'] = pd.to_numeric(df['high'], errors='coerce')
        result['Low'] = pd.to_numeric(df['low'], errors='coerce')
        result['Close'] = pd.to_numeric(df['close'], errors='coerce')
        result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        result = result.set_index('Time')
        dfs.append(result)
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    sm = compute_smart_money(
        combined['Close'].values, combined['Volume'].values, **MES_SM
    )
    combined['SM_Net'] = sm
    print(f"    {len(combined):,} bars total ({combined.index[0].date()} to {combined.index[-1].date()})")
    return combined


def prepare_arrays(df_1m, rsi_len):
    df_5m = resample_to_5min(df_1m)
    rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
        df_1m.index.values, df_5m.index.values, df_5m['Close'].values, rsi_len
    )
    rsi_dummy = np.full(len(df_1m), 50.0)
    return {
        'opens': df_1m['Open'].values, 'highs': df_1m['High'].values,
        'lows': df_1m['Low'].values, 'closes': df_1m['Close'].values,
        'sm': df_1m['SM_Net'].values, 'times': df_1m.index.values,
        'rsi': rsi_dummy, 'rsi_5m_curr': rsi_5m_curr, 'rsi_5m_prev': rsi_5m_prev,
    }


def split_trades(trades, split_ts):
    oos, is_trades = [], []
    for t in trades:
        exit_ts = pd.Timestamp(t['exit_time'])
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize('UTC')
        if exit_ts < split_ts:
            oos.append(t)
        else:
            is_trades.append(t)
    return oos, is_trades


# ---------------------------------------------------------------------------
# MES v9.4 backtest (SM flip exit, EOD=15:30, no SL)
# Reused from oos_12month_check.py
# ---------------------------------------------------------------------------

def run_mes_v94(opens, highs, lows, closes, sm, times,
                rsi_5m_curr, rsi_5m_prev,
                rsi_buy=55, rsi_sell=45,
                cooldown_bars=15, eod_minutes_et=930):
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]
        sm_bull = sm_prev > 0
        sm_bear = sm_prev < 0
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0
        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue
        if trade_state == 1:
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
        elif trade_state == -1:
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= cooldown_bars
            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
    return trades


# ---------------------------------------------------------------------------
# Daily range computation
# ---------------------------------------------------------------------------

def get_daily_ranges(df_1m):
    et_mins = compute_et_minutes(df_1m.index.values)
    mask = (et_mins >= NY_OPEN_ET) & (et_mins < NY_CLOSE_ET)
    rth = df_1m[mask].copy()
    rth['date'] = rth.index.date
    daily = rth.groupby('date').agg(
        High=('High', 'max'), Low=('Low', 'min'),
    )
    daily['Range'] = daily['High'] - daily['Low']
    return daily


def monthly_breakdown(trades, label, comm=MES_COMM, dpp=MES_DPP):
    monthly = defaultdict(list)
    for t in trades:
        m = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        monthly[m].append(t)

    print(f"\n  Monthly: {label}")
    print(f"  {'Month':<10} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Net $':>10}")
    print("  " + "-" * 50)
    for m in sorted(monthly.keys()):
        sc = score_trades(monthly[m], commission_per_side=comm, dollar_per_pt=dpp)
        if sc:
            print(f"  {m:<10} {sc['count']:>7d} {sc['win_rate']:>5.1f}% "
                  f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("TASK 4: MES DEEP DIVE")
    print("=" * 80)

    # Load data
    print("\nLoading combined 12-month MES data...")
    mes = load_combined()

    print("Preparing arrays (RSI=10)...")
    arr = prepare_arrays(mes, rsi_len=10)

    # ================================================================
    # Run MES v9.4 baseline
    # ================================================================
    print("\n" + "=" * 80)
    print("MES v9.4 BASELINE (SM flip, no SL, EOD 15:30)")
    print("=" * 80)

    v94_trades = run_mes_v94(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], arr['times'],
        arr['rsi_5m_curr'], arr['rsi_5m_prev'],
        rsi_buy=55, rsi_sell=45, cooldown_bars=15, eod_minutes_et=MES_EOD_ET,
    )
    compute_mfe_mae(v94_trades, arr['highs'], arr['lows'])

    oos, is_t = split_trades(v94_trades, SPLIT)
    sc_full = score_trades(v94_trades, MES_COMM, MES_DPP)
    sc_is = score_trades(is_t, MES_COMM, MES_DPP)
    sc_oos = score_trades(oos, MES_COMM, MES_DPP)

    print(f"\n  FULL: {sc_full['count']} trades, WR {sc_full['win_rate']}%, "
          f"PF {sc_full['pf']}, Net ${sc_full['net_dollar']:+.2f}")
    print(f"  IS:   {sc_is['count']} trades, WR {sc_is['win_rate']}%, "
          f"PF {sc_is['pf']}, Net ${sc_is['net_dollar']:+.2f}")
    print(f"  OOS:  {sc_oos['count']} trades, WR {sc_oos['win_rate']}%, "
          f"PF {sc_oos['pf']}, Net ${sc_oos['net_dollar']:+.2f}")

    # IS sanity check
    print(f"\n  IS sanity check: {sc_is['count']} trades, PF {sc_is['pf']}")
    print(f"  Expected:        359 trades, WR 54.9%, PF 1.266, +$1,188")

    monthly_breakdown(v94_trades, "MES v9.4 full 12-month")

    # ================================================================
    # Step A: MFE/MAE Autopsy
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP A: MFE/MAE AUTOPSY — MES v9.4")
    print("=" * 80)

    # Crash exclusion
    daily = get_daily_ranges(mes)
    big_move_dates = set(daily[daily['Range'] > BIG_MOVE_THRESHOLD].index)
    print(f"\n  Big-move days (range > {BIG_MOVE_THRESHOLD} pts): {len(big_move_dates)}")
    for d in sorted(big_move_dates)[:10]:
        print(f"    {d}: {daily.loc[d, 'Range']:.0f} pts")
    if len(big_move_dates) > 10:
        print(f"    ... and {len(big_move_dates) - 10} more")

    comm_pts = (MES_COMM * 2) / MES_DPP

    # OOS trade autopsy
    print(f"\n  OOS TRADE AUTOPSY: {len(oos)} trades")

    # MES has no SL, so categorize by exit type
    sm_flip_trades = [t for t in oos if t['result'] == 'SM_FLIP']
    eod_trades = [t for t in oos if t['result'] == 'EOD']

    winners = [t for t in oos if t['pts'] - comm_pts > 0]
    losers = [t for t in oos if t['pts'] - comm_pts <= 0]

    print(f"\n  Exit types: SM_FLIP={len(sm_flip_trades)}, EOD={len(eod_trades)}")
    print(f"  Winners: {len(winners)}, Losers: {len(losers)}")

    for label, subset in [("ALL", oos), ("WINNERS", winners), ("LOSERS", losers)]:
        if not subset:
            continue
        mfes = np.array([t['mfe'] for t in subset])
        maes = np.array([t['mae'] for t in subset])
        pts = np.array([t['pts'] for t in subset])
        bars = np.array([t['bars'] for t in subset])
        print(f"\n  {label}: {len(subset)} trades")
        print(f"    Avg MFE:  {np.mean(mfes):>7.1f} pts  (median {np.median(mfes):.1f})")
        print(f"    Avg MAE:  {np.mean(maes):>7.1f} pts  (median {np.median(maes):.1f})")
        print(f"    Avg PnL:  {np.mean(pts):>+7.1f} pts")
        print(f"    Avg bars: {np.mean(bars):>7.0f}")

    # Key question: were losing trades profitable first?
    profitable_first = [t for t in losers if t['mfe'] > 0]
    print(f"\n  Losing trades profitable first (MFE > 0): "
          f"{len(profitable_first)}/{len(losers)} ({len(profitable_first)/len(losers)*100:.0f}%)")
    if profitable_first:
        mfes_pf = np.array([t['mfe'] for t in profitable_first])
        print(f"    Avg MFE of 'profitable first' losers: {np.mean(mfes_pf):.1f} pts")
        print(f"    Median MFE: {np.median(mfes_pf):.1f} pts")

    # TP analysis for losers
    print(f"\n  TP analysis for losing trades:")
    for tp in [2, 3, 5, 8, 10, 15, 20]:
        saved = sum(1 for t in losers if t['mfe'] >= tp)
        print(f"    TP={tp:>2} would have saved {saved}/{len(losers)} losers "
              f"({saved/len(losers)*100:.0f}%)")

    # MFE distribution for losers
    buckets = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 999)]
    loser_mfes = np.array([t['mfe'] for t in losers])
    print(f"\n  MFE distribution (losing trades):")
    for lo, hi in buckets:
        cnt = np.sum((loser_mfes >= lo) & (loser_mfes < hi))
        pct = cnt / len(loser_mfes) * 100 if len(loser_mfes) > 0 else 0
        hi_label = f"{hi}" if hi < 999 else "+"
        print(f"    [{lo:>3d}-{hi_label:>3s}) pts: {cnt:>3d} ({pct:>5.1f}%)")

    # How bad do losses get without SL?
    loser_pts = np.array([t['pts'] for t in losers])
    print(f"\n  Loss severity (no SL):")
    print(f"    Avg loss: {np.mean(loser_pts):+.1f} pts")
    print(f"    Median loss: {np.median(loser_pts):+.1f} pts")
    print(f"    Worst loss: {np.min(loser_pts):+.1f} pts")
    print(f"    Worst 5 losses: {sorted(loser_pts)[:5]}")

    # IS vs OOS comparison
    print(f"\n  MFE/MAE comparison: IS vs OOS")
    for label, subset in [("IS ", is_t), ("OOS", oos)]:
        all_mfe = np.array([t['mfe'] for t in subset])
        all_mae = np.array([t['mae'] for t in subset])
        print(f"    {label}: avg MFE={np.mean(all_mfe):.1f}, avg MAE={np.mean(all_mae):.1f}")

    # ================================================================
    # Step B: Exit Model Comparison
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP B: EXIT MODEL COMPARISON — MES")
    print("=" * 80)

    tp_values = [2, 3, 5, 8, 10, 15, 20]
    sl_values = [0, 10, 15, 25, 35]

    print(f"\n  Testing TP exits with MES v9.4 entry params")
    print(f"  SM(20/12/400/255), RSI(10/55/45), CD=15, EOD=15:30")

    results = []

    for sl in sl_values:
        for tp in tp_values:
            trades = run_backtest_tp_exit(
                arr['opens'], arr['highs'], arr['lows'], arr['closes'],
                arr['sm'], arr['times'],
                arr['rsi_5m_curr'], arr['rsi_5m_prev'],
                rsi_buy=55, rsi_sell=45, sm_threshold=0.0,
                cooldown_bars=15, max_loss_pts=sl, tp_pts=tp,
                eod_minutes_et=MES_EOD_ET,
            )
            compute_mfe_mae(trades, arr['highs'], arr['lows'])

            oos_t, is_t_tp = split_trades(trades, SPLIT)
            sc_is_tp = score_trades(is_t_tp, MES_COMM, MES_DPP)
            sc_oos_tp = score_trades(oos_t, MES_COMM, MES_DPP)
            sc_full_tp = score_trades(trades, MES_COMM, MES_DPP)

            results.append({
                'tp': tp, 'sl': sl,
                'trades': trades, 'oos': oos_t, 'is': is_t_tp,
                'sc_is': sc_is_tp, 'sc_oos': sc_oos_tp, 'sc_full': sc_full_tp,
            })

    # Add SM flip reference
    results.append({
        'tp': 0, 'sl': 0,
        'trades': v94_trades,
        'oos': split_trades(v94_trades, SPLIT)[0],
        'is': split_trades(v94_trades, SPLIT)[1],
        'sc_is': score_trades(split_trades(v94_trades, SPLIT)[1], MES_COMM, MES_DPP),
        'sc_oos': score_trades(split_trades(v94_trades, SPLIT)[0], MES_COMM, MES_DPP),
        'sc_full': score_trades(v94_trades, MES_COMM, MES_DPP),
    })

    # Print grid
    print(f"\n  {'TP':>3} {'SL':>3} | {'IS Tr':>5} {'IS WR%':>7} {'IS PF':>7} {'IS Net$':>10} | "
          f"{'OOS Tr':>6} {'OOS WR%':>8} {'OOS PF':>7} {'OOS Net$':>10} | "
          f"{'Full PF':>7} {'Full $':>10}")
    print("  " + "-" * 115)

    # SM flip reference first
    r_ref = results[-1]
    si, so, sf = r_ref['sc_is'], r_ref['sc_oos'], r_ref['sc_full']
    print(f"  {'SMf':>3} {'  0':>3} | "
          f"{si['count']:>5} {si['win_rate']:>6.1f}% {si['pf']:>7.3f} {si['net_dollar']:>+10.2f} | "
          f"{so['count']:>6} {so['win_rate']:>7.1f}% {so['pf']:>7.3f} {so['net_dollar']:>+10.2f} | "
          f"{sf['pf']:>7.3f} {sf['net_dollar']:>+10.2f}  <-- SM flip baseline")

    for r in results[:-1]:
        si, so, sf = r['sc_is'], r['sc_oos'], r['sc_full']
        if si and so and sf:
            print(f"  {r['tp']:>3} {r['sl']:>3} | "
                  f"{si['count']:>5} {si['win_rate']:>6.1f}% {si['pf']:>7.3f} {si['net_dollar']:>+10.2f} | "
                  f"{so['count']:>6} {so['win_rate']:>7.1f}% {so['pf']:>7.3f} {so['net_dollar']:>+10.2f} | "
                  f"{sf['pf']:>7.3f} {sf['net_dollar']:>+10.2f}")

    # IS-OOS correlation
    is_pfs = []
    oos_pfs = []
    for r in results:
        if r['sc_is'] and r['sc_oos']:
            is_pfs.append(r['sc_is']['pf'])
            oos_pfs.append(r['sc_oos']['pf'])
    if len(is_pfs) > 2:
        corr = np.corrcoef(is_pfs, oos_pfs)[0, 1]
        print(f"\n  IS-OOS PF correlation: {corr:.3f}")

    # Combos profitable on both IS and OOS
    both_profitable = [r for r in results
                       if r['sc_is'] and r['sc_oos']
                       and r['sc_is']['net_dollar'] > 0 and r['sc_oos']['net_dollar'] > 0]
    if both_profitable:
        print(f"\n  Combos profitable on BOTH IS and OOS: {len(both_profitable)}")
        for r in sorted(both_profitable, key=lambda r: r['sc_full']['net_dollar'], reverse=True)[:5]:
            tp_label = f"TP={r['tp']}" if r['tp'] > 0 else "SM flip"
            print(f"    {tp_label} SL={r['sl']}: IS ${r['sc_is']['net_dollar']:+.2f}, "
                  f"OOS ${r['sc_oos']['net_dollar']:+.2f}, Full ${r['sc_full']['net_dollar']:+.2f}")
    else:
        print(f"\n  No TP/SL combo is profitable on BOTH IS and OOS for MES.")

    # Best OOS
    best_oos_tp = sorted([r for r in results if r['sc_oos']],
                          key=lambda r: r['sc_oos']['net_dollar'], reverse=True)
    print(f"\n  Top 5 by OOS Net$:")
    for r in best_oos_tp[:5]:
        tp_label = f"TP={r['tp']}" if r['tp'] > 0 else "SM flip"
        si, so = r['sc_is'], r['sc_oos']
        print(f"    {tp_label} SL={r['sl']}: OOS ${so['net_dollar']:+.2f} (PF {so['pf']}), "
              f"IS ${si['net_dollar']:+.2f} (PF {si['pf']})")

    # Best monthly breakdown for top 2 TP combos
    for r in best_oos_tp[:2]:
        tp_label = f"TP={r['tp']}" if r['tp'] > 0 else "SM flip"
        monthly_breakdown(r['trades'], f"MES {tp_label} SL={r['sl']}")

    # ================================================================
    # Step C: Param Sweep (if TP shows promise)
    # ================================================================
    # Only run if at least one TP combo is profitable on OOS
    tp_works = any(r['sc_oos'] and r['sc_oos']['net_dollar'] > 0
                   for r in results if r['tp'] > 0)

    if tp_works:
        print("\n" + "=" * 80)
        print("STEP C: PARAM SWEEP — MES with best TP exit")
        print("=" * 80)

        # Find best TP from Step B
        best_tp_result = max([r for r in results if r['tp'] > 0 and r['sc_oos'] and r['sc_oos']['net_dollar'] > 0],
                             key=lambda r: r['sc_full']['net_dollar'], default=None)

        if best_tp_result:
            best_tp = best_tp_result['tp']
            best_sl = best_tp_result['sl']
            print(f"\n  Using TP={best_tp}, SL={best_sl} from Step B")

            # Sweep SM threshold, RSI, CD
            sm_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
            rsi_periods = [8, 10, 12, 14]
            rsi_bands = [(55, 45), (60, 40), (65, 35)]
            cooldowns = [10, 15, 20, 25]

            sweep_results = []
            total = len(sm_thresholds) * len(rsi_periods) * len(rsi_bands) * len(cooldowns)
            count = 0

            for smt in sm_thresholds:
                for rsi_p in rsi_periods:
                    # Recompute RSI mapping for this period
                    arr_rsi = prepare_arrays(mes, rsi_len=rsi_p)

                    for rsi_buy, rsi_sell in rsi_bands:
                        for cd in cooldowns:
                            trades = run_backtest_tp_exit(
                                arr_rsi['opens'], arr_rsi['highs'], arr_rsi['lows'], arr_rsi['closes'],
                                arr_rsi['sm'], arr_rsi['times'],
                                arr_rsi['rsi_5m_curr'], arr_rsi['rsi_5m_prev'],
                                rsi_buy=rsi_buy, rsi_sell=rsi_sell, sm_threshold=smt,
                                cooldown_bars=cd, max_loss_pts=best_sl, tp_pts=best_tp,
                                eod_minutes_et=MES_EOD_ET,
                            )

                            oos_t, is_t_s = split_trades(trades, SPLIT)
                            sc_is_s = score_trades(is_t_s, MES_COMM, MES_DPP)
                            sc_oos_s = score_trades(oos_t, MES_COMM, MES_DPP)
                            sc_full_s = score_trades(trades, MES_COMM, MES_DPP)

                            if sc_full_s and sc_full_s['count'] >= 50:
                                sweep_results.append({
                                    'smt': smt, 'rsi_p': rsi_p,
                                    'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell, 'cd': cd,
                                    'sc_is': sc_is_s, 'sc_oos': sc_oos_s, 'sc_full': sc_full_s,
                                })
                            count += 1

            print(f"\n  Swept {count} combos, {len(sweep_results)} with >= 50 trades")

            # Filter: profitable on both IS and OOS
            both_prof = [r for r in sweep_results
                         if r['sc_is'] and r['sc_oos']
                         and r['sc_is']['net_dollar'] > 0 and r['sc_oos']['net_dollar'] > 0]

            print(f"  Profitable on BOTH IS and OOS: {len(both_prof)}")

            if both_prof:
                # Sort by full net$
                both_prof.sort(key=lambda r: r['sc_full']['net_dollar'], reverse=True)

                print(f"\n  Top 10 robust MES combos (TP={best_tp}, SL={best_sl}):")
                print(f"  {'SM_T':>5} {'RSI':>4} {'B/S':>5} {'CD':>3} | "
                      f"{'IS Tr':>5} {'IS PF':>6} {'IS $':>10} | "
                      f"{'OOS Tr':>6} {'OOS PF':>7} {'OOS $':>10} | "
                      f"{'Full PF':>7} {'Full $':>10}")
                print("  " + "-" * 95)
                for r in both_prof[:10]:
                    si, so, sf = r['sc_is'], r['sc_oos'], r['sc_full']
                    print(f"  {r['smt']:>5.2f} {r['rsi_p']:>4} {r['rsi_buy']}/{r['rsi_sell']:>3} {r['cd']:>3} | "
                          f"{si['count']:>5} {si['pf']:>6.3f} {si['net_dollar']:>+10.2f} | "
                          f"{so['count']:>6} {so['pf']:>7.3f} {so['net_dollar']:>+10.2f} | "
                          f"{sf['pf']:>7.3f} {sf['net_dollar']:>+10.2f}")

                # IS-OOS correlation for sweep
                is_pf_s = [r['sc_is']['pf'] for r in sweep_results if r['sc_is'] and r['sc_oos']]
                oos_pf_s = [r['sc_oos']['pf'] for r in sweep_results if r['sc_is'] and r['sc_oos']]
                if len(is_pf_s) > 2:
                    corr_s = np.corrcoef(is_pf_s, oos_pf_s)[0, 1]
                    print(f"\n  IS-OOS PF correlation across sweep: {corr_s:.3f}")
            else:
                print(f"\n  No combo profitable on both IS and OOS. MES TP exit may not generalize.")

    else:
        print("\n" + "=" * 80)
        print("STEP C: SKIPPED — No TP combo is profitable on OOS")
        print("MES may need a fundamentally different exit approach.")
        print("=" * 80)

    # ================================================================
    # Step D: Regime interaction
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP D: REGIME INTERACTION")
    print("Does MNQ daily range predict MES performance?")
    print("=" * 80)

    # Load MNQ data for cross-instrument regime check
    mnq_files = sorted(DATA_DIR.glob("databento_MNQ_1min_2025-*.csv"))
    if mnq_files:
        print("\n  Loading MNQ data for cross-instrument regime analysis...")
        mnq_dfs = []
        for f in mnq_files:
            df = pd.read_csv(f)
            result = pd.DataFrame()
            result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
            result['High'] = pd.to_numeric(df['high'], errors='coerce')
            result['Low'] = pd.to_numeric(df['low'], errors='coerce')
            result = result.set_index('Time')
            mnq_dfs.append(result)
        mnq_combined = pd.concat(mnq_dfs)
        mnq_combined = mnq_combined[~mnq_combined.index.duplicated(keep='last')].sort_index()

        et_mins_mnq = compute_et_minutes(mnq_combined.index.values)
        mask_rth = (et_mins_mnq >= NY_OPEN_ET) & (et_mins_mnq < NY_CLOSE_ET)
        mnq_rth = mnq_combined[mask_rth].copy()
        mnq_rth['date'] = mnq_rth.index.date
        mnq_daily = mnq_rth.groupby('date').agg(High=('High', 'max'), Low=('Low', 'min'))
        mnq_daily['Range'] = mnq_daily['High'] - mnq_daily['Low']

        # MES daily P&L
        comm_pts_mes = (MES_COMM * 2) / MES_DPP
        mes_daily_pnl = defaultdict(float)
        for t in v94_trades:
            d = pd.Timestamp(t['exit_time']).date()
            mes_daily_pnl[d] += (t['pts'] - comm_pts_mes) * MES_DPP

        # Correlate MNQ range with MES daily P&L
        common_dates = sorted(set(mnq_daily.index) & set(mes_daily_pnl.keys()))
        if len(common_dates) > 10:
            mnq_ranges = [mnq_daily.loc[d, 'Range'] for d in common_dates]
            mes_pnls = [mes_daily_pnl[d] for d in common_dates]

            corr = np.corrcoef(mnq_ranges, mes_pnls)[0, 1]
            print(f"\n  MNQ daily range vs MES daily P&L correlation: {corr:.3f}")

            # Bucket by MNQ range
            ranges = np.array(mnq_ranges)
            pnls = np.array(mes_pnls)

            buckets = [(0, 150), (150, 250), (250, 350), (350, 500), (500, 9999)]
            print(f"\n  MES P&L by MNQ daily range bucket:")
            print(f"  {'MNQ Range':>12} {'Days':>5} {'MES avg $':>10} {'MES total $':>12} {'MES WR%':>8}")
            print("  " + "-" * 55)
            for lo, hi in buckets:
                mask = (ranges >= lo) & (ranges < hi)
                if mask.sum() > 0:
                    bucket_pnls = pnls[mask]
                    avg = np.mean(bucket_pnls)
                    total = np.sum(bucket_pnls)
                    wr = (bucket_pnls > 0).mean() * 100
                    hi_label = f"{hi}" if hi < 9999 else "+"
                    print(f"  {lo:>4}-{hi_label:>4} pts {mask.sum():>5} {avg:>+10.2f} "
                          f"{total:>+12.2f} {wr:>7.1f}%")

            # Does filtering by MNQ range help MES?
            for mnq_thresh in [200, 250, 300, 400, 500]:
                calm_dates = {d for d, r in zip(common_dates, mnq_ranges) if r < mnq_thresh}
                calm_trades = [t for t in v94_trades if pd.Timestamp(t['exit_time']).date() in calm_dates]
                sc_calm = score_trades(calm_trades, MES_COMM, MES_DPP) if calm_trades else None
                if sc_calm:
                    print(f"\n  MES v9.4 on days with MNQ range < {mnq_thresh}:")
                    print(f"    {sc_calm['count']} trades, WR {sc_calm['win_rate']}%, "
                          f"PF {sc_calm['pf']}, Net ${sc_calm['net_dollar']:+.2f}")
    else:
        print("\n  No MNQ data found for cross-instrument analysis.")

    # ================================================================
    # Conclusions
    # ================================================================
    print("\n" + "=" * 80)
    print("MES DEEP DIVE CONCLUSIONS")
    print("=" * 80)

    print(f"\n  1. MES v9.4 baseline: IS PF={sc_is['pf']}, OOS PF={sc_oos['pf']}")

    if losers:
        avg_loser_mfe = np.mean([t['mfe'] for t in losers])
        pct_pf = len(profitable_first) / len(losers) * 100
        print(f"  2. {pct_pf:.0f}% of losing trades were profitable first (avg MFE {avg_loser_mfe:.1f} pts)")

    if tp_works:
        print(f"  3. TP exit DOES improve MES on OOS — see Step B results above")
    else:
        print(f"  3. TP exit does NOT improve MES on OOS — SM flip may be the right exit for MES")
        print(f"     (MES uses very slow SM EMA=255, which may better capture the full move)")

    print(f"\n  Key difference from MNQ:")
    print(f"    MNQ SM EMA=100 (fast) → misses reversals → TP=5 is better exit")
    print(f"    MES SM EMA=255 (slow) → catches the full trend → SM flip may still work")
    print(f"    MES has no SL → unlimited downside but also unlimited upside on winners")

    print("\n" + "=" * 80)
    print("DONE — Task 4 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
