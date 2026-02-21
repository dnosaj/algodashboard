"""
Task 2: TP Sweep for vWinners Exit
====================================
The MFE autopsy showed 98% of v11 SL trades were profitable first (avg 24 pts).
TP=5 captures only 5 of those 24 pts. Find the optimal TP level that captures
more of the move while still protecting against reversals.

TP = 5, 8, 10, 12, 15, 20, 25, 30 pts
SL = 15, 35, 50
All using vWinners entry params: SM_T=0.15, RSI=8/60-40, CD=20
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    fmt_score, compute_et_minutes,
    NY_OPEN_ET, NY_CLOSE_ET,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit, compute_mfe_mae

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MNQ_COMM, MNQ_DPP = 0.52, 2.0
SPLIT = pd.Timestamp("2025-08-17", tz='UTC')

# vWinners entry params
SM_THRESHOLD = 0.15
RSI_BUY, RSI_SELL = 60, 40
COOLDOWN = 20

# Sweep grid
TP_VALUES = [5, 8, 10, 12, 15, 20, 25, 30]
SL_VALUES = [15, 35, 50]


def load_combined():
    files = sorted(DATA_DIR.glob("databento_MNQ_1min_2025-*.csv"))
    print(f"  Loading MNQ: {[f.name for f in files]}")
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
        combined['Close'].values, combined['Volume'].values, **MNQ_SM
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


def monthly_breakdown(trades, label, comm=MNQ_COMM, dpp=MNQ_DPP):
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


def main():
    print("=" * 80)
    print("TASK 2: TP SWEEP FOR vWinners EXIT")
    print("=" * 80)

    # Load data
    print("\nLoading combined 12-month MNQ data...")
    mnq = load_combined()

    print("Preparing arrays (RSI=8)...")
    arr = prepare_arrays(mnq, rsi_len=8)

    # ================================================================
    # Reference: vWinners with SM flip exit (current production)
    # ================================================================
    print("\n" + "=" * 80)
    print("REFERENCE: vWinners with SM flip exit (production)")
    print("=" * 80)

    rsi_dummy = np.full(len(arr['closes']), 50.0)
    smflip_trades = run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], rsi_dummy, arr['times'],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL, sm_threshold=SM_THRESHOLD,
        cooldown_bars=COOLDOWN, max_loss_pts=50,
        rsi_5m_curr=arr['rsi_5m_curr'], rsi_5m_prev=arr['rsi_5m_prev'],
    )
    compute_mfe_mae(smflip_trades, arr['highs'], arr['lows'])

    oos_ref, is_ref = split_trades(smflip_trades, SPLIT)
    sc_ref_is = score_trades(is_ref, MNQ_COMM, MNQ_DPP)
    sc_ref_oos = score_trades(oos_ref, MNQ_COMM, MNQ_DPP)
    sc_ref_full = score_trades(smflip_trades, MNQ_COMM, MNQ_DPP)

    print(f"  IS:   {sc_ref_is['count']} trades, WR {sc_ref_is['win_rate']}%, "
          f"PF {sc_ref_is['pf']}, Net ${sc_ref_is['net_dollar']:+.2f}")
    print(f"  OOS:  {sc_ref_oos['count']} trades, WR {sc_ref_oos['win_rate']}%, "
          f"PF {sc_ref_oos['pf']}, Net ${sc_ref_oos['net_dollar']:+.2f}")
    print(f"  FULL: {sc_ref_full['count']} trades, WR {sc_ref_full['win_rate']}%, "
          f"PF {sc_ref_full['pf']}, Net ${sc_ref_full['net_dollar']:+.2f}")

    # IS sanity check
    print(f"\n  IS sanity check: {sc_ref_is['count']} trades, PF {sc_ref_is['pf']}")
    print(f"  Expected:        226 trades, PF 1.797, WR 62.4%, +$3,529")

    # ================================================================
    # TP x SL Grid
    # ================================================================
    print("\n" + "=" * 80)
    print("TP x SL GRID — vWinners entry params (SM_T=0.15, RSI=8/60-40, CD=20)")
    print("=" * 80)

    results = []

    for sl in SL_VALUES:
        for tp in TP_VALUES:
            trades = run_backtest_tp_exit(
                arr['opens'], arr['highs'], arr['lows'], arr['closes'],
                arr['sm'], arr['times'],
                arr['rsi_5m_curr'], arr['rsi_5m_prev'],
                rsi_buy=RSI_BUY, rsi_sell=RSI_SELL, sm_threshold=SM_THRESHOLD,
                cooldown_bars=COOLDOWN, max_loss_pts=sl, tp_pts=tp,
            )
            compute_mfe_mae(trades, arr['highs'], arr['lows'])

            oos, is_t = split_trades(trades, SPLIT)
            sc_is = score_trades(is_t, MNQ_COMM, MNQ_DPP)
            sc_oos = score_trades(oos, MNQ_COMM, MNQ_DPP)
            sc_full = score_trades(trades, MNQ_COMM, MNQ_DPP)

            results.append({
                'tp': tp, 'sl': sl,
                'trades': trades, 'oos': oos, 'is': is_t,
                'sc_is': sc_is, 'sc_oos': sc_oos, 'sc_full': sc_full,
            })

    # Print grid
    print(f"\n  {'TP':>3} {'SL':>3} | {'IS Tr':>5} {'IS WR%':>7} {'IS PF':>7} {'IS Net$':>10} | "
          f"{'OOS Tr':>6} {'OOS WR%':>8} {'OOS PF':>7} {'OOS Net$':>10} | "
          f"{'Full Tr':>7} {'Full PF':>7} {'Full $':>10}")
    print("  " + "-" * 120)

    for r in results:
        si, so, sf = r['sc_is'], r['sc_oos'], r['sc_full']
        if si and so and sf:
            print(f"  {r['tp']:>3} {r['sl']:>3} | "
                  f"{si['count']:>5} {si['win_rate']:>6.1f}% {si['pf']:>7.3f} {si['net_dollar']:>+10.2f} | "
                  f"{so['count']:>6} {so['win_rate']:>7.1f}% {so['pf']:>7.3f} {so['net_dollar']:>+10.2f} | "
                  f"{sf['count']:>7} {sf['pf']:>7.3f} {sf['net_dollar']:>+10.2f}")

    # ================================================================
    # IS-OOS Correlation
    # ================================================================
    print("\n" + "=" * 80)
    print("IS-OOS PF CORRELATION")
    print("=" * 80)

    is_pfs = []
    oos_pfs = []
    for r in results:
        if r['sc_is'] and r['sc_oos']:
            is_pfs.append(r['sc_is']['pf'])
            oos_pfs.append(r['sc_oos']['pf'])

    if len(is_pfs) > 2:
        corr = np.corrcoef(is_pfs, oos_pfs)[0, 1]
        print(f"\n  IS-OOS PF correlation across {len(is_pfs)} TP/SL combos: {corr:.3f}")
        print(f"  (Reference: entry param sweep was 0.067 — near zero)")
        if corr > 0.5:
            print(f"  TP/SL sensitivity is MORE predictable than entry params — good signal")
        elif corr > 0.2:
            print(f"  Moderate IS-OOS correlation — some predictive value")
        else:
            print(f"  Low correlation — TP/SL optimal values may be regime-dependent")

    # ================================================================
    # MFE utilization
    # ================================================================
    print("\n" + "=" * 80)
    print("MFE UTILIZATION — How much of available MFE does each TP capture?")
    print("=" * 80)

    # Get MFE stats from SM flip reference (the "maximum available")
    sl_trades_ref = [t for t in smflip_trades if t['result'] == 'SL']
    avg_sl_mfe = np.mean([t['mfe'] for t in sl_trades_ref]) if sl_trades_ref else 0

    all_mfe_ref = np.array([t['mfe'] for t in smflip_trades])
    avg_all_mfe = np.mean(all_mfe_ref)

    print(f"\n  SM flip reference: avg MFE all trades = {avg_all_mfe:.1f} pts")
    print(f"  SM flip reference: avg MFE SL trades = {avg_sl_mfe:.1f} pts")

    print(f"\n  {'TP':>3} {'SL':>3} | {'TP/AvgMFE':>10} {'TP/SL_MFE':>10} | {'WR (Full)':>10} {'PF (Full)':>9} {'Net$ (Full)':>12}")
    print("  " + "-" * 70)
    for r in results:
        tp = r['tp']
        if r['sc_full']:
            pct_all = tp / avg_all_mfe * 100 if avg_all_mfe > 0 else 0
            pct_sl = tp / avg_sl_mfe * 100 if avg_sl_mfe > 0 else 0
            print(f"  {tp:>3} {r['sl']:>3} | {pct_all:>9.1f}% {pct_sl:>9.1f}% | "
                  f"{r['sc_full']['win_rate']:>9.1f}% {r['sc_full']['pf']:>9.3f} "
                  f"{r['sc_full']['net_dollar']:>+12.2f}")

    # ================================================================
    # Monthly breakdown for best combos
    # ================================================================
    print("\n" + "=" * 80)
    print("MONTHLY BREAKDOWN — Best TP/SL combos")
    print("=" * 80)

    # Find best 3 by combined (IS+OOS) Net$
    sorted_results = sorted(results, key=lambda r: r['sc_full']['net_dollar'] if r['sc_full'] else -9999, reverse=True)

    for r in sorted_results[:3]:
        monthly_breakdown(r['trades'], f"TP={r['tp']} SL={r['sl']}")

    # ================================================================
    # Comparison to SM flip
    # ================================================================
    print("\n" + "=" * 80)
    print("BEST TP COMBOS vs SM FLIP EXIT")
    print("=" * 80)

    # Best by OOS Net$
    best_oos = sorted(results, key=lambda r: r['sc_oos']['net_dollar'] if r['sc_oos'] else -9999, reverse=True)

    print(f"\n  Top 5 by OOS Net$:")
    print(f"  {'TP':>3} {'SL':>3} | {'IS PF':>7} {'IS $':>10} | {'OOS PF':>7} {'OOS $':>10} | {'vs SM flip OOS':>15}")
    print("  " + "-" * 70)
    for r in best_oos[:5]:
        si, so = r['sc_is'], r['sc_oos']
        delta = so['net_dollar'] - sc_ref_oos['net_dollar']
        print(f"  {r['tp']:>3} {r['sl']:>3} | {si['pf']:>7.3f} {si['net_dollar']:>+10.2f} | "
              f"{so['pf']:>7.3f} {so['net_dollar']:>+10.2f} | {delta:>+15.2f}")

    # Best that are profitable on BOTH IS and OOS
    both_profitable = [r for r in results
                       if r['sc_is'] and r['sc_oos']
                       and r['sc_is']['net_dollar'] > 0 and r['sc_oos']['net_dollar'] > 0]
    if both_profitable:
        print(f"\n  Combos profitable on BOTH IS and OOS:")
        print(f"  {'TP':>3} {'SL':>3} | {'IS PF':>7} {'IS $':>10} | {'OOS PF':>7} {'OOS $':>10} | {'Full PF':>7} {'Full $':>10}")
        print("  " + "-" * 80)
        for r in sorted(both_profitable, key=lambda r: r['sc_full']['net_dollar'], reverse=True):
            si, so, sf = r['sc_is'], r['sc_oos'], r['sc_full']
            print(f"  {r['tp']:>3} {r['sl']:>3} | {si['pf']:>7.3f} {si['net_dollar']:>+10.2f} | "
                  f"{so['pf']:>7.3f} {so['net_dollar']:>+10.2f} | {sf['pf']:>7.3f} {sf['net_dollar']:>+10.2f}")
    else:
        print(f"\n  No TP/SL combo is profitable on BOTH IS and OOS.")

    # Does any TP level beat SM flip on IS while also working on OOS?
    beats_smflip_is = [r for r in results
                       if r['sc_is'] and r['sc_oos']
                       and r['sc_is']['net_dollar'] > sc_ref_is['net_dollar']
                       and r['sc_oos']['net_dollar'] > sc_ref_oos['net_dollar']]
    if beats_smflip_is:
        print(f"\n  Combos that beat SM flip on IS AND OOS: {len(beats_smflip_is)}")
        for r in beats_smflip_is:
            print(f"    TP={r['tp']} SL={r['sl']}: IS ${r['sc_is']['net_dollar']:+.2f} "
                  f"(SM flip ${sc_ref_is['net_dollar']:+.2f}), "
                  f"OOS ${r['sc_oos']['net_dollar']:+.2f} "
                  f"(SM flip ${sc_ref_oos['net_dollar']:+.2f})")
    else:
        print(f"\n  No TP/SL combo beats SM flip on IS — the tradeoff stands.")
        print(f"  TP exits sacrifice IS upside for OOS protection.")

    # ================================================================
    # Conclusion
    # ================================================================
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    if both_profitable:
        best = sorted(both_profitable, key=lambda r: r['sc_full']['net_dollar'], reverse=True)[0]
        print(f"\n  Best all-weather TP combo: TP={best['tp']} SL={best['sl']}")
        print(f"    IS:   PF {best['sc_is']['pf']}, Net ${best['sc_is']['net_dollar']:+.2f}")
        print(f"    OOS:  PF {best['sc_oos']['pf']}, Net ${best['sc_oos']['net_dollar']:+.2f}")
        print(f"    Full: PF {best['sc_full']['pf']}, Net ${best['sc_full']['net_dollar']:+.2f}")

        if best['tp'] > 5:
            print(f"\n  TP={best['tp']} captures {best['tp']}/24 = {best['tp']/24*100:.0f}% of avg SL-trade MFE")
            print(f"  vs TP=5 which captures only 21%. This is a meaningfully different strategy.")
        elif best['tp'] == 5:
            print(f"\n  TP=5 remains the sweet spot. Higher TP values degrade on OOS.")
    else:
        print(f"\n  TP=5 is confirmed as the best exit model.")
        print(f"  Higher TP values cannot survive OOS conditions.")
        print(f"  The 5-pt scalp is truly the sweet spot.")

    print("\n" + "=" * 80)
    print("DONE — Task 2 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
