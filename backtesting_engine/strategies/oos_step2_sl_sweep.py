"""
OOS Step 2: Stop-Loss Sweep
============================

Tests SL = 0, 15, 20, 25, 30, 35, 40, 50, 75, 100 for MNQ v11.1.

Data variants:
  - Full IS
  - Full OOS
  - OOS minus big-move days (daily range > 500 pts, from Step 1)

Output: Grid — SL value x data variant -> Trades / WR% / PF / Net$ / SL_count

Usage:
    python3 oos_step2_sl_sweep.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    compute_et_minutes, NY_OPEN_ET, NY_CLOSE_ET,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    'MNQ_OOS': 'databento_MNQ_1min_2025-02-17_to_2025-08-17.csv',
    'MNQ_IS':  'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
}

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MNQ_COMM, MNQ_DPP = 0.52, 2.0

SL_VALUES = [0, 15, 20, 25, 30, 35, 40, 50, 75, 100]
BIG_MOVE_THRESHOLD = 500


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
# SL sweep
# ---------------------------------------------------------------------------

def run_v11_with_sl(arr, sl):
    """Run MNQ v11.1 with a specific SL value."""
    return run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], arr['rsi'], arr['times'],
        rsi_buy=60, rsi_sell=40,
        sm_threshold=0.15, cooldown_bars=20,
        max_loss_pts=sl,
        rsi_5m_curr=arr['rsi_5m_curr'],
        rsi_5m_prev=arr['rsi_5m_prev'],
    )


def main():
    print("=" * 80)
    print("OOS STEP 2: STOP-LOSS SWEEP")
    print("MNQ v11.1 — SM(10/12/200/100) RSI(8/60/40) CD=20 threshold=0.15")
    print("=" * 80)

    # ---- Load data ----
    print("\nLoading data...")
    mnq_oos = load_databento(DATA_DIR / FILES['MNQ_OOS'], MNQ_SM)
    mnq_is = load_databento(DATA_DIR / FILES['MNQ_IS'], MNQ_SM)

    print("Preparing arrays...")
    mnq_oos_arr = prepare_arrays(mnq_oos, rsi_len=8)
    mnq_is_arr = prepare_arrays(mnq_is, rsi_len=8)

    # Identify big-move days
    daily_oos = get_daily_ranges(mnq_oos)
    big_move_dates = set(daily_oos[daily_oos['Range'] > BIG_MOVE_THRESHOLD].index)
    print(f"  Big-move days in OOS (range > {BIG_MOVE_THRESHOLD}): {len(big_move_dates)}")

    # IS sanity check with default SL=50
    trades_check = run_v11_with_sl(mnq_is_arr, 50)
    sc_check = score_trades(trades_check, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
    print(f"\n  IS sanity check (SL=50): {sc_check['count']} trades, PF {sc_check['pf']}, "
          f"WR {sc_check['win_rate']}%, Net ${sc_check['net_dollar']:+.2f}")

    # ================================================================
    # SL SWEEP GRID
    # ================================================================
    print("\n" + "=" * 80)
    print("SL SWEEP RESULTS")
    print("=" * 80)

    # Header
    header = (f"  {'SL':>4} | {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10} {'SL#':>4} {'SL%':>5} "
              f"| {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10} {'SL#':>4} {'SL%':>5} "
              f"| {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10} {'SL#':>4} {'SL%':>5}")
    print(f"\n       | {'Full IS':^46} | {'Full OOS':^46} | {'OOS minus crash':^46}")
    print(header)
    print("  " + "-" * 150)

    # Store results for analysis
    results = {}

    for sl in SL_VALUES:
        row_parts = [f"  {sl:>4} |"]

        for variant_name, arr, exclude_dates in [
            ("IS", mnq_is_arr, set()),
            ("OOS", mnq_oos_arr, set()),
            ("OOS_nc", mnq_oos_arr, big_move_dates),
        ]:
            trades = run_v11_with_sl(arr, sl)
            if exclude_dates:
                trades = filter_trades_by_date(trades, exclude_dates)
            sc = score_trades(trades, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
            sl_count = sum(1 for t in trades if t['result'] == 'SL')
            sl_pct = sl_count / len(trades) * 100 if trades else 0

            results[(sl, variant_name)] = {
                'trades': len(trades), 'sc': sc,
                'sl_count': sl_count, 'sl_pct': sl_pct,
            }

            if sc:
                row_parts.append(
                    f" {sc['count']:>6d} {sc['win_rate']:>5.1f}% {sc['pf']:>6.3f} "
                    f"{sc['net_dollar']:>+10.2f} {sl_count:>4d} {sl_pct:>4.1f}% |"
                )
            else:
                row_parts.append(f" {'NO TRADES':>46} |")

        print("".join(row_parts))

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Best SL by PF for each variant
    for variant_name, variant_label in [("IS", "Full IS"), ("OOS", "Full OOS"), ("OOS_nc", "OOS minus crash")]:
        print(f"\n  {variant_label}:")
        best_pf_sl = None
        best_pf = 0
        best_net_sl = None
        best_net = -999999
        for sl in SL_VALUES:
            r = results.get((sl, variant_name))
            if r and r['sc']:
                if r['sc']['pf'] > best_pf:
                    best_pf = r['sc']['pf']
                    best_pf_sl = sl
                if r['sc']['net_dollar'] > best_net:
                    best_net = r['sc']['net_dollar']
                    best_net_sl = sl
        print(f"    Best PF:   SL={best_pf_sl} (PF {best_pf:.3f})")
        print(f"    Best Net$: SL={best_net_sl} (${best_net:+.2f})")

    # SL=0 comparison (pure SM flip, no stop)
    print(f"\n  SL=0 (pure SM flip, no stop):")
    for variant_name, variant_label in [("IS", "IS"), ("OOS", "OOS"), ("OOS_nc", "OOS-crash")]:
        r = results.get((0, variant_name))
        if r and r['sc']:
            print(f"    {variant_label:>10}: {r['sc']['count']} trades, PF {r['sc']['pf']:.3f}, "
                  f"Net ${r['sc']['net_dollar']:+.2f}")

    # Tighter SL ranking for OOS
    print(f"\n  OOS ranking by Net $ (tighter SL hypothesis):")
    oos_ranked = []
    for sl in SL_VALUES:
        r = results.get((sl, "OOS"))
        if r and r['sc']:
            oos_ranked.append((sl, r['sc']['net_dollar'], r['sc']['pf'], r['sc']['win_rate']))
    oos_ranked.sort(key=lambda x: x[1], reverse=True)
    for rank, (sl, net, pf, wr) in enumerate(oos_ranked, 1):
        print(f"    #{rank}: SL={sl:>3d}  Net ${net:>+10.2f}  PF {pf:.3f}  WR {wr:.1f}%")

    # IS-OOS correlation
    print(f"\n  IS vs OOS PF correlation:")
    is_pfs = []
    oos_pfs = []
    for sl in SL_VALUES:
        r_is = results.get((sl, "IS"))
        r_oos = results.get((sl, "OOS"))
        if r_is and r_is['sc'] and r_oos and r_oos['sc']:
            is_pfs.append(r_is['sc']['pf'])
            oos_pfs.append(r_oos['sc']['pf'])
            print(f"    SL={sl:>3d}: IS PF={r_is['sc']['pf']:.3f}, OOS PF={r_oos['sc']['pf']:.3f}")
    if len(is_pfs) > 2:
        corr = np.corrcoef(is_pfs, oos_pfs)[0, 1]
        print(f"    Correlation: {corr:.3f}")

    # Top 3 SL values for Step 4
    print(f"\n  TOP 3 SL VALUES for Step 3/4:")
    top3 = oos_ranked[:3]
    for sl, net, pf, wr in top3:
        print(f"    SL={sl}")

    print("\n" + "=" * 80)
    print("DONE — Step 2 complete. Best SL values feed into Step 3 (exit models).")
    print("=" * 80)


if __name__ == "__main__":
    main()
