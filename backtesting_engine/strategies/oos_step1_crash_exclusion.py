"""
OOS Step 1: Crash Exclusion + Trade-Level MAE/MFE Autopsy
==========================================================

1a. Crash exclusion — defines big-move days by market data (daily range > 500 pts),
    NOT by strategy P&L. Shows v11.1 results on 4 data variants:
      - Full OOS (all months)
      - OOS minus big-move days (range > 500)
      - OOS minus March + April entirely
      - IS (reference)

1b. Trade-level MAE/MFE autopsy for MNQ v11.1 on OOS:
      - SL trades: were they profitable first? Avg MFE before reversal?
      - Non-SL winners vs losers excursion profiles
      - Key question: should SL be tighter or wider?

Usage:
    python3 oos_step1_crash_exclusion.py
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
    NY_OPEN_ET, NY_CLOSE_ET,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    'MNQ_OOS': 'databento_MNQ_1min_2025-02-17_to_2025-08-17.csv',
    'MNQ_IS':  'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
}

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MNQ_COMM, MNQ_DPP = 0.52, 2.0

# Crash exclusion threshold: daily RTH range in points
BIG_MOVE_THRESHOLD = 500


# ---------------------------------------------------------------------------
# Data loading (shared pattern)
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


# ---------------------------------------------------------------------------
# Daily range computation
# ---------------------------------------------------------------------------

def get_daily_ranges(df_1m):
    """Compute daily RTH range (High - Low) for each trading day."""
    et_mins = compute_et_minutes(df_1m.index.values)
    mask = (et_mins >= NY_OPEN_ET) & (et_mins < NY_CLOSE_ET)
    rth = df_1m[mask].copy()
    rth['date'] = rth.index.date
    daily = rth.groupby('date').agg(
        High=('High', 'max'), Low=('Low', 'min'),
        Open=('Open', 'first'), Close=('Close', 'last'),
    )
    daily['Range'] = daily['High'] - daily['Low']
    return daily


def get_big_move_dates(daily, threshold=BIG_MOVE_THRESHOLD):
    """Return set of dates with daily range > threshold."""
    return set(daily[daily['Range'] > threshold].index)


# ---------------------------------------------------------------------------
# Trade filtering helpers
# ---------------------------------------------------------------------------

def filter_trades_by_date(trades, exclude_dates):
    """Post-filter: remove trades whose ENTRY date is in exclude_dates."""
    kept = []
    for t in trades:
        entry_date = pd.Timestamp(t['entry_time']).date()
        if entry_date not in exclude_dates:
            kept.append(t)
    return kept


def compute_mfe_mae(trades, highs, lows):
    """Add MFE/MAE (in points) to each trade dict, using bar-by-bar data."""
    for t in trades:
        ei, xi, ep = t['entry_idx'], t['exit_idx'], t['entry']
        if xi <= ei:
            t['mfe'] = t['mae'] = 0.0
            continue
        bh, bl = highs[ei:xi + 1], lows[ei:xi + 1]
        if t['side'] == 'long':
            t['mfe'] = float(np.max(bh) - ep)
            t['mae'] = float(ep - np.min(bl))
        else:
            t['mfe'] = float(ep - np.min(bl))
            t['mae'] = float(np.max(bh) - ep)
    return trades


# ---------------------------------------------------------------------------
# Run v11.1 backtest
# ---------------------------------------------------------------------------

def run_v11(arr):
    """Run MNQ v11.1 with production params."""
    return run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], arr['rsi'], arr['times'],
        rsi_buy=60, rsi_sell=40,
        sm_threshold=0.15, cooldown_bars=20,
        max_loss_pts=50,
        rsi_5m_curr=arr['rsi_5m_curr'],
        rsi_5m_prev=arr['rsi_5m_prev'],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("OOS STEP 1: CRASH EXCLUSION + TRADE AUTOPSY")
    print("=" * 80)

    # ---- Load data ----
    print("\nLoading data...")
    mnq_oos = load_databento(DATA_DIR / FILES['MNQ_OOS'], MNQ_SM)
    mnq_is = load_databento(DATA_DIR / FILES['MNQ_IS'], MNQ_SM)
    print(f"  MNQ OOS: {len(mnq_oos):,} bars ({mnq_oos.index[0].date()} to {mnq_oos.index[-1].date()})")
    print(f"  MNQ IS:  {len(mnq_is):,} bars ({mnq_is.index[0].date()} to {mnq_is.index[-1].date()})")

    print("Preparing arrays...")
    mnq_oos_arr = prepare_arrays(mnq_oos, rsi_len=8)
    mnq_is_arr = prepare_arrays(mnq_is, rsi_len=8)

    # ================================================================
    # 1a. CRASH EXCLUSION
    # ================================================================
    print("\n" + "=" * 80)
    print("1a. CRASH EXCLUSION — Daily Range Filter")
    print("=" * 80)

    # Compute daily ranges
    daily_oos = get_daily_ranges(mnq_oos)
    daily_is = get_daily_ranges(mnq_is)
    overall_avg = pd.concat([daily_oos, daily_is])['Range'].mean()

    print(f"\n  Overall avg daily range: {overall_avg:.1f} pts")
    print(f"  Crash threshold: {BIG_MOVE_THRESHOLD} pts ({BIG_MOVE_THRESHOLD/overall_avg:.1f}x avg)")

    # Identify big-move days
    big_move_dates = get_big_move_dates(daily_oos)
    print(f"\n  Big-move days in OOS (range > {BIG_MOVE_THRESHOLD} pts): {len(big_move_dates)}")
    for d in sorted(big_move_dates):
        r = daily_oos.loc[d, 'Range']
        print(f"    {d}: {r:.0f} pts")

    # March + April dates
    mar_apr_dates = set()
    for d in daily_oos.index:
        dt = pd.Timestamp(d)
        if dt.month in (3, 4):
            mar_apr_dates.add(d)
    print(f"\n  March + April trading days: {len(mar_apr_dates)}")

    # Run v11.1 on all OOS
    trades_oos_full = run_v11(mnq_oos_arr)
    trades_is = run_v11(mnq_is_arr)

    # Filter variants
    trades_oos_no_bigmove = filter_trades_by_date(trades_oos_full, big_move_dates)
    trades_oos_no_marapr = filter_trades_by_date(trades_oos_full, mar_apr_dates)

    # Score all variants
    variants = [
        ("IS (reference)", trades_is),
        ("Full OOS", trades_oos_full),
        (f"OOS minus big-move days ({len(big_move_dates)}d)", trades_oos_no_bigmove),
        (f"OOS minus Mar+Apr ({len(mar_apr_dates)}d)", trades_oos_no_marapr),
    ]

    print(f"\n  {'Variant':<45} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10} {'SL':>4} {'SL%':>5}")
    print("  " + "-" * 90)
    for label, trades in variants:
        sc = score_trades(trades, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
        sl_count = sum(1 for t in trades if t['result'] == 'SL')
        sl_pct = sl_count / len(trades) * 100 if trades else 0
        if sc:
            print(f"  {label:<45} {sc['count']:>6d} {sc['win_rate']:>5.1f}% {sc['pf']:>6.3f} "
                  f"{sc['net_dollar']:>+10.2f} {sl_count:>4d} {sl_pct:>4.1f}%")

    # IS sanity check
    sc_is = score_trades(trades_is, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
    print(f"\n  IS sanity check: {sc_is['count']} trades, PF {sc_is['pf']}, "
          f"WR {sc_is['win_rate']}%, Net ${sc_is['net_dollar']:+.2f}")
    expected_ok = sc_is and sc_is['count'] >= 220 and sc_is['pf'] > 1.5
    print(f"  Expected: 226 trades, PF 1.797, WR 62.4%, +$3,529 -> {'PASS' if expected_ok else 'FAIL'}")

    # Monthly OOS breakdown (with variant tagging)
    print(f"\n  Monthly OOS breakdown:")
    monthly = defaultdict(list)
    for t in trades_oos_full:
        m = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        monthly[m].append(t)

    print(f"  {'Month':<10} {'Trades':>6} {'SL':>4} {'SL%':>5} {'WR%':>6} {'PF':>6} {'Net $':>10} {'BigMove':>8}")
    print("  " + "-" * 70)
    for m in sorted(monthly.keys()):
        mt = monthly[m]
        sc = score_trades(mt, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
        sl_m = sum(1 for t in mt if t['result'] == 'SL')
        # Count big-move day trades in this month
        bm_trades = sum(1 for t in mt if pd.Timestamp(t['entry_time']).date() in big_move_dates)
        if sc:
            print(f"  {m:<10} {sc['count']:>6d} {sl_m:>4d} {sl_m/len(mt)*100:>4.1f}% "
                  f"{sc['win_rate']:>5.1f}% {sc['pf']:>6.3f} {sc['net_dollar']:>+10.2f} {bm_trades:>8d}")

    # ================================================================
    # 1b. TRADE-LEVEL MAE/MFE AUTOPSY
    # ================================================================
    print("\n" + "=" * 80)
    print("1b. TRADE-LEVEL MAE/MFE AUTOPSY — MNQ v11.1 OOS")
    print("=" * 80)

    trades_oos = compute_mfe_mae(trades_oos_full, mnq_oos_arr['highs'], mnq_oos_arr['lows'])

    comm_pts = (MNQ_COMM * 2) / MNQ_DPP
    sl_trades = [t for t in trades_oos if t['result'] == 'SL']
    non_sl = [t for t in trades_oos if t['result'] != 'SL']
    non_sl_winners = [t for t in non_sl if t['pts'] - comm_pts > 0]
    non_sl_losers = [t for t in non_sl if t['pts'] - comm_pts <= 0]

    # --- SL trade autopsy ---
    print(f"\n  SL TRADES: {len(sl_trades)} total")
    if sl_trades:
        mfes = np.array([t['mfe'] for t in sl_trades])
        maes = np.array([t['mae'] for t in sl_trades])
        pts = np.array([t['pts'] for t in sl_trades])
        bars = np.array([t['bars'] for t in sl_trades])

        print(f"    Avg MFE:  {np.mean(mfes):>7.1f} pts  (median {np.median(mfes):.1f})")
        print(f"    Avg MAE:  {np.mean(maes):>7.1f} pts  (median {np.median(maes):.1f})")
        print(f"    Avg PnL:  {np.mean(pts):>+7.1f} pts")
        print(f"    Avg bars: {np.mean(bars):>7.0f}")

        # Were SL trades profitable before stopping out?
        profitable_first = np.sum(mfes > 0)
        pf_5 = np.sum(mfes >= 5)
        pf_10 = np.sum(mfes >= 10)
        pf_20 = np.sum(mfes >= 20)
        print(f"\n    SL trades that were profitable first (MFE > 0): "
              f"{profitable_first}/{len(sl_trades)} ({profitable_first/len(sl_trades)*100:.0f}%)")
        print(f"    SL trades with MFE >= 5 pts:  {pf_5} ({pf_5/len(sl_trades)*100:.0f}%)")
        print(f"    SL trades with MFE >= 10 pts: {pf_10} ({pf_10/len(sl_trades)*100:.0f}%)")
        print(f"    SL trades with MFE >= 20 pts: {pf_20} ({pf_20/len(sl_trades)*100:.0f}%)")

        # MFE distribution for SL trades
        buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
        print(f"\n    MFE distribution (SL trades):")
        for lo, hi in buckets:
            cnt = np.sum((mfes >= lo) & (mfes < hi))
            pct = cnt / len(mfes) * 100
            print(f"      [{lo:>3d}-{hi:>3d}) pts: {cnt:>3d} ({pct:>5.1f}%)")

        # MAE distribution for SL trades
        print(f"\n    MAE distribution (SL trades):")
        for lo, hi in buckets:
            cnt = np.sum((maes >= lo) & (maes < hi))
            pct = cnt / len(maes) * 100
            print(f"      [{lo:>3d}-{hi:>3d}) pts: {cnt:>3d} ({pct:>5.1f}%)")

        # Key insight: if MFE > some threshold, a TP exit would have captured profits
        print(f"\n    TP analysis for SL trades:")
        for tp in [3, 5, 8, 10, 15, 20]:
            saved = np.sum(mfes >= tp)
            print(f"      TP={tp:>2d} would have saved {saved}/{len(sl_trades)} SL trades "
                  f"({saved/len(sl_trades)*100:.0f}%)")

        # Direction breakdown of SL trades
        sl_longs = [t for t in sl_trades if t['side'] == 'long']
        sl_shorts = [t for t in sl_trades if t['side'] == 'short']
        print(f"\n    SL by direction: {len(sl_longs)} longs, {len(sl_shorts)} shorts")
        if sl_longs:
            print(f"      Longs:  avg MFE={np.mean([t['mfe'] for t in sl_longs]):.1f}, "
                  f"avg MAE={np.mean([t['mae'] for t in sl_longs]):.1f}")
        if sl_shorts:
            print(f"      Shorts: avg MFE={np.mean([t['mfe'] for t in sl_shorts]):.1f}, "
                  f"avg MAE={np.mean([t['mae'] for t in sl_shorts]):.1f}")

    # --- Non-SL trade autopsy ---
    print(f"\n  NON-SL WINNERS: {len(non_sl_winners)}")
    if non_sl_winners:
        mfes = np.array([t['mfe'] for t in non_sl_winners])
        maes = np.array([t['mae'] for t in non_sl_winners])
        pts = np.array([t['pts'] for t in non_sl_winners])
        bars = np.array([t['bars'] for t in non_sl_winners])
        print(f"    Avg MFE:  {np.mean(mfes):>7.1f} pts  (median {np.median(mfes):.1f})")
        print(f"    Avg MAE:  {np.mean(maes):>7.1f} pts  (median {np.median(maes):.1f})")
        print(f"    Avg PnL:  {np.mean(pts):>+7.1f} pts")
        print(f"    Avg bars: {np.mean(bars):>7.0f}")

    print(f"\n  NON-SL LOSERS: {len(non_sl_losers)}")
    if non_sl_losers:
        mfes = np.array([t['mfe'] for t in non_sl_losers])
        maes = np.array([t['mae'] for t in non_sl_losers])
        pts = np.array([t['pts'] for t in non_sl_losers])
        bars = np.array([t['bars'] for t in non_sl_losers])
        print(f"    Avg MFE:  {np.mean(mfes):>7.1f} pts  (median {np.median(mfes):.1f})")
        print(f"    Avg MAE:  {np.mean(maes):>7.1f} pts  (median {np.median(maes):.1f})")
        print(f"    Avg PnL:  {np.mean(pts):>+7.1f} pts")
        print(f"    Avg bars: {np.mean(bars):>7.0f}")

    # --- Overall MFE/MAE comparison: IS vs OOS ---
    print(f"\n  MFE/MAE COMPARISON: IS vs OOS")
    trades_is_mfe = compute_mfe_mae(trades_is, mnq_is_arr['highs'], mnq_is_arr['lows'])

    for label, trades in [("IS ", trades_is_mfe), ("OOS", trades_oos)]:
        all_mfe = np.array([t['mfe'] for t in trades])
        all_mae = np.array([t['mae'] for t in trades])
        sl_t = [t for t in trades if t['result'] == 'SL']
        sl_mfe = np.array([t['mfe'] for t in sl_t]) if sl_t else np.array([0])
        print(f"    {label}: all avg MFE={np.mean(all_mfe):.1f} MAE={np.mean(all_mae):.1f}, "
              f"SL avg MFE={np.mean(sl_mfe):.1f}")

    # --- Key question answer ---
    print(f"\n  {'='*70}")
    print(f"  KEY QUESTION: Tighter SL or Wider SL?")
    print(f"  {'='*70}")
    if sl_trades:
        avg_sl_mfe = np.mean([t['mfe'] for t in sl_trades])
        pct_profitable_first = np.mean([t['mfe'] > 0 for t in sl_trades]) * 100
        pct_mfe_above_5 = np.mean([t['mfe'] >= 5 for t in sl_trades]) * 100

        print(f"\n  SL trades avg MFE = {avg_sl_mfe:.1f} pts")
        print(f"  {pct_profitable_first:.0f}% of SL trades were profitable at some point")
        print(f"  {pct_mfe_above_5:.0f}% had MFE >= 5 pts")

        if avg_sl_mfe > 10:
            print(f"\n  FINDING: SL trades go significantly profitable ({avg_sl_mfe:.0f} pts avg)")
            print(f"  before reversing. This suggests the SM exit is too SLOW to protect")
            print(f"  gains. A FASTER exit (TP or faster SM) could capture these profits.")
            print(f"  Tighter SL alone won't help — the issue is exit timing, not entry quality.")
        elif avg_sl_mfe > 3:
            print(f"\n  FINDING: SL trades go modestly profitable ({avg_sl_mfe:.0f} pts avg)")
            print(f"  before reversing. Both faster exits and tighter SL could help.")
        else:
            print(f"\n  FINDING: SL trades barely go profitable (MFE {avg_sl_mfe:.1f} pts).")
            print(f"  These are BAD entries. Tighter SL would cut losses faster.")

    print("\n" + "=" * 80)
    print("DONE — Step 1 complete. Results feed into Step 2 (SL sweep).")
    print("=" * 80)


if __name__ == "__main__":
    main()
