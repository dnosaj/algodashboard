"""
SM Strength at Entry — 6-Month Analysis
========================================
Test the thesis: do weak SM entries (|SM| < threshold) produce worse trades?
If so, what's the optimal minimum |SM| for entry?

Runs on 6-month MNQ 1-min data with v11 params: SM(10/12/200/100), RSI(8/60/40), CD=20, SL=50.
Also runs v9.4 MES params: SM(20/12/400/255), RSI(10/55/45), CD=15, SL=0.

Outputs:
  - Trade stats bucketed by |SM| at entry
  - Sweep of minimum |SM| thresholds
  - Monthly breakdown for each threshold
  - Train vs test split analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_databento_1min, compute_smart_money, compute_rsi,
    prepare_backtest_arrays_1min, map_5min_rsi_to_1min,
    run_backtest_v10, score_trades, DATA_DIR,
)

# ── Config ─────────────────────────────────────────────────────────────

CONFIGS = {
    'MNQ_v11': {
        'instrument': 'MNQ',
        'sm_params': (10, 12, 200, 100),
        'rsi_len': 8,
        'rsi_buy': 60, 'rsi_sell': 40,
        'cooldown': 20, 'max_loss_pts': 50,
        'dollar_per_pt': 2.0, 'commission': 0.52,
    },
    'MES_v94': {
        'instrument': 'MES',
        'sm_params': (20, 12, 400, 255),
        'rsi_len': 10,
        'rsi_buy': 55, 'rsi_sell': 45,
        'cooldown': 15, 'max_loss_pts': 0,
        'dollar_per_pt': 5.0, 'commission': 0.52,
    },
}

# Thresholds to sweep
SM_THRESHOLDS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Train/test split
TRAIN_END = pd.Timestamp("2025-11-17")


def run_with_sm_filter(arr, cfg, sm_min_threshold=0.0):
    """Run backtest, then post-filter: remove trades where |SM| at entry < threshold.

    NOTE: We can't pre-filter (that would change cooldown/episode timing).
    Instead we run the full backtest and tag each trade with SM strength,
    then analyze outcomes grouped by SM strength. This tells us what would
    happen if we HAD filtered, without changing the engine's state machine.
    """
    trades = run_backtest_v10(
        opens=arr['opens'], highs=arr['highs'], lows=arr['lows'],
        closes=arr['closes'], sm=arr['sm'], rsi=arr['rsi'],
        times=arr['times'],
        rsi_buy=cfg['rsi_buy'], rsi_sell=cfg['rsi_sell'],
        sm_threshold=0, cooldown_bars=cfg['cooldown'],
        max_loss_pts=cfg['max_loss_pts'],
        use_rsi_cross=True,
        rsi_5m_curr=arr['rsi_5m_curr'],
        rsi_5m_prev=arr['rsi_5m_prev'],
    )

    # Tag each trade with SM at entry
    sm = arr['sm']
    for t in trades:
        idx = t['entry_idx']
        # SM used for entry decision is sm[idx-1] (previous bar's SM)
        t['sm_at_entry'] = sm[idx - 1] if idx > 0 else 0.0
        t['sm_abs'] = abs(t['sm_at_entry'])

    return trades


def score_subset(trades, dollar_per_pt, commission):
    """Score a subset of trades."""
    if not trades:
        return {'n': 0, 'pf': 0, 'wr': 0, 'net': 0, 'avg': 0, 'sl_count': 0}

    pts = np.array([t['pts'] for t in trades])
    comm = commission * 2  # round trip
    pnl = pts * dollar_per_pt - comm
    wins = pnl > 0
    gross_win = pnl[wins].sum() if wins.any() else 0
    gross_loss = abs(pnl[~wins].sum()) if (~wins).any() else 0.001
    sl_count = sum(1 for t in trades if t.get('result') == 'SL')

    return {
        'n': len(trades),
        'pf': gross_win / gross_loss if gross_loss > 0 else float('inf'),
        'wr': wins.sum() / len(trades) * 100,
        'net': pnl.sum(),
        'avg': pnl.mean(),
        'sl_count': sl_count,
    }


def get_month(t):
    """Extract month string from trade entry_time."""
    ts = pd.Timestamp(t['entry_time'])
    return ts.strftime('%Y-%m')


def main():
    print("=" * 100)
    print("  SM STRENGTH AT ENTRY — 6-MONTH ANALYSIS")
    print("=" * 100)

    for config_name, cfg in CONFIGS.items():
        print(f"\n{'━' * 100}")
        print(f"  {config_name}: SM({'/'.join(map(str, cfg['sm_params']))}) "
              f"RSI({cfg['rsi_len']}/{cfg['rsi_buy']}/{cfg['rsi_sell']}) "
              f"CD={cfg['cooldown']} SL={cfg['max_loss_pts']}")
        print(f"{'━' * 100}")

        # Load data
        inst = cfg['instrument']
        print(f"\n  Loading {inst} 1-min data...")
        df = load_databento_1min(inst)

        # Recompute SM with correct params
        sm_idx, sm_flow, sm_norm, sm_ema = cfg['sm_params']
        sm = compute_smart_money(
            df['Close'].values, df['Volume'].values,
            sm_idx, sm_flow, sm_norm, sm_ema,
        )
        df['SM_Net'] = sm

        # Prepare arrays
        arr = prepare_backtest_arrays_1min(df, rsi_len=cfg['rsi_len'])
        # Override SM with our computed version
        arr['sm'] = sm

        # Run backtest
        trades = run_with_sm_filter(arr, cfg)

        if not trades:
            print("  No trades generated!")
            continue

        dpt = cfg['dollar_per_pt']
        comm = cfg['commission']

        # ── A. Overall SM strength distribution at entries ─────────
        print(f"\n  ── A. SM STRENGTH DISTRIBUTION AT ENTRIES ──")
        sm_abs = np.array([t['sm_abs'] for t in trades])
        print(f"  Total trades: {len(trades)}")
        print(f"  |SM| at entry: mean={np.mean(sm_abs):.4f}  "
              f"median={np.median(sm_abs):.4f}  "
              f"min={np.min(sm_abs):.4f}  max={np.max(sm_abs):.4f}")

        # Histogram
        bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
        print(f"\n  {'|SM| Range':>15}  {'Count':>6}  {'%':>6}  {'PF':>6}  {'WR':>6}  {'Avg$':>8}  {'Net$':>9}  {'SL':>4}")
        print(f"  {'-'*75}")
        for j in range(len(bins)-1):
            lo, hi = bins[j], bins[j+1]
            bucket = [t for t in trades if lo <= t['sm_abs'] < hi]
            if not bucket:
                continue
            s = score_subset(bucket, dpt, comm)
            pct = len(bucket) / len(trades) * 100
            label = f"{lo:.2f}-{hi:.2f}"
            print(f"  {label:>15}  {s['n']:>6}  {pct:>5.1f}%  {s['pf']:>6.2f}  "
                  f"{s['wr']:>5.1f}%  {s['avg']:>+8.2f}  {s['net']:>+9.2f}  {s['sl_count']:>4}")

        # ── B. Sweep: minimum |SM| threshold ──────────────────────
        print(f"\n  ── B. MINIMUM |SM| THRESHOLD SWEEP ──")
        print(f"  {'Min|SM|':>8}  {'Trades':>7}  {'Removed':>8}  {'PF':>6}  {'WR':>6}  "
              f"{'Avg$/trade':>10}  {'Net$':>9}  {'SL':>4}  {'SL%':>5}")
        print(f"  {'-'*80}")

        baseline_stats = None
        for thresh in SM_THRESHOLDS:
            kept = [t for t in trades if t['sm_abs'] >= thresh]
            removed = len(trades) - len(kept)
            s = score_subset(kept, dpt, comm)
            sl_pct = s['sl_count'] / s['n'] * 100 if s['n'] > 0 else 0
            if thresh == 0:
                baseline_stats = s
            marker = ""
            if baseline_stats and s['n'] > 0:
                if s['pf'] > baseline_stats['pf'] * 1.05:
                    marker = " <<"
            print(f"  {thresh:>8.2f}  {s['n']:>7}  {removed:>8}  {s['pf']:>6.3f}  "
                  f"{s['wr']:>5.1f}%  {s['avg']:>+10.2f}  {s['net']:>+9.2f}  "
                  f"{s['sl_count']:>4}  {sl_pct:>4.1f}%{marker}")

        # ── C. Removed trades analysis ────────────────────────────
        print(f"\n  ── C. WHAT DO REMOVED TRADES LOOK LIKE? ──")
        for thresh in [0.05, 0.10, 0.15]:
            removed = [t for t in trades if t['sm_abs'] < thresh]
            if not removed:
                print(f"  |SM| < {thresh}: 0 trades removed")
                continue
            s = score_subset(removed, dpt, comm)
            sl_in_removed = sum(1 for t in removed if t.get('result') == 'SL')
            print(f"\n  Trades with |SM| < {thresh} ({len(removed)} trades):")
            print(f"    PF: {s['pf']:.3f}  WR: {s['wr']:.1f}%  Avg: {s['avg']:+.2f}  "
                  f"Net: {s['net']:+.2f}  SL exits: {sl_in_removed}")

            # Side breakdown
            longs = [t for t in removed if t['side'] == 'long']
            shorts = [t for t in removed if t['side'] == 'short']
            if longs:
                ls = score_subset(longs, dpt, comm)
                print(f"    Longs:  {ls['n']} trades, PF {ls['pf']:.3f}, Net {ls['net']:+.2f}")
            if shorts:
                ss = score_subset(shorts, dpt, comm)
                print(f"    Shorts: {ss['n']} trades, PF {ss['pf']:.3f}, Net {ss['net']:+.2f}")

            # Exit type breakdown
            exits = {}
            for t in removed:
                exits[t.get('result', 'unknown')] = exits.get(t.get('result', 'unknown'), 0) + 1
            print(f"    Exit types: {exits}")

        # ── D. Weak SM entries: are they FASTER to flip? ──────────
        print(f"\n  ── D. TRADE DURATION BY SM STRENGTH ──")
        print(f"  {'|SM| Range':>15}  {'Count':>6}  {'AvgBars':>8}  {'MedianBars':>10}  {'AvgPts':>8}")
        print(f"  {'-'*55}")
        for j in range(len(bins)-1):
            lo, hi = bins[j], bins[j+1]
            bucket = [t for t in trades if lo <= t['sm_abs'] < hi]
            if not bucket:
                continue
            bars = np.array([t['bars'] for t in bucket])
            pts = np.array([t['pts'] for t in bucket])
            label = f"{lo:.2f}-{hi:.2f}"
            print(f"  {label:>15}  {len(bucket):>6}  {np.mean(bars):>8.1f}  "
                  f"{np.median(bars):>10.0f}  {np.mean(pts):>+8.2f}")

        # ── E. Monthly breakdown at key thresholds ────────────────
        print(f"\n  ── E. MONTHLY BREAKDOWN AT KEY THRESHOLDS ──")
        months = sorted(set(get_month(t) for t in trades))

        for thresh in [0.0, 0.10, 0.15]:
            kept = [t for t in trades if t['sm_abs'] >= thresh]
            print(f"\n  Min |SM| >= {thresh}:")
            print(f"  {'Month':>8}  {'Trades':>7}  {'PF':>6}  {'WR':>6}  {'Net$':>9}")
            print(f"  {'-'*45}")
            for m in months:
                month_trades = [t for t in kept if get_month(t) == m]
                if not month_trades:
                    continue
                s = score_subset(month_trades, dpt, comm)
                print(f"  {m:>8}  {s['n']:>7}  {s['pf']:>6.3f}  {s['wr']:>5.1f}%  {s['net']:>+9.2f}")

        # ── F. Train vs Test validation ───────────────────────────
        print(f"\n  ── F. TRAIN vs TEST SPLIT ──")
        train_trades = [t for t in trades if pd.Timestamp(t['entry_time']) < TRAIN_END]
        test_trades = [t for t in trades if pd.Timestamp(t['entry_time']) >= TRAIN_END]

        print(f"  Train: {len(train_trades)} trades  |  Test: {len(test_trades)} trades\n")
        print(f"  {'Min|SM|':>8}  {'Train PF':>9}  {'Train Net':>10}  {'Test PF':>8}  "
              f"{'Test Net':>9}  {'Degr%':>6}")
        print(f"  {'-'*60}")

        for thresh in SM_THRESHOLDS:
            train_k = [t for t in train_trades if t['sm_abs'] >= thresh]
            test_k = [t for t in test_trades if t['sm_abs'] >= thresh]
            ts = score_subset(train_k, dpt, comm)
            tt = score_subset(test_k, dpt, comm)
            degr = ((tt['pf'] - ts['pf']) / ts['pf'] * 100) if ts['pf'] > 0 else 0
            print(f"  {thresh:>8.2f}  {ts['pf']:>9.3f}  {ts['net']:>+10.2f}  "
                  f"{tt['pf']:>8.3f}  {tt['net']:>+9.2f}  {degr:>+5.1f}%")

        # ── G. Optimal threshold (maximize avg $/trade) ───────────
        print(f"\n  ── G. OPTIMAL THRESHOLD (maximize $/trade) ──")
        best_avg = -9999
        best_thresh = 0
        for thresh in np.arange(0, 0.31, 0.01):
            kept = [t for t in trades if t['sm_abs'] >= thresh]
            if len(kept) < 20:
                break
            s = score_subset(kept, dpt, comm)
            if s['avg'] > best_avg:
                best_avg = s['avg']
                best_thresh = thresh

        # Show the range around optimal
        print(f"  Best avg $/trade at |SM| >= {best_thresh:.2f}: ${best_avg:.2f}/trade\n")
        print(f"  {'Min|SM|':>8}  {'Trades':>7}  {'PF':>6}  {'Avg$/trade':>10}  {'Net$':>9}")
        print(f"  {'-'*50}")
        for thresh in np.arange(max(0, best_thresh-0.05), min(0.31, best_thresh+0.06), 0.01):
            kept = [t for t in trades if t['sm_abs'] >= thresh]
            if len(kept) < 10:
                continue
            s = score_subset(kept, dpt, comm)
            marker = " <-- BEST" if abs(thresh - best_thresh) < 0.005 else ""
            print(f"  {thresh:>8.2f}  {s['n']:>7}  {s['pf']:>6.3f}  "
                  f"{s['avg']:>+10.2f}  {s['net']:>+9.2f}{marker}")

    print(f"\n{'=' * 100}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
