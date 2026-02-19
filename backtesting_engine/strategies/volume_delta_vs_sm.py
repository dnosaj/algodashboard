"""
Volume Delta (Real Order Flow) vs SM (Derived Indicator)
========================================================
We have 42.6M NQ ticks with CME aggressor side ('B'=buy, 'A'=sell).
This is GROUND TRUTH order flow — not inferred, not modeled.

Question: At the 368 trade entry points, does the real volume delta
tell a different story than our computed SM? Specifically:
  - When SM is weak (|SM| < 0.15), what is real delta saying?
  - Are there divergences (SM bullish but delta bearish)?
  - Does delta strength predict trade outcome better than SM strength?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_databento_1min, compute_smart_money, compute_rsi,
    prepare_backtest_arrays_1min, run_backtest_v10, DATA_DIR,
)

TICK_FILE = DATA_DIR / "databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet"

# ── Step 1: Aggregate ticks to 1-min volume delta bars ─────────────────

def build_1min_delta(tick_file):
    """Aggregate NQ ticks to 1-min bars with buy/sell/delta volume."""
    print("  Loading tick data...")
    df = pd.read_parquet(tick_file)
    print(f"  {len(df):,} ticks loaded")

    # Check side field coverage
    sides = df['side'].value_counts()
    print(f"  Side distribution:")
    for s, c in sides.items():
        print(f"    {s}: {c:,} ({c/len(df)*100:.1f}%)")

    # Convert timestamp and floor to minute
    df['ts'] = pd.to_datetime(df['ts_event'])
    df['minute'] = df['ts'].dt.floor('1min')

    # Aggregate by minute
    buy_mask = df['side'] == 'B'
    sell_mask = df['side'] == 'A'

    buys = df[buy_mask].groupby('minute')['size'].sum().rename('buy_vol')
    sells = df[sell_mask].groupby('minute')['size'].sum().rename('sell_vol')
    total = df.groupby('minute')['size'].sum().rename('total_vol')
    count = df.groupby('minute')['size'].count().rename('tick_count')

    result = pd.DataFrame({
        'buy_vol': buys,
        'sell_vol': sells,
        'total_vol': total,
        'tick_count': count,
    }).fillna(0)

    result['delta'] = result['buy_vol'] - result['sell_vol']
    result['delta_pct'] = result['delta'] / result['total_vol']  # normalized

    print(f"  {len(result):,} 1-min bars with delta")
    print(f"  Date range: {result.index[0]} to {result.index[-1]}")

    return result


def compute_cvd_with_session_reset(delta_df):
    """Compute CVD (cumulative volume delta) with daily session resets.

    Resets at 18:00 ET (start of new futures session).
    """
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    delta = delta_df['delta'].values
    times = delta_df.index

    # Convert to ET for session detection
    times_et = times.tz_localize('UTC').tz_convert(ET) if times.tz is None else times.tz_convert(ET)

    cvd = np.zeros(len(delta))
    prev_date = None

    for i in range(len(delta)):
        t = times_et[i]
        # Session date: before 18:00 = same date, after 18:00 = next date
        session_date = t.date() if t.hour < 18 else (t + pd.Timedelta(days=1)).date()

        if session_date != prev_date:
            cvd[i] = delta[i]  # Reset
            prev_date = session_date
        else:
            cvd[i] = cvd[i-1] + delta[i]

    delta_df['CVD'] = cvd

    # Normalize CVD to [-1, 1] using rolling max (like SM normalization)
    window = 200  # ~3.3 hours at 1-min
    cvd_series = pd.Series(cvd, index=delta_df.index)
    rolling_max = cvd_series.abs().rolling(window, min_periods=20).max()
    delta_df['CVD_norm'] = cvd_series / rolling_max.replace(0, 1)

    return delta_df


# ── Step 2: Run backtest and tag trades with delta data ────────────────

def run_and_tag(delta_df):
    """Run v11 MNQ backtest and tag each trade with volume delta at entry."""

    print("\n  Loading MNQ 1-min data and running backtest...")
    df = load_databento_1min('MNQ')

    # Recompute SM with v11 params
    sm = compute_smart_money(df['Close'].values, df['Volume'].values, 10, 12, 200, 100)
    df['SM_Net'] = sm

    arr = prepare_backtest_arrays_1min(df, rsi_len=8)
    arr['sm'] = sm

    trades = run_backtest_v10(
        opens=arr['opens'], highs=arr['highs'], lows=arr['lows'],
        closes=arr['closes'], sm=arr['sm'], rsi=arr['rsi'],
        times=arr['times'],
        rsi_buy=60, rsi_sell=40, sm_threshold=0, cooldown_bars=20,
        max_loss_pts=50, use_rsi_cross=True,
        rsi_5m_curr=arr['rsi_5m_curr'], rsi_5m_prev=arr['rsi_5m_prev'],
    )

    print(f"  {len(trades)} trades generated")

    # Tag each trade with delta data at entry
    # MNQ timestamps are UTC, delta timestamps are UTC
    # Entry uses bar i-1 signals, so we look at delta at the bar BEFORE entry
    mnq_times = arr['times']

    for t in trades:
        idx = t['entry_idx']
        t['sm_at_entry'] = arr['sm'][idx - 1] if idx > 0 else 0.0
        t['sm_abs'] = abs(t['sm_at_entry'])

        # Find matching minute in delta data
        entry_time = pd.Timestamp(mnq_times[idx - 1])
        # NQ and MNQ have same timestamps — look up in delta_df
        # Use the bar time (floor to minute)
        entry_min = entry_time.floor('1min')

        # Get delta context: the bar itself + rolling window
        if entry_min in delta_df.index:
            t['delta'] = delta_df.loc[entry_min, 'delta']
            t['delta_pct'] = delta_df.loc[entry_min, 'delta_pct']
            t['cvd'] = delta_df.loc[entry_min, 'CVD']
            t['cvd_norm'] = delta_df.loc[entry_min, 'CVD_norm']
            t['buy_vol'] = delta_df.loc[entry_min, 'buy_vol']
            t['sell_vol'] = delta_df.loc[entry_min, 'sell_vol']
            t['total_vol'] = delta_df.loc[entry_min, 'total_vol']

            # Also get rolling 5-bar delta (5-min momentum)
            loc = delta_df.index.get_loc(entry_min)
            if loc >= 5:
                t['delta_5m'] = delta_df['delta'].iloc[loc-4:loc+1].sum()
                t['delta_5m_pct'] = t['delta_5m'] / delta_df['total_vol'].iloc[loc-4:loc+1].sum()
            else:
                t['delta_5m'] = 0
                t['delta_5m_pct'] = 0

            # Rolling 20-bar delta (20-min momentum — matches cooldown)
            if loc >= 20:
                t['delta_20m'] = delta_df['delta'].iloc[loc-19:loc+1].sum()
            else:
                t['delta_20m'] = 0
        else:
            t['delta'] = 0
            t['delta_pct'] = 0
            t['cvd'] = 0
            t['cvd_norm'] = 0
            t['buy_vol'] = 0
            t['sell_vol'] = 0
            t['total_vol'] = 0
            t['delta_5m'] = 0
            t['delta_5m_pct'] = 0
            t['delta_20m'] = 0

    # Check match rate
    matched = sum(1 for t in trades if t['total_vol'] > 0)
    print(f"  Delta data matched: {matched}/{len(trades)} trades ({matched/len(trades)*100:.0f}%)")

    return trades, delta_df


# ── Step 3: Analysis ───────────────────────────────────────────────────

def analyze(trades):
    dpt = 2.0
    comm = 0.52

    print("\n" + "=" * 100)
    print("  VOLUME DELTA vs SM — ANALYSIS")
    print("=" * 100)

    # Only use trades with delta data
    trades = [t for t in trades if t['total_vol'] > 0]
    print(f"\n  Trades with delta data: {len(trades)}")

    # Compute PnL
    for t in trades:
        t['pnl'] = t['pts'] * dpt - comm * 2

    # ── A. Delta direction agreement with SM ──────────────────────
    print("\n" + "─" * 100)
    print("  A. DOES REAL DELTA AGREE WITH SM?")
    print("─" * 100)

    for window_label, delta_key in [("1-bar delta", "delta"),
                                     ("5-min delta", "delta_5m"),
                                     ("20-min delta", "delta_20m")]:
        agree = 0
        disagree = 0
        agree_trades = []
        disagree_trades = []

        for t in trades:
            sm_sign = np.sign(t['sm_at_entry'])
            delta_sign = np.sign(t[delta_key])

            if sm_sign == delta_sign:
                agree += 1
                agree_trades.append(t)
            elif delta_sign != 0 and sm_sign != 0:
                disagree += 1
                disagree_trades.append(t)

        total = agree + disagree
        if total == 0:
            continue

        agree_pnl = np.mean([t['pnl'] for t in agree_trades]) if agree_trades else 0
        disagree_pnl = np.mean([t['pnl'] for t in disagree_trades]) if disagree_trades else 0
        agree_wr = np.mean([1 if t['pnl'] > 0 else 0 for t in agree_trades]) * 100 if agree_trades else 0
        disagree_wr = np.mean([1 if t['pnl'] > 0 else 0 for t in disagree_trades]) * 100 if disagree_trades else 0

        print(f"\n  {window_label}:")
        print(f"    SM & delta AGREE:    {agree}/{total} ({agree/total*100:.0f}%)  "
              f"avg PnL={agree_pnl:+.2f}  WR={agree_wr:.1f}%")
        print(f"    SM & delta DISAGREE: {disagree}/{total} ({disagree/total*100:.0f}%)  "
              f"avg PnL={disagree_pnl:+.2f}  WR={disagree_wr:.1f}%")

    # ── B. Delta at weak SM entries vs strong SM entries ──────────
    print("\n" + "─" * 100)
    print("  B. WHAT IS DELTA DOING AT WEAK SM ENTRIES?")
    print("─" * 100)

    bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.30), (0.30, 1.0)]

    print(f"\n  {'|SM| Range':>12}  {'N':>4}  {'Avg 5m Delta':>13}  {'Delta Agrees':>13}  "
          f"{'Avg PnL':>9}  {'Agree PnL':>10}  {'Disagree PnL':>13}")
    print(f"  {'-'*90}")

    for lo, hi in bins:
        bucket = [t for t in trades if lo <= t['sm_abs'] < hi]
        if not bucket:
            continue

        avg_delta = np.mean([t['delta_5m'] for t in bucket])
        n_agree = sum(1 for t in bucket if np.sign(t['sm_at_entry']) == np.sign(t['delta_5m']))
        pct_agree = n_agree / len(bucket) * 100

        agree_sub = [t for t in bucket if np.sign(t['sm_at_entry']) == np.sign(t['delta_5m'])]
        disagree_sub = [t for t in bucket if np.sign(t['sm_at_entry']) != np.sign(t['delta_5m']) and t['delta_5m'] != 0]

        avg_pnl = np.mean([t['pnl'] for t in bucket])
        agree_pnl = np.mean([t['pnl'] for t in agree_sub]) if agree_sub else 0
        disagree_pnl = np.mean([t['pnl'] for t in disagree_sub]) if disagree_sub else 0

        label = f"{lo:.2f}-{hi:.2f}"
        print(f"  {label:>12}  {len(bucket):>4}  {avg_delta:>+13.0f}  "
              f"{pct_agree:>12.0f}%  {avg_pnl:>+9.2f}  {agree_pnl:>+10.2f}  {disagree_pnl:>+13.2f}")

    # ── C. CVD direction at entry — does it predict outcome? ─────
    print("\n" + "─" * 100)
    print("  C. CVD DIRECTION AT ENTRY — PREDICTIVE VALUE")
    print("─" * 100)

    # For longs: is CVD positive good? For shorts: is CVD negative good?
    for side in ['long', 'short']:
        side_trades = [t for t in trades if t['side'] == side]
        if not side_trades:
            continue

        # CVD aligned = (long & CVD>0) or (short & CVD<0)
        aligned = [t for t in side_trades
                   if (side == 'long' and t['cvd_norm'] > 0) or
                      (side == 'short' and t['cvd_norm'] < 0)]
        against = [t for t in side_trades
                   if (side == 'long' and t['cvd_norm'] < 0) or
                      (side == 'short' and t['cvd_norm'] > 0)]

        if aligned:
            a_pnl = np.mean([t['pnl'] for t in aligned])
            a_wr = np.mean([1 if t['pnl'] > 0 else 0 for t in aligned]) * 100
        else:
            a_pnl = a_wr = 0

        if against:
            g_pnl = np.mean([t['pnl'] for t in against])
            g_wr = np.mean([1 if t['pnl'] > 0 else 0 for t in against]) * 100
        else:
            g_pnl = g_wr = 0

        print(f"\n  {side.upper()} trades ({len(side_trades)}):")
        print(f"    CVD aligned:  {len(aligned):>4} trades  avg={a_pnl:+.2f}  WR={a_wr:.1f}%")
        print(f"    CVD against:  {len(against):>4} trades  avg={g_pnl:+.2f}  WR={g_wr:.1f}%")

    # ── D. Divergence episodes: SM bullish but delta bearish ─────
    print("\n" + "─" * 100)
    print("  D. DIVERGENCE: SM SAYS ONE THING, DELTA SAYS ANOTHER")
    print("─" * 100)

    # SM bullish but 5-min delta strongly negative (or vice versa)
    divergent = [t for t in trades
                 if np.sign(t['sm_at_entry']) != np.sign(t['delta_5m'])
                 and abs(t['delta_5m']) > 100  # meaningful delta
                 and t['delta_5m'] != 0]

    if divergent:
        print(f"\n  {len(divergent)} trades with SM/delta divergence (|5m delta| > 100):")
        print(f"  {'Date':>18}  {'Side':>6}  {'SM':>8}  {'5m Delta':>9}  {'PnL':>8}  {'Result':>8}")
        print(f"  {'-'*65}")

        # Sort by entry time
        divergent.sort(key=lambda t: t['entry_time'])
        div_pnl = []
        for t in divergent[:30]:  # Cap at 30 to avoid flooding
            ts = pd.Timestamp(t['entry_time']).strftime('%m/%d %H:%M')
            div_pnl.append(t['pnl'])
            print(f"  {ts:>18}  {t['side']:>6}  {t['sm_at_entry']:>+8.4f}  "
                  f"{t['delta_5m']:>+9.0f}  {t['pnl']:>+8.2f}  {t.get('result',''):>8}")

        print(f"\n  Divergent trades: {len(divergent)}")
        print(f"  Avg PnL: {np.mean(div_pnl):+.2f}")
        print(f"  WR: {np.mean([1 if p > 0 else 0 for p in div_pnl])*100:.1f}%")
        print(f"  Total PnL: {np.sum(div_pnl):+.2f}")
    else:
        print("  No significant divergences found")

    # Compare: concordant trades (SM and delta agree, |delta| > 100)
    concordant = [t for t in trades
                  if np.sign(t['sm_at_entry']) == np.sign(t['delta_5m'])
                  and abs(t['delta_5m']) > 100]
    if concordant:
        conc_pnl = [t['pnl'] for t in concordant]
        print(f"\n  Concordant trades (SM & delta agree, |5m delta| > 100): {len(concordant)}")
        print(f"  Avg PnL: {np.mean(conc_pnl):+.2f}")
        print(f"  WR: {np.mean([1 if p > 0 else 0 for p in conc_pnl])*100:.1f}%")
        print(f"  Total PnL: {np.sum(conc_pnl):+.2f}")

    # ── E. Delta as standalone signal quality metric ──────────────
    print("\n" + "─" * 100)
    print("  E. |5-MIN DELTA| STRENGTH AS TRADE QUALITY PREDICTOR")
    print("─" * 100)

    delta_abs = [abs(t['delta_5m']) for t in trades]
    p25 = np.percentile(delta_abs, 25)
    p50 = np.percentile(delta_abs, 50)
    p75 = np.percentile(delta_abs, 75)

    delta_bins = [(0, p25, "Q1 (weakest)"), (p25, p50, "Q2"), (p50, p75, "Q3"), (p75, float('inf'), "Q4 (strongest)")]

    print(f"  |5m Delta| quartiles: Q1<{p25:.0f}, Q2<{p50:.0f}, Q3<{p75:.0f}, Q4>={p75:.0f}\n")
    print(f"  {'Quartile':>15}  {'N':>4}  {'Avg PnL':>9}  {'WR':>6}  {'SL':>4}  {'Net$':>9}")
    print(f"  {'-'*55}")

    for lo, hi, label in delta_bins:
        bucket = [t for t in trades if lo <= abs(t['delta_5m']) < hi]
        if not bucket:
            continue
        avg_pnl = np.mean([t['pnl'] for t in bucket])
        wr = np.mean([1 if t['pnl'] > 0 else 0 for t in bucket]) * 100
        sl = sum(1 for t in bucket if t.get('result') == 'SL')
        net = sum(t['pnl'] for t in bucket)
        print(f"  {label:>15}  {len(bucket):>4}  {avg_pnl:>+9.2f}  {wr:>5.1f}%  {sl:>4}  {net:>+9.2f}")

    # ── F. Combined filter: SM strength + Delta agreement ─────────
    print("\n" + "─" * 100)
    print("  F. COMBINED: SM STRENGTH x DELTA AGREEMENT")
    print("─" * 100)

    print(f"\n  {'Filter':>35}  {'N':>4}  {'PF':>6}  {'WR':>6}  {'Avg$':>8}  {'Net$':>9}  {'SL':>4}")
    print(f"  {'-'*80}")

    filters = {
        'Baseline (all trades)': trades,
        '|SM| >= 0.15': [t for t in trades if t['sm_abs'] >= 0.15],
        '5m delta agrees': [t for t in trades if np.sign(t['sm_at_entry']) == np.sign(t['delta_5m'])],
        '|SM|>=0.15 + delta agrees': [t for t in trades
                                        if t['sm_abs'] >= 0.15
                                        and np.sign(t['sm_at_entry']) == np.sign(t['delta_5m'])],
        'CVD aligned': [t for t in trades
                        if (t['side']=='long' and t['cvd_norm']>0) or
                           (t['side']=='short' and t['cvd_norm']<0)],
        '|SM|>=0.15 + CVD aligned': [t for t in trades
                                      if t['sm_abs'] >= 0.15
                                      and ((t['side']=='long' and t['cvd_norm']>0) or
                                           (t['side']=='short' and t['cvd_norm']<0))],
    }

    for label, subset in filters.items():
        if not subset:
            print(f"  {label:>35}  0 trades")
            continue
        pnls = np.array([t['pnl'] for t in subset])
        wins = pnls > 0
        gw = pnls[wins].sum() if wins.any() else 0
        gl = abs(pnls[~wins].sum()) if (~wins).any() else 0.001
        pf = gw / gl
        wr = wins.sum() / len(pnls) * 100
        sl = sum(1 for t in subset if t.get('result') == 'SL')
        print(f"  {label:>35}  {len(subset):>4}  {pf:>6.3f}  {wr:>5.1f}%  "
              f"{pnls.mean():>+8.2f}  {pnls.sum():>+9.2f}  {sl:>4}")

    # ── G. Monthly breakdown of best combined filter ──────────────
    print("\n" + "─" * 100)
    print("  G. MONTHLY: BEST COMBINED FILTER vs BASELINE")
    print("─" * 100)

    best_filter = [t for t in trades if t['sm_abs'] >= 0.15
                   and np.sign(t['sm_at_entry']) == np.sign(t['delta_5m'])]

    months = sorted(set(pd.Timestamp(t['entry_time']).strftime('%Y-%m') for t in trades))
    print(f"\n  {'Month':>8}  {'Base N':>7}  {'Base PF':>8}  {'Base Net':>9}  "
          f"{'Filt N':>7}  {'Filt PF':>8}  {'Filt Net':>9}")
    print(f"  {'-'*70}")

    for m in months:
        base_m = [t for t in trades if pd.Timestamp(t['entry_time']).strftime('%Y-%m') == m]
        filt_m = [t for t in best_filter if pd.Timestamp(t['entry_time']).strftime('%Y-%m') == m]

        def quick_stats(subset):
            if not subset:
                return 0, 0, 0
            pnls = np.array([t['pnl'] for t in subset])
            wins = pnls > 0
            gw = pnls[wins].sum() if wins.any() else 0
            gl = abs(pnls[~wins].sum()) if (~wins).any() else 0.001
            return len(subset), gw/gl, pnls.sum()

        bn, bpf, bnet = quick_stats(base_m)
        fn, fpf, fnet = quick_stats(filt_m)
        print(f"  {m:>8}  {bn:>7}  {bpf:>8.3f}  {bnet:>+9.2f}  "
              f"{fn:>7}  {fpf:>8.3f}  {fnet:>+9.2f}")

    print(f"\n{'=' * 100}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 100}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("  VOLUME DELTA (REAL ORDER FLOW) vs SM (DERIVED INDICATOR)")
    print("=" * 100)

    # Build 1-min delta bars from NQ ticks
    delta_df = build_1min_delta(TICK_FILE)

    # Add CVD
    print("\n  Computing CVD with session resets...")
    delta_df = compute_cvd_with_session_reset(delta_df)
    print(f"  CVD range: {delta_df['CVD'].min():,.0f} to {delta_df['CVD'].max():,.0f}")
    print(f"  CVD_norm range: {delta_df['CVD_norm'].min():.3f} to {delta_df['CVD_norm'].max():.3f}")

    # Run backtest and tag
    trades, delta_df = run_and_tag(delta_df)

    # Analyze
    analyze(trades)


if __name__ == "__main__":
    main()
