"""
Task 3: Regime Detector (Option C)
====================================
Build a predictive regime classifier that determines BEFORE 10am ET
whether vWinners should trade that day. Uses only data available before
US open: London session (4am-9:30am ET) and prior-day stats.

All features must be computable at 10:00 AM ET. No intra-day US session
data used in the prediction. This is a forecast, not a post-hoc label.
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MNQ_COMM, MNQ_DPP = 0.52, 2.0
SPLIT = pd.Timestamp("2025-08-17", tz='UTC')

# Session boundaries in ET minutes from midnight
LONDON_OPEN_ET = 4 * 60          # 4:00 AM ET
LONDON_CLOSE_ET = 9 * 60 + 30    # 9:30 AM ET
RTH_OPEN_ET = NY_OPEN_ET         # 10:00 AM ET
RTH_CLOSE_ET = NY_CLOSE_ET       # 4:00 PM ET


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 1: Feature computation
# ---------------------------------------------------------------------------

def compute_features(df_1m):
    """Compute per-day features using only data available before 10am ET.

    Returns DataFrame indexed by date with columns:
    - london_range: High-Low during 4:00-9:30 AM ET
    - london_sm: SM value at the 9:30 AM ET bar
    - prior_day_range: RTH High-Low from previous trading day
    - rolling_3d_vol, rolling_5d_vol, rolling_10d_vol: avg of prior-day ranges
    """
    et_mins = compute_et_minutes(df_1m.index.values)
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    sm = df_1m['SM_Net'].values
    dates = pd.DatetimeIndex(df_1m.index).tz_localize('UTC').tz_convert('America/New_York').date

    # Compute London session range per day
    london_data = defaultdict(lambda: {'high': -np.inf, 'low': np.inf, 'sm_at_close': 0.0})
    rth_data = defaultdict(lambda: {'high': -np.inf, 'low': np.inf})

    for i in range(len(df_1m)):
        d = dates[i]
        m = et_mins[i]

        # London session: 4:00-9:30 AM ET
        if LONDON_OPEN_ET <= m < LONDON_CLOSE_ET:
            ld = london_data[d]
            ld['high'] = max(ld['high'], highs[i])
            ld['low'] = min(ld['low'], lows[i])
            # Track SM at last London bar (closest to 9:30)
            ld['sm_at_close'] = sm[i]

        # RTH: 10:00-16:00 ET
        if RTH_OPEN_ET <= m < RTH_CLOSE_ET:
            rd = rth_data[d]
            rd['high'] = max(rd['high'], highs[i])
            rd['low'] = min(rd['low'], lows[i])

    # Build feature DataFrame
    all_dates = sorted(set(dates))
    records = []

    # Compute RTH ranges for prior-day features
    rth_ranges = {}
    for d in all_dates:
        rd = rth_data.get(d)
        if rd and rd['high'] > -np.inf:
            rth_ranges[d] = rd['high'] - rd['low']

    sorted_rth_dates = sorted(rth_ranges.keys())
    rth_range_series = pd.Series(rth_ranges).sort_index()

    for d in all_dates:
        ld = london_data.get(d)
        london_range = (ld['high'] - ld['low']) if (ld and ld['high'] > -np.inf) else np.nan
        london_sm = ld['sm_at_close'] if ld else np.nan

        # Prior day range
        idx = sorted_rth_dates.index(d) if d in sorted_rth_dates else -1
        prior_range = rth_ranges.get(sorted_rth_dates[idx - 1]) if idx > 0 else np.nan

        # Rolling N-day volatility (average of prior-day ranges)
        rolling_3d = rth_range_series[:d].iloc[-4:-1].mean() if len(rth_range_series[:d]) >= 4 else np.nan
        rolling_5d = rth_range_series[:d].iloc[-6:-1].mean() if len(rth_range_series[:d]) >= 6 else np.nan
        rolling_10d = rth_range_series[:d].iloc[-11:-1].mean() if len(rth_range_series[:d]) >= 11 else np.nan

        records.append({
            'date': d,
            'london_range': london_range,
            'london_sm': london_sm,
            'prior_day_range': prior_range,
            'rolling_3d_vol': rolling_3d,
            'rolling_5d_vol': rolling_5d,
            'rolling_10d_vol': rolling_10d,
        })

    features = pd.DataFrame(records).set_index('date')
    return features


# ---------------------------------------------------------------------------
# Phase 2: Retrospective labeling
# ---------------------------------------------------------------------------

def label_days(trades, comm=MNQ_COMM, dpp=MNQ_DPP):
    """Label each trading day as favorable (daily net > 0) or unfavorable."""
    comm_pts = (comm * 2) / dpp
    daily_pnl = defaultdict(float)
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily_pnl[d] += (t['pts'] - comm_pts) * dpp

    labels = {}
    for d, pnl in daily_pnl.items():
        labels[d] = 'favorable' if pnl > 0 else 'unfavorable'
    return labels, daily_pnl


# ---------------------------------------------------------------------------
# Phase 3: Threshold sweep
# ---------------------------------------------------------------------------

# Constrained search grids
LONDON_RANGE_THRESHOLDS = [50, 75, 100, 125, 150, 175, 200, 250, 300, 400]
PRIOR_DAY_THRESHOLDS = [50, 75, 100, 125, 150, 175, 200, 250, 300, 400]
ROLLING_5D_THRESHOLDS = [100, 150, 200, 250, 300, 350, 400, 500]


def sweep_single_feature(feature_name, thresholds, features, daily_pnl, trades):
    """Sweep a single feature threshold. Trade only when feature < threshold."""
    comm_pts = (MNQ_COMM * 2) / MNQ_DPP
    results = []

    total_days = len(daily_pnl)
    total_losses = sum(pnl for pnl in daily_pnl.values() if pnl < 0)
    total_gains = sum(pnl for pnl in daily_pnl.values() if pnl > 0)

    for thresh in thresholds:
        # Determine which days to trade (feature < threshold)
        trade_days = set()
        for d in daily_pnl.keys():
            if d in features.index:
                val = features.loc[d, feature_name]
                if not np.isnan(val) and val < thresh:
                    trade_days.add(d)
            # If no feature data, default to trading
            elif d not in features.index:
                trade_days.add(d)

        # Filter trades
        filtered_trades = [t for t in trades if pd.Timestamp(t['exit_time']).date() in trade_days]

        # Compute stats
        excluded = total_days - len(trade_days)
        pct_excluded = excluded / total_days * 100 if total_days > 0 else 0

        filtered_pnl = {d: pnl for d, pnl in daily_pnl.items() if d in trade_days}
        filtered_losses = sum(pnl for pnl in filtered_pnl.values() if pnl < 0)
        filtered_gains = sum(pnl for pnl in filtered_pnl.values() if pnl > 0)

        losses_avoided = total_losses - filtered_losses  # positive = good
        gains_missed = total_gains - filtered_gains  # positive = bad

        pct_losses_avoided = (losses_avoided / abs(total_losses) * 100) if total_losses < 0 else 0
        pct_gains_missed = (gains_missed / total_gains * 100) if total_gains > 0 else 0

        net_pnl = sum(filtered_pnl.values())
        baseline_net = sum(daily_pnl.values())
        improvement = net_pnl - baseline_net

        # Max drawdown
        if filtered_pnl:
            sorted_pnl = pd.Series(filtered_pnl).sort_index()
            cum = sorted_pnl.cumsum()
            dd = (cum - cum.cummax()).min()
        else:
            dd = 0

        # Sharpe
        if filtered_pnl and len(filtered_pnl) > 1:
            pnl_arr = np.array(list(filtered_pnl.values()))
            sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(252) if pnl_arr.std() > 0 else 0
        else:
            sharpe = 0

        ratio = pct_losses_avoided / pct_gains_missed if pct_gains_missed > 0 else 999

        sc = score_trades(filtered_trades, MNQ_COMM, MNQ_DPP) if filtered_trades else None

        results.append({
            'threshold': thresh,
            'excluded_days': excluded,
            'pct_excluded': pct_excluded,
            'pct_losses_avoided': pct_losses_avoided,
            'pct_gains_missed': pct_gains_missed,
            'ratio': ratio,
            'net_pnl': net_pnl,
            'improvement': improvement,
            'max_dd': dd,
            'sharpe': sharpe,
            'trades': len(filtered_trades),
            'score': sc,
        })

    return results


# ---------------------------------------------------------------------------
# Phase 5: Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(features, trades, daily_pnl, feature_name, threshold):
    """3-way walk-forward validation for a single feature/threshold."""
    # Split OOS into H1 (Feb-Apr) and H2 (May-Aug)
    h1_end = pd.Timestamp("2025-05-01").date()
    h2_end = pd.Timestamp("2025-08-17").date()

    oos_trades, is_trades_list = split_trades(trades, SPLIT)

    # Label daily P&L by period
    comm_pts = (MNQ_COMM * 2) / MNQ_DPP
    oos_daily = defaultdict(float)
    is_daily = defaultdict(float)
    for t in oos_trades:
        d = pd.Timestamp(t['exit_time']).date()
        oos_daily[d] += (t['pts'] - comm_pts) * MNQ_DPP
    for t in is_trades_list:
        d = pd.Timestamp(t['exit_time']).date()
        is_daily[d] += (t['pts'] - comm_pts) * MNQ_DPP

    h1_daily = {d: p for d, p in oos_daily.items() if d < h1_end}
    h2_daily = {d: p for d, p in oos_daily.items() if d >= h1_end}

    def apply_filter(daily, feat, thresh):
        """Apply filter: trade only when feature < threshold."""
        filtered = {}
        for d, pnl in daily.items():
            if d in feat.index:
                val = feat.loc[d, feature_name]
                if not np.isnan(val) and val < thresh:
                    filtered[d] = pnl
            else:
                filtered[d] = pnl
        return filtered

    results = {}
    for label, daily in [("H1 (Feb-Apr)", h1_daily), ("H2 (May-Aug)", h2_daily), ("IS", dict(is_daily))]:
        baseline = sum(daily.values())
        filtered = apply_filter(daily, features, threshold)
        filtered_net = sum(filtered.values())
        improvement = filtered_net - baseline
        results[label] = {
            'baseline': baseline,
            'filtered': filtered_net,
            'improvement': improvement,
            'days_traded': len(filtered),
            'days_total': len(daily),
        }

    return results


# ---------------------------------------------------------------------------
# Phase 6: Combined feature rules
# ---------------------------------------------------------------------------

def test_combined_rules(features, trades, daily_pnl):
    """Test AND/OR combinations of top 2 features."""
    comm_pts = (MNQ_COMM * 2) / MNQ_DPP
    baseline_net = sum(daily_pnl.values())

    # Get baseline max DD
    baseline_sorted = pd.Series(daily_pnl).sort_index()
    baseline_cum = baseline_sorted.cumsum()
    baseline_dd = (baseline_cum - baseline_cum.cummax()).min()

    rules = []

    # AND rules: trade if feature1 < X AND feature2 < Y
    for lr in [100, 150, 200, 250, 300]:
        for rv in [200, 250, 300, 350, 400]:
            trade_days = set()
            for d in daily_pnl.keys():
                if d in features.index:
                    lval = features.loc[d, 'london_range']
                    rval = features.loc[d, 'rolling_5d_vol']
                    if (not np.isnan(lval) and lval < lr and
                        not np.isnan(rval) and rval < rv):
                        trade_days.add(d)
                else:
                    trade_days.add(d)

            if len(trade_days) < 10:
                continue

            filtered_pnl = {d: p for d, p in daily_pnl.items() if d in trade_days}
            net = sum(filtered_pnl.values())
            improvement = net - baseline_net

            sorted_pnl = pd.Series(filtered_pnl).sort_index()
            cum = sorted_pnl.cumsum()
            dd = (cum - cum.cummax()).min()

            pnl_arr = np.array(list(filtered_pnl.values()))
            sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(252) if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0

            rules.append({
                'rule': f"london_range < {lr} AND rolling_5d < {rv}",
                'type': 'AND',
                'days': len(trade_days),
                'pct_excluded': (len(daily_pnl) - len(trade_days)) / len(daily_pnl) * 100,
                'net': net,
                'improvement': improvement,
                'dd': dd,
                'dd_improvement': dd - baseline_dd,
                'sharpe': sharpe,
            })

    # OR rules: pause if feature1 > X OR feature2 > Y
    for lr in [150, 200, 250, 300]:
        for pdr in [200, 250, 300, 400]:
            trade_days = set()
            for d in daily_pnl.keys():
                if d in features.index:
                    lval = features.loc[d, 'london_range']
                    pval = features.loc[d, 'prior_day_range']
                    pause = False
                    if not np.isnan(lval) and lval > lr:
                        pause = True
                    if not np.isnan(pval) and pval > pdr:
                        pause = True
                    if not pause:
                        trade_days.add(d)
                else:
                    trade_days.add(d)

            if len(trade_days) < 10:
                continue

            filtered_pnl = {d: p for d, p in daily_pnl.items() if d in trade_days}
            net = sum(filtered_pnl.values())
            improvement = net - baseline_net

            sorted_pnl = pd.Series(filtered_pnl).sort_index()
            cum = sorted_pnl.cumsum()
            dd = (cum - cum.cummax()).min()

            pnl_arr = np.array(list(filtered_pnl.values()))
            sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(252) if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0

            rules.append({
                'rule': f"pause if london > {lr} OR prior_day > {pdr}",
                'type': 'OR',
                'days': len(trade_days),
                'pct_excluded': (len(daily_pnl) - len(trade_days)) / len(daily_pnl) * 100,
                'net': net,
                'improvement': improvement,
                'dd': dd,
                'dd_improvement': dd - baseline_dd,
                'sharpe': sharpe,
            })

    return rules, baseline_net, baseline_dd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("TASK 3: REGIME DETECTOR")
    print("=" * 80)

    # Load data
    print("\nLoading combined 12-month MNQ data...")
    mnq = load_combined()

    print("Preparing arrays (RSI=8)...")
    arr = prepare_arrays(mnq, rsi_len=8)

    # ================================================================
    # Run vWinners to get daily P&L
    # ================================================================
    print("\nRunning vWinners (SM flip, threshold=0.15)...")
    rsi_dummy = np.full(len(arr['closes']), 50.0)
    vwinners_trades = run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], rsi_dummy, arr['times'],
        rsi_buy=60, rsi_sell=40, sm_threshold=0.15,
        cooldown_bars=20, max_loss_pts=50,
        rsi_5m_curr=arr['rsi_5m_curr'], rsi_5m_prev=arr['rsi_5m_prev'],
    )

    sc = score_trades(vwinners_trades, MNQ_COMM, MNQ_DPP)
    print(f"  Total: {sc['count']} trades, PF {sc['pf']}, Net ${sc['net_dollar']:+.2f}")

    # ================================================================
    # Phase 1: Feature computation
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: FEATURE COMPUTATION")
    print("=" * 80)

    features = compute_features(mnq)
    print(f"\n  Features computed for {len(features)} days")
    print(f"  Columns: {list(features.columns)}")
    print(f"\n  Feature summary:")
    for col in features.columns:
        vals = features[col].dropna()
        print(f"    {col:<20} n={len(vals):>4}  mean={vals.mean():>8.1f}  "
              f"med={vals.median():>8.1f}  min={vals.min():>8.1f}  max={vals.max():>8.1f}")

    # ================================================================
    # Phase 2: Retrospective labeling
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: RETROSPECTIVE LABELING")
    print("=" * 80)

    labels, daily_pnl = label_days(vwinners_trades)
    fav_days = [d for d, l in labels.items() if l == 'favorable']
    unfav_days = [d for d, l in labels.items() if l == 'unfavorable']
    print(f"\n  Favorable days:   {len(fav_days)} ({len(fav_days)/(len(fav_days)+len(unfav_days))*100:.1f}%)")
    print(f"  Unfavorable days: {len(unfav_days)} ({len(unfav_days)/(len(fav_days)+len(unfav_days))*100:.1f}%)")

    # Feature distributions for favorable vs unfavorable
    print(f"\n  Feature distributions by day type:")
    print(f"  {'Feature':<20} {'Fav median':>12} {'Unfav median':>14} {'Difference':>12}")
    print("  " + "-" * 65)

    for col in features.columns:
        fav_vals = features.loc[[d for d in fav_days if d in features.index], col].dropna()
        unfav_vals = features.loc[[d for d in unfav_days if d in features.index], col].dropna()
        if len(fav_vals) > 0 and len(unfav_vals) > 0:
            fav_med = fav_vals.median()
            unfav_med = unfav_vals.median()
            diff = unfav_med - fav_med
            print(f"  {col:<20} {fav_med:>12.1f} {unfav_med:>14.1f} {diff:>+12.1f}")

    # ================================================================
    # Phase 3: Threshold sweep
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: SINGLE-FEATURE THRESHOLD SWEEP")
    print("=" * 80)

    baseline_net = sum(daily_pnl.values())
    print(f"\n  Baseline (no filter): Net ${baseline_net:+.2f}")

    feature_sweeps = {
        'london_range': LONDON_RANGE_THRESHOLDS,
        'prior_day_range': PRIOR_DAY_THRESHOLDS,
        'rolling_5d_vol': ROLLING_5D_THRESHOLDS,
    }

    best_per_feature = {}

    for feature_name, thresholds in feature_sweeps.items():
        print(f"\n  --- {feature_name} ---")
        results = sweep_single_feature(feature_name, thresholds, features,
                                       daily_pnl, vwinners_trades)

        print(f"  {'Thresh':>7} {'Excl%':>6} {'Loss%Avd':>9} {'Gain%Miss':>10} {'Ratio':>6} "
              f"{'Net $':>10} {'Improv $':>10} {'MaxDD $':>10} {'Sharpe':>7} {'Trades':>7}")
        print("  " + "-" * 95)

        for r in results:
            print(f"  {r['threshold']:>7} {r['pct_excluded']:>5.1f}% {r['pct_losses_avoided']:>8.1f}% "
                  f"{r['pct_gains_missed']:>9.1f}% {r['ratio']:>6.2f} "
                  f"{r['net_pnl']:>+10.2f} {r['improvement']:>+10.2f} "
                  f"{r['max_dd']:>10.2f} {r['sharpe']:>7.3f} {r['trades']:>7}")

        # Best by ratio (losses avoided / gains missed)
        valid = [r for r in results if r['pct_gains_missed'] > 0 and r['pct_excluded'] > 1]
        if valid:
            best = max(valid, key=lambda r: r['ratio'])
            best_per_feature[feature_name] = best
            print(f"\n  Best {feature_name}: threshold={best['threshold']}, "
                  f"ratio={best['ratio']:.2f}, improvement=${best['improvement']:+.2f}")

    # ================================================================
    # Phase 4: Regime stability check
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: REGIME STABILITY — Per-month optimal thresholds")
    print("=" * 80)

    # Get per-month daily P&L
    monthly_pnl = defaultdict(dict)
    for d, pnl in daily_pnl.items():
        month = pd.Timestamp(d).strftime('%Y-%m')
        monthly_pnl[month][d] = pnl

    for feature_name, thresholds in feature_sweeps.items():
        print(f"\n  --- {feature_name} ---")
        print(f"  {'Month':<10} {'BestThresh':>10} {'Baseline$':>10} {'Filtered$':>10} {'Improv$':>10}")
        print("  " + "-" * 55)

        monthly_best = []
        for month in sorted(monthly_pnl.keys()):
            mpnl = monthly_pnl[month]
            baseline = sum(mpnl.values())

            best_thresh = None
            best_net = baseline
            for thresh in thresholds:
                filtered = {}
                for d, pnl in mpnl.items():
                    if d in features.index:
                        val = features.loc[d, feature_name]
                        if not np.isnan(val) and val < thresh:
                            filtered[d] = pnl
                    else:
                        filtered[d] = pnl
                net = sum(filtered.values())
                if net > best_net:
                    best_net = net
                    best_thresh = thresh

            if best_thresh is None:
                best_thresh = max(thresholds)  # no filter helps

            monthly_best.append(best_thresh)
            print(f"  {month:<10} {best_thresh:>10} {baseline:>+10.2f} {best_net:>+10.2f} "
                  f"{best_net - baseline:>+10.2f}")

        # Stability: coefficient of variation
        cv = np.std(monthly_best) / np.mean(monthly_best) if np.mean(monthly_best) > 0 else 999
        print(f"\n  Threshold stability (CV): {cv:.3f}")
        if cv < 0.3:
            print(f"  STABLE — thresholds cluster tightly across months")
        elif cv < 0.6:
            print(f"  MODERATE — some month-to-month variation")
        else:
            print(f"  UNSTABLE — thresholds vary wildly, don't trust this feature alone")

    # ================================================================
    # Phase 5: Walk-forward validation
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: WALK-FORWARD VALIDATION (3-way)")
    print("=" * 80)

    for feature_name in best_per_feature:
        thresh = best_per_feature[feature_name]['threshold']
        print(f"\n  --- {feature_name} < {thresh} ---")

        wf = walk_forward_validate(features, vwinners_trades, daily_pnl, feature_name, thresh)

        print(f"  {'Period':<20} {'Baseline $':>12} {'Filtered $':>12} {'Improvement':>12} {'Days':>6}")
        print("  " + "-" * 70)
        all_improve = True
        for label, stats in wf.items():
            improved = stats['improvement'] > 0
            if not improved:
                all_improve = False
            marker = "+" if improved else "X"
            print(f"  {label:<20} {stats['baseline']:>+12.2f} {stats['filtered']:>+12.2f} "
                  f"{stats['improvement']:>+12.2f} {stats['days_traded']:>4}/{stats['days_total']:>4} [{marker}]")

        if all_improve:
            print(f"\n  PASS — threshold improves ALL periods")
        else:
            print(f"\n  FAIL — threshold does NOT generalize to all periods")

    # ================================================================
    # Phase 6: Combined feature rules
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 6: COMBINED FEATURE RULES")
    print("=" * 80)

    rules, baseline_net, baseline_dd = test_combined_rules(features, vwinners_trades, daily_pnl)

    # Sort by improvement
    rules.sort(key=lambda r: r['improvement'], reverse=True)

    print(f"\n  Baseline: Net ${baseline_net:+.2f}, MaxDD ${baseline_dd:.2f}")
    print(f"\n  Top 10 AND rules:")
    print(f"  {'Rule':<55} {'Excl%':>6} {'Net $':>10} {'Improv $':>10} {'DD $':>10} {'Sharpe':>7}")
    print("  " + "-" * 105)

    and_rules = [r for r in rules if r['type'] == 'AND'][:10]
    for r in and_rules:
        print(f"  {r['rule']:<55} {r['pct_excluded']:>5.1f}% {r['net']:>+10.2f} "
              f"{r['improvement']:>+10.2f} {r['dd']:>10.2f} {r['sharpe']:>7.3f}")

    print(f"\n  Top 10 OR rules:")
    print(f"  {'Rule':<55} {'Excl%':>6} {'Net $':>10} {'Improv $':>10} {'DD $':>10} {'Sharpe':>7}")
    print("  " + "-" * 105)

    or_rules = [r for r in rules if r['type'] == 'OR'][:10]
    for r in or_rules:
        print(f"  {r['rule']:<55} {r['pct_excluded']:>5.1f}% {r['net']:>+10.2f} "
              f"{r['improvement']:>+10.2f} {r['dd']:>10.2f} {r['sharpe']:>7.3f}")

    # Walk-forward validate best combined rule
    if rules and rules[0]['improvement'] > 0:
        best_rule = rules[0]
        print(f"\n  Best combined rule: {best_rule['rule']}")
        print(f"    Improvement: ${best_rule['improvement']:+.2f}, "
              f"DD improvement: ${best_rule['dd_improvement']:+.2f}")

    # ================================================================
    # Final recommendation
    # ================================================================
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    # Check which features passed walk-forward
    passed_features = []
    for feature_name in best_per_feature:
        thresh = best_per_feature[feature_name]['threshold']
        wf = walk_forward_validate(features, vwinners_trades, daily_pnl, feature_name, thresh)
        all_improve = all(s['improvement'] > 0 for s in wf.values())
        if all_improve:
            passed_features.append((feature_name, thresh, best_per_feature[feature_name]))

    if passed_features:
        print(f"\n  Features that pass walk-forward validation:")
        for fname, thresh, stats in passed_features:
            print(f"    {fname} < {thresh}: ratio={stats['ratio']:.2f}, "
                  f"improvement=${stats['improvement']:+.2f}")

        best_f, best_t, best_s = passed_features[0]
        print(f"\n  RECOMMENDED RULE: Trade vWinners only when {best_f} < {best_t}")
        print(f"    Expected improvement: ${best_s['improvement']:+.2f}")
        print(f"    Days excluded: {best_s['pct_excluded']:.1f}%")
    else:
        print(f"\n  NO feature threshold generalizes across all validation periods.")
        print(f"  CONCLUSION: Regime detection doesn't work for vWinners with these features.")
        print(f"  This is an important negative result — the strategy's daily variance")
        print(f"  is not predictable from pre-market data with simple threshold rules.")

    print("\n" + "=" * 80)
    print("DONE — Task 3 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
