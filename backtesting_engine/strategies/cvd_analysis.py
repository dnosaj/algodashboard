"""
CVD vs Smart Money Analysis
=============================
Analyzes whether Cumulative Volume Delta (CVD) from real order flow data
adds predictive value beyond the Smart Money (SM) indicator.

TRAIN PERIOD ONLY: Aug 17 - Nov 16, 2025 (~3 months)
Do NOT look at test data (Nov 17, 2025 - Feb 13, 2026).

Key analyses:
  1. Correlation: Pearson r between SM_Net and CVD_norm
  2. Lead-lag: Cross-correlation at -5 to +5 min offsets
  3. Sign agreement: % of bars where SM and CVD agree on direction
  4. Trade-level: PF/WR for CVD-agreeing vs CVD-disagreeing trades
  5. Divergence: When SM and CVD disagree, what happens to price?
  6. CVD flip timing: Does CVD flip before SM?
  7. Monthly breakdown: Per month to check stability

Usage:
  python3 cvd_analysis.py --delta data/databento_NQ_delta_1min_*.csv
  python3 cvd_analysis.py --delta data/delta.csv --train-only
  python3 cvd_analysis.py --delta data/delta.csv --all-periods
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_instrument_1min, prepare_backtest_arrays_1min,
    run_v9_baseline, score_trades, fmt_score, compute_et_minutes,
    NY_OPEN_ET, NY_CLOSE_ET,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Train/test split dates
TRAIN_START = pd.Timestamp("2025-08-17", tz='UTC')
TRAIN_END = pd.Timestamp("2025-11-17", tz='UTC')
TEST_START = pd.Timestamp("2025-11-17", tz='UTC')
TEST_END = pd.Timestamp("2026-02-14", tz='UTC')


def load_delta_bars(delta_path):
    """Load volume delta bars CSV and return UTC-indexed DataFrame."""
    df = pd.read_csv(delta_path)
    df['time_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time_utc')
    return df


def align_sm_and_cvd(ohlcv_1m, delta_df, period_start, period_end):
    """Align SM and CVD on matching 1-min timestamps within a period.

    Returns DataFrame with SM_Net, CVD, CVD_norm, Close columns,
    filtered to RTH session (10 AM - 4 PM ET).
    """
    # Filter to period
    sm_data = ohlcv_1m[(ohlcv_1m.index >= period_start) &
                        (ohlcv_1m.index < period_end)].copy()
    cvd_data = delta_df[(delta_df.index >= period_start) &
                         (delta_df.index < period_end)].copy()

    if len(sm_data) == 0 or len(cvd_data) == 0:
        print(f"  WARNING: No data in period {period_start} to {period_end}")
        return pd.DataFrame()

    # Join on timestamp
    merged = sm_data[['SM_Net', 'Close']].join(
        cvd_data[['delta', 'CVD', 'CVD_norm']], how='inner'
    )

    if len(merged) == 0:
        print("  WARNING: No overlapping timestamps between SM and CVD data")
        print(f"  SM range: {sm_data.index[0]} to {sm_data.index[-1]}")
        print(f"  CVD range: {cvd_data.index[0]} to {cvd_data.index[-1]}")
        return pd.DataFrame()

    # Filter to RTH session (10 AM - 4 PM ET)
    et_mins = compute_et_minutes(merged.index)
    rth_mask = (et_mins >= NY_OPEN_ET) & (et_mins < NY_CLOSE_ET)
    merged = merged[rth_mask]

    print(f"  Aligned bars (RTH): {len(merged):,}")
    return merged


def analyze_correlation(merged):
    """1. Pearson correlation between SM_Net and CVD_norm."""
    print("\n" + "=" * 60)
    print("1. CORRELATION: SM_Net vs CVD_norm")
    print("=" * 60)

    if len(merged) < 10:
        print("  Insufficient data")
        return

    sm = merged['SM_Net'].values
    cvd = merged['CVD_norm'].values

    # Remove NaN/inf
    valid = np.isfinite(sm) & np.isfinite(cvd)
    sm = sm[valid]
    cvd = cvd[valid]

    r = np.corrcoef(sm, cvd)[0, 1]
    print(f"  Pearson r: {r:.4f}")

    if abs(r) > 0.9:
        print("  CONCLUSION: Very high correlation. CVD may be redundant with SM.")
    elif abs(r) > 0.7:
        print("  CONCLUSION: High correlation. CVD has some independent info.")
    elif abs(r) > 0.4:
        print("  CONCLUSION: Moderate correlation. CVD has meaningful independent info.")
    else:
        print("  CONCLUSION: Low correlation. CVD captures very different signal than SM.")

    # Also check correlation with raw delta
    r_delta = np.corrcoef(sm[valid], merged['delta'].values[valid])[0, 1] if 'delta' in merged.columns else 0
    print(f"  SM vs raw delta r: {r_delta:.4f}")

    return r


def analyze_lead_lag(merged, max_lag=5):
    """2. Cross-correlation at -5 to +5 min offsets."""
    print("\n" + "=" * 60)
    print("2. LEAD-LAG: Cross-correlation SM vs CVD")
    print("=" * 60)

    if len(merged) < 20:
        print("  Insufficient data")
        return

    sm = merged['SM_Net'].values
    cvd = merged['CVD_norm'].values

    valid = np.isfinite(sm) & np.isfinite(cvd)
    sm = sm[valid]
    cvd = cvd[valid]

    print(f"  Lag (min)  |  Correlation")
    print(f"  -----------+-------------")

    best_lag = 0
    best_r = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # CVD leads SM: correlate CVD[:-lag] with SM[-lag:]
            r = np.corrcoef(cvd[:lag], sm[-lag:])[0, 1]
            label = f"CVD leads by {-lag}"
        elif lag > 0:
            # SM leads CVD: correlate SM[:-lag] with CVD[lag:]
            r = np.corrcoef(sm[:-lag], cvd[lag:])[0, 1]
            label = f"SM leads by {lag}"
        else:
            r = np.corrcoef(sm, cvd)[0, 1]
            label = "Contemporaneous"

        marker = " <-- best" if abs(r) > abs(best_r) else ""
        if abs(r) > abs(best_r):
            best_r = r
            best_lag = lag

        print(f"  {lag:+3d}         |  {r:+.4f}  ({label}){marker}")

    print()
    if best_lag < 0:
        print(f"  RESULT: CVD leads SM by {-best_lag} min (r={best_r:.4f})")
        print("  -> CVD has early-warning potential!")
    elif best_lag > 0:
        print(f"  RESULT: SM leads CVD by {best_lag} min (r={best_r:.4f})")
        print("  -> SM reacts faster; CVD is lagging.")
    else:
        print(f"  RESULT: Best correlation at lag 0 (r={best_r:.4f})")
        print("  -> SM and CVD are synchronized.")


def analyze_sign_agreement(merged):
    """3. % of bars where SM and CVD agree on direction."""
    print("\n" + "=" * 60)
    print("3. SIGN AGREEMENT: SM vs CVD direction")
    print("=" * 60)

    if len(merged) < 10:
        print("  Insufficient data")
        return

    sm = merged['SM_Net'].values
    cvd = merged['CVD_norm'].values

    sm_bull = sm > 0
    sm_bear = sm < 0
    cvd_bull = cvd > 0
    cvd_bear = cvd < 0

    agree = (sm_bull & cvd_bull) | (sm_bear & cvd_bear)
    disagree = (sm_bull & cvd_bear) | (sm_bear & cvd_bull)
    neutral = ~agree & ~disagree  # one or both are exactly 0

    n = len(merged)
    print(f"  Agree:    {agree.sum():>6} ({agree.sum()/n*100:.1f}%)")
    print(f"  Disagree: {disagree.sum():>6} ({disagree.sum()/n*100:.1f}%)")
    print(f"  Neutral:  {neutral.sum():>6} ({neutral.sum()/n*100:.1f}%)")

    # What happens to price during agreement vs disagreement?
    fwd_ret = np.diff(merged['Close'].values, prepend=merged['Close'].values[0])

    if agree.sum() > 0:
        avg_agree = fwd_ret[agree].mean()
        std_agree = fwd_ret[agree].std()
        print(f"\n  Avg 1-bar return when AGREE:    {avg_agree:+.4f} (std: {std_agree:.4f})")

    if disagree.sum() > 0:
        avg_disagree = fwd_ret[disagree].mean()
        std_disagree = fwd_ret[disagree].std()
        print(f"  Avg 1-bar return when DISAGREE: {avg_disagree:+.4f} (std: {std_disagree:.4f})")

    return agree.sum() / n


def analyze_trade_level(ohlcv_1m, delta_df, period_start, period_end):
    """4. Tag v11 trades with CVD state. Compare PF/WR for agreeing vs disagreeing."""
    print("\n" + "=" * 60)
    print("4. TRADE-LEVEL: CVD agreement vs disagreement")
    print("=" * 60)

    # Run v11 MNQ baseline on the train period
    period_data = ohlcv_1m[(ohlcv_1m.index >= period_start) &
                            (ohlcv_1m.index < period_end)]
    if len(period_data) < 100:
        print("  Insufficient data")
        return

    # v11 MNQ params: SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50
    arr = prepare_backtest_arrays_1min(period_data, rsi_len=8)
    trades = run_v9_baseline(
        arr, rsi_len=8, rsi_buy=60, rsi_sell=40,
        sm_threshold=0.0, cooldown=20, max_loss_pts=50,
    )

    if not trades:
        print("  No trades in period")
        return

    print(f"  Total trades: {len(trades)}")

    # Get CVD for the period
    cvd_data = delta_df[(delta_df.index >= period_start) &
                         (delta_df.index < period_end)]

    if len(cvd_data) == 0:
        print("  No CVD data in period")
        return

    # Tag each trade with CVD state at entry
    cvd_agree_trades = []
    cvd_disagree_trades = []
    cvd_missing = 0

    for trade in trades:
        entry_time = pd.Timestamp(trade['entry_time'])
        if entry_time.tz is None:
            entry_time = entry_time.tz_localize('UTC')

        # Find closest CVD bar
        idx = cvd_data.index.get_indexer([entry_time], method='nearest')[0]
        if idx < 0 or idx >= len(cvd_data):
            cvd_missing += 1
            continue

        cvd_val = cvd_data.iloc[idx]['CVD_norm']
        trade_side = trade['side']

        # Agreement: long + CVD bullish, or short + CVD bearish
        if trade_side == 'long':
            agrees = cvd_val > 0
        else:
            agrees = cvd_val < 0

        if agrees:
            cvd_agree_trades.append(trade)
        else:
            cvd_disagree_trades.append(trade)

    if cvd_missing > 0:
        print(f"  CVD lookup misses: {cvd_missing}")

    print(f"\n  CVD-agreeing trades: {len(cvd_agree_trades)}")
    print(f"  CVD-disagreeing trades: {len(cvd_disagree_trades)}")

    sc_agree = score_trades(cvd_agree_trades) if cvd_agree_trades else None
    sc_disagree = score_trades(cvd_disagree_trades) if cvd_disagree_trades else None
    sc_all = score_trades(trades)

    print(f"\n  ALL:      {fmt_score(sc_all, 'ALL')}")
    print(f"  AGREE:    {fmt_score(sc_agree, 'AGREE')}")
    print(f"  DISAGREE: {fmt_score(sc_disagree, 'DISAGREE')}")

    if sc_agree and sc_disagree:
        pf_diff = sc_agree['pf'] - sc_disagree['pf']
        wr_diff = sc_agree['win_rate'] - sc_disagree['win_rate']
        print(f"\n  PF difference (agree - disagree): {pf_diff:+.3f}")
        print(f"  WR difference (agree - disagree): {wr_diff:+.1f}%")

        if pf_diff > 0.3 and wr_diff > 5:
            print("  CONCLUSION: CVD agreement significantly improves trade quality!")
        elif pf_diff > 0:
            print("  CONCLUSION: CVD agreement slightly improves trade quality.")
        else:
            print("  CONCLUSION: CVD agreement does NOT improve trade quality.")

    return cvd_agree_trades, cvd_disagree_trades


def analyze_divergence(merged):
    """5. When SM and CVD disagree, what happens to price over next N bars?"""
    print("\n" + "=" * 60)
    print("5. DIVERGENCE EPISODES: SM bullish + CVD bearish (and vice versa)")
    print("=" * 60)

    if len(merged) < 20:
        print("  Insufficient data")
        return

    sm = merged['SM_Net'].values
    cvd = merged['CVD_norm'].values
    close = merged['Close'].values

    # Type 1: SM bullish, CVD bearish (potential false bullish signal)
    type1 = (sm > 0) & (cvd < -0.3)  # Strong CVD disagreement
    # Type 2: SM bearish, CVD bullish (potential false bearish signal)
    type2 = (sm < 0) & (cvd > 0.3)

    for label, mask in [("SM Bull + CVD Bear", type1), ("SM Bear + CVD Bull", type2)]:
        episodes = mask.sum()
        print(f"\n  {label}: {episodes} bars ({episodes/len(merged)*100:.1f}%)")

        if episodes < 5:
            print("    Too few episodes for analysis")
            continue

        # Forward returns: 1, 5, 15 bars ahead
        for horizon in [1, 5, 15]:
            if len(close) <= horizon:
                continue
            fwd = np.zeros(len(close))
            fwd[:-horizon] = close[horizon:] - close[:-horizon]
            fwd[-horizon:] = 0

            valid_mask = mask.copy()
            valid_mask[-horizon:] = False

            if valid_mask.sum() > 0:
                avg_fwd = fwd[valid_mask].mean()
                std_fwd = fwd[valid_mask].std()
                pct_positive = (fwd[valid_mask] > 0).sum() / valid_mask.sum() * 100

                # Compare with same-direction SM (no disagreement)
                if "Bull" in label.split("+")[0]:
                    baseline_mask = (sm > 0) & (cvd > 0) & ~mask
                else:
                    baseline_mask = (sm < 0) & (cvd < 0) & ~mask
                baseline_mask[-horizon:] = False

                if baseline_mask.sum() > 0:
                    baseline_fwd = fwd[baseline_mask].mean()
                    print(f"    {horizon}-bar fwd: {avg_fwd:+.2f}pts (std {std_fwd:.2f}), "
                          f"{pct_positive:.0f}% positive | baseline: {baseline_fwd:+.2f}pts")
                else:
                    print(f"    {horizon}-bar fwd: {avg_fwd:+.2f}pts (std {std_fwd:.2f}), "
                          f"{pct_positive:.0f}% positive")


def analyze_cvd_flip_timing(merged):
    """6. Does CVD flip before SM? Compare timing of sign changes."""
    print("\n" + "=" * 60)
    print("6. CVD FLIP TIMING: Does CVD flip before SM?")
    print("=" * 60)

    if len(merged) < 20:
        print("  Insufficient data")
        return

    sm = merged['SM_Net'].values
    cvd = merged['CVD_norm'].values

    # Detect sign flips
    sm_flip = np.diff(np.sign(sm)) != 0
    cvd_flip = np.diff(np.sign(cvd)) != 0

    sm_flip_indices = np.where(sm_flip)[0]
    cvd_flip_indices = np.where(cvd_flip)[0]

    print(f"  SM flips: {len(sm_flip_indices)}")
    print(f"  CVD flips: {len(cvd_flip_indices)}")

    if len(sm_flip_indices) == 0:
        print("  No SM flips found. Cannot compare timing.")
        return

    # For each SM flip, find the nearest CVD flip and measure the lag
    lags = []
    for sm_idx in sm_flip_indices:
        # Find nearest CVD flip within +/- 30 bars
        nearby = cvd_flip_indices[np.abs(cvd_flip_indices - sm_idx) <= 30]
        if len(nearby) > 0:
            nearest = nearby[np.argmin(np.abs(nearby - sm_idx))]
            lag = nearest - sm_idx  # negative = CVD flipped first
            lags.append(lag)

    if not lags:
        print("  No matched flips found")
        return

    lags = np.array(lags)
    cvd_leads = (lags < 0).sum()
    sm_leads = (lags > 0).sum()
    simultaneous = (lags == 0).sum()

    print(f"\n  Matched flip pairs: {len(lags)}")
    print(f"  CVD flips BEFORE SM: {cvd_leads} ({cvd_leads/len(lags)*100:.1f}%)")
    print(f"  SM flips BEFORE CVD: {sm_leads} ({sm_leads/len(lags)*100:.1f}%)")
    print(f"  Simultaneous:        {simultaneous} ({simultaneous/len(lags)*100:.1f}%)")
    print(f"  Mean lag: {lags.mean():+.1f} bars (negative = CVD leads)")
    print(f"  Median lag: {np.median(lags):+.1f} bars")

    if lags.mean() < -1:
        print("\n  RESULT: CVD tends to flip BEFORE SM -- early warning potential!")
    elif lags.mean() > 1:
        print("\n  RESULT: SM tends to flip BEFORE CVD -- CVD is lagging.")
    else:
        print("\n  RESULT: CVD and SM flip at approximately the same time.")


def analyze_monthly(merged):
    """7. Per-month breakdown to check stability."""
    print("\n" + "=" * 60)
    print("7. MONTHLY BREAKDOWN")
    print("=" * 60)

    if len(merged) < 10:
        print("  Insufficient data")
        return

    merged_copy = merged.copy()
    merged_copy['month'] = merged_copy.index.to_period('M')

    sm = merged_copy['SM_Net']
    cvd = merged_copy['CVD_norm']

    print(f"\n  {'Month':>10} | {'Bars':>6} | {'r(SM,CVD)':>10} | {'Sign Agree%':>11} | {'Avg |SM|':>9} | {'Avg |CVD|':>10}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*10}-+-{'-'*11}-+-{'-'*9}-+-{'-'*10}")

    for month, group in merged_copy.groupby('month'):
        n = len(group)
        if n < 10:
            continue

        sm_vals = group['SM_Net'].values
        cvd_vals = group['CVD_norm'].values

        valid = np.isfinite(sm_vals) & np.isfinite(cvd_vals)
        if valid.sum() < 5:
            continue

        r = np.corrcoef(sm_vals[valid], cvd_vals[valid])[0, 1]

        agree = ((sm_vals > 0) & (cvd_vals > 0)) | ((sm_vals < 0) & (cvd_vals < 0))
        agree_pct = agree.sum() / n * 100

        avg_sm = np.mean(np.abs(sm_vals[valid]))
        avg_cvd = np.mean(np.abs(cvd_vals[valid]))

        print(f"  {str(month):>10} | {n:>6} | {r:>+10.4f} | {agree_pct:>10.1f}% | {avg_sm:>9.4f} | {avg_cvd:>10.4f}")


def generate_recommendation(merged, trade_results=None):
    """Generate go/no-go recommendation based on all analyses."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if len(merged) < 100:
        print("  INSUFFICIENT DATA for recommendation")
        return

    sm = merged['SM_Net'].values
    cvd = merged['CVD_norm'].values
    valid = np.isfinite(sm) & np.isfinite(cvd)

    # Score criteria
    criteria = []

    # 1. Not redundant (r < 0.9)
    r = np.corrcoef(sm[valid], cvd[valid])[0, 1]
    not_redundant = abs(r) < 0.9
    criteria.append(("Not redundant (r < 0.9)", not_redundant, f"r={r:.4f}"))

    # 2. CVD leads SM (mean lag < 0)
    sm_flip = np.diff(np.sign(sm)) != 0
    cvd_flip = np.diff(np.sign(cvd)) != 0
    sm_flip_indices = np.where(sm_flip)[0]
    cvd_flip_indices = np.where(cvd_flip)[0]

    lags = []
    for sm_idx in sm_flip_indices:
        nearby = cvd_flip_indices[np.abs(cvd_flip_indices - sm_idx) <= 30]
        if len(nearby) > 0:
            nearest = nearby[np.argmin(np.abs(nearby - sm_idx))]
            lags.append(nearest - sm_idx)

    if lags:
        mean_lag = np.mean(lags)
        leads = mean_lag < -0.5
    else:
        mean_lag = 0
        leads = False
    criteria.append(("CVD leads SM (mean lag < -0.5)", leads, f"lag={mean_lag:+.1f}"))

    # 3. Sign agreement above chance (>55%)
    agree = ((sm > 0) & (cvd > 0)) | ((sm < 0) & (cvd < 0))
    agree_pct = agree.sum() / len(sm) * 100
    agreement_meaningful = agree_pct > 55
    criteria.append(("Sign agreement > 55%", agreement_meaningful, f"{agree_pct:.1f}%"))

    # 4. Trade-level improvement (if available)
    if trade_results:
        agree_trades, disagree_trades = trade_results
        sc_agree = score_trades(agree_trades) if agree_trades else None
        sc_disagree = score_trades(disagree_trades) if disagree_trades else None
        if sc_agree and sc_disagree:
            pf_improvement = sc_agree['pf'] > sc_disagree['pf']
            criteria.append(("CVD-agree trades have higher PF",
                           pf_improvement,
                           f"agree PF={sc_agree['pf']:.3f}, disagree PF={sc_disagree['pf']:.3f}"))

    print("\n  Criteria Assessment:")
    passes = 0
    for name, passed, detail in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: {detail}")
        if passed:
            passes += 1

    print(f"\n  Score: {passes}/{len(criteria)} criteria passed")

    if passes >= 3:
        print("\n  RECOMMENDATION: PROCEED to Phase 4 (backtest integration)")
        print("  CVD shows enough independent signal to test as a feature.")
    elif passes >= 2:
        print("\n  RECOMMENDATION: MAYBE proceed to Phase 4")
        print("  CVD shows some promise but results are marginal.")
        print("  Consider testing only the strongest signal (entry filter or flip exit).")
    else:
        print("\n  RECOMMENDATION: STOP. Do NOT proceed to Phase 4.")
        print("  CVD does not add meaningful value beyond SM.")
        print("  Save the engineering effort for other research.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CVD vs Smart Money indicator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--delta", required=True,
                        help="Path to volume delta CSV from compute_volume_delta.py")
    parser.add_argument("--instrument", default="MNQ",
                        help="Instrument for SM data (default: MNQ)")
    parser.add_argument("--train-only", action="store_true", default=True,
                        help="Only analyze train period (default: True)")
    parser.add_argument("--all-periods", action="store_true",
                        help="Analyze both train and test periods")

    args = parser.parse_args()

    delta_path = Path(args.delta)
    if not delta_path.exists():
        print(f"ERROR: Delta file not found: {delta_path}")
        sys.exit(1)

    print("=" * 70)
    print("CVD vs SMART MONEY ANALYSIS")
    print("=" * 70)
    if not args.all_periods:
        print(f"  Period: TRAIN ONLY ({TRAIN_START.date()} to {TRAIN_END.date()})")
    else:
        print(f"  Period: ALL ({TRAIN_START.date()} to {TEST_END.date()})")
    print(f"  Instrument: {args.instrument}")
    print()

    # Load data
    print("Loading data...")
    delta_df = load_delta_bars(args.delta)
    ohlcv_1m = load_instrument_1min(args.instrument)

    # Ensure OHLCV has UTC timezone
    if ohlcv_1m.index.tz is None:
        ohlcv_1m.index = ohlcv_1m.index.tz_localize('UTC')

    print(f"  Delta bars: {len(delta_df):,}")
    print(f"  OHLCV bars: {len(ohlcv_1m):,}")

    # Determine analysis period
    if args.all_periods:
        period_start = TRAIN_START
        period_end = TEST_END
    else:
        period_start = TRAIN_START
        period_end = TRAIN_END

    # Align data
    merged = align_sm_and_cvd(ohlcv_1m, delta_df, period_start, period_end)
    if len(merged) == 0:
        print("\nERROR: No aligned data. Check timestamps and date ranges.")
        sys.exit(1)

    # Run all analyses
    analyze_correlation(merged)
    analyze_lead_lag(merged)
    analyze_sign_agreement(merged)
    trade_results = analyze_trade_level(ohlcv_1m, delta_df, period_start, period_end)
    analyze_divergence(merged)
    analyze_cvd_flip_timing(merged)
    analyze_monthly(merged)

    # Generate recommendation
    generate_recommendation(merged, trade_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
