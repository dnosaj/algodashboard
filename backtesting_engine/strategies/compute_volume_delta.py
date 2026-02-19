"""
Compute Volume Delta Bars from NQ Tick Data
=============================================
Aggregates tick-level trades into 1-min volume delta bars and computes
Cumulative Volume Delta (CVD) with session resets.

Input: NQ tick parquet from fetch_databento_ticks.py
Output: CSV with 1-min delta bars aligned with existing OHLCV data

Columns produced:
  time          - Unix timestamp (matches OHLCV convention)
  buy_vol       - Buy aggressor volume (side='B')
  sell_vol      - Sell aggressor volume (side='A')
  delta         - buy_vol - sell_vol
  total_vol     - buy_vol + sell_vol (excludes side='N')
  CVD           - Cumulative delta with session reset at 6 PM ET
  CVD_norm      - CVD normalized to [-1,1] via rolling max (like SM normalization)

Usage:
  python3 compute_volume_delta.py --ticks data/databento_NQ_ticks_*.parquet
  python3 compute_volume_delta.py --ticks data/databento_NQ_ticks_*.parquet --output delta_bars.csv
  python3 compute_volume_delta.py --ticks data/databento_NQ_ticks_*.parquet --validate-against MNQ
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ET_TZ = ZoneInfo("America/New_York")


def aggregate_ticks_to_1min(tick_path):
    """Aggregate tick data into 1-min volume delta bars.

    Args:
        tick_path: Path to tick parquet file

    Returns:
        DataFrame with columns: buy_vol, sell_vol, delta, total_vol
        Indexed by 1-min timestamps (UTC)
    """
    print(f"Loading ticks from: {tick_path}")
    df = pd.read_parquet(tick_path)
    print(f"  Total ticks: {len(df):,}")

    # Parse timestamps
    df['ts'] = pd.to_datetime(df['ts_event'], utc=True)

    # Side field stats
    side_counts = df['side'].value_counts()
    n_total = len(df)
    n_buy = side_counts.get('B', 0)
    n_sell = side_counts.get('A', 0)
    n_unknown = side_counts.get('N', 0) + side_counts.get('', 0)
    n_usable = n_buy + n_sell
    pct_usable = n_usable / n_total * 100

    print(f"  Buy ticks (B):  {n_buy:>12,} ({n_buy/n_total*100:.1f}%)")
    print(f"  Sell ticks (A): {n_sell:>12,} ({n_sell/n_total*100:.1f}%)")
    print(f"  Unknown (N):    {n_unknown:>12,} ({n_unknown/n_total*100:.1f}%)")
    print(f"  Usable (B+A):   {n_usable:>12,} ({pct_usable:.1f}%)")

    if pct_usable < 90:
        print("  WARNING: <90% side coverage. Delta signal may be unreliable.")

    # Separate buy and sell
    buys = df[df['side'] == 'B'].copy()
    sells = df[df['side'] == 'A'].copy()

    # Floor to 1-min
    buys['bar'] = buys['ts'].dt.floor('1min')
    sells['bar'] = sells['ts'].dt.floor('1min')

    # Aggregate by 1-min bar
    buy_agg = buys.groupby('bar')['size'].sum().rename('buy_vol')
    sell_agg = sells.groupby('bar')['size'].sum().rename('sell_vol')

    # Also aggregate all ticks for total volume validation
    df['bar'] = df['ts'].dt.floor('1min')
    total_agg = df.groupby('bar')['size'].sum().rename('total_vol_all')

    # Combine — cast to int64 to avoid uint64 underflow on subtraction
    result = pd.DataFrame(index=buy_agg.index.union(sell_agg.index).sort_values())
    result['buy_vol'] = buy_agg.reindex(result.index, fill_value=0).astype(np.int64)
    result['sell_vol'] = sell_agg.reindex(result.index, fill_value=0).astype(np.int64)
    result['delta'] = result['buy_vol'] - result['sell_vol']
    result['total_vol'] = result['buy_vol'] + result['sell_vol']
    result['total_vol_all'] = total_agg.reindex(result.index, fill_value=0).astype(np.int64)

    result.index.name = 'time'

    print(f"\n  Aggregated to {len(result):,} 1-min bars")
    print(f"  Date range: {result.index[0]} to {result.index[-1]}")
    print(f"  Avg delta/bar: {result['delta'].mean():.1f}")
    print(f"  Avg total_vol/bar: {result['total_vol'].mean():.1f}")

    return result


def compute_cvd(delta_df, norm_period=400):
    """Compute Cumulative Volume Delta with session resets.

    Session boundary: 6 PM ET (18:00 ET) = CME Globex equity futures new day.
    DST-aware using zoneinfo.

    Also computes CVD_norm: rolling-max normalized to [-1,1] range,
    similar to how SM is normalized.

    Args:
        delta_df: DataFrame with 'delta' column, UTC-indexed
        norm_period: Lookback for normalization (default 400 = ~6.5 hours)

    Returns:
        delta_df with added columns: session_id, CVD, CVD_norm
    """
    df = delta_df.copy()

    # Convert timestamps to ET for session boundary detection
    et_times = df.index.tz_convert(_ET_TZ)
    et_hours = et_times.hour
    et_minutes = et_times.minute
    et_bar_mins = et_hours * 60 + et_minutes

    # Session boundary at 6 PM ET = 1080 minutes from midnight
    SESSION_BOUNDARY_ET = 18 * 60  # 1080

    # Detect session boundaries: new session when bar crosses 18:00 ET
    bar_mins_arr = np.asarray(et_bar_mins)
    prev_bar_mins = np.roll(bar_mins_arr, 1)
    prev_bar_mins[0] = 0

    # Date array for gap detection
    date_arr = np.array([d.date() for d in et_times])
    prev_date_arr = np.roll(date_arr, 1)

    # New session when:
    # 1. We cross 18:00 ET boundary (bar >= 1080, prev < 1080)
    # 2. Date gap (weekend/holiday)
    is_new_session = (
        ((bar_mins_arr >= SESSION_BOUNDARY_ET) & (prev_bar_mins < SESSION_BOUNDARY_ET))
        | (date_arr != prev_date_arr)
    )
    is_new_session[0] = True  # First bar is always session start
    session_id = np.cumsum(is_new_session)
    df['session_id'] = session_id

    # Compute CVD per session (cumulative delta, resets each session)
    df['CVD'] = df.groupby('session_id')['delta'].cumsum()

    # Compute CVD_norm: rolling-max normalization to [-1, 1]
    # Similar to SM: CVD / rolling_max(|CVD|, norm_period)
    cvd = df['CVD'].values
    n = len(cvd)
    cvd_norm = np.zeros(n)

    for i in range(n):
        s = max(0, i - norm_period + 1)
        abs_max = np.max(np.abs(cvd[s:i + 1]))
        if abs_max > 0:
            cvd_norm[i] = cvd[i] / abs_max
        else:
            cvd_norm[i] = 0.0

    df['CVD_norm'] = cvd_norm

    n_sessions = session_id.max()
    print(f"\n  Sessions detected: {n_sessions}")
    print(f"  CVD range: {df['CVD'].min():.0f} to {df['CVD'].max():.0f}")
    print(f"  CVD_norm range: {df['CVD_norm'].min():.3f} to {df['CVD_norm'].max():.3f}")

    return df


def validate_against_ohlcv(delta_df, instrument='MNQ'):
    """Compare aggregated volume against OHLCV volume for validation.

    NQ and MNQ are different contracts with different volume, so we compare
    tick counts and trends rather than exact values.
    """
    # Load OHLCV data
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from v10_test_common import load_instrument_1min

    print(f"\n  Validating against {instrument} OHLCV volume...")

    ohlcv = load_instrument_1min(instrument)
    if 'Volume' not in ohlcv.columns:
        print(f"  WARNING: {instrument} OHLCV has no Volume column. Skipping validation.")
        return

    # Find overlapping date range
    ohlcv_tz = ohlcv.index
    if ohlcv_tz.tz is None:
        ohlcv_tz = ohlcv_tz.tz_localize('UTC')

    delta_start = delta_df.index.min()
    delta_end = delta_df.index.max()

    ohlcv_overlap = ohlcv[(ohlcv_tz >= delta_start) & (ohlcv_tz <= delta_end)]
    print(f"  Overlapping bars: {len(ohlcv_overlap)}")

    if len(ohlcv_overlap) == 0:
        print("  No overlapping bars found. Skipping validation.")
        return

    # Compare daily total volumes (NQ vs MNQ are different, so compare ratios)
    delta_daily = delta_df['total_vol_all'].resample('D').sum()
    ohlcv_daily = ohlcv_overlap['Volume'].resample('D').sum()

    # Find common dates
    common_dates = delta_daily.index.intersection(ohlcv_daily.index)
    if len(common_dates) == 0:
        print("  No common dates for volume comparison.")
        return

    nq_vol = delta_daily[common_dates].values
    mnq_vol = ohlcv_daily[common_dates].values

    # NQ typically has 5-10x the volume of MNQ
    ratios = nq_vol / np.where(mnq_vol > 0, mnq_vol, 1)
    valid = (nq_vol > 0) & (mnq_vol > 0)
    if valid.sum() > 0:
        avg_ratio = np.mean(ratios[valid])
        corr = np.corrcoef(nq_vol[valid], mnq_vol[valid])[0, 1]
        print(f"  NQ/MNQ volume ratio: {avg_ratio:.1f}x (expected: 5-10x)")
        print(f"  Daily volume correlation: {corr:.3f} (should be >0.9)")
    else:
        print("  Could not compute volume ratios (zero volume days).")


def save_output(delta_df, output_path):
    """Save delta bars as CSV with Unix timestamp for pipeline compatibility."""
    out = delta_df[['buy_vol', 'sell_vol', 'delta', 'total_vol',
                     'CVD', 'CVD_norm']].copy()

    # Convert UTC timestamps to Unix seconds (matching OHLCV convention)
    out.insert(0, 'time', out.index.astype('int64') // 10**9)
    out = out.reset_index(drop=True)

    out.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print(f"  Rows: {len(out):,}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Compute volume delta bars from NQ tick data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 compute_volume_delta.py --ticks data/databento_NQ_ticks_*.parquet

  # Custom output filename
  python3 compute_volume_delta.py --ticks data/ticks.parquet --output delta_bars.csv

  # Validate against MNQ OHLCV volume
  python3 compute_volume_delta.py --ticks data/ticks.parquet --validate-against MNQ

  # Custom normalization period
  python3 compute_volume_delta.py --ticks data/ticks.parquet --norm-period 300
        """,
    )
    parser.add_argument("--ticks", required=True,
                        help="Path to tick parquet file from fetch_databento_ticks.py")
    parser.add_argument("--output", default=None,
                        help="Output CSV filename (default: auto-generated)")
    parser.add_argument("--norm-period", type=int, default=400,
                        help="CVD normalization lookback period (default: 400)")
    parser.add_argument("--validate-against", default=None,
                        choices=['MNQ', 'MES', 'NQ'],
                        help="Validate volume against OHLCV data for instrument")

    args = parser.parse_args()

    tick_path = Path(args.ticks)
    if not tick_path.exists():
        print(f"ERROR: Tick file not found: {tick_path}")
        sys.exit(1)

    print("=" * 70)
    print("VOLUME DELTA BAR COMPUTATION")
    print("=" * 70)

    # Step 1: Aggregate ticks to 1-min bars
    delta_df = aggregate_ticks_to_1min(tick_path)

    # Step 2: Compute CVD with session resets
    delta_df = compute_cvd(delta_df, norm_period=args.norm_period)

    # Step 3: Optional validation
    if args.validate_against:
        validate_against_ohlcv(delta_df, args.validate_against)

    # Step 4: Save output
    if args.output:
        output_filename = args.output
    else:
        # Derive from input filename
        stem = tick_path.stem.replace("ticks", "delta_1min")
        output_filename = f"{stem}.csv"

    output_path = DATA_DIR / output_filename
    save_output(delta_df, output_path)

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    days = delta_df.index.normalize().nunique()
    print(f"  Trading days: {days}")
    print(f"  1-min bars: {len(delta_df):,}")
    print(f"  Avg delta/bar: {delta_df['delta'].mean():.1f}")
    print(f"  Std delta/bar: {delta_df['delta'].std():.1f}")

    # Daily delta stats
    daily_delta = delta_df.groupby(delta_df.index.date)['delta'].sum()
    pos_days = (daily_delta > 0).sum()
    neg_days = (daily_delta < 0).sum()
    print(f"  Net positive delta days: {pos_days}/{len(daily_delta)}")
    print(f"  Net negative delta days: {neg_days}/{len(daily_delta)}")

    # CVD reset verification
    sessions = delta_df['session_id'].nunique()
    print(f"  CVD sessions (resets): {sessions}")

    # Spot check: print a few bars
    print(f"\n  Sample bars (first 5):")
    sample = delta_df.head(5)[['buy_vol', 'sell_vol', 'delta', 'CVD', 'CVD_norm']]
    print(sample.to_string(index=True))

    print("\nDone!")
    print("\nNext steps:")
    print(f"  python3 cvd_analysis.py --delta {output_path}")


if __name__ == "__main__":
    main()
