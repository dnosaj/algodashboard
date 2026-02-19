"""
Fetch NQ Tick Data from Databento
==================================
Downloads NQ (full-size Nasdaq futures) tick-level trade data for volume delta
and CVD (Cumulative Volume Delta) analysis.

WHY NQ not MNQ: NQ has far more institutional volume -- the "smart money" signal
we're looking for. NQ and MNQ track the same index with identical price movement,
so NQ delta applies directly to MNQ backtests.

Each tick has a `side` field from CME's native aggressor tag:
  'A' = sell aggressor (-delta)
  'B' = buy aggressor (+delta)
  'N' = undefined (exclude from delta)

Setup:
  export DATABENTO_API_KEY="your-key-here"
  pip install databento pyarrow

Usage:
  # Pilot: check cost for 1 week
  python3 fetch_databento_ticks.py --start 2026-02-03 --end 2026-02-08 --dry-run

  # Pilot: download 1 week to validate side field
  python3 fetch_databento_ticks.py --start 2026-02-03 --end 2026-02-08

  # Full 6 months (train + test)
  python3 fetch_databento_ticks.py --start 2025-08-17 --end 2026-02-13

  # Check side field coverage on downloaded data
  python3 fetch_databento_ticks.py --validate path/to/file.parquet
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def detect_fixed_point_prices(prices):
    """Detect whether Databento prices are in fixed-point format (scaled by 1e9)."""
    if len(prices) == 0:
        return 1.0
    sample = prices[:1000]
    median_price = sorted(sample)[len(sample) // 2]
    if median_price > 1e6:
        return 1e9
    return 1.0


def validate_parquet(filepath):
    """Validate a downloaded tick parquet file for side field coverage."""
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. pip install pandas")
        sys.exit(1)

    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"Validating: {path}")
    df = pd.read_parquet(path)
    print(f"  Total ticks: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    if 'side' not in df.columns:
        print("  ERROR: No 'side' column found!")
        print(f"  Available columns: {list(df.columns)}")
        return

    # Side distribution
    side_counts = df['side'].value_counts()
    print(f"\n  Side distribution:")
    for side, count in side_counts.items():
        pct = count / len(df) * 100
        print(f"    '{side}': {count:>12,} ({pct:.1f}%)")

    # Coverage: % with B or A (usable for delta)
    usable = df['side'].isin(['B', 'A']).sum()
    usable_pct = usable / len(df) * 100
    print(f"\n  Usable (B+A): {usable:,} ({usable_pct:.1f}%)")
    if usable_pct < 90:
        print("  WARNING: Less than 90% of ticks have side info.")
        print("  Delta signal may be unreliable.")
    else:
        print("  GOOD: >90% side coverage.")

    # Date range
    if 'ts_event' in df.columns:
        ts_col = 'ts_event'
    elif 'time' in df.columns:
        ts_col = 'time'
    else:
        ts_col = df.columns[0]

    times = pd.to_datetime(df[ts_col])
    print(f"\n  Date range: {times.min()} to {times.max()}")

    # Daily tick counts
    daily = times.dt.date.value_counts().sort_index()
    print(f"  Trading days: {len(daily)}")
    print(f"  Avg ticks/day: {daily.mean():,.0f}")
    print(f"  Min ticks/day: {daily.min():,} ({daily.idxmin()})")
    print(f"  Max ticks/day: {daily.max():,} ({daily.idxmax()})")

    # Price sanity
    if 'price' in df.columns:
        prices = df['price']
        divisor = detect_fixed_point_prices(prices.values)
        if divisor > 1:
            prices = prices / divisor
        print(f"\n  Price range: {prices.min():.2f} to {prices.max():.2f}")
        print(f"  Median price: {prices.median():.2f}")

    # Size stats
    if 'size' in df.columns:
        print(f"\n  Size stats:")
        print(f"    Mean: {df['size'].mean():.1f}")
        print(f"    Median: {df['size'].median():.0f}")
        print(f"    Max: {df['size'].max():,}")
        print(f"    Total volume: {df['size'].sum():,}")


def fetch_nq_ticks(api_key, start_date, end_date, output_filename=None,
                   dry_run=False):
    """Download NQ tick data from Databento.

    Args:
        api_key: Databento API key
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_filename: Optional output filename (saved as Parquet)
        dry_run: If True, only show cost estimate

    Returns:
        Path to output Parquet file, or None if aborted/dry-run.
    """
    try:
        import databento as db
    except ImportError:
        print("ERROR: databento package not installed.")
        print("  Install with: pip install databento")
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. pip install pandas")
        sys.exit(1)

    print("Connecting to Databento...")
    client = db.Historical(api_key)

    dataset = "GLBX.MDP3"
    symbols = ["NQ.c.0"]
    schema = "trades"

    print(f"Requesting NQ tick data: {start_date} to {end_date}")
    print(f"  Dataset: {dataset}")
    print(f"  Schema:  {schema}")
    print(f"  Symbol:  NQ.c.0 (full-size Nasdaq futures)")

    # Cost estimate
    print("\nGetting cost estimate...")
    try:
        cost = client.metadata.get_cost(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            stype_in="continuous",
            start=start_date,
            end=end_date,
        )
        print(f"  Estimated cost: ${cost:.2f}")

        if cost > 150:
            print("  WARNING: Cost exceeds $150 safety threshold.")
            response = input("  Continue? (y/n): ").strip().lower()
            if response != "y":
                print("  Aborted.")
                return None
    except Exception as e:
        print(f"  Could not get cost estimate: {e}")
        if dry_run:
            print("  Cannot proceed with --dry-run without cost estimate.")
            return None
        print("  Proceeding anyway...")

    if dry_run:
        print("\n  --dry-run flag set. Stopping before download.")
        return None

    # Download
    print("\nDownloading tick data (this may take a while for large ranges)...")
    try:
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            stype_in="continuous",
            start=start_date,
            end=end_date,
        )
    except Exception as e:
        print(f"ERROR downloading data: {e}")
        sys.exit(1)

    # Convert to DataFrame
    df = data.to_df()
    print(f"  Downloaded {len(df):,} ticks")

    if len(df) == 0:
        print("  No data returned. Check date range.")
        return None

    # Build output DataFrame
    result = pd.DataFrame()

    # Timestamp: nanosecond precision from index
    if hasattr(df.index, 'name') and df.index.name == 'ts_event':
        result['ts_event'] = df.index.values
    elif 'ts_event' in df.columns:
        result['ts_event'] = df['ts_event'].values
    else:
        result['ts_event'] = df.index.values

    # Price with fixed-point detection
    raw_prices = df['price'].values
    divisor = detect_fixed_point_prices(raw_prices)
    if divisor > 1.0:
        print(f"  Detected fixed-point pricing (divisor: {divisor:.0e})")
    result['price'] = raw_prices / divisor

    # Size (number of contracts)
    result['size'] = df['size'].values

    # Side field -- the key data we need
    if 'side' in df.columns:
        result['side'] = df['side'].values
        print(f"  Side field found!")
    else:
        # Try alternative column names
        for col in ['aggressor_side', 'action', 'flags']:
            if col in df.columns:
                result['side'] = df[col].values
                print(f"  Using '{col}' as side field")
                break
        else:
            print("  WARNING: No side field found in columns:")
            print(f"    {list(df.columns)}")
            result['side'] = 'N'

    # Save as Parquet (tick data = millions of rows, CSV impractical)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"databento_NQ_ticks_{start_date}_to_{end_date}.parquet"

    output_path = DATA_DIR / output_filename
    result.to_parquet(output_path, index=False, engine='pyarrow')

    print(f"\n  Saved to: {output_path}")
    print(f"  Ticks:      {len(result):,}")
    print(f"  File size:  {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Price range: {result['price'].min():.2f} to {result['price'].max():.2f}")

    # Quick side coverage check
    if 'side' in result.columns:
        side_counts = result['side'].value_counts()
        print(f"\n  Side distribution:")
        for side, count in side_counts.items():
            pct = count / len(result) * 100
            print(f"    '{side}': {count:>12,} ({pct:.1f}%)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NQ tick data from Databento for volume delta analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check cost for 1-week pilot
  python3 fetch_databento_ticks.py --start 2026-02-03 --end 2026-02-08 --dry-run

  # Download 1-week pilot
  python3 fetch_databento_ticks.py --start 2026-02-03 --end 2026-02-08

  # Full 6 months (train + test)
  python3 fetch_databento_ticks.py --start 2025-08-17 --end 2026-02-13

  # Validate downloaded file
  python3 fetch_databento_ticks.py --validate data/databento_NQ_ticks_*.parquet
        """,
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, default=6,
                        help="Months of history (if --start not set). Default: 6")
    parser.add_argument("--output", default=None, help="Output filename")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cost estimate only")
    parser.add_argument("--api-key", default=None,
                        help="Databento API key (or set DATABENTO_API_KEY env var)")
    parser.add_argument("--validate", default=None, metavar="PARQUET_PATH",
                        help="Validate an existing parquet file (no download)")

    args = parser.parse_args()

    # Validate mode
    if args.validate:
        validate_parquet(args.validate)
        return

    # Get API key
    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: No API key provided.")
        print("  Set DATABENTO_API_KEY environment variable or use --api-key flag")
        print("")
        print("  To get a free API key with $125 credits:")
        print("  1. Go to https://databento.com")
        print("  2. Sign up for a free account")
        print("  3. Navigate to your API keys page")
        print("  4. Create a new key and copy it")
        print("  5. export DATABENTO_API_KEY='your-key-here'")
        sys.exit(1)

    # Date range
    if args.end:
        end_date = args.end
    else:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if args.start:
        start_date = args.start
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=args.months * 30)
        start_date = start_dt.strftime("%Y-%m-%d")

    print("=" * 70)
    print("DATABENTO NQ TICK DATA DOWNLOADER")
    print("=" * 70)
    print(f"  Symbol:      NQ.c.0 (full-size Nasdaq)")
    print(f"  Schema:      trades (tick-level)")
    print(f"  Date range:  {start_date} to {end_date}")
    print(f"  Output dir:  {DATA_DIR}")
    if args.dry_run:
        print(f"  Mode:        DRY RUN (cost estimate only)")
    print()

    output_path = fetch_nq_ticks(
        api_key, start_date, end_date,
        output_filename=args.output,
        dry_run=args.dry_run,
    )

    if output_path:
        print("\nNext steps:")
        print("  1. Validate side coverage:")
        print(f"     python3 fetch_databento_ticks.py --validate {output_path}")
        print("  2. Compute volume delta bars:")
        print(f"     python3 compute_volume_delta.py --ticks {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
