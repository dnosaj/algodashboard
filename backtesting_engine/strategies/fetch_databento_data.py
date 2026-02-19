"""
Fetch Extended MNQ History from Databento
==========================================
Downloads 3-6 months of MNQ 1-min OHLCV data using Databento's free $125 credits.

Setup:
1. Sign up at https://databento.com (free $125 credits)
2. Get your API key from the dashboard
3. Set environment variable: export DATABENTO_API_KEY="your-key-here"
4. Install: pip install databento
5. Run: python3 fetch_databento_data.py

The downloaded data will be saved to backtesting_engine/data/ in a format
compatible with the existing v10 test suite.

NOTE: This data will have OHLCV but NO AlgoAlpha Smart Money columns.
Price-based features (OR alignment, VWAP, prior day levels, underwater exit,
price structure exit) can still be tested on this extended data.
SM-dependent features are limited to the AlgoAlpha data window.

Cost guidance (approximate):
  - 1 month of MNQ 1-min data: ~$2-5
  - 3 months: ~$5-15
  - 6 months: ~$10-30
  These are rough estimates; use --dry-run to check actual cost before downloading.

File 6 of the v10 feature validation suite.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Columns that match the existing pipeline format
# The existing CSVs have: time, open, high, low, close, [SM columns...]
# We output: time, open, high, low, close, Volume, VWAP, SM Net Index
# SM columns are empty since Databento does not provide AlgoAlpha data.
OUTPUT_COLUMNS = [
    "time", "open", "high", "low", "close",
    "Volume", "VWAP", "SM Net Index",
]


def detect_fixed_point_prices(prices):
    """Detect whether Databento prices are in fixed-point format (scaled by 1e9).

    Databento may return futures prices in fixed-point integer format where
    prices are multiplied by 1e9 (e.g., 24650.25 becomes 24650250000000).
    This function checks the magnitude and returns the appropriate divisor.

    Returns:
        divisor (float): 1e9 if fixed-point detected, 1.0 if already normal.
    """
    if len(prices) == 0:
        return 1.0

    sample = prices[:100]  # Check first 100 bars
    median_price = sorted(sample)[len(sample) // 2]

    if median_price > 1e6:
        # Prices are in fixed-point format (scaled by 1e9)
        # Normal MNQ prices are 10,000-30,000 range
        # Fixed-point would be 10,000,000,000,000+ range
        return 1e9
    else:
        # Prices are already in normal format
        return 1.0


def fetch_futures_data(api_key, start_date, end_date, symbol="MNQ",
                       output_filename=None, dry_run=False):
    """Download futures 1-min OHLCV from Databento.

    Args:
        api_key: Databento API key
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        symbol: Instrument name (MNQ, MES, ES, NQ, MYM). Default: MNQ
        output_filename: Optional output filename (default: auto-generated)
        dry_run: If True, only show cost estimate without downloading

    Returns:
        Path to the output CSV file, or None if aborted/dry-run.
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
        print("ERROR: pandas package not installed.")
        print("  Install with: pip install pandas")
        sys.exit(1)

    print("Connecting to Databento...")
    client = db.Historical(api_key)

    # CME Globex Market Data Platform
    # Schema: ohlcv-1m (1-minute OHLCV bars)
    # Continuous front-month symbols: MNQ.c.0, MES.c.0, ES.c.0, NQ.c.0, MYM.c.0
    SYMBOL_MAP = {
        "MNQ": "MNQ.c.0",   # Micro Nasdaq
        "MES": "MES.c.0",   # Micro S&P 500
        "ES":  "ES.c.0",    # E-mini S&P 500
        "NQ":  "NQ.c.0",    # E-mini Nasdaq
        "MYM": "MYM.c.0",   # Micro Dow
    }

    symbol_upper = symbol.upper()
    if symbol_upper not in SYMBOL_MAP:
        print(f"ERROR: Unknown symbol '{symbol}'. Supported: {', '.join(SYMBOL_MAP.keys())}")
        sys.exit(1)

    databento_symbol = SYMBOL_MAP[symbol_upper]
    dataset = "GLBX.MDP3"
    symbols = [databento_symbol]
    schema = "ohlcv-1m"

    print(f"Requesting {symbol_upper} 1-min data: {start_date} to {end_date}")
    print(f"  Dataset: {dataset}")
    print(f"  Schema:  {schema}")
    print(f"  Symbol:  {databento_symbol}")

    # ---- Cost estimate ----
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

        if cost > 50:
            print("  WARNING: Cost exceeds $50 safety threshold.")
            print("  Reduce date range or confirm manually.")
            response = input("  Continue? (y/n): ").strip().lower()
            if response != "y":
                print("  Aborted.")
                return None
    except Exception as e:
        print(f"  Could not get cost estimate: {e}")
        if dry_run:
            print("  Cannot proceed with --dry-run without cost estimate.")
            return None
        print("  Proceeding anyway (download may still fail if credits are insufficient)...")

    if dry_run:
        print("\n  --dry-run flag set. Stopping before download.")
        return None

    # ---- Download data ----
    print("\nDownloading data...")
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
    print(f"  Downloaded {len(df)} bars")

    if len(df) == 0:
        print("  No data returned. Check date range and symbol.")
        print("  Common issues:")
        print("    - Weekend/holiday date range with no trading")
        print("    - Date range outside available history")
        print("    - Symbol not found (try NQ.c.0 for full-size)")
        return None

    # ---- Price format detection ----
    # Databento may use fixed-point pricing (1e9 scale) for some datasets.
    # Detect this and convert if needed.
    raw_open = df["open"].values
    divisor = detect_fixed_point_prices(raw_open)

    if divisor > 1.0:
        print(f"  Detected fixed-point pricing (divisor: {divisor:.0e})")
    else:
        print("  Prices are in normal format (no conversion needed)")

    # ---- Build output DataFrame ----
    result = pd.DataFrame()

    # Timestamp: convert index (nanosecond datetime) to Unix seconds
    result["time"] = df.index.astype("int64") // 10**9

    # OHLCV with price format handling
    result["open"] = df["open"].values / divisor
    result["high"] = df["high"].values / divisor
    result["low"] = df["low"].values / divisor
    result["close"] = df["close"].values / divisor
    result["Volume"] = df["volume"].values

    # Sanity check: verify prices are in expected range
    # MNQ/NQ: ~10,000-30,000; MES/ES: ~3,000-7,000; MYM: ~30,000-45,000
    median_close = result["close"].median()
    if median_close < 100 or median_close > 100000:
        print(f"  WARNING: Median close price is {median_close:.2f}")
        print(f"  Data may need manual inspection.")

    # Placeholder columns for pipeline compatibility
    # VWAP: empty by default, use --compute-vwap to fill
    # SM Net Index: not available from Databento (requires AlgoAlpha indicator)
    result["VWAP"] = ""
    result["SM Net Index"] = ""

    # ---- Save to CSV ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"databento_{symbol_upper}_1min_{start_date}_to_{end_date}.csv"

    output_path = DATA_DIR / output_filename
    result.to_csv(output_path, index=False)

    print(f"\n  Saved to: {output_path}")
    print(f"  Bars:       {len(result)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Price range: {result['close'].min():.2f} to {result['close'].max():.2f}")
    print(f"  Avg volume/bar: {result['Volume'].mean():.1f}")

    return output_path


def compute_vwap_from_ohlcv(filepath):
    """Post-process: compute VWAP from OHLCV data and update the CSV in place.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Resets at each session boundary.

    Session boundary: 23:00 UTC (= 6:00 PM Eastern), which is when CME Globex
    starts a new trading day for equity index futures.

    Args:
        filepath: Path to the CSV file to update.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(filepath)

    if "time" not in df.columns or "Volume" not in df.columns:
        print(f"  ERROR: CSV missing required columns (time, Volume)")
        return

    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    # Identify session boundaries
    # CME Globex equity futures session starts at 23:00 UTC (6pm ET)
    # Each bar at or after 23:00 UTC starts a new session day
    # A bar at 22:59 UTC belongs to the current session; 23:00 UTC starts the next
    session_date = df["datetime"].dt.date
    # Shift: bars from 23:00 UTC onward belong to "next day" session
    # But simpler approach: detect 23:00 UTC crossings
    hour = df["datetime"].dt.hour
    minute = df["datetime"].dt.minute
    bar_minutes = hour * 60 + minute  # minutes since midnight UTC

    # Session boundary at 23:00 UTC = 1380 minutes
    SESSION_START_MINS = 23 * 60  # 1380

    # Assign session IDs: increment when we cross 23:00 UTC
    # A bar is "new session" if bar_minutes >= 1380 and previous bar < 1380,
    # OR if there's a date gap (weekend/holiday)
    is_new_session = (
        ((bar_minutes >= SESSION_START_MINS)
         & (bar_minutes.shift(1).fillna(0) < SESSION_START_MINS))
        | (session_date != session_date.shift(1))
    )
    # First bar is always a session start
    is_new_session.iloc[0] = True
    session_id = is_new_session.cumsum()

    # Compute VWAP per session
    tpv = typical * volume
    cum_tpv = tpv.groupby(session_id).cumsum()
    cum_vol = volume.groupby(session_id).cumsum()
    vwap = np.where(cum_vol > 0, cum_tpv / cum_vol, typical)

    df["VWAP"] = np.round(vwap, 6)

    # Drop the temporary datetime column before saving
    df.drop(columns=["datetime"], inplace=True)
    df.to_csv(filepath, index=False)

    num_sessions = session_id.nunique()
    print(f"  VWAP computed and saved to {filepath}")
    print(f"  Sessions detected: {num_sessions}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch MNQ 1-min OHLCV data from Databento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download MNQ last 6 months (default)
  python3 fetch_databento_data.py --months 6

  # Download MES last 6 months
  python3 fetch_databento_data.py --symbol MES --months 6

  # Download both MNQ and MES (run twice)
  python3 fetch_databento_data.py --symbol MNQ --months 6 --compute-vwap
  python3 fetch_databento_data.py --symbol MES --months 6 --compute-vwap

  # Dry run (check cost only, no download)
  python3 fetch_databento_data.py --symbol MNQ --months 6 --dry-run

  # Download specific date range
  python3 fetch_databento_data.py --start 2025-09-01 --end 2026-02-13

  # Just compute VWAP on an existing file
  python3 fetch_databento_data.py --compute-vwap-only path/to/file.csv
        """,
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date (YYYY-MM-DD). Default: computed from --months",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (YYYY-MM-DD). Default: yesterday",
    )
    parser.add_argument(
        "--months", type=int, default=6,
        help="Months of history to download (used if --start not set). Default: 6",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output filename (saved in backtesting_engine/data/)",
    )
    parser.add_argument(
        "--compute-vwap", action="store_true",
        help="Compute VWAP after downloading data",
    )
    parser.add_argument(
        "--compute-vwap-only", default=None, metavar="CSV_PATH",
        help="Only compute VWAP on an existing CSV file (no download)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show cost estimate only, do not download",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Databento API key (or set DATABENTO_API_KEY env var)",
    )
    parser.add_argument(
        "--symbol", default="MNQ",
        help="Instrument to download (MNQ, MES, ES, NQ, MYM). Default: MNQ",
    )
    args = parser.parse_args()

    # Handle --compute-vwap-only mode (no download, no API key needed)
    if args.compute_vwap_only:
        csv_path = Path(args.compute_vwap_only)
        if not csv_path.exists():
            print(f"ERROR: File not found: {csv_path}")
            sys.exit(1)
        print(f"Computing VWAP for: {csv_path}")
        compute_vwap_from_ohlcv(str(csv_path))
        print("Done!")
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

    # Set date range
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

    instrument = args.symbol.upper()

    print("=" * 70)
    print(f"DATABENTO {instrument} DATA DOWNLOADER")
    print("=" * 70)
    print(f"  Instrument:  {instrument}")
    print(f"  Date range:  {start_date} to {end_date}")
    print(f"  Output dir:  {DATA_DIR}")
    if args.dry_run:
        print(f"  Mode:        DRY RUN (cost estimate only)")
    print()

    output_path = fetch_futures_data(
        api_key, start_date, end_date,
        symbol=instrument,
        output_filename=args.output,
        dry_run=args.dry_run,
    )

    if output_path and args.compute_vwap:
        print("\nComputing VWAP...")
        compute_vwap_from_ohlcv(str(output_path))

    print("\nDone!")
    print("\nNext steps:")
    print("  1. The CSV is ready for price-based feature testing")
    print("  2. Load with pd.read_csv() and use with v10_test_common functions")
    print("  3. SM-dependent features need AlgoAlpha data (not available from Databento)")
    print("  4. Use --compute-vwap to add session VWAP column")


if __name__ == "__main__":
    main()
