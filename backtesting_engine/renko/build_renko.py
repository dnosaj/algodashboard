"""
Build Traditional Renko Bars from NQ Tick Data
===============================================
Reads NQ tick parquet, constructs traditional Renko bricks with a
configurable box size, and saves as parquet files compatible with
the backtesting engine architecture.

Traditional Renko rules:
  - New UP brick when price reaches current brick top + box_size
  - New DOWN brick when price reaches current brick bottom - box_size
  - Reversal requires 2x box_size move from the opposite edge
  - Each brick has OHLC, volume, tick count, timestamps

Output format matches the engine's expected DataFrame:
  Index: Time (datetime, brick completion time)
  Columns: Open, High, Low, Close, Volume, Direction (+1/-1)

Usage:
  python3 build_renko.py                          # defaults: 5pt box, full 6 months
  python3 build_renko.py --box-size 10            # 10pt boxes
  python3 build_renko.py --box-size 5 --stats     # show stats without saving
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICK_FILE = DATA_DIR / "databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet"

# Train/test split matching the project convention
TRAIN_END = pd.Timestamp("2025-11-17")  # Train: Aug 17 - Nov 16
TEST_START = pd.Timestamp("2025-11-17")  # Test: Nov 17 - Feb 12


def build_renko(prices, timestamps, sizes, sides, box_size=5.0):
    """Build traditional Renko bricks from tick-level data.

    Parameters
    ----------
    prices : np.ndarray of float
        Tick prices.
    timestamps : np.ndarray of datetime64[ns]
        Tick timestamps.
    sizes : np.ndarray of int
        Tick sizes (contracts).
    sides : np.ndarray of str
        Tick aggressor side ('B', 'A', 'N').
    box_size : float
        Renko box size in price points.

    Returns
    -------
    list of dict
        Each dict is one completed Renko brick with keys:
        open, high, low, close, volume, buy_vol, sell_vol,
        tick_count, direction (+1 or -1),
        start_time, end_time (completion timestamp)
    """
    if len(prices) == 0:
        return []

    # Snap the first price to the nearest box boundary
    first_price = prices[0]
    brick_bottom = np.floor(first_price / box_size) * box_size
    brick_top = brick_bottom + box_size

    bricks = []
    # Accumulate stats for the current forming brick
    cur_volume = 0
    cur_buy_vol = 0
    cur_sell_vol = 0
    cur_tick_count = 0
    cur_high = first_price
    cur_low = first_price
    cur_start_time = timestamps[0]
    last_direction = 0  # 0 = no bricks yet

    for i in range(len(prices)):
        p = prices[i]
        sz = sizes[i]
        side = sides[i]

        cur_tick_count += 1
        cur_volume += sz
        if side == 'B':
            cur_buy_vol += sz
        elif side == 'A':
            cur_sell_vol += sz
        cur_high = max(cur_high, p)
        cur_low = min(cur_low, p)

        # Check for new bricks (can generate multiple in one tick on gaps)
        while True:
            if p >= brick_top + box_size:
                # New UP brick
                direction = 1
                if last_direction == -1:
                    # Reversal from down: need 2x box from bottom
                    # The new brick goes from brick_bottom to brick_bottom + box_size
                    # But we already checked p >= brick_top + box_size
                    # For traditional Renko, reversal creates a brick at the opposite edge
                    pass

                brick_open = brick_top if last_direction >= 0 else brick_bottom
                brick_close = brick_open + box_size

                bricks.append({
                    'open': brick_open,
                    'high': max(cur_high, brick_close),
                    'low': min(cur_low, brick_open),
                    'close': brick_close,
                    'volume': cur_volume,
                    'buy_vol': cur_buy_vol,
                    'sell_vol': cur_sell_vol,
                    'tick_count': cur_tick_count,
                    'direction': direction,
                    'start_time': cur_start_time,
                    'end_time': timestamps[i],
                })

                # Update boundaries
                brick_bottom = brick_close - box_size
                brick_top = brick_close + box_size
                last_direction = 1

                # Reset accumulators for next brick
                cur_volume = 0
                cur_buy_vol = 0
                cur_sell_vol = 0
                cur_tick_count = 0
                cur_high = p
                cur_low = p
                cur_start_time = timestamps[i]

            elif p <= brick_bottom - box_size:
                # New DOWN brick
                direction = -1

                brick_open = brick_bottom if last_direction <= 0 else brick_top
                brick_close = brick_open - box_size

                bricks.append({
                    'open': brick_open,
                    'high': max(cur_high, brick_open),
                    'low': min(cur_low, brick_close),
                    'close': brick_close,
                    'volume': cur_volume,
                    'buy_vol': cur_buy_vol,
                    'sell_vol': cur_sell_vol,
                    'tick_count': cur_tick_count,
                    'direction': direction,
                    'start_time': cur_start_time,
                    'end_time': timestamps[i],
                })

                # Update boundaries
                brick_top = brick_close + box_size
                brick_bottom = brick_close - box_size
                last_direction = -1

                # Reset accumulators
                cur_volume = 0
                cur_buy_vol = 0
                cur_sell_vol = 0
                cur_tick_count = 0
                cur_high = p
                cur_low = p
                cur_start_time = timestamps[i]

            else:
                break  # No more bricks from this tick

    return bricks


def bricks_to_dataframe(bricks):
    """Convert brick list to a DataFrame matching engine conventions.

    Index: Time (brick completion timestamp)
    Columns: Open, High, Low, Close, Volume, BuyVol, SellVol,
             TickCount, Direction
    """
    if not bricks:
        return pd.DataFrame()

    df = pd.DataFrame(bricks)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['end_time'])
    result['Open'] = df['open']
    result['High'] = df['high']
    result['Low'] = df['low']
    result['Close'] = df['close']
    result['Volume'] = df['volume']
    result['BuyVol'] = df['buy_vol']
    result['SellVol'] = df['sell_vol']
    result['TickCount'] = df['tick_count']
    result['Direction'] = df['direction']
    result['StartTime'] = pd.to_datetime(df['start_time'])
    result = result.set_index('Time')
    return result


def main():
    parser = argparse.ArgumentParser(description="Build Renko bars from NQ tick data")
    parser.add_argument('--box-size', type=float, default=5.0,
                        help='Renko box size in points (default: 5)')
    parser.add_argument('--stats', action='store_true',
                        help='Print stats only, do not save files')
    args = parser.parse_args()

    box = args.box_size
    print("=" * 90)
    print(f"  BUILD RENKO: {box:.0f}-pt Traditional Renko from NQ Ticks")
    print("=" * 90)

    # Load ticks
    print(f"\n  Loading {TICK_FILE.name}...")
    df_ticks = pd.read_parquet(TICK_FILE)
    print(f"  {len(df_ticks):,} ticks, "
          f"{df_ticks['ts_event'].iloc[0]} to {df_ticks['ts_event'].iloc[-1]}")

    prices = df_ticks['price'].values
    timestamps = df_ticks['ts_event'].values
    sizes = df_ticks['size'].values
    sides = df_ticks['side'].values

    # Build Renko
    print(f"\n  Building {box:.0f}-pt Renko bricks...")
    bricks = build_renko(prices, timestamps, sizes, sides, box_size=box)
    print(f"  {len(bricks):,} bricks completed")

    if not bricks:
        print("  No bricks generated. Check box size.")
        return

    df_renko = bricks_to_dataframe(bricks)

    # Stats
    up_bricks = (df_renko['Direction'] == 1).sum()
    dn_bricks = (df_renko['Direction'] == -1).sum()
    avg_vol = df_renko['Volume'].mean()
    avg_ticks = df_renko['TickCount'].mean()

    # Duration stats
    durations = pd.Series(df_renko.index - df_renko['StartTime']).dt.total_seconds().values
    avg_dur = np.mean(durations)
    med_dur = np.median(durations)
    p90_dur = np.percentile(durations, 90)

    print(f"\n  --- FULL DATASET STATS ---")
    print(f"  Total bricks:     {len(df_renko):,}")
    print(f"  UP bricks:        {up_bricks:,} ({up_bricks/len(df_renko)*100:.1f}%)")
    print(f"  DOWN bricks:      {dn_bricks:,} ({dn_bricks/len(df_renko)*100:.1f}%)")
    print(f"  Date range:       {df_renko.index[0]} to {df_renko.index[-1]}")
    print(f"  Avg volume/brick: {avg_vol:,.0f} contracts")
    print(f"  Avg ticks/brick:  {avg_ticks:,.0f}")
    print(f"  Avg duration:     {avg_dur:.0f}s ({avg_dur/60:.1f}min)")
    print(f"  Median duration:  {med_dur:.0f}s ({med_dur/60:.1f}min)")
    print(f"  P90 duration:     {p90_dur:.0f}s ({p90_dur/60:.1f}min)")

    # Bricks per day
    df_renko['_date'] = df_renko.index.date
    bricks_per_day = df_renko.groupby('_date').size()
    print(f"  Avg bricks/day:   {bricks_per_day.mean():.1f}")
    print(f"  Min bricks/day:   {bricks_per_day.min()}")
    print(f"  Max bricks/day:   {bricks_per_day.max()}")
    df_renko.drop(columns=['_date'], inplace=True)

    # Monthly breakdown
    print(f"\n  --- MONTHLY BREAKDOWN ---")
    df_renko['_month'] = df_renko.index.to_period('M')
    monthly = df_renko.groupby('_month').agg(
        bricks=('Direction', 'count'),
        up=('Direction', lambda x: (x == 1).sum()),
        down=('Direction', lambda x: (x == -1).sum()),
        vol=('Volume', 'sum'),
    )
    print(f"  {'Month':>10}  {'Bricks':>7}  {'UP':>5}  {'DOWN':>5}  {'Volume':>12}")
    print(f"  {'-' * 45}")
    for m, row in monthly.iterrows():
        print(f"  {str(m):>10}  {row['bricks']:>7}  {row['up']:>5}  {row['down']:>5}  {row['vol']:>12,.0f}")
    df_renko.drop(columns=['_month'], inplace=True)

    if args.stats:
        print("\n  --stats flag set, skipping file save.")
        return

    # Split into train/test
    df_train = df_renko[df_renko.index < TRAIN_END]
    df_test = df_renko[df_renko.index >= TEST_START]

    print(f"\n  --- TRAIN/TEST SPLIT ---")
    print(f"  Train: {len(df_train):,} bricks  ({df_train.index[0].date()} to {df_train.index[-1].date()})")
    print(f"  Test:  {len(df_test):,} bricks  ({df_test.index[0].date()} to {df_test.index[-1].date()})")

    # Save files
    box_str = f"{int(box)}pt" if box == int(box) else f"{box}pt"

    full_path = DATA_DIR / f"renko_NQ_{box_str}_full.parquet"
    train_path = DATA_DIR / f"renko_NQ_{box_str}_train.parquet"
    test_path = DATA_DIR / f"renko_NQ_{box_str}_test.parquet"

    df_renko.to_parquet(full_path)
    df_train.to_parquet(train_path)
    df_test.to_parquet(test_path)

    print(f"\n  --- SAVED ---")
    print(f"  Full:  {full_path.name}  ({len(df_renko):,} bricks)")
    print(f"  Train: {train_path.name}  ({len(df_train):,} bricks)")
    print(f"  Test:  {test_path.name}  ({len(df_test):,} bricks)")

    # Show how to load in a future strategy script
    print(f"""
  --- USAGE IN FUTURE STRATEGY ---

  import pandas as pd

  # Load train or test split
  df = pd.read_parquet('data/renko_NQ_{box_str}_train.parquet')

  # Standard engine columns available:
  #   df['Open'], df['High'], df['Low'], df['Close']  -- brick OHLC
  #   df['Volume']      -- total contracts during brick
  #   df['BuyVol']      -- buy aggressor volume
  #   df['SellVol']     -- sell aggressor volume
  #   df['TickCount']   -- number of trades during brick
  #   df['Direction']   -- +1 (up) or -1 (down)
  #   df['StartTime']   -- when the brick started forming
  #   df.index          -- Time (when the brick completed)

  # Compute SM on Renko closes + volume:
  #   sm = compute_smart_money(df['Close'].values, df['Volume'].values, ...)
  #   df['SM_Net'] = sm
    """)


if __name__ == "__main__":
    main()
