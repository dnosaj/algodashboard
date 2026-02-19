# Renko Bar Construction

## Overview

Traditional Renko bars built from NQ tick-level data (42.6M ticks from Databento).
Renko bars are price-based, not time-based -- a new brick forms only when price moves
a fixed number of points (the "box size") from the previous brick's close.

## Source Data

- **File**: `data/databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet`
- **Instrument**: NQ (full-size Nasdaq futures, not MNQ -- NQ has more institutional volume)
- **Fields**: `ts_event` (nanosecond), `price`, `size` (contracts), `side` (B/A/N)
- **Coverage**: Aug 17, 2025 to Feb 12, 2026 (42,590,239 ticks)
- **Cost**: $53.31 from Databento (fetched via `strategies/fetch_databento_ticks.py`)

NQ and MNQ track the same index with identical price movement, so NQ Renko bars
apply directly to MNQ strategy development.

## Generated Files (5pt Box)

| File | Bricks | Period | Purpose |
|------|--------|--------|---------|
| `data/renko_NQ_5pt_full.parquet` | 129,416 | Aug 17 - Feb 12 | Full dataset |
| `data/renko_NQ_5pt_train.parquet` | 52,757 | Aug 17 - Nov 16 | Train split |
| `data/renko_NQ_5pt_test.parquet` | 76,659 | Nov 17 - Feb 12 | Test split (held out) |

Train/test split at Nov 17 matches the project convention used in v14/v15 testing.

## Data Schema

Each parquet file is a pandas DataFrame with DatetimeIndex:

| Column | Type | Description |
|--------|------|-------------|
| **Index (Time)** | datetime64[ns] | When the brick completed |
| Open | float | Brick open price |
| High | float | Highest tick price during brick formation |
| Low | float | Lowest tick price during brick formation |
| Close | float | Brick close price (= Open +/- box_size) |
| Volume | int | Total contracts traded during brick |
| BuyVol | int | Buy aggressor volume (side='B') |
| SellVol | int | Sell aggressor volume (side='A') |
| TickCount | int | Number of individual trades during brick |
| Direction | int | +1 (up brick) or -1 (down brick) |
| StartTime | datetime64[ns] | When the brick started forming |

## Traditional Renko Rules

1. Starting price snapped to nearest box_size boundary
2. **New UP brick**: price reaches current brick top + box_size
3. **New DOWN brick**: price reaches current brick bottom - box_size
4. **Reversal**: requires 2x box_size move from the opposite edge
5. Multiple bricks can form from a single tick (gap handling)
6. Each brick accumulates volume, buy/sell split, and tick count from all ticks during its formation

## 5pt Box Statistics

```
Total bricks:      129,416
UP / DOWN:         64,759 / 64,657 (50/50)
Avg bricks/day:    835
Median duration:   9 seconds (most bricks complete fast)
P90 duration:      3 minutes (quiet periods)
Avg volume/brick:  493 contracts
Avg ticks/brick:   329
```

Monthly breakdown:
```
2025-08:  6,001 bricks
2025-09: 10,166 bricks
2025-10: 23,060 bricks
2025-11: 34,839 bricks
2025-12: 16,525 bricks
2026-01: 20,640 bricks
2026-02: 18,185 bricks
```

## Usage in Strategy Scripts

```python
import pandas as pd
from v10_test_common import compute_smart_money

# Load train or test split
df = pd.read_parquet('data/renko_NQ_5pt_train.parquet')

# Standard OHLCV columns for the engine
opens  = df['Open'].values
highs  = df['High'].values
lows   = df['Low'].values
closes = df['Close'].values
volume = df['Volume'].values
times  = df.index.values

# Compute SM on Renko closes + volume
sm = compute_smart_money(closes, volume, sm_index=10, sm_flow=12,
                         sm_norm=200, sm_ema=100)
df['SM_Net'] = sm

# Renko-specific columns
direction = df['Direction'].values    # +1 or -1
buy_vol   = df['BuyVol'].values       # aggressor side split
sell_vol  = df['SellVol'].values
delta     = buy_vol - sell_vol         # volume delta per brick
```

**Important differences from 1-min bars**:
- Bricks are NOT time-uniform. Duration varies from <1 second to minutes.
- Cooldown in "bars" means "bricks", not "minutes".
- Session filtering needs the completion timestamp (index), not a fixed time grid.
- RSI on Renko closes behaves differently than RSI on time-based closes.
- High/Low within a brick exceed the box boundaries (they capture intrabar wicks).

## Rebuilding / Different Box Sizes

```bash
# Default: 5pt box, saves train/test/full parquet
python3 renko/build_renko.py

# Different box size
python3 renko/build_renko.py --box-size 10

# Stats only (no file save)
python3 renko/build_renko.py --box-size 3 --stats
```

The build script reads from `data/databento_NQ_ticks_*.parquet` and outputs to
`data/renko_NQ_{N}pt_{split}.parquet`.

## Files

| File | Purpose |
|------|---------|
| `renko/build_renko.py` | Build script: ticks -> Renko parquet |
| `renko/README.md` | This document |
| `strategies/build_renko.py` | Original copy (also in strategies/) |
