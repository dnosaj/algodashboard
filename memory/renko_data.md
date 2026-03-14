# Renko Bar Data (Feb 18, 2026)

## What Exists

Traditional Renko bars built from 42.6M NQ tick data (Databento). 5pt box size.

### Files
- `data/renko_NQ_5pt_full.parquet` — 129,416 bricks (Aug 17 - Feb 12)
- `data/renko_NQ_5pt_train.parquet` — 52,757 bricks (Aug 17 - Nov 16)
- `data/renko_NQ_5pt_test.parquet` — 76,659 bricks (Nov 17 - Feb 12)
- `renko/build_renko.py` — Build script (also at `strategies/build_renko.py`)
- `renko/README.md` — Full documentation

### Schema
DatetimeIndex (brick completion time) + columns: Open, High, Low, Close, Volume, BuyVol, SellVol, TickCount, Direction (+1/-1), StartTime.

### Key Stats (5pt box)
- ~835 bricks/day, median duration 9 seconds, P90 duration 3 minutes
- 50/50 up/down split, avg 493 contracts per brick
- Train/test split at Nov 17 (matching project convention)

### How to Rebuild / Change Box Size
```bash
python3 renko/build_renko.py --box-size 10    # different size
python3 renko/build_renko.py --stats           # stats only
```

### Usage
```python
df = pd.read_parquet('data/renko_NQ_5pt_train.parquet')
sm = compute_smart_money(df['Close'].values, df['Volume'].values, 10, 12, 200, 100)
```

### Important Differences from 1-min Bars
- NOT time-uniform (variable brick duration)
- Cooldown in "bars" = bricks, not minutes
- Session filtering uses completion timestamp, not fixed time grid
- RSI on Renko closes behaves differently than time-based RSI
- High/Low capture intrabar wicks beyond box boundaries
- Source is NQ (full-size), not MNQ — same price, more institutional volume

### Source Data
- `data/databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet` (42.6M ticks)
- Fetched via `strategies/fetch_databento_ticks.py`, cost $53.31
