import pandas as pd
import numpy as np

# Load tick data
df = pd.read_parquet('data/databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet')
print(f"Total ticks: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}\n")

# Size distribution
print("=" * 60)
print("TRADE SIZE DISTRIBUTION")
print("=" * 60)
sizes = df['size']
print(f"Mean: {sizes.mean():.2f}")
print(f"Median: {sizes.median():.0f}")
print(f"Std: {sizes.std():.2f}")
print(f"Max: {sizes.max():,}")

# Percentiles
for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
    val = np.percentile(sizes, p)
    print(f"  P{p}: {val:.0f} contracts")

# Count by size bucket
print("\nSize bucket breakdown:")
buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 999999)]
for lo, hi in buckets:
    mask = (sizes >= lo) & (sizes <= hi)
    count = mask.sum()
    vol = sizes[mask].sum()
    pct_count = count / len(df) * 100
    pct_vol = vol / sizes.sum() * 100
    label = f"{lo}-{hi}" if hi < 999999 else f"{lo}+"
    print(f"  {label:>6} contracts: {count:>10,} ticks ({pct_count:>5.1f}%), "
          f"volume {vol:>12,} ({pct_vol:>5.1f}%)")

# Large trades (>=5 contracts) - potential "institutional" filter
print("\n\nLARGE TRADE ANALYSIS (>= 5 contracts)")
print("=" * 60)
large = df[df['size'] >= 5]
small = df[df['size'] < 5]
print(f"Large trades: {len(large):,} ({len(large)/len(df)*100:.1f}% of ticks)")
print(f"Large volume: {large['size'].sum():,} ({large['size'].sum()/sizes.sum()*100:.1f}% of total volume)")

# Delta from large trades only
large_buy = large[large['side'] == 'B']['size'].sum()
large_sell = large[large['side'] == 'A']['size'].sum()
small_buy = small[small['side'] == 'B']['size'].sum()
small_sell = small[small['side'] == 'A']['size'].sum()
print(f"\nLarge trade delta: {large_buy - large_sell:+,} ({large_buy:,} buy - {large_sell:,} sell)")
print(f"Small trade delta: {small_buy - small_sell:+,} ({small_buy:,} buy - {small_sell:,} sell)")

# Daily large trade delta vs small trade delta
df['ts'] = pd.to_datetime(df['ts_event'], utc=True)
df['date'] = df['ts'].dt.date

# For each day, compute large delta and small delta and total delta
daily = []
for date, group in df.groupby('date'):
    lg = group[group['size'] >= 5]
    sm = group[group['size'] < 5]
    
    lb = lg[lg['side'] == 'B']['size'].sum()
    ls = lg[lg['side'] == 'A']['size'].sum()
    sb = sm[sm['side'] == 'B']['size'].sum()
    ss = sm[sm['side'] == 'A']['size'].sum()
    
    daily.append({
        'date': date,
        'large_delta': int(lb) - int(ls),
        'small_delta': int(sb) - int(ss),
        'total_delta': int(lb) - int(ls) + int(sb) - int(ss),
    })

daily_df = pd.DataFrame(daily)

# Correlation between large and small daily deltas
corr = np.corrcoef(daily_df['large_delta'], daily_df['small_delta'])[0,1]
print(f"\nDaily large-delta vs small-delta correlation: {corr:.3f}")

# How often do they agree on direction?
agree = ((daily_df['large_delta'] > 0) & (daily_df['small_delta'] > 0)) | \
        ((daily_df['large_delta'] < 0) & (daily_df['small_delta'] < 0))
print(f"Daily sign agreement: {agree.mean()*100:.1f}%")

# When they disagree, who's right? (next day price direction)
# Skip for now, but print the daily data sample
print(f"\nSample daily deltas (last 10 days):")
print(daily_df.tail(10).to_string(index=False))

# VERY LARGE trades (>=10 contracts) 
print("\n\nVERY LARGE TRADE ANALYSIS (>= 10 contracts)")
print("=" * 60)
vlarge = df[df['size'] >= 10]
print(f"Very large trades: {len(vlarge):,} ({len(vlarge)/len(df)*100:.1f}% of ticks)")
print(f"Very large volume: {vlarge['size'].sum():,} ({vlarge['size'].sum()/sizes.sum()*100:.1f}% of total volume)")

# 1-min bars with large-trade-only delta
print("\n\nSAMPLE: 1-MIN BARS WITH SIZE-FILTERED DELTA")
print("=" * 60)
# Pick a random day
sample_date = daily_df.iloc[len(daily_df)//2]['date']
sample = df[df['date'] == sample_date].copy()
sample['bar'] = sample['ts'].dt.floor('1min')

# Per bar: total delta, large-only delta, small-only delta
bars = []
for bar_time, grp in sample.groupby('bar'):
    lg = grp[grp['size'] >= 5]
    sm = grp[grp['size'] < 5]
    
    total_buy = grp[grp['side']=='B']['size'].sum()
    total_sell = grp[grp['side']=='A']['size'].sum()
    large_buy = lg[lg['side']=='B']['size'].sum()
    large_sell = lg[lg['side']=='A']['size'].sum()
    small_buy = sm[sm['side']=='B']['size'].sum()
    small_sell = sm[sm['side']=='A']['size'].sum()
    
    bars.append({
        'time': bar_time,
        'total_delta': int(total_buy) - int(total_sell),
        'large_delta': int(large_buy) - int(large_sell),
        'small_delta': int(small_buy) - int(small_sell),
        'ticks': len(grp),
        'large_ticks': len(lg),
    })

bars_df = pd.DataFrame(bars)
print(f"Sample day: {sample_date} ({len(bars_df)} bars)")
print(f"\nBar-level correlation (total vs large delta): "
      f"{np.corrcoef(bars_df['total_delta'], bars_df['large_delta'])[0,1]:.3f}")
print(f"Bar-level correlation (large vs small delta): "
      f"{np.corrcoef(bars_df['large_delta'], bars_df['small_delta'])[0,1]:.3f}")

# Sign agreement at bar level
bar_agree = ((bars_df['large_delta'] > 0) & (bars_df['small_delta'] > 0)) | \
            ((bars_df['large_delta'] < 0) & (bars_df['small_delta'] < 0))
bar_either_zero = (bars_df['large_delta'] == 0) | (bars_df['small_delta'] == 0)
valid = ~bar_either_zero
print(f"Bar-level sign agreement: {bar_agree[valid].mean()*100:.1f}% (of {valid.sum()} bars with both)")

print("\nFirst 20 bars of sample day:")
print(bars_df.head(20)[['time','total_delta','large_delta','small_delta','ticks','large_ticks']].to_string(index=False))
