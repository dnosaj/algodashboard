"""
Deep analysis of MES and MNQ chart data exports.
Compare against everything we've learned about the strategy.
Devil's advocate: look for things we might be missing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ── Load data ──────────────────────────────────────────────────────────

def load_mes(path):
    df = pd.read_csv(path)
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(ET)
    return df

def load_mnq(path):
    # MNQ has duplicate column names: Long,Short,Long,Short
    # pandas will auto-suffix: Long, Short, Long.1, Short.1
    df = pd.read_csv(path)
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(ET)
    return df

mes = load_mes("/Users/jasongeorge/Downloads/CME_MINI_MES1-1_748c4.csv")
mnq = load_mnq("/Users/jasongeorge/Downloads/CME_MINI_MNQ1-1_e0438.csv")

print("=" * 90)
print("  DEEP CHART DATA ANALYSIS")
print("=" * 90)

# ── 1. Date range and basic stats ──────────────────────────────────────
print("\n" + "─" * 90)
print("  1. DATA OVERVIEW")
print("─" * 90)

for name, df in [("MES", mes), ("MNQ", mnq)]:
    print(f"\n  {name}:")
    print(f"    Bars: {len(df):,}")
    print(f"    Range: {df['dt'].iloc[0]} to {df['dt'].iloc[-1]}")
    print(f"    Columns: {list(df.columns[:15])}")
    # Count unique dates
    dates = df['dt'].dt.date.unique()
    print(f"    Trading days: {len(dates)} ({dates[0]} to {dates[-1]})")

# ── 2. MNQ duplicate columns investigation ────────────────────────────
print("\n" + "─" * 90)
print("  2. MNQ DUPLICATE Long/Short COLUMNS")
print("─" * 90)

# Check if pandas renamed them
mnq_cols = list(mnq.columns)
print(f"  MNQ column names: {mnq_cols[:15]}")

# Find which columns have signal data
for col in mnq_cols:
    if 'Long' in col or 'Short' in col or 'long' in col or 'short' in col:
        n_signals = (mnq[col] == 1).sum()
        if n_signals > 0 or 'Long' in col or 'Short' in col:
            print(f"    {col}: {n_signals} signals")

# Check if the two Long columns ever DISAGREE
if 'Long.1' in mnq_cols:
    both_have = ((mnq['Long'] == 1) & (mnq['Long.1'] == 1)).sum()
    first_only = ((mnq['Long'] == 1) & (mnq['Long.1'] == 0)).sum()
    second_only = ((mnq['Long'] == 0) & (mnq['Long.1'] == 1)).sum()
    print(f"\n  Long column comparison:")
    print(f"    Both = 1: {both_have}")
    print(f"    First only: {first_only}")
    print(f"    Second only: {second_only}")

if 'Short.1' in mnq_cols:
    both_have = ((mnq['Short'] == 1) & (mnq['Short.1'] == 1)).sum()
    first_only = ((mnq['Short'] == 1) & (mnq['Short.1'] == 0)).sum()
    second_only = ((mnq['Short'] == 0) & (mnq['Short.1'] == 1)).sum()
    print(f"  Short column comparison:")
    print(f"    Both = 1: {both_have}")
    print(f"    First only: {first_only}")
    print(f"    Second only: {second_only}")

# ── 3. Signal analysis ─────────────────────────────────────────────────
print("\n" + "─" * 90)
print("  3. SIGNAL ANALYSIS")
print("─" * 90)

def analyze_signals(df, name, long_col='Long', short_col='Short'):
    longs = df[df[long_col] == 1].copy()
    shorts = df[df[short_col] == 1].copy()
    print(f"\n  {name}:")
    print(f"    Long signals: {len(longs)}")
    print(f"    Short signals: {len(shorts)}")

    if len(longs) > 0:
        print(f"\n    Long signal bars:")
        for _, row in longs.iterrows():
            sm = row.get('SM Net Index', float('nan'))
            print(f"      {row['dt'].strftime('%m/%d %H:%M')}  close={row['close']:.2f}  SM={sm:.4f}")

    if len(shorts) > 0:
        print(f"\n    Short signal bars:")
        for _, row in shorts.iterrows():
            sm = row.get('SM Net Index', float('nan'))
            print(f"      {row['dt'].strftime('%m/%d %H:%M')}  close={row['close']:.2f}  SM={sm:.4f}")

analyze_signals(mes, "MES v9.4")

# For MNQ, analyze both column pairs
mnq_long_col = 'Long'
mnq_short_col = 'Short'
if 'Long.1' in mnq.columns:
    # Figure out which pair has the actual signals
    first_signals = (mnq['Long'] == 1).sum() + (mnq['Short'] == 1).sum()
    second_signals = (mnq['Long.1'] == 1).sum() + (mnq['Short.1'] == 1).sum()
    print(f"\n  MNQ signal count: first pair={first_signals}, second pair={second_signals}")
    if second_signals > first_signals:
        mnq_long_col = 'Long.1'
        mnq_short_col = 'Short.1'
        print(f"  --> Using second pair (Long.1/Short.1) as primary signals")
    # Show ALL signals from both pairs
    analyze_signals(mnq, "MNQ v11 (pair 1)", 'Long', 'Short')
    if second_signals > 0:
        analyze_signals(mnq, "MNQ v11 (pair 2)", 'Long.1', 'Short.1')

# ── 4. SM Net Index at entry — strength analysis ──────────────────────
print("\n" + "─" * 90)
print("  4. SM STRENGTH AT ENTRY")
print("─" * 90)

def sm_entry_analysis(df, name, long_col='Long', short_col='Short'):
    sm_col = 'SM Net Index'
    longs = df[df[long_col] == 1]
    shorts = df[df[short_col] == 1]

    if len(longs) > 0:
        sm_at_long = longs[sm_col].values
        print(f"\n  {name} — SM at Long entries:")
        print(f"    Mean: {np.mean(sm_at_long):.4f}")
        print(f"    Median: {np.median(sm_at_long):.4f}")
        print(f"    Min: {np.min(sm_at_long):.4f}")
        print(f"    Max: {np.max(sm_at_long):.4f}")
        weak = (np.abs(sm_at_long) < 0.05).sum()
        print(f"    Weak entries (|SM| < 0.05): {weak}/{len(sm_at_long)}")

    if len(shorts) > 0:
        sm_at_short = shorts[sm_col].abs().values
        print(f"\n  {name} — |SM| at Short entries:")
        print(f"    Mean: {np.mean(sm_at_short):.4f}")
        print(f"    Median: {np.median(sm_at_short):.4f}")
        print(f"    Min: {np.min(sm_at_short):.4f}")
        print(f"    Max: {np.max(sm_at_short):.4f}")

sm_entry_analysis(mes, "MES v9.4")
if 'Long.1' in mnq.columns:
    sm_entry_analysis(mnq, "MNQ v11", mnq_long_col, mnq_short_col)
else:
    sm_entry_analysis(mnq, "MNQ v11")

# ── 5. SM behavior — flip frequency, chop detection ───────────────────
print("\n" + "─" * 90)
print("  5. SM FLIP FREQUENCY & CHOP ANALYSIS")
print("─" * 90)

def sm_flip_analysis(df, name):
    sm = df['SM Net Index'].values
    dt = df['dt'].values

    # Find SM sign changes
    signs = np.sign(sm)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    print(f"\n  {name}:")
    print(f"    Total bars: {len(sm)}")
    print(f"    SM flips: {len(sign_changes)}")

    if len(sign_changes) < 2:
        return

    # Duration between flips (in bars)
    flip_gaps = np.diff(sign_changes)
    print(f"    Avg bars between flips: {np.mean(flip_gaps):.1f}")
    print(f"    Median bars between flips: {np.median(flip_gaps):.1f}")
    print(f"    Min bars between flips: {np.min(flip_gaps)}")
    print(f"    Max bars between flips: {np.max(flip_gaps)}")

    # Rapid flips (< 5 bars = < 5 minutes)
    rapid = (flip_gaps < 5).sum()
    print(f"    Rapid flips (< 5 bars): {rapid} ({rapid/len(flip_gaps)*100:.1f}%)")

    # Very rapid (< 3 bars)
    very_rapid = (flip_gaps < 3).sum()
    print(f"    Very rapid flips (< 3 bars): {very_rapid} ({very_rapid/len(flip_gaps)*100:.1f}%)")

    # SM regime durations by day
    dates = pd.to_datetime(dt).date if hasattr(dt[0], 'date') else pd.DatetimeIndex(dt).date
    unique_dates = sorted(set(dates))

    print(f"\n    Daily SM flip count:")
    for d in unique_dates[-8:]:  # last 8 days
        mask = np.array([dd == d for dd in dates])
        day_sm = sm[mask]
        day_signs = np.sign(day_sm)
        day_flips = np.sum(np.diff(day_signs) != 0)
        day_bars = len(day_sm)
        print(f"      {d}: {day_flips} flips in {day_bars} bars ({day_flips/max(day_bars,1)*60:.1f} flips/hr)")

sm_flip_analysis(mes, "MES")
sm_flip_analysis(mnq, "MNQ")

# ── 6. SM magnitude distribution — is it compressed? ──────────────────
print("\n" + "─" * 90)
print("  6. SM MAGNITUDE DISTRIBUTION (IS IT COMPRESSED?)")
print("─" * 90)

def sm_magnitude(df, name):
    valid_mask = df['SM Net Index'].notna()
    df_valid = df[valid_mask]
    sm = df_valid['SM Net Index'].values
    dt = df_valid['dt']

    # Overall
    print(f"\n  {name} overall:")
    print(f"    Mean |SM|: {np.mean(np.abs(sm)):.4f}")
    print(f"    Median |SM|: {np.median(np.abs(sm)):.4f}")
    print(f"    P10 |SM|: {np.percentile(np.abs(sm), 10):.4f}")
    print(f"    P90 |SM|: {np.percentile(np.abs(sm), 90):.4f}")
    print(f"    Max |SM|: {np.max(np.abs(sm)):.4f}")

    # By day (check if recent days have weaker SM)
    dates = dt.dt.date.values
    unique_dates = sorted(set(dates))

    print(f"\n    Daily mean |SM|:")
    for d in unique_dates[-8:]:
        mask = dates == d
        day_sm = sm[mask]
        if len(day_sm) > 0:
            print(f"      {d}: mean={np.mean(np.abs(day_sm)):.4f}  "
                  f"max={np.max(np.abs(day_sm)):.4f}  "
                  f"pct_weak(<0.05)={100*(np.abs(day_sm)<0.05).sum()/len(day_sm):.0f}%")

sm_magnitude(mes, "MES")
sm_magnitude(mnq, "MNQ")

# ── 7. VWAP relationship at entries ───────────────────────────────────
print("\n" + "─" * 90)
print("  7. VWAP RELATIONSHIP AT ENTRY")
print("─" * 90)

def vwap_analysis(df, name, long_col='Long', short_col='Short'):
    if 'VWAP' not in df.columns:
        print(f"  {name}: No VWAP column")
        return

    longs = df[df[long_col] == 1]
    shorts = df[df[short_col] == 1]

    if len(longs) > 0:
        above_vwap = (longs['close'] > longs['VWAP']).sum()
        below_vwap = (longs['close'] < longs['VWAP']).sum()
        avg_dist = ((longs['close'] - longs['VWAP']) / longs['VWAP'] * 100).mean()
        print(f"\n  {name} Long entries vs VWAP:")
        print(f"    Above VWAP: {above_vwap}/{len(longs)} ({above_vwap/len(longs)*100:.0f}%)")
        print(f"    Below VWAP: {below_vwap}/{len(longs)} ({below_vwap/len(longs)*100:.0f}%)")
        print(f"    Avg distance from VWAP: {avg_dist:+.3f}%")
        for _, row in longs.iterrows():
            dist = (row['close'] - row['VWAP']) / row['VWAP'] * 100
            side = "ABOVE" if dist > 0 else "BELOW"
            print(f"      {row['dt'].strftime('%m/%d %H:%M')}  close={row['close']:.2f}  VWAP={row['VWAP']:.2f}  {side} {abs(dist):.3f}%")

    if len(shorts) > 0:
        above_vwap = (shorts['close'] > shorts['VWAP']).sum()
        below_vwap = (shorts['close'] < shorts['VWAP']).sum()
        avg_dist = ((shorts['close'] - shorts['VWAP']) / shorts['VWAP'] * 100).mean()
        print(f"\n  {name} Short entries vs VWAP:")
        print(f"    Above VWAP: {above_vwap}/{len(shorts)} ({above_vwap/len(shorts)*100:.0f}%)")
        print(f"    Below VWAP: {below_vwap}/{len(shorts)} ({below_vwap/len(shorts)*100:.0f}%)")
        print(f"    Avg distance from VWAP: {avg_dist:+.3f}%")
        for _, row in shorts.iterrows():
            dist = (row['close'] - row['VWAP']) / row['VWAP'] * 100
            side = "ABOVE" if dist > 0 else "BELOW"
            print(f"      {row['dt'].strftime('%m/%d %H:%M')}  close={row['close']:.2f}  VWAP={row['VWAP']:.2f}  {side} {abs(dist):.3f}%")

vwap_analysis(mes, "MES v9.4")
if 'Long.1' in mnq.columns:
    vwap_analysis(mnq, "MNQ v11", mnq_long_col, mnq_short_col)
else:
    vwap_analysis(mnq, "MNQ v11")

# ── 8. Price move AFTER entry signals (next 5, 10, 20 bars) ──────────
print("\n" + "─" * 90)
print("  8. PRICE MOVE AFTER ENTRY SIGNALS (MFE/MAE)")
print("─" * 90)

def post_entry_analysis(df, name, long_col='Long', short_col='Short'):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    for sig_type, sig_col, direction in [('Long', long_col, 1), ('Short', short_col, -1)]:
        signal_idxs = df.index[df[sig_col] == 1].tolist()
        if not signal_idxs:
            continue

        print(f"\n  {name} {sig_type} signals — price evolution:")

        for idx in signal_idxs:
            iloc_idx = df.index.get_loc(idx)
            entry_price = closes[iloc_idx]
            dt_str = df['dt'].iloc[iloc_idx].strftime('%m/%d %H:%M')

            # Look forward up to 30 bars
            max_look = min(30, len(closes) - iloc_idx - 1)
            if max_look < 5:
                continue

            future_closes = closes[iloc_idx+1:iloc_idx+max_look+1]
            future_highs = highs[iloc_idx+1:iloc_idx+max_look+1]
            future_lows = lows[iloc_idx+1:iloc_idx+max_look+1]

            if direction == 1:  # Long
                mfe_5 = np.max(future_highs[:5]) - entry_price
                mae_5 = entry_price - np.min(future_lows[:5])
                mfe_10 = np.max(future_highs[:min(10,max_look)]) - entry_price
                mae_10 = entry_price - np.min(future_lows[:min(10,max_look)])
                mfe_20 = np.max(future_highs[:min(20,max_look)]) - entry_price
                mae_20 = entry_price - np.min(future_lows[:min(20,max_look)])
                move_5 = future_closes[min(4,len(future_closes)-1)] - entry_price
            else:  # Short
                mfe_5 = entry_price - np.min(future_lows[:5])
                mae_5 = np.max(future_highs[:5]) - entry_price
                mfe_10 = entry_price - np.min(future_lows[:min(10,max_look)])
                mae_10 = np.max(future_highs[:min(10,max_look)]) - entry_price
                mfe_20 = entry_price - np.min(future_lows[:min(20,max_look)])
                mae_20 = np.max(future_highs[:min(20,max_look)]) - entry_price
                move_5 = entry_price - future_closes[min(4,len(future_closes)-1)]

            win = "WIN" if move_5 > 0 else "LOSS"
            print(f"    {dt_str} entry={entry_price:.2f}")
            print(f"      5-bar:  MFE={mfe_5:+.2f}  MAE={mae_5:.2f}  net={move_5:+.2f} {win}")
            print(f"      10-bar: MFE={mfe_10:+.2f}  MAE={mae_10:.2f}")
            print(f"      20-bar: MFE={mfe_20:+.2f}  MAE={mae_20:.2f}")

post_entry_analysis(mes, "MES v9.4")
if 'Long.1' in mnq.columns:
    post_entry_analysis(mnq, "MNQ v11", mnq_long_col, mnq_short_col)
else:
    post_entry_analysis(mnq, "MNQ v11")

# ── 9. SM Bull/Bear flip bars — what's REALLY happening ──────────────
print("\n" + "─" * 90)
print("  9. SM BULL/BEAR FLIP BARS (what triggers entries/exits)")
print("─" * 90)

def flip_bar_analysis(df, name):
    bull_bars = df[df['SM Bull'] == 1]
    bear_bars = df[df['SM Bear'] == 1]

    print(f"\n  {name}:")
    print(f"    SM Bull flip bars: {len(bull_bars)}")
    print(f"    SM Bear flip bars: {len(bear_bars)}")

    # Show all bull/bear flips with context
    for flip_type, flip_df in [("BULL", bull_bars), ("BEAR", bear_bars)]:
        if len(flip_df) == 0:
            continue
        print(f"\n    SM {flip_type} flips:")
        for _, row in flip_df.iterrows():
            sm = row.get('SM Net Index', float('nan'))
            vwap = row.get('VWAP', float('nan'))
            dist_vwap = (row['close'] - vwap) / vwap * 100 if not np.isnan(vwap) else float('nan')
            print(f"      {row['dt'].strftime('%m/%d %H:%M')}  close={row['close']:.2f}  "
                  f"SM={sm:+.4f}  VWAP_dist={dist_vwap:+.3f}%")

flip_bar_analysis(mes, "MES")
flip_bar_analysis(mnq, "MNQ")

# ── 10. Time-of-day pattern for signals ───────────────────────────────
print("\n" + "─" * 90)
print("  10. TIME-OF-DAY DISTRIBUTION")
print("─" * 90)

def time_of_day(df, name, long_col='Long', short_col='Short'):
    all_signals = df[(df[long_col] == 1) | (df[short_col] == 1)]
    if len(all_signals) == 0:
        print(f"  {name}: No signals")
        return

    hours = all_signals['dt'].dt.hour.values
    print(f"\n  {name} signal distribution by hour:")
    for h in sorted(set(hours)):
        count = (hours == h).sum()
        bar = "#" * count
        print(f"    {h:02d}:00  {count:2d}  {bar}")

time_of_day(mes, "MES v9.4")
if 'Long.1' in mnq.columns:
    time_of_day(mnq, "MNQ v11", mnq_long_col, mnq_short_col)

# ── 11. Volatility analysis — is this regime different? ───────────────
print("\n" + "─" * 90)
print("  11. INTRADAY VOLATILITY BY DAY")
print("─" * 90)

def vol_by_day(df, name):
    df_copy = df.copy()
    df_copy['date'] = df_copy['dt'].dt.date
    df_copy['range'] = df_copy['high'] - df_copy['low']

    daily = df_copy.groupby('date').agg(
        bars=('close', 'count'),
        avg_range=('range', 'mean'),
        max_range=('range', 'max'),
        day_range=('high', lambda x: x.max() - df_copy.loc[x.index, 'low'].min()),
        open_price=('open', 'first'),
        close_price=('close', 'last'),
    )
    daily['day_move'] = daily['close_price'] - daily['open_price']
    daily['day_move_pct'] = daily['day_move'] / daily['open_price'] * 100

    print(f"\n  {name}:")
    print(f"    {'Date':>12}  {'Bars':>5}  {'AvgRange':>9}  {'MaxRange':>9}  "
          f"{'DayRange':>9}  {'DayMove':>9}  {'Move%':>7}")
    print(f"    {'-'*70}")
    for d, row in daily.iterrows():
        print(f"    {str(d):>12}  {row['bars']:>5}  {row['avg_range']:>9.2f}  "
              f"{row['max_range']:>9.2f}  {row['day_range']:>9.2f}  "
              f"{row['day_move']:>+9.2f}  {row['day_move_pct']:>+7.3f}%")

vol_by_day(mes, "MES")
vol_by_day(mnq, "MNQ")

# ── 12. Buy Interest / Sell Interest columns ──────────────────────────
print("\n" + "─" * 90)
print("  12. BUY/SELL INTEREST COLUMNS (are they populated?)")
print("─" * 90)

for name, df in [("MES", mes), ("MNQ", mnq)]:
    for col in ['Buy Interest', 'Sell Interest']:
        if col in df.columns:
            non_null = df[col].notna().sum()
            non_zero = (df[col] != 0).sum() if non_null > 0 else 0
            print(f"  {name} {col}: {non_null} non-null, {non_zero} non-zero")

# ── 13. Net Buy/Sell bands — are they useful? ─────────────────────────
print("\n" + "─" * 90)
print("  13. NET BUY/SELL BANDS ANALYSIS")
print("─" * 90)

for name, df in [("MES", mes), ("MNQ", mnq)]:
    if 'Net Buy Line' in df.columns and 'Net Sell Line' in df.columns:
        buy_line = df['Net Buy Line'].dropna()
        sell_line = df['Net Sell Line'].dropna()

        if len(buy_line) > 0:
            print(f"\n  {name} Net Buy Line: mean={buy_line.mean():.4f}, populated={len(buy_line)}/{len(df)}")
        if len(sell_line) > 0:
            print(f"  {name} Net Sell Line: mean={sell_line.mean():.4f}, populated={len(sell_line)}/{len(df)}")

        # When BOTH are populated, which is dominant?
        both = df[['Net Buy Line', 'Net Sell Line']].dropna()
        if len(both) > 0:
            buy_dom = (both['Net Buy Line'].abs() > both['Net Sell Line'].abs()).sum()
            print(f"  {name} Buy dominant: {buy_dom}/{len(both)} ({buy_dom/len(both)*100:.0f}%)")

# ── 14. The devil's advocate section ──────────────────────────────────
print("\n" + "=" * 90)
print("  DEVIL'S ADVOCATE: POTENTIAL ISSUES")
print("=" * 90)

# Check 1: SM barely positive/negative at entries (weak conviction)
print("\n  [A] WEAK SM CONVICTION AT ENTRY")
for name, df, lc, sc in [("MES", mes, 'Long', 'Short'),
                          ("MNQ", mnq, mnq_long_col, mnq_short_col)]:
    signals = df[(df[lc] == 1) | (df[sc] == 1)]
    if len(signals) == 0:
        continue
    weak = signals[signals['SM Net Index'].abs() < 0.1]
    print(f"  {name}: {len(weak)}/{len(signals)} entries have |SM| < 0.1 (weak)")
    for _, row in weak.iterrows():
        sig = "LONG" if row[lc] == 1 else "SHORT"
        print(f"    {row['dt'].strftime('%m/%d %H:%M')} {sig} SM={row['SM Net Index']:+.4f}")

# Check 2: Entries against VWAP trend
print("\n  [B] ENTRIES AGAINST VWAP (counter-trend)")
for name, df, lc, sc in [("MES", mes, 'Long', 'Short'),
                          ("MNQ", mnq, mnq_long_col, mnq_short_col)]:
    longs_below = df[(df[lc] == 1) & (df['close'] < df['VWAP'])]
    shorts_above = df[(df[sc] == 1) & (df['close'] > df['VWAP'])]
    total_signals = (df[lc] == 1).sum() + (df[sc] == 1).sum()
    counter = len(longs_below) + len(shorts_above)
    print(f"  {name}: {counter}/{total_signals} entries are counter-VWAP")
    for _, row in longs_below.iterrows():
        dist = (row['close'] - row['VWAP']) / row['VWAP'] * 100
        print(f"    LONG below VWAP: {row['dt'].strftime('%m/%d %H:%M')} dist={dist:+.3f}%")
    for _, row in shorts_above.iterrows():
        dist = (row['close'] - row['VWAP']) / row['VWAP'] * 100
        print(f"    SHORT above VWAP: {row['dt'].strftime('%m/%d %H:%M')} dist={dist:+.3f}%")

# Check 3: How fast SM flips AFTER entry (time to first adverse flip)
print("\n  [C] TIME-TO-ADVERSE-SM-FLIP AFTER ENTRY")
for name, df, lc, sc in [("MES", mes, 'Long', 'Short'),
                          ("MNQ", mnq, mnq_long_col, mnq_short_col)]:
    sm = df['SM Net Index'].values
    for sig_type, sig_col, adverse_col in [('Long', lc, 'SM Bear'), ('Short', sc, 'SM Bull')]:
        signal_idxs = df.index[df[sig_col] == 1].tolist()
        if not signal_idxs:
            continue
        for idx in signal_idxs:
            iloc_idx = df.index.get_loc(idx)
            # Find next adverse SM flip
            for j in range(iloc_idx+1, min(iloc_idx+60, len(df))):
                if df[adverse_col].iloc[j] == 1:
                    bars_to_flip = j - iloc_idx
                    dt_str = df['dt'].iloc[iloc_idx].strftime('%m/%d %H:%M')
                    sm_at_entry = sm[iloc_idx]
                    print(f"    {name} {sig_type} {dt_str}: SM flips adverse in {bars_to_flip} bars "
                          f"(SM at entry={sm_at_entry:+.4f})")
                    break

# Check 4: Session edge entries (within 15 min of session end at 15:45)
print("\n  [D] LATE SESSION ENTRIES (within 15 min of 15:45)")
for name, df, lc, sc in [("MES", mes, 'Long', 'Short'),
                          ("MNQ", mnq, mnq_long_col, mnq_short_col)]:
    signals = df[(df[lc] == 1) | (df[sc] == 1)]
    late = signals[signals['dt'].dt.hour * 60 + signals['dt'].dt.minute >= 15*60+30]
    if len(late) > 0:
        print(f"  {name}: {len(late)} late entries")
        for _, row in late.iterrows():
            sig = "LONG" if row[lc] == 1 else "SHORT"
            print(f"    {row['dt'].strftime('%m/%d %H:%M')} {sig}")
    else:
        print(f"  {name}: 0 late entries")

print("\n" + "=" * 90)
print("  ANALYSIS COMPLETE")
print("=" * 90)
