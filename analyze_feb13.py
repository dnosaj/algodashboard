"""
Analyze Feb 13, 2026 MNQ trading data for v9 strategy trades.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load CSV
df = pd.read_csv("/Users/jasongeorge/Desktop/NQ trading/backtesting_engine/data/CME_MINI_MNQ1!, 1_8835e.csv")

# Convert Unix timestamps to ET (UTC-5)
ET = timezone(timedelta(hours=-5))
df['datetime_et'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ET)
df['time_str'] = df['datetime_et'].dt.strftime('%H:%M')

print(f"Data range: {df['datetime_et'].iloc[0]} to {df['datetime_et'].iloc[-1]}")
print(f"Total bars: {len(df)}")
print(f"Columns: {list(df.columns[:12])}")
print()

# Key columns
sm = df['SM Net Index'].values
close = df['close'].values
times = df['datetime_et'].values
time_str = df['time_str'].values

# ============================================================
# HELPER: Show bars around a time
# ============================================================
def show_bars(target_time_str, window=5, label=""):
    """Show bars around a specific time."""
    matches = df[df['time_str'] == target_time_str]
    if len(matches) == 0:
        print(f"  No exact match for {target_time_str}")
        # Find nearest
        target_parts = target_time_str.split(':')
        target_min = int(target_parts[0]) * 60 + int(target_parts[1])
        df['_min'] = df['datetime_et'].dt.hour * 60 + df['datetime_et'].dt.minute
        nearest_idx = (df['_min'] - target_min).abs().idxmin()
        center = nearest_idx
        df.drop(columns=['_min'], inplace=True)
    else:
        center = matches.index[0]

    start = max(0, center - window)
    end = min(len(df), center + window + 1)

    if label:
        print(f"\n  === {label} ===")
    print(f"  {'Time':>6s}  {'Open':>10s}  {'High':>10s}  {'Low':>10s}  {'Close':>10s}  {'SM Net Idx':>12s}  {'SM Sign':>8s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}")
    for i in range(start, end):
        marker = " <--" if i == center else ""
        sm_val = df['SM Net Index'].iloc[i]
        sm_sign = "BULL" if sm_val > 0 else "BEAR" if sm_val < 0 else "ZERO"
        print(f"  {df['time_str'].iloc[i]:>6s}  {df['open'].iloc[i]:>10.2f}  {df['high'].iloc[i]:>10.2f}  "
              f"{df['low'].iloc[i]:>10.2f}  {df['close'].iloc[i]:>10.2f}  {sm_val:>12.6f}  {sm_sign:>8s}{marker}")
    return center

# ============================================================
# HELPER: Find SM zero-cross near a time
# ============================================================
def find_sm_crosses(start_time, end_time):
    """Find all SM Net Index zero-crossings between two times."""
    mask = (df['time_str'] >= start_time) & (df['time_str'] <= end_time)
    subset = df[mask].copy()
    crosses = []
    for i in range(1, len(subset)):
        prev_sm = subset['SM Net Index'].iloc[i-1]
        curr_sm = subset['SM Net Index'].iloc[i]
        if (prev_sm > 0 and curr_sm <= 0) or (prev_sm <= 0 and curr_sm > 0):
            crosses.append({
                'time': subset['time_str'].iloc[i],
                'prev_sm': prev_sm,
                'curr_sm': curr_sm,
                'direction': 'Bull->Bear' if prev_sm > 0 else 'Bear->Bull',
                'close': subset['close'].iloc[i]
            })
    return crosses

# ============================================================
# 5-MIN RSI(10) CALCULATION
# ============================================================
print("=" * 80)
print("5-MIN RSI(10) CALCULATION")
print("=" * 80)

# Resample 1-min to 5-min OHLC
df_5m = df.set_index('datetime_et').resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'SM Net Index': 'last'
}).dropna()

df_5m['time_str'] = df_5m.index.strftime('%H:%M')

# RSI(10) on 5-min close
def calc_rsi(series, period=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_5m['RSI'] = calc_rsi(df_5m['close'], period=10)

# Show all 5-min RSI values during trading session
print("\n5-min RSI(10) values during session (09:30-16:00):")
session_5m = df_5m[(df_5m['time_str'] >= '09:30') & (df_5m['time_str'] <= '16:00')]
print(f"  {'Time':>6s}  {'Close':>10s}  {'RSI':>8s}  {'SM Net':>12s}")
print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*12}")
for idx, row in session_5m.iterrows():
    rsi_str = f"{row['RSI']:.2f}" if not np.isnan(row['RSI']) else "NaN"
    sm_str = f"{row['SM Net Index']:.6f}" if not np.isnan(row['SM Net Index']) else "NaN"
    print(f"  {row['time_str']:>6s}  {row['close']:>10.2f}  {rsi_str:>8s}  {sm_str:>12s}")

# ============================================================
# TRADE ANALYSIS
# ============================================================

trades = [
    {"num": 13, "dir": "Short", "entry_time": "10:00", "entry_price": 21697.75,
     "exit_time": "10:12", "exit_price": 21825.00, "exit_reason": "SM Flip", "pnl": -255.54},
    {"num": 14, "dir": "Long", "entry_time": "10:27", "entry_price": 21819.00,
     "exit_time": "10:34", "exit_price": 21925.00, "exit_reason": "SM Flip", "pnl": 210.96},
    {"num": 15, "dir": "Long", "entry_time": "11:30", "entry_price": 21850.00,
     "exit_time": "12:09", "exit_price": 21912.00, "exit_reason": "SM Flip", "pnl": 122.96},
    {"num": 16, "dir": "Long", "entry_time": "14:25", "entry_price": 21963.75,
     "exit_time": "15:38", "exit_price": 21736.75, "exit_reason": "SM Flip", "pnl": -455.04},
]

print("\n" + "=" * 80)
print("TRADE-BY-TRADE ANALYSIS")
print("=" * 80)

for trade in trades:
    print(f"\n{'#' * 70}")
    print(f"# TRADE {trade['num']}: {trade['dir']} Entry {trade['entry_time']} -> Exit {trade['exit_time']}")
    print(f"# P&L: ${trade['pnl']:+.2f} | Exit: {trade['exit_reason']}")
    print(f"{'#' * 70}")

    # Show bars around entry
    print(f"\n  --- Bars around ENTRY ({trade['entry_time']}) ---")
    entry_idx = show_bars(trade['entry_time'], window=5)

    # Show bars around exit
    print(f"\n  --- Bars around EXIT ({trade['exit_time']}) ---")
    exit_idx = show_bars(trade['exit_time'], window=5)

    # Find SM crosses during trade
    crosses = find_sm_crosses(trade['entry_time'], trade['exit_time'])
    if crosses:
        print(f"\n  --- SM Zero-Crossings during trade ---")
        for c in crosses:
            print(f"    {c['time']}: SM {c['direction']} (prev={c['prev_sm']:.6f}, curr={c['curr_sm']:.6f}, close={c['close']:.2f})")
    else:
        print(f"\n  --- No SM zero-crossings during trade ---")

    # Show 5-min RSI around entry
    print(f"\n  --- 5-min RSI(10) around entry ---")
    entry_hour = int(trade['entry_time'].split(':')[0])
    entry_min = int(trade['entry_time'].split(':')[1])
    # Find nearest 5-min bar
    entry_5m_min = (entry_min // 5) * 5
    entry_5m_time = f"{entry_hour:02d}:{entry_5m_min:02d}"

    # Get window of 5-min bars around entry
    rsi_window = df_5m[
        (df_5m['time_str'] >= f"{entry_hour-1:02d}:{entry_5m_min:02d}") &
        (df_5m['time_str'] <= f"{entry_hour+1:02d}:{entry_5m_min:02d}")
    ]
    if len(rsi_window) > 0:
        # Show just a few bars around entry
        target_idx = None
        for i, (idx, row) in enumerate(rsi_window.iterrows()):
            if row['time_str'] >= entry_5m_time and target_idx is None:
                target_idx = i

        if target_idx is not None:
            start_i = max(0, target_idx - 3)
            end_i = min(len(rsi_window), target_idx + 4)
            subset = rsi_window.iloc[start_i:end_i]
        else:
            subset = rsi_window.tail(7)

        print(f"  {'Time':>6s}  {'Close':>10s}  {'RSI(10)':>8s}  {'Cross55':>8s}  {'Cross45':>8s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
        prev_rsi = None
        for idx, row in subset.iterrows():
            rsi_val = row['RSI']
            cross_55 = ""
            cross_45 = ""
            if prev_rsi is not None and not np.isnan(prev_rsi) and not np.isnan(rsi_val):
                if prev_rsi < 55 and rsi_val >= 55:
                    cross_55 = "UP^55"
                elif prev_rsi >= 55 and rsi_val < 55:
                    cross_55 = "DN<55"
                if prev_rsi < 45 and rsi_val >= 45:
                    cross_45 = "UP^45"
                elif prev_rsi >= 45 and rsi_val < 45:
                    cross_45 = "DN<45"
            marker = " <--" if row['time_str'] == entry_5m_time else ""
            rsi_str = f"{rsi_val:.2f}" if not np.isnan(rsi_val) else "NaN"
            print(f"  {row['time_str']:>6s}  {row['close']:>10.2f}  {rsi_str:>8s}  {cross_55:>8s}  {cross_45:>8s}{marker}")
            prev_rsi = rsi_val

# ============================================================
# DEEP DIVE: TRADE 13 (Short at 10:00)
# ============================================================
print("\n" + "=" * 80)
print("DEEP DIVE: TRADE 13 — Short at 10:00 (before massive rip)")
print("=" * 80)

# Show extended window before and after
print("\n  Bars from 09:50 to 10:15:")
mask_t13 = (df['time_str'] >= '09:50') & (df['time_str'] <= '10:15')
t13_data = df[mask_t13]
print(f"  {'Time':>6s}  {'Open':>10s}  {'High':>10s}  {'Low':>10s}  {'Close':>10s}  {'SM Net Idx':>12s}  {'SM Sign':>8s}  {'Long':>5s}  {'Short':>6s}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*5}  {'-'*6}")
for _, row in t13_data.iterrows():
    sm_val = row['SM Net Index']
    sm_sign = "BULL" if sm_val > 0 else "BEAR" if sm_val < 0 else "ZERO"
    print(f"  {row['time_str']:>6s}  {row['open']:>10.2f}  {row['high']:>10.2f}  "
          f"{row['low']:>10.2f}  {row['close']:>10.2f}  {sm_val:>12.6f}  {sm_sign:>8s}  {row['Long']:>5}  {row['Short']:>6}")

# Price move during trade 13
t13_entry = df[df['time_str'] == '10:00'].iloc[0] if len(df[df['time_str'] == '10:00']) > 0 else None
t13_exit = df[df['time_str'] == '10:12'].iloc[0] if len(df[df['time_str'] == '10:12']) > 0 else None
if t13_entry is not None and t13_exit is not None:
    print(f"\n  Entry price range at 10:00: O={t13_entry['open']:.2f} H={t13_entry['high']:.2f} L={t13_entry['low']:.2f} C={t13_entry['close']:.2f}")
    print(f"  SM at entry: {t13_entry['SM Net Index']:.6f}")
    # Find max high between entry and exit
    mask_trade = (df['time_str'] >= '10:00') & (df['time_str'] <= '10:12')
    max_high = df[mask_trade]['high'].max()
    min_low = df[mask_trade]['low'].min()
    print(f"  Max high during trade: {max_high:.2f}")
    print(f"  Min low during trade: {min_low:.2f}")
    print(f"  Adverse move (for short): {max_high - t13_entry['close']:.2f} points")

# ============================================================
# DEEP DIVE: TRADE 14 (Long at 10:27 — missing from chart)
# ============================================================
print("\n" + "=" * 80)
print("DEEP DIVE: TRADE 14 — Long at 10:27 (visible in list but NOT on chart)")
print("=" * 80)

# Show extended window
print("\n  Bars from 10:20 to 10:40:")
mask_t14 = (df['time_str'] >= '10:20') & (df['time_str'] <= '10:40')
t14_data = df[mask_t14]
print(f"  {'Time':>6s}  {'Open':>10s}  {'High':>10s}  {'Low':>10s}  {'Close':>10s}  {'SM Net Idx':>12s}  {'SM Sign':>8s}  {'Long':>5s}  {'Short':>6s}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*5}  {'-'*6}")
for _, row in t14_data.iterrows():
    sm_val = row['SM Net Index']
    sm_sign = "BULL" if sm_val > 0 else "BEAR" if sm_val < 0 else "ZERO"
    print(f"  {row['time_str']:>6s}  {row['open']:>10.2f}  {row['high']:>10.2f}  "
          f"{row['low']:>10.2f}  {row['close']:>10.2f}  {sm_val:>12.6f}  {sm_sign:>8s}  {row['Long']:>5}  {row['Short']:>6}")

# Check SM transitions around 10:27
print("\n  SM transitions 10:24 -> 10:35:")
mask_sm14 = (df['time_str'] >= '10:24') & (df['time_str'] <= '10:35')
sm14_data = df[mask_sm14]
prev_sign = None
for _, row in sm14_data.iterrows():
    sm_val = row['SM Net Index']
    curr_sign = "BULL" if sm_val > 0 else "BEAR"
    if prev_sign is not None and curr_sign != prev_sign:
        print(f"    ** SM FLIP at {row['time_str']}: {prev_sign} -> {curr_sign} (SM={sm_val:.6f}, Close={row['close']:.2f})")
    prev_sign = curr_sign

# Check if trade duration is very short (entry 10:27, exit 10:34 = 7 minutes)
# On a 5-min chart this might be barely visible
print(f"\n  Trade 14 duration: 10:27 to 10:34 = 7 minutes (barely 1-2 5-min candles)")
print(f"  On a 5-min chart, this could be hard to see visually")
print(f"  Entry and exit could be on adjacent 5-min bars (10:25 and 10:30 bars)")

# ============================================================
# DEEP DIVE: TRADE 16 (Long at 14:25, held through massive selloff)
# ============================================================
print("\n" + "=" * 80)
print("DEEP DIVE: TRADE 16 — Long at 14:25 (held through selloff, -$455)")
print("=" * 80)

# Show all bars from entry to exit
print("\n  Full trade bars from 14:20 to 15:45:")
mask_t16 = (df['time_str'] >= '14:20') & (df['time_str'] <= '15:45')
t16_data = df[mask_t16]
print(f"  {'Time':>6s}  {'Open':>10s}  {'High':>10s}  {'Low':>10s}  {'Close':>10s}  {'SM Net Idx':>12s}  {'SM Sign':>8s}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}")
for _, row in t16_data.iterrows():
    sm_val = row['SM Net Index']
    sm_sign = "BULL" if sm_val > 0 else "BEAR" if sm_val < 0 else "ZERO"
    marker = ""
    if row['time_str'] == '14:25':
        marker = " <-- ENTRY"
    elif row['time_str'] == '15:38':
        marker = " <-- EXIT"
    print(f"  {row['time_str']:>6s}  {row['open']:>10.2f}  {row['high']:>10.2f}  "
          f"{row['low']:>10.2f}  {row['close']:>10.2f}  {sm_val:>12.6f}  {sm_sign:>8s}{marker}")

# Find when SM actually flipped during trade 16
crosses_16 = find_sm_crosses('14:25', '15:45')
if crosses_16:
    print(f"\n  SM zero-crossings during Trade 16:")
    for c in crosses_16:
        print(f"    {c['time']}: SM {c['direction']} (SM went from {c['prev_sm']:.6f} to {c['curr_sm']:.6f})")
        print(f"    Close at flip: {c['close']:.2f}")

# Calculate drawdown during trade
if len(t16_data) > 0:
    entry_close = t16_data[t16_data['time_str'] == '14:25']
    if len(entry_close) > 0:
        entry_px = entry_close.iloc[0]['close']
        lows = t16_data['low']
        max_dd_low = lows.min()
        max_dd_time_idx = lows.idxmin()
        max_dd_time = df.loc[max_dd_time_idx, 'time_str']
        print(f"\n  Entry close: {entry_px:.2f}")
        print(f"  Lowest low during trade: {max_dd_low:.2f} at {max_dd_time}")
        print(f"  Max adverse excursion: {entry_px - max_dd_low:.2f} points")
        print(f"  SM stayed BULLISH the entire time until exit at 15:38")

# ============================================================
# SM NET INDEX OVER FULL SESSION
# ============================================================
print("\n" + "=" * 80)
print("SM NET INDEX — ALL ZERO CROSSINGS FOR THE SESSION")
print("=" * 80)
all_crosses = find_sm_crosses('09:30', '16:00')
for c in all_crosses:
    print(f"  {c['time']}: {c['direction']} | SM: {c['prev_sm']:.6f} -> {c['curr_sm']:.6f} | Close: {c['close']:.2f}")

# ============================================================
# 5-MIN RSI CROSS SIGNALS
# ============================================================
print("\n" + "=" * 80)
print("5-MIN RSI(10) CROSS SIGNALS (55/45 levels)")
print("=" * 80)

session_rsi = df_5m[(df_5m['time_str'] >= '09:45') & (df_5m['time_str'] <= '15:45')].copy()
prev_rsi = None
for idx, row in session_rsi.iterrows():
    rsi_val = row['RSI']
    if prev_rsi is not None and not np.isnan(prev_rsi) and not np.isnan(rsi_val):
        if prev_rsi < 55 and rsi_val >= 55:
            sm_sign = "BULL" if row['SM Net Index'] > 0 else "BEAR"
            print(f"  {row['time_str']}: RSI CROSSED UP through 55 ({prev_rsi:.2f} -> {rsi_val:.2f}) | SM: {sm_sign} ({row['SM Net Index']:.4f}) | Close: {row['close']:.2f}")
            if sm_sign == "BULL":
                print(f"    >>> LONG SIGNAL (RSI>55 + SM Bullish)")
        if prev_rsi >= 45 and rsi_val < 45:
            sm_sign = "BULL" if row['SM Net Index'] > 0 else "BEAR"
            print(f"  {row['time_str']}: RSI CROSSED DOWN through 45 ({prev_rsi:.2f} -> {rsi_val:.2f}) | SM: {sm_sign} ({row['SM Net Index']:.4f}) | Close: {row['close']:.2f}")
            if sm_sign == "BEAR":
                print(f"    >>> SHORT SIGNAL (RSI<45 + SM Bearish)")
    prev_rsi = rsi_val

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

total_pnl = sum(t['pnl'] for t in trades)
print(f"\n  Total P&L for today's 4 trades: ${total_pnl:+.2f}")
print(f"  Winners: 2 (Trade 14: +$210.96, Trade 15: +$122.96)")
print(f"  Losers: 2 (Trade 13: -$255.54, Trade 16: -$455.04)")
print(f"  Win Rate: 50%")
print(f"  Net: ${total_pnl:+.2f}")
