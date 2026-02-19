#!/usr/bin/env python3
"""Analyze MES Feb 13 trades from TradingView CSV with AlgoAlpha SM columns."""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv("/Users/jasongeorge/Downloads/CME_MINI_MES1!, 1_37616.csv")

# Convert unix timestamp to ET (UTC-5)
ET = timezone(timedelta(hours=-5))
df['datetime'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
df['date'] = df['datetime'].dt.date
df['time_str'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')

print("=" * 90)
print("MES DATA OVERVIEW")
print("=" * 90)
print(f"Total bars: {len(df)}")
print(f"Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
print(f"Unique dates: {sorted(df['date'].unique())}")
print(f"Columns: {list(df.columns[:15])}")

# ── Focus on Feb 13 ───────────────────────────────────────────────────
from datetime import date
feb13 = df[df['date'] == date(2026, 2, 13)].copy()
print(f"\nFeb 13 bars: {len(feb13)}")
print(f"Feb 13 time range: {feb13['datetime'].iloc[0]} to {feb13['datetime'].iloc[-1]}")
print(f"Feb 13 price range: {feb13['low'].min():.2f} - {feb13['high'].max():.2f}")
print(f"Feb 13 open: {feb13['open'].iloc[0]:.2f}, close: {feb13['close'].iloc[-1]:.2f}")

# ── Calculate 5-min RSI(10) ───────────────────────────────────────────
# Resample 1-min to 5-min OHLC
df_all = df.set_index('datetime').copy()
df_5m = df_all[['open','high','low','close']].resample('5min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()

# RSI calculation
def calc_rsi(series, period=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df_5m['rsi'] = calc_rsi(df_5m['close'], 10)

# Merge RSI back to 1-min by forward-filling the 5-min RSI
# Each 5-min bar's RSI applies to the 5-min window ending at that bar
df_all['rsi_5m'] = df_5m['rsi'].reindex(df_all.index, method='ffill')

feb13_full = df_all[df_all.index.date == date(2026, 2, 13)].copy()

# ── SM columns analysis ──────────────────────────────────────────────
print("\n" + "=" * 90)
print("SMART MONEY COLUMNS ON FEB 13")
print("=" * 90)

# Check SM Net Index range
sm = feb13_full['SM Net Index']
print(f"SM Net Index range: {sm.min():.4f} to {sm.max():.4f}")
print(f"SM Net Index mean: {sm.mean():.4f}")

# SM Bull / SM Bear
bull = feb13_full['SM Bull']
bear = feb13_full['SM Bear']
print(f"SM Bull non-zero bars: {(bull != 0).sum()} / {len(feb13_full)}")
print(f"SM Bear non-zero bars: {(bear != 0).sum()} / {len(feb13_full)}")

# Long/Short signals
longs = feb13_full[feb13_full['Long'] != 0]
shorts = feb13_full[feb13_full['Short'] != 0]
print(f"\nLong signals on Feb 13: {len(longs)}")
for idx, row in longs.iterrows():
    print(f"  {idx.strftime('%H:%M')} - Close: {row['close']:.2f}, SM Net: {row['SM Net Index']:.4f}, RSI5m: {row['rsi_5m']:.2f}")

print(f"\nShort signals on Feb 13: {len(shorts)}")
for idx, row in shorts.iterrows():
    print(f"  {idx.strftime('%H:%M')} - Close: {row['close']:.2f}, SM Net: {row['SM Net Index']:.4f}, RSI5m: {row['rsi_5m']:.2f}")

# ── TRADE 1 ANALYSIS ─────────────────────────────────────────────────
print("\n" + "=" * 90)
print("TRADE 1: LONG 11:30 -> 12:29 (WINNER +$91.46)")
print("=" * 90)

t1_entry = pd.Timestamp('2026-02-13 11:30:00', tz='US/Eastern')
t1_exit = pd.Timestamp('2026-02-13 12:29:00', tz='US/Eastern')

# Show bars around entry
print("\n--- Bars around Trade 1 Entry (11:25 - 11:35) ---")
t1_pre = feb13_full.loc['2026-02-13 11:25:00':'2026-02-13 11:35:00']
for idx, row in t1_pre.iterrows():
    marker = " <<< ENTRY" if idx.strftime('%H:%M') == '11:30' else ""
    sm_bull = f"Bull={row['SM Bull']:.4f}" if row['SM Bull'] != 0 else ""
    sm_bear = f"Bear={row['SM Bear']:.4f}" if row['SM Bear'] != 0 else ""
    long_sig = " [LONG SIGNAL]" if row['Long'] != 0 else ""
    print(f"  {idx.strftime('%H:%M')} C={row['close']:.2f} SM_Net={row['SM Net Index']:.6f} {sm_bull} {sm_bear} RSI5m={row['rsi_5m']:.2f}{long_sig}{marker}")

# Show SM Net during entire trade 1
print("\n--- SM Net Index during Trade 1 (every 5 bars) ---")
t1_bars = feb13_full.loc[t1_entry:t1_exit]
for i, (idx, row) in enumerate(t1_bars.iterrows()):
    if i % 5 == 0 or idx == t1_bars.index[-1]:
        print(f"  {idx.strftime('%H:%M')} C={row['close']:.2f} SM_Net={row['SM Net Index']:.6f} RSI5m={row['rsi_5m']:.2f} H={row['high']:.2f} L={row['low']:.2f}")

# Show bars around exit
print("\n--- Bars around Trade 1 Exit (12:25 - 12:35) ---")
t1_exit_area = feb13_full.loc['2026-02-13 12:25:00':'2026-02-13 12:35:00']
for idx, row in t1_exit_area.iterrows():
    marker = " <<< EXIT" if idx.strftime('%H:%M') == '12:29' else ""
    sm_bull = f"Bull={row['SM Bull']:.4f}" if row['SM Bull'] != 0 else ""
    sm_bear = f"Bear={row['SM Bear']:.4f}" if row['SM Bear'] != 0 else ""
    short_sig = " [SHORT SIGNAL]" if row['Short'] != 0 else ""
    print(f"  {idx.strftime('%H:%M')} C={row['close']:.2f} SM_Net={row['SM Net Index']:.6f} {sm_bull} {sm_bear} RSI5m={row['rsi_5m']:.2f}{short_sig}{marker}")

# Trade 1 stats
t1_highs = t1_bars['high'].max()
t1_lows = t1_bars['low'].min()
entry_price = 6072.75  # corrected below
print(f"\nTrade 1 price extremes: High {t1_highs:.2f}, Low {t1_lows:.2f}")
print(f"Entry at 6872.75, Max favorable: {t1_highs - 6872.75:.2f} pts, Max adverse: {6872.75 - t1_lows:.2f} pts")

# ── TRADE 2 ANALYSIS (THE LOSER) ─────────────────────────────────────
print("\n" + "=" * 90)
print("TRADE 2: LONG 13:25 -> 14:57 (LOSER -$139.79)")
print("=" * 90)

t2_entry = pd.Timestamp('2026-02-13 13:25:00', tz='US/Eastern')
t2_exit = pd.Timestamp('2026-02-13 14:57:00', tz='US/Eastern')

# Show bars around entry
print("\n--- Bars around Trade 2 Entry (13:20 - 13:30) ---")
t2_pre = feb13_full.loc['2026-02-13 13:20:00':'2026-02-13 13:30:00']
for idx, row in t2_pre.iterrows():
    marker = " <<< ENTRY" if idx.strftime('%H:%M') == '13:25' else ""
    sm_bull = f"Bull={row['SM Bull']:.4f}" if row['SM Bull'] != 0 else ""
    sm_bear = f"Bear={row['SM Bear']:.4f}" if row['SM Bear'] != 0 else ""
    long_sig = " [LONG SIGNAL]" if row['Long'] != 0 else ""
    print(f"  {idx.strftime('%H:%M')} C={row['close']:.2f} SM_Net={row['SM Net Index']:.6f} {sm_bull} {sm_bear} RSI5m={row['rsi_5m']:.2f}{long_sig}{marker}")

# DETAILED SM tracking during Trade 2
print("\n--- DETAILED SM Net Index tracking during Trade 2 (every bar) ---")
t2_bars = feb13_full.loc[t2_entry:t2_exit]
sm_entry_val = None
sm_peak = None
sm_peak_time = None
first_decline_time = None
first_decline_val = None

print(f"  {'Time':>5s} {'Close':>8s} {'SM_Net':>10s} {'SM_Chg':>8s} {'RSI5m':>6s} {'Bull':>8s} {'Bear':>8s} {'Note'}")
print(f"  {'-'*5} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*20}")

for i, (idx, row) in enumerate(t2_bars.iterrows()):
    sm_val = row['SM Net Index']
    if i == 0:
        sm_entry_val = sm_val
        sm_peak = sm_val
        sm_peak_time = idx
        sm_chg = 0
    else:
        sm_chg = sm_val - sm_entry_val
        if sm_val > sm_peak:
            sm_peak = sm_val
            sm_peak_time = idx
        if first_decline_time is None and sm_val < sm_entry_val:
            first_decline_time = idx
            first_decline_val = sm_val

    bull_str = f"{row['SM Bull']:.4f}" if row['SM Bull'] != 0 else ""
    bear_str = f"{row['SM Bear']:.4f}" if row['SM Bear'] != 0 else ""

    note = ""
    if i == 0:
        note = "ENTRY"
    elif idx == t2_bars.index[-1]:
        note = "EXIT"
    elif row['Long'] != 0:
        note = "LONG SIGNAL"
    elif row['Short'] != 0:
        note = "SHORT SIGNAL"
    elif sm_val < 0 and t2_bars.iloc[max(0,i-1)]['SM Net Index'] >= 0:
        note = "SM CROSSES ZERO!"

    # Print every 3 bars or if notable
    if i % 3 == 0 or note or bull_str or bear_str:
        print(f"  {idx.strftime('%H:%M')} {row['close']:>8.2f} {sm_val:>10.6f} {sm_chg:>+8.6f} {row['rsi_5m']:>6.2f} {bull_str:>8s} {bear_str:>8s} {note}")

print(f"\n  SM Net at entry: {sm_entry_val:.6f}")
print(f"  SM Net peak: {sm_peak:.6f} at {sm_peak_time.strftime('%H:%M')}")
if first_decline_time:
    print(f"  First time SM below entry level: {first_decline_time.strftime('%H:%M')} (SM={first_decline_val:.6f})")
print(f"  SM Net at exit: {t2_bars.iloc[-1]['SM Net Index']:.6f}")

# Price tracking during trade 2
print(f"\n--- Price action during Trade 2 ---")
running_low = 9999
running_high = 0
for i, (idx, row) in enumerate(t2_bars.iterrows()):
    running_high = max(running_high, row['high'])
    running_low = min(running_low, row['low'])
    pnl = (row['close'] - 6894.50) * 5  # $5 per point MES
    if i % 5 == 0 or i == len(t2_bars) - 1:
        print(f"  {idx.strftime('%H:%M')} C={row['close']:.2f} H={row['high']:.2f} L={row['low']:.2f} RunPnL=${pnl:+.2f} RunHigh={running_high:.2f} RunLow={running_low:.2f}")

# ── MISSED SHORT SIGNALS ─────────────────────────────────────────────
print("\n" + "=" * 90)
print("MISSED OR FILTERED SHORT SIGNALS ON FEB 13")
print("=" * 90)

# Check all Short signals
all_shorts_feb13 = feb13_full[feb13_full['Short'] != 0]
print(f"\nTotal Short signals from AlgoAlpha SM indicator: {len(all_shorts_feb13)}")
for idx, row in all_shorts_feb13.iterrows():
    rsi_val = row['rsi_5m']
    rsi_filter = "RSI < 35 (PASS)" if rsi_val < 35 else f"RSI={rsi_val:.2f} > 35 (FILTERED)"
    sm_net = row['SM Net Index']
    sm_filter = "SM < 0 (PASS)" if sm_net < 0 else f"SM={sm_net:.6f} > 0 (FILTERED)"
    print(f"  {idx.strftime('%H:%M')} Close={row['close']:.2f} SM_Net={sm_net:.6f} RSI5m={rsi_val:.2f}")
    print(f"    RSI filter: {rsi_filter}")
    print(f"    SM filter: {sm_filter}")

# Also check where SM flips negative (potential short opportunities)
print("\n--- SM Net Index zero crossings (potential short signals) ---")
prev_sm = None
for idx, row in feb13_full.iterrows():
    sm_val = row['SM Net Index']
    if prev_sm is not None:
        if prev_sm >= 0 and sm_val < 0:
            print(f"  {idx.strftime('%H:%M')} SM crossed BELOW zero: {prev_sm:.6f} -> {sm_val:.6f}, Close={row['close']:.2f}, RSI5m={row['rsi_5m']:.2f}")
        elif prev_sm < 0 and sm_val >= 0:
            print(f"  {idx.strftime('%H:%M')} SM crossed ABOVE zero: {prev_sm:.6f} -> {sm_val:.6f}, Close={row['close']:.2f}, RSI5m={row['rsi_5m']:.2f}")
    prev_sm = sm_val

# ── 5-MIN RSI AROUND TRADE ENTRIES ───────────────────────────────────
print("\n" + "=" * 90)
print("5-MIN RSI(10) AROUND TRADE ENTRIES")
print("=" * 90)

feb13_5m = df_5m[df_5m.index.date == date(2026, 2, 13)]

print("\n--- 5-min RSI around Trade 1 entry (11:00 - 12:30) ---")
rsi_t1 = feb13_5m.loc['2026-02-13 11:00:00':'2026-02-13 12:35:00']
for idx, row in rsi_t1.iterrows():
    marker = ""
    if idx.strftime('%H:%M') == '11:30':
        marker = " <<< T1 ENTRY"
    elif idx.strftime('%H:%M') == '12:25' or idx.strftime('%H:%M') == '12:30':
        marker = " <<< T1 EXIT AREA"
    rsi_level = ""
    if row['rsi'] > 65:
        rsi_level = " [OVERBOUGHT >65]"
    elif row['rsi'] < 35:
        rsi_level = " [OVERSOLD <35]"
    print(f"  {idx.strftime('%H:%M')} C={row['close']:.2f} RSI={row['rsi']:.2f}{rsi_level}{marker}")

print("\n--- 5-min RSI around Trade 2 entry (13:00 - 15:00) ---")
rsi_t2 = feb13_5m.loc['2026-02-13 13:00:00':'2026-02-13 15:05:00']
for idx, row in rsi_t2.iterrows():
    marker = ""
    if idx.strftime('%H:%M') == '13:25':
        marker = " <<< T2 ENTRY"
    elif idx.strftime('%H:%M') == '14:55' or idx.strftime('%H:%M') == '15:00':
        marker = " <<< T2 EXIT AREA"
    rsi_level = ""
    if row['rsi'] > 65:
        rsi_level = " [OVERBOUGHT >65]"
    elif row['rsi'] < 35:
        rsi_level = " [OVERSOLD <35]"
    print(f"  {idx.strftime('%H:%M')} C={row['close']:.2f} RSI={row['rsi']:.2f}{rsi_level}{marker}")

# ── MES vs MNQ COMPARISON ────────────────────────────────────────────
print("\n" + "=" * 90)
print("MES BEHAVIOR ON FEB 13 - GENERAL OBSERVATIONS")
print("=" * 90)

# Intraday stats
feb13_rth = feb13_full.between_time('09:30', '16:00')
if len(feb13_rth) > 0:
    rth_open = feb13_rth['open'].iloc[0]
    rth_close = feb13_rth['close'].iloc[-1]
    rth_high = feb13_rth['high'].max()
    rth_low = feb13_rth['low'].min()
    rth_range = rth_high - rth_low
    print(f"RTH Open: {rth_open:.2f}")
    print(f"RTH Close: {rth_close:.2f}")
    print(f"RTH High: {rth_high:.2f}")
    print(f"RTH Low: {rth_low:.2f}")
    print(f"RTH Range: {rth_range:.2f} pts")
    print(f"RTH Change: {rth_close - rth_open:+.2f} pts ({(rth_close - rth_open) / rth_open * 100:+.3f}%)")

# Volatility analysis
feb13_full['bar_range'] = feb13_full['high'] - feb13_full['low']
print(f"\n1-min bar stats:")
print(f"  Avg range: {feb13_full['bar_range'].mean():.2f} pts")
print(f"  Max range: {feb13_full['bar_range'].max():.2f} pts")
print(f"  Median range: {feb13_full['bar_range'].median():.2f} pts")

# Price action summary - simple swing point detection without scipy
print(f"\nKey price levels on Feb 13:")
close_arr = feb13_rth['close'].values
times_arr = feb13_rth.index
order = 10
print(f"  Major swing highs:")
for i in range(order, len(close_arr) - order):
    if close_arr[i] == max(close_arr[i-order:i+order+1]):
        print(f"    {times_arr[i].strftime('%H:%M')} = {close_arr[i]:.2f}")
print(f"  Major swing lows:")
for i in range(order, len(close_arr) - order):
    if close_arr[i] == min(close_arr[i-order:i+order+1]):
        print(f"    {times_arr[i].strftime('%H:%M')} = {close_arr[i]:.2f}")

# MES point value comparison
print(f"\nMES contract specs:")
print(f"  Point value: $5 per point")
print(f"  Trade 1: +18.50 pts = +$92.50 gross")
print(f"  Trade 2: -27.75 pts = -$138.75 gross")
print(f"  Net: -9.25 pts = -$46.25 gross")

# Compare to MNQ (from memory: MNQ is $2 per point)
print(f"\nMNQ equivalent (from memory - $2/pt, ~4x MES range):")
print(f"  MNQ typically moves ~4x more points than MES for same % move")
print(f"  MES 18.50 pts ~ MNQ 74 pts ($148)")
print(f"  MES 27.75 pts ~ MNQ 111 pts ($222)")

# ── WHAT WENT WRONG WITH TRADE 2 ─────────────────────────────────────
print("\n" + "=" * 90)
print("TRADE 2 DEEP DIVE: WHAT WENT WRONG?")
print("=" * 90)

# SM Net trajectory
t2_sm = t2_bars['SM Net Index']
print(f"\nSM Net Index trajectory:")
print(f"  At entry (13:25): {t2_sm.iloc[0]:.6f}")

# Find when SM peaked and started declining
sm_vals = list(t2_sm)
sm_times = list(t2_bars.index)
max_sm = max(sm_vals)
max_sm_idx = sm_vals.index(max_sm)
print(f"  Peak: {max_sm:.6f} at {sm_times[max_sm_idx].strftime('%H:%M')}")

# Find consecutive decline
decline_start = None
for i in range(1, len(sm_vals)):
    if sm_vals[i] < sm_vals[i-1]:
        # Check if this is start of sustained decline (3+ bars)
        if i + 2 < len(sm_vals) and sm_vals[i+1] < sm_vals[i] and sm_vals[i+2] < sm_vals[i+1]:
            decline_start = i
            break

if decline_start:
    print(f"  Sustained decline started: {sm_times[decline_start].strftime('%H:%M')} (SM={sm_vals[decline_start]:.6f})")
    price_at_decline = t2_bars.iloc[decline_start]['close']
    print(f"  Price at decline start: {price_at_decline:.2f} (from entry {6894.50:.2f}, change: {price_at_decline - 6894.50:+.2f} pts)")

# Find when SM went negative
for i, (idx, row) in enumerate(t2_bars.iterrows()):
    if row['SM Net Index'] < 0 and (i == 0 or t2_bars.iloc[i-1]['SM Net Index'] >= 0):
        print(f"  SM crossed zero at: {idx.strftime('%H:%M')} (SM={row['SM Net Index']:.6f}, Close={row['close']:.2f})")

# Price at various SM levels
print(f"\n  Price vs SM decline:")
entry_sm = t2_sm.iloc[0]
thresholds = [entry_sm * 0.75, entry_sm * 0.5, entry_sm * 0.25, 0, -entry_sm * 0.25]
for thresh in thresholds:
    for i, (idx, row) in enumerate(t2_bars.iterrows()):
        if row['SM Net Index'] <= thresh:
            loss = (row['close'] - 6894.50)
            print(f"    SM <= {thresh:.4f}: {idx.strftime('%H:%M')} Close={row['close']:.2f} Loss={loss:+.2f} pts (${loss*5:+.2f})")
            break

# v9.4 stop loss check
print(f"\n  Would v9.4 50pt stop have helped?")
for i, (idx, row) in enumerate(t2_bars.iterrows()):
    if row['low'] <= 6894.50 - 50:
        print(f"    50pt stop at {6894.50 - 50:.2f} would have triggered at {idx.strftime('%H:%M')} (Low={row['low']:.2f})")
        # Approximate exit price
        approx_exit = 6894.50 - 50
        loss = (approx_exit - 6894.50) * 5
        print(f"    Approximate loss: {loss:+.2f} (vs actual -$139.79)")
        break
else:
    print(f"    50pt stop at {6894.50 - 50:.2f} = {6844.50:.2f} was NOT hit (actual low in trade: {t2_bars['low'].min():.2f})")
    loss_pts = 6894.50 - t2_bars['low'].min()
    print(f"    Max adverse excursion: {loss_pts:.2f} pts")

# 10pt stop for MES?
print(f"\n  Would a 10pt MES stop have helped?")
for i, (idx, row) in enumerate(t2_bars.iterrows()):
    if row['low'] <= 6894.50 - 10:
        print(f"    10pt stop at {6894.50 - 10:.2f} would have triggered at {idx.strftime('%H:%M')} (Low={row['low']:.2f})")
        loss = -10 * 5
        print(f"    Loss: ${loss:.2f} (vs actual -$139.79)")
        break

# 15pt stop
print(f"\n  Would a 15pt MES stop have helped?")
for i, (idx, row) in enumerate(t2_bars.iterrows()):
    if row['low'] <= 6894.50 - 15:
        print(f"    15pt stop at {6894.50 - 15:.2f} would have triggered at {idx.strftime('%H:%M')} (Low={row['low']:.2f})")
        loss = -15 * 5
        print(f"    Loss: ${loss:.2f} (vs actual -$139.79)")
        break

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
