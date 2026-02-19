#!/usr/bin/env python3
"""Compare MES vs MNQ behavior on Feb 13."""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta, date

ET = timezone(timedelta(hours=-5))

# Load MES
mes = pd.read_csv("/Users/jasongeorge/Downloads/CME_MINI_MES1!, 1_37616.csv")
mes['datetime'] = pd.to_datetime(mes['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
mes = mes.set_index('datetime')
mes_feb13 = mes[mes.index.date == date(2026, 2, 13)]

# Load MNQ
mnq = pd.read_csv("/Users/jasongeorge/Desktop/NQ trading/backtesting_engine/data/CME_MINI_MNQ1!, 1_8835e.csv")
mnq['datetime'] = pd.to_datetime(mnq['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
mnq = mnq.set_index('datetime')
mnq_feb13 = mnq[mnq.index.date == date(2026, 2, 13)]

print("=" * 90)
print("MES vs MNQ COMPARISON - Feb 13, 2026")
print("=" * 90)

# RTH comparison
mes_rth = mes_feb13.between_time('09:30', '16:00')
mnq_rth = mnq_feb13.between_time('09:30', '16:00')

# Common time overlap
overlap_start = max(mes_rth.index[0], mnq_rth.index[0])
overlap_end = min(mes_rth.index[-1], mnq_rth.index[-1])

print(f"\nMES RTH: {mes_rth.index[0].strftime('%H:%M')} to {mes_rth.index[-1].strftime('%H:%M')} ({len(mes_rth)} bars)")
print(f"MNQ RTH: {mnq_rth.index[0].strftime('%H:%M')} to {mnq_rth.index[-1].strftime('%H:%M')} ({len(mnq_rth)} bars)")

# Price stats
for name, data in [("MES", mes_rth), ("MNQ", mnq_rth)]:
    o = data['open'].iloc[0]
    c = data['close'].iloc[-1]
    h = data['high'].max()
    l = data['low'].min()
    rng = h - l
    pct = (c - o) / o * 100
    print(f"\n{name}:")
    print(f"  Open: {o:.2f}, Close: {c:.2f}, Change: {c-o:+.2f} ({pct:+.3f}%)")
    print(f"  High: {h:.2f}, Low: {l:.2f}, Range: {rng:.2f} pts")
    avg_range = (data['high'] - data['low']).mean()
    print(f"  Avg 1-min bar range: {avg_range:.2f} pts")

# SM Net Index comparison at key times
print("\n" + "=" * 90)
print("SM NET INDEX COMPARISON AT KEY TIMES")
print("=" * 90)

key_times = [
    ('11:29', 'T1 Signal'),
    ('11:30', 'T1 Entry'),
    ('12:00', 'T1 Mid'),
    ('12:28', 'T1 SM Flip (bear)'),
    ('12:29', 'T1 Exit'),
    ('13:24', 'T2 Signal'),
    ('13:25', 'T2 Entry'),
    ('13:59', 'T2 SM Peak'),
    ('14:30', 'T2 Price declining'),
    ('14:56', 'T2 SM Flip (bear)'),
    ('14:57', 'T2 Exit'),
]

print(f"\n  {'Time':>5s} {'Event':<22s} {'MES_SM':>10s} {'MNQ_SM':>10s} {'MES_Close':>10s} {'MNQ_Close':>10s}")
print(f"  {'-'*5} {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for time_str, event in key_times:
    ts = pd.Timestamp(f'2026-02-13 {time_str}:00', tz='US/Eastern')
    mes_row = mes_feb13.loc[ts] if ts in mes_feb13.index else None
    mnq_row = mnq_feb13.loc[ts] if ts in mnq_feb13.index else None

    mes_sm = f"{mes_row['SM Net Index']:.6f}" if mes_row is not None and not pd.isna(mes_row['SM Net Index']) else "N/A"
    mnq_sm = f"{mnq_row['SM Net Index']:.6f}" if mnq_row is not None and not pd.isna(mnq_row['SM Net Index']) else "N/A"
    mes_c = f"{mes_row['close']:.2f}" if mes_row is not None else "N/A"
    mnq_c = f"{mnq_row['close']:.2f}" if mnq_row is not None else "N/A"

    print(f"  {time_str:>5s} {event:<22s} {mes_sm:>10s} {mnq_sm:>10s} {mes_c:>10s} {mnq_c:>10s}")

# Signal comparison
print("\n" + "=" * 90)
print("SIGNAL COMPARISON")
print("=" * 90)

mes_longs = mes_feb13[mes_feb13['Long'] != 0]
mes_shorts = mes_feb13[mes_feb13['Short'] != 0]
mnq_longs = mnq_feb13[(mnq_feb13['Long'] != 0) & (~mnq_feb13['Long'].isna())]
mnq_shorts = mnq_feb13[(mnq_feb13['Short'] != 0) & (~mnq_feb13['Short'].isna())]

print(f"\nMES Long signals: {len(mes_longs)}")
for idx, row in mes_longs.iterrows():
    print(f"  {idx.strftime('%H:%M')} Close={row['close']:.2f} SM={row['SM Net Index']:.6f}")

print(f"\nMNQ Long signals: {len(mnq_longs)}")
for idx, row in mnq_longs.iterrows():
    print(f"  {idx.strftime('%H:%M')} Close={row['close']:.2f} SM={row['SM Net Index']:.6f}")

print(f"\nMES Short signals: {len(mes_shorts)}")
for idx, row in mes_shorts.iterrows():
    print(f"  {idx.strftime('%H:%M')} Close={row['close']:.2f} SM={row['SM Net Index']:.6f}")

print(f"\nMNQ Short signals: {len(mnq_shorts)}")
for idx, row in mnq_shorts.iterrows():
    print(f"  {idx.strftime('%H:%M')} Close={row['close']:.2f} SM={row['SM Net Index']:.6f}")

# Correlation analysis during the trade windows
print("\n" + "=" * 90)
print("PRICE CORRELATION DURING TRADE WINDOWS")
print("=" * 90)

# For overlapping 1-min bars, compute % change correlation
common_idx = mes_feb13.index.intersection(mnq_feb13.index)
mes_common = mes_feb13.loc[common_idx]['close'].pct_change()
mnq_common = mnq_feb13.loc[common_idx]['close'].pct_change()
overall_corr = mes_common.corr(mnq_common)
print(f"\nOverall 1-min close-to-close correlation: {overall_corr:.4f}")

# During Trade 2
t2_start = pd.Timestamp('2026-02-13 13:25:00', tz='US/Eastern')
t2_end = pd.Timestamp('2026-02-13 14:57:00', tz='US/Eastern')
t2_common = common_idx[(common_idx >= t2_start) & (common_idx <= t2_end)]
if len(t2_common) > 5:
    mes_t2 = mes_feb13.loc[t2_common]['close'].pct_change()
    mnq_t2 = mnq_feb13.loc[t2_common]['close'].pct_change()
    t2_corr = mes_t2.corr(mnq_t2)
    print(f"Trade 2 window correlation: {t2_corr:.4f}")

    # Compare SM Net Index during Trade 2
    print(f"\n--- SM Net Index divergence during Trade 2 ---")
    print(f"  {'Time':>5s} {'MES_SM':>10s} {'MNQ_SM':>10s} {'Diff':>10s} {'MES_Close':>10s} {'MNQ_Close':>10s}")
    for i, idx in enumerate(t2_common):
        if i % 5 == 0 or idx == t2_common[-1]:
            mes_sm = mes_feb13.loc[idx, 'SM Net Index']
            mnq_sm = mnq_feb13.loc[idx, 'SM Net Index']
            diff = mes_sm - mnq_sm if not (pd.isna(mes_sm) or pd.isna(mnq_sm)) else float('nan')
            print(f"  {idx.strftime('%H:%M')} {mes_sm:>10.6f} {mnq_sm:>10.6f} {diff:>+10.6f} {mes_feb13.loc[idx, 'close']:>10.2f} {mnq_feb13.loc[idx, 'close']:>10.2f}")

# MNQ behavior during Trade 2 window
print(f"\n--- MNQ price during Trade 2 window ---")
mnq_t2_data = mnq_feb13.loc[t2_start:t2_end]
if len(mnq_t2_data) > 0:
    mnq_t2_entry = mnq_t2_data['close'].iloc[0]
    mnq_t2_high = mnq_t2_data['high'].max()
    mnq_t2_low = mnq_t2_data['low'].min()
    mnq_t2_exit = mnq_t2_data['close'].iloc[-1]
    print(f"  MNQ entry close: {mnq_t2_entry:.2f}")
    print(f"  MNQ exit close: {mnq_t2_exit:.2f}")
    print(f"  MNQ change: {mnq_t2_exit - mnq_t2_entry:+.2f} pts ({(mnq_t2_exit - mnq_t2_entry) / mnq_t2_entry * 100:+.3f}%)")
    print(f"  MNQ high: {mnq_t2_high:.2f} (+{mnq_t2_high - mnq_t2_entry:.2f} pts)")
    print(f"  MNQ low: {mnq_t2_low:.2f} ({mnq_t2_low - mnq_t2_entry:+.2f} pts)")

    # MES comparison
    mes_t2_data = mes_feb13.loc[t2_start:t2_end]
    mes_t2_entry = mes_t2_data['close'].iloc[0]
    mes_t2_high = mes_t2_data['high'].max()
    mes_t2_low = mes_t2_data['low'].min()
    mes_t2_exit = mes_t2_data['close'].iloc[-1]
    print(f"\n  MES entry close: {mes_t2_entry:.2f}")
    print(f"  MES exit close: {mes_t2_exit:.2f}")
    print(f"  MES change: {mes_t2_exit - mes_t2_entry:+.2f} pts ({(mes_t2_exit - mes_t2_entry) / mes_t2_entry * 100:+.3f}%)")
    print(f"  MES high: {mes_t2_high:.2f} (+{mes_t2_high - mes_t2_entry:.2f} pts)")
    print(f"  MES low: {mes_t2_low:.2f} ({mes_t2_low - mes_t2_entry:+.2f} pts)")

    # Percentage move comparison
    mes_pct_range = (mes_t2_high - mes_t2_low) / mes_t2_entry * 100
    mnq_pct_range = (mnq_t2_high - mnq_t2_low) / mnq_t2_entry * 100
    print(f"\n  MES % range during trade: {mes_pct_range:.3f}%")
    print(f"  MNQ % range during trade: {mnq_pct_range:.3f}%")
    print(f"  MNQ/MES range ratio: {mnq_pct_range/mes_pct_range:.2f}x" if mes_pct_range > 0 else "")

print("\n" + "=" * 90)
print("COMPARISON COMPLETE")
print("=" * 90)
