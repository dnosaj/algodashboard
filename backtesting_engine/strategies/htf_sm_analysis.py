"""
HTF SM Divergence Analysis
--------------------------
Question: Can 15-min SM direction filter improve 1-min/5-min strategy?
Specifically: Does 15-min SM going bearish/neutral while 1-min SM is still
bullish predict profit-taking selloffs like the Feb 13 -$455 trade?

Approach:
1. Load 15-min data with AlgoAlpha SM columns
2. Load 5-min backtest data
3. For each v9 trade, check what 15-min SM was doing at entry
4. Split trades by HTF SM agreement vs disagreement
5. Compare win rate and PF for each group
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# Load 15-min data
# ============================================================================
df15 = pd.read_csv("/Users/jasongeorge/Downloads/CME_MINI_MNQ1!, 15_d3b7c.csv")
df15['datetime'] = pd.to_datetime(df15['time'], unit='s')
df15['datetime'] = df15['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
df15 = df15.set_index('datetime')

# SM Net Index from 15-min AlgoAlpha
df15['sm15'] = df15['SM Net Index'].astype(float)
df15['sm15_bull'] = df15['sm15'] > 0
df15['sm15_bear'] = df15['sm15'] < 0

print("=" * 70)
print("15-MIN SM DATA OVERVIEW")
print("=" * 70)
print(f"Date range: {df15.index[0]} to {df15.index[-1]}")
print(f"Bars: {len(df15)}")
print(f"SM bullish bars: {df15['sm15_bull'].sum()} ({df15['sm15_bull'].mean()*100:.1f}%)")
print(f"SM bearish bars: {df15['sm15_bear'].sum()} ({df15['sm15_bear'].mean()*100:.1f}%)")
print(f"SM neutral (=0): {(df15['sm15'] == 0).sum()}")
print()

# ============================================================================
# Look at Feb 13 specifically
# ============================================================================
feb13 = df15[df15.index.date == pd.Timestamp('2026-02-13').date()]
print("=" * 70)
print("FEB 13 - 15-MIN SM TIMELINE")
print("=" * 70)
for idx, row in feb13.iterrows():
    direction = "BULL" if row['sm15'] > 0 else ("BEAR" if row['sm15'] < 0 else "FLAT")
    t = idx.strftime('%H:%M')
    bar = "#" * int(abs(row['sm15']) * 100)
    sign = "+" if row['sm15'] > 0 else ""
    print(f"  {t}  SM15: {sign}{row['sm15']:+.4f}  {direction:4s}  px: {row['close']:.0f}  {bar}")

print()
print("Trade 16 entered long at 13:25 on 1-min chart.")
print("Question: What was 15-min SM doing around 13:00-14:00?")
print()

# ============================================================================
# Load 5-min data and run v9 backtest to get trade list
# ============================================================================
import os
import sys
sys.path.insert(0, '/Users/jasongeorge/Desktop/NQ trading/backtesting_engine')

# Load 5-min data with SM
df5 = pd.read_csv("/Users/jasongeorge/Desktop/NQ trading/backtesting_engine/data/CME_MINI_MNQ1!, 5_46a9d.csv")
df5['datetime'] = pd.to_datetime(df5['time'], unit='s')
df5['datetime'] = df5['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
df5 = df5.set_index('datetime')

# Filter to Jan 19 - Feb 13
df5 = df5[(df5.index >= '2026-01-19') & (df5.index <= '2026-02-14')]

# SM from AlgoAlpha prebaked columns
if 'SM Net Index' in df5.columns:
    df5['sm_net'] = df5['SM Net Index'].astype(float)
elif 'Net Buy Line' in df5.columns:
    df5['sm_net'] = df5['Net Buy Line'].fillna(0) + df5['Net Sell Line'].fillna(0)
else:
    print("ERROR: No SM columns found in 5-min data")
    print("Columns:", df5.columns.tolist())
    sys.exit(1)

# RSI
def compute_rsi(series, period=10):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df5['rsi'] = compute_rsi(df5['close'], 10)

# Strategy parameters
RSI_BUY = 55
RSI_SELL = 45
SM_THRESHOLD = 0.0
COOLDOWN = 3  # 5-min bars
ENTRY_START = 10 * 60  # 10:00 ET in minutes from midnight
ENTRY_END = 15 * 60 + 45  # 15:45 ET
SESSION_END = 16 * 60  # 16:00 ET

# Run v9 backtest
trades = []
position = None
last_exit_bar = -999
long_used = False
short_used = False

for i in range(1, len(df5)):
    row = df5.iloc[i]
    prev = df5.iloc[i-1]
    dt = df5.index[i]

    minutes = dt.hour * 60 + dt.minute
    in_entry = ENTRY_START <= minutes <= ENTRY_END
    in_session = minutes <= SESSION_END

    sm = prev['sm_net']  # Use previous bar SM (no look-ahead)
    sm_bull = sm > SM_THRESHOLD
    sm_bear = sm < -SM_THRESHOLD

    # SM flip detection
    if i >= 2:
        sm_prev2 = df5.iloc[i-2]['sm_net']
        sm_flipped_bull = sm > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm < 0 and sm_prev2 >= 0
    else:
        sm_flipped_bull = False
        sm_flipped_bear = False

    # Episode reset
    if not sm_bull or sm_flipped_bear:
        long_used = False
    if not sm_bear or sm_flipped_bull:
        short_used = False

    # RSI cross
    rsi_prev = prev['rsi']
    if i >= 2:
        rsi_prev2 = df5.iloc[i-2]['rsi']
    else:
        rsi_prev2 = rsi_prev

    rsi_cross_up = rsi_prev > RSI_BUY and rsi_prev2 <= RSI_BUY
    rsi_cross_down = rsi_prev < RSI_SELL and rsi_prev2 >= RSI_SELL

    cooldown_ok = (i - last_exit_bar) >= COOLDOWN

    # Exit logic
    if position is not None:
        exit_reason = None
        exit_price = row['open']

        # SM flip exit
        if position['side'] == 'long' and sm_flipped_bear:
            exit_reason = 'SM Flip'
        elif position['side'] == 'short' and sm_flipped_bull:
            exit_reason = 'SM Flip'

        # EOD
        if not in_session and position is not None:
            exit_reason = 'EOD'

        if exit_reason:
            if position['side'] == 'long':
                pnl = (exit_price - position['entry_price']) * 5 - 1.04
            else:
                pnl = (position['entry_price'] - exit_price) * 5 - 1.04

            position['exit_price'] = exit_price
            position['exit_time'] = dt
            position['exit_reason'] = exit_reason
            position['pnl'] = pnl
            trades.append(position)
            position = None
            last_exit_bar = i

    # Entry logic
    if position is None and in_entry and cooldown_ok:
        entry_price = row['open']

        if sm_bull and rsi_cross_up and not long_used:
            position = {
                'side': 'long',
                'entry_price': entry_price,
                'entry_time': dt,
                'entry_sm': sm,
                'entry_bar': i
            }
            long_used = True
        elif sm_bear and rsi_cross_down and not short_used:
            position = {
                'side': 'short',
                'entry_price': entry_price,
                'entry_time': dt,
                'entry_sm': sm,
                'entry_bar': i
            }
            short_used = True

print(f"v9 backtest: {len(trades)} trades")
print()

# ============================================================================
# Match each trade with 15-min SM at entry time
# ============================================================================
print("=" * 70)
print("TRADE-BY-TRADE HTF SM ANALYSIS")
print("=" * 70)

# For each trade, find the 15-min bar that contains the entry time
htf_agree = []
htf_disagree = []

for t in trades:
    entry_time = t['entry_time']

    # Find the most recent 15-min bar at or before entry time
    mask = df15.index <= entry_time
    if mask.sum() == 0:
        continue

    bar15 = df15[mask].iloc[-1]
    sm15 = bar15['sm15']
    sm15_dir = "BULL" if sm15 > 0 else ("BEAR" if sm15 < 0 else "FLAT")

    t['sm15'] = sm15
    t['sm15_dir'] = sm15_dir

    # Check agreement
    if t['side'] == 'long':
        agrees = sm15 > 0
    else:
        agrees = sm15 < 0

    t['htf_agrees'] = agrees

    win = t['pnl'] > 0
    marker = "W" if win else "L"
    agree_str = "AGREE" if agrees else "DISAGREE"

    date_str = entry_time.strftime('%m/%d %H:%M')
    print(f"  {date_str}  {t['side']:5s}  SM1m: {t['entry_sm']:+.3f}  SM15m: {sm15:+.4f} ({sm15_dir:4s})  "
          f"{agree_str:9s}  PnL: ${t['pnl']:+.1f}  {marker}")

    if agrees:
        htf_agree.append(t)
    else:
        htf_disagree.append(t)

print()
print("=" * 70)
print("HTF SM AGREEMENT vs DISAGREEMENT")
print("=" * 70)

for label, group in [("HTF AGREES (trade with trend)", htf_agree),
                      ("HTF DISAGREES (trade against trend)", htf_disagree)]:
    if not group:
        print(f"\n{label}: No trades")
        continue

    wins = sum(1 for t in group if t['pnl'] > 0)
    total = len(group)
    gross_profit = sum(t['pnl'] for t in group if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in group if t['pnl'] <= 0))
    net = sum(t['pnl'] for t in group)
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_win = gross_profit / wins if wins > 0 else 0
    avg_loss = gross_loss / (total - wins) if (total - wins) > 0 else 0

    print(f"\n{label}:")
    print(f"  Trades: {total}")
    print(f"  Win Rate: {wins}/{total} = {wins/total*100:.1f}%")
    print(f"  Profit Factor: {pf:.3f}")
    print(f"  Net P&L: ${net:+.2f}")
    print(f"  Avg Win: ${avg_win:.2f}  Avg Loss: ${avg_loss:.2f}")

    # List the disagree trades for inspection
    if "DISAGREES" in label:
        print(f"\n  Individual DISAGREE trades:")
        for t in group:
            win = "W" if t['pnl'] > 0 else "L"
            print(f"    {t['entry_time'].strftime('%m/%d %H:%M')} {t['side']:5s} "
                  f"SM15: {t['sm15']:+.4f}  PnL: ${t['pnl']:+.1f} {win}")

print()
print("=" * 70)
print("WHAT IF WE ONLY TRADE WHEN HTF AGREES?")
print("=" * 70)
all_trades_pnl = sum(t['pnl'] for t in trades)
agree_pnl = sum(t['pnl'] for t in htf_agree)
disagree_pnl = sum(t['pnl'] for t in htf_disagree)
print(f"All trades:      {len(trades)} trades, ${all_trades_pnl:+.2f}")
print(f"HTF agrees only: {len(htf_agree)} trades, ${agree_pnl:+.2f}")
print(f"HTF disagrees:   {len(htf_disagree)} trades, ${disagree_pnl:+.2f}")
print(f"Improvement from filtering: ${agree_pnl - all_trades_pnl:+.2f} "
      f"({len(trades) - len(htf_agree)} trades removed)")

# ============================================================================
# Also test: what about NEUTRAL 15-min SM (close to zero)?
# ============================================================================
print()
print("=" * 70)
print("15-MIN SM STRENGTH AT ENTRY")
print("=" * 70)
print("Do trades with strong 15-min SM do better than weak?")
print()

for threshold in [0.0, 0.01, 0.02, 0.05, 0.10]:
    strong = [t for t in trades if abs(t.get('sm15', 0)) > threshold and t.get('htf_agrees', False)]
    if not strong:
        continue
    wins = sum(1 for t in strong if t['pnl'] > 0)
    net = sum(t['pnl'] for t in strong)
    gp = sum(t['pnl'] for t in strong if t['pnl'] > 0)
    gl = abs(sum(t['pnl'] for t in strong if t['pnl'] <= 0))
    pf = gp / gl if gl > 0 else float('inf')
    print(f"  |SM15| > {threshold:.2f} and agrees: {len(strong)} trades, "
          f"WR {wins/len(strong)*100:.0f}%, PF {pf:.3f}, ${net:+.1f}")
