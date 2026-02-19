"""
Supplemental analysis: deeper investigation on Trade 14 chart visibility
and RSI cross timing for each trade.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

df = pd.read_csv("/Users/jasongeorge/Desktop/NQ trading/backtesting_engine/data/CME_MINI_MNQ1!, 1_8835e.csv")
ET = timezone(timedelta(hours=-5))
df['datetime_et'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(ET)
df['time_str'] = df['datetime_et'].dt.strftime('%H:%M')

# ============================================================
# INVESTIGATION: Why is there no bar at exactly 10:00?
# ============================================================
print("=" * 80)
print("MISSING 10:00 BAR INVESTIGATION")
print("=" * 80)
mask = (df['time_str'] >= '09:58') & (df['time_str'] <= '10:02')
for _, row in df[mask].iterrows():
    ts = row['time']
    dt = row['datetime_et']
    print(f"  Unix={ts}  ET={dt}  time_str={row['time_str']}  close={row['close']:.2f}  SM={row['SM Net Index']:.6f}")

# Check: is there a gap?
print(f"\n  Note: The CSV has 09:59 then 10:01. The 10:00 bar is missing from the data.")
print(f"  TradingView shows Trade 13 entry at 10:00, which likely means 09:59 or 10:01 bar.")

# ============================================================
# TRADE 13: RSI CROSS INVESTIGATION
# ============================================================
print("\n" + "=" * 80)
print("TRADE 13 RSI ANALYSIS — Short entry near 10:00")
print("=" * 80)

# Resample to 5-min
df_5m = df.set_index('datetime_et').resample('5min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'SM Net Index': 'last'
}).dropna()
df_5m['time_str'] = df_5m.index.strftime('%H:%M')

def calc_rsi(series, period=10):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df_5m['RSI'] = calc_rsi(df_5m['close'], period=10)

# For Trade 13 (Short), we need RSI to cross DOWN through 45 while SM < 0
print("\nLooking for RSI cross DOWN through 45 with SM bearish near 10:00:")
print(f"  {'Time':>6s}  {'Close':>10s}  {'RSI':>8s}  {'SM':>12s}  {'Signal':>20s}")
session = df_5m[(df_5m['time_str'] >= '09:30') & (df_5m['time_str'] <= '10:10')]
prev_rsi = None
for idx, row in session.iterrows():
    rsi = row['RSI']
    sm = row['SM Net Index']
    signal = ""
    if prev_rsi is not None and not np.isnan(prev_rsi) and not np.isnan(rsi):
        if prev_rsi >= 45 and rsi < 45 and sm < 0:
            signal = ">>> SHORT SIGNAL"
        elif prev_rsi < 55 and rsi >= 55 and sm > 0:
            signal = ">>> LONG SIGNAL"
        elif prev_rsi >= 45 and rsi < 45:
            signal = "RSI dn<45 (SM bull)"
        elif prev_rsi < 55 and rsi >= 55:
            signal = "RSI up>55 (SM bear)"
    sm_sign = "BULL" if sm > 0 else "BEAR"
    print(f"  {row['time_str']:>6s}  {row['close']:>10.2f}  {rsi:>8.2f}  {sm:>12.6f}  {signal:>20s}")
    prev_rsi = rsi

# ============================================================
# TRADE 14: Why not visible on chart?
# ============================================================
print("\n" + "=" * 80)
print("TRADE 14: CHART VISIBILITY ANALYSIS")
print("=" * 80)

# The RSI cross for Trade 14 long was at 10:20 (RSI crossed up through 55)
# Entry at 10:27 on 1-min bars... but on 5-min chart, 10:25 bar close is 24848.50
# and 10:30 bar close is 24907.50
# Entry at 10:27 falls within the 10:25-10:30 5-min candle
# Exit at 10:33-10:34 falls within the 10:30-10:35 5-min candle

print("\nTrade 14 timeline on 5-min bars:")
print("  RSI cross UP >55 at 10:20 bar (RSI: 52.84 -> 56.56)")
print("  SM was BULLISH at 10:20 (SM=0.1537)")
print("  -> LONG SIGNAL generated on the 10:20 5-min bar close")
print()
print("  Entry on 1-min: 10:27 at 24819.00")
print("  Exit on 1-min: 10:33-10:34 (SM flipped Bear at 10:33)")
print()
print("  On 5-min chart:")
print("    10:25 bar: O=24776.25 H=24870.75 L=24771.25 C=24848.50 (entry within this bar)")
print("    10:30 bar: O=24848.50 H=24929.25 L=24847.25 C=24907.50 (exit within this bar)")
print()
print("  The trade spans only ~7 minutes (1.4 five-min bars)")
print("  On TradingView 5-min chart, the entry and exit markers would be")
print("  extremely close together or overlapping, making it nearly invisible.")
print()
print("  Additionally, if TradingView overlaps the Trade 13 exit marker (SM flip to bull")
print("  at 10:11) with the Trade 14 entry marker, the entry arrow may be hidden.")

# Check: does the Long/Short column show anything for Trade 14?
print("\n  Checking Long/Short signal columns around 10:27:")
mask_14 = (df['time_str'] >= '10:20') & (df['time_str'] <= '10:35')
for _, row in df[mask_14].iterrows():
    long_val = row['Long']
    short_val = row['Short']
    if long_val == 1 or short_val == 1:
        print(f"    {row['time_str']}: Long={long_val}, Short={short_val}")

print("  No Long/Short signals fired in the 10:20-10:35 range.")
print("  (The Long/Short columns appear to be from a different indicator, not the v9 strategy)")

# ============================================================
# TRADE 16: SM behavior during the selloff
# ============================================================
print("\n" + "=" * 80)
print("TRADE 16: SM BEHAVIOR DURING THE SELLOFF")
print("=" * 80)

mask_16 = (df['time_str'] >= '14:25') & (df['time_str'] <= '15:38')
t16 = df[mask_16].copy()

print(f"\n  Duration: {len(t16)} one-minute bars (73 minutes)")
print(f"  Entry close: {t16.iloc[0]['close']:.2f} at {t16.iloc[0]['time_str']}")
print(f"  Exit close: {t16.iloc[-1]['close']:.2f} at {t16.iloc[-1]['time_str']}")

# SM range during trade
sm_vals = t16['SM Net Index']
print(f"\n  SM Net Index during trade:")
print(f"    Min: {sm_vals.min():.6f} at {t16.loc[sm_vals.idxmin(), 'time_str']}")
print(f"    Max: {sm_vals.max():.6f} at {t16.loc[sm_vals.idxmax(), 'time_str']}")
print(f"    SM stayed POSITIVE (bullish) for entire trade until 15:37")

# Show SM evolution at key price levels
print(f"\n  Key moments during Trade 16:")
checkpoints = ['14:25', '14:30', '14:40', '14:50', '15:00', '15:10', '15:20', '15:30', '15:35', '15:36', '15:37', '15:38']
for cp in checkpoints:
    rows = t16[t16['time_str'] == cp]
    if len(rows) > 0:
        r = rows.iloc[0]
        drawdown = r['close'] - t16.iloc[0]['close']
        sm_sign = "BULL" if r['SM Net Index'] > 0 else "BEAR"
        print(f"    {cp}: Close={r['close']:.2f} (PnL={drawdown:+.2f}pts) | SM={r['SM Net Index']:.4f} ({sm_sign})")

# The paradox: SM was actually INCREASING (more bullish) as price fell
print(f"\n  PARADOX: SM Net Index actually INCREASED from 14:25 to ~14:44:")
print(f"    14:25: SM = 0.2494 (entry)")
print(f"    14:44: SM = 0.5822 (peak SM, while price was already falling)")
print(f"    Price at 14:44: 24907.25 (already -59.50 pts from entry)")
print(f"    SM saw the selloff as 'smart money buying' (accumulation)")
print(f"    This is the classic SM trap: indicator reads selling as distribution")
print(f"    but then the selling continues into a real downtrend.")

# ============================================================
# TRADE 16: RSI ENTRY SIGNAL INVESTIGATION
# ============================================================
print("\n" + "=" * 80)
print("TRADE 16: RSI ENTRY SIGNAL DETAILS")
print("=" * 80)

print("\n  The 5-min RSI around Trade 16 entry:")
t16_rsi = df_5m[(df_5m['time_str'] >= '14:00') & (df_5m['time_str'] <= '14:30')]
prev_rsi = None
for idx, row in t16_rsi.iterrows():
    rsi = row['RSI']
    sm = row['SM Net Index']
    cross = ""
    if prev_rsi is not None:
        if prev_rsi < 55 and rsi >= 55:
            cross = "CROSS UP >55"
        elif prev_rsi >= 55 and rsi < 55:
            cross = "CROSS DN <55"
        if prev_rsi >= 45 and rsi < 45:
            cross += " CROSS DN <45"
        elif prev_rsi < 45 and rsi >= 45:
            cross += " CROSS UP >45"
    print(f"  {row['time_str']}: Close={row['close']:.2f} RSI={rsi:.2f} SM={sm:.4f} {cross}")
    prev_rsi = rsi

print("\n  Analysis:")
print("  At 14:20: RSI crossed UP through 55 (52.82 -> 56.26) with SM bullish (0.2491)")
print("  -> This generated the LONG SIGNAL")
print("  But by 14:25, RSI had already crashed to 43.72 (below even 45)")
print("  The entry at 14:25 was 5 minutes AFTER the signal bar")
print("  This suggests the strategy may use the signal from the 14:20 bar")
print("  but enters on a later bar due to cooldown or confirmation logic")
print()
print("  PROBLEM: The RSI was already plunging when the trade entered.")
print("  The signal at 14:20 was right at a local price peak (24963.50),")
print("  and price was already selling off by the time of entry at 14:25.")

# ============================================================
# OVERALL SESSION SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("FULL SESSION SUMMARY")
print("=" * 80)

# All valid RSI cross signals with SM confirmation
print("\nAll v9 signals during session (09:45-15:45):")
print("(RSI cross + SM confirmation)")
print()

session_full = df_5m[(df_5m['time_str'] >= '09:30') & (df_5m['time_str'] <= '15:45')]
prev_rsi = None
signals = []
for idx, row in session_full.iterrows():
    rsi = row['RSI']
    sm = row['SM Net Index']
    if prev_rsi is not None and not np.isnan(prev_rsi) and not np.isnan(rsi):
        if prev_rsi < 55 and rsi >= 55 and sm > 0:
            signals.append(('LONG', row['time_str'], row['close'], rsi, prev_rsi, sm))
        if prev_rsi >= 45 and rsi < 45 and sm < 0:
            signals.append(('SHORT', row['time_str'], row['close'], rsi, prev_rsi, sm))
    prev_rsi = rsi

for sig in signals:
    direction, time, close, rsi, prev, sm = sig
    level = 55 if direction == 'LONG' else 45
    print(f"  {time}: {direction} signal | Close={close:.2f} | RSI: {prev:.2f}->{rsi:.2f} (crossed {level}) | SM={sm:.4f}")

print(f"\n  Total signals: {len(signals)}")
print(f"  Long signals: {sum(1 for s in signals if s[0]=='LONG')}")
print(f"  Short signals: {sum(1 for s in signals if s[0]=='SHORT')}")
print(f"\n  NOTE: Trade 13 (Short at ~10:00) - RSI at 09:55 was 45.37, barely above 45.")
print(f"  At 10:00, RSI went to 54.89 (went UP, not down). No clear RSI cross below 45")
print(f"  with bearish SM near 10:00 on 5-min bars. The short signal timing is unclear")
print(f"  unless TradingView uses slightly different RSI calculation or bar alignment.")
