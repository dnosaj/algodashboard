"""
Compare SM flip exit strategies with threshold=0.15:
  A. Exit at zero-crossing (SM crosses 0) -- what Python engine currently does
  B. Exit at threshold-crossing (SM crosses -0.15 for longs, +0.15 for shorts)

The question: with entry threshold=0.15, should exits also use the threshold?
Threshold exits hold through brief zero-dips; zero exits are faster.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from v10_test_common import (
    compute_smart_money, resample_to_5min, map_5min_rsi_to_1min,
    compute_et_minutes, DATA_DIR,
)

# ── Load and combine data ─────────────────────────────────────────
def load_databento_csv(path):
    df = pd.read_csv(path)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"}, inplace=True)
    df.index = pd.to_datetime(df["time"], unit="s")
    df.index.name = "time"
    df.drop(columns=["time"], inplace=True)
    return df

print("Loading data...")
df1 = load_databento_csv(DATA_DIR / "databento_MNQ_1min_2025-08-17_to_2026-02-13.csv")
df2 = load_databento_csv(DATA_DIR / "databento_MNQ_1min_2026-02-13_to_2026-02-19.csv")
overlap_start = df2.index[0]
df = pd.concat([df1[df1.index < overlap_start], df2])
df = df[~df.index.duplicated(keep='last')]
df.sort_index(inplace=True)
print(f"Combined: {len(df)} bars: {df.index[0]} to {df.index[-1]}")

opens = df["Open"].values
highs = df["High"].values
lows = df["Low"].values
closes = df["Close"].values
volumes = df["Volume"].values
times = df.index

# ── Indicators ─────────────────────────────────────────────────────
print("Computing SM(10/12/200/100)...")
sm = compute_smart_money(closes, volumes,
                         index_period=10, flow_period=12,
                         norm_period=200, ema_len=100)

print("Computing 5-min RSI(8) mapped to 1-min...")
df_5m_tmp = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df_5m_tmp['SM_Net'] = 0.0
df_5m = resample_to_5min(df_5m_tmp)
rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
    times, df_5m.index.values, df_5m['Close'].values, rsi_len=8)

# ── Custom engine with configurable exit mode ──────────────────────
NY_OPEN_ET = 600      # 10:00 AM ET in minutes
NY_LAST_ENTRY_ET = 945  # 15:45 ET
NY_CLOSE_ET = 960     # 16:00 ET

def run_with_exit_mode(sm, opens, highs, lows, closes, times,
                       rsi_5m_curr, rsi_5m_prev,
                       sm_threshold=0.15, rsi_buy=60, rsi_sell=40,
                       cooldown_bars=20, max_loss_pts=50,
                       exit_mode="zero"):
    """
    exit_mode:
      "zero"      = exit when SM crosses zero (current Python engine behavior)
      "threshold"  = exit when SM crosses opposite threshold (-0.15 for longs)
    """
    n = len(opens)
    et_mins = compute_et_minutes(times)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        # Entry conditions (always use threshold)
        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold

        # Zero-crossing (for episode reset)
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # Exit conditions depend on mode
        if exit_mode == "zero":
            long_exit  = sm_flipped_bear   # SM crossed below zero
            short_exit = sm_flipped_bull   # SM crossed above zero
        elif exit_mode == "threshold":
            long_exit  = sm_bear           # SM crossed below -threshold
            short_exit = sm_bull           # SM crossed above +threshold

        # RSI cross (mapped 5-min)
        rsi_curr = rsi_5m_curr[i - 1]
        rsi_prev_val = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_curr > rsi_buy and rsi_prev_val <= rsi_buy
        rsi_short_trigger = rsi_curr < rsi_sell and rsi_prev_val >= rsi_sell

        # Episode reset -- zero-crossing only (not threshold)
        if sm_flipped_bull or sm_prev <= 0:
            long_used = False
        if sm_flipped_bear or sm_prev >= 0:
            short_used = False

        # EOD Close
        if trade_state != 0 and bar_mins_et >= NY_CLOSE_ET:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # Exits
        if trade_state == 1:
            # Max loss stop
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue
            # SM flip exit
            if long_exit:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        elif trade_state == -1:
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue
            if short_exit:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        # Entries
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True

    return trades


# ── Scoring ────────────────────────────────────────────────────────
DPP = 2.0
COMM = 0.52

def score(trades, label):
    if not trades:
        print(f"  {label}: 0 trades")
        return
    pts = np.array([t['pts'] for t in trades])
    pnl = pts * DPP - COMM * 2
    wins = pnl > 0
    gw = pnl[wins].sum() if wins.any() else 0
    gl = abs(pnl[~wins].sum()) if (~wins).any() else 0.001
    pf = gw / gl
    sl = sum(1 for t in trades if t.get('result') == 'SL')
    avg_bars = np.mean([t['bars'] for t in trades])
    print(f"  {label}: {len(trades)} trades, PF {pf:.3f}, WR {wins.sum()/len(pnl)*100:.1f}%, "
          f"Net ${pnl.sum():+.2f}, SL={sl}, AvgBars={avg_bars:.1f}")

def monthly(trades, label):
    months = {}
    for t in trades:
        m = pd.Timestamp(t['exit_time']).strftime("%Y-%m")
        if m not in months:
            months[m] = []
        months[m].append(t)
    print(f"\n  {label}:")
    for m in sorted(months.keys()):
        mt = months[m]
        pts = np.array([t['pts'] for t in mt])
        pnl = pts * DPP - COMM * 2
        wins = pnl > 0
        gw = pnl[wins].sum() if wins.any() else 0
        gl = abs(pnl[~wins].sum()) if (~wins).any() else 0.001
        print(f"    {m}: {len(mt):3d} trades, PF {gw/gl:.3f}, WR {wins.sum()/len(pnl)*100:.1f}%, Net ${pnl.sum():+.2f}")


# ── Run both exit modes ────────────────────────────────────────────
print("\n" + "="*70)
print("EXIT MODE COMPARISON (entry threshold=0.15 for both)")
print("="*70)

trades_zero = run_with_exit_mode(sm, opens, highs, lows, closes, times,
                                  rsi_5m_curr, rsi_5m_prev,
                                  exit_mode="zero")
score(trades_zero, "A. EXIT AT ZERO-CROSSING")

trades_thr = run_with_exit_mode(sm, opens, highs, lows, closes, times,
                                 rsi_5m_curr, rsi_5m_prev,
                                 exit_mode="threshold")
score(trades_thr, "B. EXIT AT THRESHOLD-CROSSING")

# ── Monthly ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("MONTHLY BREAKDOWN")
print("="*70)
monthly(trades_zero, "A. EXIT AT ZERO")
monthly(trades_thr, "B. EXIT AT THRESHOLD")

# ── Bad week ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("BAD WEEK (Feb 12-18)")
print("="*70)
cutoff = pd.Timestamp("2026-02-12")
def recent(trades):
    return [t for t in trades if pd.Timestamp(t['entry_time']) >= cutoff]

score(recent(trades_zero), "A. EXIT AT ZERO (Feb 12-18)")
score(recent(trades_thr), "B. EXIT AT THRESHOLD (Feb 12-18)")

# ── Trade-by-trade for divergent trades ────────────────────────────
print("\n" + "="*70)
print("TRADE-BY-TRADE COMPARISON (Feb 12-18)")
print("="*70)

print("\n  A. Zero-crossing exits:")
for t in recent(trades_zero):
    pnl = t['pts'] * DPP - COMM * 2
    print(f"    {t['side']:5s} {str(t['entry_time'])[:16]} -> {str(t['exit_time'])[:16]} "
          f"bars={t['bars']:3d} PnL=${pnl:+7.2f} {t['result']}")

print("\n  B. Threshold exits:")
for t in recent(trades_thr):
    pnl = t['pts'] * DPP - COMM * 2
    print(f"    {t['side']:5s} {str(t['entry_time'])[:16]} -> {str(t['exit_time'])[:16]} "
          f"bars={t['bars']:3d} PnL=${pnl:+7.2f} {t['result']}")
