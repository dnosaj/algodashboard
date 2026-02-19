"""
Test SM "band exclusion" — skip only the 0.10-0.15 death zone, keep everything else.

Three approaches:
  A. Baseline: threshold=0 (take all trades)
  B. Flat threshold 0.15: skip |SM| < 0.15
  C. Band exclusion: skip 0.10 <= |SM| < 0.15, take all others

Each tested as both POST-FILTER (remove trades after) and PRE-FILTER (engine-level).
For pre-filter band exclusion, we modify sm_threshold logic inline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from v10_test_common import (
    compute_smart_money, resample_to_5min, map_5min_rsi_to_1min,
    run_backtest_v10, compute_et_minutes, DATA_DIR,
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
df1_trimmed = df1[df1.index < overlap_start]
df = pd.concat([df1_trimmed, df2])
df = df[~df.index.duplicated(keep='last')]
df.sort_index(inplace=True)
print(f"Combined: {len(df)} bars: {df.index[0]} to {df.index[-1]}")

opens = df["Open"].values
highs = df["High"].values
lows = df["Low"].values
closes = df["Close"].values
volumes = df["Volume"].values
times = df.index

# ── Compute indicators ─────────────────────────────────────────────
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

rsi_dummy = np.full(len(closes), 50.0)

# ── Common params ──────────────────────────────────────────────────
V11 = dict(
    rsi_buy=60, rsi_sell=40,
    cooldown_bars=20, max_loss_pts=50,
    use_rsi_cross=True,
    rsi_5m_curr=rsi_5m_curr,
    rsi_5m_prev=rsi_5m_prev,
)
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
    print(f"  {label}: {len(trades)} trades, PF {pf:.3f}, WR {wins.sum()/len(pnl)*100:.1f}%, "
          f"Net ${pnl.sum():+.2f}, SL={sl}")

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


# ═══════════════════════════════════════════════════════════════════
# POST-FILTER COMPARISON
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("POST-FILTER COMPARISON (run baseline, then remove trades)")
print("="*70)

baseline = run_backtest_v10(opens, highs, lows, closes, sm, rsi_dummy, times,
                            sm_threshold=0, **V11)

# Tag each trade with SM at entry
for t in baseline:
    idx = t['entry_idx']
    t['sm_abs'] = abs(sm[idx - 1]) if idx > 0 else 0.0

score(baseline, "A. BASELINE (all trades)")

flat_015 = [t for t in baseline if t['sm_abs'] >= 0.15]
score(flat_015, "B. FLAT >= 0.15 (post-filter)")

band_excl = [t for t in baseline if not (0.10 <= t['sm_abs'] < 0.15)]
score(band_excl, "C. BAND EXCL 0.10-0.15 (post-filter)")

# Also show what the death zone trades look like
death_zone = [t for t in baseline if 0.10 <= t['sm_abs'] < 0.15]
score(death_zone, "   Death zone trades (0.10-0.15)")

low_sm = [t for t in baseline if t['sm_abs'] < 0.10]
score(low_sm, "   Low SM trades (< 0.10)")


# ═══════════════════════════════════════════════════════════════════
# PRE-FILTER: BAND EXCLUSION (engine-level)
# ═══════════════════════════════════════════════════════════════════
# The engine doesn't support band exclusion natively.
# We'll implement it by modifying the SM array: set SM to 0.0 for bars
# where 0.10 <= |SM| < 0.15. This makes sm_bull/sm_bear false for those
# bars, blocking entries. Episode reset and flip exits still use zero-crossing.
print("\n" + "="*70)
print("PRE-FILTER COMPARISON (engine-level blocking)")
print("="*70)

# Flat threshold pre-filter (already tested but included for comparison)
prefilter_flat = run_backtest_v10(opens, highs, lows, closes, sm, rsi_dummy, times,
                                  sm_threshold=0.15, **V11)
score(prefilter_flat, "B. FLAT >= 0.15 (pre-filter)")

# Band exclusion pre-filter: zero out the death zone in SM
sm_band = sm.copy()
death_mask = (np.abs(sm_band) >= 0.10) & (np.abs(sm_band) < 0.15)
sm_band[death_mask] = 0.0  # Block entries in death zone
print(f"\n  Death zone bars zeroed: {death_mask.sum()} / {len(sm)} ({death_mask.sum()/len(sm)*100:.1f}%)")

prefilter_band = run_backtest_v10(opens, highs, lows, closes, sm_band, rsi_dummy, times,
                                   sm_threshold=0, **V11)
score(prefilter_band, "C. BAND EXCL 0.10-0.15 (pre-filter)")


# ═══════════════════════════════════════════════════════════════════
# MONTHLY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("MONTHLY BREAKDOWN")
print("="*70)
monthly(baseline, "A. BASELINE")
monthly(prefilter_flat, "B. FLAT 0.15 (pre-filter)")
monthly(prefilter_band, "C. BAND EXCL (pre-filter)")


# ═══════════════════════════════════════════════════════════════════
# BAD WEEK COMPARISON
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("BAD WEEK (Feb 12-18)")
print("="*70)
cutoff = pd.Timestamp("2026-02-12")
def recent(trades):
    return [t for t in trades if pd.Timestamp(t['entry_time']) >= cutoff]

score(recent(baseline), "A. BASELINE (Feb 12-18)")
score(recent(prefilter_flat), "B. FLAT 0.15 (Feb 12-18)")
score(recent(prefilter_band), "C. BAND EXCL (Feb 12-18)")

# Trade details for bad week
print("\n  BAND EXCL trades (Feb 12-18):")
for t in recent(prefilter_band):
    pnl = t['pts'] * DPP - COMM * 2
    # Look up SM from original (not zeroed) array
    sm_entry = sm[t['entry_idx']-1] if t['entry_idx'] > 0 else 0
    print(f"    {t['side']:5s} {str(t['entry_time'])[:16]} -> {str(t['exit_time'])[:16]} "
          f"PnL=${pnl:+7.2f} |SM|={abs(sm_entry):.3f} {t['result']}")
