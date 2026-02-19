"""
Quick test: post-filter vs pre-filter SM threshold comparison.

Post-filter: run with threshold=0, remove trades where |SM| < 0.15
Pre-filter: run with threshold=0.15 in the engine (what Pine does)

This reveals whether the validated PF 2.154 post-filter result holds
when actually implemented as a pre-filter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from v10_test_common import (
    load_databento_1min, compute_smart_money, compute_rsi,
    resample_to_5min, map_5min_rsi_to_1min,
    run_backtest_v10, score_trades, DATA_DIR,
)

# ── Load MNQ data ──────────────────────────────────────────────────
print("Loading MNQ 1-min data...")
df = load_databento_1min("MNQ")
opens = df["Open"].values
highs = df["High"].values
lows = df["Low"].values
closes = df["Close"].values
volumes = df["Volume"].values
times = df.index

print(f"Data: {len(closes)} bars, {times[0]} to {times[-1]}")

# SM with v11 params
sm = compute_smart_money(closes, volumes,
                         index_period=10, flow_period=12,
                         norm_period=200, ema_len=100)

# RSI on 5-min mapped to 1-min
df_5m_tmp = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df_5m_tmp['SM_Net'] = 0.0
df_5m = resample_to_5min(df_5m_tmp)
rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
    times, df_5m.index.values, df_5m['Close'].values, rsi_len=8)

# Dummy rsi array (not used when mapped RSI provided)
rsi_dummy = np.full(len(closes), 50.0)

# ── Common params ──────────────────────────────────────────────────
V11_PARAMS = dict(
    rsi_buy=60, rsi_sell=40,
    cooldown_bars=20, max_loss_pts=50,
    use_rsi_cross=True,
    rsi_5m_curr=rsi_5m_curr,
    rsi_5m_prev=rsi_5m_prev,
)
DOLLAR_PER_PT = 2.0
COMMISSION = 0.52

def score(trades, label):
    if not trades:
        print(f"  {label}: 0 trades")
        return
    pts = np.array([t['pts'] for t in trades])
    comm = COMMISSION * 2
    pnl = pts * DOLLAR_PER_PT - comm
    wins = pnl > 0
    gross_w = pnl[wins].sum() if wins.any() else 0
    gross_l = abs(pnl[~wins].sum()) if (~wins).any() else 0.001
    pf = gross_w / gross_l
    wr = wins.sum() / len(pnl) * 100
    net = pnl.sum()
    sl = sum(1 for t in trades if t.get('result') == 'SL')
    print(f"  {label}: {len(trades)} trades, PF {pf:.3f}, WR {wr:.1f}%, "
          f"Net ${net:.2f}, SL exits {sl}")
    return trades

# ── 1. Baseline: threshold=0 ──────────────────────────────────────
print("\n=== BASELINE (threshold=0) ===")
baseline_trades = run_backtest_v10(
    opens, highs, lows, closes, sm, rsi_dummy, times,
    sm_threshold=0, **V11_PARAMS)
score(baseline_trades, "BASELINE")

# ── 2. Post-filter: run baseline, keep |SM|>=0.15 ────────────────
print("\n=== POST-FILTER (threshold=0 engine, keep |SM|>=0.15) ===")
for t in baseline_trades:
    idx = t['entry_idx']
    t['sm_at_entry'] = sm[idx - 1] if idx > 0 else 0.0
    t['sm_abs'] = abs(t['sm_at_entry'])

postfilter_trades = [t for t in baseline_trades if t['sm_abs'] >= 0.15]
score(postfilter_trades, "POST-FILTER")

# ── 3. Pre-filter: threshold=0.15 in engine ──────────────────────
print("\n=== PRE-FILTER (threshold=0.15 in engine) ===")
prefilter_trades = run_backtest_v10(
    opens, highs, lows, closes, sm, rsi_dummy, times,
    sm_threshold=0.15, **V11_PARAMS)
score(prefilter_trades, "PRE-FILTER")

# ── 4. Monthly breakdown for all three ───────────────────────────
print("\n=== MONTHLY BREAKDOWN ===")
import pandas as pd

def monthly_breakdown(trades, label):
    if not trades:
        return
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
        pnl = pts * DOLLAR_PER_PT - COMMISSION * 2
        wins = pnl > 0
        gw = pnl[wins].sum() if wins.any() else 0
        gl = abs(pnl[~wins].sum()) if (~wins).any() else 0.001
        print(f"    {m}: {len(mt):3d} trades, PF {gw/gl:.3f}, "
              f"WR {wins.sum()/len(pnl)*100:.1f}%, Net ${pnl.sum():+.2f}")

monthly_breakdown(baseline_trades, "BASELINE")
monthly_breakdown(postfilter_trades, "POST-FILTER")
monthly_breakdown(prefilter_trades, "PRE-FILTER")

# ── 5. Last 14 trading days specifically ─────────────────────────
print("\n=== LAST 14 TRADING DAYS (approx Feb 3-13) ===")
cutoff = pd.Timestamp("2026-02-03")

def filter_period(trades, start):
    return [t for t in trades if pd.Timestamp(t['entry_time']) >= start]

recent_baseline = filter_period(baseline_trades, cutoff)
recent_postfilter = filter_period(postfilter_trades, cutoff)
recent_prefilter = filter_period(prefilter_trades, cutoff)

score(recent_baseline, "BASELINE (Feb 3-13)")
score(recent_postfilter, "POST-FILTER (Feb 3-13)")
score(recent_prefilter, "PRE-FILTER (Feb 3-13)")
