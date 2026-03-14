"""
TPX Regime Gate — Paper Trade Validation
==========================================
Checks which of the ~29 vScalpB paper trades would have been BLOCKED
by the TPX regime gate (L=30, S=12):
  - LONG  blocked if TPX[i-1] <= 0
  - SHORT blocked if TPX[i-1] >= 0

Loads all vScalpB trades from session JSONs, computes TPX from the
Databento 1-min MNQ bars, and classifies each trade as ALLOWED or BLOCKED.

Usage:
    cd backtesting_engine/strategies
    python3 tpx_paper_trade_check.py
"""

import sys
import json
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "live_trading"))

from compute_tpx import compute_tpx
from generate_session import load_instrument_1min

# --- Config ---
TPX_LENGTH = 30
TPX_SMOOTH = 12
STRATEGY_ID = "MNQ_VSCALPB"
SESSIONS_DIR = SCRIPT_DIR.parent.parent / "live_trading" / "sessions"
MNQ_DOLLAR_PER_PT = 2.0
MNQ_COMMISSION_PER_SIDE = 0.52


def load_vscalpb_trades():
    """Load all vScalpB trades from session JSON files."""
    trades = []
    for fpath in sorted(glob.glob(str(SESSIONS_DIR / "session_*.json"))):
        with open(fpath) as f:
            data = json.load(f)
        for t in data.get("trades", []):
            if t.get("strategy_id") == STRATEGY_ID:
                t["_session"] = os.path.basename(fpath)
                trades.append(t)
    return trades


def parse_entry_time(entry_time_str):
    """Parse entry_time ISO string to pandas Timestamp.

    Session JSONs store ET time with a fake +00:00 suffix.
    Databento CSVs (loaded by load_instrument_1min) use the same convention:
    ET epochs treated as UTC. So parsing as-is gives a matching timestamp.
    """
    dt = pd.Timestamp(entry_time_str)
    if dt.tzinfo is not None:
        dt = dt.tz_localize(None)
    return dt


def find_bar_index(df_index, target_ts):
    """Find the index position of the bar matching target_ts.

    Uses searchsorted to find the nearest bar, then verifies the match
    is within 60 seconds.
    """
    idx = df_index.searchsorted(target_ts)

    # Check exact match or nearest
    best_pos = None
    best_diff = pd.Timedelta(days=999)
    for candidate in [max(0, idx - 1), min(idx, len(df_index) - 1)]:
        diff = abs(df_index[candidate] - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_pos = candidate

    if best_diff > pd.Timedelta(seconds=60):
        return None  # No matching bar found
    return best_pos


def main():
    print("=" * 90)
    print("  TPX REGIME GATE — Paper Trade Validation")
    print(f"  TPX params: L={TPX_LENGTH}, S={TPX_SMOOTH}")
    print(f"  Rule: LONG only when TPX>0, SHORT only when TPX<0")
    print("=" * 90)

    # --- Load MNQ 1-min bars ---
    print("\nLoading MNQ 1-min bars...")
    df = load_instrument_1min("MNQ")
    print(f"  {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # --- Compute TPX ---
    print(f"\nComputing TPX(L={TPX_LENGTH}, S={TPX_SMOOTH})...")
    highs = df["High"].values
    lows = df["Low"].values
    tpx, avgbulls, avgbears = compute_tpx(highs, lows,
                                           length=TPX_LENGTH,
                                           smooth=TPX_SMOOTH)
    pct_pos = np.sum(tpx > 0) / len(tpx) * 100
    print(f"  TPX range: [{np.min(tpx):.1f}, {np.max(tpx):.1f}], "
          f"mean={np.mean(tpx):.1f}, {pct_pos:.0f}% bullish")

    # --- Load paper trades ---
    print("\nLoading vScalpB paper trades...")
    trades = load_vscalpb_trades()
    print(f"  Found {len(trades)} trades across session files")

    # --- Classify each trade ---
    results = []
    unmatched = 0

    for t in trades:
        entry_ts = parse_entry_time(t["entry_time"])
        bar_pos = find_bar_index(df.index, entry_ts)

        if bar_pos is None or bar_pos < 1:
            unmatched += 1
            continue

        # TPX gate uses bar[i-1] (previous bar's TPX)
        tpx_at_entry = tpx[bar_pos - 1]
        side = t["side"].upper()

        # Gate logic: LONG blocked if TPX <= 0, SHORT blocked if TPX >= 0
        if side == "LONG":
            blocked = tpx_at_entry <= 0
        else:  # SHORT
            blocked = tpx_at_entry >= 0

        # Compute net PnL (subtract commission)
        gross_pnl = t["pnl"]  # Already in dollars (pts * $/pt * qty)
        qty = t.get("qty", 1)
        commission = MNQ_COMMISSION_PER_SIDE * 2 * qty
        net_pnl = gross_pnl - commission

        results.append({
            "date": entry_ts.strftime("%Y-%m-%d"),
            "time_et": entry_ts.strftime("%H:%M"),
            "side": side,
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "pts": t["pts"],
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "exit_reason": t["exit_reason"],
            "tpx_prev": tpx_at_entry,
            "blocked": blocked,
            "action": "BLOCKED" if blocked else "ALLOWED",
            "winner": net_pnl > 0,
        })

    if unmatched:
        print(f"  WARNING: {unmatched} trades could not be matched to bar data")

    # --- Display results table ---
    print(f"\n{'─' * 110}")
    print(f"  {'#':>2s}  {'Date':10s}  {'Time':5s}  {'Side':5s}  {'Entry':>10s}  "
          f"{'Pts':>7s}  {'Net$':>8s}  {'Exit':4s}  "
          f"{'TPX[i-1]':>8s}  {'Action':7s}  {'W/L':3s}")
    print(f"{'─' * 110}")

    for i, r in enumerate(results, 1):
        wl = "W" if r["winner"] else "L"
        print(f"  {i:>2d}  {r['date']}  {r['time_et']}  {r['side']:5s}  "
              f"{r['entry_price']:>10.2f}  {r['pts']:>+7.2f}  "
              f"${r['net_pnl']:>+7.2f}  {r['exit_reason']:4s}  "
              f"{r['tpx_prev']:>+8.2f}  {r['action']:7s}  {wl}")

    # --- Summary statistics ---
    print(f"\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")

    total = len(results)
    blocked_trades = [r for r in results if r["blocked"]]
    allowed_trades = [r for r in results if not r["blocked"]]

    print(f"\n  Total paper trades:  {total}")
    print(f"  ALLOWED by TPX:     {len(allowed_trades)}")
    print(f"  BLOCKED by TPX:     {len(blocked_trades)}")

    # Allowed breakdown
    if allowed_trades:
        aw = sum(1 for r in allowed_trades if r["winner"])
        al = len(allowed_trades) - aw
        allowed_pnl = sum(r["net_pnl"] for r in allowed_trades)
        print(f"\n  ALLOWED trades ({len(allowed_trades)}):")
        print(f"    Winners: {aw}  |  Losers: {al}  |  WR: {aw/len(allowed_trades)*100:.1f}%")
        print(f"    Total Net P&L: ${allowed_pnl:+.2f}")

    # Blocked breakdown
    if blocked_trades:
        bw = sum(1 for r in blocked_trades if r["winner"])
        bl = len(blocked_trades) - bw
        blocked_pnl = sum(r["net_pnl"] for r in blocked_trades)
        print(f"\n  BLOCKED trades ({len(blocked_trades)}):")
        print(f"    Winners: {bw}  |  Losers: {bl}  |  WR: {bw/len(blocked_trades)*100:.1f}%")
        print(f"    Total Net P&L: ${blocked_pnl:+.2f}")
        print(f"    P&L saved by blocking: ${-blocked_pnl:+.2f}")

        print(f"\n  Blocked trade details:")
        for i, r in enumerate(blocked_trades, 1):
            wl = "W" if r["winner"] else "L"
            print(f"    {i}. {r['date']} {r['time_et']} {r['side']:5s} "
                  f"TPX={r['tpx_prev']:+.2f}  ${r['net_pnl']:+.2f} ({r['exit_reason']}) [{wl}]")
    else:
        print("\n  No trades would have been blocked!")

    # Overall impact
    total_pnl_all = sum(r["net_pnl"] for r in results)
    total_pnl_allowed = sum(r["net_pnl"] for r in allowed_trades) if allowed_trades else 0
    total_winners_all = sum(1 for r in results if r["winner"])
    total_winners_allowed = sum(1 for r in allowed_trades if r["winner"]) if allowed_trades else 0

    print(f"\n  Impact comparison:")
    print(f"    {'':20s} {'Without TPX':>12s}  {'With TPX':>12s}  {'Delta':>10s}")
    print(f"    {'Trades':20s} {total:>12d}  {len(allowed_trades):>12d}  {len(allowed_trades)-total:>+10d}")
    print(f"    {'Win Rate':20s} {total_winners_all/total*100 if total else 0:>11.1f}%  "
          f"{total_winners_allowed/len(allowed_trades)*100 if allowed_trades else 0:>11.1f}%  "
          f"{(total_winners_allowed/len(allowed_trades)*100 if allowed_trades else 0) - (total_winners_all/total*100 if total else 0):>+9.1f}%")
    print(f"    {'Net P&L':20s} ${total_pnl_all:>+11.2f}  ${total_pnl_allowed:>+11.2f}  "
          f"${total_pnl_allowed - total_pnl_all:>+9.2f}")

    # Backtest comparison note
    print(f"\n  {'─' * 60}")
    print(f"  NOTE: This is a POST-FILTER analysis (would this trade have been")
    print(f"  blocked?). In a real pre-filter, blocked entries change cooldowns")
    print(f"  and episode timing, so some subsequent trades may also shift.")
    print(f"  Use backtest results (tpx_regime_test.py) for true pre-filter impact.")
    print(f"  {'─' * 60}")


if __name__ == "__main__":
    main()
