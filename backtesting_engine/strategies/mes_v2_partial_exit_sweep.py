#!/usr/bin/env python3
"""
MES v2 Partial Exit Sweep — Mar 11, 2026
==========================================
Two research questions:
  1. TP1 sweep: What TP1 value maximizes risk-adjusted returns at 2 contracts?
     Currently TP1=10. Testing 3-12. TP2=20, SL=35, BE_TIME=75 fixed.
  2. Breakeven escape: When a trade wanders back near entry during the 75-bar
     window, should we cut it rather than wait for BE_TIME?

Simulates full 2-contract partial exit logic:
  - 2 contracts enter
  - If price reaches TP1: close 1 contract, move runner SL to entry (BE)
  - Runner exits at TP2 / SL(BE) / BE_TIME / EOD
  - If SL hits before TP1: both contracts exit at SL
  - BE_TIME: remaining contracts close after 75 bars

Usage:
    cd backtesting_engine/strategies && python3 mes_v2_partial_exit_sweep.py
"""

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict

import numpy as np
import pandas as pd

_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR))

from v10_test_common import (
    compute_smart_money,
    compute_et_minutes,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)

_ET = ZoneInfo("America/New_York")

# MES v2 parameters (fixed)
SM_INDEX = 20
SM_FLOW = 12
SM_NORM = 400
SM_EMA = 255
SM_THRESHOLD = 0.0
RSI_LEN = 12
RSI_BUY = 55
RSI_SELL = 45
COOLDOWN = 25
SL_PTS = 35
TP2_PTS = 20
BE_TIME_BARS = 75
EOD_ET = 15 * 60 + 30
ENTRY_END_ET = NY_LAST_ENTRY_ET
DOLLAR_PER_PT = 5.0
COMMISSION_PER_SIDE = 1.25


def load_data():
    """Load MES, compute SM + RSI, split IS/OOS."""
    df = load_instrument_1min("MES")
    sm = compute_smart_money(
        df["Close"].values, df["Volume"].values,
        index_period=SM_INDEX, flow_period=SM_FLOW,
        norm_period=SM_NORM, ema_len=SM_EMA,
    )
    df["SM_Net"] = sm
    print(f"Loaded MES: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    mid = df.index[len(df) // 2]
    is_len = (df.index < mid).sum()
    print(f"  Split: IS {is_len} bars, OOS {len(df)-is_len} bars")

    # Compute RSI for each split
    splits = {}
    for name, df_s in [("FULL", df), ("IS", df[df.index < mid]), ("OOS", df[df.index >= mid])]:
        df_5m = resample_to_5min(df_s)
        rsi_curr, rsi_prev = map_5min_rsi_to_1min(
            df_s.index.values, df_5m.index.values,
            df_5m["Close"].values, rsi_len=RSI_LEN,
        )
        splits[name] = {
            "opens": df_s["Open"].values,
            "highs": df_s["High"].values,
            "lows": df_s["Low"].values,
            "closes": df_s["Close"].values,
            "sm": df_s["SM_Net"].values,
            "times": df_s.index,
            "rsi_curr": rsi_curr,
            "rsi_prev": rsi_prev,
        }

    return df, splits, is_len


def run_partial_exit(split, tp1_pts, tp2_pts=TP2_PTS, sl_pts=SL_PTS,
                     be_time_bars=BE_TIME_BARS, be_escape_pts=0, be_escape_after=0):
    """Run 2-contract MES backtest with partial exit simulation.

    Parameters:
        tp1_pts:        Partial TP for contract 1 (0 = no partial, treat as 1 contract)
        tp2_pts:        TP for contract 2 (runner)
        sl_pts:         Stop loss in points
        be_time_bars:   Close stale trades after N bars (0 = disabled)
        be_escape_pts:  Close if trade returns within N pts of entry after be_escape_after bars (0 = disabled)
        be_escape_after: Minimum bars before breakeven escape activates (0 = disabled)

    Returns list of trade dicts with fields:
        side, entry, exit_c1, exit_c2, pts_c1, pts_c2, pnl_net,
        tp1_filled, result_c1, result_c2, entry_idx, bars, entry_time
    """
    d = split
    opens = d["opens"]
    highs = d["highs"]
    lows = d["lows"]
    closes = d["closes"]
    sm = d["sm"]
    times = d["times"]
    rsi_curr = d["rsi_curr"]
    rsi_prev = d["rsi_prev"]
    n = len(opens)

    et_mins = compute_et_minutes(times)
    trades = []

    state = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    # Partial state
    tp1_filled = False
    c1_exit_price = 0.0
    c1_result = ""

    for i in range(2, n):
        bar_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        sm_bull = sm_prev > SM_THRESHOLD
        sm_bear = sm_prev < -SM_THRESHOLD
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        rsi_p = rsi_curr[i - 1]
        rsi_p2 = rsi_prev[i - 1]
        rsi_long = rsi_p > RSI_BUY and rsi_p2 <= RSI_BUY
        rsi_short = rsi_p < RSI_SELL and rsi_p2 >= RSI_SELL

        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # ---- Exits ----
        if state != 0:
            side = "long" if state == 1 else "short"
            is_long = state == 1
            bars_held = (i - 1) - entry_idx

            # How many contracts still open?
            remaining_qty = 1 if tp1_filled else 2

            # Current P&L reference
            prev_close = closes[i - 1]
            if is_long:
                unrealized_pts = prev_close - entry_price
            else:
                unrealized_pts = entry_price - prev_close

            # Effective SL for runner (BE after TP1)
            eff_sl = 0.0 if tp1_filled else sl_pts

            # EOD
            if bar_et >= EOD_ET:
                exit_p = closes[i]
                if is_long:
                    c2_pts = exit_p - entry_price
                else:
                    c2_pts = entry_price - exit_p

                if not tp1_filled:
                    # Both contracts exit EOD
                    c1_exit_price = exit_p
                    c1_result = "EOD"
                    total_pts = c2_pts * 2
                    total_pnl = total_pts * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": exit_p, "exit_c2": exit_p,
                        "pts_c1": c2_pts, "pts_c2": c2_pts,
                        "pnl_net": total_pnl,
                        "tp1_filled": False, "result_c1": "EOD", "result_c2": "EOD",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                else:
                    # Only runner exits EOD
                    c2_pts_val = (exit_p - entry_price) if is_long else (entry_price - exit_p)
                    c1_pts_val = (c1_exit_price - entry_price) if is_long else (entry_price - c1_exit_price)
                    total_pnl = (c1_pts_val + c2_pts_val) * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": c1_exit_price, "exit_c2": exit_p,
                        "pts_c1": c1_pts_val, "pts_c2": c2_pts_val,
                        "pnl_net": total_pnl,
                        "tp1_filled": True, "result_c1": c1_result, "result_c2": "EOD",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                state = 0
                exit_bar = i
                tp1_filled = False
                continue

            # SL check (using prev bar close)
            if not tp1_filled:
                # Full SL: both contracts
                sl_hit = (is_long and unrealized_pts <= -sl_pts) or \
                         (not is_long and unrealized_pts <= -sl_pts)
                if sl_hit:
                    exit_p = opens[i]
                    c_pts = (exit_p - entry_price) if is_long else (entry_price - exit_p)
                    total_pnl = c_pts * 2 * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": exit_p, "exit_c2": exit_p,
                        "pts_c1": c_pts, "pts_c2": c_pts,
                        "pnl_net": total_pnl,
                        "tp1_filled": False, "result_c1": "SL", "result_c2": "SL",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                    state = 0
                    exit_bar = i
                    continue
            else:
                # Runner SL at breakeven (entry price)
                runner_sl_hit = (is_long and prev_close <= entry_price) or \
                                (not is_long and prev_close >= entry_price)
                if runner_sl_hit:
                    exit_p = opens[i]
                    c2_pts = (exit_p - entry_price) if is_long else (entry_price - exit_p)
                    c1_pts = (c1_exit_price - entry_price) if is_long else (entry_price - c1_exit_price)
                    total_pnl = (c1_pts + c2_pts) * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": c1_exit_price, "exit_c2": exit_p,
                        "pts_c1": c1_pts, "pts_c2": c2_pts,
                        "pnl_net": total_pnl,
                        "tp1_filled": True, "result_c1": c1_result, "result_c2": "SL_BE",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                    state = 0
                    exit_bar = i
                    tp1_filled = False
                    continue

            # TP1 check (partial — close 1 contract)
            if not tp1_filled and tp1_pts > 0:
                tp1_hit = (is_long and unrealized_pts >= tp1_pts) or \
                          (not is_long and unrealized_pts >= tp1_pts)
                if tp1_hit:
                    tp1_filled = True
                    c1_exit_price = opens[i]
                    c1_result = "TP1"
                    # Don't close the trade — runner continues
                    # Runner SL moves to entry (breakeven)

            # TP2 check (runner or both if no partial)
            tp2_hit = (is_long and unrealized_pts >= tp2_pts) or \
                      (not is_long and unrealized_pts >= tp2_pts)
            if tp2_hit:
                exit_p = opens[i]
                if tp1_filled:
                    c2_pts = (exit_p - entry_price) if is_long else (entry_price - exit_p)
                    c1_pts = (c1_exit_price - entry_price) if is_long else (entry_price - c1_exit_price)
                    total_pnl = (c1_pts + c2_pts) * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": c1_exit_price, "exit_c2": exit_p,
                        "pts_c1": c1_pts, "pts_c2": c2_pts,
                        "pnl_net": total_pnl,
                        "tp1_filled": True, "result_c1": c1_result, "result_c2": "TP",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                else:
                    c_pts = (exit_p - entry_price) if is_long else (entry_price - exit_p)
                    total_pnl = c_pts * 2 * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": exit_p, "exit_c2": exit_p,
                        "pts_c1": c_pts, "pts_c2": c_pts,
                        "pnl_net": total_pnl,
                        "tp1_filled": False, "result_c1": "TP", "result_c2": "TP",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                state = 0
                exit_bar = i
                tp1_filled = False
                continue

            # Breakeven escape: if trade wanders back near entry after N bars
            if be_escape_pts > 0 and be_escape_after > 0 and bars_held >= be_escape_after:
                if not tp1_filled and abs(unrealized_pts) <= be_escape_pts:
                    exit_p = opens[i]
                    c_pts = (exit_p - entry_price) if is_long else (entry_price - exit_p)
                    total_pnl = c_pts * 2 * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": exit_p, "exit_c2": exit_p,
                        "pts_c1": c_pts, "pts_c2": c_pts,
                        "pnl_net": total_pnl,
                        "tp1_filled": False, "result_c1": "BE_ESC", "result_c2": "BE_ESC",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                    state = 0
                    exit_bar = i
                    continue

            # BE_TIME: stale trade exit
            if be_time_bars > 0 and bars_held >= be_time_bars:
                exit_p = opens[i]
                if tp1_filled:
                    c2_pts = (exit_p - entry_price) if is_long else (entry_price - exit_p)
                    c1_pts = (c1_exit_price - entry_price) if is_long else (entry_price - c1_exit_price)
                    total_pnl = (c1_pts + c2_pts) * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": c1_exit_price, "exit_c2": exit_p,
                        "pts_c1": c1_pts, "pts_c2": c2_pts,
                        "pnl_net": total_pnl,
                        "tp1_filled": True, "result_c1": c1_result, "result_c2": "BE_TIME",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                else:
                    c_pts = (exit_p - entry_price) if is_long else (entry_price - exit_p)
                    total_pnl = c_pts * 2 * DOLLAR_PER_PT - 4 * COMMISSION_PER_SIDE
                    trades.append({
                        "side": side, "entry": entry_price,
                        "exit_c1": exit_p, "exit_c2": exit_p,
                        "pts_c1": c_pts, "pts_c2": c_pts,
                        "pnl_net": total_pnl,
                        "tp1_filled": False, "result_c1": "BE_TIME", "result_c2": "BE_TIME",
                        "entry_idx": entry_idx, "exit_idx": i,
                        "bars": i - entry_idx, "entry_time": times[entry_idx],
                    })
                state = 0
                exit_bar = i
                tp1_filled = False
                continue

        # ---- Entries ----
        if state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_et <= ENTRY_END_ET
            cd_ok = bars_since >= COOLDOWN

            if in_session and cd_ok:
                if sm_bull and rsi_long and not long_used:
                    state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    tp1_filled = False
                elif sm_bear and rsi_short and not short_used:
                    state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    tp1_filled = False

    return trades


def score_2c(trades):
    """Score 2-contract trade list."""
    if not trades:
        return {"count": 0, "pf": 0, "net": 0, "sharpe": 0, "wr": 0,
                "tp1_rate": 0, "max_dd": 0, "avg_pnl": 0}

    pnls = [t["pnl_net"] for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    net = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    wr = wins / len(pnls) * 100
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if np.std(pnls) > 0 else 0
    tp1_fills = sum(1 for t in trades if t["tp1_filled"])
    tp1_rate = tp1_fills / len(trades) * 100

    # Max drawdown
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(np.min(dd))

    return {
        "count": len(trades), "pf": pf, "net": net, "sharpe": sharpe,
        "wr": wr, "tp1_rate": tp1_rate, "max_dd": max_dd,
        "avg_pnl": net / len(trades),
    }


def print_header():
    print(f"  {'Config':>30} {'Trades':>6} {'WR':>6} {'PF':>7} {'Net$':>9} "
          f"{'Sharpe':>7} {'TP1%':>6} {'MaxDD':>8} {'Avg$':>7}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*7} {'-'*6} {'-'*8} {'-'*7}")


def print_row(label, sc, bl=None):
    dpf = ""
    if bl and bl["pf"] > 0:
        dpf = f" dPF {(sc['pf'] - bl['pf'])/bl['pf']*100:+.1f}%"
    print(f"  {label:>30} {sc['count']:>6} {sc['wr']:>5.1f}% {sc['pf']:>7.3f} ${sc['net']:>+8.0f} "
          f"{sc['sharpe']:>7.2f} {sc['tp1_rate']:>5.1f}% ${sc['max_dd']:>7.0f} ${sc['avg_pnl']:>+6.1f}{dpf}")


def main():
    print("=" * 120)
    print("MES V2 PARTIAL EXIT SWEEP")
    print("=" * 120)

    df, splits, is_len = load_data()

    # =============================================
    # PART 1: TP1 SWEEP
    # =============================================
    print("\n" + "=" * 120)
    print("PART 1: TP1 SWEEP (2 contracts, TP2=20, SL=35, BE_TIME=75)")
    print("=" * 120)

    # Baseline: TP1=10 (current config)
    for split_name in ["IS", "OOS", "FULL"]:
        print(f"\n  --- {split_name} ---")
        print_header()

        # 1-contract reference (tp1=0 means no partial)
        trades_1c = run_partial_exit(splits[split_name], tp1_pts=0)
        sc_1c = score_2c(trades_1c)
        # Adjust 1c: since tp1=0 means both exit at TP2=20 simultaneously,
        # for a fair 1-contract comparison, just halve the P&L
        # Actually, tp1=0 still runs 2 contracts to TP2. Let's just show it.
        print_row("1-contract (no partial)", sc_1c)

        bl = None
        for tp1 in [3, 4, 5, 6, 7, 8, 9, 10, 12]:
            trades = run_partial_exit(splits[split_name], tp1_pts=tp1)
            sc = score_2c(trades)
            if tp1 == 10:
                bl = sc
                print_row(f"TP1={tp1} (CURRENT) <<<", sc)
            else:
                print_row(f"TP1={tp1}", sc, bl)

    # =============================================
    # PART 2: BREAKEVEN ESCAPE
    # =============================================
    print("\n" + "=" * 120)
    print("PART 2: BREAKEVEN ESCAPE (close wandering trades near entry)")
    print("=" * 120)
    print("  Mechanism: If |unrealized| <= N pts after M bars, close both remaining contracts.")
    print("  Only fires BEFORE TP1 fills (if TP1 already filled, runner has BE SL anyway).")

    for split_name in ["IS", "OOS", "FULL"]:
        print(f"\n  --- {split_name} (TP1=10, base config) ---")
        print_header()

        # Baseline: no escape
        bl_trades = run_partial_exit(splits[split_name], tp1_pts=10)
        bl = score_2c(bl_trades)
        print_row("No escape (CURRENT)", bl)

        # Sweep: escape_pts x escape_after
        for after_bars in [15, 20, 25, 30, 40, 50]:
            for esc_pts in [2, 3, 5, 7]:
                trades = run_partial_exit(splits[split_name], tp1_pts=10,
                                          be_escape_pts=esc_pts, be_escape_after=after_bars)
                sc = score_2c(trades)
                # Count escapes
                n_esc = sum(1 for t in trades if t["result_c1"] == "BE_ESC")
                label = f"Esc ±{esc_pts}pts after {after_bars}bars ({n_esc}esc)"
                print_row(label, sc, bl)

    # =============================================
    # PART 3: BEST TP1 + BEST ESCAPE COMBINED
    # =============================================
    print("\n" + "=" * 120)
    print("PART 3: COMBINED — Best TP1 candidates + breakeven escape")
    print("=" * 120)
    print("  Testing top TP1 values with promising escape configs.")

    for split_name in ["IS", "OOS", "FULL"]:
        print(f"\n  --- {split_name} ---")
        print_header()

        bl_trades = run_partial_exit(splits[split_name], tp1_pts=10)
        bl = score_2c(bl_trades)
        print_row("TP1=10, no escape (CURRENT)", bl)

        for tp1 in [5, 6, 7, 8]:
            # Without escape
            trades = run_partial_exit(splits[split_name], tp1_pts=tp1)
            sc = score_2c(trades)
            print_row(f"TP1={tp1}, no escape", sc, bl)

            # With promising escape configs
            for after_bars, esc_pts in [(20, 3), (25, 3), (30, 5)]:
                trades = run_partial_exit(splits[split_name], tp1_pts=tp1,
                                          be_escape_pts=esc_pts, be_escape_after=after_bars)
                sc = score_2c(trades)
                n_esc = sum(1 for t in trades if t["result_c1"] == "BE_ESC")
                print_row(f"TP1={tp1} + esc±{esc_pts}/{after_bars}b ({n_esc})", sc, bl)

    # =============================================
    # PART 4: EXIT REASON BREAKDOWN
    # =============================================
    print("\n" + "=" * 120)
    print("PART 4: EXIT REASON BREAKDOWN (FULL, current TP1=10)")
    print("=" * 120)

    trades = run_partial_exit(splits["FULL"], tp1_pts=10)
    reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in trades:
        key = f"{t['result_c1']}/{t['result_c2']}"
        reasons[key]["count"] += 1
        reasons[key]["pnl"] += t["pnl_net"]

    print(f"\n  {'Exit Pattern':>25} {'Count':>6} {'Total P&L':>10} {'Avg P&L':>9}")
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*9}")
    for key in sorted(reasons.keys(), key=lambda k: reasons[k]["pnl"]):
        r = reasons[key]
        avg = r["pnl"] / r["count"]
        print(f"  {key:>25} {r['count']:>6} ${r['pnl']:>+9.0f} ${avg:>+8.1f}")

    # Show same for a promising TP1 candidate
    print(f"\n  --- With TP1=7 ---")
    trades7 = run_partial_exit(splits["FULL"], tp1_pts=7)
    reasons7 = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in trades7:
        key = f"{t['result_c1']}/{t['result_c2']}"
        reasons7[key]["count"] += 1
        reasons7[key]["pnl"] += t["pnl_net"]

    print(f"  {'Exit Pattern':>25} {'Count':>6} {'Total P&L':>10} {'Avg P&L':>9}")
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*9}")
    for key in sorted(reasons7.keys(), key=lambda k: reasons7[k]["pnl"]):
        r = reasons7[key]
        avg = r["pnl"] / r["count"]
        print(f"  {key:>25} {r['count']:>6} ${r['pnl']:>+9.0f} ${avg:>+8.1f}")


if __name__ == "__main__":
    main()
