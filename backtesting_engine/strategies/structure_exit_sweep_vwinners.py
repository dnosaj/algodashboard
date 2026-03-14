#!/usr/bin/env python3
"""
Structure-Based Exit Sweep — vWinners (MNQ v11.1)
===================================================
Tests whether swing-level structure exits can revive the shelved vWinners
strategy. vWinners uses SM flip as exit which is IS +$3,529 but OOS -$509.

Entry: v11.1 production
  SM(10/12/200/100), SM_T=0.15, RSI(8/60/40), cooldown=20
  SL=50 pts, session_end=15:45, session_close=16:00

Original exit: SM flip (run_backtest_v10 baseline)

Structure exit replaces SM flip entirely:
  - Single contract (no partial exit)
  - Exit when price approaches nearest swing high/low
  - Fallback: SL=50pts, EOD=16:00
  - No trailing, no BE_TIME, no partial exit

Sweep grid:
  swing_lookback:   [10, 15, 20, 30, 50]
  swing_type:       ["pivot", "donchian"]
  pivot_right:      [1, 2, 3]  (pivot only)
  swing_buffer_pts: [0, 2, 5]
  min_profit_pts:   [0, 5, 10]
  max_tp2_pts:      [0]  (structure-only, no hard TP cap)

Usage:
    cd backtesting_engine && python3 strategies/structure_exit_sweep_vwinners.py
"""

import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR.parent))  # backtesting_engine/
sys.path.insert(0, str(_STRAT_DIR))         # strategies/

from v10_test_common import (
    compute_smart_money,
    compute_et_minutes,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    run_backtest_v10,
    score_trades,
    fmt_score,
    fmt_exits,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)

from structure_exit_common import compute_swing_levels

# ---------------------------------------------------------------------------
# vWinners (v11.1) Parameters
# ---------------------------------------------------------------------------

SM_INDEX = 10
SM_FLOW = 12
SM_NORM = 200
SM_EMA = 100
SM_THRESHOLD = 0.15
RSI_LEN = 8
RSI_BUY = 60
RSI_SELL = 40
COOLDOWN = 20
MAX_LOSS_PTS = 50
DOLLAR_PER_PT = 2.0
COMMISSION_PER_SIDE = 0.52
SESSION_END_ET = NY_LAST_ENTRY_ET   # 15:45 ET (last entry)
EOD_ET = NY_CLOSE_ET                # 16:00 ET (force close)

# ---------------------------------------------------------------------------
# Sweep Grid
# ---------------------------------------------------------------------------

SWING_LOOKBACKS = [10, 15, 20, 30, 50]
SWING_TYPES = ["pivot", "donchian"]
PIVOT_RIGHTS = [1, 2, 3]
SWING_BUFFERS = [0, 2, 5]
MIN_PROFITS = [0, 5, 10]
MAX_TP2_VALUES = [0]  # No hard TP cap — let structure exit be the only TP


# ---------------------------------------------------------------------------
# vWinners Structure Exit Backtest (single contract, no partial)
# ---------------------------------------------------------------------------

def run_backtest_vwinners_structure_exit(
    opens, highs, lows, closes, sm, times,
    rsi_5m_curr, rsi_5m_prev,
    rsi_buy, rsi_sell, sm_threshold,
    cooldown_bars, max_loss_pts,
    # Structure exit params
    swing_highs, swing_lows,
    swing_buffer_pts=0,
    min_profit_pts=0,
    use_high_low=False,
    max_tp2_pts=0,
    # Session
    eod_minutes_et=NY_CLOSE_ET,
    session_end_et=NY_LAST_ENTRY_ET,
):
    """vWinners single-contract backtest with structure-based exit.

    Entry: IDENTICAL to run_backtest_v10 entry logic.
      SM(10/12/200/100) SM_T=0.15, RSI(8/60/40) cross, cooldown=20.
      Signal from bar[i-1], fill at bar[i] open.

    Exit priority (checked each bar):
      1. SL: close[i-1] breaches entry - max_loss_pts -> fill at open[i]
      2. Structure exit: close[i-1] approaches swing high/low
         (or high/low if use_high_low). Only if profit >= min_profit_pts.
      3. Hard TP cap: close[i-1] >= entry + max_tp2_pts (0 = disabled)
      4. EOD: bar_et >= eod_minutes_et -> fill at close[i]

    Single contract. No partial exit, no trailing, no BE_TIME.

    Returns list of trade dicts compatible with score_trades().
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    et_mins = compute_et_minutes(times)

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

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI cross from mapped 5-min
        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        # Episode reset
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # --- EOD close ---
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # --- Exits for LONG positions ---
        if trade_state == 1:
            # 1. SL check
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # 2. Structure exit: price approaches swing high
            if not np.isnan(swing_highs[i - 1]):
                runner_profit = closes[i - 1] - entry_price
                if runner_profit >= min_profit_pts:
                    target = swing_highs[i - 1] - swing_buffer_pts
                    if use_high_low:
                        price_check = highs[i - 1]
                    else:
                        price_check = closes[i - 1]
                    if price_check >= target:
                        close_trade("long", entry_price, opens[i], entry_idx, i, "structure")
                        trade_state = 0
                        exit_bar = i
                        continue

            # 3. Hard TP cap (if configured)
            if max_tp2_pts > 0 and closes[i - 1] >= entry_price + max_tp2_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP_cap")
                trade_state = 0
                exit_bar = i
                continue

        # --- Exits for SHORT positions ---
        elif trade_state == -1:
            # 1. SL check
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # 2. Structure exit: price approaches swing low
            if not np.isnan(swing_lows[i - 1]):
                runner_profit = entry_price - closes[i - 1]
                if runner_profit >= min_profit_pts:
                    target = swing_lows[i - 1] + swing_buffer_pts
                    if use_high_low:
                        price_check = lows[i - 1]
                    else:
                        price_check = closes[i - 1]
                    if price_check <= target:
                        close_trade("short", entry_price, opens[i], entry_idx, i, "structure")
                        trade_state = 0
                        exit_bar = i
                        continue

            # 3. Hard TP cap
            if max_tp2_pts > 0 and closes[i - 1] <= entry_price - max_tp2_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP_cap")
                trade_state = 0
                exit_bar = i
                continue

        # --- Entries ---
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= session_end_et
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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_mnq_data():
    """Load MNQ data, re-compute SM with vWinners params, split IS/OOS."""
    df = load_instrument_1min("MNQ")

    # Re-compute SM with vWinners params (10/12/200/100)
    # load_databento_1min uses 20/12/400/255 (MES params) by default
    sm = compute_smart_money(
        df["Close"].values, df["Volume"].values,
        index_period=SM_INDEX, flow_period=SM_FLOW,
        norm_period=SM_NORM, ema_len=SM_EMA,
    )
    df["SM_Net"] = sm
    print(f"Loaded MNQ: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    print(f"  SM params: index={SM_INDEX}, flow={SM_FLOW}, norm={SM_NORM}, ema={SM_EMA}")

    # IS/OOS split at midpoint
    mid = df.index[len(df) // 2]
    is_len = (df.index < mid).sum()
    print(f"  Split: IS {is_len} bars ({df.index[0].strftime('%Y-%m-%d')} to "
          f"{df.index[is_len-1].strftime('%Y-%m-%d')}), "
          f"OOS {len(df)-is_len} bars ({df.index[is_len].strftime('%Y-%m-%d')} to "
          f"{df.index[-1].strftime('%Y-%m-%d')})")

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


def run_structure_config(split, swing_lookback, swing_type, pivot_right,
                         swing_buffer_pts, min_profit_pts, use_high_low,
                         max_tp2_pts):
    """Run one structure exit backtest with given config."""
    d = split

    # Pre-compute swing levels for this config
    swing_highs, swing_lows = compute_swing_levels(
        d["highs"], d["lows"],
        lookback=swing_lookback,
        swing_type=swing_type,
        pivot_right=pivot_right,
    )

    trades = run_backtest_vwinners_structure_exit(
        d["opens"], d["highs"], d["lows"], d["closes"], d["sm"], d["times"],
        d["rsi_curr"], d["rsi_prev"],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=SM_THRESHOLD,
        cooldown_bars=COOLDOWN,
        max_loss_pts=MAX_LOSS_PTS,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        swing_buffer_pts=swing_buffer_pts,
        min_profit_pts=min_profit_pts,
        use_high_low=use_high_low,
        max_tp2_pts=max_tp2_pts,
        eod_minutes_et=EOD_ET,
        session_end_et=SESSION_END_ET,
    )

    return trades


def run_sm_flip_baseline(split):
    """Run SM flip baseline using run_backtest_v10 — original vWinners exit."""
    d = split
    trades = run_backtest_v10(
        d["opens"], d["highs"], d["lows"], d["closes"], d["sm"],
        rsi=None,  # unused when mapped RSI provided
        times=d["times"],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=SM_THRESHOLD,
        cooldown_bars=COOLDOWN,
        max_loss_pts=MAX_LOSS_PTS,
        rsi_5m_curr=d["rsi_curr"],
        rsi_5m_prev=d["rsi_prev"],
    )
    return trades


# ---------------------------------------------------------------------------
# Scoring & Formatting Helpers
# ---------------------------------------------------------------------------

def fmt_exit_reasons(sc):
    """Format exit reason breakdown."""
    if sc is None:
        return ""
    exits = sc.get("exits", {})
    parts = []
    for key in ["structure", "TP_cap", "SM_FLIP", "SL", "EOD"]:
        if key in exits:
            parts.append(f"{key}:{exits[key]}")
    return " ".join(parts)


def composite_score(sc):
    """Composite score: PF * Sharpe / (1 + |MaxDD|/1000). Higher = better."""
    if sc is None:
        return -999
    pf = sc["pf"]
    sharpe = sc["sharpe"]
    mdd = abs(sc["max_dd_dollar"])
    if pf <= 0 or sharpe <= 0:
        return -999
    return pf * sharpe / (1 + mdd / 1000)


# ---------------------------------------------------------------------------
# Main Sweep
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 130)
    print("STRUCTURE-BASED EXIT SWEEP — vWINNERS (MNQ v11.1)")
    print("Entry: SM(10/12/200/100) SM_T=0.15, RSI(8/60/40) CD=20 | SL=50")
    print("Original exit: SM flip (SHELVED: IS +$3,529, OOS -$509)")
    print("Test: structure exit replaces SM flip (single contract)")
    print("=" * 130)

    df, splits, is_len = load_mnq_data()

    # ==================================================================
    # BASELINE: SM flip exit (original vWinners)
    # ==================================================================
    print(f"\n{'='*130}")
    print("BASELINE: vWinners with SM flip exit (original, shelved)")
    print(f"{'='*130}")

    baselines = {}
    for split_name in ["FULL", "IS", "OOS"]:
        trades = run_sm_flip_baseline(splits[split_name])
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        baselines[split_name] = {"sc": sc, "trades": trades}
        if sc:
            print(f"  {split_name:>4}: {sc['count']:>4} trades, WR {sc['win_rate']:>5.1f}%, "
                  f"PF {sc['pf']:>6.3f}, ${sc['net_dollar']:>+9.2f}, "
                  f"Sharpe {sc['sharpe']:>6.3f}, MaxDD ${sc['max_dd_dollar']:.2f}")
            print(f"        Exits: {fmt_exit_reasons(sc)}")
        else:
            print(f"  {split_name:>4}: NO TRADES")

    # ==================================================================
    # FULL-PERIOD SWEEP
    # ==================================================================
    print(f"\n{'='*130}")
    print("SWEEPING structure exit parameter grid (FULL period)...")
    print(f"{'='*130}")

    # Build parameter combos
    combos = []

    # Pivot combos
    for lb, pr, buf, mp, mtp in product(
        SWING_LOOKBACKS, PIVOT_RIGHTS, SWING_BUFFERS,
        MIN_PROFITS, MAX_TP2_VALUES
    ):
        combos.append({
            "swing_lookback": lb,
            "swing_type": "pivot",
            "pivot_right": pr,
            "swing_buffer_pts": buf,
            "min_profit_pts": mp,
            "use_high_low": False,
            "max_tp2_pts": mtp,
        })

    # Donchian combos (no pivot_right)
    for lb, buf, mp, mtp in product(
        SWING_LOOKBACKS, SWING_BUFFERS,
        MIN_PROFITS, MAX_TP2_VALUES
    ):
        combos.append({
            "swing_lookback": lb,
            "swing_type": "donchian",
            "pivot_right": 2,  # unused but keep for consistency
            "swing_buffer_pts": buf,
            "min_profit_pts": mp,
            "use_high_low": False,
            "max_tp2_pts": mtp,
        })

    total = len(combos)
    print(f"  Total combos: {total}")

    results = []
    for idx, cfg in enumerate(combos):
        trades = run_structure_config(
            splits["FULL"],
            cfg["swing_lookback"], cfg["swing_type"], cfg["pivot_right"],
            cfg["swing_buffer_pts"], cfg["min_profit_pts"],
            cfg["use_high_low"], cfg["max_tp2_pts"],
        )
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        if sc is not None:
            results.append({
                **cfg,
                "sc": sc,
                "trades": trades,
                "composite": composite_score(sc),
            })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ... {idx + 1}/{total} combos done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Completed {len(results)} valid combos in {elapsed:.1f}s")

    # Sort by composite score
    results.sort(key=lambda r: r["composite"], reverse=True)

    # ==================================================================
    # TOP 20 BY COMPOSITE SCORE (FULL period)
    # ==================================================================
    print(f"\n{'='*130}")
    print("TOP 20 BY COMPOSITE SCORE (FULL period)")
    print(f"{'='*130}")

    bl_full = baselines["FULL"]["sc"]
    bl_pf = bl_full["pf"] if bl_full else 1.0

    header = (f"{'Rank':>4} {'Type':>7} {'LB':>3} {'PR':>3} {'Buf':>4} {'MinP':>5} "
              f"{'Cap':>4} | {'N':>4} {'WR%':>6} {'PF':>6} {'P&L':>10} "
              f"{'Sharpe':>7} {'MaxDD':>9} {'Comp':>6} {'dPF%':>6} | {'Exits':>40}")
    print(header)
    print("-" * len(header))

    top20 = results[:20]
    for rank, r in enumerate(top20, 1):
        sc = r["sc"]
        dpf = (sc["pf"] - bl_pf) / bl_pf * 100 if bl_pf > 0 else 0
        pr_str = str(r["pivot_right"]) if r["swing_type"] == "pivot" else "-"
        exits_str = fmt_exit_reasons(sc)
        print(f"{rank:>4} {r['swing_type']:>7} {r['swing_lookback']:>3} {pr_str:>3} "
              f"{r['swing_buffer_pts']:>4} {r['min_profit_pts']:>5} "
              f"{r['max_tp2_pts']:>4} | {sc['count']:>4} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>6.3f} ${sc['net_dollar']:>+9.2f} "
              f"{sc['sharpe']:>7.3f} ${sc['max_dd_dollar']:>8.2f} "
              f"{r['composite']:>6.3f} {dpf:>+5.1f}% | {exits_str:>40}")

    # ==================================================================
    # IS/OOS VALIDATION ON TOP 20
    # ==================================================================
    print(f"\n{'='*130}")
    print("IS/OOS VALIDATION (Top 20 by composite score)")
    print(f"{'='*130}")

    bl_is = baselines["IS"]["sc"]
    bl_oos = baselines["OOS"]["sc"]

    if bl_is:
        print(f"  SM flip baseline IS:  {bl_is['count']:>4} trades, PF {bl_is['pf']:.3f}, "
              f"${bl_is['net_dollar']:+.2f}, Sharpe {bl_is['sharpe']:.3f}")
        print(f"                        Exits: {fmt_exit_reasons(bl_is)}")
    if bl_oos:
        print(f"  SM flip baseline OOS: {bl_oos['count']:>4} trades, PF {bl_oos['pf']:.3f}, "
              f"${bl_oos['net_dollar']:+.2f}, Sharpe {bl_oos['sharpe']:.3f}")
        print(f"                        Exits: {fmt_exit_reasons(bl_oos)}")

    isoos_results = []
    oos_profitable_count = 0

    for rank, r in enumerate(top20, 1):
        cfg = r
        pr_str = str(r["pivot_right"]) if r["swing_type"] == "pivot" else "-"
        label = (f"{r['swing_type'][:3]} LB={r['swing_lookback']} PR={pr_str} "
                 f"Buf={r['swing_buffer_pts']} MinP={r['min_profit_pts']} "
                 f"Cap={r['max_tp2_pts']}")

        # IS
        is_trades = run_structure_config(
            splits["IS"],
            cfg["swing_lookback"], cfg["swing_type"], cfg["pivot_right"],
            cfg["swing_buffer_pts"], cfg["min_profit_pts"],
            cfg["use_high_low"], cfg["max_tp2_pts"],
        )
        is_sc = score_trades(is_trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)

        # OOS
        oos_trades = run_structure_config(
            splits["OOS"],
            cfg["swing_lookback"], cfg["swing_type"], cfg["pivot_right"],
            cfg["swing_buffer_pts"], cfg["min_profit_pts"],
            cfg["use_high_low"], cfg["max_tp2_pts"],
        )
        oos_sc = score_trades(oos_trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)

        isoos_results.append({
            "rank": rank, "config": cfg, "label": label,
            "is_sc": is_sc, "oos_sc": oos_sc,
        })

        print(f"\n  #{rank} {label}")
        full_sc = r["sc"]
        print(f"    FULL: {full_sc['count']:>4} trades, WR {full_sc['win_rate']:>5.1f}%, "
              f"PF {full_sc['pf']:>6.3f}, ${full_sc['net_dollar']:>+9.2f}, "
              f"Sharpe {full_sc['sharpe']:>6.3f}, MaxDD ${full_sc['max_dd_dollar']:.2f}")
        print(f"          Exits: {fmt_exit_reasons(full_sc)}")

        if is_sc:
            print(f"    IS  : {is_sc['count']:>4} trades, WR {is_sc['win_rate']:>5.1f}%, "
                  f"PF {is_sc['pf']:>6.3f}, ${is_sc['net_dollar']:>+9.2f}, "
                  f"Sharpe {is_sc['sharpe']:>6.3f}, MaxDD ${is_sc['max_dd_dollar']:.2f}")
            print(f"          Exits: {fmt_exit_reasons(is_sc)}")
        else:
            print(f"    IS  : NO TRADES")

        if oos_sc:
            print(f"    OOS : {oos_sc['count']:>4} trades, WR {oos_sc['win_rate']:>5.1f}%, "
                  f"PF {oos_sc['pf']:>6.3f}, ${oos_sc['net_dollar']:>+9.2f}, "
                  f"Sharpe {oos_sc['sharpe']:>6.3f}, MaxDD ${oos_sc['max_dd_dollar']:.2f}")
            print(f"          Exits: {fmt_exit_reasons(oos_sc)}")
        else:
            print(f"    OOS : NO TRADES")

        # Stability verdict
        if is_sc and oos_sc and is_sc["pf"] > 0:
            pf_ratio = oos_sc["pf"] / is_sc["pf"]
            if oos_sc["pf"] > is_sc["pf"]:
                verdict = "STRONG (OOS PF > IS PF)"
            elif pf_ratio >= 0.9:
                verdict = f"STABLE (OOS PF = {pf_ratio:.1%} of IS)"
            elif pf_ratio >= 0.7:
                verdict = f"MARGINAL (OOS PF = {pf_ratio:.1%} of IS)"
            else:
                verdict = f"WEAK (OOS PF = {pf_ratio:.1%} of IS)"

            # OOS PF > 1.0 check (critical bar)
            oos_profitable = oos_sc["pf"] > 1.0
            if oos_profitable:
                oos_profitable_count += 1

            # Compare to SM flip baseline
            bl_is_pf = bl_is["pf"] if bl_is else 1.0
            bl_oos_pf = bl_oos["pf"] if bl_oos else 1.0
            is_dpf = (is_sc["pf"] - bl_is_pf) / bl_is_pf * 100 if bl_is_pf > 0 else 0
            oos_dpf = (oos_sc["pf"] - bl_oos_pf) / bl_oos_pf * 100 if bl_oos_pf > 0 else 0
            oos_pf_tag = " *** OOS PF > 1.0 ***" if oos_profitable else ""
            print(f"    --> {verdict} | vs SM flip: IS dPF {is_dpf:+.1f}%, OOS dPF {oos_dpf:+.1f}%{oos_pf_tag}")

    # ==================================================================
    # CRITICAL BAR: How many configs achieve OOS PF > 1.0?
    # ==================================================================
    print(f"\n{'='*130}")
    print("CRITICAL BAR: OOS PF > 1.0")
    print(f"{'='*130}")

    # Check ALL configs (not just top 20)
    all_oos_profitable = 0
    all_oos_pf_list = []
    for r in results:
        oos_trades = run_structure_config(
            splits["OOS"],
            r["swing_lookback"], r["swing_type"], r["pivot_right"],
            r["swing_buffer_pts"], r["min_profit_pts"],
            r["use_high_low"], r["max_tp2_pts"],
        )
        oos_sc = score_trades(oos_trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        if oos_sc and oos_sc["pf"] > 1.0:
            all_oos_profitable += 1
            all_oos_pf_list.append((r, oos_sc))

    print(f"  Total configs tested: {len(results)}")
    print(f"  Configs with OOS PF > 1.0: {all_oos_profitable} ({all_oos_profitable/len(results)*100:.1f}%)")
    print(f"  SM flip baseline OOS PF: {bl_oos['pf']:.3f}" if bl_oos else "  SM flip baseline OOS: NO TRADES")

    if all_oos_pf_list:
        # Sort by OOS PF
        all_oos_pf_list.sort(key=lambda x: x[1]["pf"], reverse=True)
        print(f"\n  Top 10 configs by OOS PF (among those > 1.0):")
        header2 = (f"    {'Type':>7} {'LB':>3} {'PR':>3} {'Buf':>4} {'MinP':>5} "
                    f"| {'OOS_N':>5} {'OOS_WR%':>7} {'OOS_PF':>7} {'OOS_P&L':>10} {'OOS_Sharpe':>10}")
        print(header2)
        for r, oos_sc in all_oos_pf_list[:10]:
            pr_str = str(r["pivot_right"]) if r["swing_type"] == "pivot" else "-"
            print(f"    {r['swing_type']:>7} {r['swing_lookback']:>3} {pr_str:>3} "
                  f"{r['swing_buffer_pts']:>4} {r['min_profit_pts']:>5} "
                  f"| {oos_sc['count']:>5} {oos_sc['win_rate']:>6.1f}% "
                  f"{oos_sc['pf']:>7.3f} ${oos_sc['net_dollar']:>+9.2f} "
                  f"{oos_sc['sharpe']:>10.3f}")

    # ==================================================================
    # TRADE-BY-TRADE COMPARISON (Best config vs SM flip baseline)
    # ==================================================================
    if top20:
        print(f"\n{'='*130}")
        print("TRADE-BY-TRADE COMPARISON: Best structure exit vs SM flip baseline (FULL period)")
        print(f"{'='*130}")

        best = top20[0]
        best_trades = best["trades"]
        bl_trades = baselines["FULL"]["trades"]

        # Index trades by entry_idx
        best_by_eidx = {t["entry_idx"]: t for t in best_trades}
        bl_by_eidx = {t["entry_idx"]: t for t in bl_trades}

        common_eidx = sorted(set(best_by_eidx.keys()) & set(bl_by_eidx.keys()))
        best_only = sorted(set(best_by_eidx.keys()) - set(bl_by_eidx.keys()))
        bl_only = sorted(set(bl_by_eidx.keys()) - set(best_by_eidx.keys()))

        print(f"  Common entries: {len(common_eidx)}, "
              f"Structure-only: {len(best_only)}, "
              f"SM-flip-only: {len(bl_only)}")

        comm_pts = (COMMISSION_PER_SIDE * 2) / DOLLAR_PER_PT

        improved = 0
        degraded = 0
        unchanged = 0
        total_improvement = 0.0
        improvements = []

        for eidx in common_eidx:
            best_t = best_by_eidx[eidx]
            bl_t = bl_by_eidx[eidx]
            best_pnl = (best_t["pts"] - comm_pts) * DOLLAR_PER_PT
            bl_pnl = (bl_t["pts"] - comm_pts) * DOLLAR_PER_PT
            diff = best_pnl - bl_pnl
            total_improvement += diff

            if abs(diff) < 0.01:
                unchanged += 1
            elif diff > 0:
                improved += 1
            else:
                degraded += 1

            improvements.append({
                "eidx": eidx,
                "best_pnl": best_pnl,
                "bl_pnl": bl_pnl,
                "diff": diff,
                "best_reason": best_t["result"],
                "bl_reason": bl_t["result"],
                "best_bars": best_t["bars"],
                "bl_bars": bl_t["bars"],
            })

        print(f"\n  Improved: {improved}, Degraded: {degraded}, Unchanged: {unchanged}")
        print(f"  Total PnL improvement: ${total_improvement:+.2f}")
        if common_eidx:
            print(f"  Avg improvement per trade: ${total_improvement / len(common_eidx):+.2f}")

        # Show avg hold time comparison
        best_avg_bars = np.mean([best_by_eidx[e]["bars"] for e in common_eidx])
        bl_avg_bars = np.mean([bl_by_eidx[e]["bars"] for e in common_eidx])
        print(f"  Avg hold time: structure={best_avg_bars:.1f} bars, SM flip={bl_avg_bars:.1f} bars")

        # Top improvements and degradations
        improvements.sort(key=lambda x: x["diff"], reverse=True)

        print(f"\n  Top 10 IMPROVED trades:")
        print(f"    {'EntryIdx':>8} {'SMflip$':>10} {'Struct$':>10} {'Diff$':>8} "
              f"{'SM_Exit':>10} {'ST_Exit':>10} {'SM_Bars':>7} {'ST_Bars':>7}")
        for imp in improvements[:10]:
            print(f"    {imp['eidx']:>8} ${imp['bl_pnl']:>+9.2f} ${imp['best_pnl']:>+9.2f} "
                  f"${imp['diff']:>+7.2f} {imp['bl_reason']:>10} {imp['best_reason']:>10} "
                  f"{imp['bl_bars']:>7} {imp['best_bars']:>7}")

        print(f"\n  Top 10 DEGRADED trades:")
        for imp in improvements[-10:]:
            print(f"    {imp['eidx']:>8} ${imp['bl_pnl']:>+9.2f} ${imp['best_pnl']:>+9.2f} "
                  f"${imp['diff']:>+7.2f} {imp['bl_reason']:>10} {imp['best_reason']:>10} "
                  f"{imp['bl_bars']:>7} {imp['best_bars']:>7}")

    # ==================================================================
    # EXIT REASON DISTRIBUTION (all configs vs baseline)
    # ==================================================================
    print(f"\n{'='*130}")
    print("EXIT REASON DISTRIBUTION — Best config vs SM flip baseline (FULL)")
    print(f"{'='*130}")

    if top20:
        best_sc = top20[0]["sc"]
        print(f"\n  SM flip baseline:  {fmt_exit_reasons(bl_full)}")
        print(f"  Structure exit:    {fmt_exit_reasons(best_sc)}")

        # Avg bars held comparison across all configs
        avg_bars_list = [r["sc"]["avg_bars"] for r in results if r["sc"]]
        bl_avg = bl_full["avg_bars"] if bl_full else 0
        print(f"\n  Avg bars held — SM flip baseline: {bl_avg:.1f}")
        print(f"  Avg bars held — Structure exit configs: "
              f"min={min(avg_bars_list):.1f}, median={np.median(avg_bars_list):.1f}, "
              f"max={max(avg_bars_list):.1f}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*130}")
    print(f"SUMMARY")
    print(f"{'='*130}")
    print(f"  SM flip baseline: IS PF {bl_is['pf']:.3f} (${bl_is['net_dollar']:+.2f}), "
          f"OOS PF {bl_oos['pf']:.3f} (${bl_oos['net_dollar']:+.2f})"
          if bl_is and bl_oos else "  SM flip baseline: incomplete")
    print(f"  Configs tested: {len(results)}")
    print(f"  Configs with OOS PF > 1.0: {all_oos_profitable}/{len(results)}")
    if oos_profitable_count > 0:
        print(f"  Top 20 configs with OOS PF > 1.0: {oos_profitable_count}/20")
    else:
        print(f"  Top 20 configs with OOS PF > 1.0: 0/20")

    # Critic verdict
    print(f"\n  CRITIC'S PREDICTION: 'low probability of success'")
    if all_oos_profitable == 0:
        print(f"  RESULT: Critic was RIGHT. Zero configs achieve OOS PF > 1.0.")
        print(f"  CONCLUSION: Structure exits do NOT fix vWinners' OOS problem.")
        print(f"  The entry signal (SM_T=0.15) degrades OOS regardless of exit method.")
    elif oos_profitable_count >= 10:
        print(f"  RESULT: Critic was WRONG. {oos_profitable_count}/20 top configs achieve OOS PF > 1.0.")
        print(f"  CONCLUSION: Structure exits MAY revive vWinners. Further validation needed.")
    else:
        print(f"  RESULT: Mixed. {all_oos_profitable}/{len(results)} configs achieve OOS PF > 1.0,")
        print(f"  but only {oos_profitable_count}/20 in the top composite-score tier.")
        print(f"  CONCLUSION: Marginal improvement. Probably not worth the complexity.")

    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
