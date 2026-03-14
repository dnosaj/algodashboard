#!/usr/bin/env python3
"""
Structure-Based Exit Sweep — vScalpC Runner (MNQ)
===================================================
Swing-based runner exits for vScalpC partial exit on MNQ.

Entry: IDENTICAL to vScalpC production
  SM(10/12/200/100), RSI(8/60/40), sm_threshold=0.0, cooldown=20
  entry cutoff 13:00 ET

Scalp leg: UNCHANGED — TP1=7, SL=40

Runner leg: structure-based exit replaces fixed TP2=25
  - Exit when price approaches nearest swing high/low
  - Fallback: SL, BE_TIME=45, EOD=16:00, optional hard TP cap
  - SL-to-BE after TP1 fill

Sweep grid:
  swing_lookback:   [10, 15, 20, 30, 50]
  swing_type:       ["pivot", "donchian"]
  pivot_right:      [1, 2, 3]  (pivot only)
  swing_buffer_pts: [0, 2, 5]
  min_profit_pts:   [0, 5, 10]
  use_high_low:     [False]  (close-based only to reduce grid)
  max_tp2_pts:      [0, 30]  (structure-only and with generous cap)

Usage:
    cd backtesting_engine && python3 strategies/structure_exit_sweep_vscalpc.py
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
    score_trades,
    fmt_score,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)

from structure_exit_common import (
    compute_swing_levels,
    run_backtest_structure_exit,
    score_structure_trades,
)

# ---------------------------------------------------------------------------
# vScalpC Production Parameters (from config.py MNQ_VSCALPC)
# ---------------------------------------------------------------------------

SM_INDEX = 10
SM_FLOW = 12
SM_NORM = 200
SM_EMA = 100
SM_THRESHOLD = 0.0
RSI_LEN = 8
RSI_BUY = 60
RSI_SELL = 40
COOLDOWN = 20
MAX_LOSS_PTS = 40
TP1_PTS = 7             # Scalp leg — production value
TP2_PTS_BASELINE = 25   # Runner leg baseline (fixed TP)
BE_TIME_BARS = 45       # Close stale runner after 45 bars (~45 min)
MOVE_SL_TO_BE = True    # After TP1, move runner SL to entry price
ENTRY_END_ET = 13 * 60  # 13:00 ET — late-day cutoff
EOD_ET = 16 * 60        # 16:00 ET
DOLLAR_PER_PT = 2.0
COMMISSION_PER_SIDE = 0.52

# ---------------------------------------------------------------------------
# Sweep Grid
# ---------------------------------------------------------------------------

SWING_LOOKBACKS = [10, 15, 20, 30, 50]
SWING_TYPES = ["pivot", "donchian"]
PIVOT_RIGHTS = [1, 2, 3]
SWING_BUFFERS = [0, 2, 5]
MIN_PROFITS = [0, 5, 10]
USE_HIGH_LOW = [False]
MAX_TP2_VALUES = [0, 30]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_mnq_data():
    """Load MNQ data, compute SM, split IS/OOS, prepare RSI."""
    df = load_instrument_1min("MNQ")
    sm = compute_smart_money(
        df["Close"].values, df["Volume"].values,
        index_period=SM_INDEX, flow_period=SM_FLOW,
        norm_period=SM_NORM, ema_len=SM_EMA,
    )
    df["SM_Net"] = sm
    print(f"Loaded MNQ: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

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


def run_with_config(split, swing_lookback, swing_type, pivot_right,
                    swing_buffer_pts, min_profit_pts, use_high_low,
                    max_tp2_pts):
    """Run one backtest with given structure exit config."""
    d = split

    # Pre-compute swing levels for this config
    swing_highs, swing_lows = compute_swing_levels(
        d["highs"], d["lows"],
        lookback=swing_lookback,
        swing_type=swing_type,
        pivot_right=pivot_right,
    )

    trades = run_backtest_structure_exit(
        d["opens"], d["highs"], d["lows"], d["closes"], d["sm"], d["times"],
        d["rsi_curr"], d["rsi_prev"],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=SM_THRESHOLD,
        cooldown_bars=COOLDOWN,
        max_loss_pts=MAX_LOSS_PTS,
        tp1_pts=TP1_PTS,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        swing_buffer_pts=swing_buffer_pts,
        min_profit_pts=min_profit_pts,
        use_high_low=use_high_low,
        max_tp2_pts=max_tp2_pts,
        move_sl_to_be_after_tp1=MOVE_SL_TO_BE,
        breakeven_after_bars=BE_TIME_BARS,
        eod_minutes_et=EOD_ET,
        entry_end_et=ENTRY_END_ET,
    )

    return trades


def run_baseline(split):
    """Run baseline: fixed TP2=25 using structure exit with max_tp2_pts=25
    and swing arrays of all NaN (no structure exit, just TP cap).
    """
    d = split
    n = len(d["opens"])
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    trades = run_backtest_structure_exit(
        d["opens"], d["highs"], d["lows"], d["closes"], d["sm"], d["times"],
        d["rsi_curr"], d["rsi_prev"],
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=SM_THRESHOLD,
        cooldown_bars=COOLDOWN,
        max_loss_pts=MAX_LOSS_PTS,
        tp1_pts=TP1_PTS,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        swing_buffer_pts=0,
        min_profit_pts=0,
        use_high_low=False,
        max_tp2_pts=TP2_PTS_BASELINE,  # Hard cap = TP2=25
        move_sl_to_be_after_tp1=MOVE_SL_TO_BE,
        breakeven_after_bars=BE_TIME_BARS,
        eod_minutes_et=EOD_ET,
        entry_end_et=ENTRY_END_ET,
    )

    return trades


def fmt_runner_exits(sc):
    """Format runner exit reason breakdown."""
    if sc is None:
        return ""
    exits = sc.get("runner_exits", {})
    parts = []
    for key in ["structure", "TP_cap", "SL", "BE", "BE_TIME", "EOD"]:
        if key in exits:
            parts.append(f"{key}:{exits[key]}")
    return " ".join(parts)


def composite_score(sc):
    """Composite score for ranking: PF * Sharpe / (1 + |MaxDD|/1000).

    Higher is better. Penalizes large drawdowns.
    """
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
    print("STRUCTURE-BASED EXIT SWEEP — vScalpC RUNNER (MNQ)")
    print("Entry: SM(10/12/200/100) RSI(8/60/40) CD=20 | Scalp: TP1=7, SL=40")
    print("Runner: structure exit + SL(->BE) + BE_TIME=45 + EOD=16:00")
    print("Entry cutoff: 13:00 ET | $2.00/pt, $0.52/side")
    print("=" * 130)

    df, splits, is_len = load_mnq_data()

    # ==================================================================
    # BASELINE: Fixed TP2=25 (current production)
    # ==================================================================
    print(f"\n{'='*130}")
    print("BASELINE: vScalpC with fixed TP2=25 (current production)")
    print(f"{'='*130}")

    baselines = {}
    for split_name in ["FULL", "IS", "OOS"]:
        trades = run_baseline(splits[split_name])
        sc = score_structure_trades(trades, DOLLAR_PER_PT, COMMISSION_PER_SIDE)
        baselines[split_name] = {"sc": sc, "trades": trades}
        if sc:
            print(f"  {split_name:>4}: {sc['count']:>4} trades, WR {sc['win_rate']:>5.1f}%, "
                  f"PF {sc['pf']:>6.3f}, ${sc['net_dollar']:>+9.2f}, "
                  f"Sharpe {sc['sharpe']:>6.3f}, MaxDD ${sc['max_dd_dollar']:.2f}")
            print(f"        Scalp exits: {sc.get('scalp_exits', {})}")
            print(f"        Runner exits: {sc.get('runner_exits', {})}")
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
    for lb, pr, buf, mp, uhl, mtp in product(
        SWING_LOOKBACKS, PIVOT_RIGHTS, SWING_BUFFERS,
        MIN_PROFITS, USE_HIGH_LOW, MAX_TP2_VALUES
    ):
        combos.append({
            "swing_lookback": lb,
            "swing_type": "pivot",
            "pivot_right": pr,
            "swing_buffer_pts": buf,
            "min_profit_pts": mp,
            "use_high_low": uhl,
            "max_tp2_pts": mtp,
        })

    # Donchian combos (no pivot_right)
    for lb, buf, mp, uhl, mtp in product(
        SWING_LOOKBACKS, SWING_BUFFERS,
        MIN_PROFITS, USE_HIGH_LOW, MAX_TP2_VALUES
    ):
        combos.append({
            "swing_lookback": lb,
            "swing_type": "donchian",
            "pivot_right": 2,  # unused but keep for consistency
            "swing_buffer_pts": buf,
            "min_profit_pts": mp,
            "use_high_low": uhl,
            "max_tp2_pts": mtp,
        })

    total = len(combos)
    print(f"  Total combos: {total}")

    results = []
    for idx, cfg in enumerate(combos):
        trades = run_with_config(
            splits["FULL"],
            cfg["swing_lookback"], cfg["swing_type"], cfg["pivot_right"],
            cfg["swing_buffer_pts"], cfg["min_profit_pts"],
            cfg["use_high_low"], cfg["max_tp2_pts"],
        )
        sc = score_structure_trades(trades, DOLLAR_PER_PT, COMMISSION_PER_SIDE)
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
              f"{'Sharpe':>7} {'MaxDD':>9} {'Comp':>6} {'dPF%':>6} | {'Runner Exits':>35}")
    print(header)
    print("-" * len(header))

    top20 = results[:20]
    for rank, r in enumerate(top20, 1):
        sc = r["sc"]
        dpf = (sc["pf"] - bl_pf) / bl_pf * 100 if bl_pf > 0 else 0
        pr_str = str(r["pivot_right"]) if r["swing_type"] == "pivot" else "-"
        runner_str = fmt_runner_exits(sc)
        print(f"{rank:>4} {r['swing_type']:>7} {r['swing_lookback']:>3} {pr_str:>3} "
              f"{r['swing_buffer_pts']:>4} {r['min_profit_pts']:>5} "
              f"{r['max_tp2_pts']:>4} | {sc['count']:>4} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>6.3f} ${sc['net_dollar']:>+9.2f} "
              f"{sc['sharpe']:>7.3f} ${sc['max_dd_dollar']:>8.2f} "
              f"{r['composite']:>6.3f} {dpf:>+5.1f}% | {runner_str:>35}")

    # ==================================================================
    # IS/OOS VALIDATION ON TOP 20
    # ==================================================================
    print(f"\n{'='*130}")
    print("IS/OOS VALIDATION (Top 20 by composite score)")
    print(f"{'='*130}")

    bl_is = baselines["IS"]["sc"]
    bl_oos = baselines["OOS"]["sc"]

    if bl_is:
        print(f"  Baseline IS:  {bl_is['count']:>4} trades, PF {bl_is['pf']:.3f}, "
              f"${bl_is['net_dollar']:+.2f}, Sharpe {bl_is['sharpe']:.3f}")
    if bl_oos:
        print(f"  Baseline OOS: {bl_oos['count']:>4} trades, PF {bl_oos['pf']:.3f}, "
              f"${bl_oos['net_dollar']:+.2f}, Sharpe {bl_oos['sharpe']:.3f}")

    isoos_results = []

    for rank, r in enumerate(top20, 1):
        cfg = r
        pr_str = str(r["pivot_right"]) if r["swing_type"] == "pivot" else "-"
        label = (f"{r['swing_type'][:3]} LB={r['swing_lookback']} PR={pr_str} "
                 f"Buf={r['swing_buffer_pts']} MinP={r['min_profit_pts']} "
                 f"Cap={r['max_tp2_pts']}")

        # IS
        is_trades = run_with_config(
            splits["IS"],
            cfg["swing_lookback"], cfg["swing_type"], cfg["pivot_right"],
            cfg["swing_buffer_pts"], cfg["min_profit_pts"],
            cfg["use_high_low"], cfg["max_tp2_pts"],
        )
        is_sc = score_structure_trades(is_trades, DOLLAR_PER_PT, COMMISSION_PER_SIDE)

        # OOS
        oos_trades = run_with_config(
            splits["OOS"],
            cfg["swing_lookback"], cfg["swing_type"], cfg["pivot_right"],
            cfg["swing_buffer_pts"], cfg["min_profit_pts"],
            cfg["use_high_low"], cfg["max_tp2_pts"],
        )
        oos_sc = score_structure_trades(oos_trades, DOLLAR_PER_PT, COMMISSION_PER_SIDE)

        isoos_results.append({
            "rank": rank, "config": cfg, "label": label,
            "is_sc": is_sc, "oos_sc": oos_sc,
        })

        print(f"\n  #{rank} {label}")
        full_sc = r["sc"]
        print(f"    FULL: {full_sc['count']:>4} trades, WR {full_sc['win_rate']:>5.1f}%, "
              f"PF {full_sc['pf']:>6.3f}, ${full_sc['net_dollar']:>+9.2f}, "
              f"Sharpe {full_sc['sharpe']:>6.3f}, MaxDD ${full_sc['max_dd_dollar']:.2f}")
        print(f"          Runner: {fmt_runner_exits(full_sc)}")

        if is_sc:
            print(f"    IS  : {is_sc['count']:>4} trades, WR {is_sc['win_rate']:>5.1f}%, "
                  f"PF {is_sc['pf']:>6.3f}, ${is_sc['net_dollar']:>+9.2f}, "
                  f"Sharpe {is_sc['sharpe']:>6.3f}, MaxDD ${is_sc['max_dd_dollar']:.2f}")
        else:
            print(f"    IS  : NO TRADES")

        if oos_sc:
            print(f"    OOS : {oos_sc['count']:>4} trades, WR {oos_sc['win_rate']:>5.1f}%, "
                  f"PF {oos_sc['pf']:>6.3f}, ${oos_sc['net_dollar']:>+9.2f}, "
                  f"Sharpe {oos_sc['sharpe']:>6.3f}, MaxDD ${oos_sc['max_dd_dollar']:.2f}")
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
                verdict = f"MARGINAL (OOS PF = {pf_ratio:.1%} of IS PF)"
            else:
                verdict = f"WEAK (OOS PF = {pf_ratio:.1%} of IS PF)"

            # Compare to baseline
            bl_is_pf = bl_is["pf"] if bl_is else 1.0
            bl_oos_pf = bl_oos["pf"] if bl_oos else 1.0
            is_dpf = (is_sc["pf"] - bl_is_pf) / bl_is_pf * 100 if bl_is_pf > 0 else 0
            oos_dpf = (oos_sc["pf"] - bl_oos_pf) / bl_oos_pf * 100 if bl_oos_pf > 0 else 0
            print(f"    --> {verdict} | vs baseline: IS dPF {is_dpf:+.1f}%, OOS dPF {oos_dpf:+.1f}%")

    # ==================================================================
    # TRADE-BY-TRADE COMPARISON (Best config vs baseline)
    # ==================================================================
    if top20:
        print(f"\n{'='*130}")
        print("TRADE-BY-TRADE COMPARISON: Best structure exit vs baseline (FULL period)")
        print(f"{'='*130}")

        best = top20[0]
        best_trades = best["trades"]
        bl_trades = baselines["FULL"]["trades"]

        # Group trades by entry_idx
        def group_entries(trade_list):
            entries = {}
            for t in trade_list:
                eidx = t["entry_idx"]
                if eidx not in entries:
                    entries[eidx] = {"scalp": None, "runner": None, "all": []}
                entries[eidx]["all"].append(t)
                leg = t.get("leg", "unknown")
                if leg == "scalp":
                    entries[eidx]["scalp"] = t
                elif leg == "runner":
                    entries[eidx]["runner"] = t
            return entries

        best_entries = group_entries(best_trades)
        bl_entries = group_entries(bl_trades)

        # Compare entries that exist in both
        common_eidx = sorted(set(best_entries.keys()) & set(bl_entries.keys()))
        print(f"  Common entries: {len(common_eidx)}, "
              f"Best-only: {len(set(best_entries.keys()) - set(bl_entries.keys()))}, "
              f"Baseline-only: {len(set(bl_entries.keys()) - set(best_entries.keys()))}")

        improved = 0
        degraded = 0
        unchanged = 0
        total_improvement = 0.0
        improvements = []

        comm = COMMISSION_PER_SIDE * 2  # per leg

        for eidx in common_eidx:
            # Compute net PnL for each
            best_pnl = sum(t["pts"] * DOLLAR_PER_PT - comm for t in best_entries[eidx]["all"])
            bl_pnl = sum(t["pts"] * DOLLAR_PER_PT - comm for t in bl_entries[eidx]["all"])
            diff = best_pnl - bl_pnl
            total_improvement += diff

            # Runner exit reasons
            best_runner = best_entries[eidx].get("runner")
            bl_runner = bl_entries[eidx].get("runner")
            best_reason = best_runner["exit_reason"] if best_runner else "?"
            bl_reason = bl_runner["exit_reason"] if bl_runner else "?"

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
                "best_reason": best_reason,
                "bl_reason": bl_reason,
            })

        print(f"\n  Improved: {improved}, Degraded: {degraded}, Unchanged: {unchanged}")
        print(f"  Total PnL improvement: ${total_improvement:+.2f}")
        print(f"  Avg improvement per trade: ${total_improvement / len(common_eidx):+.2f}" if common_eidx else "")

        # Show top 10 improvements and top 10 degradations
        improvements.sort(key=lambda x: x["diff"], reverse=True)

        print(f"\n  Top 10 IMPROVED trades:")
        print(f"    {'EntryIdx':>8} {'Baseline$':>10} {'Structure$':>11} {'Diff$':>8} {'BL_Exit':>10} {'ST_Exit':>10}")
        for imp in improvements[:10]:
            print(f"    {imp['eidx']:>8} ${imp['bl_pnl']:>+9.2f} ${imp['best_pnl']:>+10.2f} "
                  f"${imp['diff']:>+7.2f} {imp['bl_reason']:>10} {imp['best_reason']:>10}")

        print(f"\n  Top 10 DEGRADED trades:")
        for imp in improvements[-10:]:
            print(f"    {imp['eidx']:>8} ${imp['bl_pnl']:>+9.2f} ${imp['best_pnl']:>+10.2f} "
                  f"${imp['diff']:>+7.2f} {imp['bl_reason']:>10} {imp['best_reason']:>10}")

    # ==================================================================
    # CLUSTER ANALYSIS BY SWING TYPE
    # ==================================================================
    print(f"\n{'='*130}")
    print("CLUSTER ANALYSIS")
    print(f"{'='*130}")

    # By swing_type
    print(f"\n  --- By Swing Type ---")
    print(f"  {'Type':>8} | {'Combos':>6} {'AvgPF':>6} {'AvgSharpe':>10} {'AvgP&L':>10} {'BestComp':>9}")
    print(f"  " + "-" * 65)
    for st in ["pivot", "donchian"]:
        sub = [r for r in results if r["swing_type"] == st]
        if sub:
            avg_pf = np.mean([r["sc"]["pf"] for r in sub])
            avg_sharpe = np.mean([r["sc"]["sharpe"] for r in sub])
            avg_pnl = np.mean([r["sc"]["net_dollar"] for r in sub])
            best_comp = max(r["composite"] for r in sub)
            print(f"  {st:>8} | {len(sub):>6} {avg_pf:>6.3f} {avg_sharpe:>10.3f} "
                  f"${avg_pnl:>+9.2f} {best_comp:>9.3f}")

    # By lookback
    print(f"\n  --- By Swing Lookback ---")
    print(f"  {'LB':>4} | {'Combos':>6} {'AvgPF':>6} {'AvgSharpe':>10} {'AvgP&L':>10} {'BestComp':>9}")
    print(f"  " + "-" * 55)
    for lb in SWING_LOOKBACKS:
        sub = [r for r in results if r["swing_lookback"] == lb]
        if sub:
            avg_pf = np.mean([r["sc"]["pf"] for r in sub])
            avg_sharpe = np.mean([r["sc"]["sharpe"] for r in sub])
            avg_pnl = np.mean([r["sc"]["net_dollar"] for r in sub])
            best_comp = max(r["composite"] for r in sub)
            print(f"  {lb:>4} | {len(sub):>6} {avg_pf:>6.3f} {avg_sharpe:>10.3f} "
                  f"${avg_pnl:>+9.2f} {best_comp:>9.3f}")

    # By max_tp2_pts (cap vs no cap)
    print(f"\n  --- By Max TP Cap ---")
    print(f"  {'Cap':>4} | {'Combos':>6} {'AvgPF':>6} {'AvgSharpe':>10} {'AvgP&L':>10} {'BestComp':>9}")
    print(f"  " + "-" * 55)
    for cap in MAX_TP2_VALUES:
        sub = [r for r in results if r["max_tp2_pts"] == cap]
        if sub:
            avg_pf = np.mean([r["sc"]["pf"] for r in sub])
            avg_sharpe = np.mean([r["sc"]["sharpe"] for r in sub])
            avg_pnl = np.mean([r["sc"]["net_dollar"] for r in sub])
            best_comp = max(r["composite"] for r in sub)
            cap_label = "none" if cap == 0 else str(cap)
            print(f"  {cap_label:>4} | {len(sub):>6} {avg_pf:>6.3f} {avg_sharpe:>10.3f} "
                  f"${avg_pnl:>+9.2f} {best_comp:>9.3f}")

    # By buffer
    print(f"\n  --- By Swing Buffer ---")
    print(f"  {'Buf':>4} | {'Combos':>6} {'AvgPF':>6} {'AvgSharpe':>10} {'AvgP&L':>10} {'BestComp':>9}")
    print(f"  " + "-" * 55)
    for buf in SWING_BUFFERS:
        sub = [r for r in results if r["swing_buffer_pts"] == buf]
        if sub:
            avg_pf = np.mean([r["sc"]["pf"] for r in sub])
            avg_sharpe = np.mean([r["sc"]["sharpe"] for r in sub])
            avg_pnl = np.mean([r["sc"]["net_dollar"] for r in sub])
            best_comp = max(r["composite"] for r in sub)
            print(f"  {buf:>4} | {len(sub):>6} {avg_pf:>6.3f} {avg_sharpe:>10.3f} "
                  f"${avg_pnl:>+9.2f} {best_comp:>9.3f}")

    # By min_profit_pts
    print(f"\n  --- By Min Profit Pts ---")
    print(f"  {'MinP':>4} | {'Combos':>6} {'AvgPF':>6} {'AvgSharpe':>10} {'AvgP&L':>10} {'BestComp':>9}")
    print(f"  " + "-" * 55)
    for mp in MIN_PROFITS:
        sub = [r for r in results if r["min_profit_pts"] == mp]
        if sub:
            avg_pf = np.mean([r["sc"]["pf"] for r in sub])
            avg_sharpe = np.mean([r["sc"]["sharpe"] for r in sub])
            avg_pnl = np.mean([r["sc"]["net_dollar"] for r in sub])
            best_comp = max(r["composite"] for r in sub)
            print(f"  {mp:>4} | {len(sub):>6} {avg_pf:>6.3f} {avg_sharpe:>10.3f} "
                  f"${avg_pnl:>+9.2f} {best_comp:>9.3f}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*130}")
    print(f"SWEEP COMPLETE in {elapsed:.1f}s")
    print(f"{'='*130}")

    # Quick IS/OOS pass/fail summary
    print(f"\nIS/OOS PASS/FAIL SUMMARY:")
    bl_is_pf = bl_is["pf"] if bl_is else 1.0
    bl_oos_pf = bl_oos["pf"] if bl_oos else 1.0
    pass_count = 0
    for r in isoos_results:
        is_sc = r["is_sc"]
        oos_sc = r["oos_sc"]
        if is_sc and oos_sc:
            oos_vs_bl = oos_sc["pf"] >= bl_oos_pf
            is_vs_oos = oos_sc["pf"] / is_sc["pf"] if is_sc["pf"] > 0 else 0
            passes = oos_vs_bl and is_vs_oos >= 0.7
            marker = "PASS" if passes else "FAIL"
            if passes:
                pass_count += 1
            print(f"  #{r['rank']:>2} {r['label']:>50} "
                  f"IS PF={is_sc['pf']:.3f} OOS PF={oos_sc['pf']:.3f} "
                  f"ratio={is_vs_oos:.2f} OOS>=BL={'Y' if oos_vs_bl else 'N'} --> {marker}")
    print(f"\n  {pass_count}/{len(isoos_results)} configs pass IS/OOS validation")


if __name__ == "__main__":
    main()
