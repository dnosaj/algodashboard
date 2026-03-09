"""
Fixed SL/TP Sweep for MNQ Strategies (vScalpA V15 + vScalpB)
=============================================================
Comprehensive grid sweep over TP and SL for both MNQ strategies.
Confirms current exits are optimal and maps the full landscape.

Outputs:
  1. PF heatmap grids (TP x SL)
  2. Top 20 combos per strategy by Sharpe
  3. IS/OOS validation for top 10
  4. Pareto frontier (Sharpe vs MaxDD)
  5. Strategy clusters (Scalp / Swing / Runner)

Usage:
    python3 fixed_exit_sweep.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# --- Path setup (must match project convention) ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    resample_to_5min,
    map_5min_rsi_to_1min,
    score_trades,
)

from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS, VSCALPA_ENTRY_END_ET,
    VSCALPB_RSI_LEN, VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
    VSCALPB_SM_THRESHOLD, VSCALPB_COOLDOWN,
    VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
)


# ---- Sweep grids ----
V15_TP_VALUES = [3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50]
V15_SL_VALUES = [15, 20, 25, 30, 35, 40, 50, 60, 75]

VSCALPB_TP_VALUES = [3, 4, 5, 7, 10, 12, 15, 20, 25, 30]
VSCALPB_SL_VALUES = [10, 12, 15, 20, 25, 30, 35, 40, 50]


def compute_daily_sharpe(trades, dollar_per_pt, commission):
    """Annualized Sharpe from actual daily P&L aggregation."""
    if not trades:
        return 0.0
    daily_pnl = defaultdict(float)
    for t in trades:
        day = str(t["entry_time"])[:10]
        pnl = t["pts"] * dollar_per_pt - 2 * commission
        daily_pnl[day] += pnl
    vals = np.array(list(daily_pnl.values()))
    if len(vals) < 2 or np.std(vals) == 0:
        return 0.0
    return float(np.mean(vals) / np.std(vals) * np.sqrt(252))


def compute_max_dd_dollar(trades, dollar_per_pt, commission):
    """Max drawdown in dollars from trade-level P&L."""
    if not trades:
        return 0.0
    pnls = np.array([t["pts"] * dollar_per_pt - 2 * commission for t in trades])
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def run_v15(arrays, tp, sl):
    """Run vScalpA V15 with given TP/SL."""
    opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev = arrays
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=sl, tp_pts=tp,
        entry_end_et=VSCALPA_ENTRY_END_ET,
    )
    return trades


def run_vscalpb(arrays, tp, sl):
    """Run vScalpB with given TP/SL."""
    opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev = arrays
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=sl, tp_pts=tp,
    )
    return trades


def score_combo(trades, dollar_per_pt, commission):
    """Return dict of metrics for a TP/SL combo."""
    if not trades:
        return {
            "trades": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0,
            "sharpe": 0.0, "max_dd": 0.0,
        }
    pnls = np.array([t["pts"] * dollar_per_pt - 2 * commission for t in trades])
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0.0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0.0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100 if n > 0 else 0.0

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float((cum - peak).min())

    sharpe = compute_daily_sharpe(trades, dollar_per_pt, commission)

    return {
        "trades": n,
        "wr": round(wr, 1),
        "pf": round(pf, 3),
        "pnl": round(float(pnls.sum()), 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
    }


def prepare_arrays(df, rsi_len):
    """Prepare backtest arrays from a dataframe (SM already in df)."""
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    sm = df["SM_Net"].values
    times = df.index

    df_5m = resample_to_5min(df)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values,
        df_5m["Close"].values, rsi_len=rsi_len,
    )
    return (opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev)


def print_pf_grid(results, tp_values, sl_values, baseline_tp, baseline_sl, label):
    """Print a TP x SL grid of PF values."""
    print(f"\n{'=' * 80}")
    print(f"  {label} — Profit Factor Grid (TP rows x SL columns)")
    print(f"  Baseline: TP={baseline_tp}, SL={baseline_sl}")
    print(f"{'=' * 80}")

    # Header
    hdr = f"{'TP':>6}"
    for sl in sl_values:
        hdr += f"  SL={sl:>2}"
    print(hdr)
    print("-" * len(hdr))

    for tp in tp_values:
        row = f"{tp:>6}"
        for sl in sl_values:
            key = (tp, sl)
            if key in results:
                pf = results[key]["pf"]
                marker = " *" if (tp == baseline_tp and sl == baseline_sl) else "  "
                if pf >= 999:
                    row += f"  {'INF':>5}{marker}"
                else:
                    row += f"  {pf:>5.2f}{marker}"
            else:
                row += f"  {'N/A':>5}  "
        print(row)

    print("\n  * = current production baseline")


def print_top_n(results_list, n, label):
    """Print top N combos sorted by Sharpe."""
    sorted_results = sorted(results_list, key=lambda x: x["sharpe"], reverse=True)
    top = sorted_results[:n]

    print(f"\n{'=' * 80}")
    print(f"  {label} — Top {n} by Sharpe")
    print(f"{'=' * 80}")
    print(f"{'Rank':>4}  {'TP':>4}  {'SL':>4}  {'Trades':>6}  {'WR%':>6}  {'PF':>7}  "
          f"{'P&L':>9}  {'Sharpe':>7}  {'MaxDD':>9}")
    print("-" * 75)
    for i, r in enumerate(top, 1):
        print(f"{i:>4}  {r['tp']:>4}  {r['sl']:>4}  {r['trades']:>6}  {r['wr']:>5.1f}%  "
              f"{r['pf']:>7.3f}  ${r['pnl']:>8.2f}  {r['sharpe']:>7.3f}  ${r['max_dd']:>8.2f}")

    return top


def print_is_oos_validation(top_combos, run_fn, arrays_is, arrays_oos,
                            dollar_per_pt, commission, label):
    """IS/OOS validation for top combos."""
    print(f"\n{'=' * 80}")
    print(f"  {label} — IS/OOS Validation (top 10)")
    print(f"{'=' * 80}")
    print(f"{'Rank':>4}  {'TP':>4}  {'SL':>4}  "
          f"{'IS PF':>7}  {'OOS PF':>7}  {'IS Sharpe':>9}  {'OOS Sharpe':>10}  "
          f"{'IS P&L':>9}  {'OOS P&L':>9}  {'Status':>8}")
    print("-" * 95)

    validated = []
    for i, combo in enumerate(top_combos[:10], 1):
        tp, sl = combo["tp"], combo["sl"]

        trades_is = run_fn(arrays_is, tp, sl)
        trades_oos = run_fn(arrays_oos, tp, sl)

        sc_is = score_combo(trades_is, dollar_per_pt, commission)
        sc_oos = score_combo(trades_oos, dollar_per_pt, commission)

        # Status flags
        status = ""
        if sc_oos["pf"] > 0 and sc_is["pf"] > 0:
            if sc_oos["pf"] >= sc_is["pf"]:
                status = "STRONG"
            elif sc_oos["pf"] >= 0.9 * sc_is["pf"]:
                status = "STABLE"

        print(f"{i:>4}  {tp:>4}  {sl:>4}  "
              f"{sc_is['pf']:>7.3f}  {sc_oos['pf']:>7.3f}  "
              f"{sc_is['sharpe']:>9.3f}  {sc_oos['sharpe']:>10.3f}  "
              f"${sc_is['pnl']:>8.2f}  ${sc_oos['pnl']:>8.2f}  "
              f"{status:>8}")

        validated.append({
            "tp": tp, "sl": sl,
            "is_pf": sc_is["pf"], "oos_pf": sc_oos["pf"],
            "is_sharpe": sc_is["sharpe"], "oos_sharpe": sc_oos["sharpe"],
            "is_pnl": sc_is["pnl"], "oos_pnl": sc_oos["pnl"],
            "status": status,
        })

    return validated


def print_pareto_frontier(results_list, label):
    """Find and print Pareto-optimal combos (max Sharpe, min |MaxDD|)."""
    # Filter to combos with positive Sharpe and trades
    candidates = [r for r in results_list if r["sharpe"] > 0 and r["trades"] > 0]

    pareto = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            # other dominates c if higher Sharpe AND less severe drawdown
            if other["sharpe"] > c["sharpe"] and other["max_dd"] > c["max_dd"]:
                dominated = True
                break
        if not dominated:
            pareto.append(c)

    pareto.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"  {label} — Pareto Frontier (Sharpe vs MaxDD)")
    print(f"  {len(pareto)} non-dominated combos out of {len(candidates)} with positive Sharpe")
    print(f"{'=' * 80}")
    print(f"{'#':>3}  {'TP':>4}  {'SL':>4}  {'Trades':>6}  {'WR%':>6}  {'PF':>7}  "
          f"{'P&L':>9}  {'Sharpe':>7}  {'MaxDD':>9}")
    print("-" * 72)
    for i, r in enumerate(pareto, 1):
        print(f"{i:>3}  {r['tp']:>4}  {r['sl']:>4}  {r['trades']:>6}  {r['wr']:>5.1f}%  "
              f"{r['pf']:>7.3f}  ${r['pnl']:>8.2f}  {r['sharpe']:>7.3f}  ${r['max_dd']:>8.2f}")

    return pareto


def print_clusters(results_list, label):
    """Group by TP cluster and show aggregate stats."""
    clusters = {
        "Scalp (TP 3-7)": [r for r in results_list if 3 <= r["tp"] <= 7],
        "Swing (TP 10-20)": [r for r in results_list if 10 <= r["tp"] <= 20],
        "Runner (TP 25+)": [r for r in results_list if r["tp"] >= 25],
    }

    print(f"\n{'=' * 80}")
    print(f"  {label} — Strategy Clusters by TP Size")
    print(f"{'=' * 80}")
    print(f"{'Cluster':<18}  {'Combos':>6}  {'Avg Trades':>10}  {'Avg WR%':>8}  "
          f"{'Avg PF':>7}  {'Avg P&L':>9}  {'Avg Sharpe':>10}  {'Avg MaxDD':>10}  "
          f"{'Best Sharpe':>11}")
    print("-" * 110)

    for name, members in clusters.items():
        if not members:
            print(f"{name:<18}  {'N/A':>6}")
            continue
        # Filter to combos with trades
        active = [m for m in members if m["trades"] > 0]
        if not active:
            print(f"{name:<18}  {len(members):>6}  {'no trades':>10}")
            continue

        avg_trades = np.mean([m["trades"] for m in active])
        avg_wr = np.mean([m["wr"] for m in active])
        pfs = [m["pf"] for m in active if m["pf"] < 999]
        avg_pf = np.mean(pfs) if pfs else 0.0
        avg_pnl = np.mean([m["pnl"] for m in active])
        avg_sharpe = np.mean([m["sharpe"] for m in active])
        avg_dd = np.mean([m["max_dd"] for m in active])
        best_sharpe = max(m["sharpe"] for m in active)

        print(f"{name:<18}  {len(active):>6}  {avg_trades:>10.0f}  {avg_wr:>7.1f}%  "
              f"{avg_pf:>7.3f}  ${avg_pnl:>8.0f}  {avg_sharpe:>10.3f}  ${avg_dd:>9.0f}  "
              f"{best_sharpe:>11.3f}")


def main():
    print("=" * 80)
    print("  FIXED SL/TP SWEEP — MNQ Strategies (vScalpA V15 + vScalpB)")
    print(f"  V15: {len(V15_TP_VALUES)} TP x {len(V15_SL_VALUES)} SL = "
          f"{len(V15_TP_VALUES) * len(V15_SL_VALUES)} combos")
    print(f"  vScalpB: {len(VSCALPB_TP_VALUES)} TP x {len(VSCALPB_SL_VALUES)} SL = "
          f"{len(VSCALPB_TP_VALUES) * len(VSCALPB_SL_VALUES)} combos")
    print("=" * 80)

    # ---- Load MNQ data ----
    print("\n[1] Loading MNQ data...")
    df_mnq = load_instrument_1min("MNQ")
    print(f"    {len(df_mnq)} bars, {df_mnq.index[0]} to {df_mnq.index[-1]}")

    # Recompute SM with MNQ params (10/12/200/100)
    print("    Computing SM with MNQ params (10/12/200/100)...")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm

    # ---- IS/OOS split ----
    midpoint = df_mnq.index[len(df_mnq) // 2]
    df_is = df_mnq[df_mnq.index < midpoint]
    df_oos = df_mnq[df_mnq.index >= midpoint]
    print(f"    IS:  {df_is.index[0].date()} to {df_is.index[-1].date()} ({len(df_is)} bars)")
    print(f"    OOS: {df_oos.index[0].date()} to {df_oos.index[-1].date()} ({len(df_oos)} bars)")

    # ---- Prepare arrays ----
    print("\n[2] Preparing arrays (RSI mapping)...")

    # V15 uses RSI len 8, vScalpB also uses RSI len 8 but with different buy/sell thresholds
    # Both share the same SM and data, just different RSI len (both 8 here)
    arrays_full_v15 = prepare_arrays(df_mnq, VSCALPA_RSI_LEN)
    arrays_is_v15 = prepare_arrays(df_is, VSCALPA_RSI_LEN)
    arrays_oos_v15 = prepare_arrays(df_oos, VSCALPA_RSI_LEN)

    arrays_full_vb = prepare_arrays(df_mnq, VSCALPB_RSI_LEN)
    arrays_is_vb = prepare_arrays(df_is, VSCALPB_RSI_LEN)
    arrays_oos_vb = prepare_arrays(df_oos, VSCALPB_RSI_LEN)

    print("    Done.")

    # ================================================================
    #  vScalpA V15 SWEEP
    # ================================================================
    print(f"\n[3] Running vScalpA V15 sweep ({len(V15_TP_VALUES) * len(V15_SL_VALUES)} combos)...")
    v15_grid = {}       # (tp, sl) -> metrics dict
    v15_results = []    # flat list for sorting

    total = len(V15_TP_VALUES) * len(V15_SL_VALUES)
    done = 0
    for tp in V15_TP_VALUES:
        for sl in V15_SL_VALUES:
            trades = run_v15(arrays_full_v15, tp, sl)
            sc = score_combo(trades, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
            sc["tp"] = tp
            sc["sl"] = sl
            v15_grid[(tp, sl)] = sc
            v15_results.append(sc)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"    {done}/{total} done...")

    # ================================================================
    #  vScalpB SWEEP
    # ================================================================
    print(f"\n[4] Running vScalpB sweep ({len(VSCALPB_TP_VALUES) * len(VSCALPB_SL_VALUES)} combos)...")
    vb_grid = {}
    vb_results = []

    total = len(VSCALPB_TP_VALUES) * len(VSCALPB_SL_VALUES)
    done = 0
    for tp in VSCALPB_TP_VALUES:
        for sl in VSCALPB_SL_VALUES:
            trades = run_vscalpb(arrays_full_vb, tp, sl)
            sc = score_combo(trades, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
            sc["tp"] = tp
            sc["sl"] = sl
            vb_grid[(tp, sl)] = sc
            vb_results.append(sc)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"    {done}/{total} done...")

    # ================================================================
    #  OUTPUT
    # ================================================================

    # ---- 1. PF Grid ----
    print_pf_grid(v15_grid, V15_TP_VALUES, V15_SL_VALUES,
                  VSCALPA_TP_PTS, VSCALPA_MAX_LOSS_PTS, "vScalpA V15")
    print_pf_grid(vb_grid, VSCALPB_TP_VALUES, VSCALPB_SL_VALUES,
                  VSCALPB_TP_PTS, VSCALPB_MAX_LOSS_PTS, "vScalpB")

    # ---- Also print Sharpe grids ----
    print(f"\n{'=' * 80}")
    print(f"  vScalpA V15 — Sharpe Grid (TP rows x SL columns)")
    print(f"{'=' * 80}")
    hdr = f"{'TP':>6}"
    for sl in V15_SL_VALUES:
        hdr += f"  SL={sl:>2}"
    print(hdr)
    print("-" * len(hdr))
    for tp in V15_TP_VALUES:
        row = f"{tp:>6}"
        for sl in V15_SL_VALUES:
            key = (tp, sl)
            if key in v15_grid:
                s = v15_grid[key]["sharpe"]
                marker = " *" if (tp == VSCALPA_TP_PTS and sl == VSCALPA_MAX_LOSS_PTS) else "  "
                row += f"  {s:>5.2f}{marker}"
            else:
                row += f"  {'N/A':>5}  "
        print(row)
    print("\n  * = current production baseline")

    print(f"\n{'=' * 80}")
    print(f"  vScalpB — Sharpe Grid (TP rows x SL columns)")
    print(f"{'=' * 80}")
    hdr = f"{'TP':>6}"
    for sl in VSCALPB_SL_VALUES:
        hdr += f"  SL={sl:>2}"
    print(hdr)
    print("-" * len(hdr))
    for tp in VSCALPB_TP_VALUES:
        row = f"{tp:>6}"
        for sl in VSCALPB_SL_VALUES:
            key = (tp, sl)
            if key in vb_grid:
                s = vb_grid[key]["sharpe"]
                marker = " *" if (tp == VSCALPB_TP_PTS and sl == VSCALPB_MAX_LOSS_PTS) else "  "
                row += f"  {s:>5.2f}{marker}"
            else:
                row += f"  {'N/A':>5}  "
        print(row)
    print("\n  * = current production baseline")

    # ---- 2. Top 20 ----
    print("\n\n" + "#" * 80)
    print("#  TOP 20 RANKINGS")
    print("#" * 80)
    v15_top = print_top_n(v15_results, 20, "vScalpA V15")
    vb_top = print_top_n(vb_results, 20, "vScalpB")

    # ---- 3. IS/OOS Validation ----
    print("\n\n" + "#" * 80)
    print("#  IS/OOS VALIDATION")
    print("#" * 80)
    v15_validated = print_is_oos_validation(
        v15_top, run_v15, arrays_is_v15, arrays_oos_v15,
        MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpA V15")
    vb_validated = print_is_oos_validation(
        vb_top, run_vscalpb, arrays_is_vb, arrays_oos_vb,
        MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpB")

    # ---- 4. Pareto Frontier ----
    print("\n\n" + "#" * 80)
    print("#  PARETO FRONTIER")
    print("#" * 80)
    print_pareto_frontier(v15_results, "vScalpA V15")
    print_pareto_frontier(vb_results, "vScalpB")

    # ---- 5. Strategy Clusters ----
    print("\n\n" + "#" * 80)
    print("#  STRATEGY CLUSTERS")
    print("#" * 80)
    print_clusters(v15_results, "vScalpA V15")
    print_clusters(vb_results, "vScalpB")

    # ---- Summary ----
    print("\n\n" + "=" * 80)
    print("  SUMMARY — Current Baseline vs Best")
    print("=" * 80)

    v15_baseline = v15_grid.get((VSCALPA_TP_PTS, VSCALPA_MAX_LOSS_PTS))
    v15_best_sharpe = max(v15_results, key=lambda x: x["sharpe"])
    v15_best_pf = max([r for r in v15_results if r["pf"] < 999], key=lambda x: x["pf"])

    print(f"\n  vScalpA V15:")
    print(f"    Baseline (TP={VSCALPA_TP_PTS}, SL={VSCALPA_MAX_LOSS_PTS}): "
          f"{v15_baseline['trades']} trades, WR {v15_baseline['wr']}%, "
          f"PF {v15_baseline['pf']}, ${v15_baseline['pnl']:.0f}, "
          f"Sharpe {v15_baseline['sharpe']}, MaxDD ${v15_baseline['max_dd']:.0f}")
    print(f"    Best Sharpe (TP={v15_best_sharpe['tp']}, SL={v15_best_sharpe['sl']}): "
          f"{v15_best_sharpe['trades']} trades, WR {v15_best_sharpe['wr']}%, "
          f"PF {v15_best_sharpe['pf']}, ${v15_best_sharpe['pnl']:.0f}, "
          f"Sharpe {v15_best_sharpe['sharpe']}, MaxDD ${v15_best_sharpe['max_dd']:.0f}")
    print(f"    Best PF (TP={v15_best_pf['tp']}, SL={v15_best_pf['sl']}): "
          f"{v15_best_pf['trades']} trades, WR {v15_best_pf['wr']}%, "
          f"PF {v15_best_pf['pf']}, ${v15_best_pf['pnl']:.0f}, "
          f"Sharpe {v15_best_pf['sharpe']}, MaxDD ${v15_best_pf['max_dd']:.0f}")

    vb_baseline = vb_grid.get((VSCALPB_TP_PTS, VSCALPB_MAX_LOSS_PTS))
    vb_best_sharpe = max(vb_results, key=lambda x: x["sharpe"])
    vb_best_pf = max([r for r in vb_results if r["pf"] < 999], key=lambda x: x["pf"])

    print(f"\n  vScalpB:")
    print(f"    Baseline (TP={VSCALPB_TP_PTS}, SL={VSCALPB_MAX_LOSS_PTS}): "
          f"{vb_baseline['trades']} trades, WR {vb_baseline['wr']}%, "
          f"PF {vb_baseline['pf']}, ${vb_baseline['pnl']:.0f}, "
          f"Sharpe {vb_baseline['sharpe']}, MaxDD ${vb_baseline['max_dd']:.0f}")
    print(f"    Best Sharpe (TP={vb_best_sharpe['tp']}, SL={vb_best_sharpe['sl']}): "
          f"{vb_best_sharpe['trades']} trades, WR {vb_best_sharpe['wr']}%, "
          f"PF {vb_best_sharpe['pf']}, ${vb_best_sharpe['pnl']:.0f}, "
          f"Sharpe {vb_best_sharpe['sharpe']}, MaxDD ${vb_best_sharpe['max_dd']:.0f}")
    print(f"    Best PF (TP={vb_best_pf['tp']}, SL={vb_best_pf['sl']}): "
          f"{vb_best_pf['trades']} trades, WR {vb_best_pf['wr']}%, "
          f"PF {vb_best_pf['pf']}, ${vb_best_pf['pnl']:.0f}, "
          f"Sharpe {vb_best_pf['sharpe']}, MaxDD ${vb_best_pf['max_dd']:.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
