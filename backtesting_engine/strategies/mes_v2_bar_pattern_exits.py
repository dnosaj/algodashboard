"""
MES v2 Bar Pattern Exit Sweep (Studies 7a/7b/7c)
==================================================
Tests engulfing bar and consecutive adverse bar exit patterns for MES v2.

Study 7a: Engulfing Bar Exit
  - Sweep engulf_exit_pts = [0, 2, 4, 6, 8]
  - Exits on 1st bar after entry if bar body reverses by > X pts

Study 7b: Consecutive Adverse Bars
  - Sweep consec_adverse_bars = [0, 2, 3, 4]
  - Exits if N consecutive bars close against position

Study 7c: Combined
  - Best engulf + best consec together (only if both 7a and 7b pass)

All runs hold breakeven_after_bars=75 constant.
Includes counterfactual analysis (save rate vs false positive rate).

Usage:
    python3 mes_v2_bar_pattern_exits.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_test_common import (
    compute_smart_money,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
    compute_mfe_mae,
    MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA,
    MES_DOLLAR_PER_PT, MES_COMMISSION,
    MESV2_RSI_LEN, MESV2_RSI_BUY, MESV2_RSI_SELL,
    MESV2_SM_THRESHOLD, MESV2_COOLDOWN,
    MESV2_MAX_LOSS_PTS, MESV2_TP_PTS, MESV2_EOD_ET,
    MESV2_BREAKEVEN_BARS,
)

from results.save_results import save_backtest

ENGULF_SWEEP = [0, 2, 4, 6, 8]
CONSEC_SWEEP = [0, 2, 3, 4]
SCRIPT_NAME = "mes_v2_bar_pattern_exits.py"

# Pass/fail criteria
PF_IMPROVEMENT_THRESHOLD = 0.05   # 5% PF improvement over baseline
MIN_TRADE_COUNT_RATIO = 0.70      # Must retain at least 70% of baseline trades


def run_single(opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, engulf_pts=0, consec_bars=0):
    """Run MES v2 backtest with specific engulf/consec parameters."""
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        engulf_exit_pts=engulf_pts,
        consec_adverse_bars=consec_bars,
    )
    compute_mfe_mae(trades, highs, lows)
    return trades


def build_counterfactual(baseline_trades, filtered_trades, exit_reason_key):
    """Compare filtered trades against baseline to compute save rate and false positive rate.

    For each trade that exited via exit_reason_key in filtered run, find the
    matching baseline trade (by entry_time) and check if the baseline trade
    was a winner or loser. This tells us:
      - Save rate: % of early exits that would have been losers in baseline
      - False positive rate: % of early exits that would have been winners in baseline

    Returns dict with stats, or None if no early exits.
    """
    if not baseline_trades or not filtered_trades:
        return None

    # Build baseline lookup by entry_time
    baseline_by_entry = {}
    for t in baseline_trades:
        key = str(t["entry_time"])
        baseline_by_entry[key] = t

    # Commission in pts
    comm_pts = (MES_COMMISSION * 2) / MES_DOLLAR_PER_PT

    # Find trades that exited via the filter
    early_exits = [t for t in filtered_trades if t["result"] == exit_reason_key]
    if not early_exits:
        return None

    saved = 0       # Would have been losers in baseline
    false_pos = 0   # Would have been winners in baseline
    saved_pts = 0.0
    false_pos_pts = 0.0

    for t in early_exits:
        key = str(t["entry_time"])
        bl = baseline_by_entry.get(key)
        if bl is None:
            # Trade exists in filtered but not baseline (re-entry effect) -- skip
            continue
        bl_net = bl["pts"] - comm_pts
        if bl_net <= 0:
            saved += 1
            saved_pts += abs(bl_net)  # pts we avoided losing
        else:
            false_pos += 1
            false_pos_pts += bl_net   # pts we gave up

    total_matched = saved + false_pos
    if total_matched == 0:
        return None

    return {
        "total_early_exits": len(early_exits),
        "matched": total_matched,
        "saved": saved,
        "false_pos": false_pos,
        "save_rate": saved / total_matched * 100,
        "false_pos_rate": false_pos / total_matched * 100,
        "saved_pts": round(saved_pts, 2),
        "false_pos_pts": round(false_pos_pts, 2),
        "net_pts_benefit": round(saved_pts - false_pos_pts, 2),
    }


def print_results_table(results, sweep_param_name, sweep_values, baseline_counts,
                         baseline_pf, exit_reason_key):
    """Print a formatted results table for a sweep."""
    print(f"\n{'='*140}")
    print(f"{sweep_param_name:>8} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{exit_reason_key:>8} {'Re-entry':>9} {'dPF%':>7} {'dTrades%':>9}")
    print(f"{'-'*140}")

    for val, split_name, dr, sc, trades in results:
        if sc is None:
            print(f"{val:>8} {split_name:>5}   NO TRADES")
            continue

        exit_count = sc["exits"].get(exit_reason_key, 0)
        baseline = baseline_counts.get(split_name, 0)
        re_entry = sc["count"] - baseline

        bl_pf = baseline_pf.get(split_name, 0.0)
        dpf_pct = ((sc["pf"] - bl_pf) / bl_pf * 100) if bl_pf > 0 else 0.0
        bl_count = baseline_counts.get(split_name, 0)
        dtrades_pct = ((sc["count"] - bl_count) / bl_count * 100) if bl_count > 0 else 0.0

        print(f"{val:>8} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{exit_count:>8} {re_entry:>+9} "
              f"{dpf_pct:>+6.1f}% {dtrades_pct:>+8.1f}%")


def print_exit_breakdown(results, sweep_param_name):
    """Print exit reason breakdown for FULL data."""
    print(f"\n{'='*80}")
    print(f"EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")
    print(f"{sweep_param_name:>8} ", end="")
    all_reasons = set()
    for val, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())
    for reason in sorted(all_reasons):
        print(f"{reason:>10}", end="")
    print()
    print("-" * (8 + 10 * len(all_reasons)))

    for val, split_name, _, sc, _ in results:
        if split_name != "FULL":
            continue
        if sc is None:
            continue
        print(f"{val:>8} ", end="")
        for reason in sorted(all_reasons):
            print(f"{sc['exits'].get(reason, 0):>10}", end="")
        print()


def run_pass_fail(results, sweep_values, baseline_value, baseline_pf, baseline_counts):
    """Run pass/fail analysis. Returns list of passing configs and result lookup."""
    result_lookup = {}
    for val, split_name, dr, sc, trades in results:
        result_lookup[(val, split_name)] = sc

    passed_configs = []
    print(f"\n{'Val':>8} {'IS_PF':>7} {'OOS_PF':>7} {'IS_dPF%':>8} {'OOS_dPF%':>9} "
          f"{'IS_Trd':>7} {'OOS_Trd':>8} {'IS_Trd%':>8} {'OOS_Trd%':>9} {'Result':>10}")
    print("-" * 100)

    for val in sweep_values:
        is_sc = result_lookup.get((val, "IS"))
        oos_sc = result_lookup.get((val, "OOS"))

        is_pf = is_sc["pf"] if is_sc else 0.0
        oos_pf = oos_sc["pf"] if oos_sc else 0.0
        is_count = is_sc["count"] if is_sc else 0
        oos_count = oos_sc["count"] if oos_sc else 0

        bl_is_pf = baseline_pf.get("IS", 0.0)
        bl_oos_pf = baseline_pf.get("OOS", 0.0)
        bl_is_count = baseline_counts.get("IS", 0)
        bl_oos_count = baseline_counts.get("OOS", 0)

        is_dpf = ((is_pf - bl_is_pf) / bl_is_pf) if bl_is_pf > 0 else 0.0
        oos_dpf = ((oos_pf - bl_oos_pf) / bl_oos_pf) if bl_oos_pf > 0 else 0.0
        is_trd_ratio = (is_count / bl_is_count) if bl_is_count > 0 else 0.0
        oos_trd_ratio = (oos_count / bl_oos_count) if bl_oos_count > 0 else 0.0

        pf_pass = (is_dpf >= PF_IMPROVEMENT_THRESHOLD and oos_dpf >= PF_IMPROVEMENT_THRESHOLD)
        count_pass = (is_trd_ratio >= MIN_TRADE_COUNT_RATIO and oos_trd_ratio >= MIN_TRADE_COUNT_RATIO)
        passed = pf_pass and count_pass

        status = "PASS" if passed else "FAIL"
        if val == baseline_value:
            status = "BASELINE"
        if passed:
            passed_configs.append(val)

        print(f"{val:>8} {is_pf:>7.3f} {oos_pf:>7.3f} "
              f"{is_dpf:>+7.1%} {oos_dpf:>+8.1%} "
              f"{is_count:>7} {oos_count:>8} "
              f"{is_trd_ratio:>7.0%} {oos_trd_ratio:>8.0%} "
              f"{status:>10}")

    return passed_configs, result_lookup


def run_monotonicity(result_lookup, sweep_values):
    """Check monotonicity of PF and Sharpe across sweep values."""
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK")
    print(f"{'='*80}")

    for split_check in ["IS", "OOS"]:
        pf_series = []
        sharpe_series = []
        for val in sweep_values:
            sc = result_lookup.get((val, split_check))
            pf_series.append(sc["pf"] if sc else 0.0)
            sharpe_series.append(sc["sharpe"] if sc else 0.0)

        pf_diffs = [pf_series[i+1] - pf_series[i] for i in range(len(pf_series)-1)]
        pf_sign_changes = sum(1 for i in range(len(pf_diffs)-1)
                              if pf_diffs[i] * pf_diffs[i+1] < 0)

        sharpe_diffs = [sharpe_series[i+1] - sharpe_series[i] for i in range(len(sharpe_series)-1)]
        sharpe_sign_changes = sum(1 for i in range(len(sharpe_diffs)-1)
                                  if sharpe_diffs[i] * sharpe_diffs[i+1] < 0)

        pf_smooth = "SMOOTH" if pf_sign_changes <= 1 else f"NOISY ({pf_sign_changes} reversals)"
        sharpe_smooth = "SMOOTH" if sharpe_sign_changes <= 1 else f"NOISY ({sharpe_sign_changes} reversals)"

        print(f"\n  {split_check}:")
        print(f"    PF trend:     {' -> '.join(f'{p:.3f}' for p in pf_series)}")
        print(f"    PF smoothness: {pf_smooth}")
        print(f"    Sharpe trend: {' -> '.join(f'{s:.3f}' for s in sharpe_series)}")
        print(f"    Sharpe smoothness: {sharpe_smooth}")


def print_counterfactual(results, sweep_values, baseline_value, exit_reason_key, label):
    """Print counterfactual analysis for early exits."""
    print(f"\n{'='*80}")
    print(f"COUNTERFACTUAL: {label}")
    print(f"  For each {exit_reason_key} exit, what would baseline have done?")
    print(f"{'='*80}")

    # Get baseline trades per split
    baseline_trades = {}
    for val, split_name, dr, sc, trades in results:
        if val == baseline_value:
            baseline_trades[split_name] = trades

    print(f"\n{'Val':>8} {'Split':>5} {'EarlyEx':>8} {'Matched':>8} "
          f"{'Saved':>6} {'FalsePos':>9} {'SaveRate':>9} {'FPRate':>8} "
          f"{'SavedPts':>9} {'FPPts':>8} {'NetBenefit':>11}")
    print("-" * 110)

    for val in sweep_values:
        if val == baseline_value:
            continue
        for split_name in ["FULL", "IS", "OOS"]:
            # Find filtered trades for this val + split
            filtered_trades = None
            for v, sn, dr, sc, trades in results:
                if v == val and sn == split_name:
                    filtered_trades = trades
                    break

            bl_trades = baseline_trades.get(split_name)
            cf = build_counterfactual(bl_trades, filtered_trades, exit_reason_key)
            if cf is None:
                print(f"{val:>8} {split_name:>5}      --- no {exit_reason_key} exits ---")
                continue

            print(f"{val:>8} {split_name:>5} {cf['total_early_exits']:>8} {cf['matched']:>8} "
                  f"{cf['saved']:>6} {cf['false_pos']:>9} "
                  f"{cf['save_rate']:>8.1f}% {cf['false_pos_rate']:>7.1f}% "
                  f"{cf['saved_pts']:>9.2f} {cf['false_pos_pts']:>8.2f} "
                  f"{cf['net_pts_benefit']:>+11.2f}")


def main():
    print("=" * 80)
    print("MES v2 BAR PATTERN EXIT SWEEP (Studies 7a / 7b / 7c)")
    print(f"  breakeven_after_bars={MESV2_BREAKEVEN_BARS} (held constant)")
    print("=" * 80)

    # --- Load MES data ---
    print("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    print(f"  {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # Compute SM on full data (before split -- matches existing pipeline)
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm

    # Data range
    start_date = df_mes.index[0].strftime("%Y-%m-%d")
    end_date = df_mes.index[-1].strftime("%Y-%m-%d")
    data_range = f"{start_date}_to_{end_date}"

    # --- Define splits ---
    midpoint = df_mes.index[len(df_mes) // 2]
    mes_is = df_mes[df_mes.index < midpoint]
    mes_oos = df_mes[df_mes.index >= midpoint]

    is_range = f"{mes_is.index[0].strftime('%Y-%m-%d')}_to_{mes_is.index[-1].strftime('%Y-%m-%d')}"
    oos_range = f"{mes_oos.index[0].strftime('%Y-%m-%d')}_to_{mes_oos.index[-1].strftime('%Y-%m-%d')}"

    splits = [
        ("FULL", df_mes, data_range),
        ("IS", mes_is, is_range),
        ("OOS", mes_oos, oos_range),
    ]

    # --- Prepare arrays per split (RSI mapped within each split) ---
    split_arrays = {}
    for split_name, df_split, _ in splits:
        opens = df_split["Open"].values
        highs = df_split["High"].values
        lows = df_split["Low"].values
        closes = df_split["Close"].values
        sm_arr = df_split["SM_Net"].values
        times = df_split.index

        df_5m = resample_to_5min(df_split)
        rsi_curr, rsi_prev = map_5min_rsi_to_1min(
            df_split.index.values, df_5m.index.values,
            df_5m["Close"].values, rsi_len=MESV2_RSI_LEN,
        )
        split_arrays[split_name] = (opens, highs, lows, closes, sm_arr, times,
                                     rsi_curr, rsi_prev)

    # =========================================================================
    # STUDY 7a: ENGULFING BAR EXIT SWEEP
    # =========================================================================
    print(f"\n{'#'*80}")
    print(f"# STUDY 7a: ENGULFING BAR EXIT SWEEP")
    print(f"#   Sweep: engulf_exit_pts = {ENGULF_SWEEP}")
    print(f"#   Exit if 1st bar after entry reverses by > X pts")
    print(f"{'#'*80}")

    engulf_results = []
    baseline_counts_7a = {}
    baseline_pf_7a = {}
    best_oos_sharpe_7a = -999
    best_engulf = 0

    for engulf_val in ENGULF_SWEEP:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, engulf_pts=engulf_val, consec_bars=0)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            if engulf_val == 0:
                baseline_counts_7a[split_name] = sc["count"] if sc else 0
                baseline_pf_7a[split_name] = sc["pf"] if sc else 0.0

            engulf_results.append((engulf_val, split_name, dr, sc, trades))

            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe_7a:
                best_oos_sharpe_7a = sc["sharpe"]
                best_engulf = engulf_val

    # Print results
    print_results_table(engulf_results, "Engulf", ENGULF_SWEEP,
                        baseline_counts_7a, baseline_pf_7a, "ENGULF")
    print_exit_breakdown(engulf_results, "Engulf")

    # Counterfactual
    print_counterfactual(engulf_results, ENGULF_SWEEP, 0, "ENGULF",
                         "Engulfing Bar Exits")

    # Pass/fail
    print(f"\n{'='*80}")
    print("STUDY 7a PASS/FAIL ANALYSIS")
    print(f"  Criteria: IS & OOS PF improvement >= {PF_IMPROVEMENT_THRESHOLD*100:.0f}%")
    print(f"            Per-half trade count >= {MIN_TRADE_COUNT_RATIO*100:.0f}% of baseline")
    print(f"{'='*80}")

    passed_7a, lookup_7a = run_pass_fail(engulf_results, ENGULF_SWEEP, 0,
                                          baseline_pf_7a, baseline_counts_7a)
    run_monotonicity(lookup_7a, ENGULF_SWEEP)

    # Summary 7a
    total_7a = len([v for v in ENGULF_SWEEP if v != 0])
    print(f"\n  7a Pass rate: {len(passed_7a)} of {total_7a} non-baseline configs")
    if passed_7a:
        best_7a_sharpe = -999
        best_7a_val = None
        for val in passed_7a:
            oos_sc = lookup_7a.get((val, "OOS"))
            if oos_sc and oos_sc["sharpe"] > best_7a_sharpe:
                best_7a_sharpe = oos_sc["sharpe"]
                best_7a_val = val
        print(f"  Best passing engulf: {best_7a_val} (OOS Sharpe {best_7a_sharpe:.3f})")
        best_engulf_pass = best_7a_val
    else:
        print("  No engulfing configs passed.")
        best_engulf_pass = None

    # =========================================================================
    # STUDY 7b: CONSECUTIVE ADVERSE BARS SWEEP
    # =========================================================================
    print(f"\n{'#'*80}")
    print(f"# STUDY 7b: CONSECUTIVE ADVERSE BARS SWEEP")
    print(f"#   Sweep: consec_adverse_bars = {CONSEC_SWEEP}")
    print(f"#   Exit if N consecutive bars close against position")
    print(f"{'#'*80}")

    consec_results = []
    baseline_counts_7b = {}
    baseline_pf_7b = {}
    best_oos_sharpe_7b = -999
    best_consec = 0

    for consec_val in CONSEC_SWEEP:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, engulf_pts=0, consec_bars=consec_val)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            if consec_val == 0:
                baseline_counts_7b[split_name] = sc["count"] if sc else 0
                baseline_pf_7b[split_name] = sc["pf"] if sc else 0.0

            consec_results.append((consec_val, split_name, dr, sc, trades))

            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe_7b:
                best_oos_sharpe_7b = sc["sharpe"]
                best_consec = consec_val

    # Print results
    print_results_table(consec_results, "Consec", CONSEC_SWEEP,
                        baseline_counts_7b, baseline_pf_7b, "CONSEC")
    print_exit_breakdown(consec_results, "Consec")

    # Counterfactual
    print_counterfactual(consec_results, CONSEC_SWEEP, 0, "CONSEC",
                         "Consecutive Adverse Bar Exits")

    # Pass/fail
    print(f"\n{'='*80}")
    print("STUDY 7b PASS/FAIL ANALYSIS")
    print(f"  Criteria: IS & OOS PF improvement >= {PF_IMPROVEMENT_THRESHOLD*100:.0f}%")
    print(f"            Per-half trade count >= {MIN_TRADE_COUNT_RATIO*100:.0f}% of baseline")
    print(f"{'='*80}")

    passed_7b, lookup_7b = run_pass_fail(consec_results, CONSEC_SWEEP, 0,
                                          baseline_pf_7b, baseline_counts_7b)
    run_monotonicity(lookup_7b, CONSEC_SWEEP)

    # Summary 7b
    total_7b = len([v for v in CONSEC_SWEEP if v != 0])
    print(f"\n  7b Pass rate: {len(passed_7b)} of {total_7b} non-baseline configs")
    if passed_7b:
        best_7b_sharpe = -999
        best_7b_val = None
        for val in passed_7b:
            oos_sc = lookup_7b.get((val, "OOS"))
            if oos_sc and oos_sc["sharpe"] > best_7b_sharpe:
                best_7b_sharpe = oos_sc["sharpe"]
                best_7b_val = val
        print(f"  Best passing consec: {best_7b_val} (OOS Sharpe {best_7b_sharpe:.3f})")
        best_consec_pass = best_7b_val
    else:
        print("  No consecutive adverse configs passed.")
        best_consec_pass = None

    # =========================================================================
    # STUDY 7c: COMBINED (best engulf + best consec)
    # =========================================================================
    print(f"\n{'#'*80}")
    print(f"# STUDY 7c: COMBINED")
    print(f"{'#'*80}")

    if best_engulf_pass is not None and best_consec_pass is not None:
        print(f"  Combining: engulf_exit_pts={best_engulf_pass} + consec_adverse_bars={best_consec_pass}")

        combined_results = []
        baseline_counts_7c = baseline_counts_7a.copy()  # Same baseline (both=0)
        baseline_pf_7c = baseline_pf_7a.copy()

        # Run baseline (0, 0) and combined
        for combo_label, engulf_v, consec_v in [
            (f"BL(0,0)", 0, 0),
            (f"E{best_engulf_pass}", best_engulf_pass, 0),
            (f"C{best_consec_pass}", 0, best_consec_pass),
            (f"E{best_engulf_pass}+C{best_consec_pass}", best_engulf_pass, best_consec_pass),
        ]:
            for split_name, _, dr in splits:
                opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
                trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                    rsi_curr, rsi_prev,
                                    engulf_pts=engulf_v, consec_bars=consec_v)
                sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                                  dollar_per_pt=MES_DOLLAR_PER_PT)
                combined_results.append((combo_label, split_name, dr, sc, trades))

        # Print combined table
        print(f"\n{'='*140}")
        print(f"{'Config':>16} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
              f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
              f"{'ENGULF':>8} {'CONSEC':>8}")
        print(f"{'-'*140}")

        for label, split_name, dr, sc, trades in combined_results:
            if sc is None:
                print(f"{label:>16} {split_name:>5}   NO TRADES")
                continue

            engulf_count = sc["exits"].get("ENGULF", 0)
            consec_count = sc["exits"].get("CONSEC", 0)

            print(f"{label:>16} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
                  f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                  f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
                  f"{engulf_count:>8} {consec_count:>8}")

        # Exit reason breakdown for combined
        print(f"\n{'='*80}")
        print("EXIT REASON BREAKDOWN (FULL data, combined)")
        print(f"{'='*80}")
        all_reasons = set()
        for label, split_name, _, sc, _ in combined_results:
            if split_name == "FULL" and sc:
                all_reasons.update(sc["exits"].keys())
        print(f"{'Config':>16} ", end="")
        for reason in sorted(all_reasons):
            print(f"{reason:>10}", end="")
        print()
        print("-" * (16 + 10 * len(all_reasons)))
        for label, split_name, _, sc, _ in combined_results:
            if split_name != "FULL" or sc is None:
                continue
            print(f"{label:>16} ", end="")
            for reason in sorted(all_reasons):
                print(f"{sc['exits'].get(reason, 0):>10}", end="")
            print()

        # Save combined if it improves over both individuals
        combo_label = f"E{best_engulf_pass}+C{best_consec_pass}"
        combo_oos_sc = None
        combo_trades_for_save = {}
        for label, split_name, dr, sc, trades in combined_results:
            if label == combo_label:
                combo_trades_for_save[split_name] = (dr, sc, trades)
                if split_name == "OOS":
                    combo_oos_sc = sc

        bl_oos_sc = None
        for label, split_name, dr, sc, trades in combined_results:
            if label == "BL(0,0)" and split_name == "OOS":
                bl_oos_sc = sc

        combo_pass = False
        if combo_oos_sc and bl_oos_sc:
            bl_oos_pf = bl_oos_sc["pf"]
            combo_oos_pf = combo_oos_sc["pf"]
            combo_dpf = ((combo_oos_pf - bl_oos_pf) / bl_oos_pf) if bl_oos_pf > 0 else 0
            combo_trd_ratio = (combo_oos_sc["count"] / bl_oos_sc["count"]) if bl_oos_sc["count"] > 0 else 0
            combo_pass = (combo_dpf >= PF_IMPROVEMENT_THRESHOLD and
                          combo_trd_ratio >= MIN_TRADE_COUNT_RATIO)
            print(f"\n  Combined OOS: PF {combo_oos_pf:.3f} (dPF {combo_dpf:+.1%}), "
                  f"trades {combo_oos_sc['count']} ({combo_trd_ratio:.0%} of baseline), "
                  f"Sharpe {combo_oos_sc['sharpe']:.3f}")
            print(f"  Combined result: {'PASS' if combo_pass else 'FAIL'}")

        if combo_pass:
            mesv2_params = {
                "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
                "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
                "sm_threshold": MESV2_SM_THRESHOLD,
                "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
                "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
                "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
                "eod_et": MESV2_EOD_ET,
                "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
                "engulf_exit_pts": best_engulf_pass,
                "consec_adverse_bars": best_consec_pass,
            }

            print(f"\nSaving trade CSVs for combined E{best_engulf_pass}+C{best_consec_pass}...")
            for split_name, (dr, sc, trades) in combo_trades_for_save.items():
                if trades:
                    save_backtest(
                        trades, strategy="MES_V2_BAR_PATTERN", params=mesv2_params,
                        data_range=dr, split=split_name,
                        dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                        script_name=SCRIPT_NAME,
                        notes=f"engulf={best_engulf_pass}_consec={best_consec_pass}",
                    )
    else:
        skip_reasons = []
        if best_engulf_pass is None:
            skip_reasons.append("no engulfing config passed 7a")
        if best_consec_pass is None:
            skip_reasons.append("no consecutive config passed 7b")
        print(f"  SKIPPED: {', '.join(skip_reasons)}")
        print("  Combined study requires both 7a and 7b to have at least one passing config.")

    # =========================================================================
    # OVERALL SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  Study 7a (Engulfing Bar):")
    print(f"    Sweep: {ENGULF_SWEEP}")
    print(f"    Pass rate: {len(passed_7a)} of {total_7a}")
    print(f"    Best OOS Sharpe: engulf={best_engulf} (Sharpe {best_oos_sharpe_7a:.3f})")
    if passed_7a:
        print(f"    Passing configs: {passed_7a}")
        print(f"    Best passing: engulf={best_engulf_pass}")
    else:
        print(f"    No configs passed.")

    print(f"\n  Study 7b (Consecutive Adverse):")
    print(f"    Sweep: {CONSEC_SWEEP}")
    print(f"    Pass rate: {len(passed_7b)} of {total_7b}")
    print(f"    Best OOS Sharpe: consec={best_consec} (Sharpe {best_oos_sharpe_7b:.3f})")
    if passed_7b:
        print(f"    Passing configs: {passed_7b}")
        print(f"    Best passing: consec={best_consec_pass}")
    else:
        print(f"    No configs passed.")

    if best_engulf_pass is not None and best_consec_pass is not None:
        combo_status = "PASS" if combo_pass else "FAIL"
        print(f"\n  Study 7c (Combined E{best_engulf_pass}+C{best_consec_pass}): {combo_status}")
    else:
        print(f"\n  Study 7c: SKIPPED (prerequisites not met)")

    print("\nDone.")


if __name__ == "__main__":
    main()
