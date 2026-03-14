"""
MES v2 Quick Stop Sweep
========================
Sweeps quick_stop_pts (Q) and quick_stop_bars (K) parameters for MES v2.
Quick stop exits a trade early if it moves adversely by Q pts within the
first K bars after entry.

Sweep grid: Q=[5,8,10,15] x K=[1,2,3,5] = 16 combos + baseline (Q=0,K=0) = 17 configs.
breakeven_after_bars=75 held constant throughout.

Output:
  1. Standard grid table (Q rows x K columns) for FULL/IS/OOS PF
  2. Exit reason breakdown (FULL data)
  3. Counterfactual analysis: for each QS exit, did baseline run end as TP or SL?
     - Save rate: % of QS exits that would have been SL without QS
     - False positive rate: % of QS exits that would have been TP/BE_TIME/EOD without QS

Usage:
    python3 mes_v2_quick_stop_sweep.py
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

Q_VALUES = [5, 8, 10, 15]
K_VALUES = [1, 2, 3, 5]
SCRIPT_NAME = "mes_v2_quick_stop_sweep.py"


def run_single(opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, qs_pts=0, qs_bars=0):
    """Run MES v2 backtest with specific quick stop parameters."""
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        quick_stop_pts=qs_pts,
        quick_stop_bars=qs_bars,
    )
    compute_mfe_mae(trades, highs, lows)
    return trades


def build_entry_time_map(trades):
    """Build a dict mapping entry_time -> trade for counterfactual lookups."""
    return {t["entry_time"]: t for t in trades}


def counterfactual_analysis(qs_trades, baseline_map):
    """Compare QS exits against baseline outcomes.

    Returns:
        dict with keys: qs_count, saves, save_rate, false_positives, fp_rate,
                        baseline_breakdown (dict of baseline result -> count)
    """
    qs_exits = [t for t in qs_trades if t["result"] == "QS"]
    qs_count = len(qs_exits)

    if qs_count == 0:
        return {
            "qs_count": 0, "saves": 0, "save_rate": 0.0,
            "false_positives": 0, "fp_rate": 0.0,
            "baseline_breakdown": {},
        }

    saves = 0          # QS exits that would have been SL in baseline
    false_positives = 0  # QS exits that would have been TP/BE_TIME/EOD in baseline
    baseline_breakdown = {}

    for t in qs_exits:
        baseline_trade = baseline_map.get(t["entry_time"])
        if baseline_trade is None:
            # Trade exists in QS config but not baseline (different re-entry pattern)
            baseline_result = "NO_MATCH"
        else:
            baseline_result = baseline_trade["result"]

        baseline_breakdown[baseline_result] = baseline_breakdown.get(baseline_result, 0) + 1

        if baseline_result == "SL":
            saves += 1
        elif baseline_result in ("TP", "BE_TIME", "EOD"):
            false_positives += 1
        # NO_MATCH trades are neither saves nor false positives

    return {
        "qs_count": qs_count,
        "saves": saves,
        "save_rate": saves / qs_count * 100 if qs_count > 0 else 0.0,
        "false_positives": false_positives,
        "fp_rate": false_positives / qs_count * 100 if qs_count > 0 else 0.0,
        "baseline_breakdown": baseline_breakdown,
    }


def main():
    print("=" * 80)
    print("MES v2 QUICK STOP SWEEP (pre-filter)")
    print(f"  Q (quick_stop_pts) = {Q_VALUES}")
    print(f"  K (quick_stop_bars) = {K_VALUES}")
    print(f"  breakeven_after_bars = {MESV2_BREAKEVEN_BARS} (constant)")
    print("=" * 80)

    # --- Load MES data ---
    print("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    print(f"  {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # Compute SM on full data (before split — matches existing pipeline)
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

    # --- Run baseline first (Q=0, K=0) ---
    print("\nRunning baseline (QS disabled)...")
    baseline_results = {}   # split_name -> (sc, trades)
    baseline_maps = {}      # split_name -> entry_time -> trade dict

    for split_name, _, dr in splits:
        opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
        trades = run_single(opens, highs, lows, closes, sm_arr, times,
                            rsi_curr, rsi_prev, qs_pts=0, qs_bars=0)
        sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                          dollar_per_pt=MES_DOLLAR_PER_PT)
        baseline_results[split_name] = (sc, trades)
        baseline_maps[split_name] = build_entry_time_map(trades)
        if sc:
            print(f"  {split_name}: {sc['count']} trades, PF {sc['pf']}, "
                  f"Sharpe {sc['sharpe']}, Net ${sc['net_dollar']:+.2f}")

    # --- Sweep all Q x K combos ---
    print("\nRunning sweep...")
    # results[(q, k)] = {split_name: (sc, trades)}
    sweep_results = {}

    for q in Q_VALUES:
        for k in K_VALUES:
            sweep_results[(q, k)] = {}
            for split_name, _, dr in splits:
                opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
                trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                    rsi_curr, rsi_prev, qs_pts=q, qs_bars=k)
                sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                                  dollar_per_pt=MES_DOLLAR_PER_PT)
                sweep_results[(q, k)][split_name] = (sc, trades)

    # =====================================================================
    # 1. GRID TABLES: PF for FULL / IS / OOS
    # =====================================================================
    for split_name in ["FULL", "IS", "OOS"]:
        print(f"\n{'='*60}")
        print(f"PROFIT FACTOR GRID — {split_name}")
        print(f"{'='*60}")

        baseline_sc = baseline_results[split_name][0]
        baseline_pf = baseline_sc["pf"] if baseline_sc else 0.0
        print(f"Baseline (no QS): PF {baseline_pf:.3f}")
        print()

        # Header
        header = f"{'Q \\\\ K':>10}"
        for k in K_VALUES:
            header += f"  K={k:>2}"
        print(header)
        print("-" * (10 + 6 * len(K_VALUES)))

        for q in Q_VALUES:
            row = f"Q={q:>2}      "
            for k in K_VALUES:
                sc = sweep_results[(q, k)][split_name][0]
                pf = sc["pf"] if sc else 0.0
                row += f" {pf:5.3f}"
            print(row)

    # =====================================================================
    # 2. DETAILED RESULTS TABLE
    # =====================================================================
    print(f"\n{'='*130}")
    print(f"{'Q':>3} {'K':>3} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'QS':>5} {'SL':>5} {'TP':>5} {'BE_TIME':>8} {'EOD':>5}")
    print(f"{'-'*130}")

    # Print baseline first
    for split_name in ["FULL", "IS", "OOS"]:
        sc = baseline_results[split_name][0]
        if sc is None:
            print(f"{'0':>3} {'0':>3} {split_name:>5}   NO TRADES")
            continue
        print(f"{'0':>3} {'0':>3} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{sc['exits'].get('QS', 0):>5} {sc['exits'].get('SL', 0):>5} "
              f"{sc['exits'].get('TP', 0):>5} {sc['exits'].get('BE_TIME', 0):>8} "
              f"{sc['exits'].get('EOD', 0):>5}")

    print(f"{'-'*130}")

    for q in Q_VALUES:
        for k in K_VALUES:
            for split_name in ["FULL", "IS", "OOS"]:
                sc = sweep_results[(q, k)][split_name][0]
                if sc is None:
                    print(f"{q:>3} {k:>3} {split_name:>5}   NO TRADES")
                    continue
                print(f"{q:>3} {k:>3} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
                      f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                      f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
                      f"{sc['exits'].get('QS', 0):>5} {sc['exits'].get('SL', 0):>5} "
                      f"{sc['exits'].get('TP', 0):>5} {sc['exits'].get('BE_TIME', 0):>8} "
                      f"{sc['exits'].get('EOD', 0):>5}")
            if q != Q_VALUES[-1] or k != K_VALUES[-1]:
                print()

    # =====================================================================
    # 3. EXIT REASON BREAKDOWN (FULL data)
    # =====================================================================
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")

    # Collect all exit reasons across all configs
    all_reasons = set()
    for split_name in ["FULL"]:
        sc = baseline_results[split_name][0]
        if sc:
            all_reasons.update(sc["exits"].keys())
    for q in Q_VALUES:
        for k in K_VALUES:
            sc = sweep_results[(q, k)]["FULL"][0]
            if sc:
                all_reasons.update(sc["exits"].keys())

    sorted_reasons = sorted(all_reasons)
    header = f"{'Config':>10} "
    for reason in sorted_reasons:
        header += f"{reason:>10}"
    print(header)
    print("-" * (10 + 10 * len(sorted_reasons) + 1))

    # Baseline row
    sc = baseline_results["FULL"][0]
    if sc:
        row = f"{'baseline':>10} "
        for reason in sorted_reasons:
            row += f"{sc['exits'].get(reason, 0):>10}"
        print(row)

    for q in Q_VALUES:
        for k in K_VALUES:
            sc = sweep_results[(q, k)]["FULL"][0]
            if sc is None:
                continue
            label = f"Q{q}K{k}"
            row = f"{label:>10} "
            for reason in sorted_reasons:
                row += f"{sc['exits'].get(reason, 0):>10}"
            print(row)

    # =====================================================================
    # 4. COUNTERFACTUAL ANALYSIS (FULL data)
    # =====================================================================
    print(f"\n{'='*80}")
    print("COUNTERFACTUAL ANALYSIS (FULL data)")
    print("For each QS exit: what would baseline have done with that trade?")
    print(f"{'='*80}")
    print(f"{'Config':>10} {'QS exits':>10} {'saves':>7} {'save%':>7} "
          f"{'false_pos':>10} {'fp%':>7}   baseline breakdown")
    print("-" * 100)

    cf_results = {}  # (q, k) -> counterfactual dict
    for q in Q_VALUES:
        for k in K_VALUES:
            trades = sweep_results[(q, k)]["FULL"][1]
            cf = counterfactual_analysis(trades, baseline_maps["FULL"])
            cf_results[(q, k)] = cf

            breakdown_str = ", ".join(
                f"{r}={c}" for r, c in sorted(cf["baseline_breakdown"].items())
            )
            print(f"Q={q:>2},K={k:<2} {cf['qs_count']:>10} {cf['saves']:>7} "
                  f"{cf['save_rate']:>6.1f}% {cf['false_positives']:>10} "
                  f"{cf['fp_rate']:>6.1f}%   {breakdown_str}")

    # =====================================================================
    # 5. PASS/FAIL CRITERIA
    # =====================================================================
    print(f"\n{'='*80}")
    print("PASS / FAIL CRITERIA")
    print(f"{'='*80}")

    baseline_full_sc = baseline_results["FULL"][0]
    baseline_full_pf = baseline_full_sc["pf"] if baseline_full_sc else 0.0
    baseline_full_sharpe = baseline_full_sc["sharpe"] if baseline_full_sc else 0.0
    baseline_oos_sc = baseline_results["OOS"][0]
    baseline_oos_pf = baseline_oos_sc["pf"] if baseline_oos_sc else 0.0

    pass_count = 0
    total_count = len(Q_VALUES) * len(K_VALUES)

    print(f"\nBaseline: FULL PF={baseline_full_pf:.3f}, OOS PF={baseline_oos_pf:.3f}, "
          f"FULL Sharpe={baseline_full_sharpe:.3f}")
    print()
    print(f"{'Config':>10} {'FULL PF':>8} {'OOS PF':>8} {'Sharpe':>8} "
          f"{'Save%':>7} {'FP%':>7} {'Result':>8}")
    print("-" * 65)

    for q in Q_VALUES:
        for k in K_VALUES:
            full_sc = sweep_results[(q, k)]["FULL"][0]
            oos_sc = sweep_results[(q, k)]["OOS"][0]
            cf = cf_results[(q, k)]

            full_pf = full_sc["pf"] if full_sc else 0.0
            full_sharpe = full_sc["sharpe"] if full_sc else 0.0
            oos_pf = oos_sc["pf"] if oos_sc else 0.0

            # Pass criteria:
            #   1. FULL PF >= baseline FULL PF
            #   2. OOS PF >= baseline OOS PF
            #   3. Save rate > false positive rate (net positive)
            pf_pass = full_pf >= baseline_full_pf
            oos_pass = oos_pf >= baseline_oos_pf
            cf_pass = cf["save_rate"] > cf["fp_rate"] if cf["qs_count"] > 0 else False
            passed = pf_pass and oos_pass and cf_pass

            if passed:
                pass_count += 1

            label = f"Q={q:>2},K={k:<2}"
            result = "PASS" if passed else "FAIL"
            print(f"{label:>10} {full_pf:>8.3f} {oos_pf:>8.3f} {full_sharpe:>8.3f} "
                  f"{cf['save_rate']:>6.1f}% {cf['fp_rate']:>6.1f}% {result:>8}")

    print(f"\nPass rate: {pass_count}/{total_count} "
          f"({pass_count/total_count*100:.0f}%)")

    # =====================================================================
    # 6. MONOTONICITY CHECK
    # =====================================================================
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK")
    print("Checks whether increasing Q (for fixed K) monotonically changes QS count")
    print(f"{'='*80}")

    mono_pass = 0
    mono_total = len(K_VALUES)

    for k in K_VALUES:
        qs_counts = []
        for q in Q_VALUES:
            cf = cf_results[(q, k)]
            qs_counts.append(cf["qs_count"])

        # Expect: more QS exits as Q decreases (tighter stop),
        # i.e., QS count should be non-increasing as Q increases
        monotonic = all(qs_counts[i] >= qs_counts[i + 1] for i in range(len(qs_counts) - 1))
        status = "MONO" if monotonic else "NON-MONO"
        if monotonic:
            mono_pass += 1

        counts_str = " -> ".join(f"Q={q}:{c}" for q, c in zip(Q_VALUES, qs_counts))
        print(f"  K={k}: {counts_str}  [{status}]")

    print(f"\nMonotonicity pass: {mono_pass}/{mono_total}")

    # =====================================================================
    # 7. SAVE BEST RESULT
    # =====================================================================
    print(f"\n{'='*80}")
    print("BEST CONFIG SELECTION")
    print(f"{'='*80}")

    best_oos_sharpe = baseline_results["OOS"][0]["sharpe"] if baseline_results["OOS"][0] else -999
    best_config = (0, 0)  # baseline

    for q in Q_VALUES:
        for k in K_VALUES:
            oos_sc = sweep_results[(q, k)]["OOS"][0]
            if oos_sc and oos_sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = oos_sc["sharpe"]
                best_config = (q, k)

    best_q, best_k = best_config
    print(f"Best OOS Sharpe: Q={best_q}, K={best_k} (Sharpe {best_oos_sharpe:.3f})")

    if best_q > 0 and best_k > 0:
        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
            "quick_stop_pts": best_q,
            "quick_stop_bars": best_k,
        }

        print(f"\nSaving trade CSVs for best Q={best_q}, K={best_k}...")
        for split_name, _, dr in splits:
            sc, trades = sweep_results[(best_q, best_k)][split_name]
            if trades:
                save_backtest(
                    trades, strategy="MES_V2_QS", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"quick_stop_pts={best_q}, quick_stop_bars={best_k}",
                )
    else:
        print("Baseline (no QS) is best — no quick stop rule needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
