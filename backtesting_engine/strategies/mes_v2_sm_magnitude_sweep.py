"""
MES v2 SM Threshold Magnitude Sweep
====================================
Sweeps sm_threshold values for the MES v2 strategy as a proper pre-filter
backtest. Higher sm_threshold requires stronger Smart Money signal before
allowing entry, filtering out weak/ambiguous signals.

Baseline is sm_threshold=0.0 (current production).

Usage:
    python3 mes_v2_sm_magnitude_sweep.py
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

SWEEP_VALUES = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
BASELINE_VALUE = 0.0
SCRIPT_NAME = "mes_v2_sm_magnitude_sweep.py"

# Pass/fail criteria
PF_IMPROVEMENT_THRESHOLD = 0.05   # 5% PF improvement over baseline = strong pass
MIN_TRADE_COUNT_RATIO = 0.70      # Must retain at least 70% of baseline trades


def run_single(opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, sm_thresh):
    """Run MES v2 backtest with a specific sm_threshold value."""
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=sm_thresh, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
    )
    compute_mfe_mae(trades, highs, lows)
    return trades


def main():
    print("=" * 80)
    print("MES v2 SM THRESHOLD MAGNITUDE SWEEP (pre-filter)")
    print(f"  Baseline: sm_threshold={BASELINE_VALUE}")
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

    # --- Get baseline trade count and PF for pass/fail ---
    baseline_counts = {}
    baseline_pf = {}

    # --- Sweep ---
    results = []  # (sm_thresh, split, dr, sc, trades)
    best_oos_sharpe = -999
    best_thresh = BASELINE_VALUE

    for sm_val in SWEEP_VALUES:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, sm_val)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            # Track baseline stats
            if sm_val == BASELINE_VALUE:
                baseline_counts[split_name] = sc["count"] if sc else 0
                baseline_pf[split_name] = sc["pf"] if sc else 0.0

            results.append((sm_val, split_name, dr, sc, trades))

            # Track best OOS Sharpe
            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = sc["sharpe"]
                best_thresh = sm_val

    # --- Print summary table ---
    print(f"\n{'='*130}")
    print(f"{'SM_T':>6} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'dPF%':>7} {'dTrades%':>9}")
    print(f"{'-'*130}")

    for sm_val, split_name, dr, sc, trades in results:
        if sc is None:
            print(f"{sm_val:>6.2f} {split_name:>5}   NO TRADES")
            continue

        # Delta vs baseline
        bl_pf = baseline_pf.get(split_name, 0.0)
        bl_count = baseline_counts.get(split_name, 0)
        dpf_pct = ((sc["pf"] - bl_pf) / bl_pf * 100) if bl_pf > 0 else 0.0
        dtrades_pct = ((sc["count"] - bl_count) / bl_count * 100) if bl_count > 0 else 0.0

        print(f"{sm_val:>6.2f} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{dpf_pct:>+6.1f}% {dtrades_pct:>+8.1f}%")

    # --- Exit reason breakdown per threshold ---
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")
    print(f"{'SM_T':>6} ", end="")
    all_reasons = set()
    for sm_val, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())
    for reason in sorted(all_reasons):
        print(f"{reason:>10}", end="")
    print()
    print("-" * (6 + 10 * len(all_reasons)))

    for sm_val, split_name, _, sc, _ in results:
        if split_name != "FULL":
            continue
        if sc is None:
            continue
        print(f"{sm_val:>6.2f} ", end="")
        for reason in sorted(all_reasons):
            print(f"{sc['exits'].get(reason, 0):>10}", end="")
        print()

    # --- Pass/Fail Analysis ---
    print(f"\n{'='*80}")
    print("PASS/FAIL ANALYSIS")
    print(f"  Criteria: IS & OOS PF improvement >= {PF_IMPROVEMENT_THRESHOLD*100:.0f}%")
    print(f"            Per-half trade count >= {MIN_TRADE_COUNT_RATIO*100:.0f}% of baseline")
    print(f"{'='*80}")

    # Build lookup: (sm_val, split_name) -> sc
    result_lookup = {}
    for sm_val, split_name, dr, sc, trades in results:
        result_lookup[(sm_val, split_name)] = sc

    passed_configs = []
    print(f"\n{'SM_T':>6} {'IS_PF':>7} {'OOS_PF':>7} {'IS_dPF%':>8} {'OOS_dPF%':>9} "
          f"{'IS_Trd':>7} {'OOS_Trd':>8} {'IS_Trd%':>8} {'OOS_Trd%':>9} {'Result':>10}")
    print("-" * 95)

    for sm_val in SWEEP_VALUES:
        is_sc = result_lookup.get((sm_val, "IS"))
        oos_sc = result_lookup.get((sm_val, "OOS"))

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
        if sm_val == BASELINE_VALUE:
            status = "BASELINE"
        if passed:
            passed_configs.append(sm_val)

        print(f"{sm_val:>6.2f} {is_pf:>7.3f} {oos_pf:>7.3f} "
              f"{is_dpf:>+7.1%} {oos_dpf:>+8.1%} "
              f"{is_count:>7} {oos_count:>8} "
              f"{is_trd_ratio:>7.0%} {oos_trd_ratio:>8.0%} "
              f"{status:>10}")

    # --- Monotonicity check ---
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK (are results smooth across ordered thresholds?)")
    print(f"{'='*80}")

    for split_check in ["IS", "OOS"]:
        pf_series = []
        sharpe_series = []
        for sm_val in SWEEP_VALUES:
            sc = result_lookup.get((sm_val, split_check))
            pf_series.append(sc["pf"] if sc else 0.0)
            sharpe_series.append(sc["sharpe"] if sc else 0.0)

        # Check monotonicity: count direction changes
        pf_diffs = [pf_series[i+1] - pf_series[i] for i in range(len(pf_series)-1)]
        pf_sign_changes = sum(1 for i in range(len(pf_diffs)-1)
                              if pf_diffs[i] * pf_diffs[i+1] < 0)

        sharpe_diffs = [sharpe_series[i+1] - sharpe_series[i] for i in range(len(sharpe_series)-1)]
        sharpe_sign_changes = sum(1 for i in range(len(sharpe_diffs)-1)
                                  if sharpe_diffs[i] * sharpe_diffs[i+1] < 0)

        max_reversals = len(SWEEP_VALUES) - 2  # max possible sign changes
        pf_smooth = "SMOOTH" if pf_sign_changes <= 1 else f"NOISY ({pf_sign_changes} reversals)"
        sharpe_smooth = "SMOOTH" if sharpe_sign_changes <= 1 else f"NOISY ({sharpe_sign_changes} reversals)"

        print(f"\n  {split_check}:")
        print(f"    PF trend:     {' -> '.join(f'{p:.3f}' for p in pf_series)}")
        print(f"    PF smoothness: {pf_smooth}")
        print(f"    Sharpe trend: {' -> '.join(f'{s:.3f}' for s in sharpe_series)}")
        print(f"    Sharpe smoothness: {sharpe_smooth}")

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total_non_baseline = len([v for v in SWEEP_VALUES if v != BASELINE_VALUE])
    pass_count = len(passed_configs)
    print(f"\n  Pass rate: {pass_count} of {total_non_baseline} non-baseline configs passed")

    if passed_configs:
        # Find best among passed: highest OOS Sharpe
        best_passed = None
        best_passed_sharpe = -999
        for sm_val in passed_configs:
            oos_sc = result_lookup.get((sm_val, "OOS"))
            if oos_sc and oos_sc["sharpe"] > best_passed_sharpe:
                best_passed_sharpe = oos_sc["sharpe"]
                best_passed = sm_val

        print(f"  Passed configs: {passed_configs}")
        print(f"  Best passing config: sm_threshold={best_passed} (OOS Sharpe {best_passed_sharpe:.3f})")

        # Print full stats for best passing config
        print(f"\n  Best passing config detail:")
        for split_check in ["IS", "OOS", "FULL"]:
            sc = result_lookup.get((best_passed, split_check))
            if sc:
                print(f"    {split_check:>4}: {sc['count']} trades, WR {sc['win_rate']:.1f}%, "
                      f"PF {sc['pf']:.3f}, P&L {sc['net_dollar']:+.2f}, "
                      f"Sharpe {sc['sharpe']:.3f}, MaxDD {sc['max_dd_dollar']:.2f}")
    else:
        print("  No configs passed all criteria.")

    print(f"\n  Best OOS Sharpe overall: sm_threshold={best_thresh} (Sharpe {best_oos_sharpe:.3f})")
    if best_thresh == BASELINE_VALUE:
        print("  -> Baseline is best. No SM threshold filter needed.")
    elif best_thresh not in passed_configs:
        print(f"  -> WARNING: Best Sharpe config did NOT pass criteria (trade count or PF delta).")

    # --- Save best result ---
    if best_thresh != BASELINE_VALUE and passed_configs:
        save_thresh = passed_configs[-1] if best_thresh not in passed_configs else best_thresh
        # Use the best passing config
        best_for_save = None
        best_for_save_sharpe = -999
        for sm_val in passed_configs:
            oos_sc = result_lookup.get((sm_val, "OOS"))
            if oos_sc and oos_sc["sharpe"] > best_for_save_sharpe:
                best_for_save_sharpe = oos_sc["sharpe"]
                best_for_save = sm_val

        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": best_for_save,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
        }

        print(f"\nSaving trade CSVs for best passing config sm_threshold={best_for_save}...")
        for sm_val, split_name, dr, sc, trades in results:
            if sm_val == best_for_save and trades:
                save_backtest(
                    trades, strategy="MES_V2_SM_THRESH", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"sm_threshold={best_for_save}",
                )
    else:
        print("\nBaseline is best or no configs passed -- nothing to save.")

    print("\nDone.")


if __name__ == "__main__":
    main()
