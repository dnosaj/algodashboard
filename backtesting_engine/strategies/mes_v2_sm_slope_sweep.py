"""
MES v2 SM Slope Gate Sweep
==========================
Sweeps SM slope gate thresholds for MES v2. The gate requires that SM is
moving in the trade direction at entry: for longs SM slope must be positive
(> threshold), for shorts SM slope must be negative (< -threshold).

This is a pre-filter: blocking entries changes cooldowns and trade sequences.
breakeven_after_bars=75 is held constant throughout.

Usage:
    python3 mes_v2_sm_slope_sweep.py
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

SWEEP_VALUES = [None, 0.0, 0.001, 0.002, 0.005, 0.01]
SCRIPT_NAME = "mes_v2_sm_slope_sweep.py"


def build_sm_slope_gate(sm, slope_threshold, sm_threshold=0.0):
    """Gate: require SM slope away from zero at entry.
    For longs (sm > sm_threshold): slope must be > slope_threshold
    For shorts (sm < -sm_threshold): slope must be < -slope_threshold
    Bars not near a potential entry are left True (no effect).
    """
    n = len(sm)
    gate = np.ones(n, dtype=bool)
    for j in range(1, n):
        slope = sm[j] - sm[j - 1]
        if sm[j] > sm_threshold:      # potential long
            gate[j] = slope > slope_threshold
        elif sm[j] < -sm_threshold:   # potential short
            gate[j] = slope < -slope_threshold
        # else: not near entry condition, gate stays True
    return gate


def run_single(opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, entry_gate=None):
    """Run MES v2 backtest with a specific entry_gate."""
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        entry_gate=entry_gate,
    )
    compute_mfe_mae(trades, highs, lows)
    return trades


def main():
    print("=" * 80)
    print("MES v2 SM SLOPE GATE SWEEP (pre-filter)")
    print(f"  breakeven_after_bars={MESV2_BREAKEVEN_BARS} (constant)")
    print("=" * 80)

    # --- Load MES data ---
    print("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    print(f"  {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # Compute SM on full data (before split -- avoids EMA warm-up artifacts)
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

    # --- Build gates on FULL data, then slice for IS/OOS ---
    # Gate is built on full SM array so EMA warm-up doesn't create artifacts
    # at the OOS boundary. Indices correspond 1:1 with df_mes rows.
    full_sm = df_mes["SM_Net"].values
    is_len = len(mes_is)

    gates_by_val = {}  # slope_val -> {"FULL": gate, "IS": gate, "OOS": gate}
    for slope_val in SWEEP_VALUES:
        if slope_val is None:
            gates_by_val[slope_val] = {"FULL": None, "IS": None, "OOS": None}
        else:
            full_gate = build_sm_slope_gate(full_sm, slope_val, MESV2_SM_THRESHOLD)
            gates_by_val[slope_val] = {
                "FULL": full_gate,
                "IS": full_gate[:is_len],
                "OOS": full_gate[is_len:],
            }

    # --- Compute gate pass rates ---
    print(f"\n{'='*60}")
    print("GATE PASS RATES (% of bars where gate=True)")
    print(f"{'='*60}")
    print(f"{'Threshold':>12} {'FULL':>8} {'IS':>8} {'OOS':>8}")
    print(f"{'-'*40}")
    for slope_val in SWEEP_VALUES:
        label = "None" if slope_val is None else f"{slope_val:.4f}"
        if slope_val is None:
            print(f"{label:>12} {'100.0%':>8} {'100.0%':>8} {'100.0%':>8}")
        else:
            g = gates_by_val[slope_val]
            full_pct = g["FULL"].mean() * 100
            is_pct = g["IS"].mean() * 100
            oos_pct = g["OOS"].mean() * 100
            print(f"{label:>12} {full_pct:>7.1f}% {is_pct:>7.1f}% {oos_pct:>7.1f}%")

    # --- Get baseline trade count for re-entry calculation ---
    baseline_counts = {}

    # --- Sweep ---
    results = []  # (slope_val, split, dr, sc, trades)
    best_oos_sharpe = -999
    best_val = None

    for slope_val in SWEEP_VALUES:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            gate = gates_by_val[slope_val][split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, entry_gate=gate)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            # Track baseline trade counts (None = no gate)
            if slope_val is None:
                baseline_counts[split_name] = sc["count"] if sc else 0

            results.append((slope_val, split_name, dr, sc, trades))

            # Track best OOS Sharpe
            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = sc["sharpe"]
                best_val = slope_val

    # --- Print summary table ---
    print(f"\n{'='*130}")
    print(f"{'Slope':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'Blocked':>8} {'Pass/Fail':>10}")
    print(f"{'-'*130}")

    # Collect OOS PF values for monotonicity check
    oos_pf_by_val = {}

    for slope_val, split_name, dr, sc, trades in results:
        label = "None" if slope_val is None else f"{slope_val:.4f}"

        if sc is None:
            print(f"{label:>8} {split_name:>5}   NO TRADES")
            if split_name == "OOS":
                oos_pf_by_val[slope_val] = 0.0
            continue

        baseline = baseline_counts.get(split_name, 0)
        blocked = baseline - sc["count"]

        # Pass/fail: OOS PF > 1.0 and OOS Sharpe > 0
        pf_label = ""
        if split_name == "OOS":
            oos_pf_by_val[slope_val] = sc["pf"]
            passed = sc["pf"] > 1.0 and sc["sharpe"] > 0
            pf_label = "PASS" if passed else "FAIL"

        print(f"{label:>8} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{blocked:>8} {pf_label:>10}")

    # --- Monotonicity check ---
    # As slope threshold increases, we expect fewer entries and (hopefully)
    # higher PF. Check if OOS PF is monotonically non-decreasing across
    # non-None thresholds.
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK (OOS PF vs slope threshold)")
    print(f"{'='*80}")

    non_none_vals = [v for v in SWEEP_VALUES if v is not None]
    oos_pfs = [oos_pf_by_val.get(v, 0.0) for v in non_none_vals]

    monotonic = True
    for i in range(1, len(oos_pfs)):
        if oos_pfs[i] < oos_pfs[i - 1] - 0.01:  # allow 0.01 tolerance
            monotonic = False
            break

    for i, v in enumerate(non_none_vals):
        arrow = ""
        if i > 0:
            diff = oos_pfs[i] - oos_pfs[i - 1]
            arrow = f"  ({'+' if diff >= 0 else ''}{diff:.3f})"
        print(f"  slope={v:.4f}  OOS PF={oos_pfs[i]:.3f}{arrow}")

    if monotonic:
        print("  -> Monotonically non-decreasing (good: higher threshold = better PF)")
    else:
        print("  -> NOT monotonic (PF does not consistently improve with threshold)")

    # --- Exit reason breakdown per slope value ---
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")
    all_reasons = set()
    for slope_val, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())

    print(f"{'Slope':>8} ", end="")
    for reason in sorted(all_reasons):
        print(f"{reason:>10}", end="")
    print()
    print("-" * (8 + 10 * len(all_reasons)))

    for slope_val, split_name, _, sc, _ in results:
        if split_name != "FULL":
            continue
        label = "None" if slope_val is None else f"{slope_val:.4f}"
        if sc is None:
            continue
        print(f"{label:>8} ", end="")
        for reason in sorted(all_reasons):
            print(f"{sc['exits'].get(reason, 0):>10}", end="")
        print()

    # --- Summary verdict ---
    print(f"\n{'='*80}")
    best_label = "None" if best_val is None else f"{best_val:.4f}"
    print(f"BEST OOS SHARPE: slope={best_label} (Sharpe {best_oos_sharpe:.3f})")

    # Check if any non-None threshold beats baseline
    baseline_oos = oos_pf_by_val.get(None, 0.0)
    improving_vals = [v for v in non_none_vals if oos_pf_by_val.get(v, 0.0) > baseline_oos]
    if improving_vals:
        print(f"Thresholds that improve OOS PF over baseline: "
              f"{[f'{v:.4f}' for v in improving_vals]}")
    else:
        print("No slope threshold improves OOS PF over baseline (None).")
    print(f"{'='*80}")

    # --- Save best result ---
    if best_val is not None:
        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
            "sm_slope_threshold": best_val,
        }

        print(f"\nSaving trade CSVs for best slope={best_label}...")
        for slope_val, split_name, dr, sc, trades in results:
            if slope_val == best_val and trades:
                save_backtest(
                    trades, strategy="MES_V2_SM_SLOPE", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"sm_slope_threshold={best_val}",
                )
    else:
        print("Baseline (None = no gate) is best -- no SM slope gate needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
