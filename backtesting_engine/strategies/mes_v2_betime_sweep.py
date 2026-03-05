"""
MES v2 Breakeven-After-N-Bars Sweep
====================================
Sweeps breakeven_after_bars values as a proper pre-filter backtest.
Trades that stall for N+ bars without hitting TP=20 or SL=35 are closed
at next open (exit_reason=BE_TIME). This is a pre-filter test: closing
early frees the cooldown clock and may enable re-entries.

Usage:
    python3 mes_v2_betime_sweep.py
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
)

from results.save_results import save_backtest

SWEEP_VALUES = [0, 30, 45, 60, 75, 90, 105, 120, 150, 200, 275]
SCRIPT_NAME = "mes_v2_betime_sweep.py"


def run_single(opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, breakeven_n):
    """Run MES v2 backtest with a specific breakeven_after_bars value."""
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        breakeven_after_bars=breakeven_n,
    )
    compute_mfe_mae(trades, highs, lows)
    return trades


def main():
    print("=" * 80)
    print("MES v2 BREAKEVEN-AFTER-N-BARS SWEEP (pre-filter)")
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

    # --- Get baseline trade count for re-entry calculation ---
    baseline_counts = {}

    # --- Sweep ---
    results = []  # (N, split, sc, trades)
    best_oos_sharpe = -999
    best_n = 0

    for n_val in SWEEP_VALUES:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, n_val)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            # Track baseline trade counts
            if n_val == 0:
                baseline_counts[split_name] = sc["count"] if sc else 0

            results.append((n_val, split_name, dr, sc, trades))

            # Track best OOS Sharpe
            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = sc["sharpe"]
                best_n = n_val

    # --- Print summary table ---
    print(f"\n{'='*120}")
    print(f"{'N':>5} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'BE_TIME':>8} {'Re-entry':>9}")
    print(f"{'-'*120}")

    for n_val, split_name, dr, sc, trades in results:
        if sc is None:
            print(f"{n_val:>5} {split_name:>5}   NO TRADES")
            continue

        be_time_count = sc["exits"].get("BE_TIME", 0)
        baseline = baseline_counts.get(split_name, 0)
        re_entry = sc["count"] - baseline

        print(f"{n_val:>5} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{be_time_count:>8} {re_entry:>+9}")

    # --- Exit reason breakdown per N ---
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")
    print(f"{'N':>5} ", end="")
    all_reasons = set()
    for n_val, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())
    for reason in sorted(all_reasons):
        print(f"{reason:>10}", end="")
    print()
    print("-" * (5 + 10 * len(all_reasons)))

    for n_val, split_name, _, sc, _ in results:
        if split_name != "FULL":
            continue
        if sc is None:
            continue
        print(f"{n_val:>5} ", end="")
        for reason in sorted(all_reasons):
            print(f"{sc['exits'].get(reason, 0):>10}", end="")
        print()

    # --- Save best result ---
    print(f"\n{'='*80}")
    print(f"BEST OOS SHARPE: N={best_n} (Sharpe {best_oos_sharpe:.3f})")
    print(f"{'='*80}")

    if best_n > 0:
        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": best_n,
        }

        print(f"\nSaving trade CSVs for best N={best_n}...")
        for n_val, split_name, dr, sc, trades in results:
            if n_val == best_n and trades:
                save_backtest(
                    trades, strategy="MES_V2_BETIME", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"breakeven_after_bars={best_n}",
                )
    else:
        print("Baseline (N=0) is best — no BE_TIME rule needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
