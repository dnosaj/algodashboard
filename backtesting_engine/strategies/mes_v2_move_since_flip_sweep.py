"""
MES v2 Move-Since-SM-Flip Gate Sweep
=====================================
Sweeps max_move_pts threshold for MES v2. The gate blocks entries when price
has already moved too far since SM last crossed zero. The SM flip is the
"signal start" -- any move since then is already priced in.

Hypothesis: if SM flipped bullish at 21750 and price is now 21782, that's 32 pts
since the flip. A tight max_move threshold would block this stale entry.

Sweep: max_move_pts = [5, 10, 15, 20, 25, 30, 35, 40] (8 configs + baseline)
breakeven_after_bars=75 held constant throughout.

Usage:
    python3 mes_v2_move_since_flip_sweep.py
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

SWEEP_VALUES = [None, 5, 10, 15, 20, 25, 30, 35, 40]
SCRIPT_NAME = "mes_v2_move_since_flip_sweep.py"


def build_move_since_flip_gate(closes, sm, max_move_pts, sm_threshold=0.0):
    """Block entries when price has already moved too far since SM last crossed zero.

    For longs (sm > threshold): block if price rallied > max_move_pts since flip.
    For shorts (sm < -threshold): block if price dropped > max_move_pts since flip.
    """
    n = len(closes)
    gate = np.ones(n, dtype=bool)
    last_flip_idx = 0

    for j in range(1, n):
        # Detect SM zero crossing
        if (sm[j] > 0 and sm[j - 1] <= 0) or (sm[j] < 0 and sm[j - 1] >= 0):
            last_flip_idx = j

        # For potential longs: block if rallied too much since flip
        if sm[j] > sm_threshold:
            move = closes[j] - closes[last_flip_idx]
            gate[j] = move <= max_move_pts
        # For potential shorts: block if dropped too much since flip
        elif sm[j] < -sm_threshold:
            move = closes[last_flip_idx] - closes[j]
            gate[j] = move <= max_move_pts

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
    print("MES v2 MOVE-SINCE-SM-FLIP GATE SWEEP (pre-filter)")
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
    full_sm = df_mes["SM_Net"].values
    full_closes = df_mes["Close"].values
    is_len = len(mes_is)

    gates_by_val = {}
    for max_move in SWEEP_VALUES:
        if max_move is None:
            gates_by_val[max_move] = {"FULL": None, "IS": None, "OOS": None}
        else:
            full_gate = build_move_since_flip_gate(full_closes, full_sm,
                                                    max_move, MESV2_SM_THRESHOLD)
            gates_by_val[max_move] = {
                "FULL": full_gate,
                "IS": full_gate[:is_len],
                "OOS": full_gate[is_len:],
            }

    # --- Gate pass rates ---
    print(f"\n{'='*60}")
    print("GATE PASS RATES (% of bars where gate=True)")
    print(f"{'='*60}")
    print(f"{'MaxMove':>12} {'FULL':>8} {'IS':>8} {'OOS':>8}")
    print(f"{'-'*40}")
    for max_move in SWEEP_VALUES:
        label = "None" if max_move is None else f"{max_move}"
        if max_move is None:
            print(f"{label:>12} {'100.0%':>8} {'100.0%':>8} {'100.0%':>8}")
        else:
            g = gates_by_val[max_move]
            full_pct = g["FULL"].mean() * 100
            is_pct = g["IS"].mean() * 100
            oos_pct = g["OOS"].mean() * 100
            print(f"{label:>12} {full_pct:>7.1f}% {is_pct:>7.1f}% {oos_pct:>7.1f}%")

    # --- Sweep ---
    results = []
    best_oos_sharpe = -999
    best_val = None
    baseline_counts = {}

    for max_move in SWEEP_VALUES:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            gate = gates_by_val[max_move][split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, entry_gate=gate)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            if max_move is None:
                baseline_counts[split_name] = sc["count"] if sc else 0

            results.append((max_move, split_name, dr, sc, trades))

            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = sc["sharpe"]
                best_val = max_move

    # --- Print summary table ---
    print(f"\n{'='*130}")
    print(f"{'MaxMove':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'Blocked':>8} {'Pass/Fail':>10}")
    print(f"{'-'*130}")

    oos_pf_by_val = {}

    for max_move, split_name, dr, sc, trades in results:
        label = "None" if max_move is None else f"{max_move}"

        if sc is None:
            print(f"{label:>8} {split_name:>5}   NO TRADES")
            if split_name == "OOS":
                oos_pf_by_val[max_move] = 0.0
            continue

        baseline = baseline_counts.get(split_name, 0)
        blocked = baseline - sc["count"]

        pf_label = ""
        if split_name == "OOS":
            oos_pf_by_val[max_move] = sc["pf"]
            passed = sc["pf"] > 1.0 and sc["sharpe"] > 0
            pf_label = "PASS" if passed else "FAIL"

        print(f"{label:>8} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{blocked:>8} {pf_label:>10}")

    # --- Monotonicity check ---
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK (OOS PF vs max_move threshold)")
    print(f"{'='*80}")

    non_none_vals = [v for v in SWEEP_VALUES if v is not None]
    oos_pfs = [oos_pf_by_val.get(v, 0.0) for v in non_none_vals]

    # Tighter thresholds (lower max_move) should have higher PF if the filter works.
    # Since we're sweeping from tight to loose, PF should decrease as max_move increases.
    monotonic_decreasing = True
    for i in range(1, len(oos_pfs)):
        if oos_pfs[i] > oos_pfs[i - 1] + 0.01:
            monotonic_decreasing = False
            break

    for i, v in enumerate(non_none_vals):
        arrow = ""
        if i > 0:
            diff = oos_pfs[i] - oos_pfs[i - 1]
            arrow = f"  ({'+' if diff >= 0 else ''}{diff:.3f})"
        print(f"  max_move={v:>3}  OOS PF={oos_pfs[i]:.3f}{arrow}")

    if monotonic_decreasing:
        print("  -> Monotonically non-increasing (good: tighter filter = better PF)")
    else:
        print("  -> NOT monotonic (PF does not consistently improve with tighter filter)")

    # --- Exit reason breakdown ---
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")
    all_reasons = set()
    for max_move, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())

    print(f"{'MaxMove':>8} ", end="")
    for reason in sorted(all_reasons):
        print(f"{reason:>10}", end="")
    print()
    print("-" * (8 + 10 * len(all_reasons)))

    for max_move, split_name, _, sc, _ in results:
        if split_name != "FULL":
            continue
        label = "None" if max_move is None else f"{max_move}"
        if sc is None:
            continue
        print(f"{label:>8} ", end="")
        for reason in sorted(all_reasons):
            print(f"{sc['exits'].get(reason, 0):>10}", end="")
        print()

    # --- Pass/fail assessment ---
    print(f"\n{'='*80}")
    print("PASS/FAIL ASSESSMENT")
    print(f"{'='*80}")
    baseline_is_pf = None
    baseline_oos_pf = None
    baseline_is_sharpe = None
    baseline_oos_sharpe = None
    baseline_is_count = 0
    baseline_oos_count = 0

    for max_move, split_name, _, sc, _ in results:
        if max_move is None and sc:
            if split_name == "IS":
                baseline_is_pf = sc["pf"]
                baseline_is_sharpe = sc["sharpe"]
                baseline_is_count = sc["count"]
            elif split_name == "OOS":
                baseline_oos_pf = sc["pf"]
                baseline_oos_sharpe = sc["sharpe"]
                baseline_oos_count = sc["count"]

    print(f"Baseline IS:  PF={baseline_is_pf:.3f}, Sharpe={baseline_is_sharpe:.3f}, N={baseline_is_count}")
    print(f"Baseline OOS: PF={baseline_oos_pf:.3f}, Sharpe={baseline_oos_sharpe:.3f}, N={baseline_oos_count}")
    print()

    pass_count = 0
    for max_move in non_none_vals:
        is_sc = None
        oos_sc = None
        for mv, sn, _, sc, _ in results:
            if mv == max_move and sn == "IS" and sc:
                is_sc = sc
            elif mv == max_move and sn == "OOS" and sc:
                oos_sc = sc

        if is_sc is None or oos_sc is None:
            print(f"  max_move={max_move:>3}  FAIL (missing data)")
            continue

        is_pf_chg = (is_sc["pf"] - baseline_is_pf) / baseline_is_pf * 100
        oos_pf_chg = (oos_sc["pf"] - baseline_oos_pf) / baseline_oos_pf * 100
        is_count_pct = is_sc["count"] / baseline_is_count * 100 if baseline_is_count else 0
        oos_count_pct = oos_sc["count"] / baseline_oos_count * 100 if baseline_oos_count else 0

        is_profitable = is_sc["net_dollar"] > 0
        oos_profitable = oos_sc["net_dollar"] > 0

        strong = (is_pf_chg >= 5 and oos_pf_chg >= 5 and
                  is_sc["sharpe"] >= baseline_is_sharpe and
                  oos_sc["sharpe"] >= baseline_oos_sharpe and
                  is_count_pct >= 70 and oos_count_pct >= 70 and
                  is_profitable and oos_profitable)
        marginal = (not strong and
                    is_pf_chg >= -5 and oos_pf_chg >= -5 and
                    (is_sc["sharpe"] >= baseline_is_sharpe or oos_sc["sharpe"] >= baseline_oos_sharpe) and
                    is_count_pct >= 70 and oos_count_pct >= 70 and
                    is_profitable and oos_profitable)

        if strong:
            verdict = "STRONG PASS"
            pass_count += 1
        elif marginal:
            verdict = "MARGINAL PASS"
            pass_count += 1
        else:
            verdict = "FAIL"

        print(f"  max_move={max_move:>3}  {verdict:>14}  IS PF {is_pf_chg:+.1f}%  OOS PF {oos_pf_chg:+.1f}%  "
              f"IS N={is_sc['count']}({is_count_pct:.0f}%)  OOS N={oos_sc['count']}({oos_count_pct:.0f}%)")

    print(f"\nPass rate: {pass_count}/{len(non_none_vals)} ({pass_count/len(non_none_vals)*100:.0f}%)")

    # --- Summary verdict ---
    print(f"\n{'='*80}")
    best_label = "None" if best_val is None else f"{best_val}"
    print(f"BEST OOS SHARPE: max_move={best_label} (Sharpe {best_oos_sharpe:.3f})")

    baseline_oos = oos_pf_by_val.get(None, 0.0)
    improving_vals = [v for v in non_none_vals if oos_pf_by_val.get(v, 0.0) > baseline_oos]
    if improving_vals:
        print(f"Thresholds that improve OOS PF over baseline: {improving_vals}")
    else:
        print("No threshold improves OOS PF over baseline (None).")
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
            "max_move_since_flip": best_val,
        }

        print(f"\nSaving trade CSVs for best max_move={best_label}...")
        for max_move, split_name, dr, sc, trades in results:
            if max_move == best_val and trades:
                save_backtest(
                    trades, strategy="MES_V2_MOVE_FLIP", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"max_move_since_flip={best_val}",
                )
    else:
        print("Baseline (None = no gate) is best -- no move-since-flip gate needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
