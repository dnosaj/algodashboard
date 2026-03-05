"""
MES v2 Prior Session S/R Gate Sweep
=====================================
Sweeps buffer_pts for prior session high/low support/resistance gate.
Blocks entries near prior session levels:
  Longs: block if close >= prev_high - buffer (approaching/at resistance)
  Shorts: block if close <= prev_low + buffer (approaching/at support)

Sweep: buffer_pts = [0, 2, 5, 8, 10, 15, 20] (7 configs + baseline)
breakeven_after_bars=75 held constant throughout.

Usage:
    python3 mes_v2_prior_session_sr_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_test_common import (
    compute_smart_money,
    compute_prior_day_levels,
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

SWEEP_VALUES = [None, 0, 2, 5, 8, 10, 15, 20]
SCRIPT_NAME = "mes_v2_prior_session_sr_sweep.py"


def build_prior_session_sr_gate(closes, sm, prev_high, prev_low,
                                 buffer_pts, sm_threshold=0.0):
    """Block entries near prior session S/R levels.

    Longs: block if close >= prev_high - buffer (approaching/at resistance).
    Shorts: block if close <= prev_low + buffer (approaching/at support).
    """
    n = len(closes)
    gate = np.ones(n, dtype=bool)
    for j in range(n):
        ph = prev_high[j]
        pl = prev_low[j]
        if np.isnan(ph) or np.isnan(pl):
            continue
        if sm[j] > sm_threshold:      # potential long: block near prior high
            gate[j] = closes[j] < ph - buffer_pts
        elif sm[j] < -sm_threshold:   # potential short: block near prior low
            gate[j] = closes[j] > pl + buffer_pts
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
    print("MES v2 PRIOR SESSION S/R GATE SWEEP (pre-filter)")
    print(f"  breakeven_after_bars={MESV2_BREAKEVEN_BARS} (constant)")
    print("=" * 80)

    # --- Load MES data ---
    print("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    print(f"  {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # Compute SM on full data
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm

    # Compute prior day levels on full data
    prev_high, prev_low, prev_close = compute_prior_day_levels(
        df_mes.index, df_mes["High"].values, df_mes["Low"].values,
        df_mes["Close"].values,
    )

    # Verify no NaN past first day
    first_valid = np.argmax(~np.isnan(prev_high))
    nan_after_first = np.sum(np.isnan(prev_high[first_valid:]))
    print(f"  Prior day levels: first valid bar index={first_valid}, "
          f"NaN after first valid={nan_after_first}")

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

    # --- Prepare arrays per split ---
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
    for buffer_pts in SWEEP_VALUES:
        if buffer_pts is None:
            gates_by_val[buffer_pts] = {"FULL": None, "IS": None, "OOS": None}
        else:
            full_gate = build_prior_session_sr_gate(
                full_closes, full_sm, prev_high, prev_low,
                buffer_pts, MESV2_SM_THRESHOLD)
            gates_by_val[buffer_pts] = {
                "FULL": full_gate,
                "IS": full_gate[:is_len],
                "OOS": full_gate[is_len:],
            }

    # --- Gate pass rates ---
    print(f"\n{'='*60}")
    print("GATE PASS RATES (% of bars where gate=True)")
    print(f"{'='*60}")
    print(f"{'Buffer':>12} {'FULL':>8} {'IS':>8} {'OOS':>8}")
    print(f"{'-'*40}")
    for buffer_pts in SWEEP_VALUES:
        label = "None" if buffer_pts is None else f"{buffer_pts}"
        if buffer_pts is None:
            print(f"{label:>12} {'100.0%':>8} {'100.0%':>8} {'100.0%':>8}")
        else:
            g = gates_by_val[buffer_pts]
            full_pct = g["FULL"].mean() * 100
            is_pct = g["IS"].mean() * 100
            oos_pct = g["OOS"].mean() * 100
            print(f"{label:>12} {full_pct:>7.1f}% {is_pct:>7.1f}% {oos_pct:>7.1f}%")

    # --- Sweep ---
    results = []
    best_oos_sharpe = -999
    best_val = None
    baseline_counts = {}

    for buffer_pts in SWEEP_VALUES:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            gate = gates_by_val[buffer_pts][split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, entry_gate=gate)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            if buffer_pts is None:
                baseline_counts[split_name] = sc["count"] if sc else 0

            results.append((buffer_pts, split_name, dr, sc, trades))

            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = sc["sharpe"]
                best_val = buffer_pts

    # --- Diagnostic: distance to prior levels for blocked vs allowed entries ---
    print(f"\n{'='*80}")
    print("DIAGNOSTIC: Distance to prior session levels for FULL data entries")
    print(f"{'='*80}")

    # Get baseline trades (no gate) to compare
    baseline_trades = None
    for buf, sn, _, _, trades in results:
        if buf is None and sn == "FULL":
            baseline_trades = trades
            break

    if baseline_trades:
        for buffer_pts in [v for v in SWEEP_VALUES if v is not None]:
            gate = gates_by_val[buffer_pts]["FULL"]
            # Check each baseline trade entry bar against the gate
            allowed_dists_high = []
            blocked_dists_high = []
            allowed_dists_low = []
            blocked_dists_low = []

            for t in baseline_trades:
                idx = t["entry_idx"]
                if idx >= len(gate):
                    continue
                ph = prev_high[idx]
                pl = prev_low[idx]
                entry_close = full_closes[idx - 1] if idx > 0 else full_closes[idx]

                if np.isnan(ph) or np.isnan(pl):
                    continue

                if t["side"] == "long":
                    dist = ph - entry_close
                    if gate[idx - 1] if idx > 0 else gate[idx]:
                        allowed_dists_high.append(dist)
                    else:
                        blocked_dists_high.append(dist)
                else:
                    dist = entry_close - pl
                    if gate[idx - 1] if idx > 0 else gate[idx]:
                        allowed_dists_low.append(dist)
                    else:
                        blocked_dists_low.append(dist)

            n_blocked = len(blocked_dists_high) + len(blocked_dists_low)
            n_allowed = len(allowed_dists_high) + len(allowed_dists_low)
            print(f"\n  buffer={buffer_pts}: blocked {n_blocked} entries, allowed {n_allowed}")
            if blocked_dists_high:
                print(f"    Blocked longs: avg dist to prior high = {np.mean(blocked_dists_high):.1f} pts "
                      f"(min={min(blocked_dists_high):.1f}, max={max(blocked_dists_high):.1f})")
            if allowed_dists_high:
                print(f"    Allowed longs: avg dist to prior high = {np.mean(allowed_dists_high):.1f} pts "
                      f"(min={min(allowed_dists_high):.1f}, max={max(allowed_dists_high):.1f})")
            if blocked_dists_low:
                print(f"    Blocked shorts: avg dist to prior low = {np.mean(blocked_dists_low):.1f} pts "
                      f"(min={min(blocked_dists_low):.1f}, max={max(blocked_dists_low):.1f})")
            if allowed_dists_low:
                print(f"    Allowed shorts: avg dist to prior low = {np.mean(allowed_dists_low):.1f} pts "
                      f"(min={min(allowed_dists_low):.1f}, max={max(allowed_dists_low):.1f})")

    # --- Print summary table ---
    print(f"\n{'='*130}")
    print(f"{'Buffer':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'Blocked':>8} {'Pass/Fail':>10}")
    print(f"{'-'*130}")

    oos_pf_by_val = {}

    for buffer_pts, split_name, dr, sc, trades in results:
        label = "None" if buffer_pts is None else f"{buffer_pts}"

        if sc is None:
            print(f"{label:>8} {split_name:>5}   NO TRADES")
            if split_name == "OOS":
                oos_pf_by_val[buffer_pts] = 0.0
            continue

        baseline = baseline_counts.get(split_name, 0)
        blocked = baseline - sc["count"]

        pf_label = ""
        if split_name == "OOS":
            oos_pf_by_val[buffer_pts] = sc["pf"]
            passed = sc["pf"] > 1.0 and sc["sharpe"] > 0
            pf_label = "PASS" if passed else "FAIL"

        print(f"{label:>8} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{blocked:>8} {pf_label:>10}")

    # --- Monotonicity check ---
    # Wider buffer = more entries blocked. PF should increase with wider buffer
    # if the filter targets bad entries.
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK (OOS PF vs buffer_pts)")
    print(f"{'='*80}")

    non_none_vals = [v for v in SWEEP_VALUES if v is not None]
    oos_pfs = [oos_pf_by_val.get(v, 0.0) for v in non_none_vals]

    monotonic = True
    for i in range(1, len(oos_pfs)):
        if oos_pfs[i] < oos_pfs[i - 1] - 0.01:
            monotonic = False
            break

    for i, v in enumerate(non_none_vals):
        arrow = ""
        if i > 0:
            diff = oos_pfs[i] - oos_pfs[i - 1]
            arrow = f"  ({'+' if diff >= 0 else ''}{diff:.3f})"
        print(f"  buffer={v:>3}  OOS PF={oos_pfs[i]:.3f}{arrow}")

    if monotonic:
        print("  -> Monotonically non-decreasing (good: wider buffer = better PF)")
    else:
        print("  -> NOT monotonic (PF does not consistently improve with wider buffer)")

    # --- Exit reason breakdown ---
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")
    all_reasons = set()
    for buffer_pts, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())

    print(f"{'Buffer':>8} ", end="")
    for reason in sorted(all_reasons):
        print(f"{reason:>10}", end="")
    print()
    print("-" * (8 + 10 * len(all_reasons)))

    for buffer_pts, split_name, _, sc, _ in results:
        if split_name != "FULL":
            continue
        label = "None" if buffer_pts is None else f"{buffer_pts}"
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

    for buffer_pts, split_name, _, sc, _ in results:
        if buffer_pts is None and sc:
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
    for buffer_pts in non_none_vals:
        is_sc = None
        oos_sc = None
        for buf, sn, _, sc, _ in results:
            if buf == buffer_pts and sn == "IS" and sc:
                is_sc = sc
            elif buf == buffer_pts and sn == "OOS" and sc:
                oos_sc = sc

        if is_sc is None or oos_sc is None:
            print(f"  buffer={buffer_pts:>3}  FAIL (missing data)")
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

        print(f"  buffer={buffer_pts:>3}  {verdict:>14}  IS PF {is_pf_chg:+.1f}%  OOS PF {oos_pf_chg:+.1f}%  "
              f"IS N={is_sc['count']}({is_count_pct:.0f}%)  OOS N={oos_sc['count']}({oos_count_pct:.0f}%)")

    print(f"\nPass rate: {pass_count}/{len(non_none_vals)} ({pass_count/len(non_none_vals)*100:.0f}%)")

    # --- Summary ---
    print(f"\n{'='*80}")
    best_label = "None" if best_val is None else f"{best_val}"
    print(f"BEST OOS SHARPE: buffer={best_label} (Sharpe {best_oos_sharpe:.3f})")

    baseline_oos = oos_pf_by_val.get(None, 0.0)
    improving_vals = [v for v in non_none_vals if oos_pf_by_val.get(v, 0.0) > baseline_oos]
    if improving_vals:
        print(f"Thresholds that improve OOS PF over baseline: {improving_vals}")
    else:
        print("No buffer threshold improves OOS PF over baseline (None).")
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
            "prior_session_sr_buffer": best_val,
        }

        print(f"\nSaving trade CSVs for best buffer={best_label}...")
        for buffer_pts, split_name, dr, sc, trades in results:
            if buffer_pts == best_val and trades:
                save_backtest(
                    trades, strategy="MES_V2_PRIOR_SR", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"prior_session_sr_buffer={best_val}",
                )
    else:
        print("Baseline (None = no gate) is best -- no prior session S/R gate needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
