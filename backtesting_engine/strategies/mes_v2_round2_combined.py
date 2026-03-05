"""
MES v2 Round 2 Combined Entry Filters
=======================================
Tests individual and combined best-passing filters from Round 2 studies:
  1. Move-since-SM-flip gate
  2. ATR-normalized extension gate
  3. Prior session S/R gate

Also includes SM slope gate (Round 1 winner) in combinations.

Combinations tested:
  Baseline (no filter)
  Move-since-flip only
  ATR extension only
  Prior session S/R only
  SM slope only (Round 1)
  Flip + ATR (AND)
  Flip + S/R (AND)
  ATR + S/R (AND)
  SM slope + Flip (AND)
  SM slope + ATR (AND)
  SM slope + S/R (AND)
  All four (AND)

Set any threshold to None to skip that filter from combinations.
Update constants below after reviewing individual study results.

Usage:
    python3 mes_v2_round2_combined.py
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

# Import gate builders
from mes_v2_move_since_flip_sweep import build_move_since_flip_gate
from mes_v2_atr_extension_sweep import build_atr_extension_gate
from mes_v2_prior_session_sr_sweep import build_prior_session_sr_gate
from mes_v2_sm_slope_sweep import build_sm_slope_gate

# ===========================================================================
# UPDATE THESE AFTER REVIEWING INDIVIDUAL STUDY RESULTS
# Set to None to skip that filter from all combinations.
# ===========================================================================
BEST_MAX_MOVE_PTS = 5           # Study 1: move-since-flip (only OOS improver)
BEST_ATR_LOOKBACK = None        # Study 2: ATR extension — SKIP (2/20 pass, no real signal)
BEST_ATR_MULT = None            # Study 2: ATR extension — SKIP
BEST_SR_BUFFER = 0              # Study 3: prior session S/R (buffer=0 keeps 70% trades)
BEST_SM_SLOPE = 0.005           # Round 1: SM slope threshold (OOS PF 1.54, Sharpe 2.50)
ATR_PERIOD = 14

SCRIPT_NAME = "mes_v2_round2_combined.py"


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
    print("MES v2 ROUND 2 COMBINED ENTRY FILTERS")
    print(f"  breakeven_after_bars={MESV2_BREAKEVEN_BARS} (constant)")
    print("=" * 80)

    # Print active filters
    print("\nActive filters:")
    if BEST_MAX_MOVE_PTS is not None:
        print(f"  Move-since-flip: max_move={BEST_MAX_MOVE_PTS} pts")
    else:
        print("  Move-since-flip: SKIPPED")
    if BEST_ATR_LOOKBACK is not None and BEST_ATR_MULT is not None:
        print(f"  ATR extension: LB={BEST_ATR_LOOKBACK}, mult={BEST_ATR_MULT}")
    else:
        print("  ATR extension: SKIPPED")
    if BEST_SR_BUFFER is not None:
        print(f"  Prior session S/R: buffer={BEST_SR_BUFFER} pts")
    else:
        print("  Prior session S/R: SKIPPED")
    if BEST_SM_SLOPE is not None:
        print(f"  SM slope (Round 1): threshold={BEST_SM_SLOPE}")
    else:
        print("  SM slope: SKIPPED")

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

    # Prior day levels
    prev_high, prev_low, prev_close = compute_prior_day_levels(
        df_mes.index, df_mes["High"].values, df_mes["Low"].values,
        df_mes["Close"].values,
    )

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

    # --- Build individual gates on FULL data ---
    full_sm = df_mes["SM_Net"].values
    full_closes = df_mes["Close"].values
    full_highs = df_mes["High"].values
    full_lows = df_mes["Low"].values
    is_len = len(mes_is)

    # Build individual gate arrays (None if filter is skipped)
    gate_flip = None
    if BEST_MAX_MOVE_PTS is not None:
        gate_flip = build_move_since_flip_gate(
            full_closes, full_sm, BEST_MAX_MOVE_PTS, MESV2_SM_THRESHOLD)

    gate_atr = None
    if BEST_ATR_LOOKBACK is not None and BEST_ATR_MULT is not None:
        gate_atr = build_atr_extension_gate(
            full_closes, full_highs, full_lows, full_sm,
            BEST_ATR_LOOKBACK, BEST_ATR_MULT, ATR_PERIOD, MESV2_SM_THRESHOLD)

    gate_sr = None
    if BEST_SR_BUFFER is not None:
        gate_sr = build_prior_session_sr_gate(
            full_closes, full_sm, prev_high, prev_low,
            BEST_SR_BUFFER, MESV2_SM_THRESHOLD)

    gate_slope = None
    if BEST_SM_SLOPE is not None:
        gate_slope = build_sm_slope_gate(full_sm, BEST_SM_SLOPE, MESV2_SM_THRESHOLD)

    # --- Define combinations ---
    individual_gates = {
        "flip": gate_flip,
        "atr": gate_atr,
        "sr": gate_sr,
        "slope": gate_slope,
    }

    # Build named combinations
    combos = [("Baseline", None)]

    # Singles
    for name, gate in individual_gates.items():
        if gate is not None:
            combos.append((name.capitalize(), gate))

    # Pairs
    gate_names = [(n, g) for n, g in individual_gates.items() if g is not None]
    for i in range(len(gate_names)):
        for j in range(i + 1, len(gate_names)):
            n1, g1 = gate_names[i]
            n2, g2 = gate_names[j]
            combined = g1 & g2
            combos.append((f"{n1.capitalize()}+{n2.capitalize()}", combined))

    # All active filters
    if len(gate_names) >= 3:
        all_gate = gate_names[0][1].copy()
        for _, g in gate_names[1:]:
            all_gate &= g
        combo_name = "+".join(n.capitalize() for n, _ in gate_names)
        combos.append((f"ALL({combo_name})", all_gate))

    print(f"\nTesting {len(combos)} combinations:")
    for name, _ in combos:
        print(f"  - {name}")

    # --- Slice gates for IS/OOS ---
    combo_gates = {}
    for name, gate in combos:
        if gate is None:
            combo_gates[name] = {"FULL": None, "IS": None, "OOS": None}
        else:
            combo_gates[name] = {
                "FULL": gate,
                "IS": gate[:is_len],
                "OOS": gate[is_len:],
            }

    # --- Gate pass rates ---
    print(f"\n{'='*60}")
    print("GATE PASS RATES (% of bars where gate=True)")
    print(f"{'='*60}")
    print(f"{'Combo':>30} {'FULL':>8} {'IS':>8} {'OOS':>8}")
    print(f"{'-'*56}")
    for name, _ in combos:
        g = combo_gates[name]
        if g["FULL"] is None:
            print(f"{name:>30} {'100.0%':>8} {'100.0%':>8} {'100.0%':>8}")
        else:
            full_pct = g["FULL"].mean() * 100
            is_pct = g["IS"].mean() * 100
            oos_pct = g["OOS"].mean() * 100
            print(f"{name:>30} {full_pct:>7.1f}% {is_pct:>7.1f}% {oos_pct:>7.1f}%")

    # --- Sweep ---
    results = []
    best_oos_sharpe = -999
    best_combo = None
    baseline_counts = {}

    for name, _ in combos:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            gate = combo_gates[name][split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, entry_gate=gate)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            if name == "Baseline":
                baseline_counts[split_name] = sc["count"] if sc else 0

            results.append((name, split_name, dr, sc, trades))

            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = sc["sharpe"]
                best_combo = name

    # --- Print summary table ---
    print(f"\n{'='*140}")
    print(f"{'Combo':>30} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'Blocked':>8} {'Pass/Fail':>10}")
    print(f"{'-'*140}")

    for name, split_name, dr, sc, trades in results:
        if sc is None:
            print(f"{name:>30} {split_name:>5}   NO TRADES")
            continue

        baseline = baseline_counts.get(split_name, 0)
        blocked = baseline - sc["count"]

        pf_label = ""
        if split_name == "OOS":
            passed = sc["pf"] > 1.0 and sc["sharpe"] > 0
            pf_label = "PASS" if passed else "FAIL"

        print(f"{name:>30} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{blocked:>8} {pf_label:>10}")

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

    for name, split_name, _, sc, _ in results:
        if name == "Baseline" and sc:
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

    for name, _ in combos:
        if name == "Baseline":
            continue

        is_sc = None
        oos_sc = None
        for cn, sn, _, sc, _ in results:
            if cn == name and sc:
                if sn == "IS":
                    is_sc = sc
                elif sn == "OOS":
                    oos_sc = sc

        if is_sc is None or oos_sc is None:
            print(f"  {name:>30}  FAIL (missing data)")
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
        elif marginal:
            verdict = "MARGINAL PASS"
        else:
            verdict = "FAIL"

        print(f"  {name:>30}  {verdict:>14}  IS PF {is_pf_chg:+.1f}%  OOS PF {oos_pf_chg:+.1f}%  "
              f"IS N={is_sc['count']}({is_count_pct:.0f}%)  OOS N={oos_sc['count']}({oos_count_pct:.0f}%)")

    # --- Exit reason breakdown ---
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")
    all_reasons = set()
    for name, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())

    print(f"{'Combo':>30} ", end="")
    for reason in sorted(all_reasons):
        print(f"{reason:>10}", end="")
    print()
    print("-" * (30 + 10 * len(all_reasons)))

    for name, split_name, _, sc, _ in results:
        if split_name != "FULL" or sc is None:
            continue
        print(f"{name:>30} ", end="")
        for reason in sorted(all_reasons):
            print(f"{sc['exits'].get(reason, 0):>10}", end="")
        print()

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"BEST OOS SHARPE: {best_combo} (Sharpe {best_oos_sharpe:.3f})")
    print(f"{'='*80}")

    # --- Save best result ---
    if best_combo and best_combo != "Baseline":
        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
            "combined_filter": best_combo,
            "max_move_since_flip": BEST_MAX_MOVE_PTS,
            "atr_extension_lookback": BEST_ATR_LOOKBACK,
            "atr_extension_max_mult": BEST_ATR_MULT,
            "prior_session_sr_buffer": BEST_SR_BUFFER,
            "sm_slope_threshold": BEST_SM_SLOPE,
        }

        print(f"\nSaving trade CSVs for best combo: {best_combo}...")
        for name, split_name, dr, sc, trades in results:
            if name == best_combo and trades:
                save_backtest(
                    trades, strategy="MES_V2_R2_COMBINED", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"best_combo={best_combo}",
                )
    else:
        print("Baseline is best -- no combined filter improves over it.")

    print("\nDone.")


if __name__ == "__main__":
    main()
