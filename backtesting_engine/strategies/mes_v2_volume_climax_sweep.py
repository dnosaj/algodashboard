"""
MES v2 Volume Climax Gate Sweep
================================
Sweeps volume climax gate parameters for MES v2.
Blocks entries when current bar volume is a climax spike relative to
recent average. Keeps breakeven_after_bars=75 constant.

Sweep grid (3x4 = 12 combos + baseline = 13 configs):
  Lookback = [10, 20, 50] bars
  Max ratio = [1.5, 2.0, 2.5, 3.0]
  Plus baseline (no gate)

Usage:
    python3 mes_v2_volume_climax_sweep.py
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

LOOKBACKS = [10, 20, 50]
MAX_RATIOS = [1.5, 2.0, 2.5, 3.0]
SCRIPT_NAME = "mes_v2_volume_climax_sweep.py"


def build_volume_climax_gate(volumes, lookback, max_ratio):
    """Gate: block entries when current bar volume is a climax spike.
    Block if volume[j] / avg(volume[j-lookback:j]) > max_ratio.
    """
    n = len(volumes)
    gate = np.ones(n, dtype=bool)
    for j in range(lookback, n):
        avg_vol = np.mean(volumes[j - lookback:j])
        if avg_vol > 0:
            ratio = volumes[j] / avg_vol
            gate[j] = ratio <= max_ratio  # block if volume spike too large
    return gate


def run_single(opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, entry_gate=None):
    """Run MES v2 backtest with optional entry gate, BE_TIME=75 constant."""
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
    print("MES v2 VOLUME CLIMAX GATE SWEEP (pre-filter)")
    print(f"  Lookbacks: {LOOKBACKS}")
    print(f"  Max ratios: {MAX_RATIOS}")
    print(f"  breakeven_after_bars={MESV2_BREAKEVEN_BARS} (constant)")
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

    # Volume array (full data, used to build gate before split)
    volumes_full = df_mes["Volume"].values

    # Data range
    start_date = df_mes.index[0].strftime("%Y-%m-%d")
    end_date = df_mes.index[-1].strftime("%Y-%m-%d")
    data_range = f"{start_date}_to_{end_date}"

    # --- Define splits ---
    midpoint = df_mes.index[len(df_mes) // 2]
    mes_is = df_mes[df_mes.index < midpoint]
    mes_oos = df_mes[df_mes.index >= midpoint]
    is_len = len(mes_is)

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
    # Key: (lookback, max_ratio) -> {"FULL": gate_full, "IS": gate_is, "OOS": gate_oos}
    gates = {}
    for lb in LOOKBACKS:
        for mr in MAX_RATIOS:
            gate_full = build_volume_climax_gate(volumes_full, lb, mr)
            gate_is = gate_full[:is_len]
            gate_oos = gate_full[is_len:]
            gates[(lb, mr)] = {"FULL": gate_full, "IS": gate_is, "OOS": gate_oos}
            blocked = np.sum(~gate_full)
            print(f"  Gate LB={lb:>2} MR={mr:.1f}: {blocked} bars blocked "
                  f"({blocked/len(gate_full)*100:.1f}%)")

    # --- Sweep: baseline + all combos ---
    # results dict: (lookback, max_ratio) -> {split_name: (sc, trades)}
    # baseline uses key (0, 0)
    all_results = {}

    # Baseline (no gate)
    print("\nRunning baseline (no gate)...")
    all_results[(0, 0)] = {}
    for split_name, _, dr in splits:
        opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
        trades = run_single(opens, highs, lows, closes, sm_arr, times,
                            rsi_curr, rsi_prev, entry_gate=None)
        sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                          dollar_per_pt=MES_DOLLAR_PER_PT)
        all_results[(0, 0)][split_name] = (sc, trades, dr)

    # Gated combos
    for lb in LOOKBACKS:
        for mr in MAX_RATIOS:
            print(f"  Running LB={lb} MR={mr:.1f}...")
            all_results[(lb, mr)] = {}
            for split_name, _, dr in splits:
                opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
                gate = gates[(lb, mr)][split_name]
                trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                    rsi_curr, rsi_prev, entry_gate=gate)
                sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                                  dollar_per_pt=MES_DOLLAR_PER_PT)
                all_results[(lb, mr)][split_name] = (sc, trades, dr)

    # --- Print detailed summary table ---
    print(f"\n{'='*130}")
    print(f"{'LB':>4} {'MR':>5} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} {'Exits':>30}")
    print(f"{'-'*130}")

    config_keys = [(0, 0)] + [(lb, mr) for lb in LOOKBACKS for mr in MAX_RATIOS]
    for key in config_keys:
        lb, mr = key
        label_lb = "none" if lb == 0 else str(lb)
        label_mr = "none" if mr == 0 else f"{mr:.1f}"
        for split_name in ["FULL", "IS", "OOS"]:
            sc, trades, dr = all_results[key][split_name]
            if sc is None:
                print(f"{label_lb:>4} {label_mr:>5} {split_name:>5}   NO TRADES")
                continue
            exits_str = ", ".join(f"{k}:{v}" for k, v in sorted(sc["exits"].items()))
            print(f"{label_lb:>4} {label_mr:>5} {split_name:>5} {sc['count']:>7} "
                  f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                  f"{sc['net_dollar']:>+10.2f} {sc['max_dd_dollar']:>9.2f} "
                  f"{sc['sharpe']:>7.3f} {exits_str:>30}")

    # --- Grid tables: PF for FULL, IS, OOS ---
    baseline_pfs = {}
    for split_name in ["FULL", "IS", "OOS"]:
        sc_base = all_results[(0, 0)][split_name][0]
        baseline_pfs[split_name] = sc_base["pf"] if sc_base else 0

    for split_name in ["FULL", "IS", "OOS"]:
        print(f"\n{'='*60}")
        print(f"PF GRID — {split_name}  (baseline PF = {baseline_pfs[split_name]:.3f})")
        print(f"{'='*60}")
        header = f"{'LB\\MR':>8}"
        for mr in MAX_RATIOS:
            header += f"  {mr:>6.1f}"
        print(header)
        print("-" * (8 + 8 * len(MAX_RATIOS)))

        for lb in LOOKBACKS:
            row = f"{lb:>8}"
            for mr in MAX_RATIOS:
                sc = all_results[(lb, mr)][split_name][0]
                pf = sc["pf"] if sc else 0
                row += f"  {pf:>6.3f}"
            print(row)

    # --- Sharpe grid ---
    baseline_sharpes = {}
    for split_name in ["FULL", "IS", "OOS"]:
        sc_base = all_results[(0, 0)][split_name][0]
        baseline_sharpes[split_name] = sc_base["sharpe"] if sc_base else 0

    for split_name in ["FULL", "IS", "OOS"]:
        print(f"\n{'='*60}")
        print(f"SHARPE GRID — {split_name}  (baseline Sharpe = {baseline_sharpes[split_name]:.3f})")
        print(f"{'='*60}")
        header = f"{'LB\\MR':>8}"
        for mr in MAX_RATIOS:
            header += f"  {mr:>6.1f}"
        print(header)
        print("-" * (8 + 8 * len(MAX_RATIOS)))

        for lb in LOOKBACKS:
            row = f"{lb:>8}"
            for mr in MAX_RATIOS:
                sc = all_results[(lb, mr)][split_name][0]
                sharpe = sc["sharpe"] if sc else 0
                row += f"  {sharpe:>6.3f}"
            print(row)

    # --- Pass/Fail Assessment ---
    print(f"\n{'='*80}")
    print("PASS/FAIL ASSESSMENT")
    print(f"{'='*80}")
    print("Pass criteria: PF >= baseline AND Sharpe >= baseline on BOTH IS and OOS")
    print()

    pass_count = 0
    total = len(LOOKBACKS) * len(MAX_RATIOS)

    print(f"{'LB':>4} {'MR':>5} {'IS PF':>7} {'OOS PF':>8} {'IS Sh':>7} {'OOS Sh':>8} {'Result':>8}")
    print("-" * 55)

    for lb in LOOKBACKS:
        for mr in MAX_RATIOS:
            sc_is = all_results[(lb, mr)]["IS"][0]
            sc_oos = all_results[(lb, mr)]["OOS"][0]
            is_pf = sc_is["pf"] if sc_is else 0
            oos_pf = sc_oos["pf"] if sc_oos else 0
            is_sh = sc_is["sharpe"] if sc_is else 0
            oos_sh = sc_oos["sharpe"] if sc_oos else 0

            passed = (is_pf >= baseline_pfs["IS"] and
                      oos_pf >= baseline_pfs["OOS"] and
                      is_sh >= baseline_sharpes["IS"] and
                      oos_sh >= baseline_sharpes["OOS"])
            if passed:
                pass_count += 1
            verdict = "PASS" if passed else "FAIL"
            print(f"{lb:>4} {mr:>5.1f} {is_pf:>7.3f} {oos_pf:>8.3f} "
                  f"{is_sh:>7.3f} {oos_sh:>8.3f} {verdict:>8}")

    print(f"\nPass rate: {pass_count}/{total} ({pass_count/total*100:.0f}%)")

    # --- Monotonicity Check ---
    # For each lookback, check if PF decreases monotonically as max_ratio increases
    # (looser gate -> closer to baseline). Monotonicity means the gate effect is
    # smooth and predictable.
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK")
    print("For each lookback, does PF change monotonically as max_ratio increases?")
    print("(As ratio loosens, PF should approach baseline smoothly)")
    print(f"{'='*80}")

    for split_name in ["FULL", "IS", "OOS"]:
        print(f"\n  {split_name}:")
        for lb in LOOKBACKS:
            pfs = []
            for mr in MAX_RATIOS:
                sc = all_results[(lb, mr)][split_name][0]
                pfs.append(sc["pf"] if sc else 0)

            # Check monotonicity (non-decreasing as ratio loosens)
            diffs = [pfs[i+1] - pfs[i] for i in range(len(pfs)-1)]
            mono_inc = all(d >= -0.005 for d in diffs)  # small tolerance
            mono_dec = all(d <= 0.005 for d in diffs)
            if mono_inc:
                mono_label = "NON-DECREASING"
            elif mono_dec:
                mono_label = "NON-INCREASING"
            else:
                mono_label = "NON-MONOTONIC"

            pf_str = " -> ".join(f"{p:.3f}" for p in pfs)
            print(f"    LB={lb:>2}: {pf_str}  [{mono_label}]")

    # --- Best config ---
    print(f"\n{'='*80}")
    best_oos_sharpe = -999
    best_key = (0, 0)
    for key in config_keys:
        sc_oos = all_results[key]["OOS"][0]
        if sc_oos and sc_oos["sharpe"] > best_oos_sharpe:
            best_oos_sharpe = sc_oos["sharpe"]
            best_key = key

    if best_key == (0, 0):
        print("BEST OOS SHARPE: BASELINE (no volume climax gate needed)")
    else:
        lb, mr = best_key
        sc_oos = all_results[best_key]["OOS"][0]
        sc_full = all_results[best_key]["FULL"][0]
        print(f"BEST OOS SHARPE: LB={lb} MR={mr:.1f} "
              f"(OOS Sharpe {best_oos_sharpe:.3f}, PF {sc_oos['pf']:.3f})")
        print(f"  FULL: {fmt_score(sc_full, 'FULL')}")
        print(f"  OOS:  {fmt_score(sc_oos, 'OOS')}")
    print(f"{'='*80}")

    # --- Save best gated result (if it beats baseline) ---
    if best_key != (0, 0):
        lb, mr = best_key
        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
            "vol_climax_lookback": lb,
            "vol_climax_max_ratio": mr,
        }

        print(f"\nSaving trade CSVs for best config LB={lb} MR={mr:.1f}...")
        for split_name in ["FULL", "IS", "OOS"]:
            sc, trades, dr = all_results[best_key][split_name]
            if trades:
                save_backtest(
                    trades, strategy="MES_V2_VOL_CLIMAX", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"vol_climax_lookback={lb}, max_ratio={mr}",
                )
    else:
        print("\nBaseline is best — no volume climax gate improves OOS Sharpe.")

    print("\nDone.")


if __name__ == "__main__":
    main()
