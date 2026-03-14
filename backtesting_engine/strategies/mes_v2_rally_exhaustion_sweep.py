"""
MES v2 Rally Exhaustion Gate Sweep
===================================
Sweeps rally exhaustion gate parameters for MES v2.
Gate blocks entries when price has already moved strongly in the entry direction
over the lookback window — i.e., the move may be exhausted.

Sweep grid: 4 lookbacks x 6 max_move thresholds = 24 combos + baseline = 25 configs.
breakeven_after_bars=75 held constant throughout.

Usage:
    python3 mes_v2_rally_exhaustion_sweep.py
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

# --- Sweep grid ---
LOOKBACK_VALUES = [5, 10, 15, 20]        # bars
MAX_MOVE_VALUES = [10, 15, 20, 25, 30, 35]  # pts
SCRIPT_NAME = "mes_v2_rally_exhaustion_sweep.py"


def build_rally_exhaustion_gate(closes, sm, lookback_n, max_move_pts, sm_threshold=0.0):
    """Gate: block entries when price has already moved strongly in entry direction.
    For longs (sm > sm_threshold): block if price rallied > max_move_pts in last N bars.
    For shorts (sm < -sm_threshold): block if price dropped > max_move_pts in last N bars.
    """
    n = len(closes)
    gate = np.ones(n, dtype=bool)
    for j in range(lookback_n, n):
        move = closes[j] - closes[j - lookback_n]
        if sm[j] > sm_threshold:      # potential long: block if already rallied too much
            gate[j] = move <= max_move_pts
        elif sm[j] < -sm_threshold:   # potential short: block if already dropped too much
            gate[j] = (-move) <= max_move_pts
    return gate


def run_single(opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, entry_gate=None):
    """Run MES v2 backtest with optional entry gate.
    breakeven_after_bars=75 held constant."""
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
    print("MES v2 RALLY EXHAUSTION GATE SWEEP (pre-filter)")
    print(f"  Lookbacks: {LOOKBACK_VALUES}")
    print(f"  Max move:  {MAX_MOVE_VALUES} pts")
    print(f"  BE_TIME:   {MESV2_BREAKEVEN_BARS} bars (constant)")
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

    # --- Build gates on FULL data (before IS/OOS split) ---
    full_closes = df_mes["Close"].values
    full_sm = df_mes["SM_Net"].values

    gates_full = {}  # (lookback, max_move) -> gate array over full data
    for lb in LOOKBACK_VALUES:
        for mx in MAX_MOVE_VALUES:
            gates_full[(lb, mx)] = build_rally_exhaustion_gate(
                full_closes, full_sm, lb, mx,
                sm_threshold=MESV2_SM_THRESHOLD,
            )

    # --- Define splits ---
    midpoint = df_mes.index[len(df_mes) // 2]
    n_full = len(df_mes)
    n_is = (df_mes.index < midpoint).sum()

    mes_is = df_mes[df_mes.index < midpoint]
    mes_oos = df_mes[df_mes.index >= midpoint]

    is_range = f"{mes_is.index[0].strftime('%Y-%m-%d')}_to_{mes_is.index[-1].strftime('%Y-%m-%d')}"
    oos_range = f"{mes_oos.index[0].strftime('%Y-%m-%d')}_to_{mes_oos.index[-1].strftime('%Y-%m-%d')}"

    splits = [
        ("FULL", df_mes, data_range, 0, n_full),
        ("IS", mes_is, is_range, 0, n_is),
        ("OOS", mes_oos, oos_range, n_is, n_full),
    ]

    # --- Prepare arrays per split (RSI mapped within each split) ---
    split_arrays = {}
    for split_name, df_split, _, _, _ in splits:
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

    # --- Run baseline (no gate) ---
    print("\nRunning baseline (no gate)...")
    baseline_scores = {}
    baseline_trades = {}
    for split_name, _, dr, _, _ in splits:
        opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
        trades = run_single(opens, highs, lows, closes, sm_arr, times,
                            rsi_curr, rsi_prev, entry_gate=None)
        sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                          dollar_per_pt=MES_DOLLAR_PER_PT)
        baseline_scores[split_name] = sc
        baseline_trades[split_name] = trades
        if sc:
            print(f"  {split_name}: {sc['count']} trades, PF {sc['pf']:.3f}, "
                  f"Sharpe {sc['sharpe']:.3f}, Net ${sc['net_dollar']:+.2f}")

    # --- Sweep all combos ---
    print(f"\nSweeping {len(LOOKBACK_VALUES)} x {len(MAX_MOVE_VALUES)} = "
          f"{len(LOOKBACK_VALUES) * len(MAX_MOVE_VALUES)} combos...")

    # results[split_name][(lb, mx)] = (sc, trades)
    sweep_results = {s: {} for s, _, _, _, _ in splits}

    for lb in LOOKBACK_VALUES:
        for mx in MAX_MOVE_VALUES:
            gate_full = gates_full[(lb, mx)]
            for split_name, _, dr, idx_start, idx_end in splits:
                opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
                # Slice the full gate to the split range
                gate_split = gate_full[idx_start:idx_end]
                trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                    rsi_curr, rsi_prev, entry_gate=gate_split)
                sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                                  dollar_per_pt=MES_DOLLAR_PER_PT)
                sweep_results[split_name][(lb, mx)] = (sc, trades)

    # --- Print grid tables (PF) ---
    for split_name in ["FULL", "IS", "OOS"]:
        bl = baseline_scores[split_name]
        bl_pf = bl["pf"] if bl else 0.0
        bl_trades = bl["count"] if bl else 0

        print(f"\n{'='*80}")
        print(f"{split_name} PF GRID  (baseline PF={bl_pf:.3f}, {bl_trades} trades)")
        print(f"{'='*80}")

        # Header
        header = f"{'Lookback':>8}"
        for mx in MAX_MOVE_VALUES:
            header += f"  {mx:>5}pt"
        print(header)
        print("-" * len(header))

        for lb in LOOKBACK_VALUES:
            row = f"{lb:>8}"
            for mx in MAX_MOVE_VALUES:
                sc, _ = sweep_results[split_name][(lb, mx)]
                if sc and sc["count"] >= 10:
                    row += f"  {sc['pf']:>6.3f}"
                elif sc:
                    row += f"  {sc['pf']:>5.3f}*"  # asterisk = low count
                else:
                    row += f"  {'N/A':>6}"
            print(row)

    # --- Print grid tables (Trade Count) ---
    for split_name in ["FULL", "IS", "OOS"]:
        bl = baseline_scores[split_name]
        bl_trades = bl["count"] if bl else 0

        print(f"\n{'='*80}")
        print(f"{split_name} TRADE COUNT GRID  (baseline={bl_trades})")
        print(f"{'='*80}")

        header = f"{'Lookback':>8}"
        for mx in MAX_MOVE_VALUES:
            header += f"  {mx:>5}pt"
        print(header)
        print("-" * len(header))

        for lb in LOOKBACK_VALUES:
            row = f"{lb:>8}"
            for mx in MAX_MOVE_VALUES:
                sc, _ = sweep_results[split_name][(lb, mx)]
                cnt = sc["count"] if sc else 0
                row += f"  {cnt:>6}"
            print(row)

    # --- Print grid tables (Sharpe) ---
    for split_name in ["FULL", "IS", "OOS"]:
        bl = baseline_scores[split_name]
        bl_sharpe = bl["sharpe"] if bl else 0.0

        print(f"\n{'='*80}")
        print(f"{split_name} SHARPE GRID  (baseline Sharpe={bl_sharpe:.3f})")
        print(f"{'='*80}")

        header = f"{'Lookback':>8}"
        for mx in MAX_MOVE_VALUES:
            header += f"  {mx:>5}pt"
        print(header)
        print("-" * len(header))

        for lb in LOOKBACK_VALUES:
            row = f"{lb:>8}"
            for mx in MAX_MOVE_VALUES:
                sc, _ = sweep_results[split_name][(lb, mx)]
                if sc and sc["count"] >= 10:
                    row += f"  {sc['sharpe']:>6.3f}"
                elif sc:
                    row += f"  {sc['sharpe']:>5.3f}*"
                else:
                    row += f"  {'N/A':>6}"
            print(row)

    # --- Print grid tables (Net P&L) ---
    for split_name in ["FULL", "IS", "OOS"]:
        bl = baseline_scores[split_name]
        bl_pnl = bl["net_dollar"] if bl else 0.0

        print(f"\n{'='*80}")
        print(f"{split_name} NET P&L GRID  (baseline=${bl_pnl:+.2f})")
        print(f"{'='*80}")

        header = f"{'Lookback':>8}"
        for mx in MAX_MOVE_VALUES:
            header += f"  {mx:>7}pt"
        print(header)
        print("-" * len(header))

        for lb in LOOKBACK_VALUES:
            row = f"{lb:>8}"
            for mx in MAX_MOVE_VALUES:
                sc, _ = sweep_results[split_name][(lb, mx)]
                if sc:
                    row += f"  {sc['net_dollar']:>+8.0f}"
                else:
                    row += f"  {'N/A':>8}"
            print(row)

    # --- Pass/Fail assessment ---
    # Pass criteria: PF > baseline on BOTH IS and OOS, and count >= 30 on each
    print(f"\n{'='*80}")
    print("PASS/FAIL ASSESSMENT")
    print("  Pass = PF > baseline on BOTH IS and OOS, trade count >= 30 on each")
    print(f"{'='*80}")

    bl_is_pf = baseline_scores["IS"]["pf"] if baseline_scores["IS"] else 0.0
    bl_oos_pf = baseline_scores["OOS"]["pf"] if baseline_scores["OOS"] else 0.0

    header = f"{'Lookback':>8}"
    for mx in MAX_MOVE_VALUES:
        header += f"  {mx:>5}pt"
    print(header)
    print("-" * len(header))

    pass_count = 0
    total_combos = len(LOOKBACK_VALUES) * len(MAX_MOVE_VALUES)

    for lb in LOOKBACK_VALUES:
        row = f"{lb:>8}"
        for mx in MAX_MOVE_VALUES:
            sc_is, _ = sweep_results["IS"][(lb, mx)]
            sc_oos, _ = sweep_results["OOS"][(lb, mx)]

            is_ok = (sc_is and sc_is["count"] >= 30 and sc_is["pf"] > bl_is_pf)
            oos_ok = (sc_oos and sc_oos["count"] >= 30 and sc_oos["pf"] > bl_oos_pf)

            if is_ok and oos_ok:
                row += f"  {'PASS':>6}"
                pass_count += 1
            else:
                # Show which failed
                parts = []
                if not is_ok:
                    parts.append("IS")
                if not oos_ok:
                    parts.append("OOS")
                row += f"  {'FAIL':>6}"
        print(row)

    print(f"\n  {pass_count} of {total_combos} configs passed.")

    # --- Monotonicity check ---
    print(f"\n{'='*80}")
    print("MONOTONICITY CHECK")
    print("  For each lookback, does PF change smoothly as max_move increases?")
    print("  (Looser gate -> PF should trend toward baseline)")
    print(f"{'='*80}")

    for split_name in ["FULL", "IS", "OOS"]:
        print(f"\n  {split_name}:")
        for lb in LOOKBACK_VALUES:
            pf_seq = []
            for mx in MAX_MOVE_VALUES:
                sc, _ = sweep_results[split_name][(lb, mx)]
                pf_seq.append(sc["pf"] if sc else None)

            # Check monotonicity: count direction changes
            valid = [(mx, pf) for mx, pf in zip(MAX_MOVE_VALUES, pf_seq) if pf is not None]
            if len(valid) < 3:
                print(f"    Lookback {lb:>2}: insufficient data")
                continue

            diffs = []
            for k in range(1, len(valid)):
                diffs.append(valid[k][1] - valid[k - 1][1])

            sign_changes = 0
            for k in range(1, len(diffs)):
                if diffs[k] * diffs[k - 1] < 0:
                    sign_changes += 1

            pf_str = " -> ".join(f"{pf:.3f}" for _, pf in valid)
            if sign_changes == 0:
                verdict = "MONOTONIC"
            elif sign_changes <= 1:
                verdict = "MOSTLY SMOOTH"
            else:
                verdict = f"NON-MONOTONIC ({sign_changes} reversals)"
            print(f"    Lookback {lb:>2}: {pf_str}  [{verdict}]")

    # --- Detailed summary table ---
    print(f"\n{'='*120}")
    print(f"{'Lookback':>8} {'MaxMove':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} "
          f"{'PF':>7} {'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} {'Blocked':>8}")
    print(f"{'-'*120}")

    # Baseline rows first
    for split_name in ["FULL", "IS", "OOS"]:
        sc = baseline_scores[split_name]
        if sc:
            print(f"{'BASE':>8} {'---':>8} {split_name:>5} {sc['count']:>7} "
                  f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                  f"{sc['net_dollar']:>+10.2f} {sc['max_dd_dollar']:>9.2f} "
                  f"{sc['sharpe']:>7.3f} {'---':>8}")
    print(f"{'-'*120}")

    for lb in LOOKBACK_VALUES:
        for mx in MAX_MOVE_VALUES:
            for split_name in ["FULL", "IS", "OOS"]:
                sc, _ = sweep_results[split_name][(lb, mx)]
                bl = baseline_scores[split_name]
                bl_count = bl["count"] if bl else 0

                if sc is None:
                    print(f"{lb:>8} {mx:>7}pt {split_name:>5}   NO TRADES")
                    continue

                blocked = bl_count - sc["count"]
                print(f"{lb:>8} {mx:>7}pt {split_name:>5} {sc['count']:>7} "
                      f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                      f"{sc['net_dollar']:>+10.2f} {sc['max_dd_dollar']:>9.2f} "
                      f"{sc['sharpe']:>7.3f} {blocked:>8}")

    # --- Find best config by OOS Sharpe ---
    best_oos_sharpe = -999
    best_config = None

    for lb in LOOKBACK_VALUES:
        for mx in MAX_MOVE_VALUES:
            sc_oos, _ = sweep_results["OOS"][(lb, mx)]
            if sc_oos and sc_oos["count"] >= 30 and sc_oos["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = sc_oos["sharpe"]
                best_config = (lb, mx)

    bl_oos_sharpe = baseline_scores["OOS"]["sharpe"] if baseline_scores["OOS"] else 0.0

    print(f"\n{'='*80}")
    if best_config and best_oos_sharpe > bl_oos_sharpe:
        lb_best, mx_best = best_config
        sc_full, _ = sweep_results["FULL"][best_config]
        sc_is, _ = sweep_results["IS"][best_config]
        sc_oos, _ = sweep_results["OOS"][best_config]
        print(f"BEST OOS SHARPE: Lookback={lb_best}, MaxMove={mx_best}pt "
              f"(OOS Sharpe {best_oos_sharpe:.3f} vs baseline {bl_oos_sharpe:.3f})")
        print(f"  FULL: PF {sc_full['pf']:.3f}, Sharpe {sc_full['sharpe']:.3f}, "
              f"Net ${sc_full['net_dollar']:+.2f}, {sc_full['count']} trades")
        print(f"  IS:   PF {sc_is['pf']:.3f}, Sharpe {sc_is['sharpe']:.3f}, "
              f"Net ${sc_is['net_dollar']:+.2f}, {sc_is['count']} trades")
        print(f"  OOS:  PF {sc_oos['pf']:.3f}, Sharpe {sc_oos['sharpe']:.3f}, "
              f"Net ${sc_oos['net_dollar']:+.2f}, {sc_oos['count']} trades")

        # Save best result
        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
            "rally_exhaustion_lookback": lb_best,
            "rally_exhaustion_max_move": mx_best,
        }

        print(f"\nSaving trade CSVs for best config (Lookback={lb_best}, MaxMove={mx_best})...")
        for split_name, _, dr, _, _ in splits:
            sc, trades = sweep_results[split_name][best_config]
            if trades:
                save_backtest(
                    trades, strategy="MES_V2_RALLY_EXHAUST", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"rally_exhaustion_lookback={lb_best}_max_move={mx_best}",
                )
    else:
        print(f"NO CONFIG BEATS BASELINE (OOS Sharpe {bl_oos_sharpe:.3f})")
        print("Rally exhaustion gate does not improve MES v2.")
    print(f"{'='*80}")

    print("\nDone.")


if __name__ == "__main__":
    main()
