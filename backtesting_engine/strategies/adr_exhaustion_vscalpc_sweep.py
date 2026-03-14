"""
ADR Exhaustion Entry Gate Sweep — vScalpC (Partial Exit Engine)
================================================================
Same 4-layer study as the main sweep, but uses the partial exit
backtest engine for vScalpC (2-contract, TP1=7/TP2=25/SL=40/BE45).

Layer 4 tests both TP1=7 and TP2=25 as remaining range thresholds.

Usage:
    cd backtesting_engine && python3 strategies/adr_exhaustion_vscalpc_sweep.py
"""

import numpy as np

from htf_common import (
    prepare_vscalpc_data,
    slice_gate_partial,
    VSCALPC_TP1, VSCALPC_TP2, VSCALPC_SL, VSCALPC_BE_TIME, VSCALPC_SL_TO_BE,
)

from htf_filter_atr_sweep import (
    run_vscalpc,
    print_comparison,
    assess_pass,
)

from v10_test_common import resample_to_5min, map_5min_rsi_to_1min

from adr_common import (
    compute_session_tracking,
    compute_adr,
    build_range_gate,
    build_directional_gate,
    build_combined_gate,
    build_remaining_range_gate,
)

from vscalpc_partial_exit_sweep import score_partial_trades


def main():
    print("=" * 120)
    print("vScalpC ADR Exhaustion Sweep (partial exit engine)")
    print("=" * 120)

    # --- Load data ---
    df, rsi_curr, rsi_prev, is_len = prepare_vscalpc_data()

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    sm = df['SM_Net'].values
    times = df.index

    # --- IS/OOS splits ---
    df_is = df.iloc[:is_len]
    df_oos = df.iloc[is_len:]

    df_5m_is = resample_to_5min(df_is)
    rsi_is_curr, rsi_is_prev = map_5min_rsi_to_1min(
        df_is.index.values, df_5m_is.index.values, df_5m_is['Close'].values,
        rsi_len=8,
    )
    df_5m_oos = resample_to_5min(df_oos)
    rsi_oos_curr, rsi_oos_prev = map_5min_rsi_to_1min(
        df_oos.index.values, df_5m_oos.index.values, df_5m_oos['Close'].values,
        rsi_len=8,
    )

    # --- Baselines ---
    print("\nRunning baselines...")
    bl_full = score_partial_trades(run_vscalpc(
        opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev))
    bl_is = score_partial_trades(run_vscalpc(
        df_is['Open'].values, df_is['High'].values, df_is['Low'].values,
        df_is['Close'].values, df_is['SM_Net'].values, df_is.index,
        rsi_is_curr, rsi_is_prev))
    bl_oos = score_partial_trades(run_vscalpc(
        df_oos['Open'].values, df_oos['High'].values, df_oos['Low'].values,
        df_oos['Close'].values, df_oos['SM_Net'].values, df_oos.index,
        rsi_oos_curr, rsi_oos_prev))

    print_comparison("FULL baseline", bl_full, bl_full)
    print_comparison("IS   baseline", bl_is, bl_is)
    print_comparison("OOS  baseline", bl_oos, bl_oos)

    # --- Precompute session tracking and ADR ---
    print("\nComputing session tracking and ADR...")
    sess = compute_session_tracking(df)
    lookbacks = [5, 10, 14, 20]
    adr_data = {lb: compute_adr(df, lookback_days=lb) for lb in lookbacks}

    # Helper to run a gated config
    def run_gated(gate_full):
        gate_is = slice_gate_partial(gate_full, is_len, "IS")
        gate_oos = slice_gate_partial(gate_full, is_len, "OOS")

        sc_full = score_partial_trades(run_vscalpc(
            opens, highs, lows, closes, sm, times,
            rsi_curr, rsi_prev, entry_gate=gate_full))
        sc_is = score_partial_trades(run_vscalpc(
            df_is['Open'].values, df_is['High'].values, df_is['Low'].values,
            df_is['Close'].values, df_is['SM_Net'].values, df_is.index,
            rsi_is_curr, rsi_is_prev, entry_gate=gate_is))
        sc_oos = score_partial_trades(run_vscalpc(
            df_oos['Open'].values, df_oos['High'].values, df_oos['Low'].values,
            df_oos['Close'].values, df_oos['SM_Net'].values, df_oos.index,
            rsi_oos_curr, rsi_oos_prev, entry_gate=gate_oos))
        return sc_full, sc_is, sc_oos

    all_results = []

    def test_config(label, gate_full):
        sc_full, sc_is, sc_oos = run_gated(gate_full)
        verdict, detail = assess_pass(sc_is, sc_oos, bl_is, bl_oos)
        marker = " <<<" if "PASS" in verdict else ""
        print(f"\n  {label}  [{verdict}]{marker}")
        print_comparison("FULL", sc_full, bl_full, indent="    ")
        print_comparison("IS  ", sc_is, bl_is, indent="    ")
        print_comparison("OOS ", sc_oos, bl_oos, indent="    ")
        all_results.append({
            'label': label, 'sc_full': sc_full, 'sc_is': sc_is,
            'sc_oos': sc_oos, 'verdict': verdict, 'detail': detail,
        })
        return sc_full, sc_is, sc_oos, verdict

    # ========================================================================
    # LAYER 1: Basic Range Gate
    # ========================================================================
    print(f"\n{'='*120}")
    print("LAYER 1: Basic ADR Range Gate — block when today_range / ADR >= threshold")
    print(f"{'='*120}")

    range_thresholds = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    best_l1 = {"label": None, "oos_sharpe": -999}

    for lb in lookbacks:
        for thr in range_thresholds:
            gate = build_range_gate(sess['today_range'], adr_data[lb], thr)
            label = f"rng_lb{lb}_t{thr}"
            _, _, sc_oos, verdict = test_config(label, gate)
            if sc_oos and sc_oos['sharpe'] > best_l1['oos_sharpe']:
                best_l1 = {"label": label, "oos_sharpe": sc_oos['sharpe'],
                           "lb": lb, "thr": thr}

    # ========================================================================
    # LAYER 2: Directional Gate
    # ========================================================================
    print(f"\n{'='*120}")
    print("LAYER 2: Directional ADR Gate — block longs when rally/ADR >= thr")
    print(f"{'='*120}")

    dir_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_l2 = {"label": None, "oos_sharpe": -999}

    for lb in lookbacks:
        for thr in dir_thresholds:
            gate = build_directional_gate(sess['move_from_open'], adr_data[lb],
                                          sm, thr)
            label = f"dir_lb{lb}_t{thr}"
            _, _, sc_oos, verdict = test_config(label, gate)
            if sc_oos and sc_oos['sharpe'] > best_l2['oos_sharpe']:
                best_l2 = {"label": label, "oos_sharpe": sc_oos['sharpe'],
                           "lb": lb, "thr": thr}

    # ========================================================================
    # LAYER 3: Combined (best L1 + best L2)
    # ========================================================================
    print(f"\n{'='*120}")
    print("LAYER 3: Combined Range + Directional Gate")
    print(f"{'='*120}")

    if best_l1["label"] and best_l2["label"]:
        # Test best L1 x best L2, plus nearby thresholds
        l1_lb = best_l1["lb"]
        l1_thr = best_l1["thr"]
        l2_thr = best_l2["thr"]

        # Best combo
        gate = build_combined_gate(
            sess['today_range'], sess['move_from_open'],
            adr_data[l1_lb], sm, l1_thr, l2_thr)
        test_config(f"comb_lb{l1_lb}_r{l1_thr}_d{l2_thr}", gate)

        # Nearby thresholds
        for r_adj in [-0.1, 0.1]:
            for d_adj in [-0.1, 0.1]:
                r_t = l1_thr + r_adj
                d_t = l2_thr + d_adj
                if r_t < 0.5 or d_t < 0.2:
                    continue
                gate = build_combined_gate(
                    sess['today_range'], sess['move_from_open'],
                    adr_data[l1_lb], sm, r_t, d_t)
                test_config(f"comb_lb{l1_lb}_r{r_t}_d{d_t}", gate)
    else:
        print("  Skipping — no valid L1/L2 configs found.")

    # ========================================================================
    # LAYER 4: Remaining Range vs TP
    # ========================================================================
    print(f"\n{'='*120}")
    print("LAYER 4: Remaining Range Gate — block when ADR - today_range < TP")
    print(f"{'='*120}")

    # Test both TP1 (scalp leg) and TP2 (runner leg) as thresholds
    for tp_label, tp_val in [("TP1=7", VSCALPC_TP1), ("TP2=25", VSCALPC_TP2)]:
        print(f"\n  --- Threshold: {tp_label} ---")
        for lb in lookbacks:
            gate = build_remaining_range_gate(
                sess['today_range'], adr_data[lb], tp_val)
            test_config(f"remain_{tp_label}_lb{lb}", gate)

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print(f"\n{'='*120}")
    print("SUMMARY — All Configs Sorted by OOS Sharpe")
    print(f"{'='*120}")

    all_results.sort(key=lambda r: r['sc_oos']['sharpe'] if r['sc_oos'] else -999,
                     reverse=True)

    print(f"\n  {'Label':>35} {'Verdict':>15} | "
          f"{'FULL N':>6} {'FULL PF':>7} {'FULL$':>9} {'FULL Sh':>7} | "
          f"{'IS PF':>6} {'IS$':>8} {'IS Sh':>6} | "
          f"{'OOS PF':>7} {'OOS$':>8} {'OOS Sh':>7}")
    print(f"  " + "-" * 140)

    # Baseline
    print(f"  {'BASELINE':>35} {'---':>15} | "
          f"{bl_full['count']:>6} {bl_full['pf']:>7.3f} ${bl_full['net_dollar']:>+8.0f} "
          f"{bl_full['sharpe']:>7.3f} | "
          f"{bl_is['pf']:>6.3f} ${bl_is['net_dollar']:>+7.0f} {bl_is['sharpe']:>6.3f} | "
          f"{bl_oos['pf']:>7.3f} ${bl_oos['net_dollar']:>+7.0f} {bl_oos['sharpe']:>7.3f}")

    for r in all_results:
        sc_f = r['sc_full']
        sc_i = r['sc_is']
        sc_o = r['sc_oos']
        if not sc_f or not sc_i or not sc_o:
            continue
        marker = " <<<" if "PASS" in r['verdict'] else ""
        print(f"  {r['label']:>35} {r['verdict']:>15} | "
              f"{sc_f['count']:>6} {sc_f['pf']:>7.3f} ${sc_f['net_dollar']:>+8.0f} "
              f"{sc_f['sharpe']:>7.3f} | "
              f"{sc_i['pf']:>6.3f} ${sc_i['net_dollar']:>+7.0f} {sc_i['sharpe']:>6.3f} | "
              f"{sc_o['pf']:>7.3f} ${sc_o['net_dollar']:>+7.0f} {sc_o['sharpe']:>7.3f}"
              f"{marker}")

    print(f"\n{'='*120}")
    print("SWEEP COMPLETE")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
