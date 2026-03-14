"""
ADR Exhaustion Entry Gate Sweep — V15 + vScalpB + MES_V2
=========================================================
Tests whether blocking entries when the daily range is "used up"
improves strategy performance.

4 layers:
  Layer 1: Basic range gate — block when today_range / ADR >= threshold
  Layer 2: Directional gate — block longs when rally/ADR >= threshold
  Layer 3: Combined (best of L1 + L2)
  Layer 4: Remaining range vs TP — block when ADR - today_range < TP

Uses RTH session tracking (10:00-16:00 ET).
ADR = rolling N-day mean of prior completed RTH daily ranges (no look-ahead).

Usage:
    cd backtesting_engine && python3 strategies/adr_exhaustion_sweep.py
"""

import numpy as np

from sr_common import (
    STRATEGIES,
    prepare_data,
    run_sweep,
    slice_gate,
    assess_pass_fail,
)

from adr_common import (
    compute_session_tracking,
    compute_adr,
    build_range_gate,
    build_directional_gate,
    build_combined_gate,
    build_remaining_range_gate,
)


def main():
    instruments, split_arrays, split_indices = prepare_data()

    # --- Precompute session tracking and ADR for all instruments and lookbacks ---
    print("\nPrecomputing session tracking and ADR...")
    session_data = {}  # inst -> session tracking dict
    adr_data = {}      # (inst, lookback) -> adr array

    lookbacks = [5, 10, 14, 20]

    for inst, df in instruments.items():
        session_data[inst] = compute_session_tracking(df)
        for lb in lookbacks:
            adr_data[(inst, lb)] = compute_adr(df, lookback_days=lb)
        print(f"  {inst}: session tracking + ADR computed")

    # ========================================================================
    # LAYER 1: Basic Range Gate
    # ========================================================================
    print(f"\n{'='*120}")
    print("LAYER 1: Basic ADR Range Gate — block ALL entries when today_range / ADR >= threshold")
    print(f"{'='*120}")

    range_thresholds = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    l1_configs = [{"label": "None"}]
    for lb in lookbacks:
        for thr in range_thresholds:
            l1_configs.append({
                "label": f"rng_lb{lb}_t{thr}",
                "lookback": lb,
                "threshold": thr,
            })

    def build_l1_gates(config, instruments):
        gates = {}
        for inst in instruments:
            tr = session_data[inst]['today_range']
            adr = adr_data[(inst, config["lookback"])]
            gates[inst] = build_range_gate(tr, adr, config["threshold"])
        return gates

    l1_results, l1_baselines = run_sweep(
        "ADR Range Exhaustion (Layer 1)",
        STRATEGIES, split_arrays, split_indices, instruments,
        build_l1_gates, l1_configs,
        script_name="adr_exhaustion_sweep.py",
    )

    # ========================================================================
    # LAYER 2: Directional Gate
    # ========================================================================
    print(f"\n{'='*120}")
    print("LAYER 2: Directional ADR Gate — block longs when rally/ADR >= thr, shorts when selloff/ADR <= -thr")
    print(f"{'='*120}")

    dir_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    l2_configs = [{"label": "None"}]
    for lb in lookbacks:
        for thr in dir_thresholds:
            l2_configs.append({
                "label": f"dir_lb{lb}_t{thr}",
                "lookback": lb,
                "threshold": thr,
            })

    def build_l2_gates(config, instruments):
        gates = {}
        for inst, df in instruments.items():
            mfo = session_data[inst]['move_from_open']
            adr = adr_data[(inst, config["lookback"])]
            sm = df['SM_Net'].values
            gates[inst] = build_directional_gate(mfo, adr, sm, config["threshold"])
        return gates

    l2_results, _ = run_sweep(
        "ADR Directional Exhaustion (Layer 2)",
        STRATEGIES, split_arrays, split_indices, instruments,
        build_l2_gates, l2_configs,
        script_name="adr_exhaustion_sweep.py",
    )

    # ========================================================================
    # Find best L1 and L2 configs per strategy for Layer 3
    # ========================================================================
    def find_best_configs(results, baselines, prefix):
        """Find configs with best portfolio OOS Sharpe that pass."""
        # Group by label, sum OOS net$ across strategies
        label_scores = {}
        for label, strat_name, split_name, sc, trades, bl_count in results:
            if label == "None" or split_name != "OOS" or sc is None:
                continue
            if label not in label_scores:
                label_scores[label] = {"oos_sharpe_sum": 0, "all_pass": True,
                                       "oos_net": 0}
            label_scores[label]["oos_sharpe_sum"] += sc["sharpe"]
            label_scores[label]["oos_net"] += sc["net_dollar"]

            # Check pass/fail
            sc_is = None
            for l2, s2, sp2, sc2, _, _ in results:
                if l2 == label and s2 == strat_name and sp2 == "IS":
                    sc_is = sc2
                    break
            if sc_is:
                bl_is = baselines[(strat_name, "IS")]
                bl_oos = baselines[(strat_name, "OOS")]
                v, _ = assess_pass_fail(sc_is, sc, bl_is, bl_oos)
                if "FAIL" in v:
                    label_scores[label]["all_pass"] = False

        # Sort by OOS net (proxy for best overall improvement)
        sorted_labels = sorted(label_scores.keys(),
                               key=lambda l: label_scores[l]["oos_net"],
                               reverse=True)
        return sorted_labels[:3]  # top 3

    best_l1 = find_best_configs(l1_results, l1_baselines, "rng")
    best_l2 = find_best_configs(l2_results, l1_baselines, "dir")

    print(f"\nBest Layer 1 configs for Layer 3: {best_l1}")
    print(f"Best Layer 2 configs for Layer 3: {best_l2}")

    # ========================================================================
    # LAYER 3: Combined (best L1 + best L2)
    # ========================================================================
    print(f"\n{'='*120}")
    print("LAYER 3: Combined Range + Directional Gate")
    print(f"{'='*120}")

    # Parse best config params
    def parse_label(label):
        parts = label.split("_")
        lb = int(parts[1].replace("lb", ""))
        thr = float(parts[2].replace("t", ""))
        return lb, thr

    l3_configs = [{"label": "None"}]
    for l1_label in best_l1[:2]:
        l1_lb, l1_thr = parse_label(l1_label)
        for l2_label in best_l2[:2]:
            l2_lb, l2_thr = parse_label(l2_label)
            # Use same lookback for both (simpler, and avoids 2 ADR computations)
            # Pick the L1 lookback since range gate is the primary filter
            l3_configs.append({
                "label": f"comb_lb{l1_lb}_r{l1_thr}_d{l2_thr}",
                "lookback": l1_lb,
                "range_threshold": l1_thr,
                "dir_threshold": l2_thr,
            })

    def build_l3_gates(config, instruments):
        gates = {}
        for inst, df in instruments.items():
            tr = session_data[inst]['today_range']
            mfo = session_data[inst]['move_from_open']
            adr = adr_data[(inst, config["lookback"])]
            sm = df['SM_Net'].values
            gates[inst] = build_combined_gate(
                tr, mfo, adr, sm,
                config["range_threshold"], config["dir_threshold"])
        return gates

    if len(l3_configs) > 1:
        run_sweep(
            "ADR Combined (Layer 3)",
            STRATEGIES, split_arrays, split_indices, instruments,
            build_l3_gates, l3_configs,
            script_name="adr_exhaustion_sweep.py",
        )

    # ========================================================================
    # LAYER 4: Remaining Range vs TP
    # ========================================================================
    print(f"\n{'='*120}")
    print("LAYER 4: Remaining Range Gate — block when ADR - today_range < TP")
    print(f"{'='*120}")

    # TP values per instrument: MNQ strategies all TP=5, MES TP=20
    tp_by_inst = {"MNQ": 5, "MES": 20}

    l4_configs = [{"label": "None"}]
    for lb in lookbacks:
        l4_configs.append({
            "label": f"remain_lb{lb}",
            "lookback": lb,
        })

    def build_l4_gates(config, instruments):
        gates = {}
        for inst in instruments:
            tr = session_data[inst]['today_range']
            adr = adr_data[(inst, config["lookback"])]
            tp = tp_by_inst[inst]
            gates[inst] = build_remaining_range_gate(tr, adr, tp)
        return gates

    run_sweep(
        "ADR Remaining Range (Layer 4)",
        STRATEGIES, split_arrays, split_indices, instruments,
        build_l4_gates, l4_configs,
        script_name="adr_exhaustion_sweep.py",
    )

    print("\n" + "=" * 120)
    print("ALL LAYERS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
