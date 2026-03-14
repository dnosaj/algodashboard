"""
Round 3 Study 7 — Combined Best Filters
=========================================
Tests combinations of best-passing configs from Studies 1-6.
Run AFTER all individual studies complete.

Usage:
    1. Run Studies 1-6 first
    2. Fill in BEST_CONFIGS below with passing thresholds
    3. python3 sr_round3_combined.py
"""

import sys
from pathlib import Path
from itertools import combinations

import numpy as np

from sr_common import (
    STRATEGIES, prepare_data, run_sweep, compute_atr_wilder,
    compute_et_minutes, NY_OPEN_ET, NY_CLOSE_ET,
)

# Import individual study gate builders
from sr_prior_day_levels_sweep import compute_rth_volume_profile, build_prior_day_level_gate
from sr_vwap_zscore_sweep import compute_vwap_zscore, build_vwap_zscore_gate
from sr_squeeze_gate_sweep import compute_squeeze, build_squeeze_gate
from sr_leledc_exhaustion_sweep import compute_leledc_exhaustion, build_leledc_gate
from sr_initial_balance_sweep import compute_initial_balance, build_ib_gate
from sr_intraday_pivots_sweep import (
    compute_pivots_fast, score_pivots, precompute_nearest_levels, build_pivot_gate,
)
from v10_test_common import compute_prior_day_levels

# ============================================================================
# BEST CONFIGS FROM INDIVIDUAL STUDIES
# Set to None if the study had zero passing configs
# ============================================================================

BEST_CONFIGS = {
    "prior_day": None,      # FAIL — no all-3 passing config
    "vwap_z": None,         # FAIL — vScalpA OOS always degrades
    "squeeze": None,        # FAIL — hurts all strategies
    "leledc": {"maj_qual": 9, "persistence": 1},  # ALL PASS: vScalpA STRONG, vScalpB+MES MARGINAL
    "ib": {"ib_period": 60, "buffer_pts": 5},      # ALL PASS: vScalpA STRONG, vScalpB+MES MARGINAL
    "pivots": None,         # FAIL — too aggressive, blocks 70-99% of trades
}


def build_individual_gate(filter_name, config, instruments):
    """Build gate arrays for a single filter across all instruments."""
    gates = {}
    bin_widths = {"MNQ": 2, "MES": 5}

    for inst, df in instruments.items():
        times = df.index
        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values
        volumes = df["Volume"].values
        et_mins = compute_et_minutes(times)

        if filter_name == "prior_day":
            prev_high, prev_low, _ = compute_prior_day_levels(times, highs, lows, closes)
            vpoc, vah, val = compute_rth_volume_profile(
                times, closes, volumes, et_mins, bin_widths[inst])
            gate = build_prior_day_level_gate(
                closes, prev_high, prev_low, vpoc, vah, val, config["buffer_pts"])

        elif filter_name == "vwap_z":
            vwap = df["VWAP"].values
            z_score = compute_vwap_zscore(closes, vwap, times)
            gate = build_vwap_zscore_gate(z_score, config["max_z"])

        elif filter_name == "squeeze":
            squeeze_on = compute_squeeze(closes, highs, lows, kc_mult=config["kc_mult"])
            gate = build_squeeze_gate(squeeze_on, min_bars_off=config["min_bars_off"])

        elif filter_name == "leledc":
            bull_ex, bear_ex = compute_leledc_exhaustion(closes, config["maj_qual"])
            gate = build_leledc_gate(bull_ex, bear_ex, config["persistence"])

        elif filter_name == "ib":
            ib_high, ib_low = compute_initial_balance(
                times, highs, lows, et_mins, config["ib_period"])
            gate = build_ib_gate(closes, ib_high, ib_low, config["buffer_pts"])

        elif filter_name == "pivots":
            windows = [10, 20, 30, 50]
            atr = compute_atr_wilder(highs, lows, closes, period=14)
            all_ph, all_pl = {}, {}
            for w in windows:
                ph, pl = compute_pivots_fast(highs, lows, w)
                all_ph[w] = ph
                all_pl[w] = pl
            scored_h, scored_l = score_pivots(all_ph, all_pl, atr)
            n = len(closes)
            res_dist, sup_dist = precompute_nearest_levels(
                n, scored_h, scored_l, closes, config["min_score"])
            gate = build_pivot_gate(res_dist, sup_dist, config["buffer_pts"])

        gates[inst] = gate
    return gates


def main():
    instruments, split_arrays, split_indices = prepare_data()

    # Collect active filters (non-None configs)
    active_filters = {k: v for k, v in BEST_CONFIGS.items() if v is not None}

    if not active_filters:
        print("\nNo passing configs found in BEST_CONFIGS. Nothing to combine.")
        print("Fill in BEST_CONFIGS dict with results from Studies 1-6 and re-run.")
        return

    print(f"\nActive filters ({len(active_filters)}):")
    for name, cfg in active_filters.items():
        print(f"  {name}: {cfg}")

    # Pre-compute individual gate arrays per instrument
    individual_gates = {}  # filter_name -> {instrument: gate_array}
    for name, cfg in active_filters.items():
        print(f"  Building {name} gates...")
        individual_gates[name] = build_individual_gate(name, cfg, instruments)

    # Build sweep configs: individual + pairwise + triple + all
    filter_names = sorted(active_filters.keys())
    configs = [{"label": "None"}]

    # Individual
    for name in filter_names:
        configs.append({"label": name, "filters": [name]})

    # Pairwise
    if len(filter_names) >= 2:
        for combo in combinations(filter_names, 2):
            label = "+".join(combo)
            configs.append({"label": label, "filters": list(combo)})

    # Triple
    if len(filter_names) >= 3:
        for combo in combinations(filter_names, 3):
            label = "+".join(combo)
            configs.append({"label": label, "filters": list(combo)})

    # Quadruple+
    for size in range(4, len(filter_names) + 1):
        for combo in combinations(filter_names, size):
            label = "+".join(combo)
            configs.append({"label": label, "filters": list(combo)})

    def build_combined_gates(config, instruments):
        """AND together the pre-computed individual gates."""
        filter_list = config["filters"]
        gates = {}
        for inst in instruments:
            combined = None
            for fname in filter_list:
                g = individual_gates[fname][inst]
                if combined is None:
                    combined = g.copy()
                else:
                    combined = combined & g
            gates[inst] = combined
        return gates

    run_sweep("Combined Round 3", STRATEGIES, split_arrays, split_indices,
              instruments, build_combined_gates, configs,
              script_name="sr_round3_combined.py")
    print("\nDone.")


if __name__ == "__main__":
    main()
