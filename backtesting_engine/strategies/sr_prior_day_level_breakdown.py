"""
Prior Day Level Breakdown — Which level types actually help MES_V2?
===================================================================
Runs the prior-day level gate separately for each level type to determine
which ones contribute to the STRONG PASS result:

  1. H/L only (prior-day High + Low)
  2. VPOC only (volume point of control)
  3. VAH/VAL only (value area bounds)
  4. H/L + VPOC (no value area)
  5. VPOC + VAH/VAL (volume profile only, no H/L)
  6. All 5 (original buf5 — should reproduce the STRONG PASS)

buf=5 for all tests (the validated config).
MES_V2 only (the only strategy where prior-day levels passed).
"""

import numpy as np
import pandas as pd

from sr_common import (
    STRATEGIES,
    prepare_data,
    run_single,
    slice_gate,
    assess_pass_fail,
    compute_et_minutes,
    NY_OPEN_ET,
    NY_CLOSE_ET,
)
from sr_prior_day_levels_sweep import (
    compute_rth_volume_profile,
    BIN_WIDTHS,
)
from v10_test_common import compute_prior_day_levels, score_trades


BUFFER_PTS = 5


def build_level_gate(closes, levels_list, buffer_pts):
    """Build boolean gate from a list of level arrays. True=allow, False=block."""
    n = len(closes)
    gate = np.ones(n, dtype=bool)
    for i in range(n):
        c = closes[i]
        for lvl_arr in levels_list:
            if not np.isnan(lvl_arr[i]) and abs(c - lvl_arr[i]) <= buffer_pts:
                gate[i] = False
                break
    return gate


def main():
    instruments, split_arrays, split_indices = prepare_data()

    # Only MES_V2
    strat = next(s for s in STRATEGIES if s["name"] == "MES_V2")
    inst = strat["instrument"]  # "MES"
    df = instruments[inst]

    # Compute all level arrays (full length)
    times = df.index
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values
    et_mins = compute_et_minutes(times)

    prev_high, prev_low, _ = compute_prior_day_levels(times, highs, lows, closes)
    vpoc, vah, val = compute_rth_volume_profile(
        times, closes, volumes, et_mins, BIN_WIDTHS[inst]
    )

    # Define level combinations to test
    combos = [
        ("Baseline (none)", []),
        ("H/L only",        [prev_high, prev_low]),
        ("VPOC only",       [vpoc]),
        ("VAH/VAL only",    [vah, val]),
        ("H/L + VPOC",      [prev_high, prev_low, vpoc]),
        ("VPOC + VA",       [vpoc, vah, val]),
        ("All 5 (original)",[prev_high, prev_low, vpoc, vah, val]),
    ]

    # Also count how many blocks come from each level type
    print("\n" + "=" * 100)
    print(f"PRIOR DAY LEVEL BREAKDOWN — MES_V2 buf={BUFFER_PTS}")
    print("=" * 100)

    # Block attribution: for each bar, which levels would block?
    n = len(closes)
    block_counts = {"high": 0, "low": 0, "vpoc": 0, "vah": 0, "val": 0}
    bars_with_any_block = 0
    for i in range(n):
        c = closes[i]
        blocked_this_bar = False
        for name, arr in [("high", prev_high), ("low", prev_low),
                          ("vpoc", vpoc), ("vah", vah), ("val", val)]:
            if not np.isnan(arr[i]) and abs(c - arr[i]) <= BUFFER_PTS:
                block_counts[name] += 1
                blocked_this_bar = True
        if blocked_this_bar:
            bars_with_any_block += 1

    print(f"\nBlock attribution (bars within {BUFFER_PTS} pts of each level):")
    print(f"  Total bars: {n}")
    print(f"  Bars blocked by ANY level: {bars_with_any_block} ({bars_with_any_block/n*100:.1f}%)")
    for name, count in sorted(block_counts.items(), key=lambda x: -x[1]):
        print(f"    {name:>5}: {count} bars ({count/n*100:.1f}%)")

    # Run backtests
    print(f"\n{'Config':<20} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net$':>10} {'MaxDD':>9} {'Sharpe':>7} {'Blocked':>8} {'dPF%':>7} {'Verdict':>15}")
    print("-" * 120)

    baselines = {}
    all_results = {}

    for combo_name, level_arrays in combos:
        if level_arrays:
            full_gate = build_level_gate(closes, level_arrays, BUFFER_PTS)
        else:
            full_gate = None

        for split_name in ["IS", "OOS"]:
            arrays = split_arrays[(strat["name"], split_name)]
            gate = slice_gate(full_gate, inst, split_name, split_indices) if full_gate is not None else None
            trades = run_single(strat, *arrays, entry_gate=gate)
            sc = score_trades(trades, commission_per_side=strat["commission"],
                              dollar_per_pt=strat["dollar_per_pt"])

            key = (combo_name, split_name)
            all_results[key] = (sc, trades)

            if combo_name == "Baseline (none)":
                baselines[split_name] = sc

            if sc is None:
                print(f"{combo_name:<20} {split_name:>5}   NO TRADES")
                continue

            bl = baselines.get(split_name)
            bl_count = bl["count"] if bl else sc["count"]
            blocked = bl_count - sc["count"]
            dpf = 0.0
            if bl and bl["pf"] > 0:
                dpf = (sc["pf"] - bl["pf"]) / bl["pf"] * 100

            verdict = ""
            if split_name == "OOS" and combo_name != "Baseline (none)":
                sc_is = all_results.get((combo_name, "IS"), (None, None))[0]
                bl_is = baselines.get("IS")
                bl_oos = baselines.get("OOS")
                if sc_is and bl_is and bl_oos:
                    v, _ = assess_pass_fail(sc_is, sc, bl_is, bl_oos)
                    verdict = v

            print(f"{combo_name:<20} {split_name:>5} {sc['count']:>7} "
                  f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                  f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
                  f"{blocked:>8} {dpf:>+6.1f}% {verdict:>15}")

    # Blocked trade analysis: what happens to trades blocked by each level type?
    print(f"\n{'='*100}")
    print("BLOCKED TRADE QUALITY ANALYSIS")
    print("(What would have happened if the blocked trades had fired?)")
    print(f"{'='*100}")

    # For each level type, identify trades that the baseline took but this gate would block
    # Then look at the PnL of those blocked trades
    bl_full_sc, bl_full_trades = None, None
    for split_name in ["FULL"]:
        arrays = split_arrays[(strat["name"], split_name)]
        bl_full_trades = run_single(strat, *arrays, entry_gate=None)
        bl_full_sc = score_trades(bl_full_trades, commission_per_side=strat["commission"],
                                  dollar_per_pt=strat["dollar_per_pt"])

    if bl_full_trades:
        level_names_arr = [
            ("high", prev_high),
            ("low", prev_low),
            ("vpoc", vpoc),
            ("vah", vah),
            ("val", val),
        ]

        print(f"\n{'Level':<10} {'Blocked':>8} {'BL_Win':>8} {'BL_Loss':>8} {'BL_WR%':>8} "
              f"{'BL_Avg_PnL':>12} {'BL_Total$':>12} {'Quality':>12}")
        print("-" * 90)

        for level_name, level_arr in level_names_arr:
            blocked_trades = []
            for t in bl_full_trades:
                entry_bar = t["entry_idx"]
                # Gate uses bar[i-1], entry happens at bar i
                gate_bar = entry_bar - 1 if entry_bar > 0 else 0
                if not np.isnan(level_arr[gate_bar]) and abs(closes[gate_bar] - level_arr[gate_bar]) <= BUFFER_PTS:
                    blocked_trades.append(t)

            if not blocked_trades:
                print(f"{level_name:<10} {'0':>8}")
                continue

            wins = sum(1 for t in blocked_trades if t["pts"] > 0)
            losses = len(blocked_trades) - wins
            wr = wins / len(blocked_trades) * 100
            avg_pnl = np.mean([t["pts"] * strat["dollar_per_pt"] - 2 * strat["commission"] for t in blocked_trades])
            total_pnl = sum(t["pts"] * strat["dollar_per_pt"] - 2 * strat["commission"] for t in blocked_trades)

            # Quality: positive total = gate is HURTING (removing winners)
            #          negative total = gate is HELPING (removing losers)
            quality = "HELPING" if total_pnl < 0 else "HURTING"

            print(f"{level_name:<10} {len(blocked_trades):>8} {wins:>8} {losses:>8} {wr:>7.1f}% "
                  f"${avg_pnl:>+11.2f} ${total_pnl:>+11.2f} {quality:>12}")

    print("\nDone.")


if __name__ == "__main__":
    main()
