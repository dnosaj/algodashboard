"""
Cooldown Bar Sweep — IS/OOS for all active strategies
======================================================
Tests CD = [0, 5, 8, 10, 12, 15, 18, 20, 25, 30] per strategy.

Context: Gates (leledc, prior-day level, ADR) now filter bad entries,
so cooldown may be doing redundant work and blocking good re-entries.
Current values: MNQ=20, MES=25.
"""

import numpy as np
from sr_common import (
    STRATEGIES,
    prepare_data,
    run_single,
    assess_pass_fail,
)
from v10_test_common import score_trades


CD_VALUES = [0, 5, 8, 10, 12, 15, 18, 20, 25, 30]


def main():
    instruments, split_arrays, split_indices = prepare_data()

    print(f"\n{'='*130}")
    print("COOLDOWN BAR SWEEP (all strategies, IS/OOS)")
    print(f"{'='*130}")

    for strat in STRATEGIES:
        current_cd = strat["cooldown"]

        print(f"\n{'─'*130}")
        print(f"  {strat['name']} (instrument={strat['instrument']}, current CD={current_cd})")
        print(f"{'─'*130}")
        print(f"  {'CD':>4} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Net$':>10} "
              f"{'MaxDD':>9} {'Sharpe':>7} {'Avg$/tr':>9}")
        print(f"  {'-'*80}")

        baselines = {}
        all_results = {}

        for cd in CD_VALUES:
            strat_mod = dict(strat)
            strat_mod["cooldown"] = cd

            for split_name in ["IS", "OOS"]:
                arrays = split_arrays[(strat["name"], split_name)]
                trades = run_single(strat_mod, *arrays, entry_gate=None)
                sc = score_trades(trades, commission_per_side=strat["commission"],
                                  dollar_per_pt=strat["dollar_per_pt"])

                key = (cd, split_name)
                all_results[key] = sc

                if cd == current_cd:
                    baselines[split_name] = sc

                if sc is None:
                    print(f"  {cd:>4} {split_name:>5}   NO TRADES")
                    continue

                avg_pnl = sc["net_dollar"] / sc["count"] if sc["count"] > 0 else 0
                marker = " <<<" if cd == current_cd else ""

                print(f"  {cd:>4} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
                      f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} {sc['max_dd_dollar']:>9.2f} "
                      f"{sc['sharpe']:>7.3f} {avg_pnl:>+9.2f}{marker}")

        # Summary for this strategy
        print(f"\n  {'CD':>4} {'IS PF':>7} {'OOS PF':>7} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS Net$':>10} {'OOS Trades':>11} {'Verdict':>15}")
        print(f"  {'-'*85}")
        for cd in CD_VALUES:
            sc_is = all_results.get((cd, "IS"))
            sc_oos = all_results.get((cd, "OOS"))
            if not sc_is or not sc_oos:
                continue

            verdict = ""
            if baselines.get("IS") and baselines.get("OOS"):
                v, detail = assess_pass_fail(sc_is, sc_oos, baselines["IS"], baselines["OOS"])
                verdict = v

            marker = " <<<" if cd == current_cd else ""
            print(f"  {cd:>4} {sc_is['pf']:>7.3f} {sc_oos['pf']:>7.3f} "
                  f"{sc_is['sharpe']:>10.3f} {sc_oos['sharpe']:>11.3f} "
                  f"{sc_oos['net_dollar']:>+10.2f} {sc_oos['count']:>11} "
                  f"{verdict:>15}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
