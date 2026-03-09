"""
ATR Ratio Entry Gate Sweep — Test #2: Relative ATR Expansion/Contraction
=========================================================================
ATR ratio = ATR(fast) / ATR(slow)
  > 1.0 = volatility expanding (fast ATR higher than slow)
  < 1.0 = volatility contracting

Sweep:
  - Fast/slow ATR pairs: (5,20), (7,20), (7,30), (10,30), (10,50)
  - Timeframes: 1-min bars, 5-min bars (mapped back to 1-min)
  - Expansion gates: ratio > threshold  [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
  - Contraction gates: ratio < threshold [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  - All 3 strategies: V15, vScalpB (MNQ), MES_V2 (MES)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Path setup ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR))
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)
from generate_session import run_backtest_tp_exit, compute_mfe_mae
from sr_common import (
    STRATEGIES,
    prepare_data,
    run_single,
    slice_gate,
    assess_pass_fail,
    compute_atr_wilder,
)

# ---------------------------------------------------------------------------
# ATR Ratio computation
# ---------------------------------------------------------------------------

def compute_atr_ratio_1min(highs, lows, closes, fast_period, slow_period):
    """Compute ATR(fast) / ATR(slow) on 1-min bars.

    Returns numpy array same length as input. NaN where either ATR is undefined.
    """
    atr_fast = compute_atr_wilder(highs, lows, closes, period=fast_period)
    atr_slow = compute_atr_wilder(highs, lows, closes, period=slow_period)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(atr_slow > 0, atr_fast / atr_slow, np.nan)
    return ratio


def compute_atr_ratio_5min(df_1min, fast_period, slow_period):
    """Compute ATR(fast)/ATR(slow) on 5-min bars, mapped back to 1-min.

    Args:
        df_1min: DataFrame with Open, High, Low, Close, SM_Net columns and
                 DatetimeIndex (1-min bars).
        fast_period: fast ATR period (in 5-min bars).
        slow_period: slow ATR period (in 5-min bars).

    Returns:
        numpy array same length as df_1min. Each 1-min bar gets the ratio from
        its enclosing 5-min bar (forward-fill).
    """
    df_5m = df_1min.resample('5min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
    }).dropna(subset=['Open'])

    atr_fast = compute_atr_wilder(
        df_5m['High'].values, df_5m['Low'].values, df_5m['Close'].values,
        period=fast_period,
    )
    atr_slow = compute_atr_wilder(
        df_5m['High'].values, df_5m['Low'].values, df_5m['Close'].values,
        period=slow_period,
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_5m = np.where(atr_slow > 0, atr_fast / atr_slow, np.nan)

    # Map back to 1-min: assign each 5-min ratio to a Series, reindex to 1-min
    ratio_series = pd.Series(ratio_5m, index=df_5m.index)
    ratio_1min = ratio_series.reindex(df_1min.index, method='ffill').values
    return ratio_1min


# ---------------------------------------------------------------------------
# Gate builder
# ---------------------------------------------------------------------------

def build_expansion_gate(ratio, threshold):
    """True (allow entry) when ratio > threshold."""
    gate = np.ones(len(ratio), dtype=bool)
    gate[np.isnan(ratio)] = True  # fail-open on NaN
    gate[~np.isnan(ratio) & (ratio <= threshold)] = False
    return gate


def build_contraction_gate(ratio, threshold):
    """True (allow entry) when ratio < threshold."""
    gate = np.ones(len(ratio), dtype=bool)
    gate[np.isnan(ratio)] = True  # fail-open on NaN
    gate[~np.isnan(ratio) & (ratio >= threshold)] = False
    return gate


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    print("=" * 130)
    print("ATR RATIO ENTRY GATE SWEEP — Test #2: Relative ATR Expansion/Contraction")
    print("=" * 130)

    # Load data and prepare arrays
    instruments, split_arrays, split_indices = prepare_data()

    # --- Precompute all ATR ratios ---
    ATR_PAIRS = [(5, 20), (7, 20), (7, 30), (10, 30), (10, 50)]
    TIMEFRAMES = ["1min", "5min"]

    # ratios[instrument][(fast, slow, timeframe)] = full-length ratio array
    ratios = {}
    for inst_name, df in instruments.items():
        ratios[inst_name] = {}
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        for fast, slow in ATR_PAIRS:
            # 1-min
            r_1m = compute_atr_ratio_1min(highs, lows, closes, fast, slow)
            ratios[inst_name][(fast, slow, "1min")] = r_1m
            # 5-min (mapped back to 1-min)
            r_5m = compute_atr_ratio_5min(df, fast, slow)
            ratios[inst_name][(fast, slow, "5min")] = r_5m
        print(f"  Precomputed ATR ratios for {inst_name}: {len(ATR_PAIRS) * 2} combos")

    # --- Baselines ---
    print("\nRunning baselines...")
    baselines = {}
    for strat in STRATEGIES:
        for split_name in ["FULL", "IS", "OOS"]:
            arrays = split_arrays[(strat["name"], split_name)]
            trades = run_single(strat, *arrays, entry_gate=None)
            sc = score_trades(trades, commission_per_side=strat["commission"],
                              dollar_per_pt=strat["dollar_per_pt"])
            baselines[(strat["name"], split_name)] = sc
            if split_name == "FULL" and sc:
                print(f"  {strat['name']:>8} baseline: {sc['count']} trades, "
                      f"WR {sc['win_rate']}%, PF {sc['pf']}, "
                      f"Sharpe {sc['sharpe']}, ${sc['net_dollar']:+.0f}")

    # --- Build sweep configs ---
    EXP_THRESHOLDS = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    CON_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    configs = []
    for fast, slow in ATR_PAIRS:
        for tf in TIMEFRAMES:
            for thr in EXP_THRESHOLDS:
                configs.append({
                    "label": f"exp_{fast}/{slow}_{tf}_>{thr}",
                    "fast": fast, "slow": slow, "tf": tf,
                    "direction": "expansion", "threshold": thr,
                })
            for thr in CON_THRESHOLDS:
                configs.append({
                    "label": f"con_{fast}/{slow}_{tf}_<{thr}",
                    "fast": fast, "slow": slow, "tf": tf,
                    "direction": "contraction", "threshold": thr,
                })

    total_runs = len(configs) * len(STRATEGIES) * 3  # 3 splits
    print(f"\nSweep: {len(configs)} configs x {len(STRATEGIES)} strategies x 3 splits = {total_runs} backtests")

    # --- Run sweep (FULL only first for speed, IS/OOS for promising) ---
    full_results = []  # (label, strat_name, sc, trades, config)

    for ci, cfg in enumerate(configs):
        if ci % 20 == 0:
            print(f"  Progress: {ci}/{len(configs)} configs...")

        fast, slow, tf = cfg["fast"], cfg["slow"], cfg["tf"]
        direction = cfg["direction"]
        threshold = cfg["threshold"]
        label = cfg["label"]

        # Build gates for each instrument
        gates_full = {}
        for inst_name in ["MNQ", "MES"]:
            ratio = ratios[inst_name][(fast, slow, tf)]
            if direction == "expansion":
                gate = build_expansion_gate(ratio, threshold)
            else:
                gate = build_contraction_gate(ratio, threshold)
            gates_full[inst_name] = gate

        for strat in STRATEGIES:
            inst = strat["instrument"]
            arrays = split_arrays[(strat["name"], "FULL")]
            gate = gates_full[inst]
            trades = run_single(strat, *arrays, entry_gate=gate)
            sc = score_trades(trades, commission_per_side=strat["commission"],
                              dollar_per_pt=strat["dollar_per_pt"])
            bl = baselines[(strat["name"], "FULL")]
            full_results.append((label, strat["name"], sc, trades, cfg, bl))

    # --- Print full sweep grid ---
    print(f"\n{'='*155}")
    print(f"FULL SWEEP GRID — ALL CONFIGS")
    print(f"{'='*155}")
    print(f"{'Config':>35} {'Strategy':>8} {'Trades':>7} {'WR%':>6} "
          f"{'PF':>7} {'Net$':>10} {'Sharpe':>7} "
          f"{'Blocked':>8} {'dPF%':>7} {'dSharpe':>8}")
    print(f"{'-'*155}")

    for label, strat_name, sc, trades, cfg, bl in full_results:
        if sc is None:
            print(f"{label:>35} {strat_name:>8}   NO TRADES")
            continue
        blocked = bl["count"] - sc["count"] if bl else 0
        dpf = ((sc["pf"] - bl["pf"]) / bl["pf"] * 100) if (bl and bl["pf"] > 0) else 0
        dsharpe = sc["sharpe"] - bl["sharpe"] if bl else 0
        print(f"{label:>35} {strat_name:>8} {sc['count']:>7} "
              f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['sharpe']:>7.3f} "
              f"{blocked:>8} {dpf:>+6.1f}% {dsharpe:>+7.3f}")

    # --- Top 10 per strategy by Sharpe (with baseline comparison) ---
    print(f"\n{'='*155}")
    print(f"TOP 10 PER STRATEGY BY SHARPE (FULL period)")
    print(f"{'='*155}")

    for strat in STRATEGIES:
        sname = strat["name"]
        bl = baselines[(sname, "FULL")]
        print(f"\n--- {sname} (baseline: {bl['count']} trades, WR {bl['win_rate']}%, "
              f"PF {bl['pf']}, Sharpe {bl['sharpe']}, ${bl['net_dollar']:+.0f}) ---")
        print(f"{'Rank':>4} {'Config':>35} {'Trades':>7} {'WR%':>6} "
              f"{'PF':>7} {'Net$':>10} {'Sharpe':>7} "
              f"{'Blocked':>8} {'dPF%':>7} {'dSharpe':>8}")

        strat_results = [(l, sc, t, c, b) for l, sn, sc, t, c, b in full_results
                         if sn == sname and sc is not None]
        strat_results.sort(key=lambda x: x[1]["sharpe"], reverse=True)

        for rank, (label, sc, trades, cfg, bl_ref) in enumerate(strat_results[:10], 1):
            blocked = bl["count"] - sc["count"]
            dpf = ((sc["pf"] - bl["pf"]) / bl["pf"] * 100) if bl["pf"] > 0 else 0
            dsharpe = sc["sharpe"] - bl["sharpe"]
            print(f"{rank:>4} {label:>35} {sc['count']:>7} "
                  f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                  f"{sc['sharpe']:>7.3f} "
                  f"{blocked:>8} {dpf:>+6.1f}% {dsharpe:>+7.3f}")

    # --- Identify promising configs: PF improves >5% on FULL ---
    promising = set()
    for label, strat_name, sc, trades, cfg, bl in full_results:
        if sc is None or bl is None:
            continue
        if bl["pf"] > 0:
            dpf = (sc["pf"] - bl["pf"]) / bl["pf"] * 100
            if dpf > 5:
                promising.add(label)

    print(f"\n{'='*130}")
    print(f"IS/OOS VALIDATION — {len(promising)} configs with >5% PF improvement on FULL")
    print(f"{'='*130}")

    if not promising:
        print("  No configs exceeded +5% PF threshold. Sweep complete.")
    else:
        # Run IS/OOS for promising configs
        isoos_results = []

        for label in sorted(promising):
            # Recover the config
            cfg = None
            for l, sn, sc, t, c, b in full_results:
                if l == label:
                    cfg = c
                    break
            if cfg is None:
                continue

            fast, slow, tf = cfg["fast"], cfg["slow"], cfg["tf"]
            direction = cfg["direction"]
            threshold = cfg["threshold"]

            # Build gates
            gates_full = {}
            for inst_name in ["MNQ", "MES"]:
                ratio = ratios[inst_name][(fast, slow, tf)]
                if direction == "expansion":
                    gate = build_expansion_gate(ratio, threshold)
                else:
                    gate = build_contraction_gate(ratio, threshold)
                gates_full[inst_name] = gate

            for strat in STRATEGIES:
                inst = strat["instrument"]
                for split_name in ["IS", "OOS"]:
                    arrays = split_arrays[(strat["name"], split_name)]
                    gate = slice_gate(gates_full[inst], inst, split_name, split_indices)
                    trades = run_single(strat, *arrays, entry_gate=gate)
                    sc = score_trades(trades, commission_per_side=strat["commission"],
                                      dollar_per_pt=strat["dollar_per_pt"])
                    bl = baselines[(strat["name"], split_name)]
                    isoos_results.append((label, strat["name"], split_name, sc, trades, bl))

        # Print IS/OOS table
        print(f"\n{'Config':>35} {'Strategy':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} "
              f"{'PF':>7} {'Net$':>10} {'Sharpe':>7} "
              f"{'Blocked':>8} {'dPF%':>7} {'Verdict':>15}")
        print(f"{'-'*155}")

        for label, strat_name, split_name, sc, trades, bl in isoos_results:
            if sc is None:
                print(f"{label:>35} {strat_name:>8} {split_name:>5}   NO TRADES")
                continue

            blocked = bl["count"] - sc["count"] if bl else 0
            dpf = ((sc["pf"] - bl["pf"]) / bl["pf"] * 100) if (bl and bl["pf"] > 0) else 0

            # Verdict for OOS rows
            verdict = ""
            if split_name == "OOS":
                sc_is = None
                for l2, s2, sp2, sc2, _, _ in isoos_results:
                    if l2 == label and s2 == strat_name and sp2 == "IS":
                        sc_is = sc2
                        break
                if sc_is is not None:
                    bl_is = baselines[(strat_name, "IS")]
                    bl_oos = baselines[(strat_name, "OOS")]
                    v, _ = assess_pass_fail(sc_is, sc, bl_is, bl_oos)
                    verdict = v

            print(f"{label:>35} {strat_name:>8} {split_name:>5} {sc['count']:>7} "
                  f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                  f"{sc['sharpe']:>7.3f} "
                  f"{blocked:>8} {dpf:>+6.1f}% {verdict:>15}")

        # Per-strategy verdict summary for promising configs
        print(f"\n{'='*100}")
        print("PER-STRATEGY VERDICTS (promising configs)")
        print(f"{'='*100}")

        for label in sorted(promising):
            verdicts = {}
            for strat in STRATEGIES:
                sc_is = sc_oos = None
                for l2, s2, sp2, sc2, _, _ in isoos_results:
                    if l2 == label and s2 == strat["name"]:
                        if sp2 == "IS":
                            sc_is = sc2
                        elif sp2 == "OOS":
                            sc_oos = sc2
                bl_is = baselines[(strat["name"], "IS")]
                bl_oos = baselines[(strat["name"], "OOS")]
                v, detail = assess_pass_fail(sc_is, sc_oos, bl_is, bl_oos)
                verdicts[strat["name"]] = f"{v} ({detail})"
            all_pass = all("PASS" in v for v in verdicts.values())
            marker = " <<<" if all_pass else ""
            print(f"\n  {label}:{marker}")
            for sname, v in verdicts.items():
                print(f"    {sname:>8}: {v}")

    # --- Summary recommendation ---
    print(f"\n{'='*130}")
    print("SUMMARY RECOMMENDATION")
    print(f"{'='*130}")

    # Find best config per strategy (by Sharpe, among those with >5% PF lift)
    for strat in STRATEGIES:
        sname = strat["name"]
        bl = baselines[(sname, "FULL")]
        best_label = None
        best_sharpe = bl["sharpe"] if bl else 0

        for label, sn, sc, trades, cfg, bl_ref in full_results:
            if sn != sname or sc is None:
                continue
            if bl["pf"] > 0:
                dpf = (sc["pf"] - bl["pf"]) / bl["pf"] * 100
                if dpf > 5 and sc["sharpe"] > best_sharpe:
                    best_sharpe = sc["sharpe"]
                    best_label = label

        if best_label:
            # Find the FULL score for it
            for label, sn, sc, trades, cfg, bl_ref in full_results:
                if label == best_label and sn == sname:
                    dpf = (sc["pf"] - bl["pf"]) / bl["pf"] * 100
                    dsharpe = sc["sharpe"] - bl["sharpe"]
                    print(f"  {sname:>8}: BEST = {best_label}")
                    print(f"           Trades {sc['count']} (blocked {bl['count']-sc['count']}), "
                          f"WR {sc['win_rate']}%, PF {sc['pf']} ({dpf:+.1f}%), "
                          f"Sharpe {sc['sharpe']} ({dsharpe:+.3f}), ${sc['net_dollar']:+.0f}")
                    break
        else:
            print(f"  {sname:>8}: No config improved PF >5% over baseline")

    print("\nDone.")


if __name__ == "__main__":
    main()
