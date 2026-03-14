"""
MES v2 Combined Entry/Exit Filter Test
=======================================
Tests combinations of MES v2 entry gates and exit patterns that have been
individually validated by their respective sweep scripts. Run this AFTER
all individual sweeps have completed and thresholds have been identified.

Update the threshold constants at the top with values from individual studies.
Set to None/0 to disable any filter that didn't pass its individual sweep.

Usage:
    python3 mes_v2_combined_filters.py
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

# Gate builders from individual sweep scripts
from mes_v2_sm_slope_sweep import build_sm_slope_gate
from mes_v2_rally_exhaustion_sweep import build_rally_exhaustion_gate
from mes_v2_volume_climax_sweep import build_volume_climax_gate

SCRIPT_NAME = "mes_v2_combined_filters.py"

# ===== THRESHOLDS FROM INDIVIDUAL STUDIES =====
# Update these after reviewing individual sweep results.
# Set to None/0 to disable a filter that didn't pass.

# Entry gates (set to None to skip)
BEST_SM_THRESHOLD = None        # From sm_magnitude_sweep (e.g., 0.05)
BEST_SM_SLOPE = None            # From sm_slope_sweep (e.g., 0.002)
BEST_RALLY_LOOKBACK = None      # From rally_exhaustion_sweep (e.g., 15)
BEST_RALLY_MAX_MOVE = None      # From rally_exhaustion_sweep (e.g., 25)
BEST_VOL_LOOKBACK = None        # From volume_climax_sweep (e.g., 20)
BEST_VOL_MAX_RATIO = None       # From volume_climax_sweep (e.g., 2.0)

# Exit patterns (set to 0 to skip)
BEST_QS_PTS = 0                 # From quick_stop_sweep
BEST_QS_BARS = 0                # From quick_stop_sweep
BEST_ENGULF_PTS = 0             # From bar_pattern_exits (e.g., 4)
BEST_CONSEC_BARS = 0            # From bar_pattern_exits (e.g., 3)

# Pass/fail criteria
PF_IMPROVEMENT_THRESHOLD = 0.05   # 5% PF improvement over baseline = pass
MIN_TRADE_COUNT_RATIO = 0.50      # Must retain at least 50% of baseline trades


def combine_gates(*gates):
    """AND multiple boolean gate arrays. None gates are ignored."""
    active = [g for g in gates if g is not None]
    if not active:
        return None
    result = active[0].copy()
    for g in active[1:]:
        result &= g
    return result


# ===== TEST CONFIGURATIONS =====
# Each tuple: (name, config_dict)
# Config keys:
#   sm_threshold  - override SM threshold
#   slope_gate    - True to use SM slope gate
#   rally_gate    - True to use rally exhaustion gate
#   volume_gate   - True to use volume climax gate
#   qs_pts/qs_bars - quick stop params
#   engulf_pts    - engulfing bar exit
#   consec_bars   - consecutive adverse bar exit
CONFIGS = [
    ("Baseline",              {}),
    ("SM_Magnitude",          {"sm_threshold": BEST_SM_THRESHOLD}),
    ("SM_Slope",              {"slope_gate": True}),
    ("Rally_Exhaustion",      {"rally_gate": True}),
    ("Volume_Climax",         {"volume_gate": True}),
    ("Quick_Stop",            {"qs_pts": BEST_QS_PTS, "qs_bars": BEST_QS_BARS}),
    ("Engulf_Exit",           {"engulf_pts": BEST_ENGULF_PTS}),
    ("Consec_Adverse",        {"consec_bars": BEST_CONSEC_BARS}),
    ("Best_Entry_Gate",       {"slope_gate": True, "rally_gate": True, "volume_gate": True}),
    ("Best_Exit_Pattern",     {"qs_pts": BEST_QS_PTS, "qs_bars": BEST_QS_BARS,
                               "engulf_pts": BEST_ENGULF_PTS, "consec_bars": BEST_CONSEC_BARS}),
    ("Best_Entry+Exit",       {"slope_gate": True, "rally_gate": True, "volume_gate": True,
                               "qs_pts": BEST_QS_PTS, "qs_bars": BEST_QS_BARS,
                               "engulf_pts": BEST_ENGULF_PTS, "consec_bars": BEST_CONSEC_BARS}),
    ("Kitchen_Sink",          {"sm_threshold": BEST_SM_THRESHOLD,
                               "slope_gate": True, "rally_gate": True, "volume_gate": True,
                               "qs_pts": BEST_QS_PTS, "qs_bars": BEST_QS_BARS,
                               "engulf_pts": BEST_ENGULF_PTS, "consec_bars": BEST_CONSEC_BARS}),
]


def should_skip(cfg):
    """Return True if ALL filter values in this config are None/0 (nothing active)."""
    if not cfg:
        return False  # Baseline always runs

    for key, val in cfg.items():
        if key == "sm_threshold" and val is not None:
            return False
        if key in ("slope_gate", "rally_gate", "volume_gate") and val:
            # Check whether the underlying threshold is set
            if key == "slope_gate" and BEST_SM_SLOPE is not None:
                return False
            if key == "rally_gate" and BEST_RALLY_LOOKBACK is not None:
                return False
            if key == "volume_gate" and BEST_VOL_LOOKBACK is not None:
                return False
        if key in ("qs_pts", "qs_bars") and val and val > 0:
            return False
        if key == "engulf_pts" and val and val > 0:
            return False
        if key == "consec_bars" and val and val > 0:
            return False
    return True


def run_config(name, cfg, split_arrays, split_name,
               sm_slope_gates, rally_gates, volume_gates):
    """Run a single config on a single split. Returns (sc, trades)."""
    opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]

    # --- Determine SM threshold ---
    sm_thresh = cfg.get("sm_threshold", MESV2_SM_THRESHOLD)
    if sm_thresh is None:
        sm_thresh = MESV2_SM_THRESHOLD

    # --- Build combined entry gate ---
    gate_parts = []

    if cfg.get("slope_gate") and BEST_SM_SLOPE is not None:
        gate_parts.append(sm_slope_gates.get(split_name))

    if cfg.get("rally_gate") and BEST_RALLY_LOOKBACK is not None:
        gate_parts.append(rally_gates.get(split_name))

    if cfg.get("volume_gate") and BEST_VOL_LOOKBACK is not None:
        gate_parts.append(volume_gates.get(split_name))

    entry_gate = combine_gates(*gate_parts)

    # --- Exit params ---
    qs_pts = cfg.get("qs_pts", 0) or 0
    qs_bars = cfg.get("qs_bars", 0) or 0
    engulf_pts = cfg.get("engulf_pts", 0) or 0
    consec_bars = cfg.get("consec_bars", 0) or 0

    # --- Run backtest ---
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm_arr, times,
        rsi_curr, rsi_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=sm_thresh, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        entry_gate=entry_gate,
        quick_stop_pts=qs_pts,
        quick_stop_bars=qs_bars,
        engulf_exit_pts=engulf_pts,
        consec_adverse_bars=consec_bars,
    )
    compute_mfe_mae(trades, highs, lows)
    sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                      dollar_per_pt=MES_DOLLAR_PER_PT)
    return sc, trades


def main():
    print("=" * 80)
    print("MES v2 COMBINED ENTRY/EXIT FILTER TEST")
    print("=" * 80)
    print(f"\n  breakeven_after_bars = {MESV2_BREAKEVEN_BARS} (held constant)")
    print(f"\n  Entry gate thresholds:")
    print(f"    SM_THRESHOLD  = {BEST_SM_THRESHOLD}")
    print(f"    SM_SLOPE      = {BEST_SM_SLOPE}")
    print(f"    RALLY_LOOKBACK= {BEST_RALLY_LOOKBACK}, MAX_MOVE={BEST_RALLY_MAX_MOVE}")
    print(f"    VOL_LOOKBACK  = {BEST_VOL_LOOKBACK}, MAX_RATIO={BEST_VOL_MAX_RATIO}")
    print(f"\n  Exit pattern thresholds:")
    print(f"    QS_PTS={BEST_QS_PTS}, QS_BARS={BEST_QS_BARS}")
    print(f"    ENGULF_PTS={BEST_ENGULF_PTS}")
    print(f"    CONSEC_BARS={BEST_CONSEC_BARS}")

    # --- Load MES data ---
    print("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    print(f"  {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # Compute SM on full data (before split -- matches existing pipeline)
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

    # --- Pre-build gate arrays per split ---
    # SM slope gates
    sm_slope_gates = {}
    if BEST_SM_SLOPE is not None:
        print(f"\n  Building SM slope gates (slope >= {BEST_SM_SLOPE})...")
        for split_name, df_split, _ in splits:
            sm_slope_gates[split_name] = build_sm_slope_gate(
                df_split["SM_Net"].values, BEST_SM_SLOPE
            )

    # Rally exhaustion gates
    rally_gates = {}
    if BEST_RALLY_LOOKBACK is not None and BEST_RALLY_MAX_MOVE is not None:
        print(f"  Building rally exhaustion gates (lookback={BEST_RALLY_LOOKBACK}, "
              f"max_move={BEST_RALLY_MAX_MOVE})...")
        for split_name, df_split, _ in splits:
            rally_gates[split_name] = build_rally_exhaustion_gate(
                df_split["Close"].values, BEST_RALLY_LOOKBACK, BEST_RALLY_MAX_MOVE
            )

    # Volume climax gates
    volume_gates = {}
    if BEST_VOL_LOOKBACK is not None and BEST_VOL_MAX_RATIO is not None:
        print(f"  Building volume climax gates (lookback={BEST_VOL_LOOKBACK}, "
              f"max_ratio={BEST_VOL_MAX_RATIO})...")
        for split_name, df_split, _ in splits:
            volume_gates[split_name] = build_volume_climax_gate(
                df_split["Volume"].values, BEST_VOL_LOOKBACK, BEST_VOL_MAX_RATIO
            )

    # --- Determine which configs to run ---
    active_configs = []
    skipped_configs = []
    for name, cfg in CONFIGS:
        if name == "Baseline" or not should_skip(cfg):
            active_configs.append((name, cfg))
        else:
            skipped_configs.append(name)

    if skipped_configs:
        print(f"\n  Skipping configs (all thresholds None/0): {', '.join(skipped_configs)}")
    print(f"  Running {len(active_configs)} configs across {len(splits)} splits "
          f"= {len(active_configs) * len(splits)} backtests")

    # --- Run all configs ---
    results = []  # (name, split_name, dr, sc, trades)
    baseline_counts = {}
    baseline_pf = {}
    baseline_sharpe = {}

    for name, cfg in active_configs:
        for split_name, _, dr in splits:
            sc, trades = run_config(name, cfg, split_arrays, split_name,
                                    sm_slope_gates, rally_gates, volume_gates)
            results.append((name, split_name, dr, sc, trades))

            # Track baseline stats
            if name == "Baseline":
                baseline_counts[split_name] = sc["count"] if sc else 0
                baseline_pf[split_name] = sc["pf"] if sc else 0.0
                baseline_sharpe[split_name] = sc["sharpe"] if sc else 0.0

    # === SUMMARY TABLE ===
    print(f"\n{'='*140}")
    print(f"{'Config':<22} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'dPF%':>7} {'dTrades%':>9} {'dSharpe':>8}")
    print(f"{'-'*140}")

    for name, split_name, dr, sc, trades in results:
        if sc is None:
            print(f"{name:<22} {split_name:>5}   NO TRADES")
            continue

        bl_pf = baseline_pf.get(split_name, 0.0)
        bl_count = baseline_counts.get(split_name, 0)
        bl_sharpe = baseline_sharpe.get(split_name, 0.0)
        dpf_pct = ((sc["pf"] - bl_pf) / bl_pf * 100) if bl_pf > 0 else 0.0
        dtrades_pct = ((sc["count"] - bl_count) / bl_count * 100) if bl_count > 0 else 0.0
        dsharpe = sc["sharpe"] - bl_sharpe

        print(f"{name:<22} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{dpf_pct:>+6.1f}% {dtrades_pct:>+8.1f}% {dsharpe:>+7.3f}")

    # === EXIT REASON BREAKDOWN (FULL only) ===
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")

    all_reasons = set()
    for name, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())
    sorted_reasons = sorted(all_reasons)

    print(f"{'Config':<22} ", end="")
    for reason in sorted_reasons:
        print(f"{reason:>10}", end="")
    print()
    print("-" * (22 + 10 * len(sorted_reasons)))

    for name, split_name, _, sc, _ in results:
        if split_name != "FULL":
            continue
        if sc is None:
            continue
        print(f"{name:<22} ", end="")
        for reason in sorted_reasons:
            print(f"{sc['exits'].get(reason, 0):>10}", end="")
        print()

    # === PASS/FAIL ANALYSIS ===
    print(f"\n{'='*100}")
    print("PASS/FAIL ANALYSIS")
    print(f"  Criteria: IS & OOS PF improvement >= {PF_IMPROVEMENT_THRESHOLD*100:.0f}% over baseline")
    print(f"            Per-half trade count >= {MIN_TRADE_COUNT_RATIO*100:.0f}% of baseline")
    print(f"{'='*100}")

    # Build lookup: (name, split_name) -> sc
    result_lookup = {}
    for name, split_name, dr, sc, trades in results:
        result_lookup[(name, split_name)] = sc

    passed_configs = []
    print(f"\n{'Config':<22} {'IS_PF':>7} {'OOS_PF':>7} {'IS_dPF%':>8} {'OOS_dPF%':>9} "
          f"{'IS_Trd':>7} {'OOS_Trd':>8} {'IS_Trd%':>8} {'OOS_Trd%':>9} {'Result':>10}")
    print("-" * 110)

    for name, cfg in active_configs:
        is_sc = result_lookup.get((name, "IS"))
        oos_sc = result_lookup.get((name, "OOS"))

        is_pf = is_sc["pf"] if is_sc else 0.0
        oos_pf = oos_sc["pf"] if oos_sc else 0.0
        is_count = is_sc["count"] if is_sc else 0
        oos_count = oos_sc["count"] if oos_sc else 0

        bl_is_pf = baseline_pf.get("IS", 0.0)
        bl_oos_pf = baseline_pf.get("OOS", 0.0)
        bl_is_count = baseline_counts.get("IS", 0)
        bl_oos_count = baseline_counts.get("OOS", 0)

        is_dpf = ((is_pf - bl_is_pf) / bl_is_pf) if bl_is_pf > 0 else 0.0
        oos_dpf = ((oos_pf - bl_oos_pf) / bl_oos_pf) if bl_oos_pf > 0 else 0.0
        is_trd_ratio = (is_count / bl_is_count) if bl_is_count > 0 else 0.0
        oos_trd_ratio = (oos_count / bl_oos_count) if bl_oos_count > 0 else 0.0

        pf_pass = (is_dpf >= PF_IMPROVEMENT_THRESHOLD and oos_dpf >= PF_IMPROVEMENT_THRESHOLD)
        count_pass = (is_trd_ratio >= MIN_TRADE_COUNT_RATIO and oos_trd_ratio >= MIN_TRADE_COUNT_RATIO)
        passed = pf_pass and count_pass

        status = "PASS" if passed else "FAIL"
        if name == "Baseline":
            status = "BASELINE"
        if passed:
            passed_configs.append(name)

        print(f"{name:<22} {is_pf:>7.3f} {oos_pf:>7.3f} "
              f"{is_dpf:>+7.1%} {oos_dpf:>+8.1%} "
              f"{is_count:>7} {oos_count:>8} "
              f"{is_trd_ratio:>7.0%} {oos_trd_ratio:>8.0%} "
              f"{status:>10}")

    # === BEST COMBO SELECTION ===
    print(f"\n{'='*80}")
    print("BEST COMBO SELECTION")
    print(f"{'='*80}")

    total_non_baseline = len([n for n, _ in active_configs if n != "Baseline"])
    pass_count = len(passed_configs)
    print(f"\n  Pass rate: {pass_count} of {total_non_baseline} non-baseline configs passed")

    if passed_configs:
        # Find best among passed: highest OOS Sharpe
        best_name = None
        best_oos_sharpe = -999
        for name in passed_configs:
            oos_sc = result_lookup.get((name, "OOS"))
            if oos_sc and oos_sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = oos_sc["sharpe"]
                best_name = name

        print(f"  Passed configs: {passed_configs}")
        print(f"\n  >>> BEST COMBO: {best_name} (OOS Sharpe {best_oos_sharpe:.3f}) <<<")

        # Print full stats for best combo
        print(f"\n  Best combo detail:")
        for split_check in ["IS", "OOS", "FULL"]:
            sc = result_lookup.get((best_name, split_check))
            if sc:
                bl_pf = baseline_pf.get(split_check, 0.0)
                dpf_pct = ((sc["pf"] - bl_pf) / bl_pf * 100) if bl_pf > 0 else 0.0
                print(f"    {split_check:>4}: {sc['count']} trades, WR {sc['win_rate']:.1f}%, "
                      f"PF {sc['pf']:.3f} ({dpf_pct:+.1f}%), "
                      f"P&L {sc['net_dollar']:+.2f}, "
                      f"Sharpe {sc['sharpe']:.3f}, MaxDD {sc['max_dd_dollar']:.2f}")

        # --- Save best combo trades ---
        best_cfg = dict(active_configs)[best_name]
        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": best_cfg.get("sm_threshold", MESV2_SM_THRESHOLD) or MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
            "sm_slope": BEST_SM_SLOPE if best_cfg.get("slope_gate") else None,
            "rally_lookback": BEST_RALLY_LOOKBACK if best_cfg.get("rally_gate") else None,
            "rally_max_move": BEST_RALLY_MAX_MOVE if best_cfg.get("rally_gate") else None,
            "vol_lookback": BEST_VOL_LOOKBACK if best_cfg.get("volume_gate") else None,
            "vol_max_ratio": BEST_VOL_MAX_RATIO if best_cfg.get("volume_gate") else None,
            "quick_stop_pts": best_cfg.get("qs_pts", 0) or 0,
            "quick_stop_bars": best_cfg.get("qs_bars", 0) or 0,
            "engulf_exit_pts": best_cfg.get("engulf_pts", 0) or 0,
            "consec_adverse_bars": best_cfg.get("consec_bars", 0) or 0,
        }

        print(f"\nSaving trade CSVs for best combo '{best_name}'...")
        for name, split_name, dr, sc, trades in results:
            if name == best_name and trades:
                save_backtest(
                    trades, strategy="MES_V2_COMBINED", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"best_combo={best_name}",
                )
    else:
        print("  No configs passed all criteria. Baseline remains best.")

    # --- Also report best OOS Sharpe overall (even if it didn't pass) ---
    best_overall_name = None
    best_overall_sharpe = -999
    for name, _ in active_configs:
        if name == "Baseline":
            continue
        oos_sc = result_lookup.get((name, "OOS"))
        if oos_sc and oos_sc["sharpe"] > best_overall_sharpe:
            best_overall_sharpe = oos_sc["sharpe"]
            best_overall_name = name

    if best_overall_name:
        print(f"\n  Best OOS Sharpe overall: {best_overall_name} "
              f"(Sharpe {best_overall_sharpe:.3f})")
        if best_overall_name not in passed_configs:
            print(f"  -> WARNING: Best Sharpe config did NOT pass criteria "
                  f"(trade count or PF delta).")

    print("\nDone.")


if __name__ == "__main__":
    main()
