"""
Round 3 — S/R and Price Structure Entry Gates: Shared Infrastructure
=====================================================================
All 6 sweep scripts import from this module.

Provides:
  - STRATEGIES config table (all 3 active strategies)
  - prepare_data()         — load MNQ + MES, compute SM, split IS/OOS
  - run_single()           — run one backtest with strategy params from dict
  - assess_pass_fail()     — evaluate IS/OOS results
  - compute_atr_wilder()   — reusable ATR
  - print_sweep_table()    — formatted output
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Path setup ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR.parent))  # backtesting_engine/
sys.path.insert(0, str(_STRAT_DIR))         # strategies/

from v10_test_common import (
    compute_smart_money,
    compute_et_minutes,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)

sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))
from generate_session import (
    run_backtest_tp_exit,
    compute_mfe_mae,
)

from results.save_results import save_backtest

# ---------------------------------------------------------------------------
# Strategy config table
# ---------------------------------------------------------------------------

STRATEGIES = [
    {
        "name": "vScalpA", "strategy_id": "MNQ_V15", "instrument": "MNQ",
        "sm_index": 10, "sm_flow": 12, "sm_norm": 200, "sm_ema": 100,
        "rsi_len": 8, "rsi_buy": 60, "rsi_sell": 40, "sm_threshold": 0.0,
        "cooldown": 20, "max_loss_pts": 40, "tp_pts": 5,
        "eod_et": NY_CLOSE_ET, "breakeven_bars": 0,
        "entry_end_et": 13 * 60, "dollar_per_pt": 2.0, "commission": 0.52,
    },
    {
        "name": "vScalpB", "strategy_id": "MNQ_VSCALPB", "instrument": "MNQ",
        "sm_index": 10, "sm_flow": 12, "sm_norm": 200, "sm_ema": 100,
        "rsi_len": 8, "rsi_buy": 55, "rsi_sell": 45, "sm_threshold": 0.25,
        "cooldown": 20, "max_loss_pts": 15, "tp_pts": 5,
        "eod_et": NY_CLOSE_ET, "breakeven_bars": 0,
        "entry_end_et": NY_LAST_ENTRY_ET, "dollar_per_pt": 2.0, "commission": 0.52,
    },
    {
        "name": "MES_V2", "strategy_id": "MES_V2", "instrument": "MES",
        "sm_index": 20, "sm_flow": 12, "sm_norm": 400, "sm_ema": 255,
        "rsi_len": 12, "rsi_buy": 55, "rsi_sell": 45, "sm_threshold": 0.0,
        "cooldown": 25, "max_loss_pts": 35, "tp_pts": 20,
        "eod_et": 16 * 60, "breakeven_bars": 75,
        "entry_end_et": 14 * 60 + 15, "dollar_per_pt": 5.0, "commission": 1.25,
    },
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data():
    """Load MNQ + MES, compute SM, split IS/OOS, build split_arrays.

    Returns:
        instruments: dict  instrument -> df (full, with SM_Net column)
        split_arrays: dict (strategy_name, split_name) -> (opens, highs, lows,
                       closes, sm, times, rsi_curr, rsi_prev)
        split_indices: dict instrument -> {"is_len": int, "midpoint": Timestamp}
    """
    instruments = {}
    for inst in ["MNQ", "MES"]:
        df = load_instrument_1min(inst)
        # SM params differ per instrument — use the first strategy for that instrument
        strat = next(s for s in STRATEGIES if s["instrument"] == inst)
        sm = compute_smart_money(
            df["Close"].values, df["Volume"].values,
            index_period=strat["sm_index"], flow_period=strat["sm_flow"],
            norm_period=strat["sm_norm"], ema_len=strat["sm_ema"],
        )
        df["SM_Net"] = sm
        instruments[inst] = df
        print(f"Loaded {inst}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Split each instrument at midpoint
    split_indices = {}
    for inst, df in instruments.items():
        mid = df.index[len(df) // 2]
        is_len = (df.index < mid).sum()
        split_indices[inst] = {"is_len": is_len, "midpoint": mid}
        is_range = f"{df.index[0].strftime('%Y-%m-%d')}_to_{df.index[is_len-1].strftime('%Y-%m-%d')}"
        oos_range = f"{df.index[is_len].strftime('%Y-%m-%d')}_to_{df.index[-1].strftime('%Y-%m-%d')}"
        print(f"  {inst} split: IS {is_len} bars ({is_range}), OOS {len(df)-is_len} bars ({oos_range})")

    # Build per-(strategy, split) arrays with RSI mapped within each split
    split_arrays = {}
    # Track unique (instrument, rsi_len) combos to avoid duplicate RSI computation
    rsi_cache = {}

    for strat in STRATEGIES:
        inst = strat["instrument"]
        df = instruments[inst]
        mid = split_indices[inst]["midpoint"]
        is_len = split_indices[inst]["is_len"]

        splits = {
            "FULL": df,
            "IS": df[df.index < mid],
            "OOS": df[df.index >= mid],
        }

        for split_name, df_split in splits.items():
            cache_key = (inst, strat["rsi_len"], split_name)
            if cache_key in rsi_cache:
                rsi_curr, rsi_prev = rsi_cache[cache_key]
            else:
                df_5m = resample_to_5min(df_split)
                rsi_curr, rsi_prev = map_5min_rsi_to_1min(
                    df_split.index.values, df_5m.index.values,
                    df_5m["Close"].values, rsi_len=strat["rsi_len"],
                )
                rsi_cache[cache_key] = (rsi_curr, rsi_prev)

            split_arrays[(strat["name"], split_name)] = (
                df_split["Open"].values,
                df_split["High"].values,
                df_split["Low"].values,
                df_split["Close"].values,
                df_split["SM_Net"].values,
                df_split.index,
                rsi_curr,
                rsi_prev,
            )

    return instruments, split_arrays, split_indices


# ---------------------------------------------------------------------------
# Run a single backtest for a strategy
# ---------------------------------------------------------------------------

def run_single(strat, opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, entry_gate=None):
    """Run one backtest with params from a STRATEGIES dict entry."""
    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=strat["rsi_buy"], rsi_sell=strat["rsi_sell"],
        sm_threshold=strat["sm_threshold"], cooldown_bars=strat["cooldown"],
        max_loss_pts=strat["max_loss_pts"], tp_pts=strat["tp_pts"],
        eod_minutes_et=strat["eod_et"],
        breakeven_after_bars=strat["breakeven_bars"],
        entry_end_et=strat["entry_end_et"],
        entry_gate=entry_gate,
    )
    compute_mfe_mae(trades, highs, lows)
    return trades


# ---------------------------------------------------------------------------
# Gate slicing helpers
# ---------------------------------------------------------------------------

def slice_gate(full_gate, instrument, split_name, split_indices):
    """Slice a full-length gate array into IS or OOS portion.

    Args:
        full_gate: bool array same length as full instrument data (or None)
        instrument: "MNQ" or "MES"
        split_name: "FULL", "IS", or "OOS"
        split_indices: from prepare_data()

    Returns:
        Sliced gate array or None.
    """
    if full_gate is None:
        return None
    is_len = split_indices[instrument]["is_len"]
    if split_name == "FULL":
        return full_gate
    elif split_name == "IS":
        return full_gate[:is_len]
    else:
        return full_gate[is_len:]


# ---------------------------------------------------------------------------
# Pass/fail assessment
# ---------------------------------------------------------------------------

def assess_pass_fail(sc_is, sc_oos, bl_is, bl_oos):
    """Evaluate a gated config against baseline for one strategy.

    Args:
        sc_is, sc_oos: score dicts for IS/OOS with gate
        bl_is, bl_oos: score dicts for IS/OOS baseline (no gate)

    Returns:
        (verdict, detail_str) where verdict is "STRONG PASS" / "MARGINAL PASS" / "FAIL"
    """
    if sc_is is None or sc_oos is None or bl_is is None or bl_oos is None:
        return "FAIL", "no trades in one or more splits"

    # Net dollar must be positive in both halves
    if sc_is["net_dollar"] <= 0 or sc_oos["net_dollar"] <= 0:
        return "FAIL", f"not profitable (IS ${sc_is['net_dollar']:+.0f}, OOS ${sc_oos['net_dollar']:+.0f})"

    # Trade count >= 70% of baseline in each half
    is_count_pct = sc_is["count"] / bl_is["count"] * 100 if bl_is["count"] > 0 else 0
    oos_count_pct = sc_oos["count"] / bl_oos["count"] * 100 if bl_oos["count"] > 0 else 0
    if is_count_pct < 70 or oos_count_pct < 70:
        return "FAIL", f"trade count too low (IS {is_count_pct:.0f}%, OOS {oos_count_pct:.0f}%)"

    # PF change
    is_dpf = (sc_is["pf"] - bl_is["pf"]) / bl_is["pf"] * 100 if bl_is["pf"] > 0 else 0
    oos_dpf = (sc_oos["pf"] - bl_oos["pf"]) / bl_oos["pf"] * 100 if bl_oos["pf"] > 0 else 0

    # Sharpe change
    is_dsharpe = sc_is["sharpe"] - bl_is["sharpe"]
    oos_dsharpe = sc_oos["sharpe"] - bl_oos["sharpe"]

    # Fail: either half PF degrades >5%
    if is_dpf < -5 or oos_dpf < -5:
        return "FAIL", f"PF degrades (IS {is_dpf:+.1f}%, OOS {oos_dpf:+.1f}%)"

    # Strong pass: both PF +5%, both Sharpe non-negative
    if is_dpf >= 5 and oos_dpf >= 5 and is_dsharpe >= 0 and oos_dsharpe >= 0:
        return "STRONG PASS", f"IS PF {is_dpf:+.1f}%, OOS PF {oos_dpf:+.1f}%"

    # Marginal pass: PF within ±5%, at least one Sharpe improves
    if is_dpf >= -5 and oos_dpf >= -5 and (is_dsharpe > 0 or oos_dsharpe > 0):
        return "MARGINAL PASS", f"IS PF {is_dpf:+.1f}%, OOS PF {oos_dpf:+.1f}%"

    return "FAIL", f"no improvement (IS PF {is_dpf:+.1f}%, OOS PF {oos_dpf:+.1f}%)"


# ---------------------------------------------------------------------------
# ATR (Wilder)
# ---------------------------------------------------------------------------

def compute_atr_wilder(highs, lows, closes, period=14):
    """Wilder's ATR — same as Pine's ta.atr(period)."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.empty(n)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ---------------------------------------------------------------------------
# Sweep output helpers
# ---------------------------------------------------------------------------

def run_sweep(study_name, strategies, split_arrays, split_indices, instruments,
              build_gates_fn, sweep_configs, script_name=""):
    """Generic sweep runner for all studies.

    Args:
        study_name: e.g. "VWAP Z-Score"
        strategies: STRATEGIES list
        split_arrays: from prepare_data()
        split_indices: from prepare_data()
        instruments: from prepare_data()
        build_gates_fn: callable(config, instruments) -> dict[instrument -> full_gate_array]
            config is a dict from sweep_configs. Return {instrument: bool_array} or
            {instrument: None} for baseline.
        sweep_configs: list of dicts, each with at minimum {"label": str}.
            First entry should be baseline with label "None".
        script_name: for save_backtest

    Returns:
        all_results: list of (config_label, strategy_name, split_name, sc, trades, baseline_count)
    """
    print(f"\n{'='*120}")
    print(f"{study_name} SWEEP (pre-filter, all 3 strategies)")
    print(f"{'='*120}")

    # --- Baselines ---
    baselines = {}  # (strat_name, split_name) -> sc
    baseline_cfg = sweep_configs[0]
    assert baseline_cfg["label"] == "None", "First config must be baseline (label='None')"

    print("\nRunning baselines...")
    for strat in strategies:
        for split_name in ["FULL", "IS", "OOS"]:
            arrays = split_arrays[(strat["name"], split_name)]
            trades = run_single(strat, *arrays, entry_gate=None)
            sc = score_trades(trades, commission_per_side=strat["commission"],
                              dollar_per_pt=strat["dollar_per_pt"])
            baselines[(strat["name"], split_name)] = sc
            if split_name == "FULL" and sc:
                print(f"  {strat['name']:>8} baseline: {sc['count']} trades, "
                      f"WR {sc['win_rate']}%, PF {sc['pf']}, ${sc['net_dollar']:+.0f}")

    # --- Sweep ---
    all_results = []

    for cfg in sweep_configs:
        label = cfg["label"]

        # Build gates per instrument
        if label == "None":
            gates = {inst: None for inst in ["MNQ", "MES"]}
        else:
            gates = build_gates_fn(cfg, instruments)

        for strat in strategies:
            inst = strat["instrument"]
            for split_name in ["FULL", "IS", "OOS"]:
                arrays = split_arrays[(strat["name"], split_name)]
                gate = slice_gate(gates[inst], inst, split_name, split_indices)
                trades = run_single(strat, *arrays, entry_gate=gate)
                sc = score_trades(trades, commission_per_side=strat["commission"],
                                  dollar_per_pt=strat["dollar_per_pt"])
                bl = baselines[(strat["name"], split_name)]
                bl_count = bl["count"] if bl else 0
                all_results.append((label, strat["name"], split_name, sc, trades, bl_count))

    # --- Print summary table ---
    print(f"\n{'='*145}")
    print(f"{'Config':>20} {'Strategy':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} "
          f"{'PF':>7} {'Net$':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'Blocked':>8} {'dPF%':>7} {'Verdict':>15}")
    print(f"{'-'*145}")

    for label, strat_name, split_name, sc, trades, bl_count in all_results:
        if split_name == "FULL":
            continue  # Only show IS/OOS in summary

        if sc is None:
            print(f"{label:>20} {strat_name:>8} {split_name:>5}   NO TRADES")
            continue

        bl = baselines[(strat_name, split_name)]
        blocked = bl_count - sc["count"]
        dpf = 0.0
        if bl and bl["pf"] > 0:
            dpf = (sc["pf"] - bl["pf"]) / bl["pf"] * 100

        # Verdict only for complete IS+OOS pair
        verdict = ""
        if split_name == "OOS":
            sc_is = None
            for l2, s2, sp2, sc2, _, _ in all_results:
                if l2 == label and s2 == strat_name and sp2 == "IS":
                    sc_is = sc2
                    break
            if sc_is is not None:
                bl_is = baselines[(strat_name, "IS")]
                bl_oos = baselines[(strat_name, "OOS")]
                v, _ = assess_pass_fail(sc_is, sc, bl_is, bl_oos)
                verdict = v

        print(f"{label:>20} {strat_name:>8} {split_name:>5} {sc['count']:>7} "
              f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{blocked:>8} {dpf:>+6.1f}% {verdict:>15}")

    # --- Portfolio aggregate ---
    print(f"\n{'='*100}")
    print(f"PORTFOLIO AGGREGATE (sum across strategies)")
    print(f"{'='*100}")
    print(f"{'Config':>20} {'IS_Net$':>10} {'OOS_Net$':>10} {'IS_Sharpe':>10} {'OOS_Sharpe':>10} {'Verdict':>15}")
    print(f"{'-'*100}")

    seen_labels = []
    for cfg in sweep_configs:
        label = cfg["label"]
        if label in seen_labels:
            continue
        seen_labels.append(label)

        port = {"IS": {"net": 0, "pnls": []}, "OOS": {"net": 0, "pnls": []}}
        all_valid = True
        for strat in strategies:
            for split_name in ["IS", "OOS"]:
                sc = None
                for l2, s2, sp2, sc2, _, _ in all_results:
                    if l2 == label and s2 == strat["name"] and sp2 == split_name:
                        sc = sc2
                        break
                if sc is None:
                    all_valid = False
                    continue
                port[split_name]["net"] += sc["net_dollar"]
                # Approximate per-trade pnls for portfolio Sharpe
                for l2, s2, sp2, sc2, trades2, _ in all_results:
                    if l2 == label and s2 == strat["name"] and sp2 == split_name:
                        for t in trades2:
                            pnl = t["pts"] * strat["dollar_per_pt"] - 2 * strat["commission"]
                            port[split_name]["pnls"].append(pnl)
                        break

        if not all_valid:
            print(f"{label:>20}  incomplete data")
            continue

        def portfolio_sharpe(pnls):
            if len(pnls) < 2:
                return 0.0
            arr = np.array(pnls)
            if np.std(arr) == 0:
                return 0.0
            return float(np.mean(arr) / np.std(arr) * np.sqrt(252))

        is_sharpe = portfolio_sharpe(port["IS"]["pnls"])
        oos_sharpe = portfolio_sharpe(port["OOS"]["pnls"])

        is_label = "baseline" if label == "None" else ""
        print(f"{label:>20} {port['IS']['net']:>+10.2f} {port['OOS']['net']:>+10.2f} "
              f"{is_sharpe:>10.3f} {oos_sharpe:>10.3f} {is_label:>15}")

    # --- Per-strategy verdict summary ---
    print(f"\n{'='*80}")
    print("PER-STRATEGY VERDICTS")
    print(f"{'='*80}")
    for cfg in sweep_configs:
        label = cfg["label"]
        if label == "None":
            continue
        verdicts = {}
        for strat in strategies:
            sc_is = sc_oos = None
            for l2, s2, sp2, sc2, _, _ in all_results:
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

    return all_results, baselines
