"""
MES v2 ATR-Normalized Extension Gate Sweep
============================================
Sweeps lookback and max_atr_mult thresholds for MES v2. The gate blocks entries
when price has moved more than max_atr_mult * ATR(14) over the lookback window.

Hypothesis: Raw point thresholds are noisy because 30 pts means different things
in different volatility regimes. Normalizing by ATR adapts: 30 pts when ATR=15
is a 2x extension (extreme); 30 pts when ATR=40 is 0.75x (normal).

Sweep grid: 4 lookbacks x 5 max_atr_mult = 20 combos + baseline = 21 configs.
breakeven_after_bars=75 held constant throughout.

Usage:
    python3 mes_v2_atr_extension_sweep.py
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
MAX_ATR_MULT_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
ATR_PERIOD = 14
SCRIPT_NAME = "mes_v2_atr_extension_sweep.py"


def compute_atr(highs, lows, closes, atr_period=14):
    """Compute Wilder's ATR."""
    n = len(closes)
    tr = np.zeros(n)
    for j in range(1, n):
        tr[j] = max(highs[j] - lows[j],
                     abs(highs[j] - closes[j - 1]),
                     abs(lows[j] - closes[j - 1]))
    atr = np.zeros(n)
    if n > atr_period:
        atr[atr_period] = np.mean(tr[1:atr_period + 1])
        for j in range(atr_period + 1, n):
            atr[j] = (atr[j - 1] * (atr_period - 1) + tr[j]) / atr_period
    return atr


def build_atr_extension_gate(closes, highs, lows, sm, lookback, max_atr_mult,
                              atr_period=14, sm_threshold=0.0):
    """Block entries when price move over lookback bars exceeds max_atr_mult * ATR.

    For longs (sm > threshold): block if upward move / ATR > max_atr_mult.
    For shorts (sm < -threshold): block if downward move / ATR > max_atr_mult.
    Only blocks directional moves (long: block upward extension, short: block downward).
    """
    n = len(closes)
    atr = compute_atr(highs, lows, closes, atr_period)

    gate = np.ones(n, dtype=bool)
    start = max(lookback, atr_period + 1)
    for j in range(start, n):
        if atr[j] <= 0:
            continue
        move = closes[j] - closes[j - lookback]
        extension = abs(move) / atr[j]
        if sm[j] > sm_threshold and move > 0:      # potential long, already rallied
            gate[j] = extension <= max_atr_mult
        elif sm[j] < -sm_threshold and move < 0:   # potential short, already dropped
            gate[j] = extension <= max_atr_mult
    return gate


def run_single(opens, highs, lows, closes, sm, times,
               rsi_curr, rsi_prev, entry_gate=None):
    """Run MES v2 backtest with a specific entry_gate."""
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
    print("MES v2 ATR-NORMALIZED EXTENSION GATE SWEEP (pre-filter)")
    print(f"  ATR period={ATR_PERIOD}, breakeven_after_bars={MESV2_BREAKEVEN_BARS} (constant)")
    print("=" * 80)

    # --- Load MES data ---
    print("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    print(f"  {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # Compute SM on full data
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm

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

    # --- Prepare arrays per split ---
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
    full_sm = df_mes["SM_Net"].values
    full_closes = df_mes["Close"].values
    full_highs = df_mes["High"].values
    full_lows = df_mes["Low"].values
    is_len = len(mes_is)

    # Build all combo gates
    sweep_configs = [(None, None)]  # baseline
    for lb in LOOKBACK_VALUES:
        for mult in MAX_ATR_MULT_VALUES:
            sweep_configs.append((lb, mult))

    gates_by_config = {}
    for lb, mult in sweep_configs:
        if lb is None:
            gates_by_config[(lb, mult)] = {"FULL": None, "IS": None, "OOS": None}
        else:
            full_gate = build_atr_extension_gate(
                full_closes, full_highs, full_lows, full_sm,
                lb, mult, ATR_PERIOD, MESV2_SM_THRESHOLD)
            gates_by_config[(lb, mult)] = {
                "FULL": full_gate,
                "IS": full_gate[:is_len],
                "OOS": full_gate[is_len:],
            }

    # --- Gate pass rates ---
    print(f"\n{'='*70}")
    print("GATE PASS RATES (% of bars where gate=True)")
    print(f"{'='*70}")
    print(f"{'LB':>4} {'Mult':>6} {'FULL':>8} {'IS':>8} {'OOS':>8}")
    print(f"{'-'*40}")
    for lb, mult in sweep_configs:
        lb_label = "None" if lb is None else f"{lb}"
        mult_label = "None" if mult is None else f"{mult:.1f}"
        if lb is None:
            print(f"{lb_label:>4} {mult_label:>6} {'100.0%':>8} {'100.0%':>8} {'100.0%':>8}")
        else:
            g = gates_by_config[(lb, mult)]
            full_pct = g["FULL"].mean() * 100
            is_pct = g["IS"].mean() * 100
            oos_pct = g["OOS"].mean() * 100
            print(f"{lb_label:>4} {mult_label:>6} {full_pct:>7.1f}% {is_pct:>7.1f}% {oos_pct:>7.1f}%")

    # --- Sweep ---
    results = []
    best_oos_sharpe = -999
    best_config = (None, None)
    baseline_counts = {}

    for lb, mult in sweep_configs:
        for split_name, _, dr in splits:
            opens, highs, lows, closes, sm_arr, times, rsi_curr, rsi_prev = split_arrays[split_name]
            gate = gates_by_config[(lb, mult)][split_name]
            trades = run_single(opens, highs, lows, closes, sm_arr, times,
                                rsi_curr, rsi_prev, entry_gate=gate)
            sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                              dollar_per_pt=MES_DOLLAR_PER_PT)

            if lb is None:
                baseline_counts[split_name] = sc["count"] if sc else 0

            results.append((lb, mult, split_name, dr, sc, trades))

            if split_name == "OOS" and sc and sc["sharpe"] > best_oos_sharpe:
                best_oos_sharpe = sc["sharpe"]
                best_config = (lb, mult)

    # --- Print summary table ---
    print(f"\n{'='*140}")
    print(f"{'LB':>4} {'Mult':>6} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net P&L':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'Blocked':>8} {'Pass/Fail':>10}")
    print(f"{'-'*140}")

    oos_results_by_config = {}

    for lb, mult, split_name, dr, sc, trades in results:
        lb_label = "None" if lb is None else f"{lb}"
        mult_label = "None" if mult is None else f"{mult:.1f}"
        config_label = f"{lb_label}/{mult_label}"

        if sc is None:
            print(f"{lb_label:>4} {mult_label:>6} {split_name:>5}   NO TRADES")
            if split_name == "OOS":
                oos_results_by_config[(lb, mult)] = {"pf": 0.0, "sharpe": -999}
            continue

        baseline = baseline_counts.get(split_name, 0)
        blocked = baseline - sc["count"]

        pf_label = ""
        if split_name == "OOS":
            oos_results_by_config[(lb, mult)] = {"pf": sc["pf"], "sharpe": sc["sharpe"]}
            passed = sc["pf"] > 1.0 and sc["sharpe"] > 0
            pf_label = "PASS" if passed else "FAIL"

        print(f"{lb_label:>4} {mult_label:>6} {split_name:>5} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
              f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
              f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
              f"{blocked:>8} {pf_label:>10}")

    # --- Grid tables (lookback rows x mult columns) for FULL/IS/OOS ---
    for metric_name, metric_key in [("PF", "pf"), ("Sharpe", "sharpe")]:
        for split_name in ["FULL", "IS", "OOS"]:
            print(f"\n{'='*60}")
            print(f"{metric_name} GRID — {split_name}")
            print(f"{'='*60}")
            # Header
            print(f"{'LB':>4}", end="")
            for mult in MAX_ATR_MULT_VALUES:
                print(f" {mult:>7.1f}", end="")
            print()
            print("-" * (4 + 8 * len(MAX_ATR_MULT_VALUES)))

            for lb in LOOKBACK_VALUES:
                print(f"{lb:>4}", end="")
                for mult in MAX_ATR_MULT_VALUES:
                    val = None
                    for rlb, rmult, rsplit, _, rsc, _ in results:
                        if rlb == lb and rmult == mult and rsplit == split_name and rsc:
                            val = rsc[metric_key]
                            break
                    if val is not None:
                        print(f" {val:>7.3f}", end="")
                    else:
                        print(f" {'N/A':>7}", end="")
                print()

    # --- Pass/fail assessment ---
    print(f"\n{'='*80}")
    print("PASS/FAIL ASSESSMENT")
    print(f"{'='*80}")

    baseline_is_pf = None
    baseline_oos_pf = None
    baseline_is_sharpe = None
    baseline_oos_sharpe = None
    baseline_is_count = 0
    baseline_oos_count = 0

    for lb, mult, split_name, _, sc, _ in results:
        if lb is None and sc:
            if split_name == "IS":
                baseline_is_pf = sc["pf"]
                baseline_is_sharpe = sc["sharpe"]
                baseline_is_count = sc["count"]
            elif split_name == "OOS":
                baseline_oos_pf = sc["pf"]
                baseline_oos_sharpe = sc["sharpe"]
                baseline_oos_count = sc["count"]

    print(f"Baseline IS:  PF={baseline_is_pf:.3f}, Sharpe={baseline_is_sharpe:.3f}, N={baseline_is_count}")
    print(f"Baseline OOS: PF={baseline_oos_pf:.3f}, Sharpe={baseline_oos_sharpe:.3f}, N={baseline_oos_count}")
    print()

    pass_count = 0
    total_non_baseline = 0
    for lb in LOOKBACK_VALUES:
        for mult in MAX_ATR_MULT_VALUES:
            total_non_baseline += 1
            is_sc = None
            oos_sc = None
            for rlb, rmult, rsplit, _, rsc, _ in results:
                if rlb == lb and rmult == mult and rsc:
                    if rsplit == "IS":
                        is_sc = rsc
                    elif rsplit == "OOS":
                        oos_sc = rsc

            if is_sc is None or oos_sc is None:
                print(f"  LB={lb:>2} Mult={mult:.1f}  FAIL (missing data)")
                continue

            is_pf_chg = (is_sc["pf"] - baseline_is_pf) / baseline_is_pf * 100
            oos_pf_chg = (oos_sc["pf"] - baseline_oos_pf) / baseline_oos_pf * 100
            is_count_pct = is_sc["count"] / baseline_is_count * 100 if baseline_is_count else 0
            oos_count_pct = oos_sc["count"] / baseline_oos_count * 100 if baseline_oos_count else 0

            is_profitable = is_sc["net_dollar"] > 0
            oos_profitable = oos_sc["net_dollar"] > 0

            strong = (is_pf_chg >= 5 and oos_pf_chg >= 5 and
                      is_sc["sharpe"] >= baseline_is_sharpe and
                      oos_sc["sharpe"] >= baseline_oos_sharpe and
                      is_count_pct >= 70 and oos_count_pct >= 70 and
                      is_profitable and oos_profitable)
            marginal = (not strong and
                        is_pf_chg >= -5 and oos_pf_chg >= -5 and
                        (is_sc["sharpe"] >= baseline_is_sharpe or oos_sc["sharpe"] >= baseline_oos_sharpe) and
                        is_count_pct >= 70 and oos_count_pct >= 70 and
                        is_profitable and oos_profitable)

            if strong:
                verdict = "STRONG PASS"
                pass_count += 1
            elif marginal:
                verdict = "MARGINAL PASS"
                pass_count += 1
            else:
                verdict = "FAIL"

            print(f"  LB={lb:>2} Mult={mult:.1f}  {verdict:>14}  IS PF {is_pf_chg:+.1f}%  OOS PF {oos_pf_chg:+.1f}%  "
                  f"IS N={is_sc['count']}({is_count_pct:.0f}%)  OOS N={oos_sc['count']}({oos_count_pct:.0f}%)")

    print(f"\nPass rate: {pass_count}/{total_non_baseline} ({pass_count/total_non_baseline*100:.0f}%)")

    # --- Exit reason breakdown ---
    print(f"\n{'='*80}")
    print("EXIT REASON BREAKDOWN (FULL data)")
    print(f"{'='*80}")
    all_reasons = set()
    for lb, mult, split_name, _, sc, _ in results:
        if split_name == "FULL" and sc:
            all_reasons.update(sc["exits"].keys())

    print(f"{'LB':>4} {'Mult':>6} ", end="")
    for reason in sorted(all_reasons):
        print(f"{reason:>10}", end="")
    print()
    print("-" * (12 + 10 * len(all_reasons)))

    for lb, mult, split_name, _, sc, _ in results:
        if split_name != "FULL":
            continue
        lb_label = "None" if lb is None else f"{lb}"
        mult_label = "None" if mult is None else f"{mult:.1f}"
        if sc is None:
            continue
        print(f"{lb_label:>4} {mult_label:>6} ", end="")
        for reason in sorted(all_reasons):
            print(f"{sc['exits'].get(reason, 0):>10}", end="")
        print()

    # --- Summary ---
    print(f"\n{'='*80}")
    best_lb, best_mult = best_config
    best_lb_label = "None" if best_lb is None else f"{best_lb}"
    best_mult_label = "None" if best_mult is None else f"{best_mult:.1f}"
    print(f"BEST OOS SHARPE: LB={best_lb_label} Mult={best_mult_label} (Sharpe {best_oos_sharpe:.3f})")
    print(f"{'='*80}")

    # --- Save best result ---
    if best_config != (None, None):
        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
            "atr_extension_lookback": best_lb,
            "atr_extension_max_mult": best_mult,
            "atr_period": ATR_PERIOD,
        }

        print(f"\nSaving trade CSVs for best LB={best_lb_label} Mult={best_mult_label}...")
        for lb, mult, split_name, dr, sc, trades in results:
            if lb == best_lb and mult == best_mult and trades:
                save_backtest(
                    trades, strategy="MES_V2_ATR_EXT", params=mesv2_params,
                    data_range=dr, split=split_name,
                    dollar_per_pt=MES_DOLLAR_PER_PT, commission=MES_COMMISSION,
                    script_name=SCRIPT_NAME,
                    notes=f"atr_extension LB={best_lb} mult={best_mult}",
                )
    else:
        print("Baseline (None = no gate) is best -- no ATR extension gate needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
