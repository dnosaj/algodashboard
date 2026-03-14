"""
vScalpC Phase 2b — ATR & Volatility Gate Sweep
=================================================
Tests entry gates based on HTF ATR levels and prior-day volatility regime.

Diagnostic found ATR across all timeframes is the strongest, most stable
predictor of TP2 hit rate (r=0.14-0.18, all IS/OOS stable).

Sweep:
  A. Intraday ATR gate:  block entries when {5min, 15min, 30min, 1h} ATR < threshold
  B. Prior-day ATR gate: block entries when prior-day ATR < threshold
  C. Vol regime gate:    block entries when prior-day vol regime = Low (0)

All gates are pre-filters (block entry, change cooldown/episode dynamics).
Each is tested as: baseline (no gate) vs gated, IS/OOS split validation.

Usage:
    cd backtesting_engine && python3 strategies/htf_filter_atr_sweep.py
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

# --- Path setup ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR))
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))

from htf_common import (
    compute_htf_indicators,
    compute_volatility_regime,
    compute_prior_day_atr,
    prepare_vscalpc_data,
    map_htf_to_1min,
    resample_to_timeframe,
    HTF_TIMEFRAMES,
    VSCALPC_TP1, VSCALPC_TP2, VSCALPC_SL, VSCALPC_BE_TIME, VSCALPC_SL_TO_BE,
)

from vscalpc_partial_exit_sweep import (
    run_backtest_partial_exit,
    score_partial_trades,
)

from generate_session import (
    VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_ENTRY_END_ET,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
)

from v10_test_common import resample_to_5min, map_5min_rsi_to_1min, NY_CLOSE_ET


# ============================================================================
# Helpers
# ============================================================================

def run_vscalpc(opens, highs, lows, closes, sm, times,
                rsi_curr, rsi_prev, entry_gate=None):
    """Run vScalpC backtest with production config and optional entry gate."""
    return run_backtest_partial_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD,
        cooldown_bars=VSCALPA_COOLDOWN,
        sl_pts=VSCALPC_SL, tp1_pts=VSCALPC_TP1, tp2_pts=VSCALPC_TP2,
        sl_to_be_after_tp1=VSCALPC_SL_TO_BE,
        be_time_bars=VSCALPC_BE_TIME,
        entry_end_et=VSCALPA_ENTRY_END_ET,
        entry_gate=entry_gate,
    )


def print_comparison(label, sc, bl_sc, indent="  "):
    """Print scored result vs baseline."""
    if sc is None:
        print(f"{indent}{label}: NO TRADES")
        return
    blocked = bl_sc['count'] - sc['count'] if bl_sc else 0
    pct_blocked = blocked / bl_sc['count'] * 100 if bl_sc and bl_sc['count'] > 0 else 0
    dpf = ((sc['pf'] - bl_sc['pf']) / bl_sc['pf'] * 100
           if bl_sc and bl_sc['pf'] > 0 else 0)
    dsharpe = sc['sharpe'] - bl_sc['sharpe'] if bl_sc else 0
    dpnl = sc['net_dollar'] - bl_sc['net_dollar'] if bl_sc else 0

    print(f"{indent}{label}: {sc['count']:>4} trades ({blocked:>3} blocked, "
          f"{pct_blocked:.0f}%), WR {sc['win_rate']:>5.1f}%, PF {sc['pf']:>6.3f} "
          f"(dPF {dpf:>+5.1f}%), ${sc['net_dollar']:>+9.2f} (d${dpnl:>+7.0f}), "
          f"Sharpe {sc['sharpe']:>6.3f} (d{dsharpe:>+5.3f}), "
          f"MaxDD ${sc['max_dd_dollar']:>8.2f}")

    # TP2 hit rate
    tp2_hits = sc['leg2_exits'].get('TP2', 0)
    tp2_rate = tp2_hits / sc['count'] * 100 if sc['count'] > 0 else 0
    bl_tp2 = bl_sc['leg2_exits'].get('TP2', 0) if bl_sc else 0
    bl_tp2_rate = bl_tp2 / bl_sc['count'] * 100 if bl_sc and bl_sc['count'] > 0 else 0
    print(f"{indent}  TP2 rate: {tp2_rate:.1f}% (baseline {bl_tp2_rate:.1f}%)")


def assess_pass(sc_is, sc_oos, bl_is, bl_oos):
    """Assess IS/OOS pass/fail for vScalpC gate."""
    if sc_is is None or sc_oos is None:
        return "FAIL", "no trades"
    if sc_is['net_dollar'] <= 0 or sc_oos['net_dollar'] <= 0:
        return "FAIL", f"negative P&L (IS ${sc_is['net_dollar']:+.0f}, OOS ${sc_oos['net_dollar']:+.0f})"

    # PF improvement vs baseline
    is_dpf = ((sc_is['pf'] - bl_is['pf']) / bl_is['pf'] * 100
              if bl_is and bl_is['pf'] > 0 else 0)
    oos_dpf = ((sc_oos['pf'] - bl_oos['pf']) / bl_oos['pf'] * 100
               if bl_oos and bl_oos['pf'] > 0 else 0)

    # Sharpe improvement
    is_dsharpe = sc_is['sharpe'] - bl_is['sharpe'] if bl_is else 0
    oos_dsharpe = sc_oos['sharpe'] - bl_oos['sharpe'] if bl_oos else 0

    if is_dpf < -5 or oos_dpf < -5:
        return "FAIL", f"PF degrades (IS {is_dpf:+.1f}%, OOS {oos_dpf:+.1f}%)"

    if is_dpf >= 5 and oos_dpf >= 5:
        return "STRONG PASS", f"IS PF {is_dpf:+.1f}%, OOS PF {oos_dpf:+.1f}%"

    if is_dpf >= 0 and oos_dpf >= 0 and (is_dsharpe > 0 or oos_dsharpe > 0):
        return "MARGINAL PASS", f"IS PF {is_dpf:+.1f}%, OOS PF {oos_dpf:+.1f}%"

    return "FAIL", f"no improvement (IS {is_dpf:+.1f}%, OOS {oos_dpf:+.1f}%)"


# ============================================================================
# Main Sweep
# ============================================================================

def run_sweep():
    print("=" * 120)
    print("vScalpC Phase 2b — ATR & Volatility Gate Sweep")
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

    # --- Compute HTF indicators ---
    print("\nComputing HTF indicators...")
    htf_indicators = compute_htf_indicators(df)

    # --- Compute prior-day features ---
    print("Computing prior-day ATR and vol regime...")
    vol_regime = compute_volatility_regime(df)
    prior_day_atr = compute_prior_day_atr(df)

    # ========================================================================
    # SWEEP A: Intraday ATR gate
    # ========================================================================
    print(f"\n{'=' * 120}")
    print("SWEEP A: Intraday ATR Gate — block entries when ATR(14) < threshold")
    print(f"{'=' * 120}")

    atr_timeframes = ['5min', '15min', '30min', '1h']
    # Percentile-based thresholds (more robust than absolute values)
    atr_pct_thresholds = [10, 15, 20, 25, 30]

    results_a = []

    for tf in atr_timeframes:
        atr_key = (tf, 'atr')
        atr_values = htf_indicators[atr_key]

        # Compute percentile thresholds from the full ATR distribution
        valid_atr = atr_values[~np.isnan(atr_values)]
        if len(valid_atr) < 100:
            continue

        for pct in atr_pct_thresholds:
            threshold = np.percentile(valid_atr, pct)

            # Gate: allow entry when ATR >= threshold (i.e., not in bottom X%)
            gate_full = atr_values >= threshold
            gate_full[np.isnan(atr_values)] = True  # fail-open for warmup

            gate_is = gate_full[:is_len]
            gate_oos = gate_full[is_len:]

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

            verdict, detail = assess_pass(sc_is, sc_oos, bl_is, bl_oos)

            label = f"{tf} ATR p{pct} (>={threshold:.1f})"
            results_a.append({
                'label': label, 'tf': tf, 'pct': pct, 'threshold': threshold,
                'sc_full': sc_full, 'sc_is': sc_is, 'sc_oos': sc_oos,
                'verdict': verdict, 'detail': detail,
            })

            marker = " <<<" if "PASS" in verdict else ""
            print(f"\n  {label}  [{verdict}]{marker}")
            print_comparison("FULL", sc_full, bl_full, indent="    ")
            print_comparison("IS  ", sc_is, bl_is, indent="    ")
            print_comparison("OOS ", sc_oos, bl_oos, indent="    ")

    # ========================================================================
    # SWEEP B: Prior-day ATR gate
    # ========================================================================
    print(f"\n{'=' * 120}")
    print("SWEEP B: Prior-Day ATR Gate — block entries when prior-day ATR < threshold")
    print(f"{'=' * 120}")

    valid_pda = prior_day_atr[~np.isnan(prior_day_atr)]
    pda_pct_thresholds = [10, 15, 20, 25, 30, 35]

    results_b = []
    for pct in pda_pct_thresholds:
        threshold = np.percentile(valid_pda, pct)
        gate_full = prior_day_atr >= threshold
        gate_full[np.isnan(prior_day_atr)] = True

        gate_is = gate_full[:is_len]
        gate_oos = gate_full[is_len:]

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

        verdict, detail = assess_pass(sc_is, sc_oos, bl_is, bl_oos)

        label = f"PriorDay ATR p{pct} (>={threshold:.1f})"
        results_b.append({
            'label': label, 'pct': pct, 'threshold': threshold,
            'sc_full': sc_full, 'sc_is': sc_is, 'sc_oos': sc_oos,
            'verdict': verdict, 'detail': detail,
        })

        marker = " <<<" if "PASS" in verdict else ""
        print(f"\n  {label}  [{verdict}]{marker}")
        print_comparison("FULL", sc_full, bl_full, indent="    ")
        print_comparison("IS  ", sc_is, bl_is, indent="    ")
        print_comparison("OOS ", sc_oos, bl_oos, indent="    ")

    # ========================================================================
    # SWEEP C: Vol regime gate
    # ========================================================================
    print(f"\n{'=' * 120}")
    print("SWEEP C: Vol Regime Gate — block entries on Low-vol days (prior-day classification)")
    print(f"{'=' * 120}")

    results_c = []
    for blocked_regimes, label in [
        ([0], "Block Low"),
        ([0, 1], "Block Low+Medium (High only)"),
    ]:
        gate_full = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            if not np.isnan(vol_regime[i]) and int(vol_regime[i]) in blocked_regimes:
                gate_full[i] = False

        gate_is = gate_full[:is_len]
        gate_oos = gate_full[is_len:]

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

        verdict, detail = assess_pass(sc_is, sc_oos, bl_is, bl_oos)

        results_c.append({
            'label': label, 'blocked': blocked_regimes,
            'sc_full': sc_full, 'sc_is': sc_is, 'sc_oos': sc_oos,
            'verdict': verdict, 'detail': detail,
        })

        marker = " <<<" if "PASS" in verdict else ""
        print(f"\n  {label}  [{verdict}]{marker}")
        print_comparison("FULL", sc_full, bl_full, indent="    ")
        print_comparison("IS  ", sc_is, bl_is, indent="    ")
        print_comparison("OOS ", sc_oos, bl_oos, indent="    ")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print(f"\n{'=' * 120}")
    print("SUMMARY — All Configs Sorted by OOS Sharpe")
    print(f"{'=' * 120}")

    all_results = results_a + results_b + results_c
    # Sort by OOS Sharpe (descending)
    all_results.sort(key=lambda r: r['sc_oos']['sharpe'] if r['sc_oos'] else -999,
                     reverse=True)

    print(f"\n  {'Label':>35} {'Verdict':>15} | "
          f"{'FULL N':>6} {'FULL PF':>7} {'FULL$':>9} {'FULL Sh':>7} | "
          f"{'IS PF':>6} {'IS$':>8} {'IS Sh':>6} | "
          f"{'OOS PF':>7} {'OOS$':>8} {'OOS Sh':>7}")
    print(f"  " + "-" * 140)

    # Baseline first
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

    print(f"\n{'=' * 120}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    run_sweep()
