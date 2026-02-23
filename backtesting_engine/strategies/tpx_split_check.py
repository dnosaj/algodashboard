"""
TPX Regime Gate — Split-Half Stability Check (vScalpB)
=======================================================
Tests whether L=30 S=12 TPX regime gate works on BOTH halves of the data
independently, using the SAME parameters (no re-optimization).

This is NOT parameter selection — we already picked L=30 S=12 from the
full-data sweep. This checks if that setting is stable across different
market regimes, or if it only worked in one half.

Usage:
    python3 tpx_split_check.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
)
from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
    compute_mfe_mae,
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    VSCALPB_RSI_LEN, VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
    VSCALPB_SM_THRESHOLD, VSCALPB_COOLDOWN,
    VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
)
from compute_tpx import compute_tpx
from tpx_regime_test import run_backtest_tpx_regime


def run_split(label, df, dollar_per_pt, commission):
    """Run baseline and TPX-gated vScalpB on a data slice."""
    sm = compute_smart_money(
        df["Close"].values, df["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df = df.copy()
    df["SM_Net"] = sm

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    times = df.index

    df_5m = resample_to_5min(df)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values,
        df_5m["Close"].values, rsi_len=VSCALPB_RSI_LEN,
    )

    # TPX L=30 S=12
    tpx, _, _ = compute_tpx(highs, lows, length=30, smooth=12)

    # Baseline
    baseline = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
    )
    compute_mfe_mae(baseline, highs, lows)
    sc_base = score_trades(baseline, commission_per_side=commission,
                           dollar_per_pt=dollar_per_pt)

    # TPX gated
    gated, blocked = run_backtest_tpx_regime(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev, tpx,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
    )
    compute_mfe_mae(gated, highs, lows)
    sc_gated = score_trades(gated, commission_per_side=commission,
                            dollar_per_pt=dollar_per_pt)

    return sc_base, sc_gated, blocked


def print_comparison(label, sc_base, sc_gated, blocked):
    """Print side-by-side comparison."""
    print(f"\n  --- {label} ---")
    print(f"  Baseline:  {fmt_score(sc_base)}")
    print(f"  TPX gated: {fmt_score(sc_gated)}")
    print(f"  Blocked: {blocked} entries")

    if sc_base and sc_gated:
        print(f"\n  {'Metric':<16s} {'Baseline':>10s} {'TPX Gate':>10s} {'Delta':>10s} {'Better?':>8s}")
        print(f"  {'-'*58}")

        comparisons = [
            ("Trades",   sc_base["count"],      sc_gated["count"],      None),
            ("Win Rate", sc_base["win_rate"],    sc_gated["win_rate"],   True),
            ("PF",       sc_base["pf"],          sc_gated["pf"],         True),
            ("Net $",    sc_base["net_dollar"],  sc_gated["net_dollar"], True),
            ("MaxDD $",  sc_base["max_dd_dollar"], sc_gated["max_dd_dollar"], False),
            ("Sharpe",   sc_base["sharpe"],      sc_gated["sharpe"],     True),
            ("Avg pts",  sc_base["avg_pts"],     sc_gated["avg_pts"],    True),
        ]

        for name, base, gate, higher_is_better in comparisons:
            delta = gate - base
            if higher_is_better is None:
                marker = ""
            elif higher_is_better:
                marker = "YES" if delta > 0 else "no"
            else:
                # For MaxDD, less negative is better
                marker = "YES" if delta > 0 else "no"

            print(f"  {name:<16s} {base:>10.2f} {gate:>10.2f} {delta:>+10.2f} {marker:>8s}")


def main():
    print("=" * 75)
    print("  vScalpB — TPX L=30 S=12 Regime Gate — Split-Half Stability Check")
    print("=" * 75)

    print("\nLoading MNQ data...")
    df_mnq = load_instrument_1min("MNQ")
    print(f"  {len(df_mnq)} bars: {df_mnq.index[0]} to {df_mnq.index[-1]}")

    # Split at midpoint
    midpoint = df_mnq.index[len(df_mnq) // 2]
    df_is = df_mnq[df_mnq.index < midpoint].copy()
    df_oos = df_mnq[df_mnq.index >= midpoint].copy()

    print(f"\n  IS  (first half):  {len(df_is):>7d} bars, "
          f"{df_is.index[0].strftime('%Y-%m-%d')} to {df_is.index[-1].strftime('%Y-%m-%d')}")
    print(f"  OOS (second half): {len(df_oos):>7d} bars, "
          f"{df_oos.index[0].strftime('%Y-%m-%d')} to {df_oos.index[-1].strftime('%Y-%m-%d')}")

    # Run on each half
    sc_base_full, sc_gated_full, blocked_full = run_split(
        "FULL", df_mnq, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
    sc_base_is, sc_gated_is, blocked_is = run_split(
        "IS", df_is, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
    sc_base_oos, sc_gated_oos, blocked_oos = run_split(
        "OOS", df_oos, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)

    print_comparison("FULL (12 months)", sc_base_full, sc_gated_full, blocked_full)
    print_comparison("IS (first 6 months)", sc_base_is, sc_gated_is, blocked_is)
    print_comparison("OOS (second 6 months)", sc_base_oos, sc_gated_oos, blocked_oos)

    # Verdict
    print(f"\n{'='*75}")
    print(f"  STABILITY VERDICT")
    print(f"{'='*75}")

    if sc_gated_is and sc_gated_oos and sc_base_is and sc_base_oos:
        is_pf_improved = sc_gated_is["pf"] > sc_base_is["pf"]
        oos_pf_improved = sc_gated_oos["pf"] > sc_base_oos["pf"]
        is_sharpe_improved = sc_gated_is["sharpe"] > sc_base_is["sharpe"]
        oos_sharpe_improved = sc_gated_oos["sharpe"] > sc_base_oos["sharpe"]
        is_dd_improved = sc_gated_is["max_dd_dollar"] > sc_base_is["max_dd_dollar"]
        oos_dd_improved = sc_gated_oos["max_dd_dollar"] > sc_base_oos["max_dd_dollar"]

        print(f"  PF improved on IS:     {'YES' if is_pf_improved else 'NO'} "
              f"({sc_base_is['pf']:.3f} -> {sc_gated_is['pf']:.3f})")
        print(f"  PF improved on OOS:    {'YES' if oos_pf_improved else 'NO'} "
              f"({sc_base_oos['pf']:.3f} -> {sc_gated_oos['pf']:.3f})")
        print(f"  Sharpe improved on IS: {'YES' if is_sharpe_improved else 'NO'} "
              f"({sc_base_is['sharpe']:.3f} -> {sc_gated_is['sharpe']:.3f})")
        print(f"  Sharpe improved on OOS:{'YES' if oos_sharpe_improved else 'NO'} "
              f"({sc_base_oos['sharpe']:.3f} -> {sc_gated_oos['sharpe']:.3f})")
        print(f"  MaxDD improved on IS:  {'YES' if is_dd_improved else 'NO'} "
              f"({sc_base_is['max_dd_dollar']:.2f} -> {sc_gated_is['max_dd_dollar']:.2f})")
        print(f"  MaxDD improved on OOS: {'YES' if oos_dd_improved else 'NO'} "
              f"({sc_base_oos['max_dd_dollar']:.2f} -> {sc_gated_oos['max_dd_dollar']:.2f})")

        both_pf = is_pf_improved and oos_pf_improved
        both_sharpe = is_sharpe_improved and oos_sharpe_improved
        print(f"\n  PF stable across both halves:     {'PASS' if both_pf else 'FAIL'}")
        print(f"  Sharpe stable across both halves:  {'PASS' if both_sharpe else 'FAIL'}")


if __name__ == "__main__":
    main()
