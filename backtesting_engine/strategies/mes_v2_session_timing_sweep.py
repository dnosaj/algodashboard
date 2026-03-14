"""
MES v2 Session Timing Sweep — IS/OOS
=====================================
Tests both session_close_et (force close) and session_end_et (last entry).

Context: session_close_et=15:30 was validated on v9.4 (SM flip exit) but never
re-validated for v2 (TP=20 + partial exit). session_end_et defaults to 15:45
which is AFTER the 15:30 close — structurally broken.

Sweep:
  - EOD close times: 15:00, 15:15, 15:30 (current), 15:45, 16:00
  - For each close time, entry cutoffs: close-60, close-45, close-30, close-15, close
  - Plus time-of-day bucket diagnostics
"""

import sys
from pathlib import Path

import numpy as np

_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR))

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
from generate_session import run_backtest_tp_exit, compute_mfe_mae


# MES_V2 params (matches live config.py)
MES_V2 = {
    "sm_index": 20, "sm_flow": 12, "sm_norm": 400, "sm_ema": 255,
    "rsi_len": 12, "rsi_buy": 55, "rsi_sell": 45, "sm_threshold": 0.0,
    "cooldown": 25, "max_loss_pts": 35, "tp_pts": 20,
    "breakeven_bars": 75,
    "dollar_per_pt": 5.0, "commission": 1.25,
}

# Sweep parameters
EOD_CLOSE_TIMES = [
    (900, "15:00"),
    (915, "15:15"),
    (930, "15:30"),  # current
    (945, "15:45"),
    (960, "16:00"),
]

# Entry cutoff offsets (minutes before EOD close)
ENTRY_OFFSETS = [60, 45, 30, 15, 0]  # 0 = same as close time


def prepare_mes_data():
    """Load MES, compute SM, split IS/OOS."""
    df = load_instrument_1min("MES")
    sm = compute_smart_money(
        df["Close"].values, df["Volume"].values,
        index_period=MES_V2["sm_index"], flow_period=MES_V2["sm_flow"],
        norm_period=MES_V2["sm_norm"], ema_len=MES_V2["sm_ema"],
    )
    df["SM_Net"] = sm
    print(f"Loaded MES: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Split at midpoint
    mid = df.index[len(df) // 2]
    is_len = (df.index < mid).sum()
    print(f"  Split: IS {is_len} bars, OOS {len(df)-is_len} bars")

    # RSI
    df_5m = resample_to_5min(df)
    rsi_curr_full, rsi_prev_full = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values,
        df_5m["Close"].values, rsi_len=MES_V2["rsi_len"],
    )

    splits = {}
    for name, df_slice, rsi_c, rsi_p in [
        ("FULL", df, rsi_curr_full, rsi_prev_full),
        ("IS", df.iloc[:is_len], rsi_curr_full[:is_len], rsi_prev_full[:is_len]),
        ("OOS", df.iloc[is_len:], rsi_curr_full[is_len:], rsi_prev_full[is_len:]),
    ]:
        splits[name] = (
            df_slice["Open"].values, df_slice["High"].values,
            df_slice["Low"].values, df_slice["Close"].values,
            df_slice["SM_Net"].values, df_slice.index,
            rsi_c, rsi_p,
        )

    return splits


def run_mes(splits, split_name, eod_et, entry_end_et):
    """Run MES_V2 backtest with given timing params."""
    arrays = splits[split_name]
    trades = run_backtest_tp_exit(
        *arrays,
        rsi_buy=MES_V2["rsi_buy"], rsi_sell=MES_V2["rsi_sell"],
        sm_threshold=MES_V2["sm_threshold"], cooldown_bars=MES_V2["cooldown"],
        max_loss_pts=MES_V2["max_loss_pts"], tp_pts=MES_V2["tp_pts"],
        eod_minutes_et=eod_et,
        breakeven_after_bars=MES_V2["breakeven_bars"],
        entry_end_et=entry_end_et,
        entry_gate=None,
    )
    compute_mfe_mae(trades, arrays[1], arrays[2])
    return trades


def time_bucket_diagnostic(splits):
    """Analyze baseline trade performance by entry time bucket."""
    print(f"\n{'='*100}")
    print("DIAGNOSTIC: MES_V2 Baseline — Performance by Entry Time Bucket (FULL period)")
    print(f"{'='*100}")

    # Run baseline (current settings: eod=930, entry_end=NY_LAST_ENTRY_ET)
    trades = run_mes(splits, "FULL", 930, NY_LAST_ENTRY_ET)

    # Get ET minutes for each trade entry
    times = splits["FULL"][5]  # times array
    et_minutes = compute_et_minutes(times)

    # Bucket trades by entry time (30-min buckets)
    buckets = {}
    for t in trades:
        entry_bar = t["entry_idx"]
        entry_et = et_minutes[entry_bar]
        # Round down to 30-min bucket
        bucket = (entry_et // 30) * 30
        h, m = divmod(int(bucket), 60)
        bucket_label = f"{h:02d}:{m:02d}"
        if bucket_label not in buckets:
            buckets[bucket_label] = []
        pnl_net = t["pts"] * MES_V2["dollar_per_pt"] - 2 * MES_V2["commission"]
        buckets[bucket_label].append({
            "pnl": pnl_net,
            "exit_reason": t.get("exit_reason", "unknown"),
            "pts": t["pts"],
        })

    print(f"\n  {'Bucket':>8} {'Trades':>7} {'WR%':>6} {'Net$':>10} {'Avg$/tr':>9} {'PF':>7} {'EOD_exits':>10}")
    print(f"  {'-'*65}")

    for bucket_label in sorted(buckets.keys()):
        trades_b = buckets[bucket_label]
        n = len(trades_b)
        winners = sum(1 for t in trades_b if t["pnl"] > 0)
        wr = winners / n * 100 if n > 0 else 0
        net = sum(t["pnl"] for t in trades_b)
        avg = net / n if n > 0 else 0
        gross_win = sum(t["pnl"] for t in trades_b if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades_b if t["pnl"] < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        eod_exits = sum(1 for t in trades_b if t.get("exit_reason") == "eod")
        print(f"  {bucket_label:>8} {n:>7} {wr:>5.1f}% {net:>+10.2f} {avg:>+9.2f} {pf:>7.3f} {eod_exits:>10}")

    # Also show EOD exit analysis
    print(f"\n{'='*100}")
    print("DIAGNOSTIC: EOD Exits — by Entry Time (how much does force-close cost?)")
    print(f"{'='*100}")

    eod_trades = [t for t in trades if t.get("exit_reason") == "eod"]
    print(f"\n  Total EOD exits: {len(eod_trades)} of {len(trades)} ({len(eod_trades)/len(trades)*100:.1f}%)")

    eod_by_entry = {}
    for t in eod_trades:
        entry_bar = t["entry_idx"]
        entry_et = et_minutes[entry_bar]
        bucket = (entry_et // 30) * 30
        h, m = divmod(int(bucket), 60)
        bucket_label = f"{h:02d}:{m:02d}"
        pnl_net = t["pts"] * MES_V2["dollar_per_pt"] - 2 * MES_V2["commission"]
        if bucket_label not in eod_by_entry:
            eod_by_entry[bucket_label] = []
        eod_by_entry[bucket_label].append(pnl_net)

    print(f"\n  {'Entry Bucket':>12} {'EOD Exits':>10} {'Net$':>10} {'Avg$/tr':>9} {'Winners':>8}")
    print(f"  {'-'*55}")
    for bucket_label in sorted(eod_by_entry.keys()):
        pnls = eod_by_entry[bucket_label]
        n = len(pnls)
        net = sum(pnls)
        avg = net / n if n > 0 else 0
        winners = sum(1 for p in pnls if p > 0)
        print(f"  {bucket_label:>12} {n:>10} {net:>+10.2f} {avg:>+9.2f} {winners:>8}")


def main():
    splits = prepare_mes_data()

    # --- Diagnostic first ---
    time_bucket_diagnostic(splits)

    # --- Main sweep ---
    print(f"\n\n{'='*140}")
    print("MES_V2 SESSION TIMING SWEEP (IS/OOS)")
    print(f"{'='*140}")
    print(f"  Testing: EOD close × entry cutoff combos")
    print(f"  EOD close times: {', '.join(t[1] for t in EOD_CLOSE_TIMES)}")
    print(f"  Entry offsets before close: {ENTRY_OFFSETS} minutes")

    # Run baseline first (current: eod=930, entry_end=NY_LAST_ENTRY_ET)
    baselines = {}
    for split_name in ["IS", "OOS"]:
        trades = run_mes(splits, split_name, 930, NY_LAST_ENTRY_ET)
        sc = score_trades(trades, commission_per_side=MES_V2["commission"],
                          dollar_per_pt=MES_V2["dollar_per_pt"])
        baselines[split_name] = sc

    print(f"\n  Baseline (EOD=15:30, entry=15:45):")
    for sp in ["IS", "OOS"]:
        sc = baselines[sp]
        if sc:
            print(f"    {sp}: {sc['count']} trades, WR {sc['win_rate']:.1f}%, "
                  f"PF {sc['pf']:.3f}, ${sc['net_dollar']:+.2f}, Sharpe {sc['sharpe']:.3f}")

    # Header
    print(f"\n  {'EOD':>6} {'Entry':>6} {'Split':>5} {'Trades':>7} {'WR%':>6} "
          f"{'PF':>7} {'Net$':>10} {'MaxDD':>9} {'Sharpe':>7} "
          f"{'dTrades':>8} {'dPF%':>7} {'dSharpe':>8}")
    print(f"  {'-'*105}")

    # Collect results for summary
    all_results = {}

    for eod_mins, eod_label in EOD_CLOSE_TIMES:
        for offset in ENTRY_OFFSETS:
            entry_end = eod_mins - offset
            if entry_end < 600:  # Don't go before 10:00
                continue

            eh, em = divmod(entry_end, 60)
            entry_label = f"{eh:02d}:{em:02d}"
            combo_key = f"{eod_label}/{entry_label}"

            is_current = (eod_mins == 930 and entry_end == NY_LAST_ENTRY_ET)

            for split_name in ["IS", "OOS"]:
                trades = run_mes(splits, split_name, eod_mins, entry_end)
                sc = score_trades(trades, commission_per_side=MES_V2["commission"],
                                  dollar_per_pt=MES_V2["dollar_per_pt"])

                all_results[(combo_key, split_name)] = sc

                if sc is None:
                    print(f"  {eod_label:>6} {entry_label:>6} {split_name:>5}   NO TRADES")
                    continue

                bl = baselines[split_name]
                d_trades = sc["count"] - bl["count"]
                d_pf = (sc["pf"] - bl["pf"]) / bl["pf"] * 100 if bl["pf"] > 0 else 0
                d_sharpe = sc["sharpe"] - bl["sharpe"]
                marker = " <<<" if is_current else ""

                print(f"  {eod_label:>6} {entry_label:>6} {split_name:>5} {sc['count']:>7} "
                      f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                      f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
                      f"{d_trades:>+8} {d_pf:>+6.1f}% {d_sharpe:>+8.3f}{marker}")

    # --- Summary table with verdicts ---
    print(f"\n\n{'='*120}")
    print("SUMMARY — IS/OOS Verdicts (vs current baseline EOD=15:30 / Entry=15:45)")
    print(f"{'='*120}")
    print(f"  {'EOD':>6} {'Entry':>6} {'IS PF':>7} {'OOS PF':>7} {'IS Shrp':>8} "
          f"{'OOS Shrp':>9} {'OOS Net$':>10} {'OOS Trades':>11} {'dOOS_PF%':>9} {'Verdict':>15}")
    print(f"  {'-'*100}")

    for eod_mins, eod_label in EOD_CLOSE_TIMES:
        for offset in ENTRY_OFFSETS:
            entry_end = eod_mins - offset
            if entry_end < 600:
                continue

            eh, em = divmod(entry_end, 60)
            entry_label = f"{eh:02d}:{em:02d}"
            combo_key = f"{eod_label}/{entry_label}"

            sc_is = all_results.get((combo_key, "IS"))
            sc_oos = all_results.get((combo_key, "OOS"))

            if not sc_is or not sc_oos:
                continue

            bl_is = baselines["IS"]
            bl_oos = baselines["OOS"]

            # Verdict
            # Net dollar must be positive in both halves
            if sc_is["net_dollar"] <= 0 or sc_oos["net_dollar"] <= 0:
                verdict = "FAIL (neg P&L)"
            elif sc_oos["count"] < 30:
                verdict = "FAIL (low N)"
            else:
                oos_dpf = (sc_oos["pf"] - bl_oos["pf"]) / bl_oos["pf"] * 100 if bl_oos["pf"] > 0 else 0
                is_dpf = (sc_is["pf"] - bl_is["pf"]) / bl_is["pf"] * 100 if bl_is["pf"] > 0 else 0
                oos_dsharpe = sc_oos["sharpe"] - bl_oos["sharpe"]
                is_dsharpe = sc_is["sharpe"] - bl_is["sharpe"]

                if is_dpf < -5 or oos_dpf < -5:
                    verdict = "FAIL"
                elif is_dpf >= 5 and oos_dpf >= 5 and is_dsharpe >= 0 and oos_dsharpe >= 0:
                    verdict = "STRONG PASS"
                elif is_dpf >= -5 and oos_dpf >= -5 and (is_dsharpe > 0 or oos_dsharpe > 0):
                    verdict = "MARGINAL PASS"
                else:
                    verdict = "FAIL"

            is_current = (eod_mins == 930 and entry_end == NY_LAST_ENTRY_ET)
            oos_dpf = (sc_oos["pf"] - bl_oos["pf"]) / bl_oos["pf"] * 100 if bl_oos["pf"] > 0 else 0
            marker = " <<<" if is_current else ""

            print(f"  {eod_label:>6} {entry_label:>6} {sc_is['pf']:>7.3f} {sc_oos['pf']:>7.3f} "
                  f"{sc_is['sharpe']:>8.3f} {sc_oos['sharpe']:>9.3f} "
                  f"{sc_oos['net_dollar']:>+10.2f} {sc_oos['count']:>11} "
                  f"{oos_dpf:>+8.1f}% {verdict:>15}{marker}")

        # Visual separator between EOD groups
        print()

    print("Done.")


if __name__ == "__main__":
    main()
