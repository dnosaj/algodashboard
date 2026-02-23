"""
TPX / SM Agreement Filter Test
================================
Tests the hypothesis: when RedK TPX and Smart Money disagree at entry time,
trades fail more often.

For each trade, checks the previous bar's signals (no look-ahead):
  - SM direction (what triggered the entry)
  - TPX sign (>0 bullish, <0 bearish)
  - Agreement: LONG entry with TPX>0, or SHORT entry with TPX<0
  - Disagreement: LONG entry with TPX<0, or SHORT entry with TPX>0

Usage:
    python3 tpx_filter_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS,
    VSCALPB_RSI_LEN, VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
    VSCALPB_SM_THRESHOLD, VSCALPB_COOLDOWN,
    VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
    MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA,
    MES_DOLLAR_PER_PT, MES_COMMISSION,
    MESV2_RSI_LEN, MESV2_RSI_BUY, MESV2_RSI_SELL,
    MESV2_SM_THRESHOLD, MESV2_COOLDOWN,
    MESV2_MAX_LOSS_PTS, MESV2_TP_PTS, MESV2_EOD_ET,
)
from compute_tpx import compute_tpx


def analyze_tpx_filter(trades, tpx, sm, strategy_name,
                       dollar_per_pt, commission):
    """Classify trades by TPX/SM agreement and compare outcomes."""
    agree_trades = []
    disagree_trades = []
    neutral_trades = []  # TPX exactly 0 (rare)

    for t in trades:
        entry_idx = t["entry_idx"]
        # Use previous bar's TPX (same convention as SM — no look-ahead)
        if entry_idx < 1:
            continue
        tpx_prev = tpx[entry_idx - 1]
        side = t["side"]

        if side == "long":
            if tpx_prev > 0:
                agree_trades.append(t)
            elif tpx_prev < 0:
                disagree_trades.append(t)
            else:
                neutral_trades.append(t)
        else:  # short
            if tpx_prev < 0:
                agree_trades.append(t)
            elif tpx_prev > 0:
                disagree_trades.append(t)
            else:
                neutral_trades.append(t)

    # Score each group
    sc_all = score_trades(trades, commission_per_side=commission,
                          dollar_per_pt=dollar_per_pt)
    sc_agree = score_trades(agree_trades, commission_per_side=commission,
                            dollar_per_pt=dollar_per_pt)
    sc_disagree = score_trades(disagree_trades, commission_per_side=commission,
                               dollar_per_pt=dollar_per_pt)

    print(f"\n{'='*75}")
    print(f"  {strategy_name} — TPX Filter Analysis (default: length=7, smooth=3)")
    print(f"{'='*75}")
    print(f"  {fmt_score(sc_all, 'ALL TRADES')}")
    print()
    print(f"  {fmt_score(sc_agree, 'SM+TPX AGREE')}")
    print(f"  {fmt_score(sc_disagree, 'SM+TPX DISAGREE')}")
    if neutral_trades:
        sc_neutral = score_trades(neutral_trades, commission_per_side=commission,
                                  dollar_per_pt=dollar_per_pt)
        print(f"  {fmt_score(sc_neutral, 'TPX NEUTRAL (=0)')}")

    # Detailed comparison
    if sc_agree and sc_disagree:
        print(f"\n  {'Metric':<20s} {'Agree':>12s} {'Disagree':>12s} {'Delta':>12s}")
        print(f"  {'-'*56}")

        metrics = [
            ("Trades", sc_agree["count"], sc_disagree["count"], ""),
            ("Win Rate %", sc_agree["win_rate"], sc_disagree["win_rate"], "%"),
            ("Profit Factor", sc_agree["pf"], sc_disagree["pf"], ""),
            ("Net P&L $", sc_agree["net_dollar"], sc_disagree["net_dollar"], "$"),
            ("Avg pts/trade", sc_agree["avg_pts"], sc_disagree["avg_pts"], "pt"),
            ("MaxDD $", sc_agree["max_dd_dollar"], sc_disagree["max_dd_dollar"], "$"),
            ("Sharpe", sc_agree["sharpe"], sc_disagree["sharpe"], ""),
        ]

        for name, a, d, unit in metrics:
            delta = a - d if isinstance(a, (int, float)) else ""
            if isinstance(delta, float):
                print(f"  {name:<20s} {a:>12.2f} {d:>12.2f} {delta:>+12.2f}")
            else:
                print(f"  {name:<20s} {a:>12} {d:>12}")

        # What-if: skip disagreement trades
        skip_pnl = sc_agree["net_dollar"]
        all_pnl = sc_all["net_dollar"]
        delta = skip_pnl - all_pnl
        pct_skipped = sc_disagree["count"] / sc_all["count"] * 100

        print(f"\n  WHAT-IF: Skip disagreement trades")
        print(f"    Trades skipped: {sc_disagree['count']} ({pct_skipped:.1f}%)")
        print(f"    P&L keeping only agreement: ${skip_pnl:+.2f} "
              f"(was ${all_pnl:+.2f}, delta ${delta:+.2f})")

    # Exit reason breakdown for disagree trades
    if disagree_trades:
        print(f"\n  Disagree trades — exit reasons:")
        exits = {}
        for t in disagree_trades:
            r = t.get("result", "?")
            exits[r] = exits.get(r, 0) + 1
        for r, c in sorted(exits.items(), key=lambda x: -x[1]):
            print(f"    {r}: {c}")

    # Worst disagreement trades
    if disagree_trades:
        comm_pts = (commission * 2) / dollar_per_pt
        sorted_d = sorted(disagree_trades,
                          key=lambda t: (t["pts"] - comm_pts) * dollar_per_pt)
        print(f"\n  Worst 10 disagreement trades:")
        print(f"  {'Side':<6s} {'Entry':>10s} {'Exit':>10s} {'Pts':>7s} {'P&L':>9s} "
              f"{'TPX':>6s} {'SM':>6s} {'Exit':>5s}")
        print(f"  {'-'*62}")
        for t in sorted_d[:10]:
            pnl = (t["pts"] - comm_pts) * dollar_per_pt
            tpx_val = tpx[t["entry_idx"] - 1]
            sm_val = sm[t["entry_idx"] - 1]
            print(f"  {t['side']:<6s} {t['entry']:>10.2f} {t['exit']:>10.2f} "
                  f"{t['pts']:>+7.2f} ${pnl:>+8.2f} {tpx_val:>+6.1f} {sm_val:>+6.3f} "
                  f"{t.get('result', '?'):>5s}")

    return {
        "strategy": strategy_name,
        "all": sc_all,
        "agree": sc_agree,
        "disagree": sc_disagree,
        "n_agree": len(agree_trades),
        "n_disagree": len(disagree_trades),
    }


def main():
    print("=" * 75)
    print("  TPX / SM AGREEMENT FILTER TEST")
    print("  Hypothesis: SM+TPX disagreement → worse trades")
    print("=" * 75)

    # --- Load MNQ ---
    print("\nLoading MNQ data...")
    df_mnq = load_instrument_1min("MNQ")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm
    print(f"  {len(df_mnq)} bars")

    # --- Load MES ---
    print("Loading MES data...")
    df_mes = load_instrument_1min("MES")
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm
    print(f"  {len(df_mes)} bars")

    # --- Compute TPX ---
    print("\nComputing TPX (default: length=7, smooth=3)...")
    tpx_mnq, _, _ = compute_tpx(df_mnq["High"].values, df_mnq["Low"].values,
                                  length=7, smooth=3)
    tpx_mes, _, _ = compute_tpx(df_mes["High"].values, df_mes["Low"].values,
                                  length=7, smooth=3)
    print(f"  MNQ TPX: mean={np.mean(tpx_mnq):.1f}, "
          f"{np.sum(tpx_mnq > 0)/len(tpx_mnq)*100:.0f}% bullish")
    print(f"  MES TPX: mean={np.mean(tpx_mes):.1f}, "
          f"{np.sum(tpx_mes > 0)/len(tpx_mes)*100:.0f}% bullish")

    # --- MNQ RSI (shared by vScalpA and vScalpB, both use len=8) ---
    df_mnq_5m = resample_to_5min(df_mnq)
    rsi_mnq_curr, rsi_mnq_prev = map_5min_rsi_to_1min(
        df_mnq.index.values, df_mnq_5m.index.values,
        df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN,
    )

    # --- vScalpA ---
    mnq_arrays = (df_mnq["Open"].values, df_mnq["High"].values,
                  df_mnq["Low"].values, df_mnq["Close"].values,
                  df_mnq["SM_Net"].values, df_mnq.index)

    vscalpa_trades = run_backtest_tp_exit(
        *mnq_arrays, rsi_mnq_curr, rsi_mnq_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPA_MAX_LOSS_PTS, tp_pts=VSCALPA_TP_PTS,
    )
    compute_mfe_mae(vscalpa_trades, df_mnq["High"].values, df_mnq["Low"].values)

    r_vscalpa = analyze_tpx_filter(
        vscalpa_trades, tpx_mnq, mnq_sm,
        "vScalpA (MNQ v15)", MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    )

    # --- vScalpB ---
    vscalpb_trades = run_backtest_tp_exit(
        *mnq_arrays, rsi_mnq_curr, rsi_mnq_prev,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
    )
    compute_mfe_mae(vscalpb_trades, df_mnq["High"].values, df_mnq["Low"].values)

    r_vscalpb = analyze_tpx_filter(
        vscalpb_trades, tpx_mnq, mnq_sm,
        "vScalpB (MNQ)", MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    )

    # --- MES v2 ---
    mes_arrays = (df_mes["Open"].values, df_mes["High"].values,
                  df_mes["Low"].values, df_mes["Close"].values,
                  df_mes["SM_Net"].values, df_mes.index)

    df_mes_5m = resample_to_5min(df_mes)
    rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
        df_mes.index.values, df_mes_5m.index.values,
        df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN,
    )

    mesv2_trades = run_backtest_tp_exit(
        *mes_arrays, rsi_mes_curr, rsi_mes_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
    )
    compute_mfe_mae(mesv2_trades, df_mes["High"].values, df_mes["Low"].values)

    r_mesv2 = analyze_tpx_filter(
        mesv2_trades, tpx_mes, mes_sm,
        "MES v2", MES_DOLLAR_PER_PT, MES_COMMISSION,
    )

    # --- Portfolio summary ---
    print(f"\n{'='*75}")
    print(f"  PORTFOLIO SUMMARY")
    print(f"{'='*75}")

    results = [r_vscalpa, r_vscalpb, r_mesv2]

    total_all = sum(r["all"]["net_dollar"] for r in results if r["all"])
    total_agree = sum(r["agree"]["net_dollar"] for r in results if r["agree"])
    total_disagree = sum(r["disagree"]["net_dollar"] for r in results if r["disagree"])
    n_all = sum(r["all"]["count"] for r in results if r["all"])
    n_agree = sum(r["n_agree"] for r in results)
    n_disagree = sum(r["n_disagree"] for r in results)

    print(f"  All trades:         {n_all:>5d} trades  ${total_all:>+10.2f}")
    print(f"  SM+TPX agree:       {n_agree:>5d} trades  ${total_agree:>+10.2f}  "
          f"(avg ${total_agree/n_agree:>+.2f}/trade)")
    print(f"  SM+TPX disagree:    {n_disagree:>5d} trades  ${total_disagree:>+10.2f}  "
          f"(avg ${total_disagree/n_disagree:>+.2f}/trade)")
    print(f"\n  Filter impact:      skip {n_disagree} trades ({n_disagree/n_all*100:.1f}%), "
          f"delta ${total_agree - total_all:>+.2f}")


if __name__ == "__main__":
    main()
