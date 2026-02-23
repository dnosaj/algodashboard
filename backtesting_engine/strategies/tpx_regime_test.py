"""
TPX Regime Gate Test
=====================
Tests TPX zero-cross as a directional regime filter:
  - Only allow LONG entries when TPX > 0 (green zone)
  - Only allow SHORT entries when TPX < 0 (red zone)

This is a PRE-FILTER test: entries blocked by TPX disagreement are skipped
entirely, which changes cooldowns and episode timing (unlike a post-filter).

Tests multiple TPX parameter combos to find the best regime gate settings.

Usage:
    python3 tpx_regime_test.py
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
    compute_et_minutes,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)
from generate_session import (
    load_instrument_1min,
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


def run_backtest_tpx_regime(opens, highs, lows, closes, sm, times,
                            rsi_5m_curr, rsi_5m_prev, tpx,
                            rsi_buy, rsi_sell, sm_threshold,
                            cooldown_bars, max_loss_pts, tp_pts,
                            eod_minutes_et=NY_CLOSE_ET):
    """Backtest with TPX regime gate as a PRE-FILTER.

    Same as run_backtest_tp_exit but adds:
      - LONG entry blocked if tpx[i-1] <= 0
      - SHORT entry blocked if tpx[i-1] >= 0

    Blocked entries still consume the episode flag (long_used/short_used
    remain set), matching how a Pine pre-filter would behave — the SM
    episode triggered, the entry was just rejected.
    """
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    blocked = 0

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # EOD close
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # Exits
        if trade_state == 1:
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue
            if tp_pts > 0 and closes[i - 1] >= entry_price + tp_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

        elif trade_state == -1:
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue
            if tp_pts > 0 and closes[i - 1] <= entry_price - tp_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

        # Entries with TPX regime gate
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    long_used = True  # episode consumed regardless
                    # TPX REGIME GATE: only enter long if TPX > 0
                    if tpx[i - 1] > 0:
                        trade_state = 1
                        entry_price = opens[i]
                        entry_idx = i
                    else:
                        blocked += 1

                elif sm_bear and rsi_short_trigger and not short_used:
                    short_used = True  # episode consumed regardless
                    # TPX REGIME GATE: only enter short if TPX < 0
                    if tpx[i - 1] < 0:
                        trade_state = -1
                        entry_price = opens[i]
                        entry_idx = i
                    else:
                        blocked += 1

    return trades, blocked


def test_strategy(name, strategy_id, opens, highs, lows, closes, sm, times,
                  rsi_curr, rsi_prev, rsi_buy, rsi_sell, sm_threshold,
                  cooldown, max_loss_pts, tp_pts, dollar_per_pt, commission,
                  tpx_variants, eod_et=None):
    """Test one strategy across multiple TPX parameter combos."""
    from generate_session import run_backtest_tp_exit

    # Baseline (no TPX filter)
    kwargs = dict(
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts, tp_pts=tp_pts,
    )
    if eod_et is not None:
        kwargs["eod_minutes_et"] = eod_et

    baseline_trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev, **kwargs,
    )
    compute_mfe_mae(baseline_trades, highs, lows)
    sc_base = score_trades(baseline_trades, commission_per_side=commission,
                           dollar_per_pt=dollar_per_pt)

    print(f"\n{'='*80}")
    print(f"  {name} — TPX Regime Gate Sweep")
    print(f"{'='*80}")
    print(f"  BASELINE (no filter): {fmt_score(sc_base)}")

    print(f"\n  {'TPX Params':<18s} {'Trades':>6s} {'Blocked':>7s} {'WR%':>6s} "
          f"{'PF':>6s} {'Net$':>10s} {'MaxDD$':>9s} {'Sharpe':>7s} {'delta$':>9s}")
    print(f"  {'-'*82}")

    results = []
    for label, tpx_arr in tpx_variants:
        eod_kw = {"eod_minutes_et": eod_et} if eod_et is not None else {}
        trades, blocked_count = run_backtest_tpx_regime(
            opens, highs, lows, closes, sm, times,
            rsi_curr, rsi_prev, tpx_arr,
            rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            sm_threshold=sm_threshold, cooldown_bars=cooldown,
            max_loss_pts=max_loss_pts, tp_pts=tp_pts,
            **eod_kw,
        )
        compute_mfe_mae(trades, highs, lows)
        sc = score_trades(trades, commission_per_side=commission,
                          dollar_per_pt=dollar_per_pt)

        if sc:
            delta = sc["net_dollar"] - sc_base["net_dollar"]
            print(f"  {label:<18s} {sc['count']:>6d} {blocked_count:>7d} "
                  f"{sc['win_rate']:>5.1f}% {sc['pf']:>6.3f} "
                  f"${sc['net_dollar']:>+9.2f} ${sc['max_dd_dollar']:>8.2f} "
                  f"{sc['sharpe']:>7.3f} ${delta:>+8.2f}")
            results.append({
                "label": label, "sc": sc, "delta": delta,
                "blocked": blocked_count, "trades": trades,
            })
        else:
            print(f"  {label:<18s}      0 {blocked_count:>7d}    —      —          —         —       —        —")

    # Best result
    if results:
        best = max(results, key=lambda r: r["sc"]["net_dollar"])
        print(f"\n  BEST: {best['label']} — ${best['sc']['net_dollar']:+.2f} "
              f"(delta ${best['delta']:+.2f}, "
              f"blocked {best['blocked']}, "
              f"PF {best['sc']['pf']:.3f}, Sharpe {best['sc']['sharpe']:.3f})")

    return sc_base, results


def main():
    print("=" * 80)
    print("  TPX REGIME GATE TEST — Pre-Filter Sweep")
    print("  Only LONG in TPX green zones, only SHORT in TPX red zones")
    print("=" * 80)

    # --- Load data ---
    print("\nLoading data...")
    df_mnq = load_instrument_1min("MNQ")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm

    df_mes = load_instrument_1min("MES")
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm

    print(f"  MNQ: {len(df_mnq)} bars, MES: {len(df_mes)} bars")

    # --- Compute TPX variants ---
    print("\nComputing TPX variants...")
    mnq_h, mnq_l = df_mnq["High"].values, df_mnq["Low"].values
    mes_h, mes_l = df_mes["High"].values, df_mes["Low"].values

    # Sweep: (length, smooth) combos — from fast to slow
    tpx_params = [
        (7, 3),     # default
        (10, 4),
        (12, 5),
        (15, 6),    # what you saw on the chart
        (20, 8),
        (25, 10),
        (30, 12),
    ]

    mnq_tpx_variants = []
    mes_tpx_variants = []
    for length, smooth in tpx_params:
        label = f"L={length:>2d} S={smooth:>2d}"
        tpx_mnq, _, _ = compute_tpx(mnq_h, mnq_l, length=length, smooth=smooth)
        tpx_mes, _, _ = compute_tpx(mes_h, mes_l, length=length, smooth=smooth)
        mnq_tpx_variants.append((label, tpx_mnq))
        mes_tpx_variants.append((label, tpx_mes))
        print(f"  {label}: MNQ {np.sum(tpx_mnq>0)/len(tpx_mnq)*100:.0f}% bull, "
              f"MES {np.sum(tpx_mes>0)/len(tpx_mes)*100:.0f}% bull")

    # --- RSI prep ---
    df_mnq_5m = resample_to_5min(df_mnq)
    rsi_mnq_curr, rsi_mnq_prev = map_5min_rsi_to_1min(
        df_mnq.index.values, df_mnq_5m.index.values,
        df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN,
    )

    df_mes_5m = resample_to_5min(df_mes)
    rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
        df_mes.index.values, df_mes_5m.index.values,
        df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN,
    )

    mnq_arrays = (df_mnq["Open"].values, df_mnq["High"].values,
                  df_mnq["Low"].values, df_mnq["Close"].values,
                  df_mnq["SM_Net"].values, df_mnq.index)

    mes_arrays = (df_mes["Open"].values, df_mes["High"].values,
                  df_mes["Low"].values, df_mes["Close"].values,
                  df_mes["SM_Net"].values, df_mes.index)

    # --- Test each strategy ---
    test_strategy(
        "vScalpA (MNQ v15)", "MNQ_V15",
        *mnq_arrays, rsi_mnq_curr, rsi_mnq_prev,
        VSCALPA_RSI_BUY, VSCALPA_RSI_SELL, VSCALPA_SM_THRESHOLD,
        VSCALPA_COOLDOWN, VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS,
        MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
        mnq_tpx_variants,
    )

    test_strategy(
        "vScalpB (MNQ)", "MNQ_VSCALPB",
        *mnq_arrays, rsi_mnq_curr, rsi_mnq_prev,
        VSCALPB_RSI_BUY, VSCALPB_RSI_SELL, VSCALPB_SM_THRESHOLD,
        VSCALPB_COOLDOWN, VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
        MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
        mnq_tpx_variants,
    )

    test_strategy(
        "MES v2", "MES_V2",
        *mes_arrays, rsi_mes_curr, rsi_mes_prev,
        MESV2_RSI_BUY, MESV2_RSI_SELL, MESV2_SM_THRESHOLD,
        MESV2_COOLDOWN, MESV2_MAX_LOSS_PTS, MESV2_TP_PTS,
        MES_DOLLAR_PER_PT, MES_COMMISSION,
        mes_tpx_variants, eod_et=MESV2_EOD_ET,
    )


if __name__ == "__main__":
    main()
