"""
Run active portfolio strategies on full 12-month data and save trade logs.

Runs vScalpA (v15), vScalpB, vScalpC, and MES v2 on Databento 1-min bars with
production parameters. Saves per-trade CSVs to backtesting_engine/results/.

Usage:
    python3 run_and_save_portfolio.py              # full 12 months
    python3 run_and_save_portfolio.py --split       # IS + OOS halves
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_test_common import (
    compute_rsi,
    compute_smart_money,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
)

# Use generate_session's data loader + backtest engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
    compute_mfe_mae,
    # MNQ shared SM params
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    # vScalpA params
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS, VSCALPA_ENTRY_END_ET,
    # vScalpB params
    VSCALPB_RSI_LEN, VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
    VSCALPB_SM_THRESHOLD, VSCALPB_COOLDOWN,
    VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
    # MES SM params
    MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA,
    MES_DOLLAR_PER_PT, MES_COMMISSION,
    # MES v2 params
    MESV2_RSI_LEN, MESV2_RSI_BUY, MESV2_RSI_SELL,
    MESV2_SM_THRESHOLD, MESV2_COOLDOWN,
    MESV2_MAX_LOSS_PTS, MESV2_TP_PTS, MESV2_EOD_ET, MESV2_BREAKEVEN_BARS,
)

from vscalpc_partial_exit_sweep import run_backtest_partial_exit
from results.save_results import save_backtest

SCRIPT_NAME = "run_and_save_portfolio.py"

# vScalpC production params (same entries as V15, partial exit runner)
VSCALPC_TP1_PTS = 7     # Scalp leg: close 1 contract at +7 pts
VSCALPC_TP2_PTS = 25    # Runner leg: target +25 pts
VSCALPC_SL_PTS = 40     # Same SL as V15
VSCALPC_SL_TO_BE = True  # Move runner SL to entry after TP1
VSCALPC_BE_TIME = 45     # Close stale runners after 45 bars


def run_strategy(name, strategy_id, opens, highs, lows, closes, sm, times,
                 rsi_curr, rsi_prev, rsi_buy, rsi_sell, sm_threshold,
                 cooldown, max_loss_pts, tp_pts, dollar_per_pt, commission,
                 params_dict, data_range, split="FULL", eod_et=None,
                 entry_end_et=None, breakeven_after_bars=0):
    """Run a single strategy and save results."""
    kwargs = dict(
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts, tp_pts=tp_pts,
    )
    if eod_et is not None:
        kwargs["eod_minutes_et"] = eod_et
    if entry_end_et is not None:
        kwargs["entry_end_et"] = entry_end_et
    if breakeven_after_bars > 0:
        kwargs["breakeven_after_bars"] = breakeven_after_bars

    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev, **kwargs,
    )
    compute_mfe_mae(trades, highs, lows)

    sc = score_trades(trades, commission_per_side=commission,
                      dollar_per_pt=dollar_per_pt)
    label = f"{name} ({split})"
    print(f"\n  {fmt_score(sc, label)}")

    save_backtest(
        trades, strategy=strategy_id, params=params_dict,
        data_range=data_range, split=split,
        dollar_per_pt=dollar_per_pt, commission=commission,
        script_name=SCRIPT_NAME,
    )
    return trades, sc


def run_partial_strategy(name, strategy_id, opens, highs, lows, closes, sm,
                         times, rsi_curr, rsi_prev, rsi_buy, rsi_sell,
                         sm_threshold, cooldown, sl_pts, tp1_pts, tp2_pts,
                         dollar_per_pt, commission, params_dict, data_range,
                         split="FULL", entry_end_et=None, sl_to_be=False,
                         be_time_bars=0):
    """Run a partial exit strategy (2 contracts) and save results."""
    kwargs = {}
    if entry_end_et is not None:
        kwargs["entry_end_et"] = entry_end_et

    trades = run_backtest_partial_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        sl_pts=sl_pts, tp1_pts=tp1_pts, tp2_pts=tp2_pts,
        sl_to_be_after_tp1=sl_to_be, be_time_bars=be_time_bars,
        **kwargs,
    )
    compute_mfe_mae(trades, highs, lows)

    # 2 contracts: pass commission*2 so score_trades deducts 4×commission total
    sc = score_trades(trades, commission_per_side=commission * 2,
                      dollar_per_pt=dollar_per_pt)
    label = f"{name} ({split})"
    print(f"\n  {fmt_score(sc, label)}")

    save_backtest(
        trades, strategy=strategy_id, params=params_dict,
        data_range=data_range, split=split,
        dollar_per_pt=dollar_per_pt, commission=commission,
        script_name=SCRIPT_NAME, qty=2,
    )
    return trades, sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true",
                        help="Also save IS and OOS halves")
    args = parser.parse_args()

    print("=" * 70)
    print("PORTFOLIO BACKTEST — vScalpA + vScalpB + vScalpC + MES v2")
    print("=" * 70)

    # --- Load MNQ ---
    print("\nLoading MNQ data...")
    df_mnq = load_instrument_1min("MNQ")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm
    print(f"  MNQ: {len(df_mnq)} bars, {df_mnq.index[0]} to {df_mnq.index[-1]}")

    # --- Load MES ---
    print("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm
    print(f"  MES: {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # Data range string
    start_date = df_mnq.index[0].strftime("%Y-%m-%d")
    end_date = df_mnq.index[-1].strftime("%Y-%m-%d")
    data_range = f"{start_date}_to_{end_date}"

    # --- Define splits ---
    splits = [("FULL", df_mnq, df_mes)]

    if args.split:
        midpoint = df_mnq.index[len(df_mnq) // 2]
        mnq_is = df_mnq[df_mnq.index < midpoint]
        mnq_oos = df_mnq[df_mnq.index >= midpoint]
        mes_is = df_mes[df_mes.index < midpoint]
        mes_oos = df_mes[df_mes.index >= midpoint]

        is_range = f"{mnq_is.index[0].strftime('%Y-%m-%d')}_to_{mnq_is.index[-1].strftime('%Y-%m-%d')}"
        oos_range = f"{mnq_oos.index[0].strftime('%Y-%m-%d')}_to_{mnq_oos.index[-1].strftime('%Y-%m-%d')}"

        splits.append(("IS", mnq_is, mes_is))
        splits.append(("OOS", mnq_oos, mes_oos))

    for split_name, mnq_data, mes_data in splits:
        if split_name == "IS":
            dr = is_range
        elif split_name == "OOS":
            dr = oos_range
        else:
            dr = data_range

        print(f"\n{'='*70}")
        print(f"  SPLIT: {split_name}  ({len(mnq_data)} MNQ bars, {len(mes_data)} MES bars)")
        print(f"{'='*70}")

        # --- MNQ arrays ---
        mnq_opens = mnq_data["Open"].values
        mnq_highs = mnq_data["High"].values
        mnq_lows = mnq_data["Low"].values
        mnq_closes = mnq_data["Close"].values
        mnq_sm_arr = mnq_data["SM_Net"].values
        mnq_times = mnq_data.index

        # 5-min RSI mapping for MNQ (RSI len=8 shared by both)
        df_mnq_5m = resample_to_5min(mnq_data)
        rsi_mnq_curr, rsi_mnq_prev = map_5min_rsi_to_1min(
            mnq_data.index.values, df_mnq_5m.index.values,
            df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN,
        )

        # --- vScalpA ---
        vscalpa_params = {
            "sm_index": MNQ_SM_INDEX, "sm_flow": MNQ_SM_FLOW,
            "sm_norm": MNQ_SM_NORM, "sm_ema": MNQ_SM_EMA,
            "sm_threshold": VSCALPA_SM_THRESHOLD,
            "rsi_len": VSCALPA_RSI_LEN, "rsi_buy": VSCALPA_RSI_BUY,
            "rsi_sell": VSCALPA_RSI_SELL, "cooldown": VSCALPA_COOLDOWN,
            "max_loss_pts": VSCALPA_MAX_LOSS_PTS, "tp_pts": VSCALPA_TP_PTS,
        }
        run_strategy(
            "vScalpA", "MNQ_V15",
            mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
            rsi_mnq_curr, rsi_mnq_prev,
            VSCALPA_RSI_BUY, VSCALPA_RSI_SELL, VSCALPA_SM_THRESHOLD,
            VSCALPA_COOLDOWN, VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS,
            MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
            vscalpa_params, dr, split_name,
            entry_end_et=VSCALPA_ENTRY_END_ET,
        )

        # --- vScalpB ---
        vscalpb_params = {
            "sm_index": MNQ_SM_INDEX, "sm_flow": MNQ_SM_FLOW,
            "sm_norm": MNQ_SM_NORM, "sm_ema": MNQ_SM_EMA,
            "sm_threshold": VSCALPB_SM_THRESHOLD,
            "rsi_len": VSCALPB_RSI_LEN, "rsi_buy": VSCALPB_RSI_BUY,
            "rsi_sell": VSCALPB_RSI_SELL, "cooldown": VSCALPB_COOLDOWN,
            "max_loss_pts": VSCALPB_MAX_LOSS_PTS, "tp_pts": VSCALPB_TP_PTS,
        }
        run_strategy(
            "vScalpB", "MNQ_VSCALPB",
            mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
            rsi_mnq_curr, rsi_mnq_prev,
            VSCALPB_RSI_BUY, VSCALPB_RSI_SELL, VSCALPB_SM_THRESHOLD,
            VSCALPB_COOLDOWN, VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
            MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
            vscalpb_params, dr, split_name,
        )

        # --- vScalpC (partial exit runner) ---
        vscalpc_params = {
            "sm_index": MNQ_SM_INDEX, "sm_flow": MNQ_SM_FLOW,
            "sm_norm": MNQ_SM_NORM, "sm_ema": MNQ_SM_EMA,
            "sm_threshold": VSCALPA_SM_THRESHOLD,
            "rsi_len": VSCALPA_RSI_LEN, "rsi_buy": VSCALPA_RSI_BUY,
            "rsi_sell": VSCALPA_RSI_SELL, "cooldown": VSCALPA_COOLDOWN,
            "sl_pts": VSCALPC_SL_PTS, "tp1_pts": VSCALPC_TP1_PTS,
            "tp2_pts": VSCALPC_TP2_PTS, "sl_to_be": VSCALPC_SL_TO_BE,
            "be_time_bars": VSCALPC_BE_TIME,
        }
        run_partial_strategy(
            "vScalpC", "MNQ_VSCALPC",
            mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
            rsi_mnq_curr, rsi_mnq_prev,
            VSCALPA_RSI_BUY, VSCALPA_RSI_SELL, VSCALPA_SM_THRESHOLD,
            VSCALPA_COOLDOWN, VSCALPC_SL_PTS, VSCALPC_TP1_PTS, VSCALPC_TP2_PTS,
            MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
            vscalpc_params, dr, split_name,
            entry_end_et=VSCALPA_ENTRY_END_ET,
            sl_to_be=VSCALPC_SL_TO_BE, be_time_bars=VSCALPC_BE_TIME,
        )

        # --- MES v2 ---
        mes_opens = mes_data["Open"].values
        mes_highs = mes_data["High"].values
        mes_lows = mes_data["Low"].values
        mes_closes = mes_data["Close"].values
        mes_sm_arr = mes_data["SM_Net"].values
        mes_times = mes_data.index

        df_mes_5m = resample_to_5min(mes_data)
        rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
            mes_data.index.values, df_mes_5m.index.values,
            df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN,
        )

        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
        }
        run_strategy(
            "MES v2", "MES_V2",
            mes_opens, mes_highs, mes_lows, mes_closes, mes_sm_arr, mes_times,
            rsi_mes_curr, rsi_mes_prev,
            MESV2_RSI_BUY, MESV2_RSI_SELL, MESV2_SM_THRESHOLD,
            MESV2_COOLDOWN, MESV2_MAX_LOSS_PTS, MESV2_TP_PTS,
            MES_DOLLAR_PER_PT, MES_COMMISSION,
            mesv2_params, dr, split_name, eod_et=MESV2_EOD_ET,
            breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        )

    print(f"\n{'='*70}")
    print("Done. Results saved to backtesting_engine/results/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
