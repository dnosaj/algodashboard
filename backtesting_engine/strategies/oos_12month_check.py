"""
12-Month Combined Validation
=============================
Loads both 6-month files as one continuous series, computes SM once,
runs backtests once, then splits trades by date.

This validates whether SM warmup effects cause differences vs running
each 6-month file independently.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    fmt_score, compute_et_minutes,
    NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MES_SM = dict(index_period=20, flow_period=12, norm_period=400, ema_len=255)
MNQ_COMM, MNQ_DPP = 0.52, 2.0
MES_COMM, MES_DPP = 1.25, 5.0

SPLIT = pd.Timestamp("2025-08-17", tz='UTC')


def load_combined(inst, sm_params):
    files = sorted(DATA_DIR.glob(f"databento_{inst}_1min_2025-*.csv"))
    print(f"  Loading {inst}: {[f.name for f in files]}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        result = pd.DataFrame()
        result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
        result['Open'] = pd.to_numeric(df['open'], errors='coerce')
        result['High'] = pd.to_numeric(df['high'], errors='coerce')
        result['Low'] = pd.to_numeric(df['low'], errors='coerce')
        result['Close'] = pd.to_numeric(df['close'], errors='coerce')
        result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        result = result.set_index('Time')
        dfs.append(result)
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    sm = compute_smart_money(
        combined['Close'].values, combined['Volume'].values, **sm_params
    )
    combined['SM_Net'] = sm
    print(f"    {len(combined):,} bars total ({combined.index[0].date()} to {combined.index[-1].date()})")
    return combined


def prepare_arrays(df_1m, rsi_len):
    df_5m = resample_to_5min(df_1m)
    rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
        df_1m.index.values, df_5m.index.values, df_5m['Close'].values, rsi_len
    )
    rsi_dummy = np.full(len(df_1m), 50.0)
    return {
        'opens': df_1m['Open'].values,
        'highs': df_1m['High'].values,
        'lows': df_1m['Low'].values,
        'closes': df_1m['Close'].values,
        'sm': df_1m['SM_Net'].values,
        'times': df_1m.index.values,
        'rsi': rsi_dummy,
        'rsi_5m_curr': rsi_5m_curr,
        'rsi_5m_prev': rsi_5m_prev,
    }


def run_mes_v94(opens, highs, lows, closes, sm, times,
                rsi_5m_curr, rsi_5m_prev,
                rsi_buy=55, rsi_sell=45,
                cooldown_bars=15, eod_minutes_et=930):
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
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
        sm_bull = sm_prev > 0
        sm_bear = sm_prev < 0
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
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue
        if trade_state == 1:
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
        elif trade_state == -1:
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= cooldown_bars
            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
    return trades


def split_trades(trades, split_ts):
    oos, is_trades = [], []
    for t in trades:
        exit_ts = pd.Timestamp(t['exit_time'])
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize('UTC')
        if exit_ts < split_ts:
            oos.append(t)
        else:
            is_trades.append(t)
    return oos, is_trades


def main():
    print("=" * 80)
    print("12-MONTH COMBINED VALIDATION")
    print("Run on full 12 months, split trades at Aug 17 boundary")
    print("=" * 80)

    print("\nLoading combined data...")
    mnq_full = load_combined('MNQ', MNQ_SM)
    mes_full = load_combined('MES', MES_SM)

    print("\nPreparing arrays...")
    mnq_arr = prepare_arrays(mnq_full, rsi_len=8)
    mes_arr = prepare_arrays(mes_full, rsi_len=10)

    # Reference: split-file results from oos_validation.py
    print("\n" + "=" * 80)
    print("REFERENCE (split-file approach from oos_validation.py):")
    print(f"  MNQ v11.1  IS: 226 trades, WR 62.4%, PF 1.797, +$3528.96")
    print(f"  MNQ v11.1 OOS: 220 trades, WR 44.1%, PF 0.600, -$3433.80")
    print(f"  MNQ v15    IS: 363 trades, WR 88.2%, PF 1.272, +$1324.48")
    print(f"  MNQ v15   OOS: 385 trades, WR 81.8%, PF 0.932, -$508.90")
    print(f"  MES v9.4   IS: 359 trades, WR 54.9%, PF 1.266, +$1187.50")
    print(f"  MES v9.4  OOS: 374 trades, WR 50.8%, PF 0.897, -$848.75")
    print("=" * 80)

    print(f"\n{'Strategy':<14} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Net $':>11} {'SL':>4}")
    print("-" * 80)

    # MNQ v11.1
    trades = run_backtest_v10(
        mnq_arr['opens'], mnq_arr['highs'], mnq_arr['lows'], mnq_arr['closes'],
        mnq_arr['sm'], mnq_arr['rsi'], mnq_arr['times'],
        rsi_buy=60, rsi_sell=40, sm_threshold=0.15, cooldown_bars=20,
        max_loss_pts=50,
        rsi_5m_curr=mnq_arr['rsi_5m_curr'], rsi_5m_prev=mnq_arr['rsi_5m_prev'],
    )
    oos, is_t = split_trades(trades, SPLIT)
    for label, t_list in [("OOS", oos), ("IS", is_t), ("FULL", trades)]:
        sc = score_trades(t_list, MNQ_COMM, MNQ_DPP)
        sl = sum(1 for t in t_list if t['result'] == 'SL')
        if sc:
            print(f"{'MNQ v11.1':<14} {label:>5} {sc['count']:>7d} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+11.2f} {sl:>4d}")

    # MNQ v15
    trades = run_backtest_tp_exit(
        mnq_arr['opens'], mnq_arr['highs'], mnq_arr['lows'], mnq_arr['closes'],
        mnq_arr['sm'], mnq_arr['times'],
        mnq_arr['rsi_5m_curr'], mnq_arr['rsi_5m_prev'],
        rsi_buy=60, rsi_sell=40, sm_threshold=0.0, cooldown_bars=20,
        max_loss_pts=50, tp_pts=5,
    )
    oos, is_t = split_trades(trades, SPLIT)
    for label, t_list in [("OOS", oos), ("IS", is_t), ("FULL", trades)]:
        sc = score_trades(t_list, MNQ_COMM, MNQ_DPP)
        sl = sum(1 for t in t_list if t['result'] == 'SL')
        if sc:
            print(f"{'MNQ v15':<14} {label:>5} {sc['count']:>7d} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+11.2f} {sl:>4d}")

    # MES v9.4
    trades = run_mes_v94(
        mes_arr['opens'], mes_arr['highs'], mes_arr['lows'], mes_arr['closes'],
        mes_arr['sm'], mes_arr['times'],
        mes_arr['rsi_5m_curr'], mes_arr['rsi_5m_prev'],
        rsi_buy=55, rsi_sell=45, cooldown_bars=15, eod_minutes_et=930,
    )
    oos, is_t = split_trades(trades, SPLIT)
    for label, t_list in [("OOS", oos), ("IS", is_t), ("FULL", trades)]:
        sc = score_trades(t_list, MES_COMM, MES_DPP)
        sl = sum(1 for t in t_list if t['result'] == 'SL')
        if sc:
            print(f"{'MES v9.4':<14} {label:>5} {sc['count']:>7d} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+11.2f} {sl:>4d}")

    print("\n" + "=" * 80)
    print("If combined IS numbers match split-file IS numbers, methodology is sound.")
    print("If they differ, SM warmup at file boundaries is affecting results.")
    print("=" * 80)


if __name__ == "__main__":
    main()
