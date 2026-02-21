"""
OOS Validation — Test production strategies on Feb–Aug 2025 data
================================================================

Runs MNQ v11.1, MNQ v15, and MES v9.4 on the new 6-month OOS period
(Feb 17 – Aug 17, 2025) and compares to the IS period (Aug 17, 2025 –
Feb 13, 2026).

All backtests use 1-min bars with 5-min RSI mapped back (matching Pine).

Usage:
    python3 oos_validation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    fmt_score, compute_et_minutes,
    NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)

# Import v15 TP-exit backtest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---- Data files ----
FILES = {
    'MNQ_OOS': 'databento_MNQ_1min_2025-02-17_to_2025-08-17.csv',
    'MNQ_IS':  'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
    'MES_OOS': 'databento_MES_1min_2025-02-17_to_2025-08-17.csv',
    'MES_IS':  'databento_MES_1min_2025-08-17_to_2026-02-13.csv',
}

# ---- SM params per instrument ----
MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MES_SM = dict(index_period=20, flow_period=12, norm_period=400, ema_len=255)

# ---- Commission per instrument ----
MNQ_COMM = 0.52   # $/side
MNQ_DPP  = 2.0    # $/point
MES_COMM = 1.25   # $/side
MES_DPP  = 5.0    # $/point


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_databento(filepath, sm_params):
    """Load Databento CSV and compute SM with given params."""
    df = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    if 'VWAP' in df.columns:
        result['VWAP'] = pd.to_numeric(df['VWAP'], errors='coerce')
    result = result.set_index('Time')

    sm = compute_smart_money(
        result['Close'].values, result['Volume'].values,
        **sm_params
    )
    result['SM_Net'] = sm
    return result


def prepare_arrays(df_1m, rsi_len):
    """Prepare 1-min arrays with 5-min RSI mapped back."""
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


# ---------------------------------------------------------------------------
# MES v9.4 custom backtest (needs configurable EOD time)
# ---------------------------------------------------------------------------

def run_mes_v94(opens, highs, lows, closes, sm, times,
                rsi_5m_curr, rsi_5m_prev,
                rsi_buy=55, rsi_sell=45,
                cooldown_bars=15, eod_minutes_et=930):
    """MES v9.4 backtest: SM flip exit, no SL, no threshold, EOD 15:30.

    Custom function because run_backtest_v10 hardcodes NY_CLOSE_ET=960.
    """
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

        # RSI cross from mapped 5-min
        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        # Episode reset
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # EOD close at custom time (15:30 = 930 ET mins)
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # SM flip exits
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

        # Entries
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("OUT-OF-SAMPLE VALIDATION")
    print("OOS: Feb 17 – Aug 17, 2025  |  IS: Aug 17, 2025 – Feb 13, 2026")
    print("=" * 80)

    # ---- Load data ----
    print("\nLoading data...")
    mnq_oos = load_databento(DATA_DIR / FILES['MNQ_OOS'], MNQ_SM)
    mnq_is  = load_databento(DATA_DIR / FILES['MNQ_IS'], MNQ_SM)
    mes_oos = load_databento(DATA_DIR / FILES['MES_OOS'], MES_SM)
    mes_is  = load_databento(DATA_DIR / FILES['MES_IS'], MES_SM)

    print(f"  MNQ OOS: {len(mnq_oos):,} bars ({mnq_oos.index[0].date()} to {mnq_oos.index[-1].date()})")
    print(f"  MNQ IS:  {len(mnq_is):,} bars ({mnq_is.index[0].date()} to {mnq_is.index[-1].date()})")
    print(f"  MES OOS: {len(mes_oos):,} bars ({mes_oos.index[0].date()} to {mes_oos.index[-1].date()})")
    print(f"  MES IS:  {len(mes_is):,} bars ({mes_is.index[0].date()} to {mes_is.index[-1].date()})")

    # ---- Prepare arrays ----
    print("\nPreparing arrays (SM + 5-min RSI mapping)...")
    mnq_oos_arr = prepare_arrays(mnq_oos, rsi_len=8)
    mnq_is_arr  = prepare_arrays(mnq_is, rsi_len=8)
    mes_oos_arr = prepare_arrays(mes_oos, rsi_len=10)
    mes_is_arr  = prepare_arrays(mes_is, rsi_len=10)

    results = {}

    # ================================================================
    # MNQ v11.1
    # ================================================================
    print("\n" + "-" * 80)
    print("MNQ v11.1  SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50 threshold=0.15")
    print("-" * 80)

    for label, arr in [("IS ", mnq_is_arr), ("OOS", mnq_oos_arr)]:
        trades = run_backtest_v10(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['rsi'], arr['times'],
            rsi_buy=60, rsi_sell=40,
            sm_threshold=0.15, cooldown_bars=20,
            max_loss_pts=50,
            rsi_5m_curr=arr['rsi_5m_curr'],
            rsi_5m_prev=arr['rsi_5m_prev'],
        )
        sc = score_trades(trades, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
        results[f'v11.1_{label.strip()}'] = sc
        print(f"  {label}: {fmt_score(sc)}")
        if sc:
            exits = sc.get('exits', {})
            print(f"       Exits: {' '.join(f'{k}:{v}' for k, v in sorted(exits.items()))}")

    # ================================================================
    # MNQ v15
    # ================================================================
    print("\n" + "-" * 80)
    print("MNQ v15   SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50 TP=5 threshold=0.0")
    print("-" * 80)

    for label, arr in [("IS ", mnq_is_arr), ("OOS", mnq_oos_arr)]:
        trades = run_backtest_tp_exit(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['times'],
            arr['rsi_5m_curr'], arr['rsi_5m_prev'],
            rsi_buy=60, rsi_sell=40,
            sm_threshold=0.0, cooldown_bars=20,
            max_loss_pts=50, tp_pts=5,
        )
        sc = score_trades(trades, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
        results[f'v15_{label.strip()}'] = sc
        print(f"  {label}: {fmt_score(sc)}")
        if sc:
            exits = sc.get('exits', {})
            print(f"       Exits: {' '.join(f'{k}:{v}' for k, v in sorted(exits.items()))}")

    # ================================================================
    # MES v9.4
    # ================================================================
    print("\n" + "-" * 80)
    print("MES v9.4  SM(20/12/400/255) RSI(10/55/45) CD=15 SL=0 EOD=15:30")
    print("-" * 80)

    for label, arr in [("IS ", mes_is_arr), ("OOS", mes_oos_arr)]:
        trades = run_mes_v94(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['times'],
            arr['rsi_5m_curr'], arr['rsi_5m_prev'],
            rsi_buy=55, rsi_sell=45,
            cooldown_bars=15, eod_minutes_et=930,
        )
        sc = score_trades(trades, commission_per_side=MES_COMM, dollar_per_pt=MES_DPP)
        results[f'v94_{label.strip()}'] = sc
        print(f"  {label}: {fmt_score(sc)}")
        if sc:
            exits = sc.get('exits', {})
            print(f"       Exits: {' '.join(f'{k}:{v}' for k, v in sorted(exits.items()))}")

    # ================================================================
    # Summary table
    # ================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    header = f"{'Strategy':<14} {'Period':>6}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Net $':>10}  {'MaxDD $':>10}  {'Sharpe':>7}"
    print(header)
    print("-" * 80)

    strat_labels = {'v11.1': 'MNQ v11.1', 'v15': 'MNQ v15', 'v94': 'MES v9.4'}
    for key in ['v11.1_IS', 'v11.1_OOS', 'v15_IS', 'v15_OOS', 'v94_IS', 'v94_OOS']:
        sc = results.get(key)
        parts = key.rsplit('_', 1)
        strat = parts[0]
        period = parts[1]
        strat_label = strat_labels.get(strat, strat)

        if sc:
            print(f"{strat_label:<14} {period:>6}  {sc['count']:>6d}  "
                  f"{sc['win_rate']:>5.1f}%  {sc['pf']:>6.3f}  "
                  f"{sc['net_dollar']:>+10.2f}  {sc['max_dd_dollar']:>10.2f}  "
                  f"{sc['sharpe']:>7.3f}")
        else:
            print(f"{strat_label:<14} {period:>6}  {'NO TRADES':>6}")

    # ================================================================
    # Per-month breakdown for OOS
    # ================================================================
    print("\n" + "=" * 80)
    print("PER-MONTH BREAKDOWN — OOS (Feb–Aug 2025)")
    print("=" * 80)

    strategies = [
        ("MNQ v11.1", mnq_oos_arr,
         lambda a: run_backtest_v10(
             a['opens'], a['highs'], a['lows'], a['closes'],
             a['sm'], a['rsi'], a['times'],
             rsi_buy=60, rsi_sell=40, sm_threshold=0.15,
             cooldown_bars=20, max_loss_pts=50,
             rsi_5m_curr=a['rsi_5m_curr'], rsi_5m_prev=a['rsi_5m_prev']),
         dict(commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)),

        ("MNQ v15", mnq_oos_arr,
         lambda a: run_backtest_tp_exit(
             a['opens'], a['highs'], a['lows'], a['closes'],
             a['sm'], a['times'],
             a['rsi_5m_curr'], a['rsi_5m_prev'],
             rsi_buy=60, rsi_sell=40, sm_threshold=0.0,
             cooldown_bars=20, max_loss_pts=50, tp_pts=5),
         dict(commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)),

        ("MES v9.4", mes_oos_arr,
         lambda a: run_mes_v94(
             a['opens'], a['highs'], a['lows'], a['closes'],
             a['sm'], a['times'],
             a['rsi_5m_curr'], a['rsi_5m_prev'],
             rsi_buy=55, rsi_sell=45,
             cooldown_bars=15, eod_minutes_et=930),
         dict(commission_per_side=MES_COMM, dollar_per_pt=MES_DPP)),
    ]

    for strat_name, arr, run_fn, score_params in strategies:
        trades = run_fn(arr)

        # Group by month
        monthly = {}
        for t in trades:
            month = pd.Timestamp(t['exit_time']).strftime('%Y-%m')
            if month not in monthly:
                monthly[month] = []
            monthly[month].append(t)

        print(f"\n{strat_name}:")
        print(f"  {'Month':<10} {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Net $':>10}")
        for month in sorted(monthly.keys()):
            sc = score_trades(monthly[month], **score_params)
            if sc:
                print(f"  {month:<10} {sc['count']:>6d}  "
                      f"{sc['win_rate']:>5.1f}%  {sc['pf']:>6.3f}  "
                      f"{sc['net_dollar']:>+10.2f}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
