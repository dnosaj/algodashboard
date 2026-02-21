"""
OOS Regime Analysis — Why do all strategies fail on Feb-Aug 2025?
================================================================
Compares volatility, SM signal quality, SL rates, directional bias,
and market regime between OOS (Feb-Aug 2025) and IS (Aug 2025-Feb 2026).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    compute_et_minutes, compute_rsi,
    NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    'MNQ_OOS': 'databento_MNQ_1min_2025-02-17_to_2025-08-17.csv',
    'MNQ_IS':  'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
    'MES_OOS': 'databento_MES_1min_2025-02-17_to_2025-08-17.csv',
    'MES_IS':  'databento_MES_1min_2025-08-17_to_2026-02-13.csv',
}

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MES_SM = dict(index_period=20, flow_period=12, norm_period=400, ema_len=255)
MNQ_COMM, MNQ_DPP = 0.52, 2.0
MES_COMM, MES_DPP = 1.25, 5.0


def load_databento(filepath, sm_params):
    df = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    sm = compute_smart_money(result['Close'].values, result['Volume'].values, **sm_params)
    result['SM_Net'] = sm
    return result


def prepare_arrays(df_1m, rsi_len):
    df_5m = resample_to_5min(df_1m)
    rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
        df_1m.index.values, df_5m.index.values, df_5m['Close'].values, rsi_len
    )
    rsi_dummy = np.full(len(df_1m), 50.0)
    return {
        'opens': df_1m['Open'].values, 'highs': df_1m['High'].values,
        'lows': df_1m['Low'].values, 'closes': df_1m['Close'].values,
        'sm': df_1m['SM_Net'].values, 'times': df_1m.index.values,
        'rsi': rsi_dummy, 'rsi_5m_curr': rsi_5m_curr, 'rsi_5m_prev': rsi_5m_prev,
    }


def run_mes_v94(opens, highs, lows, closes, sm, times,
                rsi_5m_curr, rsi_5m_prev,
                rsi_buy=55, rsi_sell=45, cooldown_bars=15, eod_minutes_et=930):
    n = len(opens)
    trades, trade_state, entry_price, entry_idx = [], 0, 0.0, 0
    exit_bar, long_used, short_used = -9999, False, False
    et_mins = compute_et_minutes(times)

    def close_trade(side, ep, xp, ei, xi, result):
        pts = (xp - ep) if side == "long" else (ep - xp)
        trades.append({"side": side, "entry": ep, "exit": xp, "pts": pts,
                        "entry_time": times[ei], "exit_time": times[xi],
                        "entry_idx": ei, "exit_idx": xi, "bars": xi - ei, "result": result})

    for i in range(2, n):
        bm = et_mins[i]
        sp, sp2 = sm[i-1], sm[i-2]
        sb, sbe = sp > 0, sp < 0
        sfb = sp > 0 and sp2 <= 0
        sfbe = sp < 0 and sp2 >= 0
        rp, rp2 = rsi_5m_curr[i-1], rsi_5m_prev[i-1]
        rlt = rp > rsi_buy and rp2 <= rsi_buy
        rst = rp < rsi_sell and rp2 >= rsi_sell
        if sfb or not sb: long_used = False
        if sfbe or not sbe: short_used = False
        if trade_state != 0 and bm >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state, exit_bar = 0, i
            continue
        if trade_state == 1 and sp < 0 and sp2 >= 0:
            close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
            trade_state, exit_bar = 0, i
        elif trade_state == -1 and sp > 0 and sp2 <= 0:
            close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
            trade_state, exit_bar = 0, i
        if trade_state == 0:
            bs = i - exit_bar
            ins = NY_OPEN_ET <= bm <= NY_LAST_ENTRY_ET
            if ins and bs >= cooldown_bars:
                if sb and rlt and not long_used:
                    trade_state, entry_price, entry_idx, long_used = 1, opens[i], i, True
                elif sbe and rst and not short_used:
                    trade_state, entry_price, entry_idx, short_used = -1, opens[i], i, True
    return trades


def compute_mfe_mae(trades, highs, lows):
    for t in trades:
        ei, xi, ep = t['entry_idx'], t['exit_idx'], t['entry']
        if xi <= ei:
            t['mfe'] = t['mae'] = 0.0
            continue
        bh, bl = highs[ei:xi+1], lows[ei:xi+1]
        if t['side'] == 'long':
            t['mfe'] = float(np.max(bh) - ep)
            t['mae'] = float(ep - np.min(bl))
        else:
            t['mfe'] = float(ep - np.min(bl))
            t['mae'] = float(np.max(bh) - ep)
    return trades


def get_daily_stats(df):
    """Get daily OHLCV stats from 1-min data (RTH only)."""
    et_mins = compute_et_minutes(df.index.values)
    mask = (et_mins >= NY_OPEN_ET) & (et_mins < NY_CLOSE_ET)
    rth = df[mask].copy()
    rth['date'] = rth.index.date
    daily = rth.groupby('date').agg(
        Open=('Open', 'first'), High=('High', 'max'),
        Low=('Low', 'min'), Close=('Close', 'last'),
        Volume=('Volume', 'sum'),
    )
    daily['Range'] = daily['High'] - daily['Low']
    return daily


def main():
    print("=" * 80)
    print("OOS REGIME ANALYSIS — WHY DO ALL STRATEGIES FAIL?")
    print("OOS: Feb 17 – Aug 17, 2025  |  IS: Aug 17, 2025 – Feb 13, 2026")
    print("=" * 80)

    print("\nLoading data...")
    mnq_oos = load_databento(DATA_DIR / FILES['MNQ_OOS'], MNQ_SM)
    mnq_is  = load_databento(DATA_DIR / FILES['MNQ_IS'], MNQ_SM)
    mes_oos = load_databento(DATA_DIR / FILES['MES_OOS'], MES_SM)
    mes_is  = load_databento(DATA_DIR / FILES['MES_IS'], MES_SM)

    print("Preparing arrays...")
    mnq_oos_arr = prepare_arrays(mnq_oos, rsi_len=8)
    mnq_is_arr  = prepare_arrays(mnq_is, rsi_len=8)
    mes_oos_arr = prepare_arrays(mes_oos, rsi_len=10)
    mes_is_arr  = prepare_arrays(mes_is, rsi_len=10)

    # ================================================================
    # 1. VOLATILITY
    # ================================================================
    print("\n" + "=" * 80)
    print("1. VOLATILITY REGIME")
    print("=" * 80)

    for inst, oos_df, is_df in [("MNQ", mnq_oos, mnq_is), ("MES", mes_oos, mes_is)]:
        d_oos = get_daily_stats(oos_df)
        d_is = get_daily_stats(is_df)
        print(f"\n  {inst}:")
        print(f"    {'':12} {'OOS':>10} {'IS':>10} {'Ratio':>8}")
        print(f"    {'Avg Range':12} {d_oos['Range'].mean():>10.1f} {d_is['Range'].mean():>10.1f} {d_oos['Range'].mean()/d_is['Range'].mean():>8.2f}x")
        print(f"    {'Med Range':12} {d_oos['Range'].median():>10.1f} {d_is['Range'].median():>10.1f} {d_oos['Range'].median()/d_is['Range'].median():>8.2f}x")
        print(f"    {'Std Range':12} {d_oos['Range'].std():>10.1f} {d_is['Range'].std():>10.1f}")
        print(f"    {'Avg Volume':12} {d_oos['Volume'].mean():>10,.0f} {d_is['Volume'].mean():>10,.0f} {d_oos['Volume'].mean()/d_is['Volume'].mean():>8.2f}x")
        print(f"    {'Trading Days':12} {len(d_oos):>10d} {len(d_is):>10d}")

        # Monthly range
        d_oos_copy = d_oos.copy()
        d_oos_copy.index = pd.to_datetime(d_oos_copy.index)
        d_is_copy = d_is.copy()
        d_is_copy.index = pd.to_datetime(d_is_copy.index)
        dall = pd.concat([d_oos_copy, d_is_copy])
        dall['month'] = dall.index.to_period('M')
        monthly = dall.groupby('month')['Range'].agg(['mean', 'std', 'count'])
        print(f"\n    Monthly avg daily range ({inst}):")
        for m, row in monthly.iterrows():
            period = "OOS" if m <= pd.Period("2025-08") else "IS "
            print(f"      {m}  {row['mean']:>7.1f} pts  (std {row['std']:>6.1f}, {int(row['count'])} days)  [{period}]")

    # ================================================================
    # 2. SM SIGNAL QUALITY AT ENTRIES
    # ================================================================
    print("\n" + "=" * 80)
    print("2. SM SIGNAL QUALITY AT ENTRIES")
    print("=" * 80)

    # v11.1 trades
    for label, arr, comm, dpp, params in [
        ("MNQ v11.1", mnq_oos_arr, MNQ_COMM, MNQ_DPP,
         dict(rsi_buy=60, rsi_sell=40, sm_threshold=0.15, cooldown_bars=20, max_loss_pts=50)),
    ]:
        print(f"\n  {label}:")
        for period, a in [("OOS", mnq_oos_arr), ("IS ", mnq_is_arr)]:
            trades = run_backtest_v10(
                a['opens'], a['highs'], a['lows'], a['closes'],
                a['sm'], a['rsi'], a['times'],
                rsi_5m_curr=a['rsi_5m_curr'], rsi_5m_prev=a['rsi_5m_prev'], **params,
            )
            sm_at_entry = [a['sm'][t['entry_idx']-1] for t in trades if t['entry_idx'] > 0]
            abs_sm = np.abs(sm_at_entry)
            print(f"    {period} ({len(trades)} trades): |SM| mean={np.mean(abs_sm):.4f} "
                  f"median={np.median(abs_sm):.4f} std={np.std(abs_sm):.4f}")
            # Bucket distribution
            buckets = [(0.15, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]
            parts = []
            for lo, hi in buckets:
                cnt = np.sum((abs_sm >= lo) & (abs_sm < hi))
                pct = cnt / len(abs_sm) * 100
                parts.append(f"[{lo:.2f}-{hi:.2f}]:{cnt}({pct:.0f}%)")
            print(f"         Buckets: {' '.join(parts)}")

    # Global SM distribution (all RTH bars)
    print(f"\n  Global SM distribution (all RTH bars):")
    for inst, oos_arr, is_arr in [("MNQ", mnq_oos_arr, mnq_is_arr), ("MES", mes_oos_arr, mes_is_arr)]:
        for period, arr in [("OOS", oos_arr), ("IS ", is_arr)]:
            et_m = compute_et_minutes(arr['times'])
            mask = (et_m >= NY_OPEN_ET) & (et_m < NY_CLOSE_ET)
            sm_sess = np.abs(arr['sm'][mask])
            sm_sess = sm_sess[~np.isnan(sm_sess)]
            print(f"    {inst} {period}: |SM| mean={np.mean(sm_sess):.4f} "
                  f"median={np.median(sm_sess):.4f} "
                  f"pct>0.15={np.mean(sm_sess > 0.15)*100:.1f}% "
                  f"pct>0.30={np.mean(sm_sess > 0.30)*100:.1f}%")

    # ================================================================
    # 3. SL ANALYSIS FOR v11.1
    # ================================================================
    print("\n" + "=" * 80)
    print("3. SL ANALYSIS — MNQ v11.1 (SL=50 pts)")
    print("=" * 80)

    for period, arr in [("OOS", mnq_oos_arr), ("IS ", mnq_is_arr)]:
        trades = run_backtest_v10(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['rsi'], arr['times'],
            rsi_buy=60, rsi_sell=40, sm_threshold=0.15, cooldown_bars=20, max_loss_pts=50,
            rsi_5m_curr=arr['rsi_5m_curr'], rsi_5m_prev=arr['rsi_5m_prev'],
        )
        trades = compute_mfe_mae(trades, arr['highs'], arr['lows'])

        sl = [t for t in trades if t['result'] == 'SL']
        non_sl = [t for t in trades if t['result'] != 'SL']
        comm_pts = (MNQ_COMM * 2) / MNQ_DPP

        print(f"\n  {period} ({len(trades)} trades):")
        print(f"    SL: {len(sl)} ({len(sl)/len(trades)*100:.1f}%)")
        if sl:
            print(f"      Avg pts:  {np.mean([t['pts'] for t in sl]):+.2f}")
            print(f"      Avg MAE:  {np.mean([t['mae'] for t in sl]):.1f} pts")
            print(f"      Avg MFE:  {np.mean([t['mfe'] for t in sl]):.1f} pts (was profitable before SL)")
            print(f"      Avg bars: {np.mean([t['bars'] for t in sl]):.0f}")
        if non_sl:
            winners = [t for t in non_sl if t['pts'] - comm_pts > 0]
            losers = [t for t in non_sl if t['pts'] - comm_pts <= 0]
            print(f"    Non-SL: {len(non_sl)} (WR {len(winners)/len(non_sl)*100:.1f}%)")
            print(f"      Avg pts:  {np.mean([t['pts'] for t in non_sl]):+.2f}")
            print(f"      Avg MFE:  {np.mean([t['mfe'] for t in non_sl]):.1f} pts")

        # Monthly SL breakdown
        print(f"    Monthly SL breakdown:")
        by_month = defaultdict(list)
        for t in trades:
            m = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
            by_month[m].append(t)
        for m in sorted(by_month.keys()):
            mt = by_month[m]
            sl_m = sum(1 for t in mt if t['result'] == 'SL')
            wr_m = sum(1 for t in mt if t['pts'] - comm_pts > 0) / len(mt) * 100
            print(f"      {m}: {len(mt):>3d} trades, {sl_m:>2d} SL ({sl_m/len(mt)*100:>5.1f}%), WR {wr_m:.0f}%")

    # ================================================================
    # 4. DIRECTIONAL BIAS
    # ================================================================
    print("\n" + "=" * 80)
    print("4. DIRECTIONAL BIAS — Long vs Short")
    print("=" * 80)

    configs = [
        ("MNQ v11.1", "v10", MNQ_COMM, MNQ_DPP,
         [(mnq_oos_arr, "OOS"), (mnq_is_arr, "IS ")],
         dict(rsi_buy=60, rsi_sell=40, sm_threshold=0.15, cooldown_bars=20, max_loss_pts=50)),
        ("MNQ v15", "v15", MNQ_COMM, MNQ_DPP,
         [(mnq_oos_arr, "OOS"), (mnq_is_arr, "IS ")],
         dict(rsi_buy=60, rsi_sell=40, sm_threshold=0.0, cooldown_bars=20, max_loss_pts=50, tp_pts=5)),
        ("MES v9.4", "v94", MES_COMM, MES_DPP,
         [(mes_oos_arr, "OOS"), (mes_is_arr, "IS ")],
         dict(rsi_buy=55, rsi_sell=45, cooldown_bars=15, eod_minutes_et=930)),
    ]

    print(f"\n  {'Strategy':<12} {'Period':>5} {'Side':>6} {'Trades':>6} {'WR%':>6} {'Net pts':>10} {'Avg pts':>8}")
    print("  " + "-" * 65)

    for strat_name, engine, comm, dpp, arr_list, params in configs:
        for arr, period in arr_list:
            if engine == "v10":
                trades = run_backtest_v10(
                    arr['opens'], arr['highs'], arr['lows'], arr['closes'],
                    arr['sm'], arr['rsi'], arr['times'],
                    rsi_5m_curr=arr['rsi_5m_curr'], rsi_5m_prev=arr['rsi_5m_prev'], **params)
            elif engine == "v15":
                trades = run_backtest_tp_exit(
                    arr['opens'], arr['highs'], arr['lows'], arr['closes'],
                    arr['sm'], arr['times'],
                    arr['rsi_5m_curr'], arr['rsi_5m_prev'], **params)
            else:
                trades = run_mes_v94(
                    arr['opens'], arr['highs'], arr['lows'], arr['closes'],
                    arr['sm'], arr['times'],
                    arr['rsi_5m_curr'], arr['rsi_5m_prev'], **params)

            cp = (comm * 2) / dpp
            for side in ["long", "short"]:
                st = [t for t in trades if t['side'] == side]
                if not st: continue
                net_each = np.array([t['pts'] - cp for t in st])
                wins = np.sum(net_each > 0)
                wr = wins / len(st) * 100
                print(f"  {strat_name:<12} {period:>5} {side:>6} {len(st):>6d} {wr:>5.1f}% "
                      f"{net_each.sum():>+10.2f} {net_each.mean():>+8.2f}")

    # ================================================================
    # 5. MARKET REGIME
    # ================================================================
    print("\n" + "=" * 80)
    print("5. MARKET REGIME — Price Levels & Returns")
    print("=" * 80)

    d_oos = get_daily_stats(mnq_oos)
    d_is = get_daily_stats(mnq_is)
    d_oos.index = pd.to_datetime(d_oos.index)
    d_is.index = pd.to_datetime(d_is.index)
    dall = pd.concat([d_oos, d_is])
    dall['month'] = dall.index.to_period('M')

    monthly = dall.groupby('month').agg(
        AvgClose=('Close', 'mean'), Open=('Open', 'first'),
        Close=('Close', 'last'), High=('High', 'max'),
        Low=('Low', 'min'), AvgRange=('Range', 'mean'),
        Days=('Close', 'count'),
    )
    monthly['Return%'] = (monthly['Close'] - monthly['Open']) / monthly['Open'] * 100
    monthly['MaxDraw%'] = (monthly['Low'] - monthly['High']) / monthly['High'] * 100

    print(f"\n  {'Month':<10} {'AvgClose':>10} {'Return%':>9} {'AvgRange':>9} {'MaxDraw%':>9} {'Period':>6}")
    print("  " + "-" * 60)
    for m, row in monthly.iterrows():
        per = "OOS" if m <= pd.Period("2025-08") else "IS "
        print(f"  {str(m):<10} {row['AvgClose']:>10,.0f} {row['Return%']:>+8.2f}% "
              f"{row['AvgRange']:>8.1f} {row['MaxDraw%']:>+8.2f}% {per:>6}")

    # Big-move days
    overall_avg = dall['Range'].mean()
    bm_oos = (d_oos['Range'] > 2 * overall_avg).sum()
    bm_is = (d_is['Range'] > 2 * overall_avg).sum()
    print(f"\n  Big-move days (range > {2*overall_avg:.0f} pts = 2x avg):")
    print(f"    OOS: {bm_oos} days ({bm_oos/len(d_oos)*100:.1f}%)")
    print(f"    IS:  {bm_is} days ({bm_is/len(d_is)*100:.1f}%)")

    # Price range extremes
    print(f"\n  Price extremes:")
    print(f"    OOS: {d_oos['Low'].min():,.0f} – {d_oos['High'].max():,.0f} "
          f"(drawdown {(d_oos['Low'].min()-d_oos['High'].max())/d_oos['High'].max()*100:+.1f}%)")
    print(f"    IS:  {d_is['Low'].min():,.0f} – {d_is['High'].max():,.0f} "
          f"(drawdown {(d_is['Low'].min()-d_is['High'].max())/d_is['High'].max()*100:+.1f}%)")

    # ================================================================
    # 6. SYNTHESIS
    # ================================================================
    print("\n" + "=" * 80)
    print("6. SYNTHESIS: OVERFITTING vs REGIME CHANGE")
    print("=" * 80)

    # Gather evidence
    vol_ratio = d_oos['Range'].mean() / d_is['Range'].mean()

    # v11.1 trades for analysis
    v11_oos = run_backtest_v10(
        mnq_oos_arr['opens'], mnq_oos_arr['highs'], mnq_oos_arr['lows'], mnq_oos_arr['closes'],
        mnq_oos_arr['sm'], mnq_oos_arr['rsi'], mnq_oos_arr['times'],
        rsi_buy=60, rsi_sell=40, sm_threshold=0.15, cooldown_bars=20, max_loss_pts=50,
        rsi_5m_curr=mnq_oos_arr['rsi_5m_curr'], rsi_5m_prev=mnq_oos_arr['rsi_5m_prev'],
    )
    v11_is = run_backtest_v10(
        mnq_is_arr['opens'], mnq_is_arr['highs'], mnq_is_arr['lows'], mnq_is_arr['closes'],
        mnq_is_arr['sm'], mnq_is_arr['rsi'], mnq_is_arr['times'],
        rsi_buy=60, rsi_sell=40, sm_threshold=0.15, cooldown_bars=20, max_loss_pts=50,
        rsi_5m_curr=mnq_is_arr['rsi_5m_curr'], rsi_5m_prev=mnq_is_arr['rsi_5m_prev'],
    )

    sl_oos = sum(1 for t in v11_oos if t['result'] == 'SL')
    sl_is = sum(1 for t in v11_is if t['result'] == 'SL')
    sl_rate_oos = sl_oos / len(v11_oos) * 100
    sl_rate_is = sl_is / len(v11_is) * 100

    sm_oos = np.abs([mnq_oos_arr['sm'][t['entry_idx']-1] for t in v11_oos if t['entry_idx'] > 0])
    sm_is = np.abs([mnq_is_arr['sm'][t['entry_idx']-1] for t in v11_is if t['entry_idx'] > 0])

    cp = (MNQ_COMM * 2) / MNQ_DPP
    long_oos = [t for t in v11_oos if t['side'] == 'long']
    short_oos = [t for t in v11_oos if t['side'] == 'short']
    long_is = [t for t in v11_is if t['side'] == 'long']
    short_is = [t for t in v11_is if t['side'] == 'short']

    long_wr_oos = np.mean([1 if t['pts'] - cp > 0 else 0 for t in long_oos]) * 100 if long_oos else 0
    short_wr_oos = np.mean([1 if t['pts'] - cp > 0 else 0 for t in short_oos]) * 100 if short_oos else 0
    long_wr_is = np.mean([1 if t['pts'] - cp > 0 else 0 for t in long_is]) * 100 if long_is else 0
    short_wr_is = np.mean([1 if t['pts'] - cp > 0 else 0 for t in short_is]) * 100 if short_is else 0

    # Monthly P&L
    monthly_pnl = defaultdict(float)
    for t in v11_oos:
        m = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        monthly_pnl[m] += (t['pts'] - cp) * MNQ_DPP
    losing_months = sum(1 for v in monthly_pnl.values() if v < 0)

    print(f"\n  EVIDENCE:")
    print(f"")

    evidence_regime, evidence_overfit = 0, 0

    # Volatility
    print(f"  1. Volatility: OOS avg range = {d_oos['Range'].mean():.1f} pts, "
          f"IS = {d_is['Range'].mean():.1f} pts ({vol_ratio:.2f}x)")
    if vol_ratio > 1.15:
        print(f"     -> [REGIME] Higher volatility in OOS = more SL hits")
        evidence_regime += 1
    elif vol_ratio < 0.85:
        print(f"     -> [REGIME] Lower volatility in OOS")
        evidence_regime += 1
    else:
        print(f"     -> [NEUTRAL] Similar volatility")

    # SL rate
    print(f"\n  2. SL rate: OOS = {sl_rate_oos:.1f}% ({sl_oos}/{len(v11_oos)}), "
          f"IS = {sl_rate_is:.1f}% ({sl_is}/{len(v11_is)})")
    if sl_rate_oos > sl_rate_is * 1.5:
        print(f"     -> [REGIME] SL rate {sl_rate_oos/sl_rate_is:.1f}x higher in OOS")
        evidence_regime += 1
    else:
        print(f"     -> [NEUTRAL] SL rates comparable")

    # SM quality
    print(f"\n  3. SM at entry: OOS |SM| = {np.mean(sm_oos):.4f}, IS = {np.mean(sm_is):.4f}")
    if np.mean(sm_oos) < np.mean(sm_is) * 0.85:
        print(f"     -> [REGIME] SM signals weaker in OOS")
        evidence_regime += 1
    elif np.mean(sm_oos) > np.mean(sm_is) * 1.15:
        print(f"     -> [OVERFIT] SM stronger in OOS but still losing")
        evidence_overfit += 1
    else:
        print(f"     -> [NEUTRAL] SM signal strength similar")

    # Directional
    long_drop = long_wr_is - long_wr_oos
    short_drop = short_wr_is - short_wr_oos
    print(f"\n  4. Direction: Long WR drop = {long_drop:+.1f}pp, Short WR drop = {short_drop:+.1f}pp")
    print(f"     OOS: L={long_wr_oos:.0f}% ({len(long_oos)}t), S={short_wr_oos:.0f}% ({len(short_oos)}t)")
    print(f"     IS : L={long_wr_is:.0f}% ({len(long_is)}t), S={short_wr_is:.0f}% ({len(short_is)}t)")
    if abs(long_drop - short_drop) > 10:
        worse = "longs" if long_drop > short_drop else "shorts"
        print(f"     -> [REGIME] Asymmetric: {worse} degraded much more")
        evidence_regime += 1
    else:
        print(f"     -> [OVERFIT] Both sides degraded equally")
        evidence_overfit += 1

    # Months
    print(f"\n  5. Losing months: {losing_months}/{len(monthly_pnl)} OOS months lost money")
    if losing_months >= len(monthly_pnl) * 0.7:
        print(f"     -> [OVERFIT] Broad failure across most months")
        evidence_overfit += 1
    else:
        print(f"     -> [REGIME] Losses concentrated in specific months")
        evidence_regime += 1

    # Market events
    oos_draw = (d_oos['Low'].min() - d_oos['High'].max()) / d_oos['High'].max() * 100
    is_draw = (d_is['Low'].min() - d_is['High'].max()) / d_is['High'].max() * 100
    print(f"\n  6. Market drawdown: OOS = {oos_draw:+.1f}%, IS = {is_draw:+.1f}%")
    if abs(oos_draw) > abs(is_draw) * 1.5:
        print(f"     -> [REGIME] OOS had much larger market drawdown")
        evidence_regime += 1
    else:
        print(f"     -> [NEUTRAL] Similar market drawdown")

    print(f"\n  {'='*60}")
    print(f"  SCORE: REGIME CHANGE = {evidence_regime}, OVERFITTING = {evidence_overfit}")
    print(f"  {'='*60}")
    if evidence_regime > evidence_overfit:
        print(f"  VERDICT: Primarily REGIME CHANGE")
        print(f"  The OOS period had materially different market conditions.")
        print(f"  Strategies may work in similar-regime periods but aren't all-weather.")
    elif evidence_overfit > evidence_regime:
        print(f"  VERDICT: Primarily OVERFITTING")
        print(f"  Parameters were fitted to IS noise. Strategy logic is edge-weak.")
    else:
        print(f"  VERDICT: MIXED — both regime change AND parameter sensitivity")
        print(f"  Strategies have some edge but it's narrow and regime-dependent.")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
