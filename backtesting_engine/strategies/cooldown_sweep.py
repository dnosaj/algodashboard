"""
Cooldown Sweep: How much does CD=20 cost us in missed trades?
=============================================================
Runs v11 MNQ with CD = 0, 5, 10, 15, 20, 25, 30, 40, 60
to show exactly what the cooldown does.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_test_common import (
    compute_smart_money, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
)

DATA_FILE = (Path(__file__).resolve().parent.parent
             / "data" / "databento_MNQ_1min_2025-08-17_to_2026-02-13.csv")

COMMISSION = 0.52
DOLLAR_PER_PT = 2.0

# v11 params (all fixed except cooldown)
SM_INDEX, SM_FLOW, SM_NORM, SM_EMA = 10, 12, 200, 100
RSI_LEN, RSI_BUY, RSI_SELL = 8, 60, 40
MAX_LOSS_PTS = 50


def main():
    print("=" * 110)
    print("  COOLDOWN SWEEP: v11 MNQ with varying cooldown values")
    print("  All other params fixed: SM(10/12/200/100) RSI(8/60/40) SL=50")
    print("=" * 110)

    # Load data
    df = pd.read_csv(DATA_FILE)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')

    closes = result['Close'].values
    volumes = result['Volume'].values
    sm = compute_smart_money(closes, volumes, SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)

    opens = result['Open'].values
    highs = result['High'].values
    lows = result['Low'].values
    times = result.index.values

    df_for_5m = result[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_5m['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_for_5m)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=RSI_LEN)

    # =========================================================================
    # Sweep cooldowns
    # =========================================================================
    cooldowns = [0, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 60]

    results = []
    for cd in cooldowns:
        trades = run_backtest_v10(
            opens, highs, lows, closes, sm, rsi_curr, times,
            rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
            sm_threshold=0.0, cooldown_bars=cd,
            max_loss_pts=MAX_LOSS_PTS,
            rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
        )
        sc = score_trades(trades, commission_per_side=COMMISSION,
                          dollar_per_pt=DOLLAR_PER_PT)

        n_sl = sum(1 for t in trades if t['result'] == 'SL')
        n_sm = sum(1 for t in trades if t['result'] == 'SM_FLIP')
        n_eod = sum(1 for t in trades if t['result'] == 'EOD')

        # Compute avg trade P&L
        if sc and sc['count'] > 0:
            avg_pnl = sc['net_dollar'] / sc['count']
        else:
            avg_pnl = 0

        results.append({
            'cd': cd,
            'trades': sc['count'] if sc else 0,
            'wr': sc['win_rate'] if sc else 0,
            'pf': sc['pf'] if sc else 0,
            'net': sc['net_dollar'] if sc else 0,
            'max_dd': sc['max_dd_dollar'] if sc else 0,
            'sharpe': sc.get('sharpe', 0) if sc else 0,
            'sl': n_sl,
            'sm': n_sm,
            'eod': n_eod,
            'avg_pnl': avg_pnl,
        })

    # =========================================================================
    # Print results
    # =========================================================================
    print(f"\n  {'CD':>4}  {'Trades':>6}  {'WR':>6}  {'PF':>6}  {'Net $':>10}  "
          f"{'MaxDD':>8}  {'Sharpe':>7}  {'SL':>4}  {'SM':>4}  {'EOD':>4}  "
          f"{'Avg$/Tr':>8}  {'Note':>15}")
    print(f"  {'-' * 105}")

    cd20_net = None
    cd20_trades = None
    for r in results:
        note = ""
        if r['cd'] == 20:
            note = "<-- CURRENT"
            cd20_net = r['net']
            cd20_trades = r['trades']
        elif r['cd'] == 15:
            note = "(old v9.4)"

        print(f"  {r['cd']:>4}  {r['trades']:>6}  {r['wr']:>5.1f}%  {r['pf']:>6.3f}  "
              f"${r['net']:>+9.2f}  ${r['max_dd']:>7.2f}  {r['sharpe']:>7.3f}  "
              f"{r['sl']:>4}  {r['sm']:>4}  {r['eod']:>4}  "
              f"${r['avg_pnl']:>+7.2f}  {note:>15}")

    # =========================================================================
    # Delta analysis: what does CD=20 block vs CD=0?
    # =========================================================================
    print(f"\n{'=' * 110}")
    print(f"  TRADE DELTA ANALYSIS: CD=20 vs CD=0")
    print(f"{'=' * 110}")

    # Run both and compare
    trades_cd0 = run_backtest_v10(
        opens, highs, lows, closes, sm, rsi_curr, times,
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=0.0, cooldown_bars=0,
        max_loss_pts=MAX_LOSS_PTS,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )
    trades_cd20 = run_backtest_v10(
        opens, highs, lows, closes, sm, rsi_curr, times,
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=0.0, cooldown_bars=20,
        max_loss_pts=MAX_LOSS_PTS,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )

    # Find trades in CD=0 that don't exist in CD=20
    cd20_entries = set()
    for t in trades_cd20:
        cd20_entries.add(t['entry_idx'])

    blocked_trades = []
    for t in trades_cd0:
        if t['entry_idx'] not in cd20_entries:
            entry_ts = pd.Timestamp(t['entry_time'])
            exit_ts = pd.Timestamp(t['exit_time'])
            pnl_pts = t['pts']
            pnl_dollar = pnl_pts * DOLLAR_PER_PT - 2 * COMMISSION
            blocked_trades.append({
                'entry_time': entry_ts,
                'exit_time': exit_ts,
                'side': t['side'],
                'entry_price': t['entry'],
                'pnl_pts': pnl_pts,
                'pnl_dollar': pnl_dollar,
                'exit_reason': t['result'],
                'bars': t['bars'],
            })

    print(f"\n  CD=0: {len(trades_cd0)} trades")
    print(f"  CD=20: {len(trades_cd20)} trades")
    print(f"  Blocked by cooldown: {len(blocked_trades)} trades")

    if blocked_trades:
        blocked_df = pd.DataFrame(blocked_trades)
        total_blocked_pnl = blocked_df['pnl_dollar'].sum()
        blocked_wins = (blocked_df['pnl_dollar'] > 0).sum()
        blocked_losses = (blocked_df['pnl_dollar'] <= 0).sum()
        blocked_sl = (blocked_df['exit_reason'] == 'SL').sum()

        print(f"\n  Blocked trade summary:")
        print(f"    Total P&L if taken:    ${total_blocked_pnl:+.2f}")
        print(f"    Winners:               {blocked_wins}")
        print(f"    Losers:                {blocked_losses}")
        print(f"    Win rate:              {blocked_wins/len(blocked_df)*100:.1f}%")
        print(f"    SL stops among blocked:{blocked_sl}")
        print(f"    Avg P&L per blocked:   ${blocked_df['pnl_dollar'].mean():+.2f}")

        # Exit reason breakdown of blocked trades
        print(f"\n  Blocked trade exit reasons:")
        for reason in blocked_df['exit_reason'].unique():
            n = (blocked_df['exit_reason'] == reason).sum()
            rpnl = blocked_df.loc[blocked_df['exit_reason'] == reason, 'pnl_dollar'].sum()
            print(f"    {reason:>10}: {n:>4} trades, P&L ${rpnl:>+8.2f}")

        # Show the blocked trades
        print(f"\n  All {len(blocked_trades)} blocked trades:")
        print(f"  {'Entry Date':>16}  {'Side':>5}  {'Entry':>10}  {'PnL pts':>8}  "
              f"{'PnL $':>8}  {'Exit':>8}  {'Bars':>5}")
        print(f"  {'-' * 70}")
        for bt in sorted(blocked_trades, key=lambda x: x['entry_time']):
            print(f"  {bt['entry_time'].strftime('%m/%d %H:%M'):>16}  {bt['side']:>5}  "
                  f"{bt['entry_price']:>10.2f}  {bt['pnl_pts']:>+7.2f}  "
                  f"${bt['pnl_dollar']:>+7.2f}  {bt['exit_reason']:>8}  {bt['bars']:>5}")

        # Monthly breakdown of blocked trades
        blocked_df['month'] = blocked_df['entry_time'].dt.to_period('M')
        print(f"\n  Blocked trades by month:")
        for m in sorted(blocked_df['month'].unique()):
            mdf = blocked_df[blocked_df['month'] == m]
            print(f"    {str(m):>10}: {len(mdf)} trades, P&L ${mdf['pnl_dollar'].sum():+.2f}")

    # =========================================================================
    # Efficiency analysis
    # =========================================================================
    print(f"\n{'=' * 110}")
    print(f"  EFFICIENCY: Net $ per trade at each cooldown")
    print(f"{'=' * 110}")
    print(f"\n  The best cooldown isn't the one with the most trades or highest gross --")
    print(f"  it's the one with the highest NET $ per trade (quality over quantity).\n")

    best_avg = max(results, key=lambda r: r['avg_pnl'])
    best_net = max(results, key=lambda r: r['net'])
    best_pf = max(results, key=lambda r: r['pf'])

    print(f"  Best avg $/trade: CD={best_avg['cd']} (${best_avg['avg_pnl']:+.2f}/trade)")
    print(f"  Best total net:   CD={best_net['cd']} (${best_net['net']:+.2f})")
    print(f"  Best PF:          CD={best_pf['cd']} (PF {best_pf['pf']:.3f})")


if __name__ == "__main__":
    main()
