"""
TradingView Validation Preview
================================
Runs v11MNQ and v94MES on the Jan 19 - Feb 13 window and prints
trade-by-trade details so you know exactly what to expect in TradingView.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
)


def load_databento(instrument):
    files = {
        'MNQ': 'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
        'MES': 'databento_MES_1min_2025-08-17_to_2026-02-13.csv',
    }
    filepath = Path(__file__).resolve().parent.parent / "data" / files[instrument]
    df = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    return result


def run_and_print(instrument, label, sm_params, rsi_len, rsi_buy, rsi_sell,
                  cooldown, max_loss_pts, commission, dollar_per_pt):
    """Run backtest on full data (for SM warmup) but only score Jan 19 - Feb 13 trades."""
    print(f"\n{'='*90}")
    print(f"{label} on {instrument}")
    print(f"SM({sm_params[0]}/{sm_params[1]}/{sm_params[2]}/{sm_params[3]})  "
          f"RSI({rsi_len}/{rsi_buy}/{rsi_sell})  CD={cooldown}  "
          f"SL={'OFF' if max_loss_pts == 0 else str(max_loss_pts)+'pts'}")
    print(f"Commission: ${commission}/side  |  ${dollar_per_pt}/point")
    print(f"{'='*90}")

    df = load_databento(instrument)

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    volumes = df['Volume'].values
    times = df.index.values

    # Compute SM on full data (warmup needs history)
    sm = compute_smart_money(closes, volumes, *sm_params)

    # Resample to 5-min for RSI
    df_r = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_r['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_r)
    fivemin_times = df_5m.index.values
    fivemin_closes = df_5m['Close'].values

    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, fivemin_times, fivemin_closes, rsi_len=rsi_len)

    # Run backtest on full data
    trades = run_backtest_v10(
        opens, highs, lows, closes, sm, rsi_curr, times,
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=0.0, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )

    # Filter to Jan 19 - Feb 13 window
    start = pd.Timestamp("2026-01-19")
    end = pd.Timestamp("2026-02-14")  # Feb 13 inclusive
    filtered = [t for t in trades
                if start <= pd.Timestamp(t['entry_time']) < end]

    # Score
    sc = score_trades(filtered, commission_per_side=commission,
                      dollar_per_pt=dollar_per_pt)

    # Print summary
    if sc:
        print(f"\nSUMMARY: {sc['count']} trades | WR {sc['win_rate']:.1f}% | "
              f"PF {sc['pf']:.3f} | Net ${sc['net_dollar']:+.2f} | "
              f"MaxDD ${sc['max_dd_dollar']:.2f}")
        if sc.get('exits'):
            exits_str = "  ".join(f"{k}:{v}" for k, v in sorted(sc['exits'].items()))
            print(f"  Exits: {exits_str}")

    # Print trade-by-trade
    print(f"\n{'#':>3}  {'Entry Time':>20}  {'Dir':>5}  {'Entry':>10}  "
          f"{'Exit':>10}  {'Pts':>8}  {'Net$':>9}  {'Exit Reason':>12}  {'Bars':>5}")
    print(f"  {'-'*98}")

    cum_pnl = 0.0
    for i, t in enumerate(filtered, 1):
        side = t['side']
        entry_px = t['entry']
        exit_px = t['exit']
        pts = (exit_px - entry_px) if side == 'long' else (entry_px - exit_px)
        gross = pts * dollar_per_pt
        net = gross - 2 * commission
        cum_pnl += net
        bars = t.get('bars', 0)
        reason = t.get('exit_reason', '?')

        entry_ts = pd.Timestamp(t['entry_time'])
        entry_str = entry_ts.strftime('%m/%d %H:%M')

        print(f"{i:>3}  {entry_str:>20}  {side:>5}  {entry_px:>10.2f}  "
              f"{exit_px:>10.2f}  {pts:>+8.2f}  ${net:>+8.2f}  {reason:>12}  {bars:>5}")

    print(f"\n  Cumulative Net P&L: ${cum_pnl:+.2f}")
    return filtered, sc


def main():
    print("TRADINGVIEW VALIDATION PREVIEW")
    print("Date range: Jan 19 - Feb 13, 2026")
    print("Run these strategies in TV on the same dates and compare.\n")

    # v11MNQ
    run_and_print(
        instrument='MNQ',
        label='v11 MNQ',
        sm_params=(10, 12, 200, 100),
        rsi_len=8, rsi_buy=60, rsi_sell=40,
        cooldown=20, max_loss_pts=50,
        commission=0.52, dollar_per_pt=2.0,
    )

    # v94MES
    run_and_print(
        instrument='MES',
        label='v9.4 MES',
        sm_params=(20, 12, 400, 255),
        rsi_len=10, rsi_buy=55, rsi_sell=45,
        cooldown=15, max_loss_pts=0,
        commission=0.52, dollar_per_pt=5.0,
    )


if __name__ == "__main__":
    main()
