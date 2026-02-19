"""
Debug SL Clustering: Analyze stop-loss patterns across 6-month history
======================================================================
Answers: "Is the Feb 12-17 regime (8 ML stops in 5 days) unprecedented?"

Strategy: MNQ v11 SM(10/12/200/100), RSI(8/60/40), CD=20, SL=50
Data: Databento MNQ 1-min, Aug 17 2025 - Feb 13 2026

Analysis:
  1. Per-trade detail: direction, entry/exit times, PnL, exit reason
  2. Daily P&L with SL count per day
  3. Worst rolling windows (1/3/5/10-day)
  4. Consecutive SL streaks
  5. SL stops by month
  6. Comparison with TradingView Feb 12-17 actuals
  7. Summary: is Feb 12-17 unprecedented?
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    fmt_score, fmt_exits,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_FILE = (Path(__file__).resolve().parent.parent
             / "data" / "databento_MNQ_1min_2025-08-17_to_2026-02-13.csv")

COMMISSION = 0.52
DOLLAR_PER_PT = 2.0  # MNQ

# v11 MNQ params
SM_INDEX = 10
SM_FLOW = 12
SM_NORM = 200
SM_EMA = 100
RSI_LEN = 8
RSI_BUY = 60
RSI_SELL = 40
COOLDOWN = 20
MAX_LOSS_PTS = 50


def load_mnq_databento():
    """Load MNQ 1-min Databento data."""
    df = pd.read_csv(DATA_FILE)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    return result


def main():
    print("=" * 120)
    print("  DEBUG SL CLUSTERING: Stop-Loss Pattern Analysis (6-month MNQ)")
    print("  MNQ v11: SM(10/12/200/100), RSI(8/60/40), CD=20, SL=50")
    print("=" * 120)

    # =========================================================================
    # 1. Load data and run backtest
    # =========================================================================
    print("\n[1] Loading MNQ 1-min Databento data and running backtest...")
    df = load_mnq_databento()
    print(f"    {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # Compute SM with v11 params
    closes_full = df['Close'].values
    volumes_full = df['Volume'].values
    sm_full = compute_smart_money(closes_full, volumes_full,
                                  SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)

    # Prepare 1-min arrays with 5-min RSI mapping
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    times = df.index.values

    df_for_5m = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_5m['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_for_5m)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=RSI_LEN)

    # Run backtest
    trades = run_backtest_v10(
        opens, highs, lows, closes, sm_full, rsi_curr, times,
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=0.0, cooldown_bars=COOLDOWN,
        max_loss_pts=MAX_LOSS_PTS,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )
    sc = score_trades(trades, commission_per_side=COMMISSION,
                      dollar_per_pt=DOLLAR_PER_PT)
    print(f"    {fmt_score(sc, 'v11 SL=50')}")
    if sc:
        print(f"    Exit types: {fmt_exits(sc['exits'])}")

    # =========================================================================
    # 2. Build trade records with computed fields
    # =========================================================================
    print(f"\n[2] Building trade records ({len(trades)} trades)...")
    comm_pts = (COMMISSION * 2) / DOLLAR_PER_PT
    records = []
    for i, t in enumerate(trades):
        pnl_pts = t['pts'] - comm_pts
        pnl_dollar = pnl_pts * DOLLAR_PER_PT
        entry_ts = pd.Timestamp(t['entry_time'])
        exit_ts = pd.Timestamp(t['exit_time'])
        records.append({
            'trade_num': i + 1,
            'direction': t['side'].upper(),
            'entry_time': entry_ts,
            'exit_time': exit_ts,
            'entry_date': entry_ts.date(),
            'exit_date': exit_ts.date(),
            'pnl_pts': round(pnl_pts, 2),
            'pnl_dollar': round(pnl_dollar, 2),
            'exit_reason': t['result'],
        })

    df_trades = pd.DataFrame(records)

    # Quick summary of exit reasons
    print(f"\n    Exit reason breakdown:")
    for reason in df_trades['exit_reason'].unique():
        n = (df_trades['exit_reason'] == reason).sum()
        total_pnl = df_trades.loc[df_trades['exit_reason'] == reason, 'pnl_dollar'].sum()
        print(f"      {reason:>10}: {n:>4} trades, total P&L ${total_pnl:>+10.2f}")

    # =========================================================================
    # 3. Daily P&L and SL counts
    # =========================================================================
    print(f"\n{'=' * 120}")
    print(f"  [3] DAILY P&L WITH SL STOP COUNTS")
    print(f"{'=' * 120}")

    # Group by entry_date (the day the trade was opened)
    daily_pnl = df_trades.groupby('entry_date')['pnl_dollar'].sum()
    daily_sl = df_trades[df_trades['exit_reason'] == 'SL'].groupby('entry_date').size()
    daily_trades = df_trades.groupby('entry_date').size()

    all_dates = sorted(daily_pnl.index)
    cum_pnl = 0.0

    # Print all days with at least 1 SL stop
    print(f"\n  {'Date':>12}  {'# Trades':>8}  {'# SL':>5}  {'Daily P&L':>12}  {'Cum P&L':>12}")
    print(f"  {'-' * 55}")

    sl_days = []
    for d in all_dates:
        day_pnl = daily_pnl.get(d, 0.0)
        cum_pnl += day_pnl
        n_sl = daily_sl.get(d, 0)
        n_trades_day = daily_trades.get(d, 0)
        if n_sl >= 1:
            print(f"  {str(d):>12}  {n_trades_day:>8}  {n_sl:>5}  ${day_pnl:>+10.2f}  ${cum_pnl:>+10.2f}")
            sl_days.append({
                'date': d, 'n_trades': n_trades_day, 'n_sl': n_sl,
                'daily_pnl': day_pnl, 'cum_pnl': cum_pnl,
            })

    print(f"\n  Total days with >= 1 SL: {len(sl_days)} out of {len(all_dates)} trading days")
    if sl_days:
        max_sl_day = max(sl_days, key=lambda x: x['n_sl'])
        print(f"  Worst day by SL count: {max_sl_day['date']} with {max_sl_day['n_sl']} SL stops, "
              f"P&L ${max_sl_day['daily_pnl']:+.2f}")

    # =========================================================================
    # 4. Worst rolling windows (1/3/5/10-day)
    # =========================================================================
    print(f"\n{'=' * 120}")
    print(f"  [4] WORST ROLLING WINDOWS (1-day, 3-day, 5-day, 10-day)")
    print(f"{'=' * 120}")

    # Build a daily series with all calendar trading days
    daily_series = pd.Series(dtype=float)
    daily_sl_series = pd.Series(dtype=int)
    daily_trade_count = pd.Series(dtype=int)

    for d in all_dates:
        daily_series[d] = daily_pnl.get(d, 0.0)
        daily_sl_series[d] = daily_sl.get(d, 0)
        daily_trade_count[d] = daily_trades.get(d, 0)

    for window_size in [1, 3, 5, 10]:
        print(f"\n  --- Worst {window_size}-day rolling window ---")
        if len(daily_series) < window_size:
            print(f"    Not enough data for {window_size}-day window")
            continue

        worst_pnl = float('inf')
        worst_start = None
        worst_end = None
        worst_sl_count = 0
        worst_trade_count = 0

        dates_list = list(daily_series.index)
        for j in range(len(dates_list) - window_size + 1):
            window_dates = dates_list[j:j + window_size]
            window_pnl = sum(daily_series[d] for d in window_dates)
            if window_pnl < worst_pnl:
                worst_pnl = window_pnl
                worst_start = window_dates[0]
                worst_end = window_dates[-1]
                worst_sl_count = sum(daily_sl_series.get(d, 0) for d in window_dates)
                worst_trade_count = sum(daily_trade_count.get(d, 0) for d in window_dates)

        print(f"    Dates:      {worst_start} to {worst_end}")
        print(f"    Total P&L:  ${worst_pnl:+.2f}")
        print(f"    # Trades:   {worst_trade_count}")
        print(f"    # SL stops: {worst_sl_count}")

    # =========================================================================
    # 5. Consecutive SL streaks (2+ in a row)
    # =========================================================================
    print(f"\n{'=' * 120}")
    print(f"  [5] CONSECUTIVE SL STREAKS (2+ in a row)")
    print(f"{'=' * 120}")

    streaks = []
    current_streak = []

    for _, row in df_trades.iterrows():
        if row['exit_reason'] == 'SL':
            current_streak.append(row)
        else:
            if len(current_streak) >= 2:
                streaks.append(list(current_streak))
            current_streak = []

    # Don't forget trailing streak
    if len(current_streak) >= 2:
        streaks.append(list(current_streak))

    if streaks:
        print(f"\n  Found {len(streaks)} streaks of 2+ consecutive SL stops:\n")
        for s_idx, streak in enumerate(streaks):
            total_loss = sum(r['pnl_dollar'] for r in streak)
            trade_nums = [r['trade_num'] for r in streak]
            dates = [str(r['entry_time'].date()) for r in streak]
            unique_dates = sorted(set(dates))
            print(f"  Streak #{s_idx + 1}: {len(streak)} consecutive SL stops")
            print(f"    Trade numbers: {trade_nums}")
            print(f"    Dates:         {unique_dates}")
            print(f"    Total loss:    ${total_loss:+.2f}")
            for r in streak:
                print(f"      #{r['trade_num']:>3} {r['direction']:>5} "
                      f"{r['entry_time'].strftime('%m/%d %H:%M')} -> "
                      f"{r['exit_time'].strftime('%m/%d %H:%M')}  "
                      f"${r['pnl_dollar']:>+8.2f}")
            print()

        max_streak_len = max(len(s) for s in streaks)
        print(f"  Longest consecutive SL streak: {max_streak_len}")
    else:
        print(f"\n  No streaks of 2+ consecutive SL stops found.")
        max_streak_len = 0

    # Also track consecutive LOSSES (any exit, not just SL)
    print(f"\n  --- Consecutive LOSSES (any exit reason, 3+ in a row) ---")
    loss_streaks = []
    current_loss_streak = []

    for _, row in df_trades.iterrows():
        if row['pnl_dollar'] < 0:
            current_loss_streak.append(row)
        else:
            if len(current_loss_streak) >= 3:
                loss_streaks.append(list(current_loss_streak))
            current_loss_streak = []
    if len(current_loss_streak) >= 3:
        loss_streaks.append(list(current_loss_streak))

    if loss_streaks:
        print(f"\n  Found {len(loss_streaks)} streaks of 3+ consecutive losses:\n")
        for s_idx, streak in enumerate(loss_streaks):
            total_loss = sum(r['pnl_dollar'] for r in streak)
            trade_nums = [r['trade_num'] for r in streak]
            reasons = [r['exit_reason'] for r in streak]
            dates = sorted(set(str(r['entry_time'].date()) for r in streak))
            print(f"  Loss Streak #{s_idx + 1}: {len(streak)} consecutive losses")
            print(f"    Trade numbers: {trade_nums}")
            print(f"    Dates: {dates}")
            print(f"    Exit reasons: {reasons}")
            print(f"    Total loss: ${total_loss:+.2f}")
            print()

        max_loss_streak = max(len(s) for s in loss_streaks)
        print(f"  Longest consecutive loss streak: {max_loss_streak}")
    else:
        print(f"\n  No streaks of 3+ consecutive losses found.")
        max_loss_streak = 0

    # =========================================================================
    # 6. SL stops by MONTH
    # =========================================================================
    print(f"\n{'=' * 120}")
    print(f"  [6] SL STOPS BY MONTH")
    print(f"{'=' * 120}")

    df_trades['month'] = df_trades['entry_time'].dt.to_period('M')
    monthly_trades = df_trades.groupby('month').size()
    monthly_sl = df_trades[df_trades['exit_reason'] == 'SL'].groupby('month').size()
    monthly_pnl = df_trades.groupby('month')['pnl_dollar'].sum()
    monthly_sl_pnl = df_trades[df_trades['exit_reason'] == 'SL'].groupby('month')['pnl_dollar'].sum()

    print(f"\n  {'Month':>10}  {'Total':>6}  {'SL':>4}  {'SL%':>6}  {'Total P&L':>12}  {'SL P&L':>12}  {'Non-SL P&L':>12}")
    print(f"  {'-' * 75}")

    for m in sorted(monthly_trades.index):
        n_total = monthly_trades.get(m, 0)
        n_sl = monthly_sl.get(m, 0)
        sl_pct = (n_sl / n_total * 100) if n_total > 0 else 0
        m_pnl = monthly_pnl.get(m, 0)
        m_sl_pnl = monthly_sl_pnl.get(m, 0)
        m_nonsl_pnl = m_pnl - m_sl_pnl
        print(f"  {str(m):>10}  {n_total:>6}  {n_sl:>4}  {sl_pct:>5.1f}%  ${m_pnl:>+10.2f}  ${m_sl_pnl:>+10.2f}  ${m_nonsl_pnl:>+10.2f}")

    # =========================================================================
    # 7. TradingView Feb 12-17 comparison
    # =========================================================================
    print(f"\n{'=' * 120}")
    print(f"  [7] TRADINGVIEW FEB 12-17 COMPARISON")
    print(f"  TV Feb 12-17: 13 trades, 8 ML stops, ~-$598 P&L")
    print(f"{'=' * 120}")

    # --- Question A: Has 6-month data EVER seen 8 SL stops in a 5-day window? ---
    print(f"\n  --- A: Has 6-month data EVER seen 8+ SL stops in a 5-day window? ---")

    # Sliding 5-day window over daily SL counts
    max_sl_in_5day = 0
    max_sl_5day_window = None
    dates_list = list(daily_sl_series.index)

    for j in range(len(dates_list) - 4):
        window_dates = dates_list[j:j + 5]
        window_sl = sum(daily_sl_series.get(d, 0) for d in window_dates)
        if window_sl > max_sl_in_5day:
            max_sl_in_5day = window_sl
            max_sl_5day_window = (window_dates[0], window_dates[-1])

    print(f"    Max SL stops in any 5-day window: {max_sl_in_5day}")
    if max_sl_5day_window:
        print(f"    Window: {max_sl_5day_window[0]} to {max_sl_5day_window[1]}")

    if max_sl_in_5day >= 8:
        print(f"    --> YES, 8+ SL stops in 5-day window has occurred before.")
    else:
        print(f"    --> NO, 8 SL stops in 5-day window is UNPRECEDENTED in 6-month history.")
        print(f"    --> The worst 5-day window had only {max_sl_in_5day} SL stops.")

    # Also check all 5-day windows with their SL counts
    print(f"\n    Top 10 worst 5-day windows by SL count:")
    windows_5d = []
    for j in range(len(dates_list) - 4):
        window_dates = dates_list[j:j + 5]
        window_sl = sum(daily_sl_series.get(d, 0) for d in window_dates)
        window_pnl = sum(daily_series.get(d, 0) for d in window_dates)
        windows_5d.append((window_dates[0], window_dates[-1], window_sl, window_pnl))

    windows_5d.sort(key=lambda x: -x[2])
    for k, (start, end, n_sl, wpnl) in enumerate(windows_5d[:10]):
        print(f"      {k+1:>2}. {start} to {end}: {n_sl} SL stops, P&L ${wpnl:+.2f}")

    # --- Question B: Has 6-month data EVER seen 5 consecutive losses? ---
    print(f"\n  --- B: Has 6-month data EVER seen 5+ consecutive losses? ---")

    # Find all consecutive loss streaks of any length
    all_loss_streaks = []
    current_loss = []
    for _, row in df_trades.iterrows():
        if row['pnl_dollar'] < 0:
            current_loss.append(row)
        else:
            if len(current_loss) >= 1:
                all_loss_streaks.append(len(current_loss))
            current_loss = []
    if current_loss:
        all_loss_streaks.append(len(current_loss))

    max_consecutive_losses = max(all_loss_streaks) if all_loss_streaks else 0
    count_5plus = sum(1 for s in all_loss_streaks if s >= 5)

    print(f"    Longest consecutive loss streak: {max_consecutive_losses}")
    print(f"    Number of 5+ loss streaks: {count_5plus}")

    if max_consecutive_losses >= 5:
        print(f"    --> YES, 5+ consecutive losses has occurred before.")
    else:
        print(f"    --> NO, 5+ consecutive losses is UNPRECEDENTED in 6-month history.")

    # Distribution of consecutive loss streak lengths
    print(f"\n    Distribution of consecutive loss streak lengths:")
    from collections import Counter
    streak_counts = Counter(all_loss_streaks)
    for length in sorted(streak_counts.keys()):
        count = streak_counts[length]
        print(f"      {length}-loss streak: {count} occurrences")

    # --- Question C: SL-specific consecutive streaks ---
    print(f"\n  --- C: Consecutive SL stop streaks ---")

    all_sl_streaks_lens = []
    current_sl = 0
    for _, row in df_trades.iterrows():
        if row['exit_reason'] == 'SL':
            current_sl += 1
        else:
            if current_sl >= 1:
                all_sl_streaks_lens.append(current_sl)
            current_sl = 0
    if current_sl >= 1:
        all_sl_streaks_lens.append(current_sl)

    max_consec_sl = max(all_sl_streaks_lens) if all_sl_streaks_lens else 0
    sl_streak_counts = Counter(all_sl_streaks_lens)

    print(f"    Longest consecutive SL streak: {max_consec_sl}")
    print(f"    Distribution:")
    for length in sorted(sl_streak_counts.keys()):
        count = sl_streak_counts[length]
        print(f"      {length}-SL streak: {count} occurrences")

    # =========================================================================
    # 8. SUMMARY
    # =========================================================================
    print(f"\n{'=' * 120}")
    print(f"  [8] SUMMARY: Is the Feb 12-17 regime unprecedented in 6-month history?")
    print(f"{'=' * 120}")

    # Gather key stats for summary
    total_sl = (df_trades['exit_reason'] == 'SL').sum()
    total_trades = len(df_trades)
    sl_pct = total_sl / total_trades * 100 if total_trades > 0 else 0

    print(f"""
  6-MONTH BASELINE (Aug 2025 - Feb 2026):
    Total trades:               {total_trades}
    Total SL stops:             {total_sl} ({sl_pct:.1f}% of all trades)
    Max SL stops in 1 day:      {max(daily_sl_series) if len(daily_sl_series) > 0 else 0}
    Max SL stops in 5-day:      {max_sl_in_5day}
    Longest consec. SL streak:  {max_consec_sl}
    Longest consec. loss streak:{max_consecutive_losses}

  TRADINGVIEW FEB 12-17 (5 trading days):
    Trades:                     13
    SL stops:                   8
    SL rate:                    61.5%
    ~P&L:                       ~-$598

  COMPARISON:""")

    # Is 8 SL in 5 days unprecedented?
    if max_sl_in_5day >= 8:
        print(f"    8 SL stops in 5 days:  HAS occurred before (worst: {max_sl_in_5day})")
        print(f"    --> NOT unprecedented, but still a bad cluster.")
    else:
        print(f"    8 SL stops in 5 days:  UNPRECEDENTED (worst historical: {max_sl_in_5day})")
        print(f"    --> This is the worst SL clustering in the entire 6-month dataset.")

    # Is 5+ consecutive losses unprecedented?
    if max_consecutive_losses >= 5:
        print(f"    5+ consecutive losses: HAS occurred before (longest: {max_consecutive_losses})")
    else:
        print(f"    5+ consecutive losses: UNPRECEDENTED (longest historical: {max_consecutive_losses})")

    # Is the SL rate (61.5%) unprecedented for a 5-day window?
    # Check what the highest SL-rate 5-day window is
    max_sl_rate_5d = 0
    max_sl_rate_window = None
    for j in range(len(dates_list) - 4):
        window_dates = dates_list[j:j + 5]
        w_sl = sum(daily_sl_series.get(d, 0) for d in window_dates)
        w_total = sum(daily_trade_count.get(d, 0) for d in window_dates)
        if w_total > 0:
            rate = w_sl / w_total * 100
            if rate > max_sl_rate_5d:
                max_sl_rate_5d = rate
                max_sl_rate_window = (window_dates[0], window_dates[-1], w_sl, w_total)

    if max_sl_rate_window:
        print(f"    Highest SL rate in 5-day window: {max_sl_rate_5d:.1f}% "
              f"({max_sl_rate_window[2]} SL / {max_sl_rate_window[3]} trades, "
              f"{max_sl_rate_window[0]} to {max_sl_rate_window[1]})")
        if max_sl_rate_5d >= 61.5:
            print(f"    --> 61.5% SL rate has occurred before.")
        else:
            print(f"    --> 61.5% SL rate in 5 days is UNPRECEDENTED (worst: {max_sl_rate_5d:.1f}%).")

    print(f"""
  VERDICT:
    The Feb 12-17 TradingView regime with 8 SL stops in 5 days represents a
    statistically extreme event relative to the 6-month backtest history.
    The Python backtest data (which uses Databento data feed rather than TV's
    data feed) shows the worst historical 5-day SL cluster was {max_sl_in_5day} stops.

    Note: Python and TV have known SM divergences near zero-crossings, so some
    of TV's 8 SL stops may appear as SM_FLIP exits in Python. The PATTERN of
    concentrated losses is what matters for risk management.

    Key risk insight: whether or not 8 SL stops in 5 days has occurred in Python,
    the WORST 5-day P&L window (shown in section 4) quantifies the maximum pain
    the strategy has historically produced, and should be the drawdown tolerance
    threshold for live trading.""")


if __name__ == "__main__":
    main()
