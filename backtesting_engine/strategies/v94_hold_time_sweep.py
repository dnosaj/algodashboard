"""
MES v9.4 Hold Time Sweep — Find the optimal max hold cutoff.

Also answers: what does the P&L look like AT bar N for trades that
last longer than N bars? Are they underwater or profitable when we'd cut them?
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, resample_to_5min, map_5min_rsi_to_1min,
    score_trades, compute_et_minutes, NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)

COMMISSION_PER_SIDE = 1.25
DOLLAR_PER_PT = 5.0
COMM_PTS = (COMMISSION_PER_SIDE * 2) / DOLLAR_PER_PT
SM_INDEX, SM_FLOW, SM_NORM, SM_EMA = 20, 12, 400, 255
RSI_LEN = 10
RSI_BUY, RSI_SELL = 55, 45
COOLDOWN = 15


def run_backtest_record_curve(
    opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
    max_hold_bars=0,
):
    """Run baseline backtest but record the unrealized P&L at every bar during
    each trade, so we can analyze what happens at bar N."""
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    trade_curve = []  # unrealized P&L curve for current trade

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        mfe = max(trade_curve) if trade_curve else 0
        mae = min(trade_curve) if trade_curve else 0
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
            "curve": list(trade_curve),  # snapshot the P&L curve
            "mfe": mfe, "mae": mae,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]
        sm_bull = sm_prev > 0
        sm_bear = sm_prev < 0
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        rsi_p = rsi_curr[i - 1]
        rsi_p2 = rsi_prev[i - 1]
        rsi_long_trigger = rsi_p > RSI_BUY and rsi_p2 <= RSI_BUY
        rsi_short_trigger = rsi_p < RSI_SELL and rsi_p2 >= RSI_SELL

        if sm_flipped_bull or sm_prev <= 0:
            long_used = False
        if sm_flipped_bear or sm_prev >= 0:
            short_used = False

        # Track P&L curve
        if trade_state == 1:
            trade_curve.append(closes[i - 1] - entry_price)
        elif trade_state == -1:
            trade_curve.append(entry_price - closes[i - 1])

        # EOD
        if trade_state != 0 and bar_mins_et >= NY_CLOSE_ET:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            trade_curve = []
            continue

        if trade_state == 1:
            bars_held = i - entry_idx
            if max_hold_bars > 0 and bars_held >= max_hold_bars:
                close_trade("long", entry_price, opens[i], entry_idx, i, "MAX_HOLD")
                trade_state = 0
                exit_bar = i
                trade_curve = []
                continue
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                trade_curve = []

        elif trade_state == -1:
            bars_held = i - entry_idx
            if max_hold_bars > 0 and bars_held >= max_hold_bars:
                close_trade("short", entry_price, opens[i], entry_idx, i, "MAX_HOLD")
                trade_state = 0
                exit_bar = i
                trade_curve = []
                continue
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                trade_curve = []

        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= COOLDOWN
            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    trade_curve = []
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    trade_curve = []

    return trades


def load_mes_databento():
    filepath = (Path(__file__).resolve().parent.parent
                / "data" / "databento_MES_1min_2025-08-17_to_2026-02-13.csv")
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


def trades_in_month(trades, year, month):
    return [t for t in trades
            if pd.Timestamp(t['entry_time']).year == year
            and pd.Timestamp(t['entry_time']).month == month]


def main():
    print("=" * 115)
    print("MES v9.4 HOLD TIME ANALYSIS")
    print("=" * 115)

    df_1m = load_mes_databento()
    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    volumes = df_1m['Volume'].values
    times = df_1m.index.values

    sm = compute_smart_money(closes, volumes, SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)
    df_r = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_r['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_r)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=RSI_LEN)

    # ==================================================================
    # Baseline with P&L curves
    # ==================================================================
    print("\nRunning baseline (no hold cap)...")
    baseline = run_backtest_record_curve(
        opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times)

    for t in baseline:
        t['pnl'] = (t['pts'] - COMM_PTS) * DOLLAR_PER_PT

    # ==================================================================
    # SECTION 1: What is the P&L at bar N for trades that last > N bars?
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 1: UNREALIZED P&L AT BAR N (for trades lasting > N bars)")
    print("  This shows what you'd be crystallizing by force-exiting at bar N")
    print("=" * 115)

    checkpoints = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100]

    print(f"\n  {'Bar N':>6} | {'Trades':>6} | "
          f"{'Avg unreal':>10} {'Med unreal':>10} | "
          f"{'# Profit':>8} {'# Losing':>8} | "
          f"{'Avg if P':>8} {'Avg if L':>8} | "
          f"{'Final Avg':>9}")
    print(f"  {'-' * 100}")

    for n_bar in checkpoints:
        # Get trades that lasted at least n_bar bars
        long_enough = [t for t in baseline if len(t['curve']) >= n_bar]
        if not long_enough:
            continue

        unreals = [t['curve'][n_bar - 1] for t in long_enough]
        finals = [t['pts'] for t in long_enough]

        profitable = [u for u in unreals if u > 0]
        losing = [u for u in unreals if u <= 0]

        avg_u = np.mean(unreals)
        med_u = np.median(unreals)
        avg_p = np.mean(profitable) if profitable else 0
        avg_l = np.mean(losing) if losing else 0
        avg_final = np.mean(finals)

        print(f"  {n_bar:>6} | {len(long_enough):>6} | "
              f"{avg_u:>+9.2f}p {med_u:>+9.2f}p | "
              f"{len(profitable):>8} {len(losing):>8} | "
              f"{avg_p:>+7.2f}p {avg_l:>+7.2f}p | "
              f"{avg_final:>+8.2f}p")

    # ==================================================================
    # SECTION 2: For trades > 30 bars, compare P&L at bar 30 vs final
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 2: TRADES > 30 BARS — P&L at bar 30 vs FINAL P&L")
    print("  Shows whether holding past bar 30 helps or hurts each trade")
    print("=" * 115)

    long_trades = [t for t in baseline if len(t['curve']) >= 30]
    long_trades.sort(key=lambda t: t['curve'][29])  # sort by unrealized at bar 30

    print(f"\n  {'#':>3}  {'Side':<5}  {'@Bar30':>8}  {'Final':>8}  "
          f"{'Held→':>7}  {'Bars':>5}  {'MFE':>6}  {'Exit':>9}  {'Date':<12}")
    print(f"  {'-' * 85}")

    better_count = 0
    worse_count = 0
    held_gain_total = 0
    held_loss_total = 0

    for idx, t in enumerate(long_trades):
        at_30 = t['curve'][29]
        final = t['pts']
        delta = final - at_30
        held_gain = delta if delta > 0 else 0
        held_loss = delta if delta < 0 else 0
        held_gain_total += held_gain
        held_loss_total += held_loss
        if delta > COMM_PTS:
            better_count += 1
        elif delta < -COMM_PTS:
            worse_count += 1

        marker = "HELP" if delta > 1 else ("HURT" if delta < -1 else "~")
        et = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
        if idx < 60 or abs(delta) > 5:  # show first 60 or any big delta
            print(f"  {idx+1:>3}  {t['side']:<5}  {at_30:>+7.2f}p  {final:>+7.2f}p  "
                  f"{delta:>+6.2f}p  {t['bars']:>5}  {t['mfe']:>5.1f}p  "
                  f"{t['result']:>9}  {et}  {marker}")

    print(f"\n  Summary of {len(long_trades)} trades lasting > 30 bars:")
    print(f"    Holding past bar 30 HELPED: {better_count} trades (gained {held_gain_total:+.2f}pts total)")
    print(f"    Holding past bar 30 HURT:   {worse_count} trades (lost {held_loss_total:+.2f}pts total)")
    print(f"    Net effect of holding:      {held_gain_total + held_loss_total:+.2f}pts = "
          f"${(held_gain_total + held_loss_total) * DOLLAR_PER_PT:+.2f}")

    # ==================================================================
    # SECTION 3: Max hold sweep — every value from 15 to 80
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 3: MAX HOLD SWEEP (exit at bar N regardless)")
    print("=" * 115)

    bl_sc = score_trades(baseline, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
    bl_net = bl_sc['net_dollar']

    print(f"\n  Baseline: {bl_sc['count']}t, PF {bl_sc['pf']}, ${bl_net:+.2f}\n")
    print(f"  {'MaxHold':>7} | {'Trades':>6} | {'WR%':>5} | {'PF':>6} | "
          f"{'Net$':>9} | {'d$':>8} | {'dPF':>6} | {'MaxDD$':>8} | {'Sharpe':>6}")
    print(f"  {'-' * 85}")

    best_net = bl_net
    best_hold = 0

    for hold in range(15, 81, 5):
        trades = run_backtest_record_curve(
            opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
            max_hold_bars=hold)
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        if sc is None:
            continue
        d_net = sc['net_dollar'] - bl_net
        d_pf = sc['pf'] - bl_sc['pf']
        marker = " ***" if sc['net_dollar'] > best_net else ""
        if sc['net_dollar'] > best_net:
            best_net = sc['net_dollar']
            best_hold = hold
        print(f"  {hold:>7} | {sc['count']:>6} | {sc['win_rate']:>4.1f}% | "
              f"{sc['pf']:>6.3f} | ${sc['net_dollar']:>+8.2f} | "
              f"${d_net:>+7.2f} | {d_pf:>+5.3f} | ${sc['max_dd_dollar']:>7.2f} | "
              f"{sc['sharpe']:>6.3f}{marker}")

    print(f"\n  Best max hold: {best_hold} bars (${best_net:+.2f})")

    # ==================================================================
    # SECTION 4: Monthly view for best max hold + baseline
    # ==================================================================
    print("\n" + "=" * 115)
    print(f"SECTION 4: MONTHLY — BASELINE vs MAX HOLD {best_hold}")
    print("=" * 115)

    best_trades = run_backtest_record_curve(
        opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
        max_hold_bars=best_hold)

    months = sorted(set((pd.Timestamp(t['entry_time']).year,
                         pd.Timestamp(t['entry_time']).month) for t in baseline))

    print(f"\n  {'Month':>8} | {'BL Trd':>6} {'BL Net$':>9} {'BL PF':>7} | "
          f"{'MH Trd':>6} {'MH Net$':>9} {'MH PF':>7} | {'Delta$':>8}")
    print(f"  {'-' * 75}")

    for y, m in months:
        bl_mt = trades_in_month(baseline, y, m)
        mh_mt = trades_in_month(best_trades, y, m)
        bl_msc = score_trades(bl_mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        mh_msc = score_trades(mh_mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        bl_n = bl_msc['net_dollar'] if bl_msc else 0
        mh_n = mh_msc['net_dollar'] if mh_msc else 0
        bl_pf = bl_msc['pf'] if bl_msc else 0
        mh_pf = mh_msc['pf'] if mh_msc else 0
        bl_c = bl_msc['count'] if bl_msc else 0
        mh_c = mh_msc['count'] if mh_msc else 0
        delta = mh_n - bl_n
        marker = "+" if delta > 5 else ("-" if delta < -5 else " ")
        print(f"  {y}-{m:02d} | {bl_c:>6} ${bl_n:>+8.2f} {bl_pf:>7.3f} | "
              f"{mh_c:>6} ${mh_n:>+8.2f} {mh_pf:>7.3f} | ${delta:>+7.2f} {marker}")

    # ==================================================================
    # SECTION 5: What if we cap ONLY losing trades at N bars?
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 5: CONDITIONAL HOLD CAP — only exit losers at bar N, let winners run")
    print("  (exit if bars >= N AND unrealized < 0)")
    print("=" * 115)

    # We can simulate this from the baseline curves
    print(f"\n  {'Cap':>5} | {'WouldExit':>9} | {'AvgUnreal':>9} | {'AvgFinal':>9} | "
          f"{'Saved$':>8} | {'Lost$':>8} | {'Net$':>8}")
    print(f"  {'-' * 72}")

    for cap in [20, 25, 30, 35, 40, 50]:
        # Find trades that would be force-exited (lasted > cap AND losing at bar cap)
        would_exit = []
        for t in baseline:
            if len(t['curve']) >= cap and t['curve'][cap - 1] < 0:
                would_exit.append(t)

        if not would_exit:
            continue

        # P&L at cap vs final P&L
        unreals = [t['curve'][cap - 1] for t in would_exit]
        finals = [t['pts'] for t in would_exit]

        saved = sum((f - u) for u, f in zip(unreals, finals) if f < u)  # trades that got worse
        lost = sum((f - u) for u, f in zip(unreals, finals) if f > u)   # trades that recovered

        print(f"  {cap:>5} | {len(would_exit):>9} | "
              f"{np.mean(unreals):>+8.2f}p | {np.mean(finals):>+8.2f}p | "
              f"${saved * DOLLAR_PER_PT:>+7.2f} | ${lost * DOLLAR_PER_PT:>+7.2f} | "
              f"${(saved + lost) * DOLLAR_PER_PT:>+7.2f}")

    print("\n" + "=" * 115)
    print("DONE")
    print("=" * 115)


if __name__ == "__main__":
    main()
