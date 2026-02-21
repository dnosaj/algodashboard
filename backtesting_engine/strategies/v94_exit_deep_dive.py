"""
MES v9.4 Exit Deep Dive
========================
1. 50+ bar trade breakdown (winners vs losers, P&L distribution)
2. September 2025 deep dive — what made it so bad? What overfits it?
3. New exit ideas beyond RSI: max hold time, SM deceleration, breakeven stop,
   tighter EOD, losing hold cap
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, score_trades,
    compute_et_minutes, NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)

COMMISSION_PER_SIDE = 1.25
DOLLAR_PER_PT = 5.0
COMM_PTS = (COMMISSION_PER_SIDE * 2) / DOLLAR_PER_PT

SM_INDEX, SM_FLOW, SM_NORM, SM_EMA = 20, 12, 400, 255
RSI_LEN = 10
RSI_BUY, RSI_SELL = 55, 45
COOLDOWN = 15
MAX_LOSS_PTS = 0


# ---------------------------------------------------------------------------
# Extended backtest — tracks MFE, MAE, SM velocity, and supports new exits
# ---------------------------------------------------------------------------
def run_backtest_extended(
    opens, highs, lows, closes, sm, rsi_5m_curr, rsi_5m_prev, times,
    rsi_buy=55, rsi_sell=45, cooldown_bars=15, max_loss_pts=0,
    # New exit options
    max_hold_bars=0,           # Force exit after N bars (0=off)
    max_hold_losing_bars=0,    # Force exit after N bars IF losing (0=off)
    breakeven_after_bars=0,    # Move stop to breakeven after N bars (0=off)
    breakeven_min_mfe=0,       # Min MFE (pts) before breakeven activates (0=any)
    sm_decel_exit=False,       # Exit when SM decelerating (positive but falling)
    sm_decel_bars=3,           # SM must decline for N consecutive bars
    early_eod_et=0,            # Earlier EOD close in ET minutes (0=use default 960)
    rsi_exit_long_below=0,     # RSI cross exit (from v2 test)
    rsi_exit_short_above=100,
    rsi_exit_profit_gate=False,
):
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    et_mins = compute_et_minutes(times)
    eod_time = early_eod_et if early_eod_et > 0 else NY_CLOSE_ET

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result,
                    mfe=0, mae=0, sm_entry=0, sm_exit=0, rsi_entry=0, rsi_exit=0):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
            "mfe": mfe, "mae": mae,
            "sm_entry": sm_entry, "sm_exit": sm_exit,
            "rsi_entry": rsi_entry, "rsi_exit": rsi_exit,
        })

    # Per-trade tracking
    trade_mfe = 0.0  # max favorable excursion (pts)
    trade_mae = 0.0  # max adverse excursion (pts)
    sm_at_entry = 0.0
    rsi_at_entry = 0.0
    sm_declining_count = 0
    prev_sm_for_decel = 0.0

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

        if sm_flipped_bull or sm_prev <= 0:
            long_used = False
        if sm_flipped_bear or sm_prev >= 0:
            short_used = False

        # Track MFE/MAE during trade
        if trade_state == 1:
            unrealized = closes[i - 1] - entry_price
            bar_mfe = highs[i - 1] - entry_price
            bar_mae = entry_price - lows[i - 1]
            if bar_mfe > trade_mfe:
                trade_mfe = bar_mfe
            if bar_mae > trade_mae:
                trade_mae = bar_mae
            # SM deceleration tracking
            if sm_prev < prev_sm_for_decel:
                sm_declining_count += 1
            else:
                sm_declining_count = 0
            prev_sm_for_decel = sm_prev
        elif trade_state == -1:
            unrealized = entry_price - closes[i - 1]
            bar_mfe = entry_price - lows[i - 1]
            bar_mae = highs[i - 1] - entry_price
            if bar_mfe > trade_mfe:
                trade_mfe = bar_mfe
            if bar_mae > trade_mae:
                trade_mae = bar_mae
            if sm_prev > prev_sm_for_decel:
                sm_declining_count += 1  # for shorts: SM rising = bad
            else:
                sm_declining_count = 0
            prev_sm_for_decel = sm_prev

        def do_close(side, result):
            nonlocal trade_state, exit_bar, trade_mfe, trade_mae
            close_trade(side, entry_price, opens[i], entry_idx, i, result,
                        mfe=trade_mfe, mae=trade_mae,
                        sm_entry=sm_at_entry, sm_exit=sm_prev,
                        rsi_entry=rsi_at_entry, rsi_exit=rsi_prev)
            trade_state = 0
            exit_bar = i
            trade_mfe = 0.0
            trade_mae = 0.0

        # -- EOD Close --
        if trade_state != 0 and bar_mins_et >= eod_time:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD",
                        mfe=trade_mfe, mae=trade_mae,
                        sm_entry=sm_at_entry, sm_exit=sm_prev,
                        rsi_entry=rsi_at_entry, rsi_exit=rsi_prev)
            trade_state = 0
            exit_bar = i
            trade_mfe = 0.0
            trade_mae = 0.0
            continue

        # -- Exits --
        if trade_state == 1:
            bars_held = i - entry_idx
            unrealized = closes[i - 1] - entry_price

            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                do_close("long", "SL")
                continue

            # Max hold time
            if max_hold_bars > 0 and bars_held >= max_hold_bars:
                do_close("long", "MAX_HOLD")
                continue

            # Max hold if losing
            if max_hold_losing_bars > 0 and bars_held >= max_hold_losing_bars and unrealized < 0:
                do_close("long", "HOLD_LOSS")
                continue

            # Breakeven stop
            if breakeven_after_bars > 0 and bars_held >= breakeven_after_bars:
                if trade_mfe >= breakeven_min_mfe and unrealized <= 0:
                    do_close("long", "BREAKEVEN")
                    continue

            # SM deceleration exit (SM still positive but declining for N bars)
            if sm_decel_exit and sm_prev > 0 and sm_declining_count >= sm_decel_bars:
                do_close("long", "SM_DECEL")
                continue

            # RSI cross exit
            if rsi_exit_long_below > 0:
                if rsi_prev < rsi_exit_long_below and rsi_prev2 >= rsi_exit_long_below:
                    if not rsi_exit_profit_gate or unrealized > 0:
                        do_close("long", "RSI_EXIT")
                        continue

            # SM flip
            if sm_prev < 0 and sm_prev2 >= 0:
                do_close("long", "SM_FLIP")

        elif trade_state == -1:
            bars_held = i - entry_idx
            unrealized = entry_price - closes[i - 1]

            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                do_close("short", "SL")
                continue

            if max_hold_bars > 0 and bars_held >= max_hold_bars:
                do_close("short", "MAX_HOLD")
                continue

            if max_hold_losing_bars > 0 and bars_held >= max_hold_losing_bars and unrealized < 0:
                do_close("short", "HOLD_LOSS")
                continue

            if breakeven_after_bars > 0 and bars_held >= breakeven_after_bars:
                if trade_mfe >= breakeven_min_mfe and unrealized <= 0:
                    do_close("short", "BREAKEVEN")
                    continue

            if sm_decel_exit and sm_prev < 0 and sm_declining_count >= sm_decel_bars:
                do_close("short", "SM_DECEL")
                continue

            if rsi_exit_short_above < 100:
                if rsi_prev > rsi_exit_short_above and rsi_prev2 <= rsi_exit_short_above:
                    if not rsi_exit_profit_gate or unrealized > 0:
                        do_close("short", "RSI_EXIT")
                        continue

            if sm_prev > 0 and sm_prev2 <= 0:
                do_close("short", "SM_FLIP")

        # -- Entries --
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
                    trade_mfe = 0.0
                    trade_mae = 0.0
                    sm_at_entry = sm_prev
                    rsi_at_entry = rsi_prev
                    sm_declining_count = 0
                    prev_sm_for_decel = sm_prev
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    trade_mfe = 0.0
                    trade_mae = 0.0
                    sm_at_entry = sm_prev
                    rsi_at_entry = rsi_prev
                    sm_declining_count = 0
                    prev_sm_for_decel = sm_prev

    return trades


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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


def get_months(trades):
    months = set()
    for t in trades:
        ts = pd.Timestamp(t['entry_time'])
        months.add((ts.year, ts.month))
    return sorted(months)


def score_and_print(trades, label, commission=COMMISSION_PER_SIDE, dpp=DOLLAR_PER_PT):
    sc = score_trades(trades, commission, dpp)
    if sc is None:
        print(f"  {label:<42} NO TRADES")
        return sc
    exits = "  ".join(f"{k}:{v}" for k, v in sorted(sc['exits'].items()))
    print(f"  {label:<42} {sc['count']:>3}t  WR {sc['win_rate']:>5.1f}%  "
          f"PF {sc['pf']:>6.3f}  ${sc['net_dollar']:>+8.2f}  "
          f"DD ${sc['max_dd_dollar']:>7.2f}  Sh {sc['sharpe']:>5.3f}  | {exits}")
    return sc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 115)
    print("MES v9.4 EXIT DEEP DIVE")
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
    # SECTION 1: Baseline with MFE/MAE data
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 1: HOLD TIME + MFE/MAE ANALYSIS")
    print("=" * 115)

    baseline = run_backtest_extended(
        opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times)

    for t in baseline:
        t['pnl'] = (t['pts'] - COMM_PTS) * DOLLAR_PER_PT

    # Bucket by hold time
    buckets = [
        ("1-10 bars", 1, 10),
        ("11-20 bars", 11, 20),
        ("21-30 bars", 21, 30),
        ("31-50 bars", 31, 50),
        ("51-75 bars", 51, 75),
        ("76-100 bars", 76, 100),
        ("100+ bars", 101, 9999),
    ]

    print(f"\n  {'Bucket':<14} | {'Total':>5} | {'Win':>4} {'Loss':>4} | "
          f"{'WR%':>5} | {'Avg W$':>8} {'Avg L$':>8} | "
          f"{'Net$':>9} | {'Avg MFE':>7} {'Avg MAE':>7} | {'Avg Bars':>8}")
    print(f"  {'-' * 105}")

    for label, lo, hi in buckets:
        bt = [t for t in baseline if lo <= t['bars'] <= hi]
        if not bt:
            print(f"  {label:<14} |     0 |")
            continue
        wins = [t for t in bt if t['pnl'] > 0]
        losses = [t for t in bt if t['pnl'] <= 0]
        net = sum(t['pnl'] for t in bt)
        avg_w = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_l = np.mean([t['pnl'] for t in losses]) if losses else 0
        wr = len(wins) / len(bt) * 100
        avg_mfe = np.mean([t['mfe'] for t in bt])
        avg_mae = np.mean([t['mae'] for t in bt])
        avg_bars = np.mean([t['bars'] for t in bt])
        print(f"  {label:<14} | {len(bt):>5} | {len(wins):>4} {len(losses):>4} | "
              f"{wr:>4.0f}% | ${avg_w:>+7.2f} ${avg_l:>+7.2f} | "
              f"${net:>+8.2f} | {avg_mfe:>6.1f}p {avg_mae:>6.1f}p | {avg_bars:>8.1f}")

    # 50+ bar trades in detail
    long_hold = sorted([t for t in baseline if t['bars'] >= 50], key=lambda t: t['pnl'])
    print(f"\n  ALL {len(long_hold)} TRADES HELD 50+ BARS (sorted by P&L):")
    print(f"  {'#':>3}  {'Side':<5}  {'P&L$':>8}  {'Pts':>7}  {'Bars':>5}  "
          f"{'MFE':>6}  {'MAE':>6}  {'Exit':>9}  {'SM_ent':>6}  {'SM_ext':>6}  "
          f"{'RSI_e':>5}  {'RSI_x':>5}  {'Entry Time':<16}")
    print(f"  {'-' * 115}")
    for idx, t in enumerate(long_hold):
        et = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
        print(f"  {idx+1:>3}  {t['side']:<5}  ${t['pnl']:>+7.2f}  {t['pts']:>+7.2f}  "
              f"{t['bars']:>5}  {t['mfe']:>5.1f}p  {t['mae']:>5.1f}p  "
              f"{t['result']:>9}  {t['sm_entry']:>6.3f}  {t['sm_exit']:>6.3f}  "
              f"{t['rsi_entry']:>5.1f}  {t['rsi_exit']:>5.1f}  {et}")

    # ==================================================================
    # SECTION 2: September 2025 Deep Dive
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 2: SEPTEMBER 2025 — WHY SO BAD?")
    print("=" * 115)

    sept = trades_in_month(baseline, 2025, 9)
    for t in sept:
        t['pnl'] = (t['pts'] - COMM_PTS) * DOLLAR_PER_PT

    sept_wins = [t for t in sept if t['pnl'] > 0]
    sept_losses = [t for t in sept if t['pnl'] <= 0]

    print(f"\n  September 2025: {len(sept)} trades")
    print(f"  Winners: {len(sept_wins)}, avg ${np.mean([t['pnl'] for t in sept_wins]):+.2f}, "
          f"avg {np.mean([t['bars'] for t in sept_wins]):.0f} bars, "
          f"avg MFE {np.mean([t['mfe'] for t in sept_wins]):.1f}p")
    print(f"  Losers:  {len(sept_losses)}, avg ${np.mean([t['pnl'] for t in sept_losses]):+.2f}, "
          f"avg {np.mean([t['bars'] for t in sept_losses]):.0f} bars, "
          f"avg MAE {np.mean([t['mae'] for t in sept_losses]):.1f}p")

    # Side breakdown
    sept_longs = [t for t in sept if t['side'] == 'long']
    sept_shorts = [t for t in sept if t['side'] == 'short']
    sl = score_trades(sept_longs, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
    ss = score_trades(sept_shorts, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
    print(f"\n  Longs:  {sl['count']}t, WR {sl['win_rate']}%, PF {sl['pf']}, ${sl['net_dollar']:+.2f}" if sl else "  Longs: 0")
    print(f"  Shorts: {ss['count']}t, WR {ss['win_rate']}%, PF {ss['pf']}, ${ss['net_dollar']:+.2f}" if ss else "  Shorts: 0")

    # Exit type breakdown for Sept
    sept_exits = {}
    for t in sept:
        r = t['result']
        sept_exits.setdefault(r, []).append(t)
    print(f"\n  Exit type breakdown:")
    for etype, etrades in sorted(sept_exits.items()):
        esc = score_trades(etrades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        print(f"    {etype:<10}: {len(etrades):>3}t, "
              f"WR {esc['win_rate']:>4.0f}%, Net ${esc['net_dollar']:>+7.2f}, "
              f"avg {np.mean([t['bars'] for t in etrades]):.0f} bars")

    # All Sept trades detail
    print(f"\n  ALL SEPTEMBER TRADES:")
    print(f"  {'#':>3}  {'Side':<5}  {'P&L$':>8}  {'Bars':>5}  {'MFE':>6}  {'MAE':>6}  "
          f"{'Exit':>9}  {'SM_e':>6}  {'RSI_e':>5}  {'Entry':>10}  {'Date':<12}")
    print(f"  {'-' * 100}")
    for idx, t in enumerate(sorted(sept, key=lambda x: x['entry_time'])):
        et = pd.Timestamp(t['entry_time'])
        print(f"  {idx+1:>3}  {t['side']:<5}  ${t['pnl']:>+7.2f}  {t['bars']:>5}  "
              f"{t['mfe']:>5.1f}p  {t['mae']:>5.1f}p  {t['result']:>9}  "
              f"{t['sm_entry']:>6.3f}  {t['rsi_entry']:>5.1f}  "
              f"{t['entry']:>10.2f}  {et.strftime('%m-%d %H:%M')}")

    # What settings would overfit Sept?
    print(f"\n  OVERFIT ANALYSIS: What would fix September?")

    # Test various aggressive exits on Sept data only
    overfit_configs = [
        {"label": "Max hold 30 bars", "max_hold_bars": 30},
        {"label": "Max hold 40 bars", "max_hold_bars": 40},
        {"label": "Max hold 50 bars", "max_hold_bars": 50},
        {"label": "Losing hold cap 20", "max_hold_losing_bars": 20},
        {"label": "Losing hold cap 30", "max_hold_losing_bars": 30},
        {"label": "Losing hold cap 40", "max_hold_losing_bars": 40},
        {"label": "Breakeven after 20 bars", "breakeven_after_bars": 20},
        {"label": "Breakeven after 30 bars", "breakeven_after_bars": 30},
        {"label": "Breakeven 30b, MFE>3", "breakeven_after_bars": 30, "breakeven_min_mfe": 3},
        {"label": "SM deceleration 3 bars", "sm_decel_exit": True, "sm_decel_bars": 3},
        {"label": "SM deceleration 5 bars", "sm_decel_exit": True, "sm_decel_bars": 5},
        {"label": "SM deceleration 8 bars", "sm_decel_exit": True, "sm_decel_bars": 8},
        {"label": "EOD 15:30 (930 ET mins)", "early_eod_et": 930},
        {"label": "EOD 15:00 (900 ET mins)", "early_eod_et": 900},
        {"label": "RSI cross<45, PG", "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
         "rsi_exit_profit_gate": True},
        {"label": "Losing cap 30 + RSI<45 PG",
         "max_hold_losing_bars": 30, "rsi_exit_long_below": 45,
         "rsi_exit_short_above": 55, "rsi_exit_profit_gate": True},
        {"label": "Losing cap 40 + BE 30b",
         "max_hold_losing_bars": 40, "breakeven_after_bars": 30},
        {"label": "SM decel 5 + losing cap 40",
         "sm_decel_exit": True, "sm_decel_bars": 5, "max_hold_losing_bars": 40},
    ]

    # ==================================================================
    # SECTION 3: New Exit Ideas — Full Period + Monthly
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 3: NEW EXIT IDEAS — FULL PERIOD")
    print("=" * 115)

    months_list = get_months(baseline)
    month_labels = [f"{y}-{m:02d}" for y, m in months_list]

    all_results = {"BASELINE": baseline}
    print(f"\n  {'Config':<35} | {'Full':>10} |", end="")
    for ml in month_labels:
        print(f" {ml:>7}", end="")
    print(f" | {'Exits':>5}")
    print(f"  {'-' * (40 + 11 + len(months_list) * 8 + 10)}")

    # Baseline row
    bl_sc = score_trades(baseline, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
    row = f"  {'BASELINE (SM flip)':<35} | ${bl_sc['net_dollar']:>+8.2f} |"
    bl_monthly = {}
    for y, m in months_list:
        mt = trades_in_month(baseline, y, m)
        sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        mnet = sc['net_dollar'] if sc else 0
        bl_monthly[(y, m)] = mnet
        row += f" {mnet:>+6.0f} "
    row += f" | {bl_sc['count']:>5}"
    print(row)

    for cfg in overfit_configs:
        label = cfg.pop("label")
        trades = run_backtest_extended(
            opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times, **cfg)
        cfg["label"] = label  # restore
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        if sc is None:
            continue
        all_results[label] = trades

        row = f"  {label:<35} | ${sc['net_dollar']:>+8.2f} |"
        months_better = 0
        months_worse = 0
        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            msc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            mnet = msc['net_dollar'] if msc else 0
            delta = mnet - bl_monthly[(y, m)]
            if delta > 5:
                marker = "+"
                months_better += 1
            elif delta < -5:
                marker = "-"
                months_worse += 1
            else:
                marker = " "
            row += f" {mnet:>+6.0f}{marker}"
        row += f" | {sc['count']:>5}"
        print(row)

    # ==================================================================
    # SECTION 4: LOMO for promising configs
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 4: LEAVE-ONE-MONTH-OUT for configs that improved full period")
    print("=" * 115)

    # Find configs that beat baseline on full period
    promising = []
    for label, trades in all_results.items():
        if label == "BASELINE":
            continue
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        if sc and sc['net_dollar'] > bl_sc['net_dollar'] + 10:
            promising.append((label, trades, sc))

    if not promising:
        # Show top 5 by net even if they don't beat baseline
        print("  (No config beat baseline. Showing closest.)")
        for label, trades in all_results.items():
            if label == "BASELINE":
                continue
            sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            if sc:
                promising.append((label, trades, sc))
        promising.sort(key=lambda x: x[2]['net_dollar'], reverse=True)
        promising = promising[:5]

    header = f"  {'Config':<35} |"
    for ml in month_labels:
        header += f" {ml:>7}"
    header += f" | {'Wins':>4} {'dPF_avg':>7}"
    print(header)
    print(f"  {'-' * (40 + len(months_list) * 8 + 16)}")

    # Baseline LOMO
    bl_pfs = {}
    row = f"  {'BASELINE':<35} |"
    for y, m in months_list:
        mt = trades_in_month(baseline, y, m)
        sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        pf = sc['pf'] if sc else 0
        bl_pfs[(y, m)] = pf
        row += f" {pf:>7.3f}"
    row += f" |  ref"
    print(row)

    for label, trades, full_sc in promising:
        row = f"  {label:<35} |"
        wins = 0
        dpf_sum = 0
        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            pf = sc['pf'] if sc else 0
            dpf = pf - bl_pfs[(y, m)]
            dpf_sum += dpf
            if dpf > 0.005:
                wins += 1
                row += f" {pf:>6.3f}*"
            elif dpf < -0.005:
                row += f" {pf:>6.3f} "
            else:
                row += f" {pf:>6.3f}="
        avg_dpf = dpf_sum / len(months_list)
        row += f" | {wins:>4} {avg_dpf:>+7.3f}"
        print(row)

    # ==================================================================
    # SECTION 5: MFE Analysis — Are we leaving money on the table?
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 5: MFE vs ACTUAL P&L — How much are we leaving on the table?")
    print("=" * 115)

    print(f"\n  {'Bucket':<14} | {'Trades':>6} | {'Avg MFE':>8} {'Avg Pts':>8} | "
          f"{'MFE-Pts':>8} | {'%Captured':>9} | {'Avg MAE':>8}")
    print(f"  {'-' * 80}")

    for label, lo, hi in buckets:
        bt = [t for t in baseline if lo <= t['bars'] <= hi]
        if not bt:
            continue
        avg_mfe = np.mean([t['mfe'] for t in bt])
        avg_pts = np.mean([t['pts'] for t in bt])
        avg_mae = np.mean([t['mae'] for t in bt])
        left = avg_mfe - avg_pts
        captured = (avg_pts / avg_mfe * 100) if avg_mfe > 0 else 0
        print(f"  {label:<14} | {len(bt):>6} | {avg_mfe:>7.1f}p {avg_pts:>+7.1f}p | "
              f"{left:>+7.1f}p | {captured:>8.0f}% | {avg_mae:>7.1f}p")

    # Overall
    avg_mfe_all = np.mean([t['mfe'] for t in baseline])
    avg_pts_all = np.mean([t['pts'] for t in baseline])
    avg_mae_all = np.mean([t['mae'] for t in baseline])
    print(f"\n  Overall: avg MFE {avg_mfe_all:.1f}p, avg exit {avg_pts_all:+.1f}p, "
          f"capturing {avg_pts_all/avg_mfe_all*100:.0f}% of MFE, avg MAE {avg_mae_all:.1f}p")

    # Trades where MFE was great but ended as losers
    gave_back = sorted(
        [t for t in baseline if t['mfe'] >= 5 and t['pnl'] <= 0],
        key=lambda t: t['pnl'])
    print(f"\n  TRADES WITH MFE >= 5pts THAT ENDED AS LOSERS ({len(gave_back)} total):")
    print(f"  {'#':>3}  {'Side':<5}  {'P&L$':>8}  {'MFE':>6}  {'MAE':>6}  {'Bars':>5}  "
          f"{'Exit':>9}  {'Date':<12}")
    print(f"  {'-' * 75}")
    for idx, t in enumerate(gave_back[:20]):
        et = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
        print(f"  {idx+1:>3}  {t['side']:<5}  ${t['pnl']:>+7.2f}  {t['mfe']:>5.1f}p  "
              f"{t['mae']:>5.1f}p  {t['bars']:>5}  {t['result']:>9}  {et}")

    print("\n" + "=" * 115)
    print("DONE")
    print("=" * 115)


if __name__ == "__main__":
    main()
