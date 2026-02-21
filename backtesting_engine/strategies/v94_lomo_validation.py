"""
MES v9.4 LOMO Validation — Conditional Hold Cap + EOD 15:30
============================================================
Validates the promising exit parameters with Leave-One-Month-Out (LOMO):

Configs tested:
  - BASELINE:  SM flip only (no hold cap, EOD 16:00)
  - LC25:      Exit losers at bar 25 (exit if bars>=25 AND unrealized < 0)
  - LC30:      Exit losers at bar 30
  - LC35:      Exit losers at bar 35
  - MH40:      Max hold 40 bars (blanket cap, exit all trades at bar 40)
  - EOD1530:   EOD close at 15:30 ET instead of 16:00
  - LC25+EOD:  Losing cap 25 + EOD 15:30
  - LC30+EOD:  Losing cap 30 + EOD 15:30
  - MH40+EOD:  Max hold 40 + EOD 15:30

LOMO method: Hold out 1 month at a time. Compare holdout PF/net to baseline
holdout PF/net. Wins = months where config beats baseline. A robust config
should win more months than it loses.

Data: Databento MES 1-min, Aug 17 2025 - Feb 13 2026 (7 months)
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

# ---------------------------------------------------------------------------
# MES v9.4 Config
# ---------------------------------------------------------------------------
COMMISSION_PER_SIDE = 1.25
DOLLAR_PER_PT = 5.0
COMM_PTS = (COMMISSION_PER_SIDE * 2) / DOLLAR_PER_PT  # 0.50 pts
SM_INDEX, SM_FLOW, SM_NORM, SM_EMA = 20, 12, 400, 255
RSI_LEN = 10
RSI_BUY, RSI_SELL = 55, 45
COOLDOWN = 15

# EOD 15:30 = 15*60 + 30 = 930 minutes from midnight ET
NY_EOD_1530 = 15 * 60 + 30


# ---------------------------------------------------------------------------
# Backtest engine with all exit modifiers
# ---------------------------------------------------------------------------
def run_backtest_exits(
    opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
    # Exit modifiers
    max_hold_bars=0,          # 0 = off. Exit ALL trades at this bar.
    losing_hold_cap=0,        # 0 = off. Exit LOSING trades at this bar.
    eod_minutes_et=NY_CLOSE_ET,  # EOD close time in ET minutes
):
    """MES v9.4 backtest engine with configurable exit modifiers.

    Exit priority: EOD > max_hold > losing_hold_cap > SM_FLIP
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

        rsi_p = rsi_curr[i - 1]
        rsi_p2 = rsi_prev[i - 1]
        rsi_long_trigger = rsi_p > RSI_BUY and rsi_p2 <= RSI_BUY
        rsi_short_trigger = rsi_p < RSI_SELL and rsi_p2 >= RSI_SELL

        if sm_flipped_bull or sm_prev <= 0:
            long_used = False
        if sm_flipped_bear or sm_prev >= 0:
            short_used = False

        # -- EOD Close --
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # -- Exits for longs --
        if trade_state == 1:
            bars_held = i - entry_idx

            # Max hold (blanket)
            if max_hold_bars > 0 and bars_held >= max_hold_bars:
                close_trade("long", entry_price, opens[i], entry_idx, i, "MAX_HOLD")
                trade_state = 0
                exit_bar = i
                continue

            # Losing hold cap (only if underwater)
            if losing_hold_cap > 0 and bars_held >= losing_hold_cap:
                unrealized = closes[i - 1] - entry_price
                if unrealized < 0:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "LOSING_CAP")
                    trade_state = 0
                    exit_bar = i
                    continue

            # SM flip exit
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        # -- Exits for shorts --
        elif trade_state == -1:
            bars_held = i - entry_idx

            # Max hold (blanket)
            if max_hold_bars > 0 and bars_held >= max_hold_bars:
                close_trade("short", entry_price, opens[i], entry_idx, i, "MAX_HOLD")
                trade_state = 0
                exit_bar = i
                continue

            # Losing hold cap (only if underwater)
            if losing_hold_cap > 0 and bars_held >= losing_hold_cap:
                unrealized = entry_price - closes[i - 1]
                if unrealized < 0:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "LOSING_CAP")
                    trade_state = 0
                    exit_bar = i
                    continue

            # SM flip exit
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        # -- Entries --
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
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True

    return trades


# ---------------------------------------------------------------------------
# Data loader
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def fmt_sc(sc):
    if sc is None:
        return "  0t  WR  0.0%  PF  0.000  $   +0.00"
    return (f"{sc['count']:>3}t  WR {sc['win_rate']:>5.1f}%  "
            f"PF {sc['pf']:>6.3f}  ${sc['net_dollar']:>+8.2f}")


# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
CONFIGS = [
    {"label": "BASELINE",    "max_hold_bars": 0,  "losing_hold_cap": 0,  "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC20",        "max_hold_bars": 0,  "losing_hold_cap": 20, "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC25",        "max_hold_bars": 0,  "losing_hold_cap": 25, "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC30",        "max_hold_bars": 0,  "losing_hold_cap": 30, "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC35",        "max_hold_bars": 0,  "losing_hold_cap": 35, "eod_minutes_et": NY_CLOSE_ET},
    {"label": "MH40",        "max_hold_bars": 40, "losing_hold_cap": 0,  "eod_minutes_et": NY_CLOSE_ET},
    {"label": "EOD1530",     "max_hold_bars": 0,  "losing_hold_cap": 0,  "eod_minutes_et": NY_EOD_1530},
    {"label": "LC25+EOD",    "max_hold_bars": 0,  "losing_hold_cap": 25, "eod_minutes_et": NY_EOD_1530},
    {"label": "LC30+EOD",    "max_hold_bars": 0,  "losing_hold_cap": 30, "eod_minutes_et": NY_EOD_1530},
    {"label": "MH40+EOD",    "max_hold_bars": 40, "losing_hold_cap": 0,  "eod_minutes_et": NY_EOD_1530},
    {"label": "LC25+MH40",   "max_hold_bars": 40, "losing_hold_cap": 25, "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC25+MH40+EOD", "max_hold_bars": 40, "losing_hold_cap": 25, "eod_minutes_et": NY_EOD_1530},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 120)
    print("MES v9.4 LOMO VALIDATION — Conditional Hold Cap + EOD 15:30")
    print(f"Baseline: SM(20/12/400/255) RSI(10/55/45) CD=15 SL=0")
    print(f"Commission: ${COMMISSION_PER_SIDE:.2f}/side, ${DOLLAR_PER_PT:.2f}/point")
    print("=" * 120)

    # Load data
    print("\nLoading MES 1-min data...")
    df_1m = load_mes_databento()
    print(f"  {len(df_1m)} bars, {df_1m.index[0].date()} to {df_1m.index[-1].date()}")

    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    volumes = df_1m['Volume'].values
    times = df_1m.index.values

    print("Computing SM(20/12/400/255)...")
    sm = compute_smart_money(closes, volumes, SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)

    df_r = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_r['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_r)

    print(f"Mapping RSI({RSI_LEN}) 5-min -> 1-min...")
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=RSI_LEN)

    # Run all configs
    print("\nRunning all configs...")
    all_results = {}
    for cfg in CONFIGS:
        label = cfg["label"]
        trades = run_backtest_exits(
            opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
            max_hold_bars=cfg["max_hold_bars"],
            losing_hold_cap=cfg["losing_hold_cap"],
            eod_minutes_et=cfg["eod_minutes_et"],
        )
        all_results[label] = trades
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        print(f"  {label:<20} {fmt_sc(sc)}")

    months_list = get_months(all_results['BASELINE'])
    month_labels = [f"{y}-{m:02d}" for y, m in months_list]

    # ==================================================================
    # SECTION 1: Full Period Overview
    # ==================================================================
    print("\n" + "=" * 120)
    print("SECTION 1: FULL PERIOD STATS")
    print("=" * 120)

    bl_sc = score_trades(all_results['BASELINE'], COMMISSION_PER_SIDE, DOLLAR_PER_PT)

    print(f"\n  {'Config':<22} | {'Trades':>6} | {'WR%':>5} | {'PF':>6} | "
          f"{'Net$':>9} | {'d$':>8} | {'dPF':>6} | {'MaxDD$':>8} | {'Sharpe':>6} | "
          f"{'Exits':>20}")
    print(f"  {'-' * 115}")

    for cfg in CONFIGS:
        label = cfg["label"]
        trades = all_results[label]
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        if sc is None:
            continue
        d_net = sc['net_dollar'] - bl_sc['net_dollar']
        d_pf = sc['pf'] - bl_sc['pf']

        exits_str = ", ".join(f"{k}:{v}" for k, v in sorted(sc['exits'].items()))

        print(f"  {label:<22} | {sc['count']:>6} | {sc['win_rate']:>4.1f}% | "
              f"{sc['pf']:>6.3f} | ${sc['net_dollar']:>+8.2f} | "
              f"${d_net:>+7.2f} | {d_pf:>+5.3f} | ${sc['max_dd_dollar']:>7.2f} | "
              f"{sc['sharpe']:>6.3f} | {exits_str}")

    # ==================================================================
    # SECTION 2: MONTHLY P&L HEATMAP
    # ==================================================================
    print("\n" + "=" * 120)
    print("SECTION 2: MONTHLY P&L ($ net)")
    print("  + = beats baseline that month, - = worse")
    print("=" * 120)

    # Baseline monthly
    bl_monthly = {}
    for y, m in months_list:
        mt = trades_in_month(all_results['BASELINE'], y, m)
        sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        bl_monthly[(y, m)] = sc['net_dollar'] if sc else 0

    header = f"  {'Config':<22} |"
    for ml in month_labels:
        header += f" {ml:>8}"
    header += f" | {'Total':>8}"
    print(header)
    print(f"  {'-' * (26 + len(months_list) * 9 + 12)}")

    for cfg in CONFIGS:
        label = cfg["label"]
        trades = all_results[label]
        row = f"  {label:<22} |"
        total = 0
        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            mnet = sc['net_dollar'] if sc else 0
            total += mnet
            bl_m = bl_monthly[(y, m)]
            delta = mnet - bl_m
            marker = "+" if delta > 2 else ("-" if delta < -2 else " ")
            row += f" ${mnet:>+6.0f}{marker}"
        row += f" | ${total:>+7.0f}"
        print(row)

    # ==================================================================
    # SECTION 3: LEAVE-ONE-MONTH-OUT (LOMO)
    # ==================================================================
    print("\n" + "=" * 120)
    print("SECTION 3: LEAVE-ONE-MONTH-OUT CROSS-VALIDATION")
    print("  Hold out 1 month, compare config's holdout PF to baseline holdout PF.")
    print("  * = config wins that month (higher PF than baseline)")
    print("=" * 120)

    bl_holdout_pfs = {}
    bl_holdout_nets = {}
    for y, m in months_list:
        mt = trades_in_month(all_results['BASELINE'], y, m)
        sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        bl_holdout_pfs[(y, m)] = sc['pf'] if sc else 0
        bl_holdout_nets[(y, m)] = sc['net_dollar'] if sc else 0

    # Print baseline holdout PFs for reference
    header = f"  {'Config':<22} |"
    for ml in month_labels:
        header += f" {ml:>8}"
    header += f" | {'PF Wins':>7} {'$ Wins':>6} {'Avg dPF':>7} {'Avg d$':>7}"
    print(header)
    print(f"  {'-' * (26 + len(months_list) * 9 + 35)}")

    # Baseline row
    row = f"  {'BASELINE (ref PF)':<22} |"
    for y, m in months_list:
        pf = bl_holdout_pfs[(y, m)]
        row += f" {pf:>8.3f}"
    row += f" |     --     --      --      --"
    print(row)

    # Config rows
    lomo_results = {}
    for cfg in CONFIGS[1:]:  # skip baseline
        label = cfg["label"]
        trades = all_results[label]
        pf_wins = 0
        net_wins = 0
        dpf_sum = 0
        dnet_sum = 0
        row = f"  {label:<22} |"

        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            pf = sc['pf'] if sc else 0
            net = sc['net_dollar'] if sc else 0
            bl_pf = bl_holdout_pfs[(y, m)]
            bl_net = bl_holdout_nets[(y, m)]
            dpf = pf - bl_pf
            dnet = net - bl_net
            dpf_sum += dpf
            dnet_sum += dnet

            if dpf > 0.005:
                pf_wins += 1
                row += f" {pf:>7.3f}*"
            elif dpf < -0.005:
                row += f" {pf:>7.3f} "
            else:
                row += f" {pf:>7.3f}="

            if dnet > 1:
                net_wins += 1

        avg_dpf = dpf_sum / len(months_list)
        avg_dnet = dnet_sum / len(months_list)
        row += f" | {pf_wins:>7} {net_wins:>6} {avg_dpf:>+6.3f} ${avg_dnet:>+6.0f}"
        print(row)
        lomo_results[label] = {
            "pf_wins": pf_wins, "net_wins": net_wins,
            "avg_dpf": avg_dpf, "avg_dnet": avg_dnet,
        }

    print("\n  * = beats baseline PF that month")

    # ==================================================================
    # SECTION 4: LOMO NET $ DELTAS (per-month improvement in $)
    # ==================================================================
    print("\n" + "=" * 120)
    print("SECTION 4: PER-MONTH $ DELTA vs BASELINE")
    print("  How much more/less each config makes in each holdout month")
    print("=" * 120)

    header = f"  {'Config':<22} |"
    for ml in month_labels:
        header += f" {ml:>8}"
    header += f" | {'Sum d$':>8}"
    print(header)
    print(f"  {'-' * (26 + len(months_list) * 9 + 12)}")

    for cfg in CONFIGS[1:]:
        label = cfg["label"]
        trades = all_results[label]
        row = f"  {label:<22} |"
        total_delta = 0
        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            net = sc['net_dollar'] if sc else 0
            bl_net = bl_holdout_nets[(y, m)]
            delta = net - bl_net
            total_delta += delta
            marker = "+" if delta > 2 else ("-" if delta < -2 else " ")
            row += f" ${delta:>+6.0f}{marker}"
        row += f" | ${total_delta:>+7.0f}"
        print(row)

    # ==================================================================
    # SECTION 5: WALK-FORWARD VALIDATION
    # ==================================================================
    print("\n" + "=" * 120)
    print("SECTION 5: WALK-FORWARD VALIDATION")
    print("  Train months 1-3 -> test 4, train 1-4 -> test 5, etc.")
    print("  Shows if config would have been selected and profitable in OOS")
    print("=" * 120)

    wf_test_months = months_list[3:]  # month 4 onward

    header = f"  {'Config':<22} |"
    for y, m in wf_test_months:
        header += f" {y}-{m:02d}:>8"
    header += f" | {'OOS Wins':>8} {'OOS d$':>8}"
    print(header)
    print(f"  {'-' * (26 + len(wf_test_months) * 9 + 20)}")

    for cfg in CONFIGS[1:]:
        label = cfg["label"]
        trades = all_results[label]
        bl_trades = all_results['BASELINE']
        row = f"  {label:<22} |"
        oos_wins = 0
        oos_dnet = 0

        for test_idx in range(3, len(months_list)):
            ty, tm = months_list[test_idx]

            # Check IS performance (training window = all months before test)
            is_better = False
            is_net = 0
            bl_is_net = 0
            for train_idx in range(test_idx):
                iy, im = months_list[train_idx]
                mt = trades_in_month(trades, iy, im)
                bl_mt = trades_in_month(bl_trades, iy, im)
                sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
                bl_sc_m = score_trades(bl_mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
                is_net += (sc['net_dollar'] if sc else 0)
                bl_is_net += (bl_sc_m['net_dollar'] if bl_sc_m else 0)
            is_better = is_net > bl_is_net

            # OOS performance
            mt = trades_in_month(trades, ty, tm)
            bl_mt = trades_in_month(bl_trades, ty, tm)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            bl_sc_m = score_trades(bl_mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            oos_net = (sc['net_dollar'] if sc else 0)
            bl_oos_net = (bl_sc_m['net_dollar'] if bl_sc_m else 0)
            dnet = oos_net - bl_oos_net
            oos_dnet += dnet

            # Only count OOS win if IS also showed improvement
            if is_better and dnet > 1:
                oos_wins += 1
                row += f" ${dnet:>+5.0f}**"
            elif dnet > 1:
                row += f" ${dnet:>+5.0f}* "
            elif dnet < -1:
                row += f" ${dnet:>+5.0f}  "
            else:
                row += f" ${dnet:>+5.0f}= "

        row += f" | {oos_wins:>8} ${oos_dnet:>+7.0f}"
        print(row)

    print("\n  ** = IS confirms AND OOS wins   * = OOS wins only   (blank) = OOS loses")

    # ==================================================================
    # SECTION 6: TRADE-LEVEL ANALYSIS OF LOSING CAP
    # ==================================================================
    print("\n" + "=" * 120)
    print("SECTION 6: LOSING CAP TRADE-LEVEL DETAIL")
    print("  For the best losing cap config, show each trade it would exit early")
    print("  vs what baseline did with the same trade.")
    print("=" * 120)

    # Use LC25 and LC30 for comparison
    for cap_label in ["LC25", "LC30"]:
        cap_trades = all_results[cap_label]
        bl_trades = all_results['BASELINE']

        # Build entry-time -> trade maps
        cap_map = {str(t['entry_time']): t for t in cap_trades}
        bl_map = {str(t['entry_time']): t for t in bl_trades}

        # Find trades where cap exits differently than baseline
        cap_exits = [t for t in cap_trades if t['result'] == 'LOSING_CAP']

        print(f"\n  --- {cap_label}: {len(cap_exits)} trades exited by losing cap ---")
        if not cap_exits:
            continue

        print(f"  {'#':>3}  {'Side':<5}  {'Cap Exit$':>9}  {'SM Exit$':>9}  "
              f"{'Cap P&L':>8}  {'SM P&L':>8}  {'Delta':>7}  "
              f"{'Bars(C)':>7}  {'Bars(SM)':>8}  {'Entry Time':<16}")
        print(f"  {'-' * 105}")

        total_saved = 0
        helped = 0
        hurt = 0

        for idx, t in enumerate(cap_exits):
            key = str(t['entry_time'])
            bl_t = bl_map.get(key)
            if bl_t is None:
                continue

            cap_pnl = (t['pts'] - COMM_PTS) * DOLLAR_PER_PT
            bl_pnl = (bl_t['pts'] - COMM_PTS) * DOLLAR_PER_PT
            delta = cap_pnl - bl_pnl
            total_saved += delta

            if delta > 0.5:
                helped += 1
                marker = "SAVED"
            elif delta < -0.5:
                hurt += 1
                marker = "HURT"
            else:
                marker = "~"

            et = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
            print(f"  {idx+1:>3}  {t['side']:<5}  {t['exit']:>9.2f}  {bl_t['exit']:>9.2f}  "
                  f"${cap_pnl:>+7.2f}  ${bl_pnl:>+7.2f}  ${delta:>+6.2f}  "
                  f"{t['bars']:>7}  {bl_t['bars']:>8}  {et}  {marker}")

        print(f"\n  {cap_label} Summary: {len(cap_exits)} trades exited by losing cap")
        print(f"    Helped (exited before worse): {helped}")
        print(f"    Hurt (missed recovery):       {hurt}")
        print(f"    Net delta: ${total_saved:>+.2f}")

    # ==================================================================
    # SECTION 7: RECOMMENDATION
    # ==================================================================
    print("\n" + "=" * 120)
    print("SECTION 7: RANKING & RECOMMENDATION")
    print("=" * 120)

    # Rank by: LOMO PF wins DESC, then avg dPF DESC, then avg d$ DESC
    ranked = sorted(lomo_results.items(),
                    key=lambda x: (x[1]['pf_wins'], x[1]['avg_dpf'], x[1]['avg_dnet']),
                    reverse=True)

    print(f"\n  {'Rank':>4}  {'Config':<22}  {'LOMO PF Wins':>12}  "
          f"{'LOMO $ Wins':>11}  {'Avg dPF':>7}  {'Avg d$/mo':>9}")
    print(f"  {'-' * 75}")

    for rank, (label, res) in enumerate(ranked, 1):
        cfg = next(c for c in CONFIGS if c['label'] == label)
        params = []
        if cfg['losing_hold_cap'] > 0:
            params.append(f"LoseCap={cfg['losing_hold_cap']}")
        if cfg['max_hold_bars'] > 0:
            params.append(f"MaxHold={cfg['max_hold_bars']}")
        if cfg['eod_minutes_et'] != NY_CLOSE_ET:
            params.append("EOD=15:30")
        param_str = " + ".join(params) if params else "(none)"

        marker = " <<<" if rank <= 3 else ""
        print(f"  {rank:>4}  {label:<22}  {res['pf_wins']:>12}/7  "
              f"{res['net_wins']:>11}/7  {res['avg_dpf']:>+6.3f}  "
              f"${res['avg_dnet']:>+8.2f}  [{param_str}]{marker}")

    print("\n" + "=" * 120)
    print("DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
