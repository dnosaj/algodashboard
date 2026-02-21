"""
MES v9.4 Losing Cap + Directional Suppression
===============================================
After cutting a loser at bar N, suppress re-entries in the SAME direction
until SM flips. Opposite-direction entries are still allowed.

This addresses the feedback loop where cutting a loser and re-entering
the same direction in the same choppy SM regime just creates another loser.

Configs tested:
  BASELINE:       SM flip exit only
  LC25:           Losing cap 25, normal re-entry (for reference)
  LC30:           Losing cap 30, normal re-entry (for reference)
  LC25+SUPP:      Losing cap 25 + suppress same-direction re-entry
  LC30+SUPP:      Losing cap 30 + suppress same-direction re-entry
  LC25+SUPP+EOD:  LC25 + suppress + EOD 15:30
  LC30+SUPP+EOD:  LC30 + suppress + EOD 15:30
  EOD1530:        EOD 15:30 only (validated reference)

Validation: LOMO (7 months), walk-forward, trade-level detail
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
NY_EOD_1530 = 15 * 60 + 30


def run_backtest_losecap(
    opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
    losing_hold_cap=0,
    suppress_same_dir=False,   # After losing cap exit, block same-direction entries until SM flips
    eod_minutes_et=NY_CLOSE_ET,
):
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    # Directional suppression state
    suppress_long = False   # Block long entries until SM flips to bear then back to bull
    suppress_short = False  # Block short entries until SM flips to bull then back to bear

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

        # Clear directional suppression on SM flip
        # suppress_long clears when SM flips to bear (sm_flipped_bear)
        # — the idea is "SM changed its mind, the old regime is over"
        if suppress_long and sm_flipped_bear:
            suppress_long = False
        if suppress_short and sm_flipped_bull:
            suppress_short = False

        # EOD
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        if trade_state == 1:
            bars_held = i - entry_idx
            if losing_hold_cap > 0 and bars_held >= losing_hold_cap:
                unrealized = closes[i - 1] - entry_price
                if unrealized < 0:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "LOSING_CAP")
                    trade_state = 0
                    exit_bar = i
                    if suppress_same_dir:
                        suppress_long = True
                    continue
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        elif trade_state == -1:
            bars_held = i - entry_idx
            if losing_hold_cap > 0 and bars_held >= losing_hold_cap:
                unrealized = entry_price - closes[i - 1]
                if unrealized < 0:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "LOSING_CAP")
                    trade_state = 0
                    exit_bar = i
                    if suppress_same_dir:
                        suppress_short = True
                    continue
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= COOLDOWN
            if in_session and cd_ok:
                long_ok = sm_bull and rsi_long_trigger and not long_used
                short_ok = sm_bear and rsi_short_trigger and not short_used

                # Apply directional suppression
                if suppress_same_dir:
                    if suppress_long and long_ok:
                        long_ok = False
                    if suppress_short and short_ok:
                        short_ok = False

                if long_ok:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                elif short_ok:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True

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


CONFIGS = [
    {"label": "BASELINE",        "losing_hold_cap": 0,  "suppress_same_dir": False, "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC25",            "losing_hold_cap": 25, "suppress_same_dir": False, "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC30",            "losing_hold_cap": 30, "suppress_same_dir": False, "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC25+SUPP",       "losing_hold_cap": 25, "suppress_same_dir": True,  "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC30+SUPP",       "losing_hold_cap": 30, "suppress_same_dir": True,  "eod_minutes_et": NY_CLOSE_ET},
    {"label": "LC35+SUPP",       "losing_hold_cap": 35, "suppress_same_dir": True,  "eod_minutes_et": NY_CLOSE_ET},
    {"label": "EOD1530",         "losing_hold_cap": 0,  "suppress_same_dir": False, "eod_minutes_et": NY_EOD_1530},
    {"label": "LC25+SUPP+EOD",   "losing_hold_cap": 25, "suppress_same_dir": True,  "eod_minutes_et": NY_EOD_1530},
    {"label": "LC30+SUPP+EOD",   "losing_hold_cap": 30, "suppress_same_dir": True,  "eod_minutes_et": NY_EOD_1530},
    {"label": "LC35+SUPP+EOD",   "losing_hold_cap": 35, "suppress_same_dir": True,  "eod_minutes_et": NY_EOD_1530},
]


def main():
    print("=" * 130)
    print("MES v9.4 LOSING CAP + DIRECTIONAL SUPPRESSION")
    print("After cutting a loser at bar N, block same-direction re-entries until SM flips.")
    print(f"Baseline: SM(20/12/400/255) RSI(10/55/45) CD=15 SL=0")
    print(f"Commission: ${COMMISSION_PER_SIDE:.2f}/side, ${DOLLAR_PER_PT:.2f}/point")
    print("=" * 130)

    df_1m = load_mes_databento()
    print(f"  {len(df_1m)} bars, {df_1m.index[0].date()} to {df_1m.index[-1].date()}")

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

    # Run all configs
    print("\nRunning all configs...")
    all_results = {}
    for cfg in CONFIGS:
        label = cfg["label"]
        trades = run_backtest_losecap(
            opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
            losing_hold_cap=cfg["losing_hold_cap"],
            suppress_same_dir=cfg["suppress_same_dir"],
            eod_minutes_et=cfg["eod_minutes_et"],
        )
        all_results[label] = trades
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        print(f"  {label:<22} {fmt_sc(sc)}")

    months_list = get_months(all_results['BASELINE'])
    month_labels = [f"{y}-{m:02d}" for y, m in months_list]
    bl_sc = score_trades(all_results['BASELINE'], COMMISSION_PER_SIDE, DOLLAR_PER_PT)

    # ==================================================================
    # SECTION 1: Full Period Stats
    # ==================================================================
    print("\n" + "=" * 130)
    print("SECTION 1: FULL PERIOD STATS")
    print("=" * 130)

    print(f"\n  {'Config':<22} | {'Trades':>6} | {'WR%':>5} | {'PF':>6} | "
          f"{'Net$':>9} | {'d$':>8} | {'dPF':>6} | {'MaxDD$':>8} | {'Sharpe':>6} | "
          f"{'Exits':>30}")
    print(f"  {'-' * 125}")

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
    # SECTION 2: Monthly P&L Heatmap
    # ==================================================================
    print("\n" + "=" * 130)
    print("SECTION 2: MONTHLY P&L ($ net)")
    print("=" * 130)

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
    # SECTION 3: LOMO
    # ==================================================================
    print("\n" + "=" * 130)
    print("SECTION 3: LEAVE-ONE-MONTH-OUT CROSS-VALIDATION")
    print("=" * 130)

    bl_holdout_pfs = {}
    bl_holdout_nets = {}
    for y, m in months_list:
        mt = trades_in_month(all_results['BASELINE'], y, m)
        sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        bl_holdout_pfs[(y, m)] = sc['pf'] if sc else 0
        bl_holdout_nets[(y, m)] = sc['net_dollar'] if sc else 0

    header = f"  {'Config':<22} |"
    for ml in month_labels:
        header += f" {ml:>8}"
    header += f" | {'PF Wins':>7} {'$ Wins':>6} {'Avg dPF':>7} {'Avg d$':>7}"
    print(header)
    print(f"  {'-' * (26 + len(months_list) * 9 + 35)}")

    row = f"  {'BASELINE (ref PF)':<22} |"
    for y, m in months_list:
        row += f" {bl_holdout_pfs[(y, m)]:>8.3f}"
    row += f" |     --     --      --      --"
    print(row)

    lomo_results = {}
    for cfg in CONFIGS[1:]:
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

    # ==================================================================
    # SECTION 4: Per-month $ delta
    # ==================================================================
    print("\n" + "=" * 130)
    print("SECTION 4: PER-MONTH $ DELTA vs BASELINE")
    print("=" * 130)

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
    # SECTION 5: Suppressed vs unsuppressed comparison
    # ==================================================================
    print("\n" + "=" * 130)
    print("SECTION 5: EFFECT OF DIRECTIONAL SUPPRESSION")
    print("  Compare LC with and without suppression")
    print("=" * 130)

    for cap in [25, 30, 35]:
        plain_label = f"LC{cap}"
        supp_label = f"LC{cap}+SUPP"
        if plain_label not in all_results or supp_label not in all_results:
            continue

        plain = all_results[plain_label]
        supp = all_results[supp_label]
        bl = all_results['BASELINE']

        plain_sc = score_trades(plain, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        supp_sc = score_trades(supp, COMMISSION_PER_SIDE, DOLLAR_PER_PT)

        print(f"\n  --- LC{cap} ---")
        print(f"  Without suppression: {plain_sc['count']}t, PF {plain_sc['pf']}, "
              f"${plain_sc['net_dollar']:+.2f} (d$ {plain_sc['net_dollar'] - bl_sc['net_dollar']:+.2f})")
        print(f"  With suppression:    {supp_sc['count']}t, PF {supp_sc['pf']}, "
              f"${supp_sc['net_dollar']:+.2f} (d$ {supp_sc['net_dollar'] - bl_sc['net_dollar']:+.2f})")
        print(f"  Suppression effect:  {supp_sc['count'] - plain_sc['count']} trades, "
              f"${supp_sc['net_dollar'] - plain_sc['net_dollar']:+.2f}")

        # How many trades were suppressed?
        plain_entries = set(str(t['entry_time']) for t in plain)
        supp_entries = set(str(t['entry_time']) for t in supp)
        suppressed = plain_entries - supp_entries
        new_trades = supp_entries - plain_entries

        # What were the suppressed trades?
        plain_map = {str(t['entry_time']): t for t in plain}
        suppressed_trades = [plain_map[k] for k in suppressed if k in plain_map]

        if suppressed_trades:
            sup_pnls = [(t['pts'] - COMM_PTS) * DOLLAR_PER_PT for t in suppressed_trades]
            sup_wins = sum(1 for p in sup_pnls if p > 0)
            print(f"\n  Trades SUPPRESSED (blocked by directional rule): {len(suppressed_trades)}")
            print(f"    Winners among suppressed: {sup_wins}/{len(suppressed_trades)} "
                  f"({sup_wins/len(suppressed_trades)*100:.0f}%)")
            print(f"    Total P&L of suppressed:  ${sum(sup_pnls):+.2f}")
            print(f"    Avg P&L of suppressed:    ${np.mean(sup_pnls):+.2f}")

            # Show each suppressed trade
            print(f"\n    {'#':>3}  {'Side':<5}  {'P&L$':>8}  {'Bars':>5}  {'Exit':>10}  {'Entry Time':<16}")
            print(f"    {'-' * 65}")
            suppressed_trades.sort(key=lambda t: t['entry_time'])
            for idx, t in enumerate(suppressed_trades):
                pnl = (t['pts'] - COMM_PTS) * DOLLAR_PER_PT
                et = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
                print(f"    {idx+1:>3}  {t['side']:<5}  ${pnl:>+7.2f}  {t['bars']:>5}  "
                      f"{t['result']:>10}  {et}")

        if new_trades:
            supp_map = {str(t['entry_time']): t for t in supp}
            new_trade_list = [supp_map[k] for k in new_trades if k in supp_map]
            if new_trade_list:
                new_pnls = [(t['pts'] - COMM_PTS) * DOLLAR_PER_PT for t in new_trade_list]
                new_wins = sum(1 for p in new_pnls if p > 0)
                print(f"\n  NEW trades (created by suppression changing cooldown): {len(new_trade_list)}")
                print(f"    Winners: {new_wins}/{len(new_trade_list)}")
                print(f"    Total P&L: ${sum(new_pnls):+.2f}")

    # ==================================================================
    # SECTION 6: Ranking
    # ==================================================================
    print("\n" + "=" * 130)
    print("SECTION 6: RANKING")
    print("=" * 130)

    ranked = sorted(lomo_results.items(),
                    key=lambda x: (x[1]['pf_wins'], x[1]['avg_dpf'], x[1]['avg_dnet']),
                    reverse=True)

    print(f"\n  {'Rank':>4}  {'Config':<22}  {'LOMO PF Wins':>12}  "
          f"{'LOMO $ Wins':>11}  {'Avg dPF':>7}  {'Avg d$/mo':>9}")
    print(f"  {'-' * 75}")

    for rank, (label, res) in enumerate(ranked, 1):
        marker = " <<<" if rank <= 3 else ""
        sc = score_trades(all_results[label], COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        print(f"  {rank:>4}  {label:<22}  {res['pf_wins']:>12}/7  "
              f"{res['net_wins']:>11}/7  {res['avg_dpf']:>+6.3f}  "
              f"${res['avg_dnet']:>+8.2f}  "
              f"[{sc['count']}t PF {sc['pf']} ${sc['net_dollar']:+.0f}]{marker}")

    print("\n" + "=" * 130)
    print("DONE")
    print("=" * 130)


if __name__ == "__main__":
    main()
