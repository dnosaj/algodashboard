"""
MES v9.4 RSI Exit Research — v2 (Expert Validation)
=====================================================
Three validation methods:

1. LEAVE-ONE-MONTH-OUT (LOMO): Train on 6 months, test on holdout month.
   Repeat 7 times. Shows whether the exit helps across all market regimes,
   not just one lucky half.

2. WALK-FORWARD: Train months 1-3 → test month 4, train 1-4 → test 5, etc.
   Simulates real trading — never looks ahead.

3. COUNTERFACTUAL ANALYSIS: For every trade where RSI exit WOULD fire,
   compare the RSI exit P&L vs the SM flip exit P&L the trade actually got.
   Directly answers: "Would I have been better off exiting on RSI?"

Also expands the signal space beyond simple level crosses:
- RSI drop from peak: RSI fell N+ points from its high during the trade
- RSI slope: RSI declining for N consecutive 5-min bars while in a position

Data: Databento MES 1-min, Aug 17 2025 - Feb 13 2026
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    compute_et_minutes, NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)

# ---------------------------------------------------------------------------
# MES Config
# ---------------------------------------------------------------------------
COMMISSION_PER_SIDE = 1.25
DOLLAR_PER_PT = 5.0
COMM_PTS = (COMMISSION_PER_SIDE * 2) / DOLLAR_PER_PT  # 0.50 pts round-trip

SM_INDEX, SM_FLOW, SM_NORM, SM_EMA = 20, 12, 400, 255
RSI_LEN = 10
RSI_BUY, RSI_SELL = 55, 45
COOLDOWN = 15
MAX_LOSS_PTS = 0


# ---------------------------------------------------------------------------
# Backtest engine with RSI exit (same as v1, plus RSI-drop-from-peak exit)
# ---------------------------------------------------------------------------
def run_backtest_rsi_exit(
    opens, highs, lows, closes, sm, rsi_5m_curr, rsi_5m_prev, times,
    rsi_buy, rsi_sell, sm_threshold, cooldown_bars, max_loss_pts,
    # RSI exit params
    rsi_exit_long_below=0,
    rsi_exit_short_above=100,
    rsi_exit_cross_only=True,
    rsi_exit_profit_gate=False,
    rsi_exit_min_profit_pts=0,
    # RSI drop from peak exit
    rsi_drop_from_peak=0,       # Exit if RSI drops N+ pts from its peak since entry (0=off)
    rsi_drop_profit_gate=False,
):
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    max_equity = 0.0
    rsi_peak = 0.0    # track RSI peak (for longs) or trough (for shorts) during trade
    entry_rsi = 0.0   # RSI at entry time

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
            "entry_rsi": entry_rsi, "exit_rsi": rsi_5m_curr[exit_i - 1] if exit_i > 0 else 0,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]
        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
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

        # Track RSI extremes during trade
        if trade_state == 1:
            if rsi_prev > rsi_peak:
                rsi_peak = rsi_prev
        elif trade_state == -1:
            if rsi_prev < rsi_peak:  # rsi_peak = trough for shorts
                rsi_peak = rsi_prev

        # -- EOD Close --
        if trade_state != 0 and bar_mins_et >= NY_CLOSE_ET:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            max_equity = 0.0
            continue

        # -- Exits --
        if trade_state == 1:
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                continue

            # RSI level/cross exit
            if rsi_exit_long_below > 0:
                fire = False
                if rsi_exit_cross_only:
                    fire = rsi_prev < rsi_exit_long_below and rsi_prev2 >= rsi_exit_long_below
                else:
                    fire = rsi_prev < rsi_exit_long_below
                if fire:
                    unrealized = closes[i - 1] - entry_price
                    if not rsi_exit_profit_gate or unrealized > rsi_exit_min_profit_pts:
                        close_trade("long", entry_price, opens[i], entry_idx, i, "RSI_EXIT")
                        trade_state = 0
                        exit_bar = i
                        max_equity = 0.0
                        continue

            # RSI drop from peak exit (longs: RSI peaked and fell N pts)
            if rsi_drop_from_peak > 0 and rsi_peak - rsi_prev >= rsi_drop_from_peak:
                unrealized = closes[i - 1] - entry_price
                if not rsi_drop_profit_gate or unrealized > 0:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "RSI_DROP")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # SM flip exit
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0

        elif trade_state == -1:
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                continue

            # RSI level/cross exit (shorts: exit when RSI rises)
            if rsi_exit_short_above < 100:
                fire = False
                if rsi_exit_cross_only:
                    fire = rsi_prev > rsi_exit_short_above and rsi_prev2 <= rsi_exit_short_above
                else:
                    fire = rsi_prev > rsi_exit_short_above
                if fire:
                    unrealized = entry_price - closes[i - 1]
                    if not rsi_exit_profit_gate or unrealized > rsi_exit_min_profit_pts:
                        close_trade("short", entry_price, opens[i], entry_idx, i, "RSI_EXIT")
                        trade_state = 0
                        exit_bar = i
                        max_equity = 0.0
                        continue

            # RSI drop from peak for shorts (RSI troughed and rose N pts)
            if rsi_drop_from_peak > 0 and rsi_prev - rsi_peak >= rsi_drop_from_peak:
                unrealized = entry_price - closes[i - 1]
                if not rsi_drop_profit_gate or unrealized > 0:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "RSI_DROP")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0

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
                    max_equity = 0.0
                    rsi_peak = rsi_prev
                    entry_rsi = rsi_prev
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    max_equity = 0.0
                    rsi_peak = rsi_prev  # trough tracker for shorts
                    entry_rsi = rsi_prev

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trades_in_month(trades, year, month):
    out = []
    for t in trades:
        ts = pd.Timestamp(t['entry_time'])
        if ts.year == year and ts.month == month:
            out.append(t)
    return out


def get_months(trades):
    """Return sorted list of (year, month) tuples present in trades."""
    months = set()
    for t in trades:
        ts = pd.Timestamp(t['entry_time'])
        months.add((ts.year, ts.month))
    return sorted(months)


def fmt_sc(sc):
    if sc is None:
        return "NO TRADES"
    return (f"{sc['count']:>3}t  WR {sc['win_rate']:>5.1f}%  "
            f"PF {sc['pf']:>6.3f}  ${sc['net_dollar']:>+8.2f}")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
CONFIGS = [
    # Baseline
    {"label": "BASELINE (SM flip)",
     "rsi_exit_long_below": 0, "rsi_exit_short_above": 100,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    # --- RSI cross exits (ungated) ---
    {"label": "R1: cross < 45",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    {"label": "R2: cross < 40",
     "rsi_exit_long_below": 40, "rsi_exit_short_above": 60,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    {"label": "R3: cross < 35",
     "rsi_exit_long_below": 35, "rsi_exit_short_above": 65,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    # --- RSI cross exits (profit-gated) ---
    {"label": "R4: cross < 45, profit-gated",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    {"label": "R5: cross < 40, profit-gated",
     "rsi_exit_long_below": 40, "rsi_exit_short_above": 60,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    {"label": "R6: cross < 35, profit-gated",
     "rsi_exit_long_below": 35, "rsi_exit_short_above": 65,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    # --- RSI level (not cross) exits, profit-gated ---
    {"label": "R7: level < 45, profit-gated",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": False, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    {"label": "R8: level < 40, profit-gated",
     "rsi_exit_long_below": 40, "rsi_exit_short_above": 60,
     "rsi_exit_cross_only": False, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 0,
     "rsi_drop_profit_gate": False},

    # --- RSI drop from peak (the divergence signal) ---
    {"label": "D1: RSI drop 15 from peak",
     "rsi_exit_long_below": 0, "rsi_exit_short_above": 100,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 15,
     "rsi_drop_profit_gate": False},

    {"label": "D2: RSI drop 20 from peak",
     "rsi_exit_long_below": 0, "rsi_exit_short_above": 100,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 20,
     "rsi_drop_profit_gate": False},

    {"label": "D3: RSI drop 25 from peak",
     "rsi_exit_long_below": 0, "rsi_exit_short_above": 100,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 25,
     "rsi_drop_profit_gate": False},

    {"label": "D4: RSI drop 15, profit-gated",
     "rsi_exit_long_below": 0, "rsi_exit_short_above": 100,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 15,
     "rsi_drop_profit_gate": True},

    {"label": "D5: RSI drop 20, profit-gated",
     "rsi_exit_long_below": 0, "rsi_exit_short_above": 100,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 20,
     "rsi_drop_profit_gate": True},

    {"label": "D6: RSI drop 25, profit-gated",
     "rsi_exit_long_below": 0, "rsi_exit_short_above": 100,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 25,
     "rsi_drop_profit_gate": True},

    # --- Combo: RSI cross + drop from peak, profit-gated ---
    {"label": "C1: cross<45 + drop15, PG",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 15,
     "rsi_drop_profit_gate": True},

    {"label": "C2: cross<45 + drop20, PG",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0, "rsi_drop_from_peak": 20,
     "rsi_drop_profit_gate": True},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 115)
    print("MES v9.4 RSI EXIT RESEARCH — v2 (Expert Validation)")
    print(f"Baseline: SM(20/12/400/255) RSI(10/55/45) CD=15 SL=0")
    print(f"Commission: ${COMMISSION_PER_SIDE:.2f}/side, ${DOLLAR_PER_PT:.2f}/point")
    print("=" * 115)

    # Load and compute
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

    df_for_resample = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_resample['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_for_resample)

    print(f"Mapping RSI({RSI_LEN}) 5-min → 1-min...")
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=RSI_LEN)

    # ------------------------------------------------------------------
    # Run all configs on full data
    # ------------------------------------------------------------------
    print("\nRunning all configs on full period...")
    all_results = {}
    for cfg in CONFIGS:
        trades = run_backtest_rsi_exit(
            opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
            RSI_BUY, RSI_SELL, 0.0, COOLDOWN, MAX_LOSS_PTS,
            **{k: v for k, v in cfg.items() if k != 'label'})
        all_results[cfg['label']] = trades

    # ==================================================================
    # SECTION 1: Full period + monthly heatmap
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 1: FULL PERIOD + MONTHLY P&L HEATMAP")
    print("=" * 115)

    months_list = get_months(all_results['BASELINE (SM flip)'])
    month_labels = [f"{y}-{m:02d}" for y, m in months_list]

    # Header
    header = f"  {'Config':<33} | {'Full':>10} |"
    for ml in month_labels:
        header += f" {ml:>7}"
    header += " | RSI/DROP"
    print(header)
    print(f"  {'-' * (38 + 11 + len(months_list) * 8 + 12)}")

    baseline_sc = score_trades(all_results['BASELINE (SM flip)'],
                               COMMISSION_PER_SIDE, DOLLAR_PER_PT)
    bl_monthly = {}
    for y, m in months_list:
        mt = trades_in_month(all_results['BASELINE (SM flip)'], y, m)
        sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        bl_monthly[(y, m)] = sc['net_dollar'] if sc else 0

    for cfg in CONFIGS:
        label = cfg['label']
        trades = all_results[label]
        sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        full_net = sc['net_dollar'] if sc else 0
        rsi_exits = (sc['exits'].get('RSI_EXIT', 0) +
                     sc['exits'].get('RSI_DROP', 0)) if sc else 0

        row = f"  {label:<33} | ${full_net:>+8.2f} |"
        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            msc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            mnet = msc['net_dollar'] if msc else 0
            # Color code: compare to baseline month
            bl_m = bl_monthly[(y, m)]
            delta = mnet - bl_m
            if abs(delta) < 1:
                marker = " "
            elif delta > 0:
                marker = "+"
            else:
                marker = "-"
            row += f" {mnet:>+6.0f}{marker}"
        row += f" | {rsi_exits:>4}"
        print(row)

    # ==================================================================
    # SECTION 2: LEAVE-ONE-MONTH-OUT
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 2: LEAVE-ONE-MONTH-OUT CROSS-VALIDATION")
    print("  For each config, hold out 1 month. Score = holdout month PF.")
    print("  Wins = months where holdout PF > baseline holdout PF")
    print("=" * 115)

    header = f"  {'Config':<33} |"
    for ml in month_labels:
        header += f" {ml:>7}"
    header += f" | {'Wins':>4} {'Avg dPF':>7}"
    print(header)
    print(f"  {'-' * (38 + len(months_list) * 8 + 16)}")

    # Baseline holdout PFs
    bl_holdout_pfs = {}
    for y, m in months_list:
        mt = trades_in_month(all_results['BASELINE (SM flip)'], y, m)
        sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        bl_holdout_pfs[(y, m)] = sc['pf'] if sc else 0

    for cfg in CONFIGS:
        label = cfg['label']
        trades = all_results[label]
        wins = 0
        dpf_sum = 0
        row = f"  {label:<33} |"

        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            pf = sc['pf'] if sc else 0
            bl_pf = bl_holdout_pfs[(y, m)]
            dpf = pf - bl_pf
            dpf_sum += dpf
            if dpf > 0.005:
                wins += 1
                row += f" {pf:>+6.3f}*"
            elif dpf < -0.005:
                row += f" {pf:>+6.3f} "
            else:
                row += f" {pf:>+6.3f}="
        avg_dpf = dpf_sum / len(months_list)
        row += f" | {wins:>4} {avg_dpf:>+7.3f}"
        print(row)

    print("\n  * = beats baseline that month   (blank) = worse   = = same")

    # ==================================================================
    # SECTION 3: WALK-FORWARD
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 3: WALK-FORWARD VALIDATION")
    print("  Train months 1-3 → test month 4, train 1-4 → test 5, etc.")
    print("  If exit helps on train window AND also helps on next OOS month → score +1")
    print("=" * 115)

    # For each config, compare walk-forward OOS performance
    header = f"  {'Config':<33} |"
    wf_test_months = months_list[3:]  # months 4-7 are test months
    for y, m in wf_test_months:
        header += f" {y}-{m:02d}:>7"
    header += f" | {'OOS Wins':>8} {'OOS dNet':>8}"
    print(header)
    print(f"  {'-' * (38 + len(wf_test_months) * 8 + 20)}")

    for cfg in CONFIGS:
        label = cfg['label']
        trades = all_results[label]
        bl_trades = all_results['BASELINE (SM flip)']

        row = f"  {label:<33} |"
        oos_wins = 0
        oos_dnet_total = 0

        for test_idx in range(3, len(months_list)):
            ty, tm = months_list[test_idx]
            # OOS month trades
            mt = trades_in_month(trades, ty, tm)
            bl_mt = trades_in_month(bl_trades, ty, tm)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            bl_sc = score_trades(bl_mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            net = sc['net_dollar'] if sc else 0
            bl_net = bl_sc['net_dollar'] if bl_sc else 0
            dnet = net - bl_net
            oos_dnet_total += dnet
            if dnet > 1:
                oos_wins += 1
                row += f" ${dnet:>+5.0f}* "
            elif dnet < -1:
                row += f" ${dnet:>+5.0f}  "
            else:
                row += f" ${dnet:>+5.0f}= "

        row += f" | {oos_wins:>8} ${oos_dnet_total:>+7.0f}"
        print(row)

    # ==================================================================
    # SECTION 4: COUNTERFACTUAL — Per-Trade What-If
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 4: COUNTERFACTUAL — What happens to trades RSI exit touches?")
    print("  For the best config, compare each RSI-exited trade to what")
    print("  the baseline did with that SAME entry (matched by entry_time).")
    print("=" * 115)

    # Find best config by: most LOMO wins, then highest full net
    best_label = None
    best_score = (-999, -1e9)
    bl_trades = all_results['BASELINE (SM flip)']

    for cfg in CONFIGS[1:]:  # skip baseline
        label = cfg['label']
        trades = all_results[label]
        lomo_wins = 0
        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            bl_mt = trades_in_month(bl_trades, y, m)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            bl_sc = score_trades(bl_mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            if sc and bl_sc and sc['pf'] > bl_sc['pf'] + 0.005:
                lomo_wins += 1
        full_sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        full_net = full_sc['net_dollar'] if full_sc else -1e9
        if (lomo_wins, full_net) > best_score:
            best_score = (lomo_wins, full_net)
            best_label = label

    # Also find best D-series (drop from peak) config
    best_d_label = None
    best_d_score = (-999, -1e9)
    for cfg in CONFIGS:
        if not cfg['label'].startswith('D'):
            continue
        label = cfg['label']
        trades = all_results[label]
        lomo_wins = 0
        for y, m in months_list:
            mt = trades_in_month(trades, y, m)
            bl_mt = trades_in_month(bl_trades, y, m)
            sc = score_trades(mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            bl_sc = score_trades(bl_mt, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            if sc and bl_sc and sc['pf'] > bl_sc['pf'] + 0.005:
                lomo_wins += 1
        full_sc = score_trades(trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
        full_net = full_sc['net_dollar'] if full_sc else -1e9
        if (lomo_wins, full_net) > best_d_score:
            best_d_score = (lomo_wins, full_net)
            best_d_label = label

    for analysis_label in [best_label, best_d_label]:
        if analysis_label is None:
            continue
        print(f"\n  Analyzing: {analysis_label}")

        trades = all_results[analysis_label]
        bl_entry_map = {}
        for t in bl_trades:
            key = str(t['entry_time'])
            bl_entry_map[key] = t

        rsi_exited = [t for t in trades
                      if t['result'] in ('RSI_EXIT', 'RSI_DROP')]
        if not rsi_exited:
            print("  (no RSI exits fired)")
            continue

        print(f"\n  {'#':>3}  {'Side':<5}  {'Entry$':>9}  "
              f"{'RSI Exit$':>9}  {'SM Exit$':>9}  "
              f"{'RSI P&L':>8}  {'SM P&L':>8}  {'Delta':>7}  "
              f"{'EntryRSI':>8}  {'ExitRSI':>7}  {'Bars':>5}  {'Entry Time':<16}")
        print(f"  {'-' * 120}")

        total_saved = 0
        wins = 0
        for idx, t in enumerate(rsi_exited):
            key = str(t['entry_time'])
            bl_t = bl_entry_map.get(key)
            if bl_t is None:
                continue

            rsi_pts = t['pts']
            sm_pts = bl_t['pts']
            rsi_pnl = (rsi_pts - COMM_PTS) * DOLLAR_PER_PT
            sm_pnl = (sm_pts - COMM_PTS) * DOLLAR_PER_PT
            delta = rsi_pnl - sm_pnl
            total_saved += delta
            if delta > 0:
                wins += 1

            et = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
            entry_rsi = t.get('entry_rsi', 0)
            exit_rsi = t.get('exit_rsi', 0)
            marker = " <-- saved" if delta > 5 else (" <-- HURT" if delta < -5 else "")

            print(f"  {idx+1:>3}  {t['side']:<5}  {t['entry']:>9.2f}  "
                  f"{t['exit']:>9.2f}  {bl_t['exit']:>9.2f}  "
                  f"${rsi_pnl:>+7.2f}  ${sm_pnl:>+7.2f}  ${delta:>+6.2f}  "
                  f"{entry_rsi:>8.1f}  {exit_rsi:>7.1f}  {t['bars']:>5}  "
                  f"{et}{marker}")

        print(f"\n  Summary: {len(rsi_exited)} RSI exits, "
              f"{wins}/{len(rsi_exited)} better than SM flip, "
              f"Net delta: ${total_saved:+.2f}")

    # ==================================================================
    # SECTION 5: MES P&L Distribution — Where Are Losses Coming From?
    # ==================================================================
    print("\n" + "=" * 115)
    print("SECTION 5: BASELINE TRADE ANALYSIS — Where Do Losses Come From?")
    print("  Understanding the P&L distribution helps identify which trades")
    print("  an RSI exit should target.")
    print("=" * 115)

    bl_trades_scored = []
    for t in bl_trades:
        pnl = (t['pts'] - COMM_PTS) * DOLLAR_PER_PT
        bl_trades_scored.append({**t, 'pnl': pnl})

    # Bucket by: profitable at some point (had positive MFE) vs never profitable
    # We can approximate MFE from the trade data
    winners = [t for t in bl_trades_scored if t['pnl'] > 0]
    losers = [t for t in bl_trades_scored if t['pnl'] <= 0]

    # Bucket losers by how long they were held
    short_losers = [t for t in losers if t['bars'] < 15]
    medium_losers = [t for t in losers if 15 <= t['bars'] < 50]
    long_losers = [t for t in losers if t['bars'] >= 50]

    print(f"\n  Total trades: {len(bl_trades_scored)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(bl_trades_scored)*100:.0f}%), "
          f"avg ${np.mean([t['pnl'] for t in winners]):+.2f}, "
          f"avg {np.mean([t['bars'] for t in winners]):.0f} bars")
    print(f"  Losers:  {len(losers)} ({len(losers)/len(bl_trades_scored)*100:.0f}%), "
          f"avg ${np.mean([t['pnl'] for t in losers]):+.2f}, "
          f"avg {np.mean([t['bars'] for t in losers]):.0f} bars")

    print(f"\n  Loser breakdown by hold time:")
    print(f"    Short  (<15 bars):  {len(short_losers):>3} trades, "
          f"avg ${np.mean([t['pnl'] for t in short_losers]):+.2f}" if short_losers else
          f"    Short  (<15 bars):    0 trades")
    print(f"    Medium (15-49):     {len(medium_losers):>3} trades, "
          f"avg ${np.mean([t['pnl'] for t in medium_losers]):+.2f}" if medium_losers else
          f"    Medium (15-49):       0 trades")
    print(f"    Long   (50+):       {len(long_losers):>3} trades, "
          f"avg ${np.mean([t['pnl'] for t in long_losers]):+.2f}" if long_losers else
          f"    Long   (50+):         0 trades")

    # Top 10 worst trades — are they the ones RSI exit would have caught?
    worst = sorted(bl_trades_scored, key=lambda t: t['pnl'])[:15]
    print(f"\n  15 WORST TRADES (baseline SM flip exit):")
    print(f"  {'#':>3}  {'Side':<5}  {'P&L$':>8}  {'Pts':>7}  {'Bars':>5}  "
          f"{'Exit':>8}  {'Entry Time':<16}")
    print(f"  {'-' * 65}")
    for idx, t in enumerate(worst):
        et = pd.Timestamp(t['entry_time']).strftime('%m-%d %H:%M')
        print(f"  {idx+1:>3}  {t['side']:<5}  ${t['pnl']:>+7.2f}  "
              f"{t['pts']:>+7.2f}  {t['bars']:>5}  "
              f"{t['result']:>8}  {et}")

    # RSI at entry for worst trades
    print(f"\n  RSI at entry/exit for worst trades (from best RSI config):")
    if best_label:
        best_trades = all_results[best_label]
        best_entry_map = {str(t['entry_time']): t for t in best_trades}
        for idx, t in enumerate(worst[:10]):
            key = str(t['entry_time'])
            bt = best_entry_map.get(key)
            if bt:
                ersi = bt.get('entry_rsi', 0)
                xrsi = bt.get('exit_rsi', 0)
                was_rsi_exit = bt['result'] in ('RSI_EXIT', 'RSI_DROP')
                marker = " <-- RSI EXIT" if was_rsi_exit else ""
                print(f"    #{idx+1}: entry RSI={ersi:.1f}, exit RSI={xrsi:.1f}, "
                      f"best config exit: {bt['result']} ${(bt['pts']-COMM_PTS)*DOLLAR_PER_PT:+.2f}{marker}")

    print("\n" + "=" * 115)
    print("DONE")
    print("=" * 115)


if __name__ == "__main__":
    main()
