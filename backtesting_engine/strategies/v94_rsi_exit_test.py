"""
MES v9.4 RSI Exit Research
===========================
Tests RSI-based exits for MES to address the SM-flip lag problem:
  SM(20/12/400/255) is intentionally slow — great for entries, but exits
  lag badly when momentum dies before the trend indicator flips.

Hypothesis: Exit when RSI reverses against the position (crosses below
rsi_sell for longs, above rsi_buy for shorts), even if SM still agrees.
This catches momentum death 15-30 min before SM flips.

Validation: Two-way cross-validation
  Split A: Train Aug-Oct, Test Nov-Feb
  Split B: Train Nov-Feb, Test Aug-Oct
  A config only PASSES if it works in BOTH directions.

Data: Databento MES 1-min, Aug 17 2025 - Feb 13 2026
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    compute_et_minutes, NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)

# ---------------------------------------------------------------------------
# MES Config
# ---------------------------------------------------------------------------
COMMISSION_PER_SIDE = 1.25  # MES commission
DOLLAR_PER_PT = 5.0

# v9.4 baseline params
SM_INDEX = 20
SM_FLOW = 12
SM_NORM = 400
SM_EMA = 255
RSI_LEN = 10
RSI_BUY = 55
RSI_SELL = 45
COOLDOWN = 15
MAX_LOSS_PTS = 0  # SL off for MES

# Split boundary: Nov 1, 2025
SPLIT_DATE = pd.Timestamp("2025-11-01", tz="UTC")


# ---------------------------------------------------------------------------
# RSI Exit Backtest Engine (fork of run_backtest_v10 with RSI exit logic)
# ---------------------------------------------------------------------------
def run_backtest_rsi_exit(
    opens, highs, lows, closes, sm, rsi_5m_curr, rsi_5m_prev, times,
    rsi_buy, rsi_sell, sm_threshold, cooldown_bars, max_loss_pts,
    # RSI exit params
    rsi_exit_long_below=0,      # Exit long when RSI crosses below this (0=off)
    rsi_exit_short_above=100,   # Exit short when RSI crosses above this (100=off)
    rsi_exit_cross_only=True,   # True=require cross, False=level is enough
    rsi_exit_profit_gate=False, # Only fire RSI exit if trade is profitable
    rsi_exit_min_profit_pts=0,  # Min profit in pts before RSI exit can fire (0=any profit)
):
    """
    v9.4 backtest engine with optional RSI-based exit.

    RSI exit fires BEFORE SM flip in the exit priority chain:
      EOD > Max Loss > RSI Exit > SM Flip

    RSI exit uses the same mapped 5-min RSI as entries.
    Signal from bar i-1, fill at bar i open.
    """
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    max_equity = 0.0

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

        # Previous bar signals
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]
        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI (mapped 5-min)
        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        # Episode reset (zero-crossing, matches v11.1 fix)
        if sm_flipped_bull or sm_prev <= 0:
            long_used = False
        if sm_flipped_bear or sm_prev >= 0:
            short_used = False

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
            # Max loss stop
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                continue

            # RSI exit for longs
            if rsi_exit_long_below > 0:
                rsi_exit_fire = False
                if rsi_exit_cross_only:
                    rsi_exit_fire = (rsi_prev < rsi_exit_long_below
                                     and rsi_prev2 >= rsi_exit_long_below)
                else:
                    rsi_exit_fire = rsi_prev < rsi_exit_long_below

                if rsi_exit_fire:
                    unrealized = closes[i - 1] - entry_price
                    if rsi_exit_profit_gate:
                        if unrealized > rsi_exit_min_profit_pts:
                            close_trade("long", entry_price, opens[i],
                                        entry_idx, i, "RSI_EXIT")
                            trade_state = 0
                            exit_bar = i
                            max_equity = 0.0
                            continue
                    else:
                        close_trade("long", entry_price, opens[i],
                                    entry_idx, i, "RSI_EXIT")
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
            # Max loss stop
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                continue

            # RSI exit for shorts
            if rsi_exit_short_above < 100:
                rsi_exit_fire = False
                if rsi_exit_cross_only:
                    rsi_exit_fire = (rsi_prev > rsi_exit_short_above
                                     and rsi_prev2 <= rsi_exit_short_above)
                else:
                    rsi_exit_fire = rsi_prev > rsi_exit_short_above

                if rsi_exit_fire:
                    unrealized = entry_price - closes[i - 1]
                    if rsi_exit_profit_gate:
                        if unrealized > rsi_exit_min_profit_pts:
                            close_trade("short", entry_price, opens[i],
                                        entry_idx, i, "RSI_EXIT")
                            trade_state = 0
                            exit_bar = i
                            max_equity = 0.0
                            continue
                    else:
                        close_trade("short", entry_price, opens[i],
                                    entry_idx, i, "RSI_EXIT")
                        trade_state = 0
                        exit_bar = i
                        max_equity = 0.0
                        continue

            # SM flip exit
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0

        # -- Entry logic --
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
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    max_equity = 0.0

    return trades


# ---------------------------------------------------------------------------
# Data Loader
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
def split_trades(trades, split_date):
    """Split trades into before/after split_date based on entry_time."""
    before, after = [], []
    for t in trades:
        ts = pd.Timestamp(t['entry_time'])
        if ts.tz is None:
            ts = ts.tz_localize('UTC')
        if ts < split_date:
            before.append(t)
        else:
            after.append(t)
    return before, after


def print_result(sc, label=""):
    if sc is None:
        print(f"  {label:<40} NO TRADES")
        return
    exits_str = "  ".join(f"{k}:{v}" for k, v in sorted(sc['exits'].items()))
    print(f"  {label:<40} {sc['count']:>4} trades  "
          f"WR {sc['win_rate']:>5.1f}%  PF {sc['pf']:>6.3f}  "
          f"Net ${sc['net_dollar']:>+9.2f}  MaxDD ${sc['max_dd_dollar']:>8.2f}  "
          f"Sharpe {sc['sharpe']:>6.3f}  | {exits_str}")


def evaluate_config(label, trades_full, split_date,
                    commission=COMMISSION_PER_SIDE, dpp=DOLLAR_PER_PT):
    """Evaluate a config on full, split A (train early/test late),
    and split B (train late/test early). Returns pass/fail verdict."""
    early, late = split_trades(trades_full, split_date)

    sc_full = score_trades(trades_full, commission, dpp)
    sc_early = score_trades(early, commission, dpp)
    sc_late = score_trades(late, commission, dpp)

    return {
        'label': label,
        'full': sc_full,
        'early': sc_early,  # Aug-Oct
        'late': sc_late,     # Nov-Feb
        'trades_full': trades_full,
        'trades_early': early,
        'trades_late': late,
    }


# ---------------------------------------------------------------------------
# RSI Exit Configs
# ---------------------------------------------------------------------------
CONFIGS = [
    # Baseline
    {"label": "BASELINE (SM flip only)",
     "rsi_exit_long_below": 0, "rsi_exit_short_above": 100,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0},

    # R1-R4: RSI cross exit at various thresholds, no gate
    {"label": "R1: RSI cross < 45 (rsi_sell)",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0},

    {"label": "R2: RSI cross < 40",
     "rsi_exit_long_below": 40, "rsi_exit_short_above": 60,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0},

    {"label": "R3: RSI cross < 35",
     "rsi_exit_long_below": 35, "rsi_exit_short_above": 65,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0},

    {"label": "R4: RSI cross < 30",
     "rsi_exit_long_below": 30, "rsi_exit_short_above": 70,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": False,
     "rsi_exit_min_profit_pts": 0},

    # R5-R7: RSI cross exit, profit-gated (only exit if profitable)
    {"label": "R5: RSI cross < 45, profit-gated",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0},

    {"label": "R6: RSI cross < 40, profit-gated",
     "rsi_exit_long_below": 40, "rsi_exit_short_above": 60,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0},

    {"label": "R7: RSI cross < 35, profit-gated",
     "rsi_exit_long_below": 35, "rsi_exit_short_above": 65,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0},

    # R8: RSI level (not cross) below rsi_sell, profit-gated
    {"label": "R8: RSI level < 45, profit-gated",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": False, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0},

    # R9-R11: Profit-gated with minimum profit threshold
    {"label": "R9: RSI cross < 45, profit > 2pts",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 2},

    {"label": "R10: RSI cross < 40, profit > 2pts",
     "rsi_exit_long_below": 40, "rsi_exit_short_above": 60,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 2},

    {"label": "R11: RSI cross < 45, profit > 5pts",
     "rsi_exit_long_below": 45, "rsi_exit_short_above": 55,
     "rsi_exit_cross_only": True, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 5},

    # R12: RSI level (not cross) below 40, profit-gated
    {"label": "R12: RSI level < 40, profit-gated",
     "rsi_exit_long_below": 40, "rsi_exit_short_above": 60,
     "rsi_exit_cross_only": False, "rsi_exit_profit_gate": True,
     "rsi_exit_min_profit_pts": 0},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 110)
    print("MES v9.4 RSI EXIT RESEARCH")
    print(f"Data: Databento MES 1-min, Aug 17 2025 - Feb 13 2026")
    print(f"Commission: ${COMMISSION_PER_SIDE:.2f}/side, ${DOLLAR_PER_PT:.2f}/point")
    print(f"Split: {SPLIT_DATE.strftime('%Y-%m-%d')} (Early=Aug-Oct, Late=Nov-Feb)")
    print(f"Baseline: SM(20/12/400/255) RSI(10/55/45) CD=15 SL=0, SM flip exit")
    print("=" * 110)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading MES 1-min data...")
    df_1m = load_mes_databento()
    print(f"  {len(df_1m)} bars, {df_1m.index[0].date()} to {df_1m.index[-1].date()}")

    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    volumes = df_1m['Volume'].values
    times = df_1m.index.values

    # ------------------------------------------------------------------
    # Compute indicators
    # ------------------------------------------------------------------
    print("\nComputing SM(20/12/400/255) on 1-min bars...")
    sm = compute_smart_money(closes, volumes, SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)

    print("Resampling to 5-min for RSI...")
    df_for_resample = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_resample['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_for_resample)
    fivemin_times = df_5m.index.values
    fivemin_closes = df_5m['Close'].values
    print(f"  {len(df_5m)} 5-min bars")

    print(f"Mapping RSI({RSI_LEN}) from 5-min to 1-min...")
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, fivemin_times, fivemin_closes, rsi_len=RSI_LEN)

    # ==================================================================
    # SECTION 1: Full Period Results
    # ==================================================================
    print("\n" + "=" * 110)
    print("SECTION 1: FULL PERIOD (6 months)")
    print("=" * 110)

    results = []
    for cfg in CONFIGS:
        trades = run_backtest_rsi_exit(
            opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
            rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
            sm_threshold=0.0, cooldown_bars=COOLDOWN,
            max_loss_pts=MAX_LOSS_PTS,
            rsi_exit_long_below=cfg["rsi_exit_long_below"],
            rsi_exit_short_above=cfg["rsi_exit_short_above"],
            rsi_exit_cross_only=cfg["rsi_exit_cross_only"],
            rsi_exit_profit_gate=cfg["rsi_exit_profit_gate"],
            rsi_exit_min_profit_pts=cfg["rsi_exit_min_profit_pts"],
        )
        ev = evaluate_config(cfg["label"], trades, SPLIT_DATE)
        results.append(ev)
        print_result(ev['full'], cfg["label"])

    # ==================================================================
    # SECTION 2: Two-Way Cross-Validation
    # ==================================================================
    print("\n" + "=" * 110)
    print("SECTION 2: TWO-WAY CROSS-VALIDATION")
    print(f"  Split A: Train Early (Aug-Oct) -> Test Late (Nov-Feb)")
    print(f"  Split B: Train Late (Nov-Feb) -> Test Early (Aug-Oct)")
    print("  PASS = profitable (PF > 1.0) in BOTH test periods")
    print("=" * 110)

    # Header
    print(f"\n  {'Config':<40} | {'Early PF':>8} {'Early$':>9} {'E-Trd':>5}"
          f" | {'Late PF':>8} {'Late$':>9} {'L-Trd':>5}"
          f" | {'Verdict':>10}")
    print(f"  {'-' * 105}")

    baseline_full = results[0]['full']

    for ev in results:
        sc_e = ev['early']
        sc_l = ev['late']

        e_pf = sc_e['pf'] if sc_e else 0
        e_net = sc_e['net_dollar'] if sc_e else 0
        e_cnt = sc_e['count'] if sc_e else 0
        l_pf = sc_l['pf'] if sc_l else 0
        l_net = sc_l['net_dollar'] if sc_l else 0
        l_cnt = sc_l['count'] if sc_l else 0

        # Verdict: PASS if both halves are profitable
        both_profitable = e_pf > 1.0 and l_pf > 1.0
        # Compare to baseline
        bl_e = results[0]['early']
        bl_l = results[0]['late']
        bl_e_net = bl_e['net_dollar'] if bl_e else 0
        bl_l_net = bl_l['net_dollar'] if bl_l else 0

        if ev['label'].startswith("BASELINE"):
            verdict = "BASELINE"
        elif both_profitable:
            # Check if it improves on baseline in both halves
            improves_both = (e_net > bl_e_net) and (l_net > bl_l_net)
            improves_one = (e_net > bl_e_net) or (l_net > bl_l_net)
            total_better = (e_net + l_net) > (bl_e_net + bl_l_net)
            if improves_both:
                verdict = "STRONG PASS"
            elif total_better:
                verdict = "PASS"
            else:
                verdict = "PASS (weak)"
        else:
            verdict = "FAIL"

        print(f"  {ev['label']:<40} | {e_pf:>8.3f} ${e_net:>+8.2f} {e_cnt:>5}"
              f" | {l_pf:>8.3f} ${l_net:>+8.2f} {l_cnt:>5}"
              f" | {verdict:>10}")

    # ==================================================================
    # SECTION 3: Delta vs Baseline
    # ==================================================================
    print("\n" + "=" * 110)
    print("SECTION 3: DELTA vs BASELINE (SM flip only)")
    print("=" * 110)

    bl = results[0]
    bl_full_net = bl['full']['net_dollar'] if bl['full'] else 0
    bl_full_pf = bl['full']['pf'] if bl['full'] else 0

    print(f"\n  {'Config':<40} | {'dPF':>7} | {'dNet$':>9} | {'Avg Bars':>8} | {'RSI Exits':>9}")
    print(f"  {'-' * 85}")

    for ev in results:
        sc = ev['full']
        if sc is None:
            continue
        dpf = sc['pf'] - bl_full_pf
        dnet = sc['net_dollar'] - bl_full_net
        rsi_exits = sc['exits'].get('RSI_EXIT', 0)
        avg_bars = sc['avg_bars']

        print(f"  {ev['label']:<40} | {dpf:>+7.3f} | ${dnet:>+8.2f} | {avg_bars:>8.1f} | {rsi_exits:>9}")

    # ==================================================================
    # SECTION 4: Monthly breakdown for top configs
    # ==================================================================
    print("\n" + "=" * 110)
    print("SECTION 4: MONTHLY BREAKDOWN (baseline + top configs)")
    print("=" * 110)

    # Always show baseline + any that passed
    show_configs = [results[0]]  # baseline
    for ev in results[1:]:
        sc_e = ev['early']
        sc_l = ev['late']
        if sc_e and sc_l and sc_e['pf'] > 1.0 and sc_l['pf'] > 1.0:
            show_configs.append(ev)

    # If nothing passed, show top 3 by full net
    if len(show_configs) == 1:
        print("\n  (No configs passed both halves. Showing top 3 by full-period net.)")
        sorted_results = sorted(results[1:],
                                key=lambda e: e['full']['net_dollar'] if e['full'] else -1e9,
                                reverse=True)
        show_configs.extend(sorted_results[:3])

    for ev in show_configs:
        trades = ev['trades_full']
        if not trades:
            continue

        months = {}
        for t in trades:
            ts = pd.Timestamp(t['entry_time'])
            key = ts.strftime('%Y-%m')
            months.setdefault(key, []).append(t)

        print(f"\n  {ev['label']}")
        print(f"  {'Month':>8}  {'Trades':>6}  {'WR%':>6}  {'PF':>7}  {'Net$':>9}  {'RSI_EXIT':>8}  {'SM_FLIP':>7}  {'EOD':>4}")
        print(f"  {'-' * 70}")

        for mk in sorted(months.keys()):
            sc = score_trades(months[mk], COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            if sc:
                rsi_ex = sc['exits'].get('RSI_EXIT', 0)
                sm_ex = sc['exits'].get('SM_FLIP', 0)
                eod_ex = sc['exits'].get('EOD', 0)
                print(f"  {mk:>8}  {sc['count']:>6}  {sc['win_rate']:>5.1f}%  "
                      f"{sc['pf']:>7.3f}  ${sc['net_dollar']:>+8.2f}  "
                      f"{rsi_ex:>8}  {sm_ex:>7}  {eod_ex:>4}")

    # ==================================================================
    # SECTION 5: Per-trade comparison for best config vs baseline
    # ==================================================================
    print("\n" + "=" * 110)
    print("SECTION 5: TRADE-BY-TRADE RSI EXITS (showing what RSI exit captured)")
    print("=" * 110)

    # Find the best passing config, or best overall if none pass
    best = None
    for ev in results[1:]:
        sc_e = ev['early']
        sc_l = ev['late']
        if sc_e and sc_l and sc_e['pf'] > 1.0 and sc_l['pf'] > 1.0:
            if best is None or ev['full']['net_dollar'] > best['full']['net_dollar']:
                best = ev
    if best is None and len(results) > 1:
        best = max(results[1:],
                   key=lambda e: e['full']['net_dollar'] if e['full'] else -1e9)

    if best:
        print(f"\n  Best config: {best['label']}")
        rsi_trades = [t for t in best['trades_full'] if t['result'] == 'RSI_EXIT']
        bl_trades = results[0]['trades_full']

        print(f"\n  RSI EXIT trades ({len(rsi_trades)} total):")
        print(f"  {'#':>3}  {'Side':<5}  {'Entry$':>10}  {'Exit$':>10}  {'Pts':>7}  "
              f"{'P&L$':>8}  {'Bars':>5}  {'Entry Time':<20}")
        print(f"  {'-' * 80}")

        comm_pts = (COMMISSION_PER_SIDE * 2) / DOLLAR_PER_PT
        for idx, t in enumerate(rsi_trades[:30]):  # Show first 30
            pts = t['pts']
            pnl = (pts - comm_pts) * DOLLAR_PER_PT
            et = pd.Timestamp(t['entry_time']).strftime('%Y-%m-%d %H:%M')
            print(f"  {idx+1:>3}  {t['side']:<5}  {t['entry']:>10.2f}  {t['exit']:>10.2f}  "
                  f"{pts:>+7.2f}  ${pnl:>+7.2f}  {t['bars']:>5}  {et}")

        # Summary of RSI exit trades
        if rsi_trades:
            rsi_sc = score_trades(rsi_trades, COMMISSION_PER_SIDE, DOLLAR_PER_PT)
            print(f"\n  RSI EXIT summary: {rsi_sc['count']} trades, "
                  f"WR {rsi_sc['win_rate']}%, PF {rsi_sc['pf']}, "
                  f"Net ${rsi_sc['net_dollar']:+.2f}, Avg {rsi_sc['avg_bars']:.0f} bars")

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)


if __name__ == "__main__":
    main()
