"""
RSI Trendline + SM Conviction Sweep
=====================================
Tests three SM-alignment modifications for the RSI TL runner strategy:

  Option 1: Drop to 1 contract on SM-opposed entries
  Option 2: Tighter SL on SM-opposed entries
  Option 3: Hard gate -- block SM-opposed entries entirely (pre-filter)

Baseline: RSI(8), TP1=7, TP2=20, SL=40, CD=30, CutOff=13:00, 2 contracts, no SM filter.

Usage:
    python3 rsi_tl_sm_conviction_sweep.py
"""

import warnings
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

# Add parent dirs so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi_trendline_backtest import (
    compute_rsi, generate_signals, compute_et_minutes,
    score_trades, fmt_score, split_is_oos,
    NY_OPEN_ET, NY_CLOSE_ET,
    COMMISSION_PER_SIDE, DOLLAR_PER_PT,
)
from v10_test_common import compute_smart_money, load_instrument_1min

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENTRY_END_ET = 13 * 60      # 13:00 ET cutoff
ENTRY_START_ET = 9 * 60 + 30  # 09:30 ET

# Baseline runner params
BASE_TP1 = 7
BASE_TP2 = 20
BASE_SL = 40
BASE_CD = 30
RSI_PERIOD = 8

# SM params
SM_INDEX = 10
SM_FLOW = 12
SM_NORM = 200
SM_EMA = 100


# ---------------------------------------------------------------------------
# Baseline Runner Backtest (copied from rsi_trendline_backtest for reference)
# ---------------------------------------------------------------------------
def run_baseline_runner(opens, closes, times, long_signal, short_signal,
                        cooldown_bars, max_loss_pts, tp1_pts, tp2_pts,
                        entry_start_et=ENTRY_START_ET, entry_end_et=ENTRY_END_ET,
                        eod_minutes_et=NY_CLOSE_ET):
    """Baseline runner: 2 contracts, TP1 scalp + TP2 runner, SL->BE after TP1."""
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    leg1_exited = False

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result, qty=1):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result, "qty": qty,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            qty = 1 if leg1_exited else 2
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD", qty)
            trade_state = 0
            exit_bar = i
            continue

        if trade_state == 1:
            if not leg1_exited:
                if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "SL", 2)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                if closes[i - 1] <= entry_price:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "BE", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

            if not leg1_exited and tp1_pts > 0 and closes[i - 1] >= entry_price + tp1_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP1", 1)
                leg1_exited = True

            if leg1_exited and tp2_pts > 0 and closes[i - 1] >= entry_price + tp2_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP2", 1)
                trade_state = 0
                exit_bar = i
                continue

        elif trade_state == -1:
            if not leg1_exited:
                if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "SL", 2)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                if closes[i - 1] >= entry_price:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "BE", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

            if not leg1_exited and tp1_pts > 0 and closes[i - 1] <= entry_price - tp1_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP1", 1)
                leg1_exited = True

            if leg1_exited and tp2_pts > 0 and closes[i - 1] <= entry_price - tp2_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP2", 1)
                trade_state = 0
                exit_bar = i
                continue

        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = entry_start_et <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                if long_signal[i - 1]:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    leg1_exited = False
                elif short_signal[i - 1]:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    leg1_exited = False

    return trades


# ---------------------------------------------------------------------------
# Option 1: Variable qty based on SM alignment
# ---------------------------------------------------------------------------
def run_option1_qty(opens, closes, times, long_signal, short_signal, sm,
                    cooldown_bars, max_loss_pts, tp1_pts, tp2_pts,
                    sm_threshold,
                    entry_start_et=ENTRY_START_ET, entry_end_et=ENTRY_END_ET,
                    eod_minutes_et=NY_CLOSE_ET):
    """
    SM-aligned entries: 2 contracts (TP1=7 + TP2=20, SL->BE runner)
    SM-opposed entries: 1 contract (TP=20, SL=40, single exit)

    SM opposed = SM > sm_threshold for short entries, SM < -sm_threshold for long entries
    """
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    leg1_exited = False
    is_runner = False  # True = 2 contracts (runner mode), False = 1 contract
    opposed_count = 0
    aligned_count = 0

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result, qty=1):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result, "qty": qty,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        # EOD
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            if is_runner:
                qty = 1 if leg1_exited else 2
            else:
                qty = 1
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD", qty)
            trade_state = 0
            exit_bar = i
            continue

        # Exits for long
        if trade_state == 1:
            if is_runner:
                # Runner mode (2 contracts)
                if not leg1_exited:
                    if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                        close_trade("long", entry_price, opens[i], entry_idx, i, "SL", 2)
                        trade_state = 0
                        exit_bar = i
                        continue
                else:
                    if closes[i - 1] <= entry_price:
                        close_trade("long", entry_price, opens[i], entry_idx, i, "BE", 1)
                        trade_state = 0
                        exit_bar = i
                        continue

                if not leg1_exited and tp1_pts > 0 and closes[i - 1] >= entry_price + tp1_pts:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "TP1", 1)
                    leg1_exited = True

                if leg1_exited and tp2_pts > 0 and closes[i - 1] >= entry_price + tp2_pts:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "TP2", 1)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                # Single contract mode (SM opposed)
                if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "SL", 1)
                    trade_state = 0
                    exit_bar = i
                    continue
                if tp2_pts > 0 and closes[i - 1] >= entry_price + tp2_pts:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "TP", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

        elif trade_state == -1:
            if is_runner:
                if not leg1_exited:
                    if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                        close_trade("short", entry_price, opens[i], entry_idx, i, "SL", 2)
                        trade_state = 0
                        exit_bar = i
                        continue
                else:
                    if closes[i - 1] >= entry_price:
                        close_trade("short", entry_price, opens[i], entry_idx, i, "BE", 1)
                        trade_state = 0
                        exit_bar = i
                        continue

                if not leg1_exited and tp1_pts > 0 and closes[i - 1] <= entry_price - tp1_pts:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "TP1", 1)
                    leg1_exited = True

                if leg1_exited and tp2_pts > 0 and closes[i - 1] <= entry_price - tp2_pts:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "TP2", 1)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "SL", 1)
                    trade_state = 0
                    exit_bar = i
                    continue
                if tp2_pts > 0 and closes[i - 1] <= entry_price - tp2_pts:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "TP", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

        # Entries
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = entry_start_et <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                signal_dir = 0
                if long_signal[i - 1]:
                    signal_dir = 1
                elif short_signal[i - 1]:
                    signal_dir = -1

                if signal_dir != 0:
                    sm_val = sm[i - 1]  # bar[i-1] convention

                    # Determine alignment
                    if signal_dir == 1:
                        # Long entry: opposed if SM < -threshold
                        sm_opposed = sm_val < -sm_threshold
                    else:
                        # Short entry: opposed if SM > threshold
                        sm_opposed = sm_val > sm_threshold

                    if sm_opposed:
                        opposed_count += 1
                        is_runner = False  # 1 contract
                    else:
                        aligned_count += 1
                        is_runner = True   # 2 contracts

                    trade_state = signal_dir
                    entry_price = opens[i]
                    entry_idx = i
                    leg1_exited = False

    return trades, aligned_count, opposed_count


# ---------------------------------------------------------------------------
# Option 2: Tighter SL on SM-opposed
# ---------------------------------------------------------------------------
def run_option2_tighter_sl(opens, closes, times, long_signal, short_signal, sm,
                           cooldown_bars, max_loss_pts, tp1_pts, tp2_pts,
                           opposed_sl_pts, sm_threshold,
                           entry_start_et=ENTRY_START_ET, entry_end_et=ENTRY_END_ET,
                           eod_minutes_et=NY_CLOSE_ET):
    """
    Always 2 contracts.
    SM-aligned: SL = max_loss_pts (40)
    SM-opposed: SL = opposed_sl_pts (tighter)
    """
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    leg1_exited = False
    active_sl = max_loss_pts
    opposed_count = 0
    aligned_count = 0

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result, qty=1):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result, "qty": qty,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            qty = 1 if leg1_exited else 2
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD", qty)
            trade_state = 0
            exit_bar = i
            continue

        if trade_state == 1:
            if not leg1_exited:
                if active_sl > 0 and closes[i - 1] <= entry_price - active_sl:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "SL", 2)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                if closes[i - 1] <= entry_price:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "BE", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

            if not leg1_exited and tp1_pts > 0 and closes[i - 1] >= entry_price + tp1_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP1", 1)
                leg1_exited = True

            if leg1_exited and tp2_pts > 0 and closes[i - 1] >= entry_price + tp2_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP2", 1)
                trade_state = 0
                exit_bar = i
                continue

        elif trade_state == -1:
            if not leg1_exited:
                if active_sl > 0 and closes[i - 1] >= entry_price + active_sl:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "SL", 2)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                if closes[i - 1] >= entry_price:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "BE", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

            if not leg1_exited and tp1_pts > 0 and closes[i - 1] <= entry_price - tp1_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP1", 1)
                leg1_exited = True

            if leg1_exited and tp2_pts > 0 and closes[i - 1] <= entry_price - tp2_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP2", 1)
                trade_state = 0
                exit_bar = i
                continue

        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = entry_start_et <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                signal_dir = 0
                if long_signal[i - 1]:
                    signal_dir = 1
                elif short_signal[i - 1]:
                    signal_dir = -1

                if signal_dir != 0:
                    sm_val = sm[i - 1]

                    if signal_dir == 1:
                        sm_opposed = sm_val < -sm_threshold
                    else:
                        sm_opposed = sm_val > sm_threshold

                    if sm_opposed:
                        opposed_count += 1
                        active_sl = opposed_sl_pts
                    else:
                        aligned_count += 1
                        active_sl = max_loss_pts

                    trade_state = signal_dir
                    entry_price = opens[i]
                    entry_idx = i
                    leg1_exited = False

    return trades, aligned_count, opposed_count


# ---------------------------------------------------------------------------
# Option 3: Hard gate (pre-filter -- block SM-opposed entries)
# ---------------------------------------------------------------------------
def run_option3_hard_gate(opens, closes, times, long_signal, short_signal, sm,
                          cooldown_bars, max_loss_pts, tp1_pts, tp2_pts,
                          sm_threshold,
                          entry_start_et=ENTRY_START_ET, entry_end_et=ENTRY_END_ET,
                          eod_minutes_et=NY_CLOSE_ET):
    """
    Pre-filter: block entries when SM opposes. Cooldown still starts from
    the BLOCKED signal bar (pre-filter, not post-filter).
    """
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    leg1_exited = False
    blocked_count = 0
    passed_count = 0

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result, qty=1):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result, "qty": qty,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            qty = 1 if leg1_exited else 2
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD", qty)
            trade_state = 0
            exit_bar = i
            continue

        if trade_state == 1:
            if not leg1_exited:
                if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "SL", 2)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                if closes[i - 1] <= entry_price:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "BE", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

            if not leg1_exited and tp1_pts > 0 and closes[i - 1] >= entry_price + tp1_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP1", 1)
                leg1_exited = True

            if leg1_exited and tp2_pts > 0 and closes[i - 1] >= entry_price + tp2_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP2", 1)
                trade_state = 0
                exit_bar = i
                continue

        elif trade_state == -1:
            if not leg1_exited:
                if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "SL", 2)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                if closes[i - 1] >= entry_price:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "BE", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

            if not leg1_exited and tp1_pts > 0 and closes[i - 1] <= entry_price - tp1_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP1", 1)
                leg1_exited = True

            if leg1_exited and tp2_pts > 0 and closes[i - 1] <= entry_price - tp2_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP2", 1)
                trade_state = 0
                exit_bar = i
                continue

        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = entry_start_et <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                signal_dir = 0
                if long_signal[i - 1]:
                    signal_dir = 1
                elif short_signal[i - 1]:
                    signal_dir = -1

                if signal_dir != 0:
                    sm_val = sm[i - 1]

                    if signal_dir == 1:
                        sm_opposed = sm_val < -sm_threshold
                    else:
                        sm_opposed = sm_val > sm_threshold

                    if sm_opposed:
                        blocked_count += 1
                        # PRE-FILTER: cooldown starts from blocked signal bar
                        exit_bar = i
                        continue  # skip entry entirely
                    else:
                        passed_count += 1

                    trade_state = signal_dir
                    entry_price = opens[i]
                    entry_idx = i
                    leg1_exited = False

    return trades, passed_count, blocked_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_on_slice(df, run_fn, *args, **kwargs):
    """Run a backtest function on a dataframe slice."""
    opens = df['Open'].values
    closes = df['Close'].values
    times = df.index
    volumes = df['Volume'].values

    rsi = compute_rsi(closes, RSI_PERIOD)
    long_sig, short_sig = generate_signals(rsi)
    sm = compute_smart_money(closes, volumes, SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)

    return run_fn(opens, closes, times, long_sig, short_sig, sm, *args, **kwargs)


def run_baseline_on_slice(df):
    """Run baseline runner on a dataframe slice."""
    opens = df['Open'].values
    closes = df['Close'].values
    times = df.index

    rsi = compute_rsi(closes, RSI_PERIOD)
    long_sig, short_sig = generate_signals(rsi)

    trades = run_baseline_runner(opens, closes, times, long_sig, short_sig,
                                  cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
                                  tp1_pts=BASE_TP1, tp2_pts=BASE_TP2)
    return trades


def print_table_header(title, extra_cols=""):
    print(f"\n{'=' * 100}")
    print(title)
    print('=' * 100)
    hdr = (f"{'Config':>30s}  {'Split':>5s}  {'Trades':>6s}  {'WR%':>5s}  {'PF':>6s}  "
           f"{'Net$':>9s}  {'MaxDD$':>8s}  {'Sharpe':>7s}")
    if extra_cols:
        hdr += f"  {extra_cols}"
    print(hdr)
    print('-' * (100 + len(extra_cols) + 2 if extra_cols else 100))


def print_row(label, split, sc, extra=""):
    if sc is None:
        print(f"{label:>30s}  {split:>5s}  {'N/A':>6s}")
        return
    line = (f"{label:>30s}  {split:>5s}  {sc['count']:>6d}  {sc['win_rate']:>5.1f}  "
            f"{sc['pf']:>6.3f}  ${sc['net_dollar']:>8.2f}  ${sc['max_dd_dollar']:>7.2f}  "
            f"{sc['sharpe']:>7.3f}")
    if extra:
        line += f"  {extra}"
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    print("=" * 100)
    print("RSI TRENDLINE + SM CONVICTION SWEEP")
    print("=" * 100)
    print(f"Baseline: RSI({RSI_PERIOD}), TP1={BASE_TP1}, TP2={BASE_TP2}, "
          f"SL={BASE_SL}, CD={BASE_CD}, CutOff=13:00, 2 contracts")
    print(f"SM params: index={SM_INDEX}, flow={SM_FLOW}, norm={SM_NORM}, ema={SM_EMA}")

    # ----- Load Data -----
    print("\nLoading data...")
    df = load_instrument_1min('MNQ')
    print(f"Loaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")

    # IS/OOS split
    df_is, df_oos = split_is_oos(df)
    print(f"IS:  {len(df_is)} bars  {df_is.index[0]} to {df_is.index[-1]}")
    print(f"OOS: {len(df_oos)} bars  {df_oos.index[0]} to {df_oos.index[-1]}")

    # Pre-compute signals on full dataset for SM distribution check
    closes = df['Close'].values
    volumes = df['Volume'].values
    sm_full = compute_smart_money(closes, volumes, SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)
    rsi_full = compute_rsi(closes, RSI_PERIOD)
    long_sig_full, short_sig_full = generate_signals(rsi_full)

    # Quick SM distribution
    signal_bars = np.where(long_sig_full | short_sig_full)[0]
    if len(signal_bars) > 0:
        sm_at_signals = sm_full[np.clip(signal_bars - 1, 0, len(sm_full) - 1)]
        print(f"\nSM at signal bars: mean={sm_at_signals.mean():.4f}, "
              f"std={sm_at_signals.std():.4f}, "
              f"min={sm_at_signals.min():.4f}, max={sm_at_signals.max():.4f}")
        for thresh in [0.0, 0.10, 0.25]:
            long_bars = np.where(long_sig_full)[0]
            short_bars = np.where(short_sig_full)[0]
            long_opposed = np.sum(sm_full[np.clip(long_bars - 1, 0, len(sm_full) - 1)] < -thresh) if len(long_bars) > 0 else 0
            short_opposed = np.sum(sm_full[np.clip(short_bars - 1, 0, len(sm_full) - 1)] > thresh) if len(short_bars) > 0 else 0
            total_signals = len(long_bars) + len(short_bars)
            total_opposed = long_opposed + short_opposed
            print(f"  SM_T={thresh:.2f}: {total_opposed}/{total_signals} signals opposed "
                  f"({100*total_opposed/total_signals:.1f}%)")

    # ====================================================================
    # BASELINE
    # ====================================================================
    print_table_header("BASELINE (no SM filter)")

    datasets = [("FULL", df), ("IS", df_is), ("OOS", df_oos)]
    baseline_scores = {}
    for label, sub_df in datasets:
        trades = run_baseline_on_slice(sub_df)
        sc = score_trades(trades)
        baseline_scores[label] = sc
        print_row("Baseline", label, sc)

    # ====================================================================
    # OPTION 1: Variable qty
    # ====================================================================
    sm_thresholds = [0.0, 0.10, 0.25]

    print_table_header("OPTION 1: Drop to 1 contract on SM-opposed",
                       "Aligned  Opposed  %Opp")

    opt1_results = []
    for sm_t in sm_thresholds:
        config_label = f"Opt1 SM_T={sm_t:.2f}"

        for label, sub_df in datasets:
            trades, aligned, opposed = run_on_slice(
                sub_df, run_option1_qty,
                cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
                tp1_pts=BASE_TP1, tp2_pts=BASE_TP2,
                sm_threshold=sm_t,
            )
            sc = score_trades(trades)
            total = aligned + opposed
            pct = 100 * opposed / total if total > 0 else 0
            extra = f"{aligned:>7d}  {opposed:>7d}  {pct:>4.1f}%"
            print_row(config_label, label, sc, extra)

            if label == "OOS":
                opt1_results.append({
                    'config': config_label, 'sm_t': sm_t,
                    'score_full': None, 'score_oos': sc,
                    'opposed_pct': pct,
                })
            elif label == "FULL":
                if opt1_results and opt1_results[-1].get('score_full') is None:
                    pass  # will be set next iteration
                # Store for later
                opt1_results.append({'_full_temp': sc, 'sm_t': sm_t})

        print()  # blank line between SM thresholds

    # Re-collect option 1 results properly
    opt1_results_clean = []
    for sm_t in sm_thresholds:
        full_trades, full_al, full_op = run_on_slice(
            df, run_option1_qty,
            cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
            tp1_pts=BASE_TP1, tp2_pts=BASE_TP2, sm_threshold=sm_t)
        full_sc = score_trades(full_trades)

        oos_trades, oos_al, oos_op = run_on_slice(
            df_oos, run_option1_qty,
            cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
            tp1_pts=BASE_TP1, tp2_pts=BASE_TP2, sm_threshold=sm_t)
        oos_sc = score_trades(oos_trades)

        total = full_al + full_op
        pct = 100 * full_op / total if total > 0 else 0
        opt1_results_clean.append({
            'config': f"Opt1 SM_T={sm_t:.2f}",
            'sm_t': sm_t,
            'score_full': full_sc,
            'score_oos': oos_sc,
            'opposed_pct': pct,
        })

    # ====================================================================
    # OPTION 2: Tighter SL on SM-opposed
    # ====================================================================
    opposed_sl_list = [15, 20, 25, 30]

    print_table_header("OPTION 2: Tighter SL on SM-opposed entries",
                       "Aligned  Opposed  %Opp  OppSL")

    opt2_results_clean = []
    for sm_t in sm_thresholds:
        for opp_sl in opposed_sl_list:
            config_label = f"Opt2 SM_T={sm_t:.2f} OppSL={opp_sl}"

            for label, sub_df in datasets:
                trades, aligned, opposed = run_on_slice(
                    sub_df, run_option2_tighter_sl,
                    cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
                    tp1_pts=BASE_TP1, tp2_pts=BASE_TP2,
                    opposed_sl_pts=opp_sl, sm_threshold=sm_t,
                )
                sc = score_trades(trades)
                total = aligned + opposed
                pct = 100 * opposed / total if total > 0 else 0
                extra = f"{aligned:>7d}  {opposed:>7d}  {pct:>4.1f}%  {opp_sl:>5d}"
                print_row(config_label, label, sc, extra)

            print()

            # Collect for summary
            full_trades, full_al, full_op = run_on_slice(
                df, run_option2_tighter_sl,
                cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
                tp1_pts=BASE_TP1, tp2_pts=BASE_TP2,
                opposed_sl_pts=opp_sl, sm_threshold=sm_t)
            full_sc = score_trades(full_trades)

            oos_trades, oos_al, oos_op = run_on_slice(
                df_oos, run_option2_tighter_sl,
                cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
                tp1_pts=BASE_TP1, tp2_pts=BASE_TP2,
                opposed_sl_pts=opp_sl, sm_threshold=sm_t)
            oos_sc = score_trades(oos_trades)

            total = full_al + full_op
            pct = 100 * full_op / total if total > 0 else 0
            opt2_results_clean.append({
                'config': config_label,
                'sm_t': sm_t, 'opp_sl': opp_sl,
                'score_full': full_sc,
                'score_oos': oos_sc,
                'opposed_pct': pct,
            })

    # ====================================================================
    # OPTION 3: Hard gate (block SM-opposed)
    # ====================================================================
    print_table_header("OPTION 3: Hard gate -- block SM-opposed entries (pre-filter)",
                       "Passed  Blocked  %Blk")

    opt3_results_clean = []
    for sm_t in sm_thresholds:
        config_label = f"Opt3 SM_T={sm_t:.2f}"

        for label, sub_df in datasets:
            trades, passed, blocked = run_on_slice(
                sub_df, run_option3_hard_gate,
                cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
                tp1_pts=BASE_TP1, tp2_pts=BASE_TP2,
                sm_threshold=sm_t,
            )
            sc = score_trades(trades)
            total = passed + blocked
            pct = 100 * blocked / total if total > 0 else 0
            extra = f"{passed:>6d}  {blocked:>7d}  {pct:>4.1f}%"
            print_row(config_label, label, sc, extra)

        print()

        # Collect for summary
        full_trades, full_pa, full_bl = run_on_slice(
            df, run_option3_hard_gate,
            cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
            tp1_pts=BASE_TP1, tp2_pts=BASE_TP2, sm_threshold=sm_t)
        full_sc = score_trades(full_trades)

        oos_trades, oos_pa, oos_bl = run_on_slice(
            df_oos, run_option3_hard_gate,
            cooldown_bars=BASE_CD, max_loss_pts=BASE_SL,
            tp1_pts=BASE_TP1, tp2_pts=BASE_TP2, sm_threshold=sm_t)
        oos_sc = score_trades(oos_trades)

        total = full_pa + full_bl
        pct = 100 * full_bl / total if total > 0 else 0
        opt3_results_clean.append({
            'config': config_label,
            'sm_t': sm_t,
            'score_full': full_sc,
            'score_oos': oos_sc,
            'blocked_pct': pct,
        })

    # ====================================================================
    # SUMMARY TABLE
    # ====================================================================
    print(f"\n{'=' * 100}")
    print("SUMMARY: ALL OPTIONS vs BASELINE (ranked by OOS PF improvement)")
    print('=' * 100)

    baseline_oos_pf = baseline_scores['OOS']['pf'] if baseline_scores['OOS'] else 0
    baseline_oos_sharpe = baseline_scores['OOS']['sharpe'] if baseline_scores['OOS'] else 0
    baseline_full_pf = baseline_scores['FULL']['pf'] if baseline_scores['FULL'] else 0

    print(f"\nBaseline OOS: PF={baseline_oos_pf:.3f}, Sharpe={baseline_oos_sharpe:.3f}")
    print(f"Baseline FULL: PF={baseline_full_pf:.3f}")

    all_configs = []

    for r in opt1_results_clean:
        oos_pf = r['score_oos']['pf'] if r['score_oos'] else 0
        oos_sharpe = r['score_oos']['sharpe'] if r['score_oos'] else 0
        full_pf = r['score_full']['pf'] if r['score_full'] else 0
        oos_net = r['score_oos']['net_dollar'] if r['score_oos'] else 0
        oos_trades = r['score_oos']['count'] if r['score_oos'] else 0
        oos_wr = r['score_oos']['win_rate'] if r['score_oos'] else 0
        oos_dd = r['score_oos']['max_dd_dollar'] if r['score_oos'] else 0
        all_configs.append({
            'option': 'Opt1',
            'config': r['config'],
            'oos_pf': oos_pf,
            'oos_sharpe': oos_sharpe,
            'full_pf': full_pf,
            'oos_net': oos_net,
            'oos_trades': oos_trades,
            'oos_wr': oos_wr,
            'oos_dd': oos_dd,
            'pf_delta': oos_pf - baseline_oos_pf,
            'affected_pct': r['opposed_pct'],
        })

    for r in opt2_results_clean:
        oos_pf = r['score_oos']['pf'] if r['score_oos'] else 0
        oos_sharpe = r['score_oos']['sharpe'] if r['score_oos'] else 0
        full_pf = r['score_full']['pf'] if r['score_full'] else 0
        oos_net = r['score_oos']['net_dollar'] if r['score_oos'] else 0
        oos_trades = r['score_oos']['count'] if r['score_oos'] else 0
        oos_wr = r['score_oos']['win_rate'] if r['score_oos'] else 0
        oos_dd = r['score_oos']['max_dd_dollar'] if r['score_oos'] else 0
        all_configs.append({
            'option': 'Opt2',
            'config': r['config'],
            'oos_pf': oos_pf,
            'oos_sharpe': oos_sharpe,
            'full_pf': full_pf,
            'oos_net': oos_net,
            'oos_trades': oos_trades,
            'oos_wr': oos_wr,
            'oos_dd': oos_dd,
            'pf_delta': oos_pf - baseline_oos_pf,
            'affected_pct': r['opposed_pct'],
        })

    for r in opt3_results_clean:
        oos_pf = r['score_oos']['pf'] if r['score_oos'] else 0
        oos_sharpe = r['score_oos']['sharpe'] if r['score_oos'] else 0
        full_pf = r['score_full']['pf'] if r['score_full'] else 0
        oos_net = r['score_oos']['net_dollar'] if r['score_oos'] else 0
        oos_trades = r['score_oos']['count'] if r['score_oos'] else 0
        oos_wr = r['score_oos']['win_rate'] if r['score_oos'] else 0
        oos_dd = r['score_oos']['max_dd_dollar'] if r['score_oos'] else 0
        all_configs.append({
            'option': 'Opt3',
            'config': r['config'],
            'oos_pf': oos_pf,
            'oos_sharpe': oos_sharpe,
            'full_pf': full_pf,
            'oos_net': oos_net,
            'oos_trades': oos_trades,
            'oos_wr': oos_wr,
            'oos_dd': oos_dd,
            'pf_delta': oos_pf - baseline_oos_pf,
            'affected_pct': r.get('blocked_pct', 0),
        })

    # Sort by OOS PF delta
    all_configs.sort(key=lambda x: x['pf_delta'], reverse=True)

    print(f"\n{'#':>3}  {'Option':>5}  {'Config':>35s}  {'OOS PF':>7s}  {'dPF':>7s}  "
          f"{'OOS Sharpe':>10s}  {'OOS Net$':>9s}  {'OOS Trades':>10s}  "
          f"{'OOS WR%':>7s}  {'OOS DD$':>8s}  {'%Affected':>9s}")
    print('-' * 130)

    for rank, c in enumerate(all_configs, 1):
        print(f"{rank:3d}  {c['option']:>5s}  {c['config']:>35s}  "
              f"{c['oos_pf']:>7.3f}  {c['pf_delta']:>+7.3f}  "
              f"{c['oos_sharpe']:>10.3f}  ${c['oos_net']:>8.2f}  "
              f"{c['oos_trades']:>10d}  {c['oos_wr']:>7.1f}  "
              f"${c['oos_dd']:>7.2f}  {c['affected_pct']:>8.1f}%")

    # Best from each option
    print(f"\n{'=' * 100}")
    print("BEST CONFIG FROM EACH OPTION (by OOS PF)")
    print('=' * 100)
    for opt in ['Opt1', 'Opt2', 'Opt3']:
        opt_configs = [c for c in all_configs if c['option'] == opt]
        if opt_configs:
            best = max(opt_configs, key=lambda x: x['oos_pf'])
            print(f"\n  {opt}: {best['config']}")
            print(f"    OOS PF: {best['oos_pf']:.3f} (baseline {baseline_oos_pf:.3f}, "
                  f"delta {best['pf_delta']:+.3f})")
            print(f"    OOS Sharpe: {best['oos_sharpe']:.3f} (baseline {baseline_oos_sharpe:.3f})")
            print(f"    OOS Net$: ${best['oos_net']:.2f}")
            print(f"    OOS Trades: {best['oos_trades']}")
            print(f"    OOS WR%: {best['oos_wr']:.1f}")
            print(f"    OOS MaxDD$: ${best['oos_dd']:.2f}")
            print(f"    Entries affected: {best['affected_pct']:.1f}%")

    # RECOMMENDATION
    print(f"\n{'=' * 100}")
    print("RECOMMENDATION")
    print('=' * 100)

    # Find overall best by OOS PF that also has positive full PF delta
    viable = [c for c in all_configs if c['pf_delta'] > 0]
    if viable:
        best_overall = max(viable, key=lambda x: x['oos_pf'])
        print(f"\n  BEST OVERALL: {best_overall['config']}")
        print(f"    OOS PF improvement: {best_overall['pf_delta']:+.3f} "
              f"({baseline_oos_pf:.3f} -> {best_overall['oos_pf']:.3f})")
        print(f"    OOS Sharpe: {best_overall['oos_sharpe']:.3f}")
        print(f"    OOS Net$: ${best_overall['oos_net']:.2f}")
        print(f"    Entries affected: {best_overall['affected_pct']:.1f}%")
    else:
        print("\n  No config improved OOS PF over baseline.")
        print("  SM conviction filter may not add value to RSI TL entries.")
        # Still show which degraded least
        if all_configs:
            least_bad = max(all_configs, key=lambda x: x['oos_pf'])
            print(f"\n  Least degradation: {least_bad['config']}")
            print(f"    OOS PF: {least_bad['oos_pf']:.3f} (delta {least_bad['pf_delta']:+.3f})")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
