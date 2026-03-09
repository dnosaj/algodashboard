"""
vScalpC Partial Exit Sweep — Phase 1: Base Partial Exit Structure
=================================================================
V15-style entries on MNQ with 2-contract partial exit (TP1 scalp + TP2 runner).

Entry: identical to V15 (SM(10/12/200/100) SM_T=0.0 RSI(8/60/40) CD=20,
       entry cutoff 13:00 ET).
Exit:  2 contracts entered simultaneously.
       Leg 1 (scalp): TP1 or SL
       Leg 2 (runner): TP2 or SL, optional SL-to-BE after TP1, optional BE_TIME

Sweep:
  TP1: [5, 7]
  TP2: [12, 15, 20, 25, 30]
  SL:  [30, 35, 40, 50]
  BE_TIME: [0, 45, 60, 75, 90]
  SL_to_BE: [True, False]

400 combos total.

Usage:
    cd backtesting_engine && python3 strategies/vscalpc_partial_exit_sweep.py
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

# --- Path setup ---
sys.path.insert(0, 'strategies')
sys.path.insert(0, str(Path('strategies').resolve().parent))
sys.path.insert(0, str(Path('.').resolve().parent / 'live_trading'))

from v10_test_common import (
    compute_smart_money,
    compute_rsi,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
    compute_et_minutes,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)

from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_ENTRY_END_ET,
    VSCALPB_RSI_LEN, VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
    VSCALPB_SM_THRESHOLD, VSCALPB_COOLDOWN,
    VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
)


# ============================================================================
# Partial Exit Backtest Loop
# ============================================================================

def run_backtest_partial_exit(opens, highs, lows, closes, sm, times,
                              rsi_5m_curr, rsi_5m_prev,
                              rsi_buy, rsi_sell, sm_threshold,
                              cooldown_bars,
                              sl_pts, tp1_pts, tp2_pts,
                              sl_to_be_after_tp1=False,
                              be_time_bars=0,
                              entry_end_et=VSCALPA_ENTRY_END_ET,
                              eod_minutes_et=NY_CLOSE_ET):
    """Backtest with 2-contract partial exit.

    Entry logic: identical to run_backtest_tp_exit (V15 params).

    Each trade enters 2 contracts. Two legs track independently:
      Leg 1 (scalp): exits at TP1 or SL or EOD
      Leg 2 (runner): exits at TP2 or SL or EOD, optionally SL->BE after TP1

    Exit convention (matches production):
      - EOD: bar_mins >= eod_minutes_et -> fill at bar close
      - SL/TP: prev bar close breaches level -> fill at next open
      - BE_TIME: (i-1) - entry_idx >= be_time_bars -> fill at next open (runner only)

    Returns list of composite trade dicts with combined P&L for both legs.
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    # Leg tracking
    leg1_active = False  # scalp leg
    leg2_active = False  # runner leg
    leg1_exit_price = 0.0
    leg1_exit_idx = 0
    leg1_exit_reason = ""
    leg1_pts = 0.0
    runner_sl_pts = 0.0  # can change to 0 (BE) after TP1

    et_mins = compute_et_minutes(times)

    def finalize_trade(side, entry_p, entry_i, last_exit_i):
        """Record the composite 2-leg trade."""
        # Total pts is sum of both legs
        total_pts = leg1_pts + (leg2_exit_price - entry_p if side == "long"
                                else entry_p - leg2_exit_price)
        # Leg 2 result
        leg2_pts_val = (leg2_exit_price - entry_p) if side == "long" else (entry_p - leg2_exit_price)

        trades.append({
            "side": side,
            "entry": entry_p,
            "exit": leg2_exit_price,  # last exit price (runner)
            "pts": total_pts,         # combined pts for both contracts
            "entry_time": times[entry_i],
            "exit_time": times[last_exit_i],
            "entry_idx": entry_i,
            "exit_idx": last_exit_i,
            "bars": last_exit_i - entry_i,
            "result": f"{leg1_exit_reason}+{leg2_exit_reason}",
            # Extra fields for analysis
            "leg1_pts": leg1_pts,
            "leg2_pts": leg2_pts_val,
            "leg1_exit_reason": leg1_exit_reason,
            "leg2_exit_reason": leg2_exit_reason,
            "leg1_exit_idx": leg1_exit_idx,
            "leg2_exit_idx": last_exit_i,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI cross from mapped 5-min
        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        # Episode reset
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # --- EOD close (close all remaining legs at bar close) ---
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            eod_price = closes[i]

            if leg1_active:
                leg1_pts = (eod_price - entry_price) if side == "long" else (entry_price - eod_price)
                leg1_exit_price = eod_price
                leg1_exit_idx = i
                leg1_exit_reason = "EOD"
                leg1_active = False

            if leg2_active:
                leg2_exit_price = eod_price
                leg2_exit_reason = "EOD"
                leg2_active = False

            if not leg1_active and not leg2_active:
                finalize_trade(side, entry_price, entry_idx, i)
                trade_state = 0
                exit_bar = i
            continue

        # --- Exits for LONG positions ---
        if trade_state == 1:
            # Check Leg 1 (scalp) exits
            if leg1_active:
                # SL: prev bar close breaches stop
                if closes[i - 1] <= entry_price - sl_pts:
                    leg1_pts = opens[i] - entry_price
                    leg1_exit_price = opens[i]
                    leg1_exit_idx = i
                    leg1_exit_reason = "SL"
                    leg1_active = False
                # TP1: prev bar close reached TP1
                elif closes[i - 1] >= entry_price + tp1_pts:
                    leg1_pts = opens[i] - entry_price
                    leg1_exit_price = opens[i]
                    leg1_exit_idx = i
                    leg1_exit_reason = "TP1"
                    leg1_active = False
                    # Move runner SL to breakeven if configured
                    if sl_to_be_after_tp1 and leg2_active:
                        runner_sl_pts = 0.0  # BE = entry price

            # Check Leg 2 (runner) exits
            if leg2_active:
                # SL: prev bar close breaches runner stop
                effective_sl = runner_sl_pts
                if effective_sl > 0:
                    sl_hit = closes[i - 1] <= entry_price - effective_sl
                elif effective_sl == 0.0 and sl_to_be_after_tp1 and not leg1_active and leg1_exit_reason == "TP1":
                    # BE stop: close breaches entry price
                    sl_hit = closes[i - 1] <= entry_price
                else:
                    sl_hit = closes[i - 1] <= entry_price - sl_pts

                if sl_hit:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "SL" if runner_sl_pts == sl_pts else "BE"
                    leg2_active = False
                # TP2: prev bar close reached TP2
                elif closes[i - 1] >= entry_price + tp2_pts:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "TP2"
                    leg2_active = False
                # BE_TIME: runner held too long
                elif be_time_bars > 0 and not leg1_active:
                    bars_since_entry = (i - 1) - entry_idx
                    if bars_since_entry >= be_time_bars:
                        leg2_exit_price = opens[i]
                        leg2_exit_reason = "BE_TIME"
                        leg2_active = False

            # If both legs closed on same bar, or one was already closed
            if not leg1_active and not leg2_active:
                finalize_trade("long", entry_price, entry_idx, i)
                trade_state = 0
                exit_bar = i
                continue

            # If leg1 closed but leg2 still open (or vice versa), continue
            # (leg2 will keep running until it exits)

        # --- Exits for SHORT positions ---
        elif trade_state == -1:
            if leg1_active:
                # SL
                if closes[i - 1] >= entry_price + sl_pts:
                    leg1_pts = entry_price - opens[i]
                    leg1_exit_price = opens[i]
                    leg1_exit_idx = i
                    leg1_exit_reason = "SL"
                    leg1_active = False
                # TP1
                elif closes[i - 1] <= entry_price - tp1_pts:
                    leg1_pts = entry_price - opens[i]
                    leg1_exit_price = opens[i]
                    leg1_exit_idx = i
                    leg1_exit_reason = "TP1"
                    leg1_active = False
                    if sl_to_be_after_tp1 and leg2_active:
                        runner_sl_pts = 0.0

            if leg2_active:
                effective_sl = runner_sl_pts
                if effective_sl > 0:
                    sl_hit = closes[i - 1] >= entry_price + effective_sl
                elif effective_sl == 0.0 and sl_to_be_after_tp1 and not leg1_active and leg1_exit_reason == "TP1":
                    sl_hit = closes[i - 1] >= entry_price
                else:
                    sl_hit = closes[i - 1] >= entry_price + sl_pts

                if sl_hit:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "SL" if runner_sl_pts == sl_pts else "BE"
                    leg2_active = False
                elif closes[i - 1] <= entry_price - tp2_pts:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "TP2"
                    leg2_active = False
                elif be_time_bars > 0 and not leg1_active:
                    bars_since_entry = (i - 1) - entry_idx
                    if bars_since_entry >= be_time_bars:
                        leg2_exit_price = opens[i]
                        leg2_exit_reason = "BE_TIME"
                        leg2_active = False

            if not leg1_active and not leg2_active:
                finalize_trade("short", entry_price, entry_idx, i)
                trade_state = 0
                exit_bar = i
                continue

        # --- Entries ---
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    leg1_active = True
                    leg2_active = True
                    runner_sl_pts = sl_pts  # initial SL for runner
                    leg1_pts = 0.0
                    leg1_exit_reason = ""
                    leg2_exit_price = 0.0
                    leg2_exit_reason = ""
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    leg1_active = True
                    leg2_active = True
                    runner_sl_pts = sl_pts
                    leg1_pts = 0.0
                    leg1_exit_reason = ""
                    leg2_exit_price = 0.0
                    leg2_exit_reason = ""

    return trades


# ============================================================================
# Scoring for partial exit trades (2 contracts)
# ============================================================================

def score_partial_trades(trades, dollar_per_pt=MNQ_DOLLAR_PER_PT,
                         commission_per_side=MNQ_COMMISSION):
    """Score partial exit trades. Each trade is 2 contracts.

    P&L per trade:
      leg1_pnl = leg1_pts * dollar_per_pt - 2 * commission (entry + exit)
      leg2_pnl = leg2_pts * dollar_per_pt - 2 * commission
      total_pnl = leg1_pnl + leg2_pnl

    Returns dict with aggregate metrics.
    """
    if not trades:
        return None

    comm_per_leg = commission_per_side * 2  # entry + exit per contract

    pnl_list = []
    for t in trades:
        leg1_pnl = t["leg1_pts"] * dollar_per_pt - comm_per_leg
        leg2_pnl = t["leg2_pts"] * dollar_per_pt - comm_per_leg
        pnl_list.append(leg1_pnl + leg2_pnl)

    pnl_arr = np.array(pnl_list)
    n = len(pnl_arr)
    total_pnl = pnl_arr.sum()
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100

    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()

    avg_bars = np.mean([t["bars"] for t in trades])

    # Sharpe
    sharpe = 0.0
    if n > 1 and np.std(pnl_arr) > 0:
        sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)

    # Exit type breakdown
    leg1_exits = {}
    leg2_exits = {}
    for t in trades:
        r1 = t.get("leg1_exit_reason", "?")
        r2 = t.get("leg2_exit_reason", "?")
        leg1_exits[r1] = leg1_exits.get(r1, 0) + 1
        leg2_exits[r2] = leg2_exits.get(r2, 0) + 1

    return {
        "count": n,
        "net_dollar": round(total_pnl, 2),
        "pf": round(pf, 3),
        "win_rate": round(wr, 1),
        "max_dd_dollar": round(mdd, 2),
        "avg_bars": round(avg_bars, 1),
        "sharpe": round(sharpe, 3),
        "leg1_exits": leg1_exits,
        "leg2_exits": leg2_exits,
        "pnl_array": pnl_arr,  # keep for portfolio analysis
    }


def daily_pnl_series(trades, dollar_per_pt=MNQ_DOLLAR_PER_PT,
                     commission_per_side=MNQ_COMMISSION):
    """Compute daily P&L series from trade list. Returns pd.Series indexed by date."""
    if not trades:
        return pd.Series(dtype=float)

    comm_per_leg = commission_per_side * 2
    records = []
    for t in trades:
        exit_ts = pd.Timestamp(t["exit_time"])
        if exit_ts.tz is None:
            exit_ts = exit_ts.tz_localize("UTC")
        exit_et = exit_ts.tz_convert("America/New_York")
        date = exit_et.strftime("%Y-%m-%d")

        if "leg1_pts" in t:
            # Partial exit trade (2 contracts)
            pnl = (t["leg1_pts"] * dollar_per_pt - comm_per_leg +
                   t["leg2_pts"] * dollar_per_pt - comm_per_leg)
        else:
            # Single contract trade
            comm_pts = (commission_per_side * 2) / dollar_per_pt
            pnl = (t["pts"] - comm_pts) * dollar_per_pt

        records.append({"date": date, "pnl": pnl})

    df = pd.DataFrame(records)
    return df.groupby("date")["pnl"].sum()


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data():
    """Load MNQ data and compute SM."""
    df = load_instrument_1min("MNQ")
    sm = compute_smart_money(
        df['Close'].values, df['Volume'].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df['SM_Net'] = sm
    print(f"Loaded MNQ: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def prepare_rsi(df, rsi_len):
    """Compute 5-min RSI mapped back to 1-min bars."""
    df_5m = resample_to_5min(df)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values, df_5m['Close'].values,
        rsi_len=rsi_len,
    )
    return rsi_curr, rsi_prev


# ============================================================================
# Sweep Parameters
# ============================================================================

TP1_VALUES = [5, 7]
TP2_VALUES = [12, 15, 20, 25, 30]
SL_VALUES = [30, 35, 40, 50]
BE_TIME_VALUES = [0, 45, 60, 75, 90]
SL_TO_BE_VALUES = [True, False]


# ============================================================================
# Main Sweep
# ============================================================================

def run_sweep():
    print("=" * 100)
    print("vScalpC Partial Exit Sweep — Phase 1: Base Partial Exit Structure")
    print("V15 entries (SM_T=0.0, RSI 8/60/40, CD=20, cutoff 13:00) + 2-contract partial exit")
    print("=" * 100)

    # Load data
    df = prepare_data()
    rsi_curr, rsi_prev = prepare_rsi(df, VSCALPA_RSI_LEN)

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    sm = df['SM_Net'].values
    times = df.index

    # ========================================================================
    # BASELINES: V15 (1 contract)
    # ========================================================================
    print("\n" + "=" * 100)
    print("BASELINES (single contract, for reference)")
    print("=" * 100)

    # V15 current: TP=5, SL=40
    v15_trades_tp5 = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=40, tp_pts=5,
        entry_end_et=VSCALPA_ENTRY_END_ET,
    )
    v15_sc_tp5 = score_trades(v15_trades_tp5, commission_per_side=MNQ_COMMISSION,
                              dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"  V15 (TP=5, SL=40, 1ct): {fmt_score(v15_sc_tp5, '')}")

    # V15 upgrade: TP=7, SL=40
    v15_trades_tp7 = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=40, tp_pts=7,
        entry_end_et=VSCALPA_ENTRY_END_ET,
    )
    v15_sc_tp7 = score_trades(v15_trades_tp7, commission_per_side=MNQ_COMMISSION,
                              dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"  V15 (TP=7, SL=40, 1ct): {fmt_score(v15_sc_tp7, '')}")

    # vScalpB baseline for portfolio analysis
    vscalpb_trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
    )
    vscalpb_sc = score_trades(vscalpb_trades, commission_per_side=MNQ_COMMISSION,
                              dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"  vScalpB (TP=5, SL=15, 1ct): {fmt_score(vscalpb_sc, '')}")

    # ========================================================================
    # SWEEP
    # ========================================================================
    combos = list(product(TP1_VALUES, TP2_VALUES, SL_VALUES, BE_TIME_VALUES, SL_TO_BE_VALUES))
    total = len(combos)
    print(f"\n{'=' * 100}")
    print(f"SWEEPING {total} parameter combos...")
    print(f"{'=' * 100}")

    results = []
    for idx, (tp1, tp2, sl, be_time, sl_to_be) in enumerate(combos):
        trades = run_backtest_partial_exit(
            opens, highs, lows, closes, sm, times,
            rsi_curr, rsi_prev,
            rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
            sm_threshold=VSCALPA_SM_THRESHOLD,
            cooldown_bars=VSCALPA_COOLDOWN,
            sl_pts=sl, tp1_pts=tp1, tp2_pts=tp2,
            sl_to_be_after_tp1=sl_to_be,
            be_time_bars=be_time,
            entry_end_et=VSCALPA_ENTRY_END_ET,
        )

        sc = score_partial_trades(trades)
        if sc is not None:
            results.append({
                "tp1": tp1, "tp2": tp2, "sl": sl,
                "be_time": be_time, "sl_to_be": sl_to_be,
                "trades": trades,
                **sc,
            })

        if (idx + 1) % 50 == 0:
            print(f"  ... {idx + 1}/{total} combos done")

    print(f"  Completed {len(results)} valid combos")

    # Sort by Sharpe
    results_by_sharpe = sorted(results, key=lambda r: r["sharpe"], reverse=True)

    # ========================================================================
    # OUTPUT 1: Top 20 by Sharpe
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 1: Top 20 by Sharpe")
    print(f"{'=' * 100}")

    header = (f"{'Rank':>4} {'TP1':>4} {'TP2':>4} {'SL':>4} {'BET':>4} {'BE':>5} | "
              f"{'N':>4} {'WR%':>6} {'PF':>6} {'P&L':>10} {'Sharpe':>7} {'MaxDD':>9} | "
              f"{'L1 Exits':>20} {'L2 Exits':>25}")
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(results_by_sharpe[:20], 1):
        be_str = "Y" if r["sl_to_be"] else "N"
        l1_str = " ".join(f"{k}:{v}" for k, v in sorted(r["leg1_exits"].items()))
        l2_str = " ".join(f"{k}:{v}" for k, v in sorted(r["leg2_exits"].items()))
        print(f"{rank:>4} {r['tp1']:>4} {r['tp2']:>4} {r['sl']:>4} {r['be_time']:>4} {be_str:>5} | "
              f"{r['count']:>4} {r['win_rate']:>5.1f}% {r['pf']:>6.3f} "
              f"${r['net_dollar']:>+9.2f} {r['sharpe']:>7.3f} "
              f"${r['max_dd_dollar']:>8.2f} | {l1_str:>20} {l2_str:>25}")

    # ========================================================================
    # OUTPUT 2: IS/OOS Validation for Top 10
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 2: IS/OOS Split Validation (Top 10 by Sharpe)")
    print("First half = IS, Second half = OOS")
    print(f"{'=' * 100}")

    n_total = len(df)
    n_half = n_total // 2
    df_is = df.iloc[:n_half]
    df_oos = df.iloc[n_half:]

    rsi_is_curr, rsi_is_prev = prepare_rsi(df_is, VSCALPA_RSI_LEN)
    rsi_oos_curr, rsi_oos_prev = prepare_rsi(df_oos, VSCALPA_RSI_LEN)

    print(f"  IS:  {len(df_is):,} bars ({df_is.index[0]} to {df_is.index[-1]})")
    print(f"  OOS: {len(df_oos):,} bars ({df_oos.index[0]} to {df_oos.index[-1]})")

    top10 = results_by_sharpe[:10]
    is_oos_results = []

    for rank, r in enumerate(top10, 1):
        # IS
        is_trades = run_backtest_partial_exit(
            df_is['Open'].values, df_is['High'].values,
            df_is['Low'].values, df_is['Close'].values,
            df_is['SM_Net'].values, df_is.index,
            rsi_is_curr, rsi_is_prev,
            rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
            sm_threshold=VSCALPA_SM_THRESHOLD,
            cooldown_bars=VSCALPA_COOLDOWN,
            sl_pts=r["sl"], tp1_pts=r["tp1"], tp2_pts=r["tp2"],
            sl_to_be_after_tp1=r["sl_to_be"],
            be_time_bars=r["be_time"],
            entry_end_et=VSCALPA_ENTRY_END_ET,
        )
        is_sc = score_partial_trades(is_trades)

        # OOS
        oos_trades = run_backtest_partial_exit(
            df_oos['Open'].values, df_oos['High'].values,
            df_oos['Low'].values, df_oos['Close'].values,
            df_oos['SM_Net'].values, df_oos.index,
            rsi_oos_curr, rsi_oos_prev,
            rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
            sm_threshold=VSCALPA_SM_THRESHOLD,
            cooldown_bars=VSCALPA_COOLDOWN,
            sl_pts=r["sl"], tp1_pts=r["tp1"], tp2_pts=r["tp2"],
            sl_to_be_after_tp1=r["sl_to_be"],
            be_time_bars=r["be_time"],
            entry_end_et=VSCALPA_ENTRY_END_ET,
        )
        oos_sc = score_partial_trades(oos_trades)

        is_oos_results.append({
            "rank": rank, "config": r,
            "is_sc": is_sc, "oos_sc": oos_sc,
            "is_trades": is_trades, "oos_trades": oos_trades,
        })

        be_str = "Y" if r["sl_to_be"] else "N"
        label = f"TP1={r['tp1']} TP2={r['tp2']} SL={r['sl']} BET={r['be_time']} BE={be_str}"

        print(f"\n  #{rank} {label}")
        print(f"    FULL: {r['count']:>4} trades, WR {r['win_rate']:>5.1f}%, "
              f"PF {r['pf']:>6.3f}, ${r['net_dollar']:>+9.2f}, "
              f"Sharpe {r['sharpe']:>6.3f}, MaxDD ${r['max_dd_dollar']:.2f}")

        if is_sc:
            print(f"    IS  : {is_sc['count']:>4} trades, WR {is_sc['win_rate']:>5.1f}%, "
                  f"PF {is_sc['pf']:>6.3f}, ${is_sc['net_dollar']:>+9.2f}, "
                  f"Sharpe {is_sc['sharpe']:>6.3f}, MaxDD ${is_sc['max_dd_dollar']:.2f}")
        else:
            print(f"    IS  : NO TRADES")

        if oos_sc:
            print(f"    OOS : {oos_sc['count']:>4} trades, WR {oos_sc['win_rate']:>5.1f}%, "
                  f"PF {oos_sc['pf']:>6.3f}, ${oos_sc['net_dollar']:>+9.2f}, "
                  f"Sharpe {oos_sc['sharpe']:>6.3f}, MaxDD ${oos_sc['max_dd_dollar']:.2f}")
        else:
            print(f"    OOS : NO TRADES")

        # Stability flag
        if is_sc and oos_sc and is_sc["pf"] > 0:
            pf_ratio = oos_sc["pf"] / is_sc["pf"]
            if oos_sc["pf"] > is_sc["pf"]:
                print(f"    --> STRONG (OOS PF {oos_sc['pf']:.3f} > IS PF {is_sc['pf']:.3f})")
            elif pf_ratio >= 0.9:
                print(f"    --> STABLE (OOS PF is {pf_ratio:.1%} of IS PF)")
            elif pf_ratio >= 0.7:
                print(f"    --> MARGINAL (OOS PF is {pf_ratio:.1%} of IS PF)")
            else:
                print(f"    --> WEAK (OOS PF is {pf_ratio:.1%} of IS PF — likely overfit)")

    # ========================================================================
    # OUTPUT 3: Comparison vs Standalone V15 Baselines
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 3: vScalpC vs Standalone V15 Baselines")
    print(f"{'=' * 100}")

    print(f"\n  {'Config':>45} | {'N':>4} {'WR%':>6} {'PF':>6} {'P&L':>10} {'Sharpe':>7} {'MaxDD':>9}")
    print(f"  " + "-" * 100)

    print(f"  {'V15 current (1ct, TP=5, SL=40)':>45} | "
          f"{v15_sc_tp5['count']:>4} {v15_sc_tp5['win_rate']:>5.1f}% {v15_sc_tp5['pf']:>6.3f} "
          f"${v15_sc_tp5['net_dollar']:>+9.2f} {v15_sc_tp5['sharpe']:>7.3f} "
          f"${v15_sc_tp5['max_dd_dollar']:>8.2f}")

    print(f"  {'V15 upgrade (1ct, TP=7, SL=40)':>45} | "
          f"{v15_sc_tp7['count']:>4} {v15_sc_tp7['win_rate']:>5.1f}% {v15_sc_tp7['pf']:>6.3f} "
          f"${v15_sc_tp7['net_dollar']:>+9.2f} {v15_sc_tp7['sharpe']:>7.3f} "
          f"${v15_sc_tp7['max_dd_dollar']:>8.2f}")

    # Show top 3 vScalpC configs
    for rank, r in enumerate(results_by_sharpe[:3], 1):
        be_str = "Y" if r["sl_to_be"] else "N"
        label = f"vScalpC #{rank} (2ct, TP1={r['tp1']}/TP2={r['tp2']} SL={r['sl']} BET={r['be_time']} BE={be_str})"
        print(f"  {label:>45} | "
              f"{r['count']:>4} {r['win_rate']:>5.1f}% {r['pf']:>6.3f} "
              f"${r['net_dollar']:>+9.2f} {r['sharpe']:>7.3f} "
              f"${r['max_dd_dollar']:>8.2f}")

    # ========================================================================
    # OUTPUT 4: Portfolio Analysis
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 4: Portfolio Analysis — Does vScalpC add value?")
    print(f"{'=' * 100}")

    # Daily P&L series for baselines
    v15_daily = daily_pnl_series(v15_trades_tp5)
    vscalpb_daily = daily_pnl_series(vscalpb_trades)

    for rank, r in enumerate(results_by_sharpe[:3], 1):
        be_str = "Y" if r["sl_to_be"] else "N"
        label = f"vScalpC #{rank} (TP1={r['tp1']}/TP2={r['tp2']} SL={r['sl']} BET={r['be_time']} BE={be_str})"
        print(f"\n  --- {label} ---")

        vc_daily = daily_pnl_series(r["trades"])

        # Align daily series
        all_dates = sorted(set(v15_daily.index) | set(vscalpb_daily.index) | set(vc_daily.index))
        v15_aligned = v15_daily.reindex(all_dates, fill_value=0)
        vb_aligned = vscalpb_daily.reindex(all_dates, fill_value=0)
        vc_aligned = vc_daily.reindex(all_dates, fill_value=0)

        # Correlations
        corr_v15 = v15_aligned.corr(vc_aligned)
        corr_vb = vb_aligned.corr(vc_aligned)
        print(f"    Correlation with V15 daily P&L:    {corr_v15:.3f}")
        print(f"    Correlation with vScalpB daily P&L: {corr_vb:.3f}")

        # Combined portfolio: V15(1ct) + vScalpB(1ct) + vScalpC(2ct)
        combined_daily = v15_aligned + vb_aligned + vc_aligned
        total_pnl = combined_daily.sum()
        cum = combined_daily.cumsum()
        peak = cum.cummax()
        max_dd = (cum - peak).min()

        # Sharpe of combined daily
        if len(combined_daily) > 1 and combined_daily.std() > 0:
            port_sharpe = combined_daily.mean() / combined_daily.std() * np.sqrt(252)
        else:
            port_sharpe = 0.0

        print(f"    Combined portfolio (V15 1ct + vScalpB 1ct + vScalpC 2ct):")
        print(f"      Total P&L: ${total_pnl:+,.2f}")
        print(f"      Sharpe:    {port_sharpe:.3f}")
        print(f"      MaxDD:     ${max_dd:,.2f}")

        # Without vScalpC
        base_daily = v15_aligned + vb_aligned
        base_pnl = base_daily.sum()
        base_cum = base_daily.cumsum()
        base_peak = base_cum.cummax()
        base_dd = (base_cum - base_peak).min()
        if len(base_daily) > 1 and base_daily.std() > 0:
            base_sharpe = base_daily.mean() / base_daily.std() * np.sqrt(252)
        else:
            base_sharpe = 0.0

        print(f"    Without vScalpC (V15 1ct + vScalpB 1ct):")
        print(f"      Total P&L: ${base_pnl:+,.2f}")
        print(f"      Sharpe:    {base_sharpe:.3f}")
        print(f"      MaxDD:     ${base_dd:,.2f}")

        marginal_pnl = total_pnl - base_pnl
        marginal_sharpe = port_sharpe - base_sharpe
        print(f"    Marginal contribution of vScalpC:")
        print(f"      +${marginal_pnl:+,.2f} P&L, {marginal_sharpe:+.3f} Sharpe, "
              f"MaxDD change: ${max_dd - base_dd:+,.2f}")

        if marginal_sharpe > 0 and marginal_pnl > 0:
            print(f"    --> vScalpC ADDS value to portfolio")
        elif marginal_pnl > 0:
            print(f"    --> vScalpC adds P&L but reduces Sharpe — increases volatility")
        else:
            print(f"    --> vScalpC DOES NOT add value — duplicates/degrades portfolio")

    # ========================================================================
    # OUTPUT 5: Cluster Analysis by TP2 Level
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 5: Cluster Analysis by TP2 Runner Target")
    print(f"{'=' * 100}")

    tp2_groups = {}
    for r in results:
        tp2 = r["tp2"]
        if tp2 not in tp2_groups:
            tp2_groups[tp2] = []
        tp2_groups[tp2].append(r)

    print(f"\n  {'TP2':>4} | {'Combos':>6} {'AvgN':>5} {'AvgWR%':>7} {'AvgPF':>6} "
          f"{'AvgP&L':>10} {'AvgSharpe':>10} {'BestSharpe':>11} {'WorstSharpe':>12}")
    print(f"  " + "-" * 90)

    for tp2 in sorted(tp2_groups.keys()):
        group = tp2_groups[tp2]
        avg_n = np.mean([r["count"] for r in group])
        avg_wr = np.mean([r["win_rate"] for r in group])
        avg_pf = np.mean([r["pf"] for r in group])
        avg_pnl = np.mean([r["net_dollar"] for r in group])
        avg_sharpe = np.mean([r["sharpe"] for r in group])
        best_sharpe = max(r["sharpe"] for r in group)
        worst_sharpe = min(r["sharpe"] for r in group)

        print(f"  {tp2:>4} | {len(group):>6} {avg_n:>5.0f} {avg_wr:>6.1f}% {avg_pf:>6.3f} "
              f"${avg_pnl:>+9.2f} {avg_sharpe:>10.3f} {best_sharpe:>11.3f} {worst_sharpe:>12.3f}")

    # Sub-cluster: TP2 x SL_to_BE
    print(f"\n  --- By TP2 x SL_to_BE ---")
    print(f"  {'TP2':>4} {'BE':>3} | {'AvgPF':>6} {'AvgSharpe':>10} {'AvgP&L':>10}")
    print(f"  " + "-" * 50)

    for tp2 in sorted(tp2_groups.keys()):
        for sl_be in [True, False]:
            sub = [r for r in tp2_groups[tp2] if r["sl_to_be"] == sl_be]
            if sub:
                avg_pf = np.mean([r["pf"] for r in sub])
                avg_sharpe = np.mean([r["sharpe"] for r in sub])
                avg_pnl = np.mean([r["net_dollar"] for r in sub])
                be_str = "Y" if sl_be else "N"
                print(f"  {tp2:>4} {be_str:>3} | {avg_pf:>6.3f} {avg_sharpe:>10.3f} ${avg_pnl:>+9.2f}")

    # Sub-cluster: TP1 x TP2
    print(f"\n  --- By TP1 x TP2 ---")
    print(f"  {'TP1':>4} {'TP2':>4} | {'Combos':>6} {'AvgPF':>6} {'AvgSharpe':>10} {'AvgP&L':>10} {'BestSharpe':>11}")
    print(f"  " + "-" * 65)

    for tp1 in TP1_VALUES:
        for tp2 in sorted(TP2_VALUES):
            sub = [r for r in results if r["tp1"] == tp1 and r["tp2"] == tp2]
            if sub:
                avg_pf = np.mean([r["pf"] for r in sub])
                avg_sharpe = np.mean([r["sharpe"] for r in sub])
                avg_pnl = np.mean([r["net_dollar"] for r in sub])
                best_sharpe = max(r["sharpe"] for r in sub)
                print(f"  {tp1:>4} {tp2:>4} | {len(sub):>6} {avg_pf:>6.3f} {avg_sharpe:>10.3f} "
                      f"${avg_pnl:>+9.2f} {best_sharpe:>11.3f}")

    # Sub-cluster: BE_TIME
    print(f"\n  --- By BE_TIME ---")
    print(f"  {'BET':>4} | {'Combos':>6} {'AvgPF':>6} {'AvgSharpe':>10} {'AvgP&L':>10} {'BestSharpe':>11}")
    print(f"  " + "-" * 60)

    for bet in BE_TIME_VALUES:
        sub = [r for r in results if r["be_time"] == bet]
        if sub:
            avg_pf = np.mean([r["pf"] for r in sub])
            avg_sharpe = np.mean([r["sharpe"] for r in sub])
            avg_pnl = np.mean([r["net_dollar"] for r in sub])
            best_sharpe = max(r["sharpe"] for r in sub)
            print(f"  {bet:>4} | {len(sub):>6} {avg_pf:>6.3f} {avg_sharpe:>10.3f} "
                  f"${avg_pnl:>+9.2f} {best_sharpe:>11.3f}")

    print(f"\n{'=' * 100}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    run_sweep()
