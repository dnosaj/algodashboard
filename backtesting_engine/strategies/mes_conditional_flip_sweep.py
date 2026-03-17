"""
MES v2 Conditional Counter-Signal Flip Sweep
==============================================
Tests whether flipping a losing MES v2 trade on a counter-signal improves
performance vs the current behavior of ignoring all counter-signals.

Hypothesis: If a MES v2 trade is old AND losing AND a counter-signal fires,
closing the loser and entering the counter-direction should cut losses faster
than waiting for SL/TP/BE_TIME.

This is NOT the blanket SM flip exit (which failed OOS). This is a conditional
flip that only fires when:
  1. Trade age >= min_bars_for_flip  (sweep: 30, 45, 60, 75)
  2. Unrealized P&L is negative by >= min_loss_pts  (sweep: 0, 5, 10)
  3. A valid counter-signal fires (SM direction + RSI cross)

Sweep: 4 min_bars x 3 min_loss x 3 periods (FULL/IS/OOS) = 36 runs + 3 baselines.

MES v2 production params:
  SM(20/12/400/255) SM_T=0.0 RSI(12/55/45) CD=25 SL=35 TP=20
  entry_qty=2, partial_tp_pts=6, partial_qty=1, BE_TIME=75, entry_end=14:15 ET
  eod=16:00 ET, prior_day_level gate (VPOC+VAL buf=5)

Usage:
    cd backtesting_engine && python3 strategies/mes_conditional_flip_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Path setup ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    compute_rsi,
    map_5min_rsi_to_1min,
    resample_to_5min,
    compute_et_minutes,
    NY_OPEN_ET,
    NY_CLOSE_ET,
)

from generate_session import (
    load_instrument_1min,
    MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA,
    MES_DOLLAR_PER_PT, MES_COMMISSION,
    MESV2_RSI_LEN, MESV2_RSI_BUY, MESV2_RSI_SELL,
    MESV2_SM_THRESHOLD, MESV2_COOLDOWN,
    MESV2_MAX_LOSS_PTS, MESV2_TP_PTS, MESV2_BREAKEVEN_BARS,
)

# MES v2 corrected session params (matching config.py production)
MESV2_EOD_ET = 16 * 60          # 16:00 ET = 960 minutes
MESV2_ENTRY_END_ET = 14 * 60 + 15  # 14:15 ET = 855 minutes


# ============================================================================
# Partial Exit Backtest with Conditional Flip
# ============================================================================

def run_backtest_mes_v2_with_flip(
    opens, highs, lows, closes, sm, times,
    rsi_5m_curr, rsi_5m_prev,
    rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
    sm_threshold=MESV2_SM_THRESHOLD,
    cooldown_bars=MESV2_COOLDOWN,
    sl_pts=MESV2_MAX_LOSS_PTS,
    tp1_pts=6,      # partial TP (TP1)
    tp2_pts=MESV2_TP_PTS,  # runner TP (TP2)
    be_time_bars=MESV2_BREAKEVEN_BARS,
    entry_end_et=MESV2_ENTRY_END_ET,
    eod_minutes_et=MESV2_EOD_ET,
    entry_gate=None,
    # --- Flip parameters ---
    enable_flip=False,
    min_bars_for_flip=60,
    min_loss_pts_for_flip=0,
):
    """MES v2 backtest with 2-contract partial exit and optional conditional flip.

    This mirrors the production MES v2 logic:
      - 2 contracts entered simultaneously
      - Leg 1 (scalp): exits at TP1 or SL or EOD
      - Leg 2 (runner): exits at TP2 or SL or BE_TIME or EOD
      - No SL-to-BE after TP1 (MES v2 does not use this)

    Conditional flip (when enable_flip=True):
      On each bar, if in a position AND a counter-signal fires:
        1. Trade age >= min_bars_for_flip
        2. Unrealized P&L negative by >= min_loss_pts_for_flip
      Then: close all remaining legs at bar[i] open, enter counter-direction.
      The flip entry bypasses cooldown but respects session_end.
      After the flip, normal cooldown applies to subsequent entries.

    Returns:
        tuple: (trades, flip_stats)
          trades: list of composite trade dicts
          flip_stats: list of flip event dicts for analysis
    """
    n = len(opens)
    trades = []
    flip_stats = []
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
    partial_filled = False  # True after TP1 fills

    et_mins = compute_et_minutes(times)

    def finalize_trade(side, entry_p, entry_i, last_exit_i,
                       l1_pts, l1_reason, l1_idx,
                       l2_price, l2_reason, qty_at_close):
        """Record the composite 2-leg trade."""
        leg2_pts_val = (l2_price - entry_p) if side == "long" else (entry_p - l2_price)
        total_pts = l1_pts + leg2_pts_val

        trades.append({
            "side": side,
            "entry": entry_p,
            "exit": l2_price,
            "pts": total_pts,
            "entry_time": times[entry_i],
            "exit_time": times[last_exit_i],
            "entry_idx": entry_i,
            "exit_idx": last_exit_i,
            "bars": last_exit_i - entry_i,
            "result": f"{l1_reason}+{l2_reason}",
            "leg1_pts": l1_pts,
            "leg2_pts": leg2_pts_val,
            "leg1_exit_reason": l1_reason,
            "leg2_exit_reason": l2_reason,
            "leg1_exit_idx": l1_idx,
            "leg2_exit_idx": last_exit_i,
            "qty_at_close": qty_at_close,  # how many contracts were open at final close
        })

    def do_flip_close(i, side, flip_reason_suffix="FLIP"):
        """Close all remaining legs for a flip. Returns qty closed."""
        nonlocal leg1_active, leg2_active, leg1_pts, leg1_exit_price
        nonlocal leg1_exit_idx, leg1_exit_reason, partial_filled

        qty_closed = 0
        flip_price = opens[i]

        if leg1_active:
            leg1_pts = (flip_price - entry_price) if side == "long" else (entry_price - flip_price)
            leg1_exit_price = flip_price
            leg1_exit_idx = i
            leg1_exit_reason = flip_reason_suffix
            leg1_active = False
            qty_closed += 1

        l2_price = flip_price
        l2_reason = flip_reason_suffix
        if leg2_active:
            leg2_active = False
            qty_closed += 1

        # If leg1 already closed (TP1), only leg2 was open
        if not partial_filled and qty_closed == 0:
            # Both legs already closed somehow -- shouldn't happen
            pass

        finalize_trade(side, entry_price, entry_idx, i,
                       leg1_pts, leg1_exit_reason, leg1_exit_idx if leg1_exit_idx > 0 else i,
                       l2_price, l2_reason, qty_closed)
        return qty_closed

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

        # Episode reset (uses zero crossing, not threshold)
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

            l2_price = eod_price
            l2_reason = "EOD"
            if leg2_active:
                leg2_active = False

            qty_remaining = (1 if partial_filled else 2) if not leg1_active else 2
            finalize_trade(side, entry_price, entry_idx, i,
                           leg1_pts, leg1_exit_reason, leg1_exit_idx,
                           l2_price, l2_reason, qty_remaining)
            trade_state = 0
            exit_bar = i
            partial_filled = False
            continue

        # --- Conditional Flip Check (before normal exits) ---
        flip_triggered = False
        if enable_flip and trade_state != 0:
            side = "long" if trade_state == 1 else "short"
            trade_age = (i - 1) - entry_idx  # bars since entry (using i-1 convention)

            # Check if counter-signal fires
            if trade_state == 1:
                counter_signal = sm_bear and rsi_short_trigger
            else:
                counter_signal = sm_bull and rsi_long_trigger

            if counter_signal and trade_age >= min_bars_for_flip:
                # Check unrealized P&L (using prev bar close)
                if trade_state == 1:
                    unrealized_pts = closes[i - 1] - entry_price
                else:
                    unrealized_pts = entry_price - closes[i - 1]

                # For partial-filled trades, unrealized is only on the runner
                # But the condition checks overall trade health
                if unrealized_pts <= -min_loss_pts_for_flip:
                    # Check session time for the new entry
                    in_session_for_flip = NY_OPEN_ET <= bar_mins_et <= entry_end_et

                    if in_session_for_flip:
                        # Record flip stats
                        flip_close_price = opens[i]
                        if trade_state == 1:
                            close_pnl_pts = flip_close_price - entry_price
                        else:
                            close_pnl_pts = entry_price - flip_close_price

                        # How many contracts are being closed?
                        contracts_open = 0
                        if leg1_active:
                            contracts_open += 1
                        if leg2_active:
                            contracts_open += 1

                        # Close the losing trade
                        do_flip_close(i, side, "FLIP")
                        trade_state = 0
                        # NOTE: Do NOT set exit_bar = i here -- flip bypasses cooldown
                        # for the immediate re-entry. But we set exit_bar after the
                        # new entry to ensure cooldown applies to SUBSEQUENT entries.

                        # Enter the counter-direction
                        new_side = "short" if side == "long" else "long"
                        new_entry_price = opens[i]

                        trade_state = 1 if new_side == "long" else -1
                        entry_price = new_entry_price
                        entry_idx = i
                        leg1_active = True
                        leg2_active = True
                        partial_filled = False
                        leg1_pts = 0.0
                        leg1_exit_reason = ""
                        leg1_exit_price = 0.0
                        leg1_exit_idx = 0

                        if new_side == "long":
                            long_used = True
                        else:
                            short_used = True

                        # Record flip event for analysis
                        flip_stats.append({
                            "bar_idx": i,
                            "time": times[i],
                            "old_side": side,
                            "new_side": new_side,
                            "old_entry_price": entry_price,
                            "flip_price": flip_close_price,
                            "close_pnl_pts": close_pnl_pts,
                            "contracts_closed": contracts_open,
                            "trade_age_bars": trade_age,
                            "unrealized_pts": unrealized_pts,
                            "partial_filled_at_flip": partial_filled,
                        })

                        flip_triggered = True
                        # Set exit_bar so cooldown applies to entries AFTER this flip
                        exit_bar = i

        if flip_triggered:
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
                    partial_filled = True

            # Check Leg 2 (runner) exits
            if leg2_active:
                # SL: prev bar close breaches runner stop (always original SL for MES v2)
                if closes[i - 1] <= entry_price - sl_pts:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "SL"
                    leg2_active = False
                # TP2: prev bar close reached TP2
                elif closes[i - 1] >= entry_price + tp2_pts:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "TP2"
                    leg2_active = False
                # BE_TIME: runner held too long (only after TP1)
                elif be_time_bars > 0 and not leg1_active:
                    bars_since_entry = (i - 1) - entry_idx
                    if bars_since_entry >= be_time_bars:
                        leg2_exit_price = opens[i]
                        leg2_exit_reason = "BE_TIME"
                        leg2_active = False

            # If both legs closed
            if not leg1_active and not leg2_active:
                qty_at_close = 1 if partial_filled else 2
                finalize_trade("long", entry_price, entry_idx, i,
                               leg1_pts, leg1_exit_reason,
                               leg1_exit_idx if leg1_exit_idx > 0 else i,
                               leg2_exit_price, leg2_exit_reason, qty_at_close)
                trade_state = 0
                exit_bar = i
                partial_filled = False
                continue

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
                    partial_filled = True

            if leg2_active:
                # SL
                if closes[i - 1] >= entry_price + sl_pts:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "SL"
                    leg2_active = False
                # TP2
                elif closes[i - 1] <= entry_price - tp2_pts:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "TP2"
                    leg2_active = False
                # BE_TIME
                elif be_time_bars > 0 and not leg1_active:
                    bars_since_entry = (i - 1) - entry_idx
                    if bars_since_entry >= be_time_bars:
                        leg2_exit_price = opens[i]
                        leg2_exit_reason = "BE_TIME"
                        leg2_active = False

            if not leg1_active and not leg2_active:
                qty_at_close = 1 if partial_filled else 2
                finalize_trade("short", entry_price, entry_idx, i,
                               leg1_pts, leg1_exit_reason,
                               leg1_exit_idx if leg1_exit_idx > 0 else i,
                               leg2_exit_price, leg2_exit_reason, qty_at_close)
                trade_state = 0
                exit_bar = i
                partial_filled = False
                continue

        # --- Entries ---
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars
            gate_ok = entry_gate is None or entry_gate[i - 1]

            if in_session and cd_ok and gate_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    leg1_active = True
                    leg2_active = True
                    partial_filled = False
                    leg1_pts = 0.0
                    leg1_exit_reason = ""
                    leg1_exit_price = 0.0
                    leg1_exit_idx = 0
                    leg2_exit_price = 0.0
                    leg2_exit_reason = ""
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    leg1_active = True
                    leg2_active = True
                    partial_filled = False
                    leg1_pts = 0.0
                    leg1_exit_reason = ""
                    leg1_exit_price = 0.0
                    leg1_exit_idx = 0
                    leg2_exit_price = 0.0
                    leg2_exit_reason = ""

    return trades, flip_stats


# ============================================================================
# Scoring
# ============================================================================

def score_partial_trades(trades, dollar_per_pt=MES_DOLLAR_PER_PT,
                         commission_per_side=MES_COMMISSION):
    """Score MES v2 partial exit trades (2 contracts per entry).

    P&L per trade:
      leg1_pnl = leg1_pts * dollar_per_pt - 2 * commission (entry + exit)
      leg2_pnl = leg2_pts * dollar_per_pt - 2 * commission
      total_pnl = leg1_pnl + leg2_pnl
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
    n_trades = len(pnl_arr)
    total_pnl = pnl_arr.sum()
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n_trades * 100

    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()

    avg_bars = np.mean([t["bars"] for t in trades])

    sharpe = 0.0
    if n_trades > 1 and np.std(pnl_arr) > 0:
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
        "count": n_trades,
        "net_dollar": round(total_pnl, 2),
        "pf": round(pf, 3),
        "win_rate": round(wr, 1),
        "max_dd_dollar": round(mdd, 2),
        "avg_bars": round(avg_bars, 1),
        "sharpe": round(sharpe, 3),
        "leg1_exits": leg1_exits,
        "leg2_exits": leg2_exits,
        "pnl_array": pnl_arr,
    }


# ============================================================================
# Flip Statistics Analysis
# ============================================================================

def analyze_flips(flip_stats, trades, dollar_per_pt=MES_DOLLAR_PER_PT,
                  commission_per_side=MES_COMMISSION):
    """Analyze flip events: how much loss was cut, did the new trade win?"""
    if not flip_stats:
        return {
            "flip_count": 0,
            "avg_closed_pnl_pts": 0,
            "avg_new_trade_pnl_pts": 0,
            "flip_win_rate": 0,
        }

    comm_per_leg = commission_per_side * 2

    # For each flip, find the trade that was opened by the flip
    # (the trade whose entry_idx matches the flip bar_idx)
    flip_bar_set = {fs["bar_idx"] for fs in flip_stats}

    # Trades opened by flips
    flip_new_trades = {}
    for t in trades:
        if t["entry_idx"] in flip_bar_set:
            flip_new_trades[t["entry_idx"]] = t

    closed_pnl_pts_list = []
    new_trade_pnl_list = []  # net $ of new trade
    new_trade_wins = 0

    for fs in flip_stats:
        closed_pnl_pts_list.append(fs["close_pnl_pts"])

        # Find the new trade
        new_t = flip_new_trades.get(fs["bar_idx"])
        if new_t is not None:
            new_pnl = (new_t["leg1_pts"] * dollar_per_pt - comm_per_leg +
                       new_t["leg2_pts"] * dollar_per_pt - comm_per_leg)
            new_trade_pnl_list.append(new_pnl)
            if new_pnl > 0:
                new_trade_wins += 1
        else:
            new_trade_pnl_list.append(0)

    avg_closed = np.mean(closed_pnl_pts_list) if closed_pnl_pts_list else 0
    avg_new = np.mean(new_trade_pnl_list) if new_trade_pnl_list else 0
    flip_wr = (new_trade_wins / len(flip_stats) * 100) if flip_stats else 0

    return {
        "flip_count": len(flip_stats),
        "avg_closed_pnl_pts": round(avg_closed, 2),
        "avg_closed_pnl_dollar": round(avg_closed * dollar_per_pt, 2),
        "avg_new_trade_pnl_dollar": round(avg_new, 2),
        "flip_win_rate": round(flip_wr, 1),
        "avg_trade_age_at_flip": round(np.mean([fs["trade_age_bars"] for fs in flip_stats]), 1),
    }


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_mes_data():
    """Load MES data and compute SM."""
    df = load_instrument_1min("MES")
    sm = compute_smart_money(
        df['Close'].values, df['Volume'].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df['SM_Net'] = sm
    print(f"Loaded MES: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
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
# Prior-day level gate (VPOC+VAL only, buffer=5pts)
# ============================================================================

def compute_mes_level_gate(df):
    """Compute the prior-day level gate for MES v2 (VPOC+VAL, buf=5)."""
    from v10_test_common import compute_prior_day_levels
    from sr_prior_day_levels_sweep import compute_rth_volume_profile, build_prior_day_level_gate

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    times = df.index
    volumes = df["Volume"].values
    et_mins = compute_et_minutes(times)

    # Prior-day H/L (unused but needed for function signature)
    prev_high, prev_low, _ = compute_prior_day_levels(times, highs, lows, closes)

    # Volume profile
    vpoc, vah, val = compute_rth_volume_profile(
        times, closes, volumes, et_mins, bin_width=5,
    )

    # Build gate with VPOC + VAL only
    nan_arr = np.full(len(df), np.nan)
    gate = build_prior_day_level_gate(
        closes,
        nan_arr,   # prev_high disabled
        nan_arr,   # prev_low disabled
        vpoc,      # active
        nan_arr,   # VAH disabled
        val,       # active
        buffer_pts=5.0,
    )

    blocked = (~gate).sum()
    print(f"  Prior-day VPOC+VAL (buf=5): blocks {blocked}/{len(gate)} bars "
          f"({blocked/len(gate)*100:.1f}%)")
    return gate


# ============================================================================
# Main Sweep
# ============================================================================

def run_sweep():
    print("=" * 110)
    print("MES v2 CONDITIONAL COUNTER-SIGNAL FLIP SWEEP")
    print("  SM(20/12/400/255) SM_T=0.0 RSI(12/55/45) CD=25 SL=35 TP=20")
    print("  entry_qty=2, partial_tp_pts=6, BE_TIME=75, entry_end=14:15, EOD=16:00")
    print("  Gate: prior_day_level(VPOC+VAL, buf=5)")
    print("=" * 110)

    # Load data
    df = prepare_mes_data()
    rsi_curr, rsi_prev = prepare_rsi(df, MESV2_RSI_LEN)

    # Compute gate
    print("\n--- Computing Entry Gates ---")
    gate_full = compute_mes_level_gate(df)

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    sm = df['SM_Net'].values
    times = df.index

    # IS/OOS split
    n_total = len(df)
    n_half = n_total // 2
    print(f"\n  FULL:  {n_total:,} bars ({times[0]} to {times[-1]})")
    print(f"  IS:    {n_half:,} bars ({times[0]} to {times[n_half-1]})")
    print(f"  OOS:   {n_total - n_half:,} bars ({times[n_half]} to {times[-1]})")

    # Sweep parameters
    MIN_BARS_VALUES = [30, 45, 60, 75]
    MIN_LOSS_VALUES = [0, 5, 10]

    # ========================================================================
    # Run baselines (no flip) on FULL, IS, OOS
    # ========================================================================
    print(f"\n{'='*110}")
    print("BASELINES (no flip, current MES v2 behavior)")
    print(f"{'='*110}")

    baselines = {}
    for period_name, start, end, gate_slice in [
        ("FULL", 0, n_total, gate_full),
        ("IS", 0, n_half, gate_full[:n_half]),
        ("OOS", n_half, n_total, gate_full[n_half:]),
    ]:
        sl = slice(start, end)
        o, h, l, c, s = opens[sl], highs[sl], lows[sl], closes[sl], sm[sl]
        t = times[sl]

        # Recompute RSI for this slice
        df_slice = df.iloc[sl]
        rsi_c, rsi_p = prepare_rsi(df_slice, MESV2_RSI_LEN)

        trades_b, _ = run_backtest_mes_v2_with_flip(
            o, h, l, c, s, t, rsi_c, rsi_p,
            entry_gate=gate_slice,
            enable_flip=False,
        )
        sc = score_partial_trades(trades_b)
        baselines[period_name] = {"trades": trades_b, "score": sc}

        if sc:
            exits_str = " ".join(f"{k}:{v}" for k, v in sorted(sc["leg1_exits"].items()))
            exits2_str = " ".join(f"{k}:{v}" for k, v in sorted(sc["leg2_exits"].items()))
            print(f"\n  {period_name:>4}: {sc['count']:>4} trades, WR {sc['win_rate']:>5.1f}%, "
                  f"PF {sc['pf']:>6.3f}, ${sc['net_dollar']:>+10.2f}, "
                  f"Sharpe {sc['sharpe']:>6.3f}, MaxDD ${sc['max_dd_dollar']:.2f}")
            print(f"        L1: {exits_str}")
            print(f"        L2: {exits2_str}")

    # ========================================================================
    # Run sweep
    # ========================================================================
    print(f"\n{'='*110}")
    print("SWEEP: Conditional Flip Configs")
    print(f"{'='*110}")

    all_results = []

    for min_bars in MIN_BARS_VALUES:
        for min_loss in MIN_LOSS_VALUES:
            config_label = f"min_bars={min_bars} min_loss={min_loss}"
            config_results = {"min_bars": min_bars, "min_loss": min_loss}

            for period_name, start, end, gate_slice in [
                ("FULL", 0, n_total, gate_full),
                ("IS", 0, n_half, gate_full[:n_half]),
                ("OOS", n_half, n_total, gate_full[n_half:]),
            ]:
                sl = slice(start, end)
                o, h, l, c, s = opens[sl], highs[sl], lows[sl], closes[sl], sm[sl]
                t = times[sl]

                df_slice = df.iloc[sl]
                rsi_c, rsi_p = prepare_rsi(df_slice, MESV2_RSI_LEN)

                trades_f, flips = run_backtest_mes_v2_with_flip(
                    o, h, l, c, s, t, rsi_c, rsi_p,
                    entry_gate=gate_slice,
                    enable_flip=True,
                    min_bars_for_flip=min_bars,
                    min_loss_pts_for_flip=min_loss,
                )
                sc = score_partial_trades(trades_f)
                fa = analyze_flips(flips, trades_f)

                config_results[period_name] = {
                    "score": sc,
                    "flip_analysis": fa,
                    "trades": trades_f,
                    "flips": flips,
                }

            all_results.append(config_results)

            # Progress
            sc_full = config_results["FULL"]["score"]
            fa_full = config_results["FULL"]["flip_analysis"]
            if sc_full:
                print(f"  {config_label:>25}: FULL {sc_full['count']:>4} trades, "
                      f"PF {sc_full['pf']:>6.3f}, ${sc_full['net_dollar']:>+10.2f}, "
                      f"Sharpe {sc_full['sharpe']:>6.3f} | "
                      f"Flips: {fa_full['flip_count']}")

    # ========================================================================
    # OUTPUT 1: Summary Table — All configs ranked by OOS PF improvement
    # ========================================================================
    print(f"\n{'='*110}")
    print("TABLE 1: All Configs Ranked by OOS PF Improvement vs Baseline")
    print(f"{'='*110}")

    baseline_oos_pf = baselines["OOS"]["score"]["pf"] if baselines["OOS"]["score"] else 0
    baseline_oos_net = baselines["OOS"]["score"]["net_dollar"] if baselines["OOS"]["score"] else 0
    baseline_full_pf = baselines["FULL"]["score"]["pf"] if baselines["FULL"]["score"] else 0
    baseline_full_net = baselines["FULL"]["score"]["net_dollar"] if baselines["FULL"]["score"] else 0

    print(f"\n  Baseline OOS: PF={baseline_oos_pf:.3f}, Net=${baseline_oos_net:+.2f}")
    print(f"  Baseline FULL: PF={baseline_full_pf:.3f}, Net=${baseline_full_net:+.2f}")

    # Sort by OOS PF improvement
    ranked = []
    for r in all_results:
        oos_sc = r["OOS"]["score"]
        full_sc = r["FULL"]["score"]
        is_sc = r["IS"]["score"]
        oos_pf = oos_sc["pf"] if oos_sc else 0
        oos_pf_delta = oos_pf - baseline_oos_pf
        ranked.append({**r, "oos_pf_delta": oos_pf_delta})

    ranked.sort(key=lambda x: x["oos_pf_delta"], reverse=True)

    header = (f"  {'Rank':>4} {'MinBars':>7} {'MinLoss':>7} | "
              f"{'OOS PF':>7} {'delta':>7} {'OOS $':>10} {'OOS WR%':>7} {'OOS Sharpe':>10} | "
              f"{'FULL PF':>7} {'FULL $':>10} {'FULL Flips':>10} | "
              f"{'IS PF':>6}")
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    for rank, r in enumerate(ranked, 1):
        oos_sc = r["OOS"]["score"]
        full_sc = r["FULL"]["score"]
        is_sc = r["IS"]["score"]
        fa_full = r["FULL"]["flip_analysis"]

        oos_pf = oos_sc["pf"] if oos_sc else 0
        oos_net = oos_sc["net_dollar"] if oos_sc else 0
        oos_wr = oos_sc["win_rate"] if oos_sc else 0
        oos_sh = oos_sc["sharpe"] if oos_sc else 0
        full_pf = full_sc["pf"] if full_sc else 0
        full_net = full_sc["net_dollar"] if full_sc else 0
        is_pf = is_sc["pf"] if is_sc else 0

        print(f"  {rank:>4} {r['min_bars']:>7} {r['min_loss']:>7} | "
              f"{oos_pf:>7.3f} {r['oos_pf_delta']:>+7.3f} ${oos_net:>+9.2f} "
              f"{oos_wr:>6.1f}% {oos_sh:>10.3f} | "
              f"{full_pf:>7.3f} ${full_net:>+9.2f} {fa_full['flip_count']:>10} | "
              f"{is_pf:>6.3f}")

    # ========================================================================
    # OUTPUT 2: Detailed Breakdown — Per-config FULL/IS/OOS with flip stats
    # ========================================================================
    print(f"\n{'='*110}")
    print("TABLE 2: Detailed Breakdown — Per Config FULL/IS/OOS + Flip Statistics")
    print(f"{'='*110}")

    # Show baseline first
    print(f"\n  --- BASELINE (no flip) ---")
    for period in ["FULL", "IS", "OOS"]:
        sc = baselines[period]["score"]
        if sc:
            print(f"    {period:>4}: {sc['count']:>4} trades, WR {sc['win_rate']:>5.1f}%, "
                  f"PF {sc['pf']:>6.3f}, ${sc['net_dollar']:>+10.2f}, "
                  f"Sharpe {sc['sharpe']:>6.3f}, MaxDD ${sc['max_dd_dollar']:.2f}")

    # Show each config
    for r in ranked:
        print(f"\n  --- min_bars={r['min_bars']} min_loss={r['min_loss']} ---")
        for period in ["FULL", "IS", "OOS"]:
            sc = r[period]["score"]
            fa = r[period]["flip_analysis"]
            if sc:
                # Compute PF delta vs baseline
                bl_pf = baselines[period]["score"]["pf"] if baselines[period]["score"] else 0
                pf_delta = sc["pf"] - bl_pf

                print(f"    {period:>4}: {sc['count']:>4} trades, WR {sc['win_rate']:>5.1f}%, "
                      f"PF {sc['pf']:>6.3f} ({pf_delta:>+.3f}), "
                      f"${sc['net_dollar']:>+10.2f}, "
                      f"Sharpe {sc['sharpe']:>6.3f}, MaxDD ${sc['max_dd_dollar']:.2f}")
                if fa["flip_count"] > 0:
                    print(f"          Flips: {fa['flip_count']}, "
                          f"Avg closed P&L: {fa['avg_closed_pnl_pts']:.1f} pts "
                          f"(${fa['avg_closed_pnl_dollar']:.2f}), "
                          f"Avg new trade P&L: ${fa['avg_new_trade_pnl_dollar']:.2f}, "
                          f"Flip WR: {fa['flip_win_rate']:.1f}%, "
                          f"Avg age at flip: {fa['avg_trade_age_at_flip']:.0f} bars")
                else:
                    print(f"          No flips triggered")

    # ========================================================================
    # OUTPUT 3: Best Config Recommendation
    # ========================================================================
    print(f"\n{'='*110}")
    print("TABLE 3: Best Config Recommendation")
    print(f"{'='*110}")

    # Find best by OOS improvement
    best_oos = ranked[0] if ranked else None

    # Find best by FULL Sharpe improvement
    best_sharpe = max(all_results,
                      key=lambda r: (r["FULL"]["score"]["sharpe"]
                                     if r["FULL"]["score"] else -999))

    # Stability check: best IS/OOS consistency
    best_stable = None
    best_stability_score = -999
    for r in all_results:
        is_sc = r["IS"]["score"]
        oos_sc = r["OOS"]["score"]
        if is_sc and oos_sc and is_sc["pf"] > 0:
            pf_ratio = oos_sc["pf"] / is_sc["pf"]
            stability = min(pf_ratio, 1.0 / max(pf_ratio, 0.001))  # closer to 1.0 = more stable
            pf_sum = is_sc["pf"] + oos_sc["pf"]
            composite = stability * pf_sum  # reward stability AND performance
            if composite > best_stability_score:
                best_stability_score = composite
                best_stable = r

    print(f"\n  BASELINE:")
    bl_full = baselines["FULL"]["score"]
    bl_is = baselines["IS"]["score"]
    bl_oos = baselines["OOS"]["score"]
    if bl_full:
        print(f"    FULL: {bl_full['count']} trades, PF {bl_full['pf']:.3f}, "
              f"${bl_full['net_dollar']:+.2f}, Sharpe {bl_full['sharpe']:.3f}")
    if bl_is:
        print(f"    IS:   {bl_is['count']} trades, PF {bl_is['pf']:.3f}, "
              f"${bl_is['net_dollar']:+.2f}, Sharpe {bl_is['sharpe']:.3f}")
    if bl_oos:
        print(f"    OOS:  {bl_oos['count']} trades, PF {bl_oos['pf']:.3f}, "
              f"${bl_oos['net_dollar']:+.2f}, Sharpe {bl_oos['sharpe']:.3f}")

    if best_oos:
        print(f"\n  BEST BY OOS PF IMPROVEMENT: min_bars={best_oos['min_bars']}, min_loss={best_oos['min_loss']}")
        for period in ["FULL", "IS", "OOS"]:
            sc = best_oos[period]["score"]
            fa = best_oos[period]["flip_analysis"]
            bl = baselines[period]["score"]
            if sc and bl:
                pf_d = sc["pf"] - bl["pf"]
                net_d = sc["net_dollar"] - bl["net_dollar"]
                print(f"    {period:>4}: PF {sc['pf']:.3f} ({pf_d:>+.3f}), "
                      f"${sc['net_dollar']:+.2f} (${net_d:>+.2f}), "
                      f"Sharpe {sc['sharpe']:.3f}, Flips: {fa['flip_count']}")

    if best_stable:
        print(f"\n  BEST BY IS/OOS STABILITY: min_bars={best_stable['min_bars']}, min_loss={best_stable['min_loss']}")
        for period in ["FULL", "IS", "OOS"]:
            sc = best_stable[period]["score"]
            fa = best_stable[period]["flip_analysis"]
            bl = baselines[period]["score"]
            if sc and bl:
                pf_d = sc["pf"] - bl["pf"]
                print(f"    {period:>4}: PF {sc['pf']:.3f} ({pf_d:>+.3f}), "
                      f"${sc['net_dollar']:+.2f}, Sharpe {sc['sharpe']:.3f}, "
                      f"Flips: {fa['flip_count']}")

    # Final recommendation
    print(f"\n  RECOMMENDATION:")

    # Check if any config improves OOS AND maintains IS performance
    any_improvement = False
    for r in ranked:
        oos_sc = r["OOS"]["score"]
        is_sc = r["IS"]["score"]
        full_sc = r["FULL"]["score"]
        if oos_sc and is_sc and bl_oos and bl_is:
            oos_better = oos_sc["pf"] > bl_oos["pf"]
            is_ok = is_sc["pf"] >= bl_is["pf"] * 0.95  # at least 95% of baseline IS
            if oos_better and is_ok:
                any_improvement = True
                fa = r["FULL"]["flip_analysis"]
                print(f"    min_bars={r['min_bars']}, min_loss={r['min_loss']}: "
                      f"OOS PF +{oos_sc['pf'] - bl_oos['pf']:.3f}, "
                      f"IS PF {'+' if is_sc['pf'] >= bl_is['pf'] else ''}"
                      f"{is_sc['pf'] - bl_is['pf']:.3f}, "
                      f"FULL flips={fa['flip_count']}, "
                      f"flip WR={fa['flip_win_rate']:.1f}%")

    if not any_improvement:
        print("    NO CONFIG IMPROVES OOS PF while maintaining IS performance.")
        print("    The conditional flip does NOT add value to MES v2.")
        print("    Recommendation: KEEP current behavior (ignore counter-signals).")
    else:
        print("\n    Configs above show OOS improvement with stable IS.")
        print("    Consider paper trading the best config for validation.")

    print(f"\n{'='*110}")
    print("SWEEP COMPLETE")
    print(f"{'='*110}")


if __name__ == "__main__":
    run_sweep()
