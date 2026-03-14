"""
Structure-Based Exit Research — Shared Infrastructure
======================================================
Phase A of the structure-based exit research plan.

Provides:
  - compute_swing_levels()          — pre-compute swing high/low arrays
  - run_backtest_structure_exit()   — MES v2 partial exit with structure-based runner exit

No existing files are modified. All new code lives here.

Usage:
    from structure_exit_common import compute_swing_levels, run_backtest_structure_exit
"""

import numpy as np

import sys
from pathlib import Path

_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR.parent))  # backtesting_engine/
sys.path.insert(0, str(_STRAT_DIR))         # strategies/

from v10_test_common import (
    compute_et_minutes,
    NY_OPEN_ET,
    NY_CLOSE_ET,
)


# ---------------------------------------------------------------------------
# Swing Level Computation (Pre-Pass)
# ---------------------------------------------------------------------------

def compute_swing_levels(highs, lows, lookback, swing_type="pivot",
                         pivot_right=2):
    """Pre-compute swing high/low arrays with no look-ahead bias.

    For each bar i, swing_highs[i] and swing_lows[i] represent the most
    recent CONFIRMED swing level as of bar i.

    Args:
        highs:      numpy array of high prices
        lows:       numpy array of low prices
        lookback:   For pivot: number of bars on the LEFT of the swing point.
                    For donchian: rolling window lookback.
        swing_type: "pivot" or "donchian"
        pivot_right: Confirmation bars to the RIGHT (pivot type only).
                     A swing at bar j is confirmed at bar j + pivot_right.

    Returns:
        (swing_highs, swing_lows): arrays of same length as input.
        NaN where no swing has been confirmed yet.

    CRITICAL: No look-ahead bias.
      - Pivot: A swing high at bar j requires highs[j] >= highs[j-k] for
        k in 1..lookback AND highs[j] >= highs[j+k] for k in 1..pivot_right.
        Confirmed at bar j + pivot_right. Not visible before that bar.
      - Donchian: swing_highs[i] = max(highs[i-lookback : i]) — exclusive
        of bar i to avoid look-ahead.
    """
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    if swing_type == "pivot":
        _compute_pivot_swings(highs, lows, lookback, pivot_right,
                              swing_highs, swing_lows)
    elif swing_type == "donchian":
        _compute_donchian_swings(highs, lows, lookback,
                                 swing_highs, swing_lows)
    else:
        raise ValueError(f"Unknown swing_type: {swing_type}")

    return swing_highs, swing_lows


def _compute_pivot_swings(highs, lows, left, right, swing_highs, swing_lows):
    """Pivot-based swing detection.

    A swing high at bar j requires:
      highs[j] >= highs[j-k] for k = 1..left
      highs[j] >= highs[j+k] for k = 1..right
    Confirmed at bar j + right.

    We iterate candidate bars j from left..(n-right), check the conditions,
    and then forward-fill confirmed levels from confirmation_bar onward.
    """
    n = len(highs)
    last_sh = np.nan
    last_sl = np.nan

    # We need to process candidates in order and track when they're confirmed.
    # A candidate at bar j is confirmed at bar j + right.
    # We fill swing_highs[confirm_bar:] with the level, but only if no later
    # swing has been confirmed by then.

    # Strategy: iterate bar i from 0 to n-1. At each bar i, check if
    # bar j = i - right is a valid swing point (if j >= left).
    # If so, update last_sh/last_sl. Then fill swing_highs[i] = last_sh.

    for i in range(n):
        j = i - right  # candidate bar being confirmed at bar i

        if j >= left:
            # Check if bar j is a swing high
            is_sh = True
            for k in range(1, left + 1):
                if highs[j - k] > highs[j]:
                    is_sh = False
                    break
            if is_sh:
                for k in range(1, right + 1):
                    if highs[j + k] > highs[j]:
                        is_sh = False
                        break
            if is_sh:
                last_sh = highs[j]

            # Check if bar j is a swing low
            is_sl = True
            for k in range(1, left + 1):
                if lows[j - k] < lows[j]:
                    is_sl = False
                    break
            if is_sl:
                for k in range(1, right + 1):
                    if lows[j + k] < lows[j]:
                        is_sl = False
                        break
            if is_sl:
                last_sl = lows[j]

        swing_highs[i] = last_sh
        swing_lows[i] = last_sl


def _compute_donchian_swings(highs, lows, lookback, swing_highs, swing_lows):
    """Donchian-based swing levels.

    swing_highs[i] = max(highs[i-lookback : i])  — exclusive of bar i
    swing_lows[i]  = min(lows[i-lookback : i])    — exclusive of bar i

    For i < lookback, uses whatever history is available (min 1 bar).
    For i = 0, no prior bar exists so result is NaN.
    """
    n = len(highs)
    for i in range(n):
        start = max(0, i - lookback)
        if start >= i:
            # No prior bars available (i == 0)
            swing_highs[i] = np.nan
            swing_lows[i] = np.nan
        else:
            swing_highs[i] = np.max(highs[start:i])
            swing_lows[i] = np.min(lows[start:i])


# ---------------------------------------------------------------------------
# MES v2 Partial Exit Backtest with Structure-Based Runner Exit
# ---------------------------------------------------------------------------

def run_backtest_structure_exit(
    opens, highs, lows, closes, sm, times,
    rsi_5m_curr, rsi_5m_prev,
    rsi_buy, rsi_sell, sm_threshold,
    cooldown_bars, max_loss_pts,
    # Scalp leg (TP1) — unchanged from production
    tp1_pts,
    # Runner leg — structure-based exit
    swing_highs, swing_lows,
    swing_buffer_pts=0,
    min_profit_pts=0,
    use_high_low=False,
    max_tp2_pts=0,
    # Standard exits
    move_sl_to_be_after_tp1=True,
    breakeven_after_bars=0,
    eod_minutes_et=NY_CLOSE_ET,
    entry_end_et=14 * 60 + 15,
    entry_gate=None,
):
    """MES v2 partial exit backtest with structure-based runner exit.

    Entry logic: IDENTICAL to MES v2 production.
      SM(20/12/400/255), RSI(12/55/45), cooldown=25.
      Signal from bar[i-1], fill at bar[i] open.

    Exit logic (2-contract partial):
      Entry: 2 contracts.
      Leg 1 (scalp): TP1 or SL or EOD — UNCHANGED from production.
      Leg 2 (runner): Structure exit + SL + BE_TIME + EOD.
        - Structure exit: For LONG, exit when closes[i-1] >= swing_high - buffer
          (or highs[i-1] if use_high_low). Only if profit >= min_profit_pts.
        - Optional max_tp2_pts as hard cap (0 = disabled).
        - SL-to-BE after TP1: runner SL moves from max_loss_pts to 0 (entry price).
        - BE_TIME: close stale runner after N bars.
        - EOD: close at bar close when bar_et >= eod_minutes_et.

    Args:
        opens, highs, lows, closes, sm, times: price arrays
        rsi_5m_curr, rsi_5m_prev: mapped 5-min RSI arrays
        rsi_buy, rsi_sell: RSI thresholds
        sm_threshold: SM threshold for entry
        cooldown_bars: cooldown between trades
        max_loss_pts: stop loss in points
        tp1_pts: scalp leg TP (fixed, same as production)
        swing_highs: pre-computed swing high array (from compute_swing_levels)
        swing_lows: pre-computed swing low array (from compute_swing_levels)
        swing_buffer_pts: exit N pts before the swing level (0 = at level)
        min_profit_pts: only structure-exit if runner profit >= this
        use_high_low: True = compare H/L to level; False = compare close
        max_tp2_pts: hard cap on runner TP (0 = disabled, structure-only)
        move_sl_to_be_after_tp1: move runner SL to entry after TP1 fills
        breakeven_after_bars: close stale runner after N bars (0 = disabled)
        eod_minutes_et: EOD close time in ET minutes
        entry_end_et: last entry time in ET minutes
        entry_gate: boolean array, True = entry allowed (uses i-1)

    Returns:
        List of trade dicts, TWO per entry (one TP1 scalp, one runner).
        Each has: side, entry_idx, exit_idx, pts, entry_price, exit_price,
                  exit_reason, entry_time, exit_time, bars, leg.
        Compatible with score_trades (uses 'pts' and 'result' fields).
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

    def record_leg(side, entry_p, exit_p, entry_i, exit_i, reason, leg_name):
        """Record a single leg as a trade dict."""
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side,
            "entry": entry_p,
            "exit": exit_p,
            "pts": pts,
            "entry_time": times[entry_i],
            "exit_time": times[exit_i],
            "entry_idx": entry_i,
            "exit_idx": exit_i,
            "bars": exit_i - entry_i,
            "result": reason,
            "leg": leg_name,
            "entry_price": entry_p,
            "exit_price": exit_p,
            "exit_reason": reason,
        })

    def finalize_both_legs(side, entry_p, entry_i,
                           l1_price, l1_idx, l1_reason,
                           l2_price, l2_idx, l2_reason):
        """Record both legs as separate trades."""
        record_leg(side, entry_p, l1_price, entry_i, l1_idx, l1_reason, "scalp")
        record_leg(side, entry_p, l2_price, entry_i, l2_idx, l2_reason, "runner")

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
                finalize_both_legs(
                    side, entry_price, entry_idx,
                    leg1_exit_price, leg1_exit_idx, leg1_exit_reason,
                    leg2_exit_price, i, leg2_exit_reason,
                )
                trade_state = 0
                exit_bar = i
            continue

        # --- Exits for LONG positions ---
        if trade_state == 1:
            # Check Leg 1 (scalp) exits
            if leg1_active:
                # SL: prev bar close breaches stop
                if closes[i - 1] <= entry_price - max_loss_pts:
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
                    if move_sl_to_be_after_tp1 and leg2_active:
                        runner_sl_pts = 0.0  # BE = entry price

            # Check Leg 2 (runner) exits
            if leg2_active:
                # SL check for runner
                effective_sl = runner_sl_pts
                if effective_sl > 0:
                    sl_hit = closes[i - 1] <= entry_price - effective_sl
                elif effective_sl == 0.0 and move_sl_to_be_after_tp1 and not leg1_active and leg1_exit_reason == "TP1":
                    # BE stop: close breaches entry price
                    sl_hit = closes[i - 1] <= entry_price
                else:
                    sl_hit = closes[i - 1] <= entry_price - max_loss_pts

                if sl_hit:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "SL" if runner_sl_pts == max_loss_pts else "BE"
                    leg2_active = False

                # Hard TP2 cap (if configured)
                elif max_tp2_pts > 0 and closes[i - 1] >= entry_price + max_tp2_pts:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "TP_cap"
                    leg2_active = False

                # Structure exit: price approaches swing high
                elif not np.isnan(swing_highs[i - 1]):
                    runner_profit = closes[i - 1] - entry_price
                    if runner_profit >= min_profit_pts:
                        target = swing_highs[i - 1] - swing_buffer_pts
                        if use_high_low:
                            price_check = highs[i - 1]
                        else:
                            price_check = closes[i - 1]
                        if price_check >= target:
                            leg2_exit_price = opens[i]
                            leg2_exit_reason = "structure"
                            leg2_active = False

                # BE_TIME: runner held too long
                if leg2_active and breakeven_after_bars > 0 and not leg1_active:
                    bars_since_entry = (i - 1) - entry_idx
                    if bars_since_entry >= breakeven_after_bars:
                        leg2_exit_price = opens[i]
                        leg2_exit_reason = "BE_TIME"
                        leg2_active = False

            # If both legs closed
            if not leg1_active and not leg2_active:
                finalize_both_legs(
                    "long", entry_price, entry_idx,
                    leg1_exit_price, leg1_exit_idx, leg1_exit_reason,
                    leg2_exit_price, i, leg2_exit_reason,
                )
                trade_state = 0
                exit_bar = i
                continue

        # --- Exits for SHORT positions ---
        elif trade_state == -1:
            if leg1_active:
                # SL
                if closes[i - 1] >= entry_price + max_loss_pts:
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
                    if move_sl_to_be_after_tp1 and leg2_active:
                        runner_sl_pts = 0.0

            if leg2_active:
                effective_sl = runner_sl_pts
                if effective_sl > 0:
                    sl_hit = closes[i - 1] >= entry_price + effective_sl
                elif effective_sl == 0.0 and move_sl_to_be_after_tp1 and not leg1_active and leg1_exit_reason == "TP1":
                    sl_hit = closes[i - 1] >= entry_price
                else:
                    sl_hit = closes[i - 1] >= entry_price + max_loss_pts

                if sl_hit:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "SL" if runner_sl_pts == max_loss_pts else "BE"
                    leg2_active = False

                # Hard TP2 cap
                elif max_tp2_pts > 0 and closes[i - 1] <= entry_price - max_tp2_pts:
                    leg2_exit_price = opens[i]
                    leg2_exit_reason = "TP_cap"
                    leg2_active = False

                # Structure exit: price approaches swing low
                elif not np.isnan(swing_lows[i - 1]):
                    runner_profit = entry_price - closes[i - 1]
                    if runner_profit >= min_profit_pts:
                        target = swing_lows[i - 1] + swing_buffer_pts
                        if use_high_low:
                            price_check = lows[i - 1]
                        else:
                            price_check = closes[i - 1]
                        if price_check <= target:
                            leg2_exit_price = opens[i]
                            leg2_exit_reason = "structure"
                            leg2_active = False

                # BE_TIME
                if leg2_active and breakeven_after_bars > 0 and not leg1_active:
                    bars_since_entry = (i - 1) - entry_idx
                    if bars_since_entry >= breakeven_after_bars:
                        leg2_exit_price = opens[i]
                        leg2_exit_reason = "BE_TIME"
                        leg2_active = False

            if not leg1_active and not leg2_active:
                finalize_both_legs(
                    "short", entry_price, entry_idx,
                    leg1_exit_price, leg1_exit_idx, leg1_exit_reason,
                    leg2_exit_price, i, leg2_exit_reason,
                )
                trade_state = 0
                exit_bar = i
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
                    runner_sl_pts = max_loss_pts  # initial SL for runner
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
                    runner_sl_pts = max_loss_pts
                    leg1_pts = 0.0
                    leg1_exit_reason = ""
                    leg2_exit_price = 0.0
                    leg2_exit_reason = ""

    return trades


# ---------------------------------------------------------------------------
# Scoring for structure exit trades (2 legs per entry = 2 trades)
# ---------------------------------------------------------------------------

def score_structure_trades(trades, dollar_per_pt=5.0, commission_per_side=1.25):
    """Score structure exit trades where each entry produces 2 trade dicts.

    Groups trades by entry_idx to pair scalp + runner legs.
    P&L per leg: pts * dollar_per_pt - 2 * commission_per_side
    Total per entry: scalp_pnl + runner_pnl

    Returns dict with aggregate metrics + exit reason breakdown.
    """
    if not trades:
        return None

    comm_per_leg = commission_per_side * 2  # entry + exit per contract

    # Group by entry_idx to pair legs
    entries = {}
    for t in trades:
        eidx = t["entry_idx"]
        if eidx not in entries:
            entries[eidx] = []
        entries[eidx].append(t)

    pnl_list = []
    exit_reasons = {}
    scalp_reasons = {}
    runner_reasons = {}

    for eidx in sorted(entries.keys()):
        legs = entries[eidx]
        total_pnl = 0.0
        for leg in legs:
            leg_pnl = leg["pts"] * dollar_per_pt - comm_per_leg
            total_pnl += leg_pnl

            reason = leg.get("exit_reason", leg.get("result", "?"))
            leg_type = leg.get("leg", "?")

            if leg_type == "scalp":
                scalp_reasons[reason] = scalp_reasons.get(reason, 0) + 1
            elif leg_type == "runner":
                runner_reasons[reason] = runner_reasons.get(reason, 0) + 1
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        pnl_list.append(total_pnl)

    pnl_arr = np.array(pnl_list)
    n_entries = len(pnl_arr)
    total_pnl = pnl_arr.sum()
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n_entries * 100

    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()

    # Sharpe
    sharpe = 0.0
    if n_entries > 1 and np.std(pnl_arr) > 0:
        sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)

    # Average bars from runner leg (last exit)
    avg_bars_list = []
    for eidx in sorted(entries.keys()):
        legs = entries[eidx]
        max_exit = max(leg["exit_idx"] for leg in legs)
        avg_bars_list.append(max_exit - eidx)
    avg_bars = np.mean(avg_bars_list) if avg_bars_list else 0

    return {
        "count": n_entries,
        "net_dollar": round(total_pnl, 2),
        "pf": round(pf, 3),
        "win_rate": round(wr, 1),
        "max_dd_dollar": round(mdd, 2),
        "avg_bars": round(avg_bars, 1),
        "sharpe": round(sharpe, 3),
        "scalp_exits": scalp_reasons,
        "runner_exits": runner_reasons,
        "all_exits": exit_reasons,
        "pnl_array": pnl_arr,
    }
