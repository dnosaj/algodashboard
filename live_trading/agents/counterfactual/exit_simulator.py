"""Entry-agnostic exit simulation for counterfactual trades.

Ports the inner exit loops from generate_session.py (single-leg) and
vscalpc_partial_exit_sweep.py (2-leg partial) into standalone functions
that take a known entry point and walk forward to compute the exit.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CfResult:
    """Result of a counterfactual exit simulation."""
    exit_price: float
    exit_reason: str       # "TP", "SL", "EOD", "BE_TIME", "TP1+TP2", "TP1+SL_BE", etc.
    pnl_pts: float         # Combined for partial (both legs), NET of commission
    bars_held: int
    mfe_pts: float         # Max favorable excursion (uses H/L)
    mae_pts: float         # Max adverse excursion (uses H/L)
    leg1_exit_reason: str | None = None   # For partial strategies
    leg2_exit_reason: str | None = None


def _time_to_minutes(time_str: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


def simulate_single_exit(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, et_mins: np.ndarray,
    entry_idx: int, entry_price: float, side: str,
    tp_pts: float, sl_pts: float, eod_et: str,
    be_time_bars: int = 0,
    commission_pts: float = 0.0,
) -> CfResult | None:
    """Simulate exit for a single-leg (single-contract) strategy.

    Exit priority per bar (matches backtest):
      1. EOD: bar_mins >= eod → fill at bar close
      2. SL: prev bar close breaches stop → fill at next open
      3. TP: prev bar close reaches target → fill at next open
      4. BE_TIME: bars_held >= threshold → fill at next open

    Returns None if entry_idx is beyond bar data.
    """
    n = len(opens)
    if entry_idx >= n - 1:
        return None

    d = 1 if side == "long" else -1
    eod_mins = _time_to_minutes(eod_et)
    cme_hard_stop = 16 * 60 + 57  # 16:57 ET

    mfe = 0.0
    mae = 0.0

    for i in range(entry_idx + 1, n):
        # Track MFE/MAE using H/L
        if d == 1:
            mfe = max(mfe, highs[i] - entry_price)
            mae = max(mae, entry_price - lows[i])
        else:
            mfe = max(mfe, entry_price - lows[i])
            mae = max(mae, highs[i] - entry_price)

        bar_et = et_mins[i]

        # 1. EOD close
        if bar_et >= eod_mins or bar_et >= cme_hard_stop:
            pnl = (closes[i] - entry_price) * d - commission_pts
            return CfResult(
                exit_price=closes[i], exit_reason="EOD",
                pnl_pts=round(pnl, 4), bars_held=i - entry_idx,
                mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
            )

        # 2. SL: prev bar close breaches stop
        if sl_pts > 0 and (closes[i - 1] - entry_price) * d <= -sl_pts:
            pnl = (opens[i] - entry_price) * d - commission_pts
            return CfResult(
                exit_price=opens[i], exit_reason="SL",
                pnl_pts=round(pnl, 4), bars_held=i - entry_idx,
                mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
            )

        # 3. TP: prev bar close reaches target
        if tp_pts > 0 and (closes[i - 1] - entry_price) * d >= tp_pts:
            pnl = (opens[i] - entry_price) * d - commission_pts
            return CfResult(
                exit_price=opens[i], exit_reason="TP",
                pnl_pts=round(pnl, 4), bars_held=i - entry_idx,
                mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
            )

        # 4. BE_TIME: stale trade exit
        if be_time_bars > 0:
            bars_in_trade = (i - 1) - entry_idx
            if bars_in_trade >= be_time_bars:
                pnl = (opens[i] - entry_price) * d - commission_pts
                return CfResult(
                    exit_price=opens[i], exit_reason="BE_TIME",
                    pnl_pts=round(pnl, 4), bars_held=i - entry_idx,
                    mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
                )

    # Ran out of bars — force close at last bar
    pnl = (closes[-1] - entry_price) * d - commission_pts
    return CfResult(
        exit_price=closes[-1], exit_reason="DATA_END",
        pnl_pts=round(pnl, 4), bars_held=n - 1 - entry_idx,
        mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
    )


def simulate_partial_exit(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    closes: np.ndarray, et_mins: np.ndarray,
    entry_idx: int, entry_price: float, side: str,
    tp1_pts: float, tp2_pts: float, sl_pts: float, eod_et: str,
    sl_to_be: bool = True, be_time_bars: int = 0,
    commission_pts: float = 0.0,
) -> CfResult | None:
    """Simulate exit for a 2-leg partial exit strategy.

    Leg 1 (scalp): exits at TP1 or SL (SL checked first).
    Leg 2 (runner): exits at TP2, SL/BE, BE_TIME, or EOD.
    After TP1 fill (specifically TP1, not SL), runner SL moves to BE if sl_to_be=True.

    Returns None if entry_idx is beyond bar data.
    """
    n = len(opens)
    if entry_idx >= n - 1:
        return None

    d = 1 if side == "long" else -1
    eod_mins = _time_to_minutes(eod_et)
    cme_hard_stop = 16 * 60 + 57

    leg1_active = True
    leg2_active = True
    runner_sl_pts = sl_pts  # Initial SL for runner (moves to 0 after TP1)

    leg1_pnl = 0.0
    leg2_pnl = 0.0
    leg1_exit_reason = ""
    leg2_exit_reason = ""
    last_exit_idx = entry_idx
    last_exit_price = None  # Track actual fill price (closes for EOD, opens for SL/TP/BE)

    mfe = 0.0
    mae = 0.0

    for i in range(entry_idx + 1, n):
        # Track MFE/MAE
        if d == 1:
            mfe = max(mfe, highs[i] - entry_price)
            mae = max(mae, entry_price - lows[i])
        else:
            mfe = max(mfe, entry_price - lows[i])
            mae = max(mae, highs[i] - entry_price)

        bar_et = et_mins[i]

        # --- EOD close (close all remaining legs at bar close) ---
        if bar_et >= eod_mins or bar_et >= cme_hard_stop:
            if leg1_active:
                leg1_pnl = (closes[i] - entry_price) * d
                leg1_exit_reason = "EOD"
                leg1_active = False
            if leg2_active:
                leg2_pnl = (closes[i] - entry_price) * d
                leg2_exit_reason = "EOD"
                leg2_active = False
                last_exit_idx = i
                last_exit_price = closes[i]
            break

        # --- Leg 1 exits ---
        if leg1_active:
            prev_move = (closes[i - 1] - entry_price) * d

            # SL: prev bar close breaches stop (checked FIRST)
            if sl_pts > 0 and prev_move <= -sl_pts:
                leg1_pnl = (opens[i] - entry_price) * d
                leg1_exit_reason = "SL"
                leg1_active = False
            # TP1: prev bar close reaches target
            elif tp1_pts > 0 and prev_move >= tp1_pts:
                leg1_pnl = (opens[i] - entry_price) * d
                leg1_exit_reason = "TP1"
                leg1_active = False
                # Move runner SL to breakeven
                if sl_to_be and leg2_active:
                    runner_sl_pts = 0.0

        # --- Leg 2 exits ---
        if leg2_active:
            prev_move = (closes[i - 1] - entry_price) * d

            # Determine effective SL for runner
            if runner_sl_pts > 0:
                sl_hit = prev_move <= -runner_sl_pts
            elif runner_sl_pts == 0.0 and sl_to_be and not leg1_active and leg1_exit_reason == "TP1":
                # BE stop: price breaches entry
                sl_hit = prev_move <= 0
            else:
                sl_hit = prev_move <= -sl_pts

            if sl_hit:
                leg2_pnl = (opens[i] - entry_price) * d
                leg2_exit_reason = "SL" if runner_sl_pts == sl_pts else "BE"
                leg2_active = False
                last_exit_idx = i
                last_exit_price = opens[i]
            # TP2: prev bar close reaches runner target
            elif tp2_pts > 0 and prev_move >= tp2_pts:
                leg2_pnl = (opens[i] - entry_price) * d
                leg2_exit_reason = "TP2"
                leg2_active = False
                last_exit_idx = i
                last_exit_price = opens[i]

        # --- BE_TIME: close all remaining legs when bars >= threshold ---
        if be_time_bars > 0:
            bars_since_entry = (i - 1) - entry_idx
            if bars_since_entry >= be_time_bars:
                if leg1_active:
                    leg1_pnl = (opens[i] - entry_price) * d
                    leg1_exit_reason = "BE_TIME"
                    leg1_active = False
                if leg2_active:
                    leg2_pnl = (opens[i] - entry_price) * d
                    leg2_exit_reason = "BE_TIME"
                    leg2_active = False
                    last_exit_idx = i
                    last_exit_price = opens[i]

        # Both legs closed
        if not leg1_active and not leg2_active:
            break

    # Handle case where we ran out of bars
    if leg1_active:
        leg1_pnl = (closes[-1] - entry_price) * d
        leg1_exit_reason = "DATA_END"
    if leg2_active:
        leg2_pnl = (closes[-1] - entry_price) * d
        leg2_exit_reason = "DATA_END"
        last_exit_idx = n - 1
        last_exit_price = closes[-1]

    total_pnl = leg1_pnl + leg2_pnl - commission_pts
    exit_reason = f"{leg1_exit_reason}+{leg2_exit_reason}"
    exit_price = last_exit_price if last_exit_price is not None else closes[-1]

    return CfResult(
        exit_price=exit_price, exit_reason=exit_reason,
        pnl_pts=round(total_pnl, 4), bars_held=last_exit_idx - entry_idx,
        mfe_pts=round(mfe, 4), mae_pts=round(mae, 4),
        leg1_exit_reason=leg1_exit_reason,
        leg2_exit_reason=leg2_exit_reason,
    )
