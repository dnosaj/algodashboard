"""
ADR Exhaustion Filter — Shared Computation Module
====================================================
Computes Average Daily Range (ADR) and intraday session tracking,
then builds entry gates based on range consumption and directionality.

4 gate types:
  Layer 1: Basic range gate — block when today_range / ADR >= threshold
  Layer 2: Directional gate — block longs when rally / ADR >= threshold (and vice versa)
  Layer 3: Combined — require BOTH range consumed AND directional move
  Layer 4: Remaining range — block when ADR - today_range < TP target
"""

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


# ============================================================================
# Session Tracking (intraday running high/low/open)
# ============================================================================

def compute_session_tracking(df_1m):
    """Compute intraday session tracking on RTH bars (10:00-16:00 ET).

    For each 1-min bar, tracks:
      - session_open:  open price at 10:00 ET
      - session_high:  running high since 10:00 ET
      - session_low:   running low since 10:00 ET
      - today_range:   session_high - session_low
      - move_from_open: close - session_open (signed)

    Pre-RTH bars (overnight) carry NaN — they can't fire entries anyway.

    Returns:
        dict of numpy arrays, each length len(df_1m).
    """
    n = len(df_1m)
    idx = pd.DatetimeIndex(df_1m.index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    et = idx.tz_convert(_ET)
    et_dates = et.date
    et_mins = (et.hour * 60 + et.minute).values.astype(np.int32)

    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values

    session_open = np.full(n, np.nan)
    session_high = np.full(n, np.nan)
    session_low = np.full(n, np.nan)
    today_range = np.full(n, np.nan)
    move_from_open = np.full(n, np.nan)

    RTH_START = 600   # 10:00 ET
    RTH_END = 960     # 16:00 ET

    curr_date = None
    curr_open = np.nan
    curr_high = np.nan
    curr_low = np.nan

    for i in range(n):
        d = et_dates[i]
        m = et_mins[i]

        # Only track during RTH
        if m < RTH_START or m >= RTH_END:
            continue

        # New RTH session?
        if d != curr_date:
            curr_date = d
            curr_open = opens[i]
            curr_high = highs[i]
            curr_low = lows[i]
        else:
            if highs[i] > curr_high:
                curr_high = highs[i]
            if lows[i] < curr_low:
                curr_low = lows[i]

        session_open[i] = curr_open
        session_high[i] = curr_high
        session_low[i] = curr_low
        today_range[i] = curr_high - curr_low
        move_from_open[i] = closes[i] - curr_open

    return {
        'session_open': session_open,
        'session_high': session_high,
        'session_low': session_low,
        'today_range': today_range,
        'move_from_open': move_from_open,
    }


# ============================================================================
# Average Daily Range (ADR)
# ============================================================================

def compute_adr(df_1m, lookback_days=14):
    """Compute Average Daily Range mapped to each 1-min bar.

    ADR = rolling mean of prior N completed RTH daily ranges.
    No look-ahead: today's ADR uses only yesterday and earlier.

    Args:
        df_1m: 1-min DataFrame with High, Low columns.
        lookback_days: rolling window (trading days).

    Returns:
        numpy array of length len(df_1m) with ADR values.
        NaN for bars before enough history.
    """
    idx = pd.DatetimeIndex(df_1m.index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    et = idx.tz_convert(_ET)
    et_dates = et.date
    et_mins = (et.hour * 60 + et.minute).values.astype(np.int32)

    highs = df_1m['High'].values
    lows = df_1m['Low'].values

    RTH_START = 600
    RTH_END = 960

    # Pass 1: compute RTH daily range per date
    daily_ranges = {}
    for d in np.unique(et_dates):
        mask = (et_dates == d) & (et_mins >= RTH_START) & (et_mins < RTH_END)
        if mask.sum() == 0:
            continue
        daily_ranges[d] = highs[mask].max() - lows[mask].min()

    date_list = sorted(daily_ranges.keys())
    n_days = len(date_list)

    # Pass 2: rolling mean of prior lookback_days daily ranges
    day_to_adr = {}
    for i_date in range(1, n_days):
        # ADR for today uses days [i_date-lookback_days, i_date-1]
        start = max(0, i_date - lookback_days)
        end = i_date  # exclusive — does NOT include today
        if end - start < lookback_days:
            continue  # not enough history
        window = [daily_ranges[date_list[j]] for j in range(start, end)]
        day_to_adr[date_list[i_date]] = np.mean(window)

    # Pass 3: map to 1-min bars
    n = len(df_1m)
    result = np.full(n, np.nan)
    for d, adr_val in day_to_adr.items():
        mask = et_dates == d
        result[mask] = adr_val

    return result


# ============================================================================
# Gate Builders
# ============================================================================

def build_range_gate(today_range, adr, threshold):
    """Layer 1: Block all entries when today_range / ADR >= threshold.

    Returns boolean array: True = allow, False = block.
    NaN values -> True (fail-open).
    """
    n = len(today_range)
    gate = np.ones(n, dtype=bool)
    for i in range(n):
        if np.isnan(today_range[i]) or np.isnan(adr[i]) or adr[i] <= 0:
            continue  # fail-open
        if today_range[i] / adr[i] >= threshold:
            gate[i] = False
    return gate


def build_directional_gate(move_from_open, adr, sm, threshold):
    """Layer 2: Block entries chasing the daily move direction.

    Block LONGS when move_from_open / ADR >= +threshold  (already rallied)
    Block SHORTS when move_from_open / ADR <= -threshold (already sold off)

    SM sign determines entry direction:
      sm > 0 -> potential long, sm < 0 -> potential short, sm == 0 -> allow.

    Uses sm_threshold=0.0 (any positive SM is potential long). This is
    slightly over-conservative for vScalpB (sm_threshold=0.25) but those
    bars wouldn't trigger entries anyway — the backtest SM check catches them.

    Returns boolean array: True = allow, False = block.
    NaN values -> True (fail-open).
    """
    n = len(move_from_open)
    gate = np.ones(n, dtype=bool)
    for i in range(n):
        if np.isnan(move_from_open[i]) or np.isnan(adr[i]) or adr[i] <= 0:
            continue  # fail-open
        ratio = move_from_open[i] / adr[i]
        if sm[i] > 0 and ratio >= threshold:
            gate[i] = False  # block longs — already rallied
        elif sm[i] < 0 and ratio <= -threshold:
            gate[i] = False  # block shorts — already sold off
    return gate


def build_combined_gate(today_range, move_from_open, adr, sm,
                        range_threshold, dir_threshold):
    """Layer 3: Require BOTH range consumed AND directional move.

    Block LONGS when:
      today_range / ADR >= range_threshold AND move_from_open / ADR >= +dir_threshold
    Block SHORTS when:
      today_range / ADR >= range_threshold AND move_from_open / ADR <= -dir_threshold

    Returns boolean array: True = allow, False = block.
    """
    n = len(today_range)
    gate = np.ones(n, dtype=bool)
    for i in range(n):
        if (np.isnan(today_range[i]) or np.isnan(move_from_open[i])
                or np.isnan(adr[i]) or adr[i] <= 0):
            continue  # fail-open
        range_ratio = today_range[i] / adr[i]
        dir_ratio = move_from_open[i] / adr[i]
        if range_ratio >= range_threshold:
            if sm[i] > 0 and dir_ratio >= dir_threshold:
                gate[i] = False
            elif sm[i] < 0 and dir_ratio <= -dir_threshold:
                gate[i] = False
    return gate


def build_remaining_range_gate(today_range, adr, tp_pts):
    """Layer 4: Block when remaining range < TP target.

    remaining = ADR - today_range
    Block when remaining < tp_pts.

    Returns boolean array: True = allow, False = block.
    NaN values -> True (fail-open).
    """
    n = len(today_range)
    gate = np.ones(n, dtype=bool)
    for i in range(n):
        if np.isnan(today_range[i]) or np.isnan(adr[i]) or adr[i] <= 0:
            continue  # fail-open
        remaining = adr[i] - today_range[i]
        if remaining < tp_pts:
            gate[i] = False
    return gate
