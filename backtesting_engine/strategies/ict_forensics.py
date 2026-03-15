"""
ICT Trade Forensics — Map backtest trades against ICT structural features.

Computes Order Blocks, FVGs, BOS/MSS, Fibonacci/OTE, and Liquidity Sweeps
on both 1-min and 5-min bars, then maps each portfolio trade to its ICT
structural context at entry time.

Output: WR/PF breakdown by ICT feature, SL clustering, confluence analysis.

Usage:
    cd backtesting_engine/strategies
    python3 ict_forensics.py
"""

import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ET_TZ = ZoneInfo("America/New_York")

# --- Path setup (same as run_and_save_portfolio.py) ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR))
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    compute_et_minutes,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
)

from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS, VSCALPA_ENTRY_END_ET,
    VSCALPB_RSI_LEN, VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
    VSCALPB_SM_THRESHOLD, VSCALPB_COOLDOWN,
    VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
    MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA,
    MES_DOLLAR_PER_PT, MES_COMMISSION,
    MESV2_RSI_LEN, MESV2_RSI_BUY, MESV2_RSI_SELL,
    MESV2_SM_THRESHOLD, MESV2_COOLDOWN,
    MESV2_MAX_LOSS_PTS, MESV2_TP_PTS, MESV2_BREAKEVEN_BARS,
)

from sr_leledc_exhaustion_sweep import compute_leledc_exhaustion, build_leledc_gate
from adr_common import compute_session_tracking, compute_adr, build_directional_gate
from htf_common import compute_prior_day_atr
from sr_prior_day_levels_sweep import (
    compute_rth_volume_profile, build_prior_day_level_gate, _compute_value_area,
)
from v10_test_common import NY_OPEN_ET, NY_CLOSE_ET
from structure_exit_common import (
    compute_swing_levels,
    run_backtest_structure_exit,
    score_structure_trades,
)
from run_and_save_portfolio import (
    load_vix_gate,
    _get_bar_dates_et,
    VSCALPC_TP1_PTS, VSCALPC_MAX_TP2_PTS, VSCALPC_SL_PTS,
    VSCALPC_SWING_LB, VSCALPC_SWING_PR, VSCALPC_SWING_BUF,
    MESV2_EOD_ET, MESV2_ENTRY_END_ET,
)
from v10_test_common import compute_prior_day_levels


# ============================================================================
# Part 1: ICT Feature Computation Functions
# ============================================================================

# ---------------------------------------------------------------------------
# 1a. Order Blocks (UAlgo 3-bar engulfing pattern)
# ---------------------------------------------------------------------------

def compute_order_blocks(opens, highs, lows, closes):
    """Detect order blocks using UAlgo's 3-bar engulfing pattern.

    Bullish OB at bar i:
      bar[i-2] bearish, bar[i-1] bullish engulfing (sweeps below i-2 low),
      bar[i] bullish confirming (closes above i-1 high) = displacement.
      OB zone = [low[i-1], high[i-1]].

    Bearish OB at bar i:
      bar[i-2] bullish, bar[i-1] bearish engulfing (sweeps above i-2 high),
      bar[i] bearish confirming (closes below i-1 low) = displacement.
      OB zone = [low[i-1], high[i-1]].

    Returns:
      ob_events: list of dicts with keys:
        bar_idx, direction ('bull'/'bear'), zone_low, zone_high, mitigated_at
    """
    n = len(opens)
    ob_events = []

    for i in range(2, n):
        # Bullish OB
        bar2_bearish = opens[i - 2] > closes[i - 2]
        bar1_bullish = closes[i - 1] > opens[i - 1]
        bar0_bullish = closes[i] > opens[i]
        sweeps_below = lows[i - 1] < lows[i - 2]
        displacement_up = closes[i] > highs[i - 1]

        if bar2_bearish and bar1_bullish and bar0_bullish and sweeps_below and displacement_up:
            ob_events.append({
                'bar_idx': i,
                'direction': 'bull',
                'zone_low': lows[i - 1],
                'zone_high': highs[i - 1],
                'mitigated_at': -1,
            })

        # Bearish OB
        bar2_bullish = opens[i - 2] < closes[i - 2]
        bar1_bearish = closes[i - 1] < opens[i - 1]
        bar0_bearish = closes[i] < opens[i]
        sweeps_above = highs[i - 1] > highs[i - 2]
        displacement_down = closes[i] < lows[i - 1]

        if bar2_bullish and bar1_bearish and bar0_bearish and sweeps_above and displacement_down:
            ob_events.append({
                'bar_idx': i,
                'direction': 'bear',
                'zone_low': lows[i - 1],
                'zone_high': highs[i - 1],
                'mitigated_at': -1,
            })

    # Track mitigation: OB is mitigated when close passes through the zone
    for ob in ob_events:
        for j in range(ob['bar_idx'] + 1, n):
            if ob['direction'] == 'bull':
                # Bullish OB mitigated when close goes below zone low
                if closes[j] < ob['zone_low']:
                    ob['mitigated_at'] = j
                    break
            else:
                # Bearish OB mitigated when close goes above zone high
                if closes[j] > ob['zone_high']:
                    ob['mitigated_at'] = j
                    break

    return ob_events


# ---------------------------------------------------------------------------
# 1b. Fair Value Gaps
# ---------------------------------------------------------------------------

def compute_fvgs(highs, lows, closes):
    """Detect Fair Value Gaps (3-bar imbalance).

    Bullish FVG at bar i: low[i] > high[i-2] (gap up between candle 1 and 3)
      Zone: [high[i-2], low[i]]
    Bearish FVG at bar i: high[i] < low[i-2] (gap down)
      Zone: [high[i], low[i-2]]

    Returns:
      fvg_events: list of dicts with keys:
        bar_idx, direction ('bull'/'bear'), zone_low, zone_high, mitigated_at
    """
    n = len(highs)
    fvg_events = []

    for i in range(2, n):
        # Bullish FVG
        if lows[i] > highs[i - 2]:
            fvg_events.append({
                'bar_idx': i,
                'direction': 'bull',
                'zone_low': highs[i - 2],
                'zone_high': lows[i],
                'mitigated_at': -1,
            })
        # Bearish FVG
        if highs[i] < lows[i - 2]:
            fvg_events.append({
                'bar_idx': i,
                'direction': 'bear',
                'zone_low': highs[i],
                'zone_high': lows[i - 2],
                'mitigated_at': -1,
            })

    # Track mitigation: FVG filled when price enters the zone
    for fvg in fvg_events:
        for j in range(fvg['bar_idx'] + 1, n):
            if fvg['direction'] == 'bull':
                # Bullish FVG filled when price drops into the gap
                if lows[j] <= fvg['zone_high']:
                    fvg['mitigated_at'] = j
                    break
            else:
                # Bearish FVG filled when price rises into the gap
                if highs[j] >= fvg['zone_low']:
                    fvg['mitigated_at'] = j
                    break

    return fvg_events


# ---------------------------------------------------------------------------
# 1c. BOS/MSS Detection
# ---------------------------------------------------------------------------

def compute_bos_mss(highs, lows, closes, swing_highs, swing_lows):
    """Detect BOS (continuation) and MSS (reversal) events.

    Uses pre-computed swing highs/lows from compute_swing_levels().

    BOS: close breaks swing in trend direction (continuation).
    MSS: close breaks swing against trend (reversal).

    Trend tracking: start neutral, BOS confirms trend, MSS reverses it.

    Returns:
      events: list of dicts with keys:
        bar_idx, event_type ('BOS'/'MSS'), direction ('bull'/'bear'),
        level (the swing level that was broken)
      trend_state: array of per-bar trend: 1=bullish, -1=bearish, 0=neutral
    """
    n = len(closes)
    events = []
    trend_state = np.zeros(n, dtype=int)
    current_trend = 0  # 0=neutral, 1=bull, -1=bear
    last_sh = np.nan
    last_sl = np.nan

    for i in range(1, n):
        # Update tracked swing levels (use i-1 to avoid lookahead)
        if not np.isnan(swing_highs[i - 1]):
            last_sh = swing_highs[i - 1]
        if not np.isnan(swing_lows[i - 1]):
            last_sl = swing_lows[i - 1]

        # Check for structure breaks at bar i using close[i-1]
        broke_high = not np.isnan(last_sh) and closes[i - 1] > last_sh
        broke_low = not np.isnan(last_sl) and closes[i - 1] < last_sl

        if broke_high:
            if current_trend >= 0:
                # BOS: continuation in bullish direction
                events.append({
                    'bar_idx': i,
                    'event_type': 'BOS',
                    'direction': 'bull',
                    'level': last_sh,
                })
                current_trend = 1
            else:
                # MSS: reversal from bearish to bullish
                events.append({
                    'bar_idx': i,
                    'event_type': 'MSS',
                    'direction': 'bull',
                    'level': last_sh,
                })
                current_trend = 1

        if broke_low and not broke_high:
            if current_trend <= 0:
                # BOS: continuation in bearish direction
                events.append({
                    'bar_idx': i,
                    'event_type': 'BOS',
                    'direction': 'bear',
                    'level': last_sl,
                })
                current_trend = -1
            else:
                # MSS: reversal from bullish to bearish
                events.append({
                    'bar_idx': i,
                    'event_type': 'MSS',
                    'direction': 'bear',
                    'level': last_sl,
                })
                current_trend = -1

        trend_state[i] = current_trend

    return events, trend_state


# ---------------------------------------------------------------------------
# 1d. Fibonacci / OTE
# ---------------------------------------------------------------------------

def compute_fib_zones(highs, lows, closes, bos_mss_events, swing_highs, swing_lows):
    """Compute per-bar Fibonacci zone relative to last BOS/MSS.

    After each BOS/MSS:
      a0 = extreme of impulse (tracks with price until next event)
      a1 = the swing point opposite the break (prior swing before break)
      Fib levels: EQ=0.5, OTE zone=0.618-0.79

    For bullish structure: a1 is the swing low before break, a0 is the high.
      Premium = above 0.5 (close > midpoint)
      Discount = below 0.5
      OTE = 0.618 to 0.79 retracement zone

    For a bullish fib range, fib 0.0 = a0 (high), fib 1.0 = a1 (low).
    Price retracing from a0 toward a1 means going from 0.0 toward 1.0.
    OTE is the retracement zone 0.618-0.79 (closer to a1 = deeper pullback).

    Returns:
      fib_position: array of per-bar values:
        0 = no fib context
        1 = premium (above EQ, favorable for shorts)
        2 = discount (below EQ, favorable for longs)
        3 = OTE zone (0.618-0.79 retracement)
      fib_level: array of per-bar fib retracement level (0.0-1.0), NaN if none
      fib_direction: array of per-bar fib structure direction (1=bull, -1=bear, 0=none)
    """
    n = len(closes)
    fib_position = np.zeros(n, dtype=int)
    fib_level = np.full(n, np.nan)
    fib_direction = np.zeros(n, dtype=int)

    # Build event lookup by bar index
    event_bars = set()
    event_by_bar = {}
    for ev in bos_mss_events:
        event_bars.add(ev['bar_idx'])
        event_by_bar[ev['bar_idx']] = ev

    a0 = np.nan  # extreme of impulse (high for bull, low for bear)
    a1 = np.nan  # prior swing (low for bull, high for bear)
    struct_dir = 0  # 1=bull, -1=bear

    for i in range(n):
        # Check if a new structure event happened at this bar
        if i in event_by_bar:
            ev = event_by_bar[i]
            if ev['direction'] == 'bull':
                struct_dir = 1
                a0 = highs[i]  # will track upward
                # a1 = the swing low before the break
                a1 = swing_lows[i - 1] if not np.isnan(swing_lows[i - 1]) else lows[i]
            else:
                struct_dir = -1
                a0 = lows[i]  # will track downward
                a1 = swing_highs[i - 1] if not np.isnan(swing_highs[i - 1]) else highs[i]

        # Update a0 (track extreme with price)
        if struct_dir == 1:
            a0 = max(a0, highs[i]) if not np.isnan(a0) else highs[i]
        elif struct_dir == -1:
            a0 = min(a0, lows[i]) if not np.isnan(a0) else lows[i]

        # Compute fib level for current close
        if struct_dir != 0 and not np.isnan(a0) and not np.isnan(a1) and a0 != a1:
            fib_range = a0 - a1  # positive for bull (high - low)

            if struct_dir == 1:
                # Bullish: fib 0.0 at a0 (high), 1.0 at a1 (low)
                # Retracement = how far price pulled back from a0 toward a1
                retrace = (a0 - closes[i]) / abs(fib_range)
            else:
                # Bearish: fib 0.0 at a0 (low), 1.0 at a1 (high)
                retrace = (closes[i] - a0) / abs(fib_range)

            fib_level[i] = retrace
            fib_direction[i] = struct_dir

            if 0.618 <= retrace <= 0.79:
                fib_position[i] = 3  # OTE
            elif retrace < 0.5:
                fib_position[i] = 1  # Premium (near the extreme, hasn't retraced much)
            else:
                fib_position[i] = 2  # Discount (deep retrace toward the opposite swing)

    return fib_position, fib_level, fib_direction


# ---------------------------------------------------------------------------
# 1e. Liquidity Sweeps
# ---------------------------------------------------------------------------

def compute_liquidity_sweeps(highs, lows, closes, pivot_lookback=20):
    """Detect liquidity sweeps using pivot swing highs/lows.

    Bullish sweep: low < swing_low AND close > swing_low (wick below, close back)
    Bearish sweep: high > swing_high AND close < swing_high (wick above, close back)

    Uses a simple rolling pivot for sweep reference levels.

    Returns:
      sweep_events: list of dicts with bar_idx, direction ('bull'/'bear'), level
      last_sweep_bar: per-bar array of the most recent sweep bar index (-1 if none)
      last_sweep_dir: per-bar array of direction (1=bull, -1=bear, 0=none)
    """
    n = len(highs)
    sweep_events = []

    # Compute pivot highs and lows for sweep reference
    # Using rolling max/min as pivot reference (simpler than full pivot detection)
    for i in range(pivot_lookback, n):
        lookback_start = max(0, i - pivot_lookback)
        pivot_high = np.max(highs[lookback_start:i])
        pivot_low = np.min(lows[lookback_start:i])

        # Bearish sweep: wick above pivot high, close back inside
        if highs[i] > pivot_high and closes[i] < pivot_high:
            sweep_events.append({
                'bar_idx': i,
                'direction': 'bear',
                'level': pivot_high,
            })

        # Bullish sweep: wick below pivot low, close back inside
        if lows[i] < pivot_low and closes[i] > pivot_low:
            sweep_events.append({
                'bar_idx': i,
                'direction': 'bull',
                'level': pivot_low,
            })

    # Build per-bar last sweep tracking
    last_sweep_bar = np.full(n, -1, dtype=int)
    last_sweep_dir = np.zeros(n, dtype=int)
    _last_bar = -1
    _last_dir = 0
    sweep_idx = 0

    for i in range(n):
        while sweep_idx < len(sweep_events) and sweep_events[sweep_idx]['bar_idx'] <= i:
            ev = sweep_events[sweep_idx]
            _last_bar = ev['bar_idx']
            _last_dir = 1 if ev['direction'] == 'bull' else -1
            sweep_idx += 1
        last_sweep_bar[i] = _last_bar
        last_sweep_dir[i] = _last_dir

    return sweep_events, last_sweep_bar, last_sweep_dir


# ---------------------------------------------------------------------------
# 1f. Weekly Volume Profile (VPOC/VAH/VAL)
# ---------------------------------------------------------------------------

def compute_weekly_volume_profile(df_1min, bin_width=5):
    """Compute prior-week VPOC, VAH, VAL from RTH volume profile.

    Groups RTH bars (10:00-16:00 ET) by ISO week. For each completed week,
    computes volume profile. Maps prior-week's levels to each bar of the
    current week (no lookahead).

    Args:
        df_1min: DataFrame with OHLCV columns and DatetimeIndex.
        bin_width: Price bin width for volume profile (default 5 for NQ).

    Returns:
        (weekly_vpoc, weekly_vah, weekly_val) arrays, one value per bar.
        Bars before the first completed week get NaN.
    """
    times = df_1min.index
    closes = df_1min["Close"].values
    volumes = df_1min["Volume"].values
    et_mins = compute_et_minutes(times)
    n = len(times)

    weekly_vpoc = np.full(n, np.nan)
    weekly_vah = np.full(n, np.nan)
    weekly_val = np.full(n, np.nan)

    # Convert timestamps to ET dates for ISO week grouping
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    et_times = idx.tz_convert(_ET_TZ)

    # Pass 1: Collect per-week volume profile (using RTH bars only)
    weekly_profiles = {}  # (iso_year, iso_week) -> (vpoc, vah, val)
    current_week_key = None
    week_closes = []
    week_volumes = []

    for i in range(n):
        bar_et = et_times[i]
        iso_year, iso_week, _ = bar_et.isocalendar()
        week_key = (iso_year, iso_week)

        if week_key != current_week_key:
            # Finalize previous week
            if current_week_key is not None and len(week_closes) > 0:
                profile = _compute_value_area(week_closes, week_volumes, bin_width)
                if profile is not None:
                    weekly_profiles[current_week_key] = profile
            current_week_key = week_key
            week_closes = []
            week_volumes = []

        # Only include RTH bars (10:00-16:00 ET)
        if NY_OPEN_ET <= et_mins[i] < NY_CLOSE_ET:
            week_closes.append(closes[i])
            week_volumes.append(volumes[i])

    # Save last week
    if current_week_key is not None and len(week_closes) > 0:
        profile = _compute_value_area(week_closes, week_volumes, bin_width)
        if profile is not None:
            weekly_profiles[current_week_key] = profile

    # Pass 2: Map prior-week profile to each bar
    sorted_weeks = sorted(weekly_profiles.keys())
    week_to_prev = {}
    for j in range(1, len(sorted_weeks)):
        week_to_prev[sorted_weeks[j]] = weekly_profiles[sorted_weeks[j - 1]]

    for i in range(n):
        bar_et = et_times[i]
        iso_year, iso_week, _ = bar_et.isocalendar()
        week_key = (iso_year, iso_week)
        if week_key in week_to_prev:
            vpoc_p, vah_p, val_p = week_to_prev[week_key]
            weekly_vpoc[i] = vpoc_p
            weekly_vah[i] = vah_p
            weekly_val[i] = val_p

    return weekly_vpoc, weekly_vah, weekly_val


# ---------------------------------------------------------------------------
# 1g. Session Highs/Lows (Asia, London, Pre-NY)
# ---------------------------------------------------------------------------

def compute_session_levels(df_1min):
    """Compute per-bar session highs/lows for Asia, London, and Pre-NY sessions.

    Sessions (all in ET):
        Asia:    20:00 - 00:00 (prior evening)
        London:  02:00 - 05:00
        Pre-NY:  05:00 - 09:30

    For each bar during RTH (09:30+), the most recent completed session's
    levels are mapped. No lookahead: only completed sessions are used.

    Returns:
        dict with arrays:
            asia_high, asia_low,
            london_high, london_low,
            preny_high, preny_low
        Each array has length = len(df_1min), with NaN where no completed
        session is available.
    """
    times = df_1min.index
    highs = df_1min["High"].values
    lows = df_1min["Low"].values
    n = len(times)

    # Convert to ET
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    et_times = idx.tz_convert(_ET_TZ)

    # ET minutes from midnight
    et_mins = compute_et_minutes(times)

    # Session time boundaries in ET minutes
    ASIA_START = 20 * 60   # 20:00
    ASIA_END = 24 * 60     # 00:00 (midnight = next day)
    LONDON_START = 2 * 60  # 02:00
    LONDON_END = 5 * 60    # 05:00
    PRENY_START = 5 * 60   # 05:00
    PRENY_END = 9 * 60 + 30  # 09:30

    # Output arrays
    asia_high_out = np.full(n, np.nan)
    asia_low_out = np.full(n, np.nan)
    london_high_out = np.full(n, np.nan)
    london_low_out = np.full(n, np.nan)
    preny_high_out = np.full(n, np.nan)
    preny_low_out = np.full(n, np.nan)

    # Track current session state per calendar date
    # We track sessions by their "trading date" (the date they apply to)
    # Asia 20:00-00:00 on day D applies to trading day D+1
    # London 02:00-05:00 on day D applies to trading day D
    # Pre-NY 05:00-09:30 on day D applies to trading day D

    # Tracking variables for running session H/L
    asia_h = np.nan
    asia_l = np.nan
    london_h = np.nan
    london_l = np.nan
    preny_h = np.nan
    preny_l = np.nan

    # Completed session levels (these get mapped to subsequent bars)
    completed_asia_h = np.nan
    completed_asia_l = np.nan
    completed_london_h = np.nan
    completed_london_l = np.nan
    completed_preny_h = np.nan
    completed_preny_l = np.nan

    # Track session state transitions
    prev_in_asia = False
    prev_in_london = False
    prev_in_preny = False

    for i in range(n):
        m = et_mins[i]

        # Determine which session this bar belongs to
        in_asia = (m >= ASIA_START)  # 20:00-23:59 ET
        in_london = (LONDON_START <= m < LONDON_END)
        in_preny = (PRENY_START <= m < PRENY_END)

        # Asia session: 20:00-00:00 ET
        # When et_mins wraps (goes from 23:59 to 00:00), Asia ends
        if in_asia:
            if not prev_in_asia:
                # New Asia session starting
                asia_h = highs[i]
                asia_l = lows[i]
            else:
                asia_h = max(asia_h, highs[i])
                asia_l = min(asia_l, lows[i])
        elif prev_in_asia and m < ASIA_START:
            # Asia just ended (transitioned from 23:xx to 00:xx)
            if not np.isnan(asia_h):
                completed_asia_h = asia_h
                completed_asia_l = asia_l

        # London session: 02:00-05:00 ET
        if in_london:
            if not prev_in_london:
                london_h = highs[i]
                london_l = lows[i]
            else:
                london_h = max(london_h, highs[i])
                london_l = min(london_l, lows[i])
        elif prev_in_london and not in_london:
            if not np.isnan(london_h):
                completed_london_h = london_h
                completed_london_l = london_l

        # Pre-NY session: 05:00-09:30 ET
        if in_preny:
            if not prev_in_preny:
                preny_h = highs[i]
                preny_l = lows[i]
            else:
                preny_h = max(preny_h, highs[i])
                preny_l = min(preny_l, lows[i])
        elif prev_in_preny and not in_preny:
            if not np.isnan(preny_h):
                completed_preny_h = preny_h
                completed_preny_l = preny_l

        # Map completed session levels to this bar
        asia_high_out[i] = completed_asia_h
        asia_low_out[i] = completed_asia_l
        london_high_out[i] = completed_london_h
        london_low_out[i] = completed_london_l
        preny_high_out[i] = completed_preny_h
        preny_low_out[i] = completed_preny_l

        prev_in_asia = in_asia
        prev_in_london = in_london
        prev_in_preny = in_preny

    return {
        'asia_high': asia_high_out,
        'asia_low': asia_low_out,
        'london_high': london_high_out,
        'london_low': london_low_out,
        'preny_high': preny_high_out,
        'preny_low': preny_low_out,
    }


def check_session_sweep(highs, lows, closes, level, bar_idx, lookback=10,
                        direction=None):
    """Check if a session level was swept recently (wick beyond, close back).

    A sweep means price wicked beyond the level but closed back inside.

    Args:
        highs, lows, closes: price arrays
        level: the session H/L level to check
        bar_idx: current bar index
        lookback: how many bars back to check
        direction: 'above' to check sweep above a high (bearish sweep),
                   'below' to check sweep below a low (bullish sweep),
                   None to check both (legacy behavior).

    Returns:
        True if a sweep occurred within lookback bars before bar_idx
    """
    if np.isnan(level):
        return False
    start = max(0, bar_idx - lookback)
    for j in range(start, bar_idx):
        # Sweep above: wick above level, close back below (bearish sweep)
        if direction in (None, 'above'):
            if highs[j] > level and closes[j] < level:
                return True
        # Sweep below: wick below level, close back above (bullish sweep)
        if direction in (None, 'below'):
            if lows[j] < level and closes[j] > level:
                return True
    return False


# ============================================================================
# Part 2: Multi-Timeframe (5-min computation + mapping to 1-min)
# ============================================================================

def resample_for_ict(df_1min):
    """Resample 1-min bars to 5-min for ICT feature computation."""
    df_5min = df_1min.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }).dropna(subset=['Open'])
    return df_5min


def map_5min_idx_to_1min(onemin_times, fivemin_times):
    """Map each 1-min bar to its enclosing 5-min bar index.

    For 1-min bar at time t, finds the most recent 5-min bar whose
    timestamp <= t. Returns array of 5-min indices.
    """
    n_1m = len(onemin_times)
    mapping = np.zeros(n_1m, dtype=int)
    j = 0
    for i in range(n_1m):
        while j + 1 < len(fivemin_times) and fivemin_times[j + 1] <= onemin_times[i]:
            j += 1
        mapping[i] = j
    return mapping


def map_5min_events_to_1min(events_5min, idx_map_5to1, n_1min, fivemin_times):
    """Map 5-min ICT events (OBs, FVGs) to 1-min bar space.

    For each event, find the 1-min bar where the 5-min bar's event
    becomes available (first 1-min bar of the NEXT 5-min period).

    Args:
        events_5min: list of event dicts with 'bar_idx' in 5-min space
        idx_map_5to1: mapping from 1-min bar to enclosing 5-min bar index
        n_1min: number of 1-min bars
        fivemin_times: 5-min bar timestamps

    Returns:
        events with updated 'bar_idx_1min' field added.
    """
    # Build reverse mapping: for each 5-min bar j, find the first 1-min bar
    # whose enclosing 5-min bar index is j+1 (i.e., the first bar of next period)
    n_5m = len(fivemin_times)
    first_1min_after = {}  # 5min_idx -> first 1min bar of NEXT 5min period

    for i in range(n_1min):
        j5 = idx_map_5to1[i]
        # We want events from 5-min bar j to be visible starting at
        # the first 1-min bar that maps to 5-min bar j+1 or later
        if j5 not in first_1min_after:
            # This is still mapped to j5, but we want to find bars mapped to j5+1
            pass

    # Simpler: for each 5-min bar j, the event at j is confirmed at the close
    # of bar j, so it's available starting at the first 1-min bar of period j+1.
    # Build: 5min_idx -> first 1min bar AFTER that period ends
    fivemin_end_1min = {}
    prev_j = -1
    for i in range(n_1min):
        j = idx_map_5to1[i]
        if j != prev_j and prev_j >= 0:
            fivemin_end_1min[prev_j] = i
        prev_j = j
    if prev_j >= 0 and prev_j not in fivemin_end_1min:
        fivemin_end_1min[prev_j] = n_1min  # last 5-min period

    for ev in events_5min:
        j5 = ev['bar_idx']
        # Event at 5-min bar j5 becomes visible at the first 1-min bar of period j5+1
        if j5 in fivemin_end_1min:
            ev['bar_idx_1min'] = fivemin_end_1min[j5]
        else:
            # Fallback: visible at the bar after the 5-min period ends
            ev['bar_idx_1min'] = n_1min  # not visible in data range

    return events_5min


def compute_all_ict_features(opens, highs, lows, closes, swing_lb=50, pivot_right=2,
                              sweep_lookback=20):
    """Compute all ICT features for a set of OHLC arrays.

    Returns dict with all ICT feature results.
    """
    # Swing levels
    swing_highs, swing_lows = compute_swing_levels(
        highs, lows, lookback=swing_lb, swing_type="pivot", pivot_right=pivot_right)

    # Order blocks
    ob_events = compute_order_blocks(opens, highs, lows, closes)

    # FVGs
    fvg_events = compute_fvgs(highs, lows, closes)

    # BOS/MSS
    bos_mss_events, trend_state = compute_bos_mss(highs, lows, closes,
                                                   swing_highs, swing_lows)

    # Fibonacci/OTE
    fib_pos, fib_lvl, fib_dir = compute_fib_zones(highs, lows, closes,
                                                    bos_mss_events,
                                                    swing_highs, swing_lows)

    # Liquidity sweeps
    sweep_events, last_sweep_bar, last_sweep_dir = compute_liquidity_sweeps(
        highs, lows, closes, pivot_lookback=sweep_lookback)

    return {
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'ob_events': ob_events,
        'fvg_events': fvg_events,
        'bos_mss_events': bos_mss_events,
        'trend_state': trend_state,
        'fib_position': fib_pos,
        'fib_level': fib_lvl,
        'fib_direction': fib_dir,
        'sweep_events': sweep_events,
        'last_sweep_bar': last_sweep_bar,
        'last_sweep_dir': last_sweep_dir,
    }


# ============================================================================
# Part 3: Run Backtest and Collect Trades
# ============================================================================

def run_portfolio_trades():
    """Run all 4 strategies with gates, return trades + data.

    Returns:
        dict with keys:
          'trades': list of (trade_dict, strategy_name, instrument, dollar_per_pt, commission)
          'df_mnq', 'df_mes': DataFrames
    """
    print("=" * 70)
    print("ICT TRADE FORENSICS — Loading data and running portfolio")
    print("=" * 70)

    # --- Load MNQ ---
    print("\nLoading MNQ data...")
    df_mnq = load_instrument_1min("MNQ")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm
    print(f"  MNQ: {len(df_mnq)} bars, {df_mnq.index[0]} to {df_mnq.index[-1]}")

    # --- Load MES ---
    print("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm
    print(f"  MES: {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # --- Compute gates ---
    print("\n--- Computing Entry Gates ---")
    start_date = df_mnq.index[0].strftime("%Y-%m-%d")
    end_date = df_mnq.index[-1].strftime("%Y-%m-%d")

    mnq_closes = df_mnq["Close"].values
    mnq_highs = df_mnq["High"].values
    mnq_lows = df_mnq["Low"].values
    mnq_sm_arr = df_mnq["SM_Net"].values

    # Leledc
    print("  Computing Leledc exhaustion (mq=9)...")
    bull_ex, bear_ex = compute_leledc_exhaustion(mnq_closes, maj_qual=9)
    mnq_leledc_gate = build_leledc_gate(bull_ex, bear_ex, persistence=1)

    # ADR directional
    print("  Computing ADR directional gate (14d, 0.3)...")
    mnq_session = compute_session_tracking(df_mnq)
    mnq_adr = compute_adr(df_mnq, lookback_days=14)
    mnq_adr_gate = build_directional_gate(
        mnq_session['move_from_open'], mnq_adr, mnq_sm_arr, threshold=0.3)

    # ATR gate (vScalpC only)
    print("  Computing prior-day ATR gate (min=263.8)...")
    mnq_prior_atr = compute_prior_day_atr(df_mnq, lookback_days=14)
    mnq_atr_gate = np.ones(len(df_mnq), dtype=bool)
    for i in range(len(mnq_prior_atr)):
        if not np.isnan(mnq_prior_atr[i]) and mnq_prior_atr[i] < 263.8:
            mnq_atr_gate[i] = False

    # VIX death zone
    print("  Computing VIX death zone gate (19-22)...")
    mnq_bar_dates_et = _get_bar_dates_et(df_mnq.index)
    mnq_vix_gate = load_vix_gate(start_date, end_date, mnq_bar_dates_et,
                                  low=19.0, high=22.0)

    # MES gates
    print("  Computing MES prior-day level gate (VPOC+VAL, buf=5)...")
    mes_closes = df_mes["Close"].values
    mes_highs = df_mes["High"].values
    mes_lows = df_mes["Low"].values
    mes_times = df_mes.index
    mes_volumes = df_mes["Volume"].values
    mes_et_mins = compute_et_minutes(mes_times)

    prev_high, prev_low, _ = compute_prior_day_levels(
        mes_times, mes_highs, mes_lows, mes_closes)

    mes_vpoc, mes_vah, mes_val = compute_rth_volume_profile(
        mes_times, mes_closes, mes_volumes, mes_et_mins, bin_width=5)

    nan_arr = np.full(len(df_mes), np.nan)
    mes_level_gate = build_prior_day_level_gate(
        mes_closes, nan_arr, nan_arr, mes_vpoc, nan_arr, mes_val, buffer_pts=5.0)

    # Composite gates
    gate_vscalpa = mnq_leledc_gate & mnq_adr_gate & mnq_vix_gate
    gate_vscalpb = mnq_leledc_gate & mnq_adr_gate
    gate_vscalpc = mnq_leledc_gate & mnq_adr_gate & mnq_atr_gate & mnq_vix_gate
    gate_mesv2 = mes_level_gate

    # --- Swing levels for vScalpC ---
    print("  Computing swing levels for vScalpC structure exit...")
    mnq_swing_highs, mnq_swing_lows = compute_swing_levels(
        mnq_highs, mnq_lows,
        lookback=VSCALPC_SWING_LB, swing_type="pivot",
        pivot_right=VSCALPC_SWING_PR)

    # --- Prepare MNQ arrays ---
    mnq_opens = df_mnq["Open"].values
    mnq_times = df_mnq.index

    df_mnq_5m = resample_to_5min(df_mnq)
    rsi_mnq_curr, rsi_mnq_prev = map_5min_rsi_to_1min(
        df_mnq.index.values, df_mnq_5m.index.values,
        df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN)

    all_trades = []

    # --- vScalpA ---
    print("\n  Running vScalpA...")
    trades_a = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
        rsi_mnq_curr, rsi_mnq_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPA_MAX_LOSS_PTS, tp_pts=VSCALPA_TP_PTS,
        entry_end_et=VSCALPA_ENTRY_END_ET, entry_gate=gate_vscalpa,
    )
    print(f"    {len(trades_a)} trades")
    for t in trades_a:
        all_trades.append((t, 'vScalpA', 'MNQ', MNQ_DOLLAR_PER_PT, MNQ_COMMISSION))

    # --- vScalpB ---
    print("  Running vScalpB...")
    trades_b = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
        rsi_mnq_curr, rsi_mnq_prev,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
        entry_gate=gate_vscalpb,
    )
    print(f"    {len(trades_b)} trades")
    for t in trades_b:
        all_trades.append((t, 'vScalpB', 'MNQ', MNQ_DOLLAR_PER_PT, MNQ_COMMISSION))

    # --- vScalpC (structure exit) ---
    print("  Running vScalpC...")
    trades_c = run_backtest_structure_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
        rsi_mnq_curr, rsi_mnq_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPC_SL_PTS,
        tp1_pts=VSCALPC_TP1_PTS,
        swing_highs=mnq_swing_highs, swing_lows=mnq_swing_lows,
        swing_buffer_pts=VSCALPC_SWING_BUF,
        max_tp2_pts=VSCALPC_MAX_TP2_PTS,
        move_sl_to_be_after_tp1=True,
        breakeven_after_bars=0,
        entry_end_et=VSCALPA_ENTRY_END_ET,
        entry_gate=gate_vscalpc,
    )
    # For vScalpC, group by entry_idx to get per-entry trades
    # (each entry produces 2 trade dicts: scalp + runner)
    vscalpc_entries = {}
    for t in trades_c:
        eidx = t["entry_idx"]
        if eidx not in vscalpc_entries:
            vscalpc_entries[eidx] = []
        vscalpc_entries[eidx].append(t)

    # Compute per-entry result for forensics mapping
    comm = MNQ_COMMISSION * 2  # per leg
    n_entries_c = len(vscalpc_entries)
    print(f"    {n_entries_c} entries ({len(trades_c)} legs)")
    for eidx in sorted(vscalpc_entries.keys()):
        legs = vscalpc_entries[eidx]
        total_pts = sum(l['pts'] for l in legs)
        total_pnl = sum(l['pts'] * MNQ_DOLLAR_PER_PT - comm for l in legs)
        # Create a synthetic trade for forensics mapping
        first_leg = legs[0]
        sl_legs = [l for l in legs if l.get('result', l.get('exit_reason', '')) == 'SL']
        result_label = 'SL' if sl_legs else ('TP' if total_pnl > 0 else 'loss')
        synth = {
            'side': first_leg['side'],
            'entry': first_leg.get('entry', first_leg.get('entry_price', 0)),
            'exit': legs[-1].get('exit', legs[-1].get('exit_price', 0)),
            'pts': total_pts,
            'entry_time': first_leg['entry_time'],
            'exit_time': legs[-1]['exit_time'],
            'entry_idx': first_leg['entry_idx'],
            'exit_idx': legs[-1]['exit_idx'],
            'bars': legs[-1]['exit_idx'] - first_leg['entry_idx'],
            'result': result_label,
            'entry_price': first_leg.get('entry_price', first_leg.get('entry', 0)),
            '_pnl_dollar': total_pnl,
            '_is_structure': True,
        }
        all_trades.append((synth, 'vScalpC', 'MNQ', MNQ_DOLLAR_PER_PT, MNQ_COMMISSION))

    # --- MES v2 ---
    print("  Running MES v2...")
    mes_opens = df_mes["Open"].values
    mes_sm_arr = df_mes["SM_Net"].values

    df_mes_5m = resample_to_5min(df_mes)
    rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
        df_mes.index.values, df_mes_5m.index.values,
        df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN)

    trades_m = run_backtest_tp_exit(
        mes_opens, mes_highs, mes_lows, mes_closes, mes_sm_arr, mes_times,
        rsi_mes_curr, rsi_mes_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        entry_end_et=MESV2_ENTRY_END_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        entry_gate=gate_mesv2,
    )
    print(f"    {len(trades_m)} trades")
    for t in trades_m:
        all_trades.append((t, 'MES_v2', 'MES', MES_DOLLAR_PER_PT, MES_COMMISSION))

    print(f"\n  TOTAL: {len(all_trades)} trades across 4 strategies")

    # --- Weekly Volume Profile ---
    print("\n--- Computing Weekly Volume Profile ---")
    mnq_weekly_vpoc, mnq_weekly_vah, mnq_weekly_val = compute_weekly_volume_profile(
        df_mnq, bin_width=2)
    mes_weekly_vpoc, mes_weekly_vah, mes_weekly_val = compute_weekly_volume_profile(
        df_mes, bin_width=5)
    print(f"  MNQ: {np.sum(~np.isnan(mnq_weekly_vpoc))} bars with weekly profile")
    print(f"  MES: {np.sum(~np.isnan(mes_weekly_vpoc))} bars with weekly profile")

    # --- Session Levels ---
    print("\n--- Computing Session Levels ---")
    mnq_sessions = compute_session_levels(df_mnq)
    mes_sessions = compute_session_levels(df_mes)
    print(f"  MNQ Asia sessions: {np.sum(~np.isnan(mnq_sessions['asia_high']))} bars with levels")
    print(f"  MNQ London sessions: {np.sum(~np.isnan(mnq_sessions['london_high']))} bars with levels")
    print(f"  MNQ Pre-NY sessions: {np.sum(~np.isnan(mnq_sessions['preny_high']))} bars with levels")

    return {
        'all_trades': all_trades,
        'df_mnq': df_mnq,
        'df_mes': df_mes,
        'trades_a': trades_a,
        'trades_b': trades_b,
        'trades_c': trades_c,
        'trades_m': trades_m,
        'weekly_profiles': {
            'MNQ': (mnq_weekly_vpoc, mnq_weekly_vah, mnq_weekly_val),
            'MES': (mes_weekly_vpoc, mes_weekly_vah, mes_weekly_val),
        },
        'session_levels': {
            'MNQ': mnq_sessions,
            'MES': mes_sessions,
        },
    }


# ============================================================================
# Part 4: Map Trades to ICT Features
# ============================================================================

def find_nearest_active_zone(events, entry_idx, entry_price, direction_filter=None):
    """Find the nearest active (unmitigated) OB or FVG to the entry.

    Args:
        events: list of event dicts with bar_idx, direction, zone_low, zone_high, mitigated_at
        entry_idx: bar index of trade entry
        entry_price: entry price
        direction_filter: 'bull', 'bear', or None for any

    Returns:
        (distance_pts, inside_zone, event_dict) or (None, False, None) if none found
    """
    best_dist = None
    best_inside = False
    best_event = None

    for ev in events:
        # Must be formed before entry (use bar_idx_1min for 5-min events, bar_idx for 1-min)
        ev_bar = ev.get('bar_idx_1min', ev['bar_idx'])
        if ev_bar >= entry_idx:
            continue

        # Must be unmitigated at entry time
        if ev['mitigated_at'] != -1 and ev['mitigated_at'] <= entry_idx:
            continue

        if direction_filter and ev['direction'] != direction_filter:
            continue

        zone_low = ev['zone_low']
        zone_high = ev['zone_high']

        # Distance from entry price to zone
        if zone_low <= entry_price <= zone_high:
            dist = 0.0
            inside = True
        elif entry_price < zone_low:
            dist = zone_low - entry_price
            inside = False
        else:
            dist = entry_price - zone_high
            inside = False

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_inside = inside
            best_event = ev

    return best_dist, best_inside, best_event


def map_trade_to_ict(trade, entry_idx, entry_price, side, instrument,
                     ict_1min, ict_5min, idx_map_5to1_arr,
                     weekly_vpoc=None, weekly_vah=None, weekly_val=None,
                     session_levels=None, df_highs=None, df_lows=None,
                     df_closes=None):
    """Map a single trade to its ICT feature context at entry time.

    Returns dict of ICT features at entry.
    """
    result = {}

    # Use i-1 for all feature lookups (features must be available before entry)
    lookup_idx = entry_idx - 1 if entry_idx > 0 else 0

    # Get the 5-min bar index for the entry
    lookup_5min = idx_map_5to1_arr[lookup_idx] if lookup_idx < len(idx_map_5to1_arr) else 0

    # --- 5-min Order Blocks ---
    for tf_label, ict in [('5min', ict_5min), ('1min', ict_1min)]:
        for ob_dir in ['bull', 'bear']:
            dist, inside, _ = find_nearest_active_zone(
                ict['ob_events'], entry_idx, entry_price, direction_filter=ob_dir)
            result[f'ob_{ob_dir}_dist_{tf_label}'] = dist
            result[f'ob_{ob_dir}_inside_{tf_label}'] = inside

        # Any OB
        dist, inside, _ = find_nearest_active_zone(
            ict['ob_events'], entry_idx, entry_price)
        result[f'ob_any_dist_{tf_label}'] = dist
        result[f'ob_any_inside_{tf_label}'] = inside

    # --- 5-min FVGs ---
    for tf_label, ict in [('5min', ict_5min), ('1min', ict_1min)]:
        for fvg_dir in ['bull', 'bear']:
            dist, inside, _ = find_nearest_active_zone(
                ict['fvg_events'], entry_idx, entry_price, direction_filter=fvg_dir)
            result[f'fvg_{fvg_dir}_dist_{tf_label}'] = dist
            result[f'fvg_{fvg_dir}_inside_{tf_label}'] = inside

        dist, inside, _ = find_nearest_active_zone(
            ict['fvg_events'], entry_idx, entry_price)
        result[f'fvg_any_dist_{tf_label}'] = dist
        result[f'fvg_any_inside_{tf_label}'] = inside

    # --- Fibonacci / OTE (5-min primary) ---
    for tf_label, ict in [('5min', ict_5min), ('1min', ict_1min)]:
        fib_pos_arr = ict['fib_position']
        fib_lvl_arr = ict['fib_level']
        fib_dir_arr = ict['fib_direction']

        if tf_label == '5min':
            fib_idx = lookup_5min
        else:
            fib_idx = lookup_idx

        if fib_idx < len(fib_pos_arr):
            result[f'fib_position_{tf_label}'] = int(fib_pos_arr[fib_idx])
            result[f'fib_level_{tf_label}'] = float(fib_lvl_arr[fib_idx]) if not np.isnan(fib_lvl_arr[fib_idx]) else None
            result[f'fib_direction_{tf_label}'] = int(fib_dir_arr[fib_idx])
        else:
            result[f'fib_position_{tf_label}'] = 0
            result[f'fib_level_{tf_label}'] = None
            result[f'fib_direction_{tf_label}'] = 0

    # --- Liquidity Sweeps ---
    for tf_label, ict in [('5min', ict_5min), ('1min', ict_1min)]:
        sweep_bar_arr = ict['last_sweep_bar']
        sweep_dir_arr = ict['last_sweep_dir']

        if tf_label == '5min':
            s_idx = lookup_5min
        else:
            s_idx = lookup_idx

        if s_idx < len(sweep_bar_arr):
            last_sweep = sweep_bar_arr[s_idx]
            sweep_dir = sweep_dir_arr[s_idx]
            if last_sweep >= 0:
                if tf_label == '5min':
                    bars_since = s_idx - last_sweep
                else:
                    bars_since = lookup_idx - last_sweep
            else:
                bars_since = 9999
        else:
            sweep_dir = 0
            bars_since = 9999

        result[f'sweep_dir_{tf_label}'] = sweep_dir
        result[f'sweep_bars_ago_{tf_label}'] = bars_since

    # --- BOS/MSS State ---
    for tf_label, ict in [('5min', ict_5min), ('1min', ict_1min)]:
        trend_arr = ict['trend_state']

        if tf_label == '5min':
            t_idx = lookup_5min
        else:
            t_idx = lookup_idx

        if t_idx < len(trend_arr):
            result[f'trend_{tf_label}'] = int(trend_arr[t_idx])
        else:
            result[f'trend_{tf_label}'] = 0

        # Find most recent BOS/MSS event
        last_event = None
        last_event_bars = 9999
        for ev in ict['bos_mss_events']:
            ev_bar = ev.get('bar_idx_1min', ev['bar_idx'])
            if ev_bar < entry_idx:
                if tf_label == '5min':
                    bars_ago = t_idx - ev['bar_idx'] if 'bar_idx' in ev else 9999
                else:
                    bars_ago = lookup_idx - ev['bar_idx']
                last_event = ev
                last_event_bars = bars_ago

        result[f'last_structure_event_{tf_label}'] = last_event['event_type'] if last_event else None
        result[f'last_structure_dir_{tf_label}'] = last_event['direction'] if last_event else None
        result[f'last_structure_bars_{tf_label}'] = last_event_bars

    # --- Confluence Count ---
    # Count how many 5-min features align with the trade direction
    confluence = 0
    trade_is_long = (side == 'long')

    # 1. Near aligned OB (<10pts)
    if trade_is_long:
        ob_dist = result.get('ob_bull_dist_5min')
        if ob_dist is not None and ob_dist < 10:
            confluence += 1
    else:
        ob_dist = result.get('ob_bear_dist_5min')
        if ob_dist is not None and ob_dist < 10:
            confluence += 1

    # 2. Inside aligned FVG
    if trade_is_long and result.get('fvg_bull_inside_5min'):
        confluence += 1
    elif not trade_is_long and result.get('fvg_bear_inside_5min'):
        confluence += 1

    # 3. Fib discount for long / premium for short
    fib_pos = result.get('fib_position_5min', 0)
    fib_dir = result.get('fib_direction_5min', 0)
    if trade_is_long and fib_dir == 1 and fib_pos == 2:  # discount in bullish structure
        confluence += 1
    elif not trade_is_long and fib_dir == -1 and fib_pos == 1:  # premium in bearish structure
        confluence += 1

    # 4. OTE zone
    if fib_pos == 3:
        # OTE is favorable if fib direction matches trade direction
        if (trade_is_long and fib_dir == 1) or (not trade_is_long and fib_dir == -1):
            confluence += 1

    # 5. Recent sweep in trade direction
    sweep_dir = result.get('sweep_dir_5min', 0)
    sweep_bars = result.get('sweep_bars_ago_5min', 9999)
    if sweep_bars <= 10:
        if (trade_is_long and sweep_dir == 1) or (not trade_is_long and sweep_dir == -1):
            confluence += 1

    # 6. BOS in trade direction
    trend = result.get('trend_5min', 0)
    if (trade_is_long and trend == 1) or (not trade_is_long and trend == -1):
        confluence += 1

    result['confluence'] = confluence

    # --- Weekly Volume Profile Proximity ---
    if weekly_vpoc is not None and lookup_idx < len(weekly_vpoc):
        w_vpoc = weekly_vpoc[lookup_idx]
        w_vah = weekly_vah[lookup_idx]
        w_val = weekly_val[lookup_idx]

        result['weekly_vpoc_dist'] = abs(entry_price - w_vpoc) if not np.isnan(w_vpoc) else None
        result['weekly_vah_dist'] = abs(entry_price - w_vah) if not np.isnan(w_vah) else None
        result['weekly_val_dist'] = abs(entry_price - w_val) if not np.isnan(w_val) else None

        # Inside weekly value area?
        if not np.isnan(w_vah) and not np.isnan(w_val):
            result['weekly_inside_va'] = bool(w_val <= entry_price <= w_vah)
        else:
            result['weekly_inside_va'] = None

        # Near any weekly level (<10pts)?
        near_levels = []
        for lbl, dist in [('vpoc', result['weekly_vpoc_dist']),
                          ('vah', result['weekly_vah_dist']),
                          ('val', result['weekly_val_dist'])]:
            if dist is not None and dist < 10:
                near_levels.append(lbl)
        result['weekly_near_any'] = len(near_levels) > 0
        result['weekly_near_levels'] = near_levels

        # Directional context: long above VAH or short below VAL = chasing
        if not np.isnan(w_vah):
            result['long_above_weekly_vah'] = (side == 'long' and entry_price > w_vah)
        else:
            result['long_above_weekly_vah'] = False
        if not np.isnan(w_val):
            result['short_below_weekly_val'] = (side == 'short' and entry_price < w_val)
        else:
            result['short_below_weekly_val'] = False
    else:
        result['weekly_vpoc_dist'] = None
        result['weekly_vah_dist'] = None
        result['weekly_val_dist'] = None
        result['weekly_inside_va'] = None
        result['weekly_near_any'] = False
        result['weekly_near_levels'] = []
        result['long_above_weekly_vah'] = False
        result['short_below_weekly_val'] = False

    # --- Session Level Proximity ---
    if session_levels is not None and df_highs is not None:
        asia_h = session_levels['asia_high'][lookup_idx] if lookup_idx < len(session_levels['asia_high']) else np.nan
        asia_l = session_levels['asia_low'][lookup_idx] if lookup_idx < len(session_levels['asia_low']) else np.nan
        london_h = session_levels['london_high'][lookup_idx] if lookup_idx < len(session_levels['london_high']) else np.nan
        london_l = session_levels['london_low'][lookup_idx] if lookup_idx < len(session_levels['london_low']) else np.nan
        preny_h = session_levels['preny_high'][lookup_idx] if lookup_idx < len(session_levels['preny_high']) else np.nan
        preny_l = session_levels['preny_low'][lookup_idx] if lookup_idx < len(session_levels['preny_low']) else np.nan

        # Distance to nearest session H/L
        session_dists = {}
        for lbl, h, l in [('asia', asia_h, asia_l),
                          ('london', london_h, london_l),
                          ('preny', preny_h, preny_l)]:
            dists = []
            h_dist = abs(entry_price - h) if not np.isnan(h) else None
            l_dist = abs(entry_price - l) if not np.isnan(l) else None
            if h_dist is not None:
                dists.append(h_dist)
            if l_dist is not None:
                dists.append(l_dist)
            session_dists[lbl] = min(dists) if dists else None
            result[f'{lbl}_nearest_dist'] = session_dists[lbl]
            result[f'{lbl}_high_dist'] = h_dist
            result[f'{lbl}_low_dist'] = l_dist
            result[f'near_{lbl}_high'] = (h_dist is not None and h_dist < 5)
            result[f'near_{lbl}_low'] = (l_dist is not None and l_dist < 5)

        # Near session H/L (<5pts)?
        for lbl in ['asia', 'london', 'preny']:
            d = session_dists.get(lbl)
            result[f'{lbl}_near'] = (d is not None and d < 5)

        # Check for session sweep within 10 bars (direction-aware)
        bull_sweep = False  # sweep below a low = bullish
        bear_sweep = False  # sweep above a high = bearish
        for lbl, h, l in [('asia', asia_h, asia_l),
                          ('london', london_h, london_l),
                          ('preny', preny_h, preny_l)]:
            # Highs: check sweep above (bearish)
            if not np.isnan(h):
                if check_session_sweep(df_highs, df_lows, df_closes,
                                       h, entry_idx, lookback=10,
                                       direction='above'):
                    bear_sweep = True
            # Lows: check sweep below (bullish)
            if not np.isnan(l):
                if check_session_sweep(df_highs, df_lows, df_closes,
                                       l, entry_idx, lookback=10,
                                       direction='below'):
                    bull_sweep = True
        result['session_sweep_nearby'] = bull_sweep or bear_sweep
        result['session_sweep_bull'] = bull_sweep
        result['session_sweep_bear'] = bear_sweep
        trade_is_long = (side == 'long')
        result['session_sweep_aligned'] = (
            (trade_is_long and bull_sweep) or
            (not trade_is_long and bear_sweep)
        )
        result['session_sweep_against'] = (
            (trade_is_long and bear_sweep and not bull_sweep) or
            (not trade_is_long and bull_sweep and not bear_sweep)
        )
    else:
        for lbl in ['asia', 'london', 'preny']:
            result[f'{lbl}_nearest_dist'] = None
            result[f'{lbl}_near'] = False
            result[f'{lbl}_high_dist'] = None
            result[f'{lbl}_low_dist'] = None
            result[f'near_{lbl}_high'] = False
            result[f'near_{lbl}_low'] = False
        result['session_sweep_nearby'] = False
        result['session_sweep_bull'] = False
        result['session_sweep_bear'] = False
        result['session_sweep_aligned'] = False
        result['session_sweep_against'] = False

    return result


# ============================================================================
# Part 5: Output Statistics
# ============================================================================

def compute_group_stats(group_trades, label):
    """Compute WR, PF, avg P&L for a group of trades.

    Args:
        group_trades: list of (pnl_dollar, result) tuples

    Returns:
        dict with count, wr, pf, avg_pnl, delta_wr (vs overall)
    """
    if not group_trades:
        return None

    pnls = [t[0] for t in group_trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    avg_pnl = sum(pnls) / n

    return {
        'count': n,
        'wr': round(wr, 1),
        'pf': round(pf, 3),
        'avg_pnl': round(avg_pnl, 2),
        'total_pnl': round(sum(pnls), 2),
    }


def print_section(title, groups, baseline_wr):
    """Print a formatted section of ICT analysis results."""
    print(f"\n--- {title} ---")
    for label, stats in groups:
        if stats is None or stats['count'] == 0:
            continue
        delta = stats['wr'] - baseline_wr
        sign = '+' if delta >= 0 else ''
        print(f"  {label:<55s} {stats['count']:>4d} trades, "
              f"WR {stats['wr']:>5.1f}% ({sign}{delta:.1f}pp), "
              f"PF {stats['pf']:>6.3f}, "
              f"Avg ${stats['avg_pnl']:>+7.2f}")


def run_analysis(portfolio_data):
    """Run the full ICT forensics analysis."""

    all_trades = portfolio_data['all_trades']
    df_mnq = portfolio_data['df_mnq']
    df_mes = portfolio_data['df_mes']
    weekly_profiles = portfolio_data.get('weekly_profiles', {})
    session_levels_data = portfolio_data.get('session_levels', {})

    # -----------------------------------------------------------------------
    # Compute ICT features on 1-min and 5-min bars for both instruments
    # -----------------------------------------------------------------------
    print("\n--- Computing ICT Features ---")

    # Use a smaller swing lookback for BOS/MSS detection (more events)
    # but keep 50 for consistency with our structure exit
    BOS_SWING_LB = 50
    BOS_PIVOT_RIGHT = 2
    SWEEP_LB = 20

    results = {}
    for inst, df in [('MNQ', df_mnq), ('MES', df_mes)]:
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values

        print(f"  Computing 1-min ICT features for {inst}...")
        ict_1min = compute_all_ict_features(
            opens, highs, lows, closes,
            swing_lb=BOS_SWING_LB, pivot_right=BOS_PIVOT_RIGHT,
            sweep_lookback=SWEEP_LB)
        print(f"    OBs: {len(ict_1min['ob_events'])}, "
              f"FVGs: {len(ict_1min['fvg_events'])}, "
              f"BOS/MSS: {len(ict_1min['bos_mss_events'])}, "
              f"Sweeps: {len(ict_1min['sweep_events'])}")

        # 5-min features
        print(f"  Computing 5-min ICT features for {inst}...")
        df_5min = resample_for_ict(df)
        o5 = df_5min['Open'].values
        h5 = df_5min['High'].values
        l5 = df_5min['Low'].values
        c5 = df_5min['Close'].values

        ict_5min = compute_all_ict_features(
            o5, h5, l5, c5,
            swing_lb=BOS_SWING_LB // 5 if BOS_SWING_LB >= 10 else BOS_SWING_LB,
            pivot_right=max(1, BOS_PIVOT_RIGHT),
            sweep_lookback=SWEEP_LB // 5 if SWEEP_LB >= 10 else SWEEP_LB)
        print(f"    OBs: {len(ict_5min['ob_events'])}, "
              f"FVGs: {len(ict_5min['fvg_events'])}, "
              f"BOS/MSS: {len(ict_5min['bos_mss_events'])}, "
              f"Sweeps: {len(ict_5min['sweep_events'])}")

        # Map 5-min events to 1-min bar indices
        idx_map = map_5min_idx_to_1min(df.index.values, df_5min.index.values)

        # Add bar_idx_1min to 5-min events
        map_5min_events_to_1min(ict_5min['ob_events'], idx_map, len(df), df_5min.index.values)
        map_5min_events_to_1min(ict_5min['fvg_events'], idx_map, len(df), df_5min.index.values)
        # BOS/MSS events: add bar_idx_1min
        for ev in ict_5min['bos_mss_events']:
            j5 = ev['bar_idx']
            # Find first 1-min bar after this 5-min bar
            candidates = np.where(idx_map > j5)[0]
            ev['bar_idx_1min'] = int(candidates[0]) if len(candidates) > 0 else len(df)

        results[inst] = {
            'ict_1min': ict_1min,
            'ict_5min': ict_5min,
            'idx_map': idx_map,
            'df_5min': df_5min,
        }

    # -----------------------------------------------------------------------
    # Map each trade to its ICT context
    # -----------------------------------------------------------------------
    print("\n--- Mapping Trades to ICT Features ---")

    mapped_trades = []

    # Prepare dataframe references for session sweep checks
    df_by_inst = {'MNQ': df_mnq, 'MES': df_mes}

    for trade, strategy, instrument, dpp, comm in all_trades:
        entry_idx = trade['entry_idx']
        entry_price = trade.get('entry_price', trade.get('entry', 0))
        side = trade['side']

        inst_data = results[instrument]

        # Weekly profile arrays for this instrument
        wp = weekly_profiles.get(instrument)
        w_vpoc = wp[0] if wp else None
        w_vah = wp[1] if wp else None
        w_val = wp[2] if wp else None

        # Session level arrays for this instrument
        sl_data = session_levels_data.get(instrument)

        # DataFrame for sweep checks
        df_inst = df_by_inst[instrument]

        ict_features = map_trade_to_ict(
            trade, entry_idx, entry_price, side, instrument,
            inst_data['ict_1min'], inst_data['ict_5min'],
            inst_data['idx_map'],
            weekly_vpoc=w_vpoc, weekly_vah=w_vah, weekly_val=w_val,
            session_levels=sl_data,
            df_highs=df_inst['High'].values,
            df_lows=df_inst['Low'].values,
            df_closes=df_inst['Close'].values)

        # Compute P&L in dollars
        if trade.get('_is_structure'):
            pnl_dollar = trade['_pnl_dollar']
        else:
            comm_pts = (comm * 2) / dpp
            pnl_dollar = (trade['pts'] - comm_pts) * dpp

        mapped_trades.append({
            'strategy': strategy,
            'instrument': instrument,
            'side': side,
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'pts': trade['pts'],
            'pnl_dollar': pnl_dollar,
            'result': trade.get('result', ''),
            'is_win': pnl_dollar > 0,
            'is_sl': trade.get('result', '') == 'SL',
            **ict_features,
        })

    print(f"  Mapped {len(mapped_trades)} trades to ICT features")

    # -----------------------------------------------------------------------
    # Compute and print statistics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    n_months = "12.8"
    n_strats = 4
    total_trades = len(mapped_trades)
    total_wins = sum(1 for t in mapped_trades if t['is_win'])
    total_pnl = sum(t['pnl_dollar'] for t in mapped_trades)
    baseline_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    gross_profit = sum(t['pnl_dollar'] for t in mapped_trades if t['pnl_dollar'] > 0)
    gross_loss = abs(sum(t['pnl_dollar'] for t in mapped_trades if t['pnl_dollar'] <= 0))
    baseline_pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    print(f"ICT TRADE FORENSICS — {n_months} months, {n_strats} strategies, all gates applied")
    print("=" * 80)
    print(f"\nPORTFOLIO BASELINE: {total_trades} trades, "
          f"WR {baseline_wr:.1f}%, PF {baseline_pf:.3f}, "
          f"Net ${total_pnl:+,.2f}")

    # Helper to get grouped stats
    def group_stats(filter_fn, label):
        group = [(t['pnl_dollar'], t['result']) for t in mapped_trades if filter_fn(t)]
        return label, compute_group_stats(group, label)

    # ===== 5-MIN ORDER BLOCKS =====
    groups = []

    # Long entries near bullish OB
    groups.append(group_stats(
        lambda t: t['side'] == 'long' and t.get('ob_bull_dist_5min') is not None and t['ob_bull_dist_5min'] < 10,
        "Long entries near bullish OB (<10pts)"))
    groups.append(group_stats(
        lambda t: t['side'] == 'long' and t.get('ob_bear_dist_5min') is not None and t['ob_bear_dist_5min'] < 10,
        "Long entries near bearish OB (<10pts)"))
    groups.append(group_stats(
        lambda t: t['side'] == 'short' and t.get('ob_bull_dist_5min') is not None and t['ob_bull_dist_5min'] < 10,
        "Short entries near bullish OB (<10pts)"))
    groups.append(group_stats(
        lambda t: t['side'] == 'short' and t.get('ob_bear_dist_5min') is not None and t['ob_bear_dist_5min'] < 10,
        "Short entries near bearish OB (<10pts)"))
    groups.append(group_stats(
        lambda t: t.get('ob_any_inside_5min', False),
        "Entries inside any OB zone"))
    groups.append(group_stats(
        lambda t: t.get('ob_any_dist_5min') is None or t['ob_any_dist_5min'] >= 10,
        "Entries NOT near any OB (>10pts)"))

    print_section("5-MIN ORDER BLOCKS", groups, baseline_wr)

    # ===== 5-MIN FAIR VALUE GAPS =====
    groups = []
    groups.append(group_stats(
        lambda t: t['side'] == 'long' and t.get('fvg_bull_inside_5min', False),
        "Long entries inside bullish FVG"))
    groups.append(group_stats(
        lambda t: t['side'] == 'long' and t.get('fvg_bear_inside_5min', False),
        "Long entries inside bearish FVG"))
    groups.append(group_stats(
        lambda t: t['side'] == 'short' and t.get('fvg_bull_inside_5min', False),
        "Short entries inside bullish FVG"))
    groups.append(group_stats(
        lambda t: t['side'] == 'short' and t.get('fvg_bear_inside_5min', False),
        "Short entries inside bearish FVG"))
    groups.append(group_stats(
        lambda t: t.get('fvg_any_inside_5min', False),
        "Entries inside any FVG zone"))
    groups.append(group_stats(
        lambda t: t.get('fvg_any_dist_5min') is not None and t['fvg_any_dist_5min'] < 10 and not t.get('fvg_any_inside_5min', False),
        "Entries near FVG (<10pts, not inside)"))

    print_section("5-MIN FAIR VALUE GAPS", groups, baseline_wr)

    # ===== FIBONACCI / OTE (5-min structure) =====
    groups = []
    groups.append(group_stats(
        lambda t: t['side'] == 'long' and t.get('fib_direction_5min') == 1 and t.get('fib_position_5min') == 2,
        "Long in discount (bullish fib) = FAVORABLE"))
    groups.append(group_stats(
        lambda t: t['side'] == 'long' and t.get('fib_direction_5min') == 1 and t.get('fib_position_5min') == 1,
        "Long in premium (bullish fib) = UNFAVORABLE"))
    groups.append(group_stats(
        lambda t: t['side'] == 'short' and t.get('fib_direction_5min') == -1 and t.get('fib_position_5min') == 1,
        "Short in premium (bearish fib) = FAVORABLE"))
    groups.append(group_stats(
        lambda t: t['side'] == 'short' and t.get('fib_direction_5min') == -1 and t.get('fib_position_5min') == 2,
        "Short in discount (bearish fib) = UNFAVORABLE"))
    groups.append(group_stats(
        lambda t: t.get('fib_position_5min') == 3,
        "Entries in OTE zone (62-79% retrace)"))
    groups.append(group_stats(
        lambda t: t.get('fib_position_5min') == 3 and (
            (t['side'] == 'long' and t.get('fib_direction_5min') == 1) or
            (t['side'] == 'short' and t.get('fib_direction_5min') == -1)),
        "Entries in OTE zone, ALIGNED with structure"))
    groups.append(group_stats(
        lambda t: t.get('fib_position_5min') == 0,
        "No fib context"))

    print_section("FIBONACCI / OTE (5-min structure)", groups, baseline_wr)

    # ===== LIQUIDITY SWEEPS =====
    groups = []
    for window in [5, 10, 20]:
        groups.append(group_stats(
            lambda t, w=window: t['side'] == 'long' and t.get('sweep_dir_5min') == 1 and t.get('sweep_bars_ago_5min', 9999) <= w,
            f"Long within {window} bars of bullish sweep (5min)"))
        groups.append(group_stats(
            lambda t, w=window: t['side'] == 'short' and t.get('sweep_dir_5min') == -1 and t.get('sweep_bars_ago_5min', 9999) <= w,
            f"Short within {window} bars of bearish sweep (5min)"))

    groups.append(group_stats(
        lambda t: t.get('sweep_bars_ago_5min', 9999) > 20,
        "No recent sweep (>20 bars, 5min)"))

    print_section("LIQUIDITY SWEEPS (5-min)", groups, baseline_wr)

    # ===== STRUCTURE STATE (BOS/MSS) =====
    groups = []
    groups.append(group_stats(
        lambda t: (t['side'] == 'long' and t.get('trend_5min') == 1) or
                  (t['side'] == 'short' and t.get('trend_5min') == -1),
        "Entry WITH 5-min trend (BOS continuation)"))
    groups.append(group_stats(
        lambda t: (t['side'] == 'long' and t.get('trend_5min') == -1) or
                  (t['side'] == 'short' and t.get('trend_5min') == 1),
        "Entry AGAINST 5-min trend"))
    groups.append(group_stats(
        lambda t: t.get('trend_5min') == 0,
        "Entry with neutral/no trend"))

    groups.append(group_stats(
        lambda t: t.get('last_structure_event_5min') == 'MSS' and t.get('last_structure_bars_5min', 9999) <= 10,
        "Entry within 10 bars of MSS (reversal)"))
    groups.append(group_stats(
        lambda t: t.get('last_structure_event_5min') == 'BOS' and t.get('last_structure_bars_5min', 9999) <= 10,
        "Entry within 10 bars of BOS (continuation)"))

    print_section("STRUCTURE STATE (5-min BOS/MSS)", groups, baseline_wr)

    # ===== CONFLUENCE =====
    groups = []
    for c_min in [0, 1, 2, 3, 4]:
        groups.append(group_stats(
            lambda t, cm=c_min: t.get('confluence', 0) == cm,
            f"{c_min} features aligned"))
    groups.append(group_stats(
        lambda t: t.get('confluence', 0) >= 2,
        "2+ features aligned"))
    groups.append(group_stats(
        lambda t: t.get('confluence', 0) >= 3,
        "3+ features aligned"))

    print_section("CONFLUENCE (5-min features aligned with trade direction)", groups, baseline_wr)

    # ===== PER-STRATEGY BREAKDOWN =====
    print(f"\n--- PER-STRATEGY BREAKDOWN ---")
    strategies = ['vScalpA', 'vScalpB', 'vScalpC', 'MES_v2']

    for strat in strategies:
        strat_trades = [t for t in mapped_trades if t['strategy'] == strat]
        if not strat_trades:
            continue
        n_st = len(strat_trades)
        wins_st = sum(1 for t in strat_trades if t['is_win'])
        wr_st = wins_st / n_st * 100
        pnl_st = sum(t['pnl_dollar'] for t in strat_trades)
        gp_st = sum(t['pnl_dollar'] for t in strat_trades if t['pnl_dollar'] > 0)
        gl_st = abs(sum(t['pnl_dollar'] for t in strat_trades if t['pnl_dollar'] <= 0))
        pf_st = gp_st / gl_st if gl_st > 0 else 999.0

        print(f"\n  {strat}: {n_st} trades, WR {wr_st:.1f}%, PF {pf_st:.3f}, Net ${pnl_st:+,.2f}")

        # Key feature breakdowns per strategy
        strat_groups = []
        strat_groups.append(group_stats(
            lambda t, s=strat: t['strategy'] == s and (
                (t['side'] == 'long' and t.get('trend_5min') == 1) or
                (t['side'] == 'short' and t.get('trend_5min') == -1)),
            "  With trend"))
        strat_groups.append(group_stats(
            lambda t, s=strat: t['strategy'] == s and (
                (t['side'] == 'long' and t.get('trend_5min') == -1) or
                (t['side'] == 'short' and t.get('trend_5min') == 1)),
            "  Against trend"))
        strat_groups.append(group_stats(
            lambda t, s=strat: t['strategy'] == s and t.get('confluence', 0) >= 2,
            "  Confluence 2+"))
        strat_groups.append(group_stats(
            lambda t, s=strat: t['strategy'] == s and t.get('fib_position_5min') == 3,
            "  In OTE zone"))
        strat_groups.append(group_stats(
            lambda t, s=strat: t['strategy'] == s and t.get('ob_any_inside_5min', False),
            "  Inside OB"))

        for label, stats in strat_groups:
            if stats is not None and stats['count'] > 0:
                delta = stats['wr'] - wr_st
                sign = '+' if delta >= 0 else ''
                print(f"    {label:<50s} {stats['count']:>4d} trades, "
                      f"WR {stats['wr']:>5.1f}% ({sign}{delta:.1f}pp), "
                      f"PF {stats['pf']:>6.3f}")

    # ===== SL CLUSTERING =====
    print(f"\n--- SL CLUSTERING ---")
    sl_trades = [t for t in mapped_trades if t['is_sl']]
    non_sl_trades = [t for t in mapped_trades if not t['is_sl']]

    if sl_trades:
        n_sl = len(sl_trades)
        print(f"  Total SL trades: {n_sl} / {total_trades} ({n_sl/total_trades*100:.1f}%)")

        # Check each structural context for SL concentration
        contexts = []

        # Against trend
        against_sl = sum(1 for t in sl_trades if
            (t['side'] == 'long' and t.get('trend_5min') == -1) or
            (t['side'] == 'short' and t.get('trend_5min') == 1))
        against_all = sum(1 for t in mapped_trades if
            (t['side'] == 'long' and t.get('trend_5min') == -1) or
            (t['side'] == 'short' and t.get('trend_5min') == 1))
        if against_all > 0:
            contexts.append((against_sl, against_all,
                             f"Entered AGAINST 5-min trend",
                             against_sl / against_all * 100 if against_all > 0 else 0))

        # Near opposing OB
        near_opp_ob_sl = sum(1 for t in sl_trades if
            (t['side'] == 'long' and t.get('ob_bear_dist_5min') is not None and t['ob_bear_dist_5min'] < 10) or
            (t['side'] == 'short' and t.get('ob_bull_dist_5min') is not None and t['ob_bull_dist_5min'] < 10))
        near_opp_ob_all = sum(1 for t in mapped_trades if
            (t['side'] == 'long' and t.get('ob_bear_dist_5min') is not None and t['ob_bear_dist_5min'] < 10) or
            (t['side'] == 'short' and t.get('ob_bull_dist_5min') is not None and t['ob_bull_dist_5min'] < 10))
        if near_opp_ob_all > 0:
            contexts.append((near_opp_ob_sl, near_opp_ob_all,
                             f"Near opposing OB (<10pts, 5-min)",
                             near_opp_ob_sl / near_opp_ob_all * 100))

        # In unfavorable fib zone
        unfav_fib_sl = sum(1 for t in sl_trades if
            (t['side'] == 'long' and t.get('fib_direction_5min') == 1 and t.get('fib_position_5min') == 1) or
            (t['side'] == 'short' and t.get('fib_direction_5min') == -1 and t.get('fib_position_5min') == 2))
        unfav_fib_all = sum(1 for t in mapped_trades if
            (t['side'] == 'long' and t.get('fib_direction_5min') == 1 and t.get('fib_position_5min') == 1) or
            (t['side'] == 'short' and t.get('fib_direction_5min') == -1 and t.get('fib_position_5min') == 2))
        if unfav_fib_all > 0:
            contexts.append((unfav_fib_sl, unfav_fib_all,
                             f"In unfavorable fib zone (5-min)",
                             unfav_fib_sl / unfav_fib_all * 100))

        # Zero confluence
        zero_conf_sl = sum(1 for t in sl_trades if t.get('confluence', 0) == 0)
        zero_conf_all = sum(1 for t in mapped_trades if t.get('confluence', 0) == 0)
        if zero_conf_all > 0:
            contexts.append((zero_conf_sl, zero_conf_all,
                             f"Zero ICT confluence",
                             zero_conf_sl / zero_conf_all * 100))

        # Recent MSS (reversal just happened)
        mss_sl = sum(1 for t in sl_trades if
            t.get('last_structure_event_5min') == 'MSS' and
            t.get('last_structure_bars_5min', 9999) <= 10)
        mss_all = sum(1 for t in mapped_trades if
            t.get('last_structure_event_5min') == 'MSS' and
            t.get('last_structure_bars_5min', 9999) <= 10)
        if mss_all > 0:
            contexts.append((mss_sl, mss_all,
                             f"Within 10 bars of MSS (reversal)",
                             mss_sl / mss_all * 100))

        # Near weekly VPOC (<10pts)
        near_wk_vpoc_sl = sum(1 for t in sl_trades if
            t.get('weekly_vpoc_dist') is not None and t['weekly_vpoc_dist'] < 10)
        near_wk_vpoc_all = sum(1 for t in mapped_trades if
            t.get('weekly_vpoc_dist') is not None and t['weekly_vpoc_dist'] < 10)
        if near_wk_vpoc_all > 0:
            contexts.append((near_wk_vpoc_sl, near_wk_vpoc_all,
                             f"Near weekly VPOC (<10pts)",
                             near_wk_vpoc_sl / near_wk_vpoc_all * 100))

        # Outside weekly value area
        outside_wva_sl = sum(1 for t in sl_trades if
            t.get('weekly_inside_va') is not None and not t['weekly_inside_va'])
        outside_wva_all = sum(1 for t in mapped_trades if
            t.get('weekly_inside_va') is not None and not t['weekly_inside_va'])
        if outside_wva_all > 0:
            contexts.append((outside_wva_sl, outside_wva_all,
                             f"Outside weekly value area",
                             outside_wva_sl / outside_wva_all * 100))

        # Near session H/L (<5pts)
        near_session_sl = sum(1 for t in sl_trades if
            t.get('asia_near') or t.get('london_near') or t.get('preny_near'))
        near_session_all = sum(1 for t in mapped_trades if
            t.get('asia_near') or t.get('london_near') or t.get('preny_near'))
        if near_session_all > 0:
            contexts.append((near_session_sl, near_session_all,
                             f"Near session H/L (<5pts)",
                             near_session_sl / near_session_all * 100))

        # After session sweep (any direction)
        sess_sweep_sl = sum(1 for t in sl_trades if t.get('session_sweep_nearby'))
        sess_sweep_all = sum(1 for t in mapped_trades if t.get('session_sweep_nearby'))
        if sess_sweep_all > 0:
            contexts.append((sess_sweep_sl, sess_sweep_all,
                             f"After session level sweep (10 bars)",
                             sess_sweep_sl / sess_sweep_all * 100))
        # After aligned session sweep
        aligned_sl = sum(1 for t in sl_trades if t.get('session_sweep_aligned'))
        aligned_all = sum(1 for t in mapped_trades if t.get('session_sweep_aligned'))
        if aligned_all > 0:
            contexts.append((aligned_sl, aligned_all,
                             f"Sweep aligned with entry direction",
                             aligned_sl / aligned_all * 100))
        # After opposing session sweep
        against_sl = sum(1 for t in sl_trades if t.get('session_sweep_against'))
        against_all = sum(1 for t in mapped_trades if t.get('session_sweep_against'))
        if against_all > 0:
            contexts.append((against_sl, against_all,
                             f"Sweep against entry direction",
                             against_sl / against_all * 100))

        # Sort by SL rate (highest first)
        contexts.sort(key=lambda x: x[3], reverse=True)

        overall_sl_rate = n_sl / total_trades * 100
        print(f"\n  Top structural contexts for SL trades (baseline SL rate: {overall_sl_rate:.1f}%):")
        for i, (sl_count, total_count, desc, sl_rate) in enumerate(contexts):
            delta = sl_rate - overall_sl_rate
            sign = '+' if delta >= 0 else ''
            print(f"    {i+1}. {desc:<45s} "
                  f"{sl_count}/{total_count} SL ({sl_rate:.1f}%, {sign}{delta:.1f}pp vs baseline)")

    # ===== WEEKLY VOLUME PROFILE (PVP) =====
    groups = []
    groups.append(group_stats(
        lambda t: t.get('weekly_inside_va') is not None and bool(t['weekly_inside_va']) is True,
        "Entries inside weekly value area"))
    groups.append(group_stats(
        lambda t: t.get('weekly_inside_va') is not None and bool(t['weekly_inside_va']) is False,
        "Entries outside weekly value area"))
    groups.append(group_stats(
        lambda t: t.get('weekly_vpoc_dist') is not None and t['weekly_vpoc_dist'] < 10,
        "Entries near weekly VPOC (<10pts)"))
    groups.append(group_stats(
        lambda t: t.get('weekly_vah_dist') is not None and t['weekly_vah_dist'] < 10,
        "Entries near weekly VAH (<10pts)"))
    groups.append(group_stats(
        lambda t: t.get('weekly_val_dist') is not None and t['weekly_val_dist'] < 10,
        "Entries near weekly VAL (<10pts)"))
    groups.append(group_stats(
        lambda t: t.get('long_above_weekly_vah', False),
        "Long entries above weekly VAH (chasing?)"))
    groups.append(group_stats(
        lambda t: t.get('short_below_weekly_val', False),
        "Short entries below weekly VAL (chasing?)"))

    print_section("WEEKLY VOLUME PROFILE (PVP)", groups, baseline_wr)

    # ===== SESSION HIGHS/LOWS =====
    groups = []
    groups.append(group_stats(
        lambda t: t.get('asia_near', False),
        "Entries near Asia H/L (<5pts)"))
    groups.append(group_stats(
        lambda t: t.get('london_near', False),
        "Entries near London H/L (<5pts)"))
    groups.append(group_stats(
        lambda t: t.get('preny_near', False),
        "Entries near pre-NY H/L (<5pts)"))
    groups.append(group_stats(
        lambda t: t.get('session_sweep_nearby', False),
        "Entry after session sweep (within 10 bars)"))
    groups.append(group_stats(
        lambda t: t.get('session_sweep_bull', False),
        "Bullish sweep of session low (within 10 bars)"))
    groups.append(group_stats(
        lambda t: t.get('session_sweep_bear', False),
        "Bearish sweep of session high (within 10 bars)"))
    groups.append(group_stats(
        lambda t: t.get('session_sweep_aligned', False),
        "Entry direction aligned with sweep"))
    groups.append(group_stats(
        lambda t: t.get('session_sweep_against', False),
        "Entry direction against sweep"))
    groups.append(group_stats(
        lambda t: t.get('asia_near', False) or t.get('london_near', False) or t.get('preny_near', False),
        "Entries near ANY session H/L (<5pts)"))
    groups.append(group_stats(
        lambda t: not (t.get('asia_near', False) or t.get('london_near', False) or t.get('preny_near', False)),
        "Entries NOT near any session H/L (>5pts)"))

    print_section("SESSION HIGHS/LOWS", groups, baseline_wr)

    # ===== LONDON H/L DIRECTIONAL BREAKDOWN =====
    print(f"\n--- LONDON H/L DIRECTIONAL BREAKDOWN ---")
    london_dir_groups = [
        (lambda t: t['side'] == 'long' and t.get('near_london_high', False),
         "Long near London HIGH (<5pts)",
         "long hitting resistance"),
        (lambda t: t['side'] == 'long' and t.get('near_london_low', False),
         "Long near London LOW (<5pts)",
         "long at support"),
        (lambda t: t['side'] == 'short' and t.get('near_london_high', False),
         "Short near London HIGH (<5pts)",
         "short at resistance"),
        (lambda t: t['side'] == 'short' and t.get('near_london_low', False),
         "Short near London LOW (<5pts)",
         "short hitting support"),
    ]
    for filter_fn, label, context in london_dir_groups:
        trades_in_group = [(t['pnl_dollar'], t['result']) for t in mapped_trades if filter_fn(t)]
        stats = compute_group_stats(trades_in_group, label)
        if stats is None or stats['count'] == 0:
            print(f"  {label:<40s}    0 trades")
            continue
        delta = stats['wr'] - baseline_wr
        sign = '+' if delta >= 0 else ''
        print(f"  {label:<40s} {stats['count']:>4d} trades, "
              f"WR {stats['wr']:>5.1f}% ({sign}{delta:.1f}pp), "
              f"PF {stats['pf']:>6.3f}  <- {context}")

    # ===== 1-MIN COMPARISON =====
    print(f"\n--- 1-MIN vs 5-MIN COMPARISON (key metrics) ---")
    for tf in ['1min', '5min']:
        with_trend = sum(1 for t in mapped_trades if
            (t['side'] == 'long' and t.get(f'trend_{tf}') == 1) or
            (t['side'] == 'short' and t.get(f'trend_{tf}') == -1))
        against_trend = sum(1 for t in mapped_trades if
            (t['side'] == 'long' and t.get(f'trend_{tf}') == -1) or
            (t['side'] == 'short' and t.get(f'trend_{tf}') == 1))

        with_trend_pnl = [(t['pnl_dollar'], t['result']) for t in mapped_trades if
            (t['side'] == 'long' and t.get(f'trend_{tf}') == 1) or
            (t['side'] == 'short' and t.get(f'trend_{tf}') == -1)]
        against_trend_pnl = [(t['pnl_dollar'], t['result']) for t in mapped_trades if
            (t['side'] == 'long' and t.get(f'trend_{tf}') == -1) or
            (t['side'] == 'short' and t.get(f'trend_{tf}') == 1)]

        with_stats = compute_group_stats(with_trend_pnl, "with")
        against_stats = compute_group_stats(against_trend_pnl, "against")

        print(f"  {tf.upper():>5s} — With trend: ", end="")
        if with_stats:
            print(f"{with_stats['count']} trades, WR {with_stats['wr']}%, PF {with_stats['pf']}")
        else:
            print("N/A")

        print(f"  {tf.upper():>5s} — Against:    ", end="")
        if against_stats:
            print(f"{against_stats['count']} trades, WR {against_stats['wr']}%, PF {against_stats['pf']}")
        else:
            print("N/A")

    print(f"\n{'='*80}")
    print("ICT Forensics complete.")
    print(f"{'='*80}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    portfolio_data = run_portfolio_trades()
    run_analysis(portfolio_data)
