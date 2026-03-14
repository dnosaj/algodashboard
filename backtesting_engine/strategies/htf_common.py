"""
Higher Timeframe (HTF) Common Infrastructure — Phase 2 vScalpC
================================================================
Shared utilities for multi-timeframe analysis of vScalpC runner trades.

Provides:
  - resample_to_timeframe()    — generic resampler (5m, 15m, 30m, 1H, 4H)
  - map_htf_to_1min()          — map HTF values back to 1-min bars (no look-ahead)
  - compute_htf_indicators()   — batch compute SM, RSI, ATR, EMA on all TFs
  - prepare_vscalpc_data()     — load MNQ, compute SM + RSI, ready for backtest
  - slice_gate_partial()       — slice gate arrays for IS/OOS on partial exit backtests

Usage:
    cd backtesting_engine && python3 strategies/htf_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Path setup ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR))
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    compute_rsi,
    compute_et_minutes,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    NY_OPEN_ET,
    NY_CLOSE_ET,
)

from generate_session import (
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_ENTRY_END_ET,
)

from sr_common import compute_atr_wilder

from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


def _get_et_dates(df_1m):
    """Convert 1-min timestamps to ET calendar dates (handles EST/EDT)."""
    idx = pd.DatetimeIndex(df_1m.index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    return idx.tz_convert(_ET).date


# ============================================================================
# vScalpC production config constants
# ============================================================================

VSCALPC_TP1 = 7
VSCALPC_TP2 = 25
VSCALPC_SL = 40
VSCALPC_BE_TIME = 45
VSCALPC_SL_TO_BE = True

# Timeframes to analyze (pandas resample strings)
HTF_TIMEFRAMES = ['5min', '15min', '30min', '1h', '4h']


# ============================================================================
# Generic Resampler
# ============================================================================

def resample_to_timeframe(df_1m, tf_str):
    """Resample 1-min bars to any timeframe.

    Extends resample_to_5min pattern with generic tf_str.

    Args:
        df_1m: DataFrame with Open, High, Low, Close, SM_Net, Volume columns.
               Index must be DatetimeIndex.
        tf_str: Pandas resample string — '5min', '15min', '30min', '1h', '4h'.

    Returns:
        DataFrame with resampled OHLCV + SM_Net.
    """
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'SM_Net': 'last',
    }
    if 'Volume' in df_1m.columns:
        agg['Volume'] = 'sum'

    df_htf = df_1m.resample(tf_str).agg(agg).dropna(subset=['Open'])
    return df_htf


# ============================================================================
# Generic HTF-to-1min Mapper (no look-ahead)
# ============================================================================

def map_htf_to_1min(onemin_times, htf_times, htf_values):
    """Map higher-timeframe values back to 1-min bars without look-ahead.

    Uses the j-pointer loop from map_5min_rsi_to_1min. At each 1-min bar i,
    finds the most recent COMPLETED HTF bar (j-1) to prevent look-ahead.

    The j pointer advances to the latest HTF bar whose timestamp <= 1-min bar i.
    We use htf_values[j-1] because bar j is still forming (its values aren't
    finalized until the bar closes).

    Args:
        onemin_times: 1-min bar timestamps (numpy array of datetime64)
        htf_times:    HTF bar timestamps (numpy array of datetime64)
        htf_values:   HTF indicator values (numpy array, same length as htf_times)

    Returns:
        numpy array of length len(onemin_times) with mapped HTF values.
        First bars before any HTF bar completes get np.nan.
    """
    n_1m = len(onemin_times)
    result = np.full(n_1m, np.nan)

    j = 0
    for i in range(n_1m):
        while j + 1 < len(htf_times) and htf_times[j + 1] <= onemin_times[i]:
            j += 1
        if j >= 1:
            result[i] = htf_values[j - 1]

    return result


def map_htf_to_1min_curr_prev(onemin_times, htf_times, htf_values):
    """Map HTF values to 1-min bars, returning both current and previous.

    Same pattern as map_5min_rsi_to_1min but generic.

    Returns:
        (curr, prev) — two numpy arrays of length len(onemin_times).
        curr = htf_values[j-1], prev = htf_values[j-2].
    """
    n_1m = len(onemin_times)
    curr = np.full(n_1m, np.nan)
    prev = np.full(n_1m, np.nan)

    j = 0
    for i in range(n_1m):
        while j + 1 < len(htf_times) and htf_times[j + 1] <= onemin_times[i]:
            j += 1
        if j >= 1:
            curr[i] = htf_values[j - 1]
        if j >= 2:
            prev[i] = htf_values[j - 2]

    return curr, prev


# ============================================================================
# Batch HTF Indicator Computation
# ============================================================================

def compute_htf_indicators(df_1m, timeframes=None):
    """Compute all HTF indicators and map them back to 1-min bars.

    For each timeframe, computes:
      - SM Net Index (using MNQ SM params: 10/12/200/100)
      - RSI(8) on HTF closes
      - ATR(14) Wilder
      - EMA(20) of close
      - SM direction (sign of SM Net: +1, -1, 0)
      - SM slope (SM[j-1] - SM[j-2])

    All values are mapped to 1-min bars using the j-1 offset (no look-ahead).

    Args:
        df_1m: 1-min DataFrame with Open, High, Low, Close, SM_Net, Volume.
        timeframes: list of tf strings. Default: HTF_TIMEFRAMES.

    Returns:
        dict mapping (tf_str, indicator_name) -> numpy array (len = len(df_1m)).

    Indicator names:
        'sm'       — HTF SM Net value (mapped)
        'sm_dir'   — sign(sm): +1 if bullish, -1 if bearish, 0 if neutral
        'sm_slope' — sm[j-1] - sm[j-2] (momentum of SM)
        'rsi'      — RSI(8) on HTF closes
        'atr'      — ATR(14) Wilder on HTF bars
        'ema20'    — EMA(20) of HTF close
        'close'    — HTF close price (for S/R proximity checks)
        'high'     — HTF high (for swing detection)
        'low'      — HTF low (for swing detection)
    """
    if timeframes is None:
        timeframes = HTF_TIMEFRAMES

    onemin_times = df_1m.index.values
    result = {}

    for tf in timeframes:
        # Resample
        df_htf = resample_to_timeframe(df_1m, tf)
        htf_times = df_htf.index.values
        htf_closes = df_htf['Close'].values
        htf_highs = df_htf['High'].values
        htf_lows = df_htf['Low'].values

        # --- SM on HTF bars ---
        htf_volumes = df_htf['Volume'].values if 'Volume' in df_htf.columns else np.ones(len(htf_closes))
        htf_sm = compute_smart_money(
            htf_closes, htf_volumes,
            index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
            norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
        )

        # Map SM to 1-min
        sm_curr, sm_prev = map_htf_to_1min_curr_prev(onemin_times, htf_times, htf_sm)
        result[(tf, 'sm')] = sm_curr
        result[(tf, 'sm_dir')] = np.sign(sm_curr)
        result[(tf, 'sm_slope')] = sm_curr - sm_prev

        # --- RSI(8) on HTF closes ---
        htf_rsi = compute_rsi(htf_closes, 8)
        result[(tf, 'rsi')] = map_htf_to_1min(onemin_times, htf_times, htf_rsi)

        # --- ATR(14) on HTF bars ---
        htf_atr = compute_atr_wilder(htf_highs, htf_lows, htf_closes, period=14)
        result[(tf, 'atr')] = map_htf_to_1min(onemin_times, htf_times, htf_atr)

        # --- EMA(20) of HTF close ---
        htf_ema = _ema(htf_closes, 20)
        result[(tf, 'ema20')] = map_htf_to_1min(onemin_times, htf_times, htf_ema)

        # --- Raw HTF OHLC for structural analysis ---
        result[(tf, 'close')] = map_htf_to_1min(onemin_times, htf_times, htf_closes)
        result[(tf, 'high')] = map_htf_to_1min(onemin_times, htf_times, htf_highs)
        result[(tf, 'low')] = map_htf_to_1min(onemin_times, htf_times, htf_lows)

    return result


def _ema(arr, period):
    """EMA helper (same as compute_smart_money's internal EMA)."""
    r = np.zeros_like(arr, dtype=float)
    r[0] = arr[0]
    a = 2.0 / (period + 1)
    for i in range(1, len(arr)):
        r[i] = a * arr[i] + (1 - a) * r[i - 1]
    return r


# ============================================================================
# Volatility Regime (ATR-based)
# ============================================================================

def compute_volatility_regime(df_1m, lookback_days=20):
    """Classify each bar into a volatility regime using PRIOR-DAY range percentile.

    Uses prior day's range ranked against the 20 days before it.
    No look-ahead: today's bars are classified by yesterday's range,
    which is fully known before today's session opens.

    Returns:
        numpy array of length len(df_1m) with values 0 (low), 1 (medium), 2 (high).
        First (lookback_days + 1) days get np.nan.
    """
    # Compute daily high-low range (using ET dates to match live engine)
    et_mins = compute_et_minutes(df_1m.index)
    dates = _get_et_dates(df_1m)

    daily_ranges = {}
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() == 0:
            continue
        daily_ranges[d] = df_1m['High'].values[mask].max() - df_1m['Low'].values[mask].min()

    # Map daily range to 1-min bars, use rolling lookback
    n = len(df_1m)
    regime = np.full(n, np.nan)
    date_list = sorted(daily_ranges.keys())

    for i_date, d in enumerate(date_list):
        # Need lookback_days + 1: lookback window + prior day
        if i_date < lookback_days + 1:
            continue
        # Prior day's range ranked against the 20 days before it
        prior_day = date_list[i_date - 1]
        lookback = [daily_ranges[date_list[j]]
                    for j in range(i_date - 1 - lookback_days, i_date - 1)]
        prior_range = daily_ranges[prior_day]
        pct = np.searchsorted(sorted(lookback), prior_range) / len(lookback)

        mask = dates == d
        if pct < 0.33:
            regime[mask] = 0  # low vol
        elif pct < 0.67:
            regime[mask] = 1  # medium vol
        else:
            regime[mask] = 2  # high vol

    return regime


def compute_prior_day_atr(df_1m, lookback_days=14):
    """Compute prior-day ATR (Wilder) mapped to each 1-min bar.

    Uses daily range values with Wilder smoothing over lookback_days.
    Each bar gets the ATR as of the prior completed day (no look-ahead).

    Returns:
        numpy array of length len(df_1m) with prior-day ATR values.
        First (lookback_days + 1) days get np.nan.
    """
    dates = _get_et_dates(df_1m)

    daily_ranges = {}
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() == 0:
            continue
        daily_ranges[d] = df_1m['High'].values[mask].max() - df_1m['Low'].values[mask].min()

    date_list = sorted(daily_ranges.keys())
    n_days = len(date_list)

    # Wilder ATR on daily ranges
    daily_atr = np.full(n_days, np.nan)
    if n_days >= lookback_days:
        daily_atr[lookback_days - 1] = np.mean(
            [daily_ranges[date_list[j]] for j in range(lookback_days)]
        )
        for i in range(lookback_days, n_days):
            daily_atr[i] = (daily_atr[i - 1] * (lookback_days - 1)
                            + daily_ranges[date_list[i]]) / lookback_days

    # Map prior-day ATR to 1-min bars
    n = len(df_1m)
    result = np.full(n, np.nan)
    day_to_atr = {}
    for i, d in enumerate(date_list):
        if i >= 1 and not np.isnan(daily_atr[i - 1]):
            day_to_atr[d] = daily_atr[i - 1]  # prior day's ATR

    for d, atr_val in day_to_atr.items():
        mask = dates == d
        result[mask] = atr_val

    return result


# ============================================================================
# Session Context
# ============================================================================

def compute_session_context(df_1m, or_minutes=30):
    """Compute session context indicators.

    Returns dict with:
      'above_vwap'  — bool array, True if close > VWAP
      'or_high'     — opening range high (first or_minutes from 10:00 ET)
      'or_low'      — opening range low
      'above_or'    — bool array, True if close > or_high
      'below_or'    — bool array, True if close < or_low
      'ib_high'     — initial balance high (first 60 min from 10:00 ET)
      'ib_low'      — initial balance low
    """
    n = len(df_1m)
    et_mins = compute_et_minutes(df_1m.index)
    closes = df_1m['Close'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values

    # VWAP
    above_vwap = np.full(n, np.nan)
    if 'VWAP' in df_1m.columns:
        vwap = df_1m['VWAP'].values
        above_vwap = closes > vwap

    # Opening range + Initial balance: compute per day
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    ib_high = np.full(n, np.nan)
    ib_low = np.full(n, np.nan)

    dates = pd.DatetimeIndex(df_1m.index).date
    unique_dates = np.unique(dates)

    for d in unique_dates:
        day_mask = dates == d
        day_et = et_mins[day_mask]
        day_highs = highs[day_mask]
        day_lows = lows[day_mask]
        day_indices = np.where(day_mask)[0]

        # Opening range: first or_minutes after NY_OPEN_ET (600)
        or_mask = (day_et >= NY_OPEN_ET) & (day_et < NY_OPEN_ET + or_minutes)
        if or_mask.sum() > 0:
            or_h = day_highs[or_mask].max()
            or_l = day_lows[or_mask].min()
            # Apply OR levels to all bars AFTER the opening range
            post_or_mask = day_et >= NY_OPEN_ET + or_minutes
            for j in day_indices[post_or_mask]:
                or_high[j] = or_h
                or_low[j] = or_l

        # Initial balance: first 60 min
        ib_mask = (day_et >= NY_OPEN_ET) & (day_et < NY_OPEN_ET + 60)
        if ib_mask.sum() > 0:
            ib_h = day_highs[ib_mask].max()
            ib_l = day_lows[ib_mask].min()
            post_ib_mask = day_et >= NY_OPEN_ET + 60
            for j in day_indices[post_ib_mask]:
                ib_high[j] = ib_h
                ib_low[j] = ib_l

    above_or = closes > or_high
    below_or = closes < or_low

    return {
        'above_vwap': above_vwap,
        'or_high': or_high,
        'or_low': or_low,
        'above_or': above_or,
        'below_or': below_or,
        'ib_high': ib_high,
        'ib_low': ib_low,
    }


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_vscalpc_data():
    """Load MNQ data, compute SM + RSI, split IS/OOS.

    Returns:
        df: full 1-min DataFrame with SM_Net
        rsi_curr, rsi_prev: 5-min RSI mapped to 1-min (full period)
        is_len: index of IS/OOS split point
    """
    df = load_instrument_1min("MNQ")
    sm = compute_smart_money(
        df['Close'].values, df['Volume'].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df['SM_Net'] = sm
    print(f"Loaded MNQ: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # 5-min RSI mapped to 1-min
    df_5m = resample_to_5min(df)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values, df_5m['Close'].values,
        rsi_len=VSCALPA_RSI_LEN,
    )

    is_len = len(df) // 2
    print(f"  IS:  {is_len} bars ({df.index[0]} to {df.index[is_len - 1]})")
    print(f"  OOS: {len(df) - is_len} bars ({df.index[is_len]} to {df.index[-1]})")

    return df, rsi_curr, rsi_prev, is_len


def slice_gate_partial(full_gate, is_len, split_name):
    """Slice a gate array for IS/OOS splits (partial exit backtest).

    Args:
        full_gate: bool array same length as full data (or None)
        is_len: number of bars in IS period
        split_name: "FULL", "IS", or "OOS"

    Returns:
        Sliced gate array or None.
    """
    if full_gate is None:
        return None
    if split_name == "FULL":
        return full_gate
    elif split_name == "IS":
        return full_gate[:is_len]
    else:
        return full_gate[is_len:]
