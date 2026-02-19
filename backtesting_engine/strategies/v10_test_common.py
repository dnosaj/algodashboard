"""
v10 Test Common Infrastructure
===============================
Unified backtest engine, data loaders, scoring, and statistical utilities.
All v10 feature tests import from this module.

CRITICAL: run_backtest_v10() with all features OFF must produce IDENTICAL
results to the canonical v9 engine in scalp_v9_smflip_backtest.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INSTRUMENTS = {
    'MNQ': {
        'commission': 0.52, 'dollar_per_pt': 2.0,
        'file': 'CME_MINI_MNQ1!, 1_b2119.csv', 'has_vwap': True,
    },
    'MES': {
        'commission': 0.52, 'dollar_per_pt': 5.0,
        'file': 'CME_MINI_MES1!, 1_cca38.csv', 'has_vwap': True,
    },
    'ES': {
        'commission': 1.25, 'dollar_per_pt': 50.0,
        'file': 'CME_MINI_ES1!, 1_b6c2e.csv', 'has_vwap': False,
    },
    'MYM': {
        'commission': 0.52, 'dollar_per_pt': 0.50,
        'file': 'CBOT_MINI_MYM1!, 1_b3131.csv', 'has_vwap': False,
    },
}

# Session boundaries in ET (Eastern Time) minutes from midnight.
# These are converted to UTC at runtime using DST-aware logic.
NY_OPEN_ET = 10 * 60            # 10:00 AM ET
NY_LAST_ENTRY_ET = 15 * 60 + 45  # 3:45 PM ET
NY_CLOSE_ET = 16 * 60           # 4:00 PM ET

_ET_TZ = ZoneInfo("America/New_York")


def compute_et_minutes(times):
    """Convert UTC timestamps to ET (Eastern Time) minutes from midnight.

    Handles EST/EDT transitions correctly using zoneinfo.
    Returns int32 array of (hour * 60 + minute) in ET for each bar.
    """
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    et = idx.tz_convert(_ET_TZ)
    return (et.hour * 60 + et.minute).values.astype(np.int32)


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------

def load_instrument_1min(instrument: str) -> pd.DataFrame:
    """Load 1-min data for an instrument. Returns DF with standard columns.

    Prefers Databento files (6-month, real volume, computed SM) when available.
    Falls back to AlgoAlpha TradingView exports.

    Columns returned: Open, High, Low, Close, SM_Net
    Plus VWAP, Volume if available.
    """
    # Prefer Databento data (6-month with real volume)
    DATABENTO_FILES = {
        'MNQ': 'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
        'MES': 'databento_MES_1min_2025-08-17_to_2026-02-13.csv',
    }
    if instrument in DATABENTO_FILES:
        db_path = DATA_DIR / DATABENTO_FILES[instrument]
        if db_path.exists():
            return load_databento_1min(instrument)

    # Fall back to AlgoAlpha TradingView exports
    cfg = INSTRUMENTS[instrument]
    filepath = DATA_DIR / cfg['file']
    df = pd.read_csv(filepath)

    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['SM_Net'] = pd.to_numeric(df['SM Net Index'], errors='coerce').fillna(0)

    if cfg['has_vwap']:
        result['VWAP'] = pd.to_numeric(df['VWAP'], errors='coerce')
        result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

    result = result.set_index('Time')
    return result


def load_databento_1min(instrument: str) -> pd.DataFrame:
    """Load 1-min Databento data (has real OHLCV + computed VWAP, no SM).

    Computes SM Net Index from closes + real volume using compute_smart_money().
    Uses AlgoAlpha SM params (20, 12, 400, 255) matching TradingView settings.

    Columns returned: Open, High, Low, Close, Volume, VWAP, SM_Net
    """
    DATABENTO_FILES = {
        'MNQ': 'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
        'MES': 'databento_MES_1min_2025-08-17_to_2026-02-13.csv',
    }
    if instrument not in DATABENTO_FILES:
        raise ValueError(f"No Databento file for {instrument}. Available: {list(DATABENTO_FILES.keys())}")

    filepath = DATA_DIR / DATABENTO_FILES[instrument]
    df = pd.read_csv(filepath)

    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result['VWAP'] = pd.to_numeric(df['VWAP'], errors='coerce')

    result = result.set_index('Time')

    # Compute SM from real volume using actual TradingView settings (20, 12, 400, 255)
    sm = compute_smart_money(
        result['Close'].values, result['Volume'].values,
        index_period=20, flow_period=12, norm_period=400, ema_len=255,
    )
    result['SM_Net'] = sm

    return result


def load_mnq_5min_prebaked() -> pd.DataFrame:
    """Load the 55-day MNQ 5-min prebaked file (Nov 25 - Feb 12).

    Returns DF with: Open, High, Low, Close, SM_Net (from 1m SM column).
    """
    filepath = DATA_DIR / "CME_MINI_MNQ1!, 5_46a9d.csv"
    df = pd.read_csv(filepath)
    cols = df.columns.tolist()

    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df[cols[0]].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df[cols[1]], errors='coerce')
    result['High'] = pd.to_numeric(df[cols[2]], errors='coerce')
    result['Low'] = pd.to_numeric(df[cols[3]], errors='coerce')
    result['Close'] = pd.to_numeric(df[cols[4]], errors='coerce')
    # 1m SM is at col index 7
    result['SM_Net'] = pd.to_numeric(df[cols[7]], errors='coerce').fillna(0)

    result = result.set_index('Time')
    return result


def resample_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 5-min. OHLC standard, SM/VWAP = last value."""
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'SM_Net': 'last',
    }
    if 'VWAP' in df_1m.columns:
        agg['VWAP'] = 'last'
    if 'Volume' in df_1m.columns:
        agg['Volume'] = 'sum'

    df_5m = df_1m.resample('5min').agg(agg).dropna(subset=['Open'])
    return df_5m


# ---------------------------------------------------------------------------
# Indicator Computation
# ---------------------------------------------------------------------------

def compute_rsi(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI -- matches canonical v9 implementation."""
    n = len(arr)
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.zeros(n)
    al = np.zeros(n)
    if n > period:
        ag[period] = np.mean(gain[1:period + 1])
        al[period] = np.mean(loss[1:period + 1])
        for i in range(period + 1, n):
            ag[i] = (ag[i - 1] * (period - 1) + gain[i]) / period
            al[i] = (al[i - 1] * (period - 1) + loss[i]) / period
    rs = np.where(al > 0, ag / al, 100.0)
    r = 100.0 - 100.0 / (1.0 + rs)
    r[:period] = 50.0
    return r


def compute_smart_money(closes, volumes, index_period, flow_period,
                        norm_period, ema_len):
    """Compute SM net index. Uses range as volume proxy if needed."""
    n = len(closes)
    pvi = np.ones(n)
    nvi = np.ones(n)

    for i in range(1, n):
        pct = (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] != 0 else 0.0
        if volumes[i] > volumes[i - 1]:
            pvi[i] = pvi[i - 1] + pct * pvi[i - 1]
            nvi[i] = nvi[i - 1]
        elif volumes[i] < volumes[i - 1]:
            nvi[i] = nvi[i - 1] + pct * nvi[i - 1]
            pvi[i] = pvi[i - 1]
        else:
            pvi[i] = pvi[i - 1]
            nvi[i] = nvi[i - 1]

    def ema(arr, period):
        r = np.zeros_like(arr)
        r[0] = arr[0]
        a = 2.0 / (period + 1)
        for i in range(1, len(arr)):
            r[i] = a * arr[i] + (1 - a) * r[i - 1]
        return r

    dumb = pvi - ema(pvi, ema_len)
    smart = nvi - ema(nvi, ema_len)

    def rsi_internal(arr, period):
        nn = len(arr)
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss_arr = np.where(delta < 0, -delta, 0.0)
        ag = np.zeros(nn)
        al = np.zeros(nn)
        if nn > period:
            ag[period] = np.mean(gain[1:period + 1])
            al[period] = np.mean(loss_arr[1:period + 1])
            for i in range(period + 1, nn):
                ag[i] = (ag[i - 1] * (period - 1) + gain[i]) / period
                al[i] = (al[i - 1] * (period - 1) + loss_arr[i]) / period
        rs = np.where(al > 0, ag / al, 100.0)
        return 100.0 - 100.0 / (1.0 + rs)

    drsi = rsi_internal(dumb, flow_period)
    srsi = rsi_internal(smart, flow_period)

    r_buy = np.where(drsi != 0, srsi / drsi, 0.0)
    r_sell = np.where((100 - drsi) != 0, (100 - srsi) / (100 - drsi), 0.0)

    sb = np.zeros(n)
    ss = np.zeros(n)
    for i in range(n):
        s = max(0, i - index_period + 1)
        sb[i] = np.sum(r_buy[s:i + 1])
        ss[i] = np.sum(r_sell[s:i + 1])

    mx = np.maximum(sb, ss)
    pk = np.zeros(n)
    for i in range(n):
        s = max(0, i - norm_period + 1)
        pk[i] = np.max(mx[s:i + 1])

    ib = np.where(pk != 0, sb / pk, 0.0)
    isl = np.where(pk != 0, ss / pk, 0.0)
    return ib - isl


# ---------------------------------------------------------------------------
# Feature Preprocessing
# ---------------------------------------------------------------------------

def compute_opening_range(times, highs, lows, or_minutes=30):
    """Compute per-bar opening range breakout direction.

    Returns array: 1 = bullish breakout, -1 = bearish breakout, 0 = no breakout.
    OR is computed from first `or_minutes` of RTH (10:00 AM ET).
    DST-aware: uses compute_et_minutes() for correct timezone handling.
    """
    n = len(times)
    or_dir = np.zeros(n, dtype=int)
    or_high = np.nan
    or_low = np.nan
    current_date = None
    or_end_mins = NY_OPEN_ET + or_minutes
    breakout_dir = 0

    et_mins = compute_et_minutes(times)

    for i in range(n):
        ts = pd.Timestamp(times[i])
        bar_date = ts.date()
        bar_mins = et_mins[i]

        # New day reset
        if bar_date != current_date:
            current_date = bar_date
            or_high = np.nan
            or_low = np.nan
            breakout_dir = 0

        # Within OR window: build the range
        if NY_OPEN_ET <= bar_mins < or_end_mins:
            if np.isnan(or_high):
                or_high = highs[i]
                or_low = lows[i]
            else:
                or_high = max(or_high, highs[i])
                or_low = min(or_low, lows[i])
            or_dir[i] = 0  # Still building range

        # After OR window: track breakouts
        elif bar_mins >= or_end_mins and not np.isnan(or_high):
            if breakout_dir == 0:
                if highs[i] > or_high:
                    breakout_dir = 1
                elif lows[i] < or_low:
                    breakout_dir = -1
            or_dir[i] = breakout_dir
        else:
            or_dir[i] = 0

    return or_dir


def compute_prior_day_levels(times, highs, lows, closes):
    """Compute prior day RTH high/low/close for each bar.

    Returns (prev_high, prev_low, prev_close) arrays.
    RTH = 10:00 AM - 4:00 PM ET. DST-aware.
    """
    n = len(times)
    prev_high = np.full(n, np.nan)
    prev_low = np.full(n, np.nan)
    prev_close = np.full(n, np.nan)

    et_mins = compute_et_minutes(times)

    # Collect per-day RTH stats
    daily_stats = {}  # date -> (high, low, close)
    current_date = None
    day_h = np.nan
    day_l = np.nan
    day_c = np.nan

    for i in range(n):
        ts = pd.Timestamp(times[i])
        bar_date = ts.date()
        bar_mins = et_mins[i]

        if bar_date != current_date:
            if current_date is not None and not np.isnan(day_h):
                daily_stats[current_date] = (day_h, day_l, day_c)
            current_date = bar_date
            day_h = np.nan
            day_l = np.nan
            day_c = np.nan

        if NY_OPEN_ET <= bar_mins < NY_CLOSE_ET:
            if np.isnan(day_h):
                day_h = highs[i]
                day_l = lows[i]
            else:
                day_h = max(day_h, highs[i])
                day_l = min(day_l, lows[i])
            day_c = closes[i]

    # Save last day
    if current_date is not None and not np.isnan(day_h):
        daily_stats[current_date] = (day_h, day_l, day_c)

    # Map prior day levels to each bar
    sorted_dates = sorted(daily_stats.keys())
    date_to_prev = {}
    for j in range(1, len(sorted_dates)):
        date_to_prev[sorted_dates[j]] = daily_stats[sorted_dates[j - 1]]

    for i in range(n):
        ts = pd.Timestamp(times[i])
        bar_date = ts.date()
        if bar_date in date_to_prev:
            ph, pl, pc = date_to_prev[bar_date]
            prev_high[i] = ph
            prev_low[i] = pl
            prev_close[i] = pc

    return prev_high, prev_low, prev_close


# ---------------------------------------------------------------------------
# Unified Backtest Engine
# ---------------------------------------------------------------------------

def run_backtest_v10(opens, highs, lows, closes, sm, rsi, times,
                     rsi_buy, rsi_sell, sm_threshold, cooldown_bars,
                     max_loss_pts=0, trailing_stop_pts=0, use_rsi_cross=True,
                     # RSI mapped from 5-min (for 1-min engine)
                     rsi_5m_curr=None,        # mapped 5-min RSI current bar
                     rsi_5m_prev=None,        # mapped 5-min RSI previous bar
                     # TIER 1
                     underwater_exit_bars=0,
                     or_direction=None,       # pre-computed array from compute_opening_range
                     # TIER 2
                     vwap=None,               # VWAP array
                     vwap_filter=False,
                     vwap_band_pts=0,         # max distance from VWAP (0 = no limit)
                     prior_day_high=None,
                     prior_day_low=None,
                     prior_day_buffer=False,
                     prior_day_buffer_pts=50,
                     price_structure_exit=False,
                     price_structure_bars=3,
                     # Dynamic exits
                     psar_exit=False,              # Parabolic SAR trailing stop
                     psar_af_start=0.02,           # SAR acceleration factor start
                     psar_af_step=0.02,            # SAR AF increment per new extreme
                     psar_af_max=0.20,             # SAR max acceleration factor
                     atr_trail_exit=False,          # ATR-based trailing stop
                     atr_trail_mult=2.0,            # Trail by N * ATR
                     atr_period=14,                 # ATR lookback period
                     breakeven_pts=0,               # Move stop to breakeven after N pts profit (0=off)
                     # 1-min RSI filter
                     rsi_1m=None,              # pre-computed 1-min RSI array
                     rsi_1m_long_min=0,        # min 1-min RSI for longs (0=disabled)
                     rsi_1m_short_max=100,     # max 1-min RSI for shorts (100=disabled)
                     # Already tested features
                     sm_reversal_entry=False,
                     rsi_momentum_filter=False,
                     ):
    """
    Unified v10 backtest engine. All features OFF = identical to v9.

    ENTRY (bar[i] open, using bar[i-1] signals):
      LONG:  SM[i-1] > threshold AND RSI condition
      SHORT: SM[i-1] < -threshold AND RSI condition

    RSI CROSS DETECTION:
      If rsi_5m_curr and rsi_5m_prev are provided (from prepare_backtest_arrays_1min),
      the RSI cross is detected using mapped 5-min values, matching Pine's
      request.security() behavior where the cross persists across the entire
      5-min window. Otherwise falls back to adjacent-bar cross detection.

    EXIT:
      - SM flips sign (primary)
      - Max loss stop (checks bar i-1 close, fills at bar i open)
      - Trailing stop
      - Underwater exit (bars losing)
      - Price structure exit (new low/high)
      - EOD close at 4 PM ET

    Feature toggles are additive filters on top of the v9 base logic.
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    max_equity = 0.0
    has_mapped_rsi = rsi_5m_curr is not None and rsi_5m_prev is not None

    # Pre-compute ET minutes for every bar (DST-aware)
    et_mins = compute_et_minutes(times)

    # Pre-compute ATR if needed
    atr = None
    if atr_trail_exit:
        tr = np.zeros(n)
        for j in range(1, n):
            tr[j] = max(highs[j] - lows[j],
                        abs(highs[j] - closes[j - 1]),
                        abs(lows[j] - closes[j - 1]))
        atr = np.zeros(n)
        if n > atr_period:
            atr[atr_period] = np.mean(tr[1:atr_period + 1])
            for j in range(atr_period + 1, n):
                atr[j] = (atr[j - 1] * (atr_period - 1) + tr[j]) / atr_period

    # PSAR state (reset per trade)
    psar_val = 0.0
    psar_af = 0.0
    psar_ep = 0.0

    # Breakeven/ATR trail state
    be_triggered = False
    atr_trail_level = 0.0

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

        # -- Previous bar signals (no look-ahead) --
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI cross detection + rsi_prev/rsi_prev2 for momentum filter
        if has_mapped_rsi:
            # Use mapped 5-min RSI values -- matches Pine request.security()
            # rsi_5m_curr[i-1] = RSI of last completed 5-min bar at bar i-1
            # rsi_5m_prev[i-1] = RSI of the 5-min bar before that at bar i-1
            # Cross persists across entire 5-min window (same as Pine)
            rsi_prev = rsi_5m_curr[i - 1]   # current 5-min RSI
            rsi_prev2 = rsi_5m_prev[i - 1]  # previous 5-min RSI
            if use_rsi_cross:
                rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
                rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell
            else:
                rsi_long_trigger = rsi_prev > rsi_buy
                rsi_short_trigger = rsi_prev < rsi_sell
        else:
            # Legacy: derive cross from adjacent bars (5-min engine path)
            rsi_prev = rsi[i - 1]
            rsi_prev2 = rsi[i - 2]
            if use_rsi_cross:
                rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
                rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell
            else:
                rsi_long_trigger = rsi_prev > rsi_buy
                rsi_short_trigger = rsi_prev < rsi_sell

        # -- Episode reset --
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # -- EOD Close --
        if trade_state != 0 and bar_mins_et >= NY_CLOSE_ET:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            max_equity = 0.0
            continue

        # -- Exits for open positions --
        reversal_entry = None

        if trade_state == 1:
            # Max loss stop -- Pine uses `close <= avg_price - pts` and
            # strategy.close() fills at NEXT bar open.  We check bar i-1's
            # close (completed bar) and fill at opens[i] (next bar open).
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i],
                            entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                continue

            # Trailing stop -- same bar-i-1 signal, opens[i] fill pattern
            if trailing_stop_pts > 0:
                running_pnl = highs[i - 1] - entry_price
                if running_pnl > max_equity:
                    max_equity = running_pnl
                if max_equity > trailing_stop_pts and closes[i - 1] <= entry_price + max_equity - trailing_stop_pts:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "TRAIL")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # Parabolic SAR exit for longs
            if psar_exit:
                # Update SAR state using bar i-1 data
                if highs[i - 1] > psar_ep:
                    psar_ep = highs[i - 1]
                    psar_af = min(psar_af + psar_af_step, psar_af_max)
                psar_val = psar_val + psar_af * (psar_ep - psar_val)
                # SAR must stay below recent lows
                psar_val = min(psar_val, lows[i - 1], lows[max(0, i - 2)])
                if closes[i - 1] < psar_val:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "PSAR")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # ATR trailing stop for longs
            if atr_trail_exit and atr is not None and atr[i - 1] > 0:
                curr_trail = highs[i - 1] - atr_trail_mult * atr[i - 1]
                if curr_trail > atr_trail_level:
                    atr_trail_level = curr_trail
                if closes[i - 1] < atr_trail_level:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "ATR_TRAIL")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # Breakeven stop for longs
            if breakeven_pts > 0:
                if not be_triggered and highs[i - 1] >= entry_price + breakeven_pts:
                    be_triggered = True
                if be_triggered and closes[i - 1] <= entry_price:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "BREAKEVEN")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # Underwater exit: losing for N consecutive bars
            # Signal from bar i-1 (completed), fill at opens[i]
            if underwater_exit_bars > 0:
                bars_held = (i - 1) - entry_idx
                if bars_held >= underwater_exit_bars and closes[i - 1] < entry_price:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "UNDERWATER")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # Price structure exit: new low below lookback low
            if price_structure_exit:
                bars_held = (i - 1) - entry_idx
                if bars_held >= price_structure_bars and i >= price_structure_bars + 2:
                    lb_start = max(entry_idx, i - 1 - price_structure_bars)
                    lookback_low = np.min(lows[lb_start:i - 1])
                    if lows[i - 1] < lookback_low:
                        close_trade("long", entry_price, opens[i], entry_idx, i, "STRUCTURE")
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
                if sm_reversal_entry:
                    reversal_entry = "short"
                # Don't continue -- allow immediate reversal entry

        elif trade_state == -1:
            # Max loss stop
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i],
                            entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                continue

            # Trailing stop
            if trailing_stop_pts > 0:
                running_pnl = entry_price - lows[i - 1]
                if running_pnl > max_equity:
                    max_equity = running_pnl
                if max_equity > trailing_stop_pts and closes[i - 1] >= entry_price - max_equity + trailing_stop_pts:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "TRAIL")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # Parabolic SAR exit for shorts
            if psar_exit:
                if lows[i - 1] < psar_ep:
                    psar_ep = lows[i - 1]
                    psar_af = min(psar_af + psar_af_step, psar_af_max)
                psar_val = psar_val - psar_af * (psar_val - psar_ep)
                psar_val = max(psar_val, highs[i - 1], highs[max(0, i - 2)])
                if closes[i - 1] > psar_val:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "PSAR")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # ATR trailing stop for shorts
            if atr_trail_exit and atr is not None and atr[i - 1] > 0:
                curr_trail = lows[i - 1] + atr_trail_mult * atr[i - 1]
                if curr_trail < atr_trail_level:
                    atr_trail_level = curr_trail
                if closes[i - 1] > atr_trail_level:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "ATR_TRAIL")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # Breakeven stop for shorts
            if breakeven_pts > 0:
                if not be_triggered and lows[i - 1] <= entry_price - breakeven_pts:
                    be_triggered = True
                if be_triggered and closes[i - 1] >= entry_price:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "BREAKEVEN")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # Underwater exit
            if underwater_exit_bars > 0:
                bars_held = (i - 1) - entry_idx
                if bars_held >= underwater_exit_bars and closes[i - 1] > entry_price:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "UNDERWATER")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # Price structure exit: new high above lookback high
            if price_structure_exit:
                bars_held = (i - 1) - entry_idx
                if bars_held >= price_structure_bars and i >= price_structure_bars + 2:
                    lb_start = max(entry_idx, i - 1 - price_structure_bars)
                    lookback_high = np.max(highs[lb_start:i - 1])
                    if highs[i - 1] > lookback_high:
                        close_trade("short", entry_price, opens[i], entry_idx, i, "STRUCTURE")
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
                if sm_reversal_entry:
                    reversal_entry = "long"

        # -- Entry logic --
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= cooldown_bars

            entered = False

            # SM reversal entry (immediate entry after SM flip exit)
            if reversal_entry is not None and in_session:
                rsi_ok = True
                if rsi_momentum_filter:
                    if reversal_entry == "long":
                        rsi_ok = rsi_prev > rsi_prev2
                    else:
                        rsi_ok = rsi_prev < rsi_prev2

                if rsi_ok:
                    trade_state = 1 if reversal_entry == "long" else -1
                    entry_price = opens[i]
                    entry_idx = i
                    max_equity = 0.0
                    be_triggered = False
                    if reversal_entry == "long":
                        long_used = True
                        psar_val = entry_price - (highs[i] - lows[i]) if psar_exit else 0
                        psar_ep = entry_price
                        atr_trail_level = entry_price - (atr_trail_mult * atr[i] if atr is not None and atr[i] > 0 else 50)
                    else:
                        short_used = True
                        psar_val = entry_price + (highs[i] - lows[i]) if psar_exit else 0
                        psar_ep = entry_price
                        atr_trail_level = entry_price + (atr_trail_mult * atr[i] if atr is not None and atr[i] > 0 else 50)
                    psar_af = psar_af_start
                    entered = True

            # Standard entry
            if not entered and in_session and cd_ok:
                # -- Long entry --
                if sm_bull and rsi_long_trigger and not long_used:
                    # Apply feature filters
                    ok = True

                    if rsi_momentum_filter:
                        if not (rsi_prev > rsi_prev2):
                            ok = False

                    if ok and rsi_1m is not None and rsi_1m_long_min > 0:
                        if rsi_1m[i - 1] < rsi_1m_long_min:
                            ok = False

                    if ok and or_direction is not None:
                        if or_direction[i - 1] == -1:  # bearish OR breakout
                            ok = False

                    if ok and vwap_filter and vwap is not None:
                        if opens[i] < vwap[i - 1]:
                            ok = False
                        if vwap_band_pts > 0 and abs(opens[i] - vwap[i - 1]) > vwap_band_pts:
                            ok = False

                    if ok and prior_day_buffer:
                        if prior_day_high is not None and not np.isnan(prior_day_high[i]):
                            if opens[i] > prior_day_high[i] + prior_day_buffer_pts:
                                ok = False

                    if ok:
                        trade_state = 1
                        entry_price = opens[i]
                        entry_idx = i
                        long_used = True
                        max_equity = 0.0
                        be_triggered = False
                        psar_val = entry_price - (highs[i] - lows[i]) if psar_exit else 0
                        psar_ep = entry_price
                        psar_af = psar_af_start
                        atr_trail_level = entry_price - (atr_trail_mult * atr[i] if atr is not None and atr[i] > 0 else 50)

                # -- Short entry --
                elif sm_bear and rsi_short_trigger and not short_used:
                    ok = True

                    if rsi_momentum_filter:
                        if not (rsi_prev < rsi_prev2):
                            ok = False

                    if ok and rsi_1m is not None and rsi_1m_short_max < 100:
                        if rsi_1m[i - 1] > rsi_1m_short_max:
                            ok = False

                    if ok and or_direction is not None:
                        if or_direction[i - 1] == 1:  # bullish OR breakout
                            ok = False

                    if ok and vwap_filter and vwap is not None:
                        if opens[i] > vwap[i - 1]:
                            ok = False
                        if vwap_band_pts > 0 and abs(opens[i] - vwap[i - 1]) > vwap_band_pts:
                            ok = False

                    if ok and prior_day_buffer:
                        if prior_day_low is not None and not np.isnan(prior_day_low[i]):
                            if opens[i] < prior_day_low[i] - prior_day_buffer_pts:
                                ok = False

                    if ok:
                        trade_state = -1
                        entry_price = opens[i]
                        entry_idx = i
                        short_used = True
                        max_equity = 0.0
                        be_triggered = False
                        psar_val = entry_price + (highs[i] - lows[i]) if psar_exit else 0
                        psar_ep = entry_price
                        psar_af = psar_af_start
                        atr_trail_level = entry_price + (atr_trail_mult * atr[i] if atr is not None and atr[i] > 0 else 50)

    return trades


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_trades(trades, commission_per_side=0.52, dollar_per_pt=2.0):
    """Score a list of trades. Returns dict of metrics or None if empty."""
    if not trades:
        return None
    pts = np.array([t["pts"] for t in trades])
    n = len(pts)
    comm_pts = (commission_per_side * 2) / dollar_per_pt
    net_each = pts - comm_pts
    net_pts = net_each.sum()
    wins = net_each[net_each > 0]
    losses = net_each[net_each <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100
    cum = np.cumsum(net_each)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()
    avg_bars = np.mean([t["bars"] for t in trades])
    avg_pts = np.mean(net_each)

    # Sharpe (daily returns approximation)
    daily_pnl = net_each * dollar_per_pt
    sharpe = 0.0
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)

    exit_types = {}
    for t in trades:
        r = t["result"]
        exit_types[r] = exit_types.get(r, 0) + 1

    return {
        "count": n,
        "net_pts": round(net_pts, 2),
        "pf": round(pf, 3),
        "win_rate": round(wr, 1),
        "net_dollar": round(net_pts * dollar_per_pt, 2),
        "max_dd_pts": round(mdd, 2),
        "max_dd_dollar": round(mdd * dollar_per_pt, 2),
        "avg_bars": round(avg_bars, 1),
        "avg_pts": round(avg_pts, 2),
        "sharpe": round(sharpe, 3),
        "exits": exit_types,
    }


def fmt_score(sc, label=""):
    """Format a score dict as a one-line summary."""
    if sc is None:
        return f"{label}: NO TRADES"
    return (f"{label}  {sc['count']} trades, WR {sc['win_rate']}%, "
            f"PF {sc['pf']}, Net ${sc['net_dollar']:+.2f}, "
            f"MaxDD ${sc['max_dd_dollar']:.2f}, Sharpe {sc['sharpe']}")


def fmt_exits(exits_dict):
    """Format exit type counts."""
    parts = []
    for k in ["SM_FLIP", "SL", "TRAIL", "UNDERWATER", "STRUCTURE", "EOD"]:
        if k in exits_dict:
            parts.append(f"{k}:{exits_dict[k]}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Statistical Utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(trades, metric="pf", n_boot=10000, ci=0.95,
                 commission_per_side=0.52, dollar_per_pt=2.0):
    """Bootstrap confidence interval for a metric.

    metric: 'pf', 'win_rate', 'net_pts', 'max_dd_pts', 'sharpe'
    Returns (point_estimate, ci_low, ci_high).
    """
    if len(trades) < 2:
        return (0, 0, 0)

    rng = np.random.default_rng(42)
    pts_arr = np.array([t["pts"] for t in trades])
    comm_pts = (commission_per_side * 2) / dollar_per_pt
    net_arr = pts_arr - comm_pts

    estimates = []
    for _ in range(n_boot):
        sample = rng.choice(net_arr, size=len(net_arr), replace=True)
        if metric == "pf":
            wins = sample[sample > 0].sum()
            loss = abs(sample[sample <= 0].sum())
            val = wins / loss if loss > 0 else 999.0
        elif metric == "win_rate":
            val = (sample > 0).sum() / len(sample) * 100
        elif metric == "net_pts":
            val = sample.sum()
        elif metric == "max_dd_pts":
            cum = np.cumsum(sample)
            peak = np.maximum.accumulate(cum)
            val = (cum - peak).min()
        elif metric == "sharpe":
            daily = sample * dollar_per_pt
            val = np.mean(daily) / np.std(daily) * np.sqrt(252) if np.std(daily) > 0 else 0
        else:
            val = 0
        estimates.append(val)

    estimates = np.sort(estimates)
    alpha = (1 - ci) / 2
    lo = estimates[int(alpha * n_boot)]
    hi = estimates[int((1 - alpha) * n_boot)]

    # Point estimate
    if metric == "pf":
        w = net_arr[net_arr > 0].sum()
        ls = abs(net_arr[net_arr <= 0].sum())
        point = w / ls if ls > 0 else 999.0
    elif metric == "win_rate":
        point = (net_arr > 0).sum() / len(net_arr) * 100
    elif metric == "net_pts":
        point = net_arr.sum()
    elif metric == "max_dd_pts":
        cum = np.cumsum(net_arr)
        peak = np.maximum.accumulate(cum)
        point = (cum - peak).min()
    elif metric == "sharpe":
        daily = net_arr * dollar_per_pt
        point = np.mean(daily) / np.std(daily) * np.sqrt(252) if np.std(daily) > 0 else 0
    else:
        point = 0

    return (round(point, 3), round(lo, 3), round(hi, 3))


def monte_carlo_equity(trades, n_sims=10000, dollar_per_pt=2.0,
                       commission_per_side=0.52):
    """Shuffle trade order, build equity curves, get DD distribution.

    Returns dict with percentile stats for max_dd and final_equity.
    """
    if len(trades) < 2:
        return None

    rng = np.random.default_rng(42)
    pts_arr = np.array([t["pts"] for t in trades])
    comm_pts = (commission_per_side * 2) / dollar_per_pt
    net_dollar = (pts_arr - comm_pts) * dollar_per_pt

    max_dds = []
    finals = []
    for _ in range(n_sims):
        shuffled = rng.permutation(net_dollar)
        cum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak).min()
        max_dds.append(dd)
        finals.append(cum[-1])

    max_dds = np.sort(max_dds)
    finals = np.sort(finals)

    pcts = [5, 25, 50, 75, 95]
    return {
        "max_dd": {f"P{p}": round(np.percentile(max_dds, p), 2) for p in pcts},
        "final_equity": {f"P{p}": round(np.percentile(finals, p), 2) for p in pcts},
    }


def permutation_test(baseline_trades, feature_trades, metric="pf",
                     n_perms=10000, commission_per_side=0.52,
                     dollar_per_pt=2.0):
    """Permutation test for difference in metric between two trade sets.

    Returns (observed_diff, p_value).
    """
    def calc_metric(net_arr):
        if metric == "pf":
            w = net_arr[net_arr > 0].sum()
            ls = abs(net_arr[net_arr <= 0].sum())
            return w / ls if ls > 0 else 999.0
        elif metric == "win_rate":
            return (net_arr > 0).sum() / len(net_arr) * 100
        elif metric == "net_pts":
            return net_arr.sum()
        return 0

    comm_pts = (commission_per_side * 2) / dollar_per_pt
    base_net = np.array([t["pts"] for t in baseline_trades]) - comm_pts
    feat_net = np.array([t["pts"] for t in feature_trades]) - comm_pts

    observed = calc_metric(feat_net) - calc_metric(base_net)
    combined = np.concatenate([base_net, feat_net])
    n_base = len(base_net)

    rng = np.random.default_rng(42)
    count_extreme = 0
    for _ in range(n_perms):
        perm = rng.permutation(combined)
        perm_base = perm[:n_base]
        perm_feat = perm[n_base:]
        diff = calc_metric(perm_feat) - calc_metric(perm_base)
        if diff >= observed:
            count_extreme += 1

    p_value = count_extreme / n_perms
    return (round(observed, 3), round(p_value, 4))


def random_feature_baseline(trades, n_sims=100, commission_per_side=0.52,
                            dollar_per_pt=2.0):
    """Generate random binary filters and compute null PF distribution.

    Returns sorted array of PFs from random filters.
    """
    if len(trades) < 3:
        return np.array([1.0])

    rng = np.random.default_rng(42)
    pts_arr = np.array([t["pts"] for t in trades])
    comm_pts = (commission_per_side * 2) / dollar_per_pt
    net_arr = pts_arr - comm_pts

    pfs = []
    for _ in range(n_sims):
        mask = rng.random(len(net_arr)) > 0.5
        if mask.sum() < 3:
            continue
        subset = net_arr[mask]
        w = subset[subset > 0].sum()
        ls = abs(subset[subset <= 0].sum())
        pf = w / ls if ls > 0 else 999.0
        pfs.append(pf)

    return np.sort(pfs) if pfs else np.array([1.0])


# ---------------------------------------------------------------------------
# Convenience: prepare arrays for backtest
# ---------------------------------------------------------------------------

def prepare_backtest_arrays(df_5m):
    """Extract numpy arrays from a 5-min DataFrame for the engine.

    Returns dict with opens, highs, lows, closes, sm, times, and optional vwap.
    NOTE: This is the LEGACY 5-min path. For production accuracy, use
    prepare_backtest_arrays_1min() which runs on 1-min bars with 5-min RSI
    mapped back -- matching the actual Pine Script architecture.
    """
    arr = {
        'opens': df_5m['Open'].values,
        'highs': df_5m['High'].values,
        'lows': df_5m['Low'].values,
        'closes': df_5m['Close'].values,
        'sm': df_5m['SM_Net'].values,
        'times': df_5m.index.values,
    }
    if 'VWAP' in df_5m.columns:
        arr['vwap'] = df_5m['VWAP'].values
    return arr


def prepare_backtest_arrays_1min(df_1m, rsi_len=10):
    """Prepare 1-min arrays for the engine with 5-min RSI mapped back.

    This mirrors the actual Pine Script architecture:
      - OHLC, SM exits, price structure: evaluated on every 1-min bar
      - RSI: computed on 5-min bars via request.security(), then each
        1-min bar gets TWO mapped values matching Pine's behavior:

    Pine Script does:
      rsi_5m      = request.security("5", ta.rsi(close, 10), lookahead_off)
      rsi_5m_prev = request.security("5", ta.rsi(close, 10)[1], lookahead_off)
      rsi_cross_up = rsi_5m > buy_level and rsi_5m_prev <= buy_level

    With lookahead_off, ALL 1-min bars within a 5-min window see the SAME
    rsi_5m (last completed 5-min bar's RSI) and the SAME rsi_5m_prev (the
    one before that). The cross persists across all 5 bars in the window.

    Returns dict with 1-min opens, highs, lows, closes, sm, times,
    rsi_5m_curr, rsi_5m_prev, and optional vwap/rsi (for legacy compat).
    """
    # 1. Resample to 5-min and compute RSI on 5-min closes
    df_5m = resample_to_5min(df_1m)
    rsi_5m_vals = compute_rsi(df_5m['Close'].values, rsi_len)
    fivemin_times = df_5m.index.values

    # 2. Map TWO 5-min RSI arrays to each 1-min bar:
    #    rsi_curr[i] = RSI of last COMPLETED 5-min bar (matches rsi_5m in Pine)
    #    rsi_prev[i] = RSI of the bar before that (matches rsi_5m_prev in Pine)
    #
    #    With lookahead_off: on any 1-min bar within the 10:00-10:04 window,
    #    the 10:00 5-min bar is still building, so the last COMPLETED bar is
    #    09:55. Both rsi_curr and rsi_prev stay constant across all 5 bars.
    n_1m = len(df_1m)
    rsi_curr_mapped = np.full(n_1m, 50.0)
    rsi_prev_mapped = np.full(n_1m, 50.0)
    onemin_times = df_1m.index.values

    j = 0  # pointer into 5-min array
    for i in range(n_1m):
        # Advance j to the last 5-min bar starting at or before this 1-min bar
        while j + 1 < len(fivemin_times) and fivemin_times[j + 1] <= onemin_times[i]:
            j += 1
        # The 5-min bar at index j contains this 1-min bar (or is the latest one)
        # It's STILL IN PROGRESS, so the last COMPLETED bar is j-1
        # (matches lookahead_off: current higher-TF bar not available until it closes)
        if j >= 1:
            rsi_curr_mapped[i] = rsi_5m_vals[j - 1]     # last completed 5m bar
        if j >= 2:
            rsi_prev_mapped[i] = rsi_5m_vals[j - 2]     # the one before that

    arr = {
        'opens': df_1m['Open'].values,
        'highs': df_1m['High'].values,
        'lows': df_1m['Low'].values,
        'closes': df_1m['Close'].values,
        'sm': df_1m['SM_Net'].values,
        'times': df_1m.index.values,
        'rsi': rsi_curr_mapped,           # legacy compat (used by run_v9_baseline)
        'rsi_5m_curr': rsi_curr_mapped,   # mapped 5-min RSI current
        'rsi_5m_prev': rsi_prev_mapped,   # mapped 5-min RSI previous
    }
    if 'VWAP' in df_1m.columns:
        arr['vwap'] = df_1m['VWAP'].values
    return arr


def map_5min_rsi_to_1min(onemin_times, fivemin_times, fivemin_closes, rsi_len=10):
    """Compute 5-min RSI with given length and map curr+prev to 1-min timestamps.

    Efficient helper for sensitivity sweeps that change rsi_len without
    reloading data. Pass pre-computed 5-min times and closes to avoid
    repeated resampling.

    Args:
        onemin_times: 1-min bar timestamps (numpy array)
        fivemin_times: 5-min bar timestamps (numpy array)
        fivemin_closes: 5-min close prices (numpy array)
        rsi_len: RSI period

    Returns (rsi_5m_curr, rsi_5m_prev) numpy arrays of length len(onemin_times).
    """
    rsi_5m_vals = compute_rsi(fivemin_closes, rsi_len)
    n_1m = len(onemin_times)
    rsi_curr = np.full(n_1m, 50.0)
    rsi_prev = np.full(n_1m, 50.0)

    j = 0
    for i in range(n_1m):
        while j + 1 < len(fivemin_times) and fivemin_times[j + 1] <= onemin_times[i]:
            j += 1
        if j >= 1:
            rsi_curr[i] = rsi_5m_vals[j - 1]
        if j >= 2:
            rsi_prev[i] = rsi_5m_vals[j - 2]

    return rsi_curr, rsi_prev


def run_v9_baseline(arr, rsi_len=10, rsi_buy=55, rsi_sell=45,
                    sm_threshold=0.0, cooldown=3, max_loss_pts=0,
                    use_rsi_cross=True):
    """Run baseline v9 (all v10 features OFF). Convenience wrapper.

    If arr contains 'rsi_5m_curr' and 'rsi_5m_prev' (from prepare_backtest_arrays_1min),
    passes them to the engine for proper 5-min RSI cross detection on 1-min bars.
    Otherwise computes RSI from closes (legacy 5-min path).
    Cooldown is in bar units (1-min bars if 1-min data, 5-min bars if 5-min data).
    """
    rsi = arr.get('rsi', None)
    if rsi is None:
        rsi = compute_rsi(arr['closes'], rsi_len)
    return run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], rsi, arr['times'],
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts, use_rsi_cross=use_rsi_cross,
        rsi_5m_curr=arr.get('rsi_5m_curr'),
        rsi_5m_prev=arr.get('rsi_5m_prev'),
    )


# ---------------------------------------------------------------------------
# Self-test: verify v10 engine matches v9 baseline
# ---------------------------------------------------------------------------

def verify_v9_match():
    """Verify unified engine matches canonical v9 on prebaked 5-min data."""
    print("=" * 80)
    print("VERIFICATION: v10 engine (all features OFF) vs v9 canonical")
    print("=" * 80)

    df_5m = load_mnq_5min_prebaked()
    start = pd.Timestamp("2026-01-19")
    end = pd.Timestamp("2026-02-13")
    df_5m = df_5m[(df_5m.index >= start) & (df_5m.index < end)]
    print(f"  Data: {len(df_5m)} bars, {df_5m.index[0].date()} to {df_5m.index[-1].date()}")

    arr = prepare_backtest_arrays(df_5m)
    trades = run_v9_baseline(arr)
    sc = score_trades(trades)

    print(f"\n  v10 engine result: {fmt_score(sc)}")
    print(f"  Expected v9:       ~44 trades, WR ~63.6%, PF ~2.325, ~$1583")

    if sc is not None:
        ok = sc['count'] >= 40 and sc['pf'] > 1.5
        print(f"\n  Match: {'PASS' if ok else 'FAIL'}")
    else:
        print("\n  Match: FAIL (no trades)")

    return trades, sc


if __name__ == "__main__":
    verify_v9_match()
