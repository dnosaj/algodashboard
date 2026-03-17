#!/usr/bin/env python3
"""
ATR-Scaled TP/SL Exits Test
============================
Tests whether scaling TP/SL relative to rolling 14-day ATR improves
strategy performance across different volatility regimes.

Hypothesis: Fixed TP/SL (e.g., TP=7, SL=40) works well in the dev period
(avg daily ATR ~370 pts) but degrades in lower-volatility periods (avg ATR
~280 pts). Scaling TP/SL as a fraction of ATR should auto-adapt.

Tests 3 strategies across 3 one-year periods (prior2, prior, dev).

Usage:
    cd backtesting_engine && python3 -u strategies/atr_scaled_exits_test.py
"""

import sys
import os
import warnings
from pathlib import Path

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

# --- Path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    compute_rsi,
    compute_et_minutes,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
    NY_OPEN_ET,
    NY_CLOSE_ET,
    NY_LAST_ENTRY_ET,
)

from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
)

from zoneinfo import ZoneInfo
_ET = ZoneInfo("America/New_York")

def P(msg=""):
    """Print with immediate flush."""
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"

# MNQ SM params (shared by vScalpA, vScalpB)
MNQ_SM_INDEX = 10
MNQ_SM_FLOW = 12
MNQ_SM_NORM = 200
MNQ_SM_EMA = 100
MNQ_DOLLAR_PER_PT = 2.0
MNQ_COMMISSION = 0.52

# MES SM params
MES_SM_INDEX = 20
MES_SM_FLOW = 12
MES_SM_NORM = 400
MES_SM_EMA = 255
MES_DOLLAR_PER_PT = 5.0
MES_COMMISSION = 1.25

# vScalpA (V15) params
VSCALPA_RSI_LEN = 8
VSCALPA_RSI_BUY = 60
VSCALPA_RSI_SELL = 40
VSCALPA_SM_THRESHOLD = 0.0
VSCALPA_COOLDOWN = 20
VSCALPA_MAX_LOSS_PTS = 40
VSCALPA_TP_PTS = 7
VSCALPA_ENTRY_END_ET = 13 * 60  # 13:00 ET

# vScalpB params
VSCALPB_RSI_LEN = 8
VSCALPB_RSI_BUY = 55
VSCALPB_RSI_SELL = 45
VSCALPB_SM_THRESHOLD = 0.25
VSCALPB_COOLDOWN = 20
VSCALPB_MAX_LOSS_PTS = 10
VSCALPB_TP_PTS = 3

# MES v2 params
MESV2_RSI_LEN = 12
MESV2_RSI_BUY = 55
MESV2_RSI_SELL = 45
MESV2_SM_THRESHOLD = 0.0
MESV2_COOLDOWN = 25
MESV2_MAX_LOSS_PTS = 35
MESV2_TP_PTS = 20
MESV2_EOD_ET = 16 * 60          # 16:00 ET
MESV2_ENTRY_END_ET = 14 * 60 + 15  # 14:15 ET
MESV2_BREAKEVEN_BARS = 75

TICK_SIZE = 0.25


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_prior_data(instrument, period="prior"):
    """Load prior year data for an instrument."""
    if period == "prior":
        filepath = DATA_DIR / f"prior_databento_{instrument}_1min_2024-02-17_to_2025-02-16.csv"
    elif period == "prior2":
        filepath = DATA_DIR / f"prior2_databento_{instrument}_1min_2023-02-17_to_2024-02-16.csv"
    else:
        raise ValueError(f"Unknown period: {period}")

    df_raw = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df_raw['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df_raw['open'], errors='coerce')
    result['High'] = pd.to_numeric(df_raw['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df_raw['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df_raw['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df_raw['Volume'], errors='coerce').fillna(0)
    if 'VWAP' in df_raw.columns:
        result['VWAP'] = pd.to_numeric(df_raw['VWAP'], errors='coerce')
    result = result.set_index('Time')
    result = result[~result.index.duplicated(keep='first')]
    result = result.sort_index()
    return result


# ---------------------------------------------------------------------------
# Daily ATR computation
# ---------------------------------------------------------------------------

def compute_daily_atr(df, lookback_days=14):
    """Compute rolling 14-day ATR from 1-min bars.

    Returns array of length len(df) where each bar has the ATR(14)
    of the prior 14 completed trading days.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    et_dates = idx.tz_convert(_ET).date

    # Compute daily high, low, close
    daily_data = {}
    for i in range(len(df)):
        d = et_dates[i]
        if d not in daily_data:
            daily_data[d] = {'high': df['High'].iloc[i], 'low': df['Low'].iloc[i],
                             'close': df['Close'].iloc[i]}
        else:
            daily_data[d]['high'] = max(daily_data[d]['high'], df['High'].iloc[i])
            daily_data[d]['low'] = min(daily_data[d]['low'], df['Low'].iloc[i])
            daily_data[d]['close'] = df['Close'].iloc[i]

    sorted_dates = sorted(daily_data.keys())

    # Compute true range for each day
    daily_tr = {}
    for j, d in enumerate(sorted_dates):
        h = daily_data[d]['high']
        l = daily_data[d]['low']
        if j > 0:
            prev_close = daily_data[sorted_dates[j - 1]]['close']
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        else:
            tr = h - l
        daily_tr[d] = tr

    # Compute rolling ATR per day (using prior N completed days)
    daily_atr = {}
    for j, d in enumerate(sorted_dates):
        start_j = max(0, j - lookback_days)
        end_j = j  # exclusive: don't include current day
        if end_j <= start_j:
            daily_atr[d] = daily_tr[d]
        else:
            trs = [daily_tr[sorted_dates[k]] for k in range(start_j, end_j)]
            daily_atr[d] = np.mean(trs)

    # Map back to per-bar
    atr_array = np.zeros(len(df))
    for i in range(len(df)):
        atr_array[i] = daily_atr[et_dates[i]]

    return atr_array


# ---------------------------------------------------------------------------
# ATR-scaled backtest engine
# ---------------------------------------------------------------------------

def run_backtest_atr_scaled(opens, highs, lows, closes, sm, times,
                            rsi_5m_curr, rsi_5m_prev,
                            rsi_buy, rsi_sell, sm_threshold,
                            cooldown_bars, tp_frac, sl_frac,
                            atr_array, tick_size=0.25,
                            eod_minutes_et=NY_CLOSE_ET,
                            breakeven_after_bars=0,
                            entry_end_et=NY_LAST_ENTRY_ET):
    """Backtest with ATR-scaled TP/SL. Same entry logic as run_backtest_tp_exit."""
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    trade_tp = 0.0
    trade_sl = 0.0

    et_mins = compute_et_minutes(times)

    def round_to_tick(val, tick):
        return round(val / tick) * tick

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result, tp, sl):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
            "atr_at_entry": float(atr_array[entry_i]),
            "tp_pts": float(tp), "sl_pts": float(sl),
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # EOD close
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD",
                       trade_tp, trade_sl)
            trade_state = 0
            exit_bar = i
            continue

        # Exits
        if trade_state == 1:
            if trade_sl > 0 and closes[i - 1] <= entry_price - trade_sl:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL",
                           trade_tp, trade_sl)
                trade_state = 0; exit_bar = i; continue

            if trade_tp > 0 and closes[i - 1] >= entry_price + trade_tp:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP",
                           trade_tp, trade_sl)
                trade_state = 0; exit_bar = i; continue

            if breakeven_after_bars > 0:
                if (i - 1) - entry_idx >= breakeven_after_bars:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "BE_TIME",
                               trade_tp, trade_sl)
                    trade_state = 0; exit_bar = i; continue

        elif trade_state == -1:
            if trade_sl > 0 and closes[i - 1] >= entry_price + trade_sl:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL",
                           trade_tp, trade_sl)
                trade_state = 0; exit_bar = i; continue

            if trade_tp > 0 and closes[i - 1] <= entry_price - trade_tp:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP",
                           trade_tp, trade_sl)
                trade_state = 0; exit_bar = i; continue

            if breakeven_after_bars > 0:
                if (i - 1) - entry_idx >= breakeven_after_bars:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "BE_TIME",
                               trade_tp, trade_sl)
                    trade_state = 0; exit_bar = i; continue

        # Entries
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
                    atr_val = atr_array[i]
                    trade_tp = max(round_to_tick(atr_val * tp_frac, tick_size), tick_size)
                    trade_sl = max(round_to_tick(atr_val * sl_frac, tick_size), tick_size)
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    atr_val = atr_array[i]
                    trade_tp = max(round_to_tick(atr_val * tp_frac, tick_size), tick_size)
                    trade_sl = max(round_to_tick(atr_val * sl_frac, tick_size), tick_size)

    return trades


# ---------------------------------------------------------------------------
# Scoring with ATR diagnostics
# ---------------------------------------------------------------------------

def score_atr_trades(trades, commission_per_side=0.52, dollar_per_pt=2.0):
    """Score trades and add ATR diagnostics."""
    sc = score_trades(trades, commission_per_side=commission_per_side,
                      dollar_per_pt=dollar_per_pt)
    if sc is None:
        return None

    atrs = np.array([t.get("atr_at_entry", 0) for t in trades])
    tps = np.array([t.get("tp_pts", 0) for t in trades])
    sls = np.array([t.get("sl_pts", 0) for t in trades])

    sc["avg_atr"] = round(np.mean(atrs), 1)
    sc["avg_tp_pts"] = round(np.mean(tps), 2)
    sc["avg_sl_pts"] = round(np.mean(sls), 2)
    return sc


# ---------------------------------------------------------------------------
# Pre-computed data bundle
# ---------------------------------------------------------------------------

def prepare_bundle(df, sm_index, sm_flow, sm_norm, sm_ema, rsi_len):
    """Pre-compute SM, RSI (5min mapped to 1min), and daily ATR once.

    Returns dict of arrays needed by backtests. This is the expensive step.
    """
    P(f"    Computing SM (norm={sm_norm}, ema={sm_ema})...")
    sm = compute_smart_money(
        df['Close'].values, df['Volume'].values,
        index_period=sm_index, flow_period=sm_flow,
        norm_period=sm_norm, ema_len=sm_ema,
    )

    P(f"    Resampling to 5min and computing RSI (len={rsi_len})...")
    df_tmp = df.copy()
    df_tmp['SM_Net'] = sm
    df_5m = resample_to_5min(df_tmp)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values,
        df_5m['Close'].values, rsi_len=rsi_len,
    )

    P(f"    Computing daily ATR(14)...")
    atr_array = compute_daily_atr(df)

    return {
        'opens': df['Open'].values,
        'highs': df['High'].values,
        'lows': df['Low'].values,
        'closes': df['Close'].values,
        'sm': sm,
        'times': df.index,
        'rsi_curr': rsi_curr,
        'rsi_prev': rsi_prev,
        'atr': atr_array,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    P("=" * 110)
    P("ATR-SCALED TP/SL EXITS TEST")
    P("Testing whether scaling TP/SL relative to rolling 14-day ATR")
    P("improves performance across different volatility regimes.")
    P("=" * 110)

    # ===================================================================
    # Load all data periods
    # ===================================================================
    raw_data = {}  # {(instrument, period): df}

    P("\n--- Loading Dev Period Data ---")
    raw_data[("MNQ", "DEV")] = load_instrument_1min("MNQ")
    raw_data[("MES", "DEV")] = load_instrument_1min("MES")
    P(f"  MNQ dev: {len(raw_data[('MNQ','DEV')]):,} bars")
    P(f"  MES dev: {len(raw_data[('MES','DEV')]):,} bars")

    P("\n--- Loading Prior Period Data ---")
    raw_data[("MNQ", "PRIOR")] = load_prior_data("MNQ", "prior")
    raw_data[("MES", "PRIOR")] = load_prior_data("MES", "prior")
    P(f"  MNQ prior: {len(raw_data[('MNQ','PRIOR')]):,} bars")
    P(f"  MES prior: {len(raw_data[('MES','PRIOR')]):,} bars")

    P("\n--- Loading Prior2 Period Data ---")
    raw_data[("MNQ", "PRIOR2")] = load_prior_data("MNQ", "prior2")
    raw_data[("MES", "PRIOR2")] = load_prior_data("MES", "prior2")
    P(f"  MNQ prior2: {len(raw_data[('MNQ','PRIOR2')]):,} bars")
    P(f"  MES prior2: {len(raw_data[('MES','PRIOR2')]):,} bars")

    # ===================================================================
    # Pre-compute SM, RSI, ATR once per (instrument, period, sm_params)
    # ===================================================================
    # MNQ strategies (vScalpA, vScalpB) share SM params but differ in RSI len.
    # vScalpA RSI=8, vScalpB RSI=8 (same!). So we can share RSI too.
    # MES v2 uses different SM params and RSI len=12.

    bundles = {}  # {(instrument, period, rsi_len): bundle}

    P("\n--- Pre-computing indicators (this is the slow part) ---")

    for period in ["PRIOR2", "PRIOR", "DEV"]:
        # MNQ with SM(10/12/200/100) and RSI(8)
        P(f"\n  MNQ {period}:")
        df_mnq = raw_data[("MNQ", period)]
        bundles[("MNQ", period, 8)] = prepare_bundle(
            df_mnq, MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
            rsi_len=8,
        )

        # MES with SM(20/12/400/255) and RSI(12)
        P(f"\n  MES {period}:")
        df_mes = raw_data[("MES", period)]
        bundles[("MES", period, 12)] = prepare_bundle(
            df_mes, MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA,
            rsi_len=12,
        )

    # ===================================================================
    # ATR context
    # ===================================================================
    P(f"\n{'='*110}")
    P("DAILY ATR CONTEXT (14-day rolling)")
    P("=" * 110)

    for period in ["PRIOR2", "PRIOR", "DEV"]:
        for inst, rsi_len in [("MNQ", 8), ("MES", 12)]:
            b = bundles[(inst, period, rsi_len)]
            atr = b['atr']
            warmup = min(20000, len(atr) // 4)
            atr_valid = atr[warmup:]
            df = raw_data[(inst, period)]
            P(f"  {inst} {period:>6}: avg ATR = {np.mean(atr_valid):>7.1f}  "
              f"min = {np.min(atr_valid):>7.1f}  max = {np.max(atr_valid):>7.1f}  "
              f"std = {np.std(atr_valid):>7.1f}  "
              f"price = {df['Close'].min():,.0f} - {df['Close'].max():,.0f}")

    # ===================================================================
    # Run backtests (fast part — just loops, no SM recompute)
    # ===================================================================
    P(f"\n{'='*110}")
    P("RUNNING BACKTESTS")
    P("=" * 110)

    # Results storage: {(strat, tp_frac, sl_frac): {period: {"fixed": sc, "atr": sc}}}
    all_results = {}

    # ----- vScalpA -----
    P("\n--- vScalpA (TP=7, SL=40) ---")
    vscalpa_tp_fracs = [0.015, 0.019, 0.023]
    vscalpa_sl_fracs = [0.090, 0.108, 0.130]

    for period in ["PRIOR2", "PRIOR", "DEV"]:
        b = bundles[("MNQ", period, 8)]
        P(f"  {period}...", )

        # Fixed (only once per period)
        trades_fixed = run_backtest_tp_exit(
            b['opens'], b['highs'], b['lows'], b['closes'], b['sm'], b['times'],
            b['rsi_curr'], b['rsi_prev'],
            rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
            sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
            max_loss_pts=VSCALPA_MAX_LOSS_PTS, tp_pts=VSCALPA_TP_PTS,
            entry_end_et=VSCALPA_ENTRY_END_ET,
        )
        sc_fixed = score_trades(trades_fixed, commission_per_side=MNQ_COMMISSION,
                                dollar_per_pt=MNQ_DOLLAR_PER_PT)

        for tp_f in vscalpa_tp_fracs:
            for sl_f in vscalpa_sl_fracs:
                key = ("vScalpA", tp_f, sl_f)
                if key not in all_results:
                    all_results[key] = {}

                trades_atr = run_backtest_atr_scaled(
                    b['opens'], b['highs'], b['lows'], b['closes'], b['sm'], b['times'],
                    b['rsi_curr'], b['rsi_prev'],
                    rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
                    sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
                    tp_frac=tp_f, sl_frac=sl_f,
                    atr_array=b['atr'], tick_size=TICK_SIZE,
                    entry_end_et=VSCALPA_ENTRY_END_ET,
                )
                sc_atr = score_atr_trades(trades_atr, commission_per_side=MNQ_COMMISSION,
                                           dollar_per_pt=MNQ_DOLLAR_PER_PT)

                all_results[key][period] = {"fixed": sc_fixed, "atr": sc_atr}

        P(f"    fixed: {sc_fixed['count'] if sc_fixed else 0} trades, "
          f"9 ATR configs done")

    # ----- vScalpB -----
    P("\n--- vScalpB (TP=3, SL=10) ---")
    vscalpb_tp_fracs = [0.006, 0.008, 0.010]
    vscalpb_sl_fracs = [0.020, 0.027, 0.035]

    for period in ["PRIOR2", "PRIOR", "DEV"]:
        b = bundles[("MNQ", period, 8)]
        P(f"  {period}...")

        trades_fixed = run_backtest_tp_exit(
            b['opens'], b['highs'], b['lows'], b['closes'], b['sm'], b['times'],
            b['rsi_curr'], b['rsi_prev'],
            rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
            sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
            max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
        )
        sc_fixed = score_trades(trades_fixed, commission_per_side=MNQ_COMMISSION,
                                dollar_per_pt=MNQ_DOLLAR_PER_PT)

        for tp_f in vscalpb_tp_fracs:
            for sl_f in vscalpb_sl_fracs:
                key = ("vScalpB", tp_f, sl_f)
                if key not in all_results:
                    all_results[key] = {}

                trades_atr = run_backtest_atr_scaled(
                    b['opens'], b['highs'], b['lows'], b['closes'], b['sm'], b['times'],
                    b['rsi_curr'], b['rsi_prev'],
                    rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
                    sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
                    tp_frac=tp_f, sl_frac=sl_f,
                    atr_array=b['atr'], tick_size=TICK_SIZE,
                )
                sc_atr = score_atr_trades(trades_atr, commission_per_side=MNQ_COMMISSION,
                                           dollar_per_pt=MNQ_DOLLAR_PER_PT)

                all_results[key][period] = {"fixed": sc_fixed, "atr": sc_atr}

        P(f"    fixed: {sc_fixed['count'] if sc_fixed else 0} trades, "
          f"9 ATR configs done")

    # ----- MES v2 -----
    P("\n--- MES v2 (TP=20, SL=35, BE_TIME=75) ---")
    mes_tp_fracs = [0.040, 0.054, 0.068]
    mes_sl_fracs = [0.070, 0.095, 0.120]

    for period in ["PRIOR2", "PRIOR", "DEV"]:
        b = bundles[("MES", period, 12)]
        P(f"  {period}...")

        trades_fixed = run_backtest_tp_exit(
            b['opens'], b['highs'], b['lows'], b['closes'], b['sm'], b['times'],
            b['rsi_curr'], b['rsi_prev'],
            rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
            sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
            max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
            eod_minutes_et=MESV2_EOD_ET,
            entry_end_et=MESV2_ENTRY_END_ET,
            breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        )
        sc_fixed = score_trades(trades_fixed, commission_per_side=MES_COMMISSION,
                                dollar_per_pt=MES_DOLLAR_PER_PT)

        for tp_f in mes_tp_fracs:
            for sl_f in mes_sl_fracs:
                key = ("MES v2", tp_f, sl_f)
                if key not in all_results:
                    all_results[key] = {}

                trades_atr = run_backtest_atr_scaled(
                    b['opens'], b['highs'], b['lows'], b['closes'], b['sm'], b['times'],
                    b['rsi_curr'], b['rsi_prev'],
                    rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
                    sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
                    tp_frac=tp_f, sl_frac=sl_f,
                    atr_array=b['atr'], tick_size=TICK_SIZE,
                    eod_minutes_et=MESV2_EOD_ET,
                    entry_end_et=MESV2_ENTRY_END_ET,
                    breakeven_after_bars=MESV2_BREAKEVEN_BARS,
                )
                sc_atr = score_atr_trades(trades_atr, commission_per_side=MES_COMMISSION,
                                           dollar_per_pt=MES_DOLLAR_PER_PT)

                all_results[key][period] = {"fixed": sc_fixed, "atr": sc_atr}

        P(f"    fixed: {sc_fixed['count'] if sc_fixed else 0} trades, "
          f"9 ATR configs done")

    # ===================================================================
    # Print results
    # ===================================================================
    P(f"\n\n{'='*120}")
    P("RESULTS: FIXED vs ATR-SCALED EXITS")
    P("=" * 120)

    for strat_name in ["vScalpA", "vScalpB", "MES v2"]:
        fixed_tp = {"vScalpA": 7, "vScalpB": 3, "MES v2": 20}[strat_name]
        fixed_sl = {"vScalpA": 40, "vScalpB": 10, "MES v2": 35}[strat_name]

        P(f"\n{'='*120}")
        P(f"  {strat_name}  (fixed: TP={fixed_tp}, SL={fixed_sl})")
        P(f"{'='*120}")

        # Fixed results
        P(f"\n  FIXED EXITS (TP={fixed_tp}, SL={fixed_sl}):")
        first_key = None
        for k in all_results:
            if k[0] == strat_name:
                first_key = k
                break

        if first_key:
            for period in ["PRIOR2", "PRIOR", "DEV"]:
                sc = all_results[first_key][period]["fixed"]
                if sc:
                    P(f"    {period:>6}  {sc['count']:>4} trades  WR {sc['win_rate']:>5.1f}%  "
                      f"PF {sc['pf']:>6.3f}  Net ${sc['net_dollar']:>+9,.2f}  "
                      f"Sharpe {sc['sharpe']:>6.3f}  MaxDD ${sc['max_dd_dollar']:>8.2f}")
                else:
                    P(f"    {period:>6}  NO TRADES")

        # ATR-scaled results
        P(f"\n  ATR-SCALED EXITS:")
        P(f"  {'tp_frac':>8} {'sl_frac':>8} | {'Period':>6} | {'Trades':>6} {'WR%':>6} "
          f"{'PF':>7} {'Net$':>10} {'Sharpe':>7} {'MaxDD$':>9} | "
          f"{'avgTP':>6} {'avgSL':>6} {'avgATR':>7}")
        P(f"  {'-'*8} {'-'*8}-+-{'-'*6}-+-{'-'*6}-{'-'*6}-{'-'*7}-{'-'*10}-"
          f"{'-'*7}-{'-'*9}-+-{'-'*6}-{'-'*6}-{'-'*7}")

        strat_keys = sorted([k for k in all_results if k[0] == strat_name],
                             key=lambda x: (x[1], x[2]))

        for key in strat_keys:
            _, tp_f, sl_f = key
            for period in ["PRIOR2", "PRIOR", "DEV"]:
                sc = all_results[key][period]["atr"]
                if sc is None:
                    P(f"  {tp_f:>8.3f} {sl_f:>8.3f} | {period:>6} | NO TRADES")
                else:
                    P(f"  {tp_f:>8.3f} {sl_f:>8.3f} | {period:>6} | "
                      f"{sc['count']:>6} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                      f"${sc['net_dollar']:>+9,.2f} {sc['sharpe']:>7.3f} "
                      f"${sc['max_dd_dollar']:>8.2f} | "
                      f"{sc['avg_tp_pts']:>5.1f} {sc['avg_sl_pts']:>5.1f} "
                      f"{sc['avg_atr']:>6.0f}")
            P()

    # ===================================================================
    # Summary: best ATR-scaled vs fixed
    # ===================================================================
    P(f"\n\n{'='*120}")
    P("SUMMARY: BEST ATR-SCALED vs FIXED FOR EACH STRATEGY")
    P("(Best = highest cross-period Sharpe sum)")
    P("=" * 120)

    for strat_name in ["vScalpA", "vScalpB", "MES v2"]:
        fixed_tp = {"vScalpA": 7, "vScalpB": 3, "MES v2": 20}[strat_name]
        fixed_sl = {"vScalpA": 40, "vScalpB": 10, "MES v2": 35}[strat_name]

        strat_keys = [k for k in all_results if k[0] == strat_name]

        # Best ATR by Sharpe sum
        best_key = None
        best_sharpe_sum = -999
        for key in strat_keys:
            sharpe_sum = 0
            valid = True
            for period in ["PRIOR2", "PRIOR", "DEV"]:
                sc = all_results[key][period]["atr"]
                if sc is None:
                    valid = False; break
                sharpe_sum += sc["sharpe"]
            if valid and sharpe_sum > best_sharpe_sum:
                best_sharpe_sum = sharpe_sum
                best_key = key

        P(f"\n  {strat_name} (fixed TP={fixed_tp} SL={fixed_sl}):")
        P(f"  {'':15} | {'PRIOR2':>10} {'PRIOR':>10} {'DEV':>10} | {'Sum':>10}")
        P(f"  {'-'*15}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*10}")

        # Fixed row
        first_key = strat_keys[0]
        fixed_sharpes = []
        fixed_pfs = []
        fixed_nets = []
        for period in ["PRIOR2", "PRIOR", "DEV"]:
            sc = all_results[first_key][period]["fixed"]
            fixed_sharpes.append(sc["sharpe"] if sc else 0)
            fixed_pfs.append(sc["pf"] if sc else 0)
            fixed_nets.append(sc["net_dollar"] if sc else 0)

        P(f"  {'FIXED':15} | Sharpe: {fixed_sharpes[0]:>7.3f}   {fixed_sharpes[1]:>7.3f}   "
          f"{fixed_sharpes[2]:>7.3f}  | {sum(fixed_sharpes):>9.3f}")
        P(f"  {'':15} | PF:     {fixed_pfs[0]:>7.3f}   {fixed_pfs[1]:>7.3f}   "
          f"{fixed_pfs[2]:>7.3f}  | {sum(fixed_pfs):>9.3f}")
        P(f"  {'':15} | Net$: {fixed_nets[0]:>+9,.0f} {fixed_nets[1]:>+9,.0f} "
          f"{fixed_nets[2]:>+9,.0f}  | {sum(fixed_nets):>+9,.0f}")

        if best_key:
            _, tp_f, sl_f = best_key
            atr_sharpes = []
            atr_pfs = []
            atr_nets = []
            atr_avg_tps = []
            atr_avg_sls = []
            for period in ["PRIOR2", "PRIOR", "DEV"]:
                sc = all_results[best_key][period]["atr"]
                atr_sharpes.append(sc["sharpe"] if sc else 0)
                atr_pfs.append(sc["pf"] if sc else 0)
                atr_nets.append(sc["net_dollar"] if sc else 0)
                atr_avg_tps.append(sc.get("avg_tp_pts", 0) if sc else 0)
                atr_avg_sls.append(sc.get("avg_sl_pts", 0) if sc else 0)

            P(f"\n  {'ATR-BEST':15} | tp_frac={tp_f:.3f}  sl_frac={sl_f:.3f}")
            P(f"  {'':15} | Sharpe: {atr_sharpes[0]:>7.3f}   {atr_sharpes[1]:>7.3f}   "
              f"{atr_sharpes[2]:>7.3f}  | {sum(atr_sharpes):>9.3f}")
            P(f"  {'':15} | PF:     {atr_pfs[0]:>7.3f}   {atr_pfs[1]:>7.3f}   "
              f"{atr_pfs[2]:>7.3f}  | {sum(atr_pfs):>9.3f}")
            P(f"  {'':15} | Net$: {atr_nets[0]:>+9,.0f} {atr_nets[1]:>+9,.0f} "
              f"{atr_nets[2]:>+9,.0f}  | {sum(atr_nets):>+9,.0f}")
            P(f"  {'':15} | Avg TP:  {atr_avg_tps[0]:>6.1f}    {atr_avg_tps[1]:>6.1f}    "
              f"{atr_avg_tps[2]:>6.1f}")
            P(f"  {'':15} | Avg SL:  {atr_avg_sls[0]:>6.1f}    {atr_avg_sls[1]:>6.1f}    "
              f"{atr_avg_sls[2]:>6.1f}")

            # Delta
            sd = [a - f for a, f in zip(atr_sharpes, fixed_sharpes)]
            pd_ = [a - f for a, f in zip(atr_pfs, fixed_pfs)]
            nd = [a - f for a, f in zip(atr_nets, fixed_nets)]

            P(f"\n  {'DELTA':15} | Sharpe: {sd[0]:>+7.3f}   {sd[1]:>+7.3f}   "
              f"{sd[2]:>+7.3f}  | {sum(sd):>+9.3f}")
            P(f"  {'':15} | PF:     {pd_[0]:>+7.3f}   {pd_[1]:>+7.3f}   "
              f"{pd_[2]:>+7.3f}  | {sum(pd_):>+9.3f}")
            P(f"  {'':15} | Net$: {nd[0]:>+9,.0f} {nd[1]:>+9,.0f} "
              f"{nd[2]:>+9,.0f}  | {sum(nd):>+9,.0f}")

            atr_wins = sum(1 for d in sd if d > 0)
            if sum(sd) > 0.3 and atr_wins >= 2:
                verdict = "ATR-SCALED WINS -- meaningful improvement"
            elif sum(sd) > 0 and atr_wins >= 2:
                verdict = "ATR-SCALED MARGINAL -- small improvement"
            elif sum(sd) < -0.3:
                verdict = "FIXED WINS -- ATR-scaling hurts"
            else:
                verdict = "WASH -- no clear winner"
            P(f"\n  VERDICT: {verdict}")

    # ===================================================================
    # Cross-period consistency
    # ===================================================================
    P(f"\n\n{'='*120}")
    P("CROSS-PERIOD CONSISTENCY: Fixed vs ATR-scaled")
    P("(Sharpe CV across 3 periods -- lower = more consistent)")
    P("=" * 120)

    P(f"\n  {'Strategy':<10} {'Config':<28} | {'P2':>8} {'PR':>8} {'DV':>8} | "
      f"{'Mean':>7} {'StdDev':>7} {'CV':>7} | {'Verdict':>12}")
    P(f"  {'-'*10} {'-'*28}-+-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*12}")

    for strat_name in ["vScalpA", "vScalpB", "MES v2"]:
        strat_keys = [k for k in all_results if k[0] == strat_name]
        if not strat_keys:
            continue

        first_key = strat_keys[0]
        fixed_tp = {"vScalpA": 7, "vScalpB": 3, "MES v2": 20}[strat_name]
        fixed_sl = {"vScalpA": 40, "vScalpB": 10, "MES v2": 35}[strat_name]

        # Fixed
        sharpes = [all_results[first_key][p]["fixed"]["sharpe"]
                   if all_results[first_key][p]["fixed"] else 0
                   for p in ["PRIOR2", "PRIOR", "DEV"]]
        mean_s = np.mean(sharpes)
        std_s = np.std(sharpes)
        cv = std_s / mean_s if mean_s > 0 else 999
        consistent = "YES" if cv < 0.5 and min(sharpes) > 0 else "NO"
        P(f"  {strat_name:<10} {'FIXED TP=%d SL=%d' % (fixed_tp, fixed_sl):<28} | "
          f"{sharpes[0]:>8.3f} {sharpes[1]:>8.3f} {sharpes[2]:>8.3f} | "
          f"{mean_s:>7.3f} {std_s:>7.3f} {cv:>7.3f} | {consistent:>12}")

        # ATR: top 3 by Sharpe sum
        scored = []
        for key in strat_keys:
            ss = []
            valid = True
            for p in ["PRIOR2", "PRIOR", "DEV"]:
                sc = all_results[key][p]["atr"]
                if sc is None:
                    valid = False; break
                ss.append(sc["sharpe"])
            if valid:
                scored.append((key, sum(ss), ss))
        scored.sort(key=lambda x: x[1], reverse=True)

        for key, _, sharpes in scored[:3]:
            _, tp_f, sl_f = key
            mean_s = np.mean(sharpes)
            std_s = np.std(sharpes)
            cv = std_s / mean_s if mean_s > 0 else 999
            consistent = "YES" if cv < 0.5 and min(sharpes) > 0 else "NO"
            P(f"  {'':10} {'ATR tp=%.3f sl=%.3f' % (tp_f, sl_f):<28} | "
              f"{sharpes[0]:>8.3f} {sharpes[1]:>8.3f} {sharpes[2]:>8.3f} | "
              f"{mean_s:>7.3f} {std_s:>7.3f} {cv:>7.3f} | {consistent:>12}")
        P()

    P(f"\n{'='*120}")
    P("DONE -- ATR-Scaled Exits Test Complete")
    P("=" * 120)


if __name__ == "__main__":
    main()
