#!/usr/bin/env python3
"""
MES v2 Comprehensive Gate Sweep — Mar 11, 2026
================================================
Test EVERY available gate type on MES_V2 specifically.
Motivated by: 3 MES full-SL days in the last 5 trading days.

Gates tested:
  1. SM Threshold (0.05, 0.10, 0.15, 0.20, 0.25)
  2. Leledc exhaustion (mq5–9, persistence=1)
  3. ADR directional gate (threshold 0.1–0.5)
  4. Prior-day ATR gate (various percentile thresholds)
  5. Entry hour delay (skip entries before 10:30, 11:00, 11:30)
  6. Entry cutoff (last entry 13:00, 14:00, 15:00)
  7. VIX death zone (various bands)
  8. Overnight range gate (block when overnight range > threshold)
  9. TP1 sweep (TP1=3,5,7,10 with SL=35) — tests "faster partial exit"

Usage:
    cd backtesting_engine/strategies && python3 mes_v2_comprehensive_gate_sweep.py
"""

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# --- Path setup ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR))

from v10_test_common import (
    compute_smart_money,
    compute_et_minutes,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)

sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit, compute_mfe_mae

from sr_common import compute_atr_wilder, assess_pass_fail

_ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# MES v2 config
# ---------------------------------------------------------------------------

MES_STRAT = {
    "name": "MES_V2", "strategy_id": "MES_V2", "instrument": "MES",
    "sm_index": 20, "sm_flow": 12, "sm_norm": 400, "sm_ema": 255,
    "rsi_len": 12, "rsi_buy": 55, "rsi_sell": 45, "sm_threshold": 0.0,
    "cooldown": 25, "max_loss_pts": 35, "tp_pts": 20,
    "eod_et": 15 * 60 + 30, "breakeven_bars": 75,
    "entry_end_et": NY_LAST_ENTRY_ET, "dollar_per_pt": 5.0, "commission": 1.25,
}


# ---------------------------------------------------------------------------
# Data prep (MES only)
# ---------------------------------------------------------------------------

def prepare_mes_data():
    """Load MES, compute SM, split IS/OOS."""
    df = load_instrument_1min("MES")
    sm = compute_smart_money(
        df["Close"].values, df["Volume"].values,
        index_period=20, flow_period=12,
        norm_period=400, ema_len=255,
    )
    df["SM_Net"] = sm
    print(f"Loaded MES: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    mid = df.index[len(df) // 2]
    is_len = (df.index < mid).sum()
    is_range = f"{df.index[0].strftime('%Y-%m-%d')}_to_{df.index[is_len-1].strftime('%Y-%m-%d')}"
    oos_range = f"{df.index[is_len].strftime('%Y-%m-%d')}_to_{df.index[-1].strftime('%Y-%m-%d')}"
    print(f"  Split: IS {is_len} bars ({is_range}), OOS {len(df)-is_len} bars ({oos_range})")

    # RSI per split
    splits = {}
    for split_name, df_split in [("FULL", df), ("IS", df[df.index < mid]), ("OOS", df[df.index >= mid])]:
        df_5m = resample_to_5min(df_split)
        rsi_curr, rsi_prev = map_5min_rsi_to_1min(
            df_split.index.values, df_5m.index.values,
            df_5m["Close"].values, rsi_len=12,
        )
        splits[split_name] = (
            df_split["Open"].values,
            df_split["High"].values,
            df_split["Low"].values,
            df_split["Close"].values,
            df_split["SM_Net"].values,
            df_split.index,
            rsi_curr,
            rsi_prev,
        )

    return df, splits, is_len


def run_mes(splits, split_name, entry_gate=None, override_params=None):
    """Run MES v2 backtest on a split with optional gate and param overrides."""
    opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev = splits[split_name]
    strat = dict(MES_STRAT)
    if override_params:
        strat.update(override_params)

    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=strat["rsi_buy"], rsi_sell=strat["rsi_sell"],
        sm_threshold=strat["sm_threshold"], cooldown_bars=strat["cooldown"],
        max_loss_pts=strat["max_loss_pts"], tp_pts=strat["tp_pts"],
        eod_minutes_et=strat["eod_et"],
        breakeven_after_bars=strat["breakeven_bars"],
        entry_end_et=strat["entry_end_et"],
        entry_gate=entry_gate,
    )
    compute_mfe_mae(trades, highs, lows)
    sc = score_trades(trades, commission_per_side=strat["commission"],
                      dollar_per_pt=strat["dollar_per_pt"])
    return sc, trades


def slice_gate(full_gate, is_len, split_name):
    """Slice full-length gate to IS/OOS/FULL."""
    if full_gate is None:
        return None
    if split_name == "FULL":
        return full_gate
    elif split_name == "IS":
        return full_gate[:is_len]
    else:
        return full_gate[is_len:]


def print_result(label, sc_is, sc_oos, bl_is, bl_oos):
    """Print one result row."""
    if sc_is is None or sc_oos is None:
        print(f"  {label:>35}  NO TRADES")
        return

    dpf_is = (sc_is["pf"] - bl_is["pf"]) / bl_is["pf"] * 100 if bl_is["pf"] > 0 else 0
    dpf_oos = (sc_oos["pf"] - bl_oos["pf"]) / bl_oos["pf"] * 100 if bl_oos["pf"] > 0 else 0
    verdict, detail = assess_pass_fail(sc_is, sc_oos, bl_is, bl_oos)

    blk_is = bl_is["count"] - sc_is["count"]
    blk_oos = bl_oos["count"] - sc_oos["count"]

    print(f"  {label:>35}  IS: {sc_is['count']:>4}t {sc_is['win_rate']:>5.1f}% PF {sc_is['pf']:>5.3f} ${sc_is['net_dollar']:>+8.0f} Sh {sc_is['sharpe']:>5.2f}  "
          f"OOS: {sc_oos['count']:>4}t {sc_oos['win_rate']:>5.1f}% PF {sc_oos['pf']:>5.3f} ${sc_oos['net_dollar']:>+8.0f} Sh {sc_oos['sharpe']:>5.2f}  "
          f"Blk {blk_is:>3}/{blk_oos:>3}  dPF {dpf_is:>+5.1f}/{dpf_oos:>+5.1f}%  {verdict}")


# ---------------------------------------------------------------------------
# Gate builders
# ---------------------------------------------------------------------------

def build_leledc_gate(closes, maj_qual, lookback=4, persistence=1):
    """Leledc exhaustion gate (same as MNQ implementation)."""
    n = len(closes)
    bull_count = 0
    bear_count = 0
    gate = np.ones(n, dtype=bool)  # True = allow

    bull_exhaust = np.zeros(n, dtype=bool)
    bear_exhaust = np.zeros(n, dtype=bool)

    for i in range(n):
        if i < lookback:
            continue
        if closes[i] > closes[i - lookback]:
            bull_count += 1
        else:
            bull_count = 0
        if bull_count >= maj_qual:
            bull_exhaust[i] = True
        if closes[i] < closes[i - lookback]:
            bear_count += 1
        else:
            bear_count = 0
        if bear_count >= maj_qual:
            bear_exhaust[i] = True

    # Apply persistence: block for `persistence` bars after detection
    blocked = np.zeros(n, dtype=bool)
    for i in range(n):
        if bull_exhaust[i] or bear_exhaust[i]:
            for j in range(i, min(i + persistence, n)):
                blocked[j] = True

    # Gate uses prev bar (bar[i-1]) convention
    gate_prev = np.ones(n, dtype=bool)
    gate_prev[1:] = ~blocked[:-1]
    gate_prev[0] = True
    return gate_prev


def build_adr_gate(df, threshold, lookback=14):
    """ADR directional gate — blocks entries chasing the daily move.

    Returns (gate_long, gate_short) — True = allow.
    Since run_backtest_tp_exit doesn't know side at gate time,
    we return a combined gate: blocked if EITHER direction is blocked.

    Actually, for a proper test we need directional gating.
    Since the backtest decides side internally, we'll return a gate array
    and override — block any entry when |move_from_open/ADR| >= threshold.
    This is slightly different from the live implementation (which is directional)
    but captures the same concept: don't trade when intraday move is extended.
    """
    n = len(df)
    times_et = df.index.tz_localize("UTC").tz_convert(_ET) if df.index.tz is None else df.index.tz_convert(_ET)
    et_dates = times_et.date
    et_mins = (times_et.hour * 60 + times_et.minute).values.astype(np.int32)
    closes = df["Close"].values
    opens_col = df["Open"].values

    # RTH bounds
    rth_start = 600  # 10:00 ET
    rth_end = 960    # 16:00 ET

    # Step 1: compute daily RTH ranges for ADR
    rth_mask = (et_mins >= rth_start) & (et_mins < rth_end)
    daily_ranges = {}
    for date in sorted(set(et_dates)):
        day_mask = (et_dates == date) & rth_mask
        if not day_mask.any():
            continue
        day_highs = df["High"].values[day_mask]
        day_lows = df["Low"].values[day_mask]
        daily_ranges[date] = float(day_highs.max() - day_lows.min())

    dates_sorted = sorted(daily_ranges.keys())

    # Step 2: rolling ADR for each date
    adr_by_date = {}
    for i, d in enumerate(dates_sorted):
        start = max(0, i - lookback)
        window = [daily_ranges[dates_sorted[j]] for j in range(start, i)]
        if len(window) >= 1:
            adr_by_date[d] = sum(window) / len(window)

    # Step 3: compute per-bar gate
    gate = np.ones(n, dtype=bool)  # True = allow
    session_open = {}  # date -> first RTH bar open

    for i in range(n):
        date = et_dates[i]
        mins = et_mins[i]

        if mins < rth_start or mins >= rth_end:
            continue

        if date not in session_open:
            session_open[date] = opens_col[i]

        adr = adr_by_date.get(date)
        if adr is None or adr <= 0:
            continue

        move = closes[i] - session_open[date]
        ratio = move / adr

        if abs(ratio) >= threshold:
            gate[i] = False

    # Prev-bar convention
    gate_prev = np.ones(n, dtype=bool)
    gate_prev[1:] = gate[:-1]
    return gate_prev


def build_atr_gate(df, min_atr, period=14):
    """Prior-day ATR gate — block when ATR(14) < threshold."""
    n = len(df)
    times_et = df.index.tz_localize("UTC").tz_convert(_ET) if df.index.tz is None else df.index.tz_convert(_ET)
    et_dates = times_et.date
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    # Daily ranges (all bars, not just RTH)
    daily_data = {}
    for i in range(n):
        date = et_dates[i]
        if date not in daily_data:
            daily_data[date] = {"high": highs[i], "low": lows[i]}
        else:
            daily_data[date]["high"] = max(daily_data[date]["high"], highs[i])
            daily_data[date]["low"] = min(daily_data[date]["low"], lows[i])

    dates_sorted = sorted(daily_data.keys())
    daily_ranges = [daily_data[d]["high"] - daily_data[d]["low"] for d in dates_sorted]

    # Wilder ATR per completed day
    atr_by_date = {}
    if len(daily_ranges) >= period:
        atr = sum(daily_ranges[:period]) / period
        atr_by_date[dates_sorted[period - 1]] = atr
        for j in range(period, len(daily_ranges)):
            atr = (atr * (period - 1) + daily_ranges[j]) / period
            atr_by_date[dates_sorted[j]] = atr

    # Gate: block if prior completed day's ATR < min_atr
    gate = np.ones(n, dtype=bool)
    prev_atr = None
    prev_date = None
    for i in range(n):
        date = et_dates[i]
        if date != prev_date:
            # New day — get prior day's ATR
            if prev_date is not None and prev_date in atr_by_date:
                prev_atr = atr_by_date[prev_date]
            prev_date = date

        if prev_atr is not None and prev_atr < min_atr:
            gate[i] = False

    # Prev-bar convention
    gate_prev = np.ones(n, dtype=bool)
    gate_prev[1:] = gate[:-1]
    return gate_prev


def build_entry_delay_gate(df, delay_minutes_after_open):
    """Block entries in the first N minutes after RTH open (10:00 ET)."""
    n = len(df)
    times_et = df.index.tz_localize("UTC").tz_convert(_ET) if df.index.tz is None else df.index.tz_convert(_ET)
    et_mins = (times_et.hour * 60 + times_et.minute).values.astype(np.int32)

    cutoff = 600 + delay_minutes_after_open  # 10:00 + delay
    gate = np.ones(n, dtype=bool)
    for i in range(n):
        if 600 <= et_mins[i] < cutoff:
            gate[i] = False

    gate_prev = np.ones(n, dtype=bool)
    gate_prev[1:] = gate[:-1]
    return gate_prev


def build_entry_cutoff_gate(df, cutoff_hour_et):
    """Block entries after cutoff_hour_et (e.g. 13 = 13:00 ET)."""
    n = len(df)
    times_et = df.index.tz_localize("UTC").tz_convert(_ET) if df.index.tz is None else df.index.tz_convert(_ET)
    et_mins = (times_et.hour * 60 + times_et.minute).values.astype(np.int32)

    cutoff = cutoff_hour_et * 60
    gate = np.ones(n, dtype=bool)
    for i in range(n):
        if et_mins[i] >= cutoff:
            gate[i] = False

    gate_prev = np.ones(n, dtype=bool)
    gate_prev[1:] = gate[:-1]
    return gate_prev


def build_overnight_range_gate(df, max_range_pts):
    """Block all entries for the day if overnight/premarket range exceeds threshold.

    Overnight = bars between prior day 16:00 ET and current day 10:00 ET.
    """
    n = len(df)
    times_et = df.index.tz_localize("UTC").tz_convert(_ET) if df.index.tz is None else df.index.tz_convert(_ET)
    et_dates = times_et.date
    et_mins = (times_et.hour * 60 + times_et.minute).values.astype(np.int32)
    highs = df["High"].values
    lows = df["Low"].values

    # Compute overnight range for each trading day
    # Overnight = all bars where et_mins < 600 (before 10:00 ET)
    overnight_ranges = {}
    for i in range(n):
        date = et_dates[i]
        mins = et_mins[i]
        if mins < 600:  # Before RTH
            if date not in overnight_ranges:
                overnight_ranges[date] = {"high": highs[i], "low": lows[i]}
            else:
                overnight_ranges[date]["high"] = max(overnight_ranges[date]["high"], highs[i])
                overnight_ranges[date]["low"] = min(overnight_ranges[date]["low"], lows[i])

    on_range = {d: v["high"] - v["low"] for d, v in overnight_ranges.items()}

    # Gate: block all RTH bars on days with excessive overnight range
    gate = np.ones(n, dtype=bool)
    for i in range(n):
        date = et_dates[i]
        if date in on_range and on_range[date] > max_range_pts:
            gate[i] = False

    gate_prev = np.ones(n, dtype=bool)
    gate_prev[1:] = gate[:-1]
    return gate_prev


def build_vix_gate(df, vix_min, vix_max):
    """VIX death zone gate — block when prior-day VIX close is in [vix_min, vix_max].

    Downloads VIX data from yfinance.
    """
    import yfinance as yf

    times_et = df.index.tz_localize("UTC").tz_convert(_ET) if df.index.tz is None else df.index.tz_convert(_ET)
    start_date = times_et[0].strftime("%Y-%m-%d")
    end_date = (times_et[-1] + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"    Downloading VIX data ({start_date} to {end_date})...")
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if vix.empty:
        print("    WARNING: No VIX data — gate disabled")
        return np.ones(len(df), dtype=bool)

    # Handle MultiIndex columns
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix["date"] = vix.index.date
    vix_closes = dict(zip(vix["date"], vix["Close"].values))

    # Build gate: for each trading day, look up prior trading day's VIX close
    n = len(df)
    et_dates = times_et.date
    gate = np.ones(n, dtype=bool)

    sorted_vix_dates = sorted(vix_closes.keys())

    for i in range(n):
        date = et_dates[i]
        # Find prior VIX close date
        prior_vix = None
        for vd in reversed(sorted_vix_dates):
            if vd < date:
                prior_vix = vix_closes[vd]
                break
        if prior_vix is not None and vix_min <= prior_vix <= vix_max:
            gate[i] = False

    gate_prev = np.ones(n, dtype=bool)
    gate_prev[1:] = gate[:-1]
    return gate_prev


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("MES V2 COMPREHENSIVE GATE SWEEP")
    print("=" * 120)

    df, splits, is_len = prepare_mes_data()

    # ===== BASELINE =====
    print("\n--- BASELINE ---")
    bl_is, _ = run_mes(splits, "IS")
    bl_oos, _ = run_mes(splits, "OOS")
    bl_full, _ = run_mes(splits, "FULL")
    print(f"  BASELINE  IS: {bl_is['count']}t WR {bl_is['win_rate']:.1f}% PF {bl_is['pf']:.3f} ${bl_is['net_dollar']:+.0f} Sh {bl_is['sharpe']:.2f}  "
          f"OOS: {bl_oos['count']}t WR {bl_oos['win_rate']:.1f}% PF {bl_oos['pf']:.3f} ${bl_oos['net_dollar']:+.0f} Sh {bl_oos['sharpe']:.2f}")

    # ===== 1. SM THRESHOLD =====
    print("\n" + "=" * 120)
    print("1. SM THRESHOLD SWEEP")
    print("=" * 120)
    for smt in [0.05, 0.10, 0.15, 0.20, 0.25]:
        sc_is, _ = run_mes(splits, "IS", override_params={"sm_threshold": smt})
        sc_oos, _ = run_mes(splits, "OOS", override_params={"sm_threshold": smt})
        print_result(f"SM_T={smt:.2f}", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 2. LELEDC EXHAUSTION =====
    print("\n" + "=" * 120)
    print("2. LELEDC EXHAUSTION SWEEP")
    print("=" * 120)
    closes = df["Close"].values
    for mq in [5, 6, 7, 8, 9]:
        gate = build_leledc_gate(closes, maj_qual=mq, persistence=1)
        g_is = slice_gate(gate, is_len, "IS")
        g_oos = slice_gate(gate, is_len, "OOS")
        sc_is, _ = run_mes(splits, "IS", entry_gate=g_is)
        sc_oos, _ = run_mes(splits, "OOS", entry_gate=g_oos)
        print_result(f"Leledc mq={mq} p=1", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 3. ADR DIRECTIONAL GATE =====
    print("\n" + "=" * 120)
    print("3. ADR DIRECTIONAL GATE SWEEP")
    print("=" * 120)
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        gate = build_adr_gate(df, threshold=thresh)
        g_is = slice_gate(gate, is_len, "IS")
        g_oos = slice_gate(gate, is_len, "OOS")
        sc_is, _ = run_mes(splits, "IS", entry_gate=g_is)
        sc_oos, _ = run_mes(splits, "OOS", entry_gate=g_oos)
        print_result(f"ADR thresh={thresh:.2f}", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 4. PRIOR-DAY ATR GATE =====
    print("\n" + "=" * 120)
    print("4. PRIOR-DAY ATR GATE SWEEP")
    print("=" * 120)
    for min_atr in [30, 40, 50, 60, 70, 80]:
        gate = build_atr_gate(df, min_atr=min_atr)
        g_is = slice_gate(gate, is_len, "IS")
        g_oos = slice_gate(gate, is_len, "OOS")
        sc_is, _ = run_mes(splits, "IS", entry_gate=g_is)
        sc_oos, _ = run_mes(splits, "OOS", entry_gate=g_oos)
        print_result(f"ATR min={min_atr}", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 5. ENTRY DELAY (skip first N min) =====
    print("\n" + "=" * 120)
    print("5. ENTRY DELAY SWEEP (skip first N min after 10:00 ET)")
    print("=" * 120)
    for delay in [15, 30, 45, 60, 90]:
        gate = build_entry_delay_gate(df, delay)
        g_is = slice_gate(gate, is_len, "IS")
        g_oos = slice_gate(gate, is_len, "OOS")
        sc_is, _ = run_mes(splits, "IS", entry_gate=g_is)
        sc_oos, _ = run_mes(splits, "OOS", entry_gate=g_oos)
        print_result(f"Delay +{delay}min", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 6. ENTRY CUTOFF =====
    print("\n" + "=" * 120)
    print("6. ENTRY CUTOFF SWEEP (last entry hour ET)")
    print("=" * 120)
    for cutoff in [12, 13, 14]:
        gate = build_entry_cutoff_gate(df, cutoff)
        g_is = slice_gate(gate, is_len, "IS")
        g_oos = slice_gate(gate, is_len, "OOS")
        sc_is, _ = run_mes(splits, "IS", entry_gate=g_is)
        sc_oos, _ = run_mes(splits, "OOS", entry_gate=g_oos)
        print_result(f"Cutoff {cutoff}:00 ET", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 7. OVERNIGHT RANGE GATE =====
    print("\n" + "=" * 120)
    print("7. OVERNIGHT RANGE GATE SWEEP (block if overnight range > N pts)")
    print("=" * 120)
    for max_range in [30, 40, 50, 60, 70, 80, 100]:
        gate = build_overnight_range_gate(df, max_range)
        g_is = slice_gate(gate, is_len, "IS")
        g_oos = slice_gate(gate, is_len, "OOS")
        sc_is, _ = run_mes(splits, "IS", entry_gate=g_is)
        sc_oos, _ = run_mes(splits, "OOS", entry_gate=g_oos)

        # Count blocked days
        times_et = df.index.tz_localize("UTC").tz_convert(_ET) if df.index.tz is None else df.index.tz_convert(_ET)
        et_dates = times_et.date
        blocked_dates = set()
        for i in range(len(gate)):
            if not gate[i]:
                blocked_dates.add(et_dates[i])
        print_result(f"ON range>{max_range}pts ({len(blocked_dates)}d blocked)", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 8. VIX DEATH ZONE =====
    print("\n" + "=" * 120)
    print("8. VIX DEATH ZONE SWEEP")
    print("=" * 120)
    # Cache VIX data — build once
    vix_gate_cache = {}
    for vmin, vmax in [(15, 20), (17, 22), (18, 22), (19, 22), (19, 24), (20, 25), (22, 28)]:
        gate = build_vix_gate(df, vmin, vmax)
        g_is = slice_gate(gate, is_len, "IS")
        g_oos = slice_gate(gate, is_len, "OOS")
        sc_is, _ = run_mes(splits, "IS", entry_gate=g_is)
        sc_oos, _ = run_mes(splits, "OOS", entry_gate=g_oos)
        print_result(f"VIX [{vmin}-{vmax}]", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 9. TP SWEEP (faster partial) =====
    print("\n" + "=" * 120)
    print("9. TP SWEEP (would this be better as a scalp?)")
    print("=" * 120)
    for tp in [3, 5, 7, 10, 15, 20, 25]:
        sc_is, _ = run_mes(splits, "IS", override_params={"tp_pts": tp})
        sc_oos, _ = run_mes(splits, "OOS", override_params={"tp_pts": tp})
        print_result(f"TP={tp}", sc_is, sc_oos, bl_is, bl_oos)

    # ===== 10. SL SWEEP =====
    print("\n" + "=" * 120)
    print("10. SL SWEEP (tighter stop?)")
    print("=" * 120)
    for sl in [15, 20, 25, 30, 35, 40, 50]:
        sc_is, _ = run_mes(splits, "IS", override_params={"max_loss_pts": sl})
        sc_oos, _ = run_mes(splits, "OOS", override_params={"max_loss_pts": sl})
        print_result(f"SL={sl}", sc_is, sc_oos, bl_is, bl_oos)

    # ===== SUMMARY =====
    print("\n" + "=" * 120)
    print("SWEEP COMPLETE")
    print("=" * 120)
    print("Look for STRONG PASS or MARGINAL PASS results above.")
    print("Any gate that passes on MES can be implemented in SafetyManager.")


if __name__ == "__main__":
    main()
