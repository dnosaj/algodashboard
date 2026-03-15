"""
VCR Regime Gate + dVPOC Distance Gate — IS/OOS Pre-Filter Sweep
================================================================
Two developing-VPOC-based entry gates tested as pre-filters (blocking
entries DURING the backtest so cooldowns/episodes play out naturally).

Gate 1 — VCR Regime Gate (all strategies):
  Block entries when developing VCR at bar[i-1] > threshold.
  Thresholds: [0.10, 0.15, 0.20, 0.25, 0.30]
  Minimum RTH bars variants: [0, 30, 60]
  → 15 configurations per strategy.

Gate 2 — dVPOC Distance Gate (MES v2 only):
  Block MES entries when abs(close[i-1] - dVPOC[i-1]) < threshold_pts.
  Thresholds: [5, 7, 10] pts.

IS/OOS: 50/50 split at midpoint (same as sr_common.py).

Usage:
    cd backtesting_engine/strategies
    python3 vcr_dvpoc_gate_sweep.py
"""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ET_TZ = ZoneInfo("America/New_York")

# --- Path setup ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR))
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    compute_et_minutes,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    NY_OPEN_ET,
    NY_CLOSE_ET,
)

from generate_session import (
    run_backtest_tp_exit,
    compute_mfe_mae,
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

# Gate functions (existing production gates)
from sr_leledc_exhaustion_sweep import compute_leledc_exhaustion, build_leledc_gate
from adr_common import compute_session_tracking, compute_adr, build_directional_gate
from htf_common import compute_prior_day_atr
from sr_prior_day_levels_sweep import (
    compute_rth_volume_profile, build_prior_day_level_gate, _compute_value_area,
)
from v10_test_common import compute_prior_day_levels

# Structure exit for vScalpC
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

# Developing VPOC computation
from developing_vpoc_forensics import compute_developing_vpoc, BIN_WIDTHS


# ============================================================================
# Gate construction
# ============================================================================

def build_vcr_gate(vcr_arr, et_mins, threshold, min_rth_bars):
    """Build boolean gate: True = allow entry, False = block.

    Blocks when VCR at bar[i-1] exceeds threshold.
    The gate array is indexed by bar i (the entry bar); it checks bar[i-1].
    So gate[i] = True means "entry at bar i is allowed".

    However, the backtest engine already does bar[i-1] lookup:
        gate_ok = entry_gate is None or entry_gate[i - 1]
    So we build the gate such that gate[j] = True means "bar j's VCR is OK
    for an entry on bar j+1". The engine reads gate[i-1] to decide bar i entry.

    min_rth_bars: Only apply VCR gating after this many RTH bars have
    accumulated in the current day (0 = apply from first RTH bar).
    """
    n = len(vcr_arr)
    gate = np.ones(n, dtype=bool)

    # Convert timestamps to ET dates for RTH bar counting
    # We count RTH bars per day to handle min_rth_bars
    if min_rth_bars > 0:
        # Need to track bars-in-RTH per day
        # et_mins[i] gives ET minutes for bar i
        # RTH = NY_OPEN_ET <= et_mins[i] < NY_CLOSE_ET
        # We need dates too — but we can track day boundaries via et_mins
        # Since we only care about RTH bars and VCR is NaN outside RTH,
        # we can use a simpler approach: count consecutive RTH bars per day

        # Detect day boundaries: when et_mins[i] < et_mins[i-1], it's a new day
        rth_bar_count = 0
        prev_in_rth = False
        rth_counts = np.zeros(n, dtype=int)

        for i in range(n):
            in_rth = NY_OPEN_ET <= et_mins[i] < NY_CLOSE_ET

            # Detect day boundary: either we went from RTH to non-RTH and back,
            # or et_mins drops (overnight -> next morning)
            if in_rth:
                if not prev_in_rth:
                    # Transitioning into RTH — could be new day or resuming
                    # Check if there was a gap (non-RTH period)
                    rth_bar_count = 1
                else:
                    rth_bar_count += 1
                rth_counts[i] = rth_bar_count
            else:
                rth_counts[i] = 0

            prev_in_rth = in_rth

    for i in range(n):
        vcr_val = vcr_arr[i]
        if np.isnan(vcr_val):
            # NaN VCR = outside RTH or before first bar. Allow (fail-open).
            continue

        # min_rth_bars check: don't gate early-session bars
        if min_rth_bars > 0 and rth_counts[i] < min_rth_bars:
            continue  # Don't apply gate yet (allow entry)

        if vcr_val > threshold:
            gate[i] = False

    return gate


def build_dvpoc_distance_gate(closes, dvpoc_arr, min_distance_pts):
    """Build boolean gate: True = allow entry, False = block.

    Blocks when abs(close - dVPOC) < min_distance_pts at this bar.
    The engine reads gate[i-1] to decide bar i entry.
    So gate[j] represents the state at bar j.

    NaN dVPOC = fail-open (allow entry).
    """
    n = len(closes)
    gate = np.ones(n, dtype=bool)

    for i in range(n):
        dvpoc_val = dvpoc_arr[i]
        if np.isnan(dvpoc_val):
            continue
        if abs(closes[i] - dvpoc_val) < min_distance_pts:
            gate[i] = False

    return gate


# ============================================================================
# Scoring helpers
# ============================================================================

def score_vscalpc_as_flat(trades, dollar_per_pt, commission_per_side):
    """Score vScalpC structure trades into the same format as score_trades().

    Groups legs by entry_idx, computes per-entry P&L, returns a dict
    compatible with score_trades() output (count, pf, win_rate, net_dollar,
    max_dd_dollar, sharpe).
    """
    if not trades:
        return None

    comm_per_leg = commission_per_side * 2  # entry + exit

    entries = {}
    for t in trades:
        eidx = t["entry_idx"]
        if eidx not in entries:
            entries[eidx] = []
        entries[eidx].append(t)

    pnl_list = []
    for eidx in sorted(entries.keys()):
        legs = entries[eidx]
        total_pnl = sum(leg["pts"] * dollar_per_pt - comm_per_leg for leg in legs)
        pnl_list.append(total_pnl)

    if not pnl_list:
        return None

    pnl_arr = np.array(pnl_list)
    n = len(pnl_arr)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100

    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()

    sharpe = 0.0
    if n > 1 and np.std(pnl_arr) > 0:
        sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)

    return {
        "count": n,
        "pf": round(pf, 3),
        "win_rate": round(wr, 1),
        "net_dollar": round(pnl_arr.sum(), 2),
        "max_dd_dollar": round(mdd, 2),
        "sharpe": round(sharpe, 3),
    }


# ============================================================================
# Gate slicing
# ============================================================================

def _slice(arr, split_name, is_len):
    """Slice an array for IS/OOS. Returns full array if split_name='FULL'."""
    if arr is None:
        return None
    if split_name == "FULL":
        return arr
    elif split_name == "IS":
        return arr[:is_len]
    else:
        return arr[is_len:]


# ============================================================================
# Data loading + gate computation (once for full dataset)
# ============================================================================

def load_all_data():
    """Load instruments, compute SM, production gates, developing VPOC.

    Returns a dict with everything needed for the sweep.
    """
    print("=" * 70)
    print("VCR + dVPOC DISTANCE GATE SWEEP — Loading data")
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

    start_date = df_mnq.index[0].strftime("%Y-%m-%d")
    end_date = df_mnq.index[-1].strftime("%Y-%m-%d")

    # --- IS/OOS split indices ---
    mnq_is_len = len(df_mnq) // 2
    mes_is_len = len(df_mes) // 2

    mnq_mid = df_mnq.index[mnq_is_len]
    mes_mid = df_mes.index[mes_is_len]

    print(f"\n  MNQ split: IS {mnq_is_len} bars, OOS {len(df_mnq) - mnq_is_len} bars")
    print(f"  MES split: IS {mes_is_len} bars, OOS {len(df_mes) - mes_is_len} bars")

    # --- Production gates (full data) ---
    print("\n--- Computing Production Entry Gates ---")

    mnq_closes = df_mnq["Close"].values
    mnq_highs = df_mnq["High"].values
    mnq_lows = df_mnq["Low"].values
    mnq_sm_arr = df_mnq["SM_Net"].values

    # Leledc
    print("  Leledc exhaustion (mq=9)...")
    bull_ex, bear_ex = compute_leledc_exhaustion(mnq_closes, maj_qual=9)
    mnq_leledc_gate = build_leledc_gate(bull_ex, bear_ex, persistence=1)

    # ADR directional
    print("  ADR directional gate (14d, 0.3)...")
    mnq_session = compute_session_tracking(df_mnq)
    mnq_adr = compute_adr(df_mnq, lookback_days=14)
    mnq_adr_gate = build_directional_gate(
        mnq_session['move_from_open'], mnq_adr, mnq_sm_arr, threshold=0.3)

    # ATR gate (vScalpC only)
    print("  Prior-day ATR gate (min=263.8)...")
    mnq_prior_atr = compute_prior_day_atr(df_mnq, lookback_days=14)
    mnq_atr_gate = np.ones(len(df_mnq), dtype=bool)
    for i in range(len(mnq_prior_atr)):
        if not np.isnan(mnq_prior_atr[i]) and mnq_prior_atr[i] < 263.8:
            mnq_atr_gate[i] = False

    # VIX death zone
    print("  VIX death zone gate (19-22)...")
    mnq_bar_dates_et = _get_bar_dates_et(df_mnq.index)
    mnq_vix_gate = load_vix_gate(start_date, end_date, mnq_bar_dates_et,
                                  low=19.0, high=22.0)

    # MES gates
    print("  MES prior-day level gate (VPOC+VAL, buf=5)...")
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

    # Composite production gates
    prod_gate_vscalpa = mnq_leledc_gate & mnq_adr_gate & mnq_vix_gate
    prod_gate_vscalpb = mnq_leledc_gate & mnq_adr_gate
    prod_gate_vscalpc = mnq_leledc_gate & mnq_adr_gate & mnq_atr_gate & mnq_vix_gate
    prod_gate_mesv2 = mes_level_gate

    # --- Swing levels for vScalpC ---
    print("  Swing levels for vScalpC...")
    mnq_swing_highs, mnq_swing_lows = compute_swing_levels(
        mnq_highs, mnq_lows,
        lookback=VSCALPC_SWING_LB, swing_type="pivot",
        pivot_right=VSCALPC_SWING_PR)

    # --- Compute developing VPOC/VCR for both instruments ---
    print("\n--- Computing Developing VPOC/VCR ---")

    mnq_times = df_mnq.index
    mnq_volumes = df_mnq["Volume"].values
    mnq_et_mins = compute_et_minutes(mnq_times)

    mnq_dvpoc, mnq_vcr, mnq_stab, mnq_vwap, mnq_prior_v = compute_developing_vpoc(
        mnq_times, mnq_closes, mnq_volumes, mnq_et_mins, BIN_WIDTHS["MNQ"],
        highs=mnq_highs, lows=mnq_lows)
    n_valid = np.sum(~np.isnan(mnq_vcr))
    print(f"  MNQ: {n_valid} bars with valid VCR")

    mes_dvpoc, mes_vcr, mes_stab, mes_vwap, mes_prior_v = compute_developing_vpoc(
        mes_times, mes_closes, mes_volumes, mes_et_mins, BIN_WIDTHS["MES"],
        highs=mes_highs, lows=mes_lows)
    n_valid = np.sum(~np.isnan(mes_vcr))
    print(f"  MES: {n_valid} bars with valid VCR")

    # --- RSI pre-computation ---
    print("\n  Pre-computing RSI...")
    df_mnq_5m = resample_to_5min(df_mnq)
    rsi_mnq_curr_full, rsi_mnq_prev_full = map_5min_rsi_to_1min(
        df_mnq.index.values, df_mnq_5m.index.values,
        df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN)

    df_mes_5m = resample_to_5min(df_mes)
    rsi_mes_curr_full, rsi_mes_prev_full = map_5min_rsi_to_1min(
        df_mes.index.values, df_mes_5m.index.values,
        df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN)

    return {
        "df_mnq": df_mnq,
        "df_mes": df_mes,
        "mnq_is_len": mnq_is_len,
        "mes_is_len": mes_is_len,
        # Production gates (full)
        "prod_gate_vscalpa": prod_gate_vscalpa,
        "prod_gate_vscalpb": prod_gate_vscalpb,
        "prod_gate_vscalpc": prod_gate_vscalpc,
        "prod_gate_mesv2": prod_gate_mesv2,
        # Swing levels (full)
        "mnq_swing_highs": mnq_swing_highs,
        "mnq_swing_lows": mnq_swing_lows,
        # Developing VPOC arrays (full)
        "mnq_dvpoc": mnq_dvpoc,
        "mnq_vcr": mnq_vcr,
        "mnq_et_mins": mnq_et_mins,
        "mes_dvpoc": mes_dvpoc,
        "mes_vcr": mes_vcr,
        "mes_et_mins": mes_et_mins,
        "mes_closes": mes_closes,
        # RSI (full)
        "rsi_mnq_curr_full": rsi_mnq_curr_full,
        "rsi_mnq_prev_full": rsi_mnq_prev_full,
        "rsi_mes_curr_full": rsi_mes_curr_full,
        "rsi_mes_prev_full": rsi_mes_prev_full,
    }


# ============================================================================
# Strategy runners — one backtest per (strategy, split, gate_config)
# ============================================================================

def run_vscalpa(data, split_name, extra_gate=None):
    """Run vScalpA on the given split with production gates + optional extra gate."""
    df = data["df_mnq"]
    is_len = data["mnq_is_len"]

    # Slice data
    if split_name == "IS":
        df_s = df.iloc[:is_len]
    elif split_name == "OOS":
        df_s = df.iloc[is_len:]
    else:
        df_s = df

    # RSI must be recomputed per split for correct warmup
    df_5m = resample_to_5min(df_s)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df_s.index.values, df_5m.index.values,
        df_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN)

    # Composite gate: production AND extra
    prod = _slice(data["prod_gate_vscalpa"], split_name, is_len)
    if extra_gate is not None:
        eg = _slice(extra_gate, split_name, is_len)
        gate = prod & eg
    else:
        gate = prod

    trades = run_backtest_tp_exit(
        df_s["Open"].values, df_s["High"].values, df_s["Low"].values,
        df_s["Close"].values, df_s["SM_Net"].values, df_s.index,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPA_MAX_LOSS_PTS, tp_pts=VSCALPA_TP_PTS,
        entry_end_et=VSCALPA_ENTRY_END_ET,
        entry_gate=gate,
    )
    sc = score_trades(trades, commission_per_side=MNQ_COMMISSION,
                      dollar_per_pt=MNQ_DOLLAR_PER_PT)
    return trades, sc


def run_vscalpb(data, split_name, extra_gate=None):
    """Run vScalpB on the given split with production gates + optional extra gate."""
    df = data["df_mnq"]
    is_len = data["mnq_is_len"]

    if split_name == "IS":
        df_s = df.iloc[:is_len]
    elif split_name == "OOS":
        df_s = df.iloc[is_len:]
    else:
        df_s = df

    df_5m = resample_to_5min(df_s)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df_s.index.values, df_5m.index.values,
        df_5m["Close"].values, rsi_len=VSCALPB_RSI_LEN)

    prod = _slice(data["prod_gate_vscalpb"], split_name, is_len)
    if extra_gate is not None:
        eg = _slice(extra_gate, split_name, is_len)
        gate = prod & eg
    else:
        gate = prod

    trades = run_backtest_tp_exit(
        df_s["Open"].values, df_s["High"].values, df_s["Low"].values,
        df_s["Close"].values, df_s["SM_Net"].values, df_s.index,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
        entry_gate=gate,
    )
    sc = score_trades(trades, commission_per_side=MNQ_COMMISSION,
                      dollar_per_pt=MNQ_DOLLAR_PER_PT)
    return trades, sc


def run_vscalpc(data, split_name, extra_gate=None):
    """Run vScalpC (structure exit) with production gates + optional extra gate."""
    df = data["df_mnq"]
    is_len = data["mnq_is_len"]

    if split_name == "IS":
        df_s = df.iloc[:is_len]
    elif split_name == "OOS":
        df_s = df.iloc[is_len:]
    else:
        df_s = df

    df_5m = resample_to_5min(df_s)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df_s.index.values, df_5m.index.values,
        df_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN)

    prod = _slice(data["prod_gate_vscalpc"], split_name, is_len)
    if extra_gate is not None:
        eg = _slice(extra_gate, split_name, is_len)
        gate = prod & eg
    else:
        gate = prod

    sw_h = _slice(data["mnq_swing_highs"], split_name, is_len)
    sw_l = _slice(data["mnq_swing_lows"], split_name, is_len)

    trades = run_backtest_structure_exit(
        df_s["Open"].values, df_s["High"].values, df_s["Low"].values,
        df_s["Close"].values, df_s["SM_Net"].values, df_s.index,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPC_SL_PTS,
        tp1_pts=VSCALPC_TP1_PTS,
        swing_highs=sw_h, swing_lows=sw_l,
        swing_buffer_pts=VSCALPC_SWING_BUF,
        max_tp2_pts=VSCALPC_MAX_TP2_PTS,
        move_sl_to_be_after_tp1=True,
        breakeven_after_bars=0,
        entry_end_et=VSCALPA_ENTRY_END_ET,
        entry_gate=gate,
    )
    sc = score_vscalpc_as_flat(trades, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
    return trades, sc


def run_mesv2(data, split_name, extra_gate=None):
    """Run MES v2 with production gates + optional extra gate."""
    df = data["df_mes"]
    is_len = data["mes_is_len"]

    if split_name == "IS":
        df_s = df.iloc[:is_len]
    elif split_name == "OOS":
        df_s = df.iloc[is_len:]
    else:
        df_s = df

    df_5m = resample_to_5min(df_s)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df_s.index.values, df_5m.index.values,
        df_5m["Close"].values, rsi_len=MESV2_RSI_LEN)

    prod = _slice(data["prod_gate_mesv2"], split_name, is_len)
    if extra_gate is not None:
        eg = _slice(extra_gate, split_name, is_len)
        gate = prod & eg
    else:
        gate = prod

    trades = run_backtest_tp_exit(
        df_s["Open"].values, df_s["High"].values, df_s["Low"].values,
        df_s["Close"].values, df_s["SM_Net"].values, df_s.index,
        rsi_curr, rsi_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        entry_end_et=MESV2_ENTRY_END_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        entry_gate=gate,
    )
    sc = score_trades(trades, commission_per_side=MES_COMMISSION,
                      dollar_per_pt=MES_DOLLAR_PER_PT)
    return trades, sc


# ============================================================================
# Sweep runners
# ============================================================================

STRATEGY_RUNNERS = {
    "vScalpA": run_vscalpa,
    "vScalpB": run_vscalpb,
    "vScalpC": run_vscalpc,
    "MES_v2": run_mesv2,
}

VCR_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
VCR_MIN_BARS = [0, 30, 60]

DVPOC_DIST_THRESHOLDS = [5, 7, 10]


def pct_change(new_val, base_val):
    """Compute percentage change from base to new. Returns 0 if base is 0."""
    if base_val == 0 or base_val is None:
        return 0.0
    return (new_val - base_val) / abs(base_val) * 100


def run_vcr_sweep(data):
    """Run the VCR regime gate sweep across all strategies."""
    print("\n" + "=" * 130)
    print("GATE 1: VCR REGIME GATE SWEEP (all strategies)")
    print("  Block entries when developing VCR at bar[i-1] > threshold")
    print("=" * 130)

    all_strats = ["vScalpA", "vScalpB", "vScalpC", "MES_v2"]

    # --- Run baselines (production gates only, no VCR gate) ---
    print("\n--- Baselines (production gates, no VCR gate) ---")
    baselines = {}  # (strat, split) -> sc
    for strat_name in all_strats:
        runner = STRATEGY_RUNNERS[strat_name]
        for split in ["IS", "OOS"]:
            _, sc = runner(data, split, extra_gate=None)
            baselines[(strat_name, split)] = sc
            if sc:
                print(f"  {strat_name:>8} {split}: {sc['count']} trades, "
                      f"WR {sc['win_rate']}%, PF {sc['pf']}, "
                      f"${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']}")
            else:
                print(f"  {strat_name:>8} {split}: NO TRADES")

    # --- Sweep ---
    results = []  # list of dicts for summary

    for vcr_thresh in VCR_THRESHOLDS:
        for min_bars in VCR_MIN_BARS:
            config_label = f"VCR>{vcr_thresh:.2f}, min_bars={min_bars}"
            print(f"\n{'─'*130}")
            print(f"Config: {config_label}")
            print(f"{'─'*130}")

            # Build VCR gate for MNQ and MES
            mnq_vcr_gate = build_vcr_gate(
                data["mnq_vcr"], data["mnq_et_mins"],
                threshold=vcr_thresh, min_rth_bars=min_bars)
            mes_vcr_gate = build_vcr_gate(
                data["mes_vcr"], data["mes_et_mins"],
                threshold=vcr_thresh, min_rth_bars=min_bars)

            gate_map = {
                "vScalpA": mnq_vcr_gate,
                "vScalpB": mnq_vcr_gate,
                "vScalpC": mnq_vcr_gate,
                "MES_v2": mes_vcr_gate,
            }

            for strat_name in all_strats:
                runner = STRATEGY_RUNNERS[strat_name]
                extra_gate = gate_map[strat_name]

                sc_is = sc_oos = None
                bl_is = baselines[(strat_name, "IS")]
                bl_oos = baselines[(strat_name, "OOS")]

                for split in ["IS", "OOS"]:
                    _, sc = runner(data, split, extra_gate=extra_gate)
                    if split == "IS":
                        sc_is = sc
                    else:
                        sc_oos = sc

                # Compute deltas
                bl_is_count = bl_is["count"] if bl_is else 0
                bl_oos_count = bl_oos["count"] if bl_oos else 0
                is_count = sc_is["count"] if sc_is else 0
                oos_count = sc_oos["count"] if sc_oos else 0

                is_blocked = bl_is_count - is_count
                oos_blocked = bl_oos_count - oos_count
                total_blocked = is_blocked + oos_blocked
                total_baseline = bl_is_count + bl_oos_count
                block_rate = total_blocked / total_baseline * 100 if total_baseline > 0 else 0

                is_pf = sc_is["pf"] if sc_is else 0
                oos_pf = sc_oos["pf"] if sc_oos else 0
                bl_is_pf = bl_is["pf"] if bl_is else 0
                bl_oos_pf = bl_oos["pf"] if bl_oos else 0

                is_sharpe = sc_is["sharpe"] if sc_is else 0
                oos_sharpe = sc_oos["sharpe"] if sc_oos else 0
                bl_is_sharpe = bl_is["sharpe"] if bl_is else 0
                bl_oos_sharpe = bl_oos["sharpe"] if bl_oos else 0

                dpf_is = pct_change(is_pf, bl_is_pf)
                dpf_oos = pct_change(oos_pf, bl_oos_pf)
                dsharpe_is = pct_change(is_sharpe, bl_is_sharpe)
                dsharpe_oos = pct_change(oos_sharpe, bl_oos_sharpe)

                # Print
                def _fmt_sc(sc, label):
                    if sc is None:
                        return f"{label}: NO TRADES"
                    return (f"{label}: {sc['count']} trades, "
                            f"WR {sc['win_rate']}%, PF {sc['pf']}, "
                            f"${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']}")

                print(f"  [{strat_name}] {_fmt_sc(sc_is, 'IS')} | {_fmt_sc(sc_oos, 'OOS')}")
                print(f"    Block rate: {block_rate:.1f}% ({total_blocked}/{total_baseline} signals)")
                print(f"    dPF (IS): {dpf_is:+.1f}% | dPF (OOS): {dpf_oos:+.1f}%")
                print(f"    dSharpe (IS): {dsharpe_is:+.1f}% | dSharpe (OOS): {dsharpe_oos:+.1f}%")

                results.append({
                    "gate": "VCR",
                    "config": config_label,
                    "strategy": strat_name,
                    "vcr_thresh": vcr_thresh,
                    "min_bars": min_bars,
                    "is_count": is_count,
                    "oos_count": oos_count,
                    "is_pf": is_pf,
                    "oos_pf": oos_pf,
                    "is_sharpe": is_sharpe,
                    "oos_sharpe": oos_sharpe,
                    "is_net": sc_is["net_dollar"] if sc_is else 0,
                    "oos_net": sc_oos["net_dollar"] if sc_oos else 0,
                    "is_wr": sc_is["win_rate"] if sc_is else 0,
                    "oos_wr": sc_oos["win_rate"] if sc_oos else 0,
                    "block_rate": block_rate,
                    "dpf_is": dpf_is,
                    "dpf_oos": dpf_oos,
                    "dsharpe_is": dsharpe_is,
                    "dsharpe_oos": dsharpe_oos,
                    "bl_is_pf": bl_is_pf,
                    "bl_oos_pf": bl_oos_pf,
                })

    return results, baselines


def run_dvpoc_distance_sweep(data):
    """Run the dVPOC distance gate sweep (MES v2 only)."""
    print("\n" + "=" * 130)
    print("GATE 2: dVPOC DISTANCE GATE SWEEP (MES v2 only)")
    print("  Block MES entries when abs(close - dVPOC) < threshold at bar[i-1]")
    print("=" * 130)

    # --- Baseline ---
    print("\n--- Baseline (production gates, no dVPOC distance gate) ---")
    baselines = {}
    for split in ["IS", "OOS"]:
        _, sc = run_mesv2(data, split, extra_gate=None)
        baselines[split] = sc
        if sc:
            print(f"  MES_v2 {split}: {sc['count']} trades, "
                  f"WR {sc['win_rate']}%, PF {sc['pf']}, "
                  f"${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']}")
        else:
            print(f"  MES_v2 {split}: NO TRADES")

    # --- Sweep ---
    results = []

    for dist_pts in DVPOC_DIST_THRESHOLDS:
        config_label = f"dVPOC_dist<{dist_pts}pts"
        print(f"\n{'─'*130}")
        print(f"Config: {config_label}")
        print(f"{'─'*130}")

        # Build distance gate for MES
        mes_dist_gate = build_dvpoc_distance_gate(
            data["mes_closes"], data["mes_dvpoc"], min_distance_pts=dist_pts)

        sc_is = sc_oos = None
        bl_is = baselines["IS"]
        bl_oos = baselines["OOS"]

        for split in ["IS", "OOS"]:
            _, sc = run_mesv2(data, split, extra_gate=mes_dist_gate)
            if split == "IS":
                sc_is = sc
            else:
                sc_oos = sc

        # Compute deltas
        bl_is_count = bl_is["count"] if bl_is else 0
        bl_oos_count = bl_oos["count"] if bl_oos else 0
        is_count = sc_is["count"] if sc_is else 0
        oos_count = sc_oos["count"] if sc_oos else 0

        total_blocked = (bl_is_count - is_count) + (bl_oos_count - oos_count)
        total_baseline = bl_is_count + bl_oos_count
        block_rate = total_blocked / total_baseline * 100 if total_baseline > 0 else 0

        is_pf = sc_is["pf"] if sc_is else 0
        oos_pf = sc_oos["pf"] if sc_oos else 0
        bl_is_pf = bl_is["pf"] if bl_is else 0
        bl_oos_pf = bl_oos["pf"] if bl_oos else 0

        is_sharpe = sc_is["sharpe"] if sc_is else 0
        oos_sharpe = sc_oos["sharpe"] if sc_oos else 0
        bl_is_sharpe = bl_is["sharpe"] if bl_is else 0
        bl_oos_sharpe = bl_oos["sharpe"] if bl_oos else 0

        dpf_is = pct_change(is_pf, bl_is_pf)
        dpf_oos = pct_change(oos_pf, bl_oos_pf)
        dsharpe_is = pct_change(is_sharpe, bl_is_sharpe)
        dsharpe_oos = pct_change(oos_sharpe, bl_oos_sharpe)

        def _fmt_sc(sc, label):
            if sc is None:
                return f"{label}: NO TRADES"
            return (f"{label}: {sc['count']} trades, "
                    f"WR {sc['win_rate']}%, PF {sc['pf']}, "
                    f"${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']}")

        print(f"  [MES_v2] {_fmt_sc(sc_is, 'IS')} | {_fmt_sc(sc_oos, 'OOS')}")
        print(f"    Block rate: {block_rate:.1f}% ({total_blocked}/{total_baseline} signals)")
        print(f"    dPF (IS): {dpf_is:+.1f}% | dPF (OOS): {dpf_oos:+.1f}%")
        print(f"    dSharpe (IS): {dsharpe_is:+.1f}% | dSharpe (OOS): {dsharpe_oos:+.1f}%")

        results.append({
            "gate": "dVPOC_dist",
            "config": config_label,
            "strategy": "MES_v2",
            "dist_pts": dist_pts,
            "is_count": is_count,
            "oos_count": oos_count,
            "is_pf": is_pf,
            "oos_pf": oos_pf,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "is_net": sc_is["net_dollar"] if sc_is else 0,
            "oos_net": sc_oos["net_dollar"] if sc_oos else 0,
            "is_wr": sc_is["win_rate"] if sc_is else 0,
            "oos_wr": sc_oos["win_rate"] if sc_oos else 0,
            "block_rate": block_rate,
            "dpf_is": dpf_is,
            "dpf_oos": dpf_oos,
            "dsharpe_is": dsharpe_is,
            "dsharpe_oos": dsharpe_oos,
            "bl_is_pf": bl_is_pf,
            "bl_oos_pf": bl_oos_pf,
        })

    return results, baselines


# ============================================================================
# Summary tables
# ============================================================================

def print_vcr_summary(vcr_results):
    """Print ranked summary table for VCR gate sweep."""
    print("\n" + "=" * 150)
    print("VCR GATE SWEEP — SUMMARY (ranked by OOS PF improvement)")
    print("=" * 150)

    header = (f"{'Config':<30} {'Strategy':>8} {'Block%':>7} "
              f"{'IS Trd':>7} {'IS WR%':>7} {'IS PF':>7} {'IS Net$':>9} {'IS Shrp':>8} "
              f"{'OOS Trd':>8} {'OOS WR%':>8} {'OOS PF':>8} {'OOS Net$':>10} {'OOS Shrp':>9} "
              f"{'dPF_IS':>8} {'dPF_OOS':>8}")
    print(header)
    print("-" * 150)

    # Sort by OOS PF improvement descending
    sorted_results = sorted(vcr_results, key=lambda r: r["dpf_oos"], reverse=True)

    for r in sorted_results:
        print(f"{r['config']:<30} {r['strategy']:>8} {r['block_rate']:>6.1f}% "
              f"{r['is_count']:>7} {r['is_wr']:>6.1f}% {r['is_pf']:>7.3f} "
              f"{r['is_net']:>+9.0f} {r['is_sharpe']:>8.3f} "
              f"{r['oos_count']:>8} {r['oos_wr']:>7.1f}% {r['oos_pf']:>8.3f} "
              f"{r['oos_net']:>+10.0f} {r['oos_sharpe']:>9.3f} "
              f"{r['dpf_is']:>+7.1f}% {r['dpf_oos']:>+7.1f}%")


def print_dvpoc_summary(dvpoc_results):
    """Print ranked summary table for dVPOC distance gate sweep."""
    print("\n" + "=" * 150)
    print("dVPOC DISTANCE GATE SWEEP — SUMMARY (ranked by OOS PF improvement)")
    print("=" * 150)

    header = (f"{'Config':<30} {'Strategy':>8} {'Block%':>7} "
              f"{'IS Trd':>7} {'IS WR%':>7} {'IS PF':>7} {'IS Net$':>9} {'IS Shrp':>8} "
              f"{'OOS Trd':>8} {'OOS WR%':>8} {'OOS PF':>8} {'OOS Net$':>10} {'OOS Shrp':>9} "
              f"{'dPF_IS':>8} {'dPF_OOS':>8}")
    print(header)
    print("-" * 150)

    sorted_results = sorted(dvpoc_results, key=lambda r: r["dpf_oos"], reverse=True)

    for r in sorted_results:
        print(f"{r['config']:<30} {r['strategy']:>8} {r['block_rate']:>6.1f}% "
              f"{r['is_count']:>7} {r['is_wr']:>6.1f}% {r['is_pf']:>7.3f} "
              f"{r['is_net']:>+9.0f} {r['is_sharpe']:>8.3f} "
              f"{r['oos_count']:>8} {r['oos_wr']:>7.1f}% {r['oos_pf']:>8.3f} "
              f"{r['oos_net']:>+10.0f} {r['oos_sharpe']:>9.3f} "
              f"{r['dpf_is']:>+7.1f}% {r['dpf_oos']:>+7.1f}%")


def print_top_configs(vcr_results, dvpoc_results):
    """Print the top VCR configs per strategy + dVPOC configs."""
    print("\n" + "=" * 100)
    print("TOP CONFIGS BY STRATEGY (best OOS PF improvement with positive IS/OOS dPF)")
    print("=" * 100)

    all_strats = ["vScalpA", "vScalpB", "vScalpC", "MES_v2"]

    for strat in all_strats:
        strat_results = [r for r in vcr_results if r["strategy"] == strat]
        # Filter: both IS and OOS PF must improve (dPF > 0)
        passing = [r for r in strat_results if r["dpf_is"] > 0 and r["dpf_oos"] > 0]

        if passing:
            best = max(passing, key=lambda r: r["dpf_oos"])
            print(f"\n  {strat}: BEST VCR config = {best['config']}")
            print(f"    Block rate: {best['block_rate']:.1f}%")
            print(f"    IS:  PF {best['bl_is_pf']:.3f} -> {best['is_pf']:.3f} ({best['dpf_is']:+.1f}%), "
                  f"Sharpe {best['is_sharpe']:.3f}")
            print(f"    OOS: PF {best['bl_oos_pf']:.3f} -> {best['oos_pf']:.3f} ({best['dpf_oos']:+.1f}%), "
                  f"Sharpe {best['oos_sharpe']:.3f}")
        else:
            print(f"\n  {strat}: NO VCR config improves PF in both IS and OOS")

    # dVPOC distance
    passing_d = [r for r in dvpoc_results if r["dpf_is"] > 0 and r["dpf_oos"] > 0]
    if passing_d:
        best_d = max(passing_d, key=lambda r: r["dpf_oos"])
        print(f"\n  MES_v2 dVPOC: BEST config = {best_d['config']}")
        print(f"    Block rate: {best_d['block_rate']:.1f}%")
        print(f"    IS:  PF {best_d['bl_is_pf']:.3f} -> {best_d['is_pf']:.3f} ({best_d['dpf_is']:+.1f}%)")
        print(f"    OOS: PF {best_d['bl_oos_pf']:.3f} -> {best_d['oos_pf']:.3f} ({best_d['dpf_oos']:+.1f}%)")
    else:
        print(f"\n  MES_v2 dVPOC: NO config improves PF in both IS and OOS")


# ============================================================================
# Main
# ============================================================================

def main():
    data = load_all_data()

    vcr_results, vcr_baselines = run_vcr_sweep(data)
    dvpoc_results, dvpoc_baselines = run_dvpoc_distance_sweep(data)

    print_vcr_summary(vcr_results)
    print_dvpoc_summary(dvpoc_results)
    print_top_configs(vcr_results, dvpoc_results)

    print(f"\n{'='*70}")
    print("VCR + dVPOC Distance Gate Sweep complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
