"""
Developing Daily VPOC Forensics — Map backtest trades against developing VPOC.

Computes the developing (intraday) VPOC, VCR (strength), stability, and
developing VWAP at each RTH bar, then maps each portfolio trade to its dPOC
context at entry time (bar[i-1] convention).

Output: WR/PF breakdown by distance bands, directional context, stability,
time-of-day, VCR regime, weekly VPOC interaction, and prior-day VPOC proximity.

Usage:
    cd backtesting_engine/strategies
    python3 developing_vpoc_forensics.py
"""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ET_TZ = ZoneInfo("America/New_York")

# --- Path setup (same as ict_forensics.py) ---
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
    NY_OPEN_ET,
    NY_CLOSE_ET,
)

from generate_session import (
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
from v10_test_common import compute_prior_day_levels
from structure_exit_common import (
    compute_swing_levels,
    run_backtest_structure_exit,
)
from run_and_save_portfolio import (
    load_vix_gate,
    _get_bar_dates_et,
    VSCALPC_TP1_PTS, VSCALPC_MAX_TP2_PTS, VSCALPC_SL_PTS,
    VSCALPC_SWING_LB, VSCALPC_SWING_PR, VSCALPC_SWING_BUF,
    MESV2_EOD_ET, MESV2_ENTRY_END_ET,
)


# ============================================================================
# Part 1: Developing VPOC Computation
# ============================================================================

BIN_WIDTHS = {"MNQ": 2, "MES": 5}


def compute_developing_vpoc(times, closes, volumes, et_mins, bin_width,
                            highs=None, lows=None):
    """Compute developing daily VPOC at each bar.

    Forward-only pass. Resets on calendar date change. RTH-only accumulation.
    Returns:
        dvpoc:      numpy array of developing VPOC at each bar (NaN outside RTH
                    or before first RTH bar of the day)
        strength:   numpy array of VCR (max_bin_vol / total_vol) at each bar
        stability:  numpy array of bars since last VPOC shift at each bar
        dev_vwap:   numpy array of developing VWAP at each bar
        prior_vpoc: numpy array of prior-day final VPOC mapped to each bar
    """
    n = len(times)
    dvpoc = np.full(n, np.nan)
    strength = np.full(n, np.nan)
    stability_arr = np.full(n, np.nan)
    dev_vwap = np.full(n, np.nan)
    prior_vpoc = np.full(n, np.nan)

    # Convert timestamps to ET dates for day boundary detection
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    et_times = idx.tz_convert(_ET_TZ)
    et_dates = et_times.date

    # Accumulators — reset on day change
    current_date = None
    day_closes = []
    day_volumes = []
    prev_vpoc_price = np.nan
    bars_since_shift = 0
    cum_pv = 0.0     # cumulative price * volume for VWAP
    cum_vol = 0.0    # cumulative volume for VWAP

    # Track completed daily VPOCs for prior-day mapping
    daily_final_vpoc = {}  # date -> final VPOC price

    for i in range(n):
        bar_date = et_dates[i]

        # Day boundary: finalize prior day, reset accumulators
        if bar_date != current_date:
            # Save final VPOC for the completed day
            if current_date is not None and len(day_closes) > 0:
                profile = _compute_value_area(day_closes, day_volumes, bin_width)
                if profile is not None:
                    daily_final_vpoc[current_date] = profile[0]  # VPOC price

            current_date = bar_date
            day_closes = []
            day_volumes = []
            prev_vpoc_price = np.nan
            bars_since_shift = 0
            cum_pv = 0.0
            cum_vol = 0.0

        # Only accumulate during RTH
        if not (NY_OPEN_ET <= et_mins[i] < NY_CLOSE_ET):
            continue

        # Skip zero-volume bars (match engine's bar.volume > 0 guard)
        v = volumes[i]
        if v <= 0:
            continue

        # Accumulate bar data
        c = closes[i]
        day_closes.append(c)
        day_volumes.append(v)

        # Developing VWAP — use typical price (H+L+C)/3 to match engine
        if highs is not None and lows is not None:
            tp = (highs[i] + lows[i] + c) / 3.0
        else:
            tp = c
        cum_pv += tp * v
        cum_vol += v
        if cum_vol > 0:
            dev_vwap[i] = cum_pv / cum_vol

        # Compute developing VPOC via binning
        total_vol = sum(day_volumes)
        if total_vol <= 0:
            continue

        closes_arr = np.array(day_closes, dtype=np.float64)
        volumes_arr = np.array(day_volumes, dtype=np.float64)

        price_min = np.floor(np.min(closes_arr) / bin_width) * bin_width
        price_max = np.ceil(np.max(closes_arr) / bin_width) * bin_width
        if price_min == price_max:
            price_max = price_min + bin_width

        n_bins = int(round((price_max - price_min) / bin_width)) + 1
        bin_volumes = np.zeros(n_bins)

        for ci, vi in zip(closes_arr, volumes_arr):
            b_idx = int(round((ci - price_min) / bin_width))
            b_idx = min(max(b_idx, 0), n_bins - 1)
            bin_volumes[b_idx] += vi

        vpoc_idx = int(np.argmax(bin_volumes))
        vpoc_price = price_min + vpoc_idx * bin_width
        max_bin_vol = bin_volumes[vpoc_idx]

        dvpoc[i] = vpoc_price
        strength[i] = max_bin_vol / total_vol

        # Stability: bars since last VPOC shift
        if np.isnan(prev_vpoc_price) or abs(vpoc_price - prev_vpoc_price) < 1e-9:
            bars_since_shift += 1
        else:
            bars_since_shift = 0
        stability_arr[i] = bars_since_shift
        prev_vpoc_price = vpoc_price

    # Save final day
    if current_date is not None and len(day_closes) > 0:
        profile = _compute_value_area(day_closes, day_volumes, bin_width)
        if profile is not None:
            daily_final_vpoc[current_date] = profile[0]

    # Map prior-day VPOC to each bar
    sorted_dates = sorted(daily_final_vpoc.keys())
    date_to_prev_vpoc = {}
    for j in range(1, len(sorted_dates)):
        date_to_prev_vpoc[sorted_dates[j]] = daily_final_vpoc[sorted_dates[j - 1]]

    for i in range(n):
        bar_date = et_dates[i]
        if bar_date in date_to_prev_vpoc:
            prior_vpoc[i] = date_to_prev_vpoc[bar_date]

    return dvpoc, strength, stability_arr, dev_vwap, prior_vpoc


# ============================================================================
# Part 2: Run Backtest and Collect Trades (mirrors ict_forensics.py)
# ============================================================================

def run_portfolio_trades():
    """Run all 4 strategies with gates, return trades + data."""
    print("=" * 70)
    print("DEVELOPING VPOC FORENSICS — Loading data and running portfolio")
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
    vscalpc_entries = {}
    for t in trades_c:
        eidx = t["entry_idx"]
        if eidx not in vscalpc_entries:
            vscalpc_entries[eidx] = []
        vscalpc_entries[eidx].append(t)

    comm = MNQ_COMMISSION * 2
    n_entries_c = len(vscalpc_entries)
    print(f"    {n_entries_c} entries ({len(trades_c)} legs)")
    for eidx in sorted(vscalpc_entries.keys()):
        legs = vscalpc_entries[eidx]
        total_pts = sum(l['pts'] for l in legs)
        total_pnl = sum(l['pts'] * MNQ_DOLLAR_PER_PT - comm for l in legs)
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

    # vScalpB uses same RSI length as vScalpA but different thresholds
    # For MES, need separate RSI with MESV2_RSI_LEN
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

    return {
        'all_trades': all_trades,
        'df_mnq': df_mnq,
        'df_mes': df_mes,
    }


# ============================================================================
# Part 3: Statistics Helpers
# ============================================================================

def compute_group_stats(pnls):
    """Compute WR, PF, avg P&L for a list of P&L values.

    Returns dict with count, wr, pf, avg_pnl, total_pnl or None if empty.
    """
    if not pnls:
        return None

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


def print_table_row(label, stats, baseline_wr, baseline_pct=None):
    """Print a single row of a results table."""
    if stats is None or stats['count'] == 0:
        return
    delta = stats['wr'] - baseline_wr
    sign = '+' if delta >= 0 else ''
    base_str = f"  Base {baseline_pct:.0f}%" if baseline_pct is not None else ""
    print(f"  {label:<50s} {stats['count']:>4d} trades, "
          f"WR {stats['wr']:>5.1f}% ({sign}{delta:.1f}pp), "
          f"PF {stats['pf']:>6.3f}, "
          f"Avg ${stats['avg_pnl']:>+7.2f}{base_str}")


# ============================================================================
# Part 4: Analysis
# ============================================================================

def run_analysis(portfolio_data):
    """Run the full developing VPOC forensics analysis."""

    all_trades = portfolio_data['all_trades']
    df_mnq = portfolio_data['df_mnq']
    df_mes = portfolio_data['df_mes']

    # ------------------------------------------------------------------
    # Compute developing VPOC for both instruments
    # ------------------------------------------------------------------
    print("\n--- Computing Developing VPOC ---")

    dvpoc_data = {}
    for inst, df in [('MNQ', df_mnq), ('MES', df_mes)]:
        times = df.index
        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values
        volumes = df["Volume"].values
        et_mins = compute_et_minutes(times)
        bw = BIN_WIDTHS[inst]

        dvpoc, vcr, stab, d_vwap, prior_v = compute_developing_vpoc(
            times, closes, volumes, et_mins, bw, highs=highs, lows=lows)

        n_valid = np.sum(~np.isnan(dvpoc))
        print(f"  {inst}: {n_valid} bars with developing VPOC (bin_width={bw})")

        dvpoc_data[inst] = {
            'dvpoc': dvpoc,
            'vcr': vcr,
            'stability': stab,
            'dev_vwap': d_vwap,
            'prior_vpoc': prior_v,
            'closes': closes,
            'et_mins': et_mins,
        }

    # Also compute weekly VPOC for Tier 3 analysis
    print("\n--- Computing Weekly VPOC ---")
    from sr_prior_day_levels_sweep import compute_rth_volume_profile as _rth_vp
    weekly_vpoc = {}
    for inst, df in [('MNQ', df_mnq), ('MES', df_mes)]:
        wv, _, _ = _rth_vp(
            df.index, df["Close"].values, df["Volume"].values,
            compute_et_minutes(df.index), BIN_WIDTHS[inst])
        weekly_vpoc[inst] = wv
        n_valid = np.sum(~np.isnan(wv))
        print(f"  {inst}: {n_valid} bars with weekly VPOC")

    # ------------------------------------------------------------------
    # Map each trade to dPOC features at entry (bar[i-1] convention)
    # ------------------------------------------------------------------
    print("\n--- Mapping Trades to Developing VPOC ---")

    mapped_trades = []

    for trade, strategy, instrument, dpp, comm_per_leg in all_trades:
        entry_idx = trade['entry_idx']
        entry_price = trade.get('entry_price', trade.get('entry', 0))
        side = trade['side']

        # bar[i-1] lookup index
        lookup_idx = entry_idx - 1 if entry_idx > 0 else 0

        inst_d = dvpoc_data[instrument]
        d = inst_d['dvpoc']
        vcr_arr = inst_d['vcr']
        stab_arr = inst_d['stability']
        vwap_arr = inst_d['dev_vwap']

        # Compute P&L in dollars
        if trade.get('_is_structure'):
            pnl_dollar = trade['_pnl_dollar']
        else:
            comm_pts = (comm_per_leg * 2) / dpp
            pnl_dollar = (trade['pts'] - comm_pts) * dpp

        # dVPOC features at bar[i-1]
        dvpoc_val = d[lookup_idx] if lookup_idx < len(d) else np.nan
        vcr_val = vcr_arr[lookup_idx] if lookup_idx < len(vcr_arr) else np.nan
        stab_val = stab_arr[lookup_idx] if lookup_idx < len(stab_arr) else np.nan
        vwap_val = vwap_arr[lookup_idx] if lookup_idx < len(vwap_arr) else np.nan

        # Distance metrics
        if not np.isnan(dvpoc_val):
            dvpoc_dist = entry_price - dvpoc_val    # signed: positive = above dVPOC
            dvpoc_abs_dist = abs(dvpoc_dist)
        else:
            dvpoc_dist = np.nan
            dvpoc_abs_dist = np.nan

        # dVPOC-VWAP divergence
        if not np.isnan(dvpoc_val) and not np.isnan(vwap_val):
            dvpoc_vwap_dist = abs(dvpoc_val - vwap_val)
        else:
            dvpoc_vwap_dist = np.nan

        # ET minutes at entry
        entry_et = inst_d['et_mins'][entry_idx] if entry_idx < len(inst_d['et_mins']) else 0

        # Weekly VPOC at entry
        wv_arr = weekly_vpoc.get(instrument)
        wvpoc_val = np.nan
        if wv_arr is not None and lookup_idx < len(wv_arr):
            wvpoc_val = wv_arr[lookup_idx]

        # Prior-day VPOC
        prior_v_arr = inst_d.get('prior_vpoc', None)
        if prior_v_arr is not None:
            prior_v_val = prior_v_arr[lookup_idx] if lookup_idx < len(prior_v_arr) else np.nan
        else:
            prior_v_val = np.nan

        if not np.isnan(prior_v_val):
            prior_vpoc_dist = abs(entry_price - prior_v_val)
        else:
            prior_vpoc_dist = np.nan

        mapped_trades.append({
            'strategy': strategy,
            'instrument': instrument,
            'side': side,
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'pnl_dollar': pnl_dollar,
            'is_win': pnl_dollar > 0,
            'dvpoc_at_entry': dvpoc_val,
            'dvpoc_dist': dvpoc_dist,
            'dvpoc_abs_dist': dvpoc_abs_dist,
            'dvpoc_strength': vcr_val,
            'dvpoc_stability': stab_val,
            'dvpoc_vwap_dist': dvpoc_vwap_dist,
            'entry_et_mins': entry_et,
            'weekly_vpoc': wvpoc_val,
            'prior_vpoc_dist': prior_vpoc_dist,
        })

    # Filter to trades with valid dVPOC
    valid_trades = [t for t in mapped_trades if not np.isnan(t['dvpoc_at_entry'])]
    no_dvpoc = len(mapped_trades) - len(valid_trades)

    print(f"  Mapped {len(mapped_trades)} trades total")
    print(f"  {len(valid_trades)} with valid dVPOC, {no_dvpoc} without (pre-RTH or missing)")

    # ------------------------------------------------------------------
    # Compute baseline
    # ------------------------------------------------------------------
    total_trades = len(valid_trades)
    if total_trades == 0:
        print("\nERROR: No trades with valid developing VPOC. Cannot run analysis.")
        return

    total_wins = sum(1 for t in valid_trades if t['is_win'])
    total_pnl = sum(t['pnl_dollar'] for t in valid_trades)
    baseline_wr = total_wins / total_trades * 100

    gross_profit = sum(t['pnl_dollar'] for t in valid_trades if t['pnl_dollar'] > 0)
    gross_loss = abs(sum(t['pnl_dollar'] for t in valid_trades if t['pnl_dollar'] <= 0))
    baseline_pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    date_min = df_mnq.index[0].strftime("%Y-%m-%d")
    date_max = df_mnq.index[-1].strftime("%Y-%m-%d")

    print("\n" + "=" * 80)
    print(f"DEVELOPING VPOC FORENSICS — {date_min} to {date_max}")
    print(f"4 strategies, all gates applied")
    print("=" * 80)
    print(f"\nBASELINE (trades with valid dVPOC): {total_trades} trades, "
          f"WR {baseline_wr:.1f}%, PF {baseline_pf:.3f}, "
          f"Net ${total_pnl:+,.2f}")

    # ==================================================================
    # TIER 1: Distance Bands + VWAP Divergence + Null Baseline
    # ==================================================================
    print(f"\n{'='*80}")
    print("TIER 1: DISTANCE BANDS + VWAP DIVERGENCE + NULL BASELINE")
    print(f"{'='*80}")

    # --- 1. Distance bands: overall and per strategy ---
    distance_bands = [(0, 5), (5, 10), (10, 20), (20, 9999)]
    band_labels = ["0-5 pts", "5-10 pts", "10-20 pts", "20+ pts"]

    print(f"\n--- 1. WR/PF by Distance to Developing VPOC (Overall) ---")
    for (lo, hi), label in zip(distance_bands, band_labels):
        pnls = [t['pnl_dollar'] for t in valid_trades
                if lo <= t['dvpoc_abs_dist'] < hi]
        stats = compute_group_stats(pnls)
        print_table_row(f"dVPOC dist {label}", stats, baseline_wr)

    # Per strategy
    strategies = ['vScalpA', 'vScalpB', 'vScalpC', 'MES_v2']
    for strat in strategies:
        strat_trades = [t for t in valid_trades if t['strategy'] == strat]
        if not strat_trades:
            continue
        strat_wins = sum(1 for t in strat_trades if t['is_win'])
        strat_wr = strat_wins / len(strat_trades) * 100
        print(f"\n  [{strat}] ({len(strat_trades)} trades, WR {strat_wr:.1f}%)")
        for (lo, hi), label in zip(distance_bands, band_labels):
            pnls = [t['pnl_dollar'] for t in strat_trades
                    if lo <= t['dvpoc_abs_dist'] < hi]
            stats = compute_group_stats(pnls)
            print_table_row(f"  dVPOC dist {label}", stats, strat_wr)

    # --- 2. dVPOC-VWAP divergence bands ---
    print(f"\n--- 2. WR/PF by dVPOC-VWAP Divergence ---")
    vwap_bands = [(0, 3), (3, 7), (7, 15), (15, 9999)]
    vwap_labels = ["0-3 pts", "3-7 pts", "7-15 pts", "15+ pts"]

    vwap_valid = [t for t in valid_trades if not np.isnan(t['dvpoc_vwap_dist'])]
    for (lo, hi), label in zip(vwap_bands, vwap_labels):
        pnls = [t['pnl_dollar'] for t in vwap_valid
                if lo <= t['dvpoc_vwap_dist'] < hi]
        stats = compute_group_stats(pnls)
        print_table_row(f"|dVPOC - VWAP| {label}", stats, baseline_wr)

    # --- 3. Null baseline: what % of RTH bars fall in each distance band? ---
    print(f"\n--- 3. Null Baseline: Price Distribution Around dVPOC ---")
    print("  (What % of ALL RTH bars fall in each distance band from dVPOC?)")
    for inst in ['MNQ', 'MES']:
        d_arr = dvpoc_data[inst]['dvpoc']
        c_arr = dvpoc_data[inst]['closes']
        valid_mask = ~np.isnan(d_arr)
        if np.sum(valid_mask) == 0:
            continue
        abs_dists = np.abs(c_arr[valid_mask] - d_arr[valid_mask])
        n_bars = len(abs_dists)
        print(f"\n  {inst}: {n_bars} RTH bars with valid dVPOC")
        for (lo, hi), label in zip(distance_bands, band_labels):
            count = np.sum((abs_dists >= lo) & (abs_dists < hi))
            pct = count / n_bars * 100
            # Also compute what % of trade entries fall in this band for comparison
            inst_trades = [t for t in valid_trades
                           if t['instrument'] == inst and lo <= t['dvpoc_abs_dist'] < hi]
            inst_total = len([t for t in valid_trades if t['instrument'] == inst])
            trade_pct = len(inst_trades) / inst_total * 100 if inst_total > 0 else 0
            enrichment = trade_pct / pct if pct > 0 else 0
            print(f"    {label:<12s}  Bars: {pct:5.1f}%  Trades: {trade_pct:5.1f}%  "
                  f"Enrichment: {enrichment:.2f}x")

    # ==================================================================
    # TIER 2: DIRECTIONAL + STABILITY + TIME-OF-DAY
    # ==================================================================
    print(f"\n{'='*80}")
    print("TIER 2: DIRECTIONAL + STABILITY + TIME-OF-DAY")
    print(f"{'='*80}")

    # --- 4. Directional: entry above vs below dVPOC ---
    print(f"\n--- 4. Directional: Entry Position Relative to dVPOC ---")
    for side_label, side_val in [('Long', 'long'), ('Short', 'short')]:
        side_trades = [t for t in valid_trades if t['side'] == side_val]
        if not side_trades:
            continue
        side_wins = sum(1 for t in side_trades if t['is_win'])
        side_wr = side_wins / len(side_trades) * 100
        print(f"\n  {side_label} trades ({len(side_trades)}, WR {side_wr:.1f}%):")

        # Above dVPOC (dvpoc_dist > 0)
        pnls = [t['pnl_dollar'] for t in side_trades if t['dvpoc_dist'] > 0]
        stats = compute_group_stats(pnls)
        print_table_row(f"  Entry ABOVE dVPOC", stats, side_wr)

        # Below dVPOC (dvpoc_dist < 0)
        pnls = [t['pnl_dollar'] for t in side_trades if t['dvpoc_dist'] < 0]
        stats = compute_group_stats(pnls)
        print_table_row(f"  Entry BELOW dVPOC", stats, side_wr)

        # Near dVPOC (< 5pts)
        pnls = [t['pnl_dollar'] for t in side_trades if t['dvpoc_abs_dist'] < 5]
        stats = compute_group_stats(pnls)
        print_table_row(f"  Entry NEAR dVPOC (<5pts)", stats, side_wr)

    # --- 5. Stability: stable (>60 bars) vs unstable ---
    print(f"\n--- 5. dVPOC Stability at Entry ---")
    stab_valid = [t for t in valid_trades if not np.isnan(t['dvpoc_stability'])]

    pnls_stable = [t['pnl_dollar'] for t in stab_valid if t['dvpoc_stability'] > 60]
    stats_stable = compute_group_stats(pnls_stable)
    print_table_row("Stable dVPOC (>60 bars since shift)", stats_stable, baseline_wr)

    pnls_unstable = [t['pnl_dollar'] for t in stab_valid if t['dvpoc_stability'] <= 60]
    stats_unstable = compute_group_stats(pnls_unstable)
    print_table_row("Unstable dVPOC (<=60 bars since shift)", stats_unstable, baseline_wr)

    # Finer stability bands
    stab_bands = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 9999)]
    stab_labels = ["0-10 bars", "10-30 bars", "30-60 bars", "60-120 bars", "120+ bars"]
    print("\n  Finer stability bands:")
    for (lo, hi), label in zip(stab_bands, stab_labels):
        pnls = [t['pnl_dollar'] for t in stab_valid if lo <= t['dvpoc_stability'] < hi]
        stats = compute_group_stats(pnls)
        print_table_row(f"  Stability {label}", stats, baseline_wr)

    # --- 6. Time-of-day ---
    print(f"\n--- 6. Time-of-Day Segments ---")
    print("  (Early dVPOC expected to be noisier)")
    tod_bands = [
        (NY_OPEN_ET, NY_OPEN_ET + 30, "10:00-10:30 (first 30 min)"),
        (NY_OPEN_ET + 30, 12 * 60, "10:30-12:00 (mid-morning)"),
        (12 * 60, NY_CLOSE_ET, "12:00-16:00 (afternoon)"),
    ]
    for lo_et, hi_et, label in tod_bands:
        pnls = [t['pnl_dollar'] for t in valid_trades
                if lo_et <= t['entry_et_mins'] < hi_et]
        stats = compute_group_stats(pnls)
        print_table_row(label, stats, baseline_wr)

    # Cross-cut: time-of-day x distance
    print("\n  Time-of-day x Distance (near = <10pts):")
    for lo_et, hi_et, tod_label in tod_bands:
        tod_trades = [t for t in valid_trades if lo_et <= t['entry_et_mins'] < hi_et]
        if not tod_trades:
            continue
        pnls_near = [t['pnl_dollar'] for t in tod_trades if t['dvpoc_abs_dist'] < 10]
        pnls_far = [t['pnl_dollar'] for t in tod_trades if t['dvpoc_abs_dist'] >= 10]
        stats_near = compute_group_stats(pnls_near)
        stats_far = compute_group_stats(pnls_far)
        print(f"\n    {tod_label}:")
        print_table_row(f"      Near dVPOC (<10pts)", stats_near, baseline_wr)
        print_table_row(f"      Far from dVPOC (>=10pts)", stats_far, baseline_wr)

    # ==================================================================
    # TIER 3: VCR REGIME + WEEKLY VPOC INTERACTION
    # ==================================================================
    print(f"\n{'='*80}")
    print("TIER 3: VCR REGIME + WEEKLY VPOC INTERACTION")
    print(f"{'='*80}")

    # --- 7. VCR regime (quartile split) ---
    print(f"\n--- 7. VCR Regime (Volume Concentration Ratio) ---")
    print("  High VCR = strong concentration (trending). Low VCR = dispersed (rotational).")
    vcr_valid = [t for t in valid_trades if not np.isnan(t['dvpoc_strength'])]
    vcr_values = [t['dvpoc_strength'] for t in vcr_valid]

    if vcr_values:
        q25, q50, q75 = np.percentile(vcr_values, [25, 50, 75])
        print(f"  VCR quartiles: Q1={q25:.3f}, Q2={q50:.3f}, Q3={q75:.3f}")

        vcr_quartiles = [
            (0, q25, f"Q1 (VCR < {q25:.3f}) — dispersed"),
            (q25, q50, f"Q2 (VCR {q25:.3f}-{q50:.3f})"),
            (q50, q75, f"Q3 (VCR {q50:.3f}-{q75:.3f})"),
            (q75, 1.01, f"Q4 (VCR > {q75:.3f}) — concentrated"),
        ]
        for lo, hi, label in vcr_quartiles:
            pnls = [t['pnl_dollar'] for t in vcr_valid
                    if lo <= t['dvpoc_strength'] < hi]
            stats = compute_group_stats(pnls)
            print_table_row(label, stats, baseline_wr)

    # --- 8. dPOC vs Weekly VPOC interaction ---
    print(f"\n--- 8. dVPOC + Weekly VPOC Confluence ---")
    print("  Are dVPOC and weekly VPOC additive when both are near entry?")

    for inst in ['MNQ', 'MES']:
        inst_trades = [t for t in valid_trades if t['instrument'] == inst]
        if not inst_trades:
            continue
        inst_wins = sum(1 for t in inst_trades if t['is_win'])
        inst_wr = inst_wins / len(inst_trades) * 100
        print(f"\n  [{inst}] ({len(inst_trades)} trades, WR {inst_wr:.1f}%):")

        # Both near (dVPOC <10pts AND weekly VPOC <10pts)
        both_near = [t for t in inst_trades
                     if t['dvpoc_abs_dist'] < 10
                     and not np.isnan(t['weekly_vpoc'])
                     and abs(t['entry_price'] - t['weekly_vpoc']) < 10]
        pnls = [t['pnl_dollar'] for t in both_near]
        stats = compute_group_stats(pnls)
        print_table_row(f"  Both dVPOC + wVPOC <10pts", stats, inst_wr)

        # Only dVPOC near
        only_dpoc = [t for t in inst_trades
                     if t['dvpoc_abs_dist'] < 10
                     and (np.isnan(t['weekly_vpoc'])
                          or abs(t['entry_price'] - t['weekly_vpoc']) >= 10)]
        pnls = [t['pnl_dollar'] for t in only_dpoc]
        stats = compute_group_stats(pnls)
        print_table_row(f"  Only dVPOC <10pts (wVPOC far)", stats, inst_wr)

        # Only weekly VPOC near
        only_wvpoc = [t for t in inst_trades
                      if t['dvpoc_abs_dist'] >= 10
                      and not np.isnan(t['weekly_vpoc'])
                      and abs(t['entry_price'] - t['weekly_vpoc']) < 10]
        pnls = [t['pnl_dollar'] for t in only_wvpoc]
        stats = compute_group_stats(pnls)
        print_table_row(f"  Only wVPOC <10pts (dVPOC far)", stats, inst_wr)

        # Neither near
        neither = [t for t in inst_trades
                   if t['dvpoc_abs_dist'] >= 10
                   and (np.isnan(t['weekly_vpoc'])
                        or abs(t['entry_price'] - t['weekly_vpoc']) >= 10)]
        pnls = [t['pnl_dollar'] for t in neither]
        stats = compute_group_stats(pnls)
        print_table_row(f"  Neither near (<10pts)", stats, inst_wr)

    # ==================================================================
    # DELIVERABLE 4: PRIOR-DAY VPOC FOR MNQ
    # ==================================================================
    print(f"\n{'='*80}")
    print("DELIVERABLE 4: PRIOR-DAY VPOC PROXIMITY")
    print(f"{'='*80}")

    prior_bands = [(0, 5), (5, 7), (7, 10), (10, 9999)]
    prior_labels = ["0-5 pts", "5-7 pts", "7-10 pts", "10+ pts"]

    for inst in ['MNQ', 'MES']:
        inst_trades = [t for t in mapped_trades
                       if t['instrument'] == inst
                       and not np.isnan(t.get('prior_vpoc_dist', np.nan))]
        if not inst_trades:
            print(f"\n  [{inst}] No trades with prior-day VPOC data")
            continue

        inst_wins = sum(1 for t in inst_trades if t['is_win'])
        inst_wr = inst_wins / len(inst_trades) * 100

        gp = sum(t['pnl_dollar'] for t in inst_trades if t['pnl_dollar'] > 0)
        gl = abs(sum(t['pnl_dollar'] for t in inst_trades if t['pnl_dollar'] <= 0))
        inst_pf = gp / gl if gl > 0 else 999.0

        print(f"\n  [{inst}] Prior-Day VPOC ({len(inst_trades)} trades, "
              f"WR {inst_wr:.1f}%, PF {inst_pf:.3f}):")

        for (lo, hi), label in zip(prior_bands, prior_labels):
            pnls = [t['pnl_dollar'] for t in inst_trades
                    if lo <= t['prior_vpoc_dist'] < hi]
            stats = compute_group_stats(pnls)
            print_table_row(f"  Prior-day VPOC dist {label}", stats, inst_wr)

        # Per-strategy breakdown within instrument
        inst_strats = set(t['strategy'] for t in inst_trades)
        for strat in sorted(inst_strats):
            strat_trades = [t for t in inst_trades if t['strategy'] == strat]
            if len(strat_trades) < 5:
                continue
            s_wins = sum(1 for t in strat_trades if t['is_win'])
            s_wr = s_wins / len(strat_trades) * 100
            print(f"\n    [{strat}] ({len(strat_trades)} trades, WR {s_wr:.1f}%):")
            for (lo, hi), label in zip(prior_bands, prior_labels):
                pnls = [t['pnl_dollar'] for t in strat_trades
                        if lo <= t['prior_vpoc_dist'] < hi]
                stats = compute_group_stats(pnls)
                if stats and stats['count'] >= 3:
                    print_table_row(f"      {label}", stats, s_wr)

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*80}")
    print("SUMMARY OF KEY FINDINGS")
    print(f"{'='*80}")

    # Find the most interesting distance band
    best_band = None
    best_delta = -999
    for (lo, hi), label in zip(distance_bands, band_labels):
        pnls = [t['pnl_dollar'] for t in valid_trades
                if lo <= t['dvpoc_abs_dist'] < hi]
        stats = compute_group_stats(pnls)
        if stats and stats['count'] >= 10:
            delta = stats['wr'] - baseline_wr
            if delta > best_delta:
                best_delta = delta
                best_band = (label, stats)

    if best_band:
        label, stats = best_band
        print(f"\n  Best distance band: {label} "
              f"({stats['count']} trades, WR {stats['wr']:.1f}%, "
              f"PF {stats['pf']:.3f}, delta {best_delta:+.1f}pp)")

    # Stability finding
    if stats_stable and stats_unstable:
        stab_delta = stats_stable['wr'] - stats_unstable['wr']
        print(f"\n  Stability effect: Stable dVPOC WR {stats_stable['wr']:.1f}% vs "
              f"Unstable {stats_unstable['wr']:.1f}% "
              f"(delta {stab_delta:+.1f}pp, N={stats_stable['count']}+{stats_unstable['count']})")

    # VCR finding
    if vcr_values:
        q4_pnls = [t['pnl_dollar'] for t in vcr_valid if t['dvpoc_strength'] >= q75]
        q1_pnls = [t['pnl_dollar'] for t in vcr_valid if t['dvpoc_strength'] < q25]
        q4_stats = compute_group_stats(q4_pnls)
        q1_stats = compute_group_stats(q1_pnls)
        if q4_stats and q1_stats:
            vcr_delta = q4_stats['wr'] - q1_stats['wr']
            print(f"\n  VCR regime: Q4 (concentrated) WR {q4_stats['wr']:.1f}% vs "
                  f"Q1 (dispersed) {q1_stats['wr']:.1f}% "
                  f"(delta {vcr_delta:+.1f}pp)")

    # Prior-day VPOC finding for MNQ
    mnq_prior = [t for t in mapped_trades
                 if t['instrument'] == 'MNQ'
                 and not np.isnan(t.get('prior_vpoc_dist', np.nan))]
    if mnq_prior:
        near_5_pnls = [t['pnl_dollar'] for t in mnq_prior if t['prior_vpoc_dist'] < 5]
        far_pnls = [t['pnl_dollar'] for t in mnq_prior if t['prior_vpoc_dist'] >= 10]
        near_stats = compute_group_stats(near_5_pnls)
        far_stats = compute_group_stats(far_pnls)
        if near_stats and far_stats:
            print(f"\n  MNQ Prior-Day VPOC: Near (<5pts) WR {near_stats['wr']:.1f}%, "
                  f"PF {near_stats['pf']:.3f} vs "
                  f"Far (10+pts) WR {far_stats['wr']:.1f}%, PF {far_stats['pf']:.3f}")

    print(f"\n{'='*80}")
    print("Developing VPOC Forensics complete.")
    print(f"{'='*80}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    portfolio_data = run_portfolio_trades()
    run_analysis(portfolio_data)
