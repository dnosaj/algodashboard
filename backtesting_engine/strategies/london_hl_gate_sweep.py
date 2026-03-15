"""
London H/L Entry Gate Sweep
============================
Tests blocking entries when price is within `buffer_pts` of London session
H/L (02:00-05:00 ET) as an entry gate for the MNQ strategies.

Gate logic: block entry when close[i-1] is within `buffer_pts` of EITHER
London HIGH or London LOW. This is undirectional — the directional
decomposition in ict_forensics.py tells us where the damage is concentrated.

Sweep: buffer_pts = [3, 5, 7, 10]

Strategies:
  vScalpA:  London gate ON  + existing gates (leledc+adr+vix)
  vScalpB:  London gate OFF + existing gates (leledc+adr) — ungated by design
  vScalpC:  London gate ON  + existing gates (leledc+adr+atr+vix)
  MES v2:   London gate OFF + existing gate  (prior_day_level VPOC+VAL, buf=5)

IS/OOS: first half / second half split.

Look-ahead check: London session levels use completed sessions only (02:00-
05:00 ET must finish before the level is available). All entries are during
RTH (09:30+), so London is always completed. No look-ahead bias.

Usage:
    cd backtesting_engine/strategies
    python3 london_hl_gate_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    compute_prior_day_levels,
)

from generate_session import (
    load_instrument_1min,
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

# Gate functions
from sr_leledc_exhaustion_sweep import compute_leledc_exhaustion, build_leledc_gate
from adr_common import compute_session_tracking, compute_adr, build_directional_gate
from htf_common import compute_prior_day_atr
from sr_prior_day_levels_sweep import compute_rth_volume_profile, build_prior_day_level_gate

# Structure exit
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

# Session levels from ict_forensics
from ict_forensics import compute_session_levels


# ---------------------------------------------------------------------------
# London H/L gate construction
# ---------------------------------------------------------------------------

def build_london_hl_gate(closes, london_high, london_low, buffer_pts):
    """Build boolean gate: True = allow entry, False = block.

    Blocks when close is within `buffer_pts` of EITHER London HIGH or
    London LOW. Undirectional gate.

    Args:
        closes: close price array
        london_high: London session high array (NaN where unavailable)
        london_low: London session low array (NaN where unavailable)
        buffer_pts: proximity threshold in points

    Returns:
        Boolean numpy array (True = allow entry, False = block).
    """
    n = len(closes)
    gate = np.ones(n, dtype=bool)

    for i in range(n):
        c = closes[i]
        lh = london_high[i]
        ll = london_low[i]

        if not np.isnan(lh) and abs(c - lh) <= buffer_pts:
            gate[i] = False
            continue
        if not np.isnan(ll) and abs(c - ll) <= buffer_pts:
            gate[i] = False

    return gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice(arr, split_name, is_len):
    """Slice array for IS/OOS splits."""
    if arr is None:
        return None
    if split_name == "FULL":
        return arr
    elif split_name == "IS":
        return arr[:is_len]
    else:
        return arr[is_len:]


def score_fmt(sc, label, baseline_sc=None):
    """Format score dict as string with optional delta vs baseline."""
    if sc is None:
        return f"  {label}: NO TRADES"
    s = (f"  {label}: {sc['count']} trades, WR {sc['win_rate']:.1f}%, "
         f"PF {sc['pf']:.3f}, Sharpe {sc['sharpe']:.3f}, "
         f"P&L ${sc['net_dollar']:+,.2f}")
    if baseline_sc is not None and baseline_sc.get('pf', 0) > 0:
        dpf = (sc['pf'] - baseline_sc['pf']) / baseline_sc['pf'] * 100
        s += f"  (dPF {dpf:+.1f}%)"
    return s


def assess_verdict(sc_is, sc_oos, bl_is, bl_oos):
    """Simple pass/fail: IS PF > 1.0, OOS PF > IS PF * 0.8, count > 50."""
    if sc_is is None or sc_oos is None:
        return "FAIL", "no trades"
    if sc_is['count'] < 50:
        return "FAIL", f"IS count {sc_is['count']} < 50"
    if sc_oos['count'] < 50:
        return "FAIL", f"OOS count {sc_oos['count']} < 50"
    if sc_is['pf'] <= 1.0:
        return "FAIL", f"IS PF {sc_is['pf']:.3f} <= 1.0"
    if sc_oos['pf'] < sc_is['pf'] * 0.8:
        return "FAIL", f"OOS PF {sc_oos['pf']:.3f} < IS*0.8 ({sc_is['pf']*0.8:.3f})"

    # Check improvement vs baseline
    if bl_is is not None and bl_oos is not None:
        is_dpf = (sc_is['pf'] - bl_is['pf']) / bl_is['pf'] * 100 if bl_is['pf'] > 0 else 0
        oos_dpf = (sc_oos['pf'] - bl_oos['pf']) / bl_oos['pf'] * 100 if bl_oos['pf'] > 0 else 0
        if is_dpf >= 5 and oos_dpf >= 0:
            return "STRONG PASS", f"IS dPF {is_dpf:+.1f}%, OOS dPF {oos_dpf:+.1f}%"
        if is_dpf >= 0 and oos_dpf >= 0:
            return "MARGINAL PASS", f"IS dPF {is_dpf:+.1f}%, OOS dPF {oos_dpf:+.1f}%"
        if is_dpf < -5 or oos_dpf < -5:
            return "FAIL", f"PF degrades (IS {is_dpf:+.1f}%, OOS {oos_dpf:+.1f}%)"

    return "PASS", "meets thresholds"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_lines = []

    def tee(s=""):
        print(s)
        output_lines.append(s)

    tee("=" * 80)
    tee("LONDON H/L GATE SWEEP")
    tee("  Gate: block when close within buffer_pts of London HIGH or LOW")
    tee("  London session: 02:00-05:00 ET (completed before RTH)")
    tee("  Sweep: buffer_pts = [3, 5, 7, 10]")
    tee("  Applied to: vScalpA + vScalpC (stacked on existing gates)")
    tee("  NOT applied to: vScalpB (ungated) + MES v2 (uses prior-day levels)")
    tee("=" * 80)

    # --- Load MNQ ---
    tee("\nLoading MNQ data...")
    df_mnq = load_instrument_1min("MNQ")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm
    tee(f"  MNQ: {len(df_mnq)} bars, {df_mnq.index[0]} to {df_mnq.index[-1]}")

    # --- Load MES ---
    tee("\nLoading MES data...")
    df_mes = load_instrument_1min("MES")
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm
    tee(f"  MES: {len(df_mes)} bars, {df_mes.index[0]} to {df_mes.index[-1]}")

    # --- Compute existing gates ---
    tee("\n--- Computing Existing Entry Gates ---")

    mnq_closes = df_mnq["Close"].values
    mnq_highs = df_mnq["High"].values
    mnq_lows = df_mnq["Low"].values
    mnq_sm_arr = df_mnq["SM_Net"].values
    start_date = df_mnq.index[0].strftime("%Y-%m-%d")
    end_date = df_mnq.index[-1].strftime("%Y-%m-%d")

    # Leledc
    tee("  Computing Leledc exhaustion (mq=9)...")
    bull_ex, bear_ex = compute_leledc_exhaustion(mnq_closes, maj_qual=9)
    mnq_leledc_gate = build_leledc_gate(bull_ex, bear_ex, persistence=1)

    # ADR directional
    tee("  Computing ADR directional gate (14d, 0.3)...")
    mnq_session = compute_session_tracking(df_mnq)
    mnq_adr = compute_adr(df_mnq, lookback_days=14)
    mnq_adr_gate = build_directional_gate(
        mnq_session['move_from_open'], mnq_adr, mnq_sm_arr, threshold=0.3)

    # ATR gate (vScalpC only)
    tee("  Computing prior-day ATR gate (min=263.8)...")
    mnq_prior_atr = compute_prior_day_atr(df_mnq, lookback_days=14)
    mnq_atr_gate = np.ones(len(df_mnq), dtype=bool)
    for i in range(len(mnq_prior_atr)):
        if not np.isnan(mnq_prior_atr[i]) and mnq_prior_atr[i] < 263.8:
            mnq_atr_gate[i] = False

    # VIX death zone
    tee("  Computing VIX death zone gate (19-22)...")
    mnq_bar_dates_et = _get_bar_dates_et(df_mnq.index)
    mnq_vix_gate = load_vix_gate(start_date, end_date, mnq_bar_dates_et,
                                  low=19.0, high=22.0)

    # MES gates
    tee("  Computing MES prior-day level gate (VPOC+VAL, buf=5)...")
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

    # --- Existing composite gates (WITHOUT London) ---
    base_gate_vscalpa = mnq_leledc_gate & mnq_adr_gate & mnq_vix_gate
    base_gate_vscalpb = mnq_leledc_gate & mnq_adr_gate
    base_gate_vscalpc = mnq_leledc_gate & mnq_adr_gate & mnq_atr_gate & mnq_vix_gate
    gate_mesv2 = mes_level_gate

    # --- Compute London session levels ---
    tee("\n--- Computing London Session Levels ---")
    mnq_sessions = compute_session_levels(df_mnq)
    london_high = mnq_sessions['london_high']
    london_low = mnq_sessions['london_low']
    n_with_levels = np.sum(~np.isnan(london_high))
    tee(f"  {n_with_levels}/{len(df_mnq)} bars have London H/L levels")

    # --- Swing levels for vScalpC ---
    tee("  Computing swing levels for vScalpC structure exit...")
    mnq_swing_highs, mnq_swing_lows = compute_swing_levels(
        mnq_highs, mnq_lows,
        lookback=VSCALPC_SWING_LB, swing_type="pivot",
        pivot_right=VSCALPC_SWING_PR)

    # --- IS/OOS split ---
    mnq_is_len = len(df_mnq) // 2
    mes_is_len = len(df_mes) // 2

    # --- Prepare RSI ---
    mnq_opens = df_mnq["Open"].values
    mnq_times = df_mnq.index

    mes_opens = df_mes["Open"].values
    mes_sm_arr = df_mes["SM_Net"].values

    # ===================================================================
    # Sweep buffers
    # ===================================================================
    BUFFER_PTS_LIST = [None, 3, 5, 7, 10]

    # Storage for all results
    all_results = {}  # (buffer, strat, split) -> sc
    all_trades = {}   # (buffer, strat, split) -> trades
    baselines = {}    # (strat, split) -> sc

    for buffer_pts in BUFFER_PTS_LIST:
        is_baseline = buffer_pts is None
        label = "None (baseline)" if is_baseline else f"buf={buffer_pts}"

        tee(f"\n{'='*80}")
        tee(f"Buffer: {label}")
        tee(f"{'='*80}")

        # --- Build London gate ---
        if is_baseline:
            london_gate = None
        else:
            london_gate = build_london_hl_gate(
                mnq_closes, london_high, london_low, buffer_pts)
            blocked = (~london_gate).sum()
            tee(f"  London H/L gate (buf={buffer_pts}): blocks {blocked}/{len(london_gate)} bars "
                f"({blocked/len(london_gate)*100:.1f}%)")

        # --- Composite gates with London ---
        if london_gate is not None:
            gate_vscalpa = base_gate_vscalpa & london_gate
            gate_vscalpc = base_gate_vscalpc & london_gate
        else:
            gate_vscalpa = base_gate_vscalpa
            gate_vscalpc = base_gate_vscalpc

        gate_vscalpb = base_gate_vscalpb  # Never gets London gate
        # gate_mesv2 already set above, never gets London gate

        for split_name in ["FULL", "IS", "OOS"]:
            tee(f"\n  --- {split_name} ---")

            # Slice data
            mnq_data = df_mnq if split_name == "FULL" else (
                df_mnq.iloc[:mnq_is_len] if split_name == "IS" else df_mnq.iloc[mnq_is_len:])
            mes_data = df_mes if split_name == "FULL" else (
                df_mes.iloc[:mes_is_len] if split_name == "IS" else df_mes.iloc[mes_is_len:])

            # Slice gates
            g_a = _slice(gate_vscalpa, split_name, mnq_is_len)
            g_b = _slice(gate_vscalpb, split_name, mnq_is_len)
            g_c = _slice(gate_vscalpc, split_name, mnq_is_len)
            g_m = _slice(gate_mesv2, split_name, mes_is_len)

            # Slice swing levels
            sw_h = _slice(mnq_swing_highs, split_name, mnq_is_len)
            sw_l = _slice(mnq_swing_lows, split_name, mnq_is_len)

            # MNQ arrays
            m_opens = mnq_data["Open"].values
            m_highs = mnq_data["High"].values
            m_lows = mnq_data["Low"].values
            m_closes = mnq_data["Close"].values
            m_sm = mnq_data["SM_Net"].values
            m_times = mnq_data.index

            # MNQ RSI
            df_mnq_5m = resample_to_5min(mnq_data)
            rsi_mnq_curr, rsi_mnq_prev = map_5min_rsi_to_1min(
                mnq_data.index.values, df_mnq_5m.index.values,
                df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN)

            # === vScalpA ===
            trades_a = run_backtest_tp_exit(
                m_opens, m_highs, m_lows, m_closes, m_sm, m_times,
                rsi_mnq_curr, rsi_mnq_prev,
                rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
                sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
                max_loss_pts=VSCALPA_MAX_LOSS_PTS, tp_pts=VSCALPA_TP_PTS,
                entry_end_et=VSCALPA_ENTRY_END_ET, entry_gate=g_a,
            )
            compute_mfe_mae(trades_a, m_highs, m_lows)
            sc_a = score_trades(trades_a, commission_per_side=MNQ_COMMISSION,
                                dollar_per_pt=MNQ_DOLLAR_PER_PT)
            all_results[(buffer_pts, "vScalpA", split_name)] = sc_a
            all_trades[(buffer_pts, "vScalpA", split_name)] = trades_a

            # === vScalpB (NO London gate) ===
            trades_b = run_backtest_tp_exit(
                m_opens, m_highs, m_lows, m_closes, m_sm, m_times,
                rsi_mnq_curr, rsi_mnq_prev,
                rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
                sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
                max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
                entry_gate=g_b,
            )
            compute_mfe_mae(trades_b, m_highs, m_lows)
            sc_b = score_trades(trades_b, commission_per_side=MNQ_COMMISSION,
                                dollar_per_pt=MNQ_DOLLAR_PER_PT)
            all_results[(buffer_pts, "vScalpB", split_name)] = sc_b
            all_trades[(buffer_pts, "vScalpB", split_name)] = trades_b

            # === vScalpC (structure exit, with London gate) ===
            trades_c = run_backtest_structure_exit(
                m_opens, m_highs, m_lows, m_closes, m_sm, m_times,
                rsi_mnq_curr, rsi_mnq_prev,
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
                entry_gate=g_c,
            )
            sc_c = score_structure_trades(trades_c, dollar_per_pt=MNQ_DOLLAR_PER_PT,
                                          commission_per_side=MNQ_COMMISSION)
            all_results[(buffer_pts, "vScalpC", split_name)] = sc_c
            all_trades[(buffer_pts, "vScalpC", split_name)] = trades_c

            # === MES v2 (NO London gate) ===
            s_opens = mes_data["Open"].values
            s_highs = mes_data["High"].values
            s_lows = mes_data["Low"].values
            s_closes = mes_data["Close"].values
            s_sm = mes_data["SM_Net"].values
            s_times = mes_data.index

            df_mes_5m = resample_to_5min(mes_data)
            rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
                mes_data.index.values, df_mes_5m.index.values,
                df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN)

            trades_m = run_backtest_tp_exit(
                s_opens, s_highs, s_lows, s_closes, s_sm, s_times,
                rsi_mes_curr, rsi_mes_prev,
                rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
                sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
                max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
                eod_minutes_et=MESV2_EOD_ET,
                entry_end_et=MESV2_ENTRY_END_ET,
                breakeven_after_bars=MESV2_BREAKEVEN_BARS,
                entry_gate=g_m,
            )
            compute_mfe_mae(trades_m, s_highs, s_lows)
            sc_m = score_trades(trades_m, commission_per_side=MES_COMMISSION,
                                dollar_per_pt=MES_DOLLAR_PER_PT)
            all_results[(buffer_pts, "MES_v2", split_name)] = sc_m
            all_trades[(buffer_pts, "MES_v2", split_name)] = trades_m

            # Print split summary
            for sname, sc in [("vScalpA", sc_a), ("vScalpB", sc_b),
                               ("vScalpC", sc_c), ("MES_v2", sc_m)]:
                if sc is None:
                    tee(f"    {sname:>8}: NO TRADES")
                    continue
                ct = sc['count']
                wr = sc['win_rate'] if 'win_rate' in sc else sc.get('wr', 0)
                pf = sc['pf']
                net = sc['net_dollar'] if 'net_dollar' in sc else sc.get('total_pnl', 0)
                sharpe = sc.get('sharpe', 0)
                tee(f"    {sname:>8}: {ct:>4d} trades, WR {wr:>5.1f}%, "
                    f"PF {pf:>6.3f}, Sharpe {sharpe:>6.3f}, "
                    f"P&L ${net:>+9,.2f}")

        # Store baselines
        if is_baseline:
            for strat in ["vScalpA", "vScalpB", "vScalpC", "MES_v2"]:
                for sp in ["FULL", "IS", "OOS"]:
                    baselines[(strat, sp)] = all_results[(None, strat, sp)]

    # ===================================================================
    # Summary table
    # ===================================================================
    tee(f"\n\n{'='*120}")
    tee("LONDON H/L GATE SWEEP — SUMMARY TABLE")
    tee(f"{'='*120}")

    STRATS = ["vScalpA", "vScalpB", "vScalpC", "MES_v2"]

    for strat in STRATS:
        london_applied = strat in ("vScalpA", "vScalpC")
        gate_label = "LONDON GATE APPLIED" if london_applied else "NO LONDON GATE"
        tee(f"\n{strat} ({gate_label}):")
        tee(f"  {'Buffer':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
            f"{'Sharpe':>7} {'Net$':>10} {'dPF%':>7} {'Blocked':>8}")
        tee(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*10} {'-'*7} {'-'*8}")

        for buffer_pts in BUFFER_PTS_LIST:
            label = "None" if buffer_pts is None else f"{buffer_pts}"
            for sp in ["FULL", "IS", "OOS"]:
                sc = all_results.get((buffer_pts, strat, sp))
                bl = baselines.get((strat, sp))
                if sc is None:
                    tee(f"  {label:>8} {sp:>5}   NO TRADES")
                    continue

                ct = sc['count']
                wr = sc.get('win_rate', sc.get('wr', 0))
                pf = sc['pf']
                sharpe = sc.get('sharpe', 0)
                net = sc.get('net_dollar', sc.get('total_pnl', 0))
                bl_count = bl['count'] if bl else ct
                blocked = bl_count - ct
                dpf = 0.0
                if bl and bl['pf'] > 0:
                    dpf = (pf - bl['pf']) / bl['pf'] * 100

                tee(f"  {label:>8} {sp:>5} {ct:>7} {wr:>5.1f}% {pf:>7.3f} "
                    f"{sharpe:>7.3f} {net:>+10,.2f} {dpf:>+6.1f}% {blocked:>8}")

    # ===================================================================
    # Per-strategy verdicts
    # ===================================================================
    tee(f"\n\n{'='*80}")
    tee("PER-STRATEGY VERDICTS")
    tee(f"{'='*80}")

    for buffer_pts in [3, 5, 7, 10]:
        tee(f"\n  Buffer = {buffer_pts} pts:")
        for strat in STRATS:
            london_applied = strat in ("vScalpA", "vScalpC")
            if not london_applied:
                tee(f"    {strat:>8}: N/A (no London gate)")
                continue

            sc_is = all_results.get((buffer_pts, strat, "IS"))
            sc_oos = all_results.get((buffer_pts, strat, "OOS"))
            bl_is = baselines.get((strat, "IS"))
            bl_oos = baselines.get((strat, "OOS"))

            verdict, detail = assess_verdict(sc_is, sc_oos, bl_is, bl_oos)
            tee(f"    {strat:>8}: {verdict} — {detail}")

    # ===================================================================
    # Portfolio aggregate
    # ===================================================================
    tee(f"\n\n{'='*80}")
    tee("PORTFOLIO AGGREGATE (A+B+C+MES)")
    tee(f"{'='*80}")

    tee(f"  {'Buffer':>8} {'Split':>5} {'TotalNet$':>12} {'PortSharpe':>11}")
    tee(f"  {'-'*8} {'-'*5} {'-'*12} {'-'*11}")

    for buffer_pts in BUFFER_PTS_LIST:
        label = "None" if buffer_pts is None else f"{buffer_pts}"
        for sp in ["FULL", "IS", "OOS"]:
            total_net = 0
            all_pnls = []
            valid = True

            for strat in STRATS:
                sc = all_results.get((buffer_pts, strat, sp))
                if sc is None:
                    valid = False
                    break
                net = sc.get('net_dollar', sc.get('total_pnl', 0))
                total_net += net

                # Get per-trade pnls for portfolio Sharpe
                trades = all_trades.get((buffer_pts, strat, sp), [])
                if strat == "MES_v2":
                    dpp, comm = MES_DOLLAR_PER_PT, MES_COMMISSION
                else:
                    dpp, comm = MNQ_DOLLAR_PER_PT, MNQ_COMMISSION

                if strat == "vScalpC":
                    # Structure trades: group by entry_idx
                    entries = {}
                    for t in trades:
                        eidx = t.get("entry_idx", 0)
                        if eidx not in entries:
                            entries[eidx] = 0.0
                        entries[eidx] += t["pts"] * dpp - comm * 2
                    all_pnls.extend(entries.values())
                else:
                    for t in trades:
                        all_pnls.append(t["pts"] * dpp - comm * 2)

            if not valid:
                tee(f"  {label:>8} {sp:>5}   incomplete")
                continue

            pnl_arr = np.array(all_pnls)
            if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
                port_sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252))
            else:
                port_sharpe = 0.0

            tee(f"  {label:>8} {sp:>5} {total_net:>+12,.2f} {port_sharpe:>11.3f}")

    # ===================================================================
    # Gate blocking stats
    # ===================================================================
    tee(f"\n\n{'='*80}")
    tee("GATE BLOCKING ANALYSIS")
    tee(f"{'='*80}")

    for buffer_pts in [3, 5, 7, 10]:
        bl_a_full = baselines.get(("vScalpA", "FULL"))
        sc_a_full = all_results.get((buffer_pts, "vScalpA", "FULL"))
        bl_c_full = baselines.get(("vScalpC", "FULL"))
        sc_c_full = all_results.get((buffer_pts, "vScalpC", "FULL"))

        if bl_a_full and sc_a_full:
            bl_ct = bl_a_full['count']
            sc_ct = sc_a_full['count']
            blocked = bl_ct - sc_ct
            pct = blocked / bl_ct * 100 if bl_ct > 0 else 0
            tee(f"  Buffer {buffer_pts}: vScalpA blocks {blocked}/{bl_ct} entries ({pct:.1f}%)")

        if bl_c_full and sc_c_full:
            bl_ct = bl_c_full['count']
            sc_ct = sc_c_full['count']
            blocked = bl_ct - sc_ct
            pct = blocked / bl_ct * 100 if bl_ct > 0 else 0
            tee(f"  Buffer {buffer_pts}: vScalpC blocks {blocked}/{bl_ct} entries ({pct:.1f}%)")

    # --- Save output ---
    output_path = _STRAT_DIR.parent / "results" / "london_hl_gate_sweep_output.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    tee(f"\nOutput saved to: {output_path}")
    tee("Done.")


if __name__ == "__main__":
    main()
