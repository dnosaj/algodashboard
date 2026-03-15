"""
RSI Trendline Breakout — IS/OOS Scalp Analysis + Portfolio Correlation
======================================================================
Task 1: IS/OOS on two specific scalp configs
Task 2: Portfolio correlation analysis (daily P&L, temporal overlap, combined portfolio)

Usage:
    python3 rsi_tl_correlation_analysis.py
"""

import warnings
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")
warnings.filterwarnings("ignore", category=FutureWarning)

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))

# Import RSI trendline backtest functions
from rsi_trendline_backtest import (
    load_mnq_data, compute_rsi, generate_signals,
    run_backtest, run_backtest_runner, score_trades, split_is_oos,
    COMMISSION_PER_SIDE, DOLLAR_PER_PT, NY_OPEN_ET,
)

# Import existing portfolio strategy infrastructure
from v10_test_common import (
    compute_rsi as compute_rsi_v10,
    compute_smart_money,
    compute_et_minutes,
    compute_prior_day_levels,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades as score_trades_v10,
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

# Gate imports
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

# Production params from run_and_save_portfolio.py
VSCALPC_TP1_PTS = 7
VSCALPC_MAX_TP2_PTS = 60
VSCALPC_SL_PTS = 40
VSCALPC_SWING_LB = 50
VSCALPC_SWING_PR = 2
VSCALPC_SWING_BUF = 2.0
MESV2_EOD_ET = 16 * 60       # 16:00 ET
MESV2_ENTRY_END_ET = 14 * 60 + 15  # 14:15 ET

_ET_TZ = ZoneInfo("America/New_York")


# =========================================================================
# Helper: VIX gate
# =========================================================================

def _get_bar_dates_et(times):
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    return idx.tz_convert(_ET_TZ).date


def load_vix_gate(start_date, end_date, bar_dates_et, low=19.0, high=22.0):
    n = len(bar_dates_et)
    gate = np.ones(n, dtype=bool)
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if vix.empty:
            print("  WARNING: VIX download returned empty — fail-open")
            return gate
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix.reset_index()
        vix["date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.date
        vix_close_by_date = dict(zip(vix["date"], vix["Close"].values))
        sorted_vix_dates = sorted(vix_close_by_date.keys())
        prior_close = {}
        for i in range(1, len(sorted_vix_dates)):
            prior_close[sorted_vix_dates[i]] = vix_close_by_date[sorted_vix_dates[i - 1]]
        for i in range(n):
            d = bar_dates_et[i]
            if d in prior_close:
                pc = float(prior_close[d])
                if low <= pc <= high:
                    gate[i] = False
        blocked = (~gate).sum()
        print(f"  VIX death zone [{low}-{high}]: blocks {blocked}/{n} bars ({blocked/n*100:.1f}%)")
    except Exception as e:
        print(f"  WARNING: VIX download failed ({e}) — fail-open")
    return gate


# =========================================================================
# Helper: trades to daily P&L series
# =========================================================================

def trades_to_daily_pnl(trades, dollar_per_pt, commission_per_side, label,
                         is_structure=False, is_runner_rsi_tl=False):
    """Convert trade list to daily P&L Series (indexed by date)."""
    if not trades:
        return pd.Series(dtype=float, name=label)

    if is_structure:
        comm = commission_per_side * 2
        entries = {}
        for t in trades:
            eidx = t["entry_idx"]
            if eidx not in entries:
                entries[eidx] = {"pnl": 0.0, "time": t["entry_time"]}
            entries[eidx]["pnl"] += t["pts"] * dollar_per_pt - comm
        rows = []
        for eidx in sorted(entries.keys()):
            entry_ts = pd.Timestamp(entries[eidx]["time"])
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_et = entry_ts.tz_convert("America/New_York")
            rows.append({"trade_date": entry_et.strftime("%Y-%m-%d"),
                         "pnl": entries[eidx]["pnl"]})
    elif is_runner_rsi_tl:
        # RSI TL runner: trades have qty field
        comm_pts = (commission_per_side * 2) / dollar_per_pt
        rows = []
        for t in trades:
            qty = t.get("qty", 1)
            pnl = (t["pts"] - comm_pts) * dollar_per_pt * qty
            entry_ts = pd.Timestamp(t["entry_time"])
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_et = entry_ts.tz_convert("America/New_York")
            rows.append({"trade_date": entry_et.strftime("%Y-%m-%d"),
                         "pnl": pnl})
    else:
        comm_pts = (commission_per_side * 2) / dollar_per_pt
        rows = []
        for t in trades:
            pnl = (t["pts"] - comm_pts) * dollar_per_pt
            entry_ts = pd.Timestamp(t["entry_time"])
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_et = entry_ts.tz_convert("America/New_York")
            rows.append({"trade_date": entry_et.strftime("%Y-%m-%d"),
                         "pnl": pnl})

    if not rows:
        return pd.Series(dtype=float, name=label)
    df = pd.DataFrame(rows)
    daily = df.groupby("trade_date")["pnl"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily.name = label
    return daily


def portfolio_stats(daily_series, label="Portfolio"):
    """Compute Sharpe, PF, MaxDD, net P&L from a daily P&L series."""
    if daily_series.empty or len(daily_series) < 2:
        return None
    total = daily_series.sum()
    mean_d = daily_series.mean()
    std_d = daily_series.std()
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0
    cum = daily_series.cumsum()
    mdd = (cum - cum.cummax()).min()
    gp = daily_series[daily_series > 0].sum()
    gl = abs(daily_series[daily_series < 0].sum())
    pf = gp / gl if gl > 0 else 999.0
    wr = (daily_series > 0).sum() / len(daily_series) * 100
    return {
        "label": label,
        "net_dollar": round(total, 2),
        "sharpe": round(sharpe, 3),
        "pf": round(pf, 3),
        "max_dd": round(mdd, 2),
        "daily_wr": round(wr, 1),
        "trading_days": len(daily_series),
    }


# =========================================================================
# TASK 1: IS/OOS on scalp configs
# =========================================================================

def task1_is_oos_scalp():
    print("=" * 70)
    print("TASK 1: IS/OOS ANALYSIS ON SCALP CONFIGS")
    print("=" * 70)

    df = load_mnq_data()
    df_is, df_oos = split_is_oos(df)
    print(f"IS:  {len(df_is)} bars  {df_is.index[0]} to {df_is.index[-1]}")
    print(f"OOS: {len(df_oos)} bars  {df_oos.index[0]} to {df_oos.index[-1]}")

    configs = [
        {"label": "Config 1: TP=5, SL=40, CD=30, RSI=8, CutOff=13:00",
         "rsi": 8, "tp": 5, "sl": 40, "cd": 30, "entry_end": 13 * 60},
        {"label": "Config 2: TP=7, SL=30, CD=30, RSI=8, CutOff=13:00",
         "rsi": 8, "tp": 7, "sl": 30, "cd": 30, "entry_end": 13 * 60},
    ]

    for cfg in configs:
        print(f"\n{'─' * 70}")
        print(f"  {cfg['label']}")
        print(f"{'─' * 70}")

        for label, sub_df in [("FULL", df), ("IS", df_is), ("OOS", df_oos)]:
            sub_opens = sub_df['Open'].values
            sub_closes = sub_df['Close'].values
            sub_times = sub_df.index

            rsi_arr = compute_rsi(sub_closes, cfg['rsi'])
            long_sig, short_sig = generate_signals(rsi_arr)

            trades = run_backtest(sub_opens, sub_closes, sub_times,
                                  long_sig, short_sig,
                                  cooldown_bars=cfg['cd'],
                                  max_loss_pts=cfg['sl'],
                                  tp_pts=cfg['tp'],
                                  entry_end_et=cfg['entry_end'])
            sc = score_trades(trades)
            if sc:
                exits_str = " ".join(f"{k}:{v}" for k, v in sc['exits'].items())
                print(f"  {label:4s}  {sc['count']:4d} trades  WR {sc['win_rate']:5.1f}%  "
                      f"PF {sc['pf']:6.3f}  Sharpe {sc['sharpe']:6.3f}  "
                      f"Net ${sc['net_dollar']:>+9.2f}  MaxDD ${sc['max_dd_dollar']:>8.2f}  "
                      f"Exits: {exits_str}")
            else:
                print(f"  {label:4s}  NO TRADES")

    return df


# =========================================================================
# TASK 2: Portfolio correlation analysis
# =========================================================================

def task2_portfolio_correlation(df_mnq_rsi_tl):
    print(f"\n\n{'=' * 70}")
    print("TASK 2: PORTFOLIO CORRELATION ANALYSIS")
    print("=" * 70)

    # ----- Step 1: Run RSI TL best runner config -----
    print("\n--- Running RSI Trendline (Runner: RSI=8, TP1=7, TP2=20, SL=40, CD=30, CutOff=13:00) ---")
    rsi_tl_opens = df_mnq_rsi_tl['Open'].values
    rsi_tl_closes = df_mnq_rsi_tl['Close'].values
    rsi_tl_times = df_mnq_rsi_tl.index

    rsi_arr = compute_rsi(rsi_tl_closes, 8)
    long_sig, short_sig = generate_signals(rsi_arr)
    print(f"  Signals: {long_sig.sum()} long + {short_sig.sum()} short = {long_sig.sum() + short_sig.sum()}")

    rsi_tl_trades = run_backtest_runner(
        rsi_tl_opens, rsi_tl_closes, rsi_tl_times,
        long_sig, short_sig,
        cooldown_bars=30, max_loss_pts=40,
        tp1_pts=7, tp2_pts=20,
        entry_end_et=13 * 60,
    )
    rsi_tl_sc = score_trades(rsi_tl_trades)
    print(f"  RSI TL: {rsi_tl_sc['count']} trades, WR {rsi_tl_sc['win_rate']}%, "
          f"PF {rsi_tl_sc['pf']}, Net ${rsi_tl_sc['net_dollar']:+.2f}, "
          f"MaxDD ${rsi_tl_sc['max_dd_dollar']:.2f}, Sharpe {rsi_tl_sc['sharpe']}")

    # ----- Step 2: Run existing portfolio strategies with ALL gates -----
    print("\n--- Loading and running existing portfolio strategies with gates ---")

    # Load MNQ
    df_mnq = load_instrument_1min("MNQ")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm
    print(f"  MNQ: {len(df_mnq)} bars")

    # Load MES
    df_mes = load_instrument_1min("MES")
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm
    print(f"  MES: {len(df_mes)} bars")

    # --- Compute MNQ gates ---
    print("\n  Computing MNQ gates...")
    mnq_closes_full = df_mnq["Close"].values
    mnq_highs_full = df_mnq["High"].values
    mnq_lows_full = df_mnq["Low"].values
    mnq_sm_full = df_mnq["SM_Net"].values

    # Leledc
    bull_ex, bear_ex = compute_leledc_exhaustion(mnq_closes_full, maj_qual=9)
    mnq_leledc_gate = build_leledc_gate(bull_ex, bear_ex, persistence=1)

    # ADR directional
    mnq_session = compute_session_tracking(df_mnq)
    mnq_adr = compute_adr(df_mnq, lookback_days=14)
    mnq_adr_gate = build_directional_gate(
        mnq_session['move_from_open'], mnq_adr, mnq_sm_full, threshold=0.3
    )

    # Prior-day ATR (vScalpC)
    mnq_prior_atr = compute_prior_day_atr(df_mnq, lookback_days=14)
    mnq_atr_gate = np.ones(len(df_mnq), dtype=bool)
    for i in range(len(mnq_prior_atr)):
        if not np.isnan(mnq_prior_atr[i]) and mnq_prior_atr[i] < 263.8:
            mnq_atr_gate[i] = False

    # VIX
    start_date = df_mnq.index[0].strftime("%Y-%m-%d")
    end_date = df_mnq.index[-1].strftime("%Y-%m-%d")
    mnq_bar_dates_et = _get_bar_dates_et(df_mnq.index)
    mnq_vix_gate = load_vix_gate(start_date, end_date, mnq_bar_dates_et)

    # Composite gates
    gate_vscalpa = mnq_leledc_gate & mnq_adr_gate & mnq_vix_gate
    gate_vscalpb = mnq_leledc_gate & mnq_adr_gate  # vScalpB: no VIX, no ATR
    gate_vscalpc = mnq_leledc_gate & mnq_adr_gate & mnq_atr_gate & mnq_vix_gate

    # --- MES gates ---
    print("  Computing MES gates...")
    mes_closes_full = df_mes["Close"].values
    mes_highs_full = df_mes["High"].values
    mes_lows_full = df_mes["Low"].values
    mes_times_full = df_mes.index
    mes_volumes_full = df_mes["Volume"].values
    mes_et_mins = compute_et_minutes(mes_times_full)

    prev_high, prev_low, _ = compute_prior_day_levels(
        mes_times_full, mes_highs_full, mes_lows_full, mes_closes_full
    )
    mes_vpoc, mes_vah, mes_val = compute_rth_volume_profile(
        mes_times_full, mes_closes_full, mes_volumes_full, mes_et_mins, bin_width=5,
    )
    nan_arr = np.full(len(df_mes), np.nan)
    gate_mesv2 = build_prior_day_level_gate(
        mes_closes_full, nan_arr, nan_arr, mes_vpoc, nan_arr, mes_val, buffer_pts=5.0,
    )

    # --- Swing levels for vScalpC ---
    print("  Computing swing levels for vScalpC...")
    mnq_swing_highs, mnq_swing_lows = compute_swing_levels(
        mnq_highs_full, mnq_lows_full,
        lookback=VSCALPC_SWING_LB, swing_type="pivot",
        pivot_right=VSCALPC_SWING_PR,
    )

    # --- Run strategies ---
    mnq_opens = df_mnq["Open"].values
    mnq_highs = df_mnq["High"].values
    mnq_lows = df_mnq["Low"].values
    mnq_closes = df_mnq["Close"].values
    mnq_sm_arr = df_mnq["SM_Net"].values
    mnq_times = df_mnq.index

    # 5-min RSI for MNQ
    df_mnq_5m = resample_to_5min(df_mnq)
    rsi_mnq_curr, rsi_mnq_prev = map_5min_rsi_to_1min(
        df_mnq.index.values, df_mnq_5m.index.values,
        df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN,
    )

    # vScalpA
    print("\n  Running vScalpA (V15)...")
    trades_a = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
        rsi_mnq_curr, rsi_mnq_prev,
        VSCALPA_RSI_BUY, VSCALPA_RSI_SELL, VSCALPA_SM_THRESHOLD,
        VSCALPA_COOLDOWN, VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS,
        entry_end_et=VSCALPA_ENTRY_END_ET,
        entry_gate=gate_vscalpa,
    )
    sc_a = score_trades_v10(trades_a, commission_per_side=MNQ_COMMISSION,
                            dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"    vScalpA: {sc_a['count']} trades, WR {sc_a['win_rate']}%, "
          f"PF {sc_a['pf']}, Net ${sc_a['net_dollar']:+.2f}, Sharpe {sc_a['sharpe']}")

    # vScalpB
    print("  Running vScalpB...")
    trades_b = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
        rsi_mnq_curr, rsi_mnq_prev,
        VSCALPB_RSI_BUY, VSCALPB_RSI_SELL, VSCALPB_SM_THRESHOLD,
        VSCALPB_COOLDOWN, VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
        entry_gate=gate_vscalpb,
    )
    sc_b = score_trades_v10(trades_b, commission_per_side=MNQ_COMMISSION,
                            dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"    vScalpB: {sc_b['count']} trades, WR {sc_b['win_rate']}%, "
          f"PF {sc_b['pf']}, Net ${sc_b['net_dollar']:+.2f}, Sharpe {sc_b['sharpe']}")

    # vScalpC (structure exit)
    print("  Running vScalpC (structure exit)...")
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
    sc_c = score_structure_trades(trades_c, dollar_per_pt=MNQ_DOLLAR_PER_PT,
                                   commission_per_side=MNQ_COMMISSION)
    if sc_c:
        print(f"    vScalpC: {sc_c['count']} entries, WR {sc_c['win_rate']}%, "
              f"PF {sc_c['pf']}, Net ${sc_c['net_dollar']:+.2f}, Sharpe {sc_c['sharpe']}")

    # MES v2
    print("  Running MES v2...")
    mes_opens = df_mes["Open"].values
    mes_highs = df_mes["High"].values
    mes_lows = df_mes["Low"].values
    mes_closes = df_mes["Close"].values
    mes_sm_arr = df_mes["SM_Net"].values
    mes_times = df_mes.index

    df_mes_5m = resample_to_5min(df_mes)
    rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
        df_mes.index.values, df_mes_5m.index.values,
        df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN,
    )

    trades_m = run_backtest_tp_exit(
        mes_opens, mes_highs, mes_lows, mes_closes, mes_sm_arr, mes_times,
        rsi_mes_curr, rsi_mes_prev,
        MESV2_RSI_BUY, MESV2_RSI_SELL, MESV2_SM_THRESHOLD,
        MESV2_COOLDOWN, MESV2_MAX_LOSS_PTS, MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        entry_end_et=MESV2_ENTRY_END_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
        entry_gate=gate_mesv2,
    )
    sc_m = score_trades_v10(trades_m, commission_per_side=MES_COMMISSION,
                            dollar_per_pt=MES_DOLLAR_PER_PT)
    print(f"    MES v2: {sc_m['count']} trades, WR {sc_m['win_rate']}%, "
          f"PF {sc_m['pf']}, Net ${sc_m['net_dollar']:+.2f}, Sharpe {sc_m['sharpe']}")

    # ----- Step 3: Build daily P&L series for each strategy -----
    print(f"\n{'─' * 70}")
    print("DAILY P&L CORRELATION MATRIX")
    print(f"{'─' * 70}")

    daily_rsi_tl = trades_to_daily_pnl(rsi_tl_trades, DOLLAR_PER_PT,
                                        COMMISSION_PER_SIDE, "RSI_TL",
                                        is_runner_rsi_tl=True)
    daily_a = trades_to_daily_pnl(trades_a, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpA")
    daily_b = trades_to_daily_pnl(trades_b, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpB")
    daily_c = trades_to_daily_pnl(trades_c, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpC",
                                   is_structure=True)
    daily_m = trades_to_daily_pnl(trades_m, MES_DOLLAR_PER_PT, MES_COMMISSION, "MES_v2")

    # Combine into DataFrame
    all_dates = set()
    for s in [daily_rsi_tl, daily_a, daily_b, daily_c, daily_m]:
        all_dates |= set(s.index)
    idx = pd.DatetimeIndex(sorted(all_dates))

    daily_df = pd.DataFrame(index=idx)
    for label, s in [("RSI_TL", daily_rsi_tl), ("vScalpA", daily_a),
                     ("vScalpB", daily_b), ("vScalpC", daily_c), ("MES_v2", daily_m)]:
        daily_df[label] = s.reindex(idx, fill_value=0.0)

    # Correlation matrix
    corr = daily_df.corr()
    print("\nPearson correlation of daily P&L:")
    print(f"{'':>10}", end="")
    for col in corr.columns:
        print(f"{col:>10}", end="")
    print()
    for row in corr.index:
        print(f"{row:>10}", end="")
        for col in corr.columns:
            print(f"{corr.loc[row, col]:>10.3f}", end="")
        print()

    # ----- Step 4: Temporal overlap analysis -----
    print(f"\n{'─' * 70}")
    print("TEMPORAL OVERLAP ANALYSIS (±5 bars)")
    print(f"{'─' * 70}")

    # Get entry indices for RSI TL — group by entry_idx since runner has multiple legs
    rsi_tl_entry_idxs = set()
    for t in rsi_tl_trades:
        rsi_tl_entry_idxs.add(t["entry_idx"])
    rsi_tl_entry_idxs = sorted(rsi_tl_entry_idxs)

    # Get entry indices for SM strategies (all use MNQ bar indices)
    def get_entry_idxs(trades):
        seen = set()
        for t in trades:
            seen.add(t["entry_idx"])
        return sorted(seen)

    entry_idxs_a = get_entry_idxs(trades_a)
    entry_idxs_b = get_entry_idxs(trades_b)
    entry_idxs_c = get_entry_idxs(trades_c)
    # MES v2 is on different data (MES bars), so temporal overlap is in timestamp space
    entry_idxs_m = get_entry_idxs(trades_m)

    # For MNQ strategies: overlap in bar index space (±5 bars)
    def compute_overlap(rsi_tl_idxs, other_idxs, window=5):
        """What % of RSI TL entries have an other-strategy entry within ±window bars?"""
        if not rsi_tl_idxs or not other_idxs:
            return 0.0, 0
        other_set = set()
        for idx in other_idxs:
            for delta in range(-window, window + 1):
                other_set.add(idx + delta)
        overlap_count = sum(1 for idx in rsi_tl_idxs if idx in other_set)
        return overlap_count / len(rsi_tl_idxs) * 100, overlap_count

    # MNQ strategies are on the same bar index space as RSI TL (both use load_mnq_data)
    # BUT: RSI TL uses its own loader, and SM strategies use load_instrument_1min
    # Both load the same databento_MNQ files, so bar indices should align.
    # However, dedup strategy may differ (keep='first' vs keep='last').
    # For temporal overlap, we'll use timestamps instead for robustness.

    # Timestamp-based overlap (±5 minutes for 1-min bars)
    def compute_timestamp_overlap(rsi_tl_trades_list, other_trades_list, window_minutes=5):
        """What % of RSI TL entries have an other-strategy entry within ±window_minutes?"""
        if not rsi_tl_trades_list or not other_trades_list:
            return 0.0, 0

        # Get unique entry timestamps for RSI TL
        rsi_tl_times_set = set()
        for t in rsi_tl_trades_list:
            rsi_tl_times_set.add(pd.Timestamp(t["entry_time"]))
        rsi_tl_times_list = sorted(rsi_tl_times_set)

        # Get unique entry timestamps for other strategy
        other_times = set()
        for t in other_trades_list:
            other_times.add(pd.Timestamp(t["entry_time"]))
        other_times = sorted(other_times)

        if not other_times:
            return 0.0, 0

        # For each RSI TL entry, check if any other entry is within ±window_minutes
        other_arr = np.array([t.value for t in other_times])  # nanoseconds
        window_ns = window_minutes * 60 * 1_000_000_000

        overlap_count = 0
        for ts in rsi_tl_times_list:
            ts_ns = ts.value
            diffs = np.abs(other_arr - ts_ns)
            if np.min(diffs) <= window_ns:
                overlap_count += 1

        return overlap_count / len(rsi_tl_times_list) * 100, overlap_count

    n_rsi_tl_entries = len(rsi_tl_entry_idxs)
    print(f"\nRSI TL unique entries: {n_rsi_tl_entries}")
    print()

    for name, other_trades in [("vScalpA", trades_a), ("vScalpB", trades_b),
                                ("vScalpC", trades_c), ("MES_v2", trades_m)]:
        pct, cnt = compute_timestamp_overlap(rsi_tl_trades, other_trades)
        n_other = len(set(t["entry_idx"] for t in other_trades))
        print(f"  RSI_TL <-> {name:>8s}:  {pct:5.1f}% overlap ({cnt}/{n_rsi_tl_entries} RSI TL entries "
              f"have a {name} entry within ±5 min)   [{name} has {n_other} entries]")

    # Combined: any SM strategy within ±5 min
    all_sm_trades = trades_a + trades_b + trades_c + trades_m
    pct_any, cnt_any = compute_timestamp_overlap(rsi_tl_trades, all_sm_trades)
    print(f"\n  RSI_TL <-> ANY SM:     {pct_any:5.1f}% overlap ({cnt_any}/{n_rsi_tl_entries} RSI TL entries "
          f"have ANY SM strategy entry within ±5 min)")

    # ----- Step 5: Combined portfolio stats -----
    print(f"\n{'─' * 70}")
    print("COMBINED PORTFOLIO ANALYSIS")
    print(f"{'─' * 70}")

    # Existing portfolio: A(1) + B(1) + C(2) + MES(2)
    # Weighting: vScalpC trades at 2x contracts, MES v2 at 2x contracts
    # For daily P&L, multiply vScalpC and MES v2 by their respective contract counts
    daily_df_weighted = pd.DataFrame(index=idx)
    daily_df_weighted["vScalpA(1)"] = daily_a.reindex(idx, fill_value=0.0) * 1
    daily_df_weighted["vScalpB(1)"] = daily_b.reindex(idx, fill_value=0.0) * 1
    daily_df_weighted["vScalpC(2)"] = daily_c.reindex(idx, fill_value=0.0) * 2
    daily_df_weighted["MES_v2(2)"] = daily_m.reindex(idx, fill_value=0.0) * 2

    port_existing = daily_df_weighted.sum(axis=1)
    port_existing.name = "Existing_Portfolio"

    # RSI TL at 1 contract (runner = 2 contracts already included in trade qty)
    rsi_tl_daily_weighted = daily_rsi_tl.reindex(idx, fill_value=0.0)

    port_with_rsi_tl = port_existing + rsi_tl_daily_weighted

    # Stats
    stats_existing = portfolio_stats(port_existing, "Existing (A1+B1+C2+MES2)")
    stats_rsi_tl_alone = portfolio_stats(daily_rsi_tl, "RSI TL Alone (runner)")
    stats_combined = portfolio_stats(port_with_rsi_tl, "Existing + RSI TL")

    print(f"\n{'Label':<35s}  {'Net$':>10s}  {'Sharpe':>7s}  {'PF':>7s}  {'MaxDD':>10s}  {'DayWR':>7s}  {'Days':>5s}")
    print("-" * 90)
    for s in [stats_existing, stats_rsi_tl_alone, stats_combined]:
        if s:
            print(f"  {s['label']:<33s}  ${s['net_dollar']:>9,.2f}  {s['sharpe']:>7.3f}  "
                  f"{s['pf']:>7.3f}  ${s['max_dd']:>9,.2f}  {s['daily_wr']:>6.1f}%  {s['trading_days']:>5d}")

    # Improvement metrics
    if stats_existing and stats_combined:
        print(f"\n  Impact of adding RSI TL:")
        pnl_delta = stats_combined['net_dollar'] - stats_existing['net_dollar']
        sharpe_delta = stats_combined['sharpe'] - stats_existing['sharpe']
        mdd_delta = stats_combined['max_dd'] - stats_existing['max_dd']
        print(f"    P&L change:   ${pnl_delta:>+10,.2f}")
        print(f"    Sharpe change: {sharpe_delta:>+7.3f}")
        print(f"    MaxDD change:  ${mdd_delta:>+10,.2f}")
        if stats_existing['max_dd'] != 0:
            # Marginal Sharpe contribution
            print(f"    Return/Risk:   RSI TL adds ${pnl_delta:,.0f} P&L "
                  f"with {abs(mdd_delta):+,.0f} MaxDD impact")

    # ----- Step 6: Assessment -----
    print(f"\n{'─' * 70}")
    print("ASSESSMENT: Does RSI TL add diversification value?")
    print(f"{'─' * 70}")

    # Key metrics
    max_corr_with_sm = 0
    for col in ["vScalpA", "vScalpB", "vScalpC", "MES_v2"]:
        c = corr.loc["RSI_TL", col]
        if abs(c) > abs(max_corr_with_sm):
            max_corr_with_sm = c

    avg_corr = corr.loc["RSI_TL", ["vScalpA", "vScalpB", "vScalpC", "MES_v2"]].mean()

    print(f"\n  Max daily P&L correlation with any SM strategy:  {max_corr_with_sm:+.3f}")
    print(f"  Avg daily P&L correlation with SM strategies:    {avg_corr:+.3f}")
    print(f"  Temporal overlap with ANY SM strategy:           {pct_any:.1f}%")

    if abs(avg_corr) < 0.3 and pct_any < 30:
        print(f"\n  VERDICT: RSI TL provides GOOD diversification.")
        print(f"  - Low correlation (<0.3) means it wins/loses on different days")
        print(f"  - Low temporal overlap means it catches different moves")
    elif abs(avg_corr) < 0.3:
        print(f"\n  VERDICT: RSI TL provides MODERATE diversification.")
        print(f"  - Low correlation, but high temporal overlap suggests some signal redundancy")
    elif pct_any < 30:
        print(f"\n  VERDICT: RSI TL provides MODERATE diversification.")
        print(f"  - Independent timing, but correlated daily P&L suggests similar market sensitivity")
    else:
        print(f"\n  VERDICT: RSI TL provides LIMITED diversification.")
        print(f"  - High correlation and high temporal overlap suggest redundancy")

    # Sharpe improvement check
    if stats_existing and stats_combined:
        if stats_combined['sharpe'] > stats_existing['sharpe']:
            print(f"  - Combined Sharpe improves ({stats_existing['sharpe']:.3f} -> {stats_combined['sharpe']:.3f})")
        else:
            print(f"  - Combined Sharpe degrades ({stats_existing['sharpe']:.3f} -> {stats_combined['sharpe']:.3f})")


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time.time()

    # Task 1
    df_mnq_rsi_tl = task1_is_oos_scalp()

    # Task 2
    task2_portfolio_correlation(df_mnq_rsi_tl)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DONE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
