"""
Run active portfolio strategies on full 12-month data and save trade logs.

Runs vScalpA (v15), vScalpB, vScalpC, and MES v2 on Databento 1-min bars with
production parameters INCLUDING all entry gates. Saves per-trade CSVs to
backtesting_engine/results/.

Gates applied (matching live engine config.py):
  MNQ (vScalpA):  leledc(mq=9,p=1) & adr_dir(14d,0.3) & vix_death_zone(19-22)
  MNQ (vScalpB):  leledc(mq=9,p=1) & adr_dir(14d,0.3)
  MNQ (vScalpC):  leledc(mq=9,p=1) & adr_dir(14d,0.3) & atr_min(263.8) & vix(19-22)
  MES (MES v2):   prior_day_level(VPOC+VAL, buf=5)

Usage:
    python3 run_and_save_portfolio.py              # full 12 months
    python3 run_and_save_portfolio.py --split       # IS + OOS halves
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_test_common import (
    compute_rsi,
    compute_smart_money,
    compute_et_minutes,
    compute_prior_day_levels,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
)

# Use generate_session's data loader + backtest engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
    compute_mfe_mae,
    # MNQ shared SM params
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    # vScalpA params
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS, VSCALPA_ENTRY_END_ET,
    # vScalpB params
    VSCALPB_RSI_LEN, VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
    VSCALPB_SM_THRESHOLD, VSCALPB_COOLDOWN,
    VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
    # MES SM params
    MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA,
    MES_DOLLAR_PER_PT, MES_COMMISSION,
    # MES v2 params
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

from results.save_results import save_backtest

SCRIPT_NAME = "run_and_save_portfolio.py"

# vScalpC production params (structure exit runner)
VSCALPC_TP1_PTS = 7          # Scalp leg: close 1 contract at +7 pts
VSCALPC_MAX_TP2_PTS = 60     # Crash-safety OCO cap on exchange
VSCALPC_SL_PTS = 40          # Same SL as V15
VSCALPC_SL_TO_BE = True      # Move runner SL to entry after TP1
VSCALPC_SWING_LB = 50        # Pivot lookback
VSCALPC_SWING_PR = 2         # Pivot right confirmation bars
VSCALPC_SWING_BUF = 2.0      # Exit 2 pts before swing level

# MES v2 corrected params (matching config.py production)
MESV2_EOD_ET = 16 * 60       # 16:00 ET = 960 minutes (was 15:30 in generate_session)
MESV2_ENTRY_END_ET = 14 * 60 + 15  # 14:15 ET = 855 minutes


# ---------------------------------------------------------------------------
# VIX data loader
# ---------------------------------------------------------------------------

def load_vix_gate(start_date, end_date, bar_dates_et, low=19.0, high=22.0):
    """Download VIX data, return per-bar gate (True=allowed, False=blocked).

    Gate blocks when prior-day VIX close is in [low, high] (death zone).
    Fail-open on download error (returns all-True).

    Args:
        start_date: str "YYYY-MM-DD" — earliest bar date
        end_date:   str "YYYY-MM-DD" — latest bar date
        bar_dates_et: array of datetime.date for each bar (ET calendar date)
        low:  VIX death zone lower bound
        high: VIX death zone upper bound

    Returns:
        Boolean numpy array (True = allow entry, False = block).
    """
    n = len(bar_dates_et)
    gate = np.ones(n, dtype=bool)

    try:
        import yfinance as yf
        # Fetch a bit extra to get prior-day close for the first trading day
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if vix.empty:
            print("  WARNING: VIX download returned empty — VIX gate fail-open")
            return gate

        # Flatten MultiIndex columns if needed
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        vix = vix.reset_index()
        vix["date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.date
        vix_close_by_date = dict(zip(vix["date"], vix["Close"].values))

        # Build sorted date list for prior-day lookup
        sorted_vix_dates = sorted(vix_close_by_date.keys())
        prior_close = {}
        for i in range(1, len(sorted_vix_dates)):
            prior_close[sorted_vix_dates[i]] = vix_close_by_date[sorted_vix_dates[i - 1]]

        # Map to bars
        for i in range(n):
            d = bar_dates_et[i]
            if d in prior_close:
                pc = float(prior_close[d])
                if low <= pc <= high:
                    gate[i] = False

        blocked = (~gate).sum()
        total = n
        print(f"  VIX death zone [{low}-{high}]: blocks {blocked}/{total} bars "
              f"({blocked/total*100:.1f}%)")

    except Exception as e:
        print(f"  WARNING: VIX download failed ({e}) — VIX gate fail-open")

    return gate


# ---------------------------------------------------------------------------
# Helper: get ET dates for bar array
# ---------------------------------------------------------------------------

def _get_bar_dates_et(times):
    """Convert bar timestamps to ET calendar dates."""
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    return idx.tz_convert(_ET).date


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def run_strategy(name, strategy_id, opens, highs, lows, closes, sm, times,
                 rsi_curr, rsi_prev, rsi_buy, rsi_sell, sm_threshold,
                 cooldown, max_loss_pts, tp_pts, dollar_per_pt, commission,
                 params_dict, data_range, split="FULL", eod_et=None,
                 entry_end_et=None, breakeven_after_bars=0, entry_gate=None):
    """Run a single strategy and save results."""
    kwargs = dict(
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts, tp_pts=tp_pts,
    )
    if eod_et is not None:
        kwargs["eod_minutes_et"] = eod_et
    if entry_end_et is not None:
        kwargs["entry_end_et"] = entry_end_et
    if breakeven_after_bars > 0:
        kwargs["breakeven_after_bars"] = breakeven_after_bars
    if entry_gate is not None:
        kwargs["entry_gate"] = entry_gate

    trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev, **kwargs,
    )
    compute_mfe_mae(trades, highs, lows)

    sc = score_trades(trades, commission_per_side=commission,
                      dollar_per_pt=dollar_per_pt)
    label = f"{name} ({split})"
    print(f"\n  {fmt_score(sc, label)}")

    save_backtest(
        trades, strategy=strategy_id, params=params_dict,
        data_range=data_range, split=split,
        dollar_per_pt=dollar_per_pt, commission=commission,
        script_name=SCRIPT_NAME,
    )
    return trades, sc


def run_structure_strategy(name, strategy_id, opens, highs, lows, closes, sm,
                           times, rsi_curr, rsi_prev, rsi_buy, rsi_sell,
                           sm_threshold, cooldown, max_loss_pts,
                           tp1_pts, max_tp2_pts, swing_highs, swing_lows,
                           swing_buffer_pts, dollar_per_pt, commission,
                           params_dict, data_range, split="FULL",
                           entry_end_et=None, entry_gate=None):
    """Run vScalpC with structure-based runner exit."""
    trades = run_backtest_structure_exit(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts,
        tp1_pts=tp1_pts,
        swing_highs=swing_highs, swing_lows=swing_lows,
        swing_buffer_pts=swing_buffer_pts,
        max_tp2_pts=max_tp2_pts,
        move_sl_to_be_after_tp1=True,
        breakeven_after_bars=0,  # Structure monitor replaces BE_TIME
        entry_end_et=entry_end_et if entry_end_et is not None else (13 * 60),
        entry_gate=entry_gate,
    )

    # Score with structure_trades scorer (2 legs per entry)
    sc = score_structure_trades(trades, dollar_per_pt=dollar_per_pt,
                                commission_per_side=commission)
    label = f"{name} ({split})"
    if sc:
        print(f"\n  {label}  {sc['count']} entries, WR {sc['win_rate']}%, "
              f"PF {sc['pf']}, Net ${sc['net_dollar']:+.2f}, "
              f"MaxDD ${sc['max_dd_dollar']:.2f}, Sharpe {sc['sharpe']}")
        print(f"    Scalp exits: {sc['scalp_exits']}")
        print(f"    Runner exits: {sc['runner_exits']}")
    else:
        print(f"\n  {label}: NO TRADES")

    save_backtest(
        trades, strategy=strategy_id, params=params_dict,
        data_range=data_range, split=split,
        dollar_per_pt=dollar_per_pt, commission=commission,
        script_name=SCRIPT_NAME, qty=2,
    )
    return trades, sc


# ---------------------------------------------------------------------------
# Gate slicing for IS/OOS
# ---------------------------------------------------------------------------

def _slice_gate(full_gate, full_len, split_name, is_len):
    """Slice a gate array for IS/OOS splits.

    Args:
        full_gate: boolean array (same length as full data), or None
        full_len: total bars in full data
        split_name: "FULL", "IS", or "OOS"
        is_len: number of bars in IS period

    Returns:
        Sliced gate array or None.
    """
    if full_gate is None:
        return None
    if split_name == "FULL":
        return full_gate
    elif split_name == "IS":
        return full_gate[:is_len]
    else:  # OOS
        return full_gate[is_len:]


# ---------------------------------------------------------------------------
# Daily P&L extraction from trade list
# ---------------------------------------------------------------------------

def _trades_to_daily_pnl(trades, dollar_per_pt, commission_per_side, label,
                          is_structure=False):
    """Convert a trade list to a daily P&L Series.

    For structure trades (2 legs per entry), groups by entry_idx to
    compute per-entry P&L, then aggregates to daily.
    """
    if not trades:
        return pd.Series(dtype=float, name=label)

    if is_structure:
        # Group by entry_idx, sum P&L per entry
        comm = commission_per_side * 2  # entry+exit per leg
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
            rows.append({
                "trade_date": entry_et.strftime("%Y-%m-%d"),
                "pnl": entries[eidx]["pnl"],
            })
    else:
        comm_pts = (commission_per_side * 2) / dollar_per_pt
        rows = []
        for t in trades:
            pnl = (t["pts"] - comm_pts) * dollar_per_pt
            entry_ts = pd.Timestamp(t["entry_time"])
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_et = entry_ts.tz_convert("America/New_York")
            rows.append({
                "trade_date": entry_et.strftime("%Y-%m-%d"),
                "pnl": pnl,
            })

    if not rows:
        return pd.Series(dtype=float, name=label)

    df = pd.DataFrame(rows)
    daily = df.groupby("trade_date")["pnl"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily.name = label
    return daily


# ---------------------------------------------------------------------------
# Monthly breakdown output
# ---------------------------------------------------------------------------

def print_monthly_breakdown(daily_dict, all_trades_dict):
    """Print month-by-month table with per-strategy and portfolio totals.

    Args:
        daily_dict: dict of label -> daily P&L Series
        all_trades_dict: dict of label -> (trades, sc, is_structure)
    """
    # Combine all daily series into a DataFrame
    all_dates = set()
    for s in daily_dict.values():
        all_dates |= set(s.index)
    if not all_dates:
        print("\n  No trades to summarize.")
        return

    idx = pd.DatetimeIndex(sorted(all_dates))
    combined = pd.DataFrame(index=idx)
    for label, series in daily_dict.items():
        combined[label] = series.reindex(idx, fill_value=0.0)
    combined["Portfolio"] = combined.sum(axis=1)

    # Monthly breakdown
    labels = list(daily_dict.keys()) + ["Portfolio"]
    monthly = combined.resample("ME").sum()
    monthly.index = monthly.index.strftime("%Y-%m")

    # Header
    col_w = 10
    header_labels = [f"{l:>{col_w}}" for l in labels]
    print(f"\n{'='*70}")
    print(f"  MONTHLY P&L BREAKDOWN")
    print(f"{'='*70}")
    print(f"  {'Month':<10} {''.join(header_labels)}")
    print(f"  {'-'*10} {(' ' + '-'*col_w) * len(labels)}")

    for month in monthly.index:
        vals = [f"{monthly.loc[month, l]:>{col_w},.0f}" for l in labels]
        print(f"  {month:<10} {''.join(vals)}")

    # Totals row
    print(f"  {'-'*10} {(' ' + '-'*col_w) * len(labels)}")
    totals = [f"{monthly[l].sum():>{col_w},.0f}" for l in labels]
    print(f"  {'TOTAL':<10} {''.join(totals)}")

    # Portfolio summary metrics
    port_daily = combined["Portfolio"]
    total = port_daily.sum()
    mean_d = port_daily.mean()
    std_d = port_daily.std()
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0
    cum = port_daily.cumsum()
    mdd = (cum - cum.cummax()).min()
    gp = port_daily[port_daily > 0].sum()
    gl = abs(port_daily[port_daily < 0].sum())
    pf = gp / gl if gl > 0 else 999.0
    wr = (port_daily > 0).sum() / len(port_daily) * 100 if len(port_daily) > 0 else 0

    print(f"\n  PORTFOLIO SUMMARY:")
    print(f"    Total P&L:   ${total:>10,.2f}")
    print(f"    Sharpe:      {sharpe:>10.2f}")
    print(f"    PF (daily):  {pf:>10.3f}")
    print(f"    MaxDD:       ${mdd:>10,.2f}")
    print(f"    Daily WR:    {wr:>9.1f}%")
    print(f"    Trading Days: {len(port_daily)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true",
                        help="Also save IS and OOS halves")
    args = parser.parse_args()

    print("=" * 70)
    print("PORTFOLIO BACKTEST — vScalpA + vScalpB + vScalpC + MES v2")
    print("  WITH PRODUCTION ENTRY GATES")
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

    # Data range string
    start_date = df_mnq.index[0].strftime("%Y-%m-%d")
    end_date = df_mnq.index[-1].strftime("%Y-%m-%d")
    data_range = f"{start_date}_to_{end_date}"

    # ===================================================================
    # COMPUTE ALL GATES ON FULL DATA (before splitting IS/OOS)
    # ===================================================================
    print("\n--- Computing Entry Gates ---")

    mnq_closes_full = df_mnq["Close"].values
    mnq_highs_full = df_mnq["High"].values
    mnq_lows_full = df_mnq["Low"].values
    mnq_sm_full = df_mnq["SM_Net"].values

    # --- MNQ gates ---

    # 1. Leledc exhaustion gate (mq=9, persistence=1)
    print("  Computing Leledc exhaustion (mq=9)...")
    bull_ex, bear_ex = compute_leledc_exhaustion(mnq_closes_full, maj_qual=9)
    mnq_leledc_gate = build_leledc_gate(bull_ex, bear_ex, persistence=1)
    blocked = (~mnq_leledc_gate).sum()
    print(f"    Leledc: blocks {blocked}/{len(mnq_leledc_gate)} bars "
          f"({blocked/len(mnq_leledc_gate)*100:.1f}%)")

    # 2. ADR directional gate (lookback=14 days, threshold=0.3)
    print("  Computing ADR directional gate (14d, 0.3)...")
    mnq_session = compute_session_tracking(df_mnq)
    mnq_adr = compute_adr(df_mnq, lookback_days=14)
    mnq_adr_gate = build_directional_gate(
        mnq_session['move_from_open'], mnq_adr, mnq_sm_full, threshold=0.3
    )
    blocked = (~mnq_adr_gate).sum()
    print(f"    ADR directional: blocks {blocked}/{len(mnq_adr_gate)} bars "
          f"({blocked/len(mnq_adr_gate)*100:.1f}%)")

    # 3. Prior-day ATR gate (atr >= 263.8, vScalpC only)
    print("  Computing prior-day ATR gate (min=263.8)...")
    mnq_prior_atr = compute_prior_day_atr(df_mnq, lookback_days=14)
    mnq_atr_gate = np.ones(len(df_mnq), dtype=bool)
    for i in range(len(mnq_prior_atr)):
        if not np.isnan(mnq_prior_atr[i]) and mnq_prior_atr[i] < 263.8:
            mnq_atr_gate[i] = False
    # NaN = fail-open (already True)
    blocked = (~mnq_atr_gate).sum()
    print(f"    ATR(14) min 263.8: blocks {blocked}/{len(mnq_atr_gate)} bars "
          f"({blocked/len(mnq_atr_gate)*100:.1f}%)")

    # 4. VIX death zone gate (19-22, vScalpA + vScalpC)
    print("  Computing VIX death zone gate (19-22)...")
    mnq_bar_dates_et = _get_bar_dates_et(df_mnq.index)
    mnq_vix_gate = load_vix_gate(start_date, end_date, mnq_bar_dates_et,
                                  low=19.0, high=22.0)

    # --- MES gates ---

    # Prior-day level gate: VPOC + VAL only, buffer=5 pts
    print("  Computing MES prior-day level gate (VPOC+VAL, buf=5)...")
    mes_closes_full = df_mes["Close"].values
    mes_highs_full = df_mes["High"].values
    mes_lows_full = df_mes["Low"].values
    mes_times_full = df_mes.index
    mes_volumes_full = df_mes["Volume"].values
    mes_et_mins = compute_et_minutes(mes_times_full)

    # Prior-day H/L (we pass NaN arrays for high/low — only VPOC+VAL active)
    prev_high, prev_low, _ = compute_prior_day_levels(
        mes_times_full, mes_highs_full, mes_lows_full, mes_closes_full
    )

    # Volume profile: VPOC, VAH, VAL
    mes_vpoc, mes_vah, mes_val = compute_rth_volume_profile(
        mes_times_full, mes_closes_full, mes_volumes_full, mes_et_mins,
        bin_width=5,  # MES bin width
    )

    # Build gate with VPOC + VAL only (pass NaN for high, low, VAH)
    nan_arr = np.full(len(df_mes), np.nan)
    mes_level_gate = build_prior_day_level_gate(
        mes_closes_full,
        nan_arr,     # prev_high — disabled
        nan_arr,     # prev_low — disabled
        mes_vpoc,    # VPOC — active
        nan_arr,     # VAH — disabled
        mes_val,     # VAL — active
        buffer_pts=5.0,
    )
    blocked = (~mes_level_gate).sum()
    print(f"    Prior-day VPOC+VAL (buf=5): blocks {blocked}/{len(mes_level_gate)} bars "
          f"({blocked/len(mes_level_gate)*100:.1f}%)")

    # --- Composite gates (AND all applicable) ---
    gate_vscalpa = mnq_leledc_gate & mnq_adr_gate & mnq_vix_gate
    gate_vscalpb = mnq_leledc_gate & mnq_adr_gate
    gate_vscalpc = mnq_leledc_gate & mnq_adr_gate & mnq_atr_gate & mnq_vix_gate
    gate_mesv2 = mes_level_gate

    # Print composite stats
    for gname, garr in [("vScalpA", gate_vscalpa), ("vScalpB", gate_vscalpb),
                        ("vScalpC", gate_vscalpc), ("MES v2", gate_mesv2)]:
        blocked = (~garr).sum()
        print(f"  Composite {gname}: blocks {blocked}/{len(garr)} bars "
              f"({blocked/len(garr)*100:.1f}%)")

    # --- Pre-compute swing levels for vScalpC structure exit (on full MNQ data) ---
    print("\n  Computing swing levels for vScalpC structure exit "
          f"(LB={VSCALPC_SWING_LB}, PR={VSCALPC_SWING_PR})...")
    mnq_swing_highs, mnq_swing_lows = compute_swing_levels(
        mnq_highs_full, mnq_lows_full,
        lookback=VSCALPC_SWING_LB, swing_type="pivot",
        pivot_right=VSCALPC_SWING_PR,
    )

    # ===================================================================
    # Define splits
    # ===================================================================
    mnq_is_len = len(df_mnq) // 2
    mes_is_len = len(df_mes) // 2

    splits = [("FULL", df_mnq, df_mes)]

    if args.split:
        midpoint = df_mnq.index[mnq_is_len]
        mnq_is = df_mnq[df_mnq.index < midpoint]
        mnq_oos = df_mnq[df_mnq.index >= midpoint]
        mes_midpoint = df_mes.index[mes_is_len]
        mes_is = df_mes[df_mes.index < mes_midpoint]
        mes_oos = df_mes[df_mes.index >= mes_midpoint]

        is_range = f"{mnq_is.index[0].strftime('%Y-%m-%d')}_to_{mnq_is.index[-1].strftime('%Y-%m-%d')}"
        oos_range = f"{mnq_oos.index[0].strftime('%Y-%m-%d')}_to_{mnq_oos.index[-1].strftime('%Y-%m-%d')}"

        splits.append(("IS", mnq_is, mes_is))
        splits.append(("OOS", mnq_oos, mes_oos))

    for split_name, mnq_data, mes_data in splits:
        if split_name == "IS":
            dr = is_range
        elif split_name == "OOS":
            dr = oos_range
        else:
            dr = data_range

        print(f"\n{'='*70}")
        print(f"  SPLIT: {split_name}  ({len(mnq_data)} MNQ bars, {len(mes_data)} MES bars)")
        print(f"{'='*70}")

        # --- Slice gates for this split ---
        g_vscalpa = _slice_gate(gate_vscalpa, len(df_mnq), split_name, mnq_is_len)
        g_vscalpb = _slice_gate(gate_vscalpb, len(df_mnq), split_name, mnq_is_len)
        g_vscalpc = _slice_gate(gate_vscalpc, len(df_mnq), split_name, mnq_is_len)
        g_mesv2 = _slice_gate(gate_mesv2, len(df_mes), split_name, mes_is_len)

        # Slice swing levels for vScalpC
        sw_highs = _slice_gate(mnq_swing_highs, len(df_mnq), split_name, mnq_is_len)
        sw_lows = _slice_gate(mnq_swing_lows, len(df_mnq), split_name, mnq_is_len)

        # --- MNQ arrays ---
        mnq_opens = mnq_data["Open"].values
        mnq_highs = mnq_data["High"].values
        mnq_lows = mnq_data["Low"].values
        mnq_closes = mnq_data["Close"].values
        mnq_sm_arr = mnq_data["SM_Net"].values
        mnq_times = mnq_data.index

        # 5-min RSI mapping for MNQ (RSI len=8 shared by both)
        df_mnq_5m = resample_to_5min(mnq_data)
        rsi_mnq_curr, rsi_mnq_prev = map_5min_rsi_to_1min(
            mnq_data.index.values, df_mnq_5m.index.values,
            df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN,
        )

        # --- vScalpA ---
        vscalpa_params = {
            "sm_index": MNQ_SM_INDEX, "sm_flow": MNQ_SM_FLOW,
            "sm_norm": MNQ_SM_NORM, "sm_ema": MNQ_SM_EMA,
            "sm_threshold": VSCALPA_SM_THRESHOLD,
            "rsi_len": VSCALPA_RSI_LEN, "rsi_buy": VSCALPA_RSI_BUY,
            "rsi_sell": VSCALPA_RSI_SELL, "cooldown": VSCALPA_COOLDOWN,
            "max_loss_pts": VSCALPA_MAX_LOSS_PTS, "tp_pts": VSCALPA_TP_PTS,
            "gates": "leledc+adr+vix",
        }
        trades_a, sc_a = run_strategy(
            "vScalpA", "MNQ_V15",
            mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
            rsi_mnq_curr, rsi_mnq_prev,
            VSCALPA_RSI_BUY, VSCALPA_RSI_SELL, VSCALPA_SM_THRESHOLD,
            VSCALPA_COOLDOWN, VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS,
            MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
            vscalpa_params, dr, split_name,
            entry_end_et=VSCALPA_ENTRY_END_ET,
            entry_gate=g_vscalpa,
        )

        # --- vScalpB ---
        vscalpb_params = {
            "sm_index": MNQ_SM_INDEX, "sm_flow": MNQ_SM_FLOW,
            "sm_norm": MNQ_SM_NORM, "sm_ema": MNQ_SM_EMA,
            "sm_threshold": VSCALPB_SM_THRESHOLD,
            "rsi_len": VSCALPB_RSI_LEN, "rsi_buy": VSCALPB_RSI_BUY,
            "rsi_sell": VSCALPB_RSI_SELL, "cooldown": VSCALPB_COOLDOWN,
            "max_loss_pts": VSCALPB_MAX_LOSS_PTS, "tp_pts": VSCALPB_TP_PTS,
            "gates": "leledc+adr",
        }
        trades_b, sc_b = run_strategy(
            "vScalpB", "MNQ_VSCALPB",
            mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
            rsi_mnq_curr, rsi_mnq_prev,
            VSCALPB_RSI_BUY, VSCALPB_RSI_SELL, VSCALPB_SM_THRESHOLD,
            VSCALPB_COOLDOWN, VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
            MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
            vscalpb_params, dr, split_name,
            entry_gate=g_vscalpb,
        )

        # --- vScalpC (structure exit runner) ---
        vscalpc_params = {
            "sm_index": MNQ_SM_INDEX, "sm_flow": MNQ_SM_FLOW,
            "sm_norm": MNQ_SM_NORM, "sm_ema": MNQ_SM_EMA,
            "sm_threshold": VSCALPA_SM_THRESHOLD,
            "rsi_len": VSCALPA_RSI_LEN, "rsi_buy": VSCALPA_RSI_BUY,
            "rsi_sell": VSCALPA_RSI_SELL, "cooldown": VSCALPA_COOLDOWN,
            "sl_pts": VSCALPC_SL_PTS, "tp1_pts": VSCALPC_TP1_PTS,
            "max_tp2_pts": VSCALPC_MAX_TP2_PTS,
            "swing_lookback": VSCALPC_SWING_LB,
            "swing_pivot_right": VSCALPC_SWING_PR,
            "swing_buffer_pts": VSCALPC_SWING_BUF,
            "gates": "leledc+adr+atr+vix",
        }
        trades_c, sc_c = run_structure_strategy(
            "vScalpC", "MNQ_VSCALPC",
            mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
            rsi_mnq_curr, rsi_mnq_prev,
            VSCALPA_RSI_BUY, VSCALPA_RSI_SELL, VSCALPA_SM_THRESHOLD,
            VSCALPA_COOLDOWN, VSCALPC_SL_PTS,
            VSCALPC_TP1_PTS, VSCALPC_MAX_TP2_PTS,
            sw_highs, sw_lows, VSCALPC_SWING_BUF,
            MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
            vscalpc_params, dr, split_name,
            entry_end_et=VSCALPA_ENTRY_END_ET,
            entry_gate=g_vscalpc,
        )

        # --- MES v2 ---
        mes_opens = mes_data["Open"].values
        mes_highs = mes_data["High"].values
        mes_lows = mes_data["Low"].values
        mes_closes = mes_data["Close"].values
        mes_sm_arr = mes_data["SM_Net"].values
        mes_times = mes_data.index

        df_mes_5m = resample_to_5min(mes_data)
        rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
            mes_data.index.values, df_mes_5m.index.values,
            df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN,
        )

        mesv2_params = {
            "sm_index": MES_SM_INDEX, "sm_flow": MES_SM_FLOW,
            "sm_norm": MES_SM_NORM, "sm_ema": MES_SM_EMA,
            "sm_threshold": MESV2_SM_THRESHOLD,
            "rsi_len": MESV2_RSI_LEN, "rsi_buy": MESV2_RSI_BUY,
            "rsi_sell": MESV2_RSI_SELL, "cooldown": MESV2_COOLDOWN,
            "max_loss_pts": MESV2_MAX_LOSS_PTS, "tp_pts": MESV2_TP_PTS,
            "eod_et": MESV2_EOD_ET,
            "entry_end_et": MESV2_ENTRY_END_ET,
            "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
            "gates": "prior_day_level(VPOC+VAL,buf=5)",
        }
        trades_m, sc_m = run_strategy(
            "MES v2", "MES_V2",
            mes_opens, mes_highs, mes_lows, mes_closes, mes_sm_arr, mes_times,
            rsi_mes_curr, rsi_mes_prev,
            MESV2_RSI_BUY, MESV2_RSI_SELL, MESV2_SM_THRESHOLD,
            MESV2_COOLDOWN, MESV2_MAX_LOSS_PTS, MESV2_TP_PTS,
            MES_DOLLAR_PER_PT, MES_COMMISSION,
            mesv2_params, dr, split_name,
            eod_et=MESV2_EOD_ET,
            entry_end_et=MESV2_ENTRY_END_ET,
            breakeven_after_bars=MESV2_BREAKEVEN_BARS,
            entry_gate=g_mesv2,
        )

        # --- Monthly breakdown (FULL split only) ---
        if split_name == "FULL":
            daily_a = _trades_to_daily_pnl(
                trades_a, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpA")
            daily_b = _trades_to_daily_pnl(
                trades_b, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpB")
            daily_c = _trades_to_daily_pnl(
                trades_c, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpC",
                is_structure=True)
            daily_m = _trades_to_daily_pnl(
                trades_m, MES_DOLLAR_PER_PT, MES_COMMISSION, "MES_v2")

            print_monthly_breakdown(
                {"vScalpA": daily_a, "vScalpB": daily_b,
                 "vScalpC": daily_c, "MES_v2": daily_m},
                {"vScalpA": (trades_a, sc_a, False),
                 "vScalpB": (trades_b, sc_b, False),
                 "vScalpC": (trades_c, sc_c, True),
                 "MES_v2": (trades_m, sc_m, False)},
            )

    print(f"\n{'='*70}")
    print("Done. Results saved to backtesting_engine/results/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
