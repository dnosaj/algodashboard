"""
Prior Year Validation — True Out-of-Sample Test
=================================================
Runs ALL production strategies on a NEVER-BEFORE-SEEN 12-month dataset
(Feb 2024 – Feb 2025) that was NOT used during strategy development.

This is the most important test in the project: if the strategies work on
completely unseen data, the edge is real. If they don't, it's overfit.

Strategies tested (production params, NO gates):
  1. vScalpA (MNQ_V15)  — SM(10/12/200/100) SM_T=0.0 RSI(8/60/40) CD=20 SL=40 TP=7
  2. vScalpB (MNQ_VSCALPB) — SM(10/12/200/100) SM_T=0.25 RSI(8/55/45) CD=20 SL=10 TP=3
  3. vScalpC (MNQ_VSCALPC) — SM(10/12/200/100) SM_T=0.0 RSI(8/60/40) CD=20 SL=40 TP1=7 TP2=25
  4. MES v2 (MES_V2) — SM(20/12/400/255) SM_T=0.0 RSI(12/55/45) CD=25 SL=35 TP=20 BE_TIME=75
  5. RSI Trendline (MNQ_RSI_TL) — RSI(8) trendline breakout, TP1=7 TP2=20 SL=40 CD=30

Price context: MNQ 17,120–22,216 (prior) vs 20,000–24,700 (dev period).
Fixed-point TP/SL represent different % moves in each regime.

Usage:
    cd backtesting_engine && python3 strategies/prior_year_validation.py
"""

import sys
import warnings
from pathlib import Path

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
)

from generate_session import run_backtest_tp_exit

from vscalpc_partial_exit_sweep import (
    run_backtest_partial_exit,
    score_partial_trades,
    daily_pnl_series,
)

from rsi_trendline_backtest import (
    compute_rsi as compute_rsi_1min,
    generate_signals as generate_rsi_tl_signals,
    run_backtest_runner as run_rsi_tl_backtest_runner,
    score_trades as score_rsi_tl_trades,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"

# MNQ SM params (shared by vScalpA, vScalpB, vScalpC)
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

# vScalpC params (runner, TP2=25 pre-structure-exit config)
VSCALPC_SL_PTS = 40
VSCALPC_TP1_PTS = 7
VSCALPC_TP2_PTS = 25
VSCALPC_ENTRY_END_ET = 13 * 60

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

# RSI Trendline params
RSI_TL_RSI_LEN = 8
RSI_TL_TP1 = 7
RSI_TL_TP2 = 20
RSI_TL_SL = 40
RSI_TL_CD = 30
RSI_TL_ENTRY_START_ET = 9 * 60 + 30  # 09:30 ET
RSI_TL_ENTRY_END_ET = 13 * 60         # 13:00 ET


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_prior_data(instrument):
    """Load prior year (Feb 2024 – Feb 2025) data for an instrument."""
    filepath = DATA_DIR / f"prior_databento_{instrument}_1min_2024-02-17_to_2025-02-16.csv"
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


def prepare_instrument(instrument, sm_index, sm_flow, sm_norm, sm_ema):
    """Load data and compute SM for an instrument."""
    df = load_prior_data(instrument)
    sm = compute_smart_money(
        df['Close'].values, df['Volume'].values,
        index_period=sm_index, flow_period=sm_flow,
        norm_period=sm_norm, ema_len=sm_ema,
    )
    df['SM_Net'] = sm
    price_range = f"{df['Close'].min():.0f}–{df['Close'].max():.0f}"
    print(f"  {instrument}: {len(df):,} bars, {df.index[0]} to {df.index[-1]}")
    print(f"    Price range: {price_range}")
    return df


# ---------------------------------------------------------------------------
# Daily P&L helper (single-contract trades)
# ---------------------------------------------------------------------------

def trades_to_daily_pnl(trades, dollar_per_pt, commission_per_side, label,
                         is_partial=False):
    """Convert a trade list to a daily P&L Series."""
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")

    if not trades:
        return pd.Series(dtype=float, name=label)

    rows = []
    if is_partial:
        comm_per_leg = commission_per_side * 2
        for t in trades:
            pnl = t["leg1_pts"] * dollar_per_pt - comm_per_leg + \
                  t["leg2_pts"] * dollar_per_pt - comm_per_leg
            entry_ts = pd.Timestamp(t["entry_time"])
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_et = entry_ts.tz_convert(_ET)
            rows.append({"date": entry_et.strftime("%Y-%m-%d"), "pnl": pnl})
    else:
        comm_pts = (commission_per_side * 2) / dollar_per_pt
        for t in trades:
            qty = t.get("qty", 1)
            pnl = (t["pts"] - comm_pts) * dollar_per_pt * qty
            entry_ts = pd.Timestamp(t["entry_time"])
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_et = entry_ts.tz_convert(_ET)
            rows.append({"date": entry_et.strftime("%Y-%m-%d"), "pnl": pnl})

    if not rows:
        return pd.Series(dtype=float, name=label)

    df = pd.DataFrame(rows)
    daily = df.groupby("date")["pnl"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily.name = label
    return daily


# ---------------------------------------------------------------------------
# RSI TL daily P&L helper
# ---------------------------------------------------------------------------

def rsi_tl_trades_to_daily_pnl(trades, label):
    """Convert RSI TL trades (with qty) to daily P&L."""
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")

    if not trades:
        return pd.Series(dtype=float, name=label)

    comm_pts = (MNQ_COMMISSION * 2) / MNQ_DOLLAR_PER_PT
    rows = []
    for t in trades:
        qty = t.get("qty", 1)
        pnl = (t["pts"] - comm_pts) * MNQ_DOLLAR_PER_PT * qty
        entry_ts = pd.Timestamp(t["entry_time"])
        if entry_ts.tz is None:
            entry_ts = entry_ts.tz_localize("UTC")
        entry_et = entry_ts.tz_convert(_ET)
        rows.append({"date": entry_et.strftime("%Y-%m-%d"), "pnl": pnl})

    df = pd.DataFrame(rows)
    daily = df.groupby("date")["pnl"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily.name = label
    return daily


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PRIOR YEAR VALIDATION — TRUE OUT-OF-SAMPLE TEST")
    print("Data: Feb 2024 – Feb 2025 (NEVER seen during development)")
    print("NO GATES — raw strategy signals only")
    print("=" * 80)

    # --- Load Data ---
    print("\n--- Loading Data ---")
    df_mnq = prepare_instrument("MNQ", MNQ_SM_INDEX, MNQ_SM_FLOW,
                                 MNQ_SM_NORM, MNQ_SM_EMA)
    df_mes = prepare_instrument("MES", MES_SM_INDEX, MES_SM_FLOW,
                                 MES_SM_NORM, MES_SM_EMA)

    # --- Price context ---
    mnq_min = df_mnq['Close'].min()
    mnq_max = df_mnq['Close'].max()
    print(f"\n  PRICE CONTEXT:")
    print(f"    Prior year MNQ: {mnq_min:,.0f} – {mnq_max:,.0f}")
    print(f"    Dev period MNQ: ~20,000 – 24,700")
    print(f"    7pt TP as % of price: {7/mnq_min*100:.3f}% – {7/mnq_max*100:.3f}% (prior)")
    print(f"                          {7/20000*100:.3f}% – {7/24700*100:.3f}% (dev)")

    # --- Prepare MNQ arrays ---
    mnq_opens = df_mnq['Open'].values
    mnq_highs = df_mnq['High'].values
    mnq_lows = df_mnq['Low'].values
    mnq_closes = df_mnq['Close'].values
    mnq_sm = df_mnq['SM_Net'].values
    mnq_times = df_mnq.index

    # 5-min RSI mapping for MNQ (RSI len=8)
    df_mnq_5m = resample_to_5min(df_mnq)
    rsi_mnq_curr, rsi_mnq_prev = map_5min_rsi_to_1min(
        mnq_times.values, df_mnq_5m.index.values,
        df_mnq_5m['Close'].values, rsi_len=VSCALPA_RSI_LEN,
    )

    # --- Prepare MES arrays ---
    mes_opens = df_mes['Open'].values
    mes_highs = df_mes['High'].values
    mes_lows = df_mes['Low'].values
    mes_closes = df_mes['Close'].values
    mes_sm = df_mes['SM_Net'].values
    mes_times = df_mes.index

    df_mes_5m = resample_to_5min(df_mes)
    rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
        mes_times.values, df_mes_5m.index.values,
        df_mes_5m['Close'].values, rsi_len=MESV2_RSI_LEN,
    )

    # ===================================================================
    # 1. vScalpA (MNQ_V15)
    # ===================================================================
    print(f"\n{'='*80}")
    print("1. vScalpA (MNQ_V15) — SM(10/12/200/100) SM_T=0.0 RSI(8/60/40)")
    print("   SL=40 TP=7 CD=20 Entry cutoff 13:00 ET")
    print("=" * 80)

    trades_a = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm, mnq_times,
        rsi_mnq_curr, rsi_mnq_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPA_MAX_LOSS_PTS, tp_pts=VSCALPA_TP_PTS,
        entry_end_et=VSCALPA_ENTRY_END_ET,
    )
    sc_a = score_trades(trades_a, commission_per_side=MNQ_COMMISSION,
                        dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"  {fmt_score(sc_a, 'vScalpA')}")
    if sc_a:
        exits_str = "  ".join(f"{k}:{v}" for k, v in sc_a['exits'].items())
        print(f"    Exits: {exits_str}")

    # ===================================================================
    # 2. vScalpB (MNQ_VSCALPB)
    # ===================================================================
    print(f"\n{'='*80}")
    print("2. vScalpB (MNQ_VSCALPB) — SM(10/12/200/100) SM_T=0.25 RSI(8/55/45)")
    print("   SL=10 TP=3 CD=20 No entry cutoff")
    print("=" * 80)

    trades_b = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm, mnq_times,
        rsi_mnq_curr, rsi_mnq_prev,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
        # No entry cutoff for vScalpB (uses default 15:45)
    )
    sc_b = score_trades(trades_b, commission_per_side=MNQ_COMMISSION,
                        dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"  {fmt_score(sc_b, 'vScalpB')}")
    if sc_b:
        exits_str = "  ".join(f"{k}:{v}" for k, v in sc_b['exits'].items())
        print(f"    Exits: {exits_str}")

    # ===================================================================
    # 3. vScalpC (MNQ_VSCALPC) — Runner with partial exit
    # ===================================================================
    print(f"\n{'='*80}")
    print("3. vScalpC (MNQ_VSCALPC) — SM(10/12/200/100) SM_T=0.0 RSI(8/60/40)")
    print("   SL=40 TP1=7 TP2=25 CD=20 SL->BE after TP1, Entry cutoff 13:00 ET")
    print("   (Using TP2=25 pre-structure-exit config)")
    print("=" * 80)

    trades_c = run_backtest_partial_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm, mnq_times,
        rsi_mnq_curr, rsi_mnq_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD,
        cooldown_bars=VSCALPA_COOLDOWN,
        sl_pts=VSCALPC_SL_PTS, tp1_pts=VSCALPC_TP1_PTS, tp2_pts=VSCALPC_TP2_PTS,
        sl_to_be_after_tp1=True,
        be_time_bars=0,  # No BE_TIME for vScalpC (structure exit replaces it)
        entry_end_et=VSCALPC_ENTRY_END_ET,
    )
    sc_c = score_partial_trades(trades_c, dollar_per_pt=MNQ_DOLLAR_PER_PT,
                                 commission_per_side=MNQ_COMMISSION)
    if sc_c:
        print(f"  vScalpC  {sc_c['count']} trades, WR {sc_c['win_rate']}%, "
              f"PF {sc_c['pf']}, Net ${sc_c['net_dollar']:+.2f}, "
              f"MaxDD ${sc_c['max_dd_dollar']:.2f}, Sharpe {sc_c['sharpe']}")
        l1_str = "  ".join(f"{k}:{v}" for k, v in sc_c['leg1_exits'].items())
        l2_str = "  ".join(f"{k}:{v}" for k, v in sc_c['leg2_exits'].items())
        print(f"    Leg1 exits: {l1_str}")
        print(f"    Leg2 exits: {l2_str}")
    else:
        print("  vScalpC: NO TRADES")

    # ===================================================================
    # 4. MES v2 (MES_V2)
    # ===================================================================
    print(f"\n{'='*80}")
    print("4. MES v2 (MES_V2) — SM(20/12/400/255) SM_T=0.0 RSI(12/55/45)")
    print("   SL=35 TP=20 CD=25 BE_TIME=75 Entry cutoff 14:15 ET, EOD 16:00")
    print("=" * 80)

    trades_m = run_backtest_tp_exit(
        mes_opens, mes_highs, mes_lows, mes_closes, mes_sm, mes_times,
        rsi_mes_curr, rsi_mes_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        entry_end_et=MESV2_ENTRY_END_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
    )
    sc_m = score_trades(trades_m, commission_per_side=MES_COMMISSION,
                        dollar_per_pt=MES_DOLLAR_PER_PT)
    print(f"  {fmt_score(sc_m, 'MES v2')}")
    if sc_m:
        exits_str = "  ".join(f"{k}:{v}" for k, v in sc_m['exits'].items())
        print(f"    Exits: {exits_str}")

    # ===================================================================
    # 5. RSI Trendline (MNQ_RSI_TL) — Runner
    # ===================================================================
    print(f"\n{'='*80}")
    print("5. RSI Trendline (MNQ_RSI_TL) — RSI(8) trendline breakout")
    print("   SL=40 TP1=7 TP2=20 CD=30 Entry window 09:30-13:00 ET")
    print("   NO Smart Money filter — pure RSI trendline signal")
    print("=" * 80)

    # RSI TL uses 1-min RSI directly (not 5-min mapped)
    rsi_1min = compute_rsi_1min(mnq_closes, RSI_TL_RSI_LEN)
    long_sig, short_sig = generate_rsi_tl_signals(rsi_1min)
    print(f"  Raw signals: {long_sig.sum()} long + {short_sig.sum()} short = {long_sig.sum() + short_sig.sum()}")

    trades_tl = run_rsi_tl_backtest_runner(
        mnq_opens, mnq_closes, mnq_times,
        long_sig, short_sig,
        cooldown_bars=RSI_TL_CD,
        max_loss_pts=RSI_TL_SL,
        tp1_pts=RSI_TL_TP1,
        tp2_pts=RSI_TL_TP2,
        entry_start_et=RSI_TL_ENTRY_START_ET,
        entry_end_et=RSI_TL_ENTRY_END_ET,
    )
    sc_tl = score_rsi_tl_trades(trades_tl, commission_per_side=MNQ_COMMISSION,
                                 dollar_per_pt=MNQ_DOLLAR_PER_PT)
    if sc_tl:
        print(f"  RSI TL  {sc_tl['count']} trades, WR {sc_tl['win_rate']}%, "
              f"PF {sc_tl['pf']}, Net ${sc_tl['net_dollar']:+.2f}, "
              f"MaxDD ${sc_tl['max_dd_dollar']:.2f}, Sharpe {sc_tl['sharpe']}")
        exits_str = "  ".join(f"{k}:{v}" for k, v in sc_tl['exits'].items())
        print(f"    Exits: {exits_str}")
    else:
        print("  RSI TL: NO TRADES")

    # ===================================================================
    # COMPARISON TABLE
    # ===================================================================
    print(f"\n{'='*80}")
    print("SIDE-BY-SIDE: Prior Year (Feb 2024 – Feb 2025) vs Dev Period (Feb 2025 – Mar 2026)")
    print("=" * 80)

    # Dev period baselines from MEMORY.md / portfolio_backtest_mar14.md
    # Note: dev period results include gates; prior year has no gates
    dev_results = {
        "vScalpA":  {"count": 273, "wr": 85.3, "pf": 1.598, "sharpe": 2.885, "net": 2172, "maxdd": None,    "gates": "leledc+adr+vix"},
        "vScalpB":  {"count": 389, "wr": 73.5, "pf": 1.435, "sharpe": 2.151, "net": 1536, "maxdd": -281,    "gates": "none (ungated)"},
        "vScalpC":  {"count": 211, "wr": 79.6, "pf": 1.941, "sharpe": 3.932, "net": 5112, "maxdd": None,    "gates": "leledc+adr+atr+vix + structure exit"},
        "MES v2":   {"count": 384, "wr": None, "pf": 1.435, "sharpe": 2.283, "net": 4602, "maxdd": None,    "gates": "prior-day VPOC+VAL"},
        "RSI TL":   {"count": 2314, "wr": 71.8, "pf": 1.140, "sharpe": 0.699, "net": 7686, "maxdd": -3174, "gates": "none"},
    }

    prior_results = {}
    for name, sc in [("vScalpA", sc_a), ("vScalpB", sc_b), ("MES v2", sc_m)]:
        if sc:
            prior_results[name] = {
                "count": sc["count"], "wr": sc["win_rate"], "pf": sc["pf"],
                "sharpe": sc["sharpe"], "net": sc["net_dollar"],
                "maxdd": sc["max_dd_dollar"],
            }
        else:
            prior_results[name] = None

    if sc_c:
        prior_results["vScalpC"] = {
            "count": sc_c["count"], "wr": sc_c["win_rate"], "pf": sc_c["pf"],
            "sharpe": sc_c["sharpe"], "net": sc_c["net_dollar"],
            "maxdd": sc_c["max_dd_dollar"],
        }
    else:
        prior_results["vScalpC"] = None

    if sc_tl:
        prior_results["RSI TL"] = {
            "count": sc_tl["count"], "wr": sc_tl["win_rate"], "pf": sc_tl["pf"],
            "sharpe": sc_tl["sharpe"], "net": sc_tl["net_dollar"],
            "maxdd": sc_tl["max_dd_dollar"],
        }
    else:
        prior_results["RSI TL"] = None

    print(f"\n  {'Strategy':<12} | {'Period':<7} | {'Trades':>6} | {'WR%':>6} | {'PF':>6} | {'Sharpe':>7} | {'Net$':>10} | {'MaxDD$':>9} | Notes")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*10}-+-{'-'*9}-+-{'-'*20}")

    for name in ["vScalpA", "vScalpB", "vScalpC", "MES v2", "RSI TL"]:
        dev = dev_results[name]
        pri = prior_results.get(name)

        # Dev row
        dev_wr = f"{dev['wr']:.1f}" if dev['wr'] is not None else "N/A"
        dev_dd = f"${dev['maxdd']:,.0f}" if dev['maxdd'] is not None else "N/A"
        print(f"  {name:<12} | {'DEV':>7} | {dev['count']:>6} | {dev_wr:>6} | {dev['pf']:>6.3f} | {dev['sharpe']:>7.3f} | ${dev['net']:>+9,.0f} | {dev_dd:>9} | {dev['gates']}")

        # Prior row
        if pri:
            pri_dd = f"${pri['maxdd']:,.0f}" if pri['maxdd'] is not None else "N/A"
            print(f"  {'':12} | {'PRIOR':>7} | {pri['count']:>6} | {pri['wr']:>6.1f} | {pri['pf']:>6.3f} | {pri['sharpe']:>7.3f} | ${pri['net']:>+9,.0f} | {pri_dd:>9} | no gates")
        else:
            print(f"  {'':12} | {'PRIOR':>7} | {'NO TRADES':>6} |")

        # Verdict
        if pri and pri['pf'] > 1.0 and pri['net'] > 0:
            pf_ratio = pri['pf'] / dev['pf'] if dev['pf'] > 0 else 0
            if pf_ratio >= 0.9:
                verdict = "STRONG"
            elif pf_ratio >= 0.7:
                verdict = "HOLDS"
            elif pf_ratio >= 0.5:
                verdict = "MARGINAL"
            else:
                verdict = "WEAK"
            print(f"  {'':12} | {'VERDICT':>7} | PF ratio: {pf_ratio:.1%} — {verdict}")
        elif pri and pri['net'] > 0:
            print(f"  {'':12} | {'VERDICT':>7} | Profitable but PF < 1.0")
        elif pri:
            print(f"  {'':12} | {'VERDICT':>7} | NEGATIVE — strategy FAILS on prior year data")
        print()

    # ===================================================================
    # PORTFOLIO ANALYSIS
    # ===================================================================
    print(f"\n{'='*80}")
    print("PORTFOLIO — A(1) + B(1) + C(2) + MES(2)")
    print("=" * 80)

    daily_a = trades_to_daily_pnl(trades_a, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpA")
    daily_b = trades_to_daily_pnl(trades_b, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpB")
    daily_c = trades_to_daily_pnl(trades_c, MNQ_DOLLAR_PER_PT, MNQ_COMMISSION, "vScalpC",
                                   is_partial=True)
    daily_m = trades_to_daily_pnl(trades_m, MES_DOLLAR_PER_PT, MES_COMMISSION, "MES_v2")
    daily_tl = rsi_tl_trades_to_daily_pnl(trades_tl, "RSI_TL")

    # Combine all dates
    all_dates = set()
    for s in [daily_a, daily_b, daily_c, daily_m, daily_tl]:
        all_dates |= set(s.index)
    if not all_dates:
        print("  No trading days found!")
        return

    idx = pd.DatetimeIndex(sorted(all_dates))
    combined = pd.DataFrame(index=idx)
    combined["vScalpA"] = daily_a.reindex(idx, fill_value=0.0)
    combined["vScalpB"] = daily_b.reindex(idx, fill_value=0.0)
    combined["vScalpC"] = daily_c.reindex(idx, fill_value=0.0)
    combined["MES_v2"] = daily_m.reindex(idx, fill_value=0.0)
    combined["RSI_TL"] = daily_tl.reindex(idx, fill_value=0.0)

    # Portfolio = A(1) + B(1) + C(2 contracts already in partial exit P&L) + MES(2 single-contract)
    # Note: vScalpC partial exit already accounts for 2 contracts in P&L
    # MES v2 is single-contract in this backtest; multiply by 2 for target allocation
    combined["Portfolio_4strat"] = (combined["vScalpA"] + combined["vScalpB"] +
                                    combined["vScalpC"] + combined["MES_v2"] * 2)
    combined["Portfolio_5strat"] = (combined["vScalpA"] + combined["vScalpB"] +
                                    combined["vScalpC"] + combined["MES_v2"] * 2 +
                                    combined["RSI_TL"])

    # Monthly breakdown
    labels = ["vScalpA", "vScalpB", "vScalpC", "MES_v2", "RSI_TL", "Port4", "Port5"]
    monthly = combined.copy()
    monthly["Port4"] = combined["Portfolio_4strat"]
    monthly["Port5"] = combined["Portfolio_5strat"]
    monthly = monthly[labels].resample("ME").sum()
    monthly.index = monthly.index.strftime("%Y-%m")

    col_w = 10
    header_labels = [f"{l:>{col_w}}" for l in labels]
    print(f"\n  MONTHLY P&L BREAKDOWN")
    print(f"  {'Month':<10} {''.join(header_labels)}")
    print(f"  {'-'*10} {(' ' + '-'*col_w) * len(labels)}")

    for month in monthly.index:
        vals = [f"{monthly.loc[month, l]:>{col_w},.0f}" for l in labels]
        print(f"  {month:<10} {''.join(vals)}")

    # Totals
    print(f"  {'-'*10} {(' ' + '-'*col_w) * len(labels)}")
    totals = [f"{monthly[l].sum():>{col_w},.0f}" for l in labels]
    print(f"  {'TOTAL':<10} {''.join(totals)}")

    # Portfolio summary (4-strategy: A+B+C+MES)
    for port_label, port_col in [("4-strategy (A+B+C+MES)", "Portfolio_4strat"),
                                  ("5-strategy (+RSI TL)", "Portfolio_5strat")]:
        port_daily = combined[port_col]
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
        port_short = "Port4" if port_col == "Portfolio_4strat" else "Port5"
        positive_months = sum(1 for m in monthly.index if monthly.loc[m, port_short] > 0)

        print(f"\n  {port_label} PORTFOLIO SUMMARY:")
        print(f"    Total P&L:      ${total:>10,.2f}")
        print(f"    Sharpe:         {sharpe:>10.2f}")
        print(f"    PF (daily):     {pf:>10.3f}")
        print(f"    MaxDD:          ${mdd:>10,.2f}")
        print(f"    Daily WR:       {wr:>9.1f}%")
        print(f"    Trading Days:   {len(port_daily)}")
        print(f"    Positive Months: {positive_months}/{len(monthly)}")

    # ===================================================================
    # STRATEGY CORRELATIONS
    # ===================================================================
    print(f"\n{'='*80}")
    print("DAILY P&L CORRELATIONS")
    print("=" * 80)

    corr_df = combined[["vScalpA", "vScalpB", "vScalpC", "MES_v2", "RSI_TL"]].corr()
    print(f"\n  {'':>10}", end="")
    for col in corr_df.columns:
        print(f"  {col:>10}", end="")
    print()
    for row_name in corr_df.index:
        print(f"  {row_name:>10}", end="")
        for col in corr_df.columns:
            print(f"  {corr_df.loc[row_name, col]:>10.3f}", end="")
        print()

    # ===================================================================
    # FINAL VERDICT
    # ===================================================================
    print(f"\n{'='*80}")
    print("FINAL VERDICT — Prior Year Validation")
    print("=" * 80)

    strategies_passed = []
    strategies_failed = []
    strategies_marginal = []

    for name, sc_val in [("vScalpA", sc_a), ("vScalpB", sc_b), ("MES v2", sc_m)]:
        if sc_val and sc_val['net_dollar'] > 0 and sc_val['pf'] > 1.0:
            strategies_passed.append(name)
        elif sc_val and sc_val['net_dollar'] > 0:
            strategies_marginal.append(name)
        else:
            strategies_failed.append(name)

    if sc_c and sc_c['net_dollar'] > 0 and sc_c['pf'] > 1.0:
        strategies_passed.append("vScalpC")
    elif sc_c and sc_c['net_dollar'] > 0:
        strategies_marginal.append("vScalpC")
    else:
        strategies_failed.append("vScalpC")

    if sc_tl and sc_tl['net_dollar'] > 0 and sc_tl['pf'] > 1.0:
        strategies_passed.append("RSI TL")
    elif sc_tl and sc_tl['net_dollar'] > 0:
        strategies_marginal.append("RSI TL")
    else:
        strategies_failed.append("RSI TL")

    print(f"\n  PASSED (profitable, PF > 1.0): {', '.join(strategies_passed) if strategies_passed else 'NONE'}")
    print(f"  MARGINAL (profitable, PF < 1.0): {', '.join(strategies_marginal) if strategies_marginal else 'NONE'}")
    print(f"  FAILED (negative): {', '.join(strategies_failed) if strategies_failed else 'NONE'}")

    port4_total = combined["Portfolio_4strat"].sum()
    port5_total = combined["Portfolio_5strat"].sum()
    print(f"\n  4-strategy portfolio total: ${port4_total:+,.2f}")
    print(f"  5-strategy portfolio total: ${port5_total:+,.2f}")

    if port4_total > 0:
        print(f"\n  THE EDGE IS REAL: Portfolio profitable on completely unseen data.")
    else:
        print(f"\n  WARNING: Portfolio negative on prior year data. Possible overfit.")

    print(f"\n{'='*80}")
    print("DONE — Prior Year Validation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
