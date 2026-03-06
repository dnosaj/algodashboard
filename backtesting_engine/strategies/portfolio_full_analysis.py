"""
Comprehensive Portfolio Analysis
=================================
Runs all 3 active strategies (vScalpA, vScalpB, MES v2) on full Databento data
with ALL current production settings, then analyzes:

1. Baseline portfolio stats (full, IS, OOS)
2. Monthly P&L breakdown
3. Day-of-week performance (Friday exclusion)
4. Time cutoff analysis (1:00 PM, 12:00 PM, etc.)
5. VIX gate impact (19-22 on MNQ only)
6. Daily loss limit simulation + recovery analysis
7. Per-strategy and portfolio-level metrics

Usage:
    python3 portfolio_full_analysis.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))

from v10_test_common import (
    compute_rsi,
    compute_smart_money,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
    compute_et_minutes,
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
    MESV2_MAX_LOSS_PTS, MESV2_TP_PTS, MESV2_EOD_ET, MESV2_BREAKEVEN_BARS,
)


def score_portfolio(all_trades, label="Portfolio"):
    """Score combined portfolio trades."""
    if not all_trades:
        return None
    pnls = np.array([t.get("pnl_dollar", 0) for t in all_trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = wins.sum() if len(wins) else 0
    gross_loss = abs(losses.sum()) if len(losses) else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    wr = len(wins) / len(pnls) * 100 if len(pnls) else 0
    net = pnls.sum()

    # Sharpe on daily P&L
    dates = sorted(set(t["entry_date"] for t in all_trades))
    daily_pnl = []
    for d in dates:
        dp = sum(t["pnl_dollar"] for t in all_trades if t["entry_date"] == d)
        daily_pnl.append(dp)
    daily_pnl = np.array(daily_pnl)
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0

    # Max drawdown
    cum = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    print(f"  {label}: {len(pnls)} trades, WR {wr:.1f}%, PF {pf:.3f}, "
          f"${net:+,.0f}, Sharpe {sharpe:.2f}, MaxDD ${max_dd:,.0f}")
    return {"n": len(pnls), "wr": wr, "pf": pf, "net": net,
            "sharpe": sharpe, "max_dd": max_dd, "daily_pnl": daily_pnl,
            "dates": dates}


def add_trade_metadata(trades, strategy, dollar_per_pt, commission):
    """Add metadata fields to each trade dict."""
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
    _UTC = ZoneInfo("UTC")
    for t in trades:
        entry_time = t["entry_time"]
        # Convert from UTC to ET for time-of-day fields
        if entry_time.tzinfo is None:
            entry_et = entry_time.replace(tzinfo=_UTC).astimezone(_ET)
        else:
            entry_et = entry_time.astimezone(_ET)
        t["strategy"] = strategy
        t["entry_date"] = entry_et.strftime("%Y-%m-%d")
        t["entry_dow"] = entry_et.strftime("%A")
        t["entry_hour"] = entry_et.hour
        t["entry_minute"] = entry_et.minute
        t["entry_et_minutes"] = entry_et.hour * 60 + entry_et.minute
        pnl_pts = t["pts"]
        t["pnl_dollar"] = pnl_pts * dollar_per_pt - 2 * commission


def fetch_vix_history():
    """Fetch VIX daily close history for the backtest period."""
    try:
        import yfinance as yf
        import socket
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(15)
        try:
            vix = yf.download("^VIX", start="2025-02-01", end="2026-03-07", progress=False)
        finally:
            socket.setdefaulttimeout(old_timeout)
        if vix.empty:
            print("  WARNING: No VIX data returned")
            return {}
        # Build date → prior-day close mapping
        closes = vix["Close"].squeeze()
        vix_map = {}
        dates = closes.index.tolist()
        for i in range(1, len(dates)):
            # For each trading day, prior-day VIX close is the previous entry
            d = dates[i].strftime("%Y-%m-%d")
            prev_close = float(closes.iloc[i-1].item() if hasattr(closes.iloc[i-1], 'item') else closes.iloc[i-1])
            vix_map[d] = prev_close
        # Also map the first date
        d0 = dates[0].strftime("%Y-%m-%d")
        vix_map[d0] = float(closes.iloc[0].item() if hasattr(closes.iloc[0], 'item') else closes.iloc[0])
        return vix_map
    except Exception as e:
        print(f"  WARNING: VIX fetch failed: {e}")
        return {}


def apply_vix_gate(trades, vix_map, vix_min=19.0, vix_max=22.0):
    """Filter out trades whose entry date falls in VIX death zone."""
    allowed = []
    blocked = []
    for t in trades:
        vix = vix_map.get(t["entry_date"])
        t["vix_prev_close"] = vix
        if vix is not None and vix_min <= vix <= vix_max:
            blocked.append(t)
        else:
            allowed.append(t)
    return allowed, blocked


def simulate_daily_loss_limit(trades, limit_per_strategy):
    """Simulate daily P&L limits. Returns (surviving_trades, blocked_trades).

    limit_per_strategy: dict of strategy_name → max_daily_loss (positive number).
    Walks through trades in chronological order per strategy per day.
    Once cumulative daily loss exceeds limit, remaining trades for that
    strategy that day are blocked.
    """
    # Sort by entry time
    sorted_trades = sorted(trades, key=lambda t: t["entry_time"])

    surviving = []
    blocked = []

    # Track daily P&L per strategy
    daily_pnl = defaultdict(lambda: defaultdict(float))  # date → strategy → cumulative
    daily_paused = defaultdict(set)  # date → set of paused strategies

    for t in sorted_trades:
        date = t["entry_date"]
        strat = t["strategy"]
        limit = limit_per_strategy.get(strat, 0)

        if limit > 0 and strat in daily_paused[date]:
            blocked.append(t)
            continue

        surviving.append(t)
        daily_pnl[date][strat] += t["pnl_dollar"]

        if limit > 0 and daily_pnl[date][strat] <= -limit:
            daily_paused[date].add(strat)

    return surviving, blocked


def print_monthly_breakdown(trades, label=""):
    """Print monthly P&L table."""
    months = defaultdict(list)
    for t in trades:
        m = t["entry_date"][:7]  # YYYY-MM
        months[m].append(t["pnl_dollar"])

    print(f"\n  Monthly Breakdown{' — ' + label if label else ''}:")
    print(f"  {'Month':<10} {'Trades':>6} {'Net $':>10} {'Win%':>6} {'PF':>6}")
    print(f"  {'-'*42}")
    for m in sorted(months.keys()):
        pnls = np.array(months[m])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gp = wins.sum() if len(wins) else 0
        gl = abs(losses.sum()) if len(losses) else 0.001
        pf = gp / gl if gl > 0 else 999
        wr = len(wins) / len(pnls) * 100
        print(f"  {m:<10} {len(pnls):>6} {pnls.sum():>+10,.0f} {wr:>5.1f}% {pf:>5.2f}")


def print_dow_breakdown(trades, label=""):
    """Print day-of-week P&L table."""
    days = defaultdict(list)
    for t in trades:
        days[t["entry_dow"]].append(t["pnl_dollar"])

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print(f"\n  Day-of-Week Breakdown{' — ' + label if label else ''}:")
    print(f"  {'Day':<12} {'Trades':>6} {'Net $':>10} {'Win%':>6} {'PF':>6} {'Avg $':>8}")
    print(f"  {'-'*52}")
    for d in order:
        if d not in days:
            continue
        pnls = np.array(days[d])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gp = wins.sum() if len(wins) else 0
        gl = abs(losses.sum()) if len(losses) else 0.001
        pf = gp / gl if gl > 0 else 999
        wr = len(wins) / len(pnls) * 100
        print(f"  {d:<12} {len(pnls):>6} {pnls.sum():>+10,.0f} {wr:>5.1f}% {pf:>5.2f} {pnls.mean():>+7,.1f}")


def analyze_time_cutoff(all_trades, cutoffs_et):
    """Analyze portfolio performance with different entry cutoff times."""
    print(f"\n  Entry Cutoff Analysis (all strategies):")
    print(f"  {'Cutoff':>8} {'Trades':>7} {'Net $':>10} {'PF':>6} {'Sharpe':>7} {'Blocked':>8}")
    print(f"  {'-'*52}")

    baseline = len(all_trades)
    for cutoff_min in cutoffs_et:
        h = cutoff_min // 60
        m = cutoff_min % 60
        label = f"{h:02d}:{m:02d} ET"

        kept = [t for t in all_trades if t["entry_et_minutes"] < cutoff_min]
        if not kept:
            print(f"  {label:>8} {0:>7} {'N/A':>10}")
            continue

        pnls = np.array([t["pnl_dollar"] for t in kept])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gp = wins.sum() if len(wins) else 0
        gl = abs(losses.sum()) if len(losses) else 0.001
        pf = gp / gl if gl > 0 else 999

        dates = sorted(set(t["entry_date"] for t in kept))
        daily = [sum(t["pnl_dollar"] for t in kept if t["entry_date"] == d) for d in dates]
        daily = np.array(daily)
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

        blocked = baseline - len(kept)
        print(f"  {label:>8} {len(kept):>7} {pnls.sum():>+10,.0f} {pf:>5.2f} {sharpe:>6.2f} {blocked:>8}")


def analyze_daily_loss_recovery(trades):
    """Analyze: when portfolio is down $X intraday, does it recover?"""
    # Group trades by date, sort within day
    daily_trades = defaultdict(list)
    for t in trades:
        daily_trades[t["entry_date"]].append(t)

    thresholds = [100, 150, 200, 250, 300]

    print(f"\n  Daily Loss Recovery Analysis:")
    print(f"  Question: When the portfolio is down $X during the day, what happens after?")
    print(f"  {'Threshold':>10} {'Days Hit':>9} {'Recovered':>10} {'Avg Recovery':>13} {'End-of-Day Avg':>15}")
    print(f"  {'-'*62}")

    for thresh in thresholds:
        days_hit = 0
        recoveries = 0
        recovery_amounts = []
        eod_pnls = []

        for date, day_trades in sorted(daily_trades.items()):
            day_trades.sort(key=lambda t: t["entry_time"])
            cum = 0
            hit_threshold = False
            cum_at_threshold = 0

            for t in day_trades:
                cum += t["pnl_dollar"]
                if not hit_threshold and cum <= -thresh:
                    hit_threshold = True
                    cum_at_threshold = cum
                    days_hit += 1

            if hit_threshold:
                eod_pnls.append(cum)
                recovery = cum - cum_at_threshold
                recovery_amounts.append(recovery)
                if cum > cum_at_threshold:
                    recoveries += 1

        if days_hit == 0:
            print(f"  ${thresh:>8} {'0':>9}")
            continue

        avg_recovery = np.mean(recovery_amounts)
        avg_eod = np.mean(eod_pnls)
        print(f"  ${thresh:>8} {days_hit:>9} {recoveries:>9} ({recoveries/days_hit*100:.0f}%) "
              f"${avg_recovery:>+10,.0f}  ${avg_eod:>+12,.0f}")


def main():
    print("=" * 74)
    print("  COMPREHENSIVE PORTFOLIO ANALYSIS")
    print("  vScalpA (V15) + vScalpB + MES v2 — All Current Settings")
    print("=" * 74)

    # ─── Load data ───
    print("\n[1] Loading data...")
    df_mnq = load_instrument_1min("MNQ")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm
    print(f"  MNQ: {len(df_mnq):,} bars, {df_mnq.index[0].date()} → {df_mnq.index[-1].date()}")

    df_mes = load_instrument_1min("MES")
    mes_sm = compute_smart_money(
        df_mes["Close"].values, df_mes["Volume"].values,
        index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
        norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
    )
    df_mes["SM_Net"] = mes_sm
    print(f"  MES: {len(df_mes):,} bars, {df_mes.index[0].date()} → {df_mes.index[-1].date()}")

    # ─── Run strategies ───
    print("\n[2] Running strategies...")

    # MNQ arrays + RSI
    mnq_opens = df_mnq["Open"].values
    mnq_highs = df_mnq["High"].values
    mnq_lows = df_mnq["Low"].values
    mnq_closes = df_mnq["Close"].values
    mnq_sm_arr = df_mnq["SM_Net"].values
    mnq_times = df_mnq.index

    df_mnq_5m = resample_to_5min(df_mnq)
    rsi_mnq_a_curr, rsi_mnq_a_prev = map_5min_rsi_to_1min(
        mnq_times.values, df_mnq_5m.index.values,
        df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN,
    )

    # vScalpA
    trades_a = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
        rsi_mnq_a_curr, rsi_mnq_a_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPA_MAX_LOSS_PTS, tp_pts=VSCALPA_TP_PTS,
        entry_end_et=VSCALPA_ENTRY_END_ET,
    )
    add_trade_metadata(trades_a, "vScalpA", MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
    print(f"  vScalpA: {len(trades_a)} trades")

    # vScalpB (same RSI len as A, but different thresholds)
    rsi_mnq_b_curr, rsi_mnq_b_prev = map_5min_rsi_to_1min(
        mnq_times.values, df_mnq_5m.index.values,
        df_mnq_5m["Close"].values, rsi_len=VSCALPB_RSI_LEN,
    )
    trades_b = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm_arr, mnq_times,
        rsi_mnq_b_curr, rsi_mnq_b_prev,
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
    )
    add_trade_metadata(trades_b, "vScalpB", MNQ_DOLLAR_PER_PT, MNQ_COMMISSION)
    print(f"  vScalpB: {len(trades_b)} trades")

    # MES v2
    mes_opens = df_mes["Open"].values
    mes_highs = df_mes["High"].values
    mes_lows = df_mes["Low"].values
    mes_closes = df_mes["Close"].values
    mes_sm_arr = df_mes["SM_Net"].values
    mes_times = df_mes.index

    df_mes_5m = resample_to_5min(df_mes)
    rsi_mes_curr, rsi_mes_prev = map_5min_rsi_to_1min(
        mes_times.values, df_mes_5m.index.values,
        df_mes_5m["Close"].values, rsi_len=MESV2_RSI_LEN,
    )
    trades_m = run_backtest_tp_exit(
        mes_opens, mes_highs, mes_lows, mes_closes, mes_sm_arr, mes_times,
        rsi_mes_curr, rsi_mes_prev,
        rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
        sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
        max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
        eod_minutes_et=MESV2_EOD_ET,
        breakeven_after_bars=MESV2_BREAKEVEN_BARS,
    )
    add_trade_metadata(trades_m, "MES_v2", MES_DOLLAR_PER_PT, MES_COMMISSION)
    print(f"  MES v2:  {len(trades_m)} trades")

    all_trades = trades_a + trades_b + trades_m

    # ═══════════════════════════════════════════════════════════════════════
    # [3] BASELINE PORTFOLIO STATS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print("  [3] BASELINE PORTFOLIO STATS (no VIX gate, no daily loss limit)")
    print(f"{'='*74}")

    score_portfolio(trades_a, "vScalpA (V15)")
    score_portfolio(trades_b, "vScalpB")
    score_portfolio(trades_m, "MES v2")
    port_sc = score_portfolio(all_trades, "PORTFOLIO A(1)+B(1)+MES(1)")

    # ─── IS / OOS split ───
    all_dates = sorted(set(t["entry_date"] for t in all_trades))
    mid_date = all_dates[len(all_dates) // 2]
    print(f"\n  IS/OOS split at {mid_date}")

    is_trades = [t for t in all_trades if t["entry_date"] < mid_date]
    oos_trades = [t for t in all_trades if t["entry_date"] >= mid_date]
    score_portfolio(is_trades, "PORTFOLIO IS")
    score_portfolio(oos_trades, "PORTFOLIO OOS")

    # Per-strategy IS/OOS
    for name, trades in [("vScalpA", trades_a), ("vScalpB", trades_b), ("MES_v2", trades_m)]:
        is_t = [t for t in trades if t["entry_date"] < mid_date]
        oos_t = [t for t in trades if t["entry_date"] >= mid_date]
        print(f"  --- {name} ---")
        score_portfolio(is_t, f"  {name} IS")
        score_portfolio(oos_t, f"  {name} OOS")

    # ═══════════════════════════════════════════════════════════════════════
    # [4] MONTHLY BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print("  [4] MONTHLY BREAKDOWN")
    print(f"{'='*74}")
    print_monthly_breakdown(all_trades, "Portfolio")
    for name, trades in [("vScalpA", trades_a), ("vScalpB", trades_b), ("MES_v2", trades_m)]:
        print_monthly_breakdown(trades, name)

    # ═══════════════════════════════════════════════════════════════════════
    # [5] DAY-OF-WEEK ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print("  [5] DAY-OF-WEEK ANALYSIS")
    print(f"{'='*74}")
    print_dow_breakdown(all_trades, "Portfolio")
    for name, trades in [("vScalpA", trades_a), ("vScalpB", trades_b), ("MES_v2", trades_m)]:
        print_dow_breakdown(trades, name)

    # Friday exclusion
    print(f"\n  --- What if we skip Fridays? ---")
    no_fri = [t for t in all_trades if t["entry_dow"] != "Friday"]
    score_portfolio(no_fri, "Portfolio (no Fridays)")
    score_portfolio(all_trades, "Portfolio (with Fridays)")

    fri_trades = [t for t in all_trades if t["entry_dow"] == "Friday"]
    fri_pnl = sum(t["pnl_dollar"] for t in fri_trades)
    print(f"  Friday net P&L: ${fri_pnl:+,.0f} from {len(fri_trades)} trades")

    for name, trades in [("vScalpA", trades_a), ("vScalpB", trades_b), ("MES_v2", trades_m)]:
        fri_t = [t for t in trades if t["entry_dow"] == "Friday"]
        no_fri_t = [t for t in trades if t["entry_dow"] != "Friday"]
        if fri_t:
            fri_p = sum(t["pnl_dollar"] for t in fri_t)
            print(f"  {name} Friday: {len(fri_t)} trades, ${fri_p:+,.0f}")

    # ═══════════════════════════════════════════════════════════════════════
    # [6] TIME CUTOFF ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print("  [6] TIME CUTOFF ANALYSIS")
    print(f"{'='*74}")
    cutoffs = [11*60, 11*60+30, 12*60, 12*60+30, 13*60, 13*60+30, 14*60, 14*60+30, 15*60, 15*60+30, 16*60]
    analyze_time_cutoff(all_trades, cutoffs)

    # Per-strategy cutoff
    for name, trades in [("vScalpA", trades_a), ("vScalpB", trades_b), ("MES_v2", trades_m)]:
        print(f"\n  {name}:")
        analyze_time_cutoff(trades, cutoffs)

    # ═══════════════════════════════════════════════════════════════════════
    # [7] VIX GATE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print("  [7] VIX GATE ANALYSIS")
    print(f"{'='*74}")

    print("\n  Fetching VIX history...")
    vix_map = fetch_vix_history()
    if vix_map:
        print(f"  VIX data: {len(vix_map)} trading days")

        # Assign VIX to all trades
        for t in all_trades:
            t["vix_prev_close"] = vix_map.get(t["entry_date"])

        # Current gate: 19-22 on MNQ only
        mnq_trades = trades_a + trades_b
        mnq_allowed, mnq_blocked = apply_vix_gate(mnq_trades, vix_map, 19.0, 22.0)
        vix_gated = mnq_allowed + trades_m  # MES unaffected
        print(f"\n  VIX 19-22 gate on MNQ only (current production config):")
        print(f"  Blocked: {len(mnq_blocked)} MNQ trades")
        score_portfolio(vix_gated, "Portfolio (VIX gated)")
        score_portfolio(all_trades, "Portfolio (no gate)")

        # What if gate >19 (block everything above 19)?
        mnq_gt19, mnq_gt19_blocked = apply_vix_gate(mnq_trades, vix_map, 19.0, 999.0)
        gt19_portfolio = mnq_gt19 + trades_m
        print(f"\n  VIX >19 gate on MNQ (alternative):")
        print(f"  Blocked: {len(mnq_gt19_blocked)} MNQ trades")
        score_portfolio(gt19_portfolio, "Portfolio (VIX >19 gate)")

        # What about gating MES too?
        all_allowed, all_blocked = apply_vix_gate(all_trades, vix_map, 19.0, 22.0)
        print(f"\n  VIX 19-22 gate on ALL strategies:")
        print(f"  Blocked: {len(all_blocked)} trades")
        score_portfolio(all_allowed, "Portfolio (all gated 19-22)")

        # VIX bucket analysis for portfolio
        print(f"\n  Portfolio P&L by VIX Bucket:")
        buckets = [(0, 14), (14, 17), (17, 19), (19, 22), (22, 25), (25, 30), (30, 100)]
        print(f"  {'VIX Range':<12} {'Trades':>7} {'Net $':>10} {'PF':>6} {'Win%':>6} {'Days':>5}")
        print(f"  {'-'*50}")
        for lo, hi in buckets:
            bt = [t for t in all_trades
                  if t.get("vix_prev_close") is not None and lo <= t["vix_prev_close"] < hi]
            if not bt:
                continue
            pnls = np.array([t["pnl_dollar"] for t in bt])
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            gp = wins.sum() if len(wins) else 0
            gl = abs(losses.sum()) if len(losses) else 0.001
            pf = gp / gl if gl > 0 else 999
            wr = len(wins) / len(pnls) * 100
            days = len(set(t["entry_date"] for t in bt))
            label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
            print(f"  {label:<12} {len(bt):>7} {pnls.sum():>+10,.0f} {pf:>5.2f} {wr:>5.1f}% {days:>5}")
    else:
        print("  Skipping VIX analysis (no data)")

    # ═══════════════════════════════════════════════════════════════════════
    # [8] DAILY LOSS LIMIT SIMULATION
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print("  [8] DAILY LOSS LIMIT SIMULATION")
    print(f"{'='*74}")

    limits = {
        "vScalpA": 100,
        "vScalpB": 100,
        "MES_v2": 200,
    }
    surviving, blocked = simulate_daily_loss_limit(all_trades, limits)
    print(f"\n  Per-strategy limits: vScalpA=$100, vScalpB=$100, MES_v2=$200")
    print(f"  Trades blocked by daily loss limit: {len(blocked)}")
    score_portfolio(surviving, "Portfolio (with daily limits)")
    score_portfolio(all_trades, "Portfolio (no daily limits)")

    if blocked:
        blocked_pnl = sum(t["pnl_dollar"] for t in blocked)
        print(f"  Blocked trades net P&L: ${blocked_pnl:+,.0f} ({len(blocked)} trades)")
        # Were the blocked trades net negative? If so, limit helps
        blocked_wins = len([t for t in blocked if t["pnl_dollar"] > 0])
        blocked_losses = len([t for t in blocked if t["pnl_dollar"] <= 0])
        print(f"  Blocked winners: {blocked_wins}, losers: {blocked_losses}")

    # ─── Recovery analysis ───
    analyze_daily_loss_recovery(all_trades)

    # Per-strategy recovery
    for name, trades in [("vScalpA", trades_a), ("vScalpB", trades_b), ("MES_v2", trades_m)]:
        print(f"\n  --- {name} ---")
        analyze_daily_loss_recovery(trades)

    # ═══════════════════════════════════════════════════════════════════════
    # [9] COMBINED: VIX GATE + DAILY LIMITS + PRODUCTION SETTINGS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*74}")
    print("  [9] FULL PRODUCTION CONFIG (VIX gate + daily limits)")
    print(f"{'='*74}")

    if vix_map:
        # Apply VIX gate to MNQ strategies
        a_allowed, _ = apply_vix_gate(trades_a, vix_map, 19.0, 22.0)
        b_allowed, _ = apply_vix_gate(trades_b, vix_map, 19.0, 22.0)
        prod_trades = a_allowed + b_allowed + trades_m

        # Then apply daily loss limits
        prod_surviving, prod_blocked = simulate_daily_loss_limit(prod_trades, limits)

        print(f"  After VIX gate (19-22 MNQ): {len(prod_trades)} trades")
        print(f"  After daily loss limits: {len(prod_surviving)} trades")
        score_portfolio(prod_surviving, "PRODUCTION PORTFOLIO")
        score_portfolio(all_trades, "Baseline (no gates)")

        # IS/OOS on production
        prod_is = [t for t in prod_surviving if t["entry_date"] < mid_date]
        prod_oos = [t for t in prod_surviving if t["entry_date"] >= mid_date]
        score_portfolio(prod_is, "PRODUCTION IS")
        score_portfolio(prod_oos, "PRODUCTION OOS")

    print(f"\n{'='*74}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()
