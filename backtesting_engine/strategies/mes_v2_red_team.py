#!/usr/bin/env python3
"""
MES v2 Gate Red Team — Mar 11, 2026
=====================================
Forensic validation of VIX [20-25] and Entry Delay +30min findings.

Red team questions:
  1. Which SPECIFIC trades does each gate block? Are they actually losers?
  2. Does the PF improvement come from blocking bad trades or random variance?
  3. Is the VIX [20-25] band a real edge or just noise in a small sample?
  4. Does the entry delay capture the "first 11 min SL" pattern?
  5. Do both gates stack without killing trade count?
  6. What does the P&L distribution of blocked vs kept trades look like?
  7. Is there calendar clustering (same few bad days driving everything)?

Usage:
    cd backtesting_engine/strategies && python3 mes_v2_red_team.py
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

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
from mes_v2_comprehensive_gate_sweep import (
    MES_STRAT,
    prepare_mes_data,
    run_mes,
    slice_gate,
    build_vix_gate,
    build_entry_delay_gate,
)

_ET = ZoneInfo("America/New_York")


def get_trade_details(trades, times, dollar_per_pt=5.0, commission=1.25):
    """Extract detailed trade info for analysis."""
    details = []
    for t in trades:
        entry_idx = t["entry_idx"]
        exit_idx = t["exit_idx"]
        entry_time_utc = pd.Timestamp(t["entry_time"])
        exit_time_utc = pd.Timestamp(t["exit_time"])

        if entry_time_utc.tz is None:
            entry_time_et = entry_time_utc.tz_localize("UTC").tz_convert(_ET)
            exit_time_et = exit_time_utc.tz_localize("UTC").tz_convert(_ET)
        else:
            entry_time_et = entry_time_utc.tz_convert(_ET)
            exit_time_et = exit_time_utc.tz_convert(_ET)

        pts = float(t["pts"])
        side = t["side"].upper()
        raw_pnl = pts * dollar_per_pt
        net_pnl = raw_pnl - 2 * commission  # round-trip

        details.append({
            "entry_bar": entry_idx,
            "exit_bar": exit_idx,
            "entry_time_et": entry_time_et,
            "exit_time_et": exit_time_et,
            "entry_date": entry_time_et.date(),
            "entry_hour": entry_time_et.hour,
            "entry_minute": entry_time_et.minute,
            "entry_hhmm": f"{entry_time_et.hour:02d}:{entry_time_et.minute:02d}",
            "side": side,
            "entry_price": float(t["entry"]),
            "exit_price": float(t["exit"]),
            "pts": pts,
            "raw_pnl": raw_pnl,
            "net_pnl": net_pnl,
            "exit_reason": t.get("result", "unknown"),
            "mfe_pts": float(t.get("mfe", 0) or 0),
            "mae_pts": float(t.get("mae", 0) or 0),
            "bars_held": t.get("bars", exit_idx - entry_idx),
            "is_win": net_pnl > 0,
        })
    return details


def get_vix_lookup(df):
    """Build VIX prior-day close lookup."""
    import yfinance as yf

    times_et = df.index.tz_localize("UTC").tz_convert(_ET) if df.index.tz is None else df.index.tz_convert(_ET)
    start_date = times_et[0].strftime("%Y-%m-%d")
    end_date = (times_et[-1] + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"Downloading VIX data ({start_date} to {end_date})...")
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix["date"] = vix.index.date
    vix_closes = dict(zip(vix["date"], vix["Close"].values))
    sorted_dates = sorted(vix_closes.keys())

    def prior_vix(trade_date):
        for vd in reversed(sorted_dates):
            if vd < trade_date:
                return float(vix_closes[vd])
        return None

    return prior_vix, vix_closes


def analyze_blocked_trades(all_trades, gate_array, is_len, split_name, gate_name):
    """Classify each trade as blocked or kept by the gate, return stats."""
    blocked = []
    kept = []

    for t in all_trades:
        entry_bar = t["entry_bar"]
        # Map to split-relative index if needed
        if split_name == "OOS":
            gate_idx = entry_bar  # gate is already sliced to OOS
        else:
            gate_idx = entry_bar

        if gate_idx < len(gate_array) and not gate_array[gate_idx]:
            blocked.append(t)
        else:
            kept.append(t)

    return blocked, kept


def print_trade_table(trades, label, max_show=50):
    """Print a compact trade table."""
    if not trades:
        print(f"  {label}: 0 trades")
        return

    wins = sum(1 for t in trades if t["is_win"])
    losses = len(trades) - wins
    total_pnl = sum(t["net_pnl"] for t in trades)
    avg_pnl = total_pnl / len(trades)
    gross_win = sum(t["net_pnl"] for t in trades if t["is_win"])
    gross_loss = abs(sum(t["net_pnl"] for t in trades if not t["is_win"]))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    print(f"\n  {label}: {len(trades)} trades, {wins}W/{losses}L, WR {wins/len(trades)*100:.1f}%, "
          f"PF {pf:.3f}, Net ${total_pnl:+.0f}, Avg ${avg_pnl:+.1f}")
    print(f"  {'Date':>12} {'Time':>5} {'Side':>5} {'Entry':>8} {'Exit':>8} {'Pts':>6} {'P&L':>8} {'Exit':>6} {'Bars':>4} {'MFE':>6} {'MAE':>6}")
    print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*4} {'-'*6} {'-'*6}")

    for t in trades[:max_show]:
        print(f"  {str(t['entry_date']):>12} {t['entry_hhmm']:>5} {t['side']:>5} "
              f"{t['entry_price']:>8.1f} {t['exit_price']:>8.1f} {t['pts']:>+6.1f} "
              f"${t['net_pnl']:>+7.0f} {t['exit_reason']:>6} {t['bars_held']:>4} "
              f"{t['mfe_pts']:>+6.1f} {t['mae_pts']:>+6.1f}")
    if len(trades) > max_show:
        print(f"  ... ({len(trades) - max_show} more)")


def red_team_vix_gate(df, splits, is_len, all_full_trades_detail, prior_vix_fn):
    """Deep forensic analysis of VIX [20-25] gate."""
    print("\n" + "=" * 120)
    print("RED TEAM #1: VIX [20-25] GATE")
    print("=" * 120)

    # 1. What VIX was on each trading day with trades
    print("\n  --- VIX values on ALL MES trade dates ---")
    trade_dates = sorted(set(t["entry_date"] for t in all_full_trades_detail))
    vix_by_date = {}
    for d in trade_dates:
        vix_val = prior_vix_fn(d)
        vix_by_date[d] = vix_val

    # Show VIX distribution
    vix_vals = [v for v in vix_by_date.values() if v is not None]
    print(f"  VIX range: {min(vix_vals):.1f} — {max(vix_vals):.1f}")
    print(f"  VIX mean: {np.mean(vix_vals):.1f}, median: {np.median(vix_vals):.1f}")

    # Count days in 20-25 zone
    zone_days = [d for d, v in vix_by_date.items() if v is not None and 20 <= v <= 25]
    non_zone_days = [d for d, v in vix_by_date.items() if v is not None and not (20 <= v <= 25)]
    print(f"  Days with VIX 20-25: {len(zone_days)} / {len(trade_dates)} ({len(zone_days)/len(trade_dates)*100:.0f}%)")

    # 2. P&L on VIX 20-25 days vs other days
    zone_trades = [t for t in all_full_trades_detail if t["entry_date"] in zone_days]
    non_zone_trades = [t for t in all_full_trades_detail if t["entry_date"] in non_zone_days]

    print_trade_table(zone_trades, "BLOCKED by VIX [20-25] (these trades would be removed)")
    print_trade_table(non_zone_trades, "KEPT (VIX outside [20-25])")

    # 3. Daily P&L breakdown on VIX zone days
    print("\n  --- Daily P&L on VIX [20-25] days ---")
    print(f"  {'Date':>12} {'VIX':>6} {'Trades':>6} {'Net P&L':>8} {'W/L':>5} {'Avg':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*5} {'-'*8}")

    zone_day_pnl = {}
    for d in sorted(zone_days):
        day_trades = [t for t in zone_trades if t["entry_date"] == d]
        day_pnl = sum(t["net_pnl"] for t in day_trades)
        day_wins = sum(1 for t in day_trades if t["is_win"])
        avg = day_pnl / len(day_trades) if day_trades else 0
        zone_day_pnl[d] = day_pnl
        vix_val = vix_by_date.get(d)
        print(f"  {str(d):>12} {vix_val:>6.1f} {len(day_trades):>6} ${day_pnl:>+7.0f} "
              f"{day_wins}/{len(day_trades)-day_wins} ${avg:>+7.1f}")

    winning_zone_days = sum(1 for p in zone_day_pnl.values() if p > 0)
    losing_zone_days = sum(1 for p in zone_day_pnl.values() if p <= 0)
    print(f"\n  VIX zone days: {winning_zone_days} profitable, {losing_zone_days} losing")
    print(f"  Total zone P&L: ${sum(zone_day_pnl.values()):+.0f}")

    # 4. KEY RED TEAM CHECK: Is the zone P&L driven by a few outlier days?
    if zone_day_pnl:
        sorted_zone_pnl = sorted(zone_day_pnl.values())
        print(f"\n  Worst zone day: ${sorted_zone_pnl[0]:+.0f}")
        print(f"  Best zone day: ${sorted_zone_pnl[-1]:+.0f}")
        if len(sorted_zone_pnl) > 2:
            trimmed = sorted_zone_pnl[1:-1]  # Remove best and worst
            print(f"  P&L without best/worst zone day: ${sum(trimmed):+.0f} ({len(trimmed)} days)")
            # Would the gate still help if we remove the worst zone day (most favorable to the gate)?
            without_worst = sorted_zone_pnl[1:]
            print(f"  P&L without worst zone day ONLY: ${sum(without_worst):+.0f} (robustness check)")

    # 5. Is-Out-of-Sample split of VIX zone trades
    print("\n  --- IS vs OOS breakdown of VIX zone trades ---")
    mid_date = sorted(trade_dates)[len(trade_dates) // 2]
    is_zone = [t for t in zone_trades if t["entry_date"] < mid_date]
    oos_zone = [t for t in zone_trades if t["entry_date"] >= mid_date]
    print_trade_table(is_zone, f"IS zone trades (before {mid_date})", max_show=30)
    print_trade_table(oos_zone, f"OOS zone trades (on/after {mid_date})", max_show=30)

    # 6. Bootstrap significance test
    print("\n  --- Bootstrap significance test ---")
    if zone_trades:
        zone_mean = np.mean([t["net_pnl"] for t in zone_trades])
        all_pnl = [t["net_pnl"] for t in all_full_trades_detail]
        n_zone = len(zone_trades)

        np.random.seed(42)
        n_boot = 10000
        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(all_pnl, size=n_zone, replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.array(boot_means)
        pct_worse = np.mean(boot_means <= zone_mean) * 100
        print(f"  Zone avg P&L: ${zone_mean:+.1f}")
        print(f"  Random {n_zone}-trade sample avg P&L: ${np.mean(boot_means):+.1f} ± ${np.std(boot_means):.1f}")
        print(f"  {pct_worse:.1f}% of random samples are worse than the zone")
        if pct_worse < 5:
            print(f"  >>> STATISTICALLY SIGNIFICANT: VIX zone trades are genuinely worse (p < 0.05)")
        elif pct_worse < 10:
            print(f"  >>> MARGINALLY SIGNIFICANT: VIX zone trades are somewhat worse (p < 0.10)")
        else:
            print(f"  >>> NOT SIGNIFICANT: VIX zone P&L could be random (p = {pct_worse/100:.2f})")

    return zone_trades, non_zone_trades


def red_team_entry_delay(df, splits, is_len, all_full_trades_detail):
    """Deep forensic analysis of entry delay +30min gate."""
    print("\n" + "=" * 120)
    print("RED TEAM #2: ENTRY DELAY +30min (block 10:00-10:30 ET)")
    print("=" * 120)

    # 1. Classify trades by entry time
    early_trades = [t for t in all_full_trades_detail
                    if t["entry_hour"] == 10 and t["entry_minute"] < 30]
    later_trades = [t for t in all_full_trades_detail
                    if not (t["entry_hour"] == 10 and t["entry_minute"] < 30)]

    print_trade_table(early_trades, "BLOCKED by delay +30min (entries 10:00-10:29 ET)")
    print_trade_table(later_trades, "KEPT (entries 10:30+ ET)")

    # 2. Entry time histogram
    print("\n  --- Entry time distribution (30-min buckets) ---")
    buckets = defaultdict(list)
    for t in all_full_trades_detail:
        h = t["entry_hour"]
        m = t["entry_minute"]
        bucket = f"{h:02d}:{(m // 30) * 30:02d}"
        buckets[bucket].append(t)

    print(f"  {'Bucket':>8} {'Count':>5} {'WR':>6} {'Net P&L':>9} {'Avg P&L':>9} {'SLs':>4} {'PF':>6}")
    print(f"  {'-'*8} {'-'*5} {'-'*6} {'-'*9} {'-'*9} {'-'*4} {'-'*6}")
    for bucket in sorted(buckets.keys()):
        bt = buckets[bucket]
        wins = sum(1 for t in bt if t["is_win"])
        total = sum(t["net_pnl"] for t in bt)
        avg = total / len(bt)
        sls = sum(1 for t in bt if t["exit_reason"] == "stop_loss")
        gwin = sum(t["net_pnl"] for t in bt if t["is_win"])
        gloss = abs(sum(t["net_pnl"] for t in bt if not t["is_win"]))
        pf = gwin / gloss if gloss > 0 else float("inf")
        marker = " <<<" if bucket == "10:00" else ""
        print(f"  {bucket:>8} {len(bt):>5} {wins/len(bt)*100:>5.1f}% ${total:>+8.0f} ${avg:>+8.1f} {sls:>4} {pf:>6.2f}{marker}")

    # 3. Early trade exit reasons
    print("\n  --- Exit reason breakdown for early trades (10:00-10:29) ---")
    exit_counts = defaultdict(int)
    exit_pnl = defaultdict(float)
    for t in early_trades:
        exit_counts[t["exit_reason"]] += 1
        exit_pnl[t["exit_reason"]] += t["net_pnl"]
    for reason in sorted(exit_counts.keys()):
        print(f"    {reason}: {exit_counts[reason]} trades, ${exit_pnl[reason]:+.0f}")

    # 4. How many of the early SLs are within first 11 minutes?
    first_11_trades = [t for t in early_trades if t["entry_minute"] < 11]
    first_11_sls = [t for t in first_11_trades if t["exit_reason"] == "stop_loss"]
    print(f"\n  Trades entered 10:00-10:10 ET: {len(first_11_trades)}")
    print(f"  Of those, hit SL: {len(first_11_sls)}")
    if first_11_trades:
        pnl_11 = sum(t["net_pnl"] for t in first_11_trades)
        print(f"  Net P&L of 10:00-10:10 entries: ${pnl_11:+.0f}")

    # 5. Are early SLs also early EXITS? (i.e., do they fail fast or drift to SL?)
    if early_trades:
        early_sl_trades = [t for t in early_trades if t["exit_reason"] == "stop_loss"]
        if early_sl_trades:
            avg_bars = np.mean([t["bars_held"] for t in early_sl_trades])
            med_bars = np.median([t["bars_held"] for t in early_sl_trades])
            print(f"\n  Early SL trades: avg {avg_bars:.0f} bars held, median {med_bars:.0f} bars")
            for t in early_sl_trades[:20]:
                print(f"    {t['entry_date']} {t['entry_hhmm']} {t['side']} "
                      f"→ SL in {t['bars_held']} bars, ${t['net_pnl']:+.0f}")

    # 6. Bootstrap significance
    print("\n  --- Bootstrap significance test ---")
    if early_trades:
        early_mean = np.mean([t["net_pnl"] for t in early_trades])
        all_pnl = [t["net_pnl"] for t in all_full_trades_detail]
        n_early = len(early_trades)

        np.random.seed(42)
        n_boot = 10000
        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(all_pnl, size=n_early, replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.array(boot_means)
        pct_worse = np.mean(boot_means <= early_mean) * 100
        print(f"  Early avg P&L: ${early_mean:+.1f}")
        print(f"  Random {n_early}-trade sample avg P&L: ${np.mean(boot_means):+.1f} ± ${np.std(boot_means):.1f}")
        print(f"  {pct_worse:.1f}% of random samples are worse than early trades")
        if pct_worse < 5:
            print(f"  >>> STATISTICALLY SIGNIFICANT (p < 0.05)")
        elif pct_worse < 10:
            print(f"  >>> MARGINALLY SIGNIFICANT (p < 0.10)")
        else:
            print(f"  >>> NOT SIGNIFICANT (p = {pct_worse/100:.2f})")

    # 7. IS vs OOS split
    print("\n  --- IS vs OOS breakdown of early trades ---")
    trade_dates_sorted = sorted(set(t["entry_date"] for t in all_full_trades_detail))
    mid_date = trade_dates_sorted[len(trade_dates_sorted) // 2]
    is_early = [t for t in early_trades if t["entry_date"] < mid_date]
    oos_early = [t for t in early_trades if t["entry_date"] >= mid_date]
    print_trade_table(is_early, f"IS early trades (before {mid_date})", max_show=25)
    print_trade_table(oos_early, f"OOS early trades (on/after {mid_date})", max_show=25)

    return early_trades, later_trades


def red_team_stacked(df, splits, is_len, all_full_trades_detail, prior_vix_fn):
    """Test both gates stacked together."""
    print("\n" + "=" * 120)
    print("RED TEAM #3: STACKED GATES (VIX [20-25] + Entry Delay +30min)")
    print("=" * 120)

    # Get zone days
    zone_days = set()
    trade_dates = sorted(set(t["entry_date"] for t in all_full_trades_detail))
    for d in trade_dates:
        vix_val = prior_vix_fn(d)
        if vix_val is not None and 20 <= vix_val <= 25:
            zone_days.add(d)

    # Classify: blocked by VIX, blocked by delay, blocked by both, kept
    blocked_vix_only = []
    blocked_delay_only = []
    blocked_both = []
    kept = []

    for t in all_full_trades_detail:
        in_vix_zone = t["entry_date"] in zone_days
        is_early = t["entry_hour"] == 10 and t["entry_minute"] < 30

        if in_vix_zone and is_early:
            blocked_both.append(t)
        elif in_vix_zone:
            blocked_vix_only.append(t)
        elif is_early:
            blocked_delay_only.append(t)
        else:
            kept.append(t)

    total_blocked = len(blocked_vix_only) + len(blocked_delay_only) + len(blocked_both)
    total = len(all_full_trades_detail)

    print(f"\n  Total trades: {total}")
    print(f"  Blocked by VIX only: {len(blocked_vix_only)}")
    print(f"  Blocked by delay only: {len(blocked_delay_only)}")
    print(f"  Blocked by BOTH: {len(blocked_both)}")
    print(f"  Total blocked: {total_blocked} ({total_blocked/total*100:.1f}%)")
    print(f"  Remaining: {len(kept)} ({len(kept)/total*100:.1f}%)")

    all_blocked = blocked_vix_only + blocked_delay_only + blocked_both
    print_trade_table(all_blocked, "ALL BLOCKED (stacked)")
    print_trade_table(kept, "KEPT (stacked)")

    # Compare PF
    if kept:
        kept_wins_pnl = sum(t["net_pnl"] for t in kept if t["is_win"])
        kept_loss_pnl = abs(sum(t["net_pnl"] for t in kept if not t["is_win"]))
        kept_pf = kept_wins_pnl / kept_loss_pnl if kept_loss_pnl > 0 else float("inf")
        kept_wr = sum(1 for t in kept if t["is_win"]) / len(kept) * 100
        kept_sharpe = compute_sharpe([t["net_pnl"] for t in kept])

        base_wins_pnl = sum(t["net_pnl"] for t in all_full_trades_detail if t["is_win"])
        base_loss_pnl = abs(sum(t["net_pnl"] for t in all_full_trades_detail if not t["is_win"]))
        base_pf = base_wins_pnl / base_loss_pnl if base_loss_pnl > 0 else float("inf")
        base_wr = sum(1 for t in all_full_trades_detail if t["is_win"]) / len(all_full_trades_detail) * 100
        base_sharpe = compute_sharpe([t["net_pnl"] for t in all_full_trades_detail])

        print(f"\n  BASELINE: {total}t, WR {base_wr:.1f}%, PF {base_pf:.3f}, Sharpe {base_sharpe:.2f}")
        print(f"  STACKED:  {len(kept)}t, WR {kept_wr:.1f}%, PF {kept_pf:.3f}, Sharpe {kept_sharpe:.2f}")
        print(f"  PF change: {(kept_pf - base_pf) / base_pf * 100:+.1f}%")
        print(f"  Trade count reduction: {total_blocked}/{total} = {total_blocked/total*100:.1f}%")

    # IS/OOS of stacked
    print("\n  --- IS vs OOS with stacked gates ---")
    trade_dates_sorted = sorted(set(t["entry_date"] for t in all_full_trades_detail))
    mid_date = trade_dates_sorted[len(trade_dates_sorted) // 2]

    is_kept = [t for t in kept if t["entry_date"] < mid_date]
    oos_kept = [t for t in kept if t["entry_date"] >= mid_date]

    is_base = [t for t in all_full_trades_detail if t["entry_date"] < mid_date]
    oos_base = [t for t in all_full_trades_detail if t["entry_date"] >= mid_date]

    for label, trades_g, trades_b in [("IS", is_kept, is_base), ("OOS", oos_kept, oos_base)]:
        if not trades_g or not trades_b:
            continue
        gw = sum(t["net_pnl"] for t in trades_g if t["is_win"])
        gl = abs(sum(t["net_pnl"] for t in trades_g if not t["is_win"]))
        gpf = gw / gl if gl > 0 else float("inf")
        bw = sum(t["net_pnl"] for t in trades_b if t["is_win"])
        bl = abs(sum(t["net_pnl"] for t in trades_b if not t["is_win"]))
        bpf = bw / bl if bl > 0 else float("inf")
        dpf = (gpf - bpf) / bpf * 100
        print(f"  {label}: Base PF {bpf:.3f} ({len(trades_b)}t) → Gated PF {gpf:.3f} ({len(trades_g)}t), dPF {dpf:+.1f}%")


def compute_sharpe(pnl_list):
    """Compute Sharpe ratio from a list of P&L values."""
    if len(pnl_list) < 2:
        return 0
    arr = np.array(pnl_list)
    if arr.std() == 0:
        return 0
    return arr.mean() / arr.std() * np.sqrt(252)


def red_team_alternative_vix_bands(df, splits, is_len, all_full_trades_detail, prior_vix_fn):
    """Test whether VIX [20-25] is special or if any 5-pt band looks similar."""
    print("\n" + "=" * 120)
    print("RED TEAM #4: IS VIX [20-25] SPECIAL? (Compare all 5-pt bands)")
    print("=" * 120)

    all_pnl = sum(t["net_pnl"] for t in all_full_trades_detail)
    total = len(all_full_trades_detail)

    print(f"  {'Band':>12} {'Zone Trades':>11} {'Zone P&L':>9} {'Zone Avg':>9} {'Kept PF':>8} {'dPF':>7} {'Zone WR':>7}")
    print(f"  {'-'*12} {'-'*11} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*7}")

    # Test every 5-pt band from 10 to 40
    for vmin in range(10, 36):
        vmax = vmin + 5
        zone_days = set()
        trade_dates = sorted(set(t["entry_date"] for t in all_full_trades_detail))
        for d in trade_dates:
            vix_val = prior_vix_fn(d)
            if vix_val is not None and vmin <= vix_val <= vmax:
                zone_days.add(d)

        zone_trades = [t for t in all_full_trades_detail if t["entry_date"] in zone_days]
        kept_trades = [t for t in all_full_trades_detail if t["entry_date"] not in zone_days]

        if not zone_trades or not kept_trades:
            continue

        zone_pnl = sum(t["net_pnl"] for t in zone_trades)
        zone_avg = zone_pnl / len(zone_trades)
        zone_wr = sum(1 for t in zone_trades if t["is_win"]) / len(zone_trades) * 100

        kw = sum(t["net_pnl"] for t in kept_trades if t["is_win"])
        kl = abs(sum(t["net_pnl"] for t in kept_trades if not t["is_win"]))
        kept_pf = kw / kl if kl > 0 else float("inf")

        bw = sum(t["net_pnl"] for t in all_full_trades_detail if t["is_win"])
        bl = abs(sum(t["net_pnl"] for t in all_full_trades_detail if not t["is_win"]))
        base_pf = bw / bl if bl > 0 else float("inf")
        dpf = (kept_pf - base_pf) / base_pf * 100

        marker = " <<<" if vmin == 20 else ""
        print(f"  [{vmin:>2}-{vmax:>2}] {len(zone_trades):>11} ${zone_pnl:>+8.0f} ${zone_avg:>+8.1f} "
              f"{kept_pf:>8.3f} {dpf:>+6.1f}% {zone_wr:>6.1f}%{marker}")


def red_team_alternative_delays(df, splits, is_len, all_full_trades_detail):
    """Test whether +30min delay is special or if any window looks similar."""
    print("\n" + "=" * 120)
    print("RED TEAM #5: IS +30min SPECIAL? (Compare all delay windows)")
    print("=" * 120)

    print(f"  {'Delay':>8} {'Blocked':>7} {'Blk P&L':>9} {'Blk Avg':>9} {'Kept PF':>8} {'dPF':>7} {'Blk WR':>7}")
    print(f"  {'-'*8} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*7}")

    for delay_min in [5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90, 120]:
        early = [t for t in all_full_trades_detail
                 if (t["entry_hour"] * 60 + t["entry_minute"]) < (600 + delay_min)
                 and (t["entry_hour"] * 60 + t["entry_minute"]) >= 600]
        later = [t for t in all_full_trades_detail
                 if not ((t["entry_hour"] * 60 + t["entry_minute"]) < (600 + delay_min)
                         and (t["entry_hour"] * 60 + t["entry_minute"]) >= 600)]

        if not early or not later:
            continue

        early_pnl = sum(t["net_pnl"] for t in early)
        early_avg = early_pnl / len(early)
        early_wr = sum(1 for t in early if t["is_win"]) / len(early) * 100

        kw = sum(t["net_pnl"] for t in later if t["is_win"])
        kl = abs(sum(t["net_pnl"] for t in later if not t["is_win"]))
        kept_pf = kw / kl if kl > 0 else float("inf")

        bw = sum(t["net_pnl"] for t in all_full_trades_detail if t["is_win"])
        bl = abs(sum(t["net_pnl"] for t in all_full_trades_detail if not t["is_win"]))
        base_pf = bw / bl if bl > 0 else float("inf")
        dpf = (kept_pf - base_pf) / base_pf * 100

        marker = " <<<" if delay_min == 30 else ""
        print(f"  +{delay_min:>3}min {len(early):>7} ${early_pnl:>+8.0f} ${early_avg:>+8.1f} "
              f"{kept_pf:>8.3f} {dpf:>+6.1f}% {early_wr:>6.1f}%{marker}")


def main():
    print("=" * 120)
    print("MES V2 RED TEAM ANALYSIS")
    print("Forensic validation of VIX [20-25] and Entry Delay +30min")
    print("=" * 120)

    # Load data and run baseline
    df, splits, is_len = prepare_mes_data()

    # Run FULL baseline to get all trades
    print("\nRunning FULL baseline...")
    bl_full, bl_trades = run_mes(splits, "FULL")
    times = splits["FULL"][5]  # index/times

    # Get detailed trade info
    all_trades = get_trade_details(bl_trades, times)
    print(f"\nBaseline: {len(all_trades)} trades, Net ${sum(t['net_pnl'] for t in all_trades):+.0f}")

    # Get VIX lookup
    prior_vix_fn, vix_closes = get_vix_lookup(df)

    # === RED TEAM TESTS ===

    # 1. VIX [20-25] forensic analysis
    zone_trades, non_zone_trades = red_team_vix_gate(
        df, splits, is_len, all_trades, prior_vix_fn
    )

    # 2. Entry delay +30min forensic analysis
    early_trades, later_trades = red_team_entry_delay(
        df, splits, is_len, all_trades
    )

    # 3. Stacked gates
    red_team_stacked(df, splits, is_len, all_trades, prior_vix_fn)

    # 4. Is VIX [20-25] special vs other bands?
    red_team_alternative_vix_bands(df, splits, is_len, all_trades, prior_vix_fn)

    # 5. Is +30min special vs other delays?
    red_team_alternative_delays(df, splits, is_len, all_trades)

    # === FINAL VERDICT ===
    print("\n" + "=" * 120)
    print("RED TEAM VERDICT")
    print("=" * 120)
    print("Review the above analysis to determine:")
    print("  1. Are blocked trades genuinely worse, or just random?")
    print("  2. Is the improvement driven by a few outlier days?")
    print("  3. Do both IS and OOS halves show the same pattern?")
    print("  4. Is [20-25] special or does any VIX band look similar?")
    print("  5. Is +30min special or does any delay look similar?")
    print("  6. Do stacked gates maintain enough trades?")


if __name__ == "__main__":
    main()
