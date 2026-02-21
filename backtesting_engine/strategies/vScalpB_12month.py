"""
Task 1: vScalpB 12-Month Validation
=====================================
Validates vScalpB (SM_T=0.25, RSI=8/55-45, SL=15, TP=5) on full 12-month data.
Compares all 3 strategies side-by-side. Answers: does vScalpB + vScalpA outperform
vScalpA alone?

Naming convention:
  vScalpA = v15 (TP=5, no threshold, wide net scalp)
  vScalpB = v16 (TP=5, threshold=0.25, selective scalp)
  vWinners = v11 (SM flip exit, lets winners run)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    fmt_score, compute_et_minutes,
    NY_OPEN_ET, NY_CLOSE_ET,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit, compute_mfe_mae

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MNQ_COMM, MNQ_DPP = 0.52, 2.0
SPLIT = pd.Timestamp("2025-08-17", tz='UTC')


# ---------------------------------------------------------------------------
# Data loading (from oos_12month_check.py pattern)
# ---------------------------------------------------------------------------

def load_combined():
    files = sorted(DATA_DIR.glob("databento_MNQ_1min_2025-*.csv"))
    print(f"  Loading MNQ: {[f.name for f in files]}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        result = pd.DataFrame()
        result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
        result['Open'] = pd.to_numeric(df['open'], errors='coerce')
        result['High'] = pd.to_numeric(df['high'], errors='coerce')
        result['Low'] = pd.to_numeric(df['low'], errors='coerce')
        result['Close'] = pd.to_numeric(df['close'], errors='coerce')
        result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        result = result.set_index('Time')
        dfs.append(result)
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    sm = compute_smart_money(
        combined['Close'].values, combined['Volume'].values, **MNQ_SM
    )
    combined['SM_Net'] = sm
    print(f"    {len(combined):,} bars total ({combined.index[0].date()} to {combined.index[-1].date()})")
    return combined


def prepare_arrays(df_1m, rsi_len):
    df_5m = resample_to_5min(df_1m)
    rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
        df_1m.index.values, df_5m.index.values, df_5m['Close'].values, rsi_len
    )
    rsi_dummy = np.full(len(df_1m), 50.0)
    return {
        'opens': df_1m['Open'].values,
        'highs': df_1m['High'].values,
        'lows': df_1m['Low'].values,
        'closes': df_1m['Close'].values,
        'sm': df_1m['SM_Net'].values,
        'times': df_1m.index.values,
        'rsi': rsi_dummy,
        'rsi_5m_curr': rsi_5m_curr,
        'rsi_5m_prev': rsi_5m_prev,
    }


def split_trades(trades, split_ts):
    oos, is_trades = [], []
    for t in trades:
        exit_ts = pd.Timestamp(t['exit_time'])
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize('UTC')
        if exit_ts < split_ts:
            oos.append(t)
        else:
            is_trades.append(t)
    return oos, is_trades


# ---------------------------------------------------------------------------
# Monthly breakdown helper
# ---------------------------------------------------------------------------

def monthly_breakdown(trades, label, comm=MNQ_COMM, dpp=MNQ_DPP):
    monthly = defaultdict(list)
    for t in trades:
        m = pd.Timestamp(t['entry_time']).strftime('%Y-%m')
        monthly[m].append(t)

    print(f"\n  Monthly breakdown: {label}")
    print(f"  {'Month':<10} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Net $':>10}")
    print("  " + "-" * 50)
    for m in sorted(monthly.keys()):
        sc = score_trades(monthly[m], commission_per_side=comm, dollar_per_pt=dpp)
        if sc:
            print(f"  {m:<10} {sc['count']:>7d} {sc['win_rate']:>5.1f}% "
                  f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f}")
    return monthly


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def trades_to_daily_pnl(trades, comm=MNQ_COMM, dpp=MNQ_DPP):
    """Convert trade list to daily P&L Series."""
    comm_pts = (comm * 2) / dpp
    daily = defaultdict(float)
    for t in trades:
        exit_date = pd.Timestamp(t['exit_time']).date()
        net_dollar = (t['pts'] - comm_pts) * dpp
        daily[exit_date] += net_dollar
    return pd.Series(daily).sort_index()


def portfolio_stats(daily_pnl, label):
    """Compute portfolio stats from daily P&L Series."""
    if daily_pnl.empty:
        print(f"  {label:<35} NO DATA")
        return
    cum = daily_pnl.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    net = daily_pnl.sum()
    sharpe = 0.0
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
    print(f"  {label:<35} Net ${net:>+10.2f}  MaxDD ${dd:>10.2f}  Sharpe {sharpe:>6.3f}  Days {len(daily_pnl)}")
    return {'net': net, 'max_dd': dd, 'sharpe': sharpe}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("TASK 1: vScalpB 12-MONTH VALIDATION")
    print("=" * 80)

    # Load data
    print("\nLoading combined 12-month MNQ data...")
    mnq = load_combined()

    print("Preparing arrays (RSI=8)...")
    arr = prepare_arrays(mnq, rsi_len=8)

    # ================================================================
    # Run all 3 strategies on full 12-month data
    # ================================================================
    print("\n" + "=" * 80)
    print("RUNNING STRATEGIES")
    print("=" * 80)

    # vScalpA: SM_T=0.0, RSI=8/60-40, CD=20, SL=50, TP=5
    print("  Running vScalpA (TP=5, no threshold)...")
    vscalpa_trades = run_backtest_tp_exit(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], arr['times'],
        arr['rsi_5m_curr'], arr['rsi_5m_prev'],
        rsi_buy=60, rsi_sell=40, sm_threshold=0.0,
        cooldown_bars=20, max_loss_pts=50, tp_pts=5,
    )
    compute_mfe_mae(vscalpa_trades, arr['highs'], arr['lows'])

    # vScalpB: SM_T=0.25, RSI=8/55-45, CD=20, SL=15, TP=5
    print("  Running vScalpB (TP=5, threshold=0.25)...")
    vscalpb_trades = run_backtest_tp_exit(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], arr['times'],
        arr['rsi_5m_curr'], arr['rsi_5m_prev'],
        rsi_buy=55, rsi_sell=45, sm_threshold=0.25,
        cooldown_bars=20, max_loss_pts=15, tp_pts=5,
    )
    compute_mfe_mae(vscalpb_trades, arr['highs'], arr['lows'])

    # vWinners: SM_T=0.15, RSI=8/60-40, CD=20, SL=50, SM flip exit
    print("  Running vWinners (SM flip, threshold=0.15)...")
    rsi_dummy = np.full(len(arr['closes']), 50.0)
    vwinners_trades = run_backtest_v10(
        arr['opens'], arr['highs'], arr['lows'], arr['closes'],
        arr['sm'], rsi_dummy, arr['times'],
        rsi_buy=60, rsi_sell=40, sm_threshold=0.15,
        cooldown_bars=20, max_loss_pts=50,
        rsi_5m_curr=arr['rsi_5m_curr'], rsi_5m_prev=arr['rsi_5m_prev'],
    )
    compute_mfe_mae(vwinners_trades, arr['highs'], arr['lows'])

    # ================================================================
    # 1. 12-month combined results
    # ================================================================
    print("\n" + "=" * 80)
    print("1. 12-MONTH COMBINED RESULTS")
    print("=" * 80)

    strategies = [
        ("vScalpA  (TP=5, T=0.0)", vscalpa_trades),
        ("vScalpB  (TP=5, T=0.25)", vscalpb_trades),
        ("vWinners (SM flip, T=0.15)", vwinners_trades),
    ]

    print(f"\n  {'Strategy':<30} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Net $':>11} "
          f"{'MaxDD $':>10} {'Sharpe':>7}")
    print("  " + "-" * 85)
    for label, trades in strategies:
        sc = score_trades(trades, MNQ_COMM, MNQ_DPP)
        if sc:
            print(f"  {label:<30} {sc['count']:>7d} {sc['win_rate']:>5.1f}% "
                  f"{sc['pf']:>7.3f} {sc['net_dollar']:>+11.2f} "
                  f"{sc['max_dd_dollar']:>10.2f} {sc['sharpe']:>7.3f}")

    # ================================================================
    # 2. IS vs OOS breakdown
    # ================================================================
    print("\n" + "=" * 80)
    print("2. IS vs OOS BREAKDOWN")
    print("=" * 80)

    print(f"\n  {'Strategy':<30} {'Split':>5} {'Trades':>7} {'WR%':>6} {'PF':>7} "
          f"{'Net $':>11} {'MaxDD $':>10}")
    print("  " + "-" * 85)
    for label, trades in strategies:
        oos, is_t = split_trades(trades, SPLIT)
        for period, t_list in [("OOS", oos), ("IS", is_t)]:
            sc = score_trades(t_list, MNQ_COMM, MNQ_DPP)
            if sc:
                print(f"  {label:<30} {period:>5} {sc['count']:>7d} {sc['win_rate']:>5.1f}% "
                      f"{sc['pf']:>7.3f} {sc['net_dollar']:>+11.2f} "
                      f"{sc['max_dd_dollar']:>10.2f}")

    # IS sanity checks
    _, is_vwinners = split_trades(vwinners_trades, SPLIT)
    _, is_vscalpa = split_trades(vscalpa_trades, SPLIT)
    sc_is_w = score_trades(is_vwinners, MNQ_COMM, MNQ_DPP)
    sc_is_a = score_trades(is_vscalpa, MNQ_COMM, MNQ_DPP)
    print(f"\n  IS sanity checks:")
    print(f"    vWinners IS: {sc_is_w['count']} trades, PF {sc_is_w['pf']}, "
          f"WR {sc_is_w['win_rate']}%, Net ${sc_is_w['net_dollar']:+.2f}")
    print(f"    Expected:    226 trades, PF 1.797, WR 62.4%, +$3,529")
    print(f"    vScalpA  IS: {sc_is_a['count']} trades, PF {sc_is_a['pf']}, "
          f"WR {sc_is_a['win_rate']}%, Net ${sc_is_a['net_dollar']:+.2f}")
    print(f"    Expected:    363 trades, WR 88.2%, PF 1.272, +$1,324")

    # ================================================================
    # 3. Monthly breakdown for all 3
    # ================================================================
    print("\n" + "=" * 80)
    print("3. MONTHLY BREAKDOWN")
    print("=" * 80)

    for label, trades in strategies:
        monthly_breakdown(trades, label)

    # ================================================================
    # 4. Trade overlap analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("4. TRADE OVERLAP ANALYSIS (vScalpA vs vScalpB)")
    print("=" * 80)

    # Match vScalpB entries to vScalpA entries (same bar ±1)
    vscalpa_entry_indices = {t['entry_idx'] for t in vscalpa_trades}
    vscalpa_entry_map = {t['entry_idx']: t for t in vscalpa_trades}

    overlapping_b = []
    unique_b = []
    for t in vscalpb_trades:
        matched = False
        for offset in [0, -1, 1]:
            if t['entry_idx'] + offset in vscalpa_entry_indices:
                overlapping_b.append((t, vscalpa_entry_map[t['entry_idx'] + offset]))
                matched = True
                break
        if not matched:
            unique_b.append(t)

    # vScalpA-only trades (entries not matched to vScalpB)
    vscalpb_entry_indices = {t['entry_idx'] for t in vscalpb_trades}
    vscalpa_only = []
    for t in vscalpa_trades:
        matched = False
        for offset in [0, -1, 1]:
            if t['entry_idx'] + offset in vscalpb_entry_indices:
                matched = True
                break
        if not matched:
            vscalpa_only.append(t)

    comm_pts = (MNQ_COMM * 2) / MNQ_DPP

    print(f"\n  Total vScalpB trades: {len(vscalpb_trades)}")
    print(f"  Total vScalpA trades: {len(vscalpa_trades)}")
    print(f"  Overlapping (same bar ±1): {len(overlapping_b)} "
          f"({len(overlapping_b)/len(vscalpb_trades)*100:.1f}% of vScalpB)")
    print(f"  vScalpB unique entries: {len(unique_b)}")
    print(f"  vScalpA-only entries (SM < 0.25): {len(vscalpa_only)}")

    # Performance on overlapping trades
    if overlapping_b:
        b_pts = np.array([t[0]['pts'] - comm_pts for t in overlapping_b])
        a_pts = np.array([t[1]['pts'] - comm_pts for t in overlapping_b])
        print(f"\n  On overlapping trades:")
        print(f"    vScalpB avg net pts: {np.mean(b_pts):+.2f}  (SL=15)")
        print(f"    vScalpA avg net pts: {np.mean(a_pts):+.2f}  (SL=50)")
        print(f"    vScalpB WR: {(b_pts > 0).mean()*100:.1f}%")
        print(f"    vScalpA WR: {(a_pts > 0).mean()*100:.1f}%")
        print(f"    vScalpB total $: {np.sum(b_pts)*MNQ_DPP:+.2f}")
        print(f"    vScalpA total $: {np.sum(a_pts)*MNQ_DPP:+.2f}")

    # Performance on vScalpA-only trades (weak SM signal)
    if vscalpa_only:
        sc_aonly = score_trades(vscalpa_only, MNQ_COMM, MNQ_DPP)
        if sc_aonly:
            print(f"\n  vScalpA-only trades (SM < 0.25, filtered out by vScalpB):")
            print(f"    {sc_aonly['count']} trades, WR {sc_aonly['win_rate']}%, "
                  f"PF {sc_aonly['pf']}, Net ${sc_aonly['net_dollar']:+.2f}")

            # Split by IS/OOS
            oos_only, is_only = split_trades(vscalpa_only, SPLIT)
            sc_oos_only = score_trades(oos_only, MNQ_COMM, MNQ_DPP)
            sc_is_only = score_trades(is_only, MNQ_COMM, MNQ_DPP)
            if sc_oos_only:
                print(f"    OOS: {sc_oos_only['count']} trades, WR {sc_oos_only['win_rate']}%, "
                      f"PF {sc_oos_only['pf']}, Net ${sc_oos_only['net_dollar']:+.2f}")
            if sc_is_only:
                print(f"    IS:  {sc_is_only['count']} trades, WR {sc_is_only['win_rate']}%, "
                      f"PF {sc_is_only['pf']}, Net ${sc_is_only['net_dollar']:+.2f}")

    # ================================================================
    # 5. Portfolio simulation
    # ================================================================
    print("\n" + "=" * 80)
    print("5. PORTFOLIO SIMULATION")
    print("=" * 80)

    daily_a = trades_to_daily_pnl(vscalpa_trades)
    daily_b = trades_to_daily_pnl(vscalpb_trades)
    daily_w = trades_to_daily_pnl(vwinners_trades)

    # Combine daily P&L for portfolio variants
    all_dates = sorted(set(daily_a.index) | set(daily_b.index) | set(daily_w.index))
    combined_index = pd.Index(all_dates)

    daily_a_full = daily_a.reindex(combined_index, fill_value=0)
    daily_b_full = daily_b.reindex(combined_index, fill_value=0)
    daily_w_full = daily_w.reindex(combined_index, fill_value=0)

    print(f"\n  12-MONTH PORTFOLIO COMPARISON:")
    print(f"  {'Portfolio':<35} {'Net $':>11} {'MaxDD $':>10} {'Sharpe':>7} {'Days':>5}")
    print("  " + "-" * 75)
    portfolio_stats(daily_a_full[daily_a_full != 0], "vScalpA alone")
    portfolio_stats(daily_b_full[daily_b_full != 0], "vScalpB alone")
    portfolio_stats(daily_w_full[daily_w_full != 0], "vWinners alone")
    portfolio_stats(daily_a_full + daily_b_full, "vScalpA + vScalpB")
    portfolio_stats(daily_a_full + daily_b_full + daily_w_full, "vScalpA + vScalpB + vWinners")

    # Split by IS/OOS
    oos_mask = combined_index < pd.Timestamp("2025-08-17").date()
    is_mask = ~oos_mask

    print(f"\n  OOS PERIOD PORTFOLIO:")
    print(f"  {'Portfolio':<35} {'Net $':>11} {'MaxDD $':>10} {'Sharpe':>7} {'Days':>5}")
    print("  " + "-" * 75)
    portfolio_stats(daily_a_full[oos_mask][daily_a_full[oos_mask] != 0], "vScalpA alone")
    portfolio_stats(daily_b_full[oos_mask][daily_b_full[oos_mask] != 0], "vScalpB alone")
    portfolio_stats(daily_w_full[oos_mask][daily_w_full[oos_mask] != 0], "vWinners alone")
    portfolio_stats((daily_a_full + daily_b_full)[oos_mask], "vScalpA + vScalpB")
    portfolio_stats((daily_a_full + daily_b_full + daily_w_full)[oos_mask], "vScalpA + vScalpB + vWinners")

    print(f"\n  IS PERIOD PORTFOLIO:")
    print(f"  {'Portfolio':<35} {'Net $':>11} {'MaxDD $':>10} {'Sharpe':>7} {'Days':>5}")
    print("  " + "-" * 75)
    portfolio_stats(daily_a_full[is_mask][daily_a_full[is_mask] != 0], "vScalpA alone")
    portfolio_stats(daily_b_full[is_mask][daily_b_full[is_mask] != 0], "vScalpB alone")
    portfolio_stats(daily_w_full[is_mask][daily_w_full[is_mask] != 0], "vWinners alone")
    portfolio_stats((daily_a_full + daily_b_full)[is_mask], "vScalpA + vScalpB")
    portfolio_stats((daily_a_full + daily_b_full + daily_w_full)[is_mask], "vScalpA + vScalpB + vWinners")

    # ================================================================
    # Key question
    # ================================================================
    print("\n" + "=" * 80)
    print("KEY QUESTION: Does vScalpB improve the portfolio?")
    print("=" * 80)

    ab_daily = daily_a_full + daily_b_full
    a_only_daily = daily_a_full

    net_a = a_only_daily.sum()
    net_ab = ab_daily.sum()
    improvement = net_ab - net_a

    cum_a = a_only_daily.cumsum()
    cum_ab = ab_daily.cumsum()
    dd_a = (cum_a - cum_a.cummax()).min()
    dd_ab = (cum_ab - cum_ab.cummax()).min()

    sharpe_a = a_only_daily.mean() / a_only_daily.std() * np.sqrt(252) if a_only_daily.std() > 0 else 0
    sharpe_ab = ab_daily.mean() / ab_daily.std() * np.sqrt(252) if ab_daily.std() > 0 else 0

    print(f"\n  vScalpA alone:       Net ${net_a:+.2f}, MaxDD ${dd_a:.2f}, Sharpe {sharpe_a:.3f}")
    print(f"  vScalpA + vScalpB:   Net ${net_ab:+.2f}, MaxDD ${dd_ab:.2f}, Sharpe {sharpe_ab:.3f}")
    print(f"  Improvement:         ${improvement:+.2f}")
    print(f"  Sharpe delta:        {sharpe_ab - sharpe_a:+.3f}")
    print(f"  DrawDown delta:      ${dd_ab - dd_a:+.2f}")

    if improvement > 0 and sharpe_ab > sharpe_a:
        print(f"\n  VERDICT: YES — vScalpB improves the portfolio on both P&L and risk-adjusted basis")
    elif improvement > 0:
        print(f"\n  VERDICT: MIXED — vScalpB adds P&L but risk-adjusted return is worse")
    else:
        print(f"\n  VERDICT: NO — vScalpB does NOT improve the portfolio")

    print("\n" + "=" * 80)
    print("DONE — Task 1 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
