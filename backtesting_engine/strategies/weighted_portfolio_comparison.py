"""
Weighted Portfolio Comparison: Scaling vScalpB with/without vScalpA insurance.

Tests 6 portfolio configurations with different contract quantities:
  1. Baseline:              vScalpA(1) + vScalpB(1) + MES(1)
  2. Drop vScalpA:          vScalpB(1) + MES(1)
  3. Scale vScalpB only:    vScalpB(2) + MES(1)
  4. Keep both, vScalpB 2x: vScalpA(1) + vScalpB(2) + MES(1)
  5. Keep both, vScalpB 3x: vScalpA(1) + vScalpB(3) + MES(1)
  6. Scale both 2x:         vScalpA(2) + vScalpB(2) + MES(1)

Key question: Does vScalpA provide uncorrelated volatile-market insurance
that improves risk-adjusted returns when vScalpB is scaled up?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Setup imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))

from v10_test_common import (
    compute_rsi, compute_smart_money, map_5min_rsi_to_1min,
    resample_to_5min, score_trades, fmt_score,
)
from generate_session import (
    load_instrument_1min, run_backtest_tp_exit, compute_mfe_mae,
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN, VSCALPA_TP_PTS,
    VSCALPA_ENTRY_END_ET,
)
from results.save_results import load_backtest

# ── Constants ──
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
VSCALPA_SL40 = 40

# Portfolio configurations: (label, vScalpA_qty, vScalpB_qty, MES_qty)
CONFIGS = [
    ("Baseline (1/1/1)",               1, 1, 1),
    ("Drop vScalpA (0/1/1)",           0, 1, 1),
    ("Scale vScalpB only (0/2/1)",     0, 2, 1),
    ("Keep both + vScalpB 2x (1/2/1)", 1, 2, 1),
    ("Keep both + vScalpB 3x (1/3/1)", 1, 3, 1),
    ("Scale both 2x (2/2/1)",         2, 2, 1),
]

# Configs for monthly comparison
MONTHLY_CONFIGS = [0, 2, 3]  # indices into CONFIGS


def run_vscalpa_sl40():
    """Re-run vScalpA with SL=40 and return trade-level DataFrame."""
    print("Running vScalpA with SL=40...")
    df_mnq = load_instrument_1min("MNQ")
    mnq_sm = compute_smart_money(
        df_mnq["Close"].values, df_mnq["Volume"].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    df_mnq["SM_Net"] = mnq_sm

    df_mnq_5m = resample_to_5min(df_mnq)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df_mnq.index.values, df_mnq_5m.index.values,
        df_mnq_5m["Close"].values, rsi_len=VSCALPA_RSI_LEN,
    )

    trades = run_backtest_tp_exit(
        df_mnq["Open"].values, df_mnq["High"].values,
        df_mnq["Low"].values, df_mnq["Close"].values,
        mnq_sm, df_mnq.index,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPA_SL40, tp_pts=VSCALPA_TP_PTS,
        entry_end_et=VSCALPA_ENTRY_END_ET,
    )

    sc = score_trades(trades, commission_per_side=MNQ_COMMISSION,
                      dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"  {fmt_score(sc, 'vScalpA SL=40')}")

    rows = []
    cum_pnl = 0.0
    for i, t in enumerate(trades):
        pnl_pts = t["pts"]
        pnl_dollar = pnl_pts * MNQ_DOLLAR_PER_PT - 2 * MNQ_COMMISSION
        cum_pnl += pnl_dollar

        entry_ts = pd.Timestamp(t["entry_time"])
        if entry_ts.tz is None:
            entry_ts = entry_ts.tz_localize("UTC")
        entry_et = entry_ts.tz_convert("America/New_York")
        trade_date = entry_et.strftime("%Y-%m-%d")

        rows.append({
            "trade_num": i + 1,
            "trade_date": trade_date,
            "side": t["side"],
            "entry_time": entry_ts.isoformat(),
            "exit_time": pd.Timestamp(t["exit_time"]).isoformat(),
            "entry_price": round(t["entry"], 2),
            "exit_price": round(t["exit"], 2),
            "pts": round(pnl_pts, 2),
            "pnl_dollar": round(pnl_dollar, 2),
            "cumulative_pnl": round(cum_pnl, 2),
            "exit_reason": t.get("result", ""),
            "bars_held": t.get("bars", 0),
        })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} trades, total P&L: ${cum_pnl:.2f}")
    return df


def load_saved_strategy(strategy, split="FULL"):
    """Load most recent saved results for a strategy."""
    files = sorted(RESULTS_DIR.glob(f"{strategy}_{split}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No saved results for {strategy}/{split}")
    filepath = files[-1]
    meta, df = load_backtest(filepath)
    print(f"  Loaded {strategy}: {len(df)} trades, total P&L: ${meta['total_pnl']:.2f} from {filepath.name}")
    return meta, df


def daily_pnl(df, label):
    """Aggregate trade-level data to daily P&L series."""
    daily = df.groupby("trade_date")["pnl_dollar"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily.name = label
    return daily


def compute_metrics(daily_series, label):
    """Compute full suite of portfolio metrics from daily P&L series."""
    total = daily_series.sum()
    n_days = len(daily_series)
    mean_daily = daily_series.mean()
    std_daily = daily_series.std()

    # Annualized Sharpe
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

    # Max drawdown
    cum = daily_series.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()

    # Profit factor (daily)
    gross_profit = daily_series[daily_series > 0].sum()
    gross_loss = abs(daily_series[daily_series < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sortino ratio
    downside = daily_series[daily_series < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0
    sortino = (mean_daily / downside_dev * np.sqrt(252)) if downside_dev > 0 else float("inf")

    # Monthly stats
    monthly = daily_series.resample("ME").sum()
    losing_months = (monthly < 0).sum()
    worst_month = monthly.min()
    worst_month_label = monthly.idxmin().strftime("%Y-%m") if len(monthly) > 0 else "N/A"

    # Worst single day
    worst_day = daily_series.min()
    worst_day_label = daily_series.idxmin().strftime("%Y-%m-%d") if len(daily_series) > 0 else "N/A"

    # Win/loss days
    win_days = (daily_series > 0).sum()
    loss_days = (daily_series < 0).sum()

    return {
        "label": label,
        "total_pnl": total,
        "n_trading_days": n_days,
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "pf": pf,
        "sortino": sortino,
        "losing_months": losing_months,
        "worst_day": worst_day,
        "worst_day_label": worst_day_label,
        "worst_month": worst_month,
        "worst_month_label": worst_month_label,
        "win_days": win_days,
        "loss_days": loss_days,
    }


def build_portfolio_daily(combined, qty_a, qty_b, qty_c):
    """Build weighted portfolio daily P&L from individual strategy series."""
    return (combined["vScalpA_SL40"] * qty_a +
            combined["vScalpB"] * qty_b +
            combined["MES_v2"] * qty_c)


def fmt_dollar(val):
    if val >= 0:
        return f"${val:,.2f}"
    return f"-${abs(val):,.2f}"


def print_results_table(all_metrics):
    """Print the main comparison table."""
    print(f"\n{'='*130}")
    print(f"  WEIGHTED PORTFOLIO COMPARISON — 6 CONFIGURATIONS")
    print(f"{'='*130}")

    header = (f"  {'Config':<36} {'Total P&L':>11} {'Sharpe':>8} {'Sortino':>9} "
              f"{'MaxDD':>11} {'PF':>7} {'LoseMonths':>11} "
              f"{'WorstDay':>11} {'WorstMonth':>12}")
    print(header)
    print(f"  {'-'*36} {'-'*11} {'-'*8} {'-'*9} {'-'*11} {'-'*7} {'-'*11} {'-'*11} {'-'*12}")

    for m in all_metrics:
        line = (f"  {m['label']:<36} "
                f"{fmt_dollar(m['total_pnl']):>11} "
                f"{m['sharpe']:>8.2f} "
                f"{m['sortino']:>9.2f} "
                f"{fmt_dollar(m['max_dd']):>11} "
                f"{m['pf']:>7.2f} "
                f"{m['losing_months']:>11d} "
                f"{fmt_dollar(m['worst_day']):>11} "
                f"{fmt_dollar(m['worst_month']):>12}")
        print(line)


def print_monthly_comparison(combined, configs_to_compare):
    """Print monthly breakdown for selected configs."""
    print(f"\n{'='*130}")
    print(f"  MONTHLY P&L BREAKDOWN — KEY CONFIGURATIONS")
    print(f"{'='*130}")

    # Build monthly series for each config
    monthly_data = {}
    for idx in configs_to_compare:
        label, qa, qb, qc = CONFIGS[idx]
        port_daily = build_portfolio_daily(combined, qa, qb, qc)
        monthly = port_daily.resample("ME").sum()
        monthly.index = monthly.index.strftime("%Y-%m")
        monthly_data[label] = monthly

    # Also compute individual strategy monthly for context
    strat_monthly = {}
    for strat in ["vScalpA_SL40", "vScalpB", "MES_v2"]:
        m = combined[strat].resample("ME").sum()
        m.index = m.index.strftime("%Y-%m")
        strat_monthly[strat] = m

    # Get union of all months
    all_months = sorted(set().union(*[set(s.index) for s in monthly_data.values()]))

    # Print header
    config_labels = [CONFIGS[i][0] for i in configs_to_compare]
    short_labels = ["Baseline", "vScalpB(2)+MES", "A(1)+B(2)+MES"]

    header = f"  {'Month':<10} {'vScalpA':>10} {'vScalpB':>10} {'MES_v2':>10} | "
    header += " | ".join(f"{sl:>16}" for sl in short_labels)
    header += " | {'A helps?':>10}"
    print(header)
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}-+-" +
          "-+-".join(['-'*16] * len(short_labels)) +
          "-+-{'-'*10}")

    # Simpler header
    print()
    h1 = f"  {'Month':<10}"
    h1 += f" {'vScalpA':>10} {'vScalpB':>10} {'MES_v2':>10}"
    for sl in short_labels:
        h1 += f" | {sl:>16}"
    h1 += f" | {'A delta':>10}"
    print(h1)
    print(f"  {'-'*10}" + f" {'-'*10}" * 3 + (" |" + f" {'-'*16}") * 3 + f" | {'-'*10}")

    total_delta = 0.0
    for month in all_months:
        va = strat_monthly["vScalpA_SL40"].get(month, 0.0)
        vb = strat_monthly["vScalpB"].get(month, 0.0)
        vc = strat_monthly["MES_v2"].get(month, 0.0)

        vals = []
        for idx in configs_to_compare:
            label = CONFIGS[idx][0]
            vals.append(monthly_data[label].get(month, 0.0))

        # Delta: Config 3 (A+B2+MES) minus Config 2 (B2+MES) = marginal contribution of vScalpA
        delta = vals[2] - vals[1]
        total_delta += delta

        line = f"  {month:<10}"
        line += f" {va:>10,.2f} {vb:>10,.2f} {vc:>10,.2f}"
        for v in vals:
            line += f" | {v:>16,.2f}"
        marker = " *" if delta > 0 else ""
        line += f" | {delta:>+10,.2f}{marker}"
        print(line)

    # Totals row
    print(f"  {'-'*10}" + f" {'-'*10}" * 3 + (" |" + f" {'-'*16}") * 3 + f" | {'-'*10}")
    tot_va = strat_monthly["vScalpA_SL40"].sum()
    tot_vb = strat_monthly["vScalpB"].sum()
    tot_vc = strat_monthly["MES_v2"].sum()
    tot_vals = []
    for idx in configs_to_compare:
        label = CONFIGS[idx][0]
        tot_vals.append(monthly_data[label].sum())

    line = f"  {'TOTAL':<10}"
    line += f" {tot_va:>10,.2f} {tot_vb:>10,.2f} {tot_vc:>10,.2f}"
    for v in tot_vals:
        line += f" | {v:>16,.2f}"
    line += f" | {total_delta:>+10,.2f}"
    print(line)

    # Summary: months where vScalpA helped vs hurt
    help_count = 0
    hurt_count = 0
    help_total = 0.0
    hurt_total = 0.0
    for month in all_months:
        vals = []
        for idx in configs_to_compare:
            label = CONFIGS[idx][0]
            vals.append(monthly_data[label].get(month, 0.0))
        delta = vals[2] - vals[1]
        if delta > 0:
            help_count += 1
            help_total += delta
        elif delta < 0:
            hurt_count += 1
            hurt_total += delta

    print(f"\n  vScalpA insurance summary (A(1)+B(2)+MES vs B(2)+MES):")
    print(f"    Months where vScalpA helped: {help_count} (total: {fmt_dollar(help_total)})")
    print(f"    Months where vScalpA hurt:   {hurt_count} (total: {fmt_dollar(hurt_total)})")
    print(f"    Net marginal contribution:   {fmt_dollar(total_delta)}")


def print_correlation(combined):
    """Print daily P&L correlation matrix."""
    print(f"\n{'='*60}")
    print(f"  DAILY P&L CORRELATION MATRIX")
    print(f"{'='*60}")
    corr = combined[["vScalpA_SL40", "vScalpB", "MES_v2"]].corr()
    print(f"\n  {'':>16} {'vScalpA_SL40':>14} {'vScalpB':>14} {'MES_v2':>14}")
    for row_name in corr.index:
        vals = "  ".join(f"{corr.loc[row_name, c]:>12.3f}" for c in corr.columns)
        print(f"  {row_name:>16} {vals}")


def print_marginal_analysis(all_metrics):
    """Analyze whether vScalpA improves risk-adjusted returns at each scale."""
    print(f"\n{'='*100}")
    print(f"  MARGINAL ANALYSIS: Does vScalpA improve risk-adjusted returns?")
    print(f"{'='*100}")

    # Compare pairs: (with vScalpA) vs (without vScalpA)
    pairs = [
        (0, 1, "1 contract vScalpB: Baseline(1/1/1) vs Drop(0/1/1)"),
        (3, 2, "2 contract vScalpB: A(1)+B(2)+MES vs B(2)+MES"),
    ]

    print(f"\n  {'Comparison':<55} {'dP&L':>10} {'dSharpe':>10} {'dSortino':>10} {'dMaxDD':>10} {'dPF':>8}")
    print(f"  {'-'*55} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for with_idx, without_idx, desc in pairs:
        mw = all_metrics[with_idx]
        mo = all_metrics[without_idx]
        d_pnl = mw["total_pnl"] - mo["total_pnl"]
        d_sharpe = mw["sharpe"] - mo["sharpe"]
        d_sortino = mw["sortino"] - mo["sortino"]
        d_dd = mw["max_dd"] - mo["max_dd"]
        d_pf = mw["pf"] - mo["pf"]

        print(f"  {desc:<55} {d_pnl:>+10,.2f} {d_sharpe:>+10.2f} {d_sortino:>+10.2f} {d_dd:>+10,.2f} {d_pf:>+8.2f}")

    # Scale comparison: does scaling vScalpB improve things?
    print(f"\n  Scaling analysis:")
    print(f"  {'Comparison':<55} {'dP&L':>10} {'dSharpe':>10} {'dSortino':>10} {'dMaxDD':>10}")
    print(f"  {'-'*55} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    scale_pairs = [
        (3, 0, "B 1x->2x (keeping A=1):  (1/2/1) vs (1/1/1)"),
        (4, 0, "B 1x->3x (keeping A=1):  (1/3/1) vs (1/1/1)"),
        (5, 3, "A 1x->2x (keeping B=2):  (2/2/1) vs (1/2/1)"),
        (2, 1, "B 1x->2x (no A):         (0/2/1) vs (0/1/1)"),
    ]

    for to_idx, from_idx, desc in scale_pairs:
        mt = all_metrics[to_idx]
        mf = all_metrics[from_idx]
        d_pnl = mt["total_pnl"] - mf["total_pnl"]
        d_sharpe = mt["sharpe"] - mf["sharpe"]
        d_sortino = mt["sortino"] - mf["sortino"]
        d_dd = mt["max_dd"] - mf["max_dd"]

        print(f"  {desc:<55} {d_pnl:>+10,.2f} {d_sharpe:>+10.2f} {d_sortino:>+10.2f} {d_dd:>+10,.2f}")


def main():
    print("=" * 130)
    print("WEIGHTED PORTFOLIO COMPARISON: Scaling vScalpB with/without vScalpA insurance")
    print("=" * 130)

    # ── Step 1: Load strategy trade data ──
    print("\n--- Loading strategies ---")

    df_vscalpa = run_vscalpa_sl40()
    _, df_vscalpb = load_saved_strategy("MNQ_VSCALPB", "FULL")
    _, df_mesv2 = load_saved_strategy("MES_V2", "FULL")

    # ── Step 2: Build daily P&L series ──
    print("\n--- Computing daily P&L ---")
    daily_a = daily_pnl(df_vscalpa, "vScalpA_SL40")
    daily_b = daily_pnl(df_vscalpb, "vScalpB")
    daily_c = daily_pnl(df_mesv2, "MES_v2")

    print(f"  vScalpA (SL=40): {len(daily_a)} trading days, total ${daily_a.sum():,.2f}")
    print(f"  vScalpB:         {len(daily_b)} trading days, total ${daily_b.sum():,.2f}")
    print(f"  MES v2:          {len(daily_c)} trading days, total ${daily_c.sum():,.2f}")

    # Combine into single DataFrame with aligned dates
    all_dates = sorted(set(daily_a.index) | set(daily_b.index) | set(daily_c.index))
    idx = pd.DatetimeIndex(all_dates)

    combined = pd.DataFrame(index=idx)
    combined["vScalpA_SL40"] = daily_a.reindex(idx, fill_value=0.0)
    combined["vScalpB"] = daily_b.reindex(idx, fill_value=0.0)
    combined["MES_v2"] = daily_c.reindex(idx, fill_value=0.0)

    # ── Step 3: Compute metrics for all 6 configurations ──
    print("\n--- Computing portfolio metrics ---")
    all_metrics = []
    for label, qa, qb, qc in CONFIGS:
        port_daily = build_portfolio_daily(combined, qa, qb, qc)
        m = compute_metrics(port_daily, label)
        all_metrics.append(m)
        print(f"  {label}: P&L={fmt_dollar(m['total_pnl'])}, Sharpe={m['sharpe']:.2f}, "
              f"MaxDD={fmt_dollar(m['max_dd'])}, Sortino={m['sortino']:.2f}")

    # ── Step 4: Print results ──
    print_results_table(all_metrics)
    print_correlation(combined)
    print_monthly_comparison(combined, MONTHLY_CONFIGS)
    print_marginal_analysis(all_metrics)

    # ── Step 5: Individual strategy standalone metrics ──
    print(f"\n{'='*80}")
    print(f"  INDIVIDUAL STRATEGY STANDALONE METRICS")
    print(f"{'='*80}")
    for strat, label in [("vScalpA_SL40", "vScalpA (SL=40)"),
                         ("vScalpB", "vScalpB"),
                         ("MES_v2", "MES v2")]:
        m = compute_metrics(combined[strat], label)
        print(f"\n  {m['label']}")
        print(f"    Total P&L:     {fmt_dollar(m['total_pnl']):>12}")
        print(f"    Sharpe:        {m['sharpe']:>12.2f}")
        print(f"    Sortino:       {m['sortino']:>12.2f}")
        print(f"    Max DD:        {fmt_dollar(m['max_dd']):>12}")
        print(f"    PF:            {m['pf']:>12.2f}")
        print(f"    Worst Day:     {fmt_dollar(m['worst_day']):>12} ({m['worst_day_label']})")
        print(f"    Worst Month:   {fmt_dollar(m['worst_month']):>12} ({m['worst_month_label']})")
        print(f"    Losing Months: {m['losing_months']:>12d}")

    # ── Step 6: Final verdict ──
    print(f"\n{'='*130}")
    print(f"  VERDICT: KEY QUESTION")
    print(f"  Does vScalpA(1) + vScalpB(2) + MES(1) beat vScalpB(2) + MES(1)?")
    print(f"{'='*130}")

    m_with = all_metrics[3]    # Keep both + vScalpB 2x (1/2/1)
    m_without = all_metrics[2]  # Scale vScalpB only (0/2/1)

    delta_pnl = m_with["total_pnl"] - m_without["total_pnl"]
    delta_sharpe = m_with["sharpe"] - m_without["sharpe"]
    delta_sortino = m_with["sortino"] - m_without["sortino"]
    delta_dd = m_with["max_dd"] - m_without["max_dd"]
    delta_pf = m_with["pf"] - m_without["pf"]

    print(f"\n  Config WITH vScalpA:    {m_with['label']}")
    print(f"    P&L={fmt_dollar(m_with['total_pnl'])}, Sharpe={m_with['sharpe']:.2f}, "
          f"Sortino={m_with['sortino']:.2f}, MaxDD={fmt_dollar(m_with['max_dd'])}, "
          f"PF={m_with['pf']:.2f}, Losing Months={m_with['losing_months']}")

    print(f"\n  Config WITHOUT vScalpA: {m_without['label']}")
    print(f"    P&L={fmt_dollar(m_without['total_pnl'])}, Sharpe={m_without['sharpe']:.2f}, "
          f"Sortino={m_without['sortino']:.2f}, MaxDD={fmt_dollar(m_without['max_dd'])}, "
          f"PF={m_without['pf']:.2f}, Losing Months={m_without['losing_months']}")

    print(f"\n  DELTAS (with - without):")
    print(f"    P&L:           {delta_pnl:>+10,.2f}")
    print(f"    Sharpe:        {delta_sharpe:>+10.2f}")
    print(f"    Sortino:       {delta_sortino:>+10.2f}")
    print(f"    Max DD:        {delta_dd:>+10,.2f} ({'worse' if delta_dd < 0 else 'better'})")
    print(f"    PF:            {delta_pf:>+10.2f}")
    print(f"    Losing Months: {m_with['losing_months'] - m_without['losing_months']:>+10d}")

    # Decision logic
    if delta_sharpe > 0.05 and delta_sortino > 0:
        print(f"\n  >>> KEEP vScalpA as insurance — clearly improves risk-adjusted returns")
    elif delta_sharpe > 0 and delta_dd >= 0:
        print(f"\n  >>> MARGINAL KEEP — slight improvement with no drawdown cost")
    elif delta_sharpe > 0 and delta_dd < -200:
        print(f"\n  >>> MIXED — improves Sharpe but worsens drawdown significantly")
    elif delta_sharpe <= 0 and delta_pnl > 0:
        print(f"\n  >>> DROP vScalpA — adds raw P&L but worsens risk-adjusted returns")
    elif delta_sharpe <= 0 and delta_pnl <= 0:
        print(f"\n  >>> DROP vScalpA — hurts both returns and risk-adjusted metrics")
    else:
        print(f"\n  >>> INCONCLUSIVE — review monthly breakdown for regime-specific value")

    # Best overall config
    print(f"\n  {'='*80}")
    print(f"  BEST CONFIGURATION BY METRIC:")
    best_sharpe = max(all_metrics, key=lambda m: m["sharpe"])
    best_sortino = max(all_metrics, key=lambda m: m["sortino"])
    best_pf = max(all_metrics, key=lambda m: m["pf"])
    best_pnl = max(all_metrics, key=lambda m: m["total_pnl"])
    least_dd = max(all_metrics, key=lambda m: m["max_dd"])  # max_dd is negative, so max = least bad

    print(f"    Best Sharpe:   {best_sharpe['label']} ({best_sharpe['sharpe']:.2f})")
    print(f"    Best Sortino:  {best_sortino['label']} ({best_sortino['sortino']:.2f})")
    print(f"    Best PF:       {best_pf['label']} ({best_pf['pf']:.2f})")
    print(f"    Best P&L:      {best_pnl['label']} ({fmt_dollar(best_pnl['total_pnl'])})")
    print(f"    Least MaxDD:   {least_dd['label']} ({fmt_dollar(least_dd['max_dd'])})")


if __name__ == "__main__":
    main()
