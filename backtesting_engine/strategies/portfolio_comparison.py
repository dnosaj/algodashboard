"""
Portfolio Comparison: Is vScalpA worth keeping?

Compare:
  Portfolio A: vScalpA (SL=40) + vScalpB + MES v2
  Portfolio B: vScalpB + MES v2 only

Metrics: total P&L, Sharpe, max drawdown, monthly breakdown, correlations.
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
)
from results.save_results import load_backtest

# ── Constants ──
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
VSCALPA_SL40 = 40  # New SL for vScalpA


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
    )

    sc = score_trades(trades, commission_per_side=MNQ_COMMISSION,
                      dollar_per_pt=MNQ_DOLLAR_PER_PT)
    print(f"  {fmt_score(sc, 'vScalpA SL=40')}")

    # Convert to DataFrame matching saved CSV format
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
    filepath = files[-1]  # most recent
    meta, df = load_backtest(filepath)
    print(f"  Loaded {strategy}: {len(df)} trades, total P&L: ${meta['total_pnl']:.2f} from {filepath.name}")
    return meta, df


def daily_pnl(df, label):
    """Aggregate trade-level data to daily P&L series."""
    daily = df.groupby("trade_date")["pnl_dollar"].sum()
    daily.index = pd.to_datetime(daily.index)
    daily.name = label
    return daily


def compute_metrics(daily_pnl_series, label):
    """Compute key portfolio metrics from a daily P&L series."""
    total = daily_pnl_series.sum()
    n_days = len(daily_pnl_series)
    mean_daily = daily_pnl_series.mean()
    std_daily = daily_pnl_series.std()

    # Annualized Sharpe (252 trading days)
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

    # Max drawdown from cumulative equity curve
    cum = daily_pnl_series.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()

    # Win rate of days
    win_days = (daily_pnl_series > 0).sum()
    loss_days = (daily_pnl_series < 0).sum()
    flat_days = (daily_pnl_series == 0).sum()

    # Profit factor (sum of winning days / abs sum of losing days)
    gross_profit = daily_pnl_series[daily_pnl_series > 0].sum()
    gross_loss = abs(daily_pnl_series[daily_pnl_series < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "label": label,
        "total_pnl": total,
        "n_trading_days": n_days,
        "mean_daily_pnl": mean_daily,
        "std_daily_pnl": std_daily,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "win_days": win_days,
        "loss_days": loss_days,
        "flat_days": flat_days,
        "daily_win_rate": win_days / (win_days + loss_days) if (win_days + loss_days) > 0 else 0,
    }


def monthly_breakdown(daily_pnl_series, label):
    """Monthly P&L breakdown."""
    monthly = daily_pnl_series.resample("ME").sum()
    monthly.index = monthly.index.strftime("%Y-%m")
    return monthly


def print_metrics(m):
    """Pretty-print metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {m['label']}")
    print(f"{'='*60}")
    print(f"  Total P&L:        ${m['total_pnl']:>10,.2f}")
    print(f"  Trading Days:     {m['n_trading_days']:>10d}")
    print(f"  Mean Daily P&L:   ${m['mean_daily_pnl']:>10,.2f}")
    print(f"  Std Daily P&L:    ${m['std_daily_pnl']:>10,.2f}")
    print(f"  Sharpe (ann.):    {m['sharpe']:>10.2f}")
    print(f"  Max Drawdown:     ${m['max_drawdown']:>10,.2f}")
    print(f"  Profit Factor:    {m['profit_factor']:>10.2f}")
    print(f"  Win Days:         {m['win_days']:>10d}")
    print(f"  Loss Days:        {m['loss_days']:>10d}")
    print(f"  Daily Win Rate:   {m['daily_win_rate']*100:>9.1f}%")


def fmt_dollar(val):
    """Format dollar value with comma separator."""
    return f"${val:,.2f}"


def print_comparison(ma, mb):
    """Side-by-side comparison."""
    print(f"\n{'='*70}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Metric':<22} {'Portfolio A (3-strat)':>22} {'Portfolio B (2-strat)':>22}"
    print(header)
    print(f"  {'-'*22} {'-'*22} {'-'*22}")

    rows = [
        ("Total P&L", fmt_dollar(ma['total_pnl']), fmt_dollar(mb['total_pnl'])),
        ("Mean Daily P&L", fmt_dollar(ma['mean_daily_pnl']), fmt_dollar(mb['mean_daily_pnl'])),
        ("Std Daily P&L", fmt_dollar(ma['std_daily_pnl']), fmt_dollar(mb['std_daily_pnl'])),
        ("Sharpe (ann.)", f"{ma['sharpe']:.2f}", f"{mb['sharpe']:.2f}"),
        ("Max Drawdown", fmt_dollar(ma['max_drawdown']), fmt_dollar(mb['max_drawdown'])),
        ("Profit Factor", f"{ma['profit_factor']:.2f}", f"{mb['profit_factor']:.2f}"),
        ("Daily Win Rate", f"{ma['daily_win_rate']*100:.1f}%", f"{mb['daily_win_rate']*100:.1f}%"),
    ]
    for metric, val_a, val_b in rows:
        print(f"  {metric:<22} {val_a:>22} {val_b:>22}")

    # Marginal contribution
    delta_pnl = ma['total_pnl'] - mb['total_pnl']
    delta_sharpe = ma['sharpe'] - mb['sharpe']
    delta_dd = ma['max_drawdown'] - mb['max_drawdown']
    dd_label = "worse" if delta_dd < 0 else "better"
    print(f"\n  MARGINAL CONTRIBUTION OF vScalpA (SL=40):")
    print(f"  {'Delta P&L':<22} {fmt_dollar(delta_pnl):>22}")
    print(f"  {'Delta Sharpe':<22} {delta_sharpe:>+22.2f}")
    print(f"  {'Delta Max DD':<22} {fmt_dollar(delta_dd):>22} ({dd_label})")


def main():
    print("=" * 70)
    print("PORTFOLIO COMPARISON: Is vScalpA worth keeping?")
    print("=" * 70)

    # ── Step 1: Load / generate strategy trade data ──
    print("\n--- Loading strategies ---")

    # vScalpA SL=40 — must re-run
    df_vscalpa = run_vscalpa_sl40()

    # vScalpB — load from saved
    _, df_vscalpb = load_saved_strategy("MNQ_VSCALPB", "FULL")

    # MES v2 — load from saved
    _, df_mesv2 = load_saved_strategy("MES_V2", "FULL")

    # ── Step 2: Daily P&L series ──
    print("\n--- Computing daily P&L ---")
    daily_a = daily_pnl(df_vscalpa, "vScalpA_SL40")
    daily_b = daily_pnl(df_vscalpb, "vScalpB")
    daily_c = daily_pnl(df_mesv2, "MES_v2")

    print(f"  vScalpA (SL=40): {len(daily_a)} trading days, total ${daily_a.sum():,.2f}")
    print(f"  vScalpB:         {len(daily_b)} trading days, total ${daily_b.sum():,.2f}")
    print(f"  MES v2:          {len(daily_c)} trading days, total ${daily_c.sum():,.2f}")

    # ── Step 3: Build portfolio daily P&L ──
    # Combine all three into a single DataFrame, fill missing days with 0
    all_dates = sorted(set(daily_a.index) | set(daily_b.index) | set(daily_c.index))
    idx = pd.DatetimeIndex(all_dates)

    combined = pd.DataFrame(index=idx)
    combined["vScalpA_SL40"] = daily_a.reindex(idx, fill_value=0.0)
    combined["vScalpB"] = daily_b.reindex(idx, fill_value=0.0)
    combined["MES_v2"] = daily_c.reindex(idx, fill_value=0.0)

    # Portfolio A: all 3
    combined["Portfolio_A"] = combined["vScalpA_SL40"] + combined["vScalpB"] + combined["MES_v2"]
    # Portfolio B: vScalpB + MES v2 only
    combined["Portfolio_B"] = combined["vScalpB"] + combined["MES_v2"]

    # ── Step 4: Compute metrics ──
    print("\n--- Individual Strategy Metrics ---")
    m_a = compute_metrics(combined["vScalpA_SL40"], "vScalpA (SL=40) — standalone")
    m_b = compute_metrics(combined["vScalpB"], "vScalpB — standalone")
    m_c = compute_metrics(combined["MES_v2"], "MES v2 — standalone")

    print_metrics(m_a)
    print_metrics(m_b)
    print_metrics(m_c)

    print("\n--- Portfolio Metrics ---")
    m_pa = compute_metrics(combined["Portfolio_A"], "Portfolio A: vScalpA + vScalpB + MES v2")
    m_pb = compute_metrics(combined["Portfolio_B"], "Portfolio B: vScalpB + MES v2 only")

    print_metrics(m_pa)
    print_metrics(m_pb)

    print_comparison(m_pa, m_pb)

    # ── Step 5: Correlation matrix ──
    print(f"\n{'='*70}")
    print(f"  DAILY P&L CORRELATION MATRIX")
    print(f"{'='*70}")
    corr = combined[["vScalpA_SL40", "vScalpB", "MES_v2"]].corr()
    print(f"\n  {'':>16} {'vScalpA_SL40':>14} {'vScalpB':>14} {'MES_v2':>14}")
    for row_name in corr.index:
        vals = "  ".join(f"{corr.loc[row_name, c]:>12.3f}" for c in corr.columns)
        print(f"  {row_name:>16} {vals}")

    # ── Step 6: Monthly breakdown ──
    print(f"\n{'='*70}")
    print(f"  MONTHLY P&L BREAKDOWN")
    print(f"{'='*70}")

    monthly_a = monthly_breakdown(combined["vScalpA_SL40"], "vScalpA")
    monthly_b = monthly_breakdown(combined["vScalpB"], "vScalpB")
    monthly_c = monthly_breakdown(combined["MES_v2"], "MES_v2")
    monthly_pa = monthly_breakdown(combined["Portfolio_A"], "Port A")
    monthly_pb = monthly_breakdown(combined["Portfolio_B"], "Port B")

    monthly_all = pd.DataFrame({
        "vScalpA_SL40": monthly_a,
        "vScalpB": monthly_b,
        "MES_v2": monthly_c,
        "Port_A (3-strat)": monthly_pa,
        "Port_B (2-strat)": monthly_pb,
    }).fillna(0)

    print(f"\n  {'Month':<10} {'vScalpA':>10} {'vScalpB':>10} {'MES_v2':>10} {'Port A':>10} {'Port B':>10} {'A-B delta':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for month in monthly_all.index:
        va = monthly_all.loc[month, "vScalpA_SL40"]
        vb = monthly_all.loc[month, "vScalpB"]
        vc = monthly_all.loc[month, "MES_v2"]
        pa = monthly_all.loc[month, "Port_A (3-strat)"]
        pb = monthly_all.loc[month, "Port_B (2-strat)"]
        delta = pa - pb
        print(f"  {month:<10} {va:>10,.2f} {vb:>10,.2f} {vc:>10,.2f} {pa:>10,.2f} {pb:>10,.2f} {delta:>+10,.2f}")

    # Totals
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<10} {monthly_all['vScalpA_SL40'].sum():>10,.2f} {monthly_all['vScalpB'].sum():>10,.2f} {monthly_all['MES_v2'].sum():>10,.2f} {monthly_all['Port_A (3-strat)'].sum():>10,.2f} {monthly_all['Port_B (2-strat)'].sum():>10,.2f} {(monthly_all['Port_A (3-strat)'].sum() - monthly_all['Port_B (2-strat)'].sum()):>+10,.2f}")

    # ── Step 7: Risk-adjusted analysis ──
    print(f"\n{'='*70}")
    print(f"  RISK-ADJUSTED ANALYSIS")
    print(f"{'='*70}")

    # Calmar ratio (annualized return / max drawdown)
    for m in [m_pa, m_pb]:
        ann_return = m['mean_daily_pnl'] * 252
        calmar = ann_return / abs(m['max_drawdown']) if m['max_drawdown'] != 0 else float('inf')
        print(f"  {m['label']}")
        print(f"    Annualized Return: ${ann_return:,.2f}")
        print(f"    Calmar Ratio:      {calmar:.2f}")

    # Downside deviation and Sortino
    for label, series in [("Portfolio A", combined["Portfolio_A"]),
                          ("Portfolio B", combined["Portfolio_B"])]:
        downside = series[series < 0]
        downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0
        sortino = (series.mean() / downside_dev * np.sqrt(252)) if downside_dev > 0 else float('inf')
        print(f"  {label}")
        print(f"    Downside Deviation: ${downside_dev:,.2f}")
        print(f"    Sortino Ratio:      {sortino:.2f}")

    # ── Step 8: Worst streaks ──
    print(f"\n{'='*70}")
    print(f"  WORST DRAWDOWN PERIODS")
    print(f"{'='*70}")
    for label, series in [("Portfolio A (3-strat)", combined["Portfolio_A"]),
                          ("Portfolio B (2-strat)", combined["Portfolio_B"])]:
        cum = series.cumsum()
        running_max = cum.cummax()
        dd = cum - running_max
        worst_idx = dd.idxmin()
        # Find the peak before the worst drawdown
        peak_idx = cum[:worst_idx].idxmax()
        print(f"  {label}")
        print(f"    Peak:     {peak_idx.strftime('%Y-%m-%d')} (equity ${cum[peak_idx]:,.2f})")
        print(f"    Trough:   {worst_idx.strftime('%Y-%m-%d')} (equity ${cum[worst_idx]:,.2f})")
        print(f"    Drawdown: ${dd[worst_idx]:,.2f}")
        print(f"    Duration: {(worst_idx - peak_idx).days} calendar days")

    # ── Step 9: Overlap analysis ──
    print(f"\n{'='*70}")
    print(f"  TRADE OVERLAP ANALYSIS (same-day activity)")
    print(f"{'='*70}")
    dates_a = set(df_vscalpa["trade_date"])
    dates_b = set(df_vscalpb["trade_date"])
    dates_c = set(df_mesv2["trade_date"])

    overlap_ab = dates_a & dates_b
    overlap_ac = dates_a & dates_c
    overlap_bc = dates_b & dates_c
    overlap_all = dates_a & dates_b & dates_c

    print(f"  vScalpA active days:     {len(dates_a)}")
    print(f"  vScalpB active days:     {len(dates_b)}")
    print(f"  MES v2 active days:      {len(dates_c)}")
    print(f"  vScalpA & vScalpB:       {len(overlap_ab)} days overlap")
    print(f"  vScalpA & MES v2:        {len(overlap_ac)} days overlap")
    print(f"  vScalpB & MES v2:        {len(overlap_bc)} days overlap")
    print(f"  All three:               {len(overlap_all)} days overlap")

    # Days where vScalpA trades but vScalpB doesn't (unique contribution)
    unique_a = dates_a - dates_b - dates_c
    unique_a_not_b = dates_a - dates_b
    print(f"\n  vScalpA-only days (not in vScalpB or MES): {len(unique_a)}")
    print(f"  vScalpA days not in vScalpB:               {len(unique_a_not_b)}")

    # P&L on unique vScalpA days
    if unique_a_not_b:
        unique_pnl = df_vscalpa[df_vscalpa["trade_date"].isin(unique_a_not_b)]["pnl_dollar"].sum()
        print(f"  P&L on vScalpA-only days (vs vScalpB):     ${unique_pnl:,.2f}")

    # ── Final verdict ──
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    delta_sharpe = m_pa['sharpe'] - m_pb['sharpe']
    delta_dd = m_pa['max_drawdown'] - m_pb['max_drawdown']
    delta_pnl = m_pa['total_pnl'] - m_pb['total_pnl']

    if delta_sharpe > 0.05 and delta_dd >= -50:
        verdict = "KEEP vScalpA — improves risk-adjusted returns"
    elif delta_sharpe > 0 and delta_pnl > 0:
        verdict = "MARGINAL KEEP — small improvement, worth monitoring"
    elif delta_sharpe <= 0 and delta_pnl <= 0:
        verdict = "DROP vScalpA — hurts both returns and risk metrics"
    elif delta_sharpe <= 0 and delta_pnl > 0:
        verdict = "DROP vScalpA — adds P&L but worsens risk-adjusted returns"
    else:
        verdict = "INCONCLUSIVE — review monthly breakdown and correlation"

    print(f"\n  {verdict}")
    print(f"  Delta P&L:    ${delta_pnl:>+,.2f}")
    print(f"  Delta Sharpe: {delta_sharpe:>+.2f}")
    print(f"  Delta Max DD: ${delta_dd:>+,.2f}")


if __name__ == "__main__":
    main()
