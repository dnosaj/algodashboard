#!/usr/bin/env python3
"""
VIX Correlation Analysis — Phase 1
====================================
Analyzes whether VIX levels correlate with trade outcomes across all
active strategies. If VIX predicts bad trades, it can be used as a
pre-filter to improve Sharpe/PF.

Usage:
    python3 vix_correlation_analysis.py
    python3 vix_correlation_analysis.py --plot   # Save charts to disk

Output:
    - Console tables: stats by VIX quintile, threshold sweep, per-strategy breakdown
    - Charts (optional): scatter, box plots, equity curves saved to results/vix_analysis/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "vix_analysis"

# Trade archive files (FULL 12-month runs)
TRADE_FILES = {
    "MNQ_V15": "MNQ_V15_FULL_2025-02-17_to_2026-02-19_20260223_002951.csv",
    "MNQ_VSCALPB": "MNQ_VSCALPB_FULL_2025-02-17_to_2026-02-19_20260223_002951.csv",
    "MES_V2": "MES_V2_FULL_2025-02-17_to_2026-02-19_20260223_002952.csv",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades() -> pd.DataFrame:
    """Load all trade CSVs and combine into a single DataFrame."""
    frames = []
    for strategy, filename in TRADE_FILES.items():
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"  WARNING: {filepath.name} not found, skipping")
            continue
        df = pd.read_csv(filepath, comment="#")
        df["strategy"] = strategy
        frames.append(df)
        print(f"  Loaded {len(df)} trades from {filename}")

    if not frames:
        print("ERROR: No trade files found!")
        sys.exit(1)

    trades = pd.concat(frames, ignore_index=True)
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades["win"] = trades["pnl_dollar"] > 0
    print(f"\n  Total: {len(trades)} trades across {trades['strategy'].nunique()} strategies")
    return trades


def load_vix(start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily VIX data from Yahoo Finance."""
    import yfinance as yf

    print(f"\n  Downloading VIX data ({start_date} to {end_date})...")
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)

    if vix.empty:
        print("ERROR: No VIX data returned from Yahoo Finance!")
        sys.exit(1)

    # yfinance may return MultiIndex columns; flatten if needed
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vix = vix.reset_index()
    vix = vix.rename(columns={"Date": "date", "Open": "vix_open", "High": "vix_high",
                               "Low": "vix_low", "Close": "vix_close"})
    vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
    vix = vix[["date", "vix_open", "vix_high", "vix_low", "vix_close"]].copy()

    # Day-over-day VIX change
    vix["vix_change"] = vix["vix_close"].pct_change() * 100  # percent
    vix["vix_change_abs"] = vix["vix_close"].diff()  # absolute points

    print(f"  Got {len(vix)} VIX trading days (range: {vix['vix_close'].min():.1f} - {vix['vix_close'].max():.1f})")
    return vix


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame, label: str = "") -> dict:
    """Compute trading stats for a group of trades."""
    if len(df) == 0:
        return {"label": label, "n": 0}

    wins = df["pnl_dollar"] > 0
    gross_profit = df.loc[wins, "pnl_dollar"].sum()
    gross_loss = abs(df.loc[~wins, "pnl_dollar"].sum())
    total_pnl = df["pnl_dollar"].sum()

    # Daily P&L for Sharpe
    daily = df.groupby("trade_date")["pnl_dollar"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0

    return {
        "label": label,
        "n": len(df),
        "wr": wins.mean() * 100,
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(df),
        "sharpe": sharpe,
        "avg_win": df.loc[wins, "pnl_dollar"].mean() if wins.any() else 0,
        "avg_loss": df.loc[~wins, "pnl_dollar"].mean() if (~wins).any() else 0,
    }


def print_stats_table(rows: list[dict], title: str):
    """Print a formatted stats table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Bucket':<20} {'N':>5} {'WR%':>7} {'PF':>7} {'Total$':>9} {'Avg$':>8} {'Sharpe':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*7}")
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['label']:<20} {r['n']:>5}   {'---':>5} {'---':>5} {'---':>7} {'---':>6} {'---':>5}")
            continue
        pf_str = f"{r['pf']:.3f}" if r['pf'] < 100 else "inf"
        print(f"  {r['label']:<20} {r['n']:>5} {r['wr']:>6.1f}% {pf_str:>7} {r['total_pnl']:>+9.0f} {r['avg_pnl']:>+7.2f} {r['sharpe']:>7.2f}")


def analyze_vix_quintiles(merged: pd.DataFrame):
    """Break trades into VIX quintiles and compare stats."""
    merged["vix_quintile"] = pd.qcut(merged["vix_close"], 5, labels=False, duplicates="drop")
    quintile_edges = merged.groupby("vix_quintile")["vix_close"].agg(["min", "max"])

    rows = []
    for q in sorted(merged["vix_quintile"].unique()):
        subset = merged[merged["vix_quintile"] == q]
        lo, hi = quintile_edges.loc[q, "min"], quintile_edges.loc[q, "max"]
        label = f"Q{q+1} ({lo:.1f}-{hi:.1f})"
        rows.append(compute_stats(subset, label))

    rows.append(compute_stats(merged, "ALL TRADES"))
    print_stats_table(rows, "Trade Stats by VIX Quintile (all strategies)")
    return rows


def analyze_vix_quintiles_per_strategy(merged: pd.DataFrame):
    """VIX quintile breakdown per strategy."""
    merged["vix_quintile"] = pd.qcut(merged["vix_close"], 5, labels=False, duplicates="drop")
    quintile_edges = merged.groupby("vix_quintile")["vix_close"].agg(["min", "max"])

    for strat in sorted(merged["strategy"].unique()):
        strat_df = merged[merged["strategy"] == strat]
        rows = []
        for q in sorted(merged["vix_quintile"].unique()):
            subset = strat_df[strat_df["vix_quintile"] == q]
            lo, hi = quintile_edges.loc[q, "min"], quintile_edges.loc[q, "max"]
            label = f"Q{q+1} ({lo:.1f}-{hi:.1f})"
            rows.append(compute_stats(subset, label))
        rows.append(compute_stats(strat_df, "ALL"))
        print_stats_table(rows, f"VIX Quintile Breakdown — {strat}")


def analyze_vix_change(merged: pd.DataFrame):
    """Analyze if day-over-day VIX change predicts trade outcomes."""
    # VIX rising vs falling vs flat
    merged["vix_regime"] = pd.cut(
        merged["vix_change"],
        bins=[-999, -5, -1, 1, 5, 999],
        labels=["Big drop (<-5%)", "Small drop", "Flat (±1%)", "Small rise", "Big rise (>+5%)"]
    )

    rows = []
    for regime in merged["vix_regime"].cat.categories:
        subset = merged[merged["vix_regime"] == regime]
        rows.append(compute_stats(subset, str(regime)))
    rows.append(compute_stats(merged, "ALL TRADES"))
    print_stats_table(rows, "Trade Stats by Day-over-Day VIX Change")


def analyze_vix_intraday_range(merged: pd.DataFrame):
    """Analyze if VIX intraday range (high-low) predicts trade outcomes."""
    merged["vix_range"] = merged["vix_high"] - merged["vix_low"]
    merged["vix_range_q"] = pd.qcut(merged["vix_range"], 4, labels=False, duplicates="drop")
    range_edges = merged.groupby("vix_range_q")["vix_range"].agg(["min", "max"])

    rows = []
    for q in sorted(merged["vix_range_q"].unique()):
        subset = merged[merged["vix_range_q"] == q]
        lo, hi = range_edges.loc[q, "min"], range_edges.loc[q, "max"]
        label = f"Q{q+1} ({lo:.1f}-{hi:.1f})"
        rows.append(compute_stats(subset, label))
    rows.append(compute_stats(merged, "ALL TRADES"))
    print_stats_table(rows, "Trade Stats by VIX Intraday Range (High - Low)")


def sweep_vix_threshold(merged: pd.DataFrame):
    """Sweep 'skip trades when VIX > X' and find optimal threshold."""
    print(f"\n{'='*80}")
    print(f"  VIX Threshold Sweep: 'Skip trades when VIX close > X'")
    print(f"{'='*80}")
    print(f"  {'VIX Max':<10} {'N':>5} {'Skipped':>8} {'WR%':>7} {'PF':>7} {'Total$':>9} {'Sharpe':>7} {'$/trade':>8}")
    print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*8}")

    baseline = compute_stats(merged, "baseline")
    thresholds = list(range(14, 40, 1))
    best_sharpe = baseline["sharpe"]
    best_threshold = None
    results = []

    for thresh in thresholds:
        subset = merged[merged["vix_close"] <= thresh]
        skipped = len(merged) - len(subset)
        if len(subset) < 20:
            continue
        s = compute_stats(subset, f"VIX≤{thresh}")
        pf_str = f"{s['pf']:.3f}" if s['pf'] < 100 else "inf"
        marker = " <-- BEST" if s["sharpe"] > best_sharpe else ""
        if s["sharpe"] > best_sharpe:
            best_sharpe = s["sharpe"]
            best_threshold = thresh
        print(f"  VIX≤{thresh:<6} {s['n']:>5} {skipped:>8} {s['wr']:>6.1f}% {pf_str:>7} {s['total_pnl']:>+9.0f} {s['sharpe']:>7.2f} {s['avg_pnl']:>+7.2f}{marker}")
        results.append({"threshold": thresh, **s})

    print(f"\n  Baseline: {baseline['n']} trades, Sharpe {baseline['sharpe']:.2f}, PF {baseline['pf']:.3f}")
    if best_threshold:
        print(f"  Best:     VIX≤{best_threshold}, Sharpe {best_sharpe:.2f}")
    else:
        print(f"  No threshold improved over baseline.")
    return results


def sweep_vix_threshold_per_strategy(merged: pd.DataFrame):
    """Per-strategy VIX threshold sweep."""
    for strat in sorted(merged["strategy"].unique()):
        strat_df = merged[merged["strategy"] == strat]
        print(f"\n{'='*80}")
        print(f"  VIX Threshold Sweep — {strat}")
        print(f"{'='*80}")
        print(f"  {'VIX Max':<10} {'N':>5} {'Skipped':>8} {'WR%':>7} {'PF':>7} {'Total$':>9} {'Sharpe':>7}")
        print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*7}")

        baseline = compute_stats(strat_df, "baseline")
        best_sharpe = baseline["sharpe"]
        best_threshold = None

        for thresh in range(14, 40, 1):
            subset = strat_df[strat_df["vix_close"] <= thresh]
            skipped = len(strat_df) - len(subset)
            if len(subset) < 10:
                continue
            s = compute_stats(subset, f"VIX≤{thresh}")
            pf_str = f"{s['pf']:.3f}" if s['pf'] < 100 else "inf"
            marker = ""
            if s["sharpe"] > best_sharpe:
                best_sharpe = s["sharpe"]
                best_threshold = thresh
                marker = " <--"
            print(f"  VIX≤{thresh:<6} {s['n']:>5} {skipped:>8} {s['wr']:>6.1f}% {pf_str:>7} {s['total_pnl']:>+9.0f} {s['sharpe']:>7.2f}{marker}")

        print(f"\n  Baseline: {baseline['n']} trades, Sharpe {baseline['sharpe']:.2f}, PF {baseline['pf']:.3f}")
        if best_threshold:
            print(f"  Best:     VIX≤{best_threshold}, Sharpe {best_sharpe:.2f}")


def analyze_sl_clustering(merged: pd.DataFrame):
    """Check if stop-loss hits cluster at specific VIX levels."""
    sl_trades = merged[merged["exit_reason"] == "SL"]
    non_sl = merged[merged["exit_reason"] != "SL"]

    print(f"\n{'='*80}")
    print(f"  Stop-Loss Clustering Analysis")
    print(f"{'='*80}")
    print(f"  SL trades: {len(sl_trades)}  |  Non-SL trades: {len(non_sl)}")
    print(f"  SL mean VIX:     {sl_trades['vix_close'].mean():.2f}")
    print(f"  Non-SL mean VIX: {non_sl['vix_close'].mean():.2f}")
    print(f"  Difference:      {sl_trades['vix_close'].mean() - non_sl['vix_close'].mean():+.2f}")
    print()

    # VIX distribution comparison
    print(f"  VIX percentiles:")
    print(f"  {'':20} {'SL':>8} {'Non-SL':>8} {'All':>8}")
    for pct in [10, 25, 50, 75, 90]:
        sl_val = sl_trades["vix_close"].quantile(pct/100) if len(sl_trades) else 0
        nsl_val = non_sl["vix_close"].quantile(pct/100) if len(non_sl) else 0
        all_val = merged["vix_close"].quantile(pct/100)
        print(f"  P{pct:<18} {sl_val:>8.2f} {nsl_val:>8.2f} {all_val:>8.2f}")

    # SL rate by VIX quintile
    merged["vix_q"] = pd.qcut(merged["vix_close"], 5, labels=False, duplicates="drop")
    print(f"\n  SL Rate by VIX Quintile:")
    print(f"  {'Quintile':<20} {'Trades':>6} {'SLs':>5} {'SL%':>7}")
    for q in sorted(merged["vix_q"].unique()):
        subset = merged[merged["vix_q"] == q]
        sl_count = (subset["exit_reason"] == "SL").sum()
        sl_rate = sl_count / len(subset) * 100
        vix_lo = subset["vix_close"].min()
        vix_hi = subset["vix_close"].max()
        print(f"  Q{q+1} ({vix_lo:.1f}-{vix_hi:.1f}){'':<5} {len(subset):>6} {sl_count:>5} {sl_rate:>6.1f}%")


def generate_plots(merged: pd.DataFrame, output_dir: Path):
    """Generate and save analysis charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scatter: VIX at entry vs trade P&L
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, strat in enumerate(sorted(merged["strategy"].unique())):
        ax = axes[i]
        subset = merged[merged["strategy"] == strat]
        colors = ["green" if p > 0 else "red" for p in subset["pnl_dollar"]]
        ax.scatter(subset["vix_close"], subset["pnl_dollar"], c=colors, alpha=0.4, s=15)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.set_xlabel("VIX Close")
        ax.set_ylabel("Trade P&L ($)")
        ax.set_title(f"{strat}")
    fig.suptitle("VIX at Entry vs Trade P&L", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "vix_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/vix_scatter.png")

    # 2. Box plot: P&L by VIX quintile
    merged["vix_quintile"] = pd.qcut(merged["vix_close"], 5, labels=False, duplicates="drop")
    quintile_edges = merged.groupby("vix_quintile")["vix_close"].agg(["min", "max"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, strat in enumerate(sorted(merged["strategy"].unique())):
        ax = axes[i]
        subset = merged[merged["strategy"] == strat]
        data = [subset[subset["vix_quintile"] == q]["pnl_dollar"].values
                for q in sorted(subset["vix_quintile"].unique())]
        labels = [f"Q{q+1}\n{quintile_edges.loc[q, 'min']:.0f}-{quintile_edges.loc[q, 'max']:.0f}"
                  for q in sorted(subset["vix_quintile"].unique())]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4a90d9")
            patch.set_alpha(0.6)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.set_xlabel("VIX Quintile")
        ax.set_ylabel("Trade P&L ($)")
        ax.set_title(f"{strat}")
    fig.suptitle("Trade P&L Distribution by VIX Quintile", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "vix_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/vix_boxplot.png")

    # 3. SL rate by VIX quintile (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    for strat in sorted(merged["strategy"].unique()):
        subset = merged[merged["strategy"] == strat]
        sl_rates = []
        for q in sorted(subset["vix_quintile"].unique()):
            q_data = subset[subset["vix_quintile"] == q]
            sl_rate = (q_data["exit_reason"] == "SL").mean() * 100
            sl_rates.append(sl_rate)
        ax.plot(sorted(subset["vix_quintile"].unique()), sl_rates, marker="o", label=strat)
    ax.set_xlabel("VIX Quintile")
    ax.set_ylabel("SL Hit Rate (%)")
    ax.set_title("Stop-Loss Rate by VIX Quintile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "vix_sl_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/vix_sl_rate.png")

    # 4. Equity curve: filtered vs unfiltered (using best threshold from sweep)
    # Find best threshold per strategy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, strat in enumerate(sorted(merged["strategy"].unique())):
        ax = axes[i]
        strat_df = merged[merged["strategy"] == strat].sort_values("entry_time")

        # Baseline equity curve
        baseline_eq = strat_df["pnl_dollar"].cumsum()
        ax.plot(range(len(baseline_eq)), baseline_eq, label="All trades", alpha=0.7)

        # Find best VIX threshold for this strategy
        best_sharpe = -999
        best_thresh = None
        for thresh in range(14, 40):
            filtered = strat_df[strat_df["vix_close"] <= thresh]
            if len(filtered) < 10:
                continue
            daily = filtered.groupby("trade_date")["pnl_dollar"].sum()
            s = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0
            if s > best_sharpe:
                best_sharpe = s
                best_thresh = thresh

        if best_thresh:
            filtered = strat_df[strat_df["vix_close"] <= best_thresh]
            filtered_eq = filtered["pnl_dollar"].cumsum()
            ax.plot(range(len(filtered_eq)), filtered_eq,
                    label=f"VIX≤{best_thresh} (Sharpe {best_sharpe:.2f})", alpha=0.7)

        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title(f"{strat}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Equity Curves: All Trades vs VIX-Filtered", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "vix_equity_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/vix_equity_curves.png")

    # 5. VIX level over time with trade outcomes overlaid
    fig, ax1 = plt.subplots(figsize=(14, 5))
    vix_daily = merged.drop_duplicates("trade_date").sort_values("trade_date")
    ax1.fill_between(vix_daily["trade_date"], vix_daily["vix_close"], alpha=0.2, color="purple")
    ax1.plot(vix_daily["trade_date"], vix_daily["vix_close"], color="purple", linewidth=0.8, label="VIX")
    ax1.set_ylabel("VIX Close", color="purple")
    ax1.tick_params(axis="y", labelcolor="purple")

    ax2 = ax1.twinx()
    daily_pnl = merged.groupby("trade_date")["pnl_dollar"].sum().reset_index()
    colors = ["green" if p > 0 else "red" for p in daily_pnl["pnl_dollar"]]
    ax2.bar(daily_pnl["trade_date"], daily_pnl["pnl_dollar"], alpha=0.5, color=colors, width=1.5)
    ax2.set_ylabel("Daily P&L ($)")
    ax2.axhline(y=0, color="gray", linewidth=0.5)

    fig.suptitle("VIX Level vs Daily Portfolio P&L", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "vix_vs_daily_pnl.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/vix_vs_daily_pnl.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VIX Correlation Analysis")
    parser.add_argument("--plot", action="store_true", help="Generate and save charts")
    args = parser.parse_args()

    print("=" * 80)
    print("  VIX CORRELATION ANALYSIS — Phase 1")
    print("=" * 80)

    # Load trades
    print("\nLoading trade archives...")
    trades = load_trades()

    # Load VIX
    start = trades["trade_date"].min().strftime("%Y-%m-%d")
    end = trades["trade_date"].max().strftime("%Y-%m-%d")
    vix = load_vix(start, end)

    # Merge on trade date
    trades["_merge_date"] = trades["trade_date"].dt.normalize()
    vix["_merge_date"] = vix["date"].dt.normalize()
    merged = trades.merge(vix, on="_merge_date", how="left").drop(columns=["_merge_date", "date"])

    # Drop rows where VIX data is missing (weekends, holidays shouldn't exist in trades)
    missing_vix = merged["vix_close"].isna().sum()
    if missing_vix > 0:
        print(f"\n  WARNING: {missing_vix} trades have no VIX data (dropped)")
        merged = merged.dropna(subset=["vix_close"])

    print(f"\n  Merged dataset: {len(merged)} trades with VIX data")
    print(f"  Date range: {merged['trade_date'].min().date()} to {merged['trade_date'].max().date()}")
    print(f"  VIX range: {merged['vix_close'].min():.1f} to {merged['vix_close'].max():.1f}")

    # --- Run all analyses ---

    # 1. VIX quintile analysis (all strategies)
    analyze_vix_quintiles(merged)

    # 2. Per-strategy VIX quintile breakdown
    analyze_vix_quintiles_per_strategy(merged)

    # 3. VIX day-over-day change analysis
    analyze_vix_change(merged)

    # 4. VIX intraday range analysis
    analyze_vix_intraday_range(merged)

    # 5. SL clustering analysis
    analyze_sl_clustering(merged)

    # 6. VIX threshold sweep (all strategies combined)
    sweep_vix_threshold(merged)

    # 7. Per-strategy threshold sweep
    sweep_vix_threshold_per_strategy(merged)

    # 8. Correlation coefficient
    print(f"\n{'='*80}")
    print(f"  Correlation Summary")
    print(f"{'='*80}")
    for strat in sorted(merged["strategy"].unique()):
        subset = merged[merged["strategy"] == strat]
        corr_level = subset["vix_close"].corr(subset["pnl_dollar"])
        corr_change = subset["vix_change"].corr(subset["pnl_dollar"])
        corr_range = (subset["vix_high"] - subset["vix_low"]).corr(subset["pnl_dollar"])
        print(f"  {strat}:")
        print(f"    VIX level  <-> P&L:  r = {corr_level:+.4f}")
        print(f"    VIX change <-> P&L:  r = {corr_change:+.4f}")
        print(f"    VIX range  <-> P&L:  r = {corr_range:+.4f}")

    # Plots
    if args.plot:
        print(f"\nGenerating charts...")
        generate_plots(merged, OUTPUT_DIR)

    # Save merged dataset for further analysis
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_DIR / "trades_with_vix.csv", index=False)
    print(f"\n  Saved merged dataset: {OUTPUT_DIR}/trades_with_vix.csv")

    print(f"\n{'='*80}")
    print(f"  DONE. Review results above to decide if VIX filtering is worth pursuing.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
