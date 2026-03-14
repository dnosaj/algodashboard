#!/usr/bin/env python3
"""
VIX Relative Position Analysis — Phase 1b
============================================
Tests from Dylan O'Neal's VIX framework:
  1. VIX relative to its own PDH/PDL as regime classifier
  2. The 1% Rule (ES and VIX both up/down 1%+)
  4. VIX direction (intraday trend) vs VIX level

Usage:
    python3 vix_relative_analysis.py
    python3 vix_relative_analysis.py --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "vix_analysis"

TRADE_FILES = {
    "MNQ_V15": "MNQ_V15_FULL_2025-02-17_to_2026-02-19_20260223_002951.csv",
    "MNQ_VSCALPB": "MNQ_VSCALPB_FULL_2025-02-17_to_2026-02-19_20260223_002951.csv",
    "MES_V2": "MES_V2_FULL_2025-02-17_to_2026-02-19_20260223_002952.csv",
}


def load_trades() -> pd.DataFrame:
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
        sys.exit("ERROR: No trade files found!")
    trades = pd.concat(frames, ignore_index=True)
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["win"] = trades["pnl_dollar"] > 0
    return trades


def load_market_data(start: str, end: str) -> pd.DataFrame:
    """Download daily VIX + ES data from Yahoo Finance."""
    import yfinance as yf

    print(f"\n  Downloading VIX + ES data ({start} to {end})...")
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    es = yf.download("ES=F", start=start, end=end, progress=False)

    if vix.empty or es.empty:
        sys.exit("ERROR: No market data returned!")

    # Flatten MultiIndex columns if present
    for df in [vix, es]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    vix = vix.reset_index().rename(columns={
        "Date": "date", "Open": "vix_open", "High": "vix_high",
        "Low": "vix_low", "Close": "vix_close"
    })[["date", "vix_open", "vix_high", "vix_low", "vix_close"]]

    es = es.reset_index().rename(columns={
        "Date": "date", "Open": "es_open", "High": "es_high",
        "Low": "es_low", "Close": "es_close"
    })[["date", "es_open", "es_high", "es_low", "es_close"]]

    vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
    es["date"] = pd.to_datetime(es["date"]).dt.tz_localize(None)

    # Compute previous day references
    vix["vix_pdh"] = vix["vix_high"].shift(1)  # previous day high
    vix["vix_pdl"] = vix["vix_low"].shift(1)   # previous day low
    vix["vix_prev_close"] = vix["vix_close"].shift(1)
    vix["vix_pct_change"] = (vix["vix_close"] / vix["vix_prev_close"] - 1) * 100

    es["es_prev_close"] = es["es_close"].shift(1)
    es["es_pct_change"] = (es["es_close"] / es["es_prev_close"] - 1) * 100

    merged = vix.merge(es, on="date", how="inner")
    print(f"  Got {len(merged)} trading days with VIX + ES data")
    return merged


def compute_stats(df: pd.DataFrame, label: str = "") -> dict:
    if len(df) == 0:
        return {"label": label, "n": 0, "wr": 0, "pf": 0, "total_pnl": 0,
                "avg_pnl": 0, "sharpe": 0}
    wins = df["pnl_dollar"] > 0
    gross_profit = df.loc[wins, "pnl_dollar"].sum()
    gross_loss = abs(df.loc[~wins, "pnl_dollar"].sum())
    total_pnl = df["pnl_dollar"].sum()
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
        "sl_rate": (df["exit_reason"] == "SL").mean() * 100 if "exit_reason" in df else 0,
    }


def print_table(rows: list[dict], title: str):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Bucket':<25} {'N':>5} {'WR%':>7} {'PF':>7} {'Total$':>9} {'Avg$':>8} {'Sharpe':>7} {'SL%':>6}")
    print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*7} {'-'*6}")
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['label']:<25} {r['n']:>5}   {'---':>53}")
            continue
        pf = f"{r['pf']:.3f}" if r['pf'] < 100 else "inf"
        print(f"  {r['label']:<25} {r['n']:>5} {r['wr']:>6.1f}% {pf:>7} "
              f"{r['total_pnl']:>+9.0f} {r['avg_pnl']:>+7.2f} {r['sharpe']:>7.2f} {r['sl_rate']:>5.1f}%")


# =====================================================================
# TEST 1: VIX Relative to PDH/PDL
# =====================================================================

def test_vix_pdh_pdl(merged: pd.DataFrame):
    """Classify days by VIX position relative to its previous day's high/low."""

    # Regime classification
    def classify(row):
        if pd.isna(row["vix_pdh"]) or pd.isna(row["vix_pdl"]):
            return "unknown"
        if row["vix_close"] > row["vix_pdh"]:
            return "VIX > PDH (bearish)"
        elif row["vix_close"] < row["vix_pdl"]:
            return "VIX < PDL (bullish)"
        else:
            return "VIX neutral"

    merged["vix_regime"] = merged.apply(classify, axis=1)

    # Also check intraday: did VIX breach PDH at any point?
    def classify_intraday(row):
        if pd.isna(row["vix_pdh"]) or pd.isna(row["vix_pdl"]):
            return "unknown"
        breached_pdh = row["vix_high"] >= row["vix_pdh"]
        breached_pdl = row["vix_low"] <= row["vix_pdl"]
        if breached_pdh and not breached_pdl:
            return "VIX touched PDH (bearish)"
        elif breached_pdl and not breached_pdh:
            return "VIX touched PDL (bullish)"
        elif breached_pdh and breached_pdl:
            return "VIX wide range (both)"
        else:
            return "VIX inside range"

    merged["vix_intraday"] = merged.apply(classify_intraday, axis=1)

    # Print regime distribution
    regime_counts = merged.drop_duplicates("trade_date").groupby("vix_regime").size()
    print(f"\n  Day distribution by VIX close vs PDH/PDL:")
    for regime, count in regime_counts.items():
        if regime != "unknown":
            print(f"    {regime}: {count} days")

    # All strategies combined
    rows = []
    for regime in ["VIX > PDH (bearish)", "VIX neutral", "VIX < PDL (bullish)"]:
        subset = merged[merged["vix_regime"] == regime]
        rows.append(compute_stats(subset, regime))
    rows.append(compute_stats(merged[merged["vix_regime"] != "unknown"], "ALL"))
    print_table(rows, "TEST 1: Trade Stats by VIX Close vs Previous Day High/Low")

    # Per strategy
    for strat in sorted(merged["strategy"].unique()):
        strat_df = merged[merged["strategy"] == strat]
        rows = []
        for regime in ["VIX > PDH (bearish)", "VIX neutral", "VIX < PDL (bullish)"]:
            subset = strat_df[strat_df["vix_regime"] == regime]
            rows.append(compute_stats(subset, regime))
        rows.append(compute_stats(strat_df[strat_df["vix_regime"] != "unknown"], "ALL"))
        print_table(rows, f"TEST 1: VIX PDH/PDL — {strat}")

    # Intraday breach analysis
    rows = []
    for regime in ["VIX touched PDH (bearish)", "VIX inside range",
                    "VIX touched PDL (bullish)", "VIX wide range (both)"]:
        subset = merged[merged["vix_intraday"] == regime]
        rows.append(compute_stats(subset, regime))
    rows.append(compute_stats(merged[merged["vix_intraday"] != "unknown"], "ALL"))
    print_table(rows, "TEST 1b: Trade Stats by VIX Intraday Breach of PDH/PDL")


# =====================================================================
# TEST 2: The 1% Rule
# =====================================================================

def test_one_percent_rule(merged: pd.DataFrame):
    """Flag days where ES and VIX are both up/down 1%+."""

    def classify_1pct(row):
        if pd.isna(row["es_pct_change"]) or pd.isna(row["vix_pct_change"]):
            return "unknown"
        es_up = row["es_pct_change"] >= 1.0
        es_down = row["es_pct_change"] <= -1.0
        vix_up = row["vix_pct_change"] >= 1.0
        vix_down = row["vix_pct_change"] <= -1.0

        if es_up and vix_up:
            return "BOTH UP 1%+ (bearish)"
        elif es_down and vix_down:
            return "BOTH DOWN 1%+ (bullish)"
        elif es_up and vix_down:
            return "Normal bull (ES+, VIX-)"
        elif es_down and vix_up:
            return "Normal bear (ES-, VIX+)"
        else:
            return "Small moves (<1%)"

    merged["pct_regime"] = merged.apply(classify_1pct, axis=1)

    # Distribution
    regime_counts = merged.drop_duplicates("trade_date").groupby("pct_regime").size()
    print(f"\n  Day distribution by 1% Rule:")
    for regime, count in sorted(regime_counts.items()):
        if regime != "unknown":
            print(f"    {regime}: {count} days")

    # All strategies
    rows = []
    for regime in ["BOTH UP 1%+ (bearish)", "BOTH DOWN 1%+ (bullish)",
                    "Normal bull (ES+, VIX-)", "Normal bear (ES-, VIX+)",
                    "Small moves (<1%)"]:
        subset = merged[merged["pct_regime"] == regime]
        rows.append(compute_stats(subset, regime))
    rows.append(compute_stats(merged[merged["pct_regime"] != "unknown"], "ALL"))
    print_table(rows, "TEST 2: The 1% Rule — ES and VIX same-direction days")

    # Per strategy
    for strat in sorted(merged["strategy"].unique()):
        strat_df = merged[merged["strategy"] == strat]
        rows = []
        for regime in ["BOTH UP 1%+ (bearish)", "BOTH DOWN 1%+ (bullish)",
                        "Normal bull (ES+, VIX-)", "Normal bear (ES-, VIX+)",
                        "Small moves (<1%)"]:
            subset = strat_df[strat_df["pct_regime"] == regime]
            rows.append(compute_stats(subset, regime))
        rows.append(compute_stats(strat_df[strat_df["pct_regime"] != "unknown"], "ALL"))
        print_table(rows, f"TEST 2: 1% Rule — {strat}")

    # Deep dive: what happens the NEXT day after a 1% rule trigger?
    print(f"\n{'='*90}")
    print(f"  TEST 2b: Next-day effect after 1% Rule triggers")
    print(f"{'='*90}")

    daily_regimes = merged.drop_duplicates("trade_date")[["trade_date", "pct_regime"]].sort_values("trade_date")
    daily_regimes["next_date"] = daily_regimes["trade_date"].shift(-1)

    for trigger in ["BOTH UP 1%+ (bearish)", "BOTH DOWN 1%+ (bullish)"]:
        trigger_dates = daily_regimes[daily_regimes["pct_regime"] == trigger]["next_date"].dropna()
        next_day_trades = merged[merged["trade_date"].isin(trigger_dates)]
        if len(next_day_trades) > 0:
            s = compute_stats(next_day_trades, f"Day AFTER {trigger}")
            pf = f"{s['pf']:.3f}" if s['pf'] < 100 else "inf"
            print(f"  {s['label']}")
            print(f"    {s['n']} trades, WR {s['wr']:.1f}%, PF {pf}, P&L {s['total_pnl']:+.0f}, Sharpe {s['sharpe']:.2f}")
        else:
            print(f"  Day AFTER {trigger}: no trades found")


# =====================================================================
# TEST 4: VIX Direction (intraday trend) vs VIX Level
# =====================================================================

def test_vix_direction(merged: pd.DataFrame):
    """Compare VIX direction (trending up/down) vs VIX level as predictor."""

    # VIX intraday direction: close vs open
    merged["vix_intraday_dir"] = np.where(
        merged["vix_close"] > merged["vix_open"], "VIX up intraday",
        np.where(merged["vix_close"] < merged["vix_open"], "VIX down intraday", "VIX flat")
    )

    # VIX day-over-day direction: close vs prev close
    merged["vix_dod_dir"] = np.where(
        merged["vix_close"] > merged["vix_prev_close"], "VIX rising (close>prev)",
        np.where(merged["vix_close"] < merged["vix_prev_close"], "VIX falling (close<prev)", "VIX unchanged")
    )

    # VIX momentum: combine direction + magnitude
    def classify_momentum(row):
        if pd.isna(row["vix_pct_change"]):
            return "unknown"
        pct = row["vix_pct_change"]
        if pct > 5:
            return "VIX surging (>+5%)"
        elif pct > 0:
            return "VIX drifting up (0-5%)"
        elif pct > -5:
            return "VIX drifting down (0-5%)"
        else:
            return "VIX crashing (<-5%)"

    merged["vix_momentum"] = merged.apply(classify_momentum, axis=1)

    # Test A: Intraday direction
    rows = []
    for d in ["VIX up intraday", "VIX flat", "VIX down intraday"]:
        subset = merged[merged["vix_intraday_dir"] == d]
        rows.append(compute_stats(subset, d))
    rows.append(compute_stats(merged, "ALL"))
    print_table(rows, "TEST 4a: VIX Intraday Direction (close vs open)")

    # Test B: Day-over-day direction
    rows = []
    for d in ["VIX rising (close>prev)", "VIX unchanged", "VIX falling (close<prev)"]:
        subset = merged[merged["vix_dod_dir"] == d]
        rows.append(compute_stats(subset, d))
    rows.append(compute_stats(merged, "ALL"))
    print_table(rows, "TEST 4b: VIX Day-over-Day Direction")

    # Test C: VIX momentum buckets
    rows = []
    for m in ["VIX surging (>+5%)", "VIX drifting up (0-5%)",
              "VIX drifting down (0-5%)", "VIX crashing (<-5%)"]:
        subset = merged[merged["vix_momentum"] == m]
        rows.append(compute_stats(subset, m))
    rows.append(compute_stats(merged[merged["vix_momentum"] != "unknown"], "ALL"))
    print_table(rows, "TEST 4c: VIX Momentum Buckets")

    # Per strategy for momentum
    for strat in sorted(merged["strategy"].unique()):
        strat_df = merged[merged["strategy"] == strat]
        rows = []
        for m in ["VIX surging (>+5%)", "VIX drifting up (0-5%)",
                  "VIX drifting down (0-5%)", "VIX crashing (<-5%)"]:
            subset = strat_df[strat_df["vix_momentum"] == m]
            rows.append(compute_stats(subset, m))
        rows.append(compute_stats(strat_df[strat_df["vix_momentum"] != "unknown"], "ALL"))
        print_table(rows, f"TEST 4c: VIX Momentum — {strat}")

    # Test D: Combined signal — VIX direction + VIX level
    # Dylan's insight: VIX RELATIVE position matters more than level
    # Let's test: VIX direction (up/down) COMBINED with VIX level quintile
    merged["vix_q"] = pd.qcut(merged["vix_close"], 3, labels=["Low VIX", "Mid VIX", "High VIX"],
                               duplicates="drop")
    merged["vix_combo"] = merged["vix_intraday_dir"].str.replace(" intraday", "") + " + " + merged["vix_q"].astype(str)

    print(f"\n{'='*90}")
    print(f"  TEST 4d: VIX Direction × VIX Level (combined signal)")
    print(f"{'='*90}")
    print(f"  {'Combo':<30} {'N':>5} {'WR%':>7} {'PF':>7} {'Total$':>9} {'Sharpe':>7} {'SL%':>6}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*6}")

    combos = sorted(merged["vix_combo"].unique())
    for combo in combos:
        subset = merged[merged["vix_combo"] == combo]
        if len(subset) < 5:
            continue
        s = compute_stats(subset, combo)
        pf = f"{s['pf']:.3f}" if s['pf'] < 100 else "inf"
        print(f"  {combo:<30} {s['n']:>5} {s['wr']:>6.1f}% {pf:>7} {s['total_pnl']:>+9.0f} {s['sharpe']:>7.2f} {s['sl_rate']:>5.1f}%")


# =====================================================================
# PLOTS
# =====================================================================

def generate_plots(merged: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. VIX regime bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, strat in enumerate(sorted(merged["strategy"].unique())):
        ax = axes[i]
        strat_df = merged[merged["strategy"] == strat]
        regimes = ["VIX > PDH (bearish)", "VIX neutral", "VIX < PDL (bullish)"]
        stats = [compute_stats(strat_df[strat_df["vix_regime"] == r], r) for r in regimes]
        colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
        bars = ax.bar(range(3), [s["avg_pnl"] for s in stats], color=colors, alpha=0.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["VIX>PDH\n(bearish)", "Neutral", "VIX<PDL\n(bullish)"], fontsize=8)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.set_ylabel("Avg $/trade")
        ax.set_title(f"{strat}")
        for bar, s in zip(bars, stats):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"n={s['n']}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("TEST 1: Avg P&L by VIX Position vs PDH/PDL", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vix_pdh_pdl_bars.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/vix_pdh_pdl_bars.png")

    # 2. 1% Rule
    fig, ax = plt.subplots(figsize=(10, 5))
    regimes = ["BOTH UP 1%+ (bearish)", "Normal bull (ES+, VIX-)",
               "Small moves (<1%)", "Normal bear (ES-, VIX+)", "BOTH DOWN 1%+ (bullish)"]
    stats = [compute_stats(merged[merged["pct_regime"] == r], r) for r in regimes]
    colors = ["#e74c3c", "#2ecc71", "#95a5a6", "#e67e22", "#3498db"]
    bars = ax.bar(range(len(regimes)), [s["avg_pnl"] for s in stats], color=colors, alpha=0.7)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(["Both\nUP 1%+", "Normal\nbull", "Small\nmoves", "Normal\nbear", "Both\nDOWN 1%+"], fontsize=9)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_ylabel("Avg $/trade")
    for bar, s in zip(bars, stats):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"n={s['n']}\nPF {s['pf']:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("TEST 2: The 1% Rule — Avg P&L by ES/VIX Same-Direction Days", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vix_1pct_rule.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/vix_1pct_rule.png")

    # 3. VIX momentum
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, strat in enumerate(sorted(merged["strategy"].unique())):
        ax = axes[i]
        strat_df = merged[merged["strategy"] == strat]
        momenta = ["VIX crashing (<-5%)", "VIX drifting down (0-5%)",
                   "VIX drifting up (0-5%)", "VIX surging (>+5%)"]
        stats = [compute_stats(strat_df[strat_df["vix_momentum"] == m], m) for m in momenta]
        colors = ["#2ecc71", "#a8d5a2", "#f5a6a6", "#e74c3c"]
        bars = ax.bar(range(4), [s["avg_pnl"] for s in stats], color=colors, alpha=0.7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Crash\n<-5%", "Drift\ndown", "Drift\nup", "Surge\n>+5%"], fontsize=8)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.set_ylabel("Avg $/trade")
        ax.set_title(f"{strat}")
        for bar, s in zip(bars, stats):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"n={s['n']}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("TEST 4: Avg P&L by VIX Momentum", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vix_momentum_bars.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/vix_momentum_bars.png")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print("=" * 90)
    print("  VIX RELATIVE POSITION ANALYSIS — Dylan O'Neal Framework")
    print("=" * 90)

    # Load trades
    print("\nLoading trade archives...")
    trades = load_trades()

    # Load VIX + ES daily data
    start = trades["trade_date"].min().strftime("%Y-%m-%d")
    end = trades["trade_date"].max().strftime("%Y-%m-%d")
    market = load_market_data(start, end)

    # Merge
    trades["_merge_date"] = trades["trade_date"].dt.normalize()
    market["_merge_date"] = market["date"].dt.normalize()
    merged = trades.merge(market, on="_merge_date", how="left").drop(columns=["_merge_date", "date"])

    missing = merged["vix_close"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} trades missing market data (dropped)")
        merged = merged.dropna(subset=["vix_close"])

    print(f"  Merged: {len(merged)} trades with VIX + ES data")

    # Run tests
    test_vix_pdh_pdl(merged)
    test_one_percent_rule(merged)
    test_vix_direction(merged)

    if args.plot:
        print(f"\nGenerating charts...")
        generate_plots(merged)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_DIR / "trades_with_vix_relative.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR}/trades_with_vix_relative.csv")

    print(f"\n{'='*90}")
    print(f"  DONE.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
