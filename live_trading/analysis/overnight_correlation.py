#!/usr/bin/env python3
"""
Overnight snapshot correlation analysis.

Reads logs/overnight_snapshots.csv and correlates morning metrics with
daily trading performance. Run manually after 20+ days of data.

Usage:
    cd live_trading
    python -m analysis.overnight_correlation
    python -m analysis.overnight_correlation --csv logs/overnight_snapshots.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = "logs/overnight_snapshots.csv"

# Morning metrics to correlate against outcomes
NUMERIC_MORNING = [
    "coverage_hours",
    "overnight_range_pts", "overnight_direction_pts", "position_in_range",
    "gap_from_prev_close",
    "support_tests", "resistance_tests", "vpoc_distance_from_close",
    "eu_direction_pts",
    "asia_range_pts", "eu_range_pts", "eu_volume_ratio",
    "overnight_high_hour_et", "overnight_low_hour_et",
    "avg_bar_range_pts", "max_bar_range_pts", "total_overnight_volume",
    "wide_bar_count", "volume_spike_count",
    "overnight_swings", "largest_move_pts", "up_bar_pct", "time_above_mid_pct",
    "mnq_sm_value", "mnq_sm_flips", "mes_sm_value", "mes_sm_flips",
    "vix_close",
    "prev_day_pnl", "prev_day_sl_count",
]

OUTCOME_COLS = ["total_pnl", "win_rate", "sl_count", "trade_count"]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the snapshot CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Filter to rows with valid coverage and evening data
    df["coverage_hours"] = pd.to_numeric(df["coverage_hours"], errors="coerce")
    df["total_pnl"] = pd.to_numeric(df["total_pnl"], errors="coerce")

    valid = df[(df["coverage_hours"] >= 10) & df["total_pnl"].notna()].copy()
    print(f"Valid rows (coverage ≥10h + evening data): {len(valid)}")

    if len(valid) < 5:
        print("Not enough data for meaningful analysis. Need at least 5 valid days.")
        sys.exit(0)

    # Convert numeric columns
    for col in NUMERIC_MORNING + OUTCOME_COLS:
        if col in valid.columns:
            valid[col] = pd.to_numeric(valid[col], errors="coerce")

    return valid


def correlation_matrix(df: pd.DataFrame) -> None:
    """Print correlation of each morning metric vs outcomes."""
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX: Morning Metrics vs Trading Outcomes")
    print("=" * 70)

    available = [c for c in NUMERIC_MORNING if c in df.columns and df[c].notna().sum() >= 5]
    outcomes = [c for c in OUTCOME_COLS if c in df.columns and df[c].notna().sum() >= 5]

    if not available or not outcomes:
        print("Not enough numeric data for correlations.")
        return

    results = []
    for metric in available:
        row = {"metric": metric}
        for outcome in outcomes:
            valid = df[[metric, outcome]].dropna()
            if len(valid) >= 5:
                r = valid[metric].corr(valid[outcome])
                row[outcome] = r
            else:
                row[outcome] = np.nan
        results.append(row)

    corr_df = pd.DataFrame(results).set_index("metric")
    # Sort by absolute correlation with total_pnl
    if "total_pnl" in corr_df.columns:
        corr_df["abs_corr"] = corr_df["total_pnl"].abs()
        corr_df = corr_df.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])

    pd.set_option("display.float_format", "{:+.3f}".format)
    pd.set_option("display.max_rows", 50)
    print(corr_df.to_string())


def conditional_stats(df: pd.DataFrame) -> None:
    """Conditional P&L statistics for various morning conditions."""
    print("\n" + "=" * 70)
    print("CONDITIONAL STATISTICS")
    print("=" * 70)

    analyses = []

    # Position in range
    if "position_in_range" in df.columns:
        low = df[df["position_in_range"] < 0.33]
        mid = df[(df["position_in_range"] >= 0.33) & (df["position_in_range"] <= 0.67)]
        high = df[df["position_in_range"] > 0.67]
        analyses.append(("Position in range", [
            (f"Low (<0.33, n={len(low)})", low["total_pnl"]),
            (f"Mid (0.33-0.67, n={len(mid)})", mid["total_pnl"]),
            (f"High (>0.67, n={len(high)})", high["total_pnl"]),
        ]))

    # Support/resistance tests
    if "support_tests" in df.columns:
        few = df[df["support_tests"] < 3]
        many = df[df["support_tests"] >= 3]
        analyses.append(("Support tests", [
            (f"Few (<3, n={len(few)})", few["total_pnl"]),
            (f"Many (≥3, n={len(many)})", many["total_pnl"]),
        ]))

    # VPOC above/below price
    if "vpoc_distance_from_close" in df.columns:
        vpoc_col = pd.to_numeric(df["vpoc_distance_from_close"], errors="coerce")
        above = df[vpoc_col > 0]
        below = df[vpoc_col < 0]
        analyses.append(("VPOC vs price", [
            (f"VPOC above (n={len(above)})", above["total_pnl"]),
            (f"VPOC below (n={len(below)})", below["total_pnl"]),
        ]))

    # SM state (MNQ)
    if "mnq_sm_value" in df.columns:
        sm_col = pd.to_numeric(df["mnq_sm_value"], errors="coerce")
        bullish = df[sm_col > 0]
        bearish = df[sm_col < 0]
        analyses.append(("MNQ SM at open", [
            (f"Bullish (>0, n={len(bullish)})", bullish["total_pnl"]),
            (f"Bearish (<0, n={len(bearish)})", bearish["total_pnl"]),
        ]))

    # Dominant session
    if "dominant_session" in df.columns:
        asia_dom = df[df["dominant_session"] == "asia"]
        eu_dom = df[df["dominant_session"] == "eu"]
        analyses.append(("Dominant session", [
            (f"Asia (n={len(asia_dom)})", asia_dom["total_pnl"]),
            (f"EU (n={len(eu_dom)})", eu_dom["total_pnl"]),
        ]))

    # Gap filled vs unfilled
    if "gap_filled" in df.columns:
        filled = df[df["gap_filled"].astype(str).str.lower() == "true"]
        unfilled = df[df["gap_filled"].astype(str).str.lower() == "false"]
        analyses.append(("Gap filled", [
            (f"Filled (n={len(filled)})", filled["total_pnl"]),
            (f"Unfilled (n={len(unfilled)})", unfilled["total_pnl"]),
        ]))

    # Overnight character: choppy vs trending
    if "overnight_swings" in df.columns:
        choppy = df[df["overnight_swings"] > 8]
        trending = df[df["overnight_swings"] < 4]
        moderate = df[(df["overnight_swings"] >= 4) & (df["overnight_swings"] <= 8)]
        analyses.append(("Overnight character", [
            (f"Trending (<4 swings, n={len(trending)})", trending["total_pnl"]),
            (f"Moderate (4-8, n={len(moderate)})", moderate["total_pnl"]),
            (f"Choppy (>8 swings, n={len(choppy)})", choppy["total_pnl"]),
        ]))

    # Time of overnight high (Asia vs EU)
    if "overnight_high_hour_et" in df.columns:
        h_col = pd.to_numeric(df["overnight_high_hour_et"], errors="coerce")
        asia_high = df[h_col < 3]
        eu_high = df[(h_col >= 3) & (h_col < 10)]
        analyses.append(("Overnight high set in", [
            (f"Asia (<3 AM, n={len(asia_high)})", asia_high["total_pnl"]),
            (f"EU (3-9 AM, n={len(eu_high)})", eu_high["total_pnl"]),
        ]))

    # Previous day momentum
    if "prev_day_pnl" in df.columns:
        prev = pd.to_numeric(df["prev_day_pnl"], errors="coerce")
        after_loss = df[prev < 0]
        after_win = df[prev > 0]
        analyses.append(("After previous day", [
            (f"After loss (n={len(after_loss)})", after_loss["total_pnl"]),
            (f"After win (n={len(after_win)})", after_win["total_pnl"]),
        ]))

    # Day of week
    if "day_of_week" in df.columns:
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
            day_df = df[df["day_of_week"] == day]
            if len(day_df) > 0:
                analyses.append((f"Day: {day}", [
                    (f"n={len(day_df)}", day_df["total_pnl"]),
                ]))

    # Overnight range bins
    if "overnight_range_pts" in df.columns:
        q33 = df["overnight_range_pts"].quantile(0.33)
        q67 = df["overnight_range_pts"].quantile(0.67)
        tight = df[df["overnight_range_pts"] <= q33]
        medium = df[(df["overnight_range_pts"] > q33) & (df["overnight_range_pts"] <= q67)]
        wide = df[df["overnight_range_pts"] > q67]
        analyses.append((f"Overnight range (tight≤{q33:.0f}, wide>{q67:.0f})", [
            (f"Tight (n={len(tight)})", tight["total_pnl"]),
            (f"Medium (n={len(medium)})", medium["total_pnl"]),
            (f"Wide (n={len(wide)})", wide["total_pnl"]),
        ]))

    # Print all analyses
    for title, groups in analyses:
        print(f"\n  {title}:")
        for label, pnl_series in groups:
            pnl = pnl_series.dropna()
            if len(pnl) > 0:
                print(f"    {label}: avg P&L=${pnl.mean():+.2f}, "
                      f"median=${pnl.median():+.2f}, "
                      f"total=${pnl.sum():+.2f}")
            else:
                print(f"    {label}: no data")


def per_strategy_breakdown(df: pd.DataFrame) -> None:
    """Per-strategy P&L under different morning conditions."""
    print("\n" + "=" * 70)
    print("PER-STRATEGY BREAKDOWN: SM State at Open")
    print("=" * 70)

    strategies = [
        ("v15_pnl", "vScalpA (V15)"),
        ("vscalpb_pnl", "vScalpB"),
        ("mesv2_pnl", "MES v2"),
    ]

    if "mnq_sm_value" not in df.columns:
        print("No SM data available.")
        return

    sm = pd.to_numeric(df["mnq_sm_value"], errors="coerce")
    bullish = df[sm > 0]
    bearish = df[sm < 0]

    for col, name in strategies:
        if col not in df.columns:
            continue
        pnl_col = pd.to_numeric(df[col], errors="coerce")
        bull_pnl = pnl_col[sm > 0].dropna()
        bear_pnl = pnl_col[sm < 0].dropna()
        print(f"\n  {name}:")
        if len(bull_pnl) > 0:
            print(f"    SM bullish (n={len(bull_pnl)}): avg=${bull_pnl.mean():+.2f}")
        if len(bear_pnl) > 0:
            print(f"    SM bearish (n={len(bear_pnl)}): avg=${bear_pnl.mean():+.2f}")


def best_worst_profiles(df: pd.DataFrame) -> None:
    """Feature profiles of best and worst trading days."""
    print("\n" + "=" * 70)
    print("BEST & WORST DAY PROFILES")
    print("=" * 70)

    if len(df) < 5:
        print("Need more data.")
        return

    n = max(3, len(df) // 5)  # Top/bottom 20%, min 3
    sorted_df = df.sort_values("total_pnl")
    worst = sorted_df.head(n)
    best = sorted_df.tail(n)

    profile_cols = [
        "overnight_range_pts", "position_in_range", "overnight_swings",
        "avg_bar_range_pts", "eu_volume_ratio", "mnq_sm_value",
        "vix_close", "gap_from_prev_close", "dominant_session",
    ]

    print(f"\n  Best {n} days (avg P&L=${best['total_pnl'].mean():+.2f}):")
    for col in profile_cols:
        if col in df.columns:
            vals = pd.to_numeric(best[col], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"    {col}: {vals.mean():.2f}")
            elif col == "dominant_session":
                counts = best[col].value_counts()
                print(f"    {col}: {dict(counts)}")

    print(f"\n  Worst {n} days (avg P&L=${worst['total_pnl'].mean():+.2f}):")
    for col in profile_cols:
        if col in df.columns:
            vals = pd.to_numeric(worst[col], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"    {col}: {vals.mean():.2f}")
            elif col == "dominant_session":
                counts = worst[col].value_counts()
                print(f"    {col}: {dict(counts)}")


def main():
    parser = argparse.ArgumentParser(description="Overnight snapshot correlation analysis")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to snapshot CSV")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    df = load_data(args.csv)

    correlation_matrix(df)
    conditional_stats(df)
    per_strategy_breakdown(df)
    best_worst_profiles(df)

    print("\n" + "=" * 70)
    print(f"Analysis complete. {len(df)} valid trading days analyzed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
