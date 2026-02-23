"""
News Day Impact Analysis
=========================
Cross-references backtest trade logs against a macro economic calendar
to quantify how much of our losses come from news days.

Usage:
    python3 news_day_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from results.save_results import load_backtest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ===========================================================================
# MACRO ECONOMIC CALENDAR  (Feb 2025 – Feb 2026)
# ===========================================================================
# Dates are the ACTUAL release dates (accounting for gov shutdown delays).
# Impact: "high" = FOMC/NFP/CPI, "medium" = PPI/GDP/PCE/Retail Sales,
#         "low" = ISM/FOMC Minutes

NEWS_CALENDAR = [
    # FEBRUARY 2025
    ("2025-02-07", "NFP", "high"),
    ("2025-02-12", "CPI", "high"),
    ("2025-02-13", "PPI", "medium"),
    ("2025-02-14", "Retail Sales", "medium"),
    ("2025-02-19", "FOMC Minutes", "low"),
    ("2025-02-27", "GDP Q4 2024 2nd", "medium"),
    ("2025-02-28", "PCE", "medium"),
    # MARCH 2025
    ("2025-03-03", "ISM Manufacturing", "low"),
    ("2025-03-05", "ISM Services", "low"),
    ("2025-03-07", "NFP", "high"),
    ("2025-03-12", "CPI", "high"),
    ("2025-03-13", "PPI", "medium"),
    ("2025-03-17", "Retail Sales", "medium"),
    ("2025-03-19", "FOMC Rate Decision", "high"),
    ("2025-03-27", "GDP Q4 2024 3rd", "medium"),
    ("2025-03-28", "PCE", "medium"),
    # APRIL 2025
    ("2025-04-01", "ISM Manufacturing", "low"),
    ("2025-04-03", "ISM Services", "low"),
    ("2025-04-04", "NFP", "high"),
    ("2025-04-09", "FOMC Minutes", "low"),
    ("2025-04-10", "CPI", "high"),
    ("2025-04-11", "PPI", "medium"),
    ("2025-04-16", "Retail Sales", "medium"),
    ("2025-04-30", "GDP Q1 2025 Adv", "medium"),
    ("2025-04-30", "PCE", "medium"),
    # MAY 2025
    ("2025-05-01", "ISM Manufacturing", "low"),
    ("2025-05-02", "NFP", "high"),
    ("2025-05-05", "ISM Services", "low"),
    ("2025-05-07", "FOMC Rate Decision", "high"),
    ("2025-05-13", "CPI", "high"),
    ("2025-05-15", "PPI", "medium"),
    ("2025-05-15", "Retail Sales", "medium"),
    ("2025-05-28", "FOMC Minutes", "low"),
    ("2025-05-29", "GDP Q1 2025 2nd", "medium"),
    ("2025-05-30", "PCE", "medium"),
    # JUNE 2025
    ("2025-06-02", "ISM Manufacturing", "low"),
    ("2025-06-04", "ISM Services", "low"),
    ("2025-06-06", "NFP", "high"),
    ("2025-06-11", "CPI", "high"),
    ("2025-06-12", "PPI", "medium"),
    ("2025-06-17", "Retail Sales", "medium"),
    ("2025-06-18", "FOMC Rate Decision", "high"),
    ("2025-06-26", "GDP Q1 2025 3rd", "medium"),
    ("2025-06-27", "PCE", "medium"),
    # JULY 2025
    ("2025-07-01", "ISM Manufacturing", "low"),
    ("2025-07-03", "NFP", "high"),
    ("2025-07-03", "ISM Services", "low"),
    ("2025-07-09", "FOMC Minutes", "low"),
    ("2025-07-15", "CPI", "high"),
    ("2025-07-16", "PPI", "medium"),
    ("2025-07-17", "Retail Sales", "medium"),
    ("2025-07-30", "GDP Q2 2025 Adv", "medium"),
    ("2025-07-30", "FOMC Rate Decision", "high"),
    ("2025-07-31", "PCE", "medium"),
    # AUGUST 2025
    ("2025-08-01", "ISM Manufacturing", "low"),
    ("2025-08-01", "NFP", "high"),
    ("2025-08-05", "ISM Services", "low"),
    ("2025-08-12", "CPI", "high"),
    ("2025-08-14", "PPI", "medium"),
    ("2025-08-15", "Retail Sales", "medium"),
    ("2025-08-20", "FOMC Minutes", "low"),
    ("2025-08-28", "GDP Q2 2025 2nd", "medium"),
    ("2025-08-29", "PCE", "medium"),
    # SEPTEMBER 2025
    ("2025-09-02", "ISM Manufacturing", "low"),
    ("2025-09-03", "ISM Services", "low"),
    ("2025-09-05", "NFP", "high"),
    ("2025-09-10", "PPI", "medium"),
    ("2025-09-11", "CPI", "high"),
    ("2025-09-16", "Retail Sales", "medium"),
    ("2025-09-17", "FOMC Rate Decision", "high"),
    ("2025-09-25", "GDP Q2 2025 3rd", "medium"),
    ("2025-09-26", "PCE", "medium"),
    # OCTOBER 2025 (gov shutdown disruptions)
    ("2025-10-01", "ISM Manufacturing", "low"),
    ("2025-10-03", "ISM Services", "low"),
    ("2025-10-08", "FOMC Minutes", "low"),
    ("2025-10-24", "CPI (delayed)", "high"),
    # NOVEMBER 2025 (shutdown recovery)
    ("2025-11-19", "FOMC Minutes", "low"),
    ("2025-11-20", "NFP (delayed Sep)", "high"),
    ("2025-11-25", "Retail Sales (delayed Sep)", "medium"),
    # DECEMBER 2025 (catch-up releases)
    ("2025-12-01", "ISM Manufacturing", "low"),
    ("2025-12-03", "ISM Services", "low"),
    ("2025-12-05", "PCE (delayed Sep)", "medium"),
    ("2025-12-10", "FOMC Rate Decision", "high"),
    ("2025-12-16", "NFP (delayed Oct+Nov)", "high"),
    ("2025-12-16", "Retail Sales (delayed Oct)", "medium"),
    ("2025-12-18", "CPI (Nov)", "high"),
    ("2025-12-23", "GDP Q3 2025 Initial", "medium"),
    ("2025-12-23", "PCE (delayed)", "medium"),
    ("2025-12-30", "FOMC Minutes", "low"),
    # JANUARY 2026
    ("2026-01-05", "ISM Manufacturing", "low"),
    ("2026-01-07", "ISM Services", "low"),
    ("2026-01-09", "NFP (Dec)", "high"),
    ("2026-01-13", "CPI (Dec)", "high"),
    ("2026-01-14", "PPI (Nov, inc Oct)", "medium"),
    ("2026-01-22", "GDP Q3 2025 Updated", "medium"),
    ("2026-01-22", "PCE (Oct+Nov)", "medium"),
    ("2026-01-28", "FOMC Rate Decision", "high"),
    ("2026-01-30", "PPI (Dec)", "medium"),
    # FEBRUARY 2026
    ("2026-02-02", "ISM Manufacturing", "low"),
    ("2026-02-04", "ISM Services", "low"),
    ("2026-02-10", "Retail Sales (delayed Dec)", "medium"),
    ("2026-02-11", "CPI (Jan)", "high"),
    ("2026-02-11", "NFP (delayed Jan)", "high"),
    ("2026-02-18", "FOMC Minutes", "low"),
    ("2026-02-19", "Retail Sales (Dec)", "medium"),
    ("2026-02-20", "GDP Q4 2025 Adv", "medium"),
    ("2026-02-20", "PCE (Dec)", "medium"),
]

# Build lookup sets
NEWS_DATES_HIGH = {d for d, _, lvl in NEWS_CALENDAR if lvl == "high"}
NEWS_DATES_MEDIUM = {d for d, _, lvl in NEWS_CALENDAR if lvl == "medium"}
NEWS_DATES_ALL = {d for d, _, _ in NEWS_CALENDAR}

# Add weekly jobless claims (every Thursday)
_thursdays = pd.date_range("2025-02-17", "2026-02-19", freq="W-THU")
JOBLESS_CLAIMS_DATES = {t.strftime("%Y-%m-%d") for t in _thursdays}

# News lookup: date -> list of event names
NEWS_LOOKUP = {}
for d, name, lvl in NEWS_CALENDAR:
    NEWS_LOOKUP.setdefault(d, []).append(f"{name} [{lvl}]")


def classify_news_day(trade_date):
    """Classify a trade date by news impact level."""
    if trade_date in NEWS_DATES_HIGH:
        return "high"
    elif trade_date in NEWS_DATES_MEDIUM:
        return "medium"
    elif trade_date in JOBLESS_CLAIMS_DATES:
        return "claims"  # weekly jobless claims (low impact)
    else:
        return "none"


def analyze_strategy(filepath):
    """Analyze a single strategy's trade log against the news calendar."""
    meta, df = load_backtest(filepath)
    strategy = meta.get("strategy", filepath.stem)
    split = meta.get("split", "?")

    if len(df) == 0:
        return None

    # Classify each trade
    df["news_level"] = df["trade_date"].apply(classify_news_day)
    df["news_events"] = df["trade_date"].apply(
        lambda d: ", ".join(NEWS_LOOKUP.get(d, []))
    )
    df["is_loser"] = df["pnl_dollar"] < 0

    print(f"\n{'='*70}")
    print(f"  {strategy} ({split}) — {len(df)} trades, ${meta.get('total_pnl', 0):+.2f}")
    print(f"{'='*70}")

    # Overall stats
    total_pnl = df["pnl_dollar"].sum()
    n_trades = len(df)
    n_losers = df["is_loser"].sum()

    # --- By news level ---
    levels = ["high", "medium", "claims", "none"]
    level_labels = {
        "high": "HIGH (FOMC/NFP/CPI)",
        "medium": "MEDIUM (PPI/GDP/PCE/Retail)",
        "claims": "WEEKLY CLAIMS (Thu)",
        "none": "NO NEWS",
    }

    print(f"\n  {'Category':<32s} {'Trades':>6s} {'Losers':>7s} {'LoseRate':>9s} "
          f"{'P&L':>10s} {'Avg P&L':>8s} {'% of Loss':>9s}")
    print(f"  {'-'*83}")

    total_losses = df.loc[df["is_loser"], "pnl_dollar"].sum()

    for level in levels:
        mask = df["news_level"] == level
        subset = df[mask]
        if len(subset) == 0:
            continue
        n = len(subset)
        losers = subset["is_loser"].sum()
        pnl = subset["pnl_dollar"].sum()
        avg_pnl = pnl / n
        loss_pnl = subset.loc[subset["is_loser"], "pnl_dollar"].sum()
        pct_of_losses = (loss_pnl / total_losses * 100) if total_losses < 0 else 0

        lr = losers / n * 100
        print(f"  {level_labels[level]:<32s} {n:>6d} {losers:>7d} {lr:>8.1f}% "
              f"${pnl:>+9.2f} ${avg_pnl:>+7.2f} {pct_of_losses:>8.1f}%")

    # Aggregate: any news vs no news
    print(f"\n  --- AGGREGATE ---")
    news_mask = df["news_level"] != "none"
    no_news = df[~news_mask]
    news = df[news_mask]

    for label, subset in [("NEWS DAYS", news), ("NO-NEWS DAYS", no_news)]:
        n = len(subset)
        if n == 0:
            continue
        losers = subset["is_loser"].sum()
        pnl = subset["pnl_dollar"].sum()
        avg_pnl = pnl / n
        lr = losers / n * 100
        print(f"  {label:<20s} {n:>4d} trades, {losers:>3d} losers ({lr:.1f}%), "
              f"P&L ${pnl:>+9.2f}, avg ${avg_pnl:>+6.2f}/trade")

    # --- HIGH impact only: what if we skip? ---
    high_mask = df["news_level"] == "high"
    if high_mask.any():
        high_trades = df[high_mask]
        without_high = df[~high_mask]
        high_pnl = high_trades["pnl_dollar"].sum()
        without_pnl = without_high["pnl_dollar"].sum()
        print(f"\n  WHAT-IF: Skip HIGH-impact days only")
        print(f"    Trades skipped: {high_mask.sum()} ({high_mask.sum()/n_trades*100:.1f}%)")
        print(f"    P&L on HIGH days:     ${high_pnl:>+9.2f}")
        print(f"    P&L WITHOUT high days: ${without_pnl:>+9.2f} "
              f"(delta: ${without_pnl - total_pnl:>+.2f})")

    # --- HIGH + MEDIUM: what if we skip? ---
    hm_mask = df["news_level"].isin(["high", "medium"])
    if hm_mask.any():
        hm_trades = df[hm_mask]
        without_hm = df[~hm_mask]
        hm_pnl = hm_trades["pnl_dollar"].sum()
        without_pnl = without_hm["pnl_dollar"].sum()
        print(f"\n  WHAT-IF: Skip HIGH + MEDIUM days")
        print(f"    Trades skipped: {hm_mask.sum()} ({hm_mask.sum()/n_trades*100:.1f}%)")
        print(f"    P&L on H+M days:      ${hm_pnl:>+9.2f}")
        print(f"    P&L WITHOUT H+M days:  ${without_pnl:>+9.2f} "
              f"(delta: ${without_pnl - total_pnl:>+.2f})")

    # --- Show the worst losing trades on news days ---
    news_losers = df[news_mask & df["is_loser"]].sort_values("pnl_dollar")
    if len(news_losers) > 0:
        print(f"\n  WORST LOSERS ON NEWS DAYS (top 15):")
        print(f"  {'Date':<12s} {'Day':<10s} {'Side':<6s} {'P&L':>9s} {'MAE':>7s} "
              f"{'Exit':>5s} {'News Events'}")
        print(f"  {'-'*90}")
        for _, row in news_losers.head(15).iterrows():
            print(f"  {row['trade_date']:<12s} {row['day_of_week']:<10s} "
                  f"{row['side']:<6s} ${row['pnl_dollar']:>+8.2f} "
                  f"{row['mae_pts']:>6.1f} {row['exit_reason']:>5s} "
                  f"{row['news_events']}")

    # --- Per-event-type breakdown ---
    print(f"\n  P&L BY EVENT TYPE:")
    event_pnl = {}
    for _, row in df.iterrows():
        for entry in NEWS_LOOKUP.get(row["trade_date"], []):
            event_name = entry.split(" [")[0]
            if event_name not in event_pnl:
                event_pnl[event_name] = {"pnl": 0, "trades": 0, "losers": 0}
            event_pnl[event_name]["pnl"] += row["pnl_dollar"]
            event_pnl[event_name]["trades"] += 1
            if row["pnl_dollar"] < 0:
                event_pnl[event_name]["losers"] += 1

    sorted_events = sorted(event_pnl.items(), key=lambda x: x[1]["pnl"])
    print(f"  {'Event':<30s} {'Trades':>6s} {'Losers':>7s} {'P&L':>10s} {'Avg':>8s}")
    print(f"  {'-'*65}")
    for event, stats in sorted_events:
        avg = stats["pnl"] / stats["trades"] if stats["trades"] > 0 else 0
        print(f"  {event:<30s} {stats['trades']:>6d} {stats['losers']:>7d} "
              f"${stats['pnl']:>+9.2f} ${avg:>+7.2f}")

    return {
        "strategy": strategy,
        "split": split,
        "total_pnl": total_pnl,
        "n_trades": n_trades,
        "high_pnl": df.loc[high_mask, "pnl_dollar"].sum() if high_mask.any() else 0,
        "high_trades": high_mask.sum(),
    }


def main():
    print("=" * 70)
    print("  NEWS DAY IMPACT ANALYSIS — Active Portfolio")
    print("=" * 70)

    # Find FULL split results for all 3 strategies
    full_files = sorted(RESULTS_DIR.glob("*_FULL_*.csv"))
    if not full_files:
        print("\nNo FULL results found. Run run_and_save_portfolio.py first.")
        return

    print(f"\nFound {len(full_files)} result files:")
    for f in full_files:
        print(f"  {f.name}")

    # Calendar summary
    print(f"\nNews calendar: {len(NEWS_CALENDAR)} events")
    print(f"  HIGH impact: {len(NEWS_DATES_HIGH)} days "
          f"(FOMC: {sum(1 for _,n,_ in NEWS_CALENDAR if 'FOMC Rate' in n)}, "
          f"NFP: {sum(1 for _,n,_ in NEWS_CALENDAR if 'NFP' in n)}, "
          f"CPI: {sum(1 for _,n,_ in NEWS_CALENDAR if 'CPI' in n)})")
    print(f"  MEDIUM impact: {len(NEWS_DATES_MEDIUM)} days")
    print(f"  Unique news dates: {len(NEWS_DATES_ALL)}")

    results = []
    for f in full_files:
        r = analyze_strategy(f)
        if r:
            results.append(r)

    # --- Portfolio summary ---
    if results:
        print(f"\n{'='*70}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"{'='*70}")
        total_pnl = sum(r["total_pnl"] for r in results)
        total_high_pnl = sum(r["high_pnl"] for r in results)
        total_trades = sum(r["n_trades"] for r in results)
        total_high_trades = sum(r["high_trades"] for r in results)

        print(f"  Total P&L:              ${total_pnl:>+10.2f} ({total_trades} trades)")
        print(f"  P&L on HIGH days:       ${total_high_pnl:>+10.2f} ({total_high_trades} trades)")
        print(f"  P&L WITHOUT high days:  ${total_pnl - total_high_pnl:>+10.2f} "
              f"({total_trades - total_high_trades} trades)")
        delta = (total_pnl - total_high_pnl) - total_pnl
        direction = "BETTER" if delta > 0 else "WORSE"
        print(f"  Impact of skipping:     ${delta:>+10.2f} ({direction})")


if __name__ == "__main__":
    main()
