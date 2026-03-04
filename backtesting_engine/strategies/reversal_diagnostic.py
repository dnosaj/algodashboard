#!/usr/bin/env python3
"""
Immediate Reversal Diagnostic & Filter Analysis
=================================================
Step 0: SL exit speed — how fast do SL hits happen? (1-3 bars vs gradual)
Step 1: Entry-level signals — SM slope, SM flap count, pre-entry momentum,
        volume spike, entry bar candle structure
Step 2: Daily regime (lagged) — Rule of 16, VIX overnight gap, multi-day
        VIX momentum, VIX Q4 band exclusion
Step 3: Interaction effects — do entry-level filters work better on certain days?

Usage:
    python3 reversal_diagnostic.py
    python3 reversal_diagnostic.py --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & imports from backtest engine
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STRATEGIES_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "reversal_analysis"

sys.path.insert(0, str(STRATEGIES_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "live_trading"))
from v10_test_common import load_instrument_1min

TRADE_FILES = {
    "MNQ_V15": "MNQ_V15_FULL_2025-02-17_to_2026-02-19_20260223_002951.csv",
    "MNQ_VSCALPB": "MNQ_VSCALPB_FULL_2025-02-17_to_2026-02-19_20260223_002951.csv",
    "MES_V2": "MES_V2_FULL_2025-02-17_to_2026-02-19_20260223_002952.csv",
}

# SM params per strategy (for SM recompute if needed)
SM_PARAMS = {
    "MNQ_V15": {"index": 10, "flow": 12, "norm": 200, "ema": 100},
    "MNQ_VSCALPB": {"index": 10, "flow": 12, "norm": 200, "ema": 100},
    "MES_V2": {"index": 20, "flow": 12, "norm": 400, "ema": 255},
}

SL_PTS = {"MNQ_V15": 40, "MNQ_VSCALPB": 15, "MES_V2": 35}
TP_PTS = {"MNQ_V15": 5, "MNQ_VSCALPB": 5, "MES_V2": 20}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trades() -> pd.DataFrame:
    """Load all trade archive CSVs."""
    frames = []
    for strategy, filename in TRADE_FILES.items():
        fp = RESULTS_DIR / filename
        if not fp.exists():
            print(f"  WARNING: {fp.name} not found, skipping")
            continue
        df = pd.read_csv(fp, comment="#")
        df["strategy"] = strategy
        frames.append(df)
        print(f"  Loaded {len(df)} trades from {filename}")
    if not frames:
        sys.exit("ERROR: No trade files found!")
    trades = pd.concat(frames, ignore_index=True)
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades["win"] = trades["pnl_dollar"] > 0
    print(f"  Total: {len(trades)} trades")
    return trades


def load_1min_bars() -> dict[str, pd.DataFrame]:
    """Load 1-min bar data for MNQ and MES."""
    bars = {}
    for inst in ["MNQ", "MES"]:
        print(f"  Loading 1-min bars for {inst}...")
        df = load_instrument_1min(inst)
        # SM_Net is already computed in load_instrument_1min
        bars[inst] = df
        print(f"    {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    return bars


def load_vix_data(start: str, end: str) -> pd.DataFrame:
    """Download daily VIX + ES data from Yahoo Finance."""
    import yfinance as yf

    print(f"\n  Downloading VIX + ES data ({start} to {end})...")
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    es = yf.download("ES=F", start=start, end=end, progress=False)

    if vix.empty or es.empty:
        sys.exit("ERROR: No market data returned!")

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

    # Previous day references
    vix["vix_prev_close"] = vix["vix_close"].shift(1)
    vix["vix_prev_high"] = vix["vix_high"].shift(1)
    vix["vix_prev_low"] = vix["vix_low"].shift(1)
    vix["vix_pct_change"] = (vix["vix_close"] / vix["vix_prev_close"] - 1) * 100
    vix["vix_overnight_gap"] = (vix["vix_open"] / vix["vix_prev_close"] - 1) * 100

    es["es_prev_close"] = es["es_close"].shift(1)
    es["es_pct_change"] = (es["es_close"] / es["es_prev_close"] - 1) * 100

    merged = vix.merge(es, on="date", how="inner")
    print(f"  Got {len(merged)} trading days")
    return merged


# ---------------------------------------------------------------------------
# Helper: match trade entry times to 1-min bar indices
# ---------------------------------------------------------------------------

def enrich_trades_with_bar_context(trades: pd.DataFrame,
                                   bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each trade, look up the 1-min bar context at entry.

    Adds columns: sm_at_entry, sm_slope, sm_slope_3bar, sm_zero_cross_count,
    pre_entry_momentum_5, pre_entry_momentum_10, volume_ratio,
    entry_bar_wick_ratio, entry_bar_body_pct
    """
    print("\n  Enriching trades with 1-min bar context...")

    # Pre-index: build a time→index lookup for each instrument
    # Bar data may be tz-naive or tz-aware; normalize to tz-naive for matching
    bar_indices = {}
    for inst, df in bars.items():
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        bar_indices[inst] = {t: i for i, t in enumerate(idx)}
        print(f"    {inst}: indexed {len(bar_indices[inst])} bars "
              f"({idx[0]} to {idx[-1]})")

    # Map strategy to instrument
    strat_to_inst = {
        "MNQ_V15": "MNQ", "MNQ_VSCALPB": "MNQ", "MES_V2": "MES"
    }

    # Pre-extract numpy arrays for speed
    bar_arrays = {}
    for inst, df in bars.items():
        bar_arrays[inst] = {
            "sm": df["SM_Net"].values,
            "opens": df["Open"].values,
            "highs": df["High"].values,
            "lows": df["Low"].values,
            "closes": df["Close"].values,
            "volume": df["Volume"].values if "Volume" in df.columns else None,
            "times": df.index,
        }

    # New columns to populate
    n = len(trades)
    sm_at_entry = np.full(n, np.nan)
    sm_slope = np.full(n, np.nan)
    sm_slope_3bar = np.full(n, np.nan)
    sm_zero_cross_60 = np.full(n, np.nan)
    sm_zero_cross_120 = np.full(n, np.nan)
    sm_bars_since_flip = np.full(n, np.nan)
    pre_momentum_5 = np.full(n, np.nan)
    pre_momentum_10 = np.full(n, np.nan)
    pre_momentum_20 = np.full(n, np.nan)
    volume_ratio = np.full(n, np.nan)
    entry_wick_ratio = np.full(n, np.nan)
    entry_body_pct = np.full(n, np.nan)
    bars_to_sl = np.full(n, np.nan)  # for SL exits: bar-by-bar time to SL

    matched = 0
    unmatched = 0

    for idx, row in trades.iterrows():
        inst = strat_to_inst.get(row["strategy"])
        if inst is None:
            continue

        arr = bar_arrays[inst]
        entry_time = row["entry_time"]

        # Find bar index matching entry_time
        # Normalize to tz-naive for lookup (bar index is tz-naive)
        entry_naive = entry_time.tz_localize(None) if entry_time.tzinfo else entry_time
        entry_min = entry_naive.floor("min")

        if entry_min in bar_indices[inst]:
            bar_idx = bar_indices[inst][entry_min]
        else:
            unmatched += 1
            continue

        matched += 1
        sm = arr["sm"]
        closes = arr["closes"]
        highs = arr["highs"]
        lows = arr["lows"]
        opens = arr["opens"]
        vol = arr["volume"]

        # Signal bar is bar_idx - 1 (entry fills at bar_idx open)
        sig = bar_idx - 1
        if sig < 2:
            continue

        # --- SM at entry ---
        sm_at_entry[idx] = sm[sig]

        # --- SM slope (1-bar) ---
        sm_slope[idx] = sm[sig] - sm[sig - 1]

        # --- SM slope (3-bar average) ---
        if sig >= 3:
            sm_slope_3bar[idx] = (sm[sig] - sm[sig - 3]) / 3

        # --- SM zero-crossing count in last N bars ---
        for lookback, out_arr in [(60, sm_zero_cross_60), (120, sm_zero_cross_120)]:
            start_i = max(1, sig - lookback + 1)
            crossings = 0
            for j in range(start_i, sig + 1):
                if (sm[j] > 0 and sm[j - 1] <= 0) or (sm[j] < 0 and sm[j - 1] >= 0):
                    crossings += 1
            out_arr[idx] = crossings

        # --- Bars since last SM flip ---
        bars_since = 0
        for j in range(sig, 0, -1):
            if (sm[j] > 0 and sm[j - 1] <= 0) or (sm[j] < 0 and sm[j - 1] >= 0):
                break
            bars_since += 1
        sm_bars_since_flip[idx] = bars_since

        # --- Pre-entry momentum (price move in last N bars) ---
        side = row["side"]
        for lookback, out_arr in [(5, pre_momentum_5), (10, pre_momentum_10),
                                   (20, pre_momentum_20)]:
            if sig >= lookback:
                move = closes[sig] - closes[sig - lookback]
                # Directional: positive = move in trade direction
                if side == "short":
                    move = -move
                out_arr[idx] = move

        # --- Volume ratio (entry bar vs 20-bar average) ---
        if vol is not None and sig >= 20:
            avg_vol = np.mean(vol[sig - 20:sig])
            if avg_vol > 0:
                volume_ratio[idx] = vol[sig] / avg_vol

        # --- Entry bar candle structure ---
        o, h, l, c = opens[bar_idx], highs[bar_idx], lows[bar_idx], closes[bar_idx]
        body = abs(c - o)
        bar_range = h - l
        if bar_range > 0:
            entry_body_pct[idx] = body / bar_range

            # Wick in trade direction (rejection wick)
            if side == "long":
                # Upper wick = rejection for longs
                upper_wick = h - max(o, c)
                entry_wick_ratio[idx] = upper_wick / bar_range
            else:
                # Lower wick = rejection for shorts
                lower_wick = min(o, c) - l
                entry_wick_ratio[idx] = lower_wick / bar_range

        # --- For SL exits: count bars until MAE first exceeds SL ---
        if row["exit_reason"] == "SL":
            sl_pts = SL_PTS.get(row["strategy"], 40)
            exit_idx_approx = bar_idx + int(row["bars_held"]) if not pd.isna(row["bars_held"]) else bar_idx + 100
            exit_idx_approx = min(exit_idx_approx, len(closes) - 1)

            first_sl_bar = np.nan
            for j in range(bar_idx, exit_idx_approx + 1):
                if side == "long":
                    adverse = row["entry_price"] - lows[j]
                else:
                    adverse = highs[j] - row["entry_price"]
                if adverse >= sl_pts:
                    first_sl_bar = j - bar_idx
                    break
            bars_to_sl[idx] = first_sl_bar

    print(f"    Matched {matched} trades, {unmatched} unmatched")

    trades["sm_at_entry"] = sm_at_entry
    trades["sm_slope"] = sm_slope
    trades["sm_slope_3bar"] = sm_slope_3bar
    trades["sm_zero_cross_60"] = sm_zero_cross_60
    trades["sm_zero_cross_120"] = sm_zero_cross_120
    trades["sm_bars_since_flip"] = sm_bars_since_flip
    trades["pre_momentum_5"] = pre_momentum_5
    trades["pre_momentum_10"] = pre_momentum_10
    trades["pre_momentum_20"] = pre_momentum_20
    trades["volume_ratio"] = volume_ratio
    trades["entry_wick_ratio"] = entry_wick_ratio
    trades["entry_body_pct"] = entry_body_pct
    trades["bars_to_sl"] = bars_to_sl

    return trades


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame, label: str = "") -> dict:
    if len(df) == 0:
        return {"label": label, "n": 0, "wr": 0, "pf": 0, "total_pnl": 0,
                "avg_pnl": 0, "sharpe": 0, "sl_rate": 0}
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
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    print(f"  {'Bucket':<30} {'N':>5} {'WR%':>7} {'PF':>7} {'Total$':>9} {'Avg$':>8} {'Sharpe':>7} {'SL%':>6}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*7} {'-'*9} {'-'*8} {'-'*7} {'-'*6}")
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['label']:<30} {r['n']:>5}   {'---':>53}")
            continue
        pf = f"{r['pf']:.3f}" if r['pf'] < 100 else "inf"
        print(f"  {r['label']:<30} {r['n']:>5} {r['wr']:>6.1f}% {pf:>7} "
              f"{r['total_pnl']:>+9.0f} {r['avg_pnl']:>+7.2f} {r['sharpe']:>7.2f} {r['sl_rate']:>5.1f}%")


def print_percentile_table(values: pd.Series, title: str):
    print(f"\n  {title}")
    pcts = [0, 10, 25, 50, 75, 90, 100]
    vals = np.nanpercentile(values.dropna(), pcts)
    row = "    "
    for p, v in zip(pcts, vals):
        row += f"P{p}={v:.1f}  "
    print(row)
    print(f"    Mean={values.mean():.1f}  Median={values.median():.1f}  N={values.notna().sum()}")


# =====================================================================
# STEP 0: SL EXIT DIAGNOSTIC
# =====================================================================

def step0_sl_diagnostic(trades: pd.DataFrame, do_plot: bool = False):
    """How fast do SL exits happen? Are they immediate (1-3 bars) or gradual?"""
    print("\n" + "=" * 95)
    print("  STEP 0: SL EXIT SPEED DIAGNOSTIC")
    print("=" * 95)

    sl_trades = trades[trades["exit_reason"] == "SL"].copy()
    non_sl_trades = trades[trades["exit_reason"] != "SL"].copy()

    print(f"\n  Total trades: {len(trades)}")
    print(f"  SL exits: {len(sl_trades)} ({len(sl_trades)/len(trades)*100:.1f}%)")
    print(f"  Non-SL exits: {len(non_sl_trades)} ({len(non_sl_trades)/len(trades)*100:.1f}%)")

    # Bars held distribution for SL vs non-SL
    for strat in sorted(trades["strategy"].unique()):
        strat_sl = sl_trades[sl_trades["strategy"] == strat]
        strat_wins = trades[(trades["strategy"] == strat) & (trades["exit_reason"] == "TP")]
        print(f"\n  --- {strat} ---")
        print(f"  SL exits: {len(strat_sl)}, SL pts: {SL_PTS.get(strat)}")

        if len(strat_sl) > 0:
            print_percentile_table(strat_sl["bars_held"], "SL exit bars_held distribution:")

            # Bars to SL (first bar where MAE >= SL)
            bars_to = strat_sl["bars_to_sl"].dropna()
            if len(bars_to) > 0:
                print_percentile_table(bars_to, "Bars until price first reaches SL level:")
                immediate = (bars_to <= 3).sum()
                gradual = (bars_to > 10).sum()
                print(f"    Immediate (<=3 bars): {immediate} ({immediate/len(bars_to)*100:.1f}%)")
                print(f"    Medium (4-10 bars): {len(bars_to)-immediate-gradual} "
                      f"({(len(bars_to)-immediate-gradual)/len(bars_to)*100:.1f}%)")
                print(f"    Gradual (>10 bars): {gradual} ({gradual/len(bars_to)*100:.1f}%)")

        if len(strat_wins) > 0:
            print_percentile_table(strat_wins["bars_held"], "TP exit bars_held distribution (for comparison):")

    # MAE distribution for winners vs losers
    print(f"\n  --- MAE Distribution (all strategies) ---")
    for strat in sorted(trades["strategy"].unique()):
        strat_df = trades[trades["strategy"] == strat]
        w = strat_df[strat_df["win"]]
        l = strat_df[~strat_df["win"]]
        print(f"\n  {strat}:")
        if len(w) > 0:
            print_percentile_table(w["mae_pts"], "  Winners MAE (pts):")
        if len(l) > 0:
            print_percentile_table(l["mae_pts"], "  Losers MAE (pts):")

    if do_plot:
        _plot_step0(trades, sl_trades)


def _plot_step0(trades, sl_trades):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("STEP 0: SL Exit Speed Diagnostic", fontsize=14, fontweight="bold")

    strategies = sorted(trades["strategy"].unique())
    for i, strat in enumerate(strategies):
        ax = axes[0, i]
        strat_sl = sl_trades[sl_trades["strategy"] == strat]
        bars_to = strat_sl["bars_to_sl"].dropna()
        if len(bars_to) > 0:
            bins = np.arange(0, min(bars_to.max() + 2, 50), 1)
            ax.hist(bars_to, bins=bins, color="crimson", alpha=0.7, edgecolor="black")
            ax.axvline(3, color="orange", ls="--", lw=2, label="3-bar cutoff")
            ax.axvline(bars_to.median(), color="blue", ls="--", lw=2, label=f"Median={bars_to.median():.0f}")
        ax.set_title(f"{strat} — Bars to SL Hit")
        ax.set_xlabel("Bars from entry to SL level reached")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for i, strat in enumerate(strategies):
        ax = axes[1, i]
        strat_df = trades[trades["strategy"] == strat]
        w = strat_df[strat_df["win"]]["mae_pts"]
        l = strat_df[~strat_df["win"]]["mae_pts"]
        ax.hist(w, bins=30, alpha=0.6, color="green", label=f"Winners (N={len(w)})", edgecolor="black")
        ax.hist(l, bins=30, alpha=0.6, color="red", label=f"Losers (N={len(l)})", edgecolor="black")
        ax.set_title(f"{strat} — MAE Distribution")
        ax.set_xlabel("Max Adverse Excursion (pts)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step0_sl_diagnostic.png", dpi=150)
    print(f"\n  Saved step0_sl_diagnostic.png")
    plt.close()


# =====================================================================
# STEP 1: ENTRY-LEVEL SIGNALS
# =====================================================================

def step1_entry_signals(trades: pd.DataFrame, do_plot: bool = False):
    """Test entry-level signals that might predict immediate reversals."""
    print("\n" + "=" * 95)
    print("  STEP 1: ENTRY-LEVEL SIGNALS")
    print("=" * 95)

    valid = trades[trades["sm_at_entry"].notna()].copy()
    print(f"\n  Trades with bar context: {len(valid)} / {len(trades)}")

    # ---- 1A: SM Slope at Entry ----
    print("\n  === 1A: SM Slope at Entry (1-bar) ===")
    print("  Hypothesis: SM rolling over at entry (slope negative while SM positive) = reversal risk")

    # For each trade, check if SM is moving WITH the trade or AGAINST it
    # Long: SM positive, slope should be positive (SM strengthening)
    # Short: SM negative, slope should be negative (SM strengthening)
    valid["sm_aligned_slope"] = np.where(
        valid["side"] == "long", valid["sm_slope"], -valid["sm_slope"]
    )

    # Buckets: SM strengthening vs weakening at entry
    bins = [-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf]
    labels = ["strong against", "weak against", "flat", "weak with", "strong with"]
    valid["sm_slope_bucket"] = pd.cut(valid["sm_aligned_slope"], bins=bins, labels=labels)

    rows = []
    for bucket in labels:
        subset = valid[valid["sm_slope_bucket"] == bucket]
        rows.append(compute_stats(subset, bucket))
    rows.append(compute_stats(valid, "ALL"))
    print_table(rows, "1A: Performance by SM Slope Direction at Entry")

    # Per-strategy
    for strat in sorted(valid["strategy"].unique()):
        sdf = valid[valid["strategy"] == strat]
        rows = []
        for bucket in labels:
            subset = sdf[sdf["sm_slope_bucket"] == bucket]
            rows.append(compute_stats(subset, bucket))
        rows.append(compute_stats(sdf, "ALL"))
        print_table(rows, f"1A: SM Slope — {strat}")

    # ---- 1B: SM Zero-Crossing Count (SM "confused") ----
    print("\n  === 1B: SM Zero-Crossing Count (last 60 bars) ===")
    print("  Hypothesis: More flips = SM confused = entries are random")

    bins_zc = [-np.inf, 0.5, 1.5, 2.5, 4.5, np.inf]
    labels_zc = ["0 flips", "1 flip", "2 flips", "3-4 flips", "5+ flips"]
    valid["zc_bucket_60"] = pd.cut(valid["sm_zero_cross_60"], bins=bins_zc, labels=labels_zc)

    rows = []
    for bucket in labels_zc:
        subset = valid[valid["zc_bucket_60"] == bucket]
        rows.append(compute_stats(subset, bucket))
    rows.append(compute_stats(valid, "ALL"))
    print_table(rows, "1B: Performance by SM Zero-Crossing Count (60 bars)")

    # Also 120-bar window
    bins_zc2 = [-np.inf, 1.5, 3.5, 5.5, 8.5, np.inf]
    labels_zc2 = ["0-1 flips", "2-3 flips", "4-5 flips", "6-8 flips", "9+ flips"]
    valid["zc_bucket_120"] = pd.cut(valid["sm_zero_cross_120"], bins=bins_zc2, labels=labels_zc2)

    rows = []
    for bucket in labels_zc2:
        subset = valid[valid["zc_bucket_120"] == bucket]
        rows.append(compute_stats(subset, bucket))
    rows.append(compute_stats(valid, "ALL"))
    print_table(rows, "1B: Performance by SM Zero-Crossing Count (120 bars)")

    # Per-strategy for 60-bar
    for strat in sorted(valid["strategy"].unique()):
        sdf = valid[valid["strategy"] == strat]
        rows = []
        for bucket in labels_zc:
            subset = sdf[sdf["zc_bucket_60"] == bucket]
            rows.append(compute_stats(subset, bucket))
        rows.append(compute_stats(sdf, "ALL"))
        print_table(rows, f"1B: SM Flips (60 bars) — {strat}")

    # ---- 1C: Bars Since Last SM Flip (Fresh vs Stale) ----
    print("\n  === 1C: Bars Since Last SM Flip ===")
    print("  Hypothesis: Fresh SM flip (few bars ago) = catching the turn. Stale = late entry.")

    bins_fresh = [-np.inf, 5, 15, 30, 60, np.inf]
    labels_fresh = ["0-5 (very fresh)", "6-15 (fresh)", "16-30 (moderate)",
                     "31-60 (stale)", "60+ (very stale)"]
    valid["freshness"] = pd.cut(valid["sm_bars_since_flip"], bins=bins_fresh, labels=labels_fresh)

    rows = []
    for bucket in labels_fresh:
        subset = valid[valid["freshness"] == bucket]
        rows.append(compute_stats(subset, bucket))
    rows.append(compute_stats(valid, "ALL"))
    print_table(rows, "1C: Performance by SM Signal Freshness")

    for strat in sorted(valid["strategy"].unique()):
        sdf = valid[valid["strategy"] == strat]
        rows = []
        for bucket in labels_fresh:
            subset = sdf[sdf["freshness"] == bucket]
            rows.append(compute_stats(subset, bucket))
        rows.append(compute_stats(sdf, "ALL"))
        print_table(rows, f"1C: SM Freshness — {strat}")

    # ---- 1D: Pre-Entry Momentum (did we chase?) ----
    print("\n  === 1D: Pre-Entry Momentum (10-bar) ===")
    print("  Hypothesis: Large move in trade direction before entry = chasing = reversal risk")

    for strat in sorted(valid["strategy"].unique()):
        sdf = valid[valid["strategy"] == strat].copy()
        if len(sdf) == 0:
            continue

        # Strategy-specific momentum bins
        q_labels = ["Q1 (counter-move)", "Q2", "Q3", "Q4 (chasing)"]
        try:
            sdf["mom_quartile"] = pd.qcut(sdf["pre_momentum_10"].dropna(),
                                           q=4, labels=q_labels, duplicates="drop")
        except ValueError:
            continue

        rows = []
        for q in q_labels:
            subset = sdf[sdf["mom_quartile"] == q]
            rows.append(compute_stats(subset, q))
        rows.append(compute_stats(sdf[sdf["mom_quartile"].notna()], "ALL"))
        print_table(rows, f"1D: Pre-Entry Momentum (10-bar) — {strat}")

    # ---- 1E: Volume Spike at Entry ----
    print("\n  === 1E: Volume Ratio at Entry (vs 20-bar avg) ===")
    print("  Hypothesis: Volume spike at entry = exhaustion = reversal risk")

    vol_valid = valid[valid["volume_ratio"].notna()]
    if len(vol_valid) > 0:
        bins_vol = [0, 0.5, 0.8, 1.2, 2.0, np.inf]
        labels_vol = ["<0.5x (quiet)", "0.5-0.8x (low)", "0.8-1.2x (normal)",
                       "1.2-2.0x (elevated)", "2.0x+ (spike)"]
        vol_valid = vol_valid.copy()
        vol_valid["vol_bucket"] = pd.cut(vol_valid["volume_ratio"], bins=bins_vol, labels=labels_vol)

        rows = []
        for bucket in labels_vol:
            subset = vol_valid[vol_valid["vol_bucket"] == bucket]
            rows.append(compute_stats(subset, bucket))
        rows.append(compute_stats(vol_valid, "ALL"))
        print_table(rows, "1E: Performance by Volume Ratio at Entry")

        for strat in sorted(vol_valid["strategy"].unique()):
            sdf = vol_valid[vol_valid["strategy"] == strat]
            rows = []
            for bucket in labels_vol:
                subset = sdf[sdf["vol_bucket"] == bucket]
                rows.append(compute_stats(subset, bucket))
            rows.append(compute_stats(sdf, "ALL"))
            print_table(rows, f"1E: Volume Ratio — {strat}")

    # ---- 1F: Entry Bar Candle Structure ----
    print("\n  === 1F: Entry Bar Rejection Wick ===")
    print("  Hypothesis: Long rejection wick in trade direction on entry bar = trap")

    wick_valid = valid[valid["entry_wick_ratio"].notna()]
    if len(wick_valid) > 0:
        bins_wick = [0, 0.15, 0.30, 0.50, 1.0]
        labels_wick = ["<15% wick", "15-30% wick", "30-50% wick", "50%+ wick (trap bar)"]
        wick_valid = wick_valid.copy()
        wick_valid["wick_bucket"] = pd.cut(wick_valid["entry_wick_ratio"],
                                            bins=bins_wick, labels=labels_wick)

        rows = []
        for bucket in labels_wick:
            subset = wick_valid[wick_valid["wick_bucket"] == bucket]
            rows.append(compute_stats(subset, bucket))
        rows.append(compute_stats(wick_valid, "ALL"))
        print_table(rows, "1F: Performance by Entry Bar Rejection Wick")

        for strat in sorted(wick_valid["strategy"].unique()):
            sdf = wick_valid[wick_valid["strategy"] == strat]
            rows = []
            for bucket in labels_wick:
                subset = sdf[sdf["wick_bucket"] == bucket]
                rows.append(compute_stats(subset, bucket))
            rows.append(compute_stats(sdf, "ALL"))
            print_table(rows, f"1F: Rejection Wick — {strat}")

    if do_plot:
        _plot_step1(valid)


def _plot_step1(valid):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("STEP 1: Entry-Level Signal Analysis", fontsize=14, fontweight="bold")

    # 1A: SM Slope scatter
    ax = axes[0, 0]
    for strat, color in [("MNQ_V15", "blue"), ("MNQ_VSCALPB", "green"), ("MES_V2", "red")]:
        sdf = valid[valid["strategy"] == strat]
        ax.scatter(sdf["sm_aligned_slope"], sdf["pnl_dollar"], alpha=0.3, s=10,
                   color=color, label=strat)
    ax.axvline(0, color="black", lw=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("SM Slope (aligned to trade direction)")
    ax.set_ylabel("P&L ($)")
    ax.set_title("1A: SM Slope vs P&L")
    ax.legend(fontsize=7)

    # 1B: Zero-crossing count vs SL rate
    ax = axes[0, 1]
    for strat, color in [("MNQ_V15", "blue"), ("MNQ_VSCALPB", "green"), ("MES_V2", "red")]:
        sdf = valid[valid["strategy"] == strat]
        if sdf["sm_zero_cross_60"].notna().sum() > 0:
            grouped = sdf.groupby(sdf["sm_zero_cross_60"].clip(upper=8).astype(int))
            sl_rates = grouped.apply(lambda x: (x["exit_reason"] == "SL").mean() * 100)
            ax.plot(sl_rates.index, sl_rates.values, marker="o", label=strat, color=color)
    ax.set_xlabel("SM Zero-Crossings (60 bars)")
    ax.set_ylabel("SL Rate (%)")
    ax.set_title("1B: SM Flap Count vs SL Rate")
    ax.legend(fontsize=7)

    # 1C: Bars since flip vs avg P&L
    ax = axes[0, 2]
    for strat, color in [("MNQ_V15", "blue"), ("MNQ_VSCALPB", "green"), ("MES_V2", "red")]:
        sdf = valid[valid["strategy"] == strat]
        bins = [0, 5, 15, 30, 60, 200]
        labels = ["0-5", "6-15", "16-30", "31-60", "60+"]
        sdf = sdf.copy()
        sdf["fb"] = pd.cut(sdf["sm_bars_since_flip"], bins=bins, labels=labels)
        grouped = sdf.groupby("fb", observed=True)["pnl_dollar"].mean()
        ax.bar([f"{l}\n{strat[:5]}" for l in grouped.index],
               grouped.values, alpha=0.7, color=color)
    ax.set_xlabel("Bars Since SM Flip")
    ax.set_ylabel("Avg P&L ($)")
    ax.set_title("1C: Signal Freshness vs Avg P&L")
    ax.tick_params(axis='x', labelsize=6)

    # 1D: Pre-entry momentum vs P&L scatter
    ax = axes[1, 0]
    for strat, color in [("MNQ_V15", "blue"), ("MNQ_VSCALPB", "green"), ("MES_V2", "red")]:
        sdf = valid[valid["strategy"] == strat]
        ax.scatter(sdf["pre_momentum_10"], sdf["pnl_dollar"], alpha=0.3, s=10,
                   color=color, label=strat)
    ax.axvline(0, color="black", lw=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Pre-Entry Momentum 10-bar (directional)")
    ax.set_ylabel("P&L ($)")
    ax.set_title("1D: Pre-Entry Momentum vs P&L")
    ax.legend(fontsize=7)

    # 1E: Volume ratio vs SL rate
    ax = axes[1, 1]
    vol_valid = valid[valid["volume_ratio"].notna()]
    if len(vol_valid) > 0:
        for strat, color in [("MNQ_V15", "blue"), ("MNQ_VSCALPB", "green"), ("MES_V2", "red")]:
            sdf = vol_valid[vol_valid["strategy"] == strat]
            if len(sdf) > 10:
                bins = [0, 0.5, 0.8, 1.2, 2.0, 10.0]
                labels = ["<0.5x", "0.5-0.8x", "0.8-1.2x", "1.2-2x", "2x+"]
                sdf = sdf.copy()
                sdf["vb"] = pd.cut(sdf["volume_ratio"], bins=bins, labels=labels)
                grouped = sdf.groupby("vb", observed=True)
                sl_rates = grouped.apply(lambda x: (x["exit_reason"] == "SL").mean() * 100)
                ax.plot(range(len(sl_rates)), sl_rates.values, marker="o",
                        label=strat, color=color)
                ax.set_xticks(range(len(sl_rates)))
                ax.set_xticklabels(sl_rates.index, fontsize=7)
    ax.set_xlabel("Volume Ratio")
    ax.set_ylabel("SL Rate (%)")
    ax.set_title("1E: Volume Ratio vs SL Rate")
    ax.legend(fontsize=7)

    # 1F: Wick ratio histogram for winners vs losers
    ax = axes[1, 2]
    w = valid[valid["win"]]["entry_wick_ratio"].dropna()
    l = valid[~valid["win"]]["entry_wick_ratio"].dropna()
    if len(w) > 0 and len(l) > 0:
        bins = np.arange(0, 1.02, 0.05)
        ax.hist(w, bins=bins, alpha=0.6, color="green", label=f"Winners ({len(w)})",
                density=True, edgecolor="black")
        ax.hist(l, bins=bins, alpha=0.6, color="red", label=f"Losers ({len(l)})",
                density=True, edgecolor="black")
    ax.set_xlabel("Rejection Wick Ratio")
    ax.set_ylabel("Density")
    ax.set_title("1F: Entry Bar Wick — Winners vs Losers")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step1_entry_signals.png", dpi=150)
    print(f"\n  Saved step1_entry_signals.png")
    plt.close()


# =====================================================================
# STEP 2: DAILY REGIME (LAGGED PREDICTORS)
# =====================================================================

def step2_daily_regime(trades: pd.DataFrame, market: pd.DataFrame,
                       do_plot: bool = False):
    """Test daily VIX-based regime signals using PRIOR day data (knowable at entry)."""
    print("\n" + "=" * 95)
    print("  STEP 2: DAILY REGIME — LAGGED PREDICTORS")
    print("=" * 95)

    # Merge trades with market data (on trade_date)
    trades["_merge_date"] = trades["trade_date"].dt.normalize()
    market["_merge_date"] = pd.to_datetime(market["date"]).dt.normalize()
    merged = trades.merge(market, on="_merge_date", how="left")
    merged = merged[merged["vix_close"].notna()]
    print(f"\n  Trades with market data: {len(merged)} / {len(trades)}")

    # ---- 2A: Rule of 16 (realized vs implied vol) ----
    print("\n  === 2A: Rule of 16 — Realized vs Implied Volatility ===")
    print("  Hypothesis: When realized daily range exceeds VIX-implied, market is chaotic")

    # Use PRIOR day's data (knowable at today's open)
    market["es_range_pct"] = (market["es_high"] - market["es_low"]) / market["es_open"] * 100
    market["vix_implied_move"] = market["vix_close"] / np.sqrt(252)
    market["realized_vs_implied"] = market["es_range_pct"] / market["vix_implied_move"]

    # Shift: use YESTERDAY's ratio to predict today
    market["prev_realized_vs_implied"] = market["realized_vs_implied"].shift(1)
    market["prev_es_range_pct"] = market["es_range_pct"].shift(1)
    market["prev_vix_implied_move"] = market["vix_implied_move"].shift(1)

    # Re-merge with lagged columns
    merged = trades.merge(
        market[["_merge_date", "prev_realized_vs_implied", "prev_es_range_pct",
                "prev_vix_implied_move", "vix_overnight_gap", "vix_close",
                "vix_pct_change", "vix_prev_close", "vix_prev_high", "vix_prev_low",
                "vix_open"]],
        on="_merge_date", how="left"
    )
    merged = merged[merged["vix_close"].notna()]

    rvi_valid = merged[merged["prev_realized_vs_implied"].notna()]
    if len(rvi_valid) > 0:
        bins = [0, 0.5, 0.75, 1.0, 1.5, np.inf]
        labels = ["<0.5 (very quiet)", "0.5-0.75 (calm)", "0.75-1.0 (normal)",
                   "1.0-1.5 (hot)", "1.5+ (over-delivering)"]
        rvi_valid = rvi_valid.copy()
        rvi_valid["r16_bucket"] = pd.cut(rvi_valid["prev_realized_vs_implied"],
                                          bins=bins, labels=labels)

        rows = []
        for bucket in labels:
            subset = rvi_valid[rvi_valid["r16_bucket"] == bucket]
            rows.append(compute_stats(subset, bucket))
        rows.append(compute_stats(rvi_valid, "ALL"))
        print_table(rows, "2A: Performance by Prior Day Realized-vs-Implied (Rule of 16)")

        for strat in sorted(rvi_valid["strategy"].unique()):
            sdf = rvi_valid[rvi_valid["strategy"] == strat]
            rows = []
            for bucket in labels:
                subset = sdf[sdf["r16_bucket"] == bucket]
                rows.append(compute_stats(subset, bucket))
            rows.append(compute_stats(sdf, "ALL"))
            print_table(rows, f"2A: Rule of 16 — {strat}")

    # ---- 2B: VIX Overnight Gap ----
    print("\n  === 2B: VIX Overnight Gap (VIX open vs prior close) ===")
    print("  Hypothesis: VIX gap-up at open = overnight fear = bad for scalps")

    gap_valid = merged[merged["vix_overnight_gap"].notna()]
    if len(gap_valid) > 0:
        bins = [-np.inf, -2, -0.5, 0.5, 2, np.inf]
        labels = ["gap down >2%", "gap down 0.5-2%", "flat gap (±0.5%)",
                   "gap up 0.5-2%", "gap up >2%"]
        gap_valid = gap_valid.copy()
        gap_valid["gap_bucket"] = pd.cut(gap_valid["vix_overnight_gap"],
                                          bins=bins, labels=labels)

        rows = []
        for bucket in labels:
            subset = gap_valid[gap_valid["gap_bucket"] == bucket]
            rows.append(compute_stats(subset, bucket))
        rows.append(compute_stats(gap_valid, "ALL"))
        print_table(rows, "2B: Performance by VIX Overnight Gap")

        for strat in sorted(gap_valid["strategy"].unique()):
            sdf = gap_valid[gap_valid["strategy"] == strat]
            rows = []
            for bucket in labels:
                subset = sdf[sdf["gap_bucket"] == bucket]
                rows.append(compute_stats(subset, bucket))
            rows.append(compute_stats(sdf, "ALL"))
            print_table(rows, f"2B: VIX Overnight Gap — {strat}")

    # ---- 2C: Multi-Day VIX Momentum (3-day and 5-day) ----
    print("\n  === 2C: Multi-Day VIX Momentum ===")
    print("  Hypothesis: VIX trending down over 3-5 days = stable, good. Trending up = bad.")

    market["vix_3d_change"] = (market["vix_close"] / market["vix_close"].shift(3) - 1) * 100
    market["vix_5d_change"] = (market["vix_close"] / market["vix_close"].shift(5) - 1) * 100

    # Use prior day's 3d/5d change
    market["prev_vix_3d_change"] = market["vix_3d_change"].shift(1)
    market["prev_vix_5d_change"] = market["vix_5d_change"].shift(1)

    merged2 = trades.merge(
        market[["_merge_date", "prev_vix_3d_change", "prev_vix_5d_change"]],
        on="_merge_date", how="left"
    )

    for days, col in [("3-day", "prev_vix_3d_change"), ("5-day", "prev_vix_5d_change")]:
        m_valid = merged2[merged2[col].notna()]
        if len(m_valid) == 0:
            continue

        bins = [-np.inf, -5, -2, 2, 5, np.inf]
        labels = [f"crashing (<-5%)", f"falling (-5 to -2%)", f"flat (±2%)",
                   f"rising (2-5%)", f"surging (>5%)"]
        m_valid = m_valid.copy()
        m_valid["mom_bucket"] = pd.cut(m_valid[col], bins=bins, labels=labels)

        rows = []
        for bucket in labels:
            subset = m_valid[m_valid["mom_bucket"] == bucket]
            rows.append(compute_stats(subset, bucket))
        rows.append(compute_stats(m_valid, "ALL"))
        print_table(rows, f"2C: Performance by Prior {days} VIX Momentum")

    # ---- 2D: VIX Q4 Band Exclusion with IS/OOS ----
    print("\n  === 2D: VIX Q4 Band Exclusion (19-22) — IS/OOS Split ===")
    print("  Hypothesis: VIX 19-22 is the death zone, especially for vScalpB")

    vix_valid = merged[merged["vix_prev_close"].notna()].copy()
    vix_valid["in_death_zone"] = (vix_valid["vix_prev_close"] >= 19) & (vix_valid["vix_prev_close"] <= 22)

    # Split IS/OOS by date (first half / second half)
    dates = sorted(vix_valid["trade_date"].unique())
    mid = dates[len(dates) // 2]
    is_mask = vix_valid["trade_date"] <= mid
    oos_mask = vix_valid["trade_date"] > mid

    print(f"\n  IS period: {dates[0].strftime('%Y-%m-%d')} to {mid.strftime('%Y-%m-%d')}")
    print(f"  OOS period: {(mid + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    for period, mask, label in [("IS", is_mask, "In-Sample"), ("OOS", oos_mask, "Out-of-Sample")]:
        subset = vix_valid[mask]
        for strat in sorted(subset["strategy"].unique()):
            sdf = subset[subset["strategy"] == strat]
            normal = sdf[~sdf["in_death_zone"]]
            death = sdf[sdf["in_death_zone"]]
            rows = [
                compute_stats(normal, f"VIX outside 19-22"),
                compute_stats(death, f"VIX 19-22 (death zone)"),
                compute_stats(sdf, "ALL"),
            ]
            print_table(rows, f"2D: VIX Q4 Band — {strat} ({label})")

    # ---- 2E: VIX Prior Day Range (compression) ----
    print("\n  === 2E: VIX Prior Day Range (compression → expansion) ===")
    print("  Hypothesis: Yesterday's tight VIX range = coiled energy = expansion today")

    market["vix_range"] = market["vix_high"] - market["vix_low"]
    market["prev_vix_range"] = market["vix_range"].shift(1)

    merged3 = trades.merge(
        market[["_merge_date", "prev_vix_range"]],
        on="_merge_date", how="left"
    )
    range_valid = merged3[merged3["prev_vix_range"].notna()]
    if len(range_valid) > 0:
        try:
            range_valid = range_valid.copy()
            range_valid["range_tercile"] = pd.qcut(range_valid["prev_vix_range"], q=3,
                                                     labels=["tight", "normal", "wide"],
                                                     duplicates="drop")
            rows = []
            for t in ["tight", "normal", "wide"]:
                subset = range_valid[range_valid["range_tercile"] == t]
                rows.append(compute_stats(subset, f"Prev VIX range: {t}"))
            rows.append(compute_stats(range_valid, "ALL"))
            print_table(rows, "2E: Performance by Prior Day VIX Range")

            for strat in sorted(range_valid["strategy"].unique()):
                sdf = range_valid[range_valid["strategy"] == strat]
                rows = []
                for t in ["tight", "normal", "wide"]:
                    subset = sdf[sdf["range_tercile"] == t]
                    rows.append(compute_stats(subset, f"Prev VIX range: {t}"))
                rows.append(compute_stats(sdf, "ALL"))
                print_table(rows, f"2E: VIX Range — {strat}")
        except ValueError:
            print("  Could not create terciles for VIX range")

    if do_plot:
        _plot_step2(merged, market)


def _plot_step2(merged, market):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("STEP 2: Daily Regime (Lagged Predictors)", fontsize=14, fontweight="bold")

    # 2A: Rule of 16 distribution
    ax = axes[0, 0]
    rvi = market["realized_vs_implied"].dropna()
    if len(rvi) > 0:
        ax.hist(rvi, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(1.0, color="red", ls="--", lw=2, label="Implied = Realized")
        ax.set_title("2A: Rule of 16 Distribution")
        ax.set_xlabel("Realized / Implied Daily Range")
        ax.legend()

    # 2B: VIX overnight gap distribution
    ax = axes[0, 1]
    gaps = market["vix_overnight_gap"].dropna()
    if len(gaps) > 0:
        ax.hist(gaps, bins=30, color="orange", alpha=0.7, edgecolor="black")
        ax.axvline(0, color="black", lw=1)
        ax.set_title("2B: VIX Overnight Gap Distribution")
        ax.set_xlabel("VIX Gap (%)")

    # 2C: 5-day VIX momentum vs daily P&L
    ax = axes[0, 2]
    if "prev_vix_5d_change" in merged.columns:
        daily = merged.groupby("_merge_date").agg(
            daily_pnl=("pnl_dollar", "sum"),
            vix_5d=("vix_close", "first")  # just for merge
        ).reset_index()
        daily2 = daily.merge(
            market[["_merge_date", "prev_vix_5d_change"]], on="_merge_date", how="left"
        )
        valid = daily2[daily2["prev_vix_5d_change"].notna()]
        if len(valid) > 0:
            ax.scatter(valid["prev_vix_5d_change"], valid["daily_pnl"],
                       alpha=0.5, s=20, color="purple")
            ax.axhline(0, color="black", lw=1)
            ax.axvline(0, color="black", lw=1)
            ax.set_xlabel("Prior 5-Day VIX Change (%)")
            ax.set_ylabel("Daily Portfolio P&L ($)")
            ax.set_title("2C: 5-Day VIX Momentum vs Daily P&L")

    # 2D: VIX Q4 death zone - SL rate by strategy
    ax = axes[1, 0]
    if "vix_close" in merged.columns:
        for strat, color in [("MNQ_V15", "blue"), ("MNQ_VSCALPB", "green"), ("MES_V2", "red")]:
            sdf = merged[merged["strategy"] == strat]
            if len(sdf) == 0:
                continue
            bins = np.arange(12, 40, 2)
            sdf = sdf.copy()
            sdf["vix_bin"] = pd.cut(sdf["vix_close"], bins=bins)
            grouped = sdf.groupby("vix_bin", observed=True)
            sl_rates = grouped.apply(lambda x: (x["exit_reason"] == "SL").mean() * 100)
            mids = [iv.mid for iv in sl_rates.index]
            ax.plot(mids, sl_rates.values, marker="o", label=strat, color=color, alpha=0.7)
        ax.axvspan(19, 22, alpha=0.2, color="red", label="Death Zone")
        ax.set_xlabel("VIX Close")
        ax.set_ylabel("SL Rate (%)")
        ax.set_title("2D: SL Rate by VIX Level")
        ax.legend(fontsize=7)

    # 2E: Prior day VIX range vs next day P&L
    ax = axes[1, 1]
    if "prev_vix_range" in market.columns:
        daily = merged.groupby("_merge_date")["pnl_dollar"].sum().reset_index()
        daily2 = daily.merge(
            market[["_merge_date", "prev_vix_range"]], on="_merge_date", how="left"
        )
        valid = daily2[daily2["prev_vix_range"].notna()]
        if len(valid) > 0:
            ax.scatter(valid["prev_vix_range"], valid["pnl_dollar"],
                       alpha=0.5, s=20, color="teal")
            ax.axhline(0, color="black", lw=1)
            ax.set_xlabel("Prior Day VIX Range (pts)")
            ax.set_ylabel("Next Day Portfolio P&L ($)")
            ax.set_title("2E: VIX Compression → Next Day P&L")

    # Summary: correlation heatmap of lagged predictors
    ax = axes[1, 2]
    pred_cols = ["prev_realized_vs_implied", "vix_overnight_gap", "vix_close"]
    available = [c for c in pred_cols if c in merged.columns]
    if available:
        corr_data = merged[available + ["pnl_dollar"]].corr()
        im = ax.imshow(corr_data.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(corr_data.columns)))
        ax.set_yticks(range(len(corr_data.columns)))
        labels = [c.replace("prev_", "").replace("_", "\n")[:15] for c in corr_data.columns]
        ax.set_xticklabels(labels, fontsize=7, rotation=45)
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(len(corr_data)):
            for j in range(len(corr_data)):
                ax.text(j, i, f"{corr_data.values[i,j]:.2f}",
                        ha="center", va="center", fontsize=8)
        ax.set_title("Correlation Matrix")
        fig.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step2_daily_regime.png", dpi=150)
    print(f"\n  Saved step2_daily_regime.png")
    plt.close()


# =====================================================================
# STEP 3: INTERACTION EFFECTS
# =====================================================================

def step3_interactions(trades: pd.DataFrame, market: pd.DataFrame,
                       do_plot: bool = False):
    """Do entry-level filters work better on certain VIX regime days?"""
    print("\n" + "=" * 95)
    print("  STEP 3: INTERACTION EFFECTS")
    print("=" * 95)
    print("  Testing: Do entry-level signals interact with daily regime?")

    # Merge
    trades["_merge_date"] = trades["trade_date"].dt.normalize()
    market["_merge_date"] = pd.to_datetime(market["date"]).dt.normalize()
    merged = trades.merge(market[["_merge_date", "vix_close"]], on="_merge_date", how="left")
    valid = merged[merged["sm_slope"].notna() & merged["vix_close"].notna()]

    if len(valid) == 0:
        print("  No data for interaction analysis")
        return

    # Create VIX regime: death zone vs safe
    valid = valid.copy()
    valid["vix_regime"] = np.where(
        (valid["vix_close"] >= 19) & (valid["vix_close"] <= 22),
        "VIX 19-22", "VIX safe"
    )

    # SM slope direction
    valid["sm_aligned_slope"] = np.where(
        valid["side"] == "long", valid["sm_slope"], -valid["sm_slope"]
    )
    valid["sm_weakening"] = valid["sm_aligned_slope"] < -0.005

    # SM flap count
    valid["sm_confused"] = valid["sm_zero_cross_60"] >= 3

    # Combined filters
    print("\n  === 3A: SM Weakening × VIX Regime ===")
    for regime in ["VIX safe", "VIX 19-22"]:
        for sm_state, sm_label in [(True, "SM weakening"), (False, "SM steady/strengthening")]:
            subset = valid[(valid["vix_regime"] == regime) & (valid["sm_weakening"] == sm_state)]
            label = f"{regime} + {sm_label}"
            rows = [compute_stats(subset, label)]
            for r in rows:
                pf = f"{r['pf']:.3f}" if r['pf'] < 100 else "inf"
                print(f"  {r['label']:<45} N={r['n']:>4}  WR={r['wr']:.1f}%  "
                      f"PF={pf}  Avg=${r['avg_pnl']:+.2f}  SL%={r['sl_rate']:.1f}%")

    print("\n  === 3B: SM Confused × VIX Regime ===")
    for regime in ["VIX safe", "VIX 19-22"]:
        for confused, label in [(True, "SM confused (3+ flips)"), (False, "SM stable (<3 flips)")]:
            subset = valid[(valid["vix_regime"] == regime) & (valid["sm_confused"] == confused)]
            combo_label = f"{regime} + {label}"
            rows = [compute_stats(subset, combo_label)]
            for r in rows:
                pf = f"{r['pf']:.3f}" if r['pf'] < 100 else "inf"
                print(f"  {r['label']:<50} N={r['n']:>4}  WR={r['wr']:.1f}%  "
                      f"PF={pf}  Avg=${r['avg_pnl']:+.2f}  SL%={r['sl_rate']:.1f}%")

    # Per-strategy interaction for vScalpB (most affected by death zone)
    print("\n  === 3C: vScalpB Specific — Combined Filter ===")
    vb = valid[valid["strategy"] == "MNQ_VSCALPB"].copy()
    if len(vb) > 0:
        # Best combo: skip when (VIX death zone AND SM weakening)
        # or (SM confused)
        vb["skip_signal"] = (
            ((vb["vix_regime"] == "VIX 19-22") & (vb["sm_weakening"]))
            | (vb["sm_confused"])
        )

        rows = [
            compute_stats(vb[~vb["skip_signal"]], "KEEP (pass filter)"),
            compute_stats(vb[vb["skip_signal"]], "SKIP (would filter)"),
            compute_stats(vb, "ALL (baseline)"),
        ]
        print_table(rows, "3C: vScalpB — Combined Skip Filter")

        # Count how many SL exits the filter would have avoided
        skipped_sl = vb[vb["skip_signal"] & (vb["exit_reason"] == "SL")]
        skipped_tp = vb[vb["skip_signal"] & (vb["exit_reason"] == "TP")]
        print(f"\n  Filter would skip {len(vb[vb['skip_signal']])} trades total:")
        print(f"    {len(skipped_sl)} would-be SL exits avoided")
        print(f"    {len(skipped_tp)} would-be TP exits sacrificed")
        if len(skipped_sl) + len(skipped_tp) > 0:
            ratio = len(skipped_sl) / (len(skipped_sl) + len(skipped_tp))
            print(f"    SL avoidance ratio: {ratio:.1%} (want >50% to be net positive)")

    # Repeat for MNQ_V15
    print("\n  === 3D: MNQ_V15 — Combined Filter ===")
    v15 = valid[valid["strategy"] == "MNQ_V15"].copy()
    if len(v15) > 0:
        v15["skip_signal"] = (
            ((v15["vix_regime"] == "VIX 19-22") & (v15["sm_weakening"]))
            | (v15["sm_confused"])
        )
        rows = [
            compute_stats(v15[~v15["skip_signal"]], "KEEP (pass filter)"),
            compute_stats(v15[v15["skip_signal"]], "SKIP (would filter)"),
            compute_stats(v15, "ALL (baseline)"),
        ]
        print_table(rows, "3D: MNQ_V15 — Combined Skip Filter")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Immediate Reversal Diagnostic")
    parser.add_argument("--plot", action="store_true", help="Save charts")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    trades = load_trades()
    bars = load_1min_bars()

    # Date range for market data
    start = trades["trade_date"].min().strftime("%Y-%m-%d")
    end = (trades["trade_date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    market = load_vix_data(start, end)

    # Enrich trades with 1-min bar context
    trades = enrich_trades_with_bar_context(trades, bars)

    # Run all steps
    step0_sl_diagnostic(trades, do_plot=args.plot)
    step1_entry_signals(trades, do_plot=args.plot)
    step2_daily_regime(trades, market, do_plot=args.plot)
    step3_interactions(trades, market, do_plot=args.plot)

    # Save enriched trades
    out_csv = OUTPUT_DIR / "trades_with_entry_context.csv"
    trades.to_csv(out_csv, index=False)
    print(f"\n  Saved enriched trades to {out_csv}")

    print("\n" + "=" * 95)
    print("  DONE — Review results above for actionable filters")
    print("=" * 95)


if __name__ == "__main__":
    main()
