"""
Backtest Results Archive
========================
Utility to persist per-trade backtest results so they can be analyzed later
(e.g. news-day impact, regime analysis, drawdown forensics).

Usage from any backtest script:
    from results.save_results import save_backtest

    trades = run_backtest_tp_exit(...)
    save_backtest(
        trades,
        strategy="MNQ_VSCALPB",
        params={"sm_t": 0.25, "rsi_buy": 55, ...},
        data_range="2025-02-17_to_2026-02-13",
        split="FULL",           # or "IS", "OOS"
        dollar_per_pt=2.0,
        commission=0.52,
        notes="Production params, 12-month run",
    )

Files are saved to backtesting_engine/results/ as:
    {strategy}_{split}_{data_range}_{timestamp}.csv
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent


def save_backtest(trades, strategy, params, data_range, split="FULL",
                  dollar_per_pt=2.0, commission=0.52, notes="",
                  script_name="", qty=1):
    """Save a list of trade dicts to a CSV with metadata header.

    Args:
        trades: list of trade dicts from run_backtest_tp_exit or run_backtest_v10.
            Required keys: side, entry, exit, pts, entry_time, exit_time,
                           entry_idx, exit_idx, bars, result
            Optional keys: mfe, mae
        strategy: strategy identifier (e.g. "MNQ_VSCALPB", "MES_V2")
        params: dict of strategy parameters used
        data_range: string like "2025-02-17_to_2026-02-13"
        split: "FULL", "IS", or "OOS"
        dollar_per_pt: dollar value per point for the instrument
        commission: per-side commission in dollars
        notes: free-form text
        script_name: name of the calling script
        qty: number of contracts (for commission calculation; pts already
             reflects total across all legs)

    Returns:
        Path to the saved CSV file.
    """
    if not trades:
        print(f"  [save_backtest] No trades for {strategy}/{split}, skipping.")
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy}_{split}_{data_range}_{timestamp}.csv"
    filepath = RESULTS_DIR / filename

    # Compute derived fields
    rows = []
    cumulative_pnl = 0.0
    for i, t in enumerate(trades):
        pnl_pts = t["pts"]
        pnl_dollar = pnl_pts * dollar_per_pt - 2 * commission * qty
        cumulative_pnl += pnl_dollar

        entry_time = t["entry_time"]
        exit_time = t["exit_time"]

        # Convert numpy timestamps to ISO strings if needed
        if isinstance(entry_time, (np.datetime64, pd.Timestamp)):
            entry_ts = pd.Timestamp(entry_time)
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_time = entry_ts.isoformat()
        if isinstance(exit_time, (np.datetime64, pd.Timestamp)):
            exit_ts = pd.Timestamp(exit_time)
            if exit_ts.tz is None:
                exit_ts = exit_ts.tz_localize("UTC")
            exit_time = exit_ts.isoformat()

        # ET date for the trade (based on entry time)
        try:
            entry_ts = pd.Timestamp(entry_time)
            if entry_ts.tz is None:
                entry_ts = entry_ts.tz_localize("UTC")
            entry_et = entry_ts.tz_convert("America/New_York")
            trade_date = entry_et.strftime("%Y-%m-%d")
            day_of_week = entry_et.strftime("%A")
        except Exception:
            trade_date = ""
            day_of_week = ""

        rows.append({
            "trade_num": i + 1,
            "trade_date": trade_date,
            "day_of_week": day_of_week,
            "side": t["side"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": round(t["entry"], 2),
            "exit_price": round(t["exit"], 2),
            "pts": round(pnl_pts, 2),
            "pnl_dollar": round(pnl_dollar, 2),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "exit_reason": t.get("result", ""),
            "bars_held": t.get("bars", 0),
            "mfe_pts": round(t.get("mfe", 0.0), 2),
            "mae_pts": round(t.get("mae", 0.0), 2),
        })

    # Metadata block (first lines of CSV as comments)
    meta = {
        "strategy": strategy,
        "split": split,
        "data_range": data_range,
        "params": params,
        "dollar_per_pt": dollar_per_pt,
        "commission_per_side": commission,
        "qty": qty,
        "total_trades": len(rows),
        "total_pnl": round(cumulative_pnl, 2),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "script": script_name,
        "notes": notes,
    }

    fieldnames = list(rows[0].keys())

    with open(filepath, "w", newline="") as f:
        # Write metadata as comment lines
        f.write(f"# BACKTEST RESULTS: {strategy} ({split})\n")
        f.write(f"# {json.dumps(meta)}\n")

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} trades -> {filepath.name}")
    return filepath


def load_backtest(filepath):
    """Load a saved backtest CSV, returning (metadata_dict, trades_dataframe).

    Reads the JSON metadata from comment line 2, then loads the CSV body.
    """
    filepath = Path(filepath)
    meta = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith("# {"):
                meta = json.loads(line[2:])
                break

    df = pd.read_csv(filepath, comment="#")
    return meta, df


def list_results(strategy=None, split=None):
    """List saved result files, optionally filtered by strategy/split.

    Returns list of (filepath, metadata) tuples.
    """
    results = []
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        try:
            meta, _ = load_backtest(f)
            if strategy and meta.get("strategy") != strategy:
                continue
            if split and meta.get("split") != split:
                continue
            results.append((f, meta))
        except Exception:
            continue
    return results
