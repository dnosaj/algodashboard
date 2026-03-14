"""
Backfill backtest trades from portfolio CSV files into Supabase.

Loads per-trade results from `backtesting_engine/results/` CSVs into the
`trades` table (source='backtest') and seeds `research_runs` with aggregate
metrics from the CSV metadata header.

Only FULL-split CSVs are inserted as trade rows (IS/OOS are subsets of FULL
and would violate the dedup index). IS/OOS CSVs create research_runs entries
with aggregate metrics only.

Usage:
    python3 backfill_backtest_trades.py                     # backfill latest
    python3 backfill_backtest_trades.py --dry-run            # show counts only
    python3 backfill_backtest_trades.py --timestamp 20260312_015635  # specific run
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_ET = ZoneInfo("America/New_York")

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "backtesting_engine" / "results"

# Strategy ID -> instrument mapping
STRATEGY_INSTRUMENT = {
    "MNQ_V15": "MNQ",
    "MNQ_VSCALPB": "MNQ",
    "MNQ_VSCALPC": "MNQ",
    "MES_V2": "MES",
    "MNQ_V11": "MNQ",
}


def parse_csv_metadata(filepath: Path) -> dict:
    """Parse the JSON metadata from line 2 of a backtest CSV."""
    with open(filepath) as f:
        next(f)  # skip line 1 (comment header)
        line2 = next(f)
    if line2.startswith("# "):
        return json.loads(line2[2:])
    return {}


def compute_param_hash(strategy: str, params: dict, split: str, data_range: str) -> str:
    """Compute a unique hash for a research run."""
    canonical = json.dumps({
        "strategy": strategy,
        "params": params,
        "split": split,
        "data_range": data_range,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_aggregate_metrics(trades: list[dict]) -> dict:
    """Compute aggregate metrics from a list of trade rows."""
    if not trades:
        return {}

    pnls = [t["pnl_net"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    # Profit factor
    pf = gross_profit / gross_loss if gross_loss > 0 else None

    # Win rate
    wr = (len(wins) / len(pnls) * 100) if pnls else 0

    # Max drawdown (from equity curve)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = cumulative - peak
        if dd < max_dd:
            max_dd = dd

    return {
        "trade_count": len(trades),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 3) if pf else None,
        "max_drawdown": round(max_dd, 2),
    }


def find_latest_csvs(timestamp_filter: str = None) -> dict:
    """Find the latest set of backtest CSVs, grouped by strategy.

    Returns: {strategy_id: {split: filepath}}
    """
    all_csvs = sorted(RESULTS_DIR.glob("*.csv"))

    # Group by strategy and split, keeping only the latest timestamp
    results = {}
    for f in all_csvs:
        meta = parse_csv_metadata(f)
        if not meta:
            continue

        strategy = meta.get("strategy", "")
        split = meta.get("split", "")
        if not strategy or not split:
            continue

        # If timestamp filter specified, only match those files
        if timestamp_filter and timestamp_filter not in f.name:
            continue

        if strategy not in results:
            results[strategy] = {}

        # Later files in sorted order overwrite earlier ones (latest wins)
        results[strategy][split] = f

    return results


def load_csv_trades(filepath: Path, meta: dict) -> list[dict]:
    """Load trade rows from a backtest CSV and convert to Supabase format.

    pnl_dollar in backtest CSVs is already NET (commission deducted by save_results.py).
    """
    strategy_id = meta["strategy"]
    instrument = STRATEGY_INSTRUMENT.get(strategy_id, "MNQ")
    commission_per_side = meta.get("commission_per_side", 0.52)
    qty = meta.get("qty", 1)
    commission_rt = round(commission_per_side * 2 * qty, 2)  # roundtrip × qty

    rows = []
    with open(filepath) as f:
        # Skip comment lines
        reader = csv.DictReader(
            (line for line in f if not line.startswith("#"))
        )
        for r in reader:
            entry_time = r["entry_time"]
            exit_time = r["exit_time"]
            trade_date = r["trade_date"]
            pnl_net = float(r["pnl_dollar"])  # already net

            row = {
                "strategy_id": strategy_id,
                "instrument": instrument,
                "side": r["side"].lower(),
                "entry_price": float(r["entry_price"]),
                "exit_price": float(r["exit_price"]),
                "entry_time": entry_time,
                "exit_time": exit_time if exit_time else None,
                "pts": float(r["pts"]),
                "pnl_net": round(pnl_net, 2),
                "commission": commission_rt,
                "exit_reason": r["exit_reason"],
                "bars_held": int(r["bars_held"]) if r.get("bars_held") else 0,
                "qty": qty,
                "is_partial": False,
                "trade_date": trade_date,
                "source": "backtest",
            }

            # MFE/MAE (always present in backtest CSVs)
            mfe = float(r.get("mfe_pts", 0))
            mae = float(r.get("mae_pts", 0))
            if mfe is not None:
                row["mfe_pts"] = round(mfe, 2)
            if mae is not None:
                row["mae_pts"] = round(mae, 2)

            rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show counts without writing")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Filter CSVs by timestamp (e.g. 20260312_015635)")
    args = parser.parse_args()

    client = None
    if not args.dry_run:
        from engine.db import get_client
        client = get_client()
        if client is None:
            print("ERROR: Supabase not configured. Set SUPABASE_URL + SUPABASE_SERVICE_KEY in .env")
            sys.exit(1)

    print("=" * 60)
    print("BACKFILL BACKTEST TRADES")
    print("=" * 60)

    # Find CSVs
    grouped = find_latest_csvs(args.timestamp)
    if not grouped:
        print("  No backtest CSVs found.")
        return

    for strategy_id in sorted(grouped.keys()):
        splits = grouped[strategy_id]
        print(f"\n  {strategy_id}:")
        for split, filepath in sorted(splits.items()):
            print(f"    {split}: {filepath.name}")

    # --- Process each strategy ---
    total_trades_inserted = 0
    total_runs_inserted = 0

    for strategy_id in sorted(grouped.keys()):
        splits = grouped[strategy_id]
        print(f"\n{'─' * 40}")
        print(f"  {strategy_id}")
        print(f"{'─' * 40}")

        for split in ["FULL", "IS", "OOS"]:
            filepath = splits.get(split)
            if not filepath:
                print(f"  [{split}] No CSV found, skipping")
                continue

            meta = parse_csv_metadata(filepath)
            print(f"\n  [{split}] {filepath.name}")
            print(f"    Trades: {meta.get('total_trades', '?')}, "
                  f"PnL: ${meta.get('total_pnl', '?')}")

            # --- Create research_run ---
            split_lower = split.lower()
            param_hash = compute_param_hash(
                meta["strategy"], meta["params"], split_lower, meta["data_range"]
            )

            # Load trades for aggregate metrics (and for FULL, for insertion)
            trades = load_csv_trades(filepath, meta)
            agg = compute_aggregate_metrics(trades)

            run_row = {
                "strategy_id": strategy_id,
                "run_name": f"{strategy_id}_{split}_{meta['data_range']}",
                "script": meta.get("script", "run_and_save_portfolio.py"),
                "data_range": meta["data_range"].replace("_", " "),
                "split": split_lower,
                "params": meta["params"],
                "param_hash": param_hash,
                "trade_count": agg.get("trade_count"),
                "total_pnl": agg.get("total_pnl"),
                "win_rate": agg.get("win_rate"),
                "profit_factor": agg.get("profit_factor"),
                "max_drawdown": agg.get("max_drawdown"),
                "notes": f"Backfilled from {filepath.name}",
            }

            print(f"    WR: {agg.get('win_rate')}%, PF: {agg.get('profit_factor')}, "
                  f"MaxDD: ${agg.get('max_drawdown')}")

            if args.dry_run:
                print(f"    [DRY RUN] Would create research_run (hash: {param_hash[:16]}...)")
                if split == "FULL":
                    print(f"    [DRY RUN] Would insert {len(trades)} trades")
                continue

            # Upsert research_run (ON CONFLICT on param_hash)
            result = (client.table("research_runs")
                      .upsert(run_row, on_conflict="param_hash",
                              ignore_duplicates=True)
                      .execute())
            if result.data:
                run_id = result.data[0]["id"]
                print(f"    research_run: {run_id[:8]}...")
                total_runs_inserted += 1
            else:
                # Fetch existing run_id for linking trades
                existing = (client.table("research_runs")
                            .select("id")
                            .eq("param_hash", param_hash)
                            .execute())
                run_id = existing.data[0]["id"] if existing.data else None
                print(f"    research_run: exists (hash match)")

            # --- Insert FULL trades only (IS/OOS are subsets) ---
            if split != "FULL":
                continue

            # Add research_run_id to each trade
            for t in trades:
                if run_id:
                    t["research_run_id"] = run_id

            print(f"    Inserting {len(trades)} trades...")
            batch_size = 200
            inserted = 0
            for i in range(0, len(trades), batch_size):
                batch = trades[i:i + batch_size]
                result = (client.table("trades")
                          .upsert(batch,
                                  on_conflict="strategy_id,entry_time,is_partial,source",
                                  ignore_duplicates=True)
                          .execute())
                count = len(result.data) if result.data else 0
                inserted += count
                print(f"      Batch {i // batch_size + 1}: {count} rows")

            total_trades_inserted += inserted

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Research runs: {total_runs_inserted}")
    print(f"  Trades inserted: {total_trades_inserted}")

    if not args.dry_run:
        # Refresh materialized views
        print("\nRefreshing materialized views...")
        try:
            client.rpc("refresh_trading_views").execute()
            print("  Views refreshed.")
        except Exception as e:
            print(f"  WARNING: View refresh failed: {e}")
            print("  Run manually: SELECT refresh_trading_views();")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
