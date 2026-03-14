"""
Backfill paper trades from session JSON files into Supabase.

Handles 5 session JSON schema variants, normalizes fields, deduplicates
overlapping multi-day sessions, and batch inserts with ON CONFLICT DO NOTHING.

Also backfills blocked signals from the CSV log.

Usage:
    python3 backfill_paper_trades.py              # backfill all
    python3 backfill_paper_trades.py --dry-run     # show counts without writing
    python3 backfill_paper_trades.py --cutoff 2026-02-23  # skip sessions before date
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_ET = ZoneInfo("America/New_York")

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"
BLOCKED_CSV = Path(__file__).resolve().parent.parent / "logs" / "blocked_signals.csv"

# Commission per side by instrument (for deducting from gross session pnl)
COMMISSION_PER_SIDE = {
    "MNQ": 0.52,
    "MES": 1.25,
}

# Gate type extraction from reason strings
GATE_PREFIXES = {
    "VIX death zone": "vix_death_zone",
    "Leledc exhaustion": "leledc",
    "Prior-day level": "prior_day_level",
    "Prior-day ATR": "prior_day_atr",
    "ADR directional": "adr_directional",
    "Strategy paused": "strategy_paused",
    "Daily P&L limit": "daily_pnl_limit",
    "Global daily loss": "global_daily_loss",
    "Consecutive losses": "consecutive_losses",
    "Max position": "max_position",
    "Engine halted": "engine_halted",
}


def parse_entry_time(entry_time_str: str) -> datetime:
    """Parse entry_time string, ensuring UTC timezone."""
    dt = datetime.fromisoformat(entry_time_str)
    if dt.tzinfo is None:
        # Bare ISO timestamps in early sessions are UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute_trade_date(entry_time: datetime) -> str:
    """Compute ET trade date from UTC entry_time."""
    return entry_time.astimezone(_ET).date().isoformat()


def is_multi_day_session(filename: str) -> bool:
    """Check if a session file covers multiple days (e.g. session_2026-02-13_to_2026-02-18.json)."""
    return "_to_" in filename


def extract_gate_type(reason: str) -> str:
    """Extract normalized gate_type from a reason string."""
    for prefix, gate_type in GATE_PREFIXES.items():
        if reason.startswith(prefix):
            return gate_type
    return "unknown"


def load_session_trades(filepath: Path, cutoff_date: date = None) -> list[dict]:
    """Load and normalize trades from a session JSON file.

    Returns list of Supabase-ready trade dicts.
    """
    with open(filepath) as f:
        data = json.load(f)

    trades = data.get("trades", [])
    if not trades:
        return []

    rows = []
    for t in trades:
        entry_time = parse_entry_time(t["entry_time"])
        trade_date = compute_trade_date(entry_time)

        # Skip trades before cutoff
        if cutoff_date and date.fromisoformat(trade_date) < cutoff_date:
            continue

        # Normalize side: UPPERCASE -> lowercase
        side = t["side"].lower()

        # qty defaults to 1 for early schemas
        qty = t.get("qty", 1)
        if qty is None:
            qty = 1

        # is_partial: MUST be bool false, never None (dedup index includes it)
        is_partial = bool(t.get("is_partial", False))

        # Compute commission and net P&L
        instrument = t["instrument"]
        comm_per_side = COMMISSION_PER_SIDE.get(instrument, 0.52)
        commission = comm_per_side * 2 * qty
        pnl_gross = t["pnl"]
        pnl_net = round(pnl_gross - commission, 2)

        # Parse exit_time
        exit_time = None
        if t.get("exit_time"):
            exit_time = parse_entry_time(t["exit_time"])

        # MFE/MAE (only in early schemas, renamed mfe->mfe_pts)
        mfe_pts = t.get("mfe")
        mae_pts = t.get("mae")

        # Synthesize trade_group_id for partial exit strategies
        trade_group_id = None
        if is_partial or t.get("strategy_id") in ("MNQ_VSCALPC", "MES_V2"):
            trade_group_id = f"{t['strategy_id']}_{entry_time.isoformat()}"

        row = {
            "strategy_id": t["strategy_id"],
            "instrument": instrument,
            "side": side,
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat() if exit_time else None,
            "pts": t["pts"],
            "pnl_net": pnl_net,
            "commission": round(commission, 2),
            "exit_reason": t["exit_reason"],
            "bars_held": t.get("bars_held", 0),
            "qty": qty,
            "is_partial": is_partial,
            "trade_date": trade_date,
            "source": "paper",
        }

        # Optional fields (only add if not None, let DB defaults apply)
        if mfe_pts is not None:
            row["mfe_pts"] = round(mfe_pts, 2)
        if mae_pts is not None:
            row["mae_pts"] = round(mae_pts, 2)
        if trade_group_id:
            row["trade_group_id"] = trade_group_id

        rows.append(row)

    return rows


def load_blocked_signals(filepath: Path) -> list[dict]:
    """Load and normalize blocked signals from CSV."""
    if not filepath.exists():
        return []

    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for r in reader:
            signal_time = parse_entry_time(r["timestamp"])
            signal_date = signal_time.astimezone(_ET).date().isoformat()

            row = {
                "signal_time": signal_time.isoformat(),
                "strategy_id": r["strategy_id"],
                "instrument": r["instrument"],
                "side": r["side"].lower(),
                "price": float(r["price"]),
                "sm_value": float(r["sm_value"]) if r.get("sm_value") else None,
                "rsi_value": float(r["rsi_value"]) if r.get("rsi_value") else None,
                "gate_type": extract_gate_type(r["reason"]),
                "reason": r["reason"],
                "signal_date": signal_date,
                "source": "paper",
            }
            # Strip None values
            row = {k: v for k, v in row.items() if v is not None}
            rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Skip sessions before this date (YYYY-MM-DD)")
    args = parser.parse_args()

    cutoff_date = None
    if args.cutoff:
        cutoff_date = date.fromisoformat(args.cutoff)

    client = None
    if not args.dry_run:
        from engine.db import get_client
        client = get_client()
        if client is None:
            print("ERROR: Supabase not configured. Set SUPABASE_URL + SUPABASE_SERVICE_KEY in .env")
            sys.exit(1)

    print("=" * 60)
    print("BACKFILL PAPER TRADES")
    print("=" * 60)

    # --- 1. Collect session files ---
    # Single-day files first (higher priority), then multi-day
    session_files = sorted(SESSIONS_DIR.glob("session_*.json"))
    single_day = [f for f in session_files if not is_multi_day_session(f.name)]
    multi_day = [f for f in session_files if is_multi_day_session(f.name)]

    # Track which (strategy_id, entry_time) we've seen to avoid loading dupes
    seen_keys = set()
    all_trades = []
    skipped_dupes = 0

    # Process single-day files first (richer schema, preferred)
    for filepath in single_day:
        trades = load_session_trades(filepath, cutoff_date)
        new_trades = []
        for t in trades:
            key = (t["strategy_id"], t["entry_time"], t["is_partial"])
            if key not in seen_keys:
                seen_keys.add(key)
                new_trades.append(t)
            else:
                skipped_dupes += 1
        all_trades.extend(new_trades)
        if new_trades:
            print(f"  {filepath.name}: {len(new_trades)} trades")

    # Process multi-day files (only trades not already seen)
    for filepath in multi_day:
        trades = load_session_trades(filepath, cutoff_date)
        new_trades = []
        for t in trades:
            key = (t["strategy_id"], t["entry_time"], t["is_partial"])
            if key not in seen_keys:
                seen_keys.add(key)
                new_trades.append(t)
            else:
                skipped_dupes += 1
        all_trades.extend(new_trades)
        if new_trades:
            print(f"  {filepath.name}: {len(new_trades)} trades (multi-day, new only)")

    # Exclude today's session (avoid overwriting live-logged enriched data)
    today = datetime.now(_ET).date().isoformat()
    before_count = len(all_trades)
    all_trades = [t for t in all_trades if t["trade_date"] != today]
    today_skipped = before_count - len(all_trades)

    print(f"\n  Total: {len(all_trades)} trades")
    print(f"  Deduped (overlap): {skipped_dupes}")
    if today_skipped:
        print(f"  Skipped (today): {today_skipped}")

    # Strategy breakdown
    by_strategy = {}
    for t in all_trades:
        sid = t["strategy_id"]
        by_strategy[sid] = by_strategy.get(sid, 0) + 1
    for sid, count in sorted(by_strategy.items()):
        print(f"    {sid}: {count} trades")

    # --- 2. Blocked signals ---
    blocked = load_blocked_signals(BLOCKED_CSV)
    print(f"\n  Blocked signals: {len(blocked)} rows")

    if args.dry_run:
        print("\n  [DRY RUN] No data written.")
        return

    # --- 3. Insert trades (ON CONFLICT DO NOTHING) ---
    print(f"\nInserting {len(all_trades)} trades...")
    batch_size = 200
    inserted = 0
    for i in range(0, len(all_trades), batch_size):
        batch = all_trades[i:i + batch_size]
        result = (client.table("trades")
                  .upsert(batch, on_conflict="strategy_id,entry_time,is_partial,source",
                           ignore_duplicates=True)
                  .execute())
        inserted += len(result.data) if result.data else 0
        print(f"  Batch {i // batch_size + 1}: {len(result.data) if result.data else 0} rows")

    print(f"  Total inserted/updated: {inserted}")

    # --- 4. Insert blocked signals ---
    if blocked:
        print(f"\nInserting {len(blocked)} blocked signals...")
        result = (client.table("blocked_signals")
                  .upsert(blocked,
                           on_conflict="signal_time,strategy_id,instrument,source",
                           ignore_duplicates=True)
                  .execute())
        print(f"  Inserted: {len(result.data) if result.data else 0}")

    # --- 5. Refresh materialized views ---
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
