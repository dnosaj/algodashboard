"""
Seed strategy_configs table with current active strategy configurations.

Reads StrategyConfig dataclasses from config.py, serializes to JSONB,
computes config_hash (SHA256 of canonical JSON), and upserts to Supabase.

Idempotent: re-running skips configs with matching hash, deactivates old
configs when hash changes, and inserts new active configs.

Usage:
    python3 seed_strategy_configs.py          # seed all active strategies
    python3 seed_strategy_configs.py --dry-run # show what would be inserted
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add live_trading to path and load .env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from engine.config import MNQ_V15, MNQ_VSCALPB, MNQ_VSCALPC, MES_V2, MNQ_V11

# --- Backtest benchmarks (from FULL backtests with CURRENT config) ---
# Source: run_and_save_portfolio.py --split (Mar 12, 2026)
# All strategies from portfolio backtest CSVs (including vScalpC partial exit)
BENCHMARKS = {
    "MNQ_V15": {
        "backtest_expected_wr": 83.0,
        "backtest_expected_pf": 1.338,
        "backtest_expected_sharpe": 1.799,
        "backtest_max_dd": -628.24,
    },
    "MNQ_VSCALPB": {
        "backtest_expected_wr": 73.2,
        "backtest_expected_pf": 1.422,
        "backtest_expected_sharpe": 2.102,
        "backtest_max_dd": -280.78,
    },
    "MNQ_VSCALPC": {
        "backtest_expected_wr": 77.0,
        "backtest_expected_pf": 1.425,
        "backtest_expected_sharpe": 2.139,
        "backtest_max_dd": -1078.90,
    },
    "MES_V2": {
        "backtest_expected_wr": 56.0,
        "backtest_expected_pf": 1.313,
        "backtest_expected_sharpe": 1.621,
        "backtest_max_dd": -1168.75,
    },
    # Shelved strategies (inactive, for historical trade attribution)
    "MNQ_V11": {
        "backtest_expected_wr": None,
        "backtest_expected_pf": None,
        "backtest_expected_sharpe": None,
        "backtest_max_dd": None,
    },
}

# Active strategies to seed
ACTIVE_CONFIGS = {
    "MNQ_V15": MNQ_V15,
    "MNQ_VSCALPB": MNQ_VSCALPB,
    "MNQ_VSCALPC": MNQ_VSCALPC,
    "MES_V2": MES_V2,
}

# Shelved strategies (active=false)
SHELVED_CONFIGS = {
    "MNQ_V11": MNQ_V11,
}


def config_to_json(config) -> dict:
    """Convert StrategyConfig to a canonical JSON-serializable dict."""
    d = asdict(config)
    # Remove strategy_id from config body (it's the lookup key, not config content)
    d.pop("strategy_id", None)
    return d


def compute_hash(config_dict: dict) -> str:
    """Compute SHA256 hash of canonical JSON representation."""
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be inserted without writing")
    args = parser.parse_args()

    from engine.db import get_client
    client = get_client()
    if client is None:
        print("ERROR: Supabase not configured. Set SUPABASE_URL + SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)

    print("=" * 60)
    print("SEED STRATEGY CONFIGS")
    print("=" * 60)

    all_configs = []

    # Active strategies
    for sid, config in ACTIVE_CONFIGS.items():
        config_dict = config_to_json(config)
        config_hash = compute_hash(config_dict)
        benchmarks = BENCHMARKS.get(sid, {})

        row = {
            "strategy_id": sid,
            "config": config_dict,
            "config_hash": config_hash,
            "active": True,
            **benchmarks,
            "notes": f"Seeded from config.py (Mar 12, 2026)",
        }
        all_configs.append(row)

    # Shelved strategies (inactive)
    for sid, config in SHELVED_CONFIGS.items():
        config_dict = config_to_json(config)
        config_hash = compute_hash(config_dict)
        benchmarks = BENCHMARKS.get(sid, {})

        row = {
            "strategy_id": sid,
            "config": config_dict,
            "config_hash": config_hash,
            "active": False,
            **benchmarks,
            "notes": f"Shelved strategy — seeded for historical trade attribution",
        }
        all_configs.append(row)

    for row in all_configs:
        sid = row["strategy_id"]
        active = row["active"]
        print(f"\n  {sid} (active={active})")
        print(f"    hash: {row['config_hash'][:16]}...")
        print(f"    WR: {row.get('backtest_expected_wr')}, PF: {row.get('backtest_expected_pf')}, "
              f"Sharpe: {row.get('backtest_expected_sharpe')}, MaxDD: {row.get('backtest_max_dd')}")

        if args.dry_run:
            print(f"    [DRY RUN] Would insert/update")
            continue

        # Check if active config with same hash exists
        existing = (client.table("strategy_configs")
                    .select("id, config_hash, active")
                    .eq("strategy_id", sid)
                    .eq("active", active)
                    .execute())

        if existing.data:
            existing_row = existing.data[0]
            if existing_row["config_hash"] == row["config_hash"]:
                print(f"    SKIP — identical config already exists")
                continue
            else:
                # Deactivate old config
                print(f"    Deactivating old config (hash: {existing_row['config_hash'][:16]}...)")
                (client.table("strategy_configs")
                 .update({"active": False})
                 .eq("id", existing_row["id"])
                 .execute())

        # Insert new config
        result = client.table("strategy_configs").insert(row).execute()
        if result.data:
            print(f"    INSERTED (id: {result.data[0]['id'][:8]}...)")
        else:
            print(f"    ERROR inserting: {result}")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
