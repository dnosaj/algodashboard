#!/usr/bin/env python3
"""
Backfill config_id on all trades that have a matching strategy_configs entry.

Links each trade to the strategy config version that produced it.
Uses the active config for each strategy_id (there should be exactly one active config per strategy).

Idempotent: skips trades that already have config_id set.
"""

import os
import sys

# Load .env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

from supabase import create_client

url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_SERVICE_KEY')

if not url or not key:
    print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    sys.exit(1)

sb = create_client(url, key)


def main():
    dry_run = '--dry-run' in sys.argv

    # 1. Load strategy_configs mapping: strategy_id -> config UUID
    configs = sb.table('strategy_configs').select('id,strategy_id,active').execute()
    config_map = {}
    for c in configs.data:
        sid = c['strategy_id']
        # Prefer active config; fall back to any config
        if sid not in config_map or c.get('active'):
            config_map[sid] = c['id']

    print(f"Strategy config mapping ({len(config_map)} strategies):")
    for sid, cid in sorted(config_map.items()):
        print(f"  {sid} -> {cid}")

    # 2. Find trades with NULL config_id, grouped by strategy_id
    # Supabase REST API has a 1000-row default limit, so paginate
    total_updated = 0
    for sid, config_id in sorted(config_map.items()):
        # Count trades needing update
        count_res = (
            sb.table('trades')
            .select('id', count='exact')
            .eq('strategy_id', sid)
            .is_('config_id', 'null')
            .execute()
        )
        count = count_res.count or 0

        if count == 0:
            print(f"\n  {sid}: 0 trades need update (all populated)")
            continue

        print(f"\n  {sid}: {count} trades need config_id = {config_id}")

        if dry_run:
            total_updated += count
            continue

        # Update in batches (Supabase REST handles this fine)
        # Fetch IDs first, then update in batches of 500
        offset = 0
        batch_size = 500
        updated = 0

        while offset < count:
            batch = (
                sb.table('trades')
                .select('id')
                .eq('strategy_id', sid)
                .is_('config_id', 'null')
                .limit(batch_size)
                .execute()
            )

            if not batch.data:
                break

            ids = [row['id'] for row in batch.data]

            # Update batch
            for trade_id in ids:
                sb.table('trades').update({'config_id': config_id}).eq('id', trade_id).execute()
                updated += 1

            offset += len(ids)
            print(f"    Updated {updated}/{count}...")

        total_updated += updated

    print(f"\n{'[DRY RUN] Would update' if dry_run else 'Updated'}: {total_updated} trades total")

    # 3. Verify
    if not dry_run:
        still_null = (
            sb.table('trades')
            .select('id', count='exact')
            .is_('config_id', 'null')
            .execute()
        )
        has_config = (
            sb.table('trades')
            .select('id', count='exact')
            .not_.is_('config_id', 'null')
            .execute()
        )
        print(f"\nVerification:")
        print(f"  With config_id: {has_config.count}")
        print(f"  Without config_id: {still_null.count}")
        if still_null.count > 0:
            # Check which strategies are missing configs
            orphans = (
                sb.table('trades')
                .select('strategy_id')
                .is_('config_id', 'null')
                .limit(10)
                .execute()
            )
            orphan_sids = set(r['strategy_id'] for r in orphans.data)
            print(f"  Missing config for: {orphan_sids}")


if __name__ == '__main__':
    main()
