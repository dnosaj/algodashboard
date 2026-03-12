#!/usr/bin/env python3
"""
CLI for running the Digest Agent.

Usage:
    # EOD digest for today (dry run — no Supabase write)
    python -m agents.digest.cli --mode eod --dry-run

    # EOD digest for a specific date
    python -m agents.digest.cli --mode eod --date 2026-03-12

    # Morning briefing
    python -m agents.digest.cli --mode morning

    # Use a different model
    python -m agents.digest.cli --mode eod --model claude-opus-4-20250514

Run from live_trading/:
    cd live_trading && python -m agents.digest.cli --mode eod --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import date


def main():
    parser = argparse.ArgumentParser(
        description="Run the Digest Agent (EOD or Morning mode)"
    )
    parser.add_argument(
        "--mode",
        choices=["eod", "morning"],
        required=True,
        help="Digest mode: 'eod' for end-of-day, 'morning' for briefing",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Target date (YYYY-MM-DD). Default: today (ET)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model (e.g., claude-opus-4-20250514)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full agent loop but skip Supabase save (writes local markdown only)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import here so --help works without dependencies installed
    from .agent import DigestAgent

    agent = DigestAgent(
        mode=args.mode,
        target_date=args.date,
        model=args.model,
        dry_run=args.dry_run,
    )

    print(f"\n{'='*60}")
    print(f"  Digest Agent — {args.mode.upper()} — {agent.target_date}")
    print(f"  Model: {agent.model}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    result = agent.run()

    # Summary
    print(f"\n{'='*60}")
    print(f"  Result: {result.get('status', 'unknown')}")
    print(f"  Turns: {result.get('turns', '?')}")
    print(f"  Tool calls: {result.get('tool_calls', '?')}")
    print(f"  Tokens: {result.get('tokens_in', '?')} in / {result.get('tokens_out', '?')} out")
    if result.get("duration_sec") is not None:
        print(f"  Duration: {result['duration_sec']}s")
    if result.get("cost_usd") is not None:
        print(f"  Est. cost: ${result['cost_usd']}")
    if result.get("digest_saved") is not None:
        print(f"  Digest saved: {result['digest_saved']}")
    if result.get("reason"):
        print(f"  Reason: {result['reason']}")
    if result.get("message"):
        print(f"  Message: {result['message'][:200]}")
    print(f"{'='*60}\n")

    status = result.get("status")
    if status == "error":
        sys.exit(1)
    elif status == "completed_no_digest":
        print("  WARNING: Agent completed but save_digest was never called.")
        sys.exit(2)
    elif status == "token_budget_exceeded":
        print("  WARNING: Agent stopped due to token budget limit.")
        sys.exit(2)
    elif status == "max_turns":
        print("  WARNING: Agent hit max turns without finishing.")
        sys.exit(2)


if __name__ == "__main__":
    main()
