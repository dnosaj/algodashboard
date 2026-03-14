"""CLI entry point for the Counterfactual Trade Engine.

Usage:
    python -m agents.counterfactual.cli                  # Process all unfilled signals
    python -m agents.counterfactual.cli --start 2026-03-06 --end 2026-03-13
    python -m agents.counterfactual.cli --dry-run        # Compute, print, don't write
    python -m agents.counterfactual.cli --force           # Recompute existing results
    python -m agents.counterfactual.cli --cooldown-override 15 --force
    python -m agents.counterfactual.cli -v                # Verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure live_trading/ is on sys.path so engine.config/engine.db are importable
# regardless of CWD (critical for launchd which may run from / or ~)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from agents.counterfactual.engine import CounterfactualEngine


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Trade Engine")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't write to DB")
    parser.add_argument("--force", action="store_true", help="Clear and recompute existing results")
    parser.add_argument("--cooldown-override", type=int, help="Override cooldown bars (for research)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose per-signal output")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(__file__).resolve().parents[2] / "logs" / "counterfactual.log",
                mode="a",
            ),
        ],
    )

    t0 = time.time()
    engine = CounterfactualEngine(
        dry_run=args.dry_run,
        force=args.force,
        cooldown_override=args.cooldown_override,
        verbose=args.verbose,
    )
    try:
        results = engine.run(start_date=args.start, end_date=args.end)
    except Exception as e:
        logging.getLogger(__name__).error(f"[CF] Engine error: {e}", exc_info=True)
        sys.exit(1)
    elapsed = time.time() - t0

    if results is None:
        print(f"Engine returned None — likely a configuration error ({elapsed:.1f}s)")
        sys.exit(1)

    # Summary
    if results:
        winners = sum(1 for r in results if r.get("cf_pnl_pts", 0) > 0)
        losers = sum(1 for r in results if r.get("cf_pnl_pts", 0) <= 0 and r.get("cf_exit_reason") != "COOLDOWN_SUPPRESSED")
        suppressed = sum(1 for r in results if r.get("cf_exit_reason") == "COOLDOWN_SUPPRESSED")
        total_pnl = sum(r.get("cf_pnl_pts", 0) for r in results)
        print(f"\n{'='*50}")
        print(f"Counterfactual Engine {'(DRY RUN) ' if args.dry_run else ''}Summary")
        print(f"{'='*50}")
        print(f"Signals processed: {len(results)}")
        print(f"  Winners:    {winners}")
        print(f"  Losers:     {losers}")
        print(f"  CD-suppressed: {suppressed}")
        print(f"Total cf P&L: {total_pnl:+.2f} pts")
        print(f"Elapsed: {elapsed:.1f}s")
    else:
        print(f"No signals to process ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
