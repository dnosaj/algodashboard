"""
Entry point for the NQ live trading engine.

Usage:
    python run.py                              # Default: mock (paper), MNQ+MES
    python run.py --broker tastytrade          # tastytrade sandbox (reads .env)
    python run.py --broker tastytrade --live   # tastytrade PRODUCTION
    python run.py --instruments MNQ            # MNQ only
    python run.py --port 9000                  # Custom API port
"""

import argparse
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

from engine.config import (
    DEFAULT_CONFIG,
    EngineConfig,
    MES_V2,
    MNQ_V15,
    MNQ_VSCALPB,
    SafetyConfig,
    StrategyConfig,
    TastytradeConfig,
)


# Each instrument maps to a list of strategy configs (multi-strategy per instrument)
INSTRUMENT_CONFIGS: dict[str, list[StrategyConfig]] = {
    "MNQ": [MNQ_V15, MNQ_VSCALPB],
    "MES": [MES_V2],
}


def load_env() -> None:
    """Load .env file into os.environ (simple key=value, no quotes needed)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and value:
                os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NQ Trading Engine -- SM+RSI strategy on MNQ/MES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--broker",
        type=str,
        default="mock",
        choices=["mock", "tastytrade"],
        help="Broker: 'mock' for paper, 'tastytrade' for real/sandbox",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use production environment (default is sandbox/paper)",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default="MNQ,MES",
        help="Comma-separated instrument list (e.g., MNQ,MES or MNQ)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EngineConfig:
    """Build EngineConfig from CLI arguments and environment."""
    paper_mode = not args.live

    # Parse instruments and flatten all strategies for selected instruments
    instrument_names = [s.strip().upper() for s in args.instruments.split(",")]
    strategies = []
    for name in instrument_names:
        if name not in INSTRUMENT_CONFIGS:
            print(f"ERROR: Unknown instrument '{name}'. Available: {list(INSTRUMENT_CONFIGS.keys())}")
            sys.exit(1)
        strategies.extend(INSTRUMENT_CONFIGS[name])

    if not strategies:
        print("ERROR: No instruments specified.")
        sys.exit(1)

    # Build tastytrade config from env vars
    # NOTE: is_sandbox controls which API endpoint (prod vs cert).
    # Paper mode (real data + mock fills) still uses the PRODUCTION API.
    # Only set is_sandbox=True if you have sandbox/cert credentials.
    tt_config = TastytradeConfig(
        client_secret=os.environ.get("TT_CLIENT_SECRET", ""),
        refresh_token=os.environ.get("TT_REFRESH_TOKEN", ""),
        account_number=os.environ.get("TT_ACCOUNT_NUMBER", ""),
        is_sandbox=False,  # Always use production API (paper mode uses mock fills, not sandbox)
    )

    config = replace(
        DEFAULT_CONFIG,
        strategies=strategies,
        safety=replace(DEFAULT_CONFIG.safety, paper_mode=paper_mode),
        tastytrade=tt_config,
        broker=args.broker,
        api_port=args.port,
    )

    return config


def main() -> None:
    load_env()
    args = parse_args()
    config = build_config(args)

    print(f"NQ Trading Engine")
    print(f"  Broker:      {config.broker}")
    if config.broker == "tastytrade":
        env = "SANDBOX" if config.tastytrade.is_sandbox else "PRODUCTION"
        acct = config.tastytrade.account_number or "(auto-detect)"
        has_secret = "yes" if config.tastytrade.client_secret else "MISSING"
        has_token = "yes" if config.tastytrade.refresh_token else "MISSING"
        print(f"  Environment: {env}")
        print(f"  Account:     {acct}")
        print(f"  Credentials: secret={has_secret}, token={has_token}")

        if not config.tastytrade.client_secret or not config.tastytrade.refresh_token:
            print()
            print("ERROR: tastytrade credentials not found.")
            print("       Fill in TT_CLIENT_SECRET and TT_REFRESH_TOKEN in .env")
            sys.exit(1)
    else:
        print(f"  Mode:        PAPER (mock)")
    print(f"  Strategies:  {[(s.strategy_id or s.instrument) for s in config.strategies]}")
    print(f"  API port:    {config.api_port}")
    print()

    if not config.safety.paper_mode:
        print("=" * 50)
        print("  WARNING: PRODUCTION MODE")
        print("  Real orders will be placed on tastytrade.")
        print("=" * 50)
        print()

    from engine.runner import run
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
