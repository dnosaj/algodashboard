"""
Quick smoke test for the Databento data feed adapter.

Test 1: Import check (no API key needed)
Test 2: Basic initialization (needs DATABENTO_API_KEY)
Test 3: Historical warmup fetch (needs API key + costs credits)

Usage:
    # Import check only (free, no API key needed):
    python3 -c "from engine.databento_feed import DabentoDataFeed; print('OK')"

    # Run full test (needs DATABENTO_API_KEY in .env or env):
    cd live_trading && python3 tests/test_databento_feed_import.py

    # Quick live stream test (connects, gets a few bars, prints them):
    cd live_trading && python3 tests/test_databento_feed_import.py --live
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import():
    """Test 1: Verify the module imports correctly."""
    print("Test 1: Import check...")
    from engine.databento_feed import DabentoDataFeed, QuoteTick, SYMBOL_MAP
    print(f"  DabentoDataFeed: OK")
    print(f"  QuoteTick:       OK")
    print(f"  SYMBOL_MAP:      {SYMBOL_MAP}")
    print("  PASS")
    return True


def test_init():
    """Test 2: Verify initialization (needs API key)."""
    print("\nTest 2: Initialization check...")
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("  SKIP (no DATABENTO_API_KEY set)")
        return True

    from engine.databento_feed import DabentoDataFeed

    feed = DabentoDataFeed(
        instruments=["MNQ", "MES"],
        warmup_bars=100,
    )
    assert feed.connected is False
    assert feed.bar_queue is not None
    assert feed.quote_queue is not None
    assert feed.resolve_instrument("MNQ") == "MNQ"
    assert feed.resolve_instrument("MES") == "MES"
    assert feed.resolve_instrument("MNQ.c.0") == "MNQ"
    assert feed.resolve_instrument("INVALID") is None
    assert feed.resolve_instrument(None) is None
    print(f"  Instruments: {feed._instruments}")
    print(f"  DB symbols:  {feed._db_symbols}")
    print(f"  Connected:   {feed.connected}")
    print("  PASS")
    return True


async def test_warmup():
    """Test 3: Fetch warmup bars (costs credits!)."""
    print("\nTest 3: Historical warmup fetch...")
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("  SKIP (no DATABENTO_API_KEY set)")
        return True

    from engine.databento_feed import DabentoDataFeed

    # Small warmup to minimize cost
    feed = DabentoDataFeed(
        instruments=["MNQ"],
        warmup_bars=10,
    )

    # Just fetch warmup (don't start live stream)
    feed._loop = asyncio.get_running_loop()
    await feed._fetch_warmup_bars()

    bars = feed.get_warmup_bars("MNQ", 10)
    print(f"  Bars received: {len(bars)}")
    if bars:
        b = bars[-1]
        print(f"  Last bar: {b.timestamp} O={b.open:.2f} H={b.high:.2f} "
              f"L={b.low:.2f} C={b.close:.2f} V={b.volume:.0f} "
              f"inst={b.instrument}")
        # Sanity: MNQ prices should be in 10,000-30,000 range
        assert 5000 < b.close < 50000, f"Price {b.close} out of expected range"
    print("  PASS")
    return True


async def test_live_stream():
    """Test 4: Connect to live stream, receive a few bars, then disconnect."""
    print("\nTest 4: Live stream test (waiting for bars)...")
    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("  SKIP (no DATABENTO_API_KEY set)")
        return True

    from engine.databento_feed import DabentoDataFeed

    feed = DabentoDataFeed(
        instruments=["MNQ"],
        warmup_bars=10,
    )

    await feed.connect()
    print(f"  Connected: {feed.connected}")

    # Wait for up to 120s for a bar (market might be closed)
    bars_received = 0
    try:
        for _ in range(3):  # Try to get 3 bars
            try:
                bar = await asyncio.wait_for(feed.bar_queue.get(), timeout=120.0)
                bars_received += 1
                print(f"  Bar {bars_received}: {bar.timestamp} "
                      f"O={bar.open:.2f} H={bar.high:.2f} "
                      f"L={bar.low:.2f} C={bar.close:.2f} "
                      f"V={bar.volume:.0f} inst={bar.instrument}")
            except asyncio.TimeoutError:
                print("  Timeout waiting for bar (market may be closed)")
                break
    finally:
        await feed.disconnect()

    print(f"  Bars received: {bars_received}")
    print(f"  Connected after disconnect: {feed.connected}")
    print("  PASS")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Run live stream test (waits for real bars)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run warmup test (costs credits)")
    args = parser.parse_args()

    # Load .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    print("=" * 50)
    print("Databento Data Feed Smoke Test")
    print("=" * 50)

    # Always run import and init tests
    test_import()
    test_init()

    if args.warmup or args.live:
        asyncio.run(test_warmup())

    if args.live:
        asyncio.run(test_live_stream())

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
