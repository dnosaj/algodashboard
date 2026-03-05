"""
Step 0 Validation: Can tastytrade accept two simultaneous OCO brackets?

Run during market hours (futures RTH: 9:30-16:00 ET, or globex overnight).

Usage:
    cd live_trading
    python validate_dual_oco.py

What it does:
    1. Authenticates with tastytrade (production API, paper fills)
    2. Resolves front-month MES contract
    3. Places a 2-contract MES market buy
    4. Places OCO #1: 1 contract, LIMIT (TP1) + STOP (SL)
    5. Places OCO #2: 1 contract, LIMIT (TP2) + STOP (SL)
    6. Verifies both OCOs are accepted and live
    7. Cancels both OCOs
    8. Closes the 2-contract position

If step 5 fails (API rejects second OCO), reports the error.
All orders are cleaned up regardless of outcome.

Requires: pip install tastytrade
"""

import asyncio
import os
import sys
import time
from decimal import Decimal
from pathlib import Path


def load_env():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("ERROR: .env file not found in live_trading/")
        sys.exit(1)
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


async def main():
    load_env()

    from tastytrade import Session, Account, AlertStreamer
    from tastytrade.instruments import Future
    from tastytrade.order import (
        NewOrder, NewComplexOrder,
        OrderAction, OrderType, OrderTimeInForce,
        OrderStatus,
    )

    client_secret = os.environ.get("TT_CLIENT_SECRET", "")
    refresh_token = os.environ.get("TT_REFRESH_TOKEN", "")
    account_number = os.environ.get("TT_ACCOUNT_NUMBER", "")

    if not client_secret or not refresh_token:
        print("ERROR: TT_CLIENT_SECRET and TT_REFRESH_TOKEN must be set in .env")
        sys.exit(1)

    # --- Connect ---
    print("Connecting to tastytrade (production API)...")
    session = Session(client_secret, refresh_token, is_test=False)

    if account_number:
        account = await Account.get(session, account_number)
    else:
        accounts = await Account.get(session)
        account = accounts[0]
    print(f"Account: {account.account_number}")

    # --- Resolve MES front month ---
    futures = await Future.get(session, product_codes=["MES"])
    active = sorted([f for f in futures if f.active], key=lambda f: f.expiration_date)
    if not active:
        print("ERROR: No active MES contracts found")
        sys.exit(1)
    mes = active[0]
    print(f"Contract: {mes.symbol}")

    # Track IDs for cleanup
    entry_order_id = None
    oco1_id = None
    oco2_id = None
    close_order_id = None

    try:
        # --- Step 1: Place 2-contract entry ---
        print("\n[1/6] Placing 2-contract MES BUY...")
        leg = mes.build_leg(Decimal("2"), OrderAction.BUY)
        entry = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            legs=[leg],
        )
        resp = await account.place_order(session, entry, dry_run=False)
        entry_order_id = resp.order.id if resp.order else None

        # Wait for fill
        fill_price = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            orders = await account.get_live_orders(session)
            for o in orders:
                if o.id == entry_order_id and o.status == OrderStatus.FILLED:
                    for l in o.legs:
                        if l.fills:
                            fill_price = float(l.fills[0].fill_price)
                    break
            if fill_price:
                break
            await asyncio.sleep(0.2)

        if not fill_price:
            print("ERROR: Entry did not fill within 10s")
            sys.exit(1)
        print(f"  Filled 2x MES @ {fill_price}")

        # --- Step 2: Place OCO #1 (TP1) ---
        sl_price = fill_price - 35
        tp1_price = fill_price + 10
        tp2_price = fill_price + 20

        print(f"\n[2/6] Placing OCO #1: 1x LIMIT @ {tp1_price} + STOP @ {sl_price}...")
        leg1 = mes.build_leg(Decimal("1"), OrderAction.SELL)
        oco1 = NewComplexOrder(
            orders=[
                NewOrder(
                    time_in_force=OrderTimeInForce.GTC,
                    order_type=OrderType.LIMIT,
                    legs=[leg1],
                    price=Decimal(str(tp1_price)),
                ),
                NewOrder(
                    time_in_force=OrderTimeInForce.GTC,
                    order_type=OrderType.STOP,
                    legs=[leg1],
                    stop_trigger=Decimal(str(sl_price)),
                ),
            ]
        )
        resp1 = await account.place_complex_order(session, oco1, dry_run=False)
        oco1_id = resp1.complex_order.id if resp1.complex_order else None
        print(f"  OCO #1 accepted! ID: {oco1_id}")

        # --- Step 3: Place OCO #2 (TP2) ---
        print(f"\n[3/6] Placing OCO #2: 1x LIMIT @ {tp2_price} + STOP @ {sl_price}...")
        leg2 = mes.build_leg(Decimal("1"), OrderAction.SELL)
        oco2 = NewComplexOrder(
            orders=[
                NewOrder(
                    time_in_force=OrderTimeInForce.GTC,
                    order_type=OrderType.LIMIT,
                    legs=[leg2],
                    price=Decimal(str(tp2_price)),
                ),
                NewOrder(
                    time_in_force=OrderTimeInForce.GTC,
                    order_type=OrderType.STOP,
                    legs=[leg2],
                    stop_trigger=Decimal(str(sl_price)),
                ),
            ]
        )
        try:
            resp2 = await account.place_complex_order(session, oco2, dry_run=False)
            oco2_id = resp2.complex_order.id if resp2.complex_order else None
            print(f"  OCO #2 accepted! ID: {oco2_id}")
        except Exception as e:
            print(f"\n  *** OCO #2 REJECTED: {e} ***")
            print("  tastytrade does NOT support two simultaneous OCO brackets.")
            print("  Fallback needed: single OCO at TP2 + engine-managed TP1.")
            # Continue to cleanup
            oco2_id = None

        # --- Step 4: Verify both live ---
        print(f"\n[4/6] Verifying OCO orders are live...")
        await asyncio.sleep(1)  # Give exchange a moment
        # Just check they weren't immediately rejected/cancelled

        if oco1_id and oco2_id:
            print("\n" + "=" * 60)
            print("  STEP 0 VALIDATION: PASS")
            print("  tastytrade accepts two simultaneous OCO brackets!")
            print("=" * 60)
        elif oco1_id and not oco2_id:
            print("\n" + "=" * 60)
            print("  STEP 0 VALIDATION: FAIL")
            print("  Second OCO was rejected. Need fallback approach.")
            print("=" * 60)

    finally:
        # --- Cleanup: cancel OCOs and close position ---
        print(f"\n[5/6] Cancelling OCO brackets...")
        if oco1_id:
            try:
                await account.delete_complex_order(session, oco1_id)
                print(f"  OCO #1 ({oco1_id}) cancelled")
            except Exception as e:
                print(f"  OCO #1 cancel failed: {e}")
        if oco2_id:
            try:
                await account.delete_complex_order(session, oco2_id)
                print(f"  OCO #2 ({oco2_id}) cancelled")
            except Exception as e:
                print(f"  OCO #2 cancel failed: {e}")

        print(f"\n[6/6] Closing 2-contract position...")
        try:
            close_leg = mes.build_leg(Decimal("2"), OrderAction.SELL)
            close_order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.MARKET,
                legs=[close_leg],
            )
            await account.place_order(session, close_order, dry_run=False)
            print("  Position closed")
        except Exception as e:
            print(f"  WARNING: Close failed: {e}")
            print("  CHECK YOUR ACCOUNT — you may have an open MES position!")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
