"""
Tastytrade order management for the live trading engine.

Handles session authentication, order placement, bracket (stop loss) management,
and position tracking via the tastytrade REST API + AlertStreamer.

Order Type Specifications by Strategy
--------------------------------------

MNQ v11 (max_loss_pts=50):
  ENTRY:  Market order (BUY or SELL) -> immediate fill on CME
  STOP:   Resting STOP order at entry_price +/- 50 pts
          Executes on CME exchange instantly when price hits level
  EXIT:   Cancel resting STOP -> Market close (SM flip or EOD)
  STOP HIT: Exchange fills STOP, AlertStreamer notifies engine,
            strategy state updated via force_close()

MES v9.4 (max_loss_pts=0):
  ENTRY:  Market order (BUY or SELL) -> immediate fill on CME
  STOP:   None (no resting order placed)
  EXIT:   Market close (SM flip or EOD)

Why resting orders matter:
  - The backtest checks stop loss at bar boundaries (every 60s poll)
  - A resting STOP order fires intra-bar the instant price touches it
  - This is BETTER than the backtest -- no 60s delay on stop exits
  - No take-profit in current strategies; if added later, use OCO bracket

Contract symbols:
  - MNQ/MES use quarterly contracts: H(Mar), M(Jun), U(Sep), Z(Dec)
  - Auto-resolved on connect() by querying active futures
  - Example: Feb 2026 -> /MNQH6, /MESH6 (March 2026 expiry)

Requirements:
  pip install tastytrade
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import TastytradeConfig, StrategyConfig

logger = logging.getLogger(__name__)

try:
    from tastytrade import Session, Account, AlertStreamer
    from tastytrade.instruments import Future
    from tastytrade.order import (
        NewOrder, NewComplexOrder,
        OrderAction, OrderType, OrderTimeInForce,
        OrderStatus as TastyOrderStatus,
        PlacedOrder,
    )
    TASTYTRADE_AVAILABLE = True
except ImportError:
    TASTYTRADE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FillResult:
    """Order fill result, compatible with MockFill interface in runner.py.

    Runner accesses: fill.order_id, fill.instrument, fill.side,
    fill.qty, fill.price, fill.timestamp
    """
    order_id: str
    instrument: str
    side: str
    qty: int
    price: float
    timestamp: datetime


@dataclass
class BracketState:
    """Tracks a resting stop-loss order for an open position."""
    instrument: str
    side: str                           # "long" or "short"
    entry_price: float
    stop_price: float
    order_id: Optional[int] = None      # tastytrade order ID (simple stop)
    complex_order_id: Optional[int] = None  # tastytrade complex order ID (OCO)
    is_oco: bool = False                # True if OCO (TP + SL), False if simple stop
    filled: bool = False
    fill_price: float = 0.0


# ---------------------------------------------------------------------------
# Main broker class
# ---------------------------------------------------------------------------

class TastytradeBroker:
    """Order management using tastytrade REST API.

    Drop-in replacement for MockOrderManager in runner.py.
    Same method signatures: connect, disconnect, place_market_order,
    close_position, get_position, reconcile_positions.

    Additionally supports:
      - Resting stop-loss orders (placed automatically after entry)
      - AlertStreamer for real-time bracket fill detection
      - Front-month contract symbol auto-resolution
    """

    def __init__(
        self,
        config: TastytradeConfig,
        strategy_configs: dict[str, StrategyConfig],
    ) -> None:
        """
        Args:
            config: tastytrade credentials and settings.
            strategy_configs: {strategy_id: StrategyConfig} for stop/TP params.
        """
        if not TASTYTRADE_AVAILABLE:
            raise ImportError(
                "tastytrade package not installed. Run: pip install tastytrade"
            )

        self._config = config
        self._strategy_configs = strategy_configs

        # Session and account (set in connect())
        self._session: Optional[Session] = None
        self._account: Optional[Account] = None
        self._connected = False

        # Contract symbol mapping: "MNQ" -> "/MNQH6"
        self._contract_symbols: dict[str, str] = {}

        # Cached Future objects for building order legs
        self._futures: dict[str, Future] = {}

        # Local position tracking: strategy_id -> {side, qty, avg_price, order_id}
        self._positions: dict[str, dict] = {}

        # Active bracket (stop) orders: strategy_id -> BracketState
        self._brackets: dict[str, BracketState] = {}

        # Queue for bracket fills detected by AlertStreamer
        self._bracket_fill_queue: asyncio.Queue = asyncio.Queue()

        # Background tasks
        self._alert_task: Optional[asyncio.Task] = None

        # Order counter for local IDs
        self._order_count = 0

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Authenticate with tastytrade and resolve contract symbols.

        Returns True if connection succeeded.
        """
        try:
            logger.info("[TT] Authenticating...")
            self._session = Session(
                self._config.client_secret,
                self._config.refresh_token,
                is_test=self._config.is_sandbox,
            )

            # Get account
            if self._config.account_number:
                self._account = await Account.get(
                    self._session, self._config.account_number
                )
            else:
                accounts = await Account.get(self._session)
                if not accounts:
                    logger.error("[TT] No accounts found")
                    return False
                self._account = accounts[0]

            logger.info(
                f"[TT] Account: {self._account.account_number} "
                f"({'sandbox' if self._config.is_sandbox else 'PRODUCTION'})"
            )

            # Resolve front-month contract for each unique instrument
            unique_instruments = list(dict.fromkeys(
                sc.instrument for sc in self._strategy_configs.values()
            ))
            for instrument in unique_instruments:
                symbol = await self._resolve_front_month(instrument)
                self._contract_symbols[instrument] = symbol
                # Cache the Future object
                self._futures[instrument] = await Future.get(
                    self._session, symbol
                )
                logger.info(f"[TT] {instrument} -> {symbol}")

            # Start AlertStreamer for fill monitoring
            self._alert_task = asyncio.create_task(
                self._alert_stream_loop(),
                name="tt_alert_stream",
            )

            self._connected = True
            logger.info("[TT] Connected and ready")
            return True

        except Exception as e:
            logger.error(f"[TT] Connection failed: {e}", exc_info=True)
            return False

    async def disconnect(self) -> None:
        """Disconnect from tastytrade."""
        self._connected = False
        if self._alert_task:
            self._alert_task.cancel()
            try:
                await self._alert_task
            except asyncio.CancelledError:
                pass
        logger.info("[TT] Disconnected")

    async def _resolve_front_month(self, root: str) -> str:
        """Find the front-month contract symbol for a product root.

        Queries tastytrade for all active contracts of the given product
        (e.g., 'MNQ') and returns the nearest expiration.

        Returns symbol like '/MNQH6'.
        """
        futures = await Future.get(self._session, product_codes=[root])
        active = [f for f in futures if f.active]
        if not active:
            raise ValueError(f"No active {root} contracts found on tastytrade")

        active.sort(key=lambda f: f.expiration_date)
        front = active[0]
        return front.symbol

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def place_market_order(
        self,
        instrument: str,
        side: str,
        qty: int,
        price_hint: float,
        strategy_id: str = "",
    ) -> FillResult:
        """Place a market entry order.

        After the entry fills, automatically places:
          - OCO bracket (TP + SL) if tp_pts > 0
          - Simple STOP if max_loss_pts > 0 and no TP
          - Nothing otherwise

        Args:
            instrument: "MNQ" or "MES"
            side: "long" or "short" (or "buy"/"sell")
            qty: Number of contracts (always 1 for us)
            price_hint: Current price for fallback if fill price unavailable
            strategy_id: Strategy identifier for position/bracket tracking

        Returns:
            FillResult with the actual fill price.
        """
        sid = strategy_id or instrument
        norm_side = self._normalize_side(side)
        future = self._futures[instrument]

        # Build order
        action = OrderAction.BUY if norm_side == "long" else OrderAction.SELL
        from decimal import Decimal
        leg = future.build_leg(Decimal(str(qty)), action)

        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            legs=[leg],
        )

        logger.info(
            f"[TT] Submitting MARKET {action.value} {qty}x {instrument} "
            f"({self._contract_symbols[instrument]}) strategy={sid}"
        )

        response = await self._account.place_order(
            self._session, order, dry_run=False
        )

        # Wait for fill
        fill_price = await self._wait_for_fill(response.order, price_hint)

        # Track position locally (keyed by strategy_id)
        self._order_count += 1
        order_id = f"TT-{self._order_count:06d}"

        self._positions[sid] = {
            "side": norm_side,
            "qty": qty,
            "avg_price": fill_price,
            "order_id": order_id,
            "instrument": instrument,
        }

        # Auto-place bracket/stop based on strategy config
        strat = self._strategy_configs.get(sid)
        if strat and strat.tp_pts > 0 and strat.max_loss_pts > 0:
            # OCO bracket with TP + SL
            await self.place_oco_bracket(
                instrument, norm_side, qty, fill_price,
                strat.max_loss_pts, strat.tp_pts,
                strategy_id=sid,
            )
        elif strat and strat.max_loss_pts > 0:
            # Simple stop only
            await self._place_stop(
                instrument, norm_side, qty, fill_price, strat.max_loss_pts,
                strategy_id=sid,
            )

        logger.info(
            f"[TT] FILLED {order_id}: {norm_side.upper()} {qty}x {instrument} "
            f"@ {fill_price:.2f} strategy={sid}"
        )

        return FillResult(
            order_id=order_id,
            instrument=instrument,
            side=norm_side,
            qty=qty,
            price=fill_price,
            timestamp=datetime.now(timezone.utc),
        )

    async def close_position(
        self,
        instrument: str,
        price_hint: float,
        strategy_id: str = "",
    ) -> Optional[FillResult]:
        """Close an open position. Cancels any resting stop first.

        Args:
            instrument: "MNQ" or "MES"
            price_hint: Current price for fallback
            strategy_id: Strategy identifier for position/bracket lookup

        Returns:
            FillResult, or None if no position to close.
        """
        sid = strategy_id or instrument
        pos = self._positions.pop(sid, None)
        if pos is None:
            logger.warning(f"[TT] No position to close for {sid}")
            return None

        # Cancel any resting stop order first
        await self._cancel_bracket(sid)

        # Place market close
        future = self._futures[instrument]
        close_action = OrderAction.SELL if pos["side"] == "long" else OrderAction.BUY

        from decimal import Decimal
        leg = future.build_leg(Decimal(str(pos["qty"])), close_action)

        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            legs=[leg],
        )

        close_side = "sell" if pos["side"] == "long" else "buy"
        logger.info(
            f"[TT] Submitting MARKET CLOSE {close_action.value} "
            f"{pos['qty']}x {instrument} strategy={sid}"
        )

        response = await self._account.place_order(
            self._session, order, dry_run=False
        )

        fill_price = await self._wait_for_fill(response.order, price_hint)

        self._order_count += 1
        order_id = f"TT-{self._order_count:06d}"

        logger.info(
            f"[TT] CLOSE FILLED {order_id}: {close_side.upper()} "
            f"{pos['qty']}x {instrument} @ {fill_price:.2f} strategy={sid}"
        )

        return FillResult(
            order_id=order_id,
            instrument=instrument,
            side=close_side,
            qty=pos["qty"],
            price=fill_price,
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Stop loss (resting STOP order on exchange)
    # ------------------------------------------------------------------

    async def _place_stop(
        self,
        instrument: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_pts: int,
        strategy_id: str = "",
    ) -> None:
        """Place a resting STOP order for max-loss protection.

        For long positions: STOP SELL at entry_price - stop_pts
        For short positions: STOP BUY at entry_price + stop_pts

        This order sits on the CME exchange and fires instantly when
        price touches the stop level -- no polling delay.
        """
        sid = strategy_id or instrument
        future = self._futures[instrument]

        # Close action is opposite of position side
        close_action = OrderAction.SELL if side == "long" else OrderAction.BUY

        # Calculate stop trigger price
        if side == "long":
            stop_price = entry_price - stop_pts
        else:
            stop_price = entry_price + stop_pts

        from decimal import Decimal
        leg = future.build_leg(Decimal(str(qty)), close_action)

        stop_order = NewOrder(
            time_in_force=OrderTimeInForce.GTC,
            order_type=OrderType.STOP,
            legs=[leg],
            stop_trigger=Decimal(str(stop_price)),
        )

        response = await self._account.place_order(
            self._session, stop_order, dry_run=False
        )

        bracket = BracketState(
            instrument=instrument,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            order_id=response.order.id if response.order else None,
            is_oco=False,
        )
        self._brackets[sid] = bracket

        logger.info(
            f"[TT] STOP placed: {sid} ({instrument}) "
            f"{'SELL' if side == 'long' else 'BUY'} "
            f"@ {stop_price:.2f} (entry={entry_price:.2f}, "
            f"max_loss={stop_pts}pts)"
        )

    async def place_oco_bracket(
        self,
        instrument: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_pts: int,
        tp_pts: int,
        strategy_id: str = "",
    ) -> None:
        """Place an OCO bracket with both stop loss and take profit.

        Used by v15 tp_scalp strategy. When one leg fills, the other is
        automatically cancelled by the exchange.

        For long: STOP SELL at entry - stop_pts, LIMIT SELL at entry + tp_pts
        For short: STOP BUY at entry + stop_pts, LIMIT BUY at entry - tp_pts
        """
        sid = strategy_id or instrument
        future = self._futures[instrument]
        close_action = OrderAction.SELL if side == "long" else OrderAction.BUY

        from decimal import Decimal
        leg = future.build_leg(Decimal(str(qty)), close_action)

        if side == "long":
            stop_price = entry_price - stop_pts
            tp_price = entry_price + tp_pts
        else:
            stop_price = entry_price + stop_pts
            tp_price = entry_price - tp_pts

        oco = NewComplexOrder(
            orders=[
                NewOrder(  # Take profit
                    time_in_force=OrderTimeInForce.GTC,
                    order_type=OrderType.LIMIT,
                    legs=[leg],
                    price=Decimal(str(tp_price)),
                ),
                NewOrder(  # Stop loss
                    time_in_force=OrderTimeInForce.GTC,
                    order_type=OrderType.STOP,
                    legs=[leg],
                    stop_trigger=Decimal(str(stop_price)),
                ),
            ]
        )

        response = await self._account.place_complex_order(
            self._session, oco, dry_run=False
        )

        bracket = BracketState(
            instrument=instrument,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            complex_order_id=response.order.id if response.order else None,
            is_oco=True,
        )
        self._brackets[sid] = bracket

        logger.info(
            f"[TT] OCO bracket placed: {sid} ({instrument}) "
            f"TP={tp_price:.2f} SL={stop_price:.2f}"
        )

    async def _cancel_bracket(self, strategy_id: str) -> None:
        """Cancel resting stop/bracket for a strategy."""
        bracket = self._brackets.pop(strategy_id, None)
        if bracket is None or bracket.filled:
            return

        try:
            if bracket.is_oco and bracket.complex_order_id is not None:
                await self._account.delete_complex_order(
                    self._session, bracket.complex_order_id
                )
            elif bracket.order_id is not None:
                await self._account.delete_order(
                    self._session, bracket.order_id
                )
            logger.info(f"[TT] Resting stop cancelled for {strategy_id}")
        except Exception as e:
            logger.error(
                f"[TT] Failed to cancel stop for {strategy_id}: {e}"
            )

    async def check_bracket_fills(self, strategy_id: str) -> Optional[dict]:
        """Check if a resting stop was triggered by the exchange.

        Non-blocking -- reads from the queue populated by AlertStreamer.
        Call this before processing each bar in the runner.

        Args:
            strategy_id: Strategy identifier to check bracket fills for.

        Returns:
            dict with fill info if stop was hit, None otherwise.
            Example: {'instrument': 'MNQ', 'strategy_id': 'MNQ_V15',
                     'type': 'stop_loss', 'price': 21450.0, 'side': 'long'}
        """
        # Drain the queue, looking for fills matching this strategy_id
        pending = []
        result = None

        while not self._bracket_fill_queue.empty():
            try:
                fill_info = self._bracket_fill_queue.get_nowait()
                if fill_info.get("strategy_id", fill_info.get("instrument")) == strategy_id:
                    result = fill_info
                    # Clean up bracket and position state
                    self._brackets.pop(strategy_id, None)
                    self._positions.pop(strategy_id, None)
                else:
                    pending.append(fill_info)
            except asyncio.QueueEmpty:
                break

        # Put back fills for other strategies
        for item in pending:
            await self._bracket_fill_queue.put(item)

        return result

    # ------------------------------------------------------------------
    # Fill monitoring
    # ------------------------------------------------------------------

    async def _wait_for_fill(
        self,
        placed_order: Optional[PlacedOrder],
        fallback_price: float,
        timeout: float = 5.0,
    ) -> float:
        """Poll for order fill within timeout.

        Market orders on liquid MNQ/MES fill in <500ms during RTH.
        Returns the actual fill price, or fallback_price on timeout.
        """
        if placed_order is None:
            logger.warning("[TT] No placed order returned, using fallback price")
            return fallback_price

        order_id = placed_order.id
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                orders = await self._account.get_live_orders(self._session)
                for order in orders:
                    if order.id == order_id:
                        if order.status == TastyOrderStatus.FILLED:
                            for leg in order.legs:
                                if leg.fills:
                                    return float(leg.fills[0].fill_price)
                        elif order.status in (
                            TastyOrderStatus.REJECTED,
                            TastyOrderStatus.CANCELLED,
                        ):
                            logger.error(
                                f"[TT] Order {order_id} "
                                f"{order.status.value}"
                            )
                            return fallback_price
            except Exception as e:
                logger.error(f"[TT] Fill poll error: {e}")

            await asyncio.sleep(0.15)

        logger.warning(
            f"[TT] Fill timeout ({timeout}s) for order {order_id}, "
            f"using fallback price {fallback_price:.2f}"
        )
        return fallback_price

    async def _alert_stream_loop(self) -> None:
        """Background task: listen for bracket fill events via AlertStreamer.

        Detects when a resting STOP order fires on the exchange between
        polling intervals. Puts fill info into the bracket_fill_queue
        for check_bracket_fills() to pick up.
        """
        retry_count = 0
        max_retries = 10

        while self._connected and retry_count < max_retries:
            try:
                async with AlertStreamer(self._session) as streamer:
                    await streamer.subscribe_accounts([self._account])
                    retry_count = 0  # Reset on successful connection

                    async for order in streamer.listen(PlacedOrder):
                        if not self._connected:
                            break

                        if order.status != TastyOrderStatus.FILLED:
                            continue

                        # Check if this is a bracket stop fill
                        await self._process_alert_fill(order)

            except asyncio.CancelledError:
                break
            except Exception as e:
                retry_count += 1
                wait = min(2 ** retry_count, 30)
                logger.error(
                    f"[TT] AlertStreamer error (retry {retry_count}/"
                    f"{max_retries}): {e}. Reconnecting in {wait}s..."
                )
                if self._connected:
                    await asyncio.sleep(wait)

        if retry_count >= max_retries:
            logger.error("[TT] AlertStreamer max retries reached, stopping")

    async def _process_alert_fill(self, order: PlacedOrder) -> None:
        """Check if a filled order matches an active bracket stop."""
        for sid, bracket in list(self._brackets.items()):
            if bracket.filled:
                continue

            # Match by order ID
            if bracket.order_id is not None and order.id == bracket.order_id:
                fill_price = self._extract_fill_price(order)
                bracket.filled = True
                bracket.fill_price = fill_price

                await self._bracket_fill_queue.put({
                    "instrument": bracket.instrument,
                    "strategy_id": sid,
                    "type": "stop_loss",
                    "price": fill_price,
                    "side": bracket.side,
                })

                logger.info(
                    f"[TT] STOP TRIGGERED: {sid} ({bracket.instrument}) "
                    f"@ {fill_price:.2f} (was {bracket.side})"
                )
                return

    @staticmethod
    def _extract_fill_price(order: PlacedOrder) -> float:
        """Extract fill price from a PlacedOrder's legs."""
        for leg in order.legs:
            if leg.fills:
                return float(leg.fills[0].fill_price)
        return 0.0

    # ------------------------------------------------------------------
    # Position tracking
    # ------------------------------------------------------------------

    def get_position(self, strategy_id: str) -> Optional[dict]:
        """Get local position state by strategy_id. Returns None if flat."""
        return self._positions.get(strategy_id)

    async def reconcile_positions(self) -> dict[str, dict]:
        """Query broker positions and compare with local state.

        Returns current broker positions as {instrument: {side, qty, avg_price}}.
        Logs warnings on mismatches.

        Note: Broker knows only about instruments (net position per symbol).
        Local positions are keyed by strategy_id, so we aggregate per instrument
        for comparison.
        """
        try:
            broker_positions = await self._account.get_positions(self._session)

            result: dict[str, dict] = {}
            for pos in broker_positions:
                # Match broker symbol back to instrument name
                for inst, symbol in self._contract_symbols.items():
                    if pos.symbol == symbol:
                        side = "long" if pos.quantity > 0 else "short"
                        qty = abs(int(pos.quantity))
                        result[inst] = {
                            "side": side,
                            "qty": qty,
                            "avg_price": float(pos.average_open_price),
                        }

            # Aggregate local positions per instrument for comparison
            local_by_inst: dict[str, int] = {}  # instrument -> signed net qty
            for sid, pos in self._positions.items():
                inst = pos.get("instrument", sid)
                signed = pos["qty"] if pos["side"] == "long" else -pos["qty"]
                local_by_inst[inst] = local_by_inst.get(inst, 0) + signed

            # Check for mismatches
            all_instruments = set(list(local_by_inst.keys()) + list(result.keys()))
            for inst in all_instruments:
                local_net = local_by_inst.get(inst, 0)
                broker = result.get(inst)
                broker_signed = 0
                if broker:
                    broker_signed = broker["qty"] if broker["side"] == "long" else -broker["qty"]

                if local_net != broker_signed:
                    logger.error(
                        f"[TT] POSITION MISMATCH {inst}: "
                        f"local_net={local_net}, broker={broker_signed}"
                    )

            return result

        except Exception as e:
            logger.error(f"[TT] Reconciliation error: {e}")
            # Return aggregated by instrument from local state
            agg: dict[str, dict] = {}
            for sid, pos in self._positions.items():
                inst = pos.get("instrument", sid)
                if inst not in agg:
                    agg[inst] = dict(pos)
                # Simple: just return first match (reconciliation is best-effort)
            return agg

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_side(side: str) -> str:
        """Normalize side to 'long' or 'short'."""
        side = side.lower()
        if side in ("long", "buy"):
            return "long"
        elif side in ("short", "sell"):
            return "short"
        raise ValueError(f"Invalid side: {side}")

    def get_contract_symbols(self) -> dict[str, str]:
        """Return resolved contract symbols: {'MNQ': '/MNQH6', ...}"""
        return dict(self._contract_symbols)
