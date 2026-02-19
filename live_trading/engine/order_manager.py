"""
Order management for the live trading engine.

Handles order placement, fill tracking, timeout alerts, and local
position state. The base class defines the interface; concrete subclasses
implement broker-specific API calls.

WebullOrderManager: production order routing (stub -- API not approved).
MockOrderManager: immediate fills for testing and paper trading.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from .config import SafetyConfig, WebullConfig
from .events import EventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order types and status
# ---------------------------------------------------------------------------

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderRecord:
    """Internal record of an order lifecycle."""
    order_id: str
    symbol: str
    side: OrderSide
    qty: int
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    fill_price: float = 0.0
    reject_reason: str = ""
    broker_order_id: str = ""


@dataclass
class LocalPosition:
    """Locally tracked position state for a single instrument."""
    symbol: str
    side: Optional[str] = None      # "long", "short", or None (flat)
    qty: int = 0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class OrderManager(ABC):
    """Base order manager with position tracking and fill monitoring.

    Subclasses must implement:
        _submit_market_order()  -- send order to broker
        _query_order_status()   -- check order status at broker
        _cancel_order()         -- cancel a pending order
        _query_positions()      -- get broker's position state
    """

    def __init__(
        self,
        event_bus: EventBus,
        safety_config: SafetyConfig,
        on_fill: Optional[Callable[[OrderRecord], None]] = None,
    ) -> None:
        self._event_bus = event_bus
        self._safety_config = safety_config
        self._on_fill_callback = on_fill
        self._orders: dict[str, OrderRecord] = {}
        self._positions: dict[str, LocalPosition] = {}
        self._fill_timeout = safety_config.order_fill_timeout_sec

    def get_position(self, symbol: str) -> LocalPosition:
        """Get local position state for a symbol."""
        if symbol not in self._positions:
            self._positions[symbol] = LocalPosition(symbol=symbol)
        return self._positions[symbol]

    def get_all_positions(self) -> dict[str, LocalPosition]:
        """Get all tracked positions."""
        return dict(self._positions)

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        """Look up an order by ID."""
        return self._orders.get(order_id)

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
    ) -> str:
        """Place a market order and monitor for fill.

        Args:
            symbol: Instrument symbol (e.g. "MNQ", "MES").
            side: BUY or SELL.
            qty: Number of contracts.

        Returns:
            order_id: Locally generated order ID for tracking.
        """
        order_id = f"ord_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        order = OrderRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            status=OrderStatus.PENDING,
            submitted_at=now,
        )
        self._orders[order_id] = order

        logger.info(
            f"[OrderManager] Submitting {side.value} {qty}x {symbol} "
            f"(order_id={order_id})"
        )

        try:
            broker_id = await self._submit_market_order(symbol, side, qty)
            order.broker_order_id = broker_id
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)
            logger.error(
                f"[OrderManager] Order REJECTED on submission: {order_id} -- {e}"
            )
            self._event_bus.emit("error", {
                "msg": f"Order rejected: {symbol} {side.value} {qty} -- {e}",
                "severity": "high",
            })
            return order_id

        # Monitor for fill with timeout
        fill_received = await self._wait_for_fill(order)
        if not fill_received:
            order.status = OrderStatus.TIMEOUT
            logger.warning(
                f"[OrderManager] Order TIMEOUT: {order_id} not filled within "
                f"{self._fill_timeout}s"
            )
            self._event_bus.emit("error", {
                "msg": (
                    f"Order timeout: {symbol} {side.value} {qty} "
                    f"(order_id={order_id}) -- no fill in {self._fill_timeout}s"
                ),
                "severity": "high",
            })

        return order_id

    async def _wait_for_fill(self, order: OrderRecord) -> bool:
        """Poll for order fill within timeout window.

        Returns True if filled, False if timed out.
        """
        deadline = time.monotonic() + self._fill_timeout
        poll_interval = 0.1  # 100ms poll

        while time.monotonic() < deadline:
            if order.status == OrderStatus.FILLED:
                return True
            if order.status == OrderStatus.REJECTED:
                return False

            # Check broker status
            try:
                status, fill_price = await self._query_order_status(order.broker_order_id)
            except Exception as e:
                logger.error(f"[OrderManager] Status query failed: {e}")
                await asyncio.sleep(poll_interval)
                continue

            if status == OrderStatus.FILLED:
                self._on_order_filled(order, fill_price)
                return True
            elif status == OrderStatus.REJECTED:
                order.status = OrderStatus.REJECTED
                logger.error(f"[OrderManager] Order REJECTED by broker: {order.order_id}")
                return False

            await asyncio.sleep(poll_interval)

        return False

    def _on_order_filled(self, order: OrderRecord, fill_price: float) -> None:
        """Handle a confirmed fill."""
        now = datetime.now(timezone.utc)
        order.status = OrderStatus.FILLED
        order.filled_at = now
        order.fill_price = fill_price

        logger.info(
            f"[OrderManager] FILLED: {order.side.value} {order.qty}x {order.symbol} "
            f"@ {fill_price:.2f} (order_id={order.order_id})"
        )

        # Update local position
        pos = self.get_position(order.symbol)
        if order.side == OrderSide.BUY:
            if pos.side == "short":
                # Closing short
                pos.side = None
                pos.qty = 0
                pos.entry_price = 0.0
                pos.entry_time = None
            else:
                # Opening or adding to long
                pos.side = "long"
                pos.qty += order.qty
                pos.entry_price = fill_price
                pos.entry_time = now
        elif order.side == OrderSide.SELL:
            if pos.side == "long":
                # Closing long
                pos.side = None
                pos.qty = 0
                pos.entry_price = 0.0
                pos.entry_time = None
            else:
                # Opening or adding to short
                pos.side = "short"
                pos.qty += order.qty
                pos.entry_price = fill_price
                pos.entry_time = now

        # Emit fill event
        fill_data = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "qty": order.qty,
            "price": fill_price,
            "filled_at": now.isoformat(),
        }
        self._event_bus.emit("fill", fill_data)

        if self._on_fill_callback:
            self._on_fill_callback(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Returns True if cancellation succeeded.
        """
        order = self._orders.get(order_id)
        if order is None:
            logger.warning(f"[OrderManager] Cannot cancel unknown order: {order_id}")
            return False
        if order.status != OrderStatus.PENDING:
            logger.warning(
                f"[OrderManager] Cannot cancel order {order_id} in "
                f"status {order.status.value}"
            )
            return False

        try:
            success = await self._cancel_order(order.broker_order_id)
            if success:
                order.status = OrderStatus.CANCELLED
                logger.info(f"[OrderManager] Order cancelled: {order_id}")
            return success
        except Exception as e:
            logger.error(f"[OrderManager] Cancel failed for {order_id}: {e}")
            return False

    async def flatten_all(self) -> list[str]:
        """Close all open positions immediately.

        Returns list of order IDs for the closing orders.
        """
        order_ids: list[str] = []
        for symbol, pos in self._positions.items():
            if pos.side is None or pos.qty == 0:
                continue

            close_side = OrderSide.SELL if pos.side == "long" else OrderSide.BUY
            logger.warning(
                f"[OrderManager] FLATTEN: closing {pos.side} {pos.qty}x {symbol}"
            )
            oid = await self.place_market_order(symbol, close_side, pos.qty)
            order_ids.append(oid)

        return order_ids

    async def reconcile_positions(self) -> dict[str, dict]:
        """Compare local position state with broker.

        Returns dict of discrepancies:
            {symbol: {"local": ..., "broker": ..., "match": bool}}
        """
        result: dict[str, dict] = {}
        try:
            broker_positions = await self._query_positions()
        except Exception as e:
            logger.error(f"[OrderManager] Position reconciliation failed: {e}")
            return result

        all_symbols = set(list(self._positions.keys()) + list(broker_positions.keys()))

        for symbol in all_symbols:
            local = self._positions.get(symbol)
            broker = broker_positions.get(symbol)

            local_qty = local.qty if local and local.side else 0
            local_side = local.side if local else None
            broker_qty = broker.qty if broker and broker.side else 0
            broker_side = broker.side if broker else None

            match = local_qty == broker_qty and local_side == broker_side

            if not match:
                logger.error(
                    f"[OrderManager] POSITION MISMATCH for {symbol}: "
                    f"local={local_side}/{local_qty}, "
                    f"broker={broker_side}/{broker_qty}"
                )
                self._event_bus.emit("error", {
                    "msg": (
                        f"Position mismatch: {symbol} "
                        f"local={local_side}/{local_qty} vs "
                        f"broker={broker_side}/{broker_qty}"
                    ),
                    "severity": "critical",
                })

            result[symbol] = {
                "local_side": local_side,
                "local_qty": local_qty,
                "broker_side": broker_side,
                "broker_qty": broker_qty,
                "match": match,
            }

        return result

    @abstractmethod
    async def _submit_market_order(
        self, symbol: str, side: OrderSide, qty: int
    ) -> str:
        """Submit a market order to the broker.

        Returns broker-assigned order ID.
        Raises on submission failure.
        """
        ...

    @abstractmethod
    async def _query_order_status(
        self, broker_order_id: str
    ) -> tuple[OrderStatus, float]:
        """Query order status from broker.

        Returns (status, fill_price). fill_price is 0 if not filled.
        """
        ...

    @abstractmethod
    async def _cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order at the broker. Returns True if successful."""
        ...

    @abstractmethod
    async def _query_positions(self) -> dict[str, LocalPosition]:
        """Query all positions from broker.

        Returns {symbol: LocalPosition}.
        """
        ...


# ---------------------------------------------------------------------------
# Webull order manager (stub -- API not approved)
# ---------------------------------------------------------------------------

class WebullOrderManager(OrderManager):
    """Production order manager using Webull API.

    All broker methods log not-implemented and return safe defaults.
    """

    def __init__(
        self,
        event_bus: EventBus,
        safety_config: SafetyConfig,
        webull_config: WebullConfig,
        on_fill: Optional[Callable[[OrderRecord], None]] = None,
    ) -> None:
        super().__init__(event_bus, safety_config, on_fill)
        self._webull_config = webull_config
        self._access_token: Optional[str] = None

    async def authenticate(self) -> bool:
        """Authenticate with Webull API."""
        logger.warning(
            "[WebullOrderManager] authenticate() not implemented -- "
            "Webull API access not yet approved. Returning False."
        )
        return False

    async def _submit_market_order(
        self, symbol: str, side: OrderSide, qty: int
    ) -> str:
        """Submit market order to Webull.

        Expected endpoint: POST /api/trade/order/place
        Body: {account_id, instrument_id, side, order_type=MARKET, qty}
        """
        logger.warning(
            f"[WebullOrderManager] _submit_market_order({symbol}, {side.value}, {qty}) "
            "not implemented -- Webull API access not yet approved."
        )
        raise RuntimeError(
            "Webull order submission not implemented -- API access not approved"
        )

    async def _query_order_status(
        self, broker_order_id: str
    ) -> tuple[OrderStatus, float]:
        """Query order status from Webull.

        Expected endpoint: GET /api/trade/order/{order_id}
        """
        logger.warning(
            f"[WebullOrderManager] _query_order_status({broker_order_id}) "
            "not implemented -- Webull API access not yet approved."
        )
        return OrderStatus.PENDING, 0.0

    async def _cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at Webull.

        Expected endpoint: POST /api/trade/order/cancel
        """
        logger.warning(
            f"[WebullOrderManager] _cancel_order({broker_order_id}) "
            "not implemented -- Webull API access not yet approved."
        )
        return False

    async def _query_positions(self) -> dict[str, LocalPosition]:
        """Query positions from Webull.

        Expected endpoint: GET /api/trade/account/{account_id}/positions
        """
        logger.warning(
            "[WebullOrderManager] _query_positions() "
            "not implemented -- Webull API access not yet approved."
        )
        return {}


# ---------------------------------------------------------------------------
# Mock order manager for testing
# ---------------------------------------------------------------------------

class MockOrderManager(OrderManager):
    """Order manager that immediately fills all orders at the requested price.

    Used for paper trading and integration testing. Fill price comes from
    the last known bar price injected via set_current_price().
    """

    def __init__(
        self,
        event_bus: EventBus,
        safety_config: SafetyConfig,
        on_fill: Optional[Callable[[OrderRecord], None]] = None,
        slippage_pts: float = 0.0,
    ) -> None:
        super().__init__(event_bus, safety_config, on_fill)
        self._current_prices: dict[str, float] = {}
        self._slippage = slippage_pts
        self._next_broker_id = 0

    def set_current_price(self, symbol: str, price: float) -> None:
        """Set the current market price for a symbol (used for fill simulation)."""
        self._current_prices[symbol] = price

    async def _submit_market_order(
        self, symbol: str, side: OrderSide, qty: int
    ) -> str:
        """Simulate order submission with immediate fill."""
        self._next_broker_id += 1
        broker_id = f"mock_{self._next_broker_id}"

        # Get fill price
        base_price = self._current_prices.get(symbol, 0.0)
        if base_price == 0.0:
            logger.warning(
                f"[MockOrderManager] No current price for {symbol}, filling at 0"
            )

        # Apply slippage
        if side == OrderSide.BUY:
            fill_price = base_price + self._slippage
        else:
            fill_price = base_price - self._slippage

        # Find the order that was just created and fill it
        # (the order was added to self._orders before _submit_market_order is called)
        for oid, order in reversed(list(self._orders.items())):
            if (order.symbol == symbol and order.side == side
                    and order.status == OrderStatus.PENDING):
                order.broker_order_id = broker_id
                self._on_order_filled(order, fill_price)
                break

        return broker_id

    async def _query_order_status(
        self, broker_order_id: str
    ) -> tuple[OrderStatus, float]:
        """Mock orders are always immediately filled."""
        for order in self._orders.values():
            if order.broker_order_id == broker_order_id:
                return order.status, order.fill_price
        return OrderStatus.FILLED, 0.0

    async def _cancel_order(self, broker_order_id: str) -> bool:
        """Mock cancellation always succeeds."""
        return True

    async def _query_positions(self) -> dict[str, LocalPosition]:
        """Return local positions (mock has no external broker state)."""
        return dict(self._positions)
