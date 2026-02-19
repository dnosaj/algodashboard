"""
Event bus for the trading engine.

Pub/sub system that allows advisors and other modules to observe and
influence engine decisions without touching core logic. Handlers must
be fast and synchronous -- no async/network calls in handlers.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SignalType(Enum):
    NONE = "NONE"
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class ExitReason(Enum):
    SM_FLIP = "SM_FLIP"
    STOP_LOSS = "SL"
    TAKE_PROFIT = "TP"
    TRAIL_STOP = "TRAIL"
    EOD = "EOD"
    KILL_SWITCH = "KILL"
    MANUAL = "MANUAL"


@dataclass
class Bar:
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument: str = ""


@dataclass
class Signal:
    """Trading signal emitted by the strategy."""
    type: SignalType
    instrument: str
    reason: str = ""
    sm_value: float = 0.0
    rsi_value: float = 0.0
    exit_reason: Optional[ExitReason] = None


@dataclass
class TradeRecord:
    """Completed trade record."""
    instrument: str
    side: str           # "long" or "short"
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pts: float
    pnl_dollar: float
    exit_reason: str
    bars_held: int = 0
    strategy_id: str = ""


@dataclass
class PreOrderContext:
    """Context passed to advisors before order placement."""
    signal: Signal
    instrument: str
    side: str           # "long" or "short"
    qty: int = 1
    skip: bool = False  # Advisors can set True to veto (logged)
    skip_reason: str = ""


class EventBus:
    """Simple synchronous pub/sub event bus.

    Events:
        "bar"           - New bar received. Payload: Bar
        "signal"        - Strategy generated a signal. Payload: Signal
        "pre_order"     - About to place order. Payload: PreOrderContext
                          Advisors can modify qty or set skip=True.
        "fill"          - Order filled. Payload: dict(side, price, qty, order_id)
        "trade_closed"  - Trade completed. Payload: TradeRecord
        "daily_summary" - End of day stats. Payload: dict(trades, pnl, win_rate)
        "error"         - Error occurred. Payload: dict(msg, severity)
        "status_change" - Engine status changed. Payload: dict(status, reason)
    """

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}

    def subscribe(self, event: str, handler: Callable) -> None:
        """Register a handler for an event."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        """Remove a handler."""
        if event in self._handlers:
            self._handlers[event] = [h for h in self._handlers[event] if h != handler]

    def emit(self, event: str, payload: Any = None) -> None:
        """Emit an event to all subscribers.

        Handlers are called synchronously in registration order.
        If a handler raises, it's logged but doesn't stop other handlers.
        """
        for handler in self._handlers.get(event, []):
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Event handler error on '{event}': {e}", exc_info=True)

    def emit_pre_order(self, context: PreOrderContext) -> PreOrderContext:
        """Special emit for pre_order -- returns modified context.

        Advisors can modify qty or set skip=True. Each advisor gets
        the context modified by previous advisors.
        """
        for handler in self._handlers.get("pre_order", []):
            try:
                result = handler(context)
                if isinstance(result, PreOrderContext):
                    context = result
            except Exception as e:
                logger.error(f"Pre-order handler error: {e}", exc_info=True)
        return context
