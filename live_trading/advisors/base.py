"""
Abstract Advisor interface for the trading engine.

Advisors are plug-in modules that observe signals, bars, and trades.
They can influence order flow (adjust qty, veto trades) via on_signal,
or passively track state via on_bar and on_trade_closed.

All advisor methods MUST be fast and synchronous -- no async, no network
calls, no blocking I/O. The EventBus calls them inline on the hot path.
"""

from abc import ABC, abstractmethod
from typing import Any

from engine.events import Bar, PreOrderContext, Signal, TradeRecord


class Advisor(ABC):
    """Base class for all pre-order advisors.

    Subclasses must set `name` and may override any of the hooks below.
    The engine registers advisors with the EventBus during startup and
    calls them at the appropriate points in the order lifecycle.

    Lifecycle:
        1. on_bar()          -- called every 1-min bar (tracking/model updates)
        2. on_signal()       -- called when strategy emits BUY/SELL, before order
                                Return {"skip": True, "reason": "..."} to veto.
                                Return {"qty": N} to adjust position size.
                                Return {} to pass through unchanged.
        3. on_trade_closed() -- called after a trade completes (learning/stats)
    """

    name: str = "unnamed_advisor"

    @abstractmethod
    def on_signal(self, signal: Signal, context: PreOrderContext) -> dict:
        """Called before an order is placed. Return modifications or veto.

        Args:
            signal: The trading signal from the strategy.
            context: The PreOrderContext with instrument, side, qty, etc.

        Returns:
            dict with optional keys:
                "qty" (int): Override position size.
                "skip" (bool): If True, veto this trade.
                "reason" (str): Reason for veto (logged).
            Return {} to pass through with no changes.
        """
        return {}

    def on_bar(self, bar: Bar) -> None:
        """Called on every 1-min bar for tracking and model updates.

        This fires for ALL instruments, not just the one the advisor
        is registered for. Filter by bar.instrument if needed.

        Args:
            bar: The latest 1-min OHLCV bar.
        """
        pass

    def on_trade_closed(self, trade: TradeRecord) -> None:
        """Called after a trade closes for learning and statistics.

        Args:
            trade: The completed trade record with entry/exit info and P&L.
        """
        pass

    def get_status(self) -> dict:
        """Return current advisor state for the dashboard API.

        Returns:
            dict with at minimum {"name": str, "active": bool}.
            Subclasses should add relevant metrics.
        """
        return {"name": self.name, "active": True}
