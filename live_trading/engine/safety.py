"""
Safety layer for the live trading engine.

Monitors risk limits, connection health, and position integrity.
Provides circuit breakers and a kill switch to protect against
runaway losses, data outages, and position mismatches.

Integrates with the EventBus to observe trade_closed events and
with the OrderManager for position reconciliation and flatten-all.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from .config import SafetyConfig
from .events import Bar, EventBus, TradeRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety state
# ---------------------------------------------------------------------------

@dataclass
class SafetyState:
    """Mutable state tracked by the SafetyManager."""
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    consecutive_losses: int = 0
    total_trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    trading_halted: bool = False
    halt_reason: str = ""
    last_bar_time: Optional[datetime] = None
    last_heartbeat: float = 0.0         # monotonic time of last bar
    connection_alert_sent: bool = False
    flatten_alert_sent: bool = False
    session_started: bool = False


# ---------------------------------------------------------------------------
# Pre-session check results
# ---------------------------------------------------------------------------

@dataclass
class PreSessionCheckResult:
    """Result of pre-session health checks."""
    passed: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safety Manager
# ---------------------------------------------------------------------------

class SafetyManager:
    """Central safety layer for the trading engine.

    Responsibilities:
        - Track daily P&L (realized + unrealized) and halt if limit breached
        - Count consecutive losses and pause after too many
        - Monitor data feed heartbeat and alert/flatten on timeout
        - Position reconciliation between local and broker state
        - Kill switch: immediate flatten-all and halt
        - Pre-session checks before trading begins

    Usage:
        safety = SafetyManager(config, event_bus, flatten_callback, pause_callback)
        # EventBus integration is automatic (subscribes to trade_closed)
        # Call on_bar() from the engine loop to feed heartbeat
        # Call start_watchdog() to begin the background monitor
    """

    def __init__(
        self,
        config: SafetyConfig,
        event_bus: EventBus,
        flatten_callback: Callable[[], asyncio.Future],
        pause_callback: Callable[[], None],
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._flatten = flatten_callback
        self._pause = pause_callback
        self.state = SafetyState()
        self._watchdog_task: Optional[asyncio.Task] = None

        # Subscribe to events
        event_bus.subscribe("trade_closed", self._on_trade_closed)
        event_bus.subscribe("bar", self._on_bar_event)

    # -------------------------------------------------------------------
    # Pre-session checks
    # -------------------------------------------------------------------

    async def pre_session_checks(
        self,
        data_feed_connected: bool,
        broker_connected: bool,
        warmup_complete: bool,
    ) -> PreSessionCheckResult:
        """Run all pre-session health checks before trading begins.

        Args:
            data_feed_connected: Whether the data feed is receiving bars.
            broker_connected: Whether broker API is authenticated.
            warmup_complete: Whether indicator warmup has completed.

        Returns:
            PreSessionCheckResult with pass/fail and details.
        """
        result = PreSessionCheckResult()

        # Check 1: Data feed connected
        result.checks["data_feed"] = data_feed_connected
        if not data_feed_connected:
            result.passed = False
            result.messages.append("Data feed not connected")

        # Check 2: Broker connected (skip in paper mode)
        if self._config.paper_mode:
            result.checks["broker"] = True
            result.messages.append("Paper mode: broker check skipped")
        else:
            result.checks["broker"] = broker_connected
            if not broker_connected:
                result.passed = False
                result.messages.append("Broker not connected")

        # Check 3: Warmup complete
        result.checks["warmup"] = warmup_complete
        if not warmup_complete:
            result.passed = False
            result.messages.append("Indicator warmup not complete")

        # Check 4: Not already halted
        result.checks["not_halted"] = not self.state.trading_halted
        if self.state.trading_halted:
            result.passed = False
            result.messages.append(
                f"Trading already halted: {self.state.halt_reason}"
            )

        # Check 5: Daily loss not already at limit
        total_pnl = self.state.daily_realized_pnl + self.state.daily_unrealized_pnl
        within_limit = total_pnl > -self._config.max_daily_loss
        result.checks["daily_loss_limit"] = within_limit
        if not within_limit:
            result.passed = False
            result.messages.append(
                f"Daily P&L ${total_pnl:.2f} already at/below "
                f"limit -${self._config.max_daily_loss:.2f}"
            )

        if result.passed:
            self.state.session_started = True
            logger.info("[Safety] Pre-session checks PASSED")
        else:
            logger.warning(
                f"[Safety] Pre-session checks FAILED: "
                f"{', '.join(result.messages)}"
            )

        for check_name, passed in result.checks.items():
            logger.info(f"  {check_name}: {'PASS' if passed else 'FAIL'}")

        return result

    # -------------------------------------------------------------------
    # Heartbeat / Bar monitoring
    # -------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        """Called by the engine loop on every new bar.

        Updates heartbeat timestamp and resets connection alerts.
        """
        self.state.last_bar_time = bar.timestamp
        self.state.last_heartbeat = time.monotonic()
        self.state.connection_alert_sent = False
        self.state.flatten_alert_sent = False

    def _on_bar_event(self, bar: Bar) -> None:
        """EventBus handler for bar events (delegates to on_bar)."""
        self.on_bar(bar)

    # -------------------------------------------------------------------
    # Trade tracking
    # -------------------------------------------------------------------

    def _on_trade_closed(self, trade: TradeRecord) -> None:
        """EventBus handler for completed trades."""
        self.state.daily_realized_pnl += trade.pnl_dollar
        self.state.total_trades_today += 1

        if trade.pnl_dollar >= 0:
            self.state.wins_today += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losses_today += 1
            self.state.consecutive_losses += 1

        logger.info(
            f"[Safety] Trade closed: {trade.instrument} {trade.side} "
            f"PnL=${trade.pnl_dollar:+.2f} | "
            f"Daily=${self.state.daily_realized_pnl:+.2f} | "
            f"ConsecLoss={self.state.consecutive_losses}"
        )

        # Check daily loss limit
        total_pnl = self.state.daily_realized_pnl + self.state.daily_unrealized_pnl
        if total_pnl <= -self._config.max_daily_loss:
            self._halt(
                f"Daily loss limit breached: ${total_pnl:.2f} <= "
                f"-${self._config.max_daily_loss:.2f}"
            )
            return

        # Check consecutive loss circuit breaker
        if self.state.consecutive_losses >= self._config.max_consecutive_losses:
            self._halt(
                f"Consecutive loss limit: {self.state.consecutive_losses} >= "
                f"{self._config.max_consecutive_losses}"
            )
            return

    def update_unrealized_pnl(self, unrealized: float) -> None:
        """Update unrealized P&L (called by engine with current mark-to-market).

        Checks if combined realized + unrealized breaches the daily limit.
        """
        self.state.daily_unrealized_pnl = unrealized
        total_pnl = self.state.daily_realized_pnl + unrealized
        if total_pnl <= -self._config.max_daily_loss and not self.state.trading_halted:
            self._halt(
                f"Daily loss limit breached (unrealized): ${total_pnl:.2f} <= "
                f"-${self._config.max_daily_loss:.2f}"
            )

    # -------------------------------------------------------------------
    # Order pre-check
    # -------------------------------------------------------------------

    def check_order_allowed(self, symbol: str, qty: int) -> tuple[bool, str]:
        """Check if an order is allowed under current safety constraints.

        Returns:
            (allowed, reason). If not allowed, reason explains why.
        """
        if self.state.trading_halted:
            return False, f"Trading halted: {self.state.halt_reason}"

        if qty > self._config.max_position_size:
            return False, (
                f"Qty {qty} exceeds max position size "
                f"{self._config.max_position_size}"
            )

        total_pnl = self.state.daily_realized_pnl + self.state.daily_unrealized_pnl
        if total_pnl <= -self._config.max_daily_loss:
            return False, f"Daily loss limit: ${total_pnl:.2f}"

        if self.state.consecutive_losses >= self._config.max_consecutive_losses:
            return False, (
                f"Consecutive losses: {self.state.consecutive_losses} >= "
                f"{self._config.max_consecutive_losses}"
            )

        return True, ""

    # -------------------------------------------------------------------
    # Kill switch
    # -------------------------------------------------------------------

    async def kill_switch(self) -> None:
        """Emergency: flatten all positions and halt trading immediately."""
        logger.critical("[Safety] KILL SWITCH ACTIVATED")
        self._halt("Kill switch activated")

        try:
            await self._flatten()
            logger.critical("[Safety] Flatten-all completed after kill switch")
        except Exception as e:
            logger.critical(f"[Safety] Flatten-all FAILED after kill switch: {e}")

        self._event_bus.emit("status_change", {
            "status": "killed",
            "reason": "Kill switch activated",
        })

    # -------------------------------------------------------------------
    # Halt / Resume
    # -------------------------------------------------------------------

    def _halt(self, reason: str) -> None:
        """Halt trading with reason."""
        if self.state.trading_halted:
            logger.warning(f"[Safety] Already halted. New reason: {reason}")
            return

        self.state.trading_halted = True
        self.state.halt_reason = reason
        logger.critical(f"[Safety] TRADING HALTED: {reason}")

        self._pause()
        self._event_bus.emit("status_change", {
            "status": "halted",
            "reason": reason,
        })
        self._event_bus.emit("error", {
            "msg": f"Trading halted: {reason}",
            "severity": "critical",
        })

    def resume(self) -> None:
        """Resume trading after a halt (manual override)."""
        if not self.state.trading_halted:
            logger.info("[Safety] Not halted, nothing to resume")
            return

        logger.warning(
            f"[Safety] RESUMING trading (was halted: {self.state.halt_reason})"
        )
        self.state.trading_halted = False
        self.state.halt_reason = ""
        self._event_bus.emit("status_change", {
            "status": "resumed",
            "reason": "Manual resume after halt",
        })

    # -------------------------------------------------------------------
    # Connection watchdog
    # -------------------------------------------------------------------

    async def start_watchdog(self) -> None:
        """Start the background connection watchdog.

        Monitors time since last bar and triggers alerts/flatten.
        """
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("[Safety] Connection watchdog started")

    async def stop_watchdog(self) -> None:
        """Stop the connection watchdog."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
            logger.info("[Safety] Connection watchdog stopped")

    async def _watchdog_loop(self) -> None:
        """Background loop checking data feed heartbeat."""
        check_interval = 5.0  # Check every 5 seconds
        try:
            while True:
                await asyncio.sleep(check_interval)

                if self.state.last_heartbeat == 0.0:
                    # No bars received yet, skip
                    continue

                elapsed = time.monotonic() - self.state.last_heartbeat

                # Alert threshold
                if (elapsed >= self._config.heartbeat_timeout_sec
                        and not self.state.connection_alert_sent):
                    self.state.connection_alert_sent = True
                    logger.warning(
                        f"[Safety] No bar for {elapsed:.0f}s "
                        f"(alert threshold: {self._config.heartbeat_timeout_sec}s)"
                    )
                    self._event_bus.emit("error", {
                        "msg": (
                            f"Data feed stale: no bar for {elapsed:.0f}s"
                        ),
                        "severity": "high",
                    })

                # Flatten threshold
                if (elapsed >= self._config.flatten_timeout_sec
                        and not self.state.flatten_alert_sent):
                    self.state.flatten_alert_sent = True
                    logger.critical(
                        f"[Safety] No bar for {elapsed:.0f}s -- "
                        f"FLATTEN threshold ({self._config.flatten_timeout_sec}s) breached"
                    )
                    self._event_bus.emit("error", {
                        "msg": (
                            f"Data feed dead: no bar for {elapsed:.0f}s, "
                            f"flattening all positions"
                        ),
                        "severity": "critical",
                    })
                    try:
                        await self._flatten()
                        self._halt(
                            f"Connection lost: no data for {elapsed:.0f}s"
                        )
                    except Exception as e:
                        logger.critical(
                            f"[Safety] Flatten-all FAILED during watchdog: {e}"
                        )

        except asyncio.CancelledError:
            pass

    # -------------------------------------------------------------------
    # Position reconciliation
    # -------------------------------------------------------------------

    async def run_reconciliation(
        self,
        reconcile_func: Callable[[], asyncio.Future],
    ) -> None:
        """Run position reconciliation and handle mismatches.

        Args:
            reconcile_func: Async callable that returns
                {symbol: {"local_side", "local_qty", "broker_side", "broker_qty", "match"}}
        """
        try:
            result = await reconcile_func()
        except Exception as e:
            logger.error(f"[Safety] Reconciliation failed: {e}")
            self._event_bus.emit("error", {
                "msg": f"Position reconciliation failed: {e}",
                "severity": "high",
            })
            return

        mismatches = {
            sym: info for sym, info in result.items() if not info["match"]
        }

        if mismatches:
            logger.error(
                f"[Safety] Position mismatches found: {mismatches}"
            )
            self._halt(f"Position mismatch detected: {list(mismatches.keys())}")
        else:
            logger.info("[Safety] Position reconciliation: all positions match")

    async def start_reconciliation_loop(
        self,
        reconcile_func: Callable[[], asyncio.Future],
    ) -> None:
        """Run periodic position reconciliation."""
        interval = self._config.recon_interval_sec
        logger.info(
            f"[Safety] Starting reconciliation loop (interval={interval}s)"
        )
        try:
            while True:
                await asyncio.sleep(interval)
                if self.state.trading_halted:
                    continue
                await self.run_reconciliation(reconcile_func)
        except asyncio.CancelledError:
            pass

    # -------------------------------------------------------------------
    # Daily reset
    # -------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset all daily counters at start of new trading day."""
        logger.info(
            f"[Safety] Daily reset. Previous day: "
            f"PnL=${self.state.daily_realized_pnl:+.2f}, "
            f"Trades={self.state.total_trades_today}, "
            f"Wins={self.state.wins_today}, "
            f"Losses={self.state.losses_today}"
        )
        self.state = SafetyState()
