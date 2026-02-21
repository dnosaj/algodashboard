"""
Intra-bar exit monitor for real-time TP/trail exits on quote ticks.

Monitors real-time bid/ask quotes and triggers exits between bar boundaries.
Only active for strategies with intra-bar exit types (tp_pts > 0).

Architecture:
  - V15 (tp_scalp): Monitor checks TP + trail on quotes. SL on exchange STOP.
  - V11 (sm_flip):  No monitor. SL on exchange STOP. SM flip is bar-close only.
  - MES V9.4:       No monitor. No SL, no TP. SM-flip-only.

Uses bid for long exits (selling into bid), ask for short exits (buying at ask).
Holds per-strategy trade_lock during close to prevent races with bar-close
exits, manual exits, and emergency flattens.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from .events import Bar, EventBus, ExitReason, TradeRecord

logger = logging.getLogger(__name__)


class IntraBarExitMonitor:
    """Monitors real-time quotes for intra-bar TP/trail exits."""

    def __init__(self, strategies, order_manager, event_bus, safety_manager):
        self._strategies = strategies
        self._order_manager = order_manager
        self._event_bus = event_bus
        self._safety = safety_manager

        # Only monitor strategies with intra-bar exits
        self._monitored: dict = {
            sid: s for sid, s in strategies.items()
            if s.config.tp_pts > 0 or s.config.exit_mode == "tp_scalp"
        }

        if self._monitored:
            logger.info(
                f"[Monitor] Intra-bar monitoring active for: "
                f"{list(self._monitored.keys())}"
            )

    async def on_quote(self, instrument: str, bid: float, ask: float):
        """Process a real-time quote tick for potential exit.

        All state reads and mutations happen inside trade_lock to prevent
        races with bar-close exits, manual exits, and emergency flattens.

        Args:
            instrument: "MNQ" or "MES"
            bid: Current best bid price
            ask: Current best ask price
        """
        for sid, strat in self._monitored.items():
            if strat.config.instrument != instrument:
                continue
            # Quick unlocked check to skip lock acquisition when flat
            if strat.state.position == 0:
                continue

            async with strat.trade_lock:
                if strat.state.position == 0:
                    continue

                # Use bid for long exits (selling), ask for short exits (buying)
                position = strat.state.position
                price = bid if position == 1 else ask
                entry = strat.state.entry_price
                unrealized = (price - entry) if position == 1 else (entry - price)

                reason = self._check_exit(strat, unrealized)
                if reason:
                    try:
                        await self._execute_exit(strat, price, reason)
                    except Exception as e:
                        logger.error(f"[Monitor] _execute_exit failed for {sid}: {e}", exc_info=True)
                        if self._event_bus:
                            self._event_bus.emit("error", {"msg": f"Intra-bar exit failed for {sid}: {e}", "severity": "CRITICAL"})

    def _check_exit(self, strat, unrealized) -> Optional[ExitReason]:
        """Check if unrealized P&L triggers a TP or trail exit.

        Updates MFE BEFORE checking trail (order matters: MFE must be
        current before computing trail level).

        NOTE: SL is NOT checked here — exchange resting STOP handles SL.
        NOTE: SM flip is NOT checked here — bar-close indicator only.
        """
        # Update MFE before checking trail
        if unrealized > strat.state.max_favorable:
            strat.state.max_favorable = unrealized
            if (strat.config.trail_activate_pts > 0
                    and unrealized >= strat.config.trail_activate_pts):
                strat.state.trail_activated = True

        # TP check
        if strat.config.tp_pts > 0 and unrealized >= strat.config.tp_pts:
            return ExitReason.TAKE_PROFIT

        # Trail check (only tp_scalp mode with trail activated)
        if strat.state.trail_activated:
            trail_level = strat.state.max_favorable - strat.config.trail_distance_pts
            if unrealized <= trail_level:
                return ExitReason.TRAIL_STOP

        return None

    async def _execute_exit(self, strat, price: float, reason: ExitReason):
        """Close position: update strategy state, then place broker order.

        Patches trade record with actual broker fill price if it differs
        from the quote price used for the initial record.
        """
        inst = strat.config.instrument
        sid = strat.config.strategy_id or inst

        # Build a synthetic bar at the quote price for force_close()
        bar = Bar(
            timestamp=datetime.now(timezone.utc),
            open=price, high=price, low=price, close=price,
            volume=0, instrument=inst,
        )

        # 1. Update strategy state + emit trade_closed (SafetyManager notified)
        strat.force_close(bar, reason)

        logger.info(
            f"[Monitor] INTRA-BAR EXIT: {sid} {reason.value} @ {price:.2f}"
        )

        # 2. Place broker close order
        if self._order_manager and self._order_manager.connected:
            try:
                fill = await self._order_manager.close_position(
                    inst, price, strategy_id=sid
                )
            except Exception as e:
                logger.critical(f"[Monitor] Broker close FAILED for {sid}: {e} — reconciliation will detect")
                if self._event_bus:
                    self._event_bus.emit("error", {"msg": f"Broker close failed for {sid}: {e}", "severity": "CRITICAL"})
                return

            # 3. Patch trade record with actual fill price
            if fill and strat.trades and abs(fill.price - price) > 0.001:
                trade = strat.trades[-1]
                trade._pre_correction_pnl = trade.pnl_dollar
                trade.exit_price = fill.price
                if trade.side == "long":
                    trade.pts = fill.price - trade.entry_price
                else:
                    trade.pts = trade.entry_price - fill.price
                trade.pnl_dollar = trade.pts * strat.config.dollar_per_pt

                # Notify SafetyManager of corrected P&L
                if self._event_bus:
                    self._event_bus.emit("trade_corrected", trade)

                logger.info(
                    f"[Monitor] Fill patched: {sid} actual={fill.price:.2f} "
                    f"(was {price:.2f}), PnL={trade.pnl_dollar:+.2f}"
                )
