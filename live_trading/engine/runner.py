"""
Main async orchestration loop for the live trading engine.

Ties together: data feed, strategy, advisors, order management,
safety checks, and the API server. Handles graceful shutdown on
SIGINT/SIGTERM.

Usage:
    from engine.runner import run
    from engine.config import DEFAULT_CONFIG
    asyncio.run(run(DEFAULT_CONFIG))
"""

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from engine.config import EngineConfig, StrategyConfig
from engine.events import (
    Bar, EventBus, ExitReason, PreOrderContext, Signal, SignalType, TradeRecord,
)
from engine.intra_bar_monitor import IntraBarExitMonitor
from engine.strategy import IncrementalStrategy
from advisors.base import Advisor
from advisors.sizing import FixedSizeAdvisor
from api.server import EngineHandle, create_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock data feed (Day 1 -- replaced with real Webull/Databento feed later)
# ---------------------------------------------------------------------------

class MockDataFeed:
    """Simulated data feed that generates flat bars for testing.

    In production, this is replaced with a real-time data feed from
    Webull gRPC or Databento live streaming.
    """

    def __init__(self, instruments: list[str]):
        self._instruments = instruments
        self._connected = False
        self._last_prices: dict[str, float] = {
            "MNQ": 21500.0,
            "MES": 5900.0,
        }

    async def connect(self) -> None:
        self._connected = True
        logger.info("[MockDataFeed] Connected (simulated)")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("[MockDataFeed] Disconnected")

    @property
    def connected(self) -> bool:
        return self._connected

    def get_warmup_bars(self, instrument: str, count: int) -> list[Bar]:
        """Return synthetic warmup bars for indicator initialization."""
        bars = []
        price = self._last_prices.get(instrument, 20000.0)
        now = datetime.now(timezone.utc)
        from datetime import timedelta

        for i in range(count):
            ts = now - timedelta(minutes=count - i)
            bar = Bar(
                timestamp=ts,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=100.0,
                instrument=instrument,
            )
            bars.append(bar)
        return bars

    async def get_latest_bar(self, instrument: str) -> Optional[Bar]:
        """Poll for the latest 1-min bar. Returns None if no new bar."""
        if not self._connected:
            return None

        price = self._last_prices.get(instrument, 20000.0)
        bar = Bar(
            timestamp=datetime.now(timezone.utc),
            open=price,
            high=price + 2.0,
            low=price - 2.0,
            close=price + 0.5,
            volume=150.0,
            instrument=instrument,
        )
        return bar


# ---------------------------------------------------------------------------
# Mock order manager (Day 1 -- replaced with Webull order API later)
# ---------------------------------------------------------------------------

@dataclass
class MockFill:
    """Simulated order fill."""
    order_id: str
    instrument: str
    side: str
    qty: int
    price: float
    timestamp: datetime


class MockOrderManager:
    """Simulated order management for paper trading.

    In production, this sends orders via Webull's REST/gRPC API
    and handles fill confirmations, partial fills, and rejects.
    """

    def __init__(self):
        self._order_count = 0
        self._positions: dict[str, dict] = {}  # instrument -> {side, qty, avg_price}
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        logger.info("[MockOrderManager] Connected (paper mode)")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("[MockOrderManager] Disconnected")

    @property
    def connected(self) -> bool:
        return self._connected

    def get_position(self, instrument: str) -> Optional[dict]:
        """Get current position for an instrument. None if flat."""
        return self._positions.get(instrument)

    async def place_market_order(
        self, instrument: str, side: str, qty: int, price_hint: float,
        strategy_id: str = "",
    ) -> MockFill:
        """Simulate a market order fill at the price hint (current bar open)."""
        sid = strategy_id or instrument
        self._order_count += 1
        order_id = f"MOCK-{self._order_count:06d}"

        fill = MockFill(
            order_id=order_id,
            instrument=instrument,
            side=side,
            qty=qty,
            price=price_hint,
            timestamp=datetime.now(timezone.utc),
        )

        self._positions[sid] = {
            "side": side,
            "qty": qty,
            "avg_price": price_hint,
            "order_id": order_id,
            "instrument": instrument,
        }

        logger.info(
            f"[MockOrderManager] FILL {order_id}: {side.upper()} {qty}x {instrument} "
            f"@ {price_hint:.2f} strategy={sid}"
        )
        return fill

    async def close_position(
        self, instrument: str, price_hint: float,
        strategy_id: str = "",
    ) -> Optional[MockFill]:
        """Close an open position."""
        sid = strategy_id or instrument
        pos = self._positions.pop(sid, None)
        if pos is None:
            logger.warning(f"[MockOrderManager] No position to close for {sid}")
            return None

        self._order_count += 1
        order_id = f"MOCK-{self._order_count:06d}"

        close_side = "sell" if pos["side"] == "long" else "buy"
        fill = MockFill(
            order_id=order_id,
            instrument=instrument,
            side=close_side,
            qty=pos["qty"],
            price=price_hint,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            f"[MockOrderManager] CLOSE {order_id}: {close_side.upper()} {pos['qty']}x "
            f"{instrument} @ {price_hint:.2f} strategy={sid}"
        )
        return fill

    async def reconcile_positions(self) -> dict[str, dict]:
        """Return current positions (mock just returns internal state)."""
        return dict(self._positions)


# SafetyManager imported from engine.safety_manager
from engine.safety_manager import SafetyManager


# ---------------------------------------------------------------------------
# Engine state (shared across runner, API, websocket)
# ---------------------------------------------------------------------------

@dataclass
class EngineState:
    """Shared mutable state for the engine.

    Accessed by the runner loop, API endpoints, and WebSocket broadcasts.
    Protected by asyncio (single-threaded event loop, no locking needed).
    """
    config: EngineConfig
    event_bus: EventBus
    strategies: dict[str, IncrementalStrategy] = field(default_factory=dict)
    advisors: list[Advisor] = field(default_factory=list)
    safety: Optional[SafetyManager] = None
    data_feed: Optional[MockDataFeed] = None
    order_manager: Optional[MockOrderManager] = None
    start_time: float = 0.0
    trading_active: bool = False
    shutdown_event: Optional[asyncio.Event] = None
    all_trades: list[TradeRecord] = field(default_factory=list)
    # Track last known prices per instrument for kill switch
    last_prices: dict[str, float] = field(default_factory=dict)
    # Track last trading day for daily reset
    last_trading_day: Optional[str] = None
    # Track last quote time for staleness detection (Fix #8)
    last_quote_time: float = 0.0
    # Flag for intra-bar monitor active state
    intrabar_monitor_active: bool = False


# ---------------------------------------------------------------------------
# Core bar processing
# ---------------------------------------------------------------------------

async def process_bar(
    bar: Bar,
    strategy: IncrementalStrategy,
    state: EngineState,
) -> None:
    """Process a single bar through the strategy and order pipeline.

    Steps:
        1. Update last known price for kill switch
        2. Check if resting stop was triggered on exchange (bracket fill)
        3. Run strategy.on_bar() to get signal
        4. If BUY/SELL: run pre_order advisors, safety check, place order
        5. If CLOSE_*: close position via order manager
    """
    sid = strategy.strategy_id

    # Track last known price for emergency closes
    state.last_prices[bar.instrument] = bar.close

    # Check if a resting stop was triggered on the exchange between polls.
    # The TastytradeBroker's AlertStreamer detects bracket fills and queues
    # them. We check here BEFORE running strategy.on_bar() so the strategy
    # sees the position as already closed and won't generate a duplicate
    # close signal.
    if hasattr(state.order_manager, 'check_bracket_fills'):
        fill_info = await state.order_manager.check_bracket_fills(sid)
        if fill_info:
            # Stop was hit on the exchange -- create synthetic bar with
            # the actual fill price so the trade record is accurate
            fill_price = fill_info["price"]
            synthetic_bar = Bar(
                timestamp=bar.timestamp,
                open=fill_price, high=fill_price,
                low=fill_price, close=fill_price,
                volume=0, instrument=bar.instrument,
            )
            strategy.force_close(synthetic_bar, ExitReason.STOP_LOSS)
            logger.info(
                f"[Runner] Resting stop triggered for {sid} "
                f"@ {fill_price:.2f}"
            )
            # Still run on_bar so indicators stay current, but the strategy
            # will see position=0 and just update SM/RSI state
            strategy.on_bar(bar)
            return

    # Run strategy (emits "bar" event internally, which feeds safety via event bus)
    sig = strategy.on_bar(bar)

    if sig.type == SignalType.NONE:
        return

    # Entry signals
    if sig.type in (SignalType.BUY, SignalType.SELL):
        side = "long" if sig.type == SignalType.BUY else "short"

        # Build pre-order context
        context = PreOrderContext(
            signal=sig,
            instrument=bar.instrument,
            side=side,
            qty=1,
            strategy_id=sid,
        )

        # Apply SafetyManager qty override before advisors
        if state.safety:
            qty_override = state.safety.get_qty_override(sid)
            if qty_override is not None:
                context.qty = qty_override
                context.qty_locked = True

        # Run advisors via event bus
        context = state.event_bus.emit_pre_order(context)

        if context.skip:
            logger.info(
                f"[Runner] Trade VETOED by advisor for {sid}: {context.skip_reason}"
            )
            return

        # Safety check: sum positions across all strategies for this instrument
        if state.safety:
            current_exposure = sum(
                abs(s.position) for s in state.strategies.values()
                if s.config.instrument == bar.instrument
            )
            ok, reason = state.safety.check_can_trade(
                bar.instrument, current_exposure + context.qty,
                strategy_id=sid,
            )
            if not ok:
                logger.warning(f"[Runner] Trade blocked by safety for {sid}: {reason}")
                return

        # Place order
        if state.order_manager:
            fill = await state.order_manager.place_market_order(
                instrument=bar.instrument,
                side=side,
                qty=context.qty,
                price_hint=bar.open,
                strategy_id=sid,
            )

            state.event_bus.emit("fill", {
                "side": side,
                "price": fill.price,
                "qty": fill.qty,
                "order_id": fill.order_id,
                "instrument": bar.instrument,
                "strategy_id": sid,
            })

    # Close signals (strategy already recorded the trade internally)
    elif sig.type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
        if state.order_manager:
            await state.order_manager.close_position(
                instrument=bar.instrument,
                price_hint=bar.open,
                strategy_id=sid,
            )


# ---------------------------------------------------------------------------
# Async loops
# ---------------------------------------------------------------------------

async def bar_processing_loop(state: EngineState) -> None:
    """Process bars immediately as they complete from the data feed.

    Replaces the 60s polling loop. Bars arrive via asyncio.Queue from the
    TastytradeDataFeed, typically within ~100ms of bar completion.
    """
    logger.info("[Runner] Bar processing loop started (event-driven)")

    while not state.shutdown_event.is_set():
        try:
            bar = await asyncio.wait_for(
                state.data_feed.bar_queue.get(), timeout=65.0
            )
        except asyncio.TimeoutError:
            continue  # No bar in 65s, loop back and check shutdown
        except AttributeError:
            # MockDataFeed doesn't have bar_queue -- fall back to sleep
            await asyncio.sleep(60.0)
            continue

        if state.shutdown_event.is_set():
            break
        if not state.trading_active:
            continue

        inst = bar.instrument
        strategies = [s for s in state.strategies.values()
                      if s.config.instrument == inst]

        logger.info(
            f"[Runner] Bar: {inst} {bar.timestamp.strftime('%H:%M')} UTC "
            f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
            f"C={bar.close:.2f} V={bar.volume:.0f}"
        )

        try:
            _check_daily_reset(bar, state)
            for strategy in strategies:
                async with strategy.trade_lock:
                    await process_bar(bar, strategy, state)
        except Exception as e:
            logger.error(f"[Runner] Error processing bar for {inst}: {e}", exc_info=True)
            state.event_bus.emit("error", {"msg": str(e), "severity": "HIGH"})

        # Staleness check: disable intra-bar monitor if quotes stopped flowing
        if state.intrabar_monitor_active and state.last_quote_time > 0:
            if time.time() - state.last_quote_time > 60:
                logger.warning("[Runner] Quote feed stale (>60s), disabling intra-bar monitor")
                state.intrabar_monitor_active = False
                for s in state.strategies.values():
                    if hasattr(s, 'intrabar_monitor_active'):
                        s.intrabar_monitor_active = False


async def intra_bar_exit_loop(state: EngineState) -> None:
    """Process real-time quotes for intra-bar TP/trail exits.

    Reads from the data feed's quote queue and dispatches to the
    IntraBarExitMonitor which checks TP/trail conditions on each tick.
    """
    logger.info("[Runner] Intra-bar exit monitor loop started")

    monitor = IntraBarExitMonitor(
        state.strategies, state.order_manager,
        state.event_bus, state.safety,
    )

    # Activate the intrabar_monitor_active flag on monitored strategies
    for sid in monitor._monitored:
        state.strategies[sid].intrabar_monitor_active = True
        logger.info(f"[Runner] Intra-bar monitoring active for {sid}")
    state.intrabar_monitor_active = bool(monitor._monitored)

    while not state.shutdown_event.is_set():
        try:
            quote = await asyncio.wait_for(
                state.data_feed.quote_queue.get(), timeout=5.0
            )
        except asyncio.TimeoutError:
            continue
        except AttributeError:
            # MockDataFeed doesn't have quote_queue
            await asyncio.sleep(60.0)
            continue

        if state.shutdown_event.is_set():
            break

        instrument = state.data_feed.resolve_instrument(quote.event_symbol)
        if instrument:
            bid = float(quote.bid_price) if quote.bid_price else 0.0
            ask = float(quote.ask_price) if quote.ask_price else 0.0
            if bid > 0 and ask > 0:
                # Update last price from quotes too
                mid = (bid + ask) / 2.0
                state.last_prices[instrument] = mid
                state.last_quote_time = time.time()
                await monitor.on_quote(instrument, bid, ask)


def _check_daily_reset(bar: Bar, state: EngineState) -> None:
    """Reset daily counters if we've entered a new trading day.

    Uses Eastern Time dates so the reset aligns with session boundaries
    (session close ~16:00 ET), not midnight UTC (which falls mid-session
    at 7-8 PM ET and would spuriously clear drawdown protection).
    """
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
    bar_et = bar.timestamp.astimezone(_ET)
    today = bar_et.strftime("%Y-%m-%d")
    if state.last_trading_day is None:
        state.last_trading_day = today
    elif today != state.last_trading_day:
        logger.info(f"[Runner] New trading day detected: {state.last_trading_day} -> {today}")
        state.last_trading_day = today
        if state.safety:
            state.safety.reset_daily()
        for strat in state.strategies.values():
            strat.reset_daily()


async def reconciliation_loop(state: EngineState) -> None:
    """Periodically reconcile positions with broker."""
    logger.info("[Runner] Position reconciliation loop started")
    interval = state.config.safety.recon_interval_sec

    while not state.shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                state.shutdown_event.wait(),
                timeout=float(interval),
            )
            break
        except asyncio.TimeoutError:
            pass

        if state.order_manager and state.order_manager.connected:
            try:
                broker_positions = await state.order_manager.reconcile_positions()

                # Sum strategy positions per instrument for comparison
                # (broker sees ONE net position per symbol)
                inst_net: dict[str, int] = {}
                for strat in state.strategies.values():
                    inst = strat.config.instrument
                    inst_net[inst] = inst_net.get(inst, 0) + strat.position

                all_instruments = set(list(inst_net.keys()) + list(broker_positions.keys()))
                for inst in all_instruments:
                    strat_net = inst_net.get(inst, 0)
                    broker_pos = broker_positions.get(inst)
                    broker_qty = broker_pos["qty"] if broker_pos else 0
                    broker_side = broker_pos["side"] if broker_pos else "flat"

                    # Convert broker to signed qty for comparison
                    broker_signed = broker_qty if broker_side == "long" else -broker_qty if broker_side == "short" else 0

                    if strat_net != broker_signed:
                        logger.warning(
                            f"[Recon] {inst}: Strategy net={strat_net} but broker={broker_signed} "
                            f"({broker_side} x{broker_qty}). Manual intervention needed."
                        )
                        state.event_bus.emit("error", {
                            "msg": f"Position mismatch on {inst}: strategy net={strat_net}, broker={broker_signed}",
                            "severity": "HIGH",
                        })

            except Exception as e:
                logger.error(f"[Recon] Reconciliation error: {e}", exc_info=True)


async def heartbeat_loop(state: EngineState) -> None:
    """Monitor data feed health and trigger alerts."""
    logger.info("[Runner] Heartbeat monitor started")

    while not state.shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                state.shutdown_event.wait(),
                timeout=10.0,
            )
            break
        except asyncio.TimeoutError:
            pass

        if state.safety and state.trading_active:
            healthy, elapsed = state.safety.check_heartbeat()
            if not healthy:
                logger.warning(
                    f"[Heartbeat] No data for {elapsed:.0f}s "
                    f"(threshold: {state.config.safety.heartbeat_timeout_sec}s)"
                )
                state.event_bus.emit("error", {
                    "msg": f"Data feed stale: {elapsed:.0f}s since last bar",
                    "severity": "MEDIUM",
                })

                # Emergency flatten if too long
                if elapsed >= state.config.safety.flatten_timeout_sec:
                    logger.warning(
                        f"[Heartbeat] Flatten timeout ({elapsed:.0f}s). "
                        f"Emergency close all positions."
                    )
                    await _emergency_flatten(state, "connection_timeout")


async def _emergency_flatten(state: EngineState, reason: str) -> None:
    """Emergency flatten all positions using last known prices."""
    state.trading_active = False
    for sid, strat in state.strategies.items():
        if strat.position != 0:
            async with strat.trade_lock:
                if strat.position == 0:
                    continue  # Closed by another path while waiting for lock
                inst = strat.config.instrument
                last_price = state.last_prices.get(inst, None)
                if last_price is None:
                    logger.warning(f"[Emergency] No last known price for {sid} ({inst}), attempting close with 0")
                    last_price = 0.0
                bar = Bar(
                    timestamp=datetime.now(timezone.utc),
                    open=last_price, high=last_price,
                    low=last_price, close=last_price,
                    volume=0, instrument=inst,
                )
                strat.force_close(bar, ExitReason.KILL_SWITCH)
                if state.order_manager and state.order_manager.connected:
                    try:
                        fill = await state.order_manager.close_position(inst, last_price, strategy_id=sid)
                        # Patch trade record with actual fill price
                        if fill and strat.trades and abs(fill.price - last_price) > 0.001:
                            trade = strat.trades[-1]
                            trade._pre_correction_pnl = trade.pnl_dollar
                            trade.exit_price = fill.price
                            trade.pts = (fill.price - trade.entry_price if trade.side == "long"
                                         else trade.entry_price - fill.price)
                            trade.pnl_dollar = trade.pts * strat.config.dollar_per_pt
                            state.event_bus.emit("trade_corrected", trade)
                    except Exception as e:
                        logger.critical(f"[Emergency] Close FAILED for {sid}: {e} — reconciliation will detect")
    logger.warning(f"[Emergency] All positions flattened ({reason})")


# ---------------------------------------------------------------------------
# EngineHandle factory (bridges EngineState to API server)
# ---------------------------------------------------------------------------

def _build_engine_handle(state: EngineState) -> EngineHandle:
    """Create an EngineHandle that exposes the engine state to the API server."""

    def get_status() -> dict:
        """Build status dict matching dashboard expectations."""
        positions = []
        instruments = {}

        for sid, strat in state.strategies.items():
            inst = strat.config.instrument
            last_price = state.last_prices.get(inst, 0.0)

            if strat.position != 0:
                # Compute unrealized P&L
                if strat.state.position == 1:
                    unrealized_pts = last_price - strat.state.entry_price
                else:
                    unrealized_pts = strat.state.entry_price - last_price
                unrealized_pnl = unrealized_pts * strat.config.dollar_per_pt

                positions.append({
                    "instrument": inst,
                    "strategy_id": sid,
                    "side": "LONG" if strat.position == 1 else "SHORT",
                    "entry_price": strat.state.entry_price,
                    "unrealized_pnl": round(unrealized_pnl, 2),
                })
            else:
                positions.append({
                    "instrument": inst,
                    "strategy_id": sid,
                    "side": "FLAT",
                    "entry_price": None,
                    "unrealized_pnl": 0.0,
                })

            # Per-strategy state for dashboard (keyed by strategy_id)
            bars_since_exit = strat.bar_idx - strat.state.exit_bar_idx
            cooldown_remaining = max(0, strat.config.cooldown - bars_since_exit)
            bars_held = (strat.bar_idx - strat.state.entry_bar_idx) if strat.position != 0 else 0

            instruments[sid] = {
                "instrument": inst,
                "strategy_id": sid,
                "last_price": last_price,
                "sm_value": round(strat.sm.value, 4),
                "rsi_value": round(strat.rsi.curr, 1),
                "cooldown_remaining": cooldown_remaining,
                "cooldown_total": strat.config.cooldown,
                "max_loss_pts": strat.config.max_loss_pts,
                "bars_held": bars_held,
                "exit_mode": strat.config.exit_mode,
                "tp_pts": strat.config.tp_pts,
                "long_used": strat.state.long_used,
                "short_used": strat.state.short_used,
            }

        safety_status = state.safety.get_status() if state.safety else {}

        return {
            "connected": (
                (state.data_feed.connected if state.data_feed else False)
                and (state.order_manager.connected if state.order_manager else False)
            ),
            "trading_active": state.trading_active,
            "paper_mode": state.config.safety.paper_mode,
            "positions": positions,
            "instruments": instruments,
            "daily_pnl": safety_status.get("daily_pnl", 0.0),
            "uptime_seconds": round(time.time() - state.start_time, 1),
            "trade_count_today": safety_status.get("trade_count_today", 0),
            "consecutive_losses": safety_status.get("consecutive_losses", 0),
            "broker": state.config.broker,
            "account": state.config.tastytrade.account_number if state.config.broker == "tastytrade" else "",
            "safety": safety_status,
        }

    def get_trades() -> list[TradeRecord]:
        """Collect all trades across all strategies."""
        trades = []
        for strat in state.strategies.values():
            trades.extend(strat.trades)
        trades.sort(key=lambda t: t.exit_time if t.exit_time else t.entry_time)
        return trades

    def pause_trading() -> None:
        state.trading_active = False
        logger.info("[API] Trading PAUSED")

    def resume_trading() -> tuple[bool, str]:
        if state.safety and state.safety.halted:
            return False, f"Cannot resume: {state.safety.halt_reason}"
        state.trading_active = True
        logger.info("[API] Trading RESUMED")
        return True, ""

    async def kill_switch() -> None:
        logger.warning("[API] KILL SWITCH activated")
        await _emergency_flatten(state, "kill_switch")

    def get_bars(instrument: str) -> list[dict]:
        """Return all completed bars for an instrument as dicts.

        Timestamps are shifted to ET epoch so the chart x-axis displays
        Eastern Time (9:30 AM, 10:00 AM, etc.) without frontend conversion.
        """
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")

        if state.data_feed is None:
            return []
        all_bars = getattr(state.data_feed, '_all_bars', None)
        if all_bars is None:
            return []
        # TastytradeDataFeed: dict[str, deque], other feeds: list
        if isinstance(all_bars, dict):
            bars_deque = all_bars.get(instrument, [])
        else:
            bars_deque = all_bars
        result = []
        for bar in sorted(bars_deque, key=lambda b: b.timestamp):
            # Shift epoch by UTC-ET offset so chart displays ET time
            et = bar.timestamp.astimezone(_ET)
            offset_seconds = int(et.utcoffset().total_seconds())
            result.append({
                "time": int(bar.timestamp.timestamp()) + offset_seconds,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })
        return result

    def set_strategy_paused(strategy_id: str, paused: bool) -> tuple[bool, str]:
        if state.safety:
            return state.safety.set_strategy_paused(strategy_id, paused)
        return False, "Safety manager not initialized"

    def set_strategy_qty(strategy_id: str, qty: int) -> tuple[bool, str]:
        if state.safety:
            return state.safety.set_strategy_qty(strategy_id, qty)
        return False, "Safety manager not initialized"

    def set_drawdown_enabled(enabled: bool) -> tuple[bool, str]:
        if state.safety:
            return state.safety.set_drawdown_enabled(enabled)
        return False, "Safety manager not initialized"

    def force_resume_all() -> tuple[bool, str]:
        if state.safety:
            return state.safety.force_resume_all()
        return False, "Safety manager not initialized"

    async def close_strategy_position(strategy_id: str) -> tuple[bool, str]:
        strat = state.strategies.get(strategy_id)
        if not strat:
            return False, f"Unknown strategy: {strategy_id}"
        async with strat.trade_lock:
            if strat.state.position == 0:
                return False, "No open position"
            inst = strat.config.instrument
            last_price = state.last_prices.get(inst, 0.0)
            bar = Bar(
                timestamp=datetime.now(timezone.utc),
                open=last_price, high=last_price,
                low=last_price, close=last_price,
                volume=0, instrument=inst,
            )
            strat.force_close(bar, ExitReason.MANUAL)
            if state.order_manager and state.order_manager.connected:
                try:
                    fill = await state.order_manager.close_position(
                        inst, last_price, strategy_id=strategy_id)
                except Exception as e:
                    logger.critical(f"[Runner] Manual close FAILED for {strategy_id}: {e} — reconciliation will detect")
                    return True, f"Strategy closed locally but broker close failed: {e}"
                # Patch trade record with actual fill price
                if fill and strat.trades and abs(fill.price - last_price) > 0.001:
                    trade = strat.trades[-1]
                    trade._pre_correction_pnl = trade.pnl_dollar
                    trade.exit_price = fill.price
                    trade.pts = (fill.price - trade.entry_price if trade.side == "long"
                                 else trade.entry_price - fill.price)
                    trade.pnl_dollar = trade.pts * strat.config.dollar_per_pt
                    state.event_bus.emit("trade_corrected", trade)
            return True, f"Closed {strategy_id}"

    handle = EngineHandle(
        event_bus=state.event_bus,
        config=state.config,
        get_status=get_status,
        get_trades=get_trades,
        get_bars=get_bars,
        pause_trading=pause_trading,
        resume_trading=resume_trading,
        kill_switch=kill_switch,
        set_strategy_paused=set_strategy_paused,
        set_strategy_qty=set_strategy_qty,
        set_drawdown_enabled=set_drawdown_enabled,
        force_resume_all=force_resume_all,
    )
    handle._close_strategy_position = close_strategy_position
    return handle


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run(config: EngineConfig) -> None:
    """Main async orchestration loop.

    Initializes all components, runs warmup, then enters the polling loop.
    Handles graceful shutdown on SIGINT/SIGTERM.
    """
    # Configure logging — force=True to override any handlers the tastytrade SDK
    # may have already installed on the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # Suppress noisy library loggers
    for noisy in ("tastytrade", "httpx", "httpcore", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("=" * 60)
    logger.info("NQ Trading Engine starting up")
    logger.info(f"  Paper mode: {config.safety.paper_mode}")
    logger.info(f"  Strategies: {[(s.strategy_id or s.instrument) for s in config.strategies]}")
    logger.info(f"  API port: {config.api_port}")
    logger.info("=" * 60)

    # --- 1. Initialize shared state ---
    event_bus = EventBus()
    shutdown_event = asyncio.Event()

    state = EngineState(
        config=config,
        event_bus=event_bus,
        start_time=time.time(),
        shutdown_event=shutdown_event,
    )

    # --- 2. Initialize advisors ---
    sizing_advisor = FixedSizeAdvisor(qty=1)
    state.advisors.append(sizing_advisor)

    # Register advisor with event bus
    def on_pre_order(context: PreOrderContext) -> PreOrderContext:
        result = sizing_advisor.on_signal(context.signal, context)
        if result.get("skip"):
            context.skip = True
            context.skip_reason = result.get("reason", "advisor veto")
        if "qty" in result and not context.qty_locked:
            context.qty = result["qty"]
        return context

    event_bus.subscribe("pre_order", on_pre_order)
    event_bus.subscribe("bar", sizing_advisor.on_bar)
    event_bus.subscribe("trade_closed", sizing_advisor.on_trade_closed)

    logger.info(f"  Advisors loaded: {[a.name for a in state.advisors]}")

    # --- 3. Initialize data feed + order manager ---
    instruments = list(dict.fromkeys(s.instrument for s in config.strategies))

    if config.broker == "tastytrade":
        from engine.tastytrade_feed import TastytradeDataFeed
        from tastytrade import Session

        # Authenticate with tastytrade for market data
        logger.info("  Connecting to tastytrade...")
        tt_session = Session(
            config.tastytrade.client_secret,
            config.tastytrade.refresh_token,
            is_test=config.tastytrade.is_sandbox,
        )
        # Re-suppress after SDK init (SDK may reconfigure loggers)
        for noisy in ("tastytrade", "httpx", "httpcore", "websockets"):
            lg = logging.getLogger(noisy)
            lg.setLevel(logging.WARNING)
            lg.handlers.clear()

        if config.safety.paper_mode:
            # PAPER MODE: real data from tastytrade, mock order fills
            # Resolve contract symbols for data feed
            from tastytrade.instruments import Future
            contract_symbols = {}
            for inst in instruments:
                futures = await Future.get(tt_session, product_codes=[inst])
                active = sorted(
                    [f for f in futures if f.active],
                    key=lambda f: f.expiration_date,
                )
                contract_symbols[inst] = active[0].symbol
                logger.info(f"  {inst} -> {active[0].symbol}")

            data_feed = TastytradeDataFeed(
                session=tt_session,
                instruments=instruments,
                contract_symbols=contract_symbols,
                warmup_bars=config.warmup_bars,
            )
            await data_feed.connect()
            state.data_feed = data_feed

            order_manager = MockOrderManager()
            await order_manager.connect()
            state.order_manager = order_manager

            logger.info("  Broker: tastytrade PAPER (real data, mock fills)")
        else:
            # LIVE MODE: real data + real orders via tastytrade
            from engine.tastytrade_broker import TastytradeBroker

            strat_map = {(s.strategy_id or s.instrument): s for s in config.strategies}
            order_manager = TastytradeBroker(config.tastytrade, strat_map)
            ok = await order_manager.connect()
            if not ok:
                logger.error("[Runner] tastytrade connection failed, aborting")
                return
            state.order_manager = order_manager

            data_feed = TastytradeDataFeed(
                session=order_manager._session,
                instruments=instruments,
                contract_symbols=order_manager.get_contract_symbols(),
                warmup_bars=config.warmup_bars,
            )
            await data_feed.connect()
            state.data_feed = data_feed

            logger.info("  Broker: tastytrade LIVE (real orders!)")
    else:
        # Mock data feed + order manager (paper trading / testing)
        data_feed = MockDataFeed(instruments)
        await data_feed.connect()
        state.data_feed = data_feed

        order_manager = MockOrderManager()
        await order_manager.connect()
        state.order_manager = order_manager

        logger.info("  Broker: mock (paper mode)")

    # --- 5. Initialize safety manager ---
    safety = SafetyManager(config, event_bus=event_bus)
    state.safety = safety

    # Register safety with event bus (only via event bus, no direct calls)
    event_bus.subscribe("bar", safety.on_bar)
    event_bus.subscribe("trade_closed", safety.on_trade_closed)
    # trade_corrected: adjust P&L delta only (trade already counted by trade_closed)
    def _on_trade_corrected(trade):
        old_pnl = getattr(trade, '_pre_correction_pnl', None)
        if old_pnl is None:
            logger.warning(f"[Safety] Trade corrected but no _pre_correction_pnl stamp: {trade.strategy_id}")
            return
        strat_config = safety._strategy_configs.get(trade.strategy_id)
        commission = 2 * (strat_config.commission_per_side if strat_config else 0.52)
        delta = (trade.pnl_dollar - commission) - (old_pnl - commission)
        if abs(delta) < 0.005:
            return
        # Warn if correction flips trade sign (consecutive_losses can't be fixed)
        if (old_pnl - commission) >= 0 and (trade.pnl_dollar - commission) < 0:
            logger.warning(f"[Safety] Correction flipped {trade.strategy_id} from win to loss — consecutive_losses may be stale")
        elif (old_pnl - commission) < 0 and (trade.pnl_dollar - commission) >= 0:
            logger.warning(f"[Safety] Correction flipped {trade.strategy_id} from loss to win — consecutive_losses may be stale")
        safety._global_daily_pnl += delta
        sid = trade.strategy_id or trade.instrument
        strat_status = safety._strategies.get(sid)
        if strat_status:
            strat_status.daily_pnl += delta
        # Re-check circuit breaker
        if safety._global_daily_pnl <= -safety._config.max_daily_loss:
            safety._halted = True
            safety._halt_reason = f"Daily loss limit hit (corrected): ${safety._global_daily_pnl:.2f}"
            logger.warning(f"[Safety] HALTED after correction: {safety._halt_reason}")
        elif safety._halted and "Daily loss limit" in safety._halt_reason:
            # Correction brought daily P&L back within limits — auto-clear halt
            safety._halted = False
            safety._halt_reason = ""
            logger.info(f"[Safety] Halt CLEARED after correction: daily=${safety._global_daily_pnl:+.2f}")
        logger.info(f"[Safety] Trade corrected: {trade.strategy_id} delta=${delta:+.2f} daily=${safety._global_daily_pnl:+.2f}")
        safety._broadcast_status()
    event_bus.subscribe("trade_corrected", _on_trade_corrected)

    # --- 6. Create strategies (keyed by strategy_id, not instrument) ---
    for strat_config in config.strategies:
        strategy = IncrementalStrategy(strat_config, event_bus)
        sid = strategy.strategy_id
        state.strategies[sid] = strategy
        logger.info(
            f"  Strategy created: {sid} ({strat_config.instrument}) "
            f"exit={strat_config.exit_mode} "
            f"SM({strat_config.sm_index}/{strat_config.sm_flow}/"
            f"{strat_config.sm_norm}/{strat_config.sm_ema}) "
            f"RSI({strat_config.rsi_len}/{strat_config.rsi_buy}/{strat_config.rsi_sell}) "
            f"CD={strat_config.cooldown} SL={strat_config.max_loss_pts}"
            + (f" TP={strat_config.tp_pts}" if strat_config.tp_pts > 0 else "")
        )

    # --- 7. Warmup (fetch once per instrument, feed to all strategies) ---
    logger.info(f"  Loading {config.warmup_bars} warmup bars per instrument...")
    # Get unique instruments
    unique_instruments = list(dict.fromkeys(
        s.config.instrument for s in state.strategies.values()
    ))
    for inst in unique_instruments:
        warmup_bars_data = data_feed.get_warmup_bars(inst, config.warmup_bars)
        # Feed to all strategies for this instrument
        strats_for_inst = [
            s for s in state.strategies.values() if s.config.instrument == inst
        ]
        for bar in warmup_bars_data:
            for strat in strats_for_inst:
                strat.warmup(bar)
        for strat in strats_for_inst:
            strat.start_trading()
        logger.info(f"  {inst}: warmup complete ({len(warmup_bars_data)} bars, "
                    f"{len(strats_for_inst)} strategies)")

    # --- 8. Signal handlers for graceful shutdown ---
    loop = asyncio.get_running_loop()

    def handle_shutdown(sig_name: str):
        logger.info(f"[Runner] Received {sig_name}, initiating graceful shutdown...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_shutdown, sig.name)

    # --- 9. Start API server (using standalone api.server module) ---
    handle = _build_engine_handle(state)
    app = create_app(handle)

    uvicorn_config = None
    uvicorn_server = None
    try:
        import uvicorn

        uvicorn_config = uvicorn.Config(
            app,
            host=config.api_host,
            port=config.api_port,
            log_level="warning",
            access_log=False,
        )
        uvicorn_server = uvicorn.Server(uvicorn_config)
    except ImportError:
        logger.warning("[Runner] uvicorn not installed, API server disabled")

    # --- 10. Mark trading active ---
    state.trading_active = True
    logger.info("[Runner] Engine is LIVE. Waiting for bars...")

    # --- 11. Run all loops concurrently ---
    tasks = [
        asyncio.create_task(bar_processing_loop(state), name="bar_process"),
        asyncio.create_task(intra_bar_exit_loop(state), name="intrabar_monitor"),
        asyncio.create_task(reconciliation_loop(state), name="recon"),
        asyncio.create_task(heartbeat_loop(state), name="heartbeat"),
    ]

    if uvicorn_server is not None:
        tasks.append(
            asyncio.create_task(uvicorn_server.serve(), name="api_server")
        )

    try:
        # Wait for shutdown signal
        await shutdown_event.wait()
        logger.info("[Runner] Shutdown signal received, cleaning up...")
    finally:
        # --- Graceful shutdown ---
        state.trading_active = False

        # Close all open positions using last known prices
        await _emergency_flatten(state, "shutdown")

        # Stop uvicorn
        if uvicorn_server is not None:
            uvicorn_server.should_exit = True

        # Cancel background tasks
        for task in tasks:
            task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*tasks, return_exceptions=True)

        # Disconnect
        await data_feed.disconnect()
        await order_manager.disconnect()

        logger.info("[Runner] Engine shut down cleanly.")
