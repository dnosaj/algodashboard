"""
Tastytrade order management for the live trading engine.

Handles session authentication, order placement, bracket (stop loss) management,
and position tracking via the tastytrade REST API + AlertStreamer.

Order Type Specifications by Strategy
--------------------------------------

Strategies with TP + SL (vScalpA, vScalpB, MES v2):
  ENTRY:  Market order (BUY or SELL) -> immediate fill on CME
  BRACKET: OCO pair: resting LIMIT (TP) + STOP (SL) on exchange
           When one fills, exchange auto-cancels the other
  EXIT:   Cancel OCO -> Market close (EOD or manual)
  FILL:   AlertStreamer detects fill, price proximity determines TP vs SL

Strategies with SL only (no TP configured):
  ENTRY:  Market order -> immediate fill on CME
  STOP:   Resting STOP order at entry_price +/- max_loss_pts
  EXIT:   Cancel STOP -> Market close (SM flip or EOD)

Strategies with no SL (max_loss_pts=0):
  ENTRY:  Market order -> immediate fill on CME
  EXIT:   Market close (SM flip or EOD)

Why resting orders matter:
  - Resting orders fire intra-bar the instant price touches the level
  - No polling delay (60s bar boundary) on stop/TP exits
  - OCO brackets survive engine crashes -- both legs remain on exchange

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
    tp_price: float = 0.0              # TP limit price (OCO only)
    tag: str = ""                      # "tp1", "tp2", or "" (single bracket)
    qty: int = 1                       # contracts in this bracket
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

        # Strategies where OCO placement failed (fallback to simple stop)
        self._oco_failed_sids: set = set()

        # Strategies where ALL protection failed (OCO + fallback stop both failed)
        self._protection_failed_sids: set = set()

        # Order counter for local IDs
        self._order_count = 0

    @property
    def connected(self) -> bool:
        return self._connected

    def _bracket_keys(self, strategy_id: str) -> list[str]:
        """Return all bracket keys for a strategy (handles composite keys).

        Single bracket: ["MES_V2"]
        Multi bracket:  ["MES_V2__tp1", "MES_V2__tp2"]
        """
        return [k for k in self._brackets
                if k == strategy_id or k.startswith(f"{strategy_id}__")]

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
          - OCO bracket (TP LIMIT + SL STOP) if tp_pts > 0 and max_loss_pts > 0
          - SL-only STOP if max_loss_pts > 0 but no TP
          - Nothing if max_loss_pts == 0

        Falls back to simple stop if OCO placement fails.

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
        if strat and strat.max_loss_pts > 0:
            if strat.tp_pts > 0:
                if strat.partial_tp_pts > 0 and qty > 1:
                    # Two independent OCO brackets for partial-exit strategies
                    await self._place_dual_oco(
                        instrument, norm_side, qty, fill_price, strat, sid,
                    )
                else:
                    # Single OCO bracket (existing behavior)
                    try:
                        await self.place_oco_bracket(
                            instrument, norm_side, qty, fill_price,
                            strat.max_loss_pts, strat.tp_pts, strategy_id=sid,
                        )
                    except Exception as e:
                        logger.critical(
                            f"[TT] OCO bracket FAILED for {sid}: {e} — "
                            f"falling back to simple stop"
                        )
                        try:
                            await self._place_stop(
                                instrument, norm_side, qty, fill_price,
                                strat.max_loss_pts, strategy_id=sid,
                            )
                            self._oco_failed_sids.add(sid)
                        except Exception as e2:
                            self._protection_failed_sids.add(sid)
                            logger.critical(
                                f"[TT] BOTH OCO and fallback stop FAILED for {sid}: {e2} — "
                                f"position has NO exchange protection, engine should halt"
                            )
            else:
                # SL-only stop (no TP configured)
                try:
                    await self._place_stop(
                        instrument, norm_side, qty, fill_price,
                        strat.max_loss_pts, strategy_id=sid,
                    )
                except Exception as e:
                    self._protection_failed_sids.add(sid)
                    logger.critical(
                        f"[TT] SL-only stop FAILED for {sid}: {e} — "
                        f"position has NO exchange protection, engine should halt"
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

        # Cancel any resting stop orders FIRST (before popping position)
        # to prevent a race where the stop fills between pop and cancel,
        # causing a double-close (market order on already-flat position).
        # Check ALL bracket keys (handles composite keys for multi-bracket).
        keys = self._bracket_keys(sid)
        for key in keys:
            bracket = self._brackets.get(key)
            if bracket and bracket.filled:
                # At least one bracket already filled — check if it was SL
                # (SL fills both brackets → full position closed on exchange)
                tp_dist = abs(bracket.fill_price - bracket.tp_price) if bracket.tp_price else float('inf')
                sl_dist = abs(bracket.fill_price - bracket.stop_price)
                if sl_dist <= tp_dist:
                    # SL hit → full position already closed on exchange
                    for k in self._bracket_keys(sid):
                        self._brackets.pop(k, None)
                    self._positions.pop(sid, None)
                    logger.info(f"[TT] SL already filled for {sid}, skipping close")
                    return None
        await self._cancel_bracket(sid)

        # Re-check: cancel may have discovered stop filled
        for key in self._bracket_keys(sid):
            bracket = self._brackets.get(key)
            if bracket and bracket.filled:
                for k in self._bracket_keys(sid):
                    self._brackets.pop(k, None)
                self._positions.pop(sid, None)
                logger.info(f"[TT] Stop filled during close for {sid}, skipping market close")
                return None

        pos = self._positions.pop(sid, None)
        if pos is None:
            logger.warning(f"[TT] No position to close for {sid}")
            return None

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

        try:
            response = await self._account.place_order(
                self._session, order, dry_run=False
            )
        except Exception as e:
            logger.critical(f"[TT] Market close order FAILED for {sid}: {e} — reconciliation will detect")
            raise

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

    async def partial_close_position(
        self,
        instrument: str,
        qty: int,
        price_hint: float,
        strategy_id: str = "",
    ) -> Optional[FillResult]:
        """Close partial position via market order. Reduces tracked qty.

        Used as fallback when OCO #2 failed and bar-close TP2 triggers
        for the runner. Also used for any engine-managed partial exit.

        Args:
            instrument: "MNQ" or "MES"
            qty: Number of contracts to close (must be < position qty)
            price_hint: Current price for fallback
            strategy_id: Strategy identifier for position lookup
        """
        sid = strategy_id or instrument
        pos = self._positions.get(sid)
        if pos is None:
            logger.warning(f"[TT] No position for partial close: {sid}")
            return None

        future = self._futures[instrument]
        close_action = OrderAction.SELL if pos["side"] == "long" else OrderAction.BUY

        from decimal import Decimal
        leg = future.build_leg(Decimal(str(qty)), close_action)

        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            legs=[leg],
        )

        close_side = "sell" if pos["side"] == "long" else "buy"
        logger.info(
            f"[TT] Submitting PARTIAL CLOSE {close_action.value} "
            f"{qty}x {instrument} strategy={sid} "
            f"(remaining will be {pos['qty'] - qty})"
        )

        response = await self._account.place_order(
            self._session, order, dry_run=False
        )

        fill_price = await self._wait_for_fill(response.order, price_hint)

        # Reduce tracked qty (don't pop position — runner still has contracts)
        pos["qty"] -= qty

        self._order_count += 1
        order_id = f"TT-{self._order_count:06d}"

        logger.info(
            f"[TT] PARTIAL CLOSE FILLED {order_id}: {close_side.upper()} "
            f"{qty}x {instrument} @ {fill_price:.2f} strategy={sid} "
            f"(remaining={pos['qty']})"
        )

        return FillResult(
            order_id=order_id,
            instrument=instrument,
            side=close_side,
            qty=qty,
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
            qty=qty,
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
        tag: str = "",
    ) -> None:
        """Place an OCO bracket with both stop loss and take profit.

        Used by v15 tp_scalp strategy. When one leg fills, the other is
        automatically cancelled by the exchange.

        For long: STOP SELL at entry - stop_pts, LIMIT SELL at entry + tp_pts
        For short: STOP BUY at entry + stop_pts, LIMIT BUY at entry - tp_pts

        Args:
            tag: "tp1", "tp2", or "" (single bracket). Used as composite key
                 suffix for multi-bracket strategies (e.g. "MES_V2__tp1").
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

        key = f"{sid}__{tag}" if tag else sid
        bracket = BracketState(
            instrument=instrument,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            complex_order_id=response.complex_order.id if response.complex_order else None,
            is_oco=True,
            tp_price=tp_price,
            tag=tag,
            qty=qty,
        )
        self._brackets[key] = bracket

        tag_label = f" [{tag}]" if tag else ""
        logger.info(
            f"[TT] OCO bracket placed: {sid}{tag_label} ({instrument}) "
            f"{qty}x TP={tp_price:.2f} SL={stop_price:.2f}"
        )

    async def _place_dual_oco(
        self,
        instrument: str,
        side: str,
        qty: int,
        fill_price: float,
        strat: StrategyConfig,
        sid: str,
    ) -> None:
        """Place two independent OCO brackets for partial-exit strategies.

        OCO #1 (tp1): partial_qty contracts at partial_tp_pts
        OCO #2 (tp2): remaining contracts at tp_pts

        Each OCO is self-contained with its own SL leg. Both survive engine
        crashes independently.
        """
        partial_qty = strat.partial_qty
        runner_qty = qty - partial_qty

        # OCO #1: TP1 partial
        try:
            await self.place_oco_bracket(
                instrument, side, partial_qty, fill_price,
                strat.max_loss_pts, strat.partial_tp_pts,
                strategy_id=sid, tag="tp1",
            )
        except Exception as e:
            logger.critical(
                f"[TT] OCO #1 (tp1) FAILED for {sid}: {e} — "
                f"falling back to single stop for all {qty} contracts"
            )
            try:
                await self._place_stop(
                    instrument, side, qty, fill_price,
                    strat.max_loss_pts, strategy_id=sid,
                )
                self._oco_failed_sids.add(sid)
            except Exception as e2:
                self._protection_failed_sids.add(sid)
                logger.critical(
                    f"[TT] BOTH OCO and fallback stop FAILED for {sid}: {e2} — "
                    f"position has NO exchange protection, engine should halt"
                )
            return

        # OCO #2: TP2 runner
        try:
            await self.place_oco_bracket(
                instrument, side, runner_qty, fill_price,
                strat.max_loss_pts, strat.tp_pts,
                strategy_id=sid, tag="tp2",
            )
        except Exception as e:
            logger.critical(
                f"[TT] OCO #2 (tp2) FAILED for {sid}: {e} — "
                f"OCO #1 (tp1) still protecting {partial_qty} contracts. "
                f"Placing simple stop for runner ({runner_qty} contracts)."
            )
            try:
                await self._place_stop(
                    instrument, side, runner_qty, fill_price,
                    strat.max_loss_pts, strategy_id=sid,
                )
                # Mark OCO failed so bar-close TP re-enables for runner
                self._oco_failed_sids.add(sid)
            except Exception as e2:
                logger.critical(
                    f"[TT] Runner stop ALSO failed for {sid}: {e2} — "
                    f"runner ({runner_qty} contracts) has NO exchange protection. "
                    f"OCO #1 still protects {partial_qty} contracts."
                )
                self._protection_failed_sids.add(sid)

    async def _cancel_bracket(self, strategy_id: str) -> None:
        """Cancel all resting stop/bracket orders for a strategy.

        Handles both single keys ("MNQ_V15") and composite keys
        ("MES_V2__tp1", "MES_V2__tp2").
        """
        keys = self._bracket_keys(strategy_id)
        if not keys:
            return

        for key in keys:
            bracket = self._brackets.get(key)
            if bracket is None or bracket.filled:
                self._brackets.pop(key, None)
                continue

            try:
                if bracket.is_oco and bracket.complex_order_id is not None:
                    await self._account.delete_complex_order(
                        self._session, bracket.complex_order_id
                    )
                elif bracket.order_id is not None:
                    await self._account.delete_order(
                        self._session, bracket.order_id
                    )
                self._brackets.pop(key, None)  # Only pop on success
                logger.info(f"[TT] Resting stop cancelled for {key}")
            except Exception as e:
                if bracket.filled:
                    self._brackets.pop(key, None)
                    logger.info(f"[TT] Stop for {key} filled during cancel @ {bracket.fill_price:.2f}")
                else:
                    # Do NOT pop -- the stop order may still be live on the exchange.
                    # Leave bracket for _process_alert_fill or reconciliation to handle.
                    logger.critical(
                        f"[TT] Failed to cancel stop for {key}, NOT confirmed filled, "
                        f"bracket KEPT for reconciliation: {e}"
                    )

    async def check_bracket_fills(self, strategy_id: str) -> list[dict]:
        """Check if resting brackets were triggered on the exchange.

        Non-blocking -- reads from the queue populated by AlertStreamer.
        Call this before processing each bar in the runner.

        Returns a list of fill dicts (empty if nothing filled). For multi-bracket
        strategies (partial exit), multiple fills may arrive in the same poll
        cycle (e.g. both SL legs on a gap down).

        Each fill dict contains:
            instrument, strategy_id, type, price, side, tag, qty
        Where type is one of: "take_profit_partial", "take_profit", "stop_loss"
        """
        # Drain the queue, collecting fills for this strategy_id
        pending = []
        results = []

        while not self._bracket_fill_queue.empty():
            try:
                fill_info = self._bracket_fill_queue.get_nowait()
                if fill_info.get("strategy_id") == strategy_id:
                    results.append(fill_info)
                else:
                    pending.append(fill_info)
            except asyncio.QueueEmpty:
                break

        # Put back fills for other strategies
        for item in pending:
            await self._bracket_fill_queue.put(item)

        # Process each fill: update bracket and position state
        for fill_info in results:
            fill_type = fill_info["type"]
            bracket_key = fill_info.get("bracket_key", strategy_id)

            if fill_type == "take_profit_partial":
                # Partial TP1: remove only this bracket.
                # pos["qty"] already reduced in _process_alert_fill().
                self._brackets.pop(bracket_key, None)
            elif fill_type in ("stop_loss", "take_profit"):
                # Full exit: remove ALL brackets for this strategy, pop position
                # (SL dedup: position may already be popped by prior full fill)
                pos = self._positions.get(strategy_id)
                if pos is None:
                    logger.warning(
                        f"[TT] Duplicate fill for {strategy_id} "
                        f"(position already closed), skipping"
                    )
                    fill_info["_duplicate"] = True
                    continue
                for key in self._bracket_keys(strategy_id):
                    self._brackets.pop(key, None)
                self._positions.pop(strategy_id, None)

        # Remove duplicate fills (SL dedup)
        results = [f for f in results if not f.get("_duplicate")]

        return results

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
        """Check if a filled order matches an active bracket.

        Matches by simple order_id OR by complex_order_id (OCO brackets).
        Handles composite keys (e.g. "MES_V2__tp1") for multi-bracket strategies.

        Fill type classification for tagged brackets:
          - tp1 tag + TP hit → "take_profit_partial" (position stays open)
          - tp2 tag + TP hit → "take_profit" (full close)
          - Any tag + SL hit → "stop_loss" (full close)
          - No tag → existing single-bracket logic
        """
        for bracket_key, bracket in list(self._brackets.items()):
            if bracket.filled:
                continue

            # Match by simple order ID or OCO complex order ID
            matched = False
            if bracket.order_id is not None and order.id == bracket.order_id:
                matched = True
            elif (bracket.is_oco and bracket.complex_order_id is not None
                  and hasattr(order, 'complex_order_id')
                  and order.complex_order_id == bracket.complex_order_id):
                matched = True

            if matched:
                fill_price = self._extract_fill_price(order)
                bracket.filled = True
                bracket.fill_price = fill_price

                # Extract base strategy_id from composite key
                sid = bracket_key.split("__")[0] if "__" in bracket_key else bracket_key

                # Determine fill type: TP or SL based on price proximity
                if bracket.is_oco and bracket.tp_price != 0:
                    tp_dist = abs(fill_price - bracket.tp_price)
                    sl_dist = abs(fill_price - bracket.stop_price)
                    is_tp = tp_dist <= sl_dist
                else:
                    is_tp = False

                # Classify based on tag
                if is_tp and bracket.tag == "tp1":
                    fill_type = "take_profit_partial"
                elif is_tp:
                    fill_type = "take_profit"
                else:
                    fill_type = "stop_loss"

                # Position cleanup is handled by check_bracket_fills(),
                # NOT here. _process_alert_fill only marks brackets filled
                # and enqueues fill info. This avoids the bug where popping
                # the position here causes check_bracket_fills() to see
                # pos=None and mark fills as duplicates.
                #
                # Exception: partial TP reduces pos["qty"] immediately so
                # close_position() sees correct remaining qty if called
                # between AlertStreamer fill and next check_bracket_fills().
                if fill_type == "take_profit_partial":
                    pos = self._positions.get(sid)
                    if pos:
                        pos["qty"] -= bracket.qty
                elif fill_type == "stop_loss":
                    # SL dedup for multi-bracket: check if another bracket
                    # already queued a full-exit fill for this strategy
                    # (both SL legs fire on gap-down). Use bracket.filled
                    # on sibling brackets as the dedup signal.
                    sibling_keys = [k for k in self._bracket_keys(sid)
                                    if k != bracket_key]
                    for sib_key in sibling_keys:
                        sib = self._brackets.get(sib_key)
                        if sib and sib.filled and sib.fill_price != 0:
                            # Sibling already processed as SL — skip this one
                            sib_tp = abs(sib.fill_price - sib.tp_price) if sib.tp_price else float('inf')
                            sib_sl = abs(sib.fill_price - sib.stop_price)
                            if sib_sl <= sib_tp:
                                logger.warning(
                                    f"[TT] Duplicate SL fill for {bracket_key} — "
                                    f"sibling {sib_key} already filled as SL, skipping"
                                )
                                return

                await self._bracket_fill_queue.put({
                    "instrument": bracket.instrument,
                    "strategy_id": sid,
                    "bracket_key": bracket_key,
                    "type": fill_type,
                    "price": fill_price,
                    "side": bracket.side,
                    "tag": bracket.tag,
                    "qty": bracket.qty,
                })

                tag_label = f" [{bracket.tag}]" if bracket.tag else ""
                type_label = {"take_profit_partial": "TP1 PARTIAL",
                              "take_profit": "TP", "stop_loss": "SL"}[fill_type]
                logger.info(
                    f"[TT] {type_label} TRIGGERED: "
                    f"{sid}{tag_label} ({bracket.instrument}) "
                    f"{bracket.qty}x @ {fill_price:.2f} (was {bracket.side})"
                )
                return

        # No bracket matched -- warn if OCO brackets are active
        active_oco = [s for s, b in self._brackets.items()
                      if b.is_oco and not b.filled]
        if active_oco:
            logger.warning(
                f"[TT] Unmatched fill (order {order.id}, "
                f"complex_oid={getattr(order, 'complex_order_id', None)}) "
                f"while OCO brackets active: {active_oco}"
            )

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

    def pop_oco_failures(self) -> set:
        """Return and clear the set of strategy IDs where OCO placement failed."""
        failed = self._oco_failed_sids.copy()
        self._oco_failed_sids.clear()
        return failed

    def pop_protection_failures(self) -> set:
        """Return and clear strategy IDs where ALL protection failed (OCO + stop)."""
        failed = self._protection_failed_sids.copy()
        self._protection_failed_sids.clear()
        return failed
