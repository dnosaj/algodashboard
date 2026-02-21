"""
Tastytrade data feed for the live trading engine.

Streams real-time 1-min OHLCV candles via DXLinkStreamer (WebSocket)
and provides warmup (historical) bars for indicator initialization.

Architecture:
  - Background task runs DXLinkStreamer, receiving candles as they form
  - Completed candles are buffered in per-instrument queues
  - get_latest_bar() pops the next completed bar (called by runner)
  - get_warmup_bars() returns historical bars from initial backfill

Data flow:
  1. connect() starts background stream task
  2. subscribe_candle() with start_time in the past -> historical backfill
  3. Historical candles arrive rapidly, buffered in _all_bars
  4. After backfill, live candles continue flowing in
  5. When a new timestamp appears, the previous bar is marked complete
  6. get_latest_bar() returns the next completed bar in sequence

Requirements:
  pip install tastytrade
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

from .events import Bar

logger = logging.getLogger(__name__)

try:
    from tastytrade import Session, DXLinkStreamer
    from tastytrade.dxfeed import Candle, Quote
    TASTYTRADE_AVAILABLE = True
except ImportError:
    TASTYTRADE_AVAILABLE = False


class TastytradeDataFeed:
    """Real-time 1-min candle data feed from tastytrade/DXLink.

    Drop-in replacement for MockDataFeed in runner.py.
    Same method signatures: connect, disconnect, get_warmup_bars,
    get_latest_bar, connected property.
    """

    def __init__(
        self,
        session: "Session",
        instruments: list[str],
        contract_symbols: dict[str, str],
        warmup_bars: int = 500,
    ) -> None:
        """
        Args:
            session: Authenticated tastytrade Session.
            instruments: List of instrument names (e.g., ["MNQ", "MES"]).
            contract_symbols: Mapping from instrument to contract symbol
                (e.g., {"MNQ": "/MNQH6"}).
            warmup_bars: Number of historical bars to load on connect.
        """
        if not TASTYTRADE_AVAILABLE:
            raise ImportError(
                "tastytrade package not installed. Run: pip install tastytrade"
            )

        self._session = session
        self._instruments = instruments
        self._contract_symbols = contract_symbols
        self._warmup_count = warmup_bars

        # Streamer symbols (resolved from trading symbols in connect())
        # DXLink requires streamer symbols like '/MNQH26:XCME', not '/MNQH6'
        self._streamer_symbols: dict[str, str] = {}

        # Completed bars: instrument -> deque of Bar (chronological)
        self._all_bars: dict[str, deque] = {
            inst: deque(maxlen=50000) for inst in instruments
        }

        # The currently forming (incomplete) bar per instrument
        self._current_bar: dict[str, Optional[Bar]] = {
            inst: None for inst in instruments
        }

        # Queue of completed bars not yet consumed by get_latest_bar()
        self._pending_bars: dict[str, deque] = {
            inst: deque() for inst in instruments
        }

        # Async queue for event-driven bar delivery (replaces polling)
        self._bar_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Async queue for real-time quote streaming (intra-bar exits)
        self._quote_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        # Background stream task
        self._stream_task: Optional[asyncio.Task] = None
        self._connected = False
        self._warmup_complete = asyncio.Event()
        self._last_prices: dict[str, float] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def bar_queue(self) -> asyncio.Queue:
        """Async queue of completed bars for event-driven processing."""
        return self._bar_queue

    @property
    def quote_queue(self) -> asyncio.Queue:
        """Async queue of real-time quotes for intra-bar exit monitoring."""
        return self._quote_queue

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Start the candle stream and wait for warmup bars to load."""
        logger.info(
            f"[TT Feed] Starting data feed for "
            f"{', '.join(self._instruments)}..."
        )

        # Resolve streamer symbols from trading symbols
        # DXLink uses different symbol format: /MNQH6 -> /MNQH26:XCME
        from tastytrade.instruments import Future
        for inst, contract in self._contract_symbols.items():
            future = await Future.get(self._session, contract)
            self._streamer_symbols[inst] = future.streamer_symbol
            logger.info(
                f"[TT Feed] {inst}: {contract} -> {future.streamer_symbol}"
            )

        self._stream_task = asyncio.create_task(
            self._stream_loop(),
            name="tt_candle_stream",
        )
        self._connected = True

        # Wait for enough warmup bars to accumulate
        logger.info(
            f"[TT Feed] Waiting for {self._warmup_count} warmup bars..."
        )
        timeout = 60  # seconds
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            min_bars = min(
                len(self._all_bars[inst]) for inst in self._instruments
            )
            if min_bars >= self._warmup_count:
                self._warmup_complete.set()
                break
            await asyncio.sleep(0.5)

        if not self._warmup_complete.is_set():
            # Partial warmup -- use what we have
            min_bars = min(
                len(self._all_bars[inst]) for inst in self._instruments
            )
            logger.warning(
                f"[TT Feed] Warmup timeout: got {min_bars}/"
                f"{self._warmup_count} bars"
            )
            self._warmup_complete.set()
        else:
            logger.info("[TT Feed] Warmup complete")

    async def disconnect(self) -> None:
        """Stop the candle stream."""
        self._connected = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        logger.info("[TT Feed] Disconnected")

    # ------------------------------------------------------------------
    # Data access (called by runner.py)
    # ------------------------------------------------------------------

    def get_warmup_bars(self, instrument: str, count: int) -> list[Bar]:
        """Return the most recent `count` completed bars for warmup.

        Called once at startup to initialize SM and RSI indicators.
        Returns bars ordered oldest-first (chronological).
        Historical bars may arrive in reverse order, so we sort here.
        """
        all_completed = sorted(
            self._all_bars.get(instrument, []),
            key=lambda b: b.timestamp,
        )
        warmup = all_completed[-count:] if len(all_completed) >= count else all_completed
        logger.info(
            f"[TT Feed] Warmup for {instrument}: "
            f"{len(warmup)}/{count} bars available"
        )
        return warmup

    async def get_latest_bar(self, instrument: str) -> Optional[Bar]:
        """Return the next completed bar not yet consumed.

        Called every ~60s by the runner polling loop.
        Returns None if no new completed bars available.

        NOTE: Multiple bars may complete between polls. This returns one
        at a time. The runner should call repeatedly until None to catch up.
        For the current architecture (1 bar per poll), this returns the
        most recent completed bar and discards any in between.
        """
        pending = self._pending_bars.get(instrument, deque())

        if not pending:
            return None

        # Return the most recent completed bar (skip intermediates)
        # The runner processes one bar per poll cycle, so we want the
        # latest completed bar to keep indicators current
        bar = None
        while pending:
            bar = pending.popleft()

        return bar

    # ------------------------------------------------------------------
    # Background streaming
    # ------------------------------------------------------------------

    async def _stream_loop(self) -> None:
        """Background task: stream 1-min candles AND real-time quotes.

        Uses a single DXLinkStreamer with both candle and quote subscriptions.
        Candle listener handles bar construction; quote listener feeds
        the intra-bar exit monitor with real-time bid/ask prices.
        """
        retry_count = 0
        max_retries = 10

        while self._connected and retry_count < max_retries:
            try:
                async with DXLinkStreamer(self._session) as streamer:
                    # Subscribe to 1-min candles starting from warmup period ago
                    lookback_minutes = int(self._warmup_count * 1.5)
                    start_time = datetime.now(timezone.utc) - timedelta(
                        minutes=lookback_minutes
                    )

                    symbols = list(self._streamer_symbols.values())
                    await streamer.subscribe_candle(
                        symbols=symbols,
                        interval="1m",
                        start_time=start_time,
                        extended_trading_hours=True,
                    )

                    # Subscribe to real-time quotes for intra-bar exit monitoring
                    await streamer.subscribe(Quote, symbols)

                    retry_count = 0  # Reset on success
                    logger.info(
                        f"[TT Feed] Candle + Quote stream connected "
                        f"(streamer symbols: {symbols})"
                    )

                    # Run candle and quote listeners concurrently
                    candle_task = asyncio.create_task(
                        self._candle_listen(streamer),
                        name="tt_candle_listen",
                    )
                    quote_task = asyncio.create_task(
                        self._quote_listen(streamer),
                        name="tt_quote_listen",
                    )
                    await asyncio.gather(candle_task, quote_task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                retry_count += 1
                wait = min(2 ** retry_count, 30)
                logger.error(
                    f"[TT Feed] Stream error (retry {retry_count}/"
                    f"{max_retries}): {e}. Reconnecting in {wait}s..."
                )
                if self._connected:
                    await asyncio.sleep(wait)

        if retry_count >= max_retries:
            logger.error("[TT Feed] Max retries reached, feed stopped")
            self._connected = False

    async def _candle_listen(self, streamer: "DXLinkStreamer") -> None:
        """Listen for candle events from the streamer."""
        async for candle in streamer.listen(Candle):
            if not self._connected:
                break
            self._process_candle(candle)

    async def _quote_listen(self, streamer: "DXLinkStreamer") -> None:
        """Listen for quote events and push to the quote queue."""
        async for quote in streamer.listen(Quote):
            if not self._connected:
                break
            try:
                self._quote_queue.put_nowait(quote)
            except asyncio.QueueFull:
                # Drop oldest quote to make room (quotes are ephemeral)
                try:
                    self._quote_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self._quote_queue.put_nowait(quote)

    def _process_candle(self, candle: "Candle") -> None:
        """Convert a DXLink Candle event to a Bar and buffer it.

        DXLink sends data in two phases:
          1. Historical backfill: bars arrive in REVERSE order (newest first)
          2. Live updates: current bar updates, then new bar starts (forward)

        Logic:
          - If bar.timestamp > current: forward progress, current is COMPLETE
          - If bar.timestamp < current: historical backfill bar (already complete)
          - If bar.timestamp == current: in-progress update of forming bar
        """
        # Map event_symbol back to instrument name
        instrument = self._resolve_instrument(candle.event_symbol)
        if instrument is None:
            return

        # Build Bar from candle
        try:
            ts = datetime.fromtimestamp(
                candle.time / 1000.0, tz=timezone.utc
            )
            o = float(candle.open)
            h = float(candle.high)
            l = float(candle.low)
            c = float(candle.close)
            v = float(candle.volume) if candle.volume else 0.0

            # Skip empty boundary bars (DXLink sends O/H/L/C=0 at start_time)
            if o == 0 and h == 0 and l == 0 and c == 0:
                return

            bar = Bar(
                timestamp=ts,
                open=o, high=h, low=l, close=c,
                volume=v,
                instrument=instrument,
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"[TT Feed] Bad candle data: {e}")
            return

        # Track last price
        self._last_prices[instrument] = bar.close

        current = self._current_bar[instrument]

        if current is None:
            # First candle received
            self._current_bar[instrument] = bar
        elif bar.timestamp > current.timestamp:
            # Forward progress -- previous bar is COMPLETE
            self._all_bars[instrument].append(current)
            self._pending_bars[instrument].append(current)
            self._current_bar[instrument] = bar
            # Push to async queue for event-driven processing (skip during warmup)
            if self._warmup_complete.is_set():
                try:
                    self._bar_queue.put_nowait(current)
                except asyncio.QueueFull:
                    # Drop oldest bar to make room
                    try:
                        self._bar_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._bar_queue.put_nowait(current)
        elif bar.timestamp < current.timestamp:
            # Historical backfill (reverse order) -- this bar is complete
            self._all_bars[instrument].append(bar)
        else:
            # Same timestamp -- update the forming bar
            self._current_bar[instrument] = bar

    def resolve_instrument(self, event_symbol: Optional[str]) -> Optional[str]:
        """Map a DXLink event_symbol back to instrument name.

        DXLink event symbols look like: '/MNQH26:XCME{=1m,tho=true}'
        We check if our streamer symbol is a prefix of the event symbol.
        """
        if event_symbol is None:
            return None

        for inst, streamer_sym in self._streamer_symbols.items():
            if event_symbol.startswith(streamer_sym):
                return inst

        return None

    # Keep private alias for backward compat within this file
    _resolve_instrument = resolve_instrument
