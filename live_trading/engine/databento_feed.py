"""
Databento data feed for the live trading engine.

Primary data feed — streams real-time 1-min OHLCV candles and top-of-book
quotes via Databento's CME GLBX.MDP3 feed. Independent of dxFeed/DXLink.
Drop-in replacement for TastytradeDataFeed (same interface).

Architecture:
  - Uses databento.Live client for real-time streaming (callback-based)
  - Uses databento.Historical client for warmup bar backfill
  - Streams both ohlcv-1m (completed bars) and mbp-1 (top-of-book quotes)
  - Completed bars pushed to bar_queue for the event-driven bar loop
  - Quotes pushed to quote_queue for intra-bar exit monitoring

Data flow:
  1. connect() fetches historical warmup bars via Historical API
  2. Live client subscribes to ohlcv-1m + mbp-1 for MNQ.c.0 / MES.c.0
  3. SymbolMappingMsg maps instrument_id -> instrument name (MNQ, MES)
  4. OHLCVMsg records converted to Bar and pushed to bar_queue
  5. MBP1Msg records converted to QuoteTick and pushed to quote_queue
  6. runner.py consumes bar_queue + quote_queue (same as TastytradeDataFeed)

Symbols:
  - MNQ -> MNQ.c.0 (continuous front-month micro Nasdaq)
  - MES -> MES.c.0 (continuous front-month micro S&P)

Requirements:
  pip install databento
"""

import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .events import Bar

logger = logging.getLogger(__name__)

try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False

# Databento fixed-point price scale: 1 unit = 1e-9 (nanodollar)
PRICE_SCALE = 1e-9

# Undefined price sentinel in Databento (2^63 - 1)
UNDEF_PRICE = 2**63 - 1

# Continuous front-month symbol mapping
SYMBOL_MAP = {
    "MNQ": "MNQ.c.0",
    "MES": "MES.c.0",
    "NQ": "NQ.c.0",
    "ES": "ES.c.0",
    "MYM": "MYM.c.0",
}

# Reverse lookup: databento symbol -> instrument name
REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}


@dataclass
class QuoteTick:
    """Lightweight quote wrapper matching the interface runner.py expects.

    runner.py accesses: quote.event_symbol, quote.bid_price, quote.ask_price
    (see intra_bar_exit_loop in runner.py, lines 822-825).
    """
    event_symbol: str   # Databento symbol (e.g. "MNQ.c.0") or instrument name
    bid_price: float
    ask_price: float


class DabentoDataFeed:
    """Live data feed using Databento's real-time CME streaming.

    Primary feed -- replaces DXLink. Same interface as TastytradeDataFeed.
    Databento streams directly from CME's GLBX.MDP3 -- independent of dxFeed.

    Interface (drop-in for TastytradeDataFeed):
      - async connect()              Start streaming, wait for warmup
      - async disconnect()           Stop streaming
      - get_warmup_bars(inst, n)     Historical bars for SM/RSI warmup
      - async get_latest_bar(inst)   Next completed bar (or None)
      - bar_queue                    asyncio.Queue of completed Bar objects
      - quote_queue                  asyncio.Queue of QuoteTick objects
      - connected                    bool property
      - resolve_instrument(symbol)   Map symbol -> instrument name
    """

    def __init__(
        self,
        instruments: list[str],
        api_key: Optional[str] = None,
        warmup_bars: int = 500,
    ) -> None:
        """
        Args:
            instruments: List of instrument names (e.g., ["MNQ", "MES"]).
            api_key: Databento API key. If None, reads DATABENTO_API_KEY env var.
            warmup_bars: Number of historical bars to load on connect.
        """
        if not DATABENTO_AVAILABLE:
            raise ImportError(
                "databento package not installed. Run: pip install databento"
            )

        self._instruments = instruments
        self._api_key = api_key or os.environ.get("DATABENTO_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "No Databento API key. Set DATABENTO_API_KEY env var "
                "or pass api_key parameter."
            )
        self._warmup_count = warmup_bars

        # Databento symbols for subscription
        self._db_symbols = [
            SYMBOL_MAP[inst] for inst in instruments if inst in SYMBOL_MAP
        ]

        # instrument_id -> instrument name mapping (populated by SymbolMappingMsg)
        self._id_to_instrument: dict[int, str] = {}

        # Also keep a direct symbol -> instrument mapping for resolve_instrument
        self._symbol_to_instrument: dict[str, str] = {}
        for inst in instruments:
            db_sym = SYMBOL_MAP.get(inst, "")
            if db_sym:
                self._symbol_to_instrument[db_sym] = inst

        # Completed bars: instrument -> deque of Bar (chronological)
        self._all_bars: dict[str, deque] = {
            inst: deque(maxlen=50000) for inst in instruments
        }

        # Queue of completed bars not yet consumed by get_latest_bar()
        self._pending_bars: dict[str, deque] = {
            inst: deque() for inst in instruments
        }

        # Async queue for event-driven bar delivery (replaces polling)
        self._bar_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Async queue for real-time quote streaming (intra-bar exits)
        self._quote_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        # Live client and background thread
        self._live_client: Optional[Any] = None
        self._live_thread: Optional[threading.Thread] = None
        self._connected = False
        self._warmup_complete = asyncio.Event()
        self._last_prices: dict[str, float] = {}

        # Event loop reference for cross-thread queue puts
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Shutdown flag
        self._shutting_down = False

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
        """Start the data feed: fetch warmup bars, then start live stream."""
        logger.info(
            f"[DB Feed] Starting Databento feed for "
            f"{', '.join(self._instruments)}..."
        )

        # Store event loop reference for cross-thread queue operations
        self._loop = asyncio.get_running_loop()

        # --- Phase 1: Fetch historical warmup bars ---
        await self._fetch_warmup_bars()

        # --- Phase 2: Start live streaming ---
        self._start_live_stream()

        self._connected = True
        self._warmup_complete.set()
        logger.info("[DB Feed] Connected and warmup complete")

    async def disconnect(self) -> None:
        """Stop the live stream and clean up."""
        logger.info("[DB Feed] Disconnecting...")
        self._shutting_down = True
        self._connected = False

        if self._live_client:
            try:
                self._live_client.terminate()
            except Exception as e:
                logger.warning(f"[DB Feed] Error terminating live client: {e}")

        if self._live_thread and self._live_thread.is_alive():
            self._live_thread.join(timeout=5.0)

        self._live_client = None
        self._live_thread = None
        logger.info("[DB Feed] Disconnected")

    # ------------------------------------------------------------------
    # Data access (called by runner.py)
    # ------------------------------------------------------------------

    def get_warmup_bars(self, instrument: str, count: int) -> list[Bar]:
        """Return the most recent `count` completed bars for warmup.

        Called once at startup to initialize SM and RSI indicators.
        Returns bars ordered oldest-first (chronological).
        """
        all_completed = sorted(
            self._all_bars.get(instrument, []),
            key=lambda b: b.timestamp,
        )
        warmup = (
            all_completed[-count:]
            if len(all_completed) >= count
            else all_completed
        )
        logger.info(
            f"[DB Feed] Warmup for {instrument}: "
            f"{len(warmup)}/{count} bars available"
        )
        return warmup

    async def get_latest_bar(self, instrument: str) -> Optional[Bar]:
        """Return the next completed bar not yet consumed.

        Returns None if no new completed bars available.
        Returns the most recent completed bar (skips intermediates).
        """
        pending = self._pending_bars.get(instrument, deque())

        if not pending:
            return None

        bar = None
        while pending:
            bar = pending.popleft()

        return bar

    def resolve_instrument(self, event_symbol: Optional[str]) -> Optional[str]:
        """Map an event symbol or instrument name back to instrument name.

        Compatible with runner.py's intra_bar_exit_loop which calls:
          instrument = state.data_feed.resolve_instrument(quote.event_symbol)

        For Databento quotes, event_symbol is already the instrument name
        (we set it in _on_quote), so this is a simple lookup/passthrough.
        """
        if event_symbol is None:
            return None

        # Direct instrument name (set by _on_quote)
        if event_symbol in self._instruments:
            return event_symbol

        # Databento symbol lookup
        return self._symbol_to_instrument.get(event_symbol)

    # Keep private alias for backward compat
    _resolve_instrument = resolve_instrument

    # ------------------------------------------------------------------
    # Historical warmup
    # ------------------------------------------------------------------

    async def _fetch_warmup_bars(self) -> None:
        """Fetch historical 1-min bars for indicator warmup.

        Uses databento.Historical to get recent bars, same API as
        fetch_databento_data.py in the backtesting engine.
        """
        logger.info(
            f"[DB Feed] Fetching {self._warmup_count} warmup bars "
            f"per instrument..."
        )

        # Request extra bars to account for gaps (weekends, holidays, maintenance)
        request_minutes = int(self._warmup_count * 2.5)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=request_minutes)

        try:
            hist_client = db.Historical(self._api_key)

            data = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: hist_client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=self._db_symbols,
                    schema="ohlcv-1m",
                    stype_in="continuous",
                    start=start_time.strftime("%Y-%m-%dT%H:%M"),
                    end=end_time.strftime("%Y-%m-%dT%H:%M"),
                ),
            )

            df = data.to_df()

            if len(df) == 0:
                logger.warning(
                    "[DB Feed] No historical bars returned for warmup. "
                    "Indicators will start cold."
                )
                return

            # Detect fixed-point pricing
            raw_open = df["open"].values
            divisor = self._detect_price_scale(raw_open)

            # The DataFrame index is a DatetimeIndex (UTC).
            # Group bars by instrument using the 'symbol' column if present,
            # otherwise assume single instrument.
            if "symbol" in df.columns:
                grouped = df.groupby("symbol")
            else:
                # Single instrument -- assign the first instrument
                grouped = [(self._db_symbols[0], df)]

            for db_symbol, group_df in grouped:
                if isinstance(db_symbol, str):
                    instrument = REVERSE_SYMBOL_MAP.get(
                        db_symbol, self._instruments[0]
                    )
                else:
                    instrument = self._instruments[0]

                count = 0
                for idx, row in group_df.iterrows():
                    ts = idx.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    bar = Bar(
                        timestamp=ts,
                        open=float(row["open"]) / divisor,
                        high=float(row["high"]) / divisor,
                        low=float(row["low"]) / divisor,
                        close=float(row["close"]) / divisor,
                        volume=float(row["volume"]),
                        instrument=instrument,
                    )

                    # Skip zero-price bars (market closed / gaps)
                    if bar.open == 0 and bar.close == 0:
                        continue

                    self._all_bars[instrument].append(bar)
                    count += 1

                # Track last price
                if count > 0:
                    last_bar = self._all_bars[instrument][-1]
                    self._last_prices[instrument] = last_bar.close

                logger.info(
                    f"[DB Feed] Warmup loaded: {instrument} = {count} bars"
                )

        except Exception as e:
            logger.error(
                f"[DB Feed] Historical warmup failed: {e}. "
                f"Indicators will start cold.",
                exc_info=True,
            )

    @staticmethod
    def _detect_price_scale(prices) -> float:
        """Detect if prices are fixed-point (scaled by 1e9).

        Returns divisor: 1e9 if fixed-point, 1.0 if normal.
        Same logic as fetch_databento_data.py.
        """
        if len(prices) == 0:
            return 1.0
        sample = prices[:100]
        median_price = sorted(sample)[len(sample) // 2]
        if median_price > 1e6:
            return 1e9
        return 1.0

    # ------------------------------------------------------------------
    # Live streaming
    # ------------------------------------------------------------------

    def _start_live_stream(self) -> None:
        """Start the Databento Live client in a background thread.

        The Live client uses a callback model (add_callback + block_for_close).
        We run block_for_close() in a daemon thread so it doesn't block the
        asyncio event loop. Callbacks push records to asyncio queues via
        loop.call_soon_threadsafe().
        """
        self._live_thread = threading.Thread(
            target=self._live_stream_worker,
            name="databento_live_stream",
            daemon=True,
        )
        self._live_thread.start()

    def _live_stream_worker(self) -> None:
        """Background thread: run the Databento Live client with reconnection.

        Infinite retry loop with exponential backoff, same pattern as the
        TastytradeDataFeed._stream_loop() but more reliable since it's
        not dependent on dxFeed.
        """
        retry_count = 0
        max_retries = 999

        while not self._shutting_down and retry_count < max_retries:
            try:
                self._live_client = db.Live(key=self._api_key)

                # Subscribe to 1-min OHLCV bars
                self._live_client.subscribe(
                    dataset="GLBX.MDP3",
                    schema="ohlcv-1m",
                    stype_in="continuous",
                    symbols=self._db_symbols,
                )

                # Subscribe to top-of-book quotes (MBP-1) for intra-bar exits
                self._live_client.subscribe(
                    dataset="GLBX.MDP3",
                    schema="mbp-1",
                    stype_in="continuous",
                    symbols=self._db_symbols,
                )

                # Register callback for all record types
                self._live_client.add_callback(self._on_record)

                retry_count = 0  # Reset on successful connection
                logger.info(
                    f"[DB Feed] Live stream connected "
                    f"(symbols: {self._db_symbols})"
                )

                # Block until connection closes (runs in background thread)
                self._live_client.block_for_close()

                # If we get here, connection was closed normally
                if self._shutting_down:
                    break
                logger.warning("[DB Feed] Live connection closed, reconnecting...")

            except Exception as e:
                if self._shutting_down:
                    break
                retry_count += 1
                wait = min(2 ** retry_count, 30)
                logger.error(
                    f"[DB Feed] Stream error (retry {retry_count}/"
                    f"{max_retries}): {e}. Reconnecting in {wait}s..."
                )
                time.sleep(wait)

        if retry_count >= max_retries:
            logger.error("[DB Feed] Max retries reached, feed stopped")
            self._connected = False

    def _on_record(self, record: Any) -> None:
        """Callback for all Databento records (called from live client thread).

        Routes records by type:
          - SymbolMappingMsg: build instrument_id -> instrument name mapping
          - OHLCVMsg: convert to Bar, push to bar_queue
          - MBP1Msg: convert to QuoteTick, push to quote_queue
          - ErrorMsg: log errors
        """
        if self._shutting_down:
            return

        try:
            if isinstance(record, db.SymbolMappingMsg):
                self._on_symbol_mapping(record)
            elif isinstance(record, db.OHLCVMsg):
                self._on_ohlcv(record)
            elif isinstance(record, db.MBP1Msg):
                self._on_quote(record)
            elif isinstance(record, db.ErrorMsg):
                logger.error(f"[DB Feed] Databento error: {record.err}")
        except Exception as e:
            logger.warning(f"[DB Feed] Error processing record: {e}")

    def _on_symbol_mapping(self, msg: Any) -> None:
        """Process SymbolMappingMsg: map instrument_id to instrument name."""
        instrument_id = msg.hd.instrument_id
        # stype_in_symbol is the continuous symbol we subscribed with (e.g. "MNQ.c.0")
        in_symbol = msg.stype_in_symbol

        instrument = REVERSE_SYMBOL_MAP.get(in_symbol)
        if instrument:
            self._id_to_instrument[instrument_id] = instrument
            logger.info(
                f"[DB Feed] Symbol mapping: id={instrument_id} "
                f"-> {in_symbol} -> {instrument}"
            )
        else:
            # Try the out_symbol or raw_symbol
            out_symbol = getattr(msg, "stype_out_symbol", "")
            logger.debug(
                f"[DB Feed] Symbol mapping (unmapped): id={instrument_id} "
                f"in={in_symbol} out={out_symbol}"
            )

    def _resolve_id(self, instrument_id: int) -> Optional[str]:
        """Resolve instrument_id to instrument name (MNQ, MES, etc)."""
        return self._id_to_instrument.get(instrument_id)

    def _on_ohlcv(self, record: Any) -> None:
        """Process OHLCVMsg: convert to Bar and push to queues.

        Databento ohlcv-1m delivers completed 1-min bars directly.
        No need for the bar-construction logic in TastytradeDataFeed --
        each record IS a completed bar.
        """
        # instrument_id is available both as record.instrument_id and
        # record.hd.instrument_id; use the direct attribute
        instrument_id = getattr(record, 'instrument_id', None)
        if instrument_id is None:
            instrument_id = record.hd.instrument_id
        instrument = self._resolve_id(instrument_id)
        if instrument is None:
            return

        # Convert fixed-point prices to float
        o = record.open * PRICE_SCALE
        h = record.high * PRICE_SCALE
        l = record.low * PRICE_SCALE
        c = record.close * PRICE_SCALE
        v = record.volume

        # Skip undefined/zero-price bars
        if record.open == UNDEF_PRICE or (o == 0 and c == 0):
            return

        # ts_event is nanoseconds since epoch
        ts_ns = record.ts_event
        ts = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)

        bar = Bar(
            timestamp=ts,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=float(v),
            instrument=instrument,
        )

        # Track last price
        self._last_prices[instrument] = bar.close

        # Add to all_bars history
        self._all_bars[instrument].append(bar)

        # Add to pending bars for get_latest_bar()
        self._pending_bars[instrument].append(bar)

        # Push to async bar_queue (cross-thread via call_soon_threadsafe)
        if self._loop and self._warmup_complete.is_set():
            self._loop.call_soon_threadsafe(
                self._safe_bar_queue_put, bar
            )

    def _safe_bar_queue_put(self, bar: Bar) -> None:
        """Put a bar on the async queue, dropping oldest if full."""
        try:
            self._bar_queue.put_nowait(bar)
        except asyncio.QueueFull:
            try:
                self._bar_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._bar_queue.put_nowait(bar)

    def _on_quote(self, record: Any) -> None:
        """Process MBP1Msg: convert to QuoteTick and push to quote_queue.

        MBP-1 (top of book) provides best bid/ask for intra-bar exit monitoring.
        runner.py's intra_bar_exit_loop accesses:
          - quote.event_symbol  (we set to instrument name for resolve_instrument)
          - quote.bid_price
          - quote.ask_price
        """
        instrument_id = getattr(record, 'instrument_id', None)
        if instrument_id is None:
            instrument_id = record.hd.instrument_id
        instrument = self._resolve_id(instrument_id)
        if instrument is None:
            return

        # Extract bid/ask from top level
        bid_px = record.levels[0].bid_px
        ask_px = record.levels[0].ask_px

        # Skip undefined prices
        if bid_px == UNDEF_PRICE or ask_px == UNDEF_PRICE:
            return

        bid = bid_px * PRICE_SCALE
        ask = ask_px * PRICE_SCALE

        if bid <= 0 or ask <= 0:
            return

        # Create quote tick with instrument name as event_symbol
        # (resolve_instrument will pass it through directly)
        quote = QuoteTick(
            event_symbol=instrument,
            bid_price=bid,
            ask_price=ask,
        )

        # Push to async quote_queue (cross-thread)
        if self._loop:
            self._loop.call_soon_threadsafe(
                self._safe_quote_queue_put, quote
            )

    def _safe_quote_queue_put(self, quote: QuoteTick) -> None:
        """Put a quote on the async queue, dropping oldest if full."""
        try:
            self._quote_queue.put_nowait(quote)
        except asyncio.QueueFull:
            try:
                self._quote_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._quote_queue.put_nowait(quote)
