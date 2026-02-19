"""
Data feed module for the live trading engine.

Provides bar data from market sources. The base class defines the interface
and common logic (gap detection, deduplication, warmup). Concrete subclasses
implement the actual data retrieval.

WebullDataFeed: production feed using Webull HTTP API (stub -- API not yet approved).
MockDataFeed: reads from CSV files for testing and development.

All feeds emit Bar objects through an asyncio callback and are designed
to be driven by the main engine loop.
"""

import asyncio
import csv
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from .events import Bar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DataFeed(ABC):
    """Base class for all data feeds.

    Subclasses must implement:
        _fetch_historical_bars() -- load N historical bars for warmup
        _fetch_latest_bars()     -- poll for new bars since last seen
    """

    def __init__(
        self,
        instrument: str,
        on_bar: Callable[[Bar], None],
        poll_interval_sec: float = 60.0,
    ) -> None:
        self.instrument = instrument
        self._on_bar = on_bar
        self._poll_interval = poll_interval_sec
        self._last_bar_time: Optional[datetime] = None
        self._running = False
        self._total_bars_received = 0
        self._gaps_detected = 0

    @property
    def last_bar_time(self) -> Optional[datetime]:
        return self._last_bar_time

    @property
    def seconds_since_last_bar(self) -> float:
        if self._last_bar_time is None:
            return float("inf")
        now = datetime.now(timezone.utc)
        last = self._last_bar_time
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last).total_seconds()

    @property
    def is_connected(self) -> bool:
        return self._running

    async def warmup(self, num_bars: int) -> list[Bar]:
        """Load historical bars for indicator warmup.

        Returns:
            List of Bar objects ordered oldest-first.
        """
        logger.info(f"[{self.instrument}] Loading {num_bars} warmup bars...")
        bars = await self._fetch_historical_bars(num_bars)
        if not bars:
            logger.warning(f"[{self.instrument}] No warmup bars returned")
            return []

        bars.sort(key=lambda b: b.timestamp)

        if len(bars) < num_bars:
            logger.warning(
                f"[{self.instrument}] Requested {num_bars} warmup bars, "
                f"got {len(bars)}"
            )

        self._last_bar_time = bars[-1].timestamp
        self._total_bars_received = len(bars)
        logger.info(
            f"[{self.instrument}] Warmup complete: {len(bars)} bars, "
            f"last={self._last_bar_time}"
        )
        return bars

    async def start_polling(self) -> None:
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        logger.info(
            f"[{self.instrument}] Data feed polling started "
            f"(interval={self._poll_interval}s)"
        )
        try:
            while self._running:
                await self._poll_once()
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            logger.info(f"[{self.instrument}] Data feed polling cancelled")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._running = False
        logger.info(f"[{self.instrument}] Data feed stopping")

    async def _poll_once(self) -> None:
        """Poll for new bars, handle dedup and gap detection."""
        try:
            new_bars = await self._fetch_latest_bars(since=self._last_bar_time)
        except Exception as e:
            logger.error(f"[{self.instrument}] Fetch error: {e}")
            return

        if not new_bars:
            return

        new_bars.sort(key=lambda b: b.timestamp)

        for bar in new_bars:
            # Deduplication: skip bars we've already seen
            if self._last_bar_time is not None and bar.timestamp <= self._last_bar_time:
                continue

            # Gap detection: more than 2 minutes between bars during RTH
            if self._last_bar_time is not None:
                expected_next = self._last_bar_time + timedelta(minutes=1)
                if bar.timestamp > expected_next + timedelta(seconds=30):
                    gap_mins = (bar.timestamp - self._last_bar_time).total_seconds() / 60.0
                    self._gaps_detected += 1
                    logger.warning(
                        f"[{self.instrument}] GAP DETECTED: {gap_mins:.1f} min gap "
                        f"(last={self._last_bar_time}, new={bar.timestamp}). "
                        f"Total gaps: {self._gaps_detected}"
                    )

            self._last_bar_time = bar.timestamp
            self._total_bars_received += 1
            self._on_bar(bar)

    @abstractmethod
    async def _fetch_historical_bars(self, num_bars: int) -> list[Bar]:
        """Fetch N historical bars for warmup.

        Must return bars ordered oldest-first. Called once at startup.
        """
        ...

    @abstractmethod
    async def _fetch_latest_bars(self, since: Optional[datetime]) -> list[Bar]:
        """Fetch bars newer than `since` timestamp.

        Called every poll_interval_sec. Must return only completed bars
        (not the currently forming bar).
        """
        ...


# ---------------------------------------------------------------------------
# Webull HTTP data feed (stub -- API not approved yet)
# ---------------------------------------------------------------------------

class WebullDataFeed(DataFeed):
    """Production data feed using Webull HTTP API.

    All API methods log a not-implemented message and return empty results.
    These will be filled in once Webull API access is approved and the
    endpoint signatures are known.
    """

    def __init__(
        self,
        instrument: str,
        on_bar: Callable[[Bar], None],
        app_key: str = "",
        app_secret: str = "",
        account_id: str = "",
        base_url: str = "https://api.webull.com",
        region_id: str = "us",
        poll_interval_sec: float = 60.0,
    ) -> None:
        super().__init__(instrument, on_bar, poll_interval_sec)
        self._app_key = app_key
        self._app_secret = app_secret
        self._account_id = account_id
        self._base_url = base_url
        self._region_id = region_id
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    async def authenticate(self) -> bool:
        """Authenticate with Webull API and obtain access token.

        Returns True if authentication succeeded.
        """
        logger.warning(
            "[WebullDataFeed] authenticate() not implemented -- "
            "Webull API access not yet approved. "
            "Returning False."
        )
        return False

    async def _fetch_historical_bars(self, num_bars: int) -> list[Bar]:
        """Fetch historical 1-min bars from Webull.

        Will call the Webull kline/history endpoint once API access is available.
        Expected endpoint: GET /api/quote/kline
        Params: instrument_id, interval=1m, count=num_bars
        """
        logger.warning(
            f"[WebullDataFeed] _fetch_historical_bars({num_bars}) not implemented -- "
            "Webull API access not yet approved. "
            "Returning empty list."
        )
        return []

    async def _fetch_latest_bars(self, since: Optional[datetime]) -> list[Bar]:
        """Fetch latest completed 1-min bars from Webull.

        Will call the Webull kline endpoint once API access is available.
        Expected endpoint: GET /api/quote/kline
        Params: instrument_id, interval=1m, since=timestamp
        Must filter out the currently-forming bar (only return completed bars).
        """
        logger.warning(
            f"[WebullDataFeed] _fetch_latest_bars(since={since}) not implemented -- "
            "Webull API access not yet approved. "
            "Returning empty list."
        )
        return []

    def _parse_webull_bar(self, raw: dict) -> Bar:
        """Parse a raw Webull kline JSON object into a Bar.

        Expected format (to be confirmed with actual API docs):
            {"time": epoch_ms, "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}
        """
        ts = datetime.fromtimestamp(raw["time"] / 1000.0, tz=timezone.utc)
        return Bar(
            timestamp=ts,
            open=float(raw["open"]),
            high=float(raw["high"]),
            low=float(raw["low"]),
            close=float(raw["close"]),
            volume=float(raw.get("volume", 0)),
            instrument=self.instrument,
        )


# ---------------------------------------------------------------------------
# Mock data feed for testing
# ---------------------------------------------------------------------------

class MockDataFeed(DataFeed):
    """Data feed that replays bars from a CSV file.

    CSV format expected (TradingView export):
        time,open,high,low,close,volume   (header row)
        2026-02-10T15:01:00Z,21500.25,...

    Or Databento format:
        ts_event,open,high,low,close,volume
        2026-02-10 10:01:00-05:00,...

    The mock feed replays bars one at a time on each poll, simulating
    real-time delivery. For warmup, it returns the first N bars immediately.
    """

    def __init__(
        self,
        instrument: str,
        on_bar: Callable[[Bar], None],
        csv_path: str,
        poll_interval_sec: float = 0.0,  # 0 = as fast as possible for testing
        replay_speed: float = 0.0,       # 0 = instant replay, >0 = simulated delay
    ) -> None:
        super().__init__(instrument, on_bar, poll_interval_sec)
        self._csv_path = Path(csv_path)
        self._replay_speed = replay_speed
        self._all_bars: list[Bar] = []
        self._cursor = 0
        self._loaded = False

    async def load(self) -> None:
        """Pre-load all bars from CSV."""
        if self._loaded:
            return

        if not self._csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self._csv_path}")

        bars: list[Bar] = []
        with open(self._csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bar = self._parse_csv_row(row)
                if bar is not None:
                    bars.append(bar)

        bars.sort(key=lambda b: b.timestamp)
        self._all_bars = bars
        self._loaded = True
        logger.info(
            f"[MockDataFeed:{self.instrument}] Loaded {len(bars)} bars from {self._csv_path}"
        )

    def _parse_csv_row(self, row: dict) -> Optional[Bar]:
        """Parse a CSV row into a Bar, handling multiple CSV formats."""
        # Determine timestamp column
        ts_raw = row.get("time") or row.get("ts_event") or row.get("timestamp")
        if ts_raw is None:
            logger.warning(f"[MockDataFeed] No timestamp column in row: {row}")
            return None

        ts_raw = ts_raw.strip()

        # Parse timestamp
        ts: Optional[datetime] = None
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",          # ISO UTC
            "%Y-%m-%dT%H:%M:%S%z",         # ISO with offset
            "%Y-%m-%d %H:%M:%S%z",         # Databento with offset
            "%Y-%m-%d %H:%M:%S",           # Naive (assume UTC)
            "%Y-%m-%dT%H:%M:%S.%fZ",       # ISO with microseconds
            "%Y-%m-%dT%H:%M:%S.%f%z",      # ISO with microseconds + offset
        ]
        for fmt in formats:
            try:
                ts = datetime.strptime(ts_raw, fmt)
                break
            except ValueError:
                continue

        if ts is None:
            # Try pandas-style parsing as last resort
            try:
                from dateutil.parser import parse as dateutil_parse
                ts = dateutil_parse(ts_raw)
            except Exception:
                logger.warning(f"[MockDataFeed] Cannot parse timestamp: {ts_raw}")
                return None

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Parse OHLCV
        try:
            o = float(row.get("open", 0))
            h = float(row.get("high", 0))
            lo = float(row.get("low", 0))
            c = float(row.get("close", 0))
            # Volume may be missing (MNQ CSVs often have no volume)
            vol_raw = row.get("volume", row.get("Volume", "0"))
            vol = float(vol_raw) if vol_raw else 0.0
        except (ValueError, TypeError) as e:
            logger.warning(f"[MockDataFeed] Cannot parse OHLCV: {e}, row={row}")
            return None

        # If no volume, synthesize from range (as per MEMORY.md)
        if vol == 0.0:
            vol = h - lo

        return Bar(
            timestamp=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=vol,
            instrument=self.instrument,
        )

    async def _fetch_historical_bars(self, num_bars: int) -> list[Bar]:
        """Return first num_bars from the CSV for warmup."""
        if not self._loaded:
            await self.load()

        available = min(num_bars, len(self._all_bars))
        warmup_bars = self._all_bars[:available]
        self._cursor = available
        return list(warmup_bars)

    async def _fetch_latest_bars(self, since: Optional[datetime]) -> list[Bar]:
        """Return next bar(s) from the CSV, simulating real-time delivery."""
        if not self._loaded:
            await self.load()

        if self._cursor >= len(self._all_bars):
            # No more bars
            if self._running:
                logger.info(f"[MockDataFeed:{self.instrument}] All bars replayed, stopping")
                self._running = False
            return []

        # Return the next bar
        bar = self._all_bars[self._cursor]
        self._cursor += 1

        if self._replay_speed > 0:
            await asyncio.sleep(self._replay_speed)

        return [bar]

    @property
    def bars_remaining(self) -> int:
        return max(0, len(self._all_bars) - self._cursor)

    @property
    def progress(self) -> float:
        if not self._all_bars:
            return 0.0
        return self._cursor / len(self._all_bars)
