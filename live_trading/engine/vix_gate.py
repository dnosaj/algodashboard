"""Fetch prior-day VIX close for death zone gating.

Primary: tastytrade DXLink Summary event for $VIX.X (has prev_day_close_price).
Fallback: yfinance (for mock mode with no tastytrade session).
"""
import asyncio
import logging
import socket
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

VIX_SYMBOL = "$VIX.X"


async def _fetch_vix_from_tastytrade(session) -> float | None:
    """Fetch prior-day VIX close via DXLink Summary event."""
    try:
        from tastytrade import DXLinkStreamer
        from tastytrade.dxfeed import Summary

        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Summary, [VIX_SYMBOL])
            summary = await asyncio.wait_for(streamer.get_event(Summary), timeout=10)

        if summary.prev_day_close_price is None:
            logger.warning("[VIX Gate] DXLink Summary returned None for prev_day_close_price")
            return None

        close = float(summary.prev_day_close_price)
        logger.info(f"[VIX Gate] Prior-day VIX close from tastytrade: {close:.2f}")
        return close
    except asyncio.TimeoutError:
        logger.warning("[VIX Gate] DXLink VIX Summary timed out after 10s")
        return None
    except Exception as e:
        logger.error(f"[VIX Gate] Failed to fetch VIX from tastytrade: {e}")
        return None


def _fetch_vix_from_yfinance() -> float | None:
    """Fallback: fetch prior-day VIX close via yfinance."""
    try:
        import yfinance as yf
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(10)
        try:
            end = datetime.now(_ET).date()
            start = end - timedelta(days=7)
            vix = yf.download("^VIX", start=str(start), end=str(end), progress=False)
        finally:
            socket.setdefaulttimeout(old_timeout)

        if vix.empty:
            logger.warning("[VIX Gate] No VIX data returned from yfinance")
            return None

        close_series = vix["Close"].iloc[-1]
        close = float(close_series.item() if hasattr(close_series, 'item') else close_series)
        data_date = vix.index[-1].date()

        days_old = (end - data_date).days
        if days_old > 3:
            logger.warning(f"[VIX Gate] VIX data is {days_old} days old ({data_date}) — may be stale")

        logger.info(f"[VIX Gate] Prior-day VIX close from yfinance: {close:.2f} ({data_date})")
        return close
    except Exception as e:
        logger.error(f"[VIX Gate] Failed to fetch VIX from yfinance: {e}")
        return None


async def fetch_prior_day_vix_close(session=None) -> float | None:
    """Return yesterday's VIX close, or None on failure.

    Uses tastytrade DXLink if session is provided, falls back to yfinance.
    Fail-open: returns None if all sources fail, so trading continues unblocked.
    """
    if session is not None:
        result = await _fetch_vix_from_tastytrade(session)
        if result is not None:
            return result
        logger.warning("[VIX Gate] tastytrade failed, falling back to yfinance")

    return _fetch_vix_from_yfinance()
