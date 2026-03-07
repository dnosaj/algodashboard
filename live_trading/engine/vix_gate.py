"""Fetch prior-day VIX close for death zone gating.

Primary: tastytrade DXLink Summary event (prev_day_close_price).
Fallback: yfinance (if tastytrade unavailable or fails).
Fail-open: returns None on failure, so trading continues unblocked.
"""
import asyncio
import logging
import socket
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


async def _fetch_vix_from_tastytrade(session) -> float | None:
    """Fetch prior-day VIX close via tastytrade DXLink Summary event."""
    try:
        from tastytrade import DXLinkStreamer
        from tastytrade.instruments import Equity
        from tastytrade.dxfeed import Summary

        # Look up the correct streamer symbol for VIX
        vix_equity = await Equity.get(session, 'VIX')
        # get() may return a list; handle both cases
        if isinstance(vix_equity, list):
            vix_equity = vix_equity[0]
        streamer_symbol = vix_equity.streamer_symbol
        logger.info(f"[VIX Gate] VIX streamer symbol: {streamer_symbol}")

        # Open a short-lived DXLink connection to get the Summary snapshot
        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Summary, [streamer_symbol])
            # Wait for the first Summary event (with timeout)
            async for summary in streamer.listen(Summary):
                if summary.prev_day_close_price is not None:
                    close = float(summary.prev_day_close_price)
                    logger.info(
                        f"[VIX Gate] Prior-day VIX close from tastytrade: {close:.2f}"
                    )
                    return close
                # Some Summary events may have None fields; keep listening briefly
                break

        logger.warning("[VIX Gate] tastytrade Summary had no prev_day_close_price")
        return None
    except Exception as e:
        logger.warning(f"[VIX Gate] tastytrade VIX fetch failed: {e}")
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
    """Return yesterday's VIX close. Tastytrade primary, yfinance fallback.

    Fail-open: returns None on failure, so trading continues unblocked.
    """
    # Try tastytrade first if session available (15s timeout)
    if session is not None:
        try:
            result = await asyncio.wait_for(
                _fetch_vix_from_tastytrade(session), timeout=15
            )
        except asyncio.TimeoutError:
            logger.warning("[VIX Gate] tastytrade VIX fetch timed out after 15s")
            result = None
        if result is not None:
            return result
        logger.info("[VIX Gate] tastytrade failed, falling back to yfinance")

    # Fallback to yfinance
    return _fetch_vix_from_yfinance()
