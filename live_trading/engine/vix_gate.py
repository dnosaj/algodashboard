"""Fetch prior-day VIX close for death zone gating.

Uses yfinance to get prior-day VIX close.
DXLink was tried but tastytrade doesn't serve Summary events for index
symbols ($VIX.X) — all requests time out. yfinance is reliable with the
.item() fix for pandas Series extraction.
"""
import logging
import socket
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


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

    Uses yfinance. Session parameter kept for API compatibility but unused
    (DXLink doesn't serve index symbols).
    Fail-open: returns None on failure, so trading continues unblocked.
    """
    return _fetch_vix_from_yfinance()
