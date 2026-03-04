"""Fetch prior-day VIX close for death zone gating."""
import logging
import socket
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


def fetch_prior_day_vix_close() -> float | None:
    """Return yesterday's VIX close, or None on failure.

    Fail-open: returns None if yfinance is down, so trading
    continues unblocked. Logs warning for operator visibility.
    """
    try:
        import yfinance as yf
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(10)  # 10s cap to avoid blocking event loop
        try:
            end = datetime.now(_ET).date()
            start = end - timedelta(days=7)
            vix = yf.download("^VIX", start=str(start), end=str(end), progress=False)
        finally:
            socket.setdefaulttimeout(old_timeout)

        if vix.empty:
            logger.warning("[VIX Gate] No VIX data returned from yfinance")
            return None

        close = float(vix["Close"].iloc[-1])
        data_date = vix.index[-1].date()

        # Staleness check: warn if data is >3 calendar days old
        days_old = (end - data_date).days
        if days_old > 3:
            logger.warning(f"[VIX Gate] VIX data is {days_old} days old ({data_date}) — may be stale")

        logger.info(f"[VIX Gate] Prior-day VIX close: {close:.2f} ({data_date})")
        return close
    except Exception as e:
        logger.error(f"[VIX Gate] Failed to fetch VIX: {e}")
        return None
