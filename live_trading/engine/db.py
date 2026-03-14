"""
Supabase client singleton with lazy initialization and graceful degradation.

If SUPABASE_URL and SUPABASE_SERVICE_KEY are not set, all database operations
are silently skipped. Trading continues unaffected.
"""

import logging
import os

logger = logging.getLogger(__name__)

_client = None
_initialized = False


def get_client():
    """Return the Supabase client, or None if not configured.

    Lazy-initializes on first call. Subsequent calls return the cached client.
    Safe to call from sync or async contexts (client init is synchronous).
    """
    global _client, _initialized

    if _initialized:
        return _client
    _initialized = True

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

    if not url or not key:
        logger.info("[DB] SUPABASE_URL/SUPABASE_SERVICE_KEY not set — database logging disabled")
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        logger.info("[DB] Supabase connected")
        return _client
    except ImportError:
        logger.warning("[DB] supabase-py not installed (pip install supabase) — database logging disabled")
        return None
    except Exception as e:
        logger.warning(f"[DB] Supabase connection failed: {e} — database logging disabled")
        return None


def reset():
    """Reset client state (for testing)."""
    global _client, _initialized
    _client = None
    _initialized = False
