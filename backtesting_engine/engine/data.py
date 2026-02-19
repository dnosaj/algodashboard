"""
Data loading utilities for backtesting.

Supports:
- TradingView CSV exports (recommended for exact OHLC match)
- Bitstamp API (free, no auth, fallback)

Important: The last bar in live data is always dropped — it represents
an unfinished candle that would produce unreliable signals.
"""

import requests
import time
import pandas as pd
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _ENGINE_DIR.parent          # dev/ or ship/
_DATA_DIR = _PROJECT_DIR / "data"
CACHE_DIR = _DATA_DIR / "cache"


def load_tv_export(filename: str = "INDEX_BTCUSD, 1D.csv") -> pd.DataFrame:
    """
    Load a TradingView-exported CSV file.

    TV export format: time (unix timestamp), open, high, low, close.
    The last bar is dropped because it is an unfinished (still-printing) candle.

    Args:
        filename: Name of the CSV file in the ``data/`` directory.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (timezone-naive, named 'Date')
    """
    filepath = _DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"TV export not found: {filepath}\n"
            f"Place CSV files in the data/ directory."
        )

    df = pd.read_csv(filepath)

    # Convert unix timestamps to datetime index
    df["Date"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("Date")

    # Rename to standard column names
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    })

    # Keep only OHLC(V) columns
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]

    # Add dummy volume if not present (TV exports don't include volume)
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.sort_index()
    df = df.dropna()

    # Drop the last bar — it is an unfinished candle
    dropped_date = df.index[-1]
    df = df.iloc[:-1]

    print(f"Loaded TV export: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Dropped last bar (unfinished candle): {dropped_date.date()}")
    return df


# ---------------------------------------------------------------------------
# Bitstamp API (fallback — prices differ slightly from INDEX:BTCUSD)
# ---------------------------------------------------------------------------

BITSTAMP_OHLC_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"


def _fetch_bitstamp_chunk(end_ts: int, step: int = 86400, limit: int = 1000) -> list[dict]:
    """Fetch up to `limit` daily candles ending before `end_ts` from Bitstamp."""
    params = {"step": step, "limit": limit, "end": end_ts}
    resp = requests.get(BITSTAMP_OHLC_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("ohlc", [])


def fetch_btc_daily(
    start: str = "2017-01-01",
    end: str = "2026-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch BTC/USD daily OHLCV data from Bitstamp.

    The last bar is dropped because it is an unfinished (still-printing) candle.

    Args:
        start: Start date for data fetch (include warmup period).
        end:   End date for data fetch.
        use_cache: Cache data locally as CSV.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (timezone-naive, named 'Date')
    """
    cache_file = CACHE_DIR / f"BITSTAMP-BTCUSD_{start}_{end}_1d.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded cached data: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    print(f"Fetching BITSTAMP:BTCUSD daily data from {start} to {end}...")

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    all_candles: list[dict] = []
    cursor = end_ts

    while cursor > start_ts:
        chunk = _fetch_bitstamp_chunk(end_ts=cursor, step=86400, limit=1000)
        if not chunk:
            break

        all_candles.extend(chunk)

        first_ts = int(chunk[0]["timestamp"])
        if first_ts >= cursor:
            break
        cursor = first_ts
        print(f"  Fetched {len(all_candles)} candles so far "
              f"(back to {pd.Timestamp(first_ts, unit='s').date()})...")
        time.sleep(0.3)

    if not all_candles:
        raise ValueError("No data returned from Bitstamp API")

    # Build DataFrame
    df = pd.DataFrame(all_candles)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("Date")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    df = df.dropna()
    df = df[df["Volume"] > 0]

    # Drop the last bar — it is an unfinished candle
    dropped_date = df.index[-1]
    df = df.iloc[:-1]

    print(f"Fetched {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Dropped last bar (unfinished candle): {dropped_date.date()}")

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_csv(cache_file)
        print(f"  Cached to {cache_file}")

    return df
