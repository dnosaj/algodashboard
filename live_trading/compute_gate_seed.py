"""
Compute gate seed values from Databento historical data.
=========================================================
Pre-computes ADR, Wilder ATR(14), and prior-day levels from the full
1-min bar history so the live engine's SafetyManager starts with warm gates.

Run before market open (or after downloading fresh data):
    cd live_trading && python3 compute_gate_seed.py

Reads all databento_{instrument}_1min_*.csv files from backtesting_engine/data/.
Outputs: live_trading/data/gate_seed.json

The live engine loads this seed at startup. After startup, SafetyManager
accumulates new daily data from live bars — no need to re-run this script
daily unless the engine hasn't run for 14+ days.
"""

import json
import math
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ET = ZoneInfo("America/New_York")
DATA_DIR = Path(__file__).resolve().parent.parent / "backtesting_engine" / "data"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "gate_seed.json"

RTH_START = 600   # 10:00 ET in minutes
RTH_END = 960     # 16:00 ET in minutes

INSTRUMENTS = ["MNQ", "MES"]
ADR_LOOKBACK = 14
ATR_PERIOD = 14


def load_1min(instrument: str) -> pd.DataFrame:
    """Load and concatenate all databento 1-min CSVs for an instrument."""
    files = sorted(DATA_DIR.glob(f"databento_{instrument}_1min_*.csv"))
    if not files:
        print(f"  WARNING: No databento files for {instrument}")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    result = pd.DataFrame()
    result["Time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    result["Open"] = pd.to_numeric(df["open"], errors="coerce")
    result["High"] = pd.to_numeric(df["high"], errors="coerce")
    result["Low"] = pd.to_numeric(df["low"], errors="coerce")
    result["Close"] = pd.to_numeric(df["close"], errors="coerce")
    result["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    result = result.set_index("Time")
    result = result[~result.index.duplicated(keep="first")]
    result = result.sort_index()

    print(f"  {instrument}: {len(result)} bars, {result.index[0]} to {result.index[-1]}")
    return result


def compute_rth_daily(df: pd.DataFrame) -> list[dict]:
    """Compute RTH (10:00-16:00 ET) daily stats: high, low, open, close, closes, volumes."""
    et_index = df.index.tz_convert(_ET)
    et_dates = et_index.date
    et_mins = (et_index.hour * 60 + et_index.minute).values.astype(np.int32)

    rth_mask = (et_mins >= RTH_START) & (et_mins < RTH_END)
    rth_df = df[rth_mask].copy()
    rth_df["et_date"] = et_dates[rth_mask]

    days = []
    for date, group in rth_df.groupby("et_date"):
        days.append({
            "date": str(date),
            "high": float(group["High"].max()),
            "low": float(group["Low"].min()),
            "open": float(group["Open"].iloc[0]),
            "close": float(group["Close"].iloc[-1]),
            "range": float(group["High"].max() - group["Low"].min()),
            "closes": group["Close"].tolist(),
            "volumes": group["Volume"].tolist(),
        })

    return days


def compute_wilder_atr(daily_ranges: list[float], period: int = 14) -> float | None:
    """Compute Wilder ATR from a list of daily ranges."""
    if len(daily_ranges) < period:
        return None

    # Seed with simple mean of first `period` ranges
    atr = sum(daily_ranges[:period]) / period

    # Wilder smoothing for remaining
    for r in daily_ranges[period:]:
        atr = (atr * (period - 1) + r) / period

    return atr


def compute_value_area(closes: list, volumes: list, bin_width: float = 5.0):
    """Compute VPOC, VAH, VAL from a day's closes and volumes."""
    if not closes or not volumes:
        return None

    total_vol = sum(volumes)
    if total_vol <= 0:
        return None

    price_min = math.floor(min(closes) / bin_width) * bin_width
    price_max = math.ceil(max(closes) / bin_width) * bin_width
    if price_min == price_max:
        price_max = price_min + bin_width

    n_bins = int(round((price_max - price_min) / bin_width)) + 1
    bin_volumes = [0.0] * n_bins

    for c, v in zip(closes, volumes):
        idx = int(round((c - price_min) / bin_width))
        idx = min(max(idx, 0), n_bins - 1)
        bin_volumes[idx] += v

    vpoc_idx = max(range(n_bins), key=lambda i: bin_volumes[i])
    vpoc_price = price_min + vpoc_idx * bin_width

    va_target = total_vol * 0.70
    va_vol = bin_volumes[vpoc_idx]
    lo_idx = vpoc_idx
    hi_idx = vpoc_idx

    while va_vol < va_target:
        can_go_lo = lo_idx > 0
        can_go_hi = hi_idx < n_bins - 1
        if not can_go_lo and not can_go_hi:
            break
        lo_vol = bin_volumes[lo_idx - 1] if can_go_lo else -1.0
        hi_vol = bin_volumes[hi_idx + 1] if can_go_hi else -1.0
        if lo_vol >= hi_vol:
            lo_idx -= 1
            va_vol += bin_volumes[lo_idx]
        else:
            hi_idx += 1
            va_vol += bin_volumes[hi_idx]

    val_price = price_min + lo_idx * bin_width
    vah_price = price_min + hi_idx * bin_width
    return vpoc_price, vah_price, val_price


def compute_all_daily_ranges(df: pd.DataFrame) -> list[float]:
    """Compute daily ranges across ALL bars (not just RTH) for Wilder ATR."""
    et_index = df.index.tz_convert(_ET)
    et_dates = et_index.date
    df_copy = df.copy()
    df_copy["et_date"] = et_dates

    ranges = []
    for _, group in df_copy.groupby("et_date"):
        day_range = float(group["High"].max() - group["Low"].min())
        ranges.append(day_range)
    return ranges


def main():
    print("=" * 70)
    print("GATE SEED COMPUTATION")
    print("=" * 70)

    seed = {}

    for inst in INSTRUMENTS:
        print(f"\n--- {inst} ---")
        df = load_1min(inst)
        if df.empty:
            continue

        rth_days = compute_rth_daily(df)
        print(f"  RTH days: {len(rth_days)}")

        if len(rth_days) < 2:
            print(f"  WARNING: Not enough RTH days for {inst}")
            continue

        # --- ADR: rolling mean of last N RTH daily ranges ---
        rth_ranges = [d["range"] for d in rth_days]
        adr_ranges = rth_ranges[-ADR_LOOKBACK:]  # Last 14 for seeding
        adr_value = sum(adr_ranges) / len(adr_ranges) if adr_ranges else None
        print(f"  ADR({ADR_LOOKBACK}): {adr_value:.1f} (from {len(adr_ranges)} days)")

        # --- Wilder ATR(14): from ALL-bar daily ranges ---
        all_ranges = compute_all_daily_ranges(df)
        atr_value = compute_wilder_atr(all_ranges, ATR_PERIOD)
        print(f"  Wilder ATR({ATR_PERIOD}): {atr_value:.1f}" if atr_value else
              f"  Wilder ATR({ATR_PERIOD}): None (not enough data)")

        # --- Prior-day levels: last completed RTH day ---
        prior_day = rth_days[-1]  # Most recent completed day
        va = compute_value_area(prior_day["closes"], prior_day["volumes"])
        prior_day_levels = {
            "high": prior_day["high"],
            "low": prior_day["low"],
            "vpoc": va[0] if va else None,
            "vah": va[1] if va else None,
            "val": va[2] if va else None,
        }
        print(f"  Prior-day ({prior_day['date']}): "
              f"H={prior_day['high']:.2f} L={prior_day['low']:.2f} "
              f"VPOC={prior_day_levels['vpoc']}")

        # --- Leledc: last 20 closes for counter seeding ---
        last_closes = df["Close"].values[-20:].tolist()

        seed[inst] = {
            # ADR gate seeding: completed RTH daily ranges (last 30 for rolling window)
            "adr_completed_ranges": rth_ranges[-30:],
            "adr_value": round(adr_value, 2) if adr_value else None,

            # Prior-day ATR gate seeding: all-bar daily ranges + ATR value
            "atr_daily_ranges": all_ranges[-20:],
            "atr_value": round(atr_value, 2) if atr_value else None,

            # Prior-day level gate seeding
            "prior_day_levels": prior_day_levels,
            "prior_day_date": prior_day["date"],

            # Leledc seeding: last 20 closes
            "leledc_closes": [round(c, 4) for c in last_closes],

            # RTH session state (for ADR intraday tracking)
            # Last completed day's data for reference
            "last_rth_date": rth_days[-1]["date"],
        }

    # Add timestamp for staleness detection
    from datetime import datetime
    seed["_saved_at"] = datetime.now(_ET).isoformat()

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(seed, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Instruments: {list(seed.keys())}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
