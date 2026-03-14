"""
Round 3 — Study 1: Prior Day H/L + Volume Profile Levels (VPOC/VAH/VAL)
========================================================================
Gate that blocks entries when price is within `buffer_pts` of the nearest
prior-day support or resistance level.

Levels considered:
  - Prior RTH day High / Low  (from v10_test_common.compute_prior_day_levels)
  - Prior RTH day VPOC / VAH / VAL  (computed here from 1-min volume profile)

Volume profile is built per RTH day (10:00-16:00 ET) by binning closes into
fixed-width buckets and accumulating volume. The value area is 70% of total
volume, expanding outward from the VPOC bin.

Sweep: buffer_pts = [None, 0, 2, 5, 8, 10, 15]

Look-ahead check: All levels use completed prior RTH day. No look-ahead bias.
"""

import numpy as np
import pandas as pd

from sr_common import (
    STRATEGIES,
    prepare_data,
    run_sweep,
    compute_et_minutes,
    NY_OPEN_ET,
    NY_CLOSE_ET,
)
from v10_test_common import compute_prior_day_levels


# ---------------------------------------------------------------------------
# Volume profile computation
# ---------------------------------------------------------------------------

def compute_rth_volume_profile(times, closes, volumes, et_mins, bin_width):
    """Compute prior-day VPOC, VAH, VAL from RTH volume profile.

    For each RTH day (10:00-16:00 ET), bins 1-min closes into `bin_width`-pt
    buckets and accumulates volume.

    VPOC  = price bucket with maximum volume.
    VAH   = upper bound of 70% value area (expanding from VPOC).
    VAL   = lower bound of 70% value area (expanding from VPOC).

    Returns (vpoc, vah, val) arrays of prior-day values mapped to each bar.
    Bars before the first completed RTH day get NaN.
    """
    n = len(times)
    vpoc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)

    # --- Pass 1: collect per-day volume profile ---
    daily_profiles = {}  # date -> (vpoc_price, vah_price, val_price)
    current_date = None
    day_closes = []
    day_volumes = []

    for i in range(n):
        ts = pd.Timestamp(times[i])
        bar_date = ts.date()

        if bar_date != current_date:
            # Finalize previous day
            if current_date is not None and len(day_closes) > 0:
                profile = _compute_value_area(day_closes, day_volumes, bin_width)
                if profile is not None:
                    daily_profiles[current_date] = profile
            current_date = bar_date
            day_closes = []
            day_volumes = []

        if NY_OPEN_ET <= et_mins[i] < NY_CLOSE_ET:
            day_closes.append(closes[i])
            day_volumes.append(volumes[i])

    # Save last day
    if current_date is not None and len(day_closes) > 0:
        profile = _compute_value_area(day_closes, day_volumes, bin_width)
        if profile is not None:
            daily_profiles[current_date] = profile

    # --- Pass 2: map prior-day profile to each bar ---
    sorted_dates = sorted(daily_profiles.keys())
    date_to_prev = {}
    for j in range(1, len(sorted_dates)):
        date_to_prev[sorted_dates[j]] = daily_profiles[sorted_dates[j - 1]]

    for i in range(n):
        ts = pd.Timestamp(times[i])
        bar_date = ts.date()
        if bar_date in date_to_prev:
            vpoc_p, vah_p, val_p = date_to_prev[bar_date]
            vpoc[i] = vpoc_p
            vah[i] = vah_p
            val[i] = val_p

    return vpoc, vah, val


def _compute_value_area(day_closes, day_volumes, bin_width):
    """Compute VPOC, VAH, VAL from a single day's closes and volumes.

    Returns (vpoc_price, vah_price, val_price) or None if no data.
    """
    if len(day_closes) == 0:
        return None

    closes_arr = np.array(day_closes, dtype=np.float64)
    volumes_arr = np.array(day_volumes, dtype=np.float64)

    total_vol = np.sum(volumes_arr)
    if total_vol <= 0:
        return None

    # Bin prices
    price_min = np.floor(np.min(closes_arr) / bin_width) * bin_width
    price_max = np.ceil(np.max(closes_arr) / bin_width) * bin_width
    if price_min == price_max:
        price_max = price_min + bin_width

    n_bins = int(round((price_max - price_min) / bin_width)) + 1
    bin_volumes = np.zeros(n_bins)

    for c, v in zip(closes_arr, volumes_arr):
        idx = int(round((c - price_min) / bin_width))
        idx = min(max(idx, 0), n_bins - 1)
        bin_volumes[idx] += v

    # VPOC = bin with max volume
    vpoc_idx = int(np.argmax(bin_volumes))
    vpoc_price = price_min + vpoc_idx * bin_width

    # Value area: expand from VPOC until 70% of volume captured
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


# ---------------------------------------------------------------------------
# Gate construction
# ---------------------------------------------------------------------------

def build_prior_day_level_gate(closes, prev_high, prev_low, vpoc, vah, val,
                               buffer_pts):
    """Build boolean gate: True = allow entry, False = block.

    Blocks when close is within `buffer_pts` of the nearest prior-day
    resistance (above) or support (below).

    Resistance candidates: prev_high, vpoc (if above close), vah (if above close)
    Support candidates:    prev_low,  vpoc (if below close), val (if below close)

    If all levels are NaN for a bar, the entry is allowed.
    """
    n = len(closes)
    gate = np.ones(n, dtype=bool)

    for i in range(n):
        c = closes[i]

        # Collect all non-NaN levels
        levels = []
        if not np.isnan(prev_high[i]):
            levels.append(prev_high[i])
        if not np.isnan(prev_low[i]):
            levels.append(prev_low[i])
        if not np.isnan(vpoc[i]):
            levels.append(vpoc[i])
        if not np.isnan(vah[i]):
            levels.append(vah[i])
        if not np.isnan(val[i]):
            levels.append(val[i])

        if len(levels) == 0:
            continue

        # Block if close is within buffer_pts of ANY level
        for lvl in levels:
            if abs(c - lvl) <= buffer_pts:
                gate[i] = False
                break

    return gate


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

BIN_WIDTHS = {"MNQ": 2, "MES": 5}

SWEEP_CONFIGS = [{"label": "None"}]
for _buf in [0, 2, 5, 8, 10, 15]:
    SWEEP_CONFIGS.append({"label": f"buf{_buf}", "buffer_pts": _buf})


def build_gates(config, instruments):
    """build_gates_fn for run_sweep: prior-day H/L + volume profile gate."""
    gates = {}
    for inst, df in instruments.items():
        times = df.index
        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values
        volumes = df["Volume"].values
        et_mins = compute_et_minutes(times)

        # Prior day H/L from v10_test_common
        prev_high, prev_low, _ = compute_prior_day_levels(times, highs, lows, closes)

        # Volume profile levels
        vpoc, vah, val = compute_rth_volume_profile(
            times, closes, volumes, et_mins, BIN_WIDTHS[inst]
        )

        gate = build_prior_day_level_gate(
            closes, prev_high, prev_low, vpoc, vah, val, config["buffer_pts"]
        )
        gates[inst] = gate
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    instruments, split_arrays, split_indices = prepare_data()

    run_sweep(
        study_name="Prior Day H/L + Volume Profile (VPOC/VAH/VAL)",
        strategies=STRATEGIES,
        split_arrays=split_arrays,
        split_indices=split_indices,
        instruments=instruments,
        build_gates_fn=build_gates,
        sweep_configs=SWEEP_CONFIGS,
        script_name="sr_prior_day_levels_sweep.py",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
