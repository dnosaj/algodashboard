"""
Round 3, Study 5 -- Initial Balance Proximity Entry Gate Sweep
==============================================================
Hypothesis: Block entries near RTH Initial Balance (IB) high/low.
IB levels are well-known institutional reaction zones on NQ/ES.

The Initial Balance is the range established during the first N minutes
of Regular Trading Hours (9:30 ET).  Common IB periods are 30 min
(9:30-10:00) and 60 min (9:30-10:30).

Gate: block entry if price (close) is within buffer_pts of either
the IB high or IB low.  During the IB period itself, levels are
"developing" (running max high / min low).  After the IB period
closes, levels are finalized for the rest of the session.

Note: 30-min IB (9:30-10:00) is fully formed before entries start
at 10:00 ET.  60-min IB (9:30-10:30) uses developing IB for bars
10:00-10:30.

Sweep: ib_period = [30, 60] x buffer_pts = [0, 2, 5, 8, 10]
       -> 10 configs + baseline
"""

import numpy as np
import pandas as pd

from sr_common import (
    STRATEGIES,
    compute_et_minutes,
    prepare_data,
    run_sweep,
)


# ---------------------------------------------------------------------------
# Initial Balance computation
# ---------------------------------------------------------------------------

def compute_initial_balance(times, highs, lows, et_mins, ib_period_minutes=30):
    """Compute developing/finalized Initial Balance high and low arrays.

    RTH starts at 9:30 ET (570 minutes from midnight).  The IB period
    covers 9:30 to 9:30 + ib_period_minutes.

    During the IB period, IB high/low are *developing* -- each bar sees
    the running max(high) and min(low) since 9:30 of that day.

    After the IB period ends, levels are finalized and carried forward
    for the rest of the session day.

    Before the first IB bar of each day (pre-9:30), values are NaN.

    Args:
        times: DatetimeIndex (UTC) of bar timestamps.
        highs: 1-D numpy array of bar high prices.
        lows: 1-D numpy array of bar low prices.
        et_mins: integer array of ET minutes from midnight (from
                 compute_et_minutes).
        ib_period_minutes: length of IB window in minutes (default 30).

    Returns:
        (ib_high, ib_low): numpy float64 arrays, same length as *times*.
    """
    RTH_START = 570  # 9:30 ET in minutes from midnight
    ib_end = RTH_START + ib_period_minutes  # e.g. 600 for 30-min IB

    n = len(times)
    ib_high = np.full(n, np.nan, dtype=np.float64)
    ib_low = np.full(n, np.nan, dtype=np.float64)

    # Detect day boundaries by extracting the date from each UTC timestamp.
    # Use pd.Timestamp to convert so we can call .date() reliably.
    dates = np.array([pd.Timestamp(t).date() for t in times])

    current_date = None
    running_high = np.nan
    running_low = np.nan
    ib_finalized = False

    for i in range(n):
        bar_date = dates[i]
        bar_et = et_mins[i]

        # New trading day
        if bar_date != current_date:
            current_date = bar_date
            running_high = np.nan
            running_low = np.nan
            ib_finalized = False

        # Before RTH open -- no IB yet
        if bar_et < RTH_START:
            # ib_high / ib_low remain NaN (or carry from yesterday -- we reset)
            continue

        # Within IB period (developing)
        if bar_et < ib_end:
            if np.isnan(running_high):
                running_high = highs[i]
                running_low = lows[i]
            else:
                running_high = max(running_high, highs[i])
                running_low = min(running_low, lows[i])
            ib_high[i] = running_high
            ib_low[i] = running_low
            continue

        # After IB period -- finalize if not yet done
        if not ib_finalized:
            ib_finalized = True
            # If we somehow never saw an IB bar (e.g. data gap), stay NaN

        # Carry finalized levels forward
        ib_high[i] = running_high
        ib_low[i] = running_low

    return ib_high, ib_low


# ---------------------------------------------------------------------------
# Gate builder
# ---------------------------------------------------------------------------

def build_ib_gate(closes, ib_high, ib_low, buffer_pts):
    """Build a boolean gate that blocks entries near IB high/low.

    A bar is blocked (gate = False) if the close price is within
    *buffer_pts* of either the IB high or the IB low:
        blocked if (ib_high - buffer <= close <= ib_high + buffer)
              OR   (ib_low  - buffer <= close <= ib_low  + buffer)

    If ib_high or ib_low is NaN for a bar, that bar is allowed (True).

    Args:
        closes: 1-D numpy array of close prices.
        ib_high: 1-D numpy array of IB high levels.
        ib_low: 1-D numpy array of IB low levels.
        buffer_pts: proximity buffer in price points.

    Returns:
        Boolean numpy array (True = allow entry, False = block entry).
    """
    n = len(closes)
    gate = np.ones(n, dtype=bool)

    for i in range(n):
        h = ib_high[i]
        lo = ib_low[i]

        # NaN means no IB computed yet -- allow
        if np.isnan(h) or np.isnan(lo):
            continue

        c = closes[i]

        # Near IB high
        if (h - buffer_pts) <= c <= (h + buffer_pts):
            gate[i] = False
            continue

        # Near IB low
        if (lo - buffer_pts) <= c <= (lo + buffer_pts):
            gate[i] = False

    return gate


# ---------------------------------------------------------------------------
# build_gates_fn for run_sweep
# ---------------------------------------------------------------------------

def build_gates(config, instruments):
    """Build per-instrument gate arrays for a single sweep config.

    Args:
        config: dict with "ib_period" and "buffer_pts" keys.
        instruments: dict instrument_name -> DataFrame.

    Returns:
        dict instrument_name -> boolean gate array (same length as DataFrame).
    """
    gates = {}
    for inst, df in instruments.items():
        times = df.index
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        et_mins = compute_et_minutes(times)
        ib_high, ib_low = compute_initial_balance(
            times, highs, lows, et_mins, config["ib_period"],
        )
        gate = build_ib_gate(closes, ib_high, ib_low, config["buffer_pts"])
        gates[inst] = gate
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    instruments, split_arrays, split_indices = prepare_data()

    configs = [{"label": "None"}]
    for ib_min in [30, 60]:
        for buf in [0, 2, 5, 8, 10]:
            configs.append({
                "label": f"ib{ib_min}_buf{buf}",
                "ib_period": ib_min,
                "buffer_pts": buf,
            })

    run_sweep(
        "Initial Balance",
        STRATEGIES,
        split_arrays,
        split_indices,
        instruments,
        build_gates,
        configs,
        script_name="sr_initial_balance_sweep.py",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
