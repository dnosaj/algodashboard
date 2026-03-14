"""
Round 3, Study 2 — VWAP Z-Score Entry Gate Sweep
=================================================
Hypothesis: Block entries when price is >N sigma from session VWAP
(overextended, likely to mean-revert into the stop).

Gate logic:
  1. Compute deviation = close - VWAP each bar.
  2. Session resets at 18:00 ET (futures session open).
  3. Within each session, compute rolling mean and std of deviation
     over a window of bars.
  4. z_score = (deviation - rolling_mean) / rolling_std.
  5. Block entry when |z_score| > max_z.

Sweep: max_z = [1.0, 1.5, 2.0, 2.5, 3.0]  ->  5 configs + baseline
"""

import numpy as np

from sr_common import (
    STRATEGIES,
    compute_et_minutes,
    prepare_data,
    run_sweep,
)


# ---------------------------------------------------------------------------
# VWAP Z-Score computation
# ---------------------------------------------------------------------------

def compute_vwap_zscore(closes, vwap, times, rolling_window=60):
    """Compute z-score of the close-VWAP deviation within each futures session.

    Sessions reset at 18:00 ET (1080 minutes from midnight).  Within each
    session the deviation (close - VWAP) is z-scored using a rolling mean and
    std over *rolling_window* bars.

    Args:
        closes: 1-D numpy array of close prices.
        vwap: 1-D numpy array of VWAP values (may contain NaN).
        times: DatetimeIndex (UTC) aligned with closes/vwap.
        rolling_window: number of bars for rolling mean/std (default 60).

    Returns:
        z_score: 1-D numpy float64 array, same length as closes.
                 Bars with insufficient data or NaN VWAP get z_score = 0.
    """
    n = len(closes)
    z_score = np.zeros(n, dtype=np.float64)

    et_mins = compute_et_minutes(times)

    # --- Forward-fill VWAP NaN within sessions ---
    vwap_filled = vwap.copy().astype(np.float64)

    # Detect session starts: bar where et_mins < previous bar's et_mins
    # (day rollover past midnight) OR et_mins crosses the 1080 boundary.
    SESSION_BOUNDARY = 18 * 60  # 1080

    session_start = np.zeros(n, dtype=bool)
    session_start[0] = True
    for i in range(1, n):
        # Rollover: minutes decreased (crossed midnight) or jumped to/past 18:00
        if et_mins[i] < et_mins[i - 1]:
            session_start[i] = True
        elif et_mins[i - 1] < SESSION_BOUNDARY <= et_mins[i]:
            session_start[i] = True

    # Forward-fill VWAP within each session
    last_valid = np.nan
    for i in range(n):
        if session_start[i]:
            last_valid = np.nan
        if np.isnan(vwap_filled[i]):
            if not np.isnan(last_valid):
                vwap_filled[i] = last_valid
        else:
            last_valid = vwap_filled[i]

    # --- Compute z-score per session ---
    # Use a running buffer per session for rolling mean/std.
    warmup = min(10, rolling_window)
    deviation_buf = np.empty(rolling_window, dtype=np.float64)
    buf_idx = 0       # circular write index
    buf_count = 0     # how many values stored so far (up to rolling_window)
    bars_in_session = 0

    for i in range(n):
        if session_start[i]:
            buf_idx = 0
            buf_count = 0
            bars_in_session = 0

        bars_in_session += 1

        if np.isnan(vwap_filled[i]):
            # Still NaN after forward-fill — leave z_score = 0
            continue

        dev = closes[i] - vwap_filled[i]

        # Write into circular buffer
        deviation_buf[buf_idx] = dev
        buf_idx = (buf_idx + 1) % rolling_window
        buf_count = min(buf_count + 1, rolling_window)

        # Need at least warmup bars in the session
        if bars_in_session <= warmup:
            continue

        # Compute rolling mean and std from the buffer
        active = deviation_buf[:buf_count]
        roll_mean = np.mean(active)
        roll_std = np.std(active, ddof=0)

        if roll_std == 0:
            z_score[i] = 0.0
        else:
            z_score[i] = (dev - roll_mean) / roll_std

    return z_score


# ---------------------------------------------------------------------------
# Gate builder
# ---------------------------------------------------------------------------

def build_vwap_zscore_gate(z_score, max_z):
    """Build a boolean gate: allow entry when |z_score| <= max_z.

    Args:
        z_score: 1-D numpy array of z-score values.
        max_z: threshold — block if |z_score| > max_z.

    Returns:
        Boolean numpy array (True = allow entry, False = block entry).
        NaN z_scores map to True (allow).
    """
    gate = np.ones(len(z_score), dtype=bool)
    finite = np.isfinite(z_score)
    gate[finite] = np.abs(z_score[finite]) <= max_z
    return gate


# ---------------------------------------------------------------------------
# build_gates_fn for run_sweep
# ---------------------------------------------------------------------------

def build_gates(config, instruments):
    """Build per-instrument VWAP z-score gate arrays for a single sweep config.

    Args:
        config: dict with "max_z" key.
        instruments: dict instrument_name -> DataFrame with Close, VWAP columns.

    Returns:
        dict instrument_name -> boolean gate array (same length as DataFrame).
    """
    gates = {}
    for inst, df in instruments.items():
        closes = df["Close"].values
        vwap = df["VWAP"].values
        times = df.index
        z_score = compute_vwap_zscore(closes, vwap, times)
        gate = build_vwap_zscore_gate(z_score, config["max_z"])
        gates[inst] = gate
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    instruments, split_arrays, split_indices = prepare_data()

    # Check VWAP NaN rates
    for inst, df in instruments.items():
        nan_pct = df["VWAP"].isna().mean() * 100
        print(f"  {inst} VWAP NaN: {nan_pct:.2f}%")
        if nan_pct > 1:
            print(f"  WARNING: {inst} has {nan_pct:.1f}% NaN VWAP values")

    configs = [{"label": "None"}]
    for mz in [1.0, 1.5, 2.0, 2.5, 3.0]:
        configs.append({"label": f"z{mz}", "max_z": mz})

    run_sweep(
        "VWAP Z-Score",
        STRATEGIES,
        split_arrays,
        split_indices,
        instruments,
        build_gates,
        configs,
        script_name="sr_vwap_zscore_sweep.py",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
