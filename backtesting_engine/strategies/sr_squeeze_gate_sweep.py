"""
Round 3, Study 3 — Squeeze Momentum (TTM Squeeze) Entry Gate Sweep
===================================================================
Hypothesis: Block entries during BB-inside-KC squeeze (choppy, directionless
periods where SM+RSI whipsaw).

Squeeze ON: Bollinger Band upper < Keltner Channel upper AND
            Bollinger Band lower > Keltner Channel lower.

Gate: block entries during squeeze AND for min_bars_off bars after squeeze
releases.  Allow entries when not in squeeze and min_bars_off bars have
passed since the last squeeze.

Sweep: kc_mult = [1.0, 1.5, 2.0] x min_bars_off = [0, 5, 10] -> 9 configs + baseline

bb_len=20, bb_mult=2.0, kc_len=20 held constant.  Only kc_mult and
min_bars_off vary.  Smaller kc_mult = tighter KC = more frequent squeeze =
more blocking.
"""

import numpy as np

from sr_common import (
    STRATEGIES,
    compute_atr_wilder,
    prepare_data,
    run_sweep,
)


# ---------------------------------------------------------------------------
# Squeeze detection
# ---------------------------------------------------------------------------

def compute_squeeze(closes, highs, lows,
                    bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5):
    """Detect TTM Squeeze — Bollinger Bands inside Keltner Channels.

    Squeeze is ON (True) when the BB envelope sits entirely inside the KC
    envelope, indicating low-volatility compression.

    Args:
        closes: 1-D numpy array of close prices.
        highs:  1-D numpy array of high prices.
        lows:   1-D numpy array of low prices.
        bb_len: Bollinger Band SMA lookback (default 20).
        bb_mult: Bollinger Band standard-deviation multiplier (default 2.0).
        kc_len: Keltner Channel SMA lookback (default 20).
        kc_mult: Keltner Channel ATR multiplier (default 1.5).

    Returns:
        Boolean numpy array (True = squeeze ON, False = squeeze OFF).
        First bb_len bars are always False (insufficient data).
    """
    n = len(closes)
    squeeze_on = np.zeros(n, dtype=bool)

    # --- Bollinger Bands ---
    # Rolling SMA and StdDev over bb_len
    bb_sma = np.empty(n)
    bb_std = np.empty(n)
    bb_sma[:] = np.nan
    bb_std[:] = np.nan

    for i in range(bb_len - 1, n):
        window = closes[i - bb_len + 1 : i + 1]
        bb_sma[i] = np.mean(window)
        bb_std[i] = np.std(window, ddof=0)  # population std (matches Pine)

    bb_upper = bb_sma + bb_mult * bb_std
    bb_lower = bb_sma - bb_mult * bb_std

    # --- Keltner Channels ---
    # SMA of close for KC midline
    kc_sma = np.empty(n)
    kc_sma[:] = np.nan
    for i in range(kc_len - 1, n):
        kc_sma[i] = np.mean(closes[i - kc_len + 1 : i + 1])

    atr = compute_atr_wilder(highs, lows, closes, period=kc_len)

    kc_upper = kc_sma + kc_mult * atr
    kc_lower = kc_sma - kc_mult * atr

    # --- Squeeze: BB inside KC ---
    for i in range(bb_len, n):
        if (not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i])
                and bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]):
            squeeze_on[i] = True

    return squeeze_on


# ---------------------------------------------------------------------------
# Gate builder
# ---------------------------------------------------------------------------

def build_squeeze_gate(squeeze_on, min_bars_off=0):
    """Build a boolean gate that blocks during squeeze and for min_bars_off
    bars after squeeze releases.

    A bar is blocked (gate = False) when squeeze is ON, or when fewer than
    min_bars_off bars have elapsed since the last squeeze bar.

    Args:
        squeeze_on: boolean array — True when squeeze is active.
        min_bars_off: number of additional bars to keep the gate closed
                      after the squeeze releases (default 0 = no delay).

    Returns:
        Boolean numpy array (True = allow entry, False = block entry).
    """
    n = len(squeeze_on)
    gate = np.ones(n, dtype=bool)

    bars_since_squeeze = min_bars_off + 1  # start with "no recent squeeze"

    for i in range(n):
        if squeeze_on[i]:
            gate[i] = False
            bars_since_squeeze = 0
        elif bars_since_squeeze < min_bars_off:
            gate[i] = False
            bars_since_squeeze += 1
        else:
            # Not in squeeze and enough bars have passed — allow
            bars_since_squeeze += 1

    return gate


# ---------------------------------------------------------------------------
# build_gates_fn for run_sweep
# ---------------------------------------------------------------------------

def build_gates(config, instruments):
    """Build per-instrument gate arrays for a single sweep config.

    Args:
        config: dict with "kc_mult" and "min_bars_off" keys.
        instruments: dict instrument_name -> DataFrame with Close, High, Low.

    Returns:
        dict instrument_name -> boolean gate array (same length as DataFrame).
    """
    gates = {}
    for inst, df in instruments.items():
        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values
        squeeze_on = compute_squeeze(closes, highs, lows, kc_mult=config["kc_mult"])
        gate = build_squeeze_gate(squeeze_on, min_bars_off=config["min_bars_off"])
        gates[inst] = gate
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    instruments, split_arrays, split_indices = prepare_data()

    configs = [{"label": "None"}]
    for kc in [1.0, 1.5, 2.0]:
        for mbo in [0, 5, 10]:
            configs.append({
                "label": f"kc{kc}_off{mbo}",
                "kc_mult": kc,
                "min_bars_off": mbo,
            })

    run_sweep(
        "Squeeze Momentum",
        STRATEGIES,
        split_arrays,
        split_indices,
        instruments,
        build_gates,
        configs,
        script_name="sr_squeeze_gate_sweep.py",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
