"""
Round 3, Study 4 — Leledc Exhaustion Entry Gate Sweep
=====================================================
Hypothesis: Block entries on exhaustion bars — the "last buyer" in an uptrend
(or last seller in a downtrend).  Count consecutive bars where
close > close[lookback].  When count reaches maj_qual, that bar is exhaustion.

Gate: single boolean (not directional) — blocks if EITHER direction is exhausted,
and stays blocked for `persistence` bars after detection.

Sweep: maj_qual = [5, 6, 7, 8, 9] x persistence = [1, 3, 5]  ->  15 configs + baseline
"""

import numpy as np

from sr_common import (
    STRATEGIES,
    prepare_data,
    run_sweep,
)


# ---------------------------------------------------------------------------
# Leledc exhaustion detection
# ---------------------------------------------------------------------------

def compute_leledc_exhaustion(closes, maj_qual=6, lookback=4):
    """Detect Leledc exhaustion bars.

    Bull exhaustion: consecutive bars where close[j] > close[j - lookback]
    reaches *maj_qual*.

    Bear exhaustion: consecutive bars where close[j] < close[j - lookback]
    reaches *maj_qual*.

    The counter resets when the condition breaks.

    Args:
        closes: 1-D numpy array of close prices.
        maj_qual: number of consecutive bars needed to trigger exhaustion.
        lookback: how far back to compare each close (default 4).

    Returns:
        (bull_exhaustion, bear_exhaustion) — boolean numpy arrays, same length
        as *closes*.
    """
    n = len(closes)
    bull_exhaustion = np.zeros(n, dtype=bool)
    bear_exhaustion = np.zeros(n, dtype=bool)

    bull_count = 0
    bear_count = 0

    for i in range(n):
        if i < lookback:
            # Not enough history — leave False, counters stay 0
            continue

        # --- Bull count ---
        if closes[i] > closes[i - lookback]:
            bull_count += 1
        else:
            bull_count = 0

        if bull_count >= maj_qual:
            bull_exhaustion[i] = True

        # --- Bear count ---
        if closes[i] < closes[i - lookback]:
            bear_count += 1
        else:
            bear_count = 0

        if bear_count >= maj_qual:
            bear_exhaustion[i] = True

    return bull_exhaustion, bear_exhaustion


# ---------------------------------------------------------------------------
# Gate builder
# ---------------------------------------------------------------------------

def build_leledc_gate(bull_ex, bear_ex, persistence):
    """Build a boolean gate that blocks for *persistence* bars after exhaustion.

    A bar is blocked (gate = False) if either bull or bear exhaustion was
    detected within the last *persistence* bars (inclusive of the detection bar).

    Args:
        bull_ex: boolean array — True on bull exhaustion bars.
        bear_ex: boolean array — True on bear exhaustion bars.
        persistence: how many bars to keep the gate closed after detection.

    Returns:
        Boolean numpy array (True = allow entry, False = block entry).
    """
    n = len(bull_ex)
    gate = np.ones(n, dtype=bool)
    either = bull_ex | bear_ex

    bars_since = persistence + 1  # start with "no recent exhaustion"

    for i in range(n):
        if either[i]:
            bars_since = 0
        if bars_since < persistence:
            gate[i] = False
            bars_since += 1
        elif either[i]:
            # Detection bar itself is also blocked
            gate[i] = False
            bars_since += 1

    return gate


# ---------------------------------------------------------------------------
# build_gates_fn for run_sweep
# ---------------------------------------------------------------------------

def build_gates(config, instruments):
    """Build per-instrument gate arrays for a single sweep config.

    Args:
        config: dict with "maj_qual" and "persistence" keys.
        instruments: dict instrument_name -> DataFrame with Close column.

    Returns:
        dict instrument_name -> boolean gate array (same length as DataFrame).
    """
    gates = {}
    for inst, df in instruments.items():
        closes = df["Close"].values
        bull_ex, bear_ex = compute_leledc_exhaustion(closes, config["maj_qual"])
        gate = build_leledc_gate(bull_ex, bear_ex, config["persistence"])
        gates[inst] = gate
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    instruments, split_arrays, split_indices = prepare_data()

    configs = [{"label": "None"}]
    for mq in [5, 6, 7, 8, 9]:
        for pers in [1, 3, 5]:
            configs.append({
                "label": f"mq{mq}_p{pers}",
                "maj_qual": mq,
                "persistence": pers,
            })

    run_sweep(
        "Leledc Exhaustion",
        STRATEGIES,
        split_arrays,
        split_indices,
        instruments,
        build_gates,
        configs,
        script_name="sr_leledc_exhaustion_sweep.py",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
