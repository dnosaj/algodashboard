"""
Round 3, Study 6 — Intraday S/R (Rolling Pivots + Multi-Window + Clustering)
=============================================================================
Hypothesis: Block entries near confirmed intraday pivot highs/lows.
Score levels by multi-timeframe confirmation and clustering.

Layers:
  1. Pivot detection — scipy.ndimage maximum/minimum_filter1d for O(N) pivot
     detection across 4 windows (10, 20, 30, 50 bars ~ 10m/20m/30m/1h).
     Pivots are confirmed with a look-ahead delay of `window` bars.
  2. Level scoring — each pivot starts score=1.  +1 for each other window
     that detected a pivot at the same level (within cluster_tolerance *
     median ATR), +1 for each prior pivot from any window clustering at the
     same level.  Capped at 5.
  3. Nearest level lookup — for each bar, find the nearest confirmed
     resistance above and support below with score >= min_score, using
     bisect for O(log M) lookup.
  4. Gate — block entry if nearest resistance or support is within
     buffer_pts of the current close.

Sweep: buffer_pts = [2, 5, 8, 10, 15] x min_score = [1, 2, 3]
       -> 15 configs + baseline
"""

import bisect

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from sr_common import (
    STRATEGIES,
    compute_atr_wilder,
    prepare_data,
    run_sweep,
)


# ---------------------------------------------------------------------------
# Windows used for pivot detection
# ---------------------------------------------------------------------------

PIVOT_WINDOWS = [10, 20, 30, 50]


# ---------------------------------------------------------------------------
# Layer 1: Pivot detection
# ---------------------------------------------------------------------------

def compute_pivots_fast(highs, lows, window):
    """Detect pivot highs and pivot lows using scipy rolling extrema.

    A pivot high at index j means highs[j] == max(highs[j-window : j+window+1]).
    A pivot low  at index j means lows[j]  == min(lows[j-window : j+window+1]).

    Each pivot is confirmed at bar j + window (no look-ahead).

    Args:
        highs: 1-D numpy array of high prices.
        lows: 1-D numpy array of low prices.
        window: half-window size for pivot detection.

    Returns:
        pivot_highs: list of (price, confirm_idx) tuples.
        pivot_lows: list of (price, confirm_idx) tuples.
    """
    n = len(highs)
    kernel = 2 * window + 1

    # maximum_filter1d with size=kernel centered on each bar
    max_filt = maximum_filter1d(highs, size=kernel, mode="nearest")
    min_filt = minimum_filter1d(lows, size=kernel, mode="nearest")

    pivot_highs = []
    pivot_lows = []

    for j in range(n):
        confirm_idx = j + window
        if confirm_idx >= n:
            break
        if highs[j] == max_filt[j]:
            pivot_highs.append((highs[j], confirm_idx))
        if lows[j] == min_filt[j]:
            pivot_lows.append((lows[j], confirm_idx))

    return pivot_highs, pivot_lows


# ---------------------------------------------------------------------------
# Layer 2: Level scoring
# ---------------------------------------------------------------------------

def score_pivots(all_pivot_highs_by_window, all_pivot_lows_by_window, atr,
                 cluster_tolerance_atr_mult=1.0):
    """Score pivot levels by multi-window confirmation and clustering.

    Each pivot starts with score=1.
      +1 for each OTHER window that detected a pivot at the same level
         (within cluster_tolerance_atr_mult * median ATR).
      +1 for each PRIOR pivot (from any window) clustering at the same level.
    Score capped at 5.

    Args:
        all_pivot_highs_by_window: dict[window -> list of (price, confirm_idx)].
        all_pivot_lows_by_window: dict[window -> list of (price, confirm_idx)].
        atr: 1-D numpy array of ATR values (from compute_atr_wilder).
        cluster_tolerance_atr_mult: multiplier on median ATR for clustering.

    Returns:
        scored_highs: list of (price, confirm_idx, score) sorted by confirm_idx.
        scored_lows: list of (price, confirm_idx, score) sorted by confirm_idx.
    """
    median_atr = np.nanmedian(atr)
    tol = cluster_tolerance_atr_mult * median_atr

    def _score_one_side(all_by_window):
        # Flatten all pivots with their originating window
        all_pivots = []  # (price, confirm_idx, window)
        for w, pivots in all_by_window.items():
            for price, cidx in pivots:
                all_pivots.append((price, cidx, w))

        # Sort by confirm_idx for chronological processing
        all_pivots.sort(key=lambda x: (x[1], x[0]))

        scored = []
        for i, (price, cidx, win) in enumerate(all_pivots):
            score = 1

            # Check OTHER windows for pivots at same level
            confirmed_windows = {win}
            for j, (p2, c2, w2) in enumerate(all_pivots):
                if j == i:
                    continue
                if w2 in confirmed_windows:
                    continue
                if abs(p2 - price) <= tol:
                    score += 1
                    confirmed_windows.add(w2)

            # Check PRIOR pivots from ANY window clustering at same level
            for j in range(i):
                p2, c2, w2 = all_pivots[j]
                if c2 < cidx and abs(p2 - price) <= tol:
                    score += 1

            score = min(score, 5)
            scored.append((price, cidx, score))

        # Sort by confirm_idx
        scored.sort(key=lambda x: x[1])
        return scored

    scored_highs = _score_one_side(all_pivot_highs_by_window)
    scored_lows = _score_one_side(all_pivot_lows_by_window)

    return scored_highs, scored_lows


# ---------------------------------------------------------------------------
# Layer 3: Nearest level lookup
# ---------------------------------------------------------------------------

def precompute_nearest_levels(n, scored_highs, scored_lows, closes, min_score):
    """For each bar, find distance to nearest confirmed S/R with score >= min_score.

    Uses bisect for O(log M) lookup per bar, iterating chronologically and
    adding newly confirmed pivots as we reach their confirmation bar.

    Args:
        n: number of bars.
        scored_highs: list of (price, confirm_idx, score) sorted by confirm_idx.
        scored_lows: list of (price, confirm_idx, score) sorted by confirm_idx.
        closes: 1-D numpy array of close prices.
        min_score: minimum score for a level to be considered.

    Returns:
        nearest_res_dist: 1-D numpy array (positive distance to nearest
                          resistance above close, or np.inf if none).
        nearest_sup_dist: 1-D numpy array (positive distance to nearest
                          support below close, or np.inf if none).
    """
    nearest_res_dist = np.full(n, np.inf, dtype=np.float64)
    nearest_sup_dist = np.full(n, np.inf, dtype=np.float64)

    # Filter by min_score and sort by confirm_idx
    res_levels = [(p, cidx) for p, cidx, s in scored_highs if s >= min_score]
    sup_levels = [(p, cidx) for p, cidx, s in scored_lows if s >= min_score]
    res_levels.sort(key=lambda x: x[1])
    sup_levels.sort(key=lambda x: x[1])

    # Active sorted price lists for bisect lookup
    active_res = []  # sorted list of resistance prices
    active_sup = []  # sorted list of support prices

    res_ptr = 0
    sup_ptr = 0

    for i in range(n):
        # Add all pivots confirmed at or before bar i
        while res_ptr < len(res_levels) and res_levels[res_ptr][1] <= i:
            price = res_levels[res_ptr][0]
            bisect.insort(active_res, price)
            res_ptr += 1

        while sup_ptr < len(sup_levels) and sup_levels[sup_ptr][1] <= i:
            price = sup_levels[sup_ptr][0]
            bisect.insort(active_sup, price)
            sup_ptr += 1

        c = closes[i]

        # Nearest resistance ABOVE close
        if active_res:
            idx_r = bisect.bisect_right(active_res, c)
            if idx_r < len(active_res):
                nearest_res_dist[i] = active_res[idx_r] - c

        # Nearest support BELOW close
        if active_sup:
            idx_s = bisect.bisect_left(active_sup, c)
            if idx_s > 0:
                nearest_sup_dist[i] = c - active_sup[idx_s - 1]

    return nearest_res_dist, nearest_sup_dist


# ---------------------------------------------------------------------------
# Gate builder
# ---------------------------------------------------------------------------

def build_pivot_gate(nearest_res_dist, nearest_sup_dist, buffer_pts):
    """Build a boolean gate: block if nearest S/R is within buffer_pts.

    Args:
        nearest_res_dist: 1-D numpy array (distance to nearest resistance).
        nearest_sup_dist: 1-D numpy array (distance to nearest support).
        buffer_pts: distance threshold in points.

    Returns:
        Boolean numpy array (True = allow entry, False = block entry).
    """
    gate = (nearest_res_dist > buffer_pts) & (nearest_sup_dist > buffer_pts)
    return gate


# ---------------------------------------------------------------------------
# Pre-computation cache
# ---------------------------------------------------------------------------

_precomputed = {}  # instrument -> (scored_highs, scored_lows, closes)


def precompute_pivots(instruments):
    """Pre-compute scored pivot levels for all instruments and windows.

    Populates the module-level _precomputed cache so the sweep loop only
    needs to vary the nearest-level lookup parameters.
    """
    for inst, df in instruments.items():
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        atr = compute_atr_wilder(highs, lows, closes, period=14)

        all_ph = {}
        all_pl = {}
        for w in PIVOT_WINDOWS:
            ph, pl = compute_pivots_fast(highs, lows, w)
            all_ph[w] = ph
            all_pl[w] = pl

        scored_h, scored_l = score_pivots(all_ph, all_pl, atr)
        _precomputed[inst] = (scored_h, scored_l, closes)


# ---------------------------------------------------------------------------
# build_gates_fn for run_sweep
# ---------------------------------------------------------------------------

def build_gates(config, instruments):
    """Build per-instrument pivot proximity gate arrays for a single sweep config.

    Args:
        config: dict with "buffer_pts" and "min_score" keys.
        instruments: dict instrument_name -> DataFrame.

    Returns:
        dict instrument_name -> boolean gate array (same length as DataFrame).
    """
    gates = {}
    for inst in instruments:
        scored_h, scored_l, closes = _precomputed[inst]
        n = len(closes)
        res_dist, sup_dist = precompute_nearest_levels(
            n, scored_h, scored_l, closes, config["min_score"]
        )
        gate = build_pivot_gate(res_dist, sup_dist, config["buffer_pts"])
        gates[inst] = gate
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    instruments, split_arrays, split_indices = prepare_data()

    print("Pre-computing intraday pivots...")
    precompute_pivots(instruments)
    for inst in instruments:
        sh, sl, _ = _precomputed[inst]
        print(f"  {inst}: {len(sh)} resistance pivots, {len(sl)} support pivots")

    configs = [{"label": "None"}]
    for buf in [2, 5, 8, 10, 15]:
        for ms in [1, 2, 3]:
            configs.append({
                "label": f"buf{buf}_ms{ms}",
                "buffer_pts": buf,
                "min_score": ms,
            })

    run_sweep(
        "Intraday Pivots",
        STRATEGIES,
        split_arrays,
        split_indices,
        instruments,
        build_gates,
        configs,
        script_name="sr_intraday_pivots_sweep.py",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
