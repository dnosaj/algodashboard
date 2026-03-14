"""
Parity test: IncrementalSwingTracker (live) vs compute_swing_levels (backtest vectorized).

Loads full MNQ dataset, runs both implementations, and compares bar-by-bar.
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

# --- Setup paths ---
BACKTEST_DIR = Path(__file__).resolve().parent.parent / "backtesting_engine"
STRAT_DIR = BACKTEST_DIR / "strategies"
DATA_DIR = BACKTEST_DIR / "data"
LIVE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(STRAT_DIR))
sys.path.insert(0, str(BACKTEST_DIR))
sys.path.insert(0, str(LIVE_DIR))

# Import vectorized version
from structure_exit_common import compute_swing_levels

# Import IncrementalSwingTracker directly — the module has relative imports
# that pull in config/events, so we use importlib to load just the class
import importlib.util
import types

# Create a minimal engine package stub so relative imports resolve
engine_pkg = types.ModuleType("engine")
engine_pkg.__path__ = [str(LIVE_DIR / "engine")]
engine_pkg.__package__ = "engine"
sys.modules["engine"] = engine_pkg

# Load engine.events (needed by structure_monitor)
events_spec = importlib.util.spec_from_file_location(
    "engine.events", str(LIVE_DIR / "engine" / "events.py"))
events_mod = importlib.util.module_from_spec(events_spec)
sys.modules["engine.events"] = events_mod
events_spec.loader.exec_module(events_mod)

# Load engine.config (needed by structure_monitor)
config_spec = importlib.util.spec_from_file_location(
    "engine.config", str(LIVE_DIR / "engine" / "config.py"))
config_mod = importlib.util.module_from_spec(config_spec)
sys.modules["engine.config"] = config_mod
config_spec.loader.exec_module(config_mod)

# Now load structure_monitor
sm_spec = importlib.util.spec_from_file_location(
    "engine.structure_monitor", str(LIVE_DIR / "engine" / "structure_monitor.py"))
sm_mod = importlib.util.module_from_spec(sm_spec)
sys.modules["engine.structure_monitor"] = sm_mod
sm_spec.loader.exec_module(sm_mod)

IncrementalSwingTracker = sm_mod.IncrementalSwingTracker


def load_mnq_data():
    """Load and concatenate all MNQ 1-min databento CSV files."""
    files = sorted(DATA_DIR.glob("databento_MNQ_1min_*.csv"))
    if not files:
        raise ValueError(f"No MNQ data files found in {DATA_DIR}")

    print(f"Loading {len(files)} MNQ data files...")
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
        print(f"  {f.name}: {len(dfs[-1])} rows")

    df = pd.concat(dfs, ignore_index=True)

    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')

    result = result.set_index('Time')
    result = result[~result.index.duplicated(keep='first')]
    result = result.sort_index()

    print(f"Total bars after dedup+sort: {len(result)}")
    return result


def values_match(vec_val, inc_val):
    """Compare vectorized (float/NaN) with incremental (float/None)."""
    vec_nan = (vec_val is None) or (isinstance(vec_val, float) and math.isnan(vec_val))
    inc_none = inc_val is None

    if vec_nan and inc_none:
        return True
    if vec_nan != inc_none:
        return False
    # Both are real numbers — compare with tiny tolerance
    return abs(vec_val - inc_val) < 1e-6


def run_parity_test():
    # Load data
    df = load_mnq_data()
    highs = df['High'].values
    lows = df['Low'].values
    n = len(highs)

    lookback = 50
    pivot_right = 2

    print(f"\n--- Running parity test ---")
    print(f"Lookback={lookback}, pivot_right={pivot_right}")
    print(f"Total bars: {n}")

    # 1. Vectorized computation
    print("\nRunning vectorized compute_swing_levels...")
    vec_sh, vec_sl = compute_swing_levels(highs, lows, lookback=lookback,
                                           swing_type="pivot", pivot_right=pivot_right)
    print(f"  Done. swing_highs shape: {vec_sh.shape}, swing_lows shape: {vec_sl.shape}")

    # Count non-NaN values in vectorized output
    vec_sh_valid = np.sum(~np.isnan(vec_sh))
    vec_sl_valid = np.sum(~np.isnan(vec_sl))
    print(f"  Non-NaN swing_highs: {vec_sh_valid}, swing_lows: {vec_sl_valid}")

    # First non-NaN index
    first_sh_idx = np.argmax(~np.isnan(vec_sh)) if vec_sh_valid > 0 else -1
    first_sl_idx = np.argmax(~np.isnan(vec_sl)) if vec_sl_valid > 0 else -1
    print(f"  First non-NaN swing_high at bar {first_sh_idx}: {vec_sh[first_sh_idx]:.2f}")
    print(f"  First non-NaN swing_low at bar {first_sl_idx}: {vec_sl[first_sl_idx]:.2f}")

    # 2. Incremental computation
    print("\nRunning incremental IncrementalSwingTracker bar-by-bar...")
    tracker = IncrementalSwingTracker(lookback=lookback, pivot_right=pivot_right)

    inc_sh = [None] * n
    inc_sl = [None] * n

    for i in range(n):
        tracker.update(highs[i], lows[i])
        inc_sh[i] = tracker.swing_high
        inc_sl[i] = tracker.swing_low

    print("  Done.")

    # 3. Compare
    print("\nComparing bar-by-bar...")
    sh_mismatches = []
    sl_mismatches = []

    for i in range(n):
        if not values_match(vec_sh[i], inc_sh[i]):
            sh_mismatches.append(i)
        if not values_match(vec_sl[i], inc_sl[i]):
            sl_mismatches.append(i)

    total_mismatches = len(sh_mismatches) + len(sl_mismatches)

    print(f"\n{'='*60}")
    print(f"PARITY TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total bars compared:        {n}")
    print(f"Swing HIGH mismatches:      {len(sh_mismatches)}")
    print(f"Swing LOW mismatches:       {len(sl_mismatches)}")
    print(f"Total mismatches:           {total_mismatches}")

    if total_mismatches == 0:
        print(f"\nPASS: IncrementalSwingTracker produces IDENTICAL output to compute_swing_levels")
        print(f"      across all {n:,} bars of MNQ data.")
    else:
        print(f"\nFAIL: {total_mismatches} mismatches found!")

        if sh_mismatches:
            print(f"\n--- First 10 swing HIGH mismatches ---")
            for idx in sh_mismatches[:10]:
                v = vec_sh[idx]
                inc = inc_sh[idx]
                ts = df.index[idx]
                print(f"  Bar {idx} ({ts}): vectorized={v}, incremental={inc}, "
                      f"high={highs[idx]:.2f}, low={lows[idx]:.2f}")

        if sl_mismatches:
            print(f"\n--- First 10 swing LOW mismatches ---")
            for idx in sl_mismatches[:10]:
                v = vec_sl[idx]
                inc = inc_sl[idx]
                ts = df.index[idx]
                print(f"  Bar {idx} ({ts}): vectorized={v}, incremental={inc}, "
                      f"high={highs[idx]:.2f}, low={lows[idx]:.2f}")

    # Spot-check: print a few values around first non-NaN to sanity-check
    print(f"\n--- Spot check around first confirmed swing high (bar {first_sh_idx}) ---")
    start = max(0, first_sh_idx - 2)
    end = min(n, first_sh_idx + 3)
    for i in range(start, end):
        v_sh = vec_sh[i]
        i_sh = inc_sh[i]
        v_sl = vec_sl[i]
        i_sl = inc_sl[i]
        match_sh = "OK" if values_match(v_sh, i_sh) else "MISMATCH"
        match_sl = "OK" if values_match(v_sl, i_sl) else "MISMATCH"
        ts = df.index[i]
        print(f"  Bar {i} ({ts}): SH vec={v_sh} inc={i_sh} [{match_sh}] | "
              f"SL vec={v_sl} inc={i_sl} [{match_sl}]")

    return total_mismatches


if __name__ == "__main__":
    mismatches = run_parity_test()
    sys.exit(0 if mismatches == 0 else 1)
