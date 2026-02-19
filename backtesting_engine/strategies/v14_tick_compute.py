"""
v14_tick_compute.py — Process 42.6M NQ futures ticks into per-1-min-bar
microstructure features and save as CSV.

Run ONCE, then reuse the features CSV from v14_tick_test.py.

Output columns:
  time, buy_impact, sell_impact, impact_log_ratio, impact_ema3,
  buy_sweep_pts, sell_sweep_pts, top_ratio, bot_ratio, tick_cv,
  buy_iceberg_vol, sell_iceberg_vol

Usage:
  python3 v14_tick_compute.py
  python3 v14_tick_compute.py --tick-file data/databento_NQ_ticks_*.parquet
  python3 v14_tick_compute.py --output custom_features.csv
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_ticks(tick_file=None):
    """Load tick parquet, cast types, return DataFrame."""
    if tick_file is None:
        candidates = sorted(glob.glob(
            str(DATA_DIR / "databento_NQ_ticks_2025-08-17_*.parquet")
        ))
        if not candidates:
            raise FileNotFoundError(
                "No databento_NQ_ticks_2025-08-17_*.parquet found in "
                f"{DATA_DIR}"
            )
        tick_file = candidates[0]

    print(f"Loading ticks from: {tick_file}")
    t0 = time.time()
    df = pd.read_parquet(tick_file)
    print(f"  Loaded {len(df):,} ticks in {time.time() - t0:.1f}s")

    # CRITICAL: cast size to int64 immediately
    df["size"] = df["size"].astype(np.int64)

    return df


def build_bar_index(df: pd.DataFrame):
    """Floor ts_event to 1-min bars, factorize for integer indexing.

    Returns:
        bars_raw: numpy array of datetime64 bar timestamps (per tick)
        bar_idx: integer index per tick into unique_bars
        unique_bars: sorted unique bar timestamps
        n_bars: number of unique bars
        ts_ns: nanosecond timestamps as int64
    """
    ts = pd.to_datetime(df["ts_event"], utc=True)
    bars_raw = ts.dt.floor("1min").values  # numpy datetime64 array
    bar_idx, unique_bars = pd.factorize(bars_raw, sort=True)
    n_bars = len(unique_bars)
    # Nanosecond timestamps as int64 for time arithmetic
    ts_ns = df["ts_event"].values.astype(np.int64)
    print(f"  {n_bars:,} unique 1-min bars")
    return bars_raw, bar_idx, unique_bars, n_bars, ts_ns


# ---------------------------------------------------------------------------
# Feature 1: Price Impact Asymmetry
# ---------------------------------------------------------------------------
def compute_price_impact(prices, sides, bars_raw, bar_idx, n_bars):
    """Compute buy_impact, sell_impact, impact_log_ratio, impact_ema3."""
    t0 = time.time()

    # Price changes (absolute)
    price_change = np.abs(np.diff(prices, prepend=prices[0]))
    price_change[0] = 0.0

    # Zero out cross-bar transitions
    cross_bar = np.concatenate([[True], bars_raw[1:] != bars_raw[:-1]])
    price_change[cross_bar] = 0.0

    # Tag by side
    buy_pc = np.where(sides == "B", price_change, 0.0)
    sell_pc = np.where(sides == "A", price_change, 0.0)
    buy_count = (sides == "B").astype(np.float64)
    sell_count = (sides == "A").astype(np.float64)

    # Aggregate per bar using bincount
    buy_pc_sum = np.bincount(bar_idx, weights=buy_pc, minlength=n_bars)
    sell_pc_sum = np.bincount(bar_idx, weights=sell_pc, minlength=n_bars)
    buy_n = np.bincount(bar_idx, weights=buy_count, minlength=n_bars)
    sell_n = np.bincount(bar_idx, weights=sell_count, minlength=n_bars)

    buy_impact = np.where(buy_n > 0, buy_pc_sum / buy_n, 0.0)
    sell_impact = np.where(sell_n > 0, sell_pc_sum / sell_n, 0.0)

    # Log ratio: log(buy/sell) where both > 0, else 0
    both_pos = (buy_impact > 0) & (sell_impact > 0)
    impact_log_ratio = np.where(
        both_pos, np.log(buy_impact / np.where(sell_impact > 0, sell_impact, 1.0)), 0.0
    )

    # EMA with alpha = 0.5 (span=3 -> alpha=2/(3+1)=0.5)
    alpha = 0.5
    impact_ema3 = np.zeros(n_bars)
    if n_bars > 0:
        impact_ema3[0] = impact_log_ratio[0]
        for i in range(1, n_bars):
            impact_ema3[i] = alpha * impact_log_ratio[i] + (1 - alpha) * impact_ema3[i - 1]

    elapsed = time.time() - t0
    print(f"  Feature 1 (Price Impact):    {elapsed:.1f}s")
    print(f"    buy_impact  mean={buy_impact.mean():.6f}  nonzero={np.count_nonzero(buy_impact):,}")
    print(f"    sell_impact mean={sell_impact.mean():.6f}  nonzero={np.count_nonzero(sell_impact):,}")

    return buy_impact, sell_impact, impact_log_ratio, impact_ema3


# ---------------------------------------------------------------------------
# Feature 2: Sweep Velocity Detection
# ---------------------------------------------------------------------------
def compute_sweeps(prices, sides, bars_raw, bar_idx, ts_ns, n_bars):
    """Detect buy/sell sweeps, compute total price range per bar."""
    t0 = time.time()

    # Pairwise conditions between consecutive ticks
    same_side = sides[1:] == sides[:-1]
    same_bar = bars_raw[1:] == bars_raw[:-1]
    time_ok = (ts_ns[1:] - ts_ns[:-1]) < 50_000_000  # 50ms
    price_up = prices[1:] > prices[:-1]
    price_down = prices[1:] < prices[:-1]
    is_buy_next = sides[1:] == "B"
    is_sell_next = sides[1:] == "A"

    buy_cont = same_side & same_bar & time_ok & price_up & is_buy_next
    sell_cont = same_side & same_bar & time_ok & price_down & is_sell_next
    sweep_cont = buy_cont | sell_cont

    # Find runs of True using diff trick
    padded = np.concatenate([[False], sweep_cont, [False]])
    d = np.diff(padded.astype(np.int8))
    run_starts = np.where(d == 1)[0]
    run_ends = np.where(d == -1)[0]
    run_lengths = run_ends - run_starts

    # Filter: run length >= 2 means 3+ ticks in sweep
    valid = run_lengths >= 2
    v_starts = run_starts[valid]
    v_ends = run_ends[valid]

    buy_sweep_pts = np.zeros(n_bars)
    sell_sweep_pts = np.zeros(n_bars)

    for vs, ve in zip(v_starts, v_ends):
        tick_start = vs
        tick_end = ve  # inclusive: ticks vs..ve
        sweep_side = sides[tick_start]
        bar_i = bar_idx[tick_start]
        price_range = abs(float(prices[tick_end]) - float(prices[tick_start]))

        if sweep_side == "B":
            buy_sweep_pts[bar_i] += price_range
        else:
            sell_sweep_pts[bar_i] += price_range

    elapsed = time.time() - t0
    n_valid = int(valid.sum())
    print(f"  Feature 2 (Sweeps):          {elapsed:.1f}s")
    print(f"    {n_valid:,} sweeps detected (buy_cont={buy_cont.sum():,}, sell_cont={sell_cont.sum():,})")
    print(f"    buy_sweep_pts  nonzero_bars={np.count_nonzero(buy_sweep_pts):,}")
    print(f"    sell_sweep_pts nonzero_bars={np.count_nonzero(sell_sweep_pts):,}")

    return buy_sweep_pts, sell_sweep_pts


# ---------------------------------------------------------------------------
# Feature 3: Footprint Imbalance at Bar Extremes
# ---------------------------------------------------------------------------
def compute_footprint_imbalance(prices, sides, sizes, bar_idx, n_bars):
    """Compute top_ratio and bot_ratio at bar high/low."""
    t0 = time.time()

    # Compute bar high and bar low using pandas groupby (fast for this)
    bar_extremes = pd.DataFrame({"price": prices, "bar_idx": bar_idx})
    grouped = bar_extremes.groupby("bar_idx")["price"]
    bar_high = grouped.max().reindex(range(n_bars), fill_value=np.nan).values
    bar_low = grouped.min().reindex(range(n_bars), fill_value=np.nan).values

    # Map back to tick level
    tick_bar_high = bar_high[bar_idx]
    tick_bar_low = bar_low[bar_idx]

    # Ticks within 0.25 points (1 NQ tick) of bar extremes
    at_high = prices >= tick_bar_high - 0.25
    at_low = prices <= tick_bar_low + 0.25

    sizes_f = sizes.astype(np.float64)

    buy_at_high = np.where((sides == "B") & at_high, sizes_f, 0.0)
    sell_at_high = np.where((sides == "A") & at_high, sizes_f, 0.0)
    buy_at_low = np.where((sides == "B") & at_low, sizes_f, 0.0)
    sell_at_low = np.where((sides == "A") & at_low, sizes_f, 0.0)

    top_buy = np.bincount(bar_idx, weights=buy_at_high, minlength=n_bars)
    top_sell = np.bincount(bar_idx, weights=sell_at_high, minlength=n_bars)
    bot_sell = np.bincount(bar_idx, weights=sell_at_low, minlength=n_bars)
    bot_buy = np.bincount(bar_idx, weights=buy_at_low, minlength=n_bars)

    top_ratio = top_buy / (top_sell + 1)
    bot_ratio = bot_sell / (bot_buy + 1)

    elapsed = time.time() - t0
    print(f"  Feature 3 (Footprint):       {elapsed:.1f}s")
    print(f"    top_ratio mean={top_ratio.mean():.4f}  bot_ratio mean={bot_ratio.mean():.4f}")

    return top_ratio, bot_ratio


# ---------------------------------------------------------------------------
# Feature 4: Tick Arrival Time CV
# ---------------------------------------------------------------------------
def compute_tick_cv(bars_raw, bar_idx, ts_ns, n_bars):
    """Coefficient of variation of inter-tick arrival times per bar."""
    t0 = time.time()

    # Inter-tick intervals in nanoseconds
    dt = np.diff(ts_ns, prepend=ts_ns[0]).astype(np.float64)
    dt[0] = np.nan

    # Mark cross-bar boundaries
    cross_bar = np.concatenate([[True], bars_raw[1:] != bars_raw[:-1]])
    dt[cross_bar] = np.nan

    # Valid = within bar, positive interval
    valid_mask = (~np.isnan(dt)) & (dt > 0)
    dt_clean = np.where(valid_mask, dt, 0.0)

    # Per-bar sum, sum-of-squares, count
    dt_sum = np.bincount(bar_idx, weights=dt_clean, minlength=n_bars)
    dt_sq = np.where(valid_mask, dt ** 2, 0.0)
    dt_sq_sum = np.bincount(bar_idx, weights=dt_sq, minlength=n_bars)
    dt_n = np.bincount(bar_idx, weights=valid_mask.astype(np.float64), minlength=n_bars)

    dt_mean = np.where(dt_n > 0, dt_sum / dt_n, 0.0)
    dt_var = np.where(dt_n > 1, dt_sq_sum / dt_n - dt_mean ** 2, 0.0)
    dt_var = np.maximum(dt_var, 0.0)  # numerical stability
    dt_std = np.sqrt(dt_var)

    tick_cv = np.where(dt_mean > 0, dt_std / dt_mean, 0.0)

    elapsed = time.time() - t0
    print(f"  Feature 4 (Tick CV):         {elapsed:.1f}s")
    print(f"    tick_cv mean={tick_cv.mean():.4f}  median={np.median(tick_cv):.4f}")

    return tick_cv


# ---------------------------------------------------------------------------
# Feature 5: Iceberg Detection
# ---------------------------------------------------------------------------
def compute_icebergs(prices, sides, sizes, bars_raw, bar_idx, ts_ns, n_bars):
    """Detect iceberg orders: 4+ consecutive ticks, same size/price/side/bar,
    within 500ms of each other."""
    t0 = time.time()

    # Pairwise conditions
    same_size = sizes[1:] == sizes[:-1]
    same_price = prices[1:] == prices[:-1]
    same_side = sides[1:] == sides[:-1]
    same_bar = bars_raw[1:] == bars_raw[:-1]
    ice_time_ok = (ts_ns[1:] - ts_ns[:-1]) < 500_000_000  # 500ms

    iceberg_cont = same_size & same_price & same_side & same_bar & ice_time_ok

    # Find runs of length >= 3 (meaning 4+ ticks)
    padded_ice = np.concatenate([[False], iceberg_cont, [False]])
    d_ice = np.diff(padded_ice.astype(np.int8))
    ice_starts = np.where(d_ice == 1)[0]
    ice_ends = np.where(d_ice == -1)[0]
    ice_lengths = ice_ends - ice_starts
    valid_ice = ice_lengths >= 3

    buy_iceberg_vol = np.zeros(n_bars)
    sell_iceberg_vol = np.zeros(n_bars)

    vis_arr = ice_starts[valid_ice]
    vie_arr = ice_ends[valid_ice]

    for vis, vie in zip(vis_arr, vie_arr):
        tick_start = vis
        tick_end = vie  # inclusive
        n_ticks = tick_end - tick_start + 1
        ice_side = sides[tick_start]
        bar_i = bar_idx[tick_start]
        ice_vol = int(sizes[tick_start]) * n_ticks

        if ice_side == "B":
            buy_iceberg_vol[bar_i] += ice_vol
        else:
            sell_iceberg_vol[bar_i] += ice_vol

    elapsed = time.time() - t0
    n_valid_ice = int(valid_ice.sum())
    print(f"  Feature 5 (Icebergs):        {elapsed:.1f}s")
    print(f"    {n_valid_ice:,} iceberg patterns detected")
    print(f"    buy_iceberg_vol  total={buy_iceberg_vol.sum():.0f}  nonzero_bars={np.count_nonzero(buy_iceberg_vol):,}")
    print(f"    sell_iceberg_vol total={sell_iceberg_vol.sum():.0f}  nonzero_bars={np.count_nonzero(sell_iceberg_vol):,}")

    return buy_iceberg_vol, sell_iceberg_vol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute tick microstructure features per 1-min bar"
    )
    parser.add_argument(
        "--tick-file", type=str, default=None,
        help="Path to tick parquet file (auto-detected if omitted)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: data/tick_microstructure_features.csv)"
    )
    args = parser.parse_args()

    output_path = args.output or str(DATA_DIR / "tick_microstructure_features.csv")

    total_t0 = time.time()

    # ---- Load ----
    df = load_ticks(args.tick_file)

    # ---- Build bar index ----
    print("Building bar index...")
    bars_raw, bar_idx, unique_bars, n_bars, ts_ns = build_bar_index(df)

    # Extract numpy arrays for speed
    prices = df["price"].values.astype(np.float64)
    sides = df["side"].values.astype(str)  # ensure string array
    sizes = df["size"].values  # already int64

    print(f"\nComputing features for {len(df):,} ticks across {n_bars:,} bars...")

    # ---- Feature 1: Price Impact Asymmetry ----
    buy_impact, sell_impact, impact_log_ratio, impact_ema3 = compute_price_impact(
        prices, sides, bars_raw, bar_idx, n_bars
    )

    # ---- Feature 2: Sweep Velocity ----
    buy_sweep_pts, sell_sweep_pts = compute_sweeps(
        prices, sides, bars_raw, bar_idx, ts_ns, n_bars
    )

    # ---- Feature 3: Footprint Imbalance ----
    top_ratio, bot_ratio = compute_footprint_imbalance(
        prices, sides, sizes, bar_idx, n_bars
    )

    # ---- Feature 4: Tick Arrival CV ----
    tick_cv = compute_tick_cv(bars_raw, bar_idx, ts_ns, n_bars)

    # ---- Feature 5: Iceberg Detection ----
    buy_iceberg_vol, sell_iceberg_vol = compute_icebergs(
        prices, sides, sizes, bars_raw, bar_idx, ts_ns, n_bars
    )

    # ---- Assemble output DataFrame ----
    print("\nAssembling output...")

    # Convert unique_bars (datetime64[ns, UTC]) to Unix seconds
    bar_timestamps_ns = unique_bars.astype(np.int64)
    bar_timestamps_s = bar_timestamps_ns // 1_000_000_000

    result = pd.DataFrame({
        "time": bar_timestamps_s,
        "buy_impact": buy_impact,
        "sell_impact": sell_impact,
        "impact_log_ratio": impact_log_ratio,
        "impact_ema3": impact_ema3,
        "buy_sweep_pts": buy_sweep_pts,
        "sell_sweep_pts": sell_sweep_pts,
        "top_ratio": top_ratio,
        "bot_ratio": bot_ratio,
        "tick_cv": tick_cv,
        "buy_iceberg_vol": buy_iceberg_vol,
        "sell_iceberg_vol": sell_iceberg_vol,
    })

    # ---- Save ----
    result.to_csv(output_path, index=False)
    total_elapsed = time.time() - total_t0

    print(f"\nSaved {len(result):,} bars to: {output_path}")
    print(f"Total computation time: {total_elapsed:.1f}s")
    print(f"\nColumn summary:")
    print(result.describe().to_string())


if __name__ == "__main__":
    main()
