"""
v13 Tick-Data Feature Test
============================
Tests three novel tick-derived features on the v11 MNQ baseline.

Tick data: 42.6M NQ ticks aggregated to 1-min bars.
Features:
  I1/I2 - Size-Filtered Institutional Delta (entry filter, rolling 15/30 bars)
  J1/J2 - Absorption Detection (tighten stop to 25/35 pts when absorbed)
  K1/K2 - Participation Structure (entry filter, avg size > trailing 50/100 avg)

Train/Test split:
  TRAIN: Aug 17 - Nov 16, 2025 (~3 months)
  TEST:  Nov 17, 2025 - Feb 13, 2026 (~3 months)

Usage:
  python3 v13_tick_features.py                    # TRAIN only
  python3 v13_tick_features.py --oos              # TEST only
  python3 v13_tick_features.py --variant I1       # Single variant
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_instrument_1min, prepare_backtest_arrays_1min,
    run_v9_baseline, score_trades, fmt_score,
    compute_et_minutes, compute_rsi,
    bootstrap_ci, permutation_test,
    NY_OPEN_ET, NY_CLOSE_ET, NY_LAST_ENTRY_ET,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Train/test split
TRAIN_START = pd.Timestamp("2025-08-17", tz='UTC')
TRAIN_END = pd.Timestamp("2025-11-17", tz='UTC')
TEST_START = pd.Timestamp("2025-11-17", tz='UTC')
TEST_END = pd.Timestamp("2026-02-14", tz='UTC')

# v11 MNQ production params
V11_RSI_LEN = 8
V11_RSI_BUY = 60
V11_RSI_SELL = 40
V11_COOLDOWN = 20
V11_MAX_LOSS = 50
V11_SM_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Step 1: Tick Data Loading and Aggregation
# ---------------------------------------------------------------------------

def load_and_aggregate_ticks(tick_path):
    """Load tick parquet and aggregate to 1-min bars.

    Columns produced per 1-min bar:
      buy_vol, sell_vol, delta, total_vol
      large_buy_vol, large_sell_vol, large_delta
      small_buy_vol, small_sell_vol, small_delta
      tick_count, large_tick_count, avg_trade_size

    Args:
        tick_path: Path to parquet file with ts_event, price, size, side

    Returns:
        DataFrame indexed by 1-min UTC timestamp with aggregated columns.
    """
    print(f"  Loading tick data from {tick_path}...")
    t0 = time.time()
    df = pd.read_parquet(tick_path, columns=['ts_event', 'price', 'size', 'side'],
                         engine='pyarrow')
    print(f"  Loaded {len(df):,} ticks in {time.time() - t0:.1f}s")

    # CRITICAL: cast size to int64 before any subtraction to avoid uint32 overflow
    df['size'] = df['size'].astype(np.int64)

    # Parse timestamps
    print(f"  Parsing timestamps...")
    t0 = time.time()
    df['ts'] = pd.to_datetime(df['ts_event'], utc=True)
    df['bar'] = df['ts'].dt.floor('1min')
    print(f"  Timestamps parsed in {time.time() - t0:.1f}s")

    # Classify trades
    is_buy = df['side'] == 'B'
    is_sell = df['side'] == 'A'
    is_large = df['size'] >= 5
    is_small = df['size'] < 5

    # Build aggregation columns
    print(f"  Computing per-tick fields...")
    t0 = time.time()
    df['buy_vol_t'] = np.where(is_buy, df['size'], 0)
    df['sell_vol_t'] = np.where(is_sell, df['size'], 0)
    df['large_buy_vol_t'] = np.where(is_buy & is_large, df['size'], 0)
    df['large_sell_vol_t'] = np.where(is_sell & is_large, df['size'], 0)
    df['small_buy_vol_t'] = np.where(is_buy & is_small, df['size'], 0)
    df['small_sell_vol_t'] = np.where(is_sell & is_small, df['size'], 0)
    df['is_large'] = is_large.astype(np.int64)
    print(f"  Per-tick fields computed in {time.time() - t0:.1f}s")

    # Aggregate to 1-min bars
    print(f"  Aggregating to 1-min bars (this takes ~30-60s)...")
    t0 = time.time()
    agg = df.groupby('bar').agg(
        buy_vol=('buy_vol_t', 'sum'),
        sell_vol=('sell_vol_t', 'sum'),
        total_vol=('size', 'sum'),
        large_buy_vol=('large_buy_vol_t', 'sum'),
        large_sell_vol=('large_sell_vol_t', 'sum'),
        small_buy_vol=('small_buy_vol_t', 'sum'),
        small_sell_vol=('small_sell_vol_t', 'sum'),
        tick_count=('size', 'count'),
        large_tick_count=('is_large', 'sum'),
    )
    print(f"  Aggregation complete in {time.time() - t0:.1f}s")

    # Derived columns
    agg['delta'] = agg['buy_vol'] - agg['sell_vol']
    agg['large_delta'] = agg['large_buy_vol'] - agg['large_sell_vol']
    agg['small_delta'] = agg['small_buy_vol'] - agg['small_sell_vol']
    agg['avg_trade_size'] = np.where(
        agg['tick_count'] > 0,
        agg['total_vol'] / agg['tick_count'],
        0.0
    )

    print(f"  Result: {len(agg):,} 1-min bars with tick features")
    print(f"  Date range: {agg.index[0]} to {agg.index[-1]}")
    return agg


# ---------------------------------------------------------------------------
# Step 2: Align tick features to OHLCV index
# ---------------------------------------------------------------------------

def align_tick_features(ohlcv_1m, tick_agg):
    """Align tick-aggregated data to OHLCV timestamps.

    Uses reindex with ffill to handle any timestamp mismatches between
    MNQ OHLCV and NQ tick data.

    Returns dict of numpy arrays aligned to ohlcv_1m.index.
    """
    cols_to_align = [
        'buy_vol', 'sell_vol', 'delta', 'total_vol',
        'large_buy_vol', 'large_sell_vol', 'large_delta',
        'small_buy_vol', 'small_sell_vol', 'small_delta',
        'tick_count', 'large_tick_count', 'avg_trade_size',
    ]

    aligned = {}
    for col in cols_to_align:
        aligned[col] = tick_agg[col].reindex(ohlcv_1m.index, method='ffill').fillna(0).values

    # Stats
    nonzero_delta = np.count_nonzero(aligned['delta'])
    nonzero_large = np.count_nonzero(aligned['large_delta'])
    print(f"  Aligned {len(aligned['delta']):,} bars to OHLCV index")
    print(f"  Non-zero delta: {nonzero_delta:,}, Non-zero large_delta: {nonzero_large:,}")

    return aligned


# ---------------------------------------------------------------------------
# Step 3: Feature Computation
# ---------------------------------------------------------------------------

def compute_institutional_delta_filter(large_delta, window):
    """Rolling sum of large_delta over `window` bars. Returns sign array."""
    n = len(large_delta)
    roll_sum = np.zeros(n)
    for i in range(n):
        s = max(0, i - window + 1)
        roll_sum[i] = np.sum(large_delta[s:i + 1])
    return np.sign(roll_sum)


def compute_participation_filter(avg_trade_size, window):
    """Rolling mean of avg_trade_size. Returns (values, rolling_mean) arrays."""
    n = len(avg_trade_size)
    rolling_mean = np.zeros(n)
    for i in range(n):
        s = max(0, i - window + 1)
        rolling_mean[i] = np.mean(avg_trade_size[s:i + 1])
    return avg_trade_size, rolling_mean


# ---------------------------------------------------------------------------
# Step 4: Custom Engine (handles ALL variants)
# ---------------------------------------------------------------------------

def _run_custom_engine(arr, entry_filter_arr=None,
                       participation_filter=None, participation_threshold=None,
                       absorption_delta=None, tight_stop=0,
                       absorption_window=5, absorption_count=3):
    """Custom v11 engine supporting all v13 tick-feature variants.

    Args:
        arr: standard backtest arrays from prepare_backtest_arrays_1min
        entry_filter_arr: directional filter (>0 for longs, <0 for shorts)
        participation_filter: avg_trade_size array for non-directional filter
        participation_threshold: rolling mean array; entry requires filter > threshold
        absorption_delta: delta array for absorption detection
        tight_stop: tightened stop in pts when absorption detected (0 = disabled)
        absorption_window: how many bars to look back for absorption
        absorption_count: how many bars must show absorption to trigger

    Returns:
        list of trade dicts
    """
    n = len(arr['opens'])
    opens = arr['opens']
    closes = arr['closes']
    sm = arr['sm']
    times = arr['times']
    rsi_5m_curr = arr.get('rsi_5m_curr')
    rsi_5m_prev = arr.get('rsi_5m_prev')

    et_mins = compute_et_minutes(times)
    rsi = arr.get('rsi', compute_rsi(closes, V11_RSI_LEN))

    trades = []
    trade_state = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    def close_trade(side, ep, xp, ei, xi, result):
        pts = (xp - ep) if side == "long" else (ep - xp)
        trades.append({"side": side, "entry": ep, "exit": xp,
                        "pts": pts, "entry_time": times[ei],
                        "exit_time": times[xi], "entry_idx": ei,
                        "exit_idx": xi, "bars": xi - ei, "result": result})

    for i in range(2, n):
        bar_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]
        sm_bull = sm_prev > V11_SM_THRESHOLD
        sm_bear = sm_prev < -V11_SM_THRESHOLD
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI cross (mapped 5-min)
        if rsi_5m_curr is not None and rsi_5m_prev is not None:
            rsi_prev = rsi_5m_curr[i - 1]
            rsi_prev2 = rsi_5m_prev[i - 1]
            rsi_long_trigger = rsi_prev > V11_RSI_BUY and rsi_prev2 <= V11_RSI_BUY
            rsi_short_trigger = rsi_prev < V11_RSI_SELL and rsi_prev2 >= V11_RSI_SELL
        else:
            rsi_prev = rsi[i - 1]
            rsi_prev2 = rsi[i - 2]
            rsi_long_trigger = rsi_prev > V11_RSI_BUY and rsi_prev2 <= V11_RSI_BUY
            rsi_short_trigger = rsi_prev < V11_RSI_SELL and rsi_prev2 >= V11_RSI_SELL

        # Episode reset
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # EOD close
        if trade_state != 0 and bar_et >= NY_CLOSE_ET:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # --- Exits ---
        if trade_state == 1:
            # Determine effective stop
            effective_stop = V11_MAX_LOSS
            if tight_stop > 0 and absorption_delta is not None and i >= absorption_window:
                # Check absorption: how many of last absorption_window bars have
                # price up (close >= open) AND delta negative?
                absorb_count = 0
                for j in range(i - absorption_window, i):
                    if closes[j] >= opens[j] and absorption_delta[j] < 0:
                        absorb_count += 1
                if absorb_count >= absorption_count:
                    effective_stop = tight_stop

            # Max loss stop (bar i-1 data, fill at bar i open)
            if effective_stop > 0 and closes[i - 1] <= entry_price - effective_stop:
                result = "SL_TIGHT" if effective_stop < V11_MAX_LOSS else "SL"
                close_trade("long", entry_price, opens[i], entry_idx, i, result)
                trade_state = 0
                exit_bar = i
                continue

            # SM flip exit
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        elif trade_state == -1:
            effective_stop = V11_MAX_LOSS
            if tight_stop > 0 and absorption_delta is not None and i >= absorption_window:
                absorb_count = 0
                for j in range(i - absorption_window, i):
                    if closes[j] <= opens[j] and absorption_delta[j] > 0:
                        absorb_count += 1
                if absorb_count >= absorption_count:
                    effective_stop = tight_stop

            if effective_stop > 0 and closes[i - 1] >= entry_price + effective_stop:
                result = "SL_TIGHT" if effective_stop < V11_MAX_LOSS else "SL"
                close_trade("short", entry_price, opens[i], entry_idx, i, result)
                trade_state = 0
                exit_bar = i
                continue

            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        # --- Entries ---
        if trade_state == 0:
            in_session = NY_OPEN_ET <= bar_et <= NY_LAST_ENTRY_ET
            cd_ok = (i - exit_bar) >= V11_COOLDOWN

            if in_session and cd_ok:
                # Long entry
                if sm_bull and rsi_long_trigger and not long_used:
                    # Entry filters
                    filter_ok = True
                    if entry_filter_arr is not None:
                        filter_ok = entry_filter_arr[i - 1] > 0
                    if filter_ok and participation_filter is not None:
                        filter_ok = participation_filter[i - 1] > participation_threshold[i - 1]

                    if filter_ok:
                        trade_state = 1
                        entry_price = opens[i]
                        entry_idx = i
                        long_used = True

                # Short entry
                elif sm_bear and rsi_short_trigger and not short_used:
                    filter_ok = True
                    if entry_filter_arr is not None:
                        filter_ok = entry_filter_arr[i - 1] < 0
                    if filter_ok and participation_filter is not None:
                        filter_ok = participation_filter[i - 1] > participation_threshold[i - 1]

                    if filter_ok:
                        trade_state = -1
                        entry_price = opens[i]
                        entry_idx = i
                        short_used = True

    return trades


# ---------------------------------------------------------------------------
# Variant Runners
# ---------------------------------------------------------------------------

def run_variant_i1(arr, tick_aligned, **kw):
    """I1: Large delta rolling 15-bar sum, entry filter."""
    filt = compute_institutional_delta_filter(tick_aligned['large_delta'], window=15)
    return _run_custom_engine(arr, entry_filter_arr=filt)


def run_variant_i2(arr, tick_aligned, **kw):
    """I2: Large delta rolling 30-bar sum, entry filter."""
    filt = compute_institutional_delta_filter(tick_aligned['large_delta'], window=30)
    return _run_custom_engine(arr, entry_filter_arr=filt)


def run_variant_j1(arr, tick_aligned, **kw):
    """J1: Absorption detection, tighten stop to 25pts."""
    return _run_custom_engine(arr,
                              absorption_delta=tick_aligned['delta'],
                              tight_stop=25,
                              absorption_window=5,
                              absorption_count=3)


def run_variant_j2(arr, tick_aligned, **kw):
    """J2: Absorption detection, tighten stop to 35pts."""
    return _run_custom_engine(arr,
                              absorption_delta=tick_aligned['delta'],
                              tight_stop=35,
                              absorption_window=5,
                              absorption_count=3)


def run_variant_k1(arr, tick_aligned, **kw):
    """K1: Participation filter, avg size > 50-bar trailing average."""
    vals, rolling_mean = compute_participation_filter(
        tick_aligned['avg_trade_size'], window=50)
    return _run_custom_engine(arr,
                              participation_filter=vals,
                              participation_threshold=rolling_mean)


def run_variant_k2(arr, tick_aligned, **kw):
    """K2: Participation filter, avg size > 100-bar trailing average."""
    vals, rolling_mean = compute_participation_filter(
        tick_aligned['avg_trade_size'], window=100)
    return _run_custom_engine(arr,
                              participation_filter=vals,
                              participation_threshold=rolling_mean)


# ---------------------------------------------------------------------------
# Evaluation (7-criteria framework)
# ---------------------------------------------------------------------------

VARIANTS = {
    'I1': ('Inst delta 15-bar entry filter', run_variant_i1),
    'I2': ('Inst delta 30-bar entry filter', run_variant_i2),
    'J1': ('Absorption stop tighten 25pts',  run_variant_j1),
    'J2': ('Absorption stop tighten 35pts',  run_variant_j2),
    'K1': ('Participation 50-bar filter',    run_variant_k1),
    'K2': ('Participation 100-bar filter',   run_variant_k2),
}

ALL_VARIANT_KEYS = ['I1', 'I2', 'J1', 'J2', 'K1', 'K2']


def evaluate_variant(name, desc, trades_baseline, trades_feature,
                     commission=0.52, dollar_per_pt=2.0):
    """Evaluate a feature variant using the 7-criteria framework."""
    sc_base = score_trades(trades_baseline, commission, dollar_per_pt)
    sc_feat = score_trades(trades_feature, commission, dollar_per_pt)

    if sc_base is None:
        print(f"  {name} ({desc}): SKIP (no baseline trades)")
        return None
    if sc_feat is None:
        print(f"  {name} ({desc}): SKIP (no feature trades)")
        return None

    print(f"\n  --- Variant {name}: {desc} ---")
    print(f"  Baseline: {fmt_score(sc_base, 'BASE')}")
    print(f"  Feature:  {fmt_score(sc_feat, name)}")

    # Show exit types
    if sc_feat.get('exits'):
        exit_parts = [f"{k}:{v}" for k, v in sorted(sc_feat['exits'].items())]
        print(f"  Exits:    {' '.join(exit_parts)}")

    # For J variants, show SL_TIGHT breakdown specifically
    if name.startswith('J') and sc_feat.get('exits'):
        exits = sc_feat['exits']
        sl_std = exits.get('SL', 0)
        sl_tight = exits.get('SL_TIGHT', 0)
        sm_flip = exits.get('SM_FLIP', 0)
        eod = exits.get('EOD', 0)
        total = sc_feat['count']
        print(f"  Stop breakdown: SL={sl_std} ({100*sl_std/total:.1f}%), "
              f"SL_TIGHT={sl_tight} ({100*sl_tight/total:.1f}%), "
              f"SM_FLIP={sm_flip} ({100*sm_flip/total:.1f}%), "
              f"EOD={eod} ({100*eod/total:.1f}%)")

    # 1. PF improvement
    dpf = sc_feat['pf'] - sc_base['pf']
    pf_pass = dpf > 0.1

    # 2. Bootstrap CI
    ci_point, ci_lo, ci_hi = bootstrap_ci(trades_feature, metric='pf',
                                           commission_per_side=commission,
                                           dollar_per_pt=dollar_per_pt)
    ci_excludes_base = ci_lo > sc_base['pf']

    # 3. Permutation test
    obs_diff, p_val = permutation_test(trades_baseline, trades_feature,
                                        commission_per_side=commission,
                                        dollar_per_pt=dollar_per_pt)

    # 4. Trade count preservation
    count_ratio = sc_feat['count'] / sc_base['count'] if sc_base['count'] > 0 else 0
    count_ok = count_ratio > 0.5

    # 5. Win rate
    dwr = sc_feat['win_rate'] - sc_base['win_rate']
    wr_ok = dwr > -5

    # 6. Drawdown
    base_dd = sc_base['max_dd_dollar']
    feat_dd = sc_feat['max_dd_dollar']
    dd_ok = feat_dd >= base_dd * 1.2  # not >20% worse (dd is negative)

    # 7. Lucky trade removal
    if len(trades_feature) > 1:
        pts = np.array([t['pts'] for t in trades_feature])
        best_idx = np.argmax(pts)
        trades_no_best = [t for j, t in enumerate(trades_feature) if j != best_idx]
        sc_no_best = score_trades(trades_no_best, commission, dollar_per_pt)
        lucky_ok = sc_no_best is not None and sc_no_best['pf'] > 1.0
    else:
        lucky_ok = False
        sc_no_best = None

    criteria = [
        ("PF improvement > 0.1", pf_pass, f"dPF={dpf:+.3f}"),
        ("Bootstrap CI excludes baseline", ci_excludes_base,
         f"CI=[{ci_lo:.3f}, {ci_hi:.3f}], base={sc_base['pf']:.3f}"),
        ("Permutation p < 0.05", p_val < 0.05, f"p={p_val:.4f}"),
        ("Trade count > 50% of baseline", count_ok,
         f"ratio={count_ratio:.2f} ({sc_feat['count']}/{sc_base['count']})"),
        ("Win rate not degraded > 5%", wr_ok, f"dWR={dwr:+.1f}%"),
        ("Drawdown not >20% worse", dd_ok,
         f"feat DD=${feat_dd:.0f}, base DD=${base_dd:.0f}"),
        ("Survives lucky trade removal", lucky_ok,
         f"PF w/o best={sc_no_best['pf']:.3f}" if sc_no_best else "N/A"),
    ]

    passes = 0
    print(f"\n  Criteria:")
    for crit_name, passed, detail in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {crit_name}: {detail}")
        if passed:
            passes += 1

    verdict = "ADOPT" if passes >= 5 else ("MAYBE" if passes >= 3 else "REJECT")
    print(f"\n  Score: {passes}/7 -> {verdict}")

    return {
        'name': name, 'desc': desc, 'verdict': verdict,
        'passes': passes, 'sc_base': sc_base, 'sc_feat': sc_feat,
        'dpf': dpf, 'p_val': p_val,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_tick_file():
    """Auto-detect tick parquet file in DATA_DIR."""
    candidates = sorted(DATA_DIR.glob("databento_NQ_ticks_2025-08-17_*.parquet"))
    if candidates:
        return candidates[-1]  # most recent
    # Fall back to any tick parquet
    candidates = sorted(DATA_DIR.glob("*NQ*tick*.parquet"))
    if candidates:
        return candidates[-1]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="v13 tick-data feature test on v11 MNQ baseline",
    )
    parser.add_argument("--tick-file", default=None,
                        help="Path to tick parquet (default: auto-detect in data/)")
    parser.add_argument("--instrument", default="MNQ",
                        help="Instrument (default: MNQ)")
    parser.add_argument("--variant", default=None,
                        help="Test specific variant (I1,I2,J1,J2,K1,K2)")
    parser.add_argument("--oos", action="store_true",
                        help="Run on out-of-sample (test) period")

    args = parser.parse_args()

    # Resolve tick file
    if args.tick_file:
        tick_path = Path(args.tick_file)
    else:
        tick_path = find_tick_file()
    if tick_path is None or not tick_path.exists():
        print(f"ERROR: Tick file not found. Searched in {DATA_DIR}")
        print(f"  Expected: databento_NQ_ticks_2025-08-17_to_2026-02-13.parquet")
        sys.exit(1)

    if args.oos:
        period_start, period_end = TEST_START, TEST_END
        period_name = "TEST (OOS)"
    else:
        period_start, period_end = TRAIN_START, TRAIN_END
        period_name = "TRAIN"

    print("=" * 70)
    print(f"v13 TICK-DATA FEATURE TEST -- {period_name}")
    print("=" * 70)
    print(f"  Period:     {period_start.date()} to {period_end.date()}")
    print(f"  Instrument: {args.instrument}")
    print(f"  Tick file:  {tick_path.name}")
    print(f"  Commission: $0.52/side (0.005%)")
    print()

    # -----------------------------------------------------------------------
    # 1. Load and aggregate tick data (expensive -- do once)
    # -----------------------------------------------------------------------
    print("Step 1: Load and aggregate tick data")
    t_total = time.time()
    tick_agg = load_and_aggregate_ticks(tick_path)
    print(f"  Total tick processing time: {time.time() - t_total:.1f}s\n")

    # -----------------------------------------------------------------------
    # 2. Load OHLCV and filter to period
    # -----------------------------------------------------------------------
    print("Step 2: Load OHLCV data")
    ohlcv_1m = load_instrument_1min(args.instrument)
    if ohlcv_1m.index.tz is None:
        ohlcv_1m.index = ohlcv_1m.index.tz_localize('UTC')

    period_data = ohlcv_1m[(ohlcv_1m.index >= period_start) &
                            (ohlcv_1m.index < period_end)]
    if len(period_data) < 100:
        print(f"ERROR: Only {len(period_data)} bars in period. Need more data.")
        sys.exit(1)
    print(f"  OHLCV bars: {len(period_data):,}")
    print(f"  Date range: {period_data.index[0]} to {period_data.index[-1]}\n")

    # -----------------------------------------------------------------------
    # 3. Align tick features to OHLCV
    # -----------------------------------------------------------------------
    print("Step 3: Align tick features to OHLCV index")
    tick_aligned = align_tick_features(period_data, tick_agg)

    # Print tick feature stats
    ld = tick_aligned['large_delta']
    ld_nz = ld[ld != 0]
    if len(ld_nz) > 0:
        print(f"  Large delta stats: mean={ld_nz.mean():.1f}, "
              f"std={ld_nz.std():.1f}, "
              f"min={ld_nz.min():.0f}, max={ld_nz.max():.0f}")
    ats = tick_aligned['avg_trade_size']
    ats_nz = ats[ats > 0]
    if len(ats_nz) > 0:
        print(f"  Avg trade size stats: mean={ats_nz.mean():.2f}, "
              f"std={ats_nz.std():.2f}, "
              f"min={ats_nz.min():.2f}, max={ats_nz.max():.2f}")
    d = tick_aligned['delta']
    d_nz = d[d != 0]
    if len(d_nz) > 0:
        print(f"  Total delta stats: mean={d_nz.mean():.1f}, "
              f"std={d_nz.std():.1f}")
    print()

    # -----------------------------------------------------------------------
    # 4. Prepare backtest arrays
    # -----------------------------------------------------------------------
    print("Step 4: Prepare backtest arrays (1-min with mapped 5-min RSI)")
    arr = prepare_backtest_arrays_1min(period_data, rsi_len=V11_RSI_LEN)
    print(f"  Arrays ready: {len(arr['opens']):,} bars\n")

    # -----------------------------------------------------------------------
    # 5. Run baseline
    # -----------------------------------------------------------------------
    print("Step 5: Run v11 baseline")
    trades_baseline = run_v9_baseline(
        arr, rsi_len=V11_RSI_LEN, rsi_buy=V11_RSI_BUY,
        rsi_sell=V11_RSI_SELL, cooldown=V11_COOLDOWN,
        max_loss_pts=V11_MAX_LOSS,
    )
    sc_baseline = score_trades(trades_baseline)
    print(f"  {fmt_score(sc_baseline, 'v11 BASELINE')}")
    if sc_baseline and sc_baseline.get('exits'):
        exit_parts = [f"{k}:{v}" for k, v in sorted(sc_baseline['exits'].items())]
        print(f"  Exits: {' '.join(exit_parts)}")
    print()

    # -----------------------------------------------------------------------
    # 6. Run variants
    # -----------------------------------------------------------------------
    if args.variant:
        variants_to_test = [args.variant.upper()]
    else:
        variants_to_test = ALL_VARIANT_KEYS

    results = []

    for var_key in variants_to_test:
        if var_key not in VARIANTS:
            print(f"\n  WARNING: Unknown variant '{var_key}', skipping")
            continue

        desc, run_fn = VARIANTS[var_key]
        print(f"Running variant {var_key}: {desc}...")
        t0 = time.time()
        trades_feat = run_fn(arr, tick_aligned)
        elapsed = time.time() - t0
        print(f"  Engine time: {elapsed:.1f}s, Trades: {len(trades_feat)}")
        result = evaluate_variant(var_key, desc, trades_baseline, trades_feat)
        if result:
            results.append(result)

    # -----------------------------------------------------------------------
    # 7. Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY -- {period_name}")
    print("=" * 70)

    if not results:
        print("  No results to summarize.")
        return

    print(f"\n  {'Variant':<36} | {'Verdict':>8} | {'Score':>5} | "
          f"{'dPF':>8} | {'p-val':>8} | {'Trades':>7}")
    print(f"  {'-'*36}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    best = None
    for r in results:
        n_trades = r['sc_feat']['count']
        line = (f"  {r['name']+': '+r['desc']:<36} | {r['verdict']:>8} | "
                f"{r['passes']:>3}/7 | {r['dpf']:>+8.3f} | "
                f"{r['p_val']:>8.4f} | {n_trades:>7}")
        print(line)
        if best is None or r['passes'] > best['passes']:
            best = r
        elif r['passes'] == best['passes'] and r['dpf'] > best['dpf']:
            best = r

    # Baseline reference
    if sc_baseline:
        print(f"\n  Baseline: {sc_baseline['count']} trades, "
              f"PF {sc_baseline['pf']}, Net ${sc_baseline['net_dollar']:+.2f}, "
              f"MaxDD ${sc_baseline['max_dd_dollar']:.2f}")

    if best and best['verdict'] != 'REJECT':
        print(f"\n  BEST VARIANT: {best['name']} ({best['desc']})")
        print(f"    PF: {best['sc_feat']['pf']:.3f} "
              f"(baseline {best['sc_base']['pf']:.3f})")
        if not args.oos:
            print(f"\n  Next step: python3 v13_tick_features.py "
                  f"--variant {best['name']} --oos")
    else:
        print(f"\n  No variant passed on {period_name} data.")
        if not args.oos:
            print("  Consider running --oos anyway to check test period behavior.")

    # OOS validation check
    if args.oos and best:
        print(f"\n  OOS VALIDATION:")
        feat_pf = best['sc_feat']['pf']
        base_pf = best['sc_base']['pf']
        if feat_pf < 1.0:
            print(f"    PF = {feat_pf:.3f} < 1.0 -> REJECT (unprofitable OOS)")
        elif feat_pf < base_pf * 0.7:
            print(f"    PF degraded >30% from baseline -> REJECT")
        else:
            print(f"    PF = {feat_pf:.3f} (baseline {base_pf:.3f}) -> PASS")


if __name__ == "__main__":
    main()
