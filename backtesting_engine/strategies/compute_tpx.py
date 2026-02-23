"""
RedK Trader Pressure Index (TPX) — Python port from Pine v5.
=============================================================
Original: RedKTrader (Mozilla Public License 2.0)
Pine source: strategies/RedK_TPX_v5.pine

Measures buying vs selling pressure using 2-bar range normalization.
- Bull pressure: how much highs and lows moved UP relative to the 2-bar range
- Bear pressure: how much highs and lows moved DOWN relative to the 2-bar range
- TPX = WMA(avgbulls - avgbears, smooth)  — net pressure, >0 bullish, <0 bearish

Usage:
    from compute_tpx import compute_tpx

    tpx, avgbulls, avgbears = compute_tpx(highs, lows, length=7, smooth=3)
    # tpx > 0 = bullish, tpx < 0 = bearish
    # avgbulls, avgbears are the individual pressure components

Validation:
    python3 compute_tpx.py   # runs self-test with assertions
"""

import numpy as np


def _wma(arr, period):
    """Weighted Moving Average matching Pine's ta.wma().

    Pine WMA: weight = period for most recent bar, period-1 for previous, etc.
    WMA[i] = sum(arr[i-k] * (period - k) for k in 0..period-1) / sum(1..period)

    For bars where we don't have enough history (i < period-1), we compute
    WMA over available bars to match Pine's warm-up behavior.
    """
    n = len(arr)
    result = np.zeros(n)
    denom_full = period * (period + 1) / 2  # sum of weights 1..period

    for i in range(n):
        # Available bars: min(i+1, period)
        avail = min(i + 1, period)
        # Weights: avail (most recent) down to 1
        numer = 0.0
        denom = 0.0
        for k in range(avail):
            w = avail - k  # most recent bar gets highest weight
            numer += arr[i - k] * w
            denom += w
        result[i] = numer / denom if denom > 0 else 0.0

    return result


def compute_tpx(highs, lows, length=7, smooth=3,
                pre_smooth=False, pre_smooth_len=3):
    """Compute RedK TPX indicator from high/low arrays.

    Args:
        highs: numpy array of high prices
        lows: numpy array of low prices
        length: WMA averaging period for bull/bear pressure (Pine default: 7)
        smooth: WMA smoothing period for final TPX line (Pine default: 3)
        pre_smooth: enable optional pre-smoothing (Pine default: False)
        pre_smooth_len: pre-smoothing WMA period (Pine default: 3)

    Returns:
        (tpx, avgbulls, avgbears) — all numpy arrays of same length as input.
        tpx: net pressure line (>0 bullish, <0 bearish)
        avgbulls: smoothed bull pressure (0-100 scale)
        avgbears: smoothed bear pressure (0-100 scale)
    """
    n = len(highs)
    assert len(lows) == n, "highs and lows must be same length"

    # R = 2-bar range: ta.highest(2) - ta.lowest(2)
    # ta.highest(2) = max(high[i], high[i-1]), ta.lowest(2) = min(low[i], low[i-1])
    R = np.zeros(n)
    R[0] = highs[0] - lows[0]  # first bar: only 1 bar available
    for i in range(1, n):
        hi2 = max(highs[i], highs[i - 1])
        lo2 = min(lows[i], lows[i - 1])
        R[i] = hi2 - lo2

    # Bull pressure: how much high and low moved UP
    # hiup = max(change(high), 0) = max(high[i] - high[i-1], 0)
    # loup = max(change(low), 0)  = max(low[i] - low[i-1], 0)
    # bulls = min((hiup + loup) / R, 1) * 100
    bulls = np.zeros(n)
    for i in range(1, n):
        hiup = max(highs[i] - highs[i - 1], 0.0)
        loup = max(lows[i] - lows[i - 1], 0.0)
        if R[i] > 0:
            bulls[i] = min((hiup + loup) / R[i], 1.0) * 100.0
        else:
            bulls[i] = 0.0  # nz() in Pine: NaN -> 0

    # Bear pressure: how much high and low moved DOWN
    # hidn = min(change(high), 0)
    # lodn = min(change(low), 0)
    # bears = max((hidn + lodn) / R, -1) * -100  (converted to positive)
    bears = np.zeros(n)
    for i in range(1, n):
        hidn = min(highs[i] - highs[i - 1], 0.0)
        lodn = min(lows[i] - lows[i - 1], 0.0)
        if R[i] > 0:
            bears[i] = max((hidn + lodn) / R[i], -1.0) * -100.0
        else:
            bears[i] = 0.0

    # Average with WMA
    avgbull = _wma(bulls, length)
    avgbear = _wma(bears, length)

    # Optional pre-smoothing
    if pre_smooth:
        avgbulls = _wma(avgbull, pre_smooth_len)
        avgbears = _wma(avgbear, pre_smooth_len)
    else:
        avgbulls = avgbull
        avgbears = avgbear

    # Net pressure + final smoothing
    net = avgbulls - avgbears
    tpx = _wma(net, smooth)

    return tpx, avgbulls, avgbears


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_wma():
    """Verify WMA matches Pine's behavior on known values."""
    # Pine WMA(3) of [1, 2, 3, 4, 5]:
    #   i=0: WMA = 1*1/1 = 1.0
    #   i=1: WMA = (2*2 + 1*1) / (2+1) = 5/3 = 1.667
    #   i=2: WMA = (3*3 + 2*2 + 1*1) / (3+2+1) = 14/6 = 2.333
    #   i=3: WMA = (4*3 + 3*2 + 2*1) / 6 = 20/6 = 3.333
    #   i=4: WMA = (5*3 + 4*2 + 3*1) / 6 = 26/6 = 4.333
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _wma(arr, 3)
    expected = [1.0, 5/3, 14/6, 20/6, 26/6]
    for i in range(5):
        assert abs(result[i] - expected[i]) < 1e-10, \
            f"WMA mismatch at i={i}: got {result[i]}, expected {expected[i]}"
    print("  WMA: PASS")


def _test_tpx_basic():
    """Test TPX on a simple up-trend scenario."""
    # 5 bars of steady uptrend: each bar moves up by 1 point
    n = 20
    highs = np.array([100.0 + i for i in range(n)])
    lows = np.array([99.0 + i for i in range(n)])

    tpx, avgbulls, avgbears = compute_tpx(highs, lows, length=3, smooth=2)

    # In a steady uptrend:
    # - hiup = 1.0 every bar, loup = 1.0 every bar
    # - R = max(h[i],h[i-1]) - min(l[i],l[i-1]) = (100+i) - (99+i-1) = 2.0
    # - bulls = min((1+1)/2, 1) * 100 = 100.0
    # - bears = max((0+0)/2, -1) * -100 = 0.0
    # So after warmup, avgbulls should approach 100, avgbears ~0, TPX ~100

    # After warmup (past bar ~5), TPX should be strongly positive
    assert tpx[-1] > 80, f"Expected TPX > 80 in uptrend, got {tpx[-1]:.1f}"
    assert avgbulls[-1] > 90, f"Expected avgbulls > 90, got {avgbulls[-1]:.1f}"
    assert avgbears[-1] < 5, f"Expected avgbears < 5, got {avgbears[-1]:.1f}"
    print(f"  Uptrend: TPX={tpx[-1]:.1f}, bulls={avgbulls[-1]:.1f}, bears={avgbears[-1]:.1f} — PASS")


def _test_tpx_downtrend():
    """Test TPX on a simple down-trend scenario."""
    n = 20
    highs = np.array([120.0 - i for i in range(n)])
    lows = np.array([119.0 - i for i in range(n)])

    tpx, avgbulls, avgbears = compute_tpx(highs, lows, length=3, smooth=2)

    assert tpx[-1] < -80, f"Expected TPX < -80 in downtrend, got {tpx[-1]:.1f}"
    assert avgbears[-1] > 90, f"Expected avgbears > 90, got {avgbears[-1]:.1f}"
    assert avgbulls[-1] < 5, f"Expected avgbulls < 5, got {avgbulls[-1]:.1f}"
    print(f"  Downtrend: TPX={tpx[-1]:.1f}, bulls={avgbulls[-1]:.1f}, bears={avgbears[-1]:.1f} — PASS")


def _test_tpx_flat():
    """Test TPX on flat/choppy market."""
    n = 20
    # Alternating up/down bars
    highs = np.array([100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(n)])
    lows = highs - 1.0

    tpx, avgbulls, avgbears = compute_tpx(highs, lows, length=7, smooth=3)

    # In choppy market, TPX should be near zero
    assert abs(tpx[-1]) < 30, f"Expected |TPX| < 30 in chop, got {tpx[-1]:.1f}"
    print(f"  Chop: TPX={tpx[-1]:.1f}, bulls={avgbulls[-1]:.1f}, bears={avgbears[-1]:.1f} — PASS")


def _test_tpx_zero_range():
    """Ensure R=0 (identical bars) doesn't cause division by zero."""
    n = 10
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)

    tpx, avgbulls, avgbears = compute_tpx(highs, lows)

    assert not np.any(np.isnan(tpx)), "NaN found in TPX with zero range"
    assert not np.any(np.isinf(tpx)), "Inf found in TPX with zero range"
    print(f"  Zero-range: no NaN/Inf — PASS")


def _test_pre_smoothing():
    """Verify pre-smoothing produces different (more lagged) output."""
    n = 30
    highs = np.array([100.0 + i * 0.5 for i in range(n)])
    lows = highs - 1.0

    tpx_no_pre, _, _ = compute_tpx(highs, lows, length=7, smooth=3,
                                     pre_smooth=False)
    tpx_pre, _, _ = compute_tpx(highs, lows, length=7, smooth=3,
                                 pre_smooth=True, pre_smooth_len=3)

    # Pre-smoothing should produce different values (more lag)
    diff = np.abs(tpx_no_pre - tpx_pre)
    assert np.max(diff) > 0.01, "Pre-smoothing had no effect"
    print(f"  Pre-smoothing: max diff={np.max(diff):.2f} — PASS")


def _test_on_real_data():
    """Run TPX on actual MNQ data and print sample values for visual inspection."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))

    try:
        from generate_session import load_instrument_1min
    except ImportError:
        print("  Real data test: SKIP (generate_session not importable)")
        return

    df = load_instrument_1min("MNQ")
    if len(df) == 0:
        print("  Real data test: SKIP (no MNQ data)")
        return

    highs = df["High"].values
    lows = df["Low"].values

    tpx, avgbulls, avgbears = compute_tpx(highs, lows, length=7, smooth=3)

    # Sanity checks on real data
    assert not np.any(np.isnan(tpx)), "NaN in TPX on real data"
    assert not np.any(np.isinf(tpx)), "Inf in TPX on real data"

    # TPX should be bounded roughly -100 to +100
    assert np.min(tpx) > -120, f"TPX min {np.min(tpx):.1f} below -120"
    assert np.max(tpx) < 120, f"TPX max {np.max(tpx):.1f} above 120"

    # Bull/bear pressure should be 0-100
    assert np.min(avgbulls) >= -1, f"avgbulls min {np.min(avgbulls):.1f} < 0"
    assert np.max(avgbulls) <= 101, f"avgbulls max {np.max(avgbulls):.1f} > 100"
    assert np.min(avgbears) >= -1, f"avgbears min {np.min(avgbears):.1f} < 0"
    assert np.max(avgbears) <= 101, f"avgbears max {np.max(avgbears):.1f} > 100"

    # Distribution check: TPX should be roughly centered, not stuck at one extreme
    pct_positive = np.sum(tpx > 0) / len(tpx) * 100
    assert 20 < pct_positive < 80, f"TPX {pct_positive:.0f}% positive — suspiciously skewed"

    print(f"  Real MNQ data ({len(df)} bars):")
    print(f"    TPX range: [{np.min(tpx):.1f}, {np.max(tpx):.1f}], "
          f"mean={np.mean(tpx):.1f}, {pct_positive:.0f}% bullish")
    print(f"    avgbulls range: [{np.min(avgbulls):.1f}, {np.max(avgbulls):.1f}]")
    print(f"    avgbears range: [{np.min(avgbears):.1f}, {np.max(avgbears):.1f}]")

    # Print 10 sample values to eyeball
    sample_idx = np.linspace(1000, len(tpx) - 1, 10, dtype=int)
    print(f"\n    {'Bar':>8s} {'High':>10s} {'Low':>10s} {'Bulls':>7s} {'Bears':>7s} {'TPX':>7s}")
    for idx in sample_idx:
        print(f"    {idx:>8d} {highs[idx]:>10.2f} {lows[idx]:>10.2f} "
              f"{avgbulls[idx]:>7.1f} {avgbears[idx]:>7.1f} {tpx[idx]:>7.1f}")

    print("  Real data test: PASS")


if __name__ == "__main__":
    print("RedK TPX Python port — self-test")
    print("=" * 50)
    _test_wma()
    _test_tpx_basic()
    _test_tpx_downtrend()
    _test_tpx_flat()
    _test_tpx_zero_range()
    _test_pre_smoothing()
    _test_on_real_data()
    print("=" * 50)
    print("All tests passed.")
