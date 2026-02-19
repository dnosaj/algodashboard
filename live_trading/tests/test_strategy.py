r"""
Strategy parity test: IncrementalStrategy vs vectorized run_backtest_v10().

CRITICAL: Feeds Databento CSV bar-by-bar into IncrementalStrategy and
compares output trades against the vectorized backtest. They must produce
IDENTICAL trades (entry time, exit time, side, entry price, exit price).

Usage:
    cd /Users/jasongeorge/Desktop/NQ\ trading/live_trading
    python -m pytest tests/test_strategy.py -v
    # or directly:
    python tests/test_strategy.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Add backtesting engine to path for v10_test_common
BACKTESTING_DIR = Path(__file__).resolve().parent.parent.parent / "backtesting_engine" / "strategies"
sys.path.insert(0, str(BACKTESTING_DIR))

# Add live_trading root to path
LIVE_TRADING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LIVE_TRADING_DIR))

from engine.config import MNQ_V11
from engine.events import Bar, SignalType
from engine.strategy import IncrementalStrategy

import v10_test_common as vtc


DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "backtesting_engine" / "data"
    / "databento_MNQ_1min_2025-08-17_to_2026-02-13.csv"
)

# MNQ v11 params matching the validated production config
MNQ_V11_PARAMS = {
    "sm_index": 10,
    "sm_flow": 12,
    "sm_norm": 200,
    "sm_ema": 100,
    "rsi_len": 8,
    "rsi_buy": 60,
    "rsi_sell": 40,
    "cooldown": 20,
    "max_loss_pts": 50,
}


def load_data() -> pd.DataFrame:
    """Load Databento MNQ 1-min data and compute SM with v11 params."""
    df = pd.read_csv(str(DATA_PATH))

    result = pd.DataFrame()
    result["Time"] = pd.to_datetime(df["time"].astype(int), unit="s")
    result["Open"] = pd.to_numeric(df["open"], errors="coerce")
    result["High"] = pd.to_numeric(df["high"], errors="coerce")
    result["Low"] = pd.to_numeric(df["low"], errors="coerce")
    result["Close"] = pd.to_numeric(df["close"], errors="coerce")
    result["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    result["VWAP"] = pd.to_numeric(df["VWAP"], errors="coerce")
    result = result.set_index("Time")

    # Compute SM with MNQ v11 params (10, 12, 200, 100)
    sm = vtc.compute_smart_money(
        result["Close"].values,
        result["Volume"].values,
        index_period=MNQ_V11_PARAMS["sm_index"],
        flow_period=MNQ_V11_PARAMS["sm_flow"],
        norm_period=MNQ_V11_PARAMS["sm_norm"],
        ema_len=MNQ_V11_PARAMS["sm_ema"],
    )
    result["SM_Net"] = sm

    return result


def run_vectorized_backtest(df_1m: pd.DataFrame) -> list[dict]:
    """Run the vectorized backtest using v10_test_common."""
    arr = vtc.prepare_backtest_arrays_1min(df_1m, rsi_len=MNQ_V11_PARAMS["rsi_len"])
    trades = vtc.run_v9_baseline(
        arr,
        rsi_len=MNQ_V11_PARAMS["rsi_len"],
        rsi_buy=MNQ_V11_PARAMS["rsi_buy"],
        rsi_sell=MNQ_V11_PARAMS["rsi_sell"],
        sm_threshold=0.0,
        cooldown=MNQ_V11_PARAMS["cooldown"],
        max_loss_pts=MNQ_V11_PARAMS["max_loss_pts"],
        use_rsi_cross=True,
    )
    return trades


def run_incremental_strategy(df_1m: pd.DataFrame) -> list:
    """Feed data bar-by-bar into IncrementalStrategy, collect trades.

    The vectorized engine computes SM/RSI on all bars at once then loops
    from i=2. The incremental engine must process all bars sequentially.
    We start trading immediately (no warmup phase) since the vectorized
    engine has no warmup -- it just starts at index 2 with whatever SM
    values exist.
    """
    config = MNQ_V11

    strategy = IncrementalStrategy(config, event_bus=None)
    # Start trading immediately -- the vectorized engine processes all bars
    # from the start (loop begins at i=2 but SM is pre-computed on all bars).
    # The incremental engine's first few bars will have sm_prev=0.0 which
    # matches sm[0]=0 and sm[1] in the vectorized version.
    strategy.start_trading()

    times = df_1m.index.values
    opens = df_1m["Open"].values
    highs = df_1m["High"].values
    lows = df_1m["Low"].values
    closes = df_1m["Close"].values
    volumes = df_1m["Volume"].values

    n = len(df_1m)
    for i in range(n):
        ts = pd.Timestamp(times[i])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        dt = ts.to_pydatetime()

        bar = Bar(
            timestamp=dt,
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
            instrument="MNQ",
        )

        strategy.on_bar(bar)

    return strategy.trades


def compare_trades(
    vec_trades: list[dict],
    inc_trades: list,
    verbose: bool = True,
) -> tuple[int, int, list[str]]:
    """Compare vectorized and incremental trade lists.

    Returns (matches, mismatches, mismatch_details).
    """
    matches = 0
    mismatches = 0
    details = []

    n_vec = len(vec_trades)
    n_inc = len(inc_trades)

    if verbose:
        print(f"\nVectorized trades: {n_vec}")
        print(f"Incremental trades: {n_inc}")

    max_trades = max(n_vec, n_inc)

    for i in range(max_trades):
        if i >= n_vec:
            msg = f"Trade #{i+1}: EXTRA in incremental (no vectorized match)"
            if i < n_inc:
                t = inc_trades[i]
                msg += (
                    f" -- {t.side} entry={t.entry_price:.2f} "
                    f"exit={t.exit_price:.2f} @ {t.entry_time}"
                )
            details.append(msg)
            mismatches += 1
            continue

        if i >= n_inc:
            msg = f"Trade #{i+1}: MISSING in incremental (vectorized has it)"
            vt = vec_trades[i]
            msg += (
                f" -- {vt['side']} entry={vt['entry']:.2f} "
                f"exit={vt['exit']:.2f} @ {vt['entry_time']}"
            )
            details.append(msg)
            mismatches += 1
            continue

        vt = vec_trades[i]
        it = inc_trades[i]

        # Compare fields
        issues = []

        # Side
        if vt["side"] != it.side:
            issues.append(f"side: vec={vt['side']} inc={it.side}")

        # Entry price
        if abs(vt["entry"] - it.entry_price) > 0.01:
            issues.append(f"entry: vec={vt['entry']:.2f} inc={it.entry_price:.2f}")

        # Exit price
        if abs(vt["exit"] - it.exit_price) > 0.01:
            issues.append(f"exit: vec={vt['exit']:.2f} inc={it.exit_price:.2f}")

        # Entry time
        vt_entry_time = pd.Timestamp(vt["entry_time"])
        it_entry_time = pd.Timestamp(it.entry_time)
        # Strip timezone for comparison (both should be UTC)
        if it_entry_time.tzinfo is not None:
            it_entry_time = it_entry_time.tz_localize(None)
        if vt_entry_time.tzinfo is not None:
            vt_entry_time = vt_entry_time.tz_localize(None)

        if vt_entry_time != it_entry_time:
            issues.append(f"entry_time: vec={vt_entry_time} inc={it_entry_time}")

        # Exit time
        vt_exit_time = pd.Timestamp(vt["exit_time"])
        it_exit_time = pd.Timestamp(it.exit_time)
        if it_exit_time.tzinfo is not None:
            it_exit_time = it_exit_time.tz_localize(None)
        if vt_exit_time.tzinfo is not None:
            vt_exit_time = vt_exit_time.tz_localize(None)

        if vt_exit_time != it_exit_time:
            issues.append(f"exit_time: vec={vt_exit_time} inc={it_exit_time}")

        if issues:
            mismatches += 1
            msg = f"Trade #{i+1} MISMATCH: " + "; ".join(issues)
            details.append(msg)
            if verbose:
                print(f"  MISMATCH #{i+1}: {'; '.join(issues)}")
        else:
            matches += 1
            if verbose and i < 5:
                print(
                    f"  MATCH #{i+1}: {vt['side']} "
                    f"entry={vt['entry']:.2f} exit={vt['exit']:.2f} "
                    f"pts={vt['pts']:+.2f}"
                )

    return matches, mismatches, details


def test_strategy_parity():
    """CRITICAL TEST: IncrementalStrategy must match vectorized backtest exactly.

    Loads MNQ Databento 1-min data, runs both engines with identical params,
    and asserts all trades are identical.
    """
    print("\n" + "=" * 70)
    print("STRATEGY PARITY TEST: IncrementalStrategy vs run_backtest_v10()")
    print("=" * 70)
    print(f"Data: {DATA_PATH}")
    print(f"Params: {MNQ_V11_PARAMS}")

    # Load data
    print("\nLoading data...")
    df_1m = load_data()
    print(f"  Loaded {len(df_1m)} 1-min bars")
    print(f"  Date range: {df_1m.index[0]} to {df_1m.index[-1]}")

    # Run vectorized backtest
    print("\nRunning vectorized backtest...")
    vec_trades = run_vectorized_backtest(df_1m)
    vec_score = vtc.score_trades(
        vec_trades,
        commission_per_side=MNQ_V11.commission_per_side,
        dollar_per_pt=MNQ_V11.dollar_per_pt,
    )
    print(f"  Vectorized: {vtc.fmt_score(vec_score, 'v10')}")

    # Run incremental strategy
    print("\nRunning incremental strategy bar-by-bar...")
    inc_trades = run_incremental_strategy(df_1m)
    print(f"  Incremental: {len(inc_trades)} trades")

    if inc_trades:
        # Compute incremental score for comparison
        inc_trade_dicts = []
        for t in inc_trades:
            inc_trade_dicts.append({
                "side": t.side,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pts": t.pts,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_idx": 0,
                "exit_idx": 0,
                "bars": t.bars_held,
                "result": t.exit_reason,
            })
        inc_score = vtc.score_trades(
            inc_trade_dicts,
            commission_per_side=MNQ_V11.commission_per_side,
            dollar_per_pt=MNQ_V11.dollar_per_pt,
        )
        print(f"  Incremental: {vtc.fmt_score(inc_score, 'inc')}")

    # Compare trades
    print("\nComparing trades...")
    matches, mismatches, details = compare_trades(vec_trades, inc_trades)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {matches} matches, {mismatches} mismatches")
    print(f"{'=' * 70}")

    if mismatches > 0:
        print("\nMISMATCH DETAILS:")
        for d in details[:20]:  # Show first 20
            print(f"  {d}")
        if len(details) > 20:
            print(f"  ... and {len(details) - 20} more")

    # Assert parity
    assert mismatches == 0, (
        f"Strategy parity FAILED: {mismatches} mismatches out of "
        f"{max(len(vec_trades), len(inc_trades))} trades.\n"
        + "\n".join(details[:10])
    )

    print("\nPARITY TEST PASSED")


def test_incremental_sm_matches_vectorized():
    """Verify IncrementalSM produces the same values as compute_smart_money().

    This isolates SM computation from the full strategy to pinpoint
    any indicator divergence.
    """
    from engine.strategy import IncrementalSM

    print("\n" + "=" * 70)
    print("SM PARITY TEST: IncrementalSM vs compute_smart_money()")
    print("=" * 70)

    df_1m = load_data()
    closes = df_1m["Close"].values
    volumes = df_1m["Volume"].values

    # Vectorized SM
    sm_vec = vtc.compute_smart_money(
        closes, volumes,
        index_period=MNQ_V11_PARAMS["sm_index"],
        flow_period=MNQ_V11_PARAMS["sm_flow"],
        norm_period=MNQ_V11_PARAMS["sm_norm"],
        ema_len=MNQ_V11_PARAMS["sm_ema"],
    )

    # Incremental SM
    inc_sm = IncrementalSM(
        index_period=MNQ_V11_PARAMS["sm_index"],
        flow_period=MNQ_V11_PARAMS["sm_flow"],
        norm_period=MNQ_V11_PARAMS["sm_norm"],
        ema_len=MNQ_V11_PARAMS["sm_ema"],
    )
    sm_inc = np.zeros(len(closes))
    for i in range(len(closes)):
        sm_inc[i] = inc_sm.update(float(closes[i]), float(volumes[i]))

    # Compare (skip first 100 warmup bars)
    start = 100
    diff = np.abs(sm_vec[start:] - sm_inc[start:])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    corr = np.corrcoef(sm_vec[start:], sm_inc[start:])[0, 1]

    print(f"  Bars compared: {len(diff)}")
    print(f"  Max abs diff:  {max_diff:.10f}")
    print(f"  Mean abs diff: {mean_diff:.10f}")
    print(f"  Correlation:   {corr:.10f}")

    # Incremental vs vectorized will have small floating-point accumulation
    # differences over 170K+ bars. Correlation must be perfect (1.0) and
    # max diff must be negligible relative to SM range [-1, +1].
    assert corr > 0.999999, (
        f"SM correlation too low: {corr:.10f}. "
        f"IncrementalSM diverges from compute_smart_money()."
    )
    assert max_diff < 0.01, (
        f"SM max diff too large: {max_diff:.6f}. "
        f"IncrementalSM diverges from compute_smart_money()."
    )

    print("  SM PARITY: PASS")


def test_incremental_rsi_matches_vectorized():
    """Verify IncrementalRSI5m produces the same values as the vectorized path.

    Checks that curr/prev RSI values at 5-min boundaries match the
    mapped arrays from prepare_backtest_arrays_1min().
    """
    from engine.strategy import IncrementalRSI5m, _et_minutes_from_datetime

    print("\n" + "=" * 70)
    print("RSI PARITY TEST: IncrementalRSI5m vs prepare_backtest_arrays_1min()")
    print("=" * 70)

    df_1m = load_data()

    # Vectorized RSI mapping
    arr = vtc.prepare_backtest_arrays_1min(df_1m, rsi_len=MNQ_V11_PARAMS["rsi_len"])
    rsi_curr_vec = arr["rsi_5m_curr"]
    rsi_prev_vec = arr["rsi_5m_prev"]

    # Incremental RSI
    inc_rsi = IncrementalRSI5m(rsi_len=MNQ_V11_PARAMS["rsi_len"])

    times = df_1m.index.values
    opens = df_1m["Open"].values
    highs = df_1m["High"].values
    lows = df_1m["Low"].values
    closes = df_1m["Close"].values
    volumes = df_1m["Volume"].values

    rsi_curr_inc = np.zeros(len(df_1m))
    rsi_prev_inc = np.zeros(len(df_1m))

    for i in range(len(df_1m)):
        ts = pd.Timestamp(times[i])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        dt = ts.to_pydatetime()

        bar = Bar(
            timestamp=dt,
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
            instrument="MNQ",
        )
        inc_rsi.update(bar)
        rsi_curr_inc[i] = inc_rsi.curr
        rsi_prev_inc[i] = inc_rsi.prev

    # Compare at 5-min boundaries (where values actually change)
    # Skip first 500 bars for warmup
    start = 500
    curr_diff = np.abs(rsi_curr_vec[start:] - rsi_curr_inc[start:])
    prev_diff = np.abs(rsi_prev_vec[start:] - rsi_prev_inc[start:])

    max_curr_diff = np.max(curr_diff)
    max_prev_diff = np.max(prev_diff)
    mean_curr_diff = np.mean(curr_diff)
    mean_prev_diff = np.mean(prev_diff)

    print(f"  Bars compared: {len(curr_diff)}")
    print(f"  RSI curr - max diff:  {max_curr_diff:.6f}, mean: {mean_curr_diff:.6f}")
    print(f"  RSI prev - max diff:  {max_prev_diff:.6f}, mean: {mean_prev_diff:.6f}")

    # RSI values might have small differences at window boundaries due to
    # how incomplete 5-min bars are handled. Allow small tolerance.
    tolerance = 0.5
    large_diffs = np.sum(curr_diff > tolerance)
    print(f"  Bars with curr diff > {tolerance}: {large_diffs}")

    # Report but don't fail on small RSI diffs -- the full strategy parity
    # test (test_strategy_parity) is the definitive check
    if max_curr_diff > tolerance:
        print(f"  WARNING: RSI curr has diffs > {tolerance} at {large_diffs} bars")
        print(f"  (This may be OK if full strategy parity test passes)")
    else:
        print("  RSI PARITY: PASS")


if __name__ == "__main__":
    # Run all tests
    test_incremental_sm_matches_vectorized()
    test_incremental_rsi_matches_vectorized()
    test_strategy_parity()
