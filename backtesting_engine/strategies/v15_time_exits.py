"""
v15_time_exits.py — Fixed-Time / Target-Stop / Trailing Exit Sweep
===================================================================
Uses IDENTICAL SM-flip + RSI-cross entry signals from v11 MNQ,
but replaces SM-flip exit with mechanical exits:

  Time exits (T):    Hold for N bars, exit. 50pt backstop.
  Target/Stop (TS):  Exit at TP or SL, whichever first.
  Trailing (TR):     Trail by D pts after reaching A pts profit. 50pt backstop.

Tests whether the entry signal has persistent edge over different
holding periods / exit mechanics — no indicators needed for exits.

Usage:
  python3 v15_time_exits.py                 # TRAIN period
  python3 v15_time_exits.py --oos           # TEST (OOS) period
  python3 v15_time_exits.py --all           # Both TRAIN and OOS
"""

import sys
import time as _time
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

# ---------------------------------------------------------------------------
# Constants (v11 MNQ baseline)
# ---------------------------------------------------------------------------

TRAIN_START = pd.Timestamp("2025-08-17", tz='UTC')
TRAIN_END = pd.Timestamp("2025-11-17", tz='UTC')
TEST_START = pd.Timestamp("2025-11-17", tz='UTC')
TEST_END = pd.Timestamp("2026-02-14", tz='UTC')

V11_RSI_LEN = 8
V11_RSI_BUY = 60
V11_RSI_SELL = 40
V11_COOLDOWN = 20
V11_MAX_LOSS = 50
V11_SM_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Sweep Configurations
# ---------------------------------------------------------------------------

# Time exits: hold for N bars then exit at open of bar N+1
# Still has 50pt max loss backstop and EOD close
TIME_CONFIGS = {}
for n in [3, 5, 10, 15, 20, 30]:
    TIME_CONFIGS[f'T{n}'] = {'exit_type': 'time', 'hold_bars': n}

# Target/Stop pairs: exit at TP or SL (pts), whichever first, plus EOD
TPSL_CONFIGS = {}
for tp, sl in [(10, 10), (15, 10), (20, 10),
               (15, 15), (20, 15), (25, 15),
               (20, 20), (25, 20), (30, 20),
               (30, 25), (40, 25), (50, 30)]:
    key = f'TS_{tp}_{sl}'
    TPSL_CONFIGS[key] = {'exit_type': 'tpsl', 'tp_pts': tp, 'sl_pts': sl}

# Trailing stops: activate trail after A pts profit, trail by D pts
TRAIL_CONFIGS = {}
for a, d in [(5, 8), (5, 10), (10, 10), (10, 15), (15, 15), (20, 15)]:
    key = f'TR_{a}_{d}'
    TRAIL_CONFIGS[key] = {'exit_type': 'trail', 'trail_activate': a,
                          'trail_dist': d}

ALL_CONFIGS = {}
ALL_CONFIGS.update(TIME_CONFIGS)
ALL_CONFIGS.update(TPSL_CONFIGS)
ALL_CONFIGS.update(TRAIL_CONFIGS)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_mechanical_exits(arr, config):
    """Run v11 entries with mechanical exits.

    Entry logic: IDENTICAL to v11 baseline (SM flip + RSI cross).
    Exit logic: determined by config['exit_type']:
      'time'  - exit after hold_bars bars. 50pt backstop.
      'tpsl'  - exit at tp_pts profit or sl_pts loss.
      'trail' - trail by trail_dist after trail_activate profit. 50pt backstop.

    All exits: check bar i-1 data, fill at bar i open (no look-ahead).
    EOD close is always highest priority.
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

    exit_type = config['exit_type']

    trades = []
    trade_state = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    # Trail state
    max_fav = 0.0  # max favorable excursion (pts)
    trail_active = False

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

        # Episode reset (IDENTICAL to baseline — NEVER modify)
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # ===== EXITS =====

        if trade_state != 0:
            side = "long" if trade_state == 1 else "short"
            unrealized = (closes[i - 1] - entry_price) * trade_state
            bars_held = i - entry_idx

            # 1. EOD close (highest priority)
            if bar_et >= NY_CLOSE_ET:
                close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
                trade_state = 0
                exit_bar = i
                trail_active = False
                max_fav = 0.0
                continue

            if exit_type == 'time':
                # 2a. Max loss backstop (50pt)
                if V11_MAX_LOSS > 0 and unrealized <= -V11_MAX_LOSS:
                    close_trade(side, entry_price, opens[i], entry_idx, i, "SL")
                    trade_state = 0
                    exit_bar = i
                    continue

                # 2b. Time exit
                if bars_held >= config['hold_bars']:
                    close_trade(side, entry_price, opens[i], entry_idx, i, "TIME")
                    trade_state = 0
                    exit_bar = i
                    continue

            elif exit_type == 'tpsl':
                tp = config['tp_pts']
                sl = config['sl_pts']

                # 2a. Stop loss
                if unrealized <= -sl:
                    close_trade(side, entry_price, opens[i], entry_idx, i, "SL")
                    trade_state = 0
                    exit_bar = i
                    continue

                # 2b. Take profit
                if unrealized >= tp:
                    close_trade(side, entry_price, opens[i], entry_idx, i, "TP")
                    trade_state = 0
                    exit_bar = i
                    continue

            elif exit_type == 'trail':
                # Track max favorable excursion
                if unrealized > max_fav:
                    max_fav = unrealized

                # 2a. Max loss backstop (50pt)
                if V11_MAX_LOSS > 0 and unrealized <= -V11_MAX_LOSS:
                    close_trade(side, entry_price, opens[i], entry_idx, i, "SL")
                    trade_state = 0
                    exit_bar = i
                    trail_active = False
                    max_fav = 0.0
                    continue

                # 2b. Activate trail
                if max_fav >= config['trail_activate']:
                    trail_active = True

                # 2c. Trail stop
                if trail_active:
                    trail_stop_level = max_fav - config['trail_dist']
                    if unrealized <= trail_stop_level:
                        close_trade(side, entry_price, opens[i], entry_idx, i,
                                    "TRAIL")
                        trade_state = 0
                        exit_bar = i
                        trail_active = False
                        max_fav = 0.0
                        continue

        # ===== ENTRIES (IDENTICAL to baseline — NEVER modify) =====

        if trade_state == 0:
            in_session = NY_OPEN_ET <= bar_et <= NY_LAST_ENTRY_ET
            cd_ok = (i - exit_bar) >= V11_COOLDOWN

            if in_session and cd_ok:
                # Long entry
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    max_fav = 0.0
                    trail_active = False

                # Short entry
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    max_fav = 0.0
                    trail_active = False

    return trades


# ---------------------------------------------------------------------------
# Run Sweep
# ---------------------------------------------------------------------------

def run_sweep(arr, trades_baseline, sc_baseline, configs, period_name):
    """Run all configs and return results sorted by PF."""
    results = []

    for key, config in configs.items():
        trades = run_mechanical_exits(arr, config)
        sc = score_trades(trades)
        if sc is None:
            results.append({
                'key': key, 'config': config, 'trades': 0,
                'pf': 0, 'wr': 0, 'net': 0, 'dd': 0, 'dpf': 0,
                'sharpe': 0, 'avg_bars': 0, 'exits': {},
            })
            continue

        dpf = sc['pf'] - sc_baseline['pf']
        results.append({
            'key': key, 'config': config,
            'trades': sc['count'], 'pf': sc['pf'],
            'wr': sc['win_rate'], 'net': sc['net_dollar'],
            'dd': sc['max_dd_dollar'], 'dpf': dpf,
            'sharpe': sc['sharpe'], 'avg_bars': sc['avg_bars'],
            'exits': sc.get('exits', {}),
            'sc': sc, 'trades_list': trades,
        })

    return results


def print_results_table(results, sc_baseline, category_name):
    """Print a formatted results table for one category."""
    print(f"\n  --- {category_name} ---")
    print(f"  {'Config':<14} | {'Trades':>6} {'PF':>7} {'dPF':>7} {'WR':>6} "
          f"{'Net$':>9} {'DD$':>8} {'Sharpe':>7} {'AvgBars':>7} | Exits")
    print(f"  {'-'*14}-+-{'-'*63}-+-------")

    for r in results:
        exit_str = ' '.join(f"{k}:{v}" for k, v in sorted(r['exits'].items()))
        line = (f"  {r['key']:<14} | "
                f"{r['trades']:>6} {r['pf']:>7.3f} {r['dpf']:>+7.3f} "
                f"{r['wr']:>5.1f}% ${r['net']:>+8.2f} ${r['dd']:>7.2f} "
                f"{r['sharpe']:>7.3f} {r['avg_bars']:>7.1f} | {exit_str}")
        print(line)


# ---------------------------------------------------------------------------
# Statistical Validation (for top configs)
# ---------------------------------------------------------------------------

def validate_top(key, trades_baseline, trades_feature, sc_baseline, sc_feature):
    """Run bootstrap CI and permutation test on a top config."""
    dpf = sc_feature['pf'] - sc_baseline['pf']

    ci_point, ci_lo, ci_hi = bootstrap_ci(trades_feature, metric='pf')
    ci_excludes = ci_lo > sc_baseline['pf']

    obs_diff, p_val = permutation_test(trades_baseline, trades_feature)

    count_ratio = sc_feature['count'] / sc_baseline['count']
    dwr = sc_feature['win_rate'] - sc_baseline['win_rate']

    # Lucky trade removal
    pts = np.array([t['pts'] for t in trades_feature])
    best_idx = np.argmax(pts)
    trades_no_best = [t for j, t in enumerate(trades_feature) if j != best_idx]
    sc_no_best = score_trades(trades_no_best)
    lucky_ok = sc_no_best is not None and sc_no_best['pf'] > 1.0

    criteria = [
        ("PF improvement > 0.1", dpf > 0.1, f"dPF={dpf:+.3f}"),
        ("Bootstrap CI excludes baseline", ci_excludes,
         f"CI=[{ci_lo:.3f}, {ci_hi:.3f}], base={sc_baseline['pf']:.3f}"),
        ("Permutation p < 0.05", p_val < 0.05, f"p={p_val:.4f}"),
        ("Trade count > 50%", count_ratio > 0.5,
         f"ratio={count_ratio:.2f} ({sc_feature['count']}/{sc_baseline['count']})"),
        ("Win rate not degraded > 5%", dwr > -5,
         f"dWR={dwr:+.1f}%"),
        ("Drawdown not >20% worse",
         sc_feature['max_dd_dollar'] >= sc_baseline['max_dd_dollar'] * 1.2,
         f"feat DD=${sc_feature['max_dd_dollar']:.0f}, "
         f"base DD=${sc_baseline['max_dd_dollar']:.0f}"),
        ("Survives lucky trade removal", lucky_ok,
         f"PF w/o best={sc_no_best['pf']:.3f}" if sc_no_best else "N/A"),
    ]

    passes = sum(1 for _, p, _ in criteria if p)
    verdict = "ADOPT" if passes >= 5 else ("MAYBE" if passes >= 3 else "REJECT")

    print(f"\n    {key}: {passes}/7 -> {verdict}")
    for name, passed, detail in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"      [{status}] {name}: {detail}")

    return {'key': key, 'verdict': verdict, 'passes': passes, 'dpf': dpf,
            'p_val': p_val}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_period(ohlcv_1m, period_start, period_end, period_name):
    """Run full sweep on one period."""
    period_data = ohlcv_1m[(ohlcv_1m.index >= period_start) &
                            (ohlcv_1m.index < period_end)]
    if len(period_data) < 100:
        print(f"ERROR: Only {len(period_data)} bars")
        return None

    print(f"\n{'='*75}")
    print(f"  {period_name}: {period_start.date()} to {period_end.date()}")
    print(f"  {len(period_data):,} bars")
    print(f"{'='*75}")

    # Prepare arrays
    arr = prepare_backtest_arrays_1min(period_data, rsi_len=V11_RSI_LEN)

    # Baseline (SM flip exits)
    trades_baseline = run_v9_baseline(
        arr, rsi_len=V11_RSI_LEN, rsi_buy=V11_RSI_BUY,
        rsi_sell=V11_RSI_SELL, cooldown=V11_COOLDOWN,
        max_loss_pts=V11_MAX_LOSS,
    )
    sc_baseline = score_trades(trades_baseline)
    print(f"\n  BASELINE (SM flip exit): {fmt_score(sc_baseline, 'v11')}")
    if sc_baseline and sc_baseline.get('exits'):
        exit_parts = [f"{k}:{v}" for k, v in sorted(sc_baseline['exits'].items())]
        print(f"  Exits: {' '.join(exit_parts)}")

    # Run each category
    time_results = run_sweep(arr, trades_baseline, sc_baseline,
                             TIME_CONFIGS, period_name)
    tpsl_results = run_sweep(arr, trades_baseline, sc_baseline,
                             TPSL_CONFIGS, period_name)
    trail_results = run_sweep(arr, trades_baseline, sc_baseline,
                              TRAIL_CONFIGS, period_name)

    # Print tables
    print_results_table(time_results, sc_baseline, "TIME EXITS (hold N bars)")
    print_results_table(tpsl_results, sc_baseline, "TARGET/STOP EXITS")
    print_results_table(trail_results, sc_baseline, "TRAILING STOP EXITS")

    # Find top configs (positive dPF, PF > 1.0)
    all_results = time_results + tpsl_results + trail_results
    top = [r for r in all_results if r['dpf'] > 0 and r['pf'] > 1.0]
    top.sort(key=lambda r: r['dpf'], reverse=True)

    print(f"\n{'='*75}")
    print(f"  TOP CONFIGS (positive dPF, PF > 1.0)")
    print(f"{'='*75}")

    if not top:
        print("  None — all configs worse than baseline SM-flip exit.")
        return {'baseline': sc_baseline, 'top': [], 'all': all_results}

    print(f"\n  {'Config':<14} | {'PF':>7} {'dPF':>7} {'Net$':>9} "
          f"{'DD$':>8} {'Sharpe':>7} {'Trades':>6} {'AvgBars':>7}")
    print(f"  {'-'*14}-+-{'-'*57}")
    for r in top[:10]:
        print(f"  {r['key']:<14} | {r['pf']:>7.3f} {r['dpf']:>+7.3f} "
              f"${r['net']:>+8.2f} ${r['dd']:>7.2f} {r['sharpe']:>7.3f} "
              f"{r['trades']:>6} {r['avg_bars']:>7.1f}")

    # Validate top 5
    print(f"\n  --- Statistical Validation (top 5) ---")
    validations = []
    for r in top[:5]:
        v = validate_top(r['key'], trades_baseline, r['trades_list'],
                         sc_baseline, r['sc'])
        validations.append(v)

    return {'baseline': sc_baseline, 'top': top, 'validations': validations,
            'all': all_results, 'trades_baseline': trades_baseline}


def main():
    parser = argparse.ArgumentParser(
        description="v15: mechanical exit sweep on v11 MNQ entries")
    parser.add_argument("--instrument", default="MNQ")
    parser.add_argument("--oos", action="store_true",
                        help="Run on OOS (test) period only")
    parser.add_argument("--all", action="store_true",
                        help="Run on both TRAIN and OOS")
    args = parser.parse_args()

    print("=" * 75)
    print("v15 MECHANICAL EXIT SWEEP")
    print("Same v11 entries (SM flip + RSI cross), different exits")
    print("=" * 75)
    print(f"  Instrument: {args.instrument}")
    print(f"  Commission: $0.52/side")

    # Load data
    ohlcv_1m = load_instrument_1min(args.instrument)
    if ohlcv_1m.index.tz is None:
        ohlcv_1m.index = ohlcv_1m.index.tz_localize('UTC')
    print(f"  OHLCV: {len(ohlcv_1m):,} bars total")

    if args.oos:
        run_period(ohlcv_1m, TEST_START, TEST_END, "TEST (OOS)")
    elif args.all:
        train_result = run_period(ohlcv_1m, TRAIN_START, TRAIN_END, "TRAIN")
        test_result = run_period(ohlcv_1m, TEST_START, TEST_END, "TEST (OOS)")

        # Cross-period comparison
        if train_result and test_result:
            train_top = train_result.get('top', [])
            test_all = {r['key']: r for r in test_result.get('all', [])}

            print(f"\n{'='*75}")
            print("TRAIN vs OOS COMPARISON (top train configs)")
            print(f"{'='*75}")
            print(f"\n  {'Config':<14} | {'TRAIN PF':>9} {'TRAIN dPF':>9} | "
                  f"{'OOS PF':>9} {'OOS dPF':>9} | {'Held up?':>8}")
            print(f"  {'-'*14}-+-{'-'*20}-+-{'-'*20}-+-{'-'*8}")

            for r in train_top[:10]:
                oos = test_all.get(r['key'])
                if oos:
                    held = "YES" if oos['dpf'] > 0 and oos['pf'] > 1.0 else "NO"
                    print(f"  {r['key']:<14} | {r['pf']:>9.3f} {r['dpf']:>+9.3f} | "
                          f"{oos['pf']:>9.3f} {oos['dpf']:>+9.3f} | {held:>8}")
    else:
        run_period(ohlcv_1m, TRAIN_START, TRAIN_END, "TRAIN")

    print("\nDone!")


if __name__ == "__main__":
    main()
