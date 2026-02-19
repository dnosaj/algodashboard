"""
v12 CVD Integration Test (Revised)
====================================
Tests CVD / Volume Delta as a feature on top of v11 MNQ baseline.

Key changes from v1:
  - 400-bar normalization replaced with fast 20-30 bar windows
  - Entry filter uses custom engine (doesn't corrupt SM exit logic)
  - Delta burst detector uses raw per-bar delta (no normalization at all)
  - Conviction sizing analysis (variable position size, not filtering)

Train/Test split:
  TRAIN: Aug 17 - Nov 16, 2025 (~3 months)
  TEST:  Nov 17, 2025 - Feb 13, 2026 (~3 months)

Variants:
  F1. Delta burst exit (5-bar sum, threshold=300)
  F2. Delta burst exit (10-bar sum, threshold=300)
  F3. Delta burst exit (10-bar sum, threshold=500)
  G1. Proper entry filter (fast CVD 20-bar norm)
  G2. Proper entry filter (fast CVD 30-bar norm)
  H.  Conviction sizing (2x on CVD agree, 1x on disagree)

Usage:
  python3 v12_cvd_test.py --delta data/databento_NQ_delta_1min_*.csv
  python3 v12_cvd_test.py --delta data/delta.csv --variant F1
  python3 v12_cvd_test.py --delta data/delta.csv --oos
"""

import sys
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


def load_delta_bars(delta_path):
    """Load volume delta bars CSV and return UTC-indexed DataFrame."""
    df = pd.read_csv(delta_path)
    df['time_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time_utc')
    return df


def align_delta_data(ohlcv_1m, delta_df):
    """Align all delta data to OHLCV timestamps.

    Computes fast signals from raw delta (no 400-bar normalization).

    Returns dict with:
      delta       - raw per-bar delta (buy_vol - sell_vol)
      roll_5      - rolling 5-bar sum of delta
      roll_10     - rolling 10-bar sum of delta
      fast_cvd_20 - CVD normalized with 20-bar rolling max
      fast_cvd_30 - CVD normalized with 30-bar rolling max
    """
    # Align raw delta to OHLCV index
    raw_delta = delta_df['delta'].reindex(ohlcv_1m.index, method='ffill').fillna(0).values
    cvd_raw = delta_df['CVD'].reindex(ohlcv_1m.index, method='ffill').fillna(0).values

    n = len(raw_delta)

    # Rolling sums of raw delta (no normalization, pure speed)
    roll_5 = np.zeros(n)
    roll_10 = np.zeros(n)
    for i in range(n):
        s5 = max(0, i - 4)
        roll_5[i] = np.sum(raw_delta[s5:i + 1])
        s10 = max(0, i - 9)
        roll_10[i] = np.sum(raw_delta[s10:i + 1])

    # Fast CVD normalization (20-bar and 30-bar windows)
    def fast_normalize(arr, window):
        out = np.zeros(len(arr))
        for i in range(len(arr)):
            s = max(0, i - window + 1)
            abs_max = np.max(np.abs(arr[s:i + 1]))
            out[i] = arr[i] / abs_max if abs_max > 0 else 0.0
        return out

    fast_cvd_20 = fast_normalize(cvd_raw, 20)
    fast_cvd_30 = fast_normalize(cvd_raw, 30)

    return {
        'delta': raw_delta,
        'roll_5': roll_5,
        'roll_10': roll_10,
        'fast_cvd_20': fast_cvd_20,
        'fast_cvd_30': fast_cvd_30,
    }


# ---------------------------------------------------------------------------
# Shared engine for custom variants
# ---------------------------------------------------------------------------

def _run_baseline(arr):
    """Run v11 baseline."""
    return run_v9_baseline(
        arr, rsi_len=V11_RSI_LEN, rsi_buy=V11_RSI_BUY,
        rsi_sell=V11_RSI_SELL, cooldown=V11_COOLDOWN,
        max_loss_pts=V11_MAX_LOSS,
    )


def _run_custom_engine(arr, delta_data,
                       burst_window=0, burst_threshold=0,
                       entry_filter_arr=None):
    """Custom v11 engine with optional delta burst exit and CVD entry filter.

    This is a clean reimplementation that:
    - Uses SM for entries and SM flip for exits (standard v11)
    - Optionally adds a delta burst exit (raw rolling delta sum)
    - Optionally filters entries by CVD agreement (checked ONLY at entry bar)
    - Exit logic (SM flip, SL, EOD) is NEVER modified

    Args:
        burst_window: rolling delta window for burst exit (0 = disabled)
        burst_threshold: abs threshold for burst exit
        entry_filter_arr: array of CVD values; if provided, entry requires
                          agreement (>0 for longs, <0 for shorts)
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

    # Select rolling delta array for burst detection
    if burst_window == 5:
        roll_delta = delta_data['roll_5']
    elif burst_window == 10:
        roll_delta = delta_data['roll_10']
    else:
        roll_delta = None

    trades = []
    trade_state = 0
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

        # Episode reset (standard v11 logic, untouched)
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

        # --- Exits for open positions ---
        if trade_state == 1:
            # Max loss stop (checks bar i-1 close, fills at bar i open)
            if V11_MAX_LOSS > 0 and closes[i - 1] <= entry_price - V11_MAX_LOSS:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # Delta burst exit: sudden aggressive selling
            if roll_delta is not None and roll_delta[i - 1] < -burst_threshold:
                close_trade("long", entry_price, opens[i], entry_idx, i, "BURST")
                trade_state = 0
                exit_bar = i
                continue

            # SM flip exit (standard, untouched)
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        elif trade_state == -1:
            if V11_MAX_LOSS > 0 and closes[i - 1] >= entry_price + V11_MAX_LOSS:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # Delta burst exit: sudden aggressive buying
            if roll_delta is not None and roll_delta[i - 1] > burst_threshold:
                close_trade("short", entry_price, opens[i], entry_idx, i, "BURST")
                trade_state = 0
                exit_bar = i
                continue

            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        # --- Entry logic ---
        if trade_state == 0:
            in_session = NY_OPEN_ET <= bar_et <= NY_LAST_ENTRY_ET
            cd_ok = (i - exit_bar) >= V11_COOLDOWN

            if in_session and cd_ok:
                # Long entry
                if sm_bull and rsi_long_trigger and not long_used:
                    # CVD entry filter (checked ONLY here, exits untouched)
                    cvd_ok = True
                    if entry_filter_arr is not None:
                        cvd_ok = entry_filter_arr[i - 1] > 0

                    if cvd_ok:
                        trade_state = 1
                        entry_price = opens[i]
                        entry_idx = i
                        long_used = True

                # Short entry
                elif sm_bear and rsi_short_trigger and not short_used:
                    cvd_ok = True
                    if entry_filter_arr is not None:
                        cvd_ok = entry_filter_arr[i - 1] < 0

                    if cvd_ok:
                        trade_state = -1
                        entry_price = opens[i]
                        entry_idx = i
                        short_used = True

    return trades


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------

def run_variant_f1(arr, delta_data, **kw):
    """F1: Delta burst exit (5-bar, threshold=300)."""
    return _run_custom_engine(arr, delta_data, burst_window=5, burst_threshold=300)


def run_variant_f2(arr, delta_data, **kw):
    """F2: Delta burst exit (10-bar, threshold=300)."""
    return _run_custom_engine(arr, delta_data, burst_window=10, burst_threshold=300)


def run_variant_f3(arr, delta_data, **kw):
    """F3: Delta burst exit (10-bar, threshold=500)."""
    return _run_custom_engine(arr, delta_data, burst_window=10, burst_threshold=500)


def run_variant_g1(arr, delta_data, **kw):
    """G1: Proper entry filter (fast CVD 20-bar norm)."""
    return _run_custom_engine(arr, delta_data,
                              entry_filter_arr=delta_data['fast_cvd_20'])


def run_variant_g2(arr, delta_data, **kw):
    """G2: Proper entry filter (fast CVD 30-bar norm)."""
    return _run_custom_engine(arr, delta_data,
                              entry_filter_arr=delta_data['fast_cvd_30'])


def run_variant_fg(arr, delta_data, **kw):
    """FG: Best burst exit + entry filter combined."""
    return _run_custom_engine(arr, delta_data,
                              burst_window=5, burst_threshold=300,
                              entry_filter_arr=delta_data['fast_cvd_20'])


# ---------------------------------------------------------------------------
# Conviction sizing (Variant H)
# ---------------------------------------------------------------------------

def run_conviction_sizing(arr, delta_data, trades_baseline):
    """H: Conviction sizing -- 2x size when CVD agrees, 1x when it disagrees.

    Doesn't change entries or exits. Changes position size per trade.
    Uses fast_cvd_20 at entry bar to determine conviction.

    Returns modified trades with 'size_mult' field and custom scoring.
    """
    fast_cvd = delta_data['fast_cvd_20']
    times = arr['times']

    # Build a time -> index lookup for fast_cvd
    time_to_idx = {}
    for idx in range(len(times)):
        time_to_idx[times[idx]] = idx

    trades_sized = []
    n_agree = 0
    n_disagree = 0

    for trade in trades_baseline:
        t = dict(trade)  # copy

        # Find CVD value at entry
        entry_time = trade['entry_time']
        idx = time_to_idx.get(entry_time, None)
        if idx is None or idx < 1:
            t['size_mult'] = 1
            trades_sized.append(t)
            continue

        cvd_at_entry = fast_cvd[idx - 1]  # bar before entry (signal bar)

        if trade['side'] == 'long':
            agrees = cvd_at_entry > 0
        else:
            agrees = cvd_at_entry < 0

        if agrees:
            t['size_mult'] = 2
            n_agree += 1
        else:
            t['size_mult'] = 1
            n_disagree += 1

        trades_sized.append(t)

    return trades_sized, n_agree, n_disagree


def score_trades_sized(trades, commission_per_side=0.52, dollar_per_pt=2.0):
    """Score trades with variable position sizes.

    Each trade has a 'size_mult' field (1 or 2).
    Commission and P&L scale with size.
    """
    if not trades:
        return None

    n = len(trades)
    comm_pts = (commission_per_side * 2) / dollar_per_pt

    net_each = []
    for t in trades:
        mult = t.get('size_mult', 1)
        # Net pts per contract, then multiply by contracts
        net_per_contract = t['pts'] - comm_pts
        net_each.append(net_per_contract * mult)

    net_each = np.array(net_each)
    net_pts = net_each.sum()
    wins = net_each[net_each > 0]
    losses = net_each[net_each <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100
    cum = np.cumsum(net_each)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()
    avg_pts = np.mean(net_each)

    daily_pnl = net_each * dollar_per_pt
    sharpe = 0.0
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)

    # Total contracts traded
    total_contracts = sum(t.get('size_mult', 1) for t in trades)

    return {
        "count": n,
        "net_pts": round(net_pts, 2),
        "pf": round(pf, 3),
        "win_rate": round(wr, 1),
        "net_dollar": round(net_pts * dollar_per_pt, 2),
        "max_dd_pts": round(mdd, 2),
        "max_dd_dollar": round(mdd * dollar_per_pt, 2),
        "avg_pts": round(avg_pts, 2),
        "sharpe": round(sharpe, 3),
        "total_contracts": total_contracts,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

VARIANTS = {
    'F1': ('Delta burst exit (5bar/300)',   run_variant_f1),
    'F2': ('Delta burst exit (10bar/300)',  run_variant_f2),
    'F3': ('Delta burst exit (10bar/500)',  run_variant_f3),
    'G1': ('Entry filter (fast CVD 20)',    run_variant_g1),
    'G2': ('Entry filter (fast CVD 30)',    run_variant_g2),
    'FG': ('Burst exit + entry filter',     run_variant_fg),
}

ALL_VARIANT_KEYS = ['F1', 'F2', 'F3', 'G1', 'G2', 'FG', 'H']


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
    dd_ok = feat_dd >= base_dd * 1.2

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


def evaluate_conviction(trades_baseline, trades_sized, n_agree, n_disagree,
                         commission=0.52, dollar_per_pt=2.0):
    """Evaluate conviction sizing (Variant H)."""
    print(f"\n  --- Variant H: Conviction Sizing (2x agree, 1x disagree) ---")

    sc_base = score_trades(trades_baseline, commission, dollar_per_pt)
    sc_sized = score_trades_sized(trades_sized, commission, dollar_per_pt)

    if sc_base is None or sc_sized is None:
        print("  SKIP (no trades)")
        return None

    # Baseline with uniform 1x sizing
    base_net = sc_base['net_dollar']
    # Sized result
    sized_net = sc_sized['net_dollar']

    print(f"  CVD-agree trades:    {n_agree} (2x size)")
    print(f"  CVD-disagree trades: {n_disagree} (1x size)")
    print(f"  Total contracts:     {sc_sized['total_contracts']} "
          f"(baseline: {sc_base['count']})")
    print(f"")
    print(f"  Baseline (uniform 1x): {sc_base['count']} trades, "
          f"PF {sc_base['pf']}, Net ${base_net:+.2f}, "
          f"MaxDD ${sc_base['max_dd_dollar']:.2f}")
    print(f"  Conviction sizing:     {sc_sized['count']} trades, "
          f"PF {sc_sized['pf']}, Net ${sized_net:+.2f}, "
          f"MaxDD ${sc_sized['max_dd_dollar']:.2f}")
    print(f"")
    print(f"  Net $ improvement: ${sized_net - base_net:+.2f} "
          f"({(sized_net/base_net - 1)*100:+.1f}%)" if base_net != 0 else "")
    print(f"  MaxDD change: ${sc_sized['max_dd_dollar'] - sc_base['max_dd_dollar']:+.2f}")

    # Breakdown: what would agree-only and disagree-only look like?
    agree_trades = [t for t in trades_sized if t.get('size_mult', 1) == 2]
    disagree_trades = [t for t in trades_sized if t.get('size_mult', 1) == 1]

    sc_agree = score_trades(agree_trades, commission, dollar_per_pt) if agree_trades else None
    sc_disagree = score_trades(disagree_trades, commission, dollar_per_pt) if disagree_trades else None

    print(f"\n  Breakdown (1x sizing for comparison):")
    if sc_agree:
        print(f"    Agree trades:    {fmt_score(sc_agree, 'AGREE')}")
    if sc_disagree:
        print(f"    Disagree trades: {fmt_score(sc_disagree, 'DISAGREE')}")

    if sc_agree and sc_disagree:
        pf_spread = sc_agree['pf'] - sc_disagree['pf']
        wr_spread = sc_agree['win_rate'] - sc_disagree['win_rate']
        print(f"\n    PF spread (agree - disagree): {pf_spread:+.3f}")
        print(f"    WR spread (agree - disagree): {wr_spread:+.1f}%")

        if pf_spread > 0.3:
            print(f"    STRONG separation -- conviction sizing is well-justified")
        elif pf_spread > 0.1:
            print(f"    Moderate separation -- conviction sizing has some value")
        else:
            print(f"    Weak separation -- conviction sizing adds little value")

    return {
        'name': 'H', 'desc': 'Conviction sizing',
        'sc_base': sc_base, 'sc_sized': sc_sized,
        'n_agree': n_agree, 'n_disagree': n_disagree,
        'sc_agree': sc_agree, 'sc_disagree': sc_disagree,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v12 CVD backtest integration test (revised)",
    )
    parser.add_argument("--delta", required=True,
                        help="Path to volume delta CSV")
    parser.add_argument("--instrument", default="MNQ",
                        help="Instrument (default: MNQ)")
    parser.add_argument("--variant", default=None,
                        help="Test specific variant (F1,F2,F3,G1,G2,FG,H)")
    parser.add_argument("--oos", action="store_true",
                        help="Run on out-of-sample (test) period")

    args = parser.parse_args()

    delta_path = Path(args.delta)
    if not delta_path.exists():
        print(f"ERROR: Delta file not found: {delta_path}")
        sys.exit(1)

    if args.oos:
        period_start, period_end = TEST_START, TEST_END
        period_name = "TEST (OOS)"
    else:
        period_start, period_end = TRAIN_START, TRAIN_END
        period_name = "TRAIN"

    print("=" * 70)
    print(f"v12 CVD INTEGRATION TEST (REVISED) -- {period_name}")
    print("=" * 70)
    print(f"  Period: {period_start.date()} to {period_end.date()}")
    print(f"  Instrument: {args.instrument}")
    print()

    # Load data
    print("Loading data...")
    delta_df = load_delta_bars(args.delta)
    ohlcv_1m = load_instrument_1min(args.instrument)

    if ohlcv_1m.index.tz is None:
        ohlcv_1m.index = ohlcv_1m.index.tz_localize('UTC')

    period_data = ohlcv_1m[(ohlcv_1m.index >= period_start) &
                            (ohlcv_1m.index < period_end)]

    if len(period_data) < 100:
        print(f"ERROR: Only {len(period_data)} bars. Need more data.")
        sys.exit(1)

    print(f"  OHLCV bars: {len(period_data):,}")

    # Align delta data with fast signals
    delta_data = align_delta_data(period_data, delta_df)
    nonzero = np.count_nonzero(delta_data['delta'])
    print(f"  Delta bars aligned: {len(delta_data['delta']):,} ({nonzero:,} non-zero)")

    # Delta stats
    d = delta_data['delta']
    d_nz = d[d != 0]
    if len(d_nz) > 0:
        print(f"  Delta stats: mean={d_nz.mean():.1f}, std={d_nz.std():.1f}, "
              f"min={d_nz.min():.0f}, max={d_nz.max():.0f}")
        print(f"  Roll-5 stats: mean={delta_data['roll_5'][d!=0].mean():.1f}, "
              f"std={delta_data['roll_5'][d!=0].std():.1f}")
        print(f"  Roll-10 stats: mean={delta_data['roll_10'][d!=0].mean():.1f}, "
              f"std={delta_data['roll_10'][d!=0].std():.1f}")

    # Prepare arrays
    arr = prepare_backtest_arrays_1min(period_data, rsi_len=V11_RSI_LEN)

    # Run baseline
    print("\nRunning v11 baseline...")
    trades_baseline = _run_baseline(arr)
    sc_baseline = score_trades(trades_baseline)
    print(f"  Baseline: {fmt_score(sc_baseline, 'v11')}")

    # Determine which variants to run
    if args.variant:
        variants_to_test = [args.variant.upper()]
    else:
        variants_to_test = ALL_VARIANT_KEYS

    results = []

    for var_key in variants_to_test:
        if var_key == 'H':
            # Conviction sizing is special
            print(f"\nRunning variant H: Conviction sizing...")
            trades_sized, n_agree, n_disagree = run_conviction_sizing(
                arr, delta_data, trades_baseline)
            h_result = evaluate_conviction(
                trades_baseline, trades_sized, n_agree, n_disagree)
            if h_result:
                results.append(h_result)
            continue

        if var_key not in VARIANTS:
            print(f"\n  WARNING: Unknown variant '{var_key}', skipping")
            continue

        desc, run_fn = VARIANTS[var_key]
        print(f"\nRunning variant {var_key}: {desc}...")
        trades_feat = run_fn(arr, delta_data)
        result = evaluate_variant(var_key, desc, trades_baseline, trades_feat)
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY -- {period_name}")
    print("=" * 70)

    # Standard variants
    std_results = [r for r in results if r['name'] != 'H']
    if std_results:
        print(f"\n  {'Variant':<32} | {'Verdict':>8} | {'Score':>5} | "
              f"{'dPF':>8} | {'p-val':>8} | {'Trades':>7}")
        print(f"  {'-'*32}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

        best = None
        for r in std_results:
            n_trades = r['sc_feat']['count']
            print(f"  {r['name']+': '+r['desc']:<32} | {r['verdict']:>8} | "
                  f"{r['passes']:>3}/7 | {r['dpf']:>+8.3f} | "
                  f"{r['p_val']:>8.4f} | {n_trades:>7}")
            if best is None or r['passes'] > best['passes']:
                best = r
            elif r['passes'] == best['passes'] and r['dpf'] > best['dpf']:
                best = r

        if best and best['verdict'] != 'REJECT':
            print(f"\n  BEST VARIANT: {best['name']} ({best['desc']})")
            print(f"    PF: {best['sc_feat']['pf']:.3f} "
                  f"(baseline {best['sc_base']['pf']:.3f})")
            if not args.oos:
                print(f"\n  Next: python3 v12_cvd_test.py --delta {args.delta} "
                      f"--variant {best['name']} --oos")
        else:
            print(f"\n  No standard variant passed on {period_name} data.")

        # OOS check
        if args.oos and best:
            print(f"\n  OOS VALIDATION:")
            feat_pf = best['sc_feat']['pf']
            base_pf = best['sc_base']['pf']
            if feat_pf < 1.0:
                print(f"    PF = {feat_pf:.3f} < 1.0 -> REJECT")
            elif feat_pf < base_pf * 0.7:
                print(f"    PF degraded >30% -> REJECT")
            else:
                print(f"    PF = {feat_pf:.3f} (baseline {base_pf:.3f}) -> PASS")

    # Conviction sizing summary
    h_results = [r for r in results if r['name'] == 'H']
    if h_results:
        h = h_results[0]
        print(f"\n  CONVICTION SIZING (H):")
        print(f"    Agree trades: {h['n_agree']} (2x), "
              f"Disagree: {h['n_disagree']} (1x)")
        if h['sc_agree'] and h['sc_disagree']:
            print(f"    Agree PF: {h['sc_agree']['pf']:.3f}, "
                  f"Disagree PF: {h['sc_disagree']['pf']:.3f}")
        print(f"    Uniform net:    ${h['sc_base']['net_dollar']:+.2f}")
        print(f"    Conviction net: ${h['sc_sized']['net_dollar']:+.2f}")


if __name__ == "__main__":
    main()
