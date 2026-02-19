"""
v15_hybrid.py — Hybrid Exit + Portfolio + Regime Detection
============================================================
Three tests:
  1. HYBRID: Trail + SM flip exits both active, exit on whichever first
  2. PORTFOLIO: 2 contracts per entry (1 SM flip, 1 trail), sum PnL
  3. REGIME: Can first-30-min metrics predict which exit wins?

Usage:
  python3 v15_hybrid.py                 # TRAIN only
  python3 v15_hybrid.py --oos           # OOS only
  python3 v15_hybrid.py --all           # Both + cross-period comparison
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_instrument_1min, prepare_backtest_arrays_1min,
    run_v9_baseline, score_trades, fmt_score,
    compute_et_minutes, compute_rsi,
    NY_OPEN_ET, NY_CLOSE_ET, NY_LAST_ENTRY_ET,
)

# ---------------------------------------------------------------------------
# Constants
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

# Trail configs to test in hybrid
HYBRID_CONFIGS = {
    'H_5_5':  {'trail_activate': 5,  'trail_dist': 5},
    'H_5_8':  {'trail_activate': 5,  'trail_dist': 8},
    'H_5_10': {'trail_activate': 5,  'trail_dist': 10},
    'H_10_8': {'trail_activate': 10, 'trail_dist': 8},
    'H_10_10': {'trail_activate': 10, 'trail_dist': 10},
    'H_10_15': {'trail_activate': 10, 'trail_dist': 15},
    'H_15_10': {'trail_activate': 15, 'trail_dist': 10},
    'H_15_15': {'trail_activate': 15, 'trail_dist': 15},
}


# ---------------------------------------------------------------------------
# Hybrid Engine (trail + SM flip, whichever first)
# ---------------------------------------------------------------------------

def run_hybrid(arr, trail_activate=5, trail_dist=8):
    """v11 entries with hybrid exit: trailing stop OR SM flip, first wins.

    Exit priority:
      1. EOD close
      2. Max loss 50pt stop
      3. Trailing stop (once position has been up trail_activate pts)
      4. SM flip (standard)

    Returns list of trade dicts with 'result' field indicating exit type.
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
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    max_fav = 0.0
    trail_on = False

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

        # === EXITS ===
        if trade_state != 0:
            side = "long" if trade_state == 1 else "short"
            unrealized = (closes[i - 1] - entry_price) * trade_state

            # Track max favorable excursion
            if unrealized > max_fav:
                max_fav = unrealized

            # 1. EOD
            if bar_et >= NY_CLOSE_ET:
                close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
                trade_state = 0
                exit_bar = i
                trail_on = False
                max_fav = 0.0
                continue

            # 2. Max loss stop
            if V11_MAX_LOSS > 0 and unrealized <= -V11_MAX_LOSS:
                close_trade(side, entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                trail_on = False
                max_fav = 0.0
                continue

            # 3. Trailing stop (checked BEFORE SM flip)
            if max_fav >= trail_activate:
                trail_on = True
            if trail_on:
                trail_level = max_fav - trail_dist
                if unrealized <= trail_level:
                    close_trade(side, entry_price, opens[i], entry_idx, i,
                                "TRAIL")
                    trade_state = 0
                    exit_bar = i
                    trail_on = False
                    max_fav = 0.0
                    continue

            # 4. SM flip exit (standard)
            if trade_state == 1 and sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i,
                            "SM_FLIP")
                trade_state = 0
                exit_bar = i
                trail_on = False
                max_fav = 0.0

            elif trade_state == -1 and sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i,
                            "SM_FLIP")
                trade_state = 0
                exit_bar = i
                trail_on = False
                max_fav = 0.0

        # === ENTRIES ===
        if trade_state == 0:
            in_session = NY_OPEN_ET <= bar_et <= NY_LAST_ENTRY_ET
            cd_ok = (i - exit_bar) >= V11_COOLDOWN

            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    max_fav = 0.0
                    trail_on = False

                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    max_fav = 0.0
                    trail_on = False

    return trades


# ---------------------------------------------------------------------------
# Pure trailing stop engine (for portfolio leg B)
# ---------------------------------------------------------------------------

def run_trail_only(arr, trail_activate=5, trail_dist=8):
    """v11 entries, trailing stop exit only (no SM flip). 50pt backstop."""
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
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    max_fav = 0.0
    trail_on = False

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

        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        if trade_state != 0:
            side = "long" if trade_state == 1 else "short"
            unrealized = (closes[i - 1] - entry_price) * trade_state
            if unrealized > max_fav:
                max_fav = unrealized

            if bar_et >= NY_CLOSE_ET:
                close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
                trade_state = 0
                exit_bar = i
                trail_on = False
                max_fav = 0.0
                continue

            if V11_MAX_LOSS > 0 and unrealized <= -V11_MAX_LOSS:
                close_trade(side, entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                trail_on = False
                max_fav = 0.0
                continue

            if max_fav >= trail_activate:
                trail_on = True
            if trail_on:
                trail_level = max_fav - trail_dist
                if unrealized <= trail_level:
                    close_trade(side, entry_price, opens[i], entry_idx, i,
                                "TRAIL")
                    trade_state = 0
                    exit_bar = i
                    trail_on = False
                    max_fav = 0.0
                    continue

        if trade_state == 0:
            in_session = NY_OPEN_ET <= bar_et <= NY_LAST_ENTRY_ET
            cd_ok = (i - exit_bar) >= V11_COOLDOWN

            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    max_fav = 0.0
                    trail_on = False
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    max_fav = 0.0
                    trail_on = False

    return trades


# ---------------------------------------------------------------------------
# Portfolio Analysis
# ---------------------------------------------------------------------------

def analyze_portfolio(trades_sm, trades_trail, commission=0.52, dollar_per_pt=2.0):
    """Combine SM flip + trail trades into 2-contract portfolio.

    Matches trades by entry_time. Portfolio PnL = sm_pts + trail_pts.
    Returns portfolio trade list and per-contract metrics.
    """
    trail_by_entry = {t['entry_time']: t for t in trades_trail}
    comm_pts = (commission * 2) / dollar_per_pt  # per trade per contract

    portfolio_trades = []
    sm_only = 0
    trail_only = 0
    both = 0

    for sm_t in trades_sm:
        tr_t = trail_by_entry.get(sm_t['entry_time'])
        if tr_t is None:
            sm_only += 1
            continue

        both += 1
        # 2-contract portfolio: 1 SM flip + 1 trail
        sm_net = sm_t['pts'] - comm_pts
        tr_net = tr_t['pts'] - comm_pts
        combo_net = sm_net + tr_net  # total for 2 contracts
        per_contract = combo_net / 2  # per-contract average

        portfolio_trades.append({
            'entry_time': sm_t['entry_time'],
            'side': sm_t['side'],
            'sm_pts': sm_t['pts'],
            'trail_pts': tr_t['pts'],
            'sm_exit': sm_t['result'],
            'trail_exit': tr_t['result'],
            'sm_net': sm_net,
            'trail_net': tr_net,
            'combo_net': combo_net,
            'per_contract': per_contract,
            'pts': per_contract + comm_pts,  # for score_trades compat
            'bars': max(sm_t['bars'], tr_t['bars']),
            'result': f"{sm_t['result']}+{tr_t['result']}",
        })

    # Check for trail trades without SM match
    sm_entries = {t['entry_time'] for t in trades_sm}
    for tr_t in trades_trail:
        if tr_t['entry_time'] not in sm_entries:
            trail_only += 1

    return portfolio_trades, both, sm_only, trail_only


def score_portfolio(portfolio_trades, commission=0.52, dollar_per_pt=2.0):
    """Score portfolio trades (2-contract per entry)."""
    if not portfolio_trades:
        return None

    # Use combo_net (already commission-adjusted for 2 contracts)
    combo_nets = np.array([t['combo_net'] for t in portfolio_trades])
    n = len(combo_nets)

    # Dollar PnL (2 contracts * dollar_per_pt)
    combo_dollars = combo_nets * dollar_per_pt
    total_dollar = combo_dollars.sum()

    wins = combo_nets[combo_nets > 0]
    losses = combo_nets[combo_nets <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100

    cum = np.cumsum(combo_dollars)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()

    sharpe = 0.0
    if len(combo_dollars) > 1 and np.std(combo_dollars) > 0:
        sharpe = np.mean(combo_dollars) / np.std(combo_dollars) * np.sqrt(252)

    return {
        'count': n, 'pf': round(pf, 3), 'win_rate': round(wr, 1),
        'net_dollar': round(total_dollar, 2),
        'max_dd_dollar': round(mdd, 2), 'sharpe': round(sharpe, 3),
    }


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------

def compute_daily_metrics(arr, period_data):
    """Compute per-day metrics for regime detection.

    Returns dict: {date_str: {metric: value, ...}}
    """
    times = arr['times']
    opens = arr['opens']
    highs = arr['highs']
    lows = arr['lows']
    closes = arr['closes']
    sm = arr['sm']
    et_mins = compute_et_minutes(times)
    n = len(times)

    # Group bars by trading date (session = 10:00 ET to 16:00 ET)
    day_bars = defaultdict(list)
    for i in range(n):
        if NY_OPEN_ET <= et_mins[i] <= NY_CLOSE_ET:
            # Convert unix time to date
            dt = pd.Timestamp(times[i], unit='s', tz='UTC')
            day_key = dt.strftime('%Y-%m-%d')
            day_bars[day_key].append(i)

    metrics = {}
    sorted_days = sorted(day_bars.keys())

    for d_idx, day in enumerate(sorted_days):
        bars = day_bars[day]
        if len(bars) < 30:
            continue

        # Session range
        day_high = max(highs[b] for b in bars)
        day_low = min(lows[b] for b in bars)
        day_range = day_high - day_low

        # First 30-min range (first 30 bars after session open)
        first30 = bars[:30]
        f30_high = max(highs[b] for b in first30)
        f30_low = min(lows[b] for b in first30)
        f30_range = f30_high - f30_low

        # SM flip count in first 60 min
        first60 = bars[:60] if len(bars) >= 60 else bars
        sm_flips = 0
        for j in range(1, len(first60)):
            b = first60[j]
            b_prev = first60[j - 1]
            if (sm[b] > 0 and sm[b_prev] <= 0) or (sm[b] < 0 and sm[b_prev] >= 0):
                sm_flips += 1

        # Gap from prior day close
        gap = 0.0
        if d_idx > 0:
            prev_day = sorted_days[d_idx - 1]
            prev_bars = day_bars[prev_day]
            if prev_bars:
                prev_close = closes[prev_bars[-1]]
                today_open = opens[bars[0]]
                gap = abs(today_open - prev_close)

        # Prior day range
        prior_range = 0.0
        if d_idx > 0:
            prev_day = sorted_days[d_idx - 1]
            prev_bars = day_bars[prev_day]
            if prev_bars:
                prior_range = max(highs[b] for b in prev_bars) - min(
                    lows[b] for b in prev_bars)

        # First 30-min range as % of prior day range (narrow = choppy)
        f30_pct = (f30_range / prior_range * 100) if prior_range > 0 else 50.0

        metrics[day] = {
            'day_range': day_range,
            'f30_range': f30_range,
            'f30_pct': f30_pct,
            'sm_flips_60': sm_flips,
            'gap': gap,
            'prior_range': prior_range,
        }

    return metrics


def analyze_regime(trades_sm, trades_trail, daily_metrics, times_arr):
    """Split trades by regime metrics, compare SM flip vs trail performance."""

    # Map trades to dates
    def trade_to_date(t):
        dt = pd.Timestamp(t['entry_time'], unit='s', tz='UTC')
        return dt.strftime('%Y-%m-%d')

    # Build matched trade pairs
    trail_by_entry = {t['entry_time']: t for t in trades_trail}
    pairs = []
    for sm_t in trades_sm:
        tr_t = trail_by_entry.get(sm_t['entry_time'])
        if tr_t is None:
            continue
        day = trade_to_date(sm_t)
        if day not in daily_metrics:
            continue
        pairs.append({
            'day': day,
            'sm_pts': sm_t['pts'],
            'trail_pts': tr_t['pts'],
            'sm_better': sm_t['pts'] > tr_t['pts'],
            **daily_metrics[day],
        })

    if not pairs:
        print("  No matched trade pairs with metrics.")
        return

    n = len(pairs)
    print(f"\n  Matched trade pairs with daily metrics: {n}")

    # For each metric, split at median and compare
    metric_names = [
        ('f30_range', 'First 30-min range', 'wide=trend, narrow=chop'),
        ('sm_flips_60', 'SM flips (first 60min)', 'many=chop, few=trend'),
        ('gap', 'Overnight gap', 'large=consolidate, small=continue'),
        ('prior_range', 'Prior day range', 'wide=volatile, narrow=quiet'),
        ('f30_pct', 'First-30 / prior range %', 'high=trend, low=chop'),
    ]

    print(f"\n  {'Metric':<28} | {'Split':>6} | "
          f"{'SM PF':>7} {'TR PF':>7} {'Winner':>8} | "
          f"{'SM PF':>7} {'TR PF':>7} {'Winner':>8}")
    print(f"  {'':28} | {'':>6} | "
          f"{'--- LOW ---':^24} | {'--- HIGH ---':^24}")
    print(f"  {'-'*28}-+-{'-'*6}-+-{'-'*24}-+-{'-'*24}")

    best_metric = None
    best_spread = 0

    for mkey, mname, mdesc in metric_names:
        vals = [p[mkey] for p in pairs]
        median = np.median(vals)

        low_pairs = [p for p in pairs if p[mkey] <= median]
        high_pairs = [p for p in pairs if p[mkey] > median]

        if not low_pairs or not high_pairs:
            continue

        # PF for SM and trail in each bucket
        def bucket_pf(bucket, key):
            pts = [p[key] for p in bucket]
            wins = sum(p for p in pts if p > 0)
            losses = abs(sum(p for p in pts if p <= 0))
            return wins / losses if losses > 0 else 999.0

        lo_sm = bucket_pf(low_pairs, 'sm_pts')
        lo_tr = bucket_pf(low_pairs, 'trail_pts')
        hi_sm = bucket_pf(high_pairs, 'sm_pts')
        hi_tr = bucket_pf(high_pairs, 'trail_pts')

        lo_winner = "SM" if lo_sm > lo_tr else "TRAIL"
        hi_winner = "SM" if hi_sm > hi_tr else "TRAIL"

        # Best metric = one where SM wins one bucket and trail wins the other
        if lo_winner != hi_winner:
            spread = abs((lo_sm - lo_tr) + (hi_sm - hi_tr))
            if spread > best_spread:
                best_spread = spread
                best_metric = (mkey, mname, median, lo_winner, hi_winner)

        print(f"  {mname:<28} | {median:>6.1f} | "
              f"{lo_sm:>7.3f} {lo_tr:>7.3f} {lo_winner:>8} | "
              f"{hi_sm:>7.3f} {hi_tr:>7.3f} {hi_winner:>8}")

    if best_metric:
        mkey, mname, med, lo_w, hi_w = best_metric
        print(f"\n  Best regime discriminator: {mname}")
        print(f"    Threshold: {med:.1f}")
        print(f"    Below: use {lo_w} exits")
        print(f"    Above: use {hi_w} exits")

        # Calculate "oracle" strategy: always pick the right exit
        oracle_pts = []
        regime_pts = []
        for p in pairs:
            # Oracle: always pick whichever was better for this trade
            oracle_pts.append(max(p['sm_pts'], p['trail_pts']))
            # Regime: pick based on metric
            if p[mkey] <= med:
                regime_pts.append(p['sm_pts'] if lo_w == "SM" else p['trail_pts'])
            else:
                regime_pts.append(p['sm_pts'] if hi_w == "SM" else p['trail_pts'])

        oracle_arr = np.array(oracle_pts)
        regime_arr = np.array(regime_pts)
        sm_arr = np.array([p['sm_pts'] for p in pairs])
        tr_arr = np.array([p['trail_pts'] for p in pairs])

        def quick_pf(arr):
            w = arr[arr > 0].sum()
            l = abs(arr[arr <= 0].sum())
            return w / l if l > 0 else 999.0

        print(f"\n  Strategy comparison (pts):")
        print(f"    SM only:     PF {quick_pf(sm_arr):.3f}, "
              f"Net {sm_arr.sum():+.1f} pts")
        print(f"    Trail only:  PF {quick_pf(tr_arr):.3f}, "
              f"Net {tr_arr.sum():+.1f} pts")
        print(f"    Regime pick: PF {quick_pf(regime_arr):.3f}, "
              f"Net {regime_arr.sum():+.1f} pts")
        print(f"    Oracle:      PF {quick_pf(oracle_arr):.3f}, "
              f"Net {oracle_arr.sum():+.1f} pts")

    return best_metric


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_period(ohlcv_1m, period_start, period_end, period_name):
    """Run hybrid, portfolio, and regime analysis on one period."""
    period_data = ohlcv_1m[(ohlcv_1m.index >= period_start) &
                            (ohlcv_1m.index < period_end)]
    if len(period_data) < 100:
        print(f"ERROR: Only {len(period_data)} bars")
        return None

    print(f"\n{'='*75}")
    print(f"  {period_name}: {period_start.date()} to {period_end.date()}")
    print(f"  {len(period_data):,} bars")
    print(f"{'='*75}")

    arr = prepare_backtest_arrays_1min(period_data, rsi_len=V11_RSI_LEN)

    # --- 1. Baseline (SM flip exits) ---
    trades_sm = run_v9_baseline(
        arr, rsi_len=V11_RSI_LEN, rsi_buy=V11_RSI_BUY,
        rsi_sell=V11_RSI_SELL, cooldown=V11_COOLDOWN,
        max_loss_pts=V11_MAX_LOSS,
    )
    sc_sm = score_trades(trades_sm)
    print(f"\n  1. BASELINE (SM flip): {fmt_score(sc_sm, 'SM')}")
    if sc_sm.get('exits'):
        print(f"     Exits: {' '.join(f'{k}:{v}' for k,v in sorted(sc_sm['exits'].items()))}")

    # --- 2. Pure trail (TR_5_8) ---
    trades_trail = run_trail_only(arr, trail_activate=5, trail_dist=8)
    sc_trail = score_trades(trades_trail)
    print(f"\n  2. PURE TRAIL (5/8): {fmt_score(sc_trail, 'TR')}")
    if sc_trail.get('exits'):
        print(f"     Exits: {' '.join(f'{k}:{v}' for k,v in sorted(sc_trail['exits'].items()))}")

    # --- 3. Hybrid sweep ---
    print(f"\n  3. HYBRID (trail + SM flip, first wins):")
    print(f"     {'Config':<10} | {'Trades':>6} {'PF':>7} {'dPF':>7} {'WR':>6} "
          f"{'Net$':>9} {'DD$':>8} {'Sharpe':>7} | Exits")
    print(f"     {'-'*10}-+-{'-'*63}-+-------")

    hybrid_results = {}
    for key, cfg in sorted(HYBRID_CONFIGS.items()):
        trades_h = run_hybrid(arr, cfg['trail_activate'], cfg['trail_dist'])
        sc_h = score_trades(trades_h)
        dpf = sc_h['pf'] - sc_sm['pf'] if sc_h else 0
        exits_str = ' '.join(f"{k}:{v}" for k, v in
                             sorted(sc_h['exits'].items())) if sc_h else ''
        if sc_h:
            print(f"     {key:<10} | {sc_h['count']:>6} {sc_h['pf']:>7.3f} "
                  f"{dpf:>+7.3f} {sc_h['win_rate']:>5.1f}% "
                  f"${sc_h['net_dollar']:>+8.2f} ${sc_h['max_dd_dollar']:>7.2f} "
                  f"{sc_h['sharpe']:>7.3f} | {exits_str}")
        hybrid_results[key] = {
            'sc': sc_h, 'trades': trades_h, 'dpf': dpf,
        }

    # --- 4. Portfolio (2 contracts: 1 SM flip + 1 trail) ---
    print(f"\n  4. PORTFOLIO (2 contracts: 1 SM flip + 1 TR_5_8):")
    ptrades, n_both, n_sm_only, n_trail_only = analyze_portfolio(
        trades_sm, trades_trail)
    sc_port = score_portfolio(ptrades)
    if sc_port:
        print(f"     Matched entries: {n_both}, SM-only: {n_sm_only}, "
              f"Trail-only: {n_trail_only}")
        print(f"     Portfolio (2 contracts): {sc_port['count']} entries, "
              f"PF {sc_port['pf']}, WR {sc_port['win_rate']}%, "
              f"Net ${sc_port['net_dollar']:+.2f}, "
              f"MaxDD ${sc_port['max_dd_dollar']:.2f}")

        # Compare to 2x SM flip baseline
        sm_2x_net = sc_sm['net_dollar'] * 2
        port_vs_2x = sc_port['net_dollar'] - sm_2x_net
        print(f"     vs 2x SM baseline: ${sm_2x_net:+.2f} -> "
              f"portfolio ${sc_port['net_dollar']:+.2f} "
              f"(diff ${port_vs_2x:+.2f})")

        # Show how many trades trail leg won
        trail_wins = sum(1 for t in ptrades if t['trail_pts'] > t['sm_pts'])
        sm_wins = sum(1 for t in ptrades if t['sm_pts'] > t['trail_pts'])
        ties = n_both - trail_wins - sm_wins
        print(f"     Per-trade winner: SM={sm_wins}, Trail={trail_wins}, "
              f"Tie={ties}")

    # --- 5. Regime detection ---
    print(f"\n  5. REGIME DETECTION:")
    daily_metrics = compute_daily_metrics(arr, period_data)
    print(f"     Days with metrics: {len(daily_metrics)}")
    regime_result = analyze_regime(trades_sm, trades_trail, daily_metrics,
                                   arr['times'])

    return {
        'sc_sm': sc_sm, 'sc_trail': sc_trail, 'sc_port': sc_port,
        'hybrid_results': hybrid_results, 'regime': regime_result,
        'trades_sm': trades_sm, 'trades_trail': trades_trail,
    }


def main():
    parser = argparse.ArgumentParser(
        description="v15 hybrid exit + portfolio + regime detection")
    parser.add_argument("--instrument", default="MNQ")
    parser.add_argument("--oos", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    print("=" * 75)
    print("v15 HYBRID EXIT / PORTFOLIO / REGIME DETECTION")
    print("=" * 75)

    ohlcv_1m = load_instrument_1min(args.instrument)
    if ohlcv_1m.index.tz is None:
        ohlcv_1m.index = ohlcv_1m.index.tz_localize('UTC')
    print(f"  OHLCV: {len(ohlcv_1m):,} bars")

    if args.oos:
        run_period(ohlcv_1m, TEST_START, TEST_END, "TEST (OOS)")
    elif args.all:
        train = run_period(ohlcv_1m, TRAIN_START, TRAIN_END, "TRAIN")
        test = run_period(ohlcv_1m, TEST_START, TEST_END, "TEST (OOS)")

        # Cross-period summary
        if train and test:
            print(f"\n{'='*75}")
            print("CROSS-PERIOD SUMMARY")
            print(f"{'='*75}")

            print(f"\n  {'Strategy':<30} | {'TRAIN PF':>9} {'TRAIN Net':>11} | "
                  f"{'OOS PF':>9} {'OOS Net':>11} | {'Combined':>11}")
            print(f"  {'-'*30}-+-{'-'*21}-+-{'-'*21}-+-{'-'*11}")

            strategies = [
                ('SM flip (baseline)', train['sc_sm'], test['sc_sm'], 1),
                ('Trail only (TR_5_8)', train['sc_trail'], test['sc_trail'], 1),
            ]

            # Add best hybrid
            best_hybrid_key = None
            best_hybrid_combined = -9999
            for key in HYBRID_CONFIGS:
                t_sc = train['hybrid_results'][key]['sc']
                o_sc = test['hybrid_results'][key]['sc']
                if t_sc and o_sc:
                    combined = t_sc['net_dollar'] + o_sc['net_dollar']
                    if combined > best_hybrid_combined:
                        best_hybrid_combined = combined
                        best_hybrid_key = key

            if best_hybrid_key:
                bh_train = train['hybrid_results'][best_hybrid_key]['sc']
                bh_test = test['hybrid_results'][best_hybrid_key]['sc']
                strategies.append(
                    (f'Hybrid ({best_hybrid_key})', bh_train, bh_test, 1))

            # Portfolio (2 contracts — divide by 2 for per-contract comparison)
            if train['sc_port'] and test['sc_port']:
                strategies.append(
                    ('Portfolio (2 ct)', train['sc_port'], test['sc_port'], 1))

            for name, t_sc, o_sc, mult in strategies:
                if t_sc and o_sc:
                    combined = t_sc['net_dollar'] + o_sc['net_dollar']
                    print(f"  {name:<30} | {t_sc['pf']:>9.3f} "
                          f"${t_sc['net_dollar']:>+10.2f} | "
                          f"{o_sc['pf']:>9.3f} "
                          f"${o_sc['net_dollar']:>+10.2f} | "
                          f"${combined:>+10.2f}")

            # Show ALL hybrid configs
            print(f"\n  All hybrid configs (TRAIN PF / OOS PF / Combined Net):")
            for key in sorted(HYBRID_CONFIGS.keys()):
                t_sc = train['hybrid_results'][key]['sc']
                o_sc = test['hybrid_results'][key]['sc']
                if t_sc and o_sc:
                    combined = t_sc['net_dollar'] + o_sc['net_dollar']
                    print(f"    {key:<10}: TRAIN PF {t_sc['pf']:>6.3f}, "
                          f"OOS PF {o_sc['pf']:>6.3f}, "
                          f"Combined ${combined:>+9.2f}")
    else:
        run_period(ohlcv_1m, TRAIN_START, TRAIN_END, "TRAIN")

    print("\nDone!")


if __name__ == "__main__":
    main()
