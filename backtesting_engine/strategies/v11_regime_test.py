"""
v11 Regime Robustness Test
--------------------------
Test modifications to v11 MNQ that protect against SM whipsaw days
without destroying the edge in favorable regimes.

Modifications tested:
  A. SM threshold (filter marginal zero-crossings)
  B. Max loss stop variations (0=off, 50=current, 75, 100)
  C. SM confirmation bars (wait N bars after flip before entering)
  D. Combined best from A+B+C

Run on 6-month Databento data with monthly breakdown.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
from v10_test_common import (
    load_instrument_1min,
    compute_smart_money,
    resample_to_5min,
    map_5min_rsi_to_1min,
    score_trades,
)

# ---------------------------------------------------------------------------
# Load data + compute indicators with v11 params
# ---------------------------------------------------------------------------
def load_and_prepare():
    """Load MNQ 1-min data, compute SM(10/12/200/100) and RSI(8)."""
    df = load_instrument_1min("MNQ")
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values
    times = df.index  # DatetimeIndex

    # SM with v11 params
    sm = compute_smart_money(closes, volumes,
                             index_period=10, flow_period=12,
                             norm_period=200, ema_len=100)

    # RSI on 5-min, mapped back to 1-min (must resample first)
    df_for_5m = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_5m['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_for_5m)
    rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=8)

    print(f"Data: {len(closes)} bars, {times[0]} to {times[-1]}")
    print(f"SM range: [{sm.min():.4f}, {sm.max():.4f}]")
    print(f"RSI range: [{np.nanmin(rsi_5m_curr):.1f}, {np.nanmax(rsi_5m_curr):.1f}]")

    return opens, highs, lows, closes, sm, rsi_5m_curr, rsi_5m_prev, times


# ---------------------------------------------------------------------------
# Custom backtest with SM confirmation bars support
# ---------------------------------------------------------------------------
def run_backtest_custom(opens, highs, lows, closes, sm, times,
                        rsi_5m_curr, rsi_5m_prev,
                        rsi_buy=60, rsi_sell=40,
                        sm_threshold=0.0,
                        cooldown_bars=20,
                        max_loss_pts=50,
                        sm_confirm_bars=0,
                        dollar_per_pt=2.0):
    """
    Custom backtest matching v11 logic with added SM confirmation bars.

    sm_confirm_bars: if > 0, after SM crosses zero, wait this many bars
                     with SM staying on the same side before entering.
    """
    n = len(closes)
    trades = []

    position = 0       # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    entry_time = None
    exit_bar_idx = -9999
    long_used = False
    short_used = False

    # SM confirmation tracking
    sm_bull_since = 0   # consecutive bars SM > threshold
    sm_bear_since = 0   # consecutive bars SM < -threshold

    for i in range(2, n):
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        # Track SM confirmation counters
        if sm_prev > sm_threshold:
            sm_bull_since += 1
            sm_bear_since = 0
        elif sm_prev < -sm_threshold:
            sm_bear_since += 1
            sm_bull_since = 0
        else:
            # SM in dead zone (between -threshold and +threshold)
            sm_bull_since = 0
            sm_bear_since = 0

        # RSI cross detection (matches Pine request.security behavior)
        rsi_curr = rsi_5m_curr[i - 1] if rsi_5m_curr is not None else 50
        rsi_prev = rsi_5m_prev[i - 1] if rsi_5m_prev is not None else 50
        if np.isnan(rsi_curr) or np.isnan(rsi_prev):
            continue

        rsi_crossed_up = rsi_curr > rsi_buy and rsi_prev <= rsi_buy
        rsi_crossed_down = rsi_curr < rsi_sell and rsi_prev >= rsi_sell

        # Episode reset on SM flip
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        if sm_flipped_bull:
            long_used = False
        if sm_flipped_bear:
            short_used = False

        # SM state
        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold

        # Cooldown check
        bars_since_exit = i - exit_bar_idx
        in_cooldown = bars_since_exit < cooldown_bars

        # --- Position management ---
        if position != 0:
            # Check max loss stop (look at bar i-1 data, fill at bar i open)
            if max_loss_pts > 0:
                if position == 1 and closes[i - 1] <= entry_price - max_loss_pts:
                    pnl_pts = opens[i] - entry_price
                    trades.append({
                        "side": "long", "entry": entry_price, "exit": opens[i],
                        "pts": pnl_pts, "entry_time": entry_time,
                        "exit_time": times[i], "entry_idx": entry_idx,
                        "exit_idx": i, "bars": i - entry_idx, "result": "SL",
                    })
                    position = 0
                    exit_bar_idx = i
                    continue
                elif position == -1 and closes[i - 1] >= entry_price + max_loss_pts:
                    pnl_pts = entry_price - opens[i]
                    trades.append({
                        "side": "short", "entry": entry_price, "exit": opens[i],
                        "pts": pnl_pts, "entry_time": entry_time,
                        "exit_time": times[i], "entry_idx": entry_idx,
                        "exit_idx": i, "bars": i - entry_idx, "result": "SL",
                    })
                    position = 0
                    exit_bar_idx = i
                    continue

            # SM flip exit
            if position == 1 and sm_bear:
                pnl_pts = opens[i] - entry_price
                trades.append({
                    "side": "long", "entry": entry_price, "exit": opens[i],
                    "pts": pnl_pts, "entry_time": entry_time,
                    "exit_time": times[i], "entry_idx": entry_idx,
                    "exit_idx": i, "bars": i - entry_idx, "result": "SM_FLIP",
                })
                position = 0
                exit_bar_idx = i
            elif position == -1 and sm_bull:
                pnl_pts = entry_price - opens[i]
                trades.append({
                    "side": "short", "entry": entry_price, "exit": opens[i],
                    "pts": pnl_pts, "entry_time": entry_time,
                    "exit_time": times[i], "entry_idx": entry_idx,
                    "exit_idx": i, "bars": i - entry_idx, "result": "SM_FLIP",
                })
                position = 0
                exit_bar_idx = i

        # --- Entry logic ---
        if position == 0 and not in_cooldown:
            # SM confirmation check
            sm_confirmed_bull = sm_bull_since >= max(1, sm_confirm_bars)
            sm_confirmed_bear = sm_bear_since >= max(1, sm_confirm_bars)

            if sm_confirmed_bull and rsi_crossed_up and not long_used:
                position = 1
                entry_price = opens[i]
                entry_idx = i
                entry_time = times[i]
                long_used = True

            elif sm_confirmed_bear and rsi_crossed_down and not short_used:
                position = -1
                entry_price = opens[i]
                entry_idx = i
                entry_time = times[i]
                short_used = True

    # Close any open position at end
    if position != 0:
        pnl_pts = (closes[-1] - entry_price) if position == 1 else (entry_price - closes[-1])
        trades.append({
            "side": "long" if position == 1 else "short",
            "entry": entry_price, "exit": closes[-1],
            "pts": pnl_pts, "entry_time": entry_time,
            "exit_time": times[-1], "entry_idx": entry_idx,
            "exit_idx": len(closes) - 1, "bars": len(closes) - 1 - entry_idx,
            "result": "EOD",
        })

    return trades


# ---------------------------------------------------------------------------
# Monthly breakdown
# ---------------------------------------------------------------------------
def monthly_breakdown(trades, dollar_per_pt=2.0, commission=0.52):
    """Break trades into monthly buckets and score each."""
    by_month = {}
    for t in trades:
        et = t["exit_time"]
        month = str(et)[:7]  # "2025-08" format
        if month not in by_month:
            by_month[month] = []
        by_month[month].append(t)

    results = {}
    for month in sorted(by_month.keys()):
        mtrades = by_month[month]
        s = score_trades(mtrades, commission_per_side=commission, dollar_per_pt=dollar_per_pt)
        results[month] = s
    return results


def print_results(label, trades, dollar_per_pt=2.0):
    """Print compact results with monthly breakdown."""
    s = score_trades(trades, commission_per_side=0.52, dollar_per_pt=dollar_per_pt)
    if s is None:
        print(f"  {label:40s}  -- no trades --")
        return s

    exits = s.get("exits", {})
    sl_count = exits.get("SL", 0)
    sm_count = exits.get("SM_FLIP", 0)

    print(f"  {label:40s}  {s['count']:>4d} trades  PF {s['pf']:>6.3f}  "
          f"WR {s['win_rate']:>5.1f}%  Net ${s['net_dollar']:>+8.0f}  "
          f"DD ${s['max_dd_dollar']:>7.0f}  "
          f"SL:{sl_count} SM:{sm_count}")

    # Monthly
    monthly = monthly_breakdown(trades, dollar_per_pt=dollar_per_pt)
    months_str = []
    for month, ms in monthly.items():
        if ms:
            short_month = month[2:]  # "25-08"
            pnl = ms['net_dollar']
            color = "+" if pnl >= 0 else ""
            months_str.append(f"{short_month}:{color}${pnl:.0f}")
    print(f"    Monthly: {' | '.join(months_str)}")
    return s


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("v11 REGIME ROBUSTNESS TEST")
    print("=" * 80)

    opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times = load_and_prepare()
    print()

    # -----------------------------------------------------------------------
    # A. BASELINE (v11 current production)
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("A. BASELINE vs SM THRESHOLD")
    print("=" * 80)

    for thresh in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]:
        trades = run_backtest_custom(
            opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev,
            rsi_buy=60, rsi_sell=40, sm_threshold=thresh,
            cooldown_bars=20, max_loss_pts=50, sm_confirm_bars=0,
        )
        label = f"SM_thresh={thresh:.2f} SL=50"
        print_results(label, trades)

    # -----------------------------------------------------------------------
    # B. MAX LOSS STOP VARIATIONS
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("B. MAX LOSS STOP VARIATIONS")
    print("=" * 80)

    for sl in [0, 25, 50, 75, 100, 150]:
        trades = run_backtest_custom(
            opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev,
            rsi_buy=60, rsi_sell=40, sm_threshold=0.0,
            cooldown_bars=20, max_loss_pts=sl, sm_confirm_bars=0,
        )
        label = f"SL={sl}pt"
        print_results(label, trades)

    # -----------------------------------------------------------------------
    # C. SM CONFIRMATION BARS
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("C. SM CONFIRMATION BARS (wait N bars after SM flip)")
    print("=" * 80)

    for confirm in [0, 2, 3, 5, 8, 10]:
        trades = run_backtest_custom(
            opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev,
            rsi_buy=60, rsi_sell=40, sm_threshold=0.0,
            cooldown_bars=20, max_loss_pts=50, sm_confirm_bars=confirm,
        )
        label = f"SM_confirm={confirm} bars"
        print_results(label, trades)

    # -----------------------------------------------------------------------
    # D. COOLDOWN VARIATIONS
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("D. COOLDOWN VARIATIONS")
    print("=" * 80)

    for cd in [15, 20, 30, 40, 60]:
        trades = run_backtest_custom(
            opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev,
            rsi_buy=60, rsi_sell=40, sm_threshold=0.0,
            cooldown_bars=cd, max_loss_pts=50, sm_confirm_bars=0,
        )
        label = f"CD={cd}"
        print_results(label, trades)

    # -----------------------------------------------------------------------
    # E. RSI LEVELS
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("E. RSI LEVEL VARIATIONS")
    print("=" * 80)

    for rsi_buy, rsi_sell in [(55, 45), (60, 40), (65, 35), (70, 30)]:
        trades = run_backtest_custom(
            opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev,
            rsi_buy=rsi_buy, rsi_sell=rsi_sell, sm_threshold=0.0,
            cooldown_bars=20, max_loss_pts=50, sm_confirm_bars=0,
        )
        label = f"RSI {rsi_buy}/{rsi_sell}"
        print_results(label, trades)

    # -----------------------------------------------------------------------
    # F. COMBINED: Best from each dimension
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("F. PROMISING COMBINATIONS")
    print("=" * 80)

    combos = [
        # (label, thresh, sl, cd, confirm, rsi_buy, rsi_sell)
        ("v11 baseline",                0.0,  50, 20, 0, 60, 40),
        ("thresh=0.05 + SL=0",         0.05,  0, 20, 0, 60, 40),
        ("thresh=0.05 + SL=75",        0.05, 75, 20, 0, 60, 40),
        ("thresh=0.10 + SL=50",        0.10, 50, 20, 0, 60, 40),
        ("thresh=0.10 + SL=75",        0.10, 75, 20, 0, 60, 40),
        ("thresh=0.10 + SL=0",         0.10,  0, 20, 0, 60, 40),
        ("confirm=3 + SL=50",          0.0,  50, 20, 3, 60, 40),
        ("confirm=3 + SL=75",          0.0,  75, 20, 3, 60, 40),
        ("confirm=3 + thresh=0.05",    0.05, 50, 20, 3, 60, 40),
        ("confirm=5 + SL=0",           0.0,   0, 20, 5, 60, 40),
        ("thresh=0.05 + CD=30",        0.05, 50, 30, 0, 60, 40),
        ("thresh=0.05 + RSI 65/35",    0.05, 50, 20, 0, 65, 35),
        ("thresh=0.10 + CD=30 + SL=75",0.10, 75, 30, 0, 60, 40),
        ("confirm=3 + thresh=0.05 + SL=75", 0.05, 75, 20, 3, 60, 40),
        ("conservative: t=0.10 c=3 SL=75 CD=30", 0.10, 75, 30, 3, 60, 40),
    ]

    for label, thresh, sl, cd, confirm, rb, rs in combos:
        trades = run_backtest_custom(
            opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev,
            rsi_buy=rb, rsi_sell=rs, sm_threshold=thresh,
            cooldown_bars=cd, max_loss_pts=sl, sm_confirm_bars=confirm,
        )
        print_results(label, trades)


if __name__ == "__main__":
    main()
