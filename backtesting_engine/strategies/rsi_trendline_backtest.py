"""
RSI Trendline Breakout Backtest
================================
Standalone entry signal based on RSI trendline breaks (from CAE Auto-Trendlines).
NOT an SM+RSI strategy — entries come purely from RSI pivot trendline breakouts.

Algorithm:
  1. Compute Wilder's RSI on 1-min closes
  2. Detect RSI pivot highs/lows (left/right lookback)
  3. Build descending peak trendlines (long setups) and ascending trough trendlines (short setups)
  4. LONG when RSI crosses above a descending peak trendline
  5. SHORT when RSI crosses below an ascending trough trendline
  6. Exits: TP / SL / EOD (same framework as existing strategies)

Usage:
    python3 rsi_trendline_backtest.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo
from itertools import product
import time

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ET_TZ = ZoneInfo("America/New_York")

COMMISSION_PER_SIDE = 0.52      # MNQ
DOLLAR_PER_PT = 2.0             # MNQ
ROUND_TRIP_COMM = 2 * COMMISSION_PER_SIDE

NY_OPEN_ET = 9 * 60 + 30       # 09:30 ET
NY_CLOSE_ET = 16 * 60           # 16:00 ET
DEFAULT_ENTRY_END_ET = 15 * 60 + 45  # 15:45 ET

# ---------------------------------------------------------------------------
# Data Loading (matches v10_test_common pattern)
# ---------------------------------------------------------------------------
def load_mnq_data() -> pd.DataFrame:
    """Load and concatenate all Databento MNQ 1-min files."""
    files = sorted(DATA_DIR.glob("databento_MNQ_1min_*.csv"))
    if not files:
        raise FileNotFoundError(f"No Databento MNQ 1-min files in {DATA_DIR}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        result = pd.DataFrame()
        result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
        result['Open'] = pd.to_numeric(df['open'], errors='coerce')
        result['High'] = pd.to_numeric(df['high'], errors='coerce')
        result['Low'] = pd.to_numeric(df['low'], errors='coerce')
        result['Close'] = pd.to_numeric(df['close'], errors='coerce')
        result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        result = result.set_index('Time')
        dfs.append(result)

    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    print(f"Loaded MNQ: {len(combined)} bars from {combined.index[0]} to {combined.index[-1]}")
    return combined


def compute_et_minutes(times):
    """Convert UTC timestamps to ET minutes from midnight (DST-aware)."""
    idx = pd.DatetimeIndex(times)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    et = idx.tz_convert(_ET_TZ)
    return (et.hour * 60 + et.minute).values.astype(np.int32)


# ---------------------------------------------------------------------------
# RSI (Wilder's -- matches v10_test_common.compute_rsi)
# ---------------------------------------------------------------------------
def compute_rsi(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI on a price array."""
    n = len(arr)
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.zeros(n)
    al = np.zeros(n)
    if n > period:
        ag[period] = np.mean(gain[1:period + 1])
        al[period] = np.mean(loss[1:period + 1])
        for i in range(period + 1, n):
            ag[i] = (ag[i - 1] * (period - 1) + gain[i]) / period
            al[i] = (al[i - 1] * (period - 1) + loss[i]) / period
    rs = np.where(al > 0, ag / al, 100.0)
    r = 100.0 - 100.0 / (1.0 + rs)
    r[:period] = 50.0
    return r


# ---------------------------------------------------------------------------
# Pivot Detection
# ---------------------------------------------------------------------------
def detect_pivots(rsi: np.ndarray, lb_left: int, lb_right: int, min_spacing: int):
    """Detect RSI pivot highs and lows.

    Returns:
        peak_bars:   list of bar indices where peaks occur
        peak_vals:   list of RSI values at peaks
        trough_bars: list of bar indices where troughs occur
        trough_vals: list of RSI values at troughs
    """
    n = len(rsi)
    peak_bars, peak_vals = [], []
    trough_bars, trough_vals = [], []

    for i in range(lb_left, n - lb_right):
        val = rsi[i]

        # Check peak: val is strictly higher than all neighbours
        is_peak = True
        for j in range(i - lb_left, i):
            if rsi[j] >= val:
                is_peak = False
                break
        if is_peak:
            for j in range(i + 1, i + lb_right + 1):
                if rsi[j] >= val:
                    is_peak = False
                    break

        if is_peak:
            # Min spacing check
            if peak_bars and (i - peak_bars[-1]) < min_spacing:
                is_peak = False
            if is_peak:
                peak_bars.append(i)
                peak_vals.append(val)
                # Keep buffer at 20
                if len(peak_bars) > 20:
                    peak_bars.pop(0)
                    peak_vals.pop(0)

        # Check trough: val is strictly lower than all neighbours
        is_trough = True
        for j in range(i - lb_left, i):
            if rsi[j] <= val:
                is_trough = False
                break
        if is_trough:
            for j in range(i + 1, i + lb_right + 1):
                if rsi[j] <= val:
                    is_trough = False
                    break

        if is_trough:
            if trough_bars and (i - trough_bars[-1]) < min_spacing:
                is_trough = False
            if is_trough:
                trough_bars.append(i)
                trough_vals.append(val)
                if len(trough_bars) > 20:
                    trough_bars.pop(0)
                    trough_vals.pop(0)

    return peak_bars, peak_vals, trough_bars, trough_vals


# ---------------------------------------------------------------------------
# Entry Signal Generation
# ---------------------------------------------------------------------------
def generate_signals(rsi: np.ndarray, lb_left: int = 10, lb_right: int = 3,
                     min_spacing: int = 10, piv_lookback: int = 5,
                     max_age: int = 2000, broken_max: int = 1000,
                     break_thresh: float = 0.0):
    """Generate long/short signals from RSI trendline breakouts.

    Processes bar-by-bar (like the Pine indicator) to maintain trendline state.

    Returns:
        long_signal:  bool array, True on bars where a bullish break occurs
        short_signal: bool array, True on bars where a bearish break occurs
    """
    n = len(rsi)
    long_signal = np.zeros(n, dtype=bool)
    short_signal = np.zeros(n, dtype=bool)
    grace = min_spacing + 2 * lb_right

    # Rolling pivot storage
    peak_bars_list, peak_vals_list = [], []
    trough_bars_list, trough_vals_list = [], []

    # Active trendlines: lists of (x1, y1, slope, broken_bar)
    # broken_bar == 0 means active, >0 means broken at that bar
    long_tls = []   # Descending peak trendlines (long setups)
    short_tls = []   # Ascending trough trendlines (short setups)

    for i in range(lb_left, n):
        # --- Detect pivot at bar (i - lb_right), confirmed at bar i ---
        piv_bar = i - lb_right
        if piv_bar < lb_left:
            pass  # Not enough left bars
        else:
            val = rsi[piv_bar]

            # Check peak
            is_peak = True
            for j in range(piv_bar - lb_left, piv_bar):
                if rsi[j] >= val:
                    is_peak = False
                    break
            if is_peak:
                for j in range(piv_bar + 1, min(piv_bar + lb_right + 1, n)):
                    if rsi[j] >= val:
                        is_peak = False
                        break

            if is_peak:
                spacing_ok = (not peak_bars_list or
                              (piv_bar - peak_bars_list[-1]) >= min_spacing)
                if spacing_ok:
                    peak_bars_list.append(piv_bar)
                    peak_vals_list.append(val)
                    if len(peak_bars_list) > 20:
                        peak_bars_list.pop(0)
                        peak_vals_list.pop(0)

                    # Build descending peak trendlines (long setup)
                    psz = len(peak_bars_list)
                    if psz >= 2:
                        pstart = max(0, psz - 1 - piv_lookback)
                        for pk in range(psz - 2, pstart - 1, -1):
                            prev_bar = peak_bars_list[pk]
                            prev_val = peak_vals_list[pk]
                            if val < prev_val:  # Descending
                                dx = max(piv_bar - prev_bar, 1)
                                slope = (val - prev_val) / dx
                                long_tls.append([prev_bar, prev_val, slope, 0])

            # Check trough
            is_trough = True
            for j in range(piv_bar - lb_left, piv_bar):
                if rsi[j] <= val:
                    is_trough = False
                    break
            if is_trough:
                for j in range(piv_bar + 1, min(piv_bar + lb_right + 1, n)):
                    if rsi[j] <= val:
                        is_trough = False
                        break

            if is_trough:
                spacing_ok = (not trough_bars_list or
                              (piv_bar - trough_bars_list[-1]) >= min_spacing)
                if spacing_ok:
                    trough_bars_list.append(piv_bar)
                    trough_vals_list.append(val)
                    if len(trough_bars_list) > 20:
                        trough_bars_list.pop(0)
                        trough_vals_list.pop(0)

                    # Build ascending trough trendlines (short setup)
                    tsz = len(trough_bars_list)
                    if tsz >= 2:
                        tstart = max(0, tsz - 1 - piv_lookback)
                        for tk in range(tsz - 2, tstart - 1, -1):
                            prev_bar = trough_bars_list[tk]
                            prev_val = trough_vals_list[tk]
                            if val > prev_val:  # Ascending
                                dx = max(piv_bar - prev_bar, 1)
                                slope = (val - prev_val) / dx
                                short_tls.append([prev_bar, prev_val, slope, 0])

        # --- Prune and check breakouts on long trendlines (descending peaks) ---
        bull_break = False
        keep = []
        for tl in long_tls:
            x1, y1, slope, broken_bar = tl
            proj = y1 + slope * (i - x1)

            # Pruning conditions
            if broken_bar == 0 and (i - x1) > max_age:
                continue
            if broken_bar > 0 and (i - broken_bar) > broken_max:
                continue
            if broken_bar == 0 and (proj > 105 or proj < -5):
                continue

            # Check for bullish break (RSI above descending line)
            if broken_bar == 0:
                if (i - x1) > grace and rsi[i] > proj + break_thresh:
                    tl[3] = i  # Mark broken
                    bull_break = True
            keep.append(tl)
        long_tls = keep

        # --- Prune and check breakouts on short trendlines (ascending troughs) ---
        bear_break = False
        keep = []
        for tl in short_tls:
            x1, y1, slope, broken_bar = tl
            proj = y1 + slope * (i - x1)

            if broken_bar == 0 and (i - x1) > max_age:
                continue
            if broken_bar > 0 and (i - broken_bar) > broken_max:
                continue
            if broken_bar == 0 and (proj > 105 or proj < -5):
                continue

            # Check for bearish break (RSI below ascending line)
            if broken_bar == 0:
                if (i - x1) > grace and rsi[i] < proj - break_thresh:
                    tl[3] = i
                    bear_break = True
            keep.append(tl)
        short_tls = keep

        if bull_break:
            long_signal[i] = True
        if bear_break:
            short_signal[i] = True

    return long_signal, short_signal


# ---------------------------------------------------------------------------
# Backtest Loop (TP/SL exit -- matches generate_session.py pattern)
# ---------------------------------------------------------------------------
def run_backtest(opens, closes, times, long_signal, short_signal,
                 cooldown_bars, max_loss_pts, tp_pts,
                 entry_start_et=NY_OPEN_ET, entry_end_et=DEFAULT_ENTRY_END_ET,
                 eod_minutes_et=NY_CLOSE_ET):
    """Run backtest with TP/SL/EOD exits.

    Entry: signal on bar[i-1], fill at bar[i] open.
    SL/TP: check prev bar close, fill at next bar open.
    EOD: fill at bar close.
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        # EOD close
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # Exits for open positions
        if trade_state == 1:
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue
            if tp_pts > 0 and closes[i - 1] >= entry_price + tp_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

        elif trade_state == -1:
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue
            if tp_pts > 0 and closes[i - 1] <= entry_price - tp_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

        # Entries: signal on bar[i-1], fill at bar[i] open
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = entry_start_et <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                if long_signal[i - 1]:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                elif short_signal[i - 1]:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i

    return trades


# ---------------------------------------------------------------------------
# Runner Backtest (Partial Exit: 2 contracts, TP1 scalp + TP2 runner)
# ---------------------------------------------------------------------------
def run_backtest_runner(opens, closes, times, long_signal, short_signal,
                        cooldown_bars, max_loss_pts, tp1_pts, tp2_pts,
                        entry_start_et=NY_OPEN_ET, entry_end_et=DEFAULT_ENTRY_END_ET,
                        eod_minutes_et=NY_CLOSE_ET):
    """Runner mode: 2 contracts, leg1 exits at TP1, leg2 SL->BE after TP1.

    Returns trades with qty field (1 or 2) for P&L calculation.
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    leg1_exited = False  # Has the scalp leg been closed?

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result, qty=1):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result, "qty": qty,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        # EOD close -- close all remaining legs
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            qty = 1 if leg1_exited else 2
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD", qty)
            trade_state = 0
            exit_bar = i
            continue

        if trade_state == 1:
            # Determine effective SL for runner (BE after TP1)
            runner_sl = entry_price if leg1_exited else entry_price - max_loss_pts

            # SL check
            if not leg1_exited:
                # Both legs open: SL hits both
                if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "SL", 2)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                # Runner only: SL at BE (entry price)
                if closes[i - 1] <= entry_price:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "BE", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

            # TP1: close scalp leg
            if not leg1_exited and tp1_pts > 0 and closes[i - 1] >= entry_price + tp1_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP1", 1)
                leg1_exited = True

            # TP2: close runner leg
            if leg1_exited and tp2_pts > 0 and closes[i - 1] >= entry_price + tp2_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP2", 1)
                trade_state = 0
                exit_bar = i
                continue

        elif trade_state == -1:
            if not leg1_exited:
                if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "SL", 2)
                    trade_state = 0
                    exit_bar = i
                    continue
            else:
                if closes[i - 1] >= entry_price:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "BE", 1)
                    trade_state = 0
                    exit_bar = i
                    continue

            if not leg1_exited and tp1_pts > 0 and closes[i - 1] <= entry_price - tp1_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP1", 1)
                leg1_exited = True

            if leg1_exited and tp2_pts > 0 and closes[i - 1] <= entry_price - tp2_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP2", 1)
                trade_state = 0
                exit_bar = i
                continue

        # Entries
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = entry_start_et <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                if long_signal[i - 1]:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    leg1_exited = False
                elif short_signal[i - 1]:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    leg1_exited = False

    return trades


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_trades(trades, commission_per_side=COMMISSION_PER_SIDE,
                 dollar_per_pt=DOLLAR_PER_PT):
    """Score trades. Returns dict of metrics or None if no trades."""
    if not trades:
        return None
    pts = np.array([t["pts"] for t in trades])
    qtys = np.array([t.get("qty", 1) for t in trades])
    n = len(pts)
    comm_pts = (commission_per_side * 2) / dollar_per_pt
    net_each = pts - comm_pts  # Commission per contract
    net_dollar_each = net_each * dollar_per_pt * qtys
    total_dollar = net_dollar_each.sum()

    wins = net_dollar_each[net_dollar_each > 0]
    losses = net_dollar_each[net_dollar_each <= 0]
    w_sum = wins.sum() if len(wins) else 0
    l_sum = abs(losses.sum()) if len(losses) else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100

    cum = np.cumsum(net_dollar_each)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()

    avg_bars = np.mean([t["bars"] for t in trades])

    sharpe = 0.0
    if len(net_dollar_each) > 1 and np.std(net_dollar_each) > 0:
        sharpe = np.mean(net_dollar_each) / np.std(net_dollar_each) * np.sqrt(252)

    exit_types = {}
    for t in trades:
        r = t["result"]
        exit_types[r] = exit_types.get(r, 0) + 1

    return {
        "count": n,
        "pf": round(pf, 3),
        "win_rate": round(wr, 1),
        "net_dollar": round(total_dollar, 2),
        "max_dd_dollar": round(mdd, 2),
        "avg_bars": round(avg_bars, 1),
        "sharpe": round(sharpe, 3),
        "exits": exit_types,
    }


def fmt_score(sc, label=""):
    """One-line summary of a score dict."""
    if sc is None:
        return f"{label}: NO TRADES"
    return (f"{label}  {sc['count']} trades, WR {sc['win_rate']}%, "
            f"PF {sc['pf']}, Net ${sc['net_dollar']:+.2f}, "
            f"MaxDD ${sc['max_dd_dollar']:.2f}, Sharpe {sc['sharpe']}")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(opens, closes, times, rsi_periods, tp_list, sl_list,
              cd_list, entry_end_list):
    """Sweep over parameter grid. Returns list of (params, score) tuples."""
    results = []
    total = (len(rsi_periods) * len(tp_list) * len(sl_list) *
             len(cd_list) * len(entry_end_list))
    print(f"\nSweep: {total} configs")

    # Pre-compute RSI for each period
    rsi_cache = {}
    signal_cache = {}
    for rp in rsi_periods:
        rsi_arr = compute_rsi(closes, rp)
        rsi_cache[rp] = rsi_arr
        # Generate signals once per RSI period (trendline params are fixed)
        long_sig, short_sig = generate_signals(rsi_arr)
        signal_cache[rp] = (long_sig, short_sig)
        sig_count = long_sig.sum() + short_sig.sum()
        print(f"  RSI({rp}): {long_sig.sum()} long + {short_sig.sum()} short = {sig_count} raw signals")

    count = 0
    t0 = time.time()
    for rp, tp, sl, cd, entry_end in product(rsi_periods, tp_list, sl_list,
                                               cd_list, entry_end_list):
        long_sig, short_sig = signal_cache[rp]
        trades = run_backtest(opens, closes, times, long_sig, short_sig,
                              cooldown_bars=cd, max_loss_pts=sl, tp_pts=tp,
                              entry_end_et=entry_end)
        sc = score_trades(trades)
        if sc and sc['count'] >= 30:  # Min trades filter
            params = {
                'rsi': rp, 'tp': tp, 'sl': sl, 'cd': cd,
                'entry_end': f"{entry_end // 60}:{entry_end % 60:02d}",
            }
            results.append((params, sc))
        count += 1
        if count % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {count}/{total} ({elapsed:.1f}s)", end='\r')

    elapsed = time.time() - t0
    print(f"\n  Sweep done: {len(results)} valid configs in {elapsed:.1f}s")
    return results


def run_runner_sweep(opens, closes, times, rsi_periods, tp1_list, tp2_list,
                     sl_list, cd_list, entry_end_et):
    """Sweep runner mode configs."""
    results = []
    total = (len(rsi_periods) * len(tp1_list) * len(tp2_list) *
             len(sl_list) * len(cd_list))
    print(f"\nRunner sweep: {total} configs")

    signal_cache = {}
    for rp in rsi_periods:
        rsi_arr = compute_rsi(closes, rp)
        long_sig, short_sig = generate_signals(rsi_arr)
        signal_cache[rp] = (long_sig, short_sig)

    count = 0
    t0 = time.time()
    for rp, tp1, tp2, sl, cd in product(rsi_periods, tp1_list, tp2_list,
                                          sl_list, cd_list):
        if tp2 <= tp1:
            continue  # TP2 must be larger than TP1
        long_sig, short_sig = signal_cache[rp]
        trades = run_backtest_runner(opens, closes, times, long_sig, short_sig,
                                     cooldown_bars=cd, max_loss_pts=sl,
                                     tp1_pts=tp1, tp2_pts=tp2,
                                     entry_end_et=entry_end_et)
        sc = score_trades(trades)
        if sc and sc['count'] >= 30:
            params = {
                'rsi': rp, 'tp1': tp1, 'tp2': tp2, 'sl': sl, 'cd': cd,
            }
            results.append((params, sc))
        count += 1

    elapsed = time.time() - t0
    print(f"  Runner sweep done: {len(results)} valid configs in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# IS/OOS Split
# ---------------------------------------------------------------------------
def split_is_oos(df):
    """Split dataframe at chronological midpoint. Returns (is_df, oos_df)."""
    mid = len(df) // 2
    midpoint = df.index[mid]
    return df[df.index < midpoint], df[df.index >= midpoint]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("RSI TRENDLINE BREAKOUT BACKTEST")
    print("=" * 70)

    # Load data
    df = load_mnq_data()
    opens = df['Open'].values
    closes = df['Close'].values
    times = df.index

    # --- FULL DATASET SWEEP ---
    rsi_periods = [8, 11, 14]
    tp_list = [3, 5, 7, 10, 15, 20, 25]
    sl_list = [10, 15, 20, 30, 40]
    cd_list = [10, 20, 30]
    entry_end_list = [
        DEFAULT_ENTRY_END_ET,   # 15:45 (no early cutoff)
        13 * 60,                # 13:00 ET
    ]

    results = run_sweep(opens, closes, times, rsi_periods, tp_list, sl_list,
                        cd_list, entry_end_list)

    # Sort by Sharpe
    results.sort(key=lambda x: x[1]['sharpe'], reverse=True)

    # Print top 20
    print("\n" + "=" * 70)
    print("TOP 20 CONFIGS BY SHARPE (full dataset)")
    print("=" * 70)
    print(f"{'#':>3}  {'RSI':>3}  {'TP':>3}  {'SL':>3}  {'CD':>3}  {'CutOff':>5}  "
          f"{'Trades':>6}  {'WR%':>5}  {'PF':>6}  {'Net$':>9}  {'MaxDD$':>8}  {'Sharpe':>6}")
    print("-" * 85)
    for rank, (params, sc) in enumerate(results[:20], 1):
        print(f"{rank:3d}  {params['rsi']:3d}  {params['tp']:3d}  {params['sl']:3d}  "
              f"{params['cd']:3d}  {params['entry_end']:>5s}  "
              f"{sc['count']:6d}  {sc['win_rate']:5.1f}  {sc['pf']:6.3f}  "
              f"${sc['net_dollar']:>8.2f}  ${sc['max_dd_dollar']:>7.2f}  "
              f"{sc['sharpe']:6.3f}")

    # --- IS/OOS ON BEST CONFIG ---
    if not results:
        print("\nNo valid configs found. Exiting.")
        return

    best_params, best_sc = results[0]
    print(f"\n{'=' * 70}")
    print(f"IS/OOS ANALYSIS -- Best config: RSI={best_params['rsi']} "
          f"TP={best_params['tp']} SL={best_params['sl']} CD={best_params['cd']} "
          f"CutOff={best_params['entry_end']}")
    print("=" * 70)

    df_is, df_oos = split_is_oos(df)
    print(f"IS:  {len(df_is)} bars  {df_is.index[0]} to {df_is.index[-1]}")
    print(f"OOS: {len(df_oos)} bars  {df_oos.index[0]} to {df_oos.index[-1]}")

    # Parse entry_end back to int
    ee_parts = best_params['entry_end'].split(':')
    entry_end_val = int(ee_parts[0]) * 60 + int(ee_parts[1])

    for label, sub_df in [("IS", df_is), ("OOS", df_oos)]:
        sub_opens = sub_df['Open'].values
        sub_closes = sub_df['Close'].values
        sub_times = sub_df.index

        rsi_arr = compute_rsi(sub_closes, best_params['rsi'])
        long_sig, short_sig = generate_signals(rsi_arr)

        trades = run_backtest(sub_opens, sub_closes, sub_times,
                              long_sig, short_sig,
                              cooldown_bars=best_params['cd'],
                              max_loss_pts=best_params['sl'],
                              tp_pts=best_params['tp'],
                              entry_end_et=entry_end_val)
        sc = score_trades(trades)
        print(fmt_score(sc, f"  {label}:"))
        if sc:
            exits_str = "  ".join(f"{k}:{v}" for k, v in sc['exits'].items())
            print(f"       Exits: {exits_str}")

    # --- RUNNER MODE SWEEP ---
    print(f"\n{'=' * 70}")
    print("RUNNER MODE SWEEP (2 contracts: TP1 scalp + TP2 runner)")
    print("=" * 70)

    tp1_list = [3, 5, 7]
    tp2_list = [15, 20, 25, 30]
    runner_sl_list = [10, 15, 20, 30, 40]
    runner_cd_list = [10, 20, 30]

    # Use the best RSI period from single-exit sweep
    best_rsi = best_params['rsi']
    runner_results = run_runner_sweep(
        opens, closes, times,
        rsi_periods=[best_rsi],
        tp1_list=tp1_list, tp2_list=tp2_list,
        sl_list=runner_sl_list, cd_list=runner_cd_list,
        entry_end_et=entry_end_val,
    )

    runner_results.sort(key=lambda x: x[1]['sharpe'], reverse=True)

    print(f"\nTOP 10 RUNNER CONFIGS BY SHARPE (RSI={best_rsi})")
    print(f"{'#':>3}  {'TP1':>3}  {'TP2':>3}  {'SL':>3}  {'CD':>3}  "
          f"{'Trades':>6}  {'WR%':>5}  {'PF':>6}  {'Net$':>9}  {'MaxDD$':>8}  {'Sharpe':>6}")
    print("-" * 75)
    for rank, (params, sc) in enumerate(runner_results[:10], 1):
        print(f"{rank:3d}  {params['tp1']:3d}  {params['tp2']:3d}  {params['sl']:3d}  "
              f"{params['cd']:3d}  "
              f"{sc['count']:6d}  {sc['win_rate']:5.1f}  {sc['pf']:6.3f}  "
              f"${sc['net_dollar']:>8.2f}  ${sc['max_dd_dollar']:>7.2f}  "
              f"{sc['sharpe']:6.3f}")

    # IS/OOS on best runner config
    if runner_results:
        best_rp, best_rsc = runner_results[0]
        print(f"\nRunner IS/OOS -- TP1={best_rp['tp1']} TP2={best_rp['tp2']} "
              f"SL={best_rp['sl']} CD={best_rp['cd']}")
        for label, sub_df in [("IS", df_is), ("OOS", df_oos)]:
            sub_opens = sub_df['Open'].values
            sub_closes = sub_df['Close'].values
            sub_times = sub_df.index

            rsi_arr = compute_rsi(sub_closes, best_rsi)
            long_sig, short_sig = generate_signals(rsi_arr)

            trades = run_backtest_runner(
                sub_opens, sub_closes, sub_times, long_sig, short_sig,
                cooldown_bars=best_rp['cd'], max_loss_pts=best_rp['sl'],
                tp1_pts=best_rp['tp1'], tp2_pts=best_rp['tp2'],
                entry_end_et=entry_end_val,
            )
            sc = score_trades(trades)
            print(fmt_score(sc, f"  {label}:"))
            if sc:
                exits_str = "  ".join(f"{k}:{v}" for k, v in sc['exits'].items())
                print(f"       Exits: {exits_str}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
