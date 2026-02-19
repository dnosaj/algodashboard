"""
v15 MFE (Maximum Favorable Excursion) Analysis
================================================
For every v11 MNQ trade, compute how far the trade went in our favor
before exiting. Then test: at what fixed take-profit level would a
second strategy be independently profitable?

Uses the FULL 6-month dataset and TradingView validation period.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from v10_test_common import (
    load_instrument_1min, resample_to_5min, compute_rsi,
    compute_et_minutes, NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)


def prepare_arrays(instrument='MNQ', sm_params=None):
    """Load data and prepare arrays for backtesting."""
    if sm_params is None:
        sm_params = {'index_period': 10, 'flow_period': 12,
                     'norm_period': 200, 'ema_len': 100}

    df = load_instrument_1min(instrument)

    # Recompute SM with v11 params
    from v10_test_common import compute_smart_money
    sm = compute_smart_money(
        df['Close'].values, df['Volume'].values, **sm_params)
    df['SM_Net'] = sm

    # Compute 5-min RSI and map to 1-min
    df_5m = resample_to_5min(df)
    rsi_5m_raw = compute_rsi(df_5m['Close'].values, 8)

    # Map 5-min RSI back to 1-min (Pine request.security with lookahead_off)
    idx_1m = df.index
    idx_5m = df_5m.index
    rsi_5m_curr = np.full(len(idx_1m), np.nan)
    rsi_5m_prev = np.full(len(idx_1m), np.nan)

    j = 0
    for i in range(len(idx_1m)):
        while j + 1 < len(idx_5m) and idx_5m[j + 1] <= idx_1m[i]:
            j += 1
        if j < len(idx_5m) and idx_5m[j] <= idx_1m[i]:
            rsi_5m_curr[i] = rsi_5m_raw[j]
            if j > 0:
                rsi_5m_prev[i] = rsi_5m_raw[j - 1]

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    times = df.index
    et_mins = compute_et_minutes(times)

    return opens, highs, lows, closes, sm, rsi_5m_curr, rsi_5m_prev, times, et_mins


def run_v11_with_mfe(opens, highs, lows, closes, sm, rsi_curr, rsi_prev,
                     times, et_mins, rsi_buy=60, rsi_sell=40,
                     sm_threshold=0.0, cooldown=20, max_loss_pts=50):
    """Run v11 backtest and record MFE for each trade."""
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    for i in range(2, n):
        bar_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # Episode reset
        if not sm_bull or sm_flipped_bear:
            long_used = False
        if not sm_bear or sm_flipped_bull:
            short_used = False

        # RSI cross (mapped 5-min)
        rc_up = (rsi_curr[i - 1] > rsi_buy and
                 rsi_prev[i - 1] <= rsi_buy) if not np.isnan(rsi_curr[i - 1]) else False
        rc_dn = (rsi_curr[i - 1] < rsi_sell and
                 rsi_prev[i - 1] >= rsi_sell) if not np.isnan(rsi_curr[i - 1]) else False

        in_entry = NY_OPEN_ET <= bar_et < NY_LAST_ENTRY_ET
        in_session = bar_et < NY_CLOSE_ET
        cooldown_ok = (i - exit_bar) >= cooldown

        # --- EXITS ---
        if trade_state != 0:
            exit_price = None
            exit_reason = None

            # EOD
            if not in_session or bar_et >= NY_CLOSE_ET:
                exit_price = opens[i]
                exit_reason = "EOD"

            # SM flip
            if exit_price is None:
                if trade_state == 1 and sm_flipped_bear:
                    exit_price = opens[i]
                    exit_reason = "SM Flip"
                elif trade_state == -1 and sm_flipped_bull:
                    exit_price = opens[i]
                    exit_reason = "SM Flip"

            # Max loss (check prev bar close, fill at current open)
            if exit_price is None and max_loss_pts > 0:
                if trade_state == 1 and closes[i - 1] <= entry_price - max_loss_pts:
                    exit_price = opens[i]
                    exit_reason = "Max Loss"
                elif trade_state == -1 and closes[i - 1] >= entry_price + max_loss_pts:
                    exit_price = opens[i]
                    exit_reason = "Max Loss"

            if exit_price is not None:
                # Compute MFE: scan all bars from entry to exit
                side = "long" if trade_state == 1 else "short"
                pts = (exit_price - entry_price) if trade_state == 1 else (entry_price - exit_price)

                # MFE: max favorable price during trade
                mfe = 0.0
                mae = 0.0  # max adverse excursion
                for k in range(entry_idx, i + 1):
                    if trade_state == 1:
                        fav = highs[k] - entry_price
                        adv = entry_price - lows[k]
                    else:
                        fav = entry_price - lows[k]
                        adv = highs[k] - entry_price
                    if fav > mfe:
                        mfe = fav
                    if adv > mae:
                        mae = adv

                trades.append({
                    'side': side, 'entry': entry_price, 'exit': exit_price,
                    'pts': pts, 'mfe': mfe, 'mae': mae,
                    'entry_time': times[entry_idx], 'exit_time': times[i],
                    'bars': i - entry_idx, 'result': exit_reason,
                })
                trade_state = 0
                exit_bar = i

        # --- ENTRIES ---
        if trade_state == 0 and in_entry and cooldown_ok:
            if sm_bull and rc_up and not long_used:
                trade_state = 1
                entry_price = opens[i]
                entry_idx = i
                long_used = True
            elif sm_bear and rc_dn and not short_used:
                trade_state = -1
                entry_price = opens[i]
                entry_idx = i
                short_used = True

    return trades


def simulate_tp(trades, tp_pts, commission=0.52, dollar_per_pt=2.0):
    """Simulate a fixed take-profit strategy using MFE data.

    For each trade:
    - If MFE >= tp_pts: WIN at +tp_pts
    - If MFE < tp_pts: same exit as v11 (SM flip, max loss, EOD)
    """
    wins = 0
    total_pnl = 0.0
    for t in trades:
        if t['mfe'] >= tp_pts:
            # Would have hit TP
            pnl = tp_pts * dollar_per_pt - 2 * commission
            wins += 1
        else:
            # Same exit as v11
            pnl = t['pts'] * dollar_per_pt - 2 * commission
            if t['pts'] > 0:
                wins += 1
        total_pnl += pnl

    n = len(trades)
    wr = wins / n * 100 if n > 0 else 0
    gross_win = sum(
        (tp_pts * dollar_per_pt if t['mfe'] >= tp_pts else max(0, t['pts'] * dollar_per_pt))
        for t in trades
    )
    gross_loss = sum(
        (0 if t['mfe'] >= tp_pts else abs(min(0, t['pts'] * dollar_per_pt)))
        for t in trades
    )
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    return {
        'tp': tp_pts, 'trades': n, 'wins': wins, 'wr': wr,
        'pf': pf, 'net': total_pnl,
        'hit_rate': sum(1 for t in trades if t['mfe'] >= tp_pts) / n * 100,
    }


def main():
    print("Loading MNQ data and computing indicators...")
    arrays = prepare_arrays('MNQ')

    print("Running v11 backtest with MFE tracking...")
    trades = run_v11_with_mfe(*arrays)
    print(f"Total trades: {len(trades)}")

    # Split into periods
    tv_start = pd.Timestamp('2026-01-19')
    tv_end = pd.Timestamp('2026-02-14')
    train_end = pd.Timestamp('2025-11-17')
    test_start = pd.Timestamp('2025-11-17')

    all_trades = trades
    tv_trades = [t for t in trades if tv_start <= t['entry_time'] < tv_end]
    train_trades = [t for t in trades if t['entry_time'] < train_end]
    test_trades = [t for t in trades if t['entry_time'] >= test_start]

    # --- MFE Distribution ---
    for label, tlist in [("ALL (6-month)", all_trades),
                          ("TV Window (Jan 19 - Feb 13)", tv_trades),
                          ("TRAIN (Aug-Nov)", train_trades),
                          ("TEST (Nov-Feb)", test_trades)]:
        if not tlist:
            continue
        mfes = [t['mfe'] for t in tlist]
        print(f"\n{'='*60}")
        print(f"MFE Distribution: {label} ({len(tlist)} trades)")
        print(f"{'='*60}")
        print(f"  Min MFE:    {min(mfes):.1f} pts")
        print(f"  Mean MFE:   {np.mean(mfes):.1f} pts")
        print(f"  Median MFE: {np.median(mfes):.1f} pts")
        print(f"  Max MFE:    {max(mfes):.1f} pts")

        # How many trades reach each TP level
        print(f"\n  TP Level | Hit Rate | Would-Win | Would-Lose")
        print(f"  ---------|----------|-----------|----------")
        for tp in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]:
            hits = sum(1 for m in mfes if m >= tp)
            misses = len(mfes) - hits
            pct = hits / len(mfes) * 100
            print(f"  {tp:>5} pts | {pct:>5.1f}%  | {hits:>9} | {misses:>9}")

    # --- TP Simulation ---
    for label, tlist in [("ALL (6-month)", all_trades),
                          ("TV Window (Jan 19 - Feb 13)", tv_trades),
                          ("TRAIN (Aug-Nov)", train_trades),
                          ("TEST (Nov-Feb)", test_trades)]:
        if not tlist:
            continue
        print(f"\n{'='*60}")
        print(f"TP Simulation: {label} ({len(tlist)} trades)")
        print(f"{'='*60}")

        # v11 baseline
        net_v11 = sum(t['pts'] * 2.0 - 1.04 for t in tlist)
        wins_v11 = sum(1 for t in tlist if t['pts'] > 0)
        gw = sum(t['pts'] * 2.0 for t in tlist if t['pts'] > 0)
        gl = sum(abs(t['pts'] * 2.0) for t in tlist if t['pts'] <= 0)
        pf_v11 = gw / gl if gl > 0 else float('inf')
        print(f"  v11 Baseline: {len(tlist)} trades, WR {wins_v11/len(tlist)*100:.1f}%, "
              f"PF {pf_v11:.3f}, Net ${net_v11:.2f}")

        print(f"\n  {'TP':>5} | {'Hit%':>6} | {'WR':>6} | {'PF':>7} | {'Net $':>9} | {'vs v11':>8}")
        print(f"  {'---':>5}-+-{'---':>6}-+-{'---':>6}-+-{'---':>7}-+-{'---':>9}-+-{'---':>8}")

        for tp in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]:
            r = simulate_tp(tlist, tp)
            delta = r['net'] - net_v11
            marker = " ***" if r['net'] > net_v11 and r['pf'] > 1.0 else ""
            print(f"  {tp:>5} | {r['hit_rate']:>5.1f}% | {r['wr']:>5.1f}% | "
                  f"{r['pf']:>7.3f} | {r['net']:>+9.2f} | {delta:>+8.2f}{marker}")

    # --- Detailed trade list for TV window ---
    print(f"\n{'='*60}")
    print(f"Trade Detail: TV Window (Jan 19 - Feb 13)")
    print(f"{'='*60}")
    print(f"  {'#':>3} {'Side':>5} {'Entry':>10} {'Exit':>10} {'PnL':>7} {'MFE':>6} {'MAE':>6} {'Bars':>5} {'Result':>10}")
    for i, t in enumerate(tv_trades):
        print(f"  {i+1:>3} {t['side']:>5} {t['entry']:>10.2f} {t['exit']:>10.2f} "
              f"{t['pts']:>+7.2f} {t['mfe']:>6.1f} {t['mae']:>6.1f} {t['bars']:>5} {t['result']:>10}")


if __name__ == '__main__':
    main()
