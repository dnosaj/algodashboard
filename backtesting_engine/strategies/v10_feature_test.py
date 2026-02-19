"""
v10 Feature Tests: SM Flip Reversal Entry + RSI Momentum Direction
==================================================================
Tests two independent features against the v9 baseline on MNQ 5-min data.

Feature A: SM Flip Reversal Entry
  When SM flips and exits a position, immediately enter the opposite direction
  WITHOUT requiring an RSI cross. The SM flip IS the signal.

Feature B: RSI Momentum Direction
  Only enter if RSI is moving in the trade direction at entry time.
  Long: rsi > rsi_prev (RSI rising). Short: rsi < rsi_prev (RSI falling).

Feature C: Both A + B combined
"""

import pandas as pd
import numpy as np
import sys, os, glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def resample_to_5min(df):
    return df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'SM Net Index': 'last',
    }).dropna(subset=['close'])


def run_strategy(df_5m, rsi_len=10, rsi_buy=55, rsi_sell=45,
                 sm_threshold=0.0, cooldown=3, max_loss_pts=0,
                 entry_start='10:00', entry_end='15:45',
                 session_end='16:00',
                 # Feature toggles
                 use_sm_reversal=False,
                 use_rsi_momentum=False):
    """
    Run v9 SM-flip strategy with optional feature toggles.
    Returns list of trades.
    """
    from datetime import time as dt_time

    # Compute RSI
    delta = df_5m['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss_s = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean()
    avg_loss = loss_s.ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df = df_5m.copy()
    df['rsi'] = rsi
    df['rsi_prev'] = rsi.shift(1)
    df['sm'] = df['SM Net Index']
    df['sm_prev'] = df['sm'].shift(1)

    trades = []
    position = None
    long_used = False
    short_used = False
    bars_since_exit = 9999

    entry_start_t = dt_time(int(entry_start.split(':')[0]), int(entry_start.split(':')[1]))
    entry_end_t = dt_time(int(entry_end.split(':')[0]), int(entry_end.split(':')[1]))
    session_end_t = dt_time(int(session_end.split(':')[0]), int(session_end.split(':')[1]))

    for i in range(1, len(df)):
        row = df.iloc[i]
        ts = df.index[i]

        sm = row['sm']
        sm_prev = row['sm_prev']
        rsi_val = row['rsi']
        rsi_prev_val = row['rsi_prev']

        if pd.isna(sm) or pd.isna(rsi_val) or pd.isna(sm_prev) or pd.isna(rsi_prev_val):
            bars_since_exit += 1
            continue

        sm_bull = sm > sm_threshold
        sm_bear = sm < -sm_threshold
        sm_flipped_bull = sm > 0 and sm_prev <= 0
        sm_flipped_bear = sm < 0 and sm_prev >= 0

        # Episode reset
        if not sm_bull or sm_flipped_bear:
            long_used = False
        if not sm_bear or sm_flipped_bull:
            short_used = False

        rsi_cross_up = rsi_val > rsi_buy and rsi_prev_val <= rsi_buy
        rsi_cross_down = rsi_val < rsi_sell and rsi_prev_val >= rsi_sell

        # RSI momentum check (Feature B)
        rsi_rising = rsi_val > rsi_prev_val
        rsi_falling = rsi_val < rsi_prev_val

        t = ts.time()
        in_entry_window = entry_start_t <= t <= entry_end_t
        in_session = t <= session_end_t

        cooldown_ok = bars_since_exit >= cooldown
        bars_since_exit += 1

        # Track if we need a reversal entry after exit
        reversal_entry = None

        # --- EXITS ---
        if position is not None:
            exit_price = None
            exit_reason = None

            if position['dir'] == 'long' and sm_flipped_bear:
                exit_price = row['open']
                exit_reason = 'SM Flip'
                if use_sm_reversal:
                    reversal_entry = 'short'
            elif position['dir'] == 'short' and sm_flipped_bull:
                exit_price = row['open']
                exit_reason = 'SM Flip'
                if use_sm_reversal:
                    reversal_entry = 'long'

            # Max loss
            if max_loss_pts > 0 and exit_price is None:
                if position['dir'] == 'long' and row['low'] <= position['entry_price'] - max_loss_pts:
                    exit_price = position['entry_price'] - max_loss_pts
                    exit_reason = 'Max Loss'
                elif position['dir'] == 'short' and row['high'] >= position['entry_price'] + max_loss_pts:
                    exit_price = position['entry_price'] + max_loss_pts
                    exit_reason = 'Max Loss'

            # EOD
            if not in_session and exit_price is None:
                exit_price = row['open']
                exit_reason = 'EOD'

            if exit_price is not None:
                if position['dir'] == 'long':
                    pnl = (exit_price - position['entry_price']) * 2
                else:
                    pnl = (position['entry_price'] - exit_price) * 2
                net_pnl = pnl - 1.04  # commission

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': ts,
                    'dir': position['dir'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_sm': position['entry_sm'],
                    'entry_rsi': position.get('entry_rsi', 0),
                    'pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'signal_type': position.get('signal_type', 'RSI Cross'),
                })
                position = None
                bars_since_exit = 0

        # --- ENTRIES ---
        if position is None:
            entered = False

            # SM Flip Reversal Entry (Feature A) - takes priority
            if reversal_entry is not None and in_entry_window:
                # Apply RSI momentum filter if Feature B is also on
                rsi_ok = True
                if use_rsi_momentum:
                    if reversal_entry == 'long':
                        rsi_ok = rsi_rising
                    else:
                        rsi_ok = rsi_falling

                if rsi_ok:
                    position = {
                        'dir': reversal_entry,
                        'entry_price': row['open'],
                        'entry_time': ts,
                        'entry_sm': sm,
                        'entry_rsi': rsi_val,
                        'signal_type': 'SM Reversal',
                    }
                    if reversal_entry == 'long':
                        long_used = True
                    else:
                        short_used = True
                    bars_since_exit = 9999
                    entered = True

            # Standard RSI Cross Entry
            if not entered:
                # Long
                long_rsi_ok = (not use_rsi_momentum) or rsi_rising
                short_rsi_ok = (not use_rsi_momentum) or rsi_falling

                if sm_bull and rsi_cross_up and not long_used and in_entry_window and cooldown_ok and long_rsi_ok:
                    position = {
                        'dir': 'long',
                        'entry_price': row['open'],
                        'entry_time': ts,
                        'entry_sm': sm,
                        'entry_rsi': rsi_val,
                        'signal_type': 'RSI Cross',
                    }
                    long_used = True
                    bars_since_exit = 9999
                elif sm_bear and rsi_cross_down and not short_used and in_entry_window and cooldown_ok and short_rsi_ok:
                    position = {
                        'dir': 'short',
                        'entry_price': row['open'],
                        'entry_time': ts,
                        'entry_sm': sm,
                        'entry_rsi': rsi_val,
                        'signal_type': 'RSI Cross',
                    }
                    short_used = True
                    bars_since_exit = 9999

    # Close any open position
    if position is not None:
        last = df.iloc[-1]
        if position['dir'] == 'long':
            pnl = (last['close'] - position['entry_price']) * 2 - 1.04
        else:
            pnl = (position['entry_price'] - last['close']) * 2 - 1.04
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.index[-1],
            'dir': position['dir'],
            'entry_price': position['entry_price'],
            'exit_price': last['close'],
            'entry_sm': position['entry_sm'],
            'entry_rsi': position.get('entry_rsi', 0),
            'pnl': pnl,
            'exit_reason': 'End',
            'signal_type': position.get('signal_type', 'RSI Cross'),
        })

    return trades


def calc_stats(trade_list, label, show_trades=False):
    if len(trade_list) == 0:
        print(f"\n{label}: No trades")
        return {}

    pnls = [t['pnl'] for t in trade_list]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()

    print(f"\n{label}:")
    print(f"  Trades: {len(trade_list)}")
    print(f"  Win Rate: {len(wins)/len(pnls)*100:.1f}% ({len(wins)}/{len(pnls)})")
    print(f"  Profit Factor: {pf:.3f}")
    print(f"  Total P&L: ${sum(pnls):.2f}")
    print(f"  Avg Win: ${np.mean(wins):.2f}" if wins else "  Avg Win: N/A")
    print(f"  Avg Loss: ${np.mean(losses):.2f}" if losses else "  Avg Loss: N/A")
    print(f"  Avg P&L/trade: ${np.mean(pnls):.2f}")
    print(f"  Max Drawdown: ${max_dd:.2f}")

    if show_trades:
        print(f"\n  {'#':>3} {'Dir':>5} {'Type':>12} {'Entry Time':>20} {'SM':>7} {'RSI':>6} {'P&L':>10} {'Exit':>8}")
        print(f"  {'-'*80}")
        for i, t in enumerate(trade_list):
            print(f"  {i+1:>3} {t['dir']:>5} {t['signal_type']:>12} {str(t['entry_time']):>20} "
                  f"{t['entry_sm']:>7.3f} {t['entry_rsi']:>6.1f} {t['pnl']:>10.2f} {t['exit_reason']:>8}")

    return {
        'trades': len(trade_list),
        'wr': len(wins)/len(pnls)*100,
        'pf': pf,
        'pnl': sum(pnls),
        'max_dd': max_dd,
    }


def main():
    print("=" * 70)
    print("v10 FEATURE TEST: SM Flip Reversal + RSI Momentum Direction")
    print("=" * 70)

    # Load MNQ data
    print("\nLoading MNQ 1-min data...")
    mnq_files = [
        os.path.join(DATA_DIR, "CME_MINI_MNQ1!, 1_0efb1.csv"),
        os.path.join(DATA_DIR, "CME_MINI_MNQ1!, 1_8835e.csv"),
    ]
    dfs = []
    for f in mnq_files:
        df = pd.read_csv(f)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        dfs.append(df)
    mnq_1m = pd.concat(dfs)
    mnq_1m = mnq_1m[~mnq_1m.index.duplicated(keep='last')].sort_index()

    mnq_5m = resample_to_5min(mnq_1m)
    print(f"  MNQ 5-min: {len(mnq_5m)} bars, {mnq_5m.index[0].date()} to {mnq_5m.index[-1].date()}")

    # ================================================================
    # TEST 1: Baseline v9
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: BASELINE v9 (no features)")
    print("=" * 70)
    baseline = run_strategy(mnq_5m)
    s_base = calc_stats(baseline, "v9 Baseline", show_trades=True)

    # ================================================================
    # TEST 2: SM Flip Reversal Only
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: SM FLIP REVERSAL ENTRY (Feature A)")
    print("=" * 70)
    reversal = run_strategy(mnq_5m, use_sm_reversal=True)
    s_rev = calc_stats(reversal, "v9 + SM Reversal", show_trades=True)

    # Show reversal-only trades
    rev_trades = [t for t in reversal if t['signal_type'] == 'SM Reversal']
    if rev_trades:
        calc_stats(rev_trades, "  Reversal trades only")

    rsi_trades = [t for t in reversal if t['signal_type'] == 'RSI Cross']
    if rsi_trades:
        calc_stats(rsi_trades, "  RSI Cross trades only")

    # ================================================================
    # TEST 3: RSI Momentum Direction Only
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: RSI MOMENTUM DIRECTION (Feature B)")
    print("=" * 70)
    momentum = run_strategy(mnq_5m, use_rsi_momentum=True)
    s_mom = calc_stats(momentum, "v9 + RSI Momentum", show_trades=True)

    # ================================================================
    # TEST 4: Both Features
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: BOTH FEATURES (A + B)")
    print("=" * 70)
    both = run_strategy(mnq_5m, use_sm_reversal=True, use_rsi_momentum=True)
    s_both = calc_stats(both, "v9 + Reversal + RSI Momentum", show_trades=True)

    # ================================================================
    # TEST 5: Features + Max Loss 50pts
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 5: BASELINE + MAX LOSS 50 (v9.4)")
    print("=" * 70)
    base_sl = run_strategy(mnq_5m, max_loss_pts=50)
    s_base_sl = calc_stats(base_sl, "v9.4 Baseline (max loss 50)")

    print("\n" + "=" * 70)
    print("TEST 6: SM REVERSAL + MAX LOSS 50")
    print("=" * 70)
    rev_sl = run_strategy(mnq_5m, use_sm_reversal=True, max_loss_pts=50)
    s_rev_sl = calc_stats(rev_sl, "v9.4 + SM Reversal (max loss 50)")

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    configs = [
        ("v9 Baseline", s_base),
        ("+ SM Reversal", s_rev),
        ("+ RSI Momentum", s_mom),
        ("+ Both", s_both),
        ("v9.4 (max loss 50)", s_base_sl),
        ("v9.4 + Reversal", s_rev_sl),
    ]
    print(f"\n{'Config':<25} {'Trades':>7} {'WR%':>7} {'PF':>7} {'P&L':>10} {'MaxDD':>10}")
    print("-" * 70)
    for name, s in configs:
        if s:
            print(f"{name:<25} {s['trades']:>7} {s['wr']:>6.1f}% {s['pf']:>7.3f} ${s['pnl']:>9.2f} ${s['max_dd']:>9.2f}")


if __name__ == '__main__':
    main()
