"""
Cross-Instrument SM Correlation Test
=====================================
Question: When MNQ SM and MES SM agree on direction, do v9 trades perform better?

Approach:
1. Load MNQ 1-min data with SM (0efb1 + 8835e for Feb 13)
2. Load MES 1-min data with SM (016a0)
3. Align by timestamp
4. Run v9 strategy on MNQ (primary)
5. For each MNQ trade, check what MES SM was doing at entry
6. Split trades into "SM agree" vs "SM disagree" buckets
7. Compare PF, WR, avg P&L
"""

import pandas as pd
import numpy as np
import sys, os, glob

# Add engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine import engine

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_csv_with_sm(pattern):
    """Load CSV with SM Net Index column."""
    matches = glob.glob(os.path.join(DATA_DIR, pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern}")

    dfs = []
    for path in sorted(matches):
        df = pd.read_csv(path)
        if 'SM Net Index' not in df.columns:
            continue
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        dfs.append(df)

    if len(dfs) > 1:
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        return combined
    return dfs[0]


def resample_to_5min(df):
    """Resample 1-min data to 5-min bars."""
    ohlc = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'SM Net Index': 'last',
    }).dropna(subset=['close'])
    return ohlc


def run_v9_backtest(df_5m, rsi_len=10, rsi_buy=55, rsi_sell=45,
                    sm_threshold=0.0, cooldown=3, max_loss_pts=0,
                    entry_start='10:00', entry_end='15:45',
                    session_end='16:00'):
    """Run v9 SM-flip strategy on 5-min data. Returns list of trades with entry timestamps."""

    # Compute RSI
    delta = df_5m['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df_5m = df_5m.copy()
    df_5m['rsi'] = rsi
    df_5m['rsi_prev'] = rsi.shift(1)
    df_5m['sm'] = df_5m['SM Net Index']
    df_5m['sm_prev'] = df_5m['sm'].shift(1)

    trades = []
    position = None  # {'dir': 'long'/'short', 'entry_price': float, 'entry_time': ts, 'entry_sm': float}
    long_used = False
    short_used = False
    bars_since_exit = 9999

    for i in range(1, len(df_5m)):
        row = df_5m.iloc[i]
        prev = df_5m.iloc[i-1]
        ts = df_5m.index[i]

        sm = row['sm']
        sm_prev = row['sm_prev']
        rsi_val = row['rsi']
        rsi_prev_val = row['rsi_prev']

        if pd.isna(sm) or pd.isna(rsi_val) or pd.isna(sm_prev) or pd.isna(rsi_prev_val):
            bars_since_exit += 1
            continue

        # SM direction
        sm_bull = sm > sm_threshold
        sm_bear = sm < -sm_threshold
        sm_flipped_bull = sm > 0 and sm_prev <= 0
        sm_flipped_bear = sm < 0 and sm_prev >= 0

        # Episode reset
        if not sm_bull or sm_flipped_bear:
            long_used = False
        if not sm_bear or sm_flipped_bull:
            short_used = False

        # RSI cross
        rsi_cross_up = rsi_val > rsi_buy and rsi_prev_val <= rsi_buy
        rsi_cross_down = rsi_val < rsi_sell and rsi_prev_val >= rsi_sell

        # Session filter
        t = ts.time()
        from datetime import time as dt_time
        entry_start_t = dt_time(int(entry_start.split(':')[0]), int(entry_start.split(':')[1]))
        entry_end_t = dt_time(int(entry_end.split(':')[0]), int(entry_end.split(':')[1]))
        session_end_t = dt_time(int(session_end.split(':')[0]), int(session_end.split(':')[1]))

        in_entry_window = entry_start_t <= t <= entry_end_t
        in_session = t <= session_end_t

        cooldown_ok = bars_since_exit >= cooldown
        bars_since_exit += 1

        # --- EXITS ---
        if position is not None:
            exit_price = None
            exit_reason = None

            # SM flip exit
            if position['dir'] == 'long' and sm_flipped_bear:
                exit_price = row['open']
                exit_reason = 'SM Flip'
            elif position['dir'] == 'short' and sm_flipped_bull:
                exit_price = row['open']
                exit_reason = 'SM Flip'

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
                    pnl = (exit_price - position['entry_price']) * 2  # $2/pt for MNQ
                else:
                    pnl = (position['entry_price'] - exit_price) * 2

                commission = 0.52 * 2  # round trip
                net_pnl = pnl - commission

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': ts,
                    'dir': position['dir'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_sm': position['entry_sm'],
                    'pnl': net_pnl,
                    'exit_reason': exit_reason,
                })

                position = None
                bars_since_exit = 0
                continue

        # --- ENTRIES ---
        if position is None:
            # Long
            if sm_bull and rsi_cross_up and not long_used and in_entry_window and cooldown_ok:
                position = {
                    'dir': 'long',
                    'entry_price': row['open'],
                    'entry_time': ts,
                    'entry_sm': sm,
                }
                long_used = True
                bars_since_exit = 9999
            # Short
            elif sm_bear and rsi_cross_down and not short_used and in_entry_window and cooldown_ok:
                position = {
                    'dir': 'short',
                    'entry_price': row['open'],
                    'entry_time': ts,
                    'entry_sm': sm,
                }
                short_used = True
                bars_since_exit = 9999

    # Close any open position at end
    if position is not None:
        last = df_5m.iloc[-1]
        if position['dir'] == 'long':
            pnl = (last['close'] - position['entry_price']) * 2 - 1.04
        else:
            pnl = (position['entry_price'] - last['close']) * 2 - 1.04
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df_5m.index[-1],
            'dir': position['dir'],
            'entry_price': position['entry_price'],
            'exit_price': last['close'],
            'entry_sm': position['entry_sm'],
            'pnl': pnl,
            'exit_reason': 'End',
        })

    return trades


def main():
    print("=" * 70)
    print("CROSS-INSTRUMENT SM CORRELATION TEST")
    print("=" * 70)

    # Load MNQ data (with SM)
    print("\nLoading MNQ 1-min data with SM...")
    mnq_files = [
        os.path.join(DATA_DIR, "CME_MINI_MNQ1!, 1_0efb1.csv"),  # Jan 18 - Feb 12
        os.path.join(DATA_DIR, "CME_MINI_MNQ1!, 1_8835e.csv"),  # Feb 13
    ]
    mnq_dfs = []
    for f in mnq_files:
        df = pd.read_csv(f)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        mnq_dfs.append(df)
    mnq_1m = pd.concat(mnq_dfs)
    mnq_1m = mnq_1m[~mnq_1m.index.duplicated(keep='last')].sort_index()
    print(f"  MNQ: {mnq_1m.index[0]} to {mnq_1m.index[-1]} ({len(mnq_1m)} bars)")

    # Load MES data (with SM)
    print("Loading MES 1-min data with SM...")
    mes_1m = pd.read_csv(os.path.join(DATA_DIR, "CME_MINI_MES1!, 1_016a0.csv"))
    mes_1m['datetime'] = pd.to_datetime(mes_1m['time'], unit='s')
    mes_1m.set_index('datetime', inplace=True)
    print(f"  MES: {mes_1m.index[0]} to {mes_1m.index[-1]} ({len(mes_1m)} bars)")

    # Find overlap period
    overlap_start = max(mnq_1m.index[0], mes_1m.index[0])
    overlap_end = min(mnq_1m.index[-1], mes_1m.index[-1])
    print(f"\nOverlap period: {overlap_start.date()} to {overlap_end.date()}")

    # Resample both to 5-min
    print("\nResampling to 5-min bars...")
    mnq_5m = resample_to_5min(mnq_1m)
    mes_5m = resample_to_5min(mes_1m)
    print(f"  MNQ 5-min: {len(mnq_5m)} bars")
    print(f"  MES 5-min: {len(mes_5m)} bars")

    # Create MES SM lookup (1-min resolution for precise trade-time matching)
    mes_sm_series = mes_1m['SM Net Index'].copy()

    # Run v9 backtest on MNQ
    print("\nRunning v9 backtest on MNQ 5-min...")
    trades = run_v9_backtest(mnq_5m, rsi_len=10, rsi_buy=55, rsi_sell=45,
                            sm_threshold=0.0, cooldown=3, max_loss_pts=0)

    # Filter to overlap period only
    trades = [t for t in trades if t['entry_time'] >= overlap_start and t['entry_time'] <= overlap_end]
    print(f"  Total trades in overlap period: {len(trades)}")

    if len(trades) == 0:
        print("No trades found!")
        return

    # For each trade, look up MES SM at entry time
    print("\nMatching MES SM at each MNQ trade entry...")
    for trade in trades:
        entry_time = trade['entry_time']

        # Find closest MES SM value (within 5 min window)
        mask = (mes_sm_series.index >= entry_time - pd.Timedelta(minutes=5)) & \
               (mes_sm_series.index <= entry_time + pd.Timedelta(minutes=1))
        nearby = mes_sm_series[mask]

        if len(nearby) > 0:
            # Get the value at or just before entry time
            before = nearby[nearby.index <= entry_time]
            if len(before) > 0:
                trade['mes_sm'] = before.iloc[-1]
            else:
                trade['mes_sm'] = nearby.iloc[0]
        else:
            trade['mes_sm'] = np.nan

    # Classify trades
    agree_trades = []
    disagree_trades = []
    no_data = []

    for t in trades:
        if pd.isna(t.get('mes_sm', np.nan)):
            no_data.append(t)
            continue

        mnq_bull = t['entry_sm'] > 0
        mes_bull = t['mes_sm'] > 0

        if mnq_bull == mes_bull:
            t['sm_agreement'] = 'AGREE'
            agree_trades.append(t)
        else:
            t['sm_agreement'] = 'DISAGREE'
            disagree_trades.append(t)

    # Print all trades
    print("\n" + "=" * 100)
    print(f"{'#':>3} {'Dir':>5} {'Entry Time':>20} {'MNQ SM':>8} {'MES SM':>8} {'Agree':>8} {'P&L':>10} {'Exit':>8}")
    print("-" * 100)
    for i, t in enumerate(trades):
        mes_sm = t.get('mes_sm', np.nan)
        agree = t.get('sm_agreement', 'N/A')
        print(f"{i+1:>3} {t['dir']:>5} {str(t['entry_time']):>20} {t['entry_sm']:>8.3f} {mes_sm:>8.3f} {agree:>8} {t['pnl']:>10.2f} {t['exit_reason']:>8}")

    # Summary statistics
    def calc_stats(trade_list, label):
        if len(trade_list) == 0:
            print(f"\n{label}: No trades")
            return

        pnls = [t['pnl'] for t in trade_list]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        print(f"\n{label}:")
        print(f"  Trades: {len(trade_list)}")
        print(f"  Win Rate: {win_rate:.1f}% ({len(wins)}/{len(pnls)})")
        print(f"  Profit Factor: {pf:.3f}")
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Avg Win: ${avg_win:.2f}")
        print(f"  Avg Loss: ${avg_loss:.2f}")
        print(f"  Avg P&L per trade: ${np.mean(pnls):.2f}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    calc_stats(trades, "ALL TRADES (baseline)")
    calc_stats(agree_trades, "SM AGREE (MNQ + MES same direction)")
    calc_stats(disagree_trades, "SM DISAGREE (MNQ + MES opposite)")

    if no_data:
        print(f"\n  ({len(no_data)} trades had no MES SM data)")

    # Day-by-day SM correlation
    print("\n" + "=" * 70)
    print("DAILY SM CORRELATION (1-min bars)")
    print("=" * 70)

    # Align 1-min SM values
    aligned = pd.DataFrame({
        'mnq_sm': mnq_1m['SM Net Index'],
        'mes_sm': mes_1m['SM Net Index'],
    }).dropna()

    # Filter to RTH only
    rth = aligned.between_time('09:30', '16:00')

    if len(rth) > 0:
        # Overall correlation
        corr = rth['mnq_sm'].corr(rth['mes_sm'])
        print(f"\nOverall SM correlation (RTH): {corr:.4f}")

        # Direction agreement
        same_dir = ((rth['mnq_sm'] > 0) & (rth['mes_sm'] > 0)) | \
                   ((rth['mnq_sm'] < 0) & (rth['mes_sm'] < 0)) | \
                   ((rth['mnq_sm'] == 0) & (rth['mes_sm'] == 0))
        pct_agree = same_dir.mean() * 100
        print(f"Direction agreement (RTH): {pct_agree:.1f}% of bars")

        # By day
        print(f"\n{'Date':>12} {'Corr':>8} {'Agree%':>8} {'Bars':>6}")
        print("-" * 40)
        for date, group in rth.groupby(rth.index.date):
            if len(group) > 10:
                day_corr = group['mnq_sm'].corr(group['mes_sm'])
                day_agree = (((group['mnq_sm'] > 0) & (group['mes_sm'] > 0)) | \
                             ((group['mnq_sm'] < 0) & (group['mes_sm'] < 0))).mean() * 100
                print(f"{str(date):>12} {day_corr:>8.4f} {day_agree:>7.1f}% {len(group):>6}")

    # SM agreement at specific MNQ trade entry times
    print("\n" + "=" * 70)
    print("SM AGREEMENT STRENGTH AT TRADE ENTRIES")
    print("=" * 70)
    print(f"\n{'#':>3} {'Dir':>5} {'MNQ SM':>8} {'MES SM':>8} {'Agreement':>12} {'P&L':>10}")
    print("-" * 60)
    for i, t in enumerate(trades):
        mes_sm = t.get('mes_sm', np.nan)
        if pd.isna(mes_sm):
            continue

        # Compute agreement strength (both positive or both negative, and magnitude)
        mnq_dir = 'bull' if t['entry_sm'] > 0 else 'bear'
        mes_dir = 'bull' if mes_sm > 0 else 'bear'

        if mnq_dir == mes_dir:
            strength = min(abs(t['entry_sm']), abs(mes_sm))
            label = f"AGREE ({strength:.3f})"
        else:
            strength = -min(abs(t['entry_sm']), abs(mes_sm))
            label = f"DISAGREE"

        pnl_str = f"${t['pnl']:>8.2f}"
        color = ""
        print(f"{i+1:>3} {t['dir']:>5} {t['entry_sm']:>8.3f} {mes_sm:>8.3f} {label:>12} {pnl_str:>10}")


if __name__ == '__main__':
    main()
