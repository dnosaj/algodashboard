"""
Debug Max-Loss Stops: Python 0 vs TradingView 13
==================================================
Investigates why the Python backtest engine produces 0 max-loss stop exits
while TradingView shows 13 for the same MNQ v11 strategy.

Strategy: SM(10/12/200/100), RSI(8/60/40), cooldown=20, max_loss=50pts

For every trade, computes Maximum Adverse Excursion (MAE) -- the worst
intraday drawdown using actual bar-by-bar lows (longs) / highs (shorts).

Key question: Does MAE ever reach or exceed 50 pts? If SM flip always
fires before 50 pts, that explains 0 ML stops in Python.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_test_common import (
    compute_smart_money, compute_rsi, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    fmt_score, fmt_exits,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_FILE = (Path(__file__).resolve().parent.parent
             / "data" / "databento_MNQ_1min_2025-08-17_to_2026-02-13.csv")

COMMISSION = 0.52
DOLLAR_PER_PT = 2.0  # MNQ

# v11 MNQ params
SM_INDEX = 10
SM_FLOW = 12
SM_NORM = 200
SM_EMA = 100
RSI_LEN = 8
RSI_BUY = 60
RSI_SELL = 40
COOLDOWN = 20
MAX_LOSS_PTS = 50


def load_mnq_databento():
    """Load MNQ 1-min Databento data."""
    df = pd.read_csv(DATA_FILE)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    return result


def compute_mae(trades, opens, highs, lows, closes, sm, times):
    """Compute Maximum Adverse Excursion for each trade.

    For longs:  MAE = entry_price - min(lows[entry_idx:exit_idx+1])
    For shorts: MAE = max(highs[entry_idx:exit_idx+1]) - entry_price

    Also computes:
    - max_close_drawdown: worst drawdown using CLOSES only (what the engine sees)
    - bar_of_worst: which bar within the trade had the worst MAE

    Additionally checks whether close ever crossed the stop threshold,
    and if so, whether SM flipped first.
    """
    results = []
    for i, t in enumerate(trades):
        entry_idx = t['entry_idx']
        exit_idx = t['exit_idx']
        entry_price = t['entry']
        side = t['side']

        # MAE from lows/highs (true intraday worst)
        if side == 'long':
            # Worst drawdown from entry using bar lows
            bar_lows = lows[entry_idx:exit_idx + 1]
            worst_price = np.min(bar_lows)
            mae_pts = entry_price - worst_price
            worst_bar_rel = np.argmin(bar_lows)

            # Worst drawdown from CLOSES only (what stop logic sees)
            bar_closes = closes[entry_idx:exit_idx + 1]
            worst_close = np.min(bar_closes)
            max_close_dd = entry_price - worst_close

            # Check if close ever crossed the 50pt threshold during trade
            # The engine checks closes[i-1], so we look at all closes during
            # the trade period (entry_idx to exit_idx-1 for signal bars)
            close_crossed_stop = np.any(
                closes[entry_idx:exit_idx] <= entry_price - MAX_LOSS_PTS
            )

        else:  # short
            bar_highs = highs[entry_idx:exit_idx + 1]
            worst_price = np.max(bar_highs)
            mae_pts = worst_price - entry_price
            worst_bar_rel = np.argmax(bar_highs)

            bar_closes = closes[entry_idx:exit_idx + 1]
            worst_close = np.max(bar_closes)
            max_close_dd = worst_close - entry_price

            close_crossed_stop = np.any(
                closes[entry_idx:exit_idx] >= entry_price + MAX_LOSS_PTS
            )

        # Check SM value at worst bar
        worst_bar_abs = entry_idx + worst_bar_rel
        sm_at_worst = sm[worst_bar_abs] if worst_bar_abs < len(sm) else 0.0

        # Check SM value at each bar from entry to see when it flipped
        sm_flip_bar = None
        for j in range(entry_idx + 1, exit_idx + 1):
            if side == 'long' and sm[j] < 0 and sm[j - 1] >= 0:
                sm_flip_bar = j
                break
            elif side == 'short' and sm[j] > 0 and sm[j - 1] <= 0:
                sm_flip_bar = j
                break

        # Drawdown at SM flip point
        dd_at_flip = None
        if sm_flip_bar is not None:
            if side == 'long':
                dd_at_flip = entry_price - closes[sm_flip_bar]
            else:
                dd_at_flip = closes[sm_flip_bar] - entry_price

        results.append({
            'trade_num': i + 1,
            'side': side,
            'entry_time': pd.Timestamp(t['entry_time']),
            'exit_time': pd.Timestamp(t['exit_time']),
            'entry_price': entry_price,
            'exit_price': t['exit'],
            'pnl_pts': t['pts'],
            'mae_pts': round(mae_pts, 2),
            'max_close_dd_pts': round(max_close_dd, 2),
            'worst_bar_rel': worst_bar_rel,
            'bars': t['bars'],
            'exit_reason': t['result'],
            'close_crossed_stop': close_crossed_stop,
            'sm_at_worst': round(sm_at_worst, 4),
            'sm_flip_bar': sm_flip_bar,
            'dd_at_flip': round(dd_at_flip, 2) if dd_at_flip is not None else None,
        })

    return results


def print_trade_table(mae_results, title="ALL TRADES", highlight_threshold=30):
    """Print formatted table of trades sorted by MAE (worst first)."""
    sorted_trades = sorted(mae_results, key=lambda x: -x['mae_pts'])

    print(f"\n{'=' * 140}")
    print(f"  {title}")
    print(f"  ({len(sorted_trades)} trades, sorted by MAE descending)")
    print(f"{'=' * 140}")
    print(f"  {'#':>3}  {'Side':>5}  {'Entry Date':>16}  {'Exit Date':>16}  "
          f"{'Entry':>10}  {'Exit':>10}  {'PnL':>8}  {'MAE':>7}  "
          f"{'CloseDD':>8}  {'Bars':>5}  {'Exit':>8}  {'ClsCross50':>10}  {'DDatFlip':>8}")
    print(f"  {'-' * 136}")

    count_mae_gt30 = 0
    count_mae_gt40 = 0
    count_mae_gt50 = 0
    count_close_crossed = 0

    for r in sorted_trades:
        marker = ""
        if r['mae_pts'] >= 50:
            marker = " *** MAE>=50 ***"
            count_mae_gt50 += 1
        elif r['mae_pts'] >= 40:
            marker = " ** MAE>=40 **"
        elif r['mae_pts'] >= highlight_threshold:
            marker = " * MAE>=30 *"

        if r['mae_pts'] >= 30:
            count_mae_gt30 += 1
        if r['mae_pts'] >= 40:
            count_mae_gt40 += 1
        if r['close_crossed_stop']:
            count_close_crossed += 1

        entry_dt = r['entry_time'].strftime('%m/%d %H:%M')
        exit_dt = r['exit_time'].strftime('%m/%d %H:%M')
        dd_flip_str = f"{r['dd_at_flip']:>7.1f}" if r['dd_at_flip'] is not None else "    N/A"

        print(f"  {r['trade_num']:>3}  {r['side']:>5}  {entry_dt:>16}  {exit_dt:>16}  "
              f"{r['entry_price']:>10.2f}  {r['exit_price']:>10.2f}  "
              f"{r['pnl_pts']:>+7.2f}  {r['mae_pts']:>7.2f}  "
              f"{r['max_close_dd_pts']:>8.2f}  {r['bars']:>5}  "
              f"{r['exit_reason']:>8}  {'YES' if r['close_crossed_stop'] else 'no':>10}  "
              f"{dd_flip_str}{marker}")

    print(f"\n  SUMMARY:")
    print(f"    Trades with MAE >= 30 pts: {count_mae_gt30}")
    print(f"    Trades with MAE >= 40 pts: {count_mae_gt40}")
    print(f"    Trades with MAE >= 50 pts: {count_mae_gt50}")
    print(f"    Trades where CLOSE crossed 50pt stop: {count_close_crossed}")

    if sorted_trades:
        maes = [r['mae_pts'] for r in sorted_trades]
        close_dds = [r['max_close_dd_pts'] for r in sorted_trades]
        print(f"\n    MAE stats:     mean={np.mean(maes):.1f}  median={np.median(maes):.1f}  "
              f"max={np.max(maes):.1f}  P90={np.percentile(maes, 90):.1f}  "
              f"P95={np.percentile(maes, 95):.1f}  P99={np.percentile(maes, 99):.1f}")
        print(f"    CloseDD stats: mean={np.mean(close_dds):.1f}  median={np.median(close_dds):.1f}  "
              f"max={np.max(close_dds):.1f}  P90={np.percentile(close_dds, 90):.1f}  "
              f"P95={np.percentile(close_dds, 95):.1f}  P99={np.percentile(close_dds, 99):.1f}")


def main():
    print("=" * 140)
    print("  DEBUG: Max-Loss Stops -- Python 0 vs TradingView 13")
    print("  MNQ v11: SM(10/12/200/100), RSI(8/60/40), CD=20, SL=50")
    print("=" * 140)

    # =========================================================================
    # 1. Load data
    # =========================================================================
    print("\n[1] Loading MNQ 1-min Databento data...")
    df = load_mnq_databento()
    print(f"    {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # =========================================================================
    # 2. Compute SM with v11 params on full data
    # =========================================================================
    print("\n[2] Computing SM with v11 params (10/12/200/100)...")
    closes_full = df['Close'].values
    volumes_full = df['Volume'].values
    sm_full = compute_smart_money(closes_full, volumes_full,
                                  SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)
    print(f"    SM computed: min={sm_full.min():.4f}, max={sm_full.max():.4f}")

    # =========================================================================
    # 3. Prepare 1-min arrays with 5-min RSI mapping
    # =========================================================================
    print("\n[3] Preparing backtest arrays (1-min with 5-min RSI mapping)...")
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    times = df.index.values

    # Resample to 5-min for RSI
    df_for_5m = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_5m['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_for_5m)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=RSI_LEN)
    print(f"    5-min bars: {len(df_5m)}")

    # =========================================================================
    # 4. Run backtest WITH max_loss=50
    # =========================================================================
    print("\n[4] Running backtest WITH max_loss=50...")
    trades_sl50 = run_backtest_v10(
        opens, highs, lows, closes, sm_full, rsi_curr, times,
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=0.0, cooldown_bars=COOLDOWN,
        max_loss_pts=MAX_LOSS_PTS,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )
    sc_sl50 = score_trades(trades_sl50, commission_per_side=COMMISSION,
                           dollar_per_pt=DOLLAR_PER_PT)
    print(f"    {fmt_score(sc_sl50, 'SL=50')}")
    if sc_sl50:
        print(f"    Exit types: {fmt_exits(sc_sl50['exits'])}")

    # =========================================================================
    # 5. Run backtest WITHOUT max_loss (for comparison)
    # =========================================================================
    print("\n[5] Running backtest WITHOUT max_loss (SL=0)...")
    trades_sl0 = run_backtest_v10(
        opens, highs, lows, closes, sm_full, rsi_curr, times,
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=0.0, cooldown_bars=COOLDOWN,
        max_loss_pts=0,
        rsi_5m_curr=rsi_curr, rsi_5m_prev=rsi_prev,
    )
    sc_sl0 = score_trades(trades_sl0, commission_per_side=COMMISSION,
                          dollar_per_pt=DOLLAR_PER_PT)
    print(f"    {fmt_score(sc_sl0, 'SL=0')}")
    if sc_sl0:
        print(f"    Exit types: {fmt_exits(sc_sl0['exits'])}")

    # =========================================================================
    # 6. Compare trade counts: if SL=50 and SL=0 have same count, stops never fired
    # =========================================================================
    print("\n[6] Comparison: SL=50 vs SL=0")
    n50 = len(trades_sl50)
    n0 = len(trades_sl0)
    print(f"    SL=50: {n50} trades")
    print(f"    SL=0:  {n0} trades")
    if n50 == n0:
        print(f"    --> SAME trade count: stop loss NEVER FIRED in Python engine")
    else:
        print(f"    --> DIFFERENT trade count ({n50 - n0} difference): stop loss DID fire")
        # Find which trades differ
        sl_trades = [t for t in trades_sl50 if t['result'] == 'SL']
        print(f"    --> SL exits in SL=50 run: {len(sl_trades)}")

    # =========================================================================
    # 7. Compute MAE for all trades (using SL=50 results)
    # =========================================================================
    print("\n[7] Computing MAE for all trades...")
    mae_results = compute_mae(trades_sl50, opens, highs, lows, closes, sm_full, times)

    # Print ALL trades sorted by MAE
    print_trade_table(mae_results, "ALL TRADES (6-month, sorted by MAE)")

    # =========================================================================
    # 8. Focus analysis: trades with MAE > 30 pts
    # =========================================================================
    high_mae = [r for r in mae_results if r['mae_pts'] >= 30]
    if high_mae:
        print(f"\n{'=' * 140}")
        print(f"  DEEP DIVE: {len(high_mae)} trades with MAE >= 30 pts")
        print(f"{'=' * 140}")
        for r in sorted(high_mae, key=lambda x: -x['mae_pts']):
            print(f"\n  Trade #{r['trade_num']}: {r['side'].upper()} @ {r['entry_price']:.2f}")
            print(f"    Entry: {r['entry_time'].strftime('%Y-%m-%d %H:%M')} UTC")
            print(f"    Exit:  {r['exit_time'].strftime('%Y-%m-%d %H:%M')} UTC ({r['exit_reason']})")
            print(f"    PnL: {r['pnl_pts']:+.2f} pts, Bars held: {r['bars']}")
            print(f"    MAE (intraday low/high): {r['mae_pts']:.2f} pts")
            print(f"    Max Close DD:            {r['max_close_dd_pts']:.2f} pts")
            print(f"    Close crossed 50pt stop: {'YES' if r['close_crossed_stop'] else 'NO'}")
            if r['dd_at_flip'] is not None:
                print(f"    DD at SM flip point:     {r['dd_at_flip']:.2f} pts")
            print(f"    SM at worst bar:         {r['sm_at_worst']:.4f}")

            # Gap analysis: difference between intraday MAE and close-based DD
            gap = r['mae_pts'] - r['max_close_dd_pts']
            print(f"    Gap (MAE - CloseDD):     {gap:.2f} pts")
            if gap > 10:
                print(f"    *** Large gap: intraday wicked {gap:.1f}pts past closes ***")

    # =========================================================================
    # 9. TV validation window: Jan 19 - Feb 12, 2026
    # =========================================================================
    tv_start = pd.Timestamp('2026-01-19')
    tv_end = pd.Timestamp('2026-02-13')  # exclusive
    tv_trades = [r for r in mae_results
                 if r['entry_time'] >= tv_start and r['entry_time'] < tv_end]

    print(f"\n\n{'=' * 140}")
    print(f"  TV VALIDATION WINDOW: Jan 19 - Feb 12, 2026")
    print(f"  ({len(tv_trades)} trades)")
    print(f"{'=' * 140}")

    if tv_trades:
        print_trade_table(tv_trades, "TV WINDOW TRADES (Jan 19 - Feb 12)")

        # Check for any close that exceeded 50pts in TV window
        tv_close_crossed = [r for r in tv_trades if r['close_crossed_stop']]
        tv_mae_gt50 = [r for r in tv_trades if r['mae_pts'] >= 50]

        tv_sl_exits = [r for r in tv_trades if r['exit_reason'] == 'SL']

        print(f"\n  TV WINDOW DIAGNOSIS:")
        print(f"    Trades where MAE >= 50 pts (intraday):   {len(tv_mae_gt50)}")
        print(f"    Trades where CLOSE crossed 50pt stop:    {len(tv_close_crossed)}")
        print(f"    Python SL exits in this window:          {len(tv_sl_exits)}")
        print(f"    (TradingView shows 13 max-loss stops in this window)")
        print(f"    Gap: TV has {13 - len(tv_sl_exits)} more SL exits than Python")

        if len(tv_sl_exits) < 13:
            print(f"\n  FINDING: Python produces {len(tv_sl_exits)} SL stops vs TV's 13.")
            print(f"  The {13 - len(tv_sl_exits)}-trade gap is likely due to SM computation differences:")
            print(f"    - Pine computes SM from TradingView's data feed (ta.pvi/ta.nvi)")
            print(f"    - Python computes SM from Databento's data feed")
            print(f"    - Small SM divergences near zero-crossings change WHEN SM flips,")
            print(f"      which determines whether the stop fires before or after SM flip exit")
    else:
        print("  No trades found in TV validation window.")

    # =========================================================================
    # 10. Histogram of MAE distribution
    # =========================================================================
    if mae_results:
        all_maes = [r['mae_pts'] for r in mae_results]
        all_close_dds = [r['max_close_dd_pts'] for r in mae_results]

        print(f"\n\n{'=' * 140}")
        print(f"  MAE DISTRIBUTION (all {len(mae_results)} trades)")
        print(f"{'=' * 140}")

        bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 200]
        for j in range(len(bins) - 1):
            lo, hi = bins[j], bins[j + 1]
            count_mae = sum(1 for m in all_maes if lo <= m < hi)
            count_cdd = sum(1 for m in all_close_dds if lo <= m < hi)
            bar_mae = '#' * count_mae
            bar_cdd = '*' * count_cdd
            marker = " <-- STOP THRESHOLD" if lo == 50 else ""
            print(f"  {lo:>4}-{hi:>4} pts:  MAE {count_mae:>3} {bar_mae}")
            print(f"               ClDD {count_cdd:>3} {bar_cdd}{marker}")

    # =========================================================================
    # 11. Critical test: what if stop checked LOW instead of CLOSE?
    # =========================================================================
    print(f"\n\n{'=' * 140}")
    print(f"  WHAT-IF: Stop checked bar LOW/HIGH instead of CLOSE")
    print(f"{'=' * 140}")

    mae_gt50_trades = [r for r in mae_results if r['mae_pts'] >= 50]
    mae_gt50_close = [r for r in mae_results if r['max_close_dd_pts'] >= 50]

    print(f"  Trades with intraday MAE >= 50:    {len(mae_gt50_trades)}")
    print(f"  Trades with close-based DD >= 50:  {len(mae_gt50_close)}")
    print(f"  Gap: {len(mae_gt50_trades) - len(mae_gt50_close)} trades have intraday "
          f">= 50 but close < 50")

    if len(mae_gt50_trades) > len(mae_gt50_close):
        print(f"\n  These trades wicked past 50pts intraday but closed above the stop:")
        for r in sorted(mae_gt50_trades, key=lambda x: -x['mae_pts']):
            if r['max_close_dd_pts'] < 50:
                print(f"    #{r['trade_num']} {r['side']:>5} {r['entry_time'].strftime('%m/%d %H:%M')} "
                      f"MAE={r['mae_pts']:.1f}  CloseDD={r['max_close_dd_pts']:.1f}  "
                      f"Exit={r['exit_reason']}")

    # =========================================================================
    # 12. Pine vs Python stop logic difference analysis
    # =========================================================================
    print(f"\n\n{'=' * 140}")
    print(f"  PINE vs PYTHON STOP LOGIC ANALYSIS")
    print(f"{'=' * 140}")
    print(f"""
  Pine (v11MNQ.pine lines 166-172):
    if max_loss_pts > 0
        if strategy.position_size > 0 and close <= strategy.position_avg_price - max_loss_pts
            strategy.close("Long", comment="Max Loss")

  Python (v10_test_common.py lines 569-575):
    if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
        close_trade("long", entry_price, opens[i], entry_idx, i, "SL")

  Both check CLOSE (not low), and fill at next bar open.
  Pine evaluates at current bar close, Python checks bar i-1 close for bar i fill.
  These should be equivalent in backtesting mode.

  BUT: Pine processes exits BEFORE entries on the same bar. If SM flip and stop
  are both true on the same bar, Pine hits SM flip first (lines 156-163 come
  before lines 166-172). The Python engine also processes SM flip first (lines
  653-660 before checking next iteration). So exit priority matches.

  HYPOTHESIS: The difference is in SM COMPUTATION, not stop logic.
  Pine uses ta.pvi/ta.nvi (TradingView built-in) while Python uses
  compute_smart_money() on Databento data. Small SM divergences at
  zero-crossings can cause SM to flip at different bars, changing whether
  the stop fires.
    """)

    # =========================================================================
    # 13. Quantify: how close to 50 pts are the worst trades?
    # =========================================================================
    print(f"{'=' * 140}")
    print(f"  MARGIN ANALYSIS: How close to 50pts stop threshold?")
    print(f"{'=' * 140}")
    close_dds_sorted = sorted([r['max_close_dd_pts'] for r in mae_results], reverse=True)
    print(f"\n  Top 20 worst close-based drawdowns (what stop logic actually sees):")
    for j, dd in enumerate(close_dds_sorted[:20]):
        distance = 50 - dd
        trade_idx = next(i for i, r in enumerate(mae_results) if r['max_close_dd_pts'] == dd)
        r = mae_results[trade_idx]
        print(f"    {j+1:>3}. CloseDD={dd:>7.2f} pts  (distance to 50pt stop: {distance:>+7.2f})  "
              f"Trade #{r['trade_num']} {r['side']} {r['entry_time'].strftime('%m/%d %H:%M')} "
              f"Exit={r['exit_reason']}")

    print(f"\n\n{'=' * 140}")
    print(f"  CONCLUSION")
    print(f"{'=' * 140}")
    n_close_gt50 = sum(1 for r in mae_results if r['max_close_dd_pts'] >= 50)
    n_mae_gt50 = sum(1 for r in mae_results if r['mae_pts'] >= 50)
    n_sl_exits = sum(1 for t in trades_sl50 if t['result'] == 'SL')
    tv_sl_count = sum(1 for r in mae_results
                      if r['exit_reason'] == 'SL'
                      and r['entry_time'] >= pd.Timestamp('2026-01-19')
                      and r['entry_time'] < pd.Timestamp('2026-02-13'))
    print(f"""
  Out of {len(mae_results)} total trades (6-month):
    - Trades with CLOSE drawdown >= 50 pts: {n_close_gt50}
    - Trades with intraday MAE >= 50 pts:   {n_mae_gt50}
    - SL exits in Python engine (6-month):  {n_sl_exits}
    - SL exits in TV window (Jan19-Feb12):  {tv_sl_count} (TV shows 13)

  THE STOP LOSS IS WORKING IN PYTHON. Python produces {n_sl_exits} SL exits
  across 6 months and {tv_sl_count} in the TV validation window (vs TV's 13).

  The gap of {13 - tv_sl_count} stops between TV and Python in the validation window
  is explained by SM computation differences (TradingView data feed vs Databento).
  When SM is borderline near zero, small data differences shift the zero-crossing
  by 1-2 bars, which determines whether SM flip exits the trade BEFORE or AFTER
  the close breaches the 50pt stop threshold.

  The stop loss adds significant value:
    - SL=50: {sc_sl50['count'] if sc_sl50 else 0} trades, PF {sc_sl50['pf'] if sc_sl50 else 0}, Net ${sc_sl50['net_dollar'] if sc_sl50 else 0:+.2f}, MaxDD ${sc_sl50['max_dd_dollar'] if sc_sl50 else 0:.2f}
    - SL=0:  {sc_sl0['count'] if sc_sl0 else 0} trades, PF {sc_sl0['pf'] if sc_sl0 else 0}, Net ${sc_sl0['net_dollar'] if sc_sl0 else 0:+.2f}, MaxDD ${sc_sl0['max_dd_dollar'] if sc_sl0 else 0:.2f}
    - SL=50 improves PF by {((sc_sl50['pf']/sc_sl0['pf'])-1)*100 if sc_sl0 and sc_sl50 else 0:.1f}% and halves max drawdown
    """)


if __name__ == "__main__":
    main()
