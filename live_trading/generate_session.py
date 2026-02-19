"""
Generate a session JSON file from backtest data for dashboard replay.

Runs v11.1 (SM flip) and v15 (TP=5) MNQ strategies on Databento 1-min bars
and saves bars + trades in the session format the dashboard expects.

Usage:
    python3 generate_session.py                    # latest day in data
    python3 generate_session.py --date 2026-02-19  # specific date
    python3 generate_session.py --days 3           # last N days
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add backtesting engine to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backtesting_engine"))
from strategies.v10_test_common import (
    compute_rsi,
    compute_smart_money,
    map_5min_rsi_to_1min,
    resample_to_5min,
    run_backtest_v10,
    compute_et_minutes,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "backtesting_engine" / "data"
SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# Shared SM params (v11 and v15 use the same)
SM_INDEX = 10
SM_FLOW = 12
SM_NORM = 200
SM_EMA = 100
RSI_LEN = 8
RSI_BUY = 60
RSI_SELL = 40
COOLDOWN = 20
MAX_LOSS_PTS = 50
DOLLAR_PER_PT = 2.0
COMMISSION = 0.52

# v11.1-specific
V11_SM_THRESHOLD = 0.15

# v15-specific
V15_SM_THRESHOLD = 0.0  # No threshold for v15
V15_TP_PTS = 5


def run_backtest_tp_exit(opens, highs, lows, closes, sm, times,
                         rsi_5m_curr, rsi_5m_prev,
                         rsi_buy, rsi_sell, sm_threshold,
                         cooldown_bars, max_loss_pts, tp_pts):
    """v15-style backtest: same entries as v10, but TP exit instead of SM flip.

    Exit priority:
      1. SL: prev bar close breaches max_loss_pts -> fill at next open
      2. TP: prev bar close reaches tp_pts profit -> fill at next open
      3. EOD: 4 PM ET -> fill at close
    No SM flip exit.
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

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

        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI cross from mapped 5-min
        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        # Episode reset (uses zero crossing, not threshold)
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # EOD close
        if trade_state != 0 and bar_mins_et >= NY_CLOSE_ET:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # Exits for open positions
        if trade_state == 1:
            # SL: prev bar close breaches stop
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # TP: prev bar close reached TP target
            if tp_pts > 0 and closes[i - 1] >= entry_price + tp_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

        elif trade_state == -1:
            # SL
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # TP
            if tp_pts > 0 and closes[i - 1] <= entry_price - tp_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

        # Entries
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True

    return trades


def compute_mfe_mae(trades, highs, lows):
    """Add MFE/MAE (in points) to each trade dict."""
    for t in trades:
        entry_i = t['entry_idx']
        exit_i = t['exit_idx']
        entry_p = t['entry']

        if exit_i <= entry_i:
            t['mfe'] = 0.0
            t['mae'] = 0.0
            continue

        bar_highs = highs[entry_i:exit_i + 1]
        bar_lows = lows[entry_i:exit_i + 1]

        if t['side'] == 'long':
            t['mfe'] = float(np.max(bar_highs) - entry_p)
            t['mae'] = float(entry_p - np.min(bar_lows))
        else:
            t['mfe'] = float(entry_p - np.min(bar_lows))
            t['mae'] = float(np.max(bar_highs) - entry_p)


def load_all_mnq_1min() -> pd.DataFrame:
    """Load and concatenate all Databento MNQ 1-min files."""
    files = sorted(DATA_DIR.glob("databento_MNQ_1min_*.csv"))
    if not files:
        print("ERROR: No Databento MNQ 1-min files found in", DATA_DIR)
        sys.exit(1)

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
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()

    # Compute SM on 1-min
    sm = compute_smart_money(
        combined['Close'].values, combined['Volume'].values,
        index_period=SM_INDEX, flow_period=SM_FLOW,
        norm_period=SM_NORM, ema_len=SM_EMA,
    )
    combined['SM_Net'] = sm

    print(f"Loaded {len(combined)} bars from {len(files)} files")
    print(f"  Range: {combined.index[0]} to {combined.index[-1]}")
    return combined


def run_session(df_1m: pd.DataFrame, target_dates: list[str]) -> dict:
    """Run v11 + v15 backtests and extract bars + trades for target dates."""

    # Resample to 5-min for RSI mapping
    df_5m = resample_to_5min(df_1m)
    fivemin_times = df_5m.index.values
    fivemin_closes = df_5m['Close'].values

    # Prepare 1-min arrays
    onemin_times = df_1m.index.values
    rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
        onemin_times, fivemin_times, fivemin_closes, rsi_len=RSI_LEN,
    )

    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    sm = df_1m['SM_Net'].values
    times = df_1m.index

    # Dummy RSI (not used when rsi_5m_curr/prev provided)
    rsi_dummy = np.full(len(closes), 50.0)

    # --- v11 backtest (SM flip exit, threshold=0.15) ---
    v11_trades = run_backtest_v10(
        opens, highs, lows, closes, sm, rsi_dummy, times,
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=V11_SM_THRESHOLD, cooldown_bars=COOLDOWN,
        max_loss_pts=MAX_LOSS_PTS,
        rsi_5m_curr=rsi_5m_curr, rsi_5m_prev=rsi_5m_prev,
    )
    compute_mfe_mae(v11_trades, highs, lows)

    # --- v15 backtest (TP exit, threshold=0.0) ---
    v15_trades = run_backtest_tp_exit(
        opens, highs, lows, closes, sm, times,
        rsi_5m_curr, rsi_5m_prev,
        rsi_buy=RSI_BUY, rsi_sell=RSI_SELL,
        sm_threshold=V15_SM_THRESHOLD, cooldown_bars=COOLDOWN,
        max_loss_pts=MAX_LOSS_PTS, tp_pts=V15_TP_PTS,
    )
    compute_mfe_mae(v15_trades, highs, lows)

    # Filter bars and trades for target dates
    target_set = set(target_dates)

    # Get bar indices for target dates
    et_dates = pd.DatetimeIndex(times).tz_localize('UTC').tz_convert('America/New_York')
    bar_dates = et_dates.strftime('%Y-%m-%d')

    mask = np.isin(bar_dates, list(target_set))
    filtered_bars = []
    for i in np.where(mask)[0]:
        filtered_bars.append({
            "time": int(pd.Timestamp(times[i]).timestamp()),
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
            "volume": float(df_1m['Volume'].values[i]),
        })

    def format_trades(raw_trades, strategy_id):
        result = []
        for t in raw_trades:
            exit_ts = pd.Timestamp(t['exit_time'])
            if exit_ts.tz is None:
                exit_ts = exit_ts.tz_localize('UTC')
            exit_et = exit_ts.tz_convert('America/New_York')
            if exit_et.strftime('%Y-%m-%d') in target_set:
                entry_ts = pd.Timestamp(t['entry_time'])
                pnl_pts = t['pts']
                pnl_dollar = pnl_pts * DOLLAR_PER_PT - 2 * COMMISSION

                result.append({
                    "instrument": "MNQ",
                    "strategy_id": strategy_id,
                    "side": t['side'].upper(),
                    "entry_price": float(t['entry']),
                    "exit_price": float(t['exit']),
                    "entry_time": str(entry_ts.isoformat()),
                    "exit_time": str(exit_ts.isoformat()),
                    "pts": float(pnl_pts),
                    "pnl": float(pnl_dollar),
                    "exit_reason": t.get('result', 'SM_FLIP'),
                    "bars_held": int(t['bars']),
                    "mfe": round(t.get('mfe', 0.0), 2),
                    "mae": round(t.get('mae', 0.0), 2),
                })
        return result

    filtered_v11 = format_trades(v11_trades, "MNQ_V11")
    filtered_v15 = format_trades(v15_trades, "MNQ_V15")

    # Combine and sort by entry time
    all_trades = filtered_v11 + filtered_v15
    all_trades.sort(key=lambda t: t['entry_time'])

    date_label = target_dates[0] if len(target_dates) == 1 else f"{target_dates[0]}_to_{target_dates[-1]}"

    session = {
        "date": date_label,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "bars": {"MNQ": filtered_bars},
        "trades": all_trades,
    }

    return session


def main():
    parser = argparse.ArgumentParser(description="Generate session file from backtest")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="Number of recent days")
    args = parser.parse_args()

    df = load_all_mnq_1min()

    # Determine target dates
    et_dates = pd.DatetimeIndex(df.index).tz_localize('UTC').tz_convert('America/New_York')
    unique_dates = sorted(set(et_dates.strftime('%Y-%m-%d')))

    if args.date:
        target_dates = [args.date]
    else:
        target_dates = unique_dates[-args.days:]

    print(f"Target dates: {target_dates}")

    session = run_session(df, target_dates)

    bar_count = sum(len(b) for b in session['bars'].values())
    trade_count = len(session['trades'])
    v11_count = sum(1 for t in session['trades'] if t['strategy_id'] == 'MNQ_V11')
    v15_count = sum(1 for t in session['trades'] if t['strategy_id'] == 'MNQ_V15')
    print(f"Result: {bar_count} bars, {trade_count} trades (v11: {v11_count}, v15: {v15_count})")

    # Print trade summary
    if trade_count > 0:
        total_pnl = sum(t['pnl'] for t in session['trades'])
        winners = sum(1 for t in session['trades'] if t['pnl'] > 0)
        print(f"  P&L: ${total_pnl:+.2f}  W/L: {winners}/{trade_count - winners}")
        for t in session['trades']:
            strat = "v11" if "V11" in t['strategy_id'] else "v15"
            print(f"    [{strat}] {t['side']:5s} @ {t['entry_price']:.2f} -> {t['exit_price']:.2f}  "
                  f"{t['pts']:+.2f}pt  ${t['pnl']:+.2f}  MFE={t['mfe']:.1f} MAE={t['mae']:.1f}  "
                  f"[{t['exit_reason']}]")

    # Save
    date_label = session['date']
    filename = f"session_{date_label}.json"
    filepath = SESSIONS_DIR / filename
    filepath.write_text(json.dumps(session, indent=2))
    print(f"\nSaved: {filepath}")


if __name__ == "__main__":
    main()
