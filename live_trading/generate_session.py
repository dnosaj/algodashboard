"""
Generate a session JSON file from backtest data for dashboard replay.

Runs vScalpA + vScalpB (MNQ) and MES v2 strategies on Databento 1-min bars
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

# --- MNQ SM params (shared by vScalpA and vScalpB) ---
MNQ_SM_INDEX = 10
MNQ_SM_FLOW = 12
MNQ_SM_NORM = 200
MNQ_SM_EMA = 100
MNQ_DOLLAR_PER_PT = 2.0
MNQ_COMMISSION = 0.52

# vScalpA (v15) params
VSCALPA_RSI_LEN = 8
VSCALPA_RSI_BUY = 60
VSCALPA_RSI_SELL = 40
VSCALPA_SM_THRESHOLD = 0.0
VSCALPA_COOLDOWN = 20
VSCALPA_MAX_LOSS_PTS = 50
VSCALPA_TP_PTS = 5

# vScalpB params
VSCALPB_RSI_LEN = 8
VSCALPB_RSI_BUY = 55
VSCALPB_RSI_SELL = 45
VSCALPB_SM_THRESHOLD = 0.25
VSCALPB_COOLDOWN = 20
VSCALPB_MAX_LOSS_PTS = 15
VSCALPB_TP_PTS = 5

# --- MES SM params ---
MES_SM_INDEX = 20
MES_SM_FLOW = 12
MES_SM_NORM = 400
MES_SM_EMA = 255
MES_DOLLAR_PER_PT = 5.0
MES_COMMISSION = 1.25

# MES v2 params
MESV2_RSI_LEN = 12
MESV2_RSI_BUY = 55
MESV2_RSI_SELL = 45
MESV2_SM_THRESHOLD = 0.0
MESV2_COOLDOWN = 25
MESV2_MAX_LOSS_PTS = 35
MESV2_TP_PTS = 20
MESV2_EOD_ET = 15 * 60 + 30  # 15:30 ET = 930 minutes


def run_backtest_tp_exit(opens, highs, lows, closes, sm, times,
                         rsi_5m_curr, rsi_5m_prev,
                         rsi_buy, rsi_sell, sm_threshold,
                         cooldown_bars, max_loss_pts, tp_pts,
                         eod_minutes_et=NY_CLOSE_ET):
    """v15-style backtest: same entries as v10, but TP exit instead of SM flip.

    Exit priority:
      1. SL: prev bar close breaches max_loss_pts -> fill at next open
      2. TP: prev bar close reaches tp_pts profit -> fill at next open
      3. EOD: eod_minutes_et -> fill at close
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
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
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


def load_instrument_1min(instrument: str) -> pd.DataFrame:
    """Load and concatenate all Databento 1-min files for an instrument."""
    files = sorted(DATA_DIR.glob(f"databento_{instrument}_1min_*.csv"))
    if not files:
        print(f"WARNING: No Databento {instrument} 1-min files found in {DATA_DIR}")
        return pd.DataFrame()

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
    return combined


def load_all_mnq_1min() -> pd.DataFrame:
    """Load MNQ data and compute SM. Backwards-compatible wrapper."""
    combined = load_instrument_1min("MNQ")
    sm = compute_smart_money(
        combined['Close'].values, combined['Volume'].values,
        index_period=MNQ_SM_INDEX, flow_period=MNQ_SM_FLOW,
        norm_period=MNQ_SM_NORM, ema_len=MNQ_SM_EMA,
    )
    combined['SM_Net'] = sm
    print(f"Loaded MNQ: {len(combined)} bars from {combined.index[0]} to {combined.index[-1]}")
    return combined


def run_session(df_mnq: pd.DataFrame, target_dates: list[str],
                df_mes: pd.DataFrame = None) -> dict:
    """Run vScalpA + vScalpB + MES v2 backtests and extract bars + trades."""

    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")

    target_set = set(target_dates)

    def prepare_arrays(df, rsi_len):
        """Prepare 1-min arrays and 5-min RSI mapping for a dataframe."""
        df_5m = resample_to_5min(df)
        rsi_curr, rsi_prev = map_5min_rsi_to_1min(
            df.index.values, df_5m.index.values, df_5m['Close'].values,
            rsi_len=rsi_len,
        )
        return rsi_curr, rsi_prev

    def format_trades(raw_trades, strategy_id, instrument, dollar_per_pt, commission):
        """Format raw trade dicts into session JSON format."""
        result = []
        times = df_mnq.index if instrument == "MNQ" else df_mes.index
        for t in raw_trades:
            exit_ts = pd.Timestamp(t['exit_time'])
            if exit_ts.tz is None:
                exit_ts = exit_ts.tz_localize('UTC')
            exit_et = exit_ts.tz_convert('America/New_York')
            if exit_et.strftime('%Y-%m-%d') in target_set:
                entry_ts = pd.Timestamp(t['entry_time'])
                if entry_ts.tz is None:
                    entry_ts = entry_ts.tz_localize('UTC')
                pnl_pts = t['pts']
                pnl_dollar = pnl_pts * dollar_per_pt - 2 * commission

                entry_et = entry_ts.tz_convert(_ET)
                entry_et_epoch = int(entry_ts.timestamp()) + int(entry_et.utcoffset().total_seconds())
                exit_et_epoch = int(exit_ts.timestamp()) + int(exit_et.utcoffset().total_seconds())

                result.append({
                    "instrument": instrument,
                    "strategy_id": strategy_id,
                    "side": t['side'].upper(),
                    "entry_price": float(t['entry']),
                    "exit_price": float(t['exit']),
                    "entry_time": str(entry_ts.isoformat()),
                    "exit_time": str(exit_ts.isoformat()),
                    "entry_time_et_epoch": entry_et_epoch,
                    "exit_time_et_epoch": exit_et_epoch,
                    "pts": float(pnl_pts),
                    "pnl": float(pnl_dollar),
                    "exit_reason": t.get('result', 'TP'),
                    "bars_held": int(t['bars']),
                    "mfe": round(t.get('mfe', 0.0), 2),
                    "mae": round(t.get('mae', 0.0), 2),
                })
        return result

    def extract_bars(df, instrument):
        """Extract bar data for target dates."""
        et_dates = pd.DatetimeIndex(df.index).tz_localize('UTC').tz_convert('America/New_York')
        bar_dates = et_dates.strftime('%Y-%m-%d')
        mask = np.isin(bar_dates, list(target_set))
        bars = []
        for i in np.where(mask)[0]:
            utc_ts = pd.Timestamp(df.index[i])
            if utc_ts.tzinfo is None:
                utc_ts = utc_ts.tz_localize('UTC')
            et_ts = utc_ts.tz_convert(_ET)
            offset_seconds = int(et_ts.utcoffset().total_seconds())
            bars.append({
                "time": int(utc_ts.timestamp()) + offset_seconds,
                "open": float(df['Open'].values[i]),
                "high": float(df['High'].values[i]),
                "low": float(df['Low'].values[i]),
                "close": float(df['Close'].values[i]),
                "volume": float(df['Volume'].values[i]),
            })
        return bars

    # --- MNQ strategies ---
    mnq_opens = df_mnq['Open'].values
    mnq_highs = df_mnq['High'].values
    mnq_lows = df_mnq['Low'].values
    mnq_closes = df_mnq['Close'].values
    mnq_sm = df_mnq['SM_Net'].values
    mnq_times = df_mnq.index

    # vScalpA RSI (len=8)
    rsi_a_curr, rsi_a_prev = prepare_arrays(df_mnq, VSCALPA_RSI_LEN)

    # vScalpA backtest
    vscalpa_trades = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm, mnq_times,
        rsi_a_curr, rsi_a_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD, cooldown_bars=VSCALPA_COOLDOWN,
        max_loss_pts=VSCALPA_MAX_LOSS_PTS, tp_pts=VSCALPA_TP_PTS,
    )
    compute_mfe_mae(vscalpa_trades, mnq_highs, mnq_lows)

    # vScalpB RSI (also len=8, same mapping)
    vscalpb_trades = run_backtest_tp_exit(
        mnq_opens, mnq_highs, mnq_lows, mnq_closes, mnq_sm, mnq_times,
        rsi_a_curr, rsi_a_prev,  # Same RSI len, reuse
        rsi_buy=VSCALPB_RSI_BUY, rsi_sell=VSCALPB_RSI_SELL,
        sm_threshold=VSCALPB_SM_THRESHOLD, cooldown_bars=VSCALPB_COOLDOWN,
        max_loss_pts=VSCALPB_MAX_LOSS_PTS, tp_pts=VSCALPB_TP_PTS,
    )
    compute_mfe_mae(vscalpb_trades, mnq_highs, mnq_lows)

    # Format MNQ trades
    all_trades = []
    all_trades.extend(format_trades(vscalpa_trades, "MNQ_V15", "MNQ",
                                     MNQ_DOLLAR_PER_PT, MNQ_COMMISSION))
    all_trades.extend(format_trades(vscalpb_trades, "MNQ_VSCALPB", "MNQ",
                                     MNQ_DOLLAR_PER_PT, MNQ_COMMISSION))

    # Extract MNQ bars
    bars_dict = {"MNQ": extract_bars(df_mnq, "MNQ")}

    # --- MES v2 strategy ---
    if df_mes is not None and len(df_mes) > 0:
        mes_sm = compute_smart_money(
            df_mes['Close'].values, df_mes['Volume'].values,
            index_period=MES_SM_INDEX, flow_period=MES_SM_FLOW,
            norm_period=MES_SM_NORM, ema_len=MES_SM_EMA,
        )
        df_mes = df_mes.copy()
        df_mes['SM_Net'] = mes_sm

        mes_opens = df_mes['Open'].values
        mes_highs = df_mes['High'].values
        mes_lows = df_mes['Low'].values
        mes_closes = df_mes['Close'].values
        mes_times = df_mes.index

        rsi_mes_curr, rsi_mes_prev = prepare_arrays(df_mes, MESV2_RSI_LEN)

        mesv2_trades = run_backtest_tp_exit(
            mes_opens, mes_highs, mes_lows, mes_closes, mes_sm, mes_times,
            rsi_mes_curr, rsi_mes_prev,
            rsi_buy=MESV2_RSI_BUY, rsi_sell=MESV2_RSI_SELL,
            sm_threshold=MESV2_SM_THRESHOLD, cooldown_bars=MESV2_COOLDOWN,
            max_loss_pts=MESV2_MAX_LOSS_PTS, tp_pts=MESV2_TP_PTS,
            eod_minutes_et=MESV2_EOD_ET,
        )
        compute_mfe_mae(mesv2_trades, mes_highs, mes_lows)

        all_trades.extend(format_trades(mesv2_trades, "MES_V2", "MES",
                                         MES_DOLLAR_PER_PT, MES_COMMISSION))
        bars_dict["MES"] = extract_bars(df_mes, "MES")

    # Sort all trades by entry time
    all_trades.sort(key=lambda t: t['entry_time'])

    date_label = target_dates[0] if len(target_dates) == 1 else f"{target_dates[0]}_to_{target_dates[-1]}"

    session = {
        "date": date_label,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "timezone": "ET",
        "bars": bars_dict,
        "trades": all_trades,
    }

    return session


def main():
    parser = argparse.ArgumentParser(description="Generate session file from backtest")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="Number of recent days")
    args = parser.parse_args()

    # Load MNQ
    df_mnq = load_all_mnq_1min()

    # Load MES
    df_mes = load_instrument_1min("MES")
    if len(df_mes) > 0:
        print(f"Loaded MES: {len(df_mes)} bars from {df_mes.index[0]} to {df_mes.index[-1]}")
    else:
        print("No MES data found, running MNQ-only session")
        df_mes = None

    # Determine target dates (use MNQ dates as reference)
    et_dates = pd.DatetimeIndex(df_mnq.index).tz_localize('UTC').tz_convert('America/New_York')
    unique_dates = sorted(set(et_dates.strftime('%Y-%m-%d')))

    if args.date:
        target_dates = [args.date]
    else:
        target_dates = unique_dates[-args.days:]

    print(f"Target dates: {target_dates}")

    session = run_session(df_mnq, target_dates, df_mes)

    bar_count = sum(len(b) for b in session['bars'].values())
    trade_count = len(session['trades'])
    va_count = sum(1 for t in session['trades'] if t['strategy_id'] == 'MNQ_V15')
    vb_count = sum(1 for t in session['trades'] if t['strategy_id'] == 'MNQ_VSCALPB')
    mes_count = sum(1 for t in session['trades'] if t['strategy_id'] == 'MES_V2')
    print(f"Result: {bar_count} bars, {trade_count} trades "
          f"(vScalpA: {va_count}, vScalpB: {vb_count}, MES v2: {mes_count})")

    # Print trade summary
    if trade_count > 0:
        total_pnl = sum(t['pnl'] for t in session['trades'])
        winners = sum(1 for t in session['trades'] if t['pnl'] > 0)
        print(f"  P&L: ${total_pnl:+.2f}  W/L: {winners}/{trade_count - winners}")
        for t in session['trades']:
            sid = t['strategy_id']
            label = "vA" if "V15" in sid else "vB" if "VSCALPB" in sid else "MES"
            print(f"    [{label:3s}] {t['instrument']} {t['side']:5s} @ {t['entry_price']:.2f} -> "
                  f"{t['exit_price']:.2f}  {t['pts']:+.2f}pt  ${t['pnl']:+.2f}  "
                  f"MFE={t['mfe']:.1f} MAE={t['mae']:.1f}  [{t['exit_reason']}]")

    # Save
    date_label = session['date']
    filename = f"session_{date_label}.json"
    filepath = SESSIONS_DIR / filename
    filepath.write_text(json.dumps(session, indent=2))
    print(f"\nSaved: {filepath}")


if __name__ == "__main__":
    main()
