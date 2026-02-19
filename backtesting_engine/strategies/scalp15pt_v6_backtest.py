"""
15-Point Scalper v6 — OUR SM calculation (not AlgoAlpha)

Tests two approaches on the same 79-day 5-min dataset:
  A) AlgoAlpha's pre-baked 1m SM values from CSV
  B) Our own SM computed from 1-min OHLC data

This tells us if our SM is just as good (or better) than AlgoAlpha's.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# ─── SM Computation (our version) ───────────────────────────────────────────

def compute_ema(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def compute_rsi_arr(arr, period):
    n = len(arr)
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    if n > period:
        avg_gain[period] = np.mean(gain[1:period + 1])
        avg_loss[period] = np.mean(loss[1:period + 1])
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = 50.0
    return rsi


def compute_smart_money(close, volume, index_period, flow_period, norm_period, ema_len):
    """Our SM calculation: PVI/NVI from volume, RSI on flows, ratio-based index."""
    n = len(close)
    pvi = np.ones(n)
    nvi = np.ones(n)

    pct_change = np.zeros(n)
    pct_change[1:] = (close[1:] - close[:-1]) / close[:-1]

    for i in range(1, n):
        if volume[i] > volume[i - 1]:
            pvi[i] = pvi[i - 1] + pct_change[i] * pvi[i - 1]
            nvi[i] = nvi[i - 1]
        elif volume[i] < volume[i - 1]:
            nvi[i] = nvi[i - 1] + pct_change[i] * nvi[i - 1]
            pvi[i] = pvi[i - 1]
        else:
            pvi[i] = pvi[i - 1]
            nvi[i] = nvi[i - 1]

    dumb = pvi - compute_ema(pvi, ema_len)
    smart = nvi - compute_ema(nvi, ema_len)

    drsi = compute_rsi_arr(dumb, flow_period)
    srsi = compute_rsi_arr(smart, flow_period)

    r_buy = np.where(drsi != 0, srsi / drsi, 0.0)
    r_sell = np.where((100 - drsi) != 0, (100 - srsi) / (100 - drsi), 0.0)

    sums_buy = pd.Series(r_buy).rolling(index_period, min_periods=1).sum().values
    sums_sell = pd.Series(r_sell).rolling(index_period, min_periods=1).sum().values

    combined_max = np.maximum(sums_buy, sums_sell)
    peak = pd.Series(combined_max).rolling(norm_period, min_periods=1).max().values
    peak = np.where(peak > 0, peak, 1.0)

    index_buy = sums_buy / peak
    index_sell = sums_sell / peak
    return index_buy - index_sell


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_1min_data(filename):
    """Load 1-min OHLCV for computing our own SM."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close",
    })
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close"]].copy()
    # Use range as volume proxy if no volume column, otherwise try volume
    if "volume" in [c.lower() for c in df.columns]:
        pass
    else:
        df["Volume"] = df["High"] - df["Low"]
    return df


def load_5min_with_indicators(filename):
    """Load 5-min TV export with pre-baked indicators."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close",
        cols[7]: "SM_1m_AA",   # AlgoAlpha 1-min SM
        cols[8]: "SM_5m_AA",   # AlgoAlpha 5-min SM
    })
    df["RSI_TV"] = pd.to_numeric(df[cols[9]], errors="coerce")
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "SM_1m_AA", "SM_5m_AA", "RSI_TV"]].copy()
    df["SM_1m_AA"] = pd.to_numeric(df["SM_1m_AA"], errors="coerce").fillna(0)
    df["SM_5m_AA"] = pd.to_numeric(df["SM_5m_AA"], errors="coerce").fillna(0)
    df["RSI_TV"] = df["RSI_TV"].fillna(50)
    return df


def resample_1m_sm_to_5m(sm_1m_series, df_5m):
    """Take 1-min SM and get the last value at each 5-min bar close."""
    # Resample to 5-min taking the last value in each window
    sm_5m = sm_1m_series.resample("5min").last()
    # Align to the 5-min index
    sm_aligned = sm_5m.reindex(df_5m.index, method="ffill").fillna(0)
    return sm_aligned.values


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(opens, highs, lows, closes, sm_1m, rsi, times,
                 rsi_buy, rsi_sell, sm_threshold, tp, sl, cooldown_bars):
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    NY_OPEN_UTC = 15 * 60
    NY_LAST_ENTRY_UTC = 20 * 60 + 45
    NY_CLOSE_UTC = 21 * 60

    for i in range(1, n):
        bar_ts = pd.Timestamp(times[i])
        bar_mins_utc = bar_ts.hour * 60 + bar_ts.minute

        rsi_in_buy = rsi[i] > rsi_buy
        rsi_in_sell = rsi[i] < rsi_sell
        sm_bull = sm_1m[i] > sm_threshold
        sm_bear = sm_1m[i] < -sm_threshold

        if not rsi_in_buy or not sm_bull:
            long_used = False
        if not rsi_in_sell or not sm_bear:
            short_used = False

        if trade_state != 0 and bar_mins_utc >= NY_CLOSE_UTC:
            bars_held = i - entry_idx
            pts = (closes[i] - entry_price) if trade_state == 1 else (entry_price - closes[i])
            trades.append({"side": "long" if trade_state == 1 else "short",
                           "entry": entry_price, "exit": closes[i], "pts": pts,
                           "entry_time": times[entry_idx], "exit_time": times[i],
                           "bars": bars_held, "result": "EOD"})
            trade_state = 0
            exit_bar = i
            continue

        if trade_state == 1:
            bars_held = i - entry_idx
            if lows[i] <= entry_price - sl:
                trades.append({"side": "long", "entry": entry_price, "exit": entry_price - sl,
                               "pts": -sl, "entry_time": times[entry_idx],
                               "exit_time": times[i], "bars": bars_held, "result": "SL"})
                trade_state = 0; exit_bar = i
            elif highs[i] >= entry_price + tp:
                trades.append({"side": "long", "entry": entry_price, "exit": entry_price + tp,
                               "pts": tp, "entry_time": times[entry_idx],
                               "exit_time": times[i], "bars": bars_held, "result": "TP"})
                trade_state = 0; exit_bar = i
        elif trade_state == -1:
            bars_held = i - entry_idx
            if highs[i] >= entry_price + sl:
                trades.append({"side": "short", "entry": entry_price, "exit": entry_price + sl,
                               "pts": -sl, "entry_time": times[entry_idx],
                               "exit_time": times[i], "bars": bars_held, "result": "SL"})
                trade_state = 0; exit_bar = i
            elif lows[i] <= entry_price - tp:
                trades.append({"side": "short", "entry": entry_price, "exit": entry_price - tp,
                               "pts": tp, "entry_time": times[entry_idx],
                               "exit_time": times[i], "bars": bars_held, "result": "TP"})
                trade_state = 0; exit_bar = i

        if trade_state == 0:
            bars_since_exit = i - exit_bar
            in_session = NY_OPEN_UTC <= bar_mins_utc <= NY_LAST_ENTRY_UTC
            cooldown_ok = bars_since_exit >= cooldown_bars
            if in_session and cooldown_ok:
                if rsi_in_buy and sm_bull and not long_used:
                    trade_state = 1; entry_price = opens[i]; entry_idx = i; long_used = True
                elif rsi_in_sell and sm_bear and not short_used:
                    trade_state = -1; entry_price = opens[i]; entry_idx = i; short_used = True

    return trades


def score_trades(trades, commission_per_side=0.52):
    if not trades:
        return None
    pts = np.array([t["pts"] for t in trades])
    n = len(pts)
    comm_pts = (commission_per_side * 2) / 2.0
    net_pts_each = pts - comm_pts
    net_pts = net_pts_each.sum()
    wins = net_pts_each[net_pts_each > 0]
    losses = net_pts_each[net_pts_each <= 0]
    win_sum = wins.sum() if len(wins) > 0 else 0
    loss_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = win_sum / loss_sum if loss_sum > 0 else 999.0
    wr = len(wins) / n * 100
    cum = np.cumsum(net_pts_each)
    peak = np.maximum.accumulate(cum)
    max_dd = (cum - peak).min()
    avg_bars = np.mean([t["bars"] for t in trades])
    tp_ct = sum(1 for t in trades if t["result"] == "TP")
    sl_ct = sum(1 for t in trades if t["result"] == "SL")
    eod_ct = sum(1 for t in trades if t["result"] == "EOD")
    return {
        "count": n, "net_pts": round(net_pts, 2), "pf": round(pf, 3),
        "win_rate": round(wr, 1), "net_1lot": round(net_pts * 2, 2),
        "net_10lot": round(net_pts * 20, 2), "max_dd_pts": round(max_dd, 2),
        "avg_bars": round(avg_bars, 1), "tp": tp_ct, "sl": sl_ct, "eod": eod_ct,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 120)
    print("HEAD-TO-HEAD: AlgoAlpha SM vs Our SM — 79 days, 5-min MNQ")
    print("=" * 120)

    # Load 5-min data with TV indicators
    df_5m = load_5min_with_indicators("CME_MINI_MNQ1!, 5_46a9d.csv")
    print(f"\n5-min bars: {len(df_5m)}")
    print(f"Range: {df_5m.index[0]} to {df_5m.index[-1]}")

    # Load 1-min data for our SM
    df_1m = load_1min_data("CME_MINI_MNQ1!, 1_9f80c.csv")
    print(f"1-min bars: {len(df_1m)}")
    print(f"1-min range: {df_1m.index[0]} to {df_1m.index[-1]}")

    # Compute our SM on 1-min data with several param sets
    sm_param_sets = [
        ("Our_20_12_400_200",  20, 12, 400, 200),
        ("Our_20_12_400_255",  20, 12, 400, 255),  # matching AlgoAlpha EMA
        ("Our_15_10_300_150",  15, 10, 300, 150),   # F11 fast params
        ("Our_20_12_300_150",  20, 12, 300, 150),
        ("Our_25_14_500_255",  25, 14, 500, 255),   # AlgoAlpha defaults
        ("Our_10_8_200_100",   10,  8, 200, 100),   # ultra fast
    ]

    closes_1m = df_1m["Close"].values
    volumes_1m = df_1m["Volume"].values if "Volume" in df_1m.columns else (df_1m["High"] - df_1m["Low"]).values

    our_sm_1m = {}
    for name, idx, flow, norm, ema in sm_param_sets:
        sm_arr = compute_smart_money(closes_1m, volumes_1m, idx, flow, norm, ema)
        sm_series = pd.Series(sm_arr, index=df_1m.index)
        our_sm_1m[name] = resample_1m_sm_to_5m(sm_series, df_5m)
        print(f"  {name}: range {our_sm_1m[name].min():.3f} to {our_sm_1m[name].max():.3f}")

    # Add AlgoAlpha's pre-baked values
    aa_sm = df_5m["SM_1m_AA"].values
    print(f"  AlgoAlpha 1m SM: range {aa_sm.min():.3f} to {aa_sm.max():.3f}")

    # Prepare arrays
    opens = df_5m["Open"].values
    highs = df_5m["High"].values
    lows = df_5m["Low"].values
    closes = df_5m["Close"].values
    times = df_5m.index.values
    tv_rsi = df_5m["RSI_TV"].values

    # Use the best config from v5: RSI 11/12, 65/35, SM>0.05, 15/7, CD6
    test_configs = [
        ("RSI11 65/35 15/7 CD6", tv_rsi, 65, 35, 0.05, 15, 7, 6),
        ("RSI12 65/35 15/7 CD6", compute_rsi_arr(closes, 12), 65, 35, 0.05, 15, 7, 6),
        ("RSI11 65/35 15/7 CD3", tv_rsi, 65, 35, 0.05, 15, 7, 3),
        ("RSI11 65/35 20/10 CD3", tv_rsi, 65, 35, 0.00, 20, 10, 3),
        ("RSI10 55/45 20/10 CD3", compute_rsi_arr(closes, 10), 55, 45, 0.00, 20, 10, 3),
    ]

    # All SM sources
    sm_sources = {"AlgoAlpha": aa_sm}
    sm_sources.update(our_sm_1m)

    print(f"\n{'─' * 120}")
    print(f"{'SM Source':<25} {'Config':<25} {'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} {'$1lot':>9} {'$10lot':>10} {'MaxDD':>7}")
    print(f"{'─' * 120}")

    all_results = []

    for sm_name, sm_arr in sm_sources.items():
        for cfg_name, rsi_arr, rsi_buy, rsi_sell, sm_thr, tp, sl, cd in test_configs:
            trades = run_backtest(opens, highs, lows, closes, sm_arr, rsi_arr, times,
                                  rsi_buy, rsi_sell, sm_thr, tp, sl, cd)
            if len(trades) < 3:
                print(f"{sm_name:<25} {cfg_name:<25} {'< 3 trades':>5}")
                continue
            sc = score_trades(trades)
            print(f"{sm_name:<25} {cfg_name:<25} {sc['count']:>5} {sc['win_rate']:>5.1f}% "
                  f"{sc['pf']:>7.3f} {sc['net_pts']:>+8.1f} {sc['net_1lot']:>+9.2f} "
                  f"{sc['net_10lot']:>+10.2f} {sc['max_dd_pts']:>7.1f}")
            all_results.append({"sm": sm_name, "cfg": cfg_name, **sc})

    # Summary: best per SM source
    print(f"\n{'=' * 120}")
    print("BEST CONFIG PER SM SOURCE (by PF, min 5 trades)")
    print(f"{'=' * 120}")
    for sm_name in sm_sources:
        sm_results = [r for r in all_results if r["sm"] == sm_name and r["count"] >= 5]
        if sm_results:
            best = max(sm_results, key=lambda x: x["pf"])
            print(f"  {sm_name:<25} {best['cfg']:<25} Trades:{best['count']:>4} "
                  f"WR:{best['win_rate']:.1f}% PF:{best['pf']:.3f} "
                  f"Net:{best['net_pts']:>+.1f}pts $10lot:{best['net_10lot']:>+.2f}")

    print(f"\n{'=' * 120}")
    print("DONE")


if __name__ == "__main__":
    main()
