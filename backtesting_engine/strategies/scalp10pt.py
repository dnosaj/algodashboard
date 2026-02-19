"""
10-Point Scalper — Smart Money + RSI (with variable SM settings)

Computes SM indicator from scratch using range as volume proxy (MNQ has no volume).
This lets us vary all SM parameters: index_period, flow_period, norm_period, ema_len.

Constraints:
  - RSI length: 10 to 14 only
  - SM params: explore fast settings on 1-min and 5-min
  - TP target around 10 pts (also test 8, 12, 15)
  - Tight SL (5, 7, 10)
  - Quick in-and-out, OK with small profits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from itertools import product


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_ohlc(filename: str) -> pd.DataFrame:
    """Load TradingView CSV, extract just OHLC (ignore pre-baked indicators)."""
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
    # Synthesize volume from range (MNQ has no volume)
    df["Volume"] = df["High"] - df["Low"]
    df = df.iloc[:-1]  # drop last incomplete bar
    return df


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min to 5-min OHLCV."""
    return df.resample("5min").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
        "Volume": "sum"
    }).dropna()


# ─── Smart Money Indicator (computed from scratch) ───────────────────────────

def compute_ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average, matching pandas/TV behavior."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def compute_rsi(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI matching TradingView."""
    n = len(arr)
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # SMA seed
    if n > period:
        avg_gain[period] = np.mean(gain[1:period + 1])
        avg_loss[period] = np.mean(loss[1:period + 1])
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = 50.0  # undefined region
    return rsi


def compute_smart_money(close: np.ndarray, volume: np.ndarray,
                        index_period: int, flow_period: int,
                        norm_period: int, ema_len: int) -> np.ndarray:
    """
    Compute Smart Money net_index from OHLC data.
    Uses range as volume proxy (PVI/NVI).

    PVI updates when volume increases, NVI when volume decreases.
    Then: dumb = PVI - EMA(PVI, ema_len), smart = NVI - EMA(NVI, ema_len)
    Apply RSI to dumb/smart, compute buy/sell ratios, sum and normalize.
    """
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

    drsi = compute_rsi(dumb, flow_period)
    srsi = compute_rsi(smart, flow_period)

    # Buy/sell ratios
    r_buy = np.where(drsi != 0, srsi / drsi, 0.0)
    r_sell = np.where((100 - drsi) != 0, (100 - srsi) / (100 - drsi), 0.0)

    # Rolling sum
    sums_buy = pd.Series(r_buy).rolling(index_period, min_periods=1).sum().values
    sums_sell = pd.Series(r_sell).rolling(index_period, min_periods=1).sum().values

    # Normalize by peak
    combined_max = np.maximum(sums_buy, sums_sell)
    peak = pd.Series(combined_max).rolling(norm_period, min_periods=1).max().values
    peak = np.where(peak > 0, peak, 1.0)

    index_buy = sums_buy / peak
    index_sell = sums_sell / peak
    net_index = index_buy - index_sell

    return net_index


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(opens, highs, lows, closes, sm, rsi, times,
                 rsi_buy, rsi_sell, sm_threshold,
                 tp, sl, exit_on_sm_flip, max_hold):
    """Lean bar-by-bar backtest. Returns list of trade dicts."""
    n = len(opens)
    trades = []
    in_trade = False
    side = ""
    entry_price = 0.0
    entry_idx = 0

    # NY session in UTC: 9:30 AM ET = 14:30 UTC, 4:00 PM ET = 21:00 UTC
    NY_OPEN_UTC = 14 * 60 + 30    # 14:30 UTC = 9:30 AM ET
    NY_LAST_ENTRY_UTC = 20 * 60 + 45  # 20:45 UTC = 3:45 PM ET
    NY_CLOSE_UTC = 21 * 60        # 21:00 UTC = 4:00 PM ET

    for i in range(2, n):  # need i-1 for signal, i for fill
        bar_ts = pd.Timestamp(times[i])
        bar_mins_utc = bar_ts.hour * 60 + bar_ts.minute

        # Force close at end of NY session (4:00 PM ET = 21:00 UTC)
        if in_trade:
            if bar_mins_utc >= NY_CLOSE_UTC:
                bars_held = i - entry_idx
                if side == "long":
                    pts = closes[i] - entry_price
                else:
                    pts = entry_price - closes[i]
                trades.append({"side": side, "entry": entry_price,
                               "exit": closes[i], "pts": pts,
                               "entry_time": times[entry_idx], "exit_time": times[i],
                               "bars": bars_held, "result": "EOD"})
                in_trade = False
                continue

        # Check exits
        if in_trade:
            bars_held = i - entry_idx
            if side == "long":
                sl_p = entry_price - sl
                tp_p = entry_price + tp
                if lows[i] <= sl_p:
                    trades.append({"side": "long", "entry": entry_price,
                                   "exit": sl_p, "pts": -sl,
                                   "entry_time": times[entry_idx], "exit_time": times[i],
                                   "bars": bars_held, "result": "SL"})
                    in_trade = False
                elif highs[i] >= tp_p:
                    trades.append({"side": "long", "entry": entry_price,
                                   "exit": tp_p, "pts": tp,
                                   "entry_time": times[entry_idx], "exit_time": times[i],
                                   "bars": bars_held, "result": "TP"})
                    in_trade = False
                elif exit_on_sm_flip and sm[i] < 0:
                    pts = closes[i] - entry_price
                    trades.append({"side": "long", "entry": entry_price,
                                   "exit": closes[i], "pts": pts,
                                   "entry_time": times[entry_idx], "exit_time": times[i],
                                   "bars": bars_held, "result": "SM_FLIP"})
                    in_trade = False
                elif max_hold > 0 and bars_held >= max_hold:
                    pts = closes[i] - entry_price
                    trades.append({"side": "long", "entry": entry_price,
                                   "exit": closes[i], "pts": pts,
                                   "entry_time": times[entry_idx], "exit_time": times[i],
                                   "bars": bars_held, "result": "MAX_HOLD"})
                    in_trade = False
            else:  # short
                sl_p = entry_price + sl
                tp_p = entry_price - tp
                if highs[i] >= sl_p:
                    trades.append({"side": "short", "entry": entry_price,
                                   "exit": sl_p, "pts": -sl,
                                   "entry_time": times[entry_idx], "exit_time": times[i],
                                   "bars": bars_held, "result": "SL"})
                    in_trade = False
                elif lows[i] <= tp_p:
                    trades.append({"side": "short", "entry": entry_price,
                                   "exit": tp_p, "pts": tp,
                                   "entry_time": times[entry_idx], "exit_time": times[i],
                                   "bars": bars_held, "result": "TP"})
                    in_trade = False
                elif exit_on_sm_flip and sm[i] > 0:
                    pts = entry_price - closes[i]
                    trades.append({"side": "short", "entry": entry_price,
                                   "exit": closes[i], "pts": pts,
                                   "entry_time": times[entry_idx], "exit_time": times[i],
                                   "bars": bars_held, "result": "SM_FLIP"})
                    in_trade = False
                elif max_hold > 0 and bars_held >= max_hold:
                    pts = entry_price - closes[i]
                    trades.append({"side": "short", "entry": entry_price,
                                   "exit": closes[i], "pts": pts,
                                   "entry_time": times[entry_idx], "exit_time": times[i],
                                   "bars": bars_held, "result": "MAX_HOLD"})
                    in_trade = False

        # Check entries (signal on bar i-1, fill on bar i open)
        # NY session only: 9:30 AM - 3:45 PM ET = 14:30 - 20:45 UTC
        if not in_trade:
            in_ny_session = NY_OPEN_UTC <= bar_mins_utc <= NY_LAST_ENTRY_UTC

            if in_ny_session:
                # RSI crossover on bar i-1
                rsi_cross_up = rsi[i - 1] >= rsi_buy and rsi[i - 2] < rsi_buy
                rsi_cross_dn = rsi[i - 1] <= rsi_sell and rsi[i - 2] > rsi_sell

                if rsi_cross_up and sm[i - 1] > sm_threshold:
                    in_trade = True
                    side = "long"
                    entry_price = opens[i]
                    entry_idx = i
                elif rsi_cross_dn and sm[i - 1] < -sm_threshold:
                    in_trade = True
                    side = "short"
                    entry_price = opens[i]
                    entry_idx = i

    return trades


# ─── Scoring ────────────────────────────────────────────────────────────────

def score_trades(trades, commission_per_side=0.52):
    """Score trades in points and dollars. MNQ = $2/pt, $0.52/side commission."""
    if not trades:
        return None

    pts = np.array([t["pts"] for t in trades])
    n = len(pts)
    comm_pts = (commission_per_side * 2) / 2.0  # 0.52 pts per round trip
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
    fl_ct = sum(1 for t in trades if t["result"] == "SM_FLIP")

    return {
        "count": n, "net_pts": round(net_pts, 2), "pf": round(pf, 3),
        "win_rate": round(wr, 1), "net_1lot": round(net_pts * 2, 2),
        "net_10lot": round(net_pts * 20, 2), "max_dd_pts": round(max_dd, 2),
        "avg_bars": round(avg_bars, 1), "tp": tp_ct, "sl": sl_ct, "flip": fl_ct,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 110)
    print("10-POINT SCALPER — Variable SM Settings + RSI 10-14")
    print("=" * 110)

    df_1m = load_ohlc("CME_MINI_MNQ1!, 1_9f80c.csv")
    df_5m = resample_to_5min(df_1m)
    print(f"\n1-min: {len(df_1m)} bars | 5-min: {len(df_5m)} bars")
    print(f"Range: {df_1m.index[0]} to {df_1m.index[-1]}")

    # ── Parameter space ──
    # SM settings to try (index, flow, norm, ema)
    sm_param_sets = [
        ("SM_default", 25, 14, 500, 255),
        ("SM_fast1",   15, 10, 300, 150),
        ("SM_fast2",   10,  8, 200, 100),
        ("SM_fast3",   10,  6, 150,  80),
        ("SM_med",     20, 12, 400, 200),
        ("SM_tight",   12,  8, 250, 120),
    ]

    rsi_lengths = [10, 11, 12, 14]
    rsi_levels = [(60, 40), (65, 35), (55, 45)]
    sm_thresholds = [0.0, 0.05, 0.1]
    tp_sl_combos = [(8, 5), (10, 5), (10, 7), (10, 10), (12, 7), (15, 7), (15, 10)]
    exit_modes = [
        ("TP/SL", False, 0),
        ("SM_flip", True, 0),
        ("Max30", False, 30),
    ]

    # Total combos estimate
    total = (2 * len(sm_param_sets) * len(rsi_lengths) * len(rsi_levels) *
             len(sm_thresholds) * len(tp_sl_combos) * len(exit_modes))
    print(f"Parameter combos: ~{total}")

    results = []
    combo_count = 0

    for tf_name, df in [("1min", df_1m), ("5min", df_5m)]:
        closes = df["Close"].values
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        volumes = df["Volume"].values
        times = df.index.values

        for sm_name, sm_idx, sm_flow, sm_norm, sm_ema in sm_param_sets:
            # Compute SM once per param set per timeframe
            sm = compute_smart_money(closes, volumes, sm_idx, sm_flow, sm_norm, sm_ema)

            for rsi_len in rsi_lengths:
                # Compute RSI once per length
                rsi = compute_rsi(closes, rsi_len)

                for rsi_buy, rsi_sell in rsi_levels:
                    for sm_thr in sm_thresholds:
                        for tp, sl in tp_sl_combos:
                            for exit_name, sm_flip, max_hold in exit_modes:
                                combo_count += 1
                                trades = run_backtest(
                                    opens, highs, lows, closes, sm, rsi, times,
                                    rsi_buy, rsi_sell, sm_thr,
                                    tp, sl, sm_flip, max_hold)

                                if len(trades) < 3:
                                    continue

                                sc = score_trades(trades)
                                if sc is None:
                                    continue

                                results.append({
                                    "tf": tf_name, "sm": sm_name,
                                    "rsi_len": rsi_len,
                                    "rsi_lvl": f"{rsi_buy}/{rsi_sell}",
                                    "sm_thr": sm_thr,
                                    "tp_sl": f"{tp}/{sl}",
                                    "exit": exit_name,
                                    **sc
                                })

        print(f"  {tf_name}: done ({combo_count} combos so far)")

    print(f"\nTotal combos tested: {combo_count}")
    print(f"Configs with 3+ trades: {len(results)}")

    if not results:
        print("\nNo configurations produced 3+ trades. Try looser parameters.")
        return

    # ── Sort and display ──
    results.sort(key=lambda x: (-x["pf"], -x["net_pts"]))

    def print_table(title, rows, max_rows=25):
        print(f"\n{'─' * 110}")
        print(f"{title}")
        print(f"{'─' * 110}")
        print(f"{'#':>3} {'TF':>4} {'SM_Set':<11} {'RSI':>3} {'Lvl':>5} {'SM>':>5} "
              f"{'TP/SL':>5} {'Exit':<7} {'Trds':>4} {'WR%':>5} {'PF':>7} "
              f"{'NetPts':>7} {'$10lot':>9} {'MaxDD':>7} {'Bars':>5} {'TP/SL/Fl':>8}")
        for i, r in enumerate(rows[:max_rows]):
            print(f"{i+1:>3} {r['tf']:>4} {r['sm']:<11} {r['rsi_len']:>3} {r['rsi_lvl']:>5} "
                  f"{r['sm_thr']:>5.2f} {r['tp_sl']:>5} {r['exit']:<7} "
                  f"{r['count']:>4} {r['win_rate']:>4.1f}% {r['pf']:>7.3f} "
                  f"{r['net_pts']:>+7.1f} {r['net_10lot']:>+9.2f} "
                  f"{r['max_dd_pts']:>7.1f} {r['avg_bars']:>5.1f} "
                  f"{r['tp']}/{r['sl']}/{r['flip']}")

    # Top by PF
    print_table("TOP 25 BY PROFIT FACTOR", results)

    # Top by net pts
    by_net = sorted(results, key=lambda x: -x["net_pts"])
    print_table("TOP 25 BY NET POINTS", by_net)

    # High win rate
    high_wr = sorted(
        [r for r in results if r["win_rate"] > 55 and r["pf"] > 1.2 and r["count"] >= 5],
        key=lambda x: (-x["win_rate"], -x["pf"]))
    print_table("HIGH WIN RATE (>55% WR, PF>1.2, 5+ trades)", high_wr, 20)

    # Best per SM setting
    print(f"\n{'─' * 110}")
    print("BEST CONFIG PER SM SETTING (by PF, min 5 trades)")
    print(f"{'─' * 110}")
    for sm_name, _, _, _, _ in sm_param_sets:
        sm_results = [r for r in results if r["sm"] == sm_name and r["count"] >= 5]
        if sm_results:
            best = max(sm_results, key=lambda x: x["pf"])
            print(f"  {sm_name:<12} {best['tf']:>4} RSI{best['rsi_len']} {best['rsi_lvl']} "
                  f"SM>{best['sm_thr']:.2f} {best['tp_sl']} {best['exit']:<7} "
                  f"Trades:{best['count']:>3} WR:{best['win_rate']:.1f}% "
                  f"PF:{best['pf']:.3f} Net:{best['net_pts']:>+.1f}pts "
                  f"$10lot:{best['net_10lot']:>+.2f}")

    # Trade log for #1
    if results:
        best = results[0]
        print(f"\n{'=' * 110}")
        print(f"TRADE LOG — #1: {best['tf']} {best['sm']} RSI{best['rsi_len']} "
              f"{best['rsi_lvl']} SM>{best['sm_thr']} {best['tp_sl']} {best['exit']}")
        print(f"{'=' * 110}")

        # Re-run to get trades
        df_use = df_1m if best["tf"] == "1min" else df_5m
        sm_params = None
        for sname, sidx, sflow, snorm, sema in sm_param_sets:
            if sname == best["sm"]:
                sm_params = (sidx, sflow, snorm, sema)
                break
        sm_arr = compute_smart_money(df_use["Close"].values, df_use["Volume"].values, *sm_params)
        rsi_arr = compute_rsi(df_use["Close"].values, best["rsi_len"])
        rsi_buy, rsi_sell = [int(x) for x in best["rsi_lvl"].split("/")]
        tp_val, sl_val = [int(x) for x in best["tp_sl"].split("/")]

        trades = run_backtest(
            df_use["Open"].values, df_use["High"].values,
            df_use["Low"].values, df_use["Close"].values,
            sm_arr, rsi_arr, df_use.index.values,
            rsi_buy, rsi_sell, best["sm_thr"],
            tp_val, sl_val,
            best["exit"] == "SM_flip", 30 if best["exit"] == "Max30" else 0)

        cum = 0
        comm = 0.52
        for j, t in enumerate(trades):
            net = t["pts"] - comm
            cum += net
            d10 = net * 20
            et = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
            xt = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")
            print(f"  {j+1:>3}. {t['side']:>5} {et:>11} -> {xt:>11} "
                  f"E:{t['entry']:>9.2f} X:{t['exit']:>9.2f} "
                  f"Pts:{t['pts']:>+7.2f} Net:{net:>+6.2f} Cum:{cum:>+7.2f} "
                  f"$10:{d10:>+8.2f} [{t['result']}]")

    print(f"\n{'=' * 110}")
    print("DONE")


if __name__ == "__main__":
    main()
