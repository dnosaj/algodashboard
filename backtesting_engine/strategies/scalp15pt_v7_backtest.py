"""
15-Point Scalper v7 — Python Backtester (Jan 19 - Feb 12 period)
Uses NEW 1-min data file (25,977 bars) to get AlgoAlpha SM + compute our own SM.
Both are resampled to 5-min and used to drive entries on 5-min bars.

Signal: 1-min SM direction + 5-min RSI zone
  LONG  — 1m SM > threshold + RSI > rsi_buy
  SHORT — 1m SM < -threshold + RSI < rsi_sell

Zone entry: first bar both conditions align, one trade per episode.
Episode resets when RSI leaves zone OR SM flips.

NY session: 10:00 AM - 3:45 PM ET entries, force close at 4:00 PM.
Data timestamps are in UTC.

TWO SM sources tested:
  1. AlgoAlpha SM extracted from Net Buy Line / Net Sell Line columns
  2. Our own SM computed from OHLC using range as volume proxy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_1min_data(filename: str) -> pd.DataFrame:
    """Load 1-min TradingView CSV with AlgoAlpha SM data."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()

    df = df.rename(columns={
        cols[0]: "Time",
        cols[1]: "Open",
        cols[2]: "High",
        cols[3]: "Low",
        cols[4]: "Close",
    })

    # Extract AlgoAlpha SM from Net Buy Line (col 13) and Net Sell Line (col 19)
    # When SM is positive: Net Buy Line has value, Net Sell Line is empty
    # When SM is negative: Net Sell Line has value, Net Buy Line is empty
    net_buy_line = pd.to_numeric(df[cols[13]], errors="coerce").fillna(0)
    net_sell_line = pd.to_numeric(df[cols[19]], errors="coerce").fillna(0)
    # Net Sell Line values are already negative, so just add them
    df["SM_AA"] = net_buy_line + net_sell_line

    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")

    df = df[["Open", "High", "Low", "Close", "SM_AA"]].copy()
    return df


def load_5min_ohlc(filename: str) -> pd.DataFrame:
    """Load 5-min TradingView CSV for OHLC only."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()

    df = df.rename(columns={
        cols[0]: "Time",
        cols[1]: "Open",
        cols[2]: "High",
        cols[3]: "Low",
        cols[4]: "Close",
    })

    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close"]].copy()
    return df


# ─── Smart Money Computation (Our Own) ──────────────────────────────────────

def compute_smart_money(closes, volumes, index_period, flow_period, norm_period, ema_len):
    """
    Compute Smart Money Volume Index matching AlgoAlpha formula.
    Uses volume proxy (range) since MNQ CSV has no volume.
    """
    n = len(closes)

    # PVI / NVI from volume
    pvi = np.ones(n)
    nvi = np.ones(n)

    for i in range(1, n):
        pct_change = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0.0
        if volumes[i] > volumes[i-1]:
            pvi[i] = pvi[i-1] + pct_change * pvi[i-1]
            nvi[i] = nvi[i-1]
        elif volumes[i] < volumes[i-1]:
            nvi[i] = nvi[i-1] + pct_change * nvi[i-1]
            pvi[i] = pvi[i-1]
        else:
            pvi[i] = pvi[i-1]
            nvi[i] = nvi[i-1]

    # EMA of PVI and NVI
    def ema(arr, period):
        result = np.zeros_like(arr)
        result[0] = arr[0]
        alpha = 2.0 / (period + 1)
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result

    dumb = pvi - ema(pvi, ema_len)
    smart = nvi - ema(nvi, ema_len)

    # RSI of dumb and smart
    def rsi(arr, period):
        n = len(arr)
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        if n > period:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
            for i in range(period + 1, n):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        return 100.0 - 100.0 / (1.0 + rs)

    drsi = rsi(dumb, flow_period)
    srsi = rsi(smart, flow_period)

    # Buy/sell ratios
    r_buy = np.where(drsi != 0, srsi / drsi, 0.0)
    r_sell = np.where((100 - drsi) != 0, (100 - srsi) / (100 - drsi), 0.0)

    # Rolling sums
    sums_buy = np.zeros(n)
    sums_sell = np.zeros(n)
    for i in range(n):
        start = max(0, i - index_period + 1)
        sums_buy[i] = np.sum(r_buy[start:i+1])
        sums_sell[i] = np.sum(r_sell[start:i+1])

    # Peak normalization
    max_combined = np.maximum(sums_buy, sums_sell)
    peak = np.zeros(n)
    for i in range(n):
        start = max(0, i - norm_period + 1)
        peak[i] = np.max(max_combined[start:i+1])

    # Net index
    idx_buy = np.where(peak != 0, sums_buy / peak, 0.0)
    idx_sell = np.where(peak != 0, sums_sell / peak, 0.0)
    net_index = idx_buy - idx_sell

    return net_index


# ─── RSI Computation ─────────────────────────────────────────────────────────

def compute_rsi(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI matching TradingView."""
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


# ─── Resample 1-min SM to 5-min ─────────────────────────────────────────────

def resample_sm_to_5min(df_1m, df_5m):
    """
    For each 5-min bar, take the last 1-min SM value within that bar's window.
    This matches TradingView's request.security with lookahead_off.
    """
    sm_5min = np.zeros(len(df_5m))

    for i, ts in enumerate(df_5m.index):
        # 5-min bar at ts covers [ts, ts+5min)
        # Take the last 1-min bar that's <= ts + 4min (i.e. within the 5-min bar)
        mask = (df_1m.index >= ts) & (df_1m.index < ts + pd.Timedelta(minutes=5))
        subset = df_1m.loc[mask]
        if len(subset) > 0:
            sm_5min[i] = subset.iloc[-1]
        elif i > 0:
            sm_5min[i] = sm_5min[i-1]  # carry forward

    return sm_5min


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(opens, highs, lows, closes, sm_1m, rsi, times,
                 rsi_buy, rsi_sell, sm_threshold,
                 tp, sl, cooldown_bars):
    """
    Zone-entry backtest with 1-min SM driving direction on 5-min bars.
    One trade per episode. Episode resets when RSI leaves zone or SM flips.
    """
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    # NY session in UTC: 10:00 AM ET = 15:00 UTC, 3:45 PM ET = 20:45 UTC, 4:00 PM = 21:00 UTC
    NY_OPEN_UTC = 15 * 60
    NY_LAST_ENTRY_UTC = 20 * 60 + 45
    NY_CLOSE_UTC = 21 * 60

    for i in range(1, n):
        bar_ts = pd.Timestamp(times[i])
        bar_mins_utc = bar_ts.hour * 60 + bar_ts.minute

        # ── Zone detection ──
        rsi_in_buy = rsi[i] > rsi_buy
        rsi_in_sell = rsi[i] < rsi_sell
        sm_bull = sm_1m[i] > sm_threshold
        sm_bear = sm_1m[i] < -sm_threshold

        # ── Episode reset ──
        if not rsi_in_buy or not sm_bull:
            long_used = False
        if not rsi_in_sell or not sm_bear:
            short_used = False

        # ── Force close at EOD (4:00 PM ET = 21:00 UTC) ──
        if trade_state != 0 and bar_mins_utc >= NY_CLOSE_UTC:
            bars_held = i - entry_idx
            if trade_state == 1:
                pts = closes[i] - entry_price
            else:
                pts = entry_price - closes[i]
            trades.append({"side": "long" if trade_state == 1 else "short",
                           "entry": entry_price, "exit": closes[i], "pts": pts,
                           "entry_time": times[entry_idx], "exit_time": times[i],
                           "bars": bars_held, "result": "EOD"})
            trade_state = 0
            exit_bar = i
            continue

        # ── Check exits (TP/SL) ──
        if trade_state == 1:
            bars_held = i - entry_idx
            sl_p = entry_price - sl
            tp_p = entry_price + tp
            if lows[i] <= sl_p:
                trades.append({"side": "long", "entry": entry_price, "exit": sl_p,
                               "pts": -sl, "entry_time": times[entry_idx],
                               "exit_time": times[i], "bars": bars_held, "result": "SL"})
                trade_state = 0
                exit_bar = i
            elif highs[i] >= tp_p:
                trades.append({"side": "long", "entry": entry_price, "exit": tp_p,
                               "pts": tp, "entry_time": times[entry_idx],
                               "exit_time": times[i], "bars": bars_held, "result": "TP"})
                trade_state = 0
                exit_bar = i

        elif trade_state == -1:
            bars_held = i - entry_idx
            sl_p = entry_price + sl
            tp_p = entry_price - tp
            if highs[i] >= sl_p:
                trades.append({"side": "short", "entry": entry_price, "exit": sl_p,
                               "pts": -sl, "entry_time": times[entry_idx],
                               "exit_time": times[i], "bars": bars_held, "result": "SL"})
                trade_state = 0
                exit_bar = i
            elif lows[i] <= tp_p:
                trades.append({"side": "short", "entry": entry_price, "exit": tp_p,
                               "pts": tp, "entry_time": times[entry_idx],
                               "exit_time": times[i], "bars": bars_held, "result": "TP"})
                trade_state = 0
                exit_bar = i

        # ── Check entries ──
        if trade_state == 0:
            bars_since_exit = i - exit_bar
            in_session = NY_OPEN_UTC <= bar_mins_utc <= NY_LAST_ENTRY_UTC
            cooldown_ok = bars_since_exit >= cooldown_bars

            if in_session and cooldown_ok:
                if rsi_in_buy and sm_bull and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                elif rsi_in_sell and sm_bear and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True

    return trades


# ─── Scoring ────────────────────────────────────────────────────────────────

def score_trades(trades, commission_per_side=0.52):
    """Score trades. MNQ = $2/pt, $0.52/side commission."""
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
    peak_cum = np.maximum.accumulate(cum)
    max_dd = (cum - peak_cum).min()
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
    print("15-POINT SCALPER v7 — Jan 19 - Feb 12 (new 1-min data)")
    print("Using AlgoAlpha SM from CSV + Our Own SM from OHLC")
    print("=" * 120)

    # ── Load 1-min data ──
    print("\n▶ Loading 1-min data...")
    df_1m = load_1min_data("CME_MINI_MNQ1!, 1_7fdb6.csv")
    print(f"  1-min bars: {len(df_1m)}")
    print(f"  Range: {df_1m.index[0]} to {df_1m.index[-1]}")
    print(f"  AlgoAlpha SM range: {df_1m['SM_AA'].min():.4f} to {df_1m['SM_AA'].max():.4f}")

    # Count non-zero SM values
    sm_nonzero = (df_1m['SM_AA'] != 0).sum()
    print(f"  Non-zero SM bars: {sm_nonzero} / {len(df_1m)} ({sm_nonzero/len(df_1m)*100:.1f}%)")

    # ── Compute our own SM on 1-min ──
    print("\n▶ Computing our own SM on 1-min bars...")
    closes_1m = df_1m["Close"].values
    highs_1m = df_1m["High"].values
    lows_1m = df_1m["Low"].values
    volumes_1m = highs_1m - lows_1m  # range as volume proxy

    # Use AlgoAlpha matching params: flow=12, index=20, norm=400, ema=255
    our_sm_1m = compute_smart_money(closes_1m, volumes_1m,
                                     index_period=20, flow_period=12,
                                     norm_period=400, ema_len=255)
    df_1m["SM_Ours"] = our_sm_1m
    print(f"  Our SM range: {our_sm_1m.min():.4f} to {our_sm_1m.max():.4f}")

    # ── Load 5-min data ──
    print("\n▶ Loading 5-min data...")
    df_5m = load_5min_ohlc("CME_MINI_MNQ1!, 5_46a9d.csv")
    print(f"  5-min bars: {len(df_5m)}")
    print(f"  Range: {df_5m.index[0]} to {df_5m.index[-1]}")

    # Filter to Jan 19 - Feb 12 (matching TV test period)
    # Jan 19 is a Sunday, so first trading day is Jan 20 (Monday)
    start_date = pd.Timestamp("2026-01-19")
    end_date = pd.Timestamp("2026-02-13")  # exclusive
    df_5m = df_5m[(df_5m.index >= start_date) & (df_5m.index < end_date)]
    print(f"  After filtering Jan 19 - Feb 12: {len(df_5m)} bars")
    print(f"  Filtered range: {df_5m.index[0]} to {df_5m.index[-1]}")

    # ── Resample 1-min SM to 5-min ──
    print("\n▶ Resampling 1-min SM to 5-min bars...")

    # Method: for each 5-min bar, take the LAST 1-min SM value
    # This matches request.security with lookahead_off
    sm_aa_5m = resample_sm_to_5min(df_1m["SM_AA"], df_5m)
    sm_ours_5m = resample_sm_to_5min(df_1m["SM_Ours"], df_5m)

    # Also get the pre-baked SM from 5-min file for comparison
    df_5m_full = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "CME_MINI_MNQ1!, 5_46a9d.csv"
    )
    cols5 = df_5m_full.columns.tolist()
    df_5m_full["Time"] = pd.to_datetime(df_5m_full[cols5[0]].astype(int), unit="s")
    df_5m_full = df_5m_full.set_index("Time")
    df_5m_full = df_5m_full[(df_5m_full.index >= start_date) & (df_5m_full.index < end_date)]
    sm_prebaked = pd.to_numeric(df_5m_full[cols5[7]], errors="coerce").fillna(0).values

    print(f"  AlgoAlpha SM (resampled): range {sm_aa_5m.min():.4f} to {sm_aa_5m.max():.4f}")
    print(f"  Our SM (resampled): range {sm_ours_5m.min():.4f} to {sm_ours_5m.max():.4f}")
    print(f"  Pre-baked SM from 5m CSV: range {sm_prebaked.min():.4f} to {sm_prebaked.max():.4f}")

    # Check correlation between AlgoAlpha from 1-min resample vs pre-baked from 5-min CSV
    min_len = min(len(sm_aa_5m), len(sm_prebaked))
    if min_len > 0:
        corr = np.corrcoef(sm_aa_5m[:min_len], sm_prebaked[:min_len])[0, 1]
        print(f"  Correlation (AA resampled vs pre-baked): {corr:.4f}")
        # Show agreement on direction
        agree = np.sum(np.sign(sm_aa_5m[:min_len]) == np.sign(sm_prebaked[:min_len]))
        print(f"  Direction agreement: {agree}/{min_len} ({agree/min_len*100:.1f}%)")

    # ── Prepare arrays ──
    opens = df_5m["Open"].values
    highs = df_5m["High"].values
    lows = df_5m["Low"].values
    closes = df_5m["Close"].values
    times = df_5m.index.values

    # ── Parameter space ──
    rsi_lengths = [10, 11, 12, 14]
    rsi_levels = [(65, 35), (60, 40), (55, 45)]
    sm_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20]
    tp_sl_combos = [(10, 5), (10, 7), (12, 7), (15, 7), (15, 10), (20, 10)]
    cooldowns = [3, 6, 10]

    sm_sources = {
        "AA_resamp": sm_aa_5m,
        "Ours_resamp": sm_ours_5m,
        "AA_prebaked": sm_prebaked,
    }

    total = len(sm_sources) * len(rsi_lengths) * len(rsi_levels) * len(sm_thresholds) * len(tp_sl_combos) * len(cooldowns)
    print(f"\n▶ Testing {total} parameter combos across 3 SM sources...")

    results = []
    combo_count = 0

    for sm_name, sm_arr in sm_sources.items():
        for rsi_len in rsi_lengths:
            rsi_arr = compute_rsi(closes, rsi_len)
            for rsi_buy, rsi_sell in rsi_levels:
                for sm_thr in sm_thresholds:
                    for tp, sl in tp_sl_combos:
                        for cd in cooldowns:
                            combo_count += 1
                            trades = run_backtest(
                                opens, highs, lows, closes, sm_arr, rsi_arr, times,
                                rsi_buy, rsi_sell, sm_thr, tp, sl, cd)

                            if len(trades) < 3:
                                continue

                            sc = score_trades(trades)
                            if sc is None:
                                continue

                            results.append({
                                "sm_src": sm_name,
                                "rsi": f"RSI{rsi_len}",
                                "rsi_lvl": f"{rsi_buy}/{rsi_sell}",
                                "sm_thr": sm_thr,
                                "tp_sl": f"{tp}/{sl}",
                                "cd": cd,
                                **sc
                            })

    print(f"Total combos tested: {combo_count}")
    print(f"Configs with 3+ trades: {len(results)}")

    if not results:
        print("\nNo configurations produced 3+ trades.")
        return

    # ── Sort and display ──
    def print_table(title, rows, max_rows=25):
        print(f"\n{'─' * 130}")
        print(f"{title}")
        print(f"{'─' * 130}")
        print(f"{'#':>3} {'SM':>12} {'RSI':>5} {'Lvl':>5} {'SM>':>5} {'TP/SL':>5} {'CD':>3} "
              f"{'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} {'$1lot':>9} {'$10lot':>10} "
              f"{'MaxDD':>7} {'Bars':>5} {'TP/SL/EOD':>9}")
        for i, r in enumerate(rows[:max_rows]):
            print(f"{i+1:>3} {r['sm_src']:>12} {r['rsi']:>5} {r['rsi_lvl']:>5} {r['sm_thr']:>5.2f} "
                  f"{r['tp_sl']:>5} {r['cd']:>3} "
                  f"{r['count']:>5} {r['win_rate']:>5.1f}% {r['pf']:>7.3f} "
                  f"{r['net_pts']:>+8.1f} {r['net_1lot']:>+9.2f} {r['net_10lot']:>+10.2f} "
                  f"{r['max_dd_pts']:>7.1f} {r['avg_bars']:>5.1f} "
                  f"{r['tp']}/{r['sl']}/{r['eod']}")

    # ── EXACT TV CONFIG: RSI 11, 65/35, SM > 0.10, TP 15/SL 7, CD 6 ──
    print(f"\n{'=' * 130}")
    print("EXACT TV CONFIG (v6 MTF): RSI 11, 65/35, SM > 0.10, TP 15/SL 7, CD 6")
    print(f"{'=' * 130}")

    for sm_name, sm_arr in sm_sources.items():
        rsi_arr = compute_rsi(closes, 11)
        trades = run_backtest(opens, highs, lows, closes, sm_arr, rsi_arr, times,
                              65, 35, 0.10, 15, 7, 6)
        sc = score_trades(trades) if trades else None
        print(f"\n  {sm_name}: ", end="")
        if sc:
            print(f"{sc['count']} trades, WR {sc['win_rate']}%, PF {sc['pf']}, "
                  f"Net {sc['net_pts']:+.2f} pts, ${sc['net_1lot']:+.2f}/1lot, "
                  f"TP:{sc['tp']} SL:{sc['sl']} EOD:{sc['eod']}")
        else:
            print(f"{len(trades)} trades (too few or no result)")

        # Print trade log
        if trades:
            print(f"    Trade log:")
            cum = 0
            comm = 0.52
            for j, t in enumerate(trades):
                net = t["pts"] - comm
                cum += net
                d1 = net * 2
                et = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
                xt = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")
                print(f"      {j+1:>3}. {t['side']:>5} {et:>11} -> {xt:>11} "
                      f"E:{t['entry']:>9.2f} X:{t['exit']:>9.2f} "
                      f"Pts:{t['pts']:>+7.2f} Net:{net:>+6.2f} Cum:{cum:>+7.2f} "
                      f"$1:{d1:>+7.2f} [{t['result']}]")

    # ── Also test RSI 12 (our previous best) ──
    print(f"\n{'=' * 130}")
    print("PREVIOUS BEST CONFIG: RSI 12, 65/35, SM > 0.10, TP 15/SL 7, CD 6")
    print(f"{'=' * 130}")

    for sm_name, sm_arr in sm_sources.items():
        rsi_arr = compute_rsi(closes, 12)
        trades = run_backtest(opens, highs, lows, closes, sm_arr, rsi_arr, times,
                              65, 35, 0.10, 15, 7, 6)
        sc = score_trades(trades) if trades else None
        print(f"  {sm_name}: ", end="")
        if sc:
            print(f"{sc['count']} trades, WR {sc['win_rate']}%, PF {sc['pf']}, "
                  f"Net {sc['net_pts']:+.2f} pts, ${sc['net_1lot']:+.2f}/1lot")
        else:
            print(f"{len(trades) if trades else 0} trades")

    # Top by PF (min 5 trades)
    by_pf = sorted([r for r in results if r["count"] >= 5],
                   key=lambda x: (-x["pf"], -x["net_pts"]))
    print_table("TOP 25 BY PROFIT FACTOR (min 5 trades)", by_pf)

    # Top by net pts
    by_net = sorted(results, key=lambda x: -x["net_pts"])
    print_table("TOP 25 BY NET POINTS", by_net)

    # Best balanced: PF > 1.3, WR > 45%, 8+ trades
    balanced = sorted(
        [r for r in results if r["pf"] > 1.3 and r["win_rate"] > 45 and r["count"] >= 8],
        key=lambda x: (-x["net_pts"]))
    print_table("BEST BALANCED (PF>1.3, WR>45%, 8+ trades)", balanced, 25)

    # ── Summary by SM source ──
    print(f"\n{'=' * 130}")
    print("SUMMARY BY SM SOURCE")
    print(f"{'=' * 130}")

    for sm_name in sm_sources:
        src_results = [r for r in results if r["sm_src"] == sm_name]
        profitable = [r for r in src_results if r["net_pts"] > 0]
        if src_results:
            avg_pf = np.mean([r["pf"] for r in src_results])
            avg_wr = np.mean([r["win_rate"] for r in src_results])
            avg_net = np.mean([r["net_pts"] for r in src_results])
            best_net = max(r["net_pts"] for r in src_results)
            best_pf = max(r["pf"] for r in src_results)
            print(f"  {sm_name:>15}: {len(src_results):>4} configs, "
                  f"{len(profitable):>3} profitable ({len(profitable)/len(src_results)*100:.0f}%), "
                  f"Avg PF: {avg_pf:.2f}, Avg WR: {avg_wr:.1f}%, "
                  f"Avg Net: {avg_net:+.1f}, Best Net: {best_net:+.1f}, Best PF: {best_pf:.2f}")

    # ── Trade log for #1 by net pts ──
    if by_net:
        best = by_net[0]
        print(f"\n{'=' * 130}")
        print(f"TRADE LOG — #1 by Net: {best['sm_src']} {best['rsi']} {best['rsi_lvl']} "
              f"SM>{best['sm_thr']} {best['tp_sl']} CD{best['cd']}")
        print(f"{'=' * 130}")

        sm_arr = sm_sources[best["sm_src"]]
        rsi_len = int(best["rsi"].replace("RSI", ""))
        rsi_arr = compute_rsi(closes, rsi_len)
        rsi_buy, rsi_sell = [int(x) for x in best["rsi_lvl"].split("/")]
        tp_val, sl_val = [int(x) for x in best["tp_sl"].split("/")]
        trades = run_backtest(opens, highs, lows, closes, sm_arr, rsi_arr, times,
                              rsi_buy, rsi_sell, best["sm_thr"], tp_val, sl_val, best["cd"])

        cum = 0
        comm = 0.52
        for j, t in enumerate(trades):
            net = t["pts"] - comm
            cum += net
            d1 = net * 2
            et = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
            xt = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")
            print(f"  {j+1:>3}. {t['side']:>5} {et:>11} -> {xt:>11} "
                  f"E:{t['entry']:>9.2f} X:{t['exit']:>9.2f} "
                  f"Pts:{t['pts']:>+7.2f} Net:{net:>+6.2f} Cum:{cum:>+7.2f} "
                  f"$1:{d1:>+7.2f} [{t['result']}]")

    print(f"\n{'=' * 130}")
    print("DONE")


if __name__ == "__main__":
    main()
