"""
15-Point Scalper v5 MTF — Python Backtester
Uses pre-computed 1-min SM + 5-min SM from TradingView export.

Signal: 1-min SM direction + 5-min RSI zone
  LONG  — 1m SM > threshold + RSI > rsi_buy
  SHORT — 1m SM < -threshold + RSI < rsi_sell

Zone entry: first bar both conditions align, one trade per episode.
Episode resets when RSI leaves zone OR SM flips.

NY session: 10:00 AM - 3:45 PM ET entries, force close at 4:00 PM.
Data timestamps are in UTC.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_5min_with_indicators(filename: str) -> pd.DataFrame:
    """Load 5-min TradingView CSV with pre-baked 1m SM, 5m SM, and RSI."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()

    # Map columns
    df = df.rename(columns={
        cols[0]: "Time",
        cols[1]: "Open",
        cols[2]: "High",
        cols[3]: "Low",
        cols[4]: "Close",
        cols[7]: "SM_1m",    # 1-min SM net_index
        cols[8]: "SM_5m",    # 5-min SM net_index
    })

    # RSI is in the plot columns - cols[9], cols[10], cols[11] appear to be the same RSI value
    df["RSI"] = pd.to_numeric(df[cols[9]], errors="coerce")

    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")

    # Keep only what we need
    df = df[["Open", "High", "Low", "Close", "SM_1m", "SM_5m", "RSI"]].copy()

    # Convert SM columns to numeric (handle any NaN)
    df["SM_1m"] = pd.to_numeric(df["SM_1m"], errors="coerce").fillna(0)
    df["SM_5m"] = pd.to_numeric(df["SM_5m"], errors="coerce").fillna(0)
    df["RSI"] = df["RSI"].fillna(50)

    df = df.iloc[:-1]  # drop last potentially incomplete bar
    return df


# ─── RSI Computation (for testing different RSI lengths) ────────────────────

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
    trade_state = 0   # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    # NY session in UTC: 10:00 AM ET = 15:00 UTC, 3:45 PM ET = 20:45 UTC, 4:00 PM = 21:00 UTC
    NY_OPEN_UTC = 15 * 60           # 15:00 UTC = 10:00 AM ET
    NY_LAST_ENTRY_UTC = 20 * 60 + 45  # 20:45 UTC = 3:45 PM ET
    NY_CLOSE_UTC = 21 * 60          # 21:00 UTC = 4:00 PM ET

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
        if trade_state == 1:  # long
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

        elif trade_state == -1:  # short
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
    print("=" * 110)
    print("15-POINT SCALPER v5 MTF — 1-min SM driving 5-min entries")
    print("=" * 110)

    df = load_5min_with_indicators("CME_MINI_MNQ1!, 5_46a9d.csv")
    print(f"\n5-min bars: {len(df)}")
    print(f"Range: {df.index[0]} to {df.index[-1]}")
    print(f"SM_1m range: {df['SM_1m'].min():.3f} to {df['SM_1m'].max():.3f}")
    print(f"SM_5m range: {df['SM_5m'].min():.3f} to {df['SM_5m'].max():.3f}")
    print(f"RSI range: {df['RSI'].min():.1f} to {df['RSI'].max():.1f}")

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    sm_1m = df["SM_1m"].values
    times = df.index.values
    tv_rsi = df["RSI"].values  # pre-computed RSI from TV

    # ── Parameter space ──
    rsi_lengths = [10, 11, 12, 14]
    rsi_levels = [(65, 35), (60, 40), (55, 45)]
    sm_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20]
    tp_sl_combos = [(10, 5), (10, 7), (12, 7), (15, 7), (15, 10), (20, 10)]
    cooldowns = [3, 6, 10]

    # Also test with TV's pre-baked RSI (unknown length but likely 11 or 14)
    rsi_sources = {}
    rsi_sources["TV_RSI"] = tv_rsi
    for rsi_len in rsi_lengths:
        rsi_sources[f"RSI{rsi_len}"] = compute_rsi(closes, rsi_len)

    total = len(rsi_sources) * len(rsi_levels) * len(sm_thresholds) * len(tp_sl_combos) * len(cooldowns)
    print(f"\nParameter combos: {total}")

    results = []
    combo_count = 0

    for rsi_name, rsi_arr in rsi_sources.items():
        for rsi_buy, rsi_sell in rsi_levels:
            for sm_thr in sm_thresholds:
                for tp, sl in tp_sl_combos:
                    for cd in cooldowns:
                        combo_count += 1
                        trades = run_backtest(
                            opens, highs, lows, closes, sm_1m, rsi_arr, times,
                            rsi_buy, rsi_sell, sm_thr, tp, sl, cd)

                        if len(trades) < 5:
                            continue

                        sc = score_trades(trades)
                        if sc is None:
                            continue

                        results.append({
                            "rsi": rsi_name,
                            "rsi_lvl": f"{rsi_buy}/{rsi_sell}",
                            "sm_thr": sm_thr,
                            "tp_sl": f"{tp}/{sl}",
                            "cd": cd,
                            **sc
                        })

    print(f"Total combos tested: {combo_count}")
    print(f"Configs with 5+ trades: {len(results)}")

    if not results:
        print("\nNo configurations produced 5+ trades.")
        return

    # ── Sort and display ──
    def print_table(title, rows, max_rows=25):
        print(f"\n{'─' * 120}")
        print(f"{title}")
        print(f"{'─' * 120}")
        print(f"{'#':>3} {'RSI':>8} {'Lvl':>5} {'SM>':>5} {'TP/SL':>5} {'CD':>3} "
              f"{'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} {'$1lot':>9} {'$10lot':>10} "
              f"{'MaxDD':>7} {'Bars':>5} {'TP/SL/EOD':>9}")
        for i, r in enumerate(rows[:max_rows]):
            print(f"{i+1:>3} {r['rsi']:>8} {r['rsi_lvl']:>5} {r['sm_thr']:>5.2f} "
                  f"{r['tp_sl']:>5} {r['cd']:>3} "
                  f"{r['count']:>5} {r['win_rate']:>5.1f}% {r['pf']:>7.3f} "
                  f"{r['net_pts']:>+8.1f} {r['net_1lot']:>+9.2f} {r['net_10lot']:>+10.2f} "
                  f"{r['max_dd_pts']:>7.1f} {r['avg_bars']:>5.1f} "
                  f"{r['tp']}/{r['sl']}/{r['eod']}")

    # Top by PF (min 10 trades)
    by_pf = sorted([r for r in results if r["count"] >= 10],
                   key=lambda x: (-x["pf"], -x["net_pts"]))
    print_table("TOP 25 BY PROFIT FACTOR (min 10 trades)", by_pf)

    # Top by net pts
    by_net = sorted(results, key=lambda x: -x["net_pts"])
    print_table("TOP 25 BY NET POINTS", by_net)

    # High win rate
    high_wr = sorted(
        [r for r in results if r["win_rate"] > 50 and r["pf"] > 1.2 and r["count"] >= 10],
        key=lambda x: (-x["win_rate"], -x["pf"]))
    print_table("HIGH WIN RATE (>50% WR, PF>1.2, 10+ trades)", high_wr, 20)

    # Best balanced: PF > 1.5, WR > 45%, 15+ trades
    balanced = sorted(
        [r for r in results if r["pf"] > 1.5 and r["win_rate"] > 45 and r["count"] >= 15],
        key=lambda x: (-x["net_pts"]))
    print_table("BEST BALANCED (PF>1.5, WR>45%, 15+ trades)", balanced, 20)

    # ── Trade log for #1 by PF ──
    if by_pf:
        best = by_pf[0]
        print(f"\n{'=' * 120}")
        print(f"TRADE LOG — #1 by PF: {best['rsi']} {best['rsi_lvl']} SM>{best['sm_thr']} "
              f"{best['tp_sl']} CD{best['cd']}")
        print(f"{'=' * 120}")

        # Re-run
        rsi_arr = rsi_sources[best["rsi"]]
        rsi_buy, rsi_sell = [int(x) for x in best["rsi_lvl"].split("/")]
        tp_val, sl_val = [int(x) for x in best["tp_sl"].split("/")]
        trades = run_backtest(opens, highs, lows, closes, sm_1m, rsi_arr, times,
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

    # ── Trade log for #1 by net pts ──
    if by_net and by_net[0] != by_pf[0]:
        best = by_net[0]
        print(f"\n{'=' * 120}")
        print(f"TRADE LOG — #1 by Net: {best['rsi']} {best['rsi_lvl']} SM>{best['sm_thr']} "
              f"{best['tp_sl']} CD{best['cd']}")
        print(f"{'=' * 120}")

        rsi_arr = rsi_sources[best["rsi"]]
        rsi_buy, rsi_sell = [int(x) for x in best["rsi_lvl"].split("/")]
        tp_val, sl_val = [int(x) for x in best["tp_sl"].split("/")]
        trades = run_backtest(opens, highs, lows, closes, sm_1m, rsi_arr, times,
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

    print(f"\n{'=' * 120}")
    print("DONE")


if __name__ == "__main__":
    main()
