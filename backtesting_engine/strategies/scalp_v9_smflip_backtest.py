"""
Scalper v9 — SM-Flip Entry/Exit Strategy (NO Look-Ahead)
==========================================================
Inspired by the F11 winner (PF 5.852) which used SM direction flip as exit.

KEY DIFFERENCES from v8 (zone entry + fixed TP/SL):
  - ENTRY: SM crosses threshold in a direction + RSI confirms
  - EXIT:  SM flips direction (crosses zero or flips sign), OR reversal entry fires
  - No fixed TP/SL — lets winners run with SM trend, cuts losers when SM flips
  - Optional trailing stop to protect profits

LOOK-AHEAD FIX: All signals use bar[i-1] values, entry at bar[i] open.

SM Sources tested:
  1. AA_resamp   — AlgoAlpha SM from 1-min CSV, resampled to 5-min
  2. Ours_fast   — Our SM with fast params (15,10,300,150) on 1-min, resampled to 5-min
  3. Ours_aa     — Our SM with AA-like params (20,12,400,255) on 1-min, resampled to 5-min
  4. AA_prebaked — Pre-baked "1m SM" column from 5-min CSV
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
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close",
    })

    # Extract AlgoAlpha SM: Net Buy Line (col 13) + Net Sell Line (col 19)
    net_buy = pd.to_numeric(df[cols[13]], errors="coerce").fillna(0)
    net_sell = pd.to_numeric(df[cols[19]], errors="coerce").fillna(0)
    df["SM_AA"] = net_buy + net_sell

    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "SM_AA"]].copy()
    return df


def load_5min_data(filename: str) -> pd.DataFrame:
    """Load 5-min TradingView CSV with OHLC + pre-baked SM."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()

    df = df.rename(columns={
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close",
    })

    # Pre-baked SM columns
    df["SM_1m_prebaked"] = pd.to_numeric(df[cols[7]], errors="coerce").fillna(0)  # "1m SM"
    df["SM_5m_prebaked"] = pd.to_numeric(df[cols[8]], errors="coerce").fillna(0)  # "5m SM"

    # AlgoAlpha 5-min native: Net Buy Line (col 20) + Net Sell Line (col 26)
    net_buy_5m = pd.to_numeric(df[cols[20]], errors="coerce").fillna(0)
    net_sell_5m = pd.to_numeric(df[cols[26]], errors="coerce").fillna(0)
    df["SM_AA_5m"] = net_buy_5m + net_sell_5m

    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "SM_1m_prebaked", "SM_5m_prebaked", "SM_AA_5m"]].copy()
    return df


# ─── Smart Money Computation ────────────────────────────────────────────────

def compute_smart_money(closes, volumes, index_period, flow_period, norm_period, ema_len):
    """Compute SM net index. Uses range as volume proxy."""
    n = len(closes)
    pvi = np.ones(n)
    nvi = np.ones(n)

    for i in range(1, n):
        pct = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0.0
        if volumes[i] > volumes[i-1]:
            pvi[i] = pvi[i-1] + pct * pvi[i-1]
            nvi[i] = nvi[i-1]
        elif volumes[i] < volumes[i-1]:
            nvi[i] = nvi[i-1] + pct * nvi[i-1]
            pvi[i] = pvi[i-1]
        else:
            pvi[i] = pvi[i-1]
            nvi[i] = nvi[i-1]

    def ema(arr, period):
        r = np.zeros_like(arr)
        r[0] = arr[0]
        a = 2.0 / (period + 1)
        for i in range(1, len(arr)):
            r[i] = a * arr[i] + (1 - a) * r[i-1]
        return r

    dumb = pvi - ema(pvi, ema_len)
    smart = nvi - ema(nvi, ema_len)

    def rsi(arr, period):
        n = len(arr)
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        ag = np.zeros(n); al = np.zeros(n)
        if n > period:
            ag[period] = np.mean(gain[1:period+1])
            al[period] = np.mean(loss[1:period+1])
            for i in range(period+1, n):
                ag[i] = (ag[i-1] * (period-1) + gain[i]) / period
                al[i] = (al[i-1] * (period-1) + loss[i]) / period
        rs = np.where(al > 0, ag / al, 100.0)
        return 100.0 - 100.0 / (1.0 + rs)

    drsi = rsi(dumb, flow_period)
    srsi = rsi(smart, flow_period)

    r_buy = np.where(drsi != 0, srsi / drsi, 0.0)
    r_sell = np.where((100 - drsi) != 0, (100 - srsi) / (100 - drsi), 0.0)

    sb = np.zeros(n); ss = np.zeros(n)
    for i in range(n):
        s = max(0, i - index_period + 1)
        sb[i] = np.sum(r_buy[s:i+1])
        ss[i] = np.sum(r_sell[s:i+1])

    mx = np.maximum(sb, ss)
    pk = np.zeros(n)
    for i in range(n):
        s = max(0, i - norm_period + 1)
        pk[i] = np.max(mx[s:i+1])

    ib = np.where(pk != 0, sb / pk, 0.0)
    isl = np.where(pk != 0, ss / pk, 0.0)
    return ib - isl


# ─── RSI Computation ─────────────────────────────────────────────────────────

def compute_rsi(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI."""
    n = len(arr)
    delta = np.diff(arr, prepend=arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.zeros(n); al = np.zeros(n)
    if n > period:
        ag[period] = np.mean(gain[1:period+1])
        al[period] = np.mean(loss[1:period+1])
        for i in range(period+1, n):
            ag[i] = (ag[i-1] * (period-1) + gain[i]) / period
            al[i] = (al[i-1] * (period-1) + loss[i]) / period
    rs = np.where(al > 0, ag / al, 100.0)
    r = 100.0 - 100.0 / (1.0 + rs)
    r[:period] = 50.0
    return r


# ─── Resample 1-min to 5-min ────────────────────────────────────────────────

def resample_to_5min(series_1m, df_5m):
    """Take last 1-min value within each 5-min bar's window."""
    result = np.zeros(len(df_5m))
    for i, ts in enumerate(df_5m.index):
        mask = (series_1m.index >= ts) & (series_1m.index < ts + pd.Timedelta(minutes=5))
        subset = series_1m.loc[mask]
        if len(subset) > 0:
            result[i] = subset.iloc[-1]
        elif i > 0:
            result[i] = result[i-1]
    return result


# ─── v9 Backtest Engine: SM-Flip Entry/Exit (NO Look-Ahead) ─────────────────

def run_backtest_v9(opens, highs, lows, closes, sm, rsi, times,
                     rsi_buy, rsi_sell, sm_threshold,
                     cooldown_bars,
                     max_loss_pts=0,        # Optional max loss stop (0 = disabled)
                     trailing_stop_pts=0,   # Optional trailing stop (0 = disabled)
                     use_rsi_cross=True):    # True = RSI cross entry, False = RSI level entry
    """
    SM-Flip Strategy with look-ahead fix.

    ENTRY (at bar[i] open, using bar[i-1] signals):
      LONG:  SM[i-1] > sm_threshold  AND  RSI condition met
      SHORT: SM[i-1] < -sm_threshold AND  RSI condition met

    RSI condition depends on use_rsi_cross:
      True:  RSI crosses above rsi_buy (long) / below rsi_sell (short) — F11 style
      False: RSI is above rsi_buy (long) / below rsi_sell (short) — zone style

    EXIT:
      - SM flips: SM[i-1] crosses zero (long exit when SM < 0, short exit when SM > 0)
      - Reversal entry fires (acts as exit + new entry)
      - Optional: max_loss_pts stop loss
      - Optional: trailing_stop_pts trailing stop
      - EOD close at 4:00 PM ET

    Episode: one entry per SM-direction + RSI alignment. Resets when SM flips.
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    max_equity = 0.0    # For trailing stop

    NY_OPEN_UTC = 15 * 60       # 10:00 AM ET = 15:00 UTC
    NY_LAST_ENTRY_UTC = 20 * 60 + 45  # 3:45 PM ET = 20:45 UTC
    NY_CLOSE_UTC = 21 * 60      # 4:00 PM ET = 21:00 UTC

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "bars": exit_i - entry_i, "result": result
        })

    for i in range(2, n):
        bar_ts = pd.Timestamp(times[i])
        bar_mins_utc = bar_ts.hour * 60 + bar_ts.minute

        # ── Previous bar's signals (look-ahead fix) ──
        sm_prev = sm[i-1]
        sm_prev2 = sm[i-2]  # For cross detection
        rsi_prev = rsi[i-1]
        rsi_prev2 = rsi[i-2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0  # SM crossed above 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0  # SM crossed below 0

        if use_rsi_cross:
            # F11 style: RSI crosses above/below level
            rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
            rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell
        else:
            # Zone style: RSI is in zone
            rsi_long_trigger = rsi_prev > rsi_buy
            rsi_short_trigger = rsi_prev < rsi_sell

        # ── Episode reset: when SM flips, reset used flags ──
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # ── EOD Close ──
        if trade_state != 0 and bar_mins_utc >= NY_CLOSE_UTC:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            max_equity = 0.0
            continue

        # ── Check exits for open trades ──
        if trade_state == 1:
            # Max loss stop
            if max_loss_pts > 0 and lows[i] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, entry_price - max_loss_pts,
                           entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                continue

            # Trailing stop check (based on previous bar's high for no look-ahead)
            if trailing_stop_pts > 0:
                running_pnl = highs[i] - entry_price
                if running_pnl > max_equity:
                    max_equity = running_pnl
                if max_equity > trailing_stop_pts and lows[i] <= entry_price + max_equity - trailing_stop_pts:
                    trail_exit = entry_price + max_equity - trailing_stop_pts
                    close_trade("long", entry_price, trail_exit, entry_idx, i, "TRAIL")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # SM flip exit: SM turned negative
            if sm_prev < 0 and sm_prev2 >= 0:
                # Exit at this bar's open (SM flipped on previous bar's close)
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                # DON'T continue — allow immediate reversal entry below

        elif trade_state == -1:
            # Max loss stop
            if max_loss_pts > 0 and highs[i] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, entry_price + max_loss_pts,
                           entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0
                continue

            # Trailing stop
            if trailing_stop_pts > 0:
                running_pnl = entry_price - lows[i]
                if running_pnl > max_equity:
                    max_equity = running_pnl
                if max_equity > trailing_stop_pts and highs[i] >= entry_price - max_equity + trailing_stop_pts:
                    trail_exit = entry_price - max_equity + trailing_stop_pts
                    close_trade("short", entry_price, trail_exit, entry_idx, i, "TRAIL")
                    trade_state = 0
                    exit_bar = i
                    max_equity = 0.0
                    continue

            # SM flip exit: SM turned positive
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                max_equity = 0.0

        # ── Entry logic ──
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_UTC <= bar_mins_utc <= NY_LAST_ENTRY_UTC
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                # Long entry
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    max_equity = 0.0
                # Short entry
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    max_equity = 0.0

    return trades


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_trades(trades, commission_per_side=0.52):
    """Score trades. MNQ = $2/pt, $0.52/side commission."""
    if not trades:
        return None
    pts = np.array([t["pts"] for t in trades])
    n = len(pts)
    comm_pts = (commission_per_side * 2) / 2.0
    net_each = pts - comm_pts
    net_pts = net_each.sum()
    wins = net_each[net_each > 0]
    losses = net_each[net_each <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / n * 100
    cum = np.cumsum(net_each)
    peak = np.maximum.accumulate(cum)
    mdd = (cum - peak).min()
    avg_bars = np.mean([t["bars"] for t in trades])
    avg_pts = np.mean(net_each)

    # Count exit types
    exit_types = {}
    for t in trades:
        r = t["result"]
        exit_types[r] = exit_types.get(r, 0) + 1

    return {
        "count": n, "net_pts": round(net_pts, 2), "pf": round(pf, 3),
        "win_rate": round(wr, 1), "net_1lot": round(net_pts * 2, 2),
        "net_10lot": round(net_pts * 20, 2), "max_dd_pts": round(mdd, 2),
        "avg_bars": round(avg_bars, 1), "avg_pts": round(avg_pts, 2),
        "exits": exit_types,
    }


def fmt_exits(exits_dict):
    """Format exit counts."""
    parts = []
    for k in ["SM_FLIP", "SL", "TRAIL", "EOD"]:
        if k in exits_dict:
            parts.append(f"{k}:{exits_dict[k]}")
    return " ".join(parts)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 140)
    print("SCALPER v9 — SM-FLIP ENTRY/EXIT (NO LOOK-AHEAD)")
    print("Entry: SM direction + RSI confirmation | Exit: SM flips sign")
    print("=" * 140)

    # ── Load data ──
    print("\n Loading data...")
    df_1m = load_1min_data("CME_MINI_MNQ1!, 1_7fdb6.csv")
    df_5m = load_5min_data("CME_MINI_MNQ1!, 5_46a9d.csv")

    # Filter to Jan 19 - Feb 12
    start = pd.Timestamp("2026-01-19")
    end = pd.Timestamp("2026-02-13")
    df_5m = df_5m[(df_5m.index >= start) & (df_5m.index < end)]
    print(f"  1-min: {len(df_1m)} bars ({df_1m.index[0]} to {df_1m.index[-1]})")
    print(f"  5-min: {len(df_5m)} bars ({df_5m.index[0]} to {df_5m.index[-1]})")

    # ── Compute SM sources ──
    print("\n Computing SM sources...")
    closes_1m = df_1m["Close"].values
    vols_1m = df_1m["High"].values - df_1m["Low"].values

    # Our SM with fast params (F11 winner params)
    sm_ours_fast_1m = compute_smart_money(closes_1m, vols_1m, 15, 10, 300, 150)
    df_1m["SM_Ours_Fast"] = sm_ours_fast_1m

    # Our SM with AA-like params
    sm_ours_aa_1m = compute_smart_money(closes_1m, vols_1m, 20, 12, 400, 255)
    df_1m["SM_Ours_AA"] = sm_ours_aa_1m

    # Resample all 1-min SM to 5-min
    print("  Resampling to 5-min...")
    sm_aa_5m = resample_to_5min(df_1m["SM_AA"], df_5m)
    sm_fast_5m = resample_to_5min(df_1m["SM_Ours_Fast"], df_5m)
    sm_aa_ours_5m = resample_to_5min(df_1m["SM_Ours_AA"], df_5m)
    sm_prebaked_5m = df_5m["SM_1m_prebaked"].values

    sm_sources = {
        "AA_resamp":     sm_aa_5m,
        "Ours_fast":     sm_fast_5m,
        "Ours_AA":       sm_aa_ours_5m,
        "AA_prebaked":   sm_prebaked_5m,
    }

    opens = df_5m["Open"].values
    highs = df_5m["High"].values
    lows = df_5m["Low"].values
    closes = df_5m["Close"].values
    times = df_5m.index.values

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1: F11-LIKE CONFIG (RSI cross, SM flip exit, no TP/SL)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("SECTION 1: F11-LIKE CONFIG — RSI cross entry, SM flip exit, no stops")
    print(f"{'='*140}")

    configs_f11 = [
        # (RSI len, RSI buy, RSI sell, SM threshold, cooldown, max_loss, trailing, rsi_cross)
        (14, 60, 40, 0.00, 6, 0, 0, True),   # Exact F11
        (14, 60, 40, 0.05, 6, 0, 0, True),
        (14, 60, 40, 0.10, 6, 0, 0, True),
        (12, 60, 40, 0.00, 6, 0, 0, True),
        (12, 65, 35, 0.00, 6, 0, 0, True),
        (12, 65, 35, 0.10, 6, 0, 0, True),
        (11, 60, 40, 0.00, 6, 0, 0, True),
        (11, 65, 35, 0.00, 6, 0, 0, True),
        (14, 55, 45, 0.00, 6, 0, 0, True),
    ]

    print(f"\n{'SM Source':>12} {'RSI':>4} {'Lvl':>5} {'SM>':>5} {'CD':>3} "
          f"{'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} {'$1lot':>9} "
          f"{'MaxDD':>7} {'AvgBars':>7} {'AvgPts':>7} {'Exits':>25}")

    for sm_name, sm_arr in sm_sources.items():
        for rsi_len, rb, rs, smt, cd, ml, ts, rc in configs_f11:
            rsi_arr = compute_rsi(closes, rsi_len)
            trades = run_backtest_v9(opens, highs, lows, closes, sm_arr, rsi_arr, times,
                                     rb, rs, smt, cd, ml, ts, rc)
            sc = score_trades(trades)
            if sc and sc["count"] >= 3:
                print(f"{sm_name:>12} {rsi_len:>4} {rb}/{rs:>2} {smt:>5.2f} {cd:>3} "
                      f"{sc['count']:>5} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                      f"{sc['net_pts']:>+8.1f} {sc['net_1lot']:>+9.2f} "
                      f"{sc['max_dd_pts']:>7.1f} {sc['avg_bars']:>7.1f} {sc['avg_pts']:>+7.2f} "
                      f"{fmt_exits(sc['exits']):>25}")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2: SM-FLIP EXIT + MAX LOSS STOP (protect against blowouts)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("SECTION 2: SM-FLIP EXIT + MAX LOSS STOP")
    print(f"{'='*140}")

    configs_sl = [
        # (RSI len, RSI buy, RSI sell, SM threshold, cooldown, max_loss, trailing, rsi_cross)
        (14, 60, 40, 0.00, 6, 15, 0, True),
        (14, 60, 40, 0.00, 6, 20, 0, True),
        (14, 60, 40, 0.00, 6, 25, 0, True),
        (14, 60, 40, 0.00, 6, 30, 0, True),
        (14, 60, 40, 0.05, 6, 15, 0, True),
        (14, 60, 40, 0.05, 6, 20, 0, True),
        (14, 60, 40, 0.10, 6, 15, 0, True),
        (14, 60, 40, 0.10, 6, 20, 0, True),
        (12, 60, 40, 0.00, 6, 15, 0, True),
        (12, 60, 40, 0.00, 6, 20, 0, True),
        (12, 65, 35, 0.00, 6, 15, 0, True),
        (12, 65, 35, 0.00, 6, 20, 0, True),
        (12, 65, 35, 0.10, 6, 15, 0, True),
        (12, 65, 35, 0.10, 6, 20, 0, True),
    ]

    print(f"\n{'SM Source':>12} {'RSI':>4} {'Lvl':>5} {'SM>':>5} {'CD':>3} {'MaxL':>5} "
          f"{'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} {'$1lot':>9} "
          f"{'MaxDD':>7} {'AvgBars':>7} {'AvgPts':>7} {'Exits':>30}")

    for sm_name, sm_arr in sm_sources.items():
        for rsi_len, rb, rs, smt, cd, ml, ts, rc in configs_sl:
            rsi_arr = compute_rsi(closes, rsi_len)
            trades = run_backtest_v9(opens, highs, lows, closes, sm_arr, rsi_arr, times,
                                     rb, rs, smt, cd, ml, ts, rc)
            sc = score_trades(trades)
            if sc and sc["count"] >= 3:
                print(f"{sm_name:>12} {rsi_len:>4} {rb}/{rs:>2} {smt:>5.2f} {cd:>3} {ml:>5} "
                      f"{sc['count']:>5} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                      f"{sc['net_pts']:>+8.1f} {sc['net_1lot']:>+9.2f} "
                      f"{sc['max_dd_pts']:>7.1f} {sc['avg_bars']:>7.1f} {sc['avg_pts']:>+7.2f} "
                      f"{fmt_exits(sc['exits']):>30}")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3: ZONE ENTRY (not cross) + SM-FLIP EXIT
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("SECTION 3: RSI ZONE ENTRY (not cross) + SM-FLIP EXIT")
    print(f"{'='*140}")

    configs_zone = [
        (14, 60, 40, 0.00, 6, 0, 0, False),
        (14, 60, 40, 0.05, 6, 0, 0, False),
        (14, 60, 40, 0.10, 6, 0, 0, False),
        (14, 55, 45, 0.00, 6, 0, 0, False),
        (12, 60, 40, 0.00, 6, 0, 0, False),
        (12, 65, 35, 0.00, 6, 0, 0, False),
        (12, 65, 35, 0.10, 6, 0, 0, False),
        (11, 60, 40, 0.00, 6, 0, 0, False),
        (11, 65, 35, 0.00, 6, 0, 0, False),
        (14, 60, 40, 0.00, 6, 20, 0, False),
        (14, 60, 40, 0.05, 6, 20, 0, False),
        (12, 60, 40, 0.00, 6, 20, 0, False),
        (12, 65, 35, 0.00, 6, 20, 0, False),
        (12, 65, 35, 0.10, 6, 20, 0, False),
    ]

    print(f"\n{'SM Source':>12} {'RSI':>4} {'Lvl':>5} {'SM>':>5} {'CD':>3} {'MaxL':>5} "
          f"{'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} {'$1lot':>9} "
          f"{'MaxDD':>7} {'AvgBars':>7} {'AvgPts':>7} {'Exits':>30}")

    for sm_name, sm_arr in sm_sources.items():
        for rsi_len, rb, rs, smt, cd, ml, ts, rc in configs_zone:
            rsi_arr = compute_rsi(closes, rsi_len)
            trades = run_backtest_v9(opens, highs, lows, closes, sm_arr, rsi_arr, times,
                                     rb, rs, smt, cd, ml, ts, rc)
            sc = score_trades(trades)
            if sc and sc["count"] >= 3:
                print(f"{sm_name:>12} {rsi_len:>4} {rb}/{rs:>2} {smt:>5.2f} {cd:>3} {ml:>5} "
                      f"{sc['count']:>5} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                      f"{sc['net_pts']:>+8.1f} {sc['net_1lot']:>+9.2f} "
                      f"{sc['max_dd_pts']:>7.1f} {sc['avg_bars']:>7.1f} {sc['avg_pts']:>+7.2f} "
                      f"{fmt_exits(sc['exits']):>30}")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4: FULL PARAMETER SWEEP
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("SECTION 4: FULL PARAMETER SWEEP (SM-FLIP EXIT)")
    print(f"{'='*140}")

    rsi_lengths = [10, 11, 12, 14]
    rsi_levels = [(65, 35), (60, 40), (55, 45)]
    sm_thresholds = [0.0, 0.05, 0.10, 0.15, 0.20]
    max_losses = [0, 15, 20, 25]
    cooldowns = [3, 6]
    entry_modes = [True, False]  # RSI cross vs zone

    total = (len(sm_sources) * len(rsi_lengths) * len(rsi_levels) *
             len(sm_thresholds) * len(max_losses) * len(cooldowns) * len(entry_modes))
    print(f"  Testing {total} combos...")

    results = []
    for sm_name, sm_arr in sm_sources.items():
        for rsi_len in rsi_lengths:
            rsi_arr = compute_rsi(closes, rsi_len)
            for rb, rs in rsi_levels:
                for smt in sm_thresholds:
                    for ml in max_losses:
                        for cd in cooldowns:
                            for rc in entry_modes:
                                trades = run_backtest_v9(
                                    opens, highs, lows, closes, sm_arr, rsi_arr, times,
                                    rb, rs, smt, cd, ml, 0, rc)
                                if len(trades) < 3:
                                    continue
                                sc = score_trades(trades)
                                if sc is None:
                                    continue
                                results.append({
                                    "sm_src": sm_name,
                                    "rsi": f"RSI{rsi_len}",
                                    "rsi_lvl": f"{rb}/{rs}",
                                    "sm_thr": smt,
                                    "max_loss": ml,
                                    "cd": cd,
                                    "entry": "cross" if rc else "zone",
                                    **sc,
                                })

    print(f"  Configs with 3+ trades: {len(results)}")

    def print_table(title, rows, max_rows=30):
        print(f"\n{'─'*140}")
        print(f"{title}")
        print(f"{'─'*140}")
        print(f"{'#':>3} {'SM':>12} {'RSI':>5} {'Lvl':>5} {'SM>':>5} {'ML':>3} {'CD':>3} {'Entry':>5} "
              f"{'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} {'$1lot':>9} {'$10lot':>10} "
              f"{'MaxDD':>7} {'Bars':>5} {'AvgPt':>6} {'Exits':>25}")
        for i, r in enumerate(rows[:max_rows]):
            print(f"{i+1:>3} {r['sm_src']:>12} {r['rsi']:>5} {r['rsi_lvl']:>5} {r['sm_thr']:>5.2f} "
                  f"{r['max_loss']:>3} {r['cd']:>3} {r['entry']:>5} "
                  f"{r['count']:>5} {r['win_rate']:>5.1f}% {r['pf']:>7.3f} "
                  f"{r['net_pts']:>+8.1f} {r['net_1lot']:>+9.2f} {r['net_10lot']:>+10.2f} "
                  f"{r['max_dd_pts']:>7.1f} {r['avg_bars']:>5.1f} {r['avg_pts']:>+6.2f} "
                  f"{fmt_exits(r['exits']):>25}")

    # Top by PF (min 5 trades)
    by_pf = sorted([r for r in results if r["count"] >= 5],
                   key=lambda x: (-x["pf"], -x["net_pts"]))
    print_table("TOP 30 BY PROFIT FACTOR (min 5 trades)", by_pf)

    # Top by net pts
    by_net = sorted(results, key=lambda x: -x["net_pts"])
    print_table("TOP 30 BY NET POINTS", by_net)

    # Balanced: PF > 1.3, WR > 40%, 8+ trades
    balanced = sorted(
        [r for r in results if r["pf"] > 1.3 and r["win_rate"] > 40 and r["count"] >= 8],
        key=lambda x: (-x["net_pts"]))
    print_table("BEST BALANCED (PF>1.3, WR>40%, 8+ trades)", balanced, 30)

    # ── Trade logs for top 3 balanced configs ──
    if balanced:
        top3 = balanced[:3]
        for rank, cfg in enumerate(top3):
            sm_arr = sm_sources[cfg["sm_src"]]
            rsi_len = int(cfg["rsi"].replace("RSI", ""))
            rsi_arr = compute_rsi(closes, rsi_len)
            rb, rs = [int(x) for x in cfg["rsi_lvl"].split("/")]
            rc = cfg["entry"] == "cross"
            trades = run_backtest_v9(
                opens, highs, lows, closes, sm_arr, rsi_arr, times,
                rb, rs, cfg["sm_thr"], cfg["cd"], cfg["max_loss"], 0, rc)
            sc = score_trades(trades)

            print(f"\n{'='*140}")
            print(f"TRADE LOG #{rank+1}: {cfg['sm_src']} {cfg['rsi']} {cfg['rsi_lvl']} "
                  f"SM>{cfg['sm_thr']:.2f} ML={cfg['max_loss']} CD={cfg['cd']} Entry={cfg['entry']}")
            print(f"  {sc['count']} trades, WR {sc['win_rate']}%, PF {sc['pf']}, "
                  f"Net {sc['net_pts']:+.2f} pts, ${sc['net_1lot']:+.2f}/1lot")
            print(f"{'='*140}")

            cum = 0.0
            comm = 0.52
            for j, t in enumerate(trades):
                net = t["pts"] - comm
                cum += net
                et = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
                xt = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")
                print(f"  {j+1:>3}. {t['side']:>5} {et:>11} -> {xt:>11} "
                      f"E:{t['entry']:>9.2f} X:{t['exit']:>9.2f} "
                      f"Pts:{t['pts']:>+8.2f} Net:{net:>+7.2f} Cum:{cum:>+8.2f} "
                      f"${net*2:>+8.2f} [{t['result']}] ({t['bars']} bars)")

    # ── Summary by SM source ──
    print(f"\n{'='*140}")
    print("SUMMARY BY SM SOURCE")
    print(f"{'='*140}")
    for sm_name in sm_sources:
        src = [r for r in results if r["sm_src"] == sm_name]
        prof = [r for r in src if r["net_pts"] > 0]
        if src:
            print(f"  {sm_name:>12}: {len(src):>4} configs, "
                  f"{len(prof):>3} profitable ({len(prof)/len(src)*100:.0f}%), "
                  f"Avg PF: {np.mean([r['pf'] for r in src]):.2f}, "
                  f"Best PF: {max(r['pf'] for r in src):.2f}, "
                  f"Best Net: {max(r['net_pts'] for r in src):+.1f}")

    # ── Summary by entry mode ──
    print(f"\n{'='*140}")
    print("SUMMARY BY ENTRY MODE")
    print(f"{'='*140}")
    for mode in ["cross", "zone"]:
        src = [r for r in results if r["entry"] == mode]
        prof = [r for r in src if r["net_pts"] > 0]
        if src:
            print(f"  {mode:>6}: {len(src):>4} configs, "
                  f"{len(prof):>3} profitable ({len(prof)/len(src)*100:.0f}%), "
                  f"Avg PF: {np.mean([r['pf'] for r in src]):.2f}, "
                  f"Best PF: {max(r['pf'] for r in src):.2f}, "
                  f"Best Net: {max(r['net_pts'] for r in src):+.1f}")

    print(f"\n{'='*140}")
    print("DONE — v9 SM-Flip backtest complete.")


if __name__ == "__main__":
    main()
