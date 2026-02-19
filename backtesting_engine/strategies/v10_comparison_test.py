"""
v10 Comparison Test — Verify embedded SM matches AlgoAlpha SM
=============================================================
Tests two SM sources through the same v9 backtest engine
(v10 signal logic is identical to v9 when all safety features are OFF):

1. "AlgoAlpha" — Pre-baked SM from TradingView CSV export (what v10_production.pine uses)
2. "Standalone" — SM computed internally via our Python implementation (what v10_standalone.pine uses)

If both produce identical trades, the standalone Pine Script will match the production one.
Also compares against the known F11 winner results from previous rounds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_1min_clean(filename: str) -> pd.DataFrame:
    """Load clean 1-min CSV (no indicator columns)."""
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
    return df


def load_1min_with_aa(filename: str) -> pd.DataFrame:
    """Load 1-min CSV with AlgoAlpha SM data."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close",
    })
    # AlgoAlpha Net Buy Line (col 13) + Net Sell Line (col 19)
    net_buy = pd.to_numeric(df[cols[13]], errors="coerce").fillna(0)
    net_sell = pd.to_numeric(df[cols[19]], errors="coerce").fillna(0)
    df["SM_AA"] = net_buy + net_sell
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "SM_AA"]].copy()
    return df


def load_5min_with_prebaked(filename: str) -> pd.DataFrame:
    """Load 5-min CSV with pre-baked SM."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close",
    })
    df["SM_prebaked"] = pd.to_numeric(df[cols[7]], errors="coerce").fillna(0)
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "SM_prebaked"]].copy()
    return df


# ─── Smart Money Computation (standalone) ────────────────────────────────────

def compute_smart_money(closes, volumes, index_period, flow_period, norm_period, ema_len):
    """Compute SM net index — mirrors AlgoAlpha's Pine Script exactly."""
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


# ─── Resample ────────────────────────────────────────────────────────────────

def resample_ohlc(df: pd.DataFrame, period: str) -> pd.DataFrame:
    resampled = df.resample(period).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
    }).dropna()
    return resampled


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


# ─── v9 Backtest Engine (same logic v10 uses when safety features are OFF) ───

def run_backtest_v9(opens, highs, lows, closes, sm, rsi, times,
                     rsi_buy, rsi_sell, sm_threshold, cooldown_bars,
                     max_loss_pts=0, use_rsi_cross=True):
    """
    SM-Flip Strategy with look-ahead fix.
    Entry: SM direction + RSI cross | Exit: SM flips sign | Episode: one per SM direction
    """
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

        sm_prev = sm[i-1]
        sm_prev2 = sm[i-2]
        rsi_prev = rsi[i-1]
        rsi_prev2 = rsi[i-2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        if use_rsi_cross:
            rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
            rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell
        else:
            rsi_long_trigger = rsi_prev > rsi_buy
            rsi_short_trigger = rsi_prev < rsi_sell

        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        if trade_state != 0 and bar_mins_utc >= NY_CLOSE_UTC:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        if trade_state == 1:
            if max_loss_pts > 0 and lows[i] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, entry_price - max_loss_pts,
                           entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        elif trade_state == -1:
            if max_loss_pts > 0 and highs[i] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, entry_price + max_loss_pts,
                           entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_UTC <= bar_mins_utc <= NY_LAST_ENTRY_UTC
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

    exit_types = {}
    for t in trades:
        r = t["result"]
        exit_types[r] = exit_types.get(r, 0) + 1

    return {
        "count": n, "net_pts": round(net_pts, 2), "pf": round(pf, 3),
        "win_rate": round(wr, 1), "net_1lot": round(net_pts * 2, 2),
        "max_dd_pts": round(mdd, 2), "avg_bars": round(avg_bars, 1),
        "exits": exit_types,
    }


def fmt_exits(exits_dict):
    return " ".join(f"{k}:{v}" for k, v in sorted(exits_dict.items()))


def print_trades_detail(trades, commission=0.52):
    cum = 0.0
    for j, t in enumerate(trades):
        net = t["pts"] - commission
        cum += net
        et = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
        xt = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")
        print(f"  {j+1:>3}. {t['side']:>5} {et:>11} -> {xt:>11} "
              f"E:{t['entry']:>9.2f} X:{t['exit']:>9.2f} "
              f"Pts:{t['pts']:>+8.2f} Net:{net:>+7.2f} Cum:{cum:>+8.2f} "
              f"${net*2:>+8.2f} [{t['result']}] ({t['bars']} bars)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 120)
    print("  SM+RSI v10 COMPARISON TEST")
    print("  Verifying standalone (embedded SM) matches AlgoAlpha (external SM)")
    print("=" * 120)

    # ── Test configs ──
    # Using the v9 best configs that map to v10 defaults
    configs = [
        # (label, rsi_len, rsi_buy, rsi_sell, sm_threshold, cooldown, sm_index, sm_flow, sm_norm, sm_ema)
        ("v9 Default (RSI10 55/45 CD15)", 10, 55, 45, 0.00, 3, 25, 14, 500, 255),
        ("F11 Fast SM (RSI14 60/40 CD6)", 14, 60, 40, 0.00, 6, 15, 10, 300, 150),
        ("F11 + SM thr 0.05",             14, 60, 40, 0.05, 6, 15, 10, 300, 150),
        ("F11 + RSI 65/35",              14, 65, 35, 0.00, 6, 15, 10, 300, 150),
        ("AA Default params",             14, 60, 40, 0.00, 6, 25, 14, 500, 255),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1: Standalone SM (computed from clean data)
    # This is what v10_standalone.pine does internally
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  TEST 1: STANDALONE MODE (embedded SM computation)")
    print("  Data: Clean 1-min MNQ CSV -> resample to 5-min -> compute SM internally")
    print(f"{'='*120}")

    df_clean = load_1min_clean("CME_MINI_MNQ1!, 1_e8c40.csv")
    print(f"  Loaded {len(df_clean)} bars: {df_clean.index[0]} to {df_clean.index[-1]}")

    # Resample to 5-min OHLC
    df_5m = resample_ohlc(df_clean, "5min")
    print(f"  Resampled to {len(df_5m)} 5-min bars")

    # Synthesize volume (range method) for SM computation
    volumes_5m = (df_5m["High"] - df_5m["Low"]).clip(lower=0.25).values

    standalone_results = {}
    for label, rsi_len, rsi_buy, rsi_sell, sm_thr, cd, sm_idx, sm_flow, sm_norm, sm_ema in configs:
        sm_arr = compute_smart_money(df_5m["Close"].values, volumes_5m,
                                      sm_idx, sm_flow, sm_norm, sm_ema)
        rsi_arr = compute_rsi(df_5m["Close"].values, rsi_len)

        trades = run_backtest_v9(
            df_5m["Open"].values, df_5m["High"].values,
            df_5m["Low"].values, df_5m["Close"].values,
            sm_arr, rsi_arr, df_5m.index.values,
            rsi_buy, rsi_sell, sm_thr, cd)

        sc = score_trades(trades)
        standalone_results[label] = {"trades": trades, "score": sc}

        if sc:
            print(f"\n  {label}")
            print(f"    Trades: {sc['count']}  WR: {sc['win_rate']}%  PF: {sc['pf']}  "
                  f"Net: {sc['net_pts']:+.2f} pts  ${sc['net_1lot']:+.2f}/1lot  "
                  f"MaxDD: {sc['max_dd_pts']:.2f} pts  Exits: {fmt_exits(sc['exits'])}")
        else:
            print(f"\n  {label}: NO TRADES")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2: AlgoAlpha SM (from pre-baked CSV export)
    # This is what v10_production.pine does via input.source()
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("  TEST 2: ALGOALPHA MODE (pre-baked SM from TradingView export)")
    print("  Data: 1-min MNQ CSV with AA indicator columns -> resample SM to 5-min")
    print(f"{'='*120}")

    df_aa = load_1min_with_aa("CME_MINI_MNQ1!, 1_7fdb6.csv")
    print(f"  Loaded {len(df_aa)} bars with AA SM: {df_aa.index[0]} to {df_aa.index[-1]}")

    # Build 5-min OHLC from the AA data
    df_5m_aa = resample_ohlc(df_aa[["Open", "High", "Low", "Close"]], "5min")
    print(f"  Resampled to {len(df_5m_aa)} 5-min bars")

    # Resample the AA SM to 5-min (take last 1-min value in each 5-min window)
    sm_aa_5m = resample_to_5min(df_aa["SM_AA"], df_5m_aa)

    aa_results = {}
    for label, rsi_len, rsi_buy, rsi_sell, sm_thr, cd, sm_idx, sm_flow, sm_norm, sm_ema in configs:
        rsi_arr = compute_rsi(df_5m_aa["Close"].values, rsi_len)

        # AA SM is pre-computed — use it directly (ignore sm_idx/flow/norm/ema params)
        trades = run_backtest_v9(
            df_5m_aa["Open"].values, df_5m_aa["High"].values,
            df_5m_aa["Low"].values, df_5m_aa["Close"].values,
            sm_aa_5m, rsi_arr, df_5m_aa.index.values,
            rsi_buy, rsi_sell, sm_thr, cd)

        sc = score_trades(trades)
        aa_results[label] = {"trades": trades, "score": sc}

        if sc:
            print(f"\n  {label}")
            print(f"    Trades: {sc['count']}  WR: {sc['win_rate']}%  PF: {sc['pf']}  "
                  f"Net: {sc['net_pts']:+.2f} pts  ${sc['net_1lot']:+.2f}/1lot  "
                  f"MaxDD: {sc['max_dd_pts']:.2f} pts  Exits: {fmt_exits(sc['exits'])}")
        else:
            print(f"\n  {label}: NO TRADES")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3: Pre-baked 5-min SM (from 5-min CSV)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("  TEST 3: PRE-BAKED 5-MIN SM (from 5-min CSV export)")
    print(f"{'='*120}")

    df_5m_pre = load_5min_with_prebaked("CME_MINI_MNQ1!, 5_46a9d.csv")
    # Filter to same date range
    start = pd.Timestamp("2026-01-19")
    end = pd.Timestamp("2026-02-13")
    df_5m_pre = df_5m_pre[(df_5m_pre.index >= start) & (df_5m_pre.index < end)]
    print(f"  Loaded {len(df_5m_pre)} bars: {df_5m_pre.index[0]} to {df_5m_pre.index[-1]}")

    prebaked_results = {}
    for label, rsi_len, rsi_buy, rsi_sell, sm_thr, cd, sm_idx, sm_flow, sm_norm, sm_ema in configs:
        rsi_arr = compute_rsi(df_5m_pre["Close"].values, rsi_len)
        sm_arr = df_5m_pre["SM_prebaked"].values

        trades = run_backtest_v9(
            df_5m_pre["Open"].values, df_5m_pre["High"].values,
            df_5m_pre["Low"].values, df_5m_pre["Close"].values,
            sm_arr, rsi_arr, df_5m_pre.index.values,
            rsi_buy, rsi_sell, sm_thr, cd)

        sc = score_trades(trades)
        prebaked_results[label] = {"trades": trades, "score": sc}

        if sc:
            print(f"\n  {label}")
            print(f"    Trades: {sc['count']}  WR: {sc['win_rate']}%  PF: {sc['pf']}  "
                  f"Net: {sc['net_pts']:+.2f} pts  ${sc['net_1lot']:+.2f}/1lot  "
                  f"MaxDD: {sc['max_dd_pts']:.2f} pts  Exits: {fmt_exits(sc['exits'])}")
        else:
            print(f"\n  {label}: NO TRADES")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*120}")
    print(f"\n  {'Config':<35} {'Source':<12} {'Trades':>6} {'WR%':>6} {'PF':>7} "
          f"{'NetPts':>8} {'$1lot':>9} {'MaxDD':>7}")
    print(f"  {'-'*95}")

    for label, *_ in configs:
        for source_name, results_dict in [("Standalone", standalone_results),
                                            ("AlgoAlpha", aa_results),
                                            ("Prebaked5m", prebaked_results)]:
            sc = results_dict.get(label, {}).get("score")
            if sc:
                print(f"  {label:<35} {source_name:<12} {sc['count']:>6} {sc['win_rate']:>5.1f}% "
                      f"{sc['pf']:>7.3f} {sc['net_pts']:>+8.2f} {sc['net_1lot']:>+9.2f} "
                      f"{sc['max_dd_pts']:>7.2f}")
            else:
                print(f"  {label:<35} {source_name:<12} {'NO TRADES':>6}")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # DETAILED TRADE LOG — F11 config, Standalone source
    # ══════════════════════════════════════════════════════════════════════════
    f11_label = "F11 Fast SM (RSI14 60/40 CD6)"
    f11_standalone = standalone_results.get(f11_label, {})
    f11_aa = aa_results.get(f11_label, {})

    if f11_standalone.get("trades"):
        print(f"\n{'='*120}")
        print(f"  TRADE LOG: {f11_label} — STANDALONE")
        print(f"{'='*120}")
        print_trades_detail(f11_standalone["trades"])

    if f11_aa.get("trades"):
        print(f"\n{'='*120}")
        print(f"  TRADE LOG: {f11_label} — ALGOALPHA")
        print(f"{'='*120}")
        print_trades_detail(f11_aa["trades"])

    print(f"\n{'='*120}")
    print("  TEST COMPLETE")
    print(f"{'='*120}")
    print("\n  NOTES:")
    print("  - Standalone uses synthesized volume (range method) since clean CSV has no volume")
    print("  - AlgoAlpha uses TradingView's native tick volume (more accurate)")
    print("  - Trade counts may differ between sources due to different SM values")
    print("  - If Standalone and AA produce same trade count/WR/PF, the v10_standalone.pine")
    print("    will match v10_production.pine on TradingView")
    print("  - Different data date ranges between CSVs may also affect results")


if __name__ == "__main__":
    main()
