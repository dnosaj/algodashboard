"""
MYM Investigation — Why doesn't the strategy work on Micro Dow?
================================================================
1. Run v9 baseline configs on MYM with pre-baked AlgoAlpha SM
2. Compare MYM market characteristics vs MNQ/ES
3. Investigate SM behavior differences
4. Optimize settings for MYM
5. Diagnose root cause
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data_with_sm(filename):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close",
    })
    # SM Net Index directly (col 9)
    df["SM_NetIndex"] = pd.to_numeric(df[cols[9]], errors="coerce").fillna(0)
    # Net Buy Line (col 18) + Net Sell Line (col 24)
    net_buy = pd.to_numeric(df[cols[18]], errors="coerce").fillna(0)
    net_sell = pd.to_numeric(df[cols[24]], errors="coerce").fillna(0)
    df["SM_AA"] = net_buy + net_sell
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "SM_NetIndex", "SM_AA"]].copy()
    return df


def resample_ohlc(df, period):
    resampled = df[["Open", "High", "Low", "Close"]].resample(period).agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last",
    }).dropna()
    return resampled


def resample_to_5min(series_1m, df_5m):
    result = np.zeros(len(df_5m))
    for i, ts in enumerate(df_5m.index):
        mask = (series_1m.index >= ts) & (series_1m.index < ts + pd.Timedelta(minutes=5))
        subset = series_1m.loc[mask]
        if len(subset) > 0:
            result[i] = subset.iloc[-1]
        elif i > 0:
            result[i] = result[i-1]
    return result


def compute_rsi(arr, period):
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


def compute_smart_money(closes, volumes, index_period, flow_period, norm_period, ema_len):
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


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest_v9(opens, highs, lows, closes, sm, rsi, times,
                     rsi_buy, rsi_sell, sm_threshold, cooldown_bars,
                     max_loss_pts=0, use_rsi_cross=True):
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


def score_trades(trades, commission_per_side, dollar_per_pt):
    if not trades:
        return None
    pts = np.array([t["pts"] for t in trades])
    n = len(pts)
    comm_pts = (commission_per_side * 2) / dollar_per_pt
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

    # Win/loss trade detail
    win_pts = np.mean(wins) if len(wins) > 0 else 0
    loss_pts = np.mean(losses) if len(losses) > 0 else 0

    exit_types = {}
    for t in trades:
        r = t["result"]
        exit_types[r] = exit_types.get(r, 0) + 1

    return {
        "count": n, "net_pts": round(net_pts, 2), "pf": round(pf, 3),
        "win_rate": round(wr, 1), "net_dollar": round(net_pts * dollar_per_pt, 2),
        "max_dd_pts": round(mdd, 2), "avg_bars": round(avg_bars, 1),
        "avg_pts": round(avg_pts, 2), "avg_win": round(win_pts, 2),
        "avg_loss": round(loss_pts, 2), "exits": exit_types,
    }


def fmt_exits(d):
    return " ".join(f"{k}:{v}" for k, v in sorted(d.items()))


def print_trades_detail(trades, dollar_per_pt, commission):
    cum = 0.0
    comm_pts = (commission * 2) / dollar_per_pt
    for j, t in enumerate(trades):
        net = t["pts"] - comm_pts
        cum += net
        et = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
        xt = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")
        print(f"  {j+1:>3}. {t['side']:>5} {et:>11} -> {xt:>11} "
              f"E:{t['entry']:>9.0f} X:{t['exit']:>9.0f} "
              f"Pts:{t['pts']:>+8.0f} Net:{net:>+8.1f} Cum:{cum:>+9.1f} "
              f"${net*dollar_per_pt:>+8.2f} [{t['result']}] ({t['bars']} bars)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 140)
    print("  MYM (MICRO DOW) INVESTIGATION")
    print("=" * 140)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: LOAD & CHARACTERIZE DATA
    # ══════════════════════════════════════════════════════════════════════
    df_mym = load_data_with_sm("CBOT_MINI_MYM1!, 1_b3131.csv")
    df_mnq = load_data_with_sm("CME_MINI_MNQ1!, 1_7fdb6.csv")
    df_es = load_data_with_sm("CME_MINI_ES1!, 1_b6c2e.csv")

    print(f"\n  MYM: {len(df_mym)} bars | {df_mym.index[0]} to {df_mym.index[-1]}")
    print(f"  MNQ: {len(df_mnq)} bars | {df_mnq.index[0]} to {df_mnq.index[-1]}")
    print(f"  ES:  {len(df_es)} bars  | {df_es.index[0]} to {df_es.index[-1]}")

    # Resample all to 5-min
    df_mym_5m = resample_ohlc(df_mym, "5min")
    df_mnq_5m = resample_ohlc(df_mnq, "5min")
    df_es_5m = resample_ohlc(df_es, "5min")

    sm_mym = resample_to_5min(df_mym["SM_AA"], df_mym_5m)
    sm_mnq = resample_to_5min(df_mnq["SM_AA"], df_mnq_5m)
    sm_es = resample_to_5min(df_es["SM_AA"], df_es_5m)

    print(f"\n  MYM 5m: {len(df_mym_5m)} bars")
    print(f"  MNQ 5m: {len(df_mnq_5m)} bars")
    print(f"  ES 5m:  {len(df_es_5m)} bars")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: MARKET CHARACTERISTICS COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print("  SECTION 2: MARKET CHARACTERISTICS — WHY MYM MAY DIFFER")
    print(f"{'='*140}")

    for label, df5, sm_arr, dpm, comm in [
        ("MYM", df_mym_5m, sm_mym, 0.50, 0.52),
        ("MNQ", df_mnq_5m, sm_mnq, 2.00, 0.52),
        ("ES",  df_es_5m,  sm_es,  50.0, 1.25),
    ]:
        closes = df5["Close"].values
        highs = df5["High"].values
        lows = df5["Low"].values
        ranges = highs - lows
        returns_pct = np.diff(closes) / closes[:-1] * 100

        # SM characteristics
        sm_nonzero = np.count_nonzero(sm_arr)
        sm_pos = np.sum(sm_arr > 0)
        sm_neg = np.sum(sm_arr < 0)
        sm_flips = np.sum(np.diff(np.sign(sm_arr)) != 0)
        sm_abs = np.abs(sm_arr[sm_arr != 0])

        # Commission as % of typical move
        avg_range = np.mean(ranges)
        comm_pts = (comm * 2) / dpm
        comm_pct_of_range = (comm_pts / avg_range) * 100 if avg_range > 0 else 0

        print(f"\n  {label}:")
        print(f"    Price level:       ~{np.mean(closes):,.0f}")
        print(f"    $/point:           ${dpm:.2f}")
        print(f"    Commission:        ${comm:.2f}/side = {comm_pts:.2f} pts round-trip")
        print(f"    Avg 5m range:      {avg_range:.2f} pts")
        print(f"    Commission/range:  {comm_pct_of_range:.1f}% of avg bar range")
        print(f"    Avg 5m return:     {np.mean(np.abs(returns_pct)):.4f}%")
        print(f"    Volatility (std):  {np.std(returns_pct):.4f}%")
        print(f"    SM non-zero:       {sm_nonzero}/{len(sm_arr)} ({sm_nonzero/len(sm_arr)*100:.0f}%)")
        print(f"    SM pos/neg:        {sm_pos}/{sm_neg}")
        print(f"    SM flips:          {sm_flips} ({sm_flips/len(sm_arr)*100:.1f}% of bars)")
        if len(sm_abs) > 0:
            print(f"    SM |value| mean:   {np.mean(sm_abs):.4f}")
            print(f"    SM |value| median: {np.median(sm_abs):.4f}")
            print(f"    SM |value| p25:    {np.percentile(sm_abs, 25):.4f}")
            print(f"    SM |value| p75:    {np.percentile(sm_abs, 75):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: V9 BASELINE RESULTS ON MYM
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print("  SECTION 3: V9 BASELINE — MYM vs MNQ vs ES (same configs)")
    print(f"{'='*140}")

    configs = [
        ("RSI14 60/40 SM>0.00 CD6 cross",  14, 60, 40, 0.00, 6, True),
        ("RSI14 60/40 SM>0.05 CD6 cross",  14, 60, 40, 0.05, 6, True),
        ("RSI14 65/35 SM>0.00 CD6 cross",  14, 65, 35, 0.00, 6, True),
        ("RSI10 55/45 SM>0.00 CD3 cross",  10, 55, 45, 0.00, 3, True),
        ("RSI14 60/40 SM>0.00 CD6 zone",   14, 60, 40, 0.00, 6, False),
        ("RSI12 60/40 SM>0.00 CD6 cross",  12, 60, 40, 0.00, 6, True),
    ]

    instruments = [
        ("MYM", df_mym_5m, sm_mym, 0.50, 0.52),
        ("MNQ", df_mnq_5m, sm_mnq, 2.00, 0.52),
        ("ES",  df_es_5m,  sm_es,  50.0, 1.25),
    ]

    print(f"\n  {'Config':<38} {'Inst':>4} {'Trds':>5} {'WR%':>6} {'PF':>7} "
          f"{'NetPts':>8} {'Net$':>9} {'AvgWin':>7} {'AvgLoss':>8} {'MaxDD':>7} {'Exits':>25}")
    print(f"  {'-'*130}")

    for cfg_label, rsi_len, rb, rs, smt, cd, rc in configs:
        for inst_label, df5, sm_arr, dpm, comm in instruments:
            rsi_arr = compute_rsi(df5["Close"].values, rsi_len)
            trades = run_backtest_v9(
                df5["Open"].values, df5["High"].values,
                df5["Low"].values, df5["Close"].values,
                sm_arr, rsi_arr, df5.index.values,
                rb, rs, smt, cd, 0, rc)
            sc = score_trades(trades, comm, dpm)
            if sc and sc["count"] >= 1:
                print(f"  {cfg_label:<38} {inst_label:>4} "
                      f"{sc['count']:>5} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                      f"{sc['net_pts']:>+8.1f} {sc['net_dollar']:>+9.2f} "
                      f"{sc['avg_win']:>+7.1f} {sc['avg_loss']:>+8.1f} "
                      f"{sc['max_dd_pts']:>7.1f} {fmt_exits(sc['exits']):>25}")
            else:
                print(f"  {cfg_label:<38} {inst_label:>4}     0 trades")
        print()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: MYM TRADE LOG (best v9 config)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*140}")
    print("  SECTION 4: MYM TRADE LOG — RSI14 60/40 SM>0.00 CD6 cross")
    print(f"{'='*140}")

    rsi_arr = compute_rsi(df_mym_5m["Close"].values, 14)
    trades = run_backtest_v9(
        df_mym_5m["Open"].values, df_mym_5m["High"].values,
        df_mym_5m["Low"].values, df_mym_5m["Close"].values,
        sm_mym, rsi_arr, df_mym_5m.index.values,
        60, 40, 0.00, 6, 0, True)
    if trades:
        print_trades_detail(trades, 0.50, 0.52)
    else:
        print("  NO TRADES")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: SM SIGNAL QUALITY — Do SM flips predict direction on MYM?
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print("  SECTION 5: SM SIGNAL QUALITY — Does SM predict direction?")
    print(f"{'='*140}")

    for label, df5, sm_arr in [
        ("MYM", df_mym_5m, sm_mym),
        ("MNQ", df_mnq_5m, sm_mnq),
        ("ES",  df_es_5m,  sm_es),
    ]:
        closes = df5["Close"].values
        n = len(closes)
        # When SM > 0, does price tend to go up? When SM < 0, does price go down?
        sm_pos_returns = []
        sm_neg_returns = []
        for i in range(1, n):
            ret = closes[i] - closes[i-1]
            if sm_arr[i-1] > 0:
                sm_pos_returns.append(ret)
            elif sm_arr[i-1] < 0:
                sm_neg_returns.append(ret)

        sm_pos_returns = np.array(sm_pos_returns)
        sm_neg_returns = np.array(sm_neg_returns)

        # After SM flips positive, what's the avg move over next N bars?
        flip_bull_moves = {1: [], 3: [], 5: [], 10: []}
        flip_bear_moves = {1: [], 3: [], 5: [], 10: []}
        for i in range(1, n):
            if sm_arr[i] > 0 and sm_arr[i-1] <= 0:  # flip bull
                for horizon in [1, 3, 5, 10]:
                    if i + horizon < n:
                        flip_bull_moves[horizon].append(closes[i + horizon] - closes[i])
            elif sm_arr[i] < 0 and sm_arr[i-1] >= 0:  # flip bear
                for horizon in [1, 3, 5, 10]:
                    if i + horizon < n:
                        flip_bear_moves[horizon].append(-(closes[i + horizon] - closes[i]))

        print(f"\n  {label}:")
        if len(sm_pos_returns) > 0:
            print(f"    SM>0 bars: {len(sm_pos_returns)}, avg next-bar return: {np.mean(sm_pos_returns):+.2f} pts, "
                  f"% positive: {(sm_pos_returns > 0).sum()/len(sm_pos_returns)*100:.1f}%")
        if len(sm_neg_returns) > 0:
            print(f"    SM<0 bars: {len(sm_neg_returns)}, avg next-bar return: {np.mean(sm_neg_returns):+.2f} pts, "
                  f"% negative: {(sm_neg_returns < 0).sum()/len(sm_neg_returns)*100:.1f}%")

        print(f"    After SM flips BULL — avg move:")
        for h in [1, 3, 5, 10]:
            arr = np.array(flip_bull_moves[h])
            if len(arr) > 0:
                print(f"      +{h} bars: {np.mean(arr):+.2f} pts, "
                      f"win rate: {(arr > 0).sum()/len(arr)*100:.0f}%, n={len(arr)}")
        print(f"    After SM flips BEAR — avg favorable move:")
        for h in [1, 3, 5, 10]:
            arr = np.array(flip_bear_moves[h])
            if len(arr) > 0:
                print(f"      +{h} bars: {np.mean(arr):+.2f} pts, "
                      f"win rate: {(arr > 0).sum()/len(arr)*100:.0f}%, n={len(arr)}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6: OPTIMIZATION SWEEP FOR MYM
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print("  SECTION 6: MYM OPTIMIZATION SWEEP")
    print(f"{'='*140}")

    rsi_lengths = [8, 10, 12, 14, 16, 20]
    rsi_levels = [(70, 30), (65, 35), (60, 40), (55, 45), (52, 48)]
    sm_thresholds = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    cooldowns = [1, 3, 6, 10]
    entry_modes = [True, False]
    max_losses = [0, 50, 100, 150]

    total = len(rsi_lengths) * len(rsi_levels) * len(sm_thresholds) * len(cooldowns) * len(entry_modes) * len(max_losses)
    print(f"  Testing {total} combinations...")

    results = []
    for rsi_len in rsi_lengths:
        rsi_arr = compute_rsi(df_mym_5m["Close"].values, rsi_len)
        for rb, rs in rsi_levels:
            for smt in sm_thresholds:
                for cd in cooldowns:
                    for rc in entry_modes:
                        for ml in max_losses:
                            trades = run_backtest_v9(
                                df_mym_5m["Open"].values, df_mym_5m["High"].values,
                                df_mym_5m["Low"].values, df_mym_5m["Close"].values,
                                sm_mym, rsi_arr, df_mym_5m.index.values,
                                rb, rs, smt, cd, ml, rc)
                            if len(trades) < 3:
                                continue
                            sc = score_trades(trades, 0.52, 0.50)
                            if sc is None:
                                continue
                            results.append({
                                "rsi": rsi_len, "lvl": f"{rb}/{rs}", "sm_thr": smt,
                                "cd": cd, "entry": "cross" if rc else "zone",
                                "max_loss": ml, **sc,
                            })

    print(f"  Configs with 3+ trades: {len(results)}")
    profitable = [r for r in results if r["net_pts"] > 0]
    print(f"  Profitable configs: {len(profitable)}")

    def print_results_table(title, rows, max_rows=25):
        print(f"\n  {title}")
        print(f"  {'RSI':>4} {'Lvl':>5} {'SM>':>5} {'CD':>3} {'Entry':>5} {'ML':>4} "
              f"{'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} {'Net$':>8} "
              f"{'AvgWin':>7} {'AvgLoss':>8} {'MaxDD':>7} {'Exits':>25}")
        print(f"  {'-'*125}")
        for r in rows[:max_rows]:
            print(f"  {r['rsi']:>4} {r['lvl']:>5} {r['sm_thr']:>5.2f} {r['cd']:>3} "
                  f"{r['entry']:>5} {r['max_loss']:>4} "
                  f"{r['count']:>5} {r['win_rate']:>5.1f}% {r['pf']:>7.3f} "
                  f"{r['net_pts']:>+8.1f} {r['net_dollar']:>+8.2f} "
                  f"{r['avg_win']:>+7.1f} {r['avg_loss']:>+8.1f} "
                  f"{r['max_dd_pts']:>7.1f} {fmt_exits(r['exits']):>25}")

    # Top by PF (min 5 trades)
    by_pf = sorted([r for r in results if r["count"] >= 5],
                   key=lambda x: (-x["pf"], -x["net_pts"]))
    print_results_table("TOP 25 BY PROFIT FACTOR (min 5 trades)", by_pf)

    # Top by net pts (min 5 trades)
    by_net = sorted([r for r in results if r["count"] >= 5],
                   key=lambda x: -x["net_pts"])
    print_results_table("TOP 25 BY NET POINTS (min 5 trades)", by_net)

    # Best balanced: PF > 1.2, 5+ trades
    balanced = sorted(
        [r for r in results if r["pf"] > 1.2 and r["count"] >= 5],
        key=lambda x: (-x["pf"] * 2 + -x["count"] * 0.1 + x["win_rate"] / 50))
    print_results_table("BEST BALANCED (PF>1.2, 5+ trades)", balanced)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7: STANDALONE SM ON MYM (computed with range volume)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print("  SECTION 7: STANDALONE SM ON MYM (for comparison)")
    print(f"{'='*140}")

    vols_mym = (df_mym["High"] - df_mym["Low"]).clip(lower=0.5).values
    closes_mym = df_mym["Close"].values

    sm_params = [
        ("AA Default 25/14/500/255", 25, 14, 500, 255),
        ("Fast 15/10/300/150",       15, 10, 300, 150),
        ("Ultra-fast 10/8/200/100",  10, 8, 200, 100),
        ("Slow 30/20/600/300",       30, 20, 600, 300),
    ]

    for sm_label, si, sf, sn, se in sm_params:
        sm_standalone = compute_smart_money(closes_mym, vols_mym, si, sf, sn, se)
        sm_5m = resample_to_5min(pd.Series(sm_standalone, index=df_mym.index), df_mym_5m)

        # Test a few configs
        test_configs = [
            (14, 60, 40, 0.00, 6, True),
            (14, 60, 40, 0.00, 6, False),
            (14, 65, 35, 0.00, 3, True),
            (10, 55, 45, 0.00, 3, True),
        ]

        print(f"\n  SM: {sm_label}")
        for rsi_len, rb, rs, smt, cd, rc in test_configs:
            rsi_arr = compute_rsi(df_mym_5m["Close"].values, rsi_len)
            trades = run_backtest_v9(
                df_mym_5m["Open"].values, df_mym_5m["High"].values,
                df_mym_5m["Low"].values, df_mym_5m["Close"].values,
                sm_5m, rsi_arr, df_mym_5m.index.values,
                rb, rs, smt, cd, 0, rc)
            sc = score_trades(trades, 0.52, 0.50)
            mode = "cross" if rc else "zone"
            if sc and sc["count"] >= 1:
                print(f"    RSI{rsi_len} {rb}/{rs} CD{cd} {mode}: "
                      f"{sc['count']} trades, {sc['win_rate']}% WR, PF {sc['pf']}, "
                      f"Net {sc['net_pts']:+.1f} pts, ${sc['net_dollar']:+.2f}")
            else:
                print(f"    RSI{rsi_len} {rb}/{rs} CD{cd} {mode}: 0 trades")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 8: DIAGNOSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print("  SECTION 8: DIAGNOSIS — WHY MYM IS DIFFERENT")
    print(f"{'='*140}")

    # SM flip duration comparison
    for label, sm_arr, df5 in [
        ("MYM", sm_mym, df_mym_5m),
        ("MNQ", sm_mnq, df_mnq_5m),
        ("ES",  sm_es,  df_es_5m),
    ]:
        # How long does SM stay in one direction before flipping?
        episodes = []
        current_sign = 0
        current_len = 0
        for i in range(len(sm_arr)):
            s = 1 if sm_arr[i] > 0 else (-1 if sm_arr[i] < 0 else 0)
            if s == current_sign:
                current_len += 1
            else:
                if current_len > 0:
                    episodes.append(current_len)
                current_sign = s
                current_len = 1
        if current_len > 0:
            episodes.append(current_len)
        episodes = np.array(episodes)

        print(f"\n  {label} — SM episode duration (5-min bars):")
        print(f"    Total episodes: {len(episodes)}")
        if len(episodes) > 0:
            print(f"    Mean: {np.mean(episodes):.1f} bars ({np.mean(episodes)*5:.0f} min)")
            print(f"    Median: {np.median(episodes):.0f} bars ({np.median(episodes)*5:.0f} min)")
            print(f"    p25/p75: {np.percentile(episodes, 25):.0f}/{np.percentile(episodes, 75):.0f} bars")
            print(f"    Max: {np.max(episodes)} bars ({np.max(episodes)*5:.0f} min)")
            # Short episodes (< 3 bars) = SM is choppy / noisy
            short = (episodes <= 2).sum()
            print(f"    Episodes <= 2 bars: {short} ({short/len(episodes)*100:.0f}%) — choppy SM")

    print(f"\n\n{'='*140}")
    print("  INVESTIGATION COMPLETE")
    print(f"{'='*140}")


if __name__ == "__main__":
    main()
