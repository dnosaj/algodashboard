"""
Signal Quality Filters — Backtest on v9 SM-Flip Strategy
==========================================================
Tests three signal quality improvements as toggles on top of the v9 baseline:

1. RSI_CONFIRM: At entry bar, verify RSI still supports the trade
   - Long: RSI at entry bar must be > rsi_neutral (e.g., 50)
   - Short: RSI at entry bar must be < rsi_neutral
   - Rationale: Trade 16 on Feb 13 entered long while RSI was 43.7 (already bearish)

2. SM_MOMENTUM: Require SM to be trending in trade direction, not fading
   - Long: SM[i-1] > SM[i-1 - lookback] (SM is rising)
   - Short: SM[i-1] < SM[i-1 - lookback] (SM is falling)
   - Rationale: Trade 13 on Feb 13 entered short while SM was fading toward zero

3. MAX_LOSS_STOP: Hard stop loss in points
   - Already supported in v9 engine, but testing specific values
   - Rationale: Trade 16 lost 235 pts before SM flip; 25pt stop caps it

Each filter is tested independently and in combinations against the baseline.
Uses AA_prebaked SM source (the proven winner).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# ─── Data Loading (same as v9 backtest) ──────────────────────────────────────

def load_5min_data(filename: str) -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / filename)
    cols = df.columns.tolist()
    df = df.rename(columns={
        cols[0]: "Time", cols[1]: "Open", cols[2]: "High",
        cols[3]: "Low", cols[4]: "Close",
    })
    df["SM_1m_prebaked"] = pd.to_numeric(df[cols[7]], errors="coerce").fillna(0)
    df["SM_5m_prebaked"] = pd.to_numeric(df[cols[8]], errors="coerce").fillna(0)
    df["Time"] = pd.to_datetime(df["Time"].astype(int), unit="s")
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "SM_1m_prebaked", "SM_5m_prebaked"]].copy()
    return df


def compute_rsi(arr: np.ndarray, period: int) -> np.ndarray:
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


# ─── Enhanced Backtest Engine ────────────────────────────────────────────────

def run_backtest(opens, highs, lows, closes, sm, rsi, times,
                 rsi_buy, rsi_sell, sm_threshold, cooldown_bars,
                 max_loss_pts=0,
                 # Signal quality filters:
                 rsi_confirm=False,        # Filter 1: RSI confirmation at entry
                 rsi_confirm_level=50.0,   # RSI must be above/below this at entry
                 sm_momentum=False,        # Filter 2: SM must be trending in direction
                 sm_momentum_lookback=5,   # SM[i-1] vs SM[i-1-lookback]
                 ):
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

        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # EOD Close
        if trade_state != 0 and bar_mins_utc >= NY_CLOSE_UTC:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # Exits for open trades
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

        # Entry logic
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_UTC <= bar_mins_utc <= NY_LAST_ENTRY_UTC
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                # ── LONG ENTRY ──
                if sm_bull and rsi_long_trigger and not long_used:
                    # Filter 1: RSI confirmation at entry bar
                    if rsi_confirm and rsi[i-1] < rsi_confirm_level:
                        pass  # RSI doesn't confirm — skip
                    # Filter 2: SM momentum — SM must be rising
                    elif sm_momentum and i > sm_momentum_lookback + 1 and sm[i-1] <= sm[i-1-sm_momentum_lookback]:
                        pass  # SM fading — skip
                    else:
                        trade_state = 1
                        entry_price = opens[i]
                        entry_idx = i
                        long_used = True

                # ── SHORT ENTRY ──
                elif sm_bear and rsi_short_trigger and not short_used:
                    # Filter 1: RSI confirmation
                    if rsi_confirm and rsi[i-1] > (100 - rsi_confirm_level):
                        pass  # RSI doesn't confirm — skip
                    # Filter 2: SM momentum — SM must be falling
                    elif sm_momentum and i > sm_momentum_lookback + 1 and sm[i-1] >= sm[i-1-sm_momentum_lookback]:
                        pass  # SM fading — skip
                    else:
                        trade_state = -1
                        entry_price = opens[i]
                        entry_idx = i
                        short_used = True

    return trades


def score_trades(trades, commission_per_side=0.52):
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

    exit_types = {}
    for t in trades:
        r = t["result"]
        exit_types[r] = exit_types.get(r, 0) + 1

    return {
        "count": n, "net_pts": round(net_pts, 2), "pf": round(pf, 3),
        "win_rate": round(wr, 1), "net_1lot": round(net_pts * 2, 2),
        "max_dd_pts": round(mdd, 2), "avg_bars": round(avg_bars, 1),
        "avg_pts": round(avg_pts, 2), "exits": exit_types,
    }


def fmt_exits(exits_dict):
    parts = []
    for k in ["SM_FLIP", "SL", "EOD"]:
        if k in exits_dict:
            parts.append(f"{k}:{exits_dict[k]}")
    return " ".join(parts)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 150)
    print("SIGNAL QUALITY FILTERS — Testing on v9 SM-Flip Baseline")
    print("Filters: RSI_CONFIRM (RSI still valid at entry), SM_MOMENTUM (SM trending in direction)")
    print("=" * 150)

    # Load data
    print("\nLoading data...")
    df_5m = load_5min_data("CME_MINI_MNQ1!, 5_46a9d.csv")
    start = pd.Timestamp("2026-01-19")
    end = pd.Timestamp("2026-02-13")
    df_5m = df_5m[(df_5m.index >= start) & (df_5m.index < end)]
    print(f"  5-min: {len(df_5m)} bars ({df_5m.index[0]} to {df_5m.index[-1]})")

    sm = df_5m["SM_1m_prebaked"].values
    opens = df_5m["Open"].values
    highs = df_5m["High"].values
    lows = df_5m["Low"].values
    closes = df_5m["Close"].values
    times = df_5m.index.values

    # ════════════════════════════════════════════════════════════════════════════
    # Best v9 config: RSI10 55/45, SM>0.00, CD3, cross entry
    # ════════════════════════════════════════════════════════════════════════════
    rsi_arr = compute_rsi(closes, 10)
    base_params = dict(rsi_buy=55, rsi_sell=45, sm_threshold=0.00, cooldown_bars=3)

    # ── SECTION 1: Baseline (no filters) ──
    print(f"\n{'='*150}")
    print("SECTION 1: BASELINE — v9 best config, no signal quality filters")
    print(f"{'='*150}")

    header = (f"{'Config':>45} {'Trds':>5} {'WR%':>6} {'PF':>7} {'NetPts':>8} "
              f"{'$1lot':>9} {'MaxDD':>7} {'AvgBars':>7} {'AvgPts':>7} {'Exits':>25}")
    print(f"\n{header}")

    def run_and_print(label, **extra_params):
        params = {**base_params, **extra_params}
        trades = run_backtest(opens, highs, lows, closes, sm, rsi_arr, times, **params)
        sc = score_trades(trades)
        if sc and sc["count"] >= 1:
            print(f"{label:>45} {sc['count']:>5} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                  f"{sc['net_pts']:>+8.1f} {sc['net_1lot']:>+9.2f} "
                  f"{sc['max_dd_pts']:>7.1f} {sc['avg_bars']:>7.1f} {sc['avg_pts']:>+7.2f} "
                  f"{fmt_exits(sc['exits']):>25}")
        else:
            count_str = str(sc["count"]) if sc else "0"
            print(f"{label:>45}   {'NO TRADES' if not sc else count_str + ' trades (min 1)'}")
        return sc

    # Baseline
    run_and_print("BASELINE (no filters)")

    # ── SECTION 2: MAX LOSS STOP only ──
    print(f"\n{'='*150}")
    print("SECTION 2: MAX LOSS STOP (various levels)")
    print(f"{'='*150}\n{header}")

    for ml in [15, 20, 25, 30, 40, 50]:
        run_and_print(f"MaxLoss={ml}pts", max_loss_pts=ml)

    # ── SECTION 3: RSI CONFIRMATION ──
    print(f"\n{'='*150}")
    print("SECTION 3: RSI CONFIRMATION — Reject entries where RSI no longer supports direction")
    print(f"{'='*150}\n{header}")

    for level in [45, 48, 50, 52, 55]:
        run_and_print(f"RSI_confirm level={level}", rsi_confirm=True, rsi_confirm_level=level)

    # ── SECTION 4: SM MOMENTUM ──
    print(f"\n{'='*150}")
    print("SECTION 4: SM MOMENTUM — Require SM trending in trade direction")
    print(f"{'='*150}\n{header}")

    for lb in [3, 5, 8, 10]:
        run_and_print(f"SM_momentum lookback={lb}", sm_momentum=True, sm_momentum_lookback=lb)

    # ── SECTION 5: COMBINATIONS ──
    print(f"\n{'='*150}")
    print("SECTION 5: COMBINATIONS — Best filters together")
    print(f"{'='*150}\n{header}")

    # Max loss + RSI confirm
    for ml in [25, 30]:
        for rl in [48, 50]:
            run_and_print(f"MaxLoss={ml} + RSI_confirm={rl}",
                         max_loss_pts=ml, rsi_confirm=True, rsi_confirm_level=rl)

    # Max loss + SM momentum
    for ml in [25, 30]:
        for lb in [3, 5]:
            run_and_print(f"MaxLoss={ml} + SM_mom={lb}",
                         max_loss_pts=ml, sm_momentum=True, sm_momentum_lookback=lb)

    # RSI confirm + SM momentum
    for rl in [48, 50]:
        for lb in [3, 5]:
            run_and_print(f"RSI_confirm={rl} + SM_mom={lb}",
                         rsi_confirm=True, rsi_confirm_level=rl,
                         sm_momentum=True, sm_momentum_lookback=lb)

    # All three
    for ml in [25, 30]:
        for rl in [48, 50]:
            for lb in [3, 5]:
                run_and_print(f"ML={ml} + RSI={rl} + SM_mom={lb}",
                             max_loss_pts=ml, rsi_confirm=True, rsi_confirm_level=rl,
                             sm_momentum=True, sm_momentum_lookback=lb)

    # ── SECTION 6: Alternative RSI levels with best filters ──
    print(f"\n{'='*150}")
    print("SECTION 6: ALTERNATIVE RSI CONFIGS + BEST FILTERS")
    print(f"{'='*150}\n{header}")

    alt_configs = [
        (10, 55, 45, "RSI10 55/45"),
        (12, 55, 45, "RSI12 55/45"),
        (10, 65, 35, "RSI10 65/35"),
        (12, 65, 35, "RSI12 65/35"),
        (14, 60, 40, "RSI14 60/40"),
    ]

    for rlen, rb, rs, label in alt_configs:
        rsi_alt = compute_rsi(closes, rlen)
        for ml in [0, 25]:
            for rc, rl in [(False, 50), (True, 50)]:
                for sm_m, sm_lb in [(False, 5), (True, 5)]:
                    filter_label = []
                    if ml > 0:
                        filter_label.append(f"ML{ml}")
                    if rc:
                        filter_label.append(f"RC{rl}")
                    if sm_m:
                        filter_label.append(f"SM{sm_lb}")
                    flabel = "+".join(filter_label) if filter_label else "none"
                    full_label = f"{label} [{flabel}]"

                    trades = run_backtest(opens, highs, lows, closes, sm, rsi_alt, times,
                                         rsi_buy=rb, rsi_sell=rs, sm_threshold=0.00,
                                         cooldown_bars=3, max_loss_pts=ml,
                                         rsi_confirm=rc, rsi_confirm_level=rl,
                                         sm_momentum=sm_m, sm_momentum_lookback=sm_lb)
                    sc = score_trades(trades)
                    if sc and sc["count"] >= 3:
                        print(f"{full_label:>45} {sc['count']:>5} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                              f"{sc['net_pts']:>+8.1f} {sc['net_1lot']:>+9.2f} "
                              f"{sc['max_dd_pts']:>7.1f} {sc['avg_bars']:>7.1f} {sc['avg_pts']:>+7.2f} "
                              f"{fmt_exits(sc['exits']):>25}")


if __name__ == "__main__":
    main()
