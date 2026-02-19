"""
Cross-Instrument SM Agreement Analysis
=======================================
Does ES/MES Smart Money agreeing with MNQ Smart Money predict better trades?

Approach:
1. Load MNQ + ES 1-min data (both have SM Net Index from AlgoAlpha)
2. Resample both to 5-min bars (last SM value in each window, OHLC)
3. Run v9 strategy on MNQ 5-min data
4. For each MNQ trade, check if ES SM agreed at entry/during/exit
5. Split trades into Agreement vs Disagreement groups
6. Compare win rate, avg P&L, profit factor
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---- Data Loading ----

def load_1min_with_sm(filepath: str) -> pd.DataFrame:
    """Load 1-min CSV that has SM Net Index column."""
    df = pd.read_csv(filepath)
    cols = df.columns.tolist()

    # Find SM Net Index column
    sm_col = None
    for i, c in enumerate(cols):
        if 'SM Net Index' in c:
            sm_col = i
            break

    if sm_col is None:
        raise ValueError(f"No SM Net Index column found in {filepath}")

    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df[cols[0]].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df[cols[1]], errors='coerce')
    result['High'] = pd.to_numeric(df[cols[2]], errors='coerce')
    result['Low'] = pd.to_numeric(df[cols[3]], errors='coerce')
    result['Close'] = pd.to_numeric(df[cols[4]], errors='coerce')
    result['SM_Net'] = pd.to_numeric(df[cols[sm_col]], errors='coerce').fillna(0)
    result = result.set_index('Time')
    return result


def resample_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 5-min: OHLC standard, SM = last value in window."""
    df_5m = df_1m.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'SM_Net': 'last',  # Last 1-min SM value in the 5-min window
    }).dropna(subset=['Open'])
    return df_5m


# ---- RSI Computation ----

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


# ---- v9 Backtest Engine ----

def run_backtest_v9(opens, highs, lows, closes, sm, rsi, times,
                     rsi_buy, rsi_sell, sm_threshold,
                     cooldown_bars,
                     max_loss_pts=0,
                     use_rsi_cross=True):
    """v9 SM-Flip strategy -- same as the original backtest script."""
    n = len(opens)
    trades = []
    trade_state = 0
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    NY_OPEN_UTC = 15 * 60         # 10:00 AM ET = 15:00 UTC
    NY_LAST_ENTRY_UTC = 20 * 60 + 45  # 3:45 PM ET = 20:45 UTC
    NY_CLOSE_UTC = 21 * 60        # 4:00 PM ET = 21:00 UTC

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
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

        # EOD Close
        if trade_state != 0 and bar_mins_utc >= NY_CLOSE_UTC:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # Check exits
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


# ---- Scoring ----

def score_group(trades, label, commission_per_side=0.52):
    """Score a group of trades."""
    if not trades:
        return {"label": label, "count": 0}
    pts_arr = np.array([t["pts"] for t in trades])
    comm_pts = (commission_per_side * 2) / 2.0  # $0.52 * 2 sides / $2 per pt
    net_each = pts_arr - comm_pts
    wins = net_each[net_each > 0]
    losses = net_each[net_each <= 0]
    w_sum = wins.sum() if len(wins) > 0 else 0
    l_sum = abs(losses.sum()) if len(losses) > 0 else 0
    pf = w_sum / l_sum if l_sum > 0 else 999.0
    wr = len(wins) / len(net_each) * 100
    return {
        "label": label,
        "count": len(trades),
        "win_rate": round(wr, 1),
        "pf": round(pf, 3),
        "net_pts": round(net_each.sum(), 2),
        "avg_pts": round(net_each.mean(), 2),
        "net_dollar": round(net_each.sum() * 2, 2),
        "avg_dollar": round(net_each.mean() * 2, 2),
        "wins": len(wins),
        "losses": len(losses),
    }


# ---- Main Analysis ----

def main():
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    DOWNLOADS = Path("/Users/jasongeorge/Downloads")

    print("=" * 140)
    print("CROSS-INSTRUMENT SM AGREEMENT ANALYSIS")
    print("Does ES/MES Smart Money agreeing with MNQ predict trade quality?")
    print("=" * 140)

    # ================================================================
    # PART 1: MNQ + ES (Jan 18 - Feb 13 overlap)
    # ================================================================
    print("\n" + "=" * 140)
    print("PART 1: MNQ + ES (full history overlap)")
    print("=" * 140)

    print("\nLoading MNQ 1-min (0efb1, Jan 18 - Feb 13, has SM)...")
    mnq_1m = load_1min_with_sm(DATA_DIR / "CME_MINI_MNQ1!, 1_0efb1.csv")
    print(f"  {len(mnq_1m)} bars: {mnq_1m.index[0]} to {mnq_1m.index[-1]}")

    print("Loading ES 1-min (b6c2e, Jan 18 - Feb 13, has SM)...")
    es_1m = load_1min_with_sm(DATA_DIR / "CME_MINI_ES1!, 1_b6c2e.csv")
    print(f"  {len(es_1m)} bars: {es_1m.index[0]} to {es_1m.index[-1]}")

    # Find overlap
    overlap_start = max(mnq_1m.index[0], es_1m.index[0])
    overlap_end = min(mnq_1m.index[-1], es_1m.index[-1])
    print(f"\nOverlap: {overlap_start} to {overlap_end}")

    # Trim to overlap
    mnq_1m = mnq_1m[(mnq_1m.index >= overlap_start) & (mnq_1m.index <= overlap_end)]
    es_1m = es_1m[(es_1m.index >= overlap_start) & (es_1m.index <= overlap_end)]
    print(f"  MNQ trimmed: {len(mnq_1m)} bars")
    print(f"  ES trimmed: {len(es_1m)} bars")

    # Resample to 5-min
    print("\nResampling to 5-min...")
    mnq_5m = resample_to_5min(mnq_1m)
    es_5m = resample_to_5min(es_1m)
    print(f"  MNQ 5-min: {len(mnq_5m)} bars")
    print(f"  ES 5-min: {len(es_5m)} bars")

    # Align by index (inner join on timestamps)
    common_idx = mnq_5m.index.intersection(es_5m.index)
    print(f"  Common 5-min timestamps: {len(common_idx)}")

    mnq_5m_aligned = mnq_5m.loc[common_idx]
    es_5m_aligned = es_5m.loc[common_idx]

    # Run v9 strategy on MNQ
    print("\nRunning v9 strategy on MNQ 5-min data...")
    print("  Params: RSI=10, Buy=55, Sell=45, SM_thr=0.00, Cooldown=3, Cross=True")

    opens = mnq_5m_aligned['Open'].values
    highs = mnq_5m_aligned['High'].values
    lows = mnq_5m_aligned['Low'].values
    closes = mnq_5m_aligned['Close'].values
    sm_mnq = mnq_5m_aligned['SM_Net'].values
    times = mnq_5m_aligned.index.values

    rsi_arr = compute_rsi(closes, 10)

    trades = run_backtest_v9(
        opens, highs, lows, closes, sm_mnq, rsi_arr, times,
        rsi_buy=55, rsi_sell=45, sm_threshold=0.00,
        cooldown_bars=3, max_loss_pts=0, use_rsi_cross=True
    )

    print(f"\n  Total trades: {len(trades)}")
    if len(trades) == 0:
        print("  No trades generated. Check data alignment.")
    else:
        # For each trade, look up ES SM
        sm_es = es_5m_aligned['SM_Net'].values
        all_times = mnq_5m_aligned.index

        comm_pts = 0.52  # commission per side

        agree_trades = []
        disagree_trades = []

        print("\n" + "-" * 160)
        print(f"{'#':>3} {'Side':>5} {'Entry Time':>18} {'Exit Time':>18} "
              f"{'Entry':>10} {'Exit':>10} {'Pts':>8} {'Net$':>8} "
              f"{'MNQ_SM':>8} {'ES_SM':>8} {'Agree':>6} {'ES_dur':>10} {'Exit':>6}")
        print("-" * 160)

        cum_pnl = 0.0
        for j, t in enumerate(trades):
            entry_i = t["entry_idx"]
            exit_i = t["exit_idx"]

            # MNQ SM at entry (the previous bar that triggered)
            mnq_sm_at_entry = sm_mnq[entry_i - 1] if entry_i > 0 else sm_mnq[entry_i]

            # ES SM at entry
            es_sm_at_entry = sm_es[entry_i - 1] if entry_i > 0 else sm_es[entry_i]

            # Agreement: same sign
            mnq_sign = 1 if mnq_sm_at_entry > 0 else (-1 if mnq_sm_at_entry < 0 else 0)
            es_sign = 1 if es_sm_at_entry > 0 else (-1 if es_sm_at_entry < 0 else 0)
            agree_at_entry = (mnq_sign == es_sign) and (mnq_sign != 0)

            # ES SM during the trade: did it disagree at any point?
            es_during = sm_es[entry_i:exit_i+1]
            if t["side"] == "long":
                # For long trades, disagreement = ES SM goes negative
                es_disagree_during = any(es_during < 0)
            else:
                # For short trades, disagreement = ES SM goes positive
                es_disagree_during = any(es_during > 0)

            # ES SM at exit
            es_sm_at_exit = sm_es[exit_i] if exit_i < len(sm_es) else sm_es[-1]

            # Net P&L
            net_pts = t["pts"] - comm_pts
            net_dollar = net_pts * 2
            cum_pnl += net_dollar

            # Categorize
            if agree_at_entry:
                agree_trades.append(t)
                agree_str = "YES"
            else:
                disagree_trades.append(t)
                agree_str = "NO"

            es_dur_str = "disagr" if es_disagree_during else "agreed"

            entry_time_str = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
            exit_time_str = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")

            print(f"{j+1:>3} {t['side']:>5} {entry_time_str:>18} {exit_time_str:>18} "
                  f"{t['entry']:>10.2f} {t['exit']:>10.2f} {t['pts']:>+8.2f} {net_dollar:>+8.2f} "
                  f"{mnq_sm_at_entry:>+8.4f} {es_sm_at_entry:>+8.4f} {agree_str:>6} "
                  f"{es_dur_str:>10} {t['result']:>6}")

        print("-" * 160)
        print(f"  Cumulative P&L: ${cum_pnl:+.2f}")

        # ---- Score groups ----
        print("\n" + "=" * 100)
        print("RESULTS: MNQ + ES AGREEMENT ANALYSIS")
        print("=" * 100)

        all_sc = score_group(trades, "ALL TRADES")
        agr_sc = score_group(agree_trades, "AGREEMENT (ES SM same sign as MNQ SM)")
        dis_sc = score_group(disagree_trades, "DISAGREEMENT (ES SM opposite/zero vs MNQ SM)")

        for sc in [all_sc, agr_sc, dis_sc]:
            if sc["count"] > 0:
                print(f"\n  {sc['label']}:")
                print(f"    Trades: {sc['count']}, Wins: {sc['wins']}, Losses: {sc['losses']}")
                print(f"    Win Rate: {sc['win_rate']}%")
                print(f"    Profit Factor: {sc['pf']}")
                print(f"    Net Points: {sc['net_pts']:+.2f}")
                print(f"    Avg Points/trade: {sc['avg_pts']:+.2f}")
                print(f"    Net $ (1 lot): {sc['net_dollar']:+.2f}")
                print(f"    Avg $ (1 lot): {sc['avg_dollar']:+.2f}")
            else:
                print(f"\n  {sc['label']}: No trades")

        # ---- Losing trades analysis ----
        print("\n" + "-" * 100)
        print("LOSING TRADES ANALYSIS: Was ES SM disagreeing?")
        print("-" * 100)

        losing_trades = [t for t in trades if (t["pts"] - comm_pts) <= 0]
        losing_agree = 0
        losing_disagree = 0
        losing_es_flipped = 0

        for t in losing_trades:
            entry_i = t["entry_idx"]
            exit_i = t["exit_idx"]
            mnq_sm_entry = sm_mnq[entry_i - 1] if entry_i > 0 else sm_mnq[entry_i]
            es_sm_entry = sm_es[entry_i - 1] if entry_i > 0 else sm_es[entry_i]

            mnq_sign = 1 if mnq_sm_entry > 0 else (-1 if mnq_sm_entry < 0 else 0)
            es_sign = 1 if es_sm_entry > 0 else (-1 if es_sm_entry < 0 else 0)
            agree = (mnq_sign == es_sign) and (mnq_sign != 0)

            if agree:
                losing_agree += 1
            else:
                losing_disagree += 1

            # Did ES flip during the trade?
            es_during = sm_es[entry_i:exit_i+1]
            if t["side"] == "long":
                if any(es_during < 0):
                    losing_es_flipped += 1
            else:
                if any(es_during > 0):
                    losing_es_flipped += 1

        print(f"  Total losing trades: {len(losing_trades)}")
        print(f"  Losers with ES agreeing at entry: {losing_agree} ({losing_agree/max(len(losing_trades),1)*100:.1f}%)")
        print(f"  Losers with ES disagreeing at entry: {losing_disagree} ({losing_disagree/max(len(losing_trades),1)*100:.1f}%)")
        print(f"  Losers where ES flipped during trade: {losing_es_flipped} ({losing_es_flipped/max(len(losing_trades),1)*100:.1f}%)")

    # ================================================================
    # PART 2: MNQ (Feb 13) + MES (Feb 13) — today's data
    # ================================================================
    print("\n\n" + "=" * 140)
    print("PART 2: MNQ + MES (Feb 13 only -- today's data)")
    print("=" * 140)

    print("\nLoading MNQ Feb 13 1-min (8835e)...")
    mnq_feb13 = load_1min_with_sm(DATA_DIR / "CME_MINI_MNQ1!, 1_8835e.csv")
    print(f"  {len(mnq_feb13)} bars: {mnq_feb13.index[0]} to {mnq_feb13.index[-1]}")

    print("Loading MES 1-min (016a0, Jan 25 - Feb 13)...")
    mes_1m = load_1min_with_sm(DOWNLOADS / "CME_MINI_MES1!, 1_016a0.csv")
    print(f"  {len(mes_1m)} bars: {mes_1m.index[0]} to {mes_1m.index[-1]}")

    # Also load MES Feb 13 specific file
    print("Loading MES Feb 13 1-min (37616)...")
    mes_feb13 = load_1min_with_sm(DOWNLOADS / "CME_MINI_MES1!, 1_37616.csv")
    print(f"  {len(mes_feb13)} bars: {mes_feb13.index[0]} to {mes_feb13.index[-1]}")

    # Use the longer MES file, trim to overlap with MNQ Feb 13
    overlap_start2 = max(mnq_feb13.index[0], mes_1m.index[0])
    overlap_end2 = min(mnq_feb13.index[-1], mes_1m.index[-1])
    print(f"\nOverlap: {overlap_start2} to {overlap_end2}")

    mnq_feb13_trim = mnq_feb13[(mnq_feb13.index >= overlap_start2) & (mnq_feb13.index <= overlap_end2)]
    mes_trim = mes_1m[(mes_1m.index >= overlap_start2) & (mes_1m.index <= overlap_end2)]
    print(f"  MNQ trimmed: {len(mnq_feb13_trim)} bars")
    print(f"  MES trimmed: {len(mes_trim)} bars")

    if len(mnq_feb13_trim) < 10:
        print("  Not enough overlapping data for Feb 13 MNQ+MES analysis.")
    else:
        mnq_5m_2 = resample_to_5min(mnq_feb13_trim)
        mes_5m_2 = resample_to_5min(mes_trim)

        common_idx2 = mnq_5m_2.index.intersection(mes_5m_2.index)
        print(f"  Common 5-min timestamps: {len(common_idx2)}")

        if len(common_idx2) < 10:
            print("  Not enough common 5-min bars.")
        else:
            mnq_5m_2a = mnq_5m_2.loc[common_idx2]
            mes_5m_2a = mes_5m_2.loc[common_idx2]

            opens2 = mnq_5m_2a['Open'].values
            highs2 = mnq_5m_2a['High'].values
            lows2 = mnq_5m_2a['Low'].values
            closes2 = mnq_5m_2a['Close'].values
            sm_mnq2 = mnq_5m_2a['SM_Net'].values
            sm_mes2 = mes_5m_2a['SM_Net'].values
            times2 = mnq_5m_2a.index.values

            rsi_arr2 = compute_rsi(closes2, 10)

            trades2 = run_backtest_v9(
                opens2, highs2, lows2, closes2, sm_mnq2, rsi_arr2, times2,
                rsi_buy=55, rsi_sell=45, sm_threshold=0.00,
                cooldown_bars=3, max_loss_pts=0, use_rsi_cross=True
            )

            print(f"\n  Feb 13 trades: {len(trades2)}")

            if trades2:
                print("\n" + "-" * 160)
                print(f"{'#':>3} {'Side':>5} {'Entry Time':>18} {'Exit Time':>18} "
                      f"{'Entry':>10} {'Exit':>10} {'Pts':>8} {'Net$':>8} "
                      f"{'MNQ_SM':>8} {'MES_SM':>8} {'Agree':>6} {'MES_dur':>10} {'Exit':>6}")
                print("-" * 160)

                agree2 = []
                disagree2 = []
                cum2 = 0.0

                for j, t in enumerate(trades2):
                    entry_i = t["entry_idx"]
                    exit_i = t["exit_idx"]

                    mnq_sm_e = sm_mnq2[entry_i-1] if entry_i > 0 else sm_mnq2[entry_i]
                    mes_sm_e = sm_mes2[entry_i-1] if entry_i > 0 else sm_mes2[entry_i]

                    mnq_s = 1 if mnq_sm_e > 0 else (-1 if mnq_sm_e < 0 else 0)
                    mes_s = 1 if mes_sm_e > 0 else (-1 if mes_sm_e < 0 else 0)
                    agree = (mnq_s == mes_s) and (mnq_s != 0)

                    mes_during = sm_mes2[entry_i:exit_i+1]
                    if t["side"] == "long":
                        mes_disagree = any(mes_during < 0)
                    else:
                        mes_disagree = any(mes_during > 0)

                    net_pts = t["pts"] - comm_pts
                    net_dollar = net_pts * 2
                    cum2 += net_dollar

                    if agree:
                        agree2.append(t)
                        a_str = "YES"
                    else:
                        disagree2.append(t)
                        a_str = "NO"

                    dur_str = "disagr" if mes_disagree else "agreed"
                    et = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
                    xt = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")

                    print(f"{j+1:>3} {t['side']:>5} {et:>18} {xt:>18} "
                          f"{t['entry']:>10.2f} {t['exit']:>10.2f} {t['pts']:>+8.2f} {net_dollar:>+8.2f} "
                          f"{mnq_sm_e:>+8.4f} {mes_sm_e:>+8.4f} {a_str:>6} "
                          f"{dur_str:>10} {t['result']:>6}")

                print("-" * 160)
                print(f"  Cumulative P&L: ${cum2:+.2f}")

                all_sc2 = score_group(trades2, "ALL FEB 13 TRADES")
                agr_sc2 = score_group(agree2, "AGREEMENT (MES SM same sign as MNQ SM)")
                dis_sc2 = score_group(disagree2, "DISAGREEMENT (MES SM opposite/zero vs MNQ SM)")

                print("\n" + "=" * 100)
                print("RESULTS: MNQ + MES FEB 13 AGREEMENT ANALYSIS")
                print("=" * 100)

                for sc in [all_sc2, agr_sc2, dis_sc2]:
                    if sc["count"] > 0:
                        print(f"\n  {sc['label']}:")
                        print(f"    Trades: {sc['count']}, Wins: {sc['wins']}, Losses: {sc['losses']}")
                        print(f"    Win Rate: {sc['win_rate']}%")
                        print(f"    Profit Factor: {sc['pf']}")
                        print(f"    Net Points: {sc['net_pts']:+.2f}")
                        print(f"    Avg Points/trade: {sc['avg_pts']:+.2f}")
                        print(f"    Net $ (1 lot): {sc['net_dollar']:+.2f}")
                    else:
                        print(f"\n  {sc['label']}: No trades")

    # ================================================================
    # PART 3: MNQ + MES full overlap (Jan 25 - Feb 13)
    # ================================================================
    print("\n\n" + "=" * 140)
    print("PART 3: MNQ + MES (full overlap Jan 25 - Feb 13)")
    print("=" * 140)

    # Use MNQ 0efb1 (Jan 18 - Feb 13) and MES 016a0 (Jan 25 - Feb 13)
    mnq_full = load_1min_with_sm(DATA_DIR / "CME_MINI_MNQ1!, 1_0efb1.csv")
    mes_full = load_1min_with_sm(DOWNLOADS / "CME_MINI_MES1!, 1_016a0.csv")

    overlap_s3 = max(mnq_full.index[0], mes_full.index[0])
    overlap_e3 = min(mnq_full.index[-1], mes_full.index[-1])
    print(f"Overlap: {overlap_s3} to {overlap_e3}")

    mnq_full = mnq_full[(mnq_full.index >= overlap_s3) & (mnq_full.index <= overlap_e3)]
    mes_full = mes_full[(mes_full.index >= overlap_s3) & (mes_full.index <= overlap_e3)]
    print(f"  MNQ trimmed: {len(mnq_full)} bars")
    print(f"  MES trimmed: {len(mes_full)} bars")

    mnq_5m_3 = resample_to_5min(mnq_full)
    mes_5m_3 = resample_to_5min(mes_full)

    common_idx3 = mnq_5m_3.index.intersection(mes_5m_3.index)
    print(f"  Common 5-min timestamps: {len(common_idx3)}")

    mnq_5m_3a = mnq_5m_3.loc[common_idx3]
    mes_5m_3a = mes_5m_3.loc[common_idx3]

    opens3 = mnq_5m_3a['Open'].values
    highs3 = mnq_5m_3a['High'].values
    lows3 = mnq_5m_3a['Low'].values
    closes3 = mnq_5m_3a['Close'].values
    sm_mnq3 = mnq_5m_3a['SM_Net'].values
    sm_mes3 = mes_5m_3a['SM_Net'].values
    times3 = mnq_5m_3a.index.values

    rsi_arr3 = compute_rsi(closes3, 10)

    trades3 = run_backtest_v9(
        opens3, highs3, lows3, closes3, sm_mnq3, rsi_arr3, times3,
        rsi_buy=55, rsi_sell=45, sm_threshold=0.00,
        cooldown_bars=3, max_loss_pts=0, use_rsi_cross=True
    )

    print(f"\n  Total trades: {len(trades3)}")

    if trades3:
        agree3 = []
        disagree3 = []

        print("\n" + "-" * 160)
        print(f"{'#':>3} {'Side':>5} {'Entry Time':>18} {'Exit Time':>18} "
              f"{'Entry':>10} {'Exit':>10} {'Pts':>8} {'Net$':>8} "
              f"{'MNQ_SM':>8} {'MES_SM':>8} {'Agree':>6} {'MES_dur':>10} {'Exit':>6}")
        print("-" * 160)

        cum3 = 0.0
        for j, t in enumerate(trades3):
            entry_i = t["entry_idx"]
            exit_i = t["exit_idx"]

            mnq_sm_e = sm_mnq3[entry_i-1] if entry_i > 0 else sm_mnq3[entry_i]
            mes_sm_e = sm_mes3[entry_i-1] if entry_i > 0 else sm_mes3[entry_i]

            mnq_s = 1 if mnq_sm_e > 0 else (-1 if mnq_sm_e < 0 else 0)
            mes_s = 1 if mes_sm_e > 0 else (-1 if mes_sm_e < 0 else 0)
            agree = (mnq_s == mes_s) and (mnq_s != 0)

            mes_during = sm_mes3[entry_i:exit_i+1]
            if t["side"] == "long":
                mes_disagree = any(mes_during < 0)
            else:
                mes_disagree = any(mes_during > 0)

            net_pts = t["pts"] - comm_pts
            net_dollar = net_pts * 2
            cum3 += net_dollar

            if agree:
                agree3.append(t)
                a_str = "YES"
            else:
                disagree3.append(t)
                a_str = "NO"

            dur_str = "disagr" if mes_disagree else "agreed"
            et = pd.Timestamp(t["entry_time"]).strftime("%m/%d %H:%M")
            xt = pd.Timestamp(t["exit_time"]).strftime("%m/%d %H:%M")

            print(f"{j+1:>3} {t['side']:>5} {et:>18} {xt:>18} "
                  f"{t['entry']:>10.2f} {t['exit']:>10.2f} {t['pts']:>+8.2f} {net_dollar:>+8.2f} "
                  f"{mnq_sm_e:>+8.4f} {mes_sm_e:>+8.4f} {a_str:>6} "
                  f"{dur_str:>10} {t['result']:>6}")

        print("-" * 160)
        print(f"  Cumulative P&L: ${cum3:+.2f}")

        all_sc3 = score_group(trades3, "ALL TRADES (MNQ+MES overlap)")
        agr_sc3 = score_group(agree3, "AGREEMENT (MES SM same sign as MNQ SM)")
        dis_sc3 = score_group(disagree3, "DISAGREEMENT (MES SM opposite/zero vs MNQ SM)")

        print("\n" + "=" * 100)
        print("RESULTS: MNQ + MES FULL OVERLAP AGREEMENT ANALYSIS")
        print("=" * 100)

        for sc in [all_sc3, agr_sc3, dis_sc3]:
            if sc["count"] > 0:
                print(f"\n  {sc['label']}:")
                print(f"    Trades: {sc['count']}, Wins: {sc['wins']}, Losses: {sc['losses']}")
                print(f"    Win Rate: {sc['win_rate']}%")
                print(f"    Profit Factor: {sc['pf']}")
                print(f"    Net Points: {sc['net_pts']:+.2f}")
                print(f"    Avg Points/trade: {sc['avg_pts']:+.2f}")
                print(f"    Net $ (1 lot): {sc['net_dollar']:+.2f}")
                print(f"    Avg $ (1 lot): {sc['avg_dollar']:+.2f}")
            else:
                print(f"\n  {sc['label']}: No trades")

        # Losing trades analysis for MNQ+MES
        print("\n" + "-" * 100)
        print("LOSING TRADES ANALYSIS (MNQ+MES): Was MES SM disagreeing?")
        print("-" * 100)

        losing3 = [t for t in trades3 if (t["pts"] - comm_pts) <= 0]
        l_agree3 = 0
        l_disagree3 = 0
        l_flipped3 = 0

        for t in losing3:
            entry_i = t["entry_idx"]
            exit_i = t["exit_idx"]
            mnq_sm_e = sm_mnq3[entry_i-1] if entry_i > 0 else sm_mnq3[entry_i]
            mes_sm_e = sm_mes3[entry_i-1] if entry_i > 0 else sm_mes3[entry_i]

            mnq_s = 1 if mnq_sm_e > 0 else (-1 if mnq_sm_e < 0 else 0)
            mes_s = 1 if mes_sm_e > 0 else (-1 if mes_sm_e < 0 else 0)
            agree = (mnq_s == mes_s) and (mnq_s != 0)

            if agree:
                l_agree3 += 1
            else:
                l_disagree3 += 1

            mes_during = sm_mes3[entry_i:exit_i+1]
            if t["side"] == "long":
                if any(mes_during < 0):
                    l_flipped3 += 1
            else:
                if any(mes_during > 0):
                    l_flipped3 += 1

        print(f"  Total losing trades: {len(losing3)}")
        if losing3:
            print(f"  Losers with MES agreeing at entry: {l_agree3} ({l_agree3/len(losing3)*100:.1f}%)")
            print(f"  Losers with MES disagreeing at entry: {l_disagree3} ({l_disagree3/len(losing3)*100:.1f}%)")
            print(f"  Losers where MES flipped during trade: {l_flipped3} ({l_flipped3/len(losing3)*100:.1f}%)")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n\n" + "=" * 140)
    print("FINAL SUMMARY: CROSS-INSTRUMENT SM AGREEMENT AS TRADE FILTER")
    print("=" * 140)
    print("""
Key question: Should we only take MNQ trades when ES/MES SM agrees?

If Agreement group has significantly better WR, PF, and avg P&L than Disagreement:
  -> Cross-instrument SM confirmation is a valid trade quality filter
  -> Could implement as: only enter when both MNQ and ES SM agree on direction

If no significant difference:
  -> Cross-instrument SM does not add predictive value
  -> Stick with single-instrument SM signal
""")


if __name__ == "__main__":
    main()
