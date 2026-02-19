"""
SM + RSI Parameter Sweep for Aug-Dec Profitability
====================================================
Runs on 1-MIN bars matching Pine Script architecture exactly.

Phase 1: SM param sweep (400 combos) with fixed RSI 10/55/45
Phase 2: RSI sweep (5-min cross levels + 1-min RSI filter) on top SM sets
Phase 3: Top combos + ATR trailing stop
Phase 4: Monthly breakdown of best combos
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_databento_1min, resample_to_5min,
    compute_smart_money, compute_rsi, map_5min_rsi_to_1min,
    run_backtest_v10, score_trades,
)


def run_1min(opens, highs, lows, closes, sm, times,
             rsi_5m_curr, rsi_5m_prev,
             rsi_buy=55, rsi_sell=45, sm_threshold=0.0,
             cooldown=15, max_loss_pts=50,
             atr_trail_exit=False, atr_trail_mult=2.0, atr_period=14,
             rsi_1m=None, rsi_1m_long_min=0, rsi_1m_short_max=100):
    """Run backtest on 1-min arrays with mapped 5-min RSI."""
    return run_backtest_v10(
        opens, highs, lows, closes, sm, rsi_5m_curr, times,
        rsi_buy=rsi_buy, rsi_sell=rsi_sell,
        sm_threshold=sm_threshold, cooldown_bars=cooldown,
        max_loss_pts=max_loss_pts,
        rsi_5m_curr=rsi_5m_curr, rsi_5m_prev=rsi_5m_prev,
        atr_trail_exit=atr_trail_exit, atr_trail_mult=atr_trail_mult,
        atr_period=atr_period,
        rsi_1m=rsi_1m, rsi_1m_long_min=rsi_1m_long_min,
        rsi_1m_short_max=rsi_1m_short_max,
    )


def score_period(trades, start_date, end_date):
    """Score only trades within a date range."""
    filtered = [t for t in trades
                if start_date <= pd.Timestamp(t['entry_time']).date() < end_date]
    return score_trades(filtered)


def main():
    t0 = time.time()
    print("=" * 90)
    print("SM + RSI PARAMETER SWEEP (1-MIN BARS)")
    print("Finding profitable settings for Aug-Dec 2025")
    print("=" * 90)

    # Load 6-month Databento MNQ 1-min
    print("\nLoading Databento MNQ 1-min data...")
    df_1m = load_databento_1min('MNQ')
    print(f"  1-min: {len(df_1m)} bars, {df_1m.index[0].date()} to {df_1m.index[-1].date()}")

    # Pre-extract 1-min arrays
    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    volumes = df_1m['Volume'].values
    times = df_1m.index.values

    # 5-min for RSI mapping
    df_5m = resample_to_5min(df_1m)
    fivemin_times = df_5m.index.values
    fivemin_closes = df_5m['Close'].values
    print(f"  5-min: {len(df_5m)} bars (for RSI computation)")

    # Pre-compute default RSI mapping (RSI 10) for Phase 1
    rsi_curr_10, rsi_prev_10 = map_5min_rsi_to_1min(
        times, fivemin_times, fivemin_closes, rsi_len=10)

    # Pre-compute 1-min RSI for different lengths
    print("  Pre-computing 1-min RSI arrays...")
    rsi_1m_cache = {}
    for rl in [6, 8, 10, 12, 14]:
        rsi_1m_cache[rl] = compute_rsi(closes, rl)

    # Date ranges
    AUG_START = pd.Timestamp("2025-08-17").date()
    AUG_END = pd.Timestamp("2026-01-01").date()
    JAN_START = pd.Timestamp("2026-01-01").date()
    JAN_END = pd.Timestamp("2026-02-14").date()

    # =========================================================================
    # PHASE 1: SM Parameter Sweep
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHASE 1: SM PARAMETER SWEEP (fixed RSI 10/55/45, CD=15, SL=50)")
    print("=" * 90)

    sm_idx = [10, 15, 20, 25, 30]
    sm_flow = [8, 10, 12, 14]
    sm_norm = [200, 300, 400, 500]
    sm_ema = [100, 150, 200, 255, 350]

    total_sm = len(sm_idx) * len(sm_flow) * len(sm_norm) * len(sm_ema)
    print(f"  Grid: {len(sm_idx)}x{len(sm_flow)}x{len(sm_norm)}x{len(sm_ema)} = {total_sm} combos on {len(closes)} 1-min bars")

    results_p1 = []
    sm_cache = {}
    done = 0
    t1 = time.time()

    for ip, fp, np_, el in product(sm_idx, sm_flow, sm_norm, sm_ema):
        sm = compute_smart_money(closes, volumes, ip, fp, np_, el)
        sm_cache[(ip, fp, np_, el)] = sm

        trades = run_1min(opens, highs, lows, closes, sm, times,
                          rsi_curr_10, rsi_prev_10)
        sc_full = score_trades(trades)
        sc_aug = score_period(trades, AUG_START, AUG_END)
        sc_jan = score_period(trades, JAN_START, JAN_END)

        results_p1.append({
            'ip': ip, 'fp': fp, 'np': np_, 'el': el,
            'full': sc_full, 'aug': sc_aug, 'jan': sc_jan,
        })

        done += 1
        if done % 50 == 0:
            elapsed = time.time() - t1
            rate = done / elapsed
            remain = (total_sm - done) / rate
            print(f"  ... {done}/{total_sm} ({elapsed:.0f}s, ~{remain:.0f}s left)")

    elapsed_p1 = time.time() - t1
    print(f"\n  Phase 1: {total_sm} combos in {elapsed_p1:.1f}s ({total_sm/elapsed_p1:.1f}/s)")

    # Filter valid
    valid_p1 = [r for r in results_p1
                if r['aug'] is not None and r['aug']['count'] >= 50]
    valid_p1.sort(key=lambda r: r['aug']['pf'], reverse=True)

    # Both profitable
    both_p1 = [r for r in valid_p1
               if r['aug']['pf'] > 1.0
               and r['jan'] is not None and r['jan']['pf'] > 1.0]
    both_p1.sort(key=lambda r: (r['aug']['pf'] * r['jan']['pf']) ** 0.5, reverse=True)

    print(f"  Valid (>=50 Aug-Dec trades): {len(valid_p1)}")
    print(f"  Profitable BOTH periods: {len(both_p1)}")

    # Show top 20 Aug-Dec
    print(f"\n  TOP 20 by Aug-Dec PF:")
    print(f"  {'Rk':>3}  {'I':>2} {'F':>2} {'N':>3} {'E':>3}  |  "
          f"{'A-D#':>4} {'A-DWR':>5} {'A-DPF':>5} {'A-D$':>7} {'A-DD':>6}  |  "
          f"{'J-F#':>4} {'J-FPF':>5} {'J-F$':>7}  |  "
          f"{'F#':>3} {'FPF':>5} {'F$':>7} {'FDD':>6}")
    print("  " + "-" * 110)
    for rk, r in enumerate(valid_p1[:20], 1):
        a, j, f = r['aug'], r['jan'], r['full']
        j_n = j['count'] if j else 0; j_pf = j['pf'] if j else 0; j_d = j['net_dollar'] if j else 0
        f_n = f['count'] if f else 0; f_pf = f['pf'] if f else 0; f_d = f['net_dollar'] if f else 0; f_dd = f['max_dd_dollar'] if f else 0
        print(f"  {rk:>3}  {r['ip']:>2} {r['fp']:>2} {r['np']:>3} {r['el']:>3}  |  "
              f"{a['count']:>4} {a['win_rate']:>5.1f} {a['pf']:>5.3f} {a['net_dollar']:>+7.0f} {a['max_dd_dollar']:>6.0f}  |  "
              f"{j_n:>4} {j_pf:>5.3f} {j_d:>+7.0f}  |  "
              f"{f_n:>3} {f_pf:>5.3f} {f_d:>+7.0f} {f_dd:>6.0f}")

    if both_p1:
        print(f"\n  TOP 20 PROFITABLE IN BOTH (gMean PF):")
        print(f"  {'Rk':>3}  {'I':>2} {'F':>2} {'N':>3} {'E':>3}  |  "
              f"{'A-DPF':>5} {'A-D$':>7}  |  {'J-FPF':>5} {'J-F$':>7}  |  "
              f"{'FPF':>5} {'F$':>7} {'FDD':>6} {'gM':>5}")
        print("  " + "-" * 85)
        for rk, r in enumerate(both_p1[:20], 1):
            a, j, f = r['aug'], r['jan'], r['full']
            gm = (a['pf'] * j['pf']) ** 0.5
            f_pf = f['pf'] if f else 0; f_d = f['net_dollar'] if f else 0; f_dd = f['max_dd_dollar'] if f else 0
            print(f"  {rk:>3}  {r['ip']:>2} {r['fp']:>2} {r['np']:>3} {r['el']:>3}  |  "
                  f"{a['pf']:>5.3f} {a['net_dollar']:>+7.0f}  |  "
                  f"{j['pf']:>5.3f} {j['net_dollar']:>+7.0f}  |  "
                  f"{f_pf:>5.3f} {f_d:>+7.0f} {f_dd:>6.0f} {gm:>5.3f}")

    # Baseline
    base_r = next((r for r in results_p1
                   if r['ip']==20 and r['fp']==12 and r['np']==400 and r['el']==255), None)
    if base_r:
        print(f"\n  BASELINE SM(20/12/400/255):")
        for lbl, sc in [("Aug-Dec", base_r['aug']), ("Jan-Feb", base_r['jan']), ("Full", base_r['full'])]:
            if sc:
                print(f"    {lbl:>7}: {sc['count']}tr PF {sc['pf']} ${sc['net_dollar']:+.0f} DD ${sc['max_dd_dollar']:.0f}")

    # =========================================================================
    # PHASE 2: RSI Sweep (5-min cross + 1-min filter)
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHASE 2: RSI SWEEP + 1-MIN RSI FILTER on top SM sets")
    print("=" * 90)

    # Top SM sets
    if len(both_p1) >= 10:
        top_sm = both_p1[:10]
    elif len(both_p1) >= 3:
        top_sm = both_p1[:]
        extra = [r for r in valid_p1 if r not in top_sm]
        top_sm += extra[:10 - len(top_sm)]
    else:
        top_sm = valid_p1[:10]

    # Pre-compute 5-min RSI mappings
    rsi_5m_cache = {}
    for rl in [6, 8, 10, 12, 14]:
        rsi_5m_cache[rl] = map_5min_rsi_to_1min(
            times, fivemin_times, fivemin_closes, rsi_len=rl)

    # RSI sweep grid
    rsi_lens = [6, 8, 10, 12, 14]
    rsi_levels = [(50, 50), (55, 45), (60, 40), (65, 35)]
    cooldowns = [10, 15, 20, 30]
    # 1-min RSI filter: (rsi_1m_len, long_min, short_max)
    # 0/100 = disabled, otherwise acts as directional filter
    rsi_1m_configs = [
        (0, 0, 100),       # disabled (baseline)
        (10, 50, 50),      # 1m RSI(10) > 50 for longs, < 50 for shorts
        (10, 45, 55),      # looser: > 45 for longs, < 55 for shorts
        (10, 55, 45),      # tighter: > 55 for longs, < 45 for shorts
        (14, 50, 50),      # 1m RSI(14) > 50 / < 50
        (14, 45, 55),      # 1m RSI(14) looser
        (6, 50, 50),       # 1m RSI(6) > 50 / < 50 (faster)
    ]

    total_p2 = len(top_sm) * len(rsi_lens) * len(rsi_levels) * len(cooldowns) * len(rsi_1m_configs)
    print(f"  {len(top_sm)} SM x {len(rsi_lens)} RSI5m x {len(rsi_levels)} lvls x {len(cooldowns)} cd x {len(rsi_1m_configs)} rsi1m = {total_p2}")

    results_p2 = []
    t2 = time.time()
    done = 0

    for sm_r in top_sm:
        sm_key = (sm_r['ip'], sm_r['fp'], sm_r['np'], sm_r['el'])
        sm = sm_cache[sm_key]

        for rl in rsi_lens:
            rc, rp = rsi_5m_cache[rl]
            for rb, rs in rsi_levels:
                for cd in cooldowns:
                    for r1m_len, r1m_lmin, r1m_smax in rsi_1m_configs:
                        r1m = rsi_1m_cache.get(r1m_len) if r1m_len > 0 else None

                        trades = run_1min(
                            opens, highs, lows, closes, sm, times, rc, rp,
                            rsi_buy=rb, rsi_sell=rs, cooldown=cd, max_loss_pts=50,
                            rsi_1m=r1m, rsi_1m_long_min=r1m_lmin, rsi_1m_short_max=r1m_smax,
                        )
                        sc_full = score_trades(trades)
                        sc_aug = score_period(trades, AUG_START, AUG_END)
                        sc_jan = score_period(trades, JAN_START, JAN_END)

                        results_p2.append({
                            'ip': sm_r['ip'], 'fp': sm_r['fp'],
                            'np': sm_r['np'], 'el': sm_r['el'],
                            'rl': rl, 'rb': rb, 'rs': rs, 'cd': cd,
                            'r1m_len': r1m_len, 'r1m_lmin': r1m_lmin, 'r1m_smax': r1m_smax,
                            'full': sc_full, 'aug': sc_aug, 'jan': sc_jan,
                        })
                        done += 1

        if done % 500 == 0:
            elapsed = time.time() - t2
            rate = done / elapsed if elapsed > 0 else 1
            remain = (total_p2 - done) / rate
            print(f"  ... {done}/{total_p2} ({elapsed:.0f}s, ~{remain:.0f}s left)")

    elapsed_p2 = time.time() - t2
    print(f"  Phase 2: {total_p2} combos in {elapsed_p2:.1f}s ({total_p2/max(1,elapsed_p2):.1f}/s)")

    # Filter
    valid_p2 = [r for r in results_p2
                if (r['aug'] is not None and r['aug']['count'] >= 30
                    and r['jan'] is not None and r['jan']['count'] >= 10
                    and r['aug']['pf'] > 1.0 and r['jan']['pf'] > 1.0)]
    valid_p2.sort(key=lambda r: (r['aug']['pf'] * r['jan']['pf']) ** 0.5, reverse=True)

    print(f"  Profitable BOTH (>=30 A-D, >=10 J-F): {len(valid_p2)}")

    if valid_p2:
        print(f"\n  TOP 30 SM+RSI COMBOS (gMean PF):")
        print(f"  {'Rk':>3}  {'SM':>11} {'RSI5m':>9} {'CD':>2} {'RSI1m':>10}  |  "
              f"{'A-D#':>4} {'A-DPF':>5} {'A-D$':>7}  |  "
              f"{'J-F#':>4} {'J-FPF':>5} {'J-F$':>7}  |  "
              f"{'F#':>3} {'FPF':>5} {'F$':>7} {'FDD':>6} {'gM':>5}")
        print("  " + "-" * 120)
        for rk, r in enumerate(valid_p2[:30], 1):
            a, j, f = r['aug'], r['jan'], r['full']
            gm = (a['pf'] * j['pf']) ** 0.5
            sm_s = f"{r['ip']}/{r['fp']}/{r['np']}/{r['el']}"
            rsi5_s = f"{r['rl']}/{r['rb']}/{r['rs']}"
            rsi1_s = f"{r['r1m_len']}/{r['r1m_lmin']}/{r['r1m_smax']}" if r['r1m_len'] > 0 else "OFF"
            f_n = f['count'] if f else 0; f_pf = f['pf'] if f else 0
            f_d = f['net_dollar'] if f else 0; f_dd = f['max_dd_dollar'] if f else 0
            print(f"  {rk:>3}  {sm_s:>11} {rsi5_s:>9} {r['cd']:>2} {rsi1_s:>10}  |  "
                  f"{a['count']:>4} {a['pf']:>5.3f} {a['net_dollar']:>+7.0f}  |  "
                  f"{j['count']:>4} {j['pf']:>5.3f} {j['net_dollar']:>+7.0f}  |  "
                  f"{f_n:>3} {f_pf:>5.3f} {f_d:>+7.0f} {f_dd:>6.0f} {gm:>5.3f}")

    # Show best with 1-min RSI ON vs OFF
    with_1m = [r for r in valid_p2 if r['r1m_len'] > 0]
    without_1m = [r for r in valid_p2 if r['r1m_len'] == 0]
    print(f"\n  1-min RSI filter impact:")
    print(f"    With 1-min RSI ON:  {len(with_1m)} profitable combos")
    print(f"    With 1-min RSI OFF: {len(without_1m)} profitable combos")
    if with_1m:
        b = with_1m[0]
        print(f"    Best with 1m RSI:   SM({b['ip']}/{b['fp']}/{b['np']}/{b['el']}) "
              f"RSI5m({b['rl']}/{b['rb']}/{b['rs']}) RSI1m({b['r1m_len']}/{b['r1m_lmin']}/{b['r1m_smax']}) "
              f"CD={b['cd']}  Full PF={b['full']['pf'] if b['full'] else 0}")

    # =========================================================================
    # PHASE 3: Top Combos + ATR Trail
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHASE 3: TOP COMBOS + ATR TRAILING STOP")
    print("=" * 90)

    if valid_p2:
        top_c = valid_p2[:10]
    elif both_p1:
        top_c = [{**r, 'rl': 10, 'rb': 55, 'rs': 45, 'cd': 15,
                  'r1m_len': 0, 'r1m_lmin': 0, 'r1m_smax': 100}
                 for r in both_p1[:10]]
    else:
        top_c = [{**r, 'rl': 10, 'rb': 55, 'rs': 45, 'cd': 15,
                  'r1m_len': 0, 'r1m_lmin': 0, 'r1m_smax': 100}
                 for r in valid_p1[:10]]

    atr_combos = [
        (False, 0, 0),
        (True, 1.5, 14), (True, 1.5, 20),
        (True, 2.0, 14), (True, 2.0, 20),
        (True, 2.5, 20),
    ]
    sl_options = [0, 50, 75]

    total_p3 = len(top_c) * len(atr_combos) * len(sl_options)
    print(f"  {len(top_c)} combos x {len(atr_combos)} ATR x {len(sl_options)} SL = {total_p3}")

    results_p3 = []
    t3 = time.time()

    for c in top_c:
        sm_key = (c['ip'], c['fp'], c['np'], c['el'])
        sm = sm_cache[sm_key]
        rc, rp = rsi_5m_cache[c['rl']]
        r1m = rsi_1m_cache.get(c['r1m_len']) if c['r1m_len'] > 0 else None

        for atr_on, am, ap in atr_combos:
            for sl in sl_options:
                trades = run_1min(
                    opens, highs, lows, closes, sm, times, rc, rp,
                    rsi_buy=c['rb'], rsi_sell=c['rs'], cooldown=c['cd'],
                    max_loss_pts=sl,
                    atr_trail_exit=atr_on, atr_trail_mult=am, atr_period=ap,
                    rsi_1m=r1m, rsi_1m_long_min=c['r1m_lmin'], rsi_1m_short_max=c['r1m_smax'],
                )
                sc_full = score_trades(trades)
                sc_aug = score_period(trades, AUG_START, AUG_END)
                sc_jan = score_period(trades, JAN_START, JAN_END)

                results_p3.append({
                    **{k: c[k] for k in ['ip','fp','np','el','rl','rb','rs','cd',
                                          'r1m_len','r1m_lmin','r1m_smax']},
                    'atr_on': atr_on, 'am': am, 'ap': ap, 'sl': sl,
                    'full': sc_full, 'aug': sc_aug, 'jan': sc_jan,
                })

    elapsed_p3 = time.time() - t3
    print(f"  Phase 3: {total_p3} combos in {elapsed_p3:.1f}s")

    valid_p3 = [r for r in results_p3
                if (r['full'] is not None and r['full']['pf'] > 1.0
                    and r['aug'] is not None and r['aug']['pf'] > 1.0
                    and r['jan'] is not None and r['jan']['pf'] > 1.0)]
    valid_p3.sort(key=lambda r: r['full']['pf'], reverse=True)
    print(f"  Profitable ALL periods: {len(valid_p3)}")

    if valid_p3:
        print(f"\n  TOP 20 FULL CONFIGS (sorted by full PF):")
        print(f"  {'Rk':>3}  {'SM':>11} {'RSI5m':>9} {'CD':>2} {'RSI1m':>10} {'ATR':>7} {'SL':>2}  |  "
              f"{'A-DPF':>5} {'A-D$':>7}  |  {'J-FPF':>5} {'J-F$':>7}  |  "
              f"{'F#':>3} {'FPF':>5} {'F$':>7} {'FDD':>6}")
        print("  " + "-" * 120)
        for rk, r in enumerate(valid_p3[:20], 1):
            a, j, f = r['aug'], r['jan'], r['full']
            sm_s = f"{r['ip']}/{r['fp']}/{r['np']}/{r['el']}"
            rsi5_s = f"{r['rl']}/{r['rb']}/{r['rs']}"
            rsi1_s = f"{r['r1m_len']}/{r['r1m_lmin']}/{r['r1m_smax']}" if r['r1m_len'] > 0 else "OFF"
            atr_s = f"{r['am']:.1f}x{r['ap']}" if r['atr_on'] else "OFF"
            print(f"  {rk:>3}  {sm_s:>11} {rsi5_s:>9} {r['cd']:>2} {rsi1_s:>10} {atr_s:>7} {r['sl']:>2}  |  "
                  f"{a['pf']:>5.3f} {a['net_dollar']:>+7.0f}  |  "
                  f"{j['pf']:>5.3f} {j['net_dollar']:>+7.0f}  |  "
                  f"{f['count']:>3} {f['pf']:>5.3f} {f['net_dollar']:>+7.0f} {f['max_dd_dollar']:>6.0f}")

    # =========================================================================
    # PHASE 4: Monthly Breakdown
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHASE 4: MONTHLY BREAKDOWN")
    print("=" * 90)

    if valid_p3:
        final_top = valid_p3[:5]
    elif valid_p2:
        final_top = [{**r, 'atr_on': False, 'am': 0, 'ap': 0, 'sl': 50} for r in valid_p2[:5]]
    elif both_p1:
        final_top = [{**r, 'rl': 10, 'rb': 55, 'rs': 45, 'cd': 15,
                      'r1m_len': 0, 'r1m_lmin': 0, 'r1m_smax': 100,
                      'atr_on': False, 'am': 0, 'ap': 0, 'sl': 50} for r in both_p1[:5]]
    else:
        final_top = [{**r, 'rl': 10, 'rb': 55, 'rs': 45, 'cd': 15,
                      'r1m_len': 0, 'r1m_lmin': 0, 'r1m_smax': 100,
                      'atr_on': False, 'am': 0, 'ap': 0, 'sl': 50} for r in valid_p1[:5]]

    months = [
        ("Aug", pd.Timestamp("2025-08-17").date(), pd.Timestamp("2025-09-01").date()),
        ("Sep", pd.Timestamp("2025-09-01").date(), pd.Timestamp("2025-10-01").date()),
        ("Oct", pd.Timestamp("2025-10-01").date(), pd.Timestamp("2025-11-01").date()),
        ("Nov", pd.Timestamp("2025-11-01").date(), pd.Timestamp("2025-12-01").date()),
        ("Dec", pd.Timestamp("2025-12-01").date(), pd.Timestamp("2026-01-01").date()),
        ("Jan", pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-02-01").date()),
        ("Feb", pd.Timestamp("2026-02-01").date(), pd.Timestamp("2026-02-14").date()),
    ]

    for ci, c in enumerate(final_top):
        sm_key = (c['ip'], c['fp'], c['np'], c['el'])
        sm = sm_cache.get(sm_key)
        if sm is None:
            sm = compute_smart_money(closes, volumes, *sm_key)
        rc, rp = rsi_5m_cache[c['rl']]
        r1m = rsi_1m_cache.get(c['r1m_len']) if c.get('r1m_len', 0) > 0 else None

        trades = run_1min(
            opens, highs, lows, closes, sm, times, rc, rp,
            rsi_buy=c['rb'], rsi_sell=c['rs'], cooldown=c['cd'],
            max_loss_pts=c.get('sl', 50),
            atr_trail_exit=c.get('atr_on', False),
            atr_trail_mult=c.get('am', 2.0), atr_period=c.get('ap', 14),
            rsi_1m=r1m, rsi_1m_long_min=c.get('r1m_lmin', 0),
            rsi_1m_short_max=c.get('r1m_smax', 100),
        )

        sm_s = f"SM({c['ip']}/{c['fp']}/{c['np']}/{c['el']})"
        rsi5_s = f"RSI5({c['rl']}/{c['rb']}/{c['rs']})"
        rsi1_s = f"RSI1({c.get('r1m_len',0)}/{c.get('r1m_lmin',0)}/{c.get('r1m_smax',100)})" if c.get('r1m_len', 0) > 0 else ""
        atr_s = f"ATR({c.get('am',0):.1f}x{c.get('ap',0)})" if c.get('atr_on') else ""
        sl_s = f"SL{c.get('sl',50)}" if c.get('sl', 0) > 0 else ""

        print(f"\n  #{ci+1}: {sm_s} {rsi5_s} CD={c['cd']} {rsi1_s} {atr_s} {sl_s}")
        print(f"  {'Mon':>3}  {'#':>4} {'WR%':>5} {'PF':>5} {'Net$':>7} {'DD$':>6}")
        print(f"  {'-'*35}")
        for mn, ms, me in months:
            sc_m = score_period(trades, ms, me)
            if sc_m and sc_m['count'] > 0:
                print(f"  {mn:>3}  {sc_m['count']:>4} {sc_m['win_rate']:>5.1f} {sc_m['pf']:>5.3f} "
                      f"{sc_m['net_dollar']:>+7.0f} {sc_m['max_dd_dollar']:>6.0f}")
            else:
                print(f"  {mn:>3}  {'--':>4}")
        sc_f = score_trades(trades)
        if sc_f:
            print(f"  {'ALL':>3}  {sc_f['count']:>4} {sc_f['win_rate']:>5.1f} {sc_f['pf']:>5.3f} "
                  f"{sc_f['net_dollar']:>+7.0f} {sc_f['max_dd_dollar']:>6.0f}")

    # Baseline monthly
    sm_base = sm_cache.get((20, 12, 400, 255))
    if sm_base is None:
        sm_base = compute_smart_money(closes, volumes, 20, 12, 400, 255)
    trades_base = run_1min(opens, highs, lows, closes, sm_base, times,
                           rsi_curr_10, rsi_prev_10)
    print(f"\n  BASELINE: SM(20/12/400/255) RSI5(10/55/45) CD=15 SL=50")
    print(f"  {'Mon':>3}  {'#':>4} {'WR%':>5} {'PF':>5} {'Net$':>7} {'DD$':>6}")
    print(f"  {'-'*35}")
    for mn, ms, me in months:
        sc_m = score_period(trades_base, ms, me)
        if sc_m and sc_m['count'] > 0:
            print(f"  {mn:>3}  {sc_m['count']:>4} {sc_m['win_rate']:>5.1f} {sc_m['pf']:>5.3f} "
                  f"{sc_m['net_dollar']:>+7.0f} {sc_m['max_dd_dollar']:>6.0f}")
    sc_bf = score_trades(trades_base)
    if sc_bf:
        print(f"  {'ALL':>3}  {sc_bf['count']:>4} {sc_bf['win_rate']:>5.1f} {sc_bf['pf']:>5.3f} "
              f"{sc_bf['net_dollar']:>+7.0f} {sc_bf['max_dd_dollar']:>6.0f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - t0
    print("\n" + "=" * 90)
    print(f"COMPLETE -- {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 90)

    print(f"  Phase 1 SM only:   {len(both_p1)} profitable both periods")
    print(f"  Phase 2 +RSI:      {len(valid_p2)} profitable both periods")
    print(f"  Phase 3 +ATR/SL:   {len(valid_p3)} profitable both periods")

    best = valid_p3[0] if valid_p3 else (valid_p2[0] if valid_p2 else (both_p1[0] if both_p1 else None))
    if best:
        f = best.get('full')
        atr_i = f"ATR {best.get('am',0):.1f}x{best.get('ap',0)}" if best.get('atr_on') else "no ATR"
        rsi1_i = f"RSI1m({best.get('r1m_len',0)}/{best.get('r1m_lmin',0)}/{best.get('r1m_smax',100)})" if best.get('r1m_len',0) > 0 else "no RSI1m"
        print(f"\n  BEST: SM({best['ip']}/{best['fp']}/{best['np']}/{best['el']}) "
              f"RSI5m({best.get('rl',10)}/{best.get('rb',55)}/{best.get('rs',45)}) "
              f"CD={best.get('cd',15)} {rsi1_i} {atr_i} SL={best.get('sl',50)}")
        if f:
            print(f"    {f['count']} trades, PF {f['pf']}, ${f['net_dollar']:+.0f}, DD ${f['max_dd_dollar']:.0f}")


if __name__ == "__main__":
    main()
