"""
MES Parameter Sweep with Train/Test Split
==========================================
Finds optimal SM + RSI params for MES 1-min, then validates on held-out data.

Architecture: 1-MIN bars. SM on 1-min. RSI on 5-min mapped back to 1-min.

Data: Databento MES 1-min, 172,496 bars, Aug 17 2025 - Feb 12 2026
  TRAIN: Aug 17 - Nov 30 (sweep on this)
  TEST:  Dec 1 - Feb 12 (validate winners on this)

SM is computed on FULL data (needs warmup), then sliced for scoring.

Phase 1: SM param sweep (150 combos) with fixed RSI 10/55/45
Phase 2: RSI sweep + 1-min RSI filter on top SM sets (5,600 combos)
Phase 3: Top combos + ATR trailing stop (180 combos)
Phase 4: Validate top 20 on TEST data
Phase 5: Monthly breakdown of best configs on FULL data
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
    resample_to_5min,
    compute_smart_money, compute_rsi, map_5min_rsi_to_1min,
    run_backtest_v10, score_trades,
)

# MES instrument config
COMMISSION = 0.52
DOLLAR_PER_PT = 5.0  # MES = $5/point


def load_mes_databento():
    """Load MES 1-min Databento data."""
    filepath = (Path(__file__).resolve().parent.parent
                / "data" / "databento_MES_1min_2025-08-17_to_2026-02-13.csv")
    df = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    return result


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
    """Score only trades within a date range. Uses MES dollar/pt."""
    filtered = [t for t in trades
                if start_date <= pd.Timestamp(t['entry_time']).date() < end_date]
    return score_trades(filtered, commission_per_side=COMMISSION,
                        dollar_per_pt=DOLLAR_PER_PT)


def main():
    t0 = time.time()
    print("=" * 100)
    print("MES PARAMETER SWEEP WITH TRAIN/TEST SPLIT")
    print("Data: Databento MES 1-min, Aug 17 2025 - Feb 12 2026")
    print(f"Commission: ${COMMISSION:.2f}/side, ${DOLLAR_PER_PT:.2f}/point")
    print("TRAIN: Aug 17 - Nov 30  |  TEST: Dec 1 - Feb 12")
    print("=" * 100)

    # Load full data
    print("\nLoading MES 1-min Databento data...")
    df_1m = load_mes_databento()
    print(f"  {len(df_1m)} bars, {df_1m.index[0].date()} to {df_1m.index[-1].date()}")

    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    volumes = df_1m['Volume'].values
    times = df_1m.index.values

    # 5-min for RSI
    df_for_5m = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_for_5m['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_for_5m)
    fivemin_times = df_5m.index.values
    fivemin_closes = df_5m['Close'].values
    print(f"  5-min: {len(df_5m)} bars (for RSI computation)")

    # Pre-compute default RSI mapping (RSI 10) for Phase 1
    rsi_curr_10, rsi_prev_10 = map_5min_rsi_to_1min(
        times, fivemin_times, fivemin_closes, rsi_len=10)

    # Pre-compute 1-min RSI cache
    print("  Pre-computing 1-min RSI arrays...")
    rsi_1m_cache = {}
    for rl in [6, 8, 10, 12, 14]:
        rsi_1m_cache[rl] = compute_rsi(closes, rl)

    # Date ranges
    TRAIN_START = pd.Timestamp("2025-08-17").date()
    TRAIN_END = pd.Timestamp("2025-12-01").date()
    TEST_START = pd.Timestamp("2025-12-01").date()
    TEST_END = pd.Timestamp("2026-02-13").date()
    FULL_START = TRAIN_START
    FULL_END = TEST_END

    # Count bars per split
    train_mask = (df_1m.index >= pd.Timestamp(TRAIN_START)) & (df_1m.index < pd.Timestamp(TRAIN_END))
    test_mask = (df_1m.index >= pd.Timestamp(TEST_START)) & (df_1m.index < pd.Timestamp(TEST_END))
    print(f"  TRAIN bars: {train_mask.sum():,}  TEST bars: {test_mask.sum():,}")

    # =========================================================================
    # PHASE 1: SM Parameter Sweep (TRAIN only)
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 1: SM PARAMETER SWEEP on TRAIN (fixed RSI 10/55/45, CD=15, SL=50)")
    print("=" * 100)

    sm_idx = [10, 15, 20, 25, 30]
    sm_flow = [8, 12, 14]       # reduced: skip 10 (close to 12)
    sm_norm = [200, 400]         # reduced: 200=300=400=500 on MNQ
    sm_ema = [100, 150, 200, 255, 350]

    total_sm = len(sm_idx) * len(sm_flow) * len(sm_norm) * len(sm_ema)
    print(f"  Grid: {len(sm_idx)}x{len(sm_flow)}x{len(sm_norm)}x{len(sm_ema)} = {total_sm} combos")

    results_p1 = []
    sm_cache = {}
    done = 0
    t1 = time.time()

    for ip, fp, np_, el in product(sm_idx, sm_flow, sm_norm, sm_ema):
        # Compute SM on FULL data (warmup), score on TRAIN only
        sm = compute_smart_money(closes, volumes, ip, fp, np_, el)
        sm_cache[(ip, fp, np_, el)] = sm

        trades = run_1min(opens, highs, lows, closes, sm, times,
                          rsi_curr_10, rsi_prev_10)
        sc_train = score_period(trades, TRAIN_START, TRAIN_END)
        sc_test = score_period(trades, TEST_START, TEST_END)
        sc_full = score_period(trades, FULL_START, FULL_END)

        results_p1.append({
            'ip': ip, 'fp': fp, 'np': np_, 'el': el,
            'train': sc_train, 'test': sc_test, 'full': sc_full,
        })

        done += 1
        if done % 30 == 0:
            elapsed = time.time() - t1
            rate = done / elapsed
            remain = (total_sm - done) / rate
            print(f"  ... {done}/{total_sm} ({elapsed:.0f}s, ~{remain:.0f}s left)")

    elapsed_p1 = time.time() - t1
    print(f"\n  Phase 1: {total_sm} combos in {elapsed_p1:.1f}s ({total_sm/elapsed_p1:.1f}/s)")

    # Filter: profitable on train
    valid_p1 = [r for r in results_p1
                if r['train'] is not None and r['train']['count'] >= 30
                and r['train']['pf'] > 1.0]
    valid_p1.sort(key=lambda r: r['train']['pf'], reverse=True)

    print(f"  Profitable on TRAIN (>=30 trades): {len(valid_p1)}/{total_sm}")

    # Show top 20
    print(f"\n  TOP 20 by TRAIN PF:")
    print(f"  {'Rk':>3}  {'I':>2} {'F':>2} {'N':>3} {'E':>3}  |  "
          f"{'TR#':>4} {'TRWR':>5} {'TRPF':>5} {'TR$':>8} {'TRDD':>7}  |  "
          f"{'TS#':>4} {'TSPF':>5} {'TS$':>8}  |  "
          f"{'F#':>3} {'FPF':>5} {'F$':>8}")
    print("  " + "-" * 110)
    for rk, r in enumerate(valid_p1[:20], 1):
        tr = r['train']
        ts = r['test']
        f = r['full']
        ts_n = ts['count'] if ts else 0
        ts_pf = ts['pf'] if ts else 0
        ts_d = ts['net_dollar'] if ts else 0
        f_n = f['count'] if f else 0
        f_pf = f['pf'] if f else 0
        f_d = f['net_dollar'] if f else 0
        print(f"  {rk:>3}  {r['ip']:>2} {r['fp']:>2} {r['np']:>3} {r['el']:>3}  |  "
              f"{tr['count']:>4} {tr['win_rate']:>5.1f} {tr['pf']:>5.3f} "
              f"${tr['net_dollar']:>+7.0f} ${tr['max_dd_dollar']:>6.0f}  |  "
              f"{ts_n:>4} {ts_pf:>5.3f} ${ts_d:>+7.0f}  |  "
              f"{f_n:>3} {f_pf:>5.3f} ${f_d:>+7.0f}")

    # Baseline
    base_r = next((r for r in results_p1
                   if r['ip'] == 20 and r['fp'] == 12 and r['np'] == 400 and r['el'] == 255), None)
    if base_r and base_r['train']:
        print(f"\n  BASELINE SM(20/12/400/255):")
        for lbl, sc in [("TRAIN", base_r['train']), ("TEST", base_r['test']), ("FULL", base_r['full'])]:
            if sc:
                print(f"    {lbl:>5}: {sc['count']}tr WR {sc['win_rate']:.1f}% PF {sc['pf']} "
                      f"${sc['net_dollar']:+.0f} DD ${sc['max_dd_dollar']:.0f}")

    # =========================================================================
    # PHASE 2: RSI Sweep on top SM sets (TRAIN only)
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 2: RSI SWEEP + 1-MIN RSI FILTER on TRAIN")
    print("=" * 100)

    # Top SM sets from Phase 1
    if len(valid_p1) >= 10:
        top_sm = valid_p1[:10]
    else:
        top_sm = valid_p1[:]
        extra = [r for r in results_p1
                 if r['train'] is not None and r not in top_sm]
        extra.sort(key=lambda r: r['train']['pf'] if r['train'] else 0, reverse=True)
        top_sm += extra[:10 - len(top_sm)]

    # Pre-compute 5-min RSI mappings
    rsi_5m_cache = {}
    for rl in [6, 8, 10, 12, 14]:
        rsi_5m_cache[rl] = map_5min_rsi_to_1min(
            times, fivemin_times, fivemin_closes, rsi_len=rl)

    rsi_lens = [6, 8, 10, 12, 14]
    rsi_levels = [(50, 50), (55, 45), (60, 40), (65, 35)]
    cooldowns = [10, 15, 20, 30]
    rsi_1m_configs = [
        (0, 0, 100),       # disabled
        (10, 50, 50),      # RSI(10) > 50 for longs, < 50 for shorts
        (10, 45, 55),      # looser
        (10, 55, 45),      # tighter
        (14, 50, 50),      # RSI(14) > 50 / < 50
        (14, 45, 55),      # RSI(14) looser
        (6, 50, 50),       # RSI(6) faster
    ]

    total_p2 = len(top_sm) * len(rsi_lens) * len(rsi_levels) * len(cooldowns) * len(rsi_1m_configs)
    print(f"  {len(top_sm)} SM x {len(rsi_lens)} RSI5m x {len(rsi_levels)} lvls x "
          f"{len(cooldowns)} cd x {len(rsi_1m_configs)} rsi1m = {total_p2}")

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
                            rsi_1m=r1m, rsi_1m_long_min=r1m_lmin,
                            rsi_1m_short_max=r1m_smax,
                        )
                        sc_train = score_period(trades, TRAIN_START, TRAIN_END)

                        results_p2.append({
                            'ip': sm_r['ip'], 'fp': sm_r['fp'],
                            'np': sm_r['np'], 'el': sm_r['el'],
                            'rl': rl, 'rb': rb, 'rs': rs, 'cd': cd,
                            'r1m_len': r1m_len, 'r1m_lmin': r1m_lmin,
                            'r1m_smax': r1m_smax,
                            'train': sc_train,
                        })
                        done += 1

                        if done % 500 == 0:
                            elapsed = time.time() - t2
                            rate = done / elapsed if elapsed > 0 else 1
                            remain = (total_p2 - done) / rate
                            print(f"  ... {done}/{total_p2} ({elapsed:.0f}s, ~{remain:.0f}s left)")

    elapsed_p2 = time.time() - t2
    print(f"  Phase 2: {total_p2} combos in {elapsed_p2:.1f}s ({total_p2/max(1,elapsed_p2):.1f}/s)")

    # Filter: profitable on train with decent trade count
    valid_p2 = [r for r in results_p2
                if r['train'] is not None and r['train']['count'] >= 20
                and r['train']['pf'] > 1.0]
    valid_p2.sort(key=lambda r: r['train']['pf'], reverse=True)

    print(f"  Profitable on TRAIN (>=20 trades): {len(valid_p2)}/{total_p2}")

    if valid_p2:
        print(f"\n  TOP 30 by TRAIN PF:")
        print(f"  {'Rk':>3}  {'SM':>11} {'RSI5m':>9} {'CD':>2} {'RSI1m':>10}  |  "
              f"{'TR#':>4} {'TRWR':>5} {'TRPF':>6} {'TR$':>8} {'TRDD':>7}")
        print("  " + "-" * 85)
        for rk, r in enumerate(valid_p2[:30], 1):
            tr = r['train']
            sm_s = f"{r['ip']}/{r['fp']}/{r['np']}/{r['el']}"
            rsi5_s = f"{r['rl']}/{r['rb']}/{r['rs']}"
            rsi1_s = f"{r['r1m_len']}/{r['r1m_lmin']}/{r['r1m_smax']}" if r['r1m_len'] > 0 else "OFF"
            print(f"  {rk:>3}  {sm_s:>11} {rsi5_s:>9} {r['cd']:>2} {rsi1_s:>10}  |  "
                  f"{tr['count']:>4} {tr['win_rate']:>5.1f} {tr['pf']:>6.3f} "
                  f"${tr['net_dollar']:>+7.0f} ${tr['max_dd_dollar']:>6.0f}")

    # =========================================================================
    # PHASE 3: Top Combos + ATR Trail (TRAIN only)
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 3: TOP COMBOS + ATR/SL SWEEP on TRAIN")
    print("=" * 100)

    if len(valid_p2) >= 10:
        top_c = valid_p2[:10]
    elif valid_p2:
        top_c = valid_p2[:]
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
    sl_options = [0, 15, 25, 50]  # MES-specific: 15/25 scaled stops

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
                    rsi_1m=r1m, rsi_1m_long_min=c['r1m_lmin'],
                    rsi_1m_short_max=c['r1m_smax'],
                )
                sc_train = score_period(trades, TRAIN_START, TRAIN_END)

                results_p3.append({
                    **{k: c[k] for k in ['ip', 'fp', 'np', 'el', 'rl', 'rb', 'rs', 'cd',
                                          'r1m_len', 'r1m_lmin', 'r1m_smax']},
                    'atr_on': atr_on, 'am': am, 'ap': ap, 'sl': sl,
                    'train': sc_train,
                })

    elapsed_p3 = time.time() - t3
    print(f"  Phase 3: {total_p3} combos in {elapsed_p3:.1f}s")

    valid_p3 = [r for r in results_p3
                if r['train'] is not None and r['train']['count'] >= 20
                and r['train']['pf'] > 1.0]
    valid_p3.sort(key=lambda r: r['train']['pf'], reverse=True)
    print(f"  Profitable on TRAIN: {len(valid_p3)}/{total_p3}")

    if valid_p3:
        print(f"\n  TOP 20 FULL CONFIGS by TRAIN PF:")
        print(f"  {'Rk':>3}  {'SM':>11} {'RSI5m':>9} {'CD':>2} {'RSI1m':>10} {'ATR':>7} {'SL':>2}  |  "
              f"{'TR#':>4} {'TRWR':>5} {'TRPF':>6} {'TR$':>8} {'TRDD':>7}")
        print("  " + "-" * 100)
        for rk, r in enumerate(valid_p3[:20], 1):
            tr = r['train']
            sm_s = f"{r['ip']}/{r['fp']}/{r['np']}/{r['el']}"
            rsi5_s = f"{r['rl']}/{r['rb']}/{r['rs']}"
            rsi1_s = f"{r['r1m_len']}/{r['r1m_lmin']}/{r['r1m_smax']}" if r['r1m_len'] > 0 else "OFF"
            atr_s = f"{r['am']:.1f}x{r['ap']}" if r['atr_on'] else "OFF"
            print(f"  {rk:>3}  {sm_s:>11} {rsi5_s:>9} {r['cd']:>2} {rsi1_s:>10} {atr_s:>7} {r['sl']:>2}  |  "
                  f"{tr['count']:>4} {tr['win_rate']:>5.1f} {tr['pf']:>6.3f} "
                  f"${tr['net_dollar']:>+7.0f} ${tr['max_dd_dollar']:>6.0f}")

    # =========================================================================
    # PHASE 4: VALIDATE TOP CONFIGS ON TEST DATA
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 4: VALIDATE TOP 20 ON TEST (Dec 1 - Feb 12)")
    print("This is the HELD-OUT data -- never used for optimization")
    print("=" * 100)

    # Use top from Phase 3 if available, else Phase 2
    if valid_p3:
        validate_configs = valid_p3[:20]
    elif valid_p2:
        validate_configs = [{**r, 'atr_on': False, 'am': 0, 'ap': 0, 'sl': 50}
                            for r in valid_p2[:20]]
    else:
        validate_configs = [{**r, 'rl': 10, 'rb': 55, 'rs': 45, 'cd': 15,
                             'r1m_len': 0, 'r1m_lmin': 0, 'r1m_smax': 100,
                             'atr_on': False, 'am': 0, 'ap': 0, 'sl': 50}
                            for r in valid_p1[:20]]

    print(f"\n  {'Rk':>3}  {'SM':>11} {'RSI5m':>9} {'CD':>2} {'RSI1m':>10} {'ATR':>7} {'SL':>2}  |  "
          f"{'TRPF':>6} {'TR$':>8}  |  {'TSPF':>6} {'TS$':>8} {'TSDD':>7}  |  "
          f"{'Deg%':>5} {'Verdict':>7}")
    print("  " + "-" * 115)

    validated = []
    for rk, c in enumerate(validate_configs, 1):
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
        sc_train = score_period(trades, TRAIN_START, TRAIN_END)
        sc_test = score_period(trades, TEST_START, TEST_END)
        sc_full = score_period(trades, FULL_START, FULL_END)

        sm_s = f"{c['ip']}/{c['fp']}/{c['np']}/{c['el']}"
        rsi5_s = f"{c['rl']}/{c['rb']}/{c['rs']}"
        rsi1_s = f"{c['r1m_len']}/{c['r1m_lmin']}/{c['r1m_smax']}" if c.get('r1m_len', 0) > 0 else "OFF"
        atr_s = f"{c['am']:.1f}x{c['ap']}" if c.get('atr_on') else "OFF"

        if sc_train and sc_test:
            degrad = (1 - sc_test['pf'] / sc_train['pf']) * 100 if sc_train['pf'] > 0 else 999
            if sc_test['pf'] >= 1.0 and degrad < 30:
                verdict = "PASS"
            elif sc_test['pf'] >= 1.0:
                verdict = "WATCH"
            else:
                verdict = "FAIL"
            print(f"  {rk:>3}  {sm_s:>11} {rsi5_s:>9} {c['cd']:>2} {rsi1_s:>10} {atr_s:>7} {c.get('sl',50):>2}  |  "
                  f"{sc_train['pf']:>6.3f} ${sc_train['net_dollar']:>+7.0f}  |  "
                  f"{sc_test['pf']:>6.3f} ${sc_test['net_dollar']:>+7.0f} ${sc_test['max_dd_dollar']:>6.0f}  |  "
                  f"{degrad:>+5.1f} {verdict:>7}")
            validated.append({**c, 'train': sc_train, 'test': sc_test, 'full': sc_full,
                              'degrad': degrad, 'verdict': verdict})
        elif sc_train:
            print(f"  {rk:>3}  {sm_s:>11} {rsi5_s:>9} {c['cd']:>2} {rsi1_s:>10} {atr_s:>7} {c.get('sl',50):>2}  |  "
                  f"{sc_train['pf']:>6.3f} ${sc_train['net_dollar']:>+7.0f}  |  "
                  f"{'N/A':>6} {'N/A':>8} {'N/A':>7}  |  {'N/A':>5} {'N/A':>7}")

    # Also validate v9.4 baseline and v11 MNQ-optimized for comparison
    print("\n  REFERENCE CONFIGS (not from sweep):")
    ref_configs = [
        {'label': 'v9.4 baseline', 'ip': 20, 'fp': 12, 'np': 400, 'el': 255,
         'rl': 10, 'rb': 55, 'rs': 45, 'cd': 15, 'sl': 50,
         'r1m_len': 0, 'r1m_lmin': 0, 'r1m_smax': 100,
         'atr_on': False, 'am': 0, 'ap': 0},
        {'label': 'v9.4 SL=0', 'ip': 20, 'fp': 12, 'np': 400, 'el': 255,
         'rl': 10, 'rb': 55, 'rs': 45, 'cd': 15, 'sl': 0,
         'r1m_len': 0, 'r1m_lmin': 0, 'r1m_smax': 100,
         'atr_on': False, 'am': 0, 'ap': 0},
        {'label': 'v11 MNQ-opt', 'ip': 10, 'fp': 12, 'np': 200, 'el': 100,
         'rl': 8, 'rb': 60, 'rs': 40, 'cd': 20, 'sl': 50,
         'r1m_len': 0, 'r1m_lmin': 0, 'r1m_smax': 100,
         'atr_on': False, 'am': 0, 'ap': 0},
    ]

    for ref in ref_configs:
        sm_key = (ref['ip'], ref['fp'], ref['np'], ref['el'])
        sm = sm_cache.get(sm_key)
        if sm is None:
            sm = compute_smart_money(closes, volumes, *sm_key)
            sm_cache[sm_key] = sm
        rc, rp = rsi_5m_cache[ref['rl']]

        trades = run_1min(
            opens, highs, lows, closes, sm, times, rc, rp,
            rsi_buy=ref['rb'], rsi_sell=ref['rs'], cooldown=ref['cd'],
            max_loss_pts=ref['sl'])
        sc_train = score_period(trades, TRAIN_START, TRAIN_END)
        sc_test = score_period(trades, TEST_START, TEST_END)

        if sc_train and sc_test:
            degrad = (1 - sc_test['pf'] / sc_train['pf']) * 100 if sc_train['pf'] > 0 else 999
            verdict = "PASS" if sc_test['pf'] >= 1.0 and degrad < 30 else ("WATCH" if sc_test['pf'] >= 1.0 else "FAIL")
            print(f"  REF  {ref['label']:<15}  |  "
                  f"TRPF {sc_train['pf']:>6.3f} ${sc_train['net_dollar']:>+7.0f}  |  "
                  f"TSPF {sc_test['pf']:>6.3f} ${sc_test['net_dollar']:>+7.0f} "
                  f"DD ${sc_test['max_dd_dollar']:>6.0f}  |  "
                  f"Deg {degrad:>+5.1f}% {verdict:>7}")

    # =========================================================================
    # PHASE 5: Monthly Breakdown of Best Configs on FULL Data
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 5: MONTHLY BREAKDOWN (full data)")
    print("=" * 100)

    months = [
        ("Aug", pd.Timestamp("2025-08-17").date(), pd.Timestamp("2025-09-01").date()),
        ("Sep", pd.Timestamp("2025-09-01").date(), pd.Timestamp("2025-10-01").date()),
        ("Oct", pd.Timestamp("2025-10-01").date(), pd.Timestamp("2025-11-01").date()),
        ("Nov", pd.Timestamp("2025-11-01").date(), pd.Timestamp("2025-12-01").date()),
        ("Dec", pd.Timestamp("2025-12-01").date(), pd.Timestamp("2026-01-01").date()),
        ("Jan", pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-02-01").date()),
        ("Feb", pd.Timestamp("2026-02-01").date(), pd.Timestamp("2026-02-14").date()),
    ]

    # Top passing configs + reference
    monthly_configs = []
    passed = [v for v in validated if v['verdict'] == 'PASS']
    if passed:
        monthly_configs.append(passed[0])  # best passing
    # Add v9.4 baseline
    monthly_configs.append(ref_configs[0])  # v9.4 SL=50

    for ci, c in enumerate(monthly_configs):
        sm_key = (c['ip'], c['fp'], c['np'], c['el'])
        sm = sm_cache.get(sm_key)
        if sm is None:
            sm = compute_smart_money(closes, volumes, *sm_key)
        rc, rp = rsi_5m_cache[c['rl']]
        r1m = rsi_1m_cache.get(c.get('r1m_len', 0)) if c.get('r1m_len', 0) > 0 else None

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
        rsi5_s = f"RSI({c['rl']}/{c['rb']}/{c['rs']})"
        label = c.get('label', f"{sm_s} {rsi5_s} CD={c['cd']} SL={c.get('sl', 50)}")

        print(f"\n  #{ci+1}: {label}")
        print(f"  {'Mon':>3}  {'#':>4} {'WR%':>5} {'PF':>6} {'Net$':>9} {'DD$':>7}")
        print(f"  {'-'*40}")
        for mn, ms, me in months:
            sc_m = score_period(trades, ms, me)
            if sc_m and sc_m['count'] > 0:
                print(f"  {mn:>3}  {sc_m['count']:>4} {sc_m['win_rate']:>5.1f} "
                      f"{sc_m['pf']:>6.3f} ${sc_m['net_dollar']:>+8.2f} ${sc_m['max_dd_dollar']:>6.2f}")
        sc_full = score_period(trades, FULL_START, FULL_END)
        if sc_full:
            print(f"  {'ALL':>3}  {sc_full['count']:>4} {sc_full['win_rate']:>5.1f} "
                  f"{sc_full['pf']:>6.3f} ${sc_full['net_dollar']:>+8.2f} ${sc_full['max_dd_dollar']:>6.2f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - t0
    print("\n" + "=" * 100)
    print(f"COMPLETE -- {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 100)

    n_pass = sum(1 for v in validated if v['verdict'] == 'PASS')
    n_watch = sum(1 for v in validated if v['verdict'] == 'WATCH')
    n_fail = sum(1 for v in validated if v['verdict'] == 'FAIL')
    print(f"  Phase 1 (SM sweep):      {len(valid_p1)} profitable on TRAIN")
    print(f"  Phase 2 (+RSI):          {len(valid_p2)} profitable on TRAIN")
    print(f"  Phase 3 (+ATR/SL):       {len(valid_p3)} profitable on TRAIN")
    print(f"  Phase 4 (TEST validation): {n_pass} PASS, {n_watch} WATCH, {n_fail} FAIL")

    if passed:
        best = passed[0]
        sm_s = f"SM({best['ip']}/{best['fp']}/{best['np']}/{best['el']})"
        rsi5_s = f"RSI({best['rl']}/{best['rb']}/{best['rs']})"
        rsi1_s = (f"RSI1m({best['r1m_len']}/{best['r1m_lmin']}/{best['r1m_smax']})"
                  if best.get('r1m_len', 0) > 0 else "")
        atr_s = f"ATR({best.get('am',0):.1f}x{best.get('ap',0)})" if best.get('atr_on') else ""
        sl_s = f"SL={best.get('sl',50)}"

        print(f"\n  BEST VALIDATED: {sm_s} {rsi5_s} CD={best['cd']} {rsi1_s} {atr_s} {sl_s}")
        if best.get('train') and best.get('test'):
            print(f"    TRAIN: {best['train']['count']}tr PF {best['train']['pf']} "
                  f"${best['train']['net_dollar']:+.0f}")
            print(f"    TEST:  {best['test']['count']}tr PF {best['test']['pf']} "
                  f"${best['test']['net_dollar']:+.0f} DD ${best['test']['max_dd_dollar']:.0f}")
            print(f"    Degradation: {best['degrad']:+.1f}%")
    else:
        print("\n  NO configs passed validation on TEST data.")
        print("  The v9.4 baseline may be the best option for MES.")


if __name__ == "__main__":
    main()
