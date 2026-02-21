"""
OOS Step 3: Exit Model Comparison
====================================

Tests 8 exit models using v11.1 entry params (SM threshold=0.15, RSI 8/60/40, CD=20).
Uses configurable SL (set BEST_SL below based on Step 2 results).

Exit models:
  1. SM flip (current v11) — run_backtest_v10 with max_loss_pts
  2. TP=5 (current v15) — run_backtest_tp_exit with sm_threshold=0.15
  3. Tighter SL + SM flip — SM flip with BEST_SL from Step 2
  4. Time-based — cut after N bars if still losing (N=10,15,20,25,30)
  5. Faster SM exit — enter on SM(EMA=100), exit on SM(EMA=50/30/75)
  6. Faster SM + tighter SL — combine model 5 + BEST_SL
  7. Trailing stop — trail by 3pts from peak
  8. Breakeven stop — move SL to entry after +3pts profit

Dual-SM implementation: uses sm_exit parameter added to run_backtest_v10.

Whipsaw measurement for fast SM: what % of fast-SM-exited trades would have
been profitable if held to the slow SM flip?

Output: Table of 8 models x IS/OOS, per-model monthly OOS breakdown.

Usage:
    python3 oos_step3_exit_models.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, resample_to_5min,
    map_5min_rsi_to_1min, run_backtest_v10, score_trades,
    fmt_score, compute_et_minutes,
    NY_OPEN_ET, NY_CLOSE_ET,
)

# Import v15 TP-exit backtest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "live_trading"))
from generate_session import run_backtest_tp_exit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    'MNQ_OOS': 'databento_MNQ_1min_2025-02-17_to_2025-08-17.csv',
    'MNQ_IS':  'databento_MNQ_1min_2025-08-17_to_2026-02-13.csv',
}

MNQ_SM = dict(index_period=10, flow_period=12, norm_period=200, ema_len=100)
MNQ_COMM, MNQ_DPP = 0.52, 2.0

# ---- CONFIGURED FROM STEP 2 RESULTS ----
# SL=35 had best OOS PF (0.648), SL=15 had best OOS Net$ (-$2,461)
BEST_SL = 35   # Best PF on OOS
PROD_SL = 50   # Current production SL


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_databento(filepath, sm_params):
    df = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df['open'], errors='coerce')
    result['High'] = pd.to_numeric(df['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    sm = compute_smart_money(result['Close'].values, result['Volume'].values, **sm_params)
    result['SM_Net'] = sm
    return result


def prepare_arrays(df_1m, rsi_len):
    df_5m = resample_to_5min(df_1m)
    rsi_5m_curr, rsi_5m_prev = map_5min_rsi_to_1min(
        df_1m.index.values, df_5m.index.values, df_5m['Close'].values, rsi_len
    )
    rsi_dummy = np.full(len(df_1m), 50.0)
    return {
        'opens': df_1m['Open'].values, 'highs': df_1m['High'].values,
        'lows': df_1m['Low'].values, 'closes': df_1m['Close'].values,
        'sm': df_1m['SM_Net'].values, 'times': df_1m.index.values,
        'rsi': rsi_dummy, 'rsi_5m_curr': rsi_5m_curr, 'rsi_5m_prev': rsi_5m_prev,
    }


def compute_sm_with_ema(df_1m, ema_len):
    """Compute SM with a specific EMA length (all other params = MNQ default)."""
    return compute_smart_money(
        df_1m['Close'].values, df_1m['Volume'].values,
        index_period=MNQ_SM['index_period'],
        flow_period=MNQ_SM['flow_period'],
        norm_period=MNQ_SM['norm_period'],
        ema_len=ema_len,
    )


# ---------------------------------------------------------------------------
# Model runner — dispatches to the right backtest function
# ---------------------------------------------------------------------------

def run_model(arr, model_type, sm_exit_array=None, **kwargs):
    """Run a specific exit model on given arrays.

    model_type: 'sm_flip', 'tp', 'time', 'fast_sm', 'fast_sm_sl', 'trail', 'breakeven'
    """
    base_kwargs = dict(
        rsi_buy=60, rsi_sell=40, sm_threshold=0.15, cooldown_bars=20,
        rsi_5m_curr=arr['rsi_5m_curr'], rsi_5m_prev=arr['rsi_5m_prev'],
    )

    if model_type == 'sm_flip':
        return run_backtest_v10(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['rsi'], arr['times'],
            max_loss_pts=kwargs.get('sl', PROD_SL),
            **base_kwargs,
        )

    elif model_type == 'tp':
        return run_backtest_tp_exit(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['times'],
            arr['rsi_5m_curr'], arr['rsi_5m_prev'],
            rsi_buy=60, rsi_sell=40, sm_threshold=0.15, cooldown_bars=20,
            max_loss_pts=PROD_SL, tp_pts=kwargs.get('tp_pts', 5),
        )

    elif model_type == 'time':
        return run_backtest_v10(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['rsi'], arr['times'],
            max_loss_pts=PROD_SL,
            underwater_exit_bars=kwargs['max_bars'],
            **base_kwargs,
        )

    elif model_type == 'fast_sm':
        return run_backtest_v10(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['rsi'], arr['times'],
            max_loss_pts=PROD_SL,
            sm_exit=sm_exit_array,
            **base_kwargs,
        )

    elif model_type == 'fast_sm_sl':
        return run_backtest_v10(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['rsi'], arr['times'],
            max_loss_pts=kwargs.get('sl', BEST_SL),
            sm_exit=sm_exit_array,
            **base_kwargs,
        )

    elif model_type == 'trail':
        return run_backtest_v10(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['rsi'], arr['times'],
            max_loss_pts=PROD_SL,
            trailing_stop_pts=kwargs.get('trail_pts', 3),
            **base_kwargs,
        )

    elif model_type == 'breakeven':
        return run_backtest_v10(
            arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            arr['sm'], arr['rsi'], arr['times'],
            max_loss_pts=PROD_SL,
            breakeven_pts=kwargs.get('be_pts', 3),
            **base_kwargs,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Whipsaw measurement
# ---------------------------------------------------------------------------

def measure_whipsaw(fast_trades, slow_trades):
    """For fast-SM-exited trades, check if they would have been profitable on slow SM.

    Matches trades by entry_idx. For each trade that exited via SM_FLIP on fast SM,
    finds the corresponding slow SM trade and checks if it was profitable.
    Returns (total_fast_exits, matched, would_be_profitable, would_be_more_profitable).
    """
    slow_lookup = {t['entry_idx']: t for t in slow_trades}
    comm_pts = (MNQ_COMM * 2) / MNQ_DPP
    fast_sm_exits = [t for t in fast_trades if t['result'] == 'SM_FLIP']
    would_profit = 0
    would_more = 0
    matched = 0

    for ft in fast_sm_exits:
        st = slow_lookup.get(ft['entry_idx'])
        if st is None:
            continue
        matched += 1
        fast_net = ft['pts'] - comm_pts
        slow_net = st['pts'] - comm_pts
        if slow_net > 0:
            would_profit += 1
        if slow_net > fast_net:
            would_more += 1

    return len(fast_sm_exits), matched, would_profit, would_more


# ---------------------------------------------------------------------------
# Monthly breakdown
# ---------------------------------------------------------------------------

def monthly_breakdown(trades):
    """Return per-month breakdown rows."""
    monthly = defaultdict(list)
    for t in trades:
        m = pd.Timestamp(t['exit_time']).strftime('%Y-%m')
        monthly[m].append(t)

    rows = []
    for m in sorted(monthly.keys()):
        sc = score_trades(monthly[m], commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
        if sc:
            rows.append((m, sc['count'], sc['win_rate'], sc['pf'], sc['net_dollar']))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("OOS STEP 3: EXIT MODEL COMPARISON")
    print(f"v11.1 entries — SM(10/12/200/100) RSI(8/60/40) CD=20 threshold=0.15")
    print(f"Production SL={PROD_SL}, Best SL={BEST_SL} (from Step 2)")
    print("=" * 80)

    # ---- Load data ----
    print("\nLoading data...")
    mnq_oos = load_databento(DATA_DIR / FILES['MNQ_OOS'], MNQ_SM)
    mnq_is = load_databento(DATA_DIR / FILES['MNQ_IS'], MNQ_SM)

    print("Preparing arrays...")
    mnq_oos_arr = prepare_arrays(mnq_oos, rsi_len=8)
    mnq_is_arr = prepare_arrays(mnq_is, rsi_len=8)

    # Pre-compute fast SM arrays for Models 5 & 6
    fast_sm_emas = [30, 50, 75]
    print(f"Pre-computing fast SM arrays (EMA={fast_sm_emas})...")
    sm_exit_arrays = {}
    for ema in fast_sm_emas:
        sm_exit_arrays[('OOS', ema)] = compute_sm_with_ema(mnq_oos, ema)
        sm_exit_arrays[('IS', ema)] = compute_sm_with_ema(mnq_is, ema)
    print("  Done.")

    # IS sanity check
    trades_check = run_model(mnq_is_arr, 'sm_flip', sl=PROD_SL)
    sc_check = score_trades(trades_check, commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
    print(f"\n  IS sanity check (Model 1, SL={PROD_SL}): {sc_check['count']} trades, "
          f"PF {sc_check['pf']}, WR {sc_check['win_rate']}%, Net ${sc_check['net_dollar']:+.2f}")

    # ================================================================
    # Define all model configurations
    # ================================================================
    # Each: (display_name, model_type, extra_kwargs, needs_fast_sm_ema)
    model_configs = [
        ("1. SM flip (SL=50)", 'sm_flip', {'sl': PROD_SL}, None),
        ("2. TP=5 (thresh=0.15)", 'tp', {'tp_pts': 5}, None),
        (f"3. SM flip (SL={BEST_SL})", 'sm_flip', {'sl': BEST_SL}, None),
    ]
    # 4. Time-based variants
    for nb in [10, 15, 20, 25, 30]:
        model_configs.append((f"4. Time N={nb}", 'time', {'max_bars': nb}, None))
    # 5. Fast SM variants
    for ema in fast_sm_emas:
        model_configs.append((f"5. Fast SM (exit={ema})", 'fast_sm', {}, ema))
    # 6. Fast SM + tighter SL
    for ema in fast_sm_emas:
        model_configs.append((f"6. FastSM={ema}+SL={BEST_SL}", 'fast_sm_sl', {'sl': BEST_SL}, ema))
    # 7 & 8
    model_configs.append(("7. Trail 3pts", 'trail', {'trail_pts': 3}, None))
    model_configs.append(("8. Breakeven 3pts", 'breakeven', {'be_pts': 3}, None))

    # ================================================================
    # RUN ALL MODELS
    # ================================================================
    print("\n" + "=" * 80)
    print("EXIT MODEL COMPARISON — IS vs OOS")
    print("=" * 80)

    print(f"\n  {'Model':<30} | {'--- IS ---':^36} | {'--- OOS ---':^36}")
    print(f"  {'':<30} | {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10} "
          f"| {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10}")
    print("  " + "-" * 100)

    all_results = {}

    for name, mtype, kwargs, fast_ema in model_configs:
        period_trades = {}
        for period, arr in [("IS", mnq_is_arr), ("OOS", mnq_oos_arr)]:
            sm_exit_arr = sm_exit_arrays.get((period, fast_ema)) if fast_ema else None
            trades = run_model(arr, mtype, sm_exit_array=sm_exit_arr, **kwargs)
            period_trades[period] = trades

        sc_is = score_trades(period_trades['IS'], commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)
        sc_oos = score_trades(period_trades['OOS'], commission_per_side=MNQ_COMM, dollar_per_pt=MNQ_DPP)

        all_results[name] = {
            'is_trades': period_trades['IS'], 'oos_trades': period_trades['OOS'],
            'sc_is': sc_is, 'sc_oos': sc_oos,
        }

        is_str = (f"{sc_is['count']:>6d} {sc_is['win_rate']:>5.1f}% {sc_is['pf']:>6.3f} "
                  f"{sc_is['net_dollar']:>+10.2f}") if sc_is else f"{'NO TRADES':>30}"
        oos_str = (f"{sc_oos['count']:>6d} {sc_oos['win_rate']:>5.1f}% {sc_oos['pf']:>6.3f} "
                   f"{sc_oos['net_dollar']:>+10.2f}") if sc_oos else f"{'NO TRADES':>30}"

        print(f"  {name:<30} | {is_str} | {oos_str}")

    # ================================================================
    # EXIT TYPE BREAKDOWN (OOS)
    # ================================================================
    print("\n" + "=" * 80)
    print("EXIT TYPE BREAKDOWN (OOS)")
    print("=" * 80)

    for name in all_results:
        r = all_results[name]
        if r['sc_oos']:
            exits = r['sc_oos'].get('exits', {})
            exit_str = ' '.join(f"{k}:{v}" for k, v in sorted(exits.items()))
            print(f"  {name:<30}: {exit_str}")

    # ================================================================
    # WHIPSAW ANALYSIS FOR FAST SM
    # ================================================================
    print("\n" + "=" * 80)
    print("WHIPSAW ANALYSIS — Fast SM Exit vs Slow SM Flip")
    print("How many fast-SM exits would have been profitable on slow SM?")
    print("=" * 80)

    slow_is_trades = all_results["1. SM flip (SL=50)"]['is_trades']
    slow_oos_trades = all_results["1. SM flip (SL=50)"]['oos_trades']

    for ema in fast_sm_emas:
        model_name = f"5. Fast SM (exit={ema})"
        if model_name not in all_results:
            continue
        for period, fast_trades, slow_trades in [
            ("IS ", all_results[model_name]['is_trades'], slow_is_trades),
            ("OOS", all_results[model_name]['oos_trades'], slow_oos_trades),
        ]:
            total_fast, matched, would_profit, would_more = measure_whipsaw(fast_trades, slow_trades)
            if matched > 0:
                print(f"  EMA={ema} {period}: {total_fast} fast SM exits, {matched} matched, "
                      f"{would_profit} ({would_profit/matched*100:.0f}%) would be profitable on slow, "
                      f"{would_more} ({would_more/matched*100:.0f}%) would be MORE profitable")
            else:
                print(f"  EMA={ema} {period}: {total_fast} fast SM exits, 0 matched to slow trades")

    # ================================================================
    # MONTHLY OOS BREAKDOWN (SELECTED MODELS)
    # ================================================================
    print("\n" + "=" * 80)
    print("MONTHLY OOS BREAKDOWN — Selected Models")
    print("=" * 80)

    interesting = [
        "1. SM flip (SL=50)",
        "2. TP=5 (thresh=0.15)",
        f"3. SM flip (SL={BEST_SL})",
        "7. Trail 3pts",
        "8. Breakeven 3pts",
    ]
    for ema in fast_sm_emas:
        interesting.append(f"5. Fast SM (exit={ema})")

    for name in interesting:
        if name not in all_results:
            continue
        rows = monthly_breakdown(all_results[name]['oos_trades'])
        if rows:
            print(f"\n  {name}:")
            print(f"    {'Month':<10} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Net $':>10}")
            for m, cnt, wr, pf, net in rows:
                print(f"    {m:<10} {cnt:>6d} {wr:>5.1f}% {pf:>6.3f} {net:>+10.2f}")

    # ================================================================
    # RANKING
    # ================================================================
    print("\n" + "=" * 80)
    print("MODEL RANKING — By OOS Net $")
    print("=" * 80)

    ranked = []
    for name, r in all_results.items():
        if r['sc_oos'] and r['sc_is']:
            ranked.append((
                name,
                r['sc_oos']['net_dollar'], r['sc_oos']['pf'], r['sc_oos']['win_rate'],
                r['sc_is']['net_dollar'], r['sc_is']['pf'],
            ))
    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'#':>3} {'Model':<30} {'OOS Net$':>10} {'OOS PF':>7} {'OOS WR':>7} "
          f"{'IS Net$':>10} {'IS PF':>7}")
    print("  " + "-" * 80)
    for i, (name, oos_net, oos_pf, oos_wr, is_net, is_pf) in enumerate(ranked, 1):
        print(f"  {i:>3} {name:<30} {oos_net:>+10.2f} {oos_pf:>7.3f} {oos_wr:>6.1f}% "
              f"{is_net:>+10.2f} {is_pf:>7.3f}")

    if ranked:
        best = ranked[0]
        print(f"\n  BEST EXIT MODEL: {best[0]}")
        print(f"    OOS: Net ${best[1]:+.2f}, PF {best[2]:.3f}, WR {best[3]:.1f}%")
        print(f"    IS:  Net ${best[4]:+.2f}, PF {best[5]:.3f}")

    print("\n" + "=" * 80)
    print("DONE — Step 3 complete. Best exit model feeds into Step 4 (param sweep).")
    print("=" * 80)


if __name__ == "__main__":
    main()
