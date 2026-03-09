"""
ATR-Scaled Exits Sweep (Test #1)
=================================
Replace fixed SL/TP with ATR-multiple exits computed at entry time.
Custom backtest loop that mirrors run_backtest_tp_exit entry logic but uses
per-trade ATR-scaled SL and TP (capped at reasonable maximums).

Sweeps:
  - ATR periods: [10, 14, 20] on 1-min bars
  - SL multiples: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
  - TP multiples: [0.5, 1.0, 1.5, 2.0, 3.0]
  - All 3 strategies: vScalpA (V15), vScalpB, MES_V2

Usage:
    python3 atr_scaled_exits_sweep.py
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

# --- Path setup ---
sys.path.insert(0, 'strategies')
sys.path.insert(0, str(Path('strategies').resolve().parent))
sys.path.insert(0, str(Path('.').resolve().parent / 'live_trading'))

from v10_test_common import (
    compute_smart_money,
    compute_rsi,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
    compute_et_minutes,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)

from generate_session import (
    load_instrument_1min,
    run_backtest_tp_exit,
    # MNQ params
    MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
    VSCALPA_RSI_LEN, VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_MAX_LOSS_PTS, VSCALPA_TP_PTS, VSCALPA_ENTRY_END_ET,
    VSCALPB_RSI_LEN, VSCALPB_RSI_BUY, VSCALPB_RSI_SELL,
    VSCALPB_SM_THRESHOLD, VSCALPB_COOLDOWN,
    VSCALPB_MAX_LOSS_PTS, VSCALPB_TP_PTS,
    # MES params
    MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA,
    MES_DOLLAR_PER_PT, MES_COMMISSION,
    MESV2_RSI_LEN, MESV2_RSI_BUY, MESV2_RSI_SELL,
    MESV2_SM_THRESHOLD, MESV2_COOLDOWN,
    MESV2_MAX_LOSS_PTS, MESV2_TP_PTS, MESV2_EOD_ET,
    MESV2_BREAKEVEN_BARS,
)


# ============================================================================
# ATR Computation
# ============================================================================

def compute_atr(highs, lows, closes, period):
    """Compute ATR (Average True Range) on 1-min bars.

    Uses Wilder's smoothing (exponential moving average).
    Returns array of length n with ATR values.
    """
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    atr = np.zeros(n)
    # SMA for first 'period' bars
    if n > period:
        atr[period - 1] = np.mean(tr[:period])
        # Wilder's smoothing
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ============================================================================
# Custom Backtest Loop with ATR-Scaled Exits
# ============================================================================

def run_backtest_atr_exits(opens, highs, lows, closes, sm, times,
                           rsi_5m_curr, rsi_5m_prev,
                           atr,  # precomputed ATR array
                           rsi_buy, rsi_sell, sm_threshold,
                           cooldown_bars,
                           sl_mult, tp_mult,
                           sl_cap, tp_cap,
                           eod_minutes_et=NY_CLOSE_ET,
                           breakeven_after_bars=0,
                           entry_end_et=NY_LAST_ENTRY_ET):
    """Backtest with ATR-scaled SL and TP computed at entry time.

    Entry logic: identical to run_backtest_tp_exit.
    Exit logic: same priority order, but SL and TP are ATR-scaled per trade.

    At entry bar i:
        sl_pts = min(sl_mult * atr[i], sl_cap)
        tp_pts = min(tp_mult * atr[i], tp_cap)

    Exit priority (same as production):
      1. EOD -> fill at bar close
      2. SL: prev bar close breaches dynamic SL -> fill at next open
      3. TP: prev bar close reaches dynamic TP -> fill at next open
      4. BE_TIME: bars held >= breakeven_after_bars -> fill at next open
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False
    trade_sl_pts = 0.0
    trade_tp_pts = 0.0

    et_mins = compute_et_minutes(times)

    def close_trade(side, entry_p, exit_p, entry_i, exit_i, result):
        pts = (exit_p - entry_p) if side == "long" else (entry_p - exit_p)
        trades.append({
            "side": side, "entry": entry_p, "exit": exit_p,
            "pts": pts, "entry_time": times[entry_i], "exit_time": times[exit_i],
            "entry_idx": entry_i, "exit_idx": exit_i,
            "bars": exit_i - entry_i, "result": result,
        })

    for i in range(2, n):
        bar_mins_et = et_mins[i]

        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]

        sm_bull = sm_prev > sm_threshold
        sm_bear = sm_prev < -sm_threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI cross from mapped 5-min
        rsi_prev = rsi_5m_curr[i - 1]
        rsi_prev2 = rsi_5m_prev[i - 1]
        rsi_long_trigger = rsi_prev > rsi_buy and rsi_prev2 <= rsi_buy
        rsi_short_trigger = rsi_prev < rsi_sell and rsi_prev2 >= rsi_sell

        # Episode reset (uses zero crossing, not threshold)
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # EOD close
        if trade_state != 0 and bar_mins_et >= eod_minutes_et:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        # Exits for open LONG positions
        if trade_state == 1:
            # SL: prev bar close breaches dynamic stop
            if trade_sl_pts > 0 and closes[i - 1] <= entry_price - trade_sl_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # TP: prev bar close reached dynamic TP target
            if trade_tp_pts > 0 and closes[i - 1] >= entry_price + trade_tp_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

            # BE_TIME: stale trade exit after N bars
            if breakeven_after_bars > 0:
                bars_in_trade = (i - 1) - entry_idx
                if bars_in_trade >= breakeven_after_bars:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "BE_TIME")
                    trade_state = 0
                    exit_bar = i
                    continue

        # Exits for open SHORT positions
        elif trade_state == -1:
            # SL
            if trade_sl_pts > 0 and closes[i - 1] >= entry_price + trade_sl_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # TP
            if trade_tp_pts > 0 and closes[i - 1] <= entry_price - trade_tp_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

            # BE_TIME
            if breakeven_after_bars > 0:
                bars_in_trade = (i - 1) - entry_idx
                if bars_in_trade >= breakeven_after_bars:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "BE_TIME")
                    trade_state = 0
                    exit_bar = i
                    continue

        # Entries
        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= entry_end_et
            cd_ok = bars_since >= cooldown_bars

            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True
                    # Compute ATR-scaled exits at entry
                    atr_val = atr[i] if atr[i] > 0 else atr[i - 1]
                    trade_sl_pts = min(sl_mult * atr_val, sl_cap)
                    trade_tp_pts = min(tp_mult * atr_val, tp_cap)
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True
                    # Compute ATR-scaled exits at entry
                    atr_val = atr[i] if atr[i] > 0 else atr[i - 1]
                    trade_sl_pts = min(sl_mult * atr_val, sl_cap)
                    trade_tp_pts = min(tp_mult * atr_val, tp_cap)

    return trades


# ============================================================================
# Strategy Configuration
# ============================================================================

STRATEGIES = {
    "vScalpA": {
        "instrument": "MNQ",
        "sm_params": (MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA),
        "rsi_len": VSCALPA_RSI_LEN,
        "rsi_buy": VSCALPA_RSI_BUY,
        "rsi_sell": VSCALPA_RSI_SELL,
        "sm_threshold": VSCALPA_SM_THRESHOLD,
        "cooldown": VSCALPA_COOLDOWN,
        "fixed_sl": VSCALPA_MAX_LOSS_PTS,
        "fixed_tp": VSCALPA_TP_PTS,
        "dollar_per_pt": MNQ_DOLLAR_PER_PT,
        "commission": MNQ_COMMISSION,
        "entry_end_et": VSCALPA_ENTRY_END_ET,
        "eod_minutes_et": NY_CLOSE_ET,
        "breakeven_after_bars": 0,
        "sl_cap": 50,
        "tp_cap": 30,
    },
    "vScalpB": {
        "instrument": "MNQ",
        "sm_params": (MNQ_SM_INDEX, MNQ_SM_FLOW, MNQ_SM_NORM, MNQ_SM_EMA),
        "rsi_len": VSCALPB_RSI_LEN,
        "rsi_buy": VSCALPB_RSI_BUY,
        "rsi_sell": VSCALPB_RSI_SELL,
        "sm_threshold": VSCALPB_SM_THRESHOLD,
        "cooldown": VSCALPB_COOLDOWN,
        "fixed_sl": VSCALPB_MAX_LOSS_PTS,
        "fixed_tp": VSCALPB_TP_PTS,
        "dollar_per_pt": MNQ_DOLLAR_PER_PT,
        "commission": MNQ_COMMISSION,
        "entry_end_et": NY_LAST_ENTRY_ET,
        "eod_minutes_et": NY_CLOSE_ET,
        "breakeven_after_bars": 0,
        "sl_cap": 50,
        "tp_cap": 30,
    },
    "MES_V2": {
        "instrument": "MES",
        "sm_params": (MES_SM_INDEX, MES_SM_FLOW, MES_SM_NORM, MES_SM_EMA),
        "rsi_len": MESV2_RSI_LEN,
        "rsi_buy": MESV2_RSI_BUY,
        "rsi_sell": MESV2_RSI_SELL,
        "sm_threshold": MESV2_SM_THRESHOLD,
        "cooldown": MESV2_COOLDOWN,
        "fixed_sl": MESV2_MAX_LOSS_PTS,
        "fixed_tp": MESV2_TP_PTS,
        "dollar_per_pt": MES_DOLLAR_PER_PT,
        "commission": MES_COMMISSION,
        "entry_end_et": NY_LAST_ENTRY_ET,
        "eod_minutes_et": MESV2_EOD_ET,
        "breakeven_after_bars": MESV2_BREAKEVEN_BARS,
        "sl_cap": 40,
        "tp_cap": 25,
    },
}

# Sweep parameters
ATR_PERIODS = [10, 14, 20]
SL_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TP_MULTS = [0.5, 1.0, 1.5, 2.0, 3.0]


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(instrument, sm_params):
    """Load data for an instrument and compute SM with correct params."""
    # Use generate_session's loader (returns raw data without SM for Databento)
    df = load_instrument_1min(instrument)

    if 'SM_Net' not in df.columns or instrument == "MNQ":
        # Recompute SM with correct params per instrument
        sm = compute_smart_money(
            df['Close'].values, df['Volume'].values,
            index_period=sm_params[0], flow_period=sm_params[1],
            norm_period=sm_params[2], ema_len=sm_params[3],
        )
        df['SM_Net'] = sm

    print(f"Loaded {instrument}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def prepare_rsi(df, rsi_len):
    """Compute 5-min RSI mapped back to 1-min bars."""
    df_5m = resample_to_5min(df)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values, df_5m['Close'].values,
        rsi_len=rsi_len,
    )
    return rsi_curr, rsi_prev


# ============================================================================
# Main Sweep
# ============================================================================

def run_sweep():
    print("=" * 90)
    print("ATR-Scaled Exits Sweep — Test #1")
    print("=" * 90)

    # Load data once per instrument
    data_cache = {}  # instrument -> df
    rsi_cache = {}   # (instrument, rsi_len) -> (rsi_curr, rsi_prev)
    atr_cache = {}   # (instrument, period) -> atr array

    for strat_name, cfg in STRATEGIES.items():
        inst = cfg["instrument"]
        if inst not in data_cache:
            data_cache[inst] = prepare_data(inst, cfg["sm_params"])

        rsi_key = (inst, cfg["rsi_len"])
        if rsi_key not in rsi_cache:
            rsi_cache[rsi_key] = prepare_rsi(data_cache[inst], cfg["rsi_len"])

        for atr_period in ATR_PERIODS:
            atr_key = (inst, atr_period)
            if atr_key not in atr_cache:
                df = data_cache[inst]
                atr_cache[atr_key] = compute_atr(
                    df['High'].values, df['Low'].values,
                    df['Close'].values, atr_period,
                )

    # --- Run fixed baselines ---
    print("\n" + "=" * 90)
    print("FIXED BASELINES (production params)")
    print("=" * 90)

    baselines = {}
    for strat_name, cfg in STRATEGIES.items():
        inst = cfg["instrument"]
        df = data_cache[inst]
        rsi_curr, rsi_prev = rsi_cache[(inst, cfg["rsi_len"])]

        trades = run_backtest_tp_exit(
            df['Open'].values, df['High'].values, df['Low'].values,
            df['Close'].values, df['SM_Net'].values, df.index,
            rsi_curr, rsi_prev,
            rsi_buy=cfg["rsi_buy"], rsi_sell=cfg["rsi_sell"],
            sm_threshold=cfg["sm_threshold"],
            cooldown_bars=cfg["cooldown"],
            max_loss_pts=cfg["fixed_sl"],
            tp_pts=cfg["fixed_tp"],
            eod_minutes_et=cfg["eod_minutes_et"],
            breakeven_after_bars=cfg["breakeven_after_bars"],
            entry_end_et=cfg["entry_end_et"],
        )
        sc = score_trades(trades, commission_per_side=cfg["commission"],
                         dollar_per_pt=cfg["dollar_per_pt"])
        baselines[strat_name] = sc
        label = f"{strat_name} (SL={cfg['fixed_sl']}, TP={cfg['fixed_tp']})"
        print(fmt_score(sc, label))

    # --- Run ATR sweep for each strategy ---
    all_results = {}  # strat_name -> list of result dicts

    for strat_name, cfg in STRATEGIES.items():
        inst = cfg["instrument"]
        df = data_cache[inst]
        rsi_curr, rsi_prev = rsi_cache[(inst, cfg["rsi_len"])]

        opens = df['Open'].values
        highs_arr = df['High'].values
        lows_arr = df['Low'].values
        closes = df['Close'].values
        sm = df['SM_Net'].values
        times = df.index

        results = []
        combos = list(product(ATR_PERIODS, SL_MULTS, TP_MULTS))
        total = len(combos)

        print(f"\n{'=' * 90}")
        print(f"SWEEPING {strat_name} — {total} parameter combos")
        print(f"{'=' * 90}")

        for idx, (atr_period, sl_mult, tp_mult) in enumerate(combos):
            atr_arr = atr_cache[(inst, atr_period)]

            trades = run_backtest_atr_exits(
                opens, highs_arr, lows_arr, closes, sm, times,
                rsi_curr, rsi_prev,
                atr_arr,
                rsi_buy=cfg["rsi_buy"], rsi_sell=cfg["rsi_sell"],
                sm_threshold=cfg["sm_threshold"],
                cooldown_bars=cfg["cooldown"],
                sl_mult=sl_mult, tp_mult=tp_mult,
                sl_cap=cfg["sl_cap"], tp_cap=cfg["tp_cap"],
                eod_minutes_et=cfg["eod_minutes_et"],
                breakeven_after_bars=cfg["breakeven_after_bars"],
                entry_end_et=cfg["entry_end_et"],
            )

            sc = score_trades(trades, commission_per_side=cfg["commission"],
                             dollar_per_pt=cfg["dollar_per_pt"])

            if sc is not None:
                # Compute mean ATR at entry for reference
                entry_atrs = [atr_arr[t["entry_idx"]] for t in trades]
                mean_atr = np.mean(entry_atrs) if entry_atrs else 0

                results.append({
                    "atr_period": atr_period,
                    "sl_mult": sl_mult,
                    "tp_mult": tp_mult,
                    "mean_entry_atr": round(mean_atr, 2),
                    "mean_sl_pts": round(min(sl_mult * mean_atr, cfg["sl_cap"]), 1),
                    "mean_tp_pts": round(min(tp_mult * mean_atr, cfg["tp_cap"]), 1),
                    "trades": trades,
                    **sc,
                })

            if (idx + 1) % 30 == 0:
                print(f"  ... {idx + 1}/{total} combos done")

        all_results[strat_name] = results
        print(f"  Completed {len(results)} valid combos for {strat_name}")

    # ========================================================================
    # OUTPUT TABLES
    # ========================================================================

    for strat_name in STRATEGIES:
        results = all_results[strat_name]
        cfg = STRATEGIES[strat_name]
        baseline = baselines[strat_name]

        print(f"\n{'=' * 90}")
        print(f"TABLE 1: Full Sweep Grid — {strat_name}")
        print(f"Baseline: SL={cfg['fixed_sl']}, TP={cfg['fixed_tp']} → "
              f"{baseline['count']} trades, WR {baseline['win_rate']}%, "
              f"PF {baseline['pf']}, ${baseline['net_dollar']:+.2f}, "
              f"Sharpe {baseline['sharpe']}")
        print(f"{'=' * 90}")

        header = (f"{'ATR':>3} {'SL_m':>5} {'TP_m':>5} | {'~SL':>5} {'~TP':>5} | "
                  f"{'N':>4} {'WR%':>6} {'PF':>6} {'P&L':>10} {'Sharpe':>7} {'MaxDD':>8} | "
                  f"{'Exits':>20}")
        print(header)
        print("-" * len(header))

        # Sort by Sharpe descending
        sorted_results = sorted(results, key=lambda r: r["sharpe"], reverse=True)

        for r in sorted_results:
            exits_str = " ".join(f"{k}:{v}" for k, v in sorted(r["exits"].items()))
            print(f"{r['atr_period']:>3} {r['sl_mult']:>5.1f} {r['tp_mult']:>5.1f} | "
                  f"{r['mean_sl_pts']:>5.1f} {r['mean_tp_pts']:>5.1f} | "
                  f"{r['count']:>4} {r['win_rate']:>5.1f}% {r['pf']:>6.3f} "
                  f"${r['net_dollar']:>+9.2f} {r['sharpe']:>7.3f} "
                  f"${r['max_dd_dollar']:>7.2f} | {exits_str:>20}")

        # Top 10 by Sharpe
        print(f"\n{'=' * 90}")
        print(f"TABLE 2: Top 10 by Sharpe — {strat_name}")
        print(f"{'=' * 90}")
        print(f"{'Rank':>4} {'ATR':>3} {'SL_m':>5} {'TP_m':>5} | {'~SL':>5} {'~TP':>5} | "
              f"{'N':>4} {'WR%':>6} {'PF':>6} {'P&L':>10} {'Sharpe':>7} {'MaxDD':>8}")
        print("-" * 85)

        for rank, r in enumerate(sorted_results[:10], 1):
            delta_sharpe = r["sharpe"] - baseline["sharpe"]
            delta_pnl = r["net_dollar"] - baseline["net_dollar"]
            print(f"{rank:>4} {r['atr_period']:>3} {r['sl_mult']:>5.1f} {r['tp_mult']:>5.1f} | "
                  f"{r['mean_sl_pts']:>5.1f} {r['mean_tp_pts']:>5.1f} | "
                  f"{r['count']:>4} {r['win_rate']:>5.1f}% {r['pf']:>6.3f} "
                  f"${r['net_dollar']:>+9.2f} {r['sharpe']:>7.3f} "
                  f"${r['max_dd_dollar']:>7.2f}  "
                  f"(dSharpe {delta_sharpe:+.3f}, dP&L ${delta_pnl:+.0f})")

    # ========================================================================
    # IS/OOS VALIDATION for Top 5 per strategy
    # ========================================================================

    print(f"\n{'=' * 90}")
    print("TABLE 3: IS/OOS Split Validation (Top 5 per strategy by Sharpe)")
    print("First half = IS, Second half = OOS")
    print(f"{'=' * 90}")

    for strat_name in STRATEGIES:
        results = all_results[strat_name]
        cfg = STRATEGIES[strat_name]
        baseline = baselines[strat_name]
        inst = cfg["instrument"]
        df = data_cache[inst]

        # Split data in half
        n_total = len(df)
        n_half = n_total // 2

        # IS data
        df_is = df.iloc[:n_half]
        # OOS data
        df_oos = df.iloc[n_half:]

        # Prepare RSI for each half
        rsi_is_curr, rsi_is_prev = prepare_rsi(df_is, cfg["rsi_len"])
        rsi_oos_curr, rsi_oos_prev = prepare_rsi(df_oos, cfg["rsi_len"])

        # Sort results by Sharpe, take top 5
        sorted_results = sorted(results, key=lambda r: r["sharpe"], reverse=True)
        top5 = sorted_results[:5]

        print(f"\n--- {strat_name} ---")
        print(f"  IS: {len(df_is)} bars ({df_is.index[0]} to {df_is.index[-1]})")
        print(f"  OOS: {len(df_oos)} bars ({df_oos.index[0]} to {df_oos.index[-1]})")

        # Fixed baseline IS/OOS
        baseline_is_trades = run_backtest_tp_exit(
            df_is['Open'].values, df_is['High'].values, df_is['Low'].values,
            df_is['Close'].values, df_is['SM_Net'].values, df_is.index,
            rsi_is_curr, rsi_is_prev,
            rsi_buy=cfg["rsi_buy"], rsi_sell=cfg["rsi_sell"],
            sm_threshold=cfg["sm_threshold"],
            cooldown_bars=cfg["cooldown"],
            max_loss_pts=cfg["fixed_sl"],
            tp_pts=cfg["fixed_tp"],
            eod_minutes_et=cfg["eod_minutes_et"],
            breakeven_after_bars=cfg["breakeven_after_bars"],
            entry_end_et=cfg["entry_end_et"],
        )
        baseline_oos_trades = run_backtest_tp_exit(
            df_oos['Open'].values, df_oos['High'].values, df_oos['Low'].values,
            df_oos['Close'].values, df_oos['SM_Net'].values, df_oos.index,
            rsi_oos_curr, rsi_oos_prev,
            rsi_buy=cfg["rsi_buy"], rsi_sell=cfg["rsi_sell"],
            sm_threshold=cfg["sm_threshold"],
            cooldown_bars=cfg["cooldown"],
            max_loss_pts=cfg["fixed_sl"],
            tp_pts=cfg["fixed_tp"],
            eod_minutes_et=cfg["eod_minutes_et"],
            breakeven_after_bars=cfg["breakeven_after_bars"],
            entry_end_et=cfg["entry_end_et"],
        )

        bl_is = score_trades(baseline_is_trades, commission_per_side=cfg["commission"],
                            dollar_per_pt=cfg["dollar_per_pt"])
        bl_oos = score_trades(baseline_oos_trades, commission_per_side=cfg["commission"],
                             dollar_per_pt=cfg["dollar_per_pt"])

        print(f"\n  BASELINE (fixed SL={cfg['fixed_sl']}, TP={cfg['fixed_tp']}):")
        print(f"    FULL: {fmt_score(baseline, '')}")
        print(f"    IS  : {fmt_score(bl_is, '')}")
        print(f"    OOS : {fmt_score(bl_oos, '')}")

        print(f"\n  {'Rank':>4} {'ATR':>3} {'SL_m':>5} {'TP_m':>5} | "
              f"{'Period':>6} {'N':>4} {'WR%':>6} {'PF':>6} {'P&L':>10} {'Sharpe':>7}")
        print(f"  " + "-" * 80)

        for rank, r in enumerate(top5, 1):
            atr_period = r["atr_period"]
            sl_mult = r["sl_mult"]
            tp_mult = r["tp_mult"]

            # Compute ATR for IS and OOS halves
            atr_is = compute_atr(
                df_is['High'].values, df_is['Low'].values,
                df_is['Close'].values, atr_period,
            )
            atr_oos = compute_atr(
                df_oos['High'].values, df_oos['Low'].values,
                df_oos['Close'].values, atr_period,
            )

            # IS run
            is_trades = run_backtest_atr_exits(
                df_is['Open'].values, df_is['High'].values,
                df_is['Low'].values, df_is['Close'].values,
                df_is['SM_Net'].values, df_is.index,
                rsi_is_curr, rsi_is_prev,
                atr_is,
                rsi_buy=cfg["rsi_buy"], rsi_sell=cfg["rsi_sell"],
                sm_threshold=cfg["sm_threshold"],
                cooldown_bars=cfg["cooldown"],
                sl_mult=sl_mult, tp_mult=tp_mult,
                sl_cap=cfg["sl_cap"], tp_cap=cfg["tp_cap"],
                eod_minutes_et=cfg["eod_minutes_et"],
                breakeven_after_bars=cfg["breakeven_after_bars"],
                entry_end_et=cfg["entry_end_et"],
            )
            is_sc = score_trades(is_trades, commission_per_side=cfg["commission"],
                                dollar_per_pt=cfg["dollar_per_pt"])

            # OOS run
            oos_trades = run_backtest_atr_exits(
                df_oos['Open'].values, df_oos['High'].values,
                df_oos['Low'].values, df_oos['Close'].values,
                df_oos['SM_Net'].values, df_oos.index,
                rsi_oos_curr, rsi_oos_prev,
                atr_oos,
                rsi_buy=cfg["rsi_buy"], rsi_sell=cfg["rsi_sell"],
                sm_threshold=cfg["sm_threshold"],
                cooldown_bars=cfg["cooldown"],
                sl_mult=sl_mult, tp_mult=tp_mult,
                sl_cap=cfg["sl_cap"], tp_cap=cfg["tp_cap"],
                eod_minutes_et=cfg["eod_minutes_et"],
                breakeven_after_bars=cfg["breakeven_after_bars"],
                entry_end_et=cfg["entry_end_et"],
            )
            oos_sc = score_trades(oos_trades, commission_per_side=cfg["commission"],
                                 dollar_per_pt=cfg["dollar_per_pt"])

            print(f"  #{rank:>3} ATR={atr_period:>2} SL={sl_mult:.1f}x TP={tp_mult:.1f}x")
            if r:
                print(f"        FULL: {r['count']:>4} trades, WR {r['win_rate']:>5.1f}%, "
                      f"PF {r['pf']:>6.3f}, ${r['net_dollar']:>+9.2f}, "
                      f"Sharpe {r['sharpe']:>6.3f}")
            print(f"        IS  : {fmt_score(is_sc, '')}")
            print(f"        OOS : {fmt_score(oos_sc, '')}")

            # Flag if OOS degrades significantly
            if is_sc and oos_sc:
                pf_ratio = oos_sc["pf"] / is_sc["pf"] if is_sc["pf"] > 0 else 0
                if pf_ratio < 0.7:
                    print(f"        *** WARNING: OOS PF is {pf_ratio:.1%} of IS PF — likely overfit ***")
                elif oos_sc["pf"] >= 1.0 and is_sc["pf"] >= 1.0:
                    print(f"        OK: IS PF {is_sc['pf']:.3f} -> OOS PF {oos_sc['pf']:.3f} "
                          f"(ratio {pf_ratio:.2f})")

    # ========================================================================
    # TABLE 4: Comparison vs Fixed Baseline
    # ========================================================================

    print(f"\n{'=' * 90}")
    print("TABLE 4: Best ATR-Scaled vs Fixed Baseline (Full Data)")
    print(f"{'=' * 90}")

    for strat_name in STRATEGIES:
        results = all_results[strat_name]
        cfg = STRATEGIES[strat_name]
        baseline = baselines[strat_name]

        sorted_results = sorted(results, key=lambda r: r["sharpe"], reverse=True)
        best = sorted_results[0] if sorted_results else None

        print(f"\n--- {strat_name} ---")
        print(f"  Fixed Baseline (SL={cfg['fixed_sl']}, TP={cfg['fixed_tp']}):")
        print(f"    N={baseline['count']}, WR={baseline['win_rate']}%, "
              f"PF={baseline['pf']}, P&L=${baseline['net_dollar']:+.2f}, "
              f"Sharpe={baseline['sharpe']}, MaxDD=${baseline['max_dd_dollar']:.2f}")

        if best:
            print(f"  Best ATR-Scaled (ATR={best['atr_period']}, "
                  f"SL={best['sl_mult']}x, TP={best['tp_mult']}x, "
                  f"~SL={best['mean_sl_pts']:.1f}, ~TP={best['mean_tp_pts']:.1f}):")
            print(f"    N={best['count']}, WR={best['win_rate']}%, "
                  f"PF={best['pf']}, P&L=${best['net_dollar']:+.2f}, "
                  f"Sharpe={best['sharpe']}, MaxDD=${best['max_dd_dollar']:.2f}")

            d_sharpe = best['sharpe'] - baseline['sharpe']
            d_pnl = best['net_dollar'] - baseline['net_dollar']
            d_pf = best['pf'] - baseline['pf']
            print(f"  Delta: Sharpe {d_sharpe:+.3f}, PF {d_pf:+.3f}, P&L ${d_pnl:+.0f}")

            if d_sharpe > 0 and d_pf > 0:
                print(f"  --> ATR-scaled IMPROVES on baseline")
            elif d_sharpe > 0:
                print(f"  --> ATR-scaled improves Sharpe but not PF — mixed signal")
            else:
                print(f"  --> Fixed baseline WINS — ATR scaling does not help")

    # ATR stats summary
    print(f"\n{'=' * 90}")
    print("ATR Statistics at Entry (for reference)")
    print(f"{'=' * 90}")

    for strat_name in STRATEGIES:
        results = all_results[strat_name]
        if not results:
            continue
        # Use ATR=14 results for reference
        atr14_results = [r for r in results if r["atr_period"] == 14]
        if atr14_results:
            mean_atr = atr14_results[0]["mean_entry_atr"]
            cfg = STRATEGIES[strat_name]
            inst = cfg["instrument"]
            df = data_cache[inst]
            atr14 = atr_cache[(inst, 14)]
            # Overall ATR stats
            valid_atr = atr14[atr14 > 0]
            print(f"\n  {strat_name} ({inst}) ATR(14) 1-min:")
            print(f"    Mean={np.mean(valid_atr):.2f}, Median={np.median(valid_atr):.2f}, "
                  f"Std={np.std(valid_atr):.2f}")
            print(f"    P10={np.percentile(valid_atr, 10):.2f}, "
                  f"P25={np.percentile(valid_atr, 25):.2f}, "
                  f"P75={np.percentile(valid_atr, 75):.2f}, "
                  f"P90={np.percentile(valid_atr, 90):.2f}")
            print(f"    At entry (mean): {mean_atr:.2f}")
            print(f"    Fixed SL={cfg['fixed_sl']} = {cfg['fixed_sl']/mean_atr:.1f}x ATR, "
                  f"Fixed TP={cfg['fixed_tp']} = {cfg['fixed_tp']/mean_atr:.1f}x ATR")

    print(f"\n{'=' * 90}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    run_sweep()
