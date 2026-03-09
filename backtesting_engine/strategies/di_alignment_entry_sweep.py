"""
Test #3 — DI+/DI- Directional Alignment Entry Gate Sweep
=========================================================
Hypothesis: Require directional alignment from Wilder's DI+/DI- as an
entry filter. LONG only when DI+ > DI- (+ optional spread threshold);
SHORT only when DI- > DI+ (+ optional spread threshold).

Since entry_gate is a single boolean array (direction-agnostic), this test
uses a CUSTOM backtest loop copied from run_backtest_tp_exit but with an
added directional DI check at entry time.

Sweep dimensions:
  - DI period: [7, 10, 14, 20]
  - Timeframe: 1-min, 5-min (computed on 5-min, mapped back to 1-min)
  - Spread threshold: [0, 2, 5, 10, 15, 20]
    (0 = strict alignment, >0 = require DI spread exceeds threshold)
  - All 3 strategies: vScalpA, vScalpB, MES_V2
"""

import sys
from pathlib import Path

# Path setup — MUST come before imports
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR))
sys.path.insert(0, str(_STRAT_DIR.parent))  # backtesting_engine/
sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))

import numpy as np
import pandas as pd

from v10_test_common import (
    compute_rsi,
    compute_smart_money,
    compute_et_minutes,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)
from generate_session import compute_mfe_mae

# ---------------------------------------------------------------------------
# Strategy config table (production params)
# ---------------------------------------------------------------------------

STRATEGIES = [
    {
        "name": "vScalpA", "strategy_id": "MNQ_V15", "instrument": "MNQ",
        "sm_index": 10, "sm_flow": 12, "sm_norm": 200, "sm_ema": 100,
        "rsi_len": 8, "rsi_buy": 60, "rsi_sell": 40, "sm_threshold": 0.0,
        "cooldown": 20, "max_loss_pts": 40, "tp_pts": 5,
        "eod_et": NY_CLOSE_ET, "breakeven_bars": 0,
        "entry_end_et": 13 * 60, "dollar_per_pt": 2.0, "commission": 0.52,
    },
    {
        "name": "vScalpB", "strategy_id": "MNQ_VSCALPB", "instrument": "MNQ",
        "sm_index": 10, "sm_flow": 12, "sm_norm": 200, "sm_ema": 100,
        "rsi_len": 8, "rsi_buy": 55, "rsi_sell": 45, "sm_threshold": 0.25,
        "cooldown": 20, "max_loss_pts": 15, "tp_pts": 5,
        "eod_et": NY_CLOSE_ET, "breakeven_bars": 0,
        "entry_end_et": NY_LAST_ENTRY_ET, "dollar_per_pt": 2.0, "commission": 0.52,
    },
    {
        "name": "MES_V2", "strategy_id": "MES_V2", "instrument": "MES",
        "sm_index": 20, "sm_flow": 12, "sm_norm": 400, "sm_ema": 255,
        "rsi_len": 12, "rsi_buy": 55, "rsi_sell": 45, "sm_threshold": 0.0,
        "cooldown": 25, "max_loss_pts": 35, "tp_pts": 20,
        "eod_et": 15 * 60 + 30, "breakeven_bars": 75,
        "entry_end_et": NY_LAST_ENTRY_ET, "dollar_per_pt": 5.0, "commission": 1.25,
    },
]


# ---------------------------------------------------------------------------
# DI+/DI- Computation (Wilder's method)
# ---------------------------------------------------------------------------

def compute_di(highs, lows, closes, period=14):
    """Compute DI+ and DI- using Wilder's smoothing.

    Args:
        highs, lows, closes: numpy arrays of OHLC data
        period: DI smoothing period (Wilder's)

    Returns:
        (di_plus, di_minus) — numpy arrays, same length as input.
        First `period` values are NaN.
    """
    n = len(highs)
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)

    if n < period + 1:
        return di_plus, di_minus

    # Step 1: compute raw +DM, -DM, TR for each bar
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Step 2: Wilder's smoothing — first value is SMA, then smoothed
    smooth_plus_dm = np.zeros(n)
    smooth_minus_dm = np.zeros(n)
    smooth_tr = np.zeros(n)

    # Initial sum for period bars (indices 1..period)
    smooth_plus_dm[period] = np.sum(plus_dm[1:period + 1])
    smooth_minus_dm[period] = np.sum(minus_dm[1:period + 1])
    smooth_tr[period] = np.sum(tr[1:period + 1])

    for i in range(period + 1, n):
        smooth_plus_dm[i] = smooth_plus_dm[i - 1] - smooth_plus_dm[i - 1] / period + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i - 1] - smooth_minus_dm[i - 1] / period + minus_dm[i]
        smooth_tr[i] = smooth_tr[i - 1] - smooth_tr[i - 1] / period + tr[i]

    # Step 3: DI+ and DI-
    for i in range(period, n):
        if smooth_tr[i] > 0:
            di_plus[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i]
            di_minus[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i]
        else:
            di_plus[i] = 0.0
            di_minus[i] = 0.0

    return di_plus, di_minus


def map_5min_di_to_1min(onemin_times, fivemin_times, di_plus_5m, di_minus_5m):
    """Map 5-min DI+/DI- values to 1-min timestamps (forward-fill like RSI).

    Uses bar[j-1] value (completed 5-min bar) — same as RSI mapping.
    """
    n_1m = len(onemin_times)
    di_plus_1m = np.full(n_1m, np.nan)
    di_minus_1m = np.full(n_1m, np.nan)

    j = 0
    for i in range(n_1m):
        while j + 1 < len(fivemin_times) and fivemin_times[j + 1] <= onemin_times[i]:
            j += 1
        if j >= 1:
            di_plus_1m[i] = di_plus_5m[j - 1]
            di_minus_1m[i] = di_minus_5m[j - 1]

    return di_plus_1m, di_minus_1m


# ---------------------------------------------------------------------------
# Custom backtest loop with directional DI gate
# ---------------------------------------------------------------------------

def run_backtest_di_gate(opens, highs, lows, closes, sm, times,
                         rsi_5m_curr, rsi_5m_prev,
                         rsi_buy, rsi_sell, sm_threshold,
                         cooldown_bars, max_loss_pts, tp_pts,
                         eod_minutes_et, breakeven_after_bars,
                         entry_end_et,
                         di_plus, di_minus, di_spread_threshold):
    """Backtest with directional DI alignment gate.

    Copied from run_backtest_tp_exit with the addition of:
      - LONG: requires di_plus[i-1] - di_minus[i-1] > di_spread_threshold
      - SHORT: requires di_minus[i-1] - di_plus[i-1] > di_spread_threshold

    di_spread_threshold=0 means strict alignment (DI+ > DI- for longs).
    """
    n = len(opens)
    trades = []
    trade_state = 0     # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

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

        # Episode reset
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

        # Exits for open positions
        if trade_state == 1:
            # SL
            if max_loss_pts > 0 and closes[i - 1] <= entry_price - max_loss_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # TP
            if tp_pts > 0 and closes[i - 1] >= entry_price + tp_pts:
                close_trade("long", entry_price, opens[i], entry_idx, i, "TP")
                trade_state = 0
                exit_bar = i
                continue

            # BE_TIME
            if breakeven_after_bars > 0:
                bars_in_trade = (i - 1) - entry_idx
                if bars_in_trade >= breakeven_after_bars:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "BE_TIME")
                    trade_state = 0
                    exit_bar = i
                    continue

        elif trade_state == -1:
            # SL
            if max_loss_pts > 0 and closes[i - 1] >= entry_price + max_loss_pts:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SL")
                trade_state = 0
                exit_bar = i
                continue

            # TP
            if tp_pts > 0 and closes[i - 1] <= entry_price - tp_pts:
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
                # DI directional check on bar[i-1]
                di_p = di_plus[i - 1]
                di_m = di_minus[i - 1]

                # Skip if DI values are NaN (warmup period)
                di_valid = not (np.isnan(di_p) or np.isnan(di_m))

                if sm_bull and rsi_long_trigger and not long_used:
                    # LONG: require DI+ - DI- > threshold
                    if di_valid and (di_p - di_m) > di_spread_threshold:
                        trade_state = 1
                        entry_price = opens[i]
                        entry_idx = i
                        long_used = True

                elif sm_bear and rsi_short_trigger and not short_used:
                    # SHORT: require DI- - DI+ > threshold
                    if di_valid and (di_m - di_p) > di_spread_threshold:
                        trade_state = -1
                        entry_price = opens[i]
                        entry_idx = i
                        short_used = True

    return trades


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data():
    """Load MNQ + MES, compute SM, split IS/OOS, build arrays + DI cache."""
    instruments = {}
    for inst in ["MNQ", "MES"]:
        df = load_instrument_1min(inst)
        strat = next(s for s in STRATEGIES if s["instrument"] == inst)
        sm = compute_smart_money(
            df["Close"].values, df["Volume"].values,
            index_period=strat["sm_index"], flow_period=strat["sm_flow"],
            norm_period=strat["sm_norm"], ema_len=strat["sm_ema"],
        )
        df["SM_Net"] = sm
        instruments[inst] = df
        print(f"Loaded {inst}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Split each instrument at midpoint
    split_indices = {}
    for inst, df in instruments.items():
        mid = df.index[len(df) // 2]
        is_len = (df.index < mid).sum()
        split_indices[inst] = {"is_len": is_len, "midpoint": mid}
        is_range = f"{df.index[0].strftime('%Y-%m-%d')}_to_{df.index[is_len-1].strftime('%Y-%m-%d')}"
        oos_range = f"{df.index[is_len].strftime('%Y-%m-%d')}_to_{df.index[-1].strftime('%Y-%m-%d')}"
        print(f"  {inst} split: IS {is_len} bars ({is_range}), OOS {len(df)-is_len} bars ({oos_range})")

    # Build per-(strategy, split) arrays with RSI
    split_arrays = {}
    rsi_cache = {}

    for strat in STRATEGIES:
        inst = strat["instrument"]
        df = instruments[inst]
        mid = split_indices[inst]["midpoint"]

        splits = {
            "FULL": df,
            "IS": df[df.index < mid],
            "OOS": df[df.index >= mid],
        }

        for split_name, df_split in splits.items():
            cache_key = (inst, strat["rsi_len"], split_name)
            if cache_key in rsi_cache:
                rsi_curr, rsi_prev = rsi_cache[cache_key]
            else:
                df_5m = resample_to_5min(df_split)
                rsi_curr, rsi_prev = map_5min_rsi_to_1min(
                    df_split.index.values, df_5m.index.values,
                    df_5m["Close"].values, rsi_len=strat["rsi_len"],
                )
                rsi_cache[cache_key] = (rsi_curr, rsi_prev)

            split_arrays[(strat["name"], split_name)] = (
                df_split["Open"].values,
                df_split["High"].values,
                df_split["Low"].values,
                df_split["Close"].values,
                df_split["SM_Net"].values,
                df_split.index,
                rsi_curr,
                rsi_prev,
            )

    return instruments, split_arrays, split_indices


def compute_all_di(instruments, split_indices):
    """Pre-compute DI+/DI- for all periods, both 1-min and 5-min timeframes.

    Returns:
        di_cache: dict (instrument, period, timeframe, split_name)
                  -> (di_plus, di_minus)
    """
    di_cache = {}
    periods = [7, 10, 14, 20]

    for inst, df in instruments.items():
        mid = split_indices[inst]["midpoint"]
        splits = {
            "FULL": df,
            "IS": df[df.index < mid],
            "OOS": df[df.index >= mid],
        }

        for period in periods:
            # --- 1-min DI ---
            # Compute on full data, then slice
            highs_full = df["High"].values
            lows_full = df["Low"].values
            closes_full = df["Close"].values
            di_p_full, di_m_full = compute_di(highs_full, lows_full, closes_full, period)

            is_len = split_indices[inst]["is_len"]
            di_cache[(inst, period, "1min", "FULL")] = (di_p_full, di_m_full)
            di_cache[(inst, period, "1min", "IS")] = (di_p_full[:is_len], di_m_full[:is_len])
            di_cache[(inst, period, "1min", "OOS")] = (di_p_full[is_len:], di_m_full[is_len:])

            # --- 5-min DI (compute per split to avoid look-ahead) ---
            for split_name, df_split in splits.items():
                df_5m = resample_to_5min(df_split)
                di_p_5m, di_m_5m = compute_di(
                    df_5m["High"].values, df_5m["Low"].values,
                    df_5m["Close"].values, period,
                )
                # Map back to 1-min
                di_p_1m, di_m_1m = map_5min_di_to_1min(
                    df_split.index.values, df_5m.index.values,
                    di_p_5m, di_m_5m,
                )
                di_cache[(inst, period, "5min", split_name)] = (di_p_1m, di_m_1m)

        print(f"  Computed DI for {inst}: {len(periods)} periods x 2 timeframes x 3 splits")

    return di_cache


# ---------------------------------------------------------------------------
# Run single backtest with DI gate
# ---------------------------------------------------------------------------

def run_single_di(strat, arrays, di_plus, di_minus, di_spread_threshold):
    """Run one backtest with directional DI gate."""
    opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev = arrays
    trades = run_backtest_di_gate(
        opens, highs, lows, closes, sm, times,
        rsi_curr, rsi_prev,
        rsi_buy=strat["rsi_buy"], rsi_sell=strat["rsi_sell"],
        sm_threshold=strat["sm_threshold"], cooldown_bars=strat["cooldown"],
        max_loss_pts=strat["max_loss_pts"], tp_pts=strat["tp_pts"],
        eod_minutes_et=strat["eod_et"],
        breakeven_after_bars=strat["breakeven_bars"],
        entry_end_et=strat["entry_end_et"],
        di_plus=di_plus, di_minus=di_minus,
        di_spread_threshold=di_spread_threshold,
    )
    compute_mfe_mae(trades, highs, lows)
    return trades


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    print("=" * 120)
    print("DI+/DI- DIRECTIONAL ALIGNMENT ENTRY GATE SWEEP")
    print("=" * 120)

    instruments, split_arrays, split_indices = prepare_data()

    print("\nComputing DI indicators...")
    di_cache = compute_all_di(instruments, split_indices)

    # --- Baselines (no DI gate = run standard backtest) ---
    print("\n--- BASELINES (no DI gate) ---")
    baselines = {}

    # Use run_backtest_tp_exit for baseline (import from generate_session)
    from generate_session import run_backtest_tp_exit

    for strat in STRATEGIES:
        for split_name in ["FULL", "IS", "OOS"]:
            arrays = split_arrays[(strat["name"], split_name)]
            opens, highs, lows, closes, sm, times, rsi_curr, rsi_prev = arrays
            trades = run_backtest_tp_exit(
                opens, highs, lows, closes, sm, times,
                rsi_curr, rsi_prev,
                rsi_buy=strat["rsi_buy"], rsi_sell=strat["rsi_sell"],
                sm_threshold=strat["sm_threshold"], cooldown_bars=strat["cooldown"],
                max_loss_pts=strat["max_loss_pts"], tp_pts=strat["tp_pts"],
                eod_minutes_et=strat["eod_et"],
                breakeven_after_bars=strat["breakeven_bars"],
                entry_end_et=strat["entry_end_et"],
            )
            compute_mfe_mae(trades, highs, lows)
            sc = score_trades(trades, commission_per_side=strat["commission"],
                              dollar_per_pt=strat["dollar_per_pt"])
            baselines[(strat["name"], split_name)] = (sc, trades)
            if split_name == "FULL" and sc:
                print(f"  {strat['name']:>8}: {sc['count']} trades, "
                      f"WR {sc['win_rate']}%, PF {sc['pf']}, "
                      f"${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']}")

    # --- Sweep configs ---
    di_periods = [7, 10, 14, 20]
    timeframes = ["1min", "5min"]
    spread_thresholds = [0, 2, 5, 10, 15, 20]

    configs = []
    for period in di_periods:
        for tf in timeframes:
            for thresh in spread_thresholds:
                label = f"DI{period}_{tf}_t{thresh}"
                configs.append({
                    "label": label,
                    "period": period,
                    "timeframe": tf,
                    "spread_threshold": thresh,
                })

    print(f"\nSweep: {len(configs)} configs x {len(STRATEGIES)} strategies x 3 splits "
          f"= {len(configs) * len(STRATEGIES) * 3} backtests")

    # --- Run sweep ---
    all_results = []  # (label, strat_name, split_name, sc, trades)

    for ci, cfg in enumerate(configs):
        label = cfg["label"]
        if ci % 12 == 0:
            print(f"  Running config {ci+1}/{len(configs)}: {label} ...")

        for strat in STRATEGIES:
            inst = strat["instrument"]
            for split_name in ["FULL", "IS", "OOS"]:
                arrays = split_arrays[(strat["name"], split_name)]
                di_key = (inst, cfg["period"], cfg["timeframe"], split_name)
                di_plus, di_minus = di_cache[di_key]

                trades = run_single_di(strat, arrays, di_plus, di_minus,
                                       cfg["spread_threshold"])
                sc = score_trades(trades, commission_per_side=strat["commission"],
                                  dollar_per_pt=strat["dollar_per_pt"])
                all_results.append((label, strat["name"], split_name, sc, trades))

    # ---------------------------------------------------------------------------
    # 1. FULL sweep grid per strategy
    # ---------------------------------------------------------------------------
    for strat in STRATEGIES:
        sname = strat["name"]
        bl_sc = baselines[(sname, "FULL")][0]

        print(f"\n{'='*140}")
        print(f"FULL SWEEP: {sname} (baseline: {bl_sc['count']} trades, "
              f"WR {bl_sc['win_rate']}%, PF {bl_sc['pf']}, ${bl_sc['net_dollar']:+.0f}, "
              f"Sharpe {bl_sc['sharpe']})")
        print(f"{'='*140}")
        print(f"{'Config':>22} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Net$':>10} "
              f"{'MaxDD':>9} {'Sharpe':>7} {'Blocked':>8} {'dPF%':>7} {'dSharpe':>8}")
        print(f"{'-'*140}")

        for label, s, sp, sc, trades in all_results:
            if s != sname or sp != "FULL":
                continue
            if sc is None:
                print(f"{label:>22}   NO TRADES")
                continue

            blocked = bl_sc["count"] - sc["count"]
            dpf = (sc["pf"] - bl_sc["pf"]) / bl_sc["pf"] * 100 if bl_sc["pf"] > 0 else 0
            dsharpe = sc["sharpe"] - bl_sc["sharpe"]
            marker = " *" if dpf > 5 else ""

            print(f"{label:>22} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
                  f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                  f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
                  f"{blocked:>8} {dpf:>+6.1f}% {dsharpe:>+7.3f}{marker}")

    # ---------------------------------------------------------------------------
    # 2. Top 10 per strategy by Sharpe (FULL data)
    # ---------------------------------------------------------------------------
    print(f"\n{'='*140}")
    print("TOP 10 PER STRATEGY BY SHARPE (FULL data)")
    print(f"{'='*140}")

    for strat in STRATEGIES:
        sname = strat["name"]
        bl_sc = baselines[(sname, "FULL")][0]

        full_results = [(label, sc) for label, s, sp, sc, _ in all_results
                        if s == sname and sp == "FULL" and sc is not None]

        full_results.sort(key=lambda x: x[1]["sharpe"], reverse=True)
        top10 = full_results[:10]

        print(f"\n  {sname} (baseline Sharpe {bl_sc['sharpe']}, PF {bl_sc['pf']}):")
        print(f"  {'Rank':>4} {'Config':>22} {'Trades':>7} {'WR%':>6} {'PF':>7} "
              f"{'Net$':>10} {'Sharpe':>7} {'dPF%':>7}")

        for rank, (label, sc) in enumerate(top10, 1):
            dpf = (sc["pf"] - bl_sc["pf"]) / bl_sc["pf"] * 100 if bl_sc["pf"] > 0 else 0
            print(f"  {rank:>4} {label:>22} {sc['count']:>7} {sc['win_rate']:>5.1f}% "
                  f"{sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                  f"{sc['sharpe']:>7.3f} {dpf:>+6.1f}%")

    # ---------------------------------------------------------------------------
    # 3. IS/OOS for configs with FULL PF improvement >5%
    # ---------------------------------------------------------------------------
    print(f"\n{'='*150}")
    print("IS/OOS VALIDATION — configs with FULL PF improvement >5% for ANY strategy")
    print(f"{'='*150}")

    # Find labels worth validating
    promising_labels = set()
    for label, s, sp, sc, _ in all_results:
        if sp != "FULL" or sc is None:
            continue
        bl_sc = baselines[(s, "FULL")][0]
        if bl_sc and bl_sc["pf"] > 0:
            dpf = (sc["pf"] - bl_sc["pf"]) / bl_sc["pf"] * 100
            if dpf > 5:
                promising_labels.add(label)

    if not promising_labels:
        print("\n  No configs improved FULL PF by >5% for any strategy.")
    else:
        print(f"\n  {len(promising_labels)} configs to validate")

        print(f"\n{'Config':>22} {'Strategy':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} "
              f"{'PF':>7} {'Net$':>10} {'MaxDD':>9} {'Sharpe':>7} "
              f"{'Blocked':>8} {'dPF%':>7} {'Verdict':>15}")
        print(f"{'-'*150}")

        for label in sorted(promising_labels):
            for strat in STRATEGIES:
                sname = strat["name"]
                for split_name in ["IS", "OOS"]:
                    sc = None
                    for l2, s2, sp2, sc2, _ in all_results:
                        if l2 == label and s2 == sname and sp2 == split_name:
                            sc = sc2
                            break

                    bl = baselines[(sname, split_name)][0]
                    if sc is None:
                        print(f"{label:>22} {sname:>8} {split_name:>5}   NO TRADES")
                        continue

                    bl_count = bl["count"] if bl else 0
                    blocked = bl_count - sc["count"]
                    dpf = 0.0
                    if bl and bl["pf"] > 0:
                        dpf = (sc["pf"] - bl["pf"]) / bl["pf"] * 100

                    # Verdict on OOS row
                    verdict = ""
                    if split_name == "OOS":
                        sc_is = None
                        for l2, s2, sp2, sc2, _ in all_results:
                            if l2 == label and s2 == sname and sp2 == "IS":
                                sc_is = sc2
                                break
                        if sc_is is not None and bl is not None:
                            bl_is = baselines[(sname, "IS")][0]
                            bl_oos = baselines[(sname, "OOS")][0]
                            verdict = _assess(sc_is, sc, bl_is, bl_oos)

                    print(f"{label:>22} {sname:>8} {split_name:>5} {sc['count']:>7} "
                          f"{sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} {sc['net_dollar']:>+10.2f} "
                          f"{sc['max_dd_dollar']:>9.2f} {sc['sharpe']:>7.3f} "
                          f"{blocked:>8} {dpf:>+6.1f}% {verdict:>15}")

    # ---------------------------------------------------------------------------
    # 4. Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")

    for strat in STRATEGIES:
        sname = strat["name"]
        bl_sc = baselines[(sname, "FULL")][0]

        # Find best FULL config
        best_label = None
        best_sharpe = bl_sc["sharpe"]
        for label, s, sp, sc, _ in all_results:
            if s == sname and sp == "FULL" and sc is not None:
                if sc["sharpe"] > best_sharpe:
                    best_sharpe = sc["sharpe"]
                    best_label = label

        # Find best IS/OOS validated config
        best_validated = None
        best_val_sharpe = bl_sc["sharpe"]
        for label in promising_labels:
            sc_is = sc_oos = None
            for l2, s2, sp2, sc2, _ in all_results:
                if l2 == label and s2 == sname:
                    if sp2 == "IS":
                        sc_is = sc2
                    elif sp2 == "OOS":
                        sc_oos = sc2

            if sc_is and sc_oos:
                bl_is = baselines[(sname, "IS")][0]
                bl_oos = baselines[(sname, "OOS")][0]
                v = _assess(sc_is, sc_oos, bl_is, bl_oos)
                if "PASS" in v:
                    # Use FULL sharpe as tiebreaker
                    for l2, s2, sp2, sc2, _ in all_results:
                        if l2 == label and s2 == sname and sp2 == "FULL":
                            if sc2 and sc2["sharpe"] > best_val_sharpe:
                                best_val_sharpe = sc2["sharpe"]
                                best_validated = (label, v, sc2)
                            break

        print(f"\n  {sname}:")
        print(f"    Baseline: {bl_sc['count']} trades, WR {bl_sc['win_rate']}%, "
              f"PF {bl_sc['pf']}, ${bl_sc['net_dollar']:+.0f}, Sharpe {bl_sc['sharpe']}")

        if best_label:
            best_sc = None
            for l2, s2, sp2, sc2, _ in all_results:
                if l2 == best_label and s2 == sname and sp2 == "FULL":
                    best_sc = sc2
                    break
            print(f"    Best FULL:  {best_label} -> {best_sc['count']} trades, "
                  f"WR {best_sc['win_rate']}%, PF {best_sc['pf']}, "
                  f"${best_sc['net_dollar']:+.0f}, Sharpe {best_sc['sharpe']}")
        else:
            print(f"    Best FULL:  No improvement over baseline")

        if best_validated:
            lbl, verdict, sc = best_validated
            print(f"    Best validated: {lbl} -> {verdict}: {sc['count']} trades, "
                  f"WR {sc['win_rate']}%, PF {sc['pf']}, "
                  f"${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']}")
        else:
            print(f"    Best validated: No config passed IS/OOS validation")

    print(f"\n{'='*120}")
    print("Done.")


def _assess(sc_is, sc_oos, bl_is, bl_oos):
    """Quick IS/OOS pass/fail verdict."""
    if sc_is is None or sc_oos is None or bl_is is None or bl_oos is None:
        return "FAIL"

    if sc_is["net_dollar"] <= 0 or sc_oos["net_dollar"] <= 0:
        return "FAIL"

    # Trade count >= 70% of baseline
    is_pct = sc_is["count"] / bl_is["count"] * 100 if bl_is["count"] > 0 else 0
    oos_pct = sc_oos["count"] / bl_oos["count"] * 100 if bl_oos["count"] > 0 else 0
    if is_pct < 70 or oos_pct < 70:
        return "FAIL"

    is_dpf = (sc_is["pf"] - bl_is["pf"]) / bl_is["pf"] * 100 if bl_is["pf"] > 0 else 0
    oos_dpf = (sc_oos["pf"] - bl_oos["pf"]) / bl_oos["pf"] * 100 if bl_oos["pf"] > 0 else 0

    is_dsharpe = sc_is["sharpe"] - bl_is["sharpe"]
    oos_dsharpe = sc_oos["sharpe"] - bl_oos["sharpe"]

    if is_dpf < -5 or oos_dpf < -5:
        return "FAIL"

    if is_dpf >= 5 and oos_dpf >= 5 and is_dsharpe >= 0 and oos_dsharpe >= 0:
        return "STRONG PASS"

    if is_dpf >= -5 and oos_dpf >= -5 and (is_dsharpe > 0 or oos_dsharpe > 0):
        return "MARGINAL PASS"

    return "FAIL"


if __name__ == "__main__":
    main()
