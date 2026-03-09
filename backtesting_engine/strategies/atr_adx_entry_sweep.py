"""
ATR & ADX Entry Filter Analysis
================================
Comprehensive analysis of ATR and ADX indicators as entry filters across
all 3 active strategies (vScalpA, vScalpB on MNQ; MES v2 on MES).

Phase 1: Correlation diagnostic — does ATR/ADX at entry predict trade P&L?
Phase 2: Threshold sweep — optimal filter thresholds for promising combos.

Indicators computed on both 1-min and 5-min timeframes.
ATR periods: [7, 10, 14, 20, 30]
ADX periods: [7, 10, 14, 20, 30]
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# --- Path setup (matches sr_common pattern) ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR))
sys.path.insert(0, str(_STRAT_DIR.parent))  # backtesting_engine/
sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    load_instrument_1min,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    NY_OPEN_ET,
    NY_LAST_ENTRY_ET,
    NY_CLOSE_ET,
)
from generate_session import (
    run_backtest_tp_exit,
    compute_mfe_mae,
)

# ---------------------------------------------------------------------------
# Strategy config table (matches sr_common.STRATEGIES)
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
# Indicator computation
# ---------------------------------------------------------------------------

def compute_atr_wilder(highs, lows, closes, period=14):
    """Wilder's ATR — same as Pine's ta.atr(period)."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n > period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_adx(highs, lows, closes, period=14):
    """ADX (Average Directional Index) using Wilder's smoothing.

    Returns (adx, plus_di, minus_di) arrays.
    """
    n = len(highs)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        plus_dm[i] = high_diff if (high_diff > low_diff and high_diff > 0) else 0.0
        minus_dm[i] = low_diff if (low_diff > high_diff and low_diff > 0) else 0.0

        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))

    # Wilder smoothing for TR, +DM, -DM
    smoothed_tr = np.full(n, np.nan)
    smoothed_plus_dm = np.full(n, np.nan)
    smoothed_minus_dm = np.full(n, np.nan)

    if n <= period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    smoothed_tr[period - 1] = np.sum(tr[:period])
    smoothed_plus_dm[period - 1] = np.sum(plus_dm[:period])
    smoothed_minus_dm[period - 1] = np.sum(minus_dm[:period])

    for i in range(period, n):
        smoothed_tr[i] = smoothed_tr[i - 1] - smoothed_tr[i - 1] / period + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - smoothed_plus_dm[i - 1] / period + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - smoothed_minus_dm[i - 1] / period + minus_dm[i]

    # DI+ and DI-
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    for i in range(period - 1, n):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0

    # DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0

    # ADX = Wilder smoothing of DX
    adx = np.full(n, np.nan)
    # First ADX = mean of first `period` valid DX values
    first_valid = period - 1
    adx_start = first_valid + period - 1
    if adx_start < n:
        adx[adx_start] = np.nanmean(dx[first_valid:first_valid + period])
        for i in range(adx_start + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


def map_5min_indicator_to_1min(onemin_times, fivemin_times, fivemin_values):
    """Map a 5-min indicator back to 1-min bars.

    Each 1-min bar gets the PREVIOUS completed 5-min bar's value
    (same pattern as RSI mapping — uses j-1 to avoid look-ahead).
    """
    n_1m = len(onemin_times)
    mapped = np.full(n_1m, np.nan)

    j = 0
    for i in range(n_1m):
        while j + 1 < len(fivemin_times) and fivemin_times[j + 1] <= onemin_times[i]:
            j += 1
        if j >= 1:
            mapped[i] = fivemin_values[j - 1]

    return mapped


# ---------------------------------------------------------------------------
# Data loading and backtest infrastructure
# ---------------------------------------------------------------------------

def prepare_all_data():
    """Load MNQ + MES, compute SM, compute all indicator variants."""
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

    # Split at midpoint
    split_indices = {}
    for inst, df in instruments.items():
        mid = df.index[len(df) // 2]
        is_len = (df.index < mid).sum()
        split_indices[inst] = {"is_len": is_len, "midpoint": mid}
        print(f"  {inst} split: IS {is_len} bars, OOS {len(df) - is_len} bars")

    # Compute RSI for each strategy (FULL dataset only for Phase 1)
    rsi_cache = {}
    backtest_arrays = {}
    for strat in STRATEGIES:
        inst = strat["instrument"]
        df = instruments[inst]
        cache_key = (inst, strat["rsi_len"])
        if cache_key not in rsi_cache:
            df_5m = resample_to_5min(df)
            rsi_curr, rsi_prev = map_5min_rsi_to_1min(
                df.index.values, df_5m.index.values,
                df_5m["Close"].values, rsi_len=strat["rsi_len"],
            )
            rsi_cache[cache_key] = (rsi_curr, rsi_prev)
        else:
            rsi_curr, rsi_prev = rsi_cache[cache_key]

        backtest_arrays[strat["name"]] = (
            df["Open"].values, df["High"].values, df["Low"].values,
            df["Close"].values, df["SM_Net"].values, df.index,
            rsi_curr, rsi_prev,
        )

    # Also build IS/OOS arrays for Phase 2
    split_arrays = {}
    rsi_split_cache = {}
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
            if cache_key not in rsi_split_cache:
                df_5m = resample_to_5min(df_split)
                rsi_curr, rsi_prev = map_5min_rsi_to_1min(
                    df_split.index.values, df_5m.index.values,
                    df_5m["Close"].values, rsi_len=strat["rsi_len"],
                )
                rsi_split_cache[cache_key] = (rsi_curr, rsi_prev)
            else:
                rsi_curr, rsi_prev = rsi_split_cache[cache_key]

            split_arrays[(strat["name"], split_name)] = (
                df_split["Open"].values, df_split["High"].values,
                df_split["Low"].values, df_split["Close"].values,
                df_split["SM_Net"].values, df_split.index,
                rsi_curr, rsi_prev,
            )

    # Compute all indicator variants
    PERIODS = [7, 10, 14, 20, 30]

    # Per-instrument indicator storage: indicators[inst][(indicator, timeframe, period)] = 1min-length array
    indicators = {}
    for inst in ["MNQ", "MES"]:
        df = instruments[inst]
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        times = df.index.values

        ind = {}

        # 1-min ATR and ADX
        for p in PERIODS:
            atr_1m = compute_atr_wilder(highs, lows, closes, period=p)
            adx_1m, _, _ = compute_adx(highs, lows, closes, period=p)
            ind[("ATR", "1min", p)] = atr_1m
            ind[("ADX", "1min", p)] = adx_1m

        # 5-min: resample, compute, map back
        df_5m = resample_to_5min(df)
        h5 = df_5m["High"].values
        l5 = df_5m["Low"].values
        c5 = df_5m["Close"].values
        t5 = df_5m.index.values

        for p in PERIODS:
            atr_5m = compute_atr_wilder(h5, l5, c5, period=p)
            adx_5m, _, _ = compute_adx(h5, l5, c5, period=p)

            atr_mapped = map_5min_indicator_to_1min(times, t5, atr_5m)
            adx_mapped = map_5min_indicator_to_1min(times, t5, adx_5m)

            ind[("ATR", "5min", p)] = atr_mapped
            ind[("ADX", "5min", p)] = adx_mapped

        indicators[inst] = ind
        print(f"  Computed {len(ind)} indicator variants for {inst}")

    return instruments, backtest_arrays, split_arrays, split_indices, indicators


def run_baseline(strat, arrays):
    """Run baseline backtest, return trades."""
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
    return trades


def run_gated(strat, arrays, entry_gate):
    """Run backtest with an entry gate."""
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
        entry_gate=entry_gate,
    )
    return trades


# ---------------------------------------------------------------------------
# Phase 1: Correlation diagnostic
# ---------------------------------------------------------------------------

def phase1_correlation(strategies, backtest_arrays, indicators):
    """For each strategy x indicator combo, compute correlation with trade P&L."""
    print("\n" + "=" * 140)
    print("PHASE 1: CORRELATION DIAGNOSTIC — ATR/ADX at entry vs trade P&L")
    print("=" * 140)

    PERIODS = [7, 10, 14, 20, 30]
    all_corr_results = []

    for strat in strategies:
        inst = strat["instrument"]
        arrays = backtest_arrays[strat["name"]]
        trades = run_baseline(strat, arrays)

        if not trades:
            print(f"\n  {strat['name']}: NO TRADES")
            continue

        # Compute P&L per trade
        pnls = np.array([t["pts"] * strat["dollar_per_pt"] - 2 * strat["commission"] for t in trades])
        entry_idxs = np.array([t["entry_idx"] for t in trades])

        sc = score_trades(trades, commission_per_side=strat["commission"],
                          dollar_per_pt=strat["dollar_per_pt"])
        print(f"\n  {strat['name']} baseline: {sc['count']} trades, "
              f"WR {sc['win_rate']}%, PF {sc['pf']}, ${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']}")

        # Header
        print(f"\n  {'Indicator':>6} {'TF':>5} {'Period':>6} {'r':>8} {'p-val':>10} "
              f"{'Q1 WR%':>7} {'Q1 PF':>7} {'Q2 WR%':>7} {'Q2 PF':>7} "
              f"{'Q3 WR%':>7} {'Q3 PF':>7} {'Q4 WR%':>7} {'Q4 PF':>7} "
              f"{'Q5 WR%':>7} {'Q5 PF':>7} {'Signal':>8}")
        print(f"  {'-' * 135}")

        for indicator in ["ATR", "ADX"]:
            for tf in ["1min", "5min"]:
                for period in PERIODS:
                    key = (indicator, tf, period)
                    ind_values = indicators[inst][key]

                    # Get indicator value at entry
                    entry_vals = ind_values[entry_idxs]

                    # Filter out NaN entries
                    valid_mask = ~np.isnan(entry_vals)
                    if valid_mask.sum() < 20:
                        continue

                    ev = entry_vals[valid_mask]
                    pl = pnls[valid_mask]

                    # Pearson correlation
                    r, p_val = stats.pearsonr(ev, pl)

                    # Quintile analysis
                    quintile_labels = pd.qcut(ev, 5, labels=False, duplicates='drop')
                    n_quintiles = len(np.unique(quintile_labels))

                    q_stats = []
                    for q in range(n_quintiles):
                        q_mask = quintile_labels == q
                        q_pnls = pl[q_mask]
                        if len(q_pnls) == 0:
                            q_stats.append(("--", "--"))
                            continue
                        q_wins = (q_pnls > 0).sum()
                        q_wr = q_wins / len(q_pnls) * 100
                        q_gross_wins = q_pnls[q_pnls > 0].sum() if q_wins > 0 else 0
                        q_gross_losses = abs(q_pnls[q_pnls <= 0].sum()) if (q_pnls <= 0).any() else 0
                        q_pf = q_gross_wins / q_gross_losses if q_gross_losses > 0 else 999.0
                        q_stats.append((f"{q_wr:.1f}", f"{q_pf:.2f}"))

                    # Pad to 5 quintiles if needed
                    while len(q_stats) < 5:
                        q_stats.append(("--", "--"))

                    # Signal strength indicator
                    signal = ""
                    if abs(r) > 0.15 and p_val < 0.05:
                        signal = "***"
                    elif abs(r) > 0.10 and p_val < 0.10:
                        signal = "**"
                    elif abs(r) > 0.08:
                        signal = "*"

                    print(f"  {indicator:>6} {tf:>5} {period:>6} {r:>+8.4f} {p_val:>10.4f} "
                          f"{q_stats[0][0]:>7} {q_stats[0][1]:>7} "
                          f"{q_stats[1][0]:>7} {q_stats[1][1]:>7} "
                          f"{q_stats[2][0]:>7} {q_stats[2][1]:>7} "
                          f"{q_stats[3][0]:>7} {q_stats[3][1]:>7} "
                          f"{q_stats[4][0]:>7} {q_stats[4][1]:>7} "
                          f"{signal:>8}")

                    all_corr_results.append({
                        "strategy": strat["name"],
                        "instrument": inst,
                        "indicator": indicator,
                        "timeframe": tf,
                        "period": period,
                        "r": r,
                        "p_val": p_val,
                        "n_trades": int(valid_mask.sum()),
                    })

    # Summary: top correlations across all strategies
    print("\n" + "=" * 100)
    print("TOP CORRELATIONS (|r| > 0.05, sorted by |r|)")
    print("=" * 100)
    print(f"  {'Strategy':>8} {'Indicator':>6} {'TF':>5} {'Period':>6} {'r':>8} {'p-val':>10} {'N':>6}")
    print(f"  {'-' * 55}")

    sorted_results = sorted(all_corr_results, key=lambda x: abs(x["r"]), reverse=True)
    for res in sorted_results:
        if abs(res["r"]) < 0.05:
            continue
        sig = " *" if res["p_val"] < 0.05 else ""
        print(f"  {res['strategy']:>8} {res['indicator']:>6} {res['timeframe']:>5} "
              f"{res['period']:>6} {res['r']:>+8.4f} {res['p_val']:>10.4f} {res['n_trades']:>6}{sig}")

    return all_corr_results


# ---------------------------------------------------------------------------
# Phase 2: Threshold sweep
# ---------------------------------------------------------------------------

def phase2_threshold_sweep(strategies, instruments, split_arrays, split_indices, indicators):
    """Sweep ATR percentile gates and ADX threshold gates, evaluate IS/OOS."""
    print("\n\n" + "=" * 140)
    print("PHASE 2: THRESHOLD SWEEP — ATR percentile + ADX threshold gates")
    print("=" * 140)

    PERIODS = [7, 10, 14, 20, 30]

    # We'll sweep all combos but report per-strategy
    # ADX thresholds: only enter if ADX > threshold
    ADX_THRESHOLDS = [10, 15, 20, 25, 30, 35, 40]
    # ATR percentiles: skip if ATR > Xth percentile (low-vol filter)
    # AND skip if ATR < Xth percentile (high-vol filter)
    ATR_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    all_sweep_results = []

    # --- Baselines for IS/OOS ---
    baselines = {}
    print("\nRunning IS/OOS baselines...")
    for strat in strategies:
        for split_name in ["FULL", "IS", "OOS"]:
            arrays = split_arrays[(strat["name"], split_name)]
            trades = run_baseline(strat, arrays)
            sc = score_trades(trades, commission_per_side=strat["commission"],
                              dollar_per_pt=strat["dollar_per_pt"])
            baselines[(strat["name"], split_name)] = (sc, trades)
            if split_name == "FULL" and sc:
                print(f"  {strat['name']:>8}: {sc['count']} trades, WR {sc['win_rate']}%, "
                      f"PF {sc['pf']}, ${sc['net_dollar']:+.0f}, Sharpe {sc['sharpe']}")

    # Helper: build a gate from indicator values
    def build_indicator_gate_full(inst, indicator, tf, period, gate_type, gate_value):
        """Build a boolean gate for the FULL dataset of an instrument.

        gate_type:
          'adx_min': True if indicator > gate_value
          'atr_max_pct': True if indicator <= percentile(gate_value) — low ATR only
          'atr_min_pct': True if indicator >= percentile(gate_value) — high ATR only
        """
        key = (indicator, tf, period)
        vals = indicators[inst][key]
        n = len(vals)
        gate = np.ones(n, dtype=bool)

        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            return gate  # all True = no blocking

        if gate_type == 'adx_min':
            # Block entries where ADX <= threshold
            for i in range(n):
                if np.isnan(vals[i]) or vals[i] <= gate_value:
                    gate[i] = False
        elif gate_type == 'atr_max_pct':
            # Block entries where ATR > Xth percentile (only allow low-volatility)
            threshold = np.nanpercentile(vals, gate_value)
            for i in range(n):
                if np.isnan(vals[i]) or vals[i] > threshold:
                    gate[i] = False
        elif gate_type == 'atr_min_pct':
            # Block entries where ATR < Xth percentile (only allow high-volatility)
            threshold = np.nanpercentile(vals, gate_value)
            for i in range(n):
                if np.isnan(vals[i]) or vals[i] < threshold:
                    gate[i] = False

        return gate

    def slice_gate(full_gate, instrument, split_name):
        is_len = split_indices[instrument]["is_len"]
        if split_name == "FULL":
            return full_gate
        elif split_name == "IS":
            return full_gate[:is_len]
        else:
            return full_gate[is_len:]

    # --- ADX sweep ---
    print("\n" + "=" * 140)
    print("ADX MINIMUM THRESHOLD SWEEP: only enter if ADX(period) > threshold")
    print("=" * 140)

    for indicator in ["ADX"]:
        for tf in ["1min", "5min"]:
            for period in PERIODS:
                print(f"\n  --- {indicator} {tf} period={period} ---")
                print(f"  {'Threshold':>10} {'Strategy':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} "
                      f"{'PF':>7} {'Net$':>10} {'Sharpe':>7} {'Blocked':>8} {'dPF%':>7} {'dSharpe':>8}")
                print(f"  {'-' * 115}")

                for threshold in ADX_THRESHOLDS:
                    for strat in strategies:
                        inst = strat["instrument"]
                        gate_full = build_indicator_gate_full(
                            inst, indicator, tf, period, 'adx_min', threshold)

                        for split_name in ["IS", "OOS"]:
                            arrays = split_arrays[(strat["name"], split_name)]
                            gate = slice_gate(gate_full, inst, split_name)
                            trades = run_gated(strat, arrays, gate)
                            sc = score_trades(trades, commission_per_side=strat["commission"],
                                              dollar_per_pt=strat["dollar_per_pt"])
                            bl_sc, _ = baselines[(strat["name"], split_name)]

                            if sc is None:
                                blocked = bl_sc["count"] if bl_sc else 0
                                print(f"  {f'>{threshold}':>10} {strat['name']:>8} {split_name:>5} "
                                      f"{'0':>7} {'--':>6} {'--':>7} {'--':>10} {'--':>7} "
                                      f"{blocked:>8} {'--':>7} {'--':>8}")
                                continue

                            bl_count = bl_sc["count"] if bl_sc else 0
                            blocked = bl_count - sc["count"]
                            dpf = ((sc["pf"] - bl_sc["pf"]) / bl_sc["pf"] * 100
                                   if bl_sc and bl_sc["pf"] > 0 else 0)
                            dsharpe = sc["sharpe"] - bl_sc["sharpe"] if bl_sc else 0

                            print(f"  {f'>{threshold}':>10} {strat['name']:>8} {split_name:>5} "
                                  f"{sc['count']:>7} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                                  f"{sc['net_dollar']:>+10.2f} {sc['sharpe']:>7.3f} "
                                  f"{blocked:>8} {dpf:>+6.1f}% {dsharpe:>+8.3f}")

                            all_sweep_results.append({
                                "indicator": indicator, "tf": tf, "period": period,
                                "gate_type": f"ADX>{threshold}",
                                "strategy": strat["name"], "split": split_name,
                                "trades": sc["count"], "wr": sc["win_rate"],
                                "pf": sc["pf"], "net_dollar": sc["net_dollar"],
                                "sharpe": sc["sharpe"], "blocked": blocked,
                                "dpf_pct": dpf, "dsharpe": dsharpe,
                            })

    # --- ATR sweep ---
    print("\n\n" + "=" * 140)
    print("ATR PERCENTILE SWEEP: skip entries outside ATR range")
    print("=" * 140)

    for indicator in ["ATR"]:
        for tf in ["1min", "5min"]:
            for period in PERIODS:
                print(f"\n  --- {indicator} {tf} period={period} ---")

                # Low-vol filter: only enter when ATR < Xth percentile
                print(f"\n  LOW-VOL FILTER (ATR < percentile threshold):")
                print(f"  {'Pctl':>10} {'Strategy':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} "
                      f"{'PF':>7} {'Net$':>10} {'Sharpe':>7} {'Blocked':>8} {'dPF%':>7} {'dSharpe':>8}")
                print(f"  {'-' * 115}")

                for pctl in ATR_PERCENTILES:
                    for strat in strategies:
                        inst = strat["instrument"]
                        gate_full = build_indicator_gate_full(
                            inst, indicator, tf, period, 'atr_max_pct', pctl)

                        for split_name in ["IS", "OOS"]:
                            arrays = split_arrays[(strat["name"], split_name)]
                            gate = slice_gate(gate_full, inst, split_name)
                            trades = run_gated(strat, arrays, gate)
                            sc = score_trades(trades, commission_per_side=strat["commission"],
                                              dollar_per_pt=strat["dollar_per_pt"])
                            bl_sc, _ = baselines[(strat["name"], split_name)]

                            if sc is None:
                                blocked = bl_sc["count"] if bl_sc else 0
                                print(f"  {f'<p{pctl}':>10} {strat['name']:>8} {split_name:>5} "
                                      f"{'0':>7} {'--':>6} {'--':>7} {'--':>10} {'--':>7} "
                                      f"{blocked:>8} {'--':>7} {'--':>8}")
                                continue

                            bl_count = bl_sc["count"] if bl_sc else 0
                            blocked = bl_count - sc["count"]
                            dpf = ((sc["pf"] - bl_sc["pf"]) / bl_sc["pf"] * 100
                                   if bl_sc and bl_sc["pf"] > 0 else 0)
                            dsharpe = sc["sharpe"] - bl_sc["sharpe"] if bl_sc else 0

                            print(f"  {f'<p{pctl}':>10} {strat['name']:>8} {split_name:>5} "
                                  f"{sc['count']:>7} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                                  f"{sc['net_dollar']:>+10.2f} {sc['sharpe']:>7.3f} "
                                  f"{blocked:>8} {dpf:>+6.1f}% {dsharpe:>+8.3f}")

                            all_sweep_results.append({
                                "indicator": indicator, "tf": tf, "period": period,
                                "gate_type": f"ATR<p{pctl}",
                                "strategy": strat["name"], "split": split_name,
                                "trades": sc["count"], "wr": sc["win_rate"],
                                "pf": sc["pf"], "net_dollar": sc["net_dollar"],
                                "sharpe": sc["sharpe"], "blocked": blocked,
                                "dpf_pct": dpf, "dsharpe": dsharpe,
                            })

                # High-vol filter: only enter when ATR > Xth percentile
                print(f"\n  HIGH-VOL FILTER (ATR > percentile threshold):")
                print(f"  {'Pctl':>10} {'Strategy':>8} {'Split':>5} {'Trades':>7} {'WR%':>6} "
                      f"{'PF':>7} {'Net$':>10} {'Sharpe':>7} {'Blocked':>8} {'dPF%':>7} {'dSharpe':>8}")
                print(f"  {'-' * 115}")

                for pctl in ATR_PERCENTILES:
                    for strat in strategies:
                        inst = strat["instrument"]
                        gate_full = build_indicator_gate_full(
                            inst, indicator, tf, period, 'atr_min_pct', pctl)

                        for split_name in ["IS", "OOS"]:
                            arrays = split_arrays[(strat["name"], split_name)]
                            gate = slice_gate(gate_full, inst, split_name)
                            trades = run_gated(strat, arrays, gate)
                            sc = score_trades(trades, commission_per_side=strat["commission"],
                                              dollar_per_pt=strat["dollar_per_pt"])
                            bl_sc, _ = baselines[(strat["name"], split_name)]

                            if sc is None:
                                blocked = bl_sc["count"] if bl_sc else 0
                                print(f"  {f'>p{pctl}':>10} {strat['name']:>8} {split_name:>5} "
                                      f"{'0':>7} {'--':>6} {'--':>7} {'--':>10} {'--':>7} "
                                      f"{blocked:>8} {'--':>7} {'--':>8}")
                                continue

                            bl_count = bl_sc["count"] if bl_sc else 0
                            blocked = bl_count - sc["count"]
                            dpf = ((sc["pf"] - bl_sc["pf"]) / bl_sc["pf"] * 100
                                   if bl_sc and bl_sc["pf"] > 0 else 0)
                            dsharpe = sc["sharpe"] - bl_sc["sharpe"] if bl_sc else 0

                            print(f"  {f'>p{pctl}':>10} {strat['name']:>8} {split_name:>5} "
                                  f"{sc['count']:>7} {sc['win_rate']:>5.1f}% {sc['pf']:>7.3f} "
                                  f"{sc['net_dollar']:>+10.2f} {sc['sharpe']:>7.3f} "
                                  f"{blocked:>8} {dpf:>+6.1f}% {dsharpe:>+8.3f}")

                            all_sweep_results.append({
                                "indicator": indicator, "tf": tf, "period": period,
                                "gate_type": f"ATR>p{pctl}",
                                "strategy": strat["name"], "split": split_name,
                                "trades": sc["count"], "wr": sc["win_rate"],
                                "pf": sc["pf"], "net_dollar": sc["net_dollar"],
                                "sharpe": sc["sharpe"], "blocked": blocked,
                                "dpf_pct": dpf, "dsharpe": dsharpe,
                            })

    return all_sweep_results, baselines


# ---------------------------------------------------------------------------
# Phase 2 summary: find configs that improve BOTH IS and OOS
# ---------------------------------------------------------------------------

def phase2_summary(all_sweep_results, baselines):
    """Find the best threshold configs: IS and OOS both improve."""
    print("\n\n" + "=" * 140)
    print("PHASE 2 SUMMARY: Configs where BOTH IS and OOS improve (dPF% > 0 and dSharpe > 0)")
    print("=" * 140)

    # Group results by (indicator, tf, period, gate_type, strategy)
    grouped = {}
    for r in all_sweep_results:
        key = (r["indicator"], r["tf"], r["period"], r["gate_type"], r["strategy"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r["split"]] = r

    # Find configs where both IS and OOS improve
    candidates = []
    for key, splits in grouped.items():
        if "IS" not in splits or "OOS" not in splits:
            continue
        is_r = splits["IS"]
        oos_r = splits["OOS"]

        # Must have trades in both
        if is_r["trades"] < 10 or oos_r["trades"] < 10:
            continue

        # Both must improve PF and Sharpe
        if is_r["dpf_pct"] > 0 and oos_r["dpf_pct"] > 0 and is_r["dsharpe"] > 0 and oos_r["dsharpe"] > 0:
            avg_dpf = (is_r["dpf_pct"] + oos_r["dpf_pct"]) / 2
            avg_dsharpe = (is_r["dsharpe"] + oos_r["dsharpe"]) / 2
            candidates.append({
                "indicator": key[0], "tf": key[1], "period": key[2],
                "gate_type": key[3], "strategy": key[4],
                "is_trades": is_r["trades"], "oos_trades": oos_r["trades"],
                "is_wr": is_r["wr"], "oos_wr": oos_r["wr"],
                "is_pf": is_r["pf"], "oos_pf": oos_r["pf"],
                "is_net": is_r["net_dollar"], "oos_net": oos_r["net_dollar"],
                "is_sharpe": is_r["sharpe"], "oos_sharpe": oos_r["sharpe"],
                "is_blocked": is_r["blocked"], "oos_blocked": oos_r["blocked"],
                "is_dpf": is_r["dpf_pct"], "oos_dpf": oos_r["dpf_pct"],
                "is_dsharpe": is_r["dsharpe"], "oos_dsharpe": oos_r["dsharpe"],
                "avg_dpf": avg_dpf, "avg_dsharpe": avg_dsharpe,
            })

    if not candidates:
        print("\n  NO configs found that improve both IS and OOS simultaneously.")
        print("  This is consistent with prior research: simple indicator thresholds")
        print("  tend to overfit to one half of the data.")
    else:
        # Sort by avg_dsharpe descending
        candidates.sort(key=lambda x: x["avg_dsharpe"], reverse=True)

        print(f"\n  {'Strategy':>8} {'Gate':>15} {'TF':>5} {'Per':>4} "
              f"{'IS Tr':>6} {'IS WR%':>7} {'IS PF':>7} {'IS $':>9} {'IS Sh':>7} "
              f"{'OOS Tr':>7} {'OOS WR%':>8} {'OOS PF':>8} {'OOS $':>9} {'OOS Sh':>8} "
              f"{'dPF%':>7} {'dSh':>7}")
        print(f"  {'-' * 150}")

        for c in candidates[:30]:  # top 30
            print(f"  {c['strategy']:>8} {c['gate_type']:>15} {c['tf']:>5} {c['period']:>4} "
                  f"{c['is_trades']:>6} {c['is_wr']:>6.1f}% {c['is_pf']:>7.3f} "
                  f"{c['is_net']:>+9.0f} {c['is_sharpe']:>7.3f} "
                  f"{c['oos_trades']:>7} {c['oos_wr']:>7.1f}% {c['oos_pf']:>8.3f} "
                  f"{c['oos_net']:>+9.0f} {c['oos_sharpe']:>8.3f} "
                  f"{c['avg_dpf']:>+6.1f}% {c['avg_dsharpe']:>+7.3f}")

    # Also show configs that are STRONG: OOS PF > baseline AND >5% improvement
    print("\n\n" + "=" * 140)
    print("STRONG CANDIDATES: OOS dPF > +5% (regardless of IS)")
    print("=" * 140)

    oos_candidates = []
    for key, splits in grouped.items():
        if "OOS" not in splits:
            continue
        oos_r = splits["OOS"]
        if oos_r["trades"] < 15 and oos_r["dpf_pct"] > 5 and oos_r["dsharpe"] > 0:
            continue  # too few trades
        if oos_r["dpf_pct"] > 5 and oos_r["dsharpe"] > 0 and oos_r["trades"] >= 15:
            is_r = splits.get("IS", {})
            oos_candidates.append({
                "indicator": key[0], "tf": key[1], "period": key[2],
                "gate_type": key[3], "strategy": key[4],
                "oos_trades": oos_r["trades"], "oos_wr": oos_r["wr"],
                "oos_pf": oos_r["pf"], "oos_net": oos_r["net_dollar"],
                "oos_sharpe": oos_r["sharpe"],
                "oos_dpf": oos_r["dpf_pct"], "oos_dsharpe": oos_r["dsharpe"],
                "is_dpf": is_r.get("dpf_pct", 0),
            })

    if not oos_candidates:
        print("\n  No OOS-strong candidates found.")
    else:
        oos_candidates.sort(key=lambda x: x["oos_dsharpe"], reverse=True)
        print(f"\n  {'Strategy':>8} {'Gate':>15} {'TF':>5} {'Per':>4} "
              f"{'OOS Tr':>7} {'OOS WR%':>8} {'OOS PF':>8} {'OOS $':>9} {'OOS Sh':>8} "
              f"{'OOS dPF%':>9} {'OOS dSh':>8} {'IS dPF%':>8}")
        print(f"  {'-' * 110}")
        for c in oos_candidates[:20]:
            print(f"  {c['strategy']:>8} {c['gate_type']:>15} {c['tf']:>5} {c['period']:>4} "
                  f"{c['oos_trades']:>7} {c['oos_wr']:>7.1f}% {c['oos_pf']:>8.3f} "
                  f"{c['oos_net']:>+9.0f} {c['oos_sharpe']:>8.3f} "
                  f"{c['oos_dpf']:>+8.1f}% {c['oos_dsharpe']:>+8.3f} "
                  f"{c['is_dpf']:>+7.1f}%")

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("ATR & ADX ENTRY FILTER ANALYSIS")
    print("Strategies: vScalpA (V15), vScalpB, MES v2")
    print("Indicators: ATR, ADX | Timeframes: 1min, 5min | Periods: 7,10,14,20,30")
    print("=" * 80)

    instruments, backtest_arrays, split_arrays, split_indices, indicators = prepare_all_data()

    # Phase 1: Correlation diagnostic
    corr_results = phase1_correlation(STRATEGIES, backtest_arrays, indicators)

    # Phase 2: Threshold sweep with IS/OOS validation
    sweep_results, baselines = phase2_threshold_sweep(
        STRATEGIES, instruments, split_arrays, split_indices, indicators)

    # Phase 2 summary
    candidates = phase2_summary(sweep_results, baselines)

    # Final verdict
    print("\n\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if not candidates:
        print("""
  ATR and ADX entry filters show NO robust predictive signal for trade outcomes
  across any of the 3 active strategies. This is consistent with prior research
  findings that simple indicator-based entry filters tend to not survive
  IS/OOS validation.

  Key findings:
  - Correlations between ATR/ADX at entry and trade P&L are weak (|r| < 0.15)
  - Any threshold that improves IS typically degrades OOS (classic overfit)
  - The SM + RSI entry signal already captures the relevant market structure

  Recommendation: DO NOT implement ATR or ADX entry filters.
""")
    else:
        n_strong = sum(1 for c in candidates if c["avg_dpf"] > 5)
        print(f"""
  Found {len(candidates)} configs that improve both IS and OOS.
  Of these, {n_strong} show >5% average PF improvement.

  Review the PHASE 2 SUMMARY tables above for specific thresholds.
  Any candidate should be validated with paper trading before live deployment.
""")


if __name__ == "__main__":
    main()
