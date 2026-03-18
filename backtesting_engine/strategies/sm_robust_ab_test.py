"""
SM Robust A/B Test — Current vs Robust Cluster Center
======================================================
Compares current production SM params vs the robust cluster center
identified by the 3-year sweep, across all 3 years for ALL strategies.

MNQ:
  Current:  SM(10/12/200/100)
  Robust:   SM(12/12/200/80)

MES (exploratory — proportional scaling, not sweep-validated):
  Current:  SM(20/12/400/255)
  Robust:   SM(24/12/400/160)  (index +20%, ema -37%)

Strategies tested:
  vScalpA   — RSI(8/60/40), CD=20, TP=7, SL=40, cutoff 13:00, SM_T=0.0
  vScalpB   — RSI(8/55/45), CD=20, TP=3, SL=10, SM_T=0.25
  vScalpC   — RSI(8/60/40), CD=20, TP1=7, TP2=25, SL=40, cutoff 13:00, SM_T=0.0, SL->BE
  MES v2    — RSI(12/55/45), CD=25, TP=20, SL=35, cutoff 14:15, BE_TIME=75, SM_T=0.0

Usage:
    cd backtesting_engine && python3 strategies/sm_robust_ab_test.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

# --- Path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "live_trading"))

from v10_test_common import (
    compute_smart_money,
    compute_rsi,
    compute_et_minutes,
    map_5min_rsi_to_1min,
    resample_to_5min,
    score_trades,
    fmt_score,
    NY_OPEN_ET,
    NY_CLOSE_ET,
)

from generate_session import run_backtest_tp_exit

from vscalpc_partial_exit_sweep import (
    run_backtest_partial_exit,
    score_partial_trades,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"

MNQ_DOLLAR_PER_PT = 2.0
MNQ_COMMISSION = 0.52
MES_DOLLAR_PER_PT = 5.0
MES_COMMISSION = 1.25

# SM configs to compare
MNQ_CONFIGS = {
    "Current": {"index": 10, "flow": 12, "norm": 200, "ema": 100},
    "Robust":  {"index": 12, "flow": 12, "norm": 200, "ema": 80},
}

MES_CONFIGS = {
    "Current": {"index": 20, "flow": 12, "norm": 400, "ema": 255},
    "Robust":  {"index": 24, "flow": 12, "norm": 400, "ema": 160},
}

# Strategy params (all fixed — only SM changes)
STRATEGIES = {
    "vScalpA": {
        "instrument": "MNQ",
        "rsi_len": 8, "rsi_buy": 60, "rsi_sell": 40,
        "sm_threshold": 0.0, "cooldown": 20,
        "max_loss_pts": 40, "tp_pts": 7,
        "entry_end_et": 13 * 60,
        "partial": False,
    },
    "vScalpB": {
        "instrument": "MNQ",
        "rsi_len": 8, "rsi_buy": 55, "rsi_sell": 45,
        "sm_threshold": 0.25, "cooldown": 20,
        "max_loss_pts": 10, "tp_pts": 3,
        "entry_end_et": 15 * 60 + 45,  # no cutoff
        "partial": False,
    },
    "vScalpC": {
        "instrument": "MNQ",
        "rsi_len": 8, "rsi_buy": 60, "rsi_sell": 40,
        "sm_threshold": 0.0, "cooldown": 20,
        "sl_pts": 40, "tp1_pts": 7, "tp2_pts": 25,
        "entry_end_et": 13 * 60,
        "partial": True,
    },
    "MES v2": {
        "instrument": "MES",
        "rsi_len": 12, "rsi_buy": 55, "rsi_sell": 45,
        "sm_threshold": 0.0, "cooldown": 25,
        "max_loss_pts": 35, "tp_pts": 20,
        "entry_end_et": 14 * 60 + 15,
        "eod_et": 16 * 60,
        "breakeven_after_bars": 75,
        "partial": False,
    },
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_year_data(filepath):
    """Load a year of data from a CSV file."""
    df_raw = pd.read_csv(filepath)
    result = pd.DataFrame()
    result['Time'] = pd.to_datetime(df_raw['time'].astype(int), unit='s')
    result['Open'] = pd.to_numeric(df_raw['open'], errors='coerce')
    result['High'] = pd.to_numeric(df_raw['high'], errors='coerce')
    result['Low'] = pd.to_numeric(df_raw['low'], errors='coerce')
    result['Close'] = pd.to_numeric(df_raw['close'], errors='coerce')
    result['Volume'] = pd.to_numeric(df_raw['Volume'], errors='coerce').fillna(0)
    result = result.set_index('Time')
    result = result[~result.index.duplicated(keep='first')]
    result = result.sort_index()
    return result


def load_all_years(instrument):
    """Load all 3 years of data for an instrument."""
    print(f"\n  Loading {instrument} data...")

    # Year 1: Feb 2023 - Feb 2024
    y1_path = DATA_DIR / f"prior2_databento_{instrument}_1min_2023-02-17_to_2024-02-16.csv"
    df_y1 = load_year_data(y1_path)
    print(f"    Year 1 (Feb23-24): {len(df_y1):,} bars, "
          f"Price: {df_y1['Close'].min():.0f}-{df_y1['Close'].max():.0f}")

    # Year 2: Feb 2024 - Feb 2025
    y2_path = DATA_DIR / f"prior_databento_{instrument}_1min_2024-02-17_to_2025-02-16.csv"
    df_y2 = load_year_data(y2_path)
    print(f"    Year 2 (Feb24-25): {len(df_y2):,} bars, "
          f"Price: {df_y2['Close'].min():.0f}-{df_y2['Close'].max():.0f}")

    # Year 3: Feb 2025 - Mar 2026 (dev period)
    db_files = sorted(DATA_DIR.glob(f"databento_{instrument}_1min_*.csv"))
    dfs = []
    for f in db_files:
        df_tmp = load_year_data(f)
        dfs.append(df_tmp)
    df_y3 = pd.concat(dfs)
    df_y3 = df_y3[~df_y3.index.duplicated(keep='last')]
    df_y3 = df_y3.sort_index()
    print(f"    Year 3 (Feb25-26): {len(df_y3):,} bars, "
          f"Price: {df_y3['Close'].min():.0f}-{df_y3['Close'].max():.0f}")

    return {"y1": df_y1, "y2": df_y2, "y3": df_y3}


# ---------------------------------------------------------------------------
# Precompute SM + RSI for a given SM config + year
# ---------------------------------------------------------------------------

def precompute(df, sm_cfg, rsi_len):
    """Compute SM and 5-min RSI mapping for one year + one SM config."""
    closes = df['Close'].values
    volumes = df['Volume'].values

    sm = compute_smart_money(
        closes, volumes,
        index_period=sm_cfg["index"], flow_period=sm_cfg["flow"],
        norm_period=sm_cfg["norm"], ema_len=sm_cfg["ema"],
    )

    df_copy = df.copy()
    df_copy['SM_Net'] = sm
    df_5m = resample_to_5min(df_copy)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        df.index.values, df_5m.index.values,
        df_5m['Close'].values, rsi_len=rsi_len,
    )

    return {
        'opens': df['Open'].values,
        'highs': df['High'].values,
        'lows': df['Low'].values,
        'closes': closes,
        'sm': sm,
        'times': df.index,
        'rsi_curr': rsi_curr,
        'rsi_prev': rsi_prev,
    }


# ---------------------------------------------------------------------------
# Run a single strategy backtest
# ---------------------------------------------------------------------------

def run_single(arrays, strat_params, dollar_per_pt, commission):
    """Run one backtest and return metrics dict."""
    if strat_params["partial"]:
        # vScalpC partial exit
        trades = run_backtest_partial_exit(
            arrays['opens'], arrays['highs'], arrays['lows'],
            arrays['closes'], arrays['sm'], arrays['times'],
            arrays['rsi_curr'], arrays['rsi_prev'],
            rsi_buy=strat_params["rsi_buy"],
            rsi_sell=strat_params["rsi_sell"],
            sm_threshold=strat_params["sm_threshold"],
            cooldown_bars=strat_params["cooldown"],
            sl_pts=strat_params["sl_pts"],
            tp1_pts=strat_params["tp1_pts"],
            tp2_pts=strat_params["tp2_pts"],
            sl_to_be_after_tp1=True,
            be_time_bars=0,
            entry_end_et=strat_params["entry_end_et"],
        )
        sc = score_partial_trades(trades, dollar_per_pt=dollar_per_pt,
                                   commission_per_side=commission)
        if sc is None:
            return {"trades": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "sharpe": 0.0}
        return {
            "trades": sc["count"],
            "wr": sc["win_rate"],
            "pf": sc["pf"],
            "net": sc["net_dollar"],
            "sharpe": sc["sharpe"],
        }
    else:
        # Standard TP exit
        kwargs = dict(
            rsi_buy=strat_params["rsi_buy"],
            rsi_sell=strat_params["rsi_sell"],
            sm_threshold=strat_params["sm_threshold"],
            cooldown_bars=strat_params["cooldown"],
            max_loss_pts=strat_params["max_loss_pts"],
            tp_pts=strat_params["tp_pts"],
            entry_end_et=strat_params["entry_end_et"],
        )
        if "eod_et" in strat_params:
            kwargs["eod_minutes_et"] = strat_params["eod_et"]
        if "breakeven_after_bars" in strat_params:
            kwargs["breakeven_after_bars"] = strat_params["breakeven_after_bars"]

        trades = run_backtest_tp_exit(
            arrays['opens'], arrays['highs'], arrays['lows'],
            arrays['closes'], arrays['sm'], arrays['times'],
            arrays['rsi_curr'], arrays['rsi_prev'],
            **kwargs,
        )
        sc = score_trades(trades, commission_per_side=commission,
                          dollar_per_pt=dollar_per_pt)
        if sc is None:
            return {"trades": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "sharpe": 0.0}
        return {
            "trades": sc["count"],
            "wr": sc["win_rate"],
            "pf": sc["pf"],
            "net": sc["net_dollar"],
            "sharpe": sc["sharpe"],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 110)
    print("SM ROBUST A/B TEST — Current vs Robust Cluster Center")
    print("=" * 110)
    print("\nMNQ: Current SM(10/12/200/100) vs Robust SM(12/12/200/80)")
    print("MES: Current SM(20/12/400/255) vs Robust SM(24/12/400/160) [exploratory]")

    # Load data for both instruments
    mnq_years = load_all_years("MNQ")
    mes_years = load_all_years("MES")

    year_labels = {"y1": "Y1 (Feb23-24)", "y2": "Y2 (Feb24-25)", "y3": "Y3 (Feb25-26)"}
    year_keys = ["y1", "y2", "y3"]

    # Precompute SM + RSI for each instrument x config x year
    print("\n--- Precomputing SM + RSI ---")

    # Cache: (instrument, config_name, year_key, rsi_len) -> arrays
    cache = {}

    for strat_name, sp in STRATEGIES.items():
        instrument = sp["instrument"]
        sm_configs = MNQ_CONFIGS if instrument == "MNQ" else MES_CONFIGS
        years_data = mnq_years if instrument == "MNQ" else mes_years

        for cfg_name, sm_cfg in sm_configs.items():
            for yk in year_keys:
                cache_key = (instrument, cfg_name, yk, sp["rsi_len"])
                if cache_key not in cache:
                    cache[cache_key] = precompute(years_data[yk], sm_cfg, sp["rsi_len"])
                    print(f"  Computed: {instrument} {cfg_name} {yk} RSI_len={sp['rsi_len']}")

    precompute_time = time.time() - t_start
    print(f"\n  Precompute done in {precompute_time:.1f}s")

    # ===================================================================
    # Run all strategies
    # ===================================================================
    all_results = {}  # strat_name -> cfg_name -> year_key -> metrics

    for strat_name, sp in STRATEGIES.items():
        instrument = sp["instrument"]
        sm_configs = MNQ_CONFIGS if instrument == "MNQ" else MES_CONFIGS
        dollar_per_pt = MNQ_DOLLAR_PER_PT if instrument == "MNQ" else MES_DOLLAR_PER_PT
        commission = MNQ_COMMISSION if instrument == "MNQ" else MES_COMMISSION

        all_results[strat_name] = {}

        for cfg_name in ["Current", "Robust"]:
            all_results[strat_name][cfg_name] = {}
            for yk in year_keys:
                cache_key = (instrument, cfg_name, yk, sp["rsi_len"])
                arrays = cache[cache_key]
                metrics = run_single(arrays, sp, dollar_per_pt, commission)
                all_results[strat_name][cfg_name][yk] = metrics

    run_time = time.time() - t_start
    print(f"\n  All backtests done in {run_time:.1f}s")

    # ===================================================================
    # Print results per strategy
    # ===================================================================
    for strat_name in STRATEGIES:
        sp = STRATEGIES[strat_name]
        instrument = sp["instrument"]
        sm_configs = MNQ_CONFIGS if instrument == "MNQ" else MES_CONFIGS

        print(f"\n{'='*110}")
        current_sm = sm_configs["Current"]
        robust_sm = sm_configs["Robust"]
        print(f"  {strat_name} ({instrument})")
        print(f"  Current SM: ({current_sm['index']}/{current_sm['flow']}/{current_sm['norm']}/{current_sm['ema']})")
        print(f"  Robust SM:  ({robust_sm['index']}/{robust_sm['flow']}/{robust_sm['norm']}/{robust_sm['ema']})")
        print(f"{'='*110}")

        # Header
        print(f"\n  {'Year':<16} | {'Config':<8} | {'Trades':>6} | {'WR%':>6} | {'PF':>7} | {'Net$':>10} | {'Sharpe':>7}")
        print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*10}-+-{'-'*7}")

        for yk in year_keys:
            label = year_labels[yk]
            cur = all_results[strat_name]["Current"][yk]
            rob = all_results[strat_name]["Robust"][yk]

            print(f"  {label:<16} | {'Current':<8} | {cur['trades']:>6} | {cur['wr']:>5.1f}% | {cur['pf']:>7.3f} | ${cur['net']:>+9,.0f} | {cur['sharpe']:>7.3f}")
            print(f"  {'':16} | {'Robust':<8} | {rob['trades']:>6} | {rob['wr']:>5.1f}% | {rob['pf']:>7.3f} | ${rob['net']:>+9,.0f} | {rob['sharpe']:>7.3f}")

            # Delta
            d_trades = rob['trades'] - cur['trades']
            d_wr = rob['wr'] - cur['wr']
            d_pf = rob['pf'] - cur['pf']
            d_net = rob['net'] - cur['net']
            d_sharpe = rob['sharpe'] - cur['sharpe']

            print(f"  {'':16} | {'Delta':<8} | {d_trades:>+6} | {d_wr:>+5.1f}% | {d_pf:>+7.3f} | ${d_net:>+9,.0f} | {d_sharpe:>+7.3f}")
            print()

        # 3-year totals
        cur_total_net = sum(all_results[strat_name]["Current"][yk]["net"] for yk in year_keys)
        rob_total_net = sum(all_results[strat_name]["Robust"][yk]["net"] for yk in year_keys)
        cur_avg_pf = np.mean([all_results[strat_name]["Current"][yk]["pf"] for yk in year_keys])
        rob_avg_pf = np.mean([all_results[strat_name]["Robust"][yk]["pf"] for yk in year_keys])
        cur_avg_sharpe = np.mean([all_results[strat_name]["Current"][yk]["sharpe"] for yk in year_keys])
        rob_avg_sharpe = np.mean([all_results[strat_name]["Robust"][yk]["sharpe"] for yk in year_keys])
        cur_worst_pf = min(all_results[strat_name]["Current"][yk]["pf"] for yk in year_keys)
        rob_worst_pf = min(all_results[strat_name]["Robust"][yk]["pf"] for yk in year_keys)

        print(f"  3-YEAR SUMMARY:")
        print(f"    {'':20} {'Current':>10} {'Robust':>10} {'Delta':>10}")
        print(f"    {'Total Net$':20} ${cur_total_net:>+9,.0f} ${rob_total_net:>+9,.0f} ${rob_total_net - cur_total_net:>+9,.0f}")
        print(f"    {'Avg PF':20} {cur_avg_pf:>10.3f} {rob_avg_pf:>10.3f} {rob_avg_pf - cur_avg_pf:>+10.3f}")
        print(f"    {'Worst-Year PF':20} {cur_worst_pf:>10.3f} {rob_worst_pf:>10.3f} {rob_worst_pf - cur_worst_pf:>+10.3f}")
        print(f"    {'Avg Sharpe':20} {cur_avg_sharpe:>10.3f} {rob_avg_sharpe:>10.3f} {rob_avg_sharpe - cur_avg_sharpe:>+10.3f}")

        # Verdict
        if rob_total_net > cur_total_net and rob_worst_pf > cur_worst_pf:
            verdict = "ROBUST WINS — higher total P&L AND better worst-year PF"
        elif rob_total_net > cur_total_net:
            verdict = "ROBUST HIGHER P&L — but worse worst-year (less consistent)"
        elif rob_worst_pf > cur_worst_pf:
            verdict = "ROBUST MORE STABLE — better worst-year PF but lower total P&L"
        else:
            verdict = "CURRENT WINS — keep current params"
        print(f"    VERDICT: {verdict}")

    # ===================================================================
    # GRAND SUMMARY TABLE
    # ===================================================================
    print(f"\n{'='*110}")
    print("GRAND SUMMARY — All Strategies")
    print(f"{'='*110}")

    print(f"\n  {'Strategy':<12} | {'3Y Net$ Cur':>12} | {'3Y Net$ Rob':>12} | {'Delta$':>10} | "
          f"{'AvgPF Cur':>9} | {'AvgPF Rob':>9} | {'WrstPF Cur':>10} | {'WrstPF Rob':>10} | {'Verdict'}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}-+-{'-'*30}")

    total_cur = 0
    total_rob = 0

    for strat_name in STRATEGIES:
        cur_net = sum(all_results[strat_name]["Current"][yk]["net"] for yk in year_keys)
        rob_net = sum(all_results[strat_name]["Robust"][yk]["net"] for yk in year_keys)
        cur_avg_pf = np.mean([all_results[strat_name]["Current"][yk]["pf"] for yk in year_keys])
        rob_avg_pf = np.mean([all_results[strat_name]["Robust"][yk]["pf"] for yk in year_keys])
        cur_worst = min(all_results[strat_name]["Current"][yk]["pf"] for yk in year_keys)
        rob_worst = min(all_results[strat_name]["Robust"][yk]["pf"] for yk in year_keys)
        delta = rob_net - cur_net

        total_cur += cur_net
        total_rob += rob_net

        if rob_net > cur_net and rob_worst > cur_worst:
            v = "ROBUST WINS"
        elif rob_net > cur_net:
            v = "ROBUST $ (less stable)"
        elif rob_worst > cur_worst:
            v = "ROBUST STABLE (less $)"
        else:
            v = "CURRENT WINS"

        print(f"  {strat_name:<12} | ${cur_net:>+11,.0f} | ${rob_net:>+11,.0f} | ${delta:>+9,.0f} | "
              f"{cur_avg_pf:>9.3f} | {rob_avg_pf:>9.3f} | {cur_worst:>10.3f} | {rob_worst:>10.3f} | {v}")

    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}-+-{'-'*30}")
    print(f"  {'TOTAL':<12} | ${total_cur:>+11,.0f} | ${total_rob:>+11,.0f} | ${total_rob - total_cur:>+9,.0f} |")

    # ===================================================================
    # PORTFOLIO VIEW: A(1) + B(1) + C(2) + MES(2)
    # ===================================================================
    print(f"\n{'='*110}")
    print("PORTFOLIO VIEW — A(1) + B(1) + C(2) + MES(2)")
    print(f"{'='*110}")

    # Weights: vScalpA=1, vScalpB=1, vScalpC=2 (already 2-contract in partial P&L), MES=2 (multiply by 2)
    weights = {"vScalpA": 1, "vScalpB": 1, "vScalpC": 1, "MES v2": 2}
    # Note: vScalpC partial exit already accounts for 2 contracts in its P&L

    for yk in year_keys:
        label = year_labels[yk]
        cur_port = sum(all_results[s]["Current"][yk]["net"] * weights[s] for s in STRATEGIES)
        rob_port = sum(all_results[s]["Robust"][yk]["net"] * weights[s] for s in STRATEGIES)
        delta = rob_port - cur_port
        print(f"  {label:<16}  Current: ${cur_port:>+10,.0f}  Robust: ${rob_port:>+10,.0f}  Delta: ${delta:>+9,.0f}")

    cur_port_total = sum(
        sum(all_results[s]["Current"][yk]["net"] * weights[s] for s in STRATEGIES)
        for yk in year_keys
    )
    rob_port_total = sum(
        sum(all_results[s]["Robust"][yk]["net"] * weights[s] for s in STRATEGIES)
        for yk in year_keys
    )
    print(f"  {'-'*70}")
    print(f"  {'3-YEAR TOTAL':<16}  Current: ${cur_port_total:>+10,.0f}  Robust: ${rob_port_total:>+10,.0f}  Delta: ${rob_port_total - cur_port_total:>+9,.0f}")

    if rob_port_total > cur_port_total:
        print(f"\n  --> Robust SM params ADD ${rob_port_total - cur_port_total:,.0f} to the 3-year portfolio")
    else:
        print(f"\n  --> Current SM params are better by ${cur_port_total - rob_port_total:,.0f} over 3 years")

    # ===================================================================
    # YEAR-BY-YEAR CONSISTENCY CHECK
    # ===================================================================
    print(f"\n{'='*110}")
    print("CONSISTENCY CHECK — Does Robust win on EVERY year?")
    print(f"{'='*110}")

    for strat_name in STRATEGIES:
        wins = []
        for yk in year_keys:
            cur = all_results[strat_name]["Current"][yk]
            rob = all_results[strat_name]["Robust"][yk]
            robust_better = rob["net"] > cur["net"]
            wins.append(robust_better)
            symbol = "+" if robust_better else "-"
            print(f"  {strat_name:<12} {year_labels[yk]:<16} Robust {'BETTER' if robust_better else 'WORSE':>6} "
                  f"(Cur ${cur['net']:>+9,.0f}  Rob ${rob['net']:>+9,.0f}  "
                  f"Delta ${rob['net'] - cur['net']:>+9,.0f})")
        all_win = all(wins)
        print(f"  {strat_name:<12} {'ALL YEARS':16} {'CONSISTENT' if all_win else 'MIXED'}")
        print()

    total_time = time.time() - t_start
    print(f"\n{'='*110}")
    print(f"A/B TEST COMPLETE — {total_time:.0f}s total")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
