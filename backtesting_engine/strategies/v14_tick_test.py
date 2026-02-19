"""
v14 Tick Microstructure Feature Test
======================================
Loads precomputed tick microstructure features (from v14_tick_compute.py) and
tests them as exit modifiers / regime gates on the v11 MNQ strategy.

9 Variants:
  L1/L2 - Price impact asymmetry exit (threshold -0.7 / -0.5)
  M1/M2 - Sweep-based stop tighten (25pts / 35pts)
  N1/N2 - Footprint imbalance exit (ratio < 0.3 / < 0.5)
  O1    - CV regime gate (suppress entries when tick_cv > 80th pct)
  P1/P2 - Iceberg detection exit (3x / 2x opposing ratio)

Train/Test split:
  TRAIN: Aug 17 - Nov 16, 2025 (~3 months)
  TEST:  Nov 17, 2025 - Feb 13, 2026 (~3 months)

Usage:
  python3 v14_tick_test.py                              # TRAIN, all variants
  python3 v14_tick_test.py --oos                        # TEST (OOS), all variants
  python3 v14_tick_test.py --variant L1                 # Single variant
  python3 v14_tick_test.py --features data/custom.csv   # Custom features file
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    load_instrument_1min, prepare_backtest_arrays_1min,
    run_v9_baseline, score_trades, fmt_score,
    compute_et_minutes, compute_rsi,
    bootstrap_ci, permutation_test,
    NY_OPEN_ET, NY_CLOSE_ET, NY_LAST_ENTRY_ET,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRAIN_START = pd.Timestamp("2025-08-17", tz='UTC')
TRAIN_END = pd.Timestamp("2025-11-17", tz='UTC')
TEST_START = pd.Timestamp("2025-11-17", tz='UTC')
TEST_END = pd.Timestamp("2026-02-14", tz='UTC')

V11_RSI_LEN = 8
V11_RSI_BUY = 60
V11_RSI_SELL = 40
V11_COOLDOWN = 20
V11_MAX_LOSS = 50
V11_SM_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Feature Loading
# ---------------------------------------------------------------------------

def load_features(path):
    """Load precomputed tick microstructure features from CSV.

    Expects columns: time (unix seconds), buy_impact, sell_impact,
    impact_log_ratio, impact_ema3, buy_sweep_pts, sell_sweep_pts,
    top_ratio, bot_ratio, tick_cv, buy_iceberg_vol, sell_iceberg_vol.

    Returns DataFrame indexed by UTC timestamp.
    """
    df = pd.read_csv(path)
    df['time_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time_utc')
    return df


# ---------------------------------------------------------------------------
# Feature Alignment
# ---------------------------------------------------------------------------

def align_features(ohlcv_1m, feat_df):
    """Align feature columns to OHLCV index using reindex with ffill.

    Returns dict of numpy arrays keyed by feature name, aligned to
    ohlcv_1m.index. Missing values are forward-filled then zero-filled.
    """
    cols = ['buy_impact', 'sell_impact', 'impact_log_ratio', 'impact_ema3',
            'buy_sweep_pts', 'sell_sweep_pts',
            'top_ratio', 'bot_ratio',
            'tick_cv', 'buy_iceberg_vol', 'sell_iceberg_vol']
    aligned = {}
    for col in cols:
        aligned[col] = feat_df[col].reindex(ohlcv_1m.index, method='ffill').fillna(0).values

    # Stats
    n_bars = len(ohlcv_1m)
    nz_impact = np.count_nonzero(aligned['impact_ema3'])
    nz_sweep = np.count_nonzero(aligned['buy_sweep_pts']) + np.count_nonzero(aligned['sell_sweep_pts'])
    nz_iceberg = np.count_nonzero(aligned['buy_iceberg_vol']) + np.count_nonzero(aligned['sell_iceberg_vol'])
    print(f"  Aligned {n_bars:,} bars to OHLCV index")
    print(f"  Non-zero: impact_ema3={nz_impact:,}, sweeps={nz_sweep:,}, icebergs={nz_iceberg:,}")

    return aligned


# ---------------------------------------------------------------------------
# Variant Configs
# ---------------------------------------------------------------------------

VARIANT_CONFIGS = {
    'L1': {'impact_exit': True, 'impact_threshold': -0.7},
    'L2': {'impact_exit': True, 'impact_threshold': -0.5},
    'M1': {'sweep_tighten': True, 'sweep_tight_stop': 25, 'sweep_countdown_max': 5},
    'M2': {'sweep_tighten': True, 'sweep_tight_stop': 35, 'sweep_countdown_max': 5},
    'N1': {'footprint_exit': True, 'footprint_threshold': 0.3},
    'N2': {'footprint_exit': True, 'footprint_threshold': 0.5},
    'O1': {'cv_gate': True, 'cv_lookback': 100, 'cv_suppress_bars': 10},
    'P1': {'iceberg_exit': True, 'iceberg_ratio': 3.0},
    'P2': {'iceberg_exit': True, 'iceberg_ratio': 2.0},
    # Underwater-gated iceberg: only fires when position is losing >= min_loss pts
    'P3': {'iceberg_exit': True, 'iceberg_ratio': 3.0, 'iceberg_min_loss': 15},
    'P4': {'iceberg_exit': True, 'iceberg_ratio': 2.0, 'iceberg_min_loss': 15},
    'P5': {'iceberg_exit': True, 'iceberg_ratio': 3.0, 'iceberg_min_loss': 10},
    'P6': {'iceberg_exit': True, 'iceberg_ratio': 2.0, 'iceberg_min_loss': 10},
}

VARIANT_DESCS = {
    'L1': 'Impact exit (threshold -0.7)',
    'L2': 'Impact exit (threshold -0.5)',
    'M1': 'Sweep stop tighten 25pts',
    'M2': 'Sweep stop tighten 35pts',
    'N1': 'Footprint exit (ratio < 0.3)',
    'N2': 'Footprint exit (ratio < 0.5)',
    'O1': 'CV regime gate',
    'P1': 'Iceberg exit (3x ratio)',
    'P2': 'Iceberg exit (2x ratio)',
    'P3': 'Iceberg exit 3x, underwater 15pt gate',
    'P4': 'Iceberg exit 2x, underwater 15pt gate',
    'P5': 'Iceberg exit 3x, underwater 10pt gate',
    'P6': 'Iceberg exit 2x, underwater 10pt gate',
}

ALL_VARIANTS = ['L1', 'L2', 'M1', 'M2', 'N1', 'N2', 'O1', 'P1', 'P2',
                'P3', 'P4', 'P5', 'P6']


# ---------------------------------------------------------------------------
# Custom Engine (handles ALL variants via config dict)
# ---------------------------------------------------------------------------

def _run_custom_engine(arr, features, config):
    """Custom v11 engine supporting all v14 tick microstructure variants.

    config keys:
        impact_exit: bool - enable L variants
        impact_threshold: float - e.g., -0.7 or -0.5
        sweep_tighten: bool - enable M variants
        sweep_tight_stop: int - e.g., 25 or 35
        sweep_countdown_max: int - e.g., 5 (bars before reverting)
        footprint_exit: bool - enable N variants
        footprint_threshold: float - e.g., 0.3 or 0.5
        cv_gate: bool - enable O variant
        cv_lookback: int - e.g., 100
        cv_suppress_bars: int - e.g., 10
        iceberg_exit: bool - enable P variants
        iceberg_ratio: float - e.g., 3.0 or 2.0

    Exit priority: EOD > SL/SL_TIGHT > IMPACT > FOOTPRINT > ICEBERG > SM_FLIP

    All new exits check bar i-1 data and fill at bar i open (no look-ahead).
    Baseline logic follows the validated v11 pattern EXACTLY.

    Returns list of trade dicts.
    """
    n = len(arr['opens'])
    opens = arr['opens']
    closes = arr['closes']
    sm = arr['sm']
    times = arr['times']
    rsi_5m_curr = arr.get('rsi_5m_curr')
    rsi_5m_prev = arr.get('rsi_5m_prev')

    et_mins = compute_et_minutes(times)
    rsi = arr.get('rsi', compute_rsi(closes, V11_RSI_LEN))

    trades = []
    trade_state = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    exit_bar = -9999
    long_used = False
    short_used = False

    # M variant state
    sweep_tight_countdown = 0

    # O variant state
    cv_suppress_until = -1  # bar index until which entries are suppressed

    def close_trade(side, ep, xp, ei, xi, result):
        pts = (xp - ep) if side == "long" else (ep - xp)
        trades.append({"side": side, "entry": ep, "exit": xp,
                        "pts": pts, "entry_time": times[ei],
                        "exit_time": times[xi], "entry_idx": ei,
                        "exit_idx": xi, "bars": xi - ei, "result": result})

    for i in range(2, n):
        bar_et = et_mins[i]
        sm_prev = sm[i - 1]
        sm_prev2 = sm[i - 2]
        sm_bull = sm_prev > V11_SM_THRESHOLD
        sm_bear = sm_prev < -V11_SM_THRESHOLD
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI cross (mapped 5-min)
        if rsi_5m_curr is not None and rsi_5m_prev is not None:
            rsi_prev = rsi_5m_curr[i - 1]
            rsi_prev2 = rsi_5m_prev[i - 1]
            rsi_long_trigger = rsi_prev > V11_RSI_BUY and rsi_prev2 <= V11_RSI_BUY
            rsi_short_trigger = rsi_prev < V11_RSI_SELL and rsi_prev2 >= V11_RSI_SELL
        else:
            rsi_prev = rsi[i - 1]
            rsi_prev2 = rsi[i - 2]
            rsi_long_trigger = rsi_prev > V11_RSI_BUY and rsi_prev2 <= V11_RSI_BUY
            rsi_short_trigger = rsi_prev < V11_RSI_SELL and rsi_prev2 >= V11_RSI_SELL

        # Episode reset (NEVER modify this)
        if sm_flipped_bull or not sm_bull:
            long_used = False
        if sm_flipped_bear or not sm_bear:
            short_used = False

        # EOD close (highest priority exit)
        if trade_state != 0 and bar_et >= NY_CLOSE_ET:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            sweep_tight_countdown = 0
            continue

        # --- Exits for open positions ---
        if trade_state == 1:  # LONG
            # 1. Determine effective stop (M variants: sweep tighten)
            effective_stop = V11_MAX_LOSS
            if config.get('sweep_tighten') and sweep_tight_countdown > 0:
                effective_stop = config['sweep_tight_stop']
                sweep_tight_countdown -= 1

            # Check for new sweep trigger (opposing sweep with no supporting sweep)
            if config.get('sweep_tighten'):
                if features['sell_sweep_pts'][i - 1] > 0 and features['buy_sweep_pts'][i - 1] == 0:
                    sweep_tight_countdown = config.get('sweep_countdown_max', 5)
                    effective_stop = config['sweep_tight_stop']

            # 2. Max loss stop (bar i-1 data, fill at bar i open) -- look-ahead fix
            if effective_stop > 0 and closes[i - 1] <= entry_price - effective_stop:
                result = "SL_TIGHT" if effective_stop < V11_MAX_LOSS else "SL"
                close_trade("long", entry_price, opens[i], entry_idx, i, result)
                trade_state = 0
                exit_bar = i
                sweep_tight_countdown = 0
                continue

            # 3. Price impact asymmetry exit (L variants)
            if config.get('impact_exit'):
                threshold = config['impact_threshold']
                if features['impact_ema3'][i - 1] < threshold:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "IMPACT")
                    trade_state = 0
                    exit_bar = i
                    sweep_tight_countdown = 0
                    continue

            # 4. Footprint imbalance exit (N variants)
            if config.get('footprint_exit') and i >= 3:
                ft = config['footprint_threshold']
                if (features['top_ratio'][i - 1] < ft and
                        features['top_ratio'][i - 2] < ft):
                    close_trade("long", entry_price, opens[i], entry_idx, i, "FOOTPRINT")
                    trade_state = 0
                    exit_bar = i
                    sweep_tight_countdown = 0
                    continue

            # 5. Iceberg exit (P variants)
            if config.get('iceberg_exit') and i >= 3:
                min_loss = config.get('iceberg_min_loss', 0)
                unrealized = closes[i - 1] - entry_price  # positive = winning
                if unrealized <= -min_loss:  # only fire if underwater by min_loss
                    ratio = config['iceberg_ratio']
                    sell_ice = features['sell_iceberg_vol'][i - 1]
                    buy_ice = features['buy_iceberg_vol'][i - 1]
                    sell_ice_prev = features['sell_iceberg_vol'][i - 2]
                    buy_ice_prev = features['buy_iceberg_vol'][i - 2]
                    if (sell_ice > 0 and sell_ice > ratio * max(buy_ice, 1) and
                            sell_ice_prev > 0 and sell_ice_prev > ratio * max(buy_ice_prev, 1)):
                        close_trade("long", entry_price, opens[i], entry_idx, i, "ICEBERG")
                        trade_state = 0
                        exit_bar = i
                        sweep_tight_countdown = 0
                        continue

            # 6. SM flip exit (standard, NEVER modify)
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                sweep_tight_countdown = 0

        elif trade_state == -1:  # SHORT (mirror all exit logic)
            # 1. Determine effective stop (M variants: sweep tighten)
            effective_stop = V11_MAX_LOSS
            if config.get('sweep_tighten') and sweep_tight_countdown > 0:
                effective_stop = config['sweep_tight_stop']
                sweep_tight_countdown -= 1

            # Check for new sweep trigger (opposing sweep with no supporting sweep)
            if config.get('sweep_tighten'):
                if features['buy_sweep_pts'][i - 1] > 0 and features['sell_sweep_pts'][i - 1] == 0:
                    sweep_tight_countdown = config.get('sweep_countdown_max', 5)
                    effective_stop = config['sweep_tight_stop']

            # 2. Max loss stop
            if effective_stop > 0 and closes[i - 1] >= entry_price + effective_stop:
                result = "SL_TIGHT" if effective_stop < V11_MAX_LOSS else "SL"
                close_trade("short", entry_price, opens[i], entry_idx, i, result)
                trade_state = 0
                exit_bar = i
                sweep_tight_countdown = 0
                continue

            # 3. Price impact asymmetry exit (L variants) -- mirrored
            if config.get('impact_exit'):
                threshold = abs(config['impact_threshold'])
                if features['impact_ema3'][i - 1] > threshold:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "IMPACT")
                    trade_state = 0
                    exit_bar = i
                    sweep_tight_countdown = 0
                    continue

            # 4. Footprint imbalance exit (N variants) -- mirrored
            if config.get('footprint_exit') and i >= 3:
                ft = config['footprint_threshold']
                if (features['bot_ratio'][i - 1] < ft and
                        features['bot_ratio'][i - 2] < ft):
                    close_trade("short", entry_price, opens[i], entry_idx, i, "FOOTPRINT")
                    trade_state = 0
                    exit_bar = i
                    sweep_tight_countdown = 0
                    continue

            # 5. Iceberg exit (P variants) -- mirrored
            if config.get('iceberg_exit') and i >= 3:
                min_loss = config.get('iceberg_min_loss', 0)
                unrealized = entry_price - closes[i - 1]  # positive = winning (short)
                if unrealized <= -min_loss:  # only fire if underwater by min_loss
                    ratio = config['iceberg_ratio']
                    buy_ice = features['buy_iceberg_vol'][i - 1]
                    sell_ice = features['sell_iceberg_vol'][i - 1]
                    buy_ice_prev = features['buy_iceberg_vol'][i - 2]
                    sell_ice_prev = features['sell_iceberg_vol'][i - 2]
                    if (buy_ice > 0 and buy_ice > ratio * max(sell_ice, 1) and
                            buy_ice_prev > 0 and buy_ice_prev > ratio * max(sell_ice_prev, 1)):
                        close_trade("short", entry_price, opens[i], entry_idx, i, "ICEBERG")
                        trade_state = 0
                        exit_bar = i
                        sweep_tight_countdown = 0
                        continue

            # 6. SM flip exit (standard, NEVER modify)
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i
                sweep_tight_countdown = 0

        # --- Entry logic ---
        if trade_state == 0:
            in_session = NY_OPEN_ET <= bar_et <= NY_LAST_ENTRY_ET
            cd_ok = (i - exit_bar) >= V11_COOLDOWN

            # O variant: CV regime gate
            if config.get('cv_gate'):
                if i >= config.get('cv_lookback', 100):
                    lookback = config['cv_lookback']
                    cv_window = features['tick_cv'][max(0, i - lookback):i]
                    cv_window_valid = cv_window[cv_window > 0]
                    if len(cv_window_valid) > 0:
                        pct80 = np.percentile(cv_window_valid, 80)
                        if features['tick_cv'][i - 1] > pct80:
                            cv_suppress_until = i + config.get('cv_suppress_bars', 10)

                if i < cv_suppress_until:
                    continue  # skip entry entirely

            if in_session and cd_ok:
                # Long entry
                if sm_bull and rsi_long_trigger and not long_used:
                    trade_state = 1
                    entry_price = opens[i]
                    entry_idx = i
                    long_used = True

                # Short entry
                elif sm_bear and rsi_short_trigger and not short_used:
                    trade_state = -1
                    entry_price = opens[i]
                    entry_idx = i
                    short_used = True

    return trades


# ---------------------------------------------------------------------------
# Evaluation (7-criteria framework, same as v13)
# ---------------------------------------------------------------------------

def evaluate_variant(name, desc, trades_baseline, trades_feature,
                     commission=0.52, dollar_per_pt=2.0):
    """Evaluate a feature variant using the 7-criteria framework."""
    sc_base = score_trades(trades_baseline, commission, dollar_per_pt)
    sc_feat = score_trades(trades_feature, commission, dollar_per_pt)

    if sc_base is None:
        print(f"  {name} ({desc}): SKIP (no baseline trades)")
        return None
    if sc_feat is None:
        print(f"  {name} ({desc}): SKIP (no feature trades)")
        return None

    print(f"\n  --- Variant {name}: {desc} ---")
    print(f"  Baseline: {fmt_score(sc_base, 'BASE')}")
    print(f"  Feature:  {fmt_score(sc_feat, name)}")

    # Show exit types for exit-modifier variants (L, M, N, P)
    if sc_feat.get('exits'):
        exit_parts = [f"{k}:{v}" for k, v in sorted(sc_feat['exits'].items())]
        print(f"  Exits:    {' '.join(exit_parts)}")

    # For M variants, show SL_TIGHT breakdown specifically
    if name.startswith('M') and sc_feat.get('exits'):
        exits = sc_feat['exits']
        sl_std = exits.get('SL', 0)
        sl_tight = exits.get('SL_TIGHT', 0)
        sm_flip = exits.get('SM_FLIP', 0)
        eod = exits.get('EOD', 0)
        total = sc_feat['count']
        print(f"  Stop breakdown: SL={sl_std} ({100*sl_std/total:.1f}%), "
              f"SL_TIGHT={sl_tight} ({100*sl_tight/total:.1f}%), "
              f"SM_FLIP={sm_flip} ({100*sm_flip/total:.1f}%), "
              f"EOD={eod} ({100*eod/total:.1f}%)")

    # 1. PF improvement
    dpf = sc_feat['pf'] - sc_base['pf']
    pf_pass = dpf > 0.1

    # 2. Bootstrap CI
    ci_point, ci_lo, ci_hi = bootstrap_ci(trades_feature, metric='pf',
                                           commission_per_side=commission,
                                           dollar_per_pt=dollar_per_pt)
    ci_excludes_base = ci_lo > sc_base['pf']

    # 3. Permutation test
    obs_diff, p_val = permutation_test(trades_baseline, trades_feature,
                                        commission_per_side=commission,
                                        dollar_per_pt=dollar_per_pt)

    # 4. Trade count preservation
    count_ratio = sc_feat['count'] / sc_base['count'] if sc_base['count'] > 0 else 0
    count_ok = count_ratio > 0.5

    # 5. Win rate
    dwr = sc_feat['win_rate'] - sc_base['win_rate']
    wr_ok = dwr > -5

    # 6. Drawdown
    base_dd = sc_base['max_dd_dollar']
    feat_dd = sc_feat['max_dd_dollar']
    dd_ok = feat_dd >= base_dd * 1.2  # not >20% worse (dd is negative)

    # 7. Lucky trade removal
    if len(trades_feature) > 1:
        pts = np.array([t['pts'] for t in trades_feature])
        best_idx = np.argmax(pts)
        trades_no_best = [t for j, t in enumerate(trades_feature) if j != best_idx]
        sc_no_best = score_trades(trades_no_best, commission, dollar_per_pt)
        lucky_ok = sc_no_best is not None and sc_no_best['pf'] > 1.0
    else:
        lucky_ok = False
        sc_no_best = None

    criteria = [
        ("PF improvement > 0.1", pf_pass, f"dPF={dpf:+.3f}"),
        ("Bootstrap CI excludes baseline", ci_excludes_base,
         f"CI=[{ci_lo:.3f}, {ci_hi:.3f}], base={sc_base['pf']:.3f}"),
        ("Permutation p < 0.05", p_val < 0.05, f"p={p_val:.4f}"),
        ("Trade count > 50% of baseline", count_ok,
         f"ratio={count_ratio:.2f} ({sc_feat['count']}/{sc_base['count']})"),
        ("Win rate not degraded > 5%", wr_ok, f"dWR={dwr:+.1f}%"),
        ("Drawdown not >20% worse", dd_ok,
         f"feat DD=${feat_dd:.0f}, base DD=${base_dd:.0f}"),
        ("Survives lucky trade removal", lucky_ok,
         f"PF w/o best={sc_no_best['pf']:.3f}" if sc_no_best else "N/A"),
    ]

    passes = 0
    print(f"\n  Criteria:")
    for crit_name, passed, detail in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {crit_name}: {detail}")
        if passed:
            passes += 1

    verdict = "ADOPT" if passes >= 5 else ("MAYBE" if passes >= 3 else "REJECT")
    print(f"\n  Score: {passes}/7 -> {verdict}")

    return {
        'name': name, 'desc': desc, 'verdict': verdict,
        'passes': passes, 'sc_base': sc_base, 'sc_feat': sc_feat,
        'dpf': dpf, 'p_val': p_val,
    }


# ---------------------------------------------------------------------------
# Stop-Loss Rescue Analysis
# ---------------------------------------------------------------------------

def analyze_stop_rescue(trades_baseline, trades_feature, variant_name):
    """How many baseline SL trades were rescued (exited earlier/better)?

    A "rescue" is when a baseline trade that hit SL was instead exited by
    a different signal in the feature variant, AND the feature exit produced
    fewer points lost (positive pts_saved).
    """
    # Find baseline trades that hit SL
    bl_sl = [t for t in trades_baseline if t.get('result') == 'SL']
    # Find feature trades with same entry_time but different exit result
    feat_by_entry = {t['entry_time']: t for t in trades_feature}

    rescued = 0
    total_saved_pts = 0
    for bl_trade in bl_sl:
        feat_trade = feat_by_entry.get(bl_trade['entry_time'])
        if feat_trade is None:
            continue  # trade was filtered out (entry filter variant)
        if feat_trade['result'] != 'SL' and feat_trade['result'] != 'SL_TIGHT':
            # This SL trade was rescued (exited by a different signal)
            pts_saved = feat_trade['pts'] - bl_trade['pts']  # positive if less loss
            if pts_saved > 0:
                rescued += 1
                total_saved_pts += pts_saved

    n_sl = len(bl_sl)
    print(f"    SL trades rescued: {rescued}/{n_sl}")
    if rescued > 0:
        print(f"    Avg pts saved per rescue: {total_saved_pts/rescued:.1f}")
        print(f"    Total pts saved: {total_saved_pts:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v14 tick microstructure feature test on v11 MNQ baseline",
    )
    parser.add_argument("--features", default=None,
                        help="Path to precomputed features CSV "
                             "(default: data/tick_microstructure_features.csv)")
    parser.add_argument("--instrument", default="MNQ",
                        help="Instrument (default: MNQ)")
    parser.add_argument("--variant", default=None,
                        help="Test specific variant (L1,L2,M1,M2,N1,N2,O1,P1,P2)")
    parser.add_argument("--oos", action="store_true",
                        help="Run on out-of-sample (test) period")

    args = parser.parse_args()

    # Resolve features file
    if args.features:
        feat_path = Path(args.features)
    else:
        feat_path = DATA_DIR / "tick_microstructure_features.csv"
    if not feat_path.exists():
        print(f"ERROR: Features file not found: {feat_path}")
        print(f"  Run v14_tick_compute.py first to generate features.")
        sys.exit(1)

    if args.oos:
        period_start, period_end = TEST_START, TEST_END
        period_name = "TEST (OOS)"
    else:
        period_start, period_end = TRAIN_START, TRAIN_END
        period_name = "TRAIN"

    print("=" * 70)
    print(f"v14 TICK MICROSTRUCTURE FEATURE TEST -- {period_name}")
    print("=" * 70)
    print(f"  Period:     {period_start.date()} to {period_end.date()}")
    print(f"  Instrument: {args.instrument}")
    print(f"  Features:   {feat_path.name}")
    print(f"  Commission: $0.52/side (0.005%)")
    print()

    # -----------------------------------------------------------------------
    # 1. Load precomputed features
    # -----------------------------------------------------------------------
    print("Step 1: Load precomputed features")
    t0 = time.time()
    feat_df = load_features(feat_path)
    print(f"  Loaded {len(feat_df):,} rows in {time.time() - t0:.1f}s")
    print(f"  Date range: {feat_df.index[0]} to {feat_df.index[-1]}")
    print(f"  Columns: {list(feat_df.columns)}")
    print()

    # -----------------------------------------------------------------------
    # 2. Load OHLCV and filter to period
    # -----------------------------------------------------------------------
    print("Step 2: Load OHLCV data")
    ohlcv_1m = load_instrument_1min(args.instrument)
    if ohlcv_1m.index.tz is None:
        ohlcv_1m.index = ohlcv_1m.index.tz_localize('UTC')

    period_data = ohlcv_1m[(ohlcv_1m.index >= period_start) &
                            (ohlcv_1m.index < period_end)]
    if len(period_data) < 100:
        print(f"ERROR: Only {len(period_data)} bars in period. Need more data.")
        sys.exit(1)
    print(f"  OHLCV bars: {len(period_data):,}")
    print(f"  Date range: {period_data.index[0]} to {period_data.index[-1]}")
    print()

    # -----------------------------------------------------------------------
    # 3. Align features to OHLCV
    # -----------------------------------------------------------------------
    print("Step 3: Align features to OHLCV index")
    features = align_features(period_data, feat_df)

    # Print feature stats
    ie = features['impact_ema3']
    ie_nz = ie[ie != 0]
    if len(ie_nz) > 0:
        print(f"  impact_ema3 stats: mean={ie_nz.mean():.3f}, "
              f"std={ie_nz.std():.3f}, "
              f"min={ie_nz.min():.3f}, max={ie_nz.max():.3f}")
    cv = features['tick_cv']
    cv_nz = cv[cv > 0]
    if len(cv_nz) > 0:
        print(f"  tick_cv stats: mean={cv_nz.mean():.3f}, "
              f"std={cv_nz.std():.3f}, "
              f"P80={np.percentile(cv_nz, 80):.3f}")
    bs = features['buy_sweep_pts']
    ss = features['sell_sweep_pts']
    print(f"  Sweep bars: buy={np.count_nonzero(bs):,}, sell={np.count_nonzero(ss):,}")
    bi = features['buy_iceberg_vol']
    si = features['sell_iceberg_vol']
    print(f"  Iceberg bars: buy={np.count_nonzero(bi):,}, sell={np.count_nonzero(si):,}")
    print()

    # -----------------------------------------------------------------------
    # 4. Prepare backtest arrays
    # -----------------------------------------------------------------------
    print("Step 4: Prepare backtest arrays (1-min with mapped 5-min RSI)")
    arr = prepare_backtest_arrays_1min(period_data, rsi_len=V11_RSI_LEN)
    print(f"  Arrays ready: {len(arr['opens']):,} bars")
    print()

    # -----------------------------------------------------------------------
    # 5. Run baseline
    # -----------------------------------------------------------------------
    print("Step 5: Run v11 baseline")
    trades_baseline = run_v9_baseline(
        arr, rsi_len=V11_RSI_LEN, rsi_buy=V11_RSI_BUY,
        rsi_sell=V11_RSI_SELL, cooldown=V11_COOLDOWN,
        max_loss_pts=V11_MAX_LOSS,
    )
    sc_baseline = score_trades(trades_baseline)
    print(f"  {fmt_score(sc_baseline, 'v11 BASELINE')}")
    if sc_baseline and sc_baseline.get('exits'):
        exit_parts = [f"{k}:{v}" for k, v in sorted(sc_baseline['exits'].items())]
        print(f"  Exits: {' '.join(exit_parts)}")
    print()

    # -----------------------------------------------------------------------
    # 6. Run variants
    # -----------------------------------------------------------------------
    if args.variant:
        variants_to_test = [args.variant.upper()]
    else:
        variants_to_test = ALL_VARIANTS

    results = []

    for var_key in variants_to_test:
        if var_key not in VARIANT_CONFIGS:
            print(f"\n  WARNING: Unknown variant '{var_key}', skipping")
            continue

        desc = VARIANT_DESCS[var_key]
        config = VARIANT_CONFIGS[var_key]
        print(f"Running variant {var_key}: {desc}...")
        t0 = time.time()
        trades_feat = _run_custom_engine(arr, features, config)
        elapsed = time.time() - t0
        print(f"  Engine time: {elapsed:.1f}s, Trades: {len(trades_feat)}")
        result = evaluate_variant(var_key, desc, trades_baseline, trades_feat)
        if result:
            results.append(result)

    # -----------------------------------------------------------------------
    # 7. Stop-loss rescue analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STOP-LOSS RESCUE ANALYSIS")
    print("=" * 70)

    for var_key in variants_to_test:
        if var_key not in VARIANT_CONFIGS:
            continue
        desc = VARIANT_DESCS[var_key]
        config = VARIANT_CONFIGS[var_key]
        trades_feat = _run_custom_engine(arr, features, config)
        print(f"\n  {var_key} ({desc}):")
        analyze_stop_rescue(trades_baseline, trades_feat, var_key)

    # -----------------------------------------------------------------------
    # 8. Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY -- {period_name}")
    print("=" * 70)

    if not results:
        print("  No results to summarize.")
        return

    print(f"\n  {'Variant':<36} | {'Verdict':>8} | {'Score':>5} | "
          f"{'dPF':>8} | {'p-val':>8} | {'Trades':>7}")
    print(f"  {'-'*36}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")

    best = None
    for r in results:
        n_trades = r['sc_feat']['count']
        line = (f"  {r['name']+': '+r['desc']:<36} | {r['verdict']:>8} | "
                f"{r['passes']:>3}/7 | {r['dpf']:>+8.3f} | "
                f"{r['p_val']:>8.4f} | {n_trades:>7}")
        print(line)
        if best is None or r['passes'] > best['passes']:
            best = r
        elif r['passes'] == best['passes'] and r['dpf'] > best['dpf']:
            best = r

    # Baseline reference
    if sc_baseline:
        print(f"\n  Baseline: {sc_baseline['count']} trades, "
              f"PF {sc_baseline['pf']}, Net ${sc_baseline['net_dollar']:+.2f}, "
              f"MaxDD ${sc_baseline['max_dd_dollar']:.2f}")

    if best and best['verdict'] != 'REJECT':
        print(f"\n  BEST VARIANT: {best['name']} ({best['desc']})")
        print(f"    PF: {best['sc_feat']['pf']:.3f} "
              f"(baseline {best['sc_base']['pf']:.3f})")
        if not args.oos:
            print(f"\n  Next step: python3 v14_tick_test.py "
                  f"--variant {best['name']} --oos")
    else:
        print(f"\n  No variant passed on {period_name} data.")
        if not args.oos:
            print("  Consider running --oos anyway to check test period behavior.")

    # OOS validation check
    if args.oos and best:
        print(f"\n  OOS VALIDATION:")
        feat_pf = best['sc_feat']['pf']
        base_pf = best['sc_base']['pf']
        if feat_pf < 1.0:
            print(f"    PF = {feat_pf:.3f} < 1.0 -> REJECT (unprofitable OOS)")
        elif feat_pf < base_pf * 0.7:
            print(f"    PF degraded >30% from baseline -> REJECT")
        else:
            print(f"    PF = {feat_pf:.3f} (baseline {base_pf:.3f}) -> PASS")


if __name__ == "__main__":
    main()
