"""
vScalpC Phase 2a — HTF Diagnostic Analysis
=============================================
For each vScalpC trade, captures all HTF indicator values at entry time.
Correlates with runner leg P&L to identify which higher-timeframe signals
predict TP2 success.

Outputs:
  - Correlation matrix: all HTF indicators vs leg2_pts
  - Quintile analysis: split trades into 5 groups by each indicator, compare P&L
  - Direction alignment: does HTF SM direction matching 1-min direction improve runner?
  - Scatter highlights: top predictors visualized

Usage:
    cd backtesting_engine && python3 strategies/htf_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Path setup ---
_STRAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRAT_DIR))
sys.path.insert(0, str(_STRAT_DIR.parent))
sys.path.insert(0, str(_STRAT_DIR.parent.parent / "live_trading"))

from htf_common import (
    compute_htf_indicators,
    compute_volatility_regime,
    compute_prior_day_atr,
    compute_session_context,
    prepare_vscalpc_data,
    HTF_TIMEFRAMES,
    VSCALPC_TP1, VSCALPC_TP2, VSCALPC_SL, VSCALPC_BE_TIME, VSCALPC_SL_TO_BE,
)

from vscalpc_partial_exit_sweep import (
    run_backtest_partial_exit,
    score_partial_trades,
)

from generate_session import (
    VSCALPA_RSI_BUY, VSCALPA_RSI_SELL,
    VSCALPA_SM_THRESHOLD, VSCALPA_COOLDOWN,
    VSCALPA_ENTRY_END_ET,
    MNQ_DOLLAR_PER_PT, MNQ_COMMISSION,
)

from v10_test_common import NY_CLOSE_ET


# ============================================================================
# Capture HTF values at each trade entry
# ============================================================================

def capture_entry_features(trades, htf_indicators, session_ctx, vol_regime,
                           sm_1m, closes_1m, prior_day_atr, et_mins):
    """For each trade, capture all indicator values at entry bar index.

    Uses i-1 (prev bar) values since entry decisions use prev bar state.

    Args:
        trades: list of trade dicts from run_backtest_partial_exit
        htf_indicators: dict from compute_htf_indicators
        session_ctx: dict from compute_session_context
        vol_regime: array from compute_volatility_regime
        sm_1m: 1-min SM Net array
        closes_1m: 1-min close array
        prior_day_atr: array from compute_prior_day_atr
        et_mins: ET minutes array from compute_et_minutes

    Returns:
        DataFrame with one row per trade, columns for each feature + trade results.
    """
    rows = []
    for t in trades:
        idx = t['entry_idx']
        prev = max(0, idx - 1)  # prev bar for entry decision

        # Time features
        entry_et_min = et_mins[idx]
        entry_hour = entry_et_min // 60

        # Entry time as pandas Timestamp for DOW
        entry_ts = pd.Timestamp(t['entry_time'])
        if entry_ts.tz is None:
            entry_ts = entry_ts.tz_localize('UTC')
        entry_dow = entry_ts.tz_convert('America/New_York').dayofweek  # 0=Mon

        row = {
            'entry_idx': idx,
            'entry_time': t['entry_time'],
            'side': t['side'],
            'entry_price': t['entry'],
            'total_pts': t['pts'],
            'leg1_pts': t['leg1_pts'],
            'leg2_pts': t['leg2_pts'],
            'leg1_exit': t['leg1_exit_reason'],
            'leg2_exit': t['leg2_exit_reason'],
            'tp2_hit': 1.0 if t['leg2_exit_reason'] == 'TP2' else 0.0,
            'bars': t['bars'],
            'sm_1m': sm_1m[prev],
            'sm_1m_abs': abs(sm_1m[prev]),
            'hour_et': entry_hour,
            'dow': entry_dow,
            'prior_day_atr': prior_day_atr[prev] if prev < len(prior_day_atr) else np.nan,
            'bars_to_tp1': (t['leg1_exit_idx'] - idx
                            if t.get('leg1_exit_reason') == 'TP1' else np.nan),
        }

        # HTF indicators
        for (tf, ind_name), values in htf_indicators.items():
            col = f'{tf}_{ind_name}'
            row[col] = values[prev] if prev < len(values) else np.nan

        # Session context
        for key, values in session_ctx.items():
            if isinstance(values, np.ndarray):
                row[f'session_{key}'] = values[prev] if prev < len(values) else np.nan

        # Volatility regime
        row['vol_regime'] = vol_regime[prev] if prev < len(vol_regime) else np.nan

        # Derived features
        # 1. HTF SM direction alignment with 1-min SM
        sm_1m_dir = 1 if sm_1m[prev] > 0 else (-1 if sm_1m[prev] < 0 else 0)
        for tf in HTF_TIMEFRAMES:
            htf_sm_dir = htf_indicators.get((tf, 'sm_dir'))
            if htf_sm_dir is not None:
                row[f'{tf}_sm_aligned'] = 1.0 if htf_sm_dir[prev] == sm_1m_dir else 0.0

        # 2. Close relative to EMA20 on each HTF
        for tf in HTF_TIMEFRAMES:
            htf_ema = htf_indicators.get((tf, 'ema20'))
            htf_close = htf_indicators.get((tf, 'close'))
            if htf_ema is not None and htf_close is not None:
                ema_val = htf_ema[prev]
                close_val = htf_close[prev]
                if not np.isnan(ema_val) and ema_val != 0:
                    row[f'{tf}_close_vs_ema'] = (close_val - ema_val) / ema_val * 100

        # 3. RSI overbought/oversold on HTF
        for tf in HTF_TIMEFRAMES:
            htf_rsi = htf_indicators.get((tf, 'rsi'))
            if htf_rsi is not None:
                rsi_val = htf_rsi[prev]
                row[f'{tf}_rsi_ob'] = 1.0 if rsi_val > 70 else 0.0
                row[f'{tf}_rsi_os'] = 1.0 if rsi_val < 30 else 0.0

        # 4. Price distance from OR/IB levels (points)
        entry_p = t['entry']
        for level_name in ['or_high', 'or_low', 'ib_high', 'ib_low']:
            level_val = session_ctx.get(level_name)
            if level_val is not None:
                lv = level_val[prev]
                if not np.isnan(lv):
                    row[f'dist_{level_name}'] = entry_p - lv

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Analysis: Correlation
# ============================================================================

def correlation_analysis(df_features, target='leg2_pts'):
    """Compute correlation of all numeric features with target.

    Returns sorted DataFrame of (feature, correlation, abs_correlation, p-value approx).
    """
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    exclude = {'entry_idx', 'entry_price', 'total_pts', 'leg1_pts', 'leg2_pts',
                'bars', 'tp2_hit', 'bars_to_tp1'}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    correlations = []
    target_vals = df_features[target].values
    n = len(target_vals)

    for col in feature_cols:
        vals = df_features[col].values
        # Skip if all NaN or constant
        valid = ~(np.isnan(vals) | np.isnan(target_vals))
        if valid.sum() < 20:
            continue
        v = vals[valid]
        t = target_vals[valid]
        if np.std(v) == 0:
            continue

        r = np.corrcoef(v, t)[0, 1]
        # Approximate t-statistic for significance
        n_valid = valid.sum()
        if abs(r) < 1.0 and n_valid > 2:
            t_stat = r * np.sqrt((n_valid - 2) / (1 - r**2))
        else:
            t_stat = 0

        correlations.append({
            'feature': col,
            'corr': round(r, 4),
            'abs_corr': round(abs(r), 4),
            'n_valid': n_valid,
            't_stat': round(t_stat, 2),
        })

    df_corr = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)
    return df_corr


# ============================================================================
# Analysis: Quintile
# ============================================================================

def quintile_analysis(df_features, feature_col, target='leg2_pts',
                      dollar_per_pt=MNQ_DOLLAR_PER_PT,
                      commission_per_side=MNQ_COMMISSION):
    """Split trades into 5 groups by feature value, compare runner performance.

    Returns DataFrame with quintile stats.
    """
    df = df_features.dropna(subset=[feature_col, target]).copy()
    if len(df) < 25:  # Need at least 5 per quintile
        return None

    df['quintile'] = pd.qcut(df[feature_col], 5, labels=False, duplicates='drop')

    comm_per_leg = commission_per_side * 2
    results = []

    for q in sorted(df['quintile'].unique()):
        group = df[df['quintile'] == q]
        n = len(group)
        avg_leg2 = group['leg2_pts'].mean()
        avg_total = group['total_pts'].mean()
        leg2_pnl = (group['leg2_pts'] * dollar_per_pt - comm_per_leg).values
        total_pnl = ((group['leg1_pts'] + group['leg2_pts']) * dollar_per_pt
                     - 2 * comm_per_leg).values

        # Runner win rate (leg2 > 0 in pts)
        runner_wr = (group['leg2_pts'] > 0).mean() * 100

        # Feature range
        feat_min = group[feature_col].min()
        feat_max = group[feature_col].max()

        # TP2 hit rate
        tp2_hits = (group['leg2_exit'] == 'TP2').sum()
        tp2_rate = tp2_hits / n * 100

        results.append({
            'quintile': q + 1,
            'n': n,
            'feat_range': f'{feat_min:.2f} – {feat_max:.2f}',
            'avg_leg2_pts': round(avg_leg2, 2),
            'avg_total_pts': round(avg_total, 2),
            'runner_wr%': round(runner_wr, 1),
            'tp2_rate%': round(tp2_rate, 1),
            'leg2_pnl$': round(leg2_pnl.sum(), 2),
            'total_pnl$': round(total_pnl.sum(), 2),
        })

    return pd.DataFrame(results)


# ============================================================================
# Analysis: Direction Alignment
# ============================================================================

def direction_alignment_analysis(df_features, dollar_per_pt=MNQ_DOLLAR_PER_PT,
                                 commission_per_side=MNQ_COMMISSION):
    """Compare runner performance when HTF SM direction aligns vs opposes entry.

    For each HTF timeframe, splits trades into:
      - Aligned: HTF SM direction matches trade side
      - Opposed: HTF SM direction opposes trade side
      - Neutral: HTF SM is ~0

    Returns summary DataFrame.
    """
    comm_per_leg = commission_per_side * 2
    results = []

    for tf in HTF_TIMEFRAMES:
        sm_dir_col = f'{tf}_sm_dir'
        if sm_dir_col not in df_features.columns:
            continue

        df = df_features.dropna(subset=[sm_dir_col]).copy()
        if len(df) < 20:
            continue

        # Determine if trade direction aligns with HTF SM
        # LONG: aligned if HTF SM > 0. SHORT: aligned if HTF SM < 0.
        trade_dir = df['side'].map({'long': 1, 'short': -1}).values
        htf_dir = df[sm_dir_col].values

        aligned_mask = (trade_dir * htf_dir) > 0
        opposed_mask = (trade_dir * htf_dir) < 0
        neutral_mask = htf_dir == 0

        for label, mask in [('Aligned', aligned_mask),
                            ('Opposed', opposed_mask),
                            ('Neutral', neutral_mask)]:
            group = df[mask]
            n = len(group)
            if n < 5:
                continue

            avg_leg2 = group['leg2_pts'].mean()
            total_pnl = ((group['leg1_pts'] + group['leg2_pts']) * dollar_per_pt
                         - 2 * comm_per_leg).sum()
            runner_wr = (group['leg2_pts'] > 0).mean() * 100
            tp2_rate = (group['leg2_exit'] == 'TP2').mean() * 100

            results.append({
                'timeframe': tf,
                'alignment': label,
                'n': n,
                'avg_leg2_pts': round(avg_leg2, 2),
                'runner_wr%': round(runner_wr, 1),
                'tp2_rate%': round(tp2_rate, 1),
                'total_pnl$': round(total_pnl, 2),
            })

    return pd.DataFrame(results)


# ============================================================================
# Main Diagnostic
# ============================================================================

def run_diagnostic():
    print("=" * 100)
    print("vScalpC Phase 2a — HTF Diagnostic Analysis")
    print("Capturing all higher-timeframe indicators at each vScalpC entry")
    print("=" * 100)

    # --- Load data ---
    df, rsi_curr, rsi_prev, is_len = prepare_vscalpc_data()

    # --- Run baseline vScalpC ---
    print("\nRunning vScalpC baseline (production config)...")
    trades = run_backtest_partial_exit(
        df['Open'].values, df['High'].values,
        df['Low'].values, df['Close'].values,
        df['SM_Net'].values, df.index,
        rsi_curr, rsi_prev,
        rsi_buy=VSCALPA_RSI_BUY, rsi_sell=VSCALPA_RSI_SELL,
        sm_threshold=VSCALPA_SM_THRESHOLD,
        cooldown_bars=VSCALPA_COOLDOWN,
        sl_pts=VSCALPC_SL, tp1_pts=VSCALPC_TP1, tp2_pts=VSCALPC_TP2,
        sl_to_be_after_tp1=VSCALPC_SL_TO_BE,
        be_time_bars=VSCALPC_BE_TIME,
        entry_end_et=VSCALPA_ENTRY_END_ET,
    )

    sc = score_partial_trades(trades)
    print(f"  Baseline: {sc['count']} trades, WR {sc['win_rate']}%, "
          f"PF {sc['pf']}, ${sc['net_dollar']:+.2f}, Sharpe {sc['sharpe']}")
    print(f"  Leg1 exits: {sc['leg1_exits']}")
    print(f"  Leg2 exits: {sc['leg2_exits']}")

    # --- Compute HTF indicators ---
    print("\nComputing HTF indicators across all timeframes...")
    htf_indicators = compute_htf_indicators(df)
    print(f"  Computed {len(htf_indicators)} indicator series across {len(HTF_TIMEFRAMES)} timeframes")

    # --- Session context ---
    print("Computing session context (VWAP, OR, IB)...")
    session_ctx = compute_session_context(df)

    # --- Volatility regime (fixed: prior-day, no look-ahead) ---
    print("Computing volatility regime (prior-day range, no look-ahead)...")
    vol_regime = compute_volatility_regime(df)

    # --- Prior-day ATR ---
    print("Computing prior-day ATR...")
    prior_day_atr = compute_prior_day_atr(df)

    # --- ET minutes for time features ---
    from v10_test_common import compute_et_minutes
    et_mins = compute_et_minutes(df.index)

    # --- Capture features at each entry ---
    print(f"\nCapturing features at {len(trades)} entry points...")
    df_features = capture_entry_features(
        trades, htf_indicators, session_ctx, vol_regime,
        df['SM_Net'].values, df['Close'].values,
        prior_day_atr, et_mins,
    )
    print(f"  Feature matrix: {df_features.shape[0]} trades x {df_features.shape[1]} features")

    # ========================================================================
    # TABLE 1: Correlation with runner leg P&L
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 1: Feature Correlation with Runner Leg P&L (leg2_pts)")
    print(f"{'=' * 100}")

    df_corr = correlation_analysis(df_features, target='leg2_pts')
    print(f"\n{'Feature':>35} {'Corr':>8} {'|Corr|':>8} {'t-stat':>8} {'N':>6}")
    print("-" * 70)
    for _, row in df_corr.head(30).iterrows():
        marker = " ***" if abs(row['corr']) >= 0.15 else (" **" if abs(row['corr']) >= 0.10 else "")
        print(f"{row['feature']:>35} {row['corr']:>8.4f} {row['abs_corr']:>8.4f} "
              f"{row['t_stat']:>8.2f} {row['n_valid']:>6}{marker}")

    # Also check correlation with tp2_hit (binary target, uncensored)
    print(f"\n--- Correlation with tp2_hit (binary: did runner reach TP2?) ---")
    df_corr_tp2 = correlation_analysis(df_features, target='tp2_hit')
    print(f"\n{'Feature':>35} {'Corr':>8} {'|Corr|':>8} {'t-stat':>8} {'N':>6}")
    print("-" * 70)
    for _, row in df_corr_tp2.head(30).iterrows():
        marker = " ***" if abs(row['corr']) >= 0.15 else (" **" if abs(row['corr']) >= 0.10 else "")
        print(f"{row['feature']:>35} {row['corr']:>8.4f} {row['abs_corr']:>8.4f} "
              f"{row['t_stat']:>8.2f} {row['n_valid']:>6}{marker}")

    # Also check correlation with total_pts
    print(f"\n--- Also checking total_pts (both legs combined) ---")
    df_corr_total = correlation_analysis(df_features, target='total_pts')
    print(f"\n{'Feature':>35} {'Corr':>8} {'|Corr|':>8} {'t-stat':>8} {'N':>6}")
    print("-" * 70)
    for _, row in df_corr_total.head(15).iterrows():
        marker = " ***" if abs(row['corr']) >= 0.15 else (" **" if abs(row['corr']) >= 0.10 else "")
        print(f"{row['feature']:>35} {row['corr']:>8.4f} {row['abs_corr']:>8.4f} "
              f"{row['t_stat']:>8.2f} {row['n_valid']:>6}{marker}")

    # ========================================================================
    # TABLE 2: Quintile Analysis for Top Features
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 2: Quintile Analysis — Top Features by |Correlation|")
    print(f"{'=' * 100}")

    top_features = df_corr.head(10)['feature'].tolist()
    for feat in top_features:
        df_quint = quintile_analysis(df_features, feat)
        if df_quint is None:
            continue

        print(f"\n  --- {feat} (corr={df_corr[df_corr['feature']==feat]['corr'].values[0]:.4f}) ---")
        print(f"  {'Q':>3} {'N':>4} {'Range':>25} {'Leg2 Pts':>10} {'Total Pts':>11} "
              f"{'Runner WR%':>11} {'TP2 Rate%':>10} {'Leg2 P&L$':>10} {'Total P&L$':>11}")
        print(f"  " + "-" * 105)

        for _, row in df_quint.iterrows():
            print(f"  {row['quintile']:>3} {row['n']:>4} {row['feat_range']:>25} "
                  f"{row['avg_leg2_pts']:>10.2f} {row['avg_total_pts']:>11.2f} "
                  f"{row['runner_wr%']:>11.1f} {row['tp2_rate%']:>10.1f} "
                  f"${row['leg2_pnl$']:>9.2f} ${row['total_pnl$']:>10.2f}")

    # ========================================================================
    # TABLE 3: Direction Alignment Analysis
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 3: HTF SM Direction Alignment — Does matching HTF direction improve runner?")
    print(f"{'=' * 100}")

    df_align = direction_alignment_analysis(df_features)
    if len(df_align) > 0:
        print(f"\n  {'TF':>6} {'Alignment':>10} {'N':>5} {'Leg2 Pts':>10} "
              f"{'Runner WR%':>11} {'TP2 Rate%':>10} {'Total P&L$':>11}")
        print(f"  " + "-" * 75)
        for _, row in df_align.iterrows():
            print(f"  {row['timeframe']:>6} {row['alignment']:>10} {row['n']:>5} "
                  f"{row['avg_leg2_pts']:>10.2f} {row['runner_wr%']:>11.1f} "
                  f"{row['tp2_rate%']:>10.1f} ${row['total_pnl$']:>10.2f}")
    else:
        print("  No alignment data available")

    # ========================================================================
    # TABLE 4: Volatility Regime Analysis
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 4: Volatility Regime — Does vol regime affect runner success?")
    print(f"{'=' * 100}")

    comm_per_leg = MNQ_COMMISSION * 2
    df_vol = df_features.dropna(subset=['vol_regime'])
    if len(df_vol) > 0:
        print(f"\n  {'Regime':>8} {'N':>5} {'Avg Leg2':>10} {'Runner WR%':>11} "
              f"{'TP2 Rate%':>10} {'Total P&L$':>11}")
        print(f"  " + "-" * 60)
        regime_names = {0: 'Low', 1: 'Medium', 2: 'High'}
        for regime in sorted(df_vol['vol_regime'].unique()):
            group = df_vol[df_vol['vol_regime'] == regime]
            n = len(group)
            avg_leg2 = group['leg2_pts'].mean()
            runner_wr = (group['leg2_pts'] > 0).mean() * 100
            tp2_rate = (group['leg2_exit'] == 'TP2').mean() * 100
            total_pnl = ((group['leg1_pts'] + group['leg2_pts']) * MNQ_DOLLAR_PER_PT
                         - 2 * comm_per_leg).sum()
            label = regime_names.get(regime, f'{regime}')
            print(f"  {label:>8} {n:>5} {avg_leg2:>10.2f} {runner_wr:>11.1f} "
                  f"{tp2_rate:>10.1f} ${total_pnl:>10.2f}")

    # ========================================================================
    # TABLE 5: Session Context
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 5: Session Context — VWAP and Opening Range")
    print(f"{'=' * 100}")

    # Above/below VWAP
    if 'session_above_vwap' in df_features.columns:
        df_vwap = df_features.dropna(subset=['session_above_vwap'])
        if len(df_vwap) > 0:
            print(f"\n  --- VWAP Position ---")
            for label, mask_fn in [('Above VWAP', lambda x: x == True),
                                   ('Below VWAP', lambda x: x == False)]:
                # Separate longs and shorts
                for side in ['long', 'short']:
                    group = df_vwap[(mask_fn(df_vwap['session_above_vwap'])) &
                                   (df_vwap['side'] == side)]
                    if len(group) < 5:
                        continue
                    avg_leg2 = group['leg2_pts'].mean()
                    runner_wr = (group['leg2_pts'] > 0).mean() * 100
                    tp2_rate = (group['leg2_exit'] == 'TP2').mean() * 100
                    print(f"  {label + ' ' + side.upper():>25}: N={len(group):>3}, "
                          f"Avg Leg2={avg_leg2:>6.2f}pt, "
                          f"Runner WR={runner_wr:>5.1f}%, TP2 Rate={tp2_rate:>5.1f}%")

    # ========================================================================
    # TABLE 6: IS/OOS Stability of Top Correlations
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("TABLE 6: IS/OOS Stability of Top Correlations")
    print("  Do the same features predict runner success in both halves?")
    print(f"{'=' * 100}")

    # Split features by IS/OOS based on entry_idx
    df_is = df_features[df_features['entry_idx'] < is_len]
    df_oos = df_features[df_features['entry_idx'] >= is_len]
    print(f"\n  IS trades: {len(df_is)}, OOS trades: {len(df_oos)}")

    def compute_is_oos_stability(feat, target):
        """Returns (full_r, is_r, oos_r, stable_str)."""
        corr_df = df_corr if target == 'leg2_pts' else df_corr_tp2
        match = corr_df[corr_df['feature'] == feat]
        full_r = match['corr'].values[0] if len(match) > 0 else np.nan

        is_valid = df_is[[feat, target]].dropna()
        is_r = (np.corrcoef(is_valid[feat], is_valid[target])[0, 1]
                if len(is_valid) >= 20 and is_valid[feat].std() > 0 else np.nan)

        oos_valid = df_oos[[feat, target]].dropna()
        oos_r = (np.corrcoef(oos_valid[feat], oos_valid[target])[0, 1]
                 if len(oos_valid) >= 20 and oos_valid[feat].std() > 0 else np.nan)

        stable = ""
        if not np.isnan(is_r) and not np.isnan(oos_r):
            stable = ("YES" if np.sign(is_r) == np.sign(oos_r)
                      and abs(is_r) > 0.05 and abs(oos_r) > 0.05 else "NO")

        return full_r, is_r, oos_r, stable

    # Combine top features from both leg2_pts and tp2_hit analyses
    top_feats_leg2 = set(df_corr.head(15)['feature'].tolist())
    top_feats_tp2 = set(df_corr_tp2.head(15)['feature'].tolist())
    all_top_feats = sorted(top_feats_leg2 | top_feats_tp2,
                           key=lambda f: max(
                               df_corr[df_corr['feature']==f]['abs_corr'].values[0]
                               if len(df_corr[df_corr['feature']==f]) > 0 else 0,
                               df_corr_tp2[df_corr_tp2['feature']==f]['abs_corr'].values[0]
                               if len(df_corr_tp2[df_corr_tp2['feature']==f]) > 0 else 0,
                           ), reverse=True)

    print(f"\n  {'Feature':>35} {'Target':>10} {'FULL':>8} {'IS':>8} {'OOS':>8} {'Stable':>7}")
    print(f"  " + "-" * 85)

    for feat in all_top_feats:
        for target in ['leg2_pts', 'tp2_hit']:
            full_r, is_r, oos_r, stable = compute_is_oos_stability(feat, target)
            if np.isnan(full_r):
                continue
            lbl = 'leg2' if target == 'leg2_pts' else 'tp2'
            print(f"  {feat:>35} {lbl:>10} {full_r:>8.4f} {is_r:>8.4f} {oos_r:>8.4f} {stable:>7}")

    # ========================================================================
    # SUMMARY: Promising Signals
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("SUMMARY: Promising Signals for Phase 2b Filter Sweeps")
    print(f"{'=' * 100}")

    # Identify features with |corr| >= 0.08 AND stable across IS/OOS
    # Check against both leg2_pts and tp2_hit targets
    promising = []
    seen_feats = set()
    for corr_df_cur, target in [(df_corr, 'leg2_pts'), (df_corr_tp2, 'tp2_hit')]:
        for feat in corr_df_cur['feature'].values:
            if feat in seen_feats:
                continue
            full_r = corr_df_cur[corr_df_cur['feature'] == feat]['corr'].values[0]
            if abs(full_r) < 0.08:
                continue

            full_r, is_r, oos_r, stable = compute_is_oos_stability(feat, target)
            if stable == "YES":
                promising.append({
                    'feature': feat,
                    'target': target,
                    'full_corr': round(full_r, 4),
                    'is_corr': round(is_r, 4),
                    'oos_corr': round(oos_r, 4),
                })
                seen_feats.add(feat)

    if promising:
        print(f"\n  Features with |corr| >= 0.08 AND stable IS/OOS:")
        for p in promising:
            tgt = 'leg2' if p['target'] == 'leg2_pts' else 'tp2'
            print(f"    {p['feature']:>35} ({tgt})  full={p['full_corr']:+.4f}  "
                  f"IS={p['is_corr']:+.4f}  OOS={p['oos_corr']:+.4f}")
        print(f"\n  --> These {len(promising)} features are candidates for filter sweep scripts")
    else:
        print("\n  No features meet both |corr| >= 0.08 AND IS/OOS stability criteria.")
        print("  The strategy may be robust across conditions — filters may not help.")

    # Save feature matrix for further analysis
    output_dir = _STRAT_DIR.parent / "results" / "htf_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vscalpc_entry_features.csv"
    df_features.to_csv(csv_path, index=False)
    print(f"\n  Feature matrix saved to: {csv_path}")

    corr_path = output_dir / "correlation_with_leg2_pts.csv"
    df_corr.to_csv(corr_path, index=False)
    print(f"  Correlation table saved to: {corr_path}")

    print(f"\n{'=' * 100}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    run_diagnostic()
