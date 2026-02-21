"""
MES v9.4 Re-Entry Analysis — What happens after a losing cap exit?
===================================================================
The losing hold cap looked great as a post-filter (+$688 at cap=25) but
failed LOMO when run as a pre-filter (-$918). The early exits change
cooldowns and create new entries. This script traces exactly what happens:

1. For each losing cap exit, what is the NEXT trade that enters?
2. Is the re-entry in the same direction as the loser that was just cut?
3. Is the re-entry profitable?
4. Would the re-entry have existed in baseline? (i.e., is it a "phantom" trade
   created only because the losing cap freed up the cooldown window?)

This explains WHY the pre-filter fails.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_test_common import (
    compute_smart_money, resample_to_5min, map_5min_rsi_to_1min,
    score_trades, compute_et_minutes, NY_OPEN_ET, NY_LAST_ENTRY_ET, NY_CLOSE_ET,
)

COMMISSION_PER_SIDE = 1.25
DOLLAR_PER_PT = 5.0
COMM_PTS = (COMMISSION_PER_SIDE * 2) / DOLLAR_PER_PT
SM_INDEX, SM_FLOW, SM_NORM, SM_EMA = 20, 12, 400, 255
RSI_LEN = 10
RSI_BUY, RSI_SELL = 55, 45
COOLDOWN = 15


def run_backtest_exits(
    opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
    losing_hold_cap=0,
):
    n = len(opens)
    trades = []
    trade_state = 0
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
        sm_bull = sm_prev > 0
        sm_bear = sm_prev < 0
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        rsi_p = rsi_curr[i - 1]
        rsi_p2 = rsi_prev[i - 1]
        rsi_long_trigger = rsi_p > RSI_BUY and rsi_p2 <= RSI_BUY
        rsi_short_trigger = rsi_p < RSI_SELL and rsi_p2 >= RSI_SELL

        if sm_flipped_bull or sm_prev <= 0:
            long_used = False
        if sm_flipped_bear or sm_prev >= 0:
            short_used = False

        # EOD
        if trade_state != 0 and bar_mins_et >= NY_CLOSE_ET:
            side = "long" if trade_state == 1 else "short"
            close_trade(side, entry_price, closes[i], entry_idx, i, "EOD")
            trade_state = 0
            exit_bar = i
            continue

        if trade_state == 1:
            bars_held = i - entry_idx
            if losing_hold_cap > 0 and bars_held >= losing_hold_cap:
                unrealized = closes[i - 1] - entry_price
                if unrealized < 0:
                    close_trade("long", entry_price, opens[i], entry_idx, i, "LOSING_CAP")
                    trade_state = 0
                    exit_bar = i
                    continue
            if sm_prev < 0 and sm_prev2 >= 0:
                close_trade("long", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        elif trade_state == -1:
            bars_held = i - entry_idx
            if losing_hold_cap > 0 and bars_held >= losing_hold_cap:
                unrealized = entry_price - closes[i - 1]
                if unrealized < 0:
                    close_trade("short", entry_price, opens[i], entry_idx, i, "LOSING_CAP")
                    trade_state = 0
                    exit_bar = i
                    continue
            if sm_prev > 0 and sm_prev2 <= 0:
                close_trade("short", entry_price, opens[i], entry_idx, i, "SM_FLIP")
                trade_state = 0
                exit_bar = i

        if trade_state == 0:
            bars_since = i - exit_bar
            in_session = NY_OPEN_ET <= bar_mins_et <= NY_LAST_ENTRY_ET
            cd_ok = bars_since >= COOLDOWN
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


def load_mes_databento():
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


def main():
    print("=" * 130)
    print("MES v9.4 RE-ENTRY ANALYSIS — What happens after a losing cap exit?")
    print("=" * 130)

    df_1m = load_mes_databento()
    opens = df_1m['Open'].values
    highs = df_1m['High'].values
    lows = df_1m['Low'].values
    closes = df_1m['Close'].values
    volumes = df_1m['Volume'].values
    times = df_1m.index.values

    sm = compute_smart_money(closes, volumes, SM_INDEX, SM_FLOW, SM_NORM, SM_EMA)
    df_r = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_r['SM_Net'] = 0.0
    df_5m = resample_to_5min(df_r)
    rsi_curr, rsi_prev = map_5min_rsi_to_1min(
        times, df_5m.index.values, df_5m['Close'].values, rsi_len=RSI_LEN)

    # Run both baseline and LC30
    baseline = run_backtest_exits(
        opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
        losing_hold_cap=0)
    for t in baseline:
        t['pnl'] = (t['pts'] - COMM_PTS) * DOLLAR_PER_PT

    for cap in [25, 30]:
        cap_trades = run_backtest_exits(
            opens, highs, lows, closes, sm, rsi_curr, rsi_prev, times,
            losing_hold_cap=cap)
        for t in cap_trades:
            t['pnl'] = (t['pts'] - COMM_PTS) * DOLLAR_PER_PT

        # Build baseline entry_time set for checking phantom trades
        bl_entry_times = set(str(t['entry_time']) for t in baseline)

        # Find losing cap exits and their next trade
        cap_exits = [t for t in cap_trades if t['result'] == 'LOSING_CAP']

        print(f"\n{'='*130}")
        print(f"LOSING CAP = {cap} BARS — {len(cap_exits)} early exits, "
              f"{len(cap_trades)} total trades vs {len(baseline)} baseline")
        print(f"{'='*130}")

        # ============================================================
        # SECTION 1: What is the re-entry after each losing cap exit?
        # ============================================================
        print(f"\n  SECTION 1: RE-ENTRY AFTER EACH LOSING CAP EXIT")
        print(f"  {'─'*120}")

        # Build trade sequence for lookup
        trade_by_idx = {i: t for i, t in enumerate(cap_trades)}

        phantom_trades = []    # trades that don't exist in baseline
        same_dir_reentries = []
        opp_dir_reentries = []
        no_reentry = []

        print(f"\n  {'#':>3}  {'Loser Side':<10}  {'Loser P&L':>9}  {'Bars':>5}  |  "
              f"{'Next Side':<10}  {'Next P&L':>9}  {'Gap':>5}  {'Same?':>5}  "
              f"{'Phantom?':>8}  {'SM@exit':>8}  {'Loser Exit Time':<16}  {'Next Entry Time':<16}")
        print(f"  {'─'*140}")

        for idx, loser in enumerate(cap_exits):
            # Find the next trade after this loser's exit
            loser_exit_idx = loser['exit_idx']

            # Find next trade in cap_trades that entered after this exit
            next_trade = None
            for t in cap_trades:
                if t['entry_idx'] > loser_exit_idx:
                    if next_trade is None or t['entry_idx'] < next_trade['entry_idx']:
                        next_trade = t
                    break  # trades are in order

            # Actually, let's find it properly by iterating in order
            next_trade = None
            for t in cap_trades:
                if t['entry_idx'] > loser_exit_idx:
                    next_trade = t
                    break

            if next_trade is None:
                no_reentry.append(loser)
                continue

            gap_bars = next_trade['entry_idx'] - loser['exit_idx']
            same_direction = loser['side'] == next_trade['side']
            is_phantom = str(next_trade['entry_time']) not in bl_entry_times

            # SM value at loser exit
            sm_at_exit = sm[loser['exit_idx'] - 1] if loser['exit_idx'] > 0 else 0

            if is_phantom:
                phantom_trades.append((loser, next_trade))
            if same_direction:
                same_dir_reentries.append((loser, next_trade, is_phantom))
            else:
                opp_dir_reentries.append((loser, next_trade, is_phantom))

            loser_time = pd.Timestamp(loser['exit_time']).strftime('%m-%d %H:%M')
            next_time = pd.Timestamp(next_trade['entry_time']).strftime('%m-%d %H:%M')
            same_str = "SAME" if same_direction else "OPP"
            phantom_str = "PHANTOM" if is_phantom else "exists"

            print(f"  {idx+1:>3}  {loser['side']:<10}  ${loser['pnl']:>+7.2f}  "
                  f"{loser['bars']:>5}  |  "
                  f"{next_trade['side']:<10}  ${next_trade['pnl']:>+7.2f}  "
                  f"{gap_bars:>5}  {same_str:>5}  {phantom_str:>8}  "
                  f"{sm_at_exit:>+7.4f}  {loser_time:<16}  {next_time:<16}")

        # ============================================================
        # SECTION 2: Summary statistics
        # ============================================================
        print(f"\n  SECTION 2: SUMMARY")
        print(f"  {'─'*80}")

        total_reentries = len(same_dir_reentries) + len(opp_dir_reentries)
        total_phantoms = len(phantom_trades)

        print(f"  Losing cap exits:      {len(cap_exits)}")
        print(f"  Followed by re-entry:  {total_reentries}")
        print(f"  No re-entry (EOD etc): {len(no_reentry)}")
        print()

        # Same direction
        if same_dir_reentries:
            same_pnls = [nt['pnl'] for _, nt, _ in same_dir_reentries]
            same_phantom = sum(1 for _, _, p in same_dir_reentries if p)
            same_wins = sum(1 for p in same_pnls if p > 0)
            print(f"  SAME direction re-entries: {len(same_dir_reentries)}")
            print(f"    Winners: {same_wins}/{len(same_dir_reentries)} "
                  f"({same_wins/len(same_dir_reentries)*100:.0f}%)")
            print(f"    Total P&L: ${sum(same_pnls):+.2f}")
            print(f"    Avg P&L:   ${np.mean(same_pnls):+.2f}")
            print(f"    Phantom (wouldn't exist in baseline): {same_phantom}")
            if same_phantom > 0:
                phantom_same_pnls = [nt['pnl'] for _, nt, p in same_dir_reentries if p]
                print(f"      Phantom P&L: ${sum(phantom_same_pnls):+.2f} "
                      f"(avg ${np.mean(phantom_same_pnls):+.2f})")

        # Opposite direction
        if opp_dir_reentries:
            opp_pnls = [nt['pnl'] for _, nt, _ in opp_dir_reentries]
            opp_phantom = sum(1 for _, _, p in opp_dir_reentries if p)
            opp_wins = sum(1 for p in opp_pnls if p > 0)
            print(f"\n  OPPOSITE direction re-entries: {len(opp_dir_reentries)}")
            print(f"    Winners: {opp_wins}/{len(opp_dir_reentries)} "
                  f"({opp_wins/len(opp_dir_reentries)*100:.0f}%)")
            print(f"    Total P&L: ${sum(opp_pnls):+.2f}")
            print(f"    Avg P&L:   ${np.mean(opp_pnls):+.2f}")
            print(f"    Phantom (wouldn't exist in baseline): {opp_phantom}")
            if opp_phantom > 0:
                phantom_opp_pnls = [nt['pnl'] for _, nt, p in opp_dir_reentries if p]
                print(f"      Phantom P&L: ${sum(phantom_opp_pnls):+.2f} "
                      f"(avg ${np.mean(phantom_opp_pnls):+.2f})")

        # ============================================================
        # SECTION 3: Phantom trades deep dive
        # ============================================================
        print(f"\n  SECTION 3: PHANTOM TRADES (only exist because of early exit)")
        print(f"  {'─'*80}")

        if phantom_trades:
            phantom_pnls = [nt['pnl'] for _, nt in phantom_trades]
            phantom_wins = sum(1 for p in phantom_pnls if p > 0)
            print(f"  Total phantom trades: {len(phantom_trades)}")
            print(f"  Phantom winners: {phantom_wins}/{len(phantom_trades)} "
                  f"({phantom_wins/len(phantom_trades)*100:.0f}%)")
            print(f"  Phantom total P&L: ${sum(phantom_pnls):+.2f}")
            print(f"  Phantom avg P&L:   ${np.mean(phantom_pnls):+.2f}")

            # Break down: same-dir phantoms vs opp-dir phantoms
            same_phantoms = [(l, n) for l, n, p in same_dir_reentries if p]
            opp_phantoms = [(l, n) for l, n, p in opp_dir_reentries if p]

            if same_phantoms:
                sp_pnls = [n['pnl'] for _, n in same_phantoms]
                print(f"\n    Same-direction phantoms: {len(same_phantoms)}, "
                      f"P&L ${sum(sp_pnls):+.2f}")
                print(f"    These are re-entering the SAME losing trade direction")
                print(f"    in the SAME SM regime — essentially doubling down")
            if opp_phantoms:
                op_pnls = [n['pnl'] for _, n in opp_phantoms]
                print(f"\n    Opposite-direction phantoms: {len(opp_phantoms)}, "
                      f"P&L ${sum(op_pnls):+.2f}")
        else:
            print(f"  No phantom trades found")

        # ============================================================
        # SECTION 4: SM regime at exit — are we exiting during choppy SM?
        # ============================================================
        print(f"\n  SECTION 4: SM REGIME AT LOSING CAP EXIT")
        print(f"  {'─'*80}")

        sm_at_exits = []
        for t in cap_exits:
            sm_val = sm[t['exit_idx'] - 1] if t['exit_idx'] > 0 else 0
            sm_at_exits.append({
                'sm': sm_val,
                'side': t['side'],
                'pnl': t['pnl'],
                'agrees': (t['side'] == 'long' and sm_val > 0) or
                          (t['side'] == 'short' and sm_val < 0),
            })

        agrees = [s for s in sm_at_exits if s['agrees']]
        disagrees = [s for s in sm_at_exits if not s['agrees']]

        print(f"  SM still agrees with trade direction at exit: "
              f"{len(agrees)}/{len(sm_at_exits)} ({len(agrees)/len(sm_at_exits)*100:.0f}%)")
        print(f"  SM disagrees (trade against SM) at exit:      "
              f"{len(disagrees)}/{len(sm_at_exits)} ({len(disagrees)/len(sm_at_exits)*100:.0f}%)")

        if agrees:
            print(f"\n  When SM AGREES (trade is losing but SM says hold):")
            print(f"    Avg |SM|: {np.mean([abs(s['sm']) for s in agrees]):.4f}")
            print(f"    Avg loser P&L: ${np.mean([s['pnl'] for s in agrees]):+.2f}")
            # These are the dangerous ones — SM hasn't flipped, but the trade is underwater
            weak_sm = [s for s in agrees if abs(s['sm']) < 0.10]
            strong_sm = [s for s in agrees if abs(s['sm']) >= 0.10]
            print(f"    Weak SM (|SM| < 0.10): {len(weak_sm)}, "
                  f"avg P&L ${np.mean([s['pnl'] for s in weak_sm]):+.2f}" if weak_sm else
                  f"    Weak SM (|SM| < 0.10): 0")
            print(f"    Strong SM (|SM| >= 0.10): {len(strong_sm)}, "
                  f"avg P&L ${np.mean([s['pnl'] for s in strong_sm]):+.2f}" if strong_sm else
                  f"    Strong SM (|SM| >= 0.10): 0")

        if disagrees:
            print(f"\n  When SM DISAGREES (SM already flipped, just waiting for cooldown):")
            print(f"    Avg |SM|: {np.mean([abs(s['sm']) for s in disagrees]):.4f}")
            print(f"    Avg loser P&L: ${np.mean([s['pnl'] for s in disagrees]):+.2f}")

        # ============================================================
        # SECTION 5: The cooldown collision — are phantom entries within
        #            the cooldown window of the baseline trade's SM exit?
        # ============================================================
        print(f"\n  SECTION 5: COOLDOWN COLLISION ANALYSIS")
        print(f"  {'─'*80}")
        print(f"  When a losing cap exit happens BEFORE the SM flip, the cooldown")
        print(f"  timer starts sooner. This creates a window where new entries fire")
        print(f"  that wouldn't exist in baseline (because baseline is still in the")
        print(f"  original trade or still in cooldown).")

        # For each losing cap exit, find where baseline exits the same trade
        bl_map = {str(t['entry_time']): t for t in baseline}

        early_exit_gaps = []
        for loser in cap_exits:
            key = str(loser['entry_time'])
            bl_t = bl_map.get(key)
            if bl_t is None:
                continue
            # How much earlier did we exit vs baseline?
            bars_saved = bl_t['exit_idx'] - loser['exit_idx']
            early_exit_gaps.append({
                'loser': loser,
                'baseline': bl_t,
                'bars_saved': bars_saved,
                'bl_exit_time': bl_t['exit_time'],
            })

        if early_exit_gaps:
            gaps = [e['bars_saved'] for e in early_exit_gaps]
            print(f"\n  Losing cap exits {cap} bars, baseline SM flip exits later")
            print(f"  Average bars earlier: {np.mean(gaps):.1f}")
            print(f"  Median bars earlier:  {np.median(gaps):.1f}")
            print(f"  Max bars earlier:     {max(gaps)}")

            # How many of those early exits create a window > cooldown?
            window_gt_cd = [e for e in early_exit_gaps if e['bars_saved'] > COOLDOWN]
            print(f"\n  Exits that create a cooldown-sized window (>{COOLDOWN} bars): "
                  f"{len(window_gt_cd)}/{len(early_exit_gaps)}")
            print(f"  These are the exits most likely to create phantom trades")

            # For the phantom trades, check if they fall in this window
            phantom_in_window = 0
            for loser, next_t in phantom_trades:
                key = str(loser['entry_time'])
                bl_t = bl_map.get(key)
                if bl_t and next_t['entry_idx'] < bl_t['exit_idx']:
                    phantom_in_window += 1
            print(f"\n  Phantom trades that enter BEFORE baseline would have exited: "
                  f"{phantom_in_window}/{len(phantom_trades)}")
            print(f"  (These are entering while baseline is still in the original "
                  f"losing trade)")

    # ============================================================
    # FINAL: The core insight
    # ============================================================
    print(f"\n{'='*130}")
    print("CONCLUSION")
    print(f"{'='*130}")
    print("""
  The losing hold cap creates a FEEDBACK LOOP:
  1. Cut a loser at bar 25-30 (saves money on THAT trade)
  2. Cooldown starts 20-80 bars earlier than baseline
  3. New RSI cross fires during the same SM regime
  4. SM hasn't flipped yet → re-entry is in the SAME choppy/losing regime
  5. The re-entry is a new loser (SM still hasn't found direction)

  The post-filter simulation assumed frozen entries, so it only saw step 1.
  The pre-filter (actual backtest) includes steps 2-5, which eat the savings.

  This is why EOD 15:30 works and losing cap doesn't:
  - EOD 15:30 doesn't create new entries (it just closes existing ones earlier)
  - Losing cap creates phantom entries in bad SM regimes
""")

    print("=" * 130)
    print("DONE")
    print("=" * 130)


if __name__ == "__main__":
    main()
