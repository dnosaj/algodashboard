# Structure-Based Exit Research — Results (Mar 13, 2026)

## Vision
Use recent swing highs/lows as adaptive exit targets on runner legs instead of fixed TP.
Jason manually exited MES at +48 seeing a near-term swing low — this inspired the research.
Data collected for future Observation Agent calibration.

## What Was Tested
- **Swing detection**: Pivot (N-bar fractal) and Donchian (rolling N-bar high/low)
- **Parameters swept**: lookback [10,15,20,30,50], pivot_right [1,2,3], buffer [0,2,5], min_profit [0,5,10], max_tp_cap [0,30]
- **360 combos per strategy**, IS/OOS validation on top 20

## Results

### MES v2 Runner (TP2=20 → structure exit)
- **Best config**: Donchian LB=30, Buffer=2pts, Cap=30
- IS PF: 1.572→1.488 (-5.3%), **OOS PF: 1.221→1.368 (+12.0%)**
- IS Sharpe: 2.549→2.105, **OOS Sharpe: 1.110→1.780 (+60%)**
- **OOS MaxDD: -$1,908→-$1,020 (-47%)**
- IS/OOS PF ratio: 91.9% (STABLE)
- Total P&L lower by $804 but risk-adjusted metrics massively better
- 170 trades improve, 200 degrade — improvements are larger ($225 saves vs $100 haircuts)
- Structure fires on 92% of runners
- Script: `structure_exit_sweep.py`

### vScalpC Runner (TP2=25 → structure exit)
- **Best config**: Pivot LB=50, PR=2, Buffer=2pts, no cap
- IS PF: 1.413→1.426 (+0.9%), **OOS PF: 1.334→1.388 (+4.0%)**
- IS Sharpe: 2.100→2.083, **OOS Sharpe: 1.724→1.876 (+8.8%)**
- **OOS MaxDD: -$1,586→-$1,394 (-12%)**
- IS/OOS PF ratio: 97.3% (rock solid)
- Total P&L: +$5,693→+$6,099 (+$406) — better on ALL metrics
- 132 improve, 102 degrade, 225 unchanged. Net +$1,261
- **19 of 20 top configs pass IS/OOS**
- Pivot >> Donchian for MNQ. LB=50 dominates. Buffer=2 sweet spot.
- Script: `structure_exit_sweep_vscalpc.py`

### vWinners Revival (SM flip → structure exit)
- **FAIL**: Every config has FULL-period PF < 1.0 (best: 0.960)
- Problem is the entry signal (SM_T=0.15), not the exit
- IS/OOS split has flipped from memory (dataset extended since original analysis)
- vWinners stays shelved. Confirmed dead.
- Script: `structure_exit_sweep_vwinners.py`

## Key Patterns
- Longer lookbacks (30-50 bars) capture meaningful structure, not noise
- Small buffer (2pts) is the sweet spot
- Pivot type preferred for MNQ, Donchian for MES
- min_profit parameter doesn't matter much
- Neighbor stability is good — broad regions pass, not isolated peaks

## Files Created
- `backtesting_engine/strategies/structure_exit_common.py` — swing detection + backtest functions
- `backtesting_engine/strategies/structure_exit_sweep.py` — MES sweep
- `backtesting_engine/strategies/structure_exit_sweep_vscalpc.py` — vScalpC sweep
- `backtesting_engine/strategies/structure_exit_sweep_vwinners.py` — vWinners sweep

## Session Context (Mar 13)
Also completed today:
- MES session timing: removed 15:30 EOD (never triggered, hurt live), added 14:15 entry cutoff (14:30+ entries net negative). Config updated.
- Prior-day level gate narrowed to VPOC+VAL only (H/L+VAH block profitable breakouts)
- Cooldown sweep confirmed CD=20/25 optimal
- CF engine session JSON loading implemented
- db_logger "unknown" gate_type fixed

## Next Decision
Jason wants to review the code for look-ahead bias before deciding on implementation.
Options: implement on runners, save as Observation Agent data, or both.
