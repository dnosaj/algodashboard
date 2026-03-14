# Cross-Instrument Analysis & Next Steps

## MES v11 Result (Feb 14, 2026)
v11 optimized params SM(10/12/200/100) FAIL on MES:
- v11 on MES: PF 1.080, +$388, WR 53.2%, Sharpe 0.37
- v9.4 on MES: PF 1.351, +$1,653, WR 57.9%, Sharpe 1.50
- v9.4 defaults (20/12/400/255) are 4.3x more profitable on MES
- SL doesn't matter on MES (SL=0 and SL=50 identical for v9.4)
- Script: `strategies/v11_mes_backtest.py`

## Pending Investigation Options

### Option 1: Walk-Forward Validation on MNQ [DONE - PASS]
v11 MNQ OOS PF 1.713 (BETTER than IS 1.634). Not overfit. 6/7 months profitable.
Script: `strategies/v11_walk_forward.py`

### Option 2: MES Parameter Sweep [DONE - ALL SWEEP CONFIGS FAILED]
Train/test split sweep (Train: Aug-Nov, Test: Dec-Feb), 6,090 combos total.
- Phase 1: 150 SM combos, ALL profitable on train. index_period=15 dominated.
- Phase 2: 5,600 RSI combos, 3,634 profitable on train. RSI 12/65/35 topped.
- Phase 3: 240 ATR/SL combos. SL=15 was best on train.
- **Phase 4: ALL 20 top configs FAILED test validation (PF < 1.0, ~55% degradation)**
- v9.4 baseline: Train PF 1.16, Test PF 1.686 (PASSED, negative degradation)
- v11 MNQ-opt: Train PF 1.017, Test PF 1.157 (marginal pass)
- **Conclusion: Sweep overfits MES every time. v9.4 defaults genuinely optimal for MES.**
- Script: `strategies/v11_mes_param_sweep.py`

### Option 3: Intermediate Params Test (UNNECESSARY)
v9.4 defaults already validated as robust for MES. No need for compromise params.

### Option 4: Granular Isolation (UNNECESSARY)
Sweep results show the issue is systemic — MES has different microstructure, not just one param.

## Final Conclusion (Feb 14, 2026)
**Use different params per instrument — both validated with held-out data:**
- MNQ: v11 SM(10/12/200/100) RSI(8/60/40) CD=20 SL=50 — Walk-forward PASS (OOS PF 1.713)
- MES: v9.4 SM(20/12/400/255) RSI(10/55/45) CD=15 SL=0 — Train/test PASS (Test PF 1.686)
- MES has fundamentally different microstructure (S&P 500 vs Nasdaq 100)
- "Slow" SM defaults suit MES's lower volatility
- Aggressive RSI levels (65/35) overfit MES training data every time
