# Interesting Indicators Library

Reference library of indicators analyzed for potential use in our trading system.
Each file contains source code, statistical analysis, fit assessment, and test results.

## Status Key
- **Adopted**: Active in live engine
- **Rejected**: Tested, failed validation
- **Tested**: Backtested but not adopted (passed but superseded or deferred)
- **Untested**: Analyzed but not yet backtested
- **None**: Disqualified without testing (e.g., repainting)

## Index

| Indicator | File | Author | Status | Potential Fit | Priority |
|-----------|------|--------|--------|---------------|----------|
| Leledc Exhaustion V4 | `leledc_exhaustion.md` | Joy_Bangla | **Adopted** (mq=9, all MNQ) | Entry gate | -- |
| Squeeze Momentum | `squeeze_momentum.md` | LazyBear | **Rejected** (Mar 6) | Entry gate | -- |
| VWAP Deviation Z-Score | `vwap_deviation.md` | BackQuant | **Rejected** (Mar 6) | Entry gate | -- |
| Initial Balance | `initial_balance.md` | PtGambler/LuxAlgo | **Tested** (passed, superseded) | Entry gate | Low |
| ICT Concepts (SMC) | `ict_concepts.md` | LuxAlgo/UAlgo/Huddlestone | **Tested** (forensics) | OB gate candidate, London H/L gate | **HIGH** |
| Periodic Volume Profile | `periodic_volume_profile.md` | TradingView built-in | **Tested** (forensics) | Weekly VAL gate candidate | Medium |
| Z-Score Probability | `zscore_probability.md` | steversteves | **Untested** | Entry gate (extension) | Medium |
| Fair Value Gap | `fair_value_gap.md` | LuxAlgo | **Untested** | Entry gate | Low |
| Exhaustion Signal | `exhaustion_signal.md` | ChartingCycles | **Adopted** (via Leledc) | Entry gate | -- |
| Smart Money Concepts | `smc_luxalgo.md` | LuxAlgo | **Untested** | Entry confirmation | Low |
| Market Structure CHoCH | `market_structure_choch.md` | mickes/LuxAlgo | **Untested** | Directional filter | Low |
| Williams %R Exhaustion | `williams_r_exhaustion.md` | upslidedown | **Untested** | Entry gate | Low |
| Divergence Multi | `divergence_multi.md` | LonesomeTheBlue | **Untested** | Entry gate | Low |
| Nadaraya-Watson Envelope | `nadaraya_watson.md` | LuxAlgo | **None** (repaints) | -- | None |
| Liquidity Sweep | `liquidity_sweep.md` | various | **Untested** | Exit signal | Low |
| Chandelier Exit | `chandelier_exit.md` | everget | **Untested** | Trailing stop | Very Low |

## Patterns Observed

- **Exhaustion/extension gates work** (Leledc adopted, ADR adopted). Our momentum entries benefit from "don't chase" filters.
- **Oscillator-based gates fail** (Squeeze, VWAP Z-Score rejected). They fight our SM+RSI momentum edge by blocking profitable breakouts.
- **Trailing/dynamic exits fail** (trailing stops PF 0.933, SM flip exit OOS failure). Fixed TP exits are our edge.
- **Stacking filters fails** — geometric trade count reduction kills portfolio metrics every time.
- **vScalpB (SM_T=0.25) is filter-resistant** — high-conviction entries are uncorrelated with most features.
