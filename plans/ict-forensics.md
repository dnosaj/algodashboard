# ICT Trade Forensics — In Progress

**Date**: March 14, 2026
**Status**: COMPLETE — results in `backtesting_engine/results/ict_forensics_output.txt`
**Phase**: 3 (forensics first) of 3→1→2 approach

## Goal

Map every trade in our 12.8-month gated portfolio backtest against ICT structural features at entry time. Answer: do our SL trades cluster at specific ICT features?

## Approach

1. Port UAlgo ICT Concepts Pine Script logic to Python:
   - Order Blocks: 3-bar engulfing pattern (bar[2] opposing, bar[1] engulfing, bar[0] confirming)
   - Fibonacci/OTE: Structure-anchored from last BOS/MSS break extreme to prior swing
   - Liquidity Sweeps: 20-bar pivot, wick beyond but close back inside
2. Compute FVGs directly (3-bar imbalance, no library — Python lib has lookahead bias)
3. Compute BOS/MSS using our existing pivot swing infrastructure (LB=50 for swing level)
4. **Multi-timeframe**: Compute OBs, FVGs, BOS on 5-min bars (resample from 1-min), project onto 1-min entries
5. Run gated portfolio backtest to get all trades (run_and_save_portfolio.py)
6. For each trade at entry time, compute:
   - Nearest bullish/bearish OB and distance
   - Whether entry is inside an unmitigated FVG
   - Whether a liquidity sweep just occurred
   - BOS/MSS state (trending or just reversed)
   - Fib OTE zone position (premium/discount/OTE)
7. Output: WR by feature, SL clustering, confluence analysis

## Key Design Decisions

- **UAlgo OB logic preferred over Python library** — simpler 3-bar pattern, no structure-break dependency, no lookahead bias
- **5-min HTF features** — ICT is fundamentally HTF→LTF. Compute structure on 5-min, enter on 1-min (same pattern as our RSI 5-min→1-min mapping)
- **No lookahead**: All features use bar[i-1] or earlier. FVGs confirmed at bar[i] using bars [i-2], [i-1], [i] with no forward look
- **Structure-anchored fibs**: Auto-anchor to last BOS/MSS break (UAlgo approach), not arbitrary swing selection

## Source Code

- UAlgo Pine Script: provided by Jason, analyzed and documented in `memory/indicators/ict_concepts.md`
- Python library: `joshyattridge/smart-money-concepts` — HAS LOOKAHEAD BIAS, don't use for FVG/swings without patching
- Our swing infrastructure: `structure_monitor.py` PivotExitTracker (LB=50, already parity-tested)

## Data

- MNQ + MES 1-min bars: 2025-02-17 to 2026-03-12 (12.8 months)
- Trades: from gated portfolio backtest (run_and_save_portfolio.py)
- VIX: yfinance download

## Expected Output

```
TRADE FORENSICS: ~1,200 trades analyzed

5-MIN ORDER BLOCKS:
  Entries near bearish OB (within 10pts):  XX trades, WR XX% (vs avg XX%)
  Entries near bullish OB (within 10pts):  XX trades, WR XX% (vs avg XX%)
  Entries inside unmitigated OB zone:      XX trades, WR XX%

5-MIN FAIR VALUE GAPS:
  Entries inside bullish FVG:   XX trades, WR XX%
  Entries inside bearish FVG:   XX trades, WR XX%

FIBONACCI / OTE:
  Entries in OTE zone (62-79%): XX trades, WR XX%
  Entries in premium zone:      XX trades, WR XX%
  Entries in discount zone:     XX trades, WR XX%

LIQUIDITY SWEEPS:
  Entry within 5 bars of sweep: XX trades, WR XX%

BOS/MSS STATE:
  Entry with BOS (trend continuation): XX trades, WR XX%
  Entry with recent MSS (reversal):    XX trades, WR XX%

CONFLUENCE:
  2+ features aligned: XX trades, WR XX%
  3+ features aligned: XX trades, WR XX%
```

## Key Findings

- **Order Blocks**: Strongest signal. Entries inside OB zones WR 36.4% vs 73.5% baseline (-37pp). Near opposing OB: 45.5% SL rate vs 13.7% baseline. Small sample (11 trades).
- **FVGs**: Too short-lived on NQ 5-min (~2 bar median lifespan). Only 2 trades near FVG — not useful as filter.
- **Fibonacci/OTE**: Entries in premium zones perform BETTER (WR 76.7%). System is momentum-based, not mean-reversion. OTE zone entries show slightly better PF (1.777).
- **Liquidity sweeps**: Too common (nearly every trade has recent sweep). Not discriminating.
- **Structure state**: Entries AGAINST 5-min trend perform slightly better (WR 74.5% vs 73.0%). SM+RSI catches reversals.
- **Confluence**: No monotonic improvement with more features.
- **SL clustering**: "Near opposing OB" is the strongest predictor of SL (45.5% SL rate, +31.7pp vs baseline).

## After Forensics

If patterns emerge → Phase 1 (backtest as pre-filters with IS/OOS) → Phase 2 (implement in live engine)
