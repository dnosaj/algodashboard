# Research & Testing Queue

Rolling prioritized list of findings to test, investigate, or monitor.
Updated after each research session. Oldest items at bottom get dropped if not acted on.

## Ready to Test (have data, have hypothesis, need IS/OOS validation)

| # | Hypothesis | Source | Effect Size | Sample | Strategies | Priority |
|---|-----------|--------|-------------|--------|------------|----------|
| 1 | **London H/L proximity gate**: Block entries within 3pts of London H/L (undirectional) | Gate sweep (Mar 14) | vScalpC STRONG PASS (IS +7.7%, OOS +1.2%). vScalpA inconsistent. | 137 forensic / ~5 blocked per sweep | vScalpC primarily | **Observation first** — display + tag, monitor before hard gate |
| 2 | **Z-Score extension gate**: Block entries when Z-Score > threshold (e.g., 2.0) | Z-Score Probability indicator | Untested | -- | All MNQ (similar to ADR gate) | Medium |

## Monitor (promising signal, need more data or larger sample)

| # | Finding | Source | Current Sample | Trigger to Test |
|---|---------|--------|---------------|----------------|
| 1 | **Order block zone entries**: WR 36.4% (-37pp) | ICT forensics | N=11 | When N > 30 (accumulate via CF or extended backtest data) |
| 2 | **Weekly VAL proximity**: WR 50.0% (-23.5pp), PF 0.622 | ICT forensics + PVP | N=22 | When N > 40 |
| 3 | **vScalpC structure exit params**: LB=50, PR=2, buf=2 | Paper trading (started Mar 13) | ~1 day | After 10+ days acceptance criteria met |
| 4 | **TPX regime gate for vScalpB**: L=30 S=12 | TPX research | Pending paper validation | After paper data accumulates |

## Confirmed Not Useful (tested or analyzed, no edge)

| Finding | Source | Why |
|---------|--------|-----|
| FVGs on 5-min NQ | ICT forensics | Too short-lived (~2 bars). Almost no entries during active FVG. |
| Liquidity sweeps (undirected) | ICT forensics | Too common (>50% of trades). Not discriminating. |
| Confluence of ICT features | ICT forensics | No monotonic WR improvement. Features are independent. |
| Fib OTE as entry filter | ICT forensics | Our momentum entries work BETTER in premium zones. |
| Session sweep (any direction) | ICT forensics | Mixed signal when directions conflated. Aligned sweeps only mildly better. |
| VWAP Z-Score gate | Round 3 research (Mar 6) | Blocks profitable breakout entries. |
| Squeeze Momentum gate | Round 3 research (Mar 6) | Redundant with SM+RSI. Hurts all strategies. |
| ATR/ADX entry filters | ATR/ADX research (Mar 9) | FAIL on all strategies. |
| Regime detection (statistical) | Multiple sessions | London range, rolling vol, prior-day range — all fail walk-forward. |

## Deferred (valid idea, not prioritized now)

| Item | Reason for deferral |
|------|-------------------|
| CF structure exit simulation (vScalpC) | Params may change during paper trading. Revisit after 50+ blocked signals. Plan: `plans/cf-structure-exit-deferred.md` |
| Dashboard ICT overlays (weekly PVP, session H/L, OBs) | Test London H/L gate first to confirm value before building visualization |
| Weekly PVP as MES gate (extend daily VPOC+VAL to weekly) | Weekly VAL sample too small (N=22). Monitor. |
| Order block entry gate | N=11. Need extended data or different detection parameters. |
