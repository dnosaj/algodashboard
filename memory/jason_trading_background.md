# Jason's Trading Background & Strategy Ideas

## Trading History

- **Breakout trader** for many years. Liked the risk-to-reward but couldn't find the right math to define when consolidation and breakout had genuine momentum behind it.
- **RSI trendline breakout trader** — a specific methodology:

### RSI Trendline Breakout (Longs)
1. Identify descending peaks on RSI (lower highs on the oscillator)
2. Draw a trendline connecting those descending RSI peaks
3. Wait — sometimes hours — for RSI to break BACK THROUGH that descending trendline
4. That break = strong long signal, catch several points

### RSI Trendline Breakout (Shorts)
1. Identify ascending troughs on RSI (higher lows on the oscillator)
2. Draw a trendline connecting those ascending RSI troughs
3. When RSI breaks back DOWN through that ascending trendline = short signal

### Overnight RSI Trendlines into NY (Best Setup)
1. Use RSI peaks from the OVERNIGHT session (18:00-09:30 ET)
2. Draw descending/ascending trendlines on those overnight RSI peaks
3. When those trendlines break during the NY session = massive wins
4. The overnight structure sets up the institutional move; NY session executes it

## How These Connect to Our Current System

### RSI Trendline + SM
- We already compute 5-min RSI and have SM as a momentum measure
- The RSI trendline breakout could be a NEW ENTRY SIGNAL (not a filter on SM+RSI, but a standalone strategy)
- SM could serve as the "momentum confirmation" Jason was looking for in breakout trading: consolidation (tight SM) → SM fires → RSI trendline breaks → enter
- This is algorithmically quantifiable: detect RSI local maxima, fit trendline, check for cross

### Consolidation + Breakout + Momentum
- Squeeze Momentum indicator detects BB-inside-KC (consolidation)
- We rejected it as a GATE on existing entries (it fought our momentum edge)
- But as a STANDALONE entry condition: Squeeze releases + SM confirms momentum direction + RSI trendline breaks = potentially powerful
- Different question: "block entries during squeeze" (rejected) vs "ENTER when squeeze fires with SM confirmation" (untested)

### Overnight Session Structure
- We already track overnight/Asia/London session bars
- Overnight RSI peaks → trendline → NY breakout is backtestable from our Databento data
- This aligns with the ICT HTF→LTF framework: overnight builds structure, NY executes

### Chart Images
- Jason has hundreds of chart images from his breakout trading days
- These could serve as visual pattern references: "this is what a good breakout setup looks like"
- Future: vision model analysis of chart patterns at entry time
- Future: similarity search — "find past charts that look like today's setup"

## Research Priority

These ideas represent potential NEW STRATEGIES, not filters on existing ones. They should be explored when:
1. The current 4-strategy portfolio is stable and validated in live trading
2. The Frontier Agent is built and can run overnight experiments
3. We want to diversify beyond SM+RSI entries

The RSI trendline breakout is the most immediately testable — it's pure math on data we already have.
