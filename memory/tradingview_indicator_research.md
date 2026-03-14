# TradingView Open-Source Indicator Research (Mar 6, 2026)

Research into popular open-source TradingView community indicators that could serve as
entry filters or exit signals for SM+RSI scalping strategies on futures (MNQ, MES).

## Summary of Most Promising Indicators for Our Use Case

| Indicator | Category | Use Case | Implementable? | Priority |
|-----------|----------|----------|---------------|----------|
| VWAP Deviation / Z-Score | Session | Entry gate (block overextended) | Yes - simple math | HIGH |
| Squeeze Momentum (LazyBear) | Trend/Exhaustion | Entry gate (squeeze = don't enter) | Yes - BB vs KC | HIGH |
| Leledc Exhaustion | Exhaustion | Entry gate (block at exhaustion bars) | Yes - price comparison | HIGH |
| Initial Balance levels | Session | Entry gate (don't enter near IB extremes) | Yes - session H/L | MEDIUM |
| Fair Value Gap proximity | Imbalance | Entry gate (block if inside unfilled FVG) | Moderate complexity | MEDIUM |
| Market Structure (BOS/CHoCH) | Structure | Exit signal or entry confirmation | Complex to replicate | LOW |
| Divergence for Many | Divergence | Entry gate (block on bearish divergence) | Already have RSI | LOW |
| Nadaraya-Watson Envelope | Trend | Entry gate (block at envelope extremes) | Repainting concerns | LOW |

---

## 1. MARKET STRUCTURE INDICATORS

### Smart Money Concepts (SMC) [LuxAlgo]
- **Author**: LuxAlgo
- **Popularity**: ~123K likes, 12K boosts — #1 most-liked indicator on TradingView
- **Open source**: Yes (Pine Script viewable)
- **What it does**: All-in-one: BOS, CHoCH, order blocks, FVGs, premium/discount zones, equal H/L, swing point labels (HH/HL/LH/LL)
- **Key params**: Internal structure pivot length, swing structure pivot length, order block settings, FVG threshold
- **Use case**: Could use CHoCH as exit signal (trend reversal detected). BOS as entry confirmation.
- **Verdict for us**: Very complex to port to Python. Our SM indicator already does trend detection. The order block / FVG components are more interesting standalone.

### Market Structure by mickes
- **Author**: mickes
- **Popularity**: ~3.2K likes, 18K boosts
- **Open source**: Yes
- **What it does**: Shows BOS and CHoCH using swing pivots. Two modes: internal (pivot length 5) and swing (pivot length 50). Detects equal H/L zones and liquidation levels.
- **Key params**: Internal pivot length (default 5), Swing pivot length (default 50), ATR factor % (default 10%), liquidation detection
- **Use case**: Entry gate — block entries when CHoCH just occurred (trend reversal). Or use as directional filter.
- **Verdict**: Simpler than LuxAlgo SMC. Could port the CHoCH detection as a gate. But our SM already handles this.

### Market Structure CHoCH/BOS (Fractal) [LuxAlgo]
- **Author**: LuxAlgo
- **Popularity**: ~12K likes, 3.8K boosts
- **Open source**: Yes
- **What it does**: Fractal-based (not swing-point-based) BOS/CHoCH. More adaptive but can miss reversals. Includes S/R levels.
- **Key params**: Length (fractal pattern detection length)
- **Use case**: Alternative to swing-based structure. Fractal approach may be faster to detect shifts.
- **Verdict**: Interesting alternative but fractal-based = may be noisier on 1-min.

---

## 2. FAIR VALUE GAP / IMBALANCE INDICATORS

### Fair Value Gap [LuxAlgo]
- **Author**: LuxAlgo
- **Popularity**: ~7.7K likes, 515 boosts
- **Open source**: Yes
- **What it does**: Detects bullish FVG (current low > high from 2 bars ago) and bearish FVG (current high < low from 2 bars ago). Tracks fill % and avg mitigation time. "Instantaneous mitigation" flags fast gap fills = reversal signal.
- **Key params**: Threshold % (min FVG height), Auto Threshold (cumulative mean), Unmitigated Levels, Timeframe
- **Use case**: ENTRY GATE — block entry if price is inside an unmitigated FVG (imbalance zone likely to be revisited). Or: block if entry would be right after FVG creation (price moving fast, likely to retrace).
- **Verdict**: FVG is essentially a 3-bar pattern. Simple to implement. But need to test if 1-min FVGs are meaningful on NQ (probably too noisy). 5-min or 15-min FVGs projected onto 1-min might work better.

### Multitimeframe Fair Value Gap (Zeiierman)
- **Author**: Zeiierman
- **Popularity**: Not measured (premium author)
- **What it does**: Multi-timeframe FVG with smart volume logic. Shows institutional imbalances across timeframes.
- **Verdict**: Premium indicator, but concept is portable. MTF FVGs (5-min, 15-min) are more meaningful than 1-min.

---

## 3. TREND STRENGTH / EXHAUSTION INDICATORS

### Squeeze Momentum Indicator [LazyBear]
- **Author**: LazyBear
- **Popularity**: ~109K likes, 2.7M views — one of most popular indicators ever on TV
- **Open source**: Yes
- **What it does**: Based on John Carter's TTM Squeeze. When Bollinger Bands are inside Keltner Channels = "squeeze" (low volatility, consolidation). When they break out = "squeeze fires" (expansion). Momentum histogram uses linear regression.
- **Key params**: BB Length (default 20), BB MultFactor (default 2.0), KC Length (default 20), KC MultFactor (default 1.5), Use TrueRange (bool)
- **How signals work**: Black dots = squeeze on (BB inside KC). Gray dots = no squeeze. Histogram color = momentum direction. First gray dot after black = potential breakout entry.
- **Use case**: ENTRY GATE — block entries during squeeze (momentum is compressed, direction unclear). Only allow entries when squeeze has fired (momentum expanding). Or: block entries when histogram is fading (momentum exhausting).
- **Verdict**: HIGH PRIORITY. Simple to implement (BB and KC are standard). Squeeze state is a binary gate. Could prevent entering during choppy consolidation phases where our SM+RSI signals whipsaw.

### Leledc Exhaustion V4
- **Author**: Joy_Bangla (converted from original)
- **Popularity**: ~1.2K likes, 23K favorites
- **Open source**: Yes
- **What it does**: Detects "exhaustion bars" — the last buyer in an uptrend or last seller in a downtrend. Uses bar counting relative to price 4 bars ago.
- **Key params**: maj_qual (default 6), maj_len (default 30), min_qual (default 5), min_len (default 5), show Major/Minor
- **How signals work**: Counts consecutive bars closing above/below the close from 4 bars prior. When count reaches threshold (maj_qual or min_qual), signals exhaustion.
- **Use case**: ENTRY GATE — block long entries when bullish exhaustion detected (last buyer), block shorts on bearish exhaustion.
- **Verdict**: HIGH PRIORITY. Very simple math. Could be powerful for avoiding entries at stretched price levels. Test whether maj_qual=6 works on 1-min NQ/ES.

### Exhaustion Signal [ChartingCycles]
- **Author**: ChartingCycles
- **Popularity**: ~1.4K likes
- **Open source**: Yes
- **What it does**: Counts consecutive bars where close > close[4] (or < close[4]). When count hits Level 1 (9), Level 2 (12), or Level 3 (custom) = exhaustion signal.
- **Key params**: Level 1 (default 9), Level 2 (default 12), Level 3 (custom)
- **Use case**: Same as Leledc but simpler — pure count mechanism.
- **Verdict**: MEDIUM. Simpler version of Leledc. Worth testing as a quick filter.

### %R Trend Exhaustion [upslidedown]
- **Author**: upslidedown
- **Popularity**: ~5.4K likes, 7.5K boosts
- **Open source**: Yes
- **What it does**: Combines short-period Williams %R with long-period (112) %R. Confluence between both being overbought/oversold = "area of interest." Break from OB/OS triggers signal.
- **Key params**: Short %R period, Long %R period (default 112), Smoothing (default 3)
- **Use case**: ENTRY GATE — block entries when both %R periods show overextension in the entry direction.
- **Verdict**: MEDIUM. Similar concept to our RSI filter but uses %R. Could add value if %R captures different exhaustion signature than RSI.

### Momentum Exhaustion Indicator [Zeenobit]
- **Author**: Zeenobit
- **Popularity**: Moderate
- **Open source**: Yes
- **What it does**: Spots points of maximum momentum exhaustion. Isolates max risk / max opportunity periods.
- **Verdict**: Less documented. Lower priority.

---

## 4. MOMENTUM DIVERGENCE DETECTORS

### Divergence for Many Indicators v4 [LonesomeTheBlue]
- **Author**: LonesomeTheBlue
- **Popularity**: ~721 likes, 193 boosts
- **Open source**: Yes
- **What it does**: Checks divergences between price and up to 9 built-in oscillators (RSI, MACD, MACD Histogram, Stochastic, CCI, Momentum, OBV, VWMACD, CMF) + external. Scans last 16 pivot points.
- **Key params**: Pivot Period, Source (Close or H/L), Max Pivots to Check (default 16), Max Bars to Check, Min # Divergences, Show Hidden Divergences
- **Use case**: ENTRY GATE — block entry if multiple oscillators show bearish divergence (price higher high but oscillator lower high = weakening momentum).
- **Verdict**: LOW PRIORITY for us. We already use RSI. Adding CCI or Stochastic divergence might help but adds complexity. Our prior research shows divergence-type filters have mixed OOS results.

### CCI Divergence Detector [sizzlinsoft]
- **Author**: sizzlinsoft
- **Open source**: Yes
- **What it does**: CCI-based divergence only. Supports 500 markings and lines.
- **Verdict**: Narrow scope. Skip unless CCI specifically shows value.

---

## 5. SESSION-BASED INDICATORS

### VWAP Deviation Oscillator [BackQuant]
- **Author**: BackQuant
- **Popularity**: ~7.7K likes, 44K boosts
- **Open source**: Yes
- **What it does**: Measures how far price is from session VWAP. Three modes: Percent, Absolute, Z-Score. Z-Score standardizes deviation by its own mean and stdev. Fixed bucket edges at 0.5, 1.0, 2.0, 2.8 sigma.
- **Key params**: VWAP Mode (4H/Daily/Weekly/Rolling), Price reference (HLC3/Close), Deviation method (Percent/Absolute/Z-Score), Z/Std Window, Min sigma guard
- **Use case**: ENTRY GATE — block entries when price is >2 sigma from VWAP (overextended, likely to mean-revert). Especially relevant for our TP=5 strategy where a 5pt reversion against us at an extended level = full SL hit.
- **Verdict**: HIGH PRIORITY. VWAP Z-score is mathematically clean, session-anchored, and directly measures what we want to avoid (entering at extended prices). Simple to implement with standard VWAP calc. Could use daily VWAP with 2-sigma gate.

### Opening Range, Initial Balance, Opening Price [PtGambler]
- **Author**: PtGambler
- **Popularity**: ~656 likes, 1.56M views
- **Open source**: Yes
- **What it does**: Draws Opening Range (first 30-60 min H/L), Initial Balance (first two 30-min periods H/L), and Opening Price. Futures-aware with timezone support.
- **Key params**: Session times (RTH start), Opening Range duration, IB duration, timezone
- **Use case**: ENTRY GATE — block entries near IB High/Low (likely reversal zones). Or: only allow entries within IB range (mean reversion zone).
- **Verdict**: MEDIUM PRIORITY. IB levels are meaningful on ES/NQ. RTH IB (9:30-10:30 ET) high/low often acts as magnet/resistance. Could block entries within X ticks of IB extreme.

### Initial Balance Breakout Signals [LuxAlgo]
- **Author**: LuxAlgo
- **Popularity**: ~6.7K likes, 1.7K boosts
- **Open source**: Yes
- **What it does**: Auto-detects IB range, marks IB extensions (1.5x, 2x, etc.), includes Fibonacci levels within IB, weekday-filtered IB forecasts.
- **Key params**: Display Last X IBs, IB session selection, Extension %, Fibonacci levels, Forecast method
- **Use case**: Entry gate using IB extension levels as resistance/support.
- **Verdict**: More sophisticated version of IB. Extensions are useful — price often stalls at 1.5x IB.

### NQ 65 Point Futures Session Opening Range [Bostonshamrock]
- **Author**: Bostonshamrock
- **Open source**: Yes
- **What it does**: NQ-specific. 30-second opening range with 65-point interval projections. Monitors RTH, Globex, and Europe sessions.
- **Verdict**: Too NQ-specific and very short timeframe (30-sec). Not directly useful for 1-min bar strategies.

---

## 6. MULTI-TIMEFRAME CONFLUENCE INDICATORS

### Multi-Timeframe Confluence Indicator [TradeTechanalysis]
- **Author**: TradeTechanalysis
- **Open source**: Yes
- **What it does**: Combines EMA and RSI across multiple timeframes. Dynamic cooldown via ATR. Price action triggers for sharp moves.
- **Verdict**: Interesting concept but would need significant customization.

### Trinity Multi-Timeframe CCI [EMA34TRADER]
- **Author**: EMA34TRADER
- **Open source**: Yes
- **What it does**: Three CCI lines (current TF, 4H, daily) on single pane. Shows MTF momentum alignment.
- **Key params**: CCI length (same across all TFs), three timeframe selections
- **Use case**: ENTRY GATE — only enter when all three TFs show aligned momentum.
- **Verdict**: LOW PRIORITY. Adding more oscillators has diminishing returns per our research.

---

## 7. OTHER NOTABLE INDICATORS

### Nadaraya-Watson Envelope [LuxAlgo]
- **Author**: LuxAlgo
- **Popularity**: ~30K likes, 1.15M views
- **Open source**: Yes
- **What it does**: Kernel smoothing-based envelope. Estimates trend via non-parametric regression, then adds/subtracts mean absolute deviation. Cross signals when price touches envelope extremes.
- **Key params**: Bandwidth (smoothness), Mult (envelope width), Source, Repainting toggle
- **CRITICAL**: Default mode REPAINTS. Non-repainting mode available but less accurate.
- **Verdict**: LOW PRIORITY due to repainting. Non-repainting version is essentially a smoothed moving average envelope — not much different from Bollinger Bands.

### Liquidity Sweep / Stop Hunt Detectors
- Multiple authors (Quantura, wateriskey6689, H1 Liquidity Sweep Tracker)
- **What they do**: Detect when price sweeps a previous swing H/L and reverses — "stop hunt" pattern
- **Use case**: EXIT SIGNAL — if price sweeps a level and reverses against our position, exit. Or ENTRY GATE — enter only after a liquidity sweep (higher probability reversal).
- **Verdict**: Interesting for MES v2 (longer hold time). Not useful for MNQ TP=5 scalps (too slow).

### Chandelier Exit [everget]
- **What it does**: ATR-based trailing stop. Tracks local extremes and creates dynamic exit levels.
- **Use case**: EXIT SIGNAL for MES v2 runner leg (replace or supplement BE_TIME exit).
- **Verdict**: We already have SL/TP exits. Could be useful as a trailing stop for MES v2 runner.

---

## Implementation Priority for Backtesting

### Round 1 (Simple, high potential)
1. **VWAP Z-Score gate**: Block entry when |Z| > threshold (e.g., 2.0). Daily session VWAP.
2. **Squeeze Momentum gate**: Block entry during squeeze (BB inside KC). Allow after squeeze fires.
3. **Leledc Exhaustion gate**: Block entry on exhaustion bars. Simple close[0] vs close[4] counting.

### Round 2 (Moderate complexity)
4. **IB proximity gate**: Block entry within X ticks of IB High/Low.
5. **FVG proximity gate**: Block entry inside unmitigated 5-min FVG.
6. **Exhaustion count gate**: Block entry when 9+ consecutive bars in one direction.

### Round 3 (Complex, lower priority)
7. **Market Structure CHoCH**: Block entry after CHoCH against trade direction.
8. **Multi-indicator divergence**: Block entry on bearish divergence across RSI+CCI+Stoch.
