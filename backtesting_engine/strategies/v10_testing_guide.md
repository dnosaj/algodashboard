# SM+RSI v10 Testing & Operations Guide

## Files

| File | What It Is |
|------|-----------|
| `scalp_v10_production.pine` | v10 with AlgoAlpha via `input.source()` — requires AA indicator on chart |
| `scalp_v10_standalone.pine` | v10 with SM computed internally — no external indicator needed |
| `scalp_v9_smflip.pine` | Original v9 baseline for regression comparison |

**Use standalone for production.** The production version exists only for A/B validation against the original v9 approach. Once you confirm parity on TradingView, standalone is the only script you need.

---

## Setup

### Standalone (recommended)
1. Open instrument on a **1-minute** chart
2. Add `scalp_v10_standalone.pine` strategy
3. Done — SM is computed internally, no linking required

### Production (for validation only)
1. Open instrument on a **1-minute** chart
2. Add AlgoAlpha "Smart Money Volume Index" indicator → set Display Mode to **"Net"**
3. Add `scalp_v10_production.pine` strategy
4. In strategy Settings → Inputs → Smart Money group:
   - "AlgoAlpha Net Buy Line" → select AlgoAlpha's **"Net Buy Line"** plot
   - "AlgoAlpha Net Sell Line" → select AlgoAlpha's **"Net Sell Line"** plot
5. If the bottom banner shows red "LINK SM SOURCES TO ALGOALPHA IN SETTINGS" — you missed step 4

---

## Regression Test: Confirm v10 = v9

Before using any safety features, verify v10 produces the exact same trades as v9. This is the most important test.

### Settings to match v9 exactly:

**In Inputs:**
| Setting | Set To |
|---------|--------|
| Hard Stop Loss → Enable | **OFF** |
| Trailing Stop → Enable | **OFF** |
| Breakeven Stop → Enable | **OFF** |
| Daily Risk → Enable Daily Loss Limit | **OFF** |
| Daily Risk → Max Consecutive Losses | **0** |
| Daily Risk → Cooldown | **15** |
| SM Threshold | **0.00** |
| RSI Length | **10** |
| RSI Buy Level | **55** |
| RSI Sell Level | **45** |

**In Strategy Properties:**
| Setting | Set To |
|---------|--------|
| Slippage | **0** |
| Commission | **0.52** per contract |

**For standalone, also set SM params to AA defaults:**
| Setting | Set To |
|---------|--------|
| Index Period | **25** |
| Volume Flow Period | **14** |
| Normalization Period | **500** |
| PVI/NVI EMA Length | **255** |

### How to compare:
1. Add v9 to chart → note: trade count, net profit, win rate, profit factor
2. Add v10 (with settings above) to same chart
3. All four numbers should be **identical**
4. If trade count differs, check SM params first (standalone must use 25/14/500/255 to match AA defaults)

---

## Parity Test: Standalone vs Production

This confirms the embedded SM matches AlgoAlpha exactly.

1. Load both v10_production (with AA linked) and v10_standalone on the same chart
2. Set both to identical settings (all safety OFF, slippage 0)
3. Set standalone SM params to **25 / 14 / 500 / 255** (AA defaults)
4. Compare trade counts — they should be **identical**
5. If they match, the embedded SM is verified. You never need the production version again.

**Why they match on TradingView but not in Python:** Both Pine Scripts use `ta.pvi` / `ta.nvi` which read TradingView's native tick volume. The Python backtester synthesizes fake volume using `High - Low`, which produces different SM values. This is a Python-only limitation.

---

## Safety Features: What Each Does & When to Enable

### Hard Stop Loss (default: ON, ATR 3x14)

**What:** Places a standing stop-loss order at entry ± ATR distance. Fires intrabar (doesn't wait for bar close).

**Why ON by default:** This is your flash-crash protection. ATR 3x on MNQ 1-min gives ~9-15 points, which is wide enough that the SM flip exit fires first in normal conditions. The hard stop only triggers if price moves violently against you before SM has time to flip.

**When to turn OFF:** Only for regression testing against v9. Never run live without it.

**Modes:**
- **ATR (recommended):** Adapts to current volatility. Higher vol = wider stop = fewer false triggers.
- **Fixed:** Constant point distance. Use only if you want exact control.

**Tuning:**
- ATR multiplier 3.0 = conservative (rarely fires). Start here.
- ATR multiplier 2.0 = tighter, will fire on some trades that SM flip would have exited profitably.
- ATR multiplier 4.0+ = extremely wide, almost never fires.

### Trailing Stop (default: OFF)

**What:** After price moves 15 pts in your favor, a trailing stop activates 8 pts behind the best price.

**Why OFF by default:** SM flip exit already handles profit-taking well. The trailing stop can cut winners short if SM is still favorable. Enable only after you've forward-tested and want to lock in profits on runaway moves.

**When to turn ON:** After you've watched the strategy live for a while and noticed trades that were +20 pts but SM flip exited at +5 pts. The trail would have locked in more.

**Tuning:**
- Activation 15 pts = only activates on big winners (MNQ avg winning trade is ~5-10 pts)
- Trail distance 8 pts = gives room for normal pullbacks within a trend
- Tighten both numbers for faster profit-locking, widen for more room

### Breakeven Stop (default: OFF)

**What:** After price moves 8 pts in your favor, stop moves to entry + 0.5 pts (covers commission).

**Why OFF by default:** Same reasoning as trail — SM flip handles exits. BE can get you stopped out at breakeven on a trade that would have been a winner if SM flip was allowed to work.

**When to turn ON:** If you want to eliminate the possibility of a winning trade turning into a loser. Good for confidence early on.

**Tuning:**
- BE Trigger 8 pts = trade must be solidly profitable before locking in
- BE Offset 0.5 pts = covers $1.00 round-trip on MNQ so you don't lose money even at BE
- **Important:** BE only applies if it's tighter than the hard SL. If hard SL is at -10 pts and BE would be at -12 pts, BE is ignored.

### Daily Loss Limit (default: ON, 3%)

**What:** If your equity drops 3% from day-start equity, no new entries are placed. Existing positions stay open and can exit normally.

**Why ON by default:** Circuit breaker for cascading losses. MNQ's worst historical day in our data was ~1-2% drawdown. 3% is conservative — it should almost never trigger in normal conditions.

**When to turn OFF:** Only for regression testing. Keep it on in live trading.

**Tuning:**
- 3% = conservative, almost never fires in backtests
- 2% = moderate, may fire on volatile days
- 5% = very loose, only catches extreme events

### Max Consecutive Losses (default: OFF, 0)

**What:** Pauses new entries after N consecutive losing trades. Does NOT reset on new day — a losing streak spanning overnight is still a streak.

**When to turn ON:** If you want psychological protection. After 3-4 losses in a row, the market regime may have shifted.

**Tuning:**
- 0 = disabled
- 3 = aggressive pause
- 5 = moderate
- Streak resets to 0 on the next winning trade

---

## Cross-Instrument Testing

The strategy auto-adapts to any instrument via `syminfo.mintick` (for tick conversions) and ATR (for stop distances). Only the commission needs manual adjustment.

### Commission Values Per Instrument

Set in Strategy Properties → Commission → Cash Per Contract:

| Instrument | Commission Per Side | Set `commission_value` To |
|-----------|-------------------|--------------------------|
| **MNQ** (Micro Nasdaq) | $0.52 | **0.52** |
| **NQ** (E-mini Nasdaq) | $1.25 | **1.25** |
| **MES** (Micro S&P) | $0.52 | **0.52** |
| **ES** (E-mini S&P) | $1.25 | **1.25** |
| **MYM** (Micro Dow) | $0.52 | **0.52** |
| **YM** (E-mini Dow) | $1.25 | **1.25** |
| **M2K** (Micro Russell) | $0.52 | **0.52** |
| **RTY** (E-mini Russell) | $1.25 | **1.25** |
| **GC** (Gold) | $1.25 | **1.25** |
| **MGC** (Micro Gold) | $0.52 | **0.52** |
| **CL** (Crude Oil) | $1.25 | **1.25** |
| **MCL** (Micro Crude) | $0.52 | **0.52** |

### Testing Priority (strongest out-of-sample first)

1. **MES** — same underlying as ES (already validated), confirms micro contract works
2. **NQ** — same underlying as MNQ (developed on), confirms full-size contract works
3. **YM / MYM** — Dow futures, different index, true out-of-sample
4. **RTY / M2K** — Russell 2000, most different from Nasdaq, strongest equity test
5. **GC / MGC** — Gold, completely different asset class
6. **CL / MCL** — Crude oil, very different volatility profile

### What to Look For

On each instrument, check:
- **Trade count**: Should be reasonable (10-50+ over a few weeks). If 0-2 trades, the SM threshold or RSI levels may need adjustment.
- **Win rate**: Above 50% is good. Above 55% is strong.
- **Profit factor**: Above 1.3 is viable. Above 2.0 is excellent.
- **Max drawdown**: Should be manageable relative to account size.
- **Hard SL fires**: Should be rare (< 10% of exits). If the hard SL is firing frequently, the ATR multiplier is too tight or the instrument is too volatile for the default settings.

### If the Strategy Doesn't Work on an Instrument

Try adjusting in this order:
1. **SM threshold**: Lower from 0.05 to 0.00 (allow weaker SM signals)
2. **RSI levels**: Widen from 60/40 to 55/45 (more entries)
3. **Cooldown**: Reduce from 15 to 6 bars (faster re-entry)
4. **SM params (standalone only)**: Try fast params 15/10/300/150 instead of defaults
5. **Session filter**: Disable if the instrument trades different hours (e.g., gold has its own session)

---

## Python Backtest Results (for reference)

### ES with AlgoAlpha SM — Best Configs

| Config | Trades | WR% | PF | Net$ (1 lot) |
|--------|--------|-----|-----|-------------|
| RSI14 60/40 SM>0.00 CD6 cross | 29 | 62.1% | **3.184** | +$7,815 |
| RSI14 60/40 SM>0.05 CD6 cross | 24 | 62.5% | **2.934** | +$6,378 |
| RSI14 65/35 SM>0.00 CD6 cross | 22 | 54.5% | 1.676 | +$2,795 |
| RSI10 55/45 SM>0.00 CD3 cross | 57 | 59.6% | 1.640 | +$5,658 |

### MNQ with AlgoAlpha SM

| Config | Trades | WR% | PF | Net$ (1 lot) |
|--------|--------|-----|-----|-------------|
| RSI14 65/35 SM>0.00 CD6 cross | 23 | 56.5% | **2.904** | +$790 |
| RSI14 65/35 SM>0.05 CD6 cross | 20 | 50.0% | 2.717 | +$712 |
| RSI14 60/40 SM>0.05 CD6 cross | 28 | 57.1% | 1.549 | +$577 |
| RSI14 60/40 SM>0.00 CD6 cross | 35 | 54.3% | 1.312 | +$412 |

### Key Insight from Python Tests

The Python standalone SM (using synthesized volume) diverges from AlgoAlpha SM (using real tick volume). This is expected and only applies to Python backtests. On TradingView, both Pine Scripts use the same tick volume and should produce identical results.

---

## Alert Setup (for live trading)

The strategy includes 5 alert conditions:

1. **Long Entry** — fires when a long signal triggers
2. **Short Entry** — fires when a short signal triggers
3. **SM Flip Exit** — fires when SM changes sign and closes a position
4. **Daily Loss Limit Hit** — fires when daily equity drops below threshold
5. **Consecutive Loss Limit** — fires when max consecutive losses reached

To set up alerts in TradingView:
1. Right-click the strategy on chart → "Add Alert..."
2. Select the condition from the dropdown
3. Set notification method (push, email, webhook)
4. Repeat for each alert you want

---

## Dashboard Reference

The live dashboard (top-right corner) shows 16 rows:

| Row | Shows |
|-----|-------|
| Version | "v10 Standalone" + volume status |
| SM Index | Current SM value, green (bullish) / red (bearish) |
| 5m RSI | Current 5-min RSI value |
| Position | LONG / SHORT / FLAT |
| Open P&L | Unrealized dollar P&L |
| Hard SL | Stop distance in points + mode (ATR/Fixed) or OFF |
| Trail | OFF / ARMED (in position, not yet activated) / ACTIVE (trailing) |
| Breakeven | OFF / ARMED (in position, not triggered) / LOCKED (stop at entry) |
| --- Daily --- | Separator |
| Day P&L | Dollar + percentage from day start |
| Day W/L | Win/Loss count for the day |
| Consec L | Current consecutive loss streak |
| Risk | OK (green) / DAILY LIMIT (red) / CONSEC LIMIT (red) |
| --- Config --- | Separator |
| RSI Cfg | RSI length + buy/sell levels |
| SM / CD | SM params + cooldown bars |

---

## Quick Reference: Recommended Starting Settings

### Conservative (start here)
- Hard SL: **ON**, ATR 3x14
- Trail: **OFF**
- BE: **OFF**
- Daily Limit: **ON**, 3%
- Max Consec Losses: **0** (off)
- Slippage: **1**

### After 2+ weeks of forward testing, if results are good
- Trail: Consider **ON** with 15 pt activation / 8 pt trail
- BE: Consider **ON** with 8 pt trigger / 0.5 pt offset

### Never disable in live trading
- Hard Stop Loss
- Daily Loss Limit
