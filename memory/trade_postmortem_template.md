# Trade Post-Mortem Template

Standard format for daily trade analysis. Run after each trading day.

## Required Sections

### 1. Pre-Market Context
- Overnight range (pts) and position within range (percentile)
- VIX prior-day close and gate status
- Prior-day ATR and gate status (vScalpC)
- Key economic events / news calendar
- Any overnight observations (gaps, trend, momentum)

### 2. Trade-by-Trade Breakdown
For each trade (grouped by entry event, not individual fills):
- **Entry**: Time (ET), price, side, strategy, qty
- **Exit**: Time (ET), price, pts, P&L, exit reason, bars held
- **Partial trades**: Show TP1 and runner as sub-rows of the same entry
- **MFE/MAE**: Max favorable/adverse excursion in pts (from bar data if not in session)
- **Narrative**: What happened — was the entry good? Why did the exit happen where it did?
- **Gate interactions**: Was this signal blocked/delayed by any gate? Did it help or hurt?

### 3. Per-Strategy Summary Table
| Strategy | Trades | W/L | Net P&L | Notes |
**Always use commission-adjusted P&L** (matches dashboard). Commissions: MNQ $1.04 RT/contract, MES $2.50 RT/contract. Raw session file P&L does NOT include commissions.

### 4. Gate Activity
- Which gates fired (blocked signals)?
- Were blocks correct (would the blocked trade have lost)?
- Any gates that SHOULD have fired but didn't?

### 5. What-If Analysis (when applicable)
- Pending config changes: would they have improved or degraded today?
- Show specific trade-by-trade impact with bar data evidence
- Note: single-day what-if is anecdotal — always reference the backtest sample size

### 6. Behavioral Notes
- Any manual interventions (early exits, pauses)? Why?
- Did fear/greed drive any decisions that deviated from the system?
- Was the MES manual close justified or emotional?

### 7. Market Observations
- Intraday structure: trending, ranging, reversal?
- Key S/R levels tested
- Volume patterns (if notable)
- Anything relevant for next session

### 8. Action Items
- Config changes to make
- Research to queue
- Bugs or issues discovered

## Rules
- Run post-mortem EVERY trading day, even zero-trade days (document why no trades)
- Use actual ET times (convert from UTC if needed: entry_time is stored as UTC ISO string)
- Session bar timestamps are ET epoch — use `datetime.fromtimestamp(x, tz=et)` but note the DST offset
- Always check bar data for MFE/MAE when session file has None values
- Compare against pending config upgrades when they exist
- Be honest about behavioral deviations — the system works if you follow it
