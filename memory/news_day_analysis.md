# News Day Impact Analysis (Feb 22, 2026)

## What We Built

### Backtest Results Archive
- **`backtesting_engine/results/save_results.py`** — `save_backtest()` / `load_backtest()` / `list_results()`
- Any sweep/validation script can call `save_backtest(trades, strategy=..., params=..., split=...)`
  to persist per-trade CSVs with full metadata (params, data range, IS/OOS label, run timestamp)
- Saved to `backtesting_engine/results/*.csv` with JSON metadata in comment header
- Previously we were throwing away all per-trade data after computing summary stats

### Run & Save Script
- **`backtesting_engine/strategies/run_and_save_portfolio.py`** — runs vScalpA + vScalpB + MES v2
  on full 12-month Databento data, saves FULL + IS + OOS trade logs
- Usage: `python3 run_and_save_portfolio.py --split`
- 9 CSV files generated (3 strategies × 3 splits)

### News Calendar + Analysis
- **`backtesting_engine/strategies/news_day_analysis.py`** — 107-event macro calendar
  (FOMC, NFP, CPI, PPI, GDP, PCE, Retail Sales, ISM, FOMC Minutes) with actual release
  dates accounting for the Oct-Nov 2025 government shutdown delays
- Cross-references saved trade logs, classifies each trade by news impact level
- Usage: `python3 news_day_analysis.py`

## Key Findings (12-month, FULL split)

### Portfolio-wide: blanket news blackout would HURT
- Total P&L: +$6,284 (1,499 trades)
- P&L on HIGH days: +$929 — skipping would lose $929

### By strategy:

**vScalpA (MNQ v15) — NEWS DAYS ARE BAD**
- News days: -$381 (298 trades) vs No-news: +$994 (467 trades)
- Skipping HIGH days improves P&L by **+$454**
- FOMC Rate Decision is worst event: **-$414** over 24 trades
- Lose rate: 17.1% on news vs 14.3% on quiet days
- Wide SL (50 pts = $100) gets blown out by FOMC/CPI volatility

**vScalpB (MNQ) — MIXED, targeted issues**
- HIGH days are net positive (+$303) — don't skip
- **Retail Sales is disastrous: -$388** across 18 trades (worst single event type)
- NFP delayed releases also bad: -$77

**MES v2 — LOVES NEWS DAYS**
- HIGH days: +$1,080 (48 trades, avg +$22.50/trade)
- FOMC Rate Decision is BEST event: **+$720** across 13 trades
- Wider SL (35 pts) + higher TP (20 pts) captures big moves
- Do NOT add any news filter to MES

### Specific problem events for MNQ:
1. **FOMC Rate Decision** — vScalpA -$414, vScalpB +$40
2. **Retail Sales** — vScalpA -$265, vScalpB -$388
3. **CPI** — vScalpA -$114, vScalpB -$1 (mostly vScalpA problem)
4. **NFP delayed releases** — both strategies hurt (-$326 vScalpA, -$77 vScalpB)

### Important caveats
- This is post-filter analysis (counting P&L of trades that happened to land on news days)
- A real pre-filter (blocking entries on news days) would change cooldowns/episodes and
  produce different trade sequences — needs to be validated in the actual backtest engine
- Weekly jobless claims (Thursdays) show negative P&L but could be day-of-week effect, not claims-specific

## Next Steps (not yet done)
- Implement news blackout as a **pre-filter** in the backtest engine for vScalpA
- Test per-event-type blackouts (FOMC-only, FOMC+Retail, etc.)
- Consider time-window approach (skip first hour after 8:30 AM releases) vs full-day skip
- Could add news calendar to live engine as an optional pre-trade filter
- MES v2: no filter needed, trades through news profitably
