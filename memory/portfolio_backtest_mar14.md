# Portfolio Backtest — Mar 14, 2026 (Gated + Structure Exit)

First full-period backtest with all production gates + vScalpC structure exit + corrected MES v2 parameters. This is the most accurate historical simulation of the current live system.

## Configuration

Target allocation: A(1) + B(1) + C(2) + MES(2)

| Param | vScalpA | vScalpB | vScalpC | MES v2 |
|-------|---------|---------|---------|--------|
| SM | 10/12/200/100 | 10/12/200/100 | 10/12/200/100 | 20/12/400/255 |
| SM_T | 0.0 | 0.25 | 0.0 | 0.0 |
| RSI | 8/60/40 | 8/55/45 | 8/60/40 | 12/55/45 |
| CD | 20 | 20 | 20 | 25 |
| SL | 40 | 10 | 40 | 35 |
| Exit | TP=7 | TP=3 | TP1=7 + structure | TP1=6, TP2=20 |
| BE_TIME | -- | -- | -- (structure replaces) | 75 |
| SL→BE | -- | -- | after TP1 | -- |
| Entry cutoff | 13:00 | 15:45 | 13:00 | 14:15 |
| EOD | 16:00 | 16:00 | 16:00 | 16:00 |
| Contracts | 1 | 1 | 2 (partial) | 2 (partial) |

### Gates applied

| Gate | A | B | C | MES |
|------|---|---|---|-----|
| Leledc mq=9 | Y | **N** (removed Mar 14) | Y | -- |
| ADR 30% | Y | **N** (removed Mar 14) | Y | -- |
| ATR >= 263.8 | -- | -- | Y | -- |
| Prior-day VPOC+VAL buf=5 | -- | -- | -- | Y |
| VIX death zone 19-22 | Y | -- | Y | -- |

### Gate block rates (full period)

| Gate | Bars blocked |
|------|-------------|
| Leledc mq=9 | 10.5% |
| ADR directional 30% | 5.9% |
| Prior-day ATR < 263.8 | 18.9% |
| VIX death zone 19-22 | 17.4% |
| MES prior-day VPOC+VAL | 20.3% |

## Data

- MNQ: Databento 1-min OHLCV, 2025-02-17 to 2026-03-12 (12.8 months, 270 trading days)
- MES: Databento 1-min OHLCV, same range
- VIX: yfinance daily closes (for death zone gate)
- Volume data used for VPOC computation (MES prior-day levels)

## Results — Full Period

| Strategy | Trades | WR | PF | Sharpe | P&L | MaxDD |
|----------|--------|------|------|--------|-----|-------|
| vScalpA | 273 | 85.3% | 1.598 | 2.885 | +$2,172 | -- |
| vScalpB | 389 (ungated) | 73.5% | 1.435 | 2.151 | +$1,536 | -$281 |
| vScalpC | 211 entries | 79.6% | 1.941 | 3.932 | +$5,112 | -- |
| MES v2 | 384 | -- | 1.435 | 2.283 | +$4,602 | -- |
| **Portfolio** | -- | -- | **2.145** | **4.29** | **+$13,422** | **-$1,420** |

Note: vScalpB shown ungated (gates removed Mar 14). Portfolio total reflects ungated vScalpB.

### vScalpC structure exit breakdown

| Exit reason | Count |
|-------------|-------|
| Structure (pivot swing) | 114 |
| BE (breakeven stop) | 56 |
| TP cap (60pt safety) | 13 |
| SL | remaining |

## Monthly Breakdown

```
  Month         vScalpA   vScalpB   vScalpC    MES_v2 Portfolio
  ----------  ---------- ---------- ---------- ---------- ----------
  2025-02           154        61       338      -435       117
  2025-03             5        86        85     1,336     1,513
  2025-04           254        32       524     1,802     2,613
  2025-05           410        19       459      -330       558
  2025-06            38       242       437       300     1,016
  2025-07           138        51        35       412       636
  2025-08            46         3        58       250       357
  2025-09           175        19         0      -494      -300
  2025-10          -197       104        73       618       598
  2025-11           210        29       751       981     1,970
  2025-12           335       -48       695        -2       980
  2026-01           315       278       951      -392     1,151
  2026-02           386       114     1,133       361     1,994
  2026-03           -97        65      -426       195      -264
```

Note: vScalpB monthly numbers above are from the gated run. Ungated adds ~$483 spread across months. Mar 2026 is partial (~8 trading days).

## Key Observations

1. **12 of 14 months positive** at portfolio level. Only Sep 2025 (-$300) and Mar 2026 partial (-$264).
2. **vScalpC is the portfolio driver** — $5,112 (38% of total). Structure exit captures larger moves scalps leave on the table.
3. **Diversification works** — no single strategy drags the portfolio into deep drawdown. Negative months for one strategy are offset by others.
4. **MES v2 is the most volatile** — swings $1,800 to -$494 monthly. $5/pt amplifies both directions.
5. **Nov-Feb is the strongest stretch** — $6,095 (46% of total in 29% of time).
6. **MaxDD of -$1,420 is manageable** — less than 11% of total P&L.

## vScalpB Gate Removal Analysis

| Metric | Gated | Ungated | Delta |
|--------|-------|---------|-------|
| Trades | 255 | 389 | +134 |
| WR | 74.1% | 73.5% | -0.6pp |
| PF | 1.470 | 1.435 | -0.035 |
| Sharpe | 2.348 | 2.151 | -0.197 |
| P&L | $1,053 | $1,536 | +$483 |
| MaxDD | -$269 | -$281 | -$12 |

Gates blocked 152 trades: 109 winners (72%) vs 43 losers (28%). Net blocked P&L: +$537 (the blocked trades were profitable). Gates cost $483/yr for marginal quality improvement. **Decision: remove gates from vScalpB** — SM_T=0.25 already provides sufficient filtering.

## Known Limitations

1. **TP fills at next-bar open, not limit price** — inflates PF slightly vs production OCO limit fills. On 1-min bars the difference is small but nonzero.
2. **Backtest commissions** — $0.52/side MNQ, $1.25/side MES. Matches tastytrade rates.
3. **No slippage model** — fills at exact open/close. Live fills may differ by 0.25-0.50 pts.
4. **Mar 2026 partial** — only ~8 trading days. Not representative of a full month.
5. **VIX gate depends on yfinance download** — if download fails, gate is disabled (fail-open).

## Corrections Made (vs prior backtests)

- **MES v2 EOD**: Changed from 15:30 ET (v9.4 legacy) to 16:00 ET (production config). Never triggered in v2 but was killing live winners.
- **MES v2 entry cutoff**: Added 14:15 ET (production config). Prior backtests used default 15:45.
- **vScalpC exit**: Changed from fixed TP2=25 to structure exit (pivot LB=50, PR=2, buf=2pts). Matches production.
- **All gates included**: First portfolio backtest with entry gates active. Prior runs were ungated.
