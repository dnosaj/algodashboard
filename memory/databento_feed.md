---
name: Databento Live Feed
description: Databento CME direct feed — architecture, gotchas, costs, and operational notes
type: reference
---

# Databento Live Feed — Reference (March 20, 2026)

## Architecture
- **Data**: Databento GLBX.MDP3 (CME Globex direct, not dxFeed)
- **Orders**: tastytrade (unchanged)
- **Feed and broker are independent** — data comes from Databento, orders go through tastytrade

## Start command
```bash
python3 run.py --broker tastytrade --feed databento
```
- `--feed databento` — Databento CME direct (primary)
- `--feed tastytrade` — DXLink/dxFeed (fallback)
- `--feed auto` — uses Databento if DATABENTO_API_KEY is set

## Cost
- **Live license**: $179/month for CME Globex MDP3.0 (purchased Mar 20, 2026)
- **Historical**: free with license (included)
- **API key**: `DATABENTO_API_KEY` in `.env`

## Files
- Feed adapter: `live_trading/engine/databento_feed.py` (654 lines)
- Config: `data_feed` field in `EngineConfig` (config.py)
- Runner wiring: section 3b in `runner.py`
- Test: `live_trading/tests/test_databento_feed_import.py`
- Historical download: `backtesting_engine/strategies/fetch_databento_data.py`

## How it works

### Warmup (historical bars for SM/RSI)
1. Uses `db.Historical` to fetch ~1250 minutes of 1-min OHLCV
2. Probes available range (tries offsets 0, 15, 30, 45, 60, 90 min) — historical data has ~30 min delay
3. Bars converted to `Bar` dataclass and stored in `_all_bars`
4. Runner calls `get_warmup_bars()` to feed 500 bars to each strategy

### Live streaming
1. `db.Live` client runs in a daemon thread
2. Subscribes to `ohlcv-1m` (bars) and `mbp-1` (quotes) for MNQ.c.0 + MES.c.0
3. `start=` param set to warmup period ago for gap backfill
4. Callbacks route records: OHLCVMsg → Bar → bar_queue, MBP1Msg → QuoteTick → quote_queue
5. Cross-thread: `loop.call_soon_threadsafe()` pushes to asyncio queues
6. Infinite retry on connection failure (max_retries=999)

### Symbol mapping
- MNQ → MNQ.c.0 (continuous front-month)
- MES → MES.c.0
- SymbolMappingMsg maps instrument_id to instrument name at connection time

## Gotchas learned the hard way

### Prices are fixed-point (1e9 scale)
Raw `record.open` = 24113500000000 (int). Multiply by 1e-9 to get 24113.50.
The `PRICE_SCALE = 1e-9` constant handles this.
The Python iterator auto-converts, but callbacks get raw values.

### Must call `live_client.start()` before `block_for_close()`
Without `start()`, the live client subscribes but never begins receiving data.
Discovered Mar 20 — callbacks weren't firing until this was added.

### SymbolMappingMsg has no `.hd` attribute
Use `getattr(msg, 'instrument_id', None)` — don't access `msg.hd.instrument_id`.
SDK 0.71.0 changed the record structure.

### Historical data has ~30 min delay
`db.Historical` data lags real-time by ~30 min. Today's RTH data isn't available
in historical API until overnight. Use `db.Live` with `start=` param for same-day
backfill — this pulls historical bars through the live stream.

### Live backfill is sequential
When using `start=` on the live subscription, bars arrive one-by-one at ~1 bar/second.
A full day (1300 bars) takes ~20 minutes to backfill. Not instant.

### UNDEF_PRICE = 2^63 - 1
Undefined/empty bars have `open = 9223372036854775807`. Filter these out.

### No `next_record()` in SDK 0.71.0
Use `for record in live:` iterator, not `live.next_record()`.

## DXLink comparison

| | Databento | DXLink (tastytrade) |
|---|---|---|
| Source | CME direct (GLBX.MDP3) | dxFeed (3rd party) |
| Reliability | High (no outages observed) | Low (multiple outages Mar 16-20) |
| Cost | $179/month | Free with tastytrade account |
| Warmup | Historical API + live backfill | Live stream lookback |
| Latency | Low | Low |
| Quote data | MBP-1 (top of book) | DXLink Quote |
| Empty candles? | Never observed | Sent empty O=0 H=0 L=0 C=0 on Mar 20 |

## Why we switched
- Mar 16: dxFeed "Session not found: api" — all-day outage
- Mar 19: Feed worked but DXLink reconnect bug prevented auto-recovery
- Mar 20: dxFeed sending empty candles (O=0 H=0 L=0 C=0) — entire day lost
- Tastytrade status page showed "all systems operational" during all outages
- dxFeed issues are infrastructure-level, not fixable on our side
