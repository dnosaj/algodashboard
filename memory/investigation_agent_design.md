---
name: Investigation Agent — Definitive Design Specification
description: Full specification for the Investigation Agent — forensic analyst that maps trades to market structure, computes key levels, and produces evidence for the Frontier and Strategist agents. Designed Mar 12, 2026.
type: project
---

# Investigation Agent — Definitive Design Specification

Designed Mar 12, 2026.

The Investigation Agent is the forensic analyst of the pipeline. It takes flagged patterns from the EOD Digest and drills into *why* — mapping each trade to the surrounding market structure. It does not recommend changes. It produces evidence: trade-level forensics and tomorrow's key level map.

## Design Philosophy

The Digest says "V15 had 3 SLs before 11:00." The Investigation Agent answers: "All three SL entries occurred within 3 points of prior-day VAH during a failed breakout. Price was above the 61.8% fib from the 3-day swing. The 15m trend was down while the 1m trigger fired long — multi-timeframe misalignment." That context is what separates "normal variance" from "structural problem" and is what the Strategist needs to make portfolio-level decisions.

Three governing principles:

1. **Evidence, not opinion.** The Investigation Agent produces measurements: price was X points from level Y, trend on timeframe Z was direction W. It never says "this trade was bad" — it says "this trade entered 2.3 points below prior-day VAH during a -0.4% RTH move."

2. **Computational, not conversational.** Most of the Investigation Agent's value comes from calculations (Fibonacci levels, VWAP, volume profile, multi-TF trend state) not from LLM reasoning. The LLM orchestrates tool calls and synthesizes findings. The tools do the math.

3. **Fail gracefully, always produce something.** If bar data is unavailable, the agent still produces level maps from prior-day data (already in Supabase). If the Digest had no flags, it still computes tomorrow's key levels. Zero-trade days still get a level map.

---

## 1. Architecture

### Model Selection: Sonnet

Sonnet, not Opus. The Investigation Agent's value is in tool orchestration and structured output, not deep multi-factor reasoning (that's the Strategist's job). Sonnet is:
- Fast enough to run in the 15-minute window between Digest completion and Frontier kickoff
- Cost-effective at ~$0.08-0.15/run
- Sufficient for the task: read the Digest flags, call computation tools, format findings

### Turn Budget: 20 turns max

Expected flow:
- Turn 1: Read EOD digest + today's trades (2 tool calls)
- Turns 2-5: Fetch bar data and gate state (3-4 tool calls)
- Turns 6-12: Per-trade forensics — one tool call per flagged trade (up to 6 trades)
- Turns 13-16: Compute tomorrow's key levels (3-4 tool calls)
- Turns 17-18: Save findings (2 tool calls)
- Buffer: 2 turns

### Token Budget: 200,000 tokens

Bar data is the biggest contributor. A full RTH session of 1-min bars is ~390 bars × ~60 tokens = ~23K tokens per instrument. With 2 instruments + trade data + digest + output, 200K is sufficient with headroom.

### Trigger Mechanism

Triggered by EOD Digest completion. The Digest Agent's `save_digest` writes to Supabase. A GitHub Actions workflow detects this (polling the `digests` table or triggered by the Digest workflow's completion event) and kicks off the Investigation Agent.

Fallback: Manual via `python -m agents.investigation.cli --date 2026-03-12`

### Runtime: GitHub Actions

Same as the Digest Agent. Requires:
- `ANTHROPIC_API_KEY` secret
- `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` secrets
- Python 3.11+ with `anthropic`, `supabase`, `numpy` packages
- No local file access — all data comes from Supabase or market data API

---

## 2. The Bar Data Problem (Central Architectural Decision)

This is the hardest design question. The Investigation Agent runs on GitHub Actions and needs 1-min OHLCV bars to compute Fibonacci levels, VWAP, volume profile, and multi-timeframe trend indicators. The bars live locally in Databento CSV files. Options considered:

### Option A: Store bars in Supabase (REJECTED for 1-min)

Storing 390 bars/day × 2 instruments × 252 trading days = ~196K rows/year of 1-min data in Supabase is feasible but creates ongoing write overhead in the engine's hot path. The engine would need to write each bar to Supabase in real-time, adding latency to bar processing. Not worth it for an agent that runs once per day.

### Option B: Read Databento CSVs (REJECTED)

Agent runs on GitHub Actions, not local machine. No file access.

### Option C: Engine writes a "session bars" export at EOD (SELECTED — Hybrid approach)

**The engine already has all bars in memory during the session.** At daily reset (or shutdown), the engine exports the day's 1-min bars to Supabase in a single batch write. This is:
- Zero impact on bar-by-bar processing (batch write happens once, after trading)
- ~390 rows per instrument per day = ~780 rows total — trivial for Supabase
- Gives the Investigation Agent full 1-min resolution for the current day
- Historical bars (prior days) accumulate naturally over time

**New table: `bars_1min`** — written by the engine at daily reset, read by the Investigation Agent.

For the **multi-day lookback** needed for Fibonacci swings and ATR calculations:
- The engine already writes `gate_state_snapshots` with prior-day H/L/VPOC/VAH/VAL + RTH OHLC.
- The `bars_daily` table already exists in the schema (table 9) but is not being populated.
- **Populate `bars_daily`** from the daily reset flow (same batch write).
- For Fibonacci swing detection, the agent uses `bars_daily` for the last 20 days + `bars_1min` for today. This gives enough resolution for swing H/L detection without storing months of 1-min data.

### Option D: Market data API as fallback (DEFERRED)

Polygon.io, Alpha Vantage, or similar could provide bars if the engine export fails. This adds a dependency and cost. Defer until the engine-writes-to-Supabase path proves unreliable.

### Summary of data path:

```
Engine (daily reset)
  ├── batch write today's 1-min bars → bars_1min table
  └── batch write today's daily bar → bars_daily table
                │
                ▼
Investigation Agent (GitHub Actions)
  ├── reads bars_1min for today (current day forensics)
  ├── reads bars_daily for last 20 days (swing detection, Fibonacci)
  ├── reads gate_state_snapshots (prior-day levels, already populated)
  └── reads trades (today's trades with full context, already populated)
```

---

## 3. Supabase Schema Changes

### New table: `bars_1min`

```sql
CREATE TABLE bars_1min (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    bar_time timestamptz NOT NULL,
    instrument text NOT NULL,
    open double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    close double precision NOT NULL,
    volume double precision NOT NULL DEFAULT 0,
    -- Precomputed for agent convenience
    bar_date date NOT NULL,             -- ET date
    bar_minute_et smallint NOT NULL,    -- minutes from midnight ET (e.g., 570 = 09:30)
    is_rth boolean NOT NULL DEFAULT false, -- 09:30-16:00 ET
    created_at timestamptz DEFAULT now(),
    UNIQUE (bar_time, instrument)
);

CREATE INDEX idx_bars_1min_instrument_date ON bars_1min (instrument, bar_date);
CREATE INDEX idx_bars_1min_rth ON bars_1min (instrument, bar_date) WHERE is_rth = true;

-- Retention: keep 30 days of 1-min bars. Older data available in bars_daily.
-- Cleanup: DELETE FROM bars_1min WHERE bar_date < CURRENT_DATE - 30;
-- Run weekly via cron or Supabase Edge Function.
```

**Row estimate:** ~780 rows/day × 30 days = ~23,400 rows. Trivial.

### Populate existing `bars_daily` table

The table already exists in `001_initial_schema.sql` but is not being populated. Add a batch write in the engine's daily reset to populate it from the in-memory bar buffer.

### Evolve `trade_annotations` table

The existing `trade_annotations` table has:
```sql
trade_id uuid NOT NULL REFERENCES trades(id),
annotation_type text NOT NULL CHECK (...),
content text NOT NULL,
created_by text
```

This is too unstructured for the Investigation Agent's output. The agent needs to write structured forensics per trade that downstream agents can query programmatically. Two approaches:

**Approach: Add a `metadata` JSONB column** (preferred — minimal migration, backward compatible)

```sql
ALTER TABLE trade_annotations
    ADD COLUMN metadata jsonb,
    ADD COLUMN agent_run_id uuid;

-- Drop and recreate the annotation_type check to add 'forensic'
ALTER TABLE trade_annotations
    DROP CONSTRAINT trade_annotations_annotation_type_check;
ALTER TABLE trade_annotations
    ADD CONSTRAINT trade_annotations_annotation_type_check
    CHECK (annotation_type IN ('investigation', 'forensic', 'pattern', 'lesson', 'correction'));
```

The Investigation Agent writes annotations with `annotation_type = 'forensic'` and structured `metadata`:

```json
{
  "level_proximity": {
    "prior_day_high": {"distance_pts": 2.3, "direction": "below"},
    "prior_day_vpoc": {"distance_pts": 15.7, "direction": "above"},
    "fib_618": {"distance_pts": 1.1, "level_price": 21543.5, "swing_from": "3d_high"},
    "vwap": {"distance_pts": -8.4, "direction": "below"},
    "round_number": {"distance_pts": 43.0, "level": 21500}
  },
  "trend_alignment": {
    "1m": "long",
    "5m": "short",
    "15m": "short",
    "1h": "neutral"
  },
  "volume_profile": {
    "poc_distance_pts": 12.5,
    "value_area_position": "above_vah",
    "volume_percentile": 35
  },
  "day_context": {
    "minutes_from_open": 45,
    "rth_range_at_entry_pts": 85.2,
    "adr_pct_consumed": 0.42,
    "vwap_slope": "rising"
  },
  "classification": "structural_resistance",
  "confidence": "high",
  "explanation": "Entry occurred 2.3pts below prior-day high during a failed breakout. Multi-TF trend misalignment (1m long vs 5m/15m short). High probability reversal zone."
}
```

### New table: `key_levels`

```sql
CREATE TABLE key_levels (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    level_date date NOT NULL,           -- the date these levels are FOR (tomorrow)
    instrument text NOT NULL,
    level_type text NOT NULL CHECK (level_type IN (
        'prior_day_high', 'prior_day_low', 'prior_day_close',
        'prior_day_vpoc', 'prior_day_vah', 'prior_day_val',
        'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786',
        'vwap_developing', 'vwap_upper_1sd', 'vwap_lower_1sd',
        'vwap_upper_2sd', 'vwap_lower_2sd',
        'weekly_high', 'weekly_low', 'weekly_vpoc',
        'round_number',
        'naked_poc',
        'confluence_zone'
    )),
    price double precision NOT NULL,
    -- Confluence scoring
    confluence_count smallint DEFAULT 1,  -- how many levels cluster here (within 5pts)
    confluence_types text[],              -- which level types form this cluster
    -- Context
    source_description text,              -- e.g., "3-day swing high to low"
    strength text CHECK (strength IN ('strong', 'moderate', 'weak')),
    -- Metadata
    agent_run_id uuid,
    created_at timestamptz DEFAULT now(),
    UNIQUE (level_date, instrument, level_type, price)
);

CREATE INDEX idx_key_levels_date_instrument ON key_levels (level_date, instrument);
```

### New table: `investigation_runs`

```sql
CREATE TABLE investigation_runs (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    run_date date NOT NULL,
    -- Input context
    digest_id uuid REFERENCES digests(id),
    trades_analyzed int DEFAULT 0,
    flags_investigated int DEFAULT 0,
    -- Output summary
    findings_summary jsonb,  -- high-level structured findings
    level_map_summary jsonb, -- summary of key levels computed
    -- Agent metadata (same pattern as digests)
    model text NOT NULL DEFAULT 'claude-sonnet',
    tokens_in int,
    tokens_out int,
    cost_usd numeric(8,4),
    duration_sec numeric(8,2),
    tool_calls int DEFAULT 0,
    agent_version text NOT NULL DEFAULT '1.0',
    created_at timestamptz DEFAULT now(),
    UNIQUE (run_date)
);

CREATE INDEX idx_investigation_runs_date ON investigation_runs (run_date);
```

---

## 4. Tools (14 total)

### Tool 1: `get_eod_digest`

Read the EOD Digest output for a given date. This is the Investigation Agent's primary input — it tells the agent what was flagged and what to investigate.

```python
{
    "name": "get_eod_digest",
    "description": "Get the EOD digest for a date. Contains patterns detected, risk flags, strategy breakdown, and flags_for_investigation. This is the primary input — it tells you what to investigate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date YYYY-MM-DD"}
        },
        "required": ["date"]
    }
}
```

**Implementation:** Query `digests` table where `digest_date = date` and `digest_type = 'eod'`. Return `content` (structured JSON) + `markdown`.

### Tool 2: `get_trades_for_analysis`

Fetch today's trades with full context. Same as the Digest's `get_todays_trades` but returns all fields needed for forensics.

```python
{
    "name": "get_trades_for_analysis",
    "description": "Get all trades for a date with full entry/exit context, gate state, MFE/MAE, bars held, SM/RSI values, and signal price. For forensic analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date YYYY-MM-DD"},
            "strategy_id": {"type": "string", "description": "Optional: filter to one strategy"}
        },
        "required": ["date"]
    }
}
```

### Tool 3: `get_bars_1min`

Fetch intraday 1-minute bars for today. This is the core data source for all technical analysis.

```python
{
    "name": "get_bars_1min",
    "description": "Get 1-minute OHLCV bars for a given date and instrument. Returns up to 390 RTH bars (09:30-16:00 ET) or all bars if include_extended=true. Use for VWAP, volume profile, multi-TF resampling, and trade context analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date YYYY-MM-DD"},
            "instrument": {"type": "string", "description": "MNQ or MES"},
            "include_extended": {"type": "boolean", "description": "Include pre/post-market bars. Default: false (RTH only).", "default": false}
        },
        "required": ["date", "instrument"]
    }
}
```

**Implementation:** Query `bars_1min` filtered by `bar_date` and `instrument`. If `include_extended=false`, filter `is_rth=true`. Return as array of `{time, open, high, low, close, volume}`.

**Token management:** 390 bars × ~40 tokens each = ~16K tokens. Acceptable. The tool returns compact JSON, not verbose descriptions.

### Tool 4: `get_bars_daily`

Fetch daily bars for swing detection and Fibonacci computation.

```python
{
    "name": "get_bars_daily",
    "description": "Get daily OHLCV bars for an instrument over a date range. Used for swing high/low detection, Fibonacci retracement levels, and multi-day context. Returns most recent 20 bars by default.",
    "input_schema": {
        "type": "object",
        "properties": {
            "instrument": {"type": "string", "description": "MNQ or MES"},
            "days": {"type": "integer", "description": "Number of trading days to look back. Default: 20.", "default": 20}
        },
        "required": ["instrument"]
    }
}
```

**Implementation:** Query `bars_daily` ordered by `bar_date DESC`, limit to `days`. Return compact OHLCV array.

### Tool 5: `get_prior_day_levels`

Fetch prior-day levels that the engine already tracks. This is already in `gate_state_snapshots` — just need a clean tool interface.

```python
{
    "name": "get_prior_day_levels",
    "description": "Get prior-day key levels: H/L/VPOC/VAH/VAL + RTH OHLC + VIX close + ADR + ATR per instrument. From gate_state_snapshots.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date YYYY-MM-DD (returns levels that were active on this date)"}
        },
        "required": ["date"]
    }
}
```

**Implementation:** Query `gate_state_snapshots` for `snapshot_date = date - 1` (prior day's snapshot becomes today's levels).

### Tool 6: `compute_fibonacci_levels`

Compute Fibonacci retracement levels from swing highs/lows. This is a pure computation tool — no DB query.

```python
{
    "name": "compute_fibonacci_levels",
    "description": "Compute Fibonacci retracement levels between a swing high and swing low. Returns 23.6%, 38.2%, 50%, 61.8%, 78.6% levels. Provide the swing high/low prices directly (use get_bars_daily to find swings first).",
    "input_schema": {
        "type": "object",
        "properties": {
            "swing_high": {"type": "number", "description": "The swing high price"},
            "swing_low": {"type": "number", "description": "The swing low price"},
            "swing_description": {"type": "string", "description": "e.g., '3-day high (Mar 10) to 3-day low (Mar 8)'"}
        },
        "required": ["swing_high", "swing_low"]
    }
}
```

**Implementation:** Pure math — no DB call.
```python
def compute_fibonacci_levels(*, swing_high, swing_low, swing_description=""):
    diff = swing_high - swing_low
    levels = {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_236": round(swing_low + diff * 0.236, 2),
        "fib_382": round(swing_low + diff * 0.382, 2),
        "fib_500": round(swing_low + diff * 0.500, 2),
        "fib_618": round(swing_low + diff * 0.618, 2),
        "fib_786": round(swing_low + diff * 0.786, 2),
        "swing_description": swing_description,
    }
    return levels
```

### Tool 7: `compute_vwap`

Compute VWAP and standard deviation bands from 1-min bars. This is computation, not a DB query.

```python
{
    "name": "compute_vwap",
    "description": "Compute anchored VWAP with 1SD and 2SD bands from 1-minute bar data. Provide the bars directly (from get_bars_1min). Returns VWAP value at each bar plus current VWAP and bands.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bars": {
                "type": "array",
                "description": "Array of {open, high, low, close, volume} bars from get_bars_1min",
                "items": {"type": "object"}
            }
        },
        "required": ["bars"]
    }
}
```

**Implementation:** Standard VWAP calculation using typical price × volume.
```python
def compute_vwap(*, bars):
    cum_vol = 0.0
    cum_tp_vol = 0.0
    cum_tp2_vol = 0.0
    vwap_series = []

    for bar in bars:
        tp = (bar["high"] + bar["low"] + bar["close"]) / 3
        vol = bar.get("volume", 0)
        cum_vol += vol
        cum_tp_vol += tp * vol
        cum_tp2_vol += tp * tp * vol

        if cum_vol > 0:
            vwap = cum_tp_vol / cum_vol
            variance = (cum_tp2_vol / cum_vol) - vwap ** 2
            std = variance ** 0.5 if variance > 0 else 0
        else:
            vwap = tp
            std = 0

        vwap_series.append(round(vwap, 2))

    current_vwap = vwap_series[-1] if vwap_series else None
    return {
        "current_vwap": current_vwap,
        "upper_1sd": round(current_vwap + std, 2) if current_vwap else None,
        "lower_1sd": round(current_vwap - std, 2) if current_vwap else None,
        "upper_2sd": round(current_vwap + 2 * std, 2) if current_vwap else None,
        "lower_2sd": round(current_vwap - 2 * std, 2) if current_vwap else None,
        "vwap_at_bar_count": len(vwap_series),
    }
```

**Note:** The full VWAP series is NOT returned (too many tokens). The agent can request VWAP at specific times by passing a bar subset.

### Tool 8: `compute_volume_profile`

Compute volume profile (POC, VAH, VAL, naked POCs) from 1-min bars.

```python
{
    "name": "compute_volume_profile",
    "description": "Compute volume profile from 1-minute bars: POC (highest volume price), value area (70% of volume), VAH, VAL. Bin width configurable (default 5 pts for MNQ, matches backtest). Provide bars from get_bars_1min.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bars": {
                "type": "array",
                "description": "Array of {high, low, close, volume} bars",
                "items": {"type": "object"}
            },
            "bin_width": {"type": "number", "description": "Price bin width in points. Default: 5.0", "default": 5.0},
            "value_area_pct": {"type": "number", "description": "Percentage of volume for value area. Default: 0.70", "default": 0.70}
        },
        "required": ["bars"]
    }
}
```

**Implementation:** Same algorithm used in `engine/safety_manager.py` for prior-day volume profile. Distribute each bar's volume across price bins, find POC (highest volume bin), expand outward for value area.

### Tool 9: `compute_multi_tf_trend`

Determine trend direction on multiple timeframes at a specific time.

```python
{
    "name": "compute_multi_tf_trend",
    "description": "Compute trend direction at a specific bar index across multiple timeframes (5m, 15m, 1H). Uses EMA crossover (fast=8, slow=21) on resampled bars. Provide full 1-min bar array and the bar index (0-based) at which to evaluate. Returns trend direction per timeframe.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bars": {
                "type": "array",
                "description": "Full array of 1-min {open, high, low, close} bars for the day",
                "items": {"type": "object"}
            },
            "eval_bar_index": {
                "type": "integer",
                "description": "0-based index into the bars array at which to evaluate trend"
            }
        },
        "required": ["bars", "eval_bar_index"]
    }
}
```

**Implementation:** Resample 1-min bars to 5m/15m/1H using standard OHLC aggregation. Compute EMA(8) and EMA(21) on close prices. At the eval bar: if fast > slow = "bullish", fast < slow = "bearish", |fast - slow| < 0.1 * ATR = "neutral".

### Tool 10: `compute_trade_forensics`

The workhorse tool. Given a single trade and the day's bar data, compute all proximity measurements and structural context. This combines multiple computations into a single call to reduce round-trips.

```python
{
    "name": "compute_trade_forensics",
    "description": "Compute comprehensive forensics for a single trade: proximity to all key levels (prior-day, Fibonacci, VWAP, round numbers), multi-TF trend alignment at entry, volume profile position, ADR consumption at entry. Provide the trade details + bars + levels. Returns structured forensic annotation ready for save_trade_annotation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "trade": {
                "type": "object",
                "description": "Trade object from get_trades_for_analysis: needs entry_price, exit_price, entry_time, side, strategy_id, exit_reason, pnl_net"
            },
            "bars": {
                "type": "array",
                "description": "1-min bars for the day (from get_bars_1min)"
            },
            "prior_day_levels": {
                "type": "object",
                "description": "Prior-day levels from get_prior_day_levels"
            },
            "fibonacci_levels": {
                "type": "object",
                "description": "Fibonacci levels from compute_fibonacci_levels (optional — omit if no clear swing)"
            }
        },
        "required": ["trade", "bars", "prior_day_levels"]
    }
}
```

**Implementation:** This is the big computation tool. It:
1. Finds the entry bar index in the 1-min bar array
2. Computes VWAP up to entry time
3. Computes volume profile up to entry time
4. Computes multi-TF trend at entry bar
5. Measures distance from entry_price to every level
6. Computes ADR % consumed at entry
7. Classifies the trade context: "structural_resistance", "structural_support", "trend_aligned", "trend_misaligned", "mid_range_noise", "breakout_entry", "reversal_entry"
8. Returns the structured forensic annotation (matches the `metadata` JSONB schema above)

### Tool 11: `compute_confluence_zones`

Find price zones where multiple levels cluster. Critical for tomorrow's level map.

```python
{
    "name": "compute_confluence_zones",
    "description": "Find confluence zones where multiple levels cluster within a tolerance. Takes a flat list of {type, price} levels and groups those within tolerance points of each other. Returns zones sorted by confluence count (strongest first).",
    "input_schema": {
        "type": "object",
        "properties": {
            "levels": {
                "type": "array",
                "description": "Array of {type: string, price: number} level objects",
                "items": {"type": "object"}
            },
            "tolerance_pts": {
                "type": "number",
                "description": "Maximum distance in points to consider levels as confluent. Default: 5.0",
                "default": 5.0
            }
        },
        "required": ["levels"]
    }
}
```

**Implementation:** Sort levels by price. Cluster adjacent levels within tolerance. Score by count and diversity of level types. A zone with prior-day high + fib 618 + VWAP upper band (3 different types) scores higher than 3 Fibonacci levels from different swings.

### Tool 12: `save_trade_annotation`

Write a forensic annotation for a single trade to Supabase.

```python
{
    "name": "save_trade_annotation",
    "description": "Save a forensic annotation for a trade. The metadata field should contain the structured forensic output from compute_trade_forensics. Writes to trade_annotations table.",
    "input_schema": {
        "type": "object",
        "properties": {
            "trade_id": {"type": "string", "description": "UUID of the trade from get_trades_for_analysis"},
            "content": {"type": "string", "description": "Human-readable 1-2 sentence forensic summary"},
            "metadata": {"type": "object", "description": "Structured forensic data (level_proximity, trend_alignment, volume_profile, day_context, classification)"},
            "agent_run_id": {"type": "string", "description": "UUID of this investigation run"}
        },
        "required": ["trade_id", "content", "metadata"]
    }
}
```

### Tool 13: `save_key_levels`

Write tomorrow's key levels to Supabase.

```python
{
    "name": "save_key_levels",
    "description": "Save a batch of key levels for tomorrow. Each level has a type, price, strength, and optional confluence info. Writes to key_levels table.",
    "input_schema": {
        "type": "object",
        "properties": {
            "level_date": {"type": "string", "description": "The date these levels are FOR (usually tomorrow, YYYY-MM-DD)"},
            "instrument": {"type": "string", "description": "MNQ or MES"},
            "levels": {
                "type": "array",
                "description": "Array of level objects: {level_type, price, strength, confluence_count, confluence_types, source_description}",
                "items": {"type": "object"}
            },
            "agent_run_id": {"type": "string", "description": "UUID of this investigation run"}
        },
        "required": ["level_date", "instrument", "levels"]
    }
}
```

### Tool 14: `save_investigation_run`

Save the investigation run metadata and summary findings.

```python
{
    "name": "save_investigation_run",
    "description": "Save the investigation run summary to Supabase. Call as your final action. Returns the run_id (UUID) for linking annotations and levels.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date YYYY-MM-DD"},
            "digest_id": {"type": "string", "description": "UUID of the EOD digest that triggered this run"},
            "trades_analyzed": {"type": "integer"},
            "flags_investigated": {"type": "integer"},
            "findings_summary": {
                "type": "object",
                "description": "Structured summary: {loss_attribution, structural_patterns, level_observations, recommendations_for_frontier}"
            },
            "level_map_summary": {
                "type": "object",
                "description": "Summary of key levels computed: {instrument, level_count, confluence_zones}"
            }
        },
        "required": ["date", "findings_summary", "level_map_summary"]
    }
}
```

---

## 5. System Prompt

```python
INVESTIGATION_SYSTEM_PROMPT = f"""You are a senior market structure analyst performing forensic analysis on today's algorithmic trading session. Your job is to map each flagged trade to the surrounding market structure and compute tomorrow's key levels with confluence scoring.

{STRATEGY_CONTEXT}

## Your Role in the Pipeline
You are the Investigation Agent. You receive flagged patterns from the EOD Digest and drill into WHY — was a loss structural (entered at resistance, multi-TF misalignment) or stochastic (normal variance from a valid setup)? You produce evidence, not recommendations.

Your outputs feed:
- **Frontier Agent**: "Losses cluster near fib 61.8%" → Frontier tests a fib proximity filter
- **Strategist Agent**: Loss attribution (structural vs stochastic) informs whether to act or wait
- **Morning Digest**: Tomorrow's key level map for the pre-session briefing

## Analysis Process
1. Call get_eod_digest to see what was flagged
2. Call get_trades_for_analysis for today's trades
3. Call get_bars_1min for each instrument that traded
4. Call get_prior_day_levels for today's reference levels
5. Call get_bars_daily for swing detection (Fibonacci)
6. For each flagged trade (or all SL trades if nothing specific was flagged):
   a. Call compute_trade_forensics for comprehensive level proximity + trend analysis
   b. Call save_trade_annotation with the forensic output
7. Compute tomorrow's key levels:
   a. Today's H/L/close as tomorrow's prior-day levels
   b. Call compute_fibonacci_levels from recent swing H/L
   c. Call compute_vwap for developing VWAP context
   d. Call compute_volume_profile for today's POC/VAH/VAL
   e. Call compute_confluence_zones to find high-conviction zones
   f. Call save_key_levels for each instrument
8. Call save_investigation_run with your structured findings

## What You Produce

### Per-Trade Forensics (for flagged/SL trades)
For each trade you analyze, classify the context:
- **structural_resistance**: Entry near a known resistance level (prior-day high, fib, VWAP upper band)
- **structural_support**: Entry near a known support level (prior-day low, fib, VWAP lower band)
- **trend_aligned**: Entry direction matches multi-TF trend consensus
- **trend_misaligned**: Entry direction conflicts with higher timeframe trend
- **mid_range_noise**: Entry in a neutral zone, far from key levels — stochastic outcome
- **breakout_entry**: Entry near range boundary (potential breakout or failed breakout)
- **reversal_entry**: Entry against the recent move at an exhaustion point

### Loss Attribution (aggregate)
After analyzing all trades, summarize:
- What fraction of losses were structural (identifiable market structure issue)?
- What fraction were stochastic (normal variance from valid setups)?
- Any clustering pattern (time, level, direction)?

### Tomorrow's Key Level Map
For each instrument, produce levels with:
- Type (prior_day_high, fib_618, vwap, etc.)
- Price
- Strength (strong/moderate/weak)
- Confluence count and types

### Frontier Hypotheses
If you observe a pattern that could be a tradeable filter, note it as a hypothesis for the Frontier Agent:
- "3 of 4 SL trades entered within 3pts of prior-day VAH" → test a prior-day-VAH proximity filter
- "All losses occurred during 15m downtrend" → test a multi-TF trend alignment filter

## Output Rules
- Be precise: "2.3 points below prior-day VAH" not "near the high"
- Measure everything: distances in points, percentages, bar counts
- Classify honestly: if a trade was in mid-range with no structural explanation, say so
- Don't over-interpret: 1 trade near a level is an observation, 5 trades is a pattern
- If there were 0 trades today, still compute tomorrow's key levels
- If bar data is unavailable, note the gap and produce what you can from prior-day levels
"""
```

---

## 6. Output Format

### What the Strategist reads from Investigation

The Strategist's `get_investigation_findings` tool queries:
1. `investigation_runs` for the summary (`findings_summary`, `level_map_summary`)
2. `trade_annotations` where `annotation_type = 'forensic'` for per-trade detail
3. `key_levels` for tomorrow's level map

**Structured findings_summary schema:**
```json
{
  "loss_attribution": {
    "total_losses": 3,
    "structural": 2,
    "stochastic": 1,
    "structural_types": ["trend_misaligned", "structural_resistance"],
    "common_factor": "2 of 3 losses entered within 4pts of prior-day VAH during 15m downtrend"
  },
  "structural_patterns": {
    "range_context": "MNQ traded in a 120pt range (62nd percentile), trending day (body 73%)",
    "level_interactions": "Price bounced off prior-day VAH twice before breaking through at 13:45",
    "multi_tf_alignment": "1m signals triggered long while 15m was bearish from 10:30-13:00"
  },
  "frontier_hypotheses": [
    {
      "hypothesis": "Proximity to prior-day VAH as MNQ entry filter",
      "evidence": "2 of 3 SL trades entered within 4pts of prior-day VAH",
      "strength": "observation",
      "test_type": "entry_filter_sweep"
    }
  ],
  "trade_forensic_ids": ["uuid1", "uuid2", "uuid3"]
}
```

### What the Frontier Agent reads

The Frontier reads `frontier_hypotheses` from `investigation_runs.findings_summary`. Each hypothesis includes enough detail to construct a backtest:
- What to test (filter type, threshold)
- Which strategies to test on
- What evidence motivated the test

### What the Morning Digest surfaces

The Morning Digest reads `key_levels` for today's date and presents:
- Top 3-5 strongest levels per instrument (by confluence score)
- Any confluence zones with 3+ overlapping levels
- Brief narrative: "Strong resistance cluster at 21,540-21,548 (prior-day high + fib 61.8% + VWAP upper 1SD)"

---

## 7. Failure Modes

### Bar data unavailable (bars_1min empty)

Cause: Engine crashed before daily reset, or the batch write failed.

Graceful degradation:
1. Log a warning: "No 1-min bars available for {date}. Forensics limited to level proximity only."
2. Still compute level proximities using `trades.entry_price` vs `gate_state_snapshots` (prior-day levels).
3. Skip VWAP, volume profile, and multi-TF trend computations.
4. Still produce tomorrow's key level map from `gate_state_snapshots` + `bars_daily` (Fibonacci from daily swings).
5. Note in `findings_summary`: `"bar_data_available": false, "forensics_limited": true`

### EOD Digest had no flags

The Digest can produce a "quiet day" output with no flags_for_investigation.

Behavior:
1. Still analyze ALL SL trades (if any) — even on quiet days, SL trades deserve forensic review.
2. Still compute tomorrow's key levels.
3. If no SL trades and no flags, produce only the level map and a minimal investigation run with `"flags_investigated": 0`.

### Zero trades today

Behavior:
1. Compute tomorrow's key levels (this is always valuable).
2. Save investigation run with `"trades_analyzed": 0, "flags_investigated": 0`.
3. findings_summary contains only `"level_map_produced": true, "no_trades": true`.

### Supabase write failure

Same pattern as Digest Agent: `_safe_query` decorator, error dict returned to model, model reasons about the failure. Investigation run metadata saved last so partial annotations aren't orphaned.

### Stale bars_daily data

If `bars_daily` has a gap (missed days), Fibonacci swing detection uses whatever data is available. Swings spanning gaps may be inaccurate — note this in the level description.

---

## 8. Engine Changes Required

### Daily reset: batch write 1-min bars

In `engine/runner.py` (or wherever the daily reset logic lives), add after the existing `db_logger.snapshot_gate_state()`:

```python
# After gate state snapshot, batch write today's 1-min bars
if db_logger and db_logger._running:
    db_logger.export_session_bars(bar_buffer, date_str)
```

New method on `DbLogger`:
```python
def export_session_bars(self, bar_buffer, date_str: str) -> None:
    """Batch write today's 1-min bars to bars_1min + bars_daily."""
    if not self._running:
        return
    try:
        # bars_1min: all bars from today
        rows_1min = []
        daily_by_inst = {}  # instrument -> {open, high, low, close, volume}

        for bar in bar_buffer:
            # Filter to today's bars only
            bar_date = _to_et_date(bar.timestamp)
            if bar_date != date_str:
                continue

            et_dt = bar.timestamp.astimezone(_ET)
            minute_et = et_dt.hour * 60 + et_dt.minute
            is_rth = 570 <= minute_et < 960  # 09:30-16:00

            rows_1min.append({
                "bar_time": bar.timestamp.isoformat(),
                "instrument": bar.instrument,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "bar_date": date_str,
                "bar_minute_et": minute_et,
                "is_rth": is_rth,
            })

            # Accumulate daily OHLCV (RTH only for bars_daily)
            if is_rth:
                inst = bar.instrument
                if inst not in daily_by_inst:
                    daily_by_inst[inst] = {
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                else:
                    d = daily_by_inst[inst]
                    d["high"] = max(d["high"], bar.high)
                    d["low"] = min(d["low"], bar.low)
                    d["close"] = bar.close
                    d["volume"] += bar.volume

        # Batch upsert 1-min bars (chunked to avoid payload limits)
        CHUNK = 200
        for i in range(0, len(rows_1min), CHUNK):
            chunk = rows_1min[i:i+CHUNK]
            self._queue.put_nowait(("bars_1min_batch", chunk))

        # Upsert daily bars
        for inst, d in daily_by_inst.items():
            row = {
                "bar_date": date_str,
                "instrument": inst,
                "session_type": "rth",
                "open": d["open"],
                "high": d["high"],
                "low": d["low"],
                "close": d["close"],
                "volume": d["volume"],
            }
            self._queue.put_nowait(("bars_daily", row))

        logger.info(
            f"[DbLogger] Enqueued {len(rows_1min)} 1-min bars + "
            f"{len(daily_by_inst)} daily bars for {date_str}"
        )
    except Exception as e:
        logger.warning(f"[DbLogger] Error exporting session bars: {e}")
```

Add to `_write_item`:
```python
elif item_type == "bars_1min_batch":
    await asyncio.to_thread(lambda r=row: (
        self._client.table("bars_1min")
        .upsert(r, on_conflict="bar_time,instrument")
        .execute()
    ))
elif item_type == "bars_daily":
    await asyncio.to_thread(lambda r=row: (
        self._client.table("bars_daily")
        .upsert(r, on_conflict="bar_date,instrument,session_type")
        .execute()
    ))
```

### Backfill bars_daily from existing data

One-time script to backfill `bars_daily` from `gate_state_snapshots` (which already has RTH OHLC):

```sql
INSERT INTO bars_daily (bar_date, instrument, session_type, open, high, low, close)
SELECT snapshot_date, instrument, 'rth', rth_open, rth_high, rth_low, rth_close
FROM gate_state_snapshots
WHERE rth_open IS NOT NULL
ON CONFLICT (bar_date, instrument, session_type) DO NOTHING;
```

---

## 9. Cost Estimate

### Per-run token budget (conservative)

| Component | Tokens |
|-----------|--------|
| System prompt + strategy context | ~2,500 |
| EOD digest (input) | ~3,000 |
| Trades (up to 15 trades) | ~4,000 |
| 1-min bars (2 instruments × 390 bars) | ~32,000 |
| Daily bars (2 instruments × 20 days) | ~4,000 |
| Tool call overhead (14 calls × ~500 tokens each) | ~7,000 |
| Agent reasoning (text output) | ~6,000 |
| Forensic annotations (up to 6 trades × ~800 tokens) | ~5,000 |
| Level map output | ~2,000 |
| **Total input tokens** | **~55,000** |
| **Total output tokens** | **~10,000** |

### Cost per run (Sonnet pricing)

- Input: 55,000 × $3.00 / 1M = $0.165
- Output: 10,000 × $15.00 / 1M = $0.150
- **Total: ~$0.32/run**

### Monthly cost

- 22 trading days × $0.32 = **~$7.00/month**

Cheap enough to run daily without concern. Even if bar data doubles the input (e.g., extended hours), cost stays under $12/month.

### Comparison to other agents

| Agent | Model | Est. cost/run | Monthly |
|-------|-------|---------------|---------|
| Digest (EOD) | Sonnet | ~$0.08 | ~$1.76 |
| Digest (Morning) | Sonnet | ~$0.06 | ~$1.32 |
| **Investigation** | **Sonnet** | **~$0.32** | **~$7.00** |
| Strategist | Opus | ~$1.00 | ~$22.00 |

Investigation is the most expensive Sonnet agent because of bar data volume, but still far cheaper than the Strategist.

---

## 10. Implementation Plan

### Phase 1: Infrastructure (prerequisite)

1. Create `bars_1min` table in Supabase (migration 005)
2. Add `metadata` and `agent_run_id` columns to `trade_annotations` (migration 005)
3. Create `key_levels` table (migration 005)
4. Create `investigation_runs` table (migration 005)
5. Add `export_session_bars` to `DbLogger` + drain loop handling
6. Backfill `bars_daily` from `gate_state_snapshots`
7. Verify bar data flows: run engine for one session, check `bars_1min` has data

### Phase 2: Agent Implementation

1. Create `agents/investigation/` package (same structure as `agents/digest/`)
2. Implement tools.py with all 14 tools
3. Implement prompts.py with system prompt
4. Implement agent.py (copy DigestAgent pattern, adjust for Investigation)
5. Implement cli.py for manual runs

### Phase 3: Validation

1. Run manually on a recent trading day with known trades
2. Verify forensic annotations are structured correctly
3. Verify key levels match what a human would identify
4. Verify Fibonacci levels are computed correctly from swing detection
5. Check cost is within budget

### Phase 4: Integration

1. Add GitHub Actions workflow triggered by Digest completion
2. Wire up Strategist's `get_investigation_findings` tool to read investigation output
3. Wire up Morning Digest to surface key levels
4. Add Investigation status to dashboard (placeholder until AnalyticsPanel v2)

---

## 11. What This Agent Does NOT Do

- **Does not recommend changes.** It produces evidence. The Strategist recommends.
- **Does not run backtests.** If it identifies a hypothesis ("losses cluster near fib 61.8%"), it flags it for the Frontier Agent.
- **Does not compute indicators the engine already computes.** SM, RSI, and Leledc are already in the trade record. The Investigation Agent adds market structure context that the engine doesn't track.
- **Does not analyze intraday.** It runs once, after market close. Real-time monitoring is the Observation Agent's job (future).
- **Does not store raw bars permanently.** `bars_1min` has a 30-day retention policy. Historical analysis uses `bars_daily`.
