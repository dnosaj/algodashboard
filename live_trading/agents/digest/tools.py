"""
Supabase query tools for the Digest Agent.

Each function takes a Supabase client + params and returns structured data.
Errors return {"error": "message"} so the model can reason about missing data.
"""

import functools
import logging
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

# ── Tool definitions for Anthropic API ────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_todays_trades",
        "description": "Fetch all trades for a given date with full context (entry/exit prices, SM, RSI, MFE/MAE, gate state, exit reason, bars held, qty, slippage). Trades are sorted by entry_time. Capped at 100.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_daily_strategy_stats",
        "description": "Get per-strategy daily stats: trade count, wins, losses, win rate, PnL, profit factor, exit reason breakdown (TP, SL, BE_TIME, EOD counts), plus commission analysis (gross_pnl, total_commission, commission_pct).",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_daily_portfolio_context",
        "description": "Get portfolio-level P&L for a date plus week/month context: weekly running total, monthly running total, day rank vs recent days, trading day count, best/worst day in last 30 days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_drift_status",
        "description": "Get live-vs-backtest drift for all active strategies: WR deviation, PF deviation, binomial Z-score, GREEN/YELLOW/RED status.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "get_blocked_signals",
        "description": "Get all blocked signals for a date with gate type, reason, price, SM/RSI values, and which gate fired. Capped at 100.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_rolling_performance",
        "description": "Get latest rolling 20-trade WR/PF and cumulative PnL per strategy.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "get_drawdown_status",
        "description": "Get current drawdown position per strategy: cumulative PnL, high water mark, drawdown amount, drawdown percentage.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "get_gate_state",
        "description": "Get gate state snapshot: VIX close, ADR value, ATR value, Leledc counts, and prior-day levels (H/L/VPOC/VAH/VAL) per instrument.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_correlation_status",
        "description": "Get cross-strategy daily P&L correlation: Pearson r, overlap days, both-lose days, divergent days per strategy pair.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "get_runner_pairs",
        "description": "Get TP1/runner pair analysis for partial-exit strategies on a given date: TP1 P&L, runner P&L, combined P&L, runner exit reason, bars held.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_recent_digests",
        "description": "Get previous digests for narrative continuity. Returns up to 6 digests from the last 7 days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Reference date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "save_digest",
        "description": "Save the completed digest to Supabase and write a markdown file. Call this as your final action after completing the analysis. The 'content' field MUST follow the schema below for your digest_type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "digest_type": {"type": "string", "enum": ["eod", "morning"]},
                "content": {
                    "type": "object",
                    "description": (
                        "Structured JSON content. Required keys depend on digest_type.\n"
                        "\nFor EOD:\n"
                        "- portfolio_summary: {total_pnl, week_pnl, month_pnl, rank_in_30d, day_type}\n"
                        "- strategy_breakdown: [{strategy_id, trades, wins, pnl, pf, notable}]\n"
                        "- patterns: [{pattern_type, description, severity}]\n"
                        "- risk_flags: [{flag, detail, severity}]\n"
                        "- gate_summary: {blocked_count, gates_active, near_threshold}\n"
                        "- runner_summary: {tp1_fills, runners_profitable, runners_be_time, runners_sl} (if applicable)\n"
                        "- forensic_insights: [{tool, finding, severity}] (from SL velocity, clustering, near-gate-miss, level proximity)\n"
                        "- flags_for_frontier: [{hypothesis, evidence, suggested_test, priority, sample_size, recurrence, strategy_id}]\n"
                        "- tomorrow_outlook: {vix, atr, levels, watchlist}\n"
                        "\nFor Morning:\n"
                        "- status_snapshot: {week_pnl, month_pnl, consecutive_days, drift_zscores, drawdown_pcts}\n"
                        "- yesterday_recap: {pnl, headline, key_events}\n"
                        "- gate_status: {vix, atr, adr, leledc, prior_day_levels}\n"
                        "- system_health: [{strategy_id, drift_status, rolling_wr, drawdown_pct}]\n"
                        "- risk_flags: [{flag, detail, severity}]\n"
                        "- watchlist: [str]"
                    ),
                },
                "markdown": {
                    "type": "string",
                    "description": "Human-readable markdown summary of the digest"
                }
            },
            "required": ["date", "digest_type", "content", "markdown"]
        }
    },
    # ── New tools (M1-M7) ──────────────────────────────────────────────
    {
        "name": "get_tod_performance",
        "description": "Get historical time-of-day performance by hour (ET) per strategy: trade count, win rate, PF, avg P&L. Compare today's entry hours against these baselines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "Optional: filter to a single strategy. Omit for all."
                }
            },
        }
    },
    {
        "name": "get_market_regime",
        "description": "Get market regime context: RTH OHLC, range percentile vs last 30 days, day type (trending/choppy/mixed), VWAP close, opening range (initial balance), developing VPOC (dvpoc_price, dvpoc_strength/VCR 0-1, dvpoc_stability). Use days>1 for multi-day context (e.g. 'range expanding for 5 days'). Derived from gate_state_snapshots.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "instrument": {
                    "type": "string",
                    "description": "Optional: MNQ or MES. Omit for both."
                },
                "days": {
                    "type": "integer",
                    "description": "Number of recent days to return (default 1). Use >1 for multi-day context patterns."
                }
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_sl_velocity",
        "description": "Classify SL exits by speed: rapid/normal/gradual with strategy-specific thresholds. Includes mae_velocity (mae_pts/bars_held) and bars_held distribution. Helps distinguish entry-quality problems (rapid SL) from position-management issues (gradual SL).",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "days": {
                    "type": "integer",
                    "description": "Rolling window in days (default 1). Use 5-10 for meaningful aggregation."
                }
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_entry_clustering",
        "description": "Flag simultaneous entries across strategies on the same 1-min bar. Computes combined dollar SL risk, max concurrent dollar exposure vs $650 daily limit, and whether clustered trades all won/lost. Distinguishes correlated clusters (same instrument) from coincident clusters (different instruments).",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_near_gate_miss",
        "description": "Find trades that entered just outside gate thresholds. Reports BOTH winning and losing near-misses with net counterfactual P&L. Near-miss windows: VIX within 1.0 of 19/22, ATR within 5% above 263.8, ADR within 0.05 of 0.3, Leledc count 7-8 (threshold 9). Note: a single trade may appear in multiple gates; unique_trades count deduplicates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_level_proximity",
        "description": "Distance from each trade's entry to nearest prior-day levels (H/L/VPOC/VAH/VAL). Expressed as raw points, % of TP, and ADR-normalized. Reports distance to EACH level individually. Depends on gate_state_snapshots data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
            },
            "required": ["date"]
        }
    },
    {
        "name": "get_gate_effectiveness",
        "description": "Get aggregate gate effectiveness over time: signals blocked per gate type per week, counterfactual P&L (what would have happened without the gate), losses avoided vs gains missed. Assesses whether each gate is net-positive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weeks": {
                    "type": "integer",
                    "description": "Number of recent weeks to include. Default 8."
                },
                "gate_type": {
                    "type": "string",
                    "description": "Optional: filter to specific gate type (vix_death_zone, leledc, prior_day_level, adr_directional, prior_day_atr)"
                }
            },
        }
    },
    {
        "name": "get_streak_status",
        "description": "Get current consecutive win/loss streak per strategy: how many consecutive wins or losses from the most recent trade backwards. Positive = win streak, negative = loss streak.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "get_runner_stats",
        "description": "Get aggregate runner conversion statistics for partial-exit strategies: TP1 fill rate, runner conversion rate (TP2 vs BE_TIME vs SL), average runner outcome by exit type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "Optional: filter to a single strategy (e.g. MNQ_VSCALPC, MES_V2)"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of recent days to include. Default: all time (0)."
                }
            },
        }
    },
    {
        "name": "get_dow_performance",
        "description": "Get historical day-of-week performance per strategy: trade count, win rate, PF, total P&L for each weekday (Mon-Fri). Flag historically weak days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "Optional: filter to a single strategy"
                }
            },
        }
    },
]


# ── Tool implementations ──────────────────────────────────────────────────

def _safe_query(fn):
    """Decorator: catch Supabase errors and return {"error": msg}."""
    @functools.wraps(fn)
    def wrapper(client, **kwargs):
        try:
            return fn(client, **kwargs)
        except Exception as e:
            logger.warning(f"[DigestTool] {fn.__name__} failed: {e}")
            return {"error": str(e)}
    return wrapper


# ── Columns selected for trades (strip internal IDs, timestamps) ─────────
_TRADE_COLUMNS = ",".join([
    "strategy_id", "instrument", "side",
    "entry_price", "exit_price", "entry_time", "exit_time",
    "pts", "pnl_net", "commission", "exit_reason", "bars_held", "qty", "is_partial",
    "entry_sm_value", "entry_rsi_value", "entry_bar_volume",
    "mfe_pts", "mae_pts", "signal_price",
    "gate_vix_close", "gate_leledc_active", "gate_atr_value", "gate_adr_ratio", "gate_leledc_count",
    "is_runner", "trade_group_id", "source", "trade_date", "day_of_week",
])


@_safe_query
def get_todays_trades(client, *, date: str) -> list | dict:
    result = (
        client.table("trades")
        .select(_TRADE_COLUMNS)
        .eq("trade_date", date)
        .in_("source", ["paper", "live"])
        .order("entry_time")
        .limit(100)
        .execute()
    )
    trades = result.data or []

    # Enrich with derived fields
    for t in trades:
        # Slippage (if signal_price available)
        if t.get("signal_price") and t.get("entry_price"):
            side_mult = 1 if t.get("side") == "long" else -1
            t["slippage_pts"] = round(
                side_mult * (t["entry_price"] - t["signal_price"]), 2
            )

        # Entry hour ET (for time clustering)
        entry = t.get("entry_time")
        if entry:
            try:
                dt = datetime.fromisoformat(entry)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                et = dt.astimezone(_ET)
                t["entry_hour_et"] = et.hour
                t["entry_minute_et"] = et.minute
            except Exception:
                pass

        # MFE efficiency
        mfe = t.get("mfe_pts")
        pts = t.get("pts")
        if mfe and mfe > 0 and pts:
            t["mfe_efficiency"] = round(pts / mfe, 2)

    return trades


@_safe_query
def get_daily_strategy_stats(client, *, date: str) -> list | dict:
    result = (
        client.table("daily_stats")
        .select("*")
        .eq("trade_date", date)
        .in_("source", ["paper", "live"])
        .execute()
    )
    stats = result.data or []

    # M2: Enrich with commission breakdown from trades table
    if stats:
        comm_result = (
            client.table("trades")
            .select("strategy_id,source,pnl_net,commission")
            .eq("trade_date", date)
            .in_("source", ["paper", "live"])
            .execute()
        )
        trades = comm_result.data or []

        comm_map = {}
        for t in trades:
            key = (t["strategy_id"], t.get("source", "paper"))
            if key not in comm_map:
                comm_map[key] = {"total_commission": 0.0, "gross_pnl": 0.0}
            commission = float(t.get("commission") or 0)
            pnl_net = float(t.get("pnl_net") or 0)
            comm_map[key]["total_commission"] += commission
            comm_map[key]["gross_pnl"] += pnl_net + commission

        for s in stats:
            key = (s["strategy_id"], s.get("source", "paper"))
            cm = comm_map.get(key, {})
            total_comm = round(cm.get("total_commission", 0.0), 2)
            gross_pnl = round(cm.get("gross_pnl", 0.0), 2)
            s["total_commission"] = total_comm
            s["gross_pnl"] = gross_pnl
            s["commission_pct"] = (
                round(total_comm / gross_pnl * 100, 1)
                if gross_pnl > 0 else 0.0
            )

    return stats


@_safe_query
def get_daily_portfolio_context(client, *, date: str) -> dict:
    target = date
    d = _parse_date(date)
    start_30d = (d - timedelta(days=45)).isoformat()

    result = (
        client.table("portfolio_daily")
        .select("*")
        .gte("trade_date", start_30d)
        .lte("trade_date", date)
        .in_("source", ["paper", "live"])
        .order("trade_date")
        .execute()
    )
    days = result.data or []

    today_row = next((r for r in days if r["trade_date"] == target), None)
    today_pnl = today_row["portfolio_pnl"] if today_row else 0.0

    # Week (Monday-Friday containing target date)
    week_start = (d - timedelta(days=d.weekday())).isoformat()
    week_days = [r for r in days if week_start <= r["trade_date"] <= target]
    week_pnl = sum(r.get("portfolio_pnl", 0) for r in week_days)

    # Month
    month_start = d.replace(day=1).isoformat()
    month_days = [r for r in days if month_start <= r["trade_date"] <= target]
    month_pnl = sum(r.get("portfolio_pnl", 0) for r in month_days)

    # Rank today vs last 30 days (I9: proper ranking)
    last_30 = [r.get("portfolio_pnl", 0) for r in days[-30:]]
    if last_30:
        rank = sum(1 for x in last_30 if x > today_pnl) + 1
        best_30 = max(last_30)
        worst_30 = min(last_30)
    else:
        rank = None
        best_30 = None
        worst_30 = None

    return {
        "today_pnl": today_pnl,
        "week_pnl": round(week_pnl, 2),
        "month_pnl": round(month_pnl, 2),
        "week_days_traded": len(week_days),
        "month_days_traded": len(month_days),
        "rank_in_30d": rank,
        "total_30d_days": len(last_30),
        "best_day_30d": best_30,
        "worst_day_30d": worst_30,
        "trading_days_total": len(days),
        "recent_days": [
            {"date": r["trade_date"], "pnl": r.get("portfolio_pnl", 0)}
            for r in days[-7:]
        ],
    }


@_safe_query
def get_drift_status(client) -> list | dict:
    result = (
        client.table("live_vs_backtest")
        .select("*")
        .in_("source", ["paper", "live"])
        .execute()
    )
    return result.data or []


@_safe_query
def get_blocked_signals(client, *, date: str) -> list | dict:
    result = (
        client.table("blocked_signals")
        .select("*")
        .eq("signal_date", date)
        .in_("source", ["paper", "live"])
        .order("signal_time")
        .limit(100)
        .execute()
    )
    return result.data or []


@_safe_query
def get_rolling_performance(client) -> list | dict:
    result = (
        client.table("rolling_performance")
        .select("strategy_id,source,trade_num,cumulative_pnl,rolling_20_wr,rolling_20_pf,trade_date")
        .in_("source", ["paper", "live"])
        .order("trade_num", desc=True)
        .limit(50)
    ).execute()

    seen = set()
    latest = []
    for r in (result.data or []):
        key = (r["strategy_id"], r["source"])
        if key not in seen:
            seen.add(key)
            latest.append(r)
    return latest


@_safe_query
def get_drawdown_status(client) -> list | dict:
    result = (
        client.table("drawdown_hwm")
        .select("*")
        .in_("source", ["paper", "live"])
        .order("trade_num", desc=True)
        .limit(50)
    ).execute()

    seen = set()
    latest = []
    for r in (result.data or []):
        key = (r["strategy_id"], r["source"])
        if key not in seen:
            seen.add(key)
            latest.append(r)
    return latest


@_safe_query
def get_gate_state(client, *, date: str) -> list | dict:
    # Try exact date first
    result = (
        client.table("gate_state_snapshots")
        .select("*")
        .eq("snapshot_date", date)
        .execute()
    )
    if result.data:
        return result.data

    # Fallback: most recent snapshot (today's isn't written until overnight reset)
    fallback = (
        client.table("gate_state_snapshots")
        .select("*")
        .lte("snapshot_date", date)
        .order("snapshot_date", desc=True)
        .limit(10)  # up to 10 rows (multiple instruments)
        .execute()
    )
    rows = fallback.data or []
    if rows:
        latest_date = rows[0]["snapshot_date"]
        return {
            "note": f"No snapshot for {date} — showing most recent ({latest_date})",
            "snapshot_date": latest_date,
            "data": [r for r in rows if r["snapshot_date"] == latest_date],
        }
    return []


@_safe_query
def get_correlation_status(client) -> list | dict:
    result = (
        client.table("cross_strategy_correlation")
        .select("*")
        .in_("source", ["paper", "live"])
        .execute()
    )
    return result.data or []


@_safe_query
def get_runner_pairs(client, *, date: str) -> list | dict:
    result = (
        client.table("tp1_runner_pairs")
        .select("*")
        .eq("trade_date", date)
        .in_("source", ["paper", "live"])
        .execute()
    )
    return result.data or []


@_safe_query
def get_recent_digests(client, *, date: str) -> list | dict:
    d = _parse_date(date)
    start = (d - timedelta(days=7)).isoformat()  # I7: 7-day lookback
    result = (
        client.table("digests")
        .select("digest_date,digest_type,content,markdown")
        .gte("digest_date", start)
        .lt("digest_date", date)  # exclude today
        .order("digest_date", desc=True)
        .limit(6)
        .execute()
    )
    return result.data or []


# ── M1: Time-of-day performance ──────────────────────────────────────────

@_safe_query
def get_tod_performance(client, *, strategy_id: str = None) -> list | dict:
    query = (
        client.table("tod_performance")
        .select("*")
        .in_("source", ["paper", "live"])
    )
    if strategy_id:
        query = query.eq("strategy_id", strategy_id)
    result = query.order("strategy_id").order("entry_hour_et").execute()
    return result.data or []


# ── M3: Market regime context ────────────────────────────────────────────

@_safe_query
def get_market_regime(client, *, date: str, instrument: str = None, days: int = 1) -> list | dict:
    d = _parse_date(date)
    start = (d - timedelta(days=45)).isoformat()

    query = (
        client.table("gate_state_snapshots")
        .select("snapshot_date,instrument,rth_open,rth_high,rth_low,rth_close,"
                "adr_value,atr_value,vix_close,vwap_close,opening_range_high,opening_range_low,"
                "dvpoc_price,dvpoc_strength,dvpoc_stability")
        .gte("snapshot_date", start)
        .lte("snapshot_date", date)
    )
    if instrument:
        query = query.eq("instrument", instrument)
    result = query.order("snapshot_date").execute()
    rows = result.data or []

    if not rows:
        return {"error": "No gate_state_snapshots found for this date range"}

    # Group by instrument
    by_inst: dict[str, list] = {}
    for r in rows:
        inst = r["instrument"]
        if inst not in by_inst:
            by_inst[inst] = []
        by_inst[inst].append(r)

    regimes = []
    for inst, snapshots in by_inst.items():
        # Historical ranges for percentile (date -> range, skipping nulls)
        hist_range_by_date: dict[str, float] = {}
        for s in snapshots:
            h, l = s.get("rth_high"), s.get("rth_low")
            if h is not None and l is not None:
                hist_range_by_date[s["snapshot_date"]] = h - l

        # Determine which days to return (last N days of available data)
        target_dates = sorted(set(s["snapshot_date"] for s in snapshots))
        if days > 1:
            target_dates = target_dates[-days:]
        else:
            target_dates = [date] if date in target_dates else target_dates[-1:]

        for target_date in target_dates:
            snap = next((s for s in snapshots if s["snapshot_date"] == target_date), None)
            if not snap:
                continue

            rth_high = snap.get("rth_high")
            rth_low = snap.get("rth_low")
            rth_open = snap.get("rth_open")
            rth_close = snap.get("rth_close")

            if rth_high is None or rth_low is None:
                continue

            daily_range = round(rth_high - rth_low, 2)

            # Percentile vs history (excluding this day)
            prior_ranges = [r for d, r in hist_range_by_date.items()
                            if d != target_date]
            range_percentile = None
            if prior_ranges:
                below = sum(1 for r in prior_ranges if r <= daily_range)
                range_percentile = round(below / len(prior_ranges) * 100, 0)

            # Trend classification
            direction = None
            body_pct = None
            if rth_open and rth_close and daily_range > 0:
                body = rth_close - rth_open
                direction = "up" if body > 0 else "down" if body < 0 else "flat"
                body_pct = round(abs(body) / daily_range * 100, 0)

            day_type = "unknown"
            if body_pct is not None:
                if body_pct >= 60:
                    day_type = "trending"
                elif body_pct <= 30:
                    day_type = "choppy"
                else:
                    day_type = "mixed"

            regime = {
                "instrument": inst,
                "date": target_date,
                "rth_open": rth_open,
                "rth_high": rth_high,
                "rth_low": rth_low,
                "rth_close": rth_close,
                "daily_range_pts": daily_range,
                "range_percentile_30d": range_percentile,
                "direction": direction,
                "body_pct": body_pct,
                "day_type": day_type,
                "adr_value": snap.get("adr_value"),
                "atr_value": snap.get("atr_value"),
                "vix_close": snap.get("vix_close"),
                "vwap_close": snap.get("vwap_close"),
                "opening_range_high": snap.get("opening_range_high"),
                "opening_range_low": snap.get("opening_range_low"),
            }
            regimes.append(regime)

    return regimes


# ── M4: Gate effectiveness aggregate ─────────────────────────────────────

@_safe_query
def get_gate_effectiveness(client, *, weeks: int = 8, gate_type: str = None) -> dict:
    query = client.table("gate_effectiveness").select("*")
    if gate_type:
        query = query.eq("gate_type", gate_type)
    result = query.execute()
    rows = result.data or []

    if not rows:
        return {"weekly_detail": [], "totals": []}

    # Filter to recent N weeks
    if weeks:
        d_cutoff = (date.today() - timedelta(weeks=weeks)).isoformat()
        rows = [r for r in rows if r.get("week_start", "") >= d_cutoff]

    rows.sort(key=lambda r: (r.get("gate_type", ""), r.get("week_start", "")))

    # Compute per-gate-type totals
    totals: dict[str, dict] = {}
    for r in rows:
        gt = r.get("gate_type", "unknown")
        if gt not in totals:
            totals[gt] = {
                "gate_type": gt,
                "total_blocked": 0,
                "total_cf_computed": 0,
                "total_cf_net_pnl": 0.0,
                "total_losses_avoided": 0.0,
                "total_gains_missed": 0.0,
                "weeks_active": 0,
            }
        t = totals[gt]
        t["total_blocked"] += r.get("signals_blocked", 0)
        t["total_cf_computed"] += r.get("counterfactuals_computed", 0)
        t["total_cf_net_pnl"] += float(r.get("counterfactual_net_pnl") or 0)
        t["total_losses_avoided"] += float(r.get("losses_avoided") or 0)
        t["total_gains_missed"] += float(r.get("gains_missed") or 0)
        t["weeks_active"] += 1

    for t in totals.values():
        t["total_cf_net_pnl"] = round(t["total_cf_net_pnl"], 2)
        t["total_losses_avoided"] = round(t["total_losses_avoided"], 2)
        t["total_gains_missed"] = round(t["total_gains_missed"], 2)
        # Net positive means gate saved money (counterfactual P&L was negative)
        t["net_verdict"] = "POSITIVE" if t["total_cf_net_pnl"] <= 0 else "NEGATIVE"

    return {
        "weekly_detail": rows,
        "totals": list(totals.values()),
    }


# ── M5: Consecutive win/loss streaks ─────────────────────────────────────

@_safe_query
def get_streak_status(client) -> list | dict:
    result = (
        client.table("trades")
        .select("strategy_id,source,pnl_net,exit_time,trade_date,exit_reason")
        .in_("source", ["paper", "live"])
        .not_.is_("exit_time", "null")
        .order("exit_time", desc=True)
        .limit(200)
        .execute()
    )
    trades = result.data or []

    # Group by (strategy_id, source), most recent first
    groups: dict[tuple, list] = {}
    for t in trades:
        key = (t["strategy_id"], t.get("source", "paper"))
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    streaks = []
    for (strategy_id, source), trades_list in groups.items():
        if not trades_list:
            continue

        first_outcome = "win" if float(trades_list[0].get("pnl_net") or 0) > 0 else "loss"
        streak_count = 0
        for t in trades_list:
            outcome = "win" if float(t.get("pnl_net") or 0) > 0 else "loss"
            if outcome == first_outcome:
                streak_count += 1
            else:
                break

        streaks.append({
            "strategy_id": strategy_id,
            "source": source,
            "streak_type": first_outcome,
            "streak_length": streak_count,
            "streak_signed": streak_count if first_outcome == "win" else -streak_count,
            "last_trade_date": trades_list[0].get("trade_date"),
            "last_exit_reason": trades_list[0].get("exit_reason"),
        })

    return streaks


# ── M6: Runner conversion aggregate stats ────────────────────────────────

@_safe_query
def get_runner_stats(client, *, strategy_id: str = None, days: int = 0) -> dict:
    query = (
        client.table("tp1_runner_pairs")
        .select("*")
        .in_("source", ["paper", "live"])
    )
    if strategy_id:
        query = query.eq("strategy_id", strategy_id)
    if days > 0:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        query = query.gte("trade_date", cutoff)

    result = query.execute()
    pairs = result.data or []

    if not pairs:
        return {"strategies": [], "note": "No partial-exit pairs found"}

    by_strat: dict[str, dict] = {}
    for p in pairs:
        sid = p.get("strategy_id", "unknown")
        if sid not in by_strat:
            by_strat[sid] = {
                "total_entries": 0, "tp1_filled": 0,
                "runner_tp2": 0, "runner_sl": 0, "runner_be_time": 0,
                "runner_eod": 0, "runner_other": 0,
                "runner_pnl_sum": 0.0, "runner_pnl_by_exit": {},
                "combined_pnl_sum": 0.0,
                "runner_bars_held_sum": 0, "runner_count": 0,
            }
        s = by_strat[sid]
        s["total_entries"] += 1

        tp1_reason = p.get("tp1_exit_reason", "")
        if tp1_reason in ("TP", "TP1"):
            s["tp1_filled"] += 1

        runner_reason = p.get("runner_exit_reason")
        runner_pnl = float(p.get("runner_pnl") or 0)
        runner_bars = p.get("runner_bars_held") or 0

        if runner_reason:
            s["runner_count"] += 1
            s["runner_pnl_sum"] += runner_pnl
            s["runner_bars_held_sum"] += runner_bars

            if runner_reason in ("TP", "TP2"):
                s["runner_tp2"] += 1
            elif runner_reason == "SL":
                s["runner_sl"] += 1
            elif runner_reason == "BE_TIME":
                s["runner_be_time"] += 1
            elif runner_reason == "EOD":
                s["runner_eod"] += 1
            else:
                s["runner_other"] += 1

            if runner_reason not in s["runner_pnl_by_exit"]:
                s["runner_pnl_by_exit"][runner_reason] = {"count": 0, "pnl_sum": 0.0}
            s["runner_pnl_by_exit"][runner_reason]["count"] += 1
            s["runner_pnl_by_exit"][runner_reason]["pnl_sum"] += runner_pnl

        s["combined_pnl_sum"] += float(p.get("combined_pnl") or 0)

    strategies = []
    for sid, s in by_strat.items():
        tp1_rate = round(s["tp1_filled"] / s["total_entries"] * 100, 1) if s["total_entries"] else 0
        runner_conv = round(s["runner_tp2"] / s["runner_count"] * 100, 1) if s["runner_count"] else 0
        avg_runner_pnl = round(s["runner_pnl_sum"] / s["runner_count"], 2) if s["runner_count"] else 0
        avg_runner_bars = round(s["runner_bars_held_sum"] / s["runner_count"], 1) if s["runner_count"] else 0

        exit_breakdown = {}
        for reason, data in s["runner_pnl_by_exit"].items():
            exit_breakdown[reason] = {
                "count": data["count"],
                "total_pnl": round(data["pnl_sum"], 2),
                "avg_pnl": round(data["pnl_sum"] / data["count"], 2) if data["count"] else 0,
            }

        strategies.append({
            "strategy_id": sid,
            "total_entries": s["total_entries"],
            "tp1_fill_rate_pct": tp1_rate,
            "runner_count": s["runner_count"],
            "runner_conversion_rate_tp2_pct": runner_conv,
            "runner_exit_breakdown": exit_breakdown,
            "avg_runner_pnl": avg_runner_pnl,
            "avg_runner_bars_held": avg_runner_bars,
            "total_combined_pnl": round(s["combined_pnl_sum"], 2),
        })

    return {"strategies": strategies}


# ── M7: Day-of-week performance ──────────────────────────────────────────

_DOW_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday"}


@_safe_query
def get_dow_performance(client, *, strategy_id: str = None) -> list | dict:
    query = (
        client.table("trades")
        .select("strategy_id,source,day_of_week,pnl_net")
        .in_("source", ["paper", "live"])
        .not_.is_("exit_time", "null")
        .eq("is_partial", False)
    )
    if strategy_id:
        query = query.eq("strategy_id", strategy_id)

    result = query.execute()
    trades = result.data or []

    if not trades:
        return []

    buckets: dict[tuple, dict] = {}
    for t in trades:
        dow = t.get("day_of_week")
        if dow is None or dow > 5:
            continue
        key = (t["strategy_id"], t.get("source", "paper"), dow)
        if key not in buckets:
            buckets[key] = {"wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0, "total_pnl": 0.0}
        b = buckets[key]
        pnl = float(t.get("pnl_net") or 0)
        b["total_pnl"] += pnl
        if pnl > 0:
            b["wins"] += 1
            b["gross_profit"] += pnl
        else:
            b["losses"] += 1
            b["gross_loss"] += pnl

    rows = []
    for (sid, source, dow), b in sorted(buckets.items()):
        total = b["wins"] + b["losses"]
        wr = round(b["wins"] / total * 100, 1) if total else 0
        pf = round(b["gross_profit"] / abs(b["gross_loss"]), 3) if b["gross_loss"] < 0 else None
        rows.append({
            "strategy_id": sid,
            "source": source,
            "day_of_week": dow,
            "day_name": _DOW_NAMES.get(dow, f"Day{dow}"),
            "trade_count": total,
            "wins": b["wins"],
            "losses": b["losses"],
            "win_rate": wr,
            "profit_factor": pf,
            "total_pnl": round(b["total_pnl"], 2),
        })

    return rows


# ── F1: SL velocity analysis ──────────────────────────────────────────────

# Strategy-specific SL velocity thresholds (bars_held buckets)
_SL_VELOCITY_THRESHOLDS = {
    "MNQ_VSCALPB": {"rapid": 3, "normal": 10},    # rapid <3, normal 3-10, gradual 10+
    "MNQ_V15":     {"rapid": 5, "normal": 15},     # rapid <5, normal 5-15, gradual 15+
    "MNQ_VSCALPC": {"rapid": 3, "normal": 15},     # rapid <3 = TP1 never filled (2x loss)
    "MES_V2":      {"rapid": 60, "normal": 100},   # fast <60, normal 60-100, extended 100+
}

# SL distance per strategy (for dollar-risk calculations)
_SL_DISTANCE = {
    "MNQ_V15": {"pts": 40, "dollar_per_pt": 2.0},
    "MNQ_VSCALPB": {"pts": 10, "dollar_per_pt": 2.0},
    "MNQ_VSCALPC": {"pts": 40, "dollar_per_pt": 2.0},
    "MES_V2": {"pts": 35, "dollar_per_pt": 5.0},
}


@_safe_query
def get_sl_velocity(client, *, date: str, days: int = 1) -> list | dict:
    d = _parse_date(date)
    start_date = (d - timedelta(days=max(days - 1, 0))).isoformat()

    result = (
        client.table("trades")
        .select("strategy_id,instrument,bars_held,mae_pts,pnl_net,exit_reason,is_runner,trade_date,entry_sm_value,entry_rsi_value")
        .eq("exit_reason", "SL")
        .gte("trade_date", start_date)
        .lte("trade_date", date)
        .in_("source", ["paper", "live"])
        .order("entry_time")
        .execute()
    )
    sl_trades = result.data or []

    if not sl_trades:
        return []

    by_strat: dict[str, list] = {}
    for t in sl_trades:
        sid = t.get("strategy_id", "unknown")
        if sid not in by_strat:
            by_strat[sid] = []
        by_strat[sid].append(t)

    analysis = []
    for sid, trades in by_strat.items():
        thresholds = _SL_VELOCITY_THRESHOLDS.get(sid, {"rapid": 5, "normal": 15})
        buckets = {"rapid": [], "normal": [], "gradual": []}

        for t in trades:
            bh = t.get("bars_held") or 0
            mae = t.get("mae_pts") or 0
            mae_vel = round(mae / max(bh, 1), 2)
            pnl_val = float(t.get("pnl_net") or 0)
            is_be_exit = t.get("is_runner") and bh > 0 and abs(pnl_val) < 5.0  # SL at ~breakeven after TP1

            entry = {
                "bars_held": bh,
                "mae_pts": mae,
                "mae_velocity": mae_vel,
                "pnl_net": t.get("pnl_net"),
                "trade_date": t.get("trade_date"),
                "is_breakeven_exit": is_be_exit,
                "entry_sm_value": t.get("entry_sm_value"),
                "entry_rsi_value": t.get("entry_rsi_value"),
            }

            if is_be_exit:
                # SL at breakeven after TP1 is a success, not a velocity issue
                buckets["gradual"].append(entry)
            elif bh < thresholds["rapid"]:
                buckets["rapid"].append(entry)
            elif bh < thresholds["normal"]:
                buckets["normal"].append(entry)
            else:
                buckets["gradual"].append(entry)

        bars_held_values = [t.get("bars_held", 0) for t in trades]
        analysis.append({
            "strategy_id": sid,
            "total_sl_exits": len(trades),
            "date_range": f"{start_date} to {date}" if days > 1 else date,
            "rapid_count": len(buckets["rapid"]),
            "normal_count": len(buckets["normal"]),
            "gradual_count": len(buckets["gradual"]),
            "rapid_pct": round(len(buckets["rapid"]) / len(trades) * 100, 1),
            "thresholds": thresholds,
            "bars_held_min": min(bars_held_values),
            "bars_held_max": max(bars_held_values),
            "bars_held_median": sorted(bars_held_values)[len(bars_held_values) // 2],
            "rapid_trades": buckets["rapid"][:5],  # Cap detail to avoid bloat
            "be_exits": [e for e in buckets["gradual"] if e.get("is_breakeven_exit")][:3],
        })

    return analysis


# ── F2: Entry clustering ─────────────────────────────────────────────────

@_safe_query
def get_entry_clustering(client, *, date: str) -> list | dict:
    result = (
        client.table("trades")
        .select("strategy_id,instrument,side,entry_time,exit_time,entry_price,pnl_net,qty,trade_date")
        .eq("trade_date", date)
        .in_("source", ["paper", "live"])
        .order("entry_time")
        .execute()
    )
    trades = result.data or []

    if len(trades) < 2:
        return {"clusters": [], "max_concurrent_dollar_risk": 0}

    # Group by 1-minute bar (truncate to minute)
    by_minute: dict[str, list] = {}
    for t in trades:
        et = t.get("entry_time", "")[:16]  # "2026-03-13T10:05" — minute resolution
        if et not in by_minute:
            by_minute[et] = []
        by_minute[et].append(t)

    clusters = []
    max_dollar_risk = 0.0

    for minute, group in by_minute.items():
        if len(group) < 2:
            continue

        # Compute dollar SL risk for cluster
        combined_sl_risk = 0.0
        instruments = set()
        outcomes = []
        for t in group:
            sid = t.get("strategy_id", "")
            sl_info = _SL_DISTANCE.get(sid, {"pts": 40, "dollar_per_pt": 2.0})
            qty = t.get("qty", 1)
            risk = sl_info["pts"] * sl_info["dollar_per_pt"] * qty
            combined_sl_risk += risk
            instruments.add(t.get("instrument", ""))
            pnl = float(t.get("pnl_net") or 0)
            outcomes.append("win" if pnl > 0 else "loss")

        # Classify cluster type
        cluster_type = "correlated" if len(instruments) == 1 else "coincident"

        # Correlation realized
        all_same = len(set(outcomes)) == 1
        correlation = f"all_{outcomes[0]}" if all_same else "mixed"

        daily_limit = 650  # SafetyConfig.max_daily_loss — update if config changes
        risk_pct_of_limit = round(combined_sl_risk / daily_limit * 100, 1)

        clusters.append({
            "entry_minute": minute,
            "trades": len(group),
            "strategies": [t.get("strategy_id") for t in group],
            "instruments": list(instruments),
            "cluster_type": cluster_type,
            "combined_sl_risk": round(combined_sl_risk, 2),
            "risk_pct_of_daily_limit": risk_pct_of_limit,
            "correlation_realized": correlation,
            "total_pnl": round(sum(float(t.get("pnl_net") or 0) for t in group), 2),
        })

        max_dollar_risk = max(max_dollar_risk, combined_sl_risk)

    return {
        "clusters": clusters,
        "max_concurrent_dollar_risk": round(max_dollar_risk, 2),
        "daily_limit": 650,
        "risk_flag": max_dollar_risk > 325,  # >50% of daily limit
    }


# ── F3: Near-gate miss analysis ──────────────────────────────────────────

@_safe_query
def get_near_gate_miss(client, *, date: str) -> list | dict:
    result = (
        client.table("trades")
        .select(_TRADE_COLUMNS)
        .eq("trade_date", date)
        .in_("source", ["paper", "live"])
        .order("entry_time")
        .execute()
    )
    trades = result.data or []

    if not trades:
        return []

    near_misses = []

    for t in trades:
        pnl = float(t.get("pnl_net") or 0)
        sid = t.get("strategy_id", "")
        outcome = "win" if pnl > 0 else "loss"

        # VIX near-miss: within 1.0 of death zone boundaries (19-22)
        vix = t.get("gate_vix_close")
        if vix is not None and sid in ("MNQ_V15", "MNQ_VSCALPC"):
            if 18.0 <= vix < 19.0 or 22.0 < vix <= 23.0:
                near_misses.append({
                    "gate": "vix_death_zone",
                    "strategy_id": sid,
                    "gate_value": vix,
                    "threshold": "19-22",
                    "distance": round(min(abs(vix - 19), abs(vix - 22)), 2),
                    "outcome": outcome,
                    "pnl_net": pnl,
                    "counterfactual": f"Would have been blocked, saving ${abs(pnl):.2f}" if pnl < 0 else f"Would have blocked a ${pnl:.2f} winner",
                })

        # ATR near-miss: within 5% above threshold (263.8)
        atr = t.get("gate_atr_value")
        if atr is not None and sid == "MNQ_VSCALPC":
            atr_threshold = 263.8
            if atr_threshold <= atr < atr_threshold * 1.05:
                near_misses.append({
                    "gate": "prior_day_atr",
                    "strategy_id": sid,
                    "gate_value": round(atr, 2),
                    "threshold": atr_threshold,
                    "distance": round(atr - atr_threshold, 2),
                    "distance_pct": round((atr - atr_threshold) / atr_threshold * 100, 1),
                    "outcome": outcome,
                    "pnl_net": pnl,
                    "counterfactual": f"Would have been blocked, saving ${abs(pnl):.2f}" if pnl < 0 else f"Would have blocked a ${pnl:.2f} winner",
                })

        # ADR near-miss: within 0.05 of 0.3 threshold (direction-aware)
        # Gate blocks longs when ratio >= 0.3, shorts when ratio <= -0.3
        adr = t.get("gate_adr_ratio")
        side = t.get("side", "")
        if adr is not None and t.get("instrument") == "MNQ":
            adr_threshold = 0.3
            is_near = False
            if side == "long" and 0.25 <= adr < 0.30:
                is_near = True
            elif side == "short" and -0.30 < adr <= -0.25:
                is_near = True
            if is_near:
                near_misses.append({
                    "gate": "adr_directional",
                    "strategy_id": sid,
                    "gate_value": round(adr, 3),
                    "threshold": adr_threshold,
                    "side": side,
                    "distance": round(adr_threshold - abs(adr), 3),
                    "outcome": outcome,
                    "pnl_net": pnl,
                    "counterfactual": f"Would have been blocked, saving ${abs(pnl):.2f}" if pnl < 0 else f"Would have blocked a ${pnl:.2f} winner",
                })

        # Leledc near-miss: count at 7 or 8 (threshold 9)
        leledc_count = t.get("gate_leledc_count")
        if leledc_count is not None and leledc_count in (7, 8) and t.get("instrument") == "MNQ":
            near_misses.append({
                "gate": "leledc_exhaustion",
                "strategy_id": sid,
                "gate_value": leledc_count,
                "threshold": 9,
                "distance": 9 - leledc_count,
                "outcome": outcome,
                "pnl_net": pnl,
                "counterfactual": f"Would have been blocked at threshold={leledc_count}, saving ${abs(pnl):.2f}" if pnl < 0 else f"Would have blocked a ${pnl:.2f} winner at threshold={leledc_count}",
            })

    # Summary stats (deduplicate P&L by trade to avoid multi-gate inflation)
    if near_misses:
        # Per-gate counts
        loser_count = sum(1 for nm in near_misses if nm["outcome"] == "loss")
        winner_count = sum(1 for nm in near_misses if nm["outcome"] == "win")

        # Deduplicate by trade for P&L summary (a trade may trigger multiple gates)
        seen_trades: dict[str, float] = {}
        for nm in near_misses:
            key = f"{nm['strategy_id']}_{nm.get('pnl_net', 0)}"
            if key not in seen_trades:
                seen_trades[key] = nm["pnl_net"]
        deduped_pnl = sum(seen_trades.values())

        return {
            "near_misses": near_misses,
            "total_near_miss_entries": len(near_misses),
            "unique_trades": len(seen_trades),
            "losers": loser_count,
            "winners": winner_count,
            "net_counterfactual_pnl": round(deduped_pnl, 2),
            "net_verdict": "gates should be TIGHTER" if deduped_pnl < -10 else "gates working well" if deduped_pnl > 10 else "neutral",
        }

    return []


# ── F4: Level proximity analysis ─────────────────────────────────────────

@_safe_query
def get_level_proximity(client, *, date: str) -> list | dict:
    # Get trades for the date
    trade_result = (
        client.table("trades")
        .select("strategy_id,instrument,side,entry_price,pnl_net,exit_reason,trade_date")
        .eq("trade_date", date)
        .in_("source", ["paper", "live"])
        .order("entry_time")
        .execute()
    )
    trades = trade_result.data or []

    if not trades:
        return []

    # Get prior-day levels from gate_state_snapshots (use previous trading day's snapshot)
    d = _parse_date(date)
    lookback_start = (d - timedelta(days=5)).isoformat()
    snap_result = (
        client.table("gate_state_snapshots")
        .select("snapshot_date,instrument,prior_day_high,prior_day_low,prior_day_vpoc,prior_day_vah,prior_day_val,adr_value")
        .gte("snapshot_date", lookback_start)
        .lte("snapshot_date", date)
        .order("snapshot_date", desc=True)
        .execute()
    )
    snapshots = snap_result.data or []

    if not snapshots:
        return {"error": "No gate_state_snapshots data — levels unavailable"}

    # Get most recent snapshot per instrument (levels active on trade date)
    levels_by_inst: dict[str, dict] = {}
    for s in snapshots:
        inst = s["instrument"]
        if inst not in levels_by_inst:
            levels_by_inst[inst] = s

    # TP distances per strategy for % of TP calculation
    tp_by_strategy = {
        "MNQ_V15": 7, "MNQ_VSCALPB": 3, "MNQ_VSCALPC": 25,
        "MES_V2": 20,
    }

    results = []
    for t in trades:
        inst = t.get("instrument", "")
        snap = levels_by_inst.get(inst)
        if not snap:
            continue

        entry = t.get("entry_price", 0)
        sid = t.get("strategy_id", "")
        tp_pts = tp_by_strategy.get(sid, 10)
        adr = snap.get("adr_value")  # None if unavailable

        # Prior-day levels + VWAP + opening range
        level_names = [
            "prior_day_high", "prior_day_low", "prior_day_vpoc",
            "prior_day_vah", "prior_day_val",
            "vwap_close", "opening_range_high", "opening_range_low",
        ]
        distances = []
        for lname in level_names:
            level_val = snap.get(lname)
            if level_val is None:
                continue
            dist = round(entry - level_val, 2)
            abs_dist = abs(dist)
            # Clean display name
            display = lname.replace("prior_day_", "").replace("opening_range_", "or_")
            distances.append({
                "level": display,
                "level_price": level_val,
                "distance_pts": dist,
                "abs_distance_pts": abs_dist,
                "pct_of_tp": round(abs_dist / tp_pts * 100, 1) if tp_pts > 0 else None,
                "adr_normalized": round(abs_dist / adr, 3) if adr and adr > 0 else None,
            })

        # Sort by absolute distance
        distances.sort(key=lambda x: x["abs_distance_pts"])

        results.append({
            "strategy_id": sid,
            "instrument": inst,
            "entry_price": entry,
            "side": t.get("side"),
            "pnl_net": t.get("pnl_net"),
            "nearest_level": distances[0]["level"] if distances else None,
            "nearest_distance_pts": distances[0]["abs_distance_pts"] if distances else None,
            "nearest_pct_of_tp": distances[0]["pct_of_tp"] if distances else None,
            "all_levels": distances,
        })

    return results


# ── save_digest (not in TOOL_DISPATCH — handled by agent for metadata) ───

def save_digest(client, *, date: str, digest_type: str, content: dict,
                markdown: str, model: str, tokens_in: int, tokens_out: int,
                cost_usd: float, duration_sec: float, tool_calls: int) -> dict:
    """Save digest to Supabase and write markdown file."""
    from pathlib import Path

    row = {
        "digest_date": date,
        "digest_type": digest_type,
        "content": content,
        "markdown": markdown,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost_usd, 4),
        "duration_sec": round(duration_sec, 2),
        "tool_calls": tool_calls,
        "agent_version": "1.0",
    }

    # C1: honest return on DB failure
    db_saved = False
    db_error = None
    digest_id = None

    try:
        result = (
            client.table("digests")
            .upsert(row, on_conflict="digest_date,digest_type")
            .execute()
        )
        digest_id = result.data[0]["id"] if result.data else None
        db_saved = True
    except Exception as e:
        logger.error(f"[DigestTool] Failed to save digest to Supabase: {e}")
        db_error = str(e)

    # Write markdown file (always attempt — fallback even if DB fails)
    md_path_str = None
    try:
        digest_dir = Path(__file__).parent.parent.parent / "logs" / "digests"
        digest_dir.mkdir(parents=True, exist_ok=True)
        md_path = digest_dir / f"{date}_{digest_type}.md"
        md_path.write_text(markdown)
        md_path_str = str(md_path)
        logger.info(f"[DigestTool] Markdown saved to {md_path}")
    except Exception as e:
        logger.warning(f"[DigestTool] Failed to write markdown: {e}")

    if db_saved:
        return {"saved": True, "id": digest_id, "markdown_path": md_path_str}
    else:
        return {"saved": False, "error": db_error, "markdown_path": md_path_str}


# ── Helpers ───────────────────────────────────────────────────────────────

def _parse_date(s: str) -> date:
    """Parse YYYY-MM-DD string to date. Uses stdlib fromisoformat."""
    return date.fromisoformat(s)


# ── Tool dispatch ─────────────────────────────────────────────────────────

TOOL_DISPATCH = {
    "get_todays_trades": get_todays_trades,
    "get_daily_strategy_stats": get_daily_strategy_stats,
    "get_daily_portfolio_context": get_daily_portfolio_context,
    "get_drift_status": get_drift_status,
    "get_blocked_signals": get_blocked_signals,
    "get_rolling_performance": get_rolling_performance,
    "get_drawdown_status": get_drawdown_status,
    "get_gate_state": get_gate_state,
    "get_correlation_status": get_correlation_status,
    "get_runner_pairs": get_runner_pairs,
    "get_recent_digests": get_recent_digests,
    # save_digest handled separately by the agent (needs metadata injection)
    # ── New tools (M1-M7) ──
    "get_tod_performance": get_tod_performance,
    "get_market_regime": get_market_regime,
    "get_gate_effectiveness": get_gate_effectiveness,
    "get_streak_status": get_streak_status,
    "get_runner_stats": get_runner_stats,
    "get_dow_performance": get_dow_performance,
    # ── Forensic tools (Phase 4) ──
    "get_sl_velocity": get_sl_velocity,
    "get_entry_clustering": get_entry_clustering,
    "get_near_gate_miss": get_near_gate_miss,
    "get_level_proximity": get_level_proximity,
}
