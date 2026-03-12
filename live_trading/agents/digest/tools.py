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
                        "- tomorrow_outlook: {vix, atr, levels, watchlist}\n"
                        "\nFor Morning:\n"
                        "- traffic_light: {color, reasons}\n"
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
        "description": "Get market regime context: today's RTH range (high-low), range percentile vs last 30 days, close vs open direction, day type (trending/choppy/mixed). Derived from gate_state_snapshots.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "instrument": {
                    "type": "string",
                    "description": "Optional: MNQ or MES. Omit for both."
                }
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
    "gate_vix_close", "gate_leledc_active", "gate_atr_value", "gate_adr_ratio",
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
    result = (
        client.table("gate_state_snapshots")
        .select("*")
        .eq("snapshot_date", date)
        .execute()
    )
    return result.data or []


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
def get_market_regime(client, *, date: str, instrument: str = None) -> list | dict:
    d = _parse_date(date)
    start = (d - timedelta(days=45)).isoformat()

    query = (
        client.table("gate_state_snapshots")
        .select("snapshot_date,instrument,rth_open,rth_high,rth_low,rth_close,adr_value,atr_value,vix_close")
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
        today_snap = next(
            (s for s in snapshots if s["snapshot_date"] == date), None
        )
        if not today_snap:
            continue

        rth_high = today_snap.get("rth_high")
        rth_low = today_snap.get("rth_low")
        rth_open = today_snap.get("rth_open")
        rth_close = today_snap.get("rth_close")

        if rth_high is None or rth_low is None:
            continue

        daily_range = round(rth_high - rth_low, 2)

        # Historical ranges for percentile
        hist_ranges = []
        for s in snapshots:
            h, l = s.get("rth_high"), s.get("rth_low")
            if h is not None and l is not None and s["snapshot_date"] != date:
                hist_ranges.append(h - l)

        range_percentile = None
        if hist_ranges:
            below = sum(1 for r in hist_ranges if r <= daily_range)
            range_percentile = round(below / len(hist_ranges) * 100, 0)

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

        regimes.append({
            "instrument": inst,
            "date": date,
            "rth_open": rth_open,
            "rth_high": rth_high,
            "rth_low": rth_low,
            "rth_close": rth_close,
            "daily_range_pts": daily_range,
            "range_percentile_30d": range_percentile,
            "direction": direction,
            "body_pct": body_pct,
            "day_type": day_type,
            "adr_value": today_snap.get("adr_value"),
            "atr_value": today_snap.get("atr_value"),
            "vix_close": today_snap.get("vix_close"),
        })

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
}
