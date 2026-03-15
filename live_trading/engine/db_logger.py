"""
Database logger — fire-and-forget async EventBus subscriber.

Subscribes to trade_closed, trade_corrected, signal_blocked, and structure_bar events.
Enqueues rows to an asyncio.Queue; a background task drains them to Supabase.

Zero latency impact on bar processing. If Supabase is unreachable, logs a
warning and continues — session JSON + CSV are preserved as backup.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Gate type extraction from reason strings
_GATE_PREFIXES = {
    "VIX death zone": "vix_death_zone",
    "Leledc exhaustion": "leledc",
    "Prior-day level": "prior_day_level",
    "Near prior-day level": "prior_day_level",
    "Prior-day ATR": "prior_day_atr",
    "ADR directional": "adr_directional",
    "Strategy paused": "strategy_paused",
    "Daily P&L limit": "daily_pnl_limit",
    "Global daily loss": "global_daily_loss",
    "Consecutive losses": "consecutive_losses",
    "Max position": "max_position",
    "Engine halted": "engine_halted",
}


def _extract_gate_type(reason: str) -> str:
    """Extract normalized gate_type from a safety check reason string."""
    for prefix, gate_type in _GATE_PREFIXES.items():
        if reason.startswith(prefix):
            return gate_type
    return "unknown"


# Commission per side by instrument (for deducting from gross pnl_dollar)
_COMMISSION_PER_SIDE = {
    "MNQ": 0.52,
    "MES": 1.25,
}


def _compute_commission(instrument: str, qty: int) -> float:
    """Compute roundtrip commission for a trade."""
    per_side = _COMMISSION_PER_SIDE.get(instrument, 0.52)
    return per_side * 2 * qty


def _to_et_date(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to ET date string (YYYY-MM-DD)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_ET).date().isoformat()


class DbLogger:
    """Async database logger for trading events.

    Usage:
        db_logger = DbLogger(paper_mode=True)
        await db_logger.start()
        event_bus.subscribe("trade_closed", db_logger.on_trade_closed)
        event_bus.subscribe("trade_corrected", db_logger.on_trade_corrected)
        event_bus.subscribe("signal_blocked", db_logger.on_signal_blocked)
        # ... at shutdown:
        await db_logger.stop()
    """

    def __init__(self, paper_mode: bool = True):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._source = "paper" if paper_mode else "live"
        self._client = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._config_id_map: dict[str, str] = {}  # strategy_id -> config UUID

    async def start(self) -> bool:
        """Initialize client and start background drain loop.

        Returns True if Supabase is connected, False if disabled.
        """
        from .db import get_client
        self._client = get_client()

        if self._client is None:
            logger.info("[DbLogger] No database connection — logging disabled")
            return False

        self._running = True

        # Load strategy_id -> config_id mapping for FK population
        try:
            configs = self._client.table('strategy_configs').select('id,strategy_id,active').execute()
            for c in (configs.data or []):
                sid = c['strategy_id']
                if sid not in self._config_id_map or c.get('active'):
                    self._config_id_map[sid] = c['id']
            if self._config_id_map:
                logger.info(f"[DbLogger] Loaded config_id map for {len(self._config_id_map)} strategies")
        except Exception as e:
            logger.warning(f"[DbLogger] Could not load config_id map: {e}")
        self._task = asyncio.create_task(self._drain_loop(), name="db_logger")
        logger.info("[DbLogger] Started — logging trades to Supabase")
        return True

    async def stop(self) -> None:
        """Stop the background drain loop, flush remaining items."""
        self._running = False
        if self._task:
            # Give it a moment to drain
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    # ------------------------------------------------------------------
    # Synchronous event handlers (called by EventBus)
    # ------------------------------------------------------------------

    def on_trade_closed(self, trade) -> None:
        """Handle trade_closed event. Enqueues for async insert."""
        if not self._running:
            return
        try:
            row = self._trade_to_row(trade)
            self._queue.put_nowait(("trade", row))
        except asyncio.QueueFull:
            logger.warning("[DbLogger] Queue full, dropping trade record")
        except Exception as e:
            logger.warning(f"[DbLogger] Error building trade row: {e}")

    def on_trade_corrected(self, trade) -> None:
        """Handle trade_corrected event. Enqueues for async update."""
        if not self._running:
            return
        try:
            row = self._correction_to_row(trade)
            self._queue.put_nowait(("trade_correction", row))
        except asyncio.QueueFull:
            pass  # Corrections are best-effort
        except Exception as e:
            logger.warning(f"[DbLogger] Error building correction row: {e}")

    def on_signal_blocked(self, payload: dict) -> None:
        """Handle signal_blocked event. Enqueues for async insert."""
        if not self._running:
            return
        try:
            row = self._blocked_to_row(payload)
            if row is None:
                return
            self._queue.put_nowait(("blocked_signal", row))
        except asyncio.QueueFull:
            logger.warning("[DbLogger] Queue full, dropping blocked signal")
        except Exception as e:
            logger.warning(f"[DbLogger] Error building blocked signal row: {e}")

    def on_structure_bar(self, payload: dict) -> None:
        """Handle structure_bar event. Enqueues per-bar observation for async insert."""
        if not self._running:
            return
        try:
            row = self._structure_bar_to_row(payload)
            if row is None:
                return
            self._queue.put_nowait(("structure_bar", row))
        except asyncio.QueueFull:
            pass  # High-volume observation data — silently drop on backpressure
        except Exception as e:
            logger.warning(f"[DbLogger] Error building structure_bar row: {e}")

    def refresh_views(self) -> None:
        """Trigger materialized view refresh. Called during daily reset."""
        if not self._running:
            return
        try:
            self._queue.put_nowait(("refresh_views", {}))
        except asyncio.QueueFull:
            logger.warning("[DbLogger] Queue full, skipping view refresh")

    def snapshot_gate_state(self, safety_manager, date_str: str) -> None:
        """Snapshot daily gate state. Called during daily reset."""
        if not self._running or safety_manager is None:
            return
        try:
            rows = self._gate_state_rows(safety_manager, date_str)
            for row in rows:
                self._queue.put_nowait(("gate_snapshot", row))
        except asyncio.QueueFull:
            logger.warning("[DbLogger] Queue full, dropping gate snapshot")
        except Exception as e:
            logger.warning(f"[DbLogger] Error building gate snapshot: {e}")

    # ------------------------------------------------------------------
    # Row builders
    # ------------------------------------------------------------------

    def _trade_to_row(self, trade) -> dict:
        """Convert TradeRecord to Supabase trades row."""
        entry_time = trade.entry_time
        if entry_time and entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        exit_time = trade.exit_time
        if exit_time and exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)

        commission = _compute_commission(trade.instrument, trade.qty)
        pnl_net = trade.pnl_dollar - commission

        row = {
            "strategy_id": trade.strategy_id,
            "instrument": trade.instrument,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": entry_time.isoformat() if entry_time else None,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "pts": trade.pts,
            "pnl_net": round(pnl_net, 2),
            "commission": round(commission, 2),
            "exit_reason": trade.exit_reason,
            "bars_held": trade.bars_held,
            "qty": trade.qty,
            "is_partial": trade.is_partial,
            "trade_date": _to_et_date(entry_time),
            "source": self._source,
            # Entry context
            "entry_sm_value": getattr(trade, 'entry_sm_value', None),
            "entry_sm_velocity": getattr(trade, 'entry_sm_velocity', None),
            "entry_rsi_value": getattr(trade, 'entry_rsi_value', None),
            "entry_bar_volume": getattr(trade, 'entry_bar_volume', None),
            "entry_minutes_from_open": getattr(trade, 'entry_minutes_from_open', None),
            # Exit context
            "exit_sm_value": getattr(trade, 'exit_sm_value', None),
            "exit_rsi_value": getattr(trade, 'exit_rsi_value', None),
            "is_runner": getattr(trade, 'is_runner', None),
            # Gate state
            "gate_vix_close": getattr(trade, 'gate_vix_close', None),
            "gate_leledc_active": getattr(trade, 'gate_leledc_active', None),
            "gate_atr_value": getattr(trade, 'gate_atr_value', None),
            "gate_adr_ratio": getattr(trade, 'gate_adr_ratio', None),
            "gate_leledc_count": getattr(trade, 'gate_leledc_count', None),
            # MFE/MAE
            "mfe_pts": getattr(trade, 'mfe_pts', None),
            "mae_pts": getattr(trade, 'mae_pts', None),
            # Structure exit
            "structure_exit_level": getattr(trade, 'structure_exit_level', None),
            # Entry bar context
            "entry_bar_range": getattr(trade, 'entry_bar_range', None),
            "concurrent_positions": getattr(trade, 'concurrent_positions', None),
            "streak_at_entry": getattr(trade, 'streak_at_entry', None),
            # Trade grouping + slippage
            "trade_group_id": getattr(trade, 'trade_group_id', None),
            "signal_price": getattr(trade, 'signal_price', None),
            "config_id": self._config_id_map.get(trade.strategy_id),
            # ICT level proximity at entry
            "ict_near_levels": getattr(trade, 'ict_near_levels', None),
        }
        # Strip None values to let DB defaults apply
        return {k: v for k, v in row.items() if v is not None}

    def _correction_to_row(self, trade) -> dict:
        """Build update dict for trade correction."""
        commission = _compute_commission(trade.instrument, trade.qty)
        pnl_net = trade.pnl_dollar - commission

        entry_time = trade.entry_time
        if entry_time and entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        exit_time = trade.exit_time
        if exit_time and exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)

        return {
            # Match key (trade_group_id preferred, fallback to strategy_id + entry_time)
            "trade_group_id": getattr(trade, 'trade_group_id', None),
            "strategy_id": trade.strategy_id,
            "entry_time": entry_time.isoformat() if entry_time else None,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "is_partial": trade.is_partial,
            # Updated fields
            "exit_price": trade.exit_price,
            "pts": trade.pts,
            "pnl_net": round(pnl_net, 2),
        }

    def _blocked_to_row(self, payload: dict) -> dict:
        """Convert signal_blocked event payload to Supabase row."""
        signal_time = payload.get("time")
        if signal_time is None:
            logger.warning("[DbLogger] Blocked signal missing 'time' — skipping")
            return None
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)

        reason = payload.get("reason", "")
        gate_type = payload.get("gate_types") or _extract_gate_type(reason)

        row = {
            "signal_time": signal_time.isoformat(),
            "strategy_id": payload.get("strategy_id", ""),
            "instrument": payload.get("instrument", ""),
            "side": payload.get("side", ""),
            "price": payload.get("price", 0.0),
            "signal_price": payload.get("signal_price"),
            "sm_value": payload.get("sm_value"),
            "rsi_value": payload.get("rsi_value"),
            "gate_type": gate_type,
            "reason": reason,
            "gate_vix_close": payload.get("gate_vix_close"),
            "gate_leledc_active": payload.get("gate_leledc_active"),
            "gate_atr_value": payload.get("gate_atr_value"),
            "gate_adr_ratio": payload.get("gate_adr_ratio"),
            "signal_date": _to_et_date(signal_time),
            "source": self._source,
        }
        # Strip None values to let DB defaults apply
        return {k: v for k, v in row.items() if v is not None}

    def _structure_bar_to_row(self, payload: dict) -> Optional[dict]:
        """Convert structure_bar event payload to Supabase row."""
        bar_time = payload.get("bar_time")
        if bar_time is None:
            return None
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        row = {
            "bar_time": bar_time.isoformat(),
            "strategy_id": payload.get("strategy_id", ""),
            "instrument": payload.get("instrument", ""),
            "bar_close": payload.get("bar_close"),
            "swing_high": payload.get("swing_high"),
            "swing_low": payload.get("swing_low"),
            "position": payload.get("position", 0),
            "entry_price": payload.get("entry_price"),
            "runner_profit_pts": payload.get("runner_profit_pts"),
            "distance_to_level_pts": payload.get("distance_to_level_pts"),
            "near_miss": payload.get("near_miss", False),
            "trade_date": _to_et_date(bar_time),
        }
        # Strip None values to let DB defaults apply
        return {k: v for k, v in row.items() if v is not None}

    def _gate_state_rows(self, safety, date_str: str) -> list[dict]:
        """Build gate_state_snapshots rows from SafetyManager state."""
        rows = []
        # Collect all instruments from safety manager's gate tracking
        instruments = set()
        instruments.update(getattr(safety, '_prior_day_atr_value', {}).keys())
        instruments.update(getattr(safety, '_adr_value', {}).keys())
        instruments.update(getattr(safety, '_leledc_bull_count', {}).keys())
        instruments.update(getattr(safety, '_prior_day_levels', {}).keys())
        # Include instruments with VWAP data (ensures all tracked instruments appear)
        instruments.update(getattr(safety, '_vwap_num', {}).keys())

        vix_close = getattr(safety, '_vix_close', None)

        for inst in instruments:
            levels = getattr(safety, '_prior_day_levels', {}).get(inst, {})
            # RTH OHLC: prefer _adr_rth_session (MNQ), fall back to _current_rth (MES)
            session = getattr(safety, '_adr_rth_session', {}).get(inst, {})
            if not session.get("open"):
                # MES uses _current_rth for prior-day level tracking — use it for OHLC
                current_rth = getattr(safety, '_current_rth', {}).get(inst, {})
                if current_rth:
                    closes = current_rth.get("closes", [])
                    session = {
                        "open": closes[0] if closes else None,
                        "high": current_rth.get("high"),
                        "low": current_rth.get("low"),
                        "close": closes[-1] if closes else None,
                    }

            # VWAP: compute final value from accumulators
            vwap_den = getattr(safety, '_vwap_den', {}).get(inst, 0.0)
            vwap_num = getattr(safety, '_vwap_num', {}).get(inst, 0.0)
            vwap_close = round(vwap_num / vwap_den, 2) if vwap_den > 0 else None

            # Opening range
            or_high = getattr(safety, '_or_high', {}).get(inst)
            or_low = getattr(safety, '_or_low', {}).get(inst)

            row = {
                "snapshot_date": date_str,
                "instrument": inst,
                "vix_close": vix_close,
                "adr_value": getattr(safety, '_adr_value', {}).get(inst),
                "atr_value": getattr(safety, '_prior_day_atr_value', {}).get(inst),
                "leledc_bull_count": getattr(safety, '_leledc_bull_count', {}).get(inst),
                "leledc_bear_count": getattr(safety, '_leledc_bear_count', {}).get(inst),
                "prior_day_high": levels.get("high"),
                "prior_day_low": levels.get("low"),
                "prior_day_vpoc": levels.get("vpoc"),
                "prior_day_vah": levels.get("vah"),
                "prior_day_val": levels.get("val"),
                "rth_open": session.get("open"),
                "rth_high": session.get("high"),
                "rth_low": session.get("low"),
                "rth_close": session.get("close"),
                "vwap_close": vwap_close,
                "opening_range_high": or_high,
                "opening_range_low": or_low,
                "weekly_vpoc": getattr(safety, '_weekly_vpoc', {}).get(inst),
                "weekly_val": getattr(safety, '_weekly_val', {}).get(inst),
                "dvpoc_price": getattr(safety, '_developing_vpoc', {}).get(inst),
                "dvpoc_strength": getattr(safety, '_daily_vpoc_strength', {}).get(inst),
                "dvpoc_stability": getattr(safety, '_dvpoc_stability', {}).get(inst),
            }
            # Strip None values
            row = {k: v for k, v in row.items() if v is not None}
            rows.append(row)

        return rows

    def _apply_correction(self, row: dict) -> None:
        """Apply a trade correction using the best available match key."""
        update_data = {
            "exit_price": row["exit_price"],
            "pts": row["pts"],
            "pnl_net": row["pnl_net"],
        }
        # Prefer trade_group_id + is_partial (unique) over strategy_id + entry_time
        # Always filter by source to avoid cross-contaminating backtest data
        if row.get("trade_group_id"):
            result = self._client.table("trades").update(update_data).eq(
                "trade_group_id", row["trade_group_id"]
            ).eq("is_partial", row.get("is_partial", False)).eq(
                "source", self._source
            ).execute()
        else:
            result = self._client.table("trades").update(update_data).eq(
                "strategy_id", row["strategy_id"]
            ).eq("entry_time", row["entry_time"]).eq(
                "is_partial", row.get("is_partial", False)
            ).eq("source", self._source).execute()

        # Warn if no rows matched (correction target not found)
        if not result.data:
            logger.warning(
                f"[DbLogger] Correction matched 0 rows for "
                f"{row.get('strategy_id')} @ {row.get('entry_time')}"
            )

    # ------------------------------------------------------------------
    # Background drain loop
    # ------------------------------------------------------------------

    async def _write_item(self, item_type: str, row: dict) -> None:
        """Execute a single Supabase write (called from drain loop)."""
        if item_type == "trade":
            await asyncio.to_thread(
                lambda r=row: self._client.table("trades").upsert(
                    r, on_conflict="strategy_id,entry_time,is_partial,source"
                ).execute()
            )
        elif item_type == "trade_correction":
            await asyncio.to_thread(
                lambda r=row: self._apply_correction(r)
            )
        elif item_type == "blocked_signal":
            await asyncio.to_thread(
                lambda r=row: self._client.table("blocked_signals").upsert(
                    r, on_conflict="signal_time,strategy_id,instrument,source",
                    ignore_duplicates=True
                ).execute()
            )
        elif item_type == "structure_bar":
            await asyncio.to_thread(
                lambda r=row: self._client.table("structure_bar_logs").upsert(
                    r, on_conflict="bar_time,strategy_id"
                ).execute()
            )
        elif item_type == "gate_snapshot":
            await asyncio.to_thread(lambda r=row: (
                self._client.table("gate_state_snapshots")
                .upsert(r, on_conflict="snapshot_date,instrument")
                .execute()
            ))
        elif item_type == "refresh_views":
            await asyncio.to_thread(
                lambda: self._client.rpc("refresh_trading_views").execute()
            )
            logger.info("[DbLogger] Materialized views refreshed")

    async def _drain_loop(self) -> None:
        """Background task: drain queue to Supabase.

        All Supabase calls are wrapped in asyncio.to_thread() to avoid blocking
        the event loop (supabase-py client is synchronous HTTP).
        One retry with 1s delay for transient network failures.
        """
        consecutive_errors = 0

        while self._running:
            try:
                item_type, row = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._write_item(item_type, row)
                logger.debug(f"[DbLogger] {item_type} logged: {row.get('strategy_id')}")
                consecutive_errors = 0
            except Exception as first_err:
                # One retry after 1s for transient failures
                try:
                    await asyncio.sleep(1.0)
                    await self._write_item(item_type, row)
                    logger.debug(f"[DbLogger] {item_type} logged (retry): {row.get('strategy_id')}")
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= 3:
                        logger.warning(f"[DbLogger] Supabase write failed ({item_type}): {e}")
                    elif consecutive_errors == 4:
                        logger.warning("[DbLogger] Suppressing repeated Supabase errors (will log every 50th)")
                    elif consecutive_errors % 50 == 0:
                        logger.warning(f"[DbLogger] Supabase still failing ({consecutive_errors} errors): {e}")

        # Drain remaining items on shutdown (best-effort, blocking OK during shutdown)
        drain_failures = 0
        while not self._queue.empty() and drain_failures < 5:
            try:
                item_type, row = self._queue.get_nowait()
                if item_type == "trade":
                    self._client.table("trades").upsert(
                        row, on_conflict="strategy_id,entry_time,is_partial,source"
                    ).execute()
                elif item_type == "blocked_signal":
                    self._client.table("blocked_signals").upsert(
                        row, on_conflict="signal_time,strategy_id,instrument,source",
                        ignore_duplicates=True
                    ).execute()
                elif item_type == "trade_correction":
                    self._apply_correction(row)
                elif item_type == "structure_bar":
                    self._client.table("structure_bar_logs").upsert(
                        row, on_conflict="bar_time,strategy_id"
                    ).execute()
                elif item_type == "gate_snapshot":
                    self._client.table("gate_state_snapshots").upsert(
                        row, on_conflict="snapshot_date,instrument"
                    ).execute()
            except Exception:
                drain_failures += 1
                continue
