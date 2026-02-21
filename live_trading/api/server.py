"""
FastAPI server for the live trading engine.

Provides REST endpoints for status, trade history, configuration, and
engine control. A WebSocket endpoint broadcasts real-time events (bars,
signals, trades, status changes) to connected dashboard clients.

The server receives an EngineHandle at startup that provides a clean
interface to the engine internals -- the server never touches strategy
state directly.
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from engine.config import EngineConfig, SafetyConfig, StrategyConfig
from engine.events import Bar, EventBus, Signal, TradeRecord

from .models import (
    ConfigResponse,
    ControlAction,
    ControlActionType,
    DailyPnLEntry,
    DrawdownToggleBody,
    EngineStatus,
    ErrorResponse,
    PositionInfo,
    QtyOverrideBody,
    TradeResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EngineHandle -- abstraction layer between the API and the engine
# ---------------------------------------------------------------------------

class EngineHandle:
    """Interface the API server uses to interact with the engine.

    The live runner creates an EngineHandle wired to the real engine state
    and passes it to create_app(). This avoids circular imports and keeps
    the server decoupled from engine internals.

    get_status and get_trades are synchronous (engine state is in-process memory).
    kill_switch is async (needs to close positions via order manager).
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: EngineConfig,
        get_status: Any,                # Callable[[], dict]
        get_trades: Any,                # Callable[[], list[TradeRecord]]
        get_bars: Any = None,           # Callable[[str], list[dict]]
        pause_trading: Any = None,      # Callable[[], None]
        resume_trading: Any = None,     # Callable[[], tuple[bool, str]]
        kill_switch: Any = None,        # Callable[[], Awaitable[None]]
        set_strategy_paused: Any = None,   # Callable[[str, bool], tuple[bool, str]]
        set_strategy_qty: Any = None,      # Callable[[str, int], tuple[bool, str]]
        set_drawdown_enabled: Any = None,  # Callable[[bool], tuple[bool, str]]
        force_resume_all: Any = None,      # Callable[[], tuple[bool, str]]
    ):
        self.event_bus = event_bus
        self.config = config
        self._get_status = get_status
        self._get_trades = get_trades
        self._get_bars = get_bars
        self._pause_trading = pause_trading
        self._resume_trading = resume_trading
        self._kill_switch = kill_switch
        self._set_strategy_paused = set_strategy_paused
        self._set_strategy_qty = set_strategy_qty
        self._set_drawdown_enabled = set_drawdown_enabled
        self._force_resume_all = force_resume_all
        self._close_strategy_position = None  # Set later if available

    def get_status(self) -> dict:
        return self._get_status()

    def get_trades(self) -> list[TradeRecord]:
        return self._get_trades()

    def pause(self) -> None:
        self._pause_trading()

    def resume(self) -> tuple[bool, str]:
        return self._resume_trading()

    def get_bars(self, instrument: str) -> list[dict]:
        if self._get_bars:
            return self._get_bars(instrument)
        return []

    async def kill(self) -> None:
        await self._kill_switch()

    def set_strategy_paused(self, strategy_id: str, paused: bool) -> tuple[bool, str]:
        if self._set_strategy_paused:
            return self._set_strategy_paused(strategy_id, paused)
        return False, "Not available"

    def set_strategy_qty(self, strategy_id: str, qty: int) -> tuple[bool, str]:
        if self._set_strategy_qty:
            return self._set_strategy_qty(strategy_id, qty)
        return False, "Not available"

    def set_drawdown_enabled(self, enabled: bool) -> tuple[bool, str]:
        if self._set_drawdown_enabled:
            return self._set_drawdown_enabled(enabled)
        return False, "Not available"

    def force_resume_all(self) -> tuple[bool, str]:
        if self._force_resume_all:
            return self._force_resume_all()
        return False, "Not available"

    async def close_strategy_position(self, strategy_id: str) -> tuple[bool, str]:
        if self._close_strategy_position:
            return await self._close_strategy_position(strategy_id)
        return False, "Not available"

    def get_config_snapshot(self) -> ConfigResponse:
        cfg = self.config
        strategies = []
        for sc in cfg.strategies:
            strategies.append(asdict(sc))
        safety = asdict(cfg.safety)
        return ConfigResponse(
            strategies=strategies,
            safety=safety,
            paper_mode=cfg.safety.paper_mode,
            warmup_bars=cfg.warmup_bars,
        )


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------

class WebSocketManager:
    """Manages multiple WebSocket connections and broadcasts JSON events."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info(f"WebSocket client connected. Total: {len(self._connections)}")

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        logger.info(f"WebSocket client disconnected. Total: {len(self._connections)}")

    async def broadcast(self, event_type: str, payload: dict) -> None:
        """Send a JSON message to all connected clients.

        Message format: {"type": <type>, "data": <payload>, "ts": <iso timestamp>}
        Disconnected clients are silently removed.
        """
        if not self._connections:
            return

        message = json.dumps({
            "type": event_type,
            "data": payload,
            "ts": datetime.utcnow().isoformat() + "Z",
        }, default=_json_serial)

        dead: list[WebSocket] = []
        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._connections.remove(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


def _json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# ---------------------------------------------------------------------------
# Event bus bridge (sync EventBus -> async WebSocket broadcast)
# ---------------------------------------------------------------------------

class EventBridge:
    """Bridges the synchronous EventBus to the async WebSocket manager.

    Engine events are synchronous (from the strategy loop). This bridge
    enqueues them and a background task drains the queue into WebSocket
    broadcasts.
    """

    def __init__(self, event_bus: EventBus, ws_manager: WebSocketManager) -> None:
        self._event_bus = event_bus
        self._ws_manager = ws_manager
        self._queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue(maxsize=1000)
        self._task: Optional[asyncio.Task] = None

        # Subscribe to engine events
        event_bus.subscribe("bar", self._on_bar)
        event_bus.subscribe("signal", self._on_signal)
        event_bus.subscribe("trade_closed", self._on_trade_closed)
        event_bus.subscribe("trade_corrected", self._on_trade_corrected)
        event_bus.subscribe("fill", self._on_fill)
        event_bus.subscribe("status_change", self._on_status_change)
        event_bus.subscribe("error", self._on_error)

    def start(self) -> None:
        """Start the background drain task. Must be called inside a running event loop."""
        self._task = asyncio.create_task(self._drain_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _enqueue(self, event_type: str, payload: dict) -> None:
        """Non-blocking enqueue from synchronous handler."""
        try:
            self._queue.put_nowait((event_type, payload))
        except asyncio.QueueFull:
            logger.warning(f"WebSocket event queue full, dropping {event_type}")

    async def _drain_loop(self) -> None:
        """Background loop that drains queued events into WebSocket broadcasts."""
        while True:
            event_type, payload = await self._queue.get()
            try:
                await self._ws_manager.broadcast(event_type, payload)
            except Exception as e:
                logger.error(f"WebSocket broadcast error: {e}")

    # --- Event handlers (synchronous, called by EventBus) ---

    def _on_bar(self, bar: Bar) -> None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        et = bar.timestamp.astimezone(_ET)
        offset_seconds = int(et.utcoffset().total_seconds())
        self._enqueue("bar", {
            "instrument": bar.instrument,
            "timestamp": bar.timestamp.isoformat(),
            "time": int(bar.timestamp.timestamp()) + offset_seconds,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })

    def _on_signal(self, signal: Signal) -> None:
        self._enqueue("signal", {
            "type": signal.type.value,
            "instrument": signal.instrument,
            "reason": signal.reason,
            "sm_value": signal.sm_value,
            "rsi_value": signal.rsi_value,
        })

    def _on_trade_closed(self, trade: TradeRecord) -> None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        # Shift entry/exit times to ET epoch for chart marker alignment
        entry_et_epoch = None
        exit_et_epoch = None
        if trade.entry_time:
            et = trade.entry_time.astimezone(_ET)
            entry_et_epoch = int(trade.entry_time.timestamp()) + int(et.utcoffset().total_seconds())
        if trade.exit_time:
            et = trade.exit_time.astimezone(_ET)
            exit_et_epoch = int(trade.exit_time.timestamp()) + int(et.utcoffset().total_seconds())
        # Use "trade" event name (matches dashboard WSMessage type)
        self._enqueue("trade", {
            "instrument": trade.instrument,
            "strategy_id": trade.strategy_id,
            "side": trade.side.upper(),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "entry_time_et_epoch": entry_et_epoch,
            "exit_time_et_epoch": exit_et_epoch,
            "pts": trade.pts,
            "pnl": trade.pnl_dollar,
            "exit_reason": trade.exit_reason,
            "bars_held": trade.bars_held,
        })

    def _on_trade_corrected(self, trade: TradeRecord) -> None:
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        entry_et_epoch = None
        exit_et_epoch = None
        if trade.entry_time:
            et = trade.entry_time.astimezone(_ET)
            entry_et_epoch = int(trade.entry_time.timestamp()) + int(et.utcoffset().total_seconds())
        if trade.exit_time:
            et = trade.exit_time.astimezone(_ET)
            exit_et_epoch = int(trade.exit_time.timestamp()) + int(et.utcoffset().total_seconds())
        self._enqueue("trade_update", {
            "instrument": trade.instrument,
            "strategy_id": trade.strategy_id,
            "side": trade.side.upper(),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "entry_time_et_epoch": entry_et_epoch,
            "exit_time_et_epoch": exit_et_epoch,
            "pts": trade.pts,
            "pnl": trade.pnl_dollar,
            "exit_reason": trade.exit_reason,
            "bars_held": trade.bars_held,
        })

    def _on_fill(self, fill: dict) -> None:
        self._enqueue("fill", fill)

    def _on_status_change(self, status: dict) -> None:
        # Use "safety_status" to avoid overwriting full StatusData on dashboard
        self._enqueue("safety_status", status)

    def _on_error(self, error: dict) -> None:
        self._enqueue("error", error)


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(handle: EngineHandle) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        handle: EngineHandle providing access to engine state and controls.

    Returns:
        Configured FastAPI app ready to be served by uvicorn.
    """
    app = FastAPI(
        title="NQ Trading Engine",
        description="Live trading engine API for MNQ/MES SM+RSI strategy",
        version="1.0.0",
    )

    # CORS for local dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_manager = WebSocketManager()
    bridge = EventBridge(handle.event_bus, ws_manager)

    # Auto-save state
    _autosave_task: Optional[asyncio.Task] = None
    _save_lock = asyncio.Lock()
    AUTOSAVE_INTERVAL = 300  # 5 minutes

    async def _autosave_loop() -> None:
        """Background task: auto-save session every 5 minutes."""
        await asyncio.sleep(60)  # Wait 1 min after startup before first save
        while True:
            try:
                await asyncio.sleep(AUTOSAVE_INTERVAL)
                async with _save_lock:
                    trades = handle.get_trades()
                    if trades:  # Only save if there are trades
                        _do_save_session()
                        logger.info(f"[AutoSave] Session saved ({len(trades)} trades)")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AutoSave] Error: {e}")

    @app.on_event("startup")
    async def startup() -> None:
        nonlocal _autosave_task
        bridge.start()
        _autosave_task = asyncio.create_task(_autosave_loop())
        logger.info("API server started, WebSocket bridge active, auto-save enabled")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        # Auto-save on shutdown
        try:
            trades = handle.get_trades()
            if trades:
                _do_save_session()
                logger.info(f"[Shutdown] Session saved ({len(trades)} trades)")
        except Exception as e:
            logger.error(f"[Shutdown] Save failed: {e}")
        if _autosave_task:
            _autosave_task.cancel()
            try:
                await _autosave_task
            except asyncio.CancelledError:
                pass
        await bridge.stop()
        logger.info("API server shutting down")

    # -------------------------------------------------------------------
    # REST endpoints
    # -------------------------------------------------------------------

    @app.get("/api/status")
    def get_status() -> dict:
        """Get current engine status including positions and P&L."""
        return handle.get_status()

    @app.get("/api/trades")
    def get_trades() -> list[dict]:
        """Get all trades from the current session (flat array).

        Includes ET epoch timestamps for chart marker alignment.
        """
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        trades = handle.get_trades()
        result = []
        for t in trades:
            entry_et_epoch = None
            exit_et_epoch = None
            if t.entry_time:
                et = t.entry_time.astimezone(_ET)
                entry_et_epoch = int(t.entry_time.timestamp()) + int(et.utcoffset().total_seconds())
            if t.exit_time:
                et = t.exit_time.astimezone(_ET)
                exit_et_epoch = int(t.exit_time.timestamp()) + int(et.utcoffset().total_seconds())
            result.append({
                "instrument": t.instrument,
                "strategy_id": t.strategy_id,
                "side": t.side.upper(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_time_et_epoch": entry_et_epoch,
                "exit_time_et_epoch": exit_et_epoch,
                "pts": t.pts,
                "pnl": t.pnl_dollar,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            })
        return result

    @app.get("/api/daily_pnl")
    def get_daily_pnl() -> list[dict]:
        """Get daily P&L breakdown computed from trade history."""
        trades = handle.get_trades()
        daily: dict[str, float] = {}
        for t in trades:
            if t.exit_time is None:
                continue
            date = t.exit_time.strftime("%Y-%m-%d")
            if date not in daily:
                daily[date] = 0.0
            daily[date] += t.pnl_dollar

        result = []
        cumulative = 0.0
        for date in sorted(daily.keys()):
            cumulative += daily[date]
            result.append({
                "date": date,
                "pnl": round(daily[date], 2),
                "cumulative": round(cumulative, 2),
            })
        return result

    @app.get("/api/bars/{instrument}")
    def get_bars(instrument: str) -> list[dict]:
        """Get all completed 1-min bars for an instrument."""
        return handle.get_bars(instrument.upper())

    # -------------------------------------------------------------------
    # Session save/load
    # -------------------------------------------------------------------

    SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"
    SESSIONS_DIR.mkdir(exist_ok=True)

    def _do_save_session() -> dict:
        """Internal save logic — called by route, auto-save, and shutdown."""
        # Collect bars for all instruments
        bars_data: dict[str, list[dict]] = {}
        for sc in handle.config.strategies:
            inst = sc.instrument
            if inst not in bars_data:
                bars_data[inst] = handle.get_bars(inst)

        # Collect trades
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        trades = handle.get_trades()
        trades_data = []
        for t in trades:
            entry_et_epoch = None
            exit_et_epoch = None
            if t.entry_time:
                et = t.entry_time.astimezone(_ET)
                entry_et_epoch = int(t.entry_time.timestamp()) + int(et.utcoffset().total_seconds())
            if t.exit_time:
                et = t.exit_time.astimezone(_ET)
                exit_et_epoch = int(t.exit_time.timestamp()) + int(et.utcoffset().total_seconds())
            trades_data.append({
                "instrument": t.instrument,
                "strategy_id": t.strategy_id,
                "side": t.side.upper(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_time_et_epoch": entry_et_epoch,
                "exit_time_et_epoch": exit_et_epoch,
                "pts": t.pts,
                "pnl": t.pnl_dollar,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            })

        today = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"session_{today}.json"
        filepath = SESSIONS_DIR / filename

        session = {
            "date": today,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "timezone": "ET",  # Bar times are ET-shifted epochs
            "bars": bars_data,
            "trades": trades_data,
        }

        filepath.write_text(json.dumps(session, indent=2))
        logger.info(f"Session saved: {filepath} ({sum(len(b) for b in bars_data.values())} bars, {len(trades_data)} trades)")
        return {"filename": filename, "bars": sum(len(b) for b in bars_data.values()), "trades": len(trades_data)}

    @app.post("/api/session/save")
    def save_session() -> dict:
        """Save current bars + trades to a JSON file for later replay."""
        return _do_save_session()

    @app.get("/api/sessions")
    def list_sessions() -> list[dict]:
        """List all saved session files."""
        sessions = []
        for f in sorted(SESSIONS_DIR.glob("session_*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                bar_count = sum(len(b) for b in data.get("bars", {}).values())
                sessions.append({
                    "filename": f.name,
                    "date": data.get("date", ""),
                    "saved_at": data.get("saved_at", ""),
                    "bars": bar_count,
                    "trades": len(data.get("trades", [])),
                })
            except Exception:
                sessions.append({"filename": f.name, "date": "", "bars": 0, "trades": 0})
        return sessions

    @app.get("/api/session/{filename}")
    def load_session(filename: str) -> dict:
        """Load a saved session file."""
        if "/" in filename or "\\" in filename or ".." in filename:
            return {"error": "Invalid filename"}
        filepath = SESSIONS_DIR / filename
        if not filepath.exists() or not filepath.name.startswith("session_"):
            return {"error": "Session not found"}
        data = json.loads(filepath.read_text())
        return data

    @app.get("/api/config")
    def get_config() -> ConfigResponse:
        """Get current engine configuration."""
        return handle.get_config_snapshot()

    @app.post("/api/control/pause")
    def pause_trading() -> dict:
        """Pause trading (existing positions remain open)."""
        handle.pause()
        logger.info("Trading paused via API")
        return {"status": "paused"}

    @app.post("/api/control/resume")
    def resume_trading() -> dict:
        """Resume trading after a pause."""
        ok, reason = handle.resume()
        if not ok:
            return {"status": "error", "detail": reason}
        logger.info("Trading resumed via API")
        return {"status": "resumed"}

    # -------------------------------------------------------------------
    # Safety control endpoints
    # -------------------------------------------------------------------

    @app.post("/api/safety/strategy/{strategy_id}/pause")
    def pause_strategy(strategy_id: str) -> dict:
        """Pause a specific strategy."""
        ok, msg = handle.set_strategy_paused(strategy_id, True)
        return {"ok": ok, "message": msg}

    @app.post("/api/safety/strategy/{strategy_id}/resume")
    def resume_strategy(strategy_id: str) -> dict:
        """Resume a specific strategy (sets manual_override to prevent auto re-pause)."""
        ok, msg = handle.set_strategy_paused(strategy_id, False)
        return {"ok": ok, "message": msg}

    @app.post("/api/safety/strategy/{strategy_id}/qty")
    def set_strategy_qty(strategy_id: str, body: QtyOverrideBody) -> dict:
        """Set contract qty override for a strategy. Body: {qty: N}"""
        ok, msg = handle.set_strategy_qty(strategy_id, body.qty)
        return {"ok": ok, "message": msg}

    @app.post("/api/safety/drawdown/toggle")
    def toggle_drawdown(body: DrawdownToggleBody) -> dict:
        """Toggle auto drawdown rules. Body: {enabled: bool}"""
        ok, msg = handle.set_drawdown_enabled(body.enabled)
        return {"ok": ok, "message": msg}

    @app.post("/api/control/force_resume")
    def force_resume() -> dict:
        """Force resume all — clear all pauses, halts, and overrides."""
        ok, msg = handle.force_resume_all()
        return {"ok": ok, "message": msg}

    @app.post("/api/control/strategy/{strategy_id}/close")
    async def close_strategy_position(strategy_id: str) -> dict:
        """Manually close a strategy's open position."""
        ok, msg = await handle.close_strategy_position(strategy_id)
        return {"ok": ok, "message": msg}

    @app.post("/api/control/kill")
    async def kill_trading() -> dict:
        """Kill switch: flatten all positions and halt."""
        await handle.kill()
        logger.warning("KILL SWITCH activated via API")
        # Broadcast updated status to all WS clients
        try:
            status = handle.get_status()
            await ws_manager.broadcast("status", status)
        except Exception:
            pass
        return {"status": "killed", "message": "All positions flattened, trading halted"}

    # -------------------------------------------------------------------
    # WebSocket
    # -------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        """WebSocket endpoint for real-time event streaming.

        Clients receive JSON messages:
            {"type": "bar", "data": {...}, "ts": "..."}
            {"type": "signal", "data": {...}, "ts": "..."}
            {"type": "trade", "data": {...}, "ts": "..."}
            {"type": "status", "data": {...}, "ts": "..."}

        Clients can send JSON commands:
            {"command": "pause"} / {"command": "resume"} / {"command": "kill"}
        """
        await ws_manager.connect(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                    command = msg.get("command", "")
                    if command == "pause":
                        handle.pause()
                        await ws.send_text(json.dumps({"ack": "paused"}))
                    elif command == "resume":
                        ok, reason = handle.resume()
                        if ok:
                            await ws.send_text(json.dumps({"ack": "resumed"}))
                        else:
                            await ws.send_text(json.dumps({"error": reason}))
                    elif command == "kill":
                        await handle.kill()
                        await ws.send_text(json.dumps({"ack": "killed"}))
                        # Broadcast updated status
                        try:
                            status = handle.get_status()
                            await ws_manager.broadcast("status", status)
                        except Exception:
                            pass
                    elif command == "strategy_pause":
                        sid = msg.get("strategy_id", "")
                        ok, reason = handle.set_strategy_paused(sid, True)
                        await ws.send_text(json.dumps({"ack": "strategy_paused" if ok else "error", "message": reason}))
                    elif command == "strategy_resume":
                        sid = msg.get("strategy_id", "")
                        ok, reason = handle.set_strategy_paused(sid, False)
                        await ws.send_text(json.dumps({"ack": "strategy_resumed" if ok else "error", "message": reason}))
                    elif command == "strategy_qty":
                        sid = msg.get("strategy_id", "")
                        try:
                            body = QtyOverrideBody(qty=msg.get("qty"))
                        except Exception as e:
                            await ws.send_text(json.dumps({"error": str(e)}))
                            continue
                        ok, reason = handle.set_strategy_qty(sid, body.qty)
                        await ws.send_text(json.dumps({"ack": "strategy_qty_set" if ok else "error", "message": reason}))
                    elif command == "drawdown_toggle":
                        try:
                            body = DrawdownToggleBody(enabled=msg.get("enabled"))
                        except Exception as e:
                            await ws.send_text(json.dumps({"error": str(e)}))
                            continue
                        ok, reason = handle.set_drawdown_enabled(body.enabled)
                        await ws.send_text(json.dumps({"ack": "drawdown_toggled" if ok else "error", "message": reason}))
                    elif command == "force_resume":
                        ok, reason = handle.force_resume_all()
                        await ws.send_text(json.dumps({"ack": "force_resumed" if ok else "error", "message": reason}))
                    elif command == "strategy_close":
                        sid = msg.get("strategy_id", "")
                        ok, reason = await handle.close_strategy_position(sid)
                        await ws.send_text(json.dumps({"ack": "strategy_closed" if ok else "error", "message": reason}))
                    else:
                        await ws.send_text(json.dumps({"error": f"unknown command: {command}"}))
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"error": "invalid JSON"}))
        except WebSocketDisconnect:
            await ws_manager.disconnect(ws)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await ws_manager.disconnect(ws)

    return app
