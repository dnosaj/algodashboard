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
    EngineStatus,
    ErrorResponse,
    PositionInfo,
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
    ):
        self.event_bus = event_bus
        self.config = config
        self._get_status = get_status
        self._get_trades = get_trades
        self._get_bars = get_bars
        self._pause_trading = pause_trading
        self._resume_trading = resume_trading
        self._kill_switch = kill_switch

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
        self._enqueue("bar", {
            "instrument": bar.instrument,
            "timestamp": bar.timestamp.isoformat(),
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
        # Use "trade" event name (matches dashboard WSMessage type)
        self._enqueue("trade", {
            "instrument": trade.instrument,
            "strategy_id": trade.strategy_id,
            "side": trade.side.upper(),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "pts": trade.pts,
            "pnl": trade.pnl_dollar,
            "exit_reason": trade.exit_reason,
            "bars_held": trade.bars_held,
        })

    def _on_fill(self, fill: dict) -> None:
        self._enqueue("fill", fill)

    def _on_status_change(self, status: dict) -> None:
        # Use "status" event name (matches dashboard WSMessage type)
        self._enqueue("status", status)

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

    @app.on_event("startup")
    async def startup() -> None:
        bridge.start()
        logger.info("API server started, WebSocket bridge active")

    @app.on_event("shutdown")
    async def shutdown() -> None:
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
        """Get all trades from the current session (flat array)."""
        trades = handle.get_trades()
        return [
            {
                "instrument": t.instrument,
                "strategy_id": t.strategy_id,
                "side": t.side.upper(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "pts": t.pts,
                "pnl": t.pnl_dollar,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            }
            for t in trades
        ]

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

    @app.post("/api/session/save")
    def save_session() -> dict:
        """Save current bars + trades to a JSON file for later replay."""
        # Collect bars for all instruments
        bars_data: dict[str, list[dict]] = {}
        for sc in handle.config.strategies:
            inst = sc.instrument
            if inst not in bars_data:
                bars_data[inst] = handle.get_bars(inst)

        # Collect trades
        trades = handle.get_trades()
        trades_data = [
            {
                "instrument": t.instrument,
                "strategy_id": t.strategy_id,
                "side": t.side.upper(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "pts": t.pts,
                "pnl": t.pnl_dollar,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            }
            for t in trades
        ]

        today = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"session_{today}.json"
        filepath = SESSIONS_DIR / filename

        session = {
            "date": today,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "bars": bars_data,
            "trades": trades_data,
        }

        filepath.write_text(json.dumps(session, indent=2))
        logger.info(f"Session saved: {filepath} ({sum(len(b) for b in bars_data.values())} bars, {len(trades_data)} trades)")
        return {"filename": filename, "bars": sum(len(b) for b in bars_data.values()), "trades": len(trades_data)}

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
