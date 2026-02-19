"""
Pydantic models for the trading engine REST API.

These models define the shape of all API responses and request bodies.
They are used by FastAPI for automatic validation and OpenAPI docs.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ControlActionType(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    KILL = "kill"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PositionInfo(BaseModel):
    """Current position for a single instrument."""
    instrument: str
    side: str = "FLAT"                              # "LONG", "SHORT", or "FLAT"
    entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0


class EngineStatus(BaseModel):
    """Top-level engine status snapshot."""
    connected: bool = False
    trading_active: bool = False
    paper_mode: bool = True
    positions: list[PositionInfo] = Field(default_factory=list)
    daily_pnl: float = 0.0
    uptime_seconds: float = 0.0
    trade_count_today: int = 0
    consecutive_losses: int = 0


class TradeResponse(BaseModel):
    """A completed trade record for the API."""
    instrument: str
    side: str                   # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pts: float
    pnl: float                  # Commission-adjusted P&L in dollars
    exit_reason: str
    bars_held: int = 0


class DailyPnLEntry(BaseModel):
    """Single day P&L entry."""
    date: str
    pnl: float
    cumulative: float


class ControlAction(BaseModel):
    """Request body for engine control actions."""
    action: ControlActionType


class ConfigResponse(BaseModel):
    """Current engine configuration snapshot."""
    strategies: list[dict] = Field(default_factory=list)
    safety: dict = Field(default_factory=dict)
    paper_mode: bool = True
    warmup_bars: int = 500


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    code: str = "UNKNOWN"
