"""
Configuration for the live trading engine.

All strategy parameters, safety thresholds, and instrument settings.
Validated params from v11 MNQ and v9.4 MES TradingView-confirmed strategies.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyConfig:
    """Strategy parameters for a single instrument."""
    instrument: str           # "MNQ" or "MES"
    strategy_id: str = ""     # Unique ID (e.g. "MNQ_V11", "MNQ_V15"); defaults to instrument
    sm_index: int = 10        # SM summation period
    sm_flow: int = 12         # SM RSI period on volume flow
    sm_norm: int = 200        # SM peak normalization lookback
    sm_ema: int = 100         # SM EMA smoothing on PVI/NVI
    sm_threshold: float = 0.0 # Min |SM| for direction confirmation
    exit_mode: str = "sm_flip"  # "sm_flip" (default) or "tp_scalp"
    tp_pts: int = 0           # Take profit in points (tp_scalp mode)
    trail_activate_pts: int = 5   # Activate trailing stop after this MFE
    trail_distance_pts: int = 8   # Trail distance once activated
    rsi_len: int = 10         # RSI period (computed on 5-min bars)
    rsi_buy: int = 60         # RSI level for long entry
    rsi_sell: int = 40        # RSI level for short entry
    cooldown: int = 20        # Bars between trades (1-min bars)
    max_loss_pts: int = 50    # Max loss stop in points (0 = off)
    dollar_per_pt: float = 2.0  # Dollar per point
    commission_per_side: float = 0.52  # Per-contract commission
    session_start_et: str = "10:00"   # RTH start (Eastern Time)
    session_end_et: str = "15:45"     # Last entry allowed
    session_close_et: str = "16:00"   # Force close all positions


@dataclass
class SafetyConfig:
    """Safety limits and circuit breakers."""
    max_daily_loss: float = 500.0     # Max daily loss in dollars before halt
    max_position_size: int = 2        # Max contracts per instrument (across strategies)
    max_consecutive_losses: int = 5   # Consecutive losses before pause
    heartbeat_timeout_sec: int = 90   # Alert if no data for this long (polls every 60s)
    flatten_timeout_sec: int = 300   # Flatten all if no data for this long
    order_fill_timeout_sec: int = 5   # Alert if no fill within this time
    recon_interval_sec: int = 60      # Position reconciliation interval
    paper_mode: bool = True           # Start in paper mode


@dataclass
class WebullConfig:
    """Webull API connection settings (deprecated -- Webull API never approved)."""
    app_key: str = ""
    app_secret: str = ""
    account_id: str = ""
    base_url: str = "https://api.webull.com"
    region_id: str = "us"


@dataclass
class TastytradeConfig:
    """tastytrade API connection settings.

    OAuth2 authentication -- requires one-time setup:
      1. my.tastytrade.com -> OAuth Applications -> Create App -> save client_secret
      2. OAuth Applications -> Manage -> Create Grant -> save refresh_token
      3. Note your account number (e.g. '5WX00000')

    Set is_sandbox=True for certification/paper trading environment.
    """
    client_secret: str = ""
    refresh_token: str = ""
    account_number: str = ""     # Leave empty to auto-select first account
    is_sandbox: bool = True      # True = cert environment, False = production


@dataclass
class EngineConfig:
    """Top-level engine configuration."""
    strategies: list = field(default_factory=list)  # List[StrategyConfig]
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    webull: WebullConfig = field(default_factory=WebullConfig)
    tastytrade: TastytradeConfig = field(default_factory=TastytradeConfig)
    broker: str = "mock"         # "mock", "tastytrade", or "webull"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    warmup_bars: int = 500  # Historical bars to load for SM warmup
    log_dir: str = "logs"


# Pre-built configs for validated strategies
MNQ_V11 = StrategyConfig(
    instrument="MNQ",
    strategy_id="MNQ_V11",
    sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,
    rsi_len=8, rsi_buy=60, rsi_sell=40,
    cooldown=20, max_loss_pts=50,
    dollar_per_pt=2.0,
)

MNQ_V11_1 = StrategyConfig(
    instrument="MNQ",
    strategy_id="MNQ_V11",
    sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,
    sm_threshold=0.15,  # Filter whipsaw entries near SM zero
    rsi_len=8, rsi_buy=60, rsi_sell=40,
    cooldown=20, max_loss_pts=50,
    dollar_per_pt=2.0,
)

MNQ_V15 = StrategyConfig(
    instrument="MNQ",
    strategy_id="MNQ_V15",
    sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,
    sm_threshold=0.0,  # v15 validated with threshold=0 on TV
    exit_mode="tp_scalp",
    tp_pts=5,
    trail_activate_pts=5,
    trail_distance_pts=8,
    rsi_len=8, rsi_buy=60, rsi_sell=40,
    cooldown=20, max_loss_pts=50,
    dollar_per_pt=2.0,
)

MES_V94 = StrategyConfig(
    instrument="MES",
    strategy_id="MES_V94",
    sm_index=20, sm_flow=12, sm_norm=400, sm_ema=255,
    rsi_len=10, rsi_buy=55, rsi_sell=45,
    cooldown=15, max_loss_pts=0,  # Stop loss hurts MES
    dollar_per_pt=5.0,
)

DEFAULT_CONFIG = EngineConfig(
    strategies=[MNQ_V11, MES_V94],
    safety=SafetyConfig(paper_mode=True),
)
