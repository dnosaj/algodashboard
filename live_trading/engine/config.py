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
    entry_qty: int = 1                  # Default entry size (contracts)
    partial_tp_pts: int = 0              # Partial TP target in pts (0 = disabled)
    partial_qty: int = 1                 # Contracts to close at partial TP
    breakeven_after_bars: int = 0         # Close stale trades after N bars (0 = disabled)
    max_strategy_daily_loss: float = 0.0  # Max daily loss per strategy (0 = disabled)
    vix_death_zone_min: float = 0.0   # VIX death zone lower bound (0 = disabled)
    vix_death_zone_max: float = 0.0   # VIX death zone upper bound (0 = disabled)
    move_sl_to_be_after_tp1: bool = False  # After TP1 partial fill, move runner SL to entry price
    leledc_maj_qual: int = 0            # Leledc exhaustion threshold (0 = disabled)
    prior_day_level_buffer: float = 0.0 # Block within N pts of prior-day levels (0 = disabled)
    prior_day_atr_min: float = 0.0    # Block when prior-day ATR(14) < threshold (0 = disabled)
    adr_lookback_days: int = 0        # ADR lookback in trading days (0 = disabled)
    adr_directional_threshold: float = 0.0  # Block entries chasing daily direction (0 = disabled)
    session_start_et: str = "10:00"   # RTH start (Eastern Time)
    session_end_et: str = "15:45"     # Last entry allowed
    session_close_et: str = "16:00"   # Force close all positions


@dataclass
class SafetyConfig:
    """Safety limits and circuit breakers."""
    max_daily_loss: float = 650.0     # Max daily loss in dollars before halt (raised from 600: A1+B1+C2+MES2 worst-case=$619)
    max_position_size: int = 5        # Max contracts per instrument
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

# vWinners (v11.1) -- SHELVED: SM flip exit profitable on IS but loses on OOS.
# No reliable regime detector found (Task 3). Kept for reference; re-enable
# when a robust "let winners run" exit is found.
# MNQ_V11_1 = StrategyConfig(
#     instrument="MNQ",
#     strategy_id="MNQ_V11",
#     sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,
#     sm_threshold=0.15,  # Filter whipsaw entries near SM zero
#     rsi_len=8, rsi_buy=60, rsi_sell=40,
#     cooldown=20, max_loss_pts=50,
#     dollar_per_pt=2.0,
# )

MNQ_V15 = StrategyConfig(
    instrument="MNQ",
    strategy_id="MNQ_V15",
    sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,
    sm_threshold=0.0,  # v15 validated with threshold=0 on TV
    exit_mode="tp_scalp",
    tp_pts=7,             # Upgraded from 5: Sharpe 2.73 vs 2.08, PF 1.36 vs 1.29, IS/OOS STRONG
    trail_activate_pts=7,
    trail_distance_pts=8,
    rsi_len=8, rsi_buy=60, rsi_sell=40,
    cooldown=20, max_loss_pts=40,
    dollar_per_pt=2.0,
    max_strategy_daily_loss=100.0,
    vix_death_zone_min=19.0,
    vix_death_zone_max=22.0,
    leledc_maj_qual=9,  # Block entry on Leledc exhaustion (9+ consecutive directional closes)
    adr_lookback_days=14,             # ADR directional gate: 14-day lookback (STRONG PASS on V15+vScalpC)
    adr_directional_threshold=0.3,    # Block when move_from_open/ADR >= 0.3 in entry direction
    session_end_et="13:00",  # Late-day cutoff: entries after 13:00 ET are net negative
)

MNQ_VSCALPB = StrategyConfig(
    instrument="MNQ",
    strategy_id="MNQ_VSCALPB",
    sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,
    sm_threshold=0.25,  # High-conviction entries only
    exit_mode="tp_scalp",
    tp_pts=3,             # Upgraded from 5: Sharpe 3.29 vs 1.49, PF 1.47 vs 1.19, IS/OOS STRONG
    trail_activate_pts=3,
    trail_distance_pts=8,
    rsi_len=8, rsi_buy=55, rsi_sell=45,  # Tighter bands than vScalpA
    cooldown=20, max_loss_pts=10,  # Upgraded from 15: tighter SL matches tighter TP
    dollar_per_pt=2.0,
    max_strategy_daily_loss=100.0,
    leledc_maj_qual=9,  # Block entry on Leledc exhaustion (9+ consecutive directional closes)
    adr_lookback_days=14,             # ADR directional gate: 14-day lookback
    adr_directional_threshold=0.3,    # Block when move_from_open/ADR >= 0.3 in entry direction
)

# MES v9.4 -- REPLACED by MES_V2 (TP=20 exit). Kept for reference.
# MES_V94 = StrategyConfig(
#     instrument="MES",
#     strategy_id="MES_V94",
#     sm_index=20, sm_flow=12, sm_norm=400, sm_ema=255,
#     rsi_len=10, rsi_buy=55, rsi_sell=45,
#     cooldown=15, max_loss_pts=0,  # Stop loss hurts MES with SM flip exit
#     dollar_per_pt=5.0,
#     commission_per_side=1.25,
#     session_close_et="15:30",
# )

MNQ_VSCALPC = StrategyConfig(
    instrument="MNQ",
    strategy_id="MNQ_VSCALPC",
    sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,
    sm_threshold=0.0,  # Same entries as V15 — runner captures larger moves
    exit_mode="tp_scalp",
    tp_pts=25,  # TP2: runner target
    trail_activate_pts=25,
    trail_distance_pts=8,
    rsi_len=8, rsi_buy=60, rsi_sell=40,
    cooldown=20, max_loss_pts=40,
    dollar_per_pt=2.0,
    entry_qty=2,           # 2 contracts: partial at TP1, runner to TP2
    partial_tp_pts=7,      # TP1: close 1 contract at +7 pts ($14)
    partial_qty=1,         # Close 1 of 2 at TP1
    breakeven_after_bars=45,   # Close stale runners after 45 bars (~45 min)
    move_sl_to_be_after_tp1=True,  # After TP1, move runner SL to entry (risk-free runner)
    max_strategy_daily_loss=200.0,  # One 2-contract SL is $160; $200 allows recovery
    vix_death_zone_min=19.0,   # Same VIX gate as V15 (same entries)
    vix_death_zone_max=22.0,
    leledc_maj_qual=9,  # Block entry on Leledc exhaustion (9+ consecutive directional closes)
    prior_day_atr_min=263.8,   # Block on low-vol days (ATR < p20). Phase 2 sweep (ET dates): IS PF +10%, OOS PF +13%
    adr_lookback_days=14,             # ADR directional gate: 14-day lookback (STRONG PASS)
    adr_directional_threshold=0.3,    # Block when move_from_open/ADR >= 0.3 in entry direction
    session_end_et="13:00",  # Same late-day cutoff as V15
)

MES_V2 = StrategyConfig(
    instrument="MES",
    strategy_id="MES_V2",
    sm_index=20, sm_flow=12, sm_norm=400, sm_ema=255,
    sm_threshold=0.0,  # MES weak entries are profitable
    exit_mode="tp_scalp",
    tp_pts=20,  # $100/contract — replaces SM flip exit
    trail_activate_pts=20,
    trail_distance_pts=8,
    rsi_len=12, rsi_buy=55, rsi_sell=45,  # Slower RSI matches slow MES SM
    cooldown=25, max_loss_pts=35,  # $175/contract — caps v9.4's uncapped losses
    dollar_per_pt=5.0,
    commission_per_side=1.25,
    entry_qty=2,           # 2 contracts: partial at TP1, runner to TP2
    partial_tp_pts=6,      # TP1: close 1 contract at +6 pts ($30) — swept Mar 11: PF +8.2%, Sharpe +28%, TP1 fill 39%→60%
    partial_qty=1,         # Close 1 of 2 at TP1
    breakeven_after_bars=75,   # Close stale trades after 75 bars (~1h15m)
    max_strategy_daily_loss=400.0,    # One 2-contract SL is ~$355; $400 allows recovery
    prior_day_level_buffer=5.0,  # Block within 5 pts of prior-day H/L/VPOC/VAH/VAL
    session_close_et="15:30",  # EOD 15:30 ET validated for MES
)

DEFAULT_CONFIG = EngineConfig(
    strategies=[MNQ_V15, MNQ_VSCALPB, MNQ_VSCALPC, MES_V2],
    safety=SafetyConfig(paper_mode=True),
)
