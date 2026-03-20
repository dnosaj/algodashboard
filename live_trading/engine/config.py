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
    entry_signal: str = "sm_rsi"  # "sm_rsi" (default) or "rsi_trendline"
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
    prior_day_level_keys: tuple = ("high", "low", "vpoc", "vah", "val")  # Which levels to check
    prior_day_atr_min: float = 0.0    # Block when prior-day ATR(14) < threshold (0 = disabled)
    adr_lookback_days: int = 0        # ADR lookback in trading days (0 = disabled)
    adr_directional_threshold: float = 0.0  # Block entries chasing daily direction (0 = disabled)
    structure_exit_type: str = ""         # "" (disabled) or "pivot"
    structure_exit_lookback: int = 0      # Left bars for pivot detection
    structure_exit_pivot_right: int = 2   # Confirmation bars to right (pivot only)
    structure_exit_buffer_pts: float = 0.0  # Exit N pts before swing level
    session_start_et: str = "10:00"   # RTH start (Eastern Time)
    session_end_et: str = "15:45"     # Last entry allowed
    session_close_et: str = "16:00"   # Force close all positions


@dataclass
class SafetyConfig:
    """Safety limits and circuit breakers."""
    max_daily_loss: float = 2000.0    # Max daily loss before halt. 4 correlated SM strategies can lose $471 in one bar.
    max_position_size: int = 12       # Max contracts per instrument (10 MNQ possible: A(2)+B(1)+C(3)+C-SAT(2)+RSI_TL(2))
    max_consecutive_losses: int = 15  # 4 SM strategies fire together = 4 losses per bad bar. 15 allows 3 bad bars + individual losses.
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
    data_feed: str = "auto"      # "auto", "databento", or "tastytrade"
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
    sm_index=12, sm_flow=12, sm_norm=200, sm_ema=80,  # Robust: index 10→12, EMA 100→80
    sm_threshold=0.0,  # v15 validated with threshold=0 on TV
    exit_mode="tp_scalp",
    tp_pts=7,             # Upgraded from 5: Sharpe 2.73 vs 2.08, PF 1.36 vs 1.29, IS/OOS STRONG
    trail_activate_pts=7,
    trail_distance_pts=8,
    rsi_len=8, rsi_buy=60, rsi_sell=40,
    cooldown=20, max_loss_pts=40,
    dollar_per_pt=2.0,
    entry_qty=2,          # Robust: 2 contracts
    max_strategy_daily_loss=400.0,  # 2ct × SL=40 = $164. Allows 2 full SLs + recovery.
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
    # Gates REMOVED Mar 14: vScalpB is filter-resistant (SM_T=0.25 already filters).
    # Leledc+ADR gates cost $483/yr (31% of P&L) for only +0.6pp WR. Blocked 109 winners vs 43 losers.
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
    sm_index=12, sm_flow=12, sm_norm=200, sm_ema=80,  # Robust: index 10→12, EMA 100→80
    sm_threshold=0.0,  # Same entries as V15 — runner captures larger moves
    exit_mode="tp_scalp",
    tp_pts=30,  # Robust: fixed TP2=30 (validated 3-year, +$9,754, better P&L than structure exit)
    trail_activate_pts=30,
    trail_distance_pts=8,
    rsi_len=8, rsi_buy=60, rsi_sell=40,
    cooldown=20, max_loss_pts=30,  # Robust: SL=30 (was 40)
    dollar_per_pt=2.0,
    entry_qty=3,           # Robust: 3 contracts (was 2): scalp 2 at TP1, 1 runner to TP2
    partial_tp_pts=10,     # Robust: TP1=10 (was 7): close 2 contracts at +10 pts ($40)
    partial_qty=2,         # Robust: close 2 of 3 at TP1 (was 1)
    breakeven_after_bars=45,   # Close stale runners after 45 bars (~45 min)
    move_sl_to_be_after_tp1=True,  # After TP1, move runner SL to entry (risk-free runner)
    max_strategy_daily_loss=400.0,  # Robust: 3 contracts × SL=30 × $2 = $180 worst case (was 200)
    vix_death_zone_min=19.0,   # Same VIX gate as V15 (same entries)
    vix_death_zone_max=22.0,
    leledc_maj_qual=9,  # Block entry on Leledc exhaustion (9+ consecutive directional closes)
    prior_day_atr_min=263.8,   # Block on low-vol days (ATR < p20). Phase 2 sweep (ET dates): IS PF +10%, OOS PF +13%
    adr_lookback_days=14,             # ADR directional gate: 14-day lookback (STRONG PASS)
    adr_directional_threshold=0.3,    # Block when move_from_open/ADR >= 0.3 in entry direction
    session_end_et="13:00",  # Same late-day cutoff as V15
    # Structure exit REMOVED for robust config — fixed TP2=30 produces +$2,091 more P&L over 3 years
    # with comparable risk-adjusted returns. Structure exit (LB=50/PR=2/Buf=2) was only validated on
    # old SM(10/12/200/100). Best structure params for robust SM would be LB=30/PR=3/Buf=1 but still
    # underperform fixed TP2. Satellite (VSCALPC_SAT) keeps structure exit with original SM.
)

# vScalpC Satellite — exact copy of pre-robust vScalpC config (0 losing years across 3 years)
MNQ_VSCALPC_SAT = StrategyConfig(
    instrument="MNQ",
    strategy_id="MNQ_VSCALPC_SAT",
    sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,
    sm_threshold=0.0,
    exit_mode="tp_scalp",
    tp_pts=60,  # Crash-safety cap: resting OCO at 60pts. Structure monitor exits before this.
    trail_activate_pts=60,
    trail_distance_pts=8,
    rsi_len=8, rsi_buy=60, rsi_sell=40,
    cooldown=20, max_loss_pts=40,
    dollar_per_pt=2.0,
    entry_qty=2,           # 2 contracts: partial at TP1, runner to TP2
    partial_tp_pts=7,      # TP1: close 1 contract at +7 pts ($14)
    partial_qty=1,         # Close 1 of 2 at TP1
    breakeven_after_bars=45,   # Close stale runners after 45 bars (~45 min)
    move_sl_to_be_after_tp1=True,  # After TP1, move runner SL to entry (risk-free runner)
    max_strategy_daily_loss=400.0,  # 2ct × SL=40 = $164. Allows 2 full SLs + recovery.
    vix_death_zone_min=19.0,
    vix_death_zone_max=22.0,
    leledc_maj_qual=9,
    prior_day_atr_min=263.8,
    adr_lookback_days=14,
    adr_directional_threshold=0.3,
    session_end_et="13:00",
    structure_exit_type="pivot",
    structure_exit_lookback=50,
    structure_exit_pivot_right=2,
    structure_exit_buffer_pts=2.0,
)

MES_V2 = StrategyConfig(
    instrument="MES",
    strategy_id="MES_V2",
    sm_index=20, sm_flow=14, sm_norm=400, sm_ema=300,  # Robust: flow 12→14, EMA 255→300
    sm_threshold=0.25,  # Robust: 0.0→0.25 (high-conviction entries)
    exit_mode="tp_scalp",
    tp_pts=20,  # $100/contract — replaces SM flip exit
    trail_activate_pts=20,
    trail_distance_pts=8,
    rsi_len=12, rsi_buy=55, rsi_sell=45,  # Slower RSI matches slow MES SM
    cooldown=25, max_loss_pts=35,  # $175/contract — caps v9.4's uncapped losses
    dollar_per_pt=5.0,
    commission_per_side=1.25,
    entry_qty=2,           # 2 contracts: partial at TP1, runner to TP2
    partial_tp_pts=8,      # Robust: TP1=8 (was 6): close 1 contract at +8 pts ($40)
    partial_qty=1,         # Close 1 of 2 at TP1
    breakeven_after_bars=75,   # Close stale trades after 75 bars (~1h15m)
    max_strategy_daily_loss=800.0,    # 2ct × SL=35 × $5 = $355. Allows 2 full SLs + recovery.
    prior_day_level_buffer=5.0,  # Block within 5 pts of prior-day levels
    prior_day_level_keys=("vpoc", "val"),  # VPOC+VAL only — H/L and VAH remove profitable breakouts (Mar 13 breakdown)
    session_end_et="14:15",    # Last entry 14:15 ET — entries after 14:00 are net negative (PF 0.78-0.87). Mar 13 sweep.
    session_close_et="16:00",  # No forced close — TP/SL/BE_TIME resolve all trades. 15:30 was v9.4 legacy, never triggered in v2 backtest, killed a live winner Mar 13.
)

MNQ_RSI_TL = StrategyConfig(
    instrument="MNQ",
    strategy_id="MNQ_RSI_TL",
    entry_signal="rsi_trendline",
    sm_index=10, sm_flow=12, sm_norm=200, sm_ema=100,  # unused but required defaults
    sm_threshold=0.0,
    exit_mode="tp_scalp",
    tp_pts=25,                      # Robust: TP2=25 (was 20)
    trail_activate_pts=25,
    trail_distance_pts=8,
    rsi_len=10,                     # Robust: RSI(10) (was 8) on 1-min closes
    rsi_buy=60, rsi_sell=40,        # unused for trendline entry but present
    cooldown=30,
    max_loss_pts=35,                # Robust: SL=35 (was 40)
    dollar_per_pt=2.0,
    entry_qty=2,
    partial_tp_pts=7,               # TP1 scalp
    partial_qty=1,
    move_sl_to_be_after_tp1=True,
    max_strategy_daily_loss=350.0,   # 2ct × SL=35 = $144. Allows 2 full SLs + recovery.
    session_start_et="09:30",       # Match backtest (9:30 AM, not 10:00 default)
    session_end_et="13:00",
    session_close_et="16:00",
)

DEFAULT_CONFIG = EngineConfig(
    strategies=[
        MNQ_V15, MNQ_VSCALPB, MNQ_VSCALPC, MNQ_VSCALPC_SAT, MES_V2,
        MNQ_RSI_TL,  # RSI trendline breakout — paper trading
    ],
    safety=SafetyConfig(paper_mode=True),
)
