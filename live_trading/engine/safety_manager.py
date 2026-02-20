"""
SafetyManager — per-strategy drawdown protection and circuit breakers.

Replaces the inline SafetyManager from runner.py with:
- Per-strategy pause/resume with manual override
- Per-strategy qty overrides (for drawdown sizing)
- Auto drawdown rules (Rule 1: 1st SL, Rule 2: rolling 5-day)
- Per-instrument heartbeat tracking
- Commission-adjusted P&L
- EventBus status_change broadcasting on every mutation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from engine.config import EngineConfig, StrategyConfig
from engine.events import Bar, EventBus, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class StrategyStatus:
    """Per-strategy safety state."""
    strategy_id: str
    instrument: str
    paused: bool = False
    pause_reason: str = ""
    manual_override: bool = False   # True after manual resume — prevents auto re-pause
    qty_override: Optional[int] = None  # None = use default (1)
    sl_count_today: int = 0
    trade_count_today: int = 0
    daily_pnl: float = 0.0


class SafetyManager:
    """Enforces safety limits, circuit breakers, and drawdown rules.

    Tracks per-strategy state, auto drawdown rules, and global circuit
    breakers. Emits status_change events on the EventBus after every
    state mutation so dashboard clients update immediately.
    """

    def __init__(self, config: EngineConfig, event_bus: Optional[EventBus] = None):
        self._config = config.safety
        self._event_bus = event_bus

        # Per-strategy state
        self._strategies: dict[str, StrategyStatus] = {}
        self._strategy_configs: dict[str, StrategyConfig] = {}
        for s in config.strategies:
            sid = s.strategy_id or s.instrument
            self._strategies[sid] = StrategyStatus(
                strategy_id=sid,
                instrument=s.instrument,
            )
            self._strategy_configs[sid] = s

        # Global state
        self._halted: bool = False
        self._halt_reason: str = ""
        self._consecutive_losses: int = 0
        self._global_daily_pnl: float = 0.0
        self._global_trade_count: int = 0

        # Per-instrument heartbeat (wall clock of last bar processed)
        self._last_bar_time: dict[str, datetime] = {}

        # Drawdown auto-rules
        self._drawdown_enabled: bool = True
        self._extended_pause: bool = False
        self._extended_pause_reason: str = ""

        # Rolling 5-day SL history: list of {date, sl_count} dicts
        # Appended on daily reset (before clearing daily counters)
        self._daily_sl_history: list[dict] = []

    # ------------------------------------------------------------------
    # Event bus handlers (registered externally)
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        """Track per-instrument heartbeat."""
        self._last_bar_time[bar.instrument] = datetime.now(timezone.utc)

    def on_trade_closed(self, trade: TradeRecord) -> None:
        """Update per-strategy and global stats after a trade closes."""
        sid = trade.strategy_id or trade.instrument
        strat = self._strategies.get(sid)

        if not strat and sid != "":
            logger.warning(f"[Safety] Unknown strategy_id in trade: {sid}")

        # Commission from strategy config (falls back to $0.52/side)
        strat_config = self._strategy_configs.get(sid)
        commission = 2 * (strat_config.commission_per_side if strat_config else 0.52)
        adjusted_pnl = trade.pnl_dollar - commission

        # Per-strategy tracking
        if strat:
            strat.trade_count_today += 1
            strat.daily_pnl += adjusted_pnl
            # Only count actual stop-loss exits toward SL counters
            if trade.exit_reason == "SL":
                strat.sl_count_today += 1

        # Global tracking
        self._global_daily_pnl += adjusted_pnl
        self._global_trade_count += 1

        if adjusted_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        logger.info(
            f"[Safety] Trade: {sid} raw=${trade.pnl_dollar:+.2f} "
            f"comm=${commission:.2f} adj=${adjusted_pnl:+.2f} | "
            f"Daily=${self._global_daily_pnl:+.2f} ConsecLoss={self._consecutive_losses}"
        )

        # Check global circuit breakers
        if self._global_daily_pnl <= -self._config.max_daily_loss:
            self._halted = True
            self._halt_reason = (
                f"Daily loss limit hit: ${self._global_daily_pnl:.2f} "
                f"(limit: -${self._config.max_daily_loss:.2f})"
            )
            logger.warning(f"[Safety] HALTED: {self._halt_reason}")

        if self._consecutive_losses >= self._config.max_consecutive_losses:
            self._halted = True
            self._halt_reason = (
                f"Consecutive loss limit hit: {self._consecutive_losses} "
                f"(limit: {self._config.max_consecutive_losses})"
            )
            logger.warning(f"[Safety] HALTED: {self._halt_reason}")

        # Auto drawdown rules — only V11 SL exits trigger rules
        if (self._drawdown_enabled and trade.exit_reason == "SL"
                and strat and strat.strategy_id == "MNQ_V11"):
            self._apply_drawdown_rules(strat)

        self._broadcast_status()

    # ------------------------------------------------------------------
    # Drawdown rules
    # ------------------------------------------------------------------

    def _rolling_sl_5d(self) -> int:
        """Rolling 5-day V11 SL count (history + today)."""
        hist = sum(d.get("sl_count", 0) for d in self._daily_sl_history[-4:])
        today = sum(s.sl_count_today for s in self._strategies.values()
                    if s.strategy_id == "MNQ_V11")
        return hist + today

    def _apply_drawdown_rules(self, strat: StrategyStatus) -> None:
        """Apply auto drawdown rules after a V11 SL exit."""
        # Rule 1: 1st SL today on V11 -> pause V11, size up V15 to 2.
        # Timing guarantee: this fires on V11's SL close, so V11 position is
        # already 0. V15 at qty=2 keeps total exposure <= max_position_size(2).
        if strat.strategy_id == "MNQ_V11" and strat.sl_count_today == 1:
            v11 = self._strategies.get("MNQ_V11")
            v15 = self._strategies.get("MNQ_V15")

            if v11 and not v11.manual_override and not v11.paused:
                v11.paused = True
                v11.pause_reason = "Auto: 1st SL today on V11"
                logger.info("[Safety] Rule 1: Paused MNQ_V11 (1st SL)")

            if v15 and not v15.manual_override and v15.qty_override is None:
                v15.qty_override = 2
                logger.info("[Safety] Rule 1: MNQ_V15 qty -> 2")

        # Rule 2: Rolling 5-day V11 SL count >= 4 -> extended pause (halt all)
        rolling_sl = self._rolling_sl_5d()

        if rolling_sl >= 4 and not self._extended_pause:
            self._extended_pause = True
            self._extended_pause_reason = (
                f"Rolling 5-day SL count: {rolling_sl} (>= 4)"
            )
            # Pause all strategies
            for s in self._strategies.values():
                if not s.manual_override:
                    s.paused = True
                    s.pause_reason = self._extended_pause_reason
            logger.warning(f"[Safety] Rule 2: EXTENDED PAUSE — {self._extended_pause_reason}")

    # ------------------------------------------------------------------
    # Trade gating
    # ------------------------------------------------------------------

    def check_can_trade(self, instrument: str, qty: int, strategy_id: str = "") -> tuple[bool, str]:
        """Check if a trade is allowed. Returns (ok, reason)."""
        if self._halted:
            return False, f"Engine halted: {self._halt_reason}"

        if self._extended_pause:
            return False, f"Extended pause: {self._extended_pause_reason}"

        # Per-strategy pause check
        if strategy_id:
            strat = self._strategies.get(strategy_id)
            if strat and strat.paused:
                return False, f"Strategy {strategy_id} paused: {strat.pause_reason}"

        # Position size check
        if qty > self._config.max_position_size:
            return False, (
                f"Position size {qty} exceeds limit {self._config.max_position_size}"
            )

        return True, ""

    def get_qty_override(self, strategy_id: str) -> Optional[int]:
        """Return qty override for a strategy, or None for default."""
        strat = self._strategies.get(strategy_id)
        if strat:
            return strat.qty_override
        return None

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def check_heartbeat(self, instrument: Optional[str] = None) -> tuple[bool, float]:
        """Check data feed health. Returns (healthy, max_seconds_since_last_bar).

        If instrument specified, check that instrument only. Otherwise check all.
        """
        if not self._last_bar_time:
            return True, 0.0

        now = datetime.now(timezone.utc)
        max_elapsed = 0.0

        targets = (
            {instrument: self._last_bar_time[instrument]}
            if instrument and instrument in self._last_bar_time
            else self._last_bar_time
        )

        for inst, last_time in targets.items():
            if last_time.tzinfo is None:
                elapsed = (now.replace(tzinfo=None) - last_time).total_seconds()
            else:
                elapsed = (now - last_time).total_seconds()
            max_elapsed = max(max_elapsed, elapsed)

        healthy = max_elapsed < self._config.heartbeat_timeout_sec
        return healthy, max_elapsed

    # ------------------------------------------------------------------
    # Manual overrides (called from API)
    # ------------------------------------------------------------------

    def set_strategy_paused(self, strategy_id: str, paused: bool) -> tuple[bool, str]:
        """Pause or resume a strategy. Returns (ok, message)."""
        strat = self._strategies.get(strategy_id)
        if not strat:
            return False, f"Unknown strategy: {strategy_id}"

        if paused:
            strat.paused = True
            strat.pause_reason = "Manual pause"
            strat.manual_override = False  # Manual pause doesn't set override
            logger.info(f"[Safety] Manual PAUSE: {strategy_id}")
        else:
            strat.paused = False
            strat.pause_reason = ""
            strat.manual_override = True  # Prevent auto-rules from re-pausing
            logger.info(f"[Safety] Manual RESUME: {strategy_id} (manual_override=True)")

        self._broadcast_status()
        return True, f"{strategy_id} {'paused' if paused else 'resumed'}"

    def set_strategy_qty(self, strategy_id: str, qty: int) -> tuple[bool, str]:
        """Override contract size for a strategy. Returns (ok, message)."""
        strat = self._strategies.get(strategy_id)
        if not strat:
            return False, f"Unknown strategy: {strategy_id}"

        if qty < 1 or qty > self._config.max_position_size:
            return False, f"Invalid qty: {qty} (must be 1-{self._config.max_position_size})"

        strat.qty_override = qty
        logger.info(f"[Safety] Manual QTY: {strategy_id} -> {qty}")
        self._broadcast_status()
        return True, f"{strategy_id} qty set to {qty}"

    def set_drawdown_enabled(self, enabled: bool) -> tuple[bool, str]:
        """Toggle auto drawdown rules. Returns (ok, message)."""
        self._drawdown_enabled = enabled
        logger.info(f"[Safety] Drawdown rules {'ENABLED' if enabled else 'DISABLED'}")
        self._broadcast_status()
        return True, f"Drawdown rules {'enabled' if enabled else 'disabled'}"

    def force_resume_all(self) -> tuple[bool, str]:
        """Clear all pauses, halt, extended pause, manual overrides."""
        self._halted = False
        self._halt_reason = ""
        self._extended_pause = False
        self._extended_pause_reason = ""
        for strat in self._strategies.values():
            strat.paused = False
            strat.pause_reason = ""
            strat.manual_override = False
            strat.qty_override = None
        logger.info("[Safety] FORCE RESUME ALL — all pauses/halts cleared")
        self._broadcast_status()
        return True, "All pauses and halts cleared"

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily counters. Called at start of new trading day.

        Order matters:
        1. Compute rolling count BEFORE appending (avoids double-counting today)
        2. Append today's V11 SL to history
        3. Auto-clear extended pause if rolling dropped below 4
        4. Reset per-strategy counters (sees correct _extended_pause state)
        5. Reset global counters
        """
        # 1. Compute rolling BEFORE appending (history = past days, sl_count_today = today)
        rolling = self._rolling_sl_5d()

        # 2. Append today's V11 SL to history, then trim.
        #    Always append (even on zero-trade days) so old SL entries age out
        #    and extended pause can auto-clear.
        today_sl = sum(s.sl_count_today for s in self._strategies.values()
                       if s.strategy_id == "MNQ_V11")
        self._daily_sl_history.append({
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "sl_count": today_sl,
        })
        self._daily_sl_history = self._daily_sl_history[-5:]

        # 3. Auto-clear extended pause if rolling dropped below 4
        if self._extended_pause and rolling < 4:
            self._extended_pause = False
            self._extended_pause_reason = ""
            logger.info(f"[Safety] Extended pause auto-cleared (rolling 5d SL: {rolling})")

        # 4. Reset per-strategy counters (now sees _extended_pause correctly)
        for strat in self._strategies.values():
            strat.sl_count_today = 0
            strat.trade_count_today = 0
            strat.daily_pnl = 0.0
            if not self._extended_pause:
                strat.paused = False
                strat.pause_reason = ""
                strat.qty_override = None
                strat.manual_override = False

        # 5. Reset global counters
        self._global_daily_pnl = 0.0
        self._consecutive_losses = 0
        self._global_trade_count = 0
        self._halted = False
        self._halt_reason = ""

        logger.info("[Safety] Daily counters reset")
        self._broadcast_status()

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def trade_count_today(self) -> int:
        return self._global_trade_count

    @property
    def daily_pnl(self) -> float:
        return self._global_daily_pnl

    def get_status(self) -> dict:
        """Return full safety state for embedding in /api/status."""
        healthy, elapsed = self.check_heartbeat()

        # Rolling 5-day V11 SL count (history + today)
        rolling_sl_5d = self._rolling_sl_5d()

        # Determine drawdown mode
        if self._extended_pause:
            drawdown_mode = "EXTENDED_PAUSE"
        elif any(s.paused for s in self._strategies.values()):
            drawdown_mode = "REDUCED"
        else:
            drawdown_mode = "NORMAL"

        strategies = {}
        for sid, strat in self._strategies.items():
            strategies[sid] = {
                "strategy_id": strat.strategy_id,
                "instrument": strat.instrument,
                "paused": strat.paused,
                "pause_reason": strat.pause_reason,
                "manual_override": strat.manual_override,
                "qty_override": strat.qty_override,
                "sl_count_today": strat.sl_count_today,
                "trade_count_today": strat.trade_count_today,
                "daily_pnl": round(strat.daily_pnl, 2),
            }

        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": round(self._global_daily_pnl, 2),
            "consecutive_losses": self._consecutive_losses,
            "trade_count_today": self._global_trade_count,
            "data_feed_healthy": healthy,
            "seconds_since_last_bar": round(elapsed, 1),
            "drawdown_enabled": self._drawdown_enabled,
            "drawdown_mode": drawdown_mode,
            "extended_pause": self._extended_pause,
            "extended_pause_reason": self._extended_pause_reason,
            "rolling_sl_5d": rolling_sl_5d,
            "strategies": strategies,
        }

    def _broadcast_status(self) -> None:
        """Emit status_change on EventBus so WS clients update immediately."""
        if self._event_bus:
            self._event_bus.emit("status_change", self.get_status())
