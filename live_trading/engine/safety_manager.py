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
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from engine.config import EngineConfig, StrategyConfig
from engine.events import Bar, EventBus, TradeRecord

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


@dataclass
class StrategyStatus:
    """Per-strategy safety state."""
    strategy_id: str
    instrument: str
    paused: bool = False
    pause_reason: str = ""
    manual_override: bool = False   # True after manual resume — prevents auto re-pause
    qty_override: Optional[int] = None  # None = use default (1)
    partial_qty_override: Optional[int] = None  # None = use config default
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

        # Per-strategy rolling 5-day SL history: keyed by strategy_id
        # Each value is a list of the last 5 daily SL counts
        self._per_strategy_sl_history: dict[str, list[int]] = {
            (s.strategy_id or s.instrument): [] for s in config.strategies
        }

        # VIX death zone gating
        self._vix_close: float | None = None

        # Gate deduplication: on_bar is called multiple times per bar (direct +
        # per-strategy event bus). Gate updates must only run ONCE per unique bar.
        self._last_gate_bar: dict[str, tuple] = {}

        # Leledc exhaustion gate (per instrument)
        self._leledc_closes: dict[str, list] = {}
        self._leledc_bull_count: dict[str, int] = {}
        self._leledc_bear_count: dict[str, int] = {}
        self._leledc_gate_prev: dict[str, bool] = {}
        self._leledc_gate_current: dict[str, bool] = {}
        self._leledc_thresholds: dict[str, int] = {}  # min maj_qual per instrument
        for s in config.strategies:
            if s.leledc_maj_qual > 0:
                inst = s.instrument
                if inst not in self._leledc_thresholds or s.leledc_maj_qual < self._leledc_thresholds[inst]:
                    self._leledc_thresholds[inst] = s.leledc_maj_qual

        # Prior-day level gate (per instrument)
        self._prior_day_levels: dict[str, dict] = {}   # inst -> {high, low, vpoc, vah, val}
        self._current_rth: dict[str, dict] = {}        # inst -> {date, high, low, closes, volumes}
        self._prior_day_gate_prev: dict[str, bool] = {}
        self._prior_day_gate_current: dict[str, bool] = {}
        self._prior_day_buffers: dict[str, float] = {}
        for s in config.strategies:
            if s.prior_day_level_buffer > 0:
                self._prior_day_buffers[s.instrument] = s.prior_day_level_buffer

    # ------------------------------------------------------------------
    # Event bus handlers (registered externally)
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        """Track heartbeat + update entry gate state (deduplicated).

        Called multiple times per bar (direct from runner + per-strategy event bus).
        Heartbeat always updates; gate logic runs once per unique bar per instrument.
        """
        self._last_bar_time[bar.instrument] = datetime.now(timezone.utc)

        # Gate updates: process each bar ONCE per instrument
        inst = bar.instrument
        bar_key = (inst, bar.timestamp)
        if bar_key == self._last_gate_bar.get(inst):
            return
        self._last_gate_bar[inst] = bar_key

        # Swap prev ← current BEFORE updating current
        self._leledc_gate_prev[inst] = self._leledc_gate_current.get(inst, True)
        self._prior_day_gate_prev[inst] = self._prior_day_gate_current.get(inst, True)

        self._update_leledc(bar)
        self._update_prior_day_tracking(bar)

    # ------------------------------------------------------------------
    # Entry gate update methods (called from on_bar, deduplicated)
    # ------------------------------------------------------------------

    _LELEDC_LOOKBACK = 4

    def _update_leledc(self, bar: Bar) -> None:
        """Incremental Leledc exhaustion gate update.

        Persistence=1: gate is False on the exhaustion bar itself, True when
        the streak breaks. Matches build_leledc_gate(persistence=1) from backtest.
        Counter does NOT reset at daily boundary (matches backtest behavior).
        """
        inst = bar.instrument
        threshold = self._leledc_thresholds.get(inst)
        if threshold is None:
            return

        if inst not in self._leledc_closes:
            self._leledc_closes[inst] = []
            self._leledc_bull_count[inst] = 0
            self._leledc_bear_count[inst] = 0

        closes = self._leledc_closes[inst]
        closes.append(bar.close)
        if len(closes) > 20:
            closes[:] = closes[-20:]

        if len(closes) <= self._LELEDC_LOOKBACK:
            self._leledc_gate_current[inst] = True
            return

        curr = closes[-1]
        prev = closes[-(1 + self._LELEDC_LOOKBACK)]

        self._leledc_bull_count[inst] = (self._leledc_bull_count[inst] + 1) if curr > prev else 0
        self._leledc_bear_count[inst] = (self._leledc_bear_count[inst] + 1) if curr < prev else 0

        exhausted = (self._leledc_bull_count[inst] >= threshold
                     or self._leledc_bear_count[inst] >= threshold)
        self._leledc_gate_current[inst] = not exhausted

    # RTH window: 10:00 ET (600 mins) to 16:00 ET (960 mins)
    _RTH_OPEN_ET = 600
    _RTH_CLOSE_ET = 960

    def _update_prior_day_tracking(self, bar: Bar) -> None:
        """Track RTH bars incrementally and compute proximity gate.

        On calendar date change, finalizes prior-day levels (H/L + volume profile).
        Gate checks proximity of current close to all prior-day levels.
        """
        inst = bar.instrument
        if inst not in self._prior_day_buffers:
            return

        buffer = self._prior_day_buffers[inst]
        bar_et = bar.timestamp.astimezone(_ET)
        et_mins = bar_et.hour * 60 + bar_et.minute
        bar_date = bar_et.date()

        if inst not in self._current_rth:
            self._current_rth[inst] = {"date": None, "high": None, "low": None,
                                        "closes": [], "volumes": []}
        rth = self._current_rth[inst]

        # New calendar date → finalize previous day's levels
        if rth["date"] is not None and bar_date != rth["date"]:
            self._finalize_prior_day(inst)
        rth["date"] = bar_date

        # Collect RTH bars (10:00-16:00 ET)
        if self._RTH_OPEN_ET <= et_mins < self._RTH_CLOSE_ET:
            if rth["high"] is None:
                rth["high"] = bar.high
                rth["low"] = bar.low
            else:
                rth["high"] = max(rth["high"], bar.high)
                rth["low"] = min(rth["low"], bar.low)
            rth["closes"].append(bar.close)
            rth["volumes"].append(bar.volume)

        # Proximity gate vs prior-day levels
        levels = self._prior_day_levels.get(inst)
        if not levels:
            self._prior_day_gate_current[inst] = True
            return

        for key in ("high", "low", "vpoc", "vah", "val"):
            lvl = levels.get(key)
            if lvl is not None and abs(bar.close - lvl) <= buffer:
                self._prior_day_gate_current[inst] = False
                return
        self._prior_day_gate_current[inst] = True

    def _finalize_prior_day(self, inst: str) -> None:
        """Compute prior-day levels from collected RTH data and reset tracking."""
        rth = self._current_rth[inst]
        if rth["high"] is None:
            # No RTH bars collected — keep previous levels
            return

        levels: dict = {"high": rth["high"], "low": rth["low"],
                        "vpoc": None, "vah": None, "val": None}

        # Compute volume profile (VPOC, VAH, VAL)
        if rth["closes"] and rth["volumes"]:
            profile = self._compute_value_area(rth["closes"], rth["volumes"], bin_width=5.0)
            if profile is not None:
                levels["vpoc"], levels["vah"], levels["val"] = profile

        self._prior_day_levels[inst] = levels
        logger.info(
            f"[Safety] Prior-day levels for {inst}: "
            f"H={levels['high']:.2f} L={levels['low']:.2f} "
            f"VPOC={levels.get('vpoc', '?')} VAH={levels.get('vah', '?')} VAL={levels.get('val', '?')}"
        )

        # Reset tracking for new day
        rth["high"] = None
        rth["low"] = None
        rth["closes"] = []
        rth["volumes"] = []

    @staticmethod
    def _compute_value_area(day_closes: list, day_volumes: list, bin_width: float):
        """Compute VPOC, VAH, VAL from a single day's closes and volumes.

        Direct port from sr_prior_day_levels_sweep.py backtest.
        Returns (vpoc_price, vah_price, val_price) or None if no data.
        """
        if not day_closes:
            return None

        total_vol = sum(day_volumes)
        if total_vol <= 0:
            return None

        # Bin prices
        price_min = math.floor(min(day_closes) / bin_width) * bin_width
        price_max = math.ceil(max(day_closes) / bin_width) * bin_width
        if price_min == price_max:
            price_max = price_min + bin_width

        n_bins = int(round((price_max - price_min) / bin_width)) + 1
        bin_volumes = [0.0] * n_bins

        for c, v in zip(day_closes, day_volumes):
            idx = int(round((c - price_min) / bin_width))
            idx = min(max(idx, 0), n_bins - 1)
            bin_volumes[idx] += v

        # VPOC = bin with max volume
        vpoc_idx = max(range(n_bins), key=lambda i: bin_volumes[i])
        vpoc_price = price_min + vpoc_idx * bin_width

        # Value area: expand from VPOC until 70% of volume captured
        va_target = total_vol * 0.70
        va_vol = bin_volumes[vpoc_idx]
        lo_idx = vpoc_idx
        hi_idx = vpoc_idx

        while va_vol < va_target:
            can_go_lo = lo_idx > 0
            can_go_hi = hi_idx < n_bins - 1

            if not can_go_lo and not can_go_hi:
                break

            lo_vol = bin_volumes[lo_idx - 1] if can_go_lo else -1.0
            hi_vol = bin_volumes[hi_idx + 1] if can_go_hi else -1.0

            if lo_vol >= hi_vol:
                lo_idx -= 1
                va_vol += bin_volumes[lo_idx]
            else:
                hi_idx += 1
                va_vol += bin_volumes[hi_idx]

        val_price = price_min + lo_idx * bin_width
        vah_price = price_min + hi_idx * bin_width

        return vpoc_price, vah_price, val_price

    def on_trade_closed(self, trade: TradeRecord) -> None:
        """Update per-strategy and global stats after a trade closes."""
        sid = trade.strategy_id or trade.instrument
        strat = self._strategies.get(sid)

        if not strat and sid != "":
            logger.warning(f"[Safety] Unknown strategy_id in trade: {sid}")

        # Commission from strategy config, scaled by qty (falls back to $0.52/side)
        strat_config = self._strategy_configs.get(sid)
        commission = 2 * trade.qty * (strat_config.commission_per_side if strat_config else 0.52)
        adjusted_pnl = trade.pnl_dollar - commission

        # Per-strategy tracking
        if strat:
            # Only count full closes for trade_count (avoid double-counting partial + final)
            if not trade.is_partial:
                strat.trade_count_today += 1
            strat.daily_pnl += adjusted_pnl
            # Only count actual stop-loss exits toward SL counters
            if trade.exit_reason == "SL":
                strat.sl_count_today += 1

            # Per-strategy daily loss limit (auto-pause)
            strat_max_loss = strat_config.max_strategy_daily_loss if strat_config else 0.0
            if (strat_max_loss > 0 and strat.daily_pnl <= -strat_max_loss
                    and not strat.paused and not strat.manual_override):
                strat.paused = True
                strat.pause_reason = (
                    f"Auto: daily loss ${abs(strat.daily_pnl):.2f} "
                    f"exceeds limit ${strat_max_loss:.2f}"
                )
                logger.warning(
                    f"[Safety] Strategy {sid} auto-paused: daily loss "
                    f"${abs(strat.daily_pnl):.2f} exceeds limit ${strat_max_loss:.2f}"
                )

        # Global tracking
        self._global_daily_pnl += adjusted_pnl
        if not trade.is_partial:
            self._global_trade_count += 1

        # Consecutive losses: skip partial trades entirely (TP1 is always a winner
        # but resetting prematurely could mask a losing streak; don't count as loss either)
        if not trade.is_partial:
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

        # Auto drawdown rules — only V11 SL exits trigger rules.
        # NOTE: V11 (vWinners) is shelved as of Feb 2026 OOS deep dive.
        # These rules are dormant while MNQ_V11 is not in the active config.
        # TODO: Design new drawdown rules for vScalpA+vScalpB+MES_V2 portfolio
        # once paper trading data is available.
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

        # VIX death zone gate (per-strategy, based on config bounds)
        if strategy_id and self._vix_close is not None:
            cfg = self._strategy_configs.get(strategy_id)
            if cfg and cfg.vix_death_zone_min > 0:
                if cfg.vix_death_zone_min <= self._vix_close <= cfg.vix_death_zone_max:
                    strat = self._strategies.get(strategy_id)
                    if strat and not strat.manual_override:
                        return False, (
                            f"VIX death zone: VIX {self._vix_close:.1f} "
                            f"in [{cfg.vix_death_zone_min}-{cfg.vix_death_zone_max}]"
                        )

        # Leledc exhaustion gate
        if strategy_id:
            cfg = self._strategy_configs.get(strategy_id)
            if cfg and cfg.leledc_maj_qual > 0:
                strat = self._strategies.get(strategy_id)
                if strat and not strat.manual_override:
                    if not self._leledc_gate_prev.get(strat.instrument, True):
                        return False, (
                            f"Leledc exhaustion: bull={self._leledc_bull_count.get(strat.instrument, 0)} "
                            f"bear={self._leledc_bear_count.get(strat.instrument, 0)}"
                        )

        # Prior-day level proximity gate
        if strategy_id:
            cfg = self._strategy_configs.get(strategy_id)
            if cfg and cfg.prior_day_level_buffer > 0:
                strat = self._strategies.get(strategy_id)
                if strat and not strat.manual_override:
                    if not self._prior_day_gate_prev.get(strat.instrument, True):
                        levels = self._prior_day_levels.get(strat.instrument, {})
                        return False, (
                            f"Near prior-day level (buf={cfg.prior_day_level_buffer}): "
                            f"H={levels.get('high', '?')} L={levels.get('low', '?')} "
                            f"VPOC={levels.get('vpoc', '?')}"
                        )

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

    def set_vix_close(self, vix_close: float | None) -> None:
        """Update prior-day VIX close for death zone gating."""
        self._vix_close = vix_close
        logger.info(f"[Safety] VIX close set to: {vix_close}")
        self._broadcast_status()

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
            strat.partial_qty_override = None
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

        # 2b. Append each strategy's SL count to per-strategy history, then trim
        for sid, strat in self._strategies.items():
            if sid not in self._per_strategy_sl_history:
                self._per_strategy_sl_history[sid] = []
            self._per_strategy_sl_history[sid].append(strat.sl_count_today)
            self._per_strategy_sl_history[sid] = self._per_strategy_sl_history[sid][-5:]

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
                strat.partial_qty_override = None
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
            strat_config = self._strategy_configs.get(sid)
            # Per-strategy rolling 5-day SL: last 4 days of history + today
            hist = self._per_strategy_sl_history.get(sid, [])
            sl_rolling_5d = sum(hist[-4:]) + strat.sl_count_today
            # VIX death zone: gated if VIX in range AND no manual override
            cfg = self._strategy_configs.get(sid)
            vix_gated = (
                self._vix_close is not None
                and cfg is not None
                and cfg.vix_death_zone_min > 0
                and cfg.vix_death_zone_min <= self._vix_close <= cfg.vix_death_zone_max
                and not strat.manual_override
            )

            strategies[sid] = {
                "strategy_id": strat.strategy_id,
                "instrument": strat.instrument,
                "paused": strat.paused,
                "pause_reason": strat.pause_reason,
                "manual_override": strat.manual_override,
                "qty_override": strat.qty_override,
                "partial_qty_override": strat.partial_qty_override,
                "config_entry_qty": strat_config.entry_qty if strat_config else 1,
                "config_partial_qty": strat_config.partial_qty if strat_config else 1,
                "config_partial_tp_pts": strat_config.partial_tp_pts if strat_config else 0,
                "sl_count_today": strat.sl_count_today,
                "sl_rolling_5d": sl_rolling_5d,
                "trade_count_today": strat.trade_count_today,
                "daily_pnl": round(strat.daily_pnl, 2),
                "vix_gated": vix_gated,
                "leledc_gated": (
                    cfg is not None
                    and cfg.leledc_maj_qual > 0
                    and not self._leledc_gate_prev.get(strat.instrument, True)
                    and not strat.manual_override
                ),
                "prior_day_gated": (
                    cfg is not None
                    and cfg.prior_day_level_buffer > 0
                    and not self._prior_day_gate_prev.get(strat.instrument, True)
                    and not strat.manual_override
                ),
            }

        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "vix_close": self._vix_close,
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
            "prior_day_levels": {
                inst: {k: round(v, 2) if v is not None else None for k, v in lvls.items()}
                for inst, lvls in self._prior_day_levels.items()
            },
            "strategies": strategies,
        }

    def _broadcast_status(self) -> None:
        """Emit status_change on EventBus so WS clients update immediately."""
        if self._event_bus:
            self._event_bus.emit("status_change", self.get_status())
