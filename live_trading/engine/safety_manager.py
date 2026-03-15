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

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
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

        # Portfolio context — for loss day messaging (observation only)
        self._weekly_pnl: float = 0.0
        self._monthly_pnl: float = 0.0
        self._loss_days_this_month: int = 0
        self._equity_peak: float = 0.0
        self._cumulative_pnl: float = 0.0
        self._today_date: date | None = None

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
        self._prior_day_level_keys: dict[str, tuple] = {}
        for s in config.strategies:
            if s.prior_day_level_buffer > 0:
                self._prior_day_buffers[s.instrument] = s.prior_day_level_buffer
                self._prior_day_level_keys[s.instrument] = s.prior_day_level_keys

        # Prior-day ATR gate (per instrument)
        # Tracks daily ranges across ALL bars (not just RTH), computes Wilder ATR(14).
        # Gate blocks entries when prior-day ATR < threshold.
        self._atr_daily_tracking: dict[str, dict] = {}   # inst -> {date, high, low}
        self._atr_daily_ranges: dict[str, list[float]] = {}  # inst -> list of daily ranges (last 20)
        self._prior_day_atr_value: dict[str, float | None] = {}  # inst -> current ATR value
        self._prior_day_atr_gate_prev: dict[str, bool] = {}
        self._prior_day_atr_gate_current: dict[str, bool] = {}
        self._prior_day_atr_thresholds: dict[str, float] = {}  # inst -> min threshold
        self._ATR_PERIOD = 14
        for s in config.strategies:
            if s.prior_day_atr_min > 0:
                inst = s.instrument
                # Use lowest threshold if multiple strategies on same instrument
                if inst not in self._prior_day_atr_thresholds or s.prior_day_atr_min < self._prior_day_atr_thresholds[inst]:
                    self._prior_day_atr_thresholds[inst] = s.prior_day_atr_min

        # ADR directional gate (per instrument)
        # Tracks RTH session (open/high/low) + rolling ADR (simple mean of prior N daily ranges).
        # Stores move_from_open / ADR ratio for directional gating in check_can_trade.
        self._adr_rth_session: dict[str, dict] = {}        # inst -> {date, open, high, low, close}
        self._adr_completed_ranges: dict[str, list[float]] = {}  # inst -> list of completed daily ranges
        self._adr_value: dict[str, float | None] = {}      # inst -> current ADR value
        self._adr_dir_ratio_prev: dict[str, float] = {}    # inst -> ratio at prev bar (used by check_can_trade)
        self._adr_dir_ratio_current: dict[str, float] = {} # inst -> ratio at current bar
        self._adr_lookback: dict[str, int] = {}             # inst -> lookback days
        for s in config.strategies:
            if s.adr_lookback_days > 0 and s.adr_directional_threshold > 0:
                inst = s.instrument
                if inst not in self._adr_lookback:
                    self._adr_lookback[inst] = s.adr_lookback_days

        # VWAP accumulation (RTH only, for gate_state_snapshots)
        self._vwap_num: dict[str, float] = {}   # inst -> sum(typical_price * volume)
        self._vwap_den: dict[str, float] = {}   # inst -> sum(volume)

        # Opening range / Initial Balance (10:00-10:30 ET, first 30 min RTH)
        self._or_high: dict[str, float | None] = {}
        self._or_low: dict[str, float | None] = {}
        self._or_finalized: dict[str, bool] = {}
        self._OR_END_ET = 630  # 10:30 ET in minutes from midnight

        # ------------------------------------------------------------------
        # ICT dashboard levels — OBSERVATION ONLY (never enters check_can_trade)
        # ------------------------------------------------------------------

        # Weekly VPOC/VAL: accumulate RTH closes+volumes Mon-Fri, compute on Monday
        self._weekly_rth_closes: dict[str, list[float]] = {}
        self._weekly_rth_volumes: dict[str, list[float]] = {}
        self._weekly_vpoc: dict[str, float | None] = {}
        self._weekly_val: dict[str, float | None] = {}
        self._weekly_vpoc_strength: dict[str, float] = {}  # max_bin_vol / total_vol
        # Bin width per instrument for weekly volume profile
        self._weekly_bin_width: dict[str, float] = {}
        for s in config.strategies:
            inst = s.instrument
            if inst not in self._weekly_bin_width:
                self._weekly_bin_width[inst] = 2.0 if "MNQ" in inst else 5.0

        # Developing Daily VPOC — intraday volume profile (observation only)
        self._daily_vpoc_closes: dict[str, list[float]] = {}
        self._daily_vpoc_volumes: dict[str, list[float]] = {}
        self._developing_vpoc: dict[str, float | None] = {}
        self._daily_vpoc_strength: dict[str, float] = {}  # VCR: max_bin_vol / total_vol
        self._dvpoc_stability: dict[str, int] = {}  # bars since last dPOC shift

        # Order Block tracking (UAlgo 3-bar engulfing pattern on 5-MIN bars)
        # OBs persist until mitigated — no daily/weekly reset
        # Detection runs on 5-min resampled bars (matching forensics validation).
        # Mitigation runs on every 1-min bar (more responsive).
        self._active_obs: dict[str, list[dict]] = {}  # per instrument, active OB zones
        self._ob_5min_history: dict[str, deque] = {}   # last 3 completed 5-min bars per instrument
        self._ob_5min_accum: dict[str, dict] = {}      # accumulating current 5-min bar per instrument
        self._ob_5min_bar_count: dict[str, int] = {}   # 1-min bars accumulated in current 5-min bar

        # Load gate seed from pre-computed file (if available)
        self._load_gate_seed()

    # ------------------------------------------------------------------
    # Gate seed loading (pre-computed from historical data)
    # ------------------------------------------------------------------

    _SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "gate_seed.json"
    _GATE_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "gate_state.json"

    def _load_gate_seed(self) -> None:
        """Load gate state from persisted live state or pre-computed seed.

        Priority: gate_state.json (persisted from previous engine run) >
                  gate_seed.json (pre-computed from databento CSVs) >
                  fail-open (warm up from live data).

        After startup, on_bar() accumulates new data naturally. State is
        auto-persisted at each daily reset so the next startup is fresh.
        """
        if self._GATE_STATE_PATH.exists():
            path = self._GATE_STATE_PATH
            source = "persisted state"
        elif self._SEED_PATH.exists():
            path = self._SEED_PATH
            source = "pre-computed seed"
        else:
            logger.info("[Safety] No gate state or seed file — gates will warm up from live data")
            return

        try:
            with open(path) as f:
                seed = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[Safety] Failed to load gate {source}: {e}")
            return

        # Staleness check
        saved_at_str = seed.pop("_saved_at", None)
        if saved_at_str:
            try:
                saved_dt = datetime.fromisoformat(saved_at_str)
                now_et = datetime.now(_ET)
                age_days = (now_et.date() - saved_dt.astimezone(_ET).date()).days
                if age_days > 5:
                    logger.warning(
                        f"[Safety] Gate {source} is {age_days} days old — "
                        f"gate values may be stale until first full trading day"
                    )
                logger.info(f"[Safety] Loading gate {source} (age: {age_days} day{'s' if age_days != 1 else ''})")
            except (ValueError, TypeError):
                logger.info(f"[Safety] Loading gate {source}")
        else:
            logger.info(f"[Safety] Loading gate {source} (no timestamp)")

        for inst, data in seed.items():
            if not isinstance(data, dict):
                continue

            # --- ADR directional gate ---
            if inst in self._adr_lookback and "adr_completed_ranges" in data:
                self._adr_completed_ranges[inst] = data["adr_completed_ranges"]
                self._adr_value[inst] = data.get("adr_value")
                if data.get("last_rth_date"):
                    self._adr_rth_session[inst] = {
                        "date": None, "open": None, "high": None, "low": None, "close": None,
                    }
                logger.info(
                    f"[Safety] ADR seed for {inst}: ADR={data.get('adr_value')}, "
                    f"{len(data['adr_completed_ranges'])} daily ranges"
                )

            # --- Prior-day ATR gate ---
            if inst in self._prior_day_atr_thresholds and "atr_daily_ranges" in data:
                self._atr_daily_ranges[inst] = data["atr_daily_ranges"]
                self._prior_day_atr_value[inst] = data.get("atr_value")
                self._atr_daily_tracking[inst] = {"date": None, "high": None, "low": None}
                atr_val = data.get("atr_value")
                if atr_val is not None:
                    threshold = self._prior_day_atr_thresholds[inst]
                    gate_ok = atr_val >= threshold
                    self._prior_day_atr_gate_current[inst] = gate_ok
                    self._prior_day_atr_gate_prev[inst] = gate_ok
                    logger.info(
                        f"[Safety] ATR seed for {inst}: ATR={atr_val:.1f}, "
                        f"threshold={threshold:.1f}, gate={'OPEN' if gate_ok else 'BLOCKED'}"
                    )

            # --- Prior-day level gate ---
            if inst in self._prior_day_buffers and "prior_day_levels" in data:
                levels = data["prior_day_levels"]
                self._prior_day_levels[inst] = levels
                self._current_rth[inst] = {
                    "date": None, "high": None, "low": None,
                    "closes": [], "volumes": [],
                }
                logger.info(
                    f"[Safety] Prior-day levels seed for {inst}: "
                    f"H={levels.get('high')} L={levels.get('low')} "
                    f"VPOC={levels.get('vpoc')}"
                )

            # --- Leledc gate ---
            if inst in self._leledc_thresholds and "leledc_closes" in data:
                closes = data["leledc_closes"]
                self._leledc_closes[inst] = closes
                threshold = self._leledc_thresholds[inst]
                bull_count = 0
                bear_count = 0
                lookback = self._LELEDC_LOOKBACK
                for i in range(lookback, len(closes)):
                    curr = closes[i]
                    prev = closes[i - lookback]
                    bull_count = (bull_count + 1) if curr > prev else 0
                    bear_count = (bear_count + 1) if curr < prev else 0
                self._leledc_bull_count[inst] = bull_count
                self._leledc_bear_count[inst] = bear_count
                exhausted = bull_count >= threshold or bear_count >= threshold
                self._leledc_gate_current[inst] = not exhausted
                self._leledc_gate_prev[inst] = not exhausted
                logger.info(
                    f"[Safety] Leledc seed for {inst}: bull={bull_count}, "
                    f"bear={bear_count}, gate={'BLOCKED' if exhausted else 'OPEN'}"
                )

            # --- ICT: Weekly VPOC/VAL ---
            if "weekly_rth_closes" in data:
                self._weekly_rth_closes[inst] = data["weekly_rth_closes"]
                self._weekly_rth_volumes[inst] = data.get("weekly_rth_volumes", [])
            if "weekly_vpoc" in data:
                self._weekly_vpoc[inst] = data["weekly_vpoc"]
            if "weekly_val" in data:
                self._weekly_val[inst] = data["weekly_val"]
            if "weekly_vpoc_strength" in data:
                self._weekly_vpoc_strength[inst] = data["weekly_vpoc_strength"]
            if self._weekly_vpoc.get(inst) is not None:
                logger.info(
                    f"[Safety] Weekly VPOC seed for {inst}: "
                    f"VPOC={self._weekly_vpoc[inst]}, VAL={self._weekly_val.get(inst)}, "
                    f"strength={self._weekly_vpoc_strength.get(inst, 0)}"
                )

            # --- ICT: Active Order Blocks ---
            if "active_obs" in data:
                self._active_obs[inst] = data["active_obs"]
                if not self._ob_5min_history.get(inst):
                    self._ob_5min_history[inst] = deque(maxlen=3)
                    self._ob_5min_accum[inst] = {}
                    self._ob_5min_bar_count[inst] = 0
                logger.info(
                    f"[Safety] OB seed for {inst}: {len(data['active_obs'])} active OBs"
                )

        logger.info(f"[Safety] Gate data loaded from {source}")

    def _save_gate_state(self) -> None:
        """Persist current gate state to disk for next startup.

        Called at daily reset. Saves ADR ranges, ATR, prior-day levels, and
        leledc state so the next engine startup has fresh gate values without
        needing the original seed file or manual intervention.
        """
        state: dict = {"_saved_at": datetime.now(_ET).isoformat()}

        instruments: set[str] = set()
        instruments.update(self._adr_completed_ranges.keys())
        instruments.update(self._atr_daily_ranges.keys())
        instruments.update(self._prior_day_levels.keys())
        instruments.update(self._leledc_closes.keys())
        instruments.update(self._weekly_rth_closes.keys())
        instruments.update(self._active_obs.keys())
        instruments.update(self._weekly_vpoc.keys())

        for inst in instruments:
            d: dict = {}

            if inst in self._adr_completed_ranges:
                d["adr_completed_ranges"] = [round(r, 2) for r in self._adr_completed_ranges[inst]]
                adr = self._adr_value.get(inst)
                d["adr_value"] = round(adr, 2) if adr is not None else None

            if inst in self._atr_daily_ranges:
                d["atr_daily_ranges"] = [round(r, 2) for r in self._atr_daily_ranges[inst]]
                atr = self._prior_day_atr_value.get(inst)
                d["atr_value"] = round(atr, 2) if atr is not None else None

            if inst in self._prior_day_levels:
                d["prior_day_levels"] = self._prior_day_levels[inst]

            if inst in self._leledc_closes:
                d["leledc_closes"] = [round(c, 4) for c in self._leledc_closes[inst]]

            session = self._adr_rth_session.get(inst, {})
            d["last_rth_date"] = str(session.get("date", "")) if session.get("date") else None

            # ICT: weekly VPOC/VAL accumulators + computed levels
            if inst in self._weekly_rth_closes and self._weekly_rth_closes[inst]:
                d["weekly_rth_closes"] = [round(c, 2) for c in self._weekly_rth_closes[inst]]
                d["weekly_rth_volumes"] = [round(v, 2) for v in self._weekly_rth_volumes.get(inst, [])]
            if self._weekly_vpoc.get(inst) is not None:
                d["weekly_vpoc"] = round(self._weekly_vpoc[inst], 2)
            if self._weekly_val.get(inst) is not None:
                d["weekly_val"] = round(self._weekly_val[inst], 2)
            if self._weekly_vpoc_strength.get(inst):
                d["weekly_vpoc_strength"] = self._weekly_vpoc_strength[inst]

            # ICT: active order blocks
            if inst in self._active_obs and self._active_obs[inst]:
                d["active_obs"] = self._active_obs[inst]

            state[inst] = d

        try:
            self._GATE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self._GATE_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"[Safety] Gate state persisted ({len(instruments)} instruments)")
        except OSError as e:
            logger.warning(f"[Safety] Failed to save gate state: {e}")

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
        self._prior_day_atr_gate_prev[inst] = self._prior_day_atr_gate_current.get(inst, True)
        self._adr_dir_ratio_prev[inst] = self._adr_dir_ratio_current.get(inst, 0.0)

        self._update_leledc(bar)
        self._update_prior_day_tracking(bar)
        self._update_prior_day_atr(bar)
        self._update_adr_directional(bar)

        # VWAP + Opening Range + Weekly VPOC accumulation (RTH only)
        bar_et = bar.timestamp.astimezone(_ET)
        et_mins = bar_et.hour * 60 + bar_et.minute
        if self._RTH_OPEN_ET <= et_mins < self._RTH_CLOSE_ET:
            # VWAP: accumulate typical_price * volume (skip zero-volume bars)
            if bar.volume > 0:
                tp = (bar.high + bar.low + bar.close) / 3
                self._vwap_num[inst] = self._vwap_num.get(inst, 0.0) + tp * bar.volume
                self._vwap_den[inst] = self._vwap_den.get(inst, 0.0) + bar.volume
            # Opening range: first 30 min of RTH (10:00-10:30 ET)
            if et_mins < self._OR_END_ET and not self._or_finalized.get(inst, False):
                if self._or_high.get(inst) is None:
                    self._or_high[inst] = bar.high
                    self._or_low[inst] = bar.low
                else:
                    self._or_high[inst] = max(self._or_high[inst], bar.high)
                    self._or_low[inst] = min(self._or_low[inst], bar.low)
            elif et_mins >= self._OR_END_ET and not self._or_finalized.get(inst, False):
                self._or_finalized[inst] = True

            # Weekly VPOC/VAL: accumulate RTH closes + volumes for volume profile
            if inst in self._weekly_bin_width:
                if inst not in self._weekly_rth_closes:
                    self._weekly_rth_closes[inst] = []
                    self._weekly_rth_volumes[inst] = []
                self._weekly_rth_closes[inst].append(bar.close)
                self._weekly_rth_volumes[inst].append(bar.volume)

            # Developing Daily VPOC: accumulate RTH closes + volumes, recompute each bar
            if inst in self._weekly_bin_width and bar.volume > 0:
                if inst not in self._daily_vpoc_closes:
                    self._daily_vpoc_closes[inst] = []
                    self._daily_vpoc_volumes[inst] = []
                self._daily_vpoc_closes[inst].append(bar.close)
                self._daily_vpoc_volumes[inst].append(bar.volume)
                bw = self._weekly_bin_width[inst]
                result = self._compute_value_area(
                    self._daily_vpoc_closes[inst],
                    self._daily_vpoc_volumes[inst],
                    bin_width=bw,
                )
                if result is not None:
                    new_vpoc = result[0]
                    prev_vpoc = self._developing_vpoc.get(inst)
                    if prev_vpoc is not None and new_vpoc != prev_vpoc:
                        self._dvpoc_stability[inst] = 0
                    else:
                        self._dvpoc_stability[inst] = self._dvpoc_stability.get(inst, 0) + 1
                    self._developing_vpoc[inst] = new_vpoc
                    # VCR: max_bin_vol / total_vol (inline binning)
                    closes = self._daily_vpoc_closes[inst]
                    volumes = self._daily_vpoc_volumes[inst]
                    total_vol = sum(volumes)
                    if total_vol > 0:
                        price_min = math.floor(min(closes) / bw) * bw
                        price_max = math.ceil(max(closes) / bw) * bw
                        if price_min == price_max:
                            price_max = price_min + bw
                        n_bins = int(round((price_max - price_min) / bw)) + 1
                        bin_vols = [0.0] * n_bins
                        for c, v in zip(closes, volumes):
                            idx = int(round((c - price_min) / bw))
                            idx = min(max(idx, 0), n_bins - 1)
                            bin_vols[idx] += v
                        self._daily_vpoc_strength[inst] = round(max(bin_vols) / total_vol, 4)

        # Order Block tracking: accumulate 1-min into 5-min, detect on 5-min completion
        self._update_order_blocks_5min(bar)

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

        # Check only the level types specified in config (default: all 5)
        level_keys = self._prior_day_level_keys.get(inst, ("high", "low", "vpoc", "vah", "val"))
        for key in level_keys:
            lvl = levels.get(key)
            if lvl is not None and abs(bar.close - lvl) <= buffer:
                self._prior_day_gate_current[inst] = False
                return
        self._prior_day_gate_current[inst] = True

    def _update_prior_day_atr(self, bar: Bar) -> None:
        """Track daily ranges and compute Wilder ATR(14) for entry gating.

        Tracks high/low across ALL bars (not just RTH) per calendar date.
        On date change, finalizes prior day's range and updates ATR.
        Gate: blocks entries when prior-day ATR < threshold.
        Fail-open during warmup (first ATR_PERIOD days).
        """
        inst = bar.instrument
        if inst not in self._prior_day_atr_thresholds:
            return

        bar_et = bar.timestamp.astimezone(_ET)
        bar_date = bar_et.date()

        if inst not in self._atr_daily_tracking:
            self._atr_daily_tracking[inst] = {"date": None, "high": None, "low": None}
            self._atr_daily_ranges[inst] = []
            self._prior_day_atr_value[inst] = None

        tracking = self._atr_daily_tracking[inst]

        # New calendar date → finalize previous day's range and update ATR
        if tracking["date"] is not None and bar_date != tracking["date"]:
            if tracking["high"] is not None and tracking["low"] is not None:
                day_range = tracking["high"] - tracking["low"]
                ranges = self._atr_daily_ranges[inst]
                ranges.append(day_range)
                # Keep last 20 days (more than enough for ATR(14))
                if len(ranges) > 20:
                    ranges[:] = ranges[-20:]

                # Compute Wilder ATR
                period = self._ATR_PERIOD
                if len(ranges) >= period:
                    if len(ranges) == period:
                        # Seed: simple mean of first `period` ranges
                        atr = sum(ranges) / period
                    else:
                        # Wilder: ATR = (prev_ATR * (period-1) + new_range) / period
                        prev_atr = self._prior_day_atr_value[inst]
                        if prev_atr is not None:
                            atr = (prev_atr * (period - 1) + day_range) / period
                        else:
                            atr = sum(ranges[-period:]) / period
                    self._prior_day_atr_value[inst] = atr
                    logger.info(
                        f"[Safety] Prior-day ATR for {inst}: {atr:.1f} "
                        f"(range={day_range:.1f}, days={len(ranges)})"
                    )

            # Reset for new day
            tracking["high"] = None
            tracking["low"] = None

        tracking["date"] = bar_date

        # Track daily high/low across ALL bars
        if tracking["high"] is None:
            tracking["high"] = bar.high
            tracking["low"] = bar.low
        else:
            tracking["high"] = max(tracking["high"], bar.high)
            tracking["low"] = min(tracking["low"], bar.low)

        # Gate: check prior-day ATR vs threshold
        atr_val = self._prior_day_atr_value.get(inst)
        if atr_val is None:
            self._prior_day_atr_gate_current[inst] = True  # fail-open during warmup
        else:
            threshold = self._prior_day_atr_thresholds[inst]
            self._prior_day_atr_gate_current[inst] = atr_val >= threshold

    def _update_adr_directional(self, bar: Bar) -> None:
        """Track RTH session and compute ADR directional gate ratio.

        Session tracking: RTH 10:00-16:00 ET only (matches backtest adr_common.py).
        ADR: rolling N-day simple mean of prior completed RTH daily ranges.
        Stores move_from_open / ADR ratio; directional check deferred to check_can_trade
        (which knows the entry side).

        Fail-open: ratio = 0.0 during warmup or outside RTH (won't trigger any block).
        """
        inst = bar.instrument
        if inst not in self._adr_lookback:
            return

        bar_et = bar.timestamp.astimezone(_ET)
        et_mins = bar_et.hour * 60 + bar_et.minute
        bar_date = bar_et.date()

        if inst not in self._adr_rth_session:
            self._adr_rth_session[inst] = {"date": None, "open": None, "high": None, "low": None, "close": None}
            self._adr_completed_ranges[inst] = []
            self._adr_value[inst] = None

        session = self._adr_rth_session[inst]

        # New calendar date → finalize previous day's RTH range and update ADR
        if session["date"] is not None and bar_date != session["date"]:
            if session["high"] is not None and session["low"] is not None:
                day_range = session["high"] - session["low"]
                ranges = self._adr_completed_ranges[inst]
                ranges.append(day_range)
                if len(ranges) > 30:
                    ranges[:] = ranges[-30:]

                # Compute ADR as simple rolling mean of prior N days
                lookback = self._adr_lookback[inst]
                if len(ranges) >= lookback:
                    self._adr_value[inst] = sum(ranges[-lookback:]) / lookback
                    logger.info(
                        f"[Safety] ADR for {inst}: {self._adr_value[inst]:.1f} "
                        f"(range={day_range:.1f}, lb={lookback}, days={len(ranges)})"
                    )

            # Reset session for new day
            session["open"] = None
            session["high"] = None
            session["low"] = None
            session["close"] = None

        session["date"] = bar_date

        # Track RTH bars only (10:00-16:00 ET)
        if self._RTH_OPEN_ET <= et_mins < self._RTH_CLOSE_ET:
            if session["open"] is None:
                session["open"] = bar.open
                session["high"] = bar.high
                session["low"] = bar.low
            else:
                session["high"] = max(session["high"], bar.high)
                session["low"] = min(session["low"], bar.low)
            session["close"] = bar.close  # Last RTH bar close = RTH close

        # Compute ratio: move_from_open / ADR
        adr_val = self._adr_value.get(inst)
        if session["open"] is not None and adr_val is not None and adr_val > 0:
            move = bar.close - session["open"]
            self._adr_dir_ratio_current[inst] = move / adr_val
        else:
            self._adr_dir_ratio_current[inst] = 0.0  # fail-open

    # ------------------------------------------------------------------
    # ICT Order Block tracking (OBSERVATION ONLY — never enters check_can_trade)
    # ------------------------------------------------------------------

    _MAX_OBS_PER_DIRECTION = 2

    def _update_order_blocks_5min(self, bar: Bar) -> None:
        """Accumulate 1-min bars into 5-min bars, detect OBs on 5-min completion.

        Detection runs on 5-min resampled bars (matching forensics validation
        that found -37pp WR on 5-min OBs). Mitigation runs on every 1-min bar
        for responsiveness.

        OBs persist until mitigated — no daily or weekly reset.
        Max 2 per direction per instrument (FIFO eviction).
        """
        inst = bar.instrument
        if inst not in self._active_obs:
            self._active_obs[inst] = []
            self._ob_5min_history[inst] = deque(maxlen=3)
            self._ob_5min_accum[inst] = {}
            self._ob_5min_bar_count[inst] = 0

        # --- Mitigate existing OBs on every 1-min bar (responsive) ---
        surviving = []
        for ob in self._active_obs[inst]:
            if ob["is_bull"] and bar.close < ob["bottom"]:
                continue  # Bullish OB mitigated — close below zone
            if not ob["is_bull"] and bar.close > ob["top"]:
                continue  # Bearish OB mitigated — close above zone
            surviving.append(ob)
        self._active_obs[inst] = surviving

        # --- Accumulate 1-min bars into 5-min bar ---
        accum = self._ob_5min_accum[inst]
        if not accum:
            # Start new 5-min bar
            accum["o"] = bar.open
            accum["h"] = bar.high
            accum["l"] = bar.low
            accum["c"] = bar.close
            self._ob_5min_bar_count[inst] = 1
        else:
            accum["h"] = max(accum["h"], bar.high)
            accum["l"] = min(accum["l"], bar.low)
            accum["c"] = bar.close
            self._ob_5min_bar_count[inst] += 1

        # Emit completed 5-min bar every 5 bars
        if self._ob_5min_bar_count[inst] >= 5:
            completed_bar = dict(accum)
            self._ob_5min_history[inst].append(completed_bar)
            self._ob_5min_accum[inst] = {}
            self._ob_5min_bar_count[inst] = 0

            # --- Detect OBs on completed 5-min bars ---
            hist = self._ob_5min_history[inst]
            if len(hist) >= 3:
                b0, b1, b2 = hist[0], hist[1], hist[2]

                # Bullish OB: b0 bearish, b1 bullish engulfing, b2 confirms
                is_bull_ob = (
                    b0["o"] > b0["c"]
                    and b1["c"] > b1["o"]
                    and b2["c"] > b2["o"]
                    and b1["l"] < b0["l"]
                    and b2["c"] > b1["h"]
                )

                # Bearish OB: b0 bullish, b1 bearish engulfing, b2 confirms
                is_bear_ob = (
                    b0["o"] < b0["c"]
                    and b1["c"] < b1["o"]
                    and b2["c"] < b2["o"]
                    and b1["h"] > b0["h"]
                    and b2["c"] < b1["l"]
                )

                if is_bull_ob:
                    self._add_ob(inst, is_bull=True, bottom=b1["l"], top=b1["h"])
                if is_bear_ob:
                    self._add_ob(inst, is_bull=False, bottom=b1["l"], top=b1["h"])

    def _add_ob(self, inst: str, is_bull: bool, bottom: float, top: float) -> None:
        """Add an OB zone with FIFO eviction (max 2 per direction)."""
        obs = self._active_obs.setdefault(inst, [])
        same_dir = [ob for ob in obs if ob["is_bull"] == is_bull]
        while len(same_dir) >= self._MAX_OBS_PER_DIRECTION:
            # Evict oldest (FIFO)
            oldest = same_dir.pop(0)
            obs.remove(oldest)
            same_dir = [ob for ob in obs if ob["is_bull"] == is_bull]

        obs.append({
            "is_bull": is_bull,
            "bottom": round(bottom, 2),
            "top": round(top, 2),
        })

    def get_ict_proximity(self, inst: str, price: float) -> list[str]:
        """Return list of ICT level tags with distance near the given price.

        Format: "wPOC +2.3" or "BEAR_OB" (OBs show no distance — entry is inside zone).
        Called at trade entry time to tag trades. Observation only.
        Threshold: 10 pts for weekly levels, inside zone for OBs.
        """
        tags = []
        vpoc = self._weekly_vpoc.get(inst)
        if vpoc is not None and abs(price - vpoc) <= 10:
            dist = round(price - vpoc, 1)
            tags.append(f"wPOC {dist:+.1f}")
        val = self._weekly_val.get(inst)
        if val is not None and abs(price - val) <= 10:
            dist = round(price - val, 1)
            tags.append(f"wVAL {dist:+.1f}")
        for ob in self._active_obs.get(inst, []):
            if ob["bottom"] <= price <= ob["top"]:
                tags.append("BULL_OB" if ob["is_bull"] else "BEAR_OB")
        return tags

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
            profile = self._compute_value_area(rth["closes"], rth["volumes"], bin_width=self._weekly_bin_width.get(inst, 5.0))
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

        # Portfolio context accumulators (loss day messaging)
        self._weekly_pnl += adjusted_pnl
        self._monthly_pnl += adjusted_pnl
        self._cumulative_pnl += adjusted_pnl
        if self._cumulative_pnl > self._equity_peak:
            self._equity_peak = self._cumulative_pnl

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

    def check_can_trade(self, instrument: str, qty: int, strategy_id: str = "",
                        side: str = "") -> tuple[bool, str]:
        """Check if a trade is allowed. Returns (ok, reason).

        Args:
            side: "long" or "short" — needed for directional gates (ADR).
        """
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
                        active_keys = cfg.prior_day_level_keys
                        level_str = " ".join(f"{k.upper()}={levels.get(k, '?')}" for k in active_keys)
                        return False, (
                            f"Near prior-day level (buf={cfg.prior_day_level_buffer}): "
                            f"{level_str}"
                        )

        # Prior-day ATR gate
        if strategy_id:
            cfg = self._strategy_configs.get(strategy_id)
            if cfg and cfg.prior_day_atr_min > 0:
                strat = self._strategies.get(strategy_id)
                if strat and not strat.manual_override:
                    if not self._prior_day_atr_gate_prev.get(strat.instrument, True):
                        atr_val = self._prior_day_atr_value.get(strat.instrument)
                        return False, (
                            f"Prior-day ATR too low: {atr_val:.1f} < {cfg.prior_day_atr_min:.1f}"
                            if atr_val is not None else
                            f"Prior-day ATR gate: warmup (need {self._ATR_PERIOD} days)"
                        )

        # ADR directional gate
        if strategy_id and side:
            cfg = self._strategy_configs.get(strategy_id)
            if cfg and cfg.adr_directional_threshold > 0 and cfg.adr_lookback_days > 0:
                strat = self._strategies.get(strategy_id)
                if strat and not strat.manual_override:
                    ratio = self._adr_dir_ratio_prev.get(strat.instrument, 0.0)
                    thr = cfg.adr_directional_threshold
                    if side == "long" and ratio >= thr:
                        return False, (
                            f"ADR directional: ratio {ratio:.2f} >= {thr} "
                            f"(long blocked — rally already {ratio:.0%} of ADR)"
                        )
                    elif side == "short" and ratio <= -thr:
                        return False, (
                            f"ADR directional: ratio {ratio:.2f} <= -{thr} "
                            f"(short blocked — selloff already {abs(ratio):.0%} of ADR)"
                        )

        # Position size check
        if qty > self._config.max_position_size:
            return False, (
                f"Position size {qty} exceeds limit {self._config.max_position_size}"
            )

        return True, ""

    def check_all_gates_for_logging(self, instrument: str, strategy_id: str,
                                     side: str = "") -> str:
        """Check ALL gates and return comma-separated types of failing gates.

        Called only after check_can_trade() returned False, for enriched logging.
        Only checks entry gates (VIX, Leledc, prior-day level/ATR, ADR), not
        operational blocks (halted, paused, position size).
        """
        cfg = self._strategy_configs.get(strategy_id)
        strat = self._strategies.get(strategy_id)
        if not cfg or not strat or strat.manual_override:
            return ""

        failing = []

        # VIX death zone
        if self._vix_close is not None and cfg.vix_death_zone_min > 0:
            if cfg.vix_death_zone_min <= self._vix_close <= cfg.vix_death_zone_max:
                failing.append("vix_death_zone")

        # Leledc exhaustion
        if cfg.leledc_maj_qual > 0:
            if not self._leledc_gate_prev.get(strat.instrument, True):
                failing.append("leledc")

        # Prior-day level proximity
        if cfg.prior_day_level_buffer > 0:
            if not self._prior_day_gate_prev.get(strat.instrument, True):
                failing.append("prior_day_level")

        # Prior-day ATR
        if cfg.prior_day_atr_min > 0:
            if not self._prior_day_atr_gate_prev.get(strat.instrument, True):
                failing.append("prior_day_atr")

        # ADR directional
        if side and cfg.adr_directional_threshold > 0 and cfg.adr_lookback_days > 0:
            ratio = self._adr_dir_ratio_prev.get(strat.instrument, 0.0)
            thr = cfg.adr_directional_threshold
            if (side == "long" and ratio >= thr) or (side == "short" and ratio <= -thr):
                failing.append("adr_directional")

        return ",".join(failing)

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
        # 0. Portfolio context — count loss day, reset week/month boundaries
        if self._global_daily_pnl < 0 and self._today_date is not None:
            self._loss_days_this_month += 1
        today = datetime.now(_ET).date()
        if self._today_date is not None:
            # New week (Monday) — reset weekly P&L
            if today.isocalendar()[1] != self._today_date.isocalendar()[1]:
                self._weekly_pnl = 0.0
            # New month — reset monthly P&L and loss day counter
            if today.month != self._today_date.month:
                self._monthly_pnl = 0.0
                self._loss_days_this_month = 0
        self._today_date = today

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

        # 6. Reset VWAP + Opening Range accumulators for next day
        self._vwap_num.clear()
        self._vwap_den.clear()
        self._or_high.clear()
        self._or_low.clear()
        self._or_finalized.clear()

        # Reset developing daily VPOC
        self._daily_vpoc_closes.clear()
        self._daily_vpoc_volumes.clear()
        self._developing_vpoc.clear()
        self._daily_vpoc_strength.clear()
        self._dvpoc_stability.clear()

        # 6b. Weekly VPOC/VAL: on Monday, compute from prior week's accumulated data
        now_et = datetime.now(_ET)
        if now_et.weekday() == 0:  # Monday
            for inst in list(self._weekly_rth_closes.keys()):
                closes = self._weekly_rth_closes.get(inst, [])
                volumes = self._weekly_rth_volumes.get(inst, [])
                if closes and volumes:
                    bw = self._weekly_bin_width.get(inst, 5.0)
                    profile = self._compute_value_area(closes, volumes, bin_width=bw)
                    if profile is not None:
                        vpoc, _vah, val = profile
                        self._weekly_vpoc[inst] = vpoc
                        self._weekly_val[inst] = val
                        # Compute VPOC strength (conviction opacity)
                        total_vol = sum(volumes)
                        if total_vol > 0:
                            price_min = math.floor(min(closes) / bw) * bw
                            price_max = math.ceil(max(closes) / bw) * bw
                            if price_min == price_max:
                                price_max = price_min + bw
                            n_bins = int(round((price_max - price_min) / bw)) + 1
                            bin_vols = [0.0] * n_bins
                            for c, v in zip(closes, volumes):
                                idx = int(round((c - price_min) / bw))
                                idx = min(max(idx, 0), n_bins - 1)
                                bin_vols[idx] += v
                            max_bin = max(bin_vols)
                            self._weekly_vpoc_strength[inst] = round(max_bin / total_vol, 4)
                        logger.info(
                            f"[Safety] Weekly VPOC for {inst}: {vpoc:.2f}, "
                            f"VAL: {val:.2f}, strength: {self._weekly_vpoc_strength.get(inst, 0)}"
                        )
                # Clear accumulators for new week
                self._weekly_rth_closes[inst] = []
                self._weekly_rth_volumes[inst] = []
        # Note: OBs are NOT reset — they persist until mitigated

        # 7. Persist gate state for next startup
        self._save_gate_state()

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
                "atr_gated": (
                    cfg is not None
                    and cfg.prior_day_atr_min > 0
                    and not self._prior_day_atr_gate_prev.get(strat.instrument, True)
                    and not strat.manual_override
                ),
                "adr_dir_gated": (
                    cfg is not None
                    and cfg.adr_directional_threshold > 0
                    and not strat.manual_override
                    and (lambda r, t: abs(r) >= t)(
                        self._adr_dir_ratio_prev.get(strat.instrument, 0.0),
                        cfg.adr_directional_threshold if cfg else 0.0,
                    )
                ),
                "adr_dir_ratio": (
                    round(self._adr_dir_ratio_prev.get(strat.instrument, 0.0), 3)
                    if cfg and cfg.adr_directional_threshold > 0 else None
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
            "prior_day_atr": {
                inst: round(atr, 1) if atr is not None else None
                for inst, atr in self._prior_day_atr_value.items()
            },
            "adr": {
                inst: round(adr, 1) if adr is not None else None
                for inst, adr in self._adr_value.items()
            },
            "ict_levels": {
                inst: {
                    "weekly_vpoc": round(self._weekly_vpoc[inst], 2) if self._weekly_vpoc.get(inst) is not None else None,
                    "weekly_val": round(self._weekly_val[inst], 2) if self._weekly_val.get(inst) is not None else None,
                    "vpoc_strength": self._weekly_vpoc_strength.get(inst, 0),
                    "developing_vpoc": round(self._developing_vpoc[inst], 2) if self._developing_vpoc.get(inst) is not None else None,
                    "dvpoc_strength": round(self._daily_vpoc_strength.get(inst, 0.0), 4),
                    "dvpoc_stability": self._dvpoc_stability.get(inst, 0),
                }
                for inst in self._weekly_bin_width
            },
            "ob_zones": {
                inst: [
                    {
                        "is_bull": ob["is_bull"],
                        "bottom": ob["bottom"],
                        "top": ob["top"],
                        "midline": round((ob["bottom"] + ob["top"]) / 2, 2),
                    }
                    for ob in obs
                ]
                for inst, obs in self._active_obs.items()
                if obs  # only include instruments with active OBs
            },
            "portfolio_context": {
                "daily_pnl": round(self._global_daily_pnl, 2),
                "weekly_pnl": round(self._weekly_pnl, 2),
                "monthly_pnl": round(self._monthly_pnl, 2),
                "cumulative_pnl": round(self._cumulative_pnl, 2),
                "current_drawdown": round(self._equity_peak - self._cumulative_pnl, 2),
                "equity_peak": round(self._equity_peak, 2),
                "consecutive_losses": self._consecutive_losses,
                "loss_days_this_month": self._loss_days_this_month,
                "trade_count_today": self._global_trade_count,
            },
            "backtest_benchmarks": {
                "max_drawdown": 1420,
                "worst_streak": 5,
                "expected_loss_day_rate": 0.15,
                "strategy_wr": {
                    "MNQ_V15": 0.856,
                    "MNQ_VSCALPB": 0.749,
                    "MNQ_VSCALPC": 0.780,
                    "MES_V2": 0.620,
                },
            },
            "strategies": strategies,
        }

    def _broadcast_status(self) -> None:
        """Emit status_change on EventBus so WS clients update immediately."""
        if self._event_bus:
            self._event_bus.emit("status_change", self.get_status())
