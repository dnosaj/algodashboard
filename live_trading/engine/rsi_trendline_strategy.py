"""
RSI Trendline Breakout Strategy -- bar-by-bar incremental engine.

Converts the vectorized backtest in rsi_trendline_backtest.py to an
incremental engine suitable for live trading. Each call to on_bar()
processes one 1-min bar and returns a Signal.

Entry signals come from RSI trendline breaks (descending peak lines
break = long, ascending trough lines break = short). No SM, no episodes.

VALIDATION: Feed Databento CSV bar-by-bar into RsiTrendlineStrategy and
compare output trades against rsi_trendline_backtest.run_backtest(). Must
be IDENTICAL.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from .config import StrategyConfig
from .events import EventBus, Signal, SignalType, ExitReason, Bar, TradeRecord

logger = logging.getLogger(__name__)

_ET_TZ = ZoneInfo("America/New_York")

_MARKET_OPEN_ET = 9 * 60 + 30  # 9:30 AM ET in minutes from midnight


# ---------------------------------------------------------------------------
# Session helpers (ET minutes from midnight)
# ---------------------------------------------------------------------------

def _et_minutes_from_datetime(dt: datetime) -> int:
    """Convert a datetime to ET minutes from midnight (DST-aware)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    et = dt.astimezone(_ET_TZ)
    return et.hour * 60 + et.minute


def _parse_time_str(s: str) -> int:
    """Parse 'HH:MM' to minutes from midnight."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------------------
# Incremental RSI on 1-min closes (Wilder's)
# ---------------------------------------------------------------------------

class IncrementalRSI1m:
    """Wilder's RSI computed incrementally on 1-min closes.

    Matches compute_rsi() from rsi_trendline_backtest.py exactly:
      delta = np.diff(arr, prepend=arr[0])   -> delta[0] = 0
      ag[period] = mean(gain[1:period+1])    -> first real delta at bar 1
      for i in range(period+1, n): Wilder smoothing

    Bar 0 stores close (no delta). Bars 1..period accumulate deltas.
    At bar == period, RSI is initialized via SMA. Bars > period use Wilder.
    RSI for bars < period is 50.0 (matching backtest r[:period] = 50.0).
    """

    def __init__(self, period: int = 8):
        self.period = period
        self._value = 50.0
        self._prev_close: Optional[float] = None
        self._ag = 0.0
        self._al = 0.0
        self._bar_count = 0       # total calls to update()
        self._delta_count = 0     # real deltas seen (bar_count - 1)
        self._init_gains: list[float] = []
        self._init_losses: list[float] = []

    def update(self, close: float) -> float:
        """Feed one close price, return current RSI value."""
        self._bar_count += 1

        if self._prev_close is None:
            # First bar: no delta (matches prepend=arr[0] -> delta[0]=0)
            self._prev_close = close
            self._value = 50.0
            return self._value

        delta = close - self._prev_close
        self._prev_close = close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        self._delta_count += 1
        rp = self.period

        if self._delta_count <= rp:
            # Accumulation: collect rp deltas (indices 1..period in vectorized)
            self._init_gains.append(gain)
            self._init_losses.append(loss)

            if self._delta_count == rp:
                # Initialize RSI: SMA of collected gains/losses
                self._ag = sum(self._init_gains) / rp
                self._al = sum(self._init_losses) / rp
                rs = self._ag / self._al if self._al > 0 else 100.0
                self._value = 100.0 - 100.0 / (1.0 + rs)
                self._init_gains = None
                self._init_losses = None
            else:
                self._value = 50.0
        else:
            # Wilder smoothing
            self._ag = (self._ag * (rp - 1) + gain) / rp
            self._al = (self._al * (rp - 1) + loss) / rp
            rs = self._ag / self._al if self._al > 0 else 100.0
            self._value = 100.0 - 100.0 / (1.0 + rs)

        return self._value

    @property
    def value(self) -> float:
        """Current RSI value."""
        return self._value


# ---------------------------------------------------------------------------
# Trendline Tracker (incremental pivot + trendline + break detection)
# ---------------------------------------------------------------------------

@dataclass
class Trendline:
    x1: int        # start bar index
    y1: float      # start RSI value
    slope: float   # (y2-y1)/(x2-x1)
    broken_bar: int = 0  # 0=active, >0=bar where broken


class TrendlineTracker:
    """Incremental pivot detection + trendline construction + break detection.

    Direct port of generate_signals() from rsi_trendline_backtest.py,
    converted from a vectorized loop to bar-by-bar state machine.

    Pivot detection: A peak at bar ``piv_bar = current_bar - lb_right``
    if rsi[piv_bar] is strictly greater than all values in
    [piv_bar - lb_left, piv_bar + lb_right]. Same for troughs (minimum).
    We keep a rolling deque of RSI values of size ``lb_left + lb_right + 1``
    to evaluate pivots once enough right-side confirmation bars arrive.

    Trendline construction: When a new peak pivot is found, connect it to
    each of the last ``piv_lookback`` prior peaks. If the slope is negative
    (descending peaks), create a long trendline. For troughs, if slope is
    positive (ascending troughs), create a short trendline.

    Break detection: Project trendline value at current bar. If RSI crosses
    the projection beyond break_thresh and past the grace period, mark broken.
    """

    def __init__(self, lb_left: int = 10, lb_right: int = 3,
                 min_spacing: int = 10, piv_lookback: int = 5,
                 max_age: int = 2000, broken_max: int = 1000,
                 break_thresh: float = 0.0):
        self.lb_left = lb_left
        self.lb_right = lb_right
        self.min_spacing = min_spacing
        self.piv_lookback = piv_lookback
        self.max_age = max_age
        self.broken_max = broken_max
        self.break_thresh = break_thresh
        self.grace = min_spacing + 2 * lb_right

        # Rolling RSI buffer: need lb_left + lb_right + 1 values to evaluate
        # pivot at the center position (index lb_left in the window).
        self._window_size = lb_left + lb_right + 1
        self._rsi_buf: deque[float] = deque(maxlen=self._window_size)
        # Map each buffer position to its absolute bar index.  We track the
        # bar index of the most recent entry so we can derive all others.
        self._latest_bar_idx: int = -1

        # Pivot storage (capped at 20 each, matching backtest)
        self._peak_bars: list[int] = []
        self._peak_vals: list[float] = []
        self._trough_bars: list[int] = []
        self._trough_vals: list[float] = []

        # Active trendlines
        self._long_tls: list[Trendline] = []   # descending peaks -> long breaks
        self._short_tls: list[Trendline] = []  # ascending troughs -> short breaks

    def update(self, bar_idx: int, rsi_val: float) -> tuple[bool, bool]:
        """Process one bar. Returns (long_signal, short_signal).

        Must be called with monotonically increasing bar_idx starting from 0.
        """
        self._rsi_buf.append(rsi_val)
        self._latest_bar_idx = bar_idx

        # Need a full window before we can evaluate pivots
        if len(self._rsi_buf) < self._window_size:
            # Still need to check breaks on existing trendlines
            bull_break, bear_break = self._check_breaks(bar_idx, rsi_val)
            return (bull_break, bear_break)

        # --- Detect pivot at center of window ---
        # Center index in the buffer is lb_left (0-based)
        # Its absolute bar index is bar_idx - lb_right
        piv_bar = bar_idx - self.lb_right
        piv_val = self._rsi_buf[self.lb_left]

        # Check peak: piv_val strictly greater than all neighbours
        is_peak = True
        for j in range(0, self.lb_left):
            if self._rsi_buf[j] >= piv_val:
                is_peak = False
                break
        if is_peak:
            for j in range(self.lb_left + 1, self._window_size):
                if self._rsi_buf[j] >= piv_val:
                    is_peak = False
                    break

        if is_peak:
            spacing_ok = (not self._peak_bars or
                          (piv_bar - self._peak_bars[-1]) >= self.min_spacing)
            if spacing_ok:
                self._peak_bars.append(piv_bar)
                self._peak_vals.append(piv_val)
                if len(self._peak_bars) > 20:
                    self._peak_bars.pop(0)
                    self._peak_vals.pop(0)

                # Build descending peak trendlines (long setup)
                psz = len(self._peak_bars)
                if psz >= 2:
                    pstart = max(0, psz - 1 - self.piv_lookback)
                    for pk in range(psz - 2, pstart - 1, -1):
                        prev_bar = self._peak_bars[pk]
                        prev_val = self._peak_vals[pk]
                        if piv_val < prev_val:  # Descending
                            dx = max(piv_bar - prev_bar, 1)
                            slope = (piv_val - prev_val) / dx
                            self._long_tls.append(
                                Trendline(x1=prev_bar, y1=prev_val, slope=slope)
                            )

        # Check trough: piv_val strictly less than all neighbours
        is_trough = True
        for j in range(0, self.lb_left):
            if self._rsi_buf[j] <= piv_val:
                is_trough = False
                break
        if is_trough:
            for j in range(self.lb_left + 1, self._window_size):
                if self._rsi_buf[j] <= piv_val:
                    is_trough = False
                    break

        if is_trough:
            spacing_ok = (not self._trough_bars or
                          (piv_bar - self._trough_bars[-1]) >= self.min_spacing)
            if spacing_ok:
                self._trough_bars.append(piv_bar)
                self._trough_vals.append(piv_val)
                if len(self._trough_bars) > 20:
                    self._trough_bars.pop(0)
                    self._trough_vals.pop(0)

                # Build ascending trough trendlines (short setup)
                tsz = len(self._trough_bars)
                if tsz >= 2:
                    tstart = max(0, tsz - 1 - self.piv_lookback)
                    for tk in range(tsz - 2, tstart - 1, -1):
                        prev_bar = self._trough_bars[tk]
                        prev_val = self._trough_vals[tk]
                        if piv_val > prev_val:  # Ascending
                            dx = max(piv_bar - prev_bar, 1)
                            slope = (piv_val - prev_val) / dx
                            self._short_tls.append(
                                Trendline(x1=prev_bar, y1=prev_val, slope=slope)
                            )

        # --- Check breaks ---
        bull_break, bear_break = self._check_breaks(bar_idx, rsi_val)
        return (bull_break, bear_break)

    def _check_breaks(self, bar_idx: int, rsi_val: float) -> tuple[bool, bool]:
        """Prune trendlines and check for breakouts. Returns (bull, bear)."""
        bull_break = False
        bear_break = False

        # --- Long trendlines (descending peaks) ---
        keep = []
        for tl in self._long_tls:
            proj = tl.y1 + tl.slope * (bar_idx - tl.x1)

            # Pruning
            if tl.broken_bar == 0 and (bar_idx - tl.x1) > self.max_age:
                continue
            if tl.broken_bar > 0 and (bar_idx - tl.broken_bar) > self.broken_max:
                continue
            if tl.broken_bar == 0 and (proj > 105 or proj < -5):
                continue

            # Break detection
            if tl.broken_bar == 0:
                if ((bar_idx - tl.x1) > self.grace
                        and rsi_val > proj + self.break_thresh):
                    tl.broken_bar = bar_idx
                    bull_break = True
            keep.append(tl)
        self._long_tls = keep

        # --- Short trendlines (ascending troughs) ---
        keep = []
        for tl in self._short_tls:
            proj = tl.y1 + tl.slope * (bar_idx - tl.x1)

            if tl.broken_bar == 0 and (bar_idx - tl.x1) > self.max_age:
                continue
            if tl.broken_bar > 0 and (bar_idx - tl.broken_bar) > self.broken_max:
                continue
            if tl.broken_bar == 0 and (proj > 105 or proj < -5):
                continue

            if tl.broken_bar == 0:
                if ((bar_idx - tl.x1) > self.grace
                        and rsi_val < proj - self.break_thresh):
                    tl.broken_bar = bar_idx
                    bear_break = True
            keep.append(tl)
        self._short_tls = keep

        return (bull_break, bear_break)


# ---------------------------------------------------------------------------
# Trade State (shared with IncrementalStrategy)
# ---------------------------------------------------------------------------

@dataclass
class TradeState:
    """Internal trade state for RSI trendline strategy."""
    position: int = 0        # 0=flat, 1=long, -1=short
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_bar_idx: int = 0
    exit_bar_idx: int = -9999
    # Trail stop state (tp_scalp mode)
    max_favorable: float = 0.0
    trail_activated: bool = False
    # Active SL (set at entry based on SM alignment)
    active_sl_pts: float = 40.0
    # Partial exit state (scale-out)
    qty_remaining: int = 1
    partial_filled: bool = False
    partial_pnl_accum: float = 0.0
    # MAE tracking
    min_adverse: float = 0.0
    # Entry context
    entry_sm_value: float = 0.0
    entry_rsi_value: float = 0.0
    entry_bar_volume: float = 0.0
    entry_minutes_from_open: int = 0
    entry_bar_range: float = 0.0
    # Concurrent open positions at entry (injected by runner after fill)
    concurrent_positions: int = 0
    # Win/loss streak at entry
    streak_at_entry: int = 0
    # Gate state at entry (injected by runner)
    gate_vix_close: Optional[float] = None
    gate_leledc_active: Optional[bool] = None
    gate_atr_value: Optional[float] = None
    gate_adr_ratio: Optional[float] = None
    gate_leledc_count: Optional[int] = None
    # Trade group ID
    trade_group_id: Optional[str] = None
    # ICT level proximity at entry
    ict_near_levels: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# RSI Trendline Strategy
# ---------------------------------------------------------------------------

class RsiTrendlineStrategy:
    """Bar-by-bar RSI trendline breakout strategy.

    Entry signals come from TrendlineTracker (RSI trendline breaks).
    Exits follow the same TP/SL/partial/BE/EOD framework as IncrementalStrategy.

    Usage:
        strategy = RsiTrendlineStrategy(config, event_bus)
        for bar in warmup_bars:
            strategy.warmup(bar)
        strategy.start_trading()
        for bar in live_bars:
            signal = strategy.on_bar(bar)
    """

    def __init__(self, config: StrategyConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.event_bus = event_bus

        # Session boundaries in ET minutes
        self._session_start = _parse_time_str(config.session_start_et)
        self._session_end = _parse_time_str(config.session_end_et)
        self._session_close = _parse_time_str(config.session_close_et)

        # Indicators
        self.rsi_indicator = IncrementalRSI1m(period=config.rsi_len)
        self.tracker = TrendlineTracker()

        # SM indicator — passive observer (not used for entries, logged for analysis)
        from engine.strategy import IncrementalSM
        self.sm = IncrementalSM(
            config.sm_index, config.sm_flow, config.sm_norm, config.sm_ema
        )

        # Stub RSI attribute matching IncrementalStrategy's rsi.curr for exit context
        self.rsi = type('obj', (object,), {'curr': 50.0})()

        # Trade state
        self.state = TradeState()
        self.bar_idx = 0
        self.trades: list[TradeRecord] = []
        self._warming_up = True

        # Previous bar data (for stop loss checking bar i-1 close)
        self._prev_bar: Optional[Bar] = None

        # Sizing overrides (set via dashboard, cleared on force_resume/daily_reset)
        self._entry_qty_override: Optional[int] = None
        self._partial_qty_override: Optional[int] = None

        # Win/loss streak tracker
        self._streak: int = 0

        # Per-strategy lock serializes: monitor exit, bar-close exit, manual exit
        self.trade_lock = asyncio.Lock()

        # When True, intra-bar monitor owns TP/trail/SL exits; on_bar() skips them
        self.intrabar_monitor_active: bool = False

        # Pending signals from previous bar (bar[i-1] convention)
        self._pending_long: bool = False
        self._pending_short: bool = False

    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.config.instrument

    @property
    def position(self) -> int:
        return self.state.position

    @property
    def active_entry_qty(self) -> int:
        """Entry qty: override if set, else config default."""
        if self._entry_qty_override is not None:
            return self._entry_qty_override
        return self.config.entry_qty

    @property
    def active_partial_qty(self) -> int:
        """Partial close qty: override if set, else config default."""
        if self._partial_qty_override is not None:
            return self._partial_qty_override
        return self.config.partial_qty

    @property
    def is_warming_up(self) -> bool:
        return self._warming_up

    def warmup(self, bar: Bar) -> None:
        """Feed a bar during warmup (updates indicators, no trading)."""
        rsi_val = self.rsi_indicator.update(bar.close)
        self.tracker.update(self.bar_idx, rsi_val)
        self.rsi.curr = rsi_val
        self.sm.update(bar.close, bar.volume)
        self._prev_bar = bar
        self.bar_idx += 1

    def start_trading(self) -> None:
        """Switch from warmup to live trading mode."""
        self._warming_up = False
        logger.info(f"[{self.strategy_id}] Trading started. "
                    f"Warmup bars: {self.bar_idx}, "
                    f"RSI value: {self.rsi_indicator.value:.2f}")

    def on_bar(self, bar: Bar) -> Signal:
        """Process one 1-min bar and return a trading signal.

        Signal on bar[i-1] convention: TrendlineTracker is called with
        bar i's data. The break signal it returns represents a break
        detected on bar i. The ENTRY happens at the NEXT bar's open,
        which is handled by runner.py. However, internally we follow the
        backtest pattern: signal on bar[i-1], entry fill at bar[i] open.
        So we store the tracker result and act on it next bar.

        Actually, matching the backtest exactly: the generate_signals loop
        checks rsi[i] against trendlines at bar i, and if there's a break
        it sets long_signal[i] = True. Then run_backtest checks
        long_signal[i-1] and fills at bar[i] open. So in on_bar(bar_i):
          1. Check if previous bar had a signal (_pending_long/short)
          2. Update RSI + tracker with bar_i data (sets new pending signals)
          3. Handle exits
        """
        self.bar_idx += 1

        if self._warming_up:
            self.warmup(bar)
            self.bar_idx -= 1  # warmup increments bar_idx, undo double-increment
            return Signal(type=SignalType.NONE, instrument=self.config.instrument)

        prev_bar = self._prev_bar

        # ------------------------------------------------------------------
        # UPDATE indicators with current bar (bar i)
        # ------------------------------------------------------------------
        rsi_val = self.rsi_indicator.update(bar.close)
        long_break, short_break = self.tracker.update(self.bar_idx, rsi_val)
        self.rsi.curr = rsi_val
        self.sm.update(bar.close, bar.volume)

        # Emit bar event
        if self.event_bus:
            self.event_bus.emit("bar", bar)

        # ------------------------------------------------------------------
        # Track MFE/MAE (all bars while positioned, uses prev_bar)
        # ------------------------------------------------------------------
        if self.state.position != 0 and prev_bar is not None:
            if self.state.position == 1:
                excursion = prev_bar.close - self.state.entry_price
            else:
                excursion = self.state.entry_price - prev_bar.close
            if excursion > self.state.max_favorable:
                self.state.max_favorable = excursion
            if excursion < self.state.min_adverse:
                self.state.min_adverse = excursion

        # Session info
        bar_mins = _et_minutes_from_datetime(bar.timestamp)
        signal = Signal(type=SignalType.NONE, instrument=self.config.instrument,
                        rsi_value=rsi_val)

        # --- EOD Close (uses bar.close like backtest) ---
        if self.state.position != 0 and bar_mins >= self._session_close:
            signal = self._close_position(bar, ExitReason.EOD, fill_price=bar.close)
            self._prev_bar = bar
            return signal

        # --- Exits for open positions ---
        if self.state.position != 0 and not self.intrabar_monitor_active:
            # Stop loss: check bar[i-1] close, fill at bar[i] open
            # After TP1 with move_sl_to_be_after_tp1, SL moves to breakeven
            if self.config.max_loss_pts > 0 and prev_bar is not None:
                sl_pts = self.state.active_sl_pts  # Set at entry time based on SM alignment
                is_be = False
                if self.config.move_sl_to_be_after_tp1 and self.state.partial_filled:
                    sl_pts = 0  # Breakeven
                    is_be = True
                reason = ExitReason.BREAKEVEN if is_be else ExitReason.STOP_LOSS
                if self.state.position == 1 and prev_bar.close <= self.state.entry_price - sl_pts:
                    signal = self._close_position(bar, reason)
                    self._prev_bar = bar
                    return signal
                elif self.state.position == -1 and prev_bar.close >= self.state.entry_price + sl_pts:
                    signal = self._close_position(bar, reason)
                    self._prev_bar = bar
                    return signal

        # TP/trail exits (tp_scalp mode)
        if (self.config.exit_mode == "tp_scalp" and self.state.position != 0
                and not self.intrabar_monitor_active):
            if prev_bar is not None:
                if self.state.position == 1:
                    unrealized = prev_bar.close - self.state.entry_price
                else:
                    unrealized = self.state.entry_price - prev_bar.close

                # Update max favorable excursion
                if unrealized > self.state.max_favorable:
                    self.state.max_favorable = unrealized

                # Partial TP1 exit
                if (self.config.partial_tp_pts > 0
                        and not self.state.partial_filled
                        and unrealized >= self.config.partial_tp_pts):
                    signal = self._partial_close(bar, ExitReason.TAKE_PROFIT_PARTIAL,
                                                 qty=self.active_partial_qty)
                    self._prev_bar = bar
                    return signal

                # TP exit
                if self.config.tp_pts > 0 and unrealized >= self.config.tp_pts:
                    signal = self._close_position(bar, ExitReason.TAKE_PROFIT)
                    self._prev_bar = bar
                    return signal

                # Trail exit
                if self.state.max_favorable >= self.config.trail_activate_pts:
                    self.state.trail_activated = True
                if self.state.trail_activated:
                    trail_level = self.state.max_favorable - self.config.trail_distance_pts
                    if unrealized <= trail_level:
                        signal = self._close_position(bar, ExitReason.TRAIL_STOP)
                        self._prev_bar = bar
                        return signal

        # BE_TIME exit: close stale trades after N bars
        if (self.config.breakeven_after_bars > 0
                and self.state.position != 0
                and not self.config.structure_exit_type):
            bars_held = self.bar_idx - self.state.entry_bar_idx - 1
            if bars_held >= self.config.breakeven_after_bars:
                signal = self._close_position(bar, ExitReason.BE_TIME)
                self._prev_bar = bar
                return signal

        # --- Entry logic ---
        # Signal on bar[i-1] convention: long_break/short_break from the PREVIOUS
        # bar's tracker update triggers entry on THIS bar's open. But since we
        # call tracker.update() with bar_i data in on_bar(bar_i), the break is
        # detected at bar_i. Following the backtest pattern (signal[i-1], fill[i]),
        # the runner pipeline handles the 1-bar delay: on_bar returns BUY/SELL,
        # runner places order, fill happens at current bar's open price.
        #
        # Wait -- re-reading the backtest more carefully:
        # The backtest loop at bar i checks long_signal[i-1] and fills at opens[i].
        # generate_signals sets long_signal[i] = True when RSI breaks at bar i.
        # So in live: we need to store the break signal and act on it NEXT bar.
        # But IncrementalStrategy returns the signal from on_bar and the runner
        # places the order -- the fill price IS bar.open (current bar).
        # So we need: detect break at bar i-1 -> on_bar(bar_i) opens position.
        # That means we should use the PREVIOUS call's tracker results.
        #
        # To handle this cleanly: store pending signals, act next bar.
        if self.state.position == 0:
            bars_since = self.bar_idx - self.state.exit_bar_idx
            in_session = self._session_start <= bar_mins <= self._session_end
            cd_ok = bars_since >= self.config.cooldown

            if in_session and cd_ok:
                # Use PREVIOUS bar's break signals (stored from last on_bar call)
                if self._pending_long:
                    signal = self._open_position(bar, "long", rsi_val)
                elif self._pending_short:
                    signal = self._open_position(bar, "short", rsi_val)

        # Store this bar's break results for next bar's entry check
        self._pending_long = long_break
        self._pending_short = short_break

        self._prev_bar = bar
        return signal

    def _open_position(self, bar: Bar, side: str, rsi_val: float) -> Signal:
        """Open a new position at bar.open."""
        self.state.position = 1 if side == "long" else -1
        self.state.entry_price = bar.open
        self.state.entry_time = bar.timestamp
        self.state.entry_bar_idx = self.bar_idx
        self.state.qty_remaining = self.active_entry_qty
        self.state.partial_filled = False
        self.state.min_adverse = 0.0
        self.state.max_favorable = 0.0
        self.state.trail_activated = False

        # Generate trade group ID
        self.state.trade_group_id = f"{self.strategy_id}_{bar.timestamp.isoformat()}"

        # Capture entry context
        self.state.entry_sm_value = round(self.sm.value, 4)
        # SM-aware SL: tighter SL=25 when SM opposes entry (validated OOS PF +4.2%)
        sm_val = self.sm.value
        sm_opposes = ((side == "long" and sm_val < -0.25) or
                      (side == "short" and sm_val > 0.25))
        self.state.active_sl_pts = 25.0 if sm_opposes else self.config.max_loss_pts
        self.state.entry_rsi_value = rsi_val
        self.state.entry_bar_volume = bar.volume
        self.state.entry_minutes_from_open = _et_minutes_from_datetime(bar.timestamp) - _MARKET_OPEN_ET
        self.state.entry_bar_range = round(bar.high - bar.low, 2)
        self.state.streak_at_entry = self._streak

        sig_type = SignalType.BUY if side == "long" else SignalType.SELL
        signal = Signal(
            type=sig_type,
            instrument=self.config.instrument,
            reason=f"RSI trendline {'bullish' if side == 'long' else 'bearish'} break",
            rsi_value=rsi_val,
        )

        if self.event_bus:
            self.event_bus.emit("signal", signal)

        logger.info(f"[{self.strategy_id}] OPEN {side.upper()} @ {bar.open:.2f} "
                    f"RSI={rsi_val:.1f}")
        return signal

    def _partial_close(self, bar: Bar, reason: ExitReason,
                       qty: int = 1, fill_price: float = None) -> Signal:
        """Close partial position (TP1). Position stays open with reduced qty."""
        side = "long" if self.state.position == 1 else "short"
        exit_price = fill_price if fill_price is not None else bar.open

        if side == "long":
            pts = exit_price - self.state.entry_price
        else:
            pts = self.state.entry_price - exit_price

        pnl = pts * self.config.dollar_per_pt * qty

        trade = TradeRecord(
            instrument=self.config.instrument,
            side=side,
            entry_price=self.state.entry_price,
            exit_price=exit_price,
            entry_time=self.state.entry_time,
            exit_time=bar.timestamp,
            pts=pts,
            pnl_dollar=pnl,
            exit_reason=reason.value,
            bars_held=self.bar_idx - self.state.entry_bar_idx,
            strategy_id=self.strategy_id,
            qty=qty,
            is_partial=True,
            entry_sm_value=self.state.entry_sm_value,
            entry_sm_velocity=0.0,
            entry_rsi_value=self.state.entry_rsi_value,
            entry_bar_volume=self.state.entry_bar_volume,
            entry_minutes_from_open=self.state.entry_minutes_from_open,
            entry_bar_range=self.state.entry_bar_range,
            concurrent_positions=self.state.concurrent_positions,
            streak_at_entry=self.state.streak_at_entry,
            exit_sm_value=round(self.sm.value, 4),
            exit_rsi_value=self.rsi_indicator.value,
            is_runner=False,
            gate_vix_close=self.state.gate_vix_close,
            gate_leledc_active=self.state.gate_leledc_active,
            gate_atr_value=self.state.gate_atr_value,
            gate_adr_ratio=self.state.gate_adr_ratio,
            gate_leledc_count=self.state.gate_leledc_count,
            ict_near_levels=self.state.ict_near_levels,
            mfe_pts=round(self.state.max_favorable, 2),
            mae_pts=round(self.state.min_adverse, 2),
            trade_group_id=self.state.trade_group_id,
            signal_price=self.state.entry_price,
        )
        self.trades.append(trade)

        # Decrement qty, mark partial filled
        self.state.qty_remaining -= qty
        self.state.partial_filled = True
        self.state.partial_pnl_accum += pnl

        if self.event_bus:
            self.event_bus.emit("trade_closed", trade)

        logger.info(f"[{self.strategy_id}] PARTIAL CLOSE {side.upper()} x{qty} @ {exit_price:.2f} "
                    f"PnL={pts:+.2f}pts (${pnl:+.2f}) reason={reason.value} "
                    f"remaining={self.state.qty_remaining}")

        sig_type = SignalType.PARTIAL_CLOSE_LONG if side == "long" else SignalType.PARTIAL_CLOSE_SHORT
        return Signal(
            type=sig_type,
            instrument=self.config.instrument,
            reason=reason.value,
            exit_reason=reason,
            rsi_value=self.rsi_indicator.value,
        )

    def _close_position(self, bar: Bar, reason: ExitReason,
                        fill_price: float = None) -> Signal:
        """Close current position (all remaining contracts)."""
        side = "long" if self.state.position == 1 else "short"
        exit_price = fill_price if fill_price is not None else bar.open
        qty = self.state.qty_remaining

        if side == "long":
            pts = exit_price - self.state.entry_price
        else:
            pts = self.state.entry_price - exit_price

        pnl = pts * self.config.dollar_per_pt * qty

        trade = TradeRecord(
            instrument=self.config.instrument,
            side=side,
            entry_price=self.state.entry_price,
            exit_price=exit_price,
            entry_time=self.state.entry_time,
            exit_time=bar.timestamp,
            pts=pts,
            pnl_dollar=pnl,
            exit_reason=reason.value,
            bars_held=self.bar_idx - self.state.entry_bar_idx,
            strategy_id=self.strategy_id,
            qty=qty,
            entry_sm_value=self.state.entry_sm_value,
            entry_sm_velocity=0.0,
            entry_rsi_value=self.state.entry_rsi_value,
            entry_bar_volume=self.state.entry_bar_volume,
            entry_minutes_from_open=self.state.entry_minutes_from_open,
            entry_bar_range=self.state.entry_bar_range,
            concurrent_positions=self.state.concurrent_positions,
            streak_at_entry=self.state.streak_at_entry,
            exit_sm_value=round(self.sm.value, 4),
            exit_rsi_value=self.rsi_indicator.value,
            is_runner=self.state.partial_filled,
            gate_vix_close=self.state.gate_vix_close,
            gate_leledc_active=self.state.gate_leledc_active,
            gate_atr_value=self.state.gate_atr_value,
            gate_adr_ratio=self.state.gate_adr_ratio,
            gate_leledc_count=self.state.gate_leledc_count,
            ict_near_levels=self.state.ict_near_levels,
            mfe_pts=round(self.state.max_favorable, 2),
            mae_pts=round(self.state.min_adverse, 2),
            trade_group_id=self.state.trade_group_id,
            signal_price=self.state.entry_price,
        )
        self.trades.append(trade)

        # Update win/loss streak using combined P&L
        combined_pnl = pnl + self.state.partial_pnl_accum
        if combined_pnl > 0:
            self._streak = max(self._streak, 0) + 1
        elif combined_pnl < 0:
            self._streak = min(self._streak, 0) - 1

        # Reset position
        self.state.position = 0
        self.state.exit_bar_idx = self.bar_idx
        self.state.max_favorable = 0.0
        self.state.min_adverse = 0.0
        self.state.trail_activated = False
        self.state.qty_remaining = 1
        self.state.partial_filled = False
        self.state.partial_pnl_accum = 0.0
        self.state.concurrent_positions = 0

        if self.event_bus:
            self.event_bus.emit("trade_closed", trade)

        logger.info(f"[{self.strategy_id}] CLOSE {side.upper()} x{qty} @ {exit_price:.2f} "
                    f"PnL={pts:+.2f}pts (${pnl:+.2f}) reason={reason.value}")

        sig_type = SignalType.CLOSE_LONG if side == "long" else SignalType.CLOSE_SHORT
        return Signal(
            type=sig_type,
            instrument=self.config.instrument,
            reason=reason.value,
            exit_reason=reason,
            rsi_value=self.rsi_indicator.value,
        )

    def reject_entry(self) -> None:
        """Undo the state mutation from _open_position() when the runner blocks the entry."""
        if self.state.position == 0:
            return

        side = "long" if self.state.position == 1 else "short"
        logger.info(f"[{self.strategy_id}] Entry REJECTED -- reverting {side.upper()} "
                    f"@ {self.state.entry_price:.2f}")

        self.state.position = 0
        self.state.entry_price = 0.0
        self.state.entry_time = None
        self.state.qty_remaining = 1
        self.state.partial_filled = False
        self.state.max_favorable = 0.0
        self.state.min_adverse = 0.0
        self.state.trail_activated = False

    def force_close(self, bar: Bar, reason: ExitReason = ExitReason.KILL_SWITCH) -> Optional[Signal]:
        """Force close position (kill switch, connection loss, etc.)."""
        if self.state.position != 0:
            return self._close_position(bar, reason)
        return None

    def get_daily_pnl(self) -> float:
        """Sum of today's trade P&L."""
        return sum(t.pnl_dollar for t in self.trades)

    def reset_daily(self) -> None:
        """Reset daily state (call at start of new trading day).

        Does NOT clear RSI or trendline state -- they persist across days,
        matching the backtest which processes the full data range continuously.
        """
        self.trades.clear()
        # Reset pending signals to avoid stale cross-day entries
        self._pending_long = False
        self._pending_short = False
        # Safety: clear partial exit state in case position survived EOD
        if self.state.position == 0:
            self.state.partial_filled = False
            self.state.qty_remaining = 1
            self.state.max_favorable = 0.0
            self.state.min_adverse = 0.0
            self.state.trail_activated = False
