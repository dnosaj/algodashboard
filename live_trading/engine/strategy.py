"""
Incremental Strategy Engine -- bar-by-bar SM+RSI with SM-flip exits.

Converts the vectorized backtest in v10_test_common.py to an incremental
engine suitable for live trading. Each call to on_bar() processes one
1-min bar and returns a Signal.

SM is computed on 1-min bars. RSI is computed on 5-min bars and mapped
back -- matching Pine's request.security() with lookahead_off.

VALIDATION: Feed Databento CSV bar-by-bar into IncrementalStrategy and
compare output trades against run_backtest_v10(). Must be IDENTICAL.
"""

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


# ---------------------------------------------------------------------------
# Session helpers (ET minutes from midnight)
# ---------------------------------------------------------------------------

def _et_minutes_from_datetime(dt: datetime) -> int:
    """Convert a datetime to ET minutes from midnight.

    Handles EST/EDT correctly. Accepts tz-aware or naive-UTC datetimes.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    et = dt.astimezone(_ET_TZ)
    return et.hour * 60 + et.minute


def _parse_time_str(s: str) -> int:
    """Parse 'HH:MM' to minutes from midnight."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------------------
# Incremental SM computation
# ---------------------------------------------------------------------------

class IncrementalSM:
    """Incremental Smart Money index computation.

    Mirrors compute_smart_money() from v10_test_common.py exactly, but
    processes one bar at a time with running state.

    Key matching details vs vectorized code:
      - PVI/NVI start at 1.0
      - EMA initialized to arr[0] (= 1.0)
      - rsi_internal uses gain[1:period+1] for init (skips prepend delta)
      - Buy/sell use rolling sum of index_period
      - Peak uses rolling max of norm_period
    """

    def __init__(self, index_period: int, flow_period: int,
                 norm_period: int, ema_len: int):
        self.index_period = index_period
        self.flow_period = flow_period
        self.norm_period = norm_period
        self.ema_len = ema_len
        self.ema_alpha = 2.0 / (ema_len + 1)

        # PVI/NVI accumulators
        self.pvi = 1.0
        self.nvi = 1.0
        self.prev_close = None
        self.prev_volume = None

        # EMA state
        self.ema_pvi = 1.0
        self.ema_nvi = 1.0

        # Dumb/Smart previous values for delta computation
        self.dumb_prev = 0.0
        self.smart_prev = 0.0

        # Wilder RSI state for dumb and smart flow
        # In vectorized: rsi_internal(dumb, flow_period) uses np.diff(arr, prepend=arr[0])
        # delta[0] = 0 always, and ag[period] = mean(gain[1:period+1])
        # So we skip the first delta and collect the next flow_period deltas.
        self.dumb_ag = 0.0
        self.dumb_al = 0.0
        self.smart_ag = 0.0
        self.smart_al = 0.0
        self._rsi_bar_count = 0  # counts calls to _update_rsi
        self._first_delta_skipped = False
        self._dumb_init_gains = []
        self._dumb_init_losses = []
        self._smart_init_gains = []
        self._smart_init_losses = []

        # Rolling buffers for buy/sell summation
        self._r_buy_ring = deque(maxlen=index_period)
        self._r_sell_ring = deque(maxlen=index_period)

        # Rolling buffer for peak normalization
        self._mx_ring = deque(maxlen=norm_period)

        # Output
        self.value = 0.0
        self.bar_count = 0

    def update(self, close: float, volume: float) -> float:
        """Process one bar, return SM net index value."""
        self.bar_count += 1

        # --- PVI / NVI ---
        if self.prev_close is not None and self.prev_close != 0:
            pct = (close - self.prev_close) / self.prev_close
            if self.prev_volume is not None:
                if volume > self.prev_volume:
                    self.pvi = self.pvi + pct * self.pvi
                elif volume < self.prev_volume:
                    self.nvi = self.nvi + pct * self.nvi

        self.prev_close = close
        self.prev_volume = volume

        # --- EMA of PVI/NVI ---
        if self.bar_count == 1:
            self.ema_pvi = self.pvi
            self.ema_nvi = self.nvi
        else:
            self.ema_pvi = self.ema_alpha * self.pvi + (1 - self.ema_alpha) * self.ema_pvi
            self.ema_nvi = self.ema_alpha * self.nvi + (1 - self.ema_alpha) * self.ema_nvi

        # --- Dumb / Smart ---
        new_dumb = self.pvi - self.ema_pvi
        new_smart = self.nvi - self.ema_nvi

        # --- Wilder RSI on dumb/smart ---
        dumb_delta = new_dumb - self.dumb_prev
        smart_delta = new_smart - self.smart_prev
        self.dumb_prev = new_dumb
        self.smart_prev = new_smart

        drsi, srsi = self._update_rsi(dumb_delta, smart_delta)

        # --- Buy / Sell ratios ---
        r_buy = srsi / drsi if drsi != 0 else 0.0
        r_sell = (100 - srsi) / (100 - drsi) if (100 - drsi) != 0 else 0.0

        self._r_buy_ring.append(r_buy)
        self._r_sell_ring.append(r_sell)

        sb = sum(self._r_buy_ring)
        ss = sum(self._r_sell_ring)

        # --- Peak normalization ---
        mx = max(sb, ss)
        self._mx_ring.append(mx)
        pk = max(self._mx_ring) if self._mx_ring else 1.0

        ib = sb / pk if pk != 0 else 0.0
        isl = ss / pk if pk != 0 else 0.0

        self.value = ib - isl
        return self.value

    def _update_rsi(self, dumb_delta: float, smart_delta: float) -> tuple:
        """Update internal Wilder RSI for dumb/smart flow.

        Matches vectorized rsi_internal() exactly:
          delta = np.diff(arr, prepend=arr[0])  # delta[0] = 0
          ag[period] = mean(gain[1:period+1])   # skip first, use next period
          for i in range(period+1, n): Wilder smoothing

        We skip the first delta (bar 0's "prepend" equivalent, always 0),
        then collect flow_period deltas for initialization.
        """
        fp = self.flow_period

        if not self._first_delta_skipped:
            # First delta is always 0 (dumb[0] - dumb[0] in vectorized prepend).
            # Skip it, matching gain[1:period+1] which excludes index 0.
            self._first_delta_skipped = True
            # Return RSI ~99 matching vectorized (ag=0, al=0 -> rs=100 -> 99.01)
            # But since drsi ≈ srsi, the ratio r_buy ≈ 1.0 either way.
            return (99.0099, 99.0099)

        dumb_gain = max(dumb_delta, 0.0)
        dumb_loss = max(-dumb_delta, 0.0)
        smart_gain = max(smart_delta, 0.0)
        smart_loss = max(-smart_delta, 0.0)

        self._rsi_bar_count += 1

        if self._rsi_bar_count <= fp:
            # Accumulation phase: collecting flow_period deltas
            self._dumb_init_gains.append(dumb_gain)
            self._dumb_init_losses.append(dumb_loss)
            self._smart_init_gains.append(smart_gain)
            self._smart_init_losses.append(smart_loss)

            if self._rsi_bar_count == fp:
                # Initialize RSI: SMA of collected gains/losses
                self.dumb_ag = sum(self._dumb_init_gains) / fp
                self.dumb_al = sum(self._dumb_init_losses) / fp
                self.smart_ag = sum(self._smart_init_gains) / fp
                self.smart_al = sum(self._smart_init_losses) / fp

                drs = self.dumb_ag / self.dumb_al if self.dumb_al > 0 else 100.0
                drsi = 100.0 - 100.0 / (1.0 + drs)
                srs = self.smart_ag / self.smart_al if self.smart_al > 0 else 100.0
                srsi = 100.0 - 100.0 / (1.0 + srs)

                # Free init arrays
                self._dumb_init_gains = None
                self._dumb_init_losses = None
                self._smart_init_gains = None
                self._smart_init_losses = None

                return (drsi, srsi)

            # During accumulation, ag=al=0 -> rs=100 -> RSI≈99
            return (99.0099, 99.0099)
        else:
            # Wilder smoothing
            self.dumb_ag = (self.dumb_ag * (fp - 1) + dumb_gain) / fp
            self.dumb_al = (self.dumb_al * (fp - 1) + dumb_loss) / fp
            self.smart_ag = (self.smart_ag * (fp - 1) + smart_gain) / fp
            self.smart_al = (self.smart_al * (fp - 1) + smart_loss) / fp

            drs = self.dumb_ag / self.dumb_al if self.dumb_al > 0 else 100.0
            drsi = 100.0 - 100.0 / (1.0 + drs)
            srs = self.smart_ag / self.smart_al if self.smart_al > 0 else 100.0
            srsi = 100.0 - 100.0 / (1.0 + srs)

            return (drsi, srsi)


# ---------------------------------------------------------------------------
# Incremental 5-min RSI (mapped to 1-min)
# ---------------------------------------------------------------------------

class IncrementalRSI5m:
    """Computes RSI on 5-min bars, mapped to 1-min.

    Accumulates 1-min bars into 5-min OHLC. When a 5-min window ends
    (detected by next bar starting a new window), computes Wilder RSI
    on the completed 5-min close.

    Exposes curr and prev (last two completed 5-min RSI values) matching
    Pine's request.security() with lookahead_off: the values stay constant
    across all 1-min bars until the next 5-min bar completes.

    TIMING: rsi.update(bar_i) may update curr/prev if bar_i starts a new
    5-min window. The strategy captures rsi.curr/prev BEFORE calling update()
    to get the values as of bar i-1 (matching vectorized rsi_5m_curr[i-1]).
    """

    def __init__(self, rsi_len: int):
        self.rsi_len = rsi_len
        self.curr = 50.0   # Last completed 5-min RSI
        self.prev = 50.0   # The one before that

        # Wilder RSI state
        self._ag = 0.0
        self._al = 0.0
        self._prev_close_5m = None
        self._rsi_count = 0
        self._init_gains = []
        self._init_losses = []

        # 5-min bar builder
        self._5m_open = None
        self._5m_high = None
        self._5m_low = None
        self._5m_close = None
        self._current_5m_start = None  # Which 5-min window we're in

    def _get_5m_window_start(self, timestamp: datetime) -> int:
        """Get the 5-min window start as minutes from midnight (ET)."""
        et_min = _et_minutes_from_datetime(timestamp)
        return (et_min // 5) * 5

    def update(self, bar: Bar) -> None:
        """Process one 1-min bar.

        If this bar starts a new 5-min window, the previous window's bar
        is completed and RSI is updated (curr/prev shift).
        """
        window_start = self._get_5m_window_start(bar.timestamp)

        if self._current_5m_start is None or window_start != self._current_5m_start:
            # New 5-min window -- complete the old one first
            if self._5m_close is not None:
                self._on_5m_bar_complete(self._5m_close)

            # Start new 5-min bar
            self._current_5m_start = window_start
            self._5m_open = bar.open
            self._5m_high = bar.high
            self._5m_low = bar.low
            self._5m_close = bar.close
        else:
            # Continue building current 5-min bar
            self._5m_high = max(self._5m_high, bar.high)
            self._5m_low = min(self._5m_low, bar.low)
            self._5m_close = bar.close

    def _on_5m_bar_complete(self, close_5m: float) -> None:
        """Called when a 5-min bar completes. Wilder RSI update.

        Matches compute_rsi() from v10_test_common.py:
          delta = np.diff(arr, prepend=arr[0])   # delta[0] = 0
          ag[period] = mean(gain[1:period+1])    # period items from index 1
          for i in range(period+1, n): Wilder

        First call just stores prev_close (no delta). Second call onwards
        produces real deltas. After rsi_len real deltas, RSI is initialized.
        """
        if self._prev_close_5m is None:
            # First 5-min bar: just store, no delta to compute (matches bar 0)
            self._prev_close_5m = close_5m
            return

        delta = close_5m - self._prev_close_5m
        self._prev_close_5m = close_5m
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        self._rsi_count += 1
        rp = self.rsi_len

        if self._rsi_count <= rp:
            # Accumulation: collect rp deltas (matching gain[1:period+1])
            self._init_gains.append(gain)
            self._init_losses.append(loss)

            if self._rsi_count == rp:
                self._ag = sum(self._init_gains) / rp
                self._al = sum(self._init_losses) / rp
                rs = self._ag / self._al if self._al > 0 else 100.0
                rsi_val = 100.0 - 100.0 / (1.0 + rs)
                self.prev = self.curr
                self.curr = rsi_val
                self._init_gains = None
                self._init_losses = None
        else:
            # Wilder smoothing
            self._ag = (self._ag * (rp - 1) + gain) / rp
            self._al = (self._al * (rp - 1) + loss) / rp
            rs = self._ag / self._al if self._al > 0 else 100.0
            rsi_val = 100.0 - 100.0 / (1.0 + rs)
            self.prev = self.curr
            self.curr = rsi_val

    def flush(self) -> None:
        """Force-complete the current 5-min bar (e.g., at EOD)."""
        if self._5m_close is not None:
            self._on_5m_bar_complete(self._5m_close)
            self._5m_close = None
            self._current_5m_start = None


# ---------------------------------------------------------------------------
# Main Incremental Strategy
# ---------------------------------------------------------------------------

@dataclass
class TradeState:
    """Internal trade state."""
    position: int = 0        # 0=flat, 1=long, -1=short
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_bar_idx: int = 0
    long_used: bool = False
    short_used: bool = False
    exit_bar_idx: int = -9999
    sm_prev: float = 0.0    # Previous bar's SM value
    sm_prev2: float = 0.0   # Two bars ago SM value
    # Trail stop state (tp_scalp mode)
    max_favorable: float = 0.0   # Max favorable excursion in points
    trail_activated: bool = False


class IncrementalStrategy:
    """Bar-by-bar incremental strategy matching run_backtest_v10().

    CRITICAL TIMING: All signals use bar[i-1] data, fills at bar[i] open.
    - SM: state.sm_prev = SM computed from bar i-1's close/volume
    - RSI: rsi.curr/prev captured BEFORE rsi.update(bar_i), so they
      reflect the 5-min RSI state as of bar i-1
    - Stop loss: checks prev_bar.close (bar i-1), fills at bar.open (bar i)
    - EOD: checks bar_mins >= 16:00, uses bar.close for fill

    Usage:
        strategy = IncrementalStrategy(config, event_bus)
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
        self.sm = IncrementalSM(
            config.sm_index, config.sm_flow, config.sm_norm, config.sm_ema
        )
        self.rsi = IncrementalRSI5m(config.rsi_len)

        # Trade state
        self.state = TradeState()
        self.bar_idx = 0
        self.trades: list[TradeRecord] = []
        self._warming_up = True

        # Previous bar data (for stop loss checking bar i-1 close)
        self._prev_bar: Optional[Bar] = None

    @property
    def strategy_id(self) -> str:
        return self.config.strategy_id or self.config.instrument

    @property
    def position(self) -> int:
        return self.state.position

    @property
    def is_warming_up(self) -> bool:
        return self._warming_up

    def warmup(self, bar: Bar) -> None:
        """Feed a bar during warmup (updates indicators, no trading)."""
        self.sm.update(bar.close, bar.volume)
        self.rsi.update(bar)
        self.state.sm_prev2 = self.state.sm_prev
        self.state.sm_prev = self.sm.value
        self._prev_bar = bar
        self.bar_idx += 1

    def start_trading(self) -> None:
        """Switch from warmup to live trading mode."""
        self._warming_up = False
        logger.info(f"[{self.strategy_id}] Trading started. "
                    f"SM warmup bars: {self.bar_idx}, "
                    f"SM value: {self.sm.value:.4f}")

    def on_bar(self, bar: Bar) -> Signal:
        """Process one 1-min bar and return a trading signal.

        Matching run_backtest_v10() loop at bar i:
          - Signals use bar[i-1] SM and RSI (captured before update)
          - Fills at bar[i] open (or close for EOD)
          - Stop loss checks bar[i-1] close
        """
        self.bar_idx += 1

        if self._warming_up:
            self.warmup(bar)
            # warmup increments bar_idx, undo double-increment
            self.bar_idx -= 1
            return Signal(type=SignalType.NONE, instrument=self.config.instrument)

        # ------------------------------------------------------------------
        # CAPTURE signals from PREVIOUS bar BEFORE updating current bar.
        # This matches vectorized: sm[i-1], rsi_5m_curr[i-1], etc.
        # ------------------------------------------------------------------
        sm_prev = self.state.sm_prev       # SM at bar i-1
        sm_prev2 = self.state.sm_prev2     # SM at bar i-2
        rsi_curr = self.rsi.curr           # 5-min RSI at bar i-1
        rsi_prev = self.rsi.prev           # 5-min RSI before that
        prev_bar = self._prev_bar          # bar i-1 data

        # ------------------------------------------------------------------
        # UPDATE indicators with current bar (bar i)
        # ------------------------------------------------------------------
        sm_now = self.sm.update(bar.close, bar.volume)
        self.rsi.update(bar)

        # Emit bar event
        if self.event_bus:
            self.event_bus.emit("bar", bar)

        # ------------------------------------------------------------------
        # Signal computation using previous-bar data
        # ------------------------------------------------------------------
        threshold = self.config.sm_threshold

        sm_bull = sm_prev > threshold
        sm_bear = sm_prev < -threshold
        sm_flipped_bull = sm_prev > 0 and sm_prev2 <= 0
        sm_flipped_bear = sm_prev < 0 and sm_prev2 >= 0

        # RSI cross detection (5-min RSI mapped to 1-min, Pine-style)
        rsi_long_trigger = rsi_curr > self.config.rsi_buy and rsi_prev <= self.config.rsi_buy
        rsi_short_trigger = rsi_curr < self.config.rsi_sell and rsi_prev >= self.config.rsi_sell

        # Episode reset -- use zero-crossing only (not threshold).
        # With threshold > 0, `not sm_bull` flickers in the 0-to-threshold
        # zone causing repeated entries in choppy conditions.
        if sm_flipped_bull or sm_prev <= 0:
            self.state.long_used = False
        if sm_flipped_bear or sm_prev >= 0:
            self.state.short_used = False

        # Session info
        bar_mins = _et_minutes_from_datetime(bar.timestamp)
        signal = Signal(type=SignalType.NONE, instrument=self.config.instrument,
                       sm_value=sm_prev, rsi_value=rsi_curr)

        # --- EOD Close (uses bar.close like vectorized) ---
        if self.state.position != 0 and bar_mins >= self._session_close:
            signal = self._close_position(bar, ExitReason.EOD, fill_price=bar.close)
            self._update_prev(sm_now, bar)
            return signal

        # --- Exits for open positions ---
        if self.state.position != 0:
            # Max loss stop (both modes): check bar[i-1] close, fill at bar[i] open
            if self.config.max_loss_pts > 0 and prev_bar is not None:
                if self.state.position == 1 and prev_bar.close <= self.state.entry_price - self.config.max_loss_pts:
                    signal = self._close_position(bar, ExitReason.STOP_LOSS)
                    self._update_prev(sm_now, bar)
                    return signal
                elif self.state.position == -1 and prev_bar.close >= self.state.entry_price + self.config.max_loss_pts:
                    signal = self._close_position(bar, ExitReason.STOP_LOSS)
                    self._update_prev(sm_now, bar)
                    return signal

        if self.config.exit_mode == "tp_scalp" and self.state.position != 0:
            # TP/trail exits: use prev bar close (no look-ahead)
            if prev_bar is not None:
                if self.state.position == 1:
                    unrealized = prev_bar.close - self.state.entry_price
                else:
                    unrealized = self.state.entry_price - prev_bar.close

                # Update max favorable excursion
                if unrealized > self.state.max_favorable:
                    self.state.max_favorable = unrealized

                # TP exit: prev bar close reached TP target
                if self.config.tp_pts > 0 and unrealized >= self.config.tp_pts:
                    signal = self._close_position(bar, ExitReason.TAKE_PROFIT)
                    self._update_prev(sm_now, bar)
                    return signal

                # Trail exit: activated when MFE >= trail_activate, fires when
                # unrealized drops to MFE - trail_distance
                if self.state.max_favorable >= self.config.trail_activate_pts:
                    self.state.trail_activated = True
                if self.state.trail_activated:
                    trail_level = self.state.max_favorable - self.config.trail_distance_pts
                    if unrealized <= trail_level:
                        signal = self._close_position(bar, ExitReason.TRAIL_STOP)
                        self._update_prev(sm_now, bar)
                        return signal

        elif self.config.exit_mode == "sm_flip" and self.state.position != 0:
            # SM flip exits (original v11 logic)
            if self.state.position == 1:
                if sm_prev < 0 and sm_prev2 >= 0:
                    signal = self._close_position(bar, ExitReason.SM_FLIP)
                    # Don't return -- allow immediate reversal entry below
            elif self.state.position == -1:
                if sm_prev > 0 and sm_prev2 <= 0:
                    signal = self._close_position(bar, ExitReason.SM_FLIP)

        # --- Entry logic ---
        if self.state.position == 0:
            bars_since = self.bar_idx - self.state.exit_bar_idx
            in_session = self._session_start <= bar_mins <= self._session_end
            cd_ok = bars_since >= self.config.cooldown

            if in_session and cd_ok:
                if sm_bull and rsi_long_trigger and not self.state.long_used:
                    signal = self._open_position(bar, "long", sm_prev, rsi_curr)
                elif sm_bear and rsi_short_trigger and not self.state.short_used:
                    signal = self._open_position(bar, "short", sm_prev, rsi_curr)

        self._update_prev(sm_now, bar)
        return signal

    def _open_position(self, bar: Bar, side: str,
                       sm_val: float, rsi_val: float) -> Signal:
        """Open a new position at bar.open."""
        self.state.position = 1 if side == "long" else -1
        self.state.entry_price = bar.open
        self.state.entry_time = bar.timestamp
        self.state.entry_bar_idx = self.bar_idx

        if side == "long":
            self.state.long_used = True
        else:
            self.state.short_used = True

        sig_type = SignalType.BUY if side == "long" else SignalType.SELL
        signal = Signal(
            type=sig_type,
            instrument=self.config.instrument,
            reason=f"SM {'bull' if side == 'long' else 'bear'} + RSI cross",
            sm_value=sm_val,
            rsi_value=rsi_val,
        )

        if self.event_bus:
            self.event_bus.emit("signal", signal)

        logger.info(f"[{self.strategy_id}] OPEN {side.upper()} @ {bar.open:.2f} "
                    f"SM={sm_val:.4f} RSI={rsi_val:.1f}")
        return signal

    def _close_position(self, bar: Bar, reason: ExitReason,
                        fill_price: float = None) -> Signal:
        """Close current position. Fill at bar.open unless overridden."""
        side = "long" if self.state.position == 1 else "short"
        exit_price = fill_price if fill_price is not None else bar.open

        if side == "long":
            pts = exit_price - self.state.entry_price
        else:
            pts = self.state.entry_price - exit_price

        pnl = pts * self.config.dollar_per_pt

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
        )
        self.trades.append(trade)

        # Reset position BEFORE emitting trade_closed so all subscribers
        # (SafetyManager, advisors) see position == 0 during their handlers.
        self.state.position = 0
        self.state.exit_bar_idx = self.bar_idx
        self.state.max_favorable = 0.0
        self.state.trail_activated = False

        if self.event_bus:
            self.event_bus.emit("trade_closed", trade)

        logger.info(f"[{self.strategy_id}] CLOSE {side.upper()} @ {exit_price:.2f} "
                    f"PnL={pts:+.2f}pts (${pnl:+.2f}) reason={reason.value}")

        sig_type = SignalType.CLOSE_LONG if side == "long" else SignalType.CLOSE_SHORT
        return Signal(
            type=sig_type,
            instrument=self.config.instrument,
            reason=reason.value,
            exit_reason=reason,
            sm_value=self.state.sm_prev,
            rsi_value=self.rsi.curr,
        )

    def _update_prev(self, sm_now: float, bar: Bar) -> None:
        """Update previous-bar state for next iteration."""
        self.state.sm_prev2 = self.state.sm_prev
        self.state.sm_prev = sm_now
        self._prev_bar = bar

    def force_close(self, bar: Bar, reason: ExitReason = ExitReason.KILL_SWITCH) -> Optional[Signal]:
        """Force close position (kill switch, connection loss, etc.)."""
        if self.state.position != 0:
            return self._close_position(bar, reason)
        return None

    def get_daily_pnl(self) -> float:
        """Sum of today's trade P&L."""
        return sum(t.pnl_dollar for t in self.trades)

    def reset_daily(self) -> None:
        """Reset daily state (call at start of new trading day)."""
        self.trades.clear()
        self.state.long_used = False
        self.state.short_used = False
