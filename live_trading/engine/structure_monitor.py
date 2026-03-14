"""
Structure-based exit monitor for runner legs.

Provides IncrementalSwingTracker (pivot-based swing detection, one bar at a time)
and StructureExitMonitor (checks runner exit conditions against swing levels).

Used by vScalpC only. Disabled for all other strategies (structure_exit_type="").

CRITICAL: IncrementalSwingTracker must produce identical output to the vectorized
compute_swing_levels() in backtesting_engine/strategies/structure_exit_common.py.
Parity-tested on full 12.5-month MNQ CSV.

Timing convention:
  - Capture swing levels BEFORE calling tracker.update(bar_i)
  - This gives us levels as of bar[i-1], matching the backtest convention
  - Exit decision uses bar[i-1] close vs swing level, fills at bar[i] open
"""

import logging
from collections import deque
from typing import Optional

from .config import StrategyConfig
from .events import Bar, ExitReason

logger = logging.getLogger(__name__)


class IncrementalSwingTracker:
    """Incremental pivot-based swing high/low tracker.

    Processes one bar at a time via update(). After each update, exposes
    swing_high and swing_low — the most recent confirmed pivot levels.

    Pivot detection:
      A swing high at bar j requires:
        highs[j] >= highs[j-k] for k = 1..left
        highs[j] >= highs[j+k] for k = 1..right
      Confirmed at bar j + right (no look-ahead).

    Deque-based rolling buffer: maxlen = left + right + 1.
    """

    def __init__(self, lookback: int, pivot_right: int = 2):
        self.left = lookback
        self.right = pivot_right
        self._buf_size = lookback + pivot_right + 1

        # Rolling buffers for high/low prices
        self._highs: deque = deque(maxlen=self._buf_size)
        self._lows: deque = deque(maxlen=self._buf_size)

        # Confirmed swing levels (NaN until first pivot confirmed)
        self.swing_high: Optional[float] = None
        self.swing_low: Optional[float] = None

        # Bar counter (total bars fed)
        self._bar_count: int = 0

    def update(self, high: float, low: float) -> None:
        """Process one bar. Updates swing_high/swing_low if a new pivot is confirmed.

        The candidate pivot is at position j = len(buffer) - 1 - right in the
        current buffer BEFORE appending the new bar. But since we need 'right'
        bars to the right of the candidate, we check after appending.

        After appending bar i, the buffer holds the last (left+right+1) bars.
        The candidate is at index left (counting from 0) in the buffer,
        which is bar i - right.
        """
        self._highs.append(high)
        self._lows.append(low)
        self._bar_count += 1

        # Need at least left + right + 1 bars before we can check pivots
        if len(self._highs) < self._buf_size:
            return

        # Candidate is at buffer index = left (the middle of the window)
        cand_idx = self.left
        cand_high = self._highs[cand_idx]
        cand_low = self._lows[cand_idx]

        # Check swing high: candidate must be >= all left AND right neighbors
        is_sh = True
        for k in range(1, self.left + 1):
            if self._highs[cand_idx - k] > cand_high:
                is_sh = False
                break
        if is_sh:
            for k in range(1, self.right + 1):
                if self._highs[cand_idx + k] > cand_high:
                    is_sh = False
                    break
        if is_sh:
            self.swing_high = cand_high

        # Check swing low: candidate must be <= all left AND right neighbors
        is_sl = True
        for k in range(1, self.left + 1):
            if self._lows[cand_idx - k] < cand_low:
                is_sl = False
                break
        if is_sl:
            for k in range(1, self.right + 1):
                if self._lows[cand_idx + k] < cand_low:
                    is_sl = False
                    break
        if is_sl:
            self.swing_low = cand_low

    def reset(self) -> None:
        """Clear all state (for daily reset if needed)."""
        self._highs.clear()
        self._lows.clear()
        self.swing_high = None
        self.swing_low = None
        self._bar_count = 0


class StructureExitMonitor:
    """Monitors runner positions for structure-based exits.

    Called once per bar from runner.process_bar(), AFTER strategy.on_bar()
    has processed entries/exits/indicators.

    Exit logic (matching backtest in structure_exit_common.py):
      - LONG runner: exit when close[i-1] >= swing_high - buffer
      - SHORT runner: exit when close[i-1] <= swing_low + buffer
      - Only fires on runner legs (partial_filled == True)
      - Only fires if structure_exit_type is set

    Does NOT fire if:
      - No position open
      - Position is still in scalp leg (partial not yet filled)
      - No confirmed swing level exists yet
      - Swing level would exit at a loss (structure only exits profitably by design,
        since swing levels are recent extremes — runner profit is virtually guaranteed
        when close reaches swing level)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._enabled = bool(config.structure_exit_type)

        if self._enabled:
            self.tracker = IncrementalSwingTracker(
                lookback=config.structure_exit_lookback,
                pivot_right=config.structure_exit_pivot_right,
            )
            self.buffer_pts = config.structure_exit_buffer_pts
        else:
            self.tracker = None
            self.buffer_pts = 0.0

        # Current swing levels (captured BEFORE tracker.update for bar[i-1] convention)
        self._swing_high_prev: Optional[float] = None
        self._swing_low_prev: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        logger.info(f"[StructureExit] Monitor {'enabled' if enabled else 'disabled'}")

    @property
    def current_swing_high(self) -> Optional[float]:
        """Most recent confirmed swing high (as of last bar processed)."""
        return self._swing_high_prev

    @property
    def current_swing_low(self) -> Optional[float]:
        """Most recent confirmed swing low (as of last bar processed)."""
        return self._swing_low_prev

    def warmup(self, bar: Bar) -> None:
        """Feed a warmup bar to build swing history. No exit checks."""
        if not self._enabled:
            return
        # Capture prev BEFORE update (bar[i-1] convention)
        self._swing_high_prev = self.tracker.swing_high
        self._swing_low_prev = self.tracker.swing_low
        self.tracker.update(bar.high, bar.low)

    def check_exit(self, bar: Bar, position: int, entry_price: float,
                   partial_filled: bool, prev_bar: Optional[Bar]) -> Optional[dict]:
        """Check if structure exit should fire on this bar.

        Called from runner.process_bar() AFTER strategy.on_bar().

        Args:
            bar: Current bar (bar i) — used for fill price (bar.open)
            position: 1 (long) or -1 (short), 0 = flat
            entry_price: Entry price of the position
            partial_filled: True if TP1 has filled (this is the runner leg)
            prev_bar: Previous bar (bar i-1) — used for close comparison

        Returns:
            dict with exit info if structure exit fires, None otherwise.
            Keys: exit_reason (ExitReason), exit_price (float), structure_level (float)
        """
        # Always update tracker (keeps swing levels current even when exits disabled,
        # so re-enabling doesn't produce stale levels for 53 bars)
        self._swing_high_prev = self.tracker.swing_high
        self._swing_low_prev = self.tracker.swing_low
        self.tracker.update(bar.high, bar.low)

        if not self._enabled:
            return None

        # Only exit runner legs (after TP1 partial fill)
        if position == 0 or not partial_filled:
            return None

        if prev_bar is None:
            return None

        # LONG runner: exit when close[i-1] >= swing_high - buffer
        if position == 1 and self._swing_high_prev is not None:
            target = self._swing_high_prev - self.buffer_pts
            if prev_bar.close >= target:
                logger.info(
                    f"[StructureExit] LONG runner exit: close={prev_bar.close:.2f} "
                    f">= target={target:.2f} (swing_high={self._swing_high_prev:.2f} "
                    f"- buffer={self.buffer_pts})"
                )
                return {
                    "exit_reason": ExitReason.STRUCTURE,
                    "exit_price": bar.open,
                    "structure_level": self._swing_high_prev,
                }

        # SHORT runner: exit when close[i-1] <= swing_low + buffer
        if position == -1 and self._swing_low_prev is not None:
            target = self._swing_low_prev + self.buffer_pts
            if prev_bar.close <= target:
                logger.info(
                    f"[StructureExit] SHORT runner exit: close={prev_bar.close:.2f} "
                    f"<= target={target:.2f} (swing_low={self._swing_low_prev:.2f} "
                    f"+ buffer={self.buffer_pts})"
                )
                return {
                    "exit_reason": ExitReason.STRUCTURE,
                    "exit_price": bar.open,
                    "structure_level": self._swing_low_prev,
                }

        return None

    # ------------------------------------------------------------------
    # Observation data (for Supabase per-bar logging)
    # ------------------------------------------------------------------

    _NEAR_MISS_PTS = 3.0  # Fixed "near miss zone" threshold

    def get_bar_observation(self, bar: Bar, position: int, entry_price: float,
                            partial_filled: bool, prev_bar: Optional[Bar]) -> Optional[dict]:
        """Compute observation data for the current bar.

        Called on every bar when a runner is active (partial_filled=True).
        Returns dict for Supabase logging, or None if not applicable.

        Uses bar[i-1] convention: bar_close is prev_bar.close, swing levels are
        _swing_high_prev/_swing_low_prev (captured before tracker update).
        """
        if not self._enabled:
            return None
        if position == 0 or not partial_filled:
            return None
        if prev_bar is None:
            return None

        close = prev_bar.close
        sh = self._swing_high_prev
        sl = self._swing_low_prev

        # Compute unrealized runner profit (in points)
        if position == 1:
            runner_profit_pts = close - entry_price
        else:
            runner_profit_pts = entry_price - close

        # Compute distance to relevant swing level and near-miss flag
        distance_to_level = None
        near_miss = False

        if position == 1 and sh is not None:
            target = sh - self.buffer_pts
            distance_to_level = target - close  # positive = below target
            # Near miss: within 3pts below target but didn't trigger
            near_miss = (distance_to_level > 0 and distance_to_level <= self._NEAR_MISS_PTS)

        elif position == -1 and sl is not None:
            target = sl + self.buffer_pts
            distance_to_level = close - target  # positive = above target
            # Near miss: within 3pts above target but didn't trigger
            near_miss = (distance_to_level > 0 and distance_to_level <= self._NEAR_MISS_PTS)

        return {
            "bar_time": bar.timestamp,
            "instrument": bar.instrument,
            "bar_close": close,
            "swing_high": sh,
            "swing_low": sl,
            "position": position,
            "entry_price": entry_price,
            "runner_profit_pts": round(runner_profit_pts, 2),
            "distance_to_level_pts": round(distance_to_level, 2) if distance_to_level is not None else None,
            "near_miss": near_miss,
        }
