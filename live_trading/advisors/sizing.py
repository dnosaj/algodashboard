"""
Fixed-size position sizing advisor.

Day 1: always returns qty=1. Structured so it is straightforward to
add Kelly criterion, volatility scaling, ML-based sizing, or drawdown
throttling later by subclassing or extending this advisor.

Tracks win rate, average win/loss, and streak stats from completed trades
to prepare for future adaptive sizing.
"""

import logging
from dataclasses import dataclass, field

from advisors.base import Advisor
from engine.events import Bar, PreOrderContext, Signal, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class SizingStats:
    """Running statistics for position sizing decisions."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_win_pts: float = 0.0
    total_loss_pts: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    last_trade_won: bool = False

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def avg_win_pts(self) -> float:
        if self.wins == 0:
            return 0.0
        return self.total_win_pts / self.wins

    @property
    def avg_loss_pts(self) -> float:
        if self.losses == 0:
            return 0.0
        return self.total_loss_pts / self.losses

    @property
    def kelly_fraction(self) -> float:
        """Kelly criterion: f* = (bp - q) / b where b = avg_win/avg_loss.

        Returns the optimal fraction of capital to risk. Capped at 0.25
        for safety (quarter-Kelly is standard practice).
        """
        if self.wins < 5 or self.losses < 5:
            return 0.0  # Not enough data
        if self.avg_loss_pts == 0:
            return 0.0
        b = self.avg_win_pts / self.avg_loss_pts
        p = self.win_rate
        q = 1.0 - p
        kelly = (b * p - q) / b
        return max(0.0, min(kelly, 0.25))


class FixedSizeAdvisor(Advisor):
    """Fixed position size advisor.

    Always recommends qty=1 for Day 1 deployment. Tracks trade statistics
    that would feed into more sophisticated sizing algorithms later.

    The advisor subscribes to:
        - "pre_order" events to set qty
        - "trade_closed" events to update win/loss stats
        - "bar" events (no-op for now, placeholder for vol tracking)
    """

    name = "fixed_size"

    def __init__(self, qty: int = 1):
        self._qty = qty
        self._stats: dict[str, SizingStats] = {}  # Per-instrument stats

    def _get_stats(self, instrument: str) -> SizingStats:
        """Get or create stats for an instrument."""
        if instrument not in self._stats:
            self._stats[instrument] = SizingStats()
        return self._stats[instrument]

    def on_signal(self, signal: Signal, context: PreOrderContext) -> dict:
        """Return fixed qty. No veto logic for Day 1."""
        stats = self._get_stats(context.instrument)

        logger.debug(
            f"[{self.name}] {context.instrument} {context.side} signal. "
            f"WR={stats.win_rate:.1%}, streak={'W' if stats.last_trade_won else 'L'}"
            f"{stats.consecutive_wins if stats.last_trade_won else stats.consecutive_losses}. "
            f"qty={self._qty}"
        )

        return {"qty": self._qty}

    def on_bar(self, bar: Bar) -> None:
        """No-op for Day 1. Future: track rolling volatility for vol-scaling."""
        pass

    def on_trade_closed(self, trade: TradeRecord) -> None:
        """Update win/loss statistics from completed trade."""
        stats = self._get_stats(trade.instrument)
        stats.total_trades += 1

        # Commission-adjusted P&L check (using raw pts for simplicity --
        # trade.pnl_dollar already accounts for dollar_per_pt but not commission)
        won = trade.pts > 0

        if won:
            stats.wins += 1
            stats.total_win_pts += trade.pts
            stats.consecutive_wins += 1
            stats.consecutive_losses = 0
            if stats.consecutive_wins > stats.max_consecutive_wins:
                stats.max_consecutive_wins = stats.consecutive_wins
        else:
            stats.losses += 1
            stats.total_loss_pts += abs(trade.pts)
            stats.consecutive_losses += 1
            stats.consecutive_wins = 0
            if stats.consecutive_losses > stats.max_consecutive_losses:
                stats.max_consecutive_losses = stats.consecutive_losses

        stats.last_trade_won = won

        logger.info(
            f"[{self.name}] {trade.instrument} trade #{stats.total_trades}: "
            f"{'WIN' if won else 'LOSS'} {trade.pts:+.2f}pts. "
            f"WR={stats.win_rate:.1%} ({stats.wins}W/{stats.losses}L), "
            f"AvgW={stats.avg_win_pts:.1f} AvgL={stats.avg_loss_pts:.1f}, "
            f"Kelly={stats.kelly_fraction:.3f}"
        )

    def get_status(self) -> dict:
        """Return current sizing state for dashboard."""
        per_instrument = {}
        for instrument, stats in self._stats.items():
            per_instrument[instrument] = {
                "total_trades": stats.total_trades,
                "wins": stats.wins,
                "losses": stats.losses,
                "win_rate": round(stats.win_rate * 100, 1),
                "avg_win_pts": round(stats.avg_win_pts, 2),
                "avg_loss_pts": round(stats.avg_loss_pts, 2),
                "consecutive_wins": stats.consecutive_wins,
                "consecutive_losses": stats.consecutive_losses,
                "max_consecutive_wins": stats.max_consecutive_wins,
                "max_consecutive_losses": stats.max_consecutive_losses,
                "kelly_fraction": round(stats.kelly_fraction, 4),
            }

        return {
            "name": self.name,
            "active": True,
            "qty": self._qty,
            "instruments": per_instrument,
        }
