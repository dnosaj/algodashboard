"""
Overnight snapshot: capture pre-RTH market structure and end-of-day results.

Morning snapshot (9:25 AM ET): computes overnight metrics from backfilled bars.
Evening snapshot (4:05 PM ET): fills in daily trading results.

Both write to a single CSV — one row per trading day.
"""

import csv
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Optional

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# ── Column definitions ───────────────────────────────────────────────

MORNING_COLUMNS = [
    # Metadata
    "date", "day_of_week", "coverage_hours",
    # Price structure
    "overnight_high", "overnight_low", "overnight_range_pts",
    "overnight_open", "pre_rth_price", "overnight_direction_pts",
    "position_in_range", "gap_from_prev_close",
    # Key levels
    "support_tests", "resistance_tests", "vpoc_price", "vpoc_distance_from_close",
    # European session
    "eu_open_price", "eu_direction_pts",
    # Session breakdown
    "asia_range_pts", "asia_high", "asia_low",
    "eu_range_pts", "eu_volume_ratio", "dominant_session",
    # Timing of key levels
    "overnight_high_hour_et", "overnight_low_hour_et",
    # Volatility & volume
    "avg_bar_range_pts", "max_bar_range_pts", "total_overnight_volume",
    "wide_bar_count", "volume_spike_count",
    # Overnight character
    "overnight_swings", "largest_move_pts", "up_bar_pct", "time_above_mid_pct",
    # Gap
    "gap_filled",
    # SM indicator state
    "mnq_sm_value", "mnq_sm_flips", "mes_sm_value", "mes_sm_flips",
    # External
    "vix_close",
    # Previous day carryover
    "prev_day_pnl", "prev_day_sl_count",
]

EVENING_COLUMNS = [
    # Portfolio
    "total_pnl", "trade_count", "win_count", "loss_count", "win_rate", "sl_count",
    # Per-strategy
    "v15_pnl", "v15_trades", "v15_sl_count",
    "vscalpb_pnl", "vscalpb_trades", "vscalpb_sl_count",
    "mesv2_pnl", "mesv2_trades", "mesv2_sl_count",
    # First trade
    "first_trade_strategy", "first_trade_side", "first_trade_pnl",
    # Extremes
    "best_trade_pnl", "worst_trade_pnl",
]

ALL_COLUMNS = MORNING_COLUMNS + EVENING_COLUMNS

DEFAULT_CSV_PATH = "logs/overnight_snapshots.csv"


def row_exists_for_date(csv_path: str, date_str: str) -> bool:
    """Check if a morning row already exists for this date."""
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            return any(row.get("date") == date_str for row in reader)
    except Exception:
        return False


# ── Helper functions ─────────────────────────────────────────────────

def _get_overnight_bars(bars: list, now_et: datetime) -> list:
    """Filter bars from 18:00 ET (prev session open) to now_et.

    Since the user starts the engine fresh each morning, bars in the buffer
    are the DXLink backfill. We filter to 18:00-now to be explicit.
    """
    today = now_et.date()
    # Overnight session starts at 18:00 ET the previous calendar day.
    # Sunday 18:00 starts Monday session (CME reopen after weekend).
    prev_day = today - timedelta(days=1)
    start = datetime(prev_day.year, prev_day.month, prev_day.day, 18, 0, tzinfo=_ET)

    result = []
    for bar in bars:
        bar_et = bar.timestamp.astimezone(_ET)
        if start <= bar_et <= now_et:
            result.append(bar)
    return result


def _get_asia_bars(overnight_bars: list, now_et: datetime) -> list:
    """Asia session: 18:00 ET to 03:00 ET."""
    today = now_et.date()
    cutoff = datetime(today.year, today.month, today.day, 3, 0, tzinfo=_ET)
    return [b for b in overnight_bars if b.timestamp.astimezone(_ET) < cutoff]


def _get_eu_bars(overnight_bars: list, now_et: datetime) -> list:
    """European session: 03:00 ET to 09:25 ET."""
    today = now_et.date()
    eu_start = datetime(today.year, today.month, today.day, 3, 0, tzinfo=_ET)
    return [b for b in overnight_bars if b.timestamp.astimezone(_ET) >= eu_start]


def _compute_coverage_hours(bars: list) -> float:
    """Hours between first and last bar in the list."""
    if len(bars) < 2:
        return 0.0
    delta = bars[-1].timestamp - bars[0].timestamp
    return delta.total_seconds() / 3600.0


def _count_level_tests(bars: list, level: float, is_support: bool = True,
                       tolerance_pts: float = 3.0, min_gap: int = 5) -> int:
    """Count distinct visits near a price level.

    For support: checks bar.low near level. For resistance: bar.high near level.
    A new test requires at least min_gap bars since the last test.
    tolerance_pts: fixed point distance (default 3.0 — close to TP=5 scalp width).
    """
    if not bars or level == 0:
        return 0
    tolerance = tolerance_pts
    count = 0
    last_test_idx = -min_gap

    for i, bar in enumerate(bars):
        test_price = bar.low if is_support else bar.high
        if abs(test_price - level) <= tolerance:
            if i - last_test_idx >= min_gap:
                count += 1
                last_test_idx = i

    return count


def _compute_vpoc(bars: list, bin_size: float = 5.0) -> Optional[float]:
    """Volume Profile Point of Control — highest-volume price bucket center."""
    if not bars:
        return None

    bins: dict[float, float] = {}
    for bar in bars:
        mid = (bar.high + bar.low) / 2.0
        bucket = round(mid / bin_size) * bin_size
        bins[bucket] = bins.get(bucket, 0.0) + bar.volume

    if not bins:
        return None

    return max(bins, key=bins.get)


def _count_sm_flips(overnight_bars: list, sm_params: tuple) -> int:
    """Compute SM on overnight bar series and count zero-crossings.

    Feeds all bars through a fresh IncrementalSM. Only counts flips after
    warmup period (max of norm_period, ema_len) so early values are stable.

    sm_params: (index_period, flow_period, norm_period, ema_len)
    """
    from engine.strategy import IncrementalSM

    index_period, flow_period, norm_period, ema_len = sm_params
    sm = IncrementalSM(
        index_period=index_period,
        flow_period=flow_period,
        norm_period=norm_period,
        ema_len=ema_len,
    )

    flips = 0
    prev_value = 0.0
    warmup = max(norm_period, ema_len)

    for i, bar in enumerate(overnight_bars):
        value = sm.update(bar.close, bar.volume)

        if i >= warmup and prev_value != 0.0:
            if (prev_value > 0 and value < 0) or (prev_value < 0 and value > 0):
                flips += 1

        prev_value = value

    return flips


def _count_swings(bars: list, min_move: float = 5.0) -> int:
    """Count directional reversals (peaks/troughs) in close prices.

    A swing occurs when price reverses min_move pts from a running extreme.
    """
    if len(bars) < 2:
        return 0

    swings = 0
    direction = 0  # 0=undecided, 1=up, -1=down
    extreme = bars[0].close

    for bar in bars[1:]:
        price = bar.close
        if direction == 0:
            if price - extreme >= min_move:
                direction = 1
                extreme = price
            elif extreme - price >= min_move:
                direction = -1
                extreme = price
        elif direction == 1:
            if price > extreme:
                extreme = price
            elif extreme - price >= min_move:
                swings += 1
                direction = -1
                extreme = price
        elif direction == -1:
            if price < extreme:
                extreme = price
            elif price - extreme >= min_move:
                swings += 1
                direction = 1
                extreme = price

    return swings


def _largest_directional_run(bars: list) -> float:
    """Largest consecutive same-direction move in pts."""
    if len(bars) < 2:
        return 0.0

    max_run = 0.0
    run_start = bars[0].close
    direction = 0

    for i in range(1, len(bars)):
        delta = bars[i].close - bars[i - 1].close
        if delta > 0:
            new_dir = 1
        elif delta < 0:
            new_dir = -1
        else:
            new_dir = direction

        if new_dir != direction and direction != 0:
            run = abs(bars[i - 1].close - run_start)
            max_run = max(max_run, run)
            run_start = bars[i - 1].close

        direction = new_dir

    # Final run
    run = abs(bars[-1].close - run_start)
    max_run = max(max_run, run)

    return max_run


def _get_prev_day_from_csv(csv_path: str, today_str: str) -> tuple:
    """Read previous day's P&L and SL count from the CSV.

    Returns (prev_day_pnl, prev_day_sl_count) or (None, None).
    """
    if not os.path.exists(csv_path):
        return None, None

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None, None

    from datetime import date as date_type
    try:
        today_date = date_type.fromisoformat(today_str)
    except ValueError:
        return None, None

    for row in reversed(rows):
        row_date_str = row.get("date", "")
        if row_date_str < today_str and row.get("total_pnl"):
            try:
                row_date = date_type.fromisoformat(row_date_str)
                if (today_date - row_date).days > 4:
                    return None, None  # Stale — skip data older than a long weekend
                pnl = float(row["total_pnl"])
                sl = int(row.get("sl_count", 0) or 0)
                return pnl, sl
            except (ValueError, TypeError):
                return None, None

    return None, None


def _find_prev_close(bars: list, now_et: datetime) -> Optional[float]:
    """Find yesterday's ~16:00 ET close from bar buffer (if available)."""
    today = now_et.date()
    prev_day = today - timedelta(days=1)
    if today.weekday() == 0:
        prev_day = today - timedelta(days=3)  # Friday for Monday

    target_start = datetime(prev_day.year, prev_day.month, prev_day.day,
                            15, 55, tzinfo=_ET)
    target_end = datetime(prev_day.year, prev_day.month, prev_day.day,
                          16, 5, tzinfo=_ET)

    best = None
    for bar in bars:
        bar_et = bar.timestamp.astimezone(_ET)
        if target_start <= bar_et <= target_end:
            best = bar.close  # Take the latest one in the window

    return best


# ── Main compute functions ───────────────────────────────────────────

def compute_morning_snapshot(bars_by_instrument: dict, strategies: dict,
                             vix_close: Optional[float],
                             csv_path: str = DEFAULT_CSV_PATH) -> Optional[dict]:
    """Compute overnight metrics from backfilled bars + strategy SM state.

    Args:
        bars_by_instrument: dict[str, deque[Bar]] from data_feed._all_bars
        strategies: dict[str, IncrementalStrategy] from state.strategies
        vix_close: prior-day VIX close (from safety._vix_close)
        csv_path: path to snapshot CSV (for prev-day carryover)

    Returns:
        dict with all morning columns, or None if no bars available.
    """
    now_et = datetime.now(_ET)
    today_str = now_et.strftime("%Y-%m-%d")
    day_of_week = now_et.strftime("%a")

    # Primary instrument: MNQ
    mnq_all = list(bars_by_instrument.get("MNQ", []))
    overnight = _get_overnight_bars(mnq_all, now_et)

    if not overnight:
        logger.warning("[Snapshot] No overnight bars available for MNQ")
        return None

    # ── Coverage ──
    coverage_hours = _compute_coverage_hours(overnight)
    if coverage_hours < 10:
        logger.warning(f"[Snapshot] Low coverage: {coverage_hours:.1f}h (expected ≥10)")

    # ── Price structure ──
    overnight_high = max(b.high for b in overnight)
    overnight_low = min(b.low for b in overnight)
    overnight_range = overnight_high - overnight_low
    overnight_open = overnight[0].close
    pre_rth_price = overnight[-1].close
    overnight_direction = pre_rth_price - overnight_open
    position_in_range = ((pre_rth_price - overnight_low) / overnight_range
                         if overnight_range > 0 else 0.5)

    # Gap from previous close
    prev_close = _find_prev_close(mnq_all, now_et)
    gap_from_prev_close = (pre_rth_price - prev_close) if prev_close else ""

    # ── Key levels ──
    support_tests = _count_level_tests(overnight, overnight_low, is_support=True)
    resistance_tests = _count_level_tests(overnight, overnight_high, is_support=False)
    vpoc = _compute_vpoc(overnight)
    vpoc_distance = (vpoc - pre_rth_price) if vpoc else ""

    # ── Session breakdown ──
    asia = _get_asia_bars(overnight, now_et)
    eu = _get_eu_bars(overnight, now_et)

    asia_high = max(b.high for b in asia) if asia else ""
    asia_low = min(b.low for b in asia) if asia else ""
    asia_range = (asia_high - asia_low) if asia else ""

    eu_high = max(b.high for b in eu) if eu else ""
    eu_low = min(b.low for b in eu) if eu else ""
    eu_range = (eu_high - eu_low) if eu else ""

    total_vol = sum(b.volume for b in overnight)
    eu_vol = sum(b.volume for b in eu) if eu else 0
    eu_volume_ratio = round(eu_vol / total_vol, 4) if total_vol > 0 else ""

    if isinstance(asia_range, (int, float)) and isinstance(eu_range, (int, float)):
        dominant_session = "eu" if eu_range > asia_range else "asia"
    else:
        dominant_session = ""

    # EU open price (first bar at/after 03:00 ET)
    eu_open_price = eu[0].close if eu else ""
    eu_direction = (pre_rth_price - eu_open_price) if eu else ""

    # ── Timing of key levels ──
    high_bar = max(overnight, key=lambda b: b.high)
    low_bar = min(overnight, key=lambda b: b.low)
    overnight_high_hour = high_bar.timestamp.astimezone(_ET).hour
    overnight_low_hour = low_bar.timestamp.astimezone(_ET).hour

    # ── Volatility & volume ──
    bar_ranges = [b.high - b.low for b in overnight]
    avg_bar_range = sum(bar_ranges) / len(bar_ranges) if bar_ranges else 0
    max_bar_range = max(bar_ranges) if bar_ranges else 0
    wide_bar_count = sum(1 for r in bar_ranges if r > 2 * avg_bar_range) if avg_bar_range > 0 else 0

    bar_volumes = [b.volume for b in overnight]
    avg_volume = sum(bar_volumes) / len(bar_volumes) if bar_volumes else 0
    volume_spike_count = sum(1 for v in bar_volumes if v > 2 * avg_volume) if avg_volume > 0 else 0

    # ── Overnight character ──
    swings = _count_swings(overnight)
    largest_move = _largest_directional_run(overnight)
    up_bars = sum(1 for b in overnight if b.close > b.open)
    up_bar_pct = round(up_bars / len(overnight), 4) if overnight else 0

    range_mid = (overnight_high + overnight_low) / 2
    above_mid = sum(1 for b in overnight if b.close > range_mid)
    time_above_mid_pct = round(above_mid / len(overnight), 4) if overnight else 0

    # Gap filled?
    gap_filled = ""
    if prev_close is not None:
        if prev_close >= overnight_low and prev_close <= overnight_high:
            gap_filled = True
        else:
            gap_filled = False

    # ── SM indicator state ──
    mnq_sm_value = ""
    mes_sm_value = ""
    mnq_sm_params = None
    mes_sm_params = None

    for _sid, strat in strategies.items():
        inst = strat.config.instrument
        if inst == "MNQ" and mnq_sm_value == "":
            mnq_sm_value = round(strat.sm.value, 6)
            mnq_sm_params = (strat.config.sm_index, strat.config.sm_flow,
                             strat.config.sm_norm, strat.config.sm_ema)
        elif inst == "MES" and mes_sm_value == "":
            mes_sm_value = round(strat.sm.value, 6)
            mes_sm_params = (strat.config.sm_index, strat.config.sm_flow,
                             strat.config.sm_norm, strat.config.sm_ema)

    mnq_sm_flips = _count_sm_flips(overnight, mnq_sm_params) if mnq_sm_params else 0
    mes_all = list(bars_by_instrument.get("MES", []))
    mes_overnight = _get_overnight_bars(mes_all, now_et)
    mes_sm_flips = _count_sm_flips(mes_overnight, mes_sm_params) if mes_sm_params and mes_overnight else 0

    # ── Previous day carryover ──
    prev_pnl, prev_sl = _get_prev_day_from_csv(csv_path, today_str)

    # ── Build snapshot dict ──
    snapshot = {
        "date": today_str,
        "day_of_week": day_of_week,
        "coverage_hours": round(coverage_hours, 2),
        "overnight_high": round(overnight_high, 2),
        "overnight_low": round(overnight_low, 2),
        "overnight_range_pts": round(overnight_range, 2),
        "overnight_open": round(overnight_open, 2),
        "pre_rth_price": round(pre_rth_price, 2),
        "overnight_direction_pts": round(overnight_direction, 2),
        "position_in_range": round(position_in_range, 4),
        "gap_from_prev_close": round(gap_from_prev_close, 2) if isinstance(gap_from_prev_close, (int, float)) else "",
        "support_tests": support_tests,
        "resistance_tests": resistance_tests,
        "vpoc_price": round(vpoc, 2) if vpoc else "",
        "vpoc_distance_from_close": round(vpoc_distance, 2) if isinstance(vpoc_distance, (int, float)) else "",
        "eu_open_price": round(eu_open_price, 2) if isinstance(eu_open_price, (int, float)) else "",
        "eu_direction_pts": round(eu_direction, 2) if isinstance(eu_direction, (int, float)) else "",
        "asia_range_pts": round(asia_range, 2) if isinstance(asia_range, (int, float)) else "",
        "asia_high": round(asia_high, 2) if isinstance(asia_high, (int, float)) else "",
        "asia_low": round(asia_low, 2) if isinstance(asia_low, (int, float)) else "",
        "eu_range_pts": round(eu_range, 2) if isinstance(eu_range, (int, float)) else "",
        "eu_volume_ratio": eu_volume_ratio,
        "dominant_session": dominant_session,
        "overnight_high_hour_et": overnight_high_hour,
        "overnight_low_hour_et": overnight_low_hour,
        "avg_bar_range_pts": round(avg_bar_range, 4),
        "max_bar_range_pts": round(max_bar_range, 2),
        "total_overnight_volume": round(total_vol, 0),
        "wide_bar_count": wide_bar_count,
        "volume_spike_count": volume_spike_count,
        "overnight_swings": swings,
        "largest_move_pts": round(largest_move, 2),
        "up_bar_pct": up_bar_pct,
        "time_above_mid_pct": time_above_mid_pct,
        "gap_filled": gap_filled,
        "mnq_sm_value": mnq_sm_value,
        "mnq_sm_flips": mnq_sm_flips,
        "mes_sm_value": mes_sm_value,
        "mes_sm_flips": mes_sm_flips,
        "vix_close": round(vix_close, 2) if vix_close is not None else "",
        "prev_day_pnl": round(prev_pnl, 2) if prev_pnl is not None else "",
        "prev_day_sl_count": prev_sl if prev_sl is not None else "",
    }

    return snapshot


def compute_evening_snapshot(strategies: dict) -> dict:
    """Compute daily trading results from strategy trade lists.

    Args:
        strategies: dict[str, IncrementalStrategy] from state.strategies

    Returns:
        dict with all evening columns.
    """
    all_trades = []
    for strat in strategies.values():
        all_trades.extend(strat.trades)

    # Sort by exit time
    all_trades.sort(key=lambda t: t.exit_time or datetime.min)

    # Total P&L includes ALL records (partials contribute real P&L)
    total_pnl = sum(t.pnl_dollar for t in all_trades)

    # Counts exclude partial exits (is_partial=True) — one entry = one trade
    closed_trades = [t for t in all_trades if not t.is_partial]
    trade_count = len(closed_trades)
    win_count = sum(1 for t in closed_trades if t.pnl_dollar > 0)
    loss_count = sum(1 for t in closed_trades if t.pnl_dollar < 0)
    win_rate = win_count / trade_count if trade_count > 0 else 0.0
    sl_count = sum(1 for t in closed_trades if t.exit_reason == "SL")

    # Per-strategy breakdown (P&L from all records, counts from non-partial)
    strategy_map = {"MNQ_V15": "v15", "MNQ_VSCALPB": "vscalpb", "MES_V2": "mesv2"}
    per_strat = {}
    for sid, prefix in strategy_map.items():
        strat_all = [t for t in all_trades if t.strategy_id == sid]
        strat_closed = [t for t in strat_all if not t.is_partial]
        per_strat[f"{prefix}_pnl"] = round(sum(t.pnl_dollar for t in strat_all), 2)
        per_strat[f"{prefix}_trades"] = len(strat_closed)
        per_strat[f"{prefix}_sl_count"] = sum(1 for t in strat_closed if t.exit_reason == "SL")

    # First trade (non-partial — first completed entry)
    first = closed_trades[0] if closed_trades else None

    # Best/worst trade (non-partial)
    pnls = [t.pnl_dollar for t in closed_trades]
    best = max(pnls) if pnls else 0
    worst = min(pnls) if pnls else 0

    return {
        "total_pnl": round(total_pnl, 2),
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 4),
        "sl_count": sl_count,
        **per_strat,
        "first_trade_strategy": first.strategy_id if first else "",
        "first_trade_side": first.side.upper() if first else "",
        "first_trade_pnl": round(first.pnl_dollar, 2) if first else "",
        "best_trade_pnl": round(best, 2),
        "worst_trade_pnl": round(worst, 2),
    }


# ── CSV I/O ──────────────────────────────────────────────────────────

def write_morning_snapshot(snapshot: dict, csv_path: str = DEFAULT_CSV_PATH) -> None:
    """Append new row with morning columns filled, evening columns blank."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    # Build full row: morning data + empty evening columns
    row = {col: "" for col in ALL_COLUMNS}
    row.update(snapshot)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"[Snapshot] Morning row written to {csv_path}")


def update_evening_snapshot(evening_data: dict, csv_path: str = DEFAULT_CSV_PATH,
                            date_str: str = "") -> None:
    """Read CSV, find today's row, fill evening columns, write back."""
    if not date_str:
        date_str = datetime.now(_ET).strftime("%Y-%m-%d")

    if not os.path.exists(csv_path):
        logger.warning(f"[Snapshot] CSV not found: {csv_path}")
        return

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        logger.warning(f"[Snapshot] Failed to read CSV for evening update: {e}")
        return

    updated = False
    for row in rows:
        if row.get("date") == date_str:
            row.update(evening_data)
            updated = True
            break

    if not updated:
        logger.warning(f"[Snapshot] No morning row found for {date_str}, skipping evening update")
        return

    # Atomic write: temp file + rename to prevent data loss on crash
    csv_dir = os.path.dirname(csv_path) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=csv_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)  # Atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info(f"[Snapshot] Evening columns updated for {date_str}")
