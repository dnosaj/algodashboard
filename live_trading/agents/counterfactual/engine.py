"""Counterfactual Engine — orchestrates fetch, interleave, simulate, write.

Fetches unfilled blocked signals from Supabase, loads bar data from session
JSONs (primary) or Databento CSVs (fallback), runs cooldown-aware chronological
simulation, and writes results back.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from .exit_simulator import simulate_single_exit, simulate_partial_exit

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
SESSIONS_DIR = Path(__file__).resolve().parents[2] / "sessions"
DATA_DIR = Path(__file__).resolve().parents[3] / "backtesting_engine" / "data"

# Strategy configs (imported from engine config — pure dataclasses, no side effects)
from engine.config import MNQ_V15, MNQ_VSCALPB, MNQ_VSCALPC, MES_V2

STRATEGY_CONFIGS = {
    "MNQ_V15": MNQ_V15,
    "MNQ_VSCALPB": MNQ_VSCALPB,
    "MNQ_VSCALPC": MNQ_VSCALPC,
    "MES_V2": MES_V2,
}

# Commission per-contract roundtrip in POINTS
# MNQ: $1.04 / $2.00/pt = 0.52 pts.  MES: $2.50 / $5.00/pt = 0.50 pts.
_COMMISSION_PTS = {
    "MNQ": 0.52,
    "MES": 0.50,
}


class CounterfactualEngine:
    """Nightly batch engine for counterfactual trade simulation."""

    def __init__(self, dry_run=False, force=False, cooldown_override=None, verbose=False):
        self.dry_run = dry_run
        self.force = force
        self.cooldown_override = cooldown_override
        self.verbose = verbose
        self._bar_cache: dict[str, pd.DataFrame] = {}

    def run(self, start_date: str | None = None, end_date: str | None = None):
        """Main entry point. Returns list of result dicts."""
        from engine.db import get_client
        client = get_client()
        if not client:
            logger.error("[CF] Supabase not configured — cannot run")
            return []

        # Force mode: clear existing cf_* results
        if self.force:
            self._clear_results(client, start_date, end_date)

        signals = self._fetch_signals(client, start_date, end_date)
        if not signals:
            logger.info("[CF] No unfilled blocked signals to process")
            return []

        trades = self._fetch_trades(client, start_date, end_date)
        strategy_groups = self._group_by_strategy(signals, trades)

        # Collect all signal dates for session loading
        signal_dates = sorted({s["signal_date"] for s in signals if s.get("signal_date")})

        results = []
        stats = {"fetched": len(signals), "eligible": 0, "simulated": 0,
                 "cooldown_suppressed": 0, "errors": 0, "skipped": 0}

        for strategy_id, (strat_signals, strat_trades) in sorted(strategy_groups.items()):
            cfg = STRATEGY_CONFIGS.get(strategy_id)
            if not cfg:
                logger.warning(f"[CF] Unknown strategy {strategy_id} — skipping")
                stats["skipped"] += len(strat_signals)
                continue

            bar_df = self._load_bars(cfg.instrument, signal_dates)
            if bar_df is None:
                stats["errors"] += len(strat_signals)
                continue

            strat_results = self._process_strategy(
                strategy_id, strat_signals, strat_trades, cfg, bar_df, stats
            )
            results.extend(strat_results)

        # Write results
        if not self.dry_run and results:
            self._write_results(client, results)

        # Summary
        logger.info(
            f"[CF] Done: {stats['fetched']} fetched, {stats['eligible']} eligible, "
            f"{stats['simulated']} simulated, {stats['cooldown_suppressed']} CD-suppressed, "
            f"{stats['errors']} errors, {stats['skipped']} skipped"
        )
        return results

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_bars(self, instrument: str, signal_dates: list[str] | None = None) -> pd.DataFrame | None:
        """Load bars for instrument. Tries session JSONs first, falls back to Databento CSVs."""
        if instrument in self._bar_cache:
            return self._bar_cache[instrument]

        # Primary: session JSONs (already captured by the engine, free)
        df = self._load_bars_from_sessions(instrument, signal_dates)

        # Fallback: Databento CSVs
        if df is None:
            df = self._load_bars_from_databento(instrument)

        if df is not None:
            self._bar_cache[instrument] = df
        return df

    def _load_bars_from_sessions(self, instrument: str, signal_dates: list[str] | None = None) -> pd.DataFrame | None:
        """Load bars from session JSON files. Returns DataFrame in same format as Databento loader."""
        if not signal_dates:
            # Discover all available session files
            files = sorted(SESSIONS_DIR.glob("session_????-??-??.json"))
            if not files:
                return None
            signal_dates = [f.stem.replace("session_", "") for f in files]

        dfs = []
        for date_str in signal_dates:
            session_file = SESSIONS_DIR / f"session_{date_str}.json"
            if not session_file.exists():
                logger.warning(f"[CF] No session file for {date_str}")
                continue

            try:
                with open(session_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[CF] Failed to read {session_file}: {e}")
                continue

            bars = data.get("bars", {}).get(instrument)
            if not bars:
                logger.warning(f"[CF] No {instrument} bars in {session_file.name}")
                continue

            # Reverse the timestamp display hack:
            # Session stores: real_utc + ET_offset (so it displays as ET when treated as UTC)
            # To get real UTC: stored_time - ET_offset = stored_time + abs(utc_offset_seconds)
            session_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=ET)
            et_offset_s = int(session_date.utcoffset().total_seconds())  # negative (-14400 or -18000)

            rows = []
            for bar in bars:
                real_utc_ts = bar["time"] - et_offset_s  # subtract negative = add positive
                rows.append({
                    "time": real_utc_ts,
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar.get("volume", 0),
                })
            dfs.append(pd.DataFrame(rows))

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)

        # Same processing as Databento loader
        df["timestamp"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
        df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")

        et_times = df["timestamp"].dt.tz_convert(ET)
        df["et_mins"] = et_times.dt.hour * 60 + et_times.dt.minute
        df["et_date"] = et_times.dt.date

        df = df.reset_index(drop=True)
        df["unix_ts"] = df["time"].astype(int)

        logger.info(
            f"[CF] Loaded {len(df)} bars for {instrument} from {len(dfs)} session files, "
            f"range {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
        )
        return df

    def _load_bars_from_databento(self, instrument: str) -> pd.DataFrame | None:
        """Fallback: load Databento CSVs for instrument."""
        files = sorted(DATA_DIR.glob(f"databento_{instrument}_1min_*.csv"))
        if not files:
            logger.error(f"[CF] No bar data for {instrument} (no sessions or Databento CSVs)")
            return None

        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        df["timestamp"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
        df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")

        et_times = df["timestamp"].dt.tz_convert(ET)
        df["et_mins"] = et_times.dt.hour * 60 + et_times.dt.minute
        df["et_date"] = et_times.dt.date

        df = df.reset_index(drop=True)
        df["unix_ts"] = df["time"].astype(int)

        logger.info(
            f"[CF] Loaded {len(df)} bars for {instrument} from {len(files)} Databento CSVs (fallback), "
            f"range {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
        )
        return df

    # ------------------------------------------------------------------
    # Supabase queries
    # ------------------------------------------------------------------

    def _fetch_signals(self, client, start_date, end_date) -> list[dict]:
        """Fetch blocked signals with unfilled cf_exit_reason."""
        query = client.table("blocked_signals").select("*").is_("cf_exit_reason", "null")
        if start_date:
            query = query.gte("signal_date", start_date)
        if end_date:
            query = query.lte("signal_date", end_date)
        resp = query.order("signal_time").execute()
        return resp.data if resp.data else []

    def _fetch_trades(self, client, start_date, end_date) -> list[dict]:
        """Fetch real trades for cooldown interleaving."""
        query = client.table("trades").select(
            "id,strategy_id,entry_time,exit_time,side,instrument"
        )
        if start_date:
            # Fetch a few days before for cooldown seeding
            seed_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=2)).strftime("%Y-%m-%d")
            query = query.gte("entry_time", seed_date + "T00:00:00Z")
        if end_date:
            query = query.lte("entry_time", end_date + "T23:59:59Z")
        resp = query.order("entry_time").execute()
        return resp.data if resp.data else []

    def _clear_results(self, client, start_date, end_date):
        """Clear cf_* fields for force recompute."""
        query = client.table("blocked_signals").update({
            "cf_exit_price": None,
            "cf_exit_reason": None,
            "cf_pnl_pts": None,
            "cf_pnl_dollar": None,
            "cf_bars_held": None,
            "cf_mfe_pts": None,
            "cf_mae_pts": None,
            "signal_group_id": None,
        }).not_.is_("cf_exit_reason", "null")
        if start_date:
            query = query.gte("signal_date", start_date)
        if end_date:
            query = query.lte("signal_date", end_date)
        resp = query.execute()
        count = len(resp.data) if resp.data else 0
        logger.info(f"[CF] Cleared {count} existing cf results (--force)")

    # ------------------------------------------------------------------
    # Grouping and timeline walk
    # ------------------------------------------------------------------

    def _group_by_strategy(self, signals, trades):
        """Group signals and trades by strategy_id (not per-day) for continuous cooldown walk."""
        groups = {}  # strategy_id -> ([signals], [trades])

        for sig in signals:
            sid = sig["strategy_id"]
            if sid not in groups:
                groups[sid] = ([], [])
            groups[sid][0].append(sig)

        for trade in trades:
            sid = trade["strategy_id"]
            if sid not in groups:
                groups[sid] = ([], [])
            groups[sid][1].append(trade)

        return groups

    def _process_strategy(self, strategy_id, signals, trades, cfg, bar_df, stats):
        """Cooldown timeline walk for one strategy across ALL days (continuous cooldown)."""
        results = []
        cooldown_bars = self.cooldown_override if self.cooldown_override is not None else cfg.cooldown

        # Build merged timeline: real trades + blocked signals sorted by time
        events = []
        for sig in signals:
            events.append(("signal", sig["signal_time"], sig))
        for trade in trades:
            exit_time = trade.get("exit_time")
            if not exit_time:
                # Trade still open or missing exit — skip for cooldown purposes
                continue
            events.append(("trade", trade["entry_time"], trade, exit_time))

        events.sort(key=lambda e: e[1])

        # Numpy arrays for simulation
        opens = bar_df["open"].values.astype(float)
        highs = bar_df["high"].values.astype(float)
        lows = bar_df["low"].values.astype(float)
        closes = bar_df["close"].values.astype(float)
        et_mins = bar_df["et_mins"].values.astype(int)
        unix_ts = bar_df["unix_ts"].values.astype(int)

        last_exit_bar = -9999  # Seed cooldown (no prior trade)

        for event in events:
            if event[0] == "trade":
                # Real trade — update cooldown from exit time
                _, _, trade_data, exit_time_str = event
                exit_bar = self._time_to_bar_index(unix_ts, exit_time_str)
                if exit_bar is not None:
                    last_exit_bar = exit_bar
                continue

            # Blocked signal
            _, _, sig = event
            stats["eligible"] += 1

            entry_bar = self._time_to_bar_index(unix_ts, sig["signal_time"])
            if entry_bar is None:
                logger.warning(f"[CF] No bar match for signal {sig['id']} at {sig['signal_time']}")
                stats["errors"] += 1
                continue

            # Tag correlated entries (V15 ↔ vScalpC on same bar)
            side = sig.get("side", "long")
            if strategy_id in ("MNQ_V15", "MNQ_VSCALPC"):
                group_id = hashlib.md5(
                    f"{entry_bar}:{side}".encode()
                ).hexdigest()[:12]
            else:
                group_id = None

            # Cooldown check
            bars_since = entry_bar - last_exit_bar
            if bars_since < cooldown_bars:
                result = {
                    "id": sig["id"],
                    "cf_exit_price": None,
                    "cf_exit_reason": "COOLDOWN_SUPPRESSED",
                    "cf_pnl_pts": 0.0,
                    "cf_pnl_dollar": 0.0,
                    "cf_bars_held": 0,
                    "cf_mfe_pts": 0.0,
                    "cf_mae_pts": 0.0,
                    "signal_group_id": group_id,
                }
                results.append(result)
                stats["cooldown_suppressed"] += 1
                if self.verbose:
                    logger.info(f"[CF] {sig['id']}: COOLDOWN_SUPPRESSED (bars_since={bars_since} < {cooldown_bars})")
                continue

            # Determine entry price: prefer signal_price (bar.open), fall back to bar data
            entry_price = sig.get("signal_price")
            if entry_price is None:
                # Historical signals before the fix — use bar open from CSV
                entry_price = float(opens[entry_bar])

            is_partial = cfg.entry_qty > 1 and cfg.partial_tp_pts > 0
            commission_per_contract = _COMMISSION_PTS.get(cfg.instrument, 0.52)
            total_commission_pts = commission_per_contract * cfg.entry_qty

            # EOD: use session_close_et from config
            eod_et = cfg.session_close_et or "16:00"

            # Simulate
            # vScalpC uses structure exit (not BE_TIME) — skip BE_TIME in CF sim
            effective_be_time = 0 if cfg.structure_exit_type else cfg.breakeven_after_bars

            if is_partial:
                cf = simulate_partial_exit(
                    opens, highs, lows, closes, et_mins,
                    entry_idx=entry_bar, entry_price=entry_price, side=side,
                    tp1_pts=cfg.partial_tp_pts, tp2_pts=cfg.tp_pts,
                    sl_pts=cfg.max_loss_pts, eod_et=eod_et,
                    sl_to_be=cfg.move_sl_to_be_after_tp1,
                    be_time_bars=effective_be_time,
                    commission_pts=total_commission_pts,
                )
            else:
                cf = simulate_single_exit(
                    opens, highs, lows, closes, et_mins,
                    entry_idx=entry_bar, entry_price=entry_price, side=side,
                    tp_pts=cfg.tp_pts, sl_pts=cfg.max_loss_pts, eod_et=eod_et,
                    be_time_bars=effective_be_time,
                    commission_pts=total_commission_pts,
                )

            if cf is None:
                stats["errors"] += 1
                continue

            # Update cooldown from CF exit
            cf_exit_bar = entry_bar + cf.bars_held
            last_exit_bar = cf_exit_bar

            result = {
                "id": sig["id"],
                "cf_exit_price": round(float(cf.exit_price), 2),
                "cf_exit_reason": cf.exit_reason,
                "cf_pnl_pts": float(cf.pnl_pts),
                "cf_pnl_dollar": round(cf.pnl_pts * cfg.dollar_per_pt, 2),
                "cf_bars_held": int(cf.bars_held),
                "cf_mfe_pts": float(cf.mfe_pts),
                "cf_mae_pts": float(cf.mae_pts),
                "signal_group_id": group_id,
            }
            results.append(result)
            stats["simulated"] += 1

            if self.verbose:
                logger.info(
                    f"[CF] {sig['id']}: {side} entry@{entry_price:.2f} → "
                    f"{cf.exit_reason} exit@{cf.exit_price:.2f} pnl={cf.pnl_pts:+.2f}pts "
                    f"({cf.bars_held} bars)"
                )

        return results

    # ------------------------------------------------------------------
    # Timestamp → bar index mapping
    # ------------------------------------------------------------------

    def _time_to_bar_index(self, unix_ts: np.ndarray, time_str: str,
                           tolerance_s: int = 30) -> int | None:
        """Find bar index closest to the given ISO timestamp within tolerance."""
        try:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            target_ts = int(dt.timestamp())
        except (ValueError, AttributeError):
            return None

        # Binary search for closest bar
        idx = np.searchsorted(unix_ts, target_ts)

        best_idx = None
        best_diff = tolerance_s + 1

        for candidate in (idx - 1, idx, idx + 1):
            if 0 <= candidate < len(unix_ts):
                diff = abs(int(unix_ts[candidate]) - target_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = candidate

        if best_diff <= tolerance_s:
            return best_idx
        return None

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------

    def _write_results(self, client, results: list[dict]):
        """Batch UPDATE blocked_signals with cf_* results."""
        success = 0
        errors = 0

        for r in results:
            row_id = r["id"]
            update_data = {k: v for k, v in r.items() if k != "id"}
            try:
                client.table("blocked_signals").update(update_data).eq("id", row_id).execute()
                success += 1
            except Exception as e:
                logger.error(f"[CF] Failed to update {row_id}: {e}")
                errors += 1

        logger.info(f"[CF] Wrote {success} results, {errors} errors")
