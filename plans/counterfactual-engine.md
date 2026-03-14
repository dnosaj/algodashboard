# Counterfactual Trade Engine — Implementation Plan

**Phase**: 3 (Team Review) COMPLETE
**Status**: Plan reviewed, issues addressed, ready for Phase 4 execution
**Date**: March 13, 2026

## Vision

Gates block ~5-25% of trading signals. We need to know if they're net-positive by simulating what would have happened if blocked signals had traded. Results feed into the existing `gate_effectiveness` view and Digest Agent reports.

## Critical Issues Surfaced by Red Team

### Issue 1: Entry Price Bug (MUST FIX)
- **Problem**: Blocked signals log `bar.close` (runner.py:518), but real entries use `bar.open` (strategy.py:689). The logged price has look-ahead bias — it includes the move during bar_i.
- **Impact**: For vScalpB (TP=3, SL=10), a 1-2pt error can flip TP↔SL.
- **Fix**:
  1. Change runner.py to log `bar.open` as `signal_price` (new field) alongside existing `price` (bar.close kept for backward compat)
  2. CF engine uses `bar.open` from CSV data for simulation (matches backtest convention)
  3. Going forward, blocked signals have the correct entry price

### Issue 2: Cooldown Cascade Limitation (ACCEPT + DOCUMENT)
- **Problem**: CF trade N's exit shifts cooldown for CF trade N+1, and could even affect whether real trade N+2 should have fired in the alternate universe.
- **Decision**: Accept the limitation. Process chronologically per-strategy per-day. CF trades create cooldown for subsequent CF signals. Real trades create cooldown for CF signals. But we do NOT model position occupancy (can't enter while already in a trade) or the full alternate universe where a CF entry changes all subsequent real trade timing.
- **Rationale**: Full universe replay requires re-running the entire backtest engine with gates disabled — a different and heavier tool. The cooldown-aware simulation is 80%+ accurate and serves the primary use case (gate effectiveness measurement).

### Issue 3: Statistical Power (SAFEGUARDS REQUIRED)
- **Problem**: ~0.75-3 blocked signals per gate per week. Need N≥20 for directional signal, N≥50 for confidence.
- **Safeguards**:
  - Minimum N=20 before displaying per-gate P&L conclusions
  - Below threshold: show "Accumulating (N/20)" instead of P&L numbers
  - Compare CF win rate against strategy baseline WR (not 50%)
  - Never recommend gate removal — only flag for investigation
  - Display alongside original backtest evidence for the gate

## Architecture

### Location: `live_trading/agents/counterfactual/`
```
live_trading/agents/counterfactual/
  __init__.py
  cli.py              # CLI entry point: python -m agents.counterfactual.cli
  engine.py            # Orchestrator: fetch → interleave → simulate → write
  exit_simulator.py    # Entry-agnostic exit simulation (single-leg + 2-leg)
```

### Why not reuse existing backtest functions?
`run_backtest_tp_exit()` and `run_backtest_partial_exit()` are full entry+exit loops that walk all bars and detect entries via SM/RSI. We need entry-agnostic exit-only simulation from a known entry point. The exit logic is ~80 lines per function, directly ported from the inner loops — simpler and cleaner than wrapping the full backtest.

### Data Flow
```
Supabase blocked_signals (WHERE cf_exit_price IS NULL)
  + Supabase trades (real trades for cooldown interleaving)
       ↓
  Cooldown timeline walk (per strategy, per day)
       ↓
  Session JSONs / Databento CSVs (1-min OHLCV)
       ↓
  Exit simulation (single-leg or 2-leg)
       ↓
  UPDATE blocked_signals SET cf_* WHERE id = ...
```

## Key Design Decisions

### 1. Entry Price
Use `bar[i].open` from CSV/session data (matching backtest and live engine `_open_position`), NOT the logged `price` field (which is bar[i].close). Also fix the logged field going forward.

### 2. Bar Data Source
- **Primary**: Databento CSVs (`backtesting_engine/data/`) — real UTC timestamps, no display-hack conversion needed, full historical coverage.
- **Rationale**: Session JSONs have a display-adjusted `time` field (`UTC_epoch + ET_offset`) that requires DST-aware reverse conversion. Databento CSVs have real UTC epochs — simpler and less error-prone. Load once per instrument, cache in memory (~3s).
- **Session JSONs**: NOT used as primary due to timestamp display hack (Phase 3 finding C2). Could be used as fallback for dates beyond CSV coverage if the timezone conversion is implemented.

### 3. Cooldown Interleaving Algorithm
```
For each (strategy_id, date):
  events = merge(real_trades, blocked_signals) sorted by time
  last_exit_bar = seed from pre-window real trade (or -9999)

  for event in events:
    if event.is_real:
      last_exit_bar = event.exit_bar_index  # Real trade always happened
      continue

    # Blocked signal — would cooldown have suppressed it?
    bars_since = event.bar_index - last_exit_bar
    if bars_since < cooldown_bars:
      mark as cf_exit_reason = "COOLDOWN_SUPPRESSED", cf_pnl = 0
      continue

    # Simulate exit (eagerly, to get exit_bar for next iteration)
    result = simulate_exit(event, bar_data, params)
    cf_exit_bar = event.bar_index + result.bars_held
    last_exit_bar = cf_exit_bar  # CF trade creates cooldown

    write result
```

### 4. Exit Simulation Functions

```python
@dataclass
class CfResult:
    exit_price: float
    exit_reason: str       # "TP", "SL", "EOD", "BE_TIME", "TP1+TP2", "TP1+SL_BE", etc.
    pnl_pts: float         # Combined for partial (both legs)
    bars_held: int
    mfe_pts: float
    mae_pts: float
    leg1_exit_reason: str | None = None   # For partial strategies
    leg2_exit_reason: str | None = None

def simulate_single_exit(
    opens, highs, lows, closes, et_mins,
    entry_idx, entry_price, side,
    tp_pts, sl_pts, eod_et, be_time_bars=0
) -> CfResult | None

def simulate_partial_exit(
    opens, highs, lows, closes, et_mins,
    entry_idx, entry_price, side,
    tp1_pts, tp2_pts, sl_pts, eod_et,
    sl_to_be=True, be_time_bars=0
) -> CfResult | None
```

Exit priority (matches backtest — SL checked before TP on same bar):
1. EOD: bar_mins >= eod_et → fill at bar close
2. SL: prev bar close breaches SL → fill at next open (checked FIRST)
3. TP: prev bar close reaches TP → fill at next open (checked SECOND)
4. BE_TIME: bars_held >= threshold → fill at next open
Note: `bars_held = current_bar_index - entry_bar_index - 1` (the `-1` offset matches both strategy.py:653 and generate_session.py:223)
Note: For partial exit, SL→BE only activates when leg1 exited via TP1 specifically (not SL). If leg1 exits via SL, runner keeps original SL.

### 5. Config Resolution
Import directly from `engine/config.py` (pure dataclasses, no side effects). Strategy params flow in automatically — if TP changes in config, CF engine uses new value. Support `--cooldown-override N` for CD sweep research.

### 6. Commission Handling
All cf_pnl as NET (after commission). Per-contract round-trip:
- MNQ: $1.04/contract (2 × $0.52)
- MES: $2.50/contract (2 × $1.25)
- vScalpC: 2 contracts × $1.04 = $2.08 total
- MES_V2: 2 contracts × $2.50 = $5.00 total

cf_pnl_pts is in POINTS (matching gate_effectiveness view which sums cf_pnl_pts). Dollar conversion at query time using dollar_per_pt.

### 7. Time-Based Exit Rules
- V15/vScalpC: Entry cutoff 13:00 ET. No explicit EOD (exits via TP/SL/BE_TIME).
- MES_V2: EOD flatten at 15:30 ET. BE_TIME=75.
- All strategies: Force exit at 16:57 ET (CME maintenance) if still open.
- Session boundary at 16:00 ET default for non-MES strategies.

### 8. Gate Interaction Fix
Currently `check_can_trade` short-circuits on first failing gate. Fix:
- Continue checking all gates even after first failure (for logging only)
- Log comma-separated `gate_type` in blocked_signals (e.g., "vix_death_zone,leledc")
- The block decision still uses the first gate (behavior unchanged)
- CF engine can then attribute correctly: "uniquely blocked by X" vs "also blocked by Y"

### 9. Correlated Entry Grouping (V15 ↔ vScalpC)
V15 and vScalpC share entries (r=0.80). When both are blocked on the same bar:
- Simulate both independently (different exit logic → different P&L)
- Tag with `signal_group_id = hash(entry_bar, side)` for deduped analysis
- Report both per-strategy and per-signal gate cost

## Operational Plan

### Runtime: Local launchd (macOS)
GitHub Actions can't access local CSV/session data. Run locally via launchd plist.

### Schedule: 18:00 ET Mon-Fri
- After EOD digest (17:00 ET) and session save (~16:50 ET)
- During CME maintenance (16:57-18:03) — no engine resource contention
- Results available for morning digest (07:00 ET)

### CLI Interface
```bash
# Default: process all unfilled signals
python -m agents.counterfactual.cli

# Specific date range
python -m agents.counterfactual.cli --start 2026-03-06 --end 2026-03-13

# Dry run (compute, print, don't write)
python -m agents.counterfactual.cli --dry-run

# Force recompute existing results
python -m agents.counterfactual.cli --force

# Research: different cooldown
python -m agents.counterfactual.cli --cooldown-override 15 --force

# Verbose
python -m agents.counterfactual.cli -v
```

### Idempotency
- Default: only processes WHERE `cf_exit_price IS NULL`
- Re-running is safe (skips already-filled rows)
- `--force` clears and recomputes

### Monitoring
- Log to `live_trading/logs/counterfactual.log`
- Summary stats: signals fetched, eligible, simulated, written, skipped, errors, duration
- Morning digest checks for unfilled cf_* older than 1 day
- gate_effectiveness view shows computed/blocked ratio

### Performance
- ~5-10 seconds total (session JSON load dominates)
- 10-50 blocked signals per day typical
- No resource concerns

## Implementation Steps

### Step 1: Fix entry price logging + DB migration
- Add `signal_price: bar.open` to blocked signal payload in runner.py. Keep `price: bar.close` for backward compat.
- Write migration `006_cf_signal_price.sql`: `ALTER TABLE blocked_signals ADD COLUMN signal_price double precision;`
- Update `_blocked_to_row()` in db_logger.py to include `signal_price`.
- (Phase 3 finding C3: without the migration, signal_price would be silently dropped)

### Step 2: Fix gate interaction logging (safety_manager.py)
- Add `_check_all_gates_for_logging()` method that runs ALL gate checks and returns comma-separated gate types.
- Call it ONLY when `check_can_trade` already returned False (no perf impact on happy path).
- (Phase 3 finding H2: don't modify the existing short-circuit logic, add a separate logging path)

### Step 3: Write exit_simulator.py (~160 lines)
- `simulate_single_exit()` — ported from run_backtest_tp_exit inner loop
- `simulate_partial_exit()` — ported from run_backtest_partial_exit inner loop
- `CfResult` dataclass

### Step 4: Write engine.py (~200 lines)
- `CounterfactualEngine` class
- Fetch unfilled signals from Supabase
- Fetch real trades for cooldown interleaving
- Load bar data from Databento CSVs (load once per instrument, cache)
- Build timestamp-to-bar-index mapping for signal/trade time lookups (tolerance-based, ±30s)
- Cooldown timeline walk per (strategy, date)
- Batch UPDATE results to Supabase (use `.update()` not `.upsert()` — CF engine never creates rows)
- `--force` NULLs ALL cf_* fields (including cf_exit_reason) before recompute

### Step 5: Write cli.py (~50 lines)
- argparse: --start, --end, --dry-run, --force, --cooldown-override, -v
- Summary output block

### Step 6: Validation
- Gold standard test: 20 known backtest trades per strategy (all 4: V15, vScalpB, vScalpC, MES_V2), feed entry points into CF simulator, verify exit_reason + exit_bar + pnl match exactly
- Cooldown test: synthetic scenarios with real + CF trade interleaving
- Edge cases: session boundary, partial exit BE_TIME, same-bar V15+vScalpC, SL→BE only after TP1 (not after SL), short-side partial exit direction

### Step 7: Digest Agent integration
- Update `get_gate_effectiveness` tool description to note cf data now available
- Add gate effectiveness section to EOD prompt (conditional on N≥5 for any gate)
- Morning digest: check for unfilled cf_* as health signal

### Step 8: launchd plist
- `~/Library/LaunchAgents/com.nqtrading.counterfactual.plist`
- 18:00 ET Mon-Fri

## What We're NOT Building (Explicitly Deferred)

1. **Full universe replay** — would require re-running backtest with gates disabled. Different tool for different question.
2. **Real-time shadow tracking** — Creative's idea of phantom positions during live trading. Good idea but adds complexity to the hot path. Defer until nightly batch proves valuable.
3. **Dashboard Gate Health panel** — defer until we have N≥20 for at least one gate (probably 4-6 weeks of data).
4. **Inverted counterfactual** (what if gates were tighter?) — interesting but requires different data flow. Defer.
5. **Confidence intervals / bootstrap** — add after initial deployment once we have meaningful sample sizes.
6. **cf_classification field** (SAVED_DISASTER, MISSED_RUNNER, etc.) — nice to have, add in v2 after validating base results.

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Entry price discrepancy (bar.close vs bar.open) | CRITICAL | Fix logging + use CSV bar.open in simulation |
| False confidence → removing protective gate | CRITICAL | N≥20 minimum, never recommend removal, show backtest evidence |
| Cooldown cascade inaccuracy | MEDIUM | Accept + document. 80%+ accurate for primary use case |
| Config param drift (old signals, new params) | LOW | Warning if config changed within simulation window |
| Code drift between CF and backtest exit logic | MEDIUM | Gold standard validation tests catch divergence |
| Small sample sizes for weeks/months | HIGH | N threshold display, accumulation tracking |
| Gate interaction blind spot (first gate only) | MEDIUM | Fix to log all failing gates |
| Session JSON timestamp display hack | HIGH | Use Databento CSVs as primary (Phase 3 fix) |
| No bar_index on blocked signals | MEDIUM | Build timestamp→index mapping with ±30s tolerance |
| cf_pnl_pts cross-strategy aggregation | LOW | View groups by strategy_id, cross-strategy sums are in pts not dollars |

## Phase 3 Review Findings (addressed above)

### Critical (fixed in plan)
- **C1**: SL checked before TP on same bar — documented in exit priority
- **C2**: Session JSON `time` field is display-adjusted, not real UTC — switched to Databento CSVs as primary
- **C3**: Missing DB migration for `signal_price` column — added to Step 1
- **C4**: Timestamp-to-bar-index mapping needed — added to Step 4

### High (fixed in plan)
- **H1**: SL→BE only after TP1 exit (not SL exit) — documented in exit priority notes
- **H2**: "Log all gates" as separate method, only on failure path — updated Step 2
- **H3**: Config param drift — accepted, `--since` default limits window
- **H5**: Commission per-leg for partial — cf_pnl_pts formula clarified
- **H6**: Bar index lookup fragility — tolerance-based matching

### Medium (accepted/documented)
- **M1**: cf_pnl_pts cross-strategy aggregation — view groups by strategy_id, acceptable
- **M2**: signal_group_id column — add to migration in Step 1
- **M3**: --force NULLs ALL cf_* fields — documented in Step 4
- **M4**: launchd DST/sleep — date validation check in CLI
- **M5**: BE_TIME `-1` offset — documented in exit priority
- **M7**: MES_V2 partial exit — added to gold standard validation

### Low (deferred or non-issues)
- **L1-L4**: Creative ideas properly deferred, entry cutoff already validated by live engine, verbose mode detail
