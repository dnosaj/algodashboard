# Structure-Based Exit Implementation — Team-Reviewed Plan (Phase 3)

## Vision
Replace fixed TP2=25 on **vScalpC runner only** with adaptive pivot-based swing exits.
MES v2 stays with fixed TP2=20 (structure exit rejected — full-period P&L -$1,460 not justified).
Keep OCO bracket on exchange at 60pt cap as crash safety. Log structure data to Supabase
for future Observation Agent. Add dashboard overlay toggle for structure levels.

## Decisions Made (Jason, Mar 13)
1. **Skip shadow runner** — go straight to paper trading
2. **vScalpC only** — MES structure exit rejected (P&L -$1,460, MaxDD only -19%, Sharpe only +5%)
3. **60pt crash-safety cap** — resting OCO TP2 at 60pts on exchange
4. **Phantom OCO Ladder REJECTED** — tastytrade has no partial OCO modification; cancel-replace each bar has 100ms unprotected gap. Active monitoring + 60pt cap is simpler and equally safe.
5. **Wider TP not needed** — structure exits are adaptive (variable exit level), not just a wider fixed TP
6. **tp_pts=60** — Was 25. Now serves as crash-safety cap directly. No separate cap field needed.

## Backtest Results — FULL PERIOD (12.5 months, 2 contracts)
```
vScalpC (MNQ) — Pivot LB=50, PR=2, Buffer=2pts, Cap=60
                         P&L      PF   Sharpe    MaxDD      WR   Trades
Baseline (TP2=25)    $+5,693   1.374    1.914  $-1,586   76.7%     472
Structure exit       $+6,099   1.406    1.975  $-1,394   78.0%     469
─────────────────────────────────────────────────────────────────────────
  P&L:    +$406       (more money)
  MaxDD:  -12%        ($193 less drawdown)
  Sharpe: +3%
  PF:     +2.3%
  WR:     +1.3pp      (higher win rate)
```
All metrics improve. IS/OOS PF ratio: 97.3% (rock solid). 19/20 top configs pass IS/OOS.

---

## ARCHITECTURE

### New Files
| File | Purpose |
|------|---------|
| `live_trading/engine/structure_monitor.py` | IncrementalSwingTracker + StructureExitMonitor |
| `live_trading/supabase/migrations/004_structure_exits.sql` | New table + trades column |

### Modified Files
| File | Changes |
|------|---------|
| `engine/config.py` | 6 new fields on StrategyConfig + MNQ_VSCALPC update |
| `engine/events.py` | ExitReason.STRUCTURE + TradeRecord.structure_exit_level |
| `engine/runner.py` | StructureExitMonitor init, process_bar integration, warmup |
| `engine/db_logger.py` | Subscribe to structure events |
| `api/server.py` | Structure levels in status + WS events |
| `dashboard/src/types.ts` | StructureLevel type |
| `dashboard/src/hooks/useWebSocket.ts` | Handle structure_levels WS event |
| Dashboard chart component | Structure level overlay with toggle |

### NOT Modified
- `strategy.py` — stays as pure signal generator
- `safety_manager.py` — structure exits are exit logic, not safety gates
- `intra_bar_monitor.py` — structure exits are bar-close only (matching backtest)

### Config Fields (on StrategyConfig)
```python
structure_exit_type: str = ""           # "" (disabled) or "pivot"
structure_exit_lookback: int = 0        # Left bars for pivot
structure_exit_pivot_right: int = 2     # Confirmation bars (pivot only)
structure_exit_buffer_pts: float = 0.0  # Exit N pts before swing level
```

Applied to vScalpC:
```python
MNQ_VSCALPC = StrategyConfig(
    ...existing fields...
    tp_pts=60,                      # Was 25 — now crash-safety cap on exchange
    structure_exit_type="pivot",
    structure_exit_lookback=50,
    structure_exit_pivot_right=2,
    structure_exit_buffer_pts=2.0,
)
```

### Data Flow
```
Bar arrives → runner.process_bar()
  1. Check bracket fills (existing)
  2. strategy.on_bar() — SM/RSI/entries/SL/TP1 (existing)
  3. struct_monitor.on_bar() — update swing levels, check runner exit
     If triggered: cancel OCO → verify position → market close → emit events
  4. Handle entry signals (existing)
```

### Exit Priority (maintained from backtest)
SL → BE (after TP1) → Structure → TP_cap (60pt OCO) → BE_TIME → EOD

### IncrementalSwingTracker
- Deque-based rolling buffer (maxlen = lookback + pivot_right + 1 = 53)
- Pivot: candidate at j = i - right, confirmed when left + right neighbors pass
- Exposes swing_high / swing_low after each update()
- Capture prev values BEFORE update() to match bar[i-1] convention
- Parity-tested against vectorized backtest on full MNQ CSV

### OCO Strategy
- Place TP2 OCO at cap=60pts ($120/contract) as crash safety on exchange
- Engine actively monitors structure levels and exits runner before cap
- SL leg NEVER cancelled independently
- When structure fires: cancel TP2 OCO → verify bracket.filled → market close
- If cancel fails or bracket already filled: DO NOT send market order

### Phantom OCO Ladder (REJECTED)
- tastytrade has no `replace_complex_order()` — OCO is atomic, can't modify one leg
- Cancel-replace each bar leaves ~100ms unprotected gap per update
- Active monitoring + 60pt cap is simpler and equally safe

---

## PHASE 3 REVIEW FIXES

1. **tp_pts 25→60**: Was conflicting with structure_exit_cap_pts. Now tp_pts IS the cap. Removed separate cap field.
2. **BE_TIME ordering**: Skip BE_TIME on runner leg when structure monitoring active. Structure monitor handles it or 60pt cap catches it.
3. **OCO cancel → cancel TP leg only**: Keep SL resting until market fill confirmed. If market fails, SL still protects.
4. **caffeinate -di**: Add `caffeinate -di -w $! &` to NQ Trading.app launcher after engine launch line.
5. **Parity test scope**: Full 12.5-month MNQ CSV, not 2 weeks.
6. **Paper criteria**: 10 structure exits in 10 days (rate is ~1.0/day, not 2.0/day).

---

## CRITICAL RISKS AND MITIGATIONS

1. **CRITICAL: `caffeinate -di`** — Add to NQ Trading.app launcher. macOS can sleep during trading, leaving runner with only SL.
2. **CRITICAL: OCO cancel race** — After cancel await, check `bracket.filled` before sending market order. If TP2 already filled, position is flat — don't double-exit.
3. **CRITICAL: 60pt crash-safety cap** — Runner MUST have resting TP on exchange. If engine dies, cap catches tail risk. Without it, runner rides winner back to BE.
4. **HIGH: Position guard** — Before any structure exit market order, verify `strategy.state.position != 0`. Defense against race conditions.
5. **HIGH: Swing tracker update ordering** — Capture swing levels BEFORE calling tracker.update(bar). Otherwise structure level includes current bar data (look-ahead).
6. **MODERATE: ETH bars** — Verify backtest and live use same bar universe. If backtest is RTH-only but live uses continuous bars, swing levels will differ.
7. **MODERATE: Lunch-hour Donchian contraction** — N/A (vScalpC uses Pivot, not Donchian. And entry cutoff is 13:00 ET).

---

## VERIFICATION PLAN

### Unit Tests
1. `test_swing_levels.py` — 10 test cases for pivot math (basic, no-lookahead, NaN, monotone, flat)
2. `test_incremental_swing_parity.py` — feed full MNQ CSV through both versions, assert identical (**MOST CRITICAL**)
3. `test_structure_exit_logic.py` — 13 exit decision scenarios (NaN, priority, buffer, min_profit, partial interaction)
4. `test_oco_structure_interaction.py` — 5 race condition scenarios

### Paper Trade Acceptance Criteria
- **Minimum**: 10 trading days, 10+ vScalpC structure exits observed
- **Metrics**: Runner P&L per trade >= -5% vs baseline, SL rate <= baseline +5%
- **Hard fail**: Any position mismatch, any OCO cancel failure, any crash during structure processing
- **Success**: >= 80% of structure exits rated "good/acceptable" by Jason

### Backtest-to-Live Parity
- Feed full 12.5-month MNQ CSV through incremental engine with structure monitoring
- Compare every trade to vectorized backtest: zero mismatches on exit bars/reasons

---

## DEPLOYMENT

### Rollout
1. Deploy with `structure_exit_type=""` on all strategies (zero impact)
2. Enable on MNQ_VSCALPC only
3. Paper trade 10+ trading days with acceptance criteria above
4. V15, vScalpB, MES_V2: never enabled (scalp strategies / MES rejected)

### Rollback
- **Mid-session**: Dashboard "Disable Structure Exit" toggle → falls back to 60pt OCO cap
- **Overnight**: Set `structure_exit_type=""` in config, restart

### Monitoring
- exit_reason="STRUCTURE" in trade log, session JSON, Supabase
- `logs/structure_exits.csv` with gain_vs_cap comparison
- Dashboard "STRUCT" badge showing current target price or "--"
- Structure exit failure counter

---

## IMPLEMENTATION STAGES

### Stage 1: Core Exit Logic + Basic Logging
- IncrementalSwingTracker (pivot only — donchian not needed since MES rejected)
- StructureExitMonitor
- Config fields + MNQ_VSCALPC update
- ExitReason.STRUCTURE
- Parity tests
- `caffeinate -di` fix
- Basic Supabase logging (exit events only)

### Stage 2: Dashboard Overlay
- Structure level lines on chart (toggle, like PPST)
- Distance-to-level readout in SafetyPanel
- "STR" badge in trade log
- Runtime disable toggle

### Stage 3: Observation Agent Data (deferred)
- Per-bar structure level logging while runners active
- Near-miss logging (level nearby but didn't fire)
- jason_override field for manual intervention tracking
- Digest Agent tool for structure exit queries
