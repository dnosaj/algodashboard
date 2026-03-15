# ICT Dashboard Levels — Plan (Phase 2 Complete)

**Date**: March 14, 2026
**Status**: Phase 2 complete. Ready for Phase 3 review then execution.
**Depends on**: ICT forensics (`memory/ict_forensics_results.md`)

## Vision

Color-coded ICT levels on the dashboard chart with trade tagging for ongoing tracking. Show where our entries perform well (green) and poorly (red) relative to market structure.

## What to show (revised — London H/L DROPPED)

**GREEN (strong positive):**
- Weekly VPOC: WR 90.5%, PF 3.891 (N=42)
  - Conviction opacity: thicker/brighter line when volume is concentrated at POC
  - Line style toggles dashed→solid when price is within proximity

**RED (danger — our entries underperform):**
- Weekly VAL: WR 50.0%, PF 0.622 (N=22)

**ORDER BLOCK ZONES (strongest signal — WR 36.4% inside, -37pp):**
- Bullish OB: green rectangle [low[1], high[1]] extending right until mitigated (close < bottom)
- Bearish OB: red rectangle [low[1], high[1]] extending right until mitigated (close > top)
- Midline at 50% (dotted)
- Max 2 active per direction (matches UAlgo default)
- Detection: 3-bar engulfing pattern from UAlgo (already ported in ict_forensics.py)

**DROPPED:**
- London H/L: Gate blocks <1% of entries, no directional edge, low frequency. Not worth visual clutter.

**Not shown** (too close to baseline to justify visual clutter):
- Asia H/L (-3.3pp), Pre-NY H/L (-4.1pp), Weekly VAH (+0.7pp)

## Trade tagging

When a trade opens near a level or inside an OB zone:
- Badge in TradeLog with distance: "wPOC +0.8" (green), "wVAL -1.5" (red), "OB" (red if bearish OB, green if bullish)
- Tag in Supabase `trades.ict_near_levels text[]` with GIN index
- Digest Agent queries tagged trades and reports performance vs forensic baseline

## London H/L Gate Sweep Results (Mar 14)

Directional decomposition showed BOTH directions near BOTH London levels underperform. Undirectional gate is appropriate.

| Buffer | vScalpC IS PF | vScalpC OOS PF | Verdict |
|--------|---------------|----------------|---------|
| 3 pts | +7.7% | +1.2% | **STRONG PASS** |
| 5 pts | +3.7% | +0.6% | Marginal |
| 7 pts | +2.1% | -0.5% | Mixed |
| 10 pts | -0.2% | -1.1% | FAIL |

vScalpA is inconsistent across IS/OOS. Gate blocks very few trades (~1-2/month). Decision: **implement as observation (display + tagging) first, monitor before promoting to hard gate.**

## Architecture (from 9-agent team + corrections)

### Engine (~120 lines in safety_manager.py)

**Weekly volume profile** (~40 lines):
- Accumulate RTH closes+volumes Mon-Fri
- Compute VPOC/VAL on Monday daily reset using existing `_compute_value_area()`
- Persist in gate_state.json for restart survival
- Bin width: 2 for MNQ, 5 for MES (matching forensics)
- Also compute `vpoc_strength` = max_bin_volume / total_volume (for conviction opacity)

**Order Block tracking** (~60 lines):
- Port UAlgo `detectLogic()` 3-bar engulfing pattern to Python
- Track active OB zones per instrument (max 2 bullish + 2 bearish)
- Mitigation: remove OB when close passes through zone
- OB zone = [low[1], high[1]] of the engulfing candle
- Already ported in ict_forensics.py `compute_order_blocks()` — adapt for per-bar incremental use
- Extend on each bar (zone persists rightward until mitigated)

**ICT proximity check** (~20 lines):
- `get_ict_proximity(inst, price) -> list[str]`
- 10pts threshold for weekly levels (wPOC, wVAL)
- Inside OB zone check (entry_price between OB top and bottom)
- Returns list of matched tags: ["wVAL", "wPOC", "BULL_OB", "BEAR_OB"]
- Called at trade entry time in runner.py

**CRITICAL CONSTRAINT**: ICT levels are OBSERVATION ONLY. Never enter `check_can_trade()`. No `_gate_` variable naming. Comment block at top of section.

### Trade tagging (~20 lines across runner.py, events.py, strategy.py, db_logger.py)

- At trade entry: runner.py calls `safety.get_ict_proximity()`, stamps on TradeState
- TradeRecord gets `ict_near_levels: Optional[list[str]]`
- db_logger includes in `_trade_to_row()`
- Captures levels active AT ENTRY TIME (not current)

### API/WebSocket (~8 lines in server.py)

- Add `ict_levels` to `get_status()` return dict
- Rides existing `safety_status` broadcast (no new WS message type)

### Dashboard (~80 lines across PriceChart.tsx, TradeLog.tsx, types.ts)

**PriceChart.tsx** (~50 lines):
- New `useEffect` for ICT price lines (same pattern as prior-day levels)
- "ICT" toggle button in chart header (like PPST — observation, not execution)
- Weekly VPOC: green, solid when price nearby / dashed when far (Creative idea #2)
- London H/L: red, labeled "LDN H" / "LDN L"
- Weekly VAL: red, labeled "wVAL"
- VPOC conviction opacity based on volume concentration (Creative idea #1)

**TradeLog.tsx** (~15 lines):
- ICT badges with distance: "LDN +2.3" (red), "wPOC +0.8" (green) (Creative idea #3)

**types.ts** (~15 lines):
- `ICTLevels` type on SafetyStatusData
- `ict_near_levels` on Trade type

### Supabase migration (~15 lines)

```sql
ALTER TABLE trades ADD COLUMN IF NOT EXISTS ict_near_levels text[];
CREATE INDEX IF NOT EXISTS idx_trades_ict_levels ON trades USING GIN (ict_near_levels);
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS london_high double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS london_low double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS weekly_vpoc double precision;
ALTER TABLE gate_state_snapshots ADD COLUMN IF NOT EXISTS weekly_val double precision;
```

### Digest Agent (~15 lines)

- Extend `get_level_proximity` or add `get_ict_level_trades` tool
- EOD report: "3 trades today near London H/L — 1W 2L, tracking against -12pp forensic finding"

## Creative enhancements (from Creative agent)

1. **Conviction opacity on VPOC** (~8 lines): `vpoc_strength = max_bin_volume / total_volume`. Thicker/brighter green when volume is concentrated. Faint when diffuse.
2. **Line style toggle for proximity** (~15 lines): When price is within threshold of a level, line changes from dashed to solid. Real-time "heads up" without animation.
3. **Distance in trade badges** (~20 lines): "LDN +2.3" instead of just "LDN". Shows exactly how close the entry was.

## Key decisions

| Decision | Rationale |
|----------|-----------|
| London: 02:00-05:00 ET fixed | Forensics tested this window. Don't track LSE hours. |
| Prior-week VPOC only (not developing) | That's what was validated at N=42. Developing is untested. |
| Observation only, not a gate (yet) | Gate sweep shows real but small edge. Monitor via tagging first. |
| MNQ + MES both get levels | Display costs nothing. Gate would be MNQ-only (MES has its own daily gate). |
| Labels: "wPOC" not "VPOC" | Avoid confusion with existing prior-day VPOC on MES. |
| Chart-header toggle (like PPST) | Not SafetyPanel — these don't affect execution. |
| Warmup bars: increase to 720 | Ensures London session coverage on cold start. |

## Total scope

~220 lines across 11 files. Every change follows an existing pattern in the codebase.

## Phase 2 Agent Team Assessment

### What each agent contributed

| Agent | Value Added | Key Contribution |
|-------|-----------|-----------------|
| **Creative** | HIGH | Conviction opacity, line style toggle, distance badges — all adopted |
| **Architect** | HIGH | Concrete data flow, component boundaries, `text[]` array decision |
| **Domain** | HIGH | Directional decomposition gap (validated by sweep), session definition |
| **Senior Dev** | HIGH | Implementation estimates, regression risk, existing pattern identification |
| **QA** | HIGH | Edge cases (warmup bars, DST, replay mode), acceptance criteria |
| **Security** | HIGH | Observation-only constraint, naming convention, separate from gates |
| **DevOps** | MEDIUM | Deployment order, toggle placement, persistence lifecycle |
| **Product** | LOW | Over-corrected — argued to defer the feature Jason asked for |
| **Critic** | LOW | Over-corrected — recommended skipping weekly VPOC/VAL |

### Counter-correction applied

The Product Strategist and Critic optimized for "minimum viable product" rather than enhancing the user's vision. Their concerns were noted but overridden:
- Weekly VPOC/VAL IS worth displaying (it's zero-risk as observation, and WR 90.5% is a massive effect even at N=42)
- The visualization IS the feature Jason asked for — it should be built alongside tagging, not deferred
- Testing London H/L as a gate was done in parallel (sweep results above confirm small but real edge)
