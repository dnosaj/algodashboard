# Plan: Developing Daily VPOC

**Date**: March 15, 2026
**Status**: Phase 4 — Executing
**Phase 1 Vision**: Confirmed Mar 15
**Phase 2 Plan**: 9-agent team, synthesized Mar 15
**Phase 3 Review**: Completed Mar 15. Key corrections: keep new accumulator lists (can't reuse `_current_rth` — only populated for MES), drop new static method (reuse `_compute_value_area`), fix `_finalize_prior_day` bin_width bug.

## Deliverables

### Deliverable 1: Engine Accumulator (~40 lines in safety_manager.py)

**Init** (after line ~183, weekly VPOC block):
- `_daily_vpoc_closes: dict[str, list[float]]` — RTH closes today per instrument
- `_daily_vpoc_volumes: dict[str, list[float]]` — RTH volumes today per instrument
- `_developing_vpoc: dict[str, float | None]` — current developing VPOC price
- `_daily_vpoc_strength: dict[str, float]` — VCR (max_bin_vol / total_vol)
- `_dvpoc_stability: dict[str, int]` — bars since last dPOC shift

**Accumulate** (in `on_bar` RTH block, after weekly VPOC accumulation ~line 470):
- Guard: `if inst in self._weekly_bin_width and bar.volume > 0`
- Lazy init lists if inst not present
- Append bar.close and bar.volume to daily lists
- Call `_compute_value_area(closes, volumes, bin_width)[0]` to get developing VPOC
- If VPOC changed from previous value: reset stability counter to 0
- Else: increment stability counter
- Compute VCR: `max(bin_volumes) / sum(volumes)` — need to compute this separately since `_compute_value_area` doesn't return it. Use same binning as `_compute_value_area` inline, or store total_vol separately.
  - Actually: just compute `max_bin_vol / total_vol` from the closes/volumes. This requires binning. Since `_compute_value_area` already bins internally but doesn't expose the bin array, we have two options:
    1. Call `_compute_value_area` for VPOC, then compute VCR separately (re-bins, ~0.5ms extra)
    2. Create a small helper that returns (vpoc, vcr) from the bin array
  - Option 1 is simpler. Total cost <1ms at N=390. Use option 1.

**Reset** (in `reset_daily`, after VWAP/OR clear ~line 1369):
- `_daily_vpoc_closes.clear()`, `_daily_vpoc_volumes.clear()`, `_developing_vpoc.clear()`, `_daily_vpoc_strength.clear()`, `_dvpoc_stability.clear()`

**get_status()** (in `ict_levels` dict ~line 1544):
- Add `developing_vpoc`, `dvpoc_strength`, `dvpoc_stability`

**Session save** (server.py `_do_save_session` ~line 745):
- Add `developing_vpoc` to `ict_snapshot[inst]`

**Fix** `_finalize_prior_day` (line 847):
- Change `bin_width=5.0` → `bin_width=self._weekly_bin_width.get(inst, 5.0)`

**Design decisions**:
- New accumulator lists (NOT reusing `_current_rth` — only populated for MES via `_update_prior_day_tracking`)
- Reuse `_compute_value_area(...)[0]` for VPOC (NOT creating new static method)
- No persistence for daily bins (transient, reset daily)
- Do NOT skip zero-volume bars (match weekly VPOC behavior)
- RTH-only accumulation (existing `_RTH_OPEN_ET` guard)

### Deliverable 2: Dashboard (~22 lines)

**types.ts**: Add `developing_vpoc: number | null` to `ICTLevelData`

**PriceChart.tsx** (ICT useEffect, after wVAL block ~line 668):
- `createPriceLine` — cyan `#00cccc`, dotted (lineStyle 3), width 1, label "dPOC"
- Bundled under existing ICT toggle
- `if (levels.developing_vpoc != null)` guard for null safety

**Session replay**: Works automatically — same ICT level rendering path.
**Known V1 limitation**: Replay shows end-of-day dPOC snapshot, not developing trajectory.

### Deliverable 3: Forensics Script (~120 lines, new file)

**File**: `backtesting_engine/strategies/developing_vpoc_forensics.py`

**Core function**: `compute_developing_vpoc(times, closes, volumes, et_mins, bin_width)` — forward-only pass, returns per-bar array of developing VPOC. Resets on calendar date change. RTH-only accumulation.

**Analysis** (bar[i-1] convention for dVPOC at trade entry):

Tier 1 (must answer):
1. WR/PF by distance bands: 0-5, 5-10, 10-20, 20+ pts from dVPOC — per strategy
2. dVPOC-VWAP per-entry distance: `abs(dVPOC - VWAP)` at each entry, segment by distance
3. Null baseline: what % of time is price within Xpts of dVPOC anyway?

Tier 2 (informative):
4. Directional: longs below dPOC vs above, shorts below vs above
5. Stability at entry: entries near stable dPOC (>60 bars since shift) vs unstable
6. Time-of-day: segment pre-10:30, 10:30-12:00, 12:00-16:00

Tier 3 (if Tier 1 positive):
7. VCR regime: high-VCR days (strong dPOC) vs low-VCR days — quartile split
8. dPOC vs weekly VPOC: complementary or redundant?

### Deliverable 4: Prior-Day VPOC for MNQ (~25 lines)

- bin_width fix in `_finalize_prior_day` (1 line — included in D1)
- Forensic proximity analysis: MNQ trades vs prior-day VPOC at [5, 7, 10] pts
- Verify MES prior-day VPOC values unchanged after fix

### Verification

- Parity: forensics output = what engine would produce, for every RTH bar
- Daily reset: first RTH bar produces correct single-bin VPOC
- Zero-vol: not skipped (match weekly)
- ETH excluded: pre-RTH bars don't affect dPOC
- Null safety: dPOC is null before first RTH bar, dashboard handles gracefully
- Session JSON: dPOC saved and rendered on replay
- MES regression: prior-day VPOC identical before/after bin_width fix

### Total: ~190 lines across ~5 files. No new migrations needed, no new dependencies.
