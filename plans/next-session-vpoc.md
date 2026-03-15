# Next Session: Developing Daily VPOC

**Date**: March 14, 2026 (documented end of session)
**Status**: Ready to build
**Priority**: #1 on research queue

## Context

We track VWAP (volume-weighted mean) but NOT developing daily VPOC (volume mode — the price with the most volume today). These are different:
- **VWAP**: Mean price weighted by volume. Every price contributes. Smooth, moves slowly.
- **VPOC**: The single price bin with the HIGHEST volume. The mode. Can jump as volume concentrates.

VPOC is a fundamental volume profile / ICT concept that we've never tested against our entries. Weekly VPOC showed 90.5% WR (N=42) in our forensics — if daily VPOC shows a similar positive signal, it's extremely valuable.

## What we have now

| Level | Status | Where |
|-------|--------|-------|
| VWAP (developing) | Computed, in engine | `safety_manager.py` — `_vwap_num`, `_vwap_den` accumulators |
| Prior-day VPOC | Computed, MES gate only | `safety_manager.py` — `_compute_value_area()` at daily reset |
| Weekly VPOC | Computed (Mar 14) | `safety_manager.py` — Monday reset, shown on chart as green line |
| Developing daily VPOC | **NOT IMPLEMENTED** | Need running volume profile accumulator |

## Tasks (in order)

### 1. Add developing daily VPOC to the engine
- Add a running volume profile accumulator in SafetyManager (bin prices by close, track volume per bin during RTH)
- On each bar: update the bin for the current close, recompute POC = bin with max volume
- Store as `_developing_vpoc: dict[str, float]`
- Use same bin_width as weekly (2 for MNQ, 5 for MES)
- Reset daily (same as VWAP reset)
- Include in `get_status()` → `ict_levels[inst].developing_vpoc`
- Observation only — never gates

### 2. Show developing daily VPOC on the chart
- New price line in PriceChart.tsx ICT useEffect
- Color: distinct from weekly VPOC (maybe cyan or lighter green with different label "dPOC")
- This line MOVES during the session as volume concentrates — update on each status broadcast
- Save in session JSON for replay

### 3. Add to forensics — backtest against our entries
- For each trade in the 12.8-month backtest, compute the developing daily VPOC at entry time
- Questions to answer:
  - Do entries near developing daily VPOC perform better or worse?
  - How does daily VPOC compare to weekly VPOC as a proximity signal?
  - Is daily VPOC a positive signal (like weekly) or negative (like London H/L)?
  - Does the developing POC act as a magnet (price drawn toward it) or S/R (price bounces off)?

### 4. Test prior-day VPOC proximity for MNQ
- We only tested prior-day VPOC as a gate for MES (adopted, blocks ~20%)
- Never tested as an observation for MNQ: "do MNQ entries near prior-day VPOC perform well?"
- Quick addition to forensics — the prior-day levels are already computed
- If positive (like weekly VPOC), show on chart. If negative (like London H/L), potential gate.

## OB Replay Gap (noted, not blocking)

OBs in session replay don't work well — the snapshot approach only captures OBs active at save time. OBs that formed and got mitigated during the session are lost. Real fix: compute OBs from bar data during replay (client-side or server-side). Deferred — the live OB tracking works correctly during real-time trading.
