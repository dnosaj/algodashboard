---
name: Cooldown bar sweep — COMPLETED
description: Cooldown sweep completed Mar 13, 2026. Result: current values (MNQ=20, MES=25) are optimal. No change needed.
type: project
---

# Cooldown Bar Sweep — COMPLETED (Mar 13, 2026)

**Result:** Current values are in the right neighborhood. No change needed.

- MNQ strategies (CD=20): Lower CDs inflate IS but degrade OOS (classic overfit). CD=15 and CD=20 are equivalent on OOS. No improvement from changing.
- MES_V2 (CD=25): CD=20 is marginally better on OOS Sharpe but ~same PF. Not worth the risk of changing a validated parameter.

**Script:** `backtesting_engine/strategies/cooldown_sweep.py` — full IS/OOS sweep with verdicts.
