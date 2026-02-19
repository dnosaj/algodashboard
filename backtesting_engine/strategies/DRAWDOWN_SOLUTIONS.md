# Solutions for Clustered Drawdown Events

**Date**: Feb 18, 2026
**Context**: Feb 12-17 produced 8 Max Loss stops in 5 days on MNQ v11 (~-$598), which is unprecedented in 6-month backtest history. The worst prior 5-day window had 6 SL stops, longest consecutive SL streak was 2, and longest consecutive loss streak was 4.

**Key facts informing these solutions**:
- Entries are near-perfect for short-term direction (97-100% MFE hit rate)
- The problem is exits in adverse regimes — SM flip exits are too slow in chop/crash
- No regime predictor has been stable (all flip direction between train/test)
- v15 TP=5 trailing stops are countercyclical — help in chop, hurt in trends
- 50pt stop halves max drawdown vs no stop (SL adds 11% to PF)
- SL stops are ~10% of all trades (36/368 in 6 months), costing -$4,235 total
- Non-SL trades make +$8,803, so net is +$4,567

---

## Solution 1: Adaptive Circuit Breaker

**Concept**: Stop trading when the regime is clearly hostile, rather than continuing to take hits.

**Rules**:
- After 2 SL stops in a day → stop trading that day
- After 4 SL stops in a rolling 5-day window → reduce to paper-only for 1 day
- After 6 SL stops in 5 days → pause live trading entirely, paper-only until 2 consecutive wins

**Why it works**: The Feb 12-17 damage was front-loaded. Had you stopped after the 4th SL on Feb 13, you'd have avoided ~$500 of the ~$598 loss. The historical max SL stops in one day was 4 (Dec 16), so a 2/day limit would have caught most bad days earlier.

**Effort**: Low — the live trading bot's SafetyManager already has daily P&L limit and consecutive loss breaker. Just needs a rolling SL counter.

**Confidence**: High — pure risk management, no prediction needed.

**Testable**: Already partially in live bot. Can simulate on 6-month data to measure how much profit is sacrificed on good days vs saved on bad days.

---

## Solution 2: Volatility-Scaled Stop (ATR-Based)

**Concept**: Replace the fixed 50pt max loss with a dynamic stop that adapts to market conditions.

**Current problem**: 50pt is static, but NQ's intraday range varies dramatically:
- Quiet day (80-120pt range): 50pt stop = ~50% of daily range (reasonable)
- Wild day like Feb 13/17 (200-300pt range): 50pt stop = ~20% of daily range (way too tight — fires on noise)

**Implementation**: `max_loss_pts = N * ATR(20)` where N is calibrated.
- Quiet day: ATR ~15, so 3×ATR = 45pts (similar to current)
- Wild day: ATR ~40, so 3×ATR = 120pts (gives trade room to breathe)

**Why it works**: On high-vol days, price routinely swings 50pts then reverses. The fixed stop fires on noise, not on signal failure. ATR-scaling lets the stop match the market's current rhythm.

**Effort**: Medium — the Python engine already has ATR trail infrastructure in `run_backtest_v10`.

**Confidence**: Medium — needs careful calibration. Too wide = bigger individual losses. Too tight = same clustering problem.

**Testable**: Yes, on 6-month data. Can sweep ATR multipliers.

---

## Solution 3: SM Flip Frequency as "Don't Trade" Signal

**Concept**: When SM oscillates rapidly around zero (many flips per day), SM itself is unreliable. Don't trade when your primary indicator is confused.

**Observation**: On Feb 13, SM flipped multiple times while price fell 235pts — that's SM confusion, not SM signal. When SM zero-crossings are frequent, both entries and exits degrade.

**Implementation**: Count SM zero-crossings in the last 60-120 minutes. If > N crossings, suppress new entries.

**Why it's different from failed regime prediction**: This isn't predicting tomorrow's regime from today's features (which failed — every metric flipped between train/test). It's a real-time observation that the indicator you're trading on is currently unreliable. It's like checking if your compass is spinning before trusting it.

**Effort**: Medium — need to compute SM flip frequency on 1-min bars and add as entry filter.

**Confidence**: Unknown — needs testing. Could be the first stable "regime" signal because it measures indicator quality, not market direction.

**Testable**: Yes, on 6-month data. Compute SM flip frequency, correlate with trade outcomes.

---

## Solution 4: Asymmetric Sizing (Shift to v15 During Drawdown)

**Concept**: v15 TP=5 has 86.7% win rate and captures near-certain 5pt moves. During drawdown periods, shift capital from v11 (SM flip, trend-following) to v15 (TP=5, scalping).

**Sizing rules**:
- Normal: v11 = 1 contract, v15 = 1 contract
- After 2 SL stops in rolling 5 days: v11 = 1 contract, v15 = 2 contracts
- After 4 SL stops: v11 = 0 contracts (pause), v15 = 2 contracts
- After 2 consecutive wins on v15: restore v11 to 1 contract

**Why it works**: v15's TP=5 captures the first 5pts of every entry (97%+ MFE hit rate). In choppy regimes where SM flip exits fail, the TP exit has already locked profit before chaos starts. Trail exits are *countercyclical* — they help in chop, hurt in trends — which is exactly what you want during drawdown periods.

**Effort**: Low — both strategies already run independently on TradingView.

**Confidence**: Medium-High — based on validated v15 results and proven countercyclical property of trailing stops.

**Testable**: Can simulate portfolio allocation rules on 6-month data.

---

## Solution 5: Anti-Martingale Position Sizing

**Concept**: Reduce size after losses, increase after wins. Classic risk management.

**Rules**:
- Start each day at 1 contract
- After each SL stop, reduce by 1 (minimum 0 = pause)
- After each win, add 1 back (maximum 2)
- Reset to 1 at start of each day

**Feb 12-17 simulation**:
```
Trade 61: SL (-$105) → size drops to 0 (pause)
Trade 62: skipped (would have been SL -$136, SAVED)
Trade 63: skipped (would have been SL -$169, SAVED)
... need a win to resume ...
```

**Why it works**: Skipping even 2-3 of the 8 SL stops saves $300-$400. The strategy has 58% WR, so a loss is more likely to be followed by a win than another loss — but SL stops cluster (as we've seen), so pausing after an SL is protective exactly when it matters.

**Effort**: Low — simple counter logic.

**Confidence**: High — well-established risk management principle.

**Testable**: Easy to simulate on 6-month trade list.

---

## Solution 6: Time-of-Day Gating After Losses

**Concept**: After losses, restrict entries to the highest-quality time window only.

**Observation from Feb 12-17 SL stops**:
```
Feb 12 13:38 SL   ← afternoon
Feb 12 14:12 SL   ← afternoon
Feb 13 10:17 SL   ← morning
Feb 13 11:24 SL   ← morning
Feb 13 13:53 SL   ← afternoon
Feb 17 10:46 SL   ← morning
Feb 17 12:33 SL   ← midday
Feb 17 14:59 SL   ← afternoon
```

**Rule**: After 2 SL stops in a day, restrict entries to 10:00-11:30 ET only (opening window when signals are freshest). Afternoon SL stops often come in deteriorating conditions where SM was already stressed from earlier whipsaws.

**Effort**: Low — simple time filter modification.

**Confidence**: Medium — needs analysis of time-of-day SL distribution across 6 months to validate.

**Testable**: Yes, can analyze SL stop time distribution and simulate restricted windows.

---

## Solution 7: Correlated Drawdown Hedge (MES as Insurance)

**Concept**: MES is naturally more resilient in hostile regimes. When MNQ is in drawdown, shift size to MES.

**Evidence**:
- Feb 12-17: MNQ lost -$598, MES lost only -$274
- MES has **zero Max Loss stops ever** (SL=0 is optimal for MES)
- MES v9.4 uses slower SM params (20/12/400/255) that don't whipsaw as easily
- MES PF over TV period: 1.555 (vs MNQ's 1.389)

**Rules**:
- Normal: MNQ = 1 contract, MES = 1 contract
- MNQ drawdown (2+ SL in 5 days): MNQ = 0-1, MES = 2 contracts
- MNQ drawdown resolved: restore normal sizing

**Why it works**: MES's slower SM params make it less sensitive to the same whipsaw conditions that destroy MNQ. It's natural diversification without prediction.

**Effort**: Low — both already running, just manual size adjustment.

**Confidence**: Medium — only 6 months of correlated data, and both products track the same underlying market.

**Testable**: Can simulate portfolio sizing rules using both instrument backtests.

---

## Solution 8: Dynamic Cooldown After SL Stops

**Concept**: After an SL stop, the market just made a violent 50pt move. Don't rush back in.

**Current**: Fixed 20-bar cooldown (20 minutes) after any exit.

**Proposed**: After SL stops specifically, double the cooldown:
- Normal exit: 20 bars
- After 1 SL: 40 bars
- After 2 SL same day: 80 bars (essentially skip to the next session phase)
- Reset at start of next day

**Why it works**: Re-entering 20 minutes after a 50pt loss often catches the continuation of that same adverse move. A longer cooldown lets the market establish a new regime. On Feb 13, the 3 SL stops came at 10:17, 11:24, and 13:53 — each roughly 60-90 minutes apart. A 40-bar cooldown after the first would have delayed the second entry, potentially improving the entry price or skipping it entirely.

**Effort**: Medium — needs changes to both Pine script and Python engine cooldown logic.

**Confidence**: Medium — needs careful testing. Too long = miss recovery trades. Too short = same problem.

**Testable**: Yes, can sweep dynamic cooldown parameters on 6-month data.

---

## Priority Matrix

| Solution | Effort | Confidence | Risk of Hurting Good Days |
|---|---|---|---|
| 1. Circuit breaker (2 SL/day) | Low | High | Low (rare trigger) |
| 5. Anti-martingale sizing | Low | High | Low (resets daily) |
| 4. Shift to v15 during drawdown | Low | Medium-High | Medium (caps upside) |
| 7. MES hedge sizing | Low | Medium | Low (MES still trades) |
| 6. Time-of-day gating | Low | Medium | Medium (may miss PM wins) |
| 8. Dynamic cooldown | Medium | Medium | Medium (delays re-entry) |
| 2. ATR-scaled stop | Medium | Medium | Medium (wider stops = bigger individual losses) |
| 3. SM flip frequency filter | Medium | Unknown | Unknown (novel, untested) |

**Recommended first steps**: Solutions 1 + 5 (circuit breaker + anti-martingale). They're the lowest effort, highest confidence, and compose well together. A simple "stop after 2 SL stops per day, reduce size after any SL" rule would have prevented the majority of the Feb 12-17 damage with minimal impact on good days.

---

## Combinations Worth Testing

**Combo A (Conservative)**: Circuit breaker + anti-martingale
- After 1 SL: size → 0, done for the day
- Simple, brutal, effective

**Combo B (Adaptive)**: ATR stop + dynamic cooldown + v15 shift
- Wider stops in volatile markets + longer cooldown after SL + lean on v15
- More complex but addresses root cause (stop too tight for regime)

**Combo C (Portfolio)**: v15 shift + MES hedge + circuit breaker
- Normal: 1 MNQ v11 + 1 MNQ v15 + 1 MES v9.4
- Drawdown: 0 MNQ v11 + 2 MNQ v15 + 2 MES v9.4
- Maximum diversification across exit types and instruments

---

## SELECTED PLAN: Adaptive Drawdown Protection

**Decision (Feb 18)**: Implement Solutions 1 + 4 + 5 together. Manual on TradingView for now, automated in SafetyManager later.

### The Rules

**Per-instrument, per-day state tracking:**
- `sl_count_today` — number of max-loss stop exits today
- `sl_count_5day` — rolling 5-day SL stop count
- `v11_size` — current v11 position size (starts at 1)
- `v15_size` — current v15 position size (starts at 1)

**After each v11 trade closes:**

```
IF exit_reason == "Max Loss":
    sl_count_today += 1
    sl_count_5day += 1

    IF sl_count_today == 1:
        v11_size = 0          # Anti-martingale: pause v11 for rest of day
        v15_size = 2          # Shift capital to v15 (scalp TP=5)
        LOG "v11 paused after 1st SL, v15 sized up to 2"

    IF sl_count_5day >= 4:
        v11_size = 0          # Extended pause
        v15_size = 2
        LOG "v11 paused: 4+ SL stops in 5 days"

ELIF exit_reason == "SM Flip" AND pnl > 0:
    # Win on SM flip = regime may be improving
    # But do NOT auto-resume v11 same day after SL pause
    pass

AT START OF EACH DAY:
    sl_count_today = 0
    # Only restore v11 if 5-day SL count is under threshold
    IF sl_count_5day < 4:
        v11_size = 1
        v15_size = 1
    ELSE:
        v11_size = 0          # Stay paused until 5-day window clears
        v15_size = 2
        LOG "v11 still paused: sl_count_5day={sl_count_5day}"

    # Age out old SL stops from 5-day window
    remove SL stops older than 5 trading days from sl_count_5day
```

### Why 1 SL = Pause (Not 2)

The original proposal said "after 2 SL stops in a day". But looking at the data:
- The worst single SL stop was -$169 (Feb 13 trade #63)
- Average SL loss is -$118 ($50pts * $2/pt + commission)
- After 1 SL, there's a ~40% chance the next trade is also SL (vs ~10% baseline SL rate)
- Pausing after 1 SL saves the 2nd (and possible 3rd) SL at the cost of 1 potential SM flip winner (~$12 avg)

The math is clear: expected loss from 2nd trade after SL = ~$47 (40% * $118), expected gain = ~$7 (60% * $12 avg). Pause after 1 is +EV.

### Manual TradingView Protocol

**Start of day:**
1. Check: did v11 have 4+ SL stops in the last 5 trading days?
2. If YES: keep v11 OFF, run v15 at 2 contracts
3. If NO: run v11 at 1 contract, v15 at 1 contract

**During the day (MNQ v11):**
1. v11 hits a Max Loss stop → immediately disable v11 strategy on TradingView
2. Change v15 from 1 contract to 2 contracts (Properties > Qty)
3. v11 stays off for the rest of the day
4. Note the SL stop in your trading log for the 5-day rolling count

**End of day:**
1. Re-enable v11 for next day IF 5-day SL count < 4
2. Reset v15 back to 1 contract
3. MES v9.4 is UNAFFECTED by this protocol (runs independently, no SL stops)

### SafetyManager Implementation Plan

**File**: `live_trading/engine/safety.py`

**New state fields in SafetyState:**
```python
@dataclass
class SafetyState:
    # ... existing fields ...

    # Adaptive drawdown protection (per instrument)
    sl_stops_today: dict[str, int] = field(default_factory=dict)     # {instrument: count}
    sl_stops_5day: dict[str, list] = field(default_factory=dict)     # {instrument: [dates]}
    strategy_sizes: dict[str, int] = field(default_factory=dict)     # {strategy_name: qty}
    drawdown_mode: bool = False
    drawdown_since: Optional[datetime] = None
```

**New config fields in SafetyConfig:**
```python
@dataclass
class SafetyConfig:
    # ... existing fields ...

    # Adaptive drawdown protection
    sl_pause_threshold_daily: int = 1      # SL stops before pausing trend strategy
    sl_pause_threshold_5day: int = 4       # Rolling 5-day SL count threshold
    scalp_size_normal: int = 1             # v15 qty in normal mode
    scalp_size_drawdown: int = 2           # v15 qty in drawdown mode
    trend_strategy: str = "v11"            # Strategy to pause on SL cluster
    scalp_strategy: str = "v15"            # Strategy to size up during pause
```

**New method in SafetyManager:**
```python
def _on_trade_closed(self, trade: TradeRecord) -> None:
    # ... existing P&L tracking ...

    # Adaptive drawdown: track SL stops per instrument
    if trade.exit_reason == "SL":
        inst = trade.instrument
        self.state.sl_stops_today[inst] = self.state.sl_stops_today.get(inst, 0) + 1
        self.state.sl_stops_5day.setdefault(inst, []).append(datetime.now().date())

        if self.state.sl_stops_today[inst] >= self._config.sl_pause_threshold_daily:
            self._enter_drawdown_mode(inst)

    # ... existing circuit breaker checks ...

def _enter_drawdown_mode(self, instrument: str) -> None:
    """Shift from trend strategy to scalp strategy."""
    self.state.drawdown_mode = True
    self.state.drawdown_since = datetime.now()
    self.state.strategy_sizes[self._config.trend_strategy] = 0
    self.state.strategy_sizes[self._config.scalp_strategy] = self._config.scalp_size_drawdown

    logger.warning(
        f"[Safety] DRAWDOWN MODE: {instrument} hit "
        f"{self.state.sl_stops_today[instrument]} SL stops today. "
        f"Pausing {self._config.trend_strategy}, "
        f"sizing up {self._config.scalp_strategy} to "
        f"{self._config.scalp_size_drawdown}"
    )
    self._event_bus.emit("drawdown_mode", {
        "active": True,
        "instrument": instrument,
        "trend_size": 0,
        "scalp_size": self._config.scalp_size_drawdown,
    })

def check_order_allowed(self, symbol: str, qty: int, strategy: str = "") -> tuple[bool, str]:
    """Extended to check strategy-specific sizing in drawdown mode."""
    # ... existing checks ...

    # Drawdown mode: enforce strategy-specific sizing
    if self.state.drawdown_mode and strategy:
        max_qty = self.state.strategy_sizes.get(strategy, qty)
        if qty > max_qty:
            return False, (
                f"Drawdown mode: {strategy} limited to {max_qty} contracts "
                f"(requested {qty})"
            )

    return True, ""

def reset_daily(self) -> None:
    """Extended daily reset with 5-day window aging."""
    # ... existing reset ...

    # Age out SL stops older than 5 trading days
    cutoff = datetime.now().date()  # Will need proper trading day calc
    for inst in self.state.sl_stops_5day:
        self.state.sl_stops_5day[inst] = [
            d for d in self.state.sl_stops_5day[inst]
            if (cutoff - d).days <= 5
        ]

    # Check if drawdown mode should continue
    for inst, dates in self.state.sl_stops_5day.items():
        if len(dates) >= self._config.sl_pause_threshold_5day:
            self._enter_drawdown_mode(inst)
        else:
            if self.state.drawdown_mode:
                self.state.drawdown_mode = False
                self.state.strategy_sizes[self._config.trend_strategy] = 1
                self.state.strategy_sizes[self._config.scalp_strategy] = self._config.scalp_size_normal
                logger.info(f"[Safety] Drawdown mode CLEARED for {inst}")
```

**Runner integration** (`engine/runner.py`):
- Before emitting a signal, call `safety.check_order_allowed(symbol, qty, strategy_name)`
- If drawdown mode is active and strategy is the trend strategy, the order is blocked
- The scalp strategy picks up the signal with its increased size

**Dashboard integration** (`dashboard/`):
- Show drawdown mode status (yellow banner when active)
- Show per-strategy sizing (v11: 0, v15: 2 when in drawdown mode)
- Show rolling 5-day SL count

### Feb 12-17 Simulation

With this system active, the week would have played out:

```
Feb 12 Trade #61: v11 SL stop → drawdown mode ON
  v11 → 0 contracts, v15 → 2 contracts
  sl_count_today=1, sl_count_5day=1

Feb 12 Trade #62: v11 BLOCKED (saved -$136)
  v15 takes the signal at 2 contracts → TP=5 likely hits → +$18

Feb 13 morning: sl_count_5day=1, < 4 threshold
  v11 restored to 1 contract, v15 back to 1

Feb 13 Trade #63: v11 SL stop → drawdown mode ON again
  v11 → 0, v15 → 2
  sl_count_today=1, sl_count_5day=2

Feb 13 Trade #64: v11 BLOCKED (saved -$152)
Feb 13 Trade #65: v11 BLOCKED (saved -$103)

Feb 14 (Friday): sl_count_5day=3, < 4 → v11 restored

Feb 17 (Monday): sl_count_5day=3 (Feb 12 SL still in window)
  v11 at 1 contract

Feb 17 Trade #69: v11 SL stop → drawdown mode ON
  sl_count_5day=4 → EXTENDED PAUSE
  v11 → 0, v15 → 2

Feb 17 Trade #71: v11 BLOCKED (saved -$135)
Feb 17 Trade #72: v11 BLOCKED (saved -$124)

ESTIMATED SAVINGS: ~$650 in avoided SL stops
ESTIMATED COST:    ~$50-80 in missed v11 winners (v15 partially compensates)
NET BENEFIT:       ~$570-600 better than actual
```

Instead of -$598 for the week, the system would have produced roughly **-$30 to +$50**.
