---
name: Session bar epochs are ET not UTC
description: CRITICAL — session JSON bar timestamps are ET wall-clock encoded as Unix epoch, NOT UTC. Always use entry_time_et_epoch for lookups.
type: feedback
---

## The Bug That Keeps Happening

Session JSON files (`sessions/session_*.json`) store bar `time` fields as **ET (Eastern Time) wall-clock timestamps naively encoded as Unix epochs**. They are NOT UTC.

This means:
- Bar epoch 1773226860 represents **11:01 ET** (not 15:01 UTC)
- Trade `entry_time` is stored as UTC ISO 8601 (`2026-03-11T15:01:00+00:00`)
- Trade `entry_time_et_epoch` is the correct key to look up bars

**WRONG** (what I keep doing):
```python
target = datetime(2026, 3, 11, 15, 1, 0, tzinfo=timezone.utc).timestamp()  # UTC epoch
# This finds a bar 4 hours LATER in the day
```

**RIGHT**:
```python
et_epoch = trade['entry_time_et_epoch']  # Use this field directly
# Or convert: take the ET wall-clock time, treat as naive, get epoch
```

This has caused false alarms and incorrect MFE calculations multiple times. On Mar 11, 2026, it created a phantom "32-point data discrepancy" that wasted investigation time.

**Always use `entry_time_et_epoch` when looking up bars in session files.**
