# Features

One entry per feature/component, start to finish: how it was scoped, what
was tried, what worked, how it was verified. Anyone should be able to read
one entry cold and know exactly how that piece was built and validated.

---

## Feature: Environment simulator (`traffic_env.py`)

**Scoped from:** `prompt.txt` section "1. Environment Simulator".

**What was built:**
- `State` dataclass: `cars_waiting_ns`, `cars_waiting_ew`, `current_light`,
  `time_since_last_switch`.
- `TrafficEnv` class with `reset()` and `step(action)` following the
  standard `(next_state, reward, done)` RL interface.
- `discretize_state()` bucketing counts into LOW/MEDIUM/HIGH and time into
  SHORT/MEDIUM/LONG (see `DECISIONS.md` for the exact thresholds and why).
- Poisson arrivals with independently configurable `rate_ns`/`rate_ew`.
- Fixed-capacity departures (`DEPARTURE_RATE = 3`) on whichever road is
  currently green.
- Reward = `-(total waiting)` with an extra `-SWITCH_PENALTY` (5) on SWITCH.
- Rush-hour mode: 3-phase time-varying arrival rate schedule.

**What was tried / verified:**
- Ran the file directly (`python traffic_env.py`), which executes a
  10-step manual trace with a fixed seed (42) and a mixed KEEP/SWITCH
  action sequence, printing raw state + discretized state + reward every
  step.
- Confirmed by inspection:
  - Cars only accumulate (never decrease) on the red road.
  - The green road's count only decreases, capped at `DEPARTURE_RATE`.
  - SWITCH resets `time_since_last_switch` to 0 and costs an extra -5
    reward on top of the queue cost.
  - Discretized buckets flip at the documented thresholds (e.g. NS count
    hitting 4 moves the bucket from LOW to MEDIUM).
- User reviewed the printed trace and confirmed the behavior looked
  sensible before agent work began (per the explicit "confirm before
  building the agent" instruction in `prompt.txt`).

**Additional verification (2026-09-05):**
- `done` becomes `True` exactly on the step where `current_step` reaches
  `max_steps` (checked with `max_steps=5`: `done` is `False` for steps
  0-3, `True` on step 4).
- Rush-hour schedule confirmed transitioning correctly across a
  `max_steps=9` episode (phase length 3): steps 0-3 return
  `(2.0, 0.5)` (NS-heavy), steps 4-6 return `(1.0, 1.0)` (balanced),
  steps 7-8 return `(0.5, 2.0)` (EW-heavy).

**Status:** Fully verified — core mechanics, episode termination, and
rush-hour phase transitions all confirmed working.
