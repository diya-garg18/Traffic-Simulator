# Code Explained

Deep, file-by-file, line-group-by-line-group explanation of everything in
this project. Goal: you should be able to read this and then explain any
line of the actual code without re-deriving it from scratch. Updated as
each file is built.

---

## `traffic_env.py`

### Module purpose

Defines the entire MDP (Markov Decision Process) for one 4-way
intersection: what a "state" is, what actions exist, how the world changes
each step, and how reward is computed. This is the ONLY file that knows the
true simulation rules — every agent/controller interacts with it only
through `reset()` and `step()`, never by reaching into its internals
(except `value_iteration.py`, deliberately, since it needs the known model).

### `ACTION_KEEP`, `ACTION_SWITCH`, `ACTIONS`

```python
ACTION_KEEP = "KEEP"
ACTION_SWITCH = "SWITCH"
ACTIONS = (ACTION_KEEP, ACTION_SWITCH)
```

The entire action space of the MDP. There are exactly two things the
controller can do each step: leave the light as-is, or flip it. These are
plain string constants rather than an `Enum` — see `DECISIONS.md` for why.
`ACTIONS` is a tuple (immutable) used both for validation (`step()` checks
`action in ACTIONS`) and could be iterated over by an agent that needs to
try "all possible actions from this state" (e.g. picking `argmax_a Q(s,a)`).

### `DEPARTURE_RATE`, `SWITCH_PENALTY`, `TIME_SINCE_SWITCH_CAP`

Module-level constants instead of magic numbers buried in methods, so they
can be found, tuned, and referenced from `DECISIONS.md` in one place.
- `DEPARTURE_RATE = 3`: how many cars leave per step on the green road.
- `SWITCH_PENALTY = 5`: extra reward cost for choosing SWITCH.
- `TIME_SINCE_SWITCH_CAP = 20`: ceiling on the raw "steps since last
  switch" counter, so it never grows unboundedly even though only the
  bucketed version (SHORT/MEDIUM/LONG) is ever seen by an agent.

### `State` (dataclass)

```python
@dataclass
class State:
    cars_waiting_ns: int
    cars_waiting_ew: int
    current_light: str
    time_since_last_switch: int
```

A plain data container for the **raw, true** state of the intersection.
`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from the
field list — this is standard library, not a custom abstraction, and it
means the constructor (`State(cars_waiting_ns=0, ...)`) is self-documenting
via keyword arguments. This is mutated in place by `TrafficEnv.step()`
(each field reassigned as the simulation progresses) rather than
reconstructed fresh each step, which is simpler and avoids allocating a new
object every tick.

### `TrafficEnv.__init__`

Stores the configuration (`rate_ns`, `rate_ew`, `max_steps`, `rush_hour`)
and creates `self._rng = np.random.default_rng(seed)` — NumPy's modern
random generator API. Using a `Generator` object (instead of the older
global `np.random.seed()`) means multiple `TrafficEnv` instances can have
independent, non-interfering random streams, which matters once
`evaluate.py` needs several environments seeded identically for a fair
comparison. `self.state: State` and `self.current_step: int` are declared
with type hints but not assigned yet — they're populated by the call to
`self.reset()` at the end of `__init__`, so a freshly constructed
`TrafficEnv` is immediately usable.

### `TrafficEnv.reset()`

Rebuilds `self.state` from scratch: both queues at 0, `"NS"` green by
default, `time_since_last_switch` at 0, and resets `self.current_step` to
0. Returns `self.discretize_state(self.state)` — callers always get the
discretized (bucketed) view, never the raw `State` object, because that's
the only thing a Q-learning agent (or any controller) is supposed to see.

### `TrafficEnv.step(action)`

The heart of the simulator. Runs in four strict phases every call:

1. **Arrivals.** `self._current_arrival_rates()` returns whatever
   `(rate_ns, rate_ew)` apply right now (fixed, or rush-hour-adjusted — see
   below), then `self._rng.poisson(rate)` draws a random non-negative
   integer from a Poisson distribution with that rate as its mean. This
   models "on average `rate` cars arrive per step, but the exact number is
   random," which is the standard way to model independent arrival events
   (this is literally what a Poisson process is for). Both roads get
   arrivals added regardless of the action — arrivals don't care what the
   light does.

2. **Apply the action.** If `SWITCH`: flip `current_light` between `"NS"`
   and `"EW"` using a conditional expression, and reset the timer to 0. If
   `KEEP`: increment the timer by 1, clamped at `TIME_SINCE_SWITCH_CAP` via
   `min(...)`.

3. **Departures.** Whichever road `current_light` now points to (note:
   *after* step 2, so a SWITCH this very step lets the newly-green road
   depart cars immediately) loses `min(DEPARTURE_RATE, <that road's queue>)`
   cars. The `min()` is what prevents the queue from going negative when
   fewer than `DEPARTURE_RATE` cars are actually waiting.

4. **Reward.** `total_waiting = ns + ew` after all of the above;
   `reward = -total_waiting`, minus `SWITCH_PENALTY` if the action was
   SWITCH. Framed as a cost-minimization signal even though RL maximizes
   reward — negative cost is a completely standard framing (see
   `DECISIONS.md` for the reasoning).

Finally, `self.current_step` increments and `done = current_step >=
max_steps` is computed. The method returns the discretized next state,
the float reward, and the `done` flag — the exact tuple shape every
controller expects.

### `TrafficEnv.discretize_state(state)`

Takes a `State` object and returns a 4-tuple of strings:
`(ns_bucket, ew_bucket, current_light, time_bucket)`. This tuple is
hashable (tuples of strings always are), which is exactly what's needed to
use it as half of a dictionary key in the Q-table (`Q[(state, action)]`).
The docstring inside the method itself derives the resulting state-space
size (3 × 3 × 2 × 3 = 54) — worth memorizing, since "why is the state space
54 and is that a good size for tabular Q-learning" is a very likely
interview question.

### `TrafficEnv._bucket_cars(count)` / `_bucket_time(steps)`

Both `@staticmethod` — they don't touch `self` at all, they're pure
functions of their single input, so marking them static documents that
fact and lets them be called as `TrafficEnv._bucket_cars(5)` without an
instance if ever useful (e.g. in a unit test). Simple `if/elif/else`
threshold checks; see `DECISIONS.md` for why the specific cut points
(3/8 for cars, 2/7 for time) were chosen.

### `TrafficEnv._current_arrival_rates()`

Returns the fixed `(rate_ns, rate_ew)` unless `self.rush_hour` is `True`,
in which case it computes `phase_length = max_steps // 3` and
`phase = current_step // phase_length` (clamped to a max of 2 via `min(...,
2)` so integer-division rounding at the very end of the episode can't push
`phase` to 3 and fall through with no return value). Each of the three
phases returns a different multiple of the base rates — see `DECISIONS.md`
for the reasoning behind the specific multipliers (2x/0.5x, 1x/1x,
0.5x/2x).

### `if __name__ == "__main__":` block

Only runs when the file is executed directly (`python traffic_env.py`),
not when imported by another module (`import traffic_env` will NOT trigger
this). Builds one `TrafficEnv`, resets it, and steps through a hand-picked
sequence of 10 actions (mostly KEEP with a few SWITCH mixed in), printing a
formatted table of raw state + reward + discretized state after every step.
This is the "print a few manual steps" sanity check the project spec asked
for, and is the artifact that was reviewed and confirmed before any agent
code was written (see `docs/FEATURES.md`).

---

## Other files (not yet built)

This section will be filled in as each file is written:
`q_learning_agent.py`, `baseline_controller.py`, `value_iteration.py`,
`evaluate.py`, `visualize.py`.
