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

---

## Feature: Tabular Q-learning agent (`q_learning_agent.py`)

**Scoped from:** `prompt.txt` section "2. Q-Learning Agent".

**What was built:**
- `QLearningAgent` class with a `defaultdict(float)`-backed Q-table keyed by
  `(discretized_state, action)`.
- `epsilon_for_episode()`: linear decay from `epsilon_start` (1.0) to
  `epsilon_end` (0.05) over `epsilon_decay_episodes`, then held flat.
- `choose_action()`: epsilon-greedy selection.
- `best_action()` / `max_q_value()`: greedy argmax / max helpers used by both
  action selection and the update rule.
- `update()`: the Bellman equation update, `Q(s,a) <- Q(s,a) + alpha*(r +
  gamma*max_a' Q(s',a') - Q(s,a))`, with each intermediate value
  (`current_q`, `best_next_q`, `td_target`, `td_error`, `new_q`) broken into
  its own named line and commented against the equation.
- `train()`: runs full episodes against a `TrafficEnv`, updating online after
  every step, returning per-episode total reward for later plotting.
- Configurable hyperparameters (alpha, gamma, epsilon schedule) — see
  `DECISIONS.md` for the chosen defaults and reasoning.

**What was tried / verified:**
- Ran `python q_learning_agent.py`: trained 500 episodes on a fixed seed
  (`rate_ns=0.8, rate_ew=0.5, max_steps=100, seed=7`).
- Confirmed epsilon decay is monotonic and hits the right values: 1.000 at
  episode 0, 0.525 at episode 150 (roughly the midpoint of the 300-episode
  decay window), 0.050 at episode 500 (floor, past the decay window).
- Confirmed learning is actually happening: average total reward per episode
  rose from -423.0 (first 50 episodes) to -294.0 (last 50 episodes) — a
  clear, large improvement, not noise.
- Q-table ended with 54 visited (state, action) pairs out of a possible 108
  (54 states x 2 actions) — expected, since low arrival rates mean HIGH-queue
  states are rarely if ever visited in this particular scenario.

**Status:** Smoke-tested, learning confirmed via reward trend. Deeper
validation (does the resulting policy actually beat a naive baseline?) is
deferred to `evaluate.py`.

---

## Feature: Fixed-timer baseline (`baseline_controller.py`)

**Scoped from:** `prompt.txt` section "3. Baseline Comparison".

**What was built:**
- `FixedTimerController` class: switches every `switch_every` steps
  (default 10), tracked via its own internal counter, ignoring the
  environment's state entirely.
- `choose_action(state)` keeps the same call shape as
  `QLearningAgent.choose_action` (even though `state` is unused) so
  `evaluate.py` can swap controllers without special-casing either one.
- `reset()` to resync the controller's internal clock with a fresh episode.

**What was tried / verified:**
- Ran `python baseline_controller.py`: 20 steps, `switch_every=5`, fixed
  seed. Confirmed SWITCH fires exactly at steps 4, 9, 14, 19 (every 5th
  step, 0-indexed) regardless of queue size — e.g. step 17 shows 7 cars
  waiting on EW and the controller still doesn't switch early, proving it
  truly ignores traffic.

**Status:** Fully verified.

---

## Feature: Value iteration planner (`value_iteration.py`, stretch goal)

**Scoped from:** `prompt.txt` section "4. Dynamic Programming Comparison".

**What was built:**
- `_build_road_transition_table(rate)`: precomputes, per road, a
  bucket-to-bucket transition distribution and expected count-after, using
  one representative raw count per bucket (LOW=2, MEDIUM=6, HIGH=12) and
  summing the Poisson arrival distribution (via `scipy.stats.poisson.pmf`)
  up to `MAX_ARRIVALS_TO_SUM=30`.
- `ValueIteration` class: builds the 378-state finite MDP
  (3 car-buckets x 3 car-buckets x 2 lights x 21 exact time-counter values),
  runs the Bellman optimality update to convergence (`solve()`), and
  extracts the greedy policy.
- `choose_action(env)`: reads `env.state` directly (raw counts, exact time
  counter) — the one module allowed to do this per `ARCHITECTURE.md`.

**What was tried / verified:**
- Ran `python value_iteration.py`: converged in 197 iterations (well under
  the 1000-iteration cap), policy covers all 378 states.
- Ran the resulting policy against the REAL environment (same seed=42,
  rates as `traffic_env.py`'s own trace). Cross-checked the printed raw
  counts step-by-step against the original `traffic_env.py` trace (same
  seed) to confirm the environment's random arrival draws are positional
  (depend only on step index, not action history) — confirmed consistent,
  ruling out an RNG/state-tracking bug.
- **Found and diagnosed a real model limitation, not a code bug:** the
  policy sometimes refuses to SWITCH out of a badly backed-up HIGH queue,
  and visibly flickers (SWITCH immediately followed by SWITCH back) when
  run against the real environment. Traced this to the HIGH bucket's
  single representative count (12): departing 3 cars (12 -> 9) still
  counts as HIGH, so the model's future-value term sees no benefit from
  switching, only the -5 switch-penalty cost. Verified directly by
  printing `Q(KEEP)=-263.71` vs `Q(SWITCH)=-276.81` at state
  `('LOW','HIGH','NS',10)` and confirming the underlying reward/transition
  breakdown matches this explanation exactly. Full writeup in
  `DECISIONS.md`.

**Status:** Code verified correct; the resulting POLICY has a known,
diagnosed, documented limitation stemming from the bucket-approximation
model, not a bug. This is intentionally kept (not patched around) because
it's a genuinely useful, concrete illustration of "planning is only as
good as its model" for the interview-prep discussion — see `DECISIONS.md`.
