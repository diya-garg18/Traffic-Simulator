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

## `q_learning_agent.py`

### Module purpose

Implements Q-learning itself: how the agent picks actions, and how it
updates its belief about how good each action is in each state, purely
from trial and error against `traffic_env.TrafficEnv` — no knowledge of
arrival rates or departure rates is used anywhere in this file.

### `StateType`

```python
StateType = Tuple[str, str, str, str]
```
Just a type-hint alias for "whatever `discretize_state()` returns" — makes
function signatures below (`choose_action`, `update`, etc.) readable
without repeating `Tuple[str, str, str, str]` everywhere.

### `QLearningAgent.__init__`

Stores the five hyperparameters (`alpha`, `gamma`, `epsilon_start`,
`epsilon_end`, `epsilon_decay_episodes`) exactly as passed in — no
validation/clamping, because this is a coursework tool used by one person
who is expected to pass sane values. Creates `self._rng`, a NumPy
`Generator`, used for every random decision the *agent* makes (as opposed
to `self._rng` inside `TrafficEnv`, which is a separate, independent
stream — this separation means changing how many random numbers the
environment consumes doesn't shift which random numbers the agent
consumes, keeping experiments reproducible and separable).

`self.q_table: Dict[..., float] = defaultdict(float)` is the entire
"brain" of the agent. A normal `dict` would raise `KeyError` the first
time you looked up a `(state, action)` pair that's never been seen;
`defaultdict(float)` instead silently creates that entry with value `0.0`
on first access. This is exactly the standard "initialize Q-values to
zero" step in the Q-learning algorithm, just implemented via a Python
language feature instead of a manual pre-fill loop over all 108 possible
keys.

### `epsilon_for_episode(episode)`

```python
if episode >= self.epsilon_decay_episodes:
    return self.epsilon_end
progress = episode / self.epsilon_decay_episodes
return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
```
Straight-line (linear) interpolation between `epsilon_start` and
`epsilon_end`. At `episode=0`, `progress=0`, so it returns exactly
`epsilon_start`. As `episode` approaches `epsilon_decay_episodes`,
`progress` approaches 1, so it approaches exactly `epsilon_end`. Past that
point, the `if` clamps it flat at `epsilon_end` forever — without this
clamp, the formula would keep extrapolating past `epsilon_end` (e.g. going
negative), which makes no sense for a probability.

### `choose_action(state, epsilon)`

```python
if self._rng.random() < epsilon:
    return self._rng.choice(ACTIONS)
return self.best_action(state)
```
`self._rng.random()` draws a uniform random float in `[0, 1)`. If it lands
below `epsilon`, that's the "explore" branch — pick uniformly at random
from `ACTIONS` (`KEEP` or `SWITCH`), ignoring the Q-table entirely. This
happens with probability exactly `epsilon`, which is the definition of
epsilon-greedy. Otherwise ("exploit" branch, probability `1 - epsilon`),
defer to `best_action`, which looks at the Q-table.

### `best_action(state)`

```python
q_keep = self.q_table[(state, ACTIONS[0])]
q_switch = self.q_table[(state, ACTIONS[1])]
return ACTIONS[0] if q_keep >= q_switch else ACTIONS[1]
```
Reads both Q-values for this state directly out of the dict (triggering
`defaultdict`'s 0.0 default for either one if unseen), and returns
whichever action has the higher value. `>=` (not `>`) means ties go to
`KEEP` — an arbitrary but deterministic tie-break, which matters for a
never-visited state where both values are 0.0 and would otherwise tie.

### `max_q_value(state)`

```python
return max(self.q_table[(state, a)] for a in ACTIONS)
```
This is `max_a' Q(s', a')` from the Bellman equation, computed directly:
loop over both possible actions for the given state, take the larger
Q-value. Used only inside `update()`, always applied to the *next* state.

### `update(state, action, reward, next_state)` — the Bellman update

This is the function the whole project is built to be able to explain.
Line by line:

```python
current_q = self.q_table[(state, action)]
```
`Q(s, a)` — the agent's current, possibly wrong, estimate of how good it
is to take `action` in `state`. This is what's about to get corrected.

```python
best_next_q = self.max_q_value(next_state)
```
`max_a' Q(s', a')` — the best value the agent currently believes is
achievable from wherever it ended up (`next_state`), assuming it acts
optimally from then on. This is the "look one step into the future"
component of the update.

```python
td_target = reward + self.gamma * best_next_q
```
The "TD target" — what `Q(s, a)` *should* be, based on this one real
transition: the reward actually received, plus the discounted value of
the best thing achievable next. `gamma` (0.95 by default) controls how
much that future value counts relative to the immediate reward.

```python
td_error = td_target - current_q
```
The "TD error" — the gap between what the estimate should be
(`td_target`) and what it currently is (`current_q`). If this is 0, the
old estimate was already exactly right for this transition and nothing
changes below.

```python
new_q = current_q + self.alpha * td_error
```
Move the estimate toward the target, but only by a fraction `alpha`
(0.1 by default) of the gap — not all the way. This is what makes
learning stable when transitions are noisy (Poisson-random arrivals mean
the same `(state, action)` pair won't always lead to the same reward or
next state) — a small step size averages out the noise over many visits
instead of overreacting to any single (possibly unlucky) transition.

```python
self.q_table[(state, action)] = new_q
```
Write the corrected estimate back into the table, overwriting the old
one. Next time this exact `(state, action)` pair is visited, `current_q`
in the next call will start from this updated value.

**Reciting this from memory:** the five named intermediate variables
(`current_q`, `best_next_q`, `td_target`, `td_error`, `new_q`) exist
specifically so each line of the equation
`Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a'Q(s',a') - Q(s,a)]` has a
one-to-one, nameable counterpart in code — that's the mapping to practice
reciting on a whiteboard.

### `train(env, num_episodes)`

```python
for episode in range(num_episodes):
    state = env.reset()
    epsilon = self.epsilon_for_episode(episode)
    total_reward = 0.0
    done = False
    while not done:
        action = self.choose_action(state, epsilon)
        next_state, reward, done = env.step(action)
        self.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    episode_rewards.append(total_reward)
```
One episode = one full call to `env.reset()` followed by stepping until
`done`. Epsilon is computed once per episode (not per step) — the agent's
exploration rate stays constant for the whole episode, then drops for the
next one, per the schedule. Each step follows the standard RL loop: pick
an action, take it, learn from what happened (`update`), advance
(`state = next_state`), and accumulate `total_reward` purely for logging —
it plays no role in the learning itself (that's entirely driven by
per-step `update()` calls). The list of per-episode totals is returned so
`evaluate.py` can later plot how reward changes across training.

### `if __name__ == "__main__":` block

Trains one agent for 500 episodes on a fixed-seed environment, then prints
the Q-table size and average reward for the first vs. last 50 episodes, so
an improving (less negative) trend can be checked immediately without
needing `evaluate.py`'s plotting yet. Also prints epsilon at three points
to sanity-check the decay schedule directly.

---

## `baseline_controller.py`

### Module purpose

The "dumb" comparison point: a controller that switches on a fixed
schedule and never looks at traffic at all. This is what makes it possible
to later claim "the learned policy is better than X" with X being a
concrete, well-defined thing, not a strawman.

### `FixedTimerController.__init__` / `reset`

Stores `switch_every` and initializes `steps_since_switch = 0`.
`reset()` exists as its own method (rather than just re-running `__init__`
logic) because `evaluate.py` will call it every time `env.reset()` is
called, to keep the controller's internal clock in sync with a fresh
episode — without this, the controller's very first action of episode 2
would be based on however many steps happened to elapse at the end of
episode 1, which would make the "switches every N steps" guarantee false
across episode boundaries.

### `choose_action(state)`

```python
if self.steps_since_switch >= self.switch_every - 1:
    self.steps_since_switch = 0
    return ACTION_SWITCH
self.steps_since_switch += 1
return ACTION_KEEP
```
`state` is accepted as a parameter but never read inside the method body —
this is deliberate (see `DECISIONS.md`), purely to keep the same call
shape as `QLearningAgent.choose_action(state, epsilon)` minus the epsilon
argument, so `evaluate.py` can call `controller.choose_action(state)` on
whichever controller is currently active without an if/else branch per
controller type.

The `>= switch_every - 1` (not `>= switch_every`) is what makes a light
held for exactly `switch_every` steps: counting starts at 0, so after
`switch_every - 1` KEEP-equivalent ticks have already passed since the
last switch, this step should be the switch. Off-by-one here is exactly
the kind of thing to double check with a manual trace, which is what the
`__main__` block below does.

### `if __name__ == "__main__":` block

Runs the controller against a real `TrafficEnv` for 20 steps with
`switch_every=5`, printing the light and both raw queue counts every step,
specifically so the "switches exactly every 5 steps, ignores queue size"
claim can be checked by eye (see `TEST_CHECKLIST.md` for the exact
confirmed step numbers).

---

## `value_iteration.py`

### Module purpose

Computes the mathematically optimal policy directly from a KNOWN model of
the environment's dynamics — no trial and error, no sampling, no
`env.step()` calls during learning at all. This is "planning," the
opposite paradigm from `q_learning_agent.py`'s "learning."

### Why it needs its own approximate model

`TrafficEnv`'s true state (exact car counts) is unbounded, so an exact
dynamic-programming sweep over "every possible state" is literally
impossible — there are infinitely many. This file works around that by
planning over the same LOW/MEDIUM/HIGH bucket abstraction Q-learning uses,
picking one "representative" raw count per bucket (`BUCKET_REPRESENTATIVE_COUNT`)
to stand in for "any count in this bucket" when computing arrival
probabilities. This is a real approximation with a real, observed cost —
see the "HIGH bucket" discussion below and `DECISIONS.md`.

### `CAR_BUCKETS`, `LIGHTS`, `BUCKET_REPRESENTATIVE_COUNT`, `MAX_ARRIVALS_TO_SUM`

Module-level constants. `BUCKET_REPRESENTATIVE_COUNT = {"LOW": 2, "MEDIUM":
6, "HIGH": 12}` picks the middle of each closed range (LOW 0-3, MEDIUM
4-8) and a fixed anchor above HIGH's open-ended lower bound (9+).
`MAX_ARRIVALS_TO_SUM = 30` truncates the (infinite) Poisson sum used below
— 30 is far more than enough for the arrival rates used in this project,
so the truncated-away probability mass is negligible.

### `VIState`

```python
VIState = Tuple[str, str, str, int]
```
This planner's own state representation: `(ns_bucket, ew_bucket, light,
exact_time_since_switch)`. Note the LAST element is an `int` (0-20), not a
SHORT/MEDIUM/LONG string like the Q-learning agent's states. Time-since-
switch transitions are fully deterministic given the action (no
randomness involved at all), so there's no approximation cost to keeping
it exact — only the two stochastic car-count dimensions need bucketing.

### `_build_road_transition_table(rate)`

For ONE road, precomputes a lookup table keyed by `(current_bucket,
is_green)`, each entry holding: (a) a probability distribution over the
NEXT bucket, and (b) the expected raw count after this step (used later
for reward). Built once per road before value iteration starts, since a
road's own arrivals/departures don't depend on the other road or on
light/time at all — only on its own bucket and whether it's currently
green.

Inner loop:
```python
for arrivals in range(MAX_ARRIVALS_TO_SUM + 1):
    p_arrivals = poisson.pmf(arrivals, rate)
    count_after_arrival = representative_count + arrivals
    if is_green:
        departing = min(DEPARTURE_RATE, count_after_arrival)
        count_after = count_after_arrival - departing
    else:
        count_after = count_after_arrival
    next_bucket = TrafficEnv._bucket_cars(count_after)
    next_bucket_probs[next_bucket] += p_arrivals
    expected_count_after += p_arrivals * count_after
```
This mirrors `TrafficEnv.step()`'s own arrival-then-departure logic
exactly (reusing `TrafficEnv._bucket_cars` directly rather than
duplicating the threshold logic — one source of truth for what LOW/
MEDIUM/HIGH mean). For every possible number of arrivals (weighted by how
likely that many arrivals actually is, `poisson.pmf`), compute what bucket
the road ends up in, and accumulate that probability into
`next_bucket_probs`. `expected_count_after` accumulates the
probability-weighted raw count, used later as the expected reward
contribution. After the loop, probabilities are renormalized to sum to
exactly 1.0 (correcting for the small mass lost to truncation).

### `ValueIteration.__init__`

Builds both roads' transition tables up front, enumerates all 378 states
(`itertools.product` over the two 3-bucket dimensions, 2 lights, and 21
time-counter values), and initializes every state's value to 0.0 — the
standard value-iteration starting point (like Q-learning's zero-initialized
Q-table, but here it's a full sweep table, not learned incrementally).

### `_next_light_and_time(state, action)`

Purely deterministic — applying an action to `(light, time_since_switch)`
never depends on car counts. Exactly mirrors the corresponding lines in
`TrafficEnv.step()`'s "apply the action" phase.

### `_q_value(state, action)` — one Bellman backup

```python
reward = -(ns_expected_count + ew_expected_count)
if action == ACTION_SWITCH:
    reward -= SWITCH_PENALTY

expected_future_value = 0.0
for next_ns, p_ns in ns_probs.items():
    for next_ew, p_ew in ew_probs.items():
        next_state = (next_ns, next_ew, next_light, next_time)
        expected_future_value += p_ns * p_ew * self.value[next_state]

return reward + self.gamma * expected_future_value
```
This is `R(s,a) + gamma * sum_s' P(s'|s,a) * V(s')` computed directly.
Because NS and EW arrivals are independent random variables, the joint
probability of landing in `(next_ns, next_ew)` is just `p_ns * p_ew` — no
need to model a joint distribution explicitly, a simple nested loop over
3x3=9 combinations suffices. Compare this to
`q_learning_agent.QLearningAgent.update()`: the STRUCTURE of the equation
is identical (`reward + gamma * something`), but here "something" is an
exact expectation over a known distribution, while in Q-learning it's
`max_a' Q(s',a')` from a single SAMPLED next state. That's the entire
planning-vs-learning distinction, visible directly in code.

### `solve()`

Standard value-iteration sweep: repeatedly recompute every state's value
as the best achievable `_q_value` over both actions, tracking the largest
change (`max_delta`) seen in that sweep. Stops early once `max_delta` drops
below `theta` (converged) or after `max_iterations` sweeps (safety cap).
A final pass extracts the greedy policy (`argmax` instead of `max`) once
the values have stabilized — computing the policy on possibly-still-moving
values during the main loop would be wasted work, since only the FINAL
values matter for the FINAL policy.

### `choose_action(env)`

Unlike the other two controllers' `choose_action(state)`, this one takes
the whole `env` object and reads `env.state` directly — the raw car
counts and the EXACT time counter, not the discretized tuple `env.step()`
would return to an agent. This is deliberate and matches
`ARCHITECTURE.md`: value iteration represents "planning with a known
model," which includes knowing the true state, not just an agent's-eye
bucketed view of it.

### A genuine, observed limitation (not a bug) — read this before treating the planner's policy as "the correct answer"

Diagnostic check on state `('LOW', 'HIGH', 'NS', 10)` (NS clear, EW badly
backed up, NS has had green for a while):
```
KEEP:   Q = -263.71   (ew stays classified HIGH, probability 1.0)
SWITCH: Q = -276.81   (ew stays classified HIGH, probability 1.0)
```
Both actions leave EW in the HIGH bucket with 100% probability, because
departing 3 cars from the representative HIGH count (12 -> 9) still
satisfies "9 or more." The model's FUTURE-value term therefore sees zero
benefit from switching — it only "sees" the -5 switch penalty and NS's
own queue starting to build (since NS becomes red). The actual reward term
DOES register the improvement (12.6 expected vs 9.6 expected — see
`DECISIONS.md`), but that's only a one-step effect; the future-value term,
which dominates over a long horizon, cannot tell the difference between
"just barely HIGH" and "catastrophically HIGH." The result: the planner's
policy sometimes refuses to relieve a backed-up road, and can flicker
(SWITCH immediately followed by SWITCH back) when run against the real
environment — verified directly in this file's own `__main__` trace.

This was confirmed to be a property of the APPROXIMATE MODEL, not a code
bug, by manually recomputing `_q_value` for both actions at that state and
checking the arithmetic matches the printed numbers exactly (see
`DECISIONS.md` and `FEATURES.md` for the full investigation). It is kept,
not patched around, because it is one of the strongest, most concrete,
most honest "design tradeoff" talking points in the whole project: a
hand-built model that's slightly wrong can produce a confidently-computed
"optimal" policy that is actually worse than a model-free method that
never needed a model to be right in the first place.

### `if __name__ == "__main__":` block

Solves the planner for the same rates used in `traffic_env.py`'s own
trace, prints convergence info and a few illustrative state/policy/value
triples, then runs the resulting policy against a REAL environment
(same seed) and prints a trace in the same format as the other files'
smoke tests — this is the trace that surfaced the HIGH-bucket limitation
above.

---

## `evaluate.py`

### Module purpose

Runs all three controllers — the trained Q-learning agent, the
fixed-timer baseline, and the value-iteration planner — against the SAME
seeded environment, then prints a comparison table and saves three plots.
This is the file that actually answers "did any of this work?" — every
other file is a component; this is the experiment.

### Why `TRAIN_SEED` and `EVAL_SEED` are different constants

```python
TRAIN_SEED = 7
EVAL_SEED = 123
```

If the agent were evaluated on the exact same seed it trained on, a good
score wouldn't prove it learned a general policy — it could just mean the
agent memorized the one specific sequence of arrivals it saw 500 times
during training. Using a different seed for evaluation means the agent is
being tested on a traffic pattern it has never seen before (same
statistics — same `rate_ns`/`rate_ew` — different actual random draws).

### Why all three controllers get the SAME `EVAL_SEED`

```python
eval_env_q = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, seed=EVAL_SEED)
eval_env_base = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, seed=EVAL_SEED)
eval_env_vi = TrafficEnv(rate_ns=RATE_NS, rate_ew=RATE_EW, max_steps=MAX_STEPS, seed=EVAL_SEED)
```

Three separate `TrafficEnv` objects, not one shared object — each
controller needs to run its own full episode without interfering with the
others. But because they all get the identical seed, and because
`TrafficEnv`'s arrivals are positional (the arrival draw at step *i*
depends only on *i*, never on what actions were taken before it — this was
directly verified: `numpy.random.default_rng(123)` called twice from
scratch produced the identical sequence `[1, 0, 0, 0, 2]` both times), all
three controllers face EXACTLY the same sequence of arriving cars. Any
difference in outcome (avg wait, max queue, switches) is then attributable
purely to the controller's decisions, not to one of them getting luckier
traffic.

### `run_episode(env, choose_action_fn)`

```python
def run_episode(env, choose_action_fn):
    state = env.reset()
    waiting_history = []
    switch_count = 0
    max_queue = 0
    done = False
    while not done:
        action = choose_action_fn(state)
        state, _reward, done = env.step(action)
        s = env.state
        waiting_history.append(s.cars_waiting_ns + s.cars_waiting_ew)
        max_queue = max(max_queue, s.cars_waiting_ns, s.cars_waiting_ew)
        if action == "SWITCH":
            switch_count += 1
    return waiting_history, switch_count, max_queue
```

One shared runner for all three controllers, because they all speak the
same `reset()`/`step()` interface. `choose_action_fn` is just any callable
that takes a state and returns an action string — this is what lets the
exact same loop drive the Q-learning agent, the baseline, and the
value-iteration planner without three separate copies of this logic:

- Q-learning: passed as `agent.best_action` directly — pure greedy
  (no exploration) at evaluation time, since epsilon-greedy randomness
  is only needed during training.
- Baseline: passed as `baseline.choose_action` directly (its call
  signature was deliberately built to match `agent.choose_action`'s
  shape back in `baseline_controller.py`, for exactly this reason).
- Value iteration: `choose_action(env)` needs the actual `env` object
  (raw counts), not just a discretized `state`, so it's wrapped in a
  one-line lambda: `lambda state: planner.choose_action(eval_env_vi)`.

### The three plots

1. **`waiting_comparison.png`** — one line per controller, x = step,
   y = total cars waiting (NS + EW) at that step, all three drawn on
   the same axes from the same seeded episode. This is the single most
   direct visual answer to "which controller keeps queues shorter?"
2. **`training_reward.png`** — the Q-learning agent's per-episode total
   reward across all 500 training episodes, plotted both raw (faint) and
   as a 20-episode rolling average (bold). The raw line is noisy because
   arrivals are random every episode; the rolling average is what
   actually shows the upward (less negative) learning trend clearly.
3. **`rushhour_switching.png`** — trains a SEPARATE Q-learning agent
   under `rush_hour=True` (a fresh agent, not the main one, since it
   needs to have actually experienced the changing-rate schedule during
   training to plausibly adapt to it), then plots cumulative switch
   count over time for that agent vs. the baseline, with the three
   traffic-phase boundaries marked as dashed vertical lines. The
   baseline's line is a perfectly even staircase by construction (it
   switches every `switch_every` steps no matter what); if the
   Q-learning line's slope visibly changes between phases, that's
   evidence the agent is reacting to the changing NS/EW balance rather
   than switching on a fixed clock. Value iteration is deliberately left
   out of this plot — it doesn't support `rush_hour` at all (see its own
   module docstring: a time-varying arrival rate would make its
   transition model non-stationary, which is out of scope for this
   project).

### Real run output

```
Controller          Avg wait   Max queue    Switches
----------------------------------------------------
Q-learning              1.96           5          18
Baseline                3.97          10          10
Value iteration         2.03           5          12
```

Q-learning roughly halves both the average wait and the worst-case queue
compared to the naive fixed-timer baseline — the central result the whole
project was built to produce. Value iteration performs almost as well as
Q-learning (2.03 vs 1.96), which makes sense: it's an (approximately)
optimal planner for a model that's very close to the truth at the traffic
rates used here, so the bucket-approximation limitation documented for
`value_iteration.py` doesn't bite hard in this particular scenario (it
shows up in specific backed-up states, not on average across a whole
episode — see `DECISIONS.md`).

---

## `visualize.py` (optional)

### Module purpose

A "watch it work" demo, separate from `evaluate.py`'s measurement job.
Trains a Q-learning agent, then prints one text frame per step showing
each road's queue as an ASCII bar (`#` per car) and which road is green.

### Why `DEMO_SEED` differs from `TRAIN_SEED`

Same reasoning as `evaluate.py`: a demo run on the exact training seed
would just replay a memorized trajectory, not show the agent handling a
traffic pattern it has never seen. `DEMO_SEED = 999` is fresh.

### `render_frame()`

```python
def render_frame(step, ns_count, ew_count, light, action, reward):
    ns_marker = "GREEN" if light == "NS" else "red"
    ew_marker = "GREEN" if light == "EW" else "red"
    print(f"Step {step:>3} | action={action:<6} reward={reward:>6.1f}")
    print(f"  NS [{ns_marker:>5}] {'#' * ns_count} ({ns_count})")
    print(f"  EW [{ew_marker:>5}] {'#' * ew_count} ({ew_count})")
    print()
```

One `#` character per waiting car — deliberately the simplest possible
visualization, no plotting library needed. `light` decides which road's
marker says `GREEN` vs `red`; only one is ever green at a time (mirrors
`TrafficEnv`'s own rule).

### What the smoke test showed

Running `python visualize.py`: NS starts empty and green. EW's queue
visibly grows (`#`, `##`, `###`) across several KEEP steps while NS holds
green. At step 8, once EW has backed up to 3 cars, the agent SWITCHes —
NS becomes red, EW becomes green, and EW's bar shrinks the next step as it
drains. This is the discretized-state Q-learning policy visibly reacting
to a real, growing queue — the exact behavior the whole project set out to
produce, made directly watchable instead of only measurable.

---

## Other files (not yet built)

None remaining — all files from the original spec are built.
