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

## Other files (not yet built)

This section will be filled in as each file is written:
`baseline_controller.py`, `value_iteration.py`, `evaluate.py`,
`visualize.py`.
