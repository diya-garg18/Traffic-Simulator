# Decisions

Log of meaningful design decisions and the reasoning behind them. Code shows
what changed; this shows why. Newest entries at the top.

---

## 2026-09-05 — Value iteration plans over bucketed states using one representative count per bucket

**Decision:** `value_iteration.py` solves an APPROXIMATE finite MDP: states
use the same LOW/MEDIUM/HIGH car buckets as Q-learning, but transition
probabilities and rewards are computed by pretending each bucket's true
count is always one fixed "representative" value (LOW=2, MEDIUM=6,
HIGH=12), not the real (unbounded, exact) count.

**Why:** True dynamic programming needs a FINITE state space to sweep
over. TrafficEnv's real state space is infinite (queues are unbounded), so
exact DP over it is impossible. Bucketing collapses it to 378 states
(3 x 3 x 2 x 21), small enough to solve in under 200 iterations.

**Tradeoff accepted — and this is a REAL, OBSERVED one, not theoretical:**
Collapsing an entire open-ended bucket (HIGH = "9 or more") to one anchor
value (12) hides how much a partial departure actually helps. Diagnostic
check on state `('LOW', 'HIGH', 'NS', 10)`:
```
KEEP:   ns_exp=0.5,  ew_exp=12.6, ew stays HIGH with probability 1.0, Q=-263.71
SWITCH: ns_exp=3.2,  ew_exp=9.6,  ew stays HIGH with probability 1.0, Q=-276.81
```
Departing 3 cars from the HIGH anchor (12 -> 9) is still classified HIGH
(the bucket is "9+"), so the FUTURE-value term sees zero benefit from
switching — even though the immediate reward term correctly registers the
queue as smaller (9 vs 12). Combined with the flat SWITCH_PENALTY (5) and
the cost of neglecting NS while EW is served, the planner ends up
preferring to just KEEP holding NS green even though EW is badly backed
up, and its resulting policy visibly flickers (back-to-back SWITCH/SWITCH)
when run against the real environment — see `docs/FEATURES.md` for the
concrete trace. This is exactly the "planning is only as good as the
model" lesson: a hand-built approximate model can produce a policy that is
optimal FOR THAT MODEL but visibly suboptimal against the real dynamics.
Q-learning has no equivalent failure mode here, since it learns directly
from real transitions rather than a hand-built approximation of them —
this is the single best concrete talking point for "why model-free
learning can beat planning when the model is imperfect."

**What would fix it (not implemented, out of scope):** finer buckets
specifically at the high end (e.g. splitting HIGH into 9-14 / 15+), or
tracking a full distribution over counts within each bucket instead of one
point estimate. Both add real complexity for a coursework project whose
goal is understanding the core algorithms, not building a production-grade
planner — left as a known, documented limitation instead.

---

## 2026-09-05 — Baseline switches every 10 steps by default, tracks its own clock

**Decision:** `FixedTimerController` counts steps internally
(`self.steps_since_switch`), rather than reading `time_since_last_switch`
out of the environment's discretized state.

**Why:** A real fixed-timer light has its own physical clock; it does not
sense the environment at all. Reading the env's internal timer would be
cheating relative to what this baseline is supposed to represent, and would
also break if evaluate.py ever ran the baseline against a differently
configured env. Default `switch_every=10` is a reasonable "no worse than
one long departure phase" choice, deliberately not tuned to be optimal —
the baseline is supposed to be dumb.

**Tradeoff accepted:** `choose_action(state)` still takes `state` as a
parameter even though it's ignored, purely so its call signature matches
`QLearningAgent.choose_action` and both can be swapped interchangeably in
evaluate.py's loop. This is a very small, deliberate abstraction leak in
favor of interface consistency.

---

## 2026-09-05 — Q-learning hyperparameters: alpha=0.1, gamma=0.95, epsilon 1.0->0.05 linear

**Decision:** `q_learning_agent.py` defaults to learning rate 0.1, discount
factor 0.95, epsilon decaying linearly from 1.0 to 0.05 over 500 episodes.

**Why:** Alpha 0.1 is small enough to stay stable under Poisson-noisy
transitions but large enough to learn in a few hundred episodes. Gamma 0.95
was chosen because traffic control is a long-run queue-management problem —
a myopic agent (low gamma) would never learn "hold the light a bit longer to
fully drain a queue" tradeoffs. Epsilon starts at 1.0 (pure exploration on an
empty table) and decays to a nonzero floor (0.05), not to 0, because arrivals
never stop being random — the agent should never fully stop sampling.

**Tradeoff accepted:** All three are hand-picked defaults, not tuned via a
sweep. Smoke test (`python q_learning_agent.py`) confirms they're good enough
to show learning (reward -423 -> -294 avg over 500 episodes on a fixed seed),
but a hyperparameter sweep is explicitly out of scope for this project's
goals (understanding > optimality).

---

## 2026-09-05 — Q-table uses `defaultdict(float)`, zero-initialization

**Decision:** Unvisited (state, action) pairs implicitly read as 0.0 instead
of being pre-populated or raising an error.

**Why:** Simplest possible initialization; avoids having to enumerate all 54
states up front. Zero is also a defensible optimistic-ish starting point
here since most rewards are negative — an untried action defaulting to 0
looks *better* than a poorly-performing known action, which mildly encourages
trying new things even outside of epsilon-exploration.

**Tradeoff accepted:** True optimistic initialization (e.g. starting at 0
when true values are very negative) can bias early exploration in a
specific direction; not analyzed further here since epsilon-greedy already
handles exploration explicitly.

---

## 2026-09-05 — Action representation: plain strings, not an Enum

**Decision:** `ACTIONS = ("KEEP", "SWITCH")` as module-level string constants
in `traffic_env.py`, instead of an `Enum`/`IntEnum`.

**Why:** Only two actions, ever. An Enum adds a layer of indirection
(`Action.KEEP.value` vs `"KEEP"`) for zero real benefit here, and strings are
directly readable in printed traces and directly usable as Q-table dict-key
components. `CONSTRAINTS.md` explicitly rules out unneeded abstraction —
this is the concrete example of applying that rule.

**Tradeoff accepted:** No compile-time typo protection (`"SWTICH"` would
silently be treated as an invalid action string, raising a `ValueError` at
runtime instead of failing earlier). Acceptable because `step()` validates
the action and raises immediately.

---

## 2026-09-05 — Discretization thresholds: LOW 0-3 / MEDIUM 4-8 / HIGH 9+

**Decision:** Bucket each road's queue length into 3 categories at those
specific cut points.

**Why:** `DEPARTURE_RATE = 3`, i.e. a green light clears up to 3 cars per
step. LOW (0-3) roughly means "a green phase can fully clear this in one
step." HIGH (9+) means the queue is growing faster than 3 phases could
plausibly drain it. MEDIUM is the "in between, needs attention soon" zone.
The thresholds are not arbitrary — they're anchored to the one mechanic
(departure rate) that actually determines whether a queue is under control.

**Tradeoff accepted:** Any fixed threshold coarsens information — a queue of
4 and a queue of 8 are both "MEDIUM" even though they're very different in
severity. This is the necessary cost of keeping the state space small
enough for tabular Q-learning (see the docstring on `discretize_state()`
for the full state-space-size argument).

---

## 2026-09-05 — Reward = -(total waiting) - 5 if SWITCH

**Decision:** Reward per step is the negative sum of both queues, minus an
extra flat 5 whenever the action taken was SWITCH.

**Why:** Negative total queue length is a direct, easily-justified proxy for
minimizing average wait time — no need to track individual cars' wait
timers. The switch penalty exists because, without it, an agent could learn
to flip the light every single step purely to "time" departures on whichever
road happens to be momentarily green, which is not a policy a real
intersection could implement (lights need a minimum safe green duration).
The penalty makes switching a real cost the agent has to justify with
reduced future queueing, not something free.

**Tradeoff accepted:** The magnitude of the penalty (5) is a hyperparameter
chosen by inspection, not derived analytically — it needs to be large enough
to discourage flicker but small enough not to make the agent never switch.
If evaluation shows pathological behavior (e.g. the agent never switches,
or still flickers), this is the first constant to revisit.

---

## 2026-09-05 — Reward computed AFTER arrivals + action + departures

**Decision:** Within `step()`, reward is computed from the state as it
exists after all three sub-steps (arrivals, action application, departures)
have already happened for this tick — not from the state before the action.

**Why:** This makes reward a direct measurement of "how good is the state
the agent is now in," which is the standard MDP convention (reward attached
to the transition into the next state) and keeps the Bellman equation's
`r + γ·max Q(s', a')` interpretable without any lag/offset confusion.

---

## 2026-09-05 — time_since_last_switch capped at 20 before discretizing

**Decision:** The raw counter itself is clamped at `TIME_SINCE_SWITCH_CAP =
20`, not just the discretized bucket.

**Why:** Simpler than tracking an unbounded int and bucketing it — capping
early means the value is always small and directly inspectable while
debugging, with no behavioral difference (the LONG bucket already covers
"8 or more," so nothing distinguishes step 21 from step 10000 anyway).

---

## 2026-09-05 — Rush-hour schedule: 3 equal phases via `current_step // phase_length`

**Decision:** Rush hour splits an episode into thirds (NS-heavy, balanced,
EW-heavy) using integer division on the step counter, rather than a
time-of-day clock or a config file of phases.

**Why:** Simplest possible implementation that still produces a visibly
adapting environment for the evaluation plots, and it's trivially
explainable in an interview ("the episode is divided into three regimes").
A more elaborate schedule (e.g. sinusoidal rates) would look more realistic
but adds nothing pedagogically and makes the "did the agent adapt" plot
harder to read.
