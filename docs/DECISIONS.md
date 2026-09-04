# Decisions

Log of meaningful design decisions and the reasoning behind them. Code shows
what changed; this shows why. Newest entries at the top.

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
