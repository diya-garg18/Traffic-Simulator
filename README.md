# Adaptive Traffic Light Controller using Q-Learning

A single 4-way intersection (North-South vs East-West traffic), controlled
three different ways, all built from scratch in plain Python/NumPy — no
Gymnasium, no Stable-Baselines3, no RL libraries at all. The goal of this
project is to fully understand and be able to defend every line, not just
get a working demo.

## The MDP formulation

An MDP (Markov Decision Process) is defined by: a state space, an action
space, a transition model, and a reward function. Here's this project's
choices, and why.

### State

The raw state is `(cars_waiting_ns, cars_waiting_ew, current_light,
time_since_last_switch)` — two queue lengths, which road is green, and how
long it's been since the light last changed.

For tabular Q-learning this state is **discretized** into:
`(ns_bucket, ew_bucket, current_light, time_bucket)`, where each queue is
bucketed as `LOW` (0-3 cars), `MEDIUM` (4-8), or `HIGH` (9+), and time since
switch is bucketed as `SHORT` (0-2 steps), `MEDIUM` (3-7), or `LONG` (8+).

**Why discretize at all?** Queues are unbounded in principle — a tabular
Q-table needs a *finite* number of states to have any hope of visiting each
one enough times to learn a good estimate. Bucketing collapses infinitely
many raw states into 3 × 3 × 2 × 3 = **54 discretized states**, small
enough to fully explore.

**Why these specific thresholds?** They're anchored to `DEPARTURE_RATE = 3`
(how many cars a green light clears per step) — LOW roughly means "a single
green phase can clear this," HIGH means "growing faster than any reasonable
number of phases can drain it." Not arbitrary numbers.

### Actions

Exactly two: `KEEP` (leave the light alone) or `SWITCH` (flip it). Plain
strings, not an Enum — there are only ever two of them, and strings are
directly usable as Q-table dictionary keys and directly readable in
printed traces.

### Transition dynamics

Each step, in this exact order:
1. New cars arrive on both roads, independently, drawn from a **Poisson
   distribution** with rate `rate_ns` / `rate_ew` (expected new cars per
   step) — Poisson is the standard model for "random independent arrivals
   over time" (e.g. cars reaching an intersection).
2. The chosen action is applied (switch the light, or age the "time since
   switch" counter).
3. The road that is now green discharges up to `DEPARTURE_RATE = 3` cars
   (capped at how many are actually waiting).

Order matters: reward is computed from the state *after* all three steps,
which is standard MDP convention (reward attaches to the transition into
the next state).

### Reward

```
reward = -(cars_waiting_ns + cars_waiting_ew)
reward -= 5  if the action was SWITCH
```

Negative total queue length is a direct, simple proxy for "minimize average
wait time," with no need to track individual cars. The flat -5 SWITCH
penalty exists so flipping the light is never free — without it, an agent
could learn to flicker the light every single step purely to "time"
departures, which isn't something a real intersection could safely do
(lights need a minimum green duration).

### Discount factor

`gamma = 0.95` for both Q-learning and value iteration. Close to 1 because
traffic control is a long-run queue-management problem — a myopic agent
(low gamma) would never learn "hold this light a bit longer to fully drain
a queue" tradeoffs. Not exactly 1, so long-episode value estimates stay
finite and stable.

### Rush hour (optional mode)

When `rush_hour=True`, an episode is split into three equal phases with
different arrival rates: NS-heavy → balanced → EW-heavy. Used to test
whether a controller actually *adapts* to changing traffic, not just to one
fixed average rate.

## The three controllers

| Controller | File | Idea |
|---|---|---|
| Fixed-timer baseline | `baseline_controller.py` | Switches every N steps, ignores traffic entirely — how many real, non-adaptive lights already work. |
| Q-learning agent | `q_learning_agent.py` | Learns a policy purely from trial and error — never told the arrival rates, only observes `(state, reward, next_state)`. |
| Value iteration planner | `value_iteration.py` (stretch goal) | Given the *known* arrival rates, computes the mathematically optimal policy directly via dynamic programming — no trial and error. |

The baseline and Q-learning agent never see anything but
`env.reset()`/`env.step()`. Value iteration is the one module allowed to
read the environment's true transition model, since it represents
"planning with a known model" — see `docs/ARCHITECTURE.md`.

## How to run

```bash
pip install -r requirements.txt

python traffic_env.py           # sanity-check the environment (10-step trace)
python q_learning_agent.py      # train + smoke-test the agent alone
python baseline_controller.py   # smoke-test the fixed-timer baseline
python value_iteration.py       # solve the planner + smoke-test it
python evaluate.py              # the real comparison: trains everything,
                                 # prints a summary table, saves plots/
```

`evaluate.py` is the main entry point for results — it trains the
Q-learning agent (500 episodes), solves the value-iteration planner, then
runs all three controllers against the *same* seeded traffic scenario so
the comparison is fair (identical arrivals, only the decisions differ).

## Expected results

Running `python evaluate.py` with the defaults (`rate_ns=0.8, rate_ew=0.5,
max_steps=100`) produces:

```
Controller          Avg wait   Max queue    Switches
----------------------------------------------------
Q-learning              1.96           5          18
Baseline                3.97          10          10
Value iteration         2.03           5          12
```

Q-learning roughly **halves** both the average wait and the worst-case
queue length compared to the naive fixed-timer baseline. Value iteration
performs almost as well (it's near-optimal for its model), occasionally
lagging Q-learning in specific backed-up states due to a known,
deliberately-documented model approximation (see `docs/DECISIONS.md`).

Three plots are saved to `plots/`:
- `waiting_comparison.png` — cars waiting over time, all 3 controllers.
- `training_reward.png` — Q-learning's reward curve across 500 training
  episodes (proof it's actually learning, not just running).
- `rushhour_switching.png` — Q-learning's switching behavior adapting
  across NS-heavy / balanced / EW-heavy traffic phases, vs the baseline's
  fixed-clock switching.

## Project docs

Deeper documentation lives in `docs/`:
- `ARCHITECTURE.md` — module map and data flow.
- `DECISIONS.md` — every non-obvious design choice and why, including the
  value-iteration model limitation.
- `CODE_EXPLAINED.md` — line-by-line explanation of every file.
- `FEATURES.md`, `TEST_CHECKLIST.md`, `HANDOVER.md`, `FLOW.md`,
  `CONSTRAINTS.md`, `ROLLBACK.md` — project process/tracking docs.
