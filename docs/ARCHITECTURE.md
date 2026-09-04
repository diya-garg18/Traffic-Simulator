# Architecture

High-level map of the project. Not implementation detail — see `CODE_EXPLAINED.md`
for that. This file exists so every session starts knowing the shape of the
system instead of re-deriving it from the files.

## Modules and how they connect

```
                     +-------------------+
                     |   traffic_env.py  |   <- the MDP (state, actions, reward,
                     |    TrafficEnv     |      transition dynamics). Owns the
                     +-------------------+      ONE source of truth for how the
                        ^      ^      ^          intersection behaves.
                        |      |      |
        +---------------+      |      +----------------+
        |                      |                       |
+-------------------+  +----------------------+  +--------------------+
| q_learning_agent.py| | baseline_controller.py| | value_iteration.py |
|   QLearningAgent    | |  FixedTimerController | |  (uses env's known |
|  (learns a policy   | |  (no learning, just   | |   transition model |
|   via trial+error)  | |   switches every N    | |   to solve exactly)|
+-------------------+  |   steps)              | +--------------------+
                        +----------------------+
                        ^                       ^
                        |                       |
                        +----------+------------+
                                   |
                          +------------------+
                          |   evaluate.py    |   <- runs all controllers on the
                          |                  |      SAME seeded scenario, plots
                          +------------------+      + prints comparison tables
                                   |
                          +------------------+
                          |  visualize.py    |   <- optional live demo animation
                          +------------------+
```

## Data flow

Every controller (Q-learning agent, baseline, value-iteration policy) talks
to `TrafficEnv` through exactly two calls:

- `state = env.reset()` — start an episode, get the discretized starting state.
- `next_state, reward, done = env.step(action)` — advance one tick.

`action` is always one of the two strings in `traffic_env.ACTIONS`
(`"KEEP"` / `"SWITCH"`). `state` is always the 4-tuple returned by
`discretize_state()`: `(ns_bucket, ew_bucket, current_light, time_bucket)`.

This shared interface is what makes the baseline, the learned policy, and
the value-iteration policy directly comparable in `evaluate.py` — they are
three different ways of choosing `action` given `state`, running against the
identical environment.

## Why this shape

- `traffic_env.py` has zero knowledge of any agent. Agents depend on the
  env; the env never depends on an agent. This is what lets `evaluate.py`
  swap controllers in and out against the same simulator.
- `value_iteration.py` is the only module allowed to reach into the env's
  *raw* transition dynamics (arrival rates, departure rate) to build an
  exact model — because it represents "planning with a known model," which
  only makes sense if it's actually allowed to know the model. Q-learning
  and the baseline must NOT do this; they only see `(state, reward, done)`
  like a real controller would.

## Status

- [x] `traffic_env.py` — built, manually smoke-tested.
- [x] `q_learning_agent.py` — built, smoke-tested (reward improves over training).
- [x] `baseline_controller.py` — built, smoke-tested (switches exactly every N steps).
- [x] `value_iteration.py` (stretch goal) — built, solved, smoke-tested. Has
  a known/documented model-approximation limitation, see DECISIONS.md.
- [x] `evaluate.py` — built, runs all three controllers on the same seeded
  scenario, prints summary table, saves 3 plots to `plots/`.
- [x] `visualize.py` (optional) — built, smoke-tested. Text-frame demo of
  the Q-learning agent driving the intersection.
