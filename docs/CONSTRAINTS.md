# Constraints

What any AI session (or contributor) must NOT do in this project, and what
it must always do. "Allow" on a tool call does not mean "allow anything" —
this is the scope.

## Hard constraints (never do these)

- **No RL/environment libraries.** No Gymnasium/OpenAI Gym, no
  Stable-Baselines3, no RLlib, etc. The whole point is hand-written MDP +
  hand-written Q-learning the user can defend on a whiteboard.
- **No new dependencies without asking first.** `requirements.txt` is
  meant to stay minimal: numpy, matplotlib, scipy. Anything else — ask.
- **No unnecessary abstraction.** No factory patterns, no plugin systems,
  no config-file frameworks, no dependency injection. If a plain function
  or a small dataclass does the job, use that. This is a solo undergrad
  project meant to be held entirely in one person's head.
- **No silent interface changes.** `TrafficEnv.step()`/`reset()` are the
  contract every controller (`q_learning_agent.py`, `baseline_controller.py`,
  `value_iteration.py`) depends on. If that signature changes, update
  `ARCHITECTURE.md` and `FLOW.md` in the same change, and check all three
  callers.
- **No claiming something works without running it.** Every new file gets
  smoke-tested (a `__main__` block or a quick manual run) before being
  marked done in `ARCHITECTURE.md`.

## Standing requirements

- Every non-obvious design choice (discretization buckets, reward shaping,
  hyperparameter defaults, episode length, rush-hour schedule) gets a `WHY`
  comment in the code AND an entry in `DECISIONS.md`.
- Type hints on all function signatures.
- Fixed random seed available everywhere evaluation/comparison happens, so
  runs are reproducible.
- `docs/` files get updated in the same session as the code change they
  describe — not batched up for "later."
