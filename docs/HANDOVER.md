# Handover

Living record of where the project stands right now. Read this first at the
start of every session. Updated incrementally — not a full dump, just current
state.

**Model that did the most recent work:** Claude Sonnet 5, session
`session_01XSe2ZGVYoQHfLBxGy1JwPa`, 2026-09-05.

## What's done

- `traffic_env.py` — full MDP environment. Manually smoke-tested via its
  `__main__` block (`python traffic_env.py`): arrivals accumulate correctly,
  green road drains capped at `DEPARTURE_RATE`, SWITCH resets the timer and
  applies the penalty, reward tracks total queue length. Confirmed sensible
  by the user before proceeding.
- `docs/ARCHITECTURE.md`, `docs/CONSTRAINTS.md`, `docs/DECISIONS.md`,
  `docs/FLOW.md` — guardrail + rationale docs, current as of `traffic_env.py`.

## What's in progress

- Nothing actively in progress — paused here per the user's explicit
  instruction to confirm the environment before building the agent.

## What's next

1. `q_learning_agent.py` — tabular Q-learning agent (epsilon-greedy,
   Bellman update, `train()` loop).
2. `baseline_controller.py` — fixed-timer controller, same env interface.
3. `value_iteration.py` — stretch goal, exact planner using the env's known
   transition model.
4. `evaluate.py` — run all controllers on the same seeded scenario, plot +
   print comparison.
5. `visualize.py` — optional simple animation/text demo.
6. `README.md` + `requirements.txt`.
7. `docs/TEST_CHECKLIST.md`, `docs/ROLLBACK.md`, `docs/CODE_EXPLAINED.md`,
   `docs/FEATURES.md` — created alongside the code above (some already
   exist as skeletons; see `ARCHITECTURE.md` status checklist).
8. After all code exists: interview-prep walkthrough of the Q-update
   function, 5 likely interview Q&As, and a list of genuine design
   tradeoffs to highlight — all requested explicitly in `prompt.txt`.

## What to watch out for

- No git repo initialized in this project yet — nothing to roll back to via
  `git`. See `docs/ROLLBACK.md`.
- `TrafficEnv.step()`'s three-phase order (arrivals -> action -> departures)
  is load-bearing for correctness; do not reorder without updating
  `docs/FLOW.md` and re-running the smoke test.
