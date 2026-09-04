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
- `q_learning_agent.py` — tabular Q-learning agent (Q-table, epsilon-greedy,
  Bellman `update()`, `train()`). Smoke-tested: reward improved from -423.0
  to -294.0 avg (first 50 vs last 50 of 500 episodes), epsilon decay verified
  monotonic (1.000 -> 0.525 -> 0.050).
- Project is now a git repo, pushed to
  https://github.com/diya-garg18/Traffic-Simulator (main branch). `prompt.txt`
  is intentionally NOT tracked in git (kept local-only, per user request) —
  see `.gitignore`.
- `docs/ARCHITECTURE.md`, `docs/CONSTRAINTS.md`, `docs/DECISIONS.md`,
  `docs/FLOW.md`, `docs/FEATURES.md`, `docs/HANDOVER.md` — current as of
  `q_learning_agent.py`.

## What's in progress

- Nothing actively in progress. Continuing straight to
  `baseline_controller.py` next (env->agent confirmation gate already passed;
  no further per-file pause was requested by the user).

## What's next

1. `baseline_controller.py` — fixed-timer controller, same env interface.
2. `value_iteration.py` — stretch goal, exact planner using the env's known
   transition model.
3. `evaluate.py` — run all controllers on the same seeded scenario, plot +
   print comparison.
4. `visualize.py` — optional simple animation/text demo.
5. `README.md` + `requirements.txt`.
6. `docs/TEST_CHECKLIST.md`, `docs/ROLLBACK.md`, `docs/CODE_EXPLAINED.md` —
   keep updating incrementally per file, as already done for
   `q_learning_agent.py`.
7. After all code exists: interview-prep walkthrough of the Q-update
   function, 5 likely interview Q&As, and a list of genuine design
   tradeoffs to highlight — all requested explicitly in `prompt.txt`.
8. Keep pushing to GitHub after each meaningful chunk, with simple commit
   messages (explicit user instruction).

## What to watch out for

- `TrafficEnv.step()`'s three-phase order (arrivals -> action -> departures)
  is load-bearing for correctness; do not reorder without updating
  `docs/FLOW.md` and re-running the smoke test.
- Git repo now exists — real rollback via `git revert`/`git reset` is
  possible going forward (update `docs/ROLLBACK.md` to reflect this next
  time it's touched).
