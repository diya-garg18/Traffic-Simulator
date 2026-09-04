# Handover

Living record of where the project stands right now. Read this first at the
start of every session. Updated incrementally — not a full dump, just current
state.

**Model that did the most recent work:** Claude Sonnet 5, session
`session_01XSe2ZGVYoQHfLBxGy1JwPa`, 2026-09-05.

## What's done

- `traffic_env.py` — full MDP environment. Manually smoke-tested, user
  reviewed and confirmed the trace looked sensible before agent work began.
- `q_learning_agent.py` — tabular Q-learning (Q-table, epsilon-greedy,
  Bellman `update()`, `train()`). Smoke-tested: reward improved from -423.0
  to -294.0 avg (first 50 vs last 50 of 500 episodes).
- `baseline_controller.py` — fixed-timer controller. Smoke-tested: switches
  exactly every N steps regardless of queue state.
- `value_iteration.py` (stretch goal) — exact DP planner over an
  approximate bucketed MDP. Converges in 197 iterations. Diagnosed and
  documented a real, non-bug limitation: its bucket-approximation model
  can make it refuse to SWITCH out of a backed-up HIGH queue and can
  flicker — see `DECISIONS.md`, this is the flagship interview talking
  point.
- `evaluate.py` — trains the Q-learning agent, solves the planner, runs all
  three controllers (+ raw baseline) on the SAME seeded scenario
  (`EVAL_SEED=123`, different from `TRAIN_SEED=7` so it tests
  generalization), prints a summary table, and saves 3 plots to `plots/`.
  Real result: Q-learning avg wait 1.96 vs baseline's 3.97 — roughly half.
  Value iteration close behind at 2.03.
- `requirements.txt` — `numpy`, `matplotlib`, `scipy` (scipy is a hard
  dependency of `value_iteration.py`).
- Project is a git repo, pushed to
  https://github.com/diya-garg18/Traffic-Simulator (main branch). `prompt.txt`
  is intentionally NOT tracked (deleted from both local and GitHub, per
  explicit user request — no backup of the original spec text exists
  anywhere except earlier conversation history).
- All docs (`ARCHITECTURE.md`, `CONSTRAINTS.md`, `DECISIONS.md`, `FLOW.md`,
  `FEATURES.md`, `TEST_CHECKLIST.md`, `CODE_EXPLAINED.md`, this file) are
  current as of `evaluate.py`. `ROLLBACK.md` still needs the "real git
  rollback is now possible" update noted below.

## What's in progress

- Nothing actively in progress.

## What's next

1. `visualize.py` — optional simple animation/text demo (last remaining
   code deliverable; explicitly marked optional in the original spec).
2. `README.md` — MDP formulation with justification for state/action/
   reward/discount choices, how to run training/evaluation, expected
   results summary. Not yet started.
3. `docs/ROLLBACK.md` — update to reflect that real git-based rollback
   (`git revert`/`git reset`) is now possible, since a real repo with
   real history exists.
4. After all code exists: interview-prep walkthrough of the Bellman
   `update()` function line by line, 5 likely interview Q&As, and a list
   of genuine design tradeoffs to highlight (the `value_iteration.py`
   HIGH-bucket limitation is the strongest candidate, already fully
   diagnosed).
5. Keep pushing to GitHub after each meaningful chunk, with simple commit
   messages (explicit standing user instruction).

## What to watch out for

- `TrafficEnv.step()`'s three-phase order (arrivals -> action -> departures)
  is load-bearing for correctness; do not reorder without updating
  `docs/FLOW.md` and re-running the smoke test.
- `evaluate.py` retrains a SEPARATE Q-learning agent specifically for the
  rush-hour plot (`plot_rush_hour_switching`) rather than reusing the main
  agent — the main agent was trained on fixed rates and never experienced
  the rush-hour schedule, so reusing it wouldn't fairly test adaptation.
- A handful of harmless 0-byte junk files sit in the project root (`,+`,
  `0.05`, `5}`, `7}`, `agent`, `departures),`, `optimality).,+,+---,+,+##`)
  — origin unknown (possibly a tooling artifact from an earlier session),
  confirmed empty, not part of the project, left untouched pending a user
  decision to delete them.
- Git repo now exists — real rollback via `git revert`/`git reset` is
  possible going forward; `docs/ROLLBACK.md` still describes the
  pre-git-repo state and needs updating.
