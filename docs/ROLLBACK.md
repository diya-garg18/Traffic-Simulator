# Rollback

Short plan for undoing a change if it breaks something.

## Current state: no git repo

This project is **not yet a git repository**. That means there is currently
no commit history to revert to — the only rollback mechanism right now is
manual (restoring a file you saved a copy of, or re-writing it from a
previous message in this conversation).

**Recommendation:** once the environment + agent are both working, run
`git init` and commit, so future risky changes (hyperparameter sweeps,
refactors) have a real revert point. This has NOT been done automatically —
it's a decision for you to make, not something to do silently.

## Per-file rollback notes

- **`traffic_env.py`**: If a change to `step()`'s ordering (arrivals ->
  action -> departures) breaks behavior, the fix is to restore that exact
  order — see `docs/FLOW.md` for what it should be. Re-run
  `python traffic_env.py` and compare against the trace pattern described
  in `docs/TEST_CHECKLIST.md` after restoring.
- **`q_learning_agent.py`** (once it exists): If a hyperparameter change
  (alpha/gamma/epsilon schedule) causes training reward to stop improving,
  revert to the last known-good values recorded in `docs/DECISIONS.md`
  before debugging further — isolates "bad hyperparameters" from "bad code"
  as the cause.
- **Any file**: if this session or a future one makes a change you don't
  want, the fastest recovery is to ask the AI to show the diff against what
  was here before, rather than accepting a rewritten file blind (see the
  field guide's "read the diff every time" habit).

## What to re-check after any rollback

1. Run every checked-off item in `docs/TEST_CHECKLIST.md` for the file(s)
   that were rolled back.
2. Update `docs/HANDOVER.md` to reflect the reverted state.
