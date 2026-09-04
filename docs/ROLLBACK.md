# Rollback

Short plan for undoing a change if it breaks something.

## Current state: real git repo, pushed to GitHub

This project IS now a git repository, with history pushed to
https://github.com/diya-garg18/Traffic-Simulator (`main` branch). Real
rollback is available:

- **Undo the last commit (not yet pushed further):**
  `git revert <commit-hash>` — creates a new commit that undoes the
  changes, preserving history (safer than `reset` since it doesn't rewrite
  anything already pushed).
- **See what changed in any past commit:** `git show <commit-hash>` or
  `git log -p -- <file>` for one file's full history.
- **Restore one file to how it looked in an earlier commit:**
  `git checkout <commit-hash> -- <file>` (careful: this discards
  uncommitted changes to that file without asking twice — check
  `git status` first).

`prompt.txt` is a deliberate exception: it was intentionally removed from
both git history's tracked state and the working directory, per explicit
user request — it should NOT be "rolled back" or restored.

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
