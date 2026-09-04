# Test Checklist

Concrete commands and expected outputs — proof, not vibes. Run these before
marking any change "done."

## `traffic_env.py`

- [x] `python traffic_env.py`
      Expected: a 10-row trace table. Raw NS/EW counts on the RED road only
      ever increase between prints; counts on the GREEN road only ever
      decrease (capped at 3 per step). Any row with action `SWITCH` shows
      `t_since` reset to 0 and a reward roughly 5 more negative than a
      comparable KEEP row with the same queue lengths.
- [x] Episode termination:
      ```
      python -c "from traffic_env import TrafficEnv, ACTION_KEEP; e=TrafficEnv(max_steps=5, seed=1); e.reset(); [print(i, e.step(ACTION_KEEP)[2]) for i in range(5)]"
      ```
      Expected: `False` for i=0..3, `True` for i=4.
- [x] Rush-hour phase schedule:
      ```
      python -c "from traffic_env import TrafficEnv; e=TrafficEnv(rate_ns=1.0, rate_ew=1.0, max_steps=9, rush_hour=True, seed=1); [print(s, e._current_arrival_rates()) or setattr(e,'current_step',s) for s in range(9)]"
      ```
      Expected: rates `(2.0, 0.5)` for steps 0-3, `(1.0, 1.0)` for 4-6,
      `(0.5, 2.0)` for 7-8 (with `max_steps=9`, phase length 3).
- [ ] Invalid action raises: `env.step("NOPE")` should raise `ValueError`.

## `q_learning_agent.py`

- [x] Q-table only ever contains keys `(discretized_state, action)` where
      `action` is one of `traffic_env.ACTIONS` — by construction, every
      write to `self.q_table` goes through `(state, action)` pairs sourced
      from `ACTIONS` (`best_action`, `max_q_value`, `update`).
- [x] Epsilon starts near 1.0 and decays toward the configured floor
      (0.05) over the configured number of episodes:
      ```
      python -c "from q_learning_agent import QLearningAgent; a=QLearningAgent(epsilon_decay_episodes=300); print(a.epsilon_for_episode(0), a.epsilon_for_episode(150), a.epsilon_for_episode(500))"
      ```
      Result: `1.0 0.525 0.05` — monotonic decay confirmed.
- [x] Training reward curve trends upward (less negative) over episodes:
      `python q_learning_agent.py` on seed 7, 500 episodes ->
      first 50 avg -423.0, last 50 avg -294.0.

## `baseline_controller.py`

- [x] Switches exactly every N steps regardless of queue state:
      `python baseline_controller.py` (`switch_every=5`, seed=3, 20 steps)
      -> SWITCH at steps 4, 9, 14, 19 exactly, including step 17 where EW
      has 7 cars waiting and it still doesn't switch early.

## `value_iteration.py` (once built, stretch goal)

- [ ] Value function converges (max change between sweeps drops below a
      small threshold) within a bounded number of iterations.
- [ ] Resulting policy is deterministic and covers all 54 discretized
      states.

## `evaluate.py` (once built)

- [ ] All controllers run against `TrafficEnv(..., seed=FIXED_SEED)` with
      the identical seed — confirm by checking the first 5 arrival values
      are identical across controller runs (e.g. temporarily print them).
- [ ] Summary table prints avg wait, max queue, and switch count for every
      controller with no `NaN`/`None` values.
- [ ] Q-learning's average wait is lower than the fixed-timer baseline's
      (the whole point of the project) — if not, something is wrong with
      training, not just an unlucky seed.
