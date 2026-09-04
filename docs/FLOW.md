# Flow

How execution actually travels between files and functions. Updated as each
module is built.

## Current flow (only `traffic_env.py` exists so far)

```
traffic_env.py  __main__ block
  -> TrafficEnv(rate_ns, rate_ew, max_steps, seed)   # __init__ calls self.reset()
  -> env.reset()
       -> creates a fresh State(0, 0, "NS", 0)
       -> returns self.discretize_state(state)
  -> loop: env.step(action)
       -> self._current_arrival_rates()      # fixed rates, or rush-hour phase lookup
       -> np.random.Generator.poisson(...)    # sample arrivals for NS and EW
       -> apply action (KEEP ages the timer / SWITCH flips light + resets timer)
       -> drain up to DEPARTURE_RATE cars from whichever road is now green
       -> compute reward from resulting queue lengths (+ switch penalty)
       -> self.discretize_state(state)        # bucket for the caller
       -> return (discretized_state, reward, done)
```

## Planned flow once the agent exists

```
q_learning_agent.py  train()
  for each episode:
    state = env.reset()
    loop until done:
      action = agent.choose_action(state)      # epsilon-greedy over Q-table
      next_state, reward, done = env.step(action)
      agent.update(state, action, reward, next_state)   # Bellman update
      state = next_state
    log total_reward for this episode
```

## Planned flow for evaluate.py

```
evaluate.py
  for each controller in [trained QLearningAgent, FixedTimerController, (ValueIteration policy)]:
    env = TrafficEnv(..., seed=FIXED_SEED)      # same seed => same arrival sequence
    state = env.reset()
    loop until done:
      action = controller.act(state)            # no further learning during eval
      state, reward, done = env.step(action)
      record queue length / switches / reward
  plot all controllers' queue-length-over-time on one graph
  print summary table (avg wait, max queue, switch count) per controller
```

## Points where a bug is most likely to hide

- The action-application order inside `TrafficEnv.step()` (arrivals -> action
  -> departures). Getting this order wrong changes which road's cars can
  depart on the same tick a SWITCH happens.
- Any place a controller reads `env.state` directly instead of only using
  the discretized state returned by `step()`/`reset()` — only
  `value_iteration.py` is allowed to do this (see `CONSTRAINTS.md`).
