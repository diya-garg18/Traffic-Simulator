"""
baseline_controller.py — Fixed-timer traffic light controller (the "dumb" baseline).

This is the simplest possible controller: it switches the light every N
steps, no matter how many cars are waiting on either road. It exists so
evaluate.py can answer "is the learned Q-learning policy actually better
than doing nothing clever at all?" using the SAME environment interface
(reset()/step()) as every other controller in this project.

Why bother with such a simple baseline?
    Without it, there's no way to know whether the Q-learning agent's
    behaviour is genuinely good, or just "not obviously broken." A fixed
    N-step timer is exactly how many real, non-adaptive traffic lights
    already work, so beating it is a meaningful, concrete claim to make
    about the learned policy.
"""

from __future__ import annotations

from typing import Tuple

from traffic_env import ACTION_KEEP, ACTION_SWITCH

StateType = Tuple[str, str, str, str]


class FixedTimerController:
    """
    Switches the light every `switch_every` steps, regardless of traffic.

    This controller is stateful in exactly one way: it counts how many
    steps have passed since it last switched. That count is tracked here,
    NOT read from the environment's discretized state — a real fixed-timer
    controller would use its own internal clock, not sensor data, since
    the whole point is that it ignores traffic conditions.
    """

    def __init__(self, switch_every: int = 10) -> None:
        """
        switch_every:
            Number of steps to hold a light green before switching,
            regardless of queue lengths. 10 is a reasonable middle ground:
            long enough that a few cars can clear per phase, short enough
            that a busy road isn't left waiting an unreasonably long time.
        """
        self.switch_every = switch_every
        self.steps_since_switch = 0

    def reset(self) -> None:
        """Resets the controller's internal clock. Must be called whenever
        the environment is reset, so the controller's timer starts back at
        0 in sync with a fresh episode (mirrors TrafficEnv.reset())."""
        self.steps_since_switch = 0

    def choose_action(self, state: StateType) -> str:
        """
        Ignores `state` entirely (parameter kept only so this method has
        the same signature shape as QLearningAgent.choose_action, making
        it trivial to swap controllers in evaluate.py's loop). Switches
        exactly when the internal counter reaches switch_every, then
        resets the counter; otherwise keeps the light as-is.
        """
        if self.steps_since_switch >= self.switch_every - 1:
            self.steps_since_switch = 0
            return ACTION_SWITCH
        self.steps_since_switch += 1
        return ACTION_KEEP


if __name__ == "__main__":
    # Smoke test: run the controller against the real environment for 20
    # steps and print the light + action each step, to visually confirm it
    # switches EXACTLY every switch_every steps, independent of the queues.
    from traffic_env import TrafficEnv

    env = TrafficEnv(rate_ns=1.0, rate_ew=1.0, max_steps=20, seed=3)
    controller = FixedTimerController(switch_every=5)

    state = env.reset()
    controller.reset()
    print(f"{'step':>4} {'action':>7} {'light':>5} | {'raw NS':>6} {'raw EW':>6}")
    print("-" * 45)

    for i in range(20):
        action = controller.choose_action(state)
        state, reward, done = env.step(action)
        s = env.state
        print(f"{i:>4} {action:>7} {s.current_light:>5} | "
              f"{s.cars_waiting_ns:>6} {s.cars_waiting_ew:>6}")
        if done:
            break
